"""
Clinical Policy Engine

Produces deterministic action plans from health context and urgency.
Rules are loaded from a versioned policy manifest for governance and audit.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from models.schemas import (
    ActionPlan,
    ActionPlanAction,
    HealthContext,
)
from chatbot.knowledge_base import get_drug_interaction_db


DEFAULT_POLICY_VERSION = "2026.02.v2"


@dataclass(frozen=True)
class PolicyRule:
    """Versioned policy rule descriptor."""

    rule_id: str
    policy_ref: str
    description: str
    thresholds: Dict[str, float]
    changelog_ref: str


class ClinicalPolicyEngine:
    """Rule-based policy engine for generating deterministic action plans."""

    def __init__(self):
        self._drug_db = None
        self._manifest = self._load_policy_manifest()
        self._rules = self._build_rule_registry(self._manifest)
        self._policy_version = self._manifest.get("version", DEFAULT_POLICY_VERSION)
        self._policy_manifest_hash = self._manifest.get("manifest_hash", "unknown")
        self._changelog_index = {
            entry.get("id"): entry for entry in self._manifest.get("changelog", [])
        }

    @property
    def drug_db(self):
        """Lazy load drug interaction DB for medication policy checks."""
        if self._drug_db is None:
            self._drug_db = get_drug_interaction_db()
        return self._drug_db

    def build_action_plan(
        self,
        health_context: Optional[HealthContext],
        user_question: str,
    ) -> ActionPlan:
        """
        Build deterministic action plan from available context.
        """
        actions: List[ActionPlanAction] = []
        contraindications: List[str] = []
        changelog_refs: Set[str] = set()

        if health_context and health_context.risk_assessment:
            self._apply_fall_rules(actions, health_context, changelog_refs)
            self._apply_medication_rules(actions, contraindications, health_context, changelog_refs)
            self._apply_cognitive_sleep_rules(actions, health_context, changelog_refs)

        if not actions:
            rule = self._rule("general_monitoring")
            actions.append(
                ActionPlanAction(
                    id="routine_monitoring",
                    title="Continue routine monitoring",
                    description="Track symptoms and continue routine care observations.",
                    priority=4,
                    requires_clinician=False,
                    policy_refs=[self._policy_ref(rule)],
                )
            )
            self._track_changelog(changelog_refs, rule)

        if any(a.id.startswith("medication_") for a in actions):
            contraindications.append("Do not change medication dosing without clinician approval.")

        actions.sort(key=lambda x: x.priority, reverse=True)

        return ActionPlan(
            actions=actions,
            contraindications=sorted(set(contraindications)),
            confidence=85.0 if health_context else 60.0,
            policy_version=self._policy_version,
            policy_changelog_refs=sorted(changelog_refs),
        )

    def build_escalation_action_plan(self, urgency_level: str) -> ActionPlan:
        """Build deterministic action plan for urgent/emergency bypass flows."""
        if urgency_level == "emergency":
            return ActionPlan(
                actions=[
                    ActionPlanAction(
                        id="emergency_call_911",
                        title="Call emergency services now",
                        description="Call 911 immediately and keep the person monitored until help arrives.",
                        priority=10,
                        requires_clinician=True,
                        policy_refs=[f"policy.safety.emergency.v1@{self._policy_version}"],
                    )
                ],
                contraindications=[],
                confidence=100.0,
                policy_version=self._policy_version,
                policy_changelog_refs=["POL-005"],
            )

        if urgency_level == "urgent":
            return ActionPlan(
                actions=[
                    ActionPlanAction(
                        id="urgent_same_day_care",
                        title="Seek same-day medical evaluation",
                        description="Contact clinician or urgent care for same-day assessment.",
                        priority=9,
                        requires_clinician=True,
                        policy_refs=[f"policy.safety.urgent.v1@{self._policy_version}"],
                    )
                ],
                contraindications=[],
                confidence=95.0,
                policy_version=self._policy_version,
                policy_changelog_refs=["POL-005"],
            )

        return self.build_action_plan(health_context=None, user_question="")

    def get_policy_metadata(self) -> Dict[str, Any]:
        """Return loaded policy metadata for observability and audits."""
        return {
            "version": self._policy_version,
            "manifest_hash": self._policy_manifest_hash,
            "rules_loaded": len(self._rules),
            "changelog_count": len(self._changelog_index),
        }

    def _apply_fall_rules(
        self,
        actions: List[ActionPlanAction],
        health_context: HealthContext,
        changelog_refs: Set[str],
    ) -> None:
        risk = health_context.risk_assessment
        if not risk:
            return

        if risk.fall_risk and risk.fall_risk >= self._threshold("fall_high_risk", "fall_risk_gte", 75):
            rule = self._rule("fall_high_risk")
            actions.append(
                ActionPlanAction(
                    id="fall_safety_assessment",
                    title="Initiate clinician-led fall safety plan",
                    description="Perform immediate home-safety checks and supervised mobility review.",
                    priority=9,
                    requires_clinician=True,
                    policy_refs=[self._policy_ref(rule)],
                )
            )
            self._track_changelog(changelog_refs, rule)
        elif risk.fall_risk and risk.fall_risk >= self._threshold("fall_moderate_risk", "fall_risk_gte", 60):
            rule = self._rule("fall_moderate_risk")
            actions.append(
                ActionPlanAction(
                    id="fall_prevention_program",
                    title="Start fall prevention program",
                    description="Begin balance and strength program with home hazard reduction checklist.",
                    priority=7,
                    requires_clinician=False,
                    policy_refs=[self._policy_ref(rule)],
                )
            )
            self._track_changelog(changelog_refs, rule)

        nighttime_threshold = int(self._threshold("fall_night_safety", "nighttime_bathroom_visits_gte", 3))
        if health_context.adl_summary and health_context.adl_summary.nighttime_bathroom_visits >= nighttime_threshold:
            rule = self._rule("fall_night_safety")
            actions.append(
                ActionPlanAction(
                    id="fall_night_lighting",
                    title="Apply nighttime safety safeguards",
                    description="Improve nighttime lighting and ensure assisted path to bathroom.",
                    priority=7,
                    requires_clinician=False,
                    policy_refs=[self._policy_ref(rule)],
                )
            )
            self._track_changelog(changelog_refs, rule)

    def _apply_medication_rules(
        self,
        actions: List[ActionPlanAction],
        contraindications: List[str],
        health_context: HealthContext,
        changelog_refs: Set[str],
    ) -> None:
        profile = health_context.medical_profile
        risk = health_context.risk_assessment

        medications = [m.name for m in profile.medications] if profile and profile.medications else []
        conditions = profile.chronic_conditions if profile else []
        age = profile.age if profile and profile.age else 75

        med_risk_result = None
        if medications:
            med_risk_result = self.drug_db.assess_medication_risk(
                medications=medications,
                conditions=conditions,
                age=age,
            )

        risk_threshold = self._threshold("medication_reconciliation", "medication_risk_gte", 60)
        medication_risk_high = bool(risk and risk.medication_risk and risk.medication_risk >= risk_threshold)

        if medication_risk_high or (med_risk_result and med_risk_result["risk_level"] in {"HIGH", "MODERATE"}):
            rule = self._rule("medication_reconciliation")
            actions.append(
                ActionPlanAction(
                    id="medication_reconciliation",
                    title="Review medications for safety",
                    description="Request clinician or pharmacist medication reconciliation.",
                    priority=8,
                    requires_clinician=True,
                    policy_refs=[self._policy_ref(rule)],
                )
            )
            self._track_changelog(changelog_refs, rule)

        if med_risk_result:
            major_or_contra = [
                i for i in med_risk_result["interactions"]
                if i["severity"] in {"major", "contraindicated"}
            ]
            if major_or_contra:
                rule = self._rule("medication_major_interactions")
                actions.append(
                    ActionPlanAction(
                        id="medication_interaction_review",
                        title="Urgent medication interaction review",
                        description="Major interaction risk detected; arrange same-day clinician review.",
                        priority=9,
                        requires_clinician=True,
                        policy_refs=[self._policy_ref(rule)],
                    )
                )
                contraindications.append("Do not start new over-the-counter medicines until clinician review.")
                self._track_changelog(changelog_refs, rule)

            if med_risk_result["medication_count"] >= int(
                self._threshold("medication_polypharmacy", "medication_count_gte", 5)
            ):
                rule = self._rule("medication_polypharmacy")
                actions.append(
                    ActionPlanAction(
                        id="medication_polypharmacy_review",
                        title="Assess polypharmacy burden",
                        description="Review all medications for simplification opportunities and necessity.",
                        priority=7,
                        requires_clinician=True,
                        policy_refs=[self._policy_ref(rule)],
                    )
                )
                self._track_changelog(changelog_refs, rule)

            if med_risk_result["anticholinergic_burden"]["score"] >= self._threshold(
                "medication_anticholinergic", "acb_score_gte", 3
            ):
                rule = self._rule("medication_anticholinergic")
                actions.append(
                    ActionPlanAction(
                        id="medication_anticholinergic_burden",
                        title="Reduce anticholinergic burden",
                        description="Review alternatives to anticholinergic medications due to cognitive/fall risk.",
                        priority=8,
                        requires_clinician=True,
                        policy_refs=[self._policy_ref(rule)],
                    )
                )
                self._track_changelog(changelog_refs, rule)

    def _apply_cognitive_sleep_rules(
        self,
        actions: List[ActionPlanAction],
        health_context: HealthContext,
        changelog_refs: Set[str],
    ) -> None:
        risk = health_context.risk_assessment
        if not risk:
            return

        if risk.cognitive_decline_risk and risk.cognitive_decline_risk >= self._threshold(
            "cognitive_high_risk", "cognitive_risk_gte", 70
        ):
            rule = self._rule("cognitive_high_risk")
            actions.append(
                ActionPlanAction(
                    id="cognitive_clinical_evaluation",
                    title="Schedule clinician cognitive evaluation",
                    description="Arrange formal cognitive review and caregiver-supported observation plan.",
                    priority=8,
                    requires_clinician=True,
                    policy_refs=[self._policy_ref(rule)],
                )
            )
            self._track_changelog(changelog_refs, rule)
        elif risk.cognitive_decline_risk and risk.cognitive_decline_risk >= self._threshold(
            "cognitive_moderate_risk", "cognitive_risk_gte", 50
        ):
            rule = self._rule("cognitive_moderate_risk")
            actions.append(
                ActionPlanAction(
                    id="cognitive_monitoring",
                    title="Start cognitive monitoring routine",
                    description="Schedule structured cognitive monitoring and caregiver observations.",
                    priority=6,
                    requires_clinician=False,
                    policy_refs=[self._policy_ref(rule)],
                )
            )
            self._track_changelog(changelog_refs, rule)

        if risk.sleep_disorder_risk and risk.sleep_disorder_risk >= self._threshold(
            "sleep_high_risk", "sleep_risk_gte", 70
        ):
            rule = self._rule("sleep_high_risk")
            actions.append(
                ActionPlanAction(
                    id="sleep_clinical_review",
                    title="Schedule sleep disorder review",
                    description="Arrange clinician evaluation for persistent or severe sleep disturbances.",
                    priority=7,
                    requires_clinician=True,
                    policy_refs=[self._policy_ref(rule)],
                )
            )
            self._track_changelog(changelog_refs, rule)
        elif risk.sleep_disorder_risk and risk.sleep_disorder_risk >= self._threshold(
            "sleep_moderate_risk", "sleep_risk_gte", 50
        ):
            rule = self._rule("sleep_moderate_risk")
            actions.append(
                ActionPlanAction(
                    id="sleep_hygiene_protocol",
                    title="Apply sleep support protocol",
                    description="Apply bedtime routine checks and escalate if persistent disruption continues.",
                    priority=5,
                    requires_clinician=False,
                    policy_refs=[self._policy_ref(rule)],
                )
            )
            self._track_changelog(changelog_refs, rule)

    def _rule(self, rule_id: str) -> PolicyRule:
        if rule_id not in self._rules:
            raise KeyError(f"Policy rule not found: {rule_id}")
        return self._rules[rule_id]

    def _threshold(self, rule_id: str, key: str, default: float) -> float:
        rule = self._rule(rule_id)
        return float(rule.thresholds.get(key, default))

    def _policy_ref(self, rule: PolicyRule) -> str:
        return f"{rule.policy_ref}@{self._policy_version}"

    @staticmethod
    def _track_changelog(changelog_refs: Set[str], rule: PolicyRule) -> None:
        if rule.changelog_ref:
            changelog_refs.add(rule.changelog_ref)

    @staticmethod
    def _build_rule_registry(manifest: Dict[str, Any]) -> Dict[str, PolicyRule]:
        rules: Dict[str, PolicyRule] = {}
        for rule_data in manifest.get("rules", []):
            rule = PolicyRule(
                rule_id=rule_data["rule_id"],
                policy_ref=rule_data["policy_ref"],
                description=rule_data.get("description", ""),
                thresholds=rule_data.get("thresholds", {}),
                changelog_ref=rule_data.get("changelog_ref", ""),
            )
            rules[rule.rule_id] = rule
        return rules

    @staticmethod
    def _default_manifest() -> Dict[str, Any]:
        return {
            "version": DEFAULT_POLICY_VERSION,
            "changelog": [
                {
                    "id": "POL-005",
                    "date": "2026-02-09",
                    "summary": "Fallback policy manifest loaded.",
                }
            ],
            "rules": [],
            "manifest_hash": "fallback",
        }

    def _load_policy_manifest(self) -> Dict[str, Any]:
        manifest_path = Path(__file__).resolve().parent / "policy" / "rules_v2026_02.json"
        if not manifest_path.exists():
            return self._default_manifest()

        raw = manifest_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        data["manifest_hash"] = sha256(raw.encode("utf-8")).hexdigest()
        return data


_policy_engine = None


def get_policy_engine() -> ClinicalPolicyEngine:
    """Get or create singleton policy engine."""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = ClinicalPolicyEngine()
    return _policy_engine

