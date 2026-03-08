"""
T-80 Rollout Manager: Shadow -> Canary -> Full

Lane T-80: Controlled Rollout with Safety Gates

This module manages the staged rollout of timeline-quality improvements:
1. Shadow: Generate artifacts without affecting production
2. Canary: Limited elder cohort with observation window
3. Full: Complete rollout after canary validation

Fail-closed principles:
- No promotion without explicit signoff
- Automatic rollback on safety regression
- Canary requires fixed observation period
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Mapping, Sequence

from ml.policy_presets import (
    RolloutAutoRollbackPolicy,
    RolloutLadderPolicy,
    RolloutFallbackPolicy,
    RolloutProgressionCriteria,
    load_rollout_ladder_policy,
)

logger = logging.getLogger(__name__)


class RolloutStage(Enum):
    """Rollout stages for timeline-quality improvements."""
    SHADOW = "shadow"
    CANARY = "canary"
    FULL = "full"
    ROLLED_BACK = "rolled_back"


class PromotionDecision(Enum):
    """Promotion decision from canary evaluation."""
    PROMOTE = "promote"
    HOLD = "hold"
    ROLLBACK = "rollback"


@dataclass
class CanaryConfig:
    """Configuration for canary rollout."""
    # Elder cohort selection
    max_elders: int = 5
    
    # Observation window
    observation_days: int = 7
    
    # Safety thresholds (fail-closed)
    min_timeline_gate_pass_rate: float = 0.80
    # Note: hard gate pass rate is always 100% (enforced in code)
    
    # Auto-rollback triggers
    auto_rollback_on_safety_regression: bool = True
    auto_rollback_on_leakage_failure: bool = True


@dataclass(frozen=True)
class CanaryRealDataEvidencePolicy:
    """Policy for real-data evidence required by canary artifact evaluation."""

    require_real_data_evidence: bool = True
    min_residents_covered: int = 1
    min_resident_days_total: int = 7
    allowed_data_sources: tuple[str, ...] = ("production_shadow", "production_canary")


@dataclass
class RolloutState:
    """Current state of a rollout."""
    stage: RolloutStage
    started_at: str
    canary_elders: List[str] = field(default_factory=list)
    observation_end: Optional[str] = None
    promotion_decision: Optional[str] = None
    rollback_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "started_at": self.started_at,
            "canary_elders": self.canary_elders,
            "observation_end": self.observation_end,
            "promotion_decision": self.promotion_decision,
            "rollback_reason": self.rollback_reason,
        }


@dataclass
class RolloutSummary:
    """Summary of rollout status for reporting."""
    current_stage: str
    canary_progress: Optional[Dict[str, Any]] = None
    promotion_ready: bool = False
    blockers: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_stage": self.current_stage,
            "canary_progress": self.canary_progress,
            "promotion_ready": self.promotion_ready,
            "blockers": self.blockers,
        }


@dataclass(frozen=True)
class LadderProgressionResult:
    """Result of one rung progression evaluation."""

    can_advance: bool
    target_rung: int
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "can_advance": self.can_advance,
            "target_rung": self.target_rung,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class AutoRollbackAssessment:
    """Result of evaluating auto-rollback trigger windows."""

    should_rollback: bool
    reason_codes: tuple[str, ...] = ()
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_rollback": self.should_rollback,
            "reason_codes": list(self.reason_codes),
            "details": dict(self.details),
        }


class T80RolloutManager:
    """
    Manages T-80 rollout: Shadow -> Canary -> Full
    
    Usage:
        manager = T80RolloutManager()
        
        # Start shadow mode
        manager.start_shadow()
        
        # Promote to canary (limited cohort)
        manager.promote_to_canary(elder_ids=["HK001", "HK002"])
        
        # Evaluate canary after observation period
        decision = manager.evaluate_canary(signoff_results)
        
        # Promote to full or rollback
        if decision == PromotionDecision.PROMOTE:
            manager.promote_to_full()
        else:
            manager.rollback(reason="Safety regression detected")
    """
    
    def __init__(
        self,
        state_dir: Optional[Path] = None,
        canary_config: Optional[CanaryConfig] = None,
        rollout_ladder_policy: Optional[RolloutLadderPolicy] = None,
    ):
        # Fix [Low]: Resolve path relative to repo root (this file location)
        if state_dir is None:
            # This file is at backend/ml/t80_rollout_manager.py
            repo_root = Path(__file__).resolve().parents[2]
            state_dir = repo_root / "backend" / "validation_runs_canary"
        
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.config = canary_config or CanaryConfig()
        self.real_data_policy = self._load_real_data_evidence_policy()
        self.rollout_ladder_policy = rollout_ladder_policy or self._load_rollout_ladder_policy()
        self._state: Optional[RolloutState] = None
        self._load_state()

    def _load_real_data_evidence_policy(self) -> CanaryRealDataEvidencePolicy:
        """Load real-data canary evidence policy from Beta6 config."""
        default_policy = CanaryRealDataEvidencePolicy()
        config_path = Path(__file__).resolve().parents[1] / "config" / "beta6_canary_gate.yaml"
        if not config_path.exists():
            logger.warning(
                "T-80: canary evidence policy config missing (%s); using strict defaults",
                config_path,
            )
            return default_policy

        try:
            from ml.beta6.beta6_schema import load_validated_beta6_config

            payload = load_validated_beta6_config(
                config_path,
                expected_filename="beta6_canary_gate.yaml",
            )
            section = payload.get("canary", {})
            if not isinstance(section, dict):
                logger.warning("T-80: invalid canary policy payload; using strict defaults")
                return default_policy
            allowed_sources = section.get("allowed_data_sources", default_policy.allowed_data_sources)
            if not isinstance(allowed_sources, (list, tuple)):
                allowed_sources = default_policy.allowed_data_sources
            normalized_sources = tuple(
                str(item).strip() for item in allowed_sources if str(item).strip()
            ) or default_policy.allowed_data_sources
            return CanaryRealDataEvidencePolicy(
                require_real_data_evidence=self._parse_strict_bool(
                    section.get("require_real_data_evidence", True),
                    key="canary.require_real_data_evidence",
                ),
                min_residents_covered=max(1, int(section.get("min_residents_covered", 1) or 1)),
                min_resident_days_total=max(1, int(section.get("min_resident_days_total", 7) or 7)),
                allowed_data_sources=normalized_sources,
            )
        except Exception as exc:
            logger.warning("T-80: failed to load canary evidence policy (%s); using strict defaults", exc)
            return default_policy

    @staticmethod
    def _parse_strict_bool(value: Any, *, key: str) -> bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"Expected boolean for {key}, got {type(value).__name__}")

    def _load_rollout_ladder_policy(self) -> RolloutLadderPolicy:
        try:
            return load_rollout_ladder_policy()
        except Exception as exc:
            logger.warning("T-80: failed to load rollout ladder policy (%s); using strict defaults", exc)
            return RolloutLadderPolicy(
                rungs=(),
                progression=RolloutProgressionCriteria(),
                auto_rollback=RolloutAutoRollbackPolicy(),
                fallback=RolloutFallbackPolicy(),
            )
    
    def _state_file(self) -> Path:
        return self.state_dir / "rollout_state.json"
    
    def _load_state(self) -> None:
        """Load existing rollout state."""
        state_file = self._state_file()
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                self._state = RolloutState(
                    stage=RolloutStage(data.get("stage", "shadow")),
                    started_at=data.get("started_at", self._now_iso()),
                    canary_elders=data.get("canary_elders", []),
                    observation_end=data.get("observation_end"),
                    promotion_decision=data.get("promotion_decision"),
                    rollback_reason=data.get("rollback_reason"),
                )
            except Exception as e:
                logger.warning(f"Failed to load rollout state: {e}")
                self._state = None
    
    def _save_state(self) -> None:
        """Save current rollout state."""
        if self._state:
            self._state_file().write_text(json.dumps(self._state.to_dict(), indent=2))
    
    def _now_iso(self) -> str:
        """Current UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()
    
    def get_state(self) -> Optional[RolloutState]:
        """Get current rollout state."""
        return self._state
    
    def start_shadow(self) -> RolloutState:
        """Start or reset to shadow mode."""
        self._state = RolloutState(
            stage=RolloutStage.SHADOW,
            started_at=self._now_iso(),
        )
        self._save_state()
        logger.info("T-80: Entered SHADOW mode")
        return self._state
    
    def promote_to_canary(self, elder_ids: List[str]) -> RolloutState:
        """
        Promote from shadow to canary with limited elder cohort.
        
        Args:
            elder_ids: List of elder IDs for canary cohort (max 5)
            
        Raises:
            ValueError: If too many elders or invalid state transition
        """
        if self._state is None:
            raise ValueError("Must start shadow mode before canary")
        
        if self._state.stage != RolloutStage.SHADOW:
            raise ValueError(f"Cannot promote to canary from {self._state.stage.value}")
        
        if len(elder_ids) > self.config.max_elders:
            raise ValueError(f"Canary cohort max {self.config.max_elders} elders, got {len(elder_ids)}")
        
        observation_end = datetime.now(timezone.utc) + timedelta(days=self.config.observation_days)
        
        self._state = RolloutState(
            stage=RolloutStage.CANARY,
            started_at=self._now_iso(),
            canary_elders=list(elder_ids),
            observation_end=observation_end.isoformat(),
        )
        self._save_state()
        logger.info(f"T-80: Promoted to CANARY with elders {elder_ids}")
        return self._state
    
    def evaluate_canary(
        self,
        signoff_results: List[Dict[str, Any]],
        *,
        require_real_data_evidence: Optional[bool] = None,
    ) -> PromotionDecision:
        """
        Evaluate canary results and determine promotion decision.
        
        Args:
            signoff_results: List of signoff results from canary elders
            
        Returns:
            PromotionDecision: PROMOTE, HOLD, or ROLLBACK
        """
        if self._state is None or self._state.stage != RolloutStage.CANARY:
            logger.warning("T-80: Cannot evaluate - not in canary stage")
            return PromotionDecision.HOLD
        
        # Check if observation period complete
        if self._state.observation_end:
            end_time = datetime.fromisoformat(self._state.observation_end)
            if datetime.now(timezone.utc) < end_time:
                remaining = (end_time - datetime.now(timezone.utc)).days
                logger.info(f"T-80: Canary observation in progress ({remaining} days remaining)")
                return PromotionDecision.HOLD
        
        policy_requires_real_data_evidence = bool(self.real_data_policy.require_real_data_evidence)
        if require_real_data_evidence is None:
            enforce_real_data_evidence = policy_requires_real_data_evidence
        else:
            requested_real_data_evidence = bool(require_real_data_evidence)
            # Fail-closed: caller override may tighten gates, never relax policy-required gates.
            if policy_requires_real_data_evidence and not requested_real_data_evidence:
                logger.warning(
                    "T-80: ignored require_real_data_evidence=False override because policy requires evidence"
                )
            enforce_real_data_evidence = (
                policy_requires_real_data_evidence or requested_real_data_evidence
            )
        blockers = self._check_canary_blockers(
            signoff_results,
            require_real_data_evidence=enforce_real_data_evidence,
        )
        
        if blockers:
            # Fail-safe: clear any stale promotion approval on new blocking evidence.
            self._state.promotion_decision = None
            self._save_state()

            # Check for critical safety regressions requiring rollback
            # Hard gate failure = safety regression (must rollback)
            hard_gate_failure = any("hard gates failed" in b.lower() for b in blockers)
            leakage_failure = any("leakage" in b.lower() for b in blockers)
            
            if hard_gate_failure and self.config.auto_rollback_on_safety_regression:
                logger.error(f"T-80: Hard gate failure detected - recommending ROLLBACK")
                return PromotionDecision.ROLLBACK
            
            if leakage_failure and self.config.auto_rollback_on_leakage_failure:
                logger.error(f"T-80: Leakage audit failure detected - recommending ROLLBACK")
                return PromotionDecision.ROLLBACK
            
            logger.warning(f"T-80: Canary blockers found - recommending HOLD")
            return PromotionDecision.HOLD
        
        # Fix [High]: Persist promotion decision when all checks pass
        self._state.promotion_decision = "promote"
        self._save_state()
        
        logger.info("T-80: Canary validation passed - recommending PROMOTE")
        return PromotionDecision.PROMOTE

    def evaluate_canary_artifacts(
        self,
        artifacts: List[Union[str, Path, Dict[str, Any]]],
    ) -> PromotionDecision:
        """
        Evaluate canary using live signoff artifacts.

        Accepted artifact inputs:
        - path string or Path (elder_id must be present in artifact payload)
        - dict {"path": "...", "elder_id": "..."} for payloads without elder_id
        """
        signoff_results = self.build_canary_results_from_artifacts(artifacts)
        return self.evaluate_canary(
            signoff_results,
            require_real_data_evidence=bool(self.real_data_policy.require_real_data_evidence),
        )

    def build_canary_results_from_artifacts(
        self,
        artifacts: List[Union[str, Path, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Build normalized canary evaluation results from signoff artifacts."""
        results: List[Dict[str, Any]] = []
        for item in artifacts:
            path: Path
            elder_id_override: Optional[str] = None
            if isinstance(item, (str, Path)):
                path = Path(item)
            elif isinstance(item, dict):
                path_raw = item.get("path")
                if path_raw is None:
                    raise ValueError(f"Artifact mapping missing 'path': {item}")
                path = Path(str(path_raw))
                elder_id_raw = item.get("elder_id")
                elder_id_override = str(elder_id_raw) if elder_id_raw else None
            else:
                raise ValueError(f"Unsupported artifact input type: {type(item)}")

            if not path.exists() or not path.is_file():
                raise ValueError(f"Artifact not found: {path}")
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise ValueError(f"Artifact payload must be JSON object: {path}")
            results.append(
                self._extract_canary_result_from_signoff_payload(
                    payload=payload,
                    elder_id_override=elder_id_override,
                    source=str(path),
                )
            )
        return results

    def _extract_canary_result_from_signoff_payload(
        self,
        payload: Dict[str, Any],
        *,
        elder_id_override: Optional[str] = None,
        source: str = "artifact",
    ) -> Dict[str, Any]:
        """
        Normalize a signoff artifact payload to canary result shape:
        {elder_id, timeline_gates_pass, hard_gates_pass, leakage_audit_pass}

        Supported payloads:
        - WS-6 signoff pack (`split_seed_results`)
        - Validation signoff pack (`validation_run` + `compliance_checklist`)
        - Aggregate signoff (`gate_decision`, `seed_split_stability`, `timeline_release`)
        """
        elder_id = elder_id_override
        if not elder_id:
            vr = payload.get("validation_run")
            if isinstance(vr, dict) and vr.get("elder_id"):
                elder_id = str(vr.get("elder_id"))
            elif payload.get("elder_id"):
                elder_id = str(payload.get("elder_id"))
        if not elder_id:
            raise ValueError(f"Missing elder_id for artifact: {source}")
        evidence_pass, evidence_reason = self._evaluate_real_data_evidence(payload)

        # 1) WS-6 signoff pack
        split_seed_results = payload.get("split_seed_results")
        if isinstance(split_seed_results, list):
            if len(split_seed_results) == 0 or not all(isinstance(r, dict) for r in split_seed_results):
                return {
                    "elder_id": elder_id,
                    "timeline_gates_pass": False,
                    "hard_gates_pass": False,
                    "leakage_audit_pass": False,
                    "real_data_evidence_pass": bool(evidence_pass),
                    "real_data_evidence_reason": evidence_reason,
                }

            hard_ok = True
            leakage_ok = True
            timeline_passed = 0
            timeline_total = 0
            for r in split_seed_results:
                hard_total = int(r.get("hard_gates_total", 0) or 0)
                hard_passed = int(r.get("hard_gates_passed", 0) or 0)
                if hard_total <= 0 or hard_passed != hard_total:
                    hard_ok = False

                if r.get("leakage_audit_pass") is not True:
                    leakage_ok = False

                t_total = int(r.get("timeline_gates_total", 0) or 0)
                t_pass = int(r.get("timeline_gates_passed", 0) or 0)
                if t_total > 0:
                    timeline_total += t_total
                    timeline_passed += max(0, min(t_pass, t_total))

            # If no timeline checks were evaluated, fail closed.
            timeline_ok = False
            if timeline_total > 0:
                timeline_ok = (timeline_passed / timeline_total) >= float(self.config.min_timeline_gate_pass_rate)

            return {
                "elder_id": elder_id,
                "timeline_gates_pass": bool(timeline_ok),
                "hard_gates_pass": bool(hard_ok),
                "leakage_audit_pass": bool(leakage_ok),
                "real_data_evidence_pass": bool(evidence_pass),
                "real_data_evidence_reason": evidence_reason,
            }

        # 2) Validation signoff pack (legacy + D1 strict)
        if isinstance(payload.get("validation_run"), dict):
            compliance = payload.get("compliance_checklist", {})
            if isinstance(compliance, dict) and compliance:
                hard_ok = compliance.get("all_hard_gates_pass") is True
                leakage_ok = (
                    compliance.get("leakage_audit_present") is True
                    and compliance.get("leakage_audit_pass") is True
                )
                timeline_ok = compliance.get("timeline_gates_pass") is True
                return {
                    "elder_id": elder_id,
                    "timeline_gates_pass": bool(timeline_ok),
                    "hard_gates_pass": bool(hard_ok),
                    "leakage_audit_pass": bool(leakage_ok),
                    "real_data_evidence_pass": bool(evidence_pass),
                    "real_data_evidence_reason": evidence_reason,
                }

            # Fallback derive from per-daily gate payloads (fail-closed defaults).
            daily_results = payload.get("validation_run", {}).get("daily_results", [])
            if not isinstance(daily_results, list) or len(daily_results) == 0:
                return {
                    "elder_id": elder_id,
                    "timeline_gates_pass": False,
                    "hard_gates_pass": False,
                    "leakage_audit_pass": False,
                    "real_data_evidence_pass": bool(evidence_pass),
                    "real_data_evidence_reason": evidence_reason,
                }

            hard_ok = True
            leakage_ok = True
            timeline_passed = 0
            timeline_total = 0
            saw_hard = False
            for d in daily_results:
                if not isinstance(d, dict):
                    hard_ok = False
                    leakage_ok = False
                    continue
                gate = d.get("gate_stack_summary", {})
                if not isinstance(gate, dict):
                    hard_ok = False
                    leakage_ok = False
                    continue

                h_total = int(gate.get("hard_gates_total", 0) or 0)
                h_pass = int(gate.get("hard_gates_passed", 0) or 0)
                if h_total > 0:
                    saw_hard = True
                    if h_pass != h_total:
                        hard_ok = False
                elif gate.get("all_hard_gates_pass") is not True:
                    hard_ok = False

                leakage = gate.get("leakage_audit", {})
                if not isinstance(leakage, dict) or leakage.get("pass") is not True:
                    leakage_ok = False

                t_total = int(gate.get("timeline_gates_total", 0) or 0)
                t_pass = int(gate.get("timeline_gates_passed", 0) or 0)
                if t_total > 0:
                    timeline_total += t_total
                    timeline_passed += max(0, min(t_pass, t_total))

            timeline_ok = False
            if timeline_total > 0:
                timeline_ok = (timeline_passed / timeline_total) >= float(self.config.min_timeline_gate_pass_rate)

            return {
                "elder_id": elder_id,
                "timeline_gates_pass": bool(timeline_ok),
                "hard_gates_pass": bool(hard_ok and saw_hard),
                "leakage_audit_pass": bool(leakage_ok),
                "real_data_evidence_pass": bool(evidence_pass),
                "real_data_evidence_reason": evidence_reason,
            }

        # 3) Aggregate signoff output
        if payload.get("gate_decision") is not None:
            failed_reasons = payload.get("failed_reasons", [])
            if not isinstance(failed_reasons, list):
                failed_reasons = []
            failed_reason_text = [str(r) for r in failed_reasons]

            seed_split = payload.get("seed_split_stability", {})
            hard_ok = False
            if isinstance(seed_split, dict):
                hard_ok = seed_split.get("hard_gate_all_seeds") is True

            leakage_ok = not any(reason.startswith("leakage_audit") for reason in failed_reason_text)

            timeline_release = payload.get("timeline_release", {})
            timeline_ok = False
            if isinstance(timeline_release, dict):
                # Canary uses internal-ready threshold.
                timeline_ok = timeline_release.get("internal_ready") is True

            return {
                "elder_id": elder_id,
                "timeline_gates_pass": bool(timeline_ok),
                "hard_gates_pass": bool(hard_ok),
                "leakage_audit_pass": bool(leakage_ok),
                "real_data_evidence_pass": bool(evidence_pass),
                "real_data_evidence_reason": evidence_reason,
            }

        raise ValueError(f"Unsupported signoff artifact schema: {source}")

    def _evaluate_real_data_evidence(self, payload: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate real-data canary evidence payload.

        Expected schema:
        canary_real_data_evidence:
          is_real_data: true
          data_source: production_shadow|production_canary
          residents_covered: int
          resident_days_total: int
        """
        evidence = payload.get("canary_real_data_evidence")
        if not isinstance(evidence, dict):
            return False, "missing_canary_real_data_evidence"

        if evidence.get("is_real_data") is not True:
            return False, "is_real_data_not_true"

        data_source = str(evidence.get("data_source", "")).strip()
        if not data_source:
            return False, "missing_data_source"
        if data_source not in self.real_data_policy.allowed_data_sources:
            return False, f"invalid_data_source:{data_source}"

        try:
            residents_covered = int(evidence.get("residents_covered", 0) or 0)
        except (TypeError, ValueError):
            residents_covered = 0
        if residents_covered < self.real_data_policy.min_residents_covered:
            return False, "insufficient_residents_covered"

        try:
            resident_days_total = int(evidence.get("resident_days_total", 0) or 0)
        except (TypeError, ValueError):
            resident_days_total = 0
        if resident_days_total < self.real_data_policy.min_resident_days_total:
            return False, "insufficient_resident_days_total"

        return True, "ok"

    def _check_canary_blockers(
        self,
        signoff_results: List[Dict[str, Any]],
        *,
        require_real_data_evidence: bool = False,
    ) -> List[str]:
        """Check for blockers in canary results."""
        blockers = []
        
        if not signoff_results:
            blockers.append("No signoff results provided")
            return blockers
        if not all(isinstance(r, dict) for r in signoff_results):
            blockers.append("Invalid signoff result payload shape")
            return blockers
        
        # Fix [High]: Check full cohort coverage
        if self._state and self._state.canary_elders:
            result_elder_ids = set()
            duplicate_elders = set()
            for r in signoff_results:
                elder_id = r.get("elder_id")
                if elder_id:
                    if elder_id in result_elder_ids:
                        duplicate_elders.add(str(elder_id))
                    result_elder_ids.add(elder_id)
            
            expected_elders = set(self._state.canary_elders)
            missing_elders = expected_elders - result_elder_ids
            extra_elders = result_elder_ids - expected_elders
            
            if missing_elders:
                blockers.append(f"Missing signoff results for elders: {sorted(missing_elders)}")
            if extra_elders:
                blockers.append(f"Unexpected signoff results for elders: {sorted(extra_elders)}")
            if duplicate_elders:
                blockers.append(f"Duplicate signoff results for elders: {sorted(duplicate_elders)}")
            
            # If coverage is incomplete, return early (can't evaluate properly)
            if missing_elders or extra_elders or duplicate_elders:
                return blockers
        
        # Check timeline gate pass rate
        timeline_passes = sum(1 for r in signoff_results if r.get("timeline_gates_pass", False))
        timeline_rate = timeline_passes / len(signoff_results) if signoff_results else 0
        if timeline_rate < self.config.min_timeline_gate_pass_rate:
            blockers.append(f"Timeline gate pass rate {timeline_rate:.1%} below threshold {self.config.min_timeline_gate_pass_rate:.1%}")
        
        # Check hard gate pass rate (must be 100%)
        hard_passes = sum(1 for r in signoff_results if r.get("hard_gates_pass", False))
        if hard_passes < len(signoff_results):
            blockers.append(f"Hard gates failed for {len(signoff_results) - hard_passes} elders")
        
        # Fix [High]: Leakage audit - fail-closed (missing = fail, not pass)
        # Changed from r.get("leakage_audit_pass", True) to r.get("leakage_audit_pass", False)
        leakage_failures = sum(1 for r in signoff_results if not r.get("leakage_audit_pass", False))
        if leakage_failures > 0:
            blockers.append(f"Leakage audit failed for {leakage_failures} elders")
            if self.config.auto_rollback_on_leakage_failure:
                blockers.append("CRITICAL: Leakage failure triggers auto-rollback")

        if require_real_data_evidence:
            evidence_failures = [
                str(r.get("elder_id", "unknown"))
                for r in signoff_results
                if r.get("real_data_evidence_pass", False) is not True
            ]
            if evidence_failures:
                blockers.append(
                    "Real-data canary evidence missing/invalid for elders: "
                    f"{sorted(evidence_failures)}"
                )

        return blockers
    
    def promote_to_full(self) -> RolloutState:
        """
        Promote from canary to full rollout.
        
        Raises:
            ValueError: If not in canary stage, observation incomplete, 
                       or no prior PROMOTE decision
        """
        if self._state is None or self._state.stage != RolloutStage.CANARY:
            raise ValueError("Must be in canary stage to promote to full")
        
        # Fix [High]: Require completed observation period
        if not self._can_promote():
            raise ValueError(
                "Cannot promote to full: observation period not complete. "
                f"Wait until {self._state.observation_end}"
            )
        
        # Fix [High]: Require prior PROMOTE decision from evaluate_canary
        if self._state.promotion_decision != "promote":
            raise ValueError(
                "Cannot promote to full: no prior PROMOTE decision. "
                "Run evaluate_canary() first and ensure it returns PROMOTE."
            )
        
        self._state = RolloutState(
            stage=RolloutStage.FULL,
            started_at=self._now_iso(),
            canary_elders=self._state.canary_elders,
            promotion_decision="promoted",
        )
        self._save_state()
        logger.info("T-80: Promoted to FULL rollout")
        return self._state
    
    def rollback(self, reason: str) -> RolloutState:
        """
        Rollback to shadow mode.
        
        Args:
            reason: Reason for rollback (logged and saved)
        """
        prev_stage = self._state.stage.value if self._state else "unknown"
        
        self._state = RolloutState(
            stage=RolloutStage.ROLLED_BACK,
            started_at=self._now_iso(),
            rollback_reason=f"From {prev_stage}: {reason}",
        )
        self._save_state()
        logger.warning(f"T-80: ROLLED BACK - {reason}")
        return self._state

    @staticmethod
    def _as_bool_flag(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"true", "1", "yes", "y"}:
                return True
            if token in {"false", "0", "no", "n"}:
                return False
        return False

    def evaluate_ladder_progression(
        self,
        *,
        current_rung: int,
        gate_summary: Mapping[str, Any],
    ) -> LadderProgressionResult:
        """
        Evaluate rung progression against policy gates.

        `gate_summary` expected keys:
        - mandatory_metric_floors_pass (bool)
        - open_p0_incidents (int)
        - nightly_pipeline_success_rate (float)
        - drift_alerts_within_budget (bool)
        - timeline_hard_gates_all_rooms_pass (bool)
        - phase5_acceptance_pass (bool)
        """
        target_rung = int(current_rung) + 1
        target = self.rollout_ladder_policy.get_rung(target_rung)
        blockers: list[str] = []

        if target is None:
            blockers.append(f"No configured target rung={target_rung}")
            return LadderProgressionResult(can_advance=False, target_rung=target_rung, blockers=tuple(blockers))

        if self.rollout_ladder_policy.progression.all_mandatory_metric_floors_pass:
            if not self._as_bool_flag(gate_summary.get("mandatory_metric_floors_pass", False)):
                blockers.append("mandatory_metric_floors_not_passed")

        if self.rollout_ladder_policy.progression.block_on_open_p0_incidents:
            try:
                open_p0_incidents = int(gate_summary.get("open_p0_incidents", 0) or 0)
            except (TypeError, ValueError):
                open_p0_incidents = 1
            if open_p0_incidents > 0:
                blockers.append(f"open_p0_incidents={open_p0_incidents}")

        try:
            nightly_success = float(gate_summary.get("nightly_pipeline_success_rate", 0.0) or 0.0)
        except (TypeError, ValueError):
            nightly_success = 0.0
        if nightly_success < self.rollout_ladder_policy.progression.min_nightly_pipeline_success_rate:
            blockers.append(
                "nightly_pipeline_success_rate_below_floor"
                f"({nightly_success:.4f}<{self.rollout_ladder_policy.progression.min_nightly_pipeline_success_rate:.4f})"
            )

        if self.rollout_ladder_policy.progression.require_drift_alerts_within_budget:
            if not self._as_bool_flag(gate_summary.get("drift_alerts_within_budget", False)):
                blockers.append("drift_alerts_over_budget")

        if self.rollout_ladder_policy.progression.require_timeline_hard_gates_all_rooms:
            if not self._as_bool_flag(gate_summary.get("timeline_hard_gates_all_rooms_pass", False)):
                blockers.append("timeline_hard_gates_not_passed")

        if target.requires_phase5_acceptance:
            if not self._as_bool_flag(gate_summary.get("phase5_acceptance_pass", False)):
                blockers.append("phase5_acceptance_required_for_rung2_plus")

        return LadderProgressionResult(
            can_advance=len(blockers) == 0,
            target_rung=target_rung,
            blockers=tuple(blockers),
        )

    def evaluate_auto_rollback(
        self,
        nightly_metrics: Sequence[Mapping[str, Any]],
    ) -> AutoRollbackAssessment:
        """Evaluate automatic rollback triggers on the most recent N-night window."""
        trigger_policy = self.rollout_ladder_policy.auto_rollback
        window_size = max(1, int(trigger_policy.consecutive_nights))
        if len(nightly_metrics) < window_size:
            return AutoRollbackAssessment(
                should_rollback=False,
                reason_codes=(),
                details={
                    "status": "insufficient_history",
                    "required_nights": window_size,
                    "observed_nights": len(nightly_metrics),
                },
            )

        window = list(nightly_metrics[-window_size:])
        reasons: list[str] = []
        detail: Dict[str, Any] = {"window_size": window_size}

        if trigger_policy.pipeline_success_rate_lt >= 0.0:
            pipeline_fail_all = True
            pipeline_values: list[float] = []
            for entry in window:
                raw_value = entry.get("pipeline_success_rate", None)
                try:
                    if raw_value is None:
                        value = 1.0
                    else:
                        value = float(raw_value)
                except (TypeError, ValueError):
                    value = 0.0
                pipeline_values.append(value)
                if value >= trigger_policy.pipeline_success_rate_lt:
                    pipeline_fail_all = False
            detail["pipeline_success_rate_window"] = pipeline_values
            if pipeline_fail_all:
                reasons.append("pipeline_reliability_breach")

        if trigger_policy.alert_precision_below_threshold:
            precision_breach_all = all(
                self._as_bool_flag(entry.get("alert_precision_below_threshold", False))
                for entry in window
            )
            if precision_breach_all:
                reasons.append("alert_precision_breach")

        if trigger_policy.worst_room_f1_below_floor:
            f1_breach_all = all(
                self._as_bool_flag(entry.get("worst_room_f1_below_floor", False))
                for entry in window
            )
            if f1_breach_all:
                reasons.append("worst_room_f1_breach")

        room_fail_counts: Dict[str, int] = {}
        for entry in window:
            room_map = entry.get("mae_regression_pct_by_room")
            if not isinstance(room_map, Mapping):
                room_map = {}
            seen_for_night: set[str] = set()
            for raw_room, raw_regression in room_map.items():
                room = str(raw_room).strip().lower()
                if not room or room in seen_for_night:
                    continue
                seen_for_night.add(room)
                try:
                    regression = float(raw_regression)
                except (TypeError, ValueError):
                    continue
                if regression > trigger_policy.mae_regression_pct_gt:
                    room_fail_counts[room] = room_fail_counts.get(room, 0) + 1
        failing_rooms = sorted(
            [room for room, count in room_fail_counts.items() if count >= window_size]
        )
        detail["mae_regression_rooms"] = failing_rooms
        if failing_rooms:
            reasons.append("mae_regression_breach")

        return AutoRollbackAssessment(
            should_rollback=len(reasons) > 0,
            reason_codes=tuple(reasons),
            details=detail,
        )

    def apply_auto_rollback_protection(
        self,
        *,
        nightly_metrics: Sequence[Mapping[str, Any]],
        elder_id: Optional[str] = None,
        room: Optional[str] = None,
        rooms: Optional[Sequence[str]] = None,
        run_id: Optional[str] = None,
        registry_v2: Optional[Any] = None,
        override_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Apply rollout auto-protection: rollback state + room fallback + baseline profile.
        """
        assessment = self.evaluate_auto_rollback(nightly_metrics)
        if not assessment.should_rollback:
            return {
                "status": "no_action",
                "assessment": assessment.to_dict(),
                "registry_action": None,
                "baseline_fallback": None,
            }

        trigger_reason = ",".join(assessment.reason_codes) or "auto_rollback_triggered"
        registry_action: list[Dict[str, Any]] = []
        baseline_fallback: Optional[Dict[str, Any]] = None

        target_rooms: list[str] = []
        if rooms:
            target_rooms.extend([str(token).strip() for token in rooms if str(token).strip()])
        elif room:
            target_rooms.append(str(room).strip())
        dedup_rooms = sorted(set([token for token in target_rooms if token]))

        if registry_v2 is not None and elder_id and run_id:
            for target_room in dedup_rooms:
                try:
                    action = registry_v2.rollback_and_activate_fallback(
                        elder_id=str(elder_id),
                        room=str(target_room),
                        run_id=str(run_id),
                        trigger_reason_code=trigger_reason,
                        fallback_flags={
                            "serving_mode": self.rollout_ladder_policy.fallback.serving_mode,
                            "operator_safe_mode": True,
                        },
                        metadata={"source": "t80_rollout_manager"},
                    )
                except ValueError as exc:
                    logger.warning(
                        "T-80: failed to activate room fallback during auto rollback for %s/%s: %s",
                        elder_id,
                        target_room,
                        exc,
                    )
                    action = {
                        "elder_id": str(elder_id),
                        "room": str(target_room),
                        "run_id": str(run_id),
                        "rollback_applied": False,
                        "rollback_error": None,
                        "rollback_pointer": None,
                        "fallback_state": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                registry_action.append(action)

        if (
            override_manager is not None
            and self.rollout_ladder_policy.fallback.baseline_profile == "production"
        ):
            success, event = override_manager.activate_baseline_fallback(reason=trigger_reason)
            baseline_fallback = {
                "success": bool(success),
                "event": event,
            }

        self.rollback(reason=f"Auto rollback triggered: {trigger_reason}")
        return {
            "status": "rollback_applied",
            "assessment": assessment.to_dict(),
            "registry_action": registry_action,
            "baseline_fallback": baseline_fallback,
        }
    
    def get_summary(self) -> RolloutSummary:
        """Get human-readable rollout summary."""
        if self._state is None:
            return RolloutSummary(
                current_stage="not_started",
                blockers=["Rollout not initialized"],
            )
        
        canary_progress = None
        blockers = []
        
        if self._state.stage == RolloutStage.CANARY:
            days_remaining = 0
            if self._state.observation_end:
                end = datetime.fromisoformat(self._state.observation_end)
                days_remaining = max(0, (end - datetime.now(timezone.utc)).days)
            
            canary_progress = {
                "elders": self._state.canary_elders,
                "observation_days_total": self.config.observation_days,
                "observation_days_remaining": days_remaining,
            }
            
            # Fix [Medium]: promotion_ready considers gate outcomes + coverage
            if days_remaining > 0:
                blockers.append(f"Observation in progress ({days_remaining} days remaining)")
            if self._state.promotion_decision != "promote":
                blockers.append("No PROMOTE decision from evaluate_canary()")
        
        # Fix [Medium]: Compute promotion_ready from both time + gate outcomes
        promotion_ready = (
            self._state.stage == RolloutStage.CANARY 
            and self._can_promote()
            and self._state.promotion_decision == "promote"
        )
        
        return RolloutSummary(
            current_stage=self._state.stage.value,
            canary_progress=canary_progress,
            promotion_ready=promotion_ready,
            blockers=blockers,
        )
    
    def _can_promote(self) -> bool:
        """Check if promotion from canary is possible (time-based only)."""
        if self._state is None or self._state.stage != RolloutStage.CANARY:
            return False
        if self._state.observation_end is None:
            return False
        end_time = datetime.fromisoformat(self._state.observation_end)
        return datetime.now(timezone.utc) >= end_time


def get_rollout_manager() -> T80RolloutManager:
    """Get singleton rollout manager instance."""
    return T80RolloutManager()
