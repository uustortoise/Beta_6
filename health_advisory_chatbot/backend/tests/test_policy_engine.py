"""
Unit tests for ClinicalPolicyEngine action plan generation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.schemas import HealthContext, RiskAssessment, MedicalProfile, Medication, ADLSummary
from chatbot.core.policy_engine import ClinicalPolicyEngine


def test_policy_engine_returns_risk_based_actions():
    engine = ClinicalPolicyEngine()

    context = HealthContext(
        elder_id="elder_policy_001",
        risk_assessment=RiskAssessment(
            overall_risk_score=78,
            overall_risk_level="high",
            fall_risk=82,
            medication_risk=76,
            cognitive_decline_risk=30,
            sleep_disorder_risk=20,
        ),
    )
    plan = engine.build_action_plan(
        health_context=context,
        user_question="What should I do about safety?",
    )

    action_ids = [a.id for a in plan.actions]
    assert "fall_safety_assessment" in action_ids
    assert "medication_reconciliation" in action_ids
    assert plan.confidence >= 80
    assert "Do not change medication dosing without clinician approval." in plan.contraindications
    assert plan.policy_version.startswith("2026.02")
    assert "POL-003" in plan.policy_changelog_refs


def test_policy_engine_returns_default_action_without_context():
    engine = ClinicalPolicyEngine()
    plan = engine.build_action_plan(health_context=None, user_question="General check-in")

    assert len(plan.actions) == 1
    assert plan.actions[0].id == "routine_monitoring"
    assert plan.confidence == 60.0
    assert plan.policy_version.startswith("2026.02")


def test_policy_engine_emergency_escalation_action():
    engine = ClinicalPolicyEngine()
    plan = engine.build_escalation_action_plan("emergency")

    assert len(plan.actions) == 1
    assert plan.actions[0].id == "emergency_call_911"
    assert plan.actions[0].priority == 10
    assert plan.confidence == 100.0
    assert "POL-005" in plan.policy_changelog_refs


def test_policy_engine_adds_night_safety_for_frequent_night_bathroom_visits():
    engine = ClinicalPolicyEngine()
    context = HealthContext(
        elder_id="elder_policy_002",
        risk_assessment=RiskAssessment(
            overall_risk_score=62,
            overall_risk_level="moderate",
            fall_risk=64,
        ),
        adl_summary=ADLSummary(
            period_start="2026-02-01T00:00:00",
            period_end="2026-02-08T00:00:00",
            days_analyzed=7,
            nighttime_bathroom_visits=4,
        ),
    )

    plan = engine.build_action_plan(health_context=context, user_question="I get up many times at night.")
    action_ids = [a.id for a in plan.actions]
    assert "fall_prevention_program" in action_ids
    assert "fall_night_lighting" in action_ids


def test_policy_engine_detects_major_medication_interaction_and_polypharmacy():
    engine = ClinicalPolicyEngine()
    profile = MedicalProfile(
        elder_id="elder_policy_003",
        age=82,
        chronic_conditions=["hypertension", "atrial fibrillation"],
        medications=[
            Medication(name="warfarin", dosage="5mg", frequency="daily"),
            Medication(name="aspirin", dosage="81mg", frequency="daily"),
            Medication(name="lisinopril", dosage="10mg", frequency="daily"),
            Medication(name="metformin", dosage="500mg", frequency="twice daily"),
            Medication(name="sertraline", dosage="50mg", frequency="daily"),
        ],
    )
    context = HealthContext(
        elder_id="elder_policy_003",
        medical_profile=profile,
        risk_assessment=RiskAssessment(
            overall_risk_score=71,
            overall_risk_level="high",
            medication_risk=82,
        ),
    )

    plan = engine.build_action_plan(health_context=context, user_question="Can I keep taking all these meds?")
    action_ids = [a.id for a in plan.actions]

    assert "medication_reconciliation" in action_ids
    assert "medication_interaction_review" in action_ids
    assert "medication_polypharmacy_review" in action_ids
    assert any("Do not start new over-the-counter medicines" in c for c in plan.contraindications)


def test_policy_engine_applies_cognitive_and_sleep_rules():
    engine = ClinicalPolicyEngine()
    context = HealthContext(
        elder_id="elder_policy_004",
        risk_assessment=RiskAssessment(
            overall_risk_score=69,
            overall_risk_level="high",
            cognitive_decline_risk=72,
            sleep_disorder_risk=74,
        ),
    )

    plan = engine.build_action_plan(health_context=context, user_question="Memory and sleep are getting worse.")
    action_ids = [a.id for a in plan.actions]

    assert "cognitive_clinical_evaluation" in action_ids
    assert "sleep_clinical_review" in action_ids
    assert "POL-004" in plan.policy_changelog_refs
