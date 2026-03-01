import numpy as np
import pandas as pd

from ml.gate_integration import GateEvaluationResult, GateIntegrationPipeline
from ml.policy_config import TrainingPolicy


def _build_df(num_rows: int = 400, freq: str = "15min") -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=num_rows, freq=freq)
    activity = np.where(np.arange(num_rows) % 2 == 0, "sleep", "unoccupied")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "motion": np.random.rand(num_rows),
            "temperature": np.random.rand(num_rows),
            "activity": activity,
        }
    )


def test_pre_training_failure_returns_artifact_and_legacy_keys():
    pipeline = GateIntegrationPipeline(policy=TrainingPolicy())
    df = _build_df(num_rows=100, freq="10s")
    observed_days = {pd.Timestamp("2026-01-01")}

    result = pipeline.evaluate_pre_training_gates(
        room_name="Bedroom",
        elder_id="elder_1",
        df=df,
        processed_df=df,
        observed_days=observed_days,
        seq_length=10,
    )

    serialized = result.to_dict()
    assert serialized["pre_training_pass"] is False
    assert serialized["gate_pass"] is False
    assert serialized["gate_reasons"]
    assert serialized["rejection_artifact"] is not None


def test_pre_training_class_coverage_uses_activity_column():
    policy = TrainingPolicy()
    policy.release_gate.min_observed_days = 2
    policy.release_gate.min_calibration_support = 1
    pipeline = GateIntegrationPipeline(policy=policy)

    df = _build_df(num_rows=600, freq="10min")
    observed_days = set(pd.to_datetime(df["timestamp"]).dt.floor("D").tolist())

    result = pipeline.evaluate_pre_training_gates(
        room_name="Bedroom",
        elder_id="elder_1",
        df=df,
        processed_df=df,
        observed_days=observed_days,
        seq_length=10,
    )
    serialized = result.to_dict()
    assert serialized["pre_training_pass"] is True

    class_gate = next(g for g in serialized["gate_stack"] if g["gate_name"] == "ClassCoverageGate")
    assert class_gate["passed"] is True
    assert class_gate["details"]["train_classes"] >= 2


def test_post_training_gate_parses_prefixed_class_ids():
    policy = TrainingPolicy()
    policy.calibration.min_support_per_class = 50
    pipeline = GateIntegrationPipeline(policy=policy)

    existing = GateEvaluationResult(
        run_id="run_1",
        elder_id="elder_1",
        room_name="Bedroom",
        pre_training_pass=True,
        post_training_pass=True,
        overall_pass=True,
        gate_stack=[],
    )

    result = pipeline.evaluate_post_training_gates(
        room_name="Bedroom",
        elder_id="elder_1",
        run_id="run_1",
        calibration_support={"class_0": 10, "class_1": 10},
        validation_class_support={"class_0": 5, "class_1": 5},
        existing_gate_result=existing,
    )

    serialized = result.to_dict()
    assert serialized["post_training_pass"] is False
    assert serialized["overall_pass"] is False
    assert serialized["failure_stage"] == "statistical_validity"
    assert any("Insufficient calibration support" in (r or "") for r in serialized["gate_reasons"])


def test_post_training_statistical_gate_low_support_is_not_blocking_in_pilot_profile():
    policy = TrainingPolicy()
    policy.release_gate.evidence_profile = "pilot_stage_a"
    policy.calibration.min_support_per_class = 50
    pipeline = GateIntegrationPipeline(policy=policy)

    existing = GateEvaluationResult(
        run_id="run_1",
        elder_id="elder_1",
        room_name="Entrance",
        pre_training_pass=True,
        post_training_pass=True,
        overall_pass=True,
        gate_stack=[],
    )

    result = pipeline.evaluate_post_training_gates(
        room_name="Entrance",
        elder_id="elder_1",
        run_id="run_1",
        calibration_support={"class_0": 3, "class_1": 2},
        validation_class_support={"class_0": 2},
        existing_gate_result=existing,
    )

    serialized = result.to_dict()
    assert serialized["post_training_pass"] is True
    assert serialized["overall_pass"] is True
    gate = next(g for g in serialized["gate_stack"] if g["gate_name"] == "StatisticalValidityGate")
    assert gate["passed"] is True
    assert gate["details"]["evaluation_status"] == "not_evaluated"
    assert gate["details"]["blocking_reasons"] == []
    assert gate["details"]["non_blocking_reasons"]
