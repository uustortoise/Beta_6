import numpy as np

from ml.class_coverage_gate import ClassCoverageGate
from ml.gate_integration import GateIntegrationPipeline
from ml.policy_config import TrainingPolicy


def test_class_coverage_gate_can_warn_instead_of_block_for_low_val_support():
    gate = ClassCoverageGate(
        critical_classes=[0, 1],
        min_train_support=1,
        min_val_support=5,
        min_calib_support=5,
        block_on_val_support=False,
    )

    y_train = np.array([0] * 50 + [1] * 50, dtype=np.int32)
    y_val = np.array([0] * 20 + [1], dtype=np.int32)
    result = gate.evaluate(y_train=y_train, y_val=y_val, y_calib=None, room_name="Bedroom")

    assert result["passes"] is True
    assert any("insufficient validation support" in msg.lower() for msg in result["warnings"])


def test_class_coverage_gate_blocks_low_val_support_in_strict_mode():
    gate = ClassCoverageGate(
        critical_classes=[0, 1],
        min_train_support=1,
        min_val_support=5,
        min_calib_support=5,
        block_on_val_support=True,
    )

    y_train = np.array([0] * 50 + [1] * 50, dtype=np.int32)
    y_val = np.array([0] * 20 + [1], dtype=np.int32)
    result = gate.evaluate(y_train=y_train, y_val=y_val, y_calib=None, room_name="Bedroom")

    assert result["passes"] is False
    assert any("insufficient validation support" in msg.lower() for msg in result["reasons"])


def test_gate_integration_uses_relaxed_class_coverage_in_pilot_profile():
    policy = TrainingPolicy()
    policy.release_gate.evidence_profile = "pilot_stage_a"
    pipeline = GateIntegrationPipeline(policy=policy)

    assert pipeline.class_coverage_gate.min_val_support == 1
    assert pipeline.class_coverage_gate.block_on_val_support is False

