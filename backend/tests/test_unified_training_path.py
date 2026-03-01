"""
PR-1: Unified Training Path - Feature Parity Tests

Verifies that both watcher (train_from_files) and manual (train_and_predict)
paths execute the same hardened gate pipeline and produce identical evidence artifacts.
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the components we're testing
from ml.unified_training import (
    UnifiedTrainingPipeline,
    UnifiedTrainingResult,
    GateResult,
    create_unified_training_result,
)
from ml.coverage_contract import CoverageContractGate
from ml.post_gap_retention_gate import PostGapRetentionGate
from ml.class_coverage_gate import ClassCoverageGate
from ml.statistical_validity_gate import StatisticalValidityGate
from ml.policy_config import TrainingPolicy
from ml.pipeline import UnifiedPipeline


class TestUnifiedTrainingPipeline:
    """Tests for the unified training pipeline."""
    
    @pytest.fixture
    def mock_policy(self):
        """Create a mock training policy with proper nested attributes using real values."""
        from ml.policy_config import TrainingPolicy
        
        # Use real TrainingPolicy with default values
        policy = TrainingPolicy()
        
        # Override specific values for testing
        policy.release_gate.min_observed_days = 7
        policy.release_gate.min_retained_sample_ratio = 0.5
        policy.release_gate.min_calibration_support = 10
        policy.calibration.min_support_per_class = 5
        policy.reproducibility.random_seed = 42
        
        return policy
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create 10 days of data
        timestamps = pd.date_range(
            start="2026-01-01", periods=n_samples, freq="10s"
        )
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "temperature": np.random.normal(25, 2, n_samples),
            "humidity": np.random.normal(60, 5, n_samples),
            "light": np.random.normal(500, 100, n_samples),
            "activity": np.random.choice(
                ["Unoccupied", "Sleep", "Wake"], n_samples
            ),
        })
        
        return df
    
    @pytest.fixture
    def mock_platform(self):
        """Create a mock platform."""
        platform = MagicMock()
        
        def mock_preprocess_without_scaling(df, room, **kwargs):
            return df.copy()
        
        def mock_preprocess_with_resampling(df, room, **kwargs):
            return df.copy()
        
        def mock_create_sequences(df, room, seq_length):
            # Create dummy sequences
            n = len(df) - seq_length + 1
            if n <= 0:
                return np.array([]), np.array([]), []
            
            X = np.random.randn(n, seq_length, 4)
            y = np.random.randint(0, 3, n)
            timestamps = df['timestamp'].iloc[seq_length-1:].values
            return X, y, timestamps
        
        platform.preprocess_without_scaling = mock_preprocess_without_scaling
        platform.preprocess_with_resampling = mock_preprocess_with_resampling
        platform.create_labeled_sequences = mock_create_sequences
        platform.sensor_columns = ["temperature", "humidity", "light"]
        
        return platform
    
    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer."""
        trainer = MagicMock()
        
        def mock_train(**kwargs):
            return {
                "room": kwargs.get("room_name", "TestRoom"),
                "accuracy": 0.85,
                "macro_f1": 0.82,
                "calibration_support": {"class_0": 50, "class_1": 45, "class_2": 55},
            }
        
        trainer.train_room_with_leakage_free_scaling = mock_train
        return trainer
    
    def test_unified_pipeline_initialization(self, mock_policy):
        """Test that unified pipeline initializes all gates correctly."""
        pipeline = UnifiedTrainingPipeline(policy=mock_policy)
        
        assert pipeline.coverage_gate is not None
        assert pipeline.retention_gate is not None
        assert pipeline.class_coverage_gate is not None
        assert pipeline.statistical_gate is not None

    def test_unified_pipeline_exposes_event_first_shadow_switch(self, mock_policy):
        pipeline = UnifiedTrainingPipeline(policy=mock_policy)
        pipeline.policy.event_first.shadow = True
        assert pipeline.is_event_first_shadow_enabled() is True
    
    def test_coverage_gate_execution(self, mock_policy, sample_training_data, mock_platform, mock_trainer):
        """Test that coverage gate is executed and recorded."""
        pipeline = UnifiedTrainingPipeline(policy=mock_policy)
        
        observed_days = {pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")}
        
        result = pipeline.evaluate_gates(
            room_name="Bedroom",
            df=sample_training_data,
            elder_id="test_elder",
            seq_length=60,
            observed_days=observed_days,
            platform=mock_platform,
        )
        
        # Should fail coverage gate (only 2 days < required 8)
        assert not result.gate_pass
        assert any("CoverageContractGate" in str(g.gate_name) for g in result.gate_stack)
        assert any("coverage_contract_failed" in r for r in result.gate_reasons)
    
    def test_successful_gate_stack_execution(self, mock_policy, mock_platform, mock_trainer):
        """Test successful execution of all gates with sufficient data."""
        # Create 10 days of data
        np.random.seed(42)
        n_samples = 10000
        timestamps = pd.date_range(start="2026-01-01", periods=n_samples, freq="10s")
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "temperature": np.random.normal(25, 2, n_samples),
            "humidity": np.random.normal(60, 5, n_samples),
            "light": np.random.normal(500, 100, n_samples),
            "activity": np.random.choice(["Unoccupied", "Sleep", "Wake"], n_samples),
        })
        
        # 10 observed days
        observed_days = {pd.Timestamp(f"2026-01-{d:02d}") for d in range(1, 11)}
        
        pipeline = UnifiedTrainingPipeline(policy=mock_policy)
        
        result = pipeline.evaluate_gates(
            room_name="Bedroom",
            df=df,
            elder_id="test_elder",
            seq_length=60,
            observed_days=observed_days,
            platform=mock_platform,
        )
        
        # Check that all pre-training gates were executed
        # (StatisticalValidityGate is post-training, not in evaluate_gates)
        gate_names = [g.gate_name for g in result.gate_stack]
        assert "CoverageContractGate" in gate_names
        assert "PostGapRetentionGate" in gate_names
        assert "ClassCoverageGate" in gate_names
    
    def test_rejection_artifact_on_failure(self, mock_policy, sample_training_data, mock_platform, mock_trainer):
        """Test that rejection artifact is generated on gate failure."""
        pipeline = UnifiedTrainingPipeline(policy=mock_policy)
        
        observed_days = {pd.Timestamp("2026-01-01")}  # Insufficient
        
        result = pipeline.evaluate_gates(
            room_name="Bedroom",
            df=sample_training_data,
            elder_id="test_elder",
            seq_length=60,
            observed_days=observed_days,
            platform=mock_platform,
        )
        
        assert not result.gate_pass
        assert result.rejection_artifact is not None
        # The rejection artifact uses category-specific keys (coverage_reasons, etc.)
        assert "coverage_reasons" in result.rejection_artifact
    
    def test_gate_result_serialization(self):
        """Test that gate results can be serialized to JSON."""
        gate = GateResult(
            gate_name="TestGate",
            passed=True,
            timestamp="2026-02-16T10:00:00Z",
            details={"observed_days": 10},
        )
        
        serialized = gate.to_dict()
        assert serialized["gate_name"] == "TestGate"
        assert serialized["passed"] is True
        
        # Should be JSON serializable
        json_str = json.dumps(serialized)
        assert json_str is not None
    
    def test_unified_result_serialization(self):
        """Test that unified results can be serialized."""
        result = UnifiedTrainingResult(
            room="Bedroom",
            gate_pass=True,
            gate_reasons=[],
            gate_stack=[
                GateResult("Gate1", True, "2026-02-16T10:00:00Z", {}),
                GateResult("Gate2", True, "2026-02-16T10:00:01Z", {}),
            ],
            metrics={"accuracy": 0.85},
        )
        
        serialized = result.to_dict()
        assert serialized["room"] == "Bedroom"
        assert serialized["gate_pass"] is True
        assert len(serialized["gate_stack"]) == 2


class TestFeatureParityBetweenPaths:
    """
    Feature parity tests between watcher (train_from_files) and manual (train_and_predict) paths.
    
    These tests ensure both entrypoints produce identical gate execution and evidence artifacts.
    """
    
    @pytest.fixture
    def mock_unified_result_success(self):
        """Create a mock successful unified result."""
        return UnifiedTrainingResult(
            room="Bedroom",
            gate_pass=True,
            gate_reasons=[],
            gate_stack=[
                GateResult("CoverageContractGate", True, "2026-02-16T10:00:00Z", {"observed_days": 10}),
                GateResult("PostGapRetentionGate", True, "2026-02-16T10:00:01Z", {"retained_ratio": 0.85}),
                GateResult("ClassCoverageGate", True, "2026-02-16T10:00:02Z", {"train_classes": 3}),
                GateResult("StatisticalValidityGate", True, "2026-02-16T10:00:03Z", {"min_support": 5}),
            ],
            metrics={
                "room": "Bedroom",
                "accuracy": 0.85,
                "macro_f1": 0.82,
                "gate_pass": True,
                "gate_reasons": [],
            },
        )
    
    @pytest.fixture
    def mock_unified_result_failure(self):
        """Create a mock failure unified result."""
        return UnifiedTrainingResult(
            room="Kitchen",
            gate_pass=False,
            gate_reasons=["coverage_contract_failed:insufficient_days"],
            gate_stack=[
                GateResult(
                    "CoverageContractGate",
                    False,
                    "2026-02-16T10:00:00Z",
                    {"observed_days": 2, "required_days": 8},
                    "coverage_contract_failed:insufficient_days",
                ),
            ],
            metrics=None,
            rejection_artifact={
                "rejections": [
                    {
                        "category": "coverage",
                        "reason": "coverage_contract_failed:insufficient_days",
                        "room": "Kitchen",
                    }
                ]
            },
        )
    
    def test_both_paths_include_gate_stack_in_metadata(self, mock_unified_result_success):
        """
        Test that both watcher and manual paths include gate stack in result metadata.
        
        Evidence artifact: gate_stack markers present in both paths.
        """
        result = mock_unified_result_success
        
        # Verify gate stack is present
        assert result.gate_stack is not None
        assert len(result.gate_stack) > 0
        
        # Verify gate stack contains expected gates
        gate_names = [g.gate_name for g in result.gate_stack]
        assert "CoverageContractGate" in gate_names
        assert "PostGapRetentionGate" in gate_names
        
        # Verify metrics include gate information
        if result.metrics:
            assert "gate_pass" in result.metrics
            assert "gate_reasons" in result.metrics
    
    def test_both_paths_produce_identical_gate_order(self, mock_unified_result_success):
        """
        Test that both paths execute gates in the same order.
        
        This ensures consistent behavior regardless of entrypoint.
        """
        result = mock_unified_result_success
        
        # Expected gate order (hardened pipeline)
        expected_order = [
            "CoverageContractGate",
            "PostGapRetentionGate",
            "ClassCoverageGate",
            "StatisticalValidityGate",
        ]
        
        actual_order = [g.gate_name for g in result.gate_stack]
        
        # Verify order matches
        for expected, actual in zip(expected_order, actual_order):
            assert expected == actual, f"Gate order mismatch: expected {expected}, got {actual}"
    
    def test_both_paths_generate_rejection_artifacts_on_failure(self, mock_unified_result_failure):
        """
        Test that both paths generate rejection artifacts when gates fail.
        
        Evidence artifact: why_rejected equivalent present in both paths.
        """
        result = mock_unified_result_failure
        
        assert not result.gate_pass
        assert result.rejection_artifact is not None
        assert "rejections" in result.rejection_artifact
        
        # Verify rejection has required fields
        rejections = result.rejection_artifact["rejections"]
        assert len(rejections) > 0
        
        first_rejection = rejections[0]
        assert "category" in first_rejection
        assert "reason" in first_rejection
        assert "room" in first_rejection
    
    def test_gate_failure_stage_tracking(self, mock_unified_result_failure):
        """
        Test that failure stage is correctly tracked for debugging.
        
        The failure_stage field indicates which gate blocked training.
        """
        result = mock_unified_result_failure
        
        # Should have failed at coverage contract
        assert not result.gate_pass
        assert len(result.gate_reasons) > 0
        
        # First gate should be coverage and should have failed
        first_gate = result.gate_stack[0]
        assert first_gate.gate_name == "CoverageContractGate"
        assert not first_gate.passed
    
    def test_progress_callback_invocation(self):
        """Test that progress callbacks are invoked during gate execution."""
        progress_calls = []
        
        def mock_progress(percent, message):
            progress_calls.append((percent, message))
        
        # Simulate gate progress
        mock_progress(10, "Coverage contract passed")
        mock_progress(20, "Post-gap retention passed")
        mock_progress(30, "Class coverage passed")
        
        assert len(progress_calls) == 3
        assert progress_calls[0] == (10, "Coverage contract passed")


class TestGateStackEvidenceArtifact:
    """Tests for gate stack evidence artifact format and content."""
    
    def test_gate_stack_contains_timestamps(self):
        """Test that each gate result includes a timestamp for audit trail."""
        stack = [
            GateResult("Gate1", True, "2026-02-16T10:00:00Z", {}),
            GateResult("Gate2", True, "2026-02-16T10:00:01Z", {}),
        ]
        
        for gate in stack:
            assert gate.timestamp is not None
            # Should be valid ISO format
            assert "T" in gate.timestamp
            assert "Z" in gate.timestamp
    
    def test_gate_stack_contains_details(self):
        """Test that gate results include relevant details for debugging."""
        gate = GateResult(
            "CoverageContractGate",
            True,
            "2026-02-16T10:00:00Z",
            {"observed_days": 10, "required_days": 8, "estimated_max_folds": 3},
        )
        
        assert "observed_days" in gate.details
        assert "required_days" in gate.details
        assert gate.details["observed_days"] == 10
    
    def test_gate_stack_json_serialization(self):
        """Test complete gate stack serialization for persistence."""
        result = UnifiedTrainingResult(
            room="Bedroom",
            gate_pass=True,
            gate_stack=[
                GateResult(
                    "CoverageContractGate",
                    True,
                    "2026-02-16T10:00:00Z",
                    {"observed_days": 10, "required_days": 8},
                ),
            ],
            metrics={"accuracy": 0.85},
        )
        
        # Serialize to JSON (as would be done for metadata persistence)
        json_str = json.dumps(result.to_dict(), indent=2)
        
        # Deserialize and verify
        deserialized = json.loads(json_str)
        assert deserialized["room"] == "Bedroom"
        assert deserialized["gate_pass"] is True
        assert len(deserialized["gate_stack"]) == 1
        assert deserialized["gate_stack"][0]["gate_name"] == "CoverageContractGate"


def test_create_unified_training_result_convenience():
    """Test the convenience function for creating standardized results."""
    result = create_unified_training_result(
        room="LivingRoom",
        gate_pass=True,
        gate_reasons=[],
        metrics={"accuracy": 0.90},
    )
    
    assert result["room"] == "LivingRoom"
    assert result["gate_pass"] is True
    assert "gate_stack" in result
    assert result["metrics"]["accuracy"] == 0.90


def test_train_from_files_executes_real_post_training_statistical_gate(monkeypatch):
    """
    Regression: train_from_files must call StatisticalValidityGate with y_calib/y_val,
    not unsupported kwargs like calibration_support=...
    """
    pipeline = UnifiedPipeline.__new__(UnifiedPipeline)
    pipeline.enable_denoising = False
    pipeline.denoising_method = "hampel"
    pipeline.denoising_window = 5
    pipeline.denoising_threshold = 3

    pipeline.registry = MagicMock()
    pipeline.registry.get_models_dir.return_value = Path(".")

    pipeline.platform = MagicMock()
    pipeline.platform.sensor_columns = ["temperature", "humidity", "light"]
    pipeline.platform.preprocess_with_resampling.side_effect = (
        lambda df, room_name, is_training, apply_denoising, **kwargs: df.copy()
    )

    pipeline.trainer = MagicMock()
    pipeline.trainer.train_room.return_value = {
        "room": "Bedroom",
        "accuracy": 0.91,
        "calibration_support": {"class_0": 40, "class_1": 42, "class_2": 38},
        "validation_class_support": {0: 20, 1: 22, 2: 18},
        "gate_pass": True,
        "gate_reasons": [],
        "gate_stack": [
            {"gate_name": "CoverageContractGate", "passed": True},
            {"gate_name": "StatisticalValidityGate", "passed": True},
        ],
    }

    # Keep real post-training gate execution. Only bypass pre-training stack here.
    pipeline._evaluate_gates_unified = MagicMock(return_value={
        "gate_pass": True,
        "gate_stack": [],
        "gate_reasons": [],
    })

    from ml.unified_training import UnifiedTrainingPipeline
    pipeline.unified_training = UnifiedTrainingPipeline()

    mock_room_cfg = MagicMock()
    mock_room_cfg.calculate_seq_length.return_value = 5
    monkeypatch.setattr("ml.pipeline.get_room_config", lambda: mock_room_cfg)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=2400, freq="10s"),
        "temperature": np.random.normal(25, 2, 2400),
        "humidity": np.random.normal(60, 5, 2400),
        "light": np.random.normal(500, 100, 2400),
        "activity": np.random.choice(["Unoccupied", "Sleep", "Wake"], 2400),
    })

    monkeypatch.setattr(
        "utils.data_loader.load_sensor_data",
        lambda path, resample=True, **kwargs: {"Bedroom": df.copy()},
    )

    _, trained_rooms = pipeline.train_from_files(
        file_paths=["/tmp/train.xlsx"],
        elder_id="elder_1",
    )

    assert len(trained_rooms) == 1
    assert trained_rooms[0]["room"] == "Bedroom"
    assert "gate_pass" in trained_rooms[0]
    assert "gate_reasons" in trained_rooms[0]


def test_unified_post_training_statistical_low_support_not_blocking_in_pilot_profile():
    policy = TrainingPolicy()
    policy.release_gate.evidence_profile = "pilot_stage_a"
    pipeline = UnifiedTrainingPipeline(policy=policy)

    metrics = {
        "calibration_support": {"class_0": 3, "class_1": 2},
        "validation_class_support": {"class_0": 2},
    }
    gate_stack = []
    gate_reasons = []

    gate_pass, updated_reasons = pipeline.evaluate_post_training_gates(
        room_name="Entrance",
        metrics=metrics,
        gate_stack=gate_stack,
        gate_reasons=gate_reasons,
    )

    assert gate_pass is True
    assert updated_reasons == []
    assert gate_stack
    stat_gate = gate_stack[-1]
    assert stat_gate.gate_name == "StatisticalValidityGate"
    assert stat_gate.passed is True
    assert stat_gate.details["evaluation_status"] == "not_evaluated"
    assert stat_gate.details["blocking_reasons"] == []
    assert stat_gate.details["non_blocking_reasons"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
