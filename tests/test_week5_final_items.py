"""
Week 5 Final Items Tests (Items 12, 16, 17, 18)

Tests for:
- Item 12: Registry Canonical State Validator in CI
- Item 16: Policy Complexity Governance (presets)
- Item 17: Ops Runbook + Pilot Override Rollback
- Item 18: Transformer Head A/B (Architecture Research)
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


# ============================================================================
# Item 12: Registry Canonical State Validator Tests
# ============================================================================

class TestRegistryValidator:
    """Tests for registry validation."""
    
    @pytest.fixture
    def mock_registry(self, tmp_path):
        """Create a mock registry for testing."""
        registry = MagicMock()
        registry.base_path = tmp_path / "models"
        registry.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create registry state
        registry.state = {
            "current_version": "v2.0.0",
            "versions": {
                "v1.0.0": {
                    "promoted": False,
                    "aliases": ["stable", "previous"],
                },
                "v2.0.0": {
                    "promoted": True,
                    "aliases": ["latest", "production"],
                },
            },
            "aliases": {
                "latest": "v2.0.0",
                "production": "v2.0.0",
                "stable": "v1.0.0",
                "previous": "v1.0.0",
            },
        }
        return registry
    
    def test_registry_consistency_error(self):
        """Test RegistryConsistencyError is raised correctly."""
        from ml.registry_validator import RegistryConsistencyError
        
        with pytest.raises(RegistryConsistencyError) as exc_info:
            raise RegistryConsistencyError("Test error", "test_check")
        
        assert "Test error" in str(exc_info.value)
        assert exc_info.value.check_name == "test_check"
    
    def test_validate_current_version_consistency(self, mock_registry):
        """Test validation of current_version consistency."""
        from ml.registry_validator import RegistryValidator
        
        validator = RegistryValidator(mock_registry)
        validator._validate_current_version_consistency()
        
        assert len(validator.errors) == 0
    
    def test_validate_current_version_missing(self, mock_registry):
        """Test error when current_version is missing."""
        from ml.registry_validator import RegistryValidator
        
        del mock_registry.state["current_version"]
        
        validator = RegistryValidator(mock_registry)
        validator._validate_current_version_consistency()
        
        assert len(validator.errors) == 1
        assert "current_version" in validator.errors[0].lower()
    
    def test_validate_promoted_flags(self, mock_registry):
        """Test validation of promoted flags."""
        from ml.registry_validator import RegistryValidator
        
        validator = RegistryValidator(mock_registry)
        validator._validate_promoted_flags()
        
        assert len(validator.errors) == 0
    
    def test_validate_promoted_flags_mismatch(self, mock_registry):
        """Test error when promoted flag is inconsistent."""
        from ml.registry_validator import RegistryValidator
        
        # Set current_version to non-promoted version
        mock_registry.state["current_version"] = "v1.0.0"
        
        validator = RegistryValidator(mock_registry)
        validator._validate_promoted_flags()
        
        assert len(validator.errors) == 1
        assert "promoted" in validator.errors[0].lower()
    
    def test_validate_alias_consistency(self, mock_registry):
        """Test validation of alias consistency."""
        from ml.registry_validator import RegistryValidator
        
        validator = RegistryValidator(mock_registry)
        validator._validate_alias_consistency()
        
        assert len(validator.errors) == 0
    
    def test_validate_alias_dangling(self, mock_registry):
        """Test warning when alias points to non-existent version."""
        from ml.registry_validator import RegistryValidator
        
        mock_registry.state["aliases"]["broken"] = "v999.0.0"
        
        validator = RegistryValidator(mock_registry)
        validator._validate_alias_consistency()
        
        # Should have warning but not error
        assert len(validator.warnings) >= 1
        assert "dangling" in validator.warnings[0].lower()
    
    def test_validate_all(self, mock_registry):
        """Test complete validation."""
        from ml.registry_validator import RegistryValidator
        
        validator = RegistryValidator(mock_registry)
        valid, errors, warnings = validator.validate_all()
        
        assert valid is True
        assert len(errors) == 0
        assert len(warnings) == 0
    
    def test_chaos_test_interrupted_write(self, mock_registry):
        """Test chaos test for interrupted write recovery."""
        from ml.registry_validator import RegistryValidator
        
        validator = RegistryValidator(mock_registry)
        
        # Should handle corrupted state gracefully
        passed, recovery_time = validator.chaos_test_interrupted_write_recovery()
        
        assert passed is True
        assert recovery_time >= 0
    
    def test_validate_registry_state_function(self, mock_registry):
        """Test the convenience function."""
        from ml.registry_validator import validate_registry_state
        
        result = validate_registry_state(mock_registry)
        
        assert result["valid"] is True
        assert "checks_run" in result
        assert "timestamp" in result


# ============================================================================
# Item 16: Policy Complexity Governance Tests
# ============================================================================

class TestPolicyPresets:
    """Tests for policy complexity governance."""
    
    @pytest.fixture
    def base_policy(self):
        """Create base policy for testing."""
        from ml.policy_config import TrainingPolicy
        return TrainingPolicy()
    
    def test_policy_preset_enum(self):
        """Test PolicyPreset enum values."""
        from ml.policy_presets import PolicyPreset
        
        assert PolicyPreset.CONSERVATIVE.value == "conservative"
        assert PolicyPreset.BALANCED.value == "balanced"
        assert PolicyPreset.AGGRESSIVE.value == "aggressive"
    
    def test_preset_manager_initialization(self):
        """Test PolicyPresetManager initialization."""
        from ml.policy_presets import PolicyPresetManager, PolicyPreset
        
        manager = PolicyPresetManager(environment="production")
        
        assert manager.environment == "production"
        assert PolicyPreset.AGGRESSIVE not in manager.allowed_presets
        
        # Development allows all
        dev_manager = PolicyPresetManager(environment="development")
        assert PolicyPreset.AGGRESSIVE in dev_manager.allowed_presets
    
    def test_apply_conservative_preset(self, base_policy):
        """Test applying conservative preset."""
        from ml.policy_presets import PolicyPresetManager, PolicyPreset
        
        manager = PolicyPresetManager()
        result = manager.apply_preset(base_policy, PolicyPreset.CONSERVATIVE)
        
        assert result._applied_preset == "conservative"
        assert result.data_viability.min_observed_days == 7
        assert result.data_viability.min_post_gap_rows == 10000
        assert result.release_gate.min_training_days == 3.0
    
    def test_apply_balanced_preset(self, base_policy):
        """Test applying balanced preset."""
        from ml.policy_presets import PolicyPresetManager, PolicyPreset
        
        manager = PolicyPresetManager()
        result = manager.apply_preset(base_policy, PolicyPreset.BALANCED)
        
        assert result._applied_preset == "balanced"
        assert result.data_viability.min_observed_days == 5
    
    def test_apply_aggressive_preset(self, base_policy):
        """Test applying aggressive preset."""
        from ml.policy_presets import PolicyPresetManager, PolicyPreset
        
        manager = PolicyPresetManager(environment="development")
        result = manager.apply_preset(base_policy, PolicyPreset.AGGRESSIVE)
        
        assert result._applied_preset == "aggressive"
        assert result.data_viability.min_observed_days == 2
        assert result.release_gate.min_training_days == 1.0
    
    def test_aggressive_preset_blocked_in_production(self, base_policy):
        """Test that aggressive preset is blocked in production."""
        from ml.policy_presets import PolicyPresetManager, PolicyPreset
        
        manager = PolicyPresetManager(environment="production")
        
        with pytest.raises(ValueError) as exc_info:
            manager.apply_preset(base_policy, PolicyPreset.AGGRESSIVE)
        
        assert "not allowed" in str(exc_info.value).lower()
    
    def test_validate_environment_overrides_allowed(self, base_policy):
        """Test validation of allowed environment overrides."""
        from ml.policy_presets import PolicyPresetManager, PolicyPreset
        
        manager = PolicyPresetManager(environment="production")
        environ = {"TRAINING_PROFILE": "conservative"}  # Allowed
        
        valid, violations = manager.validate_environment_overrides(
            PolicyPreset.CONSERVATIVE, environ
        )
        
        assert valid is True
        assert len(violations) == 0
    
    def test_validate_environment_overrides_blocked(self, base_policy):
        """Test validation catches blocked overrides."""
        from ml.policy_presets import PolicyPresetManager, PolicyPreset
        
        manager = PolicyPresetManager(environment="production")
        environ = {"MIN_OBSERVED_DAYS": "1"}  # Blocked
        
        valid, violations = manager.validate_environment_overrides(
            PolicyPreset.CONSERVATIVE, environ
        )
        
        assert valid is False
        assert len(violations) > 0
        assert "blocked" in violations[0].lower()
    
    def test_get_preset_info(self):
        """Test getting preset information."""
        from ml.policy_presets import PolicyPresetManager, PolicyPreset
        
        manager = PolicyPresetManager()
        info = manager.get_preset_info(PolicyPreset.CONSERVATIVE)
        
        assert info["preset"] == "conservative"
        assert "description" in info
        assert "overrides" in info
        assert "allowed_in_environment" in info
    
    def test_list_presets(self):
        """Test listing all presets."""
        from ml.policy_presets import PolicyPresetManager
        
        manager = PolicyPresetManager()
        presets = manager.list_presets()
        
        assert len(presets) == 3
        preset_names = {p["preset"] for p in presets}
        assert preset_names == {"conservative", "balanced", "aggressive"}


# ============================================================================
# Item 17: Ops Runbook + Pilot Override Rollback Tests
# ============================================================================

class TestPilotOverrideManager:
    """Tests for pilot override management."""
    
    @pytest.fixture
    def temp_state_file(self, tmp_path):
        """Create temporary state file."""
        return tmp_path / "pilot_state.json"
    
    @pytest.fixture
    def manager(self, temp_state_file):
        """Create manager with temp state file."""
        from ml.pilot_override_manager import PilotOverrideManager
        return PilotOverrideManager(state_file=temp_state_file, default_expiry_hours=1)
    
    def test_activate_pilot(self, manager):
        """Test pilot profile activation."""
        success, message = manager.activate_pilot(
            reason="Testing new features",
            duration_hours=4,
        )
        
        assert success is True
        assert "Pilot profile activated" in message
        assert "Testing new features" in message
        assert "4 hours" in message
        assert "rollback" in message.lower()
    
    def test_get_status_active(self, manager):
        """Test getting status when pilot is active."""
        manager.activate_pilot(reason="Test", duration_hours=4)
        
        status = manager.get_status()
        
        assert status["active"] is True
        assert status["profile"] == "pilot"
        assert status["reason"] == "Test"
        assert status["is_expired"] is False
    
    def test_get_status_inactive(self, manager):
        """Test getting status when no pilot is active."""
        status = manager.get_status()
        
        assert status["active"] is False
        assert status["profile"] == "production"
    
    def test_rollback(self, manager):
        """Test rollback to production."""
        manager.activate_pilot(reason="Test", duration_hours=4)
        
        success, message = manager.rollback()
        
        assert success is True
        assert "Rolled back" in message
        
        # Verify state cleared
        status = manager.get_status()
        assert status["active"] is False
    
    def test_auto_rollback_on_expiry(self, manager, monkeypatch):
        """Test automatic rollback when override expires."""
        manager.activate_pilot(reason="Test", duration_hours=1)
        
        # Simulate expiry by modifying the state
        from datetime import datetime, timedelta
        expired_time = (datetime.utcnow() - timedelta(hours=2)).isoformat()
        
        from ml.pilot_override_manager import OverrideState
        state = OverrideState(
            profile="pilot",
            activated_at=expired_time,
            expires_at=expired_time,
            activated_by="test",
            reason="Test",
            auto_rollback=True,
        )
        manager._state = state
        manager._save_state()
        
        did_rollback, message = manager.check_and_auto_rollback()
        
        assert did_rollback is True
        assert "Rolled back" in message or "Auto-rolling back" in message
    
    def test_auto_rollback_disabled(self, manager):
        """Test that auto-rollback can be disabled."""
        manager.activate_pilot(reason="Test", duration_hours=1, auto_rollback=False)
        
        # Simulate expiry
        from datetime import datetime, timedelta
        expired_time = (datetime.utcnow() - timedelta(hours=2)).isoformat()
        
        from ml.pilot_override_manager import OverrideState
        state = OverrideState(
            profile="pilot",
            activated_at=expired_time,
            expires_at=expired_time,
            activated_by="test",
            reason="Test",
            auto_rollback=False,  # Disabled
        )
        manager._state = state
        manager._save_state()
        
        did_rollback, message = manager.check_and_auto_rollback()
        
        assert did_rollback is False
        assert "disabled" in message.lower() or "Manual rollback" in message
    
    def test_get_reminder_active(self, manager):
        """Test getting reminder when pilot is active."""
        manager.activate_pilot(reason="Test", duration_hours=4)
        
        reminder = manager.get_reminder()
        
        assert reminder is not None
        assert "PILOT MODE ACTIVE" in reminder
        assert "Test" in reminder
        assert "rollback" in reminder.lower()
    
    def test_get_reminder_expired(self, manager):
        """Test getting reminder when pilot has expired."""
        manager.activate_pilot(reason="Test", duration_hours=1)
        
        # Simulate expiry
        from datetime import datetime, timedelta
        expired_time = (datetime.utcnow() - timedelta(hours=2)).isoformat()
        
        from ml.pilot_override_manager import OverrideState
        state = OverrideState(
            profile="pilot",
            activated_at=expired_time,
            expires_at=expired_time,
            activated_by="test",
            reason="Test",
            auto_rollback=True,
        )
        manager._state = state
        manager._save_state()
        
        reminder = manager.get_reminder()
        
        assert reminder is not None
        assert "EXPIRED" in reminder
    
    def test_set_training_profile_pilot(self, manager):
        """Test set_training_profile function for pilot."""
        from ml.pilot_override_manager import set_training_profile
        
        success, message = set_training_profile(
            "pilot",
            reason="Testing",
            duration_hours=2,
        )
        
        assert success is True
        assert "Pilot profile activated" in message
    
    def test_set_training_profile_production(self, manager):
        """Test set_training_profile function for production."""
        from ml.pilot_override_manager import set_training_profile
        
        # First activate pilot
        set_training_profile("pilot", reason="Test")
        
        # Then rollback
        success, message = set_training_profile("production")
        
        assert success is True
        assert "Rolled back" in message
    
    def test_set_training_profile_requires_reason(self, manager):
        """Test that pilot activation requires reason."""
        from ml.pilot_override_manager import set_training_profile
        
        success, message = set_training_profile("pilot")
        
        assert success is False
        assert "Reason required" in message


# ============================================================================
# Item 18: Transformer Head A/B Tests
# ============================================================================

class TestTransformerHeadAB:
    """Tests for transformer head A/B testing."""
    
    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        train_data = {
            'train': pd.DataFrame({
                'features': list(np.random.randn(n_samples, 100, 64).astype(np.float32)),
                'label': np.random.randint(0, 12, n_samples),
            }),
            'val': pd.DataFrame({
                'features': list(np.random.randn(20, 100, 64).astype(np.float32)),
                'label': np.random.randint(0, 12, 20),
            }),
        }
        test_data = pd.DataFrame({
            'features': list(np.random.randn(20, 100, 64).astype(np.float32)),
            'label': np.random.randint(0, 12, 20),
        })
        
        return train_data, test_data
    
    def test_pooling_strategy_enum(self):
        """Test PoolingStrategy enum."""
        from ml.transformer_head_ab import PoolingStrategy
        
        assert PoolingStrategy.GLOBAL_AVG.value == "global_average_pooling"
        assert PoolingStrategy.ATTENTION.value == "attention_pooling"
        assert PoolingStrategy.LAST_TOKEN.value == "last_token"
    
    def test_ab_run_config(self):
        """Test ABRunConfig dataclass."""
        from ml.transformer_head_ab import ABRunConfig, PoolingStrategy
        
        config = ABRunConfig(
            pooling=PoolingStrategy.ATTENTION,
            random_seed=42,
            model_params={"hidden_dim": 128},
        )
        
        assert config.pooling == PoolingStrategy.ATTENTION
        assert config.random_seed == 42
        assert config.model_params["hidden_dim"] == 128
    
    def test_ab_comparison_result(self):
        """Test ABComparisonResult aggregation."""
        from ml.transformer_head_ab import ABComparisonResult, PoolingStrategy
        
        result = ABComparisonResult(
            strategy=PoolingStrategy.GLOBAL_AVG,
            n_runs=3,
            val_f1_mean=0.85,
            val_f1_std=0.02,
            runs=[],
        )
        
        assert result.strategy == PoolingStrategy.GLOBAL_AVG
        assert result.val_f1_mean == 0.85
        assert result.n_runs == 3
    
    def test_ab_test_report(self):
        """Test ABTestReport generation."""
        from ml.transformer_head_ab import (
            ABTestReport, ABComparisonResult, PoolingStrategy
        )
        
        report = ABTestReport(
            comparison_results={
                PoolingStrategy.GLOBAL_AVG: ABComparisonResult(
                    strategy=PoolingStrategy.GLOBAL_AVG,
                    n_runs=3,
                    val_f1_mean=0.85,
                    val_f1_std=0.02,
                    runs=[],
                ),
            },
            recommendation=PoolingStrategy.GLOBAL_AVG,
            recommendation_confidence=0.9,
        )
        
        assert report.recommendation == PoolingStrategy.GLOBAL_AVG
        assert report.recommendation_confidence == 0.9
        
        # Test JSON export
        json_str = report.to_json()
        assert "global_average_pooling" in json_str
        
        data = json.loads(json_str)
        assert data["recommendation"] == "global_average_pooling"
    
    @pytest.mark.skipif(
        not sys.modules.get("tensorflow", None),
        reason="TensorFlow not available"
    )
    def test_transformer_head_ab_initialization(self, tmp_path):
        """Test A/B tester initialization."""
        import tensorflow as tf
        from ml.transformer_head_ab import TransformerHeadAB
        
        tf.keras.utils.set_random_seed(42)
        
        ab = TransformerHeadAB(
            num_classes=12,
            num_runs=2,
            epochs=2,
            output_dir=str(tmp_path),
        )
        
        assert ab.num_classes == 12
        assert ab.num_runs == 2
        assert ab.epochs == 2
    
    def test_determine_recommendation(self, tmp_path):
        """Test recommendation logic."""
        from ml.transformer_head_ab import (
            TransformerHeadAB, ABComparisonResult, PoolingStrategy
        )
        
        ab = TransformerHeadAB(num_runs=1, epochs=1, output_dir=str(tmp_path))
        
        # Create mock comparison results
        comparison_results = {
            PoolingStrategy.GLOBAL_AVG: ABComparisonResult(
                strategy=PoolingStrategy.GLOBAL_AVG,
                n_runs=1,
                val_f1_mean=0.90,  # Higher
                val_f1_std=0.01,
                convergence_epoch_mean=5,
                training_time_mean=10,
                runs=[],
            ),
            PoolingStrategy.ATTENTION: ABComparisonResult(
                strategy=PoolingStrategy.ATTENTION,
                n_runs=1,
                val_f1_mean=0.85,  # Lower
                val_f1_std=0.02,
                convergence_epoch_mean=6,
                training_time_mean=12,
                runs=[],
            ),
        }
        
        rec, conf = ab._determine_recommendation(comparison_results)
        
        # Should recommend GLOBAL_AVG due to higher F1
        assert rec == PoolingStrategy.GLOBAL_AVG
        assert conf > 0
    
    def test_generate_summary(self, tmp_path):
        """Test summary generation."""
        from ml.transformer_head_ab import (
            TransformerHeadAB, ABComparisonResult, PoolingStrategy
        )
        
        ab = TransformerHeadAB(num_runs=1, epochs=1, output_dir=str(tmp_path))
        
        comparison_results = {
            PoolingStrategy.GLOBAL_AVG: ABComparisonResult(
                strategy=PoolingStrategy.GLOBAL_AVG,
                n_runs=1,
                val_f1_mean=0.90,
                val_f1_std=0.01,
                convergence_epoch_mean=5,
                training_time_mean=10,
                runs=[],
            ),
        }
        
        summary = ab._generate_summary(
            comparison_results,
            PoolingStrategy.GLOBAL_AVG
        )
        
        assert "TRANSFORMER HEAD A/B TEST RESULTS" in summary
        assert "GLOBAL" in summary
        assert "RECOMMENDATION" in summary


# ============================================================================
# Integration Tests
# ============================================================================

class TestWeek5Integration:
    """Integration tests for Week 5 items."""
    
    def test_policy_preset_with_pilot_override(self):
        """Test policy presets work with pilot override system."""
        from ml.policy_presets import load_policy_with_preset, PolicyPreset
        from ml.pilot_override_manager import PilotOverrideManager
        
        # Set environment for testing
        with patch.dict(os.environ, {
            "DEPLOYMENT_ENV": "development",
            "POLICY_PRESET": "conservative",
        }, clear=False):
            policy = load_policy_with_preset()
            
            assert policy._applied_preset == "conservative"
    
    def test_registry_validation_in_ci_context(self, tmp_path):
        """Test registry validation as would be used in CI."""
        from ml.registry_validator import validate_registry_state
        
        # Create mock registry
        registry = MagicMock()
        registry.state = {
            "current_version": "v1.0.0",
            "versions": {
                "v1.0.0": {"promoted": True, "aliases": ["latest"]},
            },
            "aliases": {"latest": "v1.0.0"},
        }
        
        result = validate_registry_state(registry)
        
        assert result["valid"] is True
        assert result["checks_run"] > 0
        
        # In CI, would exit with error code if not valid
        assert "timestamp" in result
        assert "duration_ms" in result


# ============================================================================
# Test Suite Runner
# ============================================================================

def run_all_tests():
    """Run all Week 5 tests."""
    import subprocess
    
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode


if __name__ == "__main__":
    import sys
    sys.exit(run_all_tests())
