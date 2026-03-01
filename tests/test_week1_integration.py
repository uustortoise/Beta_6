"""
Week 1 Integration Tests

Tests for:
- Item 1 Phase A: preprocess_without_scaling() + apply_scaling()
- Item 2: CoverageContractGate
- Item 3: TRAINING_PROFILE support
- Item 7: Walk-Forward Robustness
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from elderlycare_v1_16.platform import ElderlyCarePlatform
from ml.coverage_contract import CoverageContractGate, WalkForwardConfig
from ml.policy_config import load_policy_from_env, TrainingPolicy
from ml.week1_integration import (
    check_coverage_contract_for_room,
    prepare_training_data_with_leakage_free_scaling,
    apply_train_split_scaling,
    check_walk_forward_status,
    policy_hash,
)


class TestItem1_LeakageFreeScaling:
    """Item 1: Test preprocess_without_scaling() + apply_scaling() flow."""
    
    def test_preprocess_without_scaling_returns_unscaled_data(self):
        """Phase A: Preprocessing should NOT scale sensor data."""
        # Use custom sensor columns that match test data
        platform = ElderlyCarePlatform(
            enable_time_based_processing=False,
            sensor_columns=['sound', 'light', 'motion']  # Match test data
        )
        
        # Create test data with known values
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'sound': [50.0] * 100,
            'light': [500.0] * 100,
            'motion': [0.5] * 100,
            'activity': ['inactive'] * 100,
        })
        
        result = platform.preprocess_without_scaling(df, 'bedroom', is_training=True)
        
        # Sensor values should be unscaled (raw values preserved)
        assert result['sound'].mean() == pytest.approx(50.0, rel=0.01)
        assert result['light'].mean() == pytest.approx(500.0, rel=0.01)
        
        # Should have activity column (not encoded yet)
        assert 'activity' in result.columns
        assert 'activity_encoded' not in result.columns
    
    def test_apply_scaling_fits_on_train_only(self):
        """Phase C: Scaler should be fitted only on training data."""
        platform = ElderlyCarePlatform(
            enable_time_based_processing=False,
            sensor_columns=['sound', 'light']  # Match test data
        )
        
        # Create train data with values 0-100
        train_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'sound': np.linspace(0, 100, 100),
            'light': np.linspace(0, 1000, 100),
            'activity': ['inactive'] * 100,
        })
        
        # Create validation data with different range (50-150)
        val_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-02', periods=100, freq='1min'),
            'sound': np.linspace(50, 150, 100),
            'light': np.linspace(500, 1500, 100),
        })
        
        # Fit on train
        scaler_meta = {'fit_start_ts': '2024-01-01', 'fit_end_ts': '2024-01-01', 'fit_sample_count': 100}
        train_scaled = platform.apply_scaling(train_df, 'bedroom', is_training=True, scaler_fit_range=scaler_meta)
        
        # Transform val using the same scaler
        val_scaled = platform.apply_scaling(val_df, 'bedroom', is_training=False)
        
        # Train data should be centered around 0
        assert abs(train_scaled['sound'].mean()) < 0.1
        assert abs(train_scaled['light'].mean()) < 0.1
        
        # Val data should NOT be centered (since fitted on train range 0-100, val has 50-150)
        assert val_scaled['sound'].mean() > 0  # Should be positive since val > train
    
    def test_full_train_val_split_flow(self):
        """Test complete flow: preprocess → temporal split → scale on train only."""
        platform = ElderlyCarePlatform(
            enable_time_based_processing=False,
            sensor_columns=['sound', 'light']  # Match test data
        )
        
        # Create data spanning 5 days
        timestamps = pd.date_range('2024-01-01', periods=7200, freq='1min')  # 5 days
        df = pd.DataFrame({
            'timestamp': timestamps,
            'sound': np.random.normal(50, 10, len(timestamps)),
            'light': np.random.normal(500, 100, len(timestamps)),
            'activity': ['inactive'] * len(timestamps),
        })
        
        # Preprocess without scaling
        preprocessed = platform.preprocess_without_scaling(df, 'bedroom', is_training=True)
        
        # Temporal split (day 1-3 = train, day 4-5 = val)
        split_ts = pd.Timestamp('2024-01-04')
        train_df = preprocessed[preprocessed['timestamp'] < split_ts].copy()
        val_df = preprocessed[preprocessed['timestamp'] >= split_ts].copy()
        
        # Scale on train only
        result = apply_train_split_scaling(platform, 'bedroom', train_df, val_df)
        
        assert result['train_scaled'] is not None
        assert result['val_scaled'] is not None
        assert result['scaler_metadata']['fit_sample_count'] == len(train_df)
        assert result['scaler_metadata']['fit_start_ts'] == train_df['timestamp'].min().isoformat()


class TestItem2_CoverageContractGate:
    """Item 2: Test coverage contract for minimum fold feasibility."""
    
    def test_coverage_contract_passes_with_sufficient_days(self):
        """Should pass when sufficient observed days exist."""
        policy = TrainingPolicy()
        policy.data_viability.min_observed_days = 3
        
        # Create data with 5 days
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5*24*60, freq='1min'),
        })
        
        passes, details = check_coverage_contract_for_room('bedroom', df, policy)
        
        assert passes is True
        assert details['observed_days'] == 5
        assert details['required_days'] >= 3
    
    def test_coverage_contract_fails_with_insufficient_days(self):
        """Should fail when not enough observed days."""
        policy = TrainingPolicy()
        policy.data_viability.min_observed_days = 3
        
        # Create data with only 1 day
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24*60, freq='1min'),
        })
        
        passes, details = check_coverage_contract_for_room('bedroom', df, policy)
        
        assert passes is False
        assert 'reason' in details
        assert details['observed_days'] == 1
    
    def test_coverage_contract_fails_without_timestamp(self):
        """Should fail gracefully when timestamp column is missing."""
        policy = TrainingPolicy()
        df = pd.DataFrame({'value': [1, 2, 3]})
        
        passes, details = check_coverage_contract_for_room('bedroom', df, policy)
        
        assert passes is False
        assert details['reason'] == 'missing_timestamp_column'


class TestItem3_TrainingProfile:
    """Item 3: Test TRAINING_PROFILE support."""
    
    def test_pilot_profile_loads_with_lower_thresholds(self):
        """Pilot profile should have relaxed thresholds."""
        os.environ['TRAINING_PROFILE'] = 'pilot'
        
        policy = load_policy_from_env()
        
        assert policy.get_profile_name() == 'pilot'
        # Pilot mode sets min_observed_days to min(default, 3) = 3
        assert policy.data_viability.min_observed_days == 3
        
        # Cleanup
        del os.environ['TRAINING_PROFILE']
    
    def test_production_profile_loads_with_stricter_thresholds(self):
        """Production profile should have stricter thresholds."""
        os.environ['TRAINING_PROFILE'] = 'production'
        
        policy = load_policy_from_env()
        
        assert policy.get_profile_name() == 'production'
        assert policy.data_viability.min_observed_days >= 2
        
        # Cleanup
        del os.environ['TRAINING_PROFILE']
    
    def test_default_profile_is_production(self):
        """Default profile should be production."""
        if 'TRAINING_PROFILE' in os.environ:
            del os.environ['TRAINING_PROFILE']
        
        policy = load_policy_from_env()
        
        assert policy.get_profile_name() == 'production'
    
    def test_policy_hash_is_stable(self):
        """Policy hash should be deterministic."""
        policy = TrainingPolicy()
        
        hash1 = policy_hash(policy)
        hash2 = policy_hash(policy)
        
        assert hash1 == hash2
        assert len(hash1) == 16  # First 16 chars of sha256


class TestItem7_WalkForwardRobustness:
    """Item 7: Test walk-forward evaluation handles insufficient data gracefully."""
    
    def test_walk_forward_unavailable_status(self):
        """Should return walk_forward_unavailable when no folds possible."""
        eval_result = {
            "status": "walk_forward_unavailable",
            "folds": [],
            "summary": {"num_folds": 0, "reason": "insufficient_observed_days"}
        }
        
        status = check_walk_forward_status(eval_result)
        
        assert status['walk_forward_available'] is False
        assert status['status'] == 'walk_forward_unavailable'
        assert status['promotable'] is False
    
    def test_walk_forward_completed_status(self):
        """Should return completed status when folds exist."""
        eval_result = {
            "status": "completed",
            "folds": [{'fold': 0}, {'fold': 1}],
            "summary": {"num_folds": 2}
        }
        
        status = check_walk_forward_status(eval_result)
        
        assert status['walk_forward_available'] is True
        assert status['status'] == 'completed'
        assert status['num_folds'] == 2
        assert status['promotable'] is True


class TestWeek1Integration:
    """End-to-end integration tests for Week 1 items."""
    
    def test_full_pipeline_with_coverage_contract_pass(self):
        """Full pipeline: coverage check → preprocessing → status."""
        platform = ElderlyCarePlatform(
            enable_time_based_processing=False,
            sensor_columns=['sound', 'light', 'motion']  # Match test data
        )
        
        # Create sufficient data (8 days - production requires 7+1 for walk-forward)
        timestamps = pd.date_range('2024-01-01', periods=8*24*60, freq='1min')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'sound': np.random.normal(50, 10, len(timestamps)),
            'light': np.random.normal(500, 100, len(timestamps)),
            'motion': np.random.normal(0.5, 0.1, len(timestamps)),
            'activity': ['inactive'] * len(timestamps),
        })
        
        result = prepare_training_data_with_leakage_free_scaling(
            platform=platform,
            room_name='bedroom',
            df=df,
            elder_id='test_resident_001',
            training_profile='production',
        )
        
        assert result['status'] == 'preprocessing_complete'
        assert result['coverage_details']['passes'] is True
        assert result['preprocessed_df'] is not None
        assert result['policy_profile'] == 'production'
        assert 'policy_hash' in result
    
    def test_full_pipeline_with_coverage_contract_fail(self):
        """Full pipeline: coverage check should fail with insufficient data."""
        platform = ElderlyCarePlatform(
            enable_time_based_processing=False,
            sensor_columns=['sound', 'light']  # Match test data
        )
        
        # Create insufficient data (1 day only)
        timestamps = pd.date_range('2024-01-01', periods=24*60, freq='1min')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'sound': np.random.normal(50, 10, len(timestamps)),
            'light': np.random.normal(500, 100, len(timestamps)),
            'activity': ['inactive'] * len(timestamps),
        })
        
        result = prepare_training_data_with_leakage_free_scaling(
            platform=platform,
            room_name='bedroom',
            df=df,
            elder_id='test_resident_002',
            training_profile='production',
        )
        
        assert result['status'] == 'coverage_contract_failed'
        assert result['coverage_details']['passes'] is False
        assert result['preprocessed_df'] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
