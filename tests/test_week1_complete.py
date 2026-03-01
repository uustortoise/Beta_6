"""
Complete Week 1 Integration Tests

Tests all Week 1 items:
- Item 1 Phase A+B+C+D: Train-split scaling with feature flag
- Item 2: Coverage Contract Gate
- Item 3: Training Profile (pilot vs production)
- Item 6: Statistical Validity Gate
- Item 7: Walk-Forward Robustness
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from elderlycare_v1_16.platform import ElderlyCarePlatform
from ml.train_split_scaling_pipeline import (
    is_train_split_scaling_enabled,
    temporal_split_dataframe,
    prepare_training_data_with_train_split_scaling,
    validate_no_leakage,
)
from ml.statistical_validity_gate import (
    StatisticalValidityGate,
    create_statistical_validity_gate_from_policy,
    evaluate_promotion_with_statistical_validity,
)
from ml.policy_config import load_policy_from_env, TrainingPolicy
from ml.coverage_contract import CoverageContractGate, WalkForwardConfig


class TestItem1_TrainSplitScaling:
    """Item 1: Complete train-split scaling implementation."""
    
    def test_feature_flag_disabled_by_default(self):
        """Phase D: Feature flag should be disabled by default."""
        if 'ENABLE_TRAIN_SPLIT_SCALING' in os.environ:
            del os.environ['ENABLE_TRAIN_SPLIT_SCALING']
        
        assert is_train_split_scaling_enabled() is False
    
    def test_feature_flag_enabled_via_env(self):
        """Phase D: Feature flag can be enabled via env var."""
        os.environ['ENABLE_TRAIN_SPLIT_SCALING'] = 'true'
        assert is_train_split_scaling_enabled() is True
        
        os.environ['ENABLE_TRAIN_SPLIT_SCALING'] = '1'
        assert is_train_split_scaling_enabled() is True
        
        os.environ['ENABLE_TRAIN_SPLIT_SCALING'] = 'false'
        assert is_train_split_scaling_enabled() is False
        
        del os.environ['ENABLE_TRAIN_SPLIT_SCALING']
    
    def test_temporal_split_chronological_order(self):
        """Phase B: Temporal split should maintain chronological order."""
        # Create data with timestamps
        timestamps = pd.date_range('2024-01-01', periods=1000, freq='1min')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': np.arange(1000),
        })
        
        train_df, val_df, calib_df, metadata = temporal_split_dataframe(
            df, validation_split=0.2, calibration_fraction=0.3
        )
        
        # Train should be before validation
        assert train_df['timestamp'].max() < val_df['timestamp'].min()
        
        # Validation should be before calibration (if exists)
        if calib_df is not None:
            assert val_df['timestamp'].max() < calib_df['timestamp'].min()
    
    def test_temporal_split_metadata(self):
        """Phase B: Temporal split should provide detailed metadata."""
        timestamps = pd.date_range('2024-01-01', periods=1000, freq='1min')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': np.arange(1000),
        })
        
        train_df, val_df, calib_df, metadata = temporal_split_dataframe(
            df, validation_split=0.2, calibration_fraction=0.3
        )
        
        assert 'total_samples' in metadata
        assert 'train_samples' in metadata
        assert 'val_samples' in metadata
        assert 'calib_samples' in metadata
        assert 'train_start_ts' in metadata
        assert 'train_end_ts' in metadata
        assert metadata['total_samples'] == 1000
    
    def test_validate_no_leakage_passes(self):
        """Phase B: Leakage validation should pass for proper splits."""
        timestamps = pd.date_range('2024-01-01', periods=300, freq='1min')
        
        train_df = pd.DataFrame({
            'timestamp': timestamps[:200],
            'value': np.arange(200),
        })
        
        val_df = pd.DataFrame({
            'timestamp': timestamps[200:250],
            'value': np.arange(200, 250),
        })
        
        calib_df = pd.DataFrame({
            'timestamp': timestamps[250:],
            'value': np.arange(250, 300),
        })
        
        passes, violations = validate_no_leakage(train_df, val_df, calib_df)
        
        assert passes is True
        assert len(violations) == 0
    
    def test_validate_no_leakage_fails_on_overlap(self):
        """Phase B: Leakage validation should detect overlap."""
        timestamps = pd.date_range('2024-01-01', periods=100, freq='1min')
        
        # Create overlapping data
        train_df = pd.DataFrame({
            'timestamp': timestamps[:60],
            'value': np.arange(60),
        })
        
        val_df = pd.DataFrame({
            'timestamp': timestamps[50:],  # Overlaps with train
            'value': np.arange(50, 100),
        })
        
        passes, violations = validate_no_leakage(train_df, val_df)
        
        assert passes is False
        assert len(violations) > 0
    
    def test_full_leakage_free_pipeline(self):
        """Phase A+B+C: Complete pipeline from raw data to scaled splits."""
        platform = ElderlyCarePlatform(
            enable_time_based_processing=False,
            sensor_columns=['sound', 'light']
        )
        
        # Create 3 days of data
        timestamps = pd.date_range('2024-01-01', periods=3*24*60, freq='1min')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'sound': np.random.normal(50, 10, len(timestamps)),
            'light': np.random.normal(500, 100, len(timestamps)),
            'activity': ['inactive'] * len(timestamps),
        })
        
        result = prepare_training_data_with_train_split_scaling(
            platform=platform,
            room_name='bedroom',
            raw_df=df,
            validation_split=0.2,
        )
        
        assert result['status'] == 'success'
        assert result['train_scaled'] is not None
        assert result['val_scaled'] is not None
        assert 'split_metadata' in result
        assert 'scaler_metadata' in result
        
        # Verify scaler was fitted on train only
        scaler_meta = result['scaler_metadata']
        assert 'fit_start_ts' in scaler_meta
        assert 'fit_end_ts' in scaler_meta
        assert 'fit_sample_count' in scaler_meta
        
        # Verify no leakage
        passes, violations = validate_no_leakage(
            result['train_scaled'],
            result['val_scaled'],
            result.get('calib_scaled')
        )
        assert passes is True


class TestItem6_StatisticalValidityGate:
    """Item 6: Statistical Validity Gate Tightening."""
    
    def test_gate_passes_with_sufficient_support(self):
        """Gate should pass when all classes have sufficient support."""
        gate = StatisticalValidityGate(
            min_calibration_support=50,
            min_minority_support=10,
            min_promotable_class_count=2,
        )
        
        # Create calibration data with 3 classes, each with 20 samples
        y_calib = np.array([0] * 20 + [1] * 20 + [2] * 20)
        
        result = gate.evaluate(y_calib, room_name='bedroom')
        
        assert result['passes'] is True
        assert result['promotable'] is True
        assert len(result['reasons']) == 0
        assert result['metrics']['minority_support'] == 20
    
    def test_gate_fails_with_insufficient_minority_support(self):
        """Gate should fail when minority class has too few samples."""
        gate = StatisticalValidityGate(
            min_calibration_support=50,
            min_minority_support=10,
            min_promotable_class_count=2,
        )
        
        # Create imbalanced data
        y_calib = np.array([0] * 100 + [1] * 5)  # Class 1 has only 5 samples
        
        result = gate.evaluate(y_calib, room_name='bedroom')
        
        assert result['passes'] is False
        assert result['promotable'] is False
        assert len(result['reasons']) > 0
        assert 'minority' in ' '.join(result['reasons']).lower()
    
    def test_gate_fails_with_insufficient_class_count(self):
        """Gate should fail when not enough classes have sufficient support."""
        gate = StatisticalValidityGate(
            min_calibration_support=50,
            min_minority_support=10,
            min_promotable_class_count=3,  # Require 3 classes
        )
        
        # Only 2 classes with sufficient support
        y_calib = np.array([0] * 50 + [1] * 50)
        
        result = gate.evaluate(y_calib, room_name='bedroom')
        
        assert result['passes'] is False
        assert '2' in str(result['metrics']['classes_with_sufficient_support'])
    
    def test_gate_fails_with_insufficient_total_support(self):
        """Gate should fail when total calibration samples are insufficient."""
        gate = StatisticalValidityGate(
            min_calibration_support=100,
            min_minority_support=5,
        )
        
        y_calib = np.array([0] * 30 + [1] * 30)  # Only 60 samples
        
        result = gate.evaluate(y_calib, room_name='bedroom')
        
        assert result['passes'] is False
        assert '60' in ' '.join(result['reasons'])
        assert '100' in ' '.join(result['reasons'])
    
    def test_gate_from_policy(self):
        """Gate should be configurable from TrainingPolicy."""
        policy = TrainingPolicy()
        policy.calibration.min_samples = 75
        policy.calibration.min_support_per_class = 15
        
        gate = create_statistical_validity_gate_from_policy(policy)
        
        assert gate.min_calibration_support >= 75
        assert gate.min_minority_support >= 15
    
    def test_evaluate_promotion_with_fallback(self):
        """Promotion should be blocked when using fallback metrics."""
        y_calib = np.array([0] * 50 + [1] * 50)
        
        calib_metrics = {
            'is_fallback': True,
            'metric_source': 'fallback',
        }
        
        promotable, reasons, result = evaluate_promotion_with_statistical_validity(
            calibration_metrics=calib_metrics,
            y_calib=y_calib,
            room_name='bedroom',
        )
        
        assert promotable is False
        assert len(reasons) > 0
        assert 'fallback' in ' '.join(reasons).lower()


class TestItem2_CoverageContractGate:
    """Item 2: Data Coverage Contract Gate."""
    
    def test_coverage_contract_passes(self):
        """Coverage contract should pass with sufficient observed days."""
        config = WalkForwardConfig(min_train_days=5, valid_days=1, step_days=1)
        gate = CoverageContractGate(config)
        
        result = gate.evaluate('bedroom', observed_days=7)
        
        assert result.passes is True
        assert result.observed_days == 7
    
    def test_coverage_contract_fails(self):
        """Coverage contract should fail with insufficient observed days."""
        config = WalkForwardConfig(min_train_days=5, valid_days=1, step_days=1)
        gate = CoverageContractGate(config)
        
        result = gate.evaluate('bedroom', observed_days=3)
        
        assert result.passes is False
        assert result.observed_days == 3
        assert '5' in result.reason  # Required days mentioned


class TestItem3_TrainingProfile:
    """Item 3: Training Profile (Pilot vs Production)."""
    
    def test_pilot_profile_relaxed_thresholds(self):
        """Pilot profile should have relaxed thresholds."""
        os.environ['TRAINING_PROFILE'] = 'pilot'
        policy = load_policy_from_env()
        
        assert policy.get_profile_name() == 'pilot'
        # Pilot sets min_observed_days to min(default, 3)
        assert policy.data_viability.min_observed_days == 3
        
        del os.environ['TRAINING_PROFILE']
    
    def test_production_profile_strict_thresholds(self):
        """Production profile should have strict thresholds."""
        os.environ['TRAINING_PROFILE'] = 'production'
        policy = load_policy_from_env()
        
        assert policy.get_profile_name() == 'production'
        
        del os.environ['TRAINING_PROFILE']


class TestWeek1EndToEnd:
    """End-to-end integration tests for all Week 1 items."""
    
    def test_complete_pipeline_with_train_split_scaling(self):
        """Complete pipeline with all Week 1 features enabled."""
        # Enable feature flag
        os.environ['ENABLE_TRAIN_SPLIT_SCALING'] = 'true'
        os.environ['TRAINING_PROFILE'] = 'production'
        
        try:
            platform = ElderlyCarePlatform(
                enable_time_based_processing=False,
                sensor_columns=['sound', 'light', 'motion']
            )
            
            # Create 5 days of data
            timestamps = pd.date_range('2024-01-01', periods=5*24*60, freq='1min')
            df = pd.DataFrame({
                'timestamp': timestamps,
                'sound': np.random.normal(50, 10, len(timestamps)),
                'light': np.random.normal(500, 100, len(timestamps)),
                'motion': np.random.normal(0.5, 0.1, len(timestamps)),
                'activity': ['inactive'] * len(timestamps),
            })
            
            # Use prepare_training_data_with_train_split_scaling
            result = prepare_training_data_with_train_split_scaling(
                platform=platform,
                room_name='bedroom',
                raw_df=df,
            )
            
            assert result['status'] == 'success'
            
            # Validate no leakage
            passes, violations = validate_no_leakage(
                result['train_scaled'],
                result['val_scaled'],
                result.get('calib_scaled')
            )
            assert passes is True, f"Leakage detected: {violations}"
            
        finally:
            del os.environ['ENABLE_TRAIN_SPLIT_SCALING']
            if 'TRAINING_PROFILE' in os.environ:
                del os.environ['TRAINING_PROFILE']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
