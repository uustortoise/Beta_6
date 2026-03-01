"""
PR-3: Correctness Fixes Tests

Tests for:
1. Strict sequence-label alignment
2. Duplicate resolution policy
3. Reproducibility no-op hash
4. Pilot override CI-safe mode
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.sequence_alignment import (
    create_labeled_sequences_strict,
    assert_sequence_label_alignment,
    safe_create_sequences,
    SequenceLabelAlignmentError,
    SequenceAlignmentValidator,
    validate_stride_safety,
)
from ml.duplicate_resolution import (
    DuplicateTimestampResolver,
    DuplicateResolutionPolicy,
    TieBreaker,
    resolve_duplicate_timestamps,
)
from ml.reproducibility_report import (
    ReproducibilityTracker,
    DataFingerprint,
    CodeVersion,
    RunOutcome,
    compute_data_fingerprint,
    get_code_version,
)
from ml.pilot_override_manager import (
    PilotOverrideManager,
    is_ci_environment,
    set_training_profile,
)


class TestStrictSequenceLabelAlignment(unittest.TestCase):
    """PR-3.1: Strict sequence-label alignment"""
    
    def test_create_labeled_sequences_strict_basic(self):
        """Test basic sequence creation with strict alignment"""
        n_samples = 100
        n_features = 5
        seq_length = 10
        
        sensor_data = np.random.randn(n_samples, n_features)
        labels = np.random.randint(0, 3, n_samples)
        timestamps = pd.date_range("2026-01-01", periods=n_samples, freq="10s")
        
        X_seq, y_seq, seq_ts = create_labeled_sequences_strict(
            sensor_data, labels, seq_length, stride=1, timestamps=timestamps
        )
        
        # Verify shapes
        expected_n_seq = n_samples - seq_length + 1
        self.assertEqual(X_seq.shape, (expected_n_seq, seq_length, n_features))
        self.assertEqual(y_seq.shape, (expected_n_seq,))
        self.assertEqual(seq_ts.shape, (expected_n_seq,))
        
        # Verify alignment: y_seq[i] should be label at end of X_seq[i]
        for i in range(expected_n_seq):
            self.assertEqual(y_seq[i], labels[i + seq_length - 1])
    
    def test_create_labeled_sequences_strict_mismatch_error(self):
        """Test error on sample count mismatch"""
        sensor_data = np.random.randn(100, 5)
        labels = np.random.randint(0, 3, 90)  # Mismatched length
        
        with self.assertRaises(SequenceLabelAlignmentError):
            create_labeled_sequences_strict(sensor_data, labels, seq_length=10)
    
    def test_create_labeled_sequences_strict_insufficient_data(self):
        """Test error on insufficient data"""
        sensor_data = np.random.randn(5, 5)
        labels = np.random.randint(0, 3, 5)
        
        with self.assertRaises(SequenceLabelAlignmentError):
            create_labeled_sequences_strict(sensor_data, labels, seq_length=10)
    
    def test_assert_sequence_label_alignment_pass(self):
        """Test alignment assertion passes for valid data"""
        X_seq = np.random.randn(50, 10, 5)
        y_seq = np.random.randint(0, 3, 50)
        seq_ts = pd.date_range("2026-01-01", periods=50, freq="10s")
        
        # Should not raise
        assert_sequence_label_alignment(X_seq, y_seq, seq_ts, context="test")
    
    def test_assert_sequence_label_alignment_fail(self):
        """Test alignment assertion fails for mismatched data"""
        X_seq = np.random.randn(50, 10, 5)
        y_seq = np.random.randint(0, 3, 49)  # Mismatched
        seq_ts = pd.date_range("2026-01-01", periods=50, freq="10s")
        
        with self.assertRaises(SequenceLabelAlignmentError):
            assert_sequence_label_alignment(X_seq, y_seq, seq_ts, context="test")
    
    def test_validate_stride_safety(self):
        """Test stride safety validation"""
        # Valid config
        result = validate_stride_safety(100, seq_length=10, stride=1)
        self.assertTrue(result["safe"])
        self.assertEqual(result["expected_sequences"], 91)
        
        # Invalid stride
        result = validate_stride_safety(100, seq_length=10, stride=0)
        self.assertFalse(result["safe"])
        
        # Insufficient data
        result = validate_stride_safety(5, seq_length=10, stride=1)
        self.assertFalse(result["safe"])


class TestDuplicateResolutionPolicy(unittest.TestCase):
    """PR-3.2: Duplicate resolution policy"""
    
    def test_duplicate_resolution_majority_vote(self):
        """Test majority vote resolution"""
        df = pd.DataFrame({
            'timestamp': ['2026-01-01 10:00:00', '2026-01-01 10:00:00', '2026-01-01 10:00:01'],
            'activity': ['cooking', 'cooking', 'sleeping'],
            'sensor1': [1.0, 1.1, 2.0],
        })
        
        policy = DuplicateResolutionPolicy(method="majority_vote", tie_breaker=TieBreaker.LATEST)
        resolver = DuplicateTimestampResolver(policy)
        
        result = resolver.resolve(df, timestamp_col='timestamp', label_col='activity')
        
        # Should resolve to 2 rows
        self.assertEqual(len(result), 2)
        # The duplicate timestamp should resolve to 'cooking' (majority)
        row_at_duplicate = result[result['timestamp'] == '2026-01-01 10:00:00']
        self.assertEqual(len(row_at_duplicate), 1)
    
    def test_duplicate_resolution_no_duplicates(self):
        """Test fast path when no duplicates exist"""
        df = pd.DataFrame({
            'timestamp': ['2026-01-01 10:00:00', '2026-01-01 10:00:01', '2026-01-01 10:00:02'],
            'activity': ['cooking', 'sleeping', 'eating'],
            'sensor1': [1.0, 2.0, 3.0],
        })
        
        policy = DuplicateResolutionPolicy(method="majority_vote")
        resolver = DuplicateTimestampResolver(policy)
        
        result = resolver.resolve(df, timestamp_col='timestamp', label_col='activity')
        
        # Should return same number of rows
        self.assertEqual(len(result), 3)
    
    def test_duplicate_resolution_stats(self):
        """Test duplicate resolution statistics"""
        df = pd.DataFrame({
            'timestamp': ['2026-01-01 10:00:00'] * 3 + ['2026-01-01 10:00:01'],
            'activity': ['cooking', 'cooking', 'sleeping', 'eating'],
            'sensor1': [1.0, 1.1, 2.0, 3.0],
        })
        
        policy = DuplicateResolutionPolicy(method="majority_vote", emit_stats=True)
        resolver = DuplicateTimestampResolver(policy)
        
        resolver.resolve(df, timestamp_col='timestamp', label_col='activity')
        stats = resolver.get_stats()
        
        self.assertEqual(stats["total_timestamps"], 4)
        self.assertEqual(stats["duplicate_count"], 3)
        self.assertEqual(stats["resolution_method"], "majority_vote")


class TestReproducibilityNoOp(unittest.TestCase):
    """PR-3.3: Reproducibility no-op hash"""
    
    def test_data_fingerprint_compute_hash(self):
        """Test data fingerprint hash computation"""
        fp = DataFingerprint(
            elder_id="test_elder",
            room_names=("bedroom", "kitchen"),
            total_samples=1000,
            observed_days=7,
            raw_data_hash="abc123",
        )
        
        hash1 = fp.compute_hash()
        hash2 = fp.compute_hash()
        
        # Same inputs should produce same hash
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 16)  # Truncated to 16 chars
    
    def test_reproducibility_tracker_noop_eligibility_no_prior(self):
        """Test no-op eligibility when no prior run exists"""
        tracker = ReproducibilityTracker()
        
        # Create a data fingerprint
        fp = DataFingerprint(
            elder_id="test_elder",
            room_names=("bedroom",),
            total_samples=100,
            observed_days=3,
            raw_data_hash="hash1",
        )
        
        # Create a clean code version for testing
        clean_code_version = CodeVersion(
            git_commit="abc123",
            git_branch="main",
            git_dirty=False,
            python_version="3.12.0",
        )
        
        # First check - no prior run
        is_noop, reason, prior = tracker.check_noop_eligibility(
            fp, "policy_hash_1", code_version=clean_code_version
        )
        self.assertFalse(is_noop)
        self.assertIn("No equivalent prior run", reason)
        self.assertIsNone(prior)
    
    def test_reproducibility_tracker_create_report(self):
        """Test creating reproducibility report"""
        tracker = ReproducibilityTracker()
        
        fp = DataFingerprint(
            elder_id="test_elder",
            room_names=("bedroom",),
            total_samples=100,
            observed_days=3,
            raw_data_hash="hash1",
        )
        
        outcome = RunOutcome(promoted_rooms=["bedroom"], rejected_rooms=[])
        report = tracker.create_report(
            run_id="run_1",
            elder_id="test_elder",
            data_fingerprint=fp,
            policy_hash="policy_hash_1",
            random_seed=42,
            outcome=outcome,
        )
        
        # Verify report structure
        self.assertEqual(report.run_id, "run_1")
        self.assertEqual(report.elder_id, "test_elder")
        self.assertEqual(report.policy_hash, "policy_hash_1")
        self.assertEqual(report.outcome.promoted_rooms, ["bedroom"])
        
        # Verify composite hash is computed
        composite_hash = report.compute_composite_hash()
        self.assertEqual(len(composite_hash), 32)
        
        # Verify outcome signature
        sig = report.outcome.compute_signature()
        self.assertEqual(len(sig), 16)
    
    def test_run_outcome_signature_determinism(self):
        """Test that outcome signatures are deterministic"""
        outcome1 = RunOutcome(
            promoted_rooms=["kitchen", "bedroom"],
            rejected_rooms=["bathroom"],
        )
        outcome2 = RunOutcome(
            promoted_rooms=["bedroom", "kitchen"],
            rejected_rooms=["bathroom"],
        )
        
        # Same rooms in different order should produce same signature
        self.assertEqual(outcome1.compute_signature(), outcome2.compute_signature())


class TestPilotOverrideCISafe(unittest.TestCase):
    """PR-3.4: Pilot override CI-safe mode"""
    
    def setUp(self):
        """Save original TRAINING_PROFILE env var"""
        self._original_profile = os.environ.get('TRAINING_PROFILE')
    
    def tearDown(self):
        """Restore TRAINING_PROFILE env var and clean up state files"""
        if self._original_profile is not None:
            os.environ['TRAINING_PROFILE'] = self._original_profile
        elif 'TRAINING_PROFILE' in os.environ:
            del os.environ['TRAINING_PROFILE']
        
        # Clean up any test state files
        for state_file in ["/tmp/test_pilot_state.json", "/tmp/test_pilot_force_state.json"]:
            if os.path.exists(state_file):
                os.remove(state_file)
    
    def test_is_ci_environment_detection(self):
        """Test CI environment detection"""
        # Clear any CI env vars
        ci_vars = ['CI', 'GITHUB_ACTIONS', 'GITLAB_CI', 'CIRCLECI', 'JENKINS_URL', 
                   'BUILDKITE', 'TF_BUILD', 'DRONE', 'TRAVIS', 'CODEBUILD_BUILD_ID']
        
        # Save original values
        original_values = {var: os.environ.get(var) for var in ci_vars}
        
        try:
            # Clear all CI vars
            for var in ci_vars:
                if var in os.environ:
                    del os.environ[var]
            
            self.assertFalse(is_ci_environment())
            
            # Test each CI var
            for var in ci_vars:
                os.environ[var] = "true"
                self.assertTrue(is_ci_environment(), f"Failed to detect {var}")
                del os.environ[var]
        finally:
            # Restore original values
            for var, val in original_values.items():
                if val is not None:
                    os.environ[var] = val
                elif var in os.environ:
                    del os.environ[var]
    
    def test_pilot_activation_blocked_in_ci(self):
        """Test pilot activation is blocked in CI environment"""
        manager = PilotOverrideManager(state_file="/tmp/test_pilot_state.json")
        
        with patch('ml.pilot_override_manager.is_ci_environment', return_value=True):
            success, message = manager.activate_pilot("Testing", duration_hours=1)
            
            self.assertFalse(success)
            self.assertIn("blocked in CI environment", message)
    
    def test_pilot_activation_allowed_with_ci_safe_false(self):
        """Test pilot activation can be forced with ci_safe=False"""
        manager = PilotOverrideManager(state_file="/tmp/test_pilot_force_state.json")
        
        with patch('ml.pilot_override_manager.is_ci_environment', return_value=True):
            success, message = manager.activate_pilot(
                "Testing", duration_hours=1, ci_safe=False
            )
            
            # Should succeed when ci_safe=False
            self.assertTrue(success)
            self.assertIn("Pilot profile activated", message)
    
    def test_pilot_activation_allowed_outside_ci(self):
        """Test pilot activation works outside CI"""
        manager = PilotOverrideManager(state_file="/tmp/test_pilot_state.json")
        
        with patch('ml.pilot_override_manager.is_ci_environment', return_value=False):
            success, message = manager.activate_pilot("Testing", duration_hours=1)
            
            self.assertTrue(success)
            self.assertIn("Pilot profile activated", message)


class TestIntegrationPR3Features(unittest.TestCase):
    """Integration tests for PR-3 features"""
    
    def test_sequence_creation_with_timestamps(self):
        """End-to-end sequence creation with timestamp tracking"""
        n_samples = 50
        seq_length = 10
        
        sensor_data = np.random.randn(n_samples, 5)
        labels = np.random.randint(0, 3, n_samples)
        timestamps = pd.date_range("2026-01-01", periods=n_samples, freq="10s")
        
        X_seq, y_seq, seq_ts = create_labeled_sequences_strict(
            sensor_data, labels, seq_length, timestamps=timestamps
        )
        
        # Verify each sequence's timestamp matches its label's timestamp
        for i in range(len(X_seq)):
            label_idx = i + seq_length - 1
            self.assertEqual(seq_ts[i], timestamps[label_idx])
            self.assertEqual(y_seq[i], labels[label_idx])
    
    def test_duplicate_resolution_then_sequence_creation(self):
        """Test duplicate resolution followed by sequence creation"""
        # Create data with duplicates
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2026-01-01 10:00:00'] * 2 + 
                                        ['2026-01-01 10:00:10'] * 2 +
                                        ['2026-01-01 10:00:20']),
            'activity': ['cooking', 'cooking', 'sleeping', 'sleeping', 'eating'],
            'sensor1': [1.0, 1.1, 2.0, 2.1, 3.0],
            'sensor2': [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        
        # Apply duplicate resolution
        policy = DuplicateResolutionPolicy(method="majority_vote")
        resolver = DuplicateTimestampResolver(policy)
        resolved_df = resolver.resolve(df, timestamp_col='timestamp', label_col='activity')
        
        # Should have 3 unique timestamps
        self.assertEqual(len(resolved_df), 3)
        
        # Now create sequences (would need activity encoding in real scenario)
        # Just verify structure is preserved
        self.assertIn('sensor1', resolved_df.columns)
        self.assertIn('sensor2', resolved_df.columns)


if __name__ == '__main__':
    unittest.main()
