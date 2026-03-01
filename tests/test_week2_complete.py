"""
Complete Week 2 Integration Tests

Tests all Week 2 items:
- Item 4: Post-Gap Retention Quality Gate
- Item 5: Sequence-Label Alignment Contract Hardening
- Item 8: Duplicate-Timestamp Label Aggregation Policy
- Item 15: Class Coverage Gate (Train/Val/Calibration)
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ml.post_gap_retention_gate import (
    PostGapRetentionGate,
    create_post_gap_retention_gate_from_policy,
)
from ml.sequence_alignment import (
    create_labeled_sequences_strict,
    assert_sequence_label_alignment,
    SequenceLabelAlignmentError,
    SequenceAlignmentValidator,
    validate_stride_safety,
    safe_create_sequences,
)
from ml.duplicate_resolution import (
    DuplicateTimestampResolver,
    DuplicateResolutionPolicy,
    TieBreaker,
    resolve_duplicate_timestamps,
)
from ml.class_coverage_gate import (
    ClassCoverageGate,
    create_class_coverage_gate_from_policy,
    check_class_coverage_across_splits,
)
from ml.policy_config import TrainingPolicy


class TestItem4_PostGapRetentionGate:
    """Item 4: Post-Gap Retention Quality Gate."""
    
    def test_gate_passes_with_continuous_data(self):
        """Gate should pass with continuous, well-retained data."""
        gate = PostGapRetentionGate(
            min_retained_ratio=0.5,
            max_contiguous_segments=10,
        )
        
        # Create continuous data
        timestamps = pd.date_range('2024-01-01', periods=1000, freq='1min')
        raw_df = pd.DataFrame({'timestamp': timestamps, 'value': range(1000)})
        post_gap_df = raw_df.copy()  # No gaps
        
        result = gate.evaluate(raw_df, post_gap_df, room_name='bedroom')
        
        assert result['passes'] is True
        assert result['promotable'] is True
        assert result['metrics']['retained_ratio'] == 1.0
        assert result['metrics']['contiguous_segment_count'] == 1
    
    def test_gate_fails_with_high_fragmentation(self):
        """Gate should fail when data is highly fragmented."""
        gate = PostGapRetentionGate(
            min_retained_ratio=0.5,
            max_contiguous_segments=5,  # Low threshold
        )
        
        # Create fragmented data (many small segments)
        timestamps = []
        for i in range(10):  # 10 segments
            segment_start = datetime(2024, 1, 1) + timedelta(hours=i * 2)
            segment_ts = pd.date_range(segment_start, periods=10, freq='1min')
            timestamps.extend(segment_ts)
        
        raw_df = pd.DataFrame({'timestamp': timestamps, 'value': range(len(timestamps))})
        post_gap_df = raw_df.copy()
        
        result = gate.evaluate(raw_df, post_gap_df, room_name='bedroom')
        
        assert result['passes'] is False
        assert 'segment' in ' '.join(result['reasons']).lower()
    
    def test_gate_fails_with_low_retention(self):
        """Gate should fail when retention ratio is too low."""
        gate = PostGapRetentionGate(
            min_retained_ratio=0.8,
        )
        
        raw_df = pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min')})
        post_gap_df = raw_df.iloc[:50].copy()  # Only 50% retained
        
        result = gate.evaluate(raw_df, post_gap_df, room_name='bedroom')
        
        assert result['passes'] is False
        assert 'retention' in ' '.join(result['reasons']).lower()
        assert result['metrics']['retained_ratio'] == 0.5
    
    def test_continuity_analysis(self):
        """Continuity analysis should detect gaps correctly."""
        gate = PostGapRetentionGate()
        
        # Create data with a gap
        timestamps = list(pd.date_range('2024-01-01', periods=50, freq='1min'))
        timestamps += list(pd.date_range('2024-01-01 02:00', periods=50, freq='1min'))
        
        df = pd.DataFrame({'timestamp': timestamps, 'value': range(100)})
        
        continuity = gate.analyze_continuity(df)
        
        assert continuity['contiguous_segment_count'] == 2
        assert continuity['max_segment_length'] == 50
        assert continuity['median_segment_length'] == 50


class TestItem5_SequenceLabelAlignment:
    """Item 5: Sequence-Label Alignment Contract Hardening."""
    
    def test_strict_sequence_creation(self):
        """Strict sequence creation should maintain alignment."""
        sensor_data = np.random.randn(100, 5)
        labels = np.arange(100) % 3  # Classes 0, 1, 2
        seq_length = 10
        
        X_seq, y_seq, seq_ts = create_labeled_sequences_strict(
            sensor_data, labels, seq_length, stride=1
        )
        
        assert len(X_seq) == len(y_seq)
        assert len(X_seq) == len(seq_ts)
        assert X_seq.shape == (91, 10, 5)  # (100-10+1, 10, 5)
        
        # Verify labels are from end of sequences
        for i in range(len(X_seq)):
            expected_label = labels[i + seq_length - 1]
            assert y_seq[i] == expected_label
    
    def test_alignment_assertion_passes(self):
        """Alignment assertion should pass for valid data."""
        X_seq = np.random.randn(50, 10, 5)
        y_seq = np.arange(50) % 3
        seq_ts = np.arange(50)
        
        # Should not raise
        assert_sequence_label_alignment(X_seq, y_seq, seq_ts, context='test')
    
    def test_alignment_assertion_fails_on_mismatch(self):
        """Alignment assertion should fail on length mismatch."""
        X_seq = np.random.randn(50, 10, 5)
        y_seq = np.arange(40)  # Mismatched length
        seq_ts = np.arange(50)
        
        with pytest.raises(SequenceLabelAlignmentError):
            assert_sequence_label_alignment(X_seq, y_seq, seq_ts)
    
    def test_strict_creation_fails_on_insufficient_data(self):
        """Strict creation should fail when data is insufficient."""
        sensor_data = np.random.randn(5, 5)
        labels = np.arange(5)
        seq_length = 10
        
        with pytest.raises(SequenceLabelAlignmentError):
            create_labeled_sequences_strict(sensor_data, labels, seq_length)
    
    def test_stride_safety_validation(self):
        """Stride safety should detect misconfigurations."""
        # Valid configuration
        result = validate_stride_safety(100, 10, stride=1)
        assert result['safe'] is True
        assert result['expected_sequences'] == 91
        
        # Invalid stride
        result = validate_stride_safety(100, 10, stride=0)
        assert result['safe'] is False
        
        # Insufficient data
        result = validate_stride_safety(5, 10, stride=1)
        assert result['safe'] is False
    
    def test_alignment_validator_tracks_stages(self):
        """Validator should track validation across pipeline stages."""
        validator = SequenceAlignmentValidator('bedroom')
        
        X_seq = np.random.randn(50, 10, 5)
        y_seq = np.arange(50) % 3
        seq_ts = np.arange(50)
        
        # Validate at multiple stages
        validator.validate(X_seq, y_seq, seq_ts, "after_creation")
        validator.validate(X_seq, y_seq, seq_ts, "after_augmentation")
        
        report = validator.get_report()
        
        assert report['room'] == 'bedroom'
        assert report['all_passed'] is True
        assert len(report['validation_points']) == 2


class TestItem8_DuplicateResolution:
    """Item 8: Duplicate-Timestamp Label Aggregation Policy."""
    
    def test_no_duplicates_returns_unchanged(self):
        """Resolver should return unchanged DataFrame when no duplicates."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'activity': ['walking'] * 10,
        })
        
        policy = DuplicateResolutionPolicy(method="majority_vote")
        resolver = DuplicateTimestampResolver(policy)
        
        result = resolver.resolve(df)
        
        assert len(result) == 10
        assert resolver.get_stats()['duplicate_count'] == 0
    
    def test_majority_vote_resolution(self):
        """Majority vote should resolve duplicates correctly."""
        timestamps = [pd.Timestamp('2024-01-01 00:00')] * 3
        timestamps += [pd.Timestamp('2024-01-01 00:01')]  # No duplicate
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'activity': ['walking', 'walking', 'sleeping', 'eating'],
        })
        
        policy = DuplicateResolutionPolicy(method="majority_vote")
        resolver = DuplicateTimestampResolver(policy)
        
        result = resolver.resolve(df)
        
        assert len(result) == 2  # 2 unique timestamps
        # At 00:00, majority is 'walking' (2 vs 1)
        assert result[result['timestamp'] == '2024-01-01 00:00']['activity'].iloc[0] == 'walking'
    
    def test_tie_breaker_latest(self):
        """Latest tie-breaker should use last occurrence."""
        df = pd.DataFrame({
            'timestamp': ['2024-01-01 00:00'] * 2,
            'activity': ['walking', 'sleeping'],
        })
        
        policy = DuplicateResolutionPolicy(
            method="majority_vote",
            tie_breaker=TieBreaker.LATEST,
        )
        resolver = DuplicateTimestampResolver(policy)
        
        result = resolver.resolve(df)
        
        # Tie - should use latest (last row)
        assert result['activity'].iloc[0] == 'sleeping'
    
    def test_tie_breaker_priority(self):
        """Priority tie-breaker should use class priority map."""
        df = pd.DataFrame({
            'timestamp': ['2024-01-01 00:00'] * 2,
            'activity': ['walking', 'sleeping'],
        })
        
        policy = DuplicateResolutionPolicy(
            method="majority_vote",
            tie_breaker=TieBreaker.HIGHEST_PRIORITY,
            class_priority_map={'sleeping': 1, 'walking': 2},  # Lower number = higher priority
        )
        resolver = DuplicateTimestampResolver(policy)
        
        result = resolver.resolve(df)
        
        # sleeping has higher priority (1 < 2)
        assert result['activity'].iloc[0] == 'sleeping'
    
    def test_duplicate_stats_emitted(self):
        """Stats should be emitted when emit_stats=True."""
        df = pd.DataFrame({
            'timestamp': ['2024-01-01 00:00'] * 3 + ['2024-01-01 00:01'] * 2,
            'activity': ['a', 'a', 'b', 'c', 'c'],
        })
        
        policy = DuplicateResolutionPolicy(emit_stats=True)
        resolver = DuplicateTimestampResolver(policy)
        
        resolver.resolve(df)
        stats = resolver.get_stats()
        
        assert stats['total_timestamps'] == 5
        assert stats['unique_timestamps'] == 2
        assert stats['duplicate_count'] == 5
        assert stats['duplicate_rate'] == 1.0
    
    def test_convenience_function(self):
        """Convenience function should work correctly."""
        df = pd.DataFrame({
            'timestamp': ['2024-01-01 00:00'] * 2,
            'activity': ['a', 'b'],
        })
        
        result, stats = resolve_duplicate_timestamps(
            df, method="first", tie_breaker="first"
        )
        
        assert len(result) == 1
        assert stats['duplicate_count'] == 2


class TestItem15_ClassCoverageGate:
    """Item 15: Class Coverage Gate (Train/Val/Calibration)."""
    
    def test_gate_passes_with_full_coverage(self):
        """Gate should pass when all classes present in all splits."""
        gate = ClassCoverageGate(critical_classes=[0, 1, 2])
        
        y_train = np.array([0] * 20 + [1] * 20 + [2] * 20)
        y_val = np.array([0] * 10 + [1] * 10 + [2] * 10)
        y_calib = np.array([0] * 10 + [1] * 10 + [2] * 10)
        
        result = gate.evaluate(y_train, y_val, y_calib, room_name='bedroom')
        
        assert result['passes'] is True
        assert result['promotable'] is True
        assert len(result['reasons']) == 0
    
    def test_gate_fails_when_critical_class_absent_from_train(self):
        """Gate should fail when critical class missing from training."""
        gate = ClassCoverageGate(critical_classes=[0, 1, 2])
        
        y_train = np.array([0] * 20 + [1] * 20)  # Class 2 missing!
        y_val = np.array([0] * 10 + [1] * 10 + [2] * 10)
        
        result = gate.evaluate(y_train, y_val, room_name='bedroom')
        
        assert result['passes'] is False
        assert 'absent' in ' '.join(result['reasons']).lower()
        assert 2 in result['analysis']['absent_from_train']
    
    def test_gate_fails_when_insufficient_support(self):
        """Gate should fail when class has insufficient support."""
        gate = ClassCoverageGate(
            critical_classes=[0, 1],
            min_train_support=10,
        )
        
        y_train = np.array([0] * 20 + [1] * 5)  # Class 1 has only 5 samples
        
        result = gate.evaluate(y_train, room_name='bedroom')
        
        assert result['passes'] is False
        assert 'insufficient' in ' '.join(result['reasons']).lower()
    
    def test_gate_fails_when_class_absent_from_val(self):
        """Gate should fail when critical class missing from validation."""
        gate = ClassCoverageGate(
            critical_classes=[0, 1],
            min_val_support=5,
        )
        
        y_train = np.array([0] * 20 + [1] * 20)
        y_val = np.array([0] * 20)  # Class 1 missing from validation!
        
        result = gate.evaluate(y_train, y_val, room_name='bedroom')
        
        assert result['passes'] is False
        assert 1 in result['analysis']['val_insufficient_support']
    
    def test_coverage_ratio_check(self):
        """Gate should check minimum class coverage ratio."""
        gate = ClassCoverageGate(
            critical_classes=[0, 1, 2, 3, 4],  # 5 classes
            min_class_coverage_ratio=0.8,  # Need 4 out of 5
        )
        
        y_train = np.array([0] * 20 + [1] * 20 + [2] * 20)  # Only 3 classes
        
        result = gate.evaluate(y_train, room_name='bedroom')
        
        assert result['passes'] is False
        assert 'coverage' in ' '.join(result['reasons']).lower()
    
    def test_convenience_function(self):
        """Convenience function should work correctly."""
        y_train = np.array([0] * 20 + [1] * 15)  # Both classes have sufficient samples
        y_val = np.array([0] * 10 + [1] * 10)
        
        promotable, reasons, result = check_class_coverage_across_splits(
            y_train, y_val, None, room_name='bedroom'
        )
        
        assert promotable is True  # Should pass with default thresholds
        assert isinstance(reasons, list)


class TestWeek2EndToEnd:
    """End-to-end integration tests for Week 2."""
    
    def test_post_gap_and_class_coverage_together(self):
        """Both gates should work together in pipeline."""
        # Create fragmented data with class imbalance
        timestamps = []
        for i in range(5):
            segment_start = datetime(2024, 1, 1) + timedelta(hours=i * 3)
            segment_ts = pd.date_range(segment_start, periods=20, freq='1min')
            timestamps.extend(segment_ts)
        
        raw_df = pd.DataFrame({'timestamp': timestamps})
        post_gap_df = raw_df.copy()
        
        # Post-gap gate
        post_gap_gate = PostGapRetentionGate(max_contiguous_segments=3)
        post_gap_result = post_gap_gate.evaluate(raw_df, post_gap_df, 'bedroom')
        
        # Should fail due to fragmentation
        assert post_gap_result['passes'] is False
    
    def test_duplicate_resolution_before_sequence_creation(self):
        """Duplicates should be resolved before sequence creation."""
        # Create data with duplicates
        timestamps = ['2024-01-01 00:00'] * 2 + list(pd.date_range('2024-01-01 00:01', periods=48, freq='1min'))
        df = pd.DataFrame({
            'timestamp': timestamps,
            'activity': ['walking', 'sleeping'] + ['eating'] * 48,
        })
        
        # Resolve duplicates
        resolved_df, stats = resolve_duplicate_timestamps(df)
        
        assert stats['duplicate_count'] == 2
        assert len(resolved_df) == 49  # 50 - 1 duplicate timestamp
        
        # Now can safely create sequences
        # (Would need sensor data for actual sequence creation)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
