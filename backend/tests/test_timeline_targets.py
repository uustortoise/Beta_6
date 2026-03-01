"""
Tests for WS-1: Timeline Target Builder

Tests boundary target extraction and episode attribute targets.
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.timeline_targets import (
    BoundaryTargets,
    EpisodeAttributeTargets,
    build_boundary_targets,
    build_boundary_targets_from_episodes,
    build_episode_attribute_targets,
    build_multi_room_targets,
    targets_to_dataframe,
    validate_no_temporal_leakage,
)


class TestBoundaryTargets(unittest.TestCase):
    """Tests for boundary target generation."""
    
    def test_simple_episode(self):
        """Test boundary detection for a simple episode (excludes unoccupied transitions)."""
        labels = np.array(['unoccupied', 'sleeping', 'sleeping', 'sleeping', 'unoccupied'])
        
        targets = build_boundary_targets(labels)
        
        # Transitions to/from unoccupied are NOT boundaries
        # Only care activity episodes matter
        self.assertEqual(targets.start_flags[0], 0)  # First window, unoccupied
        self.assertEqual(targets.start_flags[1], 1)  # Sleeping episode STARTS
        self.assertEqual(targets.start_flags[2], 0)  # Continuing
        
        # Check end flags
        self.assertEqual(targets.end_flags[2], 0)    # Continuing
        self.assertEqual(targets.end_flags[3], 1)    # Sleeping episode ENDS
        self.assertEqual(targets.end_flags[4], 0)    # Unoccupied - not an episode end
    
    def test_multiple_episodes(self):
        """Test boundary detection for multiple episodes (transitions via unoccupied ignored)."""
        labels = np.array([
            'unoccupied', 'sleeping', 'sleeping',  # Sleep episode (indices 1-2)
            'unoccupied', 'unoccupied',             # Gap (excluded)
            'cooking', 'cooking',                   # Cook episode (indices 5-6)
            'unoccupied',                           # End (excluded)
        ])
        
        targets = build_boundary_targets(labels)
        
        # Sleep episode: starts at 1, ends at 2 (unoccupied transition at 3 is NOT a boundary)
        self.assertEqual(targets.start_flags[1], 1)  # Sleeping starts
        self.assertEqual(targets.end_flags[2], 1)    # Sleeping ends
        
        # Cook episode: starts at 5, ends at 6
        self.assertEqual(targets.start_flags[5], 1)  # Cooking starts
        self.assertEqual(targets.end_flags[6], 1)    # Cooking ends
    
    def test_consecutive_different_labels(self):
        """Test consecutive different care labels (boundaries between care activities)."""
        labels = np.array(['sleeping', 'cooking', 'eating', 'unoccupied'])
        
        targets = build_boundary_targets(labels)
        
        # sleeping (index 0) starts immediately
        self.assertEqual(targets.start_flags[0], 1)  # Sleeping starts
        # sleeping->cooking: boundary at index 0/1
        self.assertEqual(targets.end_flags[0], 1)    # Sleeping ends
        # cooking->eating: boundary at index 1/2  
        self.assertEqual(targets.start_flags[1], 1)  # Cooking starts
        self.assertEqual(targets.end_flags[1], 1)    # Cooking ends
        # eating starts at index 2
        self.assertEqual(targets.start_flags[2], 1)  # Eating starts
        # eating->unoccupied: eating ENDS when person leaves (care to excluded)
        self.assertEqual(targets.end_flags[2], 1)    # Eating ends
    
    def test_single_window_episode(self):
        """Test single window episode (cooking) surrounded by unoccupied."""
        labels = np.array(['unoccupied', 'cooking', 'unoccupied'])
        
        targets = build_boundary_targets(labels)
        
        # Single care activity window is both start and end
        # But transitions to/from unoccupied are NOT boundaries
        self.assertEqual(targets.start_flags[1], 1)  # Cooking starts
        self.assertEqual(targets.end_flags[1], 1)    # Cooking ends (single window episode)
    
    def test_all_unoccupied(self):
        """Test sequence with all unoccupied."""
        labels = np.array(['unoccupied', 'unoccupied', 'unoccupied'])
        
        targets = build_boundary_targets(labels)
        
        # No boundaries for unoccupied
        self.assertEqual(np.sum(targets.start_flags), 0)
        self.assertEqual(np.sum(targets.end_flags), 0)
    
    def test_empty_labels_raises_error(self):
        """Test that empty labels raise error."""
        with self.assertRaises(ValueError):
            build_boundary_targets(np.array([]))
    
    def test_single_label_raises_error(self):
        """Test that single label raises error."""
        with self.assertRaises(ValueError):
            build_boundary_targets(np.array(['sleeping']))
    
    def test_timestamps_generated(self):
        """Test that timestamps are generated correctly."""
        labels = np.array(['unoccupied', 'sleeping', 'sleeping', 'unoccupied'])
        window_duration = 10.0
        
        targets = build_boundary_targets(labels, window_duration)
        
        expected_timestamps = np.array([0.0, 10.0, 20.0, 30.0])
        np.testing.assert_array_equal(targets.timestamps, expected_timestamps)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        labels = np.array(['unoccupied', 'sleeping', 'unoccupied'])
        targets = build_boundary_targets(labels)
        
        d = targets.to_dict()
        
        self.assertIn('start_flags', d)
        self.assertIn('end_flags', d)
        self.assertIn('timestamps', d)
        self.assertEqual(len(d['start_flags']), 3)


class TestEpisodeAttributeTargets(unittest.TestCase):
    """Tests for episode attribute target generation."""
    
    def test_simple_episode_attributes(self):
        """Test attribute extraction for simple episode."""
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = np.array([
            base_time,
            base_time + timedelta(seconds=10),
            base_time + timedelta(seconds=20),
            base_time + timedelta(seconds=30),
        ])
        labels = np.array(['sleeping', 'sleeping', 'sleeping', 'unoccupied'])
        
        targets = build_episode_attribute_targets(timestamps, labels, 'bedroom')
        
        # Should have 1 sleeping episode
        self.assertEqual(targets.episode_counts.get('sleeping'), 1)
        # Duration: 3 windows * 10s = 30s = 0.5 minutes
        self.assertAlmostEqual(targets.episode_durations.get('sleeping', 0), 0.5, places=2)
    
    def test_multiple_episode_attributes(self):
        """Test attribute extraction for multiple episodes."""
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = np.array([
            base_time + timedelta(seconds=i*10)
            for i in range(10)
        ])
        labels = np.array([
            'sleeping', 'sleeping',      # 20s
            'unoccupied',                # 10s gap
            'cooking', 'cooking',        # 20s
            'unoccupied',                # 10s gap
            'sleeping', 'sleeping',      # 20s
            'unoccupied', 'unoccupied',  # end
        ])
        
        # Use 20s min duration to capture the 20s episodes
        targets = build_episode_attribute_targets(
            timestamps, labels, 'kitchen',
            min_episode_duration_seconds=20.0
        )
        
        # 2 sleeping episodes (20s each)
        self.assertEqual(targets.episode_counts.get('sleeping'), 2)
        # 1 cooking episode (20s)
        self.assertEqual(targets.episode_counts.get('cooking'), 1)

    def test_excluded_gap_splits_same_label_into_two_episodes(self):
        """Excluded labels must break episodes; no bridging across unoccupied gaps."""
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = np.array([base_time + timedelta(seconds=i * 10) for i in range(6)])
        labels = np.array([
            'sleeping', 'sleeping',      # episode 1 (20s)
            'unoccupied',                # excluded gap
            'sleeping', 'sleeping',      # episode 2 (20s)
            'unoccupied',
        ])

        targets = build_episode_attribute_targets(
            timestamps,
            labels,
            'bedroom',
            min_episode_duration_seconds=20.0,
        )
        self.assertEqual(targets.episode_counts.get('sleeping'), 2)
        self.assertAlmostEqual(targets.episode_durations.get('sleeping', 0.0), 40.0 / 60.0, places=4)

    def test_excluded_labels_not_counted_as_episodes(self):
        """Unoccupied/unknown should never appear in episode count/duration targets."""
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = np.array([base_time + timedelta(seconds=i * 10) for i in range(5)])
        labels = np.array(['unoccupied', 'unknown', 'sleeping', 'unknown', 'unoccupied'])

        targets = build_episode_attribute_targets(
            timestamps,
            labels,
            'bedroom',
            min_episode_duration_seconds=10.0,
        )
        self.assertNotIn('unoccupied', targets.episode_counts)
        self.assertNotIn('unknown', targets.episode_counts)
        self.assertEqual(targets.episode_counts.get('sleeping'), 1)
    
    def test_min_duration_filtering(self):
        """Test minimum duration filtering."""
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = np.array([
            base_time + timedelta(seconds=i*10)
            for i in range(5)
        ])
        labels = np.array(['cooking', 'unoccupied', 'unoccupied', 'unoccupied', 'unoccupied'])
        
        # Min duration 30s - single 10s window should be filtered
        targets = build_episode_attribute_targets(
            timestamps, labels, 'kitchen',
            min_episode_duration_seconds=30.0
        )
        
        self.assertNotIn('cooking', targets.episode_counts)
    
    def test_empty_labels(self):
        """Test handling of empty labels."""
        targets = build_episode_attribute_targets(
            np.array([]), np.array([]), 'bedroom'
        )
        
        self.assertEqual(targets.episode_counts, {})
        self.assertEqual(targets.episode_durations, {})
    
    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched lengths raise error."""
        timestamps = np.array([datetime.now()])
        labels = np.array(['sleeping', 'sleeping'])
        
        with self.assertRaises(ValueError):
            build_episode_attribute_targets(timestamps, labels, 'bedroom')
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = np.array([base_time, base_time + timedelta(seconds=10)])
        labels = np.array(['sleeping', 'unoccupied'])
        
        targets = build_episode_attribute_targets(timestamps, labels, 'bedroom')
        d = targets.to_dict()
        
        self.assertIn('episode_counts', d)
        self.assertIn('episode_durations', d)
        self.assertIn('episode_starts', d)
        self.assertIn('episode_ends', d)
        self.assertIn('episode_labels', d)


class TestValidateNoTemporalLeakage(unittest.TestCase):
    """Tests for temporal leakage validation."""
    
    def test_no_leakage_detected(self):
        """Test that no leakage is detected for valid targets."""
        labels = np.array(['unoccupied', 'sleeping', 'sleeping', 'cooking', 'unoccupied'])
        targets = build_boundary_targets(labels)
        
        # The validator now uses independent logic
        is_valid = validate_no_temporal_leakage(labels, targets)
        
        # If this fails, there's a mismatch between build function and validator
        if not is_valid:
            # Debug output
            print(f"labels: {labels}")
            print(f"start_flags: {targets.start_flags}")
            print(f"end_flags: {targets.end_flags}")
        
        self.assertTrue(is_valid)
    
    def test_leakage_detected_wrong_length(self):
        """Test that wrong length triggers leakage detection."""
        labels = np.array(['sleeping', 'sleeping'])
        targets = build_boundary_targets(labels)
        
        # Modify targets to have wrong length
        targets.start_flags = np.array([1, 0, 1])
        
        is_valid = validate_no_temporal_leakage(labels, targets)
        
        self.assertFalse(is_valid)
    
    def test_leakage_detected_wrong_end_flag_length(self):
        """Regression test: end_flags length mismatch should fail validation."""
        labels = np.array(['sleeping', 'sleeping'])
        targets = build_boundary_targets(labels)
        
        # Modify end_flags to have wrong length
        targets.end_flags = np.array([1, 0, 1])
        
        is_valid = validate_no_temporal_leakage(labels, targets)
        
        self.assertFalse(is_valid)
    
    def test_empty_labels_no_crash(self):
        """Regression test: Empty labels should not crash validator."""
        # Create empty targets manually since build_boundary_targets requires len >= 2
        targets = BoundaryTargets(
            start_flags=np.array([], dtype=np.int32),
            end_flags=np.array([], dtype=np.int32),
            timestamps=np.array([], dtype=float),
        )
        labels = np.array([])
        
        # Should not crash and should return True (no leakage with no data)
        is_valid = validate_no_temporal_leakage(labels, targets)
        self.assertTrue(is_valid)


class TestBuildBoundaryTargetsFromEpisodes(unittest.TestCase):
    """Tests for building targets from episode list."""
    
    def test_from_episodes(self):
        """Test building targets from episode dictionaries."""
        episodes = [
            {'start_idx': 1, 'end_idx': 3, 'label': 'sleeping'},
            {'start_idx': 5, 'end_idx': 7, 'label': 'cooking'},
        ]
        
        targets = build_boundary_targets_from_episodes(episodes, total_windows=10)
        
        self.assertEqual(targets.start_flags[1], 1)
        self.assertEqual(targets.end_flags[3], 1)
        self.assertEqual(targets.start_flags[5], 1)
        self.assertEqual(targets.end_flags[7], 1)
        self.assertEqual(np.sum(targets.start_flags), 2)
        self.assertEqual(np.sum(targets.end_flags), 2)
    
    def test_empty_episodes(self):
        """Test with empty episode list."""
        targets = build_boundary_targets_from_episodes([], total_windows=5)
        
        self.assertEqual(np.sum(targets.start_flags), 0)
        self.assertEqual(np.sum(targets.end_flags), 0)


class TestBuildMultiRoomTargets(unittest.TestCase):
    """Tests for multi-room target building."""
    
    def test_multi_room(self):
        """Test building targets for multiple rooms (excludes unoccupied)."""
        room_data = {
            'bedroom': {
                'labels': np.array(['unoccupied', 'sleeping', 'sleeping', 'unoccupied']),
            },
            'kitchen': {
                'labels': np.array(['unoccupied', 'cooking', 'unoccupied', 'unoccupied']),
            },
        }
        
        targets = build_multi_room_targets(room_data)
        
        self.assertIn('bedroom', targets)
        self.assertIn('kitchen', targets)
        
        # Bedroom should have sleeping episode start at index 1 (not 0, that's unoccupied)
        self.assertEqual(targets['bedroom'].start_flags[1], 1)  # Sleeping starts
        self.assertEqual(targets['bedroom'].end_flags[2], 1)    # Sleeping ends
        
        # Kitchen should have cooking episode
        self.assertEqual(targets['kitchen'].start_flags[1], 1)  # Cooking starts
        self.assertEqual(targets['kitchen'].end_flags[1], 1)    # Cooking ends (single window)
    
    def test_empty_room_skipped(self):
        """Test that empty rooms are skipped."""
        room_data = {
            'bedroom': {'labels': np.array(['sleeping', 'sleeping'])},
            'kitchen': {'labels': np.array([])},
        }
        
        targets = build_multi_room_targets(room_data)
        
        self.assertIn('bedroom', targets)
        self.assertNotIn('kitchen', targets)


class TestTargetsToDataframe(unittest.TestCase):
    """Tests for DataFrame conversion."""
    
    def test_conversion(self):
        """Test conversion to DataFrame."""
        labels = np.array(['unoccupied', 'sleeping', 'sleeping', 'unoccupied'])
        targets = build_boundary_targets(labels)
        
        df = targets_to_dataframe(targets)
        
        self.assertEqual(len(df), 4)
        self.assertIn('timestamp', df.columns)
        self.assertIn('start_flag', df.columns)
        self.assertIn('end_flag', df.columns)
        self.assertIn('is_boundary', df.columns)


class TestDeterminism(unittest.TestCase):
    """Tests for determinism requirements."""
    
    def test_boundary_targets_deterministic(self):
        """Test that boundary targets are deterministic."""
        labels = np.array(['unoccupied', 'sleeping', 'cooking', 'eating', 'unoccupied'])
        
        targets1 = build_boundary_targets(labels)
        targets2 = build_boundary_targets(labels)
        
        np.testing.assert_array_equal(targets1.start_flags, targets2.start_flags)
        np.testing.assert_array_equal(targets1.end_flags, targets2.end_flags)
        np.testing.assert_array_equal(targets1.timestamps, targets2.timestamps)
    
    def test_episode_attributes_deterministic(self):
        """Test that episode attributes are deterministic."""
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = np.array([base_time + timedelta(seconds=i*10) for i in range(10)])
        labels = np.array(['sleeping'] * 5 + ['unoccupied'] * 5)
        
        targets1 = build_episode_attribute_targets(timestamps, labels, 'bedroom')
        targets2 = build_episode_attribute_targets(timestamps, labels, 'bedroom')
        
        self.assertEqual(targets1.episode_counts, targets2.episode_counts)
        self.assertEqual(targets1.episode_durations, targets2.episode_durations)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""
    
    def test_rapid_transitions(self):
        """Test rapid label transitions."""
        labels = np.array(['a', 'b', 'a', 'b', 'a'])
        
        targets = build_boundary_targets(labels)
        
        # Every position should be both start and end (except first/last)
        self.assertEqual(targets.start_flags[0], 1)
        self.assertEqual(targets.end_flags[0], 1)
        for i in range(1, 4):
            self.assertEqual(targets.start_flags[i], 1)
            self.assertEqual(targets.end_flags[i], 1)
    
    def test_string_timestamps(self):
        """Test handling of string timestamps."""
        timestamps = np.array([
            '2026-02-01T10:00:00',
            '2026-02-01T10:00:10',
            '2026-02-01T10:00:20',
        ])
        labels = np.array(['sleeping', 'sleeping', 'unoccupied'])
        
        # Use 20s min duration to capture the 20s episode
        targets = build_episode_attribute_targets(
            timestamps, labels, 'bedroom',
            min_episode_duration_seconds=20.0
        )
        
        self.assertEqual(targets.episode_counts.get('sleeping'), 1)
    
    def test_datetime_serialization_with_numpy_datetime64(self):
        """Regression test: to_dict should handle numpy.datetime64 without crash."""
        base_time = np.datetime64('2026-02-01T10:00:00')
        timestamps = np.array([
            base_time,
            base_time + np.timedelta64(10, 's'),
            base_time + np.timedelta64(20, 's'),
        ])
        labels = np.array(['sleeping', 'sleeping', 'unoccupied'])
        
        targets = build_episode_attribute_targets(
            timestamps, labels, 'bedroom',
            min_episode_duration_seconds=20.0
        )
        
        # Should not raise AttributeError on isoformat()
        d = targets.to_dict()
        
        self.assertIn('episode_starts', d)
        self.assertIn('episode_ends', d)
        # Verify the timestamps are properly serialized as ISO strings
        self.assertEqual(len(d['episode_starts']), 1)
        self.assertTrue(isinstance(d['episode_starts'][0], str))


if __name__ == '__main__':
    unittest.main()
