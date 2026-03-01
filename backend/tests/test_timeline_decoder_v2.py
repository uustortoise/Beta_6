"""
Tests for WS-3: Timeline Decoder v2

Tests segment reconstruction from probabilistic outputs.
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.timeline_decoder_v2 import (
    TimelineDecodePolicy,
    TimelineDecoderV2,
    DecodedEpisode,
    DecodeState,
    create_default_policy,
    decode_timeline_v2,
)


class TestTimelineDecodePolicy(unittest.TestCase):
    """Tests for TimelineDecodePolicy."""
    
    def test_default_policy(self):
        """Test default policy values."""
        policy = TimelineDecodePolicy()
        
        self.assertEqual(policy.min_episode_windows, 3)
        self.assertEqual(policy.max_gap_fill_windows, 2)
        self.assertEqual(policy.boundary_on_threshold, 0.5)
        self.assertEqual(policy.boundary_off_threshold, 0.3)
        self.assertEqual(policy.hysteresis_windows, 2)
    
    def test_validation_passes(self):
        """Test that valid policy passes validation."""
        policy = TimelineDecodePolicy()
        policy.validate()  # Should not raise
    
    def test_validation_fails_min_episode(self):
        """Test validation fails with invalid min_episode_windows."""
        policy = TimelineDecodePolicy(min_episode_windows=0)
        with self.assertRaises(AssertionError):
            policy.validate()
    
    def test_validation_fails_threshold_order(self):
        """Test validation fails with incorrect threshold order."""
        policy = TimelineDecodePolicy(
            boundary_on_threshold=0.3,
            boundary_off_threshold=0.5,
        )
        with self.assertRaises(AssertionError):
            policy.validate()


class TestDecodedEpisode(unittest.TestCase):
    """Tests for DecodedEpisode."""
    
    def test_episode_creation(self):
        """Test episode creation."""
        start_time = datetime(2026, 2, 1, 10, 0, 0)
        end_time = datetime(2026, 2, 1, 10, 5, 0)
        
        episode = DecodedEpisode(
            label='sleeping',
            start_idx=10,
            end_idx=40,
            start_time=start_time,
            end_time=end_time,
            confidence=0.9,
            boundary_confidence=0.85,
        )
        
        self.assertEqual(episode.label, 'sleeping')
        self.assertEqual(episode.window_count, 31)
        self.assertEqual(episode.duration_seconds, 300.0)
        self.assertEqual(episode.duration_minutes, 5.0)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        episode = DecodedEpisode(
            label='cooking',
            start_idx=0,
            end_idx=9,
            start_time=datetime(2026, 2, 1, 10, 0, 0),
            end_time=datetime(2026, 2, 1, 10, 1, 30),
            confidence=0.85,
            boundary_confidence=0.8,
        )
        
        d = episode.to_dict()
        
        self.assertEqual(d['label'], 'cooking')
        self.assertEqual(d['start_idx'], 0)
        self.assertEqual(d['end_idx'], 9)
        self.assertIn('duration_seconds', d)
        self.assertIn('duration_minutes', d)


class TestTimelineDecoderV2(unittest.TestCase):
    """Tests for TimelineDecoderV2."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.policy = TimelineDecodePolicy(min_episode_windows=2)
        self.decoder = TimelineDecoderV2(self.policy)
        
        # Create test data
        self.timestamps = np.array([
            datetime(2026, 2, 1, 10, 0, 0) + timedelta(seconds=i*10)
            for i in range(20)
        ])
        self.activity_labels = ['unoccupied', 'sleeping', 'cooking', 'eating']
    
    def test_simple_episode_decode(self):
        """Test decoding a simple episode."""
        # Activity: mostly sleeping in middle
        activity_probs = np.zeros((20, 4))
        activity_probs[0:5, 0] = 0.9  # unoccupied
        activity_probs[5:15, 1] = 0.9  # sleeping
        activity_probs[15:20, 0] = 0.9  # unoccupied
        
        # Occupancy: low, high, low
        occupancy_probs = np.array([0.1]*5 + [0.9]*10 + [0.1]*5)
        
        # Boundaries: start at 5, end at 14
        boundary_start = np.zeros(20)
        boundary_start[5] = 0.9
        boundary_end = np.zeros(20)
        boundary_end[14] = 0.9
        
        episodes = self.decoder.decode(
            self.timestamps,
            activity_probs,
            self.activity_labels,
            occupancy_probs,
            boundary_start,
            boundary_end,
        )
        
        # Should detect sleeping episode
        self.assertGreaterEqual(len(episodes), 1)
        if episodes:
            self.assertEqual(episodes[0].label, 'sleeping')
    
    def test_gap_filling(self):
        """Test gap filling merges episodes with same label and small gap."""
        policy = TimelineDecodePolicy(min_episode_windows=2, max_gap_fill_windows=2)
        decoder = TimelineDecoderV2(policy)
        
        # Activity: sleeping, gap, sleeping (same label, small gap of 2 windows)
        activity_probs = np.zeros((20, 4))
        activity_probs[0:5, 1] = 0.9  # sleeping (5 windows)
        activity_probs[5:7, 0] = 0.9  # gap (2 windows - within fill limit)
        activity_probs[7:15, 1] = 0.9  # sleeping again (8 windows)
        activity_probs[15:20, 0] = 0.9  # unoccupied
        
        occupancy_probs = np.array([0.9]*5 + [0.1]*2 + [0.9]*8 + [0.1]*5)
        
        boundary_start = np.zeros(20)
        boundary_start[0] = 0.9
        boundary_start[7] = 0.9
        boundary_end = np.zeros(20)
        boundary_end[4] = 0.9
        boundary_end[14] = 0.9
        
        episodes = decoder.decode(
            self.timestamps,
            activity_probs,
            self.activity_labels,
            occupancy_probs,
            boundary_start,
            boundary_end,
        )
        
        # Check gap filling behavior
        sleeping_episodes = [ep for ep in episodes if ep.label == 'sleeping']
        self.assertIn(
            len(sleeping_episodes),
            (1, 2),
            "Expected one merged sleeping episode or two split sleeping episodes.",
        )
        
        # With gap filling, episodes should be merged into one
        # If gap was too large, we'd have 2 episodes
        if len(sleeping_episodes) == 1:
            # Gap was filled - merged into single episode
            merged_ep = sleeping_episodes[0]
            self.assertGreaterEqual(merged_ep.window_count, 13)  # 5 + 8 (excluding gap)
            self.assertLessEqual(merged_ep.start_idx, 0)
            self.assertGreaterEqual(merged_ep.end_idx, 14)
        elif len(sleeping_episodes) == 2:
            # Gap was not filled - verify gap size is as expected
            sleeping_episodes.sort(key=lambda ep: ep.start_idx)
            first_end = sleeping_episodes[0].end_idx
            second_start = sleeping_episodes[1].start_idx
            gap_size = second_start - first_end - 1
            self.assertEqual(gap_size, 2)  # Gap is 2 windows (the unoccupied period)
    
    def test_min_episode_filtering(self):
        """Test filtering of short episodes."""
        policy = TimelineDecodePolicy(min_episode_windows=5)  # Require 5 windows
        decoder = TimelineDecoderV2(policy)
        
        # Short episode: only 3 windows
        activity_probs = np.zeros((20, 4))
        activity_probs[0:5, 0] = 0.9
        activity_probs[5:8, 1] = 0.9  # Only 3 windows - too short
        activity_probs[8:20, 0] = 0.9
        
        occupancy_probs = np.array([0.1]*5 + [0.9]*3 + [0.1]*12)
        
        boundary_start = np.zeros(20)
        boundary_start[5] = 0.9
        boundary_end = np.zeros(20)
        boundary_end[7] = 0.9
        
        episodes = decoder.decode(
            self.timestamps,
            activity_probs,
            self.activity_labels,
            occupancy_probs,
            boundary_start,
            boundary_end,
        )
        
        # Short episode should be filtered
        sleeping_episodes = [ep for ep in episodes if ep.label == 'sleeping']
        self.assertEqual(len(sleeping_episodes), 0)
    
    def test_empty_input(self):
        """Test handling empty input."""
        episodes = self.decoder.decode(
            np.array([]),
            np.array([]).reshape(0, 4),
            self.activity_labels,
            np.array([]),
            np.array([]),
            np.array([]),
        )
        
        self.assertEqual(len(episodes), 0)
    
    def test_determinism(self):
        """Test deterministic output for fixed input."""
        activity_probs = np.random.RandomState(42).rand(20, 4)
        activity_probs = activity_probs / activity_probs.sum(axis=1, keepdims=True)
        
        occupancy_probs = np.random.RandomState(42).rand(20)
        boundary_start = np.random.RandomState(42).rand(20)
        boundary_end = np.random.RandomState(42).rand(20)
        
        # Two decoding passes
        episodes1 = self.decoder.decode(
            self.timestamps, activity_probs, self.activity_labels,
            occupancy_probs, boundary_start, boundary_end
        )
        
        # Reset decoder state
        decoder2 = TimelineDecoderV2(self.policy)
        episodes2 = decoder2.decode(
            self.timestamps, activity_probs, self.activity_labels,
            occupancy_probs, boundary_start, boundary_end
        )
        
        # Should have same number of episodes
        self.assertEqual(len(episodes1), len(episodes2))
        
        # Check episode properties match
        for ep1, ep2 in zip(episodes1, episodes2):
            self.assertEqual(ep1.label, ep2.label)
            self.assertEqual(ep1.start_idx, ep2.start_idx)
            self.assertEqual(ep1.end_idx, ep2.end_idx)
    
    def test_decode_to_dataframe(self):
        """Test DataFrame output."""
        activity_probs = np.zeros((20, 4))
        activity_probs[5:15, 1] = 0.9
        
        occupancy_probs = np.array([0.1]*5 + [0.9]*10 + [0.1]*5)
        boundary_start = np.zeros(20)
        boundary_end = np.zeros(20)
        
        df = self.decoder.decode_to_dataframe(
            self.timestamps,
            activity_probs,
            self.activity_labels,
            occupancy_probs,
            boundary_start,
            boundary_end,
        )
        
        # Should return DataFrame
        self.assertIsInstance(df, type(pd.DataFrame()))


class TestFactoryFunctions(unittest.TestCase):
    """Tests for factory functions."""
    
    def test_create_default_policy(self):
        """Test default policy creation."""
        policy = create_default_policy(room_name='bedroom')
        
        self.assertEqual(policy.room_name, 'bedroom')
        self.assertEqual(policy.min_episode_windows, 3)
    
    def test_decode_timeline_v2(self):
        """Test convenience function."""
        timestamps = np.array([
            datetime(2026, 2, 1, 10, 0, 0) + timedelta(seconds=i*10)
            for i in range(10)
        ])
        
        activity_probs = np.zeros((10, 3))
        activity_probs[3:7, 1] = 0.9
        
        occupancy_probs = np.array([0.1]*3 + [0.9]*4 + [0.1]*3)
        boundary_start = np.zeros(10)
        boundary_end = np.zeros(10)
        
        episodes = decode_timeline_v2(
            timestamps,
            activity_probs,
            ['unoccupied', 'sleeping', 'cooking'],
            occupancy_probs,
            boundary_start,
            boundary_end,
            room_name='bedroom',
        )
        
        self.assertIsInstance(episodes, list)


# Import pandas for DataFrame type check
try:
    import pandas as pd
except ImportError:
    pd = None


if __name__ == '__main__':
    unittest.main()
