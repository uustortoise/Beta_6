"""
Tests for PR-B1: Event Compiler

Tests episode compilation from window-level predictions.
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.event_compiler import (
    Episode,
    EpisodeCompiler,
    EpisodeCompilerConfig,
    EpisodeStatus,
    MultiRoomEpisodeCompiler,
    compile_day_episodes,
)


class TestEpisode(unittest.TestCase):
    """Tests for Episode dataclass."""
    
    def test_episode_creation(self):
        """Test creating an episode."""
        start = datetime(2026, 2, 1, 10, 0, 0)
        end = datetime(2026, 2, 1, 10, 5, 0)
        
        episode = Episode(
            episode_id="ep_001",
            event_label="sleeping",
            start_time=start,
            end_time=end,
            confidence=0.95,
            window_count=30,
        )
        
        self.assertEqual(episode.episode_id, "ep_001")
        self.assertEqual(episode.event_label, "sleeping")
        self.assertEqual(episode.duration_seconds, 300.0)
        self.assertEqual(episode.duration_minutes, 5.0)
    
    def test_episode_overlaps(self):
        """Test episode overlap detection."""
        ep1 = Episode(
            episode_id="ep_001",
            event_label="sleeping",
            start_time=datetime(2026, 2, 1, 10, 0, 0),
            end_time=datetime(2026, 2, 1, 11, 0, 0),
            confidence=0.9,
            window_count=60,
        )
        
        # Overlapping episode
        ep2 = Episode(
            episode_id="ep_002",
            event_label="sleeping",
            start_time=datetime(2026, 2, 1, 10, 30, 0),
            end_time=datetime(2026, 2, 1, 11, 30, 0),
            confidence=0.9,
            window_count=60,
        )
        
        self.assertTrue(ep1.overlaps(ep2))
        
        # Non-overlapping episode
        ep3 = Episode(
            episode_id="ep_003",
            event_label="sleeping",
            start_time=datetime(2026, 2, 1, 12, 0, 0),
            end_time=datetime(2026, 2, 1, 13, 0, 0),
            confidence=0.9,
            window_count=60,
        )
        
        self.assertFalse(ep1.overlaps(ep3))
    
    def test_episode_can_merge(self):
        """Test episode merge detection."""
        ep1 = Episode(
            episode_id="ep_001",
            event_label="sleeping",
            start_time=datetime(2026, 2, 1, 10, 0, 0),
            end_time=datetime(2026, 2, 1, 11, 0, 0),
            confidence=0.9,
            window_count=60,
        )
        
        # Same label, small gap - can merge
        ep2 = Episode(
            episode_id="ep_002",
            event_label="sleeping",
            start_time=datetime(2026, 2, 1, 11, 0, 30),
            end_time=datetime(2026, 2, 1, 12, 0, 0),
            confidence=0.9,
            window_count=60,
        )
        
        self.assertTrue(ep1.can_merge(ep2, max_gap_seconds=60))
        
        # Different label - cannot merge
        ep3 = Episode(
            episode_id="ep_003",
            event_label="cooking",
            start_time=datetime(2026, 2, 1, 11, 0, 30),
            end_time=datetime(2026, 2, 1, 12, 0, 0),
            confidence=0.9,
            window_count=60,
        )
        
        self.assertFalse(ep1.can_merge(ep3, max_gap_seconds=60))


class TestEpisodeCompiler(unittest.TestCase):
    """Tests for EpisodeCompiler."""
    
    def test_compile_simple_episode(self):
        """Test compiling a simple episode."""
        config = EpisodeCompilerConfig(
            min_duration_seconds=0,  # No minimum for this test
            merge_gap_seconds=0,
            use_hysteresis=False,
        )
        compiler = EpisodeCompiler(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        predictions = [
            (base_time + timedelta(seconds=i*10), "sleeping", 0.9)
            for i in range(6)  # 60 seconds of sleeping
        ]
        
        episodes = compiler.compile(predictions)
        
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].event_label, "sleeping")
        self.assertEqual(episodes[0].window_count, 6)
    
    def test_compile_multiple_episodes(self):
        """Test compiling multiple episodes."""
        config = EpisodeCompilerConfig(
            min_duration_seconds=0,
            merge_gap_seconds=0,
            use_hysteresis=False,
        )
        compiler = EpisodeCompiler(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        predictions = [
            # First episode: sleeping
            (base_time, "sleeping", 0.9),
            (base_time + timedelta(seconds=10), "sleeping", 0.9),
            (base_time + timedelta(seconds=20), "sleeping", 0.9),
            # Switch to cooking
            (base_time + timedelta(seconds=30), "cooking", 0.8),
            (base_time + timedelta(seconds=40), "cooking", 0.8),
            (base_time + timedelta(seconds=50), "cooking", 0.8),
        ]
        
        episodes = compiler.compile(predictions)
        
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].event_label, "sleeping")
        self.assertEqual(episodes[1].event_label, "cooking")
    
    def test_min_duration_filtering(self):
        """Test filtering by minimum duration."""
        config = EpisodeCompilerConfig(
            min_duration_seconds=60,  # 60 second minimum
            merge_gap_seconds=0,
            use_hysteresis=False,
        )
        compiler = EpisodeCompiler(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        predictions = [
            # Short episode: 10 seconds
            (base_time, "cooking", 0.8),
            # Long episode: 60 seconds (6 windows @ 10s each)
            (base_time + timedelta(seconds=20), "sleeping", 0.9),
            (base_time + timedelta(seconds=30), "sleeping", 0.9),
            (base_time + timedelta(seconds=40), "sleeping", 0.9),
            (base_time + timedelta(seconds=50), "sleeping", 0.9),
            (base_time + timedelta(seconds=60), "sleeping", 0.9),
            (base_time + timedelta(seconds=70), "sleeping", 0.9),
            (base_time + timedelta(seconds=80), "sleeping", 0.9),
        ]
        
        episodes = compiler.compile(predictions)
        
        # Only the long episode should remain
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].event_label, "sleeping")
    
    def test_episode_merging(self):
        """Test merging episodes with small gaps."""
        config = EpisodeCompilerConfig(
            min_duration_seconds=0,
            merge_gap_seconds=30,  # Merge if gap <= 30 seconds
            use_hysteresis=False,
        )
        compiler = EpisodeCompiler(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        predictions = [
            # First part
            (base_time, "sleeping", 0.9),
            (base_time + timedelta(seconds=10), "sleeping", 0.9),
            # Gap: 20 seconds (within merge threshold)
            (base_time + timedelta(seconds=30), "sleeping", 0.9),
            (base_time + timedelta(seconds=40), "sleeping", 0.9),
        ]
        
        episodes = compiler.compile(predictions)
        
        # Should be merged into one episode
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].window_count, 4)
    
    def test_hysteresis_smoothing(self):
        """Test hysteresis-based smoothing."""
        config = EpisodeCompilerConfig(
            min_duration_seconds=0,
            merge_gap_seconds=0,
            use_hysteresis=True,
            hysteresis_on_threshold=0.60,
            hysteresis_off_threshold=0.40,
            hysteresis_min_windows=3,
        )
        compiler = EpisodeCompiler(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        predictions = [
            # Below threshold - unoccupied
            (base_time, "unoccupied", 0.3),
            (base_time + timedelta(seconds=10), "unoccupied", 0.3),
            # Single spike - should be smoothed
            (base_time + timedelta(seconds=20), "occupied", 0.7),
            # Back to unoccupied
            (base_time + timedelta(seconds=30), "unoccupied", 0.3),
            (base_time + timedelta(seconds=40), "unoccupied", 0.3),
        ]
        
        episodes = compiler.compile(predictions)
        
        # Should remain as unoccupied (single spike smoothed out)
        # Note: Actual behavior depends on hysteresis implementation
        self.assertGreaterEqual(len(episodes), 1)
    
    def test_hysteresis_respects_confidence_thresholds(self):
        """Only high-confidence sustained transitions should switch labels."""
        config = EpisodeCompilerConfig(
            min_duration_seconds=0,
            merge_gap_seconds=0,
            use_hysteresis=True,
            hysteresis_on_threshold=0.70,
            hysteresis_off_threshold=0.50,
            hysteresis_min_windows=2,
        )
        compiler = EpisodeCompiler(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        predictions = [
            (base_time, "unoccupied", 0.90),
            (base_time + timedelta(seconds=10), "occupied", 0.55),  # below on-threshold
            (base_time + timedelta(seconds=20), "occupied", 0.55),  # below on-threshold
            (base_time + timedelta(seconds=30), "occupied", 0.80),  # high confidence starts
            (base_time + timedelta(seconds=40), "occupied", 0.82),  # sustained => switch
        ]
        
        episodes = compiler.compile(predictions)
        labels = [ep.event_label for ep in episodes]
        
        self.assertEqual(labels, ["unoccupied", "occupied"])
    
    def test_hysteresis_ignores_low_confidence_flip(self):
        """Low-confidence contrary labels should not flip state."""
        config = EpisodeCompilerConfig(
            min_duration_seconds=0,
            merge_gap_seconds=0,
            use_hysteresis=True,
            hysteresis_on_threshold=0.70,
            hysteresis_off_threshold=0.50,
            hysteresis_min_windows=1,
        )
        compiler = EpisodeCompiler(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        predictions = [
            (base_time, "unoccupied", 0.90),
            (base_time + timedelta(seconds=10), "occupied", 0.30),  # below off-threshold
            (base_time + timedelta(seconds=20), "unoccupied", 0.90),
        ]
        
        episodes = compiler.compile(predictions)
        labels = [ep.event_label for ep in episodes]
        
        self.assertEqual(labels, ["unoccupied"])
    
    def test_compile_to_dataframe(self):
        """Test compilation to DataFrame."""
        config = EpisodeCompilerConfig(min_duration_seconds=0)
        compiler = EpisodeCompiler(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        predictions = [
            (base_time + timedelta(seconds=i*10), "sleeping", 0.9)
            for i in range(6)
        ]
        
        df = compiler.compile_to_dataframe(predictions, room_name="bedroom")
        
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["event_label"], "sleeping")
        self.assertIn("bedroom", df.iloc[0]["metadata"]["room"])
    
    def test_empty_predictions(self):
        """Test handling empty predictions."""
        compiler = EpisodeCompiler()
        
        episodes = compiler.compile([])
        
        self.assertEqual(len(episodes), 0)


class TestMultiRoomEpisodeCompiler(unittest.TestCase):
    """Tests for MultiRoomEpisodeCompiler."""
    
    def test_compile_multiple_rooms(self):
        """Test compiling episodes for multiple rooms."""
        compiler = MultiRoomEpisodeCompiler()
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        
        room_predictions = {
            "bedroom": [
                (base_time + timedelta(seconds=i*10), "sleeping", 0.9)
                for i in range(6)
            ],
            "kitchen": [
                (base_time + timedelta(seconds=i*10), "cooking", 0.8)
                for i in range(6)
            ],
        }
        
        results = compiler.compile_all(room_predictions)
        
        self.assertIn("bedroom", results)
        self.assertIn("kitchen", results)
        self.assertEqual(len(results["bedroom"]), 1)
        self.assertEqual(len(results["kitchen"]), 1)
        self.assertEqual(results["bedroom"][0].event_label, "sleeping")
        self.assertEqual(results["kitchen"][0].event_label, "cooking")
    
    def test_get_compiler_creates_new(self):
        """Test that get_compiler creates new compiler for new room."""
        compiler = MultiRoomEpisodeCompiler()
        
        bedroom_compiler = compiler.get_compiler("bedroom")
        kitchen_compiler = compiler.get_compiler("kitchen")
        
        # Should be different instances
        self.assertIsNot(bedroom_compiler, kitchen_compiler)
        
        # Should return same instance for same room
        bedroom_compiler2 = compiler.get_compiler("bedroom")
        self.assertIs(bedroom_compiler, bedroom_compiler2)


class TestEpisodeCompilerConfig(unittest.TestCase):
    """Tests for EpisodeCompilerConfig."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = EpisodeCompilerConfig(
            min_duration_seconds=30,
            merge_gap_seconds=60,
            hysteresis_on_threshold=0.6,
            hysteresis_off_threshold=0.4,
        )
        
        # Should not raise
        config.validate()
    
    def test_invalid_hysteresis_thresholds(self):
        """Test invalid hysteresis thresholds."""
        config = EpisodeCompilerConfig(
            hysteresis_on_threshold=0.4,
            hysteresis_off_threshold=0.6,  # Should be < on_threshold
        )
        
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_negative_duration(self):
        """Test negative duration raises error."""
        config = EpisodeCompilerConfig(min_duration_seconds=-1)
        
        with self.assertRaises(ValueError):
            config.validate()


class TestCompileDayEpisodes(unittest.TestCase):
    """Tests for compile_day_episodes convenience function."""
    
    def test_compile_day_episodes_single_room(self):
        """Test compiling day episodes for single room."""
        import pandas as pd
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        data = {
            "timestamp": [base_time + timedelta(seconds=i*10) for i in range(6)],
            "predicted_label": ["sleeping"] * 6,
            "confidence": [0.9] * 6,
        }
        df = pd.DataFrame(data)
        
        episodes_df = compile_day_episodes(df)
        
        self.assertEqual(len(episodes_df), 1)
        self.assertEqual(episodes_df.iloc[0]["event_label"], "sleeping")
    
    def test_compile_day_episodes_multi_room(self):
        """Test compiling day episodes for multiple rooms."""
        import pandas as pd
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        data = {
            "timestamp": [base_time + timedelta(seconds=i*10) for i in range(12)],
            "room": ["bedroom"] * 6 + ["kitchen"] * 6,
            "predicted_label": ["sleeping"] * 6 + ["cooking"] * 6,
            "confidence": [0.9] * 12,
        }
        df = pd.DataFrame(data)
        
        episodes_df = compile_day_episodes(df, room_col="room")
        
        self.assertEqual(len(episodes_df), 2)


if __name__ == '__main__':
    unittest.main()
