"""
Tests for PR-B1: Event Decoder

Tests window-level decoding with hysteresis and smoothing.
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.event_decoder import (
    DecoderConfig,
    DecoderState,
    EventDecoder,
    RoomAwareDecoder,
    WindowPrediction,
    apply_decoder_to_predictions,
)


class TestDecoderConfig(unittest.TestCase):
    """Tests for DecoderConfig."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = DecoderConfig(
            occupancy_on_threshold=0.6,
            occupancy_off_threshold=0.4,
        )
        
        config.validate()  # Should not raise
    
    def test_invalid_threshold_order(self):
        """Test that off >= on raises error."""
        config = DecoderConfig(
            occupancy_on_threshold=0.4,
            occupancy_off_threshold=0.6,  # Should be < on
        )
        
        with self.assertRaises(AssertionError):
            config.validate()
    
    def test_invalid_threshold_range(self):
        """Test thresholds outside [0, 1] raise error."""
        config = DecoderConfig(occupancy_on_threshold=1.5)
        
        with self.assertRaises(AssertionError):
            config.validate()


class TestEventDecoder(unittest.TestCase):
    """Tests for EventDecoder."""
    
    def test_decode_occupied_sequence(self):
        """Test decoding a sequence of occupied windows."""
        config = DecoderConfig(use_hysteresis=False)
        decoder = EventDecoder(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = [base_time + timedelta(seconds=i*10) for i in range(5)]
        
        occupancy_probs = np.array([0.8, 0.85, 0.9, 0.87, 0.82])
        activity_probs = {"sleeping": np.array([0.9, 0.9, 0.9, 0.9, 0.9])}
        
        predictions = decoder.decode(occupancy_probs, activity_probs, timestamps, "bedroom")
        
        self.assertEqual(len(predictions), 5)
        for pred in predictions:
            self.assertEqual(pred.predicted_label, "sleeping")
            self.assertEqual(pred.room_name, "bedroom")
    
    def test_decode_unoccupied_sequence(self):
        """Test decoding unoccupied sequence."""
        config = DecoderConfig(use_hysteresis=False)
        decoder = EventDecoder(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = [base_time + timedelta(seconds=i*10) for i in range(5)]
        
        occupancy_probs = np.array([0.2, 0.15, 0.1, 0.12, 0.18])
        activity_probs = {"sleeping": np.array([0.1, 0.1, 0.1, 0.1, 0.1])}
        
        predictions = decoder.decode(occupancy_probs, activity_probs, timestamps, "bedroom")
        
        for pred in predictions:
            self.assertEqual(pred.predicted_label, "unoccupied")
    
    def test_decode_with_hysteresis(self):
        """Test decoding with hysteresis smoothing."""
        config = DecoderConfig(
            use_hysteresis=True,
            occupancy_on_threshold=0.6,
            occupancy_off_threshold=0.4,
            hysteresis_min_windows=3,
        )
        decoder = EventDecoder(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        # Pattern: unoccupied -> spike -> unoccupied -> sustained occupied
        # 10 windows total
        timestamps = [base_time + timedelta(seconds=i*10) for i in range(10)]
        
        occupancy_probs = np.array([
            0.3, 0.3,  # unoccupied (2)
            0.7,  # spike (1)
            0.3, 0.3, 0.3,  # back to unoccupied (3)
            0.7, 0.7, 0.7, 0.7,  # sustained occupied (4)
        ])
        activity_probs = {"sleeping": np.full(10, 0.9)}
        
        predictions = decoder.decode(occupancy_probs, activity_probs, timestamps, "bedroom")
        
        self.assertEqual(len(predictions), 10)
        # Single spike should be smoothed out
        # Last 3 should be occupied due to sustained high probability
    
    def test_decode_with_temporal_smoothing(self):
        """Test temporal smoothing of probabilities."""
        config = DecoderConfig(
            temporal_smoothing_window=3,
            use_hysteresis=False,
        )
        decoder = EventDecoder(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = [base_time + timedelta(seconds=i*10) for i in range(5)]
        
        # Noisy signal
        occupancy_probs = np.array([0.5, 0.9, 0.5, 0.9, 0.5])
        activity_probs = {"sleeping": np.full(5, 0.9)}
        
        predictions = decoder.decode(occupancy_probs, activity_probs, timestamps, "bedroom")
        
        self.assertEqual(len(predictions), 5)
        # Smoothing should reduce noise
    
    def test_decode_to_dataframe(self):
        """Test decoding to DataFrame."""
        config = DecoderConfig(use_hysteresis=False)
        decoder = EventDecoder(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = [base_time + timedelta(seconds=i*10) for i in range(5)]
        
        occupancy_probs = np.array([0.8, 0.85, 0.9, 0.87, 0.82])
        activity_probs = {"sleeping": np.array([0.9, 0.9, 0.9, 0.9, 0.9])}
        
        df = decoder.decode_to_dataframe(occupancy_probs, activity_probs, timestamps, "bedroom")
        
        self.assertEqual(len(df), 5)
        self.assertIn("timestamp", df.columns)
        self.assertIn("predicted_label", df.columns)
    
    def test_activity_threshold_forces_unknown(self):
        """Low-confidence activities should map to unknown when below activity_threshold."""
        config = DecoderConfig(
            use_hysteresis=False,
            use_unknown_fallback=False,
            activity_threshold=0.8,
            unknown_label="unknown",
        )
        decoder = EventDecoder(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = [base_time]
        
        occupancy_probs = np.array([0.9])
        activity_probs = {
            "sleeping": np.array([0.7]),
            "relaxing": np.array([0.6]),
        }
        
        predictions = decoder.decode(occupancy_probs, activity_probs, timestamps, "bedroom")
        
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].predicted_label, "unknown")
        self.assertTrue(predictions[0].is_unknown)
    
    def test_reset_decoder(self):
        """Test resetting decoder state."""
        config = DecoderConfig()
        decoder = EventDecoder(config)
        
        # Process some data
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = [base_time + timedelta(seconds=i*10) for i in range(5)]
        occupancy_probs = np.array([0.8, 0.85, 0.9, 0.87, 0.82])
        activity_probs = {"sleeping": np.full(5, 0.9)}
        
        decoder.decode(occupancy_probs, activity_probs, timestamps, "bedroom")
        
        # Reset
        decoder.reset()
        
        # Should start fresh
        self.assertEqual(decoder._state, DecoderState.UNOCCUPIED)
    
    def test_empty_predictions(self):
        """Test handling empty predictions."""
        decoder = EventDecoder()
        
        predictions = decoder.decode(
            np.array([]),
            {},
            [],
            "bedroom"
        )
        
        self.assertEqual(len(predictions), 0)
    
    def test_mismatched_lengths(self):
        """Test that mismatched array lengths raise error."""
        decoder = EventDecoder()
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        
        with self.assertRaises(ValueError):
            decoder.decode(
                np.array([0.8, 0.9]),  # 2 elements
                {"sleeping": np.array([0.9, 0.9, 0.9])},  # 3 elements
                [base_time, base_time + timedelta(seconds=10)],  # 2 elements
                "bedroom"
            )


class TestWindowPrediction(unittest.TestCase):
    """Tests for WindowPrediction dataclass."""
    
    def test_prediction_creation(self):
        """Test creating a prediction."""
        pred = WindowPrediction(
            timestamp=datetime(2026, 2, 1, 10, 0, 0),
            room_name="bedroom",
            occupancy_prob=0.85,
            activity_probs={"sleeping": 0.9},
            predicted_label="sleeping",
            confidence=0.9,
        )
        
        self.assertEqual(pred.room_name, "bedroom")
        self.assertEqual(pred.predicted_label, "sleeping")
        self.assertFalse(pred.is_unknown)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        pred = WindowPrediction(
            timestamp=datetime(2026, 2, 1, 10, 0, 0),
            room_name="bedroom",
            occupancy_prob=0.85,
            activity_probs={"sleeping": 0.9},
            predicted_label="sleeping",
            confidence=0.9,
            is_unknown=False,
        )
        
        d = pred.to_dict()
        
        self.assertEqual(d["room_name"], "bedroom")
        self.assertEqual(d["predicted_label"], "sleeping")
        self.assertEqual(d["occupancy_prob"], 0.85)


class TestRoomAwareDecoder(unittest.TestCase):
    """Tests for RoomAwareDecoder."""
    
    def test_decode_multiple_rooms(self):
        """Test decoding for multiple rooms."""
        decoder = RoomAwareDecoder()
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = [base_time + timedelta(seconds=i*10) for i in range(5)]
        
        occupancy_probs = np.array([0.8, 0.85, 0.9, 0.87, 0.82])
        activity_probs = {"sleeping": np.array([0.9, 0.9, 0.9, 0.9, 0.9])}
        
        # Decode for bedroom
        bedroom_preds = decoder.decode_room("bedroom", occupancy_probs, activity_probs, timestamps)
        self.assertEqual(len(bedroom_preds), 5)
        
        # Decode for kitchen
        kitchen_preds = decoder.decode_room("kitchen", occupancy_probs, activity_probs, timestamps)
        self.assertEqual(len(kitchen_preds), 5)
        
        # Should have separate decoders
        self.assertIsNot(
            decoder.get_decoder("bedroom"),
            decoder.get_decoder("kitchen")
        )
    
    def test_reset_room(self):
        """Test resetting specific room."""
        decoder = RoomAwareDecoder()
        
        # Use bedroom decoder
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = [base_time + timedelta(seconds=i*10) for i in range(5)]
        occupancy_probs = np.array([0.8, 0.85, 0.9, 0.87, 0.82])
        activity_probs = {"sleeping": np.full(5, 0.9)}
        
        decoder.decode_room("bedroom", occupancy_probs, activity_probs, timestamps)
        
        # Reset bedroom
        decoder.reset_room("bedroom")
        
        # Kitchen should still work
        kitchen_preds = decoder.decode_room("kitchen", occupancy_probs, activity_probs, timestamps)
        self.assertEqual(len(kitchen_preds), 5)
    
    def test_reset_all(self):
        """Test resetting all rooms."""
        decoder = RoomAwareDecoder()
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = [base_time + timedelta(seconds=i*10) for i in range(5)]
        occupancy_probs = np.array([0.8, 0.85, 0.9, 0.87, 0.82])
        activity_probs = {"sleeping": np.full(5, 0.9)}
        
        decoder.decode_room("bedroom", occupancy_probs, activity_probs, timestamps)
        decoder.decode_room("kitchen", occupancy_probs, activity_probs, timestamps)
        
        # Reset all
        decoder.reset_all()
        
        # Should still work after reset
        preds = decoder.decode_room("bedroom", occupancy_probs, activity_probs, timestamps)
        self.assertEqual(len(preds), 5)


class TestApplyDecoderToPredictions(unittest.TestCase):
    """Tests for apply_decoder_to_predictions function."""
    
    def test_apply_to_dataframe(self):
        """Test applying decoder to DataFrame."""
        import pandas as pd
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        data = {
            "timestamp": [base_time + timedelta(seconds=i*10) for i in range(5)],
            "room": ["bedroom"] * 5,
            "prob_occupied": [0.8, 0.85, 0.9, 0.87, 0.82],
            "prob_sleeping": [0.9, 0.9, 0.9, 0.9, 0.9],
        }
        df = pd.DataFrame(data)
        
        result = apply_decoder_to_predictions(df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
    
    def test_empty_dataframe(self):
        """Test handling empty DataFrame."""
        import pandas as pd
        
        df = pd.DataFrame()
        
        result = apply_decoder_to_predictions(df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)


if __name__ == '__main__':
    unittest.main()
