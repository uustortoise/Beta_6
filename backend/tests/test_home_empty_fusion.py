"""
Tests for PR-B3: Home-Empty Fusion

Tests multi-room fusion and false-empty protection.
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.home_empty_fusion import (
    EntrancePenaltyStatus,
    HomeEmptyConfig,
    HomeEmptyEpisode,
    HomeEmptyFusion,
    HomeEmptyPrediction,
    HomeEmptyState,
    HouseholdGate,
    ResidentHomeContext,
    RoomState,
    fuse_home_empty_predictions,
)
from ml.beta6.sequence.transition_builder import (
    TransitionPolicy,
    build_transition_log_matrix,
    resolve_transition_policy_for_context,
)


class TestHomeEmptyConfig(unittest.TestCase):
    """Tests for HomeEmptyConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = HomeEmptyConfig()
        
        self.assertEqual(config.min_precision, 0.95)
        self.assertEqual(config.max_false_empty_rate, 0.05)
        self.assertEqual(config.entrance_penalty_duration_seconds, 300.0)
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = HomeEmptyConfig(
            min_precision=0.96,
            entrance_penalty_duration_seconds=600.0,
        )
        
        config.validate()  # Should not raise
    
    def test_invalid_precision(self):
        """Test invalid precision value."""
        config = HomeEmptyConfig(min_precision=1.5)
        
        with self.assertRaises(ValueError):
            config.validate()

    def test_single_vs_multi_resident_context_is_typed_and_explicit(self):
        """Context contract should normalize and type household + helper + layout fields."""
        single = ResidentHomeContext.from_payload(
            {
                "household_type": "single",
                "helper_presence": False,
                "layout_topology": {"bedroom": ["entrance"]},
            }
        )
        self.assertEqual(single.household_type, "single")
        self.assertEqual(single.helper_presence, "none")
        self.assertEqual(single.layout_topology["bedroom"], ("entrance",))
        self.assertEqual(single.missing_required_fields, ())

        multi = ResidentHomeContext.from_payload(
            {
                "household_type": "double",
                "helper_presence": "part_time",
                "layout_topology": {"bedroom": ["entrance", "livingroom"]},
            }
        )
        self.assertEqual(multi.household_type, "multi")
        self.assertEqual(multi.helper_presence, "present")
        self.assertEqual(multi.layout_topology["bedroom"], ("entrance", "livingroom"))
        self.assertEqual(multi.missing_required_fields, ())

    def test_missing_required_context_is_reported_clearly(self):
        """Strict mode must report exactly which context fields are missing."""
        partial = ResidentHomeContext.from_payload({"household_type": "single"})
        config = HomeEmptyConfig(
            resident_home_context=partial,
            require_resident_context=True,
        )
        with self.assertRaisesRegex(
            ValueError,
            "resident_home_context missing required fields: helper_presence, layout_topology",
        ):
            config.validate()


class TestHomeEmptyState(unittest.TestCase):
    """Tests for HomeEmptyState enum."""
    
    def test_state_values(self):
        """Test state values."""
        self.assertEqual(HomeEmptyState.OCCUPIED.value, "occupied")
        self.assertEqual(HomeEmptyState.EMPTY.value, "empty")
        self.assertEqual(HomeEmptyState.UNCERTAIN.value, "uncertain")


class TestRoomState(unittest.TestCase):
    """Tests for RoomState."""
    
    def test_room_state_creation(self):
        """Test creating room state."""
        ts = datetime(2026, 2, 1, 10, 0, 0)
        state = RoomState(
            room_name="bedroom",
            timestamp=ts,
            occupancy_prob=0.9,
            is_occupied=True,
            activity_label="sleeping",
            confidence=0.85,
        )
        
        self.assertEqual(state.room_name, "bedroom")
        self.assertTrue(state.is_occupied)
        self.assertEqual(state.activity_label, "sleeping")


class TestHomeEmptyPrediction(unittest.TestCase):
    """Tests for HomeEmptyPrediction."""
    
    def test_prediction_creation(self):
        """Test creating prediction."""
        ts = datetime(2026, 2, 1, 10, 0, 0)
        room_states = [
            RoomState("bedroom", ts, 0.1, False),
            RoomState("kitchen", ts, 0.1, False),
        ]
        
        pred = HomeEmptyPrediction(
            timestamp=ts,
            state=HomeEmptyState.EMPTY,
            confidence=0.9,
            room_states=room_states,
            unoccupied_room_count=2,
            total_room_count=2,
        )
        
        self.assertEqual(pred.state, HomeEmptyState.EMPTY)
        self.assertEqual(pred.total_room_count, 2)
        self.assertEqual(pred.unoccupied_room_count, 2)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        ts = datetime(2026, 2, 1, 10, 0, 0)
        pred = HomeEmptyPrediction(
            timestamp=ts,
            state=HomeEmptyState.OCCUPIED,
            confidence=0.8,
            room_states=[],
            occupied_room_count=1,
            total_room_count=2,
        )
        
        d = pred.to_dict()
        
        self.assertEqual(d["state"], "occupied")
        self.assertEqual(d["occupied_room_count"], 1)


class TestHomeEmptyEpisode(unittest.TestCase):
    """Tests for HomeEmptyEpisode."""
    
    def test_episode_duration(self):
        """Test episode duration calculation."""
        start = datetime(2026, 2, 1, 10, 0, 0)
        end = datetime(2026, 2, 1, 10, 30, 0)
        
        episode = HomeEmptyEpisode(
            start_time=start,
            end_time=end,
            confidence=0.9,
            room_participation={"bedroom", "kitchen"},
        )
        
        self.assertEqual(episode.duration_seconds, 1800.0)
        self.assertEqual(episode.duration_minutes, 30.0)


class TestHomeEmptyFusion(unittest.TestCase):
    """Tests for HomeEmptyFusion."""
    
    def test_fuse_all_occupied(self):
        """Test fusion when all rooms occupied."""
        fusion = HomeEmptyFusion()
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = [base_time]
        
        room_predictions = {
            "bedroom": pd.DataFrame({
                "timestamp": [base_time],
                "occupancy_prob": [0.9],
                "predicted_label": ["sleeping"],
            }),
            "kitchen": pd.DataFrame({
                "timestamp": [base_time],
                "occupancy_prob": [0.8],
                "predicted_label": ["cooking"],
            }),
        }
        
        predictions = fusion.fuse(room_predictions, timestamps)
        
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].state, HomeEmptyState.OCCUPIED)
        self.assertEqual(predictions[0].occupied_room_count, 2)
    
    def test_fuse_all_unoccupied(self):
        """Test fusion when all rooms unoccupied."""
        fusion = HomeEmptyFusion()
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = [base_time]
        
        room_predictions = {
            "bedroom": pd.DataFrame({
                "timestamp": [base_time],
                "occupancy_prob": [0.1],
                "predicted_label": ["unoccupied"],
            }),
            "kitchen": pd.DataFrame({
                "timestamp": [base_time],
                "occupancy_prob": [0.1],
                "predicted_label": ["unoccupied"],
            }),
        }
        
        predictions = fusion.fuse(room_predictions, timestamps)
        
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].state, HomeEmptyState.EMPTY)
        self.assertEqual(predictions[0].unoccupied_room_count, 2)
    
    def test_fuse_mixed(self):
        """Test fusion with mixed occupancy."""
        fusion = HomeEmptyFusion()
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = [base_time]
        
        room_predictions = {
            "bedroom": pd.DataFrame({
                "timestamp": [base_time],
                "occupancy_prob": [0.1],
                "predicted_label": ["unoccupied"],
            }),
            "kitchen": pd.DataFrame({
                "timestamp": [base_time],
                "occupancy_prob": [0.9],
                "predicted_label": ["cooking"],
            }),
        }
        
        predictions = fusion.fuse(room_predictions, timestamps)
        
        self.assertEqual(len(predictions), 1)
        # Home is occupied if ANY room is occupied
        self.assertEqual(predictions[0].state, HomeEmptyState.OCCUPIED)

    def test_layout_context_reaches_transition_builder_without_env_reads(self):
        """Transition policy should consume explicit layout context payload only."""
        labels = ["sleep", "bedroom_normal_use", "unoccupied"]
        policy = TransitionPolicy(switch_penalty=0.20, self_transition_bias=0.03)
        baseline = build_transition_log_matrix(labels, policy=policy)

        context = ResidentHomeContext.from_payload(
            {
                "household_type": "multi",
                "helper_presence": "present",
                "layout_topology": {"bedroom": ["entrance", "livingroom", "toilet"]},
            }
        )

        with patch("os.getenv", side_effect=AssertionError("env access is not allowed")):
            contextual = build_transition_log_matrix(
                labels,
                policy=policy,
                room_name="bedroom",
                resident_home_context=context.to_runtime_payload(),
            )

        self.assertFalse(np.allclose(baseline, contextual))

    def test_household_helper_context_changes_decoder_constraints_offline_only(self):
        """Helper/household context should adjust decoder constraints from explicit payload only."""
        base_policy = TransitionPolicy(switch_penalty=0.20, self_transition_bias=0.03)
        context = ResidentHomeContext.from_payload(
            {
                "household_type": "multi",
                "helper_presence": "present",
                "layout_topology": {"bedroom": ["entrance"]},
            }
        )

        baseline = resolve_transition_policy_for_context(
            policy=base_policy,
            room_name="kitchen",
            resident_home_context=None,
        )

        with patch("os.getenv", side_effect=AssertionError("env access is not allowed")):
            contextual = resolve_transition_policy_for_context(
                policy=base_policy,
                room_name="kitchen",
                resident_home_context=context.to_runtime_payload(),
            )

        self.assertGreater(contextual.switch_penalty, baseline.switch_penalty)
        self.assertEqual(contextual.self_transition_bias, baseline.self_transition_bias)
    
    def test_entrance_penalty(self):
        """Test entrance penalty logic."""
        config = HomeEmptyConfig(entrance_penalty_duration_seconds=60.0)
        fusion = HomeEmptyFusion(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        
        # First prediction: home empty (need 2 rooms for min_rooms_for_consensus=2 default)
        timestamps = [base_time]
        room_predictions = {
            "bedroom": pd.DataFrame({
                "timestamp": [base_time],
                "occupancy_prob": [0.1],
                "predicted_label": ["unoccupied"],
            }),
            "kitchen": pd.DataFrame({
                "timestamp": [base_time],
                "occupancy_prob": [0.1],
                "predicted_label": ["unoccupied"],
            }),
        }
        
        pred1 = fusion.fuse(room_predictions, timestamps)[0]
        self.assertEqual(pred1.state, HomeEmptyState.EMPTY)
        
        # Second prediction shortly after: someone enters
        timestamps2 = [base_time + timedelta(seconds=30)]
        room_predictions2 = {
            "bedroom": pd.DataFrame({
                "timestamp": timestamps2,
                "occupancy_prob": [0.6],  # Would normally be occupied
                "predicted_label": ["sleeping"],
            }),
            "kitchen": pd.DataFrame({
                "timestamp": timestamps2,
                "occupancy_prob": [0.1],
                "predicted_label": ["unoccupied"],
            }),
        }
        
        pred2 = fusion.fuse(room_predictions2, timestamps2)[0]
        # Should track entrance (home occupied now)
        self.assertEqual(pred2.state, HomeEmptyState.OCCUPIED)
        
        # Third prediction: try to go empty again (within penalty window)
        timestamps3 = [base_time + timedelta(seconds=60)]
        room_predictions3 = {
            "bedroom": pd.DataFrame({
                "timestamp": timestamps3,
                "occupancy_prob": [0.1],
                "predicted_label": ["unoccupied"],
            }),
            "kitchen": pd.DataFrame({
                "timestamp": timestamps3,
                "occupancy_prob": [0.1],
                "predicted_label": ["unoccupied"],
            }),
        }
        
        pred3 = fusion.fuse(room_predictions3, timestamps3)[0]
        # Should have entrance penalty info
        self.assertIsNotNone(pred3.seconds_since_entrance)
    
    def test_detect_episodes(self):
        """Test episode detection."""
        # Use 0 min duration to ensure episode is detected
        config = HomeEmptyConfig(min_empty_duration_seconds=0)
        fusion = HomeEmptyFusion(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        predictions = [
            HomeEmptyPrediction(
                timestamp=base_time,
                state=HomeEmptyState.EMPTY,
                confidence=0.9,
                room_states=[],
                unoccupied_room_count=2,
                total_room_count=2,
            ),
            HomeEmptyPrediction(
                timestamp=base_time + timedelta(seconds=10),
                state=HomeEmptyState.EMPTY,
                confidence=0.9,
                room_states=[],
                unoccupied_room_count=2,
                total_room_count=2,
            ),
            HomeEmptyPrediction(
                timestamp=base_time + timedelta(seconds=20),
                state=HomeEmptyState.OCCUPIED,
                confidence=0.8,
                room_states=[],
                occupied_room_count=1,
                total_room_count=2,
            ),
        ]
        
        episodes = fusion.detect_episodes(predictions)
        
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].duration_seconds, 10.0)
    
    def test_detect_episodes_min_duration(self):
        """Test episode detection with minimum duration."""
        config = HomeEmptyConfig(min_empty_duration_seconds=60.0)
        fusion = HomeEmptyFusion(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        predictions = [
            HomeEmptyPrediction(
                timestamp=base_time,
                state=HomeEmptyState.EMPTY,
                confidence=0.9,
                room_states=[],
                unoccupied_room_count=2,
                total_room_count=2,
            ),
            HomeEmptyPrediction(
                timestamp=base_time + timedelta(seconds=10),
                state=HomeEmptyState.EMPTY,
                confidence=0.9,
                room_states=[],
                unoccupied_room_count=2,
                total_room_count=2,
            ),
            HomeEmptyPrediction(
                timestamp=base_time + timedelta(seconds=20),
                state=HomeEmptyState.OCCUPIED,
                confidence=0.8,
                room_states=[],
                occupied_room_count=1,
                total_room_count=2,
            ),
        ]
        
        episodes = fusion.detect_episodes(predictions)
        
        # Episode too short - should be filtered
        self.assertEqual(len(episodes), 0)
    
    def test_insufficient_rooms(self):
        """Test handling insufficient rooms for consensus."""
        config = HomeEmptyConfig(min_rooms_for_consensus=1)  # Allow single room
        fusion = HomeEmptyFusion(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = [base_time]
        
        # Only one room
        room_predictions = {
            "bedroom": pd.DataFrame({
                "timestamp": [base_time],
                "occupancy_prob": [0.1],
                "predicted_label": ["unoccupied"],
            }),
        }
        
        predictions = fusion.fuse(room_predictions, timestamps)
        
        # Should work with single room
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].state, HomeEmptyState.EMPTY)

    def test_find_nearest_prediction_respects_max_age(self):
        """Stale rows beyond max age should not be used."""
        config = HomeEmptyConfig(max_prediction_age_seconds=10.0)
        fusion = HomeEmptyFusion(config)
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        pred_df = pd.DataFrame({
            "timestamp": [base_time],
            "occupancy_prob": [0.1],
            "predicted_label": ["unoccupied"],
        })
        result = fusion._find_nearest_prediction(pred_df, base_time + timedelta(seconds=120))
        self.assertIsNone(result)

    def test_consensus_threshold_blocks_empty_with_many_unknowns(self):
        """High empty consensus requirement should return uncertain when unknowns dominate."""
        config = HomeEmptyConfig(room_consensus_threshold=0.8, min_rooms_for_consensus=1)
        fusion = HomeEmptyFusion(config)
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        timestamps = [base_time]
        room_predictions = {
            "bedroom": pd.DataFrame({
                "timestamp": [base_time],
                "occupancy_prob": [0.1],
                "predicted_label": ["unoccupied"],
            }),
            "kitchen": pd.DataFrame({
                "timestamp": [base_time - timedelta(minutes=10)],  # stale -> unknown
                "occupancy_prob": [0.1],
                "predicted_label": ["unoccupied"],
            }),
        }
        preds = fusion.fuse(room_predictions, timestamps)
        self.assertEqual(preds[0].state, HomeEmptyState.UNCERTAIN)


class TestHouseholdGate(unittest.TestCase):
    """Tests for HouseholdGate."""
    
    def test_check_precision_pass(self):
        """Test precision check passing."""
        gate = HouseholdGate()
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        predictions = [
            HomeEmptyPrediction(
                timestamp=base_time,
                state=HomeEmptyState.EMPTY,
                confidence=0.95,
                room_states=[],
            ),
        ]
        
        ground_truth = [(base_time, True)]  # Actually empty
        
        results = gate.check_household_gate(predictions, ground_truth)
        
        self.assertTrue(results["precision_check"]["passed"])
    
    def test_check_precision_fail(self):
        """Test precision check failing."""
        config = HomeEmptyConfig(min_precision=0.95)
        gate = HouseholdGate(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        predictions = [
            HomeEmptyPrediction(
                timestamp=base_time,
                state=HomeEmptyState.EMPTY,
                confidence=0.5,
                room_states=[],
            ),
        ]
        
        ground_truth = [(base_time, False)]  # Actually occupied
        
        results = gate.check_household_gate(predictions, ground_truth)
        
        self.assertFalse(results["precision_check"]["passed"])
    
    def test_check_false_empty_rate(self):
        """Test false-empty rate check."""
        config = HomeEmptyConfig(max_false_empty_rate=0.05)
        gate = HouseholdGate(config)
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        
        # 100 predictions, 5 false empties = 5% rate (at threshold)
        predictions = []
        ground_truth = []
        
        for i in range(95):
            predictions.append(HomeEmptyPrediction(
                timestamp=base_time + timedelta(seconds=i),
                state=HomeEmptyState.OCCUPIED,
                confidence=0.9,
                room_states=[],
            ))
            ground_truth.append((base_time + timedelta(seconds=i), False))  # Occupied
        
        for i in range(5):
            predictions.append(HomeEmptyPrediction(
                timestamp=base_time + timedelta(seconds=100+i),
                state=HomeEmptyState.EMPTY,  # False empty
                confidence=0.9,
                room_states=[],
            ))
            ground_truth.append((base_time + timedelta(seconds=100+i), False))  # Actually occupied
        
        results = gate.check_household_gate(predictions, ground_truth)
        
        # 5% false empty rate should pass (at threshold)
        self.assertTrue(results["false_empty_rate_check"]["passed"])
    
    def test_coverage_check(self):
        """Test coverage check."""
        gate = HouseholdGate()
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        predictions = [
            HomeEmptyPrediction(
                timestamp=base_time + timedelta(seconds=i),
                state=HomeEmptyState.OCCUPIED,
                confidence=0.9,
                room_states=[],
            )
            for i in range(100)
        ]
        
        results = gate.check_household_gate(predictions)
        
        self.assertTrue(results["coverage_check"]["passed"])

    def test_household_gate_alignment_tolerance(self):
        """Timestamp jitter should still align with tolerance."""
        config = HomeEmptyConfig(alignment_tolerance_seconds=15.0)
        gate = HouseholdGate(config)
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        predictions = [
            HomeEmptyPrediction(
                timestamp=base_time,
                state=HomeEmptyState.EMPTY,
                confidence=0.9,
                room_states=[],
            ),
        ]
        ground_truth = [(base_time + timedelta(seconds=10), True)]
        results = gate.check_household_gate(predictions, ground_truth)
        self.assertTrue(results["precision_check"]["passed"])
    
    def test_overall_gate(self):
        """Test overall gate evaluation."""
        gate = HouseholdGate()
        
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        predictions = [
            HomeEmptyPrediction(
                timestamp=base_time,
                state=HomeEmptyState.EMPTY,
                confidence=0.95,
                room_states=[],
            ),
        ]
        
        ground_truth = [(base_time, True)]
        
        results = gate.check_household_gate(predictions, ground_truth)
        
        self.assertTrue(results["overall_passed"])


class TestFuseHomeEmptyPredictions(unittest.TestCase):
    """Tests for fuse_home_empty_predictions convenience function."""
    
    def test_convenience_function(self):
        """Test convenience function."""
        base_time = datetime(2026, 2, 1, 10, 0, 0)
        
        room_predictions = {
            "bedroom": pd.DataFrame({
                "timestamp": [base_time, base_time + timedelta(seconds=10)],
                "occupancy_prob": [0.1, 0.9],
                "predicted_label": ["unoccupied", "sleeping"],
            }),
            "kitchen": pd.DataFrame({
                "timestamp": [base_time, base_time + timedelta(seconds=10)],
                "occupancy_prob": [0.1, 0.1],
                "predicted_label": ["unoccupied", "unoccupied"],
            }),
        }
        
        df = fuse_home_empty_predictions(room_predictions)
        
        self.assertEqual(len(df), 2)
        self.assertIn("state", df.columns)
        self.assertIn("confidence", df.columns)


if __name__ == '__main__':
    unittest.main()
