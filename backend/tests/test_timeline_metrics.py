"""
Tests for WS-4: Timeline Metrics

Tests timeline-quality metric computation.
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.timeline_metrics import (
    TimelineMetrics,
    compute_event_level_quality,
    compute_fragmentation_rate,
    compute_timeline_metrics,
    match_episodes,
    compute_segment_boundary_mae,
    compute_duration_mae,
    compute_room_timeline_metrics,
    compute_aggregate_timeline_metrics,
)


class TestTimelineMetrics(unittest.TestCase):
    """Tests for TimelineMetrics dataclass."""
    
    def test_creation(self):
        """Test creating timeline metrics."""
        metrics = TimelineMetrics(
            segment_start_mae_minutes=5.0,
            segment_end_mae_minutes=3.0,
            segment_duration_mae_minutes=10.0,
            episode_count_error=1,
            fragmentation_rate=0.2,
        )
        
        self.assertEqual(metrics.segment_start_mae_minutes, 5.0)
        self.assertEqual(metrics.fragmentation_rate, 0.2)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = TimelineMetrics(
            segment_start_mae_minutes=5.0,
            fragmentation_rate=0.2,
            num_pred_episodes=10,
            num_gt_episodes=8,
        )
        
        d = metrics.to_dict()
        
        self.assertEqual(d['segment_start_mae_minutes'], 5.0)
        self.assertEqual(d['fragmentation_rate'], 0.2)
        self.assertEqual(d['num_pred_episodes'], 10)

    def test_timeline_metrics_include_event_level_quality_fields(self):
        metrics = TimelineMetrics(event_iou=0.75, onset_tolerance_rate=0.8, offset_tolerance_rate=0.9)
        payload = metrics.to_dict()

        self.assertEqual(payload["event_iou"], 0.75)
        self.assertEqual(payload["onset_tolerance_rate"], 0.8)
        self.assertEqual(payload["offset_tolerance_rate"], 0.9)


class TestComputeFragmentationRate(unittest.TestCase):
    """Tests for fragmentation rate computation."""
    
    def test_no_fragmentation(self):
        """Test perfect match (no fragmentation)."""
        pred = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0)},
        ]
        gt = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0)},
        ]
        
        frag = compute_fragmentation_rate(pred, gt)
        
        self.assertEqual(frag, 0.0)  # No fragmentation
    
    def test_fragmentation_one_to_many(self):
        """Test fragmentation where one GT episode becomes multiple pred episodes."""
        gt = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 11, 0, 0)},
        ]
        pred = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 20, 0)},
            {'start_time': datetime(2026, 2, 1, 10, 20, 0), 'end_time': datetime(2026, 2, 1, 10, 40, 0)},
            {'start_time': datetime(2026, 2, 1, 10, 40, 0), 'end_time': datetime(2026, 2, 1, 11, 0, 0)},
        ]
        
        frag = compute_fragmentation_rate(pred, gt)
        
        # 3 pred episodes, 1 matched GT -> fragmentation = (3-1)/1 = 2.0
        self.assertEqual(frag, 2.0)
    
    def test_empty_gt(self):
        """Test with empty ground truth."""
        pred = [{'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0)}]
        gt = []
        
        frag = compute_fragmentation_rate(pred, gt)
        
        self.assertEqual(frag, 0.0)  # No GT to fragment
    
    def test_empty_pred(self):
        """Test with empty predictions."""
        pred = []
        gt = [{'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0)}]
        
        frag = compute_fragmentation_rate(pred, gt)
        
        self.assertEqual(frag, 1.0)  # Complete fragmentation (no predictions)

    def test_fragmentation_respects_label_semantics(self):
        """Predictions with wrong labels should not count as matched episodes."""
        pred = [
            {
                'start_time': datetime(2026, 2, 1, 10, 0, 0),
                'end_time': datetime(2026, 2, 1, 10, 30, 0),
                'label': 'cooking',
            }
        ]
        gt = [
            {
                'start_time': datetime(2026, 2, 1, 10, 0, 0),
                'end_time': datetime(2026, 2, 1, 10, 30, 0),
                'label': 'sleeping',
            }
        ]

        frag = compute_fragmentation_rate(pred, gt)
        self.assertEqual(frag, 1.0)


class TestMatchEpisodes(unittest.TestCase):
    """Tests for episode matching."""
    
    def test_perfect_match(self):
        """Test perfect one-to-one matching."""
        pred = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0), 'label': 'sleeping'},
        ]
        gt = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0), 'label': 'sleeping'},
        ]
        
        matches = match_episodes(pred, gt)
        
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][:2], (0, 0))  # pred_idx, gt_idx
        # Overlap includes tolerance: 30 min episode + 5 min on each side = 40 min = 2400 sec
        self.assertAlmostEqual(matches[0][2], 2400.0, places=0)
    
    def test_label_mismatch(self):
        """Test that label mismatches prevent matching."""
        pred = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0), 'label': 'cooking'},
        ]
        gt = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0), 'label': 'sleeping'},
        ]
        
        matches = match_episodes(pred, gt)
        
        self.assertEqual(len(matches), 0)  # No match due to label mismatch
    
    def test_temporal_overlap(self):
        """Test matching with temporal overlap."""
        pred = [
            {'start_time': datetime(2026, 2, 1, 10, 5, 0), 'end_time': datetime(2026, 2, 1, 10, 25, 0), 'label': 'sleeping'},
        ]
        gt = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0), 'label': 'sleeping'},
        ]
        
        matches = match_episodes(pred, gt, tolerance_minutes=0)
        
        self.assertEqual(len(matches), 1)
        # Overlap: 10:05 to 10:25 = 20 minutes = 1200 seconds
        self.assertEqual(matches[0][2], 1200.0)
    
    def test_multiple_matches(self):
        """Test matching multiple episodes."""
        pred = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0), 'label': 'sleeping'},
            {'start_time': datetime(2026, 2, 1, 11, 0, 0), 'end_time': datetime(2026, 2, 1, 11, 30, 0), 'label': 'cooking'},
        ]
        gt = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0), 'label': 'sleeping'},
            {'start_time': datetime(2026, 2, 1, 11, 0, 0), 'end_time': datetime(2026, 2, 1, 11, 30, 0), 'label': 'cooking'},
        ]
        
        matches = match_episodes(pred, gt)
        
        self.assertEqual(len(matches), 2)

    def test_compute_event_level_quality(self):
        pred = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0), 'label': 'sleeping'},
        ]
        gt = [
            {'start_time': datetime(2026, 2, 1, 10, 1, 0), 'end_time': datetime(2026, 2, 1, 10, 31, 0), 'label': 'sleeping'},
        ]
        matches = [(0, 0, 1800.0)]

        event_iou, onset_rate, offset_rate = compute_event_level_quality(
            pred,
            gt,
            matches,
            tolerance_minutes=5.0,
        )

        self.assertGreater(event_iou, 0.9)
        self.assertEqual(onset_rate, 1.0)
        self.assertEqual(offset_rate, 1.0)


class TestComputeBoundaryMae(unittest.TestCase):
    """Tests for boundary MAE computation."""
    
    def test_perfect_boundaries(self):
        """Test perfect boundary alignment."""
        pred = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0)},
        ]
        gt = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0)},
        ]
        matches = [(0, 0, 1800.0)]
        
        start_mae, end_mae = compute_segment_boundary_mae(pred, gt, matches)
        
        self.assertEqual(start_mae, 0.0)
        self.assertEqual(end_mae, 0.0)
    
    def test_boundary_drift(self):
        """Test boundary drift detection."""
        pred = [
            {'start_time': datetime(2026, 2, 1, 10, 5, 0), 'end_time': datetime(2026, 2, 1, 10, 25, 0)},
        ]
        gt = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0)},
        ]
        matches = [(0, 0, 1200.0)]
        
        start_mae, end_mae = compute_segment_boundary_mae(pred, gt, matches)
        
        self.assertEqual(start_mae, 5.0)  # 5 minutes drift
        self.assertEqual(end_mae, 5.0)    # 5 minutes drift
    
    def test_no_matches(self):
        """Test with no matches."""
        pred = []
        gt = []
        matches = []
        
        start_mae, end_mae = compute_segment_boundary_mae(pred, gt, matches)
        
        self.assertEqual(start_mae, float('inf'))
        self.assertEqual(end_mae, float('inf'))


class TestComputeDurationMae(unittest.TestCase):
    """Tests for duration MAE computation."""
    
    def test_perfect_duration(self):
        """Test perfect duration match."""
        pred = [{'duration_minutes': 30.0}]
        gt = [{'duration_minutes': 30.0}]
        matches = [(0, 0, 1800.0)]
        
        mae = compute_duration_mae(pred, gt, matches)
        
        self.assertEqual(mae, 0.0)
    
    def test_duration_error(self):
        """Test duration error detection."""
        pred = [{'duration_minutes': 25.0}]
        gt = [{'duration_minutes': 30.0}]
        matches = [(0, 0, 1500.0)]
        
        mae = compute_duration_mae(pred, gt, matches)
        
        self.assertEqual(mae, 5.0)  # 5 minutes error
    
    def test_compute_from_timestamps(self):
        """Test duration computation from timestamps when duration_minutes not provided."""
        pred = [{'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 25, 0)}]
        gt = [{'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0)}]
        matches = [(0, 0, 1500.0)]
        
        mae = compute_duration_mae(pred, gt, matches)
        
        self.assertEqual(mae, 5.0)  # 25 vs 30 minutes


class TestComputeTimelineMetrics(unittest.TestCase):
    """Tests for full timeline metrics computation."""
    
    def test_perfect_prediction(self):
        """Test metrics for perfect prediction."""
        pred = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0), 'label': 'sleeping'},
        ]
        gt = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0), 'label': 'sleeping'},
        ]
        
        metrics = compute_timeline_metrics(pred, gt)
        
        self.assertEqual(metrics.segment_start_mae_minutes, 0.0)
        self.assertEqual(metrics.segment_end_mae_minutes, 0.0)
        self.assertEqual(metrics.segment_duration_mae_minutes, 0.0)
        self.assertEqual(metrics.episode_count_error, 0)
        self.assertEqual(metrics.fragmentation_rate, 0.0)
        self.assertEqual(metrics.matched_episodes, 1)
    
    def test_fragmented_prediction(self):
        """Test metrics for fragmented prediction."""
        gt = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 11, 0, 0), 'label': 'sleeping'},
        ]
        pred = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 20, 0), 'label': 'sleeping'},
            {'start_time': datetime(2026, 2, 1, 10, 20, 0), 'end_time': datetime(2026, 2, 1, 10, 40, 0), 'label': 'sleeping'},
            {'start_time': datetime(2026, 2, 1, 10, 40, 0), 'end_time': datetime(2026, 2, 1, 11, 0, 0), 'label': 'sleeping'},
        ]
        
        metrics = compute_timeline_metrics(pred, gt)
        
        self.assertEqual(metrics.fragmentation_rate, 2.0)  # (3-1)/1 = 2
        self.assertEqual(metrics.episode_count_error, 2)  # |3-1| = 2
    
    def test_empty_predictions(self):
        """Test metrics with empty predictions."""
        pred = []
        gt = [
            {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0), 'label': 'sleeping'},
        ]
        
        metrics = compute_timeline_metrics(pred, gt)
        
        self.assertEqual(metrics.fragmentation_rate, 1.0)  # No predictions
        self.assertEqual(metrics.episode_count_error, 1)   # |0-1| = 1
        self.assertEqual(metrics.num_pred_episodes, 0)
        self.assertEqual(metrics.num_gt_episodes, 1)


class TestRoomMetrics(unittest.TestCase):
    """Tests for per-room and aggregate metrics."""
    
    def test_compute_room_metrics(self):
        """Test computing metrics per room."""
        pred_by_room = {
            'bedroom': [
                {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0), 'label': 'sleeping'},
            ],
            'kitchen': [
                {'start_time': datetime(2026, 2, 1, 11, 0, 0), 'end_time': datetime(2026, 2, 1, 11, 15, 0), 'label': 'cooking'},
            ],
        }
        gt_by_room = {
            'bedroom': [
                {'start_time': datetime(2026, 2, 1, 10, 0, 0), 'end_time': datetime(2026, 2, 1, 10, 30, 0), 'label': 'sleeping'},
            ],
            'kitchen': [
                {'start_time': datetime(2026, 2, 1, 11, 0, 0), 'end_time': datetime(2026, 2, 1, 11, 15, 0), 'label': 'cooking'},
            ],
        }
        
        metrics = compute_room_timeline_metrics(pred_by_room, gt_by_room)
        
        self.assertIn('bedroom', metrics)
        self.assertIn('kitchen', metrics)
        self.assertEqual(metrics['bedroom'].num_pred_episodes, 1)
        self.assertEqual(metrics['kitchen'].num_pred_episodes, 1)
    
    def test_aggregate_metrics(self):
        """Test aggregating metrics across rooms."""
        room_metrics = {
            'bedroom': TimelineMetrics(
                segment_start_mae_minutes=5.0,
                segment_end_mae_minutes=3.0,
                segment_duration_mae_minutes=10.0,
                fragmentation_rate=0.2,
            ),
            'kitchen': TimelineMetrics(
                segment_start_mae_minutes=3.0,
                segment_end_mae_minutes=5.0,
                segment_duration_mae_minutes=8.0,
                fragmentation_rate=0.1,
            ),
        }
        
        agg = compute_aggregate_timeline_metrics(room_metrics)
        
        self.assertEqual(agg['mean_start_mae'], 4.0)  # (5+3)/2
        self.assertEqual(agg['mean_end_mae'], 4.0)    # (3+5)/2
        self.assertEqual(agg['mean_duration_mae'], 9.0)  # (10+8)/2
        self.assertAlmostEqual(agg['mean_fragmentation'], 0.15, places=5)  # (0.2+0.1)/2
        self.assertEqual(agg['num_rooms_evaluated'], 2)


if __name__ == '__main__':
    unittest.main()
