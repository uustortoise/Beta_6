"""
Tests for WS-4: Timeline Gates

Tests timeline-quality gate checking.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.timeline_gates import (
    TimelineGateThresholds,
    TimelineGateResult,
    TimelineGateStatus,
    TimelineGateChecker,
    UnknownRateReport,
    create_default_timeline_checker,
)
from ml.timeline_metrics import TimelineMetrics


class TestTimelineGateThresholds(unittest.TestCase):
    """Tests for TimelineGateThresholds."""
    
    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = TimelineGateThresholds()
        
        self.assertEqual(thresholds.tier_1_recall_floor, 0.50)
        self.assertEqual(thresholds.fragmentation_improvement_threshold, 0.20)
        self.assertEqual(thresholds.segment_duration_mae_max, 120.0)
        self.assertEqual(thresholds.unknown_rate_global_max, 0.15)


class TestTimelineGateResult(unittest.TestCase):
    """Tests for TimelineGateResult."""
    
    def test_creation(self):
        """Test creating gate result."""
        result = TimelineGateResult(
            gate_name="duration_mae_bedroom",
            status=TimelineGateStatus.PASS,
            metric_value=50.0,
            threshold_value=120.0,
            message="Duration MAE: 50.0 min",
            room="bedroom",
        )
        
        self.assertEqual(result.gate_name, "duration_mae_bedroom")
        self.assertTrue(result.is_pass)
        self.assertEqual(result.room, "bedroom")
    
    def test_fail_status(self):
        """Test fail status."""
        result = TimelineGateResult(
            gate_name="fragmentation_bedroom",
            status=TimelineGateStatus.FAIL,
        )
        
        self.assertFalse(result.is_pass)


class TestTimelineGateChecker(unittest.TestCase):
    """Tests for TimelineGateChecker."""
    
    def test_creation(self):
        """Test creating checker."""
        checker = TimelineGateChecker()
        
        self.assertIsNotNone(checker.thresholds)
    
    def test_check_duration_mae_pass(self):
        """Test duration MAE gate passing."""
        checker = TimelineGateChecker()
        
        room_metrics = {
            'bedroom': TimelineMetrics(segment_duration_mae_minutes=50.0),
        }
        
        results = checker.check_timeline_metrics(room_metrics)
        
        duration_results = [r for r in results if 'duration_mae' in r.gate_name]
        self.assertEqual(len(duration_results), 1)
        self.assertTrue(duration_results[0].is_pass)
    
    def test_check_duration_mae_fail(self):
        """Test duration MAE gate failing."""
        checker = TimelineGateChecker()
        
        room_metrics = {
            'bedroom': TimelineMetrics(segment_duration_mae_minutes=150.0),
        }
        
        results = checker.check_timeline_metrics(room_metrics)
        
        duration_results = [r for r in results if 'duration_mae' in r.gate_name]
        self.assertEqual(len(duration_results), 1)
        self.assertEqual(duration_results[0].status, TimelineGateStatus.FAIL)
    
    def test_check_fragmentation_improvement(self):
        """Test fragmentation improvement gate."""
        checker = TimelineGateChecker()
        
        room_metrics = {
            'bedroom': TimelineMetrics(fragmentation_rate=0.3),
        }
        baseline_metrics = {
            'bedroom': TimelineMetrics(fragmentation_rate=0.5),
        }
        
        results = checker.check_timeline_metrics(room_metrics, baseline_metrics)
        
        frag_results = [r for r in results if 'fragmentation' in r.gate_name]
        self.assertEqual(len(frag_results), 1)
        
        # Improvement: (0.5 - 0.3) / 0.5 = 0.4 = 40%
        self.assertTrue(frag_results[0].is_pass)  # Above 20% threshold

    def test_fragmentation_without_baseline_fails_when_too_high(self):
        """Test absolute fragmentation hard-fail when no baseline is provided."""
        checker = TimelineGateChecker(
            TimelineGateThresholds(fragmentation_rate_max_without_baseline=0.50)
        )
        room_metrics = {
            'bedroom': TimelineMetrics(fragmentation_rate=0.75),
        }

        results = checker.check_timeline_metrics(room_metrics, baseline_metrics=None)
        frag_results = [r for r in results if 'fragmentation' in r.gate_name]
        self.assertEqual(len(frag_results), 1)
        self.assertEqual(frag_results[0].status, TimelineGateStatus.FAIL)
    
    def test_check_multiple_rooms(self):
        """Test checking multiple rooms."""
        checker = TimelineGateChecker()
        
        room_metrics = {
            'bedroom': TimelineMetrics(
                segment_duration_mae_minutes=50.0,
                segment_start_mae_minutes=30.0,
                segment_end_mae_minutes=25.0,
            ),
            'kitchen': TimelineMetrics(
                segment_duration_mae_minutes=80.0,
                segment_start_mae_minutes=40.0,
                segment_end_mae_minutes=35.0,
            ),
        }
        
        results = checker.check_timeline_metrics(room_metrics)
        
        # Should have gates for each room
        bedroom_results = [r for r in results if r.room == 'bedroom']
        kitchen_results = [r for r in results if r.room == 'kitchen']
        
        self.assertGreater(len(bedroom_results), 0)
        self.assertGreater(len(kitchen_results), 0)
    
    def test_check_unknown_rates(self):
        """Test unknown rate checking."""
        checker = TimelineGateChecker()
        
        predictions = [
            {'room': 'bedroom', 'predicted_label': 'sleeping'},
            {'room': 'bedroom', 'predicted_label': 'unknown'},
            {'room': 'bedroom', 'predicted_label': 'sleeping'},
            {'room': 'kitchen', 'predicted_label': 'cooking'},
        ]
        
        report = checker.check_unknown_rates(predictions)
        
        # Global: 1 unknown out of 4 = 0.25
        self.assertEqual(report.global_rate, 0.25)
        
        # Bedroom: 1 unknown out of 3 = 0.33
        self.assertEqual(report.per_room['bedroom'], 1.0/3.0)
        
        # Kitchen: 0 unknown out of 1 = 0
        self.assertEqual(report.per_room['kitchen'], 0.0)

    def test_check_unknown_rates_per_label_with_ground_truth(self):
        """Test per-label unknown governance when true labels are provided."""
        checker = TimelineGateChecker(
            TimelineGateThresholds(unknown_rate_per_label_max=0.25)
        )

        predictions = [
            {'room': 'bedroom', 'predicted_label': 'unknown', 'true_label': 'sleeping'},
            {'room': 'bedroom', 'predicted_label': 'unknown', 'true_label': 'sleeping'},
            {'room': 'bedroom', 'predicted_label': 'sleeping', 'true_label': 'sleeping'},
            {'room': 'kitchen', 'predicted_label': 'cooking', 'true_label': 'cooking'},
        ]

        report = checker.check_unknown_rates(predictions)
        self.assertIn('sleeping', report.per_label)
        self.assertAlmostEqual(report.per_label['sleeping'], 2.0 / 3.0)
        self.assertTrue(any('Label sleeping' in breach for breach in report.breaches))
    
    def test_unknown_rate_breaches(self):
        """Test unknown rate breach detection."""
        thresholds = TimelineGateThresholds(unknown_rate_per_room_max=0.20)
        checker = TimelineGateChecker(thresholds)
        
        predictions = [
            {'room': 'bedroom', 'predicted_label': 'unknown'},
            {'room': 'bedroom', 'predicted_label': 'unknown'},
            {'room': 'bedroom', 'predicted_label': 'sleeping'},
        ]
        
        report = checker.check_unknown_rates(predictions)
        
        # Bedroom: 2/3 = 0.67 > 0.20 threshold -> breach
        self.assertGreater(len(report.breaches), 0)
        self.assertIn('bedroom', report.breaches[0])
    
    def test_compute_pass_rate(self):
        """Test pass rate computation."""
        checker = TimelineGateChecker()
        
        results = [
            TimelineGateResult(gate_name='g1', status=TimelineGateStatus.PASS),
            TimelineGateResult(gate_name='g2', status=TimelineGateStatus.PASS),
            TimelineGateResult(gate_name='g3', status=TimelineGateStatus.FAIL),
        ]
        
        pass_rate = checker.compute_pass_rate(results)
        
        self.assertEqual(pass_rate, 2.0/3.0)
    
    def test_is_promotable_pass(self):
        """Test promotable check when all gates pass."""
        checker = TimelineGateChecker()
        
        results = [
            TimelineGateResult(gate_name='g1', status=TimelineGateStatus.PASS),
            TimelineGateResult(gate_name='g2', status=TimelineGateStatus.PASS),
        ]
        unknown_report = UnknownRateReport(global_rate=0.05)
        
        is_promotable, reasons = checker.is_promotable(results, unknown_report)
        
        self.assertTrue(is_promotable)
        self.assertEqual(len(reasons), 0)
    
    def test_is_promotable_fail(self):
        """Test promotable check when gates fail."""
        checker = TimelineGateChecker()
        
        results = [
            TimelineGateResult(
                gate_name='duration_mae_bedroom',
                status=TimelineGateStatus.FAIL,
                metric_value=150.0,
                threshold_value=120.0,
            ),
        ]
        unknown_report = UnknownRateReport(global_rate=0.05)
        
        is_promotable, reasons = checker.is_promotable(results, unknown_report)
        
        self.assertFalse(is_promotable)
        self.assertGreater(len(reasons), 0)
    
    def test_is_promotable_unknown_breach(self):
        """Test promotable check with unknown rate breach."""
        checker = TimelineGateChecker()
        
        results = [
            TimelineGateResult(gate_name='g1', status=TimelineGateStatus.PASS),
        ]
        unknown_report = UnknownRateReport(
            global_rate=0.25,
            breaches=['Global: unknown rate 25.00% > threshold 15.00%'],
        )
        
        is_promotable, reasons = checker.is_promotable(results, unknown_report)
        
        self.assertFalse(is_promotable)
        self.assertIn('unknown_rate_breach', reasons[0])


class TestUnknownRateReport(unittest.TestCase):
    """Tests for UnknownRateReport."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = UnknownRateReport(
            global_rate=0.15,
            per_room={'bedroom': 0.10, 'kitchen': 0.20},
            breaches=['Kitchen rate too high'],
        )
        
        d = report.to_dict()
        
        self.assertEqual(d['global_rate'], 0.15)
        self.assertEqual(d['per_room']['bedroom'], 0.10)
        self.assertEqual(len(d['breaches']), 1)


class TestFactoryFunction(unittest.TestCase):
    """Tests for factory function."""
    
    def test_create_default_checker(self):
        """Test creating default checker."""
        checker = create_default_timeline_checker()
        
        self.assertIsInstance(checker, TimelineGateChecker)
        self.assertIsInstance(checker.thresholds, TimelineGateThresholds)


if __name__ == '__main__':
    unittest.main()
