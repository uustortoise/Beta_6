"""
Tests for PR-B2: Event Gates

Tests tiered gate checks and promotion eligibility.
"""

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.event_gates import (
    CriticalityTier,
    EventGateChecker,
    EventGateReport,
    EventGateThresholds,
    GateResult,
    GateStatus,
    check_promotion_eligibility,
    create_default_gate_checker,
)


class TestCriticalityTier(unittest.TestCase):
    """Tests for CriticalityTier enum."""
    
    def test_tier_ordering(self):
        """Test tier value ordering."""
        self.assertLess(CriticalityTier.TIER_1.value, CriticalityTier.TIER_2.value)
        self.assertLess(CriticalityTier.TIER_2.value, CriticalityTier.TIER_3.value)
    
    def test_tier_values(self):
        """Test tier values."""
        self.assertEqual(CriticalityTier.TIER_1.value, 1)
        self.assertEqual(CriticalityTier.TIER_2.value, 2)
        self.assertEqual(CriticalityTier.TIER_3.value, 3)


class TestGateStatus(unittest.TestCase):
    """Tests for GateStatus enum."""
    
    def test_status_values(self):
        """Test status values."""
        self.assertEqual(GateStatus.PASS.value, "pass")
        self.assertEqual(GateStatus.FAIL.value, "fail")
        self.assertEqual(GateStatus.WARNING.value, "warning")
        self.assertEqual(GateStatus.NOT_EVALUATED.value, "not_evaluated")


class TestEventGateThresholds(unittest.TestCase):
    """Tests for EventGateThresholds."""
    
    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = EventGateThresholds()
        
        self.assertEqual(thresholds.home_empty_precision_min, 0.95)
        self.assertEqual(thresholds.home_empty_false_empty_rate_max, 0.05)
        self.assertEqual(thresholds.unknown_rate_global_max, 0.15)
        self.assertEqual(thresholds.tier_1_recall_min, 0.50)
        self.assertEqual(thresholds.tier_2_recall_min, 0.35)
        self.assertEqual(thresholds.tier_3_recall_min, 0.20)
    
    def test_valid_thresholds(self):
        """Test valid threshold configuration."""
        thresholds = EventGateThresholds(
            home_empty_precision_min=0.90,
            tier_1_recall_min=0.55,
            tier_2_recall_min=0.40,
            tier_3_recall_min=0.25,
        )
        
        thresholds.validate()  # Should not raise
    
    def test_invalid_recall_order(self):
        """Test invalid recall ordering."""
        thresholds = EventGateThresholds(
            tier_1_recall_min=0.30,  # Should be > tier_2
            tier_2_recall_min=0.40,
        )
        
        with self.assertRaises(ValueError):
            thresholds.validate()


class TestGateResult(unittest.TestCase):
    """Tests for GateResult."""
    
    def test_gate_result_creation(self):
        """Test creating a gate result."""
        result = GateResult(
            gate_name="test_gate",
            status=GateStatus.PASS,
            metric_value=0.95,
            threshold_value=0.90,
            message="Test passed",
        )
        
        self.assertEqual(result.gate_name, "test_gate")
        self.assertTrue(result.is_pass)
        self.assertFalse(result.is_fail)
    
    def test_gate_result_fail(self):
        """Test failed gate result."""
        result = GateResult(
            gate_name="test_gate",
            status=GateStatus.FAIL,
            metric_value=0.80,
            threshold_value=0.90,
        )
        
        self.assertFalse(result.is_pass)
        self.assertTrue(result.is_fail)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = GateResult(
            gate_name="test_gate",
            status=GateStatus.PASS,
            metric_value=0.95,
            details={"key": "value"},
        )
        
        d = result.to_dict()
        
        self.assertEqual(d["gate_name"], "test_gate")
        self.assertEqual(d["status"], "pass")
        self.assertEqual(d["metric_value"], 0.95)


class TestEventGateChecker(unittest.TestCase):
    """Tests for EventGateChecker."""
    
    def test_checker_creation(self):
        """Test creating gate checker."""
        checker = EventGateChecker()
        
        self.assertIsNotNone(checker.thresholds)
    
    def test_check_home_empty_precision_pass(self):
        """Test home-empty precision gate passing."""
        checker = EventGateChecker()
        
        metrics = {
            "home_empty_precision": 0.96,
            "home_empty_false_empty_rate": 0.04,
        }
        
        report = checker.check_all_gates(metrics, date(2026, 2, 1))
        
        precision_result = [r for r in report.results if r.gate_name == "home_empty_precision"][0]
        self.assertEqual(precision_result.status, GateStatus.PASS)
        self.assertEqual(precision_result.metric_value, 0.96)
    
    def test_check_home_empty_precision_fail(self):
        """Test home-empty precision gate failing."""
        checker = EventGateChecker()
        
        metrics = {
            "home_empty_precision": 0.93,  # Below 0.95 threshold
            "home_empty_false_empty_rate": 0.04,
        }
        
        report = checker.check_all_gates(metrics, date(2026, 2, 1))
        
        precision_result = [r for r in report.results if r.gate_name == "home_empty_precision"][0]
        self.assertEqual(precision_result.status, GateStatus.FAIL)
        self.assertTrue(precision_result.details.get("is_critical"))
    
    def test_check_false_empty_rate_pass(self):
        """Test false-empty rate gate passing."""
        checker = EventGateChecker()
        
        metrics = {
            "home_empty_precision": 0.96,
            "home_empty_false_empty_rate": 0.04,  # Below 0.05 threshold
        }
        
        report = checker.check_all_gates(metrics, date(2026, 2, 1))
        
        fer_result = [r for r in report.results if r.gate_name == "home_empty_false_empty_rate"][0]
        self.assertEqual(fer_result.status, GateStatus.PASS)
    
    def test_check_false_empty_rate_fail(self):
        """Test false-empty rate gate failing."""
        checker = EventGateChecker()
        
        metrics = {
            "home_empty_precision": 0.96,
            "home_empty_false_empty_rate": 0.06,  # Above 0.05 threshold
        }
        
        report = checker.check_all_gates(metrics, date(2026, 2, 1))
        
        fer_result = [r for r in report.results if r.gate_name == "home_empty_false_empty_rate"][0]
        self.assertEqual(fer_result.status, GateStatus.FAIL)
        self.assertTrue(fer_result.details.get("is_critical"))
    
    def test_check_unknown_rate_global(self):
        """Test global unknown rate gate."""
        checker = EventGateChecker()
        
        # Pass case
        metrics = {"unknown_rate_global": 0.10}  # Below 0.15
        report = checker.check_all_gates(metrics, date(2026, 2, 1))
        
        result = [r for r in report.results if r.gate_name == "unknown_rate_global"][0]
        self.assertEqual(result.status, GateStatus.PASS)
        
        # Fail case
        metrics = {"unknown_rate_global": 0.20}  # Above 0.15
        report = checker.check_all_gates(metrics, date(2026, 2, 1))
        
        result = [r for r in report.results if r.gate_name == "unknown_rate_global"][0]
        self.assertEqual(result.status, GateStatus.FAIL)
    
    def test_check_tier_1_recall_pass(self):
        """Test Tier-1 recall gate passing."""
        checker = EventGateChecker()
        
        metrics = {
            "event_recalls": {"sleep_duration": 0.55},  # Above 0.50
            "event_supports": {"sleep_duration": 50},
        }
        
        report = checker.check_all_gates(metrics, date(2026, 2, 1))
        
        result = [r for r in report.results if "sleep_duration" in r.gate_name][0]
        self.assertEqual(result.status, GateStatus.PASS)
        self.assertEqual(result.details.get("tier"), 1)
    
    def test_check_tier_1_recall_fail(self):
        """Test Tier-1 recall gate failing."""
        checker = EventGateChecker()
        
        metrics = {
            "event_recalls": {"sleep_duration": 0.45},  # Below 0.50
            "event_supports": {"sleep_duration": 50},
        }
        
        report = checker.check_all_gates(metrics, date(2026, 2, 1))
        
        result = [r for r in report.results if "sleep_duration" in r.gate_name][0]
        self.assertEqual(result.status, GateStatus.FAIL)
        self.assertTrue(result.details.get("is_critical"))

    def test_check_tier_1_low_support_not_evaluated(self):
        """Tier-1 with insufficient support should be marked not_evaluated."""
        checker = EventGateChecker()
        metrics = {
            "event_recalls": {"sleep_duration": 0.99},
            "event_supports": {"sleep_duration": 3},
        }
        report = checker.check_all_gates(metrics, date(2026, 2, 1))
        result = [r for r in report.results if "sleep_duration" in r.gate_name][0]
        self.assertEqual(result.status, GateStatus.NOT_EVALUATED)
        self.assertTrue(result.details.get("insufficient_support"))
        self.assertTrue(report.is_promotable)
    
    def test_check_tier_2_recall(self):
        """Test Tier-2 recall gate."""
        checker = EventGateChecker()
        
        # Pass
        metrics = {
            "event_recalls": {"bathroom_use": 0.40},  # Above 0.35
            "event_supports": {"bathroom_use": 50},
        }
        report = checker.check_all_gates(metrics, date(2026, 2, 1))
        
        result = [r for r in report.results if "bathroom_use" in r.gate_name][0]
        self.assertEqual(result.status, GateStatus.PASS)
        self.assertEqual(result.details.get("tier"), 2)
        
        # Fail
        metrics = {
            "event_recalls": {"bathroom_use": 0.30},  # Below 0.35
            "event_supports": {"bathroom_use": 50},
        }
        report = checker.check_all_gates(metrics, date(2026, 2, 1))
        
        result = [r for r in report.results if "bathroom_use" in r.gate_name][0]
        self.assertEqual(result.status, GateStatus.FAIL)
        self.assertFalse(result.details.get("is_critical"))  # Tier-2 not critical

    def test_check_tier_2_low_support_not_evaluated(self):
        """Tier-2 with insufficient support should be not_evaluated."""
        checker = EventGateChecker()
        metrics = {
            "event_recalls": {"bathroom_use": 1.0},
            "event_supports": {"bathroom_use": 5},
        }
        report = checker.check_all_gates(metrics, date(2026, 2, 1))
        result = [r for r in report.results if "bathroom_use" in r.gate_name][0]
        self.assertEqual(result.status, GateStatus.NOT_EVALUATED)
        self.assertTrue(result.details.get("insufficient_support"))
    
    def test_collapse_detection(self):
        """Test critical collapse detection."""
        checker = EventGateChecker()
        
        metrics = {
            "event_recalls": {"sleep_duration": 0.01},  # Very low recall
            "event_supports": {"sleep_duration": 50},  # High support
        }
        
        report = checker.check_all_gates(metrics, date(2026, 2, 1))
        
        collapse_results = [r for r in report.results if "collapse" in r.gate_name]
        self.assertGreater(len(collapse_results), 0)
        self.assertEqual(collapse_results[0].status, GateStatus.FAIL)
        self.assertTrue(collapse_results[0].details.get("is_critical"))
    
    def test_no_collapse_with_low_support(self):
        """Test that low support doesn't trigger collapse."""
        checker = EventGateChecker()
        
        metrics = {
            "event_recalls": {"sleep_duration": 0.01},  # Very low recall
            "event_supports": {"sleep_duration": 10},  # Low support - no collapse
        }
        
        report = checker.check_all_gates(metrics, date(2026, 2, 1))
        
        collapse_results = [r for r in report.results if "collapse" in r.gate_name]
        self.assertEqual(len(collapse_results), 0)
    
    def test_overall_status_pass(self):
        """Test overall status when all gates pass."""
        checker = EventGateChecker()
        
        metrics = {
            "home_empty_precision": 0.96,
            "home_empty_false_empty_rate": 0.04,
            "unknown_rate_global": 0.10,
            "event_recalls": {
                "sleep_duration": 0.55,
                "bathroom_use": 0.40,
            },
            "event_supports": {
                "sleep_duration": 50,
                "bathroom_use": 50,
            },
        }
        
        report = checker.check_all_gates(metrics, date(2026, 2, 1))
        
        self.assertEqual(report.overall_status, GateStatus.PASS)
        self.assertTrue(report.is_promotable)
    
    def test_overall_status_fail_critical(self):
        """Test overall status with critical failure."""
        checker = EventGateChecker()
        
        metrics = {
            "home_empty_precision": 0.93,  # Critical failure
            "home_empty_false_empty_rate": 0.04,
        }
        
        report = checker.check_all_gates(metrics, date(2026, 2, 1))
        
        self.assertEqual(report.overall_status, GateStatus.FAIL)
        self.assertFalse(report.is_promotable)
        self.assertIn("home_empty_precision", report.critical_failures)
    
    def test_check_single_event(self):
        """Test checking a single event."""
        checker = EventGateChecker()
        
        # Tier-1 event pass
        result = checker.check_single_event("sleep_duration", 0.55, 50)
        self.assertEqual(result.status, GateStatus.PASS)
        
        # Tier-1 event fail
        result = checker.check_single_event("sleep_duration", 0.45, 50)
        self.assertEqual(result.status, GateStatus.FAIL)

        # Tier-1 low support should be not_evaluated
        result = checker.check_single_event("sleep_duration", 0.95, 2)
        self.assertEqual(result.status, GateStatus.NOT_EVALUATED)
        self.assertTrue(result.details.get("insufficient_support"))
        
        # Unknown event
        result = checker.check_single_event("unknown_event", 0.50, 50)
        self.assertEqual(result.status, GateStatus.NOT_EVALUATED)


class TestEventGateReport(unittest.TestCase):
    """Tests for EventGateReport."""
    
    def test_report_creation(self):
        """Test creating a gate report."""
        results = [
            GateResult("gate1", GateStatus.PASS),
            GateResult("gate2", GateStatus.PASS),
            GateResult("gate3", GateStatus.FAIL),
        ]
        
        report = EventGateReport(
            date=date(2026, 2, 1),
            overall_status=GateStatus.WARNING,
            results=results,
        )
        
        self.assertEqual(report.pass_count, 2)
        self.assertEqual(report.fail_count, 1)
        self.assertEqual(report.pass_rate, 2/3)
    
    def test_promotable_with_no_critical_failures(self):
        """Test promotable when no critical failures."""
        results = [
            GateResult("gate1", GateStatus.PASS),
            GateResult("gate2", GateStatus.FAIL, details={"is_critical": False}),
        ]
        
        report = EventGateReport(
            date=date(2026, 2, 1),
            overall_status=GateStatus.WARNING,
            results=results,
        )
        
        self.assertTrue(report.is_promotable)
    
    def test_not_promotable_with_critical_failure(self):
        """Test not promotable with critical failure."""
        results = [
            GateResult("gate1", GateStatus.PASS),
            GateResult("gate2", GateStatus.FAIL, details={"is_critical": True}),
        ]
        
        report = EventGateReport(
            date=date(2026, 2, 1),
            overall_status=GateStatus.FAIL,
            results=results,
        )
        
        self.assertFalse(report.is_promotable)
        self.assertIn("gate2", report.critical_failures)


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""
    
    def test_create_default_gate_checker(self):
        """Test creating default gate checker."""
        checker = create_default_gate_checker()
        
        self.assertIsInstance(checker, EventGateChecker)
        self.assertIsNotNone(checker.thresholds)
    
    def test_check_promotion_eligibility_pass(self):
        """Test promotion eligibility check passing."""
        metrics = {
            "home_empty_precision": 0.96,
            "home_empty_false_empty_rate": 0.04,
        }
        
        is_promotable, report = check_promotion_eligibility(metrics, date(2026, 2, 1))
        
        self.assertTrue(is_promotable)
        self.assertEqual(report.overall_status, GateStatus.PASS)
    
    def test_check_promotion_eligibility_fail(self):
        """Test promotion eligibility check failing."""
        metrics = {
            "home_empty_precision": 0.90,  # Below threshold
            "home_empty_false_empty_rate": 0.04,
        }
        
        is_promotable, report = check_promotion_eligibility(metrics, date(2026, 2, 1))
        
        self.assertFalse(is_promotable)
        self.assertEqual(report.overall_status, GateStatus.FAIL)


if __name__ == '__main__':
    unittest.main()
