"""Tests for PR-B2: Event KPI calculator."""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.event_kpi import EventKPICalculator, EventKPIConfig


class TestEventKPICalculator(unittest.TestCase):
    def test_calculate_metrics_accepts_label_ground_truth(self):
        base = datetime(2026, 2, 1, 10, 0, 0)
        pred = pd.DataFrame(
            {
                "timestamp": [base, base, base + timedelta(seconds=10), base + timedelta(seconds=10)],
                "room": ["bedroom", "kitchen", "bedroom", "kitchen"],
                "predicted_label": ["unoccupied", "unoccupied", "sleeping", "unoccupied"],
            }
        )
        gt = pd.DataFrame(
            {
                "timestamp": [base, base, base + timedelta(seconds=10), base + timedelta(seconds=10)],
                "room": ["bedroom", "kitchen", "bedroom", "kitchen"],
                "label": ["unoccupied", "unoccupied", "sleeping", "unoccupied"],
            }
        )

        metrics = EventKPICalculator().calculate_metrics(pred, gt)

        self.assertAlmostEqual(metrics.home_empty_precision, 1.0)
        self.assertAlmostEqual(metrics.home_empty_false_empty_rate, 0.0)
        self.assertEqual(metrics.event_supports.get("sleeping"), 1)

    def test_home_empty_alignment_uses_tolerance(self):
        base = datetime(2026, 2, 1, 10, 0, 0)
        pred = pd.DataFrame(
            {
                "timestamp": [base, base],
                "room": ["bedroom", "kitchen"],
                "predicted_label": ["unoccupied", "unoccupied"],
            }
        )
        gt = pd.DataFrame(
            {
                "timestamp": [base + timedelta(seconds=5), base + timedelta(seconds=5)],
                "room": ["bedroom", "kitchen"],
                "label": ["unoccupied", "unoccupied"],
            }
        )

        calc = EventKPICalculator(EventKPIConfig(timestamp_tolerance_seconds=10.0))
        metrics = calc.calculate_metrics(pred, gt)

        self.assertGreaterEqual(metrics.home_empty_precision, 0.99)

    def test_per_event_alignment_is_room_aware(self):
        base = datetime(2026, 2, 1, 10, 0, 0)
        pred = pd.DataFrame(
            {
                "timestamp": [base, base],
                "room": ["bedroom", "kitchen"],
                "predicted_label": ["sleeping", "cooking"],
            }
        )
        gt = pd.DataFrame(
            {
                "timestamp": [base + timedelta(seconds=2), base + timedelta(seconds=2)],
                "room": ["bedroom", "kitchen"],
                "label": ["sleeping", "kitchen_normal_use"],
            }
        )

        calc = EventKPICalculator(EventKPIConfig(timestamp_tolerance_seconds=5.0))
        metrics = calc.calculate_metrics(pred, gt)

        self.assertIn("sleeping", metrics.event_recalls)
        self.assertAlmostEqual(metrics.event_recalls["sleeping"], 1.0)

    def test_episode_duration_is_inclusive(self):
        base = datetime(2026, 2, 1, 10, 0, 0)
        pred = pd.DataFrame(
            {
                "timestamp": [base, base + timedelta(seconds=10), base + timedelta(seconds=20)],
                "predicted_label": ["sleeping", "sleeping", "unoccupied"],
            }
        )

        episodes = EventKPICalculator()._predictions_to_episodes(pred)

        self.assertGreaterEqual(episodes[0]["duration_seconds"], 20.0)
        self.assertGreaterEqual(episodes[1]["duration_seconds"], 10.0)

    def test_unknown_rate_fallback_label_column(self):
        base = datetime(2026, 2, 1, 10, 0, 0)
        pred = pd.DataFrame(
            {
                "timestamp": [base, base + timedelta(seconds=10)],
                "room": ["bedroom", "bedroom"],
                "label": ["unknown", "sleeping"],
            }
        )
        calc = EventKPICalculator()
        metrics = calc.calculate_metrics(pred, None)
        self.assertAlmostEqual(metrics.unknown_rate_global, 0.5)


if __name__ == "__main__":
    unittest.main()
