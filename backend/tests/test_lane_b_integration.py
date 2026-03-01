"""Integration tests across Lane B modules (KPI -> gates -> fusion)."""

import sys
import unittest
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.event_gates import EventGateChecker, GateStatus
from ml.event_kpi import EventKPICalculator, EventKPIConfig
from ml.home_empty_fusion import HomeEmptyFusion, HomeEmptyConfig, HomeEmptyState, HouseholdGate


class TestLaneBIntegration(unittest.TestCase):
    def test_kpi_to_gate_flow_with_real_dataframes(self):
        base = datetime(2026, 2, 1, 8, 0, 0)
        pred = pd.DataFrame(
            {
                "timestamp": [base, base, base + timedelta(seconds=10), base + timedelta(seconds=10)],
                "room": ["bedroom", "kitchen", "bedroom", "kitchen"],
                "predicted_label": ["unoccupied", "unoccupied", "sleeping", "unoccupied"],
            }
        )
        gt = pd.DataFrame(
            {
                "timestamp": [base + timedelta(seconds=2), base + timedelta(seconds=2), base + timedelta(seconds=12), base + timedelta(seconds=12)],
                "room": ["bedroom", "kitchen", "bedroom", "kitchen"],
                "label": ["unoccupied", "unoccupied", "sleeping", "unoccupied"],
            }
        )

        kpi = EventKPICalculator(EventKPIConfig(timestamp_tolerance_seconds=5.0)).calculate_metrics(pred, gt)
        gate_report = EventGateChecker().check_all_gates(kpi.to_gate_metrics(), date(2026, 2, 1))

        self.assertIn(gate_report.overall_status, {GateStatus.PASS, GateStatus.WARNING, GateStatus.FAIL})
        # Safety gate should be present and evaluable from real metrics.
        names = {r.gate_name for r in gate_report.results}
        self.assertIn("home_empty_precision", names)

    def test_fusion_to_household_gate_with_timestamp_jitter(self):
        base = datetime(2026, 2, 1, 9, 0, 0)
        timestamps = [base, base + timedelta(seconds=10), base + timedelta(seconds=20)]
        room_predictions = {
            "bedroom": pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "predicted_label": ["unoccupied", "sleeping", "sleeping"],
                    "occupancy_prob": [0.1, 0.8, 0.85],
                }
            ),
            "kitchen": pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "predicted_label": ["unoccupied", "unoccupied", "unoccupied"],
                    "occupancy_prob": [0.1, 0.1, 0.1],
                }
            ),
        }

        fusion = HomeEmptyFusion(HomeEmptyConfig(alignment_tolerance_seconds=15.0, min_rooms_for_consensus=2))
        preds = fusion.fuse(room_predictions, timestamps)
        self.assertEqual(len(preds), 3)
        self.assertIn(preds[0].state, {HomeEmptyState.EMPTY, HomeEmptyState.OCCUPIED, HomeEmptyState.UNCERTAIN})

        gt = [
            (base + timedelta(seconds=3), True),
            (base + timedelta(seconds=13), False),
            (base + timedelta(seconds=23), False),
        ]
        household = HouseholdGate(HomeEmptyConfig(alignment_tolerance_seconds=15.0))
        out = household.check_household_gate(preds, gt)
        self.assertIn("overall_passed", out)
        self.assertIn("precision_check", out)


if __name__ == "__main__":
    unittest.main()
