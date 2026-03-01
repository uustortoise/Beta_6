"""
Tests for WS-3 calibration utilities.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.calibration import CalibrationConfig, MultiTaskCalibrator, TemperatureScaler


class TestTemperatureScaler(unittest.TestCase):
    """Tests for TemperatureScaler."""

    def test_from_dict_returns_instance(self):
        scaler = TemperatureScaler.from_dict({"temperature": 2.0, "fitted": True})
        self.assertIsInstance(scaler, TemperatureScaler)
        self.assertEqual(scaler.temperature, 2.0)
        self.assertTrue(scaler._fitted)


class TestMultiTaskCalibrator(unittest.TestCase):
    """Tests for MultiTaskCalibrator."""

    def test_activity_probs_renormalized_after_calibration(self):
        rng = np.random.RandomState(7)
        logits = rng.randn(128, 4)
        labels = rng.randint(0, 4, size=128)

        calibrator = MultiTaskCalibrator(CalibrationConfig())
        calibrator.fit({"activity": logits}, {"activity": labels})
        probs = calibrator.calibrate({"activity": logits})["activity"]

        row_sums = probs.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, np.ones_like(row_sums), atol=1e-6))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))


if __name__ == "__main__":
    unittest.main()
