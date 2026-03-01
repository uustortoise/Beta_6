import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from ml.exceptions import DataValidationError
from ml.pipeline import UnifiedPipeline


class TestPipelineValidationGate(unittest.TestCase):
    def _make_pipeline_stub(self) -> UnifiedPipeline:
        pipeline = UnifiedPipeline.__new__(UnifiedPipeline)
        pipeline.registry = MagicMock()
        pipeline.platform = MagicMock()
        pipeline.predictor = MagicMock()
        pipeline.denoising_window = 3
        pipeline.denoising_threshold = 4.0
        return pipeline

    @patch("ml.validation.run_validation", return_value=False)
    @patch("utils.data_loader.load_sensor_data")
    def test_predict_raises_when_integrity_validation_fails(self, mock_load_sensor_data, _mock_run_validation):
        pipeline = self._make_pipeline_stub()
        mock_load_sensor_data.return_value = {
            "room1": pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-01-01", periods=5, freq="10s"),
                    "motion": [0.1] * 5,
                    "temperature": [24.0] * 5,
                }
            )
        }
        pipeline.registry.load_models_for_elder.return_value = ["room1"]
        pipeline.predictor.run_prediction.return_value = {
            "room1": pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-01-01", periods=3, freq="10s"),
                    "predicted_activity": ["inactive", "inactive", "inactive"],
                    "confidence": [0.9, 0.9, 0.9],
                }
            )
        }
        pipeline.predictor.apply_golden_samples.side_effect = lambda raw, elder_id: raw

        with self.assertRaises(DataValidationError):
            pipeline.predict("dummy.parquet", "elder_1")

    @patch("ml.validation.run_validation", return_value=True)
    @patch("utils.data_loader.load_sensor_data")
    def test_predict_returns_results_when_validation_passes(self, mock_load_sensor_data, _mock_run_validation):
        pipeline = self._make_pipeline_stub()
        mock_load_sensor_data.return_value = {
            "room1": pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-01-01", periods=5, freq="10s"),
                    "motion": [0.1] * 5,
                    "temperature": [24.0] * 5,
                }
            )
        }
        pipeline.registry.load_models_for_elder.return_value = ["room1"]
        expected = {
            "room1": pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-01-01", periods=2, freq="10s"),
                    "predicted_activity": ["inactive", "inactive"],
                    "confidence": [0.95, 0.95],
                }
            )
        }
        pipeline.predictor.run_prediction.return_value = expected
        pipeline.predictor.apply_golden_samples.side_effect = lambda raw, elder_id: raw

        result = pipeline.predict("dummy.parquet", "elder_1")
        self.assertIn("room1", result)
        self.assertEqual(len(result["room1"]), 2)

    def test_data_viability_gate_reports_failures(self):
        report = UnifiedPipeline._evaluate_data_viability_gate(
            room_name="Bedroom",
            observed_day_count=1,
            raw_samples=1000,
            post_gap_samples=50,
            post_downsample_windows=40,
            policy={
                "resolver": None,
                "defaults": {
                    "min_observed_days": 2,
                    "min_post_gap_rows": 120,
                    "max_unresolved_drop_ratio": 0.90,
                    "min_training_windows": 100,
                },
            },
        )
        self.assertFalse(report["pass"])
        self.assertTrue(any(r.startswith("insufficient_observed_days:bedroom") for r in report["reasons"]))
        self.assertTrue(any(r.startswith("insufficient_samples:bedroom") for r in report["reasons"]))
        self.assertTrue(any(r.startswith("excessive_gap_drop_ratio:bedroom") for r in report["reasons"]))
        self.assertTrue(any(r.startswith("insufficient_training_windows:bedroom") for r in report["reasons"]))


if __name__ == "__main__":
    unittest.main()
