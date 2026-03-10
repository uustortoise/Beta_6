import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.livingroom_seed_forensic import build_livingroom_seed_forensic


class TestLivingRoomSeedForensic(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.backend_dir = Path(self.temp_dir.name) / "backend"
        self.models_dir = self.backend_dir / "models" / "HK0011_jessica"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.room = "LivingRoom"
        self._write_versions_json()
        self._write_trace(
            version=39,
            random_seed=41,
            macro_f1=0.0815,
            selection_mode="no_regress_macro_f1_fallback",
            threshold_0=0.59,
            threshold_1=0.96,
            calibrator_intercept=3.28,
            best_distribution={"livingroom_normal_use": 1673, "unoccupied": 91},
            last_distribution={"livingroom_normal_use": 1673, "unoccupied": 91},
            dominant_class_label="livingroom_normal_use",
            collapsed=False,
        )
        self._write_trace(
            version=40,
            random_seed=42,
            macro_f1=0.6590,
            selection_mode="no_regress_floor",
            threshold_0=0.0,
            threshold_1=0.5245,
            calibrator_intercept=-5.23,
            best_distribution={"livingroom_normal_use": 200, "unoccupied": 1564},
            last_distribution={"unoccupied": 1764},
            dominant_class_label="unoccupied",
            collapsed=False,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_versions_json(self) -> None:
        payload = {
            "current_version": 40,
            "versions": [
                {"version": 40, "created_at": "2026-03-10T21:34:48.943435"},
                {"version": 39, "created_at": "2026-03-10T21:30:57.888226"},
            ],
        }
        (self.models_dir / f"{self.room}_versions.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def _write_trace(
        self,
        *,
        version: int,
        random_seed: int,
        macro_f1: float,
        selection_mode: str,
        threshold_0: float,
        threshold_1: float,
        calibrator_intercept: float,
        best_distribution: dict[str, int],
        last_distribution: dict[str, int],
        dominant_class_label: str,
        collapsed: bool,
    ) -> None:
        trace_payload = {
            "policy": {
                "reproducibility": {
                    "random_seed": random_seed,
                    "multi_seed_rooms": ["livingroom"],
                    "multi_seed_candidate_seeds": [41, 42],
                }
            },
            "class_thresholds": {
                "0": threshold_0,
                "1": threshold_1,
            },
            "metrics": {
                "macro_f1": macro_f1,
                "validation_class_support": {"0": 138, "1": 1626},
                "holdout_class_support": {"0": 305, "1": 2635},
                "train_class_support_post_minority_sampling": {"0": 2138, "1": 12771},
                "minority_sampling": {
                    "post_sampling_prior_drift": {
                        "max_abs_drift_pp": 3.9662,
                    }
                },
                "unoccupied_downsample": {
                    "post_downsample_prior_drift": {
                        "max_abs_drift_pp": 0.0111,
                    }
                },
                "checkpoint_selection": {
                    "selection_mode": selection_mode,
                    "best_epoch": 1,
                    "best_floor_epoch": 1 if selection_mode == "no_regress_floor" else None,
                    "best_macro_epoch": 1,
                    "best_macro_f1": macro_f1,
                    "best_summary": {
                        "passes_no_regress_floor": selection_mode == "no_regress_floor",
                        "no_regress_macro_f1_floor": 0.4379,
                        "macro_f1": macro_f1,
                        "collapsed": collapsed,
                        "dominant_class_label": dominant_class_label,
                        "predicted_class_distribution": best_distribution,
                        "gate_aligned_score": 0.11 if selection_mode == "no_regress_floor" else 0.30,
                    },
                    "last_summary": {
                        "collapsed": collapsed,
                        "predicted_class_distribution": last_distribution,
                        "gate_aligned_score": -0.39 if selection_mode == "no_regress_floor" else 0.30,
                    },
                },
            },
        }
        calibrator_payload = {
            "intercept": calibrator_intercept,
            "coefficients": [1.0, -1.0, 2.0],
            "feature_names": ["top1_probability", "margin", "normalized_entropy"],
        }
        (self.models_dir / f"{self.room}_v{version}_decision_trace.json").write_text(
            json.dumps(trace_payload),
            encoding="utf-8",
        )
        (self.models_dir / f"{self.room}_v{version}_activity_confidence_calibrator.json").write_text(
            json.dumps(calibrator_payload),
            encoding="utf-8",
        )

    def test_build_livingroom_seed_forensic_summarizes_winner_and_loser_paths(self):
        payload = build_livingroom_seed_forensic(
            backend_dir=self.backend_dir,
            elder_id="HK0011_jessica",
            room_name=self.room,
            versions=[39, 40],
            winner_version=40,
        )

        self.assertEqual(payload["winner_version"], 40)
        self.assertEqual(payload["winner_seed"], 42)
        self.assertTrue(payload["policies_match_except_random_seed"])

        by_version = {entry["version"]: entry for entry in payload["versions"]}
        winner = by_version[40]
        loser = by_version[39]

        self.assertTrue(winner["reaches_no_regress_floor"])
        self.assertFalse(winner["fallback_selected"])
        self.assertEqual(winner["best_epoch_distribution"]["unoccupied"], 1564)

        self.assertFalse(loser["reaches_no_regress_floor"])
        self.assertTrue(loser["fallback_selected"])
        self.assertTrue(loser["active_heavy_best_epoch"])
        self.assertAlmostEqual(
            loser["comparison_to_winner"]["class_threshold_delta"]["1"],
            0.4355,
            places=4,
        )
        self.assertAlmostEqual(
            loser["comparison_to_winner"]["activity_confidence_intercept_delta"],
            8.51,
            places=2,
        )


if __name__ == "__main__":
    unittest.main()
