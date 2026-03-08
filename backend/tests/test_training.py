
import unittest
import numpy as np
import pandas as pd
import os
from unittest.mock import MagicMock, patch, ANY
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import tensorflow as tf

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.training import TrainingPipeline
from ml.policy_config import load_policy_from_env
from ml.exceptions import ModelTrainError

class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_platform = MagicMock()
        self.mock_registry = MagicMock()
        
        # Setup mock platform behavior with 3 sensors to pass validation
        self.mock_platform.sensor_columns = ['s1', 's2', 's3']
        self.mock_platform.label_encoders = {'room1': MagicMock()}
        self.mock_platform.label_encoders['room1'].classes_ = ['walking', 'sitting']
        self.mock_platform.label_encoders['room1'].transform.return_value = [0]
        self.mock_platform.scalers = {'room1': MagicMock()}
        # create_sequences returns (X, y)
        self.mock_platform.create_sequences.return_value = (np.zeros((10, 5, 3)), np.zeros(10))
        self.mock_platform.room_models = {}
        self.mock_platform.class_thresholds = {}
        self.mock_registry.save_model_artifacts.return_value = 1
        self.mock_registry.get_current_version_metadata.return_value = None
        
        self.pipeline = TrainingPipeline(self.mock_platform, self.mock_registry)

    def test_init_accepts_explicit_policy(self):
        policy = load_policy_from_env({"MINORITY_TARGET_SHARE": "0.22"})
        pipeline = TrainingPipeline(self.mock_platform, self.mock_registry, policy=policy)
        self.assertFalse(pipeline._use_env_policy)
        self.assertEqual(pipeline.policy.minority_sampling.target_share, 0.22)

    def test_policy_hash_is_stable(self):
        policy = load_policy_from_env({"MINORITY_TARGET_SHARE": "0.22", "THRESHOLD_FLOOR": "0.2"})
        h1 = self.pipeline._policy_hash(policy)
        h2 = self.pipeline._policy_hash(policy)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def test_policy_hash_changes_when_two_stage_core_policy_changes(self):
        p1 = load_policy_from_env({"ENABLE_TWO_STAGE_CORE_MODELING": "true"})
        p2 = load_policy_from_env({"ENABLE_TWO_STAGE_CORE_MODELING": "false"})
        self.assertNotEqual(self.pipeline._policy_hash(p1), self.pipeline._policy_hash(p2))

    def test_select_final_two_stage_gate_source_falls_back_to_single_stage_on_no_regress(self):
        self.mock_platform.label_encoders["Kitchen"] = MagicMock()
        self.mock_platform.label_encoders["Kitchen"].classes_ = np.array(
            ["kitchen_normal_use", "unoccupied"],
            dtype=object,
        )
        y_true = np.asarray(([0] * 50) + ([1] * 50), dtype=np.int32)
        single_probs = np.zeros((len(y_true), 2), dtype=np.float32)
        single_probs[np.arange(len(y_true)), y_true] = 1.0

        two_stage_pred = np.asarray(([0] * 70) + ([1] * 30), dtype=np.int32)
        two_stage_probs = np.zeros((len(y_true), 2), dtype=np.float32)
        two_stage_probs[np.arange(len(y_true)), two_stage_pred] = 1.0

        release_cfg = {
            "release_gates": {
                "no_regress": {
                    "max_drop_from_champion": 0.05,
                    "exempt_rooms": [],
                }
            }
        }
        champion_meta = {"metrics": {"macro_f1": 0.95}}
        with patch("ml.training.get_release_gates_config", return_value=release_cfg):
            selection = self.pipeline._select_final_two_stage_gate_source(
                room_name="Kitchen",
                y_true=y_true,
                single_stage_probs=single_probs,
                two_stage_probs=two_stage_probs,
                gate_mode="primary",
                champion_meta=champion_meta,
            )

        self.assertEqual(selection["selected_source"], "single_stage_fallback_no_regress")
        self.assertFalse(bool(selection["runtime_use_two_stage"]))
        self.assertTrue(bool(selection["single_stage"]["no_regress_ok"]))
        self.assertFalse(bool(selection["two_stage"]["no_regress_ok"]))

    def test_select_final_two_stage_gate_source_keeps_two_stage_when_single_collapses(self):
        self.mock_platform.label_encoders["LivingRoom"] = MagicMock()
        self.mock_platform.label_encoders["LivingRoom"].classes_ = np.array(
            ["livingroom_normal_use", "unoccupied"],
            dtype=object,
        )
        y_true = np.asarray(([0] * 50) + ([1] * 50), dtype=np.int32)
        single_probs = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(y_true), 1))
        two_stage_probs = np.zeros((len(y_true), 2), dtype=np.float32)
        two_stage_probs[np.arange(len(y_true)), y_true] = 1.0

        selection = self.pipeline._select_final_two_stage_gate_source(
            room_name="LivingRoom",
            y_true=y_true,
            single_stage_probs=single_probs,
            two_stage_probs=two_stage_probs,
            gate_mode="primary",
            champion_meta=None,
        )

        self.assertEqual(selection["selected_source"], "two_stage_primary")
        self.assertTrue(bool(selection["runtime_use_two_stage"]))
        self.assertTrue(bool(selection["single_stage"]["collapsed"]))
        self.assertFalse(bool(selection["two_stage"]["collapsed"]))

    def test_select_final_two_stage_gate_source_marks_both_collapsed_as_fail_closed(self):
        self.mock_platform.label_encoders["LivingRoom"] = MagicMock()
        self.mock_platform.label_encoders["LivingRoom"].classes_ = np.array(
            ["livingroom_normal_use", "unoccupied"],
            dtype=object,
        )
        y_true = np.asarray(([0] * 50) + ([1] * 50), dtype=np.int32)
        single_stage_probs = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(y_true), 1))
        two_stage_probs = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (len(y_true), 1))

        selection = self.pipeline._select_final_two_stage_gate_source(
            room_name="LivingRoom",
            y_true=y_true,
            single_stage_probs=single_stage_probs,
            two_stage_probs=two_stage_probs,
            gate_mode="primary",
            champion_meta=None,
        )

        self.assertFalse(bool(selection["runtime_use_two_stage"]))
        self.assertTrue(bool(selection.get("fail_closed")))
        self.assertFalse(bool(selection.get("selected_reliable", True)))
        self.assertEqual(selection.get("fail_closed_reason"), "no_reliable_primary_candidate")
        self.assertEqual(selection["selected_source"], "single_stage_fail_closed_unreliable")

    def test_should_shuffle_post_split_batches_matches_room_policy(self):
        self.assertTrue(self.pipeline._should_shuffle_post_split_batches("Entrance"))
        self.assertTrue(self.pipeline._should_shuffle_post_split_batches("Bedroom"))

    def test_build_beta6_parity_payload_maps_predictions_to_occupancy_trace(self):
        self.mock_platform.label_encoders["Bedroom"] = MagicMock()
        self.mock_platform.label_encoders["Bedroom"].classes_ = np.array(
            ["bedroom_normal_use", "sleep", "unoccupied", "unknown"],
            dtype=object,
        )
        y_pred_probs = np.asarray(
            [
                [0.05, 0.05, 0.80, 0.10],
                [0.80, 0.10, 0.05, 0.05],
                [0.05, 0.05, 0.10, 0.80],
                [0.10, 0.75, 0.10, 0.05],
            ],
            dtype=np.float32,
        )

        payload = self.pipeline._build_beta6_parity_payload(
            room_name="Bedroom",
            y_pred_probs=y_pred_probs,
        )

        self.assertEqual(
            payload["beta6_parity_trace"],
            [
                {"label": "unoccupied"},
                {"label": "occupied"},
                {"label": "unoccupied", "uncertainty_state": "unknown"},
                {"label": "occupied"},
            ],
        )
        self.assertEqual(payload["beta6_runtime_policy"], {"spike_suppression": True})
        self.assertEqual(payload["beta6_eval_policy"], {"spike_suppression": True})

    def test_build_beta6_parity_payload_caps_trace_length(self):
        self.mock_platform.label_encoders["Kitchen"] = MagicMock()
        self.mock_platform.label_encoders["Kitchen"].classes_ = np.array(
            ["kitchen_normal_use", "unoccupied"],
            dtype=object,
        )
        y_pred_probs = np.zeros((20, 2), dtype=np.float32)
        predicted_ids = np.asarray(([1] * 10) + ([0] * 10), dtype=np.int32)
        y_pred_probs[np.arange(len(predicted_ids)), predicted_ids] = 1.0

        payload = self.pipeline._build_beta6_parity_payload(
            room_name="Kitchen",
            y_pred_probs=y_pred_probs,
            max_steps=5,
        )

        self.assertEqual(len(payload["beta6_parity_trace"]), 5)

    def test_resolve_reported_accuracy_prefers_post_fit_summary_when_available(self):
        accuracy, source = self.pipeline._resolve_reported_accuracy(
            fit_history_accuracy=0.94,
            post_fit_train_summary={"available": True, "accuracy": 0.31},
        )
        self.assertEqual(source, "post_fit_train_predictions")
        self.assertAlmostEqual(accuracy, 0.31, places=6)

        accuracy, source = self.pipeline._resolve_reported_accuracy(
            fit_history_accuracy=0.94,
            post_fit_train_summary={"available": False, "accuracy": 0.31},
        )
        self.assertEqual(source, "fit_history")
        self.assertAlmostEqual(accuracy, 0.94, places=6)

    @patch.dict(os.environ, {"DEBUG_POST_FIT_AUDIT_ROOMS": "entrance,bedroom"}, clear=False)
    def test_is_prediction_audit_enabled_matches_normalized_room_name(self):
        self.assertTrue(self.pipeline._is_prediction_audit_enabled("Entrance"))
        self.assertTrue(self.pipeline._is_prediction_audit_enabled("bedroom"))
        self.assertFalse(self.pipeline._is_prediction_audit_enabled("Kitchen"))

    def test_summarize_prediction_audit_reports_argmax_and_probability_means(self):
        self.mock_platform.label_encoders["Entrance"] = MagicMock()
        self.mock_platform.label_encoders["Entrance"].classes_ = ["out", "unoccupied"]
        y_true = np.asarray([0, 0, 1, 1], dtype=np.int32)
        y_pred_probs = np.asarray(
            [
                [0.80, 0.20],
                [0.70, 0.30],
                [0.60, 0.40],
                [0.10, 0.90],
            ],
            dtype=np.float32,
        )

        summary = self.pipeline._summarize_prediction_audit(
            room_name="Entrance",
            y_true=y_true,
            y_pred_probs=y_pred_probs,
        )

        self.assertTrue(bool(summary.get("available")))
        self.assertAlmostEqual(float(summary.get("accuracy", 0.0)), 0.75, places=6)
        self.assertEqual(summary.get("predicted_class_distribution"), {"out": 3, "unoccupied": 1})
        means = summary.get("mean_probability_by_true_label") or {}
        self.assertAlmostEqual(
            float(means["out"]["mean_probability_by_label"]["out"]),
            0.75,
            places=6,
        )
        self.assertAlmostEqual(
            float(means["unoccupied"]["mean_probability_by_label"]["out"]),
            0.35,
            places=6,
        )

    def test_active_policy_uses_snapshot_when_present(self):
        p1 = load_policy_from_env({"MINORITY_TARGET_SHARE": "0.11"})
        p2 = load_policy_from_env({"MINORITY_TARGET_SHARE": "0.33"})
        self.pipeline.policy = p1
        self.pipeline._policy_snapshot = p2
        active = self.pipeline._active_policy()
        self.assertIs(active, p2)
        self.pipeline._policy_snapshot = None

    def test_timeline_native_sequence_weights_disabled_by_default(self):
        y_train = np.array([0, 0, 1, 1, 0, 0], dtype=np.int32)
        self.mock_platform.label_encoders["room1"].classes_ = ["unoccupied", "sleep"]
        weights, debug = self.pipeline._build_timeline_native_sequence_weights(
            room_name="room1",
            y_train=y_train,
        )
        self.assertEqual(len(weights), len(y_train))
        self.assertFalse(bool(debug.get("enabled")))
        self.assertEqual(debug.get("reason"), "feature_flag_or_room_disabled")

    def test_timeline_native_sequence_weights_enabled_for_bedroom(self):
        self.mock_platform.label_encoders["bedroom"] = MagicMock()
        self.mock_platform.label_encoders["bedroom"].classes_ = [
            "unoccupied",
            "sleep",
            "bedroom_normal_use",
        ]
        y_train = np.array([0, 0, 0, 1, 1, 2, 2, 0, 0], dtype=np.int32)
        with patch.dict(
            os.environ,
            {"ENABLE_TIMELINE_MULTITASK": "true", "TIMELINE_NATIVE_ROOMS": "bedroom,livingroom"},
            clear=False,
        ):
            weights, debug = self.pipeline._build_timeline_native_sequence_weights(
                room_name="bedroom",
                y_train=y_train,
            )
        self.assertTrue(bool(debug.get("enabled")))
        self.assertGreater(float(debug.get("transition_count", 0)), 0.0)
        self.assertGreater(float(np.max(weights)), 1.0)

    def test_timeline_activity_loss_weights_masks_unoccupied_and_unknown(self):
        self.mock_platform.label_encoders["bedroom"] = MagicMock()
        self.mock_platform.label_encoders["bedroom"].classes_ = ["unoccupied", "sleep", "unknown"]
        y_train = np.array([0, 1, 0, 2, 1], dtype=np.int32)
        global_weights = {0: 1.0, 1: 2.0, 2: 1.5}

        with patch.dict(
            os.environ,
            {
                "TIMELINE_ACTIVITY_UNOCCUPIED_WEIGHT_FLOOR": "0.05",
                "TIMELINE_ACTIVITY_REWEIGHT_MODE": "global",
            },
            clear=False,
        ):
            weights, class_weights, debug = self.pipeline._build_timeline_activity_loss_weights(
                room_name="bedroom",
                y_train=y_train,
                global_class_weight_dict=global_weights,
                timeline_native_weights=np.ones_like(y_train, dtype=np.float32),
                use_timeline_native_weighting=False,
                clinical_policy=self.pipeline._active_policy().clinical_priority,
            )

        self.assertEqual(debug["activity_class_weight_mode"], "global")
        self.assertAlmostEqual(debug["activity_loss_mask_coverage"], 3.0 / 5.0, places=6)
        self.assertAlmostEqual(weights[0], 0.05, places=6)   # unoccupied masked
        self.assertAlmostEqual(weights[3], 0.075, places=6)  # unknown masked
        self.assertAlmostEqual(weights[1], 2.0, places=6)    # occupied unchanged
        self.assertEqual(class_weights[1], 2.0)

    def test_timeline_activity_loss_weights_reweights_on_occupied_subset(self):
        self.mock_platform.label_encoders["bedroom"] = MagicMock()
        self.mock_platform.label_encoders["bedroom"].classes_ = ["unoccupied", "sleep", "reading"]
        y_train = np.array([0, 1, 1, 1, 1, 1, 1, 2, 2, 0], dtype=np.int32)
        global_weights = {0: 1.0, 1: 1.0, 2: 1.0}

        with patch.dict(
            os.environ,
            {
                "TIMELINE_ACTIVITY_UNOCCUPIED_WEIGHT_FLOOR": "0.0",
                "TIMELINE_ACTIVITY_REWEIGHT_MODE": "occupied_only",
            },
            clear=False,
        ):
            weights, class_weights, debug = self.pipeline._build_timeline_activity_loss_weights(
                room_name="bedroom",
                y_train=y_train,
                global_class_weight_dict=global_weights,
                timeline_native_weights=np.ones_like(y_train, dtype=np.float32),
                use_timeline_native_weighting=False,
                clinical_policy=self.pipeline._active_policy().clinical_priority,
            )

        self.assertEqual(debug["activity_class_weight_mode"], "occupied_only")
        self.assertEqual(debug["activity_class_weight_support"], 8)
        self.assertGreater(class_weights[2], class_weights[1])
        self.assertAlmostEqual(weights[0], 0.0, places=6)  # masked unoccupied gets floor=0.0

    def test_timeline_activity_loss_weights_fallbacks_to_global_when_occupied_support_small(self):
        self.mock_platform.label_encoders["bedroom"] = MagicMock()
        self.mock_platform.label_encoders["bedroom"].classes_ = ["unoccupied", "sleep"]
        y_train = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        global_weights = {0: 1.0, 1: 2.0}

        with patch.dict(
            os.environ,
            {
                "TIMELINE_ACTIVITY_UNOCCUPIED_WEIGHT_FLOOR": "0.0",
                "TIMELINE_ACTIVITY_REWEIGHT_MODE": "occupied_only",
            },
            clear=False,
        ):
            weights, class_weights, debug = self.pipeline._build_timeline_activity_loss_weights(
                room_name="bedroom",
                y_train=y_train,
                global_class_weight_dict=global_weights,
                timeline_native_weights=np.ones_like(y_train, dtype=np.float32),
                use_timeline_native_weighting=False,
                clinical_policy=self.pipeline._active_policy().clinical_priority,
            )

        self.assertEqual(debug["activity_class_weight_mode"], "global")
        self.assertEqual(debug["activity_class_weight_support"], 2)
        self.assertIn("activity_class_weight_fallback_reason", debug)
        self.assertEqual(class_weights[1], 2.0)
        self.assertAlmostEqual(weights[0], 0.0, places=6)

    @patch("ml.training.precision_recall_curve")
    def test_calibrate_two_stage_stage_a_threshold_uses_stage_a_bounds(self, mock_pr_curve):
        n = 80
        occupied_probs = np.linspace(0.55, 0.16, 40, dtype=np.float32)
        unoccupied_probs = np.linspace(0.33, 0.01, 40, dtype=np.float32)
        p_occ = np.concatenate([occupied_probs, unoccupied_probs]).astype(np.float32)
        stage_a_probs = np.stack([1.0 - p_occ, p_occ], axis=1).astype(np.float32)

        stage_a_model = MagicMock()
        stage_a_model.predict.return_value = stage_a_probs
        X_calib = np.zeros((n, 5, 3), dtype=np.float32)
        y_calib = np.concatenate([np.ones(40, dtype=np.int32), np.zeros(40, dtype=np.int32)])

        # Force deterministic threshold selection (raw threshold=0.22).
        mock_pr_curve.return_value = (
            np.array([0.98, 0.90, 0.70], dtype=np.float32),
            np.array([0.30, 0.60, 1.00], dtype=np.float32),
            np.array([0.22, 0.12], dtype=np.float32),
        )
        two_stage_result = {
            "enabled": True,
            "stage_a_model": stage_a_model,
            "occupied_class_ids": [1],
        }
        with patch.dict(
            os.environ,
            {
                "THRESHOLD_FLOOR": "0.35",
                "TWO_STAGE_CORE_STAGE_A_TARGET_PRECISION": "0.95",
                "TWO_STAGE_CORE_STAGE_A_RECALL_FLOOR": "0.20",
                "TWO_STAGE_CORE_STAGE_A_THRESHOLD_MIN": "0.00",
                "TWO_STAGE_CORE_STAGE_A_THRESHOLD_MAX": "0.95",
                "TWO_STAGE_CORE_STAGE_A_MIN_PRED_OCCUPIED_RATIO": "0.00",
                "TWO_STAGE_CORE_STAGE_A_MIN_PRED_OCCUPIED_ABS": "0.00",
            },
            clear=False,
        ):
            result = self.pipeline._calibrate_two_stage_stage_a_threshold(
                room_name="room1",
                two_stage_result=two_stage_result,
                calibration_data=(X_calib, y_calib),
            )

        self.assertEqual(result.get("source"), "calibration")
        self.assertAlmostEqual(float(result.get("threshold", 0.0)), 0.22, places=6)
        self.assertLess(float(result.get("threshold", 0.0)), 0.35)
        self.assertEqual((result.get("threshold_bounds") or {}).get("min"), 0.0)
        self.assertEqual((result.get("threshold_bounds") or {}).get("max"), 0.95)
        self.assertFalse(bool(result.get("predicted_occupied_floor_adjusted", False)))

    @patch("ml.training.precision_recall_curve")
    def test_calibrate_two_stage_stage_a_threshold_applies_predicted_occupied_floor(self, mock_pr_curve):
        n = 80
        occupied_probs = np.linspace(0.95, 0.20, 40, dtype=np.float32)
        unoccupied_probs = np.linspace(0.80, 0.05, 40, dtype=np.float32)
        p_occ = np.concatenate([occupied_probs, unoccupied_probs]).astype(np.float32)
        stage_a_probs = np.stack([1.0 - p_occ, p_occ], axis=1).astype(np.float32)

        stage_a_model = MagicMock()
        stage_a_model.predict.return_value = stage_a_probs
        X_calib = np.zeros((n, 5, 3), dtype=np.float32)
        y_calib = np.concatenate([np.ones(40, dtype=np.int32), np.zeros(40, dtype=np.int32)])

        # Force high raw threshold selection (0.90) so occupancy-rate guardrail must adjust downward.
        mock_pr_curve.return_value = (
            np.array([1.00, 0.95, 0.70], dtype=np.float32),
            np.array([0.20, 0.50, 1.00], dtype=np.float32),
            np.array([0.90, 0.40], dtype=np.float32),
        )
        two_stage_result = {
            "enabled": True,
            "stage_a_model": stage_a_model,
            "occupied_class_ids": [1],
        }
        with patch.dict(
            os.environ,
            {
                "TWO_STAGE_CORE_STAGE_A_TARGET_PRECISION": "0.99",
                "TWO_STAGE_CORE_STAGE_A_RECALL_FLOOR": "0.20",
                "TWO_STAGE_CORE_STAGE_A_THRESHOLD_MIN": "0.00",
                "TWO_STAGE_CORE_STAGE_A_THRESHOLD_MAX": "0.95",
                "TWO_STAGE_CORE_STAGE_A_MIN_PRED_OCCUPIED_RATIO": "1.00",
                "TWO_STAGE_CORE_STAGE_A_MIN_PRED_OCCUPIED_ABS": "0.80",
            },
            clear=False,
        ):
            result = self.pipeline._calibrate_two_stage_stage_a_threshold(
                room_name="room1",
                two_stage_result=two_stage_result,
                calibration_data=(X_calib, y_calib),
            )

        self.assertEqual(result.get("source"), "calibration")
        self.assertTrue(bool(result.get("predicted_occupied_floor_adjusted", False)))
        self.assertIn("pred_occ_floor", str(result.get("status", "")))
        self.assertLess(float(result.get("threshold", 1.0)), 0.90)
        self.assertGreaterEqual(
            float(result.get("predicted_occupied_rate", 0.0)),
            float(result.get("min_predicted_occupied_rate", 0.0)) - 1e-6,
        )

    def test_summarize_gate_aligned_validation_detects_collapse(self):
        self.mock_platform.label_encoders["room1"].classes_ = np.array(["unoccupied", "sleep"], dtype=object)
        y_true = np.asarray(([0] * 40) + ([1] * 40), dtype=np.int32)
        y_pred_probs = np.tile(np.asarray([[0.99, 0.01]], dtype=np.float32), (len(y_true), 1))
        with patch.object(self.pipeline, "_resolve_critical_labels", return_value=["sleep"]):
            summary = self.pipeline._summarize_gate_aligned_validation(
                room_name="room1",
                y_true=y_true,
                y_pred_probs=y_pred_probs,
            )
        self.assertTrue(bool(summary.get("collapsed")))
        self.assertAlmostEqual(float(summary.get("dominant_class_share", 0.0)), 1.0, places=6)
        self.assertEqual(str(summary.get("dominant_class_label")), "unoccupied")
        self.assertLess(
            float(summary.get("gate_aligned_score", 1.0)),
            float(summary.get("macro_f1", 0.0)) + 1e-8,
        )

    @patch("ml.training.get_release_gates_config")
    def test_resolve_checkpoint_no_regress_floor_uses_release_policy(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "no_regress": {
                    "max_drop_from_champion": 0.05,
                    "exempt_rooms": [],
                }
            }
        }
        ctx = self.pipeline._resolve_no_regress_macro_f1_floor(
            room_name="Bedroom",
            champion_meta={"metrics": {"macro_f1": 0.82}},
        )
        self.assertTrue(bool(ctx.get("enabled")))
        self.assertAlmostEqual(float(ctx.get("target_macro_f1_floor", 0.0)), 0.77, places=6)
        self.assertFalse(bool(ctx.get("exempt")))
    def test_write_decision_trace_persists_latest_and_versioned(self):
        with TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            self.mock_registry.get_models_dir.return_value = models_dir
            payload = {"ok": True, "room": "room1"}
            out = self.pipeline._write_decision_trace(
                elder_id="elder1",
                room_name="room1",
                saved_version=3,
                payload=payload,
            )
            self.assertIsNotNone(out)
            versioned = models_dir / "room1_v3_decision_trace.json"
            latest = models_dir / "room1_decision_trace.json"
            self.assertTrue(versioned.exists())
            self.assertTrue(latest.exists())

    @patch('ml.training.TrainingPipeline.augment_training_data')
    @patch('ml.training.build_transformer_model')
    @patch('ml.training.get_room_config')
    def test_train_room_success(self, mock_get_config, mock_build_model, mock_augment):
        """Test successful training flow."""
        # Mock room config
        mock_config = MagicMock()
        mock_config.get_sequence_window.return_value = 60
        mock_config.get_data_interval.return_value = 10
        mock_get_config.return_value = mock_config
        
        # Mock augment to return inputs as-is (or whatever)
        # It takes X_seq, y_seq as args 3 and 4 (0-indexed) or similar
        # def augment_training_data(self, room, elder, ts, X, y, ...)
        # We can just make it return (X, y) provided in processed_df setup
        # But here processed_df creates sequences via platform.create_sequences
        
        # We will make augment return dummy sequences
        mock_augment.return_value = (np.zeros((10, 5, 3)), np.zeros(10))
        
        # Mock model build
        mock_model = MagicMock()
        mock_model.fit.return_value.history = {'accuracy': [0.95]}
        mock_build_model.return_value = mock_model
        
        # Create dummy processed data
        processed_df = pd.DataFrame({
            's1': np.random.randn(100),
            's2': np.random.randn(100),
            's3': np.random.randn(100),
            'activity_encoded': np.random.randint(0, 2, 100),
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='10s')
        })
        
        # Run training
        metrics = self.pipeline.train_room(
            room_name='room1',
            processed_df=processed_df,
            seq_length=5,
            elder_id='elder1'
        )
        
        # Verifications
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics['room'], 'room1')
        self.assertEqual(metrics['accuracy'], 0.95)
        self.assertIn("policy_hash", metrics)
        
        # Verify model was compiled and fitted
        mock_model.compile.assert_called_once()
        mock_model.fit.assert_called_once()

        # Calibrated/default thresholds are applied in-memory immediately.
        self.assertTrue(hasattr(self.mock_platform, 'class_thresholds'))
        if metrics.get("promoted_to_latest", False):
            self.assertIn('room1', self.mock_platform.class_thresholds)
            self.assertIn('0', self.mock_platform.class_thresholds['room1'])
            self.assertIn('1', self.mock_platform.class_thresholds['room1'])
        
        # Verify artifact saving
        # Verify artifact saving
        self.mock_registry.save_model_artifacts.assert_called_once()
        _, kwargs = self.mock_registry.save_model_artifacts.call_args
        self.assertEqual(kwargs["accuracy"], 0.95)
        self.assertIn("samples", kwargs)
        self.assertIn("class_thresholds", kwargs)
        self.assertIn("metrics", kwargs)
        self.assertIn("model_identity", kwargs)
        self.assertEqual(kwargs["promote_to_latest"], bool(metrics.get("gate_pass", False)))
        self.assertEqual(metrics["saved_version"], 1)
        self.assertFalse(metrics["promotion_deferred"])
        self.assertEqual(metrics["model_identity"]["family"], "per_resident_full_model")
        self.assertIsNone(metrics["model_identity"]["backbone_id"])

    @patch('ml.training.TrainingPipeline.augment_training_data')
    @patch('ml.training.build_transformer_model')
    @patch('ml.training.get_room_config')
    def test_train_room_enables_post_split_shuffle_for_entrance(self, mock_get_config, mock_build_model, mock_augment):
        mock_config = MagicMock()
        mock_config.get_sequence_window.return_value = 60
        mock_config.get_data_interval.return_value = 10
        mock_get_config.return_value = mock_config
        mock_augment.return_value = (np.zeros((10, 5, 3)), np.zeros(10, dtype=np.int32))

        self.mock_platform.label_encoders["Entrance"] = MagicMock()
        self.mock_platform.label_encoders["Entrance"].classes_ = ["out", "unoccupied"]
        self.mock_platform.scalers["Entrance"] = MagicMock()

        mock_model = MagicMock()
        mock_model.fit.return_value.history = {'accuracy': [0.95]}
        mock_model.predict.return_value = np.tile(np.asarray([[0.9, 0.1]], dtype=np.float32), (10, 1))
        mock_build_model.return_value = mock_model

        processed_df = pd.DataFrame({
            's1': np.random.randn(100),
            's2': np.random.randn(100),
            's3': np.random.randn(100),
            'activity_encoded': np.zeros(100, dtype=np.int32),
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='10s')
        })

        metrics = self.pipeline.train_room(
            room_name='Entrance',
            processed_df=processed_df,
            seq_length=5,
            elder_id='elder1'
        )

        self.assertTrue(metrics["train_batch_shuffle_enabled"])
        self.assertEqual(metrics["accuracy_source"], "post_fit_train_predictions")
        self.assertAlmostEqual(metrics["accuracy"], 1.0, places=6)
        self.assertIn("holdout_class_support", metrics)
        self.assertIn("holdout_min_class_support", metrics)
        self.assertEqual(mock_model.fit.call_args.kwargs["shuffle"], True)

    @patch("ml.training.build_transformer_model")
    def test_train_two_stage_core_models_enables_post_split_shuffle_for_bedroom(self, mock_build_model):
        self.mock_platform.label_encoders["Bedroom"] = MagicMock()
        self.mock_platform.label_encoders["Bedroom"].classes_ = [
            "sleep",
            "bedroom_normal_use",
            "unoccupied",
        ]

        stage_a_model = MagicMock()
        stage_b_model = MagicMock()
        mock_build_model.side_effect = [stage_a_model, stage_b_model]

        X_train = np.zeros((12, 5, 3), dtype=np.float32)
        y_train = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
        validation_data = (
            np.zeros((6, 5, 3), dtype=np.float32),
            np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int32),
        )

        result = self.pipeline._train_two_stage_core_models(
            room_name="Bedroom",
            seq_length=5,
            X_train=X_train,
            y_train=y_train,
            validation_data=validation_data,
            max_epochs=2,
        )

        self.assertTrue(bool(result.get("enabled")))
        self.assertEqual(stage_a_model.fit.call_args.kwargs["shuffle"], True)
        self.assertEqual(stage_b_model.fit.call_args.kwargs["shuffle"], True)

    @patch("ml.training.build_transformer_model")
    def test_train_two_stage_core_models_uses_typed_policy_over_conflicting_env(self, mock_build_model):
        self.mock_platform.label_encoders["Bathroom"] = MagicMock()
        self.mock_platform.label_encoders["Bathroom"].classes_ = [
            "bathroom_normal_use",
            "shower",
            "unoccupied",
        ]
        policy = load_policy_from_env(
            {
                "ENABLE_TWO_STAGE_CORE_MODELING": "true",
                "TWO_STAGE_CORE_ROOMS": "bathroom",
                "TWO_STAGE_CORE_GATE_MODE": "primary",
                "TWO_STAGE_CORE_STAGE_A_OCCUPIED_THRESHOLD": "0.42",
            }
        )
        pipeline = TrainingPipeline(self.mock_platform, self.mock_registry, policy=policy)

        stage_a_model = MagicMock()
        stage_b_model = MagicMock()
        mock_build_model.side_effect = [stage_a_model, stage_b_model]

        X_train = np.zeros((12, 5, 3), dtype=np.float32)
        y_train = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
        validation_data = (
            np.zeros((6, 5, 3), dtype=np.float32),
            np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int32),
        )

        with patch.dict(
            os.environ,
            {
                "ENABLE_TWO_STAGE_CORE_MODELING": "false",
                "TWO_STAGE_CORE_ROOMS": "",
                "TWO_STAGE_CORE_GATE_MODE": "shadow",
                "TWO_STAGE_CORE_STAGE_A_OCCUPIED_THRESHOLD": "0.91",
            },
            clear=False,
        ):
            result = pipeline._train_two_stage_core_models(
                room_name="Bathroom",
                seq_length=5,
                X_train=X_train,
                y_train=y_train,
                validation_data=validation_data,
                max_epochs=2,
            )

        self.assertTrue(bool(result.get("enabled")))
        self.assertEqual(result.get("gate_mode"), "primary")
        self.assertAlmostEqual(float(result.get("stage_a_occupied_threshold", 0.0)), 0.42, places=6)
        self.assertEqual(result.get("stage_a_threshold_source"), "policy_default")

    @patch.object(TrainingPipeline, "_evaluate_lane_b_event_gates")
    @patch.object(TrainingPipeline, "_write_event_first_shadow_artifact")
    @patch.object(TrainingPipeline, "_evaluate_event_first_shadow")
    @patch('ml.training.TrainingPipeline.augment_training_data')
    @patch('ml.training.build_transformer_model')
    @patch('ml.training.get_room_config')
    def test_train_room_blocks_promotion_when_lane_b_fails(
        self,
        mock_get_config,
        mock_build_model,
        mock_augment,
        mock_eval_shadow,
        mock_write_shadow,
        mock_lane_b_gates,
    ):
        mock_config = MagicMock()
        mock_config.get_sequence_window.return_value = 60
        mock_config.get_data_interval.return_value = 10
        mock_get_config.return_value = mock_config
        mock_augment.return_value = (np.zeros((10, 5, 3)), np.zeros(10))

        mock_model = MagicMock()
        mock_model.fit.return_value.history = {'accuracy': [0.95]}
        mock_build_model.return_value = mock_model
        mock_lane_b_gates.return_value = (
            False,
            ["lane_b_gate_failed:room1:critical_failure"],
            {"overall_status": "fail"},
        )
        mock_eval_shadow.return_value = {"enabled": False, "evaluated": False}
        mock_write_shadow.return_value = None

        processed_df = pd.DataFrame({
            's1': np.random.randn(100),
            's2': np.random.randn(100),
            's3': np.random.randn(100),
            'activity_encoded': np.random.randint(0, 2, 100),
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='10s')
        })

        metrics = self.pipeline.train_room(
            room_name='room1',
            processed_df=processed_df,
            seq_length=5,
            elder_id='elder1'
        )

        self.assertFalse(metrics["gate_pass"])
        self.assertIn("lane_b_gate_failed:room1:critical_failure", metrics["gate_reasons"])
        self.assertIn("lane_b_gate", metrics)
        self.assertFalse(metrics["promoted_to_latest"])

    @patch.object(TrainingPipeline, "_write_event_first_shadow_artifact")
    @patch.object(TrainingPipeline, "_evaluate_event_first_shadow")
    @patch('ml.training.TrainingPipeline.augment_training_data')
    @patch('ml.training.build_transformer_model')
    @patch('ml.training.get_room_config')
    def test_train_room_shadow_disabled_is_noop(
        self,
        mock_get_config,
        mock_build_model,
        mock_augment,
        mock_eval_shadow,
        mock_write_shadow,
    ):
        mock_config = MagicMock()
        mock_config.get_sequence_window.return_value = 60
        mock_config.get_data_interval.return_value = 10
        mock_get_config.return_value = mock_config
        mock_augment.return_value = (np.zeros((10, 5, 3)), np.zeros(10))

        mock_model = MagicMock()
        mock_model.fit.return_value.history = {'accuracy': [0.95]}
        mock_build_model.return_value = mock_model

        self.pipeline.policy.event_first.shadow = False
        self.pipeline._use_env_policy = False

        processed_df = pd.DataFrame({
            's1': np.random.randn(100),
            's2': np.random.randn(100),
            's3': np.random.randn(100),
            'activity_encoded': np.random.randint(0, 2, 100),
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='10s')
        })

        metrics = self.pipeline.train_room(
            room_name='room1',
            processed_df=processed_df,
            seq_length=5,
            elder_id='elder1'
        )

        self.assertIsNotNone(metrics)
        self.assertNotIn("event_first_shadow", metrics)
        mock_eval_shadow.assert_not_called()
        mock_write_shadow.assert_not_called()

    @patch.object(TrainingPipeline, "_write_event_first_shadow_artifact")
    @patch.object(TrainingPipeline, "_evaluate_event_first_shadow")
    @patch('ml.training.TrainingPipeline.augment_training_data')
    @patch('ml.training.build_transformer_model')
    @patch('ml.training.get_room_config')
    def test_train_room_shadow_enabled_writes_artifact(
        self,
        mock_get_config,
        mock_build_model,
        mock_augment,
        mock_eval_shadow,
        mock_write_shadow,
    ):
        mock_config = MagicMock()
        mock_config.get_sequence_window.return_value = 60
        mock_config.get_data_interval.return_value = 10
        mock_get_config.return_value = mock_config
        mock_augment.return_value = (np.zeros((10, 5, 3)), np.zeros(10))

        mock_model = MagicMock()
        mock_model.fit.return_value.history = {'accuracy': [0.95]}
        mock_build_model.return_value = mock_model

        self.pipeline.policy.event_first.shadow = True
        self.pipeline._use_env_policy = False
        mock_eval_shadow.return_value = {
            "enabled": True,
            "evaluated": True,
            "summary": {"overall_status": "pass", "is_promotable": True, "critical_failures": []},
            "artifact": {"test": True},
        }
        mock_write_shadow.return_value = {"versioned": "/tmp/v.json", "latest": "/tmp/latest.json"}

        processed_df = pd.DataFrame({
            's1': np.random.randn(100),
            's2': np.random.randn(100),
            's3': np.random.randn(100),
            'activity_encoded': np.random.randint(0, 2, 100),
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='10s')
        })

        metrics = self.pipeline.train_room(
            room_name='room1',
            processed_df=processed_df,
            seq_length=5,
            elder_id='elder1'
        )

        self.assertIn("event_first_shadow", metrics)
        self.assertTrue(metrics["event_first_shadow"]["evaluated"])
        self.assertEqual(metrics["event_first_shadow"]["overall_status"], "pass")
        self.assertEqual(metrics["event_first_shadow"]["artifact_paths"]["versioned"], "/tmp/v.json")
        mock_eval_shadow.assert_called_once()
        mock_write_shadow.assert_called_once()

    @patch('ml.training.TrainingPipeline.augment_training_data')
    @patch('ml.training.build_transformer_model')
    @patch('ml.training.get_room_config')
    def test_train_room_handles_index_timestamps_with_duplicates(self, mock_get_config, mock_build_model, mock_augment):
        """Regression: duplicate resolution must work when timestamps are index-only."""
        mock_config = MagicMock()
        mock_config.get_sequence_window.return_value = 60
        mock_config.get_data_interval.return_value = 10
        mock_get_config.return_value = mock_config
        mock_augment.return_value = (np.zeros((10, 5, 3)), np.zeros(10))

        mock_model = MagicMock()
        mock_model.fit.return_value.history = {'accuracy': [0.90]}
        mock_build_model.return_value = mock_model

        base_ts = pd.date_range(start='2023-01-01', periods=99, freq='10s')
        # Duplicate the first timestamp to exercise duplicate resolver without a timestamp column.
        ts_with_dupe = pd.DatetimeIndex([base_ts[0], base_ts[0], *base_ts[1:]])
        processed_df = pd.DataFrame({
            's1': np.random.randn(100),
            's2': np.random.randn(100),
            's3': np.random.randn(100),
            'activity_encoded': np.random.randint(0, 2, 100),
        }, index=ts_with_dupe)

        metrics = self.pipeline.train_room(
            room_name='room1',
            processed_df=processed_df,
            seq_length=5,
            elder_id='elder1'
        )

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics['room'], 'room1')
        self.assertIn('accuracy', metrics)

    @patch.dict("os.environ", {"ENABLE_SHARED_BACKBONE_ADAPTERS": "true", "ACTIVE_SHARED_BACKBONE_ID": "backbone-vX"}, clear=False)
    @patch('ml.training.TrainingPipeline.augment_training_data')
    @patch('ml.training.build_transformer_model')
    @patch('ml.training.get_room_config')
    def test_train_room_shared_backbone_identity(self, mock_get_config, mock_build_model, mock_augment):
        mock_config = MagicMock()
        mock_config.get_sequence_window.return_value = 60
        mock_config.get_data_interval.return_value = 10
        mock_get_config.return_value = mock_config
        mock_augment.return_value = (np.zeros((10, 5, 3)), np.zeros(10))

        mock_model = MagicMock()
        mock_model.fit.return_value.history = {'accuracy': [0.95]}
        mock_build_model.return_value = mock_model

        processed_df = pd.DataFrame({
            's1': np.random.randn(100),
            's2': np.random.randn(100),
            's3': np.random.randn(100),
            'activity_encoded': np.random.randint(0, 2, 100),
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='10s')
        })

        metrics = self.pipeline.train_room(
            room_name='room1',
            processed_df=processed_df,
            seq_length=5,
            elder_id='elderA'
        )

        self.assertEqual(metrics["model_identity"]["family"], "shared_backbone_adapter")
        self.assertEqual(metrics["model_identity"]["backbone_id"], "backbone-vX")
        self.assertEqual(metrics["model_identity"]["adapter_id"], "elderA:room1")

    @patch.dict("os.environ", {
        "ENABLE_SHARED_BACKBONE_ADAPTERS": "true",
        "ENABLE_ADAPTER_ONLY_TRAINING": "true",
        "ACTIVE_SHARED_BACKBONE_ID": "",
    }, clear=False)
    @patch('ml.training.TrainingPipeline.augment_training_data')
    @patch('ml.training.build_transformer_model')
    @patch('ml.training.get_room_config')
    def test_train_room_shared_enabled_without_backbone_id_disables_adapter_only(
        self,
        mock_get_config,
        mock_build_model,
        mock_augment,
    ):
        mock_config = MagicMock()
        mock_config.get_sequence_window.return_value = 60
        mock_config.get_data_interval.return_value = 10
        mock_get_config.return_value = mock_config
        mock_augment.return_value = (np.zeros((10, 5, 3)), np.zeros(10))

        mock_model = MagicMock()
        mock_model.fit.return_value.history = {'accuracy': [0.95]}
        mock_build_model.return_value = mock_model

        processed_df = pd.DataFrame({
            's1': np.random.randn(100),
            's2': np.random.randn(100),
            's3': np.random.randn(100),
            'activity_encoded': np.random.randint(0, 2, 100),
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='10s')
        })

        with patch.object(self.pipeline, "_apply_shared_adapter_freeze", wraps=self.pipeline._apply_shared_adapter_freeze) as mock_freeze:
            metrics = self.pipeline.train_room(
                room_name='room1',
                processed_df=processed_df,
                seq_length=5,
                elder_id='elderA'
            )

        self.assertIsNotNone(metrics)
        self.assertFalse(metrics["adapter_training_only"])
        self.assertIsNone(metrics["shared_backbone_id"])
        self.assertEqual(metrics["shared_backbone_loaded_layers"], 0)
        self.assertEqual(metrics["learning_rate"], 1e-3)
        self.assertEqual(metrics["model_identity"]["family"], "per_resident_full_model")
        mock_freeze.assert_not_called()

    @patch('ml.training.fetch_golden_samples')
    @patch('ml.training.fetch_sensor_windows_batch')
    def test_augment_training_data(self, mock_fetch_batch, mock_fetch_golden):
        """Test augmentation with golden samples."""
        # Mock golden samples
        mock_fetch_golden.return_value = pd.DataFrame({
            'timestamp': [pd.Timestamp('2023-01-01 10:00:00')],
            'activity': ['walking'],
            'record_date': ['2023-01-01']
        })
        
        # Mock sensor batch (DB data) with 3 columns
        mock_fetch_batch.return_value = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01 09:59:00', periods=12, freq='10s'),
            's1': np.random.randn(12),
            's2': np.random.randn(12),
            's3': np.random.randn(12)
        })
        
        # Existing sequences
        X_seq = np.zeros((0, 5, 3))
        y_seq = np.zeros((0,))
        
        # Configure pipeline for augmentation test
        with patch('ml.training.get_room_config') as mock_conf:
             mock_conf.return_value.get_sequence_window.return_value = 60 # 60s window
             
             with patch('ml.training.calculate_sequence_length', return_value=5):
                 X_out, y_out = self.pipeline.augment_training_data(
                     'room1', 'elder1', [], X_seq, y_seq, window_seconds=60, interval_seconds=10
                 )
        
        # Should have added at least one sequence
        self.assertTrue(len(X_out) > 0)
        self.assertTrue(len(y_out) > 0)

    def test_train_room_insufficient_data(self):
        """Test training aborts when data is insufficient."""
        processed_df = pd.DataFrame({
            's1': [1, 2],
            's2': [1, 2],
            's3': [1, 2],
            'activity_encoded': [0, 0]
        })
        
        metrics = self.pipeline.train_room(
            'room1', processed_df, seq_length=5, elder_id='elder1'
        )
        
        self.assertIsNone(metrics)

    def test_train_room_fails_closed_on_invalid_activity_encoded(self):
        processed_df = pd.DataFrame({
            's1': np.random.randn(12),
            's2': np.random.randn(12),
            's3': np.random.randn(12),
            'activity_encoded': [0, 1, 0, 1, np.nan, 0, 1, 0, 1, 0, 1, 0],
            'timestamp': pd.date_range(start='2023-01-01', periods=12, freq='10s')
        })

        with patch.object(self.pipeline, "_log_training_history"):
            with self.assertRaises(ModelTrainError) as exc:
                self.pipeline.train_room(
                    room_name='room1',
                    processed_df=processed_df,
                    seq_length=5,
                    elder_id='elder1',
                )

        self.assertIn("activity_encoded has 1 invalid rows", str(exc.exception))

    def test_resolve_scheduled_threshold_uses_earliest_for_early_days(self):
        schedule = [
            {"min_days": 2, "max_days": 9, "min_value": 0.55},
            {"min_days": 10, "max_days": 21, "min_value": 0.65},
            {"min_days": 22, "max_days": None, "min_value": 0.75},
        ]
        threshold = self.pipeline._resolve_scheduled_threshold(schedule, training_days=0.5)
        self.assertEqual(threshold, 0.55)

    def test_resolve_scheduled_threshold_handles_gaps_by_last_reached_bucket(self):
        schedule = [
            {"min_days": 2, "max_days": 3, "min_value": 0.50},
            {"min_days": 5, "max_days": None, "min_value": 0.70},
        ]
        # Day 4 sits in a gap; should use latest reached bucket (0.50), not strictest.
        threshold = self.pipeline._resolve_scheduled_threshold(schedule, training_days=4.0)
        self.assertEqual(threshold, 0.50)

    def test_apply_replay_sampling_ratio(self):
        """Correction replay keeps corrected windows plus sampled uncorrected windows."""
        X_seq = np.random.randn(100, 5, 3).astype(np.float32)
        y_seq = np.random.randint(0, 2, size=100)
        seq_ts = np.array(pd.date_range(start='2023-01-01', periods=100, freq='10s'), dtype='datetime64[ns]')

        with patch.object(self.pipeline, '_match_corrected_sequence_indices', return_value=np.array([1, 3, 5])):
            replay_X, replay_y, replay_ts, stats = self.pipeline._apply_replay_sampling(
                elder_id='elder1',
                room_name='room1',
                X_seq=X_seq,
                y_seq=y_seq,
                seq_timestamps=seq_ts,
                replay_ratio=10,
            )

        self.assertEqual(stats['corrected_kept'], 3)
        self.assertEqual(stats['uncorrected_sampled'], 30)
        self.assertEqual(len(replay_X), 33)
        self.assertEqual(len(replay_y), 33)
        self.assertEqual(len(replay_ts), 33)

    @patch('ml.training.TrainingPipeline._apply_replay_sampling')
    @patch('ml.training.TrainingPipeline._build_model_for_room')
    @patch('ml.training.TrainingPipeline._get_fine_tuning_params')
    @patch('ml.training.TrainingPipeline.augment_training_data')
    @patch('ml.training.get_room_config')
    def test_train_room_correction_mode_uses_finetune_settings(
        self,
        mock_get_config,
        mock_augment,
        mock_get_fine_tuning_params,
        mock_build_model_for_room,
        mock_apply_replay_sampling,
    ):
        """Correction fine-tune mode uses warm-start-aware settings and replay stats."""
        mock_config = MagicMock()
        mock_config.get_sequence_window.return_value = 60
        mock_config.get_data_interval.return_value = 10
        mock_get_config.return_value = mock_config

        mock_augment.return_value = (np.zeros((10, 5, 3)), np.zeros(10))
        mock_get_fine_tuning_params.return_value = {
            "warm_start": True,
            "learning_rate": 1e-4,
            "epochs": 3,
            "patience": 1,
            "replay_enabled": True,
            "replay_ratio": 10,
        }
        replay_stats = {"total": 10, "corrected_kept": 2, "uncorrected_sampled": 8, "replay_ratio": 10}
        replay_stats["sampling_strategy"] = "random_stratified"
        mock_apply_replay_sampling.return_value = (
            np.zeros((10, 5, 3)),
            np.zeros(10),
            np.array(pd.date_range(start='2023-01-01', periods=10, freq='10s'), dtype='datetime64[ns]'),
            replay_stats,
        )

        mock_model = MagicMock()
        mock_model.fit.return_value.history = {'accuracy': [0.91, 0.92]}
        mock_build_model_for_room.return_value = (mock_model, True)

        processed_df = pd.DataFrame({
            's1': np.random.randn(100),
            's2': np.random.randn(100),
            's3': np.random.randn(100),
            'activity_encoded': np.random.randint(0, 2, 100),
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='10s')
        })

        metrics = self.pipeline.train_room(
            room_name='room1',
            processed_df=processed_df,
            seq_length=5,
            elder_id='elder1',
            training_mode='correction_fine_tune',
        )

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics['training_mode'], 'correction_fine_tune')
        self.assertTrue(metrics['warm_start_used'])
        self.assertEqual(metrics['replay'], replay_stats)

        _, compile_kwargs = mock_model.compile.call_args
        self.assertIsInstance(compile_kwargs['optimizer'], tf.keras.optimizers.Adam)
        self.assertAlmostEqual(float(compile_kwargs['optimizer'].learning_rate.numpy()), 1e-4, places=7)

        _, fit_kwargs = mock_model.fit.call_args
        self.assertEqual(fit_kwargs['epochs'], 3)

    def test_sample_uncorrected_stratified_balances_classes(self):
        """Stratified replay should include minority classes, not only majority label."""
        # Uncorrected pool labels: 0 dominates heavily, 1 is minority.
        y_seq = np.array([0] * 90 + [1] * 10)
        uncorrected_idx = np.arange(len(y_seq))
        sampled = self.pipeline._sample_uncorrected_stratified(
            y_seq=y_seq,
            uncorrected_idx=uncorrected_idx,
            target_uncorrected=20,
            rng=np.random.default_rng(42),
        )
        sampled_labels = y_seq[sampled]
        self.assertEqual(len(sampled), 20)
        self.assertIn(1, sampled_labels)

    @patch.dict(
        "os.environ",
        {
            "UNOCCUPIED_DOWNSAMPLE_MIN_SHARE": "0.20",
            "UNOCCUPIED_BOUNDARY_KEEP": "100",
            "UNOCCUPIED_MIN_RUN_LENGTH": "20",
            "UNOCCUPIED_DOWNSAMPLE_STRIDE_BY_ROOM": "bathroom:2",
            "UNOCCUPIED_BOUNDARY_KEEP_BY_ROOM": "bathroom:1",
            "UNOCCUPIED_MIN_RUN_LENGTH_BY_ROOM": "bathroom:4",
        },
        clear=False,
    )
    def test_downsample_easy_unoccupied_supports_room_overrides(self):
        # Configure bathroom encoder with explicit 'unoccupied' class.
        encoder = MagicMock()
        encoder.classes_ = np.array(["bathroom_normal_use", "unoccupied"], dtype=object)
        self.mock_platform.label_encoders["bathroom"] = encoder

        # Long unoccupied run should be downsampled only due to room-specific override.
        y_seq = np.array([1] * 12, dtype=np.int32)
        X_seq = np.random.randn(12, 5, 3).astype(np.float32)
        ts = np.array(pd.date_range("2026-01-01", periods=12, freq="10s"), dtype="datetime64[ns]")

        X_out, y_out, ts_out = self.pipeline._downsample_easy_unoccupied(
            X_seq=X_seq,
            y_seq=y_seq,
            seq_timestamps=ts,
            room_name="bathroom",
        )

        # Global config would skip/disable downsampling (min_run_length too large + boundary too large).
        # Room override forces min_run_length=4, boundary_keep=1, stride=2 => should remove interior points.
        self.assertLess(len(y_out), len(y_seq))
        self.assertEqual(len(X_out), len(y_out))
        self.assertEqual(len(ts_out), len(y_out))

    @patch.dict(
        "os.environ",
        {
            "ENABLE_MINORITY_CLASS_SAMPLING": "true",
            "MINORITY_TARGET_SHARE": "0.20",
            "MINORITY_MAX_MULTIPLIER": "3",
            "MINORITY_TARGET_SHARE_BY_ROOM": "bathroom:0.30",
            "MINORITY_MAX_MULTIPLIER_BY_ROOM": "",
        },
        clear=False,
    )
    def test_apply_minority_class_sampling_respects_room_target_share(self):
        y_train = np.array([0] * 50 + [1] * 5, dtype=np.int32)
        X_train = np.random.randn(len(y_train), 5, 3).astype(np.float32)

        X_out, y_out, stats = self.pipeline._apply_minority_class_sampling(
            X_train=X_train,
            y_train=y_train,
            room_name="bathroom",
        )

        self.assertTrue(stats["enabled"])
        self.assertGreaterEqual(len(y_out), len(y_train))
        counts_after = stats["class_counts_after"]
        minority_count = int(counts_after.get(1, 0))
        # room override target_share=0.30 on 55 rows => ceil(16.5)=17,
        # but max_multiplier=3 caps class-1 from 5 to 15.
        self.assertEqual(minority_count, 15)
        self.assertEqual(len(X_out), len(y_out))

    @patch("ml.training.TrainingPipeline._downsample_easy_unoccupied")
    @patch('ml.training.TrainingPipeline.augment_training_data')
    @patch('ml.training.build_transformer_model')
    @patch('ml.training.get_room_config')
    def test_train_room_downsamples_train_split_only(
        self,
        mock_get_config,
        mock_build_model,
        mock_augment,
        mock_downsample,
    ):
        mock_config = MagicMock()
        mock_config.get_sequence_window.return_value = 60
        mock_config.get_data_interval.return_value = 10
        mock_get_config.return_value = mock_config

        X_aug = np.zeros((200, 5, 3), dtype=np.float32)
        y_aug = np.array(([0] * 120) + ([1] * 80), dtype=np.int32)
        mock_augment.return_value = (X_aug, y_aug)
        mock_downsample.side_effect = lambda X, y, ts, *_args, **_kwargs: (X, y, ts)

        mock_model = MagicMock()
        mock_model.fit.return_value.history = {'accuracy': [0.95]}
        mock_build_model.return_value = mock_model

        processed_df = pd.DataFrame({
            's1': np.random.randn(400),
            's2': np.random.randn(400),
            's3': np.random.randn(400),
            'activity_encoded': np.random.randint(0, 2, 400),
            'timestamp': pd.date_range(start='2023-01-01', periods=400, freq='10s')
        })

        metrics = self.pipeline.train_room(
            room_name='room1',
            processed_df=processed_df,
            seq_length=5,
            elder_id='elder1'
        )

        self.assertIsNotNone(metrics)
        self.assertTrue(mock_downsample.called)
        # 200 augmented windows with 80/20 temporal split => train split has 160 windows.
        downsample_X = mock_downsample.call_args[0][0]
        self.assertEqual(len(downsample_X), 160)

    def test_apply_correction_layer_freeze_keeps_head_trainable(self):
        """Fine-tune freeze should lock cnn embedding and keep top block + head trainable."""
        mock_model = MagicMock()
        mock_model.layers = [
            MagicMock(name='input_layer'),
            MagicMock(name='cnn_embedding'),
            MagicMock(name='sinusoidal_positional_encoding'),
            MagicMock(name='transformer_block_0'),
            MagicMock(name='transformer_block_1'),
            MagicMock(name='global_average_pooling1d'),
            MagicMock(name='dropout'),
            MagicMock(name='dense'),
            MagicMock(name='dense_1'),
        ]
        for i, layer in enumerate(mock_model.layers):
            layer.name = [
                'input_layer',
                'cnn_embedding',
                'sinusoidal_positional_encoding',
                'transformer_block_0',
                'transformer_block_1',
                'global_average_pooling1d',
                'dropout',
                'dense',
                'dense_1',
            ][i]
            layer.trainable = True

        summary = self.pipeline._apply_correction_layer_freeze(mock_model, top_transformer_blocks=1)
        trainable = {layer.name for layer in mock_model.layers if layer.trainable}
        self.assertNotIn('cnn_embedding', trainable)
        self.assertIn('transformer_block_1', trainable)
        self.assertNotIn('transformer_block_0', trainable)
        self.assertIn('dense_1', trainable)
        self.assertEqual(summary['unfrozen_transformer_blocks'], [1])

    @patch('ml.training.get_release_gates_config')
    def test_evaluate_release_gate_normalizes_underscored_room_names(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "livingroom": {
                        "schedule": [
                            {"min_days": 2, "max_days": 9, "min_value": 0.5}
                        ]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            }
        }

        gate_pass, reasons = self.pipeline._evaluate_release_gate(
            room_name="living_room",
            candidate_metrics={
                "macro_f1": 0.2,
                "training_days": 3,
                "samples": 300,
                "validation_min_class_support": 30,
            },
            champion_meta=None,
        )

        self.assertFalse(gate_pass)
        self.assertTrue(any(r.startswith("room_threshold_failed:livingroom") for r in reasons))
        self.assertFalse(any(r.startswith("no_room_policy:") for r in reasons))

    @patch('ml.training.get_release_gates_config')
    def test_evaluate_release_gate_rejects_bootstrap_when_evidence_floor_not_met(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "room1": {
                        "schedule": [
                            {"min_days": 2, "max_days": 9, "min_value": 0.5}
                        ]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            }
        }
        gate_pass, reasons = self.pipeline._evaluate_release_gate(
            room_name="room1",
            candidate_metrics={"training_days": 3, "samples": 80},
            champion_meta=None,
        )
        self.assertFalse(gate_pass)
        self.assertTrue(any(r.startswith("insufficient_samples:room1") for r in reasons))
        self.assertTrue(any(r.startswith("bootstrap_small_dataset_no_holdout:room1") for r in reasons))

    @patch('ml.training.get_release_gates_config')
    def test_evaluate_release_gate_rejects_missing_macro_f1_for_non_bootstrap(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "room1": {
                        "schedule": [
                            {"min_days": 2, "max_days": 9, "min_value": 0.5}
                        ]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            }
        }
        gate_pass, reasons = self.pipeline._evaluate_release_gate(
            room_name="room1",
            candidate_metrics={"training_days": 3, "samples": 120},
            champion_meta=None,
        )
        self.assertFalse(gate_pass)
        self.assertTrue(any(r.startswith("candidate_metric_missing:room1:macro_f1") for r in reasons))

    @patch('ml.training.get_release_gates_config')
    def test_evaluate_release_gate_rejects_low_evidence_even_with_high_f1(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "room1": {
                        "schedule": [
                            {"min_days": 1, "max_days": None, "min_value": 0.5}
                        ]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            }
        }
        gate_pass, reasons = self.pipeline._evaluate_release_gate(
            room_name="room1",
            candidate_metrics={
                "macro_f1": 1.0,
                "training_days": 1.0,
                "samples": 50,
                "calibration_min_support": 10,
                "retained_sample_ratio": 0.05,
            },
            champion_meta=None,
        )
        self.assertFalse(gate_pass)
        self.assertTrue(any(r.startswith("insufficient_training_days:room1") for r in reasons))
        self.assertTrue(any(r.startswith("insufficient_samples:room1") for r in reasons))
        self.assertTrue(any(r.startswith("insufficient_retained_ratio:room1") for r in reasons))
        self.assertTrue(
            any(
                r.startswith("insufficient_calibration_support:room1")
                for r in self.pipeline._last_release_gate_watch_reasons
            )
        )

    @patch('ml.training.get_release_gates_config')
    def test_evaluate_release_gate_blocks_promotion_for_existing_champion_on_low_training_days(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "room1": {
                        "schedule": [
                            {"min_days": 1, "max_days": None, "min_value": 0.5}
                        ]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            }
        }
        champion_meta = {"version": 2, "metrics": {"macro_f1": 0.8}}
        gate_pass, reasons = self.pipeline._evaluate_release_gate(
            room_name="room1",
            candidate_metrics={"macro_f1": 0.9, "training_days": 3.0, "samples": 300},
            champion_meta=champion_meta,
        )
        self.assertFalse(gate_pass)
        self.assertTrue(any(r.startswith("promotion_ineligible_training_days:room1") for r in reasons))

    @patch('ml.training.get_release_gates_config')
    def test_evaluate_release_gate_blocks_on_label_recall_floor(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "bedroom": {
                        "schedule": [
                            {"min_days": 1, "max_days": None, "min_value": 0.2}
                        ]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            }
        }
        self.pipeline.policy.release_gate.min_recall_by_room_label = {"bedroom.unoccupied": 0.60}
        self.pipeline.policy.release_gate.min_recall_support = 30
        self.pipeline._policy_snapshot = self.pipeline.policy
        gate_pass, reasons = self.pipeline._evaluate_release_gate(
            room_name="Bedroom",
            candidate_metrics={
                "macro_f1": 0.9,
                "training_days": 3.0,
                "samples": 300,
                "per_label_recall": {"unoccupied": 0.12},
                "per_label_support": {"unoccupied": 120},
            },
            champion_meta=None,
        )
        self.assertFalse(gate_pass)
        self.assertTrue(any(r.startswith("label_recall_failed:bedroom:unoccupied") for r in reasons))

    @patch('ml.training.get_release_gates_config')
    def test_evaluate_release_gate_blocks_collapse_on_critical_labels(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "bedroom": {
                        "schedule": [
                            {"min_days": 1, "max_days": None, "min_value": 0.2}
                        ]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            }
        }
        self.pipeline.policy.release_gate.min_recall_by_room_label = {}
        self.pipeline._policy_snapshot = self.pipeline.policy
        gate_pass, reasons = self.pipeline._evaluate_release_gate(
            room_name="Bedroom",
            candidate_metrics={
                "macro_f1": 0.9,
                "training_days": 3.0,
                "samples": 300,
                "metric_source": "holdout_validation",
                "validation_min_class_support": 25,
                "per_label_recall": {
                    "unoccupied": 0.0,
                    "bedroom_normal_use": 0.12,
                    "sleep": 0.0,
                },
                "per_label_support": {
                    "unoccupied": 100,
                    "bedroom_normal_use": 90,
                    "sleep": 80,
                },
            },
            champion_meta=None,
        )
        self.assertFalse(gate_pass)
        self.assertTrue(any(r.startswith("critical_label_collapse:bedroom:unoccupied") for r in reasons))

    @patch('ml.training.get_release_gates_config')
    def test_evaluate_release_gate_pilot_low_support_critical_collapse_is_watch_only(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "entrance": {
                        "schedule": [
                            {"min_days": 1, "max_days": None, "min_value": 0.2}
                        ]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            }
        }
        self.pipeline.policy.release_gate.evidence_profile = "pilot_stage_a"
        self.pipeline._policy_snapshot = self.pipeline.policy
        gate_pass, reasons = self.pipeline._evaluate_release_gate(
            room_name="Entrance",
            candidate_metrics={
                "macro_f1": 0.9,
                "training_days": 8.0,
                "samples": 300,
                "metric_source": "holdout_validation",
                "validation_min_class_support": 3,
                "required_minority_support": 10,
                "per_label_recall": {
                    "out": 1.0,
                    "unoccupied": 0.0,
                },
                "per_label_support": {
                    "out": 3,
                    "unoccupied": 100,
                },
            },
            champion_meta=None,
        )
        self.assertTrue(gate_pass)
        self.assertFalse(any(r.startswith("critical_label_collapse:entrance:unoccupied") for r in reasons))
        self.assertTrue(
            any(
                r.startswith("critical_label_collapse:entrance:unoccupied")
                for r in self.pipeline._last_release_gate_watch_reasons
            )
        )

    @patch('ml.training.get_release_gates_config')
    def test_evaluate_release_gate_training_days_tolerance_avoids_false_promotion_block(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "room1": {
                        "schedule": [
                            {"min_days": 1, "max_days": None, "min_value": 0.2}
                        ]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.20, "exempt_rooms": []},
            }
        }
        self.pipeline.policy.promotion_eligibility.min_training_days_with_champion = 7.0
        self.pipeline.policy.release_gate.min_training_days = 7.0
        self.pipeline._policy_snapshot = self.pipeline.policy
        champion_meta = {"version": 2, "metrics": {"macro_f1": 0.8}}
        gate_pass, reasons = self.pipeline._evaluate_release_gate(
            room_name="room1",
            candidate_metrics={
                "macro_f1": 0.9,
                "training_days": 6.9997,
                "samples": 300,
            },
            champion_meta=champion_meta,
        )
        self.assertTrue(gate_pass)
        self.assertFalse(any(r.startswith("insufficient_training_days:room1") for r in reasons))
        self.assertFalse(any(r.startswith("promotion_ineligible_training_days:room1") for r in reasons))

    @patch('ml.training.get_release_gates_config')
    def test_evaluate_release_gate_blocks_missing_critical_validation_support(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "bedroom": {
                        "schedule": [
                            {"min_days": 1, "max_days": None, "min_value": 0.2}
                        ]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            }
        }
        self.pipeline.policy.release_gate.min_recall_by_room_label = {}
        self.pipeline._policy_snapshot = self.pipeline.policy
        gate_pass, reasons = self.pipeline._evaluate_release_gate(
            room_name="Bedroom",
            candidate_metrics={
                "macro_f1": 0.9,
                "training_days": 3.0,
                "samples": 300,
                "metric_source": "holdout_validation",
                "validation_min_class_support": 25,
                "per_label_recall": {
                    "bedroom_normal_use": 0.65,
                },
                "per_label_support": {
                    "bedroom_normal_use": 90,
                },
            },
            champion_meta=None,
        )
        self.assertTrue(gate_pass)
        self.assertTrue(
            any(
                r.startswith("critical_label_missing_validation:bedroom:unoccupied")
                for r in self.pipeline._last_release_gate_watch_reasons
            )
        )

    @patch('ml.training.get_release_gates_config')
    def test_evaluate_release_gate_pilot_profile_skips_low_support_room_threshold_block(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "livingroom": {
                        "schedule": [
                            {"min_days": 1, "max_days": None, "min_value": 0.5}
                        ]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            }
        }
        self.pipeline.policy.release_gate.evidence_profile = "pilot_stage_a"
        self.pipeline._policy_snapshot = self.pipeline.policy
        gate_pass, reasons = self.pipeline._evaluate_release_gate(
            room_name="LivingRoom",
            candidate_metrics={
                "macro_f1": 0.0,
                "training_days": 3.0,
                "samples": 300,
                "metric_source": "holdout_validation",
                "validation_min_class_support": 5,
                "per_label_recall": {"livingroom_normal_use": 0.0},
                "per_label_support": {"livingroom_normal_use": 5},
            },
            champion_meta=None,
        )
        self.assertTrue(gate_pass)
        self.assertFalse(any(r.startswith("insufficient_validation_support:livingroom") for r in reasons))
        self.assertFalse(any(r.startswith("room_threshold_failed:livingroom") for r in reasons))

    @patch('ml.training.get_release_gates_config')
    def test_evaluate_release_gate_low_support_reasons_go_to_watch_list(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "livingroom": {
                        "schedule": [
                            {"min_days": 1, "max_days": None, "min_value": 0.5}
                        ]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            }
        }
        self.pipeline.policy.release_gate.evidence_profile = "production"
        self.pipeline._policy_snapshot = self.pipeline.policy
        gate_pass, reasons = self.pipeline._evaluate_release_gate(
            room_name="LivingRoom",
            candidate_metrics={
                "macro_f1": 0.0,
                "training_days": 3.0,
                "samples": 300,
                "metric_source": "holdout_validation",
                "validation_min_class_support": 5,
                "per_label_recall": {"livingroom_normal_use": 0.0},
                "per_label_support": {"livingroom_normal_use": 5},
            },
            champion_meta=None,
        )
        self.assertTrue(gate_pass)
        self.assertFalse(any(r.startswith("insufficient_validation_support:livingroom") for r in reasons))
        self.assertFalse(any(r.startswith("room_threshold_failed:livingroom") for r in reasons))
        self.assertTrue(
            any(
                r.startswith("insufficient_validation_support:livingroom")
                for r in self.pipeline._last_release_gate_watch_reasons
            )
        )
        self.assertTrue(
            any(
                r.startswith("room_threshold_not_evaluable:livingroom")
                for r in self.pipeline._last_release_gate_watch_reasons
            )
        )

    @patch('ml.training.get_release_gates_config')
    def test_evaluate_release_gate_blocks_fail_closed_two_stage_primary(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "livingroom": {
                        "schedule": [
                            {"min_days": 1, "max_days": None, "min_value": 0.5}
                        ]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            }
        }
        self.pipeline.policy.release_gate.evidence_profile = "production"
        self.pipeline._policy_snapshot = self.pipeline.policy
        gate_pass, reasons = self.pipeline._evaluate_release_gate(
            room_name="LivingRoom",
            candidate_metrics={
                "macro_f1": 0.0,
                "training_days": 3.0,
                "samples": 300,
                "metric_source": "holdout_validation_single_stage_fallback",
                "validation_min_class_support": 5,
                "per_label_recall": {"livingroom_normal_use": 0.0},
                "per_label_support": {"livingroom_normal_use": 5},
                "two_stage_core": {
                    "enabled": True,
                    "gate_mode": "primary",
                    "gate_source": "single_stage_fail_closed_unreliable",
                    "fail_closed": True,
                    "selected_reliable": False,
                    "validation_summary": {"collapsed": True},
                },
            },
            champion_meta=None,
        )

        self.assertFalse(gate_pass)
        self.assertTrue(
            any(
                r.startswith(
                    "two_stage_primary_unreliable:livingroom:single_stage_fail_closed_unreliable"
                )
                for r in reasons
            )
        )

    @patch('ml.training.get_release_gates_config')
    def test_evaluate_release_gate_pilot_profile_treats_missing_critical_label_as_non_evaluable(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "bedroom": {
                        "schedule": [
                            {"min_days": 1, "max_days": None, "min_value": 0.2}
                        ]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            }
        }
        self.pipeline.policy.release_gate.evidence_profile = "pilot_stage_b"
        self.pipeline._policy_snapshot = self.pipeline.policy
        gate_pass, reasons = self.pipeline._evaluate_release_gate(
            room_name="Bedroom",
            candidate_metrics={
                "macro_f1": 0.9,
                "training_days": 3.0,
                "samples": 300,
                "metric_source": "holdout_validation",
                "validation_min_class_support": 5,
                "per_label_recall": {"bedroom_normal_use": 0.65},
                "per_label_support": {"bedroom_normal_use": 90},
            },
            champion_meta=None,
        )
        self.assertTrue(gate_pass)
        self.assertFalse(any(r.startswith("critical_label_missing_validation:bedroom:unoccupied") for r in reasons))

    @patch('ml.training.get_release_gates_config')
    def test_evaluate_release_gate_treats_alias_equivalents_as_supported_for_critical_labels(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "bedroom": {
                        "schedule": [
                            {"min_days": 1, "max_days": None, "min_value": 0.2}
                        ]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            }
        }
        self.pipeline.policy.release_gate.min_recall_by_room_label = {}
        self.pipeline._policy_snapshot = self.pipeline.policy
        with patch.object(self.pipeline, "_resolve_critical_labels", return_value=["sleeping", "unoccupied"]):
            gate_pass, reasons = self.pipeline._evaluate_release_gate(
                room_name="Bedroom",
                candidate_metrics={
                    "macro_f1": 0.9,
                    "training_days": 3.0,
                    "samples": 300,
                    "metric_source": "holdout_validation",
                    "validation_min_class_support": 25,
                    "per_label_recall": {
                        "sleep": 0.70,
                        "unoccupied": 0.80,
                    },
                    "per_label_support": {
                        "sleep": 120,
                        "unoccupied": 200,
                    },
                },
                champion_meta=None,
            )
        self.assertTrue(gate_pass)
        self.assertFalse(any(r.startswith("critical_label_missing_validation:bedroom:sleeping") for r in reasons))
        self.assertFalse(any(r.startswith("critical_label_recall_missing:bedroom:sleeping") for r in reasons))

    @patch('ml.training.get_release_gates_config')
    def test_evaluate_release_gate_treats_alias_equivalents_for_min_recall_thresholds(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "bedroom": {
                        "schedule": [
                            {"min_days": 1, "max_days": None, "min_value": 0.2}
                        ]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            }
        }
        self.pipeline.policy.release_gate.min_recall_by_room_label = {"bedroom.sleeping": 0.60}
        self.pipeline._policy_snapshot = self.pipeline.policy
        with patch.object(self.pipeline, "_resolve_critical_labels", return_value=["unoccupied"]):
            gate_pass, reasons = self.pipeline._evaluate_release_gate(
                room_name="Bedroom",
                candidate_metrics={
                    "macro_f1": 0.9,
                    "training_days": 3.0,
                    "samples": 300,
                    "metric_source": "holdout_validation",
                    "validation_min_class_support": 25,
                    "per_label_recall": {
                        "sleep": 0.70,
                        "unoccupied": 0.80,
                    },
                    "per_label_support": {
                        "sleep": 120,
                        "unoccupied": 200,
                    },
                },
                champion_meta=None,
            )
        self.assertTrue(gate_pass)
        self.assertFalse(any(r.startswith("label_recall_missing:bedroom:sleeping") for r in reasons))
        self.assertFalse(any(r.startswith("label_recall_failed:bedroom:sleeping") for r in reasons))

    def test_compute_class_prior_drift_reports_max_abs_shift(self):
        drift = self.pipeline._compute_class_prior_drift(
            {
                "train_class_support_pre_sampling": {"0": 200, "1": 800},
                "minority_sampling": {
                    "class_counts_after": {"0": 200, "1": 800},
                },
                "validation_class_support": {"0": 700, "1": 300},
                "validation_min_class_support": 300,
                "required_minority_support": 5,
                "insufficient_validation_evidence": False,
            }
        )
        self.assertTrue(drift["available"])
        self.assertTrue(drift["evaluable"])
        self.assertEqual(drift.get("drift_source"), "pre_sampling_train_vs_validation")
        self.assertEqual(drift["max_drift_class"], "1")
        self.assertAlmostEqual(float(drift["max_abs_drift"]), 0.5, places=6)
        self.assertAlmostEqual(float(drift.get("sampled_max_abs_drift", 0.0)), 0.5, places=6)

    def test_compute_class_prior_drift_prefers_pre_sampling_counts_over_sampled(self):
        drift = self.pipeline._compute_class_prior_drift(
            {
                "train_class_support_pre_sampling": {"0": 800, "1": 200},
                "train_class_support_post_minority_sampling": {"0": 500, "1": 500},
                "validation_class_support": {"0": 500, "1": 500},
                "validation_min_class_support": 300,
                "required_minority_support": 5,
            }
        )
        self.assertTrue(drift["available"])
        self.assertEqual(drift.get("drift_source"), "pre_sampling_train_vs_validation")
        self.assertAlmostEqual(float(drift.get("max_abs_drift", 0.0)), 0.3, places=6)
        self.assertAlmostEqual(float(drift.get("sampled_max_abs_drift", 1.0)), 0.0, places=6)

    def test_compute_class_prior_drift_prefers_full_holdout_support_over_validation_slice(self):
        drift = self.pipeline._compute_class_prior_drift(
            {
                "train_class_support_pre_sampling": {"0": 169, "1": 296, "2": 534},
                "validation_class_support": {"0": 114, "1": 436, "2": 448},
                "holdout_class_support": {"0": 129, "1": 321, "2": 548},
                "validation_min_class_support": 114,
                "holdout_min_class_support": 129,
                "required_minority_support": 5,
                "insufficient_validation_evidence": False,
            }
        )
        self.assertTrue(drift["available"])
        self.assertTrue(drift["evaluable"])
        self.assertEqual(drift.get("drift_source"), "pre_sampling_train_vs_holdout")
        self.assertEqual(drift["max_drift_class"], "0")
        self.assertAlmostEqual(float(drift.get("max_abs_drift", 0.0)), 0.04, places=3)

    @patch.dict(
        os.environ,
        {
            "TEMPORAL_SPLIT_OPTIMIZE_DRIFT": "true",
            "TEMPORAL_SPLIT_DRIFT_DISTANCE_PENALTY": "0.0",
            "TEMPORAL_SPLIT_MAX_SHIFT_FRACTION": "0.20",
            "TEMPORAL_SPLIT_MIN_HOLDOUT_FRACTION": "0.10",
            "TEMPORAL_SPLIT_MIN_TRAIN_FRACTION": "0.60",
        },
        clear=False,
    )
    def test_select_temporal_split_can_shift_to_reduce_prior_drift(self):
        y_seq = np.asarray(
            [0] * 150 + [1] * 250 + [2] * 400 + [0] * 20 + [1] * 140 + [2] * 40,
            dtype=np.int32,
        )
        def _max_abs_drift(split_idx: int) -> float:
            train = y_seq[:split_idx]
            holdout = y_seq[split_idx:]
            values = []
            for class_id in (0, 1, 2):
                train_share = float(np.mean(train == class_id))
                holdout_share = float(np.mean(holdout == class_id))
                values.append(abs(train_share - holdout_share))
            return float(max(values))

        selected, debug = self.pipeline._select_temporal_split_index_with_support(
            y_seq=y_seq,
            default_split_idx=800,
            required_class_ids=[0, 1, 2],
            min_support=5,
        )
        self.assertNotEqual(int(selected), 800)
        self.assertTrue(bool(debug.get("drift_optimization_enabled")))
        self.assertIn("drift_objective", debug)
        self.assertLess(_max_abs_drift(int(selected)), _max_abs_drift(800))
        self.assertGreaterEqual(int(selected), int(debug.get("min_train_samples", 0)))
        self.assertGreaterEqual(int(1000 - selected), int(debug.get("min_holdout_samples", 0)))

    @patch.dict(os.environ, {"ENABLE_MINORITY_CLASS_SAMPLING": "false"}, clear=False)
    def test_apply_minority_sampling_records_counts_even_when_disabled(self):
        x = np.zeros((6, 5, 3), dtype=np.float32)
        y = np.asarray([0, 0, 1, 1, 1, 0], dtype=np.int32)
        x_out, y_out, stats = self.pipeline._apply_minority_class_sampling(
            X_train=x,
            y_train=y,
            room_name="room1",
        )
        self.assertEqual(x_out.shape, x.shape)
        np.testing.assert_array_equal(y_out, y)
        self.assertFalse(bool(stats.get("enabled", True)))
        self.assertEqual(stats.get("class_counts_before"), {0: 3, 1: 3})
        self.assertEqual(stats.get("class_counts_after"), {0: 3, 1: 3})

    def test_apply_minority_sampling_caps_post_sampling_prior_drift(self):
        x = np.zeros((100, 5, 3), dtype=np.float32)
        y = np.asarray(([0] * 90) + ([1] * 10), dtype=np.int32)

        with patch.object(
            self.pipeline,
            "_resolve_minority_sampling_config",
            return_value={
                "enabled": True,
                "target_share": 0.30,
                "max_multiplier": 4,
                "prior_drift_guard_enabled": True,
                "max_post_sampling_prior_drift": 0.03,
            },
        ):
            _, y_out, stats = self.pipeline._apply_minority_class_sampling(
                X_train=x,
                y_train=y,
                room_name="Entrance",
            )

        before_share = float(np.mean(y == 1))
        after_share = float(np.mean(y_out == 1))
        self.assertLessEqual(after_share - before_share, 0.03 + 1e-6)
        self.assertTrue(bool(stats.get("prior_drift_guard_applied")))

    def test_apply_minority_sampling_records_post_sampling_drift_summary(self):
        x = np.zeros((100, 5, 3), dtype=np.float32)
        y = np.asarray(([0] * 90) + ([1] * 10), dtype=np.int32)

        with patch.object(
            self.pipeline,
            "_resolve_minority_sampling_config",
            return_value={
                "enabled": True,
                "target_share": 0.30,
                "max_multiplier": 4,
                "prior_drift_guard_enabled": True,
                "max_post_sampling_prior_drift": 0.03,
            },
        ):
            _, _, stats = self.pipeline._apply_minority_class_sampling(
                X_train=x,
                y_train=y,
                room_name="Entrance",
            )

        drift = stats.get("post_sampling_prior_drift") or {}
        self.assertTrue(bool(drift.get("available")))
        self.assertLessEqual(float(drift.get("max_abs_drift", 1.0)), 0.03 + 1e-6)

    def test_apply_minority_sampling_uses_holdout_reference_for_prior_drift_cap(self):
        x = np.zeros((100, 5, 3), dtype=np.float32)
        y = np.asarray(([0] * 90) + ([1] * 10), dtype=np.int32)
        holdout_reference = np.asarray(([0] * 80) + ([1] * 20), dtype=np.int32)

        with patch.object(
            self.pipeline,
            "_resolve_minority_sampling_config",
            return_value={
                "enabled": True,
                "target_share": 0.30,
                "max_multiplier": 4,
                "prior_drift_guard_enabled": True,
                "max_post_sampling_prior_drift": 0.03,
            },
        ):
            _, y_out, stats = self.pipeline._apply_minority_class_sampling(
                X_train=x,
                y_train=y,
                room_name="Entrance",
                reference_labels=holdout_reference,
            )

        after_share = float(np.mean(y_out == 1))
        self.assertGreater(after_share, 0.13)
        self.assertLessEqual(after_share, 0.23 + 1e-6)
        self.assertEqual(stats.get("prior_drift_reference"), "holdout")

    def test_downsample_easy_unoccupied_caps_post_downsample_prior_drift(self):
        self.mock_platform.label_encoders["Entrance"] = MagicMock()
        self.mock_platform.label_encoders["Entrance"].classes_ = ["out", "unoccupied"]
        x = np.zeros((100, 5, 3), dtype=np.float32)
        y = np.asarray(([0] * 10) + ([1] * 90), dtype=np.int32)
        ts = np.arange("2025-12-01T00:00:00", "2025-12-01T00:16:40", dtype="datetime64[10s]")

        _, y_out, _ = self.pipeline._downsample_easy_unoccupied(
            X_seq=x,
            y_seq=y,
            seq_timestamps=ts,
            room_name="Entrance",
            resolved_cfg={
                "min_share": 0.08,
                "stride": 14,
                "boundary_keep": 0,
                "min_run_length": 24,
                "prior_drift_guard_enabled": True,
                "max_post_downsample_prior_drift": 0.03,
            },
        )

        before_share = float(np.mean(y == 1))
        after_share = float(np.mean(y_out == 1))
        self.assertLessEqual(before_share - after_share, 0.03 + 1e-6)
        debug = self.pipeline._last_unoccupied_downsample_debug
        self.assertTrue(bool(debug.get("prior_drift_guard_applied")))
        drift = debug.get("post_downsample_prior_drift") or {}
        self.assertLessEqual(float(drift.get("max_abs_drift", 1.0)), 0.03 + 1e-6)

    def test_downsample_easy_unoccupied_uses_holdout_reference_for_prior_drift_cap(self):
        self.mock_platform.label_encoders["Entrance"] = MagicMock()
        self.mock_platform.label_encoders["Entrance"].classes_ = ["out", "unoccupied"]
        x = np.zeros((100, 5, 3), dtype=np.float32)
        y = np.asarray(([0] * 10) + ([1] * 90), dtype=np.int32)
        holdout_reference = np.asarray(([0] * 20) + ([1] * 80), dtype=np.int32)
        ts = np.arange("2025-12-01T00:00:00", "2025-12-01T00:16:40", dtype="datetime64[10s]")

        _, y_out, _ = self.pipeline._downsample_easy_unoccupied(
            X_seq=x,
            y_seq=y,
            seq_timestamps=ts,
            room_name="Entrance",
            resolved_cfg={
                "min_share": 0.08,
                "stride": 14,
                "boundary_keep": 0,
                "min_run_length": 24,
                "prior_drift_guard_enabled": True,
                "max_post_downsample_prior_drift": 0.03,
            },
            reference_labels=holdout_reference,
        )

        after_share = float(np.mean(y_out == 0))
        self.assertGreater(after_share, 0.13)
        self.assertLessEqual(after_share, 0.23 + 1e-6)
        debug = self.pipeline._last_unoccupied_downsample_debug
        self.assertEqual(debug.get("prior_drift_reference"), "holdout")

    def test_select_multi_seed_candidate_prefers_noncollapsed_no_regress_result(self):
        candidates = [
            {
                "seed": 40,
                "summary": {
                    "collapsed": True,
                    "gate_aligned_score": -0.8,
                    "macro_f1": 0.08,
                },
                "no_regress_ok": False,
            },
            {
                "seed": 45,
                "summary": {
                    "collapsed": False,
                    "gate_aligned_score": 0.9,
                    "macro_f1": 0.62,
                },
                "no_regress_ok": True,
            },
        ]

        selected = self.pipeline._select_multi_seed_candidate(candidates)
        self.assertEqual(selected["seed"], 45)

    def test_select_multi_seed_candidate_prefers_gate_passing_candidate(self):
        candidates = [
            {
                "seed": 11,
                "summary": {
                    "collapsed": False,
                    "gate_pass": False,
                    "gate_aligned_score": 0.45,
                    "macro_f1": 0.95,
                },
                "no_regress_ok": True,
            },
            {
                "seed": 13,
                "summary": {
                    "collapsed": False,
                    "gate_pass": True,
                    "gate_aligned_score": 0.40,
                    "macro_f1": 0.40,
                },
                "no_regress_ok": True,
            },
        ]

        selected = self.pipeline._select_multi_seed_candidate(candidates)
        self.assertEqual(selected["seed"], 13)

    def test_train_room_with_multi_seed_panel_reuses_selected_candidate_metrics(self):
        self.pipeline._active_policy = MagicMock(
            return_value=SimpleNamespace(
                reproducibility=SimpleNamespace(
                    multi_seed_candidate_seeds=[11, 13],
                    random_seed=11,
                )
            )
        )
        first_candidate = MagicMock()
        first_candidate.train_room.return_value = {
            "macro_f1": 0.95,
            "gate_pass": False,
            "gate_reasons": ["room_threshold_failed:livingroom"],
            "predicted_class_distribution": {
                "livingroom_normal_use": 90,
                "unoccupied": 10,
            },
            "saved_version": 101,
        }
        selected_candidate = MagicMock()
        selected_candidate.train_room.return_value = {
            "macro_f1": 0.40,
            "gate_pass": True,
            "gate_reasons": [],
            "predicted_class_distribution": {
                "livingroom_normal_use": 55,
                "unoccupied": 45,
            },
            "saved_version": 102,
        }
        unexpected_retrain = MagicMock()
        unexpected_retrain.train_room.return_value = {
            "macro_f1": 0.99,
            "gate_pass": True,
            "gate_reasons": [],
            "predicted_class_distribution": {
                "livingroom_normal_use": 50,
                "unoccupied": 50,
            },
            "saved_version": 999,
        }

        with patch.object(
            self.pipeline,
            "_spawn_candidate_pipeline",
            side_effect=[first_candidate, selected_candidate, unexpected_retrain],
        ) as spawn:
            result = self.pipeline._train_room_with_multi_seed_panel(
                room_name="LivingRoom",
                processed_df=pd.DataFrame(),
                seq_length=5,
                elder_id="elder-1",
            )

        self.assertEqual(spawn.call_count, 2)
        unexpected_retrain.train_room.assert_not_called()
        self.assertEqual(result["saved_version"], 102)
        self.assertEqual(result["seed_panel_debug"]["selected_seed"], 13)

    def test_spawn_candidate_pipeline_respects_runtime_seed_override_with_explicit_policy(self):
        policy = load_policy_from_env(
            {
                "TRAINING_RANDOM_SEED": "42",
                "MULTI_SEED_ROOMS": "entrance",
            }
        )
        pipeline = TrainingPipeline(self.mock_platform, self.mock_registry, policy=policy)

        with patch.dict(
            os.environ,
            {
                "TRAINING_RANDOM_SEED": "77",
                "TRAINING_MULTI_SEED_ACTIVE": "1",
            },
            clear=False,
        ):
            spawned = pipeline._spawn_candidate_pipeline()

        self.assertFalse(spawned._use_env_policy)
        self.assertEqual(spawned._active_policy().reproducibility.random_seed, 77)

    def test_multi_seed_enabled_for_livingroom_by_default(self):
        policy = load_policy_from_env({})
        pipeline = TrainingPipeline(self.mock_platform, self.mock_registry, policy=policy)

        self.assertTrue(pipeline._multi_seed_enabled_for_room("LivingRoom"))

    def test_resolve_training_strategy_marks_bedroom_factorized_primary(self):
        self.pipeline.policy = load_policy_from_env(
            {
                "FACTORIZED_PRIMARY_ROOMS": "bedroom",
            }
        )

        self.assertEqual(self.pipeline._resolve_training_strategy("Bedroom"), "factorized_primary")
        self.assertTrue(self.pipeline._two_stage_core_enabled_for_room("Bedroom"))

    def test_apply_transition_focus_sampling_boosts_bedroom_transition_windows(self):
        self.mock_platform.label_encoders["Bedroom"] = MagicMock()
        self.mock_platform.label_encoders["Bedroom"].classes_ = [
            "sleep",
            "bedroom_normal_use",
            "unoccupied",
        ]
        self.pipeline.policy = load_policy_from_env(
            {
                "FACTORIZED_PRIMARY_ROOMS": "bedroom",
                "TRANSITION_FOCUS_ROOM_LABELS": "bedroom:bedroom_normal_use",
                "TRANSITION_FOCUS_RADIUS_STEPS_BY_ROOM": "bedroom:1",
                "TRANSITION_FOCUS_MAX_MULTIPLIER_BY_ROOM": "bedroom:3",
            }
        )
        y = np.asarray([2, 2, 2, 0, 1, 1, 0, 0, 2, 2], dtype=np.int32)
        x = np.zeros((len(y), 5, 3), dtype=np.float32)

        x_out, y_out, stats = self.pipeline._apply_transition_focus_sampling(
            X_train=x,
            y_train=y,
            room_name="Bedroom",
        )

        self.assertTrue(bool(stats.get("enabled")))
        self.assertEqual(stats.get("focus_label"), "bedroom_normal_use")
        self.assertGreater(int(stats.get("transition_window_count", 0)), 0)
        self.assertGreater(int(stats.get("added_samples", 0)), 0)
        self.assertGreater(len(y_out), len(y))
        self.assertEqual(x_out.shape[0], len(y_out))

    def test_apply_transition_focus_sampling_caps_bedroom_post_sampling_prior_drift(self):
        self.mock_platform.label_encoders["Bedroom"] = MagicMock()
        self.mock_platform.label_encoders["Bedroom"].classes_ = [
            "sleep",
            "bedroom_normal_use",
            "unoccupied",
        ]
        self.pipeline.policy = load_policy_from_env(
            {
                "FACTORIZED_PRIMARY_ROOMS": "bedroom",
                "TRANSITION_FOCUS_ROOM_LABELS": "bedroom:bedroom_normal_use",
                "TRANSITION_FOCUS_RADIUS_STEPS_BY_ROOM": "bedroom:1",
                "TRANSITION_FOCUS_MAX_MULTIPLIER_BY_ROOM": "bedroom:4",
                "TRANSITION_FOCUS_MAX_POST_SAMPLING_PRIOR_DRIFT_BY_ROOM": "bedroom:0.08",
                "TRANSITION_FOCUS_PRIOR_DRIFT_GUARD_ROOMS": "bedroom",
            }
        )
        self.pipeline._use_env_policy = False
        y = np.asarray([2, 0, 1, 1, 1, 0, 2, 2], dtype=np.int32)
        x = np.zeros((len(y), 5, 3), dtype=np.float32)
        holdout_reference = np.asarray([0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)

        x_out, y_out, stats = self.pipeline._apply_transition_focus_sampling(
            X_train=x,
            y_train=y,
            room_name="Bedroom",
            reference_labels=holdout_reference,
        )

        self.assertTrue(bool(stats.get("enabled")))
        self.assertTrue(bool(stats.get("prior_drift_guard_applied")))
        self.assertEqual(stats.get("prior_drift_reference"), "holdout")
        drift = stats.get("post_sampling_prior_drift") or {}
        self.assertTrue(bool(drift.get("available")))
        self.assertLessEqual(float(drift.get("max_abs_drift", 1.0)), 0.08 + 1e-6)
        self.assertGreater(len(y_out), len(y))
        self.assertEqual(x_out.shape[0], len(y_out))

    def test_apply_transition_focus_sampling_skips_non_target_rooms(self):
        self.mock_platform.label_encoders["Kitchen"] = MagicMock()
        self.mock_platform.label_encoders["Kitchen"].classes_ = [
            "kitchen_normal_use",
            "unoccupied",
        ]
        self.pipeline.policy = load_policy_from_env(
            {
                "FACTORIZED_PRIMARY_ROOMS": "bedroom",
                "TRANSITION_FOCUS_ROOM_LABELS": "bedroom:bedroom_normal_use",
                "TRANSITION_FOCUS_RADIUS_STEPS_BY_ROOM": "bedroom:1",
                "TRANSITION_FOCUS_MAX_MULTIPLIER_BY_ROOM": "bedroom:3",
            }
        )
        y = np.asarray([0, 0, 1, 1], dtype=np.int32)
        x = np.zeros((len(y), 5, 3), dtype=np.float32)

        x_out, y_out, stats = self.pipeline._apply_transition_focus_sampling(
            X_train=x,
            y_train=y,
            room_name="Kitchen",
        )

        self.assertFalse(bool(stats.get("enabled")))
        self.assertEqual(stats.get("reason"), "room_not_configured")
        np.testing.assert_array_equal(y_out, y)
        self.assertEqual(x_out.shape, x.shape)

    @patch('ml.training.get_release_gates_config')
    @patch.dict(os.environ, {"RELEASE_GATE_MAX_CLASS_PRIOR_DRIFT": "0.10"}, clear=False)
    def test_evaluate_release_gate_blocks_on_excessive_class_prior_drift(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "room1": {
                        "schedule": [{"min_days": 1, "max_days": None, "min_value": 0.2}]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.2, "exempt_rooms": []},
            }
        }
        gate_pass, reasons = self.pipeline._evaluate_release_gate(
            room_name="room1",
            candidate_metrics={
                "macro_f1": 0.8,
                "training_days": 8.0,
                "samples": 2000,
                "metric_source": "holdout_validation",
                "validation_min_class_support": 100,
                "required_minority_support": 5,
                "minority_sampling": {
                    "class_counts_after": {"0": 200, "1": 800},
                },
                "validation_class_support": {"0": 700, "1": 300},
            },
            champion_meta=None,
        )
        self.assertFalse(gate_pass)
        self.assertTrue(any(r.startswith("class_prior_drift_failed:room1:") for r in reasons))

    @patch('ml.training.get_release_gates_config')
    @patch.dict(os.environ, {"RELEASE_GATE_MAX_CLASS_PRIOR_DRIFT": "0.10"}, clear=False)
    def test_evaluate_release_gate_prior_drift_low_support_is_watch_only(self, mock_get_policy):
        mock_get_policy.return_value = {
            "release_gates": {
                "rooms": {
                    "room1": {
                        "schedule": [{"min_days": 1, "max_days": None, "min_value": 0.2}]
                    }
                },
                "no_regress": {"max_drop_from_champion": 0.2, "exempt_rooms": []},
            }
        }
        gate_pass, reasons = self.pipeline._evaluate_release_gate(
            room_name="room1",
            candidate_metrics={
                "macro_f1": 0.8,
                "training_days": 8.0,
                "samples": 2000,
                "metric_source": "holdout_validation",
                "validation_min_class_support": 2,
                "required_minority_support": 5,
                "insufficient_validation_evidence": True,
                "minority_sampling": {
                    "class_counts_after": {"0": 200, "1": 800},
                },
                "validation_class_support": {"0": 700, "1": 300},
            },
            champion_meta=None,
        )
        self.assertTrue(gate_pass)
        self.assertFalse(any(r.startswith("class_prior_drift_failed:room1:") for r in reasons))
        self.assertTrue(
            any(
                r.startswith("class_prior_drift_not_evaluable:room1:")
                for r in self.pipeline._last_release_gate_watch_reasons
            )
        )

    def test_evaluate_lane_b_event_gates_blocks_bedroom_critical_collapse(self):
        gate_pass, reasons, report = self.pipeline._evaluate_lane_b_event_gates(
            room_name="Bedroom",
            candidate_metrics={
                "per_label_recall": {"sleep": 0.0},
                "per_label_support": {"sleep": 120, "unoccupied": 200},
            },
        )
        self.assertFalse(gate_pass)
        self.assertTrue(any(r.startswith("lane_b_gate_failed:bedroom:") for r in reasons))
        self.assertEqual(report.get("overall_status"), "fail")

    @patch.dict(
        os.environ,
        {
            "RELEASE_GATE_BOOTSTRAP_ENABLED": "true",
            "RELEASE_GATE_BOOTSTRAP_MAX_TRAINING_DAYS": "14",
        },
        clear=False,
    )
    def test_evaluate_release_gate_bootstrap_keeps_prior_drift_strict(self):
        self.pipeline.policy.release_gate.evidence_profile = "pilot_stage_a"
        with patch("ml.training.get_release_gates_config") as mock_get_policy:
            mock_get_policy.return_value = {
                "release_gates": {
                    "rooms": {
                        "room1": {
                            "schedule": [{"min_days": 1, "max_days": None, "min_value": 0.2}]
                        }
                    },
                    "no_regress": {"max_drop_from_champion": 0.2, "exempt_rooms": []},
                }
            }
            gate_pass, reasons = self.pipeline._evaluate_release_gate(
                room_name="room1",
                candidate_metrics={
                    "macro_f1": 0.8,
                    "training_days": 6.0,
                    "samples": 2000,
                    "metric_source": "holdout_validation",
                    "validation_min_class_support": 100,
                    "required_minority_support": 5,
                    "minority_sampling": {
                        "class_counts_after": {"0": 200, "1": 800},
                    },
                    "validation_class_support": {"0": 700, "1": 300},
                },
                champion_meta=None,
            )
        self.assertFalse(gate_pass)
        self.assertTrue(any(r.startswith("class_prior_drift_failed:room1:") for r in reasons))

    @patch.dict(
        os.environ,
        {
            "RELEASE_GATE_BOOTSTRAP_ENABLED": "true",
            "RELEASE_GATE_BOOTSTRAP_MAX_TRAINING_DAYS": "14",
        },
        clear=False,
    )
    def test_evaluate_lane_b_event_gates_bootstrap_soft_fails_non_collapse(self):
        pipeline = TrainingPipeline(
            self.mock_platform,
            self.mock_registry,
            policy=load_policy_from_env({"RELEASE_GATE_EVIDENCE_PROFILE": "pilot_stage_a"}),
        )
        gate_pass, reasons, report = pipeline._evaluate_lane_b_event_gates(
            room_name="Bedroom",
            candidate_metrics={
                "training_days": 6.0,
                "per_label_recall": {"sleep": 0.30},
                "per_label_support": {"sleep": 120, "unoccupied": 200},
            },
        )
        self.assertTrue(gate_pass)
        self.assertEqual(reasons, [])
        self.assertEqual(report.get("enforcement"), "watch_only_bootstrap")
        self.assertIn("recall_sleep_duration", report.get("soft_failed_critical_failures", []))

    @patch.dict(
        os.environ,
        {
            "RELEASE_GATE_BOOTSTRAP_ENABLED": "true",
            "RELEASE_GATE_BOOTSTRAP_MAX_TRAINING_DAYS": "14",
        },
        clear=False,
    )
    def test_evaluate_lane_b_event_gates_bootstrap_still_blocks_collapse(self):
        pipeline = TrainingPipeline(
            self.mock_platform,
            self.mock_registry,
            policy=load_policy_from_env({"RELEASE_GATE_EVIDENCE_PROFILE": "pilot_stage_a"}),
        )
        gate_pass, reasons, report = pipeline._evaluate_lane_b_event_gates(
            room_name="Bedroom",
            candidate_metrics={
                "training_days": 6.0,
                "per_label_recall": {"sleep": 0.0},
                "per_label_support": {"sleep": 120, "unoccupied": 200},
            },
        )
        self.assertFalse(gate_pass)
        self.assertTrue(any("collapse_sleep_duration" in r for r in reasons))
        self.assertEqual(report.get("enforcement"), "hard_bootstrap_collapse_guard")

    def test_evaluate_lane_b_event_gates_short_window_pilot_uses_min_support_10(self):
        pipeline = TrainingPipeline(
            self.mock_platform,
            self.mock_registry,
            policy=load_policy_from_env({"RELEASE_GATE_EVIDENCE_PROFILE": "pilot_stage_b"}),
        )
        gate_pass, reasons, report = pipeline._evaluate_lane_b_event_gates(
            room_name="Bedroom",
            candidate_metrics={
                "training_days": 6.0,
                "per_label_recall": {"sleep": 0.95},
                "per_label_support": {"sleep": 12, "unoccupied": 200},
            },
        )
        self.assertTrue(gate_pass)
        self.assertEqual(reasons, [])
        self.assertEqual(report.get("configured_min_support_for_tier_gates"), 10)
        results_by_name = {
            str(row.get("gate_name")): row
            for row in report.get("results", [])
            if isinstance(row, dict)
        }
        sleep_gate = results_by_name.get("recall_sleep_duration")
        self.assertIsNotNone(sleep_gate)
        self.assertEqual(sleep_gate.get("status"), "pass")

    def test_evaluate_lane_b_event_gates_warn_only_noncritical_room(self):
        gate_pass, reasons, report = self.pipeline._evaluate_lane_b_event_gates(
            room_name="Kitchen",
            candidate_metrics={
                "per_label_recall": {"kitchen_normal_use": 0.10},
                "per_label_support": {"kitchen_normal_use": 200, "unoccupied": 500},
            },
        )
        self.assertTrue(gate_pass)
        self.assertEqual(reasons, [])
        self.assertEqual(report.get("overall_status"), "warning")

    def test_evaluate_lane_b_event_gates_not_evaluated_without_metrics(self):
        gate_pass, reasons, report = self.pipeline._evaluate_lane_b_event_gates(
            room_name="Bedroom",
            candidate_metrics={"macro_f1": 0.7},
        )
        self.assertTrue(gate_pass)
        self.assertEqual(reasons, [])
        self.assertEqual(report.get("status"), "not_evaluated")

    @patch("ml.training.load_sensor_data")
    def test_find_archive_for_date_falls_back_to_timestamp_matching(self, mock_load_sensor_data):
        with TemporaryDirectory() as td:
            archive_dir = Path(td) / "archive" / "2026-02-14"
            archive_dir.mkdir(parents=True, exist_ok=True)
            candidate = archive_dir / "HK0011_jessica_train_5dec2025.parquet"
            candidate.write_text("placeholder")

            mock_load_sensor_data.return_value = {
                "Bedroom": pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            ["2025-12-05 00:00:00", "2025-12-05 23:59:59"]
                        ),
                        "activity": ["inactive", "reading"],
                    }
                )
            }

            found = self.pipeline._find_archive_for_date(
                record_date="2025-12-05",
                elder_id="HK0011_jessica",
                archive_dir=archive_dir.parent,
            )

            self.assertEqual(found, candidate)

    @patch("ml.training.tf.random.set_seed")
    @patch("ml.training.np.random.seed")
    @patch("ml.training.random.seed")
    def test_set_training_random_seed(self, mock_random_seed, mock_np_seed, mock_tf_seed):
        """Test that random seeds are set correctly from policy."""
        # Policy with specific seed
        self.pipeline.policy.reproducibility.random_seed = 42
        self.pipeline._set_training_random_seed(42)
        
        mock_random_seed.assert_called_with(42)
        mock_np_seed.assert_called_with(42)
        mock_tf_seed.assert_called_with(42)

    def test_build_timeline_targets_disabled_by_default(self):
        """Test that timeline targets are not built when flag is OFF."""
        y_train = np.array([0, 0, 1, 1, 0, 0], dtype=np.int32)
        self.mock_platform.label_encoders["room1"].classes_ = ["unoccupied", "sleep"]
        
        targets, debug = self.pipeline._build_timeline_targets(
            room_name="room1",
            y_train=y_train,
        )
        
        # Should return empty targets when flag is OFF
        self.assertEqual(len(targets), 0)
        self.assertFalse(debug["enabled"])
        self.assertEqual(debug["reason"], "feature_flag_or_room_disabled")

    @patch.dict(os.environ, {"ENABLE_TIMELINE_MULTITASK": "true", "TIMELINE_NATIVE_ROOMS": "bedroom,livingroom"}, clear=False)
    def test_build_timeline_targets_for_bedroom(self):
        """Test that timeline targets are built when flag is ON for eligible room."""
        y_train = np.array([0, 0, 0, 1, 1, 0, 0], dtype=np.int32)
        self.mock_platform.label_encoders["bedroom"] = MagicMock()
        self.mock_platform.label_encoders["bedroom"].classes_ = ["unoccupied", "sleep"]
        
        targets, debug = self.pipeline._build_timeline_targets(
            room_name="bedroom",
            y_train=y_train,
        )
        
        # Should return timeline targets
        self.assertTrue(debug["enabled"])
        self.assertIn("activity_labels", targets)
        self.assertIn("occupancy_labels", targets)
        self.assertIn("boundary_start_labels", targets)
        self.assertIn("boundary_end_labels", targets)
        
        # Verify shapes
        self.assertEqual(len(targets["activity_labels"]), len(y_train))
        self.assertEqual(len(targets["occupancy_labels"]), len(y_train))
        self.assertEqual(len(targets["boundary_start_labels"]), len(y_train))
        self.assertEqual(len(targets["boundary_end_labels"]), len(y_train))
        
        # Verify occupancy mapping (0=unoccupied, 1=sleep should map to 0,1)
        expected_occupancy = np.array([0, 0, 0, 1, 1, 0, 0], dtype=np.int32)
        np.testing.assert_array_equal(targets["occupancy_labels"], expected_occupancy)

    @patch.dict(os.environ, {"ENABLE_TIMELINE_MULTITASK": "true", "TIMELINE_NATIVE_ROOMS": "bedroom,livingroom"}, clear=False)
    def test_build_timeline_model_for_room_is_keras_compatible(self):
        """Timeline ON path should build a valid multi-output Keras model."""
        model, _ = self.pipeline._build_model_for_room(
            room_name="Bedroom",
            seq_length=5,
            num_classes=2,
            elder_id="elder1",
            training_mode="standard",
            warm_start=False,
        )
        self.assertTrue(bool(getattr(model, "_timeline_multitask_enabled", False)))
        self.assertTrue(isinstance(model.output, dict))
        self.assertEqual(
            set(model.output.keys()),
            {"activity_logits", "occupancy_logits", "boundary_start_logits", "boundary_end_logits"},
        )

    @patch.dict(os.environ, {"ENABLE_TIMELINE_MULTITASK": "true", "TIMELINE_NATIVE_ROOMS": "bedroom"}, clear=False)
    def test_timeline_multitask_model_can_fit_one_epoch(self):
        """Timeline ON path should support one-step fit with multitask targets."""
        model, _ = self.pipeline._build_model_for_room(
            room_name="bedroom",
            seq_length=5,
            num_classes=2,
            elder_id="elder1",
            training_mode="standard",
            warm_start=False,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss={
                "activity_logits": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                "occupancy_logits": tf.keras.losses.BinaryCrossentropy(from_logits=True),
                "boundary_start_logits": tf.keras.losses.BinaryCrossentropy(from_logits=True),
                "boundary_end_logits": tf.keras.losses.BinaryCrossentropy(from_logits=True),
            },
            metrics={"activity_logits": ["accuracy"]},
            jit_compile=False,
        )
        x = np.random.randn(8, 5, 3).astype(np.float32)
        y = {
            "activity_logits": np.random.randint(0, 2, size=(8,), dtype=np.int32),
            "occupancy_logits": np.random.randint(0, 2, size=(8, 1)).astype(np.float32),
            "boundary_start_logits": np.zeros((8, 1), dtype=np.float32),
            "boundary_end_logits": np.zeros((8, 1), dtype=np.float32),
        }
        history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)
        self.assertIn("activity_logits_accuracy", history.history)

    @patch.dict(os.environ, {"ENABLE_TIMELINE_MULTITASK": "true"}, clear=False)
    def test_timeline_multitask_flag_off_by_default(self):
        """Test that timeline multitask flag defaults to false."""
        # Clear the environment variable
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(TrainingPipeline._is_timeline_multitask_enabled())

    @patch.dict(os.environ, {"ENABLE_TIMELINE_MULTITASK": "true"}, clear=False)
    def test_timeline_multitask_flag_on_when_set(self):
        """Test that timeline multitask flag is ON when env var is set."""
        self.assertTrue(TrainingPipeline._is_timeline_multitask_enabled())

    def test_timeline_native_rooms_default(self):
        """Test default timeline native rooms."""
        with patch.dict(os.environ, {}, clear=True):
            rooms = TrainingPipeline._timeline_native_rooms()
            self.assertIn("bedroom", rooms)
            self.assertIn("livingroom", rooms)

    @patch.dict(os.environ, {"TIMELINE_NATIVE_ROOMS": "kitchen,bathroom"}, clear=False)
    def test_timeline_native_rooms_custom(self):
        """Test custom timeline native rooms."""
        rooms = TrainingPipeline._timeline_native_rooms()
        self.assertIn("kitchen", rooms)
        self.assertIn("bathroom", rooms)
        self.assertNotIn("bedroom", rooms)

    def test_build_timeline_targets_empty_y_train(self):
        """Test timeline targets with empty training data."""
        targets, debug = self.pipeline._build_timeline_targets(
            room_name="room1",
            y_train=np.array([], dtype=np.int32),
        )
        self.assertEqual(len(targets), 0)
        self.assertEqual(debug["n_samples"], 0)

    @patch.dict(os.environ, {"ENABLE_TIMELINE_MULTITASK": "true", "TIMELINE_NATIVE_ROOMS": "bedroom"}, clear=False)
    def test_build_timeline_targets_single_sample_has_boundary_keys(self):
        """Boundary targets should exist even for single-sample training batches."""
        self.mock_platform.label_encoders["bedroom"] = MagicMock()
        self.mock_platform.label_encoders["bedroom"].classes_ = ["unoccupied", "sleep"]
        targets, debug = self.pipeline._build_timeline_targets(
            room_name="bedroom",
            y_train=np.array([1], dtype=np.int32),
        )
        self.assertTrue(debug["enabled"])
        self.assertIn("boundary_start_labels", targets)
        self.assertIn("boundary_end_labels", targets)
        np.testing.assert_array_equal(targets["boundary_start_labels"], np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(targets["boundary_end_labels"], np.array([0], dtype=np.int32))

    @patch.dict(os.environ, {"ENABLE_TIMELINE_MULTITASK": "true", "TIMELINE_NATIVE_ROOMS": "unknown_room"}, clear=False)
    def test_build_timeline_targets_missing_label_encoder(self):
        """Test timeline targets when label encoder is missing."""
        y_train = np.array([0, 0, 1, 1], dtype=np.int32)
        # Don't add label encoder for "unknown_room"
        targets, debug = self.pipeline._build_timeline_targets(
            room_name="unknown_room",
            y_train=y_train,
        )
        self.assertEqual(len(targets), 0)
        self.assertEqual(debug["reason"], "missing_label_encoder")


if __name__ == '__main__':
    unittest.main()
