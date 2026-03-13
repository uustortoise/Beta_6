
import unittest
import os
import shutil
import tempfile
import json
import numpy as np
import joblib
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from ml.registry import ModelRegistry
from ml.exceptions import ModelLoadError


import unittest
import os
import shutil
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from ml.registry import ModelRegistry


class _DummyLayer:
    def __init__(self, name, weights):
        self.name = name
        self._weights = [np.array(w) for w in weights]
        self.weights = [1] if weights else []

    def get_weights(self):
        return [np.array(w) for w in self._weights]

    def set_weights(self, weights):
        self._weights = [np.array(w) for w in weights]


class _DummyModel:
    def __init__(self, layers):
        self.layers = layers

    def save(self, path):
        Path(path).write_text("dummy-model")


class _DummyEncoder:
    def __init__(self, classes):
        self.classes_ = classes

class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for models
        self.test_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(backend_dir=self.test_dir)
        self.elder_id = "test_elder"
        self.room_name = "living_room"

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_init_creates_directory(self):
        """Test that get_models_dir creates the directory."""
        path = self.registry.get_models_dir(self.elder_id)
        self.assertTrue(path.exists())

    def test_get_next_version_initial(self):
        """Test generating the first version number."""
        version = self.registry._get_next_version(self.elder_id, self.room_name)
        self.assertEqual(version, 1)

    def test_save_model_artifacts(self):
        """Test saving model artifacts with metadata and versioning."""
        accuracy = 0.85
        num_samples = 100
        
        # Mock model object
        mock_model = MagicMock()
        mock_model.save = MagicMock()
        
        # Mock scaler/encoder
        mock_scaler = MagicMock()
        mock_encoder = MagicMock()
        
        # Mock joblib
        with patch('ml.registry.joblib') as mock_joblib:
            version = self.registry.save_model_artifacts(
                self.elder_id,
                self.room_name,
                mock_model,
                mock_scaler,
                mock_encoder,
                accuracy=accuracy,
                samples=num_samples
            )
            
            self.assertEqual(version, 1)
            
            # Check version info JSON
            info = self.registry._load_version_info(self.elder_id, self.room_name)
            self.assertEqual(len(info['versions']), 1)
            self.assertEqual(info['versions'][0]['version'], 1)
            self.assertEqual(info['versions'][0]['accuracy'], accuracy)
            self.assertIsNone(info['versions'][0]['parent_version_id'])

            # verify cleanup was called (indirectly via file checks if we weren't mocking)

    def test_save_model_artifacts_persists_parent_version_id(self):
        """Fine-tuned candidates should persist lineage parent version metadata."""
        mock_model = MagicMock()
        mock_model.save = MagicMock()
        mock_scaler = MagicMock()
        mock_encoder = MagicMock()

        with patch('ml.registry.joblib'):
            version = self.registry.save_model_artifacts(
                self.elder_id,
                self.room_name,
                mock_model,
                mock_scaler,
                mock_encoder,
                accuracy=0.9,
                samples=200,
                parent_version_id=4,
            )

        self.assertEqual(version, 1)
        info = self.registry._load_version_info(self.elder_id, self.room_name)
        self.assertEqual(info['versions'][0]['parent_version_id'], 4)

    def test_rollback_to_version(self):
        """Test rolling back to a previous version."""
        # Create dummy version info
        info = {
            "versions": [
                {"version": 1, "accuracy": 0.8},
                {"version": 2, "accuracy": 0.9}
            ],
            "current_version": 2
        }
        self.registry._save_version_info(self.elder_id, self.room_name, info)
        
        # Create dummy versioned files
        models_dir = self.registry.get_models_dir(self.elder_id)
        (models_dir / f"{self.room_name}_v1_model.keras").touch()
        (models_dir / f"{self.room_name}_v2_model.keras").touch()
        
        # Perform rollback
        success = self.registry.rollback_to_version(self.elder_id, self.room_name, 1)
        self.assertTrue(success)
        
        # Check current version in info
        info = self.registry._load_version_info(self.elder_id, self.room_name)
        self.assertEqual(info['current_version'], 1)
        promoted = {int(v["version"]): bool(v.get("promoted", False)) for v in info["versions"]}
        self.assertTrue(promoted.get(1, False))
        self.assertFalse(promoted.get(2, True))
        
        # Check if "latest" file was updated (mock copy)
        # In this integration test with touch(), we can check if destination exists
        self.assertTrue((models_dir / f"{self.room_name}_model.keras").exists())

    def test_rollback_to_version_restores_latest_decision_trace(self):
        info = {
            "versions": [
                {"version": 1, "accuracy": 0.8, "promoted": False},
                {"version": 2, "accuracy": 0.9, "promoted": True},
            ],
            "current_version": 2,
        }
        self.registry._save_version_info(self.elder_id, self.room_name, info)

        models_dir = self.registry.get_models_dir(self.elder_id)
        (models_dir / f"{self.room_name}_v1_model.keras").touch()
        (models_dir / f"{self.room_name}_v2_model.keras").touch()
        (models_dir / f"{self.room_name}_v1_scaler.pkl").touch()
        (models_dir / f"{self.room_name}_v2_scaler.pkl").touch()
        (models_dir / f"{self.room_name}_v1_label_encoder.pkl").touch()
        (models_dir / f"{self.room_name}_v2_label_encoder.pkl").touch()
        (models_dir / f"{self.room_name}_v1_decision_trace.json").write_text(json.dumps({"saved_version": 1}))
        (models_dir / f"{self.room_name}_v2_decision_trace.json").write_text(json.dumps({"saved_version": 2}))
        (models_dir / f"{self.room_name}_decision_trace.json").write_text(json.dumps({"saved_version": 6}))

        success = self.registry.rollback_to_version(self.elder_id, self.room_name, 1)
        self.assertTrue(success)
        self.assertEqual(
            json.loads((models_dir / f"{self.room_name}_decision_trace.json").read_text())["saved_version"],
            1,
        )

    def test_rollback_removes_stale_threshold_when_missing(self):
        """Rollback should delete latest thresholds if target version has none."""
        info = {
            "versions": [
                {"version": 1, "accuracy": 0.8},
                {"version": 2, "accuracy": 0.9}
            ],
            "current_version": 2
        }
        self.registry._save_version_info(self.elder_id, self.room_name, info)

        models_dir = self.registry.get_models_dir(self.elder_id)
        # Target version has model but no thresholds file.
        (models_dir / f"{self.room_name}_v1_model.keras").touch()
        # Simulate stale latest threshold from a newer run.
        (models_dir / f"{self.room_name}_thresholds.json").write_text('{"0": 0.9}')

        success = self.registry.rollback_to_version(self.elder_id, self.room_name, 1)
        self.assertTrue(success)
        self.assertFalse((models_dir / f"{self.room_name}_thresholds.json").exists())

    def test_list_versions(self):
        """Test listing available versions."""
        info = {
            "versions": [
                {"version": 1, "created_at": "2023-01-01"},
                {"version": 3, "created_at": "2023-01-03"}
            ]
        }
        self.registry._save_version_info(self.elder_id, self.room_name, info)
        
        versions = self.registry.list_model_versions(self.elder_id, self.room_name)
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[1]['version'], 3)

    @patch('ml.registry.tf.keras.models.load_model')
    def test_load_room_model_supports_compile_false(self, mock_load_model):
        """Warm-start loaders can request compile=False for faster transfer-loading."""
        mock_load_model.return_value = MagicMock()
        _ = self.registry.load_room_model(
            model_path="/tmp/model.keras",
            room_name="living_room",
            compile_model=False,
        )
        _, kwargs = mock_load_model.call_args
        self.assertFalse(kwargs["compile"])

    def test_deactivate_current_version(self):
        """Deactivate should clear latest aliases and unset current_version."""
        info = {
            "versions": [
                {"version": 1, "accuracy": 0.8, "promoted": True}
            ],
            "current_version": 1
        }
        self.registry._save_version_info(self.elder_id, self.room_name, info)

        models_dir = self.registry.get_models_dir(self.elder_id)
        (models_dir / f"{self.room_name}_model.keras").touch()
        (models_dir / f"{self.room_name}_scaler.pkl").touch()
        (models_dir / f"{self.room_name}_label_encoder.pkl").touch()
        (models_dir / f"{self.room_name}_thresholds.json").write_text('{"0": 0.8}')
        (models_dir / f"{self.room_name}_decision_trace.json").write_text(json.dumps({"saved_version": 1}))

        ok = self.registry.deactivate_current_version(self.elder_id, self.room_name)
        self.assertTrue(ok)
        self.assertFalse((models_dir / f"{self.room_name}_model.keras").exists())
        self.assertFalse((models_dir / f"{self.room_name}_scaler.pkl").exists())
        self.assertFalse((models_dir / f"{self.room_name}_label_encoder.pkl").exists())
        self.assertFalse((models_dir / f"{self.room_name}_thresholds.json").exists())
        self.assertFalse((models_dir / f"{self.room_name}_decision_trace.json").exists())

        updated = self.registry._load_version_info(self.elder_id, self.room_name)
        self.assertEqual(updated["current_version"], 0)
        self.assertFalse(updated["versions"][0]["promoted"])

    def test_save_model_artifacts_reconciles_single_promoted_version(self):
        model = MagicMock()
        model.save = MagicMock()
        scaler = MagicMock()
        encoder = MagicMock()

        with patch('ml.registry.joblib'):
            v1 = self.registry.save_model_artifacts(
                self.elder_id, self.room_name, model, scaler, encoder, promote_to_latest=True
            )
            self.assertEqual(v1, 1)
            v2 = self.registry.save_model_artifacts(
                self.elder_id, self.room_name, model, scaler, encoder, promote_to_latest=True
            )
            self.assertEqual(v2, 2)

        info = self.registry._load_version_info(self.elder_id, self.room_name)
        promoted = [int(v["version"]) for v in info["versions"] if bool(v.get("promoted", False))]
        self.assertEqual(info["current_version"], 2)
        self.assertEqual(promoted, [2])

    def test_cleanup_preserves_current_champion_even_if_old(self):
        self.registry.MAX_VERSIONS_PER_ROOM = 3
        info = {
            "versions": [
                {"version": 1}, {"version": 2}, {"version": 3},
                {"version": 4}, {"version": 5}, {"version": 6},
            ],
            "current_version": 2,
        }
        self.registry._save_version_info(self.elder_id, self.room_name, info)
        models_dir = self.registry.get_models_dir(self.elder_id)
        for ver in range(1, 7):
            (models_dir / f"{self.room_name}_v{ver}_model.keras").touch()

        self.registry._cleanup_old_versions(self.elder_id, self.room_name)
        updated = self.registry._load_version_info(self.elder_id, self.room_name)
        retained = {v["version"] for v in updated["versions"]}

        self.assertIn(2, retained)
        self.assertEqual(updated["current_version"], 2)
        self.assertTrue((models_dir / f"{self.room_name}_v2_model.keras").exists())
        # One of the newest non-current versions should have been pruned.
        self.assertEqual(len(updated["versions"]), 3)

    def test_cleanup_resets_stale_current_version_reference(self):
        self.registry.MAX_VERSIONS_PER_ROOM = 3
        info = {
            "versions": [{"version": 4}, {"version": 5}, {"version": 6}],
            "current_version": 2,  # stale reference: version 2 metadata is missing
        }
        self.registry._save_version_info(self.elder_id, self.room_name, info)

        self.registry._cleanup_old_versions(self.elder_id, self.room_name)
        updated = self.registry._load_version_info(self.elder_id, self.room_name)
        self.assertEqual(updated["current_version"], 0)

    def test_save_model_artifacts_shared_adapter_writes_adapter_payload(self):
        model = _DummyModel(
            layers=[
                _DummyLayer("cnn_embedding", [np.ones((2, 2))]),
                _DummyLayer("transformer_block_0", [np.ones((2, 2)) * 2]),
                _DummyLayer("dense", [np.ones((2, 2)) * 3]),
            ]
        )
        scaler = {"s": 1}
        encoder = {"e": 1}

        version = self.registry.save_model_artifacts(
            self.elder_id,
            self.room_name,
            model,
            scaler,
            encoder,
            accuracy=0.9,
            samples=42,
            model_identity={
                "family": "shared_backbone_adapter",
                "backbone_id": "bb-v1",
                "adapter_id": f"{self.elder_id}:{self.room_name}",
            },
            promote_to_latest=True,
        )
        self.assertEqual(version, 1)

        models_dir = self.registry.get_models_dir(self.elder_id)
        self.assertTrue((models_dir / f"{self.room_name}_v1_adapter_weights.pkl").exists())
        self.assertTrue((models_dir / f"{self.room_name}_adapter_weights.pkl").exists())

        info = self.registry._load_version_info(self.elder_id, self.room_name)
        self.assertEqual(info["versions"][0]["adapter_weights_path"], f"{self.room_name}_v1_adapter_weights.pkl")
        self.assertIsNotNone(info["versions"][0]["backbone_weights_path"])

    def test_save_model_artifacts_candidate_only_does_not_seed_shared_backbone_snapshot(self):
        model = _DummyModel(
            layers=[
                _DummyLayer("cnn_embedding", [np.ones((2, 2))]),
                _DummyLayer("transformer_block_0", [np.ones((2, 2)) * 2]),
                _DummyLayer("dense", [np.ones((2, 2)) * 3]),
            ]
        )
        scaler = {"s": 1}
        encoder = {"e": 1}
        identity = {
            "family": "shared_backbone_adapter",
            "backbone_id": "bb-v1",
            "adapter_id": f"{self.elder_id}:{self.room_name}",
        }

        version = self.registry.save_model_artifacts(
            self.elder_id,
            self.room_name,
            model,
            scaler,
            encoder,
            model_identity=identity,
            promote_to_latest=False,
        )
        self.assertEqual(version, 1)

        models_dir = self.registry.get_models_dir(self.elder_id)
        backbone_path = self.registry.get_backbone_weights_path(self.elder_id, self.room_name, "bb-v1")

        self.assertTrue((models_dir / f"{self.room_name}_v1_adapter_weights.pkl").exists())
        self.assertFalse((models_dir / f"{self.room_name}_adapter_weights.pkl").exists())
        self.assertFalse(backbone_path.exists())

        info = self.registry._load_version_info(self.elder_id, self.room_name)
        self.assertIsNone(info["versions"][0]["backbone_weights_path"])

    def test_rollback_to_version_restores_latest_adapter_weights(self):
        model = _DummyModel(
            layers=[
                _DummyLayer("cnn_embedding", [np.ones((2, 2))]),
                _DummyLayer("transformer_block_0", [np.ones((2, 2)) * 2]),
                _DummyLayer("dense", [np.ones((2, 2)) * 3]),
            ]
        )
        scaler = {"s": 1}
        encoder = {"e": 1}
        identity = {
            "family": "shared_backbone_adapter",
            "backbone_id": "bb-v1",
            "adapter_id": f"{self.elder_id}:{self.room_name}",
        }

        self.registry.save_model_artifacts(
            self.elder_id, self.room_name, model, scaler, encoder, model_identity=identity, promote_to_latest=True
        )
        # Save second version with different adapter head
        model_v2 = _DummyModel(
            layers=[
                _DummyLayer("cnn_embedding", [np.ones((2, 2))]),
                _DummyLayer("transformer_block_0", [np.ones((2, 2)) * 2]),
                _DummyLayer("dense", [np.ones((2, 2)) * 9]),
            ]
        )
        self.registry.save_model_artifacts(
            self.elder_id, self.room_name, model_v2, scaler, encoder, model_identity=identity, promote_to_latest=True
        )

        ok = self.registry.rollback_to_version(self.elder_id, self.room_name, 1)
        self.assertTrue(ok)

        models_dir = self.registry.get_models_dir(self.elder_id)
        latest_adapter = joblib.load(models_dir / f"{self.room_name}_adapter_weights.pkl")
        version1_adapter = joblib.load(models_dir / f"{self.room_name}_v1_adapter_weights.pkl")
        self.assertEqual(sorted(latest_adapter.keys()), sorted(version1_adapter.keys()))

    def test_rollback_to_version_materializes_shared_backbone_snapshot(self):
        model = _DummyModel(
            layers=[
                _DummyLayer("cnn_embedding", [np.ones((2, 2))]),
                _DummyLayer("transformer_block_0", [np.ones((2, 2)) * 2]),
                _DummyLayer("dense", [np.ones((2, 2)) * 3]),
            ]
        )
        scaler = {"s": 1}
        encoder = {"e": 1}
        identity = {
            "family": "shared_backbone_adapter",
            "backbone_id": "bb-v1",
            "adapter_id": f"{self.elder_id}:{self.room_name}",
        }

        self.registry.save_model_artifacts(
            self.elder_id,
            self.room_name,
            model,
            scaler,
            encoder,
            model_identity=identity,
            promote_to_latest=False,
        )
        backbone_path = self.registry.get_backbone_weights_path(self.elder_id, self.room_name, "bb-v1")
        self.assertFalse(backbone_path.exists())

        with patch.object(self.registry, "load_room_model", return_value=model) as mock_load:
            ok = self.registry.rollback_to_version(self.elder_id, self.room_name, 1)
        self.assertTrue(ok)
        self.assertTrue(backbone_path.exists())
        mock_load.assert_called_once()

    @patch("ml.transformer_backbone.build_transformer_model")
    @patch("config.get_room_config")
    def test_load_models_for_elder_prefers_shared_adapter_composition(self, mock_get_room_config, mock_build):
        mock_cfg = MagicMock()
        mock_cfg.calculate_seq_length.return_value = 5
        mock_get_room_config.return_value = mock_cfg

        runtime_model = _DummyModel(
            layers=[
                _DummyLayer("cnn_embedding", [np.zeros((1, 1))]),
                _DummyLayer("transformer_block_0", [np.zeros((1, 1))]),
                _DummyLayer("dense", [np.zeros((1, 1))]),
            ]
        )
        mock_build.return_value = runtime_model

        models_dir = self.registry.get_models_dir(self.elder_id)
        room = self.room_name
        (models_dir / f"{room}_versions.json").write_text(
            json.dumps(
                {
                    "versions": [
                        {
                            "version": 1,
                            "model_identity": {
                                "family": "shared_backbone_adapter",
                                "backbone_id": "bb-v1",
                            },
                        }
                    ],
                    "current_version": 1,
                }
            )
        )
        joblib.dump({"scale": 1}, models_dir / f"{room}_scaler.pkl")
        encoder = _DummyEncoder(["a", "b"])
        joblib.dump(encoder, models_dir / f"{room}_label_encoder.pkl")
        (models_dir / "_shared_backbones").mkdir(exist_ok=True)
        joblib.dump({"cnn_embedding": [np.ones((1, 1))], "transformer_block_0": [np.ones((1, 1))]}, models_dir / "_shared_backbones" / f"{room}_bb-v1_backbone_weights.pkl")
        joblib.dump({"cnn_embedding": [np.ones((1, 1))], "transformer_block_0": [np.ones((1, 1))]}, models_dir / "_shared_backbones" / f"{room}_bb-v1_backbone_weights.pkl")
        joblib.dump({"dense": [np.ones((1, 1)) * 3]}, models_dir / f"{room}_v1_adapter_weights.pkl")
        joblib.dump({"dense": [np.ones((1, 1)) * 3]}, models_dir / f"{room}_adapter_weights.pkl")
        # keep a latest alias file present, but composition should be preferred.
        (models_dir / f"{room}_model.keras").write_text("dummy")

        platform = MagicMock()
        platform.room_models = {}
        platform.scalers = {}
        platform.label_encoders = {}
        platform.sensor_columns = ["s1", "s2", "s3"]

        with patch.object(self.registry, "load_room_model") as mock_load_room:
            loaded = self.registry.load_models_for_elder(self.elder_id, platform)
            self.assertIn(room, loaded)
            self.assertIn(room, platform.room_models)
            # Shared adapter path should bypass direct keras loading.
            mock_load_room.assert_not_called()

    @patch("ml.transformer_backbone.build_transformer_model")
    @patch("config.get_room_config")
    def test_load_models_for_elder_falls_back_when_shared_adapter_coverage_low(self, mock_get_room_config, mock_build):
        mock_cfg = MagicMock()
        mock_cfg.calculate_seq_length.return_value = 5
        mock_get_room_config.return_value = mock_cfg

        runtime_model = _DummyModel(
            layers=[
                _DummyLayer("cnn_embedding", [np.zeros((1, 1))]),
                _DummyLayer("transformer_block_0", [np.zeros((1, 1))]),
                _DummyLayer("dense", [np.zeros((1, 1))]),
            ]
        )
        mock_build.return_value = runtime_model

        models_dir = self.registry.get_models_dir(self.elder_id)
        room = self.room_name
        (models_dir / f"{room}_versions.json").write_text(
            json.dumps(
                {
                    "versions": [
                        {
                            "version": 1,
                            "model_identity": {
                                "family": "shared_backbone_adapter",
                                "backbone_id": "bb-v1",
                            },
                        }
                    ],
                    "current_version": 1,
                }
            )
        )
        joblib.dump({"scale": 1}, models_dir / f"{room}_scaler.pkl")
        encoder = _DummyEncoder(["a", "b"])
        joblib.dump(encoder, models_dir / f"{room}_label_encoder.pkl")
        (models_dir / "_shared_backbones").mkdir(exist_ok=True)
        # Intentionally mismatched names so loaded_layers become 0.
        joblib.dump({"missing_backbone_layer": [np.ones((1, 1))]}, models_dir / "_shared_backbones" / f"{room}_bb-v1_backbone_weights.pkl")
        joblib.dump({"missing_adapter_layer": [np.ones((1, 1))]}, models_dir / f"{room}_adapter_weights.pkl")
        (models_dir / f"{room}_model.keras").write_text("dummy")

        platform = MagicMock()
        platform.room_models = {}
        platform.scalers = {}
        platform.label_encoders = {}
        platform.sensor_columns = ["s1", "s2", "s3"]

        with patch.object(self.registry, "load_room_model", return_value="fallback_model") as mock_load_room:
            loaded = self.registry.load_models_for_elder(self.elder_id, platform)
            self.assertIn(room, loaded)
            self.assertEqual(platform.room_models[room], "fallback_model")
            mock_load_room.assert_called_once()

    @patch("ml.transformer_backbone.build_transformer_model")
    @patch("config.get_room_config")
    def test_load_models_for_elder_falls_back_when_backbone_coverage_rounding_would_otherwise_pass(
        self, mock_get_room_config, mock_build
    ):
        mock_cfg = MagicMock()
        mock_cfg.calculate_seq_length.return_value = 5
        mock_get_room_config.return_value = mock_cfg

        runtime_model = _DummyModel(
            layers=[
                _DummyLayer("cnn_embedding", [np.zeros((1, 1))]),
                _DummyLayer("transformer_block_0", [np.zeros((1, 1))]),
                _DummyLayer("dense", [np.zeros((1, 1))]),
            ]
        )
        mock_build.return_value = runtime_model

        models_dir = self.registry.get_models_dir(self.elder_id)
        room = self.room_name
        (models_dir / f"{room}_versions.json").write_text(
            json.dumps(
                {
                    "versions": [
                        {
                            "version": 1,
                            "model_identity": {
                                "family": "shared_backbone_adapter",
                                "backbone_id": "bb-v1",
                            },
                        }
                    ],
                    "current_version": 1,
                }
            )
        )
        joblib.dump({"scale": 1}, models_dir / f"{room}_scaler.pkl")
        encoder = _DummyEncoder(["a", "b"])
        joblib.dump(encoder, models_dir / f"{room}_label_encoder.pkl")
        (models_dir / "_shared_backbones").mkdir(exist_ok=True)
        # 2/4 backbone layers load successfully. With floor this could pass (min 2), with ceil it must fail (min 3).
        joblib.dump(
            {
                "cnn_embedding": [np.ones((1, 1))],
                "transformer_block_0": [np.ones((1, 1))],
                "missing_backbone_1": [np.ones((1, 1))],
                "missing_backbone_2": [np.ones((1, 1))],
            },
            models_dir / "_shared_backbones" / f"{room}_bb-v1_backbone_weights.pkl",
        )
        # Adapter fully matches to isolate backbone threshold behavior.
        joblib.dump({"dense": [np.ones((1, 1)) * 3]}, models_dir / f"{room}_adapter_weights.pkl")
        (models_dir / f"{room}_model.keras").write_text("dummy")

        platform = MagicMock()
        platform.room_models = {}
        platform.scalers = {}
        platform.label_encoders = {}
        platform.sensor_columns = ["s1", "s2", "s3"]

        with patch.object(self.registry, "load_room_model", return_value="fallback_model") as mock_load_room:
            loaded = self.registry.load_models_for_elder(self.elder_id, platform)
            self.assertIn(room, loaded)
            self.assertEqual(platform.room_models[room], "fallback_model")
            mock_load_room.assert_called_once()

    def test_load_models_for_elder_skips_bad_room_and_loads_others(self):
        models_dir = self.registry.get_models_dir(self.elder_id)
        good_room = "bedroom"
        bad_room = "kitchen"

        for room in (good_room, bad_room):
            (models_dir / f"{room}_versions.json").write_text(json.dumps({"versions": [{"version": 1}], "current_version": 1}))
            (models_dir / f"{room}_scaler.pkl").write_text("dummy")
            (models_dir / f"{room}_label_encoder.pkl").write_text("dummy")
            (models_dir / f"{room}_model.keras").write_text("dummy")

        platform = MagicMock()
        platform.room_models = {}
        platform.scalers = {}
        platform.label_encoders = {}
        platform.class_thresholds = {}

        def _joblib_load(path):
            p = str(path)
            if p.endswith("_scaler.pkl"):
                return {"scale": 1}
            if p.endswith("_label_encoder.pkl"):
                return _DummyEncoder(["a", "b"])
            return {"x": 1}

        def _load_room_model(path, room_name, compile_model=True):
            if room_name == bad_room:
                raise ModelLoadError("corrupt model")
            return f"model:{room_name}"

        with patch("ml.registry.joblib.load", side_effect=_joblib_load):
            with patch.object(self.registry, "load_room_model", side_effect=_load_room_model):
                with patch.object(self.registry, "_try_load_shared_adapter_room_model", return_value=None):
                    loaded_rooms = self.registry.load_models_for_elder(self.elder_id, platform)

        self.assertIn(good_room, loaded_rooms)
        self.assertNotIn(bad_room, loaded_rooms)
        self.assertEqual(platform.room_models.get(good_room), f"model:{good_room}")
        self.assertNotIn(bad_room, platform.room_models)
        self.assertNotIn(bad_room, platform.scalers)
        self.assertNotIn(bad_room, platform.label_encoders)

    def test_load_models_for_elder_versions_discovery_missing_latest_alias_is_skipped(self):
        models_dir = self.registry.get_models_dir(self.elder_id)
        good_room = "bedroom"
        missing_alias_room = "livingroom"

        # Good room has full latest alias.
        (models_dir / f"{good_room}_versions.json").write_text(json.dumps({"versions": [{"version": 1}], "current_version": 1}))
        (models_dir / f"{good_room}_scaler.pkl").write_text("dummy")
        (models_dir / f"{good_room}_label_encoder.pkl").write_text("dummy")
        (models_dir / f"{good_room}_model.keras").write_text("dummy")

        # Room discovered only via versions metadata; latest alias is missing.
        (models_dir / f"{missing_alias_room}_versions.json").write_text(
            json.dumps({"versions": [{"version": 1}], "current_version": 1})
        )
        (models_dir / f"{missing_alias_room}_scaler.pkl").write_text("dummy")
        (models_dir / f"{missing_alias_room}_label_encoder.pkl").write_text("dummy")
        # Intentionally no `{missing_alias_room}_model.keras`

        platform = MagicMock()
        platform.room_models = {}
        platform.scalers = {}
        platform.label_encoders = {}
        platform.class_thresholds = {}

        def _joblib_load(path):
            p = str(path)
            if p.endswith("_scaler.pkl"):
                return {"scale": 1}
            if p.endswith("_label_encoder.pkl"):
                return _DummyEncoder(["a", "b"])
            return {"x": 1}

        def _load_room_model(path, room_name, compile_model=True):
            if room_name == missing_alias_room:
                raise ModelLoadError("latest alias missing")
            return f"model:{room_name}"

        with patch("ml.registry.joblib.load", side_effect=_joblib_load):
            with patch.object(self.registry, "_try_load_shared_adapter_room_model", return_value=None):
                with patch.object(self.registry, "load_room_model", side_effect=_load_room_model):
                    loaded_rooms = self.registry.load_models_for_elder(self.elder_id, platform)

        self.assertIn(good_room, loaded_rooms)
        self.assertNotIn(missing_alias_room, loaded_rooms)
        self.assertEqual(platform.room_models.get(good_room), f"model:{good_room}")
        self.assertNotIn(missing_alias_room, platform.room_models)
        self.assertNotIn(missing_alias_room, platform.scalers)
        self.assertNotIn(missing_alias_room, platform.label_encoders)

    def test_validate_and_repair_room_registry_state(self):
        """Test the audit and repair logic for registry consistency."""
        models_dir = self.registry.get_models_dir(self.elder_id)
        
        # Scenario 1: Multiple promoted versions -> should pick winner
        # We must write manually because _save_version_info auto-reconciles!
        info = {
            "versions": [
                {"version": 1, "promoted": True, "created_at": "2023-01-01"},
                {"version": 2, "promoted": True, "created_at": "2023-01-02"}
            ],
            "current_version": 2
        }
        (models_dir / f"{self.room_name}_versions.json").write_text(json.dumps(info))
        
        # Ensure aliases exist so we don't get distracted by alias errors
        (models_dir / f"{self.room_name}_model.keras").touch()
        (models_dir / f"{self.room_name}_scaler.pkl").touch()
        (models_dir / f"{self.room_name}_label_encoder.pkl").touch()
        
        report = self.registry.validate_and_repair_room_registry_state(self.elder_id, self.room_name)
        self.assertTrue(report["repaired"])
        self.assertIn("Multiple promoted versions found", str(report["issues"]))
        
        updated = self.registry._load_version_info(self.elder_id, self.room_name)
        v1 = next(v for v in updated["versions"] if v["version"] == 1)
        v2 = next(v for v in updated["versions"] if v["version"] == 2)
        self.assertFalse(v1["promoted"])
        self.assertTrue(v2["promoted"])
        
        # Scenario 2: Current version points to missing version -> reset to 0
        info = {
            "versions": [{"version": 1}],
            "current_version": 99
        }
        (models_dir / f"{self.room_name}_versions.json").write_text(json.dumps(info))
        
        report = self.registry.validate_and_repair_room_registry_state(self.elder_id, self.room_name)
        self.assertTrue(report["repaired"])
        updated = self.registry._load_version_info(self.elder_id, self.room_name)
        self.assertEqual(updated["current_version"], 0)
        
        # Scenario 3: Orphan aliases with current_version=0 -> delete aliases
        (models_dir / f"{self.room_name}_model.keras").touch()
        report = self.registry.validate_and_repair_room_registry_state(self.elder_id, self.room_name)
        self.assertTrue(report["repaired"])
        self.assertFalse((models_dir / f"{self.room_name}_model.keras").exists())
        
        # Scenario 4: Missing aliases for valid current -> restore
        info = {
            "versions": [{"version": 1, "promoted": True}],
            "current_version": 1
        }
        (models_dir / f"{self.room_name}_versions.json").write_text(json.dumps(info))
        
        # Ensure versioned artifact exists
        (models_dir / f"{self.room_name}_v1_model.keras").touch()
        (models_dir / f"{self.room_name}_v1_scaler.pkl").touch()
        (models_dir / f"{self.room_name}_v1_label_encoder.pkl").touch()

        # Latest alias missing (ensure clean start)
        if (models_dir / f"{self.room_name}_model.keras").exists():
            (models_dir / f"{self.room_name}_model.keras").unlink()
            
        report = self.registry.validate_and_repair_room_registry_state(self.elder_id, self.room_name)
        self.assertTrue(report["repaired"])
        self.assertTrue((models_dir / f"{self.room_name}_model.keras").exists())

    def test_validate_and_repair_force_copies_stale_aliases(self):
        """
        P1 Hardening: Repair should FORCE copy versioned artifacts to aliases 
        even if aliases exist, to ensure they match the champion bit-for-bit.
        """
        models_dir = self.registry.get_models_dir(self.elder_id)
        
        # Setup: Valid promoted version 1
        info = {
            "versions": [{"version": 1, "promoted": True}],
            "current_version": 1
        }
        (models_dir / f"{self.room_name}_versions.json").write_text(json.dumps(info))
        
        # 1. Create source truth (versioned artifacts)
        (models_dir / f"{self.room_name}_v1_model.keras").write_text("correct_model_content")
        (models_dir / f"{self.room_name}_v1_scaler.pkl").write_text("correct_scaler_content")
        (models_dir / f"{self.room_name}_v1_label_encoder.pkl").write_text("correct_encoder_content")
        
        # 2. Create STALE aliases (different content)
        (models_dir / f"{self.room_name}_model.keras").write_text("stale_wrong_content")
        (models_dir / f"{self.room_name}_scaler.pkl").write_text("stale_wrong_content")
        (models_dir / f"{self.room_name}_label_encoder.pkl").write_text("stale_wrong_content")
        
        # 3. Trigger repair
        # It should detect that alias file timestamps/content logic dictates a refresh, 
        # or simply blindly overwrite as per P1 hardening.
        report = self.registry.validate_and_repair_room_registry_state(self.elder_id, self.room_name)
        
        # 4. Verify aliases were overwritten
        self.assertEqual((models_dir / f"{self.room_name}_model.keras").read_text(), "correct_model_content")
        self.assertEqual((models_dir / f"{self.room_name}_scaler.pkl").read_text(), "correct_scaler_content")
        self.assertEqual((models_dir / f"{self.room_name}_label_encoder.pkl").read_text(), "correct_encoder_content")

    def test_validate_and_repair_invalid_when_mandatory_versioned_missing(self):
        """
        Registry repair must not report valid when champion versioned artifacts are incomplete.
        """
        models_dir = self.registry.get_models_dir(self.elder_id)
        info = {
            "versions": [{"version": 1, "promoted": True}],
            "current_version": 1,
        }
        (models_dir / f"{self.room_name}_versions.json").write_text(json.dumps(info))

        # Only one mandatory versioned artifact exists.
        (models_dir / f"{self.room_name}_v1_model.keras").write_text("model")

        report = self.registry.validate_and_repair_room_registry_state(self.elder_id, self.room_name)
        self.assertFalse(report["valid"])
        self.assertIn("Missing mandatory versioned artifacts", str(report["issues"]))

    def test_validate_and_repair_skips_sync_when_aliases_already_match(self):
        """
        Repair should avoid unnecessary alias copy I/O when aliases already match champion artifacts.
        """
        models_dir = self.registry.get_models_dir(self.elder_id)
        info = {
            "versions": [{"version": 1, "promoted": True}],
            "current_version": 1,
        }
        (models_dir / f"{self.room_name}_versions.json").write_text(json.dumps(info))

        for suffix in ["_model.keras", "_scaler.pkl", "_label_encoder.pkl"]:
            (models_dir / f"{self.room_name}_v1{suffix}").write_text(f"same{suffix}")
            (models_dir / f"{self.room_name}{suffix}").write_text(f"same{suffix}")

        with patch.object(self.registry, "_ensure_latest_aliases_match_current", wraps=self.registry._ensure_latest_aliases_match_current) as sync_mock:
            report = self.registry.validate_and_repair_room_registry_state(self.elder_id, self.room_name)

        self.assertTrue(report["valid"])
        self.assertEqual(sync_mock.call_count, 0)


if __name__ == '__main__':
    unittest.main()
