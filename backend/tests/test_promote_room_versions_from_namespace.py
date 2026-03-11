import json
import shutil
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent.parent))

from ml.registry import ModelRegistry
from scripts.promote_room_versions_from_namespace import promote_room_versions_from_namespace


class TestPromoteRoomVersionsFromNamespace(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(backend_dir=self.test_dir)
        self.source_elder = "candidate_elder"
        self.target_elder = "live_elder"
        self.room = "Bedroom"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _write_version_artifacts(
        self,
        elder_id: str,
        room_name: str,
        version: int,
        *,
        payload_tag: str,
        include_two_stage: bool = False,
    ) -> None:
        models_dir = self.registry.get_models_dir(elder_id)
        for suffix in ("_model.keras", "_scaler.pkl", "_label_encoder.pkl"):
            (models_dir / f"{room_name}_v{version}{suffix}").write_text(
                f"{payload_tag}:{version}:{suffix}",
                encoding="utf-8",
            )
        (models_dir / f"{room_name}_v{version}_thresholds.json").write_text(
            json.dumps({"0": version / 100.0}),
            encoding="utf-8",
        )
        (models_dir / f"{room_name}_v{version}_activity_confidence_calibrator.json").write_text(
            json.dumps({"schema_version": "activity_acceptance_score_v1", "version": version}),
            encoding="utf-8",
        )
        (models_dir / f"{room_name}_v{version}_decision_trace.json").write_text(
            json.dumps(
                {
                    "elder_id": elder_id,
                    "room": room_name,
                    "version": version,
                    "saved_version": version,
                    "tag": payload_tag,
                    "artifact_paths": {
                        "stage_a_model_versioned": str(
                            models_dir / f"{room_name}_v{version}_two_stage_stage_a_model.keras"
                        ),
                        "meta_versioned": str(models_dir / f"{room_name}_v{version}_two_stage_meta.json"),
                    },
                }
            ),
            encoding="utf-8",
        )

        if include_two_stage:
            (models_dir / f"{room_name}_v{version}_two_stage_meta.json").write_text(
                json.dumps(
                    {
                        "schema_version": "beta6.two_stage_core.v1",
                        "elder_id": elder_id,
                        "room": room_name,
                        "saved_version": version,
                        "runtime_enabled": True,
                        "stage_b_enabled": True,
                    }
                ),
                encoding="utf-8",
            )
            (models_dir / f"{room_name}_v{version}_two_stage_stage_a_model.keras").write_text(
                f"{payload_tag}:{version}:stage_a",
                encoding="utf-8",
            )
            (models_dir / f"{room_name}_v{version}_two_stage_stage_b_model.keras").write_text(
                f"{payload_tag}:{version}:stage_b",
                encoding="utf-8",
            )

    def _write_versions_json(
        self,
        elder_id: str,
        room_name: str,
        *,
        current_version: int,
        versions: list[dict],
    ) -> None:
        payload = {
            "versions": versions,
            "current_version": current_version,
        }
        self.registry._save_version_info(elder_id, room_name, payload)

    def test_promote_room_versions_from_namespace_preserves_history_and_promotes_target(self):
        self._write_version_artifacts(self.target_elder, self.room, 7, payload_tag="live")
        self._write_version_artifacts(self.target_elder, self.room, 10, payload_tag="shared")
        self._write_versions_json(
            self.target_elder,
            self.room,
            current_version=7,
            versions=[
                {"version": 10, "created_at": "2026-03-10T00:10:00", "promoted": False, "metrics": {"tag": "shared"}},
                {"version": 7, "created_at": "2026-03-10T00:07:00", "promoted": True, "metrics": {"tag": "live"}},
            ],
        )
        self.registry.rollback_to_version(self.target_elder, self.room, 7)

        self._write_version_artifacts(self.source_elder, self.room, 10, payload_tag="shared")
        self._write_version_artifacts(
            self.source_elder,
            self.room,
            11,
            payload_tag="candidate",
            include_two_stage=True,
        )
        self._write_versions_json(
            self.source_elder,
            self.room,
            current_version=11,
            versions=[
                {"version": 11, "created_at": "2026-03-11T00:11:00", "promoted": True, "metrics": {"tag": "candidate"}},
                {"version": 10, "created_at": "2026-03-10T00:10:00", "promoted": False, "metrics": {"tag": "shared"}},
            ],
        )

        summary = promote_room_versions_from_namespace(
            backend_dir=self.test_dir,
            source_elder_id=self.source_elder,
            target_elder_id=self.target_elder,
            room_versions={self.room: 11},
        )

        room_summary = summary["rooms"][0]
        self.assertEqual(room_summary["room"], self.room)
        self.assertEqual(room_summary["target_current_version"], 11)
        self.assertEqual(room_summary["copied_versions"], [11])
        self.assertEqual(room_summary["reused_versions"], [10])
        self.assertIn(f"{self.room}_decision_trace.json", room_summary["latest_artifacts"])

        info = self.registry._load_version_info(self.target_elder, self.room)
        self.assertEqual(info["current_version"], 11)
        versions = sorted(int(item["version"]) for item in info["versions"])
        self.assertEqual(versions, [7, 10, 11])

        promoted = {int(item["version"]): bool(item.get("promoted", False)) for item in info["versions"]}
        self.assertFalse(promoted[7])
        self.assertTrue(promoted[11])

        models_dir = self.registry.get_models_dir(self.target_elder)
        self.assertEqual(
            (models_dir / f"{self.room}_model.keras").read_text(encoding="utf-8"),
            "candidate:11:_model.keras",
        )
        self.assertEqual(
            json.loads((models_dir / f"{self.room}_v11_decision_trace.json").read_text(encoding="utf-8"))["version"],
            11,
        )
        self.assertEqual(
            json.loads((models_dir / f"{self.room}_decision_trace.json").read_text(encoding="utf-8"))["version"],
            11,
        )
        versioned_trace = json.loads((models_dir / f"{self.room}_v11_decision_trace.json").read_text(encoding="utf-8"))
        latest_trace = json.loads((models_dir / f"{self.room}_decision_trace.json").read_text(encoding="utf-8"))
        latest_two_stage = json.loads((models_dir / f"{self.room}_two_stage_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(versioned_trace["elder_id"], self.target_elder)
        self.assertEqual(latest_trace["elder_id"], self.target_elder)
        self.assertEqual(latest_two_stage["elder_id"], self.target_elder)
        self.assertIn(f"/models/{self.target_elder}/", versioned_trace["artifact_paths"]["stage_a_model_versioned"])
        self.assertNotIn(f"/models/{self.source_elder}/", versioned_trace["artifact_paths"]["stage_a_model_versioned"])
        self.assertTrue((models_dir / f"{self.room}_two_stage_meta.json").exists())
        self.assertTrue((models_dir / f"{self.room}_two_stage_stage_a_model.keras").exists())
        self.assertTrue((models_dir / f"{self.room}_two_stage_stage_b_model.keras").exists())
        self.assertTrue((models_dir / f"{self.room}_v7_model.keras").exists())

    def test_promote_room_versions_from_namespace_repairs_stale_target_json_on_repeat_promotion(self):
        self._write_version_artifacts(
            self.source_elder,
            self.room,
            11,
            payload_tag="candidate",
            include_two_stage=True,
        )
        self._write_versions_json(
            self.source_elder,
            self.room,
            current_version=11,
            versions=[
                {"version": 11, "created_at": "2026-03-11T00:11:00", "promoted": True, "metrics": {"tag": "candidate"}},
            ],
        )

        promote_room_versions_from_namespace(
            backend_dir=self.test_dir,
            source_elder_id=self.source_elder,
            target_elder_id=self.target_elder,
            room_versions={self.room: 11},
        )

        models_dir = self.registry.get_models_dir(self.target_elder)
        stale_trace_path = models_dir / f"{self.room}_v11_decision_trace.json"
        stale_trace = json.loads(stale_trace_path.read_text(encoding="utf-8"))
        stale_trace["elder_id"] = self.source_elder
        stale_trace["artifact_paths"]["stage_a_model_versioned"] = str(
            self.registry.get_models_dir(self.source_elder) / f"{self.room}_v11_two_stage_stage_a_model.keras"
        )
        stale_trace_path.write_text(json.dumps(stale_trace), encoding="utf-8")

        stale_latest_path = models_dir / f"{self.room}_decision_trace.json"
        stale_latest_path.write_text(json.dumps(stale_trace), encoding="utf-8")

        stale_two_stage_path = models_dir / f"{self.room}_v11_two_stage_meta.json"
        stale_two_stage = json.loads(stale_two_stage_path.read_text(encoding="utf-8"))
        stale_two_stage["elder_id"] = self.source_elder
        stale_two_stage_path.write_text(json.dumps(stale_two_stage), encoding="utf-8")

        promote_room_versions_from_namespace(
            backend_dir=self.test_dir,
            source_elder_id=self.source_elder,
            target_elder_id=self.target_elder,
            room_versions={self.room: 11},
        )

        repaired_trace = json.loads(stale_trace_path.read_text(encoding="utf-8"))
        repaired_latest = json.loads(stale_latest_path.read_text(encoding="utf-8"))
        repaired_two_stage = json.loads(stale_two_stage_path.read_text(encoding="utf-8"))
        self.assertEqual(repaired_trace["elder_id"], self.target_elder)
        self.assertEqual(repaired_latest["elder_id"], self.target_elder)
        self.assertEqual(repaired_two_stage["elder_id"], self.target_elder)
        self.assertIn(f"/models/{self.target_elder}/", repaired_trace["artifact_paths"]["stage_a_model_versioned"])
        self.assertNotIn(f"/models/{self.source_elder}/", repaired_trace["artifact_paths"]["stage_a_model_versioned"])


if __name__ == "__main__":
    unittest.main()
