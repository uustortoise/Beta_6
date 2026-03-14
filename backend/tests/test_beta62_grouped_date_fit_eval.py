import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_beta62_grouped_date_fit_eval as script
from ml.beta6.grouped_date_fit_eval import (
    _evaluate_room_candidate,
    _fit_room_candidate,
    _prepare_segmented_training_frame,
    _resolve_saved_candidate_artifacts,
    run_grouped_date_fit_eval,
)
from ml.registry import ModelRegistry


def _prepared_split_frame(
    *,
    start: str,
    room: str,
    activity: str,
    split: str,
    segment_role: str,
    segment_date: str,
    rows: int = 6,
) -> pd.DataFrame:
    ts = pd.date_range(start, periods=rows, freq="10s")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "co2": [800.0] * rows,
            "vibration": [0.0] * rows,
            "humidity": [40.0] * rows,
            "temperature": [24.0] * rows,
            "sound": [3.0] * rows,
            "motion": [1.0] * rows,
            "light": [200.0] * rows,
            "activity": [activity] * rows,
            "__segment_role": [segment_role] * rows,
            "__segment_date": [segment_date] * rows,
            "__segment_split": [split] * rows,
            "__source_path": [f"/tmp/{room}_{segment_date}.source"] * rows,
        }
    )


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _supervised_report(tmp_path: Path, *, room: str, split_paths: dict[str, Path]) -> tuple[Path, dict]:
    manifest_path = tmp_path / "grouped_manifest.json"
    manifest = {
        "schema_version": "beta62.grouped_date_supervised_manifest.v1",
        "resident_id": "HK0011_jessica",
        "target_rooms": [room],
        "segments": [
            {"role": "baseline", "date": "2025-12-04", "split": "train", "path": "/tmp/4dec.parquet"},
            {"role": "candidate", "date": "2026-03-09", "split": "holdout", "path": "/tmp/9mar.xlsx"},
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = {
        "schema_version": "beta62.grouped_date_supervised_report.v1",
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "resident_id": "HK0011_jessica",
        "target_rooms": [room.lower()],
        "room_reports": {
            room.lower(): {
                "split_summary": {
                    split: {
                        "artifact_path": str(path),
                    }
                    for split, path in split_paths.items()
                }
            }
        },
    }
    report_path = tmp_path / "grouped_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, report


def test_grouped_date_fit_eval_uses_explicit_splits_and_preserves_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    train_path = tmp_path / "prepared" / "HK0011_jessica_bathroom_train.parquet"
    val_path = tmp_path / "prepared" / "HK0011_jessica_bathroom_validation.parquet"
    calib_path = tmp_path / "prepared" / "HK0011_jessica_bathroom_calibration.parquet"
    holdout_path = tmp_path / "prepared" / "HK0011_jessica_bathroom_holdout.parquet"

    _write_parquet(
        train_path,
        pd.concat(
            [
                _prepared_split_frame(
                    start="2025-12-04 08:00:00",
                    room="Bathroom",
                    activity="bathroom_normal_use",
                    split="train",
                    segment_role="baseline",
                    segment_date="2025-12-04",
                ),
                _prepared_split_frame(
                    start="2026-03-09 08:00:00",
                    room="Bathroom",
                    activity="bathroom_normal_use",
                    split="train",
                    segment_role="candidate",
                    segment_date="2026-03-09",
                ),
            ],
            ignore_index=True,
        ),
    )
    _write_parquet(
        val_path,
        _prepared_split_frame(
            start="2025-12-05 08:00:00",
            room="Bathroom",
            activity="bathroom_normal_use",
            split="validation",
            segment_role="baseline",
            segment_date="2025-12-05",
        ),
    )
    _write_parquet(
        calib_path,
        _prepared_split_frame(
            start="2025-12-06 08:00:00",
            room="Bathroom",
            activity="bathroom_normal_use",
            split="calibration",
            segment_role="baseline",
            segment_date="2025-12-06",
        ),
    )
    _write_parquet(
        holdout_path,
        _prepared_split_frame(
            start="2026-03-01 08:00:00",
            room="Bathroom",
            activity="bathroom_normal_use",
            split="holdout",
            segment_role="candidate",
            segment_date="2026-03-01",
        ),
    )

    report_path, report = _supervised_report(
        tmp_path,
        room="Bathroom",
        split_paths={
            "train": train_path,
            "validation": val_path,
            "calibration": calib_path,
            "holdout": holdout_path,
        },
    )

    fit_calls = []
    eval_calls = []

    def _forbid_train_from_files(*args, **kwargs):
        raise AssertionError("UnifiedPipeline.train_from_files must not be called")

    monkeypatch.setattr("ml.pipeline.UnifiedPipeline.train_from_files", _forbid_train_from_files)

    def _fake_fit_room_candidate(*, room_name, split_frames, candidate_namespace, **kwargs):
        fit_calls.append(
            {
                "room_name": room_name,
                "candidate_namespace": candidate_namespace,
                "splits": sorted(split_frames.keys()),
                "train_dates": sorted(split_frames["train"]["__segment_date"].astype(str).unique().tolist()),
            }
        )
        assert sorted(split_frames.keys()) == ["calibration", "holdout", "train", "validation"]
        assert sorted(split_frames["train"]["__segment_date"].astype(str).unique().tolist()) == [
            "2025-12-04",
            "2026-03-09",
        ]
        assert sorted(split_frames["holdout"]["__segment_date"].astype(str).unique().tolist()) == ["2026-03-01"]
        return {
            "saved_version": 3,
            "fit_metrics": {"accuracy": 0.88, "macro_f1": 0.71},
            "candidate_artifact_paths": {
                "model": f"/tmp/{candidate_namespace}/{room_name}_v3_model.keras",
                "scaler": f"/tmp/{candidate_namespace}/{room_name}_v3_scaler.pkl",
            },
        }

    def _fake_evaluate_room_candidate(*, room_name, holdout_df, fit_result, **kwargs):
        eval_calls.append(
            {
                "room_name": room_name,
                "holdout_dates": sorted(holdout_df["__segment_date"].astype(str).unique().tolist()),
                "saved_version": fit_result["saved_version"],
            }
        )
        return {
            "accuracy": 0.67,
            "macro_f1": 0.54,
            "evaluated_dates": sorted(holdout_df["__segment_date"].astype(str).unique().tolist()),
        }

    monkeypatch.setattr("ml.beta6.grouped_date_fit_eval._fit_room_candidate", _fake_fit_room_candidate)
    monkeypatch.setattr("ml.beta6.grouped_date_fit_eval._evaluate_room_candidate", _fake_evaluate_room_candidate)

    payload = run_grouped_date_fit_eval(
        supervised_report=report,
        artifact_dir=tmp_path / "prepared",
        candidate_namespace="HK0011_jessica_candidate_grouped_date_test",
    )

    assert len(fit_calls) == 1
    assert len(eval_calls) == 1
    assert payload["schema_version"] == "beta62.grouped_date_fit_eval_report.v1"
    assert payload["candidate_namespace"] == "HK0011_jessica_candidate_grouped_date_test"
    assert payload["manifest"]["path"] == str(report_path.parent / "grouped_manifest.json")
    assert len(payload["manifest"]["sha256"]) == 64

    bathroom = payload["room_results"]["bathroom"]
    assert bathroom["split_counts"] == {
        "calibration": 6,
        "holdout": 6,
        "train": 12,
        "validation": 6,
    }
    assert bathroom["artifact_paths"]["train"] == str(train_path)
    assert bathroom["artifact_paths"]["holdout"] == str(holdout_path)
    assert bathroom["lineage"]["train"]["dates"] == ["2025-12-04", "2026-03-09"]
    assert bathroom["lineage"]["validation"]["dates"] == ["2025-12-05"]
    assert bathroom["candidate_artifact_paths"]["model"].endswith("Bathroom_v3_model.keras")
    assert bathroom["holdout_metrics"]["evaluated_dates"] == ["2026-03-01"]


def test_grouped_date_fit_eval_result_payload_handles_mixed_room_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    bathroom_train = tmp_path / "prepared" / "HK0011_jessica_bathroom_train.parquet"
    bathroom_holdout = tmp_path / "prepared" / "HK0011_jessica_bathroom_holdout.parquet"
    living_train = tmp_path / "prepared" / "HK0011_jessica_livingroom_train.parquet"
    living_holdout = tmp_path / "prepared" / "HK0011_jessica_livingroom_holdout.parquet"

    _write_parquet(
        bathroom_train,
        _prepared_split_frame(
            start="2025-12-04 08:00:00",
            room="Bathroom",
            activity="bathroom_normal_use",
            split="train",
            segment_role="baseline",
            segment_date="2025-12-04",
        ),
    )
    _write_parquet(
        bathroom_holdout,
        _prepared_split_frame(
            start="2026-03-08 08:00:00",
            room="Bathroom",
            activity="bathroom_normal_use",
            split="holdout",
            segment_role="candidate",
            segment_date="2026-03-08",
        ),
    )
    _write_parquet(
        living_train,
        _prepared_split_frame(
            start="2025-12-04 09:00:00",
            room="LivingRoom",
            activity="livingroom_normal_use",
            split="train",
            segment_role="baseline",
            segment_date="2025-12-04",
        ),
    )
    _write_parquet(
        living_holdout,
        _prepared_split_frame(
            start="2026-03-08 09:00:00",
            room="LivingRoom",
            activity="livingroom_normal_use",
            split="holdout",
            segment_role="candidate",
            segment_date="2026-03-08",
        ),
    )

    report = {
        "schema_version": "beta62.grouped_date_supervised_report.v1",
        "manifest_path": str(tmp_path / "manifest.json"),
        "manifest": {
            "schema_version": "beta62.grouped_date_supervised_manifest.v1",
            "resident_id": "HK0011_jessica",
            "target_rooms": ["Bathroom", "LivingRoom"],
            "segments": [],
        },
        "resident_id": "HK0011_jessica",
        "target_rooms": ["bathroom", "livingroom"],
        "room_reports": {
            "bathroom": {"split_summary": {"train": {"artifact_path": str(bathroom_train)}, "holdout": {"artifact_path": str(bathroom_holdout)}}},
            "livingroom": {"split_summary": {"train": {"artifact_path": str(living_train)}, "holdout": {"artifact_path": str(living_holdout)}}},
        },
    }

    monkeypatch.setattr(
        "ml.beta6.grouped_date_fit_eval._fit_room_candidate",
        lambda *, room_name, candidate_namespace, **kwargs: {
            "saved_version": 1,
            "fit_metrics": {"accuracy": 0.9 if room_name == "Bathroom" else 0.8},
            "candidate_artifact_paths": {"model": f"/tmp/{candidate_namespace}/{room_name}.keras"},
        },
    )
    monkeypatch.setattr(
        "ml.beta6.grouped_date_fit_eval._evaluate_room_candidate",
        lambda *, room_name, **kwargs: {
            "accuracy": 0.7 if room_name == "Bathroom" else 0.6,
            "macro_f1": 0.5 if room_name == "Bathroom" else 0.4,
        },
    )

    payload = run_grouped_date_fit_eval(
        supervised_report=report,
        artifact_dir=tmp_path / "prepared",
        candidate_namespace="HK0011_jessica_candidate_grouped_date_test2",
    )

    assert sorted(payload["room_results"].keys()) == ["bathroom", "livingroom"]
    assert payload["room_results"]["bathroom"]["holdout_metrics"]["accuracy"] == 0.7
    assert payload["room_results"]["livingroom"]["holdout_metrics"]["macro_f1"] == 0.4


def test_grouped_date_fit_eval_accepts_manifest_plus_prepared_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    train_path = tmp_path / "prepared" / "HK0011_jessica_kitchen_train.parquet"
    holdout_path = tmp_path / "prepared" / "HK0011_jessica_kitchen_holdout.parquet"
    _write_parquet(
        train_path,
        _prepared_split_frame(
            start="2025-12-04 12:00:00",
            room="Kitchen",
            activity="kitchen_normal_use",
            split="train",
            segment_role="baseline",
            segment_date="2025-12-04",
        ),
    )
    _write_parquet(
        holdout_path,
        _prepared_split_frame(
            start="2026-03-08 12:00:00",
            room="Kitchen",
            activity="kitchen_normal_use",
            split="holdout",
            segment_role="candidate",
            segment_date="2026-03-08",
        ),
    )

    manifest = {
        "schema_version": "beta62.grouped_date_supervised_manifest.v1",
        "resident_id": "HK0011_jessica",
        "target_rooms": ["Kitchen"],
        "sequence_length_by_room": {"kitchen": 4},
        "segments": [
            {"role": "baseline", "date": "2025-12-04", "split": "train", "path": "/tmp/4dec.parquet"},
            {"role": "candidate", "date": "2026-03-08", "split": "holdout", "path": "/tmp/8mar.xlsx"},
        ],
    }

    monkeypatch.setattr(
        "ml.beta6.grouped_date_fit_eval._fit_room_candidate",
        lambda *, room_name, seq_length, **kwargs: {
            "saved_version": 2,
            "fit_metrics": {"accuracy": 0.82},
            "candidate_artifact_paths": {"model": f"/tmp/{room_name}_v2.keras"},
            "seq_length_seen": seq_length,
        },
    )
    monkeypatch.setattr(
        "ml.beta6.grouped_date_fit_eval._evaluate_room_candidate",
        lambda **kwargs: {"accuracy": 0.61, "macro_f1": 0.45},
    )

    payload = run_grouped_date_fit_eval(
        manifest=manifest,
        artifact_dir=tmp_path / "prepared",
        candidate_namespace="HK0011_jessica_candidate_manifest_only",
    )

    assert payload["manifest"]["resident_id"] == "HK0011_jessica"
    assert payload["room_results"]["kitchen"]["artifact_paths"]["train"] == str(train_path)
    assert payload["room_results"]["kitchen"]["artifact_paths"]["holdout"] == str(holdout_path)


def test_grouped_date_fit_eval_accepts_embedded_report_manifest_without_manifest_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    train_path = tmp_path / "prepared" / "HK0011_jessica_bedroom_train.parquet"
    holdout_path = tmp_path / "prepared" / "HK0011_jessica_bedroom_holdout.parquet"
    _write_parquet(
        train_path,
        _prepared_split_frame(
            start="2025-12-04 00:00:00",
            room="Bedroom",
            activity="sleep",
            split="train",
            segment_role="baseline",
            segment_date="2025-12-04",
        ),
    )
    _write_parquet(
        holdout_path,
        _prepared_split_frame(
            start="2026-03-08 00:00:00",
            room="Bedroom",
            activity="sleep",
            split="holdout",
            segment_role="candidate",
            segment_date="2026-03-08",
        ),
    )

    report = {
        "schema_version": "beta62.grouped_date_supervised_report.v1",
        "manifest": {
            "schema_version": "beta62.grouped_date_supervised_manifest.v1",
            "resident_id": "HK0011_jessica",
            "target_rooms": ["bedroom"],
            "sequence_length_by_room": {"bedroom": 5},
            "segments": [
                {"role": "baseline", "date": "2025-12-04", "split": "train", "path": "/tmp/4dec.parquet"},
                {"role": "candidate", "date": "2026-03-08", "split": "holdout", "path": "/tmp/8mar.xlsx"},
            ],
            "notes": ["embedded-only contract"],
        },
        "resident_id": "HK0011_jessica",
        "target_rooms": ["bedroom"],
        "room_reports": {
            "bedroom": {
                "split_summary": {
                    "train": {"artifact_path": str(train_path)},
                    "holdout": {"artifact_path": str(holdout_path)},
                }
            }
        },
    }

    monkeypatch.setattr(
        "ml.beta6.grouped_date_fit_eval._fit_room_candidate",
        lambda *, room_name, seq_length, **kwargs: {
            "saved_version": 4,
            "fit_metrics": {"accuracy": 0.91},
            "candidate_artifact_paths": {"model": f"/tmp/{room_name}_v4.keras"},
            "seq_length_seen": seq_length,
        },
    )
    monkeypatch.setattr(
        "ml.beta6.grouped_date_fit_eval._evaluate_room_candidate",
        lambda **kwargs: {"accuracy": 0.77, "macro_f1": 0.63},
    )

    payload = run_grouped_date_fit_eval(
        supervised_report=report,
        artifact_dir=tmp_path / "prepared",
        candidate_namespace="HK0011_jessica_candidate_embedded_report_only",
    )

    assert payload["manifest"]["resident_id"] == "HK0011_jessica"
    assert len(payload["manifest"]["sha256"]) == 64
    assert payload["room_results"]["bedroom"]["artifact_paths"]["holdout"] == str(holdout_path)


def test_grouped_date_fit_eval_resolves_latest_complete_candidate_artifacts(tmp_path: Path):
    registry = ModelRegistry(str(tmp_path))
    candidate_namespace = "HK0011_jessica_candidate_reused"
    room_name = "LivingRoom"
    models_dir = registry.get_models_dir(candidate_namespace)

    versions_path = models_dir / f"{room_name}_versions.json"
    versions_path.write_text(
        json.dumps(
            {
                "current_version": 0,
                "versions": [
                    {"version": 1, "promoted": False},
                    {"version": 2, "promoted": False},
                ],
            }
        ),
        encoding="utf-8",
    )

    (models_dir / f"{room_name}_v1_model.keras").write_text("stub", encoding="utf-8")
    for suffix in ["model.keras", "scaler.pkl", "label_encoder.pkl", "thresholds.json"]:
        (models_dir / f"{room_name}_v2_{suffix}").write_text("stub", encoding="utf-8")

    resolved = _resolve_saved_candidate_artifacts(
        registry=registry,
        candidate_namespace=candidate_namespace,
        room_name=room_name,
        requested_saved_version=1,
    )

    assert resolved["requested_saved_version"] == 1
    assert resolved["resolved_saved_version"] == 2
    assert resolved["artifact_paths"]["model"].endswith("LivingRoom_v2_model.keras")
    assert resolved["artifact_paths"]["scaler"].endswith("LivingRoom_v2_scaler.pkl")
    assert resolved["artifact_paths"]["label_encoder"].endswith("LivingRoom_v2_label_encoder.pkl")


def test_grouped_date_fit_eval_preprocesses_train_segments_independently():
    train_df = pd.concat(
        [
            _prepared_split_frame(
                start="2025-12-04 08:00:00",
                room="Bathroom",
                activity="bathroom_normal_use",
                split="train",
                segment_role="baseline",
                segment_date="2025-12-04",
            ),
            _prepared_split_frame(
                start="2026-03-09 08:00:00",
                room="Bathroom",
                activity="bathroom_normal_use",
                split="train",
                segment_role="candidate",
                segment_date="2026-03-09",
            ),
        ],
        ignore_index=True,
    )

    class DummyPlatform:
        sensor_columns = ["co2", "vibration", "humidity", "temperature", "sound", "motion", "light"]

        def __init__(self):
            self.preprocess_calls = []
            self.scale_calls = []

        def preprocess_without_scaling(self, df, room_name, is_training=False, apply_denoising=False):
            self.preprocess_calls.append(
                {
                    "room_name": room_name,
                    "rows": len(df),
                    "start": pd.to_datetime(df["timestamp"]).min().date().isoformat(),
                    "end": pd.to_datetime(df["timestamp"]).max().date().isoformat(),
                    "is_training": bool(is_training),
                }
            )
            return df[["timestamp"] + self.sensor_columns + ["activity"]].copy()

        def apply_scaling(self, df, room_name, is_training=False, scaler_fit_range=None):
            self.scale_calls.append(
                {
                    "room_name": room_name,
                    "rows": len(df),
                    "is_training": bool(is_training),
                    "fit_sample_count": None if scaler_fit_range is None else scaler_fit_range.get("fit_sample_count"),
                }
            )
            scaled = df.copy()
            scaled["activity_encoded"] = 0
            return scaled[["timestamp"] + self.sensor_columns + ["activity_encoded"]]

    platform = DummyPlatform()

    processed = _prepare_segmented_training_frame(
        platform=platform,
        room_name="Bathroom",
        train_df=train_df,
    )

    assert [call["start"] for call in platform.preprocess_calls] == ["2025-12-04", "2026-03-09"]
    assert [call["rows"] for call in platform.preprocess_calls] == [6, 6]
    assert platform.scale_calls == [
        {
            "room_name": "Bathroom",
            "rows": 12,
            "is_training": True,
            "fit_sample_count": 12,
        }
    ]
    assert len(processed) == 12


def test_grouped_date_fit_eval_evaluates_holdout_with_segmented_sequences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class DummyRegistry:
        def load_room_model(self, path, room_name, compile_model=False):
            return {"room_name": room_name, "path": path}

    class DummyPlatform:
        sensor_columns = ["co2", "vibration", "humidity", "temperature", "sound", "motion", "light"]

        def __init__(self):
            self.scalers = {}

    class DummyEncoder:
        classes_ = np.array(["bathroom_normal_use", "unoccupied"], dtype=object)

    holdout_df = _prepared_split_frame(
        start="2026-03-08 08:00:00",
        room="Bathroom",
        activity="bathroom_normal_use",
        split="holdout",
        segment_role="candidate",
        segment_date="2026-03-08",
    )

    fit_result = {
        "registry": DummyRegistry(),
        "platform": DummyPlatform(),
        "candidate_artifact_paths": {
            "model": str(tmp_path / "Bathroom_v1_model.keras"),
            "scaler": str(tmp_path / "Bathroom_v1_scaler.pkl"),
            "label_encoder": str(tmp_path / "Bathroom_v1_label_encoder.pkl"),
        },
        "seq_length": 3,
        "saved_version": 1,
    }

    monkeypatch.setattr(
        "ml.beta6.grouped_date_fit_eval._prepare_segmented_holdout_frame",
        lambda **kwargs: holdout_df[["timestamp", "co2", "vibration", "humidity", "temperature", "sound", "motion", "light", "activity"]].copy(),
    )
    monkeypatch.setattr(
        "ml.beta6.grouped_date_fit_eval.joblib.load",
        lambda path: object() if str(path).endswith("_scaler.pkl") else DummyEncoder(),
    )
    monkeypatch.setattr(
        "ml.beta6.grouped_date_fit_eval.evaluate_model",
        lambda **kwargs: {
            "status": "completed",
            "summary": {"num_folds": 0},
            "sequence_count": int(len(kwargs["X_seq"])),
        },
    )

    report = _evaluate_room_candidate(
        room_name="Bathroom",
        holdout_df=holdout_df,
        fit_result=fit_result,
    )

    assert report["status"] == "completed"
    assert report["sequence_count"] == 4


def test_fit_room_candidate_passes_explicit_validation_and_calibration_sequences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    split_frames = {
        "train": _prepared_split_frame(
            start="2025-12-04 08:00:00",
            room="Bathroom",
            activity="bathroom_normal_use",
            split="train",
            segment_role="baseline",
            segment_date="2025-12-04",
        ),
        "validation": _prepared_split_frame(
            start="2025-12-05 08:00:00",
            room="Bathroom",
            activity="bathroom_normal_use",
            split="validation",
            segment_role="baseline",
            segment_date="2025-12-05",
        ),
        "calibration": _prepared_split_frame(
            start="2025-12-06 08:00:00",
            room="Bathroom",
            activity="bathroom_normal_use",
            split="calibration",
            segment_role="baseline",
            segment_date="2025-12-06",
        ),
        "holdout": _prepared_split_frame(
            start="2026-03-08 08:00:00",
            room="Bathroom",
            activity="bathroom_normal_use",
            split="holdout",
            segment_role="candidate",
            segment_date="2026-03-08",
        ),
    }

    captured = {}

    class DummyPlatform:
        sensor_columns = ["co2", "vibration", "humidity", "temperature", "sound", "motion", "light"]

        def __init__(self):
            self.scalers = {}

        def preprocess_without_scaling(self, df, room_name, is_training=False, apply_denoising=False):
            return df.copy()

        def apply_scaling(self, df, room_name, is_training=False, scaler_fit_range=None):
            if is_training:
                self.scalers[room_name] = object()
            scaled = df.copy()
            scaled["activity_encoded"] = 0
            return scaled

    class DummyTrainingPipeline:
        def __init__(self, platform, registry):
            self.platform = platform
            self.registry = registry
            self.augment_training_data = lambda *args: args[3:6]

        def train_room(self, **kwargs):
            captured["explicit_validation_data"] = kwargs.get("explicit_validation_data")
            captured["explicit_calibration_data"] = kwargs.get("explicit_calibration_data")
            return {
                "saved_version": 3,
                "validation_source": "explicit_split",
                "calibration_source": "explicit_split",
                "validation_samples": int(len(kwargs["explicit_validation_data"][1])),
                "calibration_samples": int(len(kwargs["explicit_calibration_data"][1])),
            }

    monkeypatch.setattr("ml.beta6.grouped_date_fit_eval.ElderlyCarePlatform", DummyPlatform)
    monkeypatch.setattr("ml.beta6.grouped_date_fit_eval.TrainingPipeline", DummyTrainingPipeline)
    monkeypatch.setattr("ml.beta6.grouped_date_fit_eval.ModelRegistry", lambda _: object())
    monkeypatch.setattr(
        "ml.beta6.grouped_date_fit_eval._resolve_saved_candidate_artifacts",
        lambda **kwargs: {
            "requested_saved_version": 3,
            "resolved_saved_version": 3,
            "artifact_paths": {
                "model": str(tmp_path / "Bathroom_v3_model.keras"),
                "scaler": str(tmp_path / "Bathroom_v3_scaler.pkl"),
                "label_encoder": str(tmp_path / "Bathroom_v3_label_encoder.pkl"),
            },
        },
    )

    fit_result = _fit_room_candidate(
        room_name="Bathroom",
        split_frames=split_frames,
        candidate_namespace="HK0011_jessica_candidate_grouped_date_explicit",
        backend_dir=tmp_path,
        seq_length=3,
    )

    explicit_validation = captured["explicit_validation_data"]
    explicit_calibration = captured["explicit_calibration_data"]

    assert explicit_validation is not None
    assert explicit_calibration is not None
    assert explicit_validation[0].shape[0] > 0
    assert explicit_validation[1].shape[0] > 0
    assert explicit_validation[2].shape[0] == explicit_validation[1].shape[0]
    assert explicit_calibration[0].shape[0] > 0
    assert explicit_calibration[1].shape[0] > 0
    assert fit_result["fit_metrics"]["validation_source"] == "explicit_split"
    assert fit_result["fit_metrics"]["calibration_source"] == "explicit_split"
    assert fit_result["fit_metrics"]["validation_samples"] == int(explicit_validation[1].shape[0])
    assert fit_result["fit_metrics"]["calibration_samples"] == int(explicit_calibration[1].shape[0])


def test_grouped_date_fit_eval_cli_writes_result_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    report_path = tmp_path / "grouped_report.json"
    artifact_dir = tmp_path / "prepared"
    output_path = tmp_path / "fit_eval_result.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "beta62.grouped_date_supervised_report.v1",
                "manifest": {
                    "schema_version": "beta62.grouped_date_supervised_manifest.v1",
                    "resident_id": "HK0011_jessica",
                    "target_rooms": ["Bathroom"],
                    "segments": [],
                },
                "resident_id": "HK0011_jessica",
                "target_rooms": ["bathroom"],
                "room_reports": {"bathroom": {"split_summary": {}}},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.run_beta62_grouped_date_fit_eval.run_grouped_date_fit_eval",
        lambda **kwargs: {
            "schema_version": "beta62.grouped_date_fit_eval_report.v1",
            "candidate_namespace": kwargs["candidate_namespace"],
            "room_results": {},
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_beta62_grouped_date_fit_eval.py",
            "--supervised-report",
            str(report_path),
            "--artifact-dir",
            str(artifact_dir),
            "--candidate-namespace",
            "HK0011_jessica_candidate_cli",
            "--output",
            str(output_path),
        ],
    )

    rc = script.main()

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "beta62.grouped_date_fit_eval_report.v1"
    assert payload["candidate_namespace"] == "HK0011_jessica_candidate_cli"
