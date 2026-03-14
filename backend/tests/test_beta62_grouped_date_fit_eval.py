import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_beta62_grouped_date_fit_eval as script
from ml.beta6.grouped_date_fit_eval import run_grouped_date_fit_eval


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
