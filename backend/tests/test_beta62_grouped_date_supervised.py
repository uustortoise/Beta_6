import json
from pathlib import Path

import pandas as pd

from ml.beta6.grouped_date_supervised import run_grouped_date_supervised
from scripts import run_beta62_grouped_date_supervised as script


ROOMS = ["Bedroom", "LivingRoom", "Kitchen", "Bathroom", "Entrance"]


def _room_frame(
    start: str,
    *,
    room: str,
    activity: str,
    rows: int = 6,
    motion: float = 0.0,
    light: float = 0.0,
    sound: float = 3.0,
    co2: float = 800.0,
    humidity: float = 35.0,
) -> pd.DataFrame:
    ts = pd.date_range(start, periods=rows, freq="10s")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "co2": [co2] * rows,
            "vibration": [0.0] * rows,
            "humidity": [humidity] * rows,
            "temperature": [24.0] * rows,
            "sound": [sound] * rows,
            "motion": [motion] * rows,
            "light": [light] * rows,
            "activity": [activity] * rows,
            "room": [room] * rows,
        }
    )


def _write_day_workbook(path: Path, by_room: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        for room in ROOMS:
            frame = by_room.get(room)
            if frame is None:
                frame = _room_frame(
                    "2026-03-06 00:00:00",
                    room=room,
                    activity="sleep" if room.lower() == "bedroom" else "unoccupied",
                ).drop(columns=["room"])
            else:
                frame = frame.drop(columns=["room"], errors="ignore")
            frame.to_excel(writer, sheet_name=room, index=False)


def _write_day_parquet(path: Path, by_room: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for room in ROOMS:
        frame = by_room.get(room)
        if frame is None:
            frame = _room_frame(
                "2025-12-04 00:00:00",
                room=room,
                activity="sleep" if room.lower() == "bedroom" else "unoccupied",
            )
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)


def test_grouped_date_supervised_preserves_discontinuous_boundaries_and_splits(tmp_path: Path):
    baseline_path = tmp_path / "HK0011_jessica_train_4dec2025.parquet"
    holdout_path = tmp_path / "HK0011_jessica_train_6mar2026.xlsx"

    _write_day_parquet(
        baseline_path,
        {
            "Bathroom": _room_frame("2025-12-04 08:00:00", room="Bathroom", activity="bathroom_normal_use", motion=1.0, light=220.0, humidity=42.0),
            "LivingRoom": _room_frame("2025-12-04 09:00:00", room="LivingRoom", activity="livingroom_normal_use", motion=1.0, light=650.0, sound=4.5, co2=3300.0),
        },
    )
    _write_day_workbook(
        holdout_path,
        {
            "Bathroom": _room_frame("2026-03-06 08:00:00", room="Bathroom", activity="bathroom_normal_use", motion=1.0, light=240.0, humidity=44.0),
            "LivingRoom": _room_frame("2026-03-06 09:00:00", room="LivingRoom", activity="livingroom_normal_use", motion=1.0, light=700.0, sound=4.6, co2=3400.0),
        },
    )

    manifest = {
        "schema_version": "beta62.grouped_date_supervised_manifest.v1",
        "resident_id": "HK0011_jessica",
        "target_rooms": ["Bathroom", "LivingRoom"],
        "sequence_length_by_room": {"bathroom": 3, "livingroom": 3},
        "segments": [
            {"role": "baseline", "date": "2025-12-04", "split": "train", "path": str(baseline_path)},
            {"role": "candidate", "date": "2026-03-06", "split": "holdout", "path": str(holdout_path)},
        ],
    }

    report = run_grouped_date_supervised(manifest)

    assert [segment["split"] for segment in report["manifest"]["segments"]] == ["train", "holdout"]
    assert [segment["date"] for segment in report["manifest"]["segments"]] == ["2025-12-04", "2026-03-06"]
    assert report["manifest_summary"]["split_counts"] == {"holdout": 1, "train": 1}
    assert [row["split"] for row in report["room_reports"]["livingroom"]["grouped_by_date"]] == ["train", "holdout"]
    assert report["room_reports"]["bathroom"]["split_summary"]["train"]["sequence_count"] == 4


def test_grouped_date_supervised_keeps_late_train_segment_visible(tmp_path: Path):
    baseline_a = tmp_path / "HK0011_jessica_train_4dec2025.parquet"
    baseline_b = tmp_path / "HK0011_jessica_train_5dec2025.parquet"
    late_candidate = tmp_path / "HK0011_jessica_train_9mar2026.xlsx"

    _write_day_parquet(
        baseline_a,
        {
            "Bathroom": _room_frame("2025-12-04 08:00:00", room="Bathroom", activity="bathroom_normal_use", motion=1.0, light=220.0, humidity=42.0),
        },
    )
    _write_day_parquet(
        baseline_b,
        {
            "Bathroom": _room_frame("2025-12-05 08:00:00", room="Bathroom", activity="bathroom_normal_use", motion=1.0, light=240.0, humidity=43.0),
        },
    )
    _write_day_workbook(
        late_candidate,
        {
            "Bathroom": _room_frame("2026-03-09 08:00:00", room="Bathroom", activity="bathroom_normal_use", motion=1.0, light=230.0, humidity=43.0),
        },
    )

    manifest = {
        "schema_version": "beta62.grouped_date_supervised_manifest.v1",
        "resident_id": "HK0011_jessica",
        "target_rooms": ["Bathroom"],
        "sequence_length_by_room": {"bathroom": 4},
        "segments": [
            {"role": "baseline", "date": "2025-12-04", "split": "train", "path": str(baseline_a)},
            {"role": "baseline", "date": "2025-12-05", "split": "validation", "path": str(baseline_b)},
            {"role": "candidate", "date": "2026-03-09", "split": "train", "path": str(late_candidate)},
        ],
    }

    report = run_grouped_date_supervised(manifest)

    train_summary = report["room_reports"]["bathroom"]["split_summary"]["train"]
    assert train_summary["segment_count"] == 2
    assert train_summary["dates"] == ["2025-12-04", "2026-03-09"]
    assert train_summary["sequence_count"] == 6
    assert report["room_reports"]["bathroom"]["grouped_by_date"][-1]["date"] == "2026-03-09"


def test_grouped_date_supervised_materializes_split_artifacts_with_lineage(tmp_path: Path):
    baseline_path = tmp_path / "HK0011_jessica_train_4dec2025.parquet"
    candidate_path = tmp_path / "HK0011_jessica_train_8mar2026.xlsx"
    output_dir = tmp_path / "artifacts"

    _write_day_parquet(
        baseline_path,
        {
            "LivingRoom": _room_frame("2025-12-04 09:00:00", room="LivingRoom", activity="livingroom_normal_use", motion=1.0, light=650.0, sound=4.5, co2=3300.0),
        },
    )
    _write_day_workbook(
        candidate_path,
        {
            "LivingRoom": _room_frame("2026-03-08 09:00:00", room="LivingRoom", activity="livingroom_normal_use", motion=1.0, light=690.0, sound=4.7, co2=3450.0),
        },
    )

    manifest = {
        "schema_version": "beta62.grouped_date_supervised_manifest.v1",
        "resident_id": "HK0011_jessica",
        "target_rooms": ["LivingRoom"],
        "sequence_length_by_room": {"livingroom": 2},
        "segments": [
            {"role": "baseline", "date": "2025-12-04", "split": "train", "path": str(baseline_path)},
            {"role": "candidate", "date": "2026-03-08", "split": "holdout", "path": str(candidate_path)},
        ],
    }

    report = run_grouped_date_supervised(manifest, artifact_dir=output_dir)

    train_artifact = Path(report["room_reports"]["livingroom"]["split_summary"]["train"]["artifact_path"])
    holdout_artifact = Path(report["room_reports"]["livingroom"]["split_summary"]["holdout"]["artifact_path"])
    assert train_artifact.exists()
    assert holdout_artifact.exists()

    train_df = pd.read_parquet(train_artifact)
    holdout_df = pd.read_parquet(holdout_artifact)
    assert sorted(train_df["__segment_date"].unique().tolist()) == ["2025-12-04"]
    assert sorted(holdout_df["__segment_date"].unique().tolist()) == ["2026-03-08"]
    assert sorted(holdout_df["__segment_split"].unique().tolist()) == ["holdout"]
    assert sorted(holdout_df["__segment_role"].unique().tolist()) == ["candidate"]


def test_grouped_date_supervised_report_embeds_full_manifest_contract(tmp_path: Path):
    baseline_path = tmp_path / "HK0011_jessica_train_4dec2025.parquet"
    candidate_path = tmp_path / "HK0011_jessica_train_8mar2026.xlsx"

    _write_day_parquet(
        baseline_path,
        {
            "LivingRoom": _room_frame("2025-12-04 09:00:00", room="LivingRoom", activity="livingroom_normal_use", motion=1.0, light=650.0, sound=4.5, co2=3300.0),
        },
    )
    _write_day_workbook(
        candidate_path,
        {
            "LivingRoom": _room_frame("2026-03-08 09:00:00", room="LivingRoom", activity="livingroom_normal_use", motion=1.0, light=690.0, sound=4.7, co2=3450.0),
        },
    )

    manifest = {
        "schema_version": "beta62.grouped_date_supervised_manifest.v1",
        "resident_id": "HK0011_jessica",
        "target_rooms": ["LivingRoom"],
        "sequence_length_by_room": {"livingroom": 2},
        "segments": [
            {"role": "baseline", "date": "2025-12-04", "split": "train", "path": str(baseline_path)},
            {"role": "candidate", "date": "2026-03-08", "split": "holdout", "path": str(candidate_path)},
        ],
        "notes": ["room-scoped staged subset"],
    }

    report = run_grouped_date_supervised(manifest)

    assert report["manifest"] == {
        "schema_version": "beta62.grouped_date_supervised_manifest.v1",
        "resident_id": "HK0011_jessica",
        "target_rooms": ["livingroom"],
        "sequence_length_by_room": {"livingroom": 2},
        "segments": [
            {"role": "baseline", "date": "2025-12-04", "split": "train", "path": str(baseline_path.resolve())},
            {"role": "candidate", "date": "2026-03-08", "split": "holdout", "path": str(candidate_path.resolve())},
        ],
        "notes": ["room-scoped staged subset"],
    }


def test_grouped_date_supervised_cli_emits_grouped_report(tmp_path: Path, monkeypatch):
    baseline_path = tmp_path / "HK0011_jessica_train_4dec2025.parquet"
    candidate_path = tmp_path / "HK0011_jessica_train_8mar2026.xlsx"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "grouped_date_supervised_report.json"
    artifact_dir = tmp_path / "prepared"

    _write_day_parquet(
        baseline_path,
        {
            "Bathroom": _room_frame("2025-12-04 08:00:00", room="Bathroom", activity="bathroom_normal_use", motion=1.0, light=220.0, humidity=42.0),
        },
    )
    _write_day_workbook(
        candidate_path,
        {
            "Bathroom": _room_frame("2026-03-08 08:00:00", room="Bathroom", activity="bathroom_normal_use", motion=1.0, light=230.0, humidity=43.0),
        },
    )

    manifest = {
        "schema_version": "beta62.grouped_date_supervised_manifest.v1",
        "resident_id": "HK0011_jessica",
        "target_rooms": ["Bathroom"],
        "sequence_length_by_room": {"bathroom": 2},
        "segments": [
            {"role": "baseline", "date": "2025-12-04", "split": "train", "path": str(baseline_path)},
            {"role": "candidate", "date": "2026-03-08", "split": "holdout", "path": str(candidate_path)},
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_beta62_grouped_date_supervised.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    rc = script.main()

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "beta62.grouped_date_supervised_report.v1"
    assert payload["manifest_summary"]["segment_count"] == 2
    assert payload["manifest_summary"]["split_counts"] == {"holdout": 1, "train": 1}
    assert payload["room_reports"]["bathroom"]["grouped_by_date"][1]["split"] == "holdout"
