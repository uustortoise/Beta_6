from pathlib import Path

import pandas as pd

from scripts.validate_label_pack import ROOMS, validate_label_pack


def _write_pack_file(path: Path, room_frames: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        for room, frame in room_frames.items():
            frame.to_excel(writer, index=False, sheet_name=room)


def _base_room_frames() -> dict[str, pd.DataFrame]:
    ts = pd.date_range("2025-12-04 00:00:00", periods=6, freq="10s")
    out: dict[str, pd.DataFrame] = {}
    for room in ROOMS:
        label = "sleep" if room.lower() == "bedroom" else "unoccupied"
        out[room] = pd.DataFrame({"timestamp": ts, "activity": [label] * len(ts)})
    return out


def test_validate_label_pack_passes_valid_pack(tmp_path: Path):
    frames_day4 = _base_room_frames()
    frames_day5 = _base_room_frames()
    _write_pack_file(tmp_path / "HK0011_jessica_train_4dec2025.xlsx", frames_day4)
    _write_pack_file(tmp_path / "HK0011_jessica_train_5dec2025.xlsx", frames_day5)

    report = validate_label_pack(
        pack_dir=tmp_path,
        elder_id="HK0011_jessica",
        min_day=4,
        max_day=5,
        registry_path=None,
    )
    assert report["status"] == "pass"
    assert report["violations"] == []
    assert report["file_audit"]["selected_days"] == [4, 5]


def test_validate_label_pack_fails_with_schema_and_label_issues(tmp_path: Path):
    frames = _base_room_frames()
    frames.pop("Entrance")
    frames["Garage"] = pd.DataFrame(
        {"timestamp": ["bad_ts", "2025-12-04 00:00:10"], "activity": ["party", "unoccupied"]}
    )
    frames["Bedroom"] = pd.DataFrame(
        {"timestamp": ["not_a_time", "2025-12-04 00:00:10"], "activity": ["sleep", ""]}
    )
    _write_pack_file(tmp_path / "HK0011_jessica_train_4dec2025.xlsx", frames)

    report = validate_label_pack(
        pack_dir=tmp_path,
        elder_id="HK0011_jessica",
        min_day=4,
        max_day=4,
        registry_path=None,
    )
    violations = list(report["violations"])
    assert report["status"] == "fail"
    assert any(v.startswith("missing_room_sheet:") and ":Entrance" in v for v in violations)
    assert any(v.startswith("unknown_room_sheets:") for v in violations)
    assert any(v.startswith("timestamp_parse_fail:") and ":Bedroom:" in v for v in violations)
    assert any(v.startswith("empty_activity_rows:") and ":Bedroom:" in v for v in violations)

