from pathlib import Path

import pandas as pd

from scripts.diff_label_pack import ROOMS, diff_label_pack


def _write_label_file(path: Path, by_room: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        for room in ROOMS:
            frame = by_room.get(room)
            if frame is None:
                ts = pd.date_range("2025-12-07 00:00:00", periods=6, freq="10s")
                frame = pd.DataFrame({"timestamp": ts, "activity": ["unoccupied"] * len(ts)})
            frame.to_excel(writer, sheet_name=room, index=False)


def test_diff_label_pack_reports_room_day_changes(tmp_path: Path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"

    ts = pd.date_range("2025-12-07 08:00:00", periods=6, freq="10s")
    base_lr = pd.DataFrame({"timestamp": ts, "activity": ["unoccupied"] * len(ts)})
    cand_lr = pd.DataFrame(
        {
            "timestamp": ts,
            "activity": [
                "unoccupied",
                "livingroom_normal_use",
                "livingroom_normal_use",
                "unoccupied",
                "unoccupied",
                "unoccupied",
            ],
        }
    )
    baseline_rooms = {"LivingRoom": base_lr}
    candidate_rooms = {"LivingRoom": cand_lr}

    _write_label_file(baseline_dir / "HK0011_jessica_train_7dec2025.xlsx", baseline_rooms)
    _write_label_file(candidate_dir / "HK0011_jessica_train_7dec2025.xlsx", candidate_rooms)

    report = diff_label_pack(
        baseline_dir=baseline_dir,
        candidate_dir=candidate_dir,
        elder_id="HK0011_jessica",
        min_day=7,
        max_day=7,
    )

    assert report["summary"]["days_compared"] == 1
    assert report["summary"]["windows_changed_total"] == 2
    assert report["summary"]["minutes_changed_total"] == 2 * (10.0 / 60.0)

    rows = report["rows"]
    target = [r for r in rows if r["day"] == 7 and r["room"] == "livingroom"][0]
    assert target["windows_changed"] == 2
    assert target["transition_counts"]["unoccupied->livingroom_normal_use"] == 2
    assert target["episodes_candidate"] >= target["episodes_baseline"]

