import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from scripts.run_event_first_smoke import run_smoke


def _write_day_file(path: Path, *, livingroom_occupied_windows: int = 3, total_windows: int = 10) -> None:
    ts = pd.date_range("2025-12-07 08:00:00", periods=total_windows, freq="10s")
    livingroom_labels = ["unoccupied"] * total_windows
    for i in range(min(livingroom_occupied_windows, total_windows)):
        livingroom_labels[i] = "livingroom_normal_use"
    with pd.ExcelWriter(path) as writer:
        for room in ["Bedroom", "Kitchen", "Bathroom", "Entrance"]:
            pd.DataFrame({"timestamp": ts, "activity": ["unoccupied"] * total_windows}).to_excel(
                writer, sheet_name=room, index=False
            )
        pd.DataFrame({"timestamp": ts, "activity": livingroom_labels}).to_excel(
            writer, sheet_name="LivingRoom", index=False
        )


def test_run_smoke_passes_when_checks_satisfied(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_day_file(data_dir / "HK0011_jessica_train_7dec2025.xlsx", livingroom_occupied_windows=4)

    expectation_path = tmp_path / "expect.yaml"
    expectation_path.write_text(
        yaml.safe_dump(
            {
                "smoke": {
                    "require_data_continuity_audit": True,
                    "require_label_corrections_summary": True,
                    "min_changed_minutes_total": 10.0,
                    "min_room_day_occupied_rate": {
                        "livingroom": {"day": 7, "min_rate": 0.20},
                    },
                }
            }
        )
    )
    diff_path = tmp_path / "diff.json"
    diff_path.write_text(json.dumps({"summary": {"minutes_changed_total": 30.0}}))

    def _fake_run(cmd, capture_output, text):  # noqa: ANN001
        output_idx = cmd.index("--output")
        report_path = Path(cmd[output_idx + 1])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "data_continuity_audit": {"missing_days_in_requested_window": []},
                    "label_corrections": {"load": {"enabled": True}, "apply": {"applied_windows": 1}},
                    "splits": [{"test_day": 7}],
                }
            )
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("scripts.run_event_first_smoke.subprocess.run", _fake_run)

    out = run_smoke(
        data_dir=data_dir,
        elder_id="HK0011_jessica",
        day=7,
        seed=11,
        expectation_config=expectation_path,
        diff_report=diff_path,
        output=tmp_path / "smoke.json",
        train_context_days=1,
    )
    assert out["status"] == "pass"
    assert out["blocking_reasons"] == []


def test_run_smoke_blocks_when_changed_minutes_evidence_missing(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_day_file(data_dir / "HK0011_jessica_train_7dec2025.xlsx", livingroom_occupied_windows=3)

    expectation_path = tmp_path / "expect.yaml"
    expectation_path.write_text(yaml.safe_dump({"smoke": {"min_changed_minutes_total": 60.0}}))
    diff_path = tmp_path / "diff.json"
    diff_path.write_text(json.dumps({"summary": {"minutes_changed_total": 5.0}}))

    def _fake_run(cmd, capture_output, text):  # noqa: ANN001
        output_idx = cmd.index("--output")
        report_path = Path(cmd[output_idx + 1])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "data_continuity_audit": {},
                    "label_corrections": {"load": {"enabled": True}, "apply": {}},
                    "splits": [{"test_day": 7}],
                }
            )
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("scripts.run_event_first_smoke.subprocess.run", _fake_run)

    out = run_smoke(
        data_dir=data_dir,
        elder_id="HK0011_jessica",
        day=7,
        seed=11,
        expectation_config=expectation_path,
        diff_report=diff_path,
        output=tmp_path / "smoke.json",
        train_context_days=1,
    )
    assert out["status"] == "fail"
    assert "insufficient_changed_minutes_evidence" in out["blocking_reasons"]

