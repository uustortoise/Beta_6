import json
from datetime import date
from pathlib import Path

import pandas as pd

from scripts import run_active_learning_triage as script


def test_active_learning_triage_script_writes_outputs(tmp_path: Path, monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "room": "bedroom",
                "activity": "sleep",
                "confidence": 0.2,
                "predicted_label": "sleep",
                "baseline_label": "nap",
            },
            {
                "candidate_id": "c2",
                "room": "livingroom",
                "activity": "out",
                "confidence": 0.4,
                "predicted_label": "out",
                "baseline_label": "out",
            },
            {
                "candidate_id": "c3",
                "room": "kitchen",
                "activity": "nap",
                "confidence": 0.1,
                "predicted_label": "nap",
                "baseline_label": "sleep",
            },
        ]
    )
    input_csv = tmp_path / "candidates.csv"
    output_csv = tmp_path / "queue.csv"
    report_json = tmp_path / "report.json"
    frame.to_csv(input_csv, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_active_learning_triage.py",
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--report-json",
            str(report_json),
        ],
    )
    rc = script.main()
    assert rc == 0
    assert output_csv.exists()
    assert report_json.exists()


def test_active_learning_triage_script_reports_training_signal_counts(tmp_path: Path, monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "corr-1",
                "room": "bedroom",
                "activity": "sleep",
                "confidence": 0.42,
                "predicted_label": "nap",
                "baseline_label": "sleep",
                "corrected_event": True,
                "boundary_start_target": 1,
                "boundary_end_target": 1,
                "hard_negative_flag": True,
                "residual_review_flag": True,
                "residual_review_rows": 2,
            },
            {
                "candidate_id": "plain-1",
                "room": "bedroom",
                "activity": "sleep",
                "confidence": 0.31,
                "predicted_label": "sleep",
                "baseline_label": "sleep",
            },
        ]
    )
    input_csv = tmp_path / "candidates.csv"
    output_csv = tmp_path / "queue.csv"
    report_json = tmp_path / "report.json"
    frame.to_csv(input_csv, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_active_learning_triage.py",
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--report-json",
            str(report_json),
        ],
    )
    rc = script.main()

    report = json.loads(report_json.read_text(encoding="utf-8"))
    queue = pd.read_csv(output_csv)

    assert rc == 0
    assert report["stats"]["training_signal_counts"]["corrected_event_rows"] == 1
    assert report["stats"]["training_signal_counts"]["hard_negative_rows"] == 1
    assert "triage_priority_score" in queue.columns
    assert queue.iloc[0]["candidate_id"] == "corr-1"


def test_active_learning_triage_script_can_build_queue_from_corrections_without_input_csv(
    tmp_path: Path, monkeypatch
):
    output_csv = tmp_path / "queue.csv"
    report_json = tmp_path / "report.json"
    captured = {}

    def _fake_build_correction_learning_records(
        elder_id: str,
        room: str,
        record_date: date,
        *,
        confidence_threshold: float,
        context_minutes: int,
    ):
        captured["elder_id"] = elder_id
        captured["room"] = room
        captured["record_date"] = record_date
        captured["confidence_threshold"] = confidence_threshold
        captured["context_minutes"] = context_minutes
        return [
            {
                "candidate_id": "correction:42",
                "room": room,
                "activity": "sleep",
                "confidence": 0.45,
                "predicted_label": "nap",
                "baseline_label": "sleep",
                "corrected_event": True,
                "boundary_start_target": 1,
                "boundary_end_target": 1,
                "hard_negative_flag": True,
                "hard_negative_label": "nap",
                "residual_review_flag": True,
                "residual_review_rows": 3,
            }
        ]

    monkeypatch.setattr(script, "build_correction_learning_records", _fake_build_correction_learning_records)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_active_learning_triage.py",
            "--correction-elder-id",
            "HK0011",
            "--correction-room",
            "bedroom",
            "--correction-date",
            "2026-03-13",
            "--output-csv",
            str(output_csv),
            "--report-json",
            str(report_json),
        ],
    )

    rc = script.main()

    report = json.loads(report_json.read_text(encoding="utf-8"))
    queue = pd.read_csv(output_csv)

    assert rc == 0
    assert captured == {
        "elder_id": "HK0011",
        "room": "bedroom",
        "record_date": date(2026, 3, 13),
        "confidence_threshold": 0.6,
        "context_minutes": 15,
    }
    assert report["stats"]["input_rows"] == 1
    assert report["stats"]["training_signal_counts"]["corrected_event_rows"] == 1
    assert queue.iloc[0]["candidate_id"] == "correction:42"
