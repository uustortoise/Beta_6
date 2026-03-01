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
