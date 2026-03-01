from pathlib import Path

from ml.beta6.evaluation_engine import (
    build_room_evaluation_report,
    verify_evaluation_report_signature,
)
from ml.beta6.gate_engine import GateEngine
from ml.beta6.rejection_artifact import verify_rejection_artifact_signature


def test_gate_engine_emits_signed_eval_and_rejection_artifacts(tmp_path: Path):
    signing_key = "unit-test-key"
    room_reports = [
        build_room_evaluation_report(
            room="bedroom",
            y_true=["sleep", "sleep"],
            y_pred=["sleep", "sleep"],
            data_viable=True,
            timeline_metrics={"duration_mae_minutes": 2.0, "fragmentation_rate": 0.05},
        ),
        {
            "room": "livingroom",
            "passed": False,
            "metrics_passed": False,
            "reason_code": "fail_timeline_mae",
            "timeline_metrics": {"duration_mae_minutes": 20.0, "fragmentation_rate": 0.05},
            "details": {"source": "unit-test"},
        },
    ]

    out = GateEngine().decide_run_from_reports(
        room_reports=room_reports,
        run_id="run_123",
        elder_id="elder_123",
        signing_key=signing_key,
        output_dir=tmp_path,
    )

    eval_report = out["evaluation_report"]
    assert verify_evaluation_report_signature(eval_report, signing_key=signing_key) is True
    assert out["run_decision"]["passed"] is False
    assert out["run_decision"]["reason_code"] == "fail_timeline_mae"

    rejection = out["rejection_artifact"]
    assert rejection is not None
    assert verify_rejection_artifact_signature(rejection, signing_key=signing_key) is True

    assert (tmp_path / "run_123_evaluation_report.json").exists()
    assert (tmp_path / "run_123_rejection_artifact.json").exists()
