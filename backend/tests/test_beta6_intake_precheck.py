import json
from pathlib import Path

import pytest

from ml.beta6.intake_precheck import (
    REASON_INTAKE_INVALID_ARTIFACT,
    REASON_INTAKE_MISSING_ARTIFACT,
    REASON_INTAKE_NOT_APPROVED,
    IntakeGateBlockedError,
    enforce_approved_intake_artifact,
)


def _write_report_files(base_dir: Path) -> dict[str, str]:
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    validate_json = reports_dir / "validate.json"
    diff_json = reports_dir / "diff.json"
    diff_csv = reports_dir / "diff.csv"
    smoke_json = reports_dir / "smoke.json"
    validate_json.write_text(json.dumps({"status": "pass"}))
    diff_json.write_text(json.dumps({"summary": {"rows_compared": 1}}))
    diff_csv.write_text("day,room,windows_changed\n7,livingroom,1\n")
    smoke_json.write_text(json.dumps({"status": "pass"}))
    return {
        "validate_json": "reports/validate.json",
        "diff_json": "reports/diff.json",
        "diff_csv": "reports/diff.csv",
        "smoke_json": "reports/smoke.json",
    }


def _artifact(reports: dict[str, str], *, approved: bool) -> dict:
    return {
        "artifact_version": "v1",
        "generated_at": "2026-02-25T22:45:00+00:00",
        "status": "approved" if approved else "rejected",
        "pack": {
            "candidate_dir": "/tmp/candidate",
            "baseline_dir": "/tmp/baseline",
            "elder_id": "HK0011_jessica",
            "min_day": 4,
            "max_day": 10,
            "smoke_day": 7,
            "seed": 11,
        },
        "steps": {
            "validate": {"status": "pass"},
            "diff": {"status": "pass"},
            "smoke": {"status": "pass"},
        },
        "reports": reports,
        "gate": {
            "approved": approved,
            "blocking_reasons": [] if approved else ["validate_failed"],
        },
    }


def test_enforce_approved_intake_artifact_returns_artifact_when_approved(tmp_path: Path):
    reports = _write_report_files(tmp_path)
    payload = _artifact(reports, approved=True)
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(payload))

    loaded = enforce_approved_intake_artifact(artifact_path)
    assert loaded["status"] == "approved"
    assert loaded["gate"]["approved"] is True


def test_enforce_approved_intake_artifact_blocks_missing_path():
    with pytest.raises(IntakeGateBlockedError) as exc_info:
        enforce_approved_intake_artifact("/tmp/does_not_exist_intake_artifact.json")
    assert exc_info.value.reason_code == REASON_INTAKE_MISSING_ARTIFACT


def test_enforce_approved_intake_artifact_blocks_rejected_gate(tmp_path: Path):
    reports = _write_report_files(tmp_path)
    payload = _artifact(reports, approved=False)
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(payload))

    with pytest.raises(IntakeGateBlockedError) as exc_info:
        enforce_approved_intake_artifact(artifact_path)
    assert exc_info.value.reason_code == REASON_INTAKE_NOT_APPROVED


def test_enforce_approved_intake_artifact_blocks_invalid_schema(tmp_path: Path):
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps({"hello": "world"}))

    with pytest.raises(IntakeGateBlockedError) as exc_info:
        enforce_approved_intake_artifact(artifact_path)
    assert exc_info.value.reason_code == REASON_INTAKE_INVALID_ARTIFACT
