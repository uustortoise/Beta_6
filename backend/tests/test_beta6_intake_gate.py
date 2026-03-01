import json
from pathlib import Path

import pytest

from ml.beta6.intake_gate import assert_intake_artifact_approved, validate_intake_artifact


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
        "validate_json": str(validate_json),
        "diff_json": str(diff_json),
        "diff_csv": str(diff_csv),
        "smoke_json": str(smoke_json),
    }


def _approved_artifact(reports: dict[str, str]) -> dict:
    return {
        "artifact_version": "v1",
        "generated_at": "2026-02-25T22:45:00+00:00",
        "status": "approved",
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
        "gate": {"approved": True, "blocking_reasons": []},
    }


def test_validate_intake_artifact_accepts_approved_bundle(tmp_path: Path):
    reports = _write_report_files(tmp_path)
    artifact = _approved_artifact(reports)
    normalized = validate_intake_artifact(artifact, require_report_files=True)
    assert normalized["status"] == "approved"
    assert normalized["gate"]["approved"] is True


def test_validate_intake_artifact_rejects_policy_mismatch(tmp_path: Path):
    reports = _write_report_files(tmp_path)
    artifact = _approved_artifact(reports)
    artifact["gate"]["approved"] = False
    artifact["gate"]["blocking_reasons"] = []
    artifact["status"] = "approved"
    with pytest.raises(ValueError, match="status must be 'rejected'"):
        validate_intake_artifact(artifact)


def test_assert_intake_artifact_approved_raises_for_rejected_bundle(tmp_path: Path):
    reports = _write_report_files(tmp_path)
    artifact = _approved_artifact(reports)
    artifact["status"] = "rejected"
    artifact["gate"]["approved"] = False
    artifact["gate"]["blocking_reasons"] = ["validate_failed"]

    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="not approved"):
        assert_intake_artifact_approved(artifact_path)
