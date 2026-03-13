import json
from pathlib import Path

import pytest

from ml.beta6.data_manifest import CorpusManifestPolicy, build_pretrain_corpus_manifest
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


def test_intake_gate_auto_approves_clean_pretrain_manifest(tmp_path: Path):
    csv_path = tmp_path / "clean.csv"
    csv_path.write_text(
        "\n".join(
            [
                "elder_id,timestamp,room,activity,f1,f2",
                "HK0011_jessica,2025-12-04T07:00:00,Bedroom,sleep,1,0",
                "HK0011_jessica,2025-12-04T07:05:00,Bedroom,bedroom_normal_use,0,0",
            ]
        ),
        encoding="utf-8",
    )
    manifest = build_pretrain_corpus_manifest(
        corpus_roots=[tmp_path],
        policy=CorpusManifestPolicy(min_rows=2, min_features=2, max_missing_ratio=0.2),
    )

    normalized = validate_intake_artifact(manifest)

    assert normalized["gate"]["approved"] is True
    assert normalized["summary"]["user_tags"] == ["HK0011_jessica"]


def test_intake_gate_quarantines_red_flag_files_with_explicit_reasons(tmp_path: Path):
    clean = tmp_path / "clean.csv"
    clean.write_text(
        "\n".join(
            [
                "elder_id,timestamp,room,activity,f1,f2",
                "HK0011_jessica,2025-12-04T07:00:00,Bedroom,sleep,1,0",
                "HK0011_jessica,2025-12-04T07:05:00,Bedroom,bedroom_normal_use,0,0",
            ]
        ),
        encoding="utf-8",
    )
    bad = tmp_path / "missing_date.csv"
    bad.write_text(
        "\n".join(
            [
                "elder_id,room,activity,f1,f2",
                "HK0011_jessica,Kitchen,meal_preparation,1,0",
                "HK0011_jessica,Kitchen,meal_preparation,1,1",
            ]
        ),
        encoding="utf-8",
    )

    manifest = build_pretrain_corpus_manifest(
        corpus_roots=[tmp_path],
        policy=CorpusManifestPolicy(min_rows=2, min_features=2, max_missing_ratio=0.2),
    )
    normalized = validate_intake_artifact(manifest)

    assert normalized["gate"]["approved"] is True
    assert len(normalized["quarantine"]) == 1
    assert normalized["quarantine"][0]["path"].endswith("missing_date.csv")
    assert normalized["quarantine"][0]["quarantine_reasons"] == ["manifest_missing_date_tag"]
