from pathlib import Path

from scripts.run_beta6_label_pack_intake import run_intake


def test_run_intake_approves_when_all_steps_pass(tmp_path: Path, monkeypatch):
    candidate_dir = tmp_path / "candidate"
    baseline_dir = tmp_path / "baseline"
    candidate_dir.mkdir()
    baseline_dir.mkdir()

    monkeypatch.setattr(
        "scripts.run_beta6_label_pack_intake.validate_label_pack",
        lambda **kwargs: {"status": "pass", "violations": [], "warnings": []},
    )
    monkeypatch.setattr(
        "scripts.run_beta6_label_pack_intake.diff_label_pack",
        lambda **kwargs: {"summary": {"rows_compared": 10, "days_compared": 2, "windows_changed_total": 5}, "rows": []},
    )
    monkeypatch.setattr(
        "scripts.run_beta6_label_pack_intake.run_smoke",
        lambda **kwargs: {"status": "pass", "blocking_reasons": [], "checks": []},
    )

    artifact = run_intake(
        candidate_dir=candidate_dir,
        baseline_dir=baseline_dir,
        elder_id="HK0011_jessica",
        min_day=4,
        max_day=10,
        smoke_day=7,
        seed=11,
        expectation_config=None,
        registry_path=None,
        output_dir=tmp_path / "out",
        train_context_days=1,
    )

    assert artifact["status"] == "approved"
    assert artifact["gate"]["approved"] is True
    assert artifact["steps"]["validate"]["status"] == "pass"
    assert artifact["steps"]["diff"]["status"] == "pass"
    assert artifact["steps"]["smoke"]["status"] == "pass"


def test_run_intake_rejects_when_validation_fails_and_smoke_is_skipped(tmp_path: Path, monkeypatch):
    candidate_dir = tmp_path / "candidate"
    baseline_dir = tmp_path / "baseline"
    candidate_dir.mkdir()
    baseline_dir.mkdir()

    monkeypatch.setattr(
        "scripts.run_beta6_label_pack_intake.validate_label_pack",
        lambda **kwargs: {"status": "fail", "violations": ["missing_room_sheet"], "warnings": []},
    )
    monkeypatch.setattr(
        "scripts.run_beta6_label_pack_intake.diff_label_pack",
        lambda **kwargs: {"summary": {"rows_compared": 1, "days_compared": 1, "windows_changed_total": 0}, "rows": []},
    )
    monkeypatch.setattr(
        "scripts.run_beta6_label_pack_intake.run_smoke",
        lambda **kwargs: {"status": "pass", "blocking_reasons": [], "checks": []},
    )

    artifact = run_intake(
        candidate_dir=candidate_dir,
        baseline_dir=baseline_dir,
        elder_id="HK0011_jessica",
        min_day=4,
        max_day=10,
        smoke_day=7,
        seed=11,
        expectation_config=None,
        registry_path=None,
        output_dir=tmp_path / "out",
        train_context_days=1,
    )

    assert artifact["status"] == "rejected"
    assert artifact["gate"]["approved"] is False
    assert "validate_failed" in artifact["gate"]["blocking_reasons"]
    assert "smoke_skipped_due_validation_fail" in artifact["gate"]["blocking_reasons"]
    assert artifact["steps"]["smoke"]["status"] == "skipped"
