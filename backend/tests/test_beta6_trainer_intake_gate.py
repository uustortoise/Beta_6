import numpy as np
import json
from pathlib import Path

from ml.beta6 import beta6_trainer
from ml.beta6.intake_precheck import IntakeGateBlockedError


def test_trainer_blocks_when_intake_artifact_missing(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["beta6_trainer.py", "--dataset", "/tmp/unused_dataset.json"],
    )

    load_called = {"value": False}

    def _fail_if_called(_dataset_path):
        load_called["value"] = True
        return []

    monkeypatch.setattr(beta6_trainer, "load_all_datasets", _fail_if_called)
    rc = beta6_trainer.main()

    assert rc == 2
    assert load_called["value"] is False


def test_trainer_validate_only_does_not_require_intake_artifact(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["beta6_trainer.py", "--dataset", "/tmp/unused_dataset.json", "--validate-only"],
    )
    monkeypatch.setattr(beta6_trainer, "load_all_datasets", lambda _dataset_path: [{"samples": []}])
    monkeypatch.setattr(
        beta6_trainer,
        "prepare_training_data",
        lambda _datasets: (np.zeros((1, 1), dtype=np.float32), np.zeros((1,), dtype=np.int32), {"idle": 0}),
    )

    rc = beta6_trainer.main()
    assert rc == 0


def test_trainer_blocks_when_intake_gate_not_approved(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "beta6_trainer.py",
            "--dataset",
            "/tmp/unused_dataset.json",
            "--intake-artifact",
            "/tmp/intake.json",
        ],
    )

    def _blocked(_artifact_path):
        raise IntakeGateBlockedError(
            reason_code="intake_gate_not_approved",
            detail="Intake gate is not approved: ['validate_failed']",
        )

    monkeypatch.setattr(beta6_trainer, "enforce_approved_intake_artifact", _blocked)
    rc = beta6_trainer.main()
    assert rc == 2


def test_trainer_validate_only_pretrain_does_not_require_intake_artifact(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "beta6_trainer.py",
            "--pretrain-manifest",
            "/tmp/pretrain_manifest.json",
            "--validate-only",
        ],
    )
    monkeypatch.setattr(
        beta6_trainer,
        "load_corpus_matrix",
        lambda _manifest_path, max_files=None: np.zeros((16, 4), dtype=np.float32),
    )
    rc = beta6_trainer.main()
    assert rc == 0


def test_trainer_pretrain_mode_runs_after_intake_gate_pass(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "beta6_trainer.py",
            "--pretrain-manifest",
            "/tmp/pretrain_manifest.json",
            "--intake-artifact",
            "/tmp/intake.json",
        ],
    )
    monkeypatch.setattr(
        beta6_trainer,
        "enforce_approved_intake_artifact",
        lambda _artifact: {"generated_at": "2026-02-26T00:00:00+00:00"},
    )
    called = {"value": False}

    def _run_pretrain(**kwargs):
        called["value"] = True
        return {
            "status": "pass",
            "metrics": {"final_reconstruction_mse": 0.1},
            "artifacts": {"checkpoint_npz": "/tmp/ckpt.npz"},
        }

    monkeypatch.setattr(beta6_trainer, "run_self_supervised_pretraining", _run_pretrain)
    rc = beta6_trainer.main()
    assert rc == 0
    assert called["value"] is True


def test_trainer_safe_finetune_mode_runs_after_intake_gate_pass(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "beta6_trainer.py",
            "--safe-finetune-dataset",
            "/tmp/golden_dataset.json",
            "--intake-artifact",
            "/tmp/intake.json",
        ],
    )
    monkeypatch.setattr(
        beta6_trainer,
        "enforce_approved_intake_artifact",
        lambda _artifact: {"generated_at": "2026-02-26T00:00:00+00:00"},
    )
    called = {"value": False}

    def _run_finetune(**kwargs):
        called["value"] = True
        return {
            "status": "pass",
            "metrics": {"heldout_accuracy": 0.9},
            "artifacts": {"report_json": "/tmp/r.json"},
        }

    monkeypatch.setattr(beta6_trainer, "run_safe_class_finetune", _run_finetune)
    rc = beta6_trainer.main()
    assert rc == 0
    assert called["value"] is True


def test_trainer_rejects_multiple_modes(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "beta6_trainer.py",
            "--pretrain-manifest",
            "/tmp/pretrain.json",
            "--safe-finetune-dataset",
            "/tmp/golden_dataset.json",
        ],
    )
    rc = beta6_trainer.main()
    assert rc == 2


def test_intake_gate_rejects_incomplete_20x14_resident_windows(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": "beta6_pretrain_manifest_v1",
                "entries": [
                    {
                        "path": "/tmp/a.npy",
                        "resident_id": "HK001",
                        "days_covered": 7,
                        "views": ["shadow_cohort", "labeled_high_trust_finetune_eval"],
                        "context_completeness": {"status": "missing_required_context"},
                    }
                ],
                "stats": {"records_kept": 1, "p0_violations": 0},
                "corpus_views": {
                    "shadow_cohort": {"resident_count": 1, "min_days_covered": 7},
                    "labeled_high_trust_finetune_eval": {"resident_count": 1, "min_days_covered": 7},
                },
                "context_summary": {"ready_entries": 0},
            }
        ),
        encoding="utf-8",
    )

    gate = beta6_trainer.evaluate_pretrain_manifest_intake_gate(
        manifest_path,
        required_residents=2,
        required_days=14,
    )

    assert gate["pass"] is False
    assert "shadow_resident_coverage_below_contract" in gate["reason_codes"]
    assert "shadow_days_coverage_below_contract" in gate["reason_codes"]
    assert "resident_home_context_incomplete" in gate["reason_codes"]


def test_bad_candidates_can_be_eliminated_early_without_affecting_good_runs():
    bad = beta6_trainer.should_eliminate_candidate_early(
        {"event_iou": 0.1, "fragmentation_rate": 2.0}
    )
    good = beta6_trainer.should_eliminate_candidate_early(
        {"event_iou": 0.8, "fragmentation_rate": 0.2}
    )

    assert bad["should_eliminate"] is True
    assert "event_iou_below_floor" in bad["reason_codes"]
    assert good["should_eliminate"] is False
