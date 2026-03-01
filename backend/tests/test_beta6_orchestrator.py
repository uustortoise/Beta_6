import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.beta6.orchestrator import (
    Beta6Orchestrator,
    PhaseGateError,
    REASON_INTAKE_REQUIRED,
    REASON_RUNTIME_EVAL_PARITY_FAILED,
)
from ml.beta6.runtime_eval_parity import DecoderPolicy


def _trace():
    return [
        {"label": "occupied"},
        {"label": "unoccupied"},
        {"label": "occupied"},
    ]


def test_orchestrator_phase1_parity_gate_passes():
    orchestrator = Beta6Orchestrator(require_intake_artifact=False)
    result = orchestrator.run_phase1_parity_gate(trace_steps=_trace())
    assert result.passed is True
    assert result.mismatch_count == 0


def test_orchestrator_phase1_parity_gate_raises_on_mismatch():
    orchestrator = Beta6Orchestrator(require_intake_artifact=False)
    with pytest.raises(PhaseGateError) as exc:
        orchestrator.run_phase1_parity_gate(
            trace_steps=_trace(),
            runtime_policy=DecoderPolicy(spike_suppression=True),
            eval_policy=DecoderPolicy(spike_suppression=False),
        )
    assert exc.value.reason_code == REASON_RUNTIME_EVAL_PARITY_FAILED


def test_orchestrator_phase2_pretraining_requires_intake_by_default(tmp_path: Path):
    matrix = np.arange(80, dtype=np.float32).reshape(20, 4)
    np.save(tmp_path / "corpus.npy", matrix)
    manifest = {
        "manifest_version": "beta6_pretrain_manifest_v1",
        "entries": [{"path": str((tmp_path / "corpus.npy").resolve())}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    orchestrator = Beta6Orchestrator(require_intake_artifact=True)
    with pytest.raises(PhaseGateError) as exc:
        orchestrator.run_phase2_pretraining(
            manifest_path=manifest_path,
            output_dir=tmp_path / "out",
            config_path=None,
        )
    assert exc.value.reason_code == REASON_INTAKE_REQUIRED


def test_orchestrator_phase3_active_learning_triage_writes_artifacts(tmp_path: Path):
    candidates = pd.DataFrame(
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
        ]
    )
    in_csv = tmp_path / "candidates.csv"
    out_csv = tmp_path / "triage.csv"
    out_report = tmp_path / "triage.json"
    candidates.to_csv(in_csv, index=False)
    orchestrator = Beta6Orchestrator(require_intake_artifact=False)
    result = orchestrator.run_phase3_active_learning_triage(
        candidates_csv=in_csv,
        output_csv=out_csv,
        report_json=out_report,
    )
    assert result["status"] == "pass"
    assert out_csv.exists()
    assert out_report.exists()


def test_orchestrator_phase5_crf_ab_gate_passes_on_non_regression():
    orchestrator = Beta6Orchestrator(require_intake_artifact=False)
    labels = ["sleep", "out"]
    probs = np.asarray(
        [
            [0.92, 0.08],
            [0.90, 0.10],
            [0.15, 0.85],
            [0.10, 0.90],
        ],
        dtype=np.float64,
    )
    log_probs = np.log(np.clip(probs, 1e-9, 1.0))
    result = orchestrator.run_phase5_crf_ab_gate(
        observation_log_probs=log_probs,
        labels=labels,
        true_labels=["sleep", "sleep", "out", "out"],
        label_sequences_for_fit=[["sleep", "sleep", "out", "out"]],
    )
    assert result.status == "pass"
    assert result.pass_gate is True
    assert result.decode_length == 4


def test_orchestrator_phase6_shadow_compare_emits_signed_artifact(tmp_path: Path):
    orchestrator = Beta6Orchestrator(require_intake_artifact=False)
    out = tmp_path / "shadow_compare.json"
    result = orchestrator.run_phase6_shadow_compare(
        room_rows=[
            {
                "room": "bedroom",
                "legacy_gate_pass": True,
                "legacy_gate_reasons": [],
                "beta6_gate_pass": False,
                "beta6_reason_code": "fail_timeline_mae",
                "beta6_details": {"timeline_hard_gate_passed": False},
            }
        ],
        run_id="r_shadow",
        elder_id="HK001",
        signing_key="test-shadow-key",
        output_path=out,
    )
    assert out.exists()
    assert result.status == "watch"
    assert result.divergence_count == 1
    assert result.unexplained_divergence_count == 0
    assert result.signature.startswith("sha256:")
