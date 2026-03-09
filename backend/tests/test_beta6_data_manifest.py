from pathlib import Path
import json

import numpy as np
import pandas as pd

from ml.beta6.data_manifest import (
    CorpusManifestPolicy,
    build_pretrain_corpus_manifest,
    evaluate_beta62_corpus_contract,
)


def _write_csv(path: Path, matrix: np.ndarray) -> None:
    frame = pd.DataFrame(matrix, columns=[f"f{i}" for i in range(matrix.shape[1])])
    frame.to_csv(path, index=False)


def _write_sidecar(path: Path, payload: dict) -> None:
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(payload), encoding="utf-8")


def test_manifest_fingerprint_is_stable_across_reruns(tmp_path: Path):
    matrix = np.arange(80, dtype=np.float32).reshape(20, 4)
    _write_csv(tmp_path / "a.csv", matrix)
    _write_csv(tmp_path / "b.csv", matrix + 1.0)
    policy = CorpusManifestPolicy(min_rows=8, min_features=2, max_missing_ratio=0.2)

    manifest_a = build_pretrain_corpus_manifest(corpus_roots=[tmp_path], policy=policy)
    manifest_b = build_pretrain_corpus_manifest(corpus_roots=[tmp_path], policy=policy)

    assert manifest_a["fingerprint"]["value"] == manifest_b["fingerprint"]["value"]
    assert manifest_a["stats"]["records_kept"] == 2
    assert manifest_a["stats"]["p0_violations"] == 0


def test_manifest_dedupes_by_content_hash(tmp_path: Path):
    matrix = np.arange(40, dtype=np.float32).reshape(10, 4)
    _write_csv(tmp_path / "x.csv", matrix)
    _write_csv(tmp_path / "x_copy.csv", matrix)

    manifest = build_pretrain_corpus_manifest(
        corpus_roots=[tmp_path],
        policy=CorpusManifestPolicy(min_rows=4, min_features=2, max_missing_ratio=0.2),
    )
    assert manifest["stats"]["records_kept"] == 1
    assert manifest["stats"]["duplicates_removed"] == 1


def test_manifest_flags_missing_ratio_violation(tmp_path: Path):
    matrix = np.array(
        [
            [1.0, np.nan, np.nan],
            [2.0, np.nan, np.nan],
            [3.0, np.nan, np.nan],
            [4.0, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    _write_csv(tmp_path / "noisy.csv", matrix)
    manifest = build_pretrain_corpus_manifest(
        corpus_roots=[tmp_path],
        policy=CorpusManifestPolicy(min_rows=2, min_features=2, max_missing_ratio=0.3),
    )
    assert manifest["stats"]["records_kept"] == 0
    assert manifest["stats"]["p0_violations"] >= 1
    reasons = {item["reason_code"] for item in manifest["violations"]}
    assert "manifest_missing_ratio_violation" in reasons


def test_manifest_empty_corpus_is_p0_violation(tmp_path: Path):
    manifest = build_pretrain_corpus_manifest(
        corpus_roots=[tmp_path],
        policy=CorpusManifestPolicy(min_rows=2, min_features=2, max_missing_ratio=0.3),
    )
    assert manifest["stats"]["files_scanned"] == 0
    assert manifest["stats"]["records_kept"] == 0
    assert manifest["stats"]["p0_violations"] >= 1
    assert any(v["reason_code"] == "manifest_empty_corpus" for v in manifest["violations"])


def test_manifest_contains_shadow_pretrain_and_labeled_views(tmp_path: Path):
    shadow_dir = tmp_path / "HK001_shadow"
    labeled_dir = tmp_path / "HK002_labeled"
    shadow_dir.mkdir()
    labeled_dir.mkdir()

    _write_csv(shadow_dir / "shadow_day.csv", np.arange(80, dtype=np.float32).reshape(20, 4))
    _write_sidecar(
        shadow_dir / "shadow_day.csv",
        {
            "resident_id": "HK001",
            "days_covered": 14,
            "views": ["shadow_cohort", "unlabeled_pretrain"],
            "resident_home_context": {"status": "ready", "missing_fields": []},
        },
    )
    _write_csv(labeled_dir / "golden_day.csv", np.arange(80, dtype=np.float32).reshape(20, 4) + 2.0)
    _write_sidecar(
        labeled_dir / "golden_day.csv",
        {
            "resident_id": "HK002",
            "days_covered": 14,
            "views": ["labeled_high_trust_finetune_eval"],
            "resident_home_context": {"status": "ready", "missing_fields": []},
            "label_quality": {"trust_tier": "high", "reviewed_fraction": 1.0},
        },
    )

    manifest = build_pretrain_corpus_manifest(
        corpus_roots=[tmp_path],
        policy=CorpusManifestPolicy(min_rows=8, min_features=2, max_missing_ratio=0.2),
    )

    assert manifest["corpus_views"]["shadow_cohort"]["entry_count"] == 1
    assert manifest["corpus_views"]["unlabeled_pretrain"]["entry_count"] == 1
    assert manifest["corpus_views"]["labeled_high_trust_finetune_eval"]["entry_count"] == 1
    assert manifest["label_quality_summary"]["high_trust_entries"] == 1


def test_manifest_tracks_context_completeness_and_label_quality(tmp_path: Path):
    _write_csv(tmp_path / "candidate.csv", np.arange(80, dtype=np.float32).reshape(20, 4))
    _write_sidecar(
        tmp_path / "candidate.csv",
        {
            "resident_id": "HK003",
            "days_covered": 10,
            "views": ["shadow_cohort", "labeled_high_trust_finetune_eval"],
            "resident_home_context": {
                "status": "missing_required_context",
                "missing_fields": ["helper_presence"],
            },
            "label_quality": {"trust_tier": "high", "reviewed_fraction": 0.9},
        },
    )

    manifest = build_pretrain_corpus_manifest(
        corpus_roots=[tmp_path],
        policy=CorpusManifestPolicy(min_rows=8, min_features=2, max_missing_ratio=0.2),
    )
    entry = manifest["entries"][0]
    gate = evaluate_beta62_corpus_contract(manifest, required_residents=1, required_days=14)

    assert entry["context_completeness"]["status"] == "missing_required_context"
    assert "helper_presence" in entry["context_completeness"]["missing_fields"]
    assert entry["label_quality"]["trust_tier"] == "high"
    assert gate["pass"] is False
    assert "shadow_days_coverage_below_contract" in gate["reason_codes"]
