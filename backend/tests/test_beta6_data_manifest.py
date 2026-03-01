from pathlib import Path

import numpy as np
import pandas as pd

from ml.beta6.data_manifest import CorpusManifestPolicy, build_pretrain_corpus_manifest


def _write_csv(path: Path, matrix: np.ndarray) -> None:
    frame = pd.DataFrame(matrix, columns=[f"f{i}" for i in range(matrix.shape[1])])
    frame.to_csv(path, index=False)


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
