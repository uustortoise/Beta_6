from pathlib import Path
import json

import numpy as np
import pytest

from ml.beta6.data_manifest import CorpusManifestPolicy, build_pretrain_corpus_manifest
from ml.beta6.feature_store import compute_feature_sequence_cache_key, should_reuse_cached_tensors
from ml.beta6.self_supervised_pretrain import (
    can_reuse_pretrain_cache,
    encode_with_checkpoint,
    load_pretrain_checkpoint,
    run_self_supervised_pretraining,
)


def _build_manifest(tmp_path: Path) -> Path:
    a = np.arange(120, dtype=np.float32).reshape(30, 4)
    b = (np.arange(120, dtype=np.float32).reshape(30, 4) * 0.5) + 3.0
    np.save(tmp_path / "a.npy", a)
    np.save(tmp_path / "b.npy", b)
    manifest = build_pretrain_corpus_manifest(
        corpus_roots=[tmp_path],
        policy=CorpusManifestPolicy(min_rows=8, min_features=2, max_missing_ratio=0.2),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def test_pretrain_writes_checkpoint_and_encodes(tmp_path: Path):
    manifest_path = _build_manifest(tmp_path)
    config_path = tmp_path / "pretrain.yaml"
    config_path.write_text(
        "version: v1\npretrain:\n  embedding_dim: 3\n  mask_ratio: 0.1\n  epochs: 1\n  random_seed: 11\n",
        encoding="utf-8",
    )

    result = run_self_supervised_pretraining(
        manifest_path=manifest_path,
        config_path=config_path,
        output_dir=tmp_path / "ckpt",
    )
    checkpoint = load_pretrain_checkpoint(result["artifacts"]["checkpoint_npz"])
    sample = np.arange(20, dtype=np.float32).reshape(5, 4)
    embedding = encode_with_checkpoint(sample, checkpoint)

    assert result["status"] == "pass"
    assert Path(result["artifacts"]["checkpoint_npz"]).exists()
    assert embedding.shape == (5, 3)


def test_pretrain_seeded_rerun_is_deterministic(tmp_path: Path):
    manifest_path = _build_manifest(tmp_path)
    config_path = tmp_path / "pretrain.yaml"
    config_path.write_text(
        "version: v1\npretrain:\n  embedding_dim: 2\n  mask_ratio: 0.2\n  epochs: 2\n  random_seed: 7\n",
        encoding="utf-8",
    )

    run_a = run_self_supervised_pretraining(
        manifest_path=manifest_path,
        config_path=config_path,
        output_dir=tmp_path / "run_a",
    )
    run_b = run_self_supervised_pretraining(
        manifest_path=manifest_path,
        config_path=config_path,
        output_dir=tmp_path / "run_b",
    )

    assert run_a["metrics"]["final_reconstruction_mse"] == run_b["metrics"]["final_reconstruction_mse"]


def test_pretrain_fails_closed_when_manifest_has_p0_violations(tmp_path: Path):
    matrix = np.arange(80, dtype=np.float32).reshape(20, 4)
    np.save(tmp_path / "corpus.npy", matrix)
    manifest = {
        "manifest_version": "beta6_pretrain_manifest_v1",
        "entries": [{"path": str((tmp_path / "corpus.npy").resolve())}],
        "stats": {"p0_violations": 1, "records_kept": 1},
    }
    manifest_path = tmp_path / "manifest_bad.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_p0_violations_present"):
        run_self_supervised_pretraining(
            manifest_path=manifest_path,
            config_path=None,
            output_dir=tmp_path / "out",
        )


def test_pretrain_fails_closed_on_empty_manifest_records(tmp_path: Path):
    manifest = {
        "manifest_version": "beta6_pretrain_manifest_v1",
        "entries": [],
        "stats": {"p0_violations": 0, "records_kept": 0},
    }
    manifest_path = tmp_path / "manifest_empty.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_empty_corpus"):
        run_self_supervised_pretraining(
            manifest_path=manifest_path,
            config_path=None,
            output_dir=tmp_path / "out",
        )


def test_feature_sequence_cache_respects_manifest_and_policy_fingerprint():
    cache_key_a = compute_feature_sequence_cache_key(
        manifest_fingerprint="m1",
        policy_fingerprint="p1",
        extra={"max_files": 4},
    )
    cache_key_b = compute_feature_sequence_cache_key(
        manifest_fingerprint="m1",
        policy_fingerprint="p1",
        extra={"max_files": 4},
    )
    cache_key_c = compute_feature_sequence_cache_key(
        manifest_fingerprint="m2",
        policy_fingerprint="p1",
        extra={"max_files": 4},
    )

    assert cache_key_a == cache_key_b
    assert cache_key_a != cache_key_c


def test_tensor_reuse_is_disabled_when_inputs_change():
    cache_metadata = {
        "manifest_fingerprint": "m1",
        "policy_fingerprint": "p1",
    }
    assert should_reuse_cached_tensors(
        cache_metadata,
        manifest_fingerprint="m1",
        policy_fingerprint="p1",
    ) is True
    assert can_reuse_pretrain_cache(
        cache_metadata,
        manifest_fingerprint="m2",
        policy_fingerprint="p1",
    ) is False
