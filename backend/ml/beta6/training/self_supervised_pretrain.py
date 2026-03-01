"""Beta 6 self-supervised pretraining pipeline (masked reconstruction)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from ml.yaml_compat import load_yaml_file

from ..data.data_manifest import load_feature_matrix, load_manifest
from ..data.feature_fingerprint import hash_json_payload


PRETRAIN_CONFIG_VERSION = "v1"


@dataclass(frozen=True)
class PretrainConfig:
    embedding_dim: int = 16
    mask_ratio: float = 0.15
    epochs: int = 1
    random_seed: int = 42
    min_total_rows: int = 16


def _as_mapping(payload: Any) -> Mapping[str, Any]:
    return payload if isinstance(payload, Mapping) else {}


def load_pretrain_config(path: str | Path | None) -> PretrainConfig:
    if path is None:
        return PretrainConfig()
    raw = load_yaml_file(Path(path).resolve()) or {}
    section = _as_mapping(raw).get("pretrain")
    if not isinstance(section, Mapping):
        section = raw if isinstance(raw, Mapping) else {}
    return PretrainConfig(
        embedding_dim=max(int(section.get("embedding_dim", 16)), 2),
        mask_ratio=min(max(float(section.get("mask_ratio", 0.15)), 0.0), 0.9),
        epochs=max(int(section.get("epochs", 1)), 1),
        random_seed=int(section.get("random_seed", 42)),
        min_total_rows=max(int(section.get("min_total_rows", 16)), 1),
    )


def load_corpus_matrix(
    manifest_path: str | Path,
    *,
    max_files: Optional[int] = None,
    enforce_manifest_contract: bool = True,
) -> np.ndarray:
    manifest = load_manifest(manifest_path)
    if enforce_manifest_contract:
        _assert_manifest_contract(manifest, manifest_path=manifest_path)
    entries = manifest.get("entries", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest.entries must be a non-empty list")

    matrices = []
    selected_entries = entries if max_files is None else entries[: max(int(max_files), 0)]
    for entry in selected_entries:
        path = str(_as_mapping(entry).get("path", "")).strip()
        if not path:
            continue
        matrix = load_feature_matrix(path)
        matrices.append(matrix)

    if not matrices:
        raise ValueError("no valid corpus files in manifest")
    return np.vstack(matrices).astype(np.float32)


def _assert_manifest_contract(manifest: Mapping[str, Any], *, manifest_path: str | Path) -> None:
    stats = _as_mapping(manifest.get("stats"))
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("manifest.entries must be a list")

    p0_violations = int(stats.get("p0_violations", 0) or 0)
    records_kept = int(stats.get("records_kept", len(entries)) or 0)

    if p0_violations > 0:
        raise ValueError(
            f"manifest_p0_violations_present:path={Path(manifest_path).resolve()} count={p0_violations}"
        )
    if records_kept <= 0 or len(entries) == 0:
        raise ValueError(
            f"manifest_empty_corpus:path={Path(manifest_path).resolve()} "
            f"records_kept={records_kept} entries={len(entries)}"
        )


def _masked_view(matrix: np.ndarray, *, mask_ratio: float, rng: np.random.Generator) -> np.ndarray:
    masked = np.array(matrix, copy=True)
    if mask_ratio <= 0.0:
        return masked
    mask = rng.random(masked.shape) < mask_ratio
    masked[mask] = 0.0
    return masked


def train_self_supervised_encoder(
    matrix: np.ndarray,
    *,
    config: PretrainConfig,
) -> Dict[str, Any]:
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape={matrix.shape}")
    if matrix.shape[0] < config.min_total_rows:
        raise ValueError(
            f"insufficient rows for pretraining: rows={matrix.shape[0]} min_total_rows={config.min_total_rows}"
        )

    rng = np.random.default_rng(config.random_seed)
    clean = np.nan_to_num(matrix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = clean.mean(axis=0, keepdims=True)
    std = clean.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (clean - mean) / std

    embedding_dim = int(min(config.embedding_dim, normalized.shape[1]))
    components = None
    mse_history = []
    for _ in range(config.epochs):
        masked = _masked_view(normalized, mask_ratio=config.mask_ratio, rng=rng)
        _, _, vt = np.linalg.svd(masked, full_matrices=False)
        components = vt[:embedding_dim]
        embedding = normalized @ components.T
        reconstructed = embedding @ components
        mse = float(np.mean((normalized - reconstructed) ** 2))
        mse_history.append(mse)

    assert components is not None
    return {
        "version": PRETRAIN_CONFIG_VERSION,
        "config": {
            "embedding_dim": int(embedding_dim),
            "mask_ratio": float(config.mask_ratio),
            "epochs": int(config.epochs),
            "random_seed": int(config.random_seed),
            "min_total_rows": int(config.min_total_rows),
        },
        "state": {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "components": components.astype(np.float32),
        },
        "metrics": {
            "final_reconstruction_mse": float(mse_history[-1]),
            "mse_history": [float(v) for v in mse_history],
            "rows_seen": int(normalized.shape[0]),
            "feature_count": int(normalized.shape[1]),
        },
    }


def save_pretrain_checkpoint(
    payload: Mapping[str, Any],
    *,
    output_dir: str | Path,
    checkpoint_stem: str = "beta6_pretrain_checkpoint",
) -> Dict[str, str]:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    state = _as_mapping(payload.get("state"))
    mean = np.asarray(state.get("mean"), dtype=np.float32)
    std = np.asarray(state.get("std"), dtype=np.float32)
    components = np.asarray(state.get("components"), dtype=np.float32)
    npz_path = out_dir / f"{checkpoint_stem}.npz"
    np.savez(npz_path, mean=mean, std=std, components=components)

    metadata = {
        "version": str(payload.get("version", PRETRAIN_CONFIG_VERSION)),
        "config": dict(_as_mapping(payload.get("config"))),
        "metrics": dict(_as_mapping(payload.get("metrics"))),
        "checkpoint_fingerprint": hash_json_payload(
            {
                "config": payload.get("config", {}),
                "metrics": payload.get("metrics", {}),
            }
        ),
    }
    meta_path = out_dir / f"{checkpoint_stem}.metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "checkpoint_npz": str(npz_path),
        "checkpoint_metadata": str(meta_path),
    }


def load_pretrain_checkpoint(path: str | Path) -> Dict[str, np.ndarray]:
    npz = np.load(Path(path).resolve())
    return {
        "mean": np.asarray(npz["mean"], dtype=np.float32),
        "std": np.asarray(npz["std"], dtype=np.float32),
        "components": np.asarray(npz["components"], dtype=np.float32),
    }


def encode_with_checkpoint(matrix: np.ndarray, checkpoint: Mapping[str, np.ndarray]) -> np.ndarray:
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    components = np.asarray(checkpoint["components"], dtype=np.float32)
    normalized = (np.asarray(matrix, dtype=np.float32) - mean) / np.where(std < 1e-8, 1.0, std)
    return normalized @ components.T


def run_self_supervised_pretraining(
    *,
    manifest_path: str | Path,
    config_path: str | Path | None,
    output_dir: str | Path,
    max_files: Optional[int] = None,
) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    _assert_manifest_contract(manifest, manifest_path=manifest_path)
    config = load_pretrain_config(config_path)
    matrix = load_corpus_matrix(
        manifest_path,
        max_files=max_files,
        enforce_manifest_contract=False,
    )
    payload = train_self_supervised_encoder(matrix, config=config)
    artifacts = save_pretrain_checkpoint(payload, output_dir=output_dir)
    return {
        "status": "pass",
        "manifest_path": str(Path(manifest_path).resolve()),
        "config": payload["config"],
        "metrics": payload["metrics"],
        "artifacts": artifacts,
    }


__all__ = [
    "PRETRAIN_CONFIG_VERSION",
    "PretrainConfig",
    "encode_with_checkpoint",
    "load_corpus_matrix",
    "load_pretrain_checkpoint",
    "load_pretrain_config",
    "run_self_supervised_pretraining",
    "save_pretrain_checkpoint",
    "train_self_supervised_encoder",
]
