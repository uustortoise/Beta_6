"""Phase 5 LoRA-style resident adapter lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import uuid

import numpy as np
import pandas as pd

from ..beta6_schema import load_validated_beta6_config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _softmax(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    arr = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(arr)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom = np.where(denom <= 1e-12, 1.0, denom)
    return (exp / denom).astype(np.float64)


@dataclass(frozen=True)
class LoRAAdapterConfig:
    rank: int = 4
    alpha: float = 8.0
    l2_reg: float = 1e-3
    random_seed: int = 42
    min_rows: int = 32


@dataclass(frozen=True)
class LoRAAdapterArtifact:
    version: str
    adapter_id: str
    resident_id: str
    room: str
    backbone_id: str
    created_at: str
    feature_keys: Tuple[str, ...]
    class_labels: Tuple[str, ...]
    rank: int
    alpha: float
    low_rank_a: np.ndarray
    low_rank_b: np.ndarray
    bias: np.ndarray
    warmup_accuracy: float
    row_count: int

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(max(int(self.rank), 1))

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "adapter_id": self.adapter_id,
            "resident_id": self.resident_id,
            "room": self.room,
            "backbone_id": self.backbone_id,
            "created_at": self.created_at,
            "feature_keys": list(self.feature_keys),
            "class_labels": list(self.class_labels),
            "rank": int(self.rank),
            "alpha": float(self.alpha),
            "scaling": float(self.scaling),
            "warmup_accuracy": float(self.warmup_accuracy),
            "row_count": int(self.row_count),
        }


def load_adapter_config(path: str | Path | None) -> LoRAAdapterConfig:
    policy_path = (
        Path(path).resolve()
        if path is not None
        else Path(__file__).resolve().parents[3] / "config" / "beta6_adapter_policy.yaml"
    )
    payload = load_validated_beta6_config(
        policy_path,
        expected_filename="beta6_adapter_policy.yaml",
    )
    section = payload.get("adapter")
    if not isinstance(section, Mapping):
        section = {}
    return LoRAAdapterConfig(
        rank=max(int(section.get("rank", 4)), 1),
        alpha=max(float(section.get("alpha", 8.0)), 1e-6),
        l2_reg=max(float(section.get("l2_reg", 1e-3)), 0.0),
        random_seed=int(section.get("random_seed", 42)),
        min_rows=max(int(section.get("min_rows", 32)), 1),
    )


def _infer_feature_keys(frame: pd.DataFrame, *, label_col: str) -> Tuple[str, ...]:
    feature_cols = []
    excluded = {
        label_col,
        "label",
        "activity",
        "room",
        "resident_id",
        "elder_id",
        "timestamp",
    }
    for col in frame.columns:
        if str(col) in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            feature_cols.append(str(col))
    if not feature_cols:
        raise ValueError("adapter training requires numeric feature columns")
    return tuple(sorted(feature_cols))


def _build_low_rank_delta(
    x: np.ndarray,
    y: np.ndarray,
    *,
    class_labels: Sequence[str],
    rank: int,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Deterministic class-centroid residual map used as LoRA-style low-rank delta.
    feature_mean = np.mean(x, axis=0, keepdims=False)
    centroid_rows = []
    class_priors = []
    total_rows = max(int(len(y)), 1)
    for label in class_labels:
        mask = (y == label)
        count = int(np.sum(mask))
        class_priors.append(float(count) / float(total_rows))
        if count == 0:
            centroid_rows.append(np.zeros((x.shape[1],), dtype=np.float64))
            continue
        centroid_rows.append(np.mean(x[mask], axis=0))
    centroid = np.vstack(centroid_rows)
    delta = (centroid - feature_mean).T  # [feature_dim, class_count]
    u, s, vt = np.linalg.svd(delta, full_matrices=False)
    use_rank = max(1, min(int(rank), int(delta.shape[0]), int(delta.shape[1])))
    u_r = u[:, :use_rank]
    s_r = s[:use_rank]
    vt_r = vt[:use_rank, :]
    sqrt_s = np.sqrt(np.clip(s_r, a_min=0.0, a_max=None))
    a = u_r * sqrt_s[np.newaxis, :]
    b = (sqrt_s[:, np.newaxis] * vt_r) * (float(alpha) / float(use_rank))
    priors = np.clip(np.asarray(class_priors, dtype=np.float64), 1e-9, 1.0)
    bias = np.log(priors)
    return a.astype(np.float64), b.astype(np.float64), bias.astype(np.float64)


def score_with_adapter(
    features: np.ndarray,
    *,
    low_rank_a: np.ndarray,
    low_rank_b: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    logits = x @ (np.asarray(low_rank_a, dtype=np.float64) @ np.asarray(low_rank_b, dtype=np.float64))
    logits = logits + np.asarray(bias, dtype=np.float64)
    return _softmax(logits)


def train_lora_adapter_from_frame(
    frame: pd.DataFrame,
    *,
    resident_id: str,
    room: str,
    backbone_id: str,
    config: LoRAAdapterConfig,
    label_col: str = "activity",
    adapter_id: Optional[str] = None,
) -> LoRAAdapterArtifact:
    if label_col not in frame.columns:
        raise ValueError(f"missing label column for adapter training: {label_col}")
    if len(frame) < int(config.min_rows):
        raise ValueError(
            f"insufficient rows for adapter warm-up: rows={len(frame)} min_rows={config.min_rows}"
        )

    feature_keys = _infer_feature_keys(frame, label_col=label_col)
    work = frame[list(feature_keys) + [label_col]].copy()
    work = work.dropna(subset=[label_col])
    if work.empty:
        raise ValueError("adapter training frame is empty after label filtering")
    for key in feature_keys:
        work[key] = pd.to_numeric(work[key], errors="coerce")
    work = work.dropna(subset=list(feature_keys))
    if work.empty:
        raise ValueError("adapter training frame is empty after numeric feature filtering")

    y = work[label_col].map(_norm_token).astype(str).to_numpy(dtype=object)
    class_labels = tuple(sorted({label for label in y if label}))
    if len(class_labels) < 2:
        raise ValueError("adapter training requires at least 2 classes")

    x = work[list(feature_keys)].to_numpy(dtype=np.float64)
    a, b, bias = _build_low_rank_delta(
        x,
        y,
        class_labels=class_labels,
        rank=config.rank,
        alpha=config.alpha,
    )
    probs = score_with_adapter(x, low_rank_a=a, low_rank_b=b, bias=bias)
    pred_idx = np.argmax(probs, axis=1)
    pred = np.asarray([class_labels[idx] for idx in pred_idx], dtype=object)
    accuracy = float(np.mean(pred == y))

    resolved_adapter_id = str(adapter_id or f"adapter_{uuid.uuid4().hex[:12]}")
    return LoRAAdapterArtifact(
        version="v1",
        adapter_id=resolved_adapter_id,
        resident_id=str(resident_id).strip(),
        room=_norm_token(room),
        backbone_id=str(backbone_id).strip(),
        created_at=_utc_now(),
        feature_keys=feature_keys,
        class_labels=class_labels,
        rank=int(a.shape[1]),
        alpha=float(config.alpha),
        low_rank_a=a,
        low_rank_b=b,
        bias=bias,
        warmup_accuracy=accuracy,
        row_count=int(x.shape[0]),
    )


def save_adapter_artifact(artifact: LoRAAdapterArtifact, *, output_dir: str | Path) -> Dict[str, str]:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.npz"
    metadata_path = out_dir / "metadata.json"
    np.savez(
        weights_path,
        low_rank_a=np.asarray(artifact.low_rank_a, dtype=np.float64),
        low_rank_b=np.asarray(artifact.low_rank_b, dtype=np.float64),
        bias=np.asarray(artifact.bias, dtype=np.float64),
    )
    metadata_path.write_text(json.dumps(artifact.to_metadata(), indent=2), encoding="utf-8")
    return {
        "weights_npz": str(weights_path),
        "metadata_json": str(metadata_path),
    }


def load_adapter_artifact(path: str | Path) -> LoRAAdapterArtifact:
    adapter_dir = Path(path).resolve()
    metadata = json.loads((adapter_dir / "metadata.json").read_text(encoding="utf-8"))
    npz = np.load(adapter_dir / "weights.npz")
    return LoRAAdapterArtifact(
        version=str(metadata.get("version", "v1")),
        adapter_id=str(metadata.get("adapter_id", "")),
        resident_id=str(metadata.get("resident_id", "")),
        room=_norm_token(metadata.get("room", "")),
        backbone_id=str(metadata.get("backbone_id", "")),
        created_at=str(metadata.get("created_at", "")),
        feature_keys=tuple(str(v) for v in metadata.get("feature_keys", [])),
        class_labels=tuple(str(v) for v in metadata.get("class_labels", [])),
        rank=int(metadata.get("rank", 1)),
        alpha=float(metadata.get("alpha", 1.0)),
        low_rank_a=np.asarray(npz["low_rank_a"], dtype=np.float64),
        low_rank_b=np.asarray(npz["low_rank_b"], dtype=np.float64),
        bias=np.asarray(npz["bias"], dtype=np.float64),
        warmup_accuracy=float(metadata.get("warmup_accuracy", 0.0)),
        row_count=int(metadata.get("row_count", 0)),
    )


__all__ = [
    "LoRAAdapterArtifact",
    "LoRAAdapterConfig",
    "load_adapter_artifact",
    "load_adapter_config",
    "save_adapter_artifact",
    "score_with_adapter",
    "train_lora_adapter_from_frame",
]
