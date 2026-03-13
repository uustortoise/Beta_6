"""Window-level leakage helper utilities for Beta 6 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .feature_fingerprint import hash_json_payload


@dataclass(frozen=True)
class Window:
    """Canonical window tuple for leakage checks.

    Times are represented in seconds (epoch or relative), and comparisons are
    purely numeric.
    """

    resident_id: str
    start_ts: float
    end_ts: float


WindowLike = Window | tuple[str, float, float] | list[object]


def _coerce_window(value: WindowLike) -> Window:
    if isinstance(value, Window):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 3:
        resident_id, start_ts, end_ts = value
        return Window(
            resident_id=str(resident_id),
            start_ts=float(start_ts),
            end_ts=float(end_ts),
        )
    raise TypeError(
        "window must be Window or (resident_id, start_ts, end_ts) tuple/list; "
        f"got {type(value)!r}"
    )


def _normalize_range(start_ts: float, end_ts: float) -> tuple[float, float]:
    if start_ts <= end_ts:
        return start_ts, end_ts
    return end_ts, start_ts


def _intervals_overlap(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
    *,
    gap_seconds: float = 0.0,
) -> bool:
    a_start, a_end = _normalize_range(a_start, a_end)
    b_start, b_end = _normalize_range(b_start, b_end)
    gap = float(max(gap_seconds, 0.0))

    # Symmetric gap buffer: allow a guard band around both intervals.
    return not ((a_end + gap) < b_start or (b_end + gap) < a_start)


def has_resident_leakage(
    train_resident_ids: Iterable[str],
    validation_resident_ids: Iterable[str],
) -> bool:
    train = {str(value).strip() for value in train_resident_ids}
    validation = {str(value).strip() for value in validation_resident_ids}
    return bool(train & validation)


def has_time_leakage(
    train_windows: Sequence[WindowLike],
    validation_windows: Sequence[WindowLike],
    *,
    gap_seconds: float = 0.0,
) -> bool:
    train = [_coerce_window(value) for value in train_windows]
    validation = [_coerce_window(value) for value in validation_windows]

    for train_window in train:
        for validation_window in validation:
            if train_window.resident_id != validation_window.resident_id:
                continue
            if _intervals_overlap(
                train_window.start_ts,
                train_window.end_ts,
                validation_window.start_ts,
                validation_window.end_ts,
                gap_seconds=gap_seconds,
            ):
                return True
    return False


def has_window_overlap(
    train_windows: Sequence[WindowLike],
    validation_windows: Sequence[WindowLike],
    *,
    gap_seconds: float = 0.0,
) -> bool:
    return has_time_leakage(
        train_windows,
        validation_windows,
        gap_seconds=gap_seconds,
    )


def build_feature_sequence_cache_key(
    *,
    manifest_fingerprint: str,
    policy_fingerprint: str,
    stage: str,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Build a canonical cache key for reusable feature/sequence tensors."""
    payload = {
        "manifest_fingerprint": str(manifest_fingerprint or "").strip(),
        "policy_fingerprint": str(policy_fingerprint or "").strip(),
        "stage": str(stage or "").strip().lower(),
        "extra": dict(extra or {}),
    }
    return hash_json_payload(payload)


def load_cached_tensor(
    cache_dir: str | Path,
    *,
    cache_key: str,
) -> dict[str, Any] | None:
    """Load a cached tensor bundle when the exact key is present."""
    root = Path(cache_dir).resolve()
    tensor_path = root / f"{cache_key}.npy"
    metadata_path = root / f"{cache_key}.json"
    if not tensor_path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("cache_key") or "") != str(cache_key):
            return None
        array = np.load(tensor_path, allow_pickle=False)
    except Exception:
        return None
    return {
        "array": np.asarray(array, dtype=np.float32),
        "metadata": metadata,
        "tensor_path": str(tensor_path),
        "metadata_path": str(metadata_path),
    }


def save_cached_tensor(
    cache_dir: str | Path,
    *,
    cache_key: str,
    array: np.ndarray,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Persist a reusable tensor bundle keyed by manifest/policy fingerprint."""
    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tensor_path = root / f"{cache_key}.npy"
    metadata_path = root / f"{cache_key}.json"
    np.save(tensor_path, np.asarray(array, dtype=np.float32), allow_pickle=False)
    payload = {"cache_key": str(cache_key), **dict(metadata or {})}
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "tensor_path": str(tensor_path),
        "metadata_path": str(metadata_path),
    }


__all__ = [
    "Window",
    "build_feature_sequence_cache_key",
    "has_resident_leakage",
    "has_time_leakage",
    "has_window_overlap",
    "load_cached_tensor",
    "save_cached_tensor",
]
