"""Window-level leakage helper utilities and cache-key helpers for Beta 6 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence


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


def compute_feature_sequence_cache_key(
    *,
    manifest_fingerprint: str,
    policy_fingerprint: str,
    extra: dict | None = None,
) -> str:
    payload = {
        "manifest_fingerprint": str(manifest_fingerprint),
        "policy_fingerprint": str(policy_fingerprint),
        "extra": extra or {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def should_reuse_cached_tensors(
    cache_metadata: dict | None,
    *,
    manifest_fingerprint: str,
    policy_fingerprint: str,
) -> bool:
    if not isinstance(cache_metadata, dict):
        return False
    return (
        str(cache_metadata.get("manifest_fingerprint") or "") == str(manifest_fingerprint)
        and str(cache_metadata.get("policy_fingerprint") or "") == str(policy_fingerprint)
    )


def write_cache_metadata(
    path: str | Path,
    *,
    manifest_fingerprint: str,
    policy_fingerprint: str,
    cache_key: str,
) -> dict:
    payload = {
        "manifest_fingerprint": str(manifest_fingerprint),
        "policy_fingerprint": str(policy_fingerprint),
        "cache_key": str(cache_key),
    }
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


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


__all__ = [
    "Window",
    "compute_feature_sequence_cache_key",
    "has_resident_leakage",
    "has_time_leakage",
    "has_window_overlap",
    "should_reuse_cached_tensors",
    "write_cache_metadata",
]
