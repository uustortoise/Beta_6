"""Deterministic occupancy segment proposal utilities."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _iter_true_runs(mask: np.ndarray) -> List[tuple[int, int]]:
    out: List[tuple[int, int]] = []
    arr = np.asarray(mask, dtype=bool)
    n = int(len(arr))
    i = 0
    while i < n:
        if not arr[i]:
            i += 1
            continue
        start = i
        while i < n and arr[i]:
            i += 1
        out.append((int(start), int(i)))
    return out


def _fill_small_gaps(mask: np.ndarray, *, gap_merge_windows: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    n = int(len(out))
    if n <= 0 or int(gap_merge_windows) <= 0:
        return out
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        start = i
        while i < n and not out[i]:
            i += 1
        end = i
        has_left = start > 0 and out[start - 1]
        has_right = end < n and out[end]
        if has_left and has_right and (end - start) <= int(gap_merge_windows):
            out[start:end] = True
    return out


def _drop_short_runs(mask: np.ndarray, *, min_duration_windows: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    min_windows = int(max(min_duration_windows, 1))
    if min_windows <= 1:
        return out
    for start, end in _iter_true_runs(out):
        if (end - start) < min_windows:
            out[start:end] = False
    return out


def propose_occupancy_segments(
    occupancy_probs: Sequence[float],
    *,
    threshold: float = 0.5,
    min_duration_windows: int = 6,
    gap_merge_windows: int = 3,
) -> List[Dict[str, int]]:
    """
    Build occupied segment proposals from occupancy probabilities.

    Returns half-open intervals [start_idx, end_idx).
    """
    probs = np.asarray(occupancy_probs, dtype=float)
    if len(probs) <= 0:
        return []
    mask = np.asarray(probs >= float(threshold), dtype=bool)
    mask = _fill_small_gaps(mask, gap_merge_windows=int(max(gap_merge_windows, 0)))
    mask = _drop_short_runs(mask, min_duration_windows=int(max(min_duration_windows, 1)))
    segments: List[Dict[str, int]] = []
    for start, end in _iter_true_runs(mask):
        segments.append({"start_idx": int(start), "end_idx": int(end)})
    return segments
