"""Projection of segment labels back to window labels."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def project_segments_to_window_labels(
    *,
    n_windows: int,
    labeled_segments: Sequence[Dict[str, object]],
    default_label: str = "unoccupied",
) -> np.ndarray:
    out = np.full(shape=(int(max(n_windows, 0)),), fill_value=str(default_label), dtype=object)
    n = int(len(out))
    for seg in labeled_segments:
        start = int(seg.get("start_idx", 0))
        end = int(seg.get("end_idx", 0))
        label = str(seg.get("label", default_label))
        start = max(start, 0)
        end = min(max(end, start), n)
        if end > start:
            out[start:end] = label
    return out
