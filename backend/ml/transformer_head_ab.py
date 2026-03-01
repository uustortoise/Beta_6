"""
Head-A / Head-B output contract helpers for event-first shadow mode.

This module converts existing multi-class softmax outputs into:
- Head A: occupancy probability (occupied vs unoccupied)
- Head B: conditional activity probabilities given occupied
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class DualHeadProbabilities:
    """Container for dual-head probabilities."""

    occupancy_prob: np.ndarray
    activity_probs: Dict[str, np.ndarray]
    occupancy_label: str = "unoccupied"

    def to_probability_frame(self) -> Dict[str, np.ndarray]:
        """
        Return a dict form suitable for downstream dataframe creation.
        """
        payload: Dict[str, np.ndarray] = {"prob_occupied": self.occupancy_prob}
        for label, probs in self.activity_probs.items():
            payload[f"prob_{label}"] = probs
        return payload


def derive_dual_head_probabilities(
    multiclass_probs: np.ndarray,
    class_names: Sequence[str],
    *,
    occupancy_label: str = "unoccupied",
    epsilon: float = 1e-8,
) -> DualHeadProbabilities:
    """
    Convert a single softmax output into head-A/head-B probabilities.

    Parameters
    ----------
    multiclass_probs:
        Shape (n_samples, n_classes) softmax output.
    class_names:
        Label names aligned with output columns.
    occupancy_label:
        Label representing unoccupied state.
    epsilon:
        Numerical floor for normalization.
    """
    probs = np.asarray(multiclass_probs, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError("multiclass_probs must be 2D [n_samples, n_classes]")
    if probs.shape[1] != len(class_names):
        raise ValueError("class_names length must match multiclass_probs.shape[1]")
    if probs.shape[1] == 0:
        raise ValueError("multiclass_probs must include at least one class")

    normalized_names = [str(name).strip().lower() for name in class_names]
    occ_key = str(occupancy_label).strip().lower()
    if occ_key not in normalized_names:
        raise ValueError(f"occupancy label '{occupancy_label}' not found in class_names")

    occ_idx = normalized_names.index(occ_key)
    p_unoccupied = np.clip(probs[:, occ_idx], 0.0, 1.0)
    p_occupied = np.clip(1.0 - p_unoccupied, 0.0, 1.0)

    denom = np.maximum(p_occupied, float(epsilon))
    activity_probs: Dict[str, np.ndarray] = {}
    for idx, label in enumerate(class_names):
        label_name = str(label).strip().lower()
        if label_name == occ_key:
            continue
        conditional = probs[:, idx] / denom
        conditional = np.where(p_occupied > float(epsilon), conditional, 0.0)
        activity_probs[label_name] = np.clip(conditional.astype(np.float32), 0.0, 1.0)

    return DualHeadProbabilities(
        occupancy_prob=p_occupied.astype(np.float32),
        activity_probs=activity_probs,
        occupancy_label=occ_key,
    )

