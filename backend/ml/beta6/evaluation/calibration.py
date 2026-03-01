"""Calibration utilities for Beta 6 unknown/abstain inference path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class CalibrationReport:
    brier_score: float
    expected_calibration_error: float
    mean_confidence: float
    accuracy: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "brier_score": float(self.brier_score),
            "expected_calibration_error": float(self.expected_calibration_error),
            "mean_confidence": float(self.mean_confidence),
            "accuracy": float(self.accuracy),
        }


def _as_prob_matrix(probabilities: np.ndarray) -> np.ndarray:
    arr = np.asarray(probabilities, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"probabilities must be 2D, got {arr.shape}")
    if (arr < 0).any():
        raise ValueError("probabilities must be non-negative")
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
    return arr / row_sums


def multiclass_brier_score(probabilities: np.ndarray, true_indices: Sequence[int]) -> float:
    probs = _as_prob_matrix(probabilities)
    truth = np.asarray(true_indices, dtype=np.int64)
    if len(truth) != probs.shape[0]:
        raise ValueError("true_indices length mismatch")
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(truth)), truth] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    probabilities: np.ndarray,
    true_indices: Sequence[int],
    *,
    bins: int = 10,
) -> float:
    probs = _as_prob_matrix(probabilities)
    truth = np.asarray(true_indices, dtype=np.int64)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == truth).astype(np.float64)
    edges = np.linspace(0.0, 1.0, num=max(int(bins), 2) + 1)

    ece = 0.0
    for idx in range(len(edges) - 1):
        lo, hi = edges[idx], edges[idx + 1]
        if idx == len(edges) - 2:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc = float(correct[mask].mean())
        mean_conf = float(conf[mask].mean())
        ece += (float(mask.mean()) * abs(acc - mean_conf))
    return float(ece)


def evaluate_calibration(probabilities: np.ndarray, true_indices: Sequence[int]) -> CalibrationReport:
    probs = _as_prob_matrix(probabilities)
    truth = np.asarray(true_indices, dtype=np.int64)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    return CalibrationReport(
        brier_score=multiclass_brier_score(probs, truth),
        expected_calibration_error=expected_calibration_error(probs, truth, bins=10),
        mean_confidence=float(conf.mean()),
        accuracy=float((pred == truth).mean()),
    )


__all__ = [
    "CalibrationReport",
    "evaluate_calibration",
    "expected_calibration_error",
    "multiclass_brier_score",
]
