"""Representation quality evaluation for Beta 6 pretrained encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


@dataclass(frozen=True)
class RepresentationEvalResult:
    linear_probe_accuracy: float
    random_probe_accuracy: float
    knn_purity: float
    improvement_margin: float
    train_residents: int
    test_residents: int
    train_rows: int
    test_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "linear_probe_accuracy": float(self.linear_probe_accuracy),
            "random_probe_accuracy": float(self.random_probe_accuracy),
            "knn_purity": float(self.knn_purity),
            "improvement_margin": float(self.improvement_margin),
            "train_residents": int(self.train_residents),
            "test_residents": int(self.test_residents),
            "train_rows": int(self.train_rows),
            "test_rows": int(self.test_rows),
        }


def _resident_disjoint_mask(
    resident_ids: Sequence[str],
    *,
    seed: int,
    test_fraction: float,
) -> np.ndarray:
    unique = sorted({str(rid) for rid in resident_ids})
    if len(unique) < 2:
        raise ValueError("resident-disjoint split requires at least 2 unique residents")
    rng = np.random.default_rng(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)
    split_index = max(1, int(round(len(shuffled) * (1.0 - test_fraction))))
    split_index = min(split_index, len(shuffled) - 1)
    train_residents = set(shuffled[:split_index])
    return np.array([str(rid) in train_residents for rid in resident_ids], dtype=bool)


def _fit_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    seed: int,
) -> float:
    probe = LogisticRegression(max_iter=400, random_state=seed)
    probe.fit(train_x, train_y)
    pred = probe.predict(test_x)
    return float(accuracy_score(test_y, pred))


def evaluate_representation_quality(
    *,
    embeddings: np.ndarray,
    labels: Sequence[str] | np.ndarray,
    resident_ids: Sequence[str],
    seed: int = 42,
    test_fraction: float = 0.2,
    knn_k: int = 5,
) -> RepresentationEvalResult:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={matrix.shape}")
    label_arr = np.asarray([str(label) for label in labels], dtype=object)
    resident_arr = np.asarray([str(rid) for rid in resident_ids], dtype=object)
    if not (len(matrix) == len(label_arr) == len(resident_arr)):
        raise ValueError("embeddings, labels, and resident_ids must have identical lengths")
    if len(matrix) < 16:
        raise ValueError("representation eval requires at least 16 rows")

    train_mask = _resident_disjoint_mask(resident_arr, seed=seed, test_fraction=test_fraction)
    if train_mask.sum() == 0 or (~train_mask).sum() == 0:
        raise ValueError("resident-disjoint split produced empty train/test set")

    train_x = matrix[train_mask]
    test_x = matrix[~train_mask]
    train_y = label_arr[train_mask]
    test_y = label_arr[~train_mask]

    linear_acc = _fit_probe(train_x, train_y, test_x, test_y, seed=seed)

    rng = np.random.default_rng(seed + 1)
    random_train_x = rng.standard_normal(size=train_x.shape, dtype=np.float32)
    random_test_x = rng.standard_normal(size=test_x.shape, dtype=np.float32)
    random_acc = _fit_probe(random_train_x, train_y, random_test_x, test_y, seed=seed + 1)

    neighbors = max(1, min(int(knn_k), len(train_x)))
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(train_x, train_y)
    knn_pred = knn.predict(test_x)
    knn_purity = float(accuracy_score(test_y, knn_pred))

    return RepresentationEvalResult(
        linear_probe_accuracy=linear_acc,
        random_probe_accuracy=random_acc,
        knn_purity=knn_purity,
        improvement_margin=float(linear_acc - random_acc),
        train_residents=int(len(set(resident_arr[train_mask]))),
        test_residents=int(len(set(resident_arr[~train_mask]))),
        train_rows=int(train_mask.sum()),
        test_rows=int((~train_mask).sum()),
    )


__all__ = [
    "RepresentationEvalResult",
    "evaluate_representation_quality",
]
