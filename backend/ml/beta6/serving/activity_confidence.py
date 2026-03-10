"""Learned activity-confidence scoring for Beta 6 room heads."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve


ACTIVITY_CONFIDENCE_SCHEMA_VERSION = "activity_acceptance_score_v1"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"probabilities must be 2D, got shape={probs.shape}")
    sums = probs.sum(axis=1, keepdims=True)
    safe = np.where(sums <= 0.0, 1.0, sums)
    return probs / safe


def build_activity_confidence_features(
    *,
    probabilities: np.ndarray,
    labels: Sequence[str],
) -> dict[str, Any]:
    probs = _normalize_probabilities(probabilities)
    top1_idx = probs.argmax(axis=1).astype(np.int64, copy=False)
    top1_probs = probs[np.arange(len(probs)), top1_idx]
    if probs.shape[1] > 1:
        top2_probs = np.partition(probs, -2, axis=1)[:, -2]
    else:
        top2_probs = np.zeros(len(probs), dtype=np.float64)
    margins = top1_probs - top2_probs
    clipped = np.clip(probs, 1e-9, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    entropy_scale = float(np.log(max(probs.shape[1], 2)))
    normalized_entropy = entropy / max(entropy_scale, 1e-9)

    feature_columns = [
        top1_probs.reshape(-1, 1),
        top2_probs.reshape(-1, 1),
        margins.reshape(-1, 1),
        normalized_entropy.reshape(-1, 1),
    ]
    feature_names = ["top1_probability", "top2_probability", "margin", "normalized_entropy"]
    for class_idx, label in enumerate(labels):
        indicator = (top1_idx == int(class_idx)).astype(np.float64, copy=False).reshape(-1, 1)
        feature_columns.append(indicator)
        feature_names.append(f"predicted_class:{str(label).strip().lower()}")

    matrix = np.concatenate(feature_columns, axis=1) if feature_columns else np.zeros((len(probs), 0))
    return {
        "matrix": matrix,
        "feature_names": feature_names,
        "top1_probabilities": top1_probs.astype(np.float64, copy=False),
        "top2_probabilities": top2_probs.astype(np.float64, copy=False),
        "margins": margins.astype(np.float64, copy=False),
        "entropy": normalized_entropy.astype(np.float64, copy=False),
        "predicted_indices": top1_idx,
    }


def fit_activity_confidence_calibrator(
    *,
    probabilities: np.ndarray,
    true_indices: Sequence[int],
    labels: Sequence[str],
    min_samples: int = 80,
    min_positive: int = 10,
    min_negative: int = 10,
) -> Optional[dict[str, Any]]:
    features = build_activity_confidence_features(probabilities=probabilities, labels=labels)
    predicted_indices = np.asarray(features["predicted_indices"], dtype=np.int64)
    truth = np.asarray(true_indices, dtype=np.int64).reshape(-1)
    if len(truth) != len(predicted_indices):
        raise ValueError("true_indices length mismatch")

    outcomes = (predicted_indices == truth).astype(np.int64, copy=False)
    positives = int(np.sum(outcomes))
    negatives = int(len(outcomes) - positives)
    if len(outcomes) < int(min_samples) or positives < int(min_positive) or negatives < int(min_negative):
        return None

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    model.fit(features["matrix"], outcomes)
    scores = model.predict_proba(features["matrix"])[:, 1]

    return {
        "schema_version": ACTIVITY_CONFIDENCE_SCHEMA_VERSION,
        "labels": [str(label).strip().lower() for label in labels],
        "feature_names": list(features["feature_names"]),
        "coefficients": model.coef_[0].astype(float).tolist(),
        "intercept": float(model.intercept_[0]),
        "diagnostics": {
            "n_samples": int(len(outcomes)),
            "positive_count": int(positives),
            "negative_count": int(negatives),
            "mean_acceptance_score": float(np.mean(scores)),
        },
    }


def score_activity_confidence(
    *,
    probabilities: np.ndarray,
    labels: Sequence[str],
    artifact: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    features = build_activity_confidence_features(probabilities=probabilities, labels=labels)
    if not artifact:
        return {
            "scores": features["top1_probabilities"].astype(np.float64, copy=False),
            "confidence_source": "raw_top1_confidence",
            "top1_probabilities": features["top1_probabilities"],
            "top2_probabilities": features["top2_probabilities"],
            "margins": features["margins"],
            "entropy": features["entropy"],
            "predicted_indices": features["predicted_indices"],
        }

    coefficients = np.asarray(list(artifact.get("coefficients", [])), dtype=np.float64)
    intercept = float(artifact.get("intercept", 0.0))
    if coefficients.shape[0] != features["matrix"].shape[1]:
        raise ValueError(
            "activity confidence artifact feature width mismatch: "
            f"{coefficients.shape[0]} vs {features['matrix'].shape[1]}"
        )
    logits = features["matrix"] @ coefficients + intercept
    scores = _sigmoid(logits)
    return {
        "scores": scores.astype(np.float64, copy=False),
        "confidence_source": str(
            artifact.get("schema_version", ACTIVITY_CONFIDENCE_SCHEMA_VERSION)
        ).strip()
        or ACTIVITY_CONFIDENCE_SCHEMA_VERSION,
        "top1_probabilities": features["top1_probabilities"],
        "top2_probabilities": features["top2_probabilities"],
        "margins": features["margins"],
        "entropy": features["entropy"],
        "predicted_indices": features["predicted_indices"],
    }


def choose_activity_confidence_threshold(
    *,
    scores: np.ndarray,
    outcomes: np.ndarray,
    target_precision: float,
    recall_floor: float,
    threshold_floor: float,
    threshold_cap: float,
    stability_window: float = 0.03,
    max_near_threshold_share: float = 0.20,
) -> dict[str, Any]:
    raw_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y_true = np.asarray(outcomes, dtype=np.int64).reshape(-1)
    if len(raw_scores) != len(y_true):
        raise ValueError("scores/outcomes length mismatch")
    if len(raw_scores) == 0 or int(np.sum(y_true)) <= 0:
        final_threshold = float(np.clip(threshold_floor, threshold_floor, threshold_cap))
        return {
            "threshold": final_threshold,
            "status": "fallback_empty",
            "near_threshold_share": 0.0,
            "selected_precision": 0.0,
            "selected_recall": 0.0,
        }

    precision, recall, thresholds = precision_recall_curve(y_true, raw_scores)
    if len(thresholds) == 0:
        final_threshold = float(np.clip(threshold_floor, threshold_floor, threshold_cap))
        return {
            "threshold": final_threshold,
            "status": "fallback_no_curve",
            "near_threshold_share": 0.0,
            "selected_precision": 0.0,
            "selected_recall": 0.0,
        }

    candidates: list[dict[str, Any]] = []
    for idx, raw_threshold in enumerate(thresholds):
        threshold = float(np.clip(raw_threshold, threshold_floor, threshold_cap))
        near_share = float(np.mean(np.abs(raw_scores - threshold) <= float(stability_window)))
        p = float(precision[idx])
        r = float(recall[idx])
        f1 = float((2.0 * p * r) / max(p + r, 1e-9))
        candidates.append(
            {
                "threshold": threshold,
                "near_threshold_share": near_share,
                "precision": p,
                "recall": r,
                "f1": f1,
                "valid": bool(p >= float(target_precision) and r >= float(recall_floor)),
            }
        )

    stable_valid = [c for c in candidates if c["valid"] and c["near_threshold_share"] <= max_near_threshold_share]
    if stable_valid:
        chosen = min(stable_valid, key=lambda c: (c["threshold"], -c["recall"]))
        return {
            "threshold": float(chosen["threshold"]),
            "status": "target_met",
            "near_threshold_share": float(chosen["near_threshold_share"]),
            "selected_precision": float(chosen["precision"]),
            "selected_recall": float(chosen["recall"]),
        }

    valid = [c for c in candidates if c["valid"]]
    if valid:
        chosen = min(valid, key=lambda c: (c["near_threshold_share"], c["threshold"]))
        lowered = float(np.clip(chosen["threshold"] - float(stability_window), threshold_floor, threshold_cap))
        return {
            "threshold": lowered,
            "status": "target_met_stability_fallback",
            "near_threshold_share": float(chosen["near_threshold_share"]),
            "selected_precision": float(chosen["precision"]),
            "selected_recall": float(chosen["recall"]),
            "selected_threshold_before_fallback": float(chosen["threshold"]),
        }

    stable = [c for c in candidates if c["near_threshold_share"] <= max_near_threshold_share]
    if stable:
        chosen = max(stable, key=lambda c: (c["f1"], c["recall"], -c["threshold"]))
        return {
            "threshold": float(chosen["threshold"]),
            "status": "fallback_best_f1_stability_fallback",
            "near_threshold_share": float(chosen["near_threshold_share"]),
            "selected_precision": float(chosen["precision"]),
            "selected_recall": float(chosen["recall"]),
        }

    chosen = max(candidates, key=lambda c: (c["f1"], -c["near_threshold_share"], c["recall"]))
    lowered = float(np.clip(chosen["threshold"] - float(stability_window), threshold_floor, threshold_cap))
    return {
        "threshold": lowered,
        "status": "fallback_best_f1_stability_fallback",
        "near_threshold_share": float(chosen["near_threshold_share"]),
        "selected_precision": float(chosen["precision"]),
        "selected_recall": float(chosen["recall"]),
        "selected_threshold_before_fallback": float(chosen["threshold"]),
    }


__all__ = [
    "ACTIVITY_CONFIDENCE_SCHEMA_VERSION",
    "build_activity_confidence_features",
    "choose_activity_confidence_threshold",
    "fit_activity_confidence_calibrator",
    "score_activity_confidence",
]
