"""Beta 6 inference helpers with explicit unknown/abstain routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from ..beta6_schema import load_validated_beta6_config


@dataclass(frozen=True)
class UnknownPolicy:
    min_confidence: float = 0.55
    max_entropy: float = 1.05
    outside_sensed_space_threshold: float = 0.75
    abstain_label: str = "abstain"
    unknown_label: str = "unknown"
    low_confidence_state: str = "low_confidence"
    unknown_state: str = "unknown"
    outside_sensed_space_state: str = "outside_sensed_space"
    abstain_rate_min: float = 0.02
    abstain_rate_max: float = 0.20
    unknown_recall_min: float = 0.65


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def load_unknown_policy(path: str | Path | None) -> UnknownPolicy:
    policy_path = (
        Path(path).resolve()
        if path is not None
        else Path(__file__).resolve().parents[3] / "config" / "beta6_unknown_policy.yaml"
    )
    raw = load_validated_beta6_config(policy_path, expected_filename="beta6_unknown_policy.yaml")
    policy = _as_mapping(raw).get("unknown_policy")
    policy = _as_mapping(policy)
    targets = _as_mapping(raw).get("targets")
    targets = _as_mapping(targets)
    return UnknownPolicy(
        min_confidence=float(policy.get("min_confidence", 0.55)),
        max_entropy=float(policy.get("max_entropy", 1.05)),
        outside_sensed_space_threshold=float(policy.get("outside_sensed_space_threshold", 0.75)),
        abstain_label=str(policy.get("abstain_label", "abstain")).strip().lower(),
        unknown_label=str(policy.get("unknown_label", "unknown")).strip().lower(),
        low_confidence_state=str(policy.get("low_confidence_state", "low_confidence")).strip().lower(),
        unknown_state=str(policy.get("unknown_state", "unknown")).strip().lower(),
        outside_sensed_space_state=str(
            policy.get("outside_sensed_space_state", "outside_sensed_space")
        ).strip().lower(),
        abstain_rate_min=float(targets.get("abstain_rate_min", 0.02)),
        abstain_rate_max=float(targets.get("abstain_rate_max", 0.20)),
        unknown_recall_min=float(targets.get("unknown_recall_min", 0.65)),
    )


def _entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-9, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def infer_with_unknown_path(
    *,
    probabilities: np.ndarray,
    labels: Sequence[str],
    policy: UnknownPolicy,
    outside_sensed_space_scores: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"probabilities must be 2D, got {probs.shape}")
    if probs.shape[1] != len(labels):
        raise ValueError("labels length mismatch")
    norm = probs / np.where(probs.sum(axis=1, keepdims=True) <= 0.0, 1.0, probs.sum(axis=1, keepdims=True))

    labels_norm = [str(label).strip().lower() for label in labels]
    conf = norm.max(axis=1)
    pred_idx = norm.argmax(axis=1)
    entropy = _entropy(norm)
    if outside_sensed_space_scores is None:
        outside_scores = np.zeros(len(norm), dtype=np.float64)
    else:
        outside_scores = np.asarray(outside_sensed_space_scores, dtype=np.float64).reshape(-1)
        if len(outside_scores) != len(norm):
            raise ValueError("outside_sensed_space_scores length mismatch")

    predicted_labels = []
    uncertainty_states = []
    for i in range(len(norm)):
        if outside_scores[i] >= policy.outside_sensed_space_threshold:
            predicted_labels.append(policy.abstain_label)
            uncertainty_states.append(policy.outside_sensed_space_state)
            continue
        if conf[i] < policy.min_confidence:
            predicted_labels.append(policy.abstain_label)
            uncertainty_states.append(policy.low_confidence_state)
            continue
        if entropy[i] > policy.max_entropy:
            predicted_labels.append(policy.abstain_label)
            uncertainty_states.append(policy.unknown_state)
            continue
        predicted_labels.append(labels_norm[int(pred_idx[i])])
        uncertainty_states.append(None)

    abstain_count = int(sum(label == policy.abstain_label for label in predicted_labels))
    abstain_rate = float(abstain_count) / float(max(len(predicted_labels), 1))
    return {
        "labels": predicted_labels,
        "uncertainty_states": uncertainty_states,
        "confidence": conf.tolist(),
        "entropy": entropy.tolist(),
        "abstain_rate": abstain_rate,
        "policy_targets": {
            "abstain_rate_min": policy.abstain_rate_min,
            "abstain_rate_max": policy.abstain_rate_max,
            "unknown_recall_min": policy.unknown_recall_min,
        },
    }


def build_triage_candidates_from_inference(
    *,
    inference: Mapping[str, Any],
    room: str,
    activity_hint: str,
) -> list[Dict[str, Any]]:
    labels = list(inference.get("labels", []))
    confidence = list(inference.get("confidence", []))
    uncertainty_states = list(inference.get("uncertainty_states", []))
    rows = []
    for idx, label in enumerate(labels):
        if uncertainty_states[idx] is None:
            continue
        rows.append(
            {
                "candidate_id": f"{room}_{idx}",
                "room": str(room).strip().lower(),
                "activity": str(activity_hint).strip().lower(),
                "confidence": float(confidence[idx]),
                "predicted_label": str(label),
                "baseline_label": str(activity_hint).strip().lower(),
                "uncertainty_state": str(uncertainty_states[idx]),
            }
        )
    return rows


__all__ = [
    "UnknownPolicy",
    "build_triage_candidates_from_inference",
    "infer_with_unknown_path",
    "load_unknown_policy",
]
