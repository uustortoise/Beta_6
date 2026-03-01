"""Segment labeling with optional learned classifier and fallback."""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def _normalize_label(label: str) -> str:
    return str(label).strip().lower()


def _safe_float(value: float) -> float:
    val = float(value)
    if not np.isfinite(val):
        return 0.0
    return float(val)


def _safe_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(float)


def _feature_label_key(label: str) -> str:
    txt = _normalize_label(label)
    txt = re.sub(r"[^a-z0-9]+", "_", txt).strip("_")
    return txt or "unknown"


def _heuristic_segment_label(
    *,
    start: int,
    end: int,
    label_arrays: Mapping[str, np.ndarray],
    min_activity_prob: float,
) -> Tuple[str, float, Dict[str, float]]:
    label_scores: Dict[str, float] = {}
    best_label = "unoccupied"
    best_score = 0.0
    for label in sorted(label_arrays.keys()):
        vals = label_arrays[label]
        if len(vals) <= start:
            score = 0.0
        else:
            score = _safe_float(float(np.mean(vals[start:min(end, len(vals))])))
        label_scores[label] = float(score)
        if score > best_score:
            best_score = float(score)
            best_label = str(label)
    if float(best_score) < float(min_activity_prob):
        best_label = "unoccupied"
    return str(best_label), float(best_score), label_scores


def _build_classifier_matrix(
    *,
    segments: Sequence[Dict[str, int]],
    segment_features: Sequence[Dict[str, float]],
    label_scores: Sequence[Dict[str, float]],
) -> tuple[np.ndarray, List[str]]:
    feature_rows: List[List[float]] = []
    keys = sorted({str(k) for row in segment_features for k in row.keys() if str(k) not in {"start_idx", "end_idx"}})
    label_keys = sorted({str(k) for row in label_scores for k in row.keys()})
    ordered_columns = [f"feat::{k}" for k in keys] + [f"score::{k}" for k in label_keys]
    for idx, seg in enumerate(segments):
        row = segment_features[idx] if idx < len(segment_features) else {}
        values: List[float] = []
        for key in keys:
            values.append(_safe_float(float(row.get(key, 0.0))))
        scores = label_scores[idx] if idx < len(label_scores) else {}
        for key in label_keys:
            values.append(_safe_float(float(scores.get(key, 0.0))))
        # Include raw support as final classifier feature.
        start = int(seg.get("start_idx", 0))
        end = int(seg.get("end_idx", 0))
        values.append(float(max(end - start, 0)))
        feature_rows.append(values)
    ordered_columns.append("meta::support_windows")
    if not feature_rows:
        return np.zeros(shape=(0, len(ordered_columns)), dtype=float), ordered_columns
    return np.asarray(feature_rows, dtype=float), ordered_columns


def classify_occupied_segments(
    *,
    segments: Sequence[Dict[str, int]],
    activity_probs: Dict[str, Sequence[float]],
    segment_features: Optional[Sequence[Dict[str, float]]] = None,
    min_activity_prob: float = 0.35,
    enable_learned_classifier: bool = False,
    learned_classifier_min_segments: int = 8,
    learned_classifier_confidence_floor: float = 0.55,
    learned_classifier_min_windows: int = 6,
    random_state: int = 42,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    labels = sorted(_normalize_label(k) for k in activity_probs.keys() if _normalize_label(str(k)))
    label_arrays: Dict[str, np.ndarray] = {label: _safe_array(activity_probs.get(label, [])) for label in labels}
    normalized_features: List[Dict[str, float]] = []
    for row in list(segment_features or []):
        normalized_features.append({str(k): _safe_float(float(v)) for k, v in dict(row).items()})

    heur_labels: List[str] = []
    heur_scores: List[float] = []
    heur_label_scores: List[Dict[str, float]] = []
    valid_segments: List[Dict[str, int]] = []
    for seg in segments:
        start = int(seg.get("start_idx", 0))
        end = int(seg.get("end_idx", 0))
        if end <= start:
            continue
        valid_segments.append({"start_idx": int(start), "end_idx": int(end)})
        best_label, best_score, score_map = _heuristic_segment_label(
            start=start,
            end=end,
            label_arrays=label_arrays,
            min_activity_prob=float(min_activity_prob),
        )
        heur_labels.append(str(best_label))
        heur_scores.append(float(best_score))
        heur_label_scores.append(score_map)

    if not valid_segments:
        return out

    can_try_learned = bool(enable_learned_classifier) and bool(normalized_features) and len(normalized_features) >= len(
        valid_segments
    )
    learned_model: Optional[RandomForestClassifier] = None
    learned_probs: Optional[np.ndarray] = None
    learned_classes: List[str] = []
    learned_reason = "disabled"
    if can_try_learned:
        x_all, _ = _build_classifier_matrix(
            segments=valid_segments,
            segment_features=normalized_features,
            label_scores=heur_label_scores,
        )
        train_mask = np.asarray(
            [score >= float(min_activity_prob) and label != "unoccupied" for label, score in zip(heur_labels, heur_scores)],
            dtype=bool,
        )
        x_train = x_all[train_mask]
        y_train = np.asarray([heur_labels[i] for i, m in enumerate(train_mask.tolist()) if m], dtype=object)
        if len(x_train) < int(max(learned_classifier_min_segments, 2)):
            learned_reason = "insufficient_segments"
        elif len(set(y_train.tolist())) < 2:
            learned_reason = "insufficient_label_diversity"
        else:
            learned_model = RandomForestClassifier(
                n_estimators=120,
                max_depth=5,
                min_samples_leaf=1,
                random_state=int(random_state),
                n_jobs=1,
            )
            learned_model.fit(x_train, y_train)
            learned_probs = np.asarray(learned_model.predict_proba(x_all), dtype=float)
            learned_classes = [str(v) for v in learned_model.classes_.tolist()]
            learned_reason = "trained"
    elif bool(enable_learned_classifier):
        learned_reason = "missing_segment_features"

    for idx, seg in enumerate(valid_segments):
        start = int(seg["start_idx"])
        end = int(seg["end_idx"])
        heur_label = str(heur_labels[idx])
        heur_score = float(heur_scores[idx])
        support_windows = int(max(end - start, 0))
        final_label = heur_label
        final_score = heur_score
        selected = False
        confidence = heur_score
        fallback_reason: Optional[str] = None

        if learned_model is not None and learned_probs is not None and len(learned_classes) > 0:
            row_probs = learned_probs[idx]
            best_idx = int(np.argmax(row_probs))
            learned_label = str(learned_classes[best_idx])
            learned_conf = _safe_float(float(row_probs[best_idx]))
            confidence = learned_conf
            support_ok = support_windows >= int(max(learned_classifier_min_windows, 1))
            confidence_ok = learned_conf >= float(np.clip(learned_classifier_confidence_floor, 0.0, 1.0))
            if support_ok and confidence_ok:
                final_label = learned_label
                final_score = learned_conf
                selected = True
            else:
                fallback_reason = "low_support" if not support_ok else "low_confidence"
        elif bool(enable_learned_classifier):
            fallback_reason = str(learned_reason)

        if float(final_score) < float(min_activity_prob):
            final_label = "unoccupied"

        out.append(
            {
                "start_idx": int(start),
                "end_idx": int(end),
                "label": str(final_label),
                "score": _safe_float(float(final_score)),
                "base_label": str(heur_label),
                "base_score": _safe_float(float(heur_score)),
                "support_windows": int(support_windows),
                "classifier_mode": "learned" if learned_model is not None else "heuristic",
                "classifier_selected": bool(selected),
                "classifier_confidence": _safe_float(float(confidence)),
                "fallback_reason": fallback_reason,
            }
        )
    return out
