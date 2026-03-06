"""Evaluation helpers for Beta 6 leakage guardrails and dynamic gate artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from .calibration import evaluate_calibration
from ..data.feature_store import Window, has_resident_leakage, has_time_leakage, has_window_overlap
from ..serving.prediction import UnknownPolicy
from ...timeline_metrics import compute_timeline_metrics


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sign_payload(payload: Mapping[str, Any], *, signing_key: str) -> str:
    body = _canonical(payload).encode("utf-8")
    digest = hashlib.sha256(signing_key.encode("utf-8") + b"|" + body).hexdigest()
    return f"sha256:{digest}"


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _to_label_array(values: Sequence[Any]) -> list[str]:
    return [str(v).strip().lower() for v in values]


def _to_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _normalize_timeline_metrics(timeline_metrics: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize timeline metrics into the Step3 S3-01 payload contract."""
    if not isinstance(timeline_metrics, Mapping):
        return {}

    payload: Dict[str, Any] = dict(timeline_metrics)
    binary = payload.get("timeline_metrics_binary")
    binary_map = binary if isinstance(binary, Mapping) else {}

    derived_timeline = None
    pred_episodes = payload.get("pred_episodes")
    gt_episodes = payload.get("gt_episodes")
    if isinstance(pred_episodes, list) and isinstance(gt_episodes, list):
        try:
            derived_timeline = compute_timeline_metrics(
                pred_episodes=pred_episodes,
                gt_episodes=gt_episodes,
            )
        except Exception:
            derived_timeline = None

    duration_mae = _to_optional_float(payload.get("duration_mae_minutes"))
    if duration_mae is None:
        duration_mae = _to_optional_float(payload.get("segment_duration_mae_minutes"))
    if duration_mae is None:
        duration_mae = _to_optional_float(binary_map.get("segment_duration_mae_minutes"))
    if duration_mae is None and derived_timeline is not None:
        duration_mae = _to_optional_float(derived_timeline.segment_duration_mae_minutes)
    if duration_mae is not None:
        payload["duration_mae_minutes"] = duration_mae

    fragmentation = _to_optional_float(payload.get("fragmentation_rate"))
    if fragmentation is None:
        fragmentation = _to_optional_float(binary_map.get("fragmentation_rate"))
    if fragmentation is None and derived_timeline is not None:
        fragmentation = _to_optional_float(derived_timeline.fragmentation_rate)
    if fragmentation is not None:
        payload["fragmentation_rate"] = fragmentation

    num_pred = _to_optional_float(payload.get("num_pred_episodes"))
    if num_pred is None:
        num_pred = _to_optional_float(binary_map.get("num_pred_episodes"))
    if num_pred is None and derived_timeline is not None:
        num_pred = float(derived_timeline.num_pred_episodes)

    num_gt = _to_optional_float(payload.get("num_gt_episodes"))
    if num_gt is None:
        num_gt = _to_optional_float(binary_map.get("num_gt_episodes"))
    if num_gt is None and derived_timeline is not None:
        num_gt = float(derived_timeline.num_gt_episodes)

    matched = _to_optional_float(payload.get("matched_episodes"))
    if matched is None:
        matched = _to_optional_float(binary_map.get("matched_episodes"))
    if matched is None and derived_timeline is not None:
        matched = float(derived_timeline.matched_episodes)

    if num_pred is not None:
        payload["num_pred_episodes"] = int(num_pred)
    if num_gt is not None:
        payload["num_gt_episodes"] = int(num_gt)
    if matched is not None:
        payload["matched_episodes"] = int(matched)

    boundary_precision = _to_optional_float(payload.get("boundary_precision"))
    if boundary_precision is None:
        boundary_precision = _to_optional_float(payload.get("episode_precision"))
    if boundary_precision is None:
        boundary_precision = _to_optional_float(binary_map.get("boundary_precision"))
    if boundary_precision is None and matched is not None and num_pred is not None and num_pred > 0:
        boundary_precision = float(matched / num_pred)

    boundary_recall = _to_optional_float(payload.get("boundary_recall"))
    if boundary_recall is None:
        boundary_recall = _to_optional_float(payload.get("episode_recall"))
    if boundary_recall is None:
        boundary_recall = _to_optional_float(binary_map.get("boundary_recall"))
    if boundary_recall is None and matched is not None and num_gt is not None and num_gt > 0:
        boundary_recall = float(matched / num_gt)

    boundary_f1 = _to_optional_float(payload.get("boundary_f1"))
    if boundary_f1 is None:
        boundary_f1 = _to_optional_float(payload.get("episode_f1"))
    if boundary_f1 is None:
        boundary_f1 = _to_optional_float(binary_map.get("boundary_f1"))
    if boundary_f1 is None and boundary_precision is not None and boundary_recall is not None:
        if (boundary_precision + boundary_recall) > 0.0:
            boundary_f1 = float(
                2.0 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
            )
        else:
            boundary_f1 = 0.0

    if boundary_precision is not None:
        payload["boundary_precision"] = boundary_precision
    if boundary_recall is not None:
        payload["boundary_recall"] = boundary_recall
    if boundary_f1 is not None:
        payload["boundary_f1"] = boundary_f1

    episode_count_ratio = _to_optional_float(payload.get("episode_count_ratio"))
    if episode_count_ratio is None and num_pred is not None and num_gt is not None and num_gt > 0.0:
        episode_count_ratio = float(num_pred / num_gt)
    if episode_count_ratio is not None:
        payload["episode_count_ratio"] = episode_count_ratio

    return payload


def _classification_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "support": 0, "labels": []}

    labels = sorted(set(y_true) | set(y_pred))
    per_label_f1: list[float] = []
    per_label: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        if precision + recall <= 0.0:
            f1 = 0.0
        else:
            f1 = float(2.0 * precision * recall / (precision + recall))
        per_label_f1.append(f1)
        per_label[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(sum(1 for item in y_true if item == label)),
        }

    accuracy = _safe_div(sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp), len(y_true))
    macro_f1 = float(sum(per_label_f1) / max(len(per_label_f1), 1))
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "support": int(len(y_true)),
        "labels": per_label,
    }


def _unknown_metrics(
    *,
    y_true: Optional[Sequence[str]],
    y_pred: Sequence[str],
    uncertainty_states: Optional[Sequence[Any]],
    policy: UnknownPolicy,
) -> Dict[str, Any]:
    total = len(y_pred)
    abstain_label = str(policy.abstain_label).strip().lower()
    abstain_count = sum(1 for item in y_pred if item == abstain_label)
    abstain_rate = _safe_div(abstain_count, total)
    in_band = bool(policy.abstain_rate_min <= abstain_rate <= policy.abstain_rate_max)

    state_tokens = [None if s is None else str(s).strip().lower() for s in (uncertainty_states or [])]
    if state_tokens and len(state_tokens) != total:
        raise ValueError("uncertainty_states length mismatch")
    uncertainty_counts: Dict[str, int] = {}
    for token in state_tokens:
        if token is None:
            continue
        uncertainty_counts[token] = uncertainty_counts.get(token, 0) + 1

    unknown_recall = None
    unknown_recall_pass = None
    if y_true is not None:
        truth = _to_label_array(y_true)
        unknown_tokens = {
            str(policy.unknown_label).strip().lower(),
            str(policy.unknown_state).strip().lower(),
            str(policy.outside_sensed_space_state).strip().lower(),
        }
        positive_idx = [i for i, token in enumerate(truth) if token in unknown_tokens]
        if positive_idx:
            hits = 0
            for idx in positive_idx:
                is_hit = (
                    y_pred[idx] == abstain_label
                    or (idx < len(state_tokens) and state_tokens[idx] in unknown_tokens)
                )
                if is_hit:
                    hits += 1
            unknown_recall = _safe_div(hits, len(positive_idx))
            unknown_recall_pass = bool(unknown_recall >= policy.unknown_recall_min)

    return {
        "abstain_rate": float(abstain_rate),
        "abstain_count": int(abstain_count),
        "abstain_rate_min": float(policy.abstain_rate_min),
        "abstain_rate_max": float(policy.abstain_rate_max),
        "abstain_rate_in_band": bool(in_band),
        "unknown_recall": None if unknown_recall is None else float(unknown_recall),
        "unknown_recall_min": float(policy.unknown_recall_min),
        "unknown_recall_pass": unknown_recall_pass,
        "uncertainty_counts": uncertainty_counts,
    }


@dataclass(frozen=True)
class LeakageReport:
    """Leakage diagnostics used by gating and CI checks."""

    resident_overlap: bool
    time_overlap: bool
    window_overlap: bool

    @property
    def has_any_leakage(self) -> bool:
        return self.resident_overlap or self.time_overlap or self.window_overlap


def evaluate_leakage(
    train_resident_ids: Iterable[str],
    validation_resident_ids: Iterable[str],
    train_windows: Sequence[Window],
    validation_windows: Sequence[Window],
    gap_seconds: float = 0.0,
) -> LeakageReport:
    """Compute leakage diagnostics for a train/validation split."""
    resident_overlap = has_resident_leakage(train_resident_ids, validation_resident_ids)
    time_overlap = has_time_leakage(train_windows, validation_windows, gap_seconds=0.0)
    window_overlap = has_window_overlap(
        train_windows,
        validation_windows,
        gap_seconds=gap_seconds,
    )
    return LeakageReport(
        resident_overlap=resident_overlap,
        time_overlap=time_overlap,
        window_overlap=window_overlap,
    )


def assert_no_leakage(report: LeakageReport) -> None:
    """Raise when leakage is detected so CI can fail closed."""
    if report.resident_overlap:
        raise ValueError("Resident leakage detected")
    if report.time_overlap:
        raise ValueError("Temporal leakage detected")
    if report.window_overlap:
        raise ValueError("Window overlap leakage detected")


def build_room_evaluation_report(
    *,
    room: str,
    y_true: Optional[Sequence[Any]] = None,
    y_pred: Optional[Sequence[Any]] = None,
    probabilities: Optional[np.ndarray] = None,
    true_indices: Optional[Sequence[int]] = None,
    uncertainty_states: Optional[Sequence[Any]] = None,
    unknown_policy: Optional[UnknownPolicy] = None,
    timeline_metrics: Optional[Mapping[str, Any]] = None,
    leakage: Optional[Mapping[str, Any]] = None,
    data_viable: bool = True,
    reason_code: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    normalized_timeline_metrics = _normalize_timeline_metrics(timeline_metrics)

    y_true_norm = _to_label_array(y_true) if y_true is not None else None
    y_pred_norm = _to_label_array(y_pred) if y_pred is not None else None

    if y_true_norm is not None and y_pred_norm is not None:
        metrics.update(_classification_metrics(y_true_norm, y_pred_norm))

    if probabilities is not None and true_indices is not None:
        metrics["calibration"] = evaluate_calibration(probabilities, true_indices).to_dict()

    if y_pred_norm is not None and unknown_policy is not None:
        metrics["unknown"] = _unknown_metrics(
            y_true=y_true_norm,
            y_pred=y_pred_norm,
            uncertainty_states=uncertainty_states,
            policy=unknown_policy,
        )

    metrics_passed = True
    if "unknown" in metrics:
        unknown_metrics = metrics["unknown"]
        metrics_passed = bool(unknown_metrics.get("abstain_rate_in_band", True))
        recall_pass = unknown_metrics.get("unknown_recall_pass")
        if recall_pass is not None:
            metrics_passed = bool(metrics_passed and recall_pass)

    report = {
        "room": str(room).strip().lower(),
        "passed": bool(data_viable and metrics_passed and not reason_code),
        "metrics_passed": bool(metrics_passed),
        "data_viable": bool(data_viable),
        "metrics": metrics,
        "timeline_metrics": normalized_timeline_metrics,
        "leakage": {
            "resident_overlap": bool((leakage or {}).get("resident_overlap", False)),
            "time_overlap": bool((leakage or {}).get("time_overlap", False)),
            "window_overlap": bool((leakage or {}).get("window_overlap", False)),
        },
        "details": dict(details or {}),
    }
    if reason_code is not None:
        report["reason_code"] = str(reason_code)
    return report


def create_signed_evaluation_report(
    *,
    run_id: str,
    elder_id: str,
    room_reports: Sequence[Mapping[str, Any]],
    signing_key: str,
    output_path: str | Path | None = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "version": "beta6_eval_report_v1",
        "generated_at": _utc_now(),
        "run_id": str(run_id),
        "elder_id": str(elder_id),
        "room_reports": [dict(item) for item in room_reports],
        "metadata": dict(metadata or {}),
    }
    signature = _sign_payload(payload, signing_key=signing_key)
    artifact = {**payload, "signature": signature}
    if output_path is not None:
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def verify_evaluation_report_signature(
    artifact: Mapping[str, Any],
    *,
    signing_key: str,
) -> bool:
    if "signature" not in artifact:
        return False
    signature = str(artifact.get("signature", ""))
    payload = {k: v for k, v in artifact.items() if k != "signature"}
    return signature == _sign_payload(payload, signing_key=signing_key)


__all__ = [
    "LeakageReport",
    "assert_no_leakage",
    "build_room_evaluation_report",
    "create_signed_evaluation_report",
    "evaluate_leakage",
    "verify_evaluation_report_signature",
]
