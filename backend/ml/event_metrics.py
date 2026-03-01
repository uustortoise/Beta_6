"""Event-level metrics for ADL care-oriented evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntervalEvent:
    label: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp


def interval_overlap_seconds(a: IntervalEvent, b: IntervalEvent) -> float:
    start = max(a.start_time, b.start_time)
    end = min(a.end_time, b.end_time)
    return max((end - start).total_seconds(), 0.0)


def interval_iou(a: IntervalEvent, b: IntervalEvent) -> float:
    overlap = interval_overlap_seconds(a, b)
    if overlap <= 0:
        return 0.0
    a_len = max((a.end_time - a.start_time).total_seconds(), 0.0)
    b_len = max((b.end_time - b.start_time).total_seconds(), 0.0)
    union = a_len + b_len - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def match_events(
    gt_events: Sequence[IntervalEvent],
    pred_events: Sequence[IntervalEvent],
    *,
    label: str,
    min_iou: float = 0.10,
) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching by IoU for a single label."""
    gt_idx = [i for i, ev in enumerate(gt_events) if ev.label == label]
    pd_idx = [i for i, ev in enumerate(pred_events) if ev.label == label]

    candidates: List[Tuple[float, int, int]] = []
    for gi in gt_idx:
        for pi in pd_idx:
            iou = interval_iou(gt_events[gi], pred_events[pi])
            if iou >= min_iou:
                candidates.append((iou, gi, pi))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_gt = set()
    used_pd = set()
    matches: List[Tuple[int, int, float]] = []

    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pd:
            continue
        used_gt.add(gi)
        used_pd.add(pi)
        matches.append((gi, pi, iou))

    return matches


def event_precision_recall_f1(
    gt_events: Sequence[IntervalEvent],
    pred_events: Sequence[IntervalEvent],
    *,
    label: str,
    min_iou: float = 0.10,
) -> Dict[str, float]:
    gt_count = sum(1 for e in gt_events if e.label == label)
    pred_count = sum(1 for e in pred_events if e.label == label)

    matches = match_events(gt_events, pred_events, label=label, min_iou=min_iou)
    tp = len(matches)
    fp = max(pred_count - tp, 0)
    fn = max(gt_count - tp, 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support_gt": float(gt_count),
        "support_pred": float(pred_count),
        "matches": float(tp),
    }


def daily_duration_mae(
    gt_days: Iterable[Dict[str, float]],
    pred_days: Iterable[Dict[str, float]],
    *,
    key: str,
) -> float:
    gt = list(gt_days)
    pred = list(pred_days)
    n = min(len(gt), len(pred))
    if n == 0:
        return 0.0
    diffs = [abs(float(gt[i].get(key, 0.0)) - float(pred[i].get(key, 0.0))) for i in range(n)]
    return float(np.mean(diffs))


def binary_day_precision_recall(
    gt_days: Iterable[Dict[str, float]],
    pred_days: Iterable[Dict[str, float]],
    *,
    key: str,
) -> Dict[str, float]:
    gt = list(gt_days)
    pred = list(pred_days)
    n = min(len(gt), len(pred))
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = fp = fn = 0
    for i in range(n):
        g = 1 if float(gt[i].get(key, 0.0)) > 0 else 0
        p = 1 if float(pred[i].get(key, 0.0)) > 0 else 0
        if g == 1 and p == 1:
            tp += 1
        elif g == 0 and p == 1:
            fp += 1
        elif g == 1 and p == 0:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_care_kpi_summary(
    gt_days: Sequence[Dict[str, float]],
    pred_days: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    """Compute core event-level care KPIs from daily aggregates."""
    shower = binary_day_precision_recall(gt_days, pred_days, key="shower_day")

    return {
        "sleep_duration_mae_minutes": daily_duration_mae(gt_days, pred_days, key="sleep_minutes"),
        "livingroom_active_mae_minutes": daily_duration_mae(gt_days, pred_days, key="livingroom_active_minutes"),
        "kitchen_use_mae_minutes": daily_duration_mae(gt_days, pred_days, key="kitchen_use_minutes"),
        "shower_day_precision": shower["precision"],
        "shower_day_recall": shower["recall"],
        "shower_day_f1": shower["f1"],
    }


def compute_room_care_kpis(
    room_name: str,
    gt_days: Sequence[Dict[str, float]],
    pred_days: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    """Room-scoped KPI view used for practical promotion decisions."""
    room = room_name.strip().lower()
    out: Dict[str, float] = {}

    if room == "bedroom":
        out["sleep_duration_mae_minutes"] = daily_duration_mae(gt_days, pred_days, key="sleep_minutes")
        return out

    if room == "livingroom":
        out["livingroom_active_mae_minutes"] = daily_duration_mae(
            gt_days, pred_days, key="livingroom_active_minutes"
        )
        return out

    if room == "kitchen":
        out["kitchen_use_mae_minutes"] = daily_duration_mae(gt_days, pred_days, key="kitchen_use_minutes")
        return out

    if room == "bathroom":
        shower = binary_day_precision_recall(gt_days, pred_days, key="shower_day")
        out["shower_day_precision"] = shower["precision"]
        out["shower_day_recall"] = shower["recall"]
        out["shower_day_f1"] = shower["f1"]
        out["bathroom_use_mae_minutes"] = daily_duration_mae(gt_days, pred_days, key="bathroom_use_minutes")
        return out

    if room == "entrance":
        out["out_minutes_mae"] = daily_duration_mae(gt_days, pred_days, key="out_minutes")
        return out

    return compute_care_kpi_summary(gt_days, pred_days)
