"""Utilities for converting window labels into event episodes and daily ADL targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


DEFAULT_SAMPLE_INTERVAL_SECONDS = 10.0


@dataclass(frozen=True)
class EventEpisode:
    label: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_seconds: float
    sample_count: int


def normalize_activity_label(value: object) -> str:
    """Normalize labels to stable lowercase tokens."""
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    return text if text else "unknown"


def labels_to_episodes(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    label_col: str = "activity",
    max_gap_seconds: float = 300.0,
    min_duration_seconds: float = 0.0,
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    drop_labels: Optional[Sequence[str]] = None,
) -> List[EventEpisode]:
    """Collapse consecutive labels into episodes with gap-aware splitting."""
    if timestamp_col not in df.columns or label_col not in df.columns:
        return []

    work = df[[timestamp_col, label_col]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[label_col] = work[label_col].map(normalize_activity_label)
    work = work.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)

    if work.empty:
        return []

    drop_set = {normalize_activity_label(x) for x in (drop_labels or [])}

    episodes: List[EventEpisode] = []
    cur_label: Optional[str] = None
    start_ts: Optional[pd.Timestamp] = None
    prev_ts: Optional[pd.Timestamp] = None
    cur_count = 0

    def flush_episode() -> None:
        nonlocal cur_label, start_ts, prev_ts, cur_count
        if cur_label is None or start_ts is None or prev_ts is None:
            return
        duration = max((prev_ts - start_ts).total_seconds() + sample_interval_seconds, 0.0)
        if cur_label not in drop_set and duration >= min_duration_seconds:
            episodes.append(
                EventEpisode(
                    label=cur_label,
                    start_time=start_ts,
                    end_time=prev_ts,
                    duration_seconds=duration,
                    sample_count=cur_count,
                )
            )
        cur_label = None
        start_ts = None
        prev_ts = None
        cur_count = 0

    for _, row in work.iterrows():
        ts: pd.Timestamp = row[timestamp_col]
        label: str = row[label_col]

        if cur_label is None:
            cur_label = label
            start_ts = ts
            prev_ts = ts
            cur_count = 1
            continue

        gap = (ts - prev_ts).total_seconds() if prev_ts is not None else 0.0
        is_new_episode = label != cur_label or gap > max_gap_seconds

        if is_new_episode:
            flush_episode()
            cur_label = label
            start_ts = ts
            prev_ts = ts
            cur_count = 1
        else:
            prev_ts = ts
            cur_count += 1

    flush_episode()
    return episodes


def episodes_to_frame(episodes: Iterable[EventEpisode]) -> pd.DataFrame:
    rows = [
        {
            "label": ep.label,
            "start_time": ep.start_time,
            "end_time": ep.end_time,
            "duration_seconds": ep.duration_seconds,
            "sample_count": ep.sample_count,
        }
        for ep in episodes
    ]
    return pd.DataFrame(rows)


def build_daily_event_targets(
    episodes: Sequence[EventEpisode],
    *,
    label_groups: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, float]:
    """Aggregate event episodes into care-oriented daily targets."""
    groups: Dict[str, Sequence[str]] = label_groups or {
        "sleep": ("sleep", "nap"),
        "shower": ("shower",),
        "kitchen_use": ("kitchen_normal_use",),
        "livingroom_active": ("livingroom_normal_use",),
        "bathroom_use": ("bathroom_normal_use", "shower"),
        "out": ("out",),
    }

    label_minutes: Dict[str, float] = {}
    label_counts: Dict[str, int] = {}

    occupied_seconds = 0.0
    unoccupied_seconds = 0.0

    for ep in episodes:
        minutes = ep.duration_seconds / 60.0
        label_minutes[ep.label] = label_minutes.get(ep.label, 0.0) + minutes
        label_counts[ep.label] = label_counts.get(ep.label, 0) + 1

        if ep.label == "unoccupied":
            unoccupied_seconds += ep.duration_seconds
        else:
            occupied_seconds += ep.duration_seconds

    out: Dict[str, float] = {
        "occupied_minutes": occupied_seconds / 60.0,
        "unoccupied_minutes": unoccupied_seconds / 60.0,
        "event_count_total": float(sum(label_counts.values())),
    }

    for key, labels in groups.items():
        total_minutes = sum(label_minutes.get(normalize_activity_label(lbl), 0.0) for lbl in labels)
        event_count = sum(label_counts.get(normalize_activity_label(lbl), 0) for lbl in labels)
        out[f"{key}_minutes"] = total_minutes
        out[f"{key}_events"] = float(event_count)
        out[f"{key}_day"] = 1.0 if total_minutes > 0 else 0.0

    # Keep these as payload for richer diagnostics.
    out["label_minutes"] = label_minutes  # type: ignore[assignment]
    out["label_event_counts"] = label_counts  # type: ignore[assignment]
    return out
