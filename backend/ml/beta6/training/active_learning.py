"""Phase 3.2 active-learning triage queue generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ml.yaml_compat import load_yaml_file


ACTIVE_LEARNING_VERSION = "v1"


@dataclass(frozen=True)
class ActiveLearningPolicy:
    queue_size: int = 120
    uncertainty_fraction: float = 0.5
    disagreement_fraction: float = 0.3
    diversity_fraction: float = 0.2
    uncertainty_percentile: float = 25.0
    max_share_per_room: float = 0.35
    max_share_per_class: float = 0.35
    random_seed: int = 42
    correction_signal_boost: float = 0.55
    boundary_signal_boost: float = 0.20
    hard_negative_signal_boost: float = 0.20
    residual_review_signal_boost: float = 0.10


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def load_active_learning_policy(path: str | Path | None) -> ActiveLearningPolicy:
    if path is None:
        return ActiveLearningPolicy()
    raw = load_yaml_file(Path(path).resolve()) or {}
    section = _as_mapping(raw).get("active_learning")
    if not isinstance(section, Mapping):
        section = raw if isinstance(raw, Mapping) else {}
    return ActiveLearningPolicy(
        queue_size=max(int(section.get("queue_size", 120)), 1),
        uncertainty_fraction=min(max(float(section.get("uncertainty_fraction", 0.5)), 0.0), 1.0),
        disagreement_fraction=min(max(float(section.get("disagreement_fraction", 0.3)), 0.0), 1.0),
        diversity_fraction=min(max(float(section.get("diversity_fraction", 0.2)), 0.0), 1.0),
        uncertainty_percentile=min(max(float(section.get("uncertainty_percentile", 25)), 0.0), 100.0),
        max_share_per_room=min(max(float(section.get("max_share_per_room", 0.35)), 0.1), 1.0),
        max_share_per_class=min(max(float(section.get("max_share_per_class", 0.35)), 0.1), 1.0),
        random_seed=int(section.get("random_seed", 42)),
        correction_signal_boost=max(float(section.get("correction_signal_boost", 0.55)), 0.0),
        boundary_signal_boost=max(float(section.get("boundary_signal_boost", 0.20)), 0.0),
        hard_negative_signal_boost=max(float(section.get("hard_negative_signal_boost", 0.20)), 0.0),
        residual_review_signal_boost=max(float(section.get("residual_review_signal_boost", 0.10)), 0.0),
    )


def _empty_candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "candidate_id",
            "room",
            "activity",
            "confidence",
            "predicted_label",
            "baseline_label",
            "uncertainty_score",
            "disagreement_flag",
            "corrected_event",
            "boundary_start_target",
            "boundary_end_target",
            "hard_negative_flag",
            "hard_negative_label",
            "residual_review_flag",
            "residual_review_rows",
            "triage_priority_score",
        ]
    )


def _series_or_default(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    if isinstance(default, pd.Series):
        return default.reindex(frame.index)
    return pd.Series([default] * len(frame), index=frame.index)


def _coerce_bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    raw = _series_or_default(frame, column, default)
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(bool(default)).astype(bool)
    normalized = raw.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "on"})


def _coerce_int_series(frame: pd.DataFrame, column: str, default: int = 0) -> pd.Series:
    return pd.to_numeric(_series_or_default(frame, column, default), errors="coerce").fillna(int(default)).astype(int)


def _normalize_candidates(
    candidates: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    if isinstance(candidates, pd.DataFrame):
        frame = candidates.copy()
    else:
        frame = pd.DataFrame([dict(row) for row in candidates])
    if frame.empty:
        return _empty_candidate_frame(), None

    required_columns = ("room", "activity", "confidence")
    missing_required = [column for column in required_columns if column not in frame.columns]
    if missing_required:
        return _empty_candidate_frame(), {
            "reason": "invalid_candidate_schema",
            "details": {
                "missing_columns": missing_required,
                "required_columns": list(required_columns),
            },
            "input_rows": int(len(frame)),
        }

    frame["candidate_id"] = _series_or_default(
        frame,
        "candidate_id",
        pd.Series(range(len(frame)), index=frame.index),
    ).astype(str)
    frame["room"] = _series_or_default(frame, "room", "").astype(str).str.strip().str.lower()
    frame["activity"] = _series_or_default(frame, "activity", "").astype(str).str.strip().str.lower()
    predicted_series = _series_or_default(frame, "predicted_label", frame["activity"])
    baseline_series = _series_or_default(frame, "baseline_label", "")
    frame["predicted_label"] = predicted_series.astype(str).str.strip().str.lower()
    frame["baseline_label"] = baseline_series.astype(str).str.strip().str.lower()
    frame["confidence"] = pd.to_numeric(_series_or_default(frame, "confidence", 0.0), errors="coerce").fillna(0.0)
    frame["confidence"] = frame["confidence"].clip(0.0, 1.0)
    frame["uncertainty_score"] = 1.0 - frame["confidence"]
    frame["disagreement_flag"] = (
        (frame["baseline_label"] != "")
        & (frame["predicted_label"] != "")
        & (frame["baseline_label"] != frame["predicted_label"])
    )
    frame["corrected_event"] = _coerce_bool_series(frame, "corrected_event", False)
    frame["boundary_start_target"] = _coerce_int_series(frame, "boundary_start_target", 0).clip(lower=0)
    frame["boundary_end_target"] = _coerce_int_series(frame, "boundary_end_target", 0).clip(lower=0)
    frame["hard_negative_flag"] = _coerce_bool_series(frame, "hard_negative_flag", False)
    frame["hard_negative_label"] = _series_or_default(frame, "hard_negative_label", "").astype(str).str.strip().str.lower()
    frame["residual_review_flag"] = _coerce_bool_series(frame, "residual_review_flag", False)
    frame["residual_review_rows"] = _coerce_int_series(frame, "residual_review_rows", 0).clip(lower=0)
    frame = frame[frame["room"] != ""]
    frame = frame[frame["activity"] != ""]
    frame = frame.reset_index(drop=True)
    return frame, None


def _attach_priority_scores(frame: pd.DataFrame, policy: ActiveLearningPolicy) -> pd.DataFrame:
    if frame.empty:
        out = frame.copy()
        out["triage_priority_score"] = pd.Series(dtype=float)
        return out
    out = frame.copy()
    boundary_signal = out["boundary_start_target"].gt(0).astype(float) + out["boundary_end_target"].gt(0).astype(float)
    out["triage_priority_score"] = (
        out["uncertainty_score"].astype(float)
        + out["disagreement_flag"].astype(float) * 0.05
        + out["corrected_event"].astype(float) * float(policy.correction_signal_boost)
        + boundary_signal.astype(float) * float(policy.boundary_signal_boost)
        + out["hard_negative_flag"].astype(float) * float(policy.hard_negative_signal_boost)
        + out["residual_review_flag"].astype(float) * float(policy.residual_review_signal_boost)
    )
    return out


def _round_robin_take(groups: Sequence[pd.DataFrame], limit: int) -> pd.DataFrame:
    if limit <= 0:
        return pd.DataFrame()
    active_groups = [group.reset_index(drop=True) for group in groups if not group.empty]
    if not active_groups:
        return pd.DataFrame()

    selected: List[pd.Series] = []
    idx = 0
    while len(selected) < limit and active_groups:
        gidx = idx % len(active_groups)
        group = active_groups[gidx]
        if group.empty:
            active_groups.pop(gidx)
            continue
        selected.append(group.iloc[0])
        active_groups[gidx] = group.iloc[1:].reset_index(drop=True)
        idx += 1
    if not selected:
        return pd.DataFrame()
    return pd.DataFrame(selected).reset_index(drop=True)


def _select_uncertain(frame: pd.DataFrame, limit: int, percentile: float) -> pd.DataFrame:
    if frame.empty or limit <= 0:
        return frame.iloc[0:0]
    grouped_subsets = []
    for _, group in frame.groupby(["room", "activity"], sort=True):
        cutoff = float(np.percentile(group["confidence"].to_numpy(), percentile))
        subset = group[group["confidence"] <= cutoff].copy()
        if subset.empty:
            subset = group.sort_values(["confidence", "candidate_id"], ascending=[True, True]).head(1).copy()
        subset = subset.sort_values(
            ["triage_priority_score", "uncertainty_score", "candidate_id"],
            ascending=[False, False, True],
        )
        grouped_subsets.append(subset.reset_index(drop=True))

    selection = _round_robin_take(grouped_subsets, limit)
    if selection.empty:
        return frame.iloc[0:0]
    return selection


def _select_disagreement(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if frame.empty or limit <= 0:
        return frame.iloc[0:0]
    subset = frame[frame["disagreement_flag"]].copy()
    subset = subset.sort_values(
        ["triage_priority_score", "uncertainty_score", "candidate_id"],
        ascending=[False, False, True],
    )
    return subset.head(limit).copy()


def _select_diversity(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if frame.empty or limit <= 0:
        return frame.iloc[0:0]
    rng = np.random.default_rng(seed)
    groups = []
    for _, group in frame.groupby(["room", "activity"], sort=True):
        shuffled = group.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        groups.append(shuffled)
    selected_rows = []
    idx = 0
    while len(selected_rows) < limit and groups:
        group = groups[idx % len(groups)]
        if group.empty:
            groups.pop(idx % len(groups))
            continue
        selected_rows.append(group.iloc[0])
        groups[idx % len(groups)] = group.iloc[1:]
        idx += 1
    if not selected_rows:
        return frame.iloc[0:0]
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _apply_domination_caps(
    frame: pd.DataFrame,
    *,
    queue_size: int,
    max_share_per_room: float,
    max_share_per_class: float,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    max_room = max(1, int(np.floor(queue_size * max_share_per_room)))
    max_cls = max(1, int(np.floor(queue_size * max_share_per_class)))

    kept = []
    room_counts: Dict[str, int] = {}
    cls_counts: Dict[str, int] = {}
    for _, row in frame.iterrows():
        room = str(row["room"])
        cls = str(row["activity"])
        if room_counts.get(room, 0) >= max_room:
            continue
        if cls_counts.get(cls, 0) >= max_cls:
            continue
        kept.append(row)
        room_counts[room] = room_counts.get(room, 0) + 1
        cls_counts[cls] = cls_counts.get(cls, 0) + 1
        if len(kept) >= queue_size:
            break
    return pd.DataFrame(kept).reset_index(drop=True) if kept else frame.iloc[0:0]


def _cap_limits(queue_size: int, max_share_per_room: float, max_share_per_class: float) -> tuple[int, int]:
    max_room = max(1, int(np.floor(queue_size * max_share_per_room)))
    max_cls = max(1, int(np.floor(queue_size * max_share_per_class)))
    return max_room, max_cls


def _refill_after_caps(
    selected: pd.DataFrame,
    *,
    pool: pd.DataFrame,
    queue_size: int,
    max_share_per_room: float,
    max_share_per_class: float,
) -> pd.DataFrame:
    if selected.empty:
        selected_rows: List[pd.Series] = []
    else:
        selected_rows = [row for _, row in selected.iterrows()]

    max_room, max_cls = _cap_limits(queue_size, max_share_per_room, max_share_per_class)
    selected_ids = {str(row["candidate_id"]) for row in selected_rows}
    room_counts: Dict[str, int] = {}
    class_counts: Dict[str, int] = {}
    for row in selected_rows:
        room = str(row["room"])
        cls = str(row["activity"])
        room_counts[room] = room_counts.get(room, 0) + 1
        class_counts[cls] = class_counts.get(cls, 0) + 1

    for _, row in pool.iterrows():
        if len(selected_rows) >= queue_size:
            break
        cid = str(row["candidate_id"])
        if cid in selected_ids:
            continue
        room = str(row["room"])
        cls = str(row["activity"])
        if room_counts.get(room, 0) >= max_room:
            continue
        if class_counts.get(cls, 0) >= max_cls:
            continue
        selected_rows.append(row)
        selected_ids.add(cid)
        room_counts[room] = room_counts.get(room, 0) + 1
        class_counts[cls] = class_counts.get(cls, 0) + 1

    if not selected_rows:
        return selected.iloc[0:0]
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def build_active_learning_queue(
    candidates: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    policy: ActiveLearningPolicy,
) -> Dict[str, Any]:
    frame, normalize_error = _normalize_candidates(candidates)
    if normalize_error is not None:
        return {
            "status": "fail",
            "reason": str(normalize_error.get("reason", "invalid_candidate_schema")),
            "queue": [],
            "stats": {
                "input_rows": int(normalize_error.get("input_rows", 0)),
                "queue_rows": 0,
            },
            "details": dict(normalize_error.get("details", {})),
        }
    if frame.empty:
        return {
            "status": "fail",
            "reason": "no_candidates",
            "queue": [],
            "stats": {"input_rows": 0, "queue_rows": 0},
        }
    frame = _attach_priority_scores(frame, policy)

    q = policy.queue_size
    n_uncertain = int(round(q * policy.uncertainty_fraction))
    n_disagreement = int(round(q * policy.disagreement_fraction))
    n_diversity = int(round(q * policy.diversity_fraction))
    if n_uncertain + n_disagreement + n_diversity > q:
        overflow = n_uncertain + n_disagreement + n_diversity - q
        n_uncertain = max(0, n_uncertain - overflow)

    uncertain = _select_uncertain(frame, n_uncertain, policy.uncertainty_percentile)
    uncertain = uncertain.assign(selection_reason="uncertainty")
    disagreement = _select_disagreement(
        frame[~frame["candidate_id"].isin(uncertain["candidate_id"])],
        n_disagreement,
    ).assign(selection_reason="disagreement")
    diversity = _select_diversity(
        frame[~frame["candidate_id"].isin(pd.concat([uncertain, disagreement])["candidate_id"])],
        n_diversity,
        policy.random_seed,
    ).assign(selection_reason="diversity")

    combined = pd.concat([uncertain, disagreement, diversity], ignore_index=True)
    if len(combined) < q:
        remaining = frame[~frame["candidate_id"].isin(combined["candidate_id"])].copy()
        remaining = remaining.sort_values(
            ["triage_priority_score", "uncertainty_score", "candidate_id"],
            ascending=[False, False, True],
        )
        fill = remaining.head(q - len(combined)).assign(selection_reason="uncertainty_fill")
        combined = pd.concat([combined, fill], ignore_index=True)

    combined = combined.sort_values(
        ["triage_priority_score", "uncertainty_score", "candidate_id"],
        ascending=[False, False, True],
    )
    capped = _apply_domination_caps(
        combined,
        queue_size=q,
        max_share_per_room=policy.max_share_per_room,
        max_share_per_class=policy.max_share_per_class,
    )
    if len(capped) < q:
        refill_pool = frame[
            ~frame["candidate_id"].isin(capped["candidate_id"])
        ].copy()
        refill_pool = refill_pool.sort_values(
            ["triage_priority_score", "uncertainty_score", "candidate_id"],
            ascending=[False, False, True],
        )
        refill_pool = refill_pool.assign(selection_reason="cap_refill")
        capped = _refill_after_caps(
            capped,
            pool=refill_pool,
            queue_size=q,
            max_share_per_room=policy.max_share_per_room,
            max_share_per_class=policy.max_share_per_class,
        )

    queue_rows = capped.to_dict(orient="records")
    room_counts = capped["room"].value_counts().to_dict() if not capped.empty else {}
    class_counts = capped["activity"].value_counts().to_dict() if not capped.empty else {}
    training_signal_counts = {
        "corrected_event_rows": int(capped["corrected_event"].sum()) if not capped.empty else 0,
        "boundary_target_rows": int(
            (capped["boundary_start_target"].gt(0) | capped["boundary_end_target"].gt(0)).sum()
        )
        if not capped.empty
        else 0,
        "hard_negative_rows": int(capped["hard_negative_flag"].sum()) if not capped.empty else 0,
        "residual_review_rows": int(capped["residual_review_rows"].sum()) if not capped.empty else 0,
    }
    return {
        "status": "pass" if queue_rows else "fail",
        "version": ACTIVE_LEARNING_VERSION,
        "queue": queue_rows,
        "stats": {
            "input_rows": int(len(frame)),
            "queue_rows": int(len(queue_rows)),
            "uncertainty_rows": int((capped["selection_reason"] == "uncertainty").sum()) if not capped.empty else 0,
            "disagreement_rows": int((capped["selection_reason"] == "disagreement").sum()) if not capped.empty else 0,
            "diversity_rows": int((capped["selection_reason"] == "diversity").sum()) if not capped.empty else 0,
            "room_counts": room_counts,
            "class_counts": class_counts,
            "training_signal_counts": training_signal_counts,
            "max_triage_priority_score": float(capped["triage_priority_score"].max()) if not capped.empty else 0.0,
        },
    }


__all__ = [
    "ACTIVE_LEARNING_VERSION",
    "ActiveLearningPolicy",
    "build_active_learning_queue",
    "load_active_learning_policy",
]
