#!/usr/bin/env python3
"""Run event-first rolling backtest with calibration/tuning and split diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.isotonic import IsotonicRegression

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.event_labels import build_daily_event_targets, labels_to_episodes
from ml.event_metrics import compute_room_care_kpis
from ml.event_models import EventFirstConfig, EventFirstTwoStageModel
from ml.timeline_decoder_v2 import TimelineDecodePolicy, TimelineDecoderV2
from ml.timeline_gates import create_default_timeline_checker
from ml.timeline_metrics import TimelineMetrics, compute_timeline_metrics
from ml.timeline_targets import build_boundary_targets
from ml.segment_classifier import classify_occupied_segments
from ml.segment_features import build_segment_features
from ml.segment_projection import project_segments_to_window_labels
from ml.segment_proposal import propose_occupancy_segments


DAY_RE = re.compile(r"train_(\d{1,2})(?:[a-z]{3}\d{4})?$", re.IGNORECASE)
ROOMS = ["Bedroom", "LivingRoom", "Kitchen", "Bathroom", "Entrance"]
SENSOR_COLS = ["co2", "vibration", "humidity", "temperature", "sound", "motion", "light"]
KITCHEN_DIFF_BASE_COLS = ["motion", "sound", "light", "humidity", "co2", "temperature", "vibration"]
KITCHEN_ENGINEERED_COLS = (
    [f"{c}_d1" for c in KITCHEN_DIFF_BASE_COLS]
    + [f"{c}_d1m" for c in KITCHEN_DIFF_BASE_COLS]
    + [f"{c}_d5m" for c in KITCHEN_DIFF_BASE_COLS]
    + [
        "motion_active",
        "motion_burst_2m",
        "sound_roll_std_2m",
        "light_roll_std_2m",
        "humidity_rise_3m",
        "co2_rise_5m",
    ]
)
ROOM_OCCUPANCY_ENGINEERED_ROOMS = {"bedroom", "livingroom"}
ROOM_OCCUPANCY_ENGINEERED_BASE_COLS = ["motion", "sound", "light", "co2", "humidity", "temperature", "vibration"]
ROOM_DURATION_KEY = {
    "bedroom": "sleep_minutes",
    "livingroom": "livingroom_active_minutes",
    "kitchen": "kitchen_use_minutes",
    "bathroom": "bathroom_use_minutes",
    "entrance": "out_minutes",
}


@dataclass(frozen=True)
class Split:
    train_days: Sequence[int]
    test_day: int


def _rolling_linear_slope(series: pd.Series, *, window: int) -> pd.Series:
    """
    Causal trailing-window linear regression slope against time index [0..window-1].
    """
    n = int(max(window, 2))
    x = np.arange(n, dtype=float)
    x_centered = x - float(np.mean(x))
    denom = float(np.sum(np.square(x_centered)))
    if denom <= 0.0:
        return pd.Series(np.zeros(shape=(len(series),), dtype=float), index=series.index)

    values = pd.to_numeric(series, errors="coerce").ffill().bfill().fillna(0.0)
    slopes = values.rolling(window=n, min_periods=n).apply(
        lambda arr: float(np.dot(x_centered, np.asarray(arr, dtype=float))) / denom,
        raw=True,
    )
    return slopes.fillna(0.0)


def _time_since_last_active_windows(active_mask: np.ndarray) -> np.ndarray:
    """
    Causal counter: number of windows since the last active=True window.
    Returns 0 when active at current index.
    """
    mask = np.asarray(active_mask, dtype=bool)
    n = int(len(mask))
    out = np.zeros(shape=(n,), dtype=float)
    since = float(n + 1)
    for i in range(n):
        if bool(mask[i]):
            since = 0.0
        else:
            since = float(since + 1.0)
        out[i] = float(since)
    return out


def _add_kitchen_occupancy_features(df: pd.DataFrame) -> List[str]:
    """
    Add kitchen-focused causal features to improve occupancy discrimination.
    """
    required = list(KITCHEN_DIFF_BASE_COLS)
    for col in required:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill().bfill().fillna(0.0)

    out_cols: List[str] = []
    shift_1m = 6   # 60s at 10s interval
    shift_5m = 30  # 300s at 10s interval

    # Short-term changes capture movement bursts and sensor transitions.
    for col in KITCHEN_DIFF_BASE_COLS:
        d1 = f"{col}_d1"
        d1m = f"{col}_d1m"
        d5m = f"{col}_d5m"
        df[d1] = pd.to_numeric(df[col], errors="coerce").diff().fillna(0.0)
        df[d1m] = pd.to_numeric(df[col], errors="coerce").diff(shift_1m).fillna(0.0)
        df[d5m] = pd.to_numeric(df[col], errors="coerce").diff(shift_5m).fillna(0.0)
        out_cols.extend([d1, d1m, d5m])

    # Causal rolling descriptors.
    df["motion_active"] = (pd.to_numeric(df["motion"], errors="coerce") > 0.5).astype(float)
    df["motion_burst_2m"] = df["motion_active"].rolling(window=12, min_periods=1).sum()
    df["sound_roll_std_2m"] = pd.to_numeric(df["sound"], errors="coerce").rolling(window=12, min_periods=2).std().fillna(0.0)
    df["light_roll_std_2m"] = pd.to_numeric(df["light"], errors="coerce").rolling(window=12, min_periods=2).std().fillna(0.0)
    df["humidity_rise_3m"] = pd.to_numeric(df["humidity"], errors="coerce").diff(18).fillna(0.0)
    df["co2_rise_5m"] = pd.to_numeric(df["co2"], errors="coerce").diff(30).fillna(0.0)
    out_cols.extend(
        [
            "motion_active",
            "motion_burst_2m",
            "sound_roll_std_2m",
            "light_roll_std_2m",
            "humidity_rise_3m",
            "co2_rise_5m",
        ]
    )

    for col in out_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out_cols


def _add_room_temporal_occupancy_features(
    df: pd.DataFrame,
    *,
    room_key: Optional[str] = None,
    enable_bedroom_light_texture_features: bool = False,
    bedroom_livingroom_texture_profile: str = "mixed",
) -> List[str]:
    """
    Add generic causal temporal features for bedroom/livingroom occupancy separability.
    """
    required = list(ROOM_OCCUPANCY_ENGINEERED_BASE_COLS)
    for col in required:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill().bfill().fillna(0.0)

    out_cols: List[str] = []
    lag_1m = 6
    lag_5m = 30
    lag_15m = 90
    lag_30m = 180
    lag_60m = 360
    room_key_norm = str(room_key or "").strip().lower()
    profile = str(bedroom_livingroom_texture_profile or "mixed").strip().lower()
    if profile not in {"30m", "60m", "mixed"}:
        profile = "mixed"
    use_30m = profile in {"30m", "mixed"}
    use_60m = profile in {"60m", "mixed"}

    for col in required:
        d1 = f"occ_{col}_d1"
        d1m = f"occ_{col}_d1m"
        d5m = f"occ_{col}_d5m"
        roll_mean_2m = f"occ_{col}_roll_mean_2m"
        roll_std_2m = f"occ_{col}_roll_std_2m"
        src = pd.to_numeric(df[col], errors="coerce")
        df[d1] = src.diff().fillna(0.0)
        df[d1m] = src.diff(lag_1m).fillna(0.0)
        df[d5m] = src.diff(lag_5m).fillna(0.0)
        df[roll_mean_2m] = src.rolling(window=12, min_periods=1).mean().fillna(0.0)
        df[roll_std_2m] = src.rolling(window=12, min_periods=2).std().fillna(0.0)
        out_cols.extend([d1, d1m, d5m, roll_mean_2m, roll_std_2m])

    motion_vals = pd.to_numeric(df["motion"], errors="coerce")
    co2_vals = pd.to_numeric(df["co2"], errors="coerce")
    light_vals = pd.to_numeric(df["light"], errors="coerce")
    df["occ_motion_active"] = (motion_vals > 0.5).astype(float)
    df["occ_motion_inactive"] = (motion_vals <= 0.5).astype(float)
    df["occ_motion_active_2m"] = df["occ_motion_active"].rolling(window=12, min_periods=1).sum()
    df["occ_motion_active_5m"] = df["occ_motion_active"].rolling(window=lag_5m, min_periods=1).sum()
    df["occ_motion_activity_ratio_15m"] = df["occ_motion_active"].rolling(window=lag_15m, min_periods=1).mean()
    df["occ_time_since_motion_active_windows"] = _time_since_last_active_windows(
        df["occ_motion_active"].to_numpy(dtype=float) > 0.0
    )
    df["occ_time_since_motion_active_minutes"] = (
        df["occ_time_since_motion_active_windows"].to_numpy(dtype=float) * (10.0 / 60.0)
    )
    df["occ_light_roll_mean_10m"] = light_vals.rolling(window=60, min_periods=1).mean().fillna(0.0)
    df["occ_light_roll_std_10m"] = light_vals.rolling(window=60, min_periods=2).std().fillna(0.0)
    df["occ_co2_slope_15m"] = _rolling_linear_slope(co2_vals, window=lag_15m)
    if use_30m:
        df["occ_motion_inactivity_ratio_30m"] = df["occ_motion_inactive"].rolling(window=lag_30m, min_periods=1).mean()
        df["occ_motion_roll_std_30m"] = motion_vals.rolling(window=lag_30m, min_periods=2).std().fillna(0.0)
        df["occ_co2_slope_30m"] = _rolling_linear_slope(co2_vals, window=lag_30m)
    if use_60m:
        df["occ_motion_inactivity_ratio_60m"] = df["occ_motion_inactive"].rolling(window=lag_60m, min_periods=1).mean()
        df["occ_motion_roll_std_60m"] = motion_vals.rolling(window=lag_60m, min_periods=2).std().fillna(0.0)
        df["occ_co2_slope_60m"] = _rolling_linear_slope(co2_vals, window=lag_60m)
    df["occ_sound_light_ratio"] = (
        pd.to_numeric(df["sound"], errors="coerce")
        / np.clip(light_vals, 1e-3, None)
    ).replace([np.inf, -np.inf], 0.0)
    df["occ_temp_humidity_interaction"] = (
        pd.to_numeric(df["temperature"], errors="coerce")
        * pd.to_numeric(df["humidity"], errors="coerce")
    )
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        minutes_since_midnight = (
            ts.dt.hour.fillna(0).astype(float) * 60.0
            + ts.dt.minute.fillna(0).astype(float)
            + (ts.dt.second.fillna(0).astype(float) / 60.0)
        )
        df["occ_minutes_since_midnight"] = minutes_since_midnight
        df["occ_is_night"] = ((ts.dt.hour >= 22) | (ts.dt.hour < 7)).astype(float)
    else:
        df["occ_minutes_since_midnight"] = 0.0
        df["occ_is_night"] = 0.0

    if room_key_norm == "bedroom" and bool(enable_bedroom_light_texture_features):
        light_off = (light_vals < 1.0).astype(float)
        light_group = (light_off != light_off.shift(1).fillna(light_off.iloc[0] if len(light_off) > 0 else 0.0)).cumsum()
        light_off_streak = light_off.groupby(light_group).cumsum()
        df["occ_light_off_streak_30m"] = np.clip(light_off_streak, 0.0, float(lag_30m))
        df["occ_light_regime_switch"] = (
            light_off != light_off.shift(1).fillna(light_off.iloc[0] if len(light_off) > 0 else 0.0)
        ).astype(float)
    out_cols.extend(
        [
            "occ_motion_active",
            "occ_motion_inactive",
            "occ_motion_active_2m",
            "occ_motion_active_5m",
            "occ_motion_activity_ratio_15m",
            "occ_time_since_motion_active_windows",
            "occ_time_since_motion_active_minutes",
            "occ_light_roll_mean_10m",
            "occ_light_roll_std_10m",
            "occ_co2_slope_15m",
            "occ_sound_light_ratio",
            "occ_temp_humidity_interaction",
            "occ_minutes_since_midnight",
            "occ_is_night",
        ]
    )
    if use_30m:
        out_cols.extend(
            [
                "occ_motion_inactivity_ratio_30m",
                "occ_motion_roll_std_30m",
                "occ_co2_slope_30m",
            ]
        )
    if use_60m:
        out_cols.extend(
            [
                "occ_motion_inactivity_ratio_60m",
                "occ_motion_roll_std_60m",
                "occ_co2_slope_60m",
            ]
        )
    if room_key_norm == "bedroom" and bool(enable_bedroom_light_texture_features):
        out_cols.extend(["occ_light_off_streak_30m", "occ_light_regime_switch"])

    for col in out_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], 0.0).ffill().bfill().fillna(0.0)
    return out_cols


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(BACKEND_DIR.parent), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="ignore").strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _extract_day(path: Path) -> int | None:
    m = DAY_RE.search(path.stem)
    return int(m.group(1)) if m else None


def _align_room_sensors_to_timestamps(
    *,
    room_df: pd.DataFrame,
    target_timestamps: pd.Series,
) -> pd.DataFrame:
    """Align sensor columns from one room to target timestamps (same-time only)."""
    if room_df.empty or "timestamp" not in room_df.columns:
        return pd.DataFrame(index=range(int(len(target_timestamps))))

    sensor_cols = [c for c in SENSOR_COLS if c in room_df.columns]
    if not sensor_cols:
        return pd.DataFrame(index=range(int(len(target_timestamps))))

    aligned_src = room_df[["timestamp", *sensor_cols]].copy()
    aligned_src["timestamp"] = pd.to_datetime(aligned_src["timestamp"], errors="coerce")
    aligned_src = aligned_src.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    if aligned_src.empty:
        return pd.DataFrame(index=range(int(len(target_timestamps))))

    for col in sensor_cols:
        aligned_src[col] = pd.to_numeric(aligned_src[col], errors="coerce")
    aligned = aligned_src.set_index("timestamp")[sensor_cols].reindex(pd.to_datetime(target_timestamps, errors="coerce"))
    aligned = aligned.reset_index(drop=True)
    return aligned


def _add_cross_room_context_features_for_day(
    *,
    day_room_frames: Dict[str, pd.DataFrame],
    rooms: Sequence[str],
    context_rooms: Sequence[str] | None = None,
) -> None:
    """
    Add same-timestamp cross-room context features for singleton-resident occupancy heads.
    """
    room_order = [r for r in rooms if r in day_room_frames]
    context_scope = {
        str(r).strip().lower()
        for r in (context_rooms if context_rooms is not None else room_order)
        if str(r).strip()
    }
    for room in room_order:
        if str(room).strip().lower() not in context_scope:
            continue
        target_df = day_room_frames.get(room)
        if target_df is None or target_df.empty or "timestamp" not in target_df.columns:
            continue
        target_ts = pd.to_datetime(target_df["timestamp"], errors="coerce")
        n = int(len(target_df))
        if n <= 0:
            continue

        other_aligned: List[pd.DataFrame] = []
        for other_room in room_order:
            if other_room == room:
                continue
            other_df = day_room_frames.get(other_room)
            if other_df is None or other_df.empty:
                continue
            aligned = _align_room_sensors_to_timestamps(room_df=other_df, target_timestamps=target_ts)
            if not aligned.empty:
                other_aligned.append(aligned)

        for sensor in SENSOR_COLS:
            sensor_arrays: List[np.ndarray] = []
            for aligned in other_aligned:
                if sensor in aligned.columns:
                    sensor_arrays.append(aligned[sensor].to_numpy(dtype=float))
            if sensor_arrays:
                stacked = np.vstack(sensor_arrays)
                finite_mask = np.isfinite(stacked)
                valid_counts = np.sum(finite_mask, axis=0)
                safe_vals = np.where(finite_mask, stacked, 0.0)
                sum_vals = np.sum(safe_vals, axis=0)
                mean_vals = np.divide(
                    sum_vals,
                    np.maximum(valid_counts, 1),
                    out=np.zeros(shape=(n,), dtype=float),
                    where=valid_counts > 0,
                )
                max_vals = np.max(np.where(finite_mask, stacked, -np.inf), axis=0)
                max_vals = np.where(valid_counts > 0, max_vals, 0.0)
                mean_vals = np.nan_to_num(mean_vals, nan=0.0, posinf=0.0, neginf=0.0)
                max_vals = np.nan_to_num(max_vals, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                mean_vals = np.zeros(shape=(n,), dtype=float)
                max_vals = np.zeros(shape=(n,), dtype=float)
            target_df[f"ctx_other_{sensor}_mean"] = mean_vals
            target_df[f"ctx_other_{sensor}_max"] = max_vals

        motion_active_count = np.zeros(shape=(n,), dtype=float)
        rooms_reporting = np.zeros(shape=(n,), dtype=float)
        for aligned in other_aligned:
            arr = aligned.to_numpy(dtype=float) if len(aligned.columns) > 0 else np.empty((n, 0), dtype=float)
            room_has_signal = np.any(np.isfinite(arr), axis=1) if arr.size > 0 else np.zeros(shape=(n,), dtype=bool)
            rooms_reporting += room_has_signal.astype(float)

            if "motion" in aligned.columns:
                motion_vals = aligned["motion"].to_numpy(dtype=float)
                motion_active = np.logical_and(np.isfinite(motion_vals), motion_vals > 0.5)
                motion_active_count += motion_active.astype(float)

        target_df["ctx_other_rooms_reporting"] = np.nan_to_num(
            rooms_reporting, nan=0.0, posinf=0.0, neginf=0.0
        )
        target_df["ctx_other_motion_active_count"] = np.nan_to_num(
            motion_active_count, nan=0.0, posinf=0.0, neginf=0.0
        )
        target_df["ctx_other_motion_any"] = (target_df["ctx_other_motion_active_count"] > 0.0).astype(float)


def _load_room_day_data(
    files_by_day: Dict[int, Path],
    rooms: Sequence[str],
    *,
    enable_cross_room_context_features: bool = True,
    cross_room_context_rooms: Sequence[str] | None = None,
    enable_room_temporal_occupancy_features: bool = False,
    enable_bedroom_light_texture_features: bool = False,
    bedroom_livingroom_texture_profile: str = "mixed",
) -> Dict[str, Dict[int, pd.DataFrame]]:
    out: Dict[str, Dict[int, pd.DataFrame]] = defaultdict(dict)
    day_frames: Dict[int, Dict[str, pd.DataFrame]] = defaultdict(dict)

    for day, path in files_by_day.items():
        x = pd.ExcelFile(path)
        for room in rooms:
            if room not in x.sheet_names:
                continue

            df = pd.read_excel(path, sheet_name=room)
            if "timestamp" not in df.columns or "activity" not in df.columns:
                continue

            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

            df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24.0)
            df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24.0)
            df["dow_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek / 7.0)
            df["dow_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek / 7.0)

            feat_cols = [c for c in SENSOR_COLS if c in df.columns] + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
            for col in feat_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df[feat_cols] = df[feat_cols].ffill().bfill().fillna(0)

            # Room-specific feature engineering.
            room_key = str(room).strip().lower()
            if room_key == "kitchen":
                kitchen_cols = _add_kitchen_occupancy_features(df)
                feat_cols = list(dict.fromkeys(feat_cols + kitchen_cols))
                for col in kitchen_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df[kitchen_cols] = df[kitchen_cols].replace([np.inf, -np.inf], 0).ffill().bfill().fillna(0)
            elif bool(enable_room_temporal_occupancy_features) and room_key in ROOM_OCCUPANCY_ENGINEERED_ROOMS:
                occ_cols = _add_room_temporal_occupancy_features(
                    df,
                    room_key=room_key,
                    enable_bedroom_light_texture_features=bool(enable_bedroom_light_texture_features),
                    bedroom_livingroom_texture_profile=str(bedroom_livingroom_texture_profile),
                )
                feat_cols = list(dict.fromkeys(feat_cols + occ_cols))
                for col in occ_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df[occ_cols] = df[occ_cols].replace([np.inf, -np.inf], 0).ffill().bfill().fillna(0)

            df["activity"] = df["activity"].astype(str).str.strip().str.lower()
            out[room][day] = df
            day_frames[day][room] = df

    if bool(enable_cross_room_context_features):
        for day in sorted(day_frames.keys()):
            _add_cross_room_context_features_for_day(
                day_room_frames=day_frames.get(day, {}),
                rooms=rooms,
                context_rooms=cross_room_context_rooms,
            )

    return out


def _build_splits(days: Sequence[int]) -> List[Split]:
    return [Split(train_days=days[:i], test_day=days[i]) for i in range(1, len(days))]


def _compute_day_continuity(days: Sequence[int]) -> Dict[str, object]:
    day_list = sorted({int(d) for d in days})
    if not day_list:
        return {
            "days": [],
            "count": 0,
            "min_day": None,
            "max_day": None,
            "missing_days_between_min_max": [],
            "is_contiguous": False,
        }
    min_day = int(day_list[0])
    max_day = int(day_list[-1])
    missing = [int(d) for d in range(min_day, max_day + 1) if d not in set(day_list)]
    return {
        "days": [int(d) for d in day_list],
        "count": int(len(day_list)),
        "min_day": int(min_day),
        "max_day": int(max_day),
        "missing_days_between_min_max": missing,
        "is_contiguous": bool(len(missing) == 0),
    }


def _build_data_continuity_audit(
    *,
    elder_id: str,
    min_day: int,
    max_day: int,
    candidate_files: Sequence[Path],
    canonical_files: Sequence[Path],
    files_by_day: Dict[int, Path],
    excluded_non_canonical: Sequence[Dict[str, str]],
    invalid_day_token_files: Sequence[str],
) -> Dict[str, object]:
    selected_days = sorted(int(d) for d in files_by_day.keys())
    requested_days = [int(d) for d in range(int(min_day), int(max_day) + 1)] if int(max_day) >= int(min_day) else []
    missing_in_requested_window = [int(d) for d in requested_days if int(d) not in files_by_day]
    selected_window_missing_days: List[int] = []
    if selected_days:
        selected_window_missing_days = [
            int(d) for d in range(int(selected_days[0]), int(selected_days[-1]) + 1) if int(d) not in files_by_day
        ]
    selected_files = [str(files_by_day[d].name) for d in selected_days]
    return {
        "elder_id": str(elder_id),
        "requested_day_window": {"min_day": int(min_day), "max_day": int(max_day), "days": requested_days},
        "candidate_file_count": int(len(candidate_files)),
        "canonical_file_count": int(len(canonical_files)),
        "selected_file_count": int(len(selected_files)),
        "selected_days": selected_days,
        "selected_files": selected_files,
        "missing_days_in_requested_window": missing_in_requested_window,
        "missing_days_between_selected_min_max": selected_window_missing_days,
        "selected_day_continuity": _compute_day_continuity(selected_days),
        "excluded_non_canonical_files": list(excluded_non_canonical),
        "invalid_day_token_files": [str(v) for v in invalid_day_token_files],
    }


def _load_activity_label_corrections(
    csv_path: Optional[Path],
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    if csv_path is None:
        return [], {"enabled": False, "source": None, "rows_loaded": 0, "rows_valid": 0, "rows_invalid": 0}
    path = Path(csv_path)
    if not path.exists():
        raise ValueError(f"Correction CSV not found: {path}")
    df = pd.read_csv(path)
    columns = {str(c).strip().lower(): str(c) for c in df.columns}
    room_col = columns.get("room")
    label_col = columns.get("label")
    start_col = columns.get("start_time") or columns.get("start") or columns.get("start_ts")
    end_col = columns.get("end_time") or columns.get("end") or columns.get("end_ts")
    day_col = columns.get("day")
    if room_col is None or label_col is None or start_col is None or end_col is None:
        raise ValueError(
            "Correction CSV requires columns: room,label,start_time,end_time (aliases start/end accepted)."
        )
    out: List[Dict[str, object]] = []
    invalid = 0
    for _, row in df.iterrows():
        raw_room = row.get(room_col, "")
        raw_label = row.get(label_col, "")
        if pd.isna(raw_room) or pd.isna(raw_label):
            invalid += 1
            continue
        room = str(raw_room).strip().lower()
        label = str(raw_label).strip().lower()
        start_ts = pd.to_datetime(row.get(start_col), errors="coerce")
        end_ts = pd.to_datetime(row.get(end_col), errors="coerce")
        if not room or room == "nan" or not label or label == "nan" or pd.isna(start_ts) or pd.isna(end_ts):
            invalid += 1
            continue
        if pd.Timestamp(end_ts) < pd.Timestamp(start_ts):
            invalid += 1
            continue
        day_val: Optional[int] = None
        if day_col is not None:
            try:
                raw_day = row.get(day_col)
                if pd.notna(raw_day):
                    day_val = int(raw_day)
            except Exception:
                day_val = None
        if day_val is None:
            day_val = int(pd.Timestamp(start_ts).day)
        out.append(
            {
                "room": str(room),
                "label": str(label),
                "start_time": str(pd.Timestamp(start_ts)),
                "end_time": str(pd.Timestamp(end_ts)),
                "day": int(day_val),
            }
        )
    summary = {
        "enabled": True,
        "source": str(path),
        "rows_loaded": int(len(df)),
        "rows_valid": int(len(out)),
        "rows_invalid": int(invalid),
    }
    return out, summary


def _apply_activity_label_corrections(
    *,
    room_day_data: Dict[str, Dict[int, pd.DataFrame]],
    corrections: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    room_lookup = {str(k).strip().lower(): str(k) for k in room_day_data.keys()}
    by_room_day: Dict[str, Dict[str, int]] = defaultdict(lambda: {"rows_updated": 0, "windows_applied": 0})
    applied_windows = 0
    applied_rows = 0
    skipped_by_reason: Dict[str, int] = defaultdict(int)

    for item in corrections:
        room_key = str(item.get("room", "")).strip().lower()
        label = str(item.get("label", "")).strip().lower()
        if room_key not in room_lookup:
            skipped_by_reason["room_not_found"] += 1
            continue
        room_name = room_lookup[room_key]
        day = int(item.get("day", 0))
        day_df = room_day_data.get(room_name, {}).get(day)
        if day_df is None or day_df.empty:
            skipped_by_reason["day_not_found"] += 1
            continue
        start_ts = pd.to_datetime(item.get("start_time"), errors="coerce")
        end_ts = pd.to_datetime(item.get("end_time"), errors="coerce")
        if pd.isna(start_ts) or pd.isna(end_ts):
            skipped_by_reason["invalid_timestamp"] += 1
            continue
        if pd.Timestamp(end_ts) < pd.Timestamp(start_ts):
            skipped_by_reason["end_before_start"] += 1
            continue
        ts = pd.to_datetime(day_df["timestamp"], errors="coerce")
        mask = np.asarray((ts >= pd.Timestamp(start_ts)) & (ts <= pd.Timestamp(end_ts)), dtype=bool)
        n = int(np.sum(mask))
        if n <= 0:
            skipped_by_reason["no_rows_in_window"] += 1
            continue
        day_df.loc[mask, "activity"] = str(label)
        applied_windows += 1
        applied_rows += n
        key = f"{room_name}:day{day}"
        by_room_day[key]["rows_updated"] += int(n)
        by_room_day[key]["windows_applied"] += 1

    return {
        "enabled": bool(len(corrections) > 0),
        "requested_windows": int(len(corrections)),
        "applied_windows": int(applied_windows),
        "applied_rows": int(applied_rows),
        "skipped_by_reason": {str(k): int(v) for k, v in sorted(skipped_by_reason.items())},
        "by_room_day": {str(k): dict(v) for k, v in sorted(by_room_day.items())},
    }


def _build_stage_a_group_ids_from_timestamps(
    *,
    timestamps: Sequence[object],
    resolution_seconds: int,
) -> np.ndarray:
    n = int(len(timestamps))
    if n <= 0:
        return np.empty((0,), dtype=np.int64)
    res = int(max(resolution_seconds, 1))
    ts = pd.to_datetime(pd.Series(list(timestamps)), errors="coerce")
    valid_mask = ~ts.isna().to_numpy()
    out = np.zeros(shape=(n,), dtype=np.int64)
    if np.any(valid_mask):
        ts_ns = ts.astype("int64", copy=False).to_numpy(dtype=np.int64)
        out[valid_mask] = np.floor_divide(ts_ns[valid_mask], int(res * 1_000_000_000))
        next_gid = int(np.max(out[valid_mask])) + 1
    else:
        next_gid = 0
    invalid_idx = np.where(~valid_mask)[0]
    for idx in invalid_idx:
        out[idx] = int(next_gid)
        next_gid += 1
    return out


def _temporal_fit_calib_split(
    train_df: pd.DataFrame,
    *,
    calib_fraction: float,
    min_calib_samples: int,
    min_train_samples: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    n = int(len(train_df))
    if n < max(min_train_samples + 1, 2):
        return train_df, None

    desired_calib = max(int(math.ceil(n * float(calib_fraction))), int(min_calib_samples))
    max_calib = n - int(min_train_samples)
    if max_calib <= 0:
        return train_df, None
    calib_size = min(desired_calib, max_calib)
    if calib_size <= 0:
        return train_df, None

    split_idx = n - int(calib_size)
    fit_df = train_df.iloc[:split_idx].copy()
    calib_df = train_df.iloc[split_idx:].copy()
    if fit_df.empty or calib_df.empty:
        return train_df, None
    return fit_df, calib_df


def _room_xy(
    room_data: Dict[int, pd.DataFrame],
    train_days: Sequence[int],
    test_day: int,
    *,
    calib_fraction: float,
    min_calib_samples: int,
) -> Dict[str, object]:
    train_df = pd.concat([room_data[d] for d in train_days], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    test_df = room_data[test_day].copy().sort_values("timestamp").reset_index(drop=True)

    feat_cols = [c for c in SENSOR_COLS if c in train_df.columns] + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    kitchen_extra = [c for c in KITCHEN_ENGINEERED_COLS if c in train_df.columns]
    occupancy_extra = sorted([c for c in train_df.columns if str(c).startswith("occ_")])
    context_cols = sorted([c for c in train_df.columns if str(c).startswith("ctx_other_")])
    if kitchen_extra:
        feat_cols = list(dict.fromkeys(feat_cols + kitchen_extra))
    if occupancy_extra:
        feat_cols = list(dict.fromkeys(feat_cols + occupancy_extra))
    if context_cols:
        feat_cols = list(dict.fromkeys(feat_cols + context_cols))
    fit_df, calib_df = _temporal_fit_calib_split(
        train_df,
        calib_fraction=calib_fraction,
        min_calib_samples=min_calib_samples,
    )

    payload: Dict[str, object] = {
        "fit_size": int(len(fit_df)),
        "calib_size": int(len(calib_df)) if calib_df is not None else 0,
        "test_size": int(len(test_df)),
        "fit_end_timestamp": fit_df["timestamp"].max() if len(fit_df) > 0 else None,
        "calib_start_timestamp": calib_df["timestamp"].min() if calib_df is not None and len(calib_df) > 0 else None,
        "calib_end_timestamp": calib_df["timestamp"].max() if calib_df is not None and len(calib_df) > 0 else None,
        "test_start_timestamp": test_df["timestamp"].min() if len(test_df) > 0 else None,
    }

    x_fit = fit_df[feat_cols].to_numpy(dtype=float)
    y_fit = fit_df["activity"].to_numpy()
    x_test = test_df[feat_cols].to_numpy(dtype=float)
    y_test = test_df["activity"].to_numpy()
    payload.update(
        {
            "x_fit": x_fit,
            "y_fit": y_fit,
            "x_test": x_test,
            "y_test": y_test,
            "fit_df": fit_df.copy(),
            "test_df": test_df,
            "feature_columns": feat_cols,
            "calib_df": calib_df.copy() if calib_df is not None else pd.DataFrame(columns=train_df.columns),
        }
    )
    if calib_df is None:
        payload["x_calib"] = np.empty((0, len(feat_cols)), dtype=float)
        payload["y_calib"] = np.empty((0,), dtype=object)
    else:
        payload["x_calib"] = calib_df[feat_cols].to_numpy(dtype=float)
        payload["y_calib"] = calib_df["activity"].to_numpy()
    return payload


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "occupied_precision": 0.0,
            "occupied_recall": 0.0,
            "occupied_f1": 0.0,
        }
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    yt = np.asarray([str(v).strip().lower() for v in y_true], dtype=object)
    yp = np.asarray([str(v).strip().lower() for v in y_pred], dtype=object)

    # Occupancy hard-gates are timeline-focused; smooth short gaps/bursts
    # before computing occupied precision/recall/F1.
    def _smooth_occ(mask: np.ndarray, *, min_run_windows: int = 3, gap_fill_windows: int = 2) -> np.ndarray:
        out = np.asarray(mask, dtype=bool).copy()
        n = int(len(out))
        if n == 0:
            return out
        if int(gap_fill_windows) > 0:
            i = 0
            while i < n:
                if out[i]:
                    i += 1
                    continue
                start = i
                while i < n and not out[i]:
                    i += 1
                end = i
                has_left = start > 0 and out[start - 1]
                has_right = end < n and out[end]
                if has_left and has_right and (end - start) <= int(gap_fill_windows):
                    out[start:end] = True
        if int(min_run_windows) > 1:
            i = 0
            while i < n:
                if not out[i]:
                    i += 1
                    continue
                start = i
                while i < n and out[i]:
                    i += 1
                end = i
                if (end - start) < int(min_run_windows):
                    out[start:end] = False
        return out

    true_occ = _smooth_occ(np.asarray(yt != "unoccupied", dtype=bool))
    pred_occ = _smooth_occ(np.asarray(yp != "unoccupied", dtype=bool))
    tp = int(np.sum(np.logical_and(true_occ, pred_occ)))
    fp = int(np.sum(np.logical_and(~true_occ, pred_occ)))
    fn = int(np.sum(np.logical_and(true_occ, ~pred_occ)))
    occupied_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    occupied_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    occupied_f1 = (
        float((2.0 * occupied_precision * occupied_recall) / (occupied_precision + occupied_recall))
        if (occupied_precision + occupied_recall) > 0
        else 0.0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "occupied_precision": occupied_precision,
        "occupied_recall": occupied_recall,
        "occupied_f1": occupied_f1,
    }


def _label_recall_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Per-label recall/support summary for gate checks."""
    yt = np.asarray([str(v).strip().lower() for v in y_true], dtype=object)
    yp = np.asarray([str(v).strip().lower() for v in y_pred], dtype=object)
    labels = sorted(set(yt.tolist()))
    out: Dict[str, Dict[str, float]] = {}
    for label in labels:
        support = int(np.sum(yt == label))
        if support <= 0:
            out[label] = {"support": 0.0, "recall": 0.0}
            continue
        tp = int(np.sum((yt == label) & (yp == label)))
        out[label] = {"support": float(support), "recall": float(tp / support)}
    return out


DIAGNOSTIC_KEY_LABELS = (
    "unoccupied",
    "sleep",
    "bedroom_normal_use",
    "livingroom_normal_use",
    "kitchen_normal_use",
    "shower",
    "out",
    "unknown",
)


def _phase_occupied_snapshot(labels: np.ndarray) -> Dict[str, float]:
    arr = np.asarray([str(v).strip().lower() for v in labels], dtype=object)
    total = int(len(arr))
    if total <= 0:
        return {"windows": 0.0, "occupied_windows": 0.0, "occupied_rate": 0.0}
    occupied_windows = int(np.sum(arr != "unoccupied"))
    return {
        "windows": float(total),
        "occupied_windows": float(occupied_windows),
        "occupied_rate": float(occupied_windows / total),
    }


def _phase_label_minutes_snapshot(
    labels: np.ndarray,
    *,
    key_labels: Sequence[str] = DIAGNOSTIC_KEY_LABELS,
    window_seconds: int = 10,
) -> Dict[str, float]:
    arr = np.asarray([str(v).strip().lower() for v in labels], dtype=object)
    minutes_per_window = float(max(int(window_seconds), 1) / 60.0)
    counts: Dict[str, float] = {}
    for label in key_labels:
        label_key = str(label).strip().lower()
        counts[label_key] = float(np.sum(arr == label_key) * minutes_per_window)
    occupied_windows = int(np.sum(arr != "unoccupied"))
    counts["occupied_total"] = float(occupied_windows * minutes_per_window)
    return counts


def _build_room_data_diagnostics(
    *,
    y_fit: np.ndarray,
    y_calib: np.ndarray,
    y_test: np.ndarray,
    key_labels: Sequence[str] = DIAGNOSTIC_KEY_LABELS,
    window_seconds: int = 10,
) -> Dict[str, object]:
    return {
        "window_seconds": int(max(window_seconds, 1)),
        "key_labels": [str(v).strip().lower() for v in key_labels if str(v).strip()],
        "occupied_rate_snapshot": {
            "fit": _phase_occupied_snapshot(np.asarray(y_fit, dtype=object)),
            "calib": _phase_occupied_snapshot(np.asarray(y_calib, dtype=object)),
            "test": _phase_occupied_snapshot(np.asarray(y_test, dtype=object)),
        },
        "label_minutes_snapshot": {
            "fit": _phase_label_minutes_snapshot(
                np.asarray(y_fit, dtype=object),
                key_labels=key_labels,
                window_seconds=window_seconds,
            ),
            "calib": _phase_label_minutes_snapshot(
                np.asarray(y_calib, dtype=object),
                key_labels=key_labels,
                window_seconds=window_seconds,
            ),
            "test": _phase_label_minutes_snapshot(
                np.asarray(y_test, dtype=object),
                key_labels=key_labels,
                window_seconds=window_seconds,
            ),
        },
    }


def _label_correction_reference_for_room_day(
    *,
    label_corrections_summary: Dict[str, object],
    room: str,
    day: int,
) -> Dict[str, object]:
    load_summary = (
        label_corrections_summary.get("load", {})
        if isinstance(label_corrections_summary, dict)
        else {}
    )
    apply_summary = (
        label_corrections_summary.get("apply", {})
        if isinstance(label_corrections_summary, dict)
        else {}
    )
    by_room_day = apply_summary.get("by_room_day", {}) if isinstance(apply_summary, dict) else {}
    room_day_key = f"{str(room).strip()}:day{int(day)}"
    room_day_payload = by_room_day.get(room_day_key, {}) if isinstance(by_room_day, dict) else {}
    if not isinstance(room_day_payload, dict):
        room_day_payload = {}
    return {
        "enabled": bool(load_summary.get("enabled", False)),
        "rows_loaded": int(load_summary.get("rows_loaded", 0) or 0),
        "rows_valid": int(load_summary.get("rows_valid", 0) or 0),
        "requested_windows": int(apply_summary.get("requested_windows", 0) or 0),
        "applied_windows": int(apply_summary.get("applied_windows", 0) or 0),
        "applied_rows": int(apply_summary.get("applied_rows", 0) or 0),
        "room_day": {
            "room_day_key": room_day_key,
            "windows_applied": int(room_day_payload.get("windows_applied", 0) or 0),
            "rows_updated": int(room_day_payload.get("rows_updated", 0) or 0),
        },
    }


def _compute_train_day_label_support(
    *,
    room_data: Dict[int, pd.DataFrame],
    train_days: Sequence[int],
    label_names: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    labels = sorted({str(v).strip().lower() for v in label_names if str(v).strip()})
    for label in labels:
        day_support: Dict[str, int] = {}
        zero_support_days: List[int] = []
        for day in sorted(int(d) for d in train_days):
            day_df = room_data.get(day)
            support = 0
            if isinstance(day_df, pd.DataFrame) and not day_df.empty and "activity" in day_df.columns:
                day_labels = day_df["activity"].astype(str).str.strip().str.lower().to_numpy(dtype=object)
                support = int(np.sum(day_labels == label))
            day_support[str(int(day))] = int(support)
            if int(support) <= 0:
                zero_support_days.append(int(day))
        out[label] = {
            "label": str(label),
            "day_support": day_support,
            "zero_support_days": sorted(zero_support_days),
            "all_days_supported": bool(len(zero_support_days) == 0),
            "min_day_support": int(min(day_support.values())) if day_support else 0,
            "max_day_support": int(max(day_support.values())) if day_support else 0,
        }
    return out


def _extract_feature_importance_ranking(
    *,
    model: EventFirstTwoStageModel,
    feature_columns: Sequence[str],
    top_k: int = 12,
) -> Dict[str, object]:
    feature_names = [str(c) for c in feature_columns]
    top_n = int(max(top_k, 1))

    def _from_estimator(estimator: object, *, stage_name: str) -> Dict[str, object]:
        if estimator is None or not hasattr(estimator, "feature_importances_"):
            return {"available": False, "reason": "not_supported"}
        values = np.asarray(getattr(estimator, "feature_importances_"), dtype=float).reshape(-1)
        if len(values) != len(feature_names):
            return {
                "available": False,
                "reason": "feature_count_mismatch",
                "feature_count": int(len(values)),
                "expected_feature_count": int(len(feature_names)),
                "stage": str(stage_name),
            }
        ranked_idx = np.argsort(values)[::-1][:top_n]
        ranking = [
            {"feature": feature_names[int(idx)], "importance": float(values[int(idx)])}
            for idx in ranked_idx
        ]
        return {
            "available": True,
            "feature_count": int(len(values)),
            "top_features": ranking,
            "importance_sum": float(np.sum(values)),
        }

    return {
        "stage_a": _from_estimator(model.stage_a, stage_name="stage_a"),
        "stage_b": _from_estimator(model.stage_b, stage_name="stage_b"),
    }


def _smooth_binary_mask(
    mask: np.ndarray,
    *,
    min_run_windows: int = 3,
    gap_fill_windows: int = 2,
) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    n = int(len(out))
    if n == 0:
        return out

    if int(gap_fill_windows) > 0:
        i = 0
        while i < n:
            if out[i]:
                i += 1
                continue
            start = i
            while i < n and not out[i]:
                i += 1
            end = i
            has_left = start > 0 and out[start - 1]
            has_right = end < n and out[end]
            if has_left and has_right and (end - start) <= int(gap_fill_windows):
                out[start:end] = True

    if int(min_run_windows) > 1:
        i = 0
        while i < n:
            if not out[i]:
                i += 1
                continue
            start = i
            while i < n and out[i]:
                i += 1
            end = i
            if (end - start) < int(min_run_windows):
                out[start:end] = False
    return out


def _care_fragmentation_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    min_run_windows: int = 3,
    gap_fill_windows: int = 2,
    smooth_pred_mask: bool = True,
) -> float:
    """
    Return [0,1] care-episode fragmentation similarity score.
    1.0 means predicted care-episode count matches truth.
    """
    yt = np.asarray([str(v).strip().lower() for v in y_true], dtype=object)
    yp = np.asarray([str(v).strip().lower() for v in y_pred], dtype=object)
    excluded = {"unoccupied", "unknown"}
    true_mask_raw = np.asarray([v not in excluded for v in yt], dtype=bool)
    pred_mask_raw = np.asarray([v not in excluded for v in yp], dtype=bool)
    true_mask = _smooth_binary_mask(
        true_mask_raw,
        min_run_windows=int(max(min_run_windows, 1)),
        gap_fill_windows=int(max(gap_fill_windows, 0)),
    )
    if bool(smooth_pred_mask):
        pred_mask = _smooth_binary_mask(
            pred_mask_raw,
            min_run_windows=int(max(min_run_windows, 1)),
            gap_fill_windows=int(max(gap_fill_windows, 0)),
        )
    else:
        pred_mask = np.asarray(pred_mask_raw, dtype=bool)
    true_runs = len(_iter_true_runs(true_mask))
    pred_runs = len(_iter_true_runs(pred_mask))
    if true_runs == 0 and pred_runs == 0:
        return 1.0
    if true_runs == 0 or pred_runs == 0:
        return 0.0
    # Symmetric normalization avoids collapsing to zero too aggressively
    # when run counts differ but are still directionally close.
    err_ratio = abs(float(pred_runs - true_runs)) / float(max(true_runs, pred_runs, 1))
    return float(max(0.0, 1.0 - min(err_ratio, 1.0)))


def _apply_room_prediction_occupancy_smoothing(
    *,
    room_key: str,
    y_pred: np.ndarray,
    activity_probs: Dict[str, np.ndarray],
    min_run_windows: int,
    gap_fill_windows: int,
) -> tuple[np.ndarray, Dict[str, object]]:
    """
    Apply occupancy-mask smoothing to final predictions.

    This mutates actual timeline outputs (not just scoring) to keep what users
    see aligned with what hard-gates evaluate.
    """
    room = str(room_key).strip().lower()
    if room not in {"bedroom", "livingroom"}:
        return np.asarray(y_pred, dtype=object), {"applied": False, "reason": "not_applicable"}
    labels = np.asarray(y_pred, dtype=object).copy()
    if len(labels) == 0:
        return labels, {"applied": False, "reason": "empty_prediction"}

    lower_labels = np.asarray([str(v).strip().lower() for v in labels], dtype=object)
    occupied_raw = np.asarray(
        [lab not in {"unoccupied", "unknown", ""} for lab in lower_labels],
        dtype=bool,
    )
    occupied_smooth = _smooth_binary_mask(
        occupied_raw,
        min_run_windows=int(max(min_run_windows, 1)),
        gap_fill_windows=int(max(gap_fill_windows, 0)),
    )
    if np.array_equal(occupied_raw, occupied_smooth):
        return labels, {
            "applied": True,
            "changed_windows": 0,
            "before_occupied_windows": int(np.sum(occupied_raw)),
            "after_occupied_windows": int(np.sum(occupied_smooth)),
            "min_run_windows": int(max(min_run_windows, 1)),
            "gap_fill_windows": int(max(gap_fill_windows, 0)),
            "reason": "no_change",
        }

    activity_labels = [
        str(label).strip()
        for label in sorted(activity_probs.keys())
        if str(label).strip().lower() not in {"", "unoccupied", "unknown"}
    ]
    activity_matrix = None
    if activity_labels:
        matrix = np.zeros(shape=(len(labels), len(activity_labels)), dtype=float)
        for j, label in enumerate(activity_labels):
            arr = np.asarray(activity_probs.get(label, np.zeros(len(labels))), dtype=float)
            if len(arr) < len(labels):
                arr = np.resize(arr, len(labels)).astype(float)
            matrix[:, j] = np.clip(arr[: len(labels)], 0.0, 1.0)
        activity_matrix = matrix

    default_room_label = "bedroom_normal_use" if room == "bedroom" else "livingroom_normal_use"
    changed_windows = 0
    for idx in range(len(labels)):
        if not bool(occupied_smooth[idx]):
            if str(lower_labels[idx]) != "unoccupied":
                labels[idx] = "unoccupied"
                changed_windows += 1
            continue

        if bool(occupied_raw[idx]) and str(lower_labels[idx]) not in {"unoccupied", "unknown", ""}:
            continue

        replacement: Optional[str] = None
        if activity_matrix is not None and activity_matrix.shape[1] > 0:
            best_idx = int(np.argmax(activity_matrix[idx]))
            replacement = str(activity_labels[best_idx])
        if not replacement:
            left = idx - 1
            while left >= 0:
                lbl = str(labels[left]).strip().lower()
                if lbl not in {"unoccupied", "unknown", ""}:
                    replacement = str(labels[left])
                    break
                left -= 1
        if not replacement:
            right = idx + 1
            while right < len(labels):
                lbl = str(labels[right]).strip().lower()
                if lbl not in {"unoccupied", "unknown", ""}:
                    replacement = str(labels[right])
                    break
                right += 1
        if not replacement:
            replacement = default_room_label

        if str(labels[idx]) != str(replacement):
            labels[idx] = replacement
            changed_windows += 1

    return labels, {
        "applied": True,
        "changed_windows": int(changed_windows),
        "before_occupied_windows": int(np.sum(occupied_raw)),
        "after_occupied_windows": int(np.sum(occupied_smooth)),
        "min_run_windows": int(max(min_run_windows, 1)),
        "gap_fill_windows": int(max(gap_fill_windows, 0)),
        "activity_labels_used": list(activity_labels),
    }


def _apply_room_passive_occupancy_hysteresis(
    *,
    room_key: str,
    y_pred: np.ndarray,
    occupancy_probs: np.ndarray,
    activity_probs: Dict[str, np.ndarray],
    motion_values: Optional[np.ndarray] = None,
    hold_minutes: float = 30.0,
    exit_min_consecutive_windows: int = 18,
    entry_occ_threshold: float = 0.56,
    entry_room_prob_threshold: float = 0.40,
    stay_occ_threshold: float = 0.22,
    stay_room_prob_threshold: float = 0.10,
    exit_occ_threshold: float = 0.12,
    exit_room_prob_threshold: float = 0.06,
    motion_reset_threshold: float = 0.55,
    motion_quiet_threshold: float = 0.12,
    livingroom_strict_entry_requires_strong_signal: bool = False,
    livingroom_entry_motion_threshold: float = 0.75,
) -> tuple[np.ndarray, Dict[str, object]]:
    """
    Passive-occupancy hysteresis for Bedroom/LivingRoom.

    Extends occupied state across low-motion windows after entry evidence,
    while requiring sustained low-evidence windows before forcing exit.
    """
    out = np.asarray(y_pred, dtype=object).copy()
    room = str(room_key).strip().lower()
    if room not in {"bedroom", "livingroom"}:
        return out, {"applied": False, "reason": "not_applicable", "room": room}

    n = int(len(out))
    if n == 0:
        return out, {"applied": False, "reason": "empty_prediction", "room": room}

    # Explicit zero/negative hold disables passive hysteresis for the room.
    # This is used by LR-only tuning variants to avoid Bedroom side effects.
    if float(hold_minutes) <= 0.0:
        return out, {
            "applied": False,
            "reason": "disabled_by_hold_minutes",
            "room": room,
            "hold_minutes": float(hold_minutes),
        }

    occ = np.asarray(occupancy_probs, dtype=float)
    if len(occ) != n:
        return out, {"applied": False, "reason": "occupancy_length_mismatch", "room": room}

    room_labels = [
        str(label).strip().lower()
        for label in sorted(activity_probs.keys())
        if str(label).strip().lower() not in {"", "unoccupied", "unknown"}
    ]
    if not room_labels:
        default_room_label = "sleep" if room == "bedroom" else "livingroom_normal_use"
        room_labels = [default_room_label]
    room_prob = np.zeros(shape=(n,), dtype=float)
    for label in room_labels:
        p = np.asarray(activity_probs.get(label, np.zeros(n)), dtype=float)
        if len(p) < n:
            p = np.resize(p, n).astype(float)
        room_prob = np.maximum(room_prob, np.clip(p[:n], 0.0, 1.0))

    if motion_values is None:
        motion = np.zeros(shape=(n,), dtype=float)
    else:
        motion = np.asarray(motion_values, dtype=float)
        if len(motion) < n:
            motion = np.resize(motion, n).astype(float)
        motion = np.nan_to_num(motion[:n], nan=0.0, posinf=0.0, neginf=0.0)

    base_occupied = np.asarray(
        [str(v).strip().lower() not in {"unoccupied", "unknown", ""} for v in out],
        dtype=bool,
    )
    active_mask = np.asarray(base_occupied, dtype=bool).copy()

    hold_windows = int(max(round(float(max(hold_minutes, 0.0)) * 6.0), 0))
    exit_windows = int(max(exit_min_consecutive_windows, 1))
    strict_entry = bool(room == "livingroom" and livingroom_strict_entry_requires_strong_signal)
    strict_entry_motion_threshold = float(max(livingroom_entry_motion_threshold, 0.0))
    if strict_entry:
        # In strict-entry mode we treat Stage-A/Stage-B labels as evidence only, not state.
        active_mask[:] = False
    hold_remaining = 0
    exit_low_count = 0
    held_extension_windows = 0
    forced_exit_windows = 0

    for i in range(n):
        was_active = bool(active_mask[i - 1]) if i > 0 else False
        entry_from_prob = bool(
            occ[i] >= float(entry_occ_threshold)
            and room_prob[i] >= float(entry_room_prob_threshold)
        )
        entry_from_motion = bool(
            motion[i]
            >= (
                strict_entry_motion_threshold
                if strict_entry
                else float(motion_reset_threshold)
            )
        )
        has_entry_signal = bool(
            (entry_from_prob or entry_from_motion)
            if strict_entry
            else (base_occupied[i] or entry_from_prob or motion[i] >= float(motion_reset_threshold))
        )
        has_stay_signal = bool(
            (
                occ[i] >= float(stay_occ_threshold)
                and room_prob[i] >= float(stay_room_prob_threshold)
            )
            or motion[i] >= float(motion_reset_threshold)
        )
        hard_exit_signal = bool(
            occ[i] <= float(exit_occ_threshold)
            and room_prob[i] <= float(exit_room_prob_threshold)
            and motion[i] <= float(motion_quiet_threshold)
        )

        if has_entry_signal:
            active_mask[i] = True
            hold_remaining = int(hold_windows)
            exit_low_count = 0
            continue

        if has_stay_signal and (not strict_entry or hold_remaining > 0 or was_active):
            active_mask[i] = True
            hold_remaining = int(hold_windows)
            exit_low_count = 0
            continue

        if hold_remaining > 0:
            hold_remaining -= 1
            if hard_exit_signal:
                exit_low_count += 1
            else:
                exit_low_count = 0
            if exit_low_count >= exit_windows:
                hold_remaining = 0
                active_mask[i] = False
                forced_exit_windows += 1
            else:
                active_mask[i] = True
                if not bool(base_occupied[i]):
                    held_extension_windows += 1
            continue

        active_mask[i] = False
        if bool(base_occupied[i]):
            forced_exit_windows += 1

    activity_labels = [str(label).strip() for label in room_labels]
    room_label_matrix = None
    if activity_labels:
        room_label_matrix = np.zeros(shape=(n, len(activity_labels)), dtype=float)
        for j, label in enumerate(activity_labels):
            p = np.asarray(activity_probs.get(label, np.zeros(n)), dtype=float)
            if len(p) < n:
                p = np.resize(p, n).astype(float)
            room_label_matrix[:, j] = np.clip(p[:n], 0.0, 1.0)
    default_room_label = "sleep" if room == "bedroom" else "livingroom_normal_use"
    changed_windows = 0
    for i in range(n):
        if not bool(active_mask[i]):
            if str(out[i]).strip().lower() != "unoccupied":
                out[i] = "unoccupied"
                changed_windows += 1
            continue
        if str(out[i]).strip().lower() not in {"unoccupied", "unknown", ""}:
            continue
        replacement: Optional[str] = None
        if room_label_matrix is not None and room_label_matrix.shape[1] > 0:
            best_idx = int(np.argmax(room_label_matrix[i]))
            replacement = str(activity_labels[best_idx])
        if not replacement:
            replacement = str(default_room_label)
        out[i] = replacement
        changed_windows += 1

    after_occupied = int(np.sum(np.asarray(out != "unoccupied", dtype=bool)))
    before_occupied = int(np.sum(base_occupied))
    return out, {
        "applied": True,
        "reason": "passive_occupancy_hysteresis",
        "room": room,
        "changed_windows": int(changed_windows),
        "before_occupied_windows": int(before_occupied),
        "after_occupied_windows": int(after_occupied),
        "delta_occupied_windows": int(after_occupied - before_occupied),
        "held_extension_windows": int(held_extension_windows),
        "forced_exit_windows": int(forced_exit_windows),
        "config": {
            "hold_minutes": float(max(hold_minutes, 0.0)),
            "exit_min_consecutive_windows": int(exit_windows),
            "entry_occ_threshold": float(entry_occ_threshold),
            "entry_room_prob_threshold": float(entry_room_prob_threshold),
            "stay_occ_threshold": float(stay_occ_threshold),
            "stay_room_prob_threshold": float(stay_room_prob_threshold),
            "exit_occ_threshold": float(exit_occ_threshold),
            "exit_room_prob_threshold": float(exit_room_prob_threshold),
            "motion_reset_threshold": float(motion_reset_threshold),
            "motion_quiet_threshold": float(motion_quiet_threshold),
            "livingroom_strict_entry_requires_strong_signal": bool(
                livingroom_strict_entry_requires_strong_signal
            ),
            "livingroom_entry_motion_threshold": float(max(livingroom_entry_motion_threshold, 0.0)),
        },
    }


def _episodes_to_timeline_dicts(episodes: Sequence[object]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for ep in episodes:
        start = getattr(ep, "start_time", None)
        end = getattr(ep, "end_time", None)
        if start is None or end is None:
            continue
        out.append(
            {
                "label": str(getattr(ep, "label", "unknown")),
                "start_time": str(start),
                "end_time": str(end),
                "duration_minutes": float(max(float(getattr(ep, "duration_seconds", 0.0)) / 60.0, 0.0)),
            }
        )
    return out


def _compute_room_timeline_gate_payload(
    *,
    room_key: str,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    enabled: bool,
) -> Dict[str, object]:
    """
    Compute timeline metrics + gate results for Bedroom/LivingRoom.
    Returns payload with passed/total counts suitable for gate stack.
    """
    if not enabled:
        return {"enabled": False, "applied": False, "reason": "disabled", "metrics": {}, "gates": [], "passed": 0, "total": 0}
    if room_key not in {"bedroom", "livingroom"}:
        return {
            "enabled": True,
            "applied": False,
            "reason": "room_not_in_scope",
            "metrics": {},
            "gates": [],
            "passed": 0,
            "total": 0,
        }
    if len(test_df) == 0 or len(y_true) == 0 or len(y_pred) == 0:
        return {"enabled": True, "applied": False, "reason": "empty_inputs", "metrics": {}, "gates": [], "passed": 0, "total": 0}

    gt_df = test_df[["timestamp"]].copy()
    gt_df["activity"] = np.asarray(y_true, dtype=object)
    pred_df = test_df[["timestamp"]].copy()
    pred_df["activity"] = np.asarray(y_pred, dtype=object)

    gt_eps = labels_to_episodes(gt_df, timestamp_col="timestamp", label_col="activity", drop_labels=["unoccupied", "unknown"])
    pred_eps = labels_to_episodes(pred_df, timestamp_col="timestamp", label_col="activity", drop_labels=["unoccupied", "unknown"])
    gt_dicts = _episodes_to_timeline_dicts(gt_eps)
    pred_dicts = _episodes_to_timeline_dicts(pred_eps)

    metrics: TimelineMetrics = compute_timeline_metrics(pred_dicts, gt_dicts, tolerance_minutes=5.0)
    checker = create_default_timeline_checker()
    gate_results = checker.check_timeline_metrics({room_key: metrics})

    gates_payload: List[Dict[str, object]] = []
    passed = 0
    total = 0
    for result in gate_results:
        if str(result.room or "").strip().lower() != room_key:
            continue
        total += 1
        is_pass = bool(result.is_pass)
        if is_pass:
            passed += 1
        gates_payload.append(
            {
                "gate_name": str(result.gate_name),
                "status": str(result.status.value),
                "pass": is_pass,
                "metric_value": float(result.metric_value) if result.metric_value is not None else None,
                "threshold_value": float(result.threshold_value) if result.threshold_value is not None else None,
                "message": str(result.message),
            }
        )

    return {
        "enabled": True,
        "applied": True,
        "reason": "computed",
        "metrics": metrics.to_dict(),
        "gates": gates_payload,
        "passed": int(passed),
        "total": int(total),
    }


def _estimate_window_seconds(timestamps: pd.Series, *, default_seconds: int = 10) -> float:
    ts = pd.to_datetime(timestamps, errors="coerce")
    if len(ts) < 2:
        return float(max(default_seconds, 1))
    delta = ts.diff().dt.total_seconds().dropna()
    if len(delta) == 0:
        return float(max(default_seconds, 1))
    delta = delta[np.isfinite(delta.to_numpy(dtype=float))]
    delta = delta[delta > 0.0]
    if len(delta) == 0:
        return float(max(default_seconds, 1))
    return float(max(np.median(delta.to_numpy(dtype=float)), 1.0))


def _compute_binary_episode_metrics(
    *,
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: str = "occupied",
    tolerance_minutes: float = 5.0,
) -> Dict[str, object]:
    """
    Episode-level diagnostics for binary occupied-vs-unoccupied behavior.
    """
    yt = np.asarray([str(v).strip().lower() for v in np.asarray(y_true, dtype=object)], dtype=object)
    yp = np.asarray([str(v).strip().lower() for v in np.asarray(y_pred, dtype=object)], dtype=object)
    n = int(min(len(yt), len(yp)))
    if n <= 0:
        return {
            "positive_label": str(positive_label),
            "window_seconds": 10.0,
            "windows_total": 0,
            "occupied_windows_true": 0,
            "occupied_windows_pred": 0,
            "tp_windows": 0,
            "fp_windows": 0,
            "fn_windows": 0,
            "precision_windows": 0.0,
            "recall_windows": 0.0,
            "f1_windows": 0.0,
            "occupied_minutes_true": 0.0,
            "occupied_minutes_pred": 0.0,
            "tp_minutes": 0.0,
            "fp_minutes": 0.0,
            "fn_minutes": 0.0,
            "precision_minutes": 0.0,
            "recall_minutes": 0.0,
            "f1_minutes": 0.0,
            "episode_precision": 0.0,
            "episode_recall": 0.0,
            "episode_f1": 0.0,
            "timeline_metrics_binary": {},
        }

    ts = pd.to_datetime(pd.Series(timestamps).iloc[:n], errors="coerce")
    window_seconds = _estimate_window_seconds(ts, default_seconds=10)
    minutes_per_window = float(window_seconds / 60.0)

    true_occ = np.asarray(yt[:n] != "unoccupied", dtype=bool)
    pred_occ = np.asarray(yp[:n] != "unoccupied", dtype=bool)
    tp = int(np.sum(np.logical_and(true_occ, pred_occ)))
    fp = int(np.sum(np.logical_and(~true_occ, pred_occ)))
    fn = int(np.sum(np.logical_and(true_occ, ~pred_occ)))

    precision_windows = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall_windows = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_windows = (
        float((2.0 * precision_windows * recall_windows) / (precision_windows + recall_windows))
        if (precision_windows + recall_windows) > 0
        else 0.0
    )

    gt_df = pd.DataFrame(
        {
            "timestamp": ts,
            "activity": np.where(true_occ, str(positive_label), "unoccupied"),
        }
    )
    pred_df = pd.DataFrame(
        {
            "timestamp": ts,
            "activity": np.where(pred_occ, str(positive_label), "unoccupied"),
        }
    )
    gt_eps = labels_to_episodes(gt_df, timestamp_col="timestamp", label_col="activity", drop_labels=["unoccupied"])
    pred_eps = labels_to_episodes(pred_df, timestamp_col="timestamp", label_col="activity", drop_labels=["unoccupied"])
    gt_dicts = _episodes_to_timeline_dicts(gt_eps)
    pred_dicts = _episodes_to_timeline_dicts(pred_eps)
    tmetrics = compute_timeline_metrics(pred_dicts, gt_dicts, tolerance_minutes=float(max(tolerance_minutes, 0.0)))
    matched = int(tmetrics.matched_episodes)
    gt_count = int(tmetrics.num_gt_episodes)
    pred_count = int(tmetrics.num_pred_episodes)
    episode_precision = float(matched / pred_count) if pred_count > 0 else 0.0
    episode_recall = float(matched / gt_count) if gt_count > 0 else 0.0
    episode_f1 = (
        float((2.0 * episode_precision * episode_recall) / (episode_precision + episode_recall))
        if (episode_precision + episode_recall) > 0
        else 0.0
    )

    return {
        "positive_label": str(positive_label),
        "window_seconds": float(window_seconds),
        "windows_total": int(n),
        "occupied_windows_true": int(np.sum(true_occ)),
        "occupied_windows_pred": int(np.sum(pred_occ)),
        "tp_windows": int(tp),
        "fp_windows": int(fp),
        "fn_windows": int(fn),
        "precision_windows": float(precision_windows),
        "recall_windows": float(recall_windows),
        "f1_windows": float(f1_windows),
        "occupied_minutes_true": float(np.sum(true_occ) * minutes_per_window),
        "occupied_minutes_pred": float(np.sum(pred_occ) * minutes_per_window),
        "tp_minutes": float(tp * minutes_per_window),
        "fp_minutes": float(fp * minutes_per_window),
        "fn_minutes": float(fn * minutes_per_window),
        "precision_minutes": float(precision_windows),
        "recall_minutes": float(recall_windows),
        "f1_minutes": float(f1_windows),
        "episode_precision": float(episode_precision),
        "episode_recall": float(episode_recall),
        "episode_f1": float(episode_f1),
        "timeline_metrics_binary": tmetrics.to_dict(),
    }


def _extract_error_episodes(
    *,
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: str,
    top_k: int = 5,
) -> Dict[str, object]:
    """
    Extract top FN/FP contiguous episodes for a target label.
    """
    ts = pd.to_datetime(timestamps, errors="coerce")
    if len(ts) == 0 or len(y_true) == 0 or len(y_pred) == 0:
        return {
            "positive_label": str(positive_label),
            "fn_minutes_total": 0.0,
            "fp_minutes_total": 0.0,
            "fn_top": [],
            "fp_top": [],
        }

    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    fn_mask = np.asarray((yt == positive_label) & (yp != positive_label), dtype=bool)
    fp_mask = np.asarray((yt != positive_label) & (yp == positive_label), dtype=bool)

    def _episodes(mask: np.ndarray) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        i = 0
        n = int(len(mask))
        while i < n:
            if not bool(mask[i]):
                i += 1
                continue
            start = i
            while i < n and bool(mask[i]):
                i += 1
            end = i
            start_ts = pd.Timestamp(ts.iloc[start])
            end_ts = pd.Timestamp(ts.iloc[end - 1])
            duration_sec = max((end_ts - start_ts).total_seconds() + 10.0, 0.0)
            out.append(
                {
                    "start": str(start_ts),
                    "end": str(end_ts),
                    "duration_minutes": float(duration_sec / 60.0),
                    "start_index": int(start),
                    "end_index": int(end),
                }
            )
        return out

    fn_eps = _episodes(fn_mask)
    fp_eps = _episodes(fp_mask)
    fn_top = sorted(fn_eps, key=lambda x: float(x["duration_minutes"]), reverse=True)[: int(max(1, top_k))]
    fp_top = sorted(fp_eps, key=lambda x: float(x["duration_minutes"]), reverse=True)[: int(max(1, top_k))]
    return {
        "positive_label": str(positive_label),
        "fn_minutes_total": float(np.sum(fn_mask) * 10.0 / 60.0),
        "fp_minutes_total": float(np.sum(fp_mask) * 10.0 / 60.0),
        "fn_top": fn_top,
        "fp_top": fp_top,
    }


def _split_hard_gate(
    room: str,
    gt_daily: Dict[str, float],
    pred_daily: Dict[str, float],
    *,
    cls: Dict[str, float],
    label_recall_summary: Dict[str, Dict[str, float]],
    fragmentation_score: float,
    unknown_rate: float,
    unknown_rate_cap: float,
    room_metric_floors: Dict[str, Dict[str, float]],
    room_label_recall_floors: Dict[str, Dict[str, float]],
    room_label_recall_min_supports: Dict[str, int],
    label_recall_train_day_support: Optional[Dict[str, Dict[str, object]]] = None,
    n_train_days: int = 0,
    hard_gate_min_train_days: int = 0,
) -> Dict[str, object]:
    room_key = room.strip().lower()
    checks: Dict[str, bool] = {}
    reasons: List[str] = []

    checks["unknown_rate"] = float(unknown_rate) <= float(unknown_rate_cap)
    if not checks["unknown_rate"]:
        reasons.append(f"unknown_rate>{unknown_rate_cap:.3f}")

    if room_key == "bedroom":
        gt_sleep_day = float(gt_daily.get("sleep_day", 0.0))
        pred_sleep_day = float(pred_daily.get("sleep_day", 0.0))
        checks["sleep_day_presence"] = not (gt_sleep_day > 0 and pred_sleep_day <= 0)
        if not checks["sleep_day_presence"]:
            reasons.append("sleep_day_missed")
    elif room_key == "bathroom":
        gt_shower_day = float(gt_daily.get("shower_day", 0.0))
        pred_shower_day = float(pred_daily.get("shower_day", 0.0))
        checks["shower_day_presence"] = not (gt_shower_day > 0 and pred_shower_day <= 0)
        if not checks["shower_day_presence"]:
            reasons.append("shower_day_missed")

    metric_values: Dict[str, float] = {
        "accuracy": float(cls.get("accuracy", 0.0)),
        "macro_f1": float(cls.get("macro_f1", 0.0)),
        "macro_precision": float(cls.get("macro_precision", 0.0)),
        "macro_recall": float(cls.get("macro_recall", 0.0)),
        "occupied_precision": float(cls.get("occupied_precision", 0.0)),
        "occupied_recall": float(cls.get("occupied_recall", 0.0)),
        "occupied_f1": float(cls.get("occupied_f1", 0.0)),
        "fragmentation_score": float(fragmentation_score),
    }
    for metric_name, floor in room_metric_floors.get(room_key, {}).items():
        metric_key = str(metric_name).strip().lower()
        if metric_key not in metric_values:
            checks[f"metric_{metric_key}_configured"] = False
            reasons.append(f"metric_not_available:{metric_key}")
            continue
        passed = float(metric_values[metric_key]) >= float(floor)
        checks[metric_key] = bool(passed)
        if not passed:
            reasons.append(f"{metric_key}_lt_{float(floor):.3f}")

    for label_name, floor in room_label_recall_floors.get(room_key, {}).items():
        label_key = str(label_name).strip().lower()
        train_support = (
            label_recall_train_day_support.get(label_key, {})
            if isinstance(label_recall_train_day_support, dict)
            else {}
        )
        if isinstance(train_support, dict) and not bool(train_support.get("all_days_supported", True)):
            continue
        stats = label_recall_summary.get(label_key)
        support = int(stats.get("support", 0.0)) if isinstance(stats, dict) else 0
        min_support = int(max(room_label_recall_min_supports.get(room_key, 0), 0))
        if support <= 0 or support < min_support:
            continue
        recall_val = float(stats.get("recall", 0.0)) if isinstance(stats, dict) else 0.0
        passed = recall_val >= float(floor)
        checks[f"recall_{label_key}"] = bool(passed)
        if not passed:
            reasons.append(f"recall_{label_key}_lt_{float(floor):.3f}")

    gate_pass = all(bool(v) for v in checks.values()) if checks else True
    eligible = int(n_train_days) >= int(max(hard_gate_min_train_days, 0))
    if not eligible:
        reasons = list(reasons)
        reasons.append("not_eligible_below_min_train_days")
    return {
        "pass": bool(gate_pass),
        "eligible": bool(eligible),
        "checks": checks,
        "reasons": reasons,
        "n_train_days": int(max(n_train_days, 0)),
        "hard_gate_min_train_days": int(max(hard_gate_min_train_days, 0)),
    }


def _safe_mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(len(arr)),
    }


def _config_hash(payload: Dict[str, object]) -> str:
    txt = json.dumps(payload, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(txt.encode('utf-8')).hexdigest()}"


def _daily_target_series(
    *,
    df: pd.DataFrame,
    label_col: str,
    target_key: str,
) -> List[float]:
    if df.empty or "timestamp" not in df.columns or label_col not in df.columns:
        return []
    daily_df = df.sort_values("timestamp").copy()
    out: List[float] = []
    for _, day_df in daily_df.groupby(daily_df["timestamp"].dt.date):
        episodes = labels_to_episodes(day_df, timestamp_col="timestamp", label_col=label_col)
        targets = build_daily_event_targets(episodes)
        out.append(float(targets.get(target_key, 0.0)))
    return out


def _mean_abs_error(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    n = min(len(a), len(b))
    if n <= 0:
        return float("inf")
    arr_a = np.asarray(list(a)[:n], dtype=float)
    arr_b = np.asarray(list(b)[:n], dtype=float)
    return float(np.mean(np.abs(arr_a - arr_b)))


def _duration_mae_from_labels(
    *,
    calib_df: pd.DataFrame,
    pred_labels: Sequence[object],
    target_key: str,
) -> float:
    """
    Compute daily duration MAE for a target key from predicted window labels.
    """
    if calib_df.empty or "timestamp" not in calib_df.columns or "activity" not in calib_df.columns:
        return float("inf")
    if not str(target_key).strip():
        return float("inf")
    y_pred = np.asarray(pred_labels, dtype=object)
    if len(y_pred) != len(calib_df):
        return float("inf")
    gt_daily = _daily_target_series(df=calib_df, label_col="activity", target_key=str(target_key))
    pred_df = calib_df[["timestamp"]].copy()
    pred_df["pred"] = y_pred
    pred_daily = _daily_target_series(df=pred_df, label_col="pred", target_key=str(target_key))
    return _mean_abs_error(gt_daily, pred_daily)


def _build_kitchen_stage_a_sample_weights(
    labels: Sequence[object],
    *,
    occupied_weight: float = 2.4,
    transition_band_windows: int = 18,
    transition_weight: float = 1.8,
    long_unoccupied_downweight: float = 0.7,
    long_unoccupied_windows: int = 240,
) -> np.ndarray:
    """
    Build stage-A sample weights for kitchen occupancy learning.

    Emphasizes occupied windows and boundary windows near occupied/unoccupied transitions,
    while mildly downweighting very long unoccupied stretches.
    """
    y = np.asarray([str(v).strip().lower() for v in labels], dtype=object)
    n = int(len(y))
    if n == 0:
        return np.asarray([], dtype=float)
    weights = np.ones(shape=(n,), dtype=float)
    occupied_mask = y != "unoccupied"
    weights[occupied_mask] = float(occupied_weight)

    # Emphasize boundary regions where occupancy mistakes are most costly.
    transition_idx: List[int] = []
    for i in range(1, n):
        if y[i] != y[i - 1]:
            transition_idx.append(i)
    band = int(max(0, transition_band_windows))
    for idx in transition_idx:
        start = max(0, idx - band)
        end = min(n, idx + band + 1)
        weights[start:end] = np.maximum(weights[start:end], float(transition_weight))

    # Downweight prolonged easy negatives.
    if int(long_unoccupied_windows) > 0:
        i = 0
        while i < n:
            if y[i] != "unoccupied":
                i += 1
                continue
            start = i
            while i < n and y[i] == "unoccupied":
                i += 1
            end = i
            run_len = end - start
            if run_len >= int(long_unoccupied_windows):
                mid_start = start + int(0.20 * run_len)
                mid_end = end - int(0.20 * run_len)
                if mid_end > mid_start:
                    weights[mid_start:mid_end] = np.minimum(
                        weights[mid_start:mid_end], float(long_unoccupied_downweight)
                    )

    return np.clip(weights, 0.25, 5.0).astype(float)


def _build_room_boundary_sample_weights(
    labels: Sequence[object],
    *,
    occupied_weight: float = 1.5,
    boundary_weight: float = 2.2,
    boundary_band_windows: int = 6,
    long_unoccupied_downweight: float = 0.85,
    long_unoccupied_windows: int = 180,
) -> np.ndarray:
    """
    Build stage-A sample weights that emphasize care-episode boundaries.

    This is a timeline-native proxy objective for occupancy learning:
    windows around care starts/ends get higher weight, while long easy
    unoccupied stretches are mildly downweighted.
    """
    y = np.asarray([str(v).strip().lower() for v in labels], dtype=object)
    n = int(len(y))
    if n == 0:
        return np.asarray([], dtype=float)

    weights = np.ones(shape=(n,), dtype=float)
    occupied_mask = y != "unoccupied"
    weights[occupied_mask] = float(occupied_weight)

    if n >= 2:
        try:
            boundary = build_boundary_targets(y)
            boundary_idx = np.where((boundary.start_flags + boundary.end_flags) > 0)[0]
        except Exception:
            boundary_idx = np.asarray([], dtype=int)
        band = int(max(0, boundary_band_windows))
        for idx in boundary_idx.tolist():
            start = max(0, int(idx) - band)
            end = min(n, int(idx) + band + 1)
            weights[start:end] = np.maximum(weights[start:end], float(boundary_weight))

    if int(long_unoccupied_windows) > 0:
        i = 0
        while i < n:
            if y[i] != "unoccupied":
                i += 1
                continue
            start = i
            while i < n and y[i] == "unoccupied":
                i += 1
            end = i
            run_len = end - start
            if run_len >= int(long_unoccupied_windows):
                mid_start = start + int(0.25 * run_len)
                mid_end = end - int(0.25 * run_len)
                if mid_end > mid_start:
                    weights[mid_start:mid_end] = np.minimum(
                        weights[mid_start:mid_end], float(long_unoccupied_downweight)
                    )

    return np.clip(weights, 0.25, 5.0).astype(float)


def _build_livingroom_passive_alignment_sample_weights(
    labels: Sequence[object],
    *,
    motion_values: Optional[Sequence[float]] = None,
    direct_positive_weight: float = 1.0,
    passive_positive_weight: float = 0.25,
    unoccupied_weight: float = 1.0,
    entry_exit_band_windows: int = 24,
    motion_direct_threshold: float = 0.55,
    long_unoccupied_downweight: float = 0.90,
    long_unoccupied_windows: int = 180,
) -> tuple[np.ndarray, Dict[str, object]]:
    """
    Build LivingRoom Stage-A weights that separate direct-evidence vs propagated passive positives.

    Label semantics are episode-based; this weighting keeps strong supervision on windows with
    direct evidence (entry/exit vicinity + motion-supported windows), while downweighting passive
    propagated interior windows so Stage-A can focus on evidence and let decoder carry persistence.
    """
    y = np.asarray([str(v).strip().lower() for v in labels], dtype=object)
    n = int(len(y))
    if n == 0:
        return np.asarray([], dtype=float), {
            "applied": False,
            "reason": "empty_labels",
            "n": 0,
        }

    pos_weight_direct = float(max(direct_positive_weight, 0.05))
    pos_weight_passive = float(max(passive_positive_weight, 0.01))
    neg_weight = float(max(unoccupied_weight, 0.05))
    boundary_band = int(max(entry_exit_band_windows, 0))
    motion_thr = float(max(motion_direct_threshold, 0.0))

    occupied_mask = np.asarray(y != "unoccupied", dtype=bool)
    direct_mask = np.zeros(shape=(n,), dtype=bool)

    # Entry/exit vicinity inside each occupied run is treated as direct evidence.
    i = 0
    run_count = 0
    while i < n:
        if not bool(occupied_mask[i]):
            i += 1
            continue
        start = i
        while i < n and bool(occupied_mask[i]):
            i += 1
        end = i
        run_count += 1
        left_end = min(start + boundary_band, end)
        right_start = max(start, end - boundary_band)
        direct_mask[start:left_end] = True
        direct_mask[right_start:end] = True

    motion_arr = None
    if motion_values is not None:
        arr = np.asarray(motion_values, dtype=float)
        if len(arr) < n:
            arr = np.resize(arr, n).astype(float)
        motion_arr = np.nan_to_num(arr[:n], nan=0.0, posinf=0.0, neginf=0.0)
        direct_mask = np.logical_or(direct_mask, np.logical_and(occupied_mask, motion_arr >= motion_thr))

    passive_mask = np.logical_and(occupied_mask, ~direct_mask)

    weights = np.full(shape=(n,), fill_value=neg_weight, dtype=float)
    weights[direct_mask] = float(pos_weight_direct)
    weights[passive_mask] = float(pos_weight_passive)

    if int(long_unoccupied_windows) > 0:
        i = 0
        while i < n:
            if occupied_mask[i]:
                i += 1
                continue
            start = i
            while i < n and not occupied_mask[i]:
                i += 1
            end = i
            run_len = end - start
            if run_len >= int(long_unoccupied_windows):
                mid_start = start + int(0.25 * run_len)
                mid_end = end - int(0.25 * run_len)
                if mid_end > mid_start:
                    weights[mid_start:mid_end] = np.minimum(
                        weights[mid_start:mid_end], float(max(long_unoccupied_downweight, 0.01))
                    )

    weights = np.clip(weights, 0.01, 20.0).astype(float)
    return weights, {
        "applied": True,
        "method": "livingroom_direct_vs_passive_alignment",
        "n": int(n),
        "occupied_windows": int(np.sum(occupied_mask)),
        "occupied_runs": int(run_count),
        "direct_windows": int(np.sum(direct_mask)),
        "passive_windows": int(np.sum(passive_mask)),
        "direct_fraction_of_occupied": float(
            np.sum(direct_mask) / max(int(np.sum(occupied_mask)), 1)
        ),
        "passive_fraction_of_occupied": float(
            np.sum(passive_mask) / max(int(np.sum(occupied_mask)), 1)
        ),
        "weights": {
            "direct_positive_weight": float(pos_weight_direct),
            "passive_positive_weight": float(pos_weight_passive),
            "unoccupied_weight": float(neg_weight),
            "long_unoccupied_downweight": float(max(long_unoccupied_downweight, 0.01)),
        },
        "entry_exit_band_windows": int(boundary_band),
        "motion_direct_threshold": float(motion_thr),
        "motion_available": bool(motion_arr is not None),
        "mean": float(np.mean(weights)),
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
    }


def _derive_boundary_probabilities_from_signals(
    *,
    y_pred: np.ndarray,
    occupancy_probs: np.ndarray,
    activity_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive start/end boundary probabilities from causal occupancy/activity changes.
    """
    n = int(len(occupancy_probs))
    if n == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    occ = np.asarray(occupancy_probs, dtype=float)
    occ_prev = np.roll(occ, 1)
    occ_prev[0] = occ[0]
    occ_rise = np.clip(occ - occ_prev, 0.0, 1.0)
    occ_drop = np.clip(occ_prev - occ, 0.0, 1.0)

    start_probs = np.maximum(occ_rise, ((occ >= 0.55) & (occ_prev < 0.45)).astype(float))
    end_probs = np.maximum(occ_drop, ((occ < 0.45) & (occ_prev >= 0.55)).astype(float))

    if activity_matrix.size > 0 and activity_matrix.shape[0] == n:
        top_idx = np.argmax(activity_matrix, axis=1)
        top_prev = np.roll(top_idx, 1)
        top_prev[0] = top_idx[0]
        label_change = np.asarray(top_idx != top_prev, dtype=bool)
        label_change[0] = False
        stable_occupied = np.logical_and(occ >= 0.5, occ_prev >= 0.5)
        change_mask = np.logical_and(label_change, stable_occupied)
        start_probs = np.where(change_mask, np.maximum(start_probs, 0.75), start_probs)
        end_probs = np.where(change_mask, np.maximum(end_probs, 0.75), end_probs)

    # Label-space fallback when activity probs are weak.
    yp = np.asarray([str(v).strip().lower() for v in y_pred], dtype=object)
    yp_prev = np.roll(yp, 1)
    yp_prev[0] = yp[0]
    pred_change = np.asarray(yp != yp_prev, dtype=bool)
    pred_change[0] = False
    pred_enter = np.logical_and(pred_change, np.logical_and(yp_prev == "unoccupied", yp != "unoccupied"))
    pred_exit = np.logical_and(pred_change, np.logical_and(yp_prev != "unoccupied", yp == "unoccupied"))
    start_probs = np.where(pred_enter, np.maximum(start_probs, 0.70), start_probs)
    end_probs = np.where(pred_exit, np.maximum(end_probs, 0.70), end_probs)

    return np.clip(start_probs, 0.0, 1.0), np.clip(end_probs, 0.0, 1.0)


def _apply_room_timeline_decoder_v2(
    *,
    room_key: str,
    timestamps: pd.Series,
    y_pred: np.ndarray,
    occupancy_probs: np.ndarray,
    activity_probs: Dict[str, np.ndarray],
) -> tuple[np.ndarray, Dict[str, object]]:
    """
    Apply timeline decoder v2 for Bedroom/LivingRoom with dynamic label space.
    """
    out = np.asarray(y_pred, dtype=object).copy()
    if room_key not in {"bedroom", "livingroom"}:
        return out, {"applied": False, "reason": "not_applicable", "room": room_key}

    n = int(len(out))
    if n == 0:
        return out, {"applied": False, "reason": "empty_prediction", "room": room_key}

    occ = np.asarray(occupancy_probs, dtype=float)
    if len(occ) != n:
        return out, {"applied": False, "reason": "occupancy_length_mismatch", "room": room_key}

    activity_labels = [
        str(label).strip().lower()
        for label in sorted(activity_probs.keys())
        if str(label).strip().lower() not in {"", "unoccupied", "unknown"}
    ]
    if not activity_labels:
        return out, {"applied": False, "reason": "no_activity_labels", "room": room_key}

    matrix = np.zeros(shape=(n, len(activity_labels)), dtype=float)
    for j, label in enumerate(activity_labels):
        arr = np.asarray(activity_probs.get(label, np.zeros(n)), dtype=float)
        if len(arr) < n:
            arr = np.resize(arr, n).astype(float)
        matrix[:, j] = np.clip(arr[:n], 0.0, 1.0)
    row_sum = np.sum(matrix, axis=1, keepdims=True)
    valid = row_sum[:, 0] > 1e-9
    matrix[valid] = matrix[valid] / row_sum[valid]
    matrix[~valid] = 1.0 / max(len(activity_labels), 1)

    start_probs, end_probs = _derive_boundary_probabilities_from_signals(
        y_pred=out,
        occupancy_probs=occ,
        activity_matrix=matrix,
    )

    policy = TimelineDecodePolicy(
        room_name=room_key,
        min_episode_windows=3 if room_key == "bedroom" else 2,
        max_gap_fill_windows=2,
        boundary_on_threshold=0.55,
        boundary_off_threshold=0.35,
        hysteresis_windows=2,
    )
    decoder = TimelineDecoderV2(policy=policy)
    ts = pd.to_datetime(timestamps, errors="coerce").ffill().bfill().to_numpy()
    episodes = decoder.decode(
        timestamps=ts,
        activity_probs=matrix,
        activity_labels=activity_labels,
        occupancy_probs=np.clip(occ, 0.0, 1.0),
        boundary_start_probs=start_probs,
        boundary_end_probs=end_probs,
    )

    decoded = np.full(shape=(n,), fill_value="unoccupied", dtype=object)
    for ep in episodes:
        start_idx = int(max(ep.start_idx, 0))
        end_idx = int(min(ep.end_idx, n - 1))
        if end_idx >= start_idx:
            decoded[start_idx : end_idx + 1] = str(ep.label).strip().lower()

    return decoded, {
        "applied": True,
        "reason": "timeline_decoder_v2",
        "room": room_key,
        "activity_labels": list(activity_labels),
        "episodes_decoded": int(len(episodes)),
        "occupied_windows_before": int(np.sum(out != "unoccupied")),
        "occupied_windows_after": int(np.sum(decoded != "unoccupied")),
        "boundary_start_peak": float(np.max(start_probs)) if len(start_probs) > 0 else 0.0,
        "boundary_end_peak": float(np.max(end_probs)) if len(end_probs) > 0 else 0.0,
    }


def _apply_room_segment_mode(
    *,
    room_key: str,
    timestamps: pd.Series,
    test_df: Optional[pd.DataFrame],
    y_pred: np.ndarray,
    occupancy_probs: np.ndarray,
    activity_probs: Dict[str, np.ndarray],
    occupancy_threshold: float,
    segment_min_duration_seconds: int,
    segment_gap_merge_seconds: int,
    segment_min_activity_prob: float = 0.35,
    enable_segment_learned_classifier: bool = False,
    segment_classifier_min_segments: int = 8,
    segment_classifier_confidence_floor: float = 0.55,
    segment_classifier_min_windows: int = 6,
) -> tuple[np.ndarray, Dict[str, object]]:
    out = np.asarray(y_pred, dtype=object).copy()
    room = str(room_key).strip().lower()
    if room not in {"bedroom", "livingroom"}:
        return out, {"applied": False, "reason": "not_applicable", "room": room}

    n = int(len(out))
    if n <= 0:
        return out, {"applied": False, "reason": "empty_prediction", "room": room}
    occ = np.asarray(occupancy_probs, dtype=float)
    if len(occ) != n:
        return out, {"applied": False, "reason": "occupancy_length_mismatch", "room": room}

    min_windows = int(max(round(float(segment_min_duration_seconds) / 10.0), 1))
    gap_windows = int(max(round(float(segment_gap_merge_seconds) / 10.0), 0))
    segments = propose_occupancy_segments(
        occ,
        threshold=float(np.clip(occupancy_threshold, 0.0, 1.0)),
        min_duration_windows=min_windows,
        gap_merge_windows=gap_windows,
    )
    if not segments:
        return (
            np.full(shape=(n,), fill_value="unoccupied", dtype=object),
            {
                "applied": True,
                "reason": "segment_mode_no_occupied_segments",
                "room": room,
                "segments_proposed": 0,
                "segments_labeled": 0,
            },
        )

    sensor_series: Dict[str, np.ndarray] = {}
    if isinstance(test_df, pd.DataFrame):
        for col in SENSOR_COLS:
            if col in test_df.columns:
                sensor_series[str(col)] = (
                    pd.to_numeric(test_df[col], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )

    segment_features = build_segment_features(
        segments=segments,
        timestamps=timestamps,
        occupancy_probs=occ,
        sensor_series=sensor_series,
        activity_probs=activity_probs,
    )
    labeled = classify_occupied_segments(
        segments=segments,
        activity_probs=activity_probs,
        segment_features=segment_features,
        min_activity_prob=float(np.clip(segment_min_activity_prob, 0.0, 1.0)),
        enable_learned_classifier=bool(enable_segment_learned_classifier),
        learned_classifier_min_segments=int(max(segment_classifier_min_segments, 2)),
        learned_classifier_confidence_floor=float(np.clip(segment_classifier_confidence_floor, 0.0, 1.0)),
        learned_classifier_min_windows=int(max(segment_classifier_min_windows, 1)),
    )
    projected = project_segments_to_window_labels(
        n_windows=n,
        labeled_segments=labeled,
        default_label="unoccupied",
    )
    learned_selected = int(sum(1 for row in labeled if bool(row.get("classifier_selected", False))))
    fallback_count = int(sum(1 for row in labeled if row.get("fallback_reason")))
    confidence_vals = np.asarray(
        [float(row.get("classifier_confidence", row.get("score", 0.0))) for row in labeled],
        dtype=float,
    )
    feature_columns = sorted(
        {
            str(k)
            for row in segment_features
            for k in row.keys()
            if str(k) not in {"start_idx", "end_idx"}
        }
    )
    return projected, {
        "applied": True,
        "reason": "segment_mode",
        "room": room,
        "segments_proposed": int(len(segments)),
        "segments_labeled": int(len(labeled)),
        "occupied_windows_before": int(np.sum(out != "unoccupied")),
        "occupied_windows_after": int(np.sum(np.asarray(projected, dtype=object) != "unoccupied")),
        "min_duration_windows": int(min_windows),
        "gap_merge_windows": int(gap_windows),
        "min_activity_prob": float(np.clip(segment_min_activity_prob, 0.0, 1.0)),
        "feature_rows": int(len(segment_features)),
        "feature_numeric_columns": int(len(feature_columns)),
        "classifier_modes_seen": sorted({str(row.get("classifier_mode", "heuristic")) for row in labeled}),
        "classifier_learned_selected_segments": int(learned_selected),
        "classifier_learned_fallback_segments": int(fallback_count),
        "classifier_confidence_mean": float(np.mean(confidence_vals)) if len(confidence_vals) > 0 else 0.0,
        "classifier_confidence_min": float(np.min(confidence_vals)) if len(confidence_vals) > 0 else 0.0,
        "classifier_confidence_max": float(np.max(confidence_vals)) if len(confidence_vals) > 0 else 0.0,
    }


def _tune_room_occupancy_threshold_by_duration_mae(
    *,
    model: EventFirstTwoStageModel,
    room_key: str,
    target_key: str,
    calib_df: pd.DataFrame,
    feat_cols: Sequence[str],
    room_label_threshold: Dict[str, float],
    critical_label_rescue_min_scores: Dict[str, Dict[str, float]],
    current_occupancy_threshold: float | None = None,
    objective_name: str | None = None,
    threshold_grid: np.ndarray | None = None,
    max_threshold_delta: float = 0.25,
    min_required_mae_improvement: float = 0.0,
    stability_penalty_weight: float = 0.35,
    threshold_delta_penalty_weight: float = 25.0,
    stage_a_group_ids: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Tune room occupancy threshold against daily duration MAE on calibration data.
    """
    room_key = str(room_key).strip().lower()
    if not room_key:
        return {"used": False, "reason": "missing_room_key"}
    if not str(target_key).strip():
        return {"used": False, "reason": "missing_target_key"}
    if calib_df.empty:
        return {"used": False, "reason": "empty_calibration_data"}
    if "timestamp" not in calib_df.columns or "activity" not in calib_df.columns:
        return {"used": False, "reason": "missing_required_columns"}
    if len(feat_cols) == 0:
        return {"used": False, "reason": "missing_feature_columns"}

    x_calib = calib_df[list(feat_cols)].to_numpy(dtype=float)
    y_true = calib_df["activity"].to_numpy(dtype=object)
    if len(x_calib) == 0 or len(y_true) == 0:
        return {"used": False, "reason": "empty_arrays"}

    gt_daily = _daily_target_series(df=calib_df, label_col="activity", target_key=str(target_key))
    if len(gt_daily) == 0:
        return {"used": False, "reason": "no_daily_targets"}

    grid = (
        np.asarray(threshold_grid, dtype=float)
        if threshold_grid is not None
        else np.linspace(0.20, 0.80, num=61, dtype=float)
    )
    grid = np.clip(grid, 0.0, 1.0)
    activity_probs = model.predict_activity_proba(x_calib)

    if current_occupancy_threshold is None:
        current_occ = float(model.get_operating_points().get("occupancy_threshold", 0.35))
    else:
        current_occ = float(current_occupancy_threshold)
    if stage_a_group_ids is None:
        baseline_pred = model.predict(
            x_calib,
            occupancy_threshold=current_occ,
            label_thresholds=room_label_threshold or None,
        )
    else:
        try:
            baseline_pred = model.predict(
                x_calib,
                occupancy_threshold=current_occ,
                label_thresholds=room_label_threshold or None,
                stage_a_group_ids=stage_a_group_ids,
            )
        except TypeError as exc:
            if "stage_a_group_ids" not in str(exc):
                raise
            baseline_pred = model.predict(
                x_calib,
                occupancy_threshold=current_occ,
                label_thresholds=room_label_threshold or None,
            )
    baseline_pred, _ = _apply_critical_label_rescue(
        room_key=room_key,
        y_pred=baseline_pred,
        activity_probs=activity_probs,
        rescue_min_scores=critical_label_rescue_min_scores,
    )
    baseline_df = calib_df[["timestamp"]].copy()
    baseline_df["pred"] = baseline_pred
    baseline_daily = _daily_target_series(df=baseline_df, label_col="pred", target_key=str(target_key))
    baseline_abs = np.abs(np.asarray(gt_daily, dtype=float) - np.asarray(baseline_daily, dtype=float))
    baseline_mae = float(np.mean(baseline_abs))
    baseline_std = float(np.std(baseline_abs))

    baseline_cls = _classification_metrics(y_true, np.asarray(baseline_pred, dtype=object))
    best_threshold = float(current_occ)
    best_mae = float(baseline_mae)
    best_std = float(baseline_std)
    best_cls = dict(baseline_cls)
    best_objective = float(best_mae + (stability_penalty_weight * best_std))
    best_score = (
        -best_objective,
        -abs(best_threshold - current_occ),
        float(best_cls["macro_f1"]),
        float(best_cls["accuracy"]),
    )

    for t in grid:
        if abs(float(t) - current_occ) > float(max_threshold_delta):
            continue
        if stage_a_group_ids is None:
            y_pred = model.predict(
                x_calib,
                occupancy_threshold=float(t),
                label_thresholds=room_label_threshold or None,
            )
        else:
            try:
                y_pred = model.predict(
                    x_calib,
                    occupancy_threshold=float(t),
                    label_thresholds=room_label_threshold or None,
                    stage_a_group_ids=stage_a_group_ids,
                )
            except TypeError as exc:
                if "stage_a_group_ids" not in str(exc):
                    raise
                y_pred = model.predict(
                    x_calib,
                    occupancy_threshold=float(t),
                    label_thresholds=room_label_threshold or None,
                )
        y_pred, _ = _apply_critical_label_rescue(
            room_key=room_key,
            y_pred=y_pred,
            activity_probs=activity_probs,
            rescue_min_scores=critical_label_rescue_min_scores,
        )
        pred_df = calib_df[["timestamp"]].copy()
        pred_df["pred"] = y_pred
        pred_daily = _daily_target_series(df=pred_df, label_col="pred", target_key=str(target_key))
        abs_errors = np.abs(np.asarray(gt_daily, dtype=float) - np.asarray(pred_daily, dtype=float))
        mae = float(np.mean(abs_errors))
        std = float(np.std(abs_errors))
        cls = _classification_metrics(y_true, np.asarray(y_pred, dtype=object))
        objective = (
            mae
            + (stability_penalty_weight * std)
            + (threshold_delta_penalty_weight * abs(float(t) - current_occ))
        )
        score = (-objective, -abs(float(t) - current_occ), float(cls["macro_f1"]), float(cls["accuracy"]))
        if score > best_score:
            best_score = score
            best_threshold = float(t)
            best_mae = float(mae)
            best_std = float(std)
            best_cls = dict(cls)

    adopted = bool(best_mae <= (baseline_mae - float(min_required_mae_improvement)))
    selected_threshold = float(best_threshold if adopted else current_occ)
    selected_mae = float(best_mae if adopted else baseline_mae)
    selected_std = float(best_std if adopted else baseline_std)
    selected_method = f"{room_key}_daily_mae_tuned" if adopted else "baseline_kept"
    objective = str(objective_name or f"{target_key}_mae")

    return {
        "used": True,
        "objective": objective,
        "room": room_key,
        "target_key": str(target_key),
        "baseline_occupancy_threshold": float(current_occ),
        "selected_occupancy_threshold": float(selected_threshold),
        "baseline_daily_mae_minutes": float(baseline_mae),
        "baseline_daily_abs_error_std_minutes": float(baseline_std),
        "selected_daily_mae_minutes": float(selected_mae),
        "selected_daily_abs_error_std_minutes": float(selected_std),
        "delta_daily_mae_minutes": float(selected_mae - baseline_mae),
        "adopted": adopted,
        "selection_method": selected_method,
        "max_threshold_delta": float(max_threshold_delta),
        "min_required_mae_improvement": float(min_required_mae_improvement),
        "stability_penalty_weight": float(stability_penalty_weight),
        "threshold_delta_penalty_weight": float(threshold_delta_penalty_weight),
        "calib_days": int(len(gt_daily)),
        "calib_rows": int(len(x_calib)),
        "baseline_macro_f1": float(baseline_cls["macro_f1"]),
        "baseline_accuracy": float(baseline_cls["accuracy"]),
        "selected_macro_f1": float(best_cls["macro_f1"]) if adopted else float(baseline_cls["macro_f1"]),
        "selected_accuracy": float(best_cls["accuracy"]) if adopted else float(baseline_cls["accuracy"]),
    }


def _tune_kitchen_occupancy_threshold_by_mae(
    *,
    model: EventFirstTwoStageModel,
    calib_df: pd.DataFrame,
    feat_cols: Sequence[str],
    room_label_threshold: Dict[str, float],
    critical_label_rescue_min_scores: Dict[str, Dict[str, float]],
    threshold_grid: np.ndarray | None = None,
    max_threshold_delta: float = 0.25,
    min_required_mae_improvement: float = 0.0,
    stability_penalty_weight: float = 0.35,
    threshold_delta_penalty_weight: float = 25.0,
    stage_a_group_ids: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    return _tune_room_occupancy_threshold_by_duration_mae(
        model=model,
        room_key="kitchen",
        target_key="kitchen_use_minutes",
        objective_name="kitchen_use_minutes_mae",
        calib_df=calib_df,
        feat_cols=feat_cols,
        room_label_threshold=room_label_threshold,
        critical_label_rescue_min_scores=critical_label_rescue_min_scores,
        threshold_grid=threshold_grid,
        max_threshold_delta=max_threshold_delta,
        min_required_mae_improvement=min_required_mae_improvement,
        stability_penalty_weight=stability_penalty_weight,
        threshold_delta_penalty_weight=threshold_delta_penalty_weight,
        stage_a_group_ids=stage_a_group_ids,
    )


def _tune_room_occupancy_threshold_for_hard_gate(
    *,
    model: EventFirstTwoStageModel,
    room_key: str,
    calib_df: pd.DataFrame,
    feat_cols: Sequence[str],
    room_label_threshold: Dict[str, float],
    critical_label_rescue_min_scores: Dict[str, Dict[str, float]],
    current_occupancy_threshold: float | None = None,
    threshold_grid: np.ndarray | None = None,
    max_threshold_delta: float = 0.22,
    recall_floor: float = 0.50,
    fragmentation_floor: float = 0.00,
    min_required_f1_improvement: float = 0.00,
    recall_weight: float = 0.75,
    fragmentation_weight: float = 0.30,
    recall_floor_penalty_weight: float = 2.0,
    threshold_delta_penalty_weight: float = 0.08,
    max_allowed_duration_mae_increase_minutes: float = 30.0,
    max_allowed_duration_mae_ratio: float = 1.25,
    duration_guardrail_penalty_weight: float = 3.0,
    enable_room_occupancy_decoder: bool = True,
    stage_a_group_ids: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Tune occupancy threshold for hard-gate metrics (occupied F1/recall/fragmentation).
    """
    room_key = str(room_key).strip().lower()
    if room_key not in {"bedroom", "livingroom"}:
        return {"used": False, "reason": "room_not_supported"}
    if calib_df.empty:
        return {"used": False, "reason": "empty_calibration_data"}
    if "timestamp" not in calib_df.columns or "activity" not in calib_df.columns:
        return {"used": False, "reason": "missing_required_columns"}
    if len(feat_cols) == 0:
        return {"used": False, "reason": "missing_feature_columns"}

    x_calib = calib_df[list(feat_cols)].to_numpy(dtype=float)
    y_true = calib_df["activity"].to_numpy(dtype=object)
    if len(x_calib) == 0 or len(y_true) == 0:
        return {"used": False, "reason": "empty_arrays"}

    grid = (
        np.asarray(threshold_grid, dtype=float)
        if threshold_grid is not None
        else np.linspace(0.18, 0.82, num=65, dtype=float)
    )
    grid = np.clip(grid, 0.0, 1.0)

    if current_occupancy_threshold is None:
        current_occ = float(model.get_operating_points().get("occupancy_threshold", 0.35))
    else:
        current_occ = float(current_occupancy_threshold)

    activity_probs = model.predict_activity_proba(x_calib)
    if stage_a_group_ids is None:
        occupancy_probs = model.predict_occupancy_proba(x_calib)
    else:
        try:
            occupancy_probs = model.predict_occupancy_proba(
                x_calib,
                stage_a_group_ids=stage_a_group_ids,
            )
        except TypeError as exc:
            if "stage_a_group_ids" not in str(exc):
                raise
            occupancy_probs = model.predict_occupancy_proba(x_calib)
    duration_target_key = ROOM_DURATION_KEY.get(room_key, "")

    def _predict_with_threshold(threshold: float) -> tuple[np.ndarray, Dict[str, object]]:
        if stage_a_group_ids is None:
            y_pred = model.predict(
                x_calib,
                occupancy_threshold=float(threshold),
                label_thresholds=room_label_threshold or None,
            )
        else:
            try:
                y_pred = model.predict(
                    x_calib,
                    occupancy_threshold=float(threshold),
                    label_thresholds=room_label_threshold or None,
                    stage_a_group_ids=stage_a_group_ids,
                )
            except TypeError as exc:
                if "stage_a_group_ids" not in str(exc):
                    raise
                y_pred = model.predict(
                    x_calib,
                    occupancy_threshold=float(threshold),
                    label_thresholds=room_label_threshold or None,
                )
        y_pred, rescue_debug = _apply_critical_label_rescue(
            room_key=room_key,
            y_pred=y_pred,
            activity_probs=activity_probs,
            rescue_min_scores=critical_label_rescue_min_scores,
        )
        occ_decoder_debug: Dict[str, object] = {"applied": False, "reason": "disabled"}
        if bool(enable_room_occupancy_decoder):
            y_pred, occ_decoder_debug = _apply_room_occupancy_temporal_decoder(
                room_key=room_key,
                y_pred=y_pred,
                occupancy_probs=occupancy_probs,
                activity_probs=activity_probs,
            )
        dbg = {
            "critical_label_rescue": rescue_debug,
            "room_occupancy_temporal_decoder": occ_decoder_debug,
        }
        return np.asarray(y_pred, dtype=object), dbg

    def _score_prediction(threshold: float, y_pred: np.ndarray) -> tuple[float, Dict[str, float]]:
        cls = _classification_metrics(y_true, y_pred)
        frag = _care_fragmentation_score(
            y_true,
            y_pred,
            min_run_windows=6,
            gap_fill_windows=3,
        )
        occ_f1 = float(cls.get("occupied_f1", 0.0))
        occ_recall = float(cls.get("occupied_recall", 0.0))
        floor_deficit = max(float(recall_floor) - occ_recall, 0.0)
        frag_deficit = max(float(fragmentation_floor) - float(frag), 0.0)
        duration_mae = float("inf")
        if str(duration_target_key).strip():
            duration_mae = _duration_mae_from_labels(
                calib_df=calib_df,
                pred_labels=y_pred,
                target_key=str(duration_target_key),
            )
            if not np.isfinite(duration_mae):
                duration_mae = float("inf")

        # KPI-safe bound: do not allow large duration MAE blow-up while chasing hard-gate recall.
        allowed_mae = float("inf")
        guardrail_pass = True
        if np.isfinite(duration_mae) and np.isfinite(baseline_duration_mae):
            allowed_abs = float(baseline_duration_mae + float(max_allowed_duration_mae_increase_minutes))
            if float(baseline_duration_mae) > 1e-6:
                allowed_ratio = float(baseline_duration_mae * float(max_allowed_duration_mae_ratio))
                allowed_mae = float(min(allowed_abs, allowed_ratio))
            else:
                allowed_mae = float(allowed_abs)
            guardrail_pass = bool(duration_mae <= (allowed_mae + 1e-9))
            duration_penalty = (
                0.0
                if guardrail_pass
                else float(duration_guardrail_penalty_weight) * float(max(duration_mae - allowed_mae, 0.0))
            )
        else:
            duration_penalty = 0.0

        recall_ok = bool(occ_recall >= float(recall_floor))
        frag_ok = bool(float(frag) >= float(fragmentation_floor))
        constraints_pass = bool(recall_ok and frag_ok and guardrail_pass)

        objective = (
            occ_f1
            + (float(recall_weight) * occ_recall)
            + (float(fragmentation_weight) * float(frag))
            - (float(recall_floor_penalty_weight) * float(floor_deficit))
            - (float(recall_floor_penalty_weight) * float(frag_deficit))
            - (float(threshold_delta_penalty_weight) * abs(float(threshold) - current_occ))
            - float(duration_penalty)
        )
        return float(objective), {
            "occupied_f1": float(occ_f1),
            "occupied_recall": float(occ_recall),
            "fragmentation_score": float(frag),
            "recall_floor_pass": bool(recall_ok),
            "fragmentation_floor_pass": bool(frag_ok),
            "constraints_pass": bool(constraints_pass),
            "duration_mae_minutes": float(duration_mae) if np.isfinite(duration_mae) else float("inf"),
            "duration_guardrail_pass": bool(guardrail_pass),
            "duration_guardrail_allowed_mae_minutes": float(allowed_mae) if np.isfinite(allowed_mae) else float("inf"),
            "duration_guardrail_penalty": float(duration_penalty),
            "objective": float(objective),
        }

    baseline_pred, baseline_debug = _predict_with_threshold(current_occ)
    baseline_duration_mae = _duration_mae_from_labels(
        calib_df=calib_df,
        pred_labels=baseline_pred,
        target_key=str(duration_target_key),
    ) if str(duration_target_key).strip() else float("nan")
    baseline_obj, baseline_stats = _score_prediction(current_occ, baseline_pred)

    best_threshold = float(current_occ)
    best_obj = float(baseline_obj)
    best_stats = dict(baseline_stats)
    best_debug = dict(baseline_debug)
    best_score = (
        float(best_obj),
        float(best_stats["occupied_recall"]),
        float(best_stats["occupied_f1"]),
        float(best_stats["fragmentation_score"]),
        -abs(float(best_threshold) - current_occ),
    )

    for t in grid:
        if abs(float(t) - current_occ) > float(max_threshold_delta):
            continue
        y_pred, dbg = _predict_with_threshold(float(t))
        objective, stats = _score_prediction(float(t), y_pred)
        if not bool(stats.get("constraints_pass", False)):
            continue
        score = (
            float(objective),
            float(stats["occupied_recall"]),
            float(stats["occupied_f1"]),
            float(stats["fragmentation_score"]),
            -abs(float(t) - current_occ),
        )
        if score > best_score:
            best_score = score
            best_threshold = float(t)
            best_obj = float(objective)
            best_stats = dict(stats)
            best_debug = dict(dbg)

    baseline_f1 = float(baseline_stats["occupied_f1"])
    best_f1 = float(best_stats["occupied_f1"])
    adopted = bool(
        (best_obj > baseline_obj + 1e-6)
        and (best_f1 >= baseline_f1 + float(min_required_f1_improvement))
        and bool(best_stats.get("constraints_pass", False))
    )
    selected_threshold = float(best_threshold if adopted else current_occ)
    selected_stats = dict(best_stats if adopted else baseline_stats)
    selected_debug = dict(best_debug if adopted else baseline_debug)

    return {
        "used": True,
        "objective": "hard_gate_occupied_f1_recall_fragmentation",
        "room": room_key,
        "baseline_occupancy_threshold": float(current_occ),
        "selected_occupancy_threshold": float(selected_threshold),
        "baseline_objective": float(baseline_obj),
        "selected_objective": float(best_obj if adopted else baseline_obj),
        "delta_objective": float((best_obj - baseline_obj) if adopted else 0.0),
        "baseline_occupied_f1": float(baseline_stats["occupied_f1"]),
        "selected_occupied_f1": float(selected_stats["occupied_f1"]),
        "baseline_occupied_recall": float(baseline_stats["occupied_recall"]),
        "selected_occupied_recall": float(selected_stats["occupied_recall"]),
        "baseline_fragmentation_score": float(baseline_stats["fragmentation_score"]),
        "selected_fragmentation_score": float(selected_stats["fragmentation_score"]),
        "duration_target_key": str(duration_target_key),
        "baseline_duration_mae_minutes": (
            float(baseline_stats.get("duration_mae_minutes", float("nan")))
            if np.isfinite(float(baseline_stats.get("duration_mae_minutes", float("nan"))))
            else None
        ),
        "selected_duration_mae_minutes": (
            float(selected_stats.get("duration_mae_minutes", float("nan")))
            if np.isfinite(float(selected_stats.get("duration_mae_minutes", float("nan"))))
            else None
        ),
        "duration_guardrail_pass": bool(selected_stats.get("duration_guardrail_pass", True)),
        "duration_guardrail_allowed_mae_minutes": (
            float(selected_stats.get("duration_guardrail_allowed_mae_minutes", float("nan")))
            if np.isfinite(float(selected_stats.get("duration_guardrail_allowed_mae_minutes", float("nan"))))
            else None
        ),
        "recall_floor_pass": bool(selected_stats.get("recall_floor_pass", False)),
        "fragmentation_floor_pass": bool(selected_stats.get("fragmentation_floor_pass", False)),
        "constraints_pass": bool(selected_stats.get("constraints_pass", False)),
        "recall_floor": float(recall_floor),
        "fragmentation_floor": float(fragmentation_floor),
        "adopted": adopted,
        "selection_method": f"{room_key}_hard_gate_tuned" if adopted else "baseline_kept",
        "max_threshold_delta": float(max_threshold_delta),
        "min_required_f1_improvement": float(min_required_f1_improvement),
        "recall_weight": float(recall_weight),
        "fragmentation_weight": float(fragmentation_weight),
        "recall_floor_penalty_weight": float(recall_floor_penalty_weight),
        "threshold_delta_penalty_weight": float(threshold_delta_penalty_weight),
        "max_allowed_duration_mae_increase_minutes": float(max_allowed_duration_mae_increase_minutes),
        "max_allowed_duration_mae_ratio": float(max_allowed_duration_mae_ratio),
        "duration_guardrail_penalty_weight": float(duration_guardrail_penalty_weight),
        "calib_rows": int(len(x_calib)),
        "applied_debug": selected_debug,
    }


def _predict_labels_for_df(
    *,
    model: EventFirstTwoStageModel,
    room_key: str,
    day_df: pd.DataFrame,
    feat_cols: Sequence[str],
    room_occ_threshold: float,
    room_label_threshold: Dict[str, float],
    critical_label_rescue_min_scores: Dict[str, Dict[str, float]],
    enable_kitchen_temporal_decoder: bool,
    enable_bedroom_livingroom_occupancy_decoder: bool,
    stage_a_group_seconds: Optional[int] = None,
) -> np.ndarray:
    x = day_df[list(feat_cols)].to_numpy(dtype=float)
    stage_a_group_ids: Optional[np.ndarray] = None
    if (
        stage_a_group_seconds is not None
        and int(stage_a_group_seconds) > 10
        and room_key in {"bedroom", "livingroom"}
    ):
        stage_a_group_ids = _build_stage_a_group_ids_from_timestamps(
            timestamps=day_df["timestamp"],
            resolution_seconds=int(stage_a_group_seconds),
        )
    if stage_a_group_ids is None:
        occupancy_probs = model.predict_occupancy_proba(x)
        y_pred = model.predict(
            x,
            occupancy_threshold=room_occ_threshold,
            label_thresholds=room_label_threshold or None,
        )
    else:
        try:
            occupancy_probs = model.predict_occupancy_proba(
                x,
                stage_a_group_ids=stage_a_group_ids,
            )
            y_pred = model.predict(
                x,
                occupancy_threshold=room_occ_threshold,
                label_thresholds=room_label_threshold or None,
                stage_a_group_ids=stage_a_group_ids,
            )
        except TypeError as exc:
            if "stage_a_group_ids" not in str(exc):
                raise
            occupancy_probs = model.predict_occupancy_proba(x)
            y_pred = model.predict(
                x,
                occupancy_threshold=room_occ_threshold,
                label_thresholds=room_label_threshold or None,
            )
    activity_probs = model.predict_activity_proba(x)
    y_pred, _ = _apply_critical_label_rescue(
        room_key=room_key,
        y_pred=y_pred,
        activity_probs=activity_probs,
        rescue_min_scores=critical_label_rescue_min_scores,
    )
    y_pred, _ = _apply_bathroom_shower_fallback(
        room_key=room_key,
        y_pred=y_pred,
        timestamps=day_df["timestamp"],
        test_df=day_df,
        activity_probs=activity_probs,
    )
    if room_key == "kitchen" and bool(enable_kitchen_temporal_decoder):
        y_pred, _ = _apply_kitchen_temporal_decoder(
            y_pred=y_pred,
            occupancy_probs=occupancy_probs,
            activity_probs=activity_probs,
        )
    if room_key in {"bedroom", "livingroom"} and bool(enable_bedroom_livingroom_occupancy_decoder):
        y_pred, _ = _apply_room_occupancy_temporal_decoder(
            room_key=room_key,
            y_pred=y_pred,
            occupancy_probs=occupancy_probs,
            activity_probs=activity_probs,
        )
    return np.asarray(y_pred, dtype=object)


def _adaptive_room_occupancy_threshold(
    *,
    room_key: str,
    base_threshold: float,
    occupancy_probs: np.ndarray,
    y_fit_labels: np.ndarray,
) -> tuple[float, Dict[str, object]]:
    """
    Derive a split-regime occupancy threshold from train occupancy prior.
    Uses only train labels and test probabilities (no test labels).
    """
    room = str(room_key).strip().lower()
    if room not in {"bedroom", "livingroom"}:
        return float(base_threshold), {"applied": False, "reason": "room_not_in_scope"}

    probs = np.asarray(occupancy_probs, dtype=float)
    if len(probs) == 0:
        return float(base_threshold), {"applied": False, "reason": "empty_probs"}

    y_fit = np.asarray([str(v).strip().lower() for v in y_fit_labels], dtype=object)
    if len(y_fit) == 0:
        return float(base_threshold), {"applied": False, "reason": "empty_fit_labels"}

    train_occ_rate = float(np.mean(y_fit != "unoccupied"))
    if room == "bedroom":
        target_rate = float(min(max(train_occ_rate, 0.15), 0.80))
        max_delta = 0.12
    else:
        target_rate = float(min(max(train_occ_rate, 0.08), 0.60))
        max_delta = 0.15

    def _otsu_probability_threshold(values: np.ndarray, bins: int = 64) -> float:
        probs = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
        if len(probs) <= 1:
            return float(np.median(probs)) if len(probs) > 0 else 0.5
        hist, edges = np.histogram(probs, bins=int(max(bins, 8)), range=(0.0, 1.0))
        total = float(np.sum(hist))
        if total <= 0.0:
            return float(np.median(probs))
        centers = (edges[:-1] + edges[1:]) * 0.5
        w_bg = 0.0
        sum_bg = 0.0
        sum_all = float(np.sum(centers * hist))
        best_var = -1.0
        best_thr = float(np.median(probs))
        for i in range(len(hist)):
            w_bg += float(hist[i])
            if w_bg <= 0.0:
                continue
            w_fg = total - w_bg
            if w_fg <= 0.0:
                break
            sum_bg += float(centers[i] * hist[i])
            mean_bg = sum_bg / w_bg
            mean_fg = (sum_all - sum_bg) / w_fg
            between_var = w_bg * w_fg * ((mean_bg - mean_fg) ** 2)
            if between_var > best_var:
                best_var = between_var
                best_thr = float(centers[i])
        return float(min(max(best_thr, 0.05), 0.95))

    q = float(min(max(1.0 - target_rate, 0.0), 1.0))
    quantile_thr = float(np.quantile(np.clip(probs, 0.0, 1.0), q))
    blended = float((0.6 * float(base_threshold)) + (0.4 * quantile_thr))
    otsu_thr = _otsu_probability_threshold(probs)
    prob_mean = float(np.mean(probs))
    prob_std = float(np.std(probs))
    regime_adjusted = float(blended)
    if room == "livingroom":
        if prob_mean < 0.40:
            regime_adjusted = float(max(blended, otsu_thr + 0.04))
        elif prob_mean > 0.52:
            regime_adjusted = float(min(blended, max(otsu_thr - 0.03, 0.05)))
        else:
            regime_adjusted = float((0.55 * blended) + (0.45 * otsu_thr))
    lower = float(max(float(base_threshold) - max_delta, 0.05))
    upper = float(min(float(base_threshold) + max_delta, 0.95))
    adaptive = float(min(max(regime_adjusted, lower), upper))
    return adaptive, {
        "applied": True,
        "room": room,
        "base_threshold": float(base_threshold),
        "train_occupied_rate": train_occ_rate,
        "target_occupied_rate": target_rate,
        "quantile_threshold": quantile_thr,
        "otsu_threshold": float(otsu_thr),
        "prob_mean": prob_mean,
        "prob_std": prob_std,
        "regime_adjusted_threshold": regime_adjusted,
        "adaptive_threshold": adaptive,
    }


def _fit_affine_duration_calibrator(
    pred_values: Sequence[float],
    gt_values: Sequence[float],
    sample_weights: Sequence[float] | None = None,
) -> Dict[str, float]:
    x = np.asarray(pred_values, dtype=float)
    y = np.asarray(gt_values, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return {"enabled": False, "slope": 1.0, "intercept": 0.0}
    if len(x) == 1:
        # Single-pair calibration is high-variance; use multiplicative correction
        # instead of a large free intercept.
        denom = float(max(abs(x[0]), 1e-6))
        ratio = float(y[0]) / denom
        slope = float(min(max(ratio, 0.25), 1.75))
        return {"enabled": True, "slope": slope, "intercept": 0.0, "method": "single_point_ratio"}

    w = np.ones(shape=(len(x),), dtype=float)
    if sample_weights is not None:
        w_raw = np.asarray(sample_weights, dtype=float)
        if len(w_raw) == len(x):
            w = np.clip(w_raw, 1e-6, np.inf)
    w = w / max(float(np.sum(w)), 1e-9)

    x_mean = float(np.sum(w * x))
    y_mean = float(np.sum(w * y))
    var_x = float(np.sum(w * np.square(x - x_mean)))
    if var_x <= 1e-6:
        slope = 1.0
    else:
        cov_xy = float(np.sum(w * (x - x_mean) * (y - y_mean)))
        slope = cov_xy / var_x
    slope = float(min(max(slope, 0.25), 1.75))
    intercept = float(y_mean - slope * x_mean)
    return {"enabled": True, "slope": slope, "intercept": intercept}


def _apply_duration_calibration(
    *,
    room_key: str,
    pred_daily: Dict[str, float],
    calibrator: Dict[str, float],
) -> Dict[str, float]:
    out = dict(pred_daily)
    if not bool(calibrator.get("enabled", False)):
        return out
    target_key = ROOM_DURATION_KEY.get(room_key)
    if not target_key:
        return out
    raw_val = float(out.get(target_key, 0.0))
    corrected = float(calibrator.get("slope", 1.0)) * raw_val + float(calibrator.get("intercept", 0.0))
    corrected = float(min(max(corrected, 0.0), 1440.0))
    out[target_key] = corrected
    day_key = target_key.replace("_minutes", "_day")
    if day_key in out:
        out[day_key] = 1.0 if corrected > 0.0 else 0.0
    return out


def _calibrate_duration_from_train(
    pred_values: Sequence[float],
    gt_values: Sequence[float],
    raw_value: float,
    room_key: str = "",
    enable_kitchen_robust_duration_calibration: bool = False,
) -> Dict[str, float | str | bool]:
    x = np.asarray(pred_values, dtype=float)
    y = np.asarray(gt_values, dtype=float)
    raw = float(raw_value)
    if len(x) == 0 or len(y) == 0:
        return {"enabled": False, "method": "none", "corrected_value": raw, "train_pairs": 0}

    def _clip_minutes(arr: np.ndarray) -> np.ndarray:
        return np.clip(arr.astype(float), 0.0, 1440.0)

    def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    room = str(room_key or "").strip().lower()
    # Entrance is sparse/episodic; one-pair calibration is unstable and
    # can inflate false positives on next day.
    if room == "entrance" and len(x) < 2:
        return {
            "enabled": False,
            "method": "low_support_skip",
            "corrected_value": raw,
            "train_pairs": int(len(x)),
        }
    recency_weights = np.linspace(1.0, 2.0, num=len(x), dtype=float) if len(x) > 1 else np.ones((len(x),), dtype=float)
    affine = _fit_affine_duration_calibrator(x, y, sample_weights=recency_weights if room == "kitchen" else None)
    if not bool(affine.get("enabled", False)):
        return {"enabled": False, "method": "none", "corrected_value": raw, "train_pairs": int(len(x))}

    affine_train_pred = _clip_minutes(
        float(affine.get("slope", 1.0)) * x + float(affine.get("intercept", 0.0))
    )
    affine_train_mae = _mae(y, affine_train_pred)
    affine_corrected = float(affine.get("slope", 1.0)) * raw + float(affine.get("intercept", 0.0))
    affine_corrected = float(min(max(affine_corrected, 0.0), 1440.0))
    best = {
        "enabled": True,
        "method": "affine",
        "corrected_value": affine_corrected,
        "slope": float(affine.get("slope", 1.0)),
        "intercept": float(affine.get("intercept", 0.0)),
        "train_pairs": int(len(x)),
        "train_mae": affine_train_mae,
    }

    if room == "kitchen" and bool(enable_kitchen_robust_duration_calibration):
        # Kitchen is highly variable across days; use bounded shrinkage toward raw to reduce overfit.
        x_std = float(np.std(x))
        y_std = float(np.std(y))
        corr = 0.0
        if x_std > 1e-6 and y_std > 1e-6:
            corr = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        corr = float(min(max(corr, 0.0), 1.0))
        support = float(min(max((len(x) - 2) / 6.0, 0.0), 1.0))
        shrink = float(0.15 + (0.65 * corr * support))
        shrunk = float(raw + (shrink * (affine_corrected - raw)))
        lo = float(np.min(y))
        hi = float(np.max(y))
        span = float(max(hi - lo, 30.0))
        margin = float(max(25.0, 0.15 * span))
        bounded = float(min(max(shrunk, lo - margin), hi + margin))
        bounded = float(min(max(bounded, 0.0), 1440.0))
        return {
            "enabled": True,
            "method": "kitchen_affine_shrunk",
            "corrected_value": bounded,
            "slope": float(affine.get("slope", 1.0)),
            "intercept": float(affine.get("intercept", 0.0)),
            "shrink_factor": shrink,
            "corr": corr,
            "train_pairs": int(len(x)),
            "train_mae": affine_train_mae,
        }

    if len(x) >= 4 and len(np.unique(x)) >= 3:
        try:
            iso = IsotonicRegression(y_min=0.0, y_max=1440.0, out_of_bounds="clip")
            iso.fit(x, y)
            iso_train_pred = _clip_minutes(iso.predict(x))
            iso_train_mae = _mae(y, iso_train_pred)
            if iso_train_mae < float(best["train_mae"]):
                corrected = float(iso.predict(np.asarray([raw], dtype=float))[0])
                best = {
                    "enabled": True,
                    "method": "isotonic",
                    "corrected_value": float(min(max(corrected, 0.0), 1440.0)),
                    "train_pairs": int(len(x)),
                    "train_mae": iso_train_mae,
                }
        except Exception:
            pass

    return best


def _parse_room_thresholds(raw: str | None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not raw:
        return out
    for token in str(raw).split(","):
        item = token.strip()
        if not item:
            continue
        if "=" not in item:
            continue
        key_raw, value_raw = item.split("=", 1)
        room = key_raw.strip().lower()
        if not room:
            continue
        try:
            value = float(value_raw)
        except (TypeError, ValueError):
            continue
        out[room] = float(min(max(value, 0.0), 1.0))
    return out


def _parse_room_int_thresholds(raw: str | None, *, min_value: int = 0) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not raw:
        return out
    for token in str(raw).split(","):
        item = token.strip()
        if not item or "=" not in item:
            continue
        key_raw, value_raw = item.split("=", 1)
        room = key_raw.strip().lower()
        if not room:
            continue
        try:
            value = int(float(value_raw))
        except (TypeError, ValueError):
            continue
        out[room] = int(max(value, int(min_value)))
    return out


def _parse_tune_rooms(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {token.strip().lower() for token in str(raw).split(",") if token.strip()}


def _parse_room_label_thresholds(raw: str | None) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not raw:
        return out
    for token in str(raw).split(","):
        item = token.strip()
        if not item or "=" not in item:
            continue
        key_raw, value_raw = item.split("=", 1)
        key_txt = key_raw.strip().lower()
        if "." not in key_txt:
            continue
        room, label = key_txt.split(".", 1)
        room = room.strip()
        label = label.strip()
        if not room or not label:
            continue
        try:
            value = float(value_raw)
        except (TypeError, ValueError):
            continue
        out.setdefault(room, {})[label] = float(min(max(value, 0.0), 1.0))
    return out


def _select_room_regime_by_train_prior(
    *,
    room_key: str,
    y_fit_labels: Sequence[object],
    low_occ_thresholds: Dict[str, float] | None = None,
) -> Dict[str, object]:
    room = str(room_key).strip().lower()
    y_fit = np.asarray([str(v).strip().lower() for v in y_fit_labels], dtype=object)
    if len(y_fit) == 0:
        return {
            "room": room,
            "train_occupied_rate": 0.0,
            "low_occupancy_threshold": None,
            "regime": "normal",
            "applied": False,
            "reason": "empty_fit_labels",
        }
    train_occ_rate = float(np.mean(y_fit != "unoccupied"))
    defaults = {"bedroom": 0.16, "livingroom": 0.10}
    threshold_map = dict(low_occ_thresholds or {})
    threshold = float(threshold_map.get(room, defaults.get(room, 0.12)))
    regime = "low_occupancy" if train_occ_rate <= threshold else "normal"
    return {
        "room": room,
        "train_occupied_rate": float(train_occ_rate),
        "low_occupancy_threshold": float(threshold),
        "regime": str(regime),
        "applied": room in {"bedroom", "livingroom"},
    }


def _cap_replay_rows(
    *,
    x: np.ndarray,
    y: np.ndarray,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=object)
    n = int(len(y_arr))
    if n <= 0:
        return np.empty((0, x_arr.shape[1] if x_arr.ndim == 2 else 0), dtype=float), np.empty((0,), dtype=object)
    if n <= int(max_rows):
        return x_arr, y_arr
    step = max(int(np.floor(n / max(int(max_rows), 1))), 1)
    idx = np.arange(0, n, step, dtype=int)[: int(max_rows)]
    return x_arr[idx], y_arr[idx]


def _room_mae_tuning_flag_enabled(
    *,
    room_key: str,
    enable_room_mae_threshold_tuning: bool,
    enable_kitchen_mae_threshold_tuning: bool,
) -> tuple[bool, str]:
    room = str(room_key).strip().lower()
    if room == "kitchen":
        enabled = bool(enable_kitchen_mae_threshold_tuning)
        return enabled, "kitchen_mae_tuning_disabled"
    enabled = bool(enable_room_mae_threshold_tuning)
    return enabled, "room_mae_tuning_disabled"


def _build_room_model_config(
    *,
    seed: int,
    room_key: str,
    bedroom_livingroom_regime: str = "normal",
    enable_bedroom_livingroom_regime_routing: bool = False,
    enable_bedroom_livingroom_stage_a_hgb: bool = False,
    enable_bedroom_livingroom_stage_a_sequence_model: bool = False,
    bedroom_livingroom_stage_a_sequence_lag_windows: int = 12,
    enable_bedroom_livingroom_stage_a_transformer: bool = False,
    bedroom_livingroom_stage_a_transformer_epochs: int = 8,
    bedroom_livingroom_stage_a_transformer_batch_size: int = 256,
    bedroom_livingroom_stage_a_transformer_learning_rate: float = 7e-4,
    bedroom_livingroom_stage_a_transformer_hidden_dim: int = 48,
    bedroom_livingroom_stage_a_transformer_num_heads: int = 2,
    bedroom_livingroom_stage_a_transformer_dropout: float = 0.15,
    bedroom_livingroom_stage_a_transformer_class_weight_power: float = 0.5,
    bedroom_livingroom_stage_a_transformer_conv_kernel_size: int = 3,
    bedroom_livingroom_stage_a_transformer_conv_blocks: int = 2,
    enable_bedroom_livingroom_stage_a_transformer_sequence_filter: bool = True,
) -> EventFirstConfig:
    """
    Build model config for stage-A occupancy model.
    """
    key = str(room_key).strip().lower()
    regime = str(bedroom_livingroom_regime).strip().lower()
    if regime not in {"low_occupancy", "normal"}:
        regime = "normal"
    use_sequence_stage_a = bool(enable_bedroom_livingroom_stage_a_sequence_model) and key in {
        "bedroom",
        "livingroom",
    }
    use_transformer_stage_a = bool(enable_bedroom_livingroom_stage_a_transformer) and key in {
        "bedroom",
        "livingroom",
    }
    use_hgb_stage_a = bool(enable_bedroom_livingroom_stage_a_hgb) and key in {"bedroom", "livingroom"}
    sequence_lag_windows = int(max(bedroom_livingroom_stage_a_sequence_lag_windows, 1))
    if use_transformer_stage_a:
        stage_a_model_type = "sequence_transformer"
    elif use_sequence_stage_a and use_hgb_stage_a:
        stage_a_model_type = "sequence_hgb"
    elif use_sequence_stage_a:
        stage_a_model_type = "sequence_rf"
    elif use_hgb_stage_a:
        stage_a_model_type = "hgb"
    else:
        stage_a_model_type = "rf"
    stage_a_lag_windows = int(sequence_lag_windows if (use_sequence_stage_a or use_transformer_stage_a) else 0)

    bedroom_stage_a_trees = 260
    livingroom_stage_a_trees = 280
    if bool(enable_bedroom_livingroom_regime_routing) and regime == "low_occupancy":
        bedroom_stage_a_trees = 420
        livingroom_stage_a_trees = 460
    elif use_hgb_stage_a:
        bedroom_stage_a_trees = 320
        livingroom_stage_a_trees = 340

    if key == "bedroom":
        return EventFirstConfig(
            random_state=int(seed),
            n_estimators_stage_a=int(bedroom_stage_a_trees),
            n_estimators_stage_b=220,
            min_samples_leaf=2,
            stage_a_class_weight=("balanced_sqrt" if use_transformer_stage_a else "balanced"),
            stage_b_class_weight="balanced_subsample",
            stage_a_model_type=stage_a_model_type,
            stage_a_temporal_lag_windows=stage_a_lag_windows,
            stage_a_transformer_epochs=int(max(bedroom_livingroom_stage_a_transformer_epochs, 1)),
            stage_a_transformer_batch_size=int(max(bedroom_livingroom_stage_a_transformer_batch_size, 1)),
            stage_a_transformer_learning_rate=float(max(bedroom_livingroom_stage_a_transformer_learning_rate, 1e-5)),
            stage_a_transformer_hidden_dim=int(max(bedroom_livingroom_stage_a_transformer_hidden_dim, 8)),
            stage_a_transformer_num_heads=int(max(bedroom_livingroom_stage_a_transformer_num_heads, 1)),
            stage_a_transformer_dropout=float(
                min(max(bedroom_livingroom_stage_a_transformer_dropout, 0.0), 0.5)
            ),
            stage_a_transformer_class_weight_power=float(
                min(max(bedroom_livingroom_stage_a_transformer_class_weight_power, 0.0), 1.0)
            ),
            stage_a_transformer_conv_kernel_size=int(max(bedroom_livingroom_stage_a_transformer_conv_kernel_size, 2)),
            stage_a_transformer_conv_blocks=int(max(bedroom_livingroom_stage_a_transformer_conv_blocks, 1)),
            stage_a_transformer_use_sequence_filter=bool(
                enable_bedroom_livingroom_stage_a_transformer_sequence_filter
            ),
        )
    if key == "livingroom":
        return EventFirstConfig(
            random_state=int(seed),
            n_estimators_stage_a=int(livingroom_stage_a_trees),
            n_estimators_stage_b=220,
            min_samples_leaf=2,
            stage_a_class_weight=("balanced_sqrt" if use_transformer_stage_a else "balanced"),
            stage_b_class_weight="balanced_subsample",
            stage_a_model_type=stage_a_model_type,
            stage_a_temporal_lag_windows=stage_a_lag_windows,
            stage_a_transformer_epochs=int(max(bedroom_livingroom_stage_a_transformer_epochs, 1)),
            stage_a_transformer_batch_size=int(max(bedroom_livingroom_stage_a_transformer_batch_size, 1)),
            stage_a_transformer_learning_rate=float(max(bedroom_livingroom_stage_a_transformer_learning_rate, 1e-5)),
            stage_a_transformer_hidden_dim=int(max(bedroom_livingroom_stage_a_transformer_hidden_dim, 8)),
            stage_a_transformer_num_heads=int(max(bedroom_livingroom_stage_a_transformer_num_heads, 1)),
            stage_a_transformer_dropout=float(
                min(max(bedroom_livingroom_stage_a_transformer_dropout, 0.0), 0.5)
            ),
            stage_a_transformer_class_weight_power=float(
                min(max(bedroom_livingroom_stage_a_transformer_class_weight_power, 0.0), 1.0)
            ),
            stage_a_transformer_conv_kernel_size=int(max(bedroom_livingroom_stage_a_transformer_conv_kernel_size, 2)),
            stage_a_transformer_conv_blocks=int(max(bedroom_livingroom_stage_a_transformer_conv_blocks, 1)),
            stage_a_transformer_use_sequence_filter=bool(
                enable_bedroom_livingroom_stage_a_transformer_sequence_filter
            ),
        )
    return EventFirstConfig(random_state=int(seed))


def _parse_hour_range(raw: str | None, *, default_start: int, default_end: int) -> tuple[int, int]:
    if not raw:
        return int(default_start), int(default_end)
    txt = str(raw).strip()
    if "-" not in txt:
        return int(default_start), int(default_end)
    left, right = txt.split("-", 1)
    try:
        start = int(left.strip())
        end = int(right.strip())
    except (TypeError, ValueError):
        return int(default_start), int(default_end)
    start = int(min(max(start, 0), 23))
    end = int(min(max(end, 0), 23))
    return start, end


def _parse_hour_ranges(raw: str | None, *, default_ranges: Sequence[tuple[int, int]]) -> List[tuple[int, int]]:
    if not raw:
        return [(int(a), int(b)) for a, b in default_ranges]
    out: List[tuple[int, int]] = []
    for token in str(raw).split(","):
        item = token.strip()
        if not item:
            continue
        start, end = _parse_hour_range(item, default_start=-1, default_end=-1)
        if 0 <= start <= 23 and 0 <= end <= 23:
            out.append((start, end))
    if not out:
        return [(int(a), int(b)) for a, b in default_ranges]
    return out


def _in_hour_range(hour: int, start_hour: int, end_hour: int) -> bool:
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return bool(start_hour <= hour < end_hour)
    return bool(hour >= start_hour or hour < end_hour)


def _apply_single_resident_arbitration(
    *,
    split_room_outputs: Dict[str, Dict[str, object]],
    arbitration_rooms: Sequence[str],
    min_confidence_margin: float = 0.02,
    bedroom_night_start_hour: int = 22,
    bedroom_night_end_hour: int = 7,
    bedroom_night_min_run_windows: int = 18,
    bedroom_night_min_label_score: float = 0.30,
    kitchen_min_label_score: float = 0.60,
    kitchen_guard_hour_ranges: Sequence[tuple[int, int]] | None = None,
    continuity_min_run_windows: int = 12,
    continuity_min_occ_prob: float = 0.55,
    unoccupied_label: str = "unoccupied",
) -> Dict[str, object]:
    """
    In single-resident mode, suppress concurrent occupied labels across selected rooms.

    For each timestamp, if multiple rooms are predicted occupied, keep the strongest room
    by occupancy probability (tie-break by predicted-label probability) and set others to unoccupied.
    """
    def _in_hour_ranges(hour: int, ranges: Sequence[tuple[int, int]]) -> bool:
        for start_hour, end_hour in ranges:
            if _in_hour_range(hour, int(start_hour), int(end_hour)):
                return True
        return False

    def _run_lengths(labels: np.ndarray) -> np.ndarray:
        out = np.zeros(shape=(int(len(labels)),), dtype=int)
        i = 0
        n = int(len(labels))
        while i < n:
            label = str(labels[i]).strip().lower()
            if not label or label == str(unoccupied_label).strip().lower():
                i += 1
                continue
            start = i
            while i < n and str(labels[i]).strip().lower() == label:
                i += 1
            run_len = int(i - start)
            out[start:i] = run_len
        return out

    room_keys = [str(r).strip().lower() for r in arbitration_rooms if str(r).strip()]
    room_map = {str(room).strip().lower(): room for room in split_room_outputs.keys()}
    candidate_room_keys = [rk for rk in room_keys if rk in room_map]
    debug: Dict[str, object] = {
        "enabled": True,
        "candidate_rooms": list(candidate_room_keys),
        "min_confidence_margin": float(min_confidence_margin),
        "bedroom_night_start_hour": int(bedroom_night_start_hour),
        "bedroom_night_end_hour": int(bedroom_night_end_hour),
        "bedroom_night_min_run_windows": int(bedroom_night_min_run_windows),
        "bedroom_night_min_label_score": float(bedroom_night_min_label_score),
        "kitchen_min_label_score": float(kitchen_min_label_score),
        "kitchen_guard_hour_ranges": list(kitchen_guard_hour_ranges or []),
        "continuity_min_run_windows": int(continuity_min_run_windows),
        "continuity_min_occ_prob": float(continuity_min_occ_prob),
        "conflicts_total": 0,
        "adjustments_total": 0,
        "guard_wins": {
            "bedroom_night_sleep": 0,
            "kitchen_high_confidence": 0,
            "long_run_continuity": 0,
        },
        "per_room": {},
    }
    if len(candidate_room_keys) < 2:
        debug["reason"] = "insufficient_candidate_rooms"
        return debug

    min_confidence_margin = float(min(max(min_confidence_margin, 1.0), 0.0))
    bedroom_night_start_hour = int(min(max(bedroom_night_start_hour, 0), 23))
    bedroom_night_end_hour = int(min(max(bedroom_night_end_hour, 0), 23))
    bedroom_night_min_run_windows = int(max(bedroom_night_min_run_windows, 1))
    bedroom_night_min_label_score = float(min(max(bedroom_night_min_label_score, 0.0), 1.0))
    kitchen_min_label_score = float(min(max(kitchen_min_label_score, 0.0), 1.0))
    kitchen_guard_hour_ranges = list(kitchen_guard_hour_ranges or [])
    continuity_min_run_windows = int(max(continuity_min_run_windows, 1))
    continuity_min_occ_prob = float(min(max(continuity_min_occ_prob, 0.0), 1.0))

    ts_candidates: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for room_key in candidate_room_keys:
        room = room_map[room_key]
        payload = split_room_outputs.get(room, {})
        test_df = payload.get("test_df")
        if not isinstance(test_df, pd.DataFrame) or "timestamp" not in test_df.columns:
            continue
        timestamps = pd.to_datetime(test_df["timestamp"], errors="coerce")
        y_pred = np.asarray(payload.get("y_pred", []), dtype=object)
        occ_probs = np.asarray(payload.get("occupancy_probs", []), dtype=float)
        activity_probs = payload.get("activity_probs", {})
        if not isinstance(activity_probs, dict):
            activity_probs = {}
        n = int(min(len(timestamps), len(y_pred), len(occ_probs)))
        if n <= 0:
            continue
        payload["y_pred"] = y_pred
        run_len = _run_lengths(y_pred)
        for idx in range(n):
            ts = timestamps.iloc[idx]
            if pd.isna(ts):
                continue
            label = str(y_pred[idx]).strip().lower()
            if not label or label == str(unoccupied_label).strip().lower():
                continue
            label_scores = activity_probs.get(label)
            label_score = 0.0
            if label_scores is not None:
                try:
                    label_score = float(np.asarray(label_scores, dtype=float)[idx])
                except Exception:
                    label_score = 0.0
            ts_obj = pd.Timestamp(ts)
            hour = int(ts_obj.hour)
            guard_priority = 0
            guard_reason = "none"
            this_run_len = int(run_len[idx]) if idx < len(run_len) else 0
            if (
                room_key == "bedroom"
                and label == "sleep"
                and _in_hour_range(hour, bedroom_night_start_hour, bedroom_night_end_hour)
                and this_run_len >= bedroom_night_min_run_windows
                and label_score >= bedroom_night_min_label_score
            ):
                guard_priority = 3
                guard_reason = "bedroom_night_sleep"
            elif (
                room_key == "kitchen"
                and label_score >= kitchen_min_label_score
                and (
                    not kitchen_guard_hour_ranges
                    or _in_hour_ranges(hour, kitchen_guard_hour_ranges)
                )
            ):
                guard_priority = 2
                guard_reason = "kitchen_high_confidence"
            elif this_run_len >= continuity_min_run_windows and float(occ_probs[idx]) >= continuity_min_occ_prob:
                guard_priority = 1
                guard_reason = "long_run_continuity"
            ts_candidates[int(pd.Timestamp(ts).value)].append(
                {
                    "room": room,
                    "room_key": room_key,
                    "index": int(idx),
                    "hour": hour,
                    "run_len": this_run_len,
                    "occ_prob": float(occ_probs[idx]),
                    "label": label,
                    "label_score": float(label_score),
                    "guard_priority": int(guard_priority),
                    "guard_reason": str(guard_reason),
                }
            )

    per_room_counts: Dict[str, int] = {room_map[rk]: 0 for rk in candidate_room_keys}
    for _, candidates in ts_candidates.items():
        if len(candidates) <= 1:
            continue
        debug["conflicts_total"] = int(debug["conflicts_total"]) + 1
        ranked = sorted(
            candidates,
            key=lambda c: (
                int(c.get("guard_priority", 0)),
                float(c.get("occ_prob", 0.0)),
                float(c.get("label_score", 0.0)),
            ),
            reverse=True,
        )
        winner = ranked[0]
        if len(ranked) > 1:
            second = ranked[1]
            if (
                int(winner.get("guard_priority", 0)) == int(second.get("guard_priority", 0))
                and
                float(winner.get("occ_prob", 0.0)) - float(second.get("occ_prob", 0.0))
                <= float(min_confidence_margin)
                and float(second.get("label_score", 0.0)) > float(winner.get("label_score", 0.0))
            ):
                winner = second
        reason = str(winner.get("guard_reason", "none"))
        if reason in debug["guard_wins"]:
            debug["guard_wins"][reason] = int(debug["guard_wins"][reason]) + 1
        for cand in ranked:
            if cand is winner:
                continue
            room = str(cand["room"])
            idx = int(cand["index"])
            arr = np.asarray(split_room_outputs[room]["y_pred"], dtype=object)
            if 0 <= idx < len(arr) and str(arr[idx]).strip().lower() != str(unoccupied_label).strip().lower():
                arr[idx] = str(unoccupied_label)
                split_room_outputs[room]["y_pred"] = arr
                per_room_counts[room] = int(per_room_counts.get(room, 0)) + 1
                debug["adjustments_total"] = int(debug["adjustments_total"]) + 1

    debug["per_room"] = {
        room: {"adjusted_windows": int(cnt)} for room, cnt in sorted(per_room_counts.items(), key=lambda x: x[0])
    }
    return debug


def _apply_livingroom_cross_room_presence_decoder(
    *,
    split_room_outputs: Dict[str, Dict[str, object]],
    supporting_rooms: Sequence[str],
    hold_minutes: float = 10.0,
    max_extension_minutes: float = 24.0,
    entry_occ_threshold: float = 0.66,
    entry_room_prob_threshold: float = 0.42,
    entry_motion_threshold: float = 0.75,
    refresh_occ_threshold: float = 0.44,
    refresh_room_prob_threshold: float = 0.18,
    other_room_exit_occ_threshold: float = 0.62,
    other_room_exit_confirm_windows: int = 2,
    other_room_unoccupied_max_occ_prob: float = 0.30,
    entrance_exit_occ_threshold: float = 0.58,
    min_support_rooms: int = 2,
    require_other_room_predicted_occupied: bool = True,
    enable_bedroom_sleep_night_guard: bool = False,
    night_bedroom_guard_start_hour: int = 22,
    night_bedroom_guard_end_hour: int = 6,
    night_bedroom_sleep_occ_threshold: float = 0.66,
    night_bedroom_sleep_prob_threshold: float = 0.55,
    night_bedroom_exit_occ_threshold: float = 0.35,
    night_bedroom_exit_motion_threshold: float = 0.75,
    night_entry_occ_threshold: float = 0.66,
    night_entry_motion_threshold: float = 0.75,
    night_entry_confirm_windows: int = 2,
    night_bedroom_suppression_label: str = "unknown",
    night_bedroom_min_coverage: float = 0.80,
    night_bedroom_flatline_motion_std_max: float = 1e-4,
    night_bedroom_flatline_occ_std_max: float = 1e-4,
    night_bedroom_flatline_min_windows: int = 2160,
    unoccupied_label: str = "unoccupied",
) -> Dict[str, object]:
    """
    LivingRoom process-of-elimination decoder for passive occupancy.

    Idea:
    - Enter/refresh LR state only with LR-local evidence (probability/motion).
    - Keep LR occupied during short passive periods only when other rooms are quiet.
    - Exit LR quickly when another room has sustained occupancy evidence.
    """
    room_map = {str(room).strip().lower(): room for room in split_room_outputs.keys()}
    debug: Dict[str, object] = {
        "enabled": True,
        "reason": "not_applied",
        "livingroom_room": room_map.get("livingroom"),
        "supporting_rooms_requested": sorted(
            [str(r).strip().lower() for r in supporting_rooms if str(r).strip()]
        ),
        "supporting_rooms_used": [],
        "changed_windows": 0,
        "added_occupied_windows": 0,
        "entry_events": 0,
        "refresh_events": 0,
        "exit_events": 0,
        "maintained_by_absence_windows": 0,
        "insufficient_support_windows": 0,
        "night_bedroom_guard_applied": False,
        "night_bedroom_guard_reason": "disabled",
        "night_bedroom_guard_suppressed_windows": 0,
        "night_bedroom_guard_blocked_entries": 0,
        "night_bedroom_guard_unknown_windows": 0,
        "night_bedroom_guard_unoccupied_windows": 0,
        "night_bedroom_guard_bedroom_coverage": 0.0,
        "night_bedroom_guard_bedroom_motion_std": None,
        "night_bedroom_guard_bedroom_occ_std": None,
    }

    lr_room = room_map.get("livingroom")
    if not lr_room:
        debug["reason"] = "livingroom_not_available"
        return debug

    lr_payload = split_room_outputs.get(lr_room, {})
    test_df = lr_payload.get("test_df")
    if not isinstance(test_df, pd.DataFrame) or "timestamp" not in test_df.columns:
        debug["reason"] = "livingroom_test_df_missing"
        return debug

    timestamps = pd.to_datetime(test_df["timestamp"], errors="coerce")
    y_pred = np.asarray(lr_payload.get("y_pred", []), dtype=object).copy()
    occupancy_probs = np.asarray(lr_payload.get("occupancy_probs", []), dtype=float)
    activity_probs_raw = lr_payload.get("activity_probs", {})
    activity_probs = activity_probs_raw if isinstance(activity_probs_raw, dict) else {}
    n = int(min(len(timestamps), len(y_pred), len(occupancy_probs)))
    if n <= 0:
        debug["reason"] = "livingroom_empty_series"
        return debug

    timestamps = timestamps.iloc[:n]
    y_pred = np.asarray(y_pred[:n], dtype=object)
    occupancy_probs = np.asarray(occupancy_probs[:n], dtype=float)

    hold_windows = int(max(round(float(max(hold_minutes, 0.0)) * 6.0), 1))
    max_extension_windows = int(max(round(float(max(max_extension_minutes, 0.0)) * 6.0), 0))
    min_support_rooms = int(max(min_support_rooms, 1))
    other_room_exit_confirm_windows = int(max(other_room_exit_confirm_windows, 1))
    entry_occ_threshold = float(min(max(entry_occ_threshold, 0.0), 1.0))
    entry_room_prob_threshold = float(min(max(entry_room_prob_threshold, 0.0), 1.0))
    entry_motion_threshold = float(max(entry_motion_threshold, 0.0))
    refresh_occ_threshold = float(min(max(refresh_occ_threshold, 0.0), 1.0))
    refresh_room_prob_threshold = float(min(max(refresh_room_prob_threshold, 0.0), 1.0))
    other_room_exit_occ_threshold = float(min(max(other_room_exit_occ_threshold, 0.0), 1.0))
    other_room_unoccupied_max_occ_prob = float(min(max(other_room_unoccupied_max_occ_prob, 0.0), 1.0))
    entrance_exit_occ_threshold = float(min(max(entrance_exit_occ_threshold, 0.0), 1.0))
    enable_bedroom_sleep_night_guard = bool(enable_bedroom_sleep_night_guard)
    night_bedroom_guard_start_hour = int(min(max(night_bedroom_guard_start_hour, 0), 23))
    night_bedroom_guard_end_hour = int(min(max(night_bedroom_guard_end_hour, 0), 23))
    night_bedroom_sleep_occ_threshold = float(min(max(night_bedroom_sleep_occ_threshold, 0.0), 1.0))
    night_bedroom_sleep_prob_threshold = float(min(max(night_bedroom_sleep_prob_threshold, 0.0), 1.0))
    night_bedroom_exit_occ_threshold = float(min(max(night_bedroom_exit_occ_threshold, 0.0), 1.0))
    night_bedroom_exit_motion_threshold = float(max(night_bedroom_exit_motion_threshold, 0.0))
    night_entry_occ_threshold = float(min(max(night_entry_occ_threshold, 0.0), 1.0))
    night_entry_motion_threshold = float(max(night_entry_motion_threshold, 0.0))
    night_entry_confirm_windows = int(max(night_entry_confirm_windows, 1))
    night_bedroom_min_coverage = float(min(max(night_bedroom_min_coverage, 0.0), 1.0))
    night_bedroom_flatline_motion_std_max = float(max(night_bedroom_flatline_motion_std_max, 0.0))
    night_bedroom_flatline_occ_std_max = float(max(night_bedroom_flatline_occ_std_max, 0.0))
    night_bedroom_flatline_min_windows = int(max(night_bedroom_flatline_min_windows, 1))
    suppression_label = str(night_bedroom_suppression_label or "unknown").strip().lower()
    if suppression_label not in {"unknown", str(unoccupied_label).strip().lower()}:
        suppression_label = "unknown"

    unoccupied_aliases = {str(unoccupied_label).strip().lower(), "", "unknown"}
    motion_vals = (
        pd.to_numeric(test_df["motion"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[:n]
        if "motion" in test_df.columns
        else np.zeros(shape=(n,), dtype=float)
    )
    motion_vals = np.nan_to_num(motion_vals, nan=0.0, posinf=0.0, neginf=0.0)

    lr_labels = sorted(
        [
            str(lbl).strip().lower()
            for lbl in activity_probs.keys()
            if str(lbl).strip().lower() not in unoccupied_aliases
        ]
    )
    if not lr_labels:
        lr_labels = ["livingroom_normal_use"]

    lr_label_scores = np.zeros(shape=(n,), dtype=float)
    for label in lr_labels:
        vals = np.asarray(activity_probs.get(label, np.zeros(n)), dtype=float)
        if len(vals) < n:
            vals = np.resize(vals, n).astype(float)
        lr_label_scores = np.maximum(lr_label_scores, np.clip(vals[:n], 0.0, 1.0))

    support_keys = []
    for room in supporting_rooms:
        key = str(room).strip().lower()
        if key and key != "livingroom" and key in room_map and key not in support_keys:
            support_keys.append(key)
    debug["supporting_rooms_used"] = list(support_keys)
    if not support_keys:
        debug["reason"] = "no_supporting_rooms_available"
        return debug

    support_lookup: Dict[str, Dict[int, tuple[float, bool, float, float, bool]]] = {}
    for room_key in support_keys:
        room_name = room_map[room_key]
        payload = split_room_outputs.get(room_name, {})
        df = payload.get("test_df")
        if not isinstance(df, pd.DataFrame) or "timestamp" not in df.columns:
            continue
        ts_room = pd.to_datetime(df["timestamp"], errors="coerce")
        y_room = np.asarray(payload.get("y_pred", []), dtype=object)
        occ_room = np.asarray(payload.get("occupancy_probs", []), dtype=float)
        activity_probs_room_raw = payload.get("activity_probs", {})
        activity_probs_room = activity_probs_room_raw if isinstance(activity_probs_room_raw, dict) else {}
        motion_room = (
            pd.to_numeric(df["motion"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if "motion" in df.columns
            else np.zeros(shape=(len(ts_room),), dtype=float)
        )
        m = int(min(len(ts_room), len(y_room), len(occ_room)))
        if m <= 0:
            continue
        motion_room = np.asarray(motion_room[:m], dtype=float)
        sleep_scores = np.zeros(shape=(m,), dtype=float)
        if room_key == "bedroom":
            for label_name, label_scores_raw in activity_probs_room.items():
                lbl = str(label_name).strip().lower()
                if "sleep" not in lbl:
                    continue
                label_scores = np.asarray(label_scores_raw, dtype=float)
                if len(label_scores) < m:
                    label_scores = np.resize(label_scores, m).astype(float)
                sleep_scores = np.maximum(sleep_scores, np.clip(label_scores[:m], 0.0, 1.0))

        room_map_ts: Dict[int, tuple[float, bool, float, float, bool]] = {}
        for i in range(m):
            ts_val = ts_room.iloc[i]
            if pd.isna(ts_val):
                continue
            occ_prob = float(np.clip(occ_room[i], 0.0, 1.0))
            pred_label = str(y_room[i]).strip().lower()
            pred_occ = bool(pred_label not in unoccupied_aliases)
            motion_val = float(np.nan_to_num(motion_room[i], nan=0.0, posinf=0.0, neginf=0.0))
            sleep_prob = float(sleep_scores[i]) if room_key == "bedroom" else 0.0
            pred_sleep = bool(pred_label == "sleep") if room_key == "bedroom" else False
            room_map_ts[int(pd.Timestamp(ts_val).value)] = (
                occ_prob,
                pred_occ,
                motion_val,
                sleep_prob,
                pred_sleep,
            )
        support_lookup[room_key] = room_map_ts

    if not support_lookup:
        debug["reason"] = "supporting_room_series_unavailable"
        return debug

    bedroom_guard_ready = False
    bedroom_guard_reason = "disabled"
    bedroom_coverage = 0.0
    bedroom_motion_std: Optional[float] = None
    bedroom_occ_std: Optional[float] = None
    if enable_bedroom_sleep_night_guard:
        bedroom_map = support_lookup.get("bedroom")
        if bedroom_map is None:
            bedroom_guard_reason = "bedroom_support_unavailable"
        else:
            bedroom_count = int(len(bedroom_map))
            bedroom_coverage = float(bedroom_count / max(n, 1))
            if bedroom_coverage < night_bedroom_min_coverage:
                bedroom_guard_reason = "bedroom_coverage_below_min"
            else:
                occ_vals = np.asarray([float(v[0]) for v in bedroom_map.values()], dtype=float)
                motion_vals_bed = np.asarray([float(v[2]) for v in bedroom_map.values()], dtype=float)
                if len(occ_vals) >= night_bedroom_flatline_min_windows:
                    bedroom_motion_std = float(np.nanstd(motion_vals_bed))
                    bedroom_occ_std = float(np.nanstd(occ_vals))
                    is_flatline = bool(
                        bedroom_motion_std <= night_bedroom_flatline_motion_std_max
                        and bedroom_occ_std <= night_bedroom_flatline_occ_std_max
                    )
                    if is_flatline:
                        bedroom_guard_reason = "bedroom_sensor_flatline"
                    else:
                        bedroom_guard_ready = True
                        bedroom_guard_reason = "applied"
                else:
                    bedroom_guard_ready = True
                    bedroom_guard_reason = "applied"
    debug["night_bedroom_guard_applied"] = bool(bedroom_guard_ready)
    debug["night_bedroom_guard_reason"] = str(bedroom_guard_reason)
    debug["night_bedroom_guard_bedroom_coverage"] = float(bedroom_coverage)
    debug["night_bedroom_guard_bedroom_motion_std"] = bedroom_motion_std
    debug["night_bedroom_guard_bedroom_occ_std"] = bedroom_occ_std

    out = np.asarray(y_pred, dtype=object).copy()
    active = False
    hold_remaining = 0
    windows_since_entry = int(max_extension_windows + hold_windows + 1)
    exit_counter = 0
    changed_windows = 0
    added_windows = 0
    maintained_by_absence_windows = 0
    insufficient_support_windows = 0
    entry_events = 0
    refresh_events = 0
    exit_events = 0
    night_entry_streak = 0
    night_guard_suppressed_windows = 0
    night_guard_blocked_entries = 0
    night_guard_unknown_windows = 0
    night_guard_unoccupied_windows = 0

    for i in range(n):
        ts = timestamps.iloc[i]
        if pd.isna(ts):
            continue
        ts_key = int(pd.Timestamp(ts).value)
        support_available = 0
        max_other_occ_prob = 0.0
        any_other_pred_occ = False
        all_other_quiet = True
        entrance_exit_signal = False
        any_other_exit_signal = False

        for room_key, room_ts_map in support_lookup.items():
            sample = room_ts_map.get(ts_key)
            if sample is None:
                continue
            support_available += 1
            occ_prob, pred_occ, _, _, _ = sample
            max_other_occ_prob = max(max_other_occ_prob, float(occ_prob))
            any_other_pred_occ = bool(any_other_pred_occ or pred_occ)
            if float(occ_prob) > other_room_unoccupied_max_occ_prob or bool(pred_occ):
                all_other_quiet = False
            is_exit_occ = bool(float(occ_prob) >= other_room_exit_occ_threshold)
            if bool(require_other_room_predicted_occupied):
                is_exit_occ = bool(is_exit_occ and pred_occ)
            if is_exit_occ:
                any_other_exit_signal = True
            if room_key == "entrance":
                entrance_exit_signal = bool(
                    float(occ_prob) >= entrance_exit_occ_threshold
                    and (not bool(require_other_room_predicted_occupied) or pred_occ)
                )

        if support_available < min_support_rooms:
            insufficient_support_windows += 1
            all_other_quiet = False

        entry_signal = bool(
            (
                occupancy_probs[i] >= entry_occ_threshold
                and lr_label_scores[i] >= entry_room_prob_threshold
            )
            or motion_vals[i] >= entry_motion_threshold
        )
        refresh_signal = bool(
            (
                occupancy_probs[i] >= refresh_occ_threshold
                and lr_label_scores[i] >= refresh_room_prob_threshold
            )
            or motion_vals[i] >= entry_motion_threshold
        )

        if bedroom_guard_ready:
            hour = int(pd.Timestamp(ts).hour)
            in_guard_window = _in_hour_range(hour, night_bedroom_guard_start_hour, night_bedroom_guard_end_hour)
            if in_guard_window:
                strong_night_entry = bool(
                    occupancy_probs[i] >= night_entry_occ_threshold
                    and motion_vals[i] >= night_entry_motion_threshold
                )
                night_entry_streak = int(night_entry_streak + 1) if strong_night_entry else 0
                if entry_signal and night_entry_streak < night_entry_confirm_windows:
                    entry_signal = False
                    night_guard_blocked_entries += 1

                bedroom_sample = support_lookup.get("bedroom", {}).get(ts_key)
                if bedroom_sample is not None:
                    bedroom_occ_prob, _, bedroom_motion, bedroom_sleep_prob, bedroom_pred_sleep = bedroom_sample
                    bedroom_sleep_active = bool(
                        bedroom_occ_prob >= night_bedroom_sleep_occ_threshold
                        and (
                            bedroom_sleep_prob >= night_bedroom_sleep_prob_threshold
                            or bool(bedroom_pred_sleep)
                        )
                    )
                    bedroom_exit_signal = bool(
                        bedroom_occ_prob <= night_bedroom_exit_occ_threshold
                        and bedroom_motion >= night_bedroom_exit_motion_threshold
                    )
                    if (not active) and bedroom_sleep_active and (not bedroom_exit_signal) and (not entry_signal):
                        curr_label = str(out[i]).strip().lower()
                        if curr_label not in unoccupied_aliases:
                            out[i] = suppression_label
                            changed_windows += 1
                            if suppression_label == "unknown":
                                night_guard_unknown_windows += 1
                            else:
                                night_guard_unoccupied_windows += 1
                        night_guard_suppressed_windows += 1
                        active = False
                        hold_remaining = 0
                        windows_since_entry = int(max_extension_windows + hold_windows + 1)
                        exit_counter = 0
                        continue
            else:
                night_entry_streak = 0

        if entry_signal:
            if not active:
                entry_events += 1
            else:
                refresh_events += 1
            active = True
            hold_remaining = int(hold_windows)
            windows_since_entry = 0
            exit_counter = 0
        elif active:
            windows_since_entry += 1
            if refresh_signal:
                refresh_events += 1
                hold_remaining = int(hold_windows)
                windows_since_entry = 0
                exit_counter = 0
            elif entrance_exit_signal:
                active = False
                hold_remaining = 0
                windows_since_entry = int(max_extension_windows + hold_windows + 1)
                exit_counter = 0
                exit_events += 1
            else:
                if any_other_exit_signal:
                    exit_counter += 1
                else:
                    exit_counter = max(exit_counter - 1, 0)
                if exit_counter >= other_room_exit_confirm_windows:
                    active = False
                    hold_remaining = 0
                    windows_since_entry = int(max_extension_windows + hold_windows + 1)
                    exit_counter = 0
                    exit_events += 1
                else:
                    if all_other_quiet and windows_since_entry <= max_extension_windows:
                        maintained_by_absence_windows += 1
                    else:
                        hold_remaining = max(hold_remaining - 1, 0)
                    if hold_remaining <= 0 and not (
                        all_other_quiet and windows_since_entry <= max_extension_windows
                    ):
                        active = False
                        windows_since_entry = int(max_extension_windows + hold_windows + 1)

        if not active:
            continue

        curr_label = str(out[i]).strip().lower()
        if curr_label in unoccupied_aliases:
            replacement = "livingroom_normal_use"
            best_score = float("-inf")
            for label in lr_labels:
                vals = np.asarray(activity_probs.get(label, np.zeros(n)), dtype=float)
                if len(vals) < n:
                    vals = np.resize(vals, n).astype(float)
                score = float(vals[i])
                if score > best_score:
                    best_score = score
                    replacement = str(label)
            out[i] = replacement
            changed_windows += 1
            added_windows += 1

    lr_payload["y_pred"] = np.asarray(out, dtype=object)
    split_room_outputs[lr_room] = lr_payload
    debug.update(
        {
            "enabled": True,
            "reason": "applied",
            "changed_windows": int(changed_windows),
            "added_occupied_windows": int(added_windows),
            "entry_events": int(entry_events),
            "refresh_events": int(refresh_events),
            "exit_events": int(exit_events),
            "maintained_by_absence_windows": int(maintained_by_absence_windows),
            "insufficient_support_windows": int(insufficient_support_windows),
            "night_bedroom_guard_suppressed_windows": int(night_guard_suppressed_windows),
            "night_bedroom_guard_blocked_entries": int(night_guard_blocked_entries),
            "night_bedroom_guard_unknown_windows": int(night_guard_unknown_windows),
            "night_bedroom_guard_unoccupied_windows": int(night_guard_unoccupied_windows),
            "config": {
                "supporting_rooms": list(support_keys),
                "hold_minutes": float(max(hold_minutes, 0.0)),
                "max_extension_minutes": float(max(max_extension_minutes, 0.0)),
                "entry_occ_threshold": float(entry_occ_threshold),
                "entry_room_prob_threshold": float(entry_room_prob_threshold),
                "entry_motion_threshold": float(entry_motion_threshold),
                "refresh_occ_threshold": float(refresh_occ_threshold),
                "refresh_room_prob_threshold": float(refresh_room_prob_threshold),
                "other_room_exit_occ_threshold": float(other_room_exit_occ_threshold),
                "other_room_exit_confirm_windows": int(other_room_exit_confirm_windows),
                "other_room_unoccupied_max_occ_prob": float(other_room_unoccupied_max_occ_prob),
                "entrance_exit_occ_threshold": float(entrance_exit_occ_threshold),
                "min_support_rooms": int(min_support_rooms),
                "require_other_room_predicted_occupied": bool(require_other_room_predicted_occupied),
                "enable_bedroom_sleep_night_guard": bool(enable_bedroom_sleep_night_guard),
                "night_bedroom_guard_start_hour": int(night_bedroom_guard_start_hour),
                "night_bedroom_guard_end_hour": int(night_bedroom_guard_end_hour),
                "night_bedroom_sleep_occ_threshold": float(night_bedroom_sleep_occ_threshold),
                "night_bedroom_sleep_prob_threshold": float(night_bedroom_sleep_prob_threshold),
                "night_bedroom_exit_occ_threshold": float(night_bedroom_exit_occ_threshold),
                "night_bedroom_exit_motion_threshold": float(night_bedroom_exit_motion_threshold),
                "night_entry_occ_threshold": float(night_entry_occ_threshold),
                "night_entry_motion_threshold": float(night_entry_motion_threshold),
                "night_entry_confirm_windows": int(night_entry_confirm_windows),
                "night_bedroom_suppression_label": str(suppression_label),
                "night_bedroom_min_coverage": float(night_bedroom_min_coverage),
                "night_bedroom_flatline_motion_std_max": float(night_bedroom_flatline_motion_std_max),
                "night_bedroom_flatline_occ_std_max": float(night_bedroom_flatline_occ_std_max),
                "night_bedroom_flatline_min_windows": int(night_bedroom_flatline_min_windows),
            },
        }
    )
    return debug


def _apply_critical_label_rescue(
    *,
    room_key: str,
    y_pred: np.ndarray,
    activity_probs: Dict[str, np.ndarray],
    rescue_min_scores: Dict[str, Dict[str, float]],
) -> tuple[np.ndarray, Dict[str, object]]:
    """
    Force one critical positive label when absent but model confidence is non-trivial.

    This is a model-probability-only rescue (no ground-truth leakage).
    """
    out = np.asarray(y_pred, dtype=object).copy()
    debug: Dict[str, object] = {"applied": False, "room": room_key, "rescues": []}
    label_cfg = rescue_min_scores.get(room_key, {})
    if not label_cfg:
        return out, debug

    for label_name, min_score in label_cfg.items():
        target = str(label_name).strip().lower()
        if not target:
            continue
        if np.any(out == target):
            continue
        scores = activity_probs.get(target)
        if scores is None or len(scores) == 0:
            continue
        idx = int(np.argmax(scores))
        score = float(scores[idx])
        if score < float(min_score):
            continue
        out[idx] = target
        debug["applied"] = True
        debug["rescues"].append(
            {"label": target, "index": idx, "score": score, "min_score": float(min_score)}
        )
    return out, debug


def _apply_kitchen_temporal_decoder(
    *,
    y_pred: np.ndarray,
    occupancy_probs: np.ndarray,
    activity_probs: Dict[str, np.ndarray],
    on_occupancy_threshold: float = 0.70,
    stay_occupancy_threshold: float = 0.45,
    on_kitchen_prob_threshold: float = 0.45,
    stay_kitchen_prob_threshold: float = 0.20,
    min_on_windows: int = 2,
    min_off_windows: int = 3,
    gap_fill_windows: int = 3,
    min_run_windows: int = 4,
    keep_short_run_if_prob_ge: float = 0.70,
) -> tuple[np.ndarray, Dict[str, object]]:
    """
    Kitchen-specific temporal decoder to stabilize kitchen_normal_use episodes.
    """
    out = np.asarray(y_pred, dtype=object).copy()
    n = int(len(out))
    if n == 0:
        return out, {"applied": False, "reason": "empty_prediction"}

    occ = np.asarray(occupancy_probs, dtype=float)
    if len(occ) != n:
        return out, {"applied": False, "reason": "occupancy_length_mismatch"}
    kitchen_prob = np.asarray(activity_probs.get("kitchen_normal_use", np.zeros(n)), dtype=float)
    if len(kitchen_prob) != n:
        kitchen_prob = np.resize(kitchen_prob, n).astype(float)

    active_mask = np.zeros(shape=(n,), dtype=bool)
    active = False
    on_count = 0
    off_count = 0
    for i in range(n):
        should_turn_on = bool(
            occ[i] >= float(on_occupancy_threshold) and kitchen_prob[i] >= float(on_kitchen_prob_threshold)
        )
        should_stay_active = bool(
            occ[i] >= float(stay_occupancy_threshold) or kitchen_prob[i] >= float(stay_kitchen_prob_threshold)
        )
        if active:
            if should_stay_active:
                off_count = 0
            else:
                off_count += 1
            if off_count >= int(min_off_windows):
                active = False
                off_count = 0
        else:
            if should_turn_on:
                on_count += 1
            else:
                on_count = 0
            if on_count >= int(min_on_windows):
                active = True
                on_count = 0
        active_mask[i] = bool(active)

    # Keep strong base kitchen positives to avoid suppressing clear high-confidence windows.
    base_kitchen = np.asarray(out == "kitchen_normal_use", dtype=bool)
    active_mask = np.logical_or(active_mask, np.logical_and(base_kitchen, kitchen_prob >= 0.55))

    # Fill short off-gaps inside active periods.
    if int(gap_fill_windows) > 0 and np.any(active_mask):
        idx = 0
        while idx < n:
            if active_mask[idx]:
                idx += 1
                continue
            start = idx
            while idx < n and not active_mask[idx]:
                idx += 1
            end = idx
            has_left = start > 0 and active_mask[start - 1]
            has_right = end < n and active_mask[end]
            if has_left and has_right and (end - start) <= int(gap_fill_windows):
                active_mask[start:end] = True

    # Prune very short active bursts unless high-confidence.
    if int(min_run_windows) > 1 and np.any(active_mask):
        idx = 0
        while idx < n:
            if not active_mask[idx]:
                idx += 1
                continue
            start = idx
            while idx < n and active_mask[idx]:
                idx += 1
            end = idx
            run_len = end - start
            run_peak = float(np.max(kitchen_prob[start:end])) if run_len > 0 else 0.0
            if run_len < int(min_run_windows) and run_peak < float(keep_short_run_if_prob_ge):
                active_mask[start:end] = False

    original_kitchen_windows = int(np.sum(base_kitchen))
    decoded_kitchen_windows = int(np.sum(active_mask))
    out[active_mask] = "kitchen_normal_use"
    suppress_mask = np.logical_and(np.asarray(out == "kitchen_normal_use", dtype=bool), ~active_mask)
    out[suppress_mask] = "unoccupied"

    return out, {
        "applied": True,
        "reason": "kitchen_temporal_decoder",
        "original_kitchen_windows": original_kitchen_windows,
        "decoded_kitchen_windows": decoded_kitchen_windows,
        "delta_kitchen_windows": int(decoded_kitchen_windows - original_kitchen_windows),
        "config": {
            "on_occupancy_threshold": float(on_occupancy_threshold),
            "stay_occupancy_threshold": float(stay_occupancy_threshold),
            "on_kitchen_prob_threshold": float(on_kitchen_prob_threshold),
            "stay_kitchen_prob_threshold": float(stay_kitchen_prob_threshold),
            "min_on_windows": int(min_on_windows),
            "min_off_windows": int(min_off_windows),
            "gap_fill_windows": int(gap_fill_windows),
            "min_run_windows": int(min_run_windows),
        },
    }


def _apply_room_occupancy_temporal_decoder(
    *,
    room_key: str,
    y_pred: np.ndarray,
    occupancy_probs: np.ndarray,
    activity_probs: Dict[str, np.ndarray],
    min_on_windows: int = 2,
    min_off_windows: int = 2,
    gap_fill_windows: int = 2,
    min_run_windows: int = 3,
) -> tuple[np.ndarray, Dict[str, object]]:
    """
    Occupancy-first temporal decoder for Bedroom/LivingRoom.
    Stabilizes occupied episodes and suppresses low-confidence occupancy flicker.
    """
    out = np.asarray(y_pred, dtype=object).copy()
    n = int(len(out))
    if n == 0:
        return out, {"applied": False, "reason": "empty_prediction"}
    if room_key not in {"bedroom", "livingroom"}:
        return out, {"applied": False, "reason": "not_applicable"}

    occ = np.asarray(occupancy_probs, dtype=float)
    if len(occ) != n:
        return out, {"applied": False, "reason": "occupancy_length_mismatch"}

    candidate_room_labels = [
        str(label).strip().lower()
        for label in sorted(activity_probs.keys())
        if str(label).strip().lower() not in {"", "unoccupied", "unknown"}
    ]
    if room_key == "bedroom":
        room_labels = tuple(candidate_room_labels or ["sleep", "bedroom_normal_use"])
        on_occ_threshold = 0.48
        stay_occ_threshold = 0.34
        on_room_prob_threshold = 0.26
        stay_room_prob_threshold = 0.16
        suppress_occ_threshold = 0.14
        suppress_room_prob_threshold = 0.06
        promote_occ_threshold = 0.62
        promote_room_prob_threshold = 0.45
    else:
        room_labels = tuple(candidate_room_labels or ["livingroom_normal_use"])
        on_occ_threshold = 0.50
        stay_occ_threshold = 0.35
        on_room_prob_threshold = 0.28
        stay_room_prob_threshold = 0.18
        suppress_occ_threshold = 0.16
        suppress_room_prob_threshold = 0.08
        promote_occ_threshold = 0.64
        promote_room_prob_threshold = 0.50

    room_prob = np.zeros(shape=(n,), dtype=float)
    for label in room_labels:
        p = np.asarray(activity_probs.get(label, np.zeros(n)), dtype=float)
        if len(p) != n:
            p = np.resize(p, n).astype(float)
        room_prob = np.maximum(room_prob, p)

    base_occupied = np.asarray(out != "unoccupied", dtype=bool)
    active_mask = np.zeros(shape=(n,), dtype=bool)
    active = False
    on_count = 0
    off_count = 0
    for i in range(n):
        should_turn_on = bool(
            occ[i] >= float(on_occ_threshold) or room_prob[i] >= float(on_room_prob_threshold)
        )
        should_stay = bool(
            occ[i] >= float(stay_occ_threshold) or room_prob[i] >= float(stay_room_prob_threshold)
        )
        if active:
            if should_stay:
                off_count = 0
            else:
                off_count += 1
            if off_count >= int(min_off_windows):
                active = False
                off_count = 0
        else:
            if should_turn_on:
                on_count += 1
            else:
                on_count = 0
            if on_count >= int(min_on_windows):
                active = True
                on_count = 0
        active_mask[i] = bool(active)

    decoded_mask = np.asarray(base_occupied, dtype=bool)

    # Conservative promotion only for very high-confidence windows,
    # to avoid inflating duration MAE on low-occupancy days.
    promote_candidate = np.logical_and(
        ~base_occupied,
        np.logical_and(
            occ >= float(promote_occ_threshold),
            room_prob >= float(promote_room_prob_threshold),
        ),
    )
    if np.any(promote_candidate):
        for start, end in _iter_true_runs(promote_candidate):
            if (end - start) >= int(min_on_windows):
                decoded_mask[start:end] = True

    suppress_candidate = np.logical_and(
        base_occupied,
        np.logical_and(
            occ < float(suppress_occ_threshold),
            room_prob < float(suppress_room_prob_threshold),
        ),
    )
    if np.any(suppress_candidate):
        for start, end in _iter_true_runs(suppress_candidate):
            if (end - start) >= int(min_off_windows):
                decoded_mask[start:end] = False

    if int(gap_fill_windows) > 0 and np.any(decoded_mask):
        idx = 0
        while idx < n:
            if decoded_mask[idx]:
                idx += 1
                continue
            start = idx
            while idx < n and not decoded_mask[idx]:
                idx += 1
            end = idx
            has_left = start > 0 and decoded_mask[start - 1]
            has_right = end < n and decoded_mask[end]
            if has_left and has_right and (end - start) <= int(gap_fill_windows):
                decoded_mask[start:end] = True

    if int(min_run_windows) > 1 and np.any(decoded_mask):
        idx = 0
        while idx < n:
            if not decoded_mask[idx]:
                idx += 1
                continue
            start = idx
            while idx < n and decoded_mask[idx]:
                idx += 1
            end = idx
            if (end - start) < int(min_run_windows):
                decoded_mask[start:end] = False

    original_occupied = int(np.sum(np.asarray(out != "unoccupied", dtype=bool)))
    out[~decoded_mask] = "unoccupied"
    if room_labels:
        if len(room_labels) > 1:
            stacked = np.vstack(
                [np.asarray(activity_probs.get(lbl, np.zeros(n)), dtype=float)[:n] for lbl in room_labels]
            )
            best_idx = np.argmax(stacked, axis=0)
            best_labels = np.asarray(room_labels, dtype=object)[best_idx]
        else:
            best_labels = np.asarray([room_labels[0]] * n, dtype=object)
        fill_mask = np.logical_and(decoded_mask, np.asarray(out == "unoccupied", dtype=bool))
        out[fill_mask] = best_labels[fill_mask]

    decoded_occupied = int(np.sum(np.asarray(out != "unoccupied", dtype=bool)))
    return out, {
        "applied": True,
        "reason": "occupancy_temporal_decoder",
        "room": room_key,
        "original_occupied_windows": original_occupied,
        "decoded_occupied_windows": decoded_occupied,
        "delta_occupied_windows": int(decoded_occupied - original_occupied),
        "config": {
            "on_occ_threshold": float(on_occ_threshold),
            "stay_occ_threshold": float(stay_occ_threshold),
            "on_room_prob_threshold": float(on_room_prob_threshold),
            "stay_room_prob_threshold": float(stay_room_prob_threshold),
            "suppress_occ_threshold": float(suppress_occ_threshold),
            "suppress_room_prob_threshold": float(suppress_room_prob_threshold),
            "promote_occ_threshold": float(promote_occ_threshold),
            "promote_room_prob_threshold": float(promote_room_prob_threshold),
            "min_on_windows": int(min_on_windows),
            "min_off_windows": int(min_off_windows),
            "gap_fill_windows": int(gap_fill_windows),
            "min_run_windows": int(min_run_windows),
        },
    }


def _iter_true_runs(mask: np.ndarray) -> List[tuple[int, int]]:
    runs: List[tuple[int, int]] = []
    in_run = False
    start = 0
    for idx, flag in enumerate(mask.tolist()):
        if flag and not in_run:
            in_run = True
            start = idx
        elif not flag and in_run:
            runs.append((start, idx))
            in_run = False
    if in_run:
        runs.append((start, len(mask)))
    return runs


def _apply_bathroom_shower_fallback(
    *,
    room_key: str,
    y_pred: np.ndarray,
    timestamps: pd.Series,
    test_df: pd.DataFrame,
    activity_probs: Dict[str, np.ndarray],
    min_duration_minutes: float = 12.0,
    min_humidity_delta: float = 1.8,
    min_shower_prob_peak: float = 0.06,
) -> tuple[np.ndarray, Dict[str, object]]:
    """
    Bathroom-only fallback to recover obvious shower-day misses.

    Applied only when no shower label exists in y_pred.
    """
    out = np.asarray(y_pred, dtype=object).copy()
    debug: Dict[str, object] = {"applied": False, "room": room_key, "reason": "not_applicable"}
    if room_key != "bathroom":
        return out, debug
    if len(out) == 0:
        debug["reason"] = "empty_prediction"
        return out, debug
    if np.any(out == "shower"):
        debug["reason"] = "shower_already_present"
        return out, debug

    ts = pd.to_datetime(timestamps, errors="coerce")
    if ts.isna().all():
        debug["reason"] = "invalid_timestamps"
        return out, debug
    ts = ts.ffill().bfill()

    occupied_mask = np.asarray(out != "unoccupied", dtype=bool)
    runs = _iter_true_runs(occupied_mask)
    # No occupied run is allowed; stage-2 humidity fallback can still recover shower.

    shower_scores = np.asarray(activity_probs.get("shower", np.zeros(len(out))), dtype=float)
    humidity = None
    if "humidity" in test_df.columns:
        humidity = pd.to_numeric(test_df["humidity"], errors="coerce").to_numpy(dtype=float)
    motion = None
    if "motion" in test_df.columns:
        motion = pd.to_numeric(test_df["motion"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    best: tuple[float, tuple[int, int], Dict[str, float]] | None = None
    for start, end in runs:
        if end - start <= 1:
            continue
        start_ts = pd.Timestamp(ts.iloc[start])
        end_ts = pd.Timestamp(ts.iloc[end - 1])
        duration_min = max((end_ts - start_ts).total_seconds(), 0.0) / 60.0
        if duration_min < float(min_duration_minutes):
            continue

        hum_delta = 0.0
        if humidity is not None:
            segment = humidity[start:end]
            if len(segment) > 0 and not np.all(np.isnan(segment)):
                hum_delta = float(np.nanmax(segment) - np.nanmin(segment))

        shower_peak = 0.0
        if len(shower_scores) >= end:
            shower_peak = float(np.nanmax(shower_scores[start:end]))

        motion_mean = 0.0
        if motion is not None and len(motion) >= end:
            motion_mean = float(np.nanmean(motion[start:end]))

        if hum_delta < float(min_humidity_delta) and shower_peak < float(min_shower_prob_peak):
            continue

        # Conservative scoring to choose one best candidate.
        score = (
            min(duration_min / 30.0, 2.0)
            + min(hum_delta / 4.0, 2.0)
            + min(shower_peak / 0.2, 2.0)
            + min(motion_mean / 0.5, 1.0)
        )
        details = {
            "duration_minutes": float(duration_min),
            "humidity_delta": float(hum_delta),
            "shower_prob_peak": float(shower_peak),
            "motion_mean": float(motion_mean),
        }
        if best is None or score > best[0]:
            best = (float(score), (start, end), details)

    if best is None:
        # Stage-2 fallback: infer shower segment from raw humidity peak shape.
        if humidity is None or len(humidity) == 0 or np.all(np.isnan(humidity)):
            debug["reason"] = "no_candidate_run"
            return out, debug

        hum_range = float(np.nanmax(humidity) - np.nanmin(humidity))
        if hum_range < 2.5:
            debug["reason"] = "no_candidate_run"
            return out, debug

        peak_idx = int(np.nanargmax(humidity))
        if peak_idx < 0 or peak_idx >= len(humidity):
            debug["reason"] = "no_candidate_run"
            return out, debug

        peak_val = float(humidity[peak_idx])
        band_threshold = peak_val - 0.8
        mask = np.asarray(humidity >= band_threshold, dtype=bool)
        runs = _iter_true_runs(mask)
        selected: tuple[int, int] | None = None
        for start, end in runs:
            if start <= peak_idx < end:
                selected = (start, end)
                break
        if selected is None:
            selected = (max(0, peak_idx - 24), min(len(out), peak_idx + 24))

        start, end = selected
        if end - start <= 1:
            debug["reason"] = "no_candidate_run"
            return out, debug
        start_ts = pd.Timestamp(ts.iloc[start])
        end_ts = pd.Timestamp(ts.iloc[end - 1])
        duration_min = max((end_ts - start_ts).total_seconds(), 0.0) / 60.0
        if duration_min < 3.0:
            debug["reason"] = "no_candidate_run"
            return out, debug

        motion_mean = 0.0
        if motion is not None and len(motion) >= end:
            motion_mean = float(np.nanmean(motion[start:end]))
        if motion is not None and motion_mean < 0.12 and hum_range < 3.5:
            debug["reason"] = "no_candidate_run"
            return out, debug

        out[start:end] = "shower"
        debug.update(
            {
                "applied": True,
                "reason": "humidity_peak_fallback",
                "start_index": int(start),
                "end_index": int(end),
                "details": {
                    "duration_minutes": float(duration_min),
                    "humidity_range": float(hum_range),
                    "motion_mean": float(motion_mean),
                    "peak_index": int(peak_idx),
                },
            }
        )
        return out, debug

    _, (start, end), details = best
    out[start:end] = "shower"
    debug.update(
        {
            "applied": True,
            "reason": "fallback_run_selected",
            "start_index": int(start),
            "end_index": int(end),
            "details": details,
        }
    )
    return out, debug


def run_backtest(
    *,
    data_dir: Path,
    elder_id: str,
    min_day: int,
    max_day: int,
    seed: int,
    occupancy_threshold: float,
    calibration_method: str,
    calib_fraction: float,
    min_calib_samples: int,
    min_calib_label_support: int,
    disable_threshold_tuning: bool,
    tune_rooms: set[str],
    room_occupancy_thresholds: Dict[str, float],
    room_label_thresholds: Dict[str, Dict[str, float]],
    critical_label_rescue_min_scores: Dict[str, Dict[str, float]],
    enable_duration_calibration: bool,
    enable_room_mae_threshold_tuning: bool,
    enable_adaptive_room_threshold_policy: bool,
    enable_kitchen_mae_threshold_tuning: bool,
    enable_kitchen_robust_duration_calibration: bool,
    enable_kitchen_temporal_decoder: bool,
    enable_bedroom_livingroom_boundary_reweighting: bool,
    enable_bedroom_livingroom_timeline_gates: bool,
    enable_bedroom_livingroom_occupancy_decoder: bool,
    enable_bedroom_livingroom_timeline_decoder_v2: bool,
    enable_bedroom_livingroom_hardgate_threshold_tuning: bool,
    enable_bedroom_livingroom_regime_routing: bool,
    bedroom_livingroom_low_occ_thresholds: Dict[str, float],
    enable_bedroom_livingroom_stage_a_hgb: bool,
    enable_bedroom_livingroom_hard_negative_mining: bool,
    bedroom_livingroom_hard_negative_weight: float,
    enable_bedroom_livingroom_failure_replay: bool,
    bedroom_livingroom_failure_replay_weight: float,
    bedroom_livingroom_max_replay_rows_per_day: int,
    livingroom_occupied_sample_weight: float = 1.0,
    enable_livingroom_passive_label_alignment: bool = False,
    livingroom_direct_positive_weight: float = 1.0,
    livingroom_passive_positive_weight: float = 0.25,
    livingroom_unoccupied_weight: float = 1.0,
    livingroom_direct_entry_exit_band_windows: int = 24,
    livingroom_direct_motion_threshold: float = 0.55,
    enable_kitchen_stage_a_reweighting: bool,
    hard_gate_room_metric_floors: Dict[str, Dict[str, float]],
    hard_gate_label_recall_floors: Dict[str, Dict[str, float]],
    hard_gate_label_recall_min_supports: Dict[str, int],
    hard_gate_fragmentation_min_run_windows: Dict[str, int],
    hard_gate_fragmentation_gap_fill_windows: Dict[str, int],
    hard_gate_min_train_days: int,
    enable_livingroom_cross_room_presence_decoder: bool = False,
    livingroom_cross_room_supporting_rooms: Sequence[str] = ("bedroom", "kitchen", "bathroom", "entrance"),
    livingroom_cross_room_hold_minutes: float = 10.0,
    livingroom_cross_room_max_extension_minutes: float = 24.0,
    livingroom_cross_room_entry_occ_threshold: float = 0.66,
    livingroom_cross_room_entry_room_prob_threshold: float = 0.42,
    livingroom_cross_room_entry_motion_threshold: float = 0.75,
    livingroom_cross_room_refresh_occ_threshold: float = 0.44,
    livingroom_cross_room_refresh_room_prob_threshold: float = 0.18,
    livingroom_cross_room_other_room_exit_occ_threshold: float = 0.62,
    livingroom_cross_room_other_room_exit_confirm_windows: int = 2,
    livingroom_cross_room_other_room_unoccupied_max_occ_prob: float = 0.30,
    livingroom_cross_room_entrance_exit_occ_threshold: float = 0.58,
    livingroom_cross_room_min_support_rooms: int = 2,
    livingroom_cross_room_require_other_room_predicted_occupied: bool = True,
    livingroom_night_bedroom_sleep_guard_enabled: bool = False,
    livingroom_night_bedroom_sleep_guard_start_hour: int = 22,
    livingroom_night_bedroom_sleep_guard_end_hour: int = 6,
    livingroom_night_bedroom_sleep_guard_bedroom_sleep_occ_threshold: float = 0.66,
    livingroom_night_bedroom_sleep_guard_bedroom_sleep_prob_threshold: float = 0.55,
    livingroom_night_bedroom_sleep_guard_bedroom_exit_occ_threshold: float = 0.35,
    livingroom_night_bedroom_sleep_guard_bedroom_exit_motion_threshold: float = 0.75,
    livingroom_night_bedroom_sleep_guard_entry_occ_threshold: float = 0.66,
    livingroom_night_bedroom_sleep_guard_entry_motion_threshold: float = 0.75,
    livingroom_night_bedroom_sleep_guard_entry_confirm_windows: int = 2,
    livingroom_night_bedroom_sleep_guard_suppression_label: str = "unknown",
    livingroom_night_bedroom_sleep_guard_min_coverage: float = 0.80,
    livingroom_night_bedroom_sleep_guard_flatline_motion_std_max: float = 1e-4,
    livingroom_night_bedroom_sleep_guard_flatline_occ_std_max: float = 1e-4,
    livingroom_night_bedroom_sleep_guard_flatline_min_windows: int = 2160,
    enable_single_resident_arbitration: bool,
    single_resident_rooms: Sequence[str],
    single_resident_min_margin: float,
    single_resident_bedroom_night_start_hour: int,
    single_resident_bedroom_night_end_hour: int,
    single_resident_bedroom_night_min_run_windows: int,
    single_resident_bedroom_night_min_label_score: float,
    single_resident_kitchen_min_label_score: float,
    single_resident_kitchen_guard_hours: Sequence[tuple[int, int]],
    single_resident_continuity_min_run_windows: int,
    single_resident_continuity_min_occ_prob: float,
    enable_room_temporal_occupancy_features: bool,
    enable_bedroom_light_texture_features: bool = False,
    bedroom_livingroom_texture_profile: str = "mixed",
    enable_bedroom_livingroom_segment_mode: bool = False,
    segment_min_duration_seconds: int = 60,
    segment_gap_merge_seconds: int = 30,
    segment_min_activity_prob: float = 0.35,
    segment_bedroom_min_duration_seconds: int = 30,
    segment_bedroom_gap_merge_seconds: int = 90,
    segment_bedroom_min_activity_prob: float = 0.30,
    segment_livingroom_min_duration_seconds: int = 60,
    segment_livingroom_gap_merge_seconds: int = 30,
    segment_livingroom_min_activity_prob: float = 0.40,
    segment_livingroom_max_occupied_ratio: float = 1.15,
    segment_livingroom_max_occupied_window_delta: int = 120,
    enable_bedroom_livingroom_segment_learned_classifier: bool = False,
    segment_classifier_min_segments: int = 8,
    segment_classifier_confidence_floor: float = 0.55,
    segment_classifier_min_windows: int = 6,
    enable_bedroom_livingroom_prediction_smoothing: bool = False,
    bedroom_livingroom_prediction_smoothing_min_run_windows: int = 9,
    bedroom_livingroom_prediction_smoothing_gap_fill_windows: int = 6,
    enable_bedroom_livingroom_passive_hysteresis: bool = False,
    bedroom_passive_hold_minutes: float = 45.0,
    livingroom_passive_hold_minutes: float = 30.0,
    passive_exit_min_consecutive_windows: int = 18,
    passive_entry_occ_threshold: float = 0.56,
    passive_entry_room_prob_threshold: float = 0.40,
    passive_stay_occ_threshold: float = 0.22,
    passive_stay_room_prob_threshold: float = 0.10,
    passive_exit_occ_threshold: float = 0.12,
    passive_exit_room_prob_threshold: float = 0.06,
    passive_motion_reset_threshold: float = 0.55,
    passive_motion_quiet_threshold: float = 0.12,
    livingroom_strict_entry_requires_strong_signal: bool = False,
    livingroom_entry_motion_threshold: float = 0.75,
    enable_bedroom_livingroom_stage_a_sequence_model: bool = False,
    bedroom_livingroom_stage_a_sequence_lag_windows: int = 12,
    enable_bedroom_livingroom_stage_a_minute_grid: bool = False,
    bedroom_livingroom_stage_a_minute_grid_rooms: Sequence[str] = ("bedroom", "livingroom"),
    bedroom_livingroom_stage_a_group_seconds: int = 60,
    bedroom_livingroom_stage_a_group_occupied_ratio_threshold: float = 0.50,
    enable_bedroom_livingroom_stage_a_transformer: bool = False,
    bedroom_livingroom_stage_a_transformer_epochs: int = 8,
    bedroom_livingroom_stage_a_transformer_batch_size: int = 256,
    bedroom_livingroom_stage_a_transformer_learning_rate: float = 7e-4,
    bedroom_livingroom_stage_a_transformer_hidden_dim: int = 48,
    bedroom_livingroom_stage_a_transformer_num_heads: int = 2,
    bedroom_livingroom_stage_a_transformer_dropout: float = 0.15,
    bedroom_livingroom_stage_a_transformer_class_weight_power: float = 0.5,
    bedroom_livingroom_stage_a_transformer_conv_kernel_size: int = 3,
    bedroom_livingroom_stage_a_transformer_conv_blocks: int = 2,
    enable_bedroom_livingroom_stage_a_transformer_sequence_filter: bool = True,
    enable_cross_room_context_features: bool = False,
    cross_room_context_rooms: Sequence[str] = ("bedroom", "livingroom"),
    label_corrections_csv: Optional[Path] = None,
) -> Dict[str, object]:
    files_by_day: Dict[int, Path] = {}
    candidate_files: List[Path] = []
    for ext in ("xlsx", "xls", "parquet"):
        candidate_files.extend(sorted(data_dir.glob(f"{elder_id}_train_*.{ext}")))

    canonical_files: List[Path] = []
    excluded_non_canonical: List[Dict[str, str]] = []
    invalid_day_token_files: List[str] = []
    prefix = f"{elder_id}_train_"
    for p in sorted({str(path.resolve()): path for path in candidate_files}.values(), key=lambda x: x.name.lower()):
        stem = p.stem
        if not stem.startswith(prefix):
            excluded_non_canonical.append({"file": str(p.name), "reason": "prefix_mismatch"})
            continue
        suffix = stem[len(prefix) :]
        # Keep only canonical train files like HKxxxx_name_train_8dec2025.
        # Excludes derivatives like *_occupied_only / *_manual_*.
        if "_" in suffix:
            excluded_non_canonical.append({"file": str(p.name), "reason": "derived_or_noncanonical_suffix"})
            continue
        canonical_files.append(p)

    for p in canonical_files:
        d = _extract_day(p)
        if d is None:
            invalid_day_token_files.append(str(p.name))
            continue
        if min_day <= d <= max_day:
            existing = files_by_day.get(d)
            if existing is not None and existing != p:
                raise ValueError(
                    "Ambiguous train-day token detected for backtest selection: "
                    f"day={d} maps to both '{existing.name}' and '{p.name}'. "
                    "Use a non-overlapping file set/day window."
                )
            files_by_day[d] = p

    days = sorted(files_by_day.keys())
    data_continuity_audit = _build_data_continuity_audit(
        elder_id=elder_id,
        min_day=int(min_day),
        max_day=int(max_day),
        candidate_files=candidate_files,
        canonical_files=canonical_files,
        files_by_day=files_by_day,
        excluded_non_canonical=excluded_non_canonical,
        invalid_day_token_files=invalid_day_token_files,
    )
    if len(days) < 2:
        raise ValueError(f"Need at least 2 days. Found days={days}")

    stage_a_minute_grid_rooms = {
        str(token).strip().lower() for token in list(bedroom_livingroom_stage_a_minute_grid_rooms or [])
    }
    stage_a_minute_grid_rooms = {room for room in stage_a_minute_grid_rooms if room in {"bedroom", "livingroom"}}
    if not stage_a_minute_grid_rooms:
        stage_a_minute_grid_rooms = {"bedroom", "livingroom"}

    room_day_data = _load_room_day_data(
        files_by_day,
        ROOMS,
        enable_cross_room_context_features=bool(enable_cross_room_context_features),
        cross_room_context_rooms=[
            str(r).strip()
            for r in cross_room_context_rooms
            if str(r).strip()
        ],
        enable_room_temporal_occupancy_features=bool(enable_room_temporal_occupancy_features),
        enable_bedroom_light_texture_features=bool(enable_bedroom_light_texture_features),
        bedroom_livingroom_texture_profile=str(bedroom_livingroom_texture_profile),
    )
    label_corrections, label_corrections_load_summary = _load_activity_label_corrections(label_corrections_csv)
    label_corrections_apply_summary = _apply_activity_label_corrections(
        room_day_data=room_day_data,
        corrections=label_corrections,
    )
    label_corrections_summary = {
        "load": dict(label_corrections_load_summary),
        "apply": dict(label_corrections_apply_summary),
    }
    splits = _build_splits(days)

    base_cfg = EventFirstConfig(random_state=seed)
    room_model_profiles: Dict[str, Dict[str, object]] = {}
    for key in ["default", "bedroom", "livingroom", "kitchen", "bathroom", "entrance"]:
        cfg = _build_room_model_config(
            seed=seed,
            room_key=key if key != "default" else "",
            bedroom_livingroom_regime="normal",
            enable_bedroom_livingroom_regime_routing=bool(enable_bedroom_livingroom_regime_routing),
            enable_bedroom_livingroom_stage_a_hgb=bool(enable_bedroom_livingroom_stage_a_hgb),
            enable_bedroom_livingroom_stage_a_sequence_model=bool(
                enable_bedroom_livingroom_stage_a_sequence_model
            ),
            bedroom_livingroom_stage_a_sequence_lag_windows=int(
                max(bedroom_livingroom_stage_a_sequence_lag_windows, 1)
            ),
            enable_bedroom_livingroom_stage_a_transformer=bool(
                enable_bedroom_livingroom_stage_a_transformer
            ),
            bedroom_livingroom_stage_a_transformer_epochs=int(
                max(bedroom_livingroom_stage_a_transformer_epochs, 1)
            ),
            bedroom_livingroom_stage_a_transformer_batch_size=int(
                max(bedroom_livingroom_stage_a_transformer_batch_size, 1)
            ),
            bedroom_livingroom_stage_a_transformer_learning_rate=float(
                max(bedroom_livingroom_stage_a_transformer_learning_rate, 1e-5)
            ),
            bedroom_livingroom_stage_a_transformer_hidden_dim=int(
                max(bedroom_livingroom_stage_a_transformer_hidden_dim, 8)
            ),
            bedroom_livingroom_stage_a_transformer_num_heads=int(
                max(bedroom_livingroom_stage_a_transformer_num_heads, 1)
            ),
            bedroom_livingroom_stage_a_transformer_dropout=float(
                min(max(bedroom_livingroom_stage_a_transformer_dropout, 0.0), 0.5)
            ),
            bedroom_livingroom_stage_a_transformer_class_weight_power=float(
                min(max(bedroom_livingroom_stage_a_transformer_class_weight_power, 0.0), 1.0)
            ),
            bedroom_livingroom_stage_a_transformer_conv_kernel_size=int(
                max(bedroom_livingroom_stage_a_transformer_conv_kernel_size, 2)
            ),
            bedroom_livingroom_stage_a_transformer_conv_blocks=int(
                max(bedroom_livingroom_stage_a_transformer_conv_blocks, 1)
            ),
            enable_bedroom_livingroom_stage_a_transformer_sequence_filter=bool(
                enable_bedroom_livingroom_stage_a_transformer_sequence_filter
            ),
        )
        room_model_profiles[key] = {
            "n_estimators_stage_a": int(cfg.n_estimators_stage_a),
            "n_estimators_stage_b": int(cfg.n_estimators_stage_b),
            "min_samples_leaf": int(cfg.min_samples_leaf),
            "stage_a_class_weight": cfg.stage_a_class_weight,
            "stage_b_class_weight": cfg.stage_b_class_weight,
            "stage_a_model_type": str(cfg.stage_a_model_type),
            "stage_a_temporal_lag_windows": int(cfg.stage_a_temporal_lag_windows),
            "stage_a_transformer_epochs": int(cfg.stage_a_transformer_epochs),
            "stage_a_transformer_batch_size": int(cfg.stage_a_transformer_batch_size),
            "stage_a_transformer_learning_rate": float(cfg.stage_a_transformer_learning_rate),
            "stage_a_transformer_hidden_dim": int(cfg.stage_a_transformer_hidden_dim),
            "stage_a_transformer_num_heads": int(cfg.stage_a_transformer_num_heads),
            "stage_a_transformer_dropout": float(cfg.stage_a_transformer_dropout),
            "stage_a_transformer_class_weight_power": float(cfg.stage_a_transformer_class_weight_power),
            "stage_a_transformer_conv_kernel_size": int(cfg.stage_a_transformer_conv_kernel_size),
            "stage_a_transformer_conv_blocks": int(cfg.stage_a_transformer_conv_blocks),
            "stage_a_transformer_use_sequence_filter": bool(cfg.stage_a_transformer_use_sequence_filter),
        }
    config_payload = {
        "seed": int(seed),
        "occupancy_threshold": float(occupancy_threshold),
        "calibration_method": str(calibration_method),
        "calib_fraction": float(calib_fraction),
        "min_calib_samples": int(min_calib_samples),
        "min_calib_label_support": int(min_calib_label_support),
        "disable_threshold_tuning": bool(disable_threshold_tuning),
        "tune_rooms": sorted(list(tune_rooms)),
        "room_occupancy_thresholds": dict(room_occupancy_thresholds),
        "room_label_thresholds": dict(room_label_thresholds),
        "critical_label_rescue_min_scores": dict(critical_label_rescue_min_scores),
        "enable_duration_calibration": bool(enable_duration_calibration),
        "enable_room_mae_threshold_tuning": bool(enable_room_mae_threshold_tuning),
        "enable_adaptive_room_threshold_policy": bool(enable_adaptive_room_threshold_policy),
        "enable_kitchen_mae_threshold_tuning": bool(enable_kitchen_mae_threshold_tuning),
        "enable_kitchen_robust_duration_calibration": bool(enable_kitchen_robust_duration_calibration),
        "enable_kitchen_temporal_decoder": bool(enable_kitchen_temporal_decoder),
        "enable_bedroom_livingroom_boundary_reweighting": bool(enable_bedroom_livingroom_boundary_reweighting),
        "enable_bedroom_livingroom_timeline_gates": bool(enable_bedroom_livingroom_timeline_gates),
        "enable_bedroom_livingroom_occupancy_decoder": bool(enable_bedroom_livingroom_occupancy_decoder),
        "enable_bedroom_livingroom_timeline_decoder_v2": bool(enable_bedroom_livingroom_timeline_decoder_v2),
        "enable_bedroom_livingroom_hardgate_threshold_tuning": bool(
            enable_bedroom_livingroom_hardgate_threshold_tuning
        ),
        "enable_bedroom_livingroom_regime_routing": bool(enable_bedroom_livingroom_regime_routing),
        "bedroom_livingroom_low_occ_thresholds": dict(bedroom_livingroom_low_occ_thresholds),
        "enable_bedroom_livingroom_stage_a_hgb": bool(enable_bedroom_livingroom_stage_a_hgb),
        "enable_bedroom_livingroom_hard_negative_mining": bool(enable_bedroom_livingroom_hard_negative_mining),
        "bedroom_livingroom_hard_negative_weight": float(bedroom_livingroom_hard_negative_weight),
        "enable_bedroom_livingroom_failure_replay": bool(enable_bedroom_livingroom_failure_replay),
        "bedroom_livingroom_failure_replay_weight": float(bedroom_livingroom_failure_replay_weight),
        "bedroom_livingroom_max_replay_rows_per_day": int(bedroom_livingroom_max_replay_rows_per_day),
        "livingroom_occupied_sample_weight": float(max(livingroom_occupied_sample_weight, 1.0)),
        "enable_livingroom_passive_label_alignment": bool(enable_livingroom_passive_label_alignment),
        "livingroom_direct_positive_weight": float(max(livingroom_direct_positive_weight, 0.05)),
        "livingroom_passive_positive_weight": float(max(livingroom_passive_positive_weight, 0.01)),
        "livingroom_unoccupied_weight": float(max(livingroom_unoccupied_weight, 0.05)),
        "livingroom_direct_entry_exit_band_windows": int(max(livingroom_direct_entry_exit_band_windows, 0)),
        "livingroom_direct_motion_threshold": float(max(livingroom_direct_motion_threshold, 0.0)),
        "enable_kitchen_stage_a_reweighting": bool(enable_kitchen_stage_a_reweighting),
        "hard_gate_room_metric_floors": dict(hard_gate_room_metric_floors),
        "hard_gate_label_recall_floors": dict(hard_gate_label_recall_floors),
        "hard_gate_label_recall_min_supports": dict(hard_gate_label_recall_min_supports),
        "hard_gate_fragmentation_min_run_windows": dict(hard_gate_fragmentation_min_run_windows),
        "hard_gate_fragmentation_gap_fill_windows": dict(hard_gate_fragmentation_gap_fill_windows),
        "hard_gate_min_train_days": int(max(hard_gate_min_train_days, 0)),
        "enable_livingroom_cross_room_presence_decoder": bool(enable_livingroom_cross_room_presence_decoder),
        "livingroom_cross_room_supporting_rooms": sorted(
            [str(r).strip().lower() for r in livingroom_cross_room_supporting_rooms if str(r).strip()]
        ),
        "livingroom_cross_room_hold_minutes": float(max(livingroom_cross_room_hold_minutes, 0.0)),
        "livingroom_cross_room_max_extension_minutes": float(
            max(livingroom_cross_room_max_extension_minutes, 0.0)
        ),
        "livingroom_cross_room_entry_occ_threshold": float(
            min(max(livingroom_cross_room_entry_occ_threshold, 0.0), 1.0)
        ),
        "livingroom_cross_room_entry_room_prob_threshold": float(
            min(max(livingroom_cross_room_entry_room_prob_threshold, 0.0), 1.0)
        ),
        "livingroom_cross_room_entry_motion_threshold": float(
            max(livingroom_cross_room_entry_motion_threshold, 0.0)
        ),
        "livingroom_cross_room_refresh_occ_threshold": float(
            min(max(livingroom_cross_room_refresh_occ_threshold, 0.0), 1.0)
        ),
        "livingroom_cross_room_refresh_room_prob_threshold": float(
            min(max(livingroom_cross_room_refresh_room_prob_threshold, 0.0), 1.0)
        ),
        "livingroom_cross_room_other_room_exit_occ_threshold": float(
            min(max(livingroom_cross_room_other_room_exit_occ_threshold, 0.0), 1.0)
        ),
        "livingroom_cross_room_other_room_exit_confirm_windows": int(
            max(livingroom_cross_room_other_room_exit_confirm_windows, 1)
        ),
        "livingroom_cross_room_other_room_unoccupied_max_occ_prob": float(
            min(max(livingroom_cross_room_other_room_unoccupied_max_occ_prob, 0.0), 1.0)
        ),
        "livingroom_cross_room_entrance_exit_occ_threshold": float(
            min(max(livingroom_cross_room_entrance_exit_occ_threshold, 0.0), 1.0)
        ),
        "livingroom_cross_room_min_support_rooms": int(max(livingroom_cross_room_min_support_rooms, 1)),
        "livingroom_cross_room_require_other_room_predicted_occupied": bool(
            livingroom_cross_room_require_other_room_predicted_occupied
        ),
        "livingroom_night_bedroom_sleep_guard_enabled": bool(
            livingroom_night_bedroom_sleep_guard_enabled
        ),
        "livingroom_night_bedroom_sleep_guard_start_hour": int(
            livingroom_night_bedroom_sleep_guard_start_hour
        ),
        "livingroom_night_bedroom_sleep_guard_end_hour": int(
            livingroom_night_bedroom_sleep_guard_end_hour
        ),
        "livingroom_night_bedroom_sleep_guard_bedroom_sleep_occ_threshold": float(
            livingroom_night_bedroom_sleep_guard_bedroom_sleep_occ_threshold
        ),
        "livingroom_night_bedroom_sleep_guard_bedroom_sleep_prob_threshold": float(
            livingroom_night_bedroom_sleep_guard_bedroom_sleep_prob_threshold
        ),
        "livingroom_night_bedroom_sleep_guard_bedroom_exit_occ_threshold": float(
            livingroom_night_bedroom_sleep_guard_bedroom_exit_occ_threshold
        ),
        "livingroom_night_bedroom_sleep_guard_bedroom_exit_motion_threshold": float(
            livingroom_night_bedroom_sleep_guard_bedroom_exit_motion_threshold
        ),
        "livingroom_night_bedroom_sleep_guard_entry_occ_threshold": float(
            livingroom_night_bedroom_sleep_guard_entry_occ_threshold
        ),
        "livingroom_night_bedroom_sleep_guard_entry_motion_threshold": float(
            livingroom_night_bedroom_sleep_guard_entry_motion_threshold
        ),
        "livingroom_night_bedroom_sleep_guard_entry_confirm_windows": int(
            livingroom_night_bedroom_sleep_guard_entry_confirm_windows
        ),
        "livingroom_night_bedroom_sleep_guard_suppression_label": str(
            livingroom_night_bedroom_sleep_guard_suppression_label
        ),
        "livingroom_night_bedroom_sleep_guard_min_coverage": float(
            livingroom_night_bedroom_sleep_guard_min_coverage
        ),
        "livingroom_night_bedroom_sleep_guard_flatline_motion_std_max": float(
            livingroom_night_bedroom_sleep_guard_flatline_motion_std_max
        ),
        "livingroom_night_bedroom_sleep_guard_flatline_occ_std_max": float(
            livingroom_night_bedroom_sleep_guard_flatline_occ_std_max
        ),
        "livingroom_night_bedroom_sleep_guard_flatline_min_windows": int(
            livingroom_night_bedroom_sleep_guard_flatline_min_windows
        ),
        "room_model_profiles": room_model_profiles,
        "enable_single_resident_arbitration": bool(enable_single_resident_arbitration),
        "single_resident_rooms": sorted([str(r).strip().lower() for r in single_resident_rooms if str(r).strip()]),
        "single_resident_min_margin": float(single_resident_min_margin),
        "single_resident_bedroom_night_start_hour": int(single_resident_bedroom_night_start_hour),
        "single_resident_bedroom_night_end_hour": int(single_resident_bedroom_night_end_hour),
        "single_resident_bedroom_night_min_run_windows": int(single_resident_bedroom_night_min_run_windows),
        "single_resident_bedroom_night_min_label_score": float(single_resident_bedroom_night_min_label_score),
        "single_resident_kitchen_min_label_score": float(single_resident_kitchen_min_label_score),
        "single_resident_kitchen_guard_hours": [[int(a), int(b)] for a, b in single_resident_kitchen_guard_hours],
        "single_resident_continuity_min_run_windows": int(single_resident_continuity_min_run_windows),
        "single_resident_continuity_min_occ_prob": float(single_resident_continuity_min_occ_prob),
        "enable_room_temporal_occupancy_features": bool(enable_room_temporal_occupancy_features),
        "enable_bedroom_light_texture_features": bool(enable_bedroom_light_texture_features),
        "bedroom_livingroom_texture_profile": str(bedroom_livingroom_texture_profile),
        "enable_bedroom_livingroom_segment_mode": bool(enable_bedroom_livingroom_segment_mode),
        "segment_min_duration_seconds": int(max(segment_min_duration_seconds, 10)),
        "segment_gap_merge_seconds": int(max(segment_gap_merge_seconds, 0)),
        "segment_min_activity_prob": float(min(max(segment_min_activity_prob, 0.0), 1.0)),
        "segment_bedroom_min_duration_seconds": int(max(segment_bedroom_min_duration_seconds, 10)),
        "segment_bedroom_gap_merge_seconds": int(max(segment_bedroom_gap_merge_seconds, 0)),
        "segment_bedroom_min_activity_prob": float(min(max(segment_bedroom_min_activity_prob, 0.0), 1.0)),
        "segment_livingroom_min_duration_seconds": int(max(segment_livingroom_min_duration_seconds, 10)),
        "segment_livingroom_gap_merge_seconds": int(max(segment_livingroom_gap_merge_seconds, 0)),
        "segment_livingroom_min_activity_prob": float(min(max(segment_livingroom_min_activity_prob, 0.0), 1.0)),
        "segment_livingroom_max_occupied_ratio": float(max(segment_livingroom_max_occupied_ratio, 1.0)),
        "segment_livingroom_max_occupied_window_delta": int(max(segment_livingroom_max_occupied_window_delta, 0)),
        "enable_bedroom_livingroom_segment_learned_classifier": bool(
            enable_bedroom_livingroom_segment_learned_classifier
        ),
        "segment_classifier_min_segments": int(max(segment_classifier_min_segments, 2)),
        "segment_classifier_confidence_floor": float(min(max(segment_classifier_confidence_floor, 0.0), 1.0)),
        "segment_classifier_min_windows": int(max(segment_classifier_min_windows, 1)),
        "enable_bedroom_livingroom_prediction_smoothing": bool(enable_bedroom_livingroom_prediction_smoothing),
        "bedroom_livingroom_prediction_smoothing_min_run_windows": int(
            max(bedroom_livingroom_prediction_smoothing_min_run_windows, 1)
        ),
        "bedroom_livingroom_prediction_smoothing_gap_fill_windows": int(
            max(bedroom_livingroom_prediction_smoothing_gap_fill_windows, 0)
        ),
        "enable_bedroom_livingroom_passive_hysteresis": bool(enable_bedroom_livingroom_passive_hysteresis),
        "bedroom_passive_hold_minutes": float(max(bedroom_passive_hold_minutes, 0.0)),
        "livingroom_passive_hold_minutes": float(max(livingroom_passive_hold_minutes, 0.0)),
        "passive_exit_min_consecutive_windows": int(max(passive_exit_min_consecutive_windows, 1)),
        "passive_entry_occ_threshold": float(min(max(passive_entry_occ_threshold, 0.0), 1.0)),
        "passive_entry_room_prob_threshold": float(min(max(passive_entry_room_prob_threshold, 0.0), 1.0)),
        "passive_stay_occ_threshold": float(min(max(passive_stay_occ_threshold, 0.0), 1.0)),
        "passive_stay_room_prob_threshold": float(min(max(passive_stay_room_prob_threshold, 0.0), 1.0)),
        "passive_exit_occ_threshold": float(min(max(passive_exit_occ_threshold, 0.0), 1.0)),
        "passive_exit_room_prob_threshold": float(min(max(passive_exit_room_prob_threshold, 0.0), 1.0)),
        "passive_motion_reset_threshold": float(max(passive_motion_reset_threshold, 0.0)),
        "passive_motion_quiet_threshold": float(max(passive_motion_quiet_threshold, 0.0)),
        "livingroom_strict_entry_requires_strong_signal": bool(
            livingroom_strict_entry_requires_strong_signal
        ),
        "livingroom_entry_motion_threshold": float(max(livingroom_entry_motion_threshold, 0.0)),
        "enable_bedroom_livingroom_stage_a_sequence_model": bool(
            enable_bedroom_livingroom_stage_a_sequence_model
        ),
        "bedroom_livingroom_stage_a_sequence_lag_windows": int(
            max(bedroom_livingroom_stage_a_sequence_lag_windows, 1)
        ),
        "enable_bedroom_livingroom_stage_a_minute_grid": bool(
            enable_bedroom_livingroom_stage_a_minute_grid
        ),
        "bedroom_livingroom_stage_a_minute_grid_rooms": sorted(stage_a_minute_grid_rooms),
        "bedroom_livingroom_stage_a_group_seconds": int(
            max(bedroom_livingroom_stage_a_group_seconds, 10)
        ),
        "bedroom_livingroom_stage_a_group_occupied_ratio_threshold": float(
            min(max(bedroom_livingroom_stage_a_group_occupied_ratio_threshold, 0.0), 1.0)
        ),
        "enable_bedroom_livingroom_stage_a_transformer": bool(
            enable_bedroom_livingroom_stage_a_transformer
        ),
        "bedroom_livingroom_stage_a_transformer_epochs": int(
            max(bedroom_livingroom_stage_a_transformer_epochs, 1)
        ),
        "bedroom_livingroom_stage_a_transformer_batch_size": int(
            max(bedroom_livingroom_stage_a_transformer_batch_size, 1)
        ),
        "bedroom_livingroom_stage_a_transformer_learning_rate": float(
            max(bedroom_livingroom_stage_a_transformer_learning_rate, 1e-5)
        ),
        "bedroom_livingroom_stage_a_transformer_hidden_dim": int(
            max(bedroom_livingroom_stage_a_transformer_hidden_dim, 8)
        ),
        "bedroom_livingroom_stage_a_transformer_num_heads": int(
            max(bedroom_livingroom_stage_a_transformer_num_heads, 1)
        ),
        "bedroom_livingroom_stage_a_transformer_dropout": float(
            min(max(bedroom_livingroom_stage_a_transformer_dropout, 0.0), 0.5)
        ),
        "bedroom_livingroom_stage_a_transformer_class_weight_power": float(
            min(max(bedroom_livingroom_stage_a_transformer_class_weight_power, 0.0), 1.0)
        ),
        "bedroom_livingroom_stage_a_transformer_conv_kernel_size": int(
            max(bedroom_livingroom_stage_a_transformer_conv_kernel_size, 2)
        ),
        "bedroom_livingroom_stage_a_transformer_conv_blocks": int(
            max(bedroom_livingroom_stage_a_transformer_conv_blocks, 1)
        ),
        "enable_bedroom_livingroom_stage_a_transformer_sequence_filter": bool(
            enable_bedroom_livingroom_stage_a_transformer_sequence_filter
        ),
        "enable_cross_room_context_features": bool(enable_cross_room_context_features),
        "cross_room_context_rooms": sorted(
            [str(r).strip().lower() for r in cross_room_context_rooms if str(r).strip()]
        ),
        "label_corrections_csv": str(label_corrections_csv) if label_corrections_csv is not None else None,
    }

    report: Dict[str, object] = {
        "elder_id": elder_id,
        "run_timestamp_utc": _utc_now_iso_z(),
        "git_sha": _safe_git_sha(),
        "config_hash": _config_hash(config_payload),
        "data_version": f"dec{min_day}_to_dec{max_day}",
        "feature_schema_hash": f"sha256:{hashlib.sha256(','.join(SENSOR_COLS).encode('utf-8')).hexdigest()}",
        "seed": int(seed),
        "days": days,
        "data_continuity_audit": data_continuity_audit,
        "label_corrections": label_corrections_summary,
        "splits": [],
        "summary": {},
        "classification_summary": {},
        "gate_summary": {},
        "leakage_checklist": [
            "temporal_train_fit_then_calib_then_test",
            "fit/calib/test chronological order per split-room",
            "no test-day rows used for fit/calibration",
        ],
        "calibration_summary": {
            "method": str(calibration_method),
            "threshold_tuning_enabled": bool(not disable_threshold_tuning),
            "tune_rooms": sorted(list(tune_rooms)),
            "room_occupancy_thresholds": dict(room_occupancy_thresholds),
            "room_label_thresholds": dict(room_label_thresholds),
            "critical_label_rescue_min_scores": dict(critical_label_rescue_min_scores),
            "enable_duration_calibration": bool(enable_duration_calibration),
            "enable_room_mae_threshold_tuning": bool(enable_room_mae_threshold_tuning),
            "enable_adaptive_room_threshold_policy": bool(enable_adaptive_room_threshold_policy),
            "enable_kitchen_mae_threshold_tuning": bool(enable_kitchen_mae_threshold_tuning),
            "enable_kitchen_robust_duration_calibration": bool(enable_kitchen_robust_duration_calibration),
            "enable_kitchen_temporal_decoder": bool(enable_kitchen_temporal_decoder),
            "enable_bedroom_livingroom_boundary_reweighting": bool(
                enable_bedroom_livingroom_boundary_reweighting
            ),
            "enable_bedroom_livingroom_timeline_gates": bool(enable_bedroom_livingroom_timeline_gates),
            "enable_bedroom_livingroom_occupancy_decoder": bool(enable_bedroom_livingroom_occupancy_decoder),
            "enable_bedroom_livingroom_timeline_decoder_v2": bool(enable_bedroom_livingroom_timeline_decoder_v2),
            "enable_bedroom_livingroom_hardgate_threshold_tuning": bool(
                enable_bedroom_livingroom_hardgate_threshold_tuning
            ),
            "enable_bedroom_livingroom_regime_routing": bool(enable_bedroom_livingroom_regime_routing),
            "bedroom_livingroom_low_occ_thresholds": dict(bedroom_livingroom_low_occ_thresholds),
            "enable_bedroom_livingroom_stage_a_hgb": bool(enable_bedroom_livingroom_stage_a_hgb),
            "enable_bedroom_livingroom_hard_negative_mining": bool(
                enable_bedroom_livingroom_hard_negative_mining
            ),
            "bedroom_livingroom_hard_negative_weight": float(bedroom_livingroom_hard_negative_weight),
            "enable_bedroom_livingroom_failure_replay": bool(enable_bedroom_livingroom_failure_replay),
            "bedroom_livingroom_failure_replay_weight": float(bedroom_livingroom_failure_replay_weight),
            "bedroom_livingroom_max_replay_rows_per_day": int(bedroom_livingroom_max_replay_rows_per_day),
            "livingroom_occupied_sample_weight": float(max(livingroom_occupied_sample_weight, 1.0)),
            "enable_livingroom_passive_label_alignment": bool(enable_livingroom_passive_label_alignment),
            "livingroom_direct_positive_weight": float(max(livingroom_direct_positive_weight, 0.05)),
            "livingroom_passive_positive_weight": float(max(livingroom_passive_positive_weight, 0.01)),
            "livingroom_unoccupied_weight": float(max(livingroom_unoccupied_weight, 0.05)),
            "livingroom_direct_entry_exit_band_windows": int(max(livingroom_direct_entry_exit_band_windows, 0)),
            "livingroom_direct_motion_threshold": float(max(livingroom_direct_motion_threshold, 0.0)),
            "enable_kitchen_stage_a_reweighting": bool(enable_kitchen_stage_a_reweighting),
            "hard_gate_room_metric_floors": dict(hard_gate_room_metric_floors),
            "hard_gate_label_recall_floors": dict(hard_gate_label_recall_floors),
            "hard_gate_label_recall_min_supports": dict(hard_gate_label_recall_min_supports),
            "hard_gate_fragmentation_min_run_windows": dict(hard_gate_fragmentation_min_run_windows),
            "hard_gate_fragmentation_gap_fill_windows": dict(hard_gate_fragmentation_gap_fill_windows),
            "hard_gate_min_train_days": int(max(hard_gate_min_train_days, 0)),
            "enable_livingroom_cross_room_presence_decoder": bool(
                enable_livingroom_cross_room_presence_decoder
            ),
            "livingroom_cross_room_supporting_rooms": sorted(
                [str(r).strip().lower() for r in livingroom_cross_room_supporting_rooms if str(r).strip()]
            ),
            "livingroom_cross_room_hold_minutes": float(max(livingroom_cross_room_hold_minutes, 0.0)),
            "livingroom_cross_room_max_extension_minutes": float(
                max(livingroom_cross_room_max_extension_minutes, 0.0)
            ),
            "livingroom_cross_room_entry_occ_threshold": float(
                min(max(livingroom_cross_room_entry_occ_threshold, 0.0), 1.0)
            ),
            "livingroom_cross_room_entry_room_prob_threshold": float(
                min(max(livingroom_cross_room_entry_room_prob_threshold, 0.0), 1.0)
            ),
            "livingroom_cross_room_entry_motion_threshold": float(
                max(livingroom_cross_room_entry_motion_threshold, 0.0)
            ),
            "livingroom_cross_room_refresh_occ_threshold": float(
                min(max(livingroom_cross_room_refresh_occ_threshold, 0.0), 1.0)
            ),
            "livingroom_cross_room_refresh_room_prob_threshold": float(
                min(max(livingroom_cross_room_refresh_room_prob_threshold, 0.0), 1.0)
            ),
            "livingroom_cross_room_other_room_exit_occ_threshold": float(
                min(max(livingroom_cross_room_other_room_exit_occ_threshold, 0.0), 1.0)
            ),
            "livingroom_cross_room_other_room_exit_confirm_windows": int(
                max(livingroom_cross_room_other_room_exit_confirm_windows, 1)
            ),
            "livingroom_cross_room_other_room_unoccupied_max_occ_prob": float(
                min(max(livingroom_cross_room_other_room_unoccupied_max_occ_prob, 0.0), 1.0)
            ),
            "livingroom_cross_room_entrance_exit_occ_threshold": float(
                min(max(livingroom_cross_room_entrance_exit_occ_threshold, 0.0), 1.0)
            ),
            "livingroom_cross_room_min_support_rooms": int(max(livingroom_cross_room_min_support_rooms, 1)),
            "livingroom_cross_room_require_other_room_predicted_occupied": bool(
                livingroom_cross_room_require_other_room_predicted_occupied
            ),
            "livingroom_night_bedroom_sleep_guard_enabled": bool(
                livingroom_night_bedroom_sleep_guard_enabled
            ),
            "livingroom_night_bedroom_sleep_guard_start_hour": int(
                livingroom_night_bedroom_sleep_guard_start_hour
            ),
            "livingroom_night_bedroom_sleep_guard_end_hour": int(
                livingroom_night_bedroom_sleep_guard_end_hour
            ),
            "livingroom_night_bedroom_sleep_guard_bedroom_sleep_occ_threshold": float(
                livingroom_night_bedroom_sleep_guard_bedroom_sleep_occ_threshold
            ),
            "livingroom_night_bedroom_sleep_guard_bedroom_sleep_prob_threshold": float(
                livingroom_night_bedroom_sleep_guard_bedroom_sleep_prob_threshold
            ),
            "livingroom_night_bedroom_sleep_guard_bedroom_exit_occ_threshold": float(
                livingroom_night_bedroom_sleep_guard_bedroom_exit_occ_threshold
            ),
            "livingroom_night_bedroom_sleep_guard_bedroom_exit_motion_threshold": float(
                livingroom_night_bedroom_sleep_guard_bedroom_exit_motion_threshold
            ),
            "livingroom_night_bedroom_sleep_guard_entry_occ_threshold": float(
                livingroom_night_bedroom_sleep_guard_entry_occ_threshold
            ),
            "livingroom_night_bedroom_sleep_guard_entry_motion_threshold": float(
                livingroom_night_bedroom_sleep_guard_entry_motion_threshold
            ),
            "livingroom_night_bedroom_sleep_guard_entry_confirm_windows": int(
                livingroom_night_bedroom_sleep_guard_entry_confirm_windows
            ),
            "livingroom_night_bedroom_sleep_guard_suppression_label": str(
                livingroom_night_bedroom_sleep_guard_suppression_label
            ),
            "livingroom_night_bedroom_sleep_guard_min_coverage": float(
                livingroom_night_bedroom_sleep_guard_min_coverage
            ),
            "livingroom_night_bedroom_sleep_guard_flatline_motion_std_max": float(
                livingroom_night_bedroom_sleep_guard_flatline_motion_std_max
            ),
            "livingroom_night_bedroom_sleep_guard_flatline_occ_std_max": float(
                livingroom_night_bedroom_sleep_guard_flatline_occ_std_max
            ),
            "livingroom_night_bedroom_sleep_guard_flatline_min_windows": int(
                livingroom_night_bedroom_sleep_guard_flatline_min_windows
            ),
            "enable_single_resident_arbitration": bool(enable_single_resident_arbitration),
            "single_resident_rooms": sorted([str(r).strip().lower() for r in single_resident_rooms if str(r).strip()]),
            "single_resident_min_margin": float(single_resident_min_margin),
            "single_resident_bedroom_night_start_hour": int(single_resident_bedroom_night_start_hour),
            "single_resident_bedroom_night_end_hour": int(single_resident_bedroom_night_end_hour),
            "single_resident_bedroom_night_min_run_windows": int(single_resident_bedroom_night_min_run_windows),
            "single_resident_bedroom_night_min_label_score": float(single_resident_bedroom_night_min_label_score),
            "single_resident_kitchen_min_label_score": float(single_resident_kitchen_min_label_score),
            "single_resident_kitchen_guard_hours": [
                [int(a), int(b)] for a, b in single_resident_kitchen_guard_hours
            ],
            "single_resident_continuity_min_run_windows": int(single_resident_continuity_min_run_windows),
            "single_resident_continuity_min_occ_prob": float(single_resident_continuity_min_occ_prob),
            "enable_room_temporal_occupancy_features": bool(enable_room_temporal_occupancy_features),
            "enable_bedroom_light_texture_features": bool(enable_bedroom_light_texture_features),
            "bedroom_livingroom_texture_profile": str(bedroom_livingroom_texture_profile),
            "enable_bedroom_livingroom_segment_mode": bool(enable_bedroom_livingroom_segment_mode),
            "segment_min_duration_seconds": int(max(segment_min_duration_seconds, 10)),
            "segment_gap_merge_seconds": int(max(segment_gap_merge_seconds, 0)),
            "segment_min_activity_prob": float(min(max(segment_min_activity_prob, 0.0), 1.0)),
            "segment_bedroom_min_duration_seconds": int(max(segment_bedroom_min_duration_seconds, 10)),
            "segment_bedroom_gap_merge_seconds": int(max(segment_bedroom_gap_merge_seconds, 0)),
            "segment_bedroom_min_activity_prob": float(min(max(segment_bedroom_min_activity_prob, 0.0), 1.0)),
            "segment_livingroom_min_duration_seconds": int(max(segment_livingroom_min_duration_seconds, 10)),
            "segment_livingroom_gap_merge_seconds": int(max(segment_livingroom_gap_merge_seconds, 0)),
            "segment_livingroom_min_activity_prob": float(min(max(segment_livingroom_min_activity_prob, 0.0), 1.0)),
            "segment_livingroom_max_occupied_ratio": float(max(segment_livingroom_max_occupied_ratio, 1.0)),
            "segment_livingroom_max_occupied_window_delta": int(max(segment_livingroom_max_occupied_window_delta, 0)),
            "enable_bedroom_livingroom_segment_learned_classifier": bool(
                enable_bedroom_livingroom_segment_learned_classifier
            ),
            "segment_classifier_min_segments": int(max(segment_classifier_min_segments, 2)),
            "segment_classifier_confidence_floor": float(min(max(segment_classifier_confidence_floor, 0.0), 1.0)),
            "segment_classifier_min_windows": int(max(segment_classifier_min_windows, 1)),
            "enable_bedroom_livingroom_prediction_smoothing": bool(enable_bedroom_livingroom_prediction_smoothing),
            "bedroom_livingroom_prediction_smoothing_min_run_windows": int(
                max(bedroom_livingroom_prediction_smoothing_min_run_windows, 1)
            ),
            "bedroom_livingroom_prediction_smoothing_gap_fill_windows": int(
                max(bedroom_livingroom_prediction_smoothing_gap_fill_windows, 0)
            ),
            "enable_bedroom_livingroom_passive_hysteresis": bool(enable_bedroom_livingroom_passive_hysteresis),
            "bedroom_passive_hold_minutes": float(max(bedroom_passive_hold_minutes, 0.0)),
            "livingroom_passive_hold_minutes": float(max(livingroom_passive_hold_minutes, 0.0)),
            "passive_exit_min_consecutive_windows": int(max(passive_exit_min_consecutive_windows, 1)),
            "passive_entry_occ_threshold": float(min(max(passive_entry_occ_threshold, 0.0), 1.0)),
            "passive_entry_room_prob_threshold": float(min(max(passive_entry_room_prob_threshold, 0.0), 1.0)),
            "passive_stay_occ_threshold": float(min(max(passive_stay_occ_threshold, 0.0), 1.0)),
            "passive_stay_room_prob_threshold": float(min(max(passive_stay_room_prob_threshold, 0.0), 1.0)),
            "passive_exit_occ_threshold": float(min(max(passive_exit_occ_threshold, 0.0), 1.0)),
            "passive_exit_room_prob_threshold": float(min(max(passive_exit_room_prob_threshold, 0.0), 1.0)),
            "passive_motion_reset_threshold": float(max(passive_motion_reset_threshold, 0.0)),
            "passive_motion_quiet_threshold": float(max(passive_motion_quiet_threshold, 0.0)),
            "livingroom_strict_entry_requires_strong_signal": bool(
                livingroom_strict_entry_requires_strong_signal
            ),
            "livingroom_entry_motion_threshold": float(max(livingroom_entry_motion_threshold, 0.0)),
            "enable_bedroom_livingroom_stage_a_sequence_model": bool(
                enable_bedroom_livingroom_stage_a_sequence_model
            ),
            "bedroom_livingroom_stage_a_sequence_lag_windows": int(
                max(bedroom_livingroom_stage_a_sequence_lag_windows, 1)
            ),
            "enable_bedroom_livingroom_stage_a_minute_grid": bool(
                enable_bedroom_livingroom_stage_a_minute_grid
            ),
            "bedroom_livingroom_stage_a_minute_grid_rooms": sorted(stage_a_minute_grid_rooms),
            "bedroom_livingroom_stage_a_group_seconds": int(
                max(bedroom_livingroom_stage_a_group_seconds, 10)
            ),
            "bedroom_livingroom_stage_a_group_occupied_ratio_threshold": float(
                min(max(bedroom_livingroom_stage_a_group_occupied_ratio_threshold, 0.0), 1.0)
            ),
            "enable_bedroom_livingroom_stage_a_transformer": bool(
                enable_bedroom_livingroom_stage_a_transformer
            ),
            "bedroom_livingroom_stage_a_transformer_epochs": int(
                max(bedroom_livingroom_stage_a_transformer_epochs, 1)
            ),
            "bedroom_livingroom_stage_a_transformer_batch_size": int(
                max(bedroom_livingroom_stage_a_transformer_batch_size, 1)
            ),
            "bedroom_livingroom_stage_a_transformer_learning_rate": float(
                max(bedroom_livingroom_stage_a_transformer_learning_rate, 1e-5)
            ),
            "bedroom_livingroom_stage_a_transformer_hidden_dim": int(
                max(bedroom_livingroom_stage_a_transformer_hidden_dim, 8)
            ),
            "bedroom_livingroom_stage_a_transformer_num_heads": int(
                max(bedroom_livingroom_stage_a_transformer_num_heads, 1)
            ),
            "bedroom_livingroom_stage_a_transformer_dropout": float(
                min(max(bedroom_livingroom_stage_a_transformer_dropout, 0.0), 0.5)
            ),
            "bedroom_livingroom_stage_a_transformer_class_weight_power": float(
                min(max(bedroom_livingroom_stage_a_transformer_class_weight_power, 0.0), 1.0)
            ),
            "bedroom_livingroom_stage_a_transformer_conv_kernel_size": int(
                max(bedroom_livingroom_stage_a_transformer_conv_kernel_size, 2)
            ),
            "bedroom_livingroom_stage_a_transformer_conv_blocks": int(
                max(bedroom_livingroom_stage_a_transformer_conv_blocks, 1)
            ),
            "enable_bedroom_livingroom_stage_a_transformer_sequence_filter": bool(
                enable_bedroom_livingroom_stage_a_transformer_sequence_filter
            ),
            "enable_cross_room_context_features": bool(enable_cross_room_context_features),
            "cross_room_context_rooms": sorted(
                [str(r).strip().lower() for r in cross_room_context_rooms if str(r).strip()]
            ),
            "min_calib_samples": int(min_calib_samples),
            "min_calib_label_support": int(min_calib_label_support),
        },
    }

    room_gt_daily: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    room_pred_daily: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    room_cls_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    hard_gate_results_all: List[bool] = []
    hard_gate_results_eligible: List[bool] = []
    timeline_gate_totals: List[int] = []
    timeline_gate_passed: List[int] = []
    room_failure_replay_bank: Dict[str, Dict[int, Dict[str, np.ndarray]]] = defaultdict(dict)

    for split in splits:
        split_payload: Dict[str, object] = {
            "train_days": list(split.train_days),
            "test_day": int(split.test_day),
            "train_day_continuity": _compute_day_continuity(split.train_days),
            "rooms": {},
        }
        split_room_outputs: Dict[str, Dict[str, object]] = {}

        for room in ROOMS:
            room_data = room_day_data.get(room, {})
            if any(day not in room_data for day in [*split.train_days, split.test_day]):
                continue

            room_xy = _room_xy(
                room_data,
                split.train_days,
                split.test_day,
                calib_fraction=calib_fraction,
                min_calib_samples=min_calib_samples,
            )
            x_fit = room_xy["x_fit"]
            y_fit = room_xy["y_fit"]
            x_calib = room_xy["x_calib"]
            y_calib = room_xy["y_calib"]
            x_test = room_xy["x_test"]
            y_test = room_xy["y_test"]
            test_df = room_xy["test_df"]
            room_key = str(room).strip().lower()
            configured_occ_threshold = float(room_occupancy_thresholds.get(room_key, occupancy_threshold))
            room_occ_threshold = float(configured_occ_threshold)
            room_label_threshold = dict(room_label_thresholds.get(room_key, {}))
            label_recall_train_day_support = _compute_train_day_label_support(
                room_data=room_data,
                train_days=split.train_days,
                label_names=list(hard_gate_label_recall_floors.get(room_key, {}).keys()),
            )

            if len(x_fit) == 0 or len(x_test) == 0:
                continue

            stage_a_sample_weight = None
            stage_a_reweighting: Dict[str, object] = {"applied": False, "reason": "not_applicable"}
            room_regime_payload: Dict[str, object] = {"applied": False, "reason": "not_applicable", "regime": "normal"}
            replay_training_payload: Dict[str, object] = {"applied": False, "reason": "disabled_or_not_applicable"}
            hard_negative_payload: Dict[str, object] = {"applied": False, "reason": "disabled_or_not_applicable"}
            if room_key == "kitchen":
                if bool(enable_kitchen_stage_a_reweighting):
                    stage_a_sample_weight = _build_kitchen_stage_a_sample_weights(y_fit)
                    stage_a_reweighting = {
                        "applied": True,
                        "n": int(len(stage_a_sample_weight)),
                        "mean": float(np.mean(stage_a_sample_weight)) if len(stage_a_sample_weight) > 0 else 0.0,
                        "min": float(np.min(stage_a_sample_weight)) if len(stage_a_sample_weight) > 0 else 0.0,
                        "max": float(np.max(stage_a_sample_weight)) if len(stage_a_sample_weight) > 0 else 0.0,
                        "occupied_weight_mean": float(
                            np.mean(stage_a_sample_weight[np.asarray(y_fit, dtype=object) != "unoccupied"])
                        )
                        if np.any(np.asarray(y_fit, dtype=object) != "unoccupied")
                        else 0.0,
                        "unoccupied_weight_mean": float(
                            np.mean(stage_a_sample_weight[np.asarray(y_fit, dtype=object) == "unoccupied"])
                        )
                        if np.any(np.asarray(y_fit, dtype=object) == "unoccupied")
                        else 0.0,
                    }
                else:
                    stage_a_reweighting = {"applied": False, "reason": "disabled"}
            elif room_key in {"bedroom", "livingroom"}:
                livingroom_weight = float(max(livingroom_occupied_sample_weight, 1.0))
                use_livingroom_imbalance_weight = room_key == "livingroom" and livingroom_weight > 1.0
                should_apply_room_reweighting = (
                    bool(enable_bedroom_livingroom_boundary_reweighting)
                    or bool(enable_bedroom_livingroom_hard_negative_mining)
                    or bool(use_livingroom_imbalance_weight)
                )
                if should_apply_room_reweighting:
                    stage_a_sample_weight = _build_room_boundary_sample_weights(
                        y_fit,
                        occupied_weight=livingroom_weight if room_key == "livingroom" else 1.5,
                    )
                    if bool(enable_bedroom_livingroom_hard_negative_mining):
                        method = "boundary_emphasis_plus_hard_negative"
                    elif bool(enable_bedroom_livingroom_boundary_reweighting):
                        method = "boundary_emphasis"
                    else:
                        method = "livingroom_imbalance_weight"
                    stage_a_reweighting = {
                        "applied": True,
                        "method": method,
                        "livingroom_occupied_sample_weight": livingroom_weight if room_key == "livingroom" else 1.0,
                        "n": int(len(stage_a_sample_weight)),
                        "mean": float(np.mean(stage_a_sample_weight)) if len(stage_a_sample_weight) > 0 else 0.0,
                        "min": float(np.min(stage_a_sample_weight)) if len(stage_a_sample_weight) > 0 else 0.0,
                        "max": float(np.max(stage_a_sample_weight)) if len(stage_a_sample_weight) > 0 else 0.0,
                        "occupied_weight_mean": float(
                            np.mean(stage_a_sample_weight[np.asarray(y_fit, dtype=object) != "unoccupied"])
                        )
                        if np.any(np.asarray(y_fit, dtype=object) != "unoccupied")
                        else 0.0,
                        "unoccupied_weight_mean": float(
                            np.mean(stage_a_sample_weight[np.asarray(y_fit, dtype=object) == "unoccupied"])
                        )
                        if np.any(np.asarray(y_fit, dtype=object) == "unoccupied")
                        else 0.0,
                    }
                else:
                    stage_a_reweighting = {"applied": False, "reason": "disabled"}

                if bool(enable_livingroom_passive_label_alignment) and room_key == "livingroom":
                    motion_fit = (
                        pd.to_numeric(room_xy["fit_df"]["motion"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                        if "motion" in room_xy["fit_df"].columns
                        else None
                    )
                    alignment_weights, alignment_payload = _build_livingroom_passive_alignment_sample_weights(
                        y_fit,
                        motion_values=motion_fit,
                        direct_positive_weight=float(max(livingroom_direct_positive_weight, 0.05)),
                        passive_positive_weight=float(max(livingroom_passive_positive_weight, 0.01)),
                        unoccupied_weight=float(max(livingroom_unoccupied_weight, 0.05)),
                        entry_exit_band_windows=int(max(livingroom_direct_entry_exit_band_windows, 0)),
                        motion_direct_threshold=float(max(livingroom_direct_motion_threshold, 0.0)),
                    )
                    if stage_a_sample_weight is None:
                        stage_a_sample_weight = np.asarray(alignment_weights, dtype=float)
                        combination_method = "passive_alignment_only"
                    else:
                        stage_a_sample_weight = (
                            np.asarray(stage_a_sample_weight, dtype=float)
                            * np.asarray(alignment_weights, dtype=float)
                        )
                        combination_method = "boundary_x_passive_alignment"
                    stage_a_sample_weight = np.clip(np.asarray(stage_a_sample_weight, dtype=float), 0.01, 50.0)

                    if not bool(stage_a_reweighting.get("applied", False)):
                        stage_a_reweighting = {
                            "applied": True,
                            "method": "livingroom_passive_alignment",
                        }
                    else:
                        stage_a_reweighting["method"] = (
                            f"{str(stage_a_reweighting.get('method', 'reweighting'))}+passive_alignment"
                        )
                    stage_a_reweighting["livingroom_passive_label_alignment"] = {
                        **dict(alignment_payload),
                        "combination_method": str(combination_method),
                    }
                    stage_a_reweighting["n"] = int(len(stage_a_sample_weight))
                    stage_a_reweighting["mean"] = float(np.mean(stage_a_sample_weight))
                    stage_a_reweighting["min"] = float(np.min(stage_a_sample_weight))
                    stage_a_reweighting["max"] = float(np.max(stage_a_sample_weight))
                    y_fit_arr = np.asarray(y_fit, dtype=object)
                    stage_a_reweighting["occupied_weight_mean"] = (
                        float(np.mean(stage_a_sample_weight[y_fit_arr != "unoccupied"]))
                        if np.any(y_fit_arr != "unoccupied")
                        else 0.0
                    )
                    stage_a_reweighting["unoccupied_weight_mean"] = (
                        float(np.mean(stage_a_sample_weight[y_fit_arr == "unoccupied"]))
                        if np.any(y_fit_arr == "unoccupied")
                        else 0.0
                    )

                regime = _select_room_regime_by_train_prior(
                    room_key=room_key,
                    y_fit_labels=np.asarray(y_fit, dtype=object),
                    low_occ_thresholds=bedroom_livingroom_low_occ_thresholds,
                )
                if bool(enable_bedroom_livingroom_regime_routing):
                    room_regime_payload = dict(regime)
                    room_regime_payload["applied"] = True
                else:
                    room_regime_payload = {
                        "applied": False,
                        "reason": "disabled",
                        "room": room_key,
                        "train_occupied_rate": float(regime.get("train_occupied_rate", 0.0)),
                        "low_occupancy_threshold": float(regime.get("low_occupancy_threshold", 0.0)),
                        "regime": "normal",
                    }

            selected_regime = str(room_regime_payload.get("regime", "normal"))
            x_fit_model = np.asarray(x_fit, dtype=float)
            y_fit_model = np.asarray(y_fit, dtype=object)
            stage_a_sample_weight_model = (
                np.asarray(stage_a_sample_weight, dtype=float) if stage_a_sample_weight is not None else None
            )
            stage_a_grouping_enabled = bool(
                enable_bedroom_livingroom_stage_a_minute_grid
                and room_key in stage_a_minute_grid_rooms
                and int(max(bedroom_livingroom_stage_a_group_seconds, 10)) > 10
            )
            stage_a_group_seconds_effective = int(max(bedroom_livingroom_stage_a_group_seconds, 10))
            stage_a_group_ratio_threshold_effective = float(
                min(max(bedroom_livingroom_stage_a_group_occupied_ratio_threshold, 0.0), 1.0)
            )
            stage_a_group_ids_model: Optional[np.ndarray] = None
            stage_a_group_ids_calib: Optional[np.ndarray] = None
            stage_a_group_ids_test: Optional[np.ndarray] = None
            stage_a_grouping_payload: Dict[str, object] = {
                "enabled": bool(stage_a_grouping_enabled),
                "enabled_rooms": sorted(stage_a_minute_grid_rooms),
                "group_seconds": int(stage_a_group_seconds_effective),
                "occupied_ratio_threshold": float(stage_a_group_ratio_threshold_effective),
                "fit_group_count": int(len(x_fit_model)),
                "fit_rows": int(len(x_fit_model)),
            }
            if bool(stage_a_grouping_enabled):
                stage_a_group_ids_model = _build_stage_a_group_ids_from_timestamps(
                    timestamps=room_xy["fit_df"]["timestamp"],
                    resolution_seconds=int(stage_a_group_seconds_effective),
                )
                if isinstance(room_xy.get("calib_df"), pd.DataFrame) and not room_xy["calib_df"].empty:
                    stage_a_group_ids_calib = _build_stage_a_group_ids_from_timestamps(
                        timestamps=room_xy["calib_df"]["timestamp"],
                        resolution_seconds=int(stage_a_group_seconds_effective),
                    )
                stage_a_group_ids_test = _build_stage_a_group_ids_from_timestamps(
                    timestamps=test_df["timestamp"],
                    resolution_seconds=int(stage_a_group_seconds_effective),
                )
                stage_a_grouping_payload["fit_group_count"] = int(len(np.unique(stage_a_group_ids_model)))
                stage_a_grouping_payload["calib_group_count"] = (
                    int(len(np.unique(stage_a_group_ids_calib)))
                    if stage_a_group_ids_calib is not None and len(stage_a_group_ids_calib) > 0
                    else 0
                )
                stage_a_grouping_payload["test_group_count"] = (
                    int(len(np.unique(stage_a_group_ids_test)))
                    if stage_a_group_ids_test is not None and len(stage_a_group_ids_test) > 0
                    else 0
                )
            else:
                stage_a_grouping_payload["reason"] = "disabled_or_not_target_room"

            if (
                bool(enable_bedroom_livingroom_failure_replay)
                and room_key in {"bedroom", "livingroom"}
                and room in room_failure_replay_bank
            ):
                replay_x_chunks: List[np.ndarray] = []
                replay_y_chunks: List[np.ndarray] = []
                replay_days_used: List[int] = []
                for replay_day, payload in sorted(room_failure_replay_bank.get(room, {}).items()):
                    if int(replay_day) not in set(split.train_days):
                        continue
                    x_rep = np.asarray(payload.get("x", np.empty((0, x_fit_model.shape[1]))), dtype=float)
                    y_rep = np.asarray(payload.get("y", np.empty((0,), dtype=object)), dtype=object)
                    if len(y_rep) <= 0:
                        continue
                    replay_x_chunks.append(x_rep)
                    replay_y_chunks.append(y_rep)
                    replay_days_used.append(int(replay_day))
                if replay_x_chunks and replay_y_chunks:
                    replay_x = np.vstack(replay_x_chunks)
                    replay_y = np.concatenate(replay_y_chunks)
                    replay_x, replay_y = _cap_replay_rows(
                        x=replay_x,
                        y=replay_y,
                        max_rows=int(max(bedroom_livingroom_max_replay_rows_per_day, 1)),
                    )
                    n_base = int(len(y_fit_model))
                    n_replay = int(len(replay_y))
                    if n_replay > 0:
                        x_fit_model = np.vstack([x_fit_model, replay_x])
                        y_fit_model = np.concatenate([y_fit_model, replay_y])
                        if stage_a_group_ids_model is not None:
                            start_gid = int(np.max(stage_a_group_ids_model)) + 1 if len(stage_a_group_ids_model) > 0 else 0
                            replay_gids = np.arange(start_gid, start_gid + n_replay, dtype=np.int64)
                            stage_a_group_ids_model = np.concatenate([stage_a_group_ids_model, replay_gids])
                        if stage_a_sample_weight_model is None:
                            stage_a_sample_weight_model = np.ones(shape=(n_base,), dtype=float)
                        replay_weights = np.full(
                            shape=(n_replay,),
                            fill_value=float(max(bedroom_livingroom_failure_replay_weight, 1.0)),
                            dtype=float,
                        )
                        stage_a_sample_weight_model = np.concatenate([stage_a_sample_weight_model, replay_weights])
                        replay_training_payload = {
                            "applied": True,
                            "days_used": replay_days_used,
                            "replay_rows_added": int(n_replay),
                            "base_rows": int(n_base),
                            "total_rows_after_replay": int(len(y_fit_model)),
                            "replay_weight": float(max(bedroom_livingroom_failure_replay_weight, 1.0)),
                        }
                else:
                    replay_training_payload = {"applied": False, "reason": "no_prior_failure_rows_for_train_days"}

            room_cfg = _build_room_model_config(
                seed=seed,
                room_key=room_key,
                bedroom_livingroom_regime=str(selected_regime),
                enable_bedroom_livingroom_regime_routing=bool(enable_bedroom_livingroom_regime_routing),
                enable_bedroom_livingroom_stage_a_hgb=bool(enable_bedroom_livingroom_stage_a_hgb),
                enable_bedroom_livingroom_stage_a_sequence_model=bool(
                    enable_bedroom_livingroom_stage_a_sequence_model
                ),
                bedroom_livingroom_stage_a_sequence_lag_windows=int(
                    max(bedroom_livingroom_stage_a_sequence_lag_windows, 1)
                ),
                enable_bedroom_livingroom_stage_a_transformer=bool(
                    enable_bedroom_livingroom_stage_a_transformer
                ),
                bedroom_livingroom_stage_a_transformer_epochs=int(
                    max(bedroom_livingroom_stage_a_transformer_epochs, 1)
                ),
                bedroom_livingroom_stage_a_transformer_batch_size=int(
                    max(bedroom_livingroom_stage_a_transformer_batch_size, 1)
                ),
                bedroom_livingroom_stage_a_transformer_learning_rate=float(
                    max(bedroom_livingroom_stage_a_transformer_learning_rate, 1e-5)
                ),
                bedroom_livingroom_stage_a_transformer_hidden_dim=int(
                    max(bedroom_livingroom_stage_a_transformer_hidden_dim, 8)
                ),
                bedroom_livingroom_stage_a_transformer_num_heads=int(
                    max(bedroom_livingroom_stage_a_transformer_num_heads, 1)
                ),
                bedroom_livingroom_stage_a_transformer_dropout=float(
                    min(max(bedroom_livingroom_stage_a_transformer_dropout, 0.0), 0.5)
                ),
                bedroom_livingroom_stage_a_transformer_class_weight_power=float(
                    min(max(bedroom_livingroom_stage_a_transformer_class_weight_power, 0.0), 1.0)
                ),
                bedroom_livingroom_stage_a_transformer_conv_kernel_size=int(
                    max(bedroom_livingroom_stage_a_transformer_conv_kernel_size, 2)
                ),
                bedroom_livingroom_stage_a_transformer_conv_blocks=int(
                    max(bedroom_livingroom_stage_a_transformer_conv_blocks, 1)
                ),
                enable_bedroom_livingroom_stage_a_transformer_sequence_filter=bool(
                    enable_bedroom_livingroom_stage_a_transformer_sequence_filter
                ),
            )
            model = EventFirstTwoStageModel(room_cfg).fit(
                x_fit_model,
                y_fit_model,
                stage_a_sample_weight=stage_a_sample_weight_model,
                stage_a_group_ids=stage_a_group_ids_model,
                stage_a_group_occupied_ratio_threshold=float(stage_a_group_ratio_threshold_effective),
            )
            if bool(enable_bedroom_livingroom_hard_negative_mining) and room_key in {"bedroom", "livingroom"}:
                pre_pred = model.predict(
                    x_fit_model,
                    occupancy_threshold=0.5,
                    label_thresholds=room_label_threshold or None,
                    stage_a_group_ids=stage_a_group_ids_model,
                )
                y_fit_occ = np.asarray(y_fit_model != "unoccupied", dtype=bool)
                pre_occ = np.asarray(pre_pred != "unoccupied", dtype=bool)
                fn_mask = np.logical_and(y_fit_occ, ~pre_occ)
                n_fn = int(np.sum(fn_mask))
                if n_fn > 0:
                    if stage_a_sample_weight_model is None or len(stage_a_sample_weight_model) != len(y_fit_model):
                        stage_a_sample_weight_model = np.ones(shape=(len(y_fit_model),), dtype=float)
                    boost = float(max(bedroom_livingroom_hard_negative_weight, 1.0))
                    stage_a_sample_weight_model = np.asarray(stage_a_sample_weight_model, dtype=float)
                    stage_a_sample_weight_model[fn_mask] = stage_a_sample_weight_model[fn_mask] * boost
                    stage_a_sample_weight_model = np.clip(stage_a_sample_weight_model, 0.25, 50.0)
                    model = EventFirstTwoStageModel(room_cfg).fit(
                        x_fit_model,
                        y_fit_model,
                        stage_a_sample_weight=stage_a_sample_weight_model,
                        stage_a_group_ids=stage_a_group_ids_model,
                        stage_a_group_occupied_ratio_threshold=float(stage_a_group_ratio_threshold_effective),
                    )
                    hard_negative_payload = {
                        "applied": True,
                        "false_negative_rows_boosted": int(n_fn),
                        "boost_factor": float(boost),
                    }
                else:
                    hard_negative_payload = {"applied": False, "reason": "no_stage_a_false_negatives"}

            stage_a_reweighting = dict(stage_a_reweighting)
            stage_a_reweighting["room_regime_routing"] = dict(room_regime_payload)
            stage_a_reweighting["failure_replay_training"] = dict(replay_training_payload)
            stage_a_reweighting["hard_negative_mining"] = dict(hard_negative_payload)
            stage_a_reweighting["training_rows_original_fit"] = int(len(y_fit))
            stage_a_reweighting["training_rows_model_fit"] = int(len(y_fit_model))
            stage_a_reweighting["stage_a_grouping"] = dict(stage_a_grouping_payload)
            tuning_payload: Dict[str, object] = {"used": False, "reason": "disabled_or_insufficient_samples"}
            room_mae_tuning: Dict[str, object] = {"used": False, "reason": "not_applicable"}
            room_hardgate_tuning: Dict[str, object] = {"used": False, "reason": "not_applicable"}
            kitchen_mae_tuning: Dict[str, object] = {"used": False, "reason": "not_applicable"}
            should_tune_room = (not disable_threshold_tuning) and (room_key in tune_rooms)
            allow_operating_point_tuning = bool(should_tune_room and room_key not in {"bedroom", "livingroom"})
            if allow_operating_point_tuning and len(x_calib) >= int(min_calib_samples):
                tuning_payload = model.tune_operating_points(
                    x_calib,
                    y_calib,
                    calibration_method=calibration_method,
                    min_samples=min_calib_samples,
                    min_label_support=min_calib_label_support,
                    calib_stage_a_group_ids=stage_a_group_ids_calib,
                )
            elif allow_operating_point_tuning and len(x_calib) < int(min_calib_samples):
                tuning_payload = {
                    "used": False,
                    "reason": "insufficient_calibration_samples",
                    "required_min_calib_samples": int(min_calib_samples),
                    "actual_calib_samples": int(len(x_calib)),
                }
            elif should_tune_room and room_key in {"bedroom", "livingroom"}:
                tuning_payload = {"used": False, "reason": "duration_objective_tuning_only"}
            elif not should_tune_room:
                tuning_payload = {"used": False, "reason": "room_not_in_tune_scope"}

            effective_occ_threshold = float(configured_occ_threshold)
            if bool(tuning_payload.get("used", False)):
                effective_occ_threshold = float(tuning_payload.get("occupancy_threshold", effective_occ_threshold))
            adaptive_threshold_payload: Dict[str, object] = {"applied": False, "reason": "disabled"}

            room_target_key = ROOM_DURATION_KEY.get(room_key, "")
            room_mae_tuning_enabled, room_mae_disabled_reason = _room_mae_tuning_flag_enabled(
                room_key=room_key,
                enable_room_mae_threshold_tuning=bool(enable_room_mae_threshold_tuning),
                enable_kitchen_mae_threshold_tuning=bool(enable_kitchen_mae_threshold_tuning),
            )
            if (
                should_tune_room
                and room_mae_tuning_enabled
                and len(x_calib) >= int(min_calib_samples)
                and isinstance(room_xy.get("calib_df"), pd.DataFrame)
                and bool(room_target_key)
            ):
                max_delta = 0.12 if room_key == "kitchen" else 0.18
                min_improvement = 5.0 if room_key == "kitchen" else 2.0
                room_mae_tuning = _tune_room_occupancy_threshold_by_duration_mae(
                    model=model,
                    room_key=room_key,
                    target_key=room_target_key,
                    calib_df=room_xy["calib_df"],
                    feat_cols=room_xy["feature_columns"],
                    room_label_threshold=room_label_threshold,
                    critical_label_rescue_min_scores=critical_label_rescue_min_scores,
                    current_occupancy_threshold=effective_occ_threshold,
                    max_threshold_delta=float(max_delta),
                    min_required_mae_improvement=float(min_improvement),
                    stability_penalty_weight=0.35,
                    threshold_delta_penalty_weight=25.0,
                    stage_a_group_ids=stage_a_group_ids_calib,
                )
                if bool(room_mae_tuning.get("used", False)) and bool(room_mae_tuning.get("adopted", False)):
                    effective_occ_threshold = float(
                        room_mae_tuning.get("selected_occupancy_threshold", effective_occ_threshold)
                    )
            elif should_tune_room and not room_mae_tuning_enabled:
                room_mae_tuning = {
                    "used": False,
                    "reason": str(room_mae_disabled_reason),
                }

            if room_key in {"bedroom", "livingroom"}:
                if (
                    bool(enable_bedroom_livingroom_hardgate_threshold_tuning)
                    and len(x_calib) >= int(min_calib_samples)
                    and isinstance(room_xy.get("calib_df"), pd.DataFrame)
                ):
                    hardgate_mae_blowup_minutes = 20.0 if room_key == "livingroom" else 40.0
                    hardgate_mae_blowup_ratio = 1.12 if room_key == "livingroom" else 1.30
                    room_hardgate_tuning = _tune_room_occupancy_threshold_for_hard_gate(
                        model=model,
                        room_key=room_key,
                        calib_df=room_xy["calib_df"],
                        feat_cols=room_xy["feature_columns"],
                        room_label_threshold=room_label_threshold,
                        critical_label_rescue_min_scores=critical_label_rescue_min_scores,
                        current_occupancy_threshold=effective_occ_threshold,
                        max_threshold_delta=0.18,
                        recall_floor=float(hard_gate_room_metric_floors.get(room_key, {}).get("occupied_recall", 0.50)),
                        fragmentation_floor=float(
                            hard_gate_room_metric_floors.get(room_key, {}).get("fragmentation_score", 0.0)
                        ),
                        min_required_f1_improvement=0.0,
                        recall_weight=0.85,
                        fragmentation_weight=0.25,
                        recall_floor_penalty_weight=2.0,
                        threshold_delta_penalty_weight=0.06,
                        max_allowed_duration_mae_increase_minutes=float(hardgate_mae_blowup_minutes),
                        max_allowed_duration_mae_ratio=float(hardgate_mae_blowup_ratio),
                        duration_guardrail_penalty_weight=3.0,
                        enable_room_occupancy_decoder=bool(enable_bedroom_livingroom_occupancy_decoder),
                        stage_a_group_ids=stage_a_group_ids_calib,
                    )
                    if bool(room_hardgate_tuning.get("used", False)) and bool(room_hardgate_tuning.get("adopted", False)):
                        effective_occ_threshold = float(
                            room_hardgate_tuning.get("selected_occupancy_threshold", effective_occ_threshold)
                        )
                elif not bool(enable_bedroom_livingroom_hardgate_threshold_tuning):
                    room_hardgate_tuning = {"used": False, "reason": "disabled"}
                else:
                    room_hardgate_tuning = {
                        "used": False,
                        "reason": "insufficient_calibration_samples",
                        "required_min_calib_samples": int(min_calib_samples),
                        "actual_calib_samples": int(len(x_calib)),
                    }

            if room_key == "kitchen":
                kitchen_mae_tuning = dict(room_mae_tuning)

            if bool(enable_adaptive_room_threshold_policy):
                room_occ_threshold, adaptive_threshold_payload = _adaptive_room_occupancy_threshold(
                    room_key=room_key,
                    base_threshold=float(effective_occ_threshold),
                    occupancy_probs=model.predict_occupancy_proba(
                        x_test,
                        stage_a_group_ids=stage_a_group_ids_test,
                    ),
                    y_fit_labels=np.asarray(y_fit, dtype=object),
                )
                # KPI-safe bound for regime-aware threshold shifts on Bedroom/LivingRoom.
                if (
                    room_key in {"bedroom", "livingroom"}
                    and isinstance(room_xy.get("calib_df"), pd.DataFrame)
                    and not room_xy["calib_df"].empty
                    and bool(room_target_key)
                ):
                    guard_ratio = 1.12 if room_key == "livingroom" else 1.30
                    guard_delta_minutes = 20.0 if room_key == "livingroom" else 40.0
                    baseline_pred_for_guard = _predict_labels_for_df(
                        model=model,
                        room_key=room_key,
                        day_df=room_xy["calib_df"],
                        feat_cols=room_xy["feature_columns"],
                        room_occ_threshold=float(effective_occ_threshold),
                        room_label_threshold=room_label_threshold,
                        critical_label_rescue_min_scores=critical_label_rescue_min_scores,
                        enable_kitchen_temporal_decoder=bool(enable_kitchen_temporal_decoder),
                        enable_bedroom_livingroom_occupancy_decoder=bool(
                            enable_bedroom_livingroom_occupancy_decoder
                        ),
                        stage_a_group_seconds=(
                            int(stage_a_group_seconds_effective)
                            if bool(stage_a_grouping_enabled)
                            else None
                        ),
                    )
                    candidate_pred_for_guard = _predict_labels_for_df(
                        model=model,
                        room_key=room_key,
                        day_df=room_xy["calib_df"],
                        feat_cols=room_xy["feature_columns"],
                        room_occ_threshold=float(room_occ_threshold),
                        room_label_threshold=room_label_threshold,
                        critical_label_rescue_min_scores=critical_label_rescue_min_scores,
                        enable_kitchen_temporal_decoder=bool(enable_kitchen_temporal_decoder),
                        enable_bedroom_livingroom_occupancy_decoder=bool(
                            enable_bedroom_livingroom_occupancy_decoder
                        ),
                        stage_a_group_seconds=(
                            int(stage_a_group_seconds_effective)
                            if bool(stage_a_grouping_enabled)
                            else None
                        ),
                    )
                    baseline_guard_mae = _duration_mae_from_labels(
                        calib_df=room_xy["calib_df"],
                        pred_labels=baseline_pred_for_guard,
                        target_key=str(room_target_key),
                    )
                    candidate_guard_mae = _duration_mae_from_labels(
                        calib_df=room_xy["calib_df"],
                        pred_labels=candidate_pred_for_guard,
                        target_key=str(room_target_key),
                    )
                    allowed_abs = float(baseline_guard_mae + float(guard_delta_minutes))
                    allowed_mae = float(allowed_abs)
                    if np.isfinite(baseline_guard_mae) and baseline_guard_mae > 1e-6:
                        allowed_ratio = float(baseline_guard_mae * float(guard_ratio))
                        allowed_mae = float(min(allowed_abs, allowed_ratio))
                    pass_guard = bool(np.isfinite(candidate_guard_mae) and candidate_guard_mae <= (allowed_mae + 1e-9))
                    adaptive_threshold_payload["kpi_guardrail"] = {
                        "applied": True,
                        "room": room_key,
                        "target_key": str(room_target_key),
                        "baseline_duration_mae_minutes": float(baseline_guard_mae),
                        "candidate_duration_mae_minutes": float(candidate_guard_mae),
                        "allowed_duration_mae_minutes": float(allowed_mae),
                        "max_allowed_mae_ratio": float(guard_ratio),
                        "max_allowed_mae_increase_minutes": float(guard_delta_minutes),
                        "pass": bool(pass_guard),
                    }
                    if not bool(pass_guard):
                        room_occ_threshold = float(effective_occ_threshold)
                        adaptive_threshold_payload["adaptive_threshold_rejected_by_kpi_guardrail"] = True
                        adaptive_threshold_payload["selected_threshold"] = float(room_occ_threshold)
                    else:
                        adaptive_threshold_payload["adaptive_threshold_rejected_by_kpi_guardrail"] = False
                        adaptive_threshold_payload["selected_threshold"] = float(room_occ_threshold)
                elif room_key in {"bedroom", "livingroom"}:
                    adaptive_threshold_payload["kpi_guardrail"] = {
                        "applied": False,
                        "reason": "insufficient_calibration_or_missing_target_key",
                    }
            else:
                room_occ_threshold = float(effective_occ_threshold)
                adaptive_threshold_payload = {"applied": False, "reason": "disabled"}

            duration_calibration_occ_threshold = float(room_occ_threshold)
            kitchen_decoder_debug: Dict[str, object] = {"applied": False, "reason": "not_applicable"}
            room_occupancy_decoder_debug: Dict[str, object] = {"applied": False, "reason": "not_applicable"}
            room_timeline_decoder_v2_debug: Dict[str, object] = {"applied": False, "reason": "not_applicable"}
            room_segment_mode_debug: Dict[str, object] = {"applied": False, "reason": "not_applicable"}
            room_prediction_smoothing_debug: Dict[str, object] = {
                "applied": False,
                "reason": "not_applicable",
            }
            room_passive_hysteresis_debug: Dict[str, object] = {
                "applied": False,
                "reason": "not_applicable",
            }
            occupancy_probs = model.predict_occupancy_proba(
                x_test,
                stage_a_group_ids=stage_a_group_ids_test,
            )
            y_pred = model.predict(
                x_test,
                occupancy_threshold=room_occ_threshold,
                label_thresholds=room_label_threshold or None,
                stage_a_group_ids=stage_a_group_ids_test,
            )
            activity_probs = model.predict_activity_proba(x_test)
            y_pred, rescue_debug = _apply_critical_label_rescue(
                room_key=room_key,
                y_pred=y_pred,
                activity_probs=activity_probs,
                rescue_min_scores=critical_label_rescue_min_scores,
            )
            y_pred, shower_fallback_debug = _apply_bathroom_shower_fallback(
                room_key=room_key,
                y_pred=y_pred,
                timestamps=test_df["timestamp"],
                test_df=test_df,
                activity_probs=activity_probs,
            )
            if room_key == "kitchen":
                if bool(enable_kitchen_temporal_decoder):
                    y_pred, kitchen_decoder_debug = _apply_kitchen_temporal_decoder(
                        y_pred=y_pred,
                        occupancy_probs=occupancy_probs,
                        activity_probs=activity_probs,
                    )
                else:
                    kitchen_decoder_debug = {"applied": False, "reason": "disabled"}
            if room_key in {"bedroom", "livingroom"}:
                if bool(enable_bedroom_livingroom_occupancy_decoder):
                    y_pred, room_occupancy_decoder_debug = _apply_room_occupancy_temporal_decoder(
                        room_key=room_key,
                        y_pred=y_pred,
                        occupancy_probs=occupancy_probs,
                        activity_probs=activity_probs,
                    )
                else:
                    room_occupancy_decoder_debug = {"applied": False, "reason": "disabled"}
                if bool(enable_bedroom_livingroom_timeline_decoder_v2):
                    y_pred, room_timeline_decoder_v2_debug = _apply_room_timeline_decoder_v2(
                        room_key=room_key,
                        timestamps=test_df["timestamp"],
                        y_pred=y_pred,
                        occupancy_probs=occupancy_probs,
                        activity_probs=activity_probs,
                    )
                else:
                    room_timeline_decoder_v2_debug = {"applied": False, "reason": "disabled"}
                if bool(enable_bedroom_livingroom_segment_mode):
                    seg_min_duration = int(
                        max(
                            segment_bedroom_min_duration_seconds
                            if room_key == "bedroom"
                            else segment_livingroom_min_duration_seconds,
                            10,
                        )
                    )
                    seg_gap_merge = int(
                        max(
                            segment_bedroom_gap_merge_seconds
                            if room_key == "bedroom"
                            else segment_livingroom_gap_merge_seconds,
                            0,
                        )
                    )
                    seg_min_prob = float(
                        min(
                            max(
                                segment_bedroom_min_activity_prob
                                if room_key == "bedroom"
                                else segment_livingroom_min_activity_prob,
                                0.0,
                            ),
                            1.0,
                        )
                    )
                    pre_segment_pred = np.asarray(y_pred, dtype=object).copy()
                    segment_pred, room_segment_mode_debug = _apply_room_segment_mode(
                        room_key=room_key,
                        timestamps=test_df["timestamp"],
                        test_df=test_df,
                        y_pred=pre_segment_pred,
                        occupancy_probs=occupancy_probs,
                        activity_probs=activity_probs,
                        occupancy_threshold=float(room_occ_threshold),
                        segment_min_duration_seconds=seg_min_duration,
                        segment_gap_merge_seconds=seg_gap_merge,
                        segment_min_activity_prob=seg_min_prob,
                        enable_segment_learned_classifier=bool(
                            enable_bedroom_livingroom_segment_learned_classifier
                        ),
                        segment_classifier_min_segments=int(max(segment_classifier_min_segments, 2)),
                        segment_classifier_confidence_floor=float(
                            min(max(segment_classifier_confidence_floor, 0.0), 1.0)
                        ),
                        segment_classifier_min_windows=int(max(segment_classifier_min_windows, 1)),
                    )
                    if room_key == "livingroom":
                        before_occ_windows = int(np.sum(pre_segment_pred != "unoccupied"))
                        after_occ_windows = int(np.sum(np.asarray(segment_pred, dtype=object) != "unoccupied"))
                        ratio_cap = float(max(segment_livingroom_max_occupied_ratio, 1.0))
                        delta_cap = int(max(segment_livingroom_max_occupied_window_delta, 0))
                        ratio_val = float(after_occ_windows / max(before_occ_windows, 1))
                        delta_val = int(after_occ_windows - before_occ_windows)
                        inflation_blocked = bool(
                            ratio_val > ratio_cap and delta_val > delta_cap
                        )
                        room_segment_mode_debug["livingroom_occupancy_inflation_guardrail"] = {
                            "applied": True,
                            "before_occupied_windows": int(before_occ_windows),
                            "after_occupied_windows": int(after_occ_windows),
                            "occupied_ratio_after_vs_before": float(ratio_val),
                            "occupied_window_delta": int(delta_val),
                            "max_allowed_ratio": float(ratio_cap),
                            "max_allowed_window_delta": int(delta_cap),
                            "pass": bool(not inflation_blocked),
                        }
                        y_pred = pre_segment_pred if inflation_blocked else np.asarray(segment_pred, dtype=object)
                        if inflation_blocked:
                            room_segment_mode_debug["reason"] = "segment_mode_reverted_by_livingroom_guardrail"
                    else:
                        y_pred = np.asarray(segment_pred, dtype=object)
                else:
                    room_segment_mode_debug = {"applied": False, "reason": "disabled"}

                if bool(enable_bedroom_livingroom_prediction_smoothing):
                    y_pred, room_prediction_smoothing_debug = _apply_room_prediction_occupancy_smoothing(
                        room_key=room_key,
                        y_pred=np.asarray(y_pred, dtype=object),
                        activity_probs=activity_probs,
                        min_run_windows=int(
                            max(bedroom_livingroom_prediction_smoothing_min_run_windows, 1)
                        ),
                        gap_fill_windows=int(
                            max(bedroom_livingroom_prediction_smoothing_gap_fill_windows, 0)
                        ),
                    )
                else:
                    room_prediction_smoothing_debug = {"applied": False, "reason": "disabled"}

                if bool(enable_bedroom_livingroom_passive_hysteresis):
                    hold_minutes = (
                        float(max(bedroom_passive_hold_minutes, 0.0))
                        if room_key == "bedroom"
                        else float(max(livingroom_passive_hold_minutes, 0.0))
                    )
                    motion_vals = (
                        pd.to_numeric(test_df["motion"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                        if "motion" in test_df.columns
                        else np.zeros(shape=(len(y_pred),), dtype=float)
                    )
                    y_pred, room_passive_hysteresis_debug = _apply_room_passive_occupancy_hysteresis(
                        room_key=room_key,
                        y_pred=np.asarray(y_pred, dtype=object),
                        occupancy_probs=occupancy_probs,
                        activity_probs=activity_probs,
                        motion_values=motion_vals,
                        hold_minutes=hold_minutes,
                        exit_min_consecutive_windows=int(max(passive_exit_min_consecutive_windows, 1)),
                        entry_occ_threshold=float(min(max(passive_entry_occ_threshold, 0.0), 1.0)),
                        entry_room_prob_threshold=float(
                            min(max(passive_entry_room_prob_threshold, 0.0), 1.0)
                        ),
                        stay_occ_threshold=float(min(max(passive_stay_occ_threshold, 0.0), 1.0)),
                        stay_room_prob_threshold=float(
                            min(max(passive_stay_room_prob_threshold, 0.0), 1.0)
                        ),
                        exit_occ_threshold=float(min(max(passive_exit_occ_threshold, 0.0), 1.0)),
                        exit_room_prob_threshold=float(
                            min(max(passive_exit_room_prob_threshold, 0.0), 1.0)
                        ),
                        motion_reset_threshold=float(max(passive_motion_reset_threshold, 0.0)),
                        motion_quiet_threshold=float(max(passive_motion_quiet_threshold, 0.0)),
                        livingroom_strict_entry_requires_strong_signal=bool(
                            room_key == "livingroom" and livingroom_strict_entry_requires_strong_signal
                        ),
                        livingroom_entry_motion_threshold=float(
                            max(livingroom_entry_motion_threshold, 0.0)
                        ),
                    )
                else:
                    room_passive_hysteresis_debug = {"applied": False, "reason": "disabled"}

            failure_replay_generation: Dict[str, object] = {"applied": False, "reason": "disabled_or_not_applicable"}
            if bool(enable_bedroom_livingroom_failure_replay) and room_key in {"bedroom", "livingroom"}:
                y_true_occ = np.asarray(y_test != "unoccupied", dtype=bool)
                y_pred_occ = np.asarray(np.asarray(y_pred, dtype=object) != "unoccupied", dtype=bool)
                fn_mask = np.logical_and(y_true_occ, ~y_pred_occ)
                n_fn = int(np.sum(fn_mask))
                if n_fn > 0:
                    x_fn = np.asarray(x_test[fn_mask], dtype=float)
                    y_fn = np.asarray(y_test[fn_mask], dtype=object)
                    x_fn, y_fn = _cap_replay_rows(
                        x=x_fn,
                        y=y_fn,
                        max_rows=int(max(bedroom_livingroom_max_replay_rows_per_day, 1)),
                    )
                    room_failure_replay_bank[room][int(split.test_day)] = {"x": x_fn, "y": y_fn}
                    failure_replay_generation = {
                        "applied": True,
                        "test_day": int(split.test_day),
                        "false_negative_rows_detected": int(n_fn),
                        "false_negative_rows_stored": int(len(y_fn)),
                    }
                else:
                    failure_replay_generation = {"applied": False, "reason": "no_false_negative_rows"}

            duration_target_key = ROOM_DURATION_KEY.get(room_key) if bool(enable_duration_calibration) else None
            train_pred_vals: List[float] = []
            train_gt_vals: List[float] = []
            if bool(enable_duration_calibration):
                if duration_target_key:
                    for day in split.train_days:
                        day_df = room_data.get(day)
                        if day_df is None or day_df.empty:
                            continue
                        y_day_pred = _predict_labels_for_df(
                            model=model,
                            room_key=room_key,
                            day_df=day_df,
                            feat_cols=room_xy["feature_columns"],
                            room_occ_threshold=duration_calibration_occ_threshold,
                            room_label_threshold=room_label_threshold,
                            critical_label_rescue_min_scores=critical_label_rescue_min_scores,
                            enable_kitchen_temporal_decoder=bool(enable_kitchen_temporal_decoder),
                            enable_bedroom_livingroom_occupancy_decoder=bool(
                                enable_bedroom_livingroom_occupancy_decoder
                            ),
                            stage_a_group_seconds=(
                                int(stage_a_group_seconds_effective)
                                if bool(stage_a_grouping_enabled)
                                else None
                            ),
                        )
                        pred_day_df = day_df[["timestamp"]].copy()
                        pred_day_df["activity"] = y_day_pred
                        gt_day_targets = build_daily_event_targets(
                            labels_to_episodes(day_df, timestamp_col="timestamp", label_col="activity")
                        )
                        pred_day_targets = build_daily_event_targets(
                            labels_to_episodes(pred_day_df, timestamp_col="timestamp", label_col="activity")
                        )
                        train_pred_vals.append(float(pred_day_targets.get(duration_target_key, 0.0)))
                        train_gt_vals.append(float(gt_day_targets.get(duration_target_key, 0.0)))

            leakage_pass = True
            leakage_reasons: List[str] = []
            fit_end = room_xy.get("fit_end_timestamp")
            calib_start = room_xy.get("calib_start_timestamp")
            calib_end = room_xy.get("calib_end_timestamp")
            test_start = room_xy.get("test_start_timestamp")
            if fit_end is not None and test_start is not None and pd.Timestamp(fit_end) > pd.Timestamp(test_start):
                leakage_pass = False
                leakage_reasons.append("fit_after_test_start")
            if calib_start is not None and fit_end is not None and pd.Timestamp(calib_start) <= pd.Timestamp(fit_end):
                leakage_pass = False
                leakage_reasons.append("calib_not_after_fit")
            if calib_end is not None and test_start is not None and pd.Timestamp(calib_end) >= pd.Timestamp(test_start):
                leakage_pass = False
                leakage_reasons.append("calib_overlaps_test")

            error_positive_label = {
                "bedroom": "sleep",
                "livingroom": "livingroom_normal_use",
                "kitchen": "kitchen_normal_use",
                "bathroom": "shower",
                "entrance": "out",
            }.get(room_key, "")
            feature_importance = _extract_feature_importance_ranking(
                model=model,
                feature_columns=room_xy["feature_columns"],
            )
            room_data_diagnostics = _build_room_data_diagnostics(
                y_fit=np.asarray(room_xy.get("y_fit", np.empty((0,), dtype=object)), dtype=object),
                y_calib=np.asarray(room_xy.get("y_calib", np.empty((0,), dtype=object)), dtype=object),
                y_test=np.asarray(y_test, dtype=object),
            )
            label_correction_reference = _label_correction_reference_for_room_day(
                label_corrections_summary=label_corrections_summary,
                room=room,
                day=int(split.test_day),
            )

            split_room_outputs[room] = {
                "room": room,
                "room_key": room_key,
                "test_df": test_df,
                "y_test": np.asarray(y_test, dtype=object),
                "y_pred": np.asarray(y_pred, dtype=object),
                "occupancy_probs": np.asarray(occupancy_probs, dtype=float),
                "activity_probs": {k: np.asarray(v, dtype=float) for k, v in activity_probs.items()},
                "error_positive_label": error_positive_label,
                "leakage_pass": leakage_pass,
                "leakage_reasons": leakage_reasons,
                "fit_size": int(room_xy["fit_size"]),
                "calib_size": int(room_xy["calib_size"]),
                "test_size": int(room_xy["test_size"]),
                "fit_end_timestamp": fit_end,
                "calib_start_timestamp": calib_start,
                "calib_end_timestamp": calib_end,
                "test_start_timestamp": test_start,
                "operating_points": model.get_operating_points(),
                "model_config": {
                    "n_estimators_stage_a": int(room_cfg.n_estimators_stage_a),
                    "n_estimators_stage_b": int(room_cfg.n_estimators_stage_b),
                    "min_samples_leaf": int(room_cfg.min_samples_leaf),
                    "stage_a_class_weight": room_cfg.stage_a_class_weight,
                    "stage_b_class_weight": room_cfg.stage_b_class_weight,
                    "stage_a_model_type": str(room_cfg.stage_a_model_type),
                    "stage_a_temporal_lag_windows": int(room_cfg.stage_a_temporal_lag_windows),
                    "stage_a_grouping": dict(stage_a_grouping_payload),
                },
                "calibration": tuning_payload,
                "room_occupancy_threshold": room_occ_threshold,
                "room_occupancy_threshold_configured": configured_occ_threshold,
                "duration_calibration_occupancy_threshold": duration_calibration_occ_threshold,
                "room_label_thresholds": room_label_threshold,
                "adaptive_room_threshold": adaptive_threshold_payload,
                "critical_label_rescue": rescue_debug,
                "kitchen_temporal_decoder": kitchen_decoder_debug,
                "room_occupancy_temporal_decoder": room_occupancy_decoder_debug,
                "room_timeline_decoder_v2": room_timeline_decoder_v2_debug,
                "room_segment_mode": room_segment_mode_debug,
                "room_prediction_smoothing": room_prediction_smoothing_debug,
                "room_passive_hysteresis": room_passive_hysteresis_debug,
                "bathroom_shower_fallback": shower_fallback_debug,
                "room_mae_tuning": room_mae_tuning,
                "room_hardgate_tuning": room_hardgate_tuning,
                "kitchen_mae_tuning": kitchen_mae_tuning,
                "stage_a_reweighting": stage_a_reweighting,
                "label_recall_train_day_support": label_recall_train_day_support,
                "feature_importance_ranking": feature_importance,
                "failure_replay_generation": failure_replay_generation,
                "room_data_diagnostics": room_data_diagnostics,
                "label_correction_reference": label_correction_reference,
                "duration_target_key": duration_target_key,
                "duration_train_pred_vals": list(train_pred_vals),
                "duration_train_gt_vals": list(train_gt_vals),
            }

        livingroom_cross_room_presence_payload: Dict[str, object]
        if bool(enable_livingroom_cross_room_presence_decoder):
            livingroom_cross_room_presence_payload = _apply_livingroom_cross_room_presence_decoder(
                split_room_outputs=split_room_outputs,
                supporting_rooms=livingroom_cross_room_supporting_rooms,
                hold_minutes=float(max(livingroom_cross_room_hold_minutes, 0.0)),
                max_extension_minutes=float(max(livingroom_cross_room_max_extension_minutes, 0.0)),
                entry_occ_threshold=float(min(max(livingroom_cross_room_entry_occ_threshold, 0.0), 1.0)),
                entry_room_prob_threshold=float(
                    min(max(livingroom_cross_room_entry_room_prob_threshold, 0.0), 1.0)
                ),
                entry_motion_threshold=float(max(livingroom_cross_room_entry_motion_threshold, 0.0)),
                refresh_occ_threshold=float(min(max(livingroom_cross_room_refresh_occ_threshold, 0.0), 1.0)),
                refresh_room_prob_threshold=float(
                    min(max(livingroom_cross_room_refresh_room_prob_threshold, 0.0), 1.0)
                ),
                other_room_exit_occ_threshold=float(
                    min(max(livingroom_cross_room_other_room_exit_occ_threshold, 0.0), 1.0)
                ),
                other_room_exit_confirm_windows=int(
                    max(livingroom_cross_room_other_room_exit_confirm_windows, 1)
                ),
                other_room_unoccupied_max_occ_prob=float(
                    min(max(livingroom_cross_room_other_room_unoccupied_max_occ_prob, 0.0), 1.0)
                ),
                entrance_exit_occ_threshold=float(
                    min(max(livingroom_cross_room_entrance_exit_occ_threshold, 0.0), 1.0)
                ),
                min_support_rooms=int(max(livingroom_cross_room_min_support_rooms, 1)),
                require_other_room_predicted_occupied=bool(
                    livingroom_cross_room_require_other_room_predicted_occupied
                ),
                enable_bedroom_sleep_night_guard=bool(livingroom_night_bedroom_sleep_guard_enabled),
                night_bedroom_guard_start_hour=int(
                    min(max(livingroom_night_bedroom_sleep_guard_start_hour, 0), 23)
                ),
                night_bedroom_guard_end_hour=int(
                    min(max(livingroom_night_bedroom_sleep_guard_end_hour, 0), 23)
                ),
                night_bedroom_sleep_occ_threshold=float(
                    min(max(livingroom_night_bedroom_sleep_guard_bedroom_sleep_occ_threshold, 0.0), 1.0)
                ),
                night_bedroom_sleep_prob_threshold=float(
                    min(max(livingroom_night_bedroom_sleep_guard_bedroom_sleep_prob_threshold, 0.0), 1.0)
                ),
                night_bedroom_exit_occ_threshold=float(
                    min(max(livingroom_night_bedroom_sleep_guard_bedroom_exit_occ_threshold, 0.0), 1.0)
                ),
                night_bedroom_exit_motion_threshold=float(
                    max(livingroom_night_bedroom_sleep_guard_bedroom_exit_motion_threshold, 0.0)
                ),
                night_entry_occ_threshold=float(
                    min(max(livingroom_night_bedroom_sleep_guard_entry_occ_threshold, 0.0), 1.0)
                ),
                night_entry_motion_threshold=float(
                    max(livingroom_night_bedroom_sleep_guard_entry_motion_threshold, 0.0)
                ),
                night_entry_confirm_windows=int(
                    max(livingroom_night_bedroom_sleep_guard_entry_confirm_windows, 1)
                ),
                night_bedroom_suppression_label=str(
                    livingroom_night_bedroom_sleep_guard_suppression_label
                ),
                night_bedroom_min_coverage=float(
                    min(max(livingroom_night_bedroom_sleep_guard_min_coverage, 0.0), 1.0)
                ),
                night_bedroom_flatline_motion_std_max=float(
                    max(livingroom_night_bedroom_sleep_guard_flatline_motion_std_max, 0.0)
                ),
                night_bedroom_flatline_occ_std_max=float(
                    max(livingroom_night_bedroom_sleep_guard_flatline_occ_std_max, 0.0)
                ),
                night_bedroom_flatline_min_windows=int(
                    max(livingroom_night_bedroom_sleep_guard_flatline_min_windows, 1)
                ),
            )
        else:
            livingroom_cross_room_presence_payload = {
                "enabled": False,
                "reason": "disabled",
                "changed_windows": 0,
                "added_occupied_windows": 0,
                "entry_events": 0,
                "refresh_events": 0,
                "exit_events": 0,
                "maintained_by_absence_windows": 0,
                "insufficient_support_windows": 0,
                "night_bedroom_guard_applied": False,
                "night_bedroom_guard_reason": "disabled",
                "night_bedroom_guard_suppressed_windows": 0,
                "night_bedroom_guard_blocked_entries": 0,
                "night_bedroom_guard_unknown_windows": 0,
                "night_bedroom_guard_unoccupied_windows": 0,
            }
        if "LivingRoom" in split_room_outputs and isinstance(split_room_outputs["LivingRoom"], dict):
            split_room_outputs["LivingRoom"]["livingroom_cross_room_presence_decoder"] = (
                livingroom_cross_room_presence_payload
            )
        split_payload["livingroom_cross_room_presence_decoder"] = livingroom_cross_room_presence_payload

        arbitration_payload: Dict[str, object]
        if bool(enable_single_resident_arbitration):
            arbitration_payload = _apply_single_resident_arbitration(
                split_room_outputs=split_room_outputs,
                arbitration_rooms=single_resident_rooms,
                min_confidence_margin=float(single_resident_min_margin),
                bedroom_night_start_hour=int(single_resident_bedroom_night_start_hour),
                bedroom_night_end_hour=int(single_resident_bedroom_night_end_hour),
                bedroom_night_min_run_windows=int(single_resident_bedroom_night_min_run_windows),
                bedroom_night_min_label_score=float(single_resident_bedroom_night_min_label_score),
                kitchen_min_label_score=float(single_resident_kitchen_min_label_score),
                kitchen_guard_hour_ranges=single_resident_kitchen_guard_hours,
                continuity_min_run_windows=int(single_resident_continuity_min_run_windows),
                continuity_min_occ_prob=float(single_resident_continuity_min_occ_prob),
            )
        else:
            arbitration_payload = {
                "enabled": False,
                "reason": "disabled",
                "candidate_rooms": sorted(
                    [str(r).strip().lower() for r in single_resident_rooms if str(r).strip()]
                ),
                "conflicts_total": 0,
                "adjustments_total": 0,
                "per_room": {},
            }
        split_payload["single_resident_arbitration"] = arbitration_payload

        for room in ROOMS:
            work = split_room_outputs.get(room)
            if not isinstance(work, dict):
                continue
            test_df = work["test_df"]
            y_test = np.asarray(work["y_test"], dtype=object)
            y_pred = np.asarray(work["y_pred"], dtype=object)
            pred_df = test_df[["timestamp"]].copy()
            pred_df["activity"] = y_pred

            gt_episodes = labels_to_episodes(test_df, timestamp_col="timestamp", label_col="activity")
            pred_episodes = labels_to_episodes(pred_df, timestamp_col="timestamp", label_col="activity")

            gt_daily = build_daily_event_targets(gt_episodes)
            pred_daily = build_daily_event_targets(pred_episodes)
            duration_calibration: Dict[str, object] = {"enabled": False}
            duration_target_key = work.get("duration_target_key")
            if bool(enable_duration_calibration) and isinstance(duration_target_key, str) and duration_target_key:
                raw_val = float(pred_daily.get(duration_target_key, 0.0))
                cal_result = _calibrate_duration_from_train(
                    work.get("duration_train_pred_vals", []),
                    work.get("duration_train_gt_vals", []),
                    raw_val,
                    room_key=str(work.get("room_key", "")).lower(),
                    enable_kitchen_robust_duration_calibration=bool(enable_kitchen_robust_duration_calibration),
                )
                pred_daily[duration_target_key] = float(cal_result.get("corrected_value", raw_val))
                day_key = duration_target_key.replace("_minutes", "_day")
                if day_key in pred_daily:
                    pred_daily[day_key] = 1.0 if float(pred_daily[duration_target_key]) > 0.0 else 0.0
                duration_calibration = {
                    "enabled": bool(cal_result.get("enabled", False)),
                    "target_key": duration_target_key,
                    "method": str(cal_result.get("method", "none")),
                    "train_pairs": int(
                        cal_result.get("train_pairs", len(work.get("duration_train_pred_vals", [])))
                    ),
                    "raw_value": float(raw_val),
                    "corrected_value": float(pred_daily[duration_target_key]),
                }
                if "slope" in cal_result:
                    duration_calibration["slope"] = float(cal_result["slope"])
                if "intercept" in cal_result:
                    duration_calibration["intercept"] = float(cal_result["intercept"])

            cls = _classification_metrics(y_test, y_pred)
            label_recall_summary = _label_recall_summary(y_test, y_pred)
            room_key_eval = str(work.get("room_key", "")).strip().lower()
            prediction_smoothing_applied = bool(
                ((work.get("room_prediction_smoothing") or {}) if isinstance(work, dict) else {}).get("applied", False)
            )
            if prediction_smoothing_applied and room_key_eval in {"bedroom", "livingroom"}:
                frag_min_run = int(max(bedroom_livingroom_prediction_smoothing_min_run_windows, 1))
                frag_gap_fill = int(max(bedroom_livingroom_prediction_smoothing_gap_fill_windows, 0))
                smooth_pred_mask = False
            else:
                frag_min_run = int(max(hard_gate_fragmentation_min_run_windows.get(room_key_eval, 3), 1))
                frag_gap_fill = int(max(hard_gate_fragmentation_gap_fill_windows.get(room_key_eval, 2), 0))
                smooth_pred_mask = True
            fragmentation_score = _care_fragmentation_score(
                y_test,
                y_pred,
                min_run_windows=frag_min_run,
                gap_fill_windows=frag_gap_fill,
                smooth_pred_mask=bool(smooth_pred_mask),
            )
            timeline_payload = _compute_room_timeline_gate_payload(
                room_key=room_key_eval,
                test_df=test_df,
                y_true=np.asarray(y_test, dtype=object),
                y_pred=np.asarray(y_pred, dtype=object),
                enabled=bool(enable_bedroom_livingroom_timeline_gates),
            )
            unknown_rate = float(np.mean(np.asarray(y_pred) == base_cfg.unknown_label))

            room_gt_daily[room].append(gt_daily)
            room_pred_daily[room].append(pred_daily)
            for k, v in cls.items():
                room_cls_metrics[room][k].append(float(v))

            hard_gate = _split_hard_gate(
                room,
                gt_daily,
                pred_daily,
                cls=cls,
                label_recall_summary=label_recall_summary,
                fragmentation_score=fragmentation_score,
                unknown_rate=unknown_rate,
                unknown_rate_cap=0.20,
                room_metric_floors=hard_gate_room_metric_floors,
                room_label_recall_floors=hard_gate_label_recall_floors,
                room_label_recall_min_supports=hard_gate_label_recall_min_supports,
                label_recall_train_day_support=work.get("label_recall_train_day_support"),
                n_train_days=int(len(split.train_days)),
                hard_gate_min_train_days=int(max(hard_gate_min_train_days, 0)),
            )
            timeline_total = int(timeline_payload.get("total", 0))
            timeline_pass = int(timeline_payload.get("passed", 0))
            if timeline_total > 0 and timeline_pass < timeline_total:
                hard_gate["pass"] = False
                hard_gate.setdefault("reasons", []).append(
                    f"timeline_gates_failed:{timeline_pass}/{timeline_total}"
                )
            leakage_pass = bool(work.get("leakage_pass", True))
            hard_gate["pass"] = bool(hard_gate.get("pass", True) and leakage_pass)
            if not leakage_pass:
                hard_gate.setdefault("reasons", []).append("leakage_check_failed")
            hard_gate_pass_val = bool(hard_gate.get("pass", True))
            hard_gate_results_all.append(hard_gate_pass_val)
            if bool(hard_gate.get("eligible", True)):
                hard_gate_results_eligible.append(hard_gate_pass_val)
            timeline_gate_totals.append(timeline_total)
            timeline_gate_passed.append(timeline_pass)

            error_episodes = _extract_error_episodes(
                timestamps=test_df["timestamp"],
                y_true=y_test,
                y_pred=y_pred,
                positive_label=str(work.get("error_positive_label", "")),
            )
            episode_metrics = _compute_binary_episode_metrics(
                timestamps=test_df["timestamp"],
                y_true=y_test,
                y_pred=y_pred,
                positive_label="occupied",
                tolerance_minutes=5.0,
            )

            split_payload["rooms"][room] = {
                "gt_targets": gt_daily,
                "pred_targets": pred_daily,
                "classification": cls,
                "label_recall_summary": label_recall_summary,
                "label_recall_train_day_support": work.get("label_recall_train_day_support", {}),
                "fragmentation_score": float(fragmentation_score),
                "timeline_metrics": timeline_payload.get("metrics", {}),
                "timeline_gates": timeline_payload.get("gates", []),
                "timeline_gates_passed": int(timeline_pass),
                "timeline_gates_total": int(timeline_total),
                "unknown_rate": unknown_rate,
                "hard_gate": hard_gate,
                "leakage_audit_pass": leakage_pass,
                "leakage_audit": {
                    "pass": leakage_pass,
                    "reasons": list(work.get("leakage_reasons", [])),
                    "fit_size": int(work.get("fit_size", 0)),
                    "calib_size": int(work.get("calib_size", 0)),
                    "test_size": int(work.get("test_size", 0)),
                    "fit_end_timestamp": str(work.get("fit_end_timestamp"))
                    if work.get("fit_end_timestamp") is not None
                    else None,
                    "calib_start_timestamp": str(work.get("calib_start_timestamp"))
                    if work.get("calib_start_timestamp") is not None
                    else None,
                    "calib_end_timestamp": str(work.get("calib_end_timestamp"))
                    if work.get("calib_end_timestamp") is not None
                    else None,
                    "test_start_timestamp": str(work.get("test_start_timestamp"))
                    if work.get("test_start_timestamp") is not None
                    else None,
                },
                "operating_points": work.get("operating_points", {}),
                "calibration": work.get("calibration", {}),
                "room_occupancy_threshold": work.get("room_occupancy_threshold"),
                "room_occupancy_threshold_configured": work.get("room_occupancy_threshold_configured"),
                "duration_calibration_occupancy_threshold": work.get("duration_calibration_occupancy_threshold"),
                "room_label_thresholds": work.get("room_label_thresholds", {}),
                "adaptive_room_threshold": work.get("adaptive_room_threshold", {}),
                "critical_label_rescue": work.get("critical_label_rescue", {}),
                "kitchen_temporal_decoder": work.get("kitchen_temporal_decoder", {}),
                "room_occupancy_temporal_decoder": work.get("room_occupancy_temporal_decoder", {}),
                "room_timeline_decoder_v2": work.get("room_timeline_decoder_v2", {}),
                "room_segment_mode": work.get("room_segment_mode", {}),
                "room_passive_hysteresis": work.get("room_passive_hysteresis", {}),
                "livingroom_cross_room_presence_decoder": work.get(
                    "livingroom_cross_room_presence_decoder", {}
                ),
                "bathroom_shower_fallback": work.get("bathroom_shower_fallback", {}),
                "duration_calibration": duration_calibration,
                "room_mae_tuning": work.get("room_mae_tuning", {}),
                "room_hardgate_tuning": work.get("room_hardgate_tuning", {}),
                "kitchen_mae_tuning": work.get("kitchen_mae_tuning", {}),
                "stage_a_reweighting": work.get("stage_a_reweighting", {}),
                "feature_importance_ranking": work.get("feature_importance_ranking", {}),
                "failure_replay_generation": work.get("failure_replay_generation", {}),
                "occupied_rate_snapshot": (
                    ((work.get("room_data_diagnostics") or {}) if isinstance(work, dict) else {}).get(
                        "occupied_rate_snapshot", {}
                    )
                ),
                "label_minutes_snapshot": (
                    ((work.get("room_data_diagnostics") or {}) if isinstance(work, dict) else {}).get(
                        "label_minutes_snapshot", {}
                    )
                ),
                "label_correction_reference": work.get("label_correction_reference", {}),
                "single_resident_arbitration": (
                    arbitration_payload.get("per_room", {}).get(room, {"adjusted_windows": 0})
                    if isinstance(arbitration_payload, dict)
                    else {"adjusted_windows": 0}
                ),
                "episode_metrics": episode_metrics,
                "error_episodes": error_episodes,
            }

        report["splits"].append(split_payload)

    summary: Dict[str, Dict[str, float]] = {}
    for room in sorted(room_gt_daily.keys()):
        summary[room] = compute_room_care_kpis(room, room_gt_daily[room], room_pred_daily[room])
    report["summary"] = summary

    cls_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for room, metrics_map in room_cls_metrics.items():
        cls_summary[room] = {metric: _safe_mean_std(values) for metric, values in metrics_map.items()}
    report["classification_summary"] = cls_summary

    total_checks_all = int(len(hard_gate_results_all))
    pass_checks_all = int(sum(1 for v in hard_gate_results_all if v))
    total_checks_eligible = int(len(hard_gate_results_eligible))
    pass_checks_eligible = int(sum(1 for v in hard_gate_results_eligible if v))
    ineligible_checks = int(max(total_checks_all - total_checks_eligible, 0))
    timeline_total_checks = int(sum(timeline_gate_totals))
    timeline_pass_checks = int(sum(timeline_gate_passed))
    report["gate_summary"] = {
        "hard_gate_min_train_days": int(max(hard_gate_min_train_days, 0)),
        "hard_gate_checks_total": total_checks_eligible,
        "hard_gate_checks_passed": pass_checks_eligible,
        "hard_gate_checks_total_eligible": total_checks_eligible,
        "hard_gate_checks_passed_eligible": pass_checks_eligible,
        "hard_gate_pass_rate": (
            float(pass_checks_eligible / total_checks_eligible) if total_checks_eligible > 0 else 0.0
        ),
        "all_hard_gates_pass": (
            bool(pass_checks_eligible == total_checks_eligible) if total_checks_eligible > 0 else False
        ),
        "hard_gate_checks_total_full": total_checks_all,
        "hard_gate_checks_passed_full": pass_checks_all,
        "hard_gate_pass_rate_full": float(pass_checks_all / total_checks_all) if total_checks_all > 0 else 0.0,
        "all_hard_gates_pass_full": bool(pass_checks_all == total_checks_all) if total_checks_all > 0 else False,
        "hard_gate_checks_ineligible": int(ineligible_checks),
        "timeline_gate_checks_total": timeline_total_checks,
        "timeline_gate_checks_passed": timeline_pass_checks,
        "timeline_gate_pass_rate": (
            float(timeline_pass_checks / timeline_total_checks) if timeline_total_checks > 0 else 1.0
        ),
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run event-first rolling ADL backtest")
    parser.add_argument("--data-dir", required=True, help="Directory containing training xlsx files")
    parser.add_argument("--elder-id", required=True, help="Elder id prefix in filenames")
    parser.add_argument("--min-day", type=int, default=4)
    parser.add_argument("--max-day", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--occupancy-threshold", type=float, default=0.35)
    parser.add_argument("--calibration-method", type=str, default="isotonic", choices=["none", "isotonic", "platt"])
    parser.add_argument("--calib-fraction", type=float, default=0.20)
    parser.add_argument("--min-calib-samples", type=int, default=500)
    parser.add_argument("--min-calib-label-support", type=int, default=30)
    parser.add_argument(
        "--tune-rooms",
        type=str,
        default="entrance,bedroom,livingroom,kitchen",
        help="Comma-separated rooms to tune thresholds for",
    )
    parser.add_argument(
        "--room-occupancy-thresholds",
        type=str,
        default="entrance=0.55",
        help="Comma-separated room=threshold overrides (default: entrance=0.55)",
    )
    parser.add_argument(
        "--room-label-thresholds",
        type=str,
        default="bedroom.sleep=0.35,bathroom.shower=0.35",
        help="Comma-separated room.label=threshold overrides",
    )
    parser.add_argument(
        "--critical-label-rescue-min-scores",
        type=str,
        default="bedroom.sleep=0.05,bathroom.shower=0.05",
        help="Comma-separated room.label=min_score for probability-based critical-label rescue",
    )
    parser.add_argument(
        "--disable-duration-calibration",
        action="store_true",
        help="Disable train-day affine calibration for daily duration KPIs",
    )
    parser.add_argument(
        "--enable-room-mae-threshold-tuning",
        action="store_true",
        help="Enable room-level occupancy threshold tuning against duration MAE objective",
    )
    parser.add_argument(
        "--disable-adaptive-room-threshold-policy",
        action="store_true",
        help="Disable train-prior adaptive occupancy threshold policy for Bedroom/LivingRoom",
    )
    parser.add_argument(
        "--enable-kitchen-mae-threshold-tuning",
        action="store_true",
        help="Enable experimental kitchen occupancy threshold tuning against daily kitchen_use_minutes MAE",
    )
    parser.add_argument(
        "--enable-kitchen-robust-duration-calibration",
        action="store_true",
        help="Enable experimental kitchen robust duration calibration (shrunken affine)",
    )
    parser.add_argument(
        "--enable-kitchen-temporal-decoder",
        action="store_true",
        help="Enable experimental kitchen temporal decoder",
    )
    parser.add_argument(
        "--enable-kitchen-stage-a-reweighting",
        action="store_true",
        help="Enable experimental kitchen stage-A occupancy sample reweighting",
    )
    parser.add_argument(
        "--enable-bedroom-livingroom-occupancy-decoder",
        action="store_true",
        help="Enable occupancy-first temporal decoder for Bedroom/LivingRoom",
    )
    parser.set_defaults(enable_bedroom_livingroom_hardgate_threshold_tuning=False)
    parser.add_argument(
        "--enable-bedroom-livingroom-hardgate-threshold-tuning",
        dest="enable_bedroom_livingroom_hardgate_threshold_tuning",
        action="store_true",
        help="Enable hard-gate-aligned occupancy-threshold tuning on Bedroom/LivingRoom calibration windows",
    )
    parser.add_argument(
        "--disable-bedroom-livingroom-hardgate-threshold-tuning",
        dest="enable_bedroom_livingroom_hardgate_threshold_tuning",
        action="store_false",
        help="Disable hard-gate-aligned occupancy-threshold tuning on Bedroom/LivingRoom calibration windows",
    )
    parser.set_defaults(enable_bedroom_livingroom_regime_routing=False)
    parser.add_argument(
        "--enable-bedroom-livingroom-regime-routing",
        dest="enable_bedroom_livingroom_regime_routing",
        action="store_true",
        help="Enable regime-routed Stage-A experts (low-occupancy vs normal) for Bedroom/LivingRoom",
    )
    parser.add_argument(
        "--disable-bedroom-livingroom-regime-routing",
        dest="enable_bedroom_livingroom_regime_routing",
        action="store_false",
        help="Disable regime-routed Stage-A experts for Bedroom/LivingRoom",
    )
    parser.add_argument(
        "--bedroom-livingroom-low-occ-thresholds",
        type=str,
        default="bedroom=0.16,livingroom=0.10",
        help="Comma-separated room=low_occupancy_threshold routing prior thresholds",
    )
    parser.set_defaults(enable_bedroom_livingroom_stage_a_hgb=False)
    parser.add_argument(
        "--enable-bedroom-livingroom-stage-a-hgb",
        dest="enable_bedroom_livingroom_stage_a_hgb",
        action="store_true",
        help="Use HistGradientBoosting as Stage-A occupancy learner for Bedroom/LivingRoom",
    )
    parser.add_argument(
        "--disable-bedroom-livingroom-stage-a-hgb",
        dest="enable_bedroom_livingroom_stage_a_hgb",
        action="store_false",
        help="Disable HistGradientBoosting Stage-A learner for Bedroom/LivingRoom",
    )
    parser.set_defaults(enable_bedroom_livingroom_hard_negative_mining=False)
    parser.add_argument(
        "--enable-bedroom-livingroom-hard-negative-mining",
        dest="enable_bedroom_livingroom_hard_negative_mining",
        action="store_true",
        help="Enable Stage-A hard-negative mining (occupied->unoccupied FN boosts) for Bedroom/LivingRoom",
    )
    parser.add_argument(
        "--disable-bedroom-livingroom-hard-negative-mining",
        dest="enable_bedroom_livingroom_hard_negative_mining",
        action="store_false",
        help="Disable Stage-A hard-negative mining for Bedroom/LivingRoom",
    )
    parser.add_argument(
        "--bedroom-livingroom-hard-negative-weight",
        type=float,
        default=2.5,
        help="Weight multiplier for hard-negative boosted Stage-A rows",
    )
    parser.set_defaults(enable_bedroom_livingroom_failure_replay=False)
    parser.add_argument(
        "--enable-bedroom-livingroom-failure-replay",
        dest="enable_bedroom_livingroom_failure_replay",
        action="store_true",
        help="Replay prior split false-negative rows into current fit for Bedroom/LivingRoom",
    )
    parser.add_argument(
        "--disable-bedroom-livingroom-failure-replay",
        dest="enable_bedroom_livingroom_failure_replay",
        action="store_false",
        help="Disable prior split failure replay for Bedroom/LivingRoom",
    )
    parser.add_argument(
        "--bedroom-livingroom-failure-replay-weight",
        type=float,
        default=3.0,
        help="Sample weight for replayed false-negative rows",
    )
    parser.add_argument(
        "--bedroom-livingroom-max-replay-rows-per-day",
        type=int,
        default=1200,
        help="Max replay rows injected from prior failure day per room",
    )
    parser.add_argument(
        "--livingroom-occupied-sample-weight",
        type=float,
        default=1.0,
        help="Additional occupied-window sample weight for LivingRoom Stage-A (>=1.0)",
    )
    parser.set_defaults(enable_livingroom_passive_label_alignment=False)
    parser.add_argument(
        "--enable-livingroom-passive-label-alignment",
        dest="enable_livingroom_passive_label_alignment",
        action="store_true",
        help=(
            "Enable LivingRoom direct-vs-passive supervision weighting "
            "(entry/exit/motion windows keep full weight; passive propagated windows are downweighted)."
        ),
    )
    parser.add_argument(
        "--disable-livingroom-passive-label-alignment",
        dest="enable_livingroom_passive_label_alignment",
        action="store_false",
        help="Disable LivingRoom direct-vs-passive supervision weighting.",
    )
    parser.add_argument(
        "--livingroom-direct-positive-weight",
        type=float,
        default=1.0,
        help="Stage-A sample weight for direct-evidence occupied LivingRoom windows.",
    )
    parser.add_argument(
        "--livingroom-passive-positive-weight",
        type=float,
        default=0.25,
        help="Stage-A sample weight for passive propagated occupied LivingRoom windows.",
    )
    parser.add_argument(
        "--livingroom-unoccupied-weight",
        type=float,
        default=1.0,
        help="Stage-A sample weight for LivingRoom unoccupied windows under passive-label alignment.",
    )
    parser.add_argument(
        "--livingroom-direct-entry-exit-band-windows",
        type=int,
        default=24,
        help="Direct-evidence band size (windows) from episode entry/exit for LivingRoom passive-label alignment.",
    )
    parser.add_argument(
        "--livingroom-direct-motion-threshold",
        type=float,
        default=0.55,
        help="Motion threshold to mark a LivingRoom occupied window as direct-evidence in passive-label alignment.",
    )
    parser.add_argument(
        "--enable-bedroom-livingroom-boundary-reweighting",
        action="store_true",
        help="Enable boundary-emphasis stage-A sample weighting for Bedroom/LivingRoom",
    )
    parser.set_defaults(enable_bedroom_livingroom_stage_a_sequence_model=False)
    parser.add_argument(
        "--enable-bedroom-livingroom-stage-a-sequence-model",
        dest="enable_bedroom_livingroom_stage_a_sequence_model",
        action="store_true",
        help="Enable causal Stage-A sequence model (Markov-filtered temporal RF/HGB) for Bedroom/LivingRoom",
    )
    parser.add_argument(
        "--disable-bedroom-livingroom-stage-a-sequence-model",
        dest="enable_bedroom_livingroom_stage_a_sequence_model",
        action="store_false",
        help="Disable causal Stage-A sequence model for Bedroom/LivingRoom",
    )
    parser.add_argument(
        "--bedroom-livingroom-stage-a-sequence-lag-windows",
        type=int,
        default=12,
        help="Lag windows for Stage-A sequence model on Bedroom/LivingRoom (10s window units)",
    )
    parser.set_defaults(enable_bedroom_livingroom_stage_a_minute_grid=False)
    parser.add_argument(
        "--enable-bedroom-livingroom-stage-a-minute-grid",
        dest="enable_bedroom_livingroom_stage_a_minute_grid",
        action="store_true",
        help="Enable minute-grid Stage-A grouping for Bedroom/LivingRoom (Stage-A only; Stage-B remains 10s)",
    )
    parser.add_argument(
        "--disable-bedroom-livingroom-stage-a-minute-grid",
        dest="enable_bedroom_livingroom_stage_a_minute_grid",
        action="store_false",
        help="Disable minute-grid Stage-A grouping for Bedroom/LivingRoom",
    )
    parser.add_argument(
        "--bedroom-livingroom-stage-a-group-seconds",
        type=int,
        default=60,
        help="Stage-A grouping window size in seconds for Bedroom/LivingRoom minute-grid mode",
    )
    parser.add_argument(
        "--bedroom-livingroom-stage-a-minute-grid-rooms",
        type=str,
        default="bedroom,livingroom",
        help="Comma-separated BL rooms for minute-grid Stage-A grouping (default: bedroom,livingroom)",
    )
    parser.add_argument(
        "--bedroom-livingroom-stage-a-group-occupied-ratio-threshold",
        type=float,
        default=0.50,
        help="Occupied-ratio threshold for grouped Stage-A labels (0.0-1.0)",
    )
    parser.set_defaults(enable_bedroom_livingroom_stage_a_transformer=False)
    parser.add_argument(
        "--enable-bedroom-livingroom-stage-a-transformer",
        dest="enable_bedroom_livingroom_stage_a_transformer",
        action="store_true",
        help="Enable tiny causal Transformer Stage-A occupancy model for Bedroom/LivingRoom",
    )
    parser.add_argument(
        "--disable-bedroom-livingroom-stage-a-transformer",
        dest="enable_bedroom_livingroom_stage_a_transformer",
        action="store_false",
        help="Disable tiny causal Transformer Stage-A occupancy model for Bedroom/LivingRoom",
    )
    parser.add_argument(
        "--bedroom-livingroom-stage-a-transformer-epochs",
        type=int,
        default=8,
        help="Training epochs for tiny Transformer Stage-A model",
    )
    parser.add_argument(
        "--bedroom-livingroom-stage-a-transformer-batch-size",
        type=int,
        default=256,
        help="Batch size for tiny Transformer Stage-A model",
    )
    parser.add_argument(
        "--bedroom-livingroom-stage-a-transformer-learning-rate",
        type=float,
        default=7e-4,
        help="Learning rate for tiny Transformer Stage-A model",
    )
    parser.add_argument(
        "--bedroom-livingroom-stage-a-transformer-hidden-dim",
        type=int,
        default=48,
        help="Hidden dimension for tiny Transformer Stage-A model",
    )
    parser.add_argument(
        "--bedroom-livingroom-stage-a-transformer-num-heads",
        type=int,
        default=2,
        help="Attention heads for tiny Transformer Stage-A model",
    )
    parser.add_argument(
        "--bedroom-livingroom-stage-a-transformer-dropout",
        type=float,
        default=0.15,
        help="Dropout for tiny Transformer Stage-A model",
    )
    parser.add_argument(
        "--bedroom-livingroom-stage-a-transformer-class-weight-power",
        type=float,
        default=0.5,
        help="Power factor for class balancing weights in tiny Transformer Stage-A model (0.0-1.0)",
    )
    parser.add_argument(
        "--bedroom-livingroom-stage-a-transformer-conv-kernel-size",
        type=int,
        default=3,
        help="Causal conv kernel size in tiny Transformer Stage-A model",
    )
    parser.add_argument(
        "--bedroom-livingroom-stage-a-transformer-conv-blocks",
        type=int,
        default=2,
        help="Number of causal conv residual blocks in tiny Transformer Stage-A model",
    )
    parser.set_defaults(enable_bedroom_livingroom_stage_a_transformer_sequence_filter=True)
    parser.add_argument(
        "--enable-bedroom-livingroom-stage-a-transformer-sequence-filter",
        dest="enable_bedroom_livingroom_stage_a_transformer_sequence_filter",
        action="store_true",
        help="Enable Markov smoothing on Stage-A transformer occupancy probabilities",
    )
    parser.add_argument(
        "--disable-bedroom-livingroom-stage-a-transformer-sequence-filter",
        dest="enable_bedroom_livingroom_stage_a_transformer_sequence_filter",
        action="store_false",
        help="Disable Markov smoothing on Stage-A transformer occupancy probabilities",
    )
    parser.add_argument(
        "--enable-bedroom-livingroom-timeline-decoder-v2",
        action="store_true",
        help="Enable timeline decoder v2 for Bedroom/LivingRoom using dynamic label space",
    )
    parser.add_argument(
        "--enable-bedroom-livingroom-timeline-gates",
        action="store_true",
        help="Enable timeline-quality gates for Bedroom/LivingRoom in hard-gate stack",
    )
    parser.set_defaults(enable_bedroom_livingroom_segment_mode=False)
    parser.add_argument(
        "--enable-bedroom-livingroom-segment-mode",
        dest="enable_bedroom_livingroom_segment_mode",
        action="store_true",
        help="Enable segment-based projection mode for Bedroom/LivingRoom labels",
    )
    parser.add_argument(
        "--disable-bedroom-livingroom-segment-mode",
        dest="enable_bedroom_livingroom_segment_mode",
        action="store_false",
        help="Disable segment-based projection mode for Bedroom/LivingRoom labels",
    )
    parser.add_argument(
        "--segment-min-duration-seconds",
        type=int,
        default=60,
        help="Minimum occupied segment duration in seconds for segment mode",
    )
    parser.add_argument(
        "--segment-gap-merge-seconds",
        type=int,
        default=30,
        help="Maximum unoccupied gap (seconds) to merge between occupied segment runs",
    )
    parser.add_argument(
        "--segment-min-activity-prob",
        type=float,
        default=0.35,
        help="Minimum segment-level activity probability to keep occupied label",
    )
    parser.add_argument(
        "--segment-bedroom-min-duration-seconds",
        type=int,
        default=None,
        help="Bedroom override for minimum segment duration (seconds); defaults to --segment-min-duration-seconds",
    )
    parser.add_argument(
        "--segment-bedroom-gap-merge-seconds",
        type=int,
        default=None,
        help="Bedroom override for max gap merge (seconds); defaults to --segment-gap-merge-seconds",
    )
    parser.add_argument(
        "--segment-bedroom-min-activity-prob",
        type=float,
        default=None,
        help="Bedroom override for min activity probability; defaults to --segment-min-activity-prob",
    )
    parser.add_argument(
        "--segment-livingroom-min-duration-seconds",
        type=int,
        default=None,
        help="LivingRoom override for minimum segment duration (seconds); defaults to --segment-min-duration-seconds",
    )
    parser.add_argument(
        "--segment-livingroom-gap-merge-seconds",
        type=int,
        default=None,
        help="LivingRoom override for max gap merge (seconds); defaults to --segment-gap-merge-seconds",
    )
    parser.add_argument(
        "--segment-livingroom-min-activity-prob",
        type=float,
        default=None,
        help="LivingRoom override for min activity probability; defaults to --segment-min-activity-prob",
    )
    parser.add_argument(
        "--segment-livingroom-max-occupied-ratio",
        type=float,
        default=1.15,
        help="LivingRoom guardrail: maximum allowed occupied-window ratio after segment mode vs before",
    )
    parser.add_argument(
        "--segment-livingroom-max-occupied-window-delta",
        type=int,
        default=120,
        help="LivingRoom guardrail: maximum allowed occupied-window delta after segment mode vs before",
    )
    parser.set_defaults(enable_bedroom_livingroom_segment_learned_classifier=False)
    parser.add_argument(
        "--enable-bedroom-livingroom-segment-learned-classifier",
        dest="enable_bedroom_livingroom_segment_learned_classifier",
        action="store_true",
        help="Enable learned segment-level label classifier with confidence/support fallback",
    )
    parser.add_argument(
        "--disable-bedroom-livingroom-segment-learned-classifier",
        dest="enable_bedroom_livingroom_segment_learned_classifier",
        action="store_false",
        help="Disable learned segment-level label classifier",
    )
    parser.add_argument(
        "--segment-classifier-min-segments",
        type=int,
        default=8,
        help="Minimum high-confidence segments required before fitting learned segment classifier",
    )
    parser.add_argument(
        "--segment-classifier-confidence-floor",
        type=float,
        default=0.55,
        help="Minimum learned-classifier confidence required to use learned prediction",
    )
    parser.add_argument(
        "--segment-classifier-min-windows",
        type=int,
        default=6,
        help="Minimum segment support windows required to use learned prediction",
    )
    parser.add_argument(
        "--hard-gate-room-metric-floors",
        type=str,
        default=(
            "bedroom.occupied_f1=0.55,bedroom.occupied_recall=0.50,bedroom.fragmentation_score=0.45,"
            "livingroom.occupied_f1=0.58,livingroom.occupied_recall=0.50,livingroom.fragmentation_score=0.45"
        ),
        help="Comma-separated room.metric=floor hard-gate floors",
    )
    parser.add_argument(
        "--hard-gate-label-recall-floors",
        type=str,
        default="bedroom.sleep=0.40,livingroom.livingroom_normal_use=0.40",
        help="Comma-separated room.label=recall_floor hard-gate checks",
    )
    parser.add_argument(
        "--hard-gate-label-recall-min-supports",
        type=str,
        default="bedroom=600,livingroom=800",
        help="Comma-separated room=min_support for applying label-recall hard gates",
    )
    parser.add_argument(
        "--hard-gate-fragmentation-min-run-windows",
        type=str,
        default="bedroom=9,livingroom=9",
        help="Comma-separated room=min_run_windows for fragmentation smoothing before hard-gate checks",
    )
    parser.add_argument(
        "--hard-gate-fragmentation-gap-fill-windows",
        type=str,
        default="bedroom=6,livingroom=6",
        help="Comma-separated room=gap_fill_windows for fragmentation smoothing before hard-gate checks",
    )
    parser.add_argument(
        "--hard-gate-min-train-days",
        type=int,
        default=0,
        help="Minimum train-day count required for a room-split hard gate to be eligible (0 = strict all cells).",
    )
    parser.set_defaults(enable_bedroom_livingroom_prediction_smoothing=False)
    parser.add_argument(
        "--enable-bedroom-livingroom-prediction-smoothing",
        dest="enable_bedroom_livingroom_prediction_smoothing",
        action="store_true",
        help="Apply occupancy-mask smoothing to final Bedroom/LivingRoom predictions.",
    )
    parser.add_argument(
        "--disable-bedroom-livingroom-prediction-smoothing",
        dest="enable_bedroom_livingroom_prediction_smoothing",
        action="store_false",
        help="Disable occupancy-mask smoothing on final Bedroom/LivingRoom predictions.",
    )
    parser.add_argument(
        "--bedroom-livingroom-prediction-smoothing-min-run-windows",
        type=int,
        default=9,
        help="Minimum occupied run length for Bedroom/LivingRoom prediction smoothing.",
    )
    parser.add_argument(
        "--bedroom-livingroom-prediction-smoothing-gap-fill-windows",
        type=int,
        default=6,
        help="Maximum unoccupied gap length to fill for Bedroom/LivingRoom prediction smoothing.",
    )
    parser.set_defaults(enable_bedroom_livingroom_passive_hysteresis=False)
    parser.add_argument(
        "--enable-bedroom-livingroom-passive-hysteresis",
        dest="enable_bedroom_livingroom_passive_hysteresis",
        action="store_true",
        help="Enable room-specific passive-occupancy hysteresis for Bedroom/LivingRoom.",
    )
    parser.add_argument(
        "--disable-bedroom-livingroom-passive-hysteresis",
        dest="enable_bedroom_livingroom_passive_hysteresis",
        action="store_false",
        help="Disable passive-occupancy hysteresis for Bedroom/LivingRoom.",
    )
    parser.add_argument(
        "--bedroom-passive-hold-minutes",
        type=float,
        default=45.0,
        help="Passive hysteresis hold duration (minutes) for Bedroom.",
    )
    parser.add_argument(
        "--livingroom-passive-hold-minutes",
        type=float,
        default=30.0,
        help="Passive hysteresis hold duration (minutes) for LivingRoom.",
    )
    parser.add_argument(
        "--passive-exit-min-consecutive-windows",
        type=int,
        default=18,
        help="Consecutive low-evidence windows required to force passive-hysteresis exit.",
    )
    parser.add_argument(
        "--passive-entry-occ-threshold",
        type=float,
        default=0.56,
        help="Occupancy-probability threshold to trigger passive-hysteresis entry.",
    )
    parser.add_argument(
        "--passive-entry-room-prob-threshold",
        type=float,
        default=0.40,
        help="Room-label probability threshold to trigger passive-hysteresis entry.",
    )
    parser.add_argument(
        "--passive-stay-occ-threshold",
        type=float,
        default=0.22,
        help="Occupancy-probability threshold to refresh passive-hysteresis hold.",
    )
    parser.add_argument(
        "--passive-stay-room-prob-threshold",
        type=float,
        default=0.10,
        help="Room-label probability threshold to refresh passive-hysteresis hold.",
    )
    parser.add_argument(
        "--passive-exit-occ-threshold",
        type=float,
        default=0.12,
        help="Occupancy-probability ceiling for low-evidence exit checks in passive hysteresis.",
    )
    parser.add_argument(
        "--passive-exit-room-prob-threshold",
        type=float,
        default=0.06,
        help="Room-label probability ceiling for low-evidence exit checks in passive hysteresis.",
    )
    parser.add_argument(
        "--passive-motion-reset-threshold",
        type=float,
        default=0.55,
        help="Motion threshold to refresh passive-hysteresis hold.",
    )
    parser.add_argument(
        "--passive-motion-quiet-threshold",
        type=float,
        default=0.12,
        help="Motion threshold considered quiet for passive-hysteresis exit checks.",
    )
    parser.set_defaults(livingroom_strict_entry_requires_strong_signal=False)
    parser.add_argument(
        "--enable-livingroom-passive-strict-entry",
        dest="livingroom_strict_entry_requires_strong_signal",
        action="store_true",
        help="Require strong evidence for LivingRoom passive-hysteresis entry (reduces weak-entry false positives).",
    )
    parser.add_argument(
        "--disable-livingroom-passive-strict-entry",
        dest="livingroom_strict_entry_requires_strong_signal",
        action="store_false",
        help="Disable strict entry requirement for LivingRoom passive hysteresis.",
    )
    parser.add_argument(
        "--livingroom-passive-entry-motion-threshold",
        type=float,
        default=0.75,
        help="Motion threshold used for strict LivingRoom passive-hysteresis entry.",
    )
    parser.set_defaults(enable_livingroom_cross_room_presence_decoder=False)
    parser.add_argument(
        "--enable-livingroom-cross-room-presence-decoder",
        dest="enable_livingroom_cross_room_presence_decoder",
        action="store_true",
        help="Enable LivingRoom process-of-elimination decoder using cross-room absence/presence evidence.",
    )
    parser.add_argument(
        "--disable-livingroom-cross-room-presence-decoder",
        dest="enable_livingroom_cross_room_presence_decoder",
        action="store_false",
        help="Disable LivingRoom cross-room presence decoder.",
    )
    parser.add_argument(
        "--livingroom-cross-room-supporting-rooms",
        type=str,
        default="bedroom,kitchen,bathroom,entrance",
        help="Comma-separated supporting rooms used by LivingRoom cross-room presence decoder.",
    )
    parser.add_argument(
        "--livingroom-cross-room-hold-minutes",
        type=float,
        default=10.0,
        help="Hold duration after LR evidence before cross-room decoder allows exit.",
    )
    parser.add_argument(
        "--livingroom-cross-room-max-extension-minutes",
        type=float,
        default=24.0,
        help="Maximum passive extension when supporting rooms remain quiet.",
    )
    parser.add_argument(
        "--livingroom-cross-room-entry-occ-threshold",
        type=float,
        default=0.66,
        help="LR occupancy probability threshold to enter cross-room presence state.",
    )
    parser.add_argument(
        "--livingroom-cross-room-entry-room-prob-threshold",
        type=float,
        default=0.42,
        help="LR room-label probability threshold to enter cross-room presence state.",
    )
    parser.add_argument(
        "--livingroom-cross-room-entry-motion-threshold",
        type=float,
        default=0.75,
        help="LR motion threshold to enter/refresh cross-room presence state.",
    )
    parser.add_argument(
        "--livingroom-cross-room-refresh-occ-threshold",
        type=float,
        default=0.44,
        help="LR occupancy probability threshold to refresh cross-room presence hold.",
    )
    parser.add_argument(
        "--livingroom-cross-room-refresh-room-prob-threshold",
        type=float,
        default=0.18,
        help="LR room-label probability threshold to refresh cross-room presence hold.",
    )
    parser.add_argument(
        "--livingroom-cross-room-other-room-exit-occ-threshold",
        type=float,
        default=0.62,
        help="Supporting-room occupancy probability threshold considered explicit exit evidence.",
    )
    parser.add_argument(
        "--livingroom-cross-room-other-room-exit-confirm-windows",
        type=int,
        default=2,
        help="Consecutive windows of supporting-room exit evidence required to end LR hold.",
    )
    parser.add_argument(
        "--livingroom-cross-room-other-room-unoccupied-max-occ-prob",
        type=float,
        default=0.30,
        help="Maximum supporting-room occupancy probability to treat room as quiet.",
    )
    parser.add_argument(
        "--livingroom-cross-room-entrance-exit-occ-threshold",
        type=float,
        default=0.58,
        help="Entrance occupancy probability threshold for immediate LR exit signal.",
    )
    parser.add_argument(
        "--livingroom-cross-room-min-support-rooms",
        type=int,
        default=2,
        help="Minimum supporting rooms required at a timestamp before applying absence logic.",
    )
    parser.set_defaults(livingroom_cross_room_require_other_room_predicted_occupied=True)
    parser.add_argument(
        "--enable-livingroom-cross-room-require-other-room-predicted-occupied",
        dest="livingroom_cross_room_require_other_room_predicted_occupied",
        action="store_true",
        help="Require supporting room predicted occupied in addition to occupancy probability for exit evidence.",
    )
    parser.add_argument(
        "--disable-livingroom-cross-room-require-other-room-predicted-occupied",
        dest="livingroom_cross_room_require_other_room_predicted_occupied",
        action="store_false",
        help="Do not require supporting room predicted occupied for exit evidence (probability-only).",
    )
    parser.set_defaults(enable_livingroom_night_bedroom_sleep_guard=False)
    parser.add_argument(
        "--enable-livingroom-night-bedroom-sleep-guard",
        dest="enable_livingroom_night_bedroom_sleep_guard",
        action="store_true",
        help=(
            "Night-only Bedroom-sleep dominance guardrail: suppress LR occupied when Bedroom sleep is strong, "
            "no Bedroom exit evidence, and no strong LR entry evidence."
        ),
    )
    parser.add_argument(
        "--disable-livingroom-night-bedroom-sleep-guard",
        dest="enable_livingroom_night_bedroom_sleep_guard",
        action="store_false",
        help="Disable night Bedroom-sleep guardrail for LivingRoom cross-room decoder.",
    )
    parser.add_argument(
        "--livingroom-night-bedroom-sleep-guard-hours",
        type=str,
        default="22-6",
        help="Night guard hour window as start-end (24h), e.g. 22-6.",
    )
    parser.add_argument(
        "--livingroom-night-bedroom-sleep-guard-bedroom-sleep-occ-threshold",
        type=float,
        default=0.66,
        help="Bedroom occupancy probability threshold considered strong sleep-state occupancy.",
    )
    parser.add_argument(
        "--livingroom-night-bedroom-sleep-guard-bedroom-sleep-prob-threshold",
        type=float,
        default=0.55,
        help="Bedroom sleep-label probability threshold for strong sleep-state evidence.",
    )
    parser.add_argument(
        "--livingroom-night-bedroom-sleep-guard-bedroom-exit-occ-threshold",
        type=float,
        default=0.35,
        help="Bedroom occupancy probability ceiling for potential Bedroom exit evidence.",
    )
    parser.add_argument(
        "--livingroom-night-bedroom-sleep-guard-bedroom-exit-motion-threshold",
        type=float,
        default=0.75,
        help="Bedroom motion threshold to confirm Bedroom exit evidence.",
    )
    parser.add_argument(
        "--livingroom-night-bedroom-sleep-guard-entry-occ-threshold",
        type=float,
        default=0.66,
        help="Night LR strong-entry occupancy probability threshold (AND with motion threshold).",
    )
    parser.add_argument(
        "--livingroom-night-bedroom-sleep-guard-entry-motion-threshold",
        type=float,
        default=0.75,
        help="Night LR strong-entry motion threshold (AND with occupancy threshold).",
    )
    parser.add_argument(
        "--livingroom-night-bedroom-sleep-guard-entry-confirm-windows",
        type=int,
        default=2,
        help="Consecutive windows required for night LR strong entry.",
    )
    parser.add_argument(
        "--livingroom-night-bedroom-sleep-guard-suppression-label",
        type=str,
        default="unknown",
        choices=["unknown", "unoccupied"],
        help="Label applied when night Bedroom-sleep guard suppresses LR occupied output.",
    )
    parser.add_argument(
        "--livingroom-night-bedroom-sleep-guard-min-bedroom-coverage",
        type=float,
        default=0.80,
        help="Minimum Bedroom timestamp coverage required to activate night guard.",
    )
    parser.add_argument(
        "--livingroom-night-bedroom-sleep-guard-flatline-motion-std-max",
        type=float,
        default=1e-4,
        help="Flatline detector threshold for Bedroom motion std (guard disabled when flatline detected).",
    )
    parser.add_argument(
        "--livingroom-night-bedroom-sleep-guard-flatline-occ-std-max",
        type=float,
        default=1e-4,
        help="Flatline detector threshold for Bedroom occupancy-probability std.",
    )
    parser.add_argument(
        "--livingroom-night-bedroom-sleep-guard-flatline-min-windows",
        type=int,
        default=2160,
        help="Minimum Bedroom windows required before applying flatline health check.",
    )
    parser.add_argument(
        "--enable-single-resident-arbitration",
        action="store_true",
        help="Enable cross-room single-resident arbitration across selected rooms",
    )
    parser.set_defaults(enable_room_temporal_occupancy_features=False)
    parser.add_argument(
        "--enable-room-temporal-occupancy-features",
        dest="enable_room_temporal_occupancy_features",
        action="store_true",
        help="Enable experimental temporal occupancy feature set for Bedroom/LivingRoom",
    )
    parser.add_argument(
        "--disable-room-temporal-occupancy-features",
        dest="enable_room_temporal_occupancy_features",
        action="store_false",
        help="Disable experimental temporal occupancy feature set for Bedroom/LivingRoom",
    )
    parser.add_argument(
        "--bedroom-livingroom-texture-profile",
        type=str,
        default="mixed",
        choices=["30m", "60m", "mixed"],
        help="Texture horizon profile for Bedroom/LivingRoom temporal occupancy features",
    )
    parser.set_defaults(enable_bedroom_light_texture_features=False)
    parser.add_argument(
        "--enable-bedroom-light-texture-features",
        dest="enable_bedroom_light_texture_features",
        action="store_true",
        help="Enable Bedroom-only light texture features (off-streak/regime-switch) when temporal occupancy features are enabled",
    )
    parser.add_argument(
        "--disable-bedroom-light-texture-features",
        dest="enable_bedroom_light_texture_features",
        action="store_false",
        help="Disable Bedroom-only light texture features",
    )
    parser.set_defaults(enable_cross_room_context_features=False)
    parser.add_argument(
        "--enable-cross-room-context-features",
        dest="enable_cross_room_context_features",
        action="store_true",
        help="Enable same-timestamp cross-room context features for selected singleton rooms",
    )
    parser.add_argument(
        "--disable-cross-room-context-features",
        dest="enable_cross_room_context_features",
        action="store_false",
        help="Disable same-timestamp cross-room context features",
    )
    parser.add_argument(
        "--cross-room-context-rooms",
        type=str,
        default="bedroom,livingroom",
        help="Comma-separated rooms that receive cross-room context features",
    )
    parser.add_argument(
        "--single-resident-rooms",
        type=str,
        default="bedroom,livingroom,kitchen",
        help="Comma-separated rooms for single-resident arbitration",
    )
    parser.add_argument(
        "--single-resident-min-margin",
        type=float,
        default=0.03,
        help="Min occupancy probability margin before allowing label-score tie-break in arbitration",
    )
    parser.add_argument(
        "--single-resident-bedroom-night-hours",
        type=str,
        default="22-7",
        help="Night hour range for bedroom sleep guardrail as start-end (e.g., 22-7)",
    )
    parser.add_argument(
        "--single-resident-bedroom-night-min-run-windows",
        type=int,
        default=18,
        help="Minimum contiguous bedroom sleep windows for night guardrail",
    )
    parser.add_argument(
        "--single-resident-bedroom-night-min-score",
        type=float,
        default=0.30,
        help="Minimum bedroom sleep label score for night guardrail",
    )
    parser.add_argument(
        "--single-resident-kitchen-min-score",
        type=float,
        default=0.60,
        help="Minimum kitchen label score for kitchen guardrail",
    )
    parser.add_argument(
        "--single-resident-kitchen-guard-hours",
        type=str,
        default="6-10,11-14,17-20",
        help="Hour ranges where kitchen guardrail is active (comma-separated start-end pairs)",
    )
    parser.add_argument(
        "--single-resident-continuity-min-run-windows",
        type=int,
        default=12,
        help="Minimum contiguous occupied windows for continuity guardrail",
    )
    parser.add_argument(
        "--single-resident-continuity-min-occ-prob",
        type=float,
        default=0.55,
        help="Minimum occupancy probability for continuity guardrail",
    )
    parser.add_argument(
        "--label-corrections-csv",
        type=str,
        default=None,
        help="Optional corrections CSV: room,label,start_time,end_time[,day] to override activity labels",
    )
    parser.add_argument("--disable-threshold-tuning", action="store_true")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    tune_rooms = _parse_tune_rooms(args.tune_rooms)
    room_occupancy_thresholds = _parse_room_thresholds(args.room_occupancy_thresholds)
    room_label_thresholds = _parse_room_label_thresholds(args.room_label_thresholds)
    bedroom_livingroom_low_occ_thresholds = _parse_room_thresholds(
        args.bedroom_livingroom_low_occ_thresholds
    )
    critical_label_rescue_min_scores = _parse_room_label_thresholds(args.critical_label_rescue_min_scores)
    hard_gate_room_metric_floors = _parse_room_label_thresholds(args.hard_gate_room_metric_floors)
    hard_gate_label_recall_floors = _parse_room_label_thresholds(args.hard_gate_label_recall_floors)
    hard_gate_label_recall_min_supports = _parse_room_int_thresholds(
        args.hard_gate_label_recall_min_supports,
        min_value=0,
    )
    hard_gate_fragmentation_min_run_windows = _parse_room_int_thresholds(
        args.hard_gate_fragmentation_min_run_windows,
        min_value=1,
    )
    hard_gate_fragmentation_gap_fill_windows = _parse_room_int_thresholds(
        args.hard_gate_fragmentation_gap_fill_windows,
        min_value=0,
    )
    single_resident_rooms = sorted(_parse_tune_rooms(args.single_resident_rooms))
    stage_a_minute_grid_rooms = sorted(
        room
        for room in _parse_tune_rooms(args.bedroom_livingroom_stage_a_minute_grid_rooms)
        if room in {"bedroom", "livingroom"}
    )
    if not stage_a_minute_grid_rooms:
        stage_a_minute_grid_rooms = ["bedroom", "livingroom"]
    night_start, night_end = _parse_hour_range(
        args.single_resident_bedroom_night_hours,
        default_start=22,
        default_end=7,
    )
    livingroom_night_guard_start, livingroom_night_guard_end = _parse_hour_range(
        args.livingroom_night_bedroom_sleep_guard_hours,
        default_start=22,
        default_end=6,
    )
    kitchen_guard_hours = _parse_hour_ranges(
        args.single_resident_kitchen_guard_hours,
        default_ranges=[(6, 10), (11, 14), (17, 20)],
    )
    cross_room_context_rooms = sorted(_parse_tune_rooms(args.cross_room_context_rooms))
    livingroom_cross_room_supporting_rooms = sorted(
        room
        for room in _parse_tune_rooms(args.livingroom_cross_room_supporting_rooms)
        if room in {"bedroom", "kitchen", "bathroom", "entrance"}
    )
    if not livingroom_cross_room_supporting_rooms:
        livingroom_cross_room_supporting_rooms = ["bedroom", "kitchen", "bathroom", "entrance"]

    report = run_backtest(
        data_dir=Path(args.data_dir),
        elder_id=args.elder_id,
        min_day=args.min_day,
        max_day=args.max_day,
        seed=args.seed,
        occupancy_threshold=args.occupancy_threshold,
        calibration_method=args.calibration_method,
        calib_fraction=args.calib_fraction,
        min_calib_samples=args.min_calib_samples,
        min_calib_label_support=args.min_calib_label_support,
        disable_threshold_tuning=bool(args.disable_threshold_tuning),
        tune_rooms=tune_rooms,
        room_occupancy_thresholds=room_occupancy_thresholds,
        room_label_thresholds=room_label_thresholds,
        critical_label_rescue_min_scores=critical_label_rescue_min_scores,
        enable_duration_calibration=not bool(args.disable_duration_calibration),
        enable_room_mae_threshold_tuning=bool(args.enable_room_mae_threshold_tuning),
        enable_adaptive_room_threshold_policy=not bool(args.disable_adaptive_room_threshold_policy),
        enable_kitchen_mae_threshold_tuning=bool(args.enable_kitchen_mae_threshold_tuning),
        enable_kitchen_robust_duration_calibration=bool(args.enable_kitchen_robust_duration_calibration),
        enable_kitchen_temporal_decoder=bool(args.enable_kitchen_temporal_decoder),
        enable_bedroom_livingroom_boundary_reweighting=bool(args.enable_bedroom_livingroom_boundary_reweighting),
        enable_bedroom_livingroom_timeline_gates=bool(args.enable_bedroom_livingroom_timeline_gates),
        enable_bedroom_livingroom_occupancy_decoder=bool(args.enable_bedroom_livingroom_occupancy_decoder),
        enable_bedroom_livingroom_timeline_decoder_v2=bool(args.enable_bedroom_livingroom_timeline_decoder_v2),
        enable_bedroom_livingroom_hardgate_threshold_tuning=bool(
            args.enable_bedroom_livingroom_hardgate_threshold_tuning
        ),
        enable_bedroom_livingroom_regime_routing=bool(args.enable_bedroom_livingroom_regime_routing),
        bedroom_livingroom_low_occ_thresholds=bedroom_livingroom_low_occ_thresholds,
        enable_bedroom_livingroom_stage_a_hgb=bool(args.enable_bedroom_livingroom_stage_a_hgb),
        enable_bedroom_livingroom_hard_negative_mining=bool(
            args.enable_bedroom_livingroom_hard_negative_mining
        ),
        bedroom_livingroom_hard_negative_weight=float(max(args.bedroom_livingroom_hard_negative_weight, 1.0)),
        enable_bedroom_livingroom_failure_replay=bool(args.enable_bedroom_livingroom_failure_replay),
        bedroom_livingroom_failure_replay_weight=float(max(args.bedroom_livingroom_failure_replay_weight, 1.0)),
        bedroom_livingroom_max_replay_rows_per_day=int(max(args.bedroom_livingroom_max_replay_rows_per_day, 1)),
        livingroom_occupied_sample_weight=float(max(args.livingroom_occupied_sample_weight, 1.0)),
        enable_livingroom_passive_label_alignment=bool(args.enable_livingroom_passive_label_alignment),
        livingroom_direct_positive_weight=float(max(args.livingroom_direct_positive_weight, 0.05)),
        livingroom_passive_positive_weight=float(max(args.livingroom_passive_positive_weight, 0.01)),
        livingroom_unoccupied_weight=float(max(args.livingroom_unoccupied_weight, 0.05)),
        livingroom_direct_entry_exit_band_windows=int(max(args.livingroom_direct_entry_exit_band_windows, 0)),
        livingroom_direct_motion_threshold=float(max(args.livingroom_direct_motion_threshold, 0.0)),
        enable_kitchen_stage_a_reweighting=bool(args.enable_kitchen_stage_a_reweighting),
        hard_gate_room_metric_floors=hard_gate_room_metric_floors,
        hard_gate_label_recall_floors=hard_gate_label_recall_floors,
        hard_gate_label_recall_min_supports=hard_gate_label_recall_min_supports,
        hard_gate_fragmentation_min_run_windows=hard_gate_fragmentation_min_run_windows,
        hard_gate_fragmentation_gap_fill_windows=hard_gate_fragmentation_gap_fill_windows,
        hard_gate_min_train_days=int(max(args.hard_gate_min_train_days, 0)),
        enable_livingroom_cross_room_presence_decoder=bool(
            args.enable_livingroom_cross_room_presence_decoder
        ),
        livingroom_cross_room_supporting_rooms=livingroom_cross_room_supporting_rooms,
        livingroom_cross_room_hold_minutes=float(max(args.livingroom_cross_room_hold_minutes, 0.0)),
        livingroom_cross_room_max_extension_minutes=float(
            max(args.livingroom_cross_room_max_extension_minutes, 0.0)
        ),
        livingroom_cross_room_entry_occ_threshold=float(
            min(max(args.livingroom_cross_room_entry_occ_threshold, 0.0), 1.0)
        ),
        livingroom_cross_room_entry_room_prob_threshold=float(
            min(max(args.livingroom_cross_room_entry_room_prob_threshold, 0.0), 1.0)
        ),
        livingroom_cross_room_entry_motion_threshold=float(
            max(args.livingroom_cross_room_entry_motion_threshold, 0.0)
        ),
        livingroom_cross_room_refresh_occ_threshold=float(
            min(max(args.livingroom_cross_room_refresh_occ_threshold, 0.0), 1.0)
        ),
        livingroom_cross_room_refresh_room_prob_threshold=float(
            min(max(args.livingroom_cross_room_refresh_room_prob_threshold, 0.0), 1.0)
        ),
        livingroom_cross_room_other_room_exit_occ_threshold=float(
            min(max(args.livingroom_cross_room_other_room_exit_occ_threshold, 0.0), 1.0)
        ),
        livingroom_cross_room_other_room_exit_confirm_windows=int(
            max(args.livingroom_cross_room_other_room_exit_confirm_windows, 1)
        ),
        livingroom_cross_room_other_room_unoccupied_max_occ_prob=float(
            min(max(args.livingroom_cross_room_other_room_unoccupied_max_occ_prob, 0.0), 1.0)
        ),
        livingroom_cross_room_entrance_exit_occ_threshold=float(
            min(max(args.livingroom_cross_room_entrance_exit_occ_threshold, 0.0), 1.0)
        ),
        livingroom_cross_room_min_support_rooms=int(max(args.livingroom_cross_room_min_support_rooms, 1)),
        livingroom_cross_room_require_other_room_predicted_occupied=bool(
            args.livingroom_cross_room_require_other_room_predicted_occupied
        ),
        livingroom_night_bedroom_sleep_guard_enabled=bool(
            args.enable_livingroom_night_bedroom_sleep_guard
        ),
        livingroom_night_bedroom_sleep_guard_start_hour=int(livingroom_night_guard_start),
        livingroom_night_bedroom_sleep_guard_end_hour=int(livingroom_night_guard_end),
        livingroom_night_bedroom_sleep_guard_bedroom_sleep_occ_threshold=float(
            min(
                max(args.livingroom_night_bedroom_sleep_guard_bedroom_sleep_occ_threshold, 0.0),
                1.0,
            )
        ),
        livingroom_night_bedroom_sleep_guard_bedroom_sleep_prob_threshold=float(
            min(
                max(args.livingroom_night_bedroom_sleep_guard_bedroom_sleep_prob_threshold, 0.0),
                1.0,
            )
        ),
        livingroom_night_bedroom_sleep_guard_bedroom_exit_occ_threshold=float(
            min(
                max(args.livingroom_night_bedroom_sleep_guard_bedroom_exit_occ_threshold, 0.0),
                1.0,
            )
        ),
        livingroom_night_bedroom_sleep_guard_bedroom_exit_motion_threshold=float(
            max(args.livingroom_night_bedroom_sleep_guard_bedroom_exit_motion_threshold, 0.0)
        ),
        livingroom_night_bedroom_sleep_guard_entry_occ_threshold=float(
            min(max(args.livingroom_night_bedroom_sleep_guard_entry_occ_threshold, 0.0), 1.0)
        ),
        livingroom_night_bedroom_sleep_guard_entry_motion_threshold=float(
            max(args.livingroom_night_bedroom_sleep_guard_entry_motion_threshold, 0.0)
        ),
        livingroom_night_bedroom_sleep_guard_entry_confirm_windows=int(
            max(args.livingroom_night_bedroom_sleep_guard_entry_confirm_windows, 1)
        ),
        livingroom_night_bedroom_sleep_guard_suppression_label=str(
            args.livingroom_night_bedroom_sleep_guard_suppression_label
        ),
        livingroom_night_bedroom_sleep_guard_min_coverage=float(
            min(max(args.livingroom_night_bedroom_sleep_guard_min_bedroom_coverage, 0.0), 1.0)
        ),
        livingroom_night_bedroom_sleep_guard_flatline_motion_std_max=float(
            max(args.livingroom_night_bedroom_sleep_guard_flatline_motion_std_max, 0.0)
        ),
        livingroom_night_bedroom_sleep_guard_flatline_occ_std_max=float(
            max(args.livingroom_night_bedroom_sleep_guard_flatline_occ_std_max, 0.0)
        ),
        livingroom_night_bedroom_sleep_guard_flatline_min_windows=int(
            max(args.livingroom_night_bedroom_sleep_guard_flatline_min_windows, 1)
        ),
        enable_single_resident_arbitration=bool(args.enable_single_resident_arbitration),
        single_resident_rooms=single_resident_rooms,
        single_resident_min_margin=float(min(max(args.single_resident_min_margin, 0.0), 1.0)),
        single_resident_bedroom_night_start_hour=int(night_start),
        single_resident_bedroom_night_end_hour=int(night_end),
        single_resident_bedroom_night_min_run_windows=int(
            max(args.single_resident_bedroom_night_min_run_windows, 1)
        ),
        single_resident_bedroom_night_min_label_score=float(
            min(max(args.single_resident_bedroom_night_min_score, 0.0), 1.0)
        ),
        single_resident_kitchen_min_label_score=float(
            min(max(args.single_resident_kitchen_min_score, 0.0), 1.0)
        ),
        single_resident_kitchen_guard_hours=kitchen_guard_hours,
        single_resident_continuity_min_run_windows=int(
            max(args.single_resident_continuity_min_run_windows, 1)
        ),
        single_resident_continuity_min_occ_prob=float(
            min(max(args.single_resident_continuity_min_occ_prob, 0.0), 1.0)
        ),
        enable_room_temporal_occupancy_features=bool(args.enable_room_temporal_occupancy_features),
        enable_bedroom_light_texture_features=bool(args.enable_bedroom_light_texture_features),
        bedroom_livingroom_texture_profile=str(args.bedroom_livingroom_texture_profile),
        enable_bedroom_livingroom_segment_mode=bool(args.enable_bedroom_livingroom_segment_mode),
        segment_min_duration_seconds=int(max(args.segment_min_duration_seconds, 10)),
        segment_gap_merge_seconds=int(max(args.segment_gap_merge_seconds, 0)),
        segment_min_activity_prob=float(min(max(args.segment_min_activity_prob, 0.0), 1.0)),
        segment_bedroom_min_duration_seconds=int(
            max(
                args.segment_bedroom_min_duration_seconds
                if args.segment_bedroom_min_duration_seconds is not None
                else args.segment_min_duration_seconds,
                10,
            )
        ),
        segment_bedroom_gap_merge_seconds=int(
            max(
                args.segment_bedroom_gap_merge_seconds
                if args.segment_bedroom_gap_merge_seconds is not None
                else args.segment_gap_merge_seconds,
                0,
            )
        ),
        segment_bedroom_min_activity_prob=float(
            min(
                max(
                    args.segment_bedroom_min_activity_prob
                    if args.segment_bedroom_min_activity_prob is not None
                    else args.segment_min_activity_prob,
                    0.0,
                ),
                1.0,
            )
        ),
        segment_livingroom_min_duration_seconds=int(
            max(
                args.segment_livingroom_min_duration_seconds
                if args.segment_livingroom_min_duration_seconds is not None
                else args.segment_min_duration_seconds,
                10,
            )
        ),
        segment_livingroom_gap_merge_seconds=int(
            max(
                args.segment_livingroom_gap_merge_seconds
                if args.segment_livingroom_gap_merge_seconds is not None
                else args.segment_gap_merge_seconds,
                0,
            )
        ),
        segment_livingroom_min_activity_prob=float(
            min(
                max(
                    args.segment_livingroom_min_activity_prob
                    if args.segment_livingroom_min_activity_prob is not None
                    else args.segment_min_activity_prob,
                    0.0,
                ),
                1.0,
            )
        ),
        segment_livingroom_max_occupied_ratio=float(max(args.segment_livingroom_max_occupied_ratio, 1.0)),
        segment_livingroom_max_occupied_window_delta=int(max(args.segment_livingroom_max_occupied_window_delta, 0)),
        enable_bedroom_livingroom_segment_learned_classifier=bool(
            args.enable_bedroom_livingroom_segment_learned_classifier
        ),
        segment_classifier_min_segments=int(max(args.segment_classifier_min_segments, 2)),
        segment_classifier_confidence_floor=float(min(max(args.segment_classifier_confidence_floor, 0.0), 1.0)),
        segment_classifier_min_windows=int(max(args.segment_classifier_min_windows, 1)),
        enable_bedroom_livingroom_prediction_smoothing=bool(
            args.enable_bedroom_livingroom_prediction_smoothing
        ),
        bedroom_livingroom_prediction_smoothing_min_run_windows=int(
            max(args.bedroom_livingroom_prediction_smoothing_min_run_windows, 1)
        ),
        bedroom_livingroom_prediction_smoothing_gap_fill_windows=int(
            max(args.bedroom_livingroom_prediction_smoothing_gap_fill_windows, 0)
        ),
        enable_bedroom_livingroom_passive_hysteresis=bool(
            args.enable_bedroom_livingroom_passive_hysteresis
        ),
        bedroom_passive_hold_minutes=float(max(args.bedroom_passive_hold_minutes, 0.0)),
        livingroom_passive_hold_minutes=float(max(args.livingroom_passive_hold_minutes, 0.0)),
        passive_exit_min_consecutive_windows=int(max(args.passive_exit_min_consecutive_windows, 1)),
        passive_entry_occ_threshold=float(min(max(args.passive_entry_occ_threshold, 0.0), 1.0)),
        passive_entry_room_prob_threshold=float(
            min(max(args.passive_entry_room_prob_threshold, 0.0), 1.0)
        ),
        passive_stay_occ_threshold=float(min(max(args.passive_stay_occ_threshold, 0.0), 1.0)),
        passive_stay_room_prob_threshold=float(
            min(max(args.passive_stay_room_prob_threshold, 0.0), 1.0)
        ),
        passive_exit_occ_threshold=float(min(max(args.passive_exit_occ_threshold, 0.0), 1.0)),
        passive_exit_room_prob_threshold=float(
            min(max(args.passive_exit_room_prob_threshold, 0.0), 1.0)
        ),
        passive_motion_reset_threshold=float(max(args.passive_motion_reset_threshold, 0.0)),
        passive_motion_quiet_threshold=float(max(args.passive_motion_quiet_threshold, 0.0)),
        livingroom_strict_entry_requires_strong_signal=bool(
            args.livingroom_strict_entry_requires_strong_signal
        ),
        livingroom_entry_motion_threshold=float(max(args.livingroom_passive_entry_motion_threshold, 0.0)),
        enable_bedroom_livingroom_stage_a_sequence_model=bool(
            args.enable_bedroom_livingroom_stage_a_sequence_model
        ),
        bedroom_livingroom_stage_a_sequence_lag_windows=int(
            max(args.bedroom_livingroom_stage_a_sequence_lag_windows, 1)
        ),
        enable_bedroom_livingroom_stage_a_minute_grid=bool(
            args.enable_bedroom_livingroom_stage_a_minute_grid
        ),
        bedroom_livingroom_stage_a_minute_grid_rooms=tuple(stage_a_minute_grid_rooms),
        bedroom_livingroom_stage_a_group_seconds=int(
            max(args.bedroom_livingroom_stage_a_group_seconds, 10)
        ),
        bedroom_livingroom_stage_a_group_occupied_ratio_threshold=float(
            min(max(args.bedroom_livingroom_stage_a_group_occupied_ratio_threshold, 0.0), 1.0)
        ),
        enable_bedroom_livingroom_stage_a_transformer=bool(
            args.enable_bedroom_livingroom_stage_a_transformer
        ),
        bedroom_livingroom_stage_a_transformer_epochs=int(
            max(args.bedroom_livingroom_stage_a_transformer_epochs, 1)
        ),
        bedroom_livingroom_stage_a_transformer_batch_size=int(
            max(args.bedroom_livingroom_stage_a_transformer_batch_size, 1)
        ),
        bedroom_livingroom_stage_a_transformer_learning_rate=float(
            max(args.bedroom_livingroom_stage_a_transformer_learning_rate, 1e-5)
        ),
        bedroom_livingroom_stage_a_transformer_hidden_dim=int(
            max(args.bedroom_livingroom_stage_a_transformer_hidden_dim, 8)
        ),
        bedroom_livingroom_stage_a_transformer_num_heads=int(
            max(args.bedroom_livingroom_stage_a_transformer_num_heads, 1)
        ),
        bedroom_livingroom_stage_a_transformer_dropout=float(
            min(max(args.bedroom_livingroom_stage_a_transformer_dropout, 0.0), 0.5)
        ),
        bedroom_livingroom_stage_a_transformer_class_weight_power=float(
            min(max(args.bedroom_livingroom_stage_a_transformer_class_weight_power, 0.0), 1.0)
        ),
        bedroom_livingroom_stage_a_transformer_conv_kernel_size=int(
            max(args.bedroom_livingroom_stage_a_transformer_conv_kernel_size, 2)
        ),
        bedroom_livingroom_stage_a_transformer_conv_blocks=int(
            max(args.bedroom_livingroom_stage_a_transformer_conv_blocks, 1)
        ),
        enable_bedroom_livingroom_stage_a_transformer_sequence_filter=bool(
            args.enable_bedroom_livingroom_stage_a_transformer_sequence_filter
        ),
        enable_cross_room_context_features=bool(args.enable_cross_room_context_features),
        cross_room_context_rooms=cross_room_context_rooms,
        label_corrections_csv=Path(args.label_corrections_csv) if args.label_corrections_csv else None,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote event-first backtest report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
