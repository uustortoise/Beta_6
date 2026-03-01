"""Segment-level feature extraction helpers."""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def _finite(value: float) -> float:
    val = float(value)
    if not np.isfinite(val):
        return 0.0
    return float(val)


def _to_numeric_array(values: Sequence[float], *, size: int) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    if len(arr) >= size:
        return np.nan_to_num(arr[:size], nan=0.0, posinf=0.0, neginf=0.0).astype(float)
    out = np.zeros(shape=(size,), dtype=float)
    if len(arr) > 0:
        out[: len(arr)] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
    return out


def _series_slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    n = int(len(arr))
    if n <= 1:
        return 0.0
    x = np.arange(n, dtype=float)
    x_center = x - float(np.mean(x))
    y_center = arr - float(np.mean(arr))
    denom = float(np.sum(x_center * x_center))
    if denom <= 1e-9:
        return 0.0
    return _finite(float(np.sum(x_center * y_center) / denom))


def _add_common_stats(row: Dict[str, float], *, prefix: str, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        row[f"{prefix}_mean"] = 0.0
        row[f"{prefix}_std"] = 0.0
        row[f"{prefix}_min"] = 0.0
        row[f"{prefix}_max"] = 0.0
        row[f"{prefix}_delta"] = 0.0
        row[f"{prefix}_slope"] = 0.0
        return
    row[f"{prefix}_mean"] = _finite(float(np.mean(arr)))
    row[f"{prefix}_std"] = _finite(float(np.std(arr)))
    row[f"{prefix}_min"] = _finite(float(np.min(arr)))
    row[f"{prefix}_max"] = _finite(float(np.max(arr)))
    row[f"{prefix}_delta"] = _finite(float(arr[-1] - arr[0]))
    row[f"{prefix}_slope"] = _series_slope(arr)


def _sanitize_feature_key(name: str) -> str:
    txt = str(name).strip().lower()
    txt = re.sub(r"[^a-z0-9]+", "_", txt)
    return txt.strip("_") or "unknown"


def build_segment_features(
    *,
    segments: Sequence[Dict[str, int]],
    timestamps: Sequence[object],
    occupancy_probs: Sequence[float],
    sensor_series: Optional[Mapping[str, Sequence[float]]] = None,
    activity_probs: Optional[Mapping[str, Sequence[float]]] = None,
    feature_version: str = "v2",
) -> List[Dict[str, float]]:
    ts = pd.to_datetime(pd.Series(list(timestamps)), errors="coerce")
    occ = _to_numeric_array(occupancy_probs, size=len(ts))
    sensors: Dict[str, np.ndarray] = {}
    for key, values in (sensor_series or {}).items():
        name = _sanitize_feature_key(str(key))
        sensors[name] = _to_numeric_array(values, size=len(occ))
    activity_arrays: Dict[str, np.ndarray] = {}
    for key, values in (activity_probs or {}).items():
        name = _sanitize_feature_key(str(key))
        activity_arrays[name] = _to_numeric_array(values, size=len(occ))

    out: List[Dict[str, float]] = []
    for seg in segments:
        start = int(seg.get("start_idx", 0))
        end = int(seg.get("end_idx", 0))
        start = max(start, 0)
        end = min(max(end, start), len(occ))
        if end <= start:
            continue
        occ_seg = occ[start:end]
        duration_windows = int(end - start)
        duration_minutes = float(duration_windows * 10.0 / 60.0)
        start_ts = ts.iloc[start] if start < len(ts) else pd.NaT
        end_ts = ts.iloc[end - 1] if (end - 1) < len(ts) else pd.NaT
        start_hour = int(start_ts.hour) if pd.notna(start_ts) else 0
        end_hour = int(end_ts.hour) if pd.notna(end_ts) else start_hour
        is_night = bool(start_hour >= 22 or start_hour < 7)
        occ_diff = np.diff(occ_seg) if len(occ_seg) > 1 else np.zeros(shape=(0,), dtype=float)

        row: Dict[str, float] = {
            "feature_version": float(2.0 if str(feature_version).strip().lower() == "v2" else 1.0),
            "start_idx": float(start),
            "end_idx": float(end),
            "duration_windows": float(duration_windows),
            "duration_minutes": _finite(duration_minutes),
            "is_night_start": float(1.0 if is_night else 0.0),
            "start_hour_sin": _finite(np.sin(2.0 * np.pi * start_hour / 24.0)),
            "start_hour_cos": _finite(np.cos(2.0 * np.pi * start_hour / 24.0)),
            "end_hour_sin": _finite(np.sin(2.0 * np.pi * end_hour / 24.0)),
            "end_hour_cos": _finite(np.cos(2.0 * np.pi * end_hour / 24.0)),
            # Backward-compatible occupancy keys.
            "occ_mean": _finite(float(np.mean(occ_seg))),
            "occ_max": _finite(float(np.max(occ_seg))),
            # Enriched occupancy-shape features.
            "occ_min": _finite(float(np.min(occ_seg))),
            "occ_std": _finite(float(np.std(occ_seg))),
            "occ_p10": _finite(float(np.percentile(occ_seg, 10))),
            "occ_p50": _finite(float(np.percentile(occ_seg, 50))),
            "occ_p90": _finite(float(np.percentile(occ_seg, 90))),
            "occ_delta": _finite(float(occ_seg[-1] - occ_seg[0])),
            "occ_slope": _series_slope(occ_seg),
            "occ_rise_count": float(np.sum(occ_diff > 0.0)),
            "occ_fall_count": float(np.sum(occ_diff < 0.0)),
        }

        # Add sensor statistics.
        for sensor_name, arr in sensors.items():
            _add_common_stats(row, prefix=f"{sensor_name}", values=arr[start:end])

        # Domain-specific derived sensor ratios.
        motion_arr = sensors.get("motion")
        if motion_arr is not None:
            motion_seg = motion_arr[start:end]
            row["motion_active_ratio"] = _finite(float(np.mean(motion_seg > 0.5)))
            row["motion_burst_count"] = float(np.sum(np.diff(motion_seg > 0.5).astype(int) == 1))
        else:
            row["motion_active_ratio"] = 0.0
            row["motion_burst_count"] = 0.0

        light_arr = sensors.get("light")
        if light_arr is not None:
            light_seg = light_arr[start:end]
            row["light_low_ratio"] = _finite(float(np.mean(light_seg <= 20.0)))
            row["light_high_ratio"] = _finite(float(np.mean(light_seg >= 120.0)))
        else:
            row["light_low_ratio"] = 0.0
            row["light_high_ratio"] = 0.0

        co2_arr = sensors.get("co2")
        if co2_arr is not None:
            co2_seg = co2_arr[start:end]
            co2_diff = np.diff(co2_seg) if len(co2_seg) > 1 else np.zeros(shape=(0,), dtype=float)
            row["co2_rise_count"] = float(np.sum(co2_diff > 0.0))
            row["co2_fall_count"] = float(np.sum(co2_diff < 0.0))
        else:
            row["co2_rise_count"] = 0.0
            row["co2_fall_count"] = 0.0

        temp_arr = sensors.get("temperature")
        humid_arr = sensors.get("humidity")
        if temp_arr is not None and humid_arr is not None:
            row["temp_humidity_delta_product"] = _finite(
                float((temp_arr[end - 1] - temp_arr[start]) * (humid_arr[end - 1] - humid_arr[start]))
            )
        else:
            row["temp_humidity_delta_product"] = 0.0

        # Include segment-level activity score summaries as features.
        for label_name, arr in activity_arrays.items():
            seg_vals = arr[start:end]
            row[f"act_{label_name}_mean"] = _finite(float(np.mean(seg_vals)))
            row[f"act_{label_name}_max"] = _finite(float(np.max(seg_vals)))

        # Guarantee finite numeric payload for downstream model input.
        for key in list(row.keys()):
            row[key] = _finite(float(row[key]))
        out.append(row)
    return out
