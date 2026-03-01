#!/usr/bin/env python3
"""
Pre-upload training file screening for Beta 5.5.

This script is intended as a fast "Stage 0" gate before dropping files into
data/raw. It checks:
1) Per-room data quality contract (schema/timestamp/missingness/basic sanity).
2) Room-label validity (labels allowed for that room taxonomy).
3) Cross-room phantom-gap contradictions (sensor-elevated windows labeled as
   unoccupied/likely wrong room usage).

The script returns non-zero exit code when any file fails blocking thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add backend and repo root to path so local imports work from scripts/
_SCRIPT_PATH = Path(__file__).resolve()
_BACKEND_DIR = _SCRIPT_PATH.parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
for _p in (str(_REPO_ROOT), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml.data_quality_contract import DataQualityContract
from utils.room_utils import normalize_room_name


ROOMS = ["Bedroom", "LivingRoom", "Kitchen", "Bathroom", "Entrance"]
SENSOR_COLS = ["motion", "co2", "light", "sound", "humidity", "temperature", "vibration"]

# Keep this in sync with backend/utils/segment_utils.py.
ROOM_ACTIVITY_VALIDATION = {
    "livingroom": ["inactive", "unoccupied", "livingroom_normal_use", "watch_tv", "low_confidence", "unknown", "out"],
    "bedroom": [
        "inactive",
        "unoccupied",
        "sleep",
        "nap",
        "bedroom_normal_use",
        "room_normal_use",
        "change_clothes",
        "low_confidence",
        "unknown",
        "out",
    ],
    "bathroom": ["inactive", "unoccupied", "bathroom_normal_use", "shower", "toilet", "low_confidence", "unknown", "out"],
    "kitchen": [
        "inactive",
        "unoccupied",
        "kitchen_normal_use",
        "cooking",
        "washing_dishes",
        "low_confidence",
        "unknown",
        "out",
    ],
    "entrance": ["inactive", "unoccupied", "out", "low_confidence", "unknown"],
    "default": ["inactive", "unoccupied", "low_confidence", "unknown", "out"],
}

REQUIRED_SENSOR_COLUMNS = ["motion", "temperature", "light", "sound", "co2", "humidity", "vibration"]


@dataclass
class ScreeningThresholds:
    min_duration_minutes: int
    max_phantom_episodes: int
    max_phantom_minutes: float
    max_missingness_ratio: float
    min_label_samples: int
    max_duplicate_ratio: float
    min_timestamp_range_days: int
    require_all_core_rooms: bool


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _load_room_sheet(path: Path, room_name: str) -> pd.DataFrame | None:
    try:
        df = pd.read_excel(path, sheet_name=room_name)
    except Exception:
        return None

    if "timestamp" not in df.columns or "activity" not in df.columns:
        # Keep dataframe for quality contract to report missing columns, but ensure
        # key columns exist for downstream safe handling.
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.NaT
        if "activity" not in df.columns:
            df["activity"] = ""

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["activity"] = df["activity"].astype(str).str.strip().str.lower()
    for s in SENSOR_COLS:
        df[s] = _as_numeric(df, s)
    df["occ"] = (df["activity"] != "unoccupied").astype(int)
    return df


def _calc_room_score(df: pd.DataFrame, room_name: str) -> np.ndarray:
    motion = (_as_numeric(df, "motion") > 0.5).astype(int)
    light = _as_numeric(df, "light")
    sound = _as_numeric(df, "sound")
    co2 = _as_numeric(df, "co2")
    humidity = _as_numeric(df, "humidity")
    hour = pd.to_datetime(df["timestamp"], errors="coerce").dt.hour
    is_night = (hour >= 20) | (hour <= 9)
    is_sleeping = (is_night & (light <= 10) & (co2 > 1200)).astype(int)

    room_key = normalize_room_name(room_name)
    if room_key == "bedroom":
        has_light = (light > 50).astype(int)
        has_sound = (sound > 4.0).astype(int)
        has_co2_or_hum = ((co2 > 2800) | (humidity > 35)).astype(int)
        base = motion + has_light + has_sound + has_co2_or_hum
        return np.maximum(base.to_numpy(dtype=int), (is_sleeping * 4).to_numpy(dtype=int))
    if room_key in {"kitchen", "bathroom"}:
        has_light = (light > 200).astype(int)
        has_sound = (sound > 4.2).astype(int)
        has_humidity = (humidity > 40).astype(int)
        base = motion + has_light + has_sound + has_humidity
        return base.to_numpy(dtype=int)

    has_light = (light > 500).astype(int)
    has_sound = (sound > 4.2).astype(int)
    has_co2 = (co2 > 3100).astype(int)
    base = motion + has_light + has_sound + has_co2
    return np.maximum(base.to_numpy(dtype=int), (is_sleeping * 4).to_numpy(dtype=int))


def _detect_phantom_episodes(
    room_dfs: dict[str, pd.DataFrame],
    *,
    min_duration_minutes: int,
) -> list[dict[str, Any]]:
    if not room_dfs:
        return []

    # Union timestamps across loaded rooms for robust cross-room overlap checks.
    ts_union = pd.Index([])
    for rdf in room_dfs.values():
        ts_union = ts_union.union(pd.Index(pd.to_datetime(rdf["timestamp"], errors="coerce").dropna().unique()))
    master_df = pd.DataFrame({"timestamp": ts_union.sort_values()})

    for room, rdf in room_dfs.items():
        occ_df = rdf[["timestamp", "occ"]].copy()
        occ_df.columns = ["timestamp", f"{room}_occ"]
        master_df = pd.merge(master_df, occ_df, on="timestamp", how="left")

    episodes: list[dict[str, Any]] = []

    for target_room, df_tgt in room_dfs.items():
        work = df_tgt.copy()
        work["score"] = _calc_room_score(work, target_room)
        room_key = normalize_room_name(target_room)

        if room_key == "bedroom":
            # For bedroom, anything not bedroom_normal_use is candidate for missing
            # occupancy/sleep episodes.
            candidate = work[work["activity"] != "bedroom_normal_use"].copy()
        else:
            candidate = work[work["activity"] == "unoccupied"].copy()
        suspicious = candidate[candidate["score"] >= 2]
        if suspicious.empty:
            continue

        merged = work[
            ["timestamp", "activity", "occ", "motion", "light", "co2", "sound", "humidity", "score"]
        ].copy()
        merged = pd.merge(merged, master_df, on="timestamp", how="left")

        susp_ts = set(pd.to_datetime(suspicious["timestamp"], errors="coerce").dropna().values)
        susp_merged = merged[merged["timestamp"].isin(susp_ts)].sort_values("timestamp").copy()
        if susp_merged.empty:
            continue

        other_occ_cols = [c for c in master_df.columns if c != "timestamp" and c != f"{target_room}_occ"]
        if other_occ_cols:
            susp_merged["any_other_occ"] = (susp_merged[other_occ_cols].fillna(0).sum(axis=1) > 0).astype(bool)
        else:
            susp_merged["any_other_occ"] = False

        ts_arr = susp_merged["timestamp"].to_numpy()
        if len(ts_arr) == 0:
            continue

        segments: list[tuple[int, int]] = []
        seg_start = 0
        for i in range(1, len(ts_arr)):
            gap_min = (ts_arr[i] - ts_arr[i - 1]) / np.timedelta64(1, "m")
            hour = pd.Timestamp(ts_arr[i]).hour
            is_night = (hour >= 20) or (hour <= 9)
            gap_tolerance = 20 if (is_night and room_key == "bedroom") else 2
            if gap_min > gap_tolerance:
                segments.append((seg_start, i))
                seg_start = i
        segments.append((seg_start, len(ts_arr)))

        for s, e in segments:
            ep = susp_merged.iloc[s:e]
            if ep.empty:
                continue
            duration_min = float(len(ep) * 10 / 60)
            if duration_min < float(min_duration_minutes):
                continue

            other_occ_ratio = float(ep["any_other_occ"].astype(int).mean())
            if other_occ_ratio > 0.15:
                # Likely legit temporary transition leakage.
                continue

            t0 = pd.Timestamp(ep["timestamp"].iloc[0])
            t1 = pd.Timestamp(ep["timestamp"].iloc[-1])

            mot_mean = float(pd.to_numeric(ep["motion"], errors="coerce").mean())
            mot_max = float(pd.to_numeric(ep["motion"], errors="coerce").max())
            lgt_mean = float(pd.to_numeric(ep["light"], errors="coerce").mean())
            snd_mean = float(pd.to_numeric(ep["sound"], errors="coerce").mean())
            co2_mean = float(pd.to_numeric(ep["co2"], errors="coerce").mean())
            hum_mean = float(pd.to_numeric(ep["humidity"], errors="coerce").mean())

            flags: list[str] = []
            ep_hour = t0.hour
            is_night = (ep_hour >= 20) or (ep_hour <= 9)
            if is_night and lgt_mean <= 15 and co2_mean > 1000 and mot_mean < 25:
                flags.append("SLEEP_SIGNATURE")
            else:
                if mot_mean > 1.0 or mot_max > 50:
                    flags.append("MOTION")
                if lgt_mean > (50 if room_key == "bedroom" else 800):
                    flags.append("LIGHTS_ON")
                if snd_mean > 4.5:
                    flags.append("HIGH_SOUND")
                if room_key in {"bathroom", "kitchen"} and hum_mean > 45:
                    flags.append("HIGH_HUMIDITY")
                if room_key in {"livingroom", "bedroom"} and co2_mean > 3200:
                    flags.append("HIGH_CO2")

            if "SLEEP_SIGNATURE" in flags or (room_key == "bedroom" and lgt_mean < 50 and mot_mean < 25):
                rec_label = "sleep"
            elif room_key == "bedroom":
                rec_label = "bedroom_normal_use"
            else:
                rec_label = f"{room_key}_normal_use"

            episodes.append(
                {
                    "room": target_room,
                    "start_time": str(t0),
                    "end_time": str(t1),
                    "duration_minutes": round(duration_min, 1),
                    "flags": flags or ["AMBIENT_ELEVATED"],
                    "recommended_label": rec_label,
                    "metrics": {
                        "motion_avg": round(mot_mean, 3),
                        "motion_max": round(mot_max, 3),
                        "light_avg": round(lgt_mean, 3),
                        "sound_avg": round(snd_mean, 3),
                        "co2_avg": round(co2_mean, 3),
                        "humidity_avg": round(hum_mean, 3),
                        "other_room_occupied_ratio": round(other_occ_ratio, 3),
                    },
                }
            )
    return episodes


def _room_label_violations(df: pd.DataFrame, room_name: str) -> list[str]:
    room_key = normalize_room_name(room_name)
    allowed = set(ROOM_ACTIVITY_VALIDATION.get(room_key, ROOM_ACTIVITY_VALIDATION["default"]))
    labels = set(df["activity"].astype(str).str.strip().str.lower().unique().tolist())
    invalid = sorted(label for label in labels if label and label not in allowed)
    return invalid


def _evaluate_file(path: Path, thresholds: ScreeningThresholds) -> dict[str, Any]:
    out: dict[str, Any] = {
        "file": str(path),
        "pass": True,
        "blocking_reasons": [],
        "warnings": [],
        "rooms_loaded": [],
        "missing_core_rooms": [],
        "room_reports": {},
        "phantom_summary": {"episodes": 0, "total_minutes": 0.0, "details": []},
    }

    room_dfs: dict[str, pd.DataFrame] = {}
    for room in ROOMS:
        df = _load_room_sheet(path, room)
        if df is not None:
            room_dfs[room] = df
            out["rooms_loaded"].append(room)

    if not room_dfs:
        out["pass"] = False
        out["blocking_reasons"].append("no_readable_room_sheets")
        return out

    missing = sorted(set(ROOMS) - set(out["rooms_loaded"]))
    out["missing_core_rooms"] = missing
    if thresholds.require_all_core_rooms and missing:
        out["pass"] = False
        out["blocking_reasons"].append(f"missing_core_rooms:{','.join(missing)}")

    contract = DataQualityContract(
        required_sensor_columns=list(REQUIRED_SENSOR_COLUMNS),
        required_label_columns=["activity"],
        max_missingness_ratio=float(thresholds.max_missingness_ratio),
        min_label_samples=int(thresholds.min_label_samples),
        max_duplicate_ratio=float(thresholds.max_duplicate_ratio),
        min_timestamp_range_days=int(thresholds.min_timestamp_range_days),
        fail_on_critical=True,
    )

    for room, df in room_dfs.items():
        report = contract.validate(df, elder_id="preupload", room_name=room)
        invalid_labels = _room_label_violations(df, room)
        room_pass = bool(report.passes and len(invalid_labels) == 0)

        room_entry: dict[str, Any] = {
            "pass": room_pass,
            "quality_contract_pass": bool(report.passes),
            "violation_summary": {
                "critical": int(report.critical_violations),
                "high": int(report.high_violations),
                "medium": int(report.medium_violations),
                "low": int(report.low_violations),
            },
            "invalid_room_labels": invalid_labels,
            "check_results": {
                "required_columns": bool(report.required_columns_pass),
                "timestamp_monotonicity": bool(report.timestamp_monotonicity_pass),
                "sensor_missingness": bool(report.sensor_missingness_pass),
                "label_distribution": bool(report.label_distribution_pass),
                "timestamp_duplicates": bool(report.timestamp_duplicates_pass),
                "timestamp_range": bool(report.timestamp_range_pass),
                "value_range": bool(report.value_range_pass),
                "label_validity": bool(report.label_validity_pass),
            },
        }
        out["room_reports"][room] = room_entry

        if not room_pass:
            out["pass"] = False
            if not report.passes:
                out["blocking_reasons"].append(
                    f"room_contract_failed:{normalize_room_name(room)}:"
                    f"critical={report.critical_violations},high={report.high_violations}"
                )
            if invalid_labels:
                out["blocking_reasons"].append(
                    f"invalid_room_labels:{normalize_room_name(room)}:{','.join(invalid_labels)}"
                )

    phantom = _detect_phantom_episodes(room_dfs, min_duration_minutes=thresholds.min_duration_minutes)
    total_minutes = float(sum(float(ep.get("duration_minutes", 0.0)) for ep in phantom))
    out["phantom_summary"] = {
        "episodes": int(len(phantom)),
        "total_minutes": round(total_minutes, 2),
        "details": phantom,
    }

    if len(phantom) > int(thresholds.max_phantom_episodes):
        out["pass"] = False
        out["blocking_reasons"].append(
            f"phantom_episode_budget_exceeded:{len(phantom)}>{thresholds.max_phantom_episodes}"
        )
    if total_minutes > float(thresholds.max_phantom_minutes):
        out["pass"] = False
        out["blocking_reasons"].append(
            f"phantom_minutes_budget_exceeded:{total_minutes:.1f}>{thresholds.max_phantom_minutes:.1f}"
        )

    return out


def _print_file_summary(res: dict[str, Any]) -> None:
    status = "PASS" if bool(res.get("pass")) else "FAIL"
    print(f"\n[{status}] {res.get('file')}")
    print(
        f"  rooms_loaded={len(res.get('rooms_loaded', []))}, "
        f"phantom_episodes={res.get('phantom_summary', {}).get('episodes', 0)}, "
        f"phantom_minutes={res.get('phantom_summary', {}).get('total_minutes', 0.0)}"
    )
    reasons = list(res.get("blocking_reasons", []) or [])
    if reasons:
        print("  blocking_reasons:")
        for reason in reasons:
            print(f"    - {reason}")


def _collect_files(file_arg: str | None, dir_arg: str | None) -> list[Path]:
    if file_arg:
        p = Path(file_arg).expanduser().resolve()
        return [p]
    if dir_arg:
        d = Path(dir_arg).expanduser().resolve()
        return sorted(d.glob("*.xlsx"))
    return []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Screen training file quality before upload (Beta 5.5).")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--file", type=str, help="Single training .xlsx file path.")
    target.add_argument("--dir", type=str, help="Directory containing training .xlsx files.")

    p.add_argument("--min-duration-minutes", type=int, default=5)
    p.add_argument("--max-phantom-episodes", type=int, default=0)
    p.add_argument("--max-phantom-minutes", type=float, default=0.0)
    p.add_argument("--max-missingness-ratio", type=float, default=0.30)
    p.add_argument("--min-label-samples", type=int, default=5)
    p.add_argument("--max-duplicate-ratio", type=float, default=0.10)
    p.add_argument(
        "--min-timestamp-range-days",
        type=int,
        default=0,
        help="Use 0 for per-file screening (aggregate run enforces stronger temporal gates).",
    )
    p.add_argument(
        "--require-all-core-rooms",
        action="store_true",
        help="Fail if any of Bedroom/LivingRoom/Kitchen/Bathroom/Entrance sheets are missing.",
    )
    p.add_argument("--json-output", type=str, default="", help="Optional path to write JSON report.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    files = _collect_files(args.file, args.dir)
    if not files:
        print("No input .xlsx files found.")
        return 2

    thresholds = ScreeningThresholds(
        min_duration_minutes=max(1, int(args.min_duration_minutes)),
        max_phantom_episodes=max(0, int(args.max_phantom_episodes)),
        max_phantom_minutes=max(0.0, float(args.max_phantom_minutes)),
        max_missingness_ratio=float(min(max(args.max_missingness_ratio, 0.0), 1.0)),
        min_label_samples=max(1, int(args.min_label_samples)),
        max_duplicate_ratio=float(min(max(args.max_duplicate_ratio, 0.0), 1.0)),
        min_timestamp_range_days=max(0, int(args.min_timestamp_range_days)),
        require_all_core_rooms=bool(args.require_all_core_rooms),
    )

    results: list[dict[str, Any]] = []
    any_fail = False

    for file_path in files:
        if not file_path.exists() or not file_path.is_file():
            any_fail = True
            results.append(
                {
                    "file": str(file_path),
                    "pass": False,
                    "blocking_reasons": ["file_not_found"],
                    "warnings": [],
                    "rooms_loaded": [],
                    "missing_core_rooms": [],
                    "room_reports": {},
                    "phantom_summary": {"episodes": 0, "total_minutes": 0.0, "details": []},
                }
            )
            continue

        res = _evaluate_file(file_path, thresholds)
        results.append(res)
        _print_file_summary(res)
        if not bool(res.get("pass", False)):
            any_fail = True

    payload = {
        "generated_at_utc": _utc_now_iso_z(),
        "thresholds": {
            "min_duration_minutes": thresholds.min_duration_minutes,
            "max_phantom_episodes": thresholds.max_phantom_episodes,
            "max_phantom_minutes": thresholds.max_phantom_minutes,
            "max_missingness_ratio": thresholds.max_missingness_ratio,
            "min_label_samples": thresholds.min_label_samples,
            "max_duplicate_ratio": thresholds.max_duplicate_ratio,
            "min_timestamp_range_days": thresholds.min_timestamp_range_days,
            "require_all_core_rooms": thresholds.require_all_core_rooms,
        },
        "summary": {
            "files_total": len(results),
            "files_pass": sum(1 for r in results if bool(r.get("pass", False))),
            "files_fail": sum(1 for r in results if not bool(r.get("pass", False))),
        },
        "files": results,
    }

    if args.json_output:
        out_path = Path(args.json_output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nWrote report: {out_path}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
