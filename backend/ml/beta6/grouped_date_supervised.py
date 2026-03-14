"""Reusable Beta6.2 grouped-date supervised prep/evaluation surface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


MANIFEST_SCHEMA_VERSION = "beta62.grouped_date_supervised_manifest.v1"
REPORT_SCHEMA_VERSION = "beta62.grouped_date_supervised_report.v1"
ROOMS = ["Bedroom", "LivingRoom", "Kitchen", "Bathroom", "Entrance"]
VALID_SEGMENT_ROLES = {"baseline", "candidate"}
VALID_SPLITS = {"train", "validation", "calibration", "holdout"}


@dataclass(frozen=True)
class DaySegment:
    role: str
    date: str
    split: str
    path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_mapping(name: str, payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return payload


def _ensure_nonempty_str(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    token = value.strip()
    if not token:
        raise ValueError(f"{name} must be non-empty")
    return token


def _validate_date_token(value: str, *, field_name: str) -> str:
    token = _ensure_nonempty_str(field_name, value)
    parsed = pd.to_datetime(token, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{field_name} must be parseable as a date")
    return parsed.date().isoformat()


def _normalize_room_name(value: str) -> str:
    token = str(value or "").strip().lower()
    for room in ROOMS:
        if room.lower() == token:
            return room.lower()
    return token


def _coerce_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _prepare_room_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp").copy()
    work["activity"] = work["activity"].astype(str).str.strip().str.lower()
    for column in ["motion", "temperature", "light", "sound", "co2", "humidity", "vibration"]:
        work[column] = _coerce_numeric(work, column)
    return work.reset_index(drop=True)


def _load_room_sheet(path: Path, room_name: str) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        source = pd.read_parquet(path)
        if "room" not in source.columns:
            raise ValueError(f"Missing room column in parquet source: {path}")
        room_df = source[source["room"].astype(str).str.strip().str.lower() == room_name.lower()].copy()
    else:
        xls = pd.ExcelFile(path)
        sheet_lookup = {str(sheet).strip().lower(): str(sheet) for sheet in xls.sheet_names}
        sheet_name = sheet_lookup.get(room_name.lower())
        if sheet_name is None:
            raise ValueError(f"Worksheet named '{room_name}' not found in {path}")
        room_df = pd.read_excel(path, sheet_name=sheet_name)
    if "timestamp" not in room_df.columns or "activity" not in room_df.columns:
        raise ValueError(f"Missing required columns in {path} room={room_name}")
    return _prepare_room_frame(room_df)


def _load_day_room_frames(path: Path, *, target_rooms: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for room_name in target_rooms:
        frame = _load_room_sheet(path, room_name)
        if not frame.empty:
            out[room_name] = frame
    return out


def _normalize_sequence_lengths(payload: Any) -> dict[str, int]:
    if payload is None:
        return {}
    mapping = _ensure_mapping("sequence_length_by_room", payload)
    out: dict[str, int] = {}
    for room_name, raw_value in mapping.items():
        room_key = _normalize_room_name(str(room_name))
        out[room_key] = max(int(raw_value), 1)
    return out


def _label_counts(series: pd.Series) -> dict[str, int]:
    counts = series.astype(str).str.strip().str.lower().value_counts()
    return {str(label): int(count) for label, count in counts.sort_index().items()}


def _sequence_count(frame: pd.DataFrame, seq_length: int) -> int:
    return max(int(len(frame)) - int(seq_length) + 1, 0)


def load_grouped_date_supervised_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("grouped-date supervised manifest must decode to an object")
    return payload


def validate_grouped_date_supervised_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = _ensure_mapping("manifest", manifest)
    schema_version = _ensure_nonempty_str("schema_version", payload.get("schema_version"))
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported grouped-date supervised manifest schema: {schema_version}")

    resident_id = _ensure_nonempty_str("resident_id", payload.get("resident_id"))
    raw_target_rooms = payload.get("target_rooms")
    if not isinstance(raw_target_rooms, list) or not raw_target_rooms:
        raise ValueError("target_rooms must be a non-empty list")
    target_rooms = [_ensure_nonempty_str("target_rooms[]", item) for item in raw_target_rooms]

    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("segments must be a non-empty list")

    segments: list[DaySegment] = []
    for index, raw_segment in enumerate(raw_segments):
        item = _ensure_mapping(f"segments[{index}]", raw_segment)
        role = _ensure_nonempty_str(f"segments[{index}].role", item.get("role")).lower()
        if role not in VALID_SEGMENT_ROLES:
            raise ValueError(f"unsupported segment role: {role}")
        split = _ensure_nonempty_str(f"segments[{index}].split", item.get("split")).lower()
        if split not in VALID_SPLITS:
            raise ValueError(f"unsupported segment split: {split}")
        date = _validate_date_token(item.get("date"), field_name=f"segments[{index}].date")
        path = Path(_ensure_nonempty_str(f"segments[{index}].path", item.get("path"))).resolve()
        if not path.exists():
            raise FileNotFoundError(f"segment path not found: {path}")
        segments.append(DaySegment(role=role, date=date, split=split, path=path))

    return {
        "schema_version": schema_version,
        "resident_id": resident_id,
        "target_rooms": list(target_rooms),
        "sequence_length_by_room": _normalize_sequence_lengths(payload.get("sequence_length_by_room")),
        "segments": segments,
        "notes": [
            str(item).strip()
            for item in payload.get("notes", [])
            if str(item).strip()
        ],
    }


def run_grouped_date_supervised(
    manifest: Mapping[str, Any],
    *,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    normalized = validate_grouped_date_supervised_manifest(manifest)
    segments: list[DaySegment] = list(normalized["segments"])
    target_rooms: list[str] = list(normalized["target_rooms"])
    sequence_length_by_room: dict[str, int] = dict(normalized["sequence_length_by_room"])

    artifact_root = Path(artifact_dir).resolve() if artifact_dir is not None else None
    if artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)

    room_reports: dict[str, dict[str, Any]] = {}
    split_frame_buckets: dict[str, dict[str, list[pd.DataFrame]]] = {}
    for room_name in target_rooms:
        room_key = _normalize_room_name(room_name)
        room_reports[room_key] = {"grouped_by_date": []}
        split_frame_buckets[room_key] = {}

    for segment in segments:
        room_dfs = _load_day_room_frames(segment.path, target_rooms=target_rooms)
        for room_name in target_rooms:
            room_key = _normalize_room_name(room_name)
            frame = room_dfs.get(room_name)
            if frame is None or frame.empty:
                continue
            seq_length = int(sequence_length_by_room.get(room_key, 1))
            enriched = frame.copy()
            enriched["__segment_role"] = segment.role
            enriched["__segment_date"] = segment.date
            enriched["__segment_split"] = segment.split
            split_frame_buckets[room_key].setdefault(segment.split, []).append(enriched)
            room_reports[room_key]["grouped_by_date"].append(
                {
                    "date": segment.date,
                    "segment_role": segment.role,
                    "split": segment.split,
                    "path": str(segment.path),
                    "row_count": int(len(frame)),
                    "usable_row_count": int(len(frame)),
                    "sequence_count": _sequence_count(frame, seq_length),
                    "class_counts": _label_counts(frame["activity"]),
                    "start_timestamp": frame["timestamp"].min().isoformat() if len(frame) else None,
                    "end_timestamp": frame["timestamp"].max().isoformat() if len(frame) else None,
                }
            )

    for room_key, report in room_reports.items():
        grouped_rows = sorted(report["grouped_by_date"], key=lambda row: (str(row["date"]), str(row["split"])))
        split_summary: dict[str, dict[str, Any]] = {}
        for split in sorted({row["split"] for row in grouped_rows}):
            matching = [row for row in grouped_rows if row["split"] == split]
            summary: dict[str, Any] = {
                "segment_count": int(len(matching)),
                "dates": [str(row["date"]) for row in matching],
                "row_count": int(sum(int(row["row_count"]) for row in matching)),
                "usable_row_count": int(sum(int(row["usable_row_count"]) for row in matching)),
                "sequence_count": int(sum(int(row["sequence_count"]) for row in matching)),
            }
            if artifact_root is not None and split_frame_buckets[room_key].get(split):
                combined = pd.concat(split_frame_buckets[room_key][split], ignore_index=True)
                artifact_path = artifact_root / f"{normalized['resident_id']}_{room_key}_{split}.parquet"
                combined.to_parquet(artifact_path, index=False)
                summary["artifact_path"] = str(artifact_path)
            split_summary[split] = summary
        report["grouped_by_date"] = grouped_rows
        report["split_summary"] = split_summary

    role_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for segment in segments:
        role_counts[segment.role] = role_counts.get(segment.role, 0) + 1
        split_counts[segment.split] = split_counts.get(segment.split, 0) + 1

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "resident_id": normalized["resident_id"],
        "target_rooms": [_normalize_room_name(room) for room in target_rooms],
        "manifest": {
            "schema_version": normalized["schema_version"],
            "segments": [
                {
                    "role": segment.role,
                    "date": segment.date,
                    "split": segment.split,
                    "path": str(segment.path),
                }
                for segment in segments
            ],
            "notes": list(normalized["notes"]),
        },
        "manifest_summary": {
            "segment_count": len(segments),
            "role_counts": dict(sorted(role_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
        },
        "room_reports": room_reports,
    }


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "load_grouped_date_supervised_manifest",
    "run_grouped_date_supervised",
    "validate_grouped_date_supervised_manifest",
]
