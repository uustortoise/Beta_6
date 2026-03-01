#!/usr/bin/env python3
"""Validate incoming training label pack before backtest/matrix execution."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import yaml

DAY_RE = re.compile(r"train_(\d{1,2})(?:[a-z]{3}\d{4})?$", re.IGNORECASE)
ROOMS = ["Bedroom", "LivingRoom", "Kitchen", "Bathroom", "Entrance"]
REQUIRED_COLS = ("timestamp", "activity")

# Conservative union for current Beta 5.5 label taxonomy used in training data.
DEFAULT_VALID_LABELS = {
    "inactive",
    "unoccupied",
    "unknown",
    "low_confidence",
    "bathroom_normal_use",
    "shower",
    "toilet",
    "kitchen_normal_use",
    "cooking",
    "washing_dishes",
    "livingroom_normal_use",
    "watch_tv",
    "nap",
    "bedroom_normal_use",
    "sleep",
    "change_clothes",
    "out",
    "fall",
}


def _extract_day(path: Path) -> int | None:
    m = DAY_RE.search(path.stem)
    return int(m.group(1)) if m else None


def _canonical_train_files(pack_dir: Path, elder_id: str) -> tuple[List[Path], List[Dict[str, str]]]:
    candidate_files: List[Path] = []
    for ext in ("xlsx", "xls", "parquet"):
        candidate_files.extend(sorted(pack_dir.glob(f"{elder_id}_train_*.{ext}")))

    canonical: List[Path] = []
    excluded: List[Dict[str, str]] = []
    prefix = f"{elder_id}_train_"
    for p in sorted({str(path.resolve()): path for path in candidate_files}.values(), key=lambda x: x.name.lower()):
        stem = p.stem
        if not stem.startswith(prefix):
            excluded.append({"file": p.name, "reason": "prefix_mismatch"})
            continue
        suffix = stem[len(prefix):]
        if "_" in suffix:
            excluded.append({"file": p.name, "reason": "derived_or_noncanonical_suffix"})
            continue
        canonical.append(p)
    return canonical, excluded


def _load_valid_labels(registry_path: Path | None) -> tuple[set[str], List[str]]:
    labels = {str(v).strip().lower() for v in DEFAULT_VALID_LABELS}
    warnings: List[str] = []
    if registry_path is None:
        return labels, warnings
    if not registry_path.exists():
        warnings.append(f"registry_not_found:{registry_path}")
        return labels, warnings
    try:
        data = yaml.safe_load(registry_path.read_text()) or {}
        registry = data.get("registry", {}) if isinstance(data, dict) else {}
        if isinstance(registry, dict):
            for key, payload in registry.items():
                labels.add(str(key).strip().lower())
                if isinstance(payload, dict):
                    event_id = payload.get("event_id")
                    if event_id:
                        labels.add(str(event_id).strip().lower())
                    for alias in payload.get("aliases", []) or []:
                        labels.add(str(alias).strip().lower())
    except Exception as exc:
        warnings.append(f"registry_load_failed:{type(exc).__name__}")
    return labels, warnings


def validate_label_pack(
    *,
    pack_dir: Path,
    elder_id: str,
    min_day: int,
    max_day: int,
    registry_path: Path | None = None,
) -> Dict[str, object]:
    violations: List[str] = []
    warnings: List[str] = []

    canonical_files, excluded_non_canonical = _canonical_train_files(pack_dir, elder_id)
    files_by_day: Dict[int, Path] = {}
    invalid_day_tokens: List[str] = []

    for p in canonical_files:
        day = _extract_day(p)
        if day is None:
            invalid_day_tokens.append(p.name)
            continue
        if not (int(min_day) <= int(day) <= int(max_day)):
            continue
        if day in files_by_day:
            violations.append(f"duplicate_day_token:day{day}:{files_by_day[day].name}:{p.name}")
            continue
        files_by_day[day] = p

    selected_days = sorted(files_by_day.keys())
    requested_days = [int(d) for d in range(int(min_day), int(max_day) + 1)]
    missing_days = [int(d) for d in requested_days if d not in set(selected_days)]

    if not selected_days:
        violations.append("no_canonical_files_in_day_window")

    valid_labels, label_warnings = _load_valid_labels(registry_path)
    warnings.extend(label_warnings)

    schema_audit: Dict[str, Dict[str, object]] = {}
    label_audit: Dict[str, object] = {
        "unknown_labels_by_room": {},
        "empty_activity_rows_by_room": {},
        "rooms_with_unknown_labels": [],
    }

    allowed_rooms = {str(v).strip().lower() for v in ROOMS}

    for day in selected_days:
        path = files_by_day[day]
        day_key = f"day{day}"
        schema_audit[day_key] = {
            "file": path.name,
            "rooms_present": [],
            "rooms_missing": [],
            "unknown_room_sheets": [],
            "room_checks": {},
        }
        try:
            xls = pd.ExcelFile(path)
        except Exception as exc:
            violations.append(f"excel_open_failed:{path.name}:{type(exc).__name__}")
            continue

        sheets = list(xls.sheet_names)
        sheet_norm_to_raw = {str(sheet).strip().lower(): str(sheet) for sheet in sheets}
        unknown_sheets = sorted([sheet for sheet in sheets if str(sheet).strip().lower() not in allowed_rooms])
        if unknown_sheets:
            schema_audit[day_key]["unknown_room_sheets"] = unknown_sheets
            violations.append(f"unknown_room_sheets:{path.name}:{','.join(unknown_sheets)}")

        for room in ROOMS:
            room_key = str(room).strip().lower()
            room_raw = sheet_norm_to_raw.get(room_key)
            if room_raw is None:
                cast_missing = schema_audit[day_key]["rooms_missing"]
                if isinstance(cast_missing, list):
                    cast_missing.append(room)
                violations.append(f"missing_room_sheet:{path.name}:{room}")
                continue

            cast_present = schema_audit[day_key]["rooms_present"]
            if isinstance(cast_present, list):
                cast_present.append(room)

            try:
                df = pd.read_excel(path, sheet_name=room_raw)
            except Exception as exc:
                violations.append(f"room_read_failed:{path.name}:{room}:{type(exc).__name__}")
                continue

            room_checks: Dict[str, object] = {
                "rows": int(len(df)),
                "required_columns_present": False,
                "timestamp_parse_fail_rows": 0,
                "activity_empty_rows": 0,
                "unknown_labels": [],
            }

            lower_cols = {str(c).strip().lower(): str(c) for c in df.columns}
            if not all(col in lower_cols for col in REQUIRED_COLS):
                violations.append(f"missing_required_columns:{path.name}:{room}")
                cast_checks = schema_audit[day_key]["room_checks"]
                if isinstance(cast_checks, dict):
                    cast_checks[room] = room_checks
                continue

            room_checks["required_columns_present"] = True
            ts_col = lower_cols["timestamp"]
            activity_col = lower_cols["activity"]
            ts = pd.to_datetime(df[ts_col], errors="coerce")
            bad_ts = int(ts.isna().sum())
            room_checks["timestamp_parse_fail_rows"] = int(bad_ts)
            if bad_ts > 0:
                violations.append(f"timestamp_parse_fail:{path.name}:{room}:{bad_ts}")

            activities = df[activity_col].astype(str).str.strip().str.lower()
            empty_mask = activities.isna() | activities.eq("") | activities.eq("nan")
            empty_rows = int(empty_mask.sum())
            room_checks["activity_empty_rows"] = int(empty_rows)
            if empty_rows > 0:
                violations.append(f"empty_activity_rows:{path.name}:{room}:{empty_rows}")

            observed = sorted({str(v).strip().lower() for v in activities.tolist() if str(v).strip() and str(v).strip() != "nan"})
            unknown_labels = sorted([label for label in observed if label not in valid_labels])
            room_checks["unknown_labels"] = unknown_labels
            if unknown_labels:
                violations.append(
                    f"unknown_labels:{path.name}:{room}:{','.join(unknown_labels)}"
                )
                unknown_by_room = label_audit["unknown_labels_by_room"]
                if isinstance(unknown_by_room, dict):
                    current = set(unknown_by_room.get(room_key, []))
                    current.update(unknown_labels)
                    unknown_by_room[room_key] = sorted(current)

            empty_by_room = label_audit["empty_activity_rows_by_room"]
            if isinstance(empty_by_room, dict):
                empty_by_room[room_key] = int(empty_by_room.get(room_key, 0)) + int(empty_rows)

            cast_checks = schema_audit[day_key]["room_checks"]
            if isinstance(cast_checks, dict):
                cast_checks[room] = room_checks

    unknown_by_room = label_audit.get("unknown_labels_by_room", {})
    if isinstance(unknown_by_room, dict):
        label_audit["rooms_with_unknown_labels"] = sorted(list(unknown_by_room.keys()))

    file_audit = {
        "pack_dir": str(pack_dir),
        "elder_id": str(elder_id),
        "requested_day_window": {"min_day": int(min_day), "max_day": int(max_day), "days": requested_days},
        "canonical_file_count": int(len(canonical_files)),
        "selected_file_count": int(len(selected_days)),
        "selected_days": selected_days,
        "missing_days_in_requested_window": missing_days,
        "invalid_day_token_files": invalid_day_tokens,
        "excluded_non_canonical_files": excluded_non_canonical,
    }

    if invalid_day_tokens:
        for item in invalid_day_tokens:
            violations.append(f"invalid_day_token:{item}")

    report: Dict[str, object] = {
        "status": "pass" if len(violations) == 0 else "fail",
        "violations": sorted(set(violations)),
        "warnings": sorted(set(warnings)),
        "file_audit": file_audit,
        "schema_audit": schema_audit,
        "label_audit": label_audit,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate incoming training label pack")
    parser.add_argument("--pack-dir", required=True, help="Directory containing candidate training files")
    parser.add_argument("--elder-id", required=True, help="Elder id prefix in filenames")
    parser.add_argument("--min-day", type=int, required=True)
    parser.add_argument("--max-day", type=int, required=True)
    parser.add_argument("--registry-path", default="backend/config/adl_event_registry.v1.yaml")
    parser.add_argument("--output", required=True, help="Output JSON report path")
    args = parser.parse_args()

    report = validate_label_pack(
        pack_dir=Path(args.pack_dir),
        elder_id=str(args.elder_id),
        min_day=int(args.min_day),
        max_day=int(args.max_day),
        registry_path=Path(args.registry_path) if args.registry_path else None,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote label pack validation report: {out_path}")
    if str(report.get("status")) != "pass":
        print(f"Validation failed with {len(report.get('violations', []))} violation(s)")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
