#!/usr/bin/env python3
"""Single-seed smoke wrapper for event-first backtest readiness checks."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

DAY_RE = re.compile(r"train_(\d{1,2})(?:[a-z]{3}\d{4})?$", re.IGNORECASE)


def _extract_day(path: Path) -> int | None:
    m = DAY_RE.search(path.stem)
    return int(m.group(1)) if m else None


def _find_day_file(data_dir: Path, elder_id: str, day: int) -> Path | None:
    for ext in ("xlsx", "xls"):
        for p in sorted(data_dir.glob(f"{elder_id}_train_*.{ext}")):
            if "_" in p.stem[len(f"{elder_id}_train_") :]:
                continue
            d = _extract_day(p)
            if d == int(day):
                return p
    return None


def _room_day_occupied_rate(path: Path, room: str) -> float:
    if not path.exists():
        return 0.0
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return 0.0
    sheet = None
    for candidate in xls.sheet_names:
        if str(candidate).strip().lower() == str(room).strip().lower():
            sheet = str(candidate)
            break
    if sheet is None:
        return 0.0
    df = pd.read_excel(path, sheet_name=sheet)
    lower_cols = {str(c).strip().lower(): str(c) for c in df.columns}
    if "activity" not in lower_cols:
        return 0.0
    activities = df[lower_cols["activity"]].astype(str).str.strip().str.lower()
    if len(activities) == 0:
        return 0.0
    occupied = ~activities.isin(["", "nan", "unknown", "unoccupied"])
    return float(occupied.mean())


def _load_expectation(path: Path | None) -> Dict[str, object]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    try:
        obj = yaml.safe_load(path.read_text())
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def run_smoke(
    *,
    data_dir: Path,
    elder_id: str,
    day: int,
    seed: int,
    expectation_config: Path | None,
    diff_report: Path | None,
    output: Path,
    train_context_days: int,
) -> Dict[str, object]:
    expectation = _load_expectation(expectation_config)
    smoke_cfg = expectation.get("smoke", {}) if isinstance(expectation, dict) else {}
    if not isinstance(smoke_cfg, dict):
        smoke_cfg = {}

    min_day = max(int(day) - int(max(train_context_days, 1)), 1)
    max_day = int(day)

    backtest_output = output.parent / f"smoke_backtest_day{int(day)}_seed{int(seed)}.json"
    cmd = [
        sys.executable,
        "backend/scripts/run_event_first_backtest.py",
        "--data-dir",
        str(data_dir),
        "--elder-id",
        str(elder_id),
        "--min-day",
        str(min_day),
        "--max-day",
        str(max_day),
        "--seed",
        str(int(seed)),
        "--output",
        str(backtest_output),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    checks: List[Dict[str, object]] = []
    blocking_reasons: List[str] = []

    checks.append({
        "name": "backtest_command_exit_zero",
        "pass": bool(proc.returncode == 0),
        "details": {"returncode": int(proc.returncode)},
    })
    if proc.returncode != 0:
        blocking_reasons.append("backtest_command_failed")

    report = None
    if backtest_output.exists():
        try:
            report = json.loads(backtest_output.read_text())
        except Exception:
            report = None

    checks.append({
        "name": "backtest_report_generated",
        "pass": bool(report is not None),
        "details": {"path": str(backtest_output)},
    })
    if report is None:
        blocking_reasons.append("backtest_report_missing_or_invalid")

    require_continuity = bool(smoke_cfg.get("require_data_continuity_audit", True))
    continuity_ok = not require_continuity
    if isinstance(report, dict):
        continuity = report.get("data_continuity_audit", {})
        continuity_ok = isinstance(continuity, dict) if require_continuity else True
        if continuity_ok and bool(smoke_cfg.get("fail_on_missing_days_in_requested_window", False)):
            missing = continuity.get("missing_days_in_requested_window", [])
            continuity_ok = isinstance(missing, list) and len(missing) == 0
    checks.append({
        "name": "data_continuity_audit_present",
        "pass": bool(continuity_ok),
        "details": {},
    })
    if not continuity_ok:
        blocking_reasons.append("continuity_audit_failed")

    label_corr_present = False
    if isinstance(report, dict):
        label_corr = report.get("label_corrections")
        if isinstance(label_corr, dict):
            apply_payload = label_corr.get("apply", {})
            if isinstance(apply_payload, dict):
                label_corr_present = any(
                    key in apply_payload for key in ("requested_windows", "applied_windows", "applied_rows")
                )
    checks.append({
        "name": "label_corrections_summary_present",
        "pass": bool(label_corr_present),
        "details": {},
    })
    if bool(smoke_cfg.get("require_label_corrections_summary", True)) and not label_corr_present:
        blocking_reasons.append("label_corrections_summary_missing")

    split_for_day = False
    if isinstance(report, dict):
        splits = report.get("splits", [])
        if isinstance(splits, list):
            split_for_day = any(int(s.get("test_day", -1)) == int(day) for s in splits if isinstance(s, dict))
    checks.append({
        "name": "target_day_split_present",
        "pass": bool(split_for_day),
        "details": {"target_day": int(day)},
    })
    if not split_for_day:
        blocking_reasons.append("target_day_split_missing")

    min_changed_minutes = float(smoke_cfg.get("min_changed_minutes_total", 0.0) or 0.0)
    changed_minutes = 0.0
    diff_ok = True
    if min_changed_minutes > 0.0:
        diff_ok = False
        if diff_report is not None and diff_report.exists():
            try:
                diff_obj = json.loads(diff_report.read_text())
                summary = diff_obj.get("summary", {}) if isinstance(diff_obj, dict) else {}
                changed_minutes = float(summary.get("minutes_changed_total", 0.0) or 0.0)
                diff_ok = changed_minutes >= min_changed_minutes
            except Exception:
                diff_ok = False
        if not diff_ok:
            blocking_reasons.append("insufficient_changed_minutes_evidence")
    checks.append({
        "name": "minimum_changed_minutes_evidence",
        "pass": bool(diff_ok),
        "details": {
            "changed_minutes_total": float(changed_minutes),
            "minimum_required": float(min_changed_minutes),
            "diff_report": str(diff_report) if diff_report is not None else None,
        },
    })

    room_rate_cfg = smoke_cfg.get("min_room_day_occupied_rate", {})
    if isinstance(room_rate_cfg, dict):
        for room_key, payload in sorted(room_rate_cfg.items()):
            if not isinstance(payload, dict):
                continue
            cfg_day = int(payload.get("day", day))
            min_rate = float(payload.get("min_rate", 0.0))
            day_file = _find_day_file(data_dir, elder_id, cfg_day)
            observed = _room_day_occupied_rate(day_file, room_key) if day_file is not None else 0.0
            pass_flag = float(observed) >= float(min_rate)
            checks.append({
                "name": f"occupied_rate:{str(room_key).lower()}:day{cfg_day}",
                "pass": bool(pass_flag),
                "details": {
                    "observed_rate": float(observed),
                    "min_rate": float(min_rate),
                    "file": day_file.name if day_file is not None else None,
                },
            })
            if not pass_flag:
                blocking_reasons.append(f"occupied_rate_below_threshold:{str(room_key).lower()}:day{cfg_day}")

    status = "pass" if len(blocking_reasons) == 0 else "fail"
    result = {
        "status": status,
        "checks": checks,
        "blocking_reasons": sorted(set(blocking_reasons)),
        "backtest_report_path": str(backtest_output),
        "command": cmd,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run event-first smoke checks")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--elder-id", required=True)
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expectation-config", default=None)
    parser.add_argument("--diff-report", default=None)
    parser.add_argument("--train-context-days", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_smoke(
        data_dir=Path(args.data_dir),
        elder_id=str(args.elder_id),
        day=int(args.day),
        seed=int(args.seed),
        expectation_config=Path(args.expectation_config) if args.expectation_config else None,
        diff_report=Path(args.diff_report) if args.diff_report else None,
        output=out_path,
        train_context_days=int(max(args.train_context_days, 1)),
    )
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote smoke report: {out_path}")
    if str(result.get("status")) != "pass":
        print(f"Smoke failed: {result.get('blocking_reasons', [])}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
