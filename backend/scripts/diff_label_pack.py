#!/usr/bin/env python3
"""Diff baseline vs candidate training label packs for correction verification."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

DAY_RE = re.compile(r"train_(\d{1,2})(?:[a-z]{3}\d{4})?$", re.IGNORECASE)
ROOMS = ["Bedroom", "LivingRoom", "Kitchen", "Bathroom", "Entrance"]


def _extract_day(path: Path) -> int | None:
    m = DAY_RE.search(path.stem)
    return int(m.group(1)) if m else None


def _collect_files(pack_dir: Path, elder_id: str, *, min_day: int, max_day: int) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    candidate_files: List[Path] = []
    for ext in ("xlsx", "xls"):
        candidate_files.extend(sorted(pack_dir.glob(f"{elder_id}_train_*.{ext}")))

    prefix = f"{elder_id}_train_"
    for p in sorted({str(path.resolve()): path for path in candidate_files}.values(), key=lambda x: x.name.lower()):
        stem = p.stem
        if not stem.startswith(prefix):
            continue
        suffix = stem[len(prefix):]
        if "_" in suffix:
            continue
        day = _extract_day(p)
        if day is None:
            continue
        if not (int(min_day) <= int(day) <= int(max_day)):
            continue
        if day in out:
            continue
        out[day] = p
    return out


def _load_room_labels(path: Path, room: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "activity"])
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "activity"])

    room_sheet = None
    for sheet in xls.sheet_names:
        if str(sheet).strip().lower() == str(room).strip().lower():
            room_sheet = str(sheet)
            break
    if room_sheet is None:
        return pd.DataFrame(columns=["timestamp", "activity"])

    df = pd.read_excel(path, sheet_name=room_sheet)
    lower_cols = {str(c).strip().lower(): str(c) for c in df.columns}
    if "timestamp" not in lower_cols or "activity" not in lower_cols:
        return pd.DataFrame(columns=["timestamp", "activity"])

    ts_col = lower_cols["timestamp"]
    label_col = lower_cols["activity"]
    out = df[[ts_col, label_col]].copy()
    out.columns = ["timestamp", "activity"]
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["activity"] = out["activity"].astype(str).str.strip().str.lower()
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    out = out.reset_index(drop=True)
    return out


def _count_episodes(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    labels = df["activity"].astype(str).str.strip().str.lower().tolist()
    if not labels:
        return 0
    count = 1
    prev = labels[0]
    for label in labels[1:]:
        if label != prev:
            count += 1
            prev = label
    return int(count)


def diff_label_pack(
    *,
    baseline_dir: Path,
    candidate_dir: Path,
    elder_id: str,
    min_day: int,
    max_day: int,
) -> Dict[str, object]:
    base_by_day = _collect_files(baseline_dir, elder_id, min_day=min_day, max_day=max_day)
    cand_by_day = _collect_files(candidate_dir, elder_id, min_day=min_day, max_day=max_day)
    all_days = sorted(set(base_by_day.keys()) | set(cand_by_day.keys()))

    rows: List[Dict[str, object]] = []

    for day in all_days:
        base_file = base_by_day.get(day)
        cand_file = cand_by_day.get(day)
        for room in ROOMS:
            base_df = _load_room_labels(base_file, room) if base_file is not None else pd.DataFrame(columns=["timestamp", "activity"])
            cand_df = _load_room_labels(cand_file, room) if cand_file is not None else pd.DataFrame(columns=["timestamp", "activity"])

            merged = base_df.merge(
                cand_df,
                on="timestamp",
                how="inner",
                suffixes=("_baseline", "_candidate"),
            )
            if not merged.empty:
                changed = merged[merged["activity_baseline"] != merged["activity_candidate"]].copy()
            else:
                changed = pd.DataFrame(columns=["activity_baseline", "activity_candidate"])

            transition_counts: Dict[str, int] = {}
            if not changed.empty:
                for _, r in changed.iterrows():
                    key = f"{str(r['activity_baseline'])}->{str(r['activity_candidate'])}"
                    transition_counts[key] = int(transition_counts.get(key, 0)) + 1

            episodes_baseline = _count_episodes(base_df)
            episodes_candidate = _count_episodes(cand_df)
            episodes_added = max(int(episodes_candidate - episodes_baseline), 0)
            episodes_removed = max(int(episodes_baseline - episodes_candidate), 0)

            row = {
                "day": int(day),
                "room": str(room).strip().lower(),
                "baseline_file": base_file.name if base_file is not None else "",
                "candidate_file": cand_file.name if cand_file is not None else "",
                "baseline_rows": int(len(base_df)),
                "candidate_rows": int(len(cand_df)),
                "windows_matched": int(len(merged)),
                "windows_changed": int(len(changed)),
                "minutes_changed": float(len(changed) * 10.0 / 60.0),
                "episodes_baseline": int(episodes_baseline),
                "episodes_candidate": int(episodes_candidate),
                "episodes_added": int(episodes_added),
                "episodes_removed": int(episodes_removed),
                "transition_counts": transition_counts,
            }
            rows.append(row)

    summary = {
        "days_compared": int(len(all_days)),
        "rows_compared": int(len(rows)),
        "windows_changed_total": int(sum(int(r["windows_changed"]) for r in rows)),
        "minutes_changed_total": float(sum(float(r["minutes_changed"]) for r in rows)),
        "episodes_added_total": int(sum(int(r["episodes_added"]) for r in rows)),
        "episodes_removed_total": int(sum(int(r["episodes_removed"]) for r in rows)),
    }

    return {
        "summary": summary,
        "rows": rows,
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "elder_id": str(elder_id),
        "min_day": int(min_day),
        "max_day": int(max_day),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff baseline vs candidate label packs")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--elder-id", required=True)
    parser.add_argument("--min-day", type=int, required=True)
    parser.add_argument("--max-day", type=int, required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()

    report = diff_label_pack(
        baseline_dir=Path(args.baseline_dir),
        candidate_dir=Path(args.candidate_dir),
        elder_id=str(args.elder_id),
        min_day=int(args.min_day),
        max_day=int(args.max_day),
    )

    json_path = Path(args.json_output)
    csv_path = Path(args.csv_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2))

    fieldnames = [
        "day",
        "room",
        "baseline_file",
        "candidate_file",
        "baseline_rows",
        "candidate_rows",
        "windows_matched",
        "windows_changed",
        "minutes_changed",
        "episodes_baseline",
        "episodes_candidate",
        "episodes_added",
        "episodes_removed",
        "transition_counts",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("rows", []):
            row_out = dict(row)
            row_out["transition_counts"] = json.dumps(row_out.get("transition_counts", {}), sort_keys=True)
            writer.writerow(row_out)

    print(f"Wrote label diff JSON: {json_path}")
    print(f"Wrote label diff CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
