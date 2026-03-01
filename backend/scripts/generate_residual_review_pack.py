#!/usr/bin/env python3
"""Generate residual review pack from seeded backtest reports."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence


DAY_RE = re.compile(r"train_(\d+)dec2025", re.IGNORECASE)


def _extract_day(path: Path) -> int | None:
    m = DAY_RE.search(path.name)
    return int(m.group(1)) if m else None


def _resolve_day_files(*, data_dir: Path | None, elder_id: str | None) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if data_dir is None or elder_id is None:
        return out
    for p in sorted(data_dir.glob(f"{elder_id}_train_*dec2025.xlsx")):
        d = _extract_day(p)
        if d is not None:
            out[d] = p.name
    return out


def build_residual_pack(
    reports: Sequence[dict],
    *,
    day_file_map: Dict[int, str] | None = None,
    top_k: int = 20,
) -> dict:
    day_file_map = day_file_map or {}
    by_room_day: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
    top_windows: List[Dict[str, object]] = []

    for rep in reports:
        seed = int(rep.get("seed", -1))
        for split in rep.get("splits", []):
            if not isinstance(split, dict):
                continue
            test_day = int(split.get("test_day", -1))
            rooms = split.get("rooms", {})
            if not isinstance(rooms, dict):
                continue
            for room, payload in rooms.items():
                if not isinstance(payload, dict):
                    continue
                err = payload.get("error_episodes", {}) or {}
                if not isinstance(err, dict):
                    continue
                room_key = str(room)
                day_entry = by_room_day.setdefault(room_key, {}).setdefault(
                    test_day, {"fn_minutes_total": [], "fp_minutes_total": []}
                )
                day_entry["fn_minutes_total"].append(float(err.get("fn_minutes_total", 0.0)))
                day_entry["fp_minutes_total"].append(float(err.get("fp_minutes_total", 0.0)))

                for kind, key in [("FN", "fn_top"), ("FP", "fp_top")]:
                    for ep in err.get(key, []):
                        if not isinstance(ep, dict):
                            continue
                        top_windows.append(
                            {
                                "room": room_key,
                                "day": int(test_day),
                                "file": day_file_map.get(int(test_day), ""),
                                "seed": int(seed),
                                "kind": kind,
                                "start": str(ep.get("start", "")),
                                "end": str(ep.get("end", "")),
                                "duration_minutes": float(ep.get("duration_minutes", 0.0)),
                                "positive_label": str(err.get("positive_label", "")),
                            }
                        )

    room_day_summary: Dict[str, List[Dict[str, object]]] = {}
    for room, day_map in by_room_day.items():
        rows: List[Dict[str, object]] = []
        for day, vals in sorted(day_map.items()):
            fn_vals = vals.get("fn_minutes_total", [])
            fp_vals = vals.get("fp_minutes_total", [])
            fn_mean = sum(fn_vals) / max(len(fn_vals), 1)
            fp_mean = sum(fp_vals) / max(len(fp_vals), 1)
            rows.append(
                {
                    "day": int(day),
                    "file": day_file_map.get(int(day), ""),
                    "fn_minutes_mean": float(fn_mean),
                    "fp_minutes_mean": float(fp_mean),
                    "net_bias_minutes": float(fp_mean - fn_mean),
                }
            )
        room_day_summary[room] = rows

    top_windows_sorted = sorted(top_windows, key=lambda x: float(x["duration_minutes"]), reverse=True)[: int(max(1, top_k))]
    return {"room_day_summary": room_day_summary, "top_windows": top_windows_sorted}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate residual review pack from seed reports")
    parser.add_argument("--seed-reports", nargs="+", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--elder-id", default=None)
    args = parser.parse_args()

    reports = [json.loads(Path(p).read_text()) for p in args.seed_reports]
    day_file_map = _resolve_day_files(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        elder_id=str(args.elder_id) if args.elder_id else None,
    )
    pack = build_residual_pack(reports, day_file_map=day_file_map, top_k=int(args.top_k))

    json_path = Path(args.json_output)
    csv_path = Path(args.csv_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(pack, indent=2))

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "room",
                "day",
                "file",
                "seed",
                "kind",
                "start",
                "end",
                "duration_minutes",
                "positive_label",
            ],
        )
        writer.writeheader()
        for row in pack.get("top_windows", []):
            writer.writerow(row)

    print(f"Wrote residual pack JSON: {json_path}")
    print(f"Wrote residual pack CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

