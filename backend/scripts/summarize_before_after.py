#!/usr/bin/env python3
"""Generate before/after summary table from rolling/signoff artifacts."""

from __future__ import annotations

import argparse
import csv
import functools
import json
from pathlib import Path
from typing import Dict, List, Optional


def _load_json(path: Path) -> Dict[str, object]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _extract_metric(rolling: Dict[str, object], room: str, metric: str) -> Optional[float]:
    cls = rolling.get("classification_summary", {})
    if isinstance(cls, dict):
        room_payload = cls.get(room, {})
        if isinstance(room_payload, dict):
            metric_payload = room_payload.get(metric, {})
            if isinstance(metric_payload, dict) and "mean" in metric_payload:
                try:
                    return float(metric_payload.get("mean"))
                except Exception:
                    return None

    timeline = rolling.get("timeline_summary", {})
    if isinstance(timeline, dict):
        room_payload = timeline.get(room, {})
        if isinstance(room_payload, dict):
            metric_payload = room_payload.get(metric, {})
            if isinstance(metric_payload, dict) and "mean" in metric_payload:
                try:
                    return float(metric_payload.get("mean"))
                except Exception:
                    return None

    event_first = rolling.get("event_first", {})
    if isinstance(event_first, dict):
        timeline_summary = event_first.get("timeline_summary", {})
        if isinstance(timeline_summary, dict):
            room_payload = timeline_summary.get(room, {})
            if isinstance(room_payload, dict):
                metric_payload = room_payload.get(metric, {})
                if isinstance(metric_payload, dict) and "mean" in metric_payload:
                    try:
                        return float(metric_payload.get("mean"))
                    except Exception:
                        return None

    splits = rolling.get("splits", [])
    if isinstance(splits, list):
        values: List[float] = []
        for split in splits:
            if not isinstance(split, dict):
                continue
            rooms = split.get("rooms", {})
            if not isinstance(rooms, dict):
                continue
            room_payload = rooms.get(room, {})
            if not isinstance(room_payload, dict):
                continue
            if metric in room_payload:
                try:
                    values.append(float(room_payload.get(metric)))
                except Exception:
                    continue
        if values:
            return float(sum(values) / len(values))

    split_paths: List[str] = []
    if isinstance(splits, list):
        for split in splits:
            if not isinstance(split, dict):
                continue
            path = split.get("path")
            if isinstance(path, str) and path.strip():
                split_paths.append(path.strip())

    if split_paths:
        values: List[float] = []
        for path in sorted(set(split_paths)):
            report = _load_seed_report(path)
            if not isinstance(report, dict):
                continue
            report_splits = report.get("splits", [])
            if not isinstance(report_splits, list):
                continue
            for split in report_splits:
                if not isinstance(split, dict):
                    continue
                rooms = split.get("rooms", {})
                if not isinstance(rooms, dict):
                    continue
                room_payload = rooms.get(room)
                if not isinstance(room_payload, dict):
                    room_payload = None
                    room_norm = str(room).strip().lower()
                    for room_name, payload in rooms.items():
                        if (
                            isinstance(room_name, str)
                            and str(room_name).strip().lower() == room_norm
                            and isinstance(payload, dict)
                        ):
                            room_payload = payload
                            break
                if not isinstance(room_payload, dict):
                    continue
                if metric in room_payload:
                    try:
                        values.append(float(room_payload.get(metric)))
                    except Exception:
                        continue
        if values:
            return float(sum(values) / len(values))
    return None


@functools.lru_cache(maxsize=128)
def _load_seed_report(path: str) -> Optional[Dict[str, object]]:
    try:
        p = Path(path)
        if not p.exists():
            return None
        obj = json.loads(p.read_text())
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_gate_counts(payload: Dict[str, object]) -> Dict[str, int]:
    gate = payload.get("gate_summary", {})
    has_direct_gate_keys = (
        isinstance(gate, dict)
        and any(
            key in gate
            for key in (
                "hard_gate_checks_passed",
                "hard_gate_checks_total",
                "hard_gate_checks_passed_full",
                "hard_gate_checks_total_full",
            )
        )
    )
    if not has_direct_gate_keys:
        gate = (
            payload.get("seed_split_stability", {})
            if isinstance(payload.get("seed_split_stability"), dict)
            else {}
        )
    out = {
        "hard_gate_checks_passed": int(gate.get("hard_gate_checks_passed", gate.get("hard_gate_splits_passed", 0)) or 0),
        "hard_gate_checks_total": int(gate.get("hard_gate_checks_total", gate.get("hard_gate_splits_total", 0)) or 0),
        "hard_gate_checks_passed_full": int(
            gate.get("hard_gate_checks_passed_full", gate.get("hard_gate_splits_passed_full", 0)) or 0
        ),
        "hard_gate_checks_total_full": int(
            gate.get("hard_gate_checks_total_full", gate.get("hard_gate_splits_total_full", 0)) or 0
        ),
    }
    return out


def summarize_before_after(
    *,
    before_rolling: Dict[str, object],
    before_signoff: Dict[str, object],
    after_rolling: Dict[str, object],
    after_signoff: Dict[str, object],
) -> Dict[str, object]:
    before_cls = before_rolling.get("classification_summary", {})
    after_cls = after_rolling.get("classification_summary", {})
    before_rooms = set(before_cls.keys()) if isinstance(before_cls, dict) else set()
    after_rooms = set(after_cls.keys()) if isinstance(after_cls, dict) else set()
    rooms = sorted(set(str(r) for r in before_rooms | after_rooms))

    metrics = ["accuracy", "macro_f1", "occupied_f1", "occupied_recall", "fragmentation_score"]
    rows: List[Dict[str, object]] = []

    for room in rooms:
        row: Dict[str, object] = {"room": room}
        for metric in metrics:
            b = _extract_metric(before_rolling, room, metric)
            a = _extract_metric(after_rolling, room, metric)
            row[f"before_{metric}"] = b
            row[f"after_{metric}"] = a
            row[f"delta_{metric}"] = (a - b) if (a is not None and b is not None) else None
        rows.append(row)

    before_gate = _extract_gate_counts(before_signoff)
    after_gate = _extract_gate_counts(after_signoff)

    return {
        "rooms": rows,
        "global": {
            "before_gate": before_gate,
            "after_gate": after_gate,
            "delta_hard_gate_checks_passed": int(after_gate["hard_gate_checks_passed"] - before_gate["hard_gate_checks_passed"]),
            "delta_hard_gate_checks_passed_full": int(
                after_gate["hard_gate_checks_passed_full"] - before_gate["hard_gate_checks_passed_full"]
            ),
            "before_gate_decision": before_signoff.get("gate_decision"),
            "after_gate_decision": after_signoff.get("gate_decision"),
        },
    }


def _to_markdown(summary: Dict[str, object]) -> str:
    rows = summary.get("rooms", []) if isinstance(summary.get("rooms"), list) else []
    global_payload = summary.get("global", {}) if isinstance(summary.get("global"), dict) else {}

    lines = [
        "# Before/After Model Runtime Summary",
        "",
        "## Room Metrics",
        "",
        "| Room | before_accuracy | after_accuracy | delta_accuracy | before_macro_f1 | after_macro_f1 | delta_macro_f1 | before_occupied_f1 | after_occupied_f1 | delta_occupied_f1 | before_occupied_recall | after_occupied_recall | delta_occupied_recall | before_fragmentation_score | after_fragmentation_score | delta_fragmentation_score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        if not isinstance(row, dict):
            continue
        def fmt(v: object) -> str:
            if v is None:
                return ""
            try:
                return f"{float(v):.4f}"
            except Exception:
                return str(v)

        lines.append(
            "| "
            + str(row.get("room", ""))
            + " | "
            + " | ".join(
                [
                    fmt(row.get("before_accuracy")),
                    fmt(row.get("after_accuracy")),
                    fmt(row.get("delta_accuracy")),
                    fmt(row.get("before_macro_f1")),
                    fmt(row.get("after_macro_f1")),
                    fmt(row.get("delta_macro_f1")),
                    fmt(row.get("before_occupied_f1")),
                    fmt(row.get("after_occupied_f1")),
                    fmt(row.get("delta_occupied_f1")),
                    fmt(row.get("before_occupied_recall")),
                    fmt(row.get("after_occupied_recall")),
                    fmt(row.get("delta_occupied_recall")),
                    fmt(row.get("before_fragmentation_score")),
                    fmt(row.get("after_fragmentation_score")),
                    fmt(row.get("delta_fragmentation_score")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Global Gate Counts",
            "",
            f"- before_gate_decision: `{global_payload.get('before_gate_decision')}`",
            f"- after_gate_decision: `{global_payload.get('after_gate_decision')}`",
            f"- before_hard_gate_checks_passed: `{(global_payload.get('before_gate') or {}).get('hard_gate_checks_passed', 0)}`",
            f"- before_hard_gate_checks_total: `{(global_payload.get('before_gate') or {}).get('hard_gate_checks_total', 0)}`",
            f"- after_hard_gate_checks_passed: `{(global_payload.get('after_gate') or {}).get('hard_gate_checks_passed', 0)}`",
            f"- after_hard_gate_checks_total: `{(global_payload.get('after_gate') or {}).get('hard_gate_checks_total', 0)}`",
            f"- before_hard_gate_checks_passed_full: `{(global_payload.get('before_gate') or {}).get('hard_gate_checks_passed_full', 0)}`",
            f"- before_hard_gate_checks_total_full: `{(global_payload.get('before_gate') or {}).get('hard_gate_checks_total_full', 0)}`",
            f"- after_hard_gate_checks_passed_full: `{(global_payload.get('after_gate') or {}).get('hard_gate_checks_passed_full', 0)}`",
            f"- after_hard_gate_checks_total_full: `{(global_payload.get('after_gate') or {}).get('hard_gate_checks_total_full', 0)}`",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize before/after rolling+signoff artifacts")
    parser.add_argument("--before-rolling", required=True)
    parser.add_argument("--before-signoff", required=True)
    parser.add_argument("--after-rolling", required=True)
    parser.add_argument("--after-signoff", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()

    before_rolling = _load_json(Path(args.before_rolling))
    before_signoff = _load_json(Path(args.before_signoff))
    after_rolling = _load_json(Path(args.after_rolling))
    after_signoff = _load_json(Path(args.after_signoff))

    summary = summarize_before_after(
        before_rolling=before_rolling,
        before_signoff=before_signoff,
        after_rolling=after_rolling,
        after_signoff=after_signoff,
    )

    markdown_path = Path(args.markdown_output)
    csv_path = Path(args.csv_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    markdown_path.write_text(_to_markdown(summary))

    rows = summary.get("rooms", []) if isinstance(summary.get("rooms"), list) else []
    fieldnames = [
        "room",
        "before_accuracy",
        "after_accuracy",
        "delta_accuracy",
        "before_macro_f1",
        "after_macro_f1",
        "delta_macro_f1",
        "before_occupied_f1",
        "after_occupied_f1",
        "delta_occupied_f1",
        "before_occupied_recall",
        "after_occupied_recall",
        "delta_occupied_recall",
        "before_fragmentation_score",
        "after_fragmentation_score",
        "delta_fragmentation_score",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if isinstance(row, dict):
                writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"Wrote before/after markdown: {markdown_path}")
    print(f"Wrote before/after csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
