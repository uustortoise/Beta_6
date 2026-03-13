#!/usr/bin/env python3
"""Run Phase 3.2 active-learning triage queue generation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.beta6.active_learning import (  # noqa: E402
    build_active_learning_queue,
    load_active_learning_policy,
)
from services.correction_service import build_correction_learning_records  # noqa: E402


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO date: {value!r}") from exc


def _load_candidate_sources(args: argparse.Namespace, parser: argparse.ArgumentParser) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if args.input_csv:
        input_path = Path(args.input_csv).resolve()
        frames.append(pd.read_csv(input_path))

    correction_args = (
        args.correction_elder_id,
        args.correction_room,
        args.correction_date,
    )
    if any(value is not None for value in correction_args):
        if not all(value is not None for value in correction_args):
            parser.error(
                "--correction-elder-id, --correction-room, and --correction-date must be provided together"
            )
        correction_records = build_correction_learning_records(
            args.correction_elder_id,
            args.correction_room,
            args.correction_date,
            confidence_threshold=args.correction_confidence_threshold,
            context_minutes=args.correction_context_minutes,
        )
        frames.append(pd.DataFrame(correction_records))

    if not frames:
        parser.error("Provide --input-csv, correction-derived source arguments, or both")
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Beta 6 active-learning triage queue")
    parser.add_argument("--input-csv", help="Candidate windows CSV path")
    parser.add_argument(
        "--policy",
        default=str(BACKEND_DIR / "config" / "beta6_active_learning_policy.yaml"),
        help="Active-learning policy YAML",
    )
    parser.add_argument("--correction-elder-id", help="Resident ID for accepted-correction learning records")
    parser.add_argument("--correction-room", help="Room name for accepted-correction learning records")
    parser.add_argument(
        "--correction-date",
        type=_parse_iso_date,
        help="Record date (YYYY-MM-DD) for accepted-correction learning records",
    )
    parser.add_argument(
        "--correction-confidence-threshold",
        type=float,
        default=0.60,
        help="Residual-review confidence threshold for correction-derived records",
    )
    parser.add_argument(
        "--correction-context-minutes",
        type=int,
        default=15,
        help="Context window on each side of a correction when building residual-review packs",
    )
    parser.add_argument("--output-csv", required=True, help="Output triage queue CSV")
    parser.add_argument("--report-json", required=True, help="Output queue summary JSON")
    args = parser.parse_args()

    frame = _load_candidate_sources(args, parser)
    policy = load_active_learning_policy(args.policy)
    result = build_active_learning_queue(frame, policy=policy)

    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    queue_frame = pd.DataFrame(result.get("queue", []))
    queue_frame.to_csv(output_csv, index=False)

    report_json = Path(args.report_json).resolve()
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    stats = result.get("stats", {})
    signal_counts = stats.get("training_signal_counts", {}) or {}
    print(f"Wrote triage queue: {output_csv}")
    print(f"Wrote triage report: {report_json}")
    print(
        f"Queue rows={stats.get('queue_rows', 0)} "
        f"input_rows={stats.get('input_rows', 0)}"
    )
    print(
        "Training signals: "
        f"corrected={signal_counts.get('corrected_event_rows', 0)} "
        f"hard_negative={signal_counts.get('hard_negative_rows', 0)} "
        f"residual_review={signal_counts.get('residual_review_rows', 0)}"
    )
    return 0 if str(result.get("status")) == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
