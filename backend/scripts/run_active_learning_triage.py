#!/usr/bin/env python3
"""Run Phase 3.2 active-learning triage queue generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.beta6.active_learning import (  # noqa: E402
    build_active_learning_queue,
    load_active_learning_policy,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Beta 6 active-learning triage queue")
    parser.add_argument("--input-csv", required=True, help="Candidate windows CSV path")
    parser.add_argument(
        "--policy",
        default=str(BACKEND_DIR / "config" / "beta6_active_learning_policy.yaml"),
        help="Active-learning policy YAML",
    )
    parser.add_argument("--output-csv", required=True, help="Output triage queue CSV")
    parser.add_argument("--report-json", required=True, help="Output queue summary JSON")
    args = parser.parse_args()

    input_path = Path(args.input_csv).resolve()
    frame = pd.read_csv(input_path)
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
    print(f"Wrote triage queue: {output_csv}")
    print(f"Wrote triage report: {report_json}")
    print(
        f"Queue rows={stats.get('queue_rows', 0)} "
        f"input_rows={stats.get('input_rows', 0)}"
    )
    print(
        f"Training-ready records={len(result.get('training_ready_records', []))}"
    )
    return 0 if str(result.get("status")) == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
