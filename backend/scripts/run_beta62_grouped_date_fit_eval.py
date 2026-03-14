#!/usr/bin/env python3
"""Run the Beta6.2 grouped-date candidate fit/eval entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.beta6.grouped_date_fit_eval import run_grouped_date_fit_eval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Beta6.2 grouped-date candidate fit/eval")
    parser.add_argument("--artifact-dir", required=True, help="Prepared split artifact directory")
    parser.add_argument("--candidate-namespace", required=True, help="Fresh candidate-only namespace")
    parser.add_argument("--output", required=True, help="Output result JSON path")
    parser.add_argument("--supervised-report", default=None, help="Grouped-date supervised report JSON path")
    parser.add_argument("--manifest", default=None, help="Grouped-date supervised manifest JSON path")
    args = parser.parse_args()

    if bool(args.supervised_report) == bool(args.manifest):
        raise SystemExit("Provide exactly one of --supervised-report or --manifest")

    report = run_grouped_date_fit_eval(
        supervised_report_path=args.supervised_report,
        manifest_path=args.manifest,
        artifact_dir=args.artifact_dir,
        candidate_namespace=args.candidate_namespace,
        backend_dir=BACKEND_DIR,
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote Beta6.2 grouped-date fit/eval report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
