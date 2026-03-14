#!/usr/bin/env python3
"""Run the Beta6.2 grouped-date supervised prep/evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.beta6.grouped_date_supervised import load_grouped_date_supervised_manifest, run_grouped_date_supervised  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Beta6.2 grouped-date supervised prep/evaluation")
    parser.add_argument("--manifest", required=True, help="Input grouped-date supervised manifest JSON path")
    parser.add_argument("--output", required=True, help="Output supervised report JSON path")
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Optional directory for prepared split parquet artifacts",
    )
    args = parser.parse_args()

    manifest = load_grouped_date_supervised_manifest(args.manifest)
    report = run_grouped_date_supervised(manifest, artifact_dir=args.artifact_dir)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote Beta6.2 grouped-date supervised report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
