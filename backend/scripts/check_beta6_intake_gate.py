#!/usr/bin/env python3
"""Fail-closed precheck for Beta 6 intake-gate artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.beta6.intake_precheck import (  # noqa: E402
    IntakeGateBlockedError,
    enforce_approved_intake_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Beta 6 intake-gate artifact")
    parser.add_argument("--artifact", required=True, help="Path to intake artifact JSON")
    parser.add_argument(
        "--skip-report-file-check",
        action="store_true",
        help="Only validate schema/policy; do not require report files to exist",
    )
    args = parser.parse_args()

    artifact_path = Path(args.artifact).resolve()
    try:
        artifact = enforce_approved_intake_artifact(
            artifact_path,
            require_report_files=not bool(args.skip_report_file_check),
        )
    except IntakeGateBlockedError as exc:
        print(f"Beta6 intake gate blocked (reason_code={exc.reason_code}, detail={exc.detail})")
        return 2
    print(
        "Beta6 intake gate approved "
        f"(artifact={artifact_path}, generated_at={artifact.get('generated_at')})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
