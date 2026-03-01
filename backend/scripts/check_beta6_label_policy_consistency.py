#!/usr/bin/env python3
"""Fail-closed consistency check for Beta 6 label policy mappings."""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Beta 6 label policy consistency.")
    parser.add_argument(
        "--config-dir",
        default=str(BACKEND_DIR / "config"),
        help="Path to backend config directory",
    )
    parser.add_argument(
        "--models-dir",
        default=str(BACKEND_DIR / "models"),
        help="Path to model artifacts for observed-label audit",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Treat warnings as failures.",
    )
    args = parser.parse_args()

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
        from ml.beta6.label_policy_consistency import (  # noqa: E402
            validate_label_policy_consistency,
        )
        report = validate_label_policy_consistency(
            config_dir=Path(args.config_dir),
            models_dir=Path(args.models_dir),
            fail_on_warnings=bool(args.fail_on_warnings),
        )

    known_noise = "MessageFactory' object has no attribute 'GetPrototype'"
    for stream in (captured_stdout, captured_stderr):
        for line in stream.getvalue().splitlines():
            if not line.strip():
                continue
            if known_noise in line:
                continue
            print(line)

    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
