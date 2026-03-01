#!/usr/bin/env python3
"""Fail-closed schema check for all beta6_*.yaml config files."""

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
    parser = argparse.ArgumentParser(description="Validate Beta 6 YAML config schemas.")
    parser.add_argument(
        "--config-dir",
        default=str(BACKEND_DIR / "config"),
        help="Path to backend config directory",
    )
    args = parser.parse_args()

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
        from ml.beta6.beta6_schema import validate_all_beta6_configs  # noqa: E402

    # Filter known protobuf MessageFactory noise while preserving other diagnostics.
    known_noise = "MessageFactory' object has no attribute 'GetPrototype'"
    for stream in (captured_stdout, captured_stderr):
        for line in stream.getvalue().splitlines():
            if not line.strip():
                continue
            if known_noise in line:
                continue
            print(line)

    report = validate_all_beta6_configs(args.config_dir)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
