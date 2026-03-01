#!/usr/bin/env python3
"""Fail-closed import smoke for core Beta 6 authority modules."""

from __future__ import annotations

import importlib
import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODULES = (
    "ml.beta6.contracts",
    "ml.beta6.gate_engine",
    "ml.beta6.orchestrator",
    "run_daily_analysis",
)

_KNOWN_NOISE_SUBSTRINGS = (
    "MessageFactory' object has no attribute 'GetPrototype'",
)


def _filter_noise(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(token in line for token in _KNOWN_NOISE_SUBSTRINGS):
            continue
        lines.append(raw)
    return "\n".join(lines).strip()


def main() -> int:
    failures: list[tuple[str, str]] = []
    for module_name in MODULES:
        stderr_capture = io.StringIO()
        stdout_capture = io.StringIO()
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                importlib.import_module(module_name)
        except Exception:  # pragma: no cover - fail-closed smoke script
            failures.append((module_name, traceback.format_exc()))
        else:
            extra_stderr = _filter_noise(stderr_capture.getvalue())
            extra_stdout = _filter_noise(stdout_capture.getvalue())
            if extra_stderr:
                print(f"[warn][{module_name}] stderr:\n{extra_stderr}")
            if extra_stdout:
                print(f"[warn][{module_name}] stdout:\n{extra_stdout}")

    if failures:
        print("beta6_import_smoke: FAIL")
        for module_name, err in failures:
            print(f"\n[module] {module_name}\n{err.rstrip()}\n")
        return 1

    print("beta6_import_smoke: PASS")
    for module_name in MODULES:
        print(f"  - {module_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
