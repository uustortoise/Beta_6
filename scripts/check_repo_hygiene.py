#!/usr/bin/env python3
"""
Fail-closed repository hygiene guard for Beta_6.

Blocks committing runtime artifacts, local data, and secret env files.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BANNED_EXACT = {
    ".env",
    "backend/.env",
    "web-ui/.env",
    "web-ui/.env.local",
    "automation.log",
    "export_ui.log",
}

BANNED_PREFIXES = (
    "logs/",
    "tmp/",
    "archive/",
    "data/",
    "backend/data/",
    "backend/tmp/",
    "backend/logs/",
    "backend/models/",
    "backend/models_beta6_registry_v2/",
    "backend/validation_runs_canary/",
)

BANNED_SUFFIXES = (
    ".db",
    ".db-shm",
    ".db-wal",
    ".pem",
    ".key",
)


def _git(args: list[str]) -> str:
    out = subprocess.check_output(["git", *args], text=True)
    return out.strip()


def _tracked_paths(staged: bool) -> list[str]:
    if staged:
        cmd = ["diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    else:
        cmd = ["ls-files"]
    out = _git(cmd)
    if not out:
        return []
    return [line.strip().replace("\\", "/") for line in out.splitlines() if line.strip()]


def _is_banned(path: str) -> tuple[bool, str]:
    if path in BANNED_EXACT:
        return True, f"exact:{path}"
    for prefix in BANNED_PREFIXES:
        if path.startswith(prefix):
            return True, f"prefix:{prefix}"
    for suffix in BANNED_SUFFIXES:
        if path.endswith(suffix):
            return True, f"suffix:{suffix}"
    return False, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Check staged files only (for pre-commit hook).",
    )
    args = parser.parse_args()

    try:
        repo_root = Path(_git(["rev-parse", "--show-toplevel"]))
    except Exception as exc:
        print(f"[repo-hygiene] unable to resolve git repo root: {exc}", file=sys.stderr)
        return 2

    violations: list[tuple[str, str]] = []
    for path in _tracked_paths(staged=args.staged):
        banned, reason = _is_banned(path)
        if banned:
            violations.append((path, reason))

    if violations:
        scope = "staged files" if args.staged else "tracked files"
        print(f"[repo-hygiene] blocked: banned {scope} detected under {repo_root}")
        for path, reason in violations:
            print(f"  - {path} ({reason})")
        return 1

    scope = "staged files" if args.staged else "tracked files"
    print(f"[repo-hygiene] pass: no banned {scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

