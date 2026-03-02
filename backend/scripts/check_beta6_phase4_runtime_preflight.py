#!/usr/bin/env python3
"""CLI preflight for Beta 6 Phase-4 runtime activation safety."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.beta6.serving.runtime_preflight import validate_beta6_phase4_runtime_preflight_cohort


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Beta 6 Phase-4 runtime activation preflight for elder cohort.",
    )
    parser.add_argument(
        "--elder-id",
        action="append",
        dest="elder_ids",
        default=[],
        help="Elder ID to validate. Repeat for multiple elders.",
    )
    parser.add_argument(
        "--target-cohort",
        default="",
        help=(
            "Comma-separated elder IDs expected to be in runtime target cohort. "
            "If omitted, uses BETA6_RUNTIME_TARGET_COHORT from environment."
        ),
    )
    parser.add_argument(
        "--registry-root",
        default="",
        help="Optional override for BETA6 registry v2 root directory.",
    )
    parser.add_argument(
        "--allow-crf-canary",
        action="store_true",
        help="Allow CRF runtime mode in preflight (canary-only).",
    )
    parser.add_argument(
        "--no-require-target-cohort-for-runtime-flags",
        action="store_true",
        help="Do not fail preflight when runtime flags are enabled without target cohort.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    target_cohort = [token.strip() for token in str(args.target_cohort).split(",") if token.strip()]
    registry_root = Path(args.registry_root).expanduser().resolve() if str(args.registry_root).strip() else None

    ok, report = validate_beta6_phase4_runtime_preflight_cohort(
        elder_ids=list(args.elder_ids),
        target_cohort=target_cohort or None,
        registry_root=registry_root,
        require_target_cohort_for_runtime_flags=not bool(
            args.no_require_target_cohort_for_runtime_flags
        ),
        allow_crf_canary=bool(args.allow_crf_canary),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
