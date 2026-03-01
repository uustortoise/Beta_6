#!/usr/bin/env python3
"""Run one event-first backtest variant in-process using matrix-profile args."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import yaml

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts import run_event_first_backtest as backtest_script


def _load_yaml(path: Path) -> Dict[str, object]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root at {path}")
    return data


def _build_argv(
    *,
    profiles_yaml: Path,
    variant: str,
    data_dir: Path,
    elder_id: str,
    seed: int,
    output: Path,
    min_day_override: int | None,
    max_day_override: int | None,
) -> List[str]:
    spec = _load_yaml(profiles_yaml)
    defaults = spec.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}
    variants = spec.get("variants", {})
    if not isinstance(variants, dict) or variant not in variants:
        raise ValueError(f"Variant not found: {variant}")
    variant_payload = variants.get(variant, {})
    if not isinstance(variant_payload, dict):
        raise ValueError(f"Invalid variant payload: {variant}")

    min_day = int(min_day_override if min_day_override is not None else defaults.get("min_day", 4))
    max_day = int(max_day_override if max_day_override is not None else defaults.get("max_day", 10))
    variant_args = variant_payload.get("args", [])
    if not isinstance(variant_args, list):
        variant_args = []

    argv = [
        "run_event_first_backtest.py",
        "--data-dir",
        str(data_dir),
        "--elder-id",
        str(elder_id),
        "--min-day",
        str(min_day),
        "--max-day",
        str(max_day),
        "--seed",
        str(int(seed)),
        "--occupancy-threshold",
        str(float(defaults.get("occupancy_threshold", 0.35))),
        "--calibration-method",
        str(defaults.get("calibration_method", "isotonic")),
        "--calib-fraction",
        str(float(defaults.get("calib_fraction", 0.20))),
        "--min-calib-samples",
        str(int(defaults.get("min_calib_samples", 500))),
        "--min-calib-label-support",
        str(int(defaults.get("min_calib_label_support", 30))),
        "--output",
        str(output),
    ]
    argv.extend([str(v) for v in variant_args])
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one matrix variant in-process")
    parser.add_argument("--profiles-yaml", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--elder-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-day", type=int, default=None)
    parser.add_argument("--max-day", type=int, default=None)
    args = parser.parse_args()

    argv = _build_argv(
        profiles_yaml=Path(args.profiles_yaml),
        variant=str(args.variant),
        data_dir=Path(args.data_dir),
        elder_id=str(args.elder_id),
        seed=int(args.seed),
        output=Path(args.output),
        min_day_override=args.min_day,
        max_day_override=args.max_day,
    )

    old_argv = sys.argv
    try:
        sys.argv = argv
        return int(backtest_script.main())
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
