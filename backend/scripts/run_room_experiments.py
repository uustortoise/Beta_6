from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from ml.room_experiments import build_room_replay_report


def run_room_experiments(
    profile_names: Iterable[str] | None = None,
    *,
    fast_replay_only: bool = False,
) -> dict:
    return build_room_replay_report(profile_names, fast_replay_only=fast_replay_only)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run replay-only room policy diagnostics.")
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Named diagnostic profile to replay. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional JSON output path for the replay report.",
    )
    parser.add_argument(
        "--fast-replay-only",
        action="store_true",
        help="Only emit replay candidates eligible for fast diagnostic sweeps.",
    )
    args = parser.parse_args()

    report = run_room_experiments(args.profile or None, fast_replay_only=bool(args.fast_replay_only))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
