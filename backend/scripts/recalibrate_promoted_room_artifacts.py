from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.pipeline import UnifiedPipeline
from ml.utils import calculate_sequence_length
from utils.data_loader import load_sensor_data


def _combine_room_frames(file_paths: list[Path], room_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in file_paths:
        loaded = load_sensor_data(path, resample=True)
        room_df = loaded.get(room_name)
        if room_df is None or room_df.empty:
            continue
        frames.append(room_df.copy())

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
        combined = combined.dropna(subset=["timestamp"])
        combined = combined.sort_values("timestamp")
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    return combined.reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a new version of existing champion weights with recalibrated thresholds and activity confidence artifacts."
    )
    parser.add_argument("--elder-id", required=True)
    parser.add_argument("--source-file", action="append", required=True)
    parser.add_argument("--room", action="append")
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--summary-out")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    file_paths = [Path(raw).expanduser().resolve() for raw in args.source_file]

    pipeline = UnifiedPipeline()
    loaded_rooms = list(pipeline.registry.load_models_for_elder(args.elder_id, pipeline.platform))
    target_rooms = list(args.room or loaded_rooms)
    if not target_rooms:
        raise SystemExit(f"No promoted room artifacts found for {args.elder_id}")

    summary: dict[str, Any] = {
        "elder_id": str(args.elder_id),
        "source_files": [str(path) for path in file_paths],
        "rooms": [],
    }

    for room_name in target_rooms:
        room_df = _combine_room_frames(file_paths, room_name)
        if room_df.empty:
            raise SystemExit(f"No labeled data available for room {room_name}")

        seq_length = calculate_sequence_length(pipeline.platform, room_name)
        metrics = pipeline.trainer.recalibrate_promoted_room_artifacts(
            elder_id=args.elder_id,
            room_name=room_name,
            raw_df=room_df,
            seq_length=seq_length,
            validation_split=float(args.validation_split),
        )
        summary["rooms"].append(metrics)

    payload = json.dumps(summary, indent=2, default=str)
    if args.summary_out:
        out_path = Path(args.summary_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
