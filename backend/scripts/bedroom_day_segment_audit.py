#!/usr/bin/env python3
"""Audit a Bedroom source day by bounded time blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"

import sys

for path in (ROOT, BACKEND):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from utils.data_loader import load_sensor_data


ROOM = "Bedroom"
DEFAULT_EXPECTED_INTERVAL_SECONDS = 10
DEFAULT_BLOCK_MINUTES = 120


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _dominant_day(frame: pd.DataFrame) -> str:
    days = frame["timestamp"].dt.strftime("%Y-%m-%d")
    if days.empty:
        raise ValueError("Cannot infer a dominant day from an empty frame")
    return str(days.mode().iloc[0])


def _select_day_frame(frame: pd.DataFrame, *, day: str | None = None) -> pd.DataFrame:
    day_frame = frame.copy()
    day_frame["timestamp"] = pd.to_datetime(day_frame["timestamp"], errors="coerce")
    day_frame = day_frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    selected_day = day or _dominant_day(day_frame)
    mask = day_frame["timestamp"].dt.strftime("%Y-%m-%d") == str(selected_day)
    return day_frame.loc[mask].reset_index(drop=True)


def _sensor_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "activity", "location"}
    numeric = [col for col in frame.columns if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])]
    return sorted(numeric)


def _transition_counts(labels: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for current, nxt in zip(labels, labels[1:]):
        if current == nxt:
            continue
        key = f"{current} -> {nxt}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _run_length_seconds_by_label(
    labels: Sequence[str],
    *,
    expected_interval_seconds: int,
) -> dict[str, dict[str, float]]:
    if not labels:
        return {}

    durations_by_label: dict[str, list[float]] = {}
    current_label = str(labels[0])
    current_length = 1
    for label in labels[1:]:
        label = str(label)
        if label == current_label:
            current_length += 1
            continue
        durations_by_label.setdefault(current_label, []).append(float(current_length * expected_interval_seconds))
        current_label = label
        current_length = 1
    durations_by_label.setdefault(current_label, []).append(float(current_length * expected_interval_seconds))

    summary: dict[str, dict[str, float]] = {}
    for label, durations in sorted(durations_by_label.items()):
        summary[label] = {
            "count": float(len(durations)),
            "mean": float(sum(durations) / len(durations)),
            "max": float(max(durations)),
        }
    return summary


def _build_reference_lookup(reference_block_sets: Sequence[Sequence[Mapping[str, Any]]]) -> dict[int, list[Mapping[str, Any]]]:
    lookup: dict[int, list[Mapping[str, Any]]] = {}
    for block_set in reference_block_sets:
        for block in block_set:
            lookup.setdefault(int(block["block_index"]), []).append(block)
    return lookup


def _compare_to_reference(
    block: Mapping[str, Any],
    reference_blocks: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not reference_blocks:
        return None

    labels = sorted(
        {
            str(label)
            for ref in reference_blocks
            for label in (ref.get("label_share") or {}).keys()
        }
        | {str(label) for label in (block.get("label_share") or {}).keys()}
    )
    reference_label_share: dict[str, float] = {}
    label_share_delta: dict[str, float] = {}
    for label in labels:
        values = [float((ref.get("label_share") or {}).get(label, 0.0)) for ref in reference_blocks]
        average = float(sum(values) / len(values))
        reference_label_share[label] = average
        label_share_delta[label] = round(float((block.get("label_share") or {}).get(label, 0.0)) - average, 6)

    reference_observed_ratio = float(sum(float(ref.get("observed_ratio", 0.0)) for ref in reference_blocks) / len(reference_blocks))
    reference_largest_gap = float(
        sum(float(ref.get("largest_gap_seconds", 0.0)) for ref in reference_blocks) / len(reference_blocks)
    )
    return {
        "reference_count": len(reference_blocks),
        "reference_label_share": reference_label_share,
        "reference_observed_ratio": reference_observed_ratio,
        "reference_largest_gap_seconds": reference_largest_gap,
        "label_share_delta": label_share_delta,
        "observed_ratio_delta": round(float(block.get("observed_ratio", 0.0)) - reference_observed_ratio, 6),
        "largest_gap_delta_seconds": round(float(block.get("largest_gap_seconds", 0.0)) - reference_largest_gap, 6),
    }


def _summarize_block(
    block_frame: pd.DataFrame,
    *,
    block_index: int,
    block_start: pd.Timestamp,
    block_minutes: int,
    expected_interval_seconds: int,
) -> dict[str, Any]:
    expected_rows = max(int((block_minutes * 60) / expected_interval_seconds), 1)
    labels = block_frame["activity"].fillna("missing").astype(str).tolist()
    label_counts = {
        str(label): int(count)
        for label, count in block_frame["activity"].fillna("missing").astype(str).value_counts().sort_index().items()
    }
    label_share = {
        label: round(count / len(block_frame), 6)
        for label, count in label_counts.items()
    }
    diffs = block_frame["timestamp"].sort_values().diff().dt.total_seconds().dropna()
    largest_gap_seconds = float(diffs.max()) if not diffs.empty else 0.0
    sensor_missing_share = {
        column: round(float(block_frame[column].isna().mean()), 6)
        for column in _sensor_columns(block_frame)
    }

    flags: list[str] = []
    observed_ratio = round(len(block_frame) / expected_rows, 6)
    if observed_ratio < 0.8:
        flags.append("sparse_rows")
    if largest_gap_seconds > float(expected_interval_seconds * 2):
        flags.append("large_gap")
    if sensor_missing_share and max(sensor_missing_share.values()) > 0.2:
        flags.append("sensor_missingness")

    block_end = block_start + pd.Timedelta(minutes=block_minutes)
    return {
        "block_index": int(block_index),
        "block_start": block_start,
        "block_end": block_end,
        "block_label": f"{block_start.strftime('%H:%M')}-{block_end.strftime('%H:%M')}",
        "row_count": int(len(block_frame)),
        "expected_rows": int(expected_rows),
        "observed_ratio": observed_ratio,
        "largest_gap_seconds": largest_gap_seconds,
        "label_counts": label_counts,
        "label_share": label_share,
        "transition_counts": _transition_counts(labels),
        "run_length_seconds_by_label": _run_length_seconds_by_label(
            labels,
            expected_interval_seconds=expected_interval_seconds,
        ),
        "sensor_missing_share": sensor_missing_share,
        "flags": flags,
    }


def audit_day_frame(
    frame: pd.DataFrame,
    *,
    day: str | None = None,
    reference_frames: Sequence[pd.DataFrame] | None = None,
    block_minutes: int = DEFAULT_BLOCK_MINUTES,
    expected_interval_seconds: int = DEFAULT_EXPECTED_INTERVAL_SECONDS,
) -> dict[str, Any]:
    day_frame = _select_day_frame(frame, day=day)
    if day_frame.empty:
        raise ValueError("No rows remain after selecting the target day")

    working = day_frame.copy()
    working["block_start"] = working["timestamp"].dt.floor(f"{int(block_minutes)}min")

    blocks: list[dict[str, Any]] = []
    for block_index, (block_start, block_frame) in enumerate(working.groupby("block_start"), start=1):
        blocks.append(
            _summarize_block(
                block_frame.reset_index(drop=True),
                block_index=block_index,
                block_start=pd.Timestamp(block_start),
                block_minutes=block_minutes,
                expected_interval_seconds=expected_interval_seconds,
            )
        )

    reference_block_lookup: dict[int, list[Mapping[str, Any]]] = {}
    if reference_frames:
        reference_block_sets = [
            audit_day_frame(
                reference_frame,
                day=day,
                block_minutes=block_minutes,
                expected_interval_seconds=expected_interval_seconds,
            )["blocks"]
            for reference_frame in reference_frames
        ]
        reference_block_lookup = _build_reference_lookup(reference_block_sets)

    standout_blocks: list[dict[str, Any]] = []
    for block in blocks:
        reference_delta = _compare_to_reference(block, reference_block_lookup.get(int(block["block_index"]), []))
        if reference_delta is not None:
            block["reference_delta"] = reference_delta
            standout_score = max(abs(value) for value in reference_delta["label_share_delta"].values()) if reference_delta["label_share_delta"] else 0.0
            standout_blocks.append(
                {
                    "block_index": int(block["block_index"]),
                    "block_label": str(block["block_label"]),
                    "score": round(float(standout_score), 6),
                    "flags": list(block["flags"]),
                }
            )

    standout_blocks.sort(key=lambda item: (-float(item["score"]), int(item["block_index"])))
    return {
        "day": _dominant_day(day_frame),
        "block_minutes": int(block_minutes),
        "expected_interval_seconds": int(expected_interval_seconds),
        "row_count": int(len(day_frame)),
        "blocks": blocks,
        "standout_blocks": standout_blocks,
    }


def audit_file(
    *,
    file_path: str,
    room: str = ROOM,
    day: str | None = None,
    reference_files: Sequence[str] | None = None,
    block_minutes: int = DEFAULT_BLOCK_MINUTES,
    expected_interval_seconds: int = DEFAULT_EXPECTED_INTERVAL_SECONDS,
) -> dict[str, Any]:
    target_rooms = load_sensor_data(file_path, resample=True)
    if room not in target_rooms:
        raise KeyError(f"Room {room} not found in {file_path}")

    reference_frames = []
    for reference_file in reference_files or []:
        rooms = load_sensor_data(reference_file, resample=True)
        if room not in rooms:
            raise KeyError(f"Room {room} not found in {reference_file}")
        reference_frames.append(rooms[room])

    summary = audit_day_frame(
        target_rooms[room],
        day=day,
        reference_frames=reference_frames,
        block_minutes=block_minutes,
        expected_interval_seconds=expected_interval_seconds,
    )
    summary.update(
        {
            "room": room,
            "input_file": str(file_path),
            "reference_files": [str(path) for path in reference_files or []],
        }
    )
    return summary


def _blocks_dataframe(blocks: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for block in blocks:
        rows.append(
            {
                "block_index": int(block["block_index"]),
                "block_start": pd.Timestamp(block["block_start"]),
                "block_end": pd.Timestamp(block["block_end"]),
                "block_label": str(block["block_label"]),
                "row_count": int(block["row_count"]),
                "expected_rows": int(block["expected_rows"]),
                "observed_ratio": float(block["observed_ratio"]),
                "largest_gap_seconds": float(block["largest_gap_seconds"]),
                "flags_json": json.dumps(block.get("flags") or [], sort_keys=True),
                "label_counts_json": json.dumps(block.get("label_counts") or {}, sort_keys=True),
                "label_share_json": json.dumps(block.get("label_share") or {}, sort_keys=True),
                "transition_counts_json": json.dumps(block.get("transition_counts") or {}, sort_keys=True),
                "run_length_seconds_json": json.dumps(block.get("run_length_seconds_by_label") or {}, sort_keys=True),
                "sensor_missing_share_json": json.dumps(block.get("sensor_missing_share") or {}, sort_keys=True),
                "reference_delta_json": json.dumps(block.get("reference_delta") or {}, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a Bedroom day by bounded time blocks")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--reference-file", action="append", default=[])
    parser.add_argument("--room", default=ROOM)
    parser.add_argument("--day", default=None)
    parser.add_argument("--block-minutes", type=int, default=DEFAULT_BLOCK_MINUTES)
    parser.add_argument("--expected-interval-seconds", type=int, default=DEFAULT_EXPECTED_INTERVAL_SECONDS)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = audit_file(
        file_path=str(args.input_file),
        room=str(args.room),
        day=args.day,
        reference_files=args.reference_file,
        block_minutes=int(args.block_minutes),
        expected_interval_seconds=int(args.expected_interval_seconds),
    )

    summary_path = output_dir / "summary.json"
    blocks_path = output_dir / "blocks.parquet"
    summary_path.write_text(json.dumps(_to_jsonable(summary), indent=2), encoding="utf-8")
    _blocks_dataframe(summary["blocks"]).to_parquet(blocks_path, index=False)
    print(f"Wrote audit summary: {summary_path}")
    print(f"Wrote audit blocks: {blocks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
