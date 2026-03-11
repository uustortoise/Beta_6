import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from utils.data_loader import load_sensor_data


ELDER_ID = "HK0011_jessica"
ROOM = "LivingRoom"
SOURCE_DIR = Path("/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)")
SWEEP_ROOT = Path("tmp/jessica_livingroom_v52_crossdate_20260311T071050Z")
FEATURE_COLS = ["co2", "humidity", "temperature", "sound", "motion", "light"]


@dataclass(frozen=True)
class ProposalSegment:
    external_id: str
    date_key: str
    start: str
    end: str
    proposed_label: str
    confidence_tier: str
    proposal_score: float
    reason_codes: tuple[str, ...]
    summary_note: str


SELECTED_SEGMENTS = (
    ProposalSegment(
        external_id="livingroom-5dec-1613-occupied-plateau",
        date_key="5dec",
        start="2025-12-05 16:13:00",
        end="2025-12-05 16:16:20",
        proposed_label="livingroom_normal_use",
        confidence_tier="medium",
        proposal_score=0.78,
        reason_codes=(
            "consensus_all_models",
            "sandwiched_truth_run",
            "quiet_plateau_inside_occupied_block",
        ),
        summary_note=(
            "Ground truth inserts a 3m20s unoccupied island inside occupied labels. "
            "All three models keep the full span occupied, and sound/light stay continuous "
            "with the surrounding occupied block while motion collapses after the earlier edge spikes."
        ),
    ),
    ProposalSegment(
        external_id="livingroom-7dec-211850-boundary-shift-a",
        date_key="7dec",
        start="2025-12-07 21:18:50",
        end="2025-12-07 21:19:20",
        proposed_label="livingroom_normal_use",
        confidence_tier="high",
        proposal_score=0.92,
        reason_codes=(
            "consensus_all_models",
            "sandwiched_truth_run",
            "boundary_oscillation",
            "sensor_continuity_with_occupied",
        ),
        summary_note=(
            "The label flips to unoccupied for 40 seconds inside an occupied block, then flips back. "
            "All three models keep this span occupied, and sound/light remain continuous with the occupied context."
        ),
    ),
    ProposalSegment(
        external_id="livingroom-7dec-211940-boundary-shift-b",
        date_key="7dec",
        start="2025-12-07 21:19:40",
        end="2025-12-07 21:22:00",
        proposed_label="unoccupied",
        confidence_tier="high",
        proposal_score=0.89,
        reason_codes=(
            "consensus_all_models",
            "sandwiched_truth_run",
            "boundary_oscillation",
            "sensor_continuity_with_unoccupied",
        ),
        summary_note=(
            "After a short spike at the transition, the truth stays occupied for 2m20s while all three models "
            "move to unoccupied and the sensor profile aligns with the following long unoccupied run."
        ),
    ),
    ProposalSegment(
        external_id="livingroom-10dec-182210-unoccupied-island",
        date_key="10dec",
        start="2025-12-10 18:22:10",
        end="2025-12-10 18:25:50",
        proposed_label="unoccupied",
        confidence_tier="medium",
        proposal_score=0.74,
        reason_codes=(
            "consensus_all_models",
            "sandwiched_truth_run",
            "occupied_island_without_sustained_activity",
        ),
        summary_note=(
            "Ground truth marks a 3m40s occupied island inside long unoccupied runs. "
            "All three models keep the full span unoccupied; aside from the opening spike, "
            "motion stays near background while sound/light remain flat."
        ),
    ),
)

EXCLUDED_EXEMPLARS = (
    {
        "date_key": "4dec",
        "window": "2025-12-04 00:35:00 -> 00:42:30",
        "decision": "not proposed",
        "reason": "v46/v50 support the current occupied label; this remains a v52 undercall, not a label fix.",
    },
    {
        "date_key": "4dec",
        "window": "2025-12-04 05:22:40 -> 05:29:00",
        "decision": "not proposed",
        "reason": "v46/v50 support the current unoccupied label; this remains a v52 occupied overshoot.",
    },
    {
        "date_key": "4dec",
        "window": "2025-12-04 07:20:30 -> 07:25:00",
        "decision": "not proposed",
        "reason": "v46/v50 support the current unoccupied label; this remains a v52 occupied overshoot.",
    },
    {
        "date_key": "17dec",
        "window": "2025-12-17 14:36:50 -> 14:45:10",
        "decision": "not proposed",
        "reason": "This is a real occupied block that v52 recovers and v46 misses; correcting labels would move in the wrong direction.",
    },
    {
        "date_key": "17dec",
        "window": "2025-12-17 15:47:50 -> 15:52:40",
        "decision": "not proposed",
        "reason": "This is another real occupied block that v52 recovers and v46 misses; not a labeling issue.",
    },
    {
        "date_key": "8dec",
        "window": "2025-12-08 14:18:40 -> 14:23:20",
        "decision": "not proposed",
        "reason": "All models undercall despite repeated high-motion spikes; this looks like a shared model miss rather than a clean relabel.",
    },
)


def _source_file_for(date_key: str) -> Path:
    day = date_key.replace("dec", "")
    return SOURCE_DIR / f"{ELDER_ID}_train_{day}dec2025.xlsx"


def _models_for(date_key: str) -> tuple[str, ...]:
    return ("v46", "v52") if date_key == "17dec" else ("v46", "v50", "v52")


def _load_frame(date_key: str) -> pd.DataFrame:
    raw = load_sensor_data(_source_file_for(date_key), resample=True)[ROOM][["timestamp", "activity", *FEATURE_COLS]].copy()
    merged = raw
    for version in _models_for(date_key):
        pred = pd.read_parquet(
            SWEEP_ROOT / version / date_key / "comparison" / f"{ROOM}_merged.parquet"
        )[
            [
                "timestamp",
                "predicted_activity",
                "activity_acceptance_score",
                "predicted_top1_prob_raw",
            ]
        ].rename(
            columns={
                "predicted_activity": f"pred_{version}",
                "activity_acceptance_score": f"acceptance_{version}",
                "predicted_top1_prob_raw": f"rawprob_{version}",
            }
        )
        merged = merged.merge(pred, on="timestamp", how="inner")
    return merged.sort_values("timestamp").reset_index(drop=True)


def _segment_rationale(segment: ProposalSegment, date_df: pd.DataFrame, seg_df: pd.DataFrame) -> dict:
    run_ids = date_df["activity"].ne(date_df["activity"].shift()).cumsum()
    date_df = date_df.assign(truth_run_id=run_ids)
    seg_run_ids = date_df.loc[date_df["timestamp"].isin(seg_df["timestamp"]), "truth_run_id"].unique().tolist()
    prev_rows = next_rows = None
    prev_label = next_label = None
    if seg_run_ids:
        run_id = int(seg_run_ids[0])
        prev = date_df[date_df["truth_run_id"] == run_id - 1]
        nxt = date_df[date_df["truth_run_id"] == run_id + 1]
        if not prev.empty:
            prev_rows = int(len(prev))
            prev_label = str(prev["activity"].iloc[0])
        if not nxt.empty:
            next_rows = int(len(nxt))
            next_label = str(nxt["activity"].iloc[0])

    model_votes = {}
    acceptances = {}
    for version in _models_for(segment.date_key):
        pred_col = f"pred_{version}"
        acc_col = f"acceptance_{version}"
        model_votes[version] = str(seg_df[pred_col].mode().iloc[0])
        acceptances[version] = float(seg_df[acc_col].mean())

    seg_means = {col: float(seg_df[col].mean()) for col in FEATURE_COLS}
    return {
        "date_key": segment.date_key,
        "segment_rows": int(len(seg_df)),
        "truth_label": str(seg_df["activity"].mode().iloc[0]),
        "proposed_label": segment.proposed_label,
        "prev_truth_label": prev_label,
        "next_truth_label": next_label,
        "prev_truth_rows": prev_rows,
        "next_truth_rows": next_rows,
        "model_vote_mode": model_votes,
        "mean_acceptance": acceptances,
        "feature_means": seg_means,
        "summary_note": segment.summary_note,
    }


def main() -> None:
    timestamp_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("tmp") / f"livingroom_label_timestamp_forensic_{timestamp_token}"
    output_dir.mkdir(parents=True, exist_ok=True)

    date_frames: dict[str, pd.DataFrame] = {}
    segment_rows = []
    timestamp_rows = []

    for segment in SELECTED_SEGMENTS:
        date_df = date_frames.setdefault(segment.date_key, _load_frame(segment.date_key))
        seg_df = date_df[(date_df["timestamp"] >= segment.start) & (date_df["timestamp"] <= segment.end)].copy()
        if seg_df.empty:
            raise ValueError(f"No rows found for {segment.external_id}")

        current_label = str(seg_df["activity"].mode().iloc[0])
        source_models = "|".join(f"{v}:{str(seg_df[f'pred_{v}'].mode().iloc[0])}" for v in _models_for(segment.date_key))
        rationale = _segment_rationale(segment, date_df, seg_df)

        segment_rows.append(
            {
                "external_id": segment.external_id,
                "parent_external_id": None,
                "elder_id": ELDER_ID,
                "room": ROOM,
                "record_date": pd.to_datetime(segment.start).strftime("%Y-%m-%d"),
                "granularity": "segment",
                "timestamp_start": pd.to_datetime(segment.start),
                "timestamp_end": pd.to_datetime(segment.end),
                "current_label": current_label,
                "proposed_label": segment.proposed_label,
                "confidence_tier": segment.confidence_tier,
                "proposal_score": segment.proposal_score,
                "reason_codes": list(segment.reason_codes),
                "rationale": rationale,
                "source_model": source_models,
                "review_status": "proposed",
            }
        )

        for row_idx, (_, row) in enumerate(seg_df.iterrows(), start=1):
            timestamp_rows.append(
                {
                    "external_id": f"{segment.external_id}-ts-{row_idx:03d}",
                    "parent_external_id": segment.external_id,
                    "elder_id": ELDER_ID,
                    "room": ROOM,
                    "record_date": pd.to_datetime(segment.start).strftime("%Y-%m-%d"),
                    "granularity": "timestamp",
                    "timestamp_start": row["timestamp"],
                    "timestamp_end": row["timestamp"],
                    "current_label": str(row["activity"]),
                    "proposed_label": segment.proposed_label,
                    "confidence_tier": segment.confidence_tier,
                    "proposal_score": segment.proposal_score,
                    "reason_codes": list(segment.reason_codes),
                    "rationale": {
                        "parent_external_id": segment.external_id,
                        "timestamp": str(row["timestamp"]),
                        "source_predictions": {
                            version: str(row[f"pred_{version}"]) for version in _models_for(segment.date_key)
                        },
                        "feature_values": {col: float(row[col]) for col in FEATURE_COLS},
                    },
                    "source_model": source_models,
                    "review_status": "proposed",
                }
            )

    proposal_df = pd.DataFrame(segment_rows + timestamp_rows)
    ops_review_df = pd.DataFrame(
        [
            {
                "external_id": row["external_id"],
                "record_date": row["record_date"],
                "timestamp_start": row["timestamp_start"],
                "timestamp_end": row["timestamp_end"],
                "current_label": row["current_label"],
                "proposed_label": row["proposed_label"],
                "confidence_tier": row["confidence_tier"],
                "proposal_score": row["proposal_score"],
                "reason_codes": ", ".join(row["reason_codes"]),
                "source_model": row["source_model"],
                "summary_note": row["rationale"]["summary_note"],
                "prev_truth_label": row["rationale"]["prev_truth_label"],
                "next_truth_label": row["rationale"]["next_truth_label"],
                "prev_truth_rows": row["rationale"]["prev_truth_rows"],
                "next_truth_rows": row["rationale"]["next_truth_rows"],
                "feature_means": json.dumps(row["rationale"]["feature_means"], sort_keys=True),
            }
            for row in segment_rows
        ]
    )
    timestamp_set_df = pd.DataFrame(
        [
            {
                "timestamp": row["timestamp_start"],
                "record_date": row["record_date"],
                "parent_external_id": row["parent_external_id"],
                "current_label": row["current_label"],
                "proposed_label": row["proposed_label"],
                "confidence_tier": row["confidence_tier"],
                "proposal_score": row["proposal_score"],
                "reason_codes": ", ".join(row["reason_codes"]),
                "source_model": row["source_model"],
            }
            for row in timestamp_rows
        ]
    ).sort_values("timestamp")

    summary = {
        "elder_id": ELDER_ID,
        "room": ROOM,
        "proposal_segments": len(segment_rows),
        "proposal_timestamps": len(timestamp_rows),
        "selected_segments": [row["external_id"] for row in segment_rows],
        "excluded_exemplars": list(EXCLUDED_EXEMPLARS),
        "artifacts": {
            "ops_review_csv": str((output_dir / "ops_review.csv").resolve()),
            "proposed_timestamp_set_csv": str((output_dir / "proposed_timestamp_set.csv").resolve()),
            "proposal_pack_json": str((output_dir / "proposal_pack.json").resolve()),
        },
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    ops_review_df.to_csv(output_dir / "ops_review.csv", index=False)
    timestamp_set_df.to_csv(output_dir / "proposed_timestamp_set.csv", index=False)
    (output_dir / "proposal_pack.json").write_text(
        json.dumps({"items": proposal_df.to_dict(orient="records")}, indent=2, default=str),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
