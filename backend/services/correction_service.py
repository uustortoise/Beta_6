import json
import logging
from datetime import date as dt_date
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path

import pandas as pd

from elderlycare_v1_16.config.settings import ARCHIVE_DATA_DIR, RAW_DATA_DIR
from services.db_utils import get_dashboard_connection, query_df
from utils.room_utils import normalize_room_name

logger = logging.getLogger(__name__)

# ── Activity Label Management ────────────────────────────────────────

SENSOR_COLUMNS = ["co2", "humidity", "motion", "temperature", "sound", "light"]
ROOM_MATCH_SQL = "LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = LOWER(REPLACE(REPLACE(?, ' ', ''), '_', ''))"

DEFAULT_ACTIVITY_LABELS = [
    "inactive", "unoccupied", "low_confidence",
    "bathroom_normal_use", "shower", "toilet",
    "kitchen_normal_use", "cooking", "washing_dishes",
    "livingroom_normal_use", "watch_tv", "nap",
    "bedroom_normal_use", "sleep", "change_clothes",
    "out", "fall",
]


def get_activity_labels() -> list[str]:
    """Get activity labels from database config or return defaults."""
    try:
        with get_dashboard_connection() as conn:
            result = conn.execute(
                "SELECT value FROM household_config WHERE key = 'custom_activity_labels'"
            ).fetchone()
            if result and result[0]:
                return json.loads(result[0])
    except Exception as e:
        logger.warning(f"Failed to get activity labels from DB: {e}")
    return list(DEFAULT_ACTIVITY_LABELS)


def _parse_sensor_features(raw_value) -> dict:
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, str):
        payload = raw_value.strip()
        if not payload:
            return {}
        try:
            parsed = json.loads(payload)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _ensure_correction_detail_table(conn) -> None:
    """Create correction detail table lazily for exact rollback snapshots."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS correction_history_detail (
            correction_id INTEGER NOT NULL,
            elder_id TEXT NOT NULL,
            room TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            old_activity TEXT,
            new_activity TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_correction_detail_cid ON correction_history_detail(correction_id)"
    )


def _normalize_ts_str(ts_value) -> str | None:
    parsed = pd.to_datetime(ts_value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _majority_label(values: list[str], fallback: str = "unknown") -> str:
    counter: dict[str, int] = {}
    for val in values:
        key = str(val or "").strip()
        if not key:
            continue
        counter[key] = counter.get(key, 0) + 1
    if not counter:
        return fallback
    return max(counter.items(), key=lambda item: item[1])[0]


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if int(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)


def _normalize_timeline_frame(df: pd.DataFrame, *, default_room: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out.get("timestamp"), errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if out.empty:
        return out

    if "end_time" in out.columns:
        out["end_time"] = pd.to_datetime(out["end_time"], errors="coerce")
    else:
        out["end_time"] = pd.NaT

    if "duration_minutes" in out.columns:
        durations = pd.to_numeric(out["duration_minutes"], errors="coerce").fillna(0.0)
        derived_end = out["timestamp"] + pd.to_timedelta(durations, unit="m")
        out["end_time"] = out["end_time"].fillna(derived_end)
    out["end_time"] = out["end_time"].where(
        out["end_time"].notna() & (out["end_time"] > out["timestamp"]),
        out["timestamp"] + pd.Timedelta(seconds=10),
    )

    if "room" not in out.columns:
        out["room"] = str(default_room or "").strip()
    else:
        out["room"] = out["room"].fillna(str(default_room or "")).astype(str)

    if "confidence" not in out.columns:
        out["confidence"] = pd.NA
    if "is_corrected" not in out.columns:
        out["is_corrected"] = 0

    out["time"] = out["timestamp"].dt.strftime("%H:%M:%S")
    return out


def _resolve_correction_id(
    cursor,
    elder_id: str,
    room: str,
    ts_start: datetime,
    ts_end: datetime,
    new_activity: str,
    corrected_by: str,
) -> int | None:
    # sqlite path
    last_id = getattr(cursor, "lastrowid", None)
    if last_id:
        try:
            return int(last_id)
        except Exception:
            pass

    # postgres/compat path fallback
    cursor.execute(
        """
        SELECT id
        FROM correction_history
        WHERE elder_id = ? AND room = ? AND timestamp_start = ? AND timestamp_end = ?
          AND new_activity = ? AND corrected_by = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (elder_id, room, ts_start, ts_end, new_activity, corrected_by),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return int(row[0])


def get_activity_timeline(elder_id: str, room: str, date: datetime.date) -> pd.DataFrame:
    """
    Fetch the full day of ADL data for the Altair Gantt timeline chart.
    Returns a DataFrame with timestamp, activity_type, confidence, duration_minutes.
    """
    record_date = date.isoformat()

    query = """
        SELECT timestamp, activity_type, confidence, duration_minutes, is_corrected
        FROM adl_history
        WHERE elder_id = ? AND """ + ROOM_MATCH_SQL + """
          AND (record_date = ? OR DATE(timestamp) = ?)
        ORDER BY timestamp ASC
    """
    df = query_df(query, (elder_id, room, record_date, record_date))
    if df.empty:
        segment_query = """
            SELECT
                start_time AS timestamp,
                end_time AS end_time,
                activity_type,
                avg_confidence AS confidence,
                duration_minutes,
                is_corrected,
                room
            FROM activity_segments
            WHERE elder_id = ? AND """ + ROOM_MATCH_SQL + """
              AND record_date = ?
            ORDER BY start_time ASC
        """
        df = query_df(segment_query, (elder_id, room, record_date))

    return _normalize_timeline_frame(df, default_room=room)


def get_activity_timeline_any_room(elder_id: str, date: datetime.date) -> pd.DataFrame:
    """
    Fetch full-day ADL timeline without room filtering.
    Used as legacy fallback when selected-room rows are empty.
    """
    record_date = date.isoformat()
    query = """
        SELECT timestamp, room, activity_type, confidence, duration_minutes, is_corrected
        FROM adl_history
        WHERE elder_id = ?
          AND (record_date = ? OR DATE(timestamp) = ?)
        ORDER BY timestamp ASC
    """
    df = query_df(query, (elder_id, record_date, record_date))
    if df.empty:
        segment_query = """
            SELECT
                start_time AS timestamp,
                end_time AS end_time,
                room,
                activity_type,
                avg_confidence AS confidence,
                duration_minutes,
                is_corrected
            FROM activity_segments
            WHERE elder_id = ? AND record_date = ?
            ORDER BY start_time ASC
        """
        df = query_df(segment_query, (elder_id, record_date))
    return _normalize_timeline_frame(df)


def get_training_timeline(elder_id: str, room: str, date: datetime.date) -> pd.DataFrame:
    """
    Load the room/day timeline that represents current labeled truth for review.
    Uses adl_history because corrected/manual labels are persisted there.
    """
    if not elder_id or not room:
        return pd.DataFrame()

    record_date = date.isoformat()
    query = """
        SELECT timestamp, room, activity_type, confidence, duration_minutes, is_corrected
        FROM adl_history
        WHERE elder_id = ? AND """ + ROOM_MATCH_SQL + """
          AND (record_date = ? OR DATE(timestamp) = ?)
        ORDER BY timestamp ASC
    """
    df = query_df(query, (elder_id, room, record_date, record_date))
    df = _normalize_timeline_frame(df, default_room=room)
    if df.empty:
        return df
    df["source_label"] = df["is_corrected"].map(lambda v: "corrected" if bool(v) else "recorded")
    return df


def get_prediction_timeline(elder_id: str, room: str, date: datetime.date) -> pd.DataFrame:
    """
    Load raw model predictions for the selected room/day from the predictions table.
    """
    if not elder_id or not room:
        return pd.DataFrame()

    record_date = date.isoformat()
    query = """
        SELECT
            timestamp,
            room,
            activity AS activity_type,
            confidence,
            0 AS is_corrected
        FROM predictions
        WHERE resident_id = ? AND """ + ROOM_MATCH_SQL + """
          AND DATE(timestamp) = ?
        ORDER BY timestamp ASC
    """
    df = query_df(query, (elder_id, room, record_date))
    df = _normalize_timeline_frame(df, default_room=room)
    if df.empty:
        return df
    df["source_label"] = "prediction"
    return df


def build_compare_timeline_payload(
    elder_id: str,
    room: str,
    date: datetime.date,
    confidence_threshold: float = 0.60,
) -> dict:
    """
    Build normalized training-vs-prediction timeline payloads for compare mode.
    """
    training_df = get_training_timeline(elder_id, room, date)
    prediction_df = get_prediction_timeline(elder_id, room, date)
    if prediction_df.empty:
        training_blocks = int(len(training_df.index)) if training_df is not None else 0
        return {
            "training_timeline": training_df,
            "prediction_timeline": prediction_df,
            "summary": {
                "training_blocks": training_blocks,
                "prediction_blocks": 0,
                "review_needed_blocks": 0,
                "manual_review_rate": None,
            },
        }

    compare_df = prediction_df.copy()
    compare_df["training_activity_type"] = None
    compare_df["training_source_label"] = None
    compare_df["is_unknown"] = compare_df["activity_type"].fillna("").astype(str).str.lower().isin(
        {"unknown", "low_confidence"}
    )
    compare_df["is_low_confidence"] = (
        pd.to_numeric(compare_df["confidence"], errors="coerce").fillna(0.0) < float(confidence_threshold)
    )
    compare_df["is_mismatch"] = False
    compare_df["is_corrected"] = False
    compare_df["review_reason"] = "clear"
    compare_df["prediction_state_label"] = compare_df["activity_type"].fillna("").astype(str)

    for idx, row in compare_df.iterrows():
        overlaps = pd.DataFrame()
        if not training_df.empty:
            overlaps = training_df[
                (training_df["timestamp"] < row["end_time"]) & (training_df["end_time"] > row["timestamp"])
            ].copy()
        if not overlaps.empty:
            truth_row = overlaps.iloc[0]
            truth_label = str(truth_row.get("activity_type") or "")
            compare_df.at[idx, "training_activity_type"] = truth_label
            compare_df.at[idx, "training_source_label"] = str(truth_row.get("source_label") or "recorded")
            compare_df.at[idx, "is_corrected"] = bool(overlaps["is_corrected"].fillna(0).astype(bool).any())
            compare_df.at[idx, "is_mismatch"] = bool(
                truth_label and truth_label != str(row.get("activity_type") or "")
            )
        else:
            compare_df.at[idx, "is_corrected"] = False

        review_flags = []
        if bool(compare_df.at[idx, "is_unknown"]):
            review_flags.append("unknown")
            compare_df.at[idx, "prediction_state_label"] = "unknown / review needed"
        elif bool(compare_df.at[idx, "is_low_confidence"]):
            review_flags.append("low_confidence")
            compare_df.at[idx, "prediction_state_label"] = "low confidence / review needed"
        if bool(compare_df.at[idx, "is_mismatch"]):
            review_flags.append("mismatch")
        if bool(compare_df.at[idx, "is_corrected"]):
            review_flags.append("corrected_truth")
        compare_df.at[idx, "review_reason"] = " / ".join(review_flags) if review_flags else "clear"

    review_needed = int((compare_df["review_reason"] != "clear").sum())
    prediction_blocks = int(len(compare_df.index))
    training_blocks = int(len(training_df.index)) if training_df is not None else 0
    return {
        "training_timeline": training_df,
        "prediction_timeline": compare_df,
        "summary": {
            "training_blocks": training_blocks,
            "prediction_blocks": prediction_blocks,
            "review_needed_blocks": review_needed,
            "manual_review_rate": _safe_rate(review_needed, prediction_blocks),
        },
    }


def get_timeline_reliability_metrics(
    elder_id: str,
    days: int = 30,
    confidence_threshold: float = 0.60,
) -> dict:
    """
    Compute product-facing timeline reliability and correction-load metrics.
    """
    elder = str(elder_id or "").strip()
    if not elder:
        return {
            "days": int(days),
            "correction_volume": 0,
            "review_backlog": 0,
            "prediction_blocks": 0,
            "review_needed_blocks": 0,
            "manual_review_rate": None,
            "unknown_abstain_rate": None,
            "contradiction_rate": None,
            "fragmentation_rate": None,
            "unknown_abstain_trend": [],
        }

    days = max(1, int(days))
    cutoff_dt = datetime.now() - timedelta(days=days)
    cutoff_ts = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")

    correction_volume = 0
    review_backlog = 0
    try:
        corr_df = query_df(
            """
            SELECT COUNT(*) AS count
            FROM correction_history
            WHERE elder_id = ?
              AND COALESCE(is_deleted, 0) = 0
              AND COALESCE(corrected_at, timestamp_end, timestamp_start) >= ?
            """,
            (elder, cutoff_ts),
        )
        if not corr_df.empty:
            correction_volume = int(corr_df.iloc[0].get("count", 0) or 0)
    except Exception:
        correction_volume = 0

    try:
        backlog_df = query_df(
            """
            SELECT COUNT(*) AS count
            FROM adl_history
            WHERE elder_id = ?
              AND timestamp >= ?
              AND COALESCE(is_corrected, 0) = 0
              AND confidence < ?
            """,
            (elder, cutoff_ts, float(confidence_threshold)),
        )
        if not backlog_df.empty:
            review_backlog = int(backlog_df.iloc[0].get("count", 0) or 0)
    except Exception:
        review_backlog = 0

    pred_df = query_df(
        """
        SELECT timestamp, room, activity AS activity_type, confidence
        FROM predictions
        WHERE resident_id = ?
          AND timestamp >= ?
        ORDER BY timestamp ASC
        """,
        (elder, cutoff_ts),
    )

    if pred_df.empty:
        return {
            "days": int(days),
            "correction_volume": int(correction_volume),
            "review_backlog": int(review_backlog),
            "prediction_blocks": 0,
            "review_needed_blocks": 0,
            "manual_review_rate": None,
            "unknown_abstain_rate": None,
            "contradiction_rate": None,
            "fragmentation_rate": None,
            "unknown_abstain_trend": [],
        }

    out = pred_df.copy()
    out["timestamp"] = pd.to_datetime(out.get("timestamp"), errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if out.empty:
        return {
            "days": int(days),
            "correction_volume": int(correction_volume),
            "review_backlog": int(review_backlog),
            "prediction_blocks": 0,
            "review_needed_blocks": 0,
            "manual_review_rate": None,
            "unknown_abstain_rate": None,
            "contradiction_rate": None,
            "fragmentation_rate": None,
            "unknown_abstain_trend": [],
        }

    out["activity_norm"] = out["activity_type"].fillna("").astype(str).str.strip().str.lower()
    out["confidence_num"] = pd.to_numeric(out.get("confidence"), errors="coerce").fillna(0.0)
    out["room_norm"] = out.get("room").fillna("").astype(str).map(lambda v: normalize_room_name(v) or "")
    out["ts_key"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    unknown_labels = {"unknown", "low_confidence", "abstain"}
    out["is_unknown_abstain"] = out["activity_norm"].isin(unknown_labels)
    out["is_review_needed"] = out["is_unknown_abstain"] | (out["confidence_num"] < float(confidence_threshold))

    prediction_blocks = int(len(out.index))
    review_needed_blocks = int(out["is_review_needed"].sum())
    unknown_abstain_blocks = int(out["is_unknown_abstain"].sum())

    # Fragmentation rate: activity label switches between adjacent prediction blocks per room.
    transitions = 0
    transitions_base = 0
    for _, room_df in out.groupby("room_norm", dropna=False):
        room_sorted = room_df.sort_values("timestamp")
        if len(room_sorted.index) <= 1:
            continue
        labels = room_sorted["activity_norm"].tolist()
        transitions += int(sum(1 for idx in range(1, len(labels)) if labels[idx] != labels[idx - 1]))
        transitions_base += int(len(labels) - 1)
    fragmentation_rate = _safe_rate(transitions, transitions_base)

    # Contradiction rate: mismatch against available adl labels at identical room+timestamp keys.
    contradiction_rate = None
    truth_df = query_df(
        """
        SELECT timestamp, room, activity_type
        FROM adl_history
        WHERE elder_id = ?
          AND timestamp >= ?
        """,
        (elder, cutoff_ts),
    )
    if not truth_df.empty:
        truth = truth_df.copy()
        truth["timestamp"] = pd.to_datetime(truth.get("timestamp"), errors="coerce")
        truth = truth.dropna(subset=["timestamp"])
        if not truth.empty:
            truth["activity_norm"] = truth["activity_type"].fillna("").astype(str).str.strip().str.lower()
            truth["room_norm"] = truth.get("room").fillna("").astype(str).map(lambda v: normalize_room_name(v) or "")
            truth["ts_key"] = truth["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            merged = out.merge(
                truth[["ts_key", "room_norm", "activity_norm"]],
                on=["ts_key", "room_norm"],
                how="inner",
                suffixes=("_pred", "_truth"),
            )
            if not merged.empty:
                mismatches = int(
                    (merged["activity_norm_pred"].fillna("") != merged["activity_norm_truth"].fillna("")).sum()
                )
                contradiction_rate = _safe_rate(mismatches, int(len(merged.index)))

    trend_df = (
        out.assign(day_key=out["timestamp"].dt.strftime("%Y-%m-%d"))
        .groupby("day_key", as_index=False)
        .agg(
            prediction_blocks=("ts_key", "count"),
            unknown_abstain_blocks=("is_unknown_abstain", "sum"),
        )
    )
    trend_df["unknown_abstain_rate"] = trend_df.apply(
        lambda row: _safe_rate(int(row["unknown_abstain_blocks"]), int(row["prediction_blocks"])),
        axis=1,
    )
    trend_rows = [
        {
            "day": str(row["day_key"]),
            "prediction_blocks": int(row["prediction_blocks"]),
            "unknown_abstain_blocks": int(row["unknown_abstain_blocks"]),
            "unknown_abstain_rate": float(row["unknown_abstain_rate"])
            if row["unknown_abstain_rate"] is not None
            else None,
        }
        for _, row in trend_df.sort_values("day_key").iterrows()
    ]

    return {
        "days": int(days),
        "correction_volume": int(correction_volume),
        "review_backlog": int(review_backlog),
        "prediction_blocks": prediction_blocks,
        "review_needed_blocks": review_needed_blocks,
        "manual_review_rate": _safe_rate(review_needed_blocks, prediction_blocks),
        "unknown_abstain_rate": _safe_rate(unknown_abstain_blocks, prediction_blocks),
        "contradiction_rate": contradiction_rate,
        "fragmentation_rate": fragmentation_rate,
        "unknown_abstain_trend": trend_rows,
    }


def find_nearest_activity_date(elder_id: str, target_date: dt_date, room: str | None = None) -> dict | None:
    """
    Find nearest available activity date for one resident (optionally room-scoped).
    Searches both adl_history and activity_segments.
    """
    if not elder_id:
        return None

    scoped_room = str(room or "").strip()
    by_day: dict[dt_date, int] = {}

    adl_query = """
        SELECT record_date AS day_key, COUNT(*) AS row_count
        FROM adl_history
        WHERE elder_id = ?
    """
    adl_params: list[object] = [elder_id]
    if scoped_room:
        adl_query += " AND " + ROOM_MATCH_SQL
        adl_params.append(scoped_room)
    adl_query += " GROUP BY record_date"

    seg_query = """
        SELECT record_date AS day_key, COUNT(*) AS row_count
        FROM activity_segments
        WHERE elder_id = ?
    """
    seg_params: list[object] = [elder_id]
    if scoped_room:
        seg_query += " AND " + ROOM_MATCH_SQL
        seg_params.append(scoped_room)
    seg_query += " GROUP BY record_date"

    for query, params in ((adl_query, tuple(adl_params)), (seg_query, tuple(seg_params))):
        try:
            df = query_df(query, params)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            parsed = pd.to_datetime(row.get("day_key"), errors="coerce")
            if pd.isna(parsed):
                continue
            key = parsed.date()
            by_day[key] = by_day.get(key, 0) + int(row.get("row_count") or 0)

    if not by_day:
        return None

    nearest_day = min(by_day.keys(), key=lambda d: (abs((d - target_date).days), d))
    return {"date": nearest_day, "rows": int(by_day.get(nearest_day, 0))}


def get_low_confidence_queue(elder_id: str, room: str, date: datetime.date, threshold: float = 0.50) -> pd.DataFrame:
    """Fetch low-confidence predictions from adl_history for review."""
    if not elder_id or not room:
        return pd.DataFrame()
        
    record_date = date.isoformat()
    
    query = """
        SELECT id, elder_id, timestamp, room, activity_type, confidence, duration_minutes, is_corrected
        FROM adl_history
        WHERE elder_id = ? AND """ + ROOM_MATCH_SQL + """
          AND (record_date = ? OR DATE(timestamp) = ?)
          AND confidence < ? AND is_corrected = 0
        ORDER BY timestamp ASC
    """
    return query_df(query, (elder_id, room, record_date, record_date, threshold))


def get_sensor_context(elder_id: str, room: str, ts_start: str, ts_end: str) -> dict:
    """
    Extract aggregated sensor context (CO2, motion, etc.) from the adl_history sensor_features
    for the specified time window.
    """
    query = """
        SELECT sensor_features 
        FROM adl_history 
        WHERE elder_id = ? AND """ + ROOM_MATCH_SQL + """ AND timestamp >= ? AND timestamp <= ?
    """
    df = query_df(query, (elder_id, room, ts_start, ts_end))
    
    if df.empty:
        return {}
        
    context = {"motion": 0, "co2_max": 0, "sound_max": 0}
    
    for _, row in df.iterrows():
        features = _parse_sensor_features(row.get("sensor_features"))
        if isinstance(features, dict):
            # Aggregate over the window
            if features.get("motion", 0) > 0:
                context["motion"] += 1
            context["co2_max"] = max(context["co2_max"], features.get("co2", 0))    
            context["sound_max"] = max(context["sound_max"], features.get("sound", 0))
            
    return context


def get_sensor_timeseries(elder_id: str, room: str, date: datetime.date) -> pd.DataFrame:
    """
    Return per-timestamp sensor series extracted from adl_history.sensor_features
    for one resident/room/day, alongside activity/confidence for side-by-side review.
    """
    if not elder_id or not room:
        return pd.DataFrame()

    record_date = date.isoformat()

    query = """
        SELECT timestamp, activity_type, confidence, sensor_features
        FROM adl_history
        WHERE elder_id = ? AND """ + ROOM_MATCH_SQL + """
          AND (record_date = ? OR DATE(timestamp) = ?)
        ORDER BY timestamp ASC
    """
    df = query_df(query, (elder_id, room, record_date, record_date))
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return df

    sensor_payload = df["sensor_features"].apply(_parse_sensor_features)
    for col in SENSOR_COLUMNS:
        df[col] = pd.to_numeric(sensor_payload.apply(lambda payload: payload.get(col)), errors="coerce")

    return df[["timestamp", "activity_type", "confidence", *SENSOR_COLUMNS]]


def save_correction(
    elder_id: str, 
    room: str, 
    df_rows: list[dict], 
    new_activity: str,
    corrected_by: str = "operator"
) -> tuple[bool, str, int]:
    """
    Save manual corrections atomically.
    Updates adl_history and writes an audit trail to correction_history.
    """
    if not df_rows:
        return False, "No rows selected.", 0
        
    try:
        timestamps = [pd.to_datetime(r["timestamp"], errors="coerce") for r in df_rows]
        timestamps = [ts for ts in timestamps if pd.notna(ts)]
        if not timestamps:
            return False, "No valid timestamps selected.", 0

        ts_start = min(timestamps)
        ts_end = max(timestamps) + timedelta(minutes=1)
        ts_strs = sorted({ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps})
        placeholders = ",".join(["?"] * len(ts_strs))

        with get_dashboard_connection() as conn:
            cursor = conn.cursor()
            _ensure_correction_detail_table(conn)

            cursor.execute(
                f"""
                SELECT timestamp, activity_type
                FROM adl_history
                WHERE elder_id = ? AND {ROOM_MATCH_SQL} AND timestamp IN ({placeholders})
                ORDER BY timestamp ASC
                """,
                [elder_id, room] + ts_strs,
            )
            pre_rows = cursor.fetchall() or []
            snapshot_rows = []
            for row in pre_rows:
                ts_norm = _normalize_ts_str(row[0])
                if not ts_norm:
                    continue
                snapshot_rows.append(
                    {
                        "timestamp": ts_norm,
                        "old_activity": str(row[1] or "unknown"),
                    }
                )

            old_activity = _majority_label([r["old_activity"] for r in snapshot_rows], fallback="unknown")

            update_query = f"""
                UPDATE adl_history
                SET activity_type = ?, is_corrected = 1, confidence = 1.0
                WHERE elder_id = ? AND {ROOM_MATCH_SQL} AND timestamp IN ({placeholders})
            """
            cursor.execute(update_query, [new_activity, elder_id, room] + ts_strs)
            rows_affected = int(cursor.rowcount or 0)

            cursor.execute(
                """
                INSERT INTO correction_history
                (elder_id, room, timestamp_start, timestamp_end, old_activity, new_activity, rows_affected, corrected_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    elder_id,
                    room,
                    ts_start.to_pydatetime(),
                    ts_end.to_pydatetime(),
                    old_activity,
                    new_activity,
                    rows_affected,
                    corrected_by,
                ),
            )
            correction_id = _resolve_correction_id(
                cursor=cursor,
                elder_id=elder_id,
                room=room,
                ts_start=ts_start.to_pydatetime(),
                ts_end=ts_end.to_pydatetime(),
                new_activity=new_activity,
                corrected_by=corrected_by,
            )
            if correction_id is not None and snapshot_rows:
                cursor.executemany(
                    """
                    INSERT INTO correction_history_detail
                    (correction_id, elder_id, room, timestamp, old_activity, new_activity)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            correction_id,
                            elder_id,
                            room,
                            row["timestamp"],
                            row["old_activity"],
                            new_activity,
                        )
                        for row in snapshot_rows
                    ],
                )

            conn.commit()
            return True, "Correction saved successfully.", rows_affected
            
    except Exception as e:
        logger.exception("Failed to save correction to DB")
        return False, f"Database error: {e}", 0


def _coerce_time(value) -> dt_time:
    if isinstance(value, dt_time):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid time value: {value}")
    return parsed.time()


def apply_batch_corrections(
    elder_id: str,
    room: str,
    record_date: dt_date,
    corrections: list[dict],
    corrected_by: str = "admin_ui",
) -> dict:
    """
    Apply queued time-range corrections for one resident/room/day.
    Returns summary with per-range status.
    """
    if not elder_id or not room:
        return {"ok": False, "error": "elder_id and room are required", "ranges_applied": 0, "rows_updated": 0}
    if not corrections:
        return {"ok": False, "error": "No queued corrections", "ranges_applied": 0, "rows_updated": 0}

    ranges_applied = 0
    rows_updated = 0
    details: list[dict] = []

    try:
        with get_dashboard_connection() as conn:
            cursor = conn.cursor()
            _ensure_correction_detail_table(conn)
            for item in corrections:
                start_time = _coerce_time(item.get("start_time"))
                end_time = _coerce_time(item.get("end_time"))
                new_label = str(item.get("new_label", "")).strip()
                if not new_label:
                    details.append({"ok": False, "reason": "missing_new_label", "item": item})
                    continue
                if start_time >= end_time:
                    details.append({"ok": False, "reason": "invalid_time_range", "item": item})
                    continue

                ts_start = datetime.combine(record_date, start_time)
                ts_end = datetime.combine(record_date, end_time)

                cursor.execute(
                    """
                    SELECT timestamp, activity_type
                    FROM adl_history
                    WHERE elder_id = ? AND """ + ROOM_MATCH_SQL + """ AND timestamp >= ? AND timestamp <= ?
                    """,
                    (elder_id, room, ts_start, ts_end),
                )
                pre_rows = cursor.fetchall() or []
                snapshot_rows = []
                for row in pre_rows:
                    ts_norm = _normalize_ts_str(row[0])
                    if not ts_norm:
                        continue
                    snapshot_rows.append(
                        {
                            "timestamp": ts_norm,
                            "old_activity": str(row[1] or "unknown"),
                        }
                    )
                old_activity = _majority_label(
                    [r["old_activity"] for r in snapshot_rows],
                    fallback=str(item.get("old_activity", "batch")),
                )

                cursor.execute(
                    """
                    UPDATE adl_history
                    SET activity_type = ?, is_corrected = 1, confidence = 1.0
                    WHERE elder_id = ? AND """ + ROOM_MATCH_SQL + """ AND timestamp >= ? AND timestamp <= ?
                    """,
                    (new_label, elder_id, room, ts_start, ts_end),
                )
                affected = int(cursor.rowcount or 0)
                if affected <= 0:
                    details.append(
                        {
                            "ok": False,
                            "reason": "no_rows_matched",
                            "range": f"{start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}",
                        }
                    )
                    continue

                cursor.execute(
                    """
                    INSERT INTO correction_history
                    (elder_id, room, timestamp_start, timestamp_end, old_activity, new_activity, rows_affected, corrected_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (elder_id, room, ts_start, ts_end, old_activity, new_label, affected, corrected_by),
                )
                correction_id = _resolve_correction_id(
                    cursor=cursor,
                    elder_id=elder_id,
                    room=room,
                    ts_start=ts_start,
                    ts_end=ts_end,
                    new_activity=new_label,
                    corrected_by=corrected_by,
                )
                if correction_id is not None and snapshot_rows:
                    cursor.executemany(
                        """
                        INSERT INTO correction_history_detail
                        (correction_id, elder_id, room, timestamp, old_activity, new_activity)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                correction_id,
                                elder_id,
                                room,
                                row["timestamp"],
                                row["old_activity"],
                                new_label,
                            )
                            for row in snapshot_rows
                        ],
                    )
                ranges_applied += 1
                rows_updated += affected
                details.append(
                    {
                        "ok": True,
                        "new_label": new_label,
                        "rows_affected": affected,
                        "range": f"{start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}",
                    }
                )

            if ranges_applied > 0:
                try:
                    from utils.segment_utils import regenerate_segments

                    regenerate_segments(elder_id, room, record_date.isoformat(), conn)
                except Exception as e:
                    logger.warning("Segment regeneration failed after batch correction: %s", e)

            conn.commit()

        return {
            "ok": ranges_applied > 0,
            "ranges_applied": ranges_applied,
            "rows_updated": rows_updated,
            "details": details,
        }
    except Exception as e:
        logger.exception("Failed to apply batch corrections")
        return {
            "ok": False,
            "error": f"{e}",
            "ranges_applied": ranges_applied,
            "rows_updated": rows_updated,
            "details": details,
        }


def preview_batch_corrections(
    elder_id: str,
    room: str,
    record_date: dt_date,
    corrections: list[dict],
) -> dict:
    """
    Dry-run preview for queued corrections:
    - validates time ranges
    - detects overlaps
    - estimates affected rows per range from DB
    """
    rows: list[dict] = []
    if not elder_id or not room:
        return {
            "ok": False,
            "has_invalid": True,
            "has_overlap": False,
            "estimated_rows": 0,
            "rows": rows,
            "error": "elder_id and room are required",
        }

    normalized_items: list[dict] = []
    for idx, item in enumerate(corrections or []):
        try:
            start_time = _coerce_time(item.get("start_time"))
            end_time = _coerce_time(item.get("end_time"))
        except Exception:
            rows.append(
                {
                    "index": idx,
                    "status": "invalid",
                    "reason": "invalid_time",
                    "start": str(item.get("start_time")),
                    "end": str(item.get("end_time")),
                    "new_label": str(item.get("new_label", "")),
                    "estimated_rows": 0,
                }
            )
            continue
        new_label = str(item.get("new_label", "")).strip()
        if not new_label:
            rows.append(
                {
                    "index": idx,
                    "status": "invalid",
                    "reason": "missing_label",
                    "start": start_time.strftime("%H:%M:%S"),
                    "end": end_time.strftime("%H:%M:%S"),
                    "new_label": "",
                    "estimated_rows": 0,
                }
            )
            continue
        if start_time >= end_time:
            rows.append(
                {
                    "index": idx,
                    "status": "invalid",
                    "reason": "start_not_before_end",
                    "start": start_time.strftime("%H:%M:%S"),
                    "end": end_time.strftime("%H:%M:%S"),
                    "new_label": new_label,
                    "estimated_rows": 0,
                }
            )
            continue
        normalized_items.append(
            {
                "index": idx,
                "start_time": start_time,
                "end_time": end_time,
                "new_label": new_label,
            }
        )

    # overlap detection (within same resident/room/day queue)
    overlap_indexes: set[int] = set()
    ordered = sorted(
        normalized_items,
        key=lambda it: (it["start_time"].hour, it["start_time"].minute, it["start_time"].second),
    )
    for i in range(1, len(ordered)):
        prev = ordered[i - 1]
        cur = ordered[i]
        if cur["start_time"] < prev["end_time"]:
            overlap_indexes.add(prev["index"])
            overlap_indexes.add(cur["index"])

    estimate_map: dict[int, int] = {}
    try:
        with get_dashboard_connection() as conn:
            cursor = conn.cursor()
            for item in normalized_items:
                ts_start = datetime.combine(record_date, item["start_time"])
                ts_end = datetime.combine(record_date, item["end_time"])
                cursor.execute(
                    """
                    SELECT COUNT(*) AS cnt
                    FROM adl_history
                    WHERE elder_id = ? AND """ + ROOM_MATCH_SQL + """ AND timestamp >= ? AND timestamp <= ?
                    """,
                    (elder_id, room, ts_start, ts_end),
                )
                row = cursor.fetchone()
                estimate_map[item["index"]] = int(row[0]) if row else 0
    except Exception as e:
        logger.warning("Dry-run preview failed to estimate rows: %s", e)

    for item in normalized_items:
        idx = item["index"]
        reason = "overlap" if idx in overlap_indexes else ""
        status = "invalid" if reason else "ok"
        rows.append(
            {
                "index": idx,
                "status": status,
                "reason": reason,
                "start": item["start_time"].strftime("%H:%M:%S"),
                "end": item["end_time"].strftime("%H:%M:%S"),
                "new_label": item["new_label"],
                "estimated_rows": int(estimate_map.get(idx, 0)),
            }
        )

    has_invalid = any(str(r.get("status")) != "ok" for r in rows)
    has_overlap = any(str(r.get("reason")) == "overlap" for r in rows)
    estimated_rows = sum(int(r.get("estimated_rows") or 0) for r in rows if str(r.get("status")) == "ok")
    return {
        "ok": not has_invalid,
        "has_invalid": has_invalid,
        "has_overlap": has_overlap,
        "estimated_rows": estimated_rows,
        "rows": sorted(rows, key=lambda r: int(r.get("index", 0))),
    }


def build_correction_training_file(elder_id: str, room: str, record_date: dt_date) -> Path:
    """
    Materialize a one-room corrected training file from DB for manual retrain.
    """
    sensor_df = get_sensor_timeseries(elder_id=elder_id, room=room, date=record_date)
    if sensor_df.empty:
        raise ValueError("No rows available for corrected training file.")

    training_df = sensor_df.copy()
    training_df = training_df.rename(columns={"activity_type": "activity"})
    # Platform expects physical sensors including vibration.
    for col in ["motion", "temperature", "light", "sound", "co2", "humidity", "vibration"]:
        if col not in training_df.columns:
            training_df[col] = 0.0
    training_df["activity"] = training_df["activity"].fillna("inactive").astype(str)
    training_df = training_df[
        ["timestamp", "motion", "temperature", "light", "sound", "co2", "humidity", "vibration", "activity"]
    ].copy()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dir = Path(RAW_DATA_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_room = str(room).strip().lower().replace(" ", "_")
    out_path = raw_dir / f"{elder_id}_{safe_room}_manual_train_{record_date.isoformat()}_{ts}.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        training_df.to_excel(writer, sheet_name=safe_room[:31] or "room", index=False)
    return out_path


def retrain_after_batch_corrections(
    elder_id: str,
    room: str,
    record_date: dt_date,
    train_retro: bool = False,
) -> dict:
    """
    Best-effort one-shot retrain + repredict after batch corrections.
    Returns structured summary for UI status display.
    """
    manual_file = build_correction_training_file(elder_id=elder_id, room=room, record_date=record_date)
    file_paths = [str(manual_file)]

    if train_retro:
        try:
            from utils.data_loader import get_archive_files

            archived = get_archive_files(Path(ARCHIVE_DATA_DIR), resident_id=elder_id)
            for entry in archived:
                fpath = str(entry.get("path", "")).strip()
                if fpath and fpath not in file_paths:
                    file_paths.append(fpath)
        except Exception as e:
            logger.warning("Failed to extend retrain set with archive files: %s", e)

    from ml.pipeline import UnifiedPipeline

    pipeline = UnifiedPipeline(enable_denoising=True)
    _, metrics = pipeline.train_from_files(
        file_paths=file_paths,
        elder_id=elder_id,
        rooms={room},
        training_mode="correction_fine_tune",
    )
    pipeline.repredict_all(elder_id=elder_id, archive_dir=ARCHIVE_DATA_DIR, rooms={room})

    room_metrics = []
    for m in metrics or []:
        if normalize_room_name(str(m.get("room", ""))) == normalize_room_name(room):
            room_metrics.append(m)

    return {
        "ok": True,
        "manual_file": str(manual_file),
        "files_used": len(file_paths),
        "room_metrics": room_metrics,
    }
