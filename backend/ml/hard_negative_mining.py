import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RISKY_ROOMS = {"bathroom", "entrance", "kitchen"}


def normalize_room_token(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def _parse_sensor_features(raw_value: Any) -> Dict[str, Any]:
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, str):
        txt = raw_value.strip()
        if not txt:
            return {}
        try:
            parsed = json.loads(txt)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def ensure_hard_negative_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hard_negative_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            elder_id TEXT NOT NULL,
            room TEXT NOT NULL,
            record_date DATE NOT NULL,
            timestamp_start DATETIME NOT NULL,
            timestamp_end DATETIME NOT NULL,
            duration_minutes REAL,
            reason TEXT NOT NULL,
            score REAL NOT NULL,
            suggested_label TEXT,
            source TEXT DEFAULT 'hard_negative_miner_v1',
            status TEXT DEFAULT 'open',
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_hnq_unique
        ON hard_negative_queue(elder_id, room, timestamp_start, timestamp_end, reason)
        """
    )


def _fetch_rows(conn, elder_id: str, record_date: Optional[str] = None) -> List[Dict[str, Any]]:
    if record_date:
        cursor = conn.execute(
            """
            SELECT timestamp, room, activity_type, confidence, is_anomaly, sensor_features, record_date
            FROM adl_history
            WHERE elder_id = ? AND record_date = ?
            ORDER BY room, timestamp
            """,
            (elder_id, record_date),
        )
    else:
        cursor = conn.execute(
            """
            SELECT timestamp, room, activity_type, confidence, is_anomaly, sensor_features, record_date
            FROM adl_history
            WHERE elder_id = ?
            ORDER BY room, timestamp
            """,
            (elder_id,),
        )
    rows = cursor.fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        try:
            out.append(
                {
                    "timestamp": row["timestamp"],
                    "room": row["room"],
                    "activity_type": row["activity_type"],
                    "confidence": row["confidence"],
                    "is_anomaly": row["is_anomaly"],
                    "sensor_features": row["sensor_features"],
                    "record_date": row["record_date"],
                }
            )
        except Exception:
            out.append(
                {
                    "timestamp": row[0],
                    "room": row[1],
                    "activity_type": row[2],
                    "confidence": row[3],
                    "is_anomaly": row[4],
                    "sensor_features": row[5],
                    "record_date": row[6],
                }
            )
    return out


def _build_candidates_for_room(
    room_df: pd.DataFrame,
    top_k_per_room: int = 3,
    min_block_rows: int = 6,
) -> List[Dict[str, Any]]:
    if room_df.empty:
        return []
    room_df = room_df.sort_values("timestamp").reset_index(drop=True)
    room_df["activity_norm"] = room_df["activity_type"].astype(str).str.strip().str.lower()
    room_df["is_low_conf"] = room_df["is_low_confidence"] | (room_df["confidence"] < 0.55)
    room_df["label_change"] = room_df["activity_norm"].ne(room_df["activity_norm"].shift(1)).fillna(False)
    room_df["flag"] = room_df["is_low_conf"] | room_df["label_change"] | room_df["is_anomaly_flag"]

    if not bool(room_df["flag"].any()):
        return []

    candidates: List[Dict[str, Any]] = []
    gap_limit = pd.Timedelta(seconds=20)
    flagged = room_df[room_df["flag"]].copy().reset_index(drop=True)
    block_start = 0
    for i in range(1, len(flagged) + 1):
        new_block = False
        if i == len(flagged):
            new_block = True
        else:
            dt_gap = flagged.loc[i, "timestamp"] - flagged.loc[i - 1, "timestamp"]
            if dt_gap > gap_limit:
                new_block = True
        if not new_block:
            continue

        block = flagged.iloc[block_start:i].copy()
        block_start = i
        if len(block) < int(min_block_rows):
            continue

        ts_start = pd.to_datetime(block["timestamp"].iloc[0])
        ts_end = pd.to_datetime(block["timestamp"].iloc[-1])
        duration_minutes = max(0.0, (ts_end - ts_start).total_seconds() / 60.0)
        low_conf_rate = float(block["is_low_conf"].mean()) if len(block) > 0 else 0.0
        change_density = (
            float(block["label_change"].sum()) / float(max(1, len(block) - 1))
            if len(block) > 1
            else 0.0
        )
        anomaly_rate = float(block["is_anomaly_flag"].mean()) if len(block) > 0 else 0.0
        score = 0.55 * low_conf_rate + 0.30 * min(1.0, change_density) + 0.15 * anomaly_rate
        score += 0.05 * min(1.0, duration_minutes / 5.0)

        if low_conf_rate >= 0.5:
            reason = "low_confidence_cluster"
        elif change_density >= 0.4:
            reason = "unstable_transition_cluster"
        elif anomaly_rate > 0:
            reason = "anomaly_cluster"
        else:
            reason = "mixed_uncertainty_cluster"

        suggested = None
        for col in ("low_confidence_hint_label", "predicted_top1_label", "activity_type"):
            if col in block.columns:
                vals = block[col].dropna().astype(str).str.strip()
                vals = vals[vals != ""]
                if not vals.empty:
                    suggested = vals.mode().iloc[0]
                    break

        candidates.append(
            {
                "timestamp_start": ts_start,
                "timestamp_end": ts_end,
                "duration_minutes": float(duration_minutes),
                "reason": reason,
                "score": float(score),
                "suggested_label": suggested,
            }
        )

    candidates = sorted(
        candidates,
        key=lambda x: (float(x["score"]), float(x["duration_minutes"])),
        reverse=True,
    )
    return candidates[: max(1, int(top_k_per_room))]


def mine_hard_negative_windows(
    conn,
    elder_id: str,
    record_date: Optional[str] = None,
    risky_rooms: Optional[Iterable[str]] = None,
    top_k_per_room: int = 3,
    min_block_rows: int = 6,
    source: str = "hard_negative_miner_v1",
) -> Dict[str, int]:
    ensure_hard_negative_table(conn)
    risky = {
        normalize_room_token(r)
        for r in (risky_rooms if risky_rooms is not None else DEFAULT_RISKY_ROOMS)
        if normalize_room_token(r)
    }
    rows = _fetch_rows(conn=conn, elder_id=elder_id, record_date=record_date)
    if not rows:
        return {"candidates": 0, "inserted": 0, "updated": 0}

    df = pd.DataFrame(rows)
    if df.empty:
        return {"candidates": 0, "inserted": 0, "updated": 0}

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["room_norm"] = df["room"].map(normalize_room_token)
    df = df[df["room_norm"].isin(risky)].copy()
    if df.empty:
        return {"candidates": 0, "inserted": 0, "updated": 0}

    parsed = df["sensor_features"].apply(_parse_sensor_features)
    df["is_low_confidence"] = parsed.apply(lambda p: bool(p.get("is_low_confidence", False)))
    df["low_confidence_hint_label"] = parsed.apply(lambda p: p.get("low_confidence_hint_label"))
    df["predicted_top1_label"] = parsed.apply(lambda p: p.get("predicted_top1_label"))
    df["is_anomaly_flag"] = pd.to_numeric(df["is_anomaly"], errors="coerce").fillna(0).astype(int) > 0
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0).astype(float)

    inserted = 0
    updated = 0
    total_candidates = 0
    for room_key, room_df in df.groupby("room_norm"):
        cands = _build_candidates_for_room(
            room_df=room_df,
            top_k_per_room=int(top_k_per_room),
            min_block_rows=int(min_block_rows),
        )
        total_candidates += len(cands)
        for cand in cands:
            ts_start = pd.to_datetime(cand["timestamp_start"])
            ts_end = pd.to_datetime(cand["timestamp_end"])
            rec_date = (
                str(record_date)
                if record_date
                else ts_start.strftime("%Y-%m-%d")
            )
            row = conn.execute(
                """
                SELECT id
                FROM hard_negative_queue
                WHERE elder_id = ?
                  AND room = ?
                  AND timestamp_start = ?
                  AND timestamp_end = ?
                  AND reason = ?
                LIMIT 1
                """,
                (
                    elder_id,
                    room_key,
                    ts_start.strftime("%Y-%m-%d %H:%M:%S"),
                    ts_end.strftime("%Y-%m-%d %H:%M:%S"),
                    cand["reason"],
                ),
            ).fetchone()
            metadata = json.dumps(
                {
                    "duration_minutes": float(cand["duration_minutes"]),
                    "room_norm": room_key,
                }
            )
            if row:
                try:
                    row_id = row["id"]
                except Exception:
                    row_id = row[0]
                conn.execute(
                    """
                    UPDATE hard_negative_queue
                    SET score = ?, suggested_label = ?, source = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        float(cand["score"]),
                        cand.get("suggested_label"),
                        str(source),
                        metadata,
                        int(row_id),
                    ),
                )
                updated += 1
            else:
                conn.execute(
                    """
                    INSERT INTO hard_negative_queue
                    (elder_id, room, record_date, timestamp_start, timestamp_end, duration_minutes, reason, score, suggested_label, source, status, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
                    """,
                    (
                        elder_id,
                        room_key,
                        rec_date,
                        ts_start.strftime("%Y-%m-%d %H:%M:%S"),
                        ts_end.strftime("%Y-%m-%d %H:%M:%S"),
                        float(cand["duration_minutes"]),
                        cand["reason"],
                        float(cand["score"]),
                        cand.get("suggested_label"),
                        str(source),
                        metadata,
                    ),
                )
                inserted += 1
    return {"candidates": total_candidates, "inserted": inserted, "updated": updated}


def fetch_hard_negative_queue(
    conn,
    elder_id: str,
    room: Optional[str] = None,
    days: int = 30,
    status: str = "open",
    limit: int = 200,
) -> pd.DataFrame:
    ensure_hard_negative_table(conn)
    cutoff = (datetime.now() - timedelta(days=max(1, int(days)))).strftime("%Y-%m-%d %H:%M:%S")
    params: List[Any] = [elder_id, cutoff]
    query = """
        SELECT id, elder_id, room, record_date, timestamp_start, timestamp_end,
               duration_minutes, reason, score, suggested_label, source, status, created_at, updated_at
        FROM hard_negative_queue
        WHERE elder_id = ?
          AND created_at >= ?
    """
    if room:
        query += " AND room = ?"
        params.append(normalize_room_token(room))
    if status:
        query += " AND status = ?"
        params.append(str(status))
    query += " ORDER BY score DESC, created_at DESC LIMIT ?"
    params.append(int(limit))
    cursor = conn.execute(query, tuple(params))
    rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame()
    columns = [d[0] for d in cursor.description] if cursor.description else []
    try:
        return pd.DataFrame(rows, columns=columns)
    except Exception:
        return pd.DataFrame([dict(r) for r in rows])


def mark_hard_negative_status(
    conn,
    queue_ids: Iterable[int],
    status: str,
) -> int:
    ensure_hard_negative_table(conn)
    ids = [int(x) for x in queue_ids if x is not None]
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    params: List[Any] = [str(status)] + ids
    conn.execute(
        f"""
        UPDATE hard_negative_queue
        SET status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id IN ({placeholders})
        """,
        tuple(params),
    )
    return len(ids)


def mark_hard_negative_applied(
    conn,
    elder_id: str,
    room: str,
    timestamp_start: str,
    timestamp_end: str,
) -> int:
    ensure_hard_negative_table(conn)
    room_key = normalize_room_token(room)
    cursor = conn.execute(
        """
        UPDATE hard_negative_queue
        SET status = 'applied', updated_at = CURRENT_TIMESTAMP
        WHERE elder_id = ?
          AND room = ?
          AND timestamp_start <= ?
          AND timestamp_end >= ?
          AND status IN ('open', 'queued', 'reviewed')
        """,
        (elder_id, room_key, str(timestamp_end), str(timestamp_start)),
    )
    try:
        return int(cursor.rowcount or 0)
    except Exception:
        return 0
