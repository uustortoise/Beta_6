import logging
from datetime import datetime, timedelta

import pandas as pd

from services.db_utils import get_dashboard_connection, parse_json_object, query_df
from utils.correction_eval_history import fetch_and_enrich_correction_evaluations

logger = logging.getLogger(__name__)
ROOM_MATCH_SQL = "LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = LOWER(REPLACE(REPLACE(?, ' ', ''), '_', ''))"


def _ensure_correction_detail_table(conn) -> None:
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


def fetch_correction_trail(
    elder_id: str | None = None,
    room: str | None = None,
    corrected_by: str | None = None,
    days: int = 3650,
    include_deleted: bool = False,
) -> pd.DataFrame:
    """Fetch correction history with optional filters."""
    query = """
        SELECT 
            id, elder_id, room, timestamp_start, timestamp_end, 
            old_activity, new_activity, rows_affected,
            corrected_by, corrected_at, is_deleted
        FROM correction_history
        WHERE corrected_at >= ?
    """
    cutoff = datetime.now() - timedelta(days=days)
    params = [cutoff]

    if not include_deleted:
        query += " AND is_deleted = 0"
    if elder_id and elder_id != "All":
        query += " AND elder_id = ?"
        params.append(elder_id)
    if room and room != "All":
        query += " AND " + ROOM_MATCH_SQL
        params.append(room)
    if corrected_by and corrected_by != "All":
        query += " AND corrected_by = ?"
        params.append(corrected_by)

    query += " ORDER BY corrected_at DESC"

    df = query_df(query, tuple(params))
    
    # Format durations for UI display if needed
    if not df.empty:
        df["duration"] = pd.to_datetime(df["timestamp_end"]) - pd.to_datetime(df["timestamp_start"])
        # Format as MM:SS
        df["duration_str"] = df["duration"].dt.components.apply(
            lambda x: f"{int(x.minutes):02d}:{int(x.seconds):02d}", axis=1
        )
    return df


def rollback_correction(correction_id: int, rolled_back_by: str = "system") -> tuple[bool, str]:
    """
    Soft-delete a correction from history and revert the adl_history rows.
    Uses explicit transaction control so partial failure does not leave data inconsistent.
    """
    try:
        with get_dashboard_connection() as conn:
            cursor = conn.cursor()
            _ensure_correction_detail_table(conn)
            
            # 1. Fetch the correction details
            cursor.execute(
                "SELECT elder_id, room, timestamp_start, timestamp_end, old_activity "
                "FROM correction_history WHERE id = ? AND is_deleted = 0",
                (correction_id,)
            )
            row = cursor.fetchone()
            if not row:
                return False, "Correction not found or already deleted."
                
            elder_id, room, ts_start, ts_end, old_activity = row
            
            # 2. Soft delete the audit record
            cursor.execute(
                "UPDATE correction_history SET is_deleted = 1, deleted_at = ?, deleted_by = ? WHERE id = ?",
                (datetime.now(), rolled_back_by, correction_id)
            )

            # 3. Restore exact rows first (preferred, if detail snapshots exist)
            cursor.execute(
                """
                SELECT timestamp, old_activity
                FROM correction_history_detail
                WHERE correction_id = ?
                ORDER BY timestamp ASC
                """,
                (correction_id,),
            )
            detail_rows = cursor.fetchall() or []
            restored_rows = 0
            if detail_rows:
                for ts_value, old_value in detail_rows:
                    if not old_value:
                        continue
                    cursor.execute(
                        """
                        UPDATE adl_history
                        SET activity_type = ?, is_corrected = 0
                        WHERE elder_id = ? AND """ + ROOM_MATCH_SQL + """ AND timestamp = ?
                        """,
                        (old_value, elder_id, room, ts_value),
                    )
                    restored_rows += int(cursor.rowcount or 0)
            elif old_activity:
                # Legacy fallback for older correction records without detail snapshots.
                cursor.execute(
                    """
                    UPDATE adl_history 
                    SET activity_type = ?, is_corrected = 0 
                    WHERE elder_id = ? AND """ + ROOM_MATCH_SQL + """ AND timestamp >= ? AND timestamp < ?
                    """,
                    (old_activity, elder_id, room, ts_start, ts_end)
                )
                restored_rows = int(cursor.rowcount or 0)
            
            conn.commit()
            return True, f"Successfully rolled back correction #{correction_id} ({restored_rows} rows restored)."
        
    except Exception as e:
        # The context manager will handle connection cleanup;
        # any uncommitted changes are automatically rolled back on exception
        logger.exception(f"Failed to rollback correction {correction_id}")
        return False, f"Database error: {e}"


def fetch_evaluation_history(elder_id: str | None = None, days: int = 3650) -> pd.DataFrame:
    """Fetch correction retrain evaluation rows from training_history."""
    try:
        df = fetch_and_enrich_correction_evaluations(
            query_fn=query_df,
            elder_filter=elder_id if elder_id != "All" else None,
            days=days,
        )
        return df
    except Exception as e:
        logger.error(f"Failed to fetch evaluation history: {e}")
        return pd.DataFrame()
