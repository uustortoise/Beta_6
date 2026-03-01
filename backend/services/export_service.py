import io
import logging
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

from services.db_utils import get_dashboard_connection, query_df

logger = logging.getLogger(__name__)


def get_residents() -> list[str]:
    """Fetch resident IDs from operational tables with model-dir fallback."""
    resident_ids: set[str] = set()
    queries = [
        ("SELECT elder_id FROM elders", "elder_id"),
        ("SELECT DISTINCT elder_id FROM adl_history", "elder_id"),
        ("SELECT DISTINCT elder_id FROM correction_history", "elder_id"),
        ("SELECT DISTINCT elder_id FROM training_history", "elder_id"),
        ("SELECT DISTINCT resident_id AS elder_id FROM predictions", "elder_id"),
    ]
    for sql, col in queries:
        try:
            df = query_df(sql)
        except Exception:
            continue
        if df is None or df.empty or col not in df.columns:
            continue
        for value in df[col].tolist():
            resident = str(value or "").strip()
            if resident:
                resident_ids.add(resident)

    models_root = Path(__file__).resolve().parent.parent / "models"
    if models_root.exists():
        for p in models_root.iterdir():
            if not p.is_dir():
                continue
            name = p.name.strip()
            if name and name.lower() != "context":
                resident_ids.add(name)

    return sorted(resident_ids)


def get_db_bounds(elder_id: str) -> tuple[datetime | None, datetime | None]:
    """Get earliest and latest timestamps for a resident in adl_history."""
    try:
        with get_dashboard_connection() as conn:
            query = "SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM adl_history WHERE elder_id = ?"
            cursor = conn.cursor()
            cursor.execute(query, (elder_id,))
            row = cursor.fetchone()
            if not row or row[0] is None:
                return None, None

            def parse_date(val):
                if isinstance(val, str):
                    try:
                        return datetime.fromisoformat(val[:19].replace("Z", ""))
                    except ValueError:
                        return None
                return val

            return parse_date(row[0]), parse_date(row[1])
    except Exception as e:
        logger.error(f"Failed to fetch date bounds for {elder_id}: {e}")
        return None, None


def export_raw_adl(elder_id: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Export raw completed ADL segments from adl_history."""
    start_ts = datetime.combine(start_date, time.min)
    end_ts = datetime.combine(end_date, time.max)
    query = """
        SELECT timestamp, room, activity_type, confidence, duration_minutes, is_corrected
        FROM adl_history
        WHERE elder_id=? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    """
    return query_df(query, (elder_id, start_ts, end_ts))


def export_review_candidates(elder_id: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Export segments flagged as low confidence anomalies for manual review."""
    start_ts = datetime.combine(start_date, time.min)
    end_ts = datetime.combine(end_date, time.max)
    # Replicating original export logic (usually is_anomaly=1 or low_confidence explicitly)
    query = """
        SELECT timestamp, room, activity_type, confidence, duration_minutes
        FROM adl_history
        WHERE elder_id=? AND timestamp BETWEEN ? AND ?
          AND (activity_type = 'low_confidence' OR is_anomaly = 1)
        ORDER BY timestamp ASC
    """
    return query_df(query, (elder_id, start_ts, end_ts))


def export_predicted_results(elder_id: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Export raw predicted results + confidence from predictions table.
    This bypasses post-processing bounds and exposes true raw predictions.
    """
    start_ts = datetime.combine(start_date, time.min)
    end_ts = datetime.combine(end_date, time.max)
    query = """
        SELECT timestamp, room, activity as predicted_activity, confidence, is_anomaly
        FROM predictions
        WHERE resident_id=? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    """
    return query_df(query, (elder_id, start_ts, end_ts))


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert dataframe directly to Excel bytes payload."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Data Export")
    return output.getvalue()
