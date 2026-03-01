"""
Shared database connection utility for intelligence modules.

This provides a consistent way for all intelligence-related modules
(watchdog, classifier, trajectory_engine, etc.) to access the database
using the LegacyDatabaseAdapter for PostgreSQL compatibility.
"""

import logging
from typing import Any, Iterable

import pandas as pd

try:
    from backend.db.legacy_adapter import LegacyDatabaseAdapter
except ImportError:
    from elderlycare_v1_16.database import db as _db_adapter
else:
    _db_adapter = LegacyDatabaseAdapter()

logger = logging.getLogger(__name__)


def get_intelligence_db():
    """
    Get database connection for intelligence modules.
    
    Returns a context manager that yields a database connection.
    Uses LegacyDatabaseAdapter for PostgreSQL/SQLite compatibility.
    
    Usage:
        with get_intelligence_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM adl_history WHERE ...")
    """
    return _db_adapter.get_connection()


def get_adapter():
    """Get the underlying database adapter instance."""
    return _db_adapter


def query_to_dataframe(conn: Any, query: str, params: Iterable[Any] = ()) -> pd.DataFrame:
    """
    Execute SQL using DB-API cursor and return a DataFrame.
    Avoids pandas DBAPI warnings with adapter/shim connections.
    """
    cursor = conn.cursor()
    cursor.execute(query, tuple(params) if params is not None else ())
    columns = [col[0] for col in (cursor.description or [])]
    rows = cursor.fetchall()
    if not columns:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows, columns=columns)


def coerce_timestamp_column(
    df: pd.DataFrame,
    column: str = "timestamp",
    context: str = "dataset",
) -> pd.DataFrame:
    """
    Parse timestamp column safely and drop invalid rows with warning logs.
    """
    if df.empty:
        return df
    if column not in df.columns:
        logger.warning("%s is missing required timestamp column '%s'.", context, column)
        return df.iloc[0:0].copy()

    parsed = pd.to_datetime(df[column], errors="coerce")
    invalid_count = int(parsed.isna().sum())
    cleaned = df.copy()
    cleaned[column] = parsed
    if invalid_count:
        logger.warning(
            "%s dropped %d row(s) with invalid %s values.",
            context,
            invalid_count,
            column,
        )
    return cleaned.dropna(subset=[column])
