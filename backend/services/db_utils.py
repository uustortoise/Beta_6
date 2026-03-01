import json
import logging
from typing import Any, Dict

from elderlycare_v1_16.config.settings import DB_PATH

try:
    from backend.utils.intelligence_db import query_to_dataframe
except Exception:
    from utils.intelligence_db import query_to_dataframe

logger = logging.getLogger(__name__)

try:
    from backend.db.legacy_adapter import LegacyDatabaseAdapter
    _adapter = LegacyDatabaseAdapter()
except ImportError:
    import sqlite3
    _adapter = None


def get_dashboard_connection():
    """Get database connection compatible with PostgreSQL or SQLite."""
    if _adapter:
        return _adapter.get_connection()
    else:
        # Fallback for standalone usage
        import sqlite3
        return sqlite3.connect(DB_PATH)


def query_df(query: str, params: tuple = ()):
    """Run a query and return a pandas DataFrame."""
    with get_dashboard_connection() as conn:
        return query_to_dataframe(conn, query, params)


def parse_json_object(raw_value: Any) -> Dict[str, Any]:
    """Parse JSON payload from DB rows into a dict."""
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return {}
        try:
            parsed = json.loads(raw_value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def coerce_bool(raw_value: Any) -> bool:
    """Coerce serialized bool-like values from DB JSON."""
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return False
    if isinstance(raw_value, (int, float)):
        return raw_value != 0
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {'1', 'true', 'yes', 'y'}
    return bool(raw_value)
