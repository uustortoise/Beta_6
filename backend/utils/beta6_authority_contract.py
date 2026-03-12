"""Shared Beta 6 authority contract helpers."""

from __future__ import annotations

import os
from typing import Any

try:
    from backend.elderlycare_v1_16.config.settings import USE_POSTGRESQL
    from backend.db.database import db as dual_write_db
except Exception:
    from elderlycare_v1_16.config.settings import USE_POSTGRESQL
    from db.database import db as dual_write_db


_TRUTHY_VALUES = {"1", "true", "yes", "y", "on", "enabled"}


def is_truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    return str(value).strip().lower() in _TRUTHY_VALUES


def env_truthy(name: str, default: bool = False) -> bool:
    return is_truthy(os.getenv(name), default=default)


def check_postgresql_preflight() -> tuple[bool, dict[str, Any]]:
    conn = None
    pg_db = None
    try:
        if not bool(USE_POSTGRESQL):
            return False, {"error": "USE_POSTGRESQL=false"}

        pg_db = getattr(dual_write_db, "pg_db", None)
        if pg_db is None:
            return False, {"error": "PostgreSQL unavailable or failed to initialize"}

        conn = pg_db.get_raw_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        return True, {"status": "ok"}
    except Exception as exc:
        return False, {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        if pg_db is not None and conn is not None:
            try:
                pg_db.return_connection(conn)
            except Exception:
                pass
