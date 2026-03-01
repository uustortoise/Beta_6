import logging
import sqlite3
import re
from contextlib import contextmanager
from typing import Generator, Any
from backend.db.database import db as dual_write_db
from backend.elderlycare_v1_16.config.settings import POSTGRES_ONLY


logger = logging.getLogger(__name__)

class DualWriteCursorProxy:
    """Proxies cursor calls to enable dual-write."""
    
    def __init__(self, sqlite_cursor, db_proxy=None):
        self.sqlite_cursor = sqlite_cursor
        self.db_proxy = db_proxy

    def execute(self, sql: str, params=()) -> Any:
        # 1. Write to SQLite (Single Source of Truth)
        try:
            res = self.sqlite_cursor.execute(sql, params)
        except Exception as e:
            logger.error(f"SQLite execute failed: {e}")
            raise e

        # 2. Write to Postgres (Best Effort) if it's a modification
        # Simple heuristic: if it starts with INSERT/UPDATE/DELETE
        # Only check pg_db property if we might write (lazy check)
        if sql.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            pg = self.db_proxy.pg_db if self.db_proxy else None
            if pg:
                try:
                    pg.execute(sql, params)
                except Exception as e:
                    logger.error(f"Dual-write to PostgreSQL failed: {e}")
                    # Suppress error to keep legacy flow intact
        
        return res

    def executemany(self, sql: str, params_list: Any) -> Any:
        # 1. SQLite
        res = self.sqlite_cursor.executemany(sql, params_list)
        
        # 2. Postgres
        if sql.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            pg = self.db_proxy.pg_db if self.db_proxy else None
            if pg:
                try:
                    pg.executemany(sql, params_list)
                except Exception as e:
                     logger.error(f"Dual-write batch to PostgreSQL failed: {e}")
        
        return res

    def fetchone(self):
        return self.sqlite_cursor.fetchone()

    def fetchall(self):
        return self.sqlite_cursor.fetchall()
        
    def close(self):
        self.sqlite_cursor.close()
        
    def __getattr__(self, name):
        # Delegate other methods (description, rowcount, etc.) to SQLite cursor
        return getattr(self.sqlite_cursor, name)


class DualWriteConnectionProxy:
    """Proxies connection calls."""
    
    def __init__(self, sqlite_conn, db_proxy):
        self.sqlite_conn = sqlite_conn
        self.db_proxy = db_proxy
        self.row_factory = None # Emulate attribute

    def cursor(self):
        return DualWriteCursorProxy(self.sqlite_conn.cursor(), self.db_proxy)

    def execute(self, sql: str, params=()) -> Any:
        # Convenience method on connection
        cursor = self.cursor()
        res = cursor.execute(sql, params)
        return cursor # sqlite3 connection.execute returns cursor

    def executescript(self, sql_script: str) -> Any:
        # Only execute scripts on SQLite (Postgres schema is managed separately via schema.sql)
        # Sanitize for SQLite: skip PG-only commands, replace PG-only types
        sanitized = []
        # Simple split by semicolon
        for cmd in sql_script.split(';'):
            cmd_strip = cmd.strip()
            if not cmd_strip: continue
            
            # Remove comments for keyword checking to avoid skipping tables with "Hypertable" in comments
            # This handles both -- and /* */ comments simply
            clean_cmd = re.sub(r'--.*$', '', cmd_strip, flags=re.MULTILINE)
            clean_cmd = re.sub(r'/\*.*?\*/', '', clean_cmd, flags=re.DOTALL).strip()
            
            if not clean_cmd:
                # Command was just a comment, keep it as is (or skip)
                sanitized.append(cmd_strip)
                continue

            # Commands to SKIP entirely (Procedural or PG-Extensions)
            pg_skip_cmds = ["EXTENSION", "HYPERTABLE", "plpgsql", "DO $$", "CREATE OR REPLACE FUNCTION", "CREATE CAST", "create_hypertable"]
            if any(pg.upper() in clean_cmd.upper() for pg in pg_skip_cmds):
                logger.debug(f"DualWrite: Skipping PG-only command: {clean_cmd[:50]}...")
                continue
                
            # Type and keyword REPLACEMENTS (Keep the command, change the dialect)
            cmd_sqlite = cmd_strip
            # GIN indexes are Postgres-only
            if "USING GIN" in clean_cmd.upper():
                continue
                
            # Type Mapping
            cmd_sqlite = cmd_sqlite.replace("SERIAL", "INTEGER") 
            cmd_sqlite = cmd_sqlite.replace("JSONB", "JSON")
            cmd_sqlite = cmd_sqlite.replace("TIMESTAMP WITH TIME ZONE", "TIMESTAMP")
            cmd_sqlite = cmd_sqlite.replace("DOUBLE PRECISION", "REAL")
            cmd_sqlite = cmd_sqlite.replace("NOW()", "CURRENT_TIMESTAMP")
            
            sanitized.append(cmd_sqlite)
            
        final_script = ';'.join(sanitized)
        logger.debug(f"DualWrite: Executing sanitized script on SQLite: {final_script[:500]}...")
        try:
            return self.sqlite_conn.executescript(final_script)
        except Exception as e:
            logger.error(f"DualWrite: SQLite executescript failed: {e}")
            # Log the full script to a temp file for debugging
            with open("backend/logs/failed_sqlite_schema.sql", "w") as f:
                f.write(final_script)
            raise

    def commit(self):
        self.sqlite_conn.commit()
        # Postgres auto-commits in the helper class, so nothing to do here
    
    def close(self):
        self.sqlite_conn.close()
        
    def __getattr__(self, name):
        return getattr(self.sqlite_conn, name)


class PostgresCursorShim:
    """Adapts a psycopg2 cursor to look like a SQLite cursor."""
    def __init__(self, pg_cursor):
        self.pg_cursor = pg_cursor

    def __iter__(self):
        return iter(self.pg_cursor)

    def __iter__(self):
        return iter(self.pg_cursor)

    @staticmethod
    def _normalize_params(params):
        if params is None:
            return None
        if isinstance(params, bool):
            return int(params)
        if isinstance(params, tuple):
            return tuple(int(p) if isinstance(p, bool) else p for p in params)
        if isinstance(params, list):
            return [int(p) if isinstance(p, bool) else p for p in params]
        return params

    def _adapt_sql(self, sql):
        # Convert SQLite placeholders to psycopg2 placeholders.
        sql = sql.replace('?', '%s')

        # Replace INSERT OR IGNORE
        if "INSERT OR IGNORE" in sql.upper():
            sql = sql.replace("INSERT OR IGNORE", "INSERT")
            if "ON CONFLICT" not in sql.upper():
                 sql += " ON CONFLICT DO NOTHING"
        
        # Replace INSERT OR REPLACE -> INSERT ... ON CONFLICT DO UPDATE
        # SQLite's REPLACE is equivalent to upsert on PRIMARY KEY conflict
        if "INSERT OR REPLACE" in sql.upper():
            sql = sql.replace("INSERT OR REPLACE", "INSERT")
            # Extract table name and columns for ON CONFLICT clause
            # Pattern: INSERT INTO table (col1, col2, ...) VALUES (...)
            match = re.search(r'INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)', sql, re.IGNORECASE)
            if match:
                table_name = match.group(1).lower()
                columns_str = match.group(2)
                columns = [c.strip() for c in columns_str.split(',')]
                
                # Determine conflict column (usually first column or known PK)
                # These tables have known primary keys
                pk_map = {
                    'household_config': 'key',
                    'activity_segments': ['elder_id', 'room', 'start_time'],
                    'predictions': ['resident_id', 'timestamp', 'room'],
                    'medical_history': ['elder_id', 'category'],
                    'sleep_analysis': ['elder_id', 'analysis_date']
                }
                
                pk = pk_map.get(table_name, columns[0])
                if isinstance(pk, list):
                    pk_clause = ', '.join(pk)
                else:
                    pk_clause = pk
                
                # Build UPDATE SET clause (all non-PK columns)
                pk_set = set(pk) if isinstance(pk, list) else {pk}
                update_cols = [c for c in columns if c.strip().lower() not in pk_set]
                if update_cols:
                    update_clause = ', '.join([f"{c.strip()} = EXCLUDED.{c.strip()}" for c in update_cols])
                    sql = sql.rstrip(';').rstrip() + f" ON CONFLICT ({pk_clause}) DO UPDATE SET {update_clause}"
                else:
                    sql = sql.rstrip(';').rstrip() + f" ON CONFLICT ({pk_clause}) DO NOTHING"
        
        if "datetime('now')" in sql:
             sql = sql.replace("datetime('now')", "CURRENT_TIMESTAMP")
        if 'datetime("now")' in sql:
             sql = sql.replace('datetime("now")', "CURRENT_TIMESTAMP")
        # SQLite date('now') → PostgreSQL CURRENT_DATE
        if "date('now')" in sql:
            sql = sql.replace("date('now')", "CURRENT_DATE")
        if 'date("now")' in sql:
            sql = sql.replace('date("now")', "CURRENT_DATE")
        
        # SQLite date(column) -> PostgreSQL column::date (simple cases)
        sql = re.sub(r"date\(([\w\.]+)\)", r"\1::date", sql)
        return sql

    def execute(self, sql, params=()):
        sql = self._adapt_sql(sql)
        params = self._normalize_params(params)
        self.pg_cursor.execute(sql, params)
        return self

    def executemany(self, sql, params_list):
        sql = self._adapt_sql(sql)
        if params_list is not None:
            params_list = [self._normalize_params(row) for row in params_list]
        self.pg_cursor.executemany(sql, params_list)
        return self

    def fetchone(self):
        return self.pg_cursor.fetchone()

    def fetchall(self):
        return self.pg_cursor.fetchall()

    def close(self):
        self.pg_cursor.close()

    def __getattr__(self, name):
        return getattr(self.pg_cursor, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PostgresConnectionShim:
    """Adapts a psycopg2 connection to look like a sqlite3 connection."""
    def __init__(self, pg_conn, db_proxy):
        self.pg_conn = pg_conn
        self.db_proxy = db_proxy
        self.row_factory = None  # Ignored for Postgres (simulated via DictCursor if needed)

    def cursor(self):
        return PostgresCursorShim(self.pg_conn.cursor())

    def execute(self, sql, params=()):
        cursor = self.cursor()
        cursor.execute(sql, params)
        return cursor

    def executemany(self, sql, params_list):
        cursor = self.cursor()
        cursor.executemany(sql, params_list)
        return cursor

    def executescript(self, sql_script):
        # Postgres supports executing multiple statements
        with self.cursor() as cur:
            cur.execute(sql_script)

    def commit(self):
        self.pg_conn.commit()

    def rollback(self):
        self.pg_conn.rollback()

    def close(self):
        # Return to pool instead of closing
        self.db_proxy.pg_db.return_connection(self.pg_conn)

    def __getattr__(self, name):
        return getattr(self.pg_conn, name)


class LegacyDatabaseAdapter:
    """
    Drop-in replacement for elderlycare_v1_16.database.DatabaseManager.
    Wraps the new backend.db.database.db (DualWriteDatabase).
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LegacyDatabaseAdapter, cls).__new__(cls)
        return cls._instance

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        # Get raw connection
        conn = dual_write_db.get_connection()
        
        if POSTGRES_ONLY:
            # Wrap in Postgres Shim
            proxy = PostgresConnectionShim(conn, dual_write_db)
        else:
            conn.row_factory = sqlite3.Row # Legacy expects Row factory
            # Wrap in DualWrite Proxy
            proxy = DualWriteConnectionProxy(conn, dual_write_db)
        
        try:
            yield proxy
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if POSTGRES_ONLY:
                dual_write_db.pg_db.return_connection(conn)
            else:
                conn.close()

    def init_schema(self, schema_sql: str):
        """Initializes the database schema using a SQL script."""
        logger.info("LegacyAdapter: Initializing schema...")
        with self.get_connection() as conn:
            conn.executescript(schema_sql)

# RE-CHECKING implementation of SQLiteDatabase
# It def _get_connection(self): return sqlite3.connect(...)
# So we own the connection here.
