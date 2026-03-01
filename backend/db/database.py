import os
import sqlite3
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional, Dict
import threading
import re
import time
from contextlib import contextmanager

from backend.elderlycare_v1_16.config.settings import DB_PATH, USE_POSTGRESQL, POSTGRES_ONLY, POSTGRES_CONFIG

logger = logging.getLogger(__name__)

class Database(ABC):
    """Abstract Base Class for Database Interactions."""

    @abstractmethod
    def execute(self, sql: str, params: Tuple = ()) -> Any:
        pass

    @abstractmethod
    def fetchall(self, sql: str, params: Tuple = ()) -> List[Tuple]:
        pass
    
    @abstractmethod
    def fetchone(self, sql: str, params: Tuple = ()) -> Optional[Tuple]:
        pass

    @abstractmethod
    def executemany(self, sql: str, params_list: List[Tuple]) -> Any:
        pass


class SQLiteDatabase(Database):
    """SQLite Implementation."""
    
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path

    def _get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def execute(self, sql: str, params: Tuple = ()) -> Any:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            return cursor.lastrowid

    def fetchall(self, sql: str, params: Tuple = ()) -> List[Tuple]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchall()
            
    def fetchone(self, sql: str, params: Tuple = ()) -> Optional[Tuple]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchone()

    def executemany(self, sql: str, params_list: List[Tuple]) -> Any:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, params_list)
            conn.commit()


class PostgreSQLDatabase(Database):
    """PostgreSQL Implementation with Connection Pooling."""

    _pool = None
    _lock = threading.Lock()

    def __init__(self, config: Dict = POSTGRES_CONFIG):
        self.config = config
        self._ensure_pool()

    def _ensure_pool(self):
        if PostgreSQLDatabase._pool is None:
            with PostgreSQLDatabase._lock:
                if PostgreSQLDatabase._pool is None:
                    try:
                        import psycopg2.pool
                        # psycopg2 is imported here to avoid hard dependency if not using Postgres
                    except ImportError:
                        logger.error("psycopg2 module not found. Please install it with 'pip install psycopg2-binary'")
                        raise

                    try:
                        PostgreSQLDatabase._pool = psycopg2.pool.ThreadedConnectionPool(
                            minconn=self.config['minconn'],
                            maxconn=self.config['maxconn'],
                            host=self.config['host'],
                            port=self.config['port'],
                            user=self.config['user'],
                            password=self.config['password'],
                            dbname=self.config['dbname']
                        )
                        logger.info("PostgreSQL Connection Pool initialized.")
                    except Exception as e:
                         logger.error(f"Failed to initialize PostgreSQL pool: {e}")
                         raise

    @contextmanager
    def _get_cursor(self):
        conn = PostgreSQLDatabase._pool.getconn()
        try:
            with conn.cursor() as cur:
                yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            PostgreSQLDatabase._pool.putconn(conn)

    def get_raw_connection(self):
        """Get a raw connection from the pool for manual management."""
        self._ensure_pool()
        return PostgreSQLDatabase._pool.getconn()
    
    def return_connection(self, conn):
        """Return a raw connection to the pool."""
        if PostgreSQLDatabase._pool:
            PostgreSQLDatabase._pool.putconn(conn)

    def _adapt_sql(self, sql: str) -> str:
        # 1. Convert SQLite '?' placeholder to PostgreSQL '%s'
        sql = sql.replace('?', '%s')
        
        # 2. Handle SQLite 'INSERT OR IGNORE' -> Postgres 'ON CONFLICT DO NOTHING'
        # Regex to find "INSERT OR IGNORE INTO table ..."
        # We start by removing "OR IGNORE" from the INSERT clause
        if "INSERT OR IGNORE" in sql.upper():
            # Remove "OR IGNORE" to make it a standard INSERT
            sql = re.sub(r'INSERT\s+OR\s+IGNORE\s+INTO', 'INSERT INTO', sql, flags=re.IGNORECASE)
            
            # Append "ON CONFLICT DO NOTHING" if it's not already there
            if "ON CONFLICT" not in sql.upper():
                sql = sql.strip()
                if sql.endswith(';'):
                    sql = sql[:-1]
                sql += " ON CONFLICT DO NOTHING"

        return sql

    def _normalize_params(self, params: Any) -> Any:
        """
        Recursively convert Python booleans to integers (True->1, False->0).
        This ensures compatibility with the INTEGER schema in PostgreSQL, 
        regardless of whether the app sends 0/1 or True/False.
        """
        if params is None:
            return None
        if isinstance(params, bool):
            return int(params)
        if isinstance(params, (list, tuple)):
            # Recursive handling for nested structures (though params are usually flat)
            return type(params)(self._normalize_params(x) for x in params)
        return params

    def execute(self, sql: str, params: Tuple = ()) -> Any:
        sql = self._adapt_sql(sql)
        # Normalize params: bool -> int
        params = self._normalize_params(params)
        
        # 1. Auto-append RETURNING id for INSERTs (Simplified & Robust)
        # Avoid redundant checks or double-appending
        # NOTE: 'elders' table uses 'elder_id' text PK and has no 'id' column.
        is_insert = sql.strip().upper().startswith('INSERT')
        has_returning = 'RETURNING' in sql.upper()
        
        # Regex to detect 'elders' table robustly (handles quotes and spacing)
        # Matches: INTO elders, INTO "elders", INTO  elders
        is_elders_table = bool(re.search(r'into\s+"?elders"?\b', sql, re.IGNORECASE))
        
        if is_insert and not has_returning and not is_elders_table:
             sql += " RETURNING id"

        with self._get_cursor() as cur:
            try:
                cur.execute(sql, params)
                
                # 2. Fetch result if RETURNING matches
                if 'RETURNING' in sql.upper():
                    row = cur.fetchone()
                    return row[0] if row else None
                return None
            except Exception as e:
                # Basic Error Logging for visibility
                logger.error(f"PostgreSQL Execute Failed: {e} | SQL: {sql[:100]}...")
                raise e 

    def fetchall(self, sql: str, params: Tuple = ()) -> List[Tuple]:
        sql = self._adapt_sql(sql)
        params = self._normalize_params(params)
        with self._get_cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def fetchone(self, sql: str, params: Tuple = ()) -> Optional[Tuple]:
        sql = self._adapt_sql(sql)
        params = self._normalize_params(params)
        with self._get_cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    def executemany(self, sql: str, params_list: List[Tuple]) -> Any:
        sql = self._adapt_sql(sql)
        # Normalize all params in the list
        params_list = [self._normalize_params(p) for p in params_list]
        with self._get_cursor() as cur:
            cur.executemany(sql, params_list)


class DualWriteDatabase(Database):
    """Proxy that writes to both SQLite and PostgreSQL (if enabled).
    
    When POSTGRES_ONLY=True, SQLite is completely bypassed and PostgreSQL
    becomes the sole database. This is recommended for local debugging.
    """

    def __init__(self):
        self._sqlite_db = None if POSTGRES_ONLY else SQLiteDatabase()
        self._pg_db = None  # Internal storage for lazy init
        
    @property
    def pg_db(self):
        """Lazy initialization of PostgreSQL connection."""
        if not USE_POSTGRESQL:
            return None
            
        if self._pg_db is None:
            try:
                self._pg_db = PostgreSQLDatabase()
            except Exception as e:
                if POSTGRES_ONLY:
                    logger.error(f"PostgreSQL is required (POSTGRES_ONLY=true) but could not be initialized: {e}")
                    raise
                logger.error(f"Failed to initialize PostgreSQL connection: {e}. Falling back to SQLite-only mode.")
                self._pg_db = None
        
        return self._pg_db

    @property
    def sqlite_db(self):
        """Access SQLite database (None if POSTGRES_ONLY mode)."""
        return self._sqlite_db

    def execute(self, sql: str, params: Tuple = ()) -> Any:
        if POSTGRES_ONLY:
            # PostgreSQL is the sole database
            return self.pg_db.execute(sql, params)
        
        # Dual-Write Mode: Primary Write to SQLite (Source of Truth)
        res = self.sqlite_db.execute(sql, params)

        # Secondary Write: PostgreSQL (Best Effort)
        if not sql.strip().upper().startswith('SELECT'):
            pg = self.pg_db
            if pg:
                start_time = time.time()
                try:
                    pg.execute(sql, params)
                    duration_ms = (time.time() - start_time) * 1000
                    logger.info(f"Dual-write SUCCESS | Latency: {duration_ms:.2f}ms | SQL: {sql[:50]}...")
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    logger.error(f"Dual-write FAILURE | Latency: {duration_ms:.2f}ms | Error: {e}")
        
        return res

    def fetchall(self, sql: str, params: Tuple = ()) -> List[Tuple]:
        if POSTGRES_ONLY:
            return self.pg_db.fetchall(sql, params)
        return self.sqlite_db.fetchall(sql, params)

    def fetchone(self, sql: str, params: Tuple = ()) -> Optional[Tuple]:
        if POSTGRES_ONLY:
            return self.pg_db.fetchone(sql, params)
        return self.sqlite_db.fetchone(sql, params)

    def executemany(self, sql: str, params_list: List[Tuple]) -> Any:
        if POSTGRES_ONLY:
            return self.pg_db.executemany(sql, params_list)
        
        res = self.sqlite_db.executemany(sql, params_list)
        
        pg = self.pg_db
        if pg:
            try:
                self.pg_db.executemany(sql, params_list)
            except Exception as e:
                logger.error(f"Dual-write batch to PostgreSQL failed: {e}")
        
        return res

# Lazy Initialization Singleton
# The original eager init (`db = DualWriteDatabase()`) was creating the PostgreSQL
# ThreadedConnectionPool at import time, which conflicts with TensorFlow's threading model.
# This lazy proxy delays DB initialization until first use.
class _LazyDatabaseProxy:
    """Proxy that lazily initializes the database connection on first use."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._db = None
    
    def _get_db(self):
        if self._db is None:
            with self._lock:
                if self._db is None:
                    self._db = DualWriteDatabase()
        return self._db
    
    def execute(self, sql, params=()):
        return self._get_db().execute(sql, params)
    
    def fetchall(self, sql, params=()):
        return self._get_db().fetchall(sql, params)
    
    def fetchone(self, sql, params=()):
        return self._get_db().fetchone(sql, params)
    
    def executemany(self, sql, params_list):
        return self._get_db().executemany(sql, params_list)
    
    def get_connection(self):
        """Provide access to underlying database connection.
        
        In POSTGRES_ONLY mode, this returns a PostgreSQL connection.
        Otherwise, returns SQLite connection for legacy compatibility.
        """
        if POSTGRES_ONLY:
            return self._get_db().pg_db.get_raw_connection()
        return self._get_db().sqlite_db._get_connection()
    
    @property
    def sqlite_db(self):
        """Lazy access to the underlying SQLite database instance.
        
        Returns None in POSTGRES_ONLY mode.
        """
        return self._get_db().sqlite_db
    
    @property
    def pg_db(self):
        """Lazy access to the underlying PostgreSQL database instance."""
        return self._get_db().pg_db

# Singleton Instance (Lazy)
# Usage: from backend.db.database import db
db = _LazyDatabaseProxy()
