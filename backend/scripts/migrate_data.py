import sqlite3
import psycopg2
import logging
import sys
import os
from tqdm import tqdm
from datetime import datetime

# Adjust path to find backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.elderlycare_v1_16.config.settings import DB_PATH, POSTGRES_CONFIG

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("migration.log")
    ]
)
logger = logging.getLogger(__name__)

# Table Order (Respects Foreign Keys)
TABLES = [
    'elders',              # 1. Base table
    'emergency_contacts',
    'medical_history',
    'icope_assessments',
    'sleep_analysis',
    'alerts',
    'alert_rules',
    'alert_rules_v2',
    'routine_anomalies', 
    'trajectory_events',
    'activity_segments',
    'household_behavior',  # Hypertable
    'household_segments',  # No FK
    'household_config',    # No FK
    'predictions',         # No FK in schema
    'sensor_data',         # Hypertable
    'adl_history',         # Hypertable - Largest table, do last
    'correction_history',  # FK to elders
    'training_history',
    'model_training_history',
    'context_episodes'
]

def get_columns(cursor, table_name):
    """Get column names from SQLite table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]

def migrate_table(sqlite_conn, pg_conn, table_name):
    """Migrate a single table."""
    logger.info(f"Migrating table: {table_name}")
    
    # 1. Get source data
    sqlite_cur = sqlite_conn.cursor()
    
    # Get columns FIRST to avoid resetting cursor
    try:
        columns = get_columns(sqlite_conn.cursor(), table_name) # Use new cursor for metadata
    except Exception:
        columns = []

    if not columns:
         logger.warning(f"Table {table_name} not found or empty schema in SQLite. Skipping.")
         return

    try:
        sqlite_cur.execute(f"SELECT * FROM {table_name}")
    except sqlite3.OperationalError:
        logger.warning(f"Table {table_name} query failed. Skipping.")
        return

    rows = sqlite_cur.fetchall()
    
    if not rows:
        logger.info(f"Table {table_name} is empty. Skipping.")
        return

    logger.info(f"Found {len(rows)} rows in {table_name}.")

    # 2. Prepare Insert SQL
    placeholders = ', '.join(['%s'] * len(columns))
    cols_str = ', '.join(columns)
    
    # ID Handling:
    # If table has 'id', we generally want to keep it to preserve references.
    # Postgres schema has SERIAL for most IDs.
    # Dual-Write fix: We updated schema to match columns.
    
    # ON CONFLICT:
    # Most tables have a PK.
    # adl_history might have issues if we run this multiple times.
    # Let's try INSERT ON CONFLICT DO NOTHING if PK exists.
    
    # Check if table has PK in Postgres? No, just assume DO NOTHING is safe for migration idempotency.
    # Except separate handling for tables without unique constraint violation (like pure logs).
    # But schema defines PKs for almost everything now.
    
    sql = f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    
    # 3. Batch Insert
    batch_size = 1000
    pg_cur = pg_conn.cursor()
    
    # Identify boolean column indices
    boolean_cols = {'is_corrected', 'is_anomaly', 'is_primary', 'is_read'}
    bool_indices = [i for i, col in enumerate(columns) if col in boolean_cols]
    
    try:
        with tqdm(total=len(rows), desc=f"Migrating {table_name}", unit="rows") as pbar:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                
                # Convert processed batch
                processed_batch = []
                for row in batch:
                    # Convert to list to modify
                    row_list = list(row)
                    for idx in bool_indices:
                        if row_list[idx] is not None:
                            # forcing bool conversion from 0/1 integers
                            row_list[idx] = bool(row_list[idx])
                    processed_batch.append(tuple(row_list))
                
                pg_cur.executemany(sql, processed_batch)
                pg_conn.commit()
                pbar.update(len(batch))
                
    except Exception as e:
        logger.error(f"Failed to migrate {table_name}: {e}")
        pg_conn.rollback()
        # Optionally continue to next table or stop?
        # Let's stop to avoid broken FK chains
        raise e
    finally:
        pg_cur.close()

def migrate():
    # Connect SQLite
    logger.info(f"Connecting to SQLite: {DB_PATH}")
    sqlite_conn = sqlite3.connect(DB_PATH)
    sqlite_conn.row_factory = sqlite3.Row

    # Connect Postgres
    logger.info(f"Connecting to Postgres: {POSTGRES_CONFIG['host']}")
    # Filter pool args
    pg_connect_args = {k: v for k, v in POSTGRES_CONFIG.items() if k not in ['minconn', 'maxconn']}
    pg_conn = psycopg2.connect(**pg_connect_args)

    try:
        for table in TABLES:
            migrate_table(sqlite_conn, pg_conn, table)
        logger.info("Migration completed successfully!")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
    finally:
        sqlite_conn.close()
        pg_conn.close()

if __name__ == "__main__":
    migrate()
