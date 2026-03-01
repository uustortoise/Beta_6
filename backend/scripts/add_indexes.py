
"""
Script to add performance indexes to the `adl_history` table.
These indexes improve the efficiency of the "DB-First" training data loading strategy.
"""
import sqlite3
import logging
import sys
import os

# Add backend to path to import settings
sys.path.append(os.path.join(os.getcwd(), 'Development/Beta_5.5/backend'))

from elderlycare_v1_16.config.settings import DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_indexes():
    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}")
        return

    logger.info(f"Adding indexes to database at {DB_PATH}...")
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # 1. Composite index for efficient querying by elder, room, time, and correction status
            logger.info("Creating index: idx_adl_history_elder_room_time")
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_adl_history_elder_room_time 
                ON adl_history(elder_id, room, timestamp, is_corrected);
            """)
            
            # 2. Partial index for rows that actually have sensor snapshots
            # This speeds up queries that look for "sensor_features IS NOT NULL"
            logger.info("Creating index: idx_adl_history_sensor_features")
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_adl_history_sensor_features 
                ON adl_history(sensor_features) 
                WHERE sensor_features IS NOT NULL;
            """)
            
            conn.commit()
            logger.info("Indexes added successfully.")
            
    except sqlite3.Error as e:
        logger.error(f"SQLite error adding indexes: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    add_indexes()
