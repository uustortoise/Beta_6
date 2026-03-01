import uuid
import logging
import sys
import sqlite3
import psycopg2
from datetime import datetime, timezone
import os

# Adjust path to find backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.db.database import db
from backend.elderlycare_v1_16.config.settings import DB_PATH, POSTGRES_CONFIG

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_dual_write():
    logger.info("Starting Dual-Write Verification...")
    
    # 1. Generate unique test data
    test_uuid = uuid.uuid4().hex[:8]
    test_elder_id = f"test_elder_{test_uuid}"
    test_ts = datetime.now(timezone.utc)
    test_room = "TestRoom"
    test_activity = "verification_activity"
    
    logger.info(f"Generated Test Elder ID: {test_elder_id}")

    # 2. Insert via Dual-Write Layer
    # We first need to ensure the elder exists because of FK constraints
    try:
        logger.info("Inserting parent Elder record...")
        db.execute(
            "INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", 
            (test_elder_id, "Test Elder")
        )
        
        logger.info("Inserting ADL History record via db.execute()...")
        # Ensure we use a query that triggers the dual-write logic (INSERT)
        sql = "INSERT INTO adl_history (elder_id, timestamp, room, activity_type, record_date) VALUES (?, ?, ?, ?, ?)"
        params = (test_elder_id, test_ts, test_room, test_activity, test_ts.date())
        
        # This call should write to SQLite and try to write to Postgres (if enabled/reachable)
        # It handles the return value automatically now (Phase 1.75 fix)
        sqlite_id = db.execute(sql, params)
        logger.info(f"Write operation complete. SQLite ID: {sqlite_id}")
        
    except Exception as e:
        logger.error(f"Failed to insert records via app layer: {e}")
        return False

    # 3. Verify SQLite (Source of Truth)
    sqlite_ok = False
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT * FROM adl_history WHERE elder_id=? AND timestamp=?", 
                (test_elder_id, test_ts)
            ).fetchone()
            if row:
                logger.info("✅ Verified: Record found in SQLite.")
                sqlite_ok = True
            else:
                logger.error("❌ FAILED: Record NOT found in SQLite.")
    except Exception as e:
         logger.error(f"Error checking SQLite: {e}")

    # 4. Verify PostgreSQL (Target)
    postgres_ok = False
    # 4. Verify in PostgreSQL (if enabled)
    # Using Fuzzy Matching for Timestamps (Audit Item 4.1)
    # Postgres and Python/SQLite handle microseconds differently.
    pg_row = None
    # Assuming db.pg_db is available and represents the PostgreSQL connection/pool
    # Also assuming a USE_POSTGRESQL flag or similar logic exists in the actual application context
    # For this verification script, we'll check if POSTGRES_CONFIG is present and db.pg_db is initialized
    if POSTGRES_CONFIG and db.pg_db: # Check if Postgres is configured and db.pg_db is active
        try:
            # Check range +/- 100ms
            sql_check = """
                SELECT id FROM adl_history 
                WHERE elder_id = %s 
                AND timestamp BETWEEN %s - INTERVAL '500 milliseconds' AND %s + INTERVAL '500 milliseconds'
            """
            pg_row = db.pg_db.fetchone(sql_check, (test_elder_id, test_ts, test_ts)) # Use test_elder_id
            if pg_row:
                logger.info(f"✅ Verified: Record found in PostgreSQL. ID: {pg_row[0]}")
                postgres_ok = True
            else:
                logger.warning("⚠️ Warning: Record NOT found in PostgreSQL. (Is Dual-Write enabled?)")
        except Exception as e:
            logger.warning(f"⚠️ PostgreSQL check failed (Database might be down or unreachable): {e}")

    # 5. Cleanup
    logger.info("Cleaning up test data...")
    try:
        db.execute("DELETE FROM adl_history WHERE elder_id=?", (test_elder_id,))
        db.execute("DELETE FROM elders WHERE elder_id=?", (test_elder_id,))
        logger.info("Cleanup complete.")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

    # Final Verdict
    if sqlite_ok and postgres_ok:
        logger.info("\n🎉 SUCCESS: Dual-Write is fully operational!")
        return True
    elif sqlite_ok and not postgres_ok:
        logger.info("\n⚠️ PARTIAL SUCCESS: Written to SQLite, but not PostgreSQL.")
        return False # Fail strict check, but application is safe
    else:
        logger.error("\n❌ FAILURE: Write failed completely.")
        return False

if __name__ == "__main__":
    success = verify_dual_write()
    sys.exit(0 if success else 1)
