import sqlite3
import psycopg2
import os
import sys
from datetime import datetime
import time

# Configuration
SQLITE_DB_PATH = "../data/processed/residents_master_data.db"
POSTGRES_CONFIG = {
    'host': os.getenv("POSTGRES_HOST", "localhost"),
    'port': int(os.getenv("POSTGRES_PORT", "5432")),
    'user': os.getenv("POSTGRES_USER", "postgres"),
    'password': os.getenv("POSTGRES_PASSWORD", "password"),
    'dbname': os.getenv("POSTGRES_DB", "elderlycare")
}

TABLES_TO_MIGRATE = [
    "elders",
    "adl_history",
    "activity_segments",
    "training_history",
    "correction_history",
    "alerts",
    "sleep_analysis",
    "household_behavior" # If exists
]

def migrate():
    print("🚀 Starting Migration: SQLite -> PostgreSQL")
    
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"❌ SQLite database not found at {SQLITE_DB_PATH}")
        return

    # Connect to SQLite
    try:
        sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
        sqlite_conn.row_factory = sqlite3.Row
        sqlite_cur = sqlite_conn.cursor()
        print("✅ Connected to SQLite")
    except Exception as e:
        print(f"❌ Failed to connect to SQLite: {e}")
        return

    # Connect to Postgres
    try:
        pg_conn = psycopg2.connect(**POSTGRES_CONFIG)
        pg_cur = pg_conn.cursor()
        print("✅ Connected to PostgreSQL")
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        return

    for table in TABLES_TO_MIGRATE:
        print(f"\n📦 Migrating table: {table}")
        
        # 1. Read from SQLite
        try:
            sqlite_cur.execute(f"SELECT * FROM {table}")
            rows = sqlite_cur.fetchall()
        except sqlite3.OperationalError as e:
            print(f"   ⚠️  Skipping {table}: Table not found in SQLite.")
            continue
            
        if not rows:
            print(f"   ℹ️  Table {table} is empty. Skipping.")
            continue
            
        print(f"   📖 Found {len(rows)} rows in SQLite.")
        
        # 2. Prepare Postgres Insert
        # Get columns from first row
        columns = rows[0].keys()
        col_names = ",".join(columns)
        placeholders = ",".join(["%s"] * len(columns))
        
        insert_query = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
        
        # 3. Batch Insert
        batch_size = 1000
        total_migrated = 0
        start_time = time.time()
        
        try:
            # Convert SQLite Rows to tuples
            data_tuples = [tuple(row) for row in rows]
            
            # Executemany is faster
            # Process in chunks to avoid massive memory usage for 160k rows if needed, 
            # but for 160k executemany is fine. Let's do huge chunks.
            
            for i in range(0, len(data_tuples), batch_size):
                batch = data_tuples[i:i + batch_size]
                pg_cur.executemany(insert_query, batch)
                pg_conn.commit()
                total_migrated += len(batch)
                sys.stdout.write(f"\r   ⏳ Migrated {total_migrated}/{len(rows)} rows...")
                sys.stdout.flush()
                
            duration = time.time() - start_time
            print(f"\n   ✅ Completed {table} in {duration:.2f}s")
            
        except Exception as e:
            print(f"\n   ❌ Error migrating {table}: {e}")
            pg_conn.rollback()

    print("\n------------------------------------------------")
    print("🎉 Migration Complete!")
    
    sqlite_conn.close()
    pg_conn.close()

if __name__ == "__main__":
    migrate()
