import psycopg2
import os
import sys
from pathlib import Path

# Postgres Config
POSTGRES_CONFIG = {
    'host': os.getenv("POSTGRES_HOST", "localhost"),
    'port': int(os.getenv("POSTGRES_PORT", "5432")),
    'user': os.getenv("POSTGRES_USER", "postgres"),
    'password': os.getenv("POSTGRES_PASSWORD", "password"),
    'dbname': os.getenv("POSTGRES_DB", "elderlycare")
}

SCHEMA_PATH = Path("elderlycare_v1_16/models/schema.sql")

def init_postgres():
    print("🚀 Initializing PostgreSQL Schema...")
    
    if not SCHEMA_PATH.exists():
        print(f"❌ Schema file not found at {SCHEMA_PATH}")
        return

    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cur = conn.cursor()
        
        with open(SCHEMA_PATH, 'r') as f:
            schema_sql = f.read()
            
        # Convert SQLite syntax to PostgreSQL
        print("   🔄 Converting SQLite syntax to PostgreSQL...")
        
        # 1. AUTOINCREMENT -> SERIAL (Postgres equivalent)
        # Regex: INTEGER PRIMARY KEY AUTOINCREMENT -> SERIAL PRIMARY KEY
        import re
        schema_sql = re.sub(r'INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT', 'SERIAL PRIMARY KEY', schema_sql, flags=re.IGNORECASE)
        
        # 2. DATETIME -> TIMESTAMP
        schema_sql = re.sub(r'\bDATETIME\b', 'TIMESTAMP', schema_sql, flags=re.IGNORECASE)
        
        # 4. INSERT OR IGNORE -> ON CONFLICT DO NOTHING
        # Regex: INSERT OR IGNORE INTO table (...) VALUES (...)
        # Postgres: INSERT INTO table (...) VALUES (...) ON CONFLICT DO NOTHING
        if "INSERT OR IGNORE" in schema_sql:
             schema_sql = re.sub(r'INSERT\s+OR\s+IGNORE\s+INTO', 'INSERT INTO', schema_sql, flags=re.IGNORECASE)
             # This is tricky because ON CONFLICT goes at the end.
             # Simple regex replace is hard for full statement.
             # BUT, for the specific seed lines in schema.sql, they end with ); or similar.
             # Easier hack: Just drop those seed lines for now? Or try to append ON CONFLICT.
             # Actually, for init_db, seeding household_config is important.
             # Let's replace "VALUES (...);" with "VALUES (...) ON CONFLICT DO NOTHING;"
             # Only if it was INSERT OR IGNORE.
             # Better approach: remove "OR IGNORE" and use regex to append ON CONFLICT at end of statement if it was removed?
             # Too complex for regex.
             # Alternative: Just use TRY/EXCEPT block in python to ignore UniqueViolation?
             # But the SQL syntax error happens at parse time.
             # Simple fix: Hardcode replacement for the specific known seed lines?
             pass 

        # Execute schema
        # Split by ; to handle multiple statements properly
        statements = schema_sql.split(';')
        for stmt in statements:
            if stmt.strip():
                # Specific fix for INSERT OR IGNORE (simple version)
                if "INSERT OR IGNORE" in stmt:
                    stmt = stmt.replace("INSERT OR IGNORE", "INSERT")
                    stmt = stmt.rstrip().rstrip(';') + " ON CONFLICT DO NOTHING;"
                
                try:
                    cur.execute(stmt)
                    conn.commit() # Commit each statement immediately so one fail doesn't kill others
                except Exception as e:
                    print(f"   ⚠️  Warning executing statement: {stmt[:50]}... -> {e}")
                    conn.rollback()
        
        conn.commit()
        cur.close()
        conn.close()
        print("✅ PostgreSQL Schema Initialized Successfully")
    except Exception as e:
        print(f"❌ Failed to initialize PostgreSQL: {e}")

if __name__ == "__main__":
    init_postgres()
