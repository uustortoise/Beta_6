#!/usr/bin/env python3
"""
Migration script to add soft-delete columns to correction_history table.
Run this once to upgrade existing databases.

Usage:
    python scripts/migrate_soft_delete.py
"""
import sqlite3
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_PATH


def migrate():
    """Add soft-delete columns to correction_history if they don't exist."""
    print(f"Migrating database: {DB_PATH}")
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(correction_history)")
        columns = {row[1] for row in cursor.fetchall()}
        
        migrations = []
        
        if 'is_deleted' not in columns:
            migrations.append(
                "ALTER TABLE correction_history ADD COLUMN is_deleted INTEGER DEFAULT 0"
            )
            
        if 'deleted_at' not in columns:
            migrations.append(
                "ALTER TABLE correction_history ADD COLUMN deleted_at TIMESTAMP"
            )
            
        if 'deleted_by' not in columns:
            migrations.append(
                "ALTER TABLE correction_history ADD COLUMN deleted_by TEXT"
            )
        
        if not migrations:
            print("✓ All soft-delete columns already exist. No migration needed.")
            return
        
        # Run migrations
        for sql in migrations:
            print(f"  Running: {sql}")
            cursor.execute(sql)
        
        # Create index for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_correction_active 
            ON correction_history(elder_id, is_deleted)
        """)
        
        conn.commit()
        print(f"✓ Migration complete. Added {len(migrations)} column(s).")


if __name__ == "__main__":
    migrate()
