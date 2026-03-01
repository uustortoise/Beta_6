#!/usr/bin/env python3
"""
Migration script to populate activity_segments from existing adl_history data.

Run this once after upgrading to Beta 3.5 with the new activity_segments feature.
This script consolidates consecutive same-activity events into segments for efficient UI display.

Usage: python3 migrate_segments.py
"""

import sys
import os
import sqlite3
import pandas as pd
from pathlib import Path

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from elderlycare_v1_16.config.settings import DB_PATH

def generate_segments_for_elder(conn, elder_id):
    """Generate activity segments from adl_history for a specific elder."""
    
    # Get all ADL history for this elder
    query = """
        SELECT * FROM adl_history 
        WHERE elder_id = ? 
        ORDER BY room, timestamp ASC
    """
    df = pd.read_sql_query(query, conn, params=(elder_id,))
    
    if df.empty:
        print(f"  No ADL history found for {elder_id}")
        return 0
    
    # Process each room separately
    total_segments = 0
    for room_name in df['room'].unique():
        room_df = df[df['room'] == room_name].copy()
        room_df['timestamp'] = pd.to_datetime(room_df['timestamp'])
        room_df = room_df.sort_values('timestamp')
        
        # Group consecutive same-activity events
        room_df['group'] = (room_df['activity_type'] != room_df['activity_type'].shift()).cumsum()
        
        for group_id, group_df in room_df.groupby('group'):
            activity = group_df['activity_type'].iloc[0]
            start_time = group_df['timestamp'].min()
            end_time = group_df['timestamp'].max() + pd.Timedelta(seconds=10)
            duration_minutes = (end_time - start_time).total_seconds() / 60.0
            avg_confidence = group_df['confidence'].mean() if 'confidence' in group_df.columns else 1.0
            event_count = len(group_df)
            record_date = start_time.date().isoformat()
            
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO activity_segments 
                    (elder_id, room, activity_type, start_time, end_time, 
                     duration_minutes, avg_confidence, event_count, record_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    elder_id, room_name, activity,
                    start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    round(duration_minutes, 2),
                    round(avg_confidence, 3) if not pd.isna(avg_confidence) else 1.0,
                    event_count,
                    record_date
                ))
                total_segments += 1
            except sqlite3.IntegrityError:
                continue  # Skip duplicates
    
    conn.commit()
    return total_segments

def main():
    print("=" * 60)
    print("Activity Segments Migration Script")
    print("=" * 60)
    
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        sys.exit(1)
    
    print(f"Database: {DB_PATH}")
    
    conn = sqlite3.connect(str(DB_PATH))
    
    # Ensure activity_segments table exists
    conn.execute('''
        CREATE TABLE IF NOT EXISTS activity_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            elder_id TEXT,
            room TEXT NOT NULL,
            activity_type TEXT NOT NULL,
            start_time DATETIME NOT NULL,
            end_time DATETIME NOT NULL,
            duration_minutes REAL,
            avg_confidence REAL,
            event_count INTEGER,
            record_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_segments_date ON activity_segments(elder_id, record_date)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_segments_time ON activity_segments(elder_id, start_time)')
    conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_segments_unique ON activity_segments(elder_id, room, start_time)')
    conn.commit()
    
    # Get all elders with ADL history
    elders = conn.execute('SELECT DISTINCT elder_id FROM adl_history').fetchall()
    
    if not elders:
        print("No ADL history found. Nothing to migrate.")
        conn.close()
        return
    
    print(f"\nFound {len(elders)} elder(s) with ADL history:")
    
    total_all = 0
    for (elder_id,) in elders:
        print(f"\n  Processing {elder_id}...")
        count = generate_segments_for_elder(conn, elder_id)
        print(f"    Generated {count} segments")
        total_all += count
    
    conn.close()
    
    print("\n" + "=" * 60)
    print(f"✅ Migration complete! Total segments created: {total_all}")
    print("=" * 60)

if __name__ == "__main__":
    main()
