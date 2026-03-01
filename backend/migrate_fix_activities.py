#!/usr/bin/env python3
"""
Migration Script: Fix Invalid Room-Activity Combinations in adl_history

This script:
1. Updates validation rules in segment_utils.py
2. Fixes invalid room-activity combinations in adl_history (e.g., "nap" in living room → "inactive")
3. Normalizes activity names (e.g., "kitchen normal use" → "kitchen_normal_use")
4. Regenerates all activity_segments from the cleaned data

Run this ONCE, then retrain the model to learn correct patterns.

Usage: python migrate_fix_activities.py
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database path (backend/ is CWD, so go up one level to project root)
DB_PATH = Path(__file__).parent.parent / 'data' / 'processed' / 'residents_master_data.db'

# Activity normalization mapping (source → target)
ACTIVITY_NORMALIZATION = {
    'kitchen normal use': 'kitchen_normal_use',
    'room_normal_use': 'bedroom_normal_use',
    '5area no action': 'inactive',
    '5area_no_action': 'inactive',
    '5 area no action': 'inactive',
}

# Invalid room-activity combinations to fix
# Format: (normalized_room, invalid_activity) → replacement_activity
INVALID_COMBINATIONS = {
    ('livingroom', 'nap'): 'inactive',
    ('livingroom', 'sleep'): 'inactive',
    ('kitchen', 'nap'): 'inactive',
    ('kitchen', 'sleep'): 'inactive',
    ('bathroom', 'nap'): 'inactive',
    ('bathroom', 'sleep'): 'inactive',
    ('entrance', 'nap'): 'inactive',
    ('entrance', 'sleep'): 'inactive',
}

def normalize_room_name(room: str) -> str:
    """Normalize room name for matching."""
    return room.lower().replace(' ', '').replace('_', '')

def run_migration():
    """Execute the migration."""
    logger.info("=" * 60)
    logger.info("MIGRATION: Fix Invalid Room-Activity Combinations")
    logger.info("=" * 60)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Step 1: Normalize activity names
        logger.info("\n[1/4] Normalizing activity names...")
        total_normalized = 0
        for source, target in ACTIVITY_NORMALIZATION.items():
            cursor.execute('''
                UPDATE adl_history 
                SET activity_type = ?
                WHERE activity_type = ?
            ''', (target, source))
            count = cursor.rowcount
            if count > 0:
                logger.info(f"  '{source}' → '{target}': {count} rows")
                total_normalized += count
        logger.info(f"  Total normalized: {total_normalized} rows")
        
        # Step 2: Fix invalid room-activity combinations
        logger.info("\n[2/4] Fixing invalid room-activity combinations...")
        total_fixed = 0
        for (room, invalid_activity), replacement in INVALID_COMBINATIONS.items():
            # Match using normalized room name
            cursor.execute('''
                UPDATE adl_history 
                SET activity_type = ?, is_corrected = 0
                WHERE LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = ?
                  AND activity_type = ?
                  AND is_corrected = 0
            ''', (replacement, room, invalid_activity))
            count = cursor.rowcount
            if count > 0:
                logger.info(f"  '{invalid_activity}' in {room} → '{replacement}': {count} rows")
                total_fixed += count
        logger.info(f"  Total fixed: {total_fixed} rows")
        
        # Step 3: Commit changes
        logger.info("\n[3/4] Committing changes to database...")
        conn.commit()
        logger.info("  ✓ Changes committed")
        
        # Step 4: Regenerate all segments
        logger.info("\n[4/4] Regenerating all activity_segments...")
        from utils.segment_utils import regenerate_segments
        
        cursor.execute('SELECT DISTINCT elder_id, room, record_date FROM adl_history')
        combinations = cursor.fetchall()
        
        total_segments = 0
        for elder_id, room, record_date in combinations:
            count = regenerate_segments(elder_id, room, record_date, conn)
            total_segments += count
            logger.info(f"  {elder_id}/{room}/{record_date}: {count} segments")
        
        conn.commit()
        logger.info(f"  Total segments regenerated: {total_segments}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("MIGRATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Activities normalized: {total_normalized}")
        logger.info(f"  Invalid combinations fixed: {total_fixed}")
        logger.info(f"  Segments regenerated: {total_segments}")
        logger.info("\n⚠️  NEXT STEP: Retrain the model to learn from clean data!")
        logger.info("    Run: streamlit run export_dashboard.py → Correction Studio → Train")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == '__main__':
    # Verify DB exists
    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}")
        exit(1)
    
    # Backup warning
    logger.warning("This script will modify adl_history data!")
    logger.warning(f"Database: {DB_PATH}")
    
    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        logger.info("Aborted.")
        exit(0)
    
    run_migration()
