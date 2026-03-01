#!/usr/bin/env python3
"""
Golden Sample Importer for Beta 6 Upgrade Phase.

This script imports Golden Samples (exported via harvest_gold_samples.py) into
a freshly upgraded database. It is designed to handle schema drift gracefully.

Usage:
    # Standard import (skip duplicates)
    python import_golden_samples.py golden_samples_export/golden_samples_YYYYMMDD_HHMMSS.json

    # Force update existing records
    python import_golden_samples.py --update golden_samples_export/golden_samples_*.json

Workflow:
    1. Mac Studio (Production) runs: harvest_gold_samples.py -> exports JSON
    2. Mac Studio upgrades code: git pull origin main
    3. Mac Studio re-initializes DB: /reset workflow
    4. Mac Studio imports samples: import_golden_samples.py <json_file>
    5. Mac Studio retrains: Training uses imported golden samples

Author: Beta 5.5 Team
Date: Feb 6, 2026
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
    
# Import adapter
try:
    from backend.db.legacy_adapter import LegacyDatabaseAdapter
except ImportError:
    from elderlycare_v1_16.database import db as adapter
else:
    adapter = LegacyDatabaseAdapter()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add backend to path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

try:
    from elderlycare_v1_16.config.settings import DB_PATH
except ImportError:
    DB_PATH = BACKEND_DIR / "elderlycare_v1_16" / "data" / "elderlycare.db"


def get_db_connection():
    """Get database connection (Postgres/SQLite via adapter)."""
    return adapter.get_connection()


def ensure_elder_exists(conn, elder_id: str) -> bool:
    """Ensure the elder exists in the database, create if not."""
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM elders WHERE elder_id = ?", (elder_id,))
    if cursor.fetchone():
        return True
    
    # Create placeholder elder
    logger.warning(f"Creating placeholder elder: {elder_id}")
    cursor.execute("""
        INSERT INTO elders (elder_id, name, age, created_date)
        VALUES (?, ?, 80, ?)
    """, (elder_id, f"Imported_{elder_id}", datetime.now().isoformat()))
    conn.commit()
    return True


def import_sample(conn, sample: Dict, update_existing: bool = False) -> str:
    """
    Import a single golden sample into the database.
    
    Returns: 'inserted', 'updated', 'skipped', or 'error'
    """
    cursor = conn.cursor()
    
    elder_id = sample.get('elder_id')
    timestamp = sample.get('timestamp')
    room = sample.get('room')
    activity = sample.get('activity_type') or sample.get('activity')
    sensor_features = sample.get('sensor_features')
    
    if not all([elder_id, timestamp, activity]):
        logger.warning(f"Skipping incomplete sample: {sample}")
        return 'error'
    
    # Ensure elder exists
    ensure_elder_exists(conn, elder_id)
    
    # Check if record already exists
    cursor.execute("""
        SELECT id, is_corrected FROM adl_history 
        WHERE elder_id = ? AND timestamp = ? AND room = ?
    """, (elder_id, timestamp, room))
    existing = cursor.fetchone()
    
    if existing:
        if not update_existing:
            return 'skipped'
        
        # Update existing record
        cursor.execute("""
            UPDATE adl_history 
            SET activity_type = ?, is_corrected = 1, sensor_features = ?
            WHERE id = ?
        """, (activity, json.dumps(sensor_features) if sensor_features else None, existing['id']))
        return 'updated'
    else:
        # Insert new record
        # Extract date from timestamp
        try:
            record_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%Y-%m-%d')
        except:
            record_date = timestamp[:10]
        
        cursor.execute("""
            INSERT INTO adl_history 
            (elder_id, record_date, timestamp, activity_type, room, 
             is_corrected, is_anomaly, confidence, sensor_features)
            VALUES (?, ?, ?, ?, ?, 1, 0, 1.0, ?)
        """, (
            elder_id, record_date, timestamp, activity, room,
            json.dumps(sensor_features) if sensor_features else None
        ))
        return 'inserted'


def import_golden_samples(json_file: Path, update_existing: bool = False) -> Dict:
    """
    Import golden samples from a JSON file.
    
    Returns dict with counts: inserted, updated, skipped, errors
    """
    if not json_file.exists():
        raise FileNotFoundError(f"File not found: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Handle both formats: list of samples or dict with 'samples' key
    if isinstance(data, dict):
        samples = data.get('samples', [])
        metadata = {k: v for k, v in data.items() if k != 'samples'}
        logger.info(f"Export metadata: {metadata}")
    else:
        samples = data
    
    logger.info(f"Loaded {len(samples)} samples from {json_file.name}")
    
    ctx = get_db_connection()
    conn = ctx.__enter__()
    results = {'inserted': 0, 'updated': 0, 'skipped': 0, 'errors': 0}
    
    try:
        for i, sample in enumerate(samples):
            result = import_sample(conn, sample, update_existing)
            results[result + 's' if result != 'skipped' else 'skipped'] += 1
            
            if (i + 1) % 1000 == 0:
                conn.commit()
                logger.info(f"  Processed {i+1}/{len(samples)}...")
        
        conn.commit()
    finally:
        ctx.__exit__(None, None, None)
    
    return results


def print_summary(results: Dict):
    """Print import summary."""
    print("\n" + "="*50)
    print("📦 Golden Sample Import Summary")
    print("="*50)
    print(f"  ✅ Inserted:  {results['inserted']}")
    print(f"  🔄 Updated:   {results['updated']}")
    print(f"  ⏭️  Skipped:   {results['skipped']}")
    print(f"  ❌ Errors:    {results['errors']}")
    print("="*50)
    
    total = results['inserted'] + results['updated']
    if total > 0:
        print(f"\n🎉 Successfully imported {total} golden samples!")
        print("   You can now retrain models to incorporate these samples.")
    else:
        print("\n⚠️  No new samples imported. Use --update to overwrite existing.")


def main():
    parser = argparse.ArgumentParser(
        description='Import Golden Samples into the database for training.',
        epilog='Example: python import_golden_samples.py golden_samples_export/*.json'
    )
    parser.add_argument(
        'files',
        nargs='+',
        type=Path,
        help='JSON file(s) containing golden samples (from harvest_gold_samples.py)'
    )
    parser.add_argument(
        '--update',
        action='store_true',
        help='Update existing records (default: skip duplicates)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Parse files but do not modify database'
    )
    
    args = parser.parse_args()
    
    print("🌟 Golden Sample Importer")
    print(f"   Database: {DB_PATH}")
    print(f"   Mode: {'Update existing' if args.update else 'Skip duplicates'}")
    print()
    
    if args.dry_run:
        print("⚠️  DRY RUN - No changes will be made")
        for f in args.files:
            if f.exists():
                with open(f) as fp:
                    data = json.load(fp)
                    count = len(data.get('samples', data) if isinstance(data, dict) else data)
                    print(f"   Would import: {f.name} ({count} samples)")
        return
    
    total_results = {'inserted': 0, 'updated': 0, 'skipped': 0, 'errors': 0}
    
    for json_file in args.files:
        if not json_file.exists():
            logger.error(f"File not found: {json_file}")
            continue
        
        logger.info(f"Importing from: {json_file}")
        results = import_golden_samples(json_file, args.update)
        
        for k, v in results.items():
            total_results[k] += v
    
    print_summary(total_results)


if __name__ == '__main__':
    main()
