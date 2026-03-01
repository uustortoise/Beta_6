#!/usr/bin/env python3
"""
Golden Sample Harvester for Beta 6 Universal Backbone Training.

This script extracts verified "Golden Samples" (human-corrected activity labels)
from the database along with their sensor features, ready for Backbone training.

Usage:
    python harvest_gold_samples.py --dry-run              # Count samples without exporting
    python harvest_gold_samples.py --output ./data/       # Export to directory
    python harvest_gold_samples.py --filter-safe-only     # Only "Traffic Light Safe" activities

Traffic Light Rules (from labeling_guide.md):
    🟢 SAFE:    sleep, nap, shower (physics-consistent, mass-labelable)
    🟡 CAUTION: out (verify door sensor)
    🔴 UNSAFE:  inactive, *_normal_use (subtle, don't mass-label)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

# Add backend to path for imports (same pattern as other scripts)
_script_dir = Path(__file__).resolve().parent
_backend_dir = _script_dir.parent
sys.path.insert(0, str(_backend_dir))

# Import after path setup
from elderlycare_v1_16.config.settings import DEFAULT_SENSOR_COLUMNS, PROJECT_ROOT, DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# Traffic Light Classification
SAFE_ACTIVITIES = {'sleep', 'nap', 'shower'}
CAUTION_ACTIVITIES = {'out'}
UNSAFE_ACTIVITIES = {'inactive', 'unoccupied', 'low_confidence'}  # Plus any *_normal_use


def is_safe_activity(activity: str) -> bool:
    """Check if activity is 'Traffic Light Safe' for mass-labeling."""
    activity_lower = activity.lower().strip()
    if activity_lower in SAFE_ACTIVITIES:
        return True
    if activity_lower in CAUTION_ACTIVITIES:
        return True  # Include with caution warning
    if activity_lower in UNSAFE_ACTIVITIES:
        return False
    if '_normal_use' in activity_lower:
        return False
    # Unknown activities default to caution (include with warning)
    return True


def fetch_golden_samples_with_sensors(
    elder_id: Optional[str] = None,
    safe_only: bool = False
) -> List[Dict[str, Any]]:
    """
    Fetch all golden samples with sensor features from the database.
    
    Args:
        elder_id: If provided, filter to specific resident
        safe_only: If True, only include Traffic Light Safe activities
        
    Returns:
        List of dicts with: elder_id, room, timestamp, activity, sensor_features
    """
    # Use Legacy Adapter to support both SQLite and PostgreSQL transparently
    try:
        from backend.db.legacy_adapter import LegacyDatabaseAdapter
    except ImportError:
        # Fallback setup if running as script
        from elderlycare_v1_16.database import db as adapter
    else:
        adapter = LegacyDatabaseAdapter()
    
    query = """
        SELECT 
            elder_id,
            room,
            timestamp,
            activity_type as activity,
            sensor_features
        FROM adl_history
        WHERE is_corrected = 1
          AND sensor_features IS NOT NULL
    """
    params = []
    
    if elder_id:
        query += " AND elder_id = ?" # LegacyShim handles ? -> %s for Postgres
        params.append(elder_id)
    
    query += " ORDER BY elder_id, room, timestamp"
    
    try:
        with adapter.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

    except Exception as e:
        logger.error(f"Database query failed: {e}")
        return []
    
    samples = []
    skipped_unsafe = 0
    skipped_no_features = 0
    
    for row in rows:
        elder_id_val, room, timestamp, activity, sensor_features_json = row
        
        # Traffic Light filter
        if safe_only and not is_safe_activity(activity):
            skipped_unsafe += 1
            continue
        
        # Parse sensor features
        if not sensor_features_json:
            skipped_no_features += 1
            continue
            
        try:
            sensor_features = json.loads(sensor_features_json)
        except (json.JSONDecodeError, TypeError):
            skipped_no_features += 1
            continue
        
        samples.append({
            'elder_id': elder_id_val,
            'room': room,
            'timestamp': str(timestamp),
            'activity': activity,
            'sensor_features': sensor_features
        })
    
    if skipped_unsafe > 0:
        logger.info(f"Skipped {skipped_unsafe} unsafe activities (Traffic Light filter)")
    if skipped_no_features > 0:
        logger.warning(f"Skipped {skipped_no_features} samples with missing/invalid sensor_features")
    
    return samples


def export_samples(samples: List[Dict], output_dir: Path) -> Path:
    """Export samples to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"golden_dataset_{timestamp}.json"
    
    # Add metadata
    export_data = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'sample_count': len(samples),
            'unique_elders': len(set(s['elder_id'] for s in samples)),
            'unique_rooms': len(set(s['room'] for s in samples)),
            'activities': list(set(s['activity'] for s in samples))
        },
        'samples': samples
    }
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    return output_file


def print_statistics(samples: List[Dict]):
    """Print summary statistics of collected samples."""
    if not samples:
        logger.info("No samples found.")
        return
    
    # Group by elder
    by_elder = {}
    for s in samples:
        elder = s['elder_id']
        by_elder[elder] = by_elder.get(elder, 0) + 1
    
    # Group by activity
    by_activity = {}
    for s in samples:
        activity = s['activity']
        by_activity[activity] = by_activity.get(activity, 0) + 1
    
    # Group by room
    by_room = {}
    for s in samples:
        room = s['room']
        by_room[room] = by_room.get(room, 0) + 1
    
    print("\n" + "=" * 60)
    print("GOLDEN SAMPLE HARVEST STATISTICS")
    print("=" * 60)
    print(f"\nTotal Samples: {len(samples)}")
    print(f"Unique Elders: {len(by_elder)}")
    
    print("\n--- By Elder ---")
    for elder, count in sorted(by_elder.items()):
        print(f"  {elder}: {count} samples")
    
    print("\n--- By Activity ---")
    for activity, count in sorted(by_activity.items(), key=lambda x: -x[1]):
        safety = "🟢" if activity.lower() in SAFE_ACTIVITIES else (
            "🟡" if activity.lower() in CAUTION_ACTIVITIES else "🔴"
        )
        print(f"  {safety} {activity}: {count}")
    
    print("\n--- By Room ---")
    for room, count in sorted(by_room.items(), key=lambda x: -x[1]):
        print(f"  {room}: {count}")
    
    print("=" * 60 + "\n")


def build_harvest_quality_report(
    samples: List[Dict[str, Any]],
    *,
    min_samples_per_activity: int = 8,
) -> Dict[str, Any]:
    by_activity: Dict[str, int] = {}
    by_elder: Dict[str, int] = {}
    for row in samples:
        activity = str(row.get("activity", "")).strip().lower()
        elder = str(row.get("elder_id", "")).strip()
        if activity:
            by_activity[activity] = by_activity.get(activity, 0) + 1
        if elder:
            by_elder[elder] = by_elder.get(elder, 0) + 1

    low_support = {
        activity: count for activity, count in sorted(by_activity.items()) if count < int(min_samples_per_activity)
    }
    p0_violations: List[str] = []
    if not samples:
        p0_violations.append("empty_sample_set")
    if len(by_elder) < 2:
        p0_violations.append("insufficient_unique_elders")
    if low_support:
        p0_violations.append("low_support_safe_classes")

    return {
        "status": "pass" if not p0_violations else "fail",
        "sample_count": len(samples),
        "unique_elders": len(by_elder),
        "activity_counts": by_activity,
        "low_support_activities": low_support,
        "min_samples_per_activity": int(min_samples_per_activity),
        "p0_violations": p0_violations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Harvest Golden Samples for Beta 6 Universal Backbone training"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Count and display statistics without exporting'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=PROJECT_ROOT / 'data' / 'golden_samples',
        help='Output directory for exported JSON (default: data/golden_samples/)'
    )
    parser.add_argument(
        '--filter-safe-only',
        action='store_true',
        help='Only include Traffic Light Safe activities (sleep, nap, shower, out)'
    )
    parser.add_argument(
        '--elder-id',
        type=str,
        default=None,
        help='Filter to specific elder ID'
    )
    parser.add_argument(
        '--quality-report-output',
        type=Path,
        default=None,
        help='Optional path to write harvest quality report JSON'
    )
    parser.add_argument(
        '--min-samples-per-activity',
        type=int,
        default=8,
        help='Minimum support threshold per activity for quality report'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting Golden Sample Harvest...")
    logger.info(f"  Safe-only filter: {args.filter_safe_only}")
    logger.info(f"  Elder filter: {args.elder_id or 'ALL'}")
    
    # Fetch samples
    samples = fetch_golden_samples_with_sensors(
        elder_id=args.elder_id,
        safe_only=args.filter_safe_only
    )
    
    # Print statistics
    print_statistics(samples)
    quality_report = build_harvest_quality_report(
        samples,
        min_samples_per_activity=max(int(args.min_samples_per_activity), 1),
    )
    logger.info(
        "Harvest quality: status=%s unique_elders=%s sample_count=%s p0_violations=%s",
        quality_report.get("status"),
        quality_report.get("unique_elders"),
        quality_report.get("sample_count"),
        quality_report.get("p0_violations"),
    )
    if args.quality_report_output:
        report_path = Path(args.quality_report_output).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
        logger.info(f"Wrote harvest quality report: {report_path}")
    
    if args.dry_run:
        logger.info("Dry run complete. No files exported.")
        return
    
    if not samples:
        logger.warning("No samples to export. Exiting.")
        return
    
    # Export
    output_file = export_samples(samples, args.output)
    logger.info(f"✅ Exported {len(samples)} samples to: {output_file}")


if __name__ == '__main__':
    main()
