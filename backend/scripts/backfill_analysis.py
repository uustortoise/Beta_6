#!/usr/bin/env python3
"""
Backfill Analysis Script

This script scans `adl_history` for all available dates for an elder and runs
the full suite of downstream analysis (Sleep, ICOPE, Insights) to populate
any missing records.

Usage:
    python scripts/backfill_analysis.py [--elder-id ELDER_ID]
    
    If --elder-id is not specified, all elders in the database will be processed.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure backend is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from dotenv import load_dotenv
load_dotenv()

from elderlycare_v1_16.config.settings import DB_PATH
import pandas as pd
from backend.db.legacy_adapter import LegacyDatabaseAdapter


def get_all_elder_ids(conn) -> list:
    """Get all elder IDs from the database."""
    cursor = conn.execute("SELECT elder_id FROM elders")
    return [row[0] for row in cursor.fetchall()]


def get_dates_with_adl_data(conn, elder_id: str) -> list:
    """Get all dates that have ADL data for an elder."""
    cursor = conn.execute('''
        SELECT DISTINCT record_date FROM adl_history
        WHERE elder_id = ?
        ORDER BY record_date
    ''', (elder_id,))
    return [row[0] for row in cursor.fetchall()]


def get_dates_with_sleep_analysis(conn, elder_id: str) -> set:
    """Get all dates that already have sleep analysis for an elder."""
    cursor = conn.execute('''
        SELECT DISTINCT analysis_date FROM sleep_analysis
        WHERE elder_id = ?
    ''', (elder_id,))
    return {row[0] for row in cursor.fetchall()}


def get_adl_data_for_date(conn, elder_id: str, record_date: str) -> pd.DataFrame:
    """Get ADL data for a specific elder and date."""
    cursor = conn.execute('''
        SELECT timestamp, room, activity_type as predicted_activity, confidence, sensor_features
        FROM adl_history
        WHERE elder_id = ? AND record_date = ?
        ORDER BY timestamp
    ''', (elder_id, record_date))
    
    rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame()
    
    # Handle tuple results from Postgres adapter (no row_factory)
    if cursor.description:
        cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(rows, columns=cols)
    else:
        # Fallback if description missing (unlikely)
        df = pd.DataFrame([dict(row) for row in rows])
    
    # --- Motion Data Normalization (External Review Recommendation) ---
    # Use centralized MotionDataNormalizer for consistent handling across pipelines.
    # This replaces the previous approach of simply dropping motion data.
    try:
        from utils.motion_normalizer import MotionDataNormalizer
        df, quality = MotionDataNormalizer.normalize_for_sleep_analysis(
            df,
            source=f"backfill:{elder_id}:{record_date}"
        )
        # Note: Without raw_data_source, invalid motion will be removed and heuristics used
    except Exception as e:
        logger.warning(
            f"Motion normalization unavailable for {elder_id}/{record_date}: {e}. "
            "Falling back to heuristic-only mode."
        )
        # Fallback to original behavior if normalizer is unavailable/fails
        if 'motion' in df.columns:
            df = df.drop(columns=['motion'])
        
    return df



def run_downstream_analysis(elder_id: str, prediction_results: dict, record_date: str):
    """
    Run all downstream analysis (Sleep, ICOPE, Insights) for the given data.
    
    This is a standalone version that can be called from backfill or pipeline.
    """
    import numpy as np
    from elderlycare_v1_16.services.sleep_service import SleepService
    from elderlycare_v1_16.services.icope_service import ICOPEService
    from elderlycare_v1_16.services.insight_service import InsightService
    from ml.household_analyzer import HouseholdAnalyzer
    from ml.sleep_analyzer import SleepAnalyzer
    
    sleep_svc = SleepService()
    insight_svc = InsightService()
    
    # --- Sleep Analysis ---
    logger.info(f"Generating sleep analysis for {elder_id} on {record_date} (Standardized)...")
    try:
        analyzer = SleepAnalyzer()
        sleep_analysis = analyzer.analyze_from_predictions(elder_id, prediction_results)
        
        if sleep_analysis:
            sleep_svc.save_sleep_analysis(elder_id, sleep_analysis, record_date)
            logger.info(f"  Sleep: {sleep_analysis['total_duration_hours']:.1f}h, Quality: {sleep_analysis['quality_score']}")
        else:
            logger.info(f"  No sleep events found for {record_date}")
            
    except Exception as e:
        logger.error(f"Error in SleepAnalyzer: {e}", exc_info=True)
    
    # --- ICOPE Assessment ---
    logger.info(f"Generating ICOPE assessment for {elder_id} on {record_date}...")
    try:
        icope_svc = ICOPEService()
        icope_result = icope_svc.calculate_and_save(elder_id, prediction_results, record_date)
        if icope_result:
            logger.info(f"  ICOPE: Overall={icope_result['overall_score']}, Trend={icope_result['trend']}")
    except Exception as e:
        logger.error(f"Error generating ICOPE assessment: {e}")
    
    # --- Health Insights ---
    logger.info(f"Running health insights for {elder_id}...")
    try:
        alerts = insight_svc.run_daily_analysis(elder_id, analysis_date=record_date)
        if alerts:
            logger.info(f"  Generated {len(alerts)} health alerts.")
    except Exception as e:
        logger.error(f"Error running insights: {e}")
    
    # --- Household Analysis ---
    try:
        h_analyzer = HouseholdAnalyzer()
        h_analyzer.analyze_day(elder_id, record_date)
    except Exception as e:
        logger.error(f"Error running Household Analysis: {e}")


def backfill_elder(conn, elder_id: str, force: bool = False, date_range: tuple = None):
    """
    Backfill analysis for a single elder.
    
    Args:
        conn: Database connection
        elder_id: Elder identifier
        force: If True, reprocess even if analysis exists
        date_range: Optional tuple of (start_date, end_date) to limit processing
    """
    logger.info(f"Processing elder: {elder_id}")
    
    adl_dates = get_dates_with_adl_data(conn, elder_id)
    if not adl_dates:
        logger.warning(f"  No ADL data found for {elder_id}")
        return 0
    
    # Apply date range filter if specified
    if date_range:
        start_date, end_date = date_range
        adl_dates = [d for d in adl_dates if start_date <= d <= end_date]
        if not adl_dates:
            logger.info(f"  No ADL data in date range {start_date} to {end_date}")
            return 0
    
    existing_sleep_dates = get_dates_with_sleep_analysis(conn, elder_id)
    
    dates_to_process = adl_dates if force else [d for d in adl_dates if d not in existing_sleep_dates]
    
    if not dates_to_process:
        logger.info(f"  All dates already have analysis for {elder_id}")
        return 0
    
    logger.info(f"  Found {len(dates_to_process)} dates to process")
    processed_count = 0
    
    for record_date in dates_to_process:
        logger.info(f"  Processing date: {record_date}")
        
        # Get ADL data for this date
        adl_df = get_adl_data_for_date(conn, elder_id, record_date)
        if adl_df.empty:
            logger.warning(f"    No ADL data for {record_date}")
            continue
        
        # Format as prediction_results dict
        prediction_results = {'all_rooms': adl_df}
        
        # Run downstream analysis
        run_downstream_analysis(elder_id, prediction_results, record_date)
        processed_count += 1
    
    return processed_count


def process_elder_batch(conn, elder_ids: list, batch_size: int = 50, 
                        date_range: tuple = None, force: bool = False):
    """
    Process elders in batches with date range constraints.
    
    Implements external review recommendation for performance optimization:
    - Batch processing reduces memory pressure
    - Date range constraints limit query scope
    - Progress reporting for long-running operations
    
    Args:
        conn: Database connection
        elder_ids: List of elder IDs to process
        batch_size: Number of elders per batch (default: 50)
        date_range: Optional tuple of (start_date, end_date) strings
        force: If True, reprocess even if analysis exists
        
    Returns:
        dict with processing statistics
    """
    from datetime import datetime
    
    total_elders = len(elder_ids)
    total_dates_processed = 0
    batch_count = (total_elders + batch_size - 1) // batch_size
    
    logger.info(f"Processing {total_elders} elders in {batch_count} batches (size={batch_size})")
    if date_range:
        logger.info(f"Date range filter: {date_range[0]} to {date_range[1]}")
    
    start_time = datetime.now()
    
    for batch_idx in range(batch_count):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_elders)
        batch_elders = elder_ids[batch_start:batch_end]
        
        logger.info(f"--- Batch {batch_idx + 1}/{batch_count} ({len(batch_elders)} elders) ---")
        
        for eid in batch_elders:
            dates_processed = backfill_elder(conn, eid, force, date_range)
            total_dates_processed += dates_processed
        
        # Progress report
        elapsed = (datetime.now() - start_time).total_seconds()
        elders_done = batch_end
        rate = elders_done / elapsed if elapsed > 0 else 0
        eta_seconds = (total_elders - elders_done) / rate if rate > 0 else 0
        
        logger.info(f"Progress: {elders_done}/{total_elders} elders "
                   f"({elders_done/total_elders*100:.1f}%), "
                   f"ETA: {eta_seconds/60:.1f} min")
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    return {
        'total_elders': total_elders,
        'total_dates_processed': total_dates_processed,
        'total_time_seconds': total_time,
        'elders_per_second': total_elders / total_time if total_time > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description='Backfill sleep and health analysis from ADL history')
    parser.add_argument('--elder-id', type=str, help='Specific elder ID to process (default: all)')
    parser.add_argument('--force', action='store_true', help='Reprocess even if analysis exists')
    # Performance optimization options (External Review Recommendation)
    parser.add_argument('--batch-size', type=int, default=50, 
                        help='Number of elders per batch (default: 50)')
    parser.add_argument('--start-date', type=str, 
                        help='Start date for processing (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, 
                        help='End date for processing (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # Build date range if specified
    date_range = None
    if args.start_date or args.end_date:
        start = args.start_date or '1900-01-01'
        end = args.end_date or '2099-12-31'
        date_range = (start, end)
    
    logger.info("=" * 60)
    logger.info("Starting Backfill Analysis")
    logger.info("=" * 60)
    
    logger.info("=" * 60)
    
    with LegacyDatabaseAdapter().get_connection() as conn:
        # conn.row_factory not needed/supported for adapter. 
        # We manually handle row conversion in functions.
        
        if args.elder_id:
            backfill_elder(conn, args.elder_id, args.force, date_range)
        else:
            elder_ids = get_all_elder_ids(conn)
            logger.info(f"Found {len(elder_ids)} elders in database")
            
            # Use batch processing for multiple elders
            stats = process_elder_batch(
                conn, 
                elder_ids, 
                batch_size=args.batch_size,
                date_range=date_range,
                force=args.force
            )
            
            logger.info(f"Stats: {stats['total_dates_processed']} dates processed "
                       f"in {stats['total_time_seconds']:.1f}s "
                       f"({stats['elders_per_second']:.2f} elders/sec)")
    
    logger.info("=" * 60)
    logger.info("Backfill Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
