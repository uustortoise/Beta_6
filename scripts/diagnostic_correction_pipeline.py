#!/usr/bin/env python3
"""
================================================================================
CORRECTION PIPELINE DIAGNOSTIC SCRIPT
================================================================================
Run this script to diagnose where corrections are failing in the pipeline.

Usage:
    cd /Users/dicksonng/DT/Development/Beta_5.5
    python scripts/diagnostic_correction_pipeline.py [ELDER_ID] [RECORD_DATE]
    
Examples:
    python scripts/diagnostic_correction_pipeline.py
    python scripts/diagnostic_correction_pipeline.py samuel 2026-02-06
    python scripts/diagnostic_correction_pipeline.py "" 2026-02-06

Output:
    - Reports correction pipeline health
    - Identifies gaps between adl_history, activity_segments, and correction_history
    - Shows timestamp alignment issues
================================================================================
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

# Try to import database adapter
try:
    from backend.db.legacy_adapter import LegacyDatabaseAdapter
    ADAPTER_AVAILABLE = True
except ImportError:
    ADAPTER_AVAILABLE = False
    print("⚠️  Warning: LegacyDatabaseAdapter not found, falling back to SQLite")

# Database path
DB_PATH = Path(__file__).parent.parent / 'data' / 'processed' / 'residents_master_data.db'


def get_connection():
    """Get database connection (PostgreSQL or SQLite)"""
    if ADAPTER_AVAILABLE:
        adapter = LegacyDatabaseAdapter()
        return adapter.get_connection()
    else:
        import sqlite3
        return sqlite3.connect(DB_PATH)


def check_adl_history_corrections(elder_id=None, record_date=None):
    """Check 1: Verify corrections exist in adl_history"""
    print("\n" + "="*80)
    print("CHECK 1: ADL_HISTORY CORRECTIONS")
    print("="*80)
    
    with get_connection() as conn:
        query = """
            SELECT 
                elder_id,
                record_date,
                room,
                COUNT(*) as corrected_row_count,
                MIN(timestamp) as earliest_correction,
                MAX(timestamp) as latest_correction,
                array_agg(DISTINCT activity_type) as activities
            FROM adl_history
            WHERE is_corrected = 1
        """
        params = []
        
        if elder_id:
            query += " AND elder_id = %s" if ADAPTER_AVAILABLE else " AND elder_id = ?"
            params.append(elder_id)
        if record_date:
            query += " AND record_date = %s" if ADAPTER_AVAILABLE else " AND record_date = ?"
            params.append(record_date)
            
        query += """
            GROUP BY elder_id, record_date, room
            ORDER BY latest_correction DESC
            LIMIT 10
        """
        
        try:
            df = pd.read_sql(query, conn, params=params)
            if df.empty:
                print("❌ NO CORRECTIONS FOUND in adl_history")
                print("   → Corrections are NOT being saved to adl_history")
                return None
            else:
                print(f"✅ FOUND {len(df)} correction groups in adl_history:")
                print(df.to_string(index=False))
                return df
        except Exception as e:
            print(f"❌ ERROR querying adl_history: {e}")
            return None


def check_correction_history(elder_id=None, record_date=None):
    """Check 2: Verify correction_history audit trail"""
    print("\n" + "="*80)
    print("CHECK 2: CORRECTION_HISTORY AUDIT TRAIL")
    print("="*80)
    
    with get_connection() as conn:
        query = """
            SELECT 
                id,
                elder_id,
                room,
                timestamp_start,
                timestamp_end,
                old_activity,
                new_activity,
                rows_affected,
                corrected_at,
                is_deleted
            FROM correction_history
            WHERE 1=1
        """
        params = []
        
        if elder_id:
            query += " AND elder_id = %s" if ADAPTER_AVAILABLE else " AND elder_id = ?"
            params.append(elder_id)
        if record_date:
            query += " AND DATE(timestamp_start) = %s" if ADAPTER_AVAILABLE else " AND DATE(timestamp_start) = ?"
            params.append(record_date)
            
        query += """
            ORDER BY corrected_at DESC
            LIMIT 10
        """
        
        try:
            df = pd.read_sql(query, conn, params=params)
            if df.empty:
                print("❌ NO ENTRIES FOUND in correction_history")
                print("   → Audit trail is NOT being created")
                return None
            else:
                print(f"✅ FOUND {len(df)} audit entries:")
                print(df.to_string(index=False))
                return df
        except Exception as e:
            print(f"❌ ERROR querying correction_history: {e}")
            return None


def check_activity_segments(elder_id=None, record_date=None):
    """Check 3: Verify activity_segments reflect corrections"""
    print("\n" + "="*80)
    print("CHECK 3: ACTIVITY_SEGMENTS CORRECTION STATUS")
    print("="*80)
    
    with get_connection() as conn:
        query = """
            SELECT 
                elder_id,
                record_date,
                room,
                activity_type,
                start_time,
                end_time,
                is_corrected,
                correction_source,
                duration_minutes
            FROM activity_segments
            WHERE is_corrected = 1
        """
        params = []
        
        if elder_id:
            query += " AND elder_id = %s" if ADAPTER_AVAILABLE else " AND elder_id = ?"
            params.append(elder_id)
        if record_date:
            query += " AND record_date = %s" if ADAPTER_AVAILABLE else " AND record_date = ?"
            params.append(record_date)
            
        query += """
            ORDER BY start_time DESC
            LIMIT 10
        """
        
        try:
            df = pd.read_sql(query, conn, params=params)
            if df.empty:
                print("❌ NO CORRECTED SEGMENTS FOUND in activity_segments")
                print("   → Segment regeneration is NOT working")
                return None
            else:
                print(f"✅ FOUND {len(df)} corrected segments:")
                print(df.to_string(index=False))
                return df
        except Exception as e:
            print(f"❌ ERROR querying activity_segments: {e}")
            return None


def check_segment_adl_consistency(elder_id=None, record_date=None):
    """Check 4: Verify activity_segments match adl_history"""
    print("\n" + "="*80)
    print("CHECK 4: SEGMENT vs ADL_HISTORY CONSISTENCY")
    print("="*80)
    
    with get_connection() as conn:
        # Find corrections in adl_history that don't have matching segments
        query = """
            SELECT 
                ah.elder_id,
                ah.record_date,
                ah.room,
                ah.activity_type as adl_activity,
                ah.timestamp as adl_timestamp,
                ah.is_corrected as adl_is_corrected,
                seg.activity_type as seg_activity,
                seg.is_corrected as seg_is_corrected,
                seg.start_time as seg_start
            FROM adl_history ah
            LEFT JOIN activity_segments seg ON 
                ah.elder_id = seg.elder_id 
                AND ah.record_date = seg.record_date
                AND LOWER(REPLACE(REPLACE(ah.room, ' ', ''), '_', '')) = LOWER(REPLACE(REPLACE(seg.room, ' ', ''), '_', ''))
                AND ah.timestamp BETWEEN seg.start_time AND seg.end_time
            WHERE ah.is_corrected = 1
        """
        params = []
        
        if elder_id:
            query += " AND ah.elder_id = %s" if ADAPTER_AVAILABLE else " AND ah.elder_id = ?"
            params.append(elder_id)
        if record_date:
            query += " AND ah.record_date = %s" if ADAPTER_AVAILABLE else " AND ah.record_date = ?"
            params.append(record_date)
            
        query += " LIMIT 20"
        
        try:
            df = pd.read_sql(query, conn, params=params)
            
            # Count orphaned corrections (in adl but not in segments)
            orphaned = df[df['seg_activity'].isna()]
            mismatched = df[(df['seg_activity'].notna()) & (df['adl_activity'] != df['seg_activity'])]
            
            if not orphaned.empty:
                print(f"❌ FOUND {len(orphaned)} CORRECTIONS IN adl_history BUT NOT IN activity_segments:")
                print(orphaned[['elder_id', 'record_date', 'room', 'adl_timestamp', 'adl_activity']].to_string(index=False))
            
            if not mismatched.empty:
                print(f"❌ FOUND {len(mismatched)} ACTIVITY MISMATCHES:")
                print(mismatched[['elder_id', 'record_date', 'room', 'adl_activity', 'seg_activity']].to_string(index=False))
            
            if orphaned.empty and mismatched.empty and not df.empty:
                print("✅ All corrections in adl_history have matching segments")
                
            return df
        except Exception as e:
            print(f"❌ ERROR checking consistency: {e}")
            return None


def check_timestamp_drift(elder_id=None, record_date=None):
    """Check 5: Check for timestamp drift between tables"""
    print("\n" + "="*80)
    print("CHECK 5: TIMESTAMP DRIFT ANALYSIS")
    print("="*80)
    
    with get_connection() as conn:
        # Compare timestamps between tables
        query = """
            SELECT 
                ah.elder_id,
                ah.record_date,
                ah.room,
                TO_CHAR(ah.timestamp, 'YYYY-MM-DD HH24:MI:SS.MS TZ') as adl_timestamp,
                TO_CHAR(seg.start_time, 'YYYY-MM-DD HH24:MI:SS.MS TZ') as seg_start_time,
                EXTRACT(EPOCH FROM (ah.timestamp - seg.start_time)) as time_diff_seconds
            FROM adl_history ah
            JOIN activity_segments seg ON 
                ah.elder_id = seg.elder_id 
                AND ah.record_date = seg.record_date
                AND ah.timestamp BETWEEN seg.start_time AND seg.end_time
            WHERE ah.is_corrected = 1
        """ if ADAPTER_AVAILABLE else """
            SELECT 
                ah.elder_id,
                ah.record_date,
                ah.room,
                ah.timestamp as adl_timestamp,
                seg.start_time as seg_start_time,
                (julianday(ah.timestamp) - julianday(seg.start_time)) * 86400 as time_diff_seconds
            FROM adl_history ah
            JOIN activity_segments seg ON 
                ah.elder_id = seg.elder_id 
                AND ah.record_date = seg.record_date
                AND ah.timestamp >= seg.start_time AND ah.timestamp <= seg.end_time
            WHERE ah.is_corrected = 1
        """
        params = []
        
        if elder_id:
            query = query.replace("WHERE ah.is_corrected = 1", 
                                 "WHERE ah.is_corrected = 1 AND ah.elder_id = ?")
            params.append(elder_id)
        if record_date:
            query = query.replace("WHERE ah.is_corrected = 1", 
                                 "WHERE ah.is_corrected = 1 AND ah.record_date = ?")
            params.append(record_date)
            
        query += " LIMIT 10"
        
        try:
            df = pd.read_sql(query, conn, params=params)
            if not df.empty:
                max_drift = df['time_diff_seconds'].abs().max()
                print(f"Maximum timestamp drift: {max_drift:.3f} seconds")
                if max_drift > 10:
                    print(f"❌ WARNING: Drift exceeds 10 seconds - merge_asof may fail!")
                else:
                    print("✅ Timestamp drift within acceptable range")
                print("\nSample comparisons:")
                print(df.to_string(index=False))
            return df
        except Exception as e:
            print(f"❌ ERROR checking timestamp drift: {e}")
            return None


def generate_summary_report(elder_id=None, record_date=None):
    """Generate overall summary"""
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    with get_connection() as conn:
        # Count totals
        queries = {
            'adl_corrections': "SELECT COUNT(*) FROM adl_history WHERE is_corrected = 1",
            'correction_history': "SELECT COUNT(*) FROM correction_history",
            'segment_corrections': "SELECT COUNT(*) FROM activity_segments WHERE is_corrected = 1"
        }
        
        params = []
        filter_clause = ""
        if elder_id:
            filter_clause = " AND elder_id = %s" if ADAPTER_AVAILABLE else " AND elder_id = ?"
            params = [elder_id]
        
        results = {}
        for name, query in queries.items():
            try:
                full_query = query + filter_clause
                cursor = conn.cursor()
                cursor.execute(full_query, params)
                results[name] = cursor.fetchone()[0]
            except Exception as e:
                results[name] = f"Error: {e}"
        
        print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CORRECTION PIPELINE STATUS                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ adl_history corrections:        {results.get('adl_corrections', 'N/A'):>10}                              │
│ correction_history entries:     {results.get('correction_history', 'N/A'):>10}                              │
│ activity_segments corrected:    {results.get('segment_corrections', 'N/A'):>10}                              │
└─────────────────────────────────────────────────────────────────────────────┘
        """)
        
        # Pipeline health assessment
        adl_count = results.get('adl_corrections', 0)
        hist_count = results.get('correction_history', 0)
        seg_count = results.get('segment_corrections', 0)
        
        if isinstance(adl_count, int) and adl_count > 0:
            if seg_count == 0:
                print("🚨 CRITICAL: Corrections exist in adl_history but NOT in activity_segments")
                print("   → Segment regeneration is broken")
            elif abs(adl_count - seg_count) > adl_count * 0.1:
                print("⚠️  WARNING: Significant count mismatch between adl_history and activity_segments")
                print("   → Some corrections not reflected in segments")
            else:
                print("✅ Pipeline appears healthy: Corrections flow from adl_history to segments")
        else:
            print("ℹ️  No corrections found - cannot assess pipeline health")


def main():
    elder_id = sys.argv[1] if len(sys.argv) > 1 else None
    record_date = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║          CORRECTION PIPELINE DIAGNOSTIC TOOL - Beta 5.5                       ║
    ║                                                                               ║
    ║  This script checks the health of the correction pipeline:                    ║
    ║  Labeling Studio → adl_history → activity_segments → Web UI                   ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if elder_id:
        print(f"Filtering for elder_id: {elder_id}")
    if record_date:
        print(f"Filtering for record_date: {record_date}")
    
    # Run all checks
    adl_df = check_adl_history_corrections(elder_id, record_date)
    hist_df = check_correction_history(elder_id, record_date)
    seg_df = check_activity_segments(elder_id, record_date)
    consistency_df = check_segment_adl_consistency(elder_id, record_date)
    drift_df = check_timestamp_drift(elder_id, record_date)
    generate_summary_report(elder_id, record_date)
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("""
Next Steps:
1. If adl_history shows no corrections: Check Labeling Studio save logic
2. If correction_history is empty: Audit trail logging is broken
3. If activity_segments has no corrected entries: Segment regeneration failed
4. If timestamps don't match: Timezone/drift issue causing merge_asof to fail

See FIX_PLAN.md for detailed remediation steps.
    """)


if __name__ == "__main__":
    main()
