
import sys
import os
from pathlib import Path
import pandas as pd

# Setup paths
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir.parent)) # Beta_5.5 root

from backend.db.database import db

def verify_corrections(elder_id="HK002_samuel"):
    print(f"🔍 Verifying corrections for {elder_id}...")
    
    try:
        conn = db.get_connection()
        # 1. Check Audit Trail (correction_history)
        print("\n📋 Audit Trail (Last 5 Corrections):")
        audit_df = pd.read_sql(f"""
            SELECT id, room, timestamp_start, timestamp_end, old_activity, new_activity
            FROM correction_history
            WHERE elder_id = '{elder_id}'
            ORDER BY id DESC
            LIMIT 5
        """, conn)
        
        if not audit_df.empty:
            print(audit_df.to_string(index=False))
        else:
            print("  No corrections found in audit trail.")

        # 2. Check Live Data (adl_history)
        print("\n🗄️ Live Data (adl_history) - Corrected Rows:")
        live_df = pd.read_sql(f"""
            SELECT count(*) as count, room, activity_type, record_date
            FROM adl_history
            WHERE elder_id = '{elder_id}' AND is_corrected = 1
            GROUP BY room, activity_type, record_date
            ORDER BY record_date DESC, room
            LIMIT 10
        """, conn)
        
        if not live_df.empty:
            print(live_df.to_string(index=False))
        else:
            print("  No corrected rows found in adl_history.")

        # 3. Check Timeline (activity_segments)
        print("\n📊 Timeline Segments (activity_segments) - Corrected:")
        segments_df = pd.read_sql(f"""
            SELECT start_time, end_time, room, activity_type, correction_source
            FROM activity_segments
            WHERE elder_id = '{elder_id}' AND is_corrected = 1
            ORDER BY start_time DESC
            LIMIT 5
        """, conn)
        
        if not segments_df.empty:
            print(segments_df.to_string(index=False))
        else:
            print("  No corrected segments found on timeline.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    verify_corrections()
