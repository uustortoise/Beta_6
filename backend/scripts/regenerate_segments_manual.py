
import sys
from pathlib import Path
from datetime import datetime

# Setup paths
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir.parent))

from backend.db.legacy_adapter import LegacyDatabaseAdapter
from backend.utils.segment_utils import regenerate_segments

def run_manual_regeneration():
    print("🔧 Starting Manual Segment Regeneration...")
    
    elder_id = 'HK002_samuel'
    target_date = '2026-01-25'
    rooms = ['Bathroom', 'Bedroom', 'Living Room', 'Kitchen'] 
    
    adapter = LegacyDatabaseAdapter()
    
    try:
        with adapter.get_connection() as conn:
            total_segments = 0
            for room in rooms:
                print(f"  Processing {room}...")
                # Note: regenerate_segments expects record_date as string YYYY-MM-DD
                count = regenerate_segments(elder_id, room, target_date, conn=conn)
                print(f"    -> Generated {count} segments for {room}")
                total_segments += count
            
            print(f"✅ Regeneration Complete. Total Segments: {total_segments}")
            
    except Exception as e:
        print(f"❌ Regeneration Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_manual_regeneration()
