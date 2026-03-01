
import sys
from pathlib import Path
import pandas as pd

# Setup paths
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir.parent))

from backend.db.database import db

def check_corrected_at():
    print("🔍 Checking corrected_at values in correction_history...")
    
    try:
        conn = db.get_connection()
        df = pd.read_sql("""
            SELECT id, elder_id, room, old_activity, new_activity, corrected_at
            FROM correction_history
            ORDER BY id DESC
            LIMIT 10
        """, conn)
        
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("No records found.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_corrected_at()
