
import sys
import os
import logging

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

try:
    from backend.db.legacy_adapter import LegacyDatabaseAdapter
    adapter = LegacyDatabaseAdapter()
    print("Using LegacyDatabaseAdapter")
except ImportError:
    try:
        from elderlycare_v1_16.database import db as adapter
        print("Using elderlycare_v1_16 database adapter")
    except ImportError:
        print("Could not load database adapter")
        sys.exit(1)

def migrate_labels():
    print("Starting migration: living_normal_use -> livingroom_normal_use")
    
    with adapter.get_connection() as conn:
        # Update adl_history
        cursor = conn.execute("SELECT COUNT(*) FROM adl_history WHERE activity_type = 'living_normal_use'")
        count_adl = cursor.fetchone()[0]
        
        if count_adl > 0:
            conn.execute("UPDATE adl_history SET activity_type = 'livingroom_normal_use' WHERE activity_type = 'living_normal_use'")
            print(f"Updated {count_adl} records in adl_history")
        else:
            print("No records found in adl_history")

        # Update activity_segments
        cursor = conn.execute("SELECT COUNT(*) FROM activity_segments WHERE activity_type = 'living_normal_use'")
        count_seg = cursor.fetchone()[0]
        
        if count_seg > 0:
            conn.execute("UPDATE activity_segments SET activity_type = 'livingroom_normal_use' WHERE activity_type = 'living_normal_use'")
            print(f"Updated {count_seg} records in activity_segments")
        else:
            print("No records found in activity_segments")
            
    print("Migration complete.")

if __name__ == "__main__":
    migrate_labels()
