import sqlite3
import psycopg2
import sys
import os
import logging
from prettytable import PrettyTable

# Adjust path to find backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.elderlycare_v1_16.config.settings import DB_PATH, POSTGRES_CONFIG

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Same table list as migration
TABLES = [
    'elders', 'emergency_contacts', 'medical_history', 'icope_assessments',
    'sleep_analysis', 'alerts', 'alert_rules', 'alert_rules_v2',
    'routine_anomalies', 'trajectory_events', 'activity_segments',
    'household_behavior', 'household_segments', 'household_config',
    'predictions', 'sensor_data', 'adl_history', 'correction_history',
    'training_history', 'model_training_history', 'context_episodes'
]

def verify():
    # Connect
    sqlite_conn = sqlite3.connect(DB_PATH)
    pg_connect_args = {k: v for k, v in POSTGRES_CONFIG.items() if k not in ['minconn', 'maxconn']}
    pg_conn = psycopg2.connect(**pg_connect_args)
    
    sqlite_cur = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()
    
    table = PrettyTable()
    table.field_names = ["Table", "SQLite Rows", "Postgres Rows", "Status"]
    
    all_match = True
    
    for t in TABLES:
        # SQLite Count
        try:
            res_sl = sqlite_cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except:
            res_sl = "N/A"
            
        # Postgres Count
        try:
            pg_cur.execute(f"SELECT COUNT(*) FROM {t}")
            res_pg = pg_cur.fetchone()[0]
        except Exception as e:
            res_pg = f"Err: {e}"
        
        status = "✅ MATCH" if res_sl == res_pg else "❌ MISMATCH"
        if res_sl != res_pg:
            all_match = False
            
        table.add_row([t, res_sl, res_pg, status])
    
    print(table)
    
    sqlite_conn.close()
    pg_conn.close()
    
    if all_match:
        print("\n🎉 MIGRATION VERIFIED SUCCESSFUL!")
        sys.exit(0)
    else:
        print("\n⚠️ MIGRATION VERIFICATION FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    verify()
