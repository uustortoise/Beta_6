import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
try:
    from backend.utils.intelligence_db import query_to_dataframe
except ImportError:
    from utils.intelligence_db import query_to_dataframe
try:
    from backend.db.legacy_adapter import LegacyDatabaseAdapter
    adapter = LegacyDatabaseAdapter()
except ImportError:
    # Fallback to sqlite3 if adapter missing (should not happen in prod)
    import sqlite3
    adapter = None

logger = logging.getLogger(__name__)

class DataManager:
    """
    Advanced Data Manager for Beta_2.
    Handles high-volume SQLite storage for prediction logs and 
    user-friendly Excel reports with monthly sheets.
    """
    
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.processed_dir = data_root / "processed"
        self.db_path = self.processed_dir / "residents_master_data.db"
        
        # Ensure directories exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Database
        self._init_db()

    def get_connection(self):
        """Get database connection."""
        if adapter:
            return adapter.get_connection()
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Ensure predictions table exists (now handled by schema.sql)."""
        # In Postgres mode, schema.sql handles this.
        if not adapter:
             with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        resident_id TEXT,
                        timestamp DATETIME,
                        room TEXT,
                        activity TEXT,
                        confidence REAL,
                        is_anomaly INTEGER,
                        UNIQUE(resident_id, timestamp, room)
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_resident_ts ON predictions (resident_id, timestamp)')
                conn.commit()

    def get_resident_dir(self, resident_id: str) -> Path:
        res_dir = self.processed_dir / resident_id
        res_dir.mkdir(parents=True, exist_ok=True)
        return res_dir

    def save_prediction_batch(self, resident_id: str, room: str, df: pd.DataFrame):
        """
        Save predictions to both SQLite (for Dashboard) and Excel (for User).
        Splits Excel into monthly sheets automatically.
        """
        if df.empty:
            return

        # 1. Save to SQLite ONLY (DB-First Strategy)
        self._save_to_sqlite(resident_id, room, df)
        
        # 2. Excel is now On-Demand via export tools.
        # self._save_to_excel(resident_id, room, df)

    def _save_to_sqlite(self, resident_id: str, room: str, df: pd.DataFrame):
        try:
            # Prepare data for SQL
            sql_df = df.copy()
            sql_df['resident_id'] = resident_id
            sql_df['room'] = room
            
            # Map columns
            # Map columns - PRIORITIZE predicted_activity
            if 'predicted_activity' in df.columns:
                 # If we have both, drop the original 'activity' first to avoid collision
                 if 'activity' in sql_df.columns:
                     sql_df = sql_df.drop(columns=['activity'])
                 
                 sql_df = sql_df.rename(columns={'predicted_activity': 'activity'})
            
            # Ensure timestamp and confidence map correctly
            # Note: If 'activity' was already there and no 'predicted_activity', it stays as is.
            
            # Handle anomaly if present
            sql_df['is_anomaly'] = 0
            if 'combined_anomaly' in sql_df.columns:
                sql_df['is_anomaly'] = sql_df['combined_anomaly'].astype(int)
            elif 'final_anomaly' in sql_df.columns:
                 sql_df['is_anomaly'] = sql_df['final_anomaly'].astype(int)

            # Keep only DB columns
            db_cols = ['resident_id', 'timestamp', 'room', 'activity', 'confidence', 'is_anomaly']
            sql_df = sql_df[db_cols]

            if adapter:
                # PostgreSQL Strategy: executemany with ON CONFLICT
                # Convert to list of tuples for executemany
                # Ensure timestamps are strings/objects compatible with adapter
                sql_df['timestamp'] = sql_df['timestamp'].astype(str)
                data = list(sql_df.itertuples(index=False, name=None))
                
                query = """
                    INSERT INTO predictions (resident_id, timestamp, room, activity, confidence, is_anomaly)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (resident_id, timestamp, room) DO UPDATE SET
                    activity = EXCLUDED.activity,
                    confidence = EXCLUDED.confidence,
                    is_anomaly = EXCLUDED.is_anomaly
                """
                
                with self.get_connection() as conn:
                    # Legacy adapter handles ? -> %s conversion
                    conn.executemany(query, data)
                    conn.commit()
            else:
                # SQLite Strategy (Fallback)
                with sqlite3.connect(self.db_path) as conn:
                    sql_df.to_sql('temp_predictions', conn, if_exists='replace', index=False)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO predictions (resident_id, timestamp, room, activity, confidence, is_anomaly)
                        SELECT resident_id, timestamp, room, activity, confidence, is_anomaly FROM temp_predictions
                    ''')
                    cursor.execute('DROP TABLE temp_predictions')
                    conn.commit()
            logger.info(f"Saved {len(df)} predictions to master for {resident_id} ({room})")
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")

    def _save_to_excel(self, resident_id: str, room: str, df: pd.DataFrame):
        """
        Appends data to an Excel file, creating new monthly sheets as needed.
        """
        excel_path = self.get_resident_dir(resident_id) / f"{resident_id}_yearly_report.xlsx"
        
        try:
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Group by Month/Year to handle batch processing spanning across months
            df['month_key'] = df['timestamp'].dt.strftime('%b %Y')
            
            # If file doesn't exist, create it. Otherwise load it.
            if excel_path.exists():
                mode = 'a'
                if_sheet_exists = 'overlay'
                engine = 'openpyxl'
            else:
                mode = 'w'
                if_sheet_exists = None
                engine = 'openpyxl'

            with pd.ExcelWriter(excel_path, engine=engine, mode=mode, if_sheet_exists=if_sheet_exists) as writer:
                for month, month_df in df.groupby('month_key'):
                    # Each room gets its own data block or we just append to a sheet?
                    # User asked for "one xls for one elderly", and "tab for each room".
                    # But we also have "365 days". If we have 3 rooms * 12 months = 36 tabs.
                    # Or we combine rooms into one sheet with a 'Room' column.
                    # Since the user said "a tab for each room", let's name sheets as "Room - Month"
                    sheet_name = f"{room} - {month}"
                    
                    # Truncate if too long (Excel limit 31 chars)
                    if len(sheet_name) > 31:
                        sheet_name = sheet_name[:31]

                    # If sheet exists, read existing and append (overlay logic)
                    startrow = 0
                    if mode == 'a' and sheet_name in writer.sheets:
                         # This is complex with overlay. Simplest: read existing, concat, write.
                         try:
                             existing_df = pd.read_excel(excel_path, sheet_name=sheet_name)
                             month_df = pd.concat([existing_df, month_df.drop(columns=['month_key'])]).drop_duplicates(subset=['timestamp'])
                             month_df = month_df.sort_values('timestamp')
                         except Exception:
                             pass
                    
                    month_df.drop(columns=['month_key']).to_excel(writer, sheet_name=sheet_name, index=False)
                    
            logger.info(f"Updated Excel report for {resident_id} in {excel_path}")
        except Exception as e:
            logger.error(f"Failed to save to Excel: {e}")

    def update_history_json(self, resident_id: str, filename: str, entry: Dict[str, Any], date_key: str = 'date', max_days: int = 365):
        """
        Appends an entry to a JSON history file. Defaults to 365-day retention.
        """
        res_dir = self.get_resident_dir(resident_id)
        filepath = res_dir / filename
        
        history = []
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    history = json.load(f)
                    if not isinstance(history, list):
                        # Some files like adl_trends have a dict wrapper
                        if 'daily_records' in history:
                            history = history['daily_records']
                        else:
                            history = []
            except Exception:
                history = []

        # Remove duplicate date
        history = [h for h in history if h.get(date_key) != entry.get(date_key)]
        history.append(entry)
        history.sort(key=lambda x: x.get(date_key, ""))
        
        # Retention
        if len(history) > max_days:
            history = history[-max_days:]
            
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=4)
        
        logger.info(f"Updated {filename} for {resident_id} (History: {len(history)} days)")
        return history
    def export_data_to_excel(self, resident_id: str, start_date: str = None, end_date: str = None, output_file: Path = None):
        """
        Export data from SQLite to Excel for a specific resident and date range.
        This enables 'On-Demand' reporting.
        """
        try:
            with self.get_connection() as conn:
                # Note: adapter connection works with read_sql if it mimics DBAPI connection
                query = "SELECT * FROM predictions WHERE resident_id = ?"
                params = [resident_id]
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)
                    
                query += " ORDER BY timestamp"
                
                df = query_to_dataframe(conn, query, params)
                
            if df.empty:
                logger.warning(f"No data found for resident {resident_id} in specified range.")
                return False
                
            # Convert timestamp back to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            invalid_ts = int(df['timestamp'].isna().sum())
            if invalid_ts:
                logger.warning(
                    "Export dropped %d row(s) with invalid timestamp for resident %s.",
                    invalid_ts,
                    resident_id,
                )
                df = df.dropna(subset=['timestamp'])
            if df.empty:
                logger.warning(f"No valid timestamp rows found for resident {resident_id}.")
                return False
            
            # Default output path if not provided
            if not output_file:
                date_str = datetime.now().strftime("%Y%m%d")
                output_file = self.get_resident_dir(resident_id) / f"{resident_id}_report_{date_str}.xlsx"
                
            # Create Excel writer
            # We can split by room into tabs
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 1. Master Sheet
                df.to_excel(writer, sheet_name='All Data', index=False)
                
                # 2. Room Sheets
                for room, room_df in df.groupby('room'):
                    sheet_name = str(room)[:31] # Excel limit
                    room_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
            logger.info(f"Exported {len(df)} rows to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
