
import logging
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, Dict

from elderlycare_v1_16.config.settings import DB_PATH, DEFAULT_SENSOR_COLUMNS
from config import get_room_config
try:
    from backend.utils.intelligence_db import coerce_timestamp_column, query_to_dataframe
except ImportError:
    from utils.intelligence_db import coerce_timestamp_column, query_to_dataframe

# Import adapter fallback logic
try:
    from backend.db.legacy_adapter import LegacyDatabaseAdapter
except ImportError:
    from elderlycare_v1_16.database import db as adapter
else:
    adapter = LegacyDatabaseAdapter()

logger = logging.getLogger(__name__)

def calculate_sequence_length(platform, room_name: str) -> int:
    """
    Centralized logic for calculating sequence length.
    Ensures consistency between training and prediction.
    
    Args:
        platform: The ElderlyCarePlatform instance (for time-based config)
        room_name: The room to calculate for
        
    Returns:
        int: The calculated sequence length in samples
    """
    if platform.enable_time_based_processing:
        # Use per-room config if available via get_room_config()
        # Note: platform.sequence_time_window might be a global default
        try:
            return get_room_config().calculate_seq_length(room_name)
        except Exception:
             # Fallback to platform's global calculation if room config fails
            from elderlycare_v1_16.preprocessing.resampling import calculate_samples_from_time
            return calculate_samples_from_time(
                platform.sequence_time_window,
                platform.data_interval
            )
    else:
        return 60 # Default fallback

def fetch_golden_samples(elder_id: str, room_name: str) -> Optional[pd.DataFrame]:
    """
    Fetch 'Golden Samples' - previously corrected rows from the database.
    These are rows where is_corrected=1.
    
    Returns a DataFrame with sensor columns and 'activity' label, or None if no samples found.
    """
    try:
        # DB_PATH check is only relevant if using SQLite, adapter handles abstraction
        # if not DB_PATH.exists() and not adapter.is_postgres(): return None
            
        with adapter.get_connection() as conn:
            query = """
                SELECT timestamp, activity_type as activity, record_date
                FROM adl_history
                WHERE elder_id = ? 
                  AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = LOWER(REPLACE(REPLACE(?, ' ', ''), '_', ''))
                  AND is_corrected = 1
                ORDER BY timestamp
            """
            golden_df = query_to_dataframe(conn, query, (elder_id, room_name))
            
            if golden_df.empty:
                logger.debug(f"No Golden Samples found for {elder_id}/{room_name}")
                return None
                
            logger.info(f"Found {len(golden_df)} Golden Samples for {elder_id}/{room_name}")
            golden_df = coerce_timestamp_column(
                golden_df,
                "timestamp",
                f"Golden samples {elder_id}/{room_name}",
            )
            if golden_df.empty:
                return None
            return golden_df
    except Exception as e:
        logger.warning(f"Failed to fetch Golden Samples: {e}")
        return None

def fetch_all_golden_samples(elder_id: str) -> Optional[pd.DataFrame]:
    """
    Fetch ALL Golden Samples for an elder across all rooms.
    Used for training augmentation.
    
    Returns a DataFrame with columns: room, timestamp, activity
    """
    try:
        with adapter.get_connection() as conn:
            query = """
                SELECT room, timestamp, activity_type as activity
                FROM adl_history
                WHERE elder_id = ? 
                  AND is_corrected = 1
                ORDER BY room, timestamp
            """
            golden_df = query_to_dataframe(conn, query, (elder_id,))
            
            if golden_df.empty:
                logger.debug(f"No Golden Samples found for {elder_id}")
                return None
                
            logger.info(f"Found {len(golden_df)} total Golden Samples for {elder_id}")
            golden_df = coerce_timestamp_column(
                golden_df,
                "timestamp",
                f"Golden samples {elder_id}/all_rooms",
            )
            if golden_df.empty:
                return None
            return golden_df
    except Exception as e:
        logger.warning(f"Failed to fetch all Golden Samples: {e}")
        return None

def fetch_sensor_window_from_db(elder_id: str, 
                                room_name: str, 
                                start_time: pd.Timestamp, 
                                end_time: pd.Timestamp) -> Optional[pd.DataFrame]:
    """
    Fetch raw sensor data for a specific time window directly from adl_history.
    This uses the 'sensor_features' JSON column populated in Beta 5.5.
    
    Args:
        elder_id: Resident ID
        room_name: Room name
        start_time: Start of window
        end_time: End of window (inclusive)
        
    Returns:
        DataFrame with sensor columns and timestamp, or None if insufficient/missing data.
    """
    try:
        # Normalize inputs
        start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        
        with adapter.get_connection() as conn:
            # Query for rows with valid sensor_features
            query = """
                SELECT timestamp, sensor_features 
                FROM adl_history
                WHERE elder_id = ? 
                  AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = LOWER(REPLACE(REPLACE(?, ' ', ''), '_', ''))
                  AND timestamp BETWEEN ? AND ?
                  AND sensor_features IS NOT NULL
                ORDER BY timestamp
            """
            df = query_to_dataframe(conn, query, (elder_id, room_name, start_str, end_str))
            
            if df.empty:
                return None

            df = coerce_timestamp_column(
                df,
                "timestamp",
                f"Sensor window {elder_id}/{room_name}",
            )
            if df.empty:
                return None
                
            # Parse JSON features
            import json
            
            # Optimized parsing: list comprehension is faster than apply for simple JSON
            sensor_data = []
            for _, row in df.iterrows():
                try:
                    features = json.loads(row['sensor_features'])
                    features['timestamp'] = row['timestamp']
                    sensor_data.append(features)
                except (json.JSONDecodeError, TypeError):
                    continue
            
            if not sensor_data:
                return None
                
            # Create DataFrame
            result_df = pd.DataFrame(sensor_data)
            
            # Ensure all default sensor columns exist
            for col in DEFAULT_SENSOR_COLUMNS:
                if col not in result_df.columns:
                    # Special handling for temporal features if missing in JSON
                    if col in ['hour_sin', 'hour_cos', 'day_period']:
                        continue # Will be generated later if needed
                    result_df[col] = 0.0
            
            # Sort by timestamp
            result_df = result_df.sort_values('timestamp').reset_index(drop=True)
            
            return result_df

    except Exception as e:
        logger.warning(f"Failed to fetch sensor window from DB: {e}")
        return None

def fetch_sensor_windows_batch(elder_id: str, 
                               room_name: str, 
                               windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> Optional[pd.DataFrame]:
    """
    Fetch raw sensor data for MULTIPLE time windows in a single query.
    This optimizes the N+1 problem when processing many corrections.
    """
    try:
        if not windows:
            return None
            
        with adapter.get_connection() as conn:
            # Construct dynamic query with multiple BETWEEN clauses
            conditions = []
            params = [elder_id, room_name]
            
            for start, end in windows:
                conditions.append("(timestamp BETWEEN ? AND ?)")
                params.append(start.strftime('%Y-%m-%d %H:%M:%S'))
                params.append(end.strftime('%Y-%m-%d %H:%M:%S'))
                
            where_clause = " OR ".join(conditions)
            
            query = f"""
                SELECT timestamp, sensor_features 
                FROM adl_history
                WHERE elder_id = ? 
                  AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = LOWER(REPLACE(REPLACE(?, ' ', ''), '_', ''))
                  AND ({where_clause})
                  AND sensor_features IS NOT NULL
                ORDER BY timestamp
            """
            
            # Use chunks if result is expected to be huge (optional optimization)
            # For now, standard read_sql is acceptable as windows are usually small (10min x N)
            df = query_to_dataframe(conn, query, params)
            
            if df.empty:
                return None

            df = coerce_timestamp_column(
                df,
                "timestamp",
                f"Sensor window batch {elder_id}/{room_name}",
            )
            if df.empty:
                return None
                
            # Parse JSON features (Efficiently)
            # Use list comprehension which is faster than DataFrame.apply(lambda...)
            sensor_cols = DEFAULT_SENSOR_COLUMNS
            
            # Pre-allocate list of dicts
            parsed_data = []
            
            # We want to fail gracefully if one row is bad, but keep the rest
            import json
            for ts, features_json in zip(df['timestamp'], df['sensor_features']):
                try:
                    f = json.loads(features_json)
                    f['timestamp'] = ts # Keep original timestamp from DB
                    parsed_data.append(f)
                except (json.JSONDecodeError, TypeError):
                    continue
            
            if not parsed_data:
                return None
                
            result_df = pd.DataFrame(parsed_data)
            
            # Fill missing columns with 0.0 (except temporal ones which we might calc later)
            for col in sensor_cols:
                if col not in result_df.columns:
                     if col in ['hour_sin', 'hour_cos', 'day_period']:
                         continue
                     result_df[col] = 0.0
            
            return result_df.sort_values('timestamp').reset_index(drop=True)

    except Exception as e:
        logger.warning(f"Failed to fetch batch sensor windows from DB: {e}")
        return None
