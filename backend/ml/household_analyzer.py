import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict
from processors.profile_processor import normalize_resident_home_context

# Configuration
from config import DB_PATH
logger = logging.getLogger(__name__)

class HouseholdAnalyzer:
    def __init__(self, db_path=None):
        # Use shared intelligence DB utility
        from utils.intelligence_db import (
            coerce_timestamp_column,
            get_intelligence_db,
            query_to_dataframe,
        )
        self._get_connection = get_intelligence_db
        self._query_to_dataframe = query_to_dataframe
        self._coerce_timestamp_column = coerce_timestamp_column
        self.db_path = db_path

    def get_config(self):
        """Fetch configuration from DB with defaults."""
        defaults = {
            'empty_home_silence_threshold_min': 15,
            'empty_home_ignore_rooms': [],
            'enable_empty_home_detection': True,
            'household_type': 'single'  # 'single' or 'double'
        }
        
        try:
            with self._get_connection() as conn:
                query = "SELECT key, value FROM household_config"
                df = self._query_to_dataframe(conn, query)
                config = defaults.copy()
                for _, row in df.iterrows():
                    key = row['key']
                    val = row['value']
                    # Parse values
                    if key == 'empty_home_silence_threshold_min':
                        config[key] = int(val)
                    elif key == 'empty_home_ignore_rooms':
                        try:
                            config[key] = json.loads(val)
                        except:
                            config[key] = []
                    elif key == 'enable_empty_home_detection':
                        config[key] = (val.lower() == 'true')
                    elif key == 'household_type':
                        config[key] = val.lower()
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return defaults

    def get_resident_home_context(self, elder_id: str) -> Dict[str, object]:
        """Load typed resident/home context from elder profile_data."""
        try:
            with self._get_connection() as conn:
                df = self._query_to_dataframe(
                    conn,
                    "SELECT profile_data FROM elders WHERE elder_id = ? LIMIT 1",
                    (elder_id,),
                )
            if df.empty:
                return normalize_resident_home_context({})
            raw_value = df.iloc[0].get("profile_data")
            if isinstance(raw_value, str):
                try:
                    payload = json.loads(raw_value)
                except Exception:
                    payload = {}
            elif isinstance(raw_value, dict):
                payload = raw_value
            else:
                payload = {}
            return normalize_resident_home_context(payload)
        except Exception as e:
            logger.warning("Failed to load resident/home context for %s: %s", elder_id, e)
            return normalize_resident_home_context({})

    def apply_conflict_resolution(self, df_min: pd.DataFrame) -> pd.DataFrame:
        """
        For DOUBLE households: Filter out activity that conflicts with Elder's primary location.
        
        Algorithm:
        1. For each minute, find the room with highest-confidence activity (the "Anchor").
        2. If another room ALSO shows activity, label it as 'secondary_person_noise'.
        3. This assumes the Elder cannot be in two places at once.
        
        Args:
            df_min: DataFrame with index=timestamp, columns=rooms, values=activity
            
        Returns:
            df_min with conflicting activities replaced by 'secondary_person_noise'
        """
        # We need confidence to determine anchor; if not available, use activity priority
        # For now, use simple priority: sleep > cooking > normal_use > inactive
        ACTIVITY_PRIORITY = {
            'sleep': 100,
            'nap': 90,
            'shower': 85,
            'cooking': 80,
            'toilet': 75,
            'bathroom_normal_use': 70,
            'kitchen_normal_use': 65,
            'bedroom_normal_use': 60,
            'livingroom_normal_use': 55,
            'room_normal_use': 50,
            'out': 10,  # Out is special - should not anchor
            'unoccupied': 0,
            'inactive': 0,
            'low_confidence': 0
        }
        
        def get_priority(activity):
            return ACTIVITY_PRIORITY.get(activity, 30)  # Default for unknown activities
        
        result_df = df_min.copy()
        
        for ts in result_df.index:
            row = result_df.loc[ts]
            
            # Find anchor room (highest priority activity excluding 'out' and 'inactive')
            anchor_room = None
            anchor_priority = -1
            
            for room in row.index:
                activity = row[room]
                priority = get_priority(activity)
                
                # Skip 'out' and 'inactive' as anchor candidates
                if activity in ['out', 'inactive', 'low_confidence', None]:
                    continue
                    
                if priority > anchor_priority:
                    anchor_priority = priority
                    anchor_room = room
            
            # If we found an anchor, mark all OTHER active rooms as noise
            if anchor_room and anchor_priority > 0:
                for room in row.index:
                    if room == anchor_room:
                        continue  # Keep anchor as-is
                    
                    activity = row[room]
                    # If this room also has meaningful activity, it's conflict -> noise
                    if activity not in ['inactive', 'low_confidence', 'out', None]:
                        result_df.loc[ts, room] = 'secondary_person_noise'
        
        logger.info(f"Conflict Resolution applied. Anchor rooms identified for {len(result_df)} minutes.")
        return result_df


    def analyze_day(self, elder_id: str, date_str: str):
        """
        Run global analysis for a specific day.
        1. Fetch all room predictions.
        2. Align timelines.
        3. Apply Hybrid Logic (Entrance + Silence).
        4. Save to DB.
        
        Args:
            elder_id: Elder Identifier (non-empty string)
            date_str: Date of analysis (YYYY-MM-DD format)
            
        Raises:
            ValueError: If input parameters are invalid.
        """
        # === EARLY INPUT VALIDATION ===
        if not elder_id or not isinstance(elder_id, str):
            raise ValueError(f"elder_id must be a non-empty string, got: {type(elder_id).__name__}")
        if not date_str or not isinstance(date_str, str):
            raise ValueError(f"date_str must be a non-empty string, got: {type(date_str).__name__}")
        try:
            from datetime import datetime
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"date_str must be in YYYY-MM-DD format, got: '{date_str}'")
        
        config = self.get_config()
        resident_context = self.get_resident_home_context(elder_id)
        context_household_type = resident_context.get("household_type")
        context_helper_presence = resident_context.get("helper_presence")
        has_secondary_people = (
            context_household_type == "multi_resident"
            or context_helper_presence in {"scheduled", "live_in"}
        )
        if not config['enable_empty_home_detection']:
            logger.info("Empty Home detection disabled in config.")
            return

        logger.info(f"Running Household Analysis for {elder_id} on {date_str}")
        
        try:
            with self._get_connection() as conn:
                # 1. Fetch all ADL history for the day
                query = """
                    SELECT timestamp, room, activity_type, confidence 
                    FROM adl_history 
                    WHERE elder_id = ? AND record_date = ?
                    ORDER BY timestamp
                """
                df = self._query_to_dataframe(conn, query, (elder_id, date_str))
                
                if df.empty:
                    logger.warning(f"No data found for {date_str}")
                    return

                # 2. Resample to 1-minute grid for alignment
                df = self._coerce_timestamp_column(
                    df,
                    "timestamp",
                    f"Household Analysis {elder_id}/{date_str}",
                )
                if df.empty:
                    logger.warning(f"No valid timestamp rows for {elder_id}/{date_str}")
                    return

                df = df.sort_values('timestamp')
                df = df.set_index('timestamp')
                
                # Pivot: Columns = Rooms, Values = Activity
                # We take the 'mode' (most frequent) activity per minute or just last
                # Because sensors are high freq, 1 minute is a good aggregation
                
                # Group by minute and room
                df_min = df.groupby([pd.Grouper(freq='1min'), 'room'])['activity_type'].agg(lambda x: x.mode()[0] if not x.empty else 'inactive').unstack(fill_value='inactive')
                
                # If df_min is empty or malformed
                if df_min.empty:
                    return

                # 2b. Apply Conflict Resolution for DOUBLE households
                if config.get('household_type', 'single') == 'double' or has_secondary_people:
                    logger.info("Applying Double-Household Conflict Resolution...")
                    df_min = self.apply_conflict_resolution(df_min)

                # 3. Apply Logic
                results = []
                
                silence_threshold = config['empty_home_silence_threshold_min']
                ignored_rooms = config['empty_home_ignore_rooms']
                
                # Rolling window for silence check might be needed, but simpler first:
                # Iterate row by row (minute by minute)
                
                # Helper to check if a room is 'active' (not resting/sleeping)
                def is_active(act):
                    # Sleep/nap activities indicate the person is home but resting, not 'active'
                    resting_activities = ['sleep', 'nap', 'inactive', 'unoccupied', 'low_confidence', 'unknown', None]
                    return act not in resting_activities

                for ts, row in df_min.iterrows():
                    # ROW is a Series with index = room names, value = activity
                    
                    # A. Entrance Signal
                    # Check if 'entrance' column exists and is 'out'
                    entrance_out = False
                    if 'entrance' in row.index and row['entrance'] == 'out':
                        entrance_out = True
                        
                    # B. House Silence
                    # Check all OTHER rooms (excluding ignored ones)
                    rest_active = False
                    active_rooms = []
                    
                    for room in row.index:
                        if room == 'entrance': continue
                        if room in ignored_rooms: continue
                        
                        if is_active(row[room]):
                            rest_active = True
                            active_rooms.append(room)
                            
                    # DECISION LOGIC
                    state = 'home_active' # Default
                    evidence = {}
                    
                    if entrance_out:
                        if not rest_active:
                            state = 'empty_home'
                            evidence = {
                                "trigger": "entrance_out",
                                "silence": True,
                                "context_household_type": context_household_type,
                                "context_helper_presence": context_helper_presence,
                                "context_topology": (
                                    resident_context.get("layout", {}).get("topology")
                                    if isinstance(resident_context.get("layout"), dict)
                                    else None
                                ),
                                "context_status": resident_context.get("status"),
                            }
                        else:
                            # Entrance says out, but someone is in kitchen? 
                            # Conflict! For now, safer to assume Home Active (maybe guest, or mis-classification)
                            state = 'home_active' 
                            evidence = {
                                "trigger": "conflict",
                                "active_rooms": active_rooms,
                                "context_household_type": context_household_type,
                                "context_helper_presence": context_helper_presence,
                            }
                    else:
                        # Entrance NOT out.
                        if not rest_active:
                            state = 'home_quiet' # Everyone sleeping or reading
                        else:
                            state = 'home_active'
                            
                    results.append({
                        'timestamp': ts,
                        'state': state,
                        'evidence': json.dumps(evidence)
                    })
                    
                # 4. Post-Processing: Silence Threshold (Debounce)
                # We need to enforce that 'empty_home' must persist for X minutes to be valid?
                # Or does the rule "Entrance=Out" imply immediate effect?
                # Usually Entrance=Out + Silence is immediate.
                # But 'home_quiet' turning into 'empty_home' without entrance signal?
                # Using the Hybrid rule: We ONLY flag empty_home if Entrance is involved as the primary key.
                # If Entrance is NOT out, but house is silent -> 'home_quiet'.
                
                # Let's clean up the gathered results
                res_df = pd.DataFrame(results)
                
                # 5. Save State Stream (household_behavior)
                # First delete existing for this day
                del_query = "DELETE FROM household_behavior WHERE elder_id = ? AND date(timestamp) = ?"
                conn.execute(del_query, (elder_id, date_str))
                
                # Bulk Insert
                db_data = []
                for _, r in res_df.iterrows():
                    db_data.append((
                        elder_id, 
                        r['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), 
                        r['state'], 
                        1.0, # Confidence
                        r['evidence']
                    ))
                    
                conn.executemany("""
                    INSERT INTO household_behavior (elder_id, timestamp, state, confidence, supporting_evidence)
                    VALUES (?, ?, ?, ?, ?)
                """, db_data)
                
                # 6. Generate Segments (household_segments)
                # Group consecutive states
                res_df['group'] = (res_df['state'] != res_df['state'].shift()).cumsum()
                
                segments = []
                for _, group in res_df.groupby('group'):
                    state = group['state'].iloc[0]
                    start = group['timestamp'].min()
                    end = group['timestamp'].max() + pd.Timedelta(minutes=1) # +1 min duration
                    duration = (end - start).total_seconds() / 60.0
                    
                    segments.append((
                        elder_id,
                        state,
                        start.strftime('%Y-%m-%d %H:%M:%S'),
                        end.strftime('%Y-%m-%d %H:%M:%S'),
                        round(duration, 2)
                    ))
                    
                # Delete old segments
                conn.execute("DELETE FROM household_segments WHERE elder_id = ? AND date(start_time) = ?", (elder_id, date_str))
                
                conn.executemany("""
                    INSERT INTO household_segments (elder_id, state, start_time, end_time, duration_minutes)
                    VALUES (?, ?, ?, ?, ?)
                """, segments)
                
                logger.info(f"Household Analysis Complete. Generated {len(segments)} segments.")
                
        except Exception as e:
            logger.error(f"Household Analysis Failed: {e}", exc_info=True)
