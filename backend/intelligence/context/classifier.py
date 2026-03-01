"""
ContextClassifier - ML-Based Household Logic

This module classifies the overall "Context" of the household (e.g., Empty, Home Alone, Guest Present)
using aggregated features from ADL history.

It replaces the rigid if-then rules of previous betas with a trainable classifier approach.
"""

import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Model paths
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "context"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "context_classifier.pkl"

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "residents_master_data.db"

class ContextFeatureEngine:
    """Methods to extract features for context classification."""
    
    @staticmethod
    def extract_features(df_window: pd.DataFrame) -> Dict:
        """
        Extract features from a dataframe of ADL events (typically 15-30 min window).
        """
        if df_window.empty:
            return {
                'event_count': 0,
                'unique_rooms': 0,
                'entrance_activity': 0,
                'is_night': 0,
                'duration_span': 0
            }
        
        # 1. Event intensity
        event_count = len(df_window)
        unique_rooms = df_window['room'].nunique()
        
        # 2. Specific room activity
        entrance_activity = len(df_window[df_window['room'].str.lower() == 'entrance'])
        
        # 3. Time features
        timestamps = pd.to_datetime(df_window['timestamp'])
        hr = timestamps.iloc[0].hour
        is_night = 1 if (hr >= 23 or hr < 6) else 0
        
        # 4. Duration span (how long was there activity in this window?)
        duration_span = (timestamps.max() - timestamps.min()).total_seconds() / 60.0
        
        return {
            'event_count': event_count,
            'unique_rooms': unique_rooms,
            'entrance_activity': entrance_activity,
            'is_night': is_night,
            'duration_span': duration_span
        }

class ContextClassifier:
    """
    Classifies household context based on time windows.
    """
    
    def __init__(self, db_path=None):
        # Import shared database utility
        from utils.intelligence_db import (
            coerce_timestamp_column,
            get_intelligence_db,
            query_to_dataframe,
        )
        self._get_connection = get_intelligence_db
        self._query_to_dataframe = query_to_dataframe
        self._coerce_timestamp_column = coerce_timestamp_column
        self.model = self._load_model()
        self.window_size_min = 30
    
    def _load_model(self):
        if MODEL_PATH.exists():
            try:
                return joblib.load(MODEL_PATH)
            except Exception as e:
                logger.warning(f"Failed to load context model: {e}")
                return None
        return None

    def analyze_day(self, elder_id: str, date_str: str) -> List[Dict]:
        """
        Segment the day into windows and classify context.
        """
        logger.info(f"[ContextClassifier] Analyzing {elder_id} for {date_str}")
        
        try:
            with self._get_connection() as conn:
                # Get all events for the day
                query = """
                    SELECT timestamp, room, activity_type, confidence
                    FROM adl_history
                    WHERE elder_id = ? AND record_date = ?
                    ORDER BY timestamp
                """
                df = self._query_to_dataframe(conn, query, (elder_id, date_str))
                
                if df.empty:
                    return []
                
                df = self._coerce_timestamp_column(
                    df,
                    "timestamp",
                    f"ContextClassifier {elder_id}/{date_str}",
                )
                if df.empty:
                    return []
                
                # Create time windows (00:00 to 23:59)
                start_of_day = pd.to_datetime(date_str)
                windows = []
                
                current_time = start_of_day
                end_of_day = start_of_day + timedelta(days=1)
                
                episodes = []
                
                while current_time < end_of_day:
                    next_time = current_time + timedelta(minutes=self.window_size_min)
                    
                    # Filter data for this window
                    mask = (df['timestamp'] >= current_time) & (df['timestamp'] < next_time)
                    window_df = df[mask]
                    
                    # Extract features
                    features = ContextFeatureEngine.extract_features(window_df)
                    
                    # Predict
                    label = self._predict_context(features)
                    
                    episodes.append({
                        'start_time': current_time,
                        'end_time': next_time,
                        'context_label': label,
                        'confidence': 1.0 if self.model is None else 0.8, # Placeholder confidence
                        'features': features
                    })
                    
                    current_time = next_time
                
                # Consolidate adjacent episodes with same label
                consolidated = self._consolidate_episodes(episodes)
                
                # Save to DB
                self._save_episodes(conn, elder_id, consolidated)
                
                return consolidated
                
        except Exception as e:
            logger.error(f"[ContextClassifier] Error: {e}", exc_info=True)
            return []
    
    def _predict_context(self, features: Dict) -> str:
        """
        Predict context label. Uses ML model if available, else heuristics.
        """
        if self.model:
            # TODO: Convert features dict to array and predict
            pass
        
        # Fallback Heuristics (Bootstrapping)
        # 1. Empty detection
        if features['event_count'] == 0:
            return 'Empty'
        
        # 2. High activity = Potential Guest or Just Active
        if features['unique_rooms'] >= 3 and features['event_count'] > 50:
            return 'High_Activity' 
            
        # 3. Night inactive
        if features['is_night'] == 1 and features['event_count'] < 5:
            return 'Sleep_Quiet'
            
        return 'Normal_Activity'

    def _consolidate_episodes(self, episodes: List[Dict]) -> List[Dict]:
        """Merge consecutive episodes with same label."""
        if not episodes:
            return []
            
        merged = []
        current = episodes[0].copy()
        
        for next_ep in episodes[1:]:
            if next_ep['context_label'] == current['context_label']:
                # Extend current
                current['end_time'] = next_ep['end_time']
                # Aggregate features / confidence if needed
            else:
                merged.append(current)
                current = next_ep.copy()
        
        merged.append(current)
        return merged

    def _save_episodes(self, conn, elder_id: str, episodes: List[Dict]):
        """Save to context_episodes table."""
        # Clean old for this day? 
        # For simplicity, let's just insert. Ideally we delete for this time range first.
        # But since we do whole day analysis, careful about overwrites.
        
        if not episodes:
            return

        # Simple approach: Delete overlap based on start/end range of the whole set
        min_start = episodes[0]['start_time']
        max_end = episodes[-1]['end_time']
        
        conn.execute("""
            DELETE FROM context_episodes 
            WHERE elder_id = ? AND start_time >= ? AND end_time <= ?
        """, (elder_id, min_start.isoformat(), max_end.isoformat()))
        
        for ep in episodes:
            import json
            conn.execute("""
                INSERT INTO context_episodes
                (elder_id, start_time, end_time, context_label, confidence, features_used, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                elder_id,
                ep['start_time'].isoformat(),
                ep['end_time'].isoformat(),
                ep['context_label'],
                ep['confidence'],
                json.dumps(ep['features']),
                'v0.1_heuristic'
            ))
        conn.commit()

def run_context_analysis(elder_id: str, date_str: str, db_path=None) -> List[Dict]:
    """Entry point."""
    classifier = ContextClassifier(db_path=db_path)
    return classifier.analyze_day(elder_id, date_str)
