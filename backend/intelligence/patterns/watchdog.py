"""
PatternWatchdog - Routine Anomaly Detection

This module learns an elder's "normal" daily routine from historical data
and detects daily deviations that might indicate health/cognitive decline.

Key Metrics Tracked:
- Wake time (first activity after long sleep)
- Sleep time (start of night sleep)
- Total activity duration
- Bathroom frequency
- Room visit patterns
"""

import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from config import DB_PATH as DEFAULT_DB_PATH


class PatternWatchdog:
    """
    Learns routine patterns and detects anomalies.
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
        
        # Configuration
        self.profile_days = 30  # Days to use for building profile
        self.anomaly_z_threshold = 2.0  # Z-score threshold for anomaly
        
    def analyze_day(self, elder_id: str, date_str: str) -> List[Dict]:
        """
        Analyze a specific day for routine anomalies.
        
        Args:
            elder_id: Elder ID
            date_str: Date to analyze (YYYY-MM-DD)
            
        Returns:
            List of detected anomalies
        """
        logger.info(f"[PatternWatchdog] Analyzing {elder_id} for {date_str}")
        
        try:
            with self._get_connection() as conn:
                # 1. Build routine profile from history
                profile = self._build_profile(conn, elder_id, date_str)
                
                if not profile:
                    logger.info(f"Not enough history to build profile for {elder_id}")
                    return []
                
                # 2. Extract today's metrics
                today_metrics = self._extract_daily_metrics(conn, elder_id, date_str)
                
                if not today_metrics:
                    logger.info(f"No data for {elder_id} on {date_str}")
                    return []
                
                # 3. Compare and detect anomalies
                anomalies = self._detect_anomalies(profile, today_metrics)
                
                # 4. Save anomalies to database
                self._save_anomalies(conn, elder_id, date_str, anomalies)
                
                logger.info(f"[PatternWatchdog] Detected {len(anomalies)} anomalies for {elder_id}")
                return anomalies
                
        except Exception as e:
            logger.error(f"[PatternWatchdog] Error: {e}", exc_info=True)
            return []
    
    def _build_profile(self, conn, elder_id: str, reference_date: str) -> Optional[Dict]:
        """
        Build a routine profile from historical data.
        Returns statistics for key metrics.
        """
        # Get data from the past N days (excluding reference date)
        ref_date = datetime.strptime(reference_date, '%Y-%m-%d')
        start_date = (ref_date - timedelta(days=self.profile_days)).strftime('%Y-%m-%d')
        
        query = """
            SELECT record_date, timestamp, room, activity_type
            FROM adl_history
            WHERE elder_id = ? 
              AND record_date >= ? 
              AND record_date < ?
              AND activity_type NOT IN ('inactive', 'low_confidence', 'unknown')
            ORDER BY timestamp
        """
        
        df = self._query_to_dataframe(conn, query, (elder_id, start_date, reference_date))

        if df.empty or df['record_date'].nunique() < 7:
            # Need at least 7 days of history
            return None

        df = self._coerce_timestamp_column(
            df,
            "timestamp",
            f"Pattern profile {elder_id}/{reference_date}",
        )
        if df.empty:
            return None
        
        # Calculate daily metrics for each historical day
        daily_metrics = []
        for date, day_df in df.groupby('record_date'):
            metrics = self._calculate_day_metrics(day_df)
            if metrics:
                daily_metrics.append(metrics)
        
        if len(daily_metrics) < 5:
            return None
        
        # Build profile statistics
        metrics_df = pd.DataFrame(daily_metrics)
        
        profile = {}
        for col in metrics_df.columns:
            values = metrics_df[col].dropna()
            if len(values) > 0:
                profile[col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()) if len(values) > 1 else 0.0,
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'count': len(values)
                }
        
        return profile
    
    def _calculate_day_metrics(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Calculate key metrics for a single day.
        """
        if df.empty:
            return None
        
        df = df.sort_values('timestamp')
        
        metrics = {}
        
        # 1. Wake time (first activity hour)
        first_activity = df['timestamp'].min()
        metrics['wake_hour'] = first_activity.hour + first_activity.minute / 60.0
        
        # 2. Last activity hour (proxy for sleep time)
        last_activity = df['timestamp'].max()
        metrics['last_activity_hour'] = last_activity.hour + last_activity.minute / 60.0
        
        # 3. Total active duration (minutes)
        # Approximate by counting events * average interval
        metrics['active_events'] = len(df)
        
        # 4. Bathroom visit count
        bathroom_activities = df[df['room'].str.lower().str.contains('bathroom', na=False)]
        metrics['bathroom_visits'] = len(bathroom_activities)
        
        # 5. Unique rooms visited
        metrics['rooms_visited'] = df['room'].nunique()
        
        # 6. Sleep periods (bedroom activities labeled as sleep/nap)
        sleep_activities = df[df['activity_type'].str.lower().isin(['sleep', 'nap'])]
        metrics['sleep_events'] = len(sleep_activities)
        
        return metrics
    
    def _extract_daily_metrics(self, conn, elder_id: str, date_str: str) -> Optional[Dict]:
        """
        Extract metrics for a specific day.
        """
        query = """
            SELECT timestamp, room, activity_type
            FROM adl_history
            WHERE elder_id = ? AND record_date = ?
              AND activity_type NOT IN ('inactive', 'low_confidence', 'unknown')
            ORDER BY timestamp
        """
        
        df = self._query_to_dataframe(conn, query, (elder_id, date_str))
        
        if df.empty:
            return None
        
        df = self._coerce_timestamp_column(
            df,
            "timestamp",
            f"Pattern daily metrics {elder_id}/{date_str}",
        )
        if df.empty:
            return None
        return self._calculate_day_metrics(df)
    
    def _detect_anomalies(self, profile: Dict, today: Dict) -> List[Dict]:
        """
        Compare today's metrics against profile and detect anomalies.
        """
        anomalies = []
        
        anomaly_checks = [
            ('wake_hour', 'Late Wakeup', 'Early Wakeup', 
             lambda z: z > self.anomaly_z_threshold, lambda z: z < -self.anomaly_z_threshold),
            ('last_activity_hour', 'Late Night Activity', 'Early Sleep', 
             lambda z: z > self.anomaly_z_threshold, lambda z: z < -self.anomaly_z_threshold),
            ('bathroom_visits', 'Increased Bathroom Visits', 'Reduced Bathroom Visits',
             lambda z: z > self.anomaly_z_threshold, lambda z: z < -self.anomaly_z_threshold),
            ('active_events', 'Unusually High Activity', 'Reduced Activity',
             lambda z: z > self.anomaly_z_threshold, lambda z: z < -self.anomaly_z_threshold),
            ('rooms_visited', 'Increased Wandering', 'Limited Movement',
             lambda z: z > self.anomaly_z_threshold, lambda z: z < -self.anomaly_z_threshold),
        ]
        
        for metric, high_label, low_label, high_check, low_check in anomaly_checks:
            if metric not in profile or metric not in today:
                continue
            
            p = profile[metric]
            observed = today[metric]
            
            # Calculate Z-score
            if p['std'] > 0:
                z_score = (observed - p['mean']) / p['std']
            else:
                # No variation in history - any difference is notable
                z_score = 0 if observed == p['mean'] else (2.5 if observed > p['mean'] else -2.5)
            
            # Check for anomaly
            if high_check(z_score):
                anomalies.append({
                    'anomaly_type': high_label.lower().replace(' ', '_'),
                    'anomaly_score': min(abs(z_score) / 3.0, 1.0),  # Normalize to 0-1
                    'description': high_label,
                    'baseline_value': f"{p['mean']:.1f} ± {p['std']:.1f}",
                    'observed_value': f"{observed:.1f}",
                    'z_score': z_score
                })
            elif low_check(z_score):
                anomalies.append({
                    'anomaly_type': low_label.lower().replace(' ', '_'),
                    'anomaly_score': min(abs(z_score) / 3.0, 1.0),
                    'description': low_label,
                    'baseline_value': f"{p['mean']:.1f} ± {p['std']:.1f}",
                    'observed_value': f"{observed:.1f}",
                    'z_score': z_score
                })
        
        # Sort by score (most significant first)
        anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)
        
        return anomalies
    
    def _save_anomalies(self, conn, elder_id: str, date_str: str, anomalies: List[Dict]):
        """
        Save detected anomalies to database.
        """
        # Delete existing anomalies for this date
        conn.execute(
            "DELETE FROM routine_anomalies WHERE elder_id = ? AND detection_date = ?",
            (elder_id, date_str)
        )
        
        for anomaly in anomalies:
            conn.execute("""
                INSERT INTO routine_anomalies 
                (elder_id, detection_date, anomaly_type, anomaly_score, 
                 description, baseline_value, observed_value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                elder_id,
                date_str,
                anomaly['anomaly_type'],
                round(anomaly['anomaly_score'], 3),
                anomaly['description'],
                anomaly['baseline_value'],
                anomaly['observed_value']
            ))
        
        conn.commit()
        logger.info(f"Saved {len(anomalies)} anomalies for {elder_id} on {date_str}")


def run_pattern_analysis(elder_id: str, date_str: str, db_path=None) -> List[Dict]:
    """
    Run pattern analysis for a specific elder and date.
    Entry point for integration with process_data.py
    """
    watchdog = PatternWatchdog(db_path=db_path)
    return watchdog.analyze_day(elder_id, date_str)
