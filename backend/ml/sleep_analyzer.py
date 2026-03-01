"""
SleepAnalyzer - Centralized Sleep Analysis Module for Beta 5 (1000 POC).

This module consolidates all sleep analysis logic previously scattered across
process_data.py, backfill_analysis.py, and regenerate_sleep.py.

Responsibility:
- Analyzes raw activity data to extract sleep metrics.
- Determines sleep stages based on motion heuristics.
- Calculates quality scores using standardized formulas.
- Generates human-readable insights.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

from elderlycare_v1_16.config.settings import SLEEP_CONFIG

logger = logging.getLogger(__name__)


class SleepAnalyzer:
    """
    Centralized Sleep Analysis Module for Beta 5 (1000 POC).
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize SleepAnalyzer with configuration.
        
        Args:
            config: Optional configuration dictionary overrides.
        """
        # Load base config from settings, allow runtime overrides
        self.config = SLEEP_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.min_sleep_duration_min = self.config.get('min_sleep_duration_min', 60)
        self.motion_thresholds = self.config.get('motion_thresholds', {
            'deep': 0.05,
            'light': 0.2,
            'rem': 0.5
        })
        self.duration_penalties = self.config.get('duration_penalties', {
            'short_severe': 0.7,
            'short_moderate': 0.9,
            'long_moderate': 0.95,
            'optimal': 1.0
        })

    def analyze_day(self, elder_id: str, date_str: str, prediction_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze a single day of data for sleep patterns.
        
        Args:
            elder_id: Elder Identifier (non-empty string)
            date_str: Date of analysis (YYYY-MM-DD format)
            prediction_df: DataFrame containing 'predicted_activity' and optionally 'motion'/'timestamp'
            
        Returns:
            Dictionary containing full sleep analysis results, or None if no sleep found.
            
        Raises:
            ValueError: If input parameters are invalid.
        """
        # === EARLY INPUT VALIDATION ===
        # 1. Validate elder_id
        if not elder_id or not isinstance(elder_id, str):
            raise ValueError(f"elder_id must be a non-empty string, got: {type(elder_id).__name__}")
        
        # 2. Validate date_str format
        if not date_str or not isinstance(date_str, str):
            raise ValueError(f"date_str must be a non-empty string, got: {type(date_str).__name__}")
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"date_str must be in YYYY-MM-DD format, got: '{date_str}'")
        
        # 3. Validate prediction_df structure
        if not isinstance(prediction_df, pd.DataFrame):
            raise ValueError(f"prediction_df must be a DataFrame, got: {type(prediction_df).__name__}")
        
        if prediction_df.empty:
            logger.warning(f"Empty prediction data for {elder_id} on {date_str}")
            return None
            
        if 'predicted_activity' not in prediction_df.columns:
            raise ValueError(f"prediction_df missing required column 'predicted_activity'. Columns: {list(prediction_df.columns)}")

        # 1. Extract Sleep Events
        sleep_df = self._extract_sleep_events(prediction_df)
        if sleep_df.empty:
            logger.info(f"No sleep events found for {elder_id} on {date_str}")
            return None
            
        # 2. Add computed columns (motion, timestamp objects)
        sleep_df = self._preprocess_data(sleep_df)
        
        # 3. Calculate Core Metrics
        metrics = self._calculate_metrics(sleep_df)
        
        # 4. Determine Sleep Stages
        metrics['stages_summary'] = self._determine_stages(sleep_df)
        
        # 5. Calculate Efficiency
        metrics['sleep_efficiency'] = self._calculate_efficiency(sleep_df, metrics['total_duration_hours'])
        
        # 6. Compute Quality Score
        metrics['quality_score'] = self._compute_quality_score(metrics['sleep_efficiency'], metrics['total_duration_hours'])
        
        # 7. Generate Insights
        metrics['insights'] = self._generate_insights(metrics)
        
        return {
            'elder_id': elder_id,
            'analysis_date': date_str,
            **metrics
        }

    def analyze_from_predictions(self, elder_id: str, prediction_results: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        Analyze sleep from a dict of room -> DataFrame predictions.
        
        This is a convenience method for integration with process_data.py.
        """
        # Merge all rooms into one DataFrame
        all_events = []
        record_date = None
        
        for room_name, pred_df in prediction_results.items():
            if not isinstance(pred_df, pd.DataFrame):
                continue
            if 'predicted_activity' in pred_df.columns:
                sleep_mask = pred_df['predicted_activity'].astype(str).str.contains('sleep|nap', case=False, na=False)
                room_sleep = pred_df[sleep_mask].copy()
                if not room_sleep.empty:
                    room_sleep['room'] = room_name
                    all_events.append(room_sleep)
                    
                    # Capture record_date from first valid timestamp
                    if record_date is None and 'timestamp' in pred_df.columns:
                        record_date = pd.to_datetime(pred_df['timestamp'].iloc[0]).strftime('%Y-%m-%d')
        
        if not all_events:
            return None
            
        combined_df = pd.concat(all_events, ignore_index=True)
        combined_df['predicted_activity'] = 'sleep'  # Ensure column exists for analyze_day
        
        date_str = record_date or datetime.now().strftime('%Y-%m-%d')
        return self.analyze_day(elder_id, date_str, combined_df)

    def _extract_sleep_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data for sleep/nap events."""
        mask = df['predicted_activity'].astype(str).str.contains('sleep|nap', case=False, na=False)
        return df[mask].copy()

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure timestamps are datetime and motion data exists."""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
        # Motion fallback
        motion_col = next((c for c in df.columns if c.lower() == 'motion'), None)
        if not motion_col:
            df['motion'] = 0.1  # Default low motion for sleep
        else:
            df['motion'] = df[motion_col].fillna(0.1)
            
        return df

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate duration and basic counts."""
        if 'timestamp' in df.columns and len(df) > 1:
            # Calculate gaps between readings
            # Default gap for first record is 10 seconds (in minutes)
            intervals = df['timestamp'].diff().dt.total_seconds().div(60).fillna(10 / 60.0)
            # Only count intervals <= 60 mins as continuous sleep (handle fragmented naps)
            total_minutes = intervals[intervals <= 60].sum()
        else:
            # Fallback for single record or no timestamp (assume 10s per record)
            total_minutes = len(df) * (10 / 60.0)
            
        return {
            'total_duration_hours': round(total_minutes / 60.0, 1),
            'record_count': len(df)
        }

    def _determine_stages(self, df: pd.DataFrame) -> Dict[str, float]:
        """Classify sleep stages based on motion intensity."""
        # Check if we should use heuristics (missing motion or low variance default)
        use_heuristics = False
        if 'motion' not in df.columns:
            use_heuristics = True
        else:
            # If motion is constant (e.g. default 0.1) or all -1 (missing), use heuristics
            if df['motion'].nunique() <= 1:
                use_heuristics = True
        
        if use_heuristics:
            logger.info("Using heuristic sleep stages due to missing/invalid motion data.")
            fallback = self.config.get('fallback_stage_ratios', {
                'Light': 0.55, 'Deep': 0.15, 'REM': 0.20, 'Awake': 0.10
            })
            
            # Assign artificial stages to the dataframe for completeness (optional, simple distribution)
            # For exact percentage matching, we'd need to shuffle, but for metrics summary we just return the dict.
            # Let's assign 'Light' as dominant placeholder to df for efficiency calc logic
            df['sleep_stage'] = 'Light' 
            
            return {
                'Deep': float(fallback['Deep'] * 100),
                'Light': float(fallback['Light'] * 100),
                'REM': float(fallback['REM'] * 100),
                'Awake': float(fallback['Awake'] * 100)
            }
            
        conditions = [
            (df['motion'] < self.motion_thresholds['deep']),
            (df['motion'] >= self.motion_thresholds['deep']) & (df['motion'] < self.motion_thresholds['light']),
            (df['motion'] >= self.motion_thresholds['light']) & (df['motion'] < self.motion_thresholds['rem']),
            (df['motion'] >= self.motion_thresholds['rem'])
        ]
        choices = ['Deep', 'Light', 'REM', 'Awake']
        
        # Assign stage to each record
        df['sleep_stage'] = np.select(conditions, choices, default='Light')
        
        # Calculate percentages
        stage_counts = df['sleep_stage'].value_counts(normalize=True) * 100
        return {
            'Deep': float(round(stage_counts.get('Deep', 0.0), 1)),
            'Light': float(round(stage_counts.get('Light', 0.0), 1)),
            'REM': float(round(stage_counts.get('REM', 0.0), 1)),
            'Awake': float(round(stage_counts.get('Awake', 0.0), 1))
        }

    def _calculate_efficiency(self, df: pd.DataFrame, total_hours: float) -> float:
        """Calculate sleep efficiency (Non-Awake time / Total time)."""
        if total_hours <= 0: 
            return 0.0
            
        if 'timestamp' in df.columns and len(df) > 1:
            intervals = df['timestamp'].diff().dt.total_seconds().div(60).fillna(10 / 60.0)
            # Sum time where stage is NOT Awake
            non_awake_mask = (df['sleep_stage'] != 'Awake')
            non_awake_minutes = intervals[non_awake_mask & (intervals <= 60)].sum()
        else:
            non_awake_minutes = (df['sleep_stage'] != 'Awake').sum() * (10 / 60.0)
            
        total_minutes = total_hours * 60
        if total_minutes == 0:
            return 0.0
        
        efficiency = non_awake_minutes / total_minutes
        return round(min(1.0, max(0.0, efficiency)), 2)

    def _compute_quality_score(self, efficiency: float, duration_hours: float) -> int:
        """
        Compute Quality Score (0-100).
        Formula: (Efficiency * 100) * Duration_Factor
        """
        base_score = efficiency * 100
        
        # Duration Penalties
        if duration_hours < 5:
            duration_factor = self.duration_penalties['short_severe']
        elif duration_hours < 7:
            duration_factor = self.duration_penalties['short_moderate']
        elif duration_hours > 9:
            duration_factor = self.duration_penalties['long_moderate']
        else:
            duration_factor = self.duration_penalties['optimal']
            
        return int(base_score * duration_factor)

    def _generate_insights(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate human-readable insights based on metrics."""
        insights = []
        
        hours = metrics['total_duration_hours']
        efficiency = metrics['sleep_efficiency']
        stages = metrics['stages_summary']
        
        insights.append(f"Total sleep duration: {hours} hours")
        insights.append(f"Sleep efficiency: {int(efficiency*100)}%")
        
        if hours < 6:
            insights.append("Sleep duration below recommended 7 hours.")
        
        if stages['Deep'] < 10:
            insights.append("Low deep sleep detected (less than 10%).")
             
        if stages['Awake'] > 15:
            insights.append("High fragmentation (awake > 15% of sleep time).")
            
        return insights
