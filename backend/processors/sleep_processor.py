import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import sys

# Ensure backend directory is in sys.path for standalone execution
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Import the existing V1.16 System
# Note: We assume the path is already in sys.path from process_data.py
try:
    from elderlycare_v1_16.sleep.sleep_integration_complete_debug import EnhancedSleepAnalysisSystem
except ImportError:
    pass # Handled at runtime or by parent script

logger = logging.getLogger(__name__)

from elderlycare_v1_16.config import settings

# Heuristics for Sleep Stage Estimation (Deterministic)
# Ratios based on typical elderly sleep patterns
# Light: 50-60%, Deep: 10-20%, REM: 20-25%
STAGE_RATIOS = settings.SLEEP_STAGE_RATIOS

class SleepProcessor:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.processed_dir = data_dir / "processed"

    def get_elder_dir(self, elder_id: str) -> Path:
        return self.processed_dir / elder_id

    def process_sleep(self, elder_id: str, platform_instance: Any, room_name: str = "Bedroom") -> bool:
        """
        Run sleep analysis for a specific elder and save results to JSON.
        """
        logger.info(f"Starting Sleep Analysis for {elder_id}")
        
        # 1. Get Prediction Data for the specific room (Bedroom)
        # In a real scenario, we might merge data from multiple rooms or sensors.
        # For V1.16 port, we look for the dataframe in platform.prediction_data
        
        if not hasattr(platform_instance, 'prediction_data') or not platform_instance.prediction_data:
            logger.warning(f"No prediction data available for {elder_id}")
            return False

        # Find the bedroom or primary room data
        activity_df = None
        for room, df in platform_instance.prediction_data.items():
            if "bed" in room.lower() or "sleep" in room.lower():
                activity_df = df
                room_name = room
                break
        
        # If no explicit bedroom, just take the first room for the prototype
        if activity_df is None and platform_instance.prediction_data:
            room_name = list(platform_instance.prediction_data.keys())[0]
            activity_df = platform_instance.prediction_data[room_name]
            logger.info(f"No 'Bedroom' found, defaulting to {room_name}")

        if activity_df is None:
            logger.warning("No data found to analyze.")
            return False

        # 2. Initialize and Configure System (using V1.16 Defaults)
        try:
            import streamlit as st
            
            # V1.16 Defaults (from sleep_integration_complete_debug.py)
            V1_16_CONFIG = {
                'min_sleep_duration': 30.0,
                'awakening_gap': 5.0,
                'detection_mode': 'Elderly',
                'motion_threshold': 0.1,  # < 0.1 treated as stillness/noise
                'sleep_stage_enabled': True,
                'deep_threshold': 0.3, 
                'light_threshold': 0.6,
                'rem_threshold': 0.8,
                'min_coverage': 0.5,
                'max_missing': 0.3,
                'max_movements': 30.0,
                'min_efficiency': 0.7
            }
            
            # Inject into Streamlit session state (used by EnhancedSleepAnalysisSystem)
            if not hasattr(st, 'session_state'):
                 # Create a mock session state if running in headless environment without one
                 class MockSessionState(dict):
                     def __setattr__(self, key, value):
                         self[key] = value
                     def __getattr__(self, key):
                         return self.get(key)
                 st.session_state = MockSessionState()

            st.session_state['sleep__parameter_config'] = V1_16_CONFIG
            logger.info("Injected V1.16 Sleep Settings into Session State")
            
        except ImportError:
            logger.warning("Could not import Streamlit to configure Sleep System. Using defaults.")
        except Exception as e:
            logger.warning(f"Failed to configure Sleep System: {e}")

        # The system needs 'platform' and 'person_id'
        sleep_system = EnhancedSleepAnalysisSystem(platform_instance, person_id=elder_id)

        # 3. Run Analysis
        # V1.16's analyze_sleep_data takes (activity_data, sensor_data). 
        # For now we pass the same DF for both or None for sensor_data if not strictly required by the specific logic flow
        # The debug version seems to use sensor_data for environmental stability.
        

        try:
            results = sleep_system.analyze_sleep_data(activity_df, activity_df)
            
            # 3a. Process History and Averages
            # Determine date from data (use the latest timestamp in the dataframe)
            data_date = datetime.now().strftime('%Y-%m-%d') # Fallback
            if 'timestamp' in activity_df.columns:
                 try:
                     # Ensure datetime
                     ts_series = pd.to_datetime(activity_df['timestamp'])
                     if not ts_series.empty:
                         # Use the max date in the file (assuming file covers one day or ending of a period)
                         data_date = ts_series.max().strftime('%Y-%m-%d')
                         logger.info(f"Determined data date: {data_date}")
                 except Exception as e:
                     logger.warning(f"Failed to extract date from dataframe: {e}")

            history_data = self._update_sleep_history(elder_id, results, data_date)
            
            # 4. Extract Serializable Data
            # 'results' contains matplotlib figures and pandas objects. We need pure JSON.
            summary_json = self._extract_json_summary(results, history_data)
            
            # 5. Save
            return self._save_results(elder_id, summary_json)
            
        except Exception as e:
            logger.error(f"Sleep Analysis Failed: {e}", exc_info=True)
            return False

    def _update_sleep_history(self, elder_id: str, current_results: Dict, date_str: str) -> Dict:
        """
        Update sleep history using DataManager and calculate 7-day averages.
        """
        # We assume the parent script might not have passed DataManager, 
        # so we initialize a local one for simplicity or use the shared one.
        from data_manager import DataManager
        data_mgr = DataManager(self.data_dir)
        
        # Extract current day's key metrics
        current_metrics = current_results.get('sleep_metrics', {})
        current_score = current_results.get('quality_score', {}).get('overall_score', 0)
        
        today_entry = {
            'date': date_str,
            'timestamp': datetime.now().isoformat(),
            'duration_hours': current_metrics.get('duration_hours', 0),
            'efficiency': current_metrics.get('efficiency', 0),
            'score': current_score
        }
        
        # Update JSON via DataManager (365 days)
        history = data_mgr.update_history_json(elder_id, "sleep_history.json", today_entry, max_days=365)
            
        # Calculate 7-Day Trailing Averages
        relevant_history = [h for h in history if h['date'] <= date_str]
        last_7_days = relevant_history[-7:]
        
        avg_data = {
            'avg_duration': sum(h['duration_hours'] for h in last_7_days) / len(last_7_days) if last_7_days else 0,
            'avg_efficiency': sum(h['efficiency'] for h in last_7_days) / len(last_7_days) if last_7_days else 0,
            'avg_score': sum(h['score'] for h in last_7_days) / len(last_7_days) if last_7_days else 0,
            'days_counted': len(last_7_days),
            'history': last_7_days
        }
        
        return avg_data

    def _extract_json_summary(self, results: Dict, history_data: Dict) -> Dict[str, Any]:

        """Convert complex results dict to JSON-safe dict"""
        safe_data = {
            "timestamp": datetime.now().isoformat(),
            "quality_score": 0,
            "metrics": {},
            "sleep_periods": [],
            "insights": []
        }
        
        if not results:
            return safe_data

        # Extract Score
        if 'quality_score' in results and isinstance(results['quality_score'], dict):
             safe_data['quality_score'] = results['quality_score'].get('total_score', 0)
             # Note: V1.16 returns 'overall_score', checking both just in case
             if 'overall_score' in results['quality_score']:
                 safe_data['quality_score'] = results['quality_score']['overall_score']
                 
        
        # Extract Metrics
        if 'metrics' in results:
            # Copy only simple types
            for k, v in results['metrics'].items():
                if isinstance(v, (int, float, str, bool)):
                    safe_data['metrics'][k] = v

        if 'sleep_metrics' in results:
             for k, v in results['sleep_metrics'].items():
                if isinstance(v, (int, float, str, bool)):
                    safe_data['metrics'][k] = v
        
        # Extract Sleep Periods (for Hypnogram)
        if 'sleep_periods' in results:
            # Provide simplified list for the chart
            # Structure: [{start, end, stage, duration}, ...]
            periods = []
            for p in results['sleep_periods']:
                periods.append({
                    "start": str(p.get('start', p.get('start_time'))),
                    "end": str(p.get('end', p.get('end_time'))),
                    "type": str(p.get('type', 'Unknown')),
                    "duration": float(p.get('duration_minutes', 0))
                })
            safe_data['sleep_periods'] = periods

        # Improve Sleep Stage Data (Deterministic Estimation)
        # Calculate total duration from periods
        total_duration_min = sum(p.get('duration', 0) for p in safe_data['sleep_periods'])
        
        stage_bd = {'Light': 0, 'Deep': 0, 'REM': 0, 'Awake': 0}
        
        # If V1.16 returns actual stage analysis, use it
        if 'sleep_stage_analysis' in results and 'stage_percentages' in results['sleep_stage_analysis']:
             s_pct = results['sleep_stage_analysis']['stage_percentages']
             stage_bd = {
                 'Light': round(s_pct.get('light_sleep', 0), 1),
                 'Deep': round(s_pct.get('deep_sleep', 0), 1),
                 'REM': round(s_pct.get('rem_sleep', 0), 1),
                 'Awake': round(s_pct.get('awake', 0), 1)
             }
        elif total_duration_min > 0:
            # Fallback to ratios if not available
            stage_bd = {
                'Light': round(STAGE_RATIOS['Light'] * 100, 1),
                'Deep': round(STAGE_RATIOS['Deep'] * 100, 1),
                'REM': round(STAGE_RATIOS['REM'] * 100, 1),
                'Awake': round(STAGE_RATIOS['Awake'] * 100, 1)
            }
            
        safe_data['stage_breakdown'] = stage_bd

        # Add 7-Day Averages and History
        safe_data['seven_day_average'] = {
            'duration_hours': round(history_data['avg_duration'], 1),
            'efficiency': round(history_data['avg_efficiency'], 2),
            'score': round(history_data['avg_score'], 0)
        }
        safe_data['daily_history'] = history_data['history']
        
        # Calculate Grade based on 7-Day Average Score
        avg_score = history_data['avg_score']
        if avg_score >= 85: grade = 'A'
        elif avg_score >= 70: grade = 'B'
        elif avg_score >= 55: grade = 'C'
        elif avg_score >= 40: grade = 'D'
        else: grade = 'F'
        
        safe_data['grade'] = grade


        # Extract Recommendations/Insights
        # Merge 'insights' (problem detection) and 'recommendations' (advice)
        all_insights = []
        
        if 'insights' in results and isinstance(results['insights'], list):
            all_insights.extend(results['insights'])
            
        if 'quality_score' in results and 'recommendations' in results['quality_score']:
            recs = results['quality_score']['recommendations']
            if isinstance(recs, list):
                all_insights.extend(recs)
        elif 'recommendations' in results: # Fallback location
             if isinstance(results['recommendations'], list):
                all_insights.extend(results['recommendations'])
                
        # Deduplicate
        safe_data['insights'] = list(dict.fromkeys(all_insights))

        return safe_data

    def _save_results(self, elder_id: str, data: Dict) -> bool:
        elder_dir = self.get_elder_dir(elder_id)
        elder_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = elder_dir / "sleep_summary.json"
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Saved sleep summary to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save sleep json: {e}")
            return False
