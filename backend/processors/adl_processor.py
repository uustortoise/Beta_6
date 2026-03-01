import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from elderlycare_v1_16.config import settings

# Import V1.16 Module
try:
    from elderlycare_v1_16.adl.long_term_analysis import ADLAnalyzer
except ImportError:
    pass

logger = logging.getLogger(__name__)

class ADLProcessor:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.processed_dir = data_dir / "processed"

    def get_elder_dir(self, elder_id: str) -> Path:
        return self.processed_dir / elder_id

    def process_adl(self, elder_id: str, platform_instance: Any) -> bool:
        """
        Run ADL analysis and generate adl_trends.json
        """
        logger.info(f"Starting ADL Analysis for {elder_id}")
        
        if not hasattr(platform_instance, 'prediction_data') or not platform_instance.prediction_data:
            logger.warning(f"No prediction data for ADL analysis: {elder_id}")
            return False

        try:
            # 1. Initialize Analyzer
            analyzer = ADLAnalyzer(platform_instance)
            
            # 2. Get Daily Stats (DataFrame)
            # aggregate_daily_stats returns a single row DF for the *current* data
            # For a real "Long Term" trend, we need to append this to history.
            # In V1.16 this was done via data_manager. For prototype, we will just read/update our own JSON history.
            
            current_stats_df = analyzer.aggregate_daily_stats(platform_instance.prediction_data)
            
            if current_stats_df.empty:
                logger.warning("ADL Aggregation returned empty.")
                return False

            # 3. Update History (JSON)
            history_file = self.get_elder_dir(elder_id) / "adl_trends.json"
            history_data = self._load_history(elder_id)
            
            # Convert current stats to dict
            current_stats = current_stats_df.to_dict('records')[0]
            
            # Determine date from data (use the latest timestamp in the dataframe)
            data_date = datetime.now().strftime('%Y-%m-%d') # Fallback
            
            # Use platform_instance.prediction_data to find actual date
            # We iterate through available rooms to find max timestamp
            try:
                max_ts = None
                for _, df in platform_instance.prediction_data.items():
                    if 'timestamp' in df.columns:
                        try:
                            ts_series = pd.to_datetime(df['timestamp'])
                            if not ts_series.empty:
                                local_max = ts_series.max()
                                if max_ts is None or local_max > max_ts:
                                    max_ts = local_max
                        except: pass
                
                if max_ts:
                    data_date = max_ts.strftime('%Y-%m-%d')
                    logger.info(f"ADL Analysis: Determined data date: {data_date}")
            except Exception as e:
                logger.warning(f"ADL Analysis: Failed to extract date from data: {e}")

            current_stats['date'] = data_date
            
            # --- Label Normalization (New) ---
            ACTIVITY_TO_POSTURE = {
                'sleep': 'lying', 'sleeping': 'lying', 'bed': 'lying', 'nap': 'lying',
                'lie': 'lying', 'lying down': 'lying',
                'sit': 'sitting', 'watch tv': 'sitting', 'rest': 'sitting', 'eating': 'sitting',
                'stand': 'standing', 'cook': 'standing', 'wash': 'standing', 'bath': 'standing', 'clean': 'standing', 'cooking': 'standing',
                'walk': 'walking', 'run': 'walking', 'jog': 'walking', 'move': 'walking'
            }
            
            # Initialize standard counts if missing
            for posture in ['walking', 'standing', 'sitting', 'lying']:
                key = f'count_{posture}'
                if key not in current_stats:
                    current_stats[key] = 0
            
            # Aggregate counts
            for key, value in list(current_stats.items()):
                if key.startswith('count_'):
                    raw_label = key.replace('count_', '').lower().strip()
                    # Skip if it's already a standard key (to prevent double counting if logic changes)
                    # But need to be careful. The analyzer output might already have 'count_walking'.
                    # We want to add 'count_walk' to 'count_walking'.
                    
                    if raw_label in ['walking', 'standing', 'sitting', 'lying']:
                        continue
                        
                    target = ACTIVITY_TO_POSTURE.get(raw_label)
                    if target:
                        current_stats[f'count_{target}'] += value
                        # Optional: Remove the raw key if we want to clean up, 
                        # but keeping it might be useful for detailed drill-down later.
                        # For now, we leave it.

            # Append or Update
            self._update_history(elder_id, history_data, current_stats)
            
            # 4. Detect Anomalies & Trends (New v1.16 Feature)
            # Reconstruct DataFrame from history for analysis
            history_df = pd.DataFrame(history_data['daily_records'])
            if not history_df.empty:
                # Ensure correct types for analysis
                if 'mobility_index' in history_df.columns:
                     history_df['mobility_index'] = pd.to_numeric(history_df['mobility_index'], errors='coerce')
                
                analysis_results = analyzer.detect_trends_and_anomalies(history_df)
                history_data['anomalies'] = analysis_results.get('detected_anomalies', [])
                history_data['trends'] = analysis_results.get('trends', {})
            
            # 5. Save Trends
            self._save_history(history_file, history_data)

            # 5. Extract and Save Activity Timeline
            self._save_timeline(elder_id, platform_instance.prediction_data)
            
            # 6. Extract Review Candidates
            self._save_review_candidates(elder_id, platform_instance.prediction_data)

            return True

        except Exception as e:
            logger.error(f"ADL Analysis Failed: {e}", exc_info=True)
            return False

    def _save_review_candidates(self, elder_id: str, data: Dict[str, pd.DataFrame]):
        """
        Extracts low confidence and anomaly events for manual review.
        Output: review_candidates.json
        """
        candidates = []
        confidence_threshold = settings.ADL_CONFIDENCE_THRESHOLD
        
        try:
            for room, df in data.items():
                if df.empty: continue
                 
                act_col = 'activity' if 'activity' in df.columns else 'predicted_activity'
                if act_col not in df.columns or 'timestamp' not in df.columns:
                    continue
                    
                # Filter for low confidence OR anomaly
                for _, row in df.iterrows():
                    conf = float(row['confidence']) if 'confidence' in row else 1.0
                    is_anomaly = False
                    if 'combined_anomaly' in row and row['combined_anomaly']:
                        is_anomaly = True
                    elif 'final_anomaly' in row and row['final_anomaly']:
                        is_anomaly = True
                        
                    if conf < confidence_threshold or is_anomaly:
                         ts = row['timestamp']
                         ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
                         
                         candidates.append({
                             "resident_id": elder_id,
                             "timestamp": ts_str,
                             "room": room,
                             "activity": str(row[act_col]),
                             "confidence": conf,
                             "reason": "Anomaly" if is_anomaly else "Low Confidence"
                         })
                         
            # Save
            candidates_path = self.get_elder_dir(elder_id) / "review_candidates.json"
            with open(candidates_path, 'w') as f:
                json.dump(candidates, f, indent=2)
                
            logger.info(f"Saved {len(candidates)} review candidates for {elder_id}")
            
        except Exception as e:
            logger.error(f"Failed to save review candidates: {e}")

    def _save_timeline(self, elder_id: str, data: Dict[str, pd.DataFrame]):
        """
        Extracts and saves a simplified timeline of activities for visualization.
        Output: activity_timeline.json
        Format: [{timestamp, room, activity, confidence, ground_truth?, matching?}, ...]
        """
        timeline_events = []
        
        try:
            for room, df in data.items():
                if df.empty: continue
                
                # Determine activity column
                # Standard V1.16 Output keys: 'activity' (ground truth if training), 'predicted_activity' (model output)
                # If we are in purely prediction mode, 'predicted_activity' is the main one and 'activity' might be missing.
                
                has_ground_truth = 'activity' in df.columns
                has_prediction = 'predicted_activity' in df.columns
                
                # If neither exists, we can't do much
                if not has_ground_truth and not has_prediction:
                    continue
                    
                # Main display activity
                main_act = 'Unknown'
                if has_prediction:
                    main_act_col = 'predicted_activity' 
                else:
                     main_act_col = 'activity'

                # Ensure timestamp
                if 'timestamp' not in df.columns:
                    continue
                
                # Downsample
                sample_rate = 1
                if len(df) > 1000:
                    sample_rate = 5
                
                subset = df.iloc[::sample_rate].copy()
                
                # Normalize columns
                for _, row in subset.iterrows():
                    ts = row['timestamp']
                    if hasattr(ts, 'isoformat'):
                        ts_str = ts.isoformat()
                    else:
                        ts_str = str(ts)
                        
                    # Handle anomaly
                    is_anomaly = False
                    anomaly_type = "normal"
                    if 'combined_anomaly' in row and row['combined_anomaly']:
                        is_anomaly = True
                        anomaly_type = row.get('anomaly_type', 'unknown')
                    elif 'final_anomaly' in row and row['final_anomaly']:
                         is_anomaly = True
                         anomaly_type = "unknown"

                    # Build Event
                    event = {
                        "timestamp": ts_str,
                        "room": room,
                        "activity": str(row[main_act_col]),
                        "confidence": float(row['confidence']) if 'confidence' in row else 1.0,
                        "is_anomaly": bool(is_anomaly),
                        "anomaly_type": str(anomaly_type)
                    }
                    
                    # Add Comparison Data if available
                    if has_ground_truth and has_prediction:
                        gt = str(row['activity'])
                        pred = str(row['predicted_activity'])
                        event['ground_truth'] = gt
                        event['prediction'] = pred
                        event['match'] = (gt == pred)
                    
                    timeline_events.append(event)
            
            # Sort by timestamp
            timeline_events.sort(key=lambda x: x['timestamp'])
            
            # Save
            timeline_path = self.get_elder_dir(elder_id) / "activity_timeline.json"
            with open(timeline_path, 'w') as f:
                json.dump(timeline_events, f, indent=2)
                
            logger.info(f"Saved {len(timeline_events)} timeline events with comparison data to {timeline_path}")
            
        except Exception as e:
            logger.error(f"Failed to save timeline: {e}")

    def _load_history(self, elder_id: str) -> Dict[str, Any]:
        from data_manager import DataManager
        data_mgr = DataManager(self.data_dir)
        res_dir = data_mgr.get_resident_dir(elder_id)
        filepath = res_dir / "adl_trends.json"
        
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception:
                logger.warning("Corrupt ADL history, starting fresh.")
        
        return {
            "last_updated": None,
            "daily_records": [] 
        }

    def _update_history(self, elder_id: str, history: Dict, new_record: Dict):
        from data_manager import DataManager
        data_mgr = DataManager(self.data_dir)
        
        # Update JSON via DataManager (365 days)
        updated_records = data_mgr.update_history_json(elder_id, "adl_trends.json", new_record, max_days=365)
        
        history['daily_records'] = updated_records
        history['last_updated'] = datetime.now().isoformat()

    def _save_history(self, filepath: Path, data: Dict) -> bool:
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Saved ADL trends to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save ADL json: {e}")
            return False
