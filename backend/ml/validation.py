
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from ml.label_taxonomy import get_valid_prediction_labels

logger = logging.getLogger(__name__)

class DataIntegrityValidator:
    """
    Implements the "5-Point Data Integrity Check" mandated by the Engineering Audit (Jan 2026).
    Acts as a gatekeeper to prevent invalid data from polluting the database.
    """
    
    def __init__(self, room_config=None):
        from config import get_room_config
        self.room_config = room_config or get_room_config()
        
        # Point 4: Config/registry-driven valid label taxonomy.
        self.VALID_LABELS = set(get_valid_prediction_labels())

    def validate_predictions(self, prediction_dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """
        Run all 5 validation checks on a set of room predictions.
        
        Args:
            prediction_dfs: Dict mapping room_name -> DataFrame (must have 'timestamp', 'predicted_activity')
            
        Returns:
            Dict mapping room_name -> List of error messages (empty list if valid)
        """
        report = {}
        
        for room, df in prediction_dfs.items():
            errors = []
            
            if df.empty:
                continue
                
            # Ensure timestamp type
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception:
                    errors.append("Critical: Timestamp column is not datetime compatible")
                    report[room] = errors
                    continue

            # 1. Overlap Check (Not applicable for raw 10s predictions, but we check duplicate timestamps)
            if df['timestamp'].duplicated().any():
                dup_count = df['timestamp'].duplicated().sum()
                errors.append(f"Point 1: Found {dup_count} duplicate timestamps")

            # 2. Future Guard
            now = datetime.now()
            future_mask = df['timestamp'] > now
            if future_mask.any():
                future_count = future_mask.sum()
                max_future = df.loc[future_mask, 'timestamp'].max()
                errors.append(f"Point 2: Found {future_count} timestamps in the future (Max: {max_future})")

            # 3. Sensor Health (Check for NaN in critical sensors if columns exist)
            CRITICAL_SENSORS = ['motion', 'light']  # Modify based on specific hardware importance
            for sensor in CRITICAL_SENSORS:
                if sensor in df.columns:
                    nan_count = df[sensor].isna().sum()
                    if nan_count > 0:
                        errors.append(f"Point 3: Found {nan_count} NULL values for critical sensor '{sensor}'")

            # 4. Label Validity
            if 'predicted_activity' in df.columns:
                unique_labels = set(df['predicted_activity'].unique())
                invalid_labels = unique_labels - self.VALID_LABELS
                if invalid_labels:
                    errors.append(f"Point 4: Invalid activity labels found: {invalid_labels}")

            # 5. Confidence Bounds
            if 'confidence' in df.columns:
                conf_values = df['confidence'].dropna()
                if not conf_values.empty:
                    if (conf_values < 0.0).any() or (conf_values > 1.0).any():
                        min_c = conf_values.min()
                        max_c = conf_values.max()
                        errors.append(f"Point 5: Confidence score out of bounds [{min_c}, {max_c}]")

            if errors:
                logger.warning(f"Data Integrity Validation Failed for {room}: {errors}")
            else:
                logger.info(f"Data Integrity Validation Passed for {room}")
                
            report[room] = errors
            
        return report

def run_validation(prediction_results: Dict[str, pd.DataFrame]) -> bool:
    """
    Helper to run validation and log summary.
    Returns True if ALL rooms pass, False if ANY fail.
    """
    validator = DataIntegrityValidator()
    report = validator.validate_predictions(prediction_results)
    
    total_errors = sum(len(errs) for errs in report.values())
    
    if total_errors > 0:
        logger.error(f"❌ Integity Check FAILED with {total_errors} errors across {len(report)} rooms.")
        return False
        
    logger.info("✅ 5-Point Integrity Check PASSED")
    return True
