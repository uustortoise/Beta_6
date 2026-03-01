import numpy as np
from datetime import datetime
import os
import pandas as pd
from typing import Dict, Any, List, Optional
from elderlycare_v1_15.config.settings import DEFAULT_DATA_INTERVAL, DEFAULT_SEQUENCE_WINDOW

def create_training_history_entry(platform, rooms_trained, seq_length, epochs, confidence_threshold, time_based_settings, training_metrics, data_timestamp=None):
    """Create a training history entry record"""
    # Calculate average metrics
    avg_accuracy = np.mean([m['accuracy'] for m in training_metrics.values()]) if training_metrics else 0.0
    avg_loss = np.mean([m['loss'] for m in training_metrics.values()]) if training_metrics else 0.0
    
    return {
        "timestamp": data_timestamp or datetime.now(),
        "rooms_trained": rooms_trained,
        "rooms_count": len(rooms_trained),
        "epochs": epochs,
        "seq_length": seq_length,
        "avg_accuracy": float(avg_accuracy),
        "avg_loss": float(avg_loss),
        "metrics_detail": str(training_metrics),  # Store detailed metrics as string for reference
        "confidence_threshold": confidence_threshold,
        "time_based_enabled": time_based_settings.get('enabled', False),
        "data_interval": time_based_settings.get('data_interval', DEFAULT_DATA_INTERVAL),
        "sequence_time_window": time_based_settings.get('sequence_time_window', DEFAULT_SEQUENCE_WINDOW),
        "process_after_resample": time_based_settings.get('process_after_resample', True)
    }

def create_prediction_history_entry(elder_id, pred_data, filepath):
    """Create a prediction history entry record"""
    summary = {}
    total_predictions = 0
    start_time = None
    end_time = None
    
    for room, df in pred_data.items():
        if df.empty:
            continue
            
        # Get activity counts
        if 'predicted_activity' in df.columns:
            counts = df['predicted_activity'].value_counts().to_dict()
            total_predictions += len(df)
            
            # Time range
            if 'timestamp' in df.columns:
                min_ts = df['timestamp'].min()
                max_ts = df['timestamp'].max()
                
                if start_time is None or min_ts < start_time:
                    start_time = min_ts
                if end_time is None or max_ts > end_time:
                    end_time = max_ts
                    
            summary[room] = {
                'counts': counts,
                'count': len(df)
            }
            
    return {
        "timestamp": datetime.now(),
        "summary": summary,
        "total_predictions": total_predictions,
        "data_range_start": start_time,
        "data_range_end": end_time,
        "data_file": os.path.basename(filepath) if filepath else None,
        "models_used": "v1.14"
    }
