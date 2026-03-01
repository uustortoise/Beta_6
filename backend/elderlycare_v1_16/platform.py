"""
Elderly Care Platform Core Class
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
import tensorflow as tf
import joblib
import plotly.graph_objects as go
from .anomaly.detector import AnomalyDetector
from .preprocessing.noise import hampel_filter, clip_outliers
from .utils.uncertainty import calculate_entropy
from .preprocessing.sequences import create_sequences
from .preprocessing.resampling import resample_to_fixed_interval, calculate_samples_from_time
from .preprocessing.gap_policy import resolve_max_ffill_gap_seconds
from .config.settings import (
    DEFAULT_SENSOR_COLUMNS, DEFAULT_PHYSICAL_SENSORS, DEFAULT_DATA_INTERVAL, DEFAULT_SEQUENCE_WINDOW
)
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

logger = logging.getLogger(__name__)

def _parse_interval_to_seconds(interval: str) -> int:
    if not interval:
        return 10
    s = str(interval).strip().lower()
    try:
        # Raw numeric seconds (e.g., 10)
        if s.replace(".", "", 1).isdigit():
            return max(1, int(float(s)))
        if s.endswith("ms"):
            return max(1, int(float(s[:-2]) / 1000.0))
        if s.endswith("sec"):
            return max(1, int(float(s[:-3])))
        if s.endswith("s"):
            return max(1, int(float(s[:-1])))
        if s.endswith("min"):
            return max(1, int(float(s[:-3]) * 60))
        if s.endswith("m"):
            return max(1, int(float(s[:-1]) * 60))
        # Pandas alias: 'T' == minute
        if s.endswith("t"):
            return max(1, int(float(s[:-1]) * 60))
        if s.endswith("h"):
            return max(1, int(float(s[:-1]) * 3600))
    except Exception:
        return 10
    return 10


def _resolve_max_ffill_gap_seconds(default_seconds: float = 60.0):
    """
    Resolve maximum forward-fill bridging gap in seconds.
    """
    # Policy SSoT: runtime should receive this value explicitly from TrainingPolicy.
    # Fallback path uses only local default when no explicit value is injected.
    return resolve_max_ffill_gap_seconds(raw_value=default_seconds, default_seconds=default_seconds)


def _resolve_ffill_limit(interval: str, max_gap_seconds) -> int | None:
    """Convert max_gap_seconds policy to pandas `ffill(limit=...)` steps."""
    if max_gap_seconds is None:
        return None
    interval_seconds = _parse_interval_to_seconds(interval or DEFAULT_DATA_INTERVAL)
    if interval_seconds <= 0:
        return None
    try:
        gap_seconds = float(max_gap_seconds)
    except (TypeError, ValueError):
        return None
    if gap_seconds <= 0:
        return None
    return max(1, int(gap_seconds // interval_seconds))

def add_temporal_features(df):
    """
    Add time-of-day features for temporal awareness in activity recognition.
    
    Features added:
    - hour_sin, hour_cos: Circular encoding of hour (0-23)
    - day_period: Categorical time period (0=night, 1=morning, 2=afternoon, 3=evening)
    
    Args:
        df: DataFrame with 'timestamp' column
        
    Returns:
        DataFrame with added temporal features
    """
    if 'timestamp' not in df.columns:
        logger.warning("No timestamp column found, skipping temporal features")
        return df
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract hour
    hour = df['timestamp'].dt.hour
    
    # Circular encoding (prevents discontinuity at midnight: 23→0)
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    
    # Day period: 0-6=night, 6-12=morning, 12-18=afternoon, 18-24=evening
    df['day_period'] = pd.cut(hour, bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], include_lowest=True).astype(float)
    
    logger.debug(f"Added temporal features: hour_sin, hour_cos, day_period")
    return df


def add_rolling_features(df, sensor_columns, room_specific_window=None, standard_windows=[90, 360], interval=None):
    """
    Add rolling (moving window) statistical features to capture temporal trends.
    
    This significantly improves the Attention layer's ability to distinguish between:
    - Transient noise vs. meaningful activity changes
    - Short bursts (e.g., dropping something) vs. sustained patterns (e.g., cooking)
    
    Args:
        df: DataFrame with sensor data and 'timestamp' column
        sensor_columns: List of physical sensor columns to compute rolling features for
        room_specific_window: Optional int, room-specific window size in seconds
                             (e.g., 300 for Bathroom, 1800 for Bedroom)
        standard_windows: List of standard window sizes in seconds [short, long]
                         Default: [90s (1.5min), 360s (6min)]
    
    Returns:
        DataFrame with added rolling features (mean, std) for each sensor
    
    Example:
        For a sensor 'motion' with windows [90, 360]:
        - motion_roll_mean_90: Average motion over last 1.5 minutes
        - motion_roll_std_90: Motion variability over last 1.5 minutes
        - motion_roll_mean_360: Average motion over last 6 minutes
        - motion_roll_std_360: Motion variability over last 6 minutes
    """
    if 'timestamp' not in df.columns:
        logger.warning("No timestamp column found, skipping rolling features")
        return df
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate all window sizes to process
    windows_to_process = []
    if room_specific_window:
        windows_to_process.append(('specific', room_specific_window))
    for w in standard_windows:
        windows_to_process.append(('standard', w))
    
    # Compute rolling features for each sensor and window
    interval_sec = _parse_interval_to_seconds(interval or DEFAULT_DATA_INTERVAL)
    for sensor in sensor_columns:
        if sensor not in df.columns:
            continue
        
        for window_type, window_sec in windows_to_process:
            window_samples = max(1, window_sec // interval_sec)
            
            # Rolling mean (captures long-term trends like CO2 accumulation)
            df[f'{sensor}_roll_mean_{window_sec}'] = df[sensor].rolling(
                window=window_samples, min_periods=1
            ).mean()
            
            # Rolling std (captures activity intensity/volatility)
            df[f'{sensor}_roll_std_{window_sec}'] = df[sensor].rolling(
                window=window_samples, min_periods=1
            ).std().fillna(0)
    
    feature_count = len([c for c in df.columns if '_roll_' in c])
    logger.debug(f"Added {feature_count} rolling features")
    
    return df



class Attention(Layer):
    """
    Attention mechanism layer for temporal weighting.
    Computes a weighted sum of the input sequence based on learned attention scores.
    """
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch_size, time_steps, features)
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        # x: (batch_size, time_steps, features)
        # s: (batch_size, time_steps, 1) -> score for each time step
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1) # softmax over time axis
        output = x * a # weighted input
        return K.sum(output, axis=1) # (batch_size, features) - weighted sum

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(Attention, self).get_config()



class ElderlyCarePlatform:
    def __init__(self, 
                 sensor_columns=DEFAULT_SENSOR_COLUMNS,
                 data_interval=DEFAULT_DATA_INTERVAL,
                 sequence_time_window=DEFAULT_SEQUENCE_WINDOW,
                 enable_time_based_processing=True):
        self.sensor_columns = sensor_columns
        self.data_interval = data_interval
        self.sequence_time_window = sequence_time_window
        self.enable_time_based_processing = enable_time_based_processing
        self.scalers = {}
        self.label_encoders = {}
        self.class_thresholds = {}  # room -> {class_id(str/int): threshold}
        self.room_models = {}
        self.training_data = {}
        self.anomaly_detector = AnomalyDetector(sensor_columns)
        # Removed anomalies attribute - using confidence threshold only
        logger.info(f"Platform initialized with interval={data_interval}, time_window={sequence_time_window}, "
                   f"time_based_processing={enable_time_based_processing}")

    def load_excel_data(self, file):
        xls = pd.ExcelFile(file)
        room_data = {}
        self.timestamp_quality_report = {}  # Store quality report as instance variable
        
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            # Only validate physical sensors - temporal features are generated later
            required_cols = set(DEFAULT_PHYSICAL_SENSORS + ['timestamp'])
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                raise ValueError(f"Missing columns in '{sheet_name}': {missing}")
            
            # Store original timestamp strings for quality checking
            original_timestamps = df['timestamp'].copy()
            
            # Initialize quality report for this sheet
            quality_report = {
                'sheet_name': sheet_name,
                'original_count': len(df),
                'has_date_components': False,
                'has_time_components': False,
                'is_time_only': False,
                'default_date_detected': False,
                'default_date': None,
                'warnings': [],
                'parsed_count': 0,
                'invalid_count': 0
            }
            
            # Check timestamp format before parsing
            if 'timestamp' in df.columns:
                sample_timestamps = original_timestamps.dropna().head(10).astype(str).tolist()
                quality_report['sample_timestamps'] = sample_timestamps
                
                # Check for time-only patterns (HH:MM:SS, HH:MM, etc.)
                time_only_pattern = r'^\d{1,2}:\d{2}(:\d{2})?(\.\d+)?$'
                time_only_12hr = r'^\d{1,2}:\d{2}(:\d{2})?\s*[APap][Mm]$'
                
                time_only_count = 0
                for ts in sample_timestamps:
                    import re
                    if re.match(time_only_pattern, ts) or re.match(time_only_12hr, ts):
                        time_only_count += 1
                
                if time_only_count >= len(sample_timestamps) * 0.8:  # 80% are time-only
                    quality_report['is_time_only'] = True
                    quality_report['warnings'].append(
                        f"⚠️ TIMESTAMP WARNING: Data appears to contain only time information (no date). "
                        f"The system will use today's date ({datetime.now().date()}) for display. "
                        f"Consider adding year/month/day to your timestamp column."
                    )
            
            # DEBUG: Log activity labels if present
            if 'activity' in df.columns:
                logger.info(f"🔍 DEBUG [{sheet_name}] Raw activity labels from Excel:")
                
                # Handle mixed types (float/str) safely - drop NaN values
                activity_values = df['activity'].dropna().astype(str)
                unique_activities = sorted(activity_values.unique())
                
                logger.info(f"   Unique values: {unique_activities}")
                logger.info(f"   Value counts:\n{df['activity'].value_counts().to_dict()}")
                logger.info(f"   Data type: {df['activity'].dtype}")
                logger.info(f"   NaN count: {df['activity'].isna().sum()}")
                
                # Check for whitespace/case issues
                if len(activity_values) > 0:
                    stripped = activity_values.str.strip()
                    if not (activity_values == stripped).all():
                        logger.warning(f"   ⚠️ Whitespace detected in activity labels for {sheet_name}")
                        logger.info(f"   After strip: {sorted(stripped.unique())}")
                    
                    # Check for case variations
                    lowered = activity_values.str.lower()
                    if len(lowered.unique()) < len(activity_values.unique()):
                        logger.warning(f"   ⚠️ Case variations detected in activity labels for {sheet_name}")
                        logger.info(f"   After lowercase: {sorted(lowered.unique())}")
            
            # Robust timestamp parsing with multiple fallback strategies
            original_timestamps = df['timestamp'].copy()
            parsed_timestamps = pd.Series(index=df.index, dtype='datetime64[ns]')
            
            # Strategy 1: Try direct parsing with coerce (handles ISO strings, some Excel serials)
            parsed_timestamps = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            
            # Strategy 2: For remaining NaNs, try converting Excel serial numbers (float)
            if parsed_timestamps.isna().any():
                mask = parsed_timestamps.isna()
                # Check if the original value is numeric (Excel serial)
                numeric_mask = mask & pd.to_numeric(original_timestamps, errors='coerce').notna()
                if numeric_mask.any():
                    # Convert Excel serial numbers (Windows origin 1899-12-30)
                    excel_serials = pd.to_numeric(original_timestamps[numeric_mask])
                    # Excel for Windows uses 1899-12-30 as day 0 (with 1900 leap year bug)
                    excel_dates = pd.to_datetime(excel_serials, unit='D', origin='1899-12-30', utc=True)
                    parsed_timestamps[numeric_mask] = excel_dates
            
            # Strategy 3: For remaining NaNs, try with dayfirst=True for European dates
            if parsed_timestamps.isna().any():
                mask = parsed_timestamps.isna()
                remaining = original_timestamps[mask].astype(str)
                parsed_remaining = pd.to_datetime(remaining, errors='coerce', dayfirst=True, utc=True)
                parsed_timestamps[mask] = parsed_remaining
            
            # Strategy 4: For any remaining NaNs, try with format='mixed' (pandas 2.0+)
            if parsed_timestamps.isna().any():
                mask = parsed_timestamps.isna()
                remaining = original_timestamps[mask].astype(str)
                try:
                    parsed_remaining = pd.to_datetime(remaining, errors='coerce', format='mixed', utc=True)
                    parsed_timestamps[mask] = parsed_remaining
                except (ValueError, TypeError):
                    pass
            
            # Final fallback: try without utc=True
            if parsed_timestamps.isna().any():
                mask = parsed_timestamps.isna()
                remaining = original_timestamps[mask].astype(str)
                parsed_remaining = pd.to_datetime(remaining, errors='coerce')
                parsed_timestamps[mask] = parsed_remaining
            
            # Convert to timezone-naive (remove timezone) for consistent sorting
            if parsed_timestamps.dt.tz is not None:
                parsed_timestamps = parsed_timestamps.dt.tz_convert(None)
            
            df['timestamp'] = parsed_timestamps
            
            # Update quality report with parsed timestamp analysis
            if 'timestamp' in df.columns and not df['timestamp'].isna().all():
                years = df['timestamp'].dt.year
                quality_report['min_year'] = int(years.min())
                quality_report['max_year'] = int(years.max())
                quality_report['min_timestamp'] = df['timestamp'].min()
                quality_report['max_timestamp'] = df['timestamp'].max()
                quality_report['parsed_count'] = len(df)
                quality_report['invalid_count'] = df['timestamp'].isna().sum()

                # Check for suspicious years (too old or future)
                current_year = datetime.now().year
                if years.min() < 2010 or years.max() > current_year + 1:
                    quality_report['warnings'].append(
                        f"⚠️ TIMESTAMP WARNING: Parsed timestamps have years in the range {years.min()}-{years.max()}. "
                        f"This might indicate incorrect date parsing (e.g., two-digit year interpreted as 1900s). "
                        f"Please check that your timestamps include the correct four-digit year."
                    )
                
                # Check if all timestamps have the same date (might be time-only data)
                unique_dates = df['timestamp'].dt.date.unique()
                if len(unique_dates) == 1:
                    unique_date = unique_dates[0]
                    today = datetime.now().date()
                    if unique_date == today:
                        quality_report['warnings'].append(
                            f"⚠️ TIMESTAMP WARNING: All timestamps have today's date ({today}). "
                            f"This suggests your data might only contain time information without dates. "
                            f"Consider adding year/month/day to your timestamp column."
                        )
                    else:
                        # Single-day data is normal for daily analysis, change to informational note
                        quality_report['warnings'].append(
                            f"📅 NOTE: Data contains timestamps from a single day ({unique_date}). "
                            f"This is normal for daily analysis."
                        )
            
            # Store quality report for this sheet
            self.timestamp_quality_report[sheet_name] = quality_report
            
            # Check for invalid timestamps after all parsing attempts
            na_count = df['timestamp'].isna().sum()
            if na_count > 0:
                original_count = len(df)
                logger.warning(f"Failed to parse {na_count}/{original_count} timestamps in sheet '{sheet_name}'. Invalid rows will be removed.")
                
                # Log sample of problematic values for debugging
                invalid_indices = df[df['timestamp'].isna()].index.tolist()
                sample_problematic = []
                for idx in invalid_indices[:3]:  # Show first 3
                    if idx < len(df):
                        sample_problematic.append(f"Row {idx}: {df.iloc[idx]['timestamp'] if 'timestamp' in df.columns else 'N/A'}")
                
                if sample_problematic:
                    logger.info(f"Sample problematic timestamp values: {sample_problematic}")
                
                # Keep rows with valid timestamps
                df = df.dropna(subset=['timestamp'])
                logger.info(f"Kept {len(df)} valid rows out of {original_count}")
            
            room_data[sheet_name] = df.sort_values('timestamp').reset_index(drop=True)
        return room_data

    def create_sequences(self, data, seq_length):
        """
        Create sequences from data for model input.
        
        Parameters:
        - data: numpy array of shape (n_samples, n_features)
        - seq_length: int, length of each sequence
        
        Returns:
        - numpy array of shape (n_sequences, seq_length, n_features)
        """
        from .preprocessing.sequences import create_sequences as create_seq_func
        return create_seq_func(data, seq_length)

    def preprocess_room_data(self, df, room_name, is_training=True, scaler_type='standard'):
        """
        Preprocess room data with optional scaler type.
        
        Parameters:
        - df: DataFrame with sensor data
        - room_name: str, name of the room
        - is_training: bool, whether this is training data
        - scaler_type: str, 'standard' for StandardScaler or 'robust' for RobustScaler
        
        Returns:
        - DataFrame with preprocessed data
        """
        df = df.ffill().bfill()
        
        if is_training:
            if 'activity' not in df.columns:
                raise ValueError(f"'activity' column missing in {room_name}")
            
            # Choose scaler based on type
            if scaler_type.lower() == 'robust':
                scaler = RobustScaler()
                logger.info(f"Using RobustScaler for {room_name}")
            else:
                scaler = StandardScaler()
                logger.info(f"Using StandardScaler for {room_name}")
            
            df[self.sensor_columns] = scaler.fit_transform(df[self.sensor_columns])
            self.scalers[room_name] = scaler
            
            le = LabelEncoder()
            df['activity_encoded'] = le.fit_transform(df['activity'])
            self.label_encoders[room_name] = le
            return df[self.sensor_columns + ['activity_encoded']]
        else:
            if room_name not in self.scalers:
                raise ValueError(f"No scaler found for room: {room_name}")
            df[self.sensor_columns] = self.scalers[room_name].transform(df[self.sensor_columns])
            return df[self.sensor_columns]

    def preprocess_without_scaling(self, df, room_name, is_training=False,
                                   apply_denoising=False, denoising_method='hampel',
                                   denoising_window=5, denoising_threshold=3.0,
                                   max_ffill_gap_seconds=None):
        """
        Phase A: Preprocessing pipeline WITHOUT scaling to prevent temporal leakage.
        
        This method performs all preprocessing steps EXCEPT scaling:
        1. Resample to fixed interval (if time-based processing enabled)
        2. Handle missing data (forward/backward fill on regular intervals)
        3. Add temporal and rolling features
        4. Apply denoising if requested
        5. Normalize and encode labels (for training)
        
        The scaling step is intentionally separated to allow fitting on train split only.
        Use apply_scaling() after temporal split to apply scaling.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with sensor data
        room_name : str
            Name of the room
        is_training : bool, default=False
            Whether this is training data (if True, processes labels)
        apply_denoising : bool, default=False
            Whether to apply denoising/spike removal
        denoising_method : str, default='hampel'
            Denoising method: 'hampel' or 'clip'
        denoising_window : int, default=5
            Window size for denoising
        denoising_threshold : float, default=3.0
            Threshold for denoising
        max_ffill_gap_seconds : float | None, optional
            Maximum forward-fill bridge in seconds. If omitted, resolves from env.
        
        Returns:
        --------
        pd.DataFrame
            Preprocessed DataFrame with unscaled sensor columns, ready for sequence creation.
            Columns include: timestamp, sensor_columns, activity (if training), activity_encoded (if training)
        """
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        raw_rows_before_resample = int(len(df_processed))
        if max_ffill_gap_seconds is None:
            max_ffill_gap_seconds = _resolve_max_ffill_gap_seconds(default_seconds=60.0)
        else:
            try:
                max_ffill_gap_seconds = float(max_ffill_gap_seconds)
            except (TypeError, ValueError):
                logger.warning(
                    f"Invalid injected max_ffill_gap_seconds={max_ffill_gap_seconds!r}; falling back to env/default."
                )
                max_ffill_gap_seconds = _resolve_max_ffill_gap_seconds(default_seconds=60.0)
        ffill_limit = _resolve_ffill_limit(self.data_interval, max_ffill_gap_seconds)
        
        # Step 1: Resample if time-based processing is enabled
        if self.enable_time_based_processing and 'timestamp' in df_processed.columns:
            try:
                logger.info(f"Resampling data for {room_name} to {self.data_interval} interval")
                
                # Analyze interval consistency before resampling
                from .preprocessing.resampling import validate_interval_consistency
                validation = validate_interval_consistency(df_processed)
                
                if not validation['is_consistent']:
                    logger.warning(f"Irregular intervals detected for {room_name}: {validation['warnings']}")
                
                # Perform resampling
                from .preprocessing.resampling import resample_to_fixed_interval
                df_processed = resample_to_fixed_interval(
                    df_processed, 
                    interval=self.data_interval.lower(),
                    method='linear',
                    timestamp_col='timestamp',
                    fill_method='ffill',
                    keep_original_timestamps=False,
                    max_ffill_gap_seconds=max_ffill_gap_seconds,
                )
                logger.info(f"Resampled {room_name}: {len(df)} -> {len(df_processed)} samples")
                
            except (ValueError, TypeError) as e:
                logger.error(f"Resampling failed for {room_name}: {e}.")
                raise ValueError(f"Resampling failed for {room_name}: {e}") from e
        
        # Step 2: Handle missing data (forward/backward fill)
        logger.info(f"Handling missing data for {room_name}")
        # For time-resampled paths, avoid a second forward-fill pass that would
        # silently extend gaps beyond the configured limit.
        should_forward_fill = not (self.enable_time_based_processing and 'timestamp' in df_processed.columns)
        if should_forward_fill:
            df_processed = df_processed.ffill(limit=ffill_limit)
        # Only backfill leading NaNs (before any observation) to avoid future leakage.
        for col in df_processed.columns:
            if col == 'timestamp':
                continue
            if not df_processed[col].isna().any():
                continue
            first_valid = df_processed[col].first_valid_index()
            if first_valid is None:
                continue
            df_processed.loc[:first_valid, col] = df_processed.loc[:first_valid, col].bfill()
        
        # Step 2.5: Add temporal features for time-aware predictions
        logger.info(f"Adding temporal features for {room_name}")
        df_processed = add_temporal_features(df_processed)
        
        # Step 2.6: Add rolling context features
        # Get room-specific window from config if available
        from config import get_room_config
        try:
            room_config = get_room_config()
            room_specific_window = room_config.get_sequence_window(room_name)
            room_interval = room_config.get_data_interval(room_name)
            logger.info(f"Adding rolling features for {room_name} with room-specific window={room_specific_window}s")
            df_processed = add_rolling_features(
                df_processed, 
                DEFAULT_PHYSICAL_SENSORS,  # Only compute for physical sensors
                room_specific_window=room_specific_window,
                standard_windows=[90, 360],  # 1.5min and 6min standard windows
                interval=room_interval or self.data_interval,
            )
        except Exception as e:
            logger.warning(f"Could not add rolling features: {e}. Proceeding without rolling features.")

        
        # Step 3: Apply denoising if requested (on regular intervals)
        if apply_denoising:
            logger.info(f"Applying {denoising_method} denoising to {room_name}")
            if denoising_method.lower() == 'hampel':
                df_processed = hampel_filter(df_processed, self.sensor_columns,
                                            window=denoising_window, n_sigmas=denoising_threshold)
                logger.info(f"Applied Hampel filter to {room_name} data")
            elif denoising_method.lower() == 'clip':
                df_processed = clip_outliers(df_processed, self.sensor_columns,
                                            method='mad', factor=denoising_threshold,
                                            window=denoising_window)
                logger.info(f"Applied MAD clipping to {room_name} data")

        # Drop unresolved sensor gaps rather than hallucinating values across long outages.
        available_sensor_cols = [c for c in self.sensor_columns if c in df_processed.columns]
        if available_sensor_cols:
            missing_sensor_mask = df_processed[available_sensor_cols].isna().any(axis=1)
            if bool(missing_sensor_mask.any()):
                dropped_rows = int(missing_sensor_mask.sum())
                logger.info(
                    f"Dropping {dropped_rows} rows with unresolved sensor gaps for {room_name} "
                    f"(ffill_limit={ffill_limit}, max_gap_seconds={max_ffill_gap_seconds})."
                )
                df_processed = df_processed.loc[~missing_sensor_mask].copy()
        df_processed.attrs["raw_rows_before_resample"] = raw_rows_before_resample
        df_processed.attrs["rows_after_gap_drop"] = int(len(df_processed))
        if df_processed.empty:
            if is_training:
                raise ValueError(
                    f"No usable samples remain after bounded gap handling for {room_name}; "
                    "check data continuity or relax MAX_RESAMPLE_FFILL_GAP_SECONDS."
                )
            return df_processed
        
        # Step 4: Process labels (for training) WITHOUT encoding yet
        if is_training:
            if 'activity' not in df_processed.columns:
                raise ValueError(f"'activity' column missing in {room_name}")
            
            # Normalize labels before encoding to prevent inconsistent classes.
            # Drop empty/NaN labels rather than encoding them as a new class.
            activity_raw = df_processed['activity']
            valid_mask = activity_raw.notna() & (activity_raw.astype(str).str.strip() != "")
            if not valid_mask.all():
                dropped = int((~valid_mask).sum())
                logger.warning(f"Dropping {dropped} rows with missing/empty activity labels for {room_name}")
                df_processed = df_processed.loc[valid_mask].copy()

            # Canonicalize labels to the shared taxonomy when available.
            try:
                from utils.segment_utils import normalize_activity_name, validate_activity_for_room
                df_processed['activity'] = df_processed['activity'].apply(
                    lambda a: validate_activity_for_room(normalize_activity_name(str(a).strip().lower()), room_name)
                )
            except Exception as e:
                logger.warning(f"Could not import activity normalization utilities; using lowercase/strip only: {e}")
                df_processed['activity'] = df_processed['activity'].astype(str).str.strip().str.lower()

            # Log a capped set of labels for debugging (avoid log spam).
            unique_labels = sorted(pd.unique(df_processed['activity']).tolist())
            preview = unique_labels[:30]
            suffix = "..." if len(unique_labels) > 30 else ""
            logger.info(f"Activity labels for {room_name} ({len(unique_labels)}): {preview}{suffix}")
            
            # Return with activity column (not encoded yet - encoding happens after split)
            return df_processed[['timestamp'] + self.sensor_columns + ['activity']]
        else:
            # For inference, return all columns including any extra features
            return df_processed

    def apply_scaling(self, df, room_name, is_training=False, scaler_fit_range=None):
        """
        Phase C: Apply scaling to preprocessed data.
        
        For training: Fit scaler on provided data and store it.
        For inference: Transform using previously fitted scaler.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Preprocessed DataFrame from preprocess_without_scaling()
        room_name : str
            Name of the room
        is_training : bool, default=False
            Whether this is training data (fit scaler) or inference (transform)
        scaler_fit_range : dict, optional
            For training, metadata about the fit range to persist:
            {'fit_start_ts': timestamp, 'fit_end_ts': timestamp, 'fit_sample_count': int}
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with scaled sensor columns and encoded activity (if training)
        """
        df_processed = df.copy()
        
        if is_training:
            if 'activity' not in df_processed.columns:
                raise ValueError(f"'activity' column missing in {room_name}")
            
            # Fit scaler on this data (should be train split only)
            scaler = StandardScaler()
            df_processed[self.sensor_columns] = scaler.fit_transform(df_processed[self.sensor_columns])
            self.scalers[room_name] = scaler
            
            # Log scaler fit metadata
            if scaler_fit_range:
                logger.info(
                    f"Fitted scaler for {room_name} on range [{scaler_fit_range.get('fit_start_ts')}, "
                    f"{scaler_fit_range.get('fit_end_ts')}] with {scaler_fit_range.get('fit_sample_count')} samples"
                )
            
            # Encode labels
            le = LabelEncoder()
            df_processed['activity_encoded'] = le.fit_transform(df_processed['activity'])
            self.label_encoders[room_name] = le
            
            return df_processed[['timestamp'] + self.sensor_columns + ['activity_encoded']]
        else:
            # Inference: use existing scaler
            if room_name not in self.scalers:
                raise ValueError(f"No scaler found for room: {room_name}")
            df_processed[self.sensor_columns] = self.scalers[room_name].transform(df_processed[self.sensor_columns])
            return df_processed

    def preprocess_with_resampling(self, df, room_name, is_training=False, 
                               apply_denoising=False, denoising_method='hampel',
                               denoising_window=5, denoising_threshold=3.0,
                               max_ffill_gap_seconds=None):
        """
        Enhanced preprocessing pipeline that handles missing data and spikes AFTER resampling.
        
        .. deprecated::
            This method fits scaler on full dataset which causes temporal leakage.
            Use preprocess_without_scaling() + apply_scaling() for training.
            Kept for backward compatibility and inference paths.
        
        Recommended order for data with varying intervals:
        1. Resample to fixed interval (if time-based processing enabled)
        2. Handle missing data (forward/backward fill on regular intervals)
        3. Apply denoising/spike removal (on regular intervals)
        4. Apply scaling
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with sensor data
        room_name : str
            Name of the room
        is_training : bool, default=False
            Whether this is training data
        apply_denoising : bool, default=False
            Whether to apply denoising/spike removal
        denoising_method : str, default='hampel'
            Denoising method: 'hampel' or 'clip'
        denoising_window : int, default=5
            Window size for denoising
        denoising_threshold : float, default=3.0
            Threshold for denoising
        max_ffill_gap_seconds : float | None, optional
            Maximum forward-fill bridge in seconds. If omitted, resolves from env.
        
        Returns:
        --------
        pd.DataFrame
            Preprocessed DataFrame ready for sequence creation
        """
        # Check if we should use the new leakage-free path for training
        use_train_split_scaling = (
            is_training and 
            os.getenv('ENABLE_TRAIN_SPLIT_SCALING', 'true').lower() in ('1', 'true', 'yes', 'on', 'enabled')
        )
        
        if use_train_split_scaling:
            raise ValueError(
                f"Leakage guard violation for {room_name}: preprocess_with_resampling(is_training=True) "
                "is not allowed when ENABLE_TRAIN_SPLIT_SCALING is enabled. "
                "Use preprocess_without_scaling() + apply_scaling() through the Beta 6 training path."
            )
        
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        raw_rows_before_resample = int(len(df_processed))
        if max_ffill_gap_seconds is None:
            max_ffill_gap_seconds = _resolve_max_ffill_gap_seconds(default_seconds=60.0)
        else:
            try:
                max_ffill_gap_seconds = float(max_ffill_gap_seconds)
            except (TypeError, ValueError):
                logger.warning(
                    f"Invalid injected max_ffill_gap_seconds={max_ffill_gap_seconds!r}; falling back to env/default."
                )
                max_ffill_gap_seconds = _resolve_max_ffill_gap_seconds(default_seconds=60.0)
        ffill_limit = _resolve_ffill_limit(self.data_interval, max_ffill_gap_seconds)
        
        # Step 1: Resample if time-based processing is enabled
        if self.enable_time_based_processing and 'timestamp' in df_processed.columns:
            try:
                logger.info(f"Resampling data for {room_name} to {self.data_interval} interval")
                
                # Analyze interval consistency before resampling
                from .preprocessing.resampling import validate_interval_consistency
                validation = validate_interval_consistency(df_processed)
                
                if not validation['is_consistent']:
                    logger.warning(f"Irregular intervals detected for {room_name}: {validation['warnings']}")
                
                # Perform resampling
                from .preprocessing.resampling import resample_to_fixed_interval
                df_processed = resample_to_fixed_interval(
                    df_processed, 
                    interval=self.data_interval.lower(),
                    method='linear',
                    timestamp_col='timestamp',
                    fill_method='ffill',
                    keep_original_timestamps=False,
                    max_ffill_gap_seconds=max_ffill_gap_seconds,
                )
                logger.info(f"Resampled {room_name}: {len(df)} -> {len(df_processed)} samples")
                
            except (ValueError, TypeError) as e:
                logger.error(f"Resampling failed for {room_name}: {e}.")
                raise ValueError(f"Resampling failed for {room_name}: {e}") from e
        
        # Step 2: Handle missing data (forward/backward fill)
        logger.info(f"Handling missing data for {room_name}")
        # For time-resampled paths, avoid a second forward-fill pass that would
        # silently extend gaps beyond the configured limit.
        should_forward_fill = not (self.enable_time_based_processing and 'timestamp' in df_processed.columns)
        if should_forward_fill:
            df_processed = df_processed.ffill(limit=ffill_limit)
        # Only backfill leading NaNs (before any observation) to avoid future leakage.
        for col in df_processed.columns:
            if col == 'timestamp':
                continue
            if not df_processed[col].isna().any():
                continue
            first_valid = df_processed[col].first_valid_index()
            if first_valid is None:
                continue
            df_processed.loc[:first_valid, col] = df_processed.loc[:first_valid, col].bfill()
        
        # Step 2.5: Add temporal features for time-aware predictions
        logger.info(f"Adding temporal features for {room_name}")
        df_processed = add_temporal_features(df_processed)
        
        # Step 2.6: Add rolling context features
        # Get room-specific window from config if available
        from config import get_room_config
        try:
            room_config = get_room_config()
            room_specific_window = room_config.get_sequence_window(room_name)
            room_interval = room_config.get_data_interval(room_name)
            logger.info(f"Adding rolling features for {room_name} with room-specific window={room_specific_window}s")
            df_processed = add_rolling_features(
                df_processed, 
                DEFAULT_PHYSICAL_SENSORS,  # Only compute for physical sensors
                room_specific_window=room_specific_window,
                standard_windows=[90, 360],  # 1.5min and 6min standard windows
                interval=room_interval or self.data_interval,
            )
        except Exception as e:
            logger.warning(f"Could not add rolling features: {e}. Proceeding without rolling features.")

        
        # Step 3: Apply denoising if requested (on regular intervals)
        if apply_denoising:
            logger.info(f"Applying {denoising_method} denoising to {room_name}")
            if denoising_method.lower() == 'hampel':
                df_processed = hampel_filter(df_processed, self.sensor_columns,
                                            window=denoising_window, n_sigmas=denoising_threshold)
                logger.info(f"Applied Hampel filter to {room_name} data")
            elif denoising_method.lower() == 'clip':
                df_processed = clip_outliers(df_processed, self.sensor_columns,
                                            method='mad', factor=denoising_threshold,
                                            window=denoising_window)
                logger.info(f"Applied MAD clipping to {room_name} data")

        # Drop unresolved sensor gaps rather than hallucinating values across long outages.
        available_sensor_cols = [c for c in self.sensor_columns if c in df_processed.columns]
        if available_sensor_cols:
            missing_sensor_mask = df_processed[available_sensor_cols].isna().any(axis=1)
            if bool(missing_sensor_mask.any()):
                dropped_rows = int(missing_sensor_mask.sum())
                logger.info(
                    f"Dropping {dropped_rows} rows with unresolved sensor gaps for {room_name} "
                    f"(ffill_limit={ffill_limit}, max_gap_seconds={max_ffill_gap_seconds})."
                )
                df_processed = df_processed.loc[~missing_sensor_mask].copy()
        df_processed.attrs["raw_rows_before_resample"] = raw_rows_before_resample
        df_processed.attrs["rows_after_gap_drop"] = int(len(df_processed))
        if df_processed.empty:
            if is_training:
                raise ValueError(
                    f"No usable samples remain after bounded gap handling for {room_name}; "
                    "check data continuity or relax MAX_RESAMPLE_FFILL_GAP_SECONDS."
                )
            return df_processed
        
        # Step 4: Apply scaling
        if is_training:
            if 'activity' not in df_processed.columns:
                raise ValueError(f"'activity' column missing in {room_name}")
            
            scaler = StandardScaler()
            df_processed[self.sensor_columns] = scaler.fit_transform(df_processed[self.sensor_columns])
            self.scalers[room_name] = scaler
            
            le = LabelEncoder()

            # Normalize labels before encoding to prevent inconsistent classes.
            # Drop empty/NaN labels rather than encoding them as a new class.
            activity_raw = df_processed['activity']
            valid_mask = activity_raw.notna() & (activity_raw.astype(str).str.strip() != "")
            if not valid_mask.all():
                dropped = int((~valid_mask).sum())
                logger.warning(f"Dropping {dropped} rows with missing/empty activity labels for {room_name}")
                df_processed = df_processed.loc[valid_mask].copy()

            # Canonicalize labels to the shared taxonomy when available.
            try:
                from utils.segment_utils import normalize_activity_name, validate_activity_for_room
                df_processed['activity'] = df_processed['activity'].apply(
                    lambda a: validate_activity_for_room(normalize_activity_name(str(a).strip().lower()), room_name)
                )
            except Exception as e:
                logger.warning(f"Could not import activity normalization utilities; using lowercase/strip only: {e}")
                df_processed['activity'] = df_processed['activity'].astype(str).str.strip().str.lower()

            # Log a capped set of labels for debugging (avoid log spam).
            unique_labels = sorted(pd.unique(df_processed['activity']).tolist())
            preview = unique_labels[:30]
            suffix = "..." if len(unique_labels) > 30 else ""
            logger.info(f"Activity labels for {room_name} ({len(unique_labels)}): {preview}{suffix}")
            
            df_processed['activity_encoded'] = le.fit_transform(df_processed['activity'])
            self.label_encoders[room_name] = le
            
            return df_processed[['timestamp'] + self.sensor_columns + ['activity_encoded']]
        else:
            if room_name not in self.scalers:
                raise ValueError(f"No scaler found for room: {room_name}")
            df_processed[self.sensor_columns] = self.scalers[room_name].transform(df_processed[self.sensor_columns])
            # Return the entire processed DataFrame (with timestamp and sensor columns)
            logger.info(f"[DEBUG] Columns in df_processed: {list(df_processed.columns)}")
            logger.info(f"[DEBUG] Sensor columns expected: {self.sensor_columns}")
            logger.info(f"[DEBUG] Returning shape: {df_processed.shape}")
            return df_processed

    def predict_room_activities(self, room_name, df, seq_length=50, confidence_threshold=0.8, 
                               use_entropy=False, entropy_threshold=1.0,
                               apply_denoising=False, denoising_method='hampel', 
                               denoising_window=5, denoising_threshold=3.0,
                               process_after_resample=True):
        """
        Predict activities with enhanced preprocessing pipeline.
        New parameter:
        - process_after_resample: bool, default=True
            If True, handles missing data and spikes AFTER resampling (recommended for varying intervals)
            If False, uses original processing order (backward compatibility)
        """
        # Defensive checks for required components
        if room_name not in self.room_models:
            logger.error(f"No model found for room: {room_name}")
            raise ValueError(f"No model trained for room: {room_name}. Train or load a model first.")
        if room_name not in self.scalers:
            logger.error(f"No scaler found for room: {room_name}")
            raise ValueError(f"No scaler available for room: {room_name}. Train or load a model first.")
        if room_name not in self.label_encoders:
            logger.error(f"No label encoder found for room: {room_name}")
            raise ValueError(f"No label encoder available for room: {room_name}. Train or load a model first.")
        logger.info(f"Starting prediction for room '{room_name}': input data shape = {df.shape}, sequence length = {seq_length}")
            # If time-based processing is enabled, calculate seq_length from time window and interval
        if self.enable_time_based_processing:
            # P1 Fix: Use room-specific sequence length from config to match training window
            from config import get_room_config
            try:
                room_config = get_room_config()
                seq_length = room_config.calculate_seq_length(room_name)
                logger.info(f"Time-based processing enabled: Using room-specific seq_length={seq_length} for {room_name}")
            except Exception as e:
                logger.warning(f"Failed to get room config for {room_name}, falling back to global default: {e}")
                seq_length = calculate_samples_from_time(self.sequence_time_window, self.data_interval.lower())
                logger.info(f"Fallback: calculated seq_length={seq_length} from global window={self.sequence_time_window}")
        # Check sequence length vs data size
        if len(df) < seq_length:
            logger.warning(f"Data length ({len(df)}) < sequence length ({seq_length}) for {room_name}")
            # Create empty DataFrame with consistent schema including timestamp if present
            empty_df = pd.DataFrame({
                'predicted_activity': pd.Series(dtype=object),
                'confidence': pd.Series(dtype=float),
                'entropy': pd.Series(dtype=float)
            })
            if 'timestamp' in df.columns:
                empty_df['timestamp'] = pd.Series(dtype='datetime64[ns]')
            return empty_df
        try:
            # Choose preprocessing pipeline based on process_after_resample flag
            if process_after_resample:
                logger.info(f"Using enhanced preprocessing pipeline for {room_name} (process_after_resample=True)")
                processed_df = self.preprocess_with_resampling(
                    df, room_name, is_training=False,
                    apply_denoising=apply_denoising,
                    denoising_method=denoising_method,
                    denoising_window=denoising_window,
                    denoising_threshold=denoising_threshold
                )
                logger.info(f"After preprocessing with resampling: shape = {processed_df.shape}")
                logger.info(f"Processed data columns: {list(processed_df.columns)}")
            else:
                logger.info(f"Using original preprocessing pipeline for {room_name} (process_after_resample=False)")
                # Original processing order: denoising before scaling
                if apply_denoising:
                    df_denoised = df.copy()
                    if denoising_method.lower() == 'hampel':
                        df_denoised = hampel_filter(df_denoised, self.sensor_columns,
                                                    window=denoising_window, n_sigmas=denoising_threshold)
                        logger.info(f"Applied Hampel filter to {room_name} data before prediction")
                    elif denoising_method.lower() == 'clip':
                        df_denoised = clip_outliers(df_denoised, self.sensor_columns,
                                                    method='mad', factor=denoising_threshold,
                                                    window=denoising_window)
                        logger.info(f"Applied MAD clipping to {room_name} data before prediction")
                    processed_df = self.preprocess_room_data(df_denoised, room_name, is_training=False)
                else:
                    processed_df = self.preprocess_room_data(df, room_name, is_training=False)
                logger.info(f"After original preprocessing: shape = {processed_df.shape}")
                logger.info(f"Processed data columns: {list(processed_df.columns)}")
            # Ensure all sensor columns are present
            # Select only sensor columns for model input
            if set(self.sensor_columns).issubset(set(processed_df.columns)):
                sensor_data = processed_df[self.sensor_columns]
                logger.info(f"Selected sensor columns for prediction: {list(sensor_data.columns)}")
            else:
                # Try to match columns case-insensitively
                available_cols = processed_df.columns.tolist()
                sensor_cols = []
                for col in self.sensor_columns:
                    matches = [c for c in available_cols if c.lower() == col.lower()]
                    if matches:
                        sensor_cols.append(matches[0])
                    else:
                        raise ValueError(f"Sensor column '{col}' not found in processed data. Available columns: {available_cols}")
                sensor_data = processed_df[sensor_cols]
                logger.info(f"Selected sensor columns (case-insensitive match): {list(sensor_data.columns)}")
            
            # Safe numeric conversion for processed_df
            try:
                data = sensor_data.to_numpy(dtype=float)
                logger.info(f"Converted processed data to numpy array: shape = {data.shape}, dtype = {data.dtype}")
            except Exception:
                data = np.asarray(sensor_data.values, dtype=float)
                logger.info(f"Converted processed data with np.asarray: shape = {data.shape}, dtype = {data.dtype}")
            
            X = create_sequences(data, seq_length)
            logger.info(f"Created sequences: X shape = {X.shape}, seq_length = {seq_length}")
            
            if len(X) == 0:
                logger.warning("Not enough data in %s for prediction", room_name)
                # Create empty DataFrame with consistent schema including timestamp if present
                empty_df = pd.DataFrame({
                    'predicted_activity': pd.Series(dtype=object),
                    'confidence': pd.Series(dtype=float),
                    'entropy': pd.Series(dtype=float)
                })
                if 'timestamp' in df.columns:
                    empty_df['timestamp'] = pd.Series(dtype='datetime64[ns]')
                return empty_df

            # Get prediction probabilities with error handling for verbose parameter
            model = self.room_models[room_name]
            try:
                predictions = model.predict(X, verbose=0)
            except TypeError:
                # Some model adapters (joblib, custom wrappers) may not accept verbose parameter
                predictions = model.predict(X)
            except (ValueError, tf.errors.OpError) as e:
                logger.error(f"Model prediction failed for {room_name}: {e}")
                raise RuntimeError(f"Prediction failed for room {room_name}: {str(e)}")
            max_prob = np.max(predictions, axis=1)
            predicted_classes = np.argmax(predictions, axis=1)

            # Entropy if requested - use np.nan instead of None for numeric consistency
            if use_entropy:
                entropy_values = calculate_entropy(predictions)
            else:
                entropy_values = np.full(len(predictions), np.nan)

            predicted_activities, confidence_scores, entropy_scores = [], [], []

            # Get label encoder for safe decoding
            le = self.label_encoders[room_name]
            room_thresholds = self.class_thresholds.get(room_name, {}) if hasattr(self, 'class_thresholds') else {}

            for i, prob in enumerate(max_prob):
                class_idx = int(predicted_classes[i])
                class_threshold = room_thresholds.get(
                    str(class_idx),
                    room_thresholds.get(class_idx, confidence_threshold),
                )
                try:
                    class_threshold = float(class_threshold)
                except (TypeError, ValueError):
                    class_threshold = float(confidence_threshold)

                is_low_confidence = prob < class_threshold
                if use_entropy and not np.isnan(entropy_values[i]):
                    is_low_confidence |= (entropy_values[i] > entropy_threshold)

                if is_low_confidence:
                    predicted_activities.append("low_confidence")
                else:
                    # Safe label decoding with class index validation
                    if class_idx < 0 or class_idx >= len(le.classes_):
                        logger.warning(f"Class index {class_idx} out of range [0, {len(le.classes_)-1}] for {room_name}")
                        predicted_activities.append("unknown")
                    else:
                        try:
                            predicted_activities.append(le.inverse_transform([class_idx])[0])
                        except Exception as e:
                            logger.error(f"Label decoding failed for class {class_idx} in {room_name}: {e}")
                            predicted_activities.append("unknown")

                confidence_scores.append(prob)
                entropy_scores.append(entropy_values[i])

            # Build result dataframe with proper alignment
            # Use processed_df (which includes timestamp) to ensure correct row count
            prediction_count = len(predicted_activities)
            
            # Determine which dataframe to use for timestamps and sensor data
            base_df = processed_df if 'timestamp' in processed_df.columns else df
            
            if len(base_df) >= seq_length + prediction_count - 1:
                # Slice base_df to match predictions
                result_df = base_df.iloc[seq_length-1:seq_length-1+prediction_count].copy().reset_index(drop=True)
                # Ensure we have the right number of rows
                if len(result_df) == prediction_count:
                    result_df['predicted_activity'] = predicted_activities
                    result_df['confidence'] = confidence_scores
                    result_df['entropy'] = entropy_scores
                else:
                    # Fallback: create new dataframe
                    logger.warning(f"Row count mismatch: base_df slice gave {len(result_df)} rows, expected {prediction_count}")
                    result_df = pd.DataFrame({
                        'predicted_activity': predicted_activities,
                        'confidence': confidence_scores,
                        'entropy': entropy_scores
                    })
                    if 'timestamp' in base_df.columns:
                        result_df['timestamp'] = base_df['timestamp'].iloc[seq_length-1:seq_length-1+prediction_count].values
            else:
                # Not enough rows in base_df, create new dataframe
                logger.warning(f"Insufficient rows in base_df for alignment: {len(base_df)} < {seq_length + prediction_count - 1}")
                result_df = pd.DataFrame({
                    'predicted_activity': predicted_activities,
                    'confidence': confidence_scores,
                    'entropy': entropy_scores
                })
                if 'timestamp' in base_df.columns and len(base_df) >= seq_length + prediction_count - 1:
                    result_df['timestamp'] = base_df['timestamp'].iloc[seq_length-1:seq_length-1+prediction_count].values
                elif 'timestamp' in df.columns and len(df) >= seq_length + prediction_count - 1:
                    # Fallback to original df
                    result_df['timestamp'] = df['timestamp'].iloc[seq_length-1:seq_length-1+prediction_count].values
            
            logger.info(f"Result dataframe created with {len(result_df)} rows, {prediction_count} predictions")
            return result_df
        except (ValueError, KeyError, RuntimeError) as e:
            logger.error(f"Prediction failed for room {room_name}: {e}")
            # Return empty dataframe with consistent schema including timestamp if present
            empty_df = pd.DataFrame({
                'predicted_activity': pd.Series(dtype=object),
                'confidence': pd.Series(dtype=float),
                'entropy': pd.Series(dtype=float)
            })
            if 'timestamp' in df.columns:
                empty_df['timestamp'] = pd.Series(dtype='datetime64[ns]')
            return empty_df
    def detect_anomalies(self, room_name, df, z_threshold=3.0, temporal_threshold=2.0):
        """
        Detect anomalies in prediction data using combined methods.
        
        Parameters:
        - room_name: str, name of the room
        - df: DataFrame with sensor data (should be the same as prediction input)
        - z_threshold: float, Z-score threshold for statistical anomalies
        - temporal_threshold: float, threshold for temporal anomalies
        
        Returns:
        - anomaly_df: DataFrame with anomaly detection results
        """
        try:
            # Update detector settings
            self.anomaly_detector.z_threshold = z_threshold
            
            # Run combined anomaly detection
            anomaly_results = self.anomaly_detector.detect_combined(df)
            
            # Add room name to results
            anomaly_results['room'] = room_name
            
            # Ensure anomaly results have expected columns
            if 'final_anomaly' not in anomaly_results.columns:
                # Create final_anomaly column if missing
                anomaly_cols = [col for col in anomaly_results.columns if '_anomaly' in col]
                if anomaly_cols:
                    anomaly_results['final_anomaly'] = anomaly_results[anomaly_cols].any(axis=1)
                else:
                    anomaly_results['final_anomaly'] = False
            
            if 'final_anomaly_score' not in anomaly_results.columns:
                anomaly_results['final_anomaly_score'] = 0.0
            
            # Safer merging with proper reindexing
            if not anomaly_results.empty:
                # Reindex anomaly results to match df.index
                anomaly_results = anomaly_results.reindex(df.index)
                
                # Ensure numeric columns remain numeric after fillna
                if 'final_anomaly_score' in anomaly_results.columns:
                    anomaly_results['final_anomaly_score'] = anomaly_results['final_anomaly_score'].fillna(0.0).astype(float)
                if 'final_anomaly' in anomaly_results.columns:
                    anomaly_results['final_anomaly'] = anomaly_results['final_anomaly'].fillna(False).astype(bool)
                
                # Fill remaining NaN values with appropriate defaults
                anomaly_results = anomaly_results.fillna(False)
                
                # Merge with original dataframe
                result_df = df.copy()
                for col in anomaly_results.columns:
                    if col not in result_df.columns or col == 'room':
                        result_df[col] = anomaly_results[col]
                
                logger.info(f"Anomaly detection completed for {room_name}: {anomaly_results['final_anomaly'].sum() if 'final_anomaly' in anomaly_results.columns else 0} anomalies found")
                return result_df
            else:
                logger.warning(f"No anomaly results for {room_name}")
                # Return original dataframe with default anomaly columns
                result_df = df.copy()
                result_df['final_anomaly'] = False
                result_df['final_anomaly_score'] = 0.0
                result_df['room'] = room_name
                return result_df
                
        except Exception as e:
            logger.error(f"Anomaly detection failed for {room_name}: {e}")
            # Return original dataframe with default anomaly columns
            result_df = df.copy()
            result_df['final_anomaly'] = False
            result_df['final_anomaly_score'] = 0.0
            result_df['room'] = room_name
            return result_df
    
    def predict_with_anomalies(self, room_name, df, seq_length=50, confidence_threshold=0.8,
                              z_threshold=3.0, temporal_threshold=2.0):
        """
        Predict activities and detect anomalies in one combined operation.
        
        Returns:
        - result_df: DataFrame with predictions, confidence, and anomaly detection results
        """
        try:
            # First predict activities
            prediction_df = self.predict_room_activities(room_name, df, seq_length, confidence_threshold)
            
            if prediction_df.empty:
                logger.warning(f"Prediction returned empty dataframe for {room_name}")
                return prediction_df
            
            # Then detect anomalies on the original data
            anomaly_df = self.detect_anomalies(room_name, df, z_threshold, temporal_threshold)
            
            if anomaly_df.empty:
                logger.warning(f"Anomaly detection returned empty dataframe for {room_name}")
                # Return predictions without anomaly columns
                return prediction_df
            
            # Merge anomaly results with predictions
            # We need to align by timestamp since anomaly detection works on full dataset
            # and prediction works on seq_length-1: subset
            if 'timestamp' in prediction_df.columns and 'timestamp' in anomaly_df.columns:
                # Ensure timestamps are properly aligned
                prediction_timestamps = prediction_df['timestamp'].reset_index(drop=True)
                anomaly_timestamps = anomaly_df['timestamp'].reset_index(drop=True)
                
                # Find matching timestamps (prediction starts at seq_length-1)
                offset = seq_length - 1
                if len(anomaly_df) >= offset:
                    # Get the subset of anomaly data that corresponds to prediction timestamps
                    anomaly_subset = anomaly_df.iloc[offset:offset+len(prediction_df)].reset_index(drop=True)
                    
                    # Verify timestamp alignment
                    if len(prediction_df) == len(anomaly_subset):
                        # Check if timestamps match (within tolerance)
                        time_diff = (prediction_timestamps - anomaly_subset['timestamp']).abs()
                        max_diff = time_diff.max()
                        
                        if max_diff > pd.Timedelta('1s'):  # More than 1 second difference
                            logger.warning(f"Timestamp misalignment in {room_name}: max diff = {max_diff}")
                            # Fall back to index-based alignment
                            result_df = prediction_df.copy()
                            # Add column existence checks with defaults
                            for col in ['combined_anomaly', 'anomaly_type']:
                                if col not in anomaly_subset.columns:
                                    anomaly_subset[col] = False if col == 'combined_anomaly' else 'normal'
                            
                            # Add all anomaly columns
                            for col in ['combined_anomaly', 'anomaly_type'] + \
                                      [c for c in anomaly_subset.columns if c.endswith('_anomaly')]:
                                if col in anomaly_subset.columns:
                                    result_df[col] = anomaly_subset[col].reset_index(drop=True)
                        else:
                            # Timestamps align well, merge on timestamp
                            result_df = prediction_df.merge(
                                anomaly_subset[['timestamp', 'combined_anomaly', 'anomaly_type'] + 
                                              [col for col in anomaly_subset.columns if col.endswith('_anomaly')]],
                                on='timestamp',
                                how='left'
                            )
                    else:
                        # Length mismatch, use index-based alignment
                        logger.warning(f"Length mismatch in {room_name}: prediction={len(prediction_df)}, anomaly={len(anomaly_subset)}")
                        result_df = prediction_df.copy()
                        for col in ['combined_anomaly', 'anomaly_type']:
                            if col in anomaly_df.columns:
                                result_df[col] = False if col == 'combined_anomaly' else 'normal'
                else:
                    # Not enough anomaly data
                    logger.warning(f"Not enough anomaly data for {room_name}: {len(anomaly_df)} < offset {offset}")
                    result_df = prediction_df.copy()
                    result_df['combined_anomaly'] = False
                    result_df['anomaly_type'] = 'normal'
            else:
                # No timestamp columns, use index-based alignment
                logger.warning(f"No timestamp columns found for alignment in {room_name}")
                result_df = prediction_df.copy()
                offset = seq_length - 1
                for col in ['combined_anomaly', 'anomaly_type'] + \
                          [c for c in anomaly_df.columns if c.endswith('_anomaly')]:
                    if col in anomaly_df.columns:
                        if len(anomaly_df) >= offset + len(prediction_df):
                            result_df[col] = anomaly_df[col].iloc[offset:offset+len(prediction_df)].reset_index(drop=True)
                        else:
                            result_df[col] = False if col == 'combined_anomaly' else 'normal'
            
            # Fill NaN values
            if 'combined_anomaly' in result_df.columns:
                result_df['combined_anomaly'] = result_df['combined_anomaly'].fillna(False)
            if 'anomaly_type' in result_df.columns:
                result_df['anomaly_type'] = result_df['anomaly_type'].fillna('normal')
            
            logger.info(f"Successfully combined predictions and anomalies for {room_name}")
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to combine predictions and anomalies for {room_name}: {e}")
            # Return predictions without anomaly data
            return self.predict_room_activities(room_name, df, seq_length, confidence_threshold)

    def generate_unified_timeline(self, room_names, training_data=None, prediction_data=None, 
                                start_time=None, end_time=None, hide_activities=None, show_low_confidence=False):
        if hide_activities is None:
            hide_activities = set()
        else:
            hide_activities = set(hide_activities)
            
        all_timestamps = []
        filtered_training = {}
        filtered_prediction = {}
        
        # Validate and convert start_time and end_time to datetime if they're not None
        if start_time is not None and not isinstance(start_time, (datetime, pd.Timestamp)):
            try:
                start_time = pd.to_datetime(start_time)
                logger.info(f"Converted start_time to datetime: {start_time}")
            except Exception as e:
                logger.warning(f"Failed to convert start_time to datetime: {e}")
                start_time = None
        
        if end_time is not None and not isinstance(end_time, (datetime, pd.Timestamp)):
            try:
                end_time = pd.to_datetime(end_time)
                logger.info(f"Converted end_time to datetime: {end_time}")
            except Exception as e:
                logger.warning(f"Failed to convert end_time to datetime: {e}")
                end_time = None
        
        # Filter training data
        if training_data:
            for room, df in training_data.items():
                df = df.copy()
                
                # DEBUG: Log original data
                if 'activity' in df.columns:
                    logger.info(f"🔍 DEBUG Timeline [{room}] BEFORE processing:")
                    # Handle mixed types safely
                    activity_vals = df['activity'].dropna().astype(str)
                    logger.info(f"   Original unique activities: {sorted(activity_vals.unique())}")
                    logger.info(f"   Original value counts: {df['activity'].value_counts().to_dict()}")
                
                # Ensure timestamp column is datetime
                if 'timestamp' in df.columns:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        logger.info(f"Converted timestamp column to datetime for room {room}")
                    except Exception as e:
                        logger.error(f"Failed to convert timestamp column to datetime for room {room}: {e}")
                        continue
                
                if start_time is not None or end_time is not None:
                    mask = pd.Series([True] * len(df), index=df.index)
                    if start_time is not None:
                        mask &= df['timestamp'] >= start_time
                    if end_time is not None:
                        mask &= df['timestamp'] <= end_time
                    df = df[mask]
                    logger.info(f"   After time filtering: {len(df)} rows")
                
                # Ensure activity column contains strings for filtering
                if 'activity' in df.columns:
                    try:
                        # DEBUG: Log before string conversion (handle mixed types safely)
                        before_vals = df['activity'].dropna().astype(str).unique()
                        logger.info(f"   Before .astype(str): {sorted(before_vals)}")
                        
                        df['activity'] = df['activity'].astype(str)
                        
                        # DEBUG: Log after string conversion
                        logger.info(f"   After .astype(str): {sorted(df['activity'].unique())}")
                        logger.info(f"   Hidden activities filter: {hide_activities}")
                        
                        df = df[~df['activity'].isin(hide_activities)]
                        
                        # DEBUG: Log after filtering
                        logger.info(f"   After hiding activities: {sorted(df['activity'].unique())}")
                        logger.info(f"   Final value counts: {df['activity'].value_counts().to_dict()}")
                    except Exception as e:
                        logger.error(f"Failed to process activity column for room {room}: {e}")
                        continue
                
                if not df.empty:
                    filtered_training[room] = df
                    all_timestamps.extend(df['timestamp'])
        
        # Filter prediction data
        if prediction_data:
            logger.info(f"DEBUG generate_prediction_timeline: processing {len(prediction_data)} rooms")
            for room, df in prediction_data.items():
                logger.info(f"DEBUG processing room {room}: shape {df.shape}, columns {list(df.columns)}")
                df = df.copy()
                
                # Ensure timestamp column is datetime
                if 'timestamp' in df.columns:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        logger.info(f"DEBUG room {room}: timestamp conversion successful")
                    except Exception as e:
                        logger.error(f"Failed to convert timestamp column to datetime for room {room}: {e}")
                        continue
                
                if start_time is not None or end_time is not None:
                    logger.info(f"DEBUG room {room}: applying time filter start={start_time}, end={end_time}")
                    mask = pd.Series([True] * len(df), index=df.index)
                    if start_time is not None:
                        mask &= df['timestamp'] >= start_time
                    if end_time is not None:
                        mask &= df['timestamp'] <= end_time
                    df = df[mask]
                    logger.info(f"DEBUG room {room}: after time filtering shape {df.shape}")
                else:
                    logger.info(f"DEBUG room {room}: no time filtering applied")
                
                # Ensure predicted_activity column contains strings for filtering
                if 'predicted_activity' in df.columns:
                    try:
                        df['predicted_activity'] = df['predicted_activity'].astype(str)
                        # Always hide low_confidence if show_low_confidence is False
                        if not show_low_confidence:
                            df = df[df['predicted_activity'] != 'low_confidence']
                        df = df[~df['predicted_activity'].isin(hide_activities)]
                    except Exception as e:
                        logger.error(f"Failed to process predicted_activity column for room {room}: {e}")
                        continue
                
                if not df.empty:
                    filtered_prediction[room] = df
                    all_timestamps.extend(df['timestamp'])
        
        if not all_timestamps:
            fig = go.Figure()
            fig.update_layout(title="No data to display", height=300)
            return fig, {}, {}
            
        # Get room positions
        room_positions = {room: i for i, room in enumerate(sorted(room_names))}
        
        # Get all unique activities for coloring
        all_activities = set()
        if filtered_training:
            for df in filtered_training.values():
                all_activities.update(df['activity'].unique())
        if filtered_prediction:
            for df in filtered_prediction.values():
                all_activities.update(df['predicted_activity'].unique())
        
        color_cycle = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
        ]
        activity_colors = {
            act: color_cycle[i % len(color_cycle)] 
            for i, act in enumerate(sorted(all_activities))
        }
        # Add special color for low_confidence (though it's usually hidden)
        activity_colors['low_confidence'] = '#808080'  # Gray
        
        fig = go.Figure()
        
        # Plot training data
        if filtered_training:
            for room, df in filtered_training.items():
                y_val = room_positions[room]
                for activity in df['activity'].unique():
                    if activity in hide_activities:
                        continue
                    mask = df['activity'] == activity
                    fig.add_trace(go.Scatter(
                        x=df.loc[mask, 'timestamp'],
                        y=[y_val] * mask.sum(),
                        mode='markers',
                        marker=dict(color=activity_colors[activity], size=8, symbol='circle'),
                        name=f"{activity} (Training)",
                        legendgroup=activity,
                        showlegend=True
                    ))
        
        # Plot prediction data
        if filtered_prediction:
            for room, df in filtered_prediction.items():
                y_val = room_positions[room]
                for activity in df['predicted_activity'].unique():
                    if activity in hide_activities:
                        continue
                    mask = df['predicted_activity'] == activity
                    if activity == 'low_confidence':
                        # Only shown if explicitly enabled
                        fig.add_trace(go.Scatter(
                            x=df.loc[mask, 'timestamp'],
                            y=[y_val] * mask.sum(),
                            mode='markers',
                            marker=dict(color=activity_colors['low_confidence'], size=8, symbol='diamond'),
                            name="Low Confidence",
                            legendgroup="low_confidence",
                            showlegend=True
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=df.loc[mask, 'timestamp'],
                            y=[y_val] * mask.sum(),
                            mode='markers',
                            marker=dict(color=activity_colors.get(activity, '#000000'), size=6, symbol='square'),
                            name=f"{activity} (Prediction)",
                            legendgroup=activity,
                            showlegend=False
                        ))
        
        fig.update_layout(
            title="🏠 Unified Elderly Activity Timeline (Per Room)",
            height=150 + 60 * len(room_positions),
            hovermode='x unified',
            margin=dict(t=50, b=50, l=100),
            yaxis=dict(
                tickmode='array',
                tickvals=list(room_positions.values()),
                ticktext=list(room_positions.keys()),
                title="Room"
            ),
            xaxis_title="Time",
            legend_title="Activities"
        )
        
        return fig, filtered_training, filtered_prediction

    def compute_reference_patterns(self):
        """Compute dynamic reference patterns from training data"""
        if not self.training_data:
            return {}
        
        all_patterns = {}
        
        for room, df in self.training_data.items():
            if 'activity' not in df.columns:
                continue
                
            # Group by activity
            grouped = df.groupby('activity')
            
            for activity, group in grouped:
                if activity not in all_patterns:
                    all_patterns[activity] = {}
                
                # Process each sensor
                for sensor in self.sensor_columns:
                    if sensor not in group.columns:
                        continue
                        
                    values = group[sensor].dropna()
                    if len(values) < 2:  # Need at least 2 points for delta
                        continue
                    
                    # Compute deltas (current - previous)
                    deltas = values.diff().iloc[1:]  # Skip first NaN
                    prev_values = values.iloc[:-1]
                    
                    # Compute percentage changes (handle division by zero)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        pct_changes = (deltas / prev_values) * 100
                        pct_changes = pct_changes.replace([np.inf, -np.inf], np.nan)
                    
                    # Calculate averages (ignore NaNs)
                    avg_value = np.nanmean(values)
                    avg_delta = np.nanmean(deltas)
                    avg_pct_change = np.nanmean(pct_changes)
                    
                    # Initialize activity dict if needed
                    if sensor not in all_patterns[activity]:
                        all_patterns[activity][sensor] = {
                            'avg_value': [],
                            'avg_delta': [],
                            'avg_pct_change': []
                        }
                    
                    # Store values for aggregation across rooms
                    all_patterns[activity][sensor]['avg_value'].append(avg_value)
                    all_patterns[activity][sensor]['avg_delta'].append(avg_delta)
                    all_patterns[activity][sensor]['avg_pct_change'].append(avg_pct_change)
        
        # Aggregate across rooms (average the averages)
        final_patterns = {}
        for activity, sensors in all_patterns.items():
            final_patterns[activity] = {}
            for sensor, stats in sensors.items():
                final_patterns[activity][sensor] = {
                    'avg_value': float(np.nanmean(stats['avg_value'])),
                    'avg_delta': float(np.nanmean(stats['avg_delta'])),
                    'avg_pct_change': float(np.nanmean(stats['avg_pct_change']))
                }
        
        return final_patterns

    def generate_training_timeline(self, room_names, training_data=None,
                                  start_time=None, end_time=None, hide_activities=None):
        """
        Generate timeline visualization for training data only.
        
        Parameters:
        - room_names: list of room names to include
        - training_data: dict of room->DataFrame with training data
        - start_time: datetime for filtering start
        - end_time: datetime for filtering end
        - hide_activities: set of activities to hide from display
        
        Returns:
        - fig: Plotly figure object
        - filtered_training: dict of filtered training data
        """
        if hide_activities is None:
            hide_activities = set()
        else:
            hide_activities = set(hide_activities)
            
        all_timestamps = []
        filtered_training = {}
        
        # Validate and convert start_time and end_time to datetime if they're not None
        if start_time is not None and not isinstance(start_time, (datetime, pd.Timestamp)):
            try:
                start_time = pd.to_datetime(start_time)
                logger.info(f"Converted start_time to datetime: {start_time}")
            except Exception as e:
                logger.warning(f"Failed to convert start_time to datetime: {e}")
                start_time = None
        
        if end_time is not None and not isinstance(end_time, (datetime, pd.Timestamp)):
            try:
                end_time = pd.to_datetime(end_time)
                logger.info(f"Converted end_time to datetime: {end_time}")
            except Exception as e:
                logger.warning(f"Failed to convert end_time to datetime: {e}")
                end_time = None
        
        # Filter training data
        if training_data:
            for room, df in training_data.items():
                df = df.copy()
                
                # DEBUG: Log original data
                if 'activity' in df.columns:
                    logger.info(f"🔍 DEBUG Training Timeline [{room}] BEFORE processing:")
                    # Handle mixed types safely
                    activity_vals = df['activity'].dropna().astype(str)
                    logger.info(f"   Original unique activities: {sorted(activity_vals.unique())}")
                    logger.info(f"   Original value counts: {df['activity'].value_counts().to_dict()}")
                
                # Ensure timestamp column is datetime
                if 'timestamp' in df.columns:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        logger.info(f"Converted timestamp column to datetime for room {room}")
                    except Exception as e:
                        logger.error(f"Failed to convert timestamp column to datetime for room {room}: {e}")
                        continue
                
                if start_time is not None or end_time is not None:
                    mask = pd.Series([True] * len(df), index=df.index)
                    if start_time is not None:
                        mask &= df['timestamp'] >= start_time
                    if end_time is not None:
                        mask &= df['timestamp'] <= end_time
                    df = df[mask]
                    logger.info(f"   After time filtering: {len(df)} rows")
                
                # Ensure activity column contains strings for filtering
                if 'activity' in df.columns:
                    try:
                        # DEBUG: Log before string conversion (handle mixed types safely)
                        before_vals = df['activity'].dropna().astype(str).unique()
                        logger.info(f"   Before .astype(str): {sorted(before_vals)}")
                        
                        df['activity'] = df['activity'].astype(str)
                        
                        # DEBUG: Log after string conversion
                        logger.info(f"   After .astype(str): {sorted(df['activity'].unique())}")
                        logger.info(f"   Hidden activities filter: {hide_activities}")
                        
                        df = df[~df['activity'].isin(hide_activities)]
                        
                        # DEBUG: Log after filtering
                        logger.info(f"   After hiding activities: {sorted(df['activity'].unique())}")
                        logger.info(f"   Final value counts: {df['activity'].value_counts().to_dict()}")
                    except Exception as e:
                        logger.error(f"Failed to process activity column for room {room}: {e}")
                        continue
                
                if not df.empty:
                    filtered_training[room] = df
                    all_timestamps.extend(df['timestamp'])
        
        if not all_timestamps:
            fig = go.Figure()
            fig.update_layout(title="No training data to display", height=300)
            return fig, {}
            
        # Get room positions
        room_positions = {room: i for i, room in enumerate(sorted(room_names))}
        
        # Get all unique activities for coloring
        all_activities = set()
        if filtered_training:
            for df in filtered_training.values():
                all_activities.update(df['activity'].unique())
        
        color_cycle = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
        ]
        activity_colors = {
            act: color_cycle[i % len(color_cycle)] 
            for i, act in enumerate(sorted(all_activities))
        }
        
        fig = go.Figure()
        
        # Plot training data
        if filtered_training:
            for room, df in filtered_training.items():
                y_val = room_positions[room]
                for activity in df['activity'].unique():
                    if activity in hide_activities:
                        continue
                    mask = df['activity'] == activity
                    fig.add_trace(go.Scatter(
                        x=df.loc[mask, 'timestamp'],
                        y=[y_val] * mask.sum(),
                        mode='markers',
                        marker=dict(color=activity_colors[activity], size=8, symbol='circle'),
                        name=f"{activity}",
                        legendgroup=activity,
                        showlegend=True
                    ))
        
        fig.update_layout(
            title="📖 Training Activities Timeline",
            height=150 + 60 * len(room_positions),
            hovermode='x unified',
            margin=dict(t=50, b=50, l=100),
            yaxis=dict(
                tickmode='array',
                tickvals=list(room_positions.values()),
                ticktext=list(room_positions.keys()),
                title="Room"
            ),
            xaxis_title="Time",
            legend_title="Activities"
        )
        
        return fig, filtered_training

    def generate_prediction_timeline(self, room_names, prediction_data=None, anomaly_results=None,
                                    start_time=None, end_time=None, hide_activities=None,
                                    show_low_confidence=False, show_anomalies=False):
        """
        Generate timeline visualization for prediction data with optional anomaly display.
        
        Parameters:
        - room_names: list of room names to include
        - prediction_data: dict of room->DataFrame with prediction data
        - anomaly_results: dict of room->DataFrame with anomaly detection results
        - start_time: datetime for filtering start
        - end_time: datetime for filtering end
        - hide_activities: set of activities to hide from display
        - show_low_confidence: bool, whether to show low-confidence predictions
        - show_anomalies: bool, whether to show detected anomalies
        
        Returns:
        - fig: Plotly figure object
        - filtered_prediction: dict of filtered prediction data
        """
        if hide_activities is None:
            hide_activities = set()
        else:
            hide_activities = set(hide_activities)
            
        all_timestamps = []
        filtered_prediction = {}
        
        # Validate and convert start_time and end_time to datetime if they're not None
        if start_time is not None and not isinstance(start_time, (datetime, pd.Timestamp)):
            try:
                start_time = pd.to_datetime(start_time)
                logger.info(f"Converted start_time to datetime: {start_time}")
            except Exception as e:
                logger.warning(f"Failed to convert start_time to datetime: {e}")
                start_time = None
        
        if end_time is not None and not isinstance(end_time, (datetime, pd.Timestamp)):
            try:
                end_time = pd.to_datetime(end_time)
                logger.info(f"Converted end_time to datetime: {end_time}")
            except Exception as e:
                logger.warning(f"Failed to convert end_time to datetime: {e}")
                end_time = None
        
        # Filter prediction data
        if prediction_data:
            for room, df in prediction_data.items():
                df = df.copy()
                
                # Ensure timestamp column is datetime
                if 'timestamp' in df.columns:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        logger.info(f"Converted timestamp column to datetime for room {room}")
                    except Exception as e:
                        logger.error(f"Failed to convert timestamp column to datetime for room {room}: {e}")
                        continue
                
                if start_time is not None or end_time is not None:
                    mask = pd.Series([True] * len(df), index=df.index)
                    if start_time is not None:
                        mask &= df['timestamp'] >= start_time
                    if end_time is not None:
                        mask &= df['timestamp'] <= end_time
                    df = df[mask]
                
                # Ensure predicted_activity column contains strings for filtering
                if 'predicted_activity' in df.columns:
                    logger.info(f"DEBUG room {room}: predicted_activities before filtering: {df['predicted_activity'].unique()[:10]}")
                    try:
                        df['predicted_activity'] = df['predicted_activity'].astype(str)
                        # Hide low_confidence if not shown
                        if not show_low_confidence:
                            before = len(df)
                            df = df[df['predicted_activity'] != 'low_confidence']
                            after = len(df)
                            logger.info(f"DEBUG room {room}: low-confidence filtering removed {before - after} rows (show_low_confidence={show_low_confidence})")
                        before_hide = len(df)
                        df = df[~df['predicted_activity'].isin(hide_activities)]
                        after_hide = len(df)
                        logger.info(f"DEBUG room {room}: hide_activities removed {before_hide - after_hide} rows (hide_activities={hide_activities})")
                        logger.info(f"DEBUG room {room}: predicted_activities after filtering: {df['predicted_activity'].unique()[:10]}")
                    except Exception as e:
                        logger.error(f"Failed to process predicted_activity column for room {room}: {e}")
                        continue
                else:
                    logger.warning(f"DEBUG room {room}: no predicted_activity column found, columns: {list(df.columns)}")
                
                if not df.empty:
                    filtered_prediction[room] = df
                    all_timestamps.extend(df['timestamp'])
                    logger.info(f"DEBUG room {room}: added to filtered_prediction, shape {df.shape}")
                else:
                    logger.warning(f"DEBUG room {room}: dataframe empty after filtering, not added")
        
        if not all_timestamps:
            logger.warning(f"DEBUG generate_prediction_timeline: no timestamps after filtering. filtered_prediction keys: {list(filtered_prediction.keys())}")
            logger.warning(f"DEBUG original prediction_data keys: {list(prediction_data.keys()) if prediction_data else 'None'}")
            fig = go.Figure()
            fig.update_layout(title="No prediction data to display", height=300)
            return fig, {}
            
        # Get room positions
        room_positions = {room: i for i, room in enumerate(sorted(room_names))}
        
        # Get all unique activities for coloring
        all_activities = set()
        if filtered_prediction:
            for df in filtered_prediction.values():
                all_activities.update(df['predicted_activity'].unique())
        
        color_cycle = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
        ]
        activity_colors = {
            act: color_cycle[i % len(color_cycle)] 
            for i, act in enumerate(sorted(all_activities))
        }
        # Add special color for low_confidence
        activity_colors['low_confidence'] = '#808080'  # Gray
        
        fig = go.Figure()
        
        # Plot prediction data
        if filtered_prediction:
            for room, df in filtered_prediction.items():
                y_val = room_positions[room]
                for activity in df['predicted_activity'].unique():
                    if activity in hide_activities:
                        continue
                    mask = df['predicted_activity'] == activity
                    if activity == 'low_confidence':
                        # Only shown if explicitly enabled
                        fig.add_trace(go.Scatter(
                            x=df.loc[mask, 'timestamp'],
                            y=[y_val] * mask.sum(),
                            mode='markers',
                            marker=dict(color=activity_colors['low_confidence'], size=8, symbol='diamond'),
                            name="Low Confidence",
                            legendgroup="low_confidence",
                            showlegend=True
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=df.loc[mask, 'timestamp'],
                            y=[y_val] * mask.sum(),
                            mode='markers',
                            marker=dict(color=activity_colors.get(activity, '#000000'), size=6, symbol='square'),
                            name=f"{activity}",
                            legendgroup=activity,
                            showlegend=True
                        ))
        
        # Add anomalies if requested
        if show_anomalies and anomaly_results:
            for room_name, anomaly_df in anomaly_results.items():
                if room_name in room_positions and 'final_anomaly' in anomaly_df.columns:
                    # Get anomaly timestamps
                    anomaly_mask = anomaly_df['final_anomaly']
                    if anomaly_mask.any():
                        # Get corresponding prediction data for timestamps
                        if room_name in prediction_data and 'timestamp' in prediction_data[room_name].columns:
                            pred_df = prediction_data[room_name]
                            anomaly_times = pred_df['timestamp'][anomaly_mask]
                            
                            y_val = room_positions[room_name]
                            
                            # Add anomaly markers
                            fig.add_trace(go.Scatter(
                                x=anomaly_times,
                                y=[y_val] * len(anomaly_times),
                                mode='markers',
                                marker=dict(color='red', size=10, symbol='x', line=dict(width=2, color='white')),
                                name=f"Anomalies",
                                legendgroup="anomalies",
                                showlegend=True if room_name == sorted(room_names)[0] else False
                            ))
        
        fig.update_layout(
            title="🔮 Predicted Activities Timeline",
            height=150 + 60 * len(room_positions),
            hovermode='x unified',
            margin=dict(t=50, b=50, l=100),
            yaxis=dict(
                tickmode='array',
                tickvals=list(room_positions.values()),
                ticktext=list(room_positions.keys()),
                title="Room"
            ),
            xaxis_title="Time",
            legend_title="Activities & Anomalies"
        )
        
        return fig, filtered_prediction

    def advanced_anomaly_detection(self, room_name, df, method='combined', 
                                  z_threshold=3.0, temporal_threshold=2.0,
                                  window_size=10, use_mad=False):
        """
        Advanced Anomaly Detection with multiple methods.
        
        This method delegates to the anomaly detector's detect_combined method
        and adds additional metadata for the specific room.
        
        Parameters:
        -----------
        room_name : str
            Name of the room
        df : pandas.DataFrame
            DataFrame with sensor data
        method : str, default='combined'
            Detection method: 'zscore', 'temporal', 'combined'
        z_threshold : float, default=3.0
            Z-score threshold for statistical anomalies
        temporal_threshold : float, default=2.0
            Threshold for temporal anomalies
        window_size : int, default=10
            Window size for temporal analysis
        use_mad : bool, default=False
            Use Median Absolute Deviation for robust Z-score calculation
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with anomaly detection results
        """
        try:
            logger.info(f"Running {method} anomaly detection for {room_name}")
            
            # Update detector settings
            self.anomaly_detector.z_threshold = z_threshold
            
            # Use the detector's combined detection method
            results = self.anomaly_detector.detect_combined(df)
            
            # Add room-specific metadata
            results['room'] = room_name
            results['detection_method'] = method
            
            # Calculate anomaly confidence scores
            if 'final_anomaly_score' in results.columns:
                # Convert score to confidence (higher score = more anomalous = lower confidence)
                results['anomaly_confidence'] = 1.0 - results['final_anomaly_score']
                
                # Classify anomaly severity
                def classify_severity(confidence):
                    if confidence < 0.3:
                        return 'critical'
                    elif confidence < 0.6:
                        return 'high'
                    elif confidence < 0.8:
                        return 'medium'
                    else:
                        return 'low'
                
                results['anomaly_severity'] = results['anomaly_confidence'].apply(classify_severity)
            
            logger.info(f"Anomaly detection completed for {room_name}: {results['final_anomaly'].sum() if 'final_anomaly' in results.columns else 0} anomalies found")
            return results
            
        except Exception as e:
            logger.error(f"Anomaly detection failed for {room_name}: {e}")
            # Return empty results with basic structure
            empty_df = pd.DataFrame(index=df.index)
            empty_df['room'] = room_name
            empty_df['detection_method'] = method
            empty_df['final_anomaly'] = False
            empty_df['anomaly_confidence'] = 1.0
            empty_df['anomaly_severity'] = 'normal'
            return empty_df

    def confirm_anomaly_for_learning(self, room_name, timestamp, anomaly_type, 
                                    correct_activity=None, notes='', 
                                    add_to_training=True):
        """
        Confirm an anomaly for further learning and model improvement.
        
        Parameters:
        -----------
        room_name : str
            Name of the room where anomaly occurred
        timestamp : datetime or str
            Timestamp of the anomaly
        anomaly_type : str
            Type of anomaly detected
        correct_activity : str, optional
            Correct activity label if known
        notes : str, optional
            Additional notes about the anomaly
        add_to_training : bool, default=True
            Whether to add this confirmed anomaly to training data
            
        Returns:
        --------
        dict
            Confirmation record with metadata
        """
        confirmation = {
            'room': room_name,
            'timestamp': pd.to_datetime(timestamp),
            'anomaly_type': anomaly_type,
            'correct_activity': correct_activity,
            'notes': notes,
            'confirmed_at': datetime.now(),
            'added_to_training': False
        }
        
        # Add to training data if requested and correct activity is provided
        if add_to_training and correct_activity and room_name in self.training_data:
            try:
                # Find the row in training data
                df = self.training_data[room_name]
                timestamp_dt = pd.to_datetime(timestamp)
                
                # Find closest timestamp
                time_diff = (df['timestamp'] - timestamp_dt).abs()
                closest_idx = time_diff.idxmin()
                
                if time_diff[closest_idx] < pd.Timedelta('1min'):  # Within 1 minute
                    # Update activity label
                    df.loc[closest_idx, 'activity'] = correct_activity
                    confirmation['added_to_training'] = True
                    confirmation['training_index'] = closest_idx
                    
                    # Mark for retraining
                    self.retrain_needed = True
                    
                    logger.info(f"Confirmed anomaly added to training data: {room_name} at {timestamp}")
                else:
                    logger.warning(f"No matching timestamp found for anomaly confirmation: {timestamp}")
                    
            except Exception as e:
                logger.error(f"Failed to add confirmed anomaly to training data: {e}")
                confirmation['error'] = str(e)
        
        return confirmation

    def batch_confirm_anomalies(self, anomaly_list, add_to_training=True):
        """
        Batch confirm multiple anomalies for learning.
        
        Parameters:
        -----------
        anomaly_list : list of dict
            List of anomaly dictionaries, each containing:
            - room: room name
            - timestamp: anomaly timestamp
            - anomaly_type: type of anomaly
            - correct_activity: correct activity label (optional)
            - notes: additional notes (optional)
        add_to_training : bool, default=True
            Whether to add confirmed anomalies to training data
            
        Returns:
        --------
        list of dict
            List of confirmation records
        """
        confirmations = []
        
        for anomaly in anomaly_list:
            confirmation = self.confirm_anomaly_for_learning(
                room_name=anomaly.get('room'),
                timestamp=anomaly.get('timestamp'),
                anomaly_type=anomaly.get('anomaly_type'),
                correct_activity=anomaly.get('correct_activity'),
                notes=anomaly.get('notes', ''),
                add_to_training=add_to_training
            )
            confirmations.append(confirmation)
        
        # Log summary
        added_count = sum(1 for c in confirmations if c.get('added_to_training', False))
        logger.info(f"Batch confirmed {len(confirmations)} anomalies, {added_count} added to training data")
        
        return confirmations
