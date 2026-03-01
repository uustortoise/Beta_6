"""
Time-based resampling utilities for sensor data.

This module provides functions to resample time series data to a fixed interval,
handling missing data and ensuring consistent temporal patterns for model training
and prediction.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def resample_to_fixed_interval(
    df: pd.DataFrame,
    interval: str = '10s',
    method: str = 'linear',
    timestamp_col: str = 'timestamp',
    fill_method: str = 'ffill',
    keep_original_timestamps: bool = False,
    max_ffill_gap_seconds: Optional[float] = 60.0,
    normalize_timestamps_to_interval: bool = True,
) -> pd.DataFrame:
    """
    Resample DataFrame to fixed time intervals.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with a timestamp column or timestamp index.
    interval : str, default='10S'
        Resampling interval. Uses pandas offset aliases (e.g., '10S' for 10 seconds,
        '1T' for 1 minute, '1H' for 1 hour).
    method : str, default='linear'
        Interpolation method for resampling. Options:
        - 'linear': Linear interpolation
        - 'nearest': Nearest neighbor interpolation
        - 'pad' or 'ffill': Forward fill
        - 'bfill': Backward fill
    timestamp_col : str, default='timestamp'
        Name of the timestamp column. If the DataFrame has a DatetimeIndex,
        set timestamp_col to None.
    fill_method : str, default='ffill'
        Method to fill missing values after resampling. Options:
        - 'ffill': Forward fill
        - 'bfill': Backward fill
        - 'interpolate': Use the specified interpolation method
    keep_original_timestamps : bool, default=False
        If True, keep the original timestamps as a column named 'original_timestamp'.
    max_ffill_gap_seconds : float | None, default=60.0
        Maximum duration (in seconds) that forward-fill is allowed to bridge.
        Set to None or <= 0 for unbounded forward-fill.
    normalize_timestamps_to_interval : bool, default=True
        Normalize timestamps onto the target interval grid (nearest bucket)
        before reindexing. This hardens real-world jittered streams (e.g., 6-15s
        around a nominal 10s cadence) and avoids artificial sparsity.
    
    Returns:
    --------
    pd.DataFrame
        Resampled DataFrame with consistent intervals.
    
    Raises:
    -------
    ValueError
        If the timestamp column is not found or cannot be converted to datetime.
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Check if timestamp column exists or if we should use the index
    if timestamp_col is not None:
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            except Exception as e:
                raise ValueError(f"Failed to convert column '{timestamp_col}' to datetime: {e}")
        
        # Set timestamp as index for resampling
        df = df.set_index(timestamp_col)
        original_timestamp_col = timestamp_col
    else:
        # Assume index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Failed to convert index to datetime: {e}")
        original_timestamp_col = None
    
    # Sort by timestamp
    df = df.sort_index()

    # Normalize jittered real-world timestamps to the target interval grid first.
    # Without this step, a 10s model grid can appear sparsely populated when
    # source data drifts between ~6-15s.
    if normalize_timestamps_to_interval and len(df) > 0:
        try:
            original_index = df.index
            normalized_index = original_index.round(interval)
            # Log normalization drift summary for observability.
            drift_seconds = np.abs((normalized_index.asi8 - original_index.asi8) / 1_000_000_000.0)
            moved_ratio = float((drift_seconds > 0).mean())
            if moved_ratio > 0:
                logger.info(
                    "Timestamp normalization to %s adjusted %.1f%% rows (p50=%.2fs, p90=%.2fs, p99=%.2fs).",
                    interval,
                    moved_ratio * 100.0,
                    float(np.quantile(drift_seconds, 0.50)),
                    float(np.quantile(drift_seconds, 0.90)),
                    float(np.quantile(drift_seconds, 0.99)),
                )
            df.index = normalized_index
            df = df.sort_index()
        except Exception as e:
            logger.warning(
                "Timestamp normalization failed for interval=%s: %s. Proceeding without normalization.",
                interval,
                e,
            )
    
    # Save original timestamps if requested
    if keep_original_timestamps and original_timestamp_col is not None:
        # We already have the original timestamps as index, will add back later
        pass
    
    # Check for duplicate timestamps
    if df.index.duplicated().any():
        logger.warning(f"Duplicate timestamps found. Aggregating duplicates safely.")
        try:
            # Separate numeric and non-numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            
            agg_dict = {col: 'mean' for col in numeric_cols}
            agg_dict.update({col: 'first' for col in non_numeric_cols})
            
            df = df.groupby(df.index).agg(agg_dict)
        except Exception as e:
             logger.error(f"Duplicate aggregation failed: {e}. Keeping first occurrence.")
             df = df[~df.index.duplicated(keep='first')]
    
    # Determine the time range.
    # If we normalized, min/max are already bucket-aligned.
    start_time = df.index.min()
    end_time = df.index.max()
    
    # Create a regular time index
    regular_index = pd.date_range(start=start_time, end=end_time, freq=interval)
    
    # Resample the data
    # First, reindex to the regular index (this creates NaN for missing intervals)
    df_resampled = df.reindex(regular_index)
    
    # Resolve bounded forward-fill limit from interval and max gap policy.
    ffill_limit = None
    if max_ffill_gap_seconds is not None:
        try:
            max_gap = float(max_ffill_gap_seconds)
            if max_gap > 0:
                interval_seconds = float(pd.to_timedelta(interval).total_seconds())
                if interval_seconds > 0:
                    ffill_limit = max(1, int(max_gap // interval_seconds))
        except Exception:
            logger.warning(
                "Invalid max_ffill_gap_seconds=%s for interval=%s; using unbounded ffill.",
                max_ffill_gap_seconds,
                interval,
            )
            ffill_limit = None

    # Handle missing values based on the fill method
    if fill_method == 'ffill':
        df_resampled = df_resampled.ffill(limit=ffill_limit)
    elif fill_method == 'bfill':
        logger.warning("fill_method='bfill' is non-causal; using forward-fill instead")
        df_resampled = df_resampled.ffill(limit=ffill_limit)
    elif fill_method == 'interpolate':
        # Interpolation requires future values (non-causal). Keep resampling causal by default.
        logger.warning("fill_method='interpolate' is non-causal; using forward-fill instead")
        df_resampled = df_resampled.ffill(limit=ffill_limit)
    else:
        raise ValueError(f"Unsupported fill_method: {fill_method}")

    # Only backfill leading NaNs (before first observation), never interior gaps.
    for col in df_resampled.columns:
        if not df_resampled[col].isna().any():
            continue
        first_valid = df_resampled[col].first_valid_index()
        if first_valid is None:
            # Entire column missing across the range; keep as-is for now.
            continue
        df_resampled.loc[:first_valid, col] = df_resampled.loc[:first_valid, col].bfill()
    
    # Reset index to make timestamp a column again
    df_resampled = df_resampled.reset_index()
    df_resampled = df_resampled.rename(columns={'index': 'timestamp'})
    
    # Add original timestamps if requested
    if keep_original_timestamps:
        # We need to map original timestamps to the resampled ones.
        # For each resampled timestamp, find the closest original timestamp.
        # This is approximate but gives an idea of the original sampling.
        original_ts = df.index.values
        resampled_ts = df_resampled['timestamp'].values
        
        # Find the closest original timestamp for each resampled timestamp
        closest_original = []
        for ts in resampled_ts:
            # Find the index of the closest original timestamp
            idx = np.argmin(np.abs(original_ts - ts))
            closest_original.append(pd.Timestamp(original_ts[idx]))
        
        df_resampled['original_timestamp'] = closest_original
    
    # Log some statistics
    original_interval = None
    if len(df) > 1:
        time_diffs = df.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            original_interval = time_diffs.mean()
            logger.info(f"Original data: {len(df)} points, average interval: {original_interval}")
    
    logger.info(f"Resampled data: {len(df_resampled)} points, fixed interval: {interval}")
    logger.info(f"Time range: {start_time} to {end_time}")
    
    return df_resampled


def analyze_time_intervals(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp'
) -> Dict[str, Any]:
    """
    Analyze the time intervals in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with a timestamp column.
    timestamp_col : str, default='timestamp'
        Name of the timestamp column.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary with interval statistics.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found.")
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Sort by timestamp
    df_sorted = df.sort_values(timestamp_col)
    
    # Calculate time differences
    timestamps = df_sorted[timestamp_col].values
    if len(timestamps) < 2:
        return {
            'count': len(timestamps),
            'min_interval': None,
            'max_interval': None,
            'mean_interval': None,
            'std_interval': None,
            'interval_consistency': 'insufficient_data'
        }
    
    diffs = np.diff(timestamps) / np.timedelta64(1, 's')  # in seconds
    
    # Calculate statistics
    stats = {
        'count': len(timestamps),
        'min_interval': np.min(diffs),
        'max_interval': np.max(diffs),
        'mean_interval': np.mean(diffs),
        'median_interval': np.median(diffs),
        'std_interval': np.std(diffs),
        'interval_consistency': 'consistent' if np.std(diffs) < 1.0 else 'variable'
    }
    
    # Add human-readable strings
    stats['min_interval_str'] = f"{stats['min_interval']:.2f}s"
    stats['max_interval_str'] = f"{stats['max_interval']:.2f}s"
    stats['mean_interval_str'] = f"{stats['mean_interval']:.2f}s"
    
    return stats


def detect_irregular_intervals(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    Detect irregular time intervals in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with a timestamp column.
    timestamp_col : str, default='timestamp'
        Name of the timestamp column.
    threshold : float, default=2.0
        Threshold in seconds for considering an interval irregular.
        Intervals more than `threshold` seconds from the median are flagged.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with irregular intervals flagged.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found.")
    
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    
    # Calculate time differences in seconds
    diffs = df[timestamp_col].diff().dt.total_seconds()
    
    # Calculate median interval (skip first NaN)
    median_interval = diffs.iloc[1:].median()
    
    # Flag irregular intervals
    # Consider intervals that are more than `threshold` seconds away from median
    df['interval_seconds'] = diffs
    df['irregular_interval'] = False
    
    if not pd.isna(median_interval):
        mask = (diffs < median_interval - threshold) | (diffs > median_interval + threshold)
        df.loc[mask, 'irregular_interval'] = True
    
    # Also flag missing intervals (gaps)
    df['missing_interval'] = df['interval_seconds'] > (median_interval + threshold) if not pd.isna(median_interval) else False
    
    return df


def resample_multiple_rooms(
    room_data: Dict[str, pd.DataFrame],
    interval: str = '10s',
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Resample data for multiple rooms.
    
    Parameters:
    -----------
    room_data : Dict[str, pd.DataFrame]
        Dictionary mapping room names to DataFrames.
    interval : str, default='10S'
        Resampling interval.
    **kwargs : dict
        Additional arguments passed to resample_to_fixed_interval.
    
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with resampled DataFrames.
    """
    resampled_data = {}
    
    for room_name, df in room_data.items():
        try:
            resampled_df = resample_to_fixed_interval(df, interval=interval, **kwargs)
            resampled_data[room_name] = resampled_df
            logger.info(f"Resampled data for room '{room_name}' to {interval} interval")
        except Exception as e:
            logger.error(f"Failed to resample data for room '{room_name}': {e}")
            # Keep original data if resampling fails
            resampled_data[room_name] = df
    
    return resampled_data


def create_time_based_sequences(
    data: np.ndarray,
    timestamps: np.ndarray,
    time_window: str = '5min',
    interval: str = '10s',
    padding: str = 'pre'
) -> np.ndarray:
    """
    Create sequences based on fixed time windows.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    timestamps : np.ndarray
        Array of timestamps corresponding to each sample.
    time_window : str, default='5min'
        Time window for each sequence (pandas offset alias).
    interval : str, default='10S'
        Data interval (pandas offset alias).
    padding : str, default='pre'
        Padding strategy for short sequences: 'pre', 'post', or 'none'.
    
    Returns:
    --------
    np.ndarray
        Array of sequences with shape (n_sequences, window_samples, n_features).
    """
    # Convert time window to number of samples
    window_duration = pd.Timedelta(time_window)
    sample_interval = pd.Timedelta(interval)
    
    if sample_interval.total_seconds() <= 0:
        raise ValueError(f"Invalid interval: {interval}")
    
    window_samples = int(window_duration.total_seconds() / sample_interval.total_seconds())
    
    if window_samples <= 0:
        raise ValueError(f"Time window '{time_window}' is too short for interval '{interval}'")
    
    # Create sample-based sequences
    from .sequences import create_sequences
    return create_sequences(data, seq_length=window_samples, stride=1, padding=padding)


def calculate_samples_from_time(
    time_window: str,
    interval: str = '10s'
) -> int:
    """
    Calculate number of samples from time window and interval.
    
    Parameters:
    -----------
    time_window : str
        Time window (pandas offset alias).
    interval : str, default='10S'
        Data interval (pandas offset alias).
    
    Returns:
    --------
    int
        Number of samples in the time window.
    """
    window_duration = pd.Timedelta(time_window)
    sample_interval = pd.Timedelta(interval)
    
    if sample_interval.total_seconds() <= 0:
        raise ValueError(f"Invalid interval: {interval}")
    
    samples = int(window_duration.total_seconds() / sample_interval.total_seconds())
    
    if samples <= 0:
        raise ValueError(f"Time window '{time_window}' is too short for interval '{interval}'")
    
    return samples


def validate_interval_consistency(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    warning_threshold: float = 2.0
) -> Dict[str, Any]:
    """
    Validate interval consistency and provide warnings.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with timestamp column.
    timestamp_col : str, default='timestamp'
        Name of timestamp column.
    warning_threshold : float, default=2.0
        Threshold in seconds for warning about interval variability.
    
    Returns:
    --------
    Dict[str, Any]
        Validation results with warnings and recommendations.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found.")
    
    # Analyze intervals
    stats = analyze_time_intervals(df, timestamp_col)
    
    result = {
        'is_consistent': stats['interval_consistency'] == 'consistent',
        'stats': stats,
        'warnings': [],
        'recommendations': []
    }
    
    # Check for warnings
    if stats['std_interval'] is not None and stats['std_interval'] > warning_threshold:
        result['warnings'].append(
            f"High interval variability: std={stats['std_interval']:.2f}s. "
            f"Consider resampling to a fixed interval."
        )
        result['recommendations'].append(
            f"Use resample_to_fixed_interval() with interval='{max(1, int(stats['mean_interval']))}S'"
        )
    
    if stats['min_interval'] is not None and stats['min_interval'] < 0.1:
        result['warnings'].append(
            f"Very short intervals detected: min={stats['min_interval']:.3f}s. "
            f"May indicate duplicate timestamps."
        )
    
    if stats['max_interval'] is not None and stats['max_interval'] > 60:
        result['warnings'].append(
            f"Large gaps detected: max={stats['max_interval']:.1f}s. "
            f"Consider checking for missing data."
        )
    
    # Add overall recommendation
    if not result['is_consistent']:
        result['recommendations'].append(
            "Enable time-based resampling for consistent temporal patterns."
        )
    
    return result


# =============================================================================
# Beta 5.5: Transformer-specific Preprocessing Utilities
# =============================================================================

def detect_gaps(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    gap_threshold_seconds: float = 300.0  # 5 minutes
) -> pd.DataFrame:
    """
    Detect gaps in time series data for Transformer preprocessing.
    
    Gaps larger than the threshold are marked with a special flag to help
    the Transformer learn segment boundaries (instead of treating forward-filled
    data as continuous).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with timestamp column.
    timestamp_col : str, default='timestamp'
        Name of timestamp column.
    gap_threshold_seconds : float, default=300.0
        Threshold in seconds for marking a gap. Default is 5 minutes.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional columns:
        - 'gap_before': True if there's a significant gap before this row
        - 'gap_duration_seconds': Duration of the gap in seconds (0 if no gap)
        - 'segment_id': Integer ID for contiguous segments (increments after each gap)
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Sort by timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    # Calculate time differences
    df['_time_diff'] = df[timestamp_col].diff().dt.total_seconds()
    
    # Mark gaps
    df['gap_before'] = df['_time_diff'] > gap_threshold_seconds
    df['gap_duration_seconds'] = df['_time_diff'].where(df['gap_before'], 0)
    
    # First row has no gap before it
    df.loc[0, 'gap_before'] = False
    df.loc[0, 'gap_duration_seconds'] = 0
    
    # Assign segment IDs (increments after each gap)
    df['segment_id'] = df['gap_before'].cumsum()
    
    # Clean up temporary column
    df = df.drop(columns=['_time_diff'])
    
    # Log statistics
    num_gaps = df['gap_before'].sum()
    num_segments = df['segment_id'].max() + 1
    if num_gaps > 0:
        avg_gap = df.loc[df['gap_before'], 'gap_duration_seconds'].mean()
        logger.info(f"Detected {num_gaps} gaps (>{gap_threshold_seconds}s), creating {num_segments} segments. Avg gap: {avg_gap:.1f}s")
    
    return df


def insert_gap_tokens(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    gap_threshold_seconds: float = 300.0,
    gap_token_value: float = -1.0
) -> pd.DataFrame:
    """
    Insert GAP token rows into the DataFrame where significant gaps exist.
    
    This prepares data for Transformer models by explicitly marking discontinuities.
    The GAP token rows have special sensor values (default -1.0) that the model
    can learn to recognize as segment boundaries.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with timestamp and sensor columns.
    timestamp_col : str, default='timestamp'
        Name of timestamp column.
    gap_threshold_seconds : float, default=300.0
        Threshold in seconds for inserting a GAP token.
    gap_token_value : float, default=-1.0
        Value to use for sensor columns in GAP token rows.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with GAP token rows inserted.
    """
    # First detect gaps
    df = detect_gaps(df, timestamp_col, gap_threshold_seconds)
    
    if not df['gap_before'].any():
        logger.info("No significant gaps detected, no GAP tokens inserted.")
        return df
    
    # Get indices where gaps occur
    gap_indices = df[df['gap_before']].index.tolist()
    
    # Identify sensor columns (numeric columns that aren't gap-related)
    gap_related_cols = ['gap_before', 'gap_duration_seconds', 'segment_id']
    sensor_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                   if c not in gap_related_cols]
    
    # Create GAP token rows
    gap_rows = []
    for idx in gap_indices:
        # Create a row with GAP token values
        gap_row = df.iloc[idx].copy()
        
        # Set sensor values to gap token value
        for col in sensor_cols:
            gap_row[col] = gap_token_value
        
        # Mark as GAP token
        gap_row['is_gap_token'] = True
        gap_row['gap_before'] = False  # The GAP token itself doesn't have a gap before it
        
        # Adjust timestamp to be midpoint of the gap
        prev_ts = df.iloc[idx - 1][timestamp_col] if idx > 0 else gap_row[timestamp_col]
        curr_ts = gap_row[timestamp_col]
        gap_row[timestamp_col] = prev_ts + (curr_ts - prev_ts) / 2
        
        gap_rows.append(gap_row)
    
    # Add is_gap_token column to original df
    df['is_gap_token'] = False
    
    # Insert gap rows
    if gap_rows:
        gap_df = pd.DataFrame(gap_rows)
        df = pd.concat([df, gap_df], ignore_index=True)
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        logger.info(f"Inserted {len(gap_rows)} GAP token rows for Transformer preprocessing.")
    
    return df
