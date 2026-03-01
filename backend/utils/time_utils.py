"""
Unified timestamp handling with conflict resolution and interval support.
"""
import pandas as pd
from datetime import datetime
from typing import Union

def normalize_with_interval(
    value: Union[str, int, float, datetime, pd.Timestamp],
    interval_seconds: int = 10,
    output_format: str = '%Y-%m-%d %H:%M:%S'
) -> str:
    """
    Unified timestamp normalization with interval support.
    
    Robustly handles various input types (int/float Unix timestamps, strings, objects)
    and enforces a consistent unit interpretation (assuming seconds for scalars).
    """
    # Convert all types to pandas Timestamp
    if isinstance(value, (int, float)):
        # Handle Unix timestamps (assume seconds as per team standard)
        try:
            ts = pd.Timestamp(value, unit='s')
        except Exception:
            # Fallback for floats or out-of-bounds
            ts = pd.Timestamp(datetime.fromtimestamp(value))
    elif isinstance(value, str):
        ts = pd.to_datetime(value)
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, pd.Timestamp):
        ts = value
    else:
        raise TypeError(f"Unsupported timestamp type: {type(value)}")
    
    # Apply interval flooring
    if interval_seconds > 0:
        ts = ts.floor(f'{interval_seconds}s')
    
    return ts.strftime(output_format)


# ============================================================
# Timezone Standardization Utilities (Added for Beta 5.5)
# ============================================================

def ensure_naive(ts):
    """
    Convert UTC-aware timestamps to naive (strip timezone info).
    
    Works with pd.Series, pd.Timestamp, or pd.DatetimeIndex.
    If already naive, returns unchanged.
    
    Args:
        ts: Timestamp-like object (Series, Timestamp, DatetimeIndex)
        
    Returns:
        Same type with timezone stripped
    """
    if ts is None:
        return None
    
    if isinstance(ts, pd.Series):
        if ts.dt.tz is not None:
            return ts.dt.tz_convert(None)
        return ts
    elif isinstance(ts, pd.DatetimeIndex):
        if ts.tz is not None:
            return ts.tz_convert(None)
        return ts
    elif isinstance(ts, pd.Timestamp):
        if ts.tz is not None:
            return ts.tz_convert(None)
        return ts
    elif isinstance(ts, datetime):
        # Python datetime: just remove tzinfo
        return ts.replace(tzinfo=None)
    else:
        # Try to convert and strip
        try:
            converted = pd.to_datetime(ts)
            if hasattr(converted, 'tz') and converted.tz is not None:
                return converted.tz_convert(None)
            return converted
        except Exception:
            return ts


def ensure_utc(ts):
    """
    Convert naive timestamps to UTC-aware.
    
    Works with pd.Series, pd.Timestamp, or pd.DatetimeIndex.
    If already aware, converts to UTC.
    
    Args:
        ts: Timestamp-like object
        
    Returns:
        Same type with UTC timezone
    """
    if ts is None:
        return None
    
    if isinstance(ts, pd.Series):
        if ts.dt.tz is None:
            return ts.dt.tz_localize('UTC')
        return ts.dt.tz_convert('UTC')
    elif isinstance(ts, pd.DatetimeIndex):
        if ts.tz is None:
            return ts.tz_localize('UTC')
        return ts.tz_convert('UTC')
    elif isinstance(ts, pd.Timestamp):
        if ts.tz is None:
            return ts.tz_localize('UTC')
        return ts.tz_convert('UTC')
    else:
        try:
            converted = pd.to_datetime(ts)
            if hasattr(converted, 'tz') and converted.tz is None:
                return converted.tz_localize('UTC')
            return converted.tz_convert('UTC')
        except Exception:
            return ts


def safe_merge_timestamps(left_df, right_df, on='timestamp', tolerance='30s', direction='nearest'):
    """
    Wrapper for pd.merge_asof that normalizes timezones first.
    
    Ensures both DataFrames have naive timestamps before merging,
    preventing the common MergeError caused by mixed aware/naive types.
    
    Args:
        left_df: Left DataFrame
        right_df: Right DataFrame
        on: Column to merge on (default: 'timestamp')
        tolerance: Merge tolerance (default: '30s')
        direction: Merge direction (default: 'nearest')
        
    Returns:
        Merged DataFrame
    """
    left = left_df.copy()
    right = right_df.copy()
    
    # Normalize timestamps to naive
    if on in left.columns:
        left[on] = pd.to_datetime(left[on])
        left[on] = ensure_naive(left[on])
    
    if on in right.columns:
        right[on] = pd.to_datetime(right[on])
        right[on] = ensure_naive(right[on])
    
    # Sort (required for merge_asof)
    left = left.sort_values(on).reset_index(drop=True)
    right = right.sort_values(on).reset_index(drop=True)
    
    return pd.merge_asof(
        left,
        right,
        on=on,
        tolerance=pd.Timedelta(tolerance),
        direction=direction
    )

