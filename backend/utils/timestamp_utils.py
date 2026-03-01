"""
Timestamp normalization utilities for consistent datetime handling.

Addresses external review technical debt: "Mixed timestamp formats in DB"
"""

import logging
from datetime import datetime
from typing import Union, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Standard format for database storage
DB_TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'
DB_DATE_FORMAT = '%Y-%m-%d'

# Common input formats to handle
KNOWN_FORMATS = [
    '%Y-%m-%d %H:%M:%S',      # Standard: 2026-01-20 08:15:00
    '%Y-%m-%dT%H:%M:%S',      # ISO: 2026-01-20T08:15:00
    '%Y-%m-%dT%H:%M:%S.%f',   # ISO with microseconds
    '%Y-%m-%d %H:%M:%S.%f',   # Standard with microseconds
    '%d/%m/%Y %H:%M:%S',      # European: 20/01/2026 08:15:00
    '%m/%d/%Y %H:%M:%S',      # US: 01/20/2026 08:15:00
    '%Y/%m/%d %H:%M:%S',      # Asian: 2026/01/20 08:15:00
]


def normalize_timestamp(value: Union[str, datetime, pd.Timestamp], 
                         output_format: str = DB_TIMESTAMP_FORMAT) -> Optional[str]:
    """
    Normalize a timestamp value to a consistent string format.
    
    Args:
        value: Input timestamp (string, datetime, or pandas Timestamp)
        output_format: Desired output format (default: DB_TIMESTAMP_FORMAT)
        
    Returns:
        Normalized timestamp string, or None if parsing fails
    """
    if value is None:
        return None
    
    # Already a datetime-like object
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.strftime(output_format)
    
    # String - try to parse
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
            
        # Try pandas first (handles most formats)
        try:
            parsed = pd.to_datetime(value)
            return parsed.strftime(output_format)
        except:
            pass
        
        # Fallback to known formats
        for fmt in KNOWN_FORMATS:
            try:
                parsed = datetime.strptime(value, fmt)
                return parsed.strftime(output_format)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse timestamp: {value}")
        return None
    
    logger.warning(f"Unexpected timestamp type: {type(value)}")
    return None


def normalize_date(value: Union[str, datetime, pd.Timestamp]) -> Optional[str]:
    """
    Normalize a date value to YYYY-MM-DD format.
    
    Args:
        value: Input date/datetime value
        
    Returns:
        Normalized date string (YYYY-MM-DD), or None if parsing fails
    """
    result = normalize_timestamp(value, output_format=DB_DATE_FORMAT)
    if result and len(result) > 10:
        return result[:10]  # Truncate time portion if present
    return result


def normalize_df_timestamps(df: pd.DataFrame, 
                            columns: list = None,
                            inplace: bool = False) -> pd.DataFrame:
    """
    Normalize timestamp columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        columns: List of column names to normalize (default: auto-detect 'timestamp' columns)
        inplace: If True, modify DataFrame in place
        
    Returns:
        DataFrame with normalized timestamps
    """
    if not inplace:
        df = df.copy()
    
    # Auto-detect timestamp columns if not specified
    if columns is None:
        columns = [c for c in df.columns if 'timestamp' in c.lower() or 'time' in c.lower()]
    
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].dt.strftime(DB_TIMESTAMP_FORMAT)
    
    return df
