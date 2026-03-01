"""
Unified Data Loader for Beta_5
Handles loading from both Excel (.xlsx) and Parquet (.parquet) formats.
Includes data cleaning and resampling for 1000 POC scalability.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union
import logging

from elderlycare_v1_16.preprocessing.resampling import resample_to_fixed_interval

logger = logging.getLogger(__name__)


def clean_and_resample(
    data: Dict[str, pd.DataFrame], 
    interval_seconds: int = 10,
    max_ffill_gap_seconds: float | None = 60.0
) -> Dict[str, pd.DataFrame]:
    """
    Clean and resample sensor data to fixed intervals.
    
    This function:
    1. Floors all timestamps to the specified interval (default 10s)
    2. Aggregates duplicate timestamps by:
       - Taking mean of numeric columns (sensors)
       - Taking first of non-numeric columns (labels, activity)
    
    This is critical for 1000 POC scale to ensure:
    - Consistent data density across all residents
    - No duplicate timestamp errors during training
    - Predictable query performance
    
    Args:
        data: Dictionary mapping room names to DataFrames
        interval_seconds: Interval to resample to (default 10)
        max_ffill_gap_seconds: Max seconds to forward fill. Defaults to 60s.
            Pass None to intentionally allow unbounded forward fill.
        
    Returns:
        Dictionary with cleaned DataFrames
    """
    cleaned = {}
    
    max_gap_seconds = None if max_ffill_gap_seconds is None else float(max_ffill_gap_seconds)
    interval = f"{int(interval_seconds)}s"
    for room_name, df in data.items():
        if df.empty:
            cleaned[room_name] = df
            continue
            
        df = df.copy()
        
        # Ensure timestamp column exists and is datetime
        # Ensure timestamp column exists and is datetime
        if 'timestamp' not in df.columns:
            # P1 Hardening: Fail closed on schema violation.
            raise ValueError(f"Missing required 'timestamp' column in data for {room_name}")
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Use shared resampling implementation to align semantics with platform path.
        raw_rows = len(df)
        try:
            resampled = resample_to_fixed_interval(
                df,
                interval=interval,
                timestamp_col='timestamp',
                fill_method='ffill',
                keep_original_timestamps=False,
                max_ffill_gap_seconds=max_gap_seconds,
            )
            # Attach diagnostics for audit (AC-2/4.2)
            resampled.attrs["raw_rows_before_resample"] = raw_rows
            resampled.attrs["rows_after_resample"] = len(resampled)
            # We don't do "gap drop" here (that's platform logic), so just record resampled count.
            
            cleaned[room_name] = resampled
        except Exception as e:
            raise ValueError(
                f"Resampling failed for {room_name} in loader path: {e}"
            ) from e
        
    return cleaned


def load_sensor_data(
    file_path: Union[str, Path], 
    resample: bool = False,
    max_ffill_gap_seconds: float | None = 60.0
) -> Dict[str, pd.DataFrame]:
    """
    Load sensor data from either Excel or Parquet format.
    
    Args:
        file_path: Path to data file (.xlsx or .parquet)
        resample: If True, clean and resample to 10s intervals (recommended for training)
        max_ffill_gap_seconds: Max seconds to forward fill. Defaults to 60s.
            Pass None to intentionally allow unbounded forward fill.
        
    Returns:
        Dictionary mapping room/sheet names to DataFrames
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == '.parquet':
        # Parquet files store all rooms in one file with 'room' column
        df = pd.read_parquet(file_path)
        if 'room' in df.columns:
            # Split by room into dict
            data = {room: group.drop(columns=['room']).reset_index(drop=True) 
                    for room, group in df.groupby('room')}
        else:
            # Single room or no room column
            data = {'data': df}
            
    elif suffix in ['.xlsx', '.xls']:
        # Excel files have sheets per room
        data = pd.read_excel(file_path, sheet_name=None)
        
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    # Apply cleaning and resampling if requested
    if resample:
        data = clean_and_resample(data, max_ffill_gap_seconds=max_ffill_gap_seconds)
        
    return data


def save_to_parquet(data: Dict[str, pd.DataFrame], output_path: Union[str, Path]) -> Path:
    """
    Save sensor data to Parquet format.
    
    Args:
        data: Dictionary mapping room names to DataFrames
        output_path: Path for output .parquet file
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    
    # Combine all rooms into single DataFrame with 'room' column
    dfs = []
    for room_name, df in data.items():
        df_copy = df.copy()
        df_copy['room'] = room_name
        dfs.append(df_copy)
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_parquet(output_path, index=False, compression='snappy')
        logger.info(f"Saved {len(combined)} rows to {output_path}")
    else:
        # Create empty parquet
        pd.DataFrame().to_parquet(output_path, index=False)
        logger.warning(f"Saved empty parquet to {output_path}")
    
    return output_path


def convert_xlsx_to_parquet(xlsx_path: Union[str, Path], delete_original: bool = False) -> Path:
    """
    Convert an Excel file to Parquet format.
    
    Args:
        xlsx_path: Path to input Excel file
        delete_original: Whether to delete the original xlsx after conversion
        
    Returns:
        Path to created Parquet file
    """
    xlsx_path = Path(xlsx_path)
    parquet_path = xlsx_path.with_suffix('.parquet')
    
    # Load Excel
    data = load_sensor_data(xlsx_path)
    
    # Save as Parquet
    save_to_parquet(data, parquet_path)
    
    # Optionally delete original
    if delete_original:
        xlsx_path.unlink()
        logger.info(f"Deleted original: {xlsx_path}")
    
    return parquet_path


def get_archive_files(archive_dir: Path, file_type: str = None, resident_id: str = None) -> list:
    """
    Get archived files from the archive directory.
    
    Args:
        archive_dir: Root archive directory
        file_type: 'input', 'train', or None for all
        resident_id: Filter by resident ID (optional)
        
    Returns:
        List of dicts with date, filename, path info
    """
    files = []
    
    if not archive_dir.exists():
        return files
    
    # Support both xlsx and parquet
    patterns = ['*.xlsx', '*.parquet']
    
    for date_dir in archive_dir.iterdir():
        if not date_dir.is_dir():
            continue
            
        for pattern in patterns:
            for f in date_dir.glob(pattern):
                # Filter by file type
                fname_lower = f.name.lower()
                
                if file_type == 'input':
                    if '_train' in fname_lower:
                        continue
                elif file_type == 'train':
                    if '_input' in fname_lower or '_train' not in fname_lower:
                        continue
                
                # Filter by resident
                if resident_id and resident_id.lower() not in fname_lower:
                    continue
                
                files.append({
                    "date": date_dir.name,
                    "filename": f.name,
                    "path": str(f)
                })
    
    # Sort by date desc
    files.sort(key=lambda x: x['date'], reverse=True)
    return files
