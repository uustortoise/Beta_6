"""
MotionDataNormalizer - Centralized motion data handling for sleep analysis.

This utility ensures consistent motion data processing between:
- Real-time pipeline (process_data.py)
- Historical pipeline (backfill_analysis.py)

Created as per external review recommendation to address data consistency risks.
"""

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class MotionDataStatus(Enum):
    """Status of motion data validation."""
    VALID_RAW = "valid_raw"           # Raw sensor values (0-1 range)
    SCALED_ZSCORE = "scaled_zscore"   # Z-score normalized (mean~0, std~1)
    MISSING = "missing"               # No motion column present
    CONSTANT = "constant"             # All same value (no variance)
    INVALID = "invalid"               # Other issues


@dataclass
class MotionDataQuality:
    """Result of motion data validation."""
    status: MotionDataStatus
    message: str
    use_heuristics: bool
    stats: Optional[dict] = None


class MotionDataNormalizer:
    """
    Centralized motion data normalization for sleep analysis consistency.
    
    Implements external review recommendations:
    1. Detect scaled (Z-score) vs raw sensor data
    2. Validate data quality before sleep analysis
    3. Ensure consistent handling across all pipelines
    """
    
    # Expected range for raw motion sensor values
    RAW_MOTION_RANGE = (0.0, 1.0)
    
    # Z-score detection thresholds
    ZSCORE_MEAN_THRESHOLD = 0.5    # If |mean| < this, might be Z-score
    ZSCORE_STD_THRESHOLD = (0.5, 2.0)  # If std in this range, might be Z-score
    
    @classmethod
    def validate(cls, df: pd.DataFrame) -> MotionDataQuality:
        """
        Validate motion data quality and determine processing strategy.
        
        Args:
            df: DataFrame that may contain 'motion' column
            
        Returns:
            MotionDataQuality with validation results
        """
        # Check if motion column exists (case-insensitive)
        motion_col = next((c for c in df.columns if c.lower() == 'motion'), None)
        if not motion_col:
            return MotionDataQuality(
                status=MotionDataStatus.MISSING,
                message="No motion column present",
                use_heuristics=True
            )
        
        motion = df[motion_col].dropna()
        
        if len(motion) == 0:
            return MotionDataQuality(
                status=MotionDataStatus.MISSING,
                message="Motion column is all NaN",
                use_heuristics=True
            )
        
        # Calculate statistics
        stats = {
            'count': len(motion),
            'mean': float(motion.mean()),
            'std': float(motion.std()) if len(motion) > 1 else 0.0,
            'min': float(motion.min()),
            'max': float(motion.max()),
            'has_negatives': bool((motion < 0).any()),
            'unique_values': int(motion.nunique())
        }
        
        # Check for constant values (no variance)
        if stats['unique_values'] <= 1:
            return MotionDataQuality(
                status=MotionDataStatus.CONSTANT,
                message=f"Motion data has constant value ({stats['mean']:.3f})",
                use_heuristics=True,
                stats=stats
            )
        
        # Detect Z-score normalized data
        if cls._is_likely_zscore(stats):
            return MotionDataQuality(
                status=MotionDataStatus.SCALED_ZSCORE,
                message=f"Motion appears Z-score normalized (mean={stats['mean']:.3f}, std={stats['std']:.3f})",
                use_heuristics=True,
                stats=stats
            )
        
        # Check if values are in expected raw range
        if stats['min'] >= cls.RAW_MOTION_RANGE[0] and stats['max'] <= cls.RAW_MOTION_RANGE[1]:
            return MotionDataQuality(
                status=MotionDataStatus.VALID_RAW,
                message="Motion data is valid raw sensor values",
                use_heuristics=False,
                stats=stats
            )
        
        # Values outside expected range
        return MotionDataQuality(
            status=MotionDataStatus.INVALID,
            message=f"Motion values outside expected range [{stats['min']:.3f}, {stats['max']:.3f}]",
            use_heuristics=True,
            stats=stats
        )
    
    @classmethod
    def _is_likely_zscore(cls, stats: dict) -> bool:
        """
        Detect if data appears to be Z-score normalized.
        
        Z-score characteristics:
        - Mean close to 0
        - Std close to 1
        - Contains negative values
        """
        if stats['has_negatives']:
            return True
            
        mean_is_low = abs(stats['mean']) < cls.ZSCORE_MEAN_THRESHOLD
        std_near_one = cls.ZSCORE_STD_THRESHOLD[0] <= stats['std'] <= cls.ZSCORE_STD_THRESHOLD[1]
        
        return mean_is_low and std_near_one
    
    @classmethod
    def normalize_for_sleep_analysis(
        cls,
        df: pd.DataFrame,
        source: str = "unknown",
        raw_data_source: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, MotionDataQuality]:
        """
        Normalize motion data for sleep analysis.
        
        This method ensures consistent handling across all pipelines:
        - If valid raw data exists, use it directly
        - If Z-score/invalid data, optionally merge from raw source or flag for heuristics
        - Log validation results for transparency
        
        Args:
            df: DataFrame with predictions
            source: String identifier for logging (e.g., "process_data", "backfill")
            raw_data_source: Optional DataFrame with raw sensor values to merge from
            
        Returns:
            Tuple of (processed DataFrame, validation quality result)
        """
        result_df = df.copy()
        quality = cls.validate(result_df)
        
        # Log validation result
        logger.info(f"[{source}] Motion data validation: {quality.status.value} - {quality.message}")
        
        # If data is already valid, return as-is
        if quality.status == MotionDataStatus.VALID_RAW:
            return result_df, quality
        
        # Try to inject raw motion from source if provided
        raw_motion_col = None
        if raw_data_source is not None and hasattr(raw_data_source, "columns"):
            raw_motion_col = next((c for c in raw_data_source.columns if c.lower() == 'motion'), None)
        if raw_data_source is not None and raw_motion_col:
            raw_quality = cls.validate(raw_data_source)
            
            if raw_quality.status == MotionDataStatus.VALID_RAW:
                result_df = cls._merge_raw_motion(result_df, raw_data_source)
                
                # Re-validate after merge
                merged_quality = cls.validate(result_df)
                if merged_quality.status == MotionDataStatus.VALID_RAW:
                    logger.info(f"[{source}] Successfully injected raw motion data")
                    return result_df, merged_quality
                    
        # Could not get valid raw data - will use heuristics
        if 'motion' in result_df.columns and quality.use_heuristics:
            # Remove invalid motion to force heuristic fallback in SleepAnalyzer
            result_df = result_df.drop(columns=['motion'])
            logger.info(f"[{source}] Removed invalid motion data - sleep staging will use heuristics")
        
        return result_df, quality
    
    @classmethod
    def _merge_raw_motion(cls, pred_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge raw motion values into predictions using timestamp matching.
        """
        # Find actual column names (case-insensitive)
        pred_ts_col = next((c for c in pred_df.columns if c.lower() == 'timestamp'), None)
        raw_ts_col = next((c for c in raw_df.columns if c.lower() == 'timestamp'), None)
        
        if not pred_ts_col or not raw_ts_col:
            return pred_df
            
        try:
            # Find actual motion column name (case-insensitive)
            raw_mot_col = next((c for c in raw_df.columns if c.lower() == 'motion'), 'motion')

            # Prepare dataframes
            pred_copy = pred_df.copy()
            raw_copy = raw_df[[raw_ts_col, raw_mot_col]].copy()
            
            # Standardize column names for merge
            raw_copy = raw_copy.rename(columns={raw_ts_col: 'timestamp', raw_mot_col: 'motion'})
            pred_copy = pred_copy.rename(columns={pred_ts_col: 'timestamp'})
            
            pred_copy['timestamp'] = pd.to_datetime(pred_copy['timestamp'])
            raw_copy['timestamp'] = pd.to_datetime(raw_copy['timestamp'])
            
            pred_copy = pred_copy.sort_values('timestamp')
            raw_copy = raw_copy.sort_values('timestamp')
            
            # *** TIMEZONE NORMALIZATION (Using centralized utility) ***
            from utils.time_utils import ensure_naive
            pred_copy['timestamp'] = ensure_naive(pred_copy['timestamp'])
            raw_copy['timestamp'] = ensure_naive(raw_copy['timestamp'])
            
            # Drop existing motion if present
            if 'motion' in pred_copy.columns:
                pred_copy = pred_copy.drop(columns=['motion'])
            
            # Merge using nearest timestamp
            merged = pd.merge_asof(
                pred_copy,
                raw_copy,
                on='timestamp',
                tolerance=pd.Timedelta('1s'),
                direction='nearest'
            )
            
            # Fill any missing motion values
            if 'motion' in merged.columns:
                merged['motion'] = merged['motion'].fillna(0.1)
            
            return merged
            
        except Exception as e:
            logger.warning(f"Failed to merge raw motion: {e}")
            return pred_df
