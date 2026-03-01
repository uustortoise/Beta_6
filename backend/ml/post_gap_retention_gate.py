"""
Item 4: Post-Gap Retention Quality Gate

Prevents training on highly fragmented retained data by checking
continuity metrics after gap handling.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PostGapRetentionGate:
    """
    Gate that evaluates data quality after gap handling.
    
    Metrics:
    - retained_ratio: Fraction of rows retained after gap handling
    - contiguous_segment_count: Number of continuous data segments
    - max_segment_length: Length of longest continuous segment
    - median_segment_length: Median length of continuous segments
    """
    
    def __init__(
        self,
        min_retained_ratio: float = 0.5,
        max_contiguous_segments: int = 10,
        min_max_segment_length: int = 100,
        min_median_segment_length: int = 50,
    ):
        """
        Parameters:
        -----------
        min_retained_ratio : float
            Minimum acceptable ratio of retained rows (0.0-1.0)
        max_contiguous_segments : int
            Maximum acceptable number of contiguous segments
            (too many = highly fragmented)
        min_max_segment_length : int
            Minimum length of the longest continuous segment
        min_median_segment_length : int
            Minimum median length of continuous segments
        """
        self.min_retained_ratio = min_retained_ratio
        self.max_contiguous_segments = max_contiguous_segments
        self.min_max_segment_length = min_max_segment_length
        self.min_median_segment_length = min_median_segment_length
    
    def analyze_continuity(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
    ) -> Dict[str, Any]:
        """
        Analyze data continuity to find contiguous segments.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with timestamp column
        timestamp_col : str
            Name of timestamp column
            
        Returns:
        --------
        Dict with continuity metrics
        """
        if timestamp_col not in df.columns:
            return {
                "contiguous_segment_count": 0,
                "max_segment_length": 0,
                "median_segment_length": 0,
                "segment_lengths": [],
                "error": f"Missing {timestamp_col} column",
            }
        
        if len(df) < 2:
            return {
                "contiguous_segment_count": 1 if len(df) == 1 else 0,
                "max_segment_length": len(df),
                "median_segment_length": len(df),
                "segment_lengths": [len(df)] if len(df) > 0 else [],
            }
        
        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        timestamps = pd.to_datetime(df_sorted[timestamp_col])
        
        # Calculate time differences
        time_diffs = timestamps.diff().dropna()
        
        # Detect gaps (larger than expected interval)
        # Use median interval as baseline
        median_interval = time_diffs.median()
        gap_threshold = median_interval * 3  # 3x median interval = gap
        
        # Find segment boundaries
        is_gap = time_diffs > gap_threshold
        segment_boundaries = [0] + (is_gap[is_gap].index.tolist()) + [len(df)]
        
        # Calculate segment lengths
        segment_lengths = []
        for i in range(len(segment_boundaries) - 1):
            length = segment_boundaries[i + 1] - segment_boundaries[i]
            if length > 0:
                segment_lengths.append(length)
        
        if not segment_lengths:
            segment_lengths = [len(df)]
        
        return {
            "contiguous_segment_count": len(segment_lengths),
            "max_segment_length": max(segment_lengths),
            "median_segment_length": int(np.median(segment_lengths)),
            "segment_lengths": segment_lengths,
            "gap_threshold_seconds": gap_threshold.total_seconds(),
            "median_interval_seconds": median_interval.total_seconds(),
        }
    
    def evaluate(
        self,
        raw_df: pd.DataFrame,
        post_gap_df: pd.DataFrame,
        room_name: str = "unknown",
        timestamp_col: str = 'timestamp',
    ) -> Dict[str, Any]:
        """
        Evaluate post-gap retention quality.
        
        Parameters:
        -----------
        raw_df : pd.DataFrame
            Original DataFrame before gap handling
        post_gap_df : pd.DataFrame
            DataFrame after gap handling
        room_name : str
            Room name for logging
        timestamp_col : str
            Name of timestamp column
            
        Returns:
        --------
        Dict with:
            - passes: bool - True if all checks pass
            - promotable: bool - True if data quality is sufficient
            - reasons: List[str] - List of failure reasons
            - metrics: Dict - Detailed retention metrics
            - blocking: bool - True if failures are blocking
        """
        reasons = []
        warnings = []
        
        # Calculate retention ratio
        raw_count = len(raw_df)
        post_gap_count = len(post_gap_df)
        retained_ratio = post_gap_count / raw_count if raw_count > 0 else 0.0
        
        # Analyze continuity of post-gap data
        continuity = self.analyze_continuity(post_gap_df, timestamp_col)
        
        metrics = {
            "raw_samples": raw_count,
            "post_gap_samples": post_gap_count,
            "retained_ratio": float(retained_ratio),
            **continuity,
        }
        
        # Check retention ratio
        if retained_ratio < self.min_retained_ratio:
            reasons.append(
                f"Insufficient retention ratio: {retained_ratio:.2%} < "
                f"{self.min_retained_ratio:.2%}"
            )
        
        # Check for excessive fragmentation
        segment_count = continuity.get("contiguous_segment_count", 0)
        if segment_count > self.max_contiguous_segments:
            reasons.append(
                f"Excessive data fragmentation: {segment_count} segments > "
                f"{self.max_contiguous_segments} max"
            )
        
        # Check max segment length
        max_seg_len = continuity.get("max_segment_length", 0)
        if max_seg_len < self.min_max_segment_length:
            reasons.append(
                f"Longest segment too short: {max_seg_len} < "
                f"{self.min_max_segment_length} samples"
            )
        
        # Check median segment length
        median_seg_len = continuity.get("median_segment_length", 0)
        if median_seg_len < self.min_median_segment_length:
            warnings.append(
                f"Median segment length low: {median_seg_len} < "
                f"{self.min_median_segment_length} samples"
            )
        
        # Determine if blocking
        blocking = len(reasons) > 0
        
        result = {
            "passes": len(reasons) == 0,
            "promotable": len(reasons) == 0,
            "reasons": reasons,
            "warnings": warnings,
            "metrics": metrics,
            "blocking": blocking,
            "gate_name": "post_gap_retention",
            "room": room_name,
        }
        
        if reasons:
            logger.warning(
                f"[PostGapRetentionGate] {room_name} BLOCKED: {reasons}"
            )
        else:
            logger.info(
                f"[PostGapRetentionGate] {room_name} PASSED: "
                f"{retained_ratio:.1%} retained, {segment_count} segments"
            )
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize gate configuration."""
        return {
            "min_retained_ratio": self.min_retained_ratio,
            "max_contiguous_segments": self.max_contiguous_segments,
            "min_max_segment_length": self.min_max_segment_length,
            "min_median_segment_length": self.min_median_segment_length,
        }


def create_post_gap_retention_gate_from_policy(policy) -> PostGapRetentionGate:
    """
    Create PostGapRetentionGate from TrainingPolicy.
    
    Parameters:
    -----------
    policy : TrainingPolicy
        Training policy with data viability settings
        
    Returns:
    --------
    PostGapRetentionGate configured from policy
    """
    viability = policy.data_viability
    
    return PostGapRetentionGate(
        min_retained_ratio=getattr(viability, 'min_retained_ratio', 0.5),
        max_contiguous_segments=getattr(viability, 'max_contiguous_segments', 10),
        min_max_segment_length=getattr(viability, 'min_max_segment_length', 100),
        min_median_segment_length=getattr(viability, 'min_median_segment_length', 50),
    )
