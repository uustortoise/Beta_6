"""
Item 10: Calibration/Validation Temporal Semantics Hardening

Ensures transparent reporting of calibration vs validation temporal ordering
to prevent misleading metric interpretation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
import os

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CalibrationSemanticsMode(Enum):
    """Calibration semantics reporting modes."""
    TRANSPARENT = "transparent"  # Report both metrics with temporal metadata
    CONSERVATIVE = "conservative"  # Add warnings for temporal ordering issues
    STRICT = "strict"  # Fail if calibration is strictly later than validation


class TimeOrdering(Enum):
    """Temporal ordering between validation and calibration."""
    VALIDATION_BEFORE_CALIBRATION = "validation_before_calibration"
    CALIBRATION_BEFORE_VALIDATION = "calibration_before_validation"
    OVERLAPPING = "overlapping"
    DISJOINT = "disjoint"  # No temporal relationship (different periods)


@dataclass(frozen=True)
class TemporalPartition:
    """Represents a temporal partition with explicit bounds."""
    partition_type: str  # 'train', 'validation', 'calibration'
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None
    sample_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "partition_type": self.partition_type,
            "start_ts": self.start_ts.isoformat() if self.start_ts else None,
            "end_ts": self.end_ts.isoformat() if self.end_ts else None,
            "sample_count": self.sample_count,
        }
    
    def is_valid(self) -> bool:
        """Check if partition has valid temporal bounds."""
        if self.start_ts is None or self.end_ts is None:
            return False
        return self.start_ts <= self.end_ts


@dataclass
class TemporalSemanticsReport:
    """Comprehensive temporal semantics report."""
    train_partition: Optional[TemporalPartition] = None
    val_partition: Optional[TemporalPartition] = None
    calib_partition: Optional[TemporalPartition] = None
    
    validation_vs_calibration_order: TimeOrdering = TimeOrdering.DISJOINT
    temporal_gaps: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metric families
    validation_unthresholded_metrics: Dict[str, Any] = field(default_factory=dict)
    calibration_thresholded_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_partition": self.train_partition.to_dict() if self.train_partition else None,
            "val_partition": self.val_partition.to_dict() if self.val_partition else None,
            "calib_partition": self.calib_partition.to_dict() if self.calib_partition else None,
            "validation_vs_calibration_order": self.validation_vs_calibration_order.value,
            "temporal_gaps": self.temporal_gaps,
            "warnings": self.warnings,
            "validation_unthresholded_metrics": self.validation_unthresholded_metrics,
            "calibration_thresholded_metrics": self.calibration_thresholded_metrics,
        }


class CalibrationSemanticsTracker:
    """
    Tracks and validates temporal semantics between validation and calibration.
    
    This addresses the concern that calibration metrics may appear optimistic
    when calibration data comes from a later holdout period than validation.
    """
    
    def __init__(
        self,
        mode: Optional[CalibrationSemanticsMode] = None,
        emit_warnings: bool = True,
    ):
        """
        Initialize tracker.
        
        Parameters:
        -----------
        mode : CalibrationSemanticsMode
            Reporting mode (default: from env CALIBRATION_SEMANTICS_MODE)
        emit_warnings : bool
            Whether to emit warnings for temporal ordering issues
        """
        if mode is None:
            mode_str = os.getenv("CALIBRATION_SEMANTICS_MODE", "transparent").lower()
            try:
                mode = CalibrationSemanticsMode(mode_str)
            except ValueError:
                logger.warning(f"Invalid CALIBRATION_SEMANTICS_MODE='{mode_str}', using 'transparent'")
                mode = CalibrationSemanticsMode.TRANSPARENT
        
        self.mode = mode
        self.emit_warnings = emit_warnings
        self.report = TemporalSemanticsReport()
    
    def record_partition(
        self,
        partition_type: str,
        start_ts: Optional[Union[datetime, str, pd.Timestamp]],
        end_ts: Optional[Union[datetime, str, pd.Timestamp]],
        sample_count: int,
    ) -> None:
        """
        Record a temporal partition.
        
        Parameters:
        -----------
        partition_type : str
            'train', 'validation', or 'calibration'
        start_ts : datetime or str
            Partition start timestamp
        end_ts : datetime or str
            Partition end timestamp
        sample_count : int
            Number of samples in partition
        """
        # Normalize timestamps
        start = self._normalize_timestamp(start_ts)
        end = self._normalize_timestamp(end_ts)
        
        partition = TemporalPartition(
            partition_type=partition_type,
            start_ts=start,
            end_ts=end,
            sample_count=sample_count,
        )
        
        if partition_type == "train":
            self.report.train_partition = partition
        elif partition_type in ("validation", "val"):
            self.report.val_partition = partition
        elif partition_type in ("calibration", "calib"):
            self.report.calib_partition = partition
        else:
            raise ValueError(f"Unknown partition_type: {partition_type}")
        
        logger.debug(f"Recorded {partition_type} partition: {start} to {end} ({sample_count} samples)")
    
    def analyze_temporal_ordering(self) -> TimeOrdering:
        """
        Analyze temporal ordering between validation and calibration.
        
        Returns:
        --------
        TimeOrdering
            The temporal relationship between validation and calibration
        """
        val = self.report.val_partition
        calib = self.report.calib_partition
        
        if val is None or calib is None:
            self.report.validation_vs_calibration_order = TimeOrdering.DISJOINT
            return TimeOrdering.DISJOINT
        
        if not val.is_valid() or not calib.is_valid():
            self.report.validation_vs_calibration_order = TimeOrdering.DISJOINT
            return TimeOrdering.DISJOINT
        
        # Determine ordering
        if val.end_ts <= calib.start_ts:
            order = TimeOrdering.VALIDATION_BEFORE_CALIBRATION
        elif calib.end_ts <= val.start_ts:
            order = TimeOrdering.CALIBRATION_BEFORE_VALIDATION
        elif val.start_ts < calib.end_ts and calib.start_ts < val.end_ts:
            order = TimeOrdering.OVERLAPPING
        else:
            order = TimeOrdering.DISJOINT
        
        self.report.validation_vs_calibration_order = order
        
        # Emit warnings based on mode
        if self.emit_warnings:
            if order == TimeOrdering.VALIDATION_BEFORE_CALIBRATION:
                warning = (
                    f"Calibration ({calib.start_ts}..{calib.end_ts}) is strictly later "
                    f"than validation ({val.start_ts}..{val.end_ts}). "
                    f"Calibration metrics may appear more optimistic."
                )
                self.report.warnings.append(warning)
                logger.warning(f"[TemporalSemantics] {warning}")
            
            elif order == TimeOrdering.OVERLAPPING:
                warning = (
                    f"Validation ({val.start_ts}..{val.end_ts}) and calibration "
                    f"({calib.start_ts}..{calib.end_ts}) overlap. This may indicate "
                    f"data leakage between splits."
                )
                self.report.warnings.append(warning)
                logger.error(f"[TemporalSemantics] {warning}")
        
        return order
    
    def record_metrics(
        self,
        partition_type: str,
        metrics: Dict[str, Any],
        thresholded: bool = False,
    ) -> None:
        """
        Record metrics for a partition.
        
        Parameters:
        -----------
        partition_type : str
            'validation' or 'calibration'
        metrics : Dict
            Metrics dictionary
        thresholded : bool
            Whether metrics are thresholded (operating point) or unthresholded
        """
        metrics_copy = dict(metrics)
        metrics_copy["partition_type"] = partition_type
        metrics_copy["thresholded"] = thresholded
        metrics_copy["timestamp"] = datetime.utcnow().isoformat()
        
        if partition_type in ("validation", "val"):
            if thresholded:
                logger.warning(
                    "Validation metrics recorded as thresholded. "
                    "Typically validation should be unthresholded."
                )
            self.report.validation_unthresholded_metrics = metrics_copy
        elif partition_type in ("calibration", "calib"):
            self.report.calibration_thresholded_metrics = metrics_copy
        else:
            raise ValueError(f"Unknown partition_type for metrics: {partition_type}")
    
    def validate_strict_mode(self) -> Tuple[bool, List[str]]:
        """
        Validate in strict mode.
        
        Returns:
        --------
        (passes, errors) : Tuple
            passes: True if validation passes
            errors: List of error messages
        """
        if self.mode != CalibrationSemanticsMode.STRICT:
            return True, []
        
        errors = []
        order = self.report.validation_vs_calibration_order
        
        if order == TimeOrdering.VALIDATION_BEFORE_CALIBRATION:
            errors.append(
                "STRICT mode violation: Calibration is strictly later than validation. "
                "This can produce optimistic-looking metrics."
            )
        elif order == TimeOrdering.OVERLAPPING:
            errors.append(
                "STRICT mode violation: Validation and calibration overlap. "
                "This indicates potential data leakage."
            )
        
        return len(errors) == 0, errors
    
    def get_report(self) -> Dict[str, Any]:
        """Get comprehensive temporal semantics report."""
        # Ensure ordering is analyzed
        if self.report.validation_vs_calibration_order == TimeOrdering.DISJOINT:
            self.analyze_temporal_ordering()
        
        report_dict = self.report.to_dict()
        report_dict["mode"] = self.mode.value
        report_dict["timestamp"] = datetime.utcnow().isoformat()
        
        return report_dict
    
    @staticmethod
    def _normalize_timestamp(ts: Optional[Union[datetime, str, pd.Timestamp]]) -> Optional[datetime]:
        """Normalize various timestamp formats to datetime."""
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime()
        if isinstance(ts, str):
            return pd.to_datetime(ts).to_pydatetime()
        raise ValueError(f"Cannot normalize timestamp: {ts} (type: {type(ts)})")


def create_semantics_tracker_from_env() -> CalibrationSemanticsTracker:
    """Create tracker from environment configuration."""
    return CalibrationSemanticsTracker()


def compare_validation_calibration_metrics(
    validation_metrics: Dict[str, Any],
    calibration_metrics: Dict[str, Any],
    val_partition: TemporalPartition,
    calib_partition: TemporalPartition,
) -> Dict[str, Any]:
    """
    Compare validation and calibration metrics with temporal context.
    
    Parameters:
    -----------
    validation_metrics : Dict
        Unthresholded validation metrics
    calibration_metrics : Dict
        Thresholded calibration metrics
    val_partition : TemporalPartition
        Validation temporal bounds
    calib_partition : TemporalPartition
        Calibration temporal bounds
        
    Returns:
    --------
    Dict with comparison and temporal context
    """
    tracker = CalibrationSemanticsTracker()
    tracker.report.val_partition = val_partition
    tracker.report.calib_partition = calib_partition
    tracker.report.validation_unthresholded_metrics = validation_metrics
    tracker.report.calibration_thresholded_metrics = calibration_metrics
    
    order = tracker.analyze_temporal_ordering()
    
    comparison = {
        "temporal_ordering": order.value,
        "validation_metrics": validation_metrics,
        "calibration_metrics": calibration_metrics,
        "val_time_range": val_partition.to_dict() if val_partition else None,
        "calib_time_range": calib_partition.to_dict() if calib_partition else None,
        "interpretation_guidance": _generate_interpretation_guidance(order),
    }
    
    return comparison


def _generate_interpretation_guidance(order: TimeOrdering) -> str:
    """Generate human-readable interpretation guidance."""
    guidance = {
        TimeOrdering.VALIDATION_BEFORE_CALIBRATION: (
            "Calibration data is from a later period than validation. "
            "Calibration metrics may reflect more recent data patterns "
            "and appear more optimistic than validation metrics."
        ),
        TimeOrdering.CALIBRATION_BEFORE_VALIDATION: (
            "Validation data is from a later period than calibration. "
            "This ordering is unusual and may indicate configuration issues."
        ),
        TimeOrdering.OVERLAPPING: (
            "WARNING: Validation and calibration data overlap. "
            "This may indicate data leakage between splits. Results may be unreliable."
        ),
        TimeOrdering.DISJOINT: (
            "Validation and calibration are from disjoint time periods. "
            "No temporal ordering issues detected."
        ),
    }
    return guidance.get(order, "Unknown temporal ordering.")
