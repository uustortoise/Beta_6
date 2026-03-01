"""
Item 14: Training Data Quality Contract (Sensor/Label)

Fail early on bad upstream data with comprehensive pre-train checks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class QualityCheckType(Enum):
    """Types of data quality checks."""
    REQUIRED_COLUMNS = "required_columns"
    TIMESTAMP_MONOTONICITY = "timestamp_monotonicity"
    SENSOR_MISSINGNESS = "sensor_missingness"
    LABEL_DISTRIBUTION = "label_distribution"
    TIMESTAMP_DUPLICATES = "timestamp_duplicates"
    TIMESTAMP_RANGE = "timestamp_range"
    VALUE_RANGE = "value_range"
    LABEL_VALIDITY = "label_validity"


class QualitySeverity(Enum):
    """Severity levels for quality violations."""
    CRITICAL = "critical"  # Blocks training
    HIGH = "high"  # Warning, likely blocks
    MEDIUM = "medium"  # Warning, may proceed
    LOW = "low"  # Advisory only


@dataclass
class QualityViolation:
    """Individual quality violation."""
    check_type: QualityCheckType
    severity: QualitySeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_type": self.check_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class DataQualityReport:
    """
    Comprehensive data quality report.
    
    Generated before training to validate data meets minimum standards.
    """
    timestamp: str
    elder_id: str
    room_name: str
    
    # Overall status
    passes: bool = True
    critical_violations: int = 0
    high_violations: int = 0
    medium_violations: int = 0
    low_violations: int = 0
    
    # Violations by category
    violations: List[QualityViolation] = field(default_factory=list)
    
    # Check results
    required_columns_pass: bool = True
    timestamp_monotonicity_pass: bool = True
    sensor_missingness_pass: bool = True
    label_distribution_pass: bool = True
    timestamp_duplicates_pass: bool = True
    timestamp_range_pass: bool = True
    value_range_pass: bool = True
    label_validity_pass: bool = True
    
    # Detailed metrics
    column_stats: Dict[str, Any] = field(default_factory=dict)
    missingness_stats: Dict[str, float] = field(default_factory=dict)
    label_distribution: Dict[str, int] = field(default_factory=dict)
    timestamp_stats: Dict[str, Any] = field(default_factory=dict)
    value_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "elder_id": self.elder_id,
            "room_name": self.room_name,
            "passes": self.passes,
            "violation_summary": {
                "critical": self.critical_violations,
                "high": self.high_violations,
                "medium": self.medium_violations,
                "low": self.low_violations,
            },
            "violations": [v.to_dict() for v in self.violations],
            "check_results": {
                "required_columns": self.required_columns_pass,
                "timestamp_monotonicity": self.timestamp_monotonicity_pass,
                "sensor_missingness": self.sensor_missingness_pass,
                "label_distribution": self.label_distribution_pass,
                "timestamp_duplicates": self.timestamp_duplicates_pass,
                "timestamp_range": self.timestamp_range_pass,
                "value_range": self.value_range_pass,
                "label_validity": self.label_validity_pass,
            },
            "column_stats": self.column_stats,
            "missingness_stats": self.missingness_stats,
            "label_distribution": self.label_distribution,
            "timestamp_stats": self.timestamp_stats,
            "value_stats": self.value_stats,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save report to file."""
        filepath = Path(filepath)
        filepath.write_text(self.to_json())
        logger.info(f"Data quality report saved to {filepath}")


class DataQualityContract:
    """
    Contract for training data quality.
    
    Performs comprehensive pre-train checks to fail fast on bad data.
    """
    
    # Default required columns
    DEFAULT_REQUIRED_SENSOR_COLUMNS = [
        "motion", "temperature", "light", "sound", "co2", "humidity", "vibration"
    ]
    
    DEFAULT_REQUIRED_LABEL_COLUMNS = ["activity"]
    
    # Default thresholds
    DEFAULT_MAX_MISSINGNESS_RATIO = 0.30  # 30% max missing per sensor
    DEFAULT_MIN_LABEL_SAMPLES = 5  # Minimum samples per class
    DEFAULT_MAX_DUPLICATE_RATIO = 0.10  # 10% max duplicate timestamps
    DEFAULT_TIMESTAMP_RANGE_DAYS = 1  # At least 1 day of data
    
    def __init__(
        self,
        required_sensor_columns: Optional[List[str]] = None,
        required_label_columns: Optional[List[str]] = None,
        max_missingness_ratio: float = DEFAULT_MAX_MISSINGNESS_RATIO,
        min_label_samples: int = DEFAULT_MIN_LABEL_SAMPLES,
        max_duplicate_ratio: float = DEFAULT_MAX_DUPLICATE_RATIO,
        min_timestamp_range_days: int = DEFAULT_TIMESTAMP_RANGE_DAYS,
        fail_on_critical: bool = True,
    ):
        """
        Initialize quality contract.
        
        Parameters:
        -----------
        required_sensor_columns : List[str]
            Required sensor columns
        required_label_columns : List[str]
            Required label columns
        max_missingness_ratio : float
            Maximum allowed missingness ratio per sensor
        min_label_samples : int
            Minimum samples required per class
        max_duplicate_ratio : float
            Maximum allowed duplicate timestamp ratio
        min_timestamp_range_days : int
            Minimum required timestamp range in days
        fail_on_critical : bool
            Whether critical violations block training
        """
        self.required_sensor_columns = set(
            required_sensor_columns or self.DEFAULT_REQUIRED_SENSOR_COLUMNS
        )
        self.required_label_columns = set(
            required_label_columns or self.DEFAULT_REQUIRED_LABEL_COLUMNS
        )
        self.max_missingness_ratio = max_missingness_ratio
        self.min_label_samples = min_label_samples
        self.max_duplicate_ratio = max_duplicate_ratio
        self.min_timestamp_range_days = min_timestamp_range_days
        self.fail_on_critical = fail_on_critical
    
    def validate(
        self,
        df: pd.DataFrame,
        elder_id: str,
        room_name: str,
        timestamp_col: str = "timestamp",
    ) -> DataQualityReport:
        """
        Validate data against quality contract.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        elder_id : str
            Elder/resident ID
        room_name : str
            Room name
        timestamp_col : str
            Timestamp column name
            
        Returns:
        --------
        DataQualityReport
        """
        report = DataQualityReport(
            timestamp=datetime.utcnow().isoformat(),
            elder_id=elder_id,
            room_name=room_name,
        )
        
        # Run all checks
        report.required_columns_pass = self._check_required_columns(df, report)
        report.timestamp_monotonicity_pass = self._check_timestamp_monotonicity(
            df, timestamp_col, report
        )
        report.sensor_missingness_pass = self._check_sensor_missingness(df, report)
        report.label_distribution_pass = self._check_label_distribution(df, report)
        report.timestamp_duplicates_pass = self._check_timestamp_duplicates(
            df, timestamp_col, report
        )
        report.timestamp_range_pass = self._check_timestamp_range(
            df, timestamp_col, report
        )
        report.value_range_pass = self._check_value_ranges(df, report)
        report.label_validity_pass = self._check_label_validity(df, report)
        
        # Count violations
        report.critical_violations = sum(
            1 for v in report.violations if v.severity == QualitySeverity.CRITICAL
        )
        report.high_violations = sum(
            1 for v in report.violations if v.severity == QualitySeverity.HIGH
        )
        report.medium_violations = sum(
            1 for v in report.violations if v.severity == QualitySeverity.MEDIUM
        )
        report.low_violations = sum(
            1 for v in report.violations if v.severity == QualitySeverity.LOW
        )
        
        # Determine overall pass/fail
        if self.fail_on_critical and report.critical_violations > 0:
            report.passes = False
        elif report.high_violations > 2:  # Too many high severity issues
            report.passes = False
        
        return report
    
    def _check_required_columns(
        self,
        df: pd.DataFrame,
        report: DataQualityReport,
    ) -> bool:
        """Check required columns exist."""
        actual_columns = set(df.columns)
        
        # Record column stats
        report.column_stats = {
            "total_columns": len(actual_columns),
            "column_names": sorted(list(actual_columns)),
        }
        
        # Check sensor columns
        missing_sensors = self.required_sensor_columns - actual_columns
        if missing_sensors:
            report.violations.append(QualityViolation(
                check_type=QualityCheckType.REQUIRED_COLUMNS,
                severity=QualitySeverity.CRITICAL,
                message=f"Missing required sensor columns: {sorted(missing_sensors)}",
                details={"missing": sorted(missing_sensors)},
            ))
            return False
        
        # Check label columns
        missing_labels = self.required_label_columns - actual_columns
        if missing_labels:
            report.violations.append(QualityViolation(
                check_type=QualityCheckType.REQUIRED_COLUMNS,
                severity=QualitySeverity.CRITICAL,
                message=f"Missing required label columns: {sorted(missing_labels)}",
                details={"missing": sorted(missing_labels)},
            ))
            return False
        
        return True
    
    def _check_timestamp_monotonicity(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        report: DataQualityReport,
    ) -> bool:
        """Check timestamps are monotonically increasing."""
        if timestamp_col not in df.columns:
            return False  # Already caught in required columns
        
        timestamps = pd.to_datetime(df[timestamp_col])
        
        # Check for non-monotonic timestamps
        diffs = timestamps.diff().dropna()
        non_monotonic = (diffs < pd.Timedelta(0)).sum()
        
        report.timestamp_stats["non_monotonic_count"] = int(non_monotonic)
        
        if non_monotonic > 0:
            report.violations.append(QualityViolation(
                check_type=QualityCheckType.TIMESTAMP_MONOTONICITY,
                severity=QualitySeverity.HIGH,
                message=f"{non_monotonic} timestamps are not monotonically increasing",
                details={"non_monotonic_count": int(non_monotonic)},
            ))
            return False
        
        return True
    
    def _check_sensor_missingness(
        self,
        df: pd.DataFrame,
        report: DataQualityReport,
    ) -> bool:
        """Check sensor missingness is within bounds."""
        sensor_cols = list(self.required_sensor_columns & set(df.columns))
        
        if not sensor_cols:
            return False
        
        missingness = {}
        violations = []
        
        for col in sensor_cols:
            if col not in df.columns:
                continue
            
            missing_ratio = df[col].isna().mean()
            missingness[col] = float(missing_ratio)
            
            if missing_ratio > self.max_missingness_ratio:
                violations.append({
                    "column": col,
                    "missing_ratio": float(missing_ratio),
                })
        
        report.missingness_stats = missingness
        
        if violations:
            report.violations.append(QualityViolation(
                check_type=QualityCheckType.SENSOR_MISSINGNESS,
                severity=QualitySeverity.HIGH,
                message=f"{len(violations)} sensors exceed {self.max_missingness_ratio:.0%} missingness",
                details={"violations": violations, "threshold": self.max_missingness_ratio},
            ))
            return False
        
        return True
    
    def _check_label_distribution(
        self,
        df: pd.DataFrame,
        report: DataQualityReport,
    ) -> bool:
        """Check label distribution is sane."""
        label_col = "activity"  # Default label column
        
        if label_col not in df.columns:
            return False
        
        # Get label distribution
        label_counts = df[label_col].value_counts().to_dict()
        report.label_distribution = {str(k): int(v) for k, v in label_counts.items()}
        
        # Check for classes with too few samples
        low_support_classes = [
            str(cls) for cls, count in label_counts.items()
            if count < self.min_label_samples
        ]
        
        if low_support_classes:
            report.violations.append(QualityViolation(
                check_type=QualityCheckType.LABEL_DISTRIBUTION,
                severity=QualitySeverity.HIGH,
                message=f"Classes with insufficient support: {low_support_classes}",
                details={
                    "low_support_classes": low_support_classes,
                    "min_required": self.min_label_samples,
                },
            ))
            return False
        
        return True
    
    def _check_timestamp_duplicates(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        report: DataQualityReport,
    ) -> bool:
        """Check duplicate timestamp ratio."""
        if timestamp_col not in df.columns:
            return False
        
        total = len(df)
        unique = df[timestamp_col].nunique()
        duplicates = total - unique
        duplicate_ratio = duplicates / total if total > 0 else 0.0
        
        report.timestamp_stats["total_rows"] = total
        report.timestamp_stats["unique_timestamps"] = unique
        report.timestamp_stats["duplicate_count"] = duplicates
        report.timestamp_stats["duplicate_ratio"] = float(duplicate_ratio)
        
        if duplicate_ratio > self.max_duplicate_ratio:
            report.violations.append(QualityViolation(
                check_type=QualityCheckType.TIMESTAMP_DUPLICATES,
                severity=QualitySeverity.MEDIUM,
                message=f"High duplicate timestamp ratio: {duplicate_ratio:.1%}",
                details={
                    "duplicate_ratio": float(duplicate_ratio),
                    "threshold": self.max_duplicate_ratio,
                },
            ))
            return False
        
        return True
    
    def _check_timestamp_range(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        report: DataQualityReport,
    ) -> bool:
        """Check timestamp range is sufficient."""
        if timestamp_col not in df.columns:
            return False
        
        timestamps = pd.to_datetime(df[timestamp_col])
        
        if len(timestamps) < 2:
            report.violations.append(QualityViolation(
                check_type=QualityCheckType.TIMESTAMP_RANGE,
                severity=QualitySeverity.CRITICAL,
                message="Insufficient timestamp data (less than 2 points)",
                details={},
            ))
            return False
        
        min_ts = timestamps.min()
        max_ts = timestamps.max()
        range_days = (max_ts - min_ts).total_seconds() / 86400
        
        report.timestamp_stats["min_timestamp"] = min_ts.isoformat()
        report.timestamp_stats["max_timestamp"] = max_ts.isoformat()
        report.timestamp_stats["range_days"] = float(range_days)
        
        if range_days < self.min_timestamp_range_days:
            report.violations.append(QualityViolation(
                check_type=QualityCheckType.TIMESTAMP_RANGE,
                severity=QualitySeverity.CRITICAL,
                message=f"Insufficient timestamp range: {range_days:.1f} days < {self.min_timestamp_range_days} required",
                details={
                    "range_days": float(range_days),
                    "min_required": self.min_timestamp_range_days,
                },
            ))
            return False
        
        return True
    
    def _check_value_ranges(
        self,
        df: pd.DataFrame,
        report: DataQualityReport,
    ) -> bool:
        """Check sensor value ranges are reasonable."""
        sensor_cols = list(self.required_sensor_columns & set(df.columns))
        
        value_stats = {}
        has_issues = False
        
        for col in sensor_cols:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            col_data = df[col].dropna()
            
            stats = {
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
            }
            value_stats[col] = stats
            
            # Check for extreme values that might indicate sensor error
            # (e.g., all zeros, all same value)
            if col_data.std() == 0:
                report.violations.append(QualityViolation(
                    check_type=QualityCheckType.VALUE_RANGE,
                    severity=QualitySeverity.MEDIUM,
                    message=f"Sensor '{col}' has zero variance (constant value)",
                    details={"column": col, "constant_value": float(col_data.iloc[0])},
                ))
                has_issues = True
        
        report.value_stats = value_stats
        
        return not has_issues
    
    def _check_label_validity(
        self,
        df: pd.DataFrame,
        report: DataQualityReport,
    ) -> bool:
        """Check labels are valid."""
        label_col = "activity"
        
        if label_col not in df.columns:
            return False
        
        labels = df[label_col]
        
        # Check for NaN labels
        nan_count = labels.isna().sum()
        if nan_count > 0:
            report.violations.append(QualityViolation(
                check_type=QualityCheckType.LABEL_VALIDITY,
                severity=QualitySeverity.CRITICAL,
                message=f"{nan_count} rows have missing activity labels",
                details={"nan_count": int(nan_count)},
            ))
            return False
        
        # Check for empty string labels
        empty_count = (labels.astype(str).str.strip() == "").sum()
        if empty_count > 0:
            report.violations.append(QualityViolation(
                check_type=QualityCheckType.LABEL_VALIDITY,
                severity=QualitySeverity.CRITICAL,
                message=f"{empty_count} rows have empty activity labels",
                details={"empty_count": int(empty_count)},
            ))
            return False
        
        return True


def create_contract_from_env() -> DataQualityContract:
    """Create quality contract from environment variables."""
    required_sensors = os.getenv("REQUIRED_SENSOR_COLUMNS")
    if required_sensors:
        required_sensors = [s.strip() for s in required_sensors.split(",")]
    
    return DataQualityContract(
        required_sensor_columns=required_sensors,
        max_missingness_ratio=float(os.getenv("MAX_MISSINGNESS_RATIO", "0.30")),
        min_label_samples=int(os.getenv("MIN_LABEL_SAMPLES", "5")),
        max_duplicate_ratio=float(os.getenv("MAX_DUPLICATE_RATIO", "0.10")),
        min_timestamp_range_days=int(os.getenv("MIN_TIMESTAMP_RANGE_DAYS", "1")),
    )


def validate_training_data(
    df: pd.DataFrame,
    elder_id: str,
    room_name: str,
    output_path: Optional[str] = None,
) -> Tuple[bool, DataQualityReport]:
    """
    Convenience function to validate training data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Training data
    elder_id : str
        Elder ID
    room_name : str
        Room name
    output_path : str, optional
        Path to save quality report
        
    Returns:
    --------
    (passes, report) : Tuple
        passes: True if data passes quality checks
        report: Detailed quality report
    """
    contract = create_contract_from_env()
    report = contract.validate(df, elder_id, room_name)
    
    if output_path:
        report.save(output_path)
    
    return report.passes, report
