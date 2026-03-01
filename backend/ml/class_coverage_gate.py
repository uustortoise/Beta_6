"""
Item 15: Class Coverage Gate (Train/Val/Calibration)

Ensures rare classes are not silently unlearnable due to split sparsity.
Blocks promotion when critical classes are missing from splits.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ClassCoverageGate:
    """
    Gate that verifies class coverage across train/val/calibration splits.
    
    Prevents promotion when:
    - Critical classes are absent from training data
    - Classes have insufficient support in validation/calibration
    - Splits have imbalanced class distribution
    """
    
    def __init__(
        self,
        critical_classes: Optional[List[int]] = None,
        min_train_support: int = 10,
        min_val_support: int = 5,
        min_calib_support: int = 5,
        max_absent_critical_classes: int = 0,
        min_class_coverage_ratio: float = 0.8,
        block_on_val_support: bool = True,
        block_on_calib_support: bool = True,
    ):
        """
        Parameters:
        -----------
        critical_classes : List[int], optional
            Class IDs that must be present in all splits.
            If None, all classes found in training are considered critical.
        min_train_support : int
            Minimum samples per class in training
        min_val_support : int
            Minimum samples per class in validation
        min_calib_support : int
            Minimum samples per class in calibration
        max_absent_critical_classes : int
            Maximum number of critical classes allowed to be absent (default 0)
        min_class_coverage_ratio : float
            Minimum ratio of classes that must have sufficient support
        block_on_val_support : bool
            Whether insufficient validation support should block promotion.
        block_on_calib_support : bool
            Whether insufficient calibration support should block promotion.
        """
        self.critical_classes = set(critical_classes) if critical_classes else None
        self.min_train_support = min_train_support
        self.min_val_support = min_val_support
        self.min_calib_support = min_calib_support
        self.max_absent_critical_classes = max_absent_critical_classes
        self.min_class_coverage_ratio = min_class_coverage_ratio
        self.block_on_val_support = bool(block_on_val_support)
        self.block_on_calib_support = bool(block_on_calib_support)
    
    def analyze_split_coverage(
        self,
        y_train: np.ndarray,
        y_val: Optional[np.ndarray] = None,
        y_calib: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Analyze class coverage across splits.
        
        Parameters:
        -----------
        y_train : np.ndarray
            Training labels
        y_val : np.ndarray, optional
            Validation labels
        y_calib : np.ndarray, optional
            Calibration labels
            
        Returns:
        --------
        Dict with coverage analysis
        """
        # Get unique classes from training
        train_classes, train_counts = np.unique(y_train, return_counts=True)
        train_coverage = dict(zip(train_classes.tolist(), train_counts.tolist()))
        
        # Determine critical classes
        if self.critical_classes is None:
            critical = set(train_classes.tolist())
        else:
            critical = self.critical_classes
        
        # Analyze validation coverage
        val_coverage = {}
        if y_val is not None and len(y_val) > 0:
            val_classes, val_counts = np.unique(y_val, return_counts=True)
            val_coverage = dict(zip(val_classes.tolist(), val_counts.tolist()))
        
        # Analyze calibration coverage
        calib_coverage = {}
        if y_calib is not None and len(y_calib) > 0:
            calib_classes, calib_counts = np.unique(y_calib, return_counts=True)
            calib_coverage = dict(zip(calib_classes.tolist(), calib_counts.tolist()))
        
        # Check critical class presence
        absent_from_train = critical - set(train_coverage.keys())
        absent_from_val = critical - set(val_coverage.keys()) if y_val is not None else set()
        absent_from_calib = critical - set(calib_coverage.keys()) if y_calib is not None else set()
        
        # Check support thresholds
        train_insufficient = {
            cls for cls, count in train_coverage.items()
            if cls in critical and count < self.min_train_support
        }
        
        val_insufficient = set()
        if y_val is not None:
            val_insufficient = {
                cls for cls, count in val_coverage.items()
                if cls in critical and count < self.min_val_support
            }
            # Also include absent classes
            val_insufficient.update(absent_from_val)
        
        calib_insufficient = set()
        if y_calib is not None:
            calib_insufficient = {
                cls for cls, count in calib_coverage.items()
                if cls in critical and count < self.min_calib_support
            }
            # Also include absent classes
            calib_insufficient.update(absent_from_calib)
        
        # Calculate coverage ratios
        total_critical = len(critical)
        train_coverage_ratio = (total_critical - len(absent_from_train)) / max(1, total_critical)
        
        return {
            "critical_classes": sorted(list(critical)),
            "total_critical_classes": total_critical,
            "train_coverage": train_coverage,
            "val_coverage": val_coverage,
            "calib_coverage": calib_coverage,
            "absent_from_train": sorted(list(absent_from_train)),
            "absent_from_val": sorted(list(absent_from_val)),
            "absent_from_calib": sorted(list(absent_from_calib)),
            "train_insufficient_support": sorted(list(train_insufficient)),
            "val_insufficient_support": sorted(list(val_insufficient)),
            "calib_insufficient_support": sorted(list(calib_insufficient)),
            "train_coverage_ratio": train_coverage_ratio,
        }
    
    def evaluate(
        self,
        y_train: np.ndarray,
        y_val: Optional[np.ndarray] = None,
        y_calib: Optional[np.ndarray] = None,
        room_name: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Evaluate class coverage gate.
        
        Parameters:
        -----------
        y_train : np.ndarray
            Training labels
        y_val : np.ndarray, optional
            Validation labels
        y_calib : np.ndarray, optional
            Calibration labels
        room_name : str
            Room name for logging
            
        Returns:
        --------
        Dict with gate results
        """
        reasons = []
        warnings = []
        
        # Analyze coverage
        analysis = self.analyze_split_coverage(y_train, y_val, y_calib)
        
        # Check for absent critical classes in training
        absent_train = analysis.get("absent_from_train", [])
        if len(absent_train) > self.max_absent_critical_classes:
            reasons.append(
                f"Critical classes absent from training: {absent_train} "
                f"({len(absent_train)} > {self.max_absent_critical_classes} allowed)"
            )
        
        # Check training coverage ratio
        train_ratio = analysis.get("train_coverage_ratio", 0.0)
        if train_ratio < self.min_class_coverage_ratio:
            reasons.append(
                f"Insufficient training class coverage: {train_ratio:.1%} < "
                f"{self.min_class_coverage_ratio:.1%}"
            )
        
        # Check validation coverage (if provided)
        if y_val is not None:
            absent_val = analysis.get("absent_from_val", [])
            if absent_val:
                warnings.append(
                    f"Classes absent from validation: {absent_val}"
                )
            
            val_insufficient = analysis.get("val_insufficient_support", [])
            if val_insufficient:
                message = f"Classes with insufficient validation support: {val_insufficient}"
                if self.block_on_val_support:
                    reasons.append(message)
                else:
                    warnings.append(message)
        
        # Check calibration coverage (if provided)
        if y_calib is not None:
            absent_calib = analysis.get("absent_from_calib", [])
            if absent_calib:
                warnings.append(
                    f"Classes absent from calibration: {absent_calib}"
                )
            
            calib_insufficient = analysis.get("calib_insufficient_support", [])
            if calib_insufficient:
                message = f"Classes with insufficient calibration support: {calib_insufficient}"
                if self.block_on_calib_support:
                    reasons.append(message)
                else:
                    warnings.append(message)
        
        # Check support thresholds
        train_insufficient = analysis.get("train_insufficient_support", [])
        if train_insufficient:
            reasons.append(
                f"Classes with insufficient training support: {train_insufficient}"
            )
        
        blocking = len(reasons) > 0
        
        result = {
            "passes": len(reasons) == 0,
            "promotable": len(reasons) == 0,
            "reasons": reasons,
            "warnings": warnings,
            "analysis": analysis,
            "blocking": blocking,
            "gate_name": "class_coverage",
            "room": room_name,
        }
        
        if reasons:
            logger.warning(
                f"[ClassCoverageGate] {room_name} BLOCKED: {reasons}"
            )
        else:
            logger.info(
                f"[ClassCoverageGate] {room_name} PASSED: "
                f"{len(analysis['critical_classes'])} critical classes covered"
            )
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize gate configuration."""
        return {
            "critical_classes": sorted(list(self.critical_classes)) if self.critical_classes else None,
            "min_train_support": self.min_train_support,
            "min_val_support": self.min_val_support,
            "min_calib_support": self.min_calib_support,
            "max_absent_critical_classes": self.max_absent_critical_classes,
            "min_class_coverage_ratio": self.min_class_coverage_ratio,
            "block_on_val_support": self.block_on_val_support,
            "block_on_calib_support": self.block_on_calib_support,
        }


def create_class_coverage_gate_from_policy(
    policy,
    critical_classes: Optional[List[int]] = None,
) -> ClassCoverageGate:
    """
    Create ClassCoverageGate from TrainingPolicy.
    
    Parameters:
    -----------
    policy : TrainingPolicy
        Training policy
    critical_classes : List[int], optional
        Override critical classes from policy
        
    Returns:
    --------
    ClassCoverageGate configured from policy
    """
    calib_policy = policy.calibration
    evidence_profile = str(getattr(policy.release_gate, "evidence_profile", "production") or "production").strip().lower()
    pilot_relaxed = evidence_profile in {"pilot_stage_a", "pilot_stage_b"}
    
    return ClassCoverageGate(
        critical_classes=critical_classes,
        min_train_support=max(10, getattr(calib_policy, 'min_support_per_class', 10)),
        min_val_support=max(5, getattr(calib_policy, 'min_support_per_class', 10) // 2),
        min_calib_support=max(5, calib_policy.min_support_per_class),
        max_absent_critical_classes=0,
        min_class_coverage_ratio=0.8,
        block_on_val_support=not pilot_relaxed,
        block_on_calib_support=not pilot_relaxed,
    )


def check_class_coverage_across_splits(
    y_train: np.ndarray,
    y_val: Optional[np.ndarray],
    y_calib: Optional[np.ndarray],
    room_name: str = "unknown",
    policy = None,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Convenience function for class coverage check.
    
    Returns:
    --------
    (promotable, reasons, result) : Tuple
        promotable: True if coverage is sufficient
        reasons: List of blocking reasons
        result: Full gate result
    """
    if policy is not None:
        gate = create_class_coverage_gate_from_policy(policy)
    else:
        gate = ClassCoverageGate()
    
    result = gate.evaluate(y_train, y_val, y_calib, room_name)
    
    return result["promotable"], result["reasons"], result
