"""
Item 6: Statistical Validity Gate Tightening

Prevents low-support high-F1 illusions by enforcing minimum
calibration/validation support floors before promotion.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class StatisticalValidityGate:
    """
    Gate that blocks promotion when statistical evidence is insufficient.
    
    Enforces:
    - Minimum samples per class in calibration/validation
    - Minimum number of classes with sufficient support
    - Minimum minority class support
    """
    
    def __init__(
        self,
        min_calibration_support: int = 50,
        min_validation_support: int = 50,
        min_promotable_class_count: int = 2,
        min_minority_support: int = 10,
        min_support_per_class: Optional[Dict[int, int]] = None,
        pilot_relaxed_evidence: bool = False,
    ):
        """
        Parameters:
        -----------
        min_calibration_support : int
            Total minimum samples required in calibration set
        min_validation_support : int
            Total minimum samples required in validation set
        min_promotable_class_count : int
            Minimum number of classes that must have sufficient support
        min_minority_support : int
            Minimum samples for the least frequent class
        min_support_per_class : Dict[int, int], optional
            Per-class minimum support requirements {class_id: min_samples}
        pilot_relaxed_evidence : bool
            When enabled, low-support failures are emitted as non-blocking
            NOT_EVALUATED signals instead of blocking promotion.
        """
        self.min_calibration_support = min_calibration_support
        self.min_validation_support = min_validation_support
        self.min_promotable_class_count = min_promotable_class_count
        self.min_minority_support = min_minority_support
        self.min_support_per_class = min_support_per_class or {}
        self.pilot_relaxed_evidence = bool(pilot_relaxed_evidence)

    @staticmethod
    def _is_low_support_reason_code(reason_code: str) -> bool:
        return reason_code in {
            "insufficient_calibration_support",
            "insufficient_validation_support",
            "insufficient_classes_with_support",
            "insufficient_minority_class_support",
            "per_class_minimum_violations",
        }
    
    def evaluate(
        self,
        y_calib: np.ndarray,
        y_val: Optional[np.ndarray] = None,
        room_name: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Evaluate statistical validity for promotion.
        
        Parameters:
        -----------
        y_calib : np.ndarray
            Calibration set labels
        y_val : np.ndarray, optional
            Validation set labels (if separate from calibration)
        room_name : str
            Room name for logging
            
        Returns:
        --------
        Dict with:
            - passes: bool - True if all checks pass
            - promotable: bool - True if model can be promoted
            - reasons: List[str] - List of failure reasons (empty if passes)
            - metrics: Dict - Detailed support metrics
            - blocking: bool - True if failures are blocking (not just warnings)
        """
        reasons: List[str] = []
        warnings: List[str] = []
        reason_records: List[Dict[str, str]] = []
        metrics = {}

        def _add_reason(reason_code: str, message: str) -> None:
            reasons.append(message)
            reason_records.append(
                {
                    "code": str(reason_code),
                    "message": str(message),
                    "category": "low_support" if self._is_low_support_reason_code(reason_code) else "hard",
                }
            )
        
        # Check total calibration support
        total_calib = len(y_calib)
        metrics['total_calibration_samples'] = total_calib
        
        if total_calib < self.min_calibration_support:
            _add_reason(
                "insufficient_calibration_support",
                f"Insufficient calibration support: {total_calib} < {self.min_calibration_support}"
            )
        
        # Check validation support if provided
        if y_val is not None:
            total_val = len(y_val)
            metrics['total_validation_samples'] = total_val
            
            if total_val < self.min_validation_support:
                _add_reason(
                    "insufficient_validation_support",
                    f"Insufficient validation support: {total_val} < {self.min_validation_support}"
                )
        else:
            total_val = 0
            metrics['total_validation_samples'] = 0
        
        # Per-class support analysis
        unique_classes, class_counts = np.unique(y_calib, return_counts=True)
        class_support = dict(zip(unique_classes.tolist(), class_counts.tolist()))
        metrics['class_support'] = class_support
        
        # Check minimum promotable class count
        classes_with_sufficient_support = sum(
            1 for count in class_support.values()
            if count >= self.min_minority_support
        )
        metrics['classes_with_sufficient_support'] = classes_with_sufficient_support
        metrics['total_classes'] = len(unique_classes)
        
        if classes_with_sufficient_support < self.min_promotable_class_count:
            _add_reason(
                "insufficient_classes_with_support",
                f"Insufficient classes with support: {classes_with_sufficient_support} "
                f"< {self.min_promotable_class_count} (min_support={self.min_minority_support})"
            )
        
        # Check minority class support
        if class_counts.size > 0:
            minority_count = int(class_counts.min())
            minority_class = int(unique_classes[np.argmin(class_counts)])
            metrics['minority_class'] = minority_class
            metrics['minority_support'] = minority_count
            
            if minority_count < self.min_minority_support:
                _add_reason(
                    "insufficient_minority_class_support",
                    f"Insufficient minority class support: class {minority_class} "
                    f"has {minority_count} < {self.min_minority_support} samples"
                )
        
        # Check per-class minimums
        per_class_violations = []
        for class_id, min_required in self.min_support_per_class.items():
            actual = class_support.get(class_id, 0)
            if actual < min_required:
                per_class_violations.append(
                    f"Class {class_id}: {actual} < {min_required}"
                )
        
        if per_class_violations:
            _add_reason(
                "per_class_minimum_violations",
                f"Per-class minimum violations: {'; '.join(per_class_violations)}",
            )

        if self.pilot_relaxed_evidence:
            blocking_reasons = [r["message"] for r in reason_records if r["category"] == "hard"]
            non_blocking_reasons = [r["message"] for r in reason_records if r["category"] == "low_support"]
        else:
            blocking_reasons = list(reasons)
            non_blocking_reasons = []

        blocking = len(blocking_reasons) > 0
        if non_blocking_reasons:
            warnings.extend(non_blocking_reasons)

        if len(reasons) == 0:
            evaluation_status = "pass"
        elif not blocking and non_blocking_reasons:
            evaluation_status = "not_evaluated"
        else:
            evaluation_status = "fail"

        result = {
            "passes": len(reasons) == 0,
            "promotable": not blocking,
            "reasons": reasons,
            "warnings": warnings,
            "metrics": metrics,
            "blocking": blocking,
            "blocking_reasons": blocking_reasons,
            "non_blocking_reasons": non_blocking_reasons,
            "reason_codes": [r["code"] for r in reason_records],
            "evaluation_status": evaluation_status,
            "gate_name": "statistical_validity",
            "room": room_name,
        }

        if blocking:
            logger.warning(
                f"[StatisticalValidityGate] {room_name} BLOCKED: {blocking_reasons}"
            )
        elif non_blocking_reasons:
            logger.info(
                f"[StatisticalValidityGate] {room_name} NOT_EVALUATED: {non_blocking_reasons}"
            )
        else:
            logger.info(
                f"[StatisticalValidityGate] {room_name} PASSED: "
                f"{classes_with_sufficient_support}/{len(unique_classes)} classes with sufficient support"
            )
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize gate configuration."""
        return {
            "min_calibration_support": self.min_calibration_support,
            "min_validation_support": self.min_validation_support,
            "min_promotable_class_count": self.min_promotable_class_count,
            "min_minority_support": self.min_minority_support,
            "min_support_per_class": self.min_support_per_class,
        }


def create_statistical_validity_gate_from_policy(policy) -> StatisticalValidityGate:
    """
    Create StatisticalValidityGate from TrainingPolicy.
    
    Parameters:
    -----------
    policy : TrainingPolicy
        Training policy with calibration settings
        
    Returns:
    --------
    StatisticalValidityGate configured from policy
    """
    calib_policy = policy.calibration
    
    evidence_profile = str(
        getattr(getattr(policy, "release_gate", object()), "evidence_profile", "production") or "production"
    ).strip().lower()
    pilot_relaxed_evidence = evidence_profile in {"pilot_stage_a", "pilot_stage_b"}

    return StatisticalValidityGate(
        min_calibration_support=max(50, calib_policy.min_samples),
        min_validation_support=max(50, calib_policy.min_samples),
        min_promotable_class_count=2,
        min_minority_support=max(10, calib_policy.min_support_per_class),
        pilot_relaxed_evidence=pilot_relaxed_evidence,
    )


def evaluate_promotion_with_statistical_validity(
    calibration_metrics: Dict[str, Any],
    y_calib: np.ndarray,
    y_val: Optional[np.ndarray] = None,
    room_name: str = "unknown",
    policy=None,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Evaluate promotion eligibility with statistical validity gate.
    
    Parameters:
    -----------
    calibration_metrics : Dict
        Metrics from calibration evaluation (may contain fallback flags)
    y_calib : np.ndarray
        Calibration labels
    y_val : np.ndarray, optional
        Validation labels
    room_name : str
        Room name
    policy : TrainingPolicy, optional
        Policy for gate configuration
        
    Returns:
    --------
    (promotable, reasons, gate_result) : Tuple
        promotable: True if model can be promoted
        reasons: List of blocking reasons
        gate_result: Full gate evaluation result
    """
    # Create gate from policy or use defaults
    if policy is not None:
        gate = create_statistical_validity_gate_from_policy(policy)
    else:
        gate = StatisticalValidityGate()
    
    # Run gate evaluation
    gate_result = gate.evaluate(y_calib, y_val, room_name)
    
    # Check for fallback-only evidence in calibration metrics
    if calibration_metrics.get("is_fallback", False):
        gate_result["reasons"].append(
            "Calibration used fallback metrics (insufficient support for proper evaluation)"
        )
        gate_result["promotable"] = False
        gate_result["blocking"] = True
    
    # Check for metric source flags
    metric_source = calibration_metrics.get("metric_source", "unknown")
    if metric_source == "fallback":
        gate_result["reasons"].append(
            "Metrics derived from fallback computation (not full evaluation)"
        )
        gate_result["promotable"] = False
        gate_result["blocking"] = True
    
    reasons = gate_result.get("blocking_reasons") or gate_result["reasons"]
    promotable = gate_result["promotable"]
    
    return promotable, reasons, gate_result
