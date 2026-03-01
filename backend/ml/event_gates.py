"""
Event Gates Module

Tiered gate checks for event-first CNN-Transformer outputs.
Part of PR-B2: Event KPI + Gate Layer.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CriticalityTier(Enum):
    """Criticality tiers for event gates."""
    TIER_1 = 1  # Critical: must have >= 0.50 recall
    TIER_2 = 2  # Important: must have >= 0.35 recall
    TIER_3 = 3  # Standard: must have >= 0.20 recall


class GateStatus(Enum):
    """Status of a gate check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_EVALUATED = "not_evaluated"


@dataclass
class EventGateThresholds:
    """Thresholds for event-level gate checks."""
    
    # Home-empty safety gates (hard requirements)
    home_empty_precision_min: float = 0.95
    home_empty_false_empty_rate_max: float = 0.05
    
    # Unknown rate caps
    unknown_rate_global_max: float = 0.15
    unknown_rate_per_room_max: float = 0.20
    
    # Criticality tier recall minimums
    tier_1_recall_min: float = 0.50
    tier_2_recall_min: float = 0.35
    tier_3_recall_min: float = 0.20
    
    # Collapse detection
    collapse_recall_threshold: float = 0.02
    collapse_support_threshold: int = 30
    min_support_for_tier_gates: int = 30
    
    def validate(self) -> None:
        """Validate threshold configuration."""
        if not (0 <= self.home_empty_precision_min <= 1):
            raise ValueError("home_empty_precision_min must be in [0, 1]")
        if not (0 <= self.home_empty_false_empty_rate_max <= 1):
            raise ValueError("home_empty_false_empty_rate_max must be in [0, 1]")
        if not (0 <= self.unknown_rate_global_max <= 1):
            raise ValueError("unknown_rate_global_max must be in [0, 1]")
        if not (0 <= self.unknown_rate_per_room_max <= 1):
            raise ValueError("unknown_rate_per_room_max must be in [0, 1]")
        if not (self.tier_1_recall_min > self.tier_2_recall_min > self.tier_3_recall_min):
            raise ValueError("tier recall mins must satisfy tier_1 > tier_2 > tier_3")
        if self.collapse_support_threshold < 1:
            raise ValueError("collapse_support_threshold must be >= 1")
        if self.min_support_for_tier_gates < 1:
            raise ValueError("min_support_for_tier_gates must be >= 1")


@dataclass
class GateResult:
    """Result of a single gate check."""
    
    gate_name: str
    status: GateStatus
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_pass(self) -> bool:
        """Check if gate passed."""
        return self.status == GateStatus.PASS
    
    @property
    def is_fail(self) -> bool:
        """Check if gate failed."""
        return self.status == GateStatus.FAIL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gate_name": self.gate_name,
            "status": self.status.value,
            "metric_value": self.metric_value,
            "threshold_value": self.threshold_value,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class EventGateReport:
    """Complete gate report for a validation run."""
    
    date: date
    overall_status: GateStatus
    results: List[GateResult]
    
    # Aggregated metrics
    pass_count: int = 0
    fail_count: int = 0
    warning_count: int = 0
    
    # Critical events
    critical_failures: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate aggregated metrics."""
        self.pass_count = sum(1 for r in self.results if r.status == GateStatus.PASS)
        self.fail_count = sum(1 for r in self.results if r.status == GateStatus.FAIL)
        self.warning_count = sum(1 for r in self.results if r.status == GateStatus.WARNING)
        self.critical_failures = [
            r.gate_name for r in self.results 
            if r.status == GateStatus.FAIL and r.details.get("is_critical", False)
        ]
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if not self.results:
            return 0.0
        return self.pass_count / len(self.results)
    
    @property
    def is_promotable(self) -> bool:
        """Check if model is promotable (no critical failures)."""
        return len(self.critical_failures) == 0 and self.overall_status != GateStatus.FAIL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat(),
            "overall_status": self.overall_status.value,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "warning_count": self.warning_count,
            "pass_rate": self.pass_rate,
            "is_promotable": self.is_promotable,
            "critical_failures": self.critical_failures,
            "results": [r.to_dict() for r in self.results],
        }


class EventGateChecker:
    """Checks event-level gates for CNN-Transformer outputs."""
    
    # Event to criticality tier mapping
    EVENT_TIERS: Dict[str, CriticalityTier] = {
        # Tier 1: Critical (>= 0.50 recall required)
        "shower_day": CriticalityTier.TIER_1,
        "home_empty_false_empty_rate": CriticalityTier.TIER_1,
        "sleep_duration": CriticalityTier.TIER_1,
        
        # Tier 2: Important (>= 0.35 recall required)
        "bathroom_use": CriticalityTier.TIER_2,
        "kitchen_use": CriticalityTier.TIER_2,
        "livingroom_active": CriticalityTier.TIER_2,
        
        # Tier 3: Standard (>= 0.20 recall required)
        "out_time": CriticalityTier.TIER_3,
    }
    
    def __init__(self, thresholds: Optional[EventGateThresholds] = None):
        self.thresholds = thresholds or EventGateThresholds()
        self.thresholds.validate()
    
    def check_all_gates(
        self,
        metrics: Dict[str, Any],
        target_date: date,
    ) -> EventGateReport:
        """
        Check all event-level gates.
        
        Args:
            metrics: Dictionary of computed metrics
            target_date: Date being evaluated
            
        Returns:
            EventGateReport with all gate results
        """
        results = []
        
        # Check home-empty safety gates
        results.extend(self._check_home_empty_gates(metrics))
        
        # Check unknown rate gates
        results.extend(self._check_unknown_rate_gates(metrics))
        
        # Check criticality tier gates
        results.extend(self._check_tier_gates(metrics))
        
        # Check collapse detection
        results.extend(self._check_collapse_gates(metrics))
        
        # Determine overall status
        overall_status = self._determine_overall_status(results)
        
        return EventGateReport(
            date=target_date,
            overall_status=overall_status,
            results=results,
        )
    
    def _check_home_empty_gates(self, metrics: Dict[str, Any]) -> List[GateResult]:
        """Check home-empty safety gates."""
        results = []
        
        # Home-empty precision gate
        precision = metrics.get("home_empty_precision")
        if precision is not None:
            status = GateStatus.PASS if precision >= self.thresholds.home_empty_precision_min else GateStatus.FAIL
            results.append(GateResult(
                gate_name="home_empty_precision",
                status=status,
                metric_value=precision,
                threshold_value=self.thresholds.home_empty_precision_min,
                message=f"Home-empty precision: {precision:.4f} (threshold: {self.thresholds.home_empty_precision_min})",
                details={"is_critical": True, "tier": "safety"},
            ))
        
        # False-empty rate gate
        false_empty_rate = metrics.get("home_empty_false_empty_rate")
        if false_empty_rate is not None:
            status = GateStatus.PASS if false_empty_rate <= self.thresholds.home_empty_false_empty_rate_max else GateStatus.FAIL
            results.append(GateResult(
                gate_name="home_empty_false_empty_rate",
                status=status,
                metric_value=false_empty_rate,
                threshold_value=self.thresholds.home_empty_false_empty_rate_max,
                message=f"False-empty rate: {false_empty_rate:.4f} (threshold: {self.thresholds.home_empty_false_empty_rate_max})",
                details={"is_critical": True, "tier": "safety"},
            ))
        
        return results
    
    def _check_unknown_rate_gates(self, metrics: Dict[str, Any]) -> List[GateResult]:
        """Check unknown rate gates."""
        results = []
        
        # Global unknown rate
        global_unknown = metrics.get("unknown_rate_global")
        if global_unknown is not None:
            status = GateStatus.PASS if global_unknown <= self.thresholds.unknown_rate_global_max else GateStatus.FAIL
            results.append(GateResult(
                gate_name="unknown_rate_global",
                status=status,
                metric_value=global_unknown,
                threshold_value=self.thresholds.unknown_rate_global_max,
                message=f"Global unknown rate: {global_unknown:.4f}",
                details={"is_critical": False},
            ))
        
        # Per-room unknown rates
        room_unknown_rates = metrics.get("unknown_rate_per_room", {})
        for room, rate in room_unknown_rates.items():
            status = GateStatus.PASS if rate <= self.thresholds.unknown_rate_per_room_max else GateStatus.FAIL
            results.append(GateResult(
                gate_name=f"unknown_rate_{room}",
                status=status,
                metric_value=rate,
                threshold_value=self.thresholds.unknown_rate_per_room_max,
                message=f"Unknown rate for {room}: {rate:.4f}",
                details={"is_critical": False, "room": room},
            ))
        
        return results
    
    def _check_tier_gates(self, metrics: Dict[str, Any]) -> List[GateResult]:
        """Check criticality tier recall gates."""
        results = []
        
        # Get event recalls from metrics
        event_recalls = metrics.get("event_recalls", {})
        event_supports = metrics.get("event_supports", {})
        
        for event_name, recall in event_recalls.items():
            tier = self.EVENT_TIERS.get(event_name)
            if tier is None:
                continue
            support = int(event_supports.get(event_name, 0) or 0)
            
            # Get threshold for this tier
            if tier == CriticalityTier.TIER_1:
                threshold = self.thresholds.tier_1_recall_min
            elif tier == CriticalityTier.TIER_2:
                threshold = self.thresholds.tier_2_recall_min
            else:
                threshold = self.thresholds.tier_3_recall_min

            if support < self.thresholds.min_support_for_tier_gates:
                results.append(GateResult(
                    gate_name=f"recall_{event_name}",
                    status=GateStatus.NOT_EVALUATED,
                    metric_value=recall,
                    threshold_value=threshold,
                    message=(
                        f"{event_name} support={support} below minimum "
                        f"{self.thresholds.min_support_for_tier_gates}; marked not_evaluated."
                    ),
                    details={
                        "is_critical": False,
                        "tier": tier.value,
                        "event_name": event_name,
                        "support": support,
                        "insufficient_support": True,
                        "min_support": self.thresholds.min_support_for_tier_gates,
                    },
                ))
                continue

            status = GateStatus.PASS if recall >= threshold else GateStatus.FAIL
            
            results.append(GateResult(
                gate_name=f"recall_{event_name}",
                status=status,
                metric_value=recall,
                threshold_value=threshold,
                message=f"{event_name} recall: {recall:.4f} (Tier-{tier.value}, threshold: {threshold})",
                details={
                    "is_critical": tier == CriticalityTier.TIER_1,
                    "tier": tier.value,
                    "event_name": event_name,
                    "support": support,
                },
            ))
        
        return results
    
    def _check_collapse_gates(self, metrics: Dict[str, Any]) -> List[GateResult]:
        """Check for critical collapse (recall <= 0.02 with support >= 30)."""
        results = []
        
        event_recalls = metrics.get("event_recalls", {})
        event_supports = metrics.get("event_supports", {})
        
        for event_name, recall in event_recalls.items():
            support = event_supports.get(event_name, 0)
            
            # Check for collapse condition
            is_collapsed = (
                recall <= self.thresholds.collapse_recall_threshold and
                support >= self.thresholds.collapse_support_threshold
            )
            
            if is_collapsed:
                results.append(GateResult(
                    gate_name=f"collapse_{event_name}",
                    status=GateStatus.FAIL,
                    metric_value=recall,
                    threshold_value=self.thresholds.collapse_recall_threshold,
                    message=f"CRITICAL COLLAPSE: {event_name} recall={recall:.4f} with support={support}",
                    details={
                        "is_critical": True,
                        "event_name": event_name,
                        "recall": recall,
                        "support": support,
                    },
                ))
        
        return results
    
    def _determine_overall_status(self, results: List[GateResult]) -> GateStatus:
        """Determine overall gate status from individual results."""
        if not results:
            return GateStatus.NOT_EVALUATED
        
        # Any critical failure = overall fail
        if any(r.status == GateStatus.FAIL and r.details.get("is_critical") for r in results):
            return GateStatus.FAIL
        
        # Any failure = warning (non-critical)
        if any(r.status == GateStatus.FAIL for r in results):
            return GateStatus.WARNING
        
        # Any warning = warning
        if any(r.status == GateStatus.WARNING for r in results):
            return GateStatus.WARNING
        
        return GateStatus.PASS
    
    def check_single_event(
        self,
        event_name: str,
        recall: float,
        support: int = 0,
    ) -> GateResult:
        """
        Check a single event against its tier threshold.
        
        Args:
            event_name: Name of the event
            recall: Recall value for the event
            support: Number of samples for the event
            
        Returns:
            GateResult for this event
        """
        tier = self.EVENT_TIERS.get(event_name)
        
        if tier is None:
            return GateResult(
                gate_name=f"recall_{event_name}",
                status=GateStatus.NOT_EVALUATED,
                metric_value=recall,
                message=f"No tier defined for {event_name}",
            )
        
        # Get threshold
        if tier == CriticalityTier.TIER_1:
            threshold = self.thresholds.tier_1_recall_min
        elif tier == CriticalityTier.TIER_2:
            threshold = self.thresholds.tier_2_recall_min
        else:
            threshold = self.thresholds.tier_3_recall_min
        
        # Check for collapse
        is_collapsed = (
            recall <= self.thresholds.collapse_recall_threshold and
            support >= self.thresholds.collapse_support_threshold
        )
        
        if is_collapsed:
            return GateResult(
                gate_name=f"collapse_{event_name}",
                status=GateStatus.FAIL,
                metric_value=recall,
                threshold_value=self.thresholds.collapse_recall_threshold,
                message=f"CRITICAL COLLAPSE: {event_name} recall={recall:.4f}",
                details={"is_critical": True, "tier": tier.value},
            )
        
        if support < self.thresholds.min_support_for_tier_gates:
            return GateResult(
                gate_name=f"recall_{event_name}",
                status=GateStatus.NOT_EVALUATED,
                metric_value=recall,
                threshold_value=threshold,
                message=(
                    f"{event_name} support={support} below minimum "
                    f"{self.thresholds.min_support_for_tier_gates}; marked not_evaluated."
                ),
                details={
                    "is_critical": False,
                    "tier": tier.value,
                    "support": support,
                    "insufficient_support": True,
                },
            )

        # Check recall threshold
        status = GateStatus.PASS if recall >= threshold else GateStatus.FAIL
        
        return GateResult(
            gate_name=f"recall_{event_name}",
            status=status,
            metric_value=recall,
            threshold_value=threshold,
            message=f"{event_name} recall: {recall:.4f} (Tier-{tier.value})",
            details={"is_critical": tier == CriticalityTier.TIER_1, "tier": tier.value},
        )


def create_default_gate_checker() -> EventGateChecker:
    """Create a gate checker with default thresholds."""
    thresholds = EventGateThresholds()
    return EventGateChecker(thresholds)


def check_promotion_eligibility(
    metrics: Dict[str, Any],
    target_date: date,
    custom_thresholds: Optional[EventGateThresholds] = None,
) -> Tuple[bool, EventGateReport]:
    """
    Check if model is eligible for promotion.
    
    Args:
        metrics: Dictionary of computed metrics
        target_date: Date being evaluated
        custom_thresholds: Optional custom thresholds
        
    Returns:
        Tuple of (is_promotable, gate_report)
    """
    checker = EventGateChecker(custom_thresholds)
    report = checker.check_all_gates(metrics, target_date)
    
    return report.is_promotable, report
