"""
Timeline Gates Module

Tiered gate checking for timeline-quality metrics.
Extends event_gates.py with timeline-specific checks.

Part of WS-4: Timeline Metrics + Tiered Gates
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ml.timeline_metrics import TimelineMetrics


class TimelineGateStatus(Enum):
    """Status of timeline gate check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class TimelineGateThresholds:
    """Thresholds for timeline quality gates."""
    
    # Safety gates (must pass all split-seed cells)
    tier_1_recall_floor: float = 0.50
    tier_2_recall_floor: float = 0.35
    tier_3_recall_floor: float = 0.20
    home_empty_precision_min: float = 0.95
    false_empty_rate_max: float = 0.05
    
    # Timeline quality gates
    fragmentation_improvement_threshold: float = 0.20  # 20% relative improvement
    fragmentation_rate_max_without_baseline: float = 0.50
    segment_duration_mae_max: float = 120.0  # minutes
    segment_start_mae_max: float = 60.0  # minutes
    segment_end_mae_max: float = 60.0  # minutes
    
    # Variance/stability
    macro_f1_std_max: float = 0.05
    
    # Unknown rate governance
    unknown_rate_global_max: float = 0.15
    unknown_rate_per_room_max: float = 0.20
    unknown_rate_per_label_max: float = 0.30
    
    # Robustness criteria
    min_split_seed_pass_ratio: float = 0.80  # 80% of split-seed cells must pass


@dataclass
class TimelineGateResult:
    """Result of a timeline gate check."""
    
    gate_name: str
    status: TimelineGateStatus
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    message: str = ""
    room: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_pass(self) -> bool:
        return self.status == TimelineGateStatus.PASS


@dataclass
class UnknownRateReport:
    """Unknown rate report per room and label."""
    
    global_rate: float = 0.0
    per_room: Dict[str, float] = field(default_factory=dict)
    per_label: Dict[str, float] = field(default_factory=dict)
    breaches: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'global_rate': round(self.global_rate, 4),
            'per_room': {k: round(v, 4) for k, v in self.per_room.items()},
            'per_label': {k: round(v, 4) for k, v in self.per_label.items()},
            'breaches': self.breaches,
        }


class TimelineGateChecker:
    """Checker for timeline quality gates."""
    
    def __init__(self, thresholds: Optional[TimelineGateThresholds] = None):
        self.thresholds = thresholds or TimelineGateThresholds()
    
    def check_timeline_metrics(
        self,
        room_metrics: Dict[str, TimelineMetrics],
        baseline_metrics: Optional[Dict[str, TimelineMetrics]] = None,
    ) -> List[TimelineGateResult]:
        """
        Check timeline metrics against thresholds.
        
        Args:
            room_metrics: Dict mapping room names to TimelineMetrics
            baseline_metrics: Optional baseline for comparison
            
        Returns:
            List of gate results
        """
        results = []
        
        for room, metrics in room_metrics.items():
            # Fragmentation gate
            frag_result = self._check_fragmentation(room, metrics, baseline_metrics)
            if frag_result:
                results.append(frag_result)
            
            # Duration MAE gate
            duration_result = self._check_duration_mae(room, metrics)
            if duration_result:
                results.append(duration_result)
            
            # Start/End MAE gates
            start_result = self._check_start_mae(room, metrics)
            if start_result:
                results.append(start_result)
            
            end_result = self._check_end_mae(room, metrics)
            if end_result:
                results.append(end_result)
        
        return results
    
    def _check_fragmentation(
        self,
        room: str,
        metrics: TimelineMetrics,
        baseline_metrics: Optional[Dict[str, TimelineMetrics]] = None,
    ) -> Optional[TimelineGateResult]:
        """Check fragmentation rate improvement."""
        current_frag = metrics.fragmentation_rate
        
        if baseline_metrics and room in baseline_metrics:
            baseline_frag = baseline_metrics[room].fragmentation_rate
            if baseline_frag > 0:
                improvement = (baseline_frag - current_frag) / baseline_frag
                passed = improvement >= self.thresholds.fragmentation_improvement_threshold
                
                return TimelineGateResult(
                    gate_name=f"fragmentation_improvement_{room}",
                    status=TimelineGateStatus.PASS if passed else TimelineGateStatus.FAIL,
                    metric_value=improvement,
                    threshold_value=self.thresholds.fragmentation_improvement_threshold,
                    message=f"Fragmentation improvement: {improvement:.2%} (threshold: {self.thresholds.fragmentation_improvement_threshold:.2%})",
                    room=room,
                    details={
                        'current_fragmentation': current_frag,
                        'baseline_fragmentation': baseline_frag,
                        'improvement_ratio': improvement,
                    },
                )
        
        # No baseline - just check absolute
        passed = current_frag < float(self.thresholds.fragmentation_rate_max_without_baseline)
        
        return TimelineGateResult(
            gate_name=f"fragmentation_rate_{room}",
            status=TimelineGateStatus.PASS if passed else TimelineGateStatus.FAIL,
            metric_value=current_frag,
            threshold_value=float(self.thresholds.fragmentation_rate_max_without_baseline),
            message=f"Fragmentation rate: {current_frag:.2%}",
            room=room,
        )
    
    def _check_duration_mae(
        self,
        room: str,
        metrics: TimelineMetrics,
    ) -> Optional[TimelineGateResult]:
        """Check segment duration MAE."""
        if metrics.segment_duration_mae_minutes == float('inf'):
            return None
        
        passed = metrics.segment_duration_mae_minutes <= self.thresholds.segment_duration_mae_max
        
        return TimelineGateResult(
            gate_name=f"duration_mae_{room}",
            status=TimelineGateStatus.PASS if passed else TimelineGateStatus.FAIL,
            metric_value=metrics.segment_duration_mae_minutes,
            threshold_value=self.thresholds.segment_duration_mae_max,
            message=f"Duration MAE: {metrics.segment_duration_mae_minutes:.1f} min (threshold: {self.thresholds.segment_duration_mae_max:.1f})",
            room=room,
        )
    
    def _check_start_mae(
        self,
        room: str,
        metrics: TimelineMetrics,
    ) -> Optional[TimelineGateResult]:
        """Check segment start MAE."""
        if metrics.segment_start_mae_minutes == float('inf'):
            return None
        
        passed = metrics.segment_start_mae_minutes <= self.thresholds.segment_start_mae_max
        
        return TimelineGateResult(
            gate_name=f"start_mae_{room}",
            status=TimelineGateStatus.PASS if passed else TimelineGateStatus.FAIL,
            metric_value=metrics.segment_start_mae_minutes,
            threshold_value=self.thresholds.segment_start_mae_max,
            message=f"Start MAE: {metrics.segment_start_mae_minutes:.1f} min (threshold: {self.thresholds.segment_start_mae_max:.1f})",
            room=room,
        )
    
    def _check_end_mae(
        self,
        room: str,
        metrics: TimelineMetrics,
    ) -> Optional[TimelineGateResult]:
        """Check segment end MAE."""
        if metrics.segment_end_mae_minutes == float('inf'):
            return None
        
        passed = metrics.segment_end_mae_minutes <= self.thresholds.segment_end_mae_max
        
        return TimelineGateResult(
            gate_name=f"end_mae_{room}",
            status=TimelineGateStatus.PASS if passed else TimelineGateStatus.FAIL,
            metric_value=metrics.segment_end_mae_minutes,
            threshold_value=self.thresholds.segment_end_mae_max,
            message=f"End MAE: {metrics.segment_end_mae_minutes:.1f} min (threshold: {self.thresholds.segment_end_mae_max:.1f})",
            room=room,
        )
    
    def check_unknown_rates(
        self,
        predictions: List[Dict[str, Any]],
        room_key: str = "room",
        label_key: str = "predicted_label",
        true_label_key: str = "true_label",
    ) -> UnknownRateReport:
        """
        Check unknown rates per room and label.
        
        Args:
            predictions: List of prediction dicts
            room_key: Key for room name in prediction dict
            label_key: Key for label in prediction dict
            
        Returns:
            UnknownRateReport with breaches identified
        """
        report = UnknownRateReport()
        
        if not predictions:
            return report
        
        # Global rate
        unknown_count = sum(1 for p in predictions if p.get(label_key) == 'unknown')
        report.global_rate = unknown_count / len(predictions)
        
        # Per-room rates
        rooms = set(p.get(room_key, 'unknown') for p in predictions)
        for room in rooms:
            room_preds = [p for p in predictions if p.get(room_key) == room]
            room_unknown = sum(1 for p in room_preds if p.get(label_key) == 'unknown')
            rate = room_unknown / len(room_preds) if room_preds else 0.0
            report.per_room[room] = rate
            
            if rate > self.thresholds.unknown_rate_per_room_max:
                report.breaches.append(
                    f"Room {room}: unknown rate {rate:.2%} > threshold {self.thresholds.unknown_rate_per_room_max:.2%}"
                )
        
        # Per-label unknown rates:
        # For each true label, how often the prediction is unknown.
        has_true_labels = any(true_label_key in p for p in predictions)
        if has_true_labels:
            true_labels = set(
                str(p.get(true_label_key))
                for p in predictions
                if p.get(true_label_key) is not None and str(p.get(true_label_key)) != 'unknown'
            )
            for true_label in sorted(true_labels):
                label_rows = [p for p in predictions if str(p.get(true_label_key)) == true_label]
                if not label_rows:
                    continue
                unknown_pred_count = sum(1 for p in label_rows if p.get(label_key) == 'unknown')
                rate = unknown_pred_count / len(label_rows)
                report.per_label[true_label] = rate
                if rate > self.thresholds.unknown_rate_per_label_max:
                    report.breaches.append(
                        f"Label {true_label}: unknown rate {rate:.2%} > threshold {self.thresholds.unknown_rate_per_label_max:.2%}"
                    )
        
        # Check global threshold
        if report.global_rate > self.thresholds.unknown_rate_global_max:
            report.breaches.append(
                f"Global: unknown rate {report.global_rate:.2%} > threshold {self.thresholds.unknown_rate_global_max:.2%}"
            )
        
        return report
    
    def compute_pass_rate(
        self,
        results: List[TimelineGateResult],
    ) -> float:
        """Compute pass rate across gate results."""
        if not results:
            return 1.0
        passed = sum(1 for r in results if r.is_pass)
        return passed / len(results)
    
    def is_promotable(
        self,
        results: List[TimelineGateResult],
        unknown_report: UnknownRateReport,
    ) -> Tuple[bool, List[str]]:
        """
        Determine if model is promotable based on gates.
        
        Returns:
            Tuple of (is_promotable, failed_reasons)
        """
        failed_reasons = []
        
        # Check for any FAIL status
        for result in results:
            if result.status == TimelineGateStatus.FAIL:
                failed_reasons.append(
                    f"{result.gate_name}: {result.metric_value:.4f} vs threshold {result.threshold_value}"
                )
        
        # Check unknown rate breaches
        for breach in unknown_report.breaches:
            failed_reasons.append(f"unknown_rate_breach: {breach}")
        
        is_promotable = len(failed_reasons) == 0
        
        return is_promotable, failed_reasons


def create_default_timeline_checker() -> TimelineGateChecker:
    """Create a timeline gate checker with default thresholds."""
    return TimelineGateChecker(TimelineGateThresholds())
