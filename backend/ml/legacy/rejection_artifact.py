"""
Item 11: Unified "Why Rejected" Artifact

Produces a comprehensive, human-readable rejection summary that enables
external reviewers to understand rejection reasons in <5 minutes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    """Return timezone-aware UTC timestamp as ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


class RejectionCategory(Enum):
    """Categories of rejection reasons."""
    COVERAGE = "coverage"  # Data coverage issues
    VIABILITY = "viability"  # Data viability issues
    STATISTICAL_VALIDITY = "statistical_validity"  # Low support, etc.
    WALK_FORWARD = "walk_forward"  # Fold generation issues
    CLASS_COVERAGE = "class_coverage"  # Missing classes
    GLOBAL_GATE = "global_gate"  # Room threshold failures
    TEMPORAL_SEMANTICS = "temporal_semantics"  # Calibration/validation issues
    POST_GAP_RETENTION = "post_gap_retention"  # Data fragmentation
    DUPLICATE_RESOLUTION = "duplicate_resolution"  # Unresolved duplicates
    SEQUENCE_ALIGNMENT = "sequence_alignment"  # Label misalignment
    UNKNOWN = "unknown"


class Severity(Enum):
    """Severity levels for rejection reasons."""
    CRITICAL = "critical"  # Blocking, must fix
    HIGH = "high"  # Significant impact
    MEDIUM = "medium"  # Moderate impact
    LOW = "low"  # Advisory only
    INFO = "info"  # Informational


@dataclass
class RejectionReason:
    """Individual rejection reason with context."""
    category: RejectionCategory
    severity: Severity
    code: str  # Machine-readable code
    message: str  # Human-readable message
    room: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "room": self.room,
            "metric_value": self.metric_value,
            "threshold_value": self.threshold_value,
            "recommendation": self.recommendation,
        }


@dataclass
class RoomRejectionSummary:
    """Rejection summary for a specific room."""
    room_name: str
    passed: bool
    reasons: List[RejectionReason] = field(default_factory=list)
    metrics_at_failure: Dict[str, Any] = field(default_factory=dict)
    actionable_next_step: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_name": self.room_name,
            "passed": self.passed,
            "reasons": [r.to_dict() for r in self.reasons],
            "metrics_at_failure": self.metrics_at_failure,
            "actionable_next_step": self.actionable_next_step,
        }


@dataclass
class RunRejectionSummary:
    """
    Comprehensive rejection summary for an entire training run.
    
    This is the main artifact that explains why a run was rejected.
    """
    # Run identification
    run_id: str
    timestamp: str
    elder_id: str
    
    # Overall status
    overall_passed: bool
    overall_status: str  # 'promoted', 'rejected', 'partial', 'failed'
    
    # Categorized reasons (ordered by priority)
    coverage_reasons: List[RejectionReason] = field(default_factory=list)
    viability_reasons: List[RejectionReason] = field(default_factory=list)
    statistical_validity_reasons: List[RejectionReason] = field(default_factory=list)
    walk_forward_reasons: List[RejectionReason] = field(default_factory=list)
    class_coverage_reasons: List[RejectionReason] = field(default_factory=list)
    global_gate_reasons: List[RejectionReason] = field(default_factory=list)
    temporal_semantics_reasons: List[RejectionReason] = field(default_factory=list)
    post_gap_retention_reasons: List[RejectionReason] = field(default_factory=list)
    duplicate_resolution_reasons: List[RejectionReason] = field(default_factory=list)
    sequence_alignment_reasons: List[RejectionReason] = field(default_factory=list)
    
    # Per-room summaries
    room_summaries: Dict[str, RoomRejectionSummary] = field(default_factory=dict)
    
    # Executive summary
    executive_summary: str = ""
    top_priority_fix: str = ""
    estimated_time_to_fix: str = ""
    
    # Metadata
    policy_hash: str = ""
    data_fingerprint: str = ""
    code_version: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "elder_id": self.elder_id,
            "overall_passed": self.overall_passed,
            "overall_status": self.overall_status,
            "coverage_reasons": [r.to_dict() for r in self.coverage_reasons],
            "viability_reasons": [r.to_dict() for r in self.viability_reasons],
            "statistical_validity_reasons": [r.to_dict() for r in self.statistical_validity_reasons],
            "walk_forward_reasons": [r.to_dict() for r in self.walk_forward_reasons],
            "class_coverage_reasons": [r.to_dict() for r in self.class_coverage_reasons],
            "global_gate_reasons": [r.to_dict() for r in self.global_gate_reasons],
            "temporal_semantics_reasons": [r.to_dict() for r in self.temporal_semantics_reasons],
            "post_gap_retention_reasons": [r.to_dict() for r in self.post_gap_retention_reasons],
            "duplicate_resolution_reasons": [r.to_dict() for r in self.duplicate_resolution_reasons],
            "sequence_alignment_reasons": [r.to_dict() for r in self.sequence_alignment_reasons],
            "room_summaries": {k: v.to_dict() for k, v in self.room_summaries.items()},
            "executive_summary": self.executive_summary,
            "top_priority_fix": self.top_priority_fix,
            "estimated_time_to_fix": self.estimated_time_to_fix,
            "policy_hash": self.policy_hash,
            "data_fingerprint": self.data_fingerprint,
            "code_version": self.code_version,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save artifact to file."""
        filepath = Path(filepath)
        filepath.write_text(self.to_json())
        logger.info(f"Rejection artifact saved to {filepath}")


class RejectionArtifactBuilder:
    """
    Builder for creating comprehensive rejection artifacts.
    
    Usage:
        builder = RejectionArtifactBuilder(run_id, elder_id)
        builder.add_coverage_failure(room, reason, metric, threshold)
        builder.add_viability_failure(room, reason)
        ...
        artifact = builder.build()
        artifact.save("/path/to/run_rejection_summary.json")
    """
    
    # Priority order for categorization
    CATEGORY_PRIORITY = [
        RejectionCategory.COVERAGE,
        RejectionCategory.VIABILITY,
        RejectionCategory.POST_GAP_RETENTION,
        RejectionCategory.SEQUENCE_ALIGNMENT,
        RejectionCategory.DUPLICATE_RESOLUTION,
        RejectionCategory.WALK_FORWARD,
        RejectionCategory.CLASS_COVERAGE,
        RejectionCategory.STATISTICAL_VALIDITY,
        RejectionCategory.TEMPORAL_SEMANTICS,
        RejectionCategory.GLOBAL_GATE,
    ]
    
    def __init__(
        self,
        run_id: str,
        elder_id: str,
        policy_hash: str = "",
        data_fingerprint: str = "",
        code_version: str = "",
    ):
        self.run_id = run_id
        self.elder_id = elder_id
        self.policy_hash = policy_hash
        self.data_fingerprint = data_fingerprint
        self.code_version = code_version
        
        self.summary = RunRejectionSummary(
            run_id=run_id,
            timestamp=_utc_now_iso(),
            elder_id=elder_id,
            overall_passed=True,
            overall_status="pending",
            policy_hash=policy_hash,
            data_fingerprint=data_fingerprint,
            code_version=code_version,
        )
        
        self._room_reasons: Dict[str, List[RejectionReason]] = {}
    
    def add_reason(
        self,
        category: RejectionCategory,
        severity: Severity,
        code: str,
        message: str,
        room: Optional[str] = None,
        metric_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        recommendation: str = "",
    ) -> "RejectionArtifactBuilder":
        """Add a rejection reason."""
        reason = RejectionReason(
            category=category,
            severity=severity,
            code=code,
            message=message,
            room=room,
            metric_value=metric_value,
            threshold_value=threshold_value,
            recommendation=recommendation,
        )
        
        # Add to category list
        category_list = self._get_category_list(category)
        category_list.append(reason)
        
        # Add to room-specific tracking
        if room:
            if room not in self._room_reasons:
                self._room_reasons[room] = []
            self._room_reasons[room].append(reason)
        
        if severity in (Severity.CRITICAL, Severity.HIGH):
            self.summary.overall_passed = False
        
        return self
    
    def add_coverage_failure(
        self,
        room: str,
        observed_days: int,
        required_days: int,
        recommendation: str = "Collect more training data",
    ) -> "RejectionArtifactBuilder":
        """Add a coverage contract failure."""
        return self.add_reason(
            category=RejectionCategory.COVERAGE,
            severity=Severity.CRITICAL,
            code="insufficient_observed_days",
            message=f"Room '{room}': Insufficient observed days ({observed_days} < {required_days})",
            room=room,
            metric_value=float(observed_days),
            threshold_value=float(required_days),
            recommendation=recommendation,
        )
    
    def add_viability_failure(
        self,
        room: str,
        reason: str,
        recommendation: str = "Check data quality and gap handling",
    ) -> "RejectionArtifactBuilder":
        """Add a data viability failure."""
        return self.add_reason(
            category=RejectionCategory.VIABILITY,
            severity=Severity.CRITICAL,
            code="data_viability_failed",
            message=f"Room '{room}': {reason}",
            room=room,
            recommendation=recommendation,
        )
    
    def add_statistical_validity_failure(
        self,
        room: str,
        metric_name: str,
        metric_value: float,
        threshold_value: float,
        is_fallback: bool = False,
        recommendation: str = "Increase data collection or adjust thresholds",
    ) -> "RejectionArtifactBuilder":
        """Add a statistical validity failure."""
        code = "statistical_validity_fallback" if is_fallback else "insufficient_support"
        message = f"Room '{room}': {metric_name} insufficient ({metric_value:.2f} < {threshold_value:.2f})"
        if is_fallback:
            message += " (using fallback metrics)"
        
        return self.add_reason(
            category=RejectionCategory.STATISTICAL_VALIDITY,
            severity=Severity.HIGH,
            code=code,
            message=message,
            room=room,
            metric_value=metric_value,
            threshold_value=threshold_value,
            recommendation=recommendation,
        )
    
    def add_walk_forward_failure(
        self,
        room: str,
        observed_days: int,
        required_days: int,
        recommendation: str = "Need more historical data for walk-forward validation",
    ) -> "RejectionArtifactBuilder":
        """Add a walk-forward fold generation failure."""
        return self.add_reason(
            category=RejectionCategory.WALK_FORWARD,
            severity=Severity.CRITICAL,
            code="walk_forward_unavailable",
            message=f"Room '{room}': Cannot generate walk-forward folds ({observed_days} < {required_days} days)",
            room=room,
            metric_value=float(observed_days),
            threshold_value=float(required_days),
            recommendation=recommendation,
        )
    
    def add_class_coverage_failure(
        self,
        room: str,
        missing_classes: List[int],
        split: str = "training",
        recommendation: str = "Ensure all activity classes are represented in data",
    ) -> "RejectionArtifactBuilder":
        """Add a class coverage failure."""
        return self.add_reason(
            category=RejectionCategory.CLASS_COVERAGE,
            severity=Severity.HIGH,
            code=f"missing_classes_in_{split}",
            message=f"Room '{room}': Classes {missing_classes} missing from {split} split",
            room=room,
            recommendation=recommendation,
        )
    
    def add_global_gate_failure(
        self,
        room: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
        recommendation: str = "Model performance below required threshold",
    ) -> "RejectionArtifactBuilder":
        """Add a global release gate failure."""
        return self.add_reason(
            category=RejectionCategory.GLOBAL_GATE,
            severity=Severity.HIGH,
            code="room_threshold_failed",
            message=f"Room '{room}': {metric_name} ({metric_value:.3f}) below threshold ({threshold:.3f})",
            room=room,
            metric_value=metric_value,
            threshold_value=threshold,
            recommendation=recommendation,
        )
    
    def add_post_gap_retention_failure(
        self,
        room: str,
        retained_ratio: float,
        required_ratio: float,
        segment_count: int,
        recommendation: str = "Data too fragmented after gap handling. Check sensor continuity.",
    ) -> "RejectionArtifactBuilder":
        """Add a post-gap retention failure."""
        return self.add_reason(
            category=RejectionCategory.POST_GAP_RETENTION,
            severity=Severity.HIGH,
            code="excessive_fragmentation",
            message=f"Room '{room}': Retained ratio {retained_ratio:.1%} < {required_ratio:.1%}, {segment_count} segments",
            room=room,
            metric_value=retained_ratio,
            threshold_value=required_ratio,
            recommendation=recommendation,
        )
    
    def add_room_summary(
        self,
        room: str,
        passed: bool,
        metrics: Dict[str, Any],
        actionable_step: str,
    ) -> "RejectionArtifactBuilder":
        """Add per-room summary."""
        room_summary = RoomRejectionSummary(
            room_name=room,
            passed=passed,
            reasons=self._room_reasons.get(room, []),
            metrics_at_failure=metrics if not passed else {},
            actionable_next_step=actionable_step,
        )
        self.summary.room_summaries[room] = room_summary
        return self
    
    def build(self) -> RunRejectionSummary:
        """Build and return the final rejection artifact."""
        # Generate executive summary
        self._generate_executive_summary()
        
        # Determine overall status
        if self.summary.overall_passed:
            self.summary.overall_status = "promoted"
        elif any(r.severity == Severity.CRITICAL for r in self._get_all_reasons()):
            self.summary.overall_status = "rejected"
        else:
            self.summary.overall_status = "partial"
        
        return self.summary
    
    def _get_category_list(self, category: RejectionCategory) -> List[RejectionReason]:
        """Get the list for a specific category."""
        mapping = {
            RejectionCategory.COVERAGE: self.summary.coverage_reasons,
            RejectionCategory.VIABILITY: self.summary.viability_reasons,
            RejectionCategory.STATISTICAL_VALIDITY: self.summary.statistical_validity_reasons,
            RejectionCategory.WALK_FORWARD: self.summary.walk_forward_reasons,
            RejectionCategory.CLASS_COVERAGE: self.summary.class_coverage_reasons,
            RejectionCategory.GLOBAL_GATE: self.summary.global_gate_reasons,
            RejectionCategory.TEMPORAL_SEMANTICS: self.summary.temporal_semantics_reasons,
            RejectionCategory.POST_GAP_RETENTION: self.summary.post_gap_retention_reasons,
            RejectionCategory.DUPLICATE_RESOLUTION: self.summary.duplicate_resolution_reasons,
            RejectionCategory.SEQUENCE_ALIGNMENT: self.summary.sequence_alignment_reasons,
        }
        return mapping.get(category, [])
    
    def _get_all_reasons(self) -> List[RejectionReason]:
        """Get all reasons across all categories."""
        all_reasons = []
        for category in self.CATEGORY_PRIORITY:
            all_reasons.extend(self._get_category_list(category))
        return all_reasons
    
    def _generate_executive_summary(self) -> None:
        """Generate human-readable executive summary."""
        all_reasons = self._get_all_reasons()
        
        if not all_reasons:
            self.summary.executive_summary = "Training run completed successfully with no blocking issues."
            self.summary.top_priority_fix = "None - ready for promotion"
            self.summary.estimated_time_to_fix = "N/A"
            return
        
        # Count by severity
        critical_count = sum(1 for r in all_reasons if r.severity == Severity.CRITICAL)
        high_count = sum(1 for r in all_reasons if r.severity == Severity.HIGH)
        
        # Group by category
        category_counts = {}
        for reason in all_reasons:
            cat = reason.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Build summary
        summary_parts = [
            f"Training run failed with {critical_count} critical and {high_count} high severity issues.",
            "",
            "Issues by category:",
        ]
        
        for cat in self.CATEGORY_PRIORITY:
            count = category_counts.get(cat.value, 0)
            if count > 0:
                summary_parts.append(f"  - {cat.value.replace('_', ' ').title()}: {count}")
        
        # Add top priority
        critical_reasons = [r for r in all_reasons if r.severity == Severity.CRITICAL]
        if critical_reasons:
            self.summary.top_priority_fix = critical_reasons[0].message
            self.summary.estimated_time_to_fix = "Varies by issue - see recommendations"
        else:
            self.summary.top_priority_fix = all_reasons[0].message if all_reasons else ""
            self.summary.estimated_time_to_fix = "1-3 days"
        
        self.summary.executive_summary = "\n".join(summary_parts)


def create_rejection_artifact(
    run_id: str,
    elder_id: str,
    gate_results: Dict[str, Any],
    output_path: Optional[str] = None,
) -> RunRejectionSummary:
    """
    Convenience function to create rejection artifact from gate results.
    
    Parameters:
    -----------
    run_id : str
        Training run ID
    elder_id : str
        Elder/resident ID
    gate_results : Dict
        Dictionary of gate evaluation results by room
    output_path : str, optional
        Path to save artifact
        
    Returns:
    --------
    RunRejectionSummary
    """
    builder = RejectionArtifactBuilder(run_id=run_id, elder_id=elder_id)
    
    # Map gate names to rejection categories
    GATE_CATEGORY_MAP = {
        "coverage": RejectionCategory.COVERAGE,
        "viability": RejectionCategory.VIABILITY,
        "statistical_validity": RejectionCategory.STATISTICAL_VALIDITY,
        "walk_forward": RejectionCategory.WALK_FORWARD,
        "class_coverage": RejectionCategory.CLASS_COVERAGE,
        "global_gate": RejectionCategory.GLOBAL_GATE,
        "temporal_semantics": RejectionCategory.TEMPORAL_SEMANTICS,
        "post_gap_retention": RejectionCategory.POST_GAP_RETENTION,
        "duplicate_resolution": RejectionCategory.DUPLICATE_RESOLUTION,
        "sequence_alignment": RejectionCategory.SEQUENCE_ALIGNMENT,
    }
    
    # Process gate results and add to builder
    for room, results in gate_results.items():
        if not results.get("passes", True):
            gate_name = results.get("gate_name", "unknown")
            reasons = results.get("reasons", [])
            
            # Map gate name to category
            clean_name = gate_name.replace("_gate", "").lower()
            category = GATE_CATEGORY_MAP.get(clean_name, RejectionCategory.UNKNOWN)
            
            for reason in reasons:
                builder.add_reason(
                    category=category,
                    severity=Severity.CRITICAL if results.get("blocking", True) else Severity.HIGH,
                    code=f"{gate_name}_failed",
                    message=f"Room '{room}': {reason}",
                    room=room,
                    recommendation="See gate documentation for remediation steps",
                )
    
    artifact = builder.build()
    
    if output_path:
        artifact.save(output_path)
    
    return artifact
