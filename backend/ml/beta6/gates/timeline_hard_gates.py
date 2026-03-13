"""Authoritative Beta 6.2 timeline hard-gate entrypoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from ..serving.capability_profiles import CapabilityProfile
from ..contracts.decisions import ReasonCode


BETA62_SURFACE_NAME = "timeline_hard_gates"
BETA62_CANONICAL_IMPORT = "ml.beta6.gates.timeline_hard_gates"
BETA62_COMPAT_SHIMS = ("ml.beta6.timeline_hard_gates",)


@dataclass(frozen=True)
class TimelineHardGateResult:
    """Result of evaluating timeline MAE + fragmentation hard gates."""

    checked: bool
    passed: bool
    reason_code: Optional[ReasonCode]
    details: Dict[str, Any] = field(default_factory=dict)


def _to_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Expected float-like metric value, got {type(value).__name__}")
    return float(value)


def evaluate_timeline_hard_gates(
    report: Mapping[str, Any],
    capability_profile: CapabilityProfile,
) -> TimelineHardGateResult:
    """
    Evaluate timeline hard gates from report payload.

    Expects `report.timeline_metrics.duration_mae_minutes` and
    `report.timeline_metrics.fragmentation_rate` when timeline checks are provided.
    """
    raw_timeline_metrics = report.get("timeline_metrics")
    if raw_timeline_metrics is None:
        return TimelineHardGateResult(checked=False, passed=True, reason_code=None, details={})
    if not isinstance(raw_timeline_metrics, Mapping):
        return TimelineHardGateResult(
            checked=True,
            passed=False,
            reason_code=ReasonCode.FAIL_TIMELINE_METRICS_MISSING,
            details={"error": "timeline_metrics must be a mapping"},
        )

    if "duration_mae_minutes" not in raw_timeline_metrics or "fragmentation_rate" not in raw_timeline_metrics:
        return TimelineHardGateResult(
            checked=True,
            passed=False,
            reason_code=ReasonCode.FAIL_TIMELINE_METRICS_MISSING,
            details={"error": "timeline_metrics missing duration_mae_minutes or fragmentation_rate"},
        )

    try:
        duration_mae_minutes = _to_float(raw_timeline_metrics["duration_mae_minutes"])
        fragmentation_rate = _to_float(raw_timeline_metrics["fragmentation_rate"])
    except TypeError as exc:
        return TimelineHardGateResult(
            checked=True,
            passed=False,
            reason_code=ReasonCode.FAIL_TIMELINE_METRICS_MISSING,
            details={"error": str(exc)},
        )

    details = {
        "duration_mae_minutes": duration_mae_minutes,
        "duration_mae_threshold_minutes": float(capability_profile.max_timeline_mae_minutes),
        "fragmentation_rate": fragmentation_rate,
        "fragmentation_rate_threshold": float(capability_profile.max_fragmentation_rate),
    }

    if duration_mae_minutes > float(capability_profile.max_timeline_mae_minutes):
        return TimelineHardGateResult(
            checked=True,
            passed=False,
            reason_code=ReasonCode.FAIL_TIMELINE_MAE,
            details=details,
        )

    if fragmentation_rate > float(capability_profile.max_fragmentation_rate):
        return TimelineHardGateResult(
            checked=True,
            passed=False,
            reason_code=ReasonCode.FAIL_TIMELINE_FRAGMENTATION,
            details=details,
        )

    return TimelineHardGateResult(
        checked=True,
        passed=True,
        reason_code=None,
        details=details,
    )
