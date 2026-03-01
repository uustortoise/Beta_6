"""Model-behavior SLO evaluation and escalation routing for Beta 6."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class SLOSeverity(str, Enum):
    OK = "ok"
    WARN = "warn"
    CRITICAL = "critical"


DEFAULT_SLO_POLICY: Dict[str, Any] = {
    "thresholds": {
        "unknown_rate": {"warn": 0.10, "critical": 0.16},
        "abstain_rate": {"warn": 0.12, "critical": 0.18},
        "occupancy_drift_abs_pp": {"warn": 0.12, "critical": 0.20},
    },
    "reason_code_ratio_caps": {
        "fail_uncertainty_unknown": {"warn": 0.08, "critical": 0.12},
        "fail_timeline_mae": {"warn": 0.05, "critical": 0.10},
        "fail_timeline_fragmentation": {"warn": 0.06, "critical": 0.11},
    },
    "routing": {
        "default_owner": "mlops_oncall",
        "metric_owners": {
            "unknown_rate": "mlops_oncall",
            "abstain_rate": "mlops_oncall",
            "occupancy_drift_abs_pp": "modeling_lead",
            "reason_code_distribution": "qa_gate_owner",
            "slo_input_contract": "ml_platform_owner",
        },
        "severity_channels": {
            "warn": "slack://beta6-model-alerts",
            "critical": "pagerduty://beta6-mlops",
        },
        "severity_eta_hours": {"warn": 24, "critical": 4},
    },
}


@dataclass(frozen=True)
class SLOAlert:
    metric: str
    severity: SLOSeverity
    observed: float
    threshold: float
    owner: str
    escalation_route: str
    remediation_eta_hours: int
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "severity": self.severity.value,
            "observed": self.observed,
            "threshold": self.threshold,
            "owner": self.owner,
            "escalation_route": self.escalation_route,
            "remediation_eta_hours": self.remediation_eta_hours,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SLOEvaluation:
    status: SLOSeverity
    alerts: List[SLOAlert] = field(default_factory=list)
    observed_metrics: Dict[str, float] = field(default_factory=dict)
    report_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "report_date": self.report_date,
            "observed_metrics": dict(self.observed_metrics),
            "alerts": [alert.to_dict() for alert in self.alerts],
            "alert_count": len(self.alerts),
            "critical_count": sum(1 for alert in self.alerts if alert.severity == SLOSeverity.CRITICAL),
            "warn_count": sum(1 for alert in self.alerts if alert.severity == SLOSeverity.WARN),
        }


def _to_rate(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    normalized = float(value)
    if normalized < 0.0 or normalized > 1.0:
        raise ValueError(f"{field_name} must be in [0,1]")
    return normalized


def _resolve_severity(observed: float, warn_threshold: float, critical_threshold: float) -> SLOSeverity:
    if observed >= critical_threshold:
        return SLOSeverity.CRITICAL
    if observed >= warn_threshold:
        return SLOSeverity.WARN
    return SLOSeverity.OK


def _build_alert(
    *,
    metric: str,
    severity: SLOSeverity,
    observed: float,
    threshold: float,
    detail: str,
    routing: Mapping[str, Any],
) -> SLOAlert:
    owners = routing.get("metric_owners", {})
    owner = str(owners.get(metric, routing.get("default_owner", "mlops_oncall")))
    channels = routing.get("severity_channels", {})
    etas = routing.get("severity_eta_hours", {})
    return SLOAlert(
        metric=metric,
        severity=severity,
        observed=observed,
        threshold=threshold,
        owner=owner,
        escalation_route=str(channels.get(severity.value, "slack://beta6-model-alerts")),
        remediation_eta_hours=int(etas.get(severity.value, 24)),
        detail=detail,
    )


def evaluate_model_behavior_slo(
    daily_metrics: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] = DEFAULT_SLO_POLICY,
) -> SLOEvaluation:
    """Evaluate Beta 6 model-behavior SLOs and produce routed alerts."""
    thresholds = dict(policy.get("thresholds", {}))
    reason_caps = dict(policy.get("reason_code_ratio_caps", {}))
    routing = dict(policy.get("routing", {}))
    alerts: List[SLOAlert] = []
    observed_metrics: Dict[str, float] = {}

    def required_rate(name: str) -> Optional[float]:
        raw = daily_metrics.get(name)
        if raw is None:
            alerts.append(
                _build_alert(
                    metric="slo_input_contract",
                    severity=SLOSeverity.CRITICAL,
                    observed=1.0,
                    threshold=0.0,
                    detail=f"missing required metric: {name}",
                    routing=routing,
                )
            )
            return None
        try:
            rate = _to_rate(raw, field_name=name)
        except ValueError as exc:
            alerts.append(
                _build_alert(
                    metric="slo_input_contract",
                    severity=SLOSeverity.CRITICAL,
                    observed=1.0,
                    threshold=0.0,
                    detail=str(exc),
                    routing=routing,
                )
            )
            return None
        observed_metrics[name] = rate
        return rate

    unknown_rate = required_rate("unknown_rate")
    abstain_rate = required_rate("abstain_rate")
    occupancy_rate = required_rate("occupancy_rate")
    baseline_occupancy_rate = required_rate("baseline_occupancy_rate")

    for metric_name in ("unknown_rate", "abstain_rate"):
        observed = observed_metrics.get(metric_name)
        if observed is None:
            continue
        metric_threshold = thresholds.get(metric_name, {})
        warn = float(metric_threshold.get("warn", 1.0))
        critical = float(metric_threshold.get("critical", 1.0))
        severity = _resolve_severity(observed, warn, critical)
        if severity != SLOSeverity.OK:
            alerts.append(
                _build_alert(
                    metric=metric_name,
                    severity=severity,
                    observed=observed,
                    threshold=critical if severity == SLOSeverity.CRITICAL else warn,
                    detail=f"{metric_name} exceeded {severity.value} threshold",
                    routing=routing,
                )
            )

    if occupancy_rate is not None and baseline_occupancy_rate is not None:
        occupancy_drift = abs(occupancy_rate - baseline_occupancy_rate)
        observed_metrics["occupancy_drift_abs_pp"] = occupancy_drift
        drift_threshold = thresholds.get("occupancy_drift_abs_pp", {})
        drift_warn = float(drift_threshold.get("warn", 1.0))
        drift_critical = float(drift_threshold.get("critical", 1.0))
        drift_severity = _resolve_severity(occupancy_drift, drift_warn, drift_critical)
        if drift_severity != SLOSeverity.OK:
            alerts.append(
                _build_alert(
                    metric="occupancy_drift_abs_pp",
                    severity=drift_severity,
                    observed=occupancy_drift,
                    threshold=drift_critical if drift_severity == SLOSeverity.CRITICAL else drift_warn,
                    detail="occupancy drift exceeded threshold",
                    routing=routing,
                )
            )

    reason_distribution = daily_metrics.get("reason_code_distribution", {})
    if reason_distribution is not None and not isinstance(reason_distribution, Mapping):
        alerts.append(
            _build_alert(
                metric="slo_input_contract",
                severity=SLOSeverity.CRITICAL,
                observed=1.0,
                threshold=0.0,
                detail="reason_code_distribution must be a mapping",
                routing=routing,
            )
        )
    elif isinstance(reason_distribution, Mapping):
        for reason_code, caps in reason_caps.items():
            if reason_code not in reason_distribution:
                continue
            observed_ratio = _to_rate(reason_distribution.get(reason_code), field_name=reason_code)
            observed_metrics[f"reason_code_distribution.{reason_code}"] = observed_ratio
            warn = float(dict(caps).get("warn", 1.0))
            critical = float(dict(caps).get("critical", 1.0))
            severity = _resolve_severity(observed_ratio, warn, critical)
            if severity != SLOSeverity.OK:
                alerts.append(
                    _build_alert(
                        metric="reason_code_distribution",
                        severity=severity,
                        observed=observed_ratio,
                        threshold=critical if severity == SLOSeverity.CRITICAL else warn,
                        detail=f"{reason_code} ratio exceeded {severity.value} threshold",
                        routing=routing,
                    )
                )

    status = SLOSeverity.OK
    if any(alert.severity == SLOSeverity.CRITICAL for alert in alerts):
        status = SLOSeverity.CRITICAL
    elif alerts:
        status = SLOSeverity.WARN

    return SLOEvaluation(
        status=status,
        alerts=alerts,
        observed_metrics=observed_metrics,
        report_date=str(daily_metrics.get("date")) if daily_metrics.get("date") is not None else None,
    )


def generate_daily_slo_report(
    daily_metrics: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] = DEFAULT_SLO_POLICY,
) -> Dict[str, Any]:
    """Generate daily model-behavior SLO report with actionable alerts."""
    return evaluate_model_behavior_slo(daily_metrics, policy=policy).to_dict()
