from ml.beta6.slo_observability import (
    SLOSeverity,
    evaluate_model_behavior_slo,
    generate_daily_slo_report,
)


def test_slo_observability_ok_when_metrics_within_thresholds():
    evaluation = evaluate_model_behavior_slo(
        {
            "date": "2026-02-25",
            "unknown_rate": 0.05,
            "abstain_rate": 0.06,
            "occupancy_rate": 0.45,
            "baseline_occupancy_rate": 0.40,
            "reason_code_distribution": {
                "fail_uncertainty_unknown": 0.02,
                "fail_timeline_mae": 0.01,
            },
        }
    )
    assert evaluation.status == SLOSeverity.OK
    assert evaluation.alerts == []


def test_slo_observability_critical_unknown_rate_routes_with_eta():
    evaluation = evaluate_model_behavior_slo(
        {
            "date": "2026-02-25",
            "unknown_rate": 0.20,
            "abstain_rate": 0.06,
            "occupancy_rate": 0.45,
            "baseline_occupancy_rate": 0.40,
            "reason_code_distribution": {},
        }
    )
    assert evaluation.status == SLOSeverity.CRITICAL
    unknown_alerts = [alert for alert in evaluation.alerts if alert.metric == "unknown_rate"]
    assert len(unknown_alerts) == 1
    assert unknown_alerts[0].severity == SLOSeverity.CRITICAL
    assert unknown_alerts[0].owner == "mlops_oncall"
    assert unknown_alerts[0].remediation_eta_hours == 4


def test_slo_observability_warn_on_occupancy_drift():
    evaluation = evaluate_model_behavior_slo(
        {
            "date": "2026-02-25",
            "unknown_rate": 0.05,
            "abstain_rate": 0.06,
            "occupancy_rate": 0.61,
            "baseline_occupancy_rate": 0.48,
            "reason_code_distribution": {},
        }
    )
    assert evaluation.status == SLOSeverity.WARN
    drift_alerts = [alert for alert in evaluation.alerts if alert.metric == "occupancy_drift_abs_pp"]
    assert len(drift_alerts) == 1
    assert drift_alerts[0].severity == SLOSeverity.WARN
    assert drift_alerts[0].owner == "modeling_lead"


def test_slo_observability_reason_code_distribution_breach():
    evaluation = evaluate_model_behavior_slo(
        {
            "date": "2026-02-25",
            "unknown_rate": 0.05,
            "abstain_rate": 0.06,
            "occupancy_rate": 0.45,
            "baseline_occupancy_rate": 0.40,
            "reason_code_distribution": {
                "fail_uncertainty_unknown": 0.13,
            },
        }
    )
    assert evaluation.status == SLOSeverity.CRITICAL
    distribution_alerts = [alert for alert in evaluation.alerts if alert.metric == "reason_code_distribution"]
    assert len(distribution_alerts) == 1
    assert distribution_alerts[0].severity == SLOSeverity.CRITICAL
    assert distribution_alerts[0].owner == "qa_gate_owner"


def test_slo_observability_missing_required_input_creates_actionable_contract_alert():
    report = generate_daily_slo_report(
        {
            "date": "2026-02-25",
            "unknown_rate": 0.02,
            "abstain_rate": 0.03,
            "occupancy_rate": 0.4,
            # baseline_occupancy_rate intentionally missing
            "reason_code_distribution": {},
        }
    )
    assert report["status"] == SLOSeverity.CRITICAL.value
    assert any(alert["metric"] == "slo_input_contract" for alert in report["alerts"])
