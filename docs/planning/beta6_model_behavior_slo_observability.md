# Beta 6 Model-Behavior SLO Observability (v1)

- Date: 2026-02-25
- Status: Active
- Evaluator: `backend/ml/beta6/slo_observability.py`
- Policy file: `backend/config/beta6_model_behavior_slo.yaml`

## 1. Daily SLO Metrics

1. `unknown_rate`
2. `abstain_rate`
3. `occupancy_drift_abs_pp` (derived from `occupancy_rate` vs `baseline_occupancy_rate`)
4. reason-code distribution ratios (for selected high-risk reason codes)

## 2. Alert Routing

1. Warn alerts route to `slack://beta6-model-alerts`, ETA 24 hours.
2. Critical alerts route to `pagerduty://beta6-mlops`, ETA 4 hours.
3. Owner is determined by metric:
   - unknown/abstain -> `mlops_oncall`
   - occupancy drift -> `modeling_lead`
   - reason-code distribution -> `qa_gate_owner`
   - contract/input failures -> `ml_platform_owner`

## 3. Fail-Closed Behavior

Missing or invalid required SLO inputs emit critical `slo_input_contract` alerts. This prevents silent degradation from malformed daily payloads.

## 4. Deliverable

`generate_daily_slo_report(...)` returns actionable alerts with:
1. owner
2. escalation route
3. remediation ETA
4. observed value and threshold
