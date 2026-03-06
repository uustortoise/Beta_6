# Step 1 Go/No-Go

Date: 2026-03-06
Step: WS0 + WS0.5

## Decision
- Engineering gate: GO
- Core numeric A/B gate: GO
- Promotion gate: GO (all Step-1 gates pass under aligned conservative home-empty logic)

## Reason codes
- GO_S1_01_TEST_CONTRACTS_REPAIRED
- GO_S1_02_ACTIVITY_MASK_IMPLEMENTED
- GO_STEP1_UNIT_AND_GATE_TESTS_PASSING
- GO_STEP1_AB_VARIANT_RESULTS_AVAILABLE
- GO_STEP1_LR_MAE_IMPROVEMENT_GE_20PCT
- GO_STEP1_BEDROOM_SLEEP_DELTA_WITHIN_2MIN
- GO_STEP1_HARD_GATE_PASS_COUNT_NON_REGRESSION
- GO_STEP1_MINORITY_RECALL_PILOT_RULE_PASS
- GO_STEP1_OCCUPANCY_FALSE_EMPTY_RATE_WITHIN_CAP_DIRECT

## Post-GO caution
Observed behavior after 3-seed sweep and tuned conservative household-empty setting (600s sustained empty):
- Direct `home_empty_false_empty_rate` now `0.0304` (seed-22) and remains below `<= 0.05` for seeds `11/22/33`.
- Home-empty utility for A1 (seed-22): precision `0.2575`, recall `0.1029`, predicted-empty rate `0.0371`.
- Operational recommendation: **CANARY_ELIGIBLE** (limited canary only; no full rollout yet).

Reference artifacts:
- `/tmp/beta6_step1_ab_summary.json`
- `/Users/dicksonng/DT/Development/Beta_6/docs/beta6_step1_ab_report.md`
- `/tmp/beta6_home_empty_duration_sweep_summary.json`
- `/Users/dicksonng/DT/Development/Beta_6/docs/beta6_home_empty_duration_sweep_report.md`
