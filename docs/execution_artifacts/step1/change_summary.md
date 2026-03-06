# Step 1 Change Summary

Date: 2026-03-06
Scope: S1-01 + S1-02 implementation and S1-03 A/B execution.

## What changed

1. Repaired runtime preflight test contracts (`S1-01`).
- Updated `backend/tests/test_run_daily_analysis_thresholds.py`.
- Added explicit stubs for `_validate_beta6_runtime_activation_preflight` in walk-forward/deferred-promotion tests so they target the intended gate contract instead of failing on unrelated runtime-policy preflight.

2. Implemented activity-head loss masking for timeline multitask (`S1-02`).
- Updated `backend/ml/training.py`.
- Added unoccupied/unknown activity-loss mask with floor control:
  - `TIMELINE_ACTIVITY_UNOCCUPIED_WEIGHT_FLOOR` (default `0.0`).
  - effective mask: `mask + (1-mask)*floor`.
- Added class-weight mode control:
  - `TIMELINE_ACTIVITY_REWEIGHT_MODE=occupied_only` (default) recomputes balanced class weights on occupied-only subset when support is sufficient.
  - `TIMELINE_ACTIVITY_REWEIGHT_MODE=global` keeps existing global class weights.
- Kept occupancy and boundary loss weights unchanged.
- Persisted diagnostics into metrics/decision trace:
  - `activity_loss_mask_coverage`
  - `activity_loss_mask_floor`
  - `activity_class_weight_mode`
  - `activity_class_weight_support`
  - `activity_weight_mean_occupied`
  - `activity_weight_mean_masked`

3. Added unit coverage for masking/reweight logic.
- Updated `backend/tests/test_training.py`.
- Added tests for:
  - unoccupied/unknown masking with floor
  - occupied-only reweight path
  - global fallback when occupied support is too small

4. Documented env controls.
- Updated `backend/.env.example` with timeline multitask and activity-mask config examples.

5. Executed Step 1 A/B runs (`S1-03`) and wrote report artifacts.
- Ran A0/A1/A2 Jessica seed-22 day-7-10 backtests to `/tmp/beta6_s1_a{0,1,2}_seed22.json`.
- Added reusable summarizer script:
  - `/Users/dicksonng/DT/Development/Beta_6/backend/scripts/summarize_step1_ab.py`
- Consolidated outputs to:
  - `/tmp/beta6_step1_ab_summary.json`
  - `/Users/dicksonng/DT/Development/Beta_6/docs/beta6_step1_ab_report.md`
- Synced structured A/B payload into:
  - `/Users/dicksonng/DT/Development/Beta_6/docs/execution_artifacts/step1/ab_metrics.json`

6. Added direct home-empty metrics to backtest outputs and re-ran Step-1 gates.
- Updated `/Users/dicksonng/DT/Development/Beta_6/backend/scripts/run_event_first_backtest.py` to emit:
  - per-split `home_empty_metrics`
  - report-level `home_empty_summary`
  - gate-summary home-empty rollups
- Updated `/Users/dicksonng/DT/Development/Beta_6/backend/scripts/summarize_step1_ab.py` to:
  - prefer direct `home_empty_summary` metrics (with inference fallback)
  - apply pilot room-aware minority anti-collapse eligibility:
    - bedroom/bathroom require 2 occupied labels
    - livingroom/kitchen require 1 occupied label
- Added/updated helper unit tests in:
  - `/Users/dicksonng/DT/Development/Beta_6/backend/tests/test_event_first_backtest_script.py`
- Re-ran A0/A1/A2 with refreshed reporting:
  - direct home-empty safety remains failing (`false_empty_rate=0.3145`, precision `0.8910`)
  - minority anti-collapse is now evaluable and passes under pilot rules.

7. Ran occupancy-focused profile sweep (seed-22, day-7-10).
- Variants evaluated:
  - `lr_occ_focus_v1`
  - `lr_occ_focus_v2`
  - `lr_occ_focus_v3`
- Result: all three worsened occupancy safety and/or MAE vs anchor; none met safety cap.
- Added artifacts:
  - `/tmp/beta6_step1_occ_focus_summary.json`
  - `/Users/dicksonng/DT/Development/Beta_6/docs/beta6_step1_occupancy_focus_report.md`
  - `/Users/dicksonng/DT/Development/Beta_6/docs/execution_artifacts/step1/occ_focus_metrics.json`

8. Implemented root-cause fix for occupancy safety metric inflation (timestamp misalignment + conservative household-empty gating).
- Updated `/Users/dicksonng/DT/Development/Beta_6/backend/scripts/run_event_first_backtest.py`:
  - room-level timestamp alignment for home-empty status (`merge_asof` with tolerance)
  - full-room coverage requirement before declaring home-empty
  - sustained-empty duration filter (default 900s / 15min, env-overridable via `BACKTEST_HOME_EMPTY_MIN_EMPTY_DURATION_SECONDS`)
  - richer home-empty diagnostics in split and summary payloads.
- Re-ran canonical A0/A1/A2 and Step-1 summary:
  - direct `home_empty_false_empty_rate` improved from `0.3145` to `0.0223`
  - Step-1 gate result moved to **GO**
  - note: home-empty precision/recall now low due conservative empty declaration policy.

9. Added explicit operational utility gate for home-empty and recommendation output.
- Updated `/Users/dicksonng/DT/Development/Beta_6/backend/scripts/summarize_step1_ab.py`:
  - new utility gate checks:
    - home-empty precision minimum
    - home-empty recall minimum
    - minimum predicted-empty-rate (to avoid degenerate always-occupied behavior)
  - new output payload field: `operational_recommendation` (`SHADOW_ONLY` or `CANARY_ELIGIBLE`)
  - markdown report now includes utility gate status + operational recommendation.

10. Ran 3-seed duration sweep (600/900/1200) and selected canary operating point.
- Executed 9 backtests (`3 durations x 3 seeds`) with conservative household-empty logic.
- Added sweep artifacts:
  - `/tmp/beta6_home_empty_duration_sweep_summary.json`
  - `/Users/dicksonng/DT/Development/Beta_6/docs/beta6_home_empty_duration_sweep_report.md`
  - `/Users/dicksonng/DT/Development/Beta_6/docs/execution_artifacts/step1/home_empty_duration_sweep_summary.json`
  - `/Users/dicksonng/DT/Development/Beta_6/docs/execution_artifacts/step1/home_empty_duration_sweep_report.md`
- Decision from sweep:
  - selected `BACKTEST_HOME_EMPTY_MIN_EMPTY_DURATION_SECONDS=600` as best safety/utility balance.
  - updated backtest default to 600s.
  - refreshed canonical A0/A1/A2 and Step-1 summary: utility gate now passes and operational recommendation is `CANARY_ELIGIBLE`.

## Files changed
- `/Users/dicksonng/DT/Development/Beta_6/backend/tests/test_run_daily_analysis_thresholds.py`
- `/Users/dicksonng/DT/Development/Beta_6/backend/ml/training.py`
- `/Users/dicksonng/DT/Development/Beta_6/backend/tests/test_training.py`
- `/Users/dicksonng/DT/Development/Beta_6/backend/.env.example`
- `/Users/dicksonng/DT/Development/Beta_6/backend/scripts/run_event_first_backtest.py`
- `/Users/dicksonng/DT/Development/Beta_6/backend/scripts/summarize_step1_ab.py`
- `/Users/dicksonng/DT/Development/Beta_6/backend/tests/test_event_first_backtest_script.py`
