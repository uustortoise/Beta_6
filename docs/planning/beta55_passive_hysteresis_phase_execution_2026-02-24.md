# Beta 5.5 Phase Execution: Episode Metrics + Passive Hysteresis (2026-02-24)

## Scope Executed
Kept existing model stack (including Stage-A transformer path availability) and implemented the recommended phases:
1. Phase 0: Add episode-level diagnostics.
2. Phase 1: Add room-scoped passive occupancy hysteresis (default OFF).
3. Phase 1b: Add additional temporal context features (additive, no schema removals).

## Code Changes
### 1) Episode-Level Metrics (Phase 0)
- File: `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
- Added:
  - `_estimate_window_seconds(...)`
  - `_compute_binary_episode_metrics(...)`
- New per-split room output field:
  - `episode_metrics`
- Purpose:
  - Exposes occupied-vs-unoccupied episode precision/recall/F1 and binary timeline metrics so passive-occupancy behavior is measurable beyond window F1.

### 2) Passive Occupancy Hysteresis (Phase 1)
- File: `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
- Added:
  - `_apply_room_passive_occupancy_hysteresis(...)`
- Applied only for Bedroom/LivingRoom and controlled by feature flags.
- New per-split room output debug field:
  - `room_passive_hysteresis`

### 3) Temporal Feature Enrichment (Phase 1b)
- File: `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
- Added in `_add_room_temporal_occupancy_features(...)`:
  - `occ_motion_active_5m`
  - `occ_time_since_motion_active_windows`
  - `occ_time_since_motion_active_minutes`
  - `occ_light_roll_mean_10m`
  - `occ_light_roll_std_10m`
  - `occ_co2_slope_15m`
- Added helper:
  - `_time_since_last_active_windows(...)`

### 4) New CLI Controls (default OFF)
- File: `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
- Added flags:
  - `--enable-bedroom-livingroom-passive-hysteresis`
  - `--disable-bedroom-livingroom-passive-hysteresis`
  - `--bedroom-passive-hold-minutes`
  - `--livingroom-passive-hold-minutes`
  - `--passive-exit-min-consecutive-windows`
  - `--passive-entry-occ-threshold`
  - `--passive-entry-room-prob-threshold`
  - `--passive-stay-occ-threshold`
  - `--passive-stay-room-prob-threshold`
  - `--passive-exit-occ-threshold`
  - `--passive-exit-room-prob-threshold`
  - `--passive-motion-reset-threshold`
  - `--passive-motion-quiet-threshold`

### 5) Matrix Profile A/B Variant
- File: `/Users/dicksonng/DT/Development/Beta_5.5/backend/config/event_first_matrix_profiles.yaml`
- Added variant:
  - `anchor_top2_frag_v3_passive_hysteresis`
- Added profiles:
  - `passive_hysteresis_ablation_quick`
  - `passive_hysteresis_ablation_full`

### 6) Tests Updated
- File: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`
- Added/updated tests for:
  - new temporal features
  - passive hysteresis behavior
  - binary episode metrics output contract

## Test Execution
- Command:
  - `pytest -q /Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`
- Result:
  - `75 passed`

Additional readiness tests (already passing in this cycle):
- `tests/test_run_event_first_matrix.py`
- `tests/test_run_event_first_smoke.py`
- `tests/test_event_first_go_no_go.py`

## Data Intake + Smoke
### Validation
- Output: `/tmp/beta55_phase_hyst_validate_20260224.json`
- Result: PASS

### Diff
- Outputs:
  - `/tmp/beta55_phase_hyst_diff_20260224.json`
  - `/tmp/beta55_phase_hyst_diff_20260224.csv`

### Smoke
- Output: `/tmp/beta55_phase_hyst_smoke_20260224.json`
- Result: FAIL
- Blocking reason:
  - `occupied_rate_below_threshold:livingroom:day7`

## A/B Result (Seed 11)
Compared current anchor vs anchor + passive hysteresis defaults.

Artifacts:
- Anchor seed report:
  - `/tmp/beta55_phase_hyst_ablation_20260224/anchor_top2_frag_v3_seed11.json`
- Passive-hysteresis seed report:
  - `/tmp/beta55_phase_hyst_ablation_20260224/anchor_top2_frag_v3_passive_hysteresis_seed11.json`
- Before/after summary:
  - `/tmp/beta55_phase_hyst_ablation_20260224/before_after_seed11.md`
  - `/tmp/beta55_phase_hyst_ablation_20260224/before_after_seed11.csv`

Key outcomes (seed 11):
- Bedroom:
  - accuracy `0.7535 -> 0.5040` (down)
  - occupied_recall `0.7014 -> 1.0000` (up)
  - occupied_f1 `0.7462 -> 0.6807` (down)
- LivingRoom:
  - accuracy `0.7028 -> 0.1461` (down)
  - occupied_recall `0.4518 -> 1.0000` (up)
  - occupied_f1 `0.2912 -> 0.2477` (down)

Interpretation:
- Default passive hysteresis is too aggressive in current tuning.
- It over-holds occupancy, drives recall to 1.0, and collapses precision/F1.

## Operational Note
Full 3-seed A/B orchestration encountered intermittent runtime/session instability in this environment (stuck runner sessions around process cleanup), so this execution package is currently seed-11 complete with deterministic artifacts above.

## Recommendation
1. Keep passive hysteresis default OFF.
2. Retune hysteresis aggressively toward shorter hold + faster exit before 3-seed rerun.
3. Use the new `episode_metrics` field as the primary decision signal for persistence behavior, alongside occupied F1 and hard gates.

## Follow-up Fix (same day)
A logic fix was applied after the initial seed-11 A/B run:
- File: `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
- Change: passive hysteresis `entry/stay` checks switched from permissive OR to stricter AND between occupancy and room-label probabilities.

Why:
- Initial implementation could over-activate occupancy when room label probability alone stayed elevated.

Validation:
- Unit tests pass (`75 passed`) including passive-hysteresis tests.
- Full rerun on this host remains operationally unstable due lingering multiprocessing `resource_tracker` cleanup behavior; re-benchmark is pending in a clean runner session.
