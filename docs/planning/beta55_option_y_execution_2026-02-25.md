# Beta 5.5 Option Y Execution (2026-02-25)

## Scope

Implement Option Y end-to-end:

1. Treat LivingRoom episode gates as informational-only (non-blocking).
2. Apply Day-8 Bedroom unsensored-sleep runtime mask (no source-file deletion).
3. Add MAE non-regression guardrails vs anchor baseline.
4. Run canonical 3-seed decision flow.

## Code / Config Changes

### 1) Go/No-Go logic update

File:
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_matrix.py`

Changes:
- Added `informational_checks` support in `go_no_go` rules.
  - Failed informational checks are reported but do not block promotion.
- Added relative MAE non-regression checks:
  - `livingroom_active_mae_max_regression_pct`
  - `bedroom_sleep_mae_max_regression_pct`
  - compared against:
    - `livingroom_active_mae_baseline_minutes`
    - `bedroom_sleep_mae_baseline_minutes`
- Added Day-8 Bedroom sleep recall low-support skip behavior:
  - `day8_bedroom_sleep_recall_min_support`
  - if no eligible support, check passes with `low_support_skip=true`.

### 2) Go/No-Go policy config

File:
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/config/event_first_go_no_go.yaml`

Added/updated:
- `informational_checks`:
  - `day7_livingroom_episode_recall_min`
  - `livingroom_episode_recall_min`
  - `livingroom_episode_f1_min`
- MAE guardrails:
  - `livingroom_active_mae_baseline_minutes: 131.13459299972334`
  - `livingroom_active_mae_max_regression_pct: 10.0`
  - `bedroom_sleep_mae_baseline_minutes: 130.1962363951708`
  - `bedroom_sleep_mae_max_regression_pct: 10.0`
- Day-8 sleep support floor for gate evaluation:
  - `day8_bedroom_sleep_recall_min_support: 100`

### 3) Runtime mask artifact (Day-8 Bedroom)

File:
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/config/label_corrections/option_y_day8_bedroom_unsensored_mask.csv`

Rows:
- `2025-12-08T00:00:00` to `2025-12-08T00:25:40` -> `unoccupied`
- `2025-12-08T22:13:37` to `2025-12-08T23:59:51` -> `unoccupied`

Total masked windows:
- 702 sleep windows (117.0 minutes), matching Day-8 Bedroom sleep labels.

### 4) Matrix profile wiring

File:
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/config/event_first_matrix_profiles.yaml`

Added:
- Variant: `anchor_top2_frag_v3_option_y_masked_day8`
- Profile: `option_y_canonical_full` (3 seeds, canonical decision profile)

## Test Results

Command:
- `pytest -q tests/test_event_first_go_no_go.py tests/test_run_event_first_matrix.py tests/test_run_lr_fragmentation_sweep_clean.py`

Result:
- `11 passed`

Command:
- `pytest -q tests/test_event_first_go_no_go.py tests/test_run_event_first_matrix.py tests/test_run_lr_fragmentation_sweep_clean.py tests/test_event_first_backtest_script.py`

Result:
- `90 passed`

## Canonical Run Artifacts

Run root:
- `/tmp/beta55_option_y_canonical_20260225/option_y_canonical_full/option_y_canonical_full`

Seed reports:
- `/tmp/beta55_option_y_canonical_20260225/option_y_canonical_full/option_y_canonical_full/anchor_top2_frag_v3_option_y_masked_day8/seed_11.json`
- `/tmp/beta55_option_y_canonical_20260225/option_y_canonical_full/option_y_canonical_full/anchor_top2_frag_v3_option_y_masked_day8/seed_22.json`
- `/tmp/beta55_option_y_canonical_20260225/option_y_canonical_full/option_y_canonical_full/anchor_top2_frag_v3_option_y_masked_day8/seed_33.json`

Aggregate:
- `/tmp/beta55_option_y_canonical_20260225/option_y_canonical_full/option_y_canonical_full/anchor_top2_frag_v3_option_y_masked_day8/rolling.json`
- `/tmp/beta55_option_y_canonical_20260225/option_y_canonical_full/option_y_canonical_full/anchor_top2_frag_v3_option_y_masked_day8/signoff.json`

## Option Y Go/No-Go Outcome (Updated Policy Logic)

Using updated `_evaluate_go_no_go` on canonical seed reports:

- Status: `fail`
- Blocking failures:
  - `bedroom_max_regression_splits` (actual `10`, required `<=1`)
  - `day8_livingroom_fragmentation_min` (actual min `0.20`, required `>=0.45`)
- Informational-only failures (reported, non-blocking):
  - `livingroom_episode_recall_min`
  - `livingroom_episode_f1_min`
  - `day7_livingroom_episode_recall_min`
- Day-8 Bedroom sleep gate behavior:
  - `pass=true` with `low_support_skip=true` and `eligible_count=0` (as intended by Option Y masking policy)
- MAE non-regression guards:
  - LivingRoom MAE guard: `pass`
  - Bedroom MAE guard: `pass` (regression ~`9.61%`, under `10%` cap)

Recomputed go/no-go artifact:
- `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/beta55_option_y_go_no_go_recompute_2026-02-25.json`

## Practical Readout for New Prediction Files

From canonical rolling summary (same elder/home pattern):

- LivingRoom:
  - accuracy ~`0.716`
  - macro F1 ~`0.703`
  - occupied F1 ~`0.685`
  - active-duration MAE ~`131` min/day
- Bedroom:
  - accuracy ~`0.583`
  - macro F1 ~`0.419`
  - occupied F1 ~`0.394`
  - sleep-duration MAE ~`143` min/day
- Kitchen:
  - accuracy ~`0.891`, occupied F1 ~`0.730`, duration MAE ~`82` min/day
- Bathroom:
  - accuracy ~`0.967`, occupied F1 ~`0.626`, duration MAE ~`39` min/day
- Entrance:
  - accuracy ~`0.961`, out-duration MAE ~`56` min/day

## Recommendation

1. Option Y policy implementation is complete and working as designed.
2. Promotion is still blocked by two non-LR-episode blockers:
   - Bedroom regression guard (`bedroom_max_regression_splits`)
   - Day-8 LivingRoom fragmentation floor
3. If promotion is urgent, the next product-level decision is whether to temporarily relax either:
   - `bedroom_max_regression_splits`, and/or
   - `day8_livingroom_fragmentation_min`
   with explicit operator caveats.
