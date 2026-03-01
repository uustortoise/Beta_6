# Beta 5.5 Status Handover (2026-02-19)

## 1. Executive Status

- Current state: **not promotion-ready** for strict WS-6.
- Major contract issues are fixed (fail-closed baseline binding, leakage artifact enforcement, strict matrix enforcement behavior, dual scoreboard support).
- Kitchen KPI blocker has been reduced/fixed in the best current run.
- Primary remaining blocker is **Bedroom/LivingRoom hard-gate performance** (occupied F1/recall/fragmentation on eligible cells).

## 2. Best Known Baseline to Continue From

- Run family: `ws6_next_ab_min3_smooth_kitchen_tune`
- Artifacts:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_next_ab_min3_smooth_kitchen_tune/ws6_rolling.json`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_next_ab_min3_smooth_kitchen_tune/ws6_signoff.json`
- Score:
  - Eligible scoreboard: **23/30**
  - Full scoreboard: **41/60**
- KPI:
  - Kitchen MAE mean: **86.77** (below 90 threshold)
- Remaining fail reasons:
  - `hard_gate_all_seeds_failed`
  - `hard_gate_split_requirement_failed`

## 3. What Was Implemented

## 3.1 Core hardening and gating

- Fail-closed baseline binding + artifact hash checks.
- Leakage audit artifact contract checks (file/schema/pass status).
- Strict split-seed matrix enforcement behavior.
- Dual scoreboard plumbing (eligible vs full) in aggregate signoff.

## 3.2 A+B wiring

- `hard_gate_min_train_days` eligibility support in run + aggregate.
- Bedroom/LivingRoom prediction smoothing integrated into evaluation path.

## 3.3 Kitchen tuning bug fix

- Important fix: kitchen MAE tuning no longer silently depends on global room-MAE tuning.
- This was committed and pushed in prior work (`cde9d3c` branch history).

## 3.4 Backtest file discovery fix

- Removed hardcoded `dec2025` dependency in WS-6 backtest discovery.
- Backtest now scans canonical resident training files across months/years:
  - `*_train_*.xlsx`, `*.xls`, `*.parquet`
- Excludes derived variants with extra suffixes.
- Added fail-closed ambiguity check if multiple files map to same day token.
- Commit pushed: `724e34d`.

## 4. What Was Tested and Learned

## 4.1 Variants tested (high-level)

- Baseline strict repro (`40/60`) showed Kitchen KPI fail (~98.8).
- Kitchen threshold variants:
  - Some variants fixed Kitchen but caused LivingRoom or Entrance KPI regressions.
- A+B + kitchen-tune config achieved the best safety/perf balance:
  - Kitchen fixed, no new KPI explosions, but hard-gates still short.
- Aggressive BL packages (HGB + hard negatives + replay + boundary reweighting) regressed both gates and LivingRoom KPI.

## 4.2 Root pattern

- Threshold and decoder tuning can move errors between rooms.
- Bedroom/LivingRoom separability remains the bottleneck.
- Sparse/non-contiguous training coverage likely contributes to instability.

## 5. Outstanding Issues

1. Bedroom/LivingRoom hard-gate pass rate is below requirement.
2. LivingRoom occupied F1/recall is the most persistent hard-gate failure mode.
3. Fragmentation still fails in some Bedroom cells.
4. Data coverage is not ideal for robust split behavior (especially with gaps in training days).

## 6. Recommended Immediate Approach (for newcomer)

## 6.1 Do this first

1. Keep current best config as anchor:
   - `ws6_next_ab_min3_smooth_kitchen_tune` behavior.
2. Add contiguous canonical training files for missing day windows (especially around existing gaps).
3. Re-run strict WS-6 with the same knobs before introducing new model complexity.

## 6.2 Why this order

- Data coverage improvement is lower risk than architecture churn.
- Current codebase already has many tuning knobs; new aggressive changes caused regressions.

## 6.3 If still stuck after contiguous data

1. Run targeted BL-only objective tuning with strict KPI guardrails.
2. Avoid bundle changes that include kitchen/entrance tuning simultaneously.
3. Only escalate to architecture changes after a controlled A/B against the current anchor.

## 7. Newcomer Quick Start Checklist

1. Use branch `beta-5.5-transformer`.
2. Read:
   - `/Users/dicksonng/DT/Development/Beta_5.5/docs/Option A+B.md`
   - `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/event_first_cnn_transformer_execution_plan.md`
   - `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/beta55_status_handover_2026-02-19.md`
3. Verify tests:
   - `pytest -q tests/test_event_first_backtest_script.py tests/test_event_first_backtest_aggregate.py tests/test_d2_strict_splitseed_integration.py`
4. Confirm latest anchor artifacts exist:
   - `backend/tmp/ws6_next_ab_min3_smooth_kitchen_tune/...`
5. Prepare canonical training files only:
   - `HKxxxx_name_train_<date>.xlsx` (no derived suffix variants).
6. Drop new files to raw intake path:
   - `/Users/dicksonng/DT/Development/Beta_5.5/data/raw` (or env-overridden RAW_DATA_DIR).
7. Re-run WS-6 strict evaluation from anchor config and compare deltas to anchor artifacts.

## 8. Known Footguns

1. Mixing canonical and derived training variants in one set can create unexpected behavior.
2. Overly broad threshold tuning can fix one room while breaking another KPI.
3. Session/PTY execution can appear idle in tooling; confirm by checking output artifacts and process table.

## 9. Bottom Line

- Beta 5.5 is materially hardened and reproducible.
- Promotion is currently blocked by Bedroom/LivingRoom hard-gate quality, not by contract integrity.
- The next highest-confidence step is **data coverage hardening + strict re-eval from the current anchor**, not another aggressive model bundle.
