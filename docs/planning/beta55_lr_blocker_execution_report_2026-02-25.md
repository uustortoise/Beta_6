# Beta 5.5 LivingRoom Blocker Execution Report
**Date:** 2026-02-25  
**Scope:** LivingRoom blocker clearance attempts for reliable activity timeline promotion

## 1) Problem Statement
Beta 5.5 still fails LivingRoom hard gates despite label corrections and multiple tuning sweeps.  
Primary unresolved blocker is **LR episode-level quality** (not aggregate pass count).

Current gate failures (both anchor and latest candidate):
- `livingroom_eligible_pass_count_min` (required `>=3`, actual `0`)
- `livingroom_episode_recall_min` (required `>=0.40`)
- `livingroom_episode_f1_min` (required `>=0.35`)
- `day7_livingroom_episode_recall_min` (required `>=0.45`)

## 2) What Was Executed
## 2.1 Code changes
1. Updated passive hysteresis implementation to support **true per-room disable** when `hold_minutes <= 0`.
   - File: `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
2. Added regression test for the zero-hold disable behavior.
   - File: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`

## 2.2 Validation tests
Executed:
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend
pytest -q tests/test_event_first_backtest_script.py tests/test_event_first_go_no_go.py tests/test_run_event_first_matrix.py
```
Result: **84 passed**

## 2.3 Full matrix rerun (3 seeds)
Executed profile:
- `lr_semantic_alignment_tune_full` (anchor vs `lr_semantic_align_v6`)

Output bundle:
- `/tmp/beta55_lr_semantic_tune_full_r2_20260225/lr_semantic_alignment_tune_full`

Key artifacts:
- `/tmp/beta55_lr_semantic_tune_full_r2_20260225/lr_semantic_alignment_tune_full/ranking.csv`
- `/tmp/beta55_lr_semantic_tune_full_r2_20260225/lr_semantic_alignment_tune_full/clean_sweep_manifest.json`
- `/tmp/beta55_lr_semantic_tune_full_r2_20260225/lr_semantic_alignment_tune_full/anchor_top2_frag_v3/signoff.json`
- `/tmp/beta55_lr_semantic_tune_full_r2_20260225/lr_semantic_alignment_tune_full/lr_semantic_align_v6/signoff.json`

## 3) Results
## 3.1 Go/No-Go summary
| Variant | Eligible Passed | Eligible Total | LR Eligible Passed | Status |
|---|---:|---:|---:|---|
| `anchor_top2_frag_v3` | 48 | 60 | 0 | FAIL |
| `lr_semantic_align_v6` | 48 | 60 | 0 | FAIL |

No promotion delta on required LR gate outcomes.

## 3.2 What improved vs what did not
### Improved
1. Bedroom collateral risk from LR tuning was contained:
   - `bedroom_max_regression_splits`: pass for both anchor and `v6` (actual `0`, required max `1`).
2. LR strict recall-style checks were high in `v6`:
   - `day7_livingroom_recall_min`: pass.

### Not improved enough (blocking)
1. LR **episode recall/f1** stayed below thresholds.
2. LR **eligible pass count** stayed at `0/12`.

## 3.3 Timeline KPI impact (critical)
From signoff room summaries:

| Variant | LivingRoom `livingroom_active_mae_minutes` | Bedroom `sleep_duration_mae_minutes` |
|---|---:|---:|
| `anchor_top2_frag_v3` | 123.814 | 46.688 |
| `lr_semantic_align_v6` | 370.490 | 50.446 |

Interpretation:
- `v6` did not clear LR gates.
- `v6` also materially worsened LR timeline MAE (promotion risk increases).

## 4) Diagnosis
Observed pattern is consistent with **label-policy vs model-objective mismatch** for LivingRoom passive occupancy:
1. Labels represent episode persistence (enter room, remain positive during passive periods).
2. Window classifier optimizes instantaneous separability.
3. Hysteresis/alignment tuning shifts precision/recall mix but does not reliably recover episode boundaries required by gates.
4. As a result, LR hard-gate fails remain persistent across variants even when some local metrics improve.

## 5) Recommendation For Team Discussion
## 5.1 Immediate decision
1. Keep `top2_frag_v3` as active anchor for this cycle.
2. Do **not** promote `lr_semantic_align_v6`.

## 5.2 Next technical step (highest value)
Move from smoothing-only tactics to **objective-aligned LR episode modeling**:
1. Train/evaluate with explicit entry/exit-aware objectives for LR occupancy episodes.
2. Keep current production path default-off for any experimental decoder until LR episode gates pass.
3. Re-run 3-seed matrix with same go/no-go config after objective-aligned update.

## 5.3 Suggested acceptance criteria for next attempt
1. `livingroom_eligible_pass_count_min >= 3`
2. `livingroom_episode_recall_min >= 0.40`
3. `livingroom_episode_f1_min >= 0.35`
4. `day7_livingroom_episode_recall_min >= 0.45`
5. No regression on Bedroom guardrail and no major timeline KPI deterioration.

## 6) Bottom Line
Execution completed end-to-end and validated.  
The latest LR candidate (`v6`) is **not promotable**.  
Blocker is structural (objective alignment), not a remaining sweep/tuning gap.

