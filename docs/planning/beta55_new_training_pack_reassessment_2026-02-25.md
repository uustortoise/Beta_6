# Beta 5.5 Reassessment on New Training Pack (2026-02-25)

## Data Source Verification

This reassessment uses the **new top-level pack**:
- `/Users/dicksonng/DT/Development/New training files`

It does **not** use:
- `/Users/dicksonng/DT/Development/New training files/corrected_clones`

Evidence:
- Full run manifest data_dir:
  - `/tmp/beta55_option_y_newpack_anchor_full_20260225/option_y_newpack_anchor_full/option_y_newpack_anchor_full/clean_sweep_manifest.json`
  - `data_dir=/Users/dicksonng/DT/Development/New training files`

## Key Label Shift vs Previous Pack

LivingRoom label prevalence in new pack dropped sharply vs `corrected_clones`:

- Day 4: 0.0867 vs 0.3205
- Day 5: 0.0649 vs 0.3755
- Day 6: 0.2381 vs 0.4778
- Day 7: 0.0718 vs 0.3641
- Day 8: 0.1962 vs 0.6197
- Day 9: 0.1948 vs 0.5860
- Day 10: 0.1109 vs 0.2561

Bedroom sleep windows increased strongly in new pack (day-level deltas), and Day-8 now has 2194 sleep windows (vs 702 in corrected_clones).

## Option Y Runtime Mask Check (Quick A/B, seed 11)

Run root:
- `/tmp/beta55_option_y_newpack_compare_quick_20260225/option_y_newpack_compare_quick/option_y_newpack_compare_quick`

Result:
- No-mask and legacy Day-8 mask have identical LR outcome in quick run.
- Mask **degrades Bedroom** in quick run:
  - Bedroom occupied F1: 0.7462 (no mask) -> 0.7249 (masked)
  - Bedroom sleep MAE: 37.91 -> 59.81 minutes

Conclusion:
- Legacy Day-8 runtime mask should be removed for the new pack.

## Canonical Full Run (3 seeds, no mask)

Run root:
- `/tmp/beta55_option_y_newpack_anchor_full_20260225/option_y_newpack_anchor_full/option_y_newpack_anchor_full`

Ranking:
- `anchor_top2_frag_v3`: eligible `48/60`, LR eligible passes `0/12`, status `fail`

Current Option Y go/no-go result:
- Blocking: `livingroom_eligible_pass_count_min`
- Informational-only failures:
  - `livingroom_episode_recall_min`
  - `livingroom_episode_f1_min`
  - `day7_livingroom_episode_recall_min`

Everything else passes under current Option Y policy + MAE guards.

## Updated Performance Snapshot (new pack)

Classification means:
- Bedroom: accuracy 0.7339, macro_f1 0.6401, occupied_f1 0.7197
- LivingRoom: accuracy 0.6901, macro_f1 0.5356, occupied_f1 0.2884
- Kitchen: accuracy 0.8978, macro_f1 0.8253, occupied_f1 0.7219
- Bathroom: accuracy 0.9038, macro_f1 0.6781, occupied_f1 0.5204

Timeline MAE means:
- Bedroom sleep MAE: 46.69 min
- LivingRoom active MAE: 123.81 min
- Kitchen use MAE: 56.52 min
- Bathroom use MAE: 50.19 min
- Entrance out MAE: 55.77 min

## What Changed vs Previous Conclusion

Compared with the prior corrected_clones-based Option Y run:
- Bedroom improved substantially (classification and sleep MAE).
- LivingRoom degraded substantially (occupied F1/recall/precision), causing LR eligible pass count to collapse to 0/12.

## Honest Recommendation

1. Remove legacy Day-8 Bedroom runtime mask from standard evaluation on the new pack.
2. Keep Option Y episode checks informational-only (already implemented).
3. Product decision needed for LivingRoom hard-block policy:
   - If `livingroom_eligible_pass_count_min >= 3` remains blocking, current new-pack run fails.
   - If LR hard-block is set informational (or pass-count min=0 for Beta 5.5), current run passes.
4. Keep MAE non-regression guards active to prevent timeline regressions.

Policy simulation on same 3-seed artifacts:
- Current policy: fail (blocking: `livingroom_eligible_pass_count_min`)
- `livingroom_eligible_pass_count_min=0`: pass
- Mark `livingroom_eligible_pass_count_min` informational: pass

## Sensor Calibration / Future Compatibility

If sensor calibration improves while keeping current schema (same columns/rates), Beta 5.5 can absorb it via retrain/recalibration and threshold updates.

If new sensor channels are added, Beta 5.5 needs targeted feature wiring and config updates, but not a full architecture rewrite.
