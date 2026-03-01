# Beta 5.5 LR Sweep Follow-up (2026-02-24)

## 1) Full LR Fragmentation Sweep (3 seeds, completed via fallback)
- Manifest: `/tmp/beta55_lr_frag_sweep_clean_20260224_212051/lr_fragmentation_sweep/clean_sweep_manifest.json`
- Result: all 4 variants failed; no LivingRoom eligible pass improvement.

| Variant | Eligible Passed | LR Eligible Passed | Blocking Reasons |
|---|---:|---:|---|
| anchor_top2_frag_v3 | 48/60 | 0 | livingroom_eligible_pass_count_min |
| lr_frag_focus_v1 | 48/60 | 0 | livingroom_eligible_pass_count_min |
| lr_frag_focus_v2 | 48/60 | 0 | livingroom_eligible_pass_count_min |
| lr_frag_focus_v3 | 48/60 | 0 | livingroom_eligible_pass_count_min |

## 2) LR Occupied-F1 Quick Sweep (seed=11)
- Manifest: `/tmp/beta55_lr_occ_sweep_quick_20260224_real/lr_occupied_f1_sweep_quick/clean_sweep_manifest.json`
- Matrix child stalled; fallback ran anchor/v1/v2. v3 remained runtime-nonviable and was terminated.

| Variant | Status | Eligible Passed | LR Eligible Passed | Blocking Reasons |
|---|---|---:|---:|---|
| anchor_top2_frag_v3 | fail | 16/20 | 0 | overall_eligible_pass_count_min;livingroom_eligible_pass_count_min |
| lr_occ_focus_v1 | fail | 16/20 | 0 | overall_eligible_pass_count_min;livingroom_eligible_pass_count_min;day7_livingroom_recall_min;day8_livingroom_fragmentation_min |
| lr_occ_focus_v2 | fail | 16/20 | 0 | overall_eligible_pass_count_min;livingroom_eligible_pass_count_min;day7_livingroom_recall_min;day8_livingroom_fragmentation_min |
| lr_occ_focus_v3 | failed | 0/0 | 0 | missing_seed_reports |

### Soft-metric delta (seed=11, LivingRoom)
| Variant | occupied_precision | occupied_recall | occupied_f1 | day7 LR recall (min split) | day8 LR fragmentation (min split) |
|---|---:|---:|---:|---:|---:|
| anchor_top2_frag_v3 | 0.2669 | 0.4518 | 0.2912 | 0.6850 | 0.9643 |
| lr_occ_focus_v1 | 0.5471 | 0.2071 | 0.2323 | 0.2326 | 0.3571 |
| lr_occ_focus_v2 | 0.5492 | 0.2042 | 0.2290 | 0.2326 | 0.3571 |

## 3) Interpretation
- Fragmentation tuning improved some soft LR metrics but did not move any LR hard-gate pass (still 0/12 in full sweep).
- Precision-oriented tuning (v1/v2) increased LR precision, but collapsed LR recall and added new policy failures (day7 recall + day8 fragmentation).
- Net: current blocker remains LR hard-gate passability, not tooling/execution.

## 4) Recommendation
- Do not run full 3-seed occupied-F1 sweep for v1/v2 (quick screen already dominated by anchor).
- Keep `top2_frag_v3` as anchor while pursuing data/label semantics alignment or gate-policy recalibration for LivingRoom under corrected pack.
- Treat heavy `lr_occ_focus_v3` sequence candidate as out-of-scope for current Beta 5.5 runtime envelope.
