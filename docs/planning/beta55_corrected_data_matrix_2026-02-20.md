# Beta 5.5 Corrected-Data Matrix (2026-02-20)

## Scope

- Data dir (corrected): `/Users/dicksonng/DT/Development/New training files/corrected_clones`
- Elder: `HK0011_jessica`
- Days: `4..8`
- Seeds: `11,22,33`
- Shared flags:
  - `--enable-room-temporal-occupancy-features`
  - `--enable-bedroom-light-texture-features`
  - `--enable-bedroom-livingroom-stage-a-hgb`
  - `--enable-bedroom-livingroom-segment-mode`
  - `--hard-gate-min-train-days 3`

## Smoke Test (Label Ingest Verification)

### Command note
`run_event_first_backtest.py` requires at least 2 days. Smoke run used `min_day=6,max_day=7`.

### Evidence
- Corrected run:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/corrected_matrix_20260220/smoke_corrected_min6max7_seed11.json`
- Uncorrected comparator:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/corrected_matrix_20260220/smoke_uncorrected_min6max7_seed11.json`

### Day-7 LivingRoom GT occupancy
- Uncorrected: `77.17 min` (`5.36%`)
- Corrected: `534.95 min` (`37.17%`)

Interpretation: corrected labels are being consumed by the backtest loader.

## Matrix Variants

- `variantA_baseline`: corrected Candidate D baseline (segment mode + Stage-A HGB)
- `variantE_learned055`: baseline + learned segment classifier, `confidence_floor=0.55`
- `variantE_learned030`: baseline + learned segment classifier, `confidence_floor=0.30`
- `variantB_minute_global`: minute-grid Stage-A for Bedroom+LivingRoom
- `variantC_minute_global_plus`: minute-grid global + occupancy decoder + prediction smoothing
- `variantB_minute_lr_only`: minute-grid Stage-A for LivingRoom only
- `variantC_minute_lr_only_plus`: minute-grid LR-only + occupancy decoder + prediction smoothing

## Results

| Variant | Eligible Hard-Gate | Full Hard-Gate | LR recall | LR F1 | BR recall | BR F1 | Day7 LR recall | LR eligible pass | BR eligible pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| variantA_baseline | 24/30 | 45/60 | 0.6628 | 0.6110 | 0.4361 | 0.4241 | 0.9069 | 3/6 | 3/6 |
| variantE_learned055 | 24/30 | 45/60 | 0.6628 | 0.6110 | 0.4361 | 0.4241 | 0.9069 | 3/6 | 3/6 |
| variantE_learned030 | 24/30 | 45/60 | 0.6628 | 0.6110 | 0.4361 | 0.4241 | 0.9069 | 3/6 | 3/6 |
| variantB_minute_global | 21/30 | 42/60 | 0.6502 | 0.6126 | 0.3542 | 0.3605 | 0.8678 | 3/6 | 0/6 |
| variantC_minute_global_plus | 21/30 | 42/60 | 0.6524 | 0.6133 | 0.3568 | 0.3638 | 0.8656 | 3/6 | 0/6 |
| variantB_minute_lr_only | 24/30 | 45/60 | 0.6502 | 0.6126 | 0.4361 | 0.4241 | 0.8678 | 3/6 | 3/6 |
| variantC_minute_lr_only_plus | 24/30 | 45/60 | 0.6524 | 0.6133 | 0.4353 | 0.4240 | 0.8656 | 3/6 | 3/6 |

## Key Takeaways

1. Corrected labels improved the baseline matrix from prior `21/30` family to `24/30` eligible.
2. Learned segment classifier still shows no measurable uplift (`0.55` vs `0.30` identical to baseline).
3. Global minute-grid remains harmful due Bedroom regression (`BR 3/6 -> 0/6`).
4. LivingRoom-only minute-grid isolation removes Bedroom regression and preserves `24/30`.
5. Day-7 LivingRoom recall is strongly recovered (`~0.87-0.91` across variants), above the proposed `>=0.45` threshold.

## Artifacts Root

- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/corrected_matrix_20260220`

