# Beta 5.5 LivingRoom Fragmentation Sweep (2026-02-20)

## Goal

Resolve remaining LivingRoom Day-8 `fragmentation_score_lt_0.450` failures after corrected-data anchor (`24/30`).

## Sweep Design

- Seed-11 shortlist sweep:
  - root: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/fragmentation_sweep_seed11_20260220`
- Shared flags:
  - `--enable-room-temporal-occupancy-features`
  - `--enable-bedroom-light-texture-features`
  - `--enable-bedroom-livingroom-stage-a-hgb`
  - `--enable-bedroom-livingroom-segment-mode`
  - `--hard-gate-min-train-days 3`

Variants tested (seed 11):
- `frag_v1_more_runs`
- `frag_v2_more_runs_aggr`
- `frag_v3_hg_less_smooth`
- `frag_v4_combo1`
- `frag_v5_combo2`
- `frag_v6_pred_smooth`

## Seed-11 Shortlist Result

- Baseline Day-8 LivingRoom fragmentation score: `0.30` (fail)
- Best improvement signal:
  - `frag_v3_hg_less_smooth` (LivingRoom hard-gate smoothing from `9/6` to `6/3`) -> Day-8 fragmentation `0.60` (pass)
  - `frag_v6_pred_smooth` also passed Day-8 LR fragmentation in seed 11.

## 3-Seed Validation (Top 2)

Root:
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/fragmentation_top2_3seed_20260220`

Variants:
- `top2_frag_v3_hg_less_smooth`
- `top2_frag_v6_pred_smooth`

### Aggregate outcomes

| Variant | Eligible Hard-Gate | Full Hard-Gate | Delta vs Anchor |
|---|---:|---:|---:|
| Anchor (`variantA_baseline`) | 24/30 | 45/60 | baseline |
| `top2_frag_v3_hg_less_smooth` | **26/30** | **47/60** | **+2 eligible, +2 full** |
| `top2_frag_v6_pred_smooth` | **26/30** | **47/60** | **+2 eligible, +2 full** |

### Day-8 LivingRoom pass pattern

- Anchor: fail in all seeds (`3/3` fails)
- Top-2 variants: pass in `2/3` seeds, fail in seed 22 due fragmentation (`0.30-0.40`)

## Recommended Config

Promote `top2_frag_v3_hg_less_smooth` as next candidate (simpler change, no added prediction smoothing dependency):

- `--hard-gate-fragmentation-min-run-windows "bedroom=9,livingroom=6"`
- `--hard-gate-fragmentation-gap-fill-windows "bedroom=6,livingroom=3"`

This keeps model behavior stable while improving gate alignment for LivingRoom Day-8.

