# Beta 5.5 Pre-Arrival Readiness (2026-02-21)

## Objective
Prepare Beta 5.5 so the new corrected 4-10 Dec label pack can be validated, smoke-tested, and evaluated with full matrix + signoff in one operational cycle.

## Current Anchor
- Anchor variant: `top2_frag_v3`
- Branch: `beta-5.5-transformer`
- Matrix shape target: 3 seeds x rolling splits over selected day window
- Canonical elder for first pass: `HK0011_jessica`

## Current Baseline Snapshot (Captured February 21, 2026)
Reference run for before/after analysis when the new 4-10 Dec pack arrives.

### Snapshot Inputs
- Data path: `/Users/dicksonng/DT/Development/New training files/corrected_clones`
- Matrix profile: `pre_arrival_quick`
- Variant used as baseline snapshot: `anchor_top2_frag_v3`
- Seed(s): `11` (quick rehearsal only)
- Artifacts:
  - `/tmp/beta55_prearrival_matrix_live/pre_arrival_quick/anchor_top2_frag_v3/rolling.json`
  - `/tmp/beta55_prearrival_matrix_live/pre_arrival_quick/anchor_top2_frag_v3/signoff.json`
  - `/tmp/beta55_prearrival_matrix_live/pre_arrival_quick/manifest.json`

### Current Room Metrics (Anchor: `top2_frag_v3`)
| Room | Accuracy | Macro F1 | Occupied F1 | Occupied Recall | Fragmentation Score |
|---|---:|---:|---:|---:|---:|
| Bedroom | 0.6383 | 0.4877 | 0.4140 | 0.4393 | 0.4784 |
| LivingRoom | 0.7082 | 0.6900 | 0.6624 | 0.7271 | 0.5272 |
| Kitchen | 0.8894 | 0.8252 | 0.7294 | 0.7201 | 0.3207 |
| Bathroom | 0.9658 | 0.6614 | 0.6149 | 0.4749 | 0.6745 |
| Entrance | 0.9580 | 0.8215 | 0.0000 | 0.0000 | 0.6667 |

### Current Gate Snapshot (Anchor)
- Eligible hard-gate passes: `16/20`
- Full hard-gate passes (including ineligible): `23/30`
- LivingRoom eligible passes: `2/4`
- Bedroom eligible passes: `2/4`
- Go/no-go blocking checks:
  - `overall_eligible_pass_count_min` (`16 < 24`)
  - `livingroom_eligible_pass_count_min` (`2 < 3`)
  - `bedroom_max_regression_splits` (`2 > 1`)
  - `day8_livingroom_fragmentation_min` (`0.30 < 0.45`)

### Note on Quick-Rehearsal Signoff
- `signoff.json` gate decision is `FAIL` in this snapshot partly because `pre_arrival_quick` runs only seed `11`, while strict split-seed matrix expects full `11/22/33`.
- Use this snapshot as runtime metric baseline; promotion decisions must use full profile (`pre_arrival_full`).

## Promotion-Grade Baseline Snapshot (Full 3-Seed Anchor)
This is the canonical baseline for after-adoption analysis against the incoming corrected label pack.

### Snapshot Inputs
- Data path: `/Users/dicksonng/DT/Development/New training files/corrected_clones`
- Matrix profile: `pre_arrival_full`
- Anchor variant: `anchor_top2_frag_v3`
- Seeds: `11,22,33`
- Anchor artifacts:
  - `/tmp/beta55_prearrival_matrix_full/pre_arrival_full/anchor_top2_frag_v3/rolling.json`
  - `/tmp/beta55_prearrival_matrix_full/pre_arrival_full/anchor_top2_frag_v3/signoff.json`
  - `/tmp/beta55_prearrival_matrix_full/pre_arrival_full/anchor_top2_frag_v3/seed_11.json`
  - `/tmp/beta55_prearrival_matrix_full/pre_arrival_full/anchor_top2_frag_v3/seed_22.json`
  - `/tmp/beta55_prearrival_matrix_full/pre_arrival_full/anchor_top2_frag_v3/seed_33.json`

### Full 3-Seed Room Metrics (Anchor)
| Room | Accuracy | Macro F1 | Occupied F1 | Occupied Recall | Fragmentation Score |
|---|---:|---:|---:|---:|---:|
| Bedroom | 0.6044 | 0.4765 | 0.4178 | 0.4943 | 0.4864 |
| LivingRoom | 0.7049 | 0.6844 | 0.6626 | 0.7296 | 0.5073 |
| Kitchen | 0.8912 | 0.8262 | 0.7297 | 0.7083 | 0.3445 |
| Bathroom | 0.9673 | 0.6761 | 0.6264 | 0.4904 | 0.6717 |
| Entrance | 0.9606 | 0.8223 | 0.0000 | 0.0000 | 0.6667 |

### Full 3-Seed Gate Snapshot (Anchor)
- Strict split-seed stability: `hard_gate_splits_passed=48/60` (eligible), `68/90` (full)
- Configured go/no-go counters: `passed_eligible=48/60`, LivingRoom `7/12`, Bedroom `5/12`
- Configured go/no-go blocking reasons:
  - `bedroom_max_regression_splits` (`7 > 1`)
  - `day8_bedroom_sleep_recall_min` (`0.3419 < 0.40`)
  - `day8_livingroom_fragmentation_min` (`0.2000 < 0.45`)
- Signoff gate decision (aggregate): `FAIL`

## Active Model/Runner Config Snapshot
Configuration locked for pre-arrival evaluation reproducibility.

### Matrix Defaults (`backend/config/event_first_matrix_profiles.yaml`)
- `min_day=4`, `max_day=10`
- Seeds default: `11,22,33`
- `occupancy_threshold=0.35`
- Calibration: `isotonic`, `calib_fraction=0.20`
- `min_calib_samples=500`, `min_calib_label_support=30`
- `comparison_window=dec4_to_dec10`

### Anchor Variant Args (`anchor_top2_frag_v3`)
- Segment mode enabled for Bedroom/LivingRoom
- Bedroom segment params: `min_duration=30s`, `gap_merge=90s`, `min_activity_prob=0.30`
- LivingRoom segment params: `min_duration=60s`, `gap_merge=30s`, `min_activity_prob=0.40`
- Prediction smoothing enabled:
  - `min_run_windows=9`
  - `gap_fill_windows=6`
- Room temporal occupancy features enabled
- Texture profile: `mixed`
- Stage A HGB enabled for Bedroom/LivingRoom
- Hard-negative mining enabled (`weight=2.0`)
- Failure replay enabled (`weight=1.5`, `max_rows_per_day=1800`)
- LivingRoom occupied sample weight: `2.5`
- Hard-gate minimum train days: `3`

### Go/No-Go Thresholds (`backend/config/event_first_go_no_go.yaml`)
- Overall eligible pass count: `>=24`
- LivingRoom eligible pass count: `>=3`
- Bedroom max regression splits: `<=1`
- Day-7 LivingRoom occupied recall: `>=0.45`
- Day-8 Bedroom sleep recall: `>=0.40`
- Day-8 LivingRoom fragmentation: `>=0.45`

### Smoke Thresholds (`backend/config/event_first_go_no_go.yaml`)
- Require continuity audit: `true`
- Require label correction summary: `true`
- Minimum changed minutes evidence: `30.0`
- Minimum LivingRoom occupied rate on day 7: `0.20`

## Baseline Contract (Promotion-Grade)
- Baseline identifier required in signoff context.
- `baseline_artifact_hash` should be a verified `sha256:*` hash.
- Baseline artifact path should be captured and hash-verified before promotion-grade signoff.
- For pre-arrival dry runs, baseline binding can be temporarily disabled in matrix profile.

## Canonical Commands
### 1) Validate incoming label pack
```bash
python3 backend/scripts/validate_label_pack.py \
  --pack-dir "/path/to/candidate_pack" \
  --elder-id HK0011_jessica \
  --min-day 4 --max-day 10 \
  --output /tmp/label_pack_validation.json
```

### 2) Diff baseline vs incoming labels
```bash
python3 backend/scripts/diff_label_pack.py \
  --baseline-dir "/path/to/baseline_pack" \
  --candidate-dir "/path/to/candidate_pack" \
  --elder-id HK0011_jessica \
  --min-day 4 --max-day 10 \
  --json-output /tmp/label_pack_diff.json \
  --csv-output /tmp/label_pack_diff.csv
```

### 3) Smoke run
```bash
python3 backend/scripts/run_event_first_smoke.py \
  --data-dir "/path/to/candidate_pack" \
  --elder-id HK0011_jessica \
  --day 7 --seed 11 \
  --expectation-config backend/config/event_first_go_no_go.yaml \
  --diff-report /tmp/label_pack_diff.json \
  --output /tmp/event_first_smoke.json
```

### 4) Full matrix
```bash
python3 backend/scripts/run_event_first_matrix.py \
  --profiles-yaml backend/config/event_first_matrix_profiles.yaml \
  --profile pre_arrival_full \
  --data-dir "/path/to/candidate_pack" \
  --elder-id HK0011_jessica \
  --output-dir /tmp/event_first_matrix \
  --max-workers 3
```

### 5) Before/after summary
```bash
python3 backend/scripts/summarize_before_after.py \
  --before-rolling /tmp/before_rolling.json \
  --before-signoff /tmp/before_signoff.json \
  --after-rolling /tmp/after_rolling.json \
  --after-signoff /tmp/after_signoff.json \
  --markdown-output /tmp/before_after.md \
  --csv-output /tmp/before_after.csv
```

## Gate Expectations
- Keep both views in decision pack:
  - Eligible-only hard-gate pass counts
  - Full-matrix hard-gate pass counts (including ineligible visibility)
- Day-specific checks are tracked in go/no-go config and smoke checks.

## Locked Decisions
- Stay on Beta 5.5 mainline.
- Keep global minute-grid disabled.
- Keep learned segment classifier as non-default control variant only.
- Keep event-first in controlled evaluation flow until gates pass.

## Phase Update (2026-02-24): Episode Metrics + Passive Hysteresis
Implementation and execution record:
- `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/beta55_passive_hysteresis_phase_execution_2026-02-24.md`

Newly added runtime controls (all default OFF):
- `--enable-bedroom-livingroom-passive-hysteresis`
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

A/B result snapshot (seed 11 only, latest 4-10 Dec pack):
- Anchor (`top2_frag_v3`) vs Anchor + passive hysteresis defaults
- Outcome: passive hysteresis defaults over-hold occupancy and reduce Bedroom/LivingRoom occupied F1.
- Before/after artifact:
  - `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/beta55_passive_hysteresis_before_after_seed11_2026-02-24.csv`

Operational note:
- Smoke gate still fails on Day-7 LivingRoom occupied-rate floor (`0.20`) using current pack and policy.
- Smoke artifact:
  - `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/beta55_passive_hysteresis_smoke_2026-02-24.json`
