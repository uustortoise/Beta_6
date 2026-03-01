# Beta 5.5 Stage-A Minute-Grid Pilot (2026-02-20)

## Objective

Evaluate whether Bedroom/LivingRoom Stage-A minute-grid grouping improves WS6 hard-gate outcomes versus the current 10-second Stage-A path.

## Scope

- Data: `/Users/dicksonng/DT/Development/New training files`
- Elder: `HK0011_jessica`
- Window: day 4 to day 8
- Seeds: `11, 22, 33`
- Hard-gate eligibility: `--hard-gate-min-train-days 3`
- Shared baseline toggles:
  - `--enable-room-temporal-occupancy-features`
  - `--enable-bedroom-light-texture-features`
  - `--enable-bedroom-livingroom-stage-a-hgb`
  - `--enable-bedroom-livingroom-segment-mode`

## Variants

- Variant A (`variantA_baseline`): current 10s Stage-A path.
- Variant B (`variantB_minute`): enable minute-grid grouping:
  - `--enable-bedroom-livingroom-stage-a-minute-grid`
  - `--bedroom-livingroom-stage-a-group-seconds 60`
  - `--bedroom-livingroom-stage-a-group-occupied-ratio-threshold 0.50`
- Variant C (`variantC_minute_plus`): Variant B plus stronger temporal post-processing:
  - `--enable-bedroom-livingroom-occupancy-decoder`
  - `--enable-bedroom-livingroom-prediction-smoothing`

## Artifacts

- Root:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/stagea_minute_pilot_20260220`
- Aggregates:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/stagea_minute_pilot_20260220/variantA_baseline/ws6_rolling.json`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/stagea_minute_pilot_20260220/variantB_minute/ws6_rolling.json`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/stagea_minute_pilot_20260220/variantC_minute_plus/ws6_rolling.json`

## Results

| Variant | Eligible Hard-Gate Passed | Eligible Total | Eligible Split Pass Rate | Full Passed | Full Total |
|---|---:|---:|---:|---:|---:|
| A (10s baseline) | 21 | 30 | 0.70 | 39 | 60 |
| B (minute-grid) | 18 | 30 | 0.60 | 36 | 60 |
| C (minute-grid + decoder+smoothing) | 18 | 30 | 0.60 | 36 | 60 |

LivingRoom classification summary (mean):

| Variant | `occupied_recall` | `occupied_f1` |
|---|---:|---:|
| A | 0.2792 | 0.2623 |
| B | 0.3107 | 0.2990 |
| C | 0.3132 | 0.3027 |

Bedroom classification summary (mean):

| Variant | `occupied_recall` | `occupied_f1` |
|---|---:|---:|
| A | 0.4361 | 0.4241 |
| B | 0.3542 | 0.3605 |
| C | 0.3568 | 0.3638 |

## Interpretation

- Minute-grid variants improve LivingRoom occupied recall/F1 but regress Bedroom substantially.
- Net WS6 hard-gate result regresses from `21/30` to `18/30`.
- Under this setup, minute-grid Stage-A should remain default-off.

## Recommendation

- Keep 10s Stage-A as default for now.
- Continue with:
  - label-correction ingestion loop,
  - feature enrichment,
  - hard-negative/failure-replay strategy,
  - targeted LivingRoom-specific tuning.
- If retrying minute-grid, gate it by room-regime/condition (not global BL-on) and re-test against the same 3-seed WS6 matrix.
