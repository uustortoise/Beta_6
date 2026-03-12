# Beta6 Bedroom Upstream Lineage First-Divergence

## Scope

- Reconstruct the earliest upstream lineage difference between live `Bedroom_v38` and candidate `Bedroom_v40`.
- Stop at the first stage that already explains the later pre-sampling class-mix split.
- Do not propose a sampling or calibration fix unless the divergence appears later than source selection.

## Inputs

Primary artifacts:

- `dev_history.log`
- `backend/models/HK0011_jessica/Bedroom_v38_decision_trace.json`
- `backend/models/HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z/Bedroom_v40_decision_trace.json`
- `tmp/jessica_bedroom_sepfix_20260311T041856Z/build_combined_corrected_pack.py`
- `tmp/jessica_bedroom_sepfix_20260311T041856Z/combined_corrected_pack.parquet`

Diagnostic reconstruction used in this thread:

- `utils.data_loader.load_sensor_data(..., resample=False)` on the corrected Jessica workbooks
- raw Bedroom row counts by file / date / label
- saved pre-sampling counts from each room decision trace

## Finding 1: the first strong divergence is already at raw source selection

The most precise historical record for the support-fix Bedroom retrain is the command log in `dev_history.log`, not the higher-level summary sentence.

Precise command evidence for `Bedroom_v38`:

- `UnifiedPipeline.train_from_files(... rooms={'Bedroom'} ...)` against:
  - `HK0011_jessica_train_10dec2025.xlsx`
  - `HK0011_jessica_train_17dec2025.xlsx`

Planned / executed source evidence for `Bedroom_v40`:

- `docs/plans/2026-03-11-beta6-bedroom-separation-retrain.md` lists:
  - `HK0011_jessica_train_4dec2025.xlsx`
  - `HK0011_jessica_train_5dec2025.xlsx`
  - `HK0011_jessica_train_6dec2025.xlsx`
  - `HK0011_jessica_train_7dec2025.xlsx`
  - `HK0011_jessica_train_8dec2025.xlsx`
  - `HK0011_jessica_train_9dec2025.xlsx`
  - `HK0011_jessica_train_10dec2025.xlsx`
  - `HK0011_jessica_train_17dec2025.xlsx`
- `tmp/jessica_bedroom_sepfix_20260311T041856Z/build_combined_corrected_pack.py` materializes that exact eight-file corrected pack.

That means the two Bedroom models were not trained from the same raw workbook lineage:

- `v38`: Dec 10 + Dec 17 only
- `v40`: Dec 4-10 + Dec 17

## Finding 2: the raw Bedroom label topology is already in the two different regimes

Raw Bedroom rows reconstructed from the corrected workbooks:

| Stage | `v38` source lineage | `v40` source lineage |
| --- | --- | --- |
| Workbook dates | `2025-12-10`, `2025-12-17` | `2025-12-04..2025-12-10`, `2025-12-17` |
| Raw Bedroom rows | `14,604` | `61,552` |
| `bedroom_normal_use` | `1,276` (`8.74%`) | `9,048` (`14.70%`) |
| `sleep` | `5,977` (`40.93%`) | `19,630` (`31.89%`) |
| `unoccupied` | `7,351` (`50.34%`) | `32,874` (`53.41%`) |

The extra Dec 4-9 days are not neutral volume. They move the Bedroom source population toward:

- much more `bedroom_normal_use`
- materially less `sleep`
- slightly more `unoccupied`

This is the same directional regime shift later seen in the saved training traces.

## Finding 3: the saved pre-sampling traces mostly inherit the raw-pack shift

Saved decision-trace pre-sampling counts:

| Stage | `v38` | `v40` |
| --- | --- | --- |
| `total_sequences_pre_split` | `17,102` | `68,937` |
| `samples_before_downsample` | `14,255` | `55,342` |
| `bedroom_normal_use` pre-sampling | `1,288` (`9.04%`) | `8,954` (`16.18%`) |
| `sleep` pre-sampling | `5,808` (`40.74%`) | `17,437` (`31.51%`) |
| `unoccupied` pre-sampling | `7,159` (`50.22%`) | `28,951` (`52.31%`) |

Two important implications:

1. The scale jump in `total_sequences_pre_split` tracks the raw-row jump closely.
   - raw rows: `14,604 -> 61,552` (`4.21x`)
   - pre-split sequences: `17,102 -> 68,937` (`4.03x`)
2. The class-share shift is already present before split optimization, minority sampling, or threshold calibration.
   - `bedroom_normal_use`: `8.74% -> 14.70%` raw, then `9.04% -> 16.18%` pre-sampling
   - `sleep`: `40.93% -> 31.89%` raw, then `40.74% -> 31.51%` pre-sampling

So the later pre-sampling divergence is not primarily created by calibration or sampling. Those later stages act on an already different source population.

## Finding 4: later policy differences exist, but they are downstream of the first break

The Bedroom runs also differ in policy hash and sampling posture:

- `v38` policy hash: `a619de5a17140418527ae69529c5ef80f052f4f92951e00fd98c4f275254a078`
- `v40` policy hash: `165ffb6442a074664c7c3cf6456d4291e9871f8b7902471d79bf80f172799c09`

Examples:

- `v38` enabled Bedroom transition-focus sampling and minority expansion
- `v40` disabled both by setting the Bedroom multipliers to `1`

Those differences still matter for downstream behavior, but they are not the earliest lineage split. The models are already in different data regimes before those later stages run.

## Conclusion

The first irreversible divergence appears at **stage 1: raw source label topology / source workbook selection**.

Why:

- `Bedroom_v38` was trained from only the corrected Dec 10 + Dec 17 Bedroom data.
- `Bedroom_v40` was trained from the corrected Dec 4-10 + Dec 17 Bedroom data.
- That source-lineage change already shifts Bedroom toward higher `bedroom_normal_use` share and lower `sleep` share.
- The saved pre-sampling training distributions closely preserve that same shift.

So the earliest confirmed root-cause boundary is **before** sequence generation, split selection, sampling, checkpoint selection, or calibration.

## Remaining Unknown

What is not yet separated is the downstream contribution of the later policy change once source lineage is matched.

Specifically:

- how much of the final Dec 17 regression comes from the larger Dec 4-10 + Dec 17 pack itself
- versus the Bedroom-only sampling rollback that `v40` applied on top of that pack

## Recommended Next Step

Run one matched-lineage Bedroom comparison before changing policy:

1. either rerun the `v40` Bedroom-only rollback policy on the exact `v38` source pair (`10dec + 17dec`)
2. or rerun the `v38` policy posture on the full `v40` eight-file corrected pack

That will isolate whether the larger source pack alone is enough to move Bedroom into the bad regime, or whether the later sampling-policy rollback is required on top of it.

## Permanent Process Fix

Regardless of the next experiment, persist the exact room-level source manifest into saved training metadata (`decision_trace` / versions / train-metrics output).

This thread had conflicting human-written summaries, and the precise command log plus reconstructed raw counts were required to recover the true lineage.
