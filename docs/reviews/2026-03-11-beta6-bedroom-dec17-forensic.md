# Beta6 Bedroom Dec 17 Forensic

## Scope

- Explain why `Bedroom_v38` still lands around `0.35-0.39` macro-F1 on Dec 17 after the support-gating fix.
- Keep this separate from the LivingRoom `v52` work.
- Do not treat the older support blocker as open again unless new evidence contradicts `Bedroom_v38`.

## Canonical benchmark path

The tracked Bedroom benchmark on this branch is still the March 11 repaired live replay:

- `tmp/jessica_17dec_eval_live_livingroom_rootfix_registryfix_20260311/final/comparison/summary.json`
- Bedroom final macro-F1: `0.3511152121985108`

Why this is the canonical reference:

- it is the branch-tracked replay cited in `docs/reviews/2026-03-11-beta6-livingroom-root-fix-handoff.md`
- it uses the richer comparison harness that emits:
  - `per_room_raw_top1`
  - `per_room_raw_export`
  - `per_room_final`
  - `all_rooms_merged.parquet` with confidence / acceptance-score fields

Why the later `0.3900` recheck is not parity-safe:

- `docs/reviews/2026-03-11-beta6-livingroom-fresh-retrain-replay.md` explicitly treats it as a local same-harness recheck for candidate-vs-live deltas
- its merged parquet only carries `timestamp`, `truth_label`, `predicted_label`, and `room`
- it does not preserve the raw-top1 / raw-export / confidence fields needed to compare against the historical benchmark harness

## Holdout state of `Bedroom_v38`

`Bedroom_v38` itself is not a repeat of the old all-`sleep` collapse.

From `backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z/Bedroom_v38_decision_trace.json`:

- holdout macro-F1: `0.6257134795187449`
- predicted distribution:
  - `bedroom_normal_use=712`
  - `sleep=812`
  - `unoccupied=1323`
- per-label recall:
  - `bedroom_normal_use=0.5136`
  - `sleep=0.8196`
  - `unoccupied=0.6540`
- release gate: `pass=true`
- runtime two-stage path: rejected in validation (`two_stage_validation.macro_f1=0.4543`), so `runtime_use_two_stage=false`

Interpretation:

- the saved candidate is healthy enough on holdout to pass promotion gates
- the Dec 17 weakness is not the old support bug and not a fresh all-class collapse inside training

## Raw-vs-final Dec 17 loss

Using the canonical replay parquet
`tmp/jessica_17dec_eval_live_livingroom_rootfix_registryfix_20260311/final/comparison/all_rooms_merged.parquet`:

- Bedroom raw top-1 macro-F1: `0.43865005859309375`
- Bedroom raw export macro-F1: `0.35107096454067005`
- Bedroom final macro-F1: `0.3511152121985108`
- raw top-1 -> raw export rewrite count: `11`
- raw export -> final rewrite count: `5`
- low-confidence rate: `0.0013002364066193853`
- runtime-unknown rate: `0.0`

Dominant raw top-1 -> final rewrites:

- `bedroom_normal_use -> low_confidence`: `11`
- `bedroom_normal_use -> sleep`: `3`
- `sleep -> bedroom_normal_use`: `2`

Interpretation:

- almost the entire Dec 17 loss is already present before the final runtime pass
- the last confidence / Beta6 rewrite step is negligible here
- thresholding adds only a small metric penalty because `low_confidence` becomes an extra predicted label, but it does not explain the main Bedroom failure mode

## Main confusion pattern

Bedroom raw top-1 confusion on the canonical replay is dominated by `unoccupied -> bedroom_normal_use`, not by `sleep <-> bedroom_normal_use`.

Three-label raw top-1 metrics on Dec 17:

- `bedroom_normal_use`
  - precision `0.1669`
  - recall `0.9868`
  - F1 `0.2856`
- `sleep`
  - precision `0.9687`
  - recall `0.9664`
  - F1 `0.9675`
- `unoccupied`
  - precision `0.9914`
  - recall `0.3357`
  - F1 `0.5015`

Key confusion counts:

- true `sleep` predicted as `bedroom_normal_use`: `114`
- true `unoccupied` predicted as `bedroom_normal_use`: `2858`

Interpretation:

- `sleep` is not the main problem on Dec 17
- the main failure is occupied-vs-unoccupied separation, specifically an over-active `bedroom_normal_use` frontier

## Label-geometry check

If workbook label geometry were the main cause, the worst errors should cluster near label boundaries. They do not.

For the `2858` raw top-1 errors where true `unoccupied` was predicted as `bedroom_normal_use`:

- within `5` rows of a truth label change: `5.46%`
- within `10` rows: `9.52%`
- median distance to nearest truth label change: `76` rows
- 90th percentile distance: `244.3` rows

Interpretation:

- most wrong `unoccupied -> bedroom_normal_use` rows occur well inside long truth segments
- Dec 17 workbook boundary geometry is therefore a weak primary-cause candidate

## Threshold / confidence geometry

Thresholds are not suppressing many otherwise-usable Bedroom predictions.

From `Bedroom_v38_decision_trace.json`:

- `bedroom_normal_use` threshold: `0.17066459956573105`
- `sleep` threshold: `0.8892368714230562`
- `unoccupied` threshold: `0.7697282402092486`

From the canonical replay parquet:

- almost all raw top-1 `bedroom_normal_use` rows have activity-acceptance score about `0.20092074`
- this is true for both correct and incorrect `bedroom_normal_use` predictions
- only `11` Bedroom rows fell below the `bedroom_normal_use` threshold and were rewritten to `low_confidence`

Correct vs incorrect raw top-1 `bedroom_normal_use` score geometry:

- correct rows: median acceptance score `0.20092074482593314`
- incorrect rows: median acceptance score `0.20092073949781653`

Interpretation:

- the acceptance-score geometry for `bedroom_normal_use` is nearly flat
- recalibration-only threshold tuning has very little discriminatory leverage
- runtime confidence policy is not the main source of the Dec 17 loss

## Root-cause hypothesis

The best-supported current hypothesis is:

- `Bedroom_v38` no longer has a support-gating problem
- the main Dec 17 loss is not caused by late runtime rewrites
- the main Dec 17 loss is also not primarily `sleep` vs `bedroom_normal_use`
- the dominant issue is model-side occupied-vs-unoccupied separation under Dec 17 replay conditions, expressed as a large `unoccupied -> bedroom_normal_use` overprediction band
- threshold / activity-confidence artifacts are too flat to rescue that mistake pattern

## Single highest-signal next experiment

Run one Bedroom-only offline replay ablation on the canonical Dec 17 harness with `Bedroom_v38` weights fixed.

Use the same saved raw Bedroom predictions and compare:

1. raw top-1 direct export
2. current threshold + activity-confidence export
3. export with low-confidence / final rewrite disabled
4. optional Bedroom-only threshold sweep in the same saved score space

Success criterion:

- if the best policy-only variant still stays near the current raw top-1 confusion pattern, especially `unoccupied -> bedroom_normal_use`, then the next move should be a Bedroom-only retrain experiment focused on occupied-vs-unoccupied separation rather than more policy tuning

Why this is the narrowest next step:

- no all-room rerun
- no retrain needed up front
- it cleanly upper-bounds what runtime policy alone can recover from the existing `Bedroom_v38` outputs

## Follow-up

That ablation is now recorded in:

- `docs/reviews/2026-03-11-beta6-bedroom-policy-ablation.md`

Result:

- removing `low_confidence` improves benchmark macro-F1 only because the benchmark treats it as a fourth predicted label
- substantive three-label Bedroom activity quality does not improve materially
- the next move should therefore be a Bedroom-only retrain experiment aimed at occupied-vs-unoccupied separation, not more policy tuning
