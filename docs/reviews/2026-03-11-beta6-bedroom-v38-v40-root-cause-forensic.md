# Beta6 Bedroom v38-v40 Root-Cause Forensic

## Scope

- Explain why `Bedroom_v40` fixed the old `unoccupied -> bedroom_normal_use` failure but created a new `sleep -> unoccupied` collapse on the Dec 17 replay.
- Compare the promoted `v38` replay against the activated `v40` candidate on the exact same Dec 17 timestamps.
- Determine whether the flip is best explained by low-confidence rewriting, class-threshold changes, or two-stage occupancy routing.

## Inputs

Replay artifacts:

- `tmp/jessica_17dec_eval_live_livingroom_rootfix_registryfix_20260311/final/comparison/Bedroom_merged.parquet`
- `tmp/jessica_17dec_eval_candidate_bedroom_sepfix_20260311T041155Z/final/comparison/Bedroom_merged.parquet`

Model artifacts:

- `backend/models/HK0011_jessica/Bedroom_v38_decision_trace.json`
- `backend/models/HK0011_jessica/Bedroom_v38_two_stage_meta.json`
- `backend/models/HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z/Bedroom_v40_decision_trace.json`
- `backend/models/HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z/Bedroom_v40_two_stage_meta.json`

Machine-readable forensic output:

- `tmp/bedroom_v38_v40_root_cause_20260311/summary.json`
- `tmp/bedroom_v38_v40_root_cause_20260311/changed_rows.parquet`

## Finding 1: the biggest mechanism shift is the stage-A occupancy gate

Two-stage Bedroom stage-A occupied threshold:

- `v38`: `0.0027200691401958466`
- `v40`: `0.95`

Calibration detail:

- `v38` stage-A status: `fallback_recall_floor`
- `v38` predicted occupied rate on calibration: `1.0`
- `v40` stage-A status: `target_met`
- `v40` raw threshold before cap: `0.9928349256515503`
- `v40` threshold bound max: `0.95`
- `v40` predicted occupied rate on calibration: `0.2269216623758735`

Inference:

- `v38` effectively admits almost every frame into the occupied path
- `v40` pushes the occupancy gate all the way to the configured cap
- that is strong evidence that the Bedroom replay flip is primarily a stage-A routing change

## Finding 2: Dec 17 occupancy mass swung from over-occupied to under-occupied

Occupied-rate comparison on the same 8460 Dec 17 Bedroom rows:

- truth occupied rate (`sleep` + `bedroom_normal_use`): `0.47210401891252957`
- `v38` predicted occupied rate: `0.8204491725768321`
- `v40` predicted occupied rate: `0.2950354609929078`

Interpretation:

- `v38` was badly over-calling occupied
- `v40` corrected that too far in the other direction
- the replay behavior matches the threshold jump above

## Finding 3: the changed rows are dominated by occupied-to-unoccupied routing

Rows whose final Bedroom prediction changed between `v38` and `v40`:

- total changed rows: `5090`

Truth labels inside those changed rows:

- `unoccupied`: `3077`
- `sleep`: `1468`
- `bedroom_normal_use`: `541`

Largest prediction transitions (`v38 -> v40`):

- `bedroom_normal_use -> unoccupied`: `3348`
- `sleep -> unoccupied`: `1414`
- `unoccupied -> bedroom_normal_use`: `317`
- `low_confidence -> unoccupied`: `11`

Interpretation:

- most of the replay movement is not a small class-threshold nudge
- it is a large rerouting of Bedroom occupied predictions into `unoccupied`

## Finding 4: this is not a low-confidence rewrite problem

Candidate `v40` Bedroom replay:

- final macro-F1 = raw top-1 macro-F1 = raw export macro-F1 = `0.3931771489209137`
- low-confidence rewrite count from raw top-1: `0`

For the new `sleep -> unoccupied` errors:

- count: `1468`
- mean acceptance score: `0.8848681910522035`
- mean raw top-1 probability: `1.0`
- low-confidence rate: `0.0`

For the new `bedroom_normal_use -> unoccupied` errors:

- count: `548`
- mean acceptance score: `0.8848681910522034`
- mean raw top-1 probability: `1.0`
- low-confidence rate: `0.0`

Interpretation:

- these are confident unoccupied outputs
- the replay loss is happening before any late low-confidence rewrite stage
- the dominant mechanism is upstream of runtime abstention logic

## Finding 5: the new errors occur in long contiguous runs

Largest true-`sleep` segments that changed from `v38=sleep` to `v40=unoccupied`:

- `2025-12-17 06:10:50 -> 07:19:30`: `413` rows
- `2025-12-17 05:07:50 -> 06:10:30`: `377` rows
- `2025-12-17 21:35:40 -> 21:53:00`: `105` rows

Sleep-to-unoccupied errors by hour:

- `04`: `322`
- `05`: `359`
- `06`: `359`
- `07`: `118`
- `21`: `255`

Interpretation:

- the failure appears in long coherent sleep spans, not isolated noisy frames
- that is exactly the pattern expected from an over-strict stage-A occupancy gate

## What did improve

The targeted old failure still improved materially:

- true `unoccupied -> bedroom_normal_use`
  - `v38`: `2849`
  - `v40`: `470`

So the oversampling rollback did identify a real causal problem. It just overshot into a different Bedroom occupancy error.

## Root-cause conclusion

The evidence points to this as the dominant root cause:

- Bedroom `v40` recalibrated the two-stage stage-A occupied threshold from a near-zero permissive gate to a capped `0.95` strict gate
- that pushed too many genuinely occupied Bedroom windows into the unoccupied path
- once routed there, the outputs stayed confident and were not later repaired by low-confidence logic

This is a stronger explanation than:

- low-confidence runtime policy
- late class-threshold rewriting
- random model noise

## Recommended next experiment

Do not start with another broad retrain.

The highest-signal next move is a narrow Bedroom-only replay experiment on `v40` stage-A occupancy routing:

- keep the saved `v40` weights fixed
- sweep or override the Bedroom stage-A occupied threshold downward from `0.95`
- measure whether `sleep -> unoccupied` falls substantially without reviving the old `unoccupied -> bedroom_normal_use` failure

Why this is the right next step:

- it directly tests the strongest root-cause hypothesis surfaced here
- it separates calibration/routing failure from representation/training failure
- it is cheaper and cleaner than launching another retrain before proving whether the gate itself is the problem
