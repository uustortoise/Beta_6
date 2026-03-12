# Beta6 Bedroom Post-Fix Single-Stage Root-Cause Forensic

## Scope

- Explain the repaired-runtime Bedroom regression using only the valid single-stage path.
- Compare live `Bedroom_v38` against candidate `Bedroom_v40` at three layers:
  - saved training / validation metadata
  - corrected-pack source label topology
  - corrected Dec 17 replay behavior
- End with one root-cause statement and one next-step recommendation.

## Inputs

Primary artifacts:

- `tmp/bedroom_postfix_rebaseline_20260311T175000Z/live_v38_postfix/Bedroom_merged.parquet`
- `tmp/bedroom_postfix_rebaseline_20260311T175000Z/candidate_v40_postfix/Bedroom_merged.parquet`
- `backend/models/HK0011_jessica/Bedroom_v38_decision_trace.json`
- `backend/models/HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z/Bedroom_v40_decision_trace.json`
- `tmp/jessica_bedroom_sepfix_20260311T041856Z/combined_corrected_pack.parquet`

New forensic outputs:

- `tmp/bedroom_postfix_single_stage_root_cause_20260311/summary.json`
- `tmp/bedroom_postfix_single_stage_root_cause_20260311/changed_rows.parquet`

Replay truth source:

- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_17dec2025.xlsx`

## Finding 1: the repaired regression is single-stage model behavior, not runtime routing

Under the permanent runtime metadata fix, both namespaces run Bedroom as single-stage fallback:

- live `Bedroom_v38`: `runtime_enabled=false`
- candidate `Bedroom_v40`: `runtime_enabled=false`

So the post-fix Dec 17 gap is now a clean single-stage model / calibration problem:

- live `Bedroom_v38` final macro-F1: `0.4929066675409425`
- candidate `Bedroom_v40` final macro-F1: `0.33052062399815013`
- delta: `-0.16238604354279235`

The candidate is worse on both dominant error families:

- `sleep -> unoccupied`: `117 -> 832`
- `unoccupied -> bedroom_normal_use`: `435 -> 1734`
- `bedroom_normal_use -> unoccupied`: `132 -> 233`

## Finding 2: the Bedroom-only retrain moved into a different class-prior regime before sampling

Saved single-stage training traces show a real evidence-mix shift, not just a threshold change.

### Live `v38`

- evidence sequences before sampling: `14255`
- train share before sampling:
  - `bedroom_normal_use`: `0.0904`
  - `sleep`: `0.4074`
  - `unoccupied`: `0.5022`
- adjusted class weights:
  - `bedroom_normal_use`: `1.4129`
  - `sleep`: `1.7765`
  - `unoccupied`: `0.7186`

### Candidate `v40`

- evidence sequences before sampling: `55342`
- train share before sampling:
  - `bedroom_normal_use`: `0.1618`
  - `sleep`: `0.3151`
  - `unoccupied`: `0.5231`
- adjusted class weights:
  - `bedroom_normal_use`: `2.0602`
  - `sleep`: `1.6927`
  - `unoccupied`: `0.6372`

Meaning:

- `bedroom_normal_use` share nearly doubled: `9.0% -> 16.2%`
- `sleep` share dropped materially: `40.7% -> 31.5%`
- `bedroom_normal_use` weight increased by about `46%`
- `sleep` weight fell slightly
- `unoccupied` weight also fell

This is already enough to expect more Bedroom-normal-use calling and less sleep protection.

## Finding 3: holdout selection accepted a more unoccupied-dominant boundary because it still cleared the no-regress floor

`Bedroom_v40` was accepted because its holdout macro-F1 edged above the live champion:

- live `v38` holdout macro-F1: `0.6257134795187449`
- candidate `v40` holdout macro-F1: `0.6394460946743338`
- shared floor: `0.5757134795187449`

But the accepted holdout shape was different:

### Live `v38` best holdout summary

- recall:
  - `bedroom_normal_use`: `0.5136`
  - `sleep`: `0.8196`
  - `unoccupied`: `0.6540`
- predicted distribution on holdout:
  - `bedroom_normal_use`: `712`
  - `sleep`: `812`
  - `unoccupied`: `1323`

### Candidate `v40` best holdout summary

- recall:
  - `bedroom_normal_use`: `0.2315`
  - `sleep`: `0.7164`
  - `unoccupied`: `0.9415`
- predicted distribution on holdout:
  - `bedroom_normal_use`: `617`
  - `sleep`: `2425`
  - `unoccupied`: `5115`

So the trainer accepted a candidate that:

- roughly halved Bedroom-normal-use recall
- reduced sleep recall
- massively increased unoccupied recall

That still improved holdout macro-F1 on the larger corrected-pack validation split, so the current no-regress gate did not reject it.

## Finding 4: threshold calibration then amplified the wrong boundary in two different ways

Saved class thresholds:

- live `v38`
  - `bedroom_normal_use`: `0.17066459956573105`
  - `sleep`: `0.8892368714230562`
  - `unoccupied`: `0.7697282402092486`
- candidate `v40`
  - `bedroom_normal_use`: `0.0764292632392825`
  - `sleep`: `0.8249020877936664`
  - `unoccupied`: `0.5456312662762256`

Important changes:

- class-0 threshold dropped by about `55%`
- unoccupied threshold dropped by about `29%`
- sleep threshold also softened

### 4a. `unoccupied -> bedroom_normal_use` is strongly threshold-amplified

On Dec 17, all candidate false positives in this bucket have Bedroom-normal-use acceptance scores in `[0.1064, 0.2447]`.

Counterfactuals on the same candidate outputs:

- with `v40` class-0 threshold `0.0764`: keep all `1734`
- with live `v38` class-0 threshold `0.1707`: keep `1028`, remove `706`
- with candidate `v39` class-0 threshold `0.2840`: keep `0`, remove all `1734`

For rows that specifically flipped from live-correct `unoccupied` to candidate `bedroom_normal_use`:

- total flips: `1464`
- kept by `v40` threshold: `1464`
- kept by `v38` threshold: `991`
- kept by `v39` threshold: `0`

So the softer class-0 threshold is a direct mechanism for the midday false-occupancy explosion.

### 4b. `sleep -> unoccupied` is not threshold-only

Candidate `sleep -> unoccupied` errors are much more confident:

- count: `832`
- unoccupied acceptance quantiles:
  - min: `0.5456`
  - median: `0.7605`
  - p90: `0.8512`
  - max: `0.8718`

Counterfactual:

- with `v40` unoccupied threshold `0.5456`: keep all `832`
- with live `v38` unoccupied threshold `0.7697`: still keep `393`

So tightening the unoccupied threshold would help, but almost half of the sleep collapse remains. That means the underlying raw decision surface is already misranking these sleep windows toward unoccupied.

## Finding 5: the replay failure topology matches the corrected-pack Bedroom label shape

The Bedroom rows in the combined corrected pack are:

- `unoccupied`: `32874` (`53.4%`)
- `sleep`: `19630` (`31.9%`)
- `bedroom_normal_use`: `9048` (`14.7%`)

Bedroom-normal-use in that pack is concentrated in daytime / evening hours:

- strong `bedroom_normal_use` hours: `10:00-21:00`
- no daytime `sleep` from `08:00-20:00`

The candidate Dec 17 errors follow the same topology:

- `unoccupied -> bedroom_normal_use` clusters by hour:
  - `10`: `292`
  - `11`: `359`
  - `12`: `360`
  - `14`: `214`
  - `20`: `157`
  - `21`: `93`
- `sleep -> unoccupied` clusters by hour:
  - `05`: `179`
  - `06`: `306`
  - `07`: `116`
  - `21`: `222`

And the largest bad sleep segments are long contiguous runs:

- `2025-12-17 05:07:50` to `06:10:30` (`377` rows)
- `2025-12-17 06:10:50` to `07:16:50` (`397` rows)

This is not random noise. The candidate is imposing the corrected-pack Bedroom day/night partition too aggressively on Dec 17.

## Finding 6: Dec 17 replay class shares expose the exact runtime miss

Truth share on Dec 17:

- `bedroom_normal_use`: `7.15%`
- `sleep`: `40.13%`
- `unoccupied`: `52.72%`

Live `v38` predicted share:

- `bedroom_normal_use`: `10.90%`
- `sleep`: `38.62%`
- `unoccupied`: `48.42%`
- `low_confidence`: `2.07%`

Candidate `v40` predicted share:

- `bedroom_normal_use`: `27.61%`
- `sleep`: `24.79%`
- `unoccupied`: `44.23%`
- `low_confidence`: `3.37%`

So the repaired candidate is not merely “too unoccupied.” It is mispartitioning the occupied side:

- Bedroom-normal-use is overcalled by almost `4x` relative to truth
- sleep is materially undercalled
- a large chunk of true sleep is pushed into confident unoccupied instead

## Root Cause

The post-fix Bedroom regression is a coupled single-stage model / calibration failure:

1. The Bedroom-only retrain moved the pre-sampling evidence mix toward much more `bedroom_normal_use` and materially less `sleep`.
2. That changed the learned decision surface and raised class-0 weight while weakening sleep protection.
3. Checkpoint selection accepted a candidate with much higher unoccupied recall and much lower Bedroom-normal-use / sleep recall because macro-F1 still improved on the corrected-pack holdout.
4. Final threshold calibration then made the failure operational:
   - a very soft class-0 threshold admitted weak daytime Bedroom-normal-use logits on true unoccupied windows
   - a softer unoccupied threshold preserved many already-confident sleep-to-unoccupied mistakes

In short: `Bedroom_v40` is not failing because of the repaired runtime anymore. It fails because the Bedroom-only retrain learned the wrong day/night class partition for the corrected pack, and threshold calibration made that mispartition visible on Dec 17.

## Recommendation

Do not continue iterating on the old two-stage / threshold-sweep branch.

The next Bedroom step should be a fresh model-side experiment from repaired live `v38` with explicit safeguards:

- reject candidates whose holdout Bedroom-normal-use recall collapses below the live champion region
- reject candidates whose predicted Bedroom-normal-use share on validation grows far above the validation truth share
- keep the Bedroom-normal-use acceptance threshold out of the `0.07-0.24` band that admitted the Dec 17 false positives

If you want one concrete follow-up experiment, it should be:

- retrain Bedroom from the repaired `v38` baseline on the corrected pack, but add a gating rule that fails candidates when Bedroom-normal-use threshold softness or validation class-share distortion exceeds the live baseline envelope

That targets the actual root cause now exposed by the corrected runtime semantics.
