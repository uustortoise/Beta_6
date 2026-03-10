# Beta6 Bedroom/LivingRoom Retrain Blockers

## Scope

- Goal: hand off the remaining model-side blockers after the Beta6 confidence/runtime architecture fix
- Rooms:
  - Bedroom
  - LivingRoom
- Source pack under investigation:
  - corrected `2025-12-04` through `2025-12-10`
  - corrected `2025-12-17`

## Current Best Deployable State

The current best Dec 17, 2025 mixed candidate is:

- `Bathroom`: live `v35`
- `Bedroom`: recalibration-only confidence fix `v36`
- `Entrance`: live `v26`
- `Kitchen`: live `v27`
- `LivingRoom`: recalibration-only confidence fix `v30`

Benchmark:

- output: `tmp/jessica_17dec_eval_candidate_mixed_20260310T090755Z/final_v3/comparison/summary.json`

Relevant result:

- overall final accuracy `0.7608`
- overall final macro-F1 `0.3922`
- Bedroom final macro-F1 `0.2723`
- LivingRoom final macro-F1 `0.3856`

Interpretation:

- The confidence/runtime architecture fix is working.
- The remaining blockers are now retraining/model-selection blockers, not the old late-abstain bug.
- Two follow-up reruns were executed after this mixed benchmark.
- The first policy-first rerun fixed Bedroom and stopped the LivingRoom holdout collapse but still trailed `LivingRoom_v30` on the Dec 17 benchmark.
- The later LivingRoom no-downsample rerun below produced a stronger candidate that beats the old `v30` LivingRoom benchmark and the prior mixed-candidate overall macro-F1.

## What Is Already Fixed

These are not the open problems anymore:

1. LivingRoom late unknown-gate veto
- Fixed by moving runtime to a unified calibrated acceptance-score space and removing the duplicate global confidence veto after class-thresholding.

2. Bedroom legacy threshold-floor coupling
- Fixed by separating acceptance-score threshold bounds from raw-softmax threshold bounds.

3. Threshold-on-dense-band selector defect
- Fixed by adding a stability fallback that escapes the densest acceptance-score band instead of choosing the band itself as the decision boundary.

4. Bathroom parity drift in this branch
- Fixed by merging the missing local `legacy/registry.py` and `legacy/prediction.py` behavior before rerunning the integrated mixed-candidate benchmark.

## Bedroom Blocker

### What failed

Deterministic full retrain on the corrected pack produced candidate-only versions `v31` through `v34`, all rejected.

Primary artifact:

- `backend/models/HK0011_jessica_candidate_blr_20260310T072625Z/Bedroom_versions.json`

Observed pattern:

- every new candidate failed the same way
- best holdout prediction collapsed to `sleep` for all `10242` validation rows
- macro-F1 stuck at `0.1614`

Gate failures:

- `predicted_class_collapse:bedroom:sleep:1.000>=0.950`
- `critical_label_collapse:bedroom:bedroom_normal_use:0.000<= 0.020`
- `critical_label_collapse:bedroom:unoccupied:0.000<= 0.020`
- `room_threshold_failed:bedroom:f1=0.161<required=0.200`
- `no_regress_failed:bedroom:drop=0.263>max_drop=0.050`

### Why this is not a confidence-runtime bug anymore

The failure happens before runtime thresholding matters.

Evidence from `Bedroom_v34`:

- holdout predicted distribution: `sleep=10242`
- holdout recall:
  - `bedroom_normal_use=0.0`
  - `sleep=1.0`
  - `unoccupied=0.0`

That means the model weights themselves collapsed.

### Why the Dec 17-added recalibration is also bad

Fallback recalibration on the corrected Dec 4-10 plus Dec 17 pack made Bedroom worse.

Primary artifact:

- `tmp/jessica_17dec_eval_candidate_blr_20260310T072625Z/fallback_recalibration_bedroom.json`

Important details:

- `sleep` threshold moved to `0.40899087884201635`
- calibration debug status for `sleep`: `fallback_best_f1_stability_fallback`
- calibration support for `sleep`: `937`
- predicted support for `sleep`: `2518`

Dec 17 runtime evidence:

- benchmark output: `tmp/jessica_17dec_eval_candidate_fallback_20260310T083856Z/final/comparison/all_rooms_merged.parquet`
- Dec 17 raw `sleep` acceptance-score band:
  - mean `0.3729`
  - median `0.3734`
  - max `0.408991`

Result:

- final macro-F1 `0.1536`
- low-confidence rate `0.4674`
- almost all raw `sleep` rows rewrote to `low_confidence`

Interpretation:

- the corrected-pack recalibration placed the `sleep` threshold at the top edge of the observed Dec 17 `sleep` score band
- this is now a score-geometry/model problem, not the old runtime-confidence bug

### Current best Bedroom state

The best Bedroom state so far remains the earlier recalibration-only fix:

- artifact: `backend/models/HK0011_jessica_runtime_20260307_exact_bounds/Bedroom_v36_decision_trace.json`
- `sleep` threshold `0.20449343475148696`
- Dec 17 final macro-F1 `0.2723`
- low-confidence rate `0.0087`

### Highest-value next Bedroom experiments

1. Reproduce `v34` collapse with the corrected pack while freezing the old pre-collapse training policy as tightly as possible.
- Objective: separate data effect from recipe/policy drift.
- Compare directly against `Bedroom_v28_decision_trace.json`.

2. Inspect why the model collapses to `sleep` before calibration.
- Focus on holdout logits / raw class distribution, not threshold tuning.
- The critical question is why adding corrected data plus current recipe yields `sleep=100%` on holdout.

3. Re-run Bedroom retrain with clinical-priority / class-weight policy held to the older champion behavior.
- `v28` and `v34` have similar class counts but materially different effective weight traces.
- This is the fastest way to test whether policy drift, not labels, triggered the collapse.

4. Treat recalibration-only experiments as diagnostics, not promotion candidates, until the weight collapse is fixed.
- Threshold selection cannot save a model whose holdout predictions already collapsed to one class.

## LivingRoom Blocker

### What failed

Direct deterministic full retrain on the corrected pack produced candidate-only `v29`, which was gate-rejected.

Primary artifact:

- `backend/models/HK0011_jessica_candidate_fallback_20260310T083856Z/LivingRoom_v29_decision_trace.json`

Gate failures:

- `no_regress_failed:livingroom:drop=0.248>max_drop=0.050`
- `lane_b_gate_failed:livingroom:collapse_livingroom_active`

Holdout behavior:

- holdout predicted distribution: `unoccupied=8117`
- holdout recall:
  - `livingroom_normal_use=0.0`
  - `unoccupied=1.0`

Interpretation:

- the retrained weights collapsed to all `unoccupied`
- this is a retraining recipe failure, not a runtime-confidence failure

### Strong leading suspect: downsample-induced prior drift

The direct retrain path produced a large class-prior drift after unoccupied downsampling.

From `LivingRoom_v29_decision_trace.json`:

- pre-downsample train counts:
  - `livingroom_normal_use=6946`
  - `unoccupied=48554`
- post-downsample train counts:
  - `livingroom_normal_use=6946`
  - `unoccupied=14653`
- post-downsample prior drift:
  - max absolute drift `0.1964`
  - drift `19.64` percentage points
- warning recorded:
  - `class_prior_drift_sampled_watch:livingroom:0:0.196>0.100`

This did not block the run directly, but it is the cleanest quantitative anomaly in the rejected candidate.

### Why recalibration-only still works

LivingRoom recalibration on top of the existing strong weights is good.

Primary artifact:

- `tmp/jessica_17dec_eval_candidate_blr_20260310T072625Z/fallback_recalibration_livingroom.json`

Important details:

- `livingroom_normal_use` threshold `0.36614470337872906`
- low-confidence rate on Dec 17 `0.0`
- final macro-F1 `0.3856`

Interpretation:

- existing weights remain usable
- the blocker is the corrected-pack retraining recipe, especially sampling/model-selection behavior

### Current best LivingRoom state

The best LivingRoom state remains recalibration-only `v30`:

- artifact: `backend/models/HK0011_jessica_candidate_fallback_20260310T083856Z/LivingRoom_v30_decision_trace.json`
- Dec 17 final macro-F1 `0.3856`
- low-confidence rate `0.0000`

### Highest-value next LivingRoom experiments

1. Re-run LivingRoom retrain with downsample guardrails tightened or disabled.
- First candidate experiment: no unoccupied downsampling.
- Second candidate experiment: enforce a strict post-downsample prior-drift ceiling instead of the current effectively disabled guard.

2. Hold policy/sampling closer to the champion `v27` retrain recipe.
- Compare `LivingRoom_v27_decision_trace.json` against `LivingRoom_v29_decision_trace.json`.
- The question is whether corrected data alone caused the collapse, or whether the updated recipe/split path did.

3. Evaluate candidates on the real Dec 17 benchmark immediately, not only on saved holdout metrics.
- LivingRoom is sensitive to recipe effects that are visible in room-level active recall.

4. Keep recalibration-only `v30` as the anchor during experimentation.
- Do not replace it unless a weight retrain beats both the Dec 17 benchmark and the holdout gate.

## Investigation Update: `v28` vs `v34`, `v27` vs `v29`

### Bedroom: the strongest evidence now points to recipe drift, not pack-volume drift

- `Bedroom_v28` was created at `2026-03-07T01:15:39Z`; `Bedroom_v34` was created at `2026-03-07T09:33:18Z`.
- The sampled training support barely moved:
  - `v28` pre-sampling `8097 / 14165 / 25546`, post-minority `8606 / 14165 / 25546`
  - `v34` pre-sampling `8097 / 14181 / 25590`, post-minority `8617 / 14181 / 25590`
- Sampled prior drift was also effectively unchanged:
  - `v28` sampled max abs drift `0.0483` (`4.83` pp)
  - `v34` sampled max abs drift `0.0482` (`4.82` pp)

Interpretation:

- the corrected-pack Bedroom collapse is not explained by a large shift in sampled class counts
- the decisive change is the training recipe/policy

Observed recipe drift:

- policy hash changed from `23346140...` in `v28` to `b3d699f3...` in `v34`
- clinical-priority weights changed from a near-neutral Bedroom profile:
  - `bedroom_normal_use=1.0`
  - `sleep=1.6`
  - `unoccupied=1.0`
- to the broader production-weight profile:
  - `bedroom_normal_use=1.2`
  - `sleep=1.6`
  - `unoccupied=0.75`
  - `inactive=0.85`
  - `low_confidence=0.6`

Collapse-path difference:

- `v28` only survives because collapse retry finds a better retry pass
  - first pass was all `unoccupied`
  - retry switched to non-collapsed `sleep=5963`, `unoccupied=4208`, `bedroom_normal_use=71`
- `v31` through `v34` collapse on first pass and on retry identically
  - `sleep=10242`
  - retry is not rescuing the optimizer path anymore

Matching code-path note:

- the older trainer state that produced `v31` through `v34` matches the current local `backend/ml/training.py` single-stage fit path that hard-codes `shuffle=False` for the main fit and the collapse-retry fit
- the later typed-policy trainer adds explicit `training_profile.post_split_shuffle_rooms` / `factorized_primary_rooms`, and Bedroom is explicitly listed in the post-split shuffle set there

Inference:

- Bedroom is now best explained by recipe drift before thresholding:
  - changed class-weight policy
  - changed optimizer/batch-order behavior
- data addition alone is not a sufficient explanation for the all-`sleep` collapse

### LivingRoom: the corrected-pack failure is concentrated in unoccupied downsample drift

- `LivingRoom_v27` was created at `2026-03-07T07:20:47Z`; `LivingRoom_v29` was created at `2026-03-10T08:53:09Z`.
- The resolved downsample recipe stayed effectively the same:
  - `min_share=0.30`
  - `stride=4`
- But the corrected pack materially increased raw `unoccupied` volume:
  - `v27` pre-sampling `6452 / 42806`, post-downsample `6452 / 22931`
  - `v29` pre-sampling `6946 / 48554`, post-downsample `6946 / 14653`

Resulting sampled prior drift:

- `v27` sampled max abs drift `0.0886` (`8.86` pp)
- `v29` sampled max abs drift `0.1964` (`19.64` pp)

Why the run still allowed it:

- `v29` used the newer injected typed policy (`policy hash 30a16ccf...`)
- that policy only guards `entrance` for post-downsample / post-sampling prior drift
- LivingRoom resolves with:
  - `prior_drift_guard_enabled=false`
  - `max_post_downsample_prior_drift=1.0`
- so the run records the drift as a watch but still trains on the distorted sampled prior

Downstream effect on class weighting:

- `v27` class weights:
  - `livingroom_normal_use=2.2770`
  - `unoccupied=0.6407`
- `v29` class weights:
  - `livingroom_normal_use=1.5548`
  - `unoccupied=0.7370`

Interpretation:

- after aggressive unoccupied downsampling, the sampled prior becomes much more balanced than the real holdout prior
- that reduces the active-class correction pressure exactly when the room already depends on preserving minority active recall
- `two_stage_core` cannot rescue this because LivingRoom has only one occupied class (`stage_b_reason=single_occupied_class`)

Inference:

- LivingRoom is primarily a sampling-policy drift failure, not a threshold failure
- the same basic downsample recipe that was survivable at `8.86` pp sampled drift becomes destructive at `19.64` pp on the corrected pack

### Highest-information next runs

1. Bedroom: rerun the corrected pack with the old `v28` clinical-priority profile and Bedroom post-split shuffling restored before any threshold changes.
2. Bedroom: if that still collapses, isolate shuffling from class-weight policy in separate one-room runs.
3. LivingRoom: rerun with LivingRoom unoccupied downsampling disabled or guarded to `<= 0.10` sampled prior drift.
4. LivingRoom: keep all other policy knobs fixed in that rerun so the result answers only the downsample-drift question.

## Execution Update: Policy-First Rerun

Executed candidate namespace:

- `backend/models/HK0011_jessica_candidate_policyfix_20260310T121803Z`

Policy changes applied before rerun:

- Bedroom restored the near-neutral `v28` room-label weighting profile:
  - `bedroom_normal_use=1.0`
  - `sleep=1.6`
  - `unoccupied=1.0`
- LivingRoom added prior-drift guardrails:
  - unoccupied downsample post-drift cap `<= 0.10`
  - minority-sampling post-drift cap `<= 0.10`
- No confidence/runtime or threshold-policy changes were introduced for this rerun.

### Bedroom result: collapse fixed, residual issue is statistical-validity support

Primary artifact:

- `backend/models/HK0011_jessica_candidate_policyfix_20260310T121803Z/Bedroom_v37_decision_trace.json`

Corrected-pack retrain outcome:

- saved candidate `Bedroom_v37`
- holdout predicted distribution:
  - `bedroom_normal_use=712`
  - `sleep=812`
  - `unoccupied=1323`
- holdout macro-F1 `0.6257`
- the previous all-`sleep` collapse is gone

Residual blocker:

- the aggregate retrain metrics still fail statistical-validity promotion because the holdout minority support is too small:
  - `Insufficient minority class support: class 0 has 2 < 30 samples`
- this is a support-gating problem after the collapse is fixed, not the original optimizer collapse

Dec 17 benchmark impact:

- output: `tmp/jessica_17dec_eval_candidate_policyfix_20260310T121803Z/final/comparison/summary.json`
- Bedroom final macro-F1 improved from `0.2723` to `0.3511`
- Bedroom low-confidence rate dropped to `0.0013`

Interpretation:

- the recommended policy-first Bedroom fix worked
- Bedroom is no longer blocked by all-`sleep` behavior
- promotion still needs a decision on the statistical-validity/support path, but threshold work is no longer the right first move

### LivingRoom result under `0.10` drift-cap guardrail: collapse fixed on holdout, but benchmark still trails `v30`

Primary artifacts:

- `tmp/jessica_17dec_eval_candidate_policyfix_20260310T121803Z/train_metrics.json`
- `tmp/jessica_17dec_eval_candidate_policyfix_20260310T121803Z/train_metrics_livingroom_seed40.json`

First rerun outcome under the new guardrail:

- multi-seed panel selected seed `40`
- selected candidate summary:
  - saved version `31`
  - holdout macro-F1 `0.4879`
  - `gate_pass=true`
- sampled prior drift was capped at `9.9975` pp instead of the old `19.64` pp collapse case

Workflow bug exposed during that run:

- the selected `LivingRoom_v31_decision_trace.json` remained on disk
- the corresponding `LivingRoom_v31_model.keras` and `LivingRoom_v31_thresholds.json` had already been cleaned up
- rollback to the selected version therefore failed even though the seed panel chose it

Follow-up fix and rerun:

- training now skips per-seed version cleanup during multi-seed candidate generation
- after selection, registry cleanup runs once and explicitly preserves the winning saved version
- deterministic seed-`40` rerun produced `LivingRoom_v37` with the same non-collapsed holdout result and a successful alias write

Final LivingRoom benchmark impact:

- Dec 17 final macro-F1 is `0.3031`
- that is better than the all-`unoccupied` collapse path, but still below recalibration-only `LivingRoom_v30` at `0.3856`

Interpretation:

- the recommended policy-first LivingRoom fix answered the root-cause question: downsample-prior-drift was the main collapse driver
- the fix is sufficient to stop the all-`unoccupied` holdout collapse
- it is not yet sufficient to replace the current deployable LivingRoom anchor

## Execution Update: LivingRoom No-Downsample Rerun

Executed candidate namespace:

- `backend/models/HK0011_jessica_candidate_nodownsample_20260310T132301Z`

Policy change applied before rerun:

- `unoccupied_downsample.stride_by_room.livingroom=1`
- this disables effective LivingRoom unoccupied downsampling without changing confidence/runtime, class-threshold policy, or the new multi-seed retention fix

Primary artifacts:

- `tmp/jessica_17dec_eval_candidate_nodownsample_20260310T132301Z/train_metrics.json`
- `tmp/jessica_17dec_eval_candidate_nodownsample_20260310T132301Z/final/comparison/summary.json`

Training evidence:

- downsample debug for LivingRoom confirms the intended intervention:
  - `removed=0`
  - `reason=no_removals`
- multi-seed panel candidate seeds were `40..45`
- selected seed `42`
- selected version `LivingRoom_v40`
- selected holdout predicted distribution:
  - `livingroom_normal_use=249`
  - `unoccupied=1515`
- selected holdout macro-F1 `0.6590`
- selected holdout macro-recall `0.7222`
- candidate gates passed and the seed panel promoted `v40` inside the isolated candidate namespace

Dec 17 benchmark impact:

- LivingRoom final accuracy `0.8748`
- LivingRoom final macro-F1 `0.4340`
- LivingRoom low-confidence rate `0.0000`
- overall final accuracy `0.8048`
- overall final macro-F1 `0.4486`

Comparison against prior anchors:

- LivingRoom improved from the earlier `0.10`-cap rerun:
  - `0.3031 -> 0.4340`
- LivingRoom beat recalibration-only `v30`:
  - `0.3856 -> 0.4340`
- overall final macro-F1 improved over the previous policy-fix candidate:
  - `0.4300 -> 0.4486`
- overall final macro-F1 also improved over the old mixed candidate:
  - `0.3922 -> 0.4486`

Root-cause conclusion:

- this rerun confirms the destructive variable was LivingRoom unoccupied downsampling itself, not a generic corrected-pack retrain failure
- capping prior drift to `0.10` was enough to stop collapse, but fully disabling the downsample is what recovered benchmark quality
- the remaining LivingRoom risk is seed instability / calibration geometry, not unresolved collapse or threshold-architecture defects

Current recommendation after the rerun:

- treat `HK0011_jessica_candidate_nodownsample_20260310T132301Z` as the new leading corrected-pack candidate in this branch
- do not spend more time on LivingRoom threshold work unless a promotion/integration check exposes a new regression
- if further LivingRoom work is needed, it should be a narrow forensic on why seed `42` succeeds while the neighboring seeds fail badly

## Execution Update: Safe Mixed Candidate (`Bedroom_v36` + `LivingRoom_v40`)

Executed candidate namespace:

- `backend/models/HK0011_jessica_candidate_safev36_20260310T141724Z`

Assembly method:

- cloned `HK0011_jessica_candidate_nodownsample_20260310T132301Z`
- rolled `Bedroom` back from `v37` to `v36` using the registry rollback helper so latest aliases, thresholds, and activity-confidence artifacts stayed aligned with runtime loading

Primary artifacts:

- `tmp/jessica_17dec_eval_candidate_safev36_20260310T141724Z/final/comparison/summary.json`
- `tmp/jessica_17dec_eval_candidate_safev36_20260310T141724Z/load_sanity.json`

Replay benchmark source:

- corrected `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_17dec2025.xlsx`

Final Dec 17 benchmark impact:

- overall final accuracy `0.7944`
- overall final macro-F1 `0.3974`
- Bedroom final macro-F1 `0.2723`
- LivingRoom final macro-F1 `0.4340`
- post-registry-fix replay rerun reproduced the same metrics with no registry repair activity during load

Comparison against prior mixed candidates:

- versus the old mixed safe anchor (`Bedroom_v36` + `LivingRoom_v30`):
  - overall final macro-F1 `0.3922 -> 0.3974`
  - LivingRoom final macro-F1 `0.3856 -> 0.4340`
  - Bedroom stayed unchanged at `0.2723`
- versus the benchmark-leading no-downsample candidate (`Bedroom_v37` + `LivingRoom_v40`):
  - overall final macro-F1 `0.4486 -> 0.3974`
  - the lost upside is entirely the deliberate Bedroom rollback to `v36`

Fresh-load sanity result:

- all five rooms loaded cleanly through `UnifiedPipeline` / `load_models_for_elder()`
- current versions matched the intended safe candidate:
  - `Bathroom_v35`
  - `Bedroom_v36`
  - `Entrance_v26`
  - `Kitchen_v27`
  - `LivingRoom_v40`
- `platform.two_stage_core_models` contained:
  - `Bathroom`
  - `LivingRoom`
- all five rooms exposed thresholds and activity-confidence artifacts after load
- first clean load exposed a registry false positive: semantically equal Bathroom two-stage metadata was being compared byte-for-byte and triggered unnecessary alias sync
- fixed the registry validation path so JSON artifacts compare semantically instead of by raw bytes
- post-fix fresh-load sanity completed with `registry_repair_required=false`

Updated deployment recommendation:

- treat `HK0011_jessica_candidate_safev36_20260310T141724Z` as the safest promotion-grade candidate in this branch when Bedroom must remain fail-closed on support policy
- keep `HK0011_jessica_candidate_nodownsample_20260310T132301Z` as the benchmark-leading candidate only if an explicit Bedroom `v37` exception is acceptable
- do not spend more time on Bedroom/LivingRoom modeling before deciding between those two deployment postures

## Cross-Room Conclusion

The remaining work should split cleanly:

- Bedroom:
  - the all-`sleep` retrain collapse is fixed by restoring the older weighting policy and Bedroom shuffle behavior
  - the remaining blocker is now promotion/statistical-validity support, not collapse or threshold geometry

- LivingRoom:
  - the all-`unoccupied` retrain collapse is fixed, and the no-downsample rerun now beats deployed `v30` on the Dec 17 benchmark
  - the remaining follow-up, if any, is a narrow seed-instability / calibration-geometry forensic rather than more sampling-threshold rescue work

Do not reopen the confidence-runtime architecture thread unless new evidence shows a raw/final mismatch again. That part is already fixed enough to expose the real model-side blockers.

## Start-Here Artifacts For A New Thread

- `docs/reviews/2026-03-10-beta6-mixed-candidate-benchmark.md`
- `docs/reviews/2026-03-10-beta6-dec17-added-retrain-attempt.md`
- `backend/models/HK0011_jessica_candidate_blr_20260310T072625Z/Bedroom_versions.json`
- `backend/models/HK0011_jessica_candidate_blr_20260310T072625Z/Bedroom_v28_decision_trace.json`
- `backend/models/HK0011_jessica_candidate_blr_20260310T072625Z/Bedroom_v34_decision_trace.json`
- `backend/models/HK0011_jessica_candidate_fallback_20260310T083856Z/LivingRoom_v27_decision_trace.json`
- `backend/models/HK0011_jessica_candidate_fallback_20260310T083856Z/LivingRoom_v29_decision_trace.json`
- `backend/models/HK0011_jessica_candidate_fallback_20260310T083856Z/LivingRoom_v30_decision_trace.json`
- `backend/models/HK0011_jessica_candidate_nodownsample_20260310T132301Z/LivingRoom_v40_decision_trace.json`
- `backend/models/HK0011_jessica_candidate_safev36_20260310T141724Z/Bedroom_versions.json`
- `tmp/jessica_17dec_eval_candidate_blr_20260310T072625Z/fallback_recalibration_bedroom.json`
- `tmp/jessica_17dec_eval_candidate_blr_20260310T072625Z/fallback_recalibration_livingroom.json`
- `tmp/jessica_17dec_eval_candidate_mixed_20260310T090755Z/final_v3/comparison/summary.json`
- `tmp/jessica_17dec_eval_candidate_nodownsample_20260310T132301Z/train_metrics.json`
- `tmp/jessica_17dec_eval_candidate_nodownsample_20260310T132301Z/final/comparison/summary.json`
- `tmp/jessica_17dec_eval_candidate_safev36_20260310T141724Z/final/comparison/summary.json`
- `tmp/jessica_17dec_eval_candidate_safev36_20260310T141724Z/load_sanity.json`
