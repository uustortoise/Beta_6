# Beta6 Jessica Live Promotion And LivingRoom Seed Deep Dive

## Scope

- Promote the validated Jessica support-fix candidate into live `HK0011_jessica`
- Verify that the live namespace reproduces the benchmark-leading Dec 17 result
- Use the promoted LivingRoom seed panel artifacts to identify the highest-value next model-side deep dive

## Promotion source and backup

Promotion source namespace:

- `backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z`

Live namespace before promotion:

- `backend/models/HK0011_jessica`

Filesystem backup created before promotion:

- `/tmp/hk0011_jessica_live_backup_20260310T225829Z`

Promotion summary artifact:

- `tmp/jessica_live_promotion_20260310T225829Z.json`

## Promotion method

### Why the promotion was room-wise instead of a namespace overwrite

- live `Bedroom` still carried rollback history down to `v28`
- live `LivingRoom` still carried rollback history down to `v27`
- the support-fix candidate namespace only carried the candidate-era room histories:
  - Bedroom `v34..v38`
  - LivingRoom `v39..v43`
- overwriting the entire live namespace would have discarded the live room rollback chain

### Executed promotion path

1. Added a scripted room-wise merge helper:
   - `backend/scripts/promote_room_versions_from_namespace.py`
2. Copied the support-fix candidate namespace into the main workspace `backend/models/` so source and target lived under one backend root.
3. Promoted only the changed rooms into live `HK0011_jessica`:
   - Bedroom -> `v38`
   - LivingRoom -> `v40`
4. Preserved the older live room versions in `*_versions.json` so deterministic rollback remains available.

### Important execution note

The first promotion pass reused the main-workspace `ml.registry` import path, which still omitted `_activity_confidence_calibrator.json` from rollback alias sync. The room versions were promoted correctly, but the latest activity-confidence aliases were not materialized.

The promotion was then re-run from the `codex/jessica-promotion-deepdive` worktree code path against the same live backend directory. That second pass reused the already imported room versions and correctly materialized the latest activity-confidence aliases.

## Final live namespace state

Fresh-load sanity artifact:

- `tmp/jessica_live_load_sanity_20260310T230119Z.json`

Verified live versions after promotion:

- Bathroom `v35`
- Bedroom `v38`
- Entrance `v26`
- Kitchen `v27`
- LivingRoom `v40`

Verified runtime load result:

- all five rooms loaded through `UnifiedPipeline` / `load_models_for_elder()`
- all five rooms exposed activity-confidence artifacts
- `platform.two_stage_core_models` contained:
  - `Bathroom`
  - `Bedroom`
  - `LivingRoom`

## Live Dec 17 replay

Replay output:

- `tmp/jessica_17dec_eval_live_promoted_20260310T230119Z/final/comparison/summary.json`

Comparator baseline:

- `tmp/jessica_17dec_eval_candidate_supportfix_20260310T2312Z/final/comparison/summary.json`

Final live replay result:

- overall final accuracy `0.8048`
- overall final macro-F1 `0.4486`
- Bedroom final macro-F1 `0.3511`
- LivingRoom final macro-F1 `0.4340`

Comparison:

- the live replay matched the support-fix candidate exactly on all aggregate and per-room final metrics
- promotion therefore changed the live namespace state without changing benchmark behavior relative to the validated candidate

## LivingRoom seed instability deep dive

Primary artifacts:

- `backend/models/HK0011_jessica/LivingRoom_versions.json`
- `backend/models/HK0011_jessica/LivingRoom_v39_decision_trace.json`
- `backend/models/HK0011_jessica/LivingRoom_v40_decision_trace.json`
- `backend/models/HK0011_jessica/LivingRoom_v41_decision_trace.json`
- `backend/models/HK0011_jessica/LivingRoom_v42_decision_trace.json`
- `backend/models/HK0011_jessica/LivingRoom_v43_decision_trace.json`

### What stayed constant across the seed panel

The key no-downsample intervention stayed fixed across `v39..v43`:

- sampled prior drift stayed at roughly `3.97` percentage points for every candidate in the panel
- this is far below the earlier collapse case and is effectively not the differentiating variable anymore

Interpretation:

- the residual instability is not explained by renewed unoccupied-downsample distortion
- the no-downsample fix remains the correct structural change

### What changed sharply across seeds

Holdout outcome bifurcated by seed:

- `v40`:
  - holdout macro-F1 `0.6590`
  - holdout predicted distribution:
    - `livingroom_normal_use=249`
    - `unoccupied=1515`
  - checkpoint selection mode: `no_regress_floor`
- `v39`, `v41`, `v42`:
  - holdout macro-F1 about `0.080..0.082`
  - holdout predicted distribution:
    - roughly `livingroom_normal_use=1673`
    - `unoccupied=89..91`
  - checkpoint selection mode: `no_regress_macro_f1_fallback`
- `v43`:
  - holdout macro-F1 `0.0726`
  - holdout predicted distribution:
    - `livingroom_normal_use=1764`
    - `unoccupied=0`
  - checkpoint selection mode: `no_regress_macro_f1_fallback`

Threshold geometry also diverged sharply:

- `v40` thresholds:
  - `livingroom_normal_use=0.0`
  - `unoccupied=0.5245`
- failing neighbors generally pushed the `unoccupied` threshold to about `0.95`
- `v43` collapsed all the way to an effectively always-active configuration

Activity-confidence geometry diverged with the same seed split:

- `v40` activity-confidence intercept: `-5.2337`
- failing neighbors:
  - `v39`: `3.2822`
  - `v41`: `1.2517`
  - `v42`: `3.7302`
  - `v43`: `-1.8135`

Interpretation:

- the bad seeds are not merely suffering from post-hoc threshold tuning noise
- the raw holdout class geometry already flips into an active-heavy regime before runtime policy applies
- checkpoint selection then falls back because no seed/epoch path clears the no-regress floor
- calibration and activity-confidence artifacts reflect that instability, but they do not appear to be the root cause

## Conclusion

The promotion-grade Jessica fix is complete:

- live `HK0011_jessica` now runs Bedroom `v38` + LivingRoom `v40`
- the live replay reproduces the validated benchmark-leading candidate exactly

The highest-value next model-side thread is now narrow:

- target LivingRoom training/checkpoint stability, not more downsample-policy work
- compare the winning `v40` seed path directly against the neighboring failed seed paths before any new threshold experiments
- prioritize investigation of optimizer / initialization / checkpoint-selection sensitivity over another calibration sweep

Recommended follow-up thread:

- keep the promoted live namespace unchanged
- run a focused LivingRoom-only forensic that inspects why the no-regress floor is reachable for the winning seed but unreachable for the neighboring seeds under the same no-downsample policy
