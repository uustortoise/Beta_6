# Beta6 Jessica LivingRoom Root Fix Design

**Date:** 2026-03-11

**Goal:** Remove the underlying LivingRoom seed-instability that leaves `HK0011_jessica` dependent on a single lucky no-downsample winner (`LivingRoom_v40`) by making the training path itself more stable and by rejecting unstable seed panels at selection time.

## Current state

- Live `HK0011_jessica` is already promoted and must stay frozen:
  - Bathroom `v35`
  - Bedroom `v38`
  - Entrance `v26`
  - Kitchen `v27`
  - LivingRoom `v40`
- The seed forensic on `LivingRoom_v39..v43` already established:
  - all five versions share the same no-downsample policy except `policy.reproducibility.random_seed`
  - post-sampling prior drift is identical across the panel, so downsample distortion is not the remaining cause
  - only `v40` reaches `selection_mode = no_regress_floor`
  - failed seeds already diverge at checkpoint-selection time, before runtime thresholding can explain the split
- Additional trace evidence narrows the unstable layer further:
  - `v39`, `v41`, `v42`, and `v43` all record `metrics.two_stage_core.stage_a_calibration.status = fallback_recall_floor`
  - those same failed seeds report `predicted_occupied_rate = 1.0`
  - `v40` instead records `status = target_met` with `predicted_occupied_rate ~= 0.099`

## Problem

LivingRoom is still unstable under the current no-downsample recipe. The system can produce a promotable result, but only when one seed happens to avoid an occupancy-first collapse in the two-stage path. That is not a permanent fix.

The task is therefore not to protect runtime from that instability. The task is to make the LivingRoom training path itself harder to collapse and to encode a selection contract that refuses to bless seed panels where most seeds still fail in the same way.

## Constraints

- Stay LivingRoom-only for model changes.
- Do not reopen confidence/runtime wiring.
- Do not reopen Bedroom support-gating work.
- Do not retrain all rooms.
- Prefer a structural training-policy fix over any threshold-only or promotion-only workaround.
- Accept a lower peak than `v40` if needed, as long as the result is materially more stable across seeds.

## Approaches considered

### 1. Promotion-side guard only

Add a stronger promotion rule that rejects unstable LivingRoom seed panels, but leave the recipe unchanged.

Pros:

- low code risk
- blocks accidental re-promotion of obviously unstable winners

Cons:

- does not fix the training instability
- still leaves LivingRoom dependent on lucky seeds

Decision: reject as primary fix; keep only as a supporting safeguard.

### 2. Tune thresholds or confidence behavior to mimic `v40`

Try to compensate for failed seeds with threshold or confidence adjustments.

Pros:

- quick to try
- might recover some replay metrics without retraining

Cons:

- explicitly cosmetic relative to the observed root cause
- failed seeds are already wrong at checkpoint-selection time
- would reopen a thread that the prior forensics already retired

Decision: reject.

### 3. Stabilize the training path and encode a room-level stability contract

Make LivingRoom follow the same post-split shuffle protection that already helped other fragile rooms, then require LivingRoom seed panels to demonstrate multi-seed stability instead of accepting a single lucky winner.

Pros:

- addresses the problem at the training path, where the collapse actually emerges
- preserves the current reliability-first candidate ranking while adding a room-specific stability gate
- creates a reusable contract for future LivingRoom retrains

Cons:

- requires touching policy defaults, training selection, and regression coverage
- may reduce peak macro-F1 if the stable recipe is slightly more conservative

Decision: recommend.

## Design

### 1. Stabilize LivingRoom training batches

Add `livingroom` to the default `training_profile.post_split_shuffle_rooms` list.

Rationale:

- prior blocker notes already showed that fragile rooms can collapse when post-split ordering is left untouched
- LivingRoom currently lacks that protection with no documented reason
- the failing seed traces are consistent with an early occupancy-path collapse, which is the kind of pathology that biased post-split ordering can amplify

This is a training-side correction, not a runtime compensation.

### 2. Encode a LivingRoom stability contract

Extend training selection metadata so LivingRoom seed panels expose whether they are stable enough to trust.

For LivingRoom specifically, record:

- `seed_panel_no_regress_pass_count`
- `seed_panel_stage_a_collapse_count`
- `seed_panel_is_stable`

Treat a seed as stage-A-collapsed when the two-stage calibration falls back to the recall floor and its saved `predicted_occupied_rate` saturates to the fully occupied regime.

The contract should require more than one viable seed. A panel with one `no_regress_floor` winner and several occupancy-collapse losers is still unstable, even if the current ranking can pick the lone good seed.

### 3. Keep winner selection reliability-first, then apply the stability gate

Do not rewrite the candidate ranking itself unless evidence forces it.

Current selection already prefers:

- not collapsed
- gate pass
- no-regress pass
- better gate-aligned score
- better macro-F1

That ranking is directionally correct. The missing piece is that LivingRoom needs a room-level panel validation step after ranking, so the chosen winner cannot be treated as promotion-grade unless the broader panel is stable enough.

### 4. Verification strategy

Verification should answer two distinct questions:

1. Did the code-path change actually apply the intended protection?
2. Did the LivingRoom seed panel become materially more stable?

That means:

- unit tests proving LivingRoom now participates in post-split shuffle on the relevant training paths
- unit tests proving stage-A occupancy collapse is counted correctly in multi-seed summaries
- a LivingRoom-only rerun of the no-downsample seed panel after the recipe change
- judging success by stability first, then by replay quality

### 5. Stop condition

If the shuffle-enabled recipe produces a stable LivingRoom panel with acceptable replay quality, stop there.

If it does not, the next root-cause layer is the two-stage stage-A training path itself, not threshold tuning or runtime compensation. That would justify a second forensic focused narrowly on stage-A calibration and checkpoint dynamics.

## Success criteria

- LivingRoom is protected by the default post-split shuffle policy.
- Training artifacts explicitly report whether a LivingRoom seed panel is stable or only lucky.
- Regression tests cover both the shuffle application and the new panel-stability accounting.
- A LivingRoom-only retrain using the revised recipe shows materially improved cross-seed stability, or clearly proves that the next root cause is deeper inside stage-A.
- The resulting branch history is merge-ready and documents both the root-fix decision and its verification outcome.
