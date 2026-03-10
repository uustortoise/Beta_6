# Beta6 Activity Confidence Architecture Design

## Goal

Replace Beta6's raw-softmax confidence gating with a persisted, learned activity-confidence layer so:

- class-thresholding and unknown-abstain routing use one score space
- room heads no longer rely on globally comparable raw softmax confidence
- threshold learning avoids placing decision boundaries inside dense score bands
- Jessica's LivingRoom and Bedroom failures are fixed as a consequence of architecture, not cosmetic tuning

## Problem Summary

Jessica exposed two architectural defects:

1. `LivingRoom` passes room-level `unoccupied` thresholding, then fails later in the Beta6 unknown path because the later gate reinterprets raw top-1 softmax with a stricter global floor.
2. `Bedroom` learns a `sleep` threshold that sits directly on the densest raw score band, so tiny score drift rewrites large slices of true `sleep` into `low_confidence`.

The common root cause is that runtime policy assumes raw top-1 softmax is a stable, globally meaningful confidence value. It is not.

## Chosen Approach

Implement a per-room activity-confidence calibrator for timeline-native rooms.

For each room head, training learns a binary acceptability model answering:

`given this prediction geometry, how likely is the top-1 class to be correct?`

The calibrator consumes prediction-derived features:

- top-1 probability
- top-2 probability
- margin
- normalized entropy
- predicted class identity

The calibrator outputs an `activity_acceptance_score` in `[0, 1]`.

This score becomes the runtime authority for:

- class acceptance / low-confidence thresholding
- Beta6 unknown-abstain routing

Raw softmax remains available for diagnostics, but it is no longer the score used for final accept/reject decisions when the new artifact exists.

## Alternatives Considered

### 1. Calibrated activity-confidence layer

Recommended and selected.

Pros:

- directly fixes the score-space mismatch
- fits the existing Beta6 artifact and registry model
- easy to deploy incrementally with compatibility fallback
- supports later uncertainty upgrades without rewriting serving again

Cons:

- still requires good calibration data
- adds one more learned artifact per room

### 2. Margin- or energy-only abstention

Not selected.

Pros:

- cleaner than raw softmax gating

Cons:

- less interpretable for ops
- still requires separate threshold semantics
- more custom logic, less reusable artifact structure

### 3. Conformal or set prediction

Not selected for this fix.

Pros:

- strongest formal uncertainty framing

Cons:

- materially more invasive
- awkward fit for current single-label export contract
- does not by itself solve Jessica's threshold-cliff training issue
- relies on calibration/label assumptions that are weak in the noisy-room cases we already know about

## Architecture

### Training-time artifacts

For each upgraded room, training persists:

- existing per-class threshold JSON
- new per-room activity-confidence calibrator JSON
- calibration debug payload describing score-space quality and threshold stability decisions

The calibrator artifact stores:

- schema version
- class labels
- feature names
- logistic calibration coefficients and intercept
- fit diagnostics
- score space identifier: `activity_acceptance_score_v1`

### Runtime data flow

1. model emits room probabilities or logits
2. runtime derives top-1 / top-2 / margin / entropy features
3. room calibrator maps those features to `activity_acceptance_score`
4. class-specific thresholding uses `activity_acceptance_score`
5. Beta6 unknown-abstain routing consumes the same `activity_acceptance_score`
6. scoped runtime-unknown conversion, if still enabled, operates on the already unified uncertainty result instead of reinterpreting raw softmax

### Compatibility

- timeline-native rooms (`bedroom`, `livingroom`) use the new calibrator path once retrained
- rooms without a calibrator artifact continue using current raw-confidence behavior
- registry loading exposes whether a room is on legacy or calibrated confidence mode

This allows safe rollout without silently mixing score spaces inside a single room artifact set.

## Threshold Learning Redesign

Thresholds are now learned in calibrated acceptance-score space, not raw per-class softmax space.

For each predicted class:

1. take calibration rows where top-1 predicted class is that class
2. target is binary correctness of that prediction
3. search candidate thresholds for precision/recall targets
4. apply stability guardrails before finalizing the threshold

This changes threshold meaning from:

`raw probability of class j`

to:

`calibrated reliability of exporting class j`

That is the durable semantic we need for runtime policy.

## Stability Guardrails

Threshold selection must not bisect the densest acceptance-score band.

Guardrails:

- compute local score density around each candidate threshold using a fixed stability window
- prefer candidates meeting precision and recall targets with the lowest local density
- if the best candidate still lies inside an overly dense band, search for the nearest lower stable candidate
- if no stable candidate satisfies targets, record an explicit `stability_fallback` status and choose the least dense candidate instead of silently using the cliff point

Persisted debug fields include:

- selected threshold
- selected status
- local density near threshold
- density window
- whether stability fallback triggered

This gives us explicit evidence when the guardrail prevented another Bedroom-style cliff.

## Serving Contract Changes

### Prediction output columns

Add:

- `activity_acceptance_score`
- `activity_confidence_source`
- `predicted_top1_margin`
- `predicted_entropy`

These fields make the new runtime decisions auditable during evaluation.

### Beta6 unknown policy

The unknown path keeps its existing semantics for:

- `outside_sensed_space`

But the `min_confidence` gate now consumes externally supplied confidence scores when available. For upgraded rooms, that supplied score is `activity_acceptance_score`.

This preserves the existing API shape while removing the raw-softmax assumption.

## Persistence and Registry Changes

Model registry gains a new optional artifact suffix:

- `_activity_confidence_calibrator.json`

It must participate in:

- versioned artifact saving
- latest alias syncing
- rollback
- cleanup
- room loading

Platform runtime state gains:

- `activity_confidence_artifacts[room_name]`

## Testing Strategy

Add tests for:

- calibrator fit and scoring semantics
- threshold stability fallback on dense bands
- runtime unknown path honoring supplied acceptance scores
- legacy prediction path using acceptance score for both thresholding and unknown routing
- registry save/load/rollback of calibrator artifacts
- policy schema coverage if new calibration config keys are introduced

## Validation Plan

1. rerun targeted unit/integration tests for training, registry, and runtime
2. retrain Jessica on the existing Dec 4-10 pack with the new confidence machinery
3. rerun Dec 17, 2025 benchmark without golden correction
4. report both raw top-1 and final exported metrics
5. explicitly quantify whether the raw-vs-final gap narrowed
6. inspect calibrated score distributions for LivingRoom `unoccupied` and Bedroom `sleep`
7. run one controlled retrain with Dec 17 added to Dec 4-10 after the architecture fix is validated

## Expected Outcome

- LivingRoom no longer collapses because room acceptance and later abstain routing disagree
- Bedroom no longer rewrites large `sleep` bands due to a threshold sitting on the score mode
- runtime confidence decisions are traceable, learned, and room-aware
- future data collection happens on top of a stable confidence architecture instead of compensating for a broken one
