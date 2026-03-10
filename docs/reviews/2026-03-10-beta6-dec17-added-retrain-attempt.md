# Beta6 Dec 17 Added Retrain Attempt

## Scope

- Objective: test the next recommended step after the confidence/runtime architecture fix
- Candidate data pack:
  - corrected `2025-12-04` through `2025-12-10`
  - corrected `2025-12-17`
- Target rooms:
  - Bedroom
  - LivingRoom
- Preserved from live pack:
  - Bathroom
  - Kitchen
  - Entrance

## What Was Run

1. Cloned the live `HK0011_jessica` pack into a fresh candidate namespace.
2. Captured a Dec 17 baseline:
   - `tmp/jessica_17dec_eval_candidate_blr_20260310T072625Z/baseline/comparison/summary.json`
3. Attempted deterministic `train_from_files(..., rooms={'Bedroom','LivingRoom'})`.
4. Stopped that path after Bedroom kept generating rejected candidates and blocked useful progress to LivingRoom.
5. Fallback path:
   - Bedroom: recalibrate promoted artifacts on the corrected Dec 4-10 plus Dec 17 pack
   - LivingRoom: direct deterministic room retrain; when that candidate was gate-rejected, recalibrate promoted artifacts on the same corrected pack
6. Final fallback benchmark:
   - `tmp/jessica_17dec_eval_candidate_fallback_20260310T083856Z/final/comparison/summary.json`

## Retrain Findings

### Bedroom full retrain

- Produced candidate-only versions `v31` to `v34`
- All were `gate_pass=false`
- The combined `train_from_files()` path never yielded a promotable Bedroom retrain candidate

Interpretation:

- The confidence architecture fix is not enough to make the Dec 17-added Bedroom weight retrain promotable
- Bedroom remains a model-side problem once the abstention bug is removed

### LivingRoom direct retrain

- Produced candidate-only `v29`
- Gate rejected it for:
  - `no_regress_failed:livingroom:drop=0.248>max_drop=0.050`
  - `lane_b_gate_failed:livingroom:collapse_livingroom_active`

Interpretation:

- Direct weight retraining on the corrected pack still collapses LivingRoom active recall
- Retraining is not the right promotion vehicle yet for LivingRoom either

## Fallback Calibration Findings

### Bedroom fallback recalibration on combined pack

- Final macro-F1: `0.1536`
- Baseline live-pack copy final macro-F1: `0.1816`
- Prior `v2` architecture benchmark final macro-F1: `0.2723`
- Low-confidence rate: `0.4674`

Interpretation:

- Adding Dec 17 into Bedroom recalibration made Bedroom materially worse
- This should not replace the earlier `v2` Bedroom confidence fix

### LivingRoom fallback recalibration on combined pack

- Final macro-F1: `0.3856`
- Baseline live-pack copy final macro-F1: `0.2882`
- Prior `v2` architecture benchmark final macro-F1: `0.3856`
- Low-confidence rate: `0.0000`

Interpretation:

- LivingRoom benefits from the calibrated score-space fix
- The combined-pack recalibration reproduces the earlier strong LivingRoom result

## Recommendation

1. Do not promote the Dec 17-added candidate as one pack.
2. Keep Bedroom on the earlier `v2` confidence-runtime fix rather than the Dec 17-added Bedroom recalibration.
3. Accept the LivingRoom recalibration result as valid.
4. Treat Bedroom as requiring separate model-side work before a Dec 17-added retrain is promotable.
5. Keep Bathroom and Kitchen in their separate threads and do not use the aggregate fallback benchmark as a whole-pack release decision.
