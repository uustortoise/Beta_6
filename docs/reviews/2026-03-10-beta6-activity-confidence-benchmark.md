# Beta6 Activity Confidence Benchmark

## Scope

- Thread scope: Bedroom and LivingRoom confidence/runtime architecture
- Production forensic baseline reference: `docs/reviews/beta6_jessica_17dec_forensic_2026-03-10.md`
- Current benchmark source: `HK0011_jessica_train_17dec2025.xlsx`
- Exact mixed-version runtime source reconstructed from rollout pointers:
  - Bathroom `v34`
  - Bedroom `v28`
  - Entrance `v26`
  - Kitchen `v27`
  - LivingRoom `v27`

This note records the final in-repo evidence after:

1. adding persisted activity-acceptance calibration
2. separating acceptance-score threshold bounds from legacy raw-softmax floors
3. stopping the Beta6 unknown hook from re-applying a duplicate global low-confidence gate
4. adding a dense-band fallback in the threshold selector

## Key Artifacts

- Baseline exact-runtime benchmark:
  - `tmp/jessica_17dec_eval_activity_confidence/exact_runtimefix_baseline/comparison/summary.json`
- Final recalibrated benchmark:
  - `tmp/jessica_17dec_eval_activity_confidence/exact_runtimefix_recalibrated_v2/comparison/summary.json`
- Recalibration summary:
  - `tmp/jessica_17dec_eval_activity_confidence/recalibration_summary_exact_bounds_v2.json`

## Production Baseline From Forensic Note

The production failure mode on **December 17, 2025** was:

- Bedroom:
  - final macro-F1 `0.2203`
  - raw top-1 macro-F1 `0.3470`
  - dominant rewrite: `sleep -> low_confidence`
- LivingRoom:
  - final macro-F1 `0.0982`
  - raw top-1 macro-F1 `0.4603`
  - dominant rewrites: `unoccupied -> low_confidence/unknown`

That is the true pre-fix reference for this task.

## Final v2 Benchmark Results

### Bedroom

- raw top-1 macro-F1: `0.3470`
- final exported macro-F1: `0.2723`
- final accuracy: `0.5820`
- low-confidence rate: `0.0087`
- runtime-unknown rate: `0.0000`
- rewrite count from raw top-1: `867`

Interpretation:

- The catastrophic `sleep -> low_confidence` collapse is materially reduced for the right reason.
- The low-confidence path is now driven by `activity_acceptance_score_v1`, not a raw-softmax floor.
- The learned `sleep` threshold moved off the legacy `0.35` raw floor and now persists at `0.204493...` in calibrated score space.
- Remaining loss is now mostly confusion into `bedroom_normal_use`, not mass abstention.

### LivingRoom

- raw top-1 macro-F1: `0.4603`
- final exported macro-F1: `0.3856`
- final accuracy: `0.7062`
- low-confidence rate: `0.0000`
- runtime-unknown rate: `0.0000`
- rewrite count from raw top-1: `1140`

Interpretation:

- The old architectural mismatch is removed.
- Final LivingRoom output no longer collapses through a later global `0.55` confidence veto.
- Final exported macro-F1 moved much closer to raw top-1 than in the original forensic baseline.
- Runtime `unknown` and `low_confidence` are no longer consuming the room.

## Improvement Versus The Intermediate Recalibrated Run

Comparing `exact_runtimefix_recalibrated` to `exact_runtimefix_recalibrated_v2`:

- Bedroom:
  - low-confidence rate `0.0519 -> 0.0087`
  - rewrite count `1211 -> 867`
  - final macro-F1 `0.2775 -> 0.2723`
- LivingRoom:
  - final macro-F1 `0.2236 -> 0.3856`
  - low-confidence rate `0.0914 -> 0.0000`
  - runtime-unknown rate `0.0207 -> 0.0000`
  - raw/final macro gap `0.2367 -> 0.0747`

Interpretation:

- The `v2` selector/runtime changes are the ones that actually removed the LivingRoom collapse.
- Bedroom still needs model-side improvement, but the abstention cliff is no longer the dominant failure mode.

## Score-Space Evidence

The current promoted Bedroom and LivingRoom artifacts now persist a calibrated acceptance space:

- `Bedroom_activity_confidence_calibrator.json`
- `LivingRoom_activity_confidence_calibrator.json`

Current thresholds are saved in that calibrated space:

- Bedroom:
  - `sleep`: `0.20449343475148696`
- LivingRoom:
  - `livingroom_normal_use`: `0.3366946964641008`
  - `unoccupied`: `0.8676510498602383`

Training traces record:

- `confidence_source = activity_acceptance_score_v1`
- `threshold_floor = 0.0`
- `threshold_cap = 1.0`

That confirms the runtime is no longer relying on the old raw-softmax confidence bounds for these calibrated thresholds.

## Residual Risk

- Both Bedroom and LivingRoom calibration traces still show `near_threshold_share = 1.0` on fallback-selected thresholds.
- The new dense-band fallback improves runtime behavior, but the score geometry is still not clean enough to call “fully solved”.
- Bedroom especially still has model-side confusion after abstention is removed.

## Recommendation

1. Accept this as the permanent runtime/confidence-architecture fix for Bedroom and LivingRoom.
2. Use this calibrated/runtime-fixed pack for the next controlled retrain with corrected **December 17, 2025** added.
3. Keep Bathroom and Kitchen in their separate threads; do not let them block this confidence-architecture work.
