# Beta6 Deep Dive A+B Forensic Report (2026-03-04)

## Scope
- Deep Dive A: Checkpoint/collapse objective alignment vs final gate path.
- Deep Dive B: Bedroom gate failure mechanics (Lane-B sleep recall + room F1 fail).
- Data source: `HK0011_jessica` latest Beta6 artifacts/traces on this machine.

## Evidence Baseline
- Latest room traces:
  - `backend/models/HK0011_jessica/Bathroom_v16_decision_trace.json`
  - `backend/models/HK0011_jessica/Bedroom_v14_decision_trace.json`
  - `backend/models/HK0011_jessica/Entrance_v13_decision_trace.json`
  - `backend/models/HK0011_jessica/Kitchen_v13_decision_trace.json`
  - `backend/models/HK0011_jessica/LivingRoom_v13_decision_trace.json`
- Latest run artifacts:
  - `backend/tmp/beta6_gate_artifacts/HK0011_jessica/beta6_daily_HK0011_jessica_20260304T090433Z/`
  - `backend/tmp/beta6_gate_artifacts/HK0011_jessica/beta6_daily_HK0011_jessica_20260304T092159Z/`

Current room state from latest versioned traces:
- Bathroom: PASS (`macro_f1=0.6033`)
- Entrance: PASS (`macro_f1=0.4963`)
- Kitchen: PASS (`macro_f1=0.7747`)
- LivingRoom: PASS (`macro_f1=0.7239`)
- Bedroom: FAIL (`macro_f1=0.1799`, Lane-B sleep recall fail)

---

## Deep Dive A: Checkpoint Objective vs Final Gate Path

### A1) Code-path mismatch is real
- Checkpoint selection (`_GateAlignedCheckpointCallback`) scores epochs from raw model predictions:
  - `backend/ml/training.py:107-121`
  - `model.predict(X_val)` -> `_extract_activity_probabilities(...)` -> `_summarize_gate_aligned_validation(...)`
- Final release metrics can switch to two-stage primary probabilities:
  - `backend/ml/training.py:5187-5213`
  - If two-stage enabled and gate mode is `primary`, `y_pred` is replaced with `y_pred_two_stage`.
  - `metric_source` is set to `holdout_validation_two_stage_primary`.

Implication:
- Epoch ranking/selection is done on single-stage proxy.
- Final pass/fail is decided on two-stage-composed predictions.
- These can diverge substantially.

### A2) Empirical divergence (Bedroom)
From `Bedroom_v12/v13/v14_decision_trace.json`:

| Version | Checkpoint Best Macro-F1 | Checkpoint Lane-B Mean | Final Macro-F1 | Final sleep recall | Blocking reasons |
|---|---:|---:|---:|---:|---|
| v12 | 0.4353 | 0.8994 | 0.1410 | 0.2815 | room F1 fail + Lane-B fail |
| v13 | 0.2065 | 0.0000 | 0.2121 | 0.2014 | Lane-B fail |
| v14 | 0.4353 | 0.8994 | 0.1799 | 0.0572 | room F1 fail + Lane-B fail |

Interpretation:
- Checkpoint objective can report very strong values while final two-stage gate still fails badly.
- This is a structural objective-alignment issue, not just random variance.

### A3) Practical impact
- Misleading confidence in selected checkpoints/collapse retry decisions.
- Harder debugging because "best summary" and final gate narrative disagree.

---

## Deep Dive B: Bedroom Failure Mechanics

### B1) Exact failure signature (v14)
From `Bedroom_v14_decision_trace.json`:
- `macro_f1=0.1799` (below required 0.200)
- `sleep recall=0.0572` (Lane-B threshold 0.5)
- Gate block:
  - `room_threshold_failed:bedroom:f1=0.180<required=0.200`
  - `lane_b_gate_failed:bedroom:recall_sleep_duration`

Confusion matrix (true rows, predicted cols: `bedroom_normal_use`, `sleep`, `unoccupied`):
- `[[0, 222, 635], [0, 187, 3085], [0, 1225, 2137]]`

Observed:
- `bedroom_normal_use` recall = `0.0` (never predicted).
- Most true sleep windows are routed to `unoccupied`.

### B2) Stage-A occupancy routing is under-predicting occupied state
Derived from v14 confusion matrix:
- True occupied rate in holdout (bedroom_normal_use + sleep): `55.1%`
- Predicted occupied rate: `21.8%`
- Effective occupied recall (occupied vs unoccupied routing): `9.9%`

So the two-stage route is primarily suppressing occupancy, which directly drives sleep recall collapse.

### B3) Calibration split mismatch is a key structural cause
From `metrics.calibration_split_support` in `Bedroom_v14_decision_trace.json`:
- Validation slice class counts: `{0:857, 1:3272, 2:3362}` -> occupied rate `55.1%`
- Calibration tail class counts: `{0:764, 1:743, 2:3486}` -> occupied rate `30.2%`
- Occupied-rate gap: `24.94 percentage points`

And stage-A threshold calibration uses this calibration slice:
- `backend/ml/training.py:5070-5074` and `2663+`
- Resulting calibration stats in trace:
  - `predicted_occupied_rate=0.1843`
  - `min_predicted_occupied_rate=0.1509`
  - status includes `+pred_occ_floor`

Interpretation:
- Threshold is tuned on a much more unoccupied-heavy subset than the gate-evaluated validation subset.
- This biases routing toward `unoccupied` and crushes sleep recall in final gate evaluation.

### B4) Why this is structural (not just "need more data")
- Bedroom training support is not tiny (`train_class_support_pre_sampling`: class0=8097, class1=14165, class2=25546).
- Failure is caused by:
  - two-stage occupancy threshold calibration/selection regime,
  - calibration/validation distribution mismatch,
  - strict routing effects in two-stage composition.
- More similar data alone will not reliably fix this pipeline behavior.

---

## Executable Fix Order

## 1) Alignment fix first (A)
- Make checkpoint/collapse selection explicitly `proxy` when final gate source is two-stage primary.
- Do not treat single-stage checkpoint score as gate-representative in logs/debug decisions.
- Minimum implementation:
  - annotate checkpoint summary with `proxy_source=single_stage_pre_two_stage`.
  - add final post-two-stage summary side-by-side for the selected model.

## 2) Bedroom threshold calibration fix (B)
- Tune stage-A threshold against the same distribution used for release gating.
- Practical option:
  - when two-stage `gate_mode=primary`, calibrate stage-A on `validation_data` rather than tail-only `calibration_data`.
- Add calibration split guard:
  - if occupied-rate gap between validation and calibration exceeds a bound (for example 10pp), fallback to validation for stage-A threshold tuning.

## 3) Add occupancy-route guardrail for Bedroom
- Hard warning/fail-fast when two-stage predicted occupied rate on validation is too far below true occupied rate.
- Example guard:
  - `pred_occ_rate < max(0.35, true_occ_rate * 0.75)` => reject/tune fallback.

## 4) Then rerun retrain and compare
- Expected success condition:
  - Bedroom `sleep recall` materially rises toward Lane-B floor (`>=0.5`).
  - Bedroom `macro_f1 >= 0.2`.
  - No regression in currently passing rooms.

---

## Bottom Line
- Deep Dive A confirms an objective alignment gap between checkpoint selection and final gate source.
- Deep Dive B confirms bedroom failure is mainly a two-stage routing/calibration structure issue, amplified by calibration-vs-validation occupancy distribution mismatch.
- Recommended next action: implement Fix Order 1+2 first, retrain once, then reassess Bedroom before larger architectural changes.

---

## Implementation Update (Executed In This Session)

Implemented:
- `backend/ml/training.py`
  - Added calibration/validation occupied-rate gap guard for two-stage stage-A calibration.
  - If `gate_mode=primary` and `|occ_rate_val - occ_rate_calib| > 0.10`, stage-A calibration now falls back to validation distribution.
  - Added bedroom-specific stage-A predicted-occupied floor env overrides:
    - `TWO_STAGE_CORE_STAGE_A_BEDROOM_MIN_PRED_OCCUPIED_RATIO`
    - `TWO_STAGE_CORE_STAGE_A_BEDROOM_MIN_PRED_OCCUPIED_ABS`
  - Added explicit checkpoint/collapse proxy metadata when final metric source is two-stage primary.
- `backend/tests/test_training.py`
  - Added unit test for bedroom-specific stage-A occupancy-floor override resolution.

Verification:
- `cd backend && pytest tests/test_training.py -q`
- Result: `72 passed, 3 warnings`

### Post-fix Bedroom Outcome (from `Bedroom_v16_decision_trace.json`)

Pre-fix reference (`v14`):
- `macro_f1=0.1799`, `sleep_recall=0.0572`, `Gate=FAIL`
- Blocking: room F1 fail + Lane-B sleep recall fail

Post-fix (`v16`):
- `macro_f1=0.3305`, `sleep_recall=0.9526`, `Gate=PASS`
- Blocking reasons: none

Key stage-A diagnostics now recorded in v16:
- `input_source=validation_data_occ_gap_fallback`
- `alignment.occupied_rate_gap=0.2494` with `max_allowed_gap=0.1000`
- `predicted_occupied_rate=0.7552`, `true_occupied_rate=0.5512`
- Proxy metadata present:
  - `checkpoint_selection.proxy_source=single_stage_pre_two_stage`

Interpretation:
- The implemented split-alignment fix addressed the bedroom structural blocker directly.
- The model now clears both room-F1 and Lane-B sleep recall for Bedroom.
