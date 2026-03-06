# Beta6 Structural Training/Decoding Fix Report (2026-03-06)

## Scope
- Implement structural training/decoding fixes in `backend/ml/training.py` to address recurrent collapse/no-regress failures.
- Re-run training on the current 7-file Jessica dataset in `data/raw`.
- Compare pre/post behavior and document remaining blockers with executable next steps.

## Code Changes Implemented
1. Gate-aligned checkpoint fallback improvements
- Added macro-F1 fallback tracking in `_GateAlignedCheckpointCallback`.
- Added fallback mode resolver:
  - `GATE_ALIGNED_NO_REGRESS_FALLBACK_MODE` (`macro_f1` default, or `score`).
- Added precision-floor penalty plumbing for gate-aligned validation scoring:
  - `GATE_ALIGNED_PRECISION_FLOOR_PENALTY` (default `0.35`).
  - `GATE_ALIGNED_PRECISION_FLOOR_BY_ROOM_LABEL` map override.

2. Entrance precision-floor enforcement support
- Added default precision floor for Entrance:
  - `entrance.out >= 0.20`.
- Precision floor is now evaluated and penalized in gate-aligned summary/tuning.

3. Bedroom continuity constraint hardening
- Tightened defaults:
  - `BEDROOM_SLEEP_BRIDGE_MAX_STEPS` default `18`.
  - `BEDROOM_SLEEP_BRIDGE_MIN_OCC_PROB` default `0.45`.
- Added boundary-confidence and conversion-budget controls:
  - `TWO_STAGE_BEDROOM_BRIDGE_BOUNDARY_SLEEP_MIN_PROB` default `0.75`.
  - `TWO_STAGE_BEDROOM_BRIDGE_MAX_CONVERSION_RATIO` default `0.04`.
- Reduced aggressive sleep mass transfer and kept higher unoccupied residual.

4. Two-stage threshold tuning fallback
- Added macro-F1 rescue path when no candidate satisfies no-regress floor.
- Added `best_macro_f1_no_regress_fallback` reason path.

5. Core room set update
- Included `entrance` in default two-stage core rooms.

## Test Verification
- `pytest tests/test_training.py -q` -> 83 passed
- `pytest tests/test_beta6_hmm_duration_priors.py -q` -> 4 passed
- `pytest tests/test_beta6_policy_config_schemas.py -q` -> 9 passed

## Retrain Evidence
- Completed run artifact:
  - `backend/tmp/beta6_gate_artifacts/HK0011_jessica/beta6_daily_HK0011_jessica_20260306T030352Z`
- Evaluation result: run failed policy gate due to `bedroom` and `entrance`.
- Rejection artifact reason code: `fail_gate_policy`.

## Pre/Post Metrics (Accuracy, Macro-F1)

Baseline source: `logs/pre_structural_fix_retrain_2026-03-06.json`  
Post source: room decision traces (`Bathroom_v26`, `Bedroom_v23`, `Entrance_v22`, `Kitchen_v21`, `LivingRoom_v21`)

| Room | Pre Accuracy | Post Accuracy | Delta | Pre Macro-F1 | Post Macro-F1 | Delta | Gate (Post) |
|---|---:|---:|---:|---:|---:|---:|---|
| Bathroom | 0.3664 | 0.3664 | +0.0000 | 0.6242 | 0.6242 | +0.0000 | PASS |
| Bedroom | 0.4777 | 0.5131 | +0.0353 | 0.3305 | 0.2360 | -0.0945 | FAIL |
| Entrance | 0.9071 | 0.9027 | -0.0044 | 0.5676 | 0.4963 | -0.0713 | FAIL |
| Kitchen | 0.7925 | 0.7925 | +0.0000 | 0.7747 | 0.7747 | +0.0000 | PASS |
| LivingRoom | 0.5606 | 0.3909 | -0.1697 | 0.7275 | 0.7275 | +0.0000 | PASS |

## Forensic Findings (Current Blockers)

1. Bedroom blocker is now purely no-regress
- Blocking reason: `no_regress_failed:bedroom:drop=0.095>max_drop=0.050`.
- Candidate macro-F1 (`0.2360`) is below champion floor (`0.3305 - 0.05 = 0.2805`).
- Important nuance:
  - Collapse retry now works and recovered from full-collapse in retry stats (`gate_aligned_score -0.7268 -> 0.8000`).
  - But final selected model still underperforms champion on macro-F1, so promotion remains correctly blocked.

2. Entrance blocker is also purely no-regress
- Blocking reason: `no_regress_failed:entrance:drop=0.071>max_drop=0.050`.
- Candidate macro-F1 (`0.4963`) is below champion floor (`0.5676 - 0.05 = 0.5176`).
- Entrance collapse retry also improved from severe collapse baseline (`macro_f1 ~0.033` first pass), but final macro-F1 still not enough for promotion.

3. Drift behavior changed as intended
- Sampled class-prior drift remains high for non-blocking rooms but now recorded as watch (not hard fail):
  - Bathroom, Kitchen, LivingRoom show `class_prior_drift_sampled_watch:*`.
- This confirms drift blocker behavior moved away from immediate hard-stop in this policy path.

4. Safety posture remained intact
- Candidates for blocked rooms were rejected; existing champions stayed active.
- Phase6 stability report still not cert-ready due gate failure and fallback-active state.

## What Is Fixed vs Not Fixed

Fixed:
- Collapse prevention and retry behavior is materially improved.
- Entrance precision-floor logic is now enforced in gate-aligned scoring.
- Bedroom continuity bridging is less aggressive and bounded.

Not yet solved:
- Bedroom/Entrance still fail no-regress against existing champions.
- Structural sequence model is still HMM + heuristics; no CRF/household-latent decoder in runtime path yet.

## Executable Follow-Up Plan

1. Add no-regress diagnostics output to artifacts (must do next)
- Persist explicit champion-vs-candidate deltas by class and macro-F1 margin in rejection artifact.
- Purpose: reduce iteration latency; avoid manual trace mining each run.

2. Introduce constrained checkpoint acceptance for no-regress rooms
- For Bedroom/Entrance only, accept candidate checkpoint only if:
  - macro-F1 within no-regress floor, and
  - critical-label precision/recall floors are met.
- This avoids selecting low-loss checkpoints that are gate-incompatible.

3. Tighten Entrance `out` precision floor from penalty to hard lane condition
- Current floor is represented in scoring; move to explicit promotion blocker for Entrance to prevent high-recall/low-precision `out` behavior.

4. Implement Phase-5 structural decoder upgrade on Bedroom/Entrance path
- Add CRF or equivalent household-state-aware decoder constraints for:
  - sleep continuity,
  - out/unoccupied temporal consistency,
  - transition plausibility.
- This is the structural fix that most directly targets persistent no-regress gaps.

5. Run controlled A/B retrain (single-cycle, no watcher loop)
- Execute one-shot retrain and compare:
  - macro-F1 delta vs champion,
  - per-label recall/precision floors,
  - no-regress pass/fail margin.

## Current Verdict
- Structural fixes are implemented and validated by tests.
- They improved collapse behavior but did not yet clear Bedroom/Entrance no-regress blockers.
- Promotion safety logic is working as designed; further structural decoder work is still required to clear these blockers safely.
