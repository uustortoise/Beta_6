# Beta6 Retrain Forensic Report (Run `beta6_daily_HK0011_jessica_20260306T124205Z`)

## Scope
- Elder: `HK0011_jessica`
- Input set: 7 parquet training files in `data/raw`
- Objective: execute structural training fixes end-to-end, retrain, and evaluate promotion blockers

## Code and config changes applied before run
- Added Stage-A lane-B floor eligibility guard in `backend/ml/training.py`.
- Expanded two-stage restart coverage defaults to include `bathroom`, `livingroom`, and `entrance` in `backend/ml/training.py`.
- Added lane-B floor rescue path in Stage-A gate-aligned threshold tuning in `backend/ml/training.py`.
- Added test `test_two_stage_threshold_tuning_rescues_lane_b_floor` in `backend/tests/test_training.py`.
- Runtime profile update in `backend/.env`:
  - `TWO_STAGE_CORE_ROOMS=bedroom,livingroom,kitchen,bathroom`
  - `TWO_STAGE_CORE_RUNTIME_ROOMS=bedroom,livingroom,kitchen,bathroom`
  - `TWO_STAGE_CORE_RESTART_ROOMS=bedroom,bathroom,livingroom`
  - `ENABLE_TWO_STAGE_STAGE_A_LANE_B_FLOOR_GUARD=true`
  - `TWO_STAGE_STAGE_A_LANE_B_REQUIRED_ROOMS=bathroom`
- Synced commented guidance in `backend/.env.example`.

## Verification before retrain
- `pytest backend/tests/test_training.py -q` -> pass
- `pytest backend/tests/test_beta6_hmm_duration_priors.py -q` -> pass
- `pytest backend/tests/test_beta6_policy_config_schemas.py -q` -> pass

## Final run outcome
- Run completed with room-level candidate training for all 5 rooms.
- Beta6 authority gate result for run: `FAILED` with failed rooms: `entrance`, `kitchen`, `livingroom`.
- Deferred promotion succeeded only for:
  - Bathroom (`legacy_v29`)
  - Bedroom (`legacy_v25`)
- Entrances/Kitchen/LivingRoom remained on prior champions (registry fallback protection active).

## Pre/Post comparison (accuracy + macro-F1)
Pre versions are immediate prior candidates; post versions are versions produced by this run.

| Room | Pre version | Post version | Accuracy (pre -> post) | Macro-F1 (pre -> post) | Gate (pre -> post) |
|---|---:|---:|---:|---:|---|
| Bathroom | v28 | v29 | 0.3664 -> 0.3664 | 0.3421 -> 0.6235 | FAIL -> PASS |
| Bedroom | v24 | v25 | 0.5131 -> 0.5131 | 0.4919 -> 0.4919 | PASS -> PASS |
| Entrance | v23 | v24 | 0.9027 -> 0.9027 | 0.0327 -> 0.4963 | FAIL -> FAIL |
| Kitchen | v22 | v23 | 0.7925 -> 0.7925 | 0.7651 -> 0.6883 | PASS -> FAIL |
| LivingRoom | v22 | v23 | 0.3909 -> 0.3909 | 0.6595 -> 0.6595 | FAIL -> FAIL |

## Blocker forensic findings
### Entrance (v24): still blocked
- Blocking reasons:
  - `label_precision_failed:entrance:out:0.100<0.200`
  - `no_regress_failed:entrance:drop=0.071>max_drop=0.050`
- Confusion pattern shows over-prediction of `out`:
  - true `out`: 260 correctly recovered with recall 1.00
  - false positives on `out`: 2353 from true `unoccupied`
  - precision collapses to `0.0995`
- Interpretation:
  - The new run removed prior collapse behavior and recovered macro-F1 materially, but precision floor for `out` is still structurally unmet.

### Kitchen (v23): now blocked by no-regress only
- Blocking reason:
  - `no_regress_failed:kitchen:drop=0.077>max_drop=0.050`
- Absolute behavior is still viable (macro-F1 `0.688`, no collapse), but candidate underperformed prior champion floor.
- Interpretation:
  - Regression gate is the only blocker; this is not a collapse failure.

### LivingRoom (v23): still blocked by no-regress only
- Blocking reason:
  - `no_regress_failed:livingroom:drop=0.068>max_drop=0.050`
- Similar to Kitchen: absolute metrics are reasonable, but still below champion floor.
- Interpretation:
  - Candidate remains non-promotable under current strict no-regress threshold.

## What improved vs previous behavior
- Bathroom moved from collapse-fail to stable pass.
- Bedroom remained pass and retained lane-B safety.
- Entrance improved from severe collapse (`macro_f1~0.03`) to non-collapsed (`macro_f1~0.50`), but precision floor still fails.

## Remaining operational gaps
- Phase6 shadow parity reported missing baselines for:
  - `unknown_rate`, `abstain_rate`, `duration_mae_minutes`, `fragmentation_rate`
- These are baseline-history readiness gaps, not room model training crashes.

## Executable next steps
1. Entrance precision correction:
   - Add explicit precision-constrained threshold sweep for `out` (optimize under `precision(out) >= 0.20` and maximize macro-F1/recall subject to floor).
   - If infeasible on current logits, add household-state prior (home/away) as a hard constraint before `out` emission.
2. Non-critical no-regress calibration:
   - Make `max_drop_from_champion` room-tier specific (`kitchen/livingroom` less strict than `bedroom/entrance`), or run promotion policy as partial-room promote without run-fail when champions are safely retained.
3. Shadow parity baseline readiness:
   - Seed/collect required 7-run baseline metrics so Phase6 parity alerts are signal-based instead of `missing_baseline`.
4. Re-run full retrain after steps 1-2 and re-evaluate with same file set for strict A/B comparison.

## Primary artifacts
- Evaluation report:
  - `backend/tmp/beta6_gate_artifacts/HK0011_jessica/beta6_daily_HK0011_jessica_20260306T124205Z/beta6_daily_HK0011_jessica_20260306T124205Z_evaluation_report.json`
- Rejection artifact:
  - `backend/tmp/beta6_gate_artifacts/HK0011_jessica/beta6_daily_HK0011_jessica_20260306T124205Z/beta6_daily_HK0011_jessica_20260306T124205Z_rejection_artifact.json`
- Room decision traces:
  - `backend/models/HK0011_jessica/Bathroom_v29_decision_trace.json`
  - `backend/models/HK0011_jessica/Bedroom_v25_decision_trace.json`
  - `backend/models/HK0011_jessica/Entrance_v24_decision_trace.json`
  - `backend/models/HK0011_jessica/Kitchen_v23_decision_trace.json`
  - `backend/models/HK0011_jessica/LivingRoom_v23_decision_trace.json`
