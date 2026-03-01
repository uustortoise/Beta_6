# Beta 5.5 Occupancy Ceiling: Phase 2 + Phase 3 Implementation Checklist

Status: Draft for team review  
Date: 2026-02-19  
Scope: Bedroom + LivingRoom occupancy separability and timeline reliability under strict 4x3 WS-6 gates

## 1. Objectives and Exit Criteria

### Objective
- [ ] Break current hard-gate ceiling (`38/60`) without weakening KPI or contract controls.
- [ ] Improve Bedroom/LivingRoom occupancy reliability with timeline-consistent predictions.
- [ ] Keep fail-closed controls intact (baseline binding, leakage artifacts, strict split-seed matrix).

### Exit Criteria
- [ ] Phase 2 interim target: `>=50/60` hard-gate pass count under strict 4x3 matrix.
- [ ] Phase 2 must not regress KPI guardrails (especially LivingRoom active minutes MAE).
- [ ] Phase 3 tiered target:
  - [ ] `60/60` = Gold (promotion-ready)
  - [ ] `55/60` = Conditional accept only if all failures are in single-day train split cells (e.g., `[4]->5`) and documented
  - [ ] `<55/60` = No-Go
- [ ] No contract violations in signoff (`baseline`, `leakage_audit`, `strict_split_seed_matrix` all pass).

## 2. Constraints and Non-Goals

### Constraints
- [ ] Causal-only feature engineering (no future leakage).
- [ ] Feature flags default-off for new behavior.
- [ ] Preserve existing signoff contract schema and enforcement.
- [ ] Maintain reproducibility (`config_hash` must include all new feature/model knobs).

### Non-Goals
- [ ] No label taxonomy redesign in this cycle.
- [ ] No multi-resident redesign in this cycle.
- [ ] No relaxation of promotion gates.

## 3. Phase 2: Feature Retrofit (1 week)

### 3.0 Preconditions (Blocking)
- [ ] Label quality prerequisite: quarantine pathological day-slices from Stage-A label-recall floor evaluation.
- [ ] Rule: if any training day in a split has zero support for a room/label target (example: Bedroom sleep on Day 17), skip that label-recall hard-gate floor for that split-room and log the skip reason + day IDs.

### 3.1 Feature Engineering Tasks
- [ ] Keep existing base circadian features as-is (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`) and do not duplicate them.
- [ ] Add additional circadian features only (`minutes_since_midnight`, `is_night`) if helpful.
- [ ] Add causal 30m texture features for Bedroom/LivingRoom.
- [ ] Add causal 60m texture features for Bedroom/LivingRoom.
- [ ] Add motion inactivity ratio features with explicit definition:
  - [ ] `motion_inactivity_ratio_N = mean(motion < motion_threshold)` over trailing `N`
  - [ ] fixed `motion_threshold = 0.5` for experiment comparability.
- [ ] Add CO2 slope and first-difference texture features on 30m/60m horizons.
- [ ] Add Bedroom-specific light-gated features:
  - [ ] `light_off_streak_30m` (consecutive light-off windows in trailing 30m)
  - [ ] `light_regime_switch` (light just turned off/on)
  - [ ] Optional guarded rule experiment: `light < 1.0` and night hours implies sleep-prior occupancy.
- [ ] Add LivingRoom imbalance remediation:
  - [ ] occupied-window sample weight sweep (`2x`, `3x`, `4x`)
  - [ ] `livingroom_motion_activity_ratio_15m = mean(motion > motion_threshold)` over trailing 15m.
- [ ] Add feature profile switch (`30m`, `60m`, `mixed`) and keep default-off.

### 3.2 Phase 2 Flags (default off)
- [ ] `--enable-bedroom-livingroom-texture-features`
- [ ] `--bedroom-livingroom-texture-windows=30,60`
- [ ] `--bedroom-livingroom-texture-profile={30m,60m,mixed}`
- [ ] `--enable-bedroom-livingroom-extra-circadian-features` (only for non-base additions like `minutes_since_midnight`, `is_night`)
- [ ] `--enable-bedroom-light-texture-features` (isolated Bedroom light-feature ablation on top of temporal features)
- [ ] `--livingroom-occupied-sample-weight={1.0,2.0,3.0,4.0}`

### 3.3 Phase 2 File Change Scope
- [ ] `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
  - Add flags, feature wiring, label-support quarantine logic, config payload + hash inclusion.
  - Add per-room diagnostics for feature-importance ranking.
- [ ] `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/event_models.py`
  - No required interface change for expanded feature vectors (already supports arbitrary width `np.ndarray`).
  - Optional: expose stable helper for top-k feature importances when Stage-A/Stage-B estimator supports it.
- [ ] `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`
  - Add parsing, config serialization, causal feature integrity, and label-support quarantine tests.
- [ ] `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_models.py`
  - Add model smoke tests with texture/extra-circadian feature sets.
- [ ] `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_d2_strict_splitseed_integration.py`
  - Add strict matrix integration checks for feature profiles + imbalance sweep.

### 3.4 Phase 2 Experiment Matrix
- [ ] Baseline reference (`fix26`) rerun for same data snapshot.
- [ ] `+30m` texture only.
- [ ] `+60m` texture only.
- [ ] `+mixed` texture (`30m+60m`).
- [ ] `+extra circadian` only (`minutes_since_midnight`, `is_night`).
- [ ] `+texture+extra circadian` combined.
- [ ] Repeat each variant with LivingRoom occupied sample-weight sweep (`1x`, `2x`, `3x`, `4x`).
- [ ] Capture hard-gate count, KPI checks, and failure reason distribution for each variant.

### 3.5 Phase 2 Validation Checklist
- [ ] Unit tests pass.
- [ ] Integration tests pass.
- [ ] Strict signoff generated for each candidate.
- [ ] Compare against baseline with fixed seeds and split matrix.
- [ ] Select best candidate by hard-gate count, then KPI safety.

## 4. Phase 3: Segment-Based Architecture (Greenfield Target)

### 4.1 Architecture Tasks
- [ ] Keep window models as probability generators only:
  - [ ] Stage-A outputs `P(occupied)` per window
  - [ ] Stage-B outputs per-activity probabilities per window
- [ ] Propose segment boundaries from smoothed occupancy/texture signals.
- [ ] Extract segment-level features (duration, stability, texture, circadian).
- [ ] Classify segments (Bedroom and LivingRoom separately).
- [ ] Project segment decisions back to windows for timeline output.
- [ ] Apply continuity rules at segment boundary level, not per-window flicker logic.

### 4.2 Phase 3 New Modules
- [ ] `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/segment_proposal.py`
  - Deterministic change-point and boundary proposal.
- [ ] `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/segment_features.py`
  - Segment feature aggregation from window signals.
- [ ] `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/segment_classifier.py`
  - Segment-level classifiers for Bedroom/LivingRoom.
- [ ] `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/segment_projection.py`
  - Segment-to-window label projection and continuity cleanup.

### 4.3 Phase 3 Existing File Updates
- [ ] `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
  - Add segment-mode execution path and diagnostics output.
- [ ] `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/aggregate_event_first_backtest.py`
  - Add non-breaking support for segment diagnostics fields in signoff artifacts.
- [ ] `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`
  - Add segment-mode integration tests.

### 4.4 Phase 3 Flags (default off)
- [ ] `--enable-bedroom-livingroom-segment-mode`
- [ ] `--segment-proposal-smoothing-windows`
- [ ] `--segment-min-duration-seconds`
- [ ] `--segment-gap-merge-seconds`
- [ ] `--segment-classifier-type={rf,hgb}`

### 4.5 Phase 3 Tests
- [ ] `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_segment_proposal.py`
- [ ] `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_segment_features.py`
- [ ] `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_segment_classifier.py`
- [ ] `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_segment_projection.py`
- [ ] Strict matrix integration test coverage for segment mode.

### 4.6 Phase 3 Evaluation
- [ ] Primary pass/fail remains current per-window hard-gates (for backward compatibility and signoff continuity).
- [ ] Add informational segment metrics (non-blocking initially):
  - [ ] Segment IoU target `>=0.70`
  - [ ] Boundary error target `<=15 min`
- [ ] Include segment metrics in report artifacts, but do not override hard-gate decision in first rollout.

## 5. Data and Label Quality Controls

- [ ] Enforce minimum support checks before evaluating label-recall gates.
- [ ] Log per-split train-day label support diagnostics in backtest artifacts.
- [ ] Keep leakage checks fail-closed for all new feature/segment modules.

## 6. Signoff and Reporting Artifacts

### Required Artifacts per Candidate
- [ ] `seed11.json`, `seed22.json`, `seed33.json`
- [ ] `seed11_leakage_audit.json`, `seed22_leakage_audit.json`, `seed33_leakage_audit.json`
- [ ] `ws6_rolling.json`
- [ ] `ws6_signoff.json`

### Required Reporting Fields
- [ ] Hard-gate totals (`passed/total`).
- [ ] KPI results and CI pass/fail.
- [ ] Failure reason histogram by room and metric.
- [ ] Feature-importance ranking per room (top-k) where estimator supports importances.
- [ ] Feature/segment config snapshot included in `config_hash` payload.

## 7. Go/No-Go Decision Gates

### Gate A (End of Phase 2)
- [ ] Go if `>=50/60` and no KPI regressions.
- [ ] No-Go if `<50/60`; proceed to Phase 3 with no extra threshold-only iteration.

### Gate B (End of Phase 3)
- [ ] Gold Go: `60/60`, KPI pass, and all contracts pass.
- [ ] Conditional Go: `55/60` only when all misses are isolated to single-day train split cells and documented.
- [ ] No-Go: `<55/60`, KPI failures, or leakage/baseline/matrix contract failure.

## 8. Risks and Mitigations

- [ ] Risk: 60m features over-smooth short true events.  
  Mitigation: keep profile sweep (`30m`, `60m`, `mixed`) and choose by strict matrix.
- [ ] Risk: Segment proposal misses true boundaries.  
  Mitigation: boundary precision/recall diagnostics + min/max segment duration controls.
- [ ] Risk: Runtime inflation.  
  Mitigation: cache rolling features and segment proposals per day-room.
- [ ] Risk: Feature collinearity (30m/60m versions of same signal) reduces tree split quality.  
  Mitigation: monitor feature importance and drop redundant features per room.

## 9. Ownership and Signoff Tracker

| Workstream | Owner | Reviewer | Status | Date | Notes |
|---|---|---|---|---|---|
| Phase 2 feature coding (feature engineering + gates wiring) |  |  | Pending |  |  |
| Phase 2 experiment execution (matrix + sweeps + artifact capture) |  |  | Pending |  |  |
| Phase 2 validation + strict matrix review |  |  | Pending |  |  |
| Phase 3 segment proposal/classification implementation |  |  | Pending |  |  |
| Phase 3 validation + strict matrix review |  |  | Pending |  |  |
| Final promotion signoff |  |  | Pending |  |  |

## 10. Team Review Checklist

- [ ] Scope approved (Phase 2 + Phase 3).
- [ ] File-level plan approved.
- [ ] Experiment matrix approved.
- [ ] Go/No-Go gates approved.
- [ ] Owners assigned.
- [ ] Timeline committed.
