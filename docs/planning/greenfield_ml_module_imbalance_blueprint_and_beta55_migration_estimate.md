# Greenfield ML Module Blueprint (Imbalanced Data) + Beta 5.5 Migration Estimate

- Date: February 17, 2026
- Author: Senior ML Engineering Review
- Scope: `backend/ml` and retrain/promotion flow in Beta 5.5
- Audience: Lane A/B/C/D leads, platform, data, and product owners

## 1. Executive Summary

If we rebuild from greenfield for real-world class imbalance, the right target is:
1. Contract-first event taxonomy and compatibility gates (already strong in Beta 5.5).
2. True two-stage learning (`occupied/unoccupied` -> occupied-only activity), not only post-hoc dual-head conversion.
3. Imbalance-native training stack (class-balanced focal objective, hard-negative mining, calibrated unknown path).
4. Event-first evaluation and promotion as primary gate (already partially in place).
5. Shadow rollout + active learning loop + drift-triggered data refresh.

Current Beta 5.5 has good foundations from Lane A/B and initial Lane C/D wiring, but still uses scaffolds and partial integrations in key places (notably model training objective and backtest stack).

Estimated work to move Beta 5.5 to this target:
- Engineering effort: **34 to 46 engineer-weeks**.
- Realistic calendar with a 4 to 5 person cross-functional pod: **9 to 13 weeks**.
- Conservative calendar (fewer people / slower labeling): **14 to 18 weeks**.

## 2. Current Baseline (Beta 5.5)

### 2.1 What is already strong

1. Registry and compatibility controls (Lane A):
   - Canonical event registry, schema validation, migration checks, CI fail-closed behavior.
2. Event semantics layer (Lane B):
   - Decoder, episode compiler, derived events, event KPI and gate modules, home-empty fusion.
3. Shadow integration hooks (Lane C):
   - Event-first shadow flags and artifacts wired into training/pipeline path.
4. Validation/backtest assets (Lane D):
   - Backtest and validation scripts exist with report outputs.

### 2.2 Gaps versus ideal production design

1. Model core is not yet a true production two-head deep model end-to-end:
   - Current path includes multiclass -> derived dual-head helper (`transformer_head_ab.py`), which is useful but transitional.
2. Imbalance optimization is incomplete:
   - Missing first-class class-balanced focal/objective strategy in main training path with standardized ablation reports.
3. Backtest architecture still includes scaffold behavior:
   - `event_models.py` is RandomForest-based scaffold; not final production family.
4. Data/label loop is not yet full lifecycle:
   - No robust uncertainty-driven active labeling workflow and retrain trigger policy.
5. Monitoring/cutover playbook needs tightening:
   - Drift + unknown-rate + per-event degradation triggers should be wired to operational actions.

## 3. Greenfield Target Architecture (Imbalance-First)

### 3.1 Contracts and interfaces (must exist before modeling)

1. Event registry as single source of truth:
   - Canonical IDs, aliases, criticality tiers, room scopes, unknown fallback policy.
2. Versioned contracts:
   - `RunSpec`, `FeatureSnapshot`, `CandidateArtifact`, `EvaluationReport`, `PromotionDecision`.
3. Compatibility enforcement:
   - CI blocks removals/criticality escalations/invalid fallback references without migration entries.

### 3.2 Data and split strategy (prevent leakage and skew distortion)

1. Split by household and chronology, never random window-only split.
2. Build train/calib/test partitions as chronological blocks per fold.
3. Add dataset health reports:
   - Label prevalence, minority support, room/day coverage, disagreement hotspots.
4. Label QA loop:
   - Weekly reconciliation set for high-impact rare labels (e.g., shower day, out-time boundary events).

### 3.3 Model architecture

1. Stage A: occupancy detector (binary, high recall priority).
2. Stage B: activity classifier for occupied windows only.
3. Shared backbone:
   - CNN/TCN feature extractor + transformer temporal encoder.
4. Unknown path:
   - Explicit unknown decision with confidence/margin/temporal consistency rules.

### 3.4 Training strategy for imbalance

1. Loss:
   - Class-balanced focal (or focal + effective-number weighting) by room/task.
2. Sampling:
   - Stratified mini-batches, majority cap, minority floor, no blind oversampling.
3. Hard-negative mining:
   - Periodic mining from confusion pairs and high-confidence wrong predictions.
4. Calibration:
   - Per-head calibration on split-local calibration windows only.
5. Thresholding:
   - Per-label operating points optimized under precision/recall floor constraints.

### 3.5 Event and KPI layer

1. Decoder:
   - Hysteresis + min-on/min-off durations + room-specific thresholds.
2. Event compiler:
   - Gap-aware merge + minimum duration + deterministic ordering.
3. Derived KPIs:
   - Sleep, shower-day, kitchen-use, livingroom-active, out-time, home-empty safety.

### 3.6 Evaluation and promotion

1. Primary gates:
   - Event/KPI and household safety gates.
2. Secondary diagnostics:
   - Window metrics (macro-F1, PR-AUC, calibration error, Brier).
3. Promotion logic:
   - Pass hard safety gates on all seeds + minimum split coverage.
4. Artifacts:
   - Rejection reasons and decision traces must be complete and reproducible.

### 3.7 Runtime and operations

1. Shadow-first rollout for each cohort.
2. Canary with rollback-safe pointer updates.
3. Monitoring:
   - Drift, unknown-rate, event collapse, false-empty rate, calibration drift.
4. Active learning:
   - Uncertainty/disagreement samples queued for annotation and replay.

## 4. Beta 5.5 -> Ideal Migration Plan

### 4.1 Reuse vs rebuild

Reuse with minimal change:
1. Registry contract stack (`adl_registry.py`, schema, migration validator).
2. Event semantics modules (`event_decoder.py`, `event_compiler.py`, `derived_events.py`, `event_gates.py`).
3. Policy and gate integration skeleton (`policy_config.py`, unified gate flow).

Rebuild/upgrade:
1. Core training objective and architecture contracts in `training.py` / model modules.
2. Backtest engine to use production architecture (deprecate RF scaffold path).
3. Dataset curation and active-learning loop.
4. Operational drift-to-action workflow.

### 4.2 Phased execution (recommended)

### Phase 0 (1 week): Contract freeze and baseline audit
1. Freeze event taxonomy version and RunSpec v1.
2. Baseline current metrics and artifact completeness.
3. Define acceptance thresholds for rare/critical events.

Exit criteria:
1. Signed contract docs and baseline dashboard snapshot.

### Phase 1 (2 weeks): Data and imbalance substrate
1. Build class prevalence and support diagnostics per room/day.
2. Add stratified sampler and imbalance-aware data loader policy.
3. Create hard-negative mining job and artifact output.

Exit criteria:
1. Deterministic data diagnostics and repeatable sampling behavior.

### Phase 2 (2 to 3 weeks): True Stage A/B model implementation
1. Implement real dual-head training/inference contract in model path.
2. Integrate class-balanced focal objective and threshold optimizer.
3. Add calibration per head with leakage-safe split semantics.

Exit criteria:
1. Model outputs production dual-head probabilities without post-hoc conversion dependency.

### Phase 3 (2 weeks): Event-first integration hardening
1. Connect new model outputs to decoder/compiler/gates.
2. Validate room and household KPI consistency.
3. Add regression tests for rare-event stability and unknown behavior.

Exit criteria:
1. Event-first shadow reports generated from production model path for all rooms.

### Phase 4 (2 weeks): Backtest and reproducibility certification
1. Replace scaffold backtest path with production architecture.
2. Run rolling splits x 3 seeds with leakage checklist and CI checks.
3. Publish signoff pack with pass/fail reasons.

Exit criteria:
1. Hard safety gates pass on all seeds and required split coverage.

### Phase 5 (2 to 3 weeks): Canary and controlled cutover
1. Cohort-based shadow and limited authority switch.
2. Drift and unknown-rate alerting with runbook actions.
3. Rollback drill and post-cutover stabilization.

Exit criteria:
1. Canary period passes with no critical safety regressions.

## 5. Effort Estimate

### 5.1 Engineer-week estimate by workstream

| Workstream | Scope | Effort (Engineer-Weeks) |
|---|---|---|
| W1 Contracts + policy freeze | taxonomy/run spec/promotion contract lock | 2-3 |
| W2 Data diagnostics + imbalance sampler | prevalence profiling, stratified loader, QA reports | 4-6 |
| W3 Dual-head production model | architecture, training loop, inference contract | 8-11 |
| W4 Imbalance objective + calibration | focal/weights, threshold tuning, calibration robustness | 5-7 |
| W5 Event integration + gating regression | decoder/compiler integration and KPI stability tests | 4-5 |
| W6 Backtest + signoff pack | rolling splits, 3 seeds, reproducibility artifacts | 4-6 |
| W7 Deployment, monitoring, active learning | canary, drift actions, annotation loop | 7-8 |
| Total |  | **34-46** |

### 5.2 Calendar estimate by staffing scenario

Assumption: one engineer-week means one FTE week of effective delivery (excluding waiting time from external dependencies).

| Team shape | Effective velocity | Calendar estimate |
|---|---|---|
| 2 ML + 0.5 data + 0.25 platform | ~2.5 EW/week | 14-18 weeks |
| 3 ML + 1 data + 0.5 platform | ~4.0 EW/week | 9-13 weeks |
| 4 ML + 1 data + 1 platform | ~5.5 EW/week | 7-10 weeks |

Recommended planning number for team discussion:
1. **11 weeks target**, with **13 weeks committed buffer**.

### 5.3 Critical path items

1. Stage A/B production model contract (W3).
2. Leakage-safe calibration and thresholding (W4).
3. Backtest evidence and canary signoff (W6/W7).

Delays in any of these directly delay cutover.

## 6. Risks and Mitigations

1. Risk: minority event support too low for stable thresholds.
   - Mitigation: enforce minimum support floors and unknown fallback; add targeted labeling.
2. Risk: apparent gains at window level but KPI regressions.
   - Mitigation: promotion authority remains event-gate-driven only.
3. Risk: calibration drift after rollout.
   - Mitigation: monitor ECE/Brier and unknown-rate; auto-fallback to prior champion on hard alerts.
4. Risk: team throughput fragmented across lanes.
   - Mitigation: run weekly integration cut with strict artifact contract validation.

## 7. Decision Requests for Team Meeting

1. Approve the target architecture: true Stage A/B production model + event-first promotion gates.
2. Approve timeline baseline: 11-week target, 13-week committed.
3. Approve annotation budget for rare critical events (needed for W2/W4/W7 quality).
4. Approve canary policy: no authority switch until all hard safety gates pass across seeds/splits.

## 8. Immediate Next 10 Working Days

1. Freeze RunSpec and gate threshold contract versions.
2. Land data prevalence + minority support diagnostics into CI artifacts.
3. Build first true dual-head training branch behind feature flag.
4. Run A/B benchmark versus current Beta 5.5 shadow path on identical rolling splits.
5. Present delta report: KPI gates, unknown-rate, false-empty, calibration.
