# Beta 6.1 Executable Plan (2026-03-02)

## Objective
Deliver Beta 6.1 timeline reliability upgrade as a shadow-first rollout, then authority cutover only after Jessica HK001 and cohort gates pass.

Primary outcome:
1. Reduce timeline flicker and cross-room contradictions without unsafe regressions.
2. Preserve fast rollback to baseline.
3. Keep Ops workflow clear and actionable.

## Scope
In scope:
1. Activate existing Beta 6 sequence runtime (HMM default, CRF canary).
2. Insert pre-persistence cross-room arbitration in runtime flow.
3. Promote existing decoder/fusion modules instead of building new engines.
4. Implement required Ops guardrails (alert routing, active-system banner).
5. Add/validate monitoring and rollback contract execution.

Out of scope:
1. Full model architecture rewrite.
2. New policy families not needed for Beta 6.1 cutover.
3. Multi-resident arbitration redesign.

## Execution Principles
1. Shadow-first: no direct authority switch during early phases.
2. Fail-closed for safety: missing runtime artifacts block promotion.
3. Minimal net-new algorithm work: promote existing modules.
4. One-step rollback always available.

## Work Breakdown (Task-Level)

### T1: Runtime Activation Preflight (Flags + Artifacts)
Objective:
1. Enable existing runtime sequence path safely for shadow cohort.

Code/Config touchpoints:
1. `backend/ml/pipeline.py`
2. `backend/ml/beta6/serving/runtime_hooks.py`
3. `backend/config/beta6_duration_prior_policy.yaml`
4. runtime env config (`ENABLE_BETA6_HMM_RUNTIME`, `BETA6_PHASE4_RUNTIME_ENABLED`, `BETA6_HMM_DURATION_POLICY_PATH`)

Implementation tasks:
1. Confirm runtime bridge uses cohort-scoped activation and fallback-aware behavior.
2. Validate `BETA6_HMM_DURATION_POLICY_PATH` resolves in runtime env.
3. Add preflight check script/function:
   - verifies all shadow-cohort rooms have loadable duration-prior policy
   - verifies required flags are set only for target cohort
4. Keep `BETA6_SEQUENCE_RUNTIME_MODE=hmm` for default rollout.
5. Keep CRF disabled by default (`canary only`).

Tests:
1. Preflight: policy load success for every target room.
2. Runtime shadow smoke: no `PredictionError` from missing/invalid policy path.
3. Flag-off parity: behavior unchanged when activation flags are disabled.

Passing requirements:
1. 100% preflight pass on target shadow cohort.
2. No runtime policy load failures in 3 consecutive shadow days.
3. No behavior drift when feature flags are off.

Owner:
1. ML/Serving

Estimate:
1. 0.5 to 1.0 day

Dependencies:
1. None

---

### T2: Ops Safety Guardrails (Alert Routing + Active-System Banner)
Objective:
1. Prevent alert fatigue and shadow-mode ambiguity for non-ML Ops.

Code touchpoints:
1. `backend/export_dashboard.py`
2. `backend/health_server.py` (if extra status field needed)

Implementation tasks:
1. Route routine uncertainty to `Review Queue`, not `Today` clinical alerts.
2. Keep `Today` alerts limited to care-actionable anomalies.
3. Add persistent banner when shadow mode active:
   - `Beta 5.5 currently active (Beta 6 in Shadow)`
4. Ensure banner appears on Today, Weekly, and shadow-comparison views.

Tests:
1. Alert-routing test with mixed anomaly + uncertainty payload.
2. Banner presence test across all required pages.
3. UX sanity: Ops can identify active authority path in under 5 seconds.

Passing requirements:
1. Zero routine uncertainty items in Today clinical list during test runs.
2. Banner correctness 100% across tested views.

Owner:
1. Ops UI + Backend API support

Estimate:
1. 0.75 to 1.25 days

Dependencies:
1. None

---

### T3: Pre-Persistence Arbitration Intercept
Objective:
1. Ensure contradictory raw events are resolved before DB write.

Code touchpoints:
1. `backend/process_data.py`
2. `backend/ml/legacy/prediction.py`
3. `backend/ml/event_decoder.py`
4. `backend/ml/home_empty_fusion.py`

Implementation tasks:
1. Insert arbitration intercept between:
   - prediction return (`prediction_results`)
   - and per-row `adl_svc.save_adl_event(...)` persistence loop
2. Use existing decoder/fusion modules:
   - room-level stability: `EventDecoder` (where applicable)
   - cross-room occupancy/exclusivity logic: `HomeEmptyFusion.fuse()`
3. Produce arbitration-adjusted `prediction_results` for persistence.
4. Preserve fallback toggle to current path.

Tests:
1. Intercept-point unit/integration test:
   - arbitration executes before first persisted ADL row.
2. Contradiction test:
   - no forbidden single-resident simultaneous contradictory room states written.
3. Regression test:
   - DB schema/columns unchanged.

Passing requirements:
1. 0 forbidden contradictions in validation scenarios.
2. No write-path failures introduced by intercept.
3. Fallback path still functional.

Owner:
1. Backend + ML/Serving

Estimate:
1. 1.0 to 1.75 days

Dependencies:
1. T1 complete for runtime inputs

---

### T4: Segment Heuristic De-Authority (Controlled)
Objective:
1. Reduce brittle post-hoc correction behavior in `segment_utils` while preserving continuity.

Code touchpoints:
1. `backend/utils/segment_utils.py`
2. `backend/process_data.py`

Implementation tasks:
1. Restrict heuristic logic to structural segmentation responsibilities.
2. Remove/scale down high-impact corrective merges where decoder already provides state stability.
3. Keep conservative fallback settings configurable.

Tests:
1. Segment regeneration regression tests on historical Jessica days.
2. Sleep continuity checks to avoid over-fragmentation regressions.
3. Compare baseline vs adjusted segment counts and duration distributions.

Passing requirements:
1. No regression in hard-gate fragmentation thresholds.
2. No increase in safety-critical sleep timeline errors on Jessica.

Owner:
1. Backend + QA

Estimate:
1. 0.75 to 1.5 days

Dependencies:
1. T3 partially complete

---

### T5: Episode-Level Evaluation Upgrade
Objective:
1. Add timeline-quality metrics aligned with user perception.

Code touchpoints:
1. `backend/ml/evaluation.py`
2. `backend/ml/beta6/gates/timeline_hard_gates.py`
3. `backend/export_dashboard.py`

Implementation tasks:
1. Keep existing hard gates (duration MAE + fragmentation).
2. Add episode-oriented summaries (IoU/onset-offset tolerance).
3. Publish daily resident-room scorecards for shadow comparisons.

Tests:
1. Metric-consistency tests on known fixtures.
2. Gate-decision determinism test.
3. Dashboard reconciliation test vs underlying evaluation payloads.

Passing requirements:
1. Episode metrics available for Jessica + cohort residents.
2. Gate outputs deterministic across reruns with same artifacts.

Owner:
1. ML + QA + Ops UI

Estimate:
1. 0.75 to 1.25 days

Dependencies:
1. T3 complete (for meaningful post-intercept evaluation)

---

### T6: Canary Cutover Orchestration + Rollback Automation
Objective:
1. Execute 10% -> 50% -> full ladder with explicit rollback contract.

Code/ops touchpoints:
1. rollout config and runtime env management
2. `backend/run_daily_analysis.py` (status/reporting integration as needed)
3. dashboard/health monitoring surfaces

Implementation tasks:
1. Configure canary cohort targeting.
2. Apply rollback triggers and SLA monitoring:
   - false-empty > 5% for 2 consecutive days
   - unexplained divergence > 5% for 2 consecutive days
   - fragmentation critical ratio >= 11% for 2 consecutive days
   - unknown >= 16% or abstain >= 18% for 2 consecutive days
3. Implement scoped rollback action:
   - set `BETA6_PHASE4_RUNTIME_ENABLED=false` for affected scope
4. Validate rollback SLA:
   - cohort <= 15 min, full <= 30 min

Tests:
1. Simulated trigger test for each rollback condition.
2. End-to-end rollback drill.
3. Recovery and re-entry test after rollback.

Passing requirements:
1. All rollback triggers fire correctly.
2. SLA met in drill scenarios.
3. Promotion freeze/resume controls verified.

Owner:
1. MLOps + Backend + QA

Estimate:
1. 0.75 to 1.25 days

Dependencies:
1. T1 to T5 complete

## Phase Plan and Timeline

### Phase P0 (Day 0-1)
Tasks:
1. T1
2. T2

Exit gate:
1. Runtime/artifact preflight clean.
2. Ops guardrails active in shadow views.

### Phase P1 (Day 2-4)
Tasks:
1. T3
2. T4

Exit gate:
1. Pre-persistence arbitration verified.
2. Contradiction rate reduced with no critical regressions.

### Phase P2 (Day 4-6)
Tasks:
1. T5

Exit gate:
1. Episode-level metrics and dashboards usable for decisioning.

### Phase P3 (Day 6+)
Tasks:
1. T6

Exit gate:
1. Canary ladder complete with no rollback breach.

## Test Matrix (Execution)

### A. Unit/Component
1. Runtime preflight validator
2. Arbitration intercept function
3. Decoder/fusion integration boundaries
4. Segment heuristic behavior deltas
5. Episode metric computations

### B. Integration
1. `pipeline.predict` shadow run with runtime sequence path enabled
2. `process_data` persistence flow with arbitration intercept
3. Dashboard status/alert routing and authority banner
4. Health/report consistency checks

### C. System
1. Daily shadow replay for Jessica + 1 resident
2. Rollback trigger simulation and SLA measurement
3. End-to-end canary progression and freeze controls

## Passing Requirements (Final Go/No-Go)
1. Jessica HK001 P0/P1 reliability issues closed.
2. 7-14 consecutive stable shadow days on Jessica and at least one additional resident.
3. No unresolved hard-gate failures during promotion window.
4. Unknown/abstain/divergence/fragmentation/false-empty stay within approved thresholds.
5. Rollback contract approved and SLA proven in drill.
6. Ops/Clinical/Product sign-off recorded.

## Deliverables
1. Updated runtime configuration and preflight checklist artifact.
2. Code changes for arbitration intercept and guarded segment behavior.
3. Updated test evidence pack (unit/integration/system).
4. Daily shadow scorecards and promotion recommendation log.
5. Rollback drill report with measured SLA.

## Suggested Work Ticket Split
1. `B61-01` Runtime preflight + artifact validation (T1)
2. `B61-02` Ops guardrails (T2)
3. `B61-03` Pre-persistence arbitration intercept (T3)
4. `B61-04` Segment heuristic de-authority (T4)
5. `B61-05` Episode metrics and reporting (T5)
6. `B61-06` Canary + rollback automation (T6)
