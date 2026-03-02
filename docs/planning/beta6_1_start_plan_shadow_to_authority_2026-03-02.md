# Beta 6.1 Start Plan: Shadow-First to Authority (2026-03-02)

## Purpose
Define when Beta 6.1 work should start and how to transition safely from development to care-decision authority.

Decision baseline:
1. Start Beta 6.1 engineering now (as of March 2, 2026).
2. Do not switch authority to Beta 6.1 until Jessica HK001 critical reliability issues are closed and pass gates are met.

## Scope
In scope:
1. Runtime promotion of existing Beta 6 sequence/uncertainty infrastructure
2. Cross-room arbitration integration for single-resident consistency
3. Ops safeguards (alert routing, active-system banner, ML health panel)
4. Shadow evaluation, laddered rollout, rollback criteria

Out of scope:
1. Full architecture rewrite
2. Policy threshold redesign beyond approved gate config
3. Multi-resident arbitration redesign

## Current Runtime State (Already Built)
1. Beta 6 sequence runtime hook is already integrated in prediction flow (`_apply_beta6_hmm_runtime`), gated by runtime flags.
2. Runtime sequence decode requires valid duration-prior policy artifacts; missing/misconfigured artifacts fail closed with `PredictionError`.
3. Timeline shadow artifacts are already emitted in training pipelines; promotion work is integration/activation, not core decoder invention.

## Start Policy (What begins now vs later)

### Start Now (Immediate)
1. Build and test Beta 6.1 in shadow mode only.
2. Land non-disruptive code paths behind feature flags.
3. Add instrumentation and objective pass/fail dashboards.

### Blocked Until Jessica Criticals Are Cleared
1. Any authority switch for care-facing timeline decisions.
2. Any broad pilot cutover beyond controlled canary.

## Entry Gates and Authority Gates

### Engineering Start Gate (Today)
Required:
1. Feature-flag guardrails exist for all new runtime paths.
2. Rollback to Beta 5.5/Beta 6 baseline remains one-step.
3. Shadow reporting and divergence logging are enabled.

### Authority Promotion Gate (Post-Shadow)
Required:
1. Jessica HK001 P0/P1 issues resolved:
   - sleep fragmentation spike patterns
   - cross-room ghosting contradictions
   - false-empty safety regressions
2. Minimum stability window:
   - 7 to 14 consecutive days with no safety-critical regressions
3. Cohort breadth:
   - Jessica + at least 1 additional resident
4. Divergence and rollback thresholds:
   - unexplained divergence within approved limit
   - no unresolved hard-gate failures
5. Ops acceptance:
   - active-system banner correctness
   - Today alert routing (no low-confidence alert flood)

## Phased Execution Plan

### Phase A (2026-03-02 to 2026-03-06): Safety Guardrails + Runtime Activation Preflight
Objective:
1. Prepare safe integration surface before behavioral changes are exposed.

Implementation:
1. Enforce `Today` alert routing (`clinical actionable` vs `review queue`)
2. Enforce global active-system banner in shadow mode
3. Activate existing Beta 6 sequence runtime for shadow cohort via feature flags (`ENABLE_BETA6_HMM_RUNTIME=true`, `BETA6_PHASE4_RUNTIME_ENABLED=true`)
4. Validate duration-prior policy artifacts exist for every room in shadow cohort
5. Validate `BETA6_HMM_DURATION_POLICY_PATH` resolves to a valid policy artifact in target runtime environment
6. Add preflight validation checklist/script for runtime policy artifacts and env flags
7. Keep CRF path canary-only

Primary files:
1. `backend/export_dashboard.py`
2. `backend/ml/pipeline.py`
3. `backend/ml/beta6/serving/runtime_hooks.py`
4. `backend/config/beta6_duration_prior_policy.yaml`
5. `docs/planning/beta6_shadow_mode_runbook.md`

Exit criteria:
1. Ops UI remains unambiguous during shadow
2. No production behavior change when feature flags are off
3. Preflight confirms all required duration-prior artifacts and env pointers are valid

### Phase B (2026-03-05 to 2026-03-12): Cross-Room Arbitration in Shadow
Objective:
1. Prevent timeline contradictions before persistence.

Implementation:
1. Use `HomeEmptyFusion.fuse()` as the cross-room arbitration engine (do not build a new arbitration engine for v1)
2. Promote `EventDecoder` output as the room-level state-stable input to arbitration where applicable
3. Insert arbitration intercept in `process_data.py` between prediction return and per-row event persistence loop (current flow around `prediction_results` handling before `adl_svc.save_adl_event(...)`)
4. Ensure no contradictory raw events are persisted to DB before arbitration is applied
5. Keep `segment_utils` but reduce corrective authority (structural segmentation only)
6. Preserve fallback path to current behavior

Primary files:
1. `backend/process_data.py`
2. `backend/ml/home_empty_fusion.py`
3. `backend/ml/event_decoder.py`
4. `backend/ml/legacy/prediction.py`
5. `backend/utils/segment_utils.py`

Exit criteria:
1. Shadow outputs show reduced multi-location contradictions
2. No increase in unknown/abstain beyond policy caps
3. No contradictory events are written to DB in shadow validation runs

### Phase C (2026-03-10 to 2026-03-18): Episode-Level Evaluation Hardening
Objective:
1. Align quality checks with timeline usability.

Implementation:
1. Keep existing timeline hard gates (MAE/fragmentation)
2. Add episode-oriented evaluation views (IoU/onset-offset tolerance summaries)
3. Publish resident-room daily scorecards

Primary files:
1. `backend/ml/beta6/gates/timeline_hard_gates.py`
2. `backend/ml/evaluation.py`
3. `backend/export_dashboard.py`

Exit criteria:
1. Episode metrics are visible in weekly/technical reports
2. Gate decisions remain deterministic and auditable

### Phase D (2026-03-17 onward): Laddered Cutover
Objective:
1. Controlled authority transition only after hard gates pass.

Steps:
1. 10% resident-room canary
2. 50% ladder stage if no blocking regressions
3. Full authority only after agreed sign-off

Exit criteria:
1. No breach of rollback triggers for the agreed observation window
2. Product, clinical, ops sign-off recorded

## File-Level Change Checklist
1. `backend/ml/pipeline.py`
   - runtime bridge gating and cohort scoping
2. `backend/ml/beta6/serving/runtime_hooks.py`
   - sequence mode selection and fail-closed behavior
   - duration-prior policy path validation behavior
3. `backend/ml/legacy/prediction.py`
   - keep existing sequence runtime hook usage; add/align decoded output handoff for arbitration input
4. `backend/utils/segment_utils.py`
   - scale back heuristic correction authority
5. `backend/process_data.py`
   - insert arbitration intercept before per-row `adl_svc.save_adl_event(...)` persistence loop
6. `backend/export_dashboard.py`
   - active-system banner, alert routing, ML health snapshot
7. `backend/health_server.py`
   - ml-snapshot contract support for ops monitoring
8. `backend/ml/home_empty_fusion.py`
   - arbitration configuration and integration surface
9. `backend/ml/event_decoder.py`
   - promoted room-level decoder integration surface
10. `backend/config/beta6_duration_prior_policy.yaml`
   - required policy artifact for sequence runtime

## Testing Procedures

### Functional
1. Feature-flag OFF parity test: behavior matches current production baseline
2. Feature-flag ON shadow test: no write-path regressions or DB contract breaks
3. Cross-room contradiction test: no simultaneous conflicting states for single resident where policy forbids overlap
4. Alert-routing test: routine uncertainty appears in Review Queue, not Today clinical list
5. Active-system banner test: all core views show correct authority state
6. Duration-prior artifact preflight test: all shadow cohort rooms have valid policy artifact and load successfully at runtime
7. Intercept-point test: arbitration runs before `adl_svc.save_adl_event(...)` loop in `process_data.py`

### Model/Timeline Quality
1. Daily timeline fragmentation trend must improve or remain stable
2. Duration MAE must stay within room capability thresholds
3. Transition-related failure rate must not regress
4. Unknown/abstain rates remain within configured caps

### Operational Safety
1. Fallback trigger + rollback drill must pass end-to-end
2. Shadow divergence report generated and signed daily
3. Ops can identify active system in under 5 seconds

## Rollback Contract (Proposed Defaults for Team Approval)
Scope:
1. During 10%/50% ladder stages: rollback applies to affected cohort first.
2. During full authority stage: rollback applies system-wide unless incident is cohort-isolated by design.

Automatic rollback triggers (any trigger fires):
1. Home-empty false-empty rate > `5%` for `2` consecutive daily checks.
2. Unexplained shadow divergence > `5%` for `2` consecutive daily checks.
3. Timeline fragmentation critical breach (reason-code ratio >= `11%`) for `2` consecutive daily checks.
4. Unknown-rate >= `16%` or abstain-rate >= `18%` for `2` consecutive daily checks.
5. Timeline hard-gate critical fail persists for `2` consecutive evaluation runs on Jessica HK001.

Rollback action:
1. Set `BETA6_PHASE4_RUNTIME_ENABLED=false` for target scope.
2. Revert target scope to last known stable baseline path.
3. Freeze further promotion until incident review closes.

Rollback SLA:
1. Cohort rollback completion: <= `15 minutes` from trigger detection.
2. Full rollback completion: <= `30 minutes` from trigger detection.
3. Incident owner acknowledgement: <= `5 minutes` for critical severity.

## Passing Requirements (Go/No-Go)
1. Jessica HK001 critical reliability defects closed (P0/P1)
2. 7 to 14 consecutive stable days on Jessica and at least one additional resident
3. No unresolved hard-gate failures in promotion window
4. Divergence, false-empty, and fragmentation metrics within approved thresholds
5. Rollback contract thresholds, scope, and SLA approved by Product/Clinical/Ops/MLOps
6. Ops/Clinical/Product sign-off captured before authority switch

## Recommended Team Discussion Agenda
1. Confirm Jessica P0/P1 closure definition and owner
2. Confirm shadow-to-authority numeric thresholds
3. Confirm whether CRF remains canary-only in Beta 6.1 initial cut
4. Confirm rollback SLA and on-call ownership
5. Confirm target date for first 10% canary start

## Proposed Owners
1. ML/Serving: runtime hooks + arbitration path
2. Backend: persistence orchestration + health contract
3. Ops UI: alert routing + banner + monitoring panel
4. QA: gate and rollback validation suite
5. Product/Clinical/Ops: authority sign-off board
