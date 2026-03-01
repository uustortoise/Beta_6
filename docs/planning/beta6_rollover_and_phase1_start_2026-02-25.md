# Beta 6 Rollover and Phase 1 Start (2026-02-25)

- Status: In progress
- Date: 2026-02-25
- Workspace: `/Users/dicksonng/DT/Development/Beta_6`

## 1. Rollover Actions Completed

1. Archived pre-rollover `Beta_6` snapshot to:
   - `/tmp/Beta_6_pre_rollover_20260225_214341`
2. Rolled `Beta_5.5` baseline into `Beta_6` using full sync (`rsync -a --delete`).
3. Post-rollover validation:
   - `Beta_6` file count: `1359`
   - `Beta_6/backend/ml` file count: `68`
   - `Beta_6/docs/planning` file count: `79`

## 2. Phase 1 Start Scope (from Kick-Start Plan)

Reference docs:
1. `/Users/dicksonng/DT/Development/Beta_6/Beta6/BETA6_100_USER_PILOT_KICKSTART.md`
2. `/Users/dicksonng/DT/Development/Beta_6/docs/planning/beta6_file_mapped_implementation_checklist.md`
3. `/Users/dicksonng/DT/Development/Beta_6/docs/planning/rfc_beta6_greenfield_architecture.md`

Active Phase 1 steps:
1. Step 1.1: Freeze RunSpec v1 and YAML registry schema.
2. Step 1.2: Define gate reason codes and rollback policy.
3. Step 1.3: Implement leakage guardrails in CI.

## 3. Immediate File Targets for Execution

Step 1.1:
1. `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/contracts/run_spec.py`
2. `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/contracts/decisions.py`
3. `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/contracts/events.py`
4. `/Users/dicksonng/DT/Development/Beta_6/backend/ml/unified_training_spec.md`

Step 1.2:
1. `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/gate_engine.py`
2. `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/registry_v2.py`
3. `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/registry_events.py`
4. `/Users/dicksonng/DT/Development/Beta_6/docs/planning/template_beta55_to_beta6_go_no_go.md`

Step 1.3:
1. `/Users/dicksonng/DT/Development/Beta_6/backend/tests/test_beta6_leakage_guards.py` (new)
2. `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/evaluation_engine.py`
3. `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/feature_store.py`
4. `/Users/dicksonng/DT/Development/Beta_6/.github/workflows` (or equivalent CI config)

## 4. Execution Checklist (Closed for Phase 0/1 scope)

1. [x] Confirm/lock RunSpec v1 schema fields and hash policy.
2. [x] Define deterministic gate reason-code enum and mapping table.
3. [x] Add leakage guard tests for resident overlap, time overlap, and window-gap violations.
4. [x] Add CI blocking step for leakage guard failures.
5. [x] Produce first Phase 1 sign-off artifact with pass/fail evidence.

## 5. Deferred Item Note

1. Step 1.4 (sensor onboarding + signal-quality gates) is deferred to offline/manual Ops process and tracked in:
   - `/Users/dicksonng/DT/Development/Beta_6/Beta6/todo.md`
