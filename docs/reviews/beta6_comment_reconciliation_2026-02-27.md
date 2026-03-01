# Beta 6 Comment Reconciliation Matrix (2026-02-27)

- Workspace: `/Users/dicksonng/DT/Development/Beta_6`
- Commit checked: `697cc45` (plus local reconciliation updates listed below)
- Source comments reviewed:
1. Team/reviewer notes in `/Users/dicksonng/Desktop/beta6_pilot_progress_followup_report_2026-02-27.md`
2. Prior finding: P2 Stage 4 error-path artifact-path reporting bug

## Summary

The reviewer direction is useful, but a significant subset of file-level claims in the follow-up report are stale versus the current commit. The correct status now is:
1. Prior P2 artifact-path issue: fixed and covered by tests.
2. Many Phase 1-4 modules/configs previously reported missing: now present.
3. Real blockers still open: Phase 5 adapter lifecycle implementation and CRF production readiness.

## Resolution Update (Applied Before Phase 5 Start)

The following reconciliation actions are now implemented in code/CI:
1. Added explicit import smoke gate (including `run_daily_analysis`) via [check_beta6_import_smoke.py](/Users/dicksonng/DT/Development/Beta_6/backend/scripts/check_beta6_import_smoke.py).
2. Added fail-closed Beta6 policy YAML schema tests for unknown/duration/pretrain/fine-tune/active-learning/runtime-parity via [test_beta6_policy_config_schemas.py](/Users/dicksonng/DT/Development/Beta_6/backend/tests/test_beta6_policy_config_schemas.py).
3. Wired both into required CI workflow in [ci-gate-hardening.yml](/Users/dicksonng/DT/Development/Beta_6/.github/workflows/ci-gate-hardening.yml) (contract job + gate-integration job).
4. Kept Stage 4 regression enforcement in required subset (`tests/test_run_daily_analysis_beta6_authority.py` remains in CI list).
5. Added full 14-file Beta6 config schema validator in [beta6_schema.py](/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/beta6_schema.py) and CI gate command [check_beta6_config_schema.py](/Users/dicksonng/DT/Development/Beta_6/backend/scripts/check_beta6_config_schema.py).
6. Added shim deprecation inventory and guardrail:
   - inventory: [beta6_shim_deprecation_inventory.md](/Users/dicksonng/DT/Development/Beta_6/docs/planning/beta6_shim_deprecation_inventory.md)
   - CI guard test: [test_beta6_shim_import_guard.py](/Users/dicksonng/DT/Development/Beta_6/backend/tests/test_beta6_shim_import_guard.py)
7. Moved Beta6 runtime HMM/unknown hooks into serving layer:
   - implementation: [runtime_hooks.py](/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/serving/runtime_hooks.py)
   - legacy delegator parity tests: [test_prediction_beta6_runtime_hook_parity.py](/Users/dicksonng/DT/Development/Beta_6/backend/tests/test_prediction_beta6_runtime_hook_parity.py)

## Validation Snapshot

1. Import smoke (current tree): `ml.beta6.contracts`, `ml.beta6.gate_engine`, `ml.beta6.orchestrator`, `run_daily_analysis` all import successfully.
2. Targeted tests rerun in this review:
1. `tests/test_beta6_contracts.py`
2. `tests/test_beta6_registry_v2.py`
3. `tests/test_beta6_gate_engine.py`
4. `tests/test_beta6_orchestrator_phase4.py`
5. `tests/test_prediction.py`
6. Combined result: `46 passed`.

## Comment Validity Matrix

| Comment / Claim | Verdict | Evidence | Closure Action |
|---|---|---|---|
| P2: Stage 4 error path reports artifact file paths when artifacts were not written | Resolved | [run_daily_analysis.py](/Users/dicksonng/DT/Development/Beta_6/backend/run_daily_analysis.py:1362), [test_run_daily_analysis_beta6_authority.py](/Users/dicksonng/DT/Development/Beta_6/backend/tests/test_run_daily_analysis_beta6_authority.py:119) | Keep regression test mandatory in CI gate list |
| "Import graph/runtime inconsistency due to missing modules" | Mostly stale | Imports pass for `ml.beta6.contracts`, `ml.beta6.gate_engine`, `ml.beta6.orchestrator`, `run_daily_analysis` | Keep lightweight import smoke in CI to prevent regressions |
| "RunSpec/events/registry_v2 not present" | Stale | [run_spec.py](/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/contracts/run_spec.py:303), [events.py](/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/contracts/events.py:28), [registry_v2.py shim](/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/registry_v2.py:1), [registry impl](/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/registry/registry_v2.py:15) | Update report counts/claims to current tree |
| "Phase 2 artifacts missing (manifest/fingerprint/pretrain/representation)" | Stale | [data_manifest.py](/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/data_manifest.py), [feature_fingerprint.py](/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/feature_fingerprint.py), [self_supervised_pretrain.py](/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/self_supervised_pretrain.py), [representation_eval.py](/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/representation_eval.py) | Replace stale "missing" statements with "present + quality status" |
| "Phase 4 runtime policy YAMLs missing" | Stale | [beta6_unknown_policy.yaml](/Users/dicksonng/DT/Development/Beta_6/backend/config/beta6_unknown_policy.yaml), [beta6_duration_prior_policy.yaml](/Users/dicksonng/DT/Development/Beta_6/backend/config/beta6_duration_prior_policy.yaml), [beta6_runtime_eval_parity.yaml](/Users/dicksonng/DT/Development/Beta_6/backend/config/beta6_runtime_eval_parity.yaml) | Add schema validation checks so this stays true |
| "Local tests blocked by TensorFlow/conftest" | Stale for targeted Beta6 suites | This review reran targeted Beta6 suites with passing results | Keep targeted suite as gating minimum; treat full-suite failures separately |
| "Phase 5 missing: adapters and CRF decoder" | Partially valid | [adapters package placeholder](/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/adapters/__init__.py:1), [CRF placeholder](/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/sequence/crf_decoder.py:6) | Implement adapter lifecycle modules and production CRF decode path before rollout rung advance |
| "Hierarchy/namespace confusion risk" | Valid | Both legacy `backend/ml` and Beta6 namespaces are active | Publish explicit boundary/deprecation map and ownership table |
| "Health snapshot endpoint exists" | Valid | [health_server.py route](/Users/dicksonng/DT/Development/Beta_6/backend/health_server.py:120), [build_ml_snapshot_report](/Users/dicksonng/DT/Development/Beta_6/backend/health_server.py:798) | Keep as positive evidence in status report |

## Practical Plan To Clear All Comments

### Track A: Evidence Re-baseline (same day)
1. [x] Replace stale numeric coverage section in the follow-up report with commit-anchored evidence notes in this reconciliation file.
2. [x] Add a fixed "Evidence Header" template to future reviews:
1. commit hash
2. import-smoke output
3. test command + pass count
4. artifact existence checklist
3. [x] Publish corrected reconciliation in one thread and close stale comments explicitly.

### Track B: Hardening (48-72 hours)
1. [x] Add CI job step that imports:
1. `ml.beta6.contracts`
2. `ml.beta6.gate_engine`
3. `ml.beta6.orchestrator`
4. `run_daily_analysis`
2. [x] Add YAML schema validation for Beta6 policy files (unknown, duration prior, pretrain, fine-tune, active learning, runtime parity).
3. [x] Keep Stage 4 error-path regression tests in required CI subset.

### Track C: Real Remaining Gaps (1-2 weeks)
1. Phase 5 adapters:
1. implement `lora_adapter.py`
2. implement `adapter_store.py`
3. add lifecycle tests (create/warm-up/promote/rollback/retire)
2. CRF:
1. replace placeholder in `sequence/crf_decoder.py` with functional constrained decode
2. add HMM-vs-CRF comparison tests on fixed fixtures
3. Rollout governance:
1. block rung advancement until adapter + CRF acceptance tests pass
2. document readiness criteria in rollout checklist

## Exit Criteria

1. All comments in this matrix marked either:
1. `Resolved` with code+test evidence, or
2. `Open` with owner, due date, and acceptance test.
2. CI includes import smoke + policy schema validation + Stage 4 regression tests.
3. Phase 5 blockers have implementation PRs linked with test evidence.
