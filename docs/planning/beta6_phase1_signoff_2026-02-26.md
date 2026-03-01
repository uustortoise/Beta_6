# Beta 6 Phase 1 Sign-Off Artifact (2026-02-26)

- Status: Pass with scoped defer
- Scope: Phase 0 + Phase 1 contract/policy/registry freeze controls
- Defer note: Step 1.4 sensor onboarding moved to offline/manual Ops flow by team decision

## 1. Evidence Bundle

1. `RunSpec v1` schema hash freeze published and tested.
2. Deterministic gate reason-code mapping and rollback/fallback registry paths tested.
3. Leakage guard tests fail-closed in CI.
4. Runtime/eval parity harness + shared decoder contract implemented and tested.
5. Phase 0 intake gate contract (`validate + diff + smoke`) enforced before Phase 1+ training.

## 2. Test Evidence (local run)

Command:
`python3 -m pytest -q backend/tests/test_beta6_contracts.py backend/tests/test_beta6_gate_engine.py backend/tests/test_beta6_registry_v2.py backend/tests/test_beta6_leakage_guards.py backend/tests/test_beta6_runtime_eval_parity.py backend/tests/test_beta6_intake_gate.py backend/tests/test_beta6_intake_precheck.py backend/tests/test_beta6_capability_profiles.py backend/tests/test_beta6_timeline_hard_gates.py backend/tests/test_beta6_uncertainty_contract.py backend/tests/test_beta6_trainer_intake_gate.py backend/tests/test_run_beta6_label_pack_intake.py`

Result:
`68 passed`

## 3. Decision

1. Phase 0: Complete.
2. Phase 1: Complete for Steps 1.1, 1.2, 1.3, 1.5.
3. Step 1.4: Deferred to Ops-managed offline onboarding process.
