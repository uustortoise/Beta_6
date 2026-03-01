# Beta6 Phase 5 Readiness Checklist

- Status: Required go/no-go gate before Phase 5 kickoff
- Date: 2026-02-27

## Hard Preconditions (all required)

- [x] P2 Stage4 artifact-path regression remains green.
  - Required test: `backend/tests/test_run_daily_analysis_beta6_authority.py` (`status=error` -> artifact paths are `None`)
- [x] Full Beta6 YAML schema gate is green.
  - Required command: `python backend/scripts/check_beta6_config_schema.py`
- [x] Shim import guard is active and green.
  - Required test: `backend/tests/test_beta6_shim_import_guard.py`
- [x] Runtime-hook migration status is explicit.
  - Either merged (legacy is thin delegator only), or
  - Scheduled with named owner + due date + acceptance tests.
- [x] Real-data canary evidence gate is active for artifact-based canary promotion.
  - Required config: `backend/config/beta6_canary_gate.yaml`
  - Required test: `backend/tests/test_t80_rollout.py::test_evaluate_canary_artifacts_fails_without_real_data_evidence`

## Rollout Policy (must remain hard-blocking)

1. Rung 2 rollout is blocked until both are implemented and tested:
   - Adapter lifecycle (`lora_adapter.py`, `adapter_store.py`, lifecycle tests)
   - CRF production path (decoder implementation + HMM-vs-CRF acceptance tests)
2. Any failure in the preconditions above blocks Phase 5 kickoff.
