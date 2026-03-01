# Beta 6 Runtime/Eval Parity Contract (v1)

- Date: 2026-02-25
- Status: Active
- Harness: `backend/ml/beta6/runtime_eval_parity.py`
- Shared decoder authority: `backend/ml/beta6/sequence/hmm_decoder.py`
- Orchestration entry: `backend/ml/beta6/orchestrator.py`
- Test gate: `backend/tests/test_beta6_runtime_eval_parity.py`

## 1. Fixed-Trace Parity Rule

For the same trace input, runtime and eval decoding must match on:
1. `decoded_label`
2. `source_label` (post label-map canonicalization)
3. `uncertainty_state`

Any mismatch is fail-closed in CI.
Runtime authority integration: `run_daily_analysis.py` enforces parity when room metrics include `beta6_parity_trace`, with fail-closed reason `fail_runtime_eval_parity`.

## 2. Shared Defaults

Runtime and eval use the same default label map and decoder policy unless explicitly overridden.

Policy reference:
`backend/config/beta6_runtime_eval_parity.yaml`

## 3. Decoder Semantics Under Test

The fixed-trace harness verifies that decoder behavior is parity-safe, including spike suppression behavior. If one path changes decoder policy while the other does not, parity test fails.
