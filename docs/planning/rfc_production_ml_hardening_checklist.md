# RFC: Production ML Hardening Checklist (Post-Resampling Refactor)

- Status: Completed (Feb 15, 2026)
- Date: February 15, 2026
- Owner: Senior ML Engineer
- Scope: Training reliability, gate validity, model registry integrity, and reproducibility for Beta 5.5 production pipeline

## 1. Problem Statement

Recent fixes correctly connected policy-driven resampling across training paths. However, production risks remain:

1. Fail-open behavior still exists on resampling errors.
2. Room gate can pass on low-evidence evaluations.
3. Data viability collapse is not enforced as a blocking condition.
4. Registry champion metadata and active artifacts can drift.
5. Retraining is not fully deterministic/idempotent for repeated inputs.

This RFC defines required hardening so model quality decisions are reliable and permanent.

## 2. Non-Negotiable Production Requirements

1. Fail closed on preprocessing contract violations.
2. Do not promote candidates trained/evaluated on statistically insufficient evidence.
3. Keep `TrainingPolicy` as single source of truth for training knobs.
4. Guarantee champion consistency in registry metadata + served artifacts.
5. Guarantee deterministic and idempotent behavior for identical `(data, policy, code)` inputs.

## 3. Acceptance Criteria (Definition of Done)

### AC-1: Policy Single Source of Truth

1. No direct environment reads for training knobs outside `load_policy_from_env`.
2. CI check fails if `MAX_RESAMPLE_FFILL_GAP_SECONDS` (or other training knobs) are read outside policy module.
3. Decision trace includes resolved policy fields used in the run.

### AC-2: Fail-Closed Resampling

1. If shared resampler fails for a room, that room is marked failed with explicit reason; no legacy fallback path.
2. Run status clearly distinguishes:
   - room preprocessing failure
   - room gate failure
   - run-level gate failure
3. Unit and integration tests verify no silent fallback to legacy aggregation/original data.

### AC-3: Data Viability Gate (Blocking)

A room candidate is ineligible for training or promotion when any condition fails:

1. `observed_days < min_observed_days_for_room`
2. `post_gap_rows < min_post_gap_rows_for_room`
3. `unresolved_drop_ratio > max_unresolved_drop_ratio_for_room`
4. `post_downsample_windows < min_training_windows_for_room`

Initial policy defaults (room-specific override allowed):

| Metric | Bedroom | LivingRoom | Kitchen | Entrance | Bathroom |
|---|---:|---:|---:|---:|---:|
| min_observed_days | 7 | 7 | 7 | 7 | 7 |
| min_post_gap_rows | 12000 | 12000 | 10000 | 10000 | 8000 |
| max_unresolved_drop_ratio | 0.85 | 0.85 | 0.85 | 0.85 | 0.90 |
| min_training_windows | 3000 | 3000 | 2500 | 2500 | 2000 |

### AC-4: Statistical Validity Gate (Blocking)

A room candidate cannot pass room gate unless all are true:

1. Validation split exists and is temporal.
2. `validation_samples >= min_validation_samples` (default: 300).
3. Per-class validation support `>= min_validation_support_per_class` (default: 30).
4. Per-class calibration support `>= min_calibration_support_per_class` (default: 30) for any class whose threshold is used in pass/fail logic.
5. If fallback status is used (e.g. `fallback_low_support`), gate must fail with explicit reason.

### AC-5: Promotion Eligibility by Data Age

1. For residents with existing champion: auto-promotion requires `training_days >= 7`.
2. For `training_days < 7`: candidate can be stored as challenger only, never promoted.
3. Exception path (manual override) requires explicit senior engineer approval and audit log.

### AC-6: Registry Consistency Contract

1. At most one promoted version per room.
2. `current_version` must reference an existing version with `promoted=true`.
3. If `current_version=0`, unversioned latest aliases must not exist.
4. Loader serves only rooms with valid promoted champion.
5. Startup integrity check auto-heals safe mismatches and emits warning report.

### AC-7: Determinism and Idempotency

1. Training sets seeds (`python`, `numpy`, `tensorflow`) from policy.
2. Candidate fingerprint is computed and persisted:
   - `data_fingerprint`
   - `policy_hash`
   - `code_version` (git SHA)
3. If fingerprint matches latest candidate/champion artifacts and metrics payload, skip retrain/save as `no_op_same_fingerprint`.

### AC-8: Observability and Auditability

1. Decision trace contains:
   - data viability metrics
   - statistical validity metrics
   - final gate reasons
   - fingerprint tuple
2. `training_history` metadata stores the same fields.
3. Dashboard/monitoring exposes counts of:
   - preprocessing_fail
   - viability_fail
   - statistical_validity_fail
   - gate_fail
   - promoted

## 4. Required Code Changes by Module

### 4.1 `backend/ml/policy_config.py`

1. Add dataclasses:
   - `DataViabilityPolicy`
   - `StatisticalValidityPolicy`
   - `ReproducibilityPolicy`
   - `PromotionEligibilityPolicy`
2. Extend `TrainingPolicy` with these policies.
3. Parse defaults + env overrides in `load_policy_from_env`.
4. Include all new fields in `to_dict()` so `policy_hash` reflects them.

### 4.2 `backend/utils/data_loader.py`

1. Remove fallback block that reverts to legacy duplicate aggregation on resampling exception.
2. Raise explicit `DataResamplingError` with room/file context.
3. Return optional resampling diagnostics object (`rows_before`, `rows_after`, `ffill_limit`, `max_gap_seconds`).

### 4.3 `backend/elderlycare_v1_16/platform.py`

1. Remove fail-open branch that continues with original data when resampling fails.
2. Emit structured gap audit metrics after preprocessing:
   - `resampled_rows`
   - `rows_dropped_unresolved`
   - `unresolved_drop_ratio`
3. Return these metrics to caller or attach to processed frame metadata.

### 4.4 `backend/ml/pipeline.py`

1. Introduce `evaluate_data_viability(room_name, audit_metrics, policy)` before model training.
2. If viability fails:
   - skip room training
   - write decision trace fragment
   - set room result status to `data_viability_failed`
3. Ensure same behavior in:
   - `train_and_predict`
   - `train_from_files`
4. Include viability metrics in final run metadata.

### 4.5 `backend/ml/training.py`

1. Add deterministic seed setup at train start using `policy.reproducibility.seed`.
2. Enforce statistical validity checks inside release-gate evaluation.
3. Treat calibration fallback due to low support as gate-failing condition.
4. Compute and persist `data_fingerprint` and `code_version`.
5. Add idempotent short-circuit if fingerprint unchanged.

### 4.6 `backend/ml/registry.py`

1. Add `validate_and_repair_room_registry_state(elder_id, room_name)`.
2. On promote/rollback:
   - set exactly one `promoted=true`
   - keep `current_version` aligned.
3. On `current_version=0`:
   - remove unversioned aliases for model/scaler/encoder/thresholds.
4. Update loader to require valid promoted champion and avoid serving orphan aliases.

### 4.7 `backend/run_daily_analysis.py`

1. Treat room statuses from new viability/statistical gates as blocking for promotion.
2. Separate run-level failure reasons in persisted metadata:
   - `preprocessing_contract_failed`
   - `data_viability_failed`
   - `statistical_validity_failed`
   - `walk_forward_failed`
   - `global_gate_failed`
3. Preserve existing non-destructive handling only for config/policy resolution failures.

### 4.8 `backend/ml/evaluation.py`

1. Emit per-fold class support and usable-fold count.
2. Gate out folds below support minimum; fail evaluation when insufficient valid folds remain.
3. Persist fold exclusion reasons in evaluation report.

### 4.9 Tests

Required additions/updates:

1. `backend/tests/test_data_loader_resampling.py`
   - assert resampling exceptions fail closed (no legacy fallback).
2. `backend/tests/test_pipeline_duplicate_precedence.py`
   - keep existing precedence tests; add viability-fail path assertions.
3. `backend/tests/test_training.py`
   - add statistical-validity gate tests for low support.
   - add deterministic seed reproducibility test.
   - add idempotent fingerprint skip test.
4. `backend/tests/test_registry.py`
   - add promoted/current_version consistency tests.
   - add orphan alias cleanup tests.
5. New integration test:
   - full run with induced resampling failure; verify no promotion and explicit failure reason.

## 5. Data/DB and Monitoring Changes

### 5.1 Training History Metadata

Add fields to stored metadata:

1. `data_viability`
2. `statistical_validity`
3. `data_fingerprint`
4. `policy_hash`
5. `code_version`
6. `run_failure_stage`

### 5.2 Dashboard

Add production counters:

1. room preprocessing fails/day
2. room viability fails/day
3. room statistical validity fails/day
4. candidate promotions/day
5. no-op retrain skips/day

## 6. Rollout Plan (No Temporary Workarounds)

### Phase A: Contract Enforcement (Week 1)

1. Remove fail-open fallback paths.
2. Introduce data viability + statistical validity policies and checks.
3. Ship tests for fail-closed behavior.

### Phase B: Registry and Determinism (Week 2)

1. Registry consistency repair and strict loading semantics.
2. Deterministic seed and fingerprint-based idempotency.

### Phase C: Observability and Backfill (Week 3)

1. Backfill/repair existing room registries.
2. Deploy monitoring counters and dashboard panels.
3. Run shadow verification for 7 days before enabling auto-promotion on all rooms.

## 7. Exit Criteria for RFC Closure

All must be true for 14 consecutive days:

1. 0 silent fallback incidents.
2. 0 promotions from statistically invalid candidates.
3. 0 registry inconsistency incidents at load time.
4. Retrain idempotency skip rate reflects duplicate-trigger behavior without candidate churn.
5. No unexplained run-level failures in automation logs.

