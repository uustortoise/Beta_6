# Beta 6 Implementation Checklist (File-Mapped)

- Status: Draft
- Date: February 16, 2026
- Depends on: `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/rfc_beta6_greenfield_architecture.md`
- Start condition: Beta 5.5 exit gate passed

## 1. Scope

Translate the Beta 6 architecture RFC into concrete implementation tasks mapped to current repository files.

## 2. Week 0: Beta 5.5 Validation and Contract Freeze

### Deliverables

1. Confirm Beta 5.5 release exit metrics and freeze Beta 6 `RunSpec v1`.
2. Publish module ownership and code boundaries.
3. Freeze initial artifact schema and gate reason schema.

### Files to update

1. `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/rfc_beta6_greenfield_architecture.md`
2. `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/production_ml_hardening_execution_plan.md`
3. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/unified_training_spec.md`

### Acceptance checks

1. Team sign-off recorded on Beta 6 contract version.
2. No open P0 Beta 5.5 defects in active training/promotion path.

## 3. Week 1: Contracts and Orchestrator Skeleton

### Implementation tasks

1. Add typed contract package for run spec, artifact references, room/run decisions, and gate reasons.
2. Add validators for schema, enumerations, and required fields.
3. Add orchestration skeleton for validate -> materialize -> train -> evaluate -> gate -> finalize.

### New files

1. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/contracts/__init__.py`
2. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/contracts/run_spec.py`
3. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/contracts/artifacts.py`
4. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/contracts/decisions.py`
5. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/contracts/events.py`
6. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/orchestrator.py`

### Existing files to touch

1. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/__init__.py`
2. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/exceptions.py`

### Tests

1. New: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_beta6_contracts.py`
2. New: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_beta6_orchestrator_skeleton.py`

## 4. Week 2: Deterministic Feature Store

### Implementation tasks

1. Build deterministic feature materializer with content fingerprinting.
2. Persist `FeatureSnapshot` and `snapshot_meta.json`.
3. Add cache reuse logic keyed by feature fingerprint.

### New files

1. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/feature_store.py`
2. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/feature_fingerprint.py`

### Existing files to integrate

1. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/data_quality_contract.py`
2. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/duplicate_resolution.py`
3. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/sequence_alignment.py`
4. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/coverage_contract.py`

### Tests

1. New: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_beta6_feature_store.py`
2. Update: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_data_loader_resampling.py`

## 5. Week 3: Training Fabric (Candidate-Only)

### Implementation tasks

1. Train room candidates from `FeatureSnapshot` without direct champion promotion.
2. Persist candidate artifact bundles with metadata.
3. Carry deterministic fields: seed, policy hash, code version, dependency lock hash.

### New files

1. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/training_fabric.py`
2. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/candidate_store.py`

### Existing files to integrate

1. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/training.py`
2. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/reproducibility_report.py`
3. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/policy_config.py`

### Tests

1. New: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_beta6_training_fabric.py`
2. Update: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_training.py`

## 6. Week 4: Evaluation and Gate Engine

### Implementation tasks

1. Evaluate candidates using walk-forward and support diagnostics.
2. Compute room-level decisions with explicit rejection reasons.
3. Compute run-level decisions with ordered precedence.

### New files

1. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/evaluation_engine.py`
2. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/gate_engine.py`

### Existing files to integrate

1. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/evaluation.py`
2. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/release_gates.py`
3. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/statistical_validity_gate.py`
4. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/class_coverage_gate.py`
5. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/rejection_artifact.py`

### Tests

1. New: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_beta6_gate_engine.py`
2. New: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_beta6_evaluation_engine.py`
3. Update: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_run_daily_analysis_thresholds.py`

## 7. Week 5: Registry V2 and Serving Pointer Loader

### Implementation tasks

1. Add append-only room event log.
2. Add atomic `champion_pointer.json` update with fail-safe rollback behavior.
3. Add serving loader that only consumes pointer-selected champion bundles.

### New files

1. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/registry_v2.py`
2. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/registry_events.py`
3. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/serving_loader.py`

### Existing files to integrate

1. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/registry.py`
2. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/registry_validator.py`
3. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/prediction.py`
4. `/Users/dicksonng/DT/Development/Beta_5.5/backend/health_server.py`

### Tests

1. New: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_beta6_registry_v2.py`
2. New: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_beta6_serving_loader.py`
3. Update: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_registry.py`

## 8. Week 6: Dual-Run Shadow (No Promotion Authority Switch Yet)

### Implementation tasks

1. Add beta6 shadow execution flag and run side-by-side with Beta 5.5.
2. Persist decision comparisons and divergence reasons.
3. Keep promotion authority on Beta 5.5 only.

### Files to update

1. `/Users/dicksonng/DT/Development/Beta_5.5/backend/run_daily_analysis.py`
2. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/beta6/orchestrator.py`
3. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/reproducibility_report.py`

### Tests

1. New: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_beta6_shadow_mode.py`
2. Update: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_run_daily_analysis_thresholds.py`

## 9. Week 7: Controlled Cutover (Cohort-Based)

### Implementation tasks

1. Add resident-cohort flag for Beta 6 promotion authority.
2. Keep one-command rollback to Beta 5.5 serving and promotion.
3. Add cohort-specific incident dashboard metrics.

### Files to update

1. `/Users/dicksonng/DT/Development/Beta_5.5/backend/run_daily_analysis.py`
2. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/policy_presets.py`
3. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/pilot_override_manager.py`
4. `/Users/dicksonng/DT/Development/Beta_5.5/backend/export_dashboard.py`

### Tests

1. New: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_beta6_cutover_flags.py`
2. Update: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_health_server.py`

## 10. Week 8: Full Cutover and Beta 5.5 Promotion Path Deprecation

### Implementation tasks

1. Switch all eligible residents to Beta 6 promotion authority.
2. Retain Beta 5.5 code path behind explicit fallback flag for stabilization window.
3. Finalize docs and runbooks.

### Files to update

1. `/Users/dicksonng/DT/Development/Beta_5.5/backend/run_daily_analysis.py`
2. `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/pipeline.py`
3. `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/rfc_beta6_greenfield_architecture.md`
4. `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/roadmap_ml_stabilization.md`

### Tests

1. Full regression suite under `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/`
2. Add release certification summary document in `docs/planning/`

## 11. Cross-Cutting Guardrails (Every Week)

1. Do not allow direct promotion from training module APIs.
2. Every gate failure must map to one deterministic reason code.
3. Every persisted artifact must include `run_id`, `policy_hash`, and `code_version`.
4. Changes to contracts require version increment and migration note.

## 12. Definition of Done for Beta 6 Launch

1. 14 consecutive days without destructive promotion incident.
2. Shadow/cutover divergence explanations available for all mismatches.
3. Registry pointer integrity checks pass in CI and in daily runtime audits.
4. Run replay with same fingerprint produces no-op and identical decision summary.
