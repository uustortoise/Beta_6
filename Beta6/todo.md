# Beta 6 Follow-Up TODO

- Date: 2026-02-27 (rebaselined for pre-Phase-5 start)
- Source: Team review feedback + unresolved execution details from kick-start plan
- Scope: Open items not yet fully specified or implemented

## 0. Pre-Phase-5 Rebaseline and Entry Checks

- [x] Close P2 Stage-4 artifact-path regression:
  - Detail: `phase4_dynamic_gate` error path no longer reports nonexistent artifact files.
  - Evidence: `backend/run_daily_analysis.py` + `backend/tests/test_run_daily_analysis_beta6_authority.py::test_beta6_authority_stage4_error_does_not_report_nonexistent_paths`.
  - Acceptance: Error status returns `evaluation_report_path=None` and `rejection_artifact_path=None`.

- [x] Add fail-closed full Beta6 YAML schema gate and CI enforcement.
  - Deliverable: `backend/ml/beta6/beta6_schema.py`, `backend/scripts/check_beta6_config_schema.py`, `backend/tests/test_beta6_config_schema.py`.
  - Acceptance: `check_beta6_config_schema.py` reports pass for all expected `beta6_*.yaml`.

- [x] Add required Beta6 import smoke gate in CI.
  - Deliverable: `backend/scripts/check_beta6_import_smoke.py` wired in CI.
  - Acceptance: required imports include `run_daily_analysis`, `ml.beta6.contracts`, `ml.beta6.gate_engine`, `ml.beta6.orchestrator`.

- [x] Enforce canonical module-path policy and shim deprecation deadline.
  - Owner: ML Platform Lead
  - Deliverable: Published shim inventory + removal date and CI guard against new shim-path imports.
  - Evidence: `docs/planning/beta6_shim_deprecation_inventory.md`, `backend/tests/test_beta6_shim_import_guard.py`, `.github/workflows/ci-gate-hardening.yml`.
  - Acceptance: No new imports from deprecated `backend/ml/beta6/*.py` shim modules; CI guard enforced.

- [x] Add real-data canary gate before pilot ladder expansion beyond Rung 1.
  - Owner: Release Manager + Modeling Lead
  - Deliverable: Policy-configured real-data evidence contract checked in canary artifact path.
  - Evidence: `backend/config/beta6_canary_gate.yaml`, `backend/ml/t80_rollout_manager.py`, `backend/tests/test_t80_rollout.py::test_evaluate_canary_artifacts_fails_without_real_data_evidence`.
  - Acceptance: Canary artifact evaluation fails closed when real-data evidence is missing/invalid, blocking rung progression.

## 1. Sensor Onboarding and Signal Quality

- [ ] [Needs Ops discussion] Finalize room-type placement standard with diagrams and install tolerances.
  - Owner: Field Ops Lead
  - Deliverable: `docs/planning/beta6_sensor_onboarding_protocol.md`
  - Acceptance: Approved checklist for bedroom/livingroom/bathroom installs.

- [ ] Lock numeric signal-quality thresholds for pilot gates.
  - Owner: MLOps Lead
  - Deliverable: `backend/config/beta6_sensor_onboarding_policy.yaml`
  - Acceptance: Thresholds signed off for flatline, stuck-value, missingness, and motion/light AUC drift.

- [ ] Define activation state machine and reason codes for room readiness.
  - Owner: ML Platform Lead
  - Deliverable: `observe_only -> ml_evaluable -> blocked_for_recalibration` transition rules.
  - Acceptance: Deterministic reason-code mapping in gate artifacts.

- [ ] Add nightly sensor-health report to operations dashboard.
  - Owner: MLOps Lead
  - Deliverable: Export with per-room health status and recalibration queue.
  - Acceptance: Report generated nightly and linked in go/no-go packet.

## 2. Duration Priors Semantics and Tuning

- [ ] Specify exact scoring equations for duration-prior penalties in HMM and CRF.
  - Owner: Modeling Lead
  - Deliverable: Formula section in design doc and inline code comments in decoders.
  - Acceptance: Implementation matches documented `option (b)` semantics.

- [ ] Decide initial penalty coefficients and per-activity bounds.
  - Owner: Modeling Lead
  - Deliverable: `backend/config/beta6_duration_prior_policy.yaml`
  - Acceptance: Coefficients versioned with rationale and fallback defaults.

- [ ] Run HMM/CRF ablation to quantify prior impact.
  - Owner: ML Engineers
  - Deliverable: Before/after metrics on ping-pong rate, dwell realism, and F1.
  - Acceptance: Report attached to Phase 4/5 sign-off.

## 3. Rollback Readiness (Beta 5.5 Champions)

- [ ] Inventory all Beta 5.5 champion pointers and artifact bundles for pilot cohort.
  - Owner: Platform Engineer
  - Deliverable: Rollback inventory manifest (pointer -> artifact path -> checksum).
  - Acceptance: 100% coverage for pilot cohort residents/rooms.

- [ ] Add automated load test for rollback targets.
  - Owner: QA Lead
  - Deliverable: CI/runtime check that each rollback candidate is loadable.
  - Acceptance: Fail-closed behavior if any rollback target is missing/corrupt.

- [ ] Rehearse rollback drill using validated Beta 5.5 champions.
  - Owner: Release Manager
  - Deliverable: Drill report with timings and incident notes.
  - Acceptance: One-command rollback succeeds within operational SLA.

## 4. Policy and Documentation Follow-Through

- [ ] Update go/no-go template to include sensor-quality and rollback-target validation sections.
  - Owner: ML Platform Lead
  - Deliverable: `docs/planning/template_beta55_to_beta6_go_no_go.md` revision.
  - Acceptance: Template cannot pass without these sections filled.

- [ ] Assign named individuals (not role labels only) for all Phase 1 and Phase 4 gate items.
  - Owner: Release Manager
  - Deliverable: Ownership table with primary + backup.
  - Acceptance: No unowned gate item remains.

- [ ] Lock Week-1 sign-off package contents and review cadence.
  - Owner: Product + Clinical + Ops
  - Deliverable: Signed checklist and recurring review schedule.
  - Acceptance: Package approved before ladder rung 1 activation.

## 5. Phase 0 Data/Label Contract Gate

- [x] Finalize Beta 6 intake runbook for `validate + diff + smoke` workflow.
  - Owner: QA Lead
  - Deliverable: `docs/planning/beta6_label_pack_intake_runbook.md`
  - Acceptance: Runbook includes fail-closed criteria and artifact retention path.

- [x] Wire intake bundle requirement so no Phase 1+ model run starts without approved intake artifact.
  - Owner: ML Platform Lead
  - Deliverable: Pipeline precheck in orchestrator/scheduler.
  - Acceptance: Training job exits early with deterministic reason code when intake artifact is missing/failed.

- [x] Add CI check for intake artifact schema.
  - Owner: QA Lead
  - Deliverable: Test suite validating required keys/paths in intake bundle.
  - Acceptance: CI fails on malformed or incomplete intake artifacts.

## 6. Uncertainty Taxonomy and Capability-Aware Gates

- [ ] [Needs Ops discussion] Lock reason-code and routing mapping for `low_confidence`, `unknown`, and `outside_sensed_space`.
  - Owner: ML Platform Lead
  - Deliverable: `docs/planning/beta6_uncertainty_taxonomy_contract.md` + `backend/config/beta6_uncertainty_policy.yaml`.
  - Acceptance: No pipeline path emits ambiguous/merged uncertainty states.

- [x] Publish room capability gate profiles (for example bedroom/livingroom) with justified thresholds.
  - Owner: Modeling Lead
  - Deliverable: `backend/config/beta6_room_capability_gate_profiles.yaml` + deterministic selector in `backend/ml/beta6/capability_profiles.py`.
  - Acceptance: Gate engine selects profile deterministically by room type and outputs profile id in artifacts.

- [x] Add timeline hard-gate policy text and checks (MAE + fragmentation primary).
  - Owner: Modeling Lead
  - Deliverable: `backend/config/beta6_timeline_hard_gates.yaml` + `backend/ml/beta6/timeline_hard_gates.py` + regression tests.
  - Acceptance: Promotion is blocked when timeline hard gates fail even if F1 remains acceptable.

## 7. Runtime/Eval Parity, Fallback, and SLO Observability

- [x] Implement runtime/eval fixed-trace parity test harness.
  - Owner: QA Lead
  - Deliverable: `backend/tests/test_beta6_runtime_eval_parity.py` + `backend/ml/beta6/runtime_eval_parity.py`.
  - Acceptance: CI blocks on any label/decoder/uncertainty mismatch between runtime and eval paths.

- [x] Implement deterministic operator-safe fallback mode instrumentation.
  - Owner: Platform Engineer
  - Deliverable: `RegistryV2.activate_fallback_mode/clear_fallback_mode` + policy/doc (`backend/config/beta6_fallback_mode_policy.yaml`, `docs/planning/beta6_fallback_mode_instrumentation.md`).
  - Acceptance: Fallback transitions are fully traceable and recoverable by policy.

- [x] Lock model-behavior SLO alert thresholds and escalation routing.
  - Owner: MLOps Lead
  - Deliverable: `backend/ml/beta6/slo_observability.py` + `backend/config/beta6_model_behavior_slo.yaml` + daily report contract docs.
  - Acceptance: SLO breach creates actionable alert with owner and remediation ETA.

## 8. Phase 2-6 Implementation Gaps

- [ ] [Needs Ops discussion] Confirm Beta 6 pilot cutover policy while Phase 2-6 modules are still partial.
  - Owner: Release Manager + Ops Lead
  - Deliverable: Signed policy on whether to block pilot expansion or allow phased enablement by completed controls.
  - Acceptance: Explicit go/no-go decision with fallback SLA and owner.

## 9. Phase 2 Follow-Up (Post Implementation)

- [ ] Confirm approved unlabeled corpus source list and retention policy before large-scale pretraining runs.
  - Owner: Data Governance + MLOps
  - Deliverable: Corpus source whitelist + retention note attached to pretrain manifest runbook.
  - Acceptance: No unmanaged source path in production pretraining manifests.

- [ ] Lock representation gate minimum margins by room/capability profile for pilot promotion packets.
  - Owner: Modeling Lead
  - Deliverable: Threshold table (linear-probe margin + kNN purity) added to go/no-go template.
  - Acceptance: Representation report has deterministic pass/fail thresholds.
