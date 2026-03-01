# Beta 6 100-User Pilot Kick-Start Plan

- Status: Rebaselined to current implementation (Phase 5 execution)
- Date: 2026-02-27 (updated from 2026-02-25 draft)
- Scope: Build and launch Beta 6 to a controlled 100-user market test cohort
- Strategy: Baseline-first (`Head + HMM`) then upgrade (`LoRA + CRF`)

## 0. Current Implementation Rebaseline (2026-02-27)

1. Phase 0/1 controls are in-repo and CI-gated (contracts, leakage guards, intake gate, runtime/eval parity).
2. Phase 2/3/4 baseline implementation is in-repo (manifest/pretrain/representation, safe-fine-tune/triage, dynamic head + HMM + unknown path + dynamic gate artifacts).
3. Stage-4 artifact error-path regression (nonexistent file paths on error) is closed and protected by regression tests in `tests/test_run_daily_analysis_beta6_authority.py`.
4. Beta6 config hardening is now fail-closed in CI:
   - `backend/ml/beta6/beta6_schema.py`
   - `backend/scripts/check_beta6_config_schema.py`
   - `backend/scripts/check_beta6_import_smoke.py`
5. Phase 5 core implementation is now in-repo:
   - Step 5.1 adapter lifecycle artifacts implemented (`lora_adapter.py`, `adapter_store.py`) with lifecycle tests.
   - Step 5.2 CRF production decoder implemented with HMM-vs-CRF A/B gate path in orchestrator/runtime hooks.
   - Step 5.3 ML Health Snapshot hardening implemented in health/reporting path (`health_server.py`, `export_dashboard.py`, shared `evaluation_metrics.py` helpers).
6. Step 1.4 sensor onboarding/signal-quality controls are still pending Ops finalization and must remain tracked as open readiness debt.
7. Canonical package policy:
   - Use subpackage modules (`ml.beta6.training/*`, `ml.beta6.serving/*`, `ml.beta6.registry/*`, `ml.beta6.evaluation/*`, `ml.beta6.data/*`, `ml.beta6.gates/*`).
   - Legacy shim modules under `backend/ml/beta6/*.py` are compatibility-only and scheduled for removal by end of Phase 5.
8. Real-data canary evidence gate is fail-closed for artifact-based canary evaluation (missing/invalid evidence blocks promotion beyond Rung 1).

## 1. Objective and Non-Negotiables

### Objective
Deploy Beta 6 to 100 pilot users with measurable quality and safe rollback, while proving the architecture can scale to larger cohorts without redesign.

### Non-negotiables
1. No hard-coded room/activity labels in code paths.
2. No promotion without pass/fail artifacts and deterministic reason codes.
3. No model evaluation with resident leakage or time leakage.
4. Unknown behavior must be explicit (`unknown`/`abstain`), not force-classified.
5. One-command rollback path must be validated before production cutover.
6. No room enters ML evaluation unless sensor onboarding and signal-quality gates are passed.

## 2. Architecture Contract (Locked)

## 2.1 Pretrain -> Adapt -> Sequence
1. Pretrain: self-supervised backbone on broad unlabeled multi-resident data.
2. Adapt: per-resident lightweight adapter/head updates (LoRA-style in Phase 5).
3. Sequence: per-home sequence constraints from YAML adjacency, then duration priors.

## 2.2 Backbone policy
1. Core backbone weights frozen during pilot window except scheduled model-refresh events.
2. Resident learning happens through adapters/head only.
3. Backbone refresh cadence: monthly at most during pilot; emergency refresh only with incident review.

## 2.3 Per-home YAML registry (required)
Each deployment must provide:
1. Room definitions and sensor mapping.
2. Activity list per room (mandatory: `active_use`, `unoccupied`).
3. Adjacency graph for physically possible room transitions.
4. Optional duration priors per activity.

Example:

```yaml
home_id: HK002
rooms:
  - name: bedroom
    sensor_id: sensor_01
    activities: [sleep, nap, active_use, unoccupied]
  - name: livingroom
    sensor_id: sensor_02
    activities: [active_use, passive_rest, unoccupied]
adjacency:
  bedroom: [livingroom]
  livingroom: [bedroom]
duration_priors_minutes:
  sleep: {min: 30, max: 600}
  passive_rest: {min: 10, max: 180}
```

## 2.4 Unknown handling (required)
1. Add explicit `unknown` output path or abstention policy per room.
2. Trigger abstain when confidence/calibration criteria fail.
3. Route abstained windows to review/triage pipeline; do not silently relabel.

## 2.5 Duration prior semantics (locked)
1. Adjacency constraints are hard masks (physically impossible transitions remain impossible).
2. Duration priors are soft penalties (`option b`): apply a log-linear penalty term in sequence scoring, not a hard cutoff.
3. HMM baseline and CRF upgrade both consume the same duration-prior contract from YAML.
4. Priors bias decoding toward plausible dwell times but can be overridden when observation evidence is strong.
5. Any absolute max-duration safety caps must be explicitly documented in policy and treated separately from priors.

## 2.6 Uncertainty taxonomy (locked)
1. `low_confidence`:
   - Meaning: model is uncertain between known classes.
   - Trigger: confidence/calibration below class-specific threshold, but still within sensed state space.
   - Routing: triage queue for review and potential relabeling.
2. `unknown`:
   - Meaning: observed pattern is ambiguous or out-of-distribution relative to known activity semantics.
   - Trigger: uncertainty/OOD logic indicates semantic ambiguity.
   - Routing: triage queue with higher-priority adjudication.
3. `outside_sensed_space`:
   - Meaning: likely behavior in unsensored or unreliable coverage zone.
   - Trigger: sensor-health/context evidence indicates blind spot or coverage failure.
   - Routing: field-ops/sensor-onboarding workflow, not only label adjudication.
4. Reporting requirement:
   - Metrics and reason codes must be tracked separately for all three states.
   - These states must not be collapsed into one bucket in gate artifacts.

## 2.7 Runtime and evaluation parity contract (locked)
1. Runtime and offline evaluation/backtest must use the same:
   - Label registry and mapping.
   - Decoder semantics (adjacency masks, duration priors, unknown handling).
   - Confidence calibration and abstain decisions.
2. Any intentional divergence requires:
   - Versioned contract update.
   - Migration note + sign-off.
3. Parity is validated through fixed-trace replay tests before promotion authority switch.

## 3. Pilot Scope (100 Users)

### Included
1. Dynamic registry, dynamic heads, dynamic gating.
2. Baseline sequence model (`HMM`) with per-home adjacency + duration penalties.
3. Shadow mode comparison against Beta 5.5 before promotion authority switch.
4. Nightly retraining and promotion decision pipeline for pilot cohort.

### Excluded in initial pilot
1. Global full-fleet cutover.
2. Unlimited open-ended new class onboarding without review thresholds.
3. Backbone continuous online learning.

## 4. Success Metrics and Launch Gates (Initial v1 Thresholds)

| Category | Metric | Pilot Gate (v1) | Fail Trigger |
|---|---|---:|---:|
| Timeline | MAE per room vs Beta 5.5 champion | <= +5% median regression | > +10% any room |
| Timeline | Fragmentation index vs Beta 5.5 champion | <= +10% median regression | > +20% any room |
| Classification | Per-activity F1 | >= 0.65 for mandatory classes | < 0.55 worst mandatory class |
| Classification | Worst-room F1 floor | >= 0.60 | < 0.50 |
| Alerting | Alert precision | >= 0.85 | < 0.75 |
| Alerting | Alert recall | >= 0.80 | < 0.70 |
| Calibration | Brier score (per room) | non-regression vs Beta 5.5 | > 10% worse |
| Unknown/Abstain | Unknown detection recall | >= 0.70 on review set | < 0.60 |
| Unknown/Abstain | Abstain rate | 2% to 15% expected band | > 20% or < 1% |
| Drift | Feature distribution shift alerts | <= 3 critical shifts/week | > 5 critical shifts/week |
| Ops | Nightly job success rate | >= 99% | < 97% |

Notes:
1. Thresholds are starting defaults; lock final values during Week 1 sign-off.
2. All metrics must be resident-disjoint/time-disjoint in evaluation.
3. Timeline metrics are primary hard gates (non-regression on timeline quality is mandatory).
4. Classification and alerting thresholds are capability-aware by room profile (for example bedroom vs livingroom) with documented room-specific floors.

## 5. Data and Label Governance

## 5.1 Golden data strategy
1. Keep `N=20` Golden cohort focused on safe classes (`sleep`, `shower`, `out`) for alignment.
2. Do not over-generalize ambiguous classes from this set.
3. Expand Golden set continuously via active-learning triage feedback.

## 5.2 Leakage controls (mandatory before any score is trusted)
1. Resident-disjoint splits: no resident appears in both train and validation folds.
2. Time-disjoint splits: no overlapping windows across train/validation boundaries.
3. Window-overlap control: enforce gap buffer between train and validation windows.
4. Feature leakage audit: ensure no future-derived features in training rows.

## 5.3 Representation quality checks (replace t-SNE gate)
1. Linear probe accuracy on held-out resident-disjoint set.
2. kNN retrieval purity for safe-class neighborhoods.
3. Resident-disjoint downstream head performance delta vs random encoder baseline.

## 5.4 Active-learning triage at scale
1. Use percentile-based confidence thresholds per room/class, not fixed 0.6.
2. Add disagreement sampling (backbone/head disagreement).
3. Add diversity sampling to avoid near-duplicate review windows.
4. Review policy:
   - Green labels: 1 reviewer.
   - Amber/Red labels: 2 reviewers + adjudication if disagreement.
5. Throughput SLA: >= 50 reviewed corrections per operator per week.

## 5.5 Privacy and governance gate
1. Confirm consent scope for unlabeled pretraining corpus.
2. Confirm retention period and purge policy.
3. Confirm de-identification/anonymization rules for training artifacts.
4. Block progression if compliance sign-off is missing.

## 5.6 Sensor onboarding protocol (required before room activation)
1. Standard placement guide per room type:
   - PIR mounting height, angle, and expected field of view.
   - Minimum/maximum distance from bed, sofa, and high-traffic path.
   - Placement exclusions (windows, HVAC vents, reflective surfaces).
2. Commissioning checklist before room activation:
   - Verify sensor mapping in YAML (`room -> sensor_id`) against physical install.
   - Run a guided walk-test and occupancy/no-occupancy sanity checks.
   - Record install metadata (position, angle, firmware version, installer, timestamp).
3. Nightly sensor-health monitoring in pipeline:
   - Flatline detection.
   - Stuck-value detection.
   - Missingness/dropout and heartbeat checks.
   - Motion/light AUC drift checks against room-type baseline.
4. Minimum signal-quality gate:
   - Room remains `observe_only` until it passes configured quality floors for consecutive nights.
   - Failing rooms are blocked from promotion and routed to field-ops recalibration queue.
5. Ownership:
   - Field Ops owns placement and recalibration.
   - MLOps owns nightly health checks and gate enforcement.

## 6. Adapter Lifecycle (for 100 Users and Beyond)

1. Create: initialize adapter on resident onboarding with default template.
2. Warm-up: train adapter/head on resident historical data with fallback thresholds.
3. Promote: allow adapter promotion only after room gates pass.
4. Retire: archive inactive resident adapters after retention window.
5. Rollback: keep prior promoted adapter version for immediate swap-back.
6. Storage budget guideline: cap active adapter artifacts per resident (for example last 14 versions).
7. Retrain cadence guideline: nightly for active residents, weekly for low-activity residents.

## 7. Sequence Modeling Plan

## Phase baseline (Weeks 5-6)
1. Implement HMM with:
   - Per-home adjacency transition mask from YAML.
   - Duration penalties to reduce impossible ping-pong transitions.
2. Use forward-backward smoothing and Viterbi decode for output sequence.
3. Keep operator-safe fallback path to baseline HMM/rule profile available by feature flag.

## Phase upgrade (Weeks 7-9)
1. Replace HMM with CRF:
   - Learnable transition parameters with adjacency constraints.
   - Duration features or semi-Markov variant for dwell-time realism.
2. Keep HMM path as fallback feature flag until CRF stability proven.

## 8. 12-Week Execution Plan (Baseline-First)

## Phase 0 (Week 0): Data/Label Contract Gate (Mandatory Before Model Work)

### Step 0.1: Enforce label-pack intake contract (`validate + diff + smoke`)
- Task purpose: Prevent model tuning on invalid or inconsistent label packs.
- Code/files to be changed:
1. `backend/scripts/validate_label_pack.py`
2. `backend/scripts/diff_label_pack.py`
3. `backend/scripts/run_event_first_smoke.py` (or Beta 6 equivalent smoke runner)
4. `docs/planning/beta6_label_pack_intake_runbook.md` (new)
- Testing procedures:
1. Run validation on both clean and intentionally corrupted packs.
2. Run diff against prior approved pack and verify deterministic change report.
3. Run smoke on validated pack and verify expected outputs and schema.
- Pass condition:
1. Intake artifact includes validation report, diff report, and smoke report.
2. Any failed intake component blocks progression to Phase 1+.
3. No Phase 1+ training/evaluation job runs without an approved intake bundle.

## Phase 1 (Weeks 1-2): Contracts, Policy, and Registry Freeze

### Step 1.1: Freeze RunSpec v1 and YAML registry schema
- Task purpose: Lock all interfaces so model, evaluation, and ops can move in parallel without schema drift.
- Code/files to be changed:
1. `backend/ml/beta6/contracts/run_spec.py`
2. `backend/ml/beta6/contracts/decisions.py`
3. `backend/ml/beta6/contracts/events.py`
4. `backend/ml/unified_training_spec.md`
5. `docs/planning/rfc_beta6_greenfield_architecture.md`
- Testing procedures:
1. Add schema validator tests with valid and invalid fixtures.
2. Run contract serialization/deserialization tests in CI.
3. Run backward-compatibility checks for prior run specs.
- Pass condition:
1. Contract test suite passes.
2. Schema hash is versioned and published.
3. Team sign-off recorded on frozen contract set.

### Step 1.2: Define gate reason codes and rollback policy
- Task purpose: Ensure every reject/promotion/rollback event is deterministic and auditable.
- Code/files to be changed:
1. `backend/ml/beta6/registry/gate_engine.py`
2. `backend/ml/beta6/registry/registry_v2.py`
3. `backend/ml/beta6/registry/registry_events.py`
4. `docs/planning/template_beta55_to_beta6_go_no_go.md`
- Testing procedures:
1. Unit-test all gate branches to assert non-empty reason codes.
2. Simulate rollback path and verify pointer restoration.
3. Validate event log append order and idempotency.
- Pass condition:
1. 100% gate outcomes map to a deterministic reason code.
2. Rollback drill succeeds with no manual file edits.

### Step 1.3: Implement leakage guardrails in CI
- Task purpose: Prevent invalid model gains from resident/time/window overlap leakage.
- Code/files to be changed:
1. `backend/tests/test_beta6_leakage_guards.py` (new)
2. `backend/ml/beta6/evaluation/evaluation_engine.py`
3. `backend/ml/beta6/data/feature_store.py`
4. `backend/.github/workflows` CI config (or equivalent pipeline config)
- Testing procedures:
1. Add synthetic overlap test cases for resident and temporal leakage.
2. Add window-boundary overlap test with enforced gap buffer.
3. Run CI with negative fixtures that must fail.
- Pass condition:
1. Leakage tests fail on injected overlap and pass on clean splits.
2. CI blocks merge when leakage rules are violated.

### Step 1.4: Implement sensor onboarding and signal-quality gates
- Task purpose: Ensure model quality is not limited by preventable sensor calibration/placement faults.
- Code/files to be changed:
1. `backend/ml/beta6/sensor_health.py` (new)
2. `backend/config/beta6_sensor_onboarding_policy.yaml` (new)
3. `backend/run_daily_analysis.py`
4. `docs/planning/beta6_sensor_onboarding_protocol.md` (new)
- Testing procedures:
1. Simulate flatline/stuck-value/missingness cases and verify gate failures.
2. Validate room activation state transition (`observe_only` -> `ml_evaluable`) from nightly pass streak.
3. Validate failing rooms are excluded from promotion decisions.
- Pass condition:
1. Every pilot room has onboarding checklist artifact.
2. Sensor health gate blocks failed rooms deterministically with reason codes.
3. At least one rehearse-and-recover recalibration flow is completed end-to-end.

### Step 1.5: Lock runtime/eval parity contract and tests
- Task purpose: Prevent behavior drift between offline evaluation and production runtime.
- Code/files to be changed:
1. `backend/ml/beta6/orchestrator.py`
2. `backend/ml/beta6/sequence/hmm_decoder.py` (or shared decoder module)
3. `backend/ml/beta6/evaluation/runtime_eval_parity.py`
4. `backend/tests/test_beta6_runtime_eval_parity.py` (new)
- Testing procedures:
1. Replay fixed traces through evaluation and runtime paths.
2. Assert identical decoded labels, uncertainty states, and reason-code outputs.
3. Add negative fixture to verify parity mismatch fails CI.
- Pass condition:
1. Runtime/eval parity tests pass in CI.
2. No unapproved semantic divergence exists between runtime and eval paths.

## Phase 2 (Weeks 2-3): Self-Supervised Backbone Pretraining

### Step 2.1: Build unlabeled corpus manifest and fingerprinting
- Task purpose: Create deterministic pretraining data inputs that can be replayed and audited.
- Code/files to be changed:
1. `backend/scripts/build_pretrain_corpus_manifest.py` (new)
2. `backend/ml/beta6/data/data_manifest.py` (new)
3. `backend/ml/beta6/data/feature_fingerprint.py`
4. `docs/planning/beta6_pretrain_corpus_manifest.md` (new)
- Testing procedures:
1. Run manifest builder twice on same input and compare fingerprints.
2. Validate dedupe rules and missing-data handling.
3. Check corpus stats report against expected counts.
- Pass condition:
1. Deterministic fingerprint is stable across reruns.
2. Data quality report has no P0 data contract violations.

### Step 2.2: Implement self-supervised pretraining pipeline
- Task purpose: Learn robust sensor representations without requiring large labeled datasets.
- Code/files to be changed:
1. `backend/ml/beta6/training/self_supervised_pretrain.py` (new)
2. `backend/ml/beta6/training/beta6_trainer.py`
3. `backend/config/beta6_pretrain.yaml` (new)
4. `backend/ml/transformer_backbone.py` (if architecture hooks are needed)
- Testing procedures:
1. One-epoch smoke test on small corpus.
2. Checkpoint save/load consistency test.
3. Deterministic seed test on fixed sample.
- Pass condition:
1. Pretraining run completes and writes checkpoint artifacts.
2. Seeded rerun has bounded metric variance within agreed tolerance.

### Step 2.3: Add representation quality report (quantitative)
- Task purpose: Replace visual-only checks with measurable embedding quality gates.
- Code/files to be changed:
1. `backend/ml/beta6/evaluation/representation_eval.py` (new)
2. `backend/scripts/run_representation_eval.py` (new)
3. `docs/planning/beta6_representation_gate_report_template.md` (new)
- Testing procedures:
1. Compute linear-probe accuracy on resident-disjoint holdout.
2. Compute kNN purity for safe-class neighborhoods.
3. Compare against random encoder baseline.
- Pass condition:
1. Representation metrics exceed baseline by agreed margin.
2. Resident-disjoint checks pass with no leakage alerts.

## Phase 3 (Week 4): Golden Safe-Class Fine-Tuning + Triage Engine

### Step 3.1: Fine-tune backbone on Golden safe classes
- Task purpose: Align pretrained embeddings with clinically trusted activity states.
- Code/files to be changed:
1. `backend/scripts/harvest_gold_samples.py`
2. `backend/ml/beta6/training/fine_tune_safe_classes.py` (new)
3. `backend/config/beta6_golden_safe_finetune.yaml` (new)
- Testing procedures:
1. Validate class balance and sample quality of harvested set.
2. Train/validation run with resident-disjoint holdout.
3. Reproducibility rerun with fixed seed.
- Pass condition:
1. Held-out safe-class accuracy >= 85%.
2. No leakage warnings in fine-tune evaluation report.

### Step 3.2: Build active-learning triage service
- Task purpose: Scale labeling by prioritizing highest-value windows for review.
- Code/files to be changed:
1. `backend/ml/beta6/training/active_learning.py` (new)
2. `backend/scripts/run_active_learning_triage.py` (new)
3. `backend/config/beta6_active_learning_policy.yaml` (new)
- Testing procedures:
1. Validate confidence percentile selection per room/class.
2. Validate disagreement and diversity sampling outputs.
3. Check queue generation latency and volume.
- Pass condition:
1. Triage queue is produced daily with policy-compliant mix.
2. No single room/class dominates queue due to bias bug.

### Step 3.3: Implement reviewer workflow and adjudication rules
- Task purpose: Keep label quality high while increasing throughput.
- Code/files to be changed:
1. `backend/config/label_review_policy.yaml` (new)
2. `docs/planning/golden_sample_labeling_guide_beta6.md` (new)
3. `docs/planning/beta6_label_adjudication_runbook.md` (new)
- Testing procedures:
1. Dry-run dual-review workflow on amber/red samples.
2. Verify disagreement resolution path and audit logging.
3. Measure operator throughput during simulation.
- Pass condition:
1. Workflow supports 50+ corrections/week/operator.
2. Amber/red samples consistently receive dual review.

## Phase 4 (Weeks 5-6): Dynamic Head + HMM Baseline (MVP)

### Step 4.1: Implement dynamic head generator from YAML
- Task purpose: Remove hard-coded labels and support arbitrary flat layouts.
- Code/files to be changed:
1. `backend/ml/beta6/serving/head_factory.py` (new)
2. `backend/ml/beta6/orchestrator.py`
3. `backend/ml/beta6/contracts/label_registry.py` (new)
- Testing procedures:
1. Parameterized tests for 2-room, 3-room, and 5-room configs.
2. Validate output tensor shapes against registry activities.
3. Validate mandatory class presence (`active_use`, `unoccupied`).
- Pass condition:
1. Head shape generation is deterministic and config-driven.
2. No hard-coded room/activity names remain in generation path.

### Step 4.2: Implement per-home HMM with adjacency and duration priors
- Task purpose: Stabilize sequence outputs in small flats before CRF rollout using hard adjacency masks plus soft duration-prior penalties.
- Code/files to be changed:
1. `backend/ml/beta6/sequence/hmm_decoder.py` (new)
2. `backend/ml/beta6/sequence/transition_builder.py` (new)
3. `backend/ml/beta6/contracts/label_registry.py`
4. `backend/config/beta6_duration_prior_policy.yaml` (new)
- Testing procedures:
1. Unit-test impossible transition masking from adjacency graph.
2. Unit-test soft duration-penalty behavior on ping-pong and overlong dwell sequences.
3. Unit-test that strong observation likelihood can override priors when appropriate.
4. Run end-to-end decode on known small-flat traces.
- Pass condition:
1. Impossible transitions are not emitted.
2. Ping-pong transition rate drops below defined threshold.
3. Decoder behavior matches documented duration-prior semantics (`option b`).

### Step 4.3: Add unknown/abstain path in inference
- Task purpose: Prevent unsafe forced classifications for unseen/ambiguous states.
- Code/files to be changed:
1. `backend/ml/beta6/serving/prediction.py` (new or integrate into existing scorer)
2. `backend/ml/beta6/evaluation/calibration.py` (new)
3. `backend/ml/beta6/evaluation/evaluation_engine.py`
4. `backend/config/beta6_unknown_policy.yaml` (new)
- Testing procedures:
1. Inject low-confidence windows and verify abstain behavior.
2. Validate reliability calibration and Brier calculations.
3. Verify abstained samples are routed to triage queue.
- Pass condition:
1. Abstain rate stays in configured band.
2. Unknown detection recall meets gate threshold.

### Step 4.4: Build dynamic gating and signed evaluation artifacts
- Task purpose: Produce per-home pass/fail decisions without code edits.
- Code/files to be changed:
1. `backend/ml/beta6/evaluation/evaluation_engine.py`
2. `backend/ml/beta6/registry/gate_engine.py`
3. `backend/ml/beta6/registry/rejection_artifact.py` (new)
- Testing procedures:
1. Generate reports for heterogeneous room/activity configs.
2. Validate metric computation completeness and reason-code mapping.
3. Run non-regression checks against prior champion artifacts.
- Pass condition:
1. Signed report includes all required metrics and reason codes.
2. MVP system passes end-to-end tests on 3 pilot flats.

## Phase 5 (Weeks 7-9): LoRA Adapters + CRF Upgrade

### Phase 5 Entry Gate (Must Pass Before Step 5.1 Starts)
1. Stage-4 artifact-path regression remains closed in required CI subset (`test_run_daily_analysis_beta6_authority.py`).
2. Beta6 config-schema gate passes for all expected `beta6_*.yaml` files (`check_beta6_config_schema.py`).
3. Beta6 import-smoke gate passes (`check_beta6_import_smoke.py`) including `run_daily_analysis` authority path.
4. Canonical module-path + shim deprecation guard is enforced (`test_beta6_shim_import_guard.py`) with documented shim removal deadline (end of Phase 5).
5. Real-data canary evidence gate is fail-closed in canary artifact evaluation (missing/invalid evidence blocks promotion beyond Rung 1).
6. Phase 5 implementation artifacts are now present and validated in the required test subset:
   - `backend/ml/beta6/adapters/lora_adapter.py`
   - `backend/ml/beta6/adapters/adapter_store.py`
   - `backend/ml/beta6/sequence/crf_decoder.py` (production implementation, not placeholder)
7. Sensor onboarding/signal-quality Step 1.4 open items are explicitly tracked and cannot be silently dropped from readiness packets.

### Step 5.1: Implement adapter lifecycle and storage policy
- Task purpose: Enable continual per-resident learning without destabilizing backbone.
- Code/files to be changed:
1. `backend/ml/beta6/adapters/lora_adapter.py` (new)
2. `backend/ml/beta6/adapters/adapter_store.py` (new)
3. `backend/ml/beta6/registry/registry_v2.py`
4. `backend/config/beta6_adapter_policy.yaml` (new)
- Testing procedures:
1. Test create/warm-up/promote/retire transitions.
2. Test per-resident version retention enforcement.
3. Test rollback to prior adapter version.
- Pass condition:
1. Lifecycle transitions succeed with audit trail.
2. Rollback completes within operational SLA.

### Step 5.2: Upgrade sequence model to CRF with constraints
- Task purpose: Improve sequence consistency beyond HMM while preserving physical constraints.
- Code/files to be changed:
1. `backend/ml/beta6/sequence/crf_decoder.py` (new)
2. `backend/ml/beta6/sequence/transition_builder.py`
3. `backend/ml/beta6/orchestrator.py`
- Testing procedures:
1. A/B evaluation: HMM vs CRF on hard-split datasets.
2. Validate constrained transitions and dwell-time behavior with duration priors as log-linear penalty features.
3. Regression tests on easy-split benchmarks.
- Pass condition:
1. CRF beats HMM on targeted hard splits.
2. No material regression on easy splits.

### Step 5.3: Expand go/no-go metrics and dashboarding
- Task purpose: Gate promotion using timeline, classification, safety, calibration, drift, and unknown metrics.
- Ops UX requirement (additive, not replacing workflow-first view):
1. Add a dedicated **"ML Health Snapshot"** panel for non-ML Ops that shows the latest values **and** thresholds side-by-side.
2. Minimum fields in panel (resident + room scoped):
   - Latest candidate macro-F1 (walk-forward mean)
   - Latest champion macro-F1 (walk-forward mean)
   - Candidate vs champion delta
   - Transition F1 (walk-forward mean, when available)
   - Stability accuracy (walk-forward mean)
   - Effective walk-forward drift threshold (including room override if any)
   - Effective transition F1 threshold (including room override if any)
   - Fold count / lookback window / last updated timestamp
3. Panel must label threshold source (`default`, `env override`, `room override`, `policy`) and use plain-language status badges (`Healthy`, `Watch`, `Action Needed`) with expandable technical details.
4. `Today` workflow must route routine ML low-confidence / uncertainty items to the Review Queue unless they directly block care decisions; the `Needs Attention` list should remain clinical-actionable to prevent alert fatigue.
- Code/files to be changed:
1. `backend/ml/beta6/evaluation_metrics.py` (new)
2. `backend/export_dashboard.py`
3. `backend/run_daily_analysis.py`
4. `backend/health_server.py`
5. `backend/config/beta6_room_capability_gate_profiles.yaml` (new)
6. `docs/planning/template_beta55_to_beta6_go_no_go.md`
- Testing procedures:
1. Validate metric computation with fixture datasets.
2. Validate room-profile threshold evaluation and fail-safe behavior.
3. Validate timeline hard-gate behavior (MAE + fragmentation) across room profiles.
4. Validate dashboard export and signed artifact content.
5. Validate UI panel values match backend health endpoints / run metadata for the same resident-room-time window.
6. Validate missing metric cases render clearly (e.g., no transition coverage) without showing false green.
7. Validate `Today` alert routing keeps routine uncertainty out of the clinical `Needs Attention` list during pilot load tests.
- Pass condition:
1. Gate engine can pass/fail solely from computed metrics.
2. Metrics are complete for every promoted and rejected candidate.
3. Capability-aware thresholds are applied deterministically per room profile.
4. Ops can view current macro-F1 / transition F1 / drift thresholds without reading logs or JSON.
5. Ops `Today` view remains actionable under expected abstain/unknown rates without alert fatigue escalation.

## Phase 6 (Weeks 10-12): Shadow, Laddered Rollout, and Cutover

### Step 6.1: Enable shadow mode and divergence diagnostics
- Task purpose: Compare Beta 6 vs Beta 5.5 safely before authority switch.
- Ops UX requirement:
1. Shadow dashboard must expose divergence counts plus the same **ML Health Snapshot** panel so Ops can correlate "workflow looks green" with model-quality / drift indicators.
2. Every divergence badge should deep-link to plain-language reason text first, technical trace second.
3. Show a persistent global banner across pilot UI views: `Beta 5.5 currently active (Beta 6 in Shadow)` until Beta 6 becomes the authority.
- Code/files to be changed:
1. `backend/run_daily_analysis.py`
2. `backend/ml/beta6/orchestrator.py`
3. `backend/ml/beta6/shadow_compare.py` (new)
4. `backend/export_dashboard.py`
5. `docs/planning/beta6_shadow_mode_runbook.md` (new)
- Testing procedures:
1. Side-by-side run on pilot cohort with fixed data snapshots.
2. Validate divergence classification and reason attribution.
3. Validate signed shadow report generation.
4. Validate shadow dashboard displays resolved threshold values and timestamps correctly for sampled residents.
5. Validate the active-system banner renders consistently on Today / Weekly / Shadow views and matches current rollout state.
- Pass condition:
1. Unexplained divergence <= 5%.
2. Ops can identify the active care baseline (Beta 5.5 vs Beta 6) immediately during pilot review.
3. All divergences are traceable to persisted artifacts.
3. Ops can identify whether a red state is workflow failure, metric drift, or threshold breach within one screen.

### Step 6.2: Implement rollout ladder controls and auto-rollback
- Task purpose: Scale pilot safely from 3 to 100 users with automatic protection.
- Code/files to be changed:
1. `backend/ml/policy_presets.py`
2. `backend/ml/pilot_override_manager.py`
3. `backend/ml/beta6/registry_v2.py`
4. `backend/config/beta6_rollout_ladder.yaml` (new)
- Testing procedures:
1. Simulate ladder progression gates for each rung.
2. Simulate threshold breach and verify automatic rollback.
3. Simulate threshold breach and verify automatic fallback to baseline profile.
4. Validate alerting hooks for rollback/fallback events.
- Pass condition:
1. Ladder progression is blocked on failed gates.
2. Rollback pointer swap succeeds automatically on trigger.
3. Baseline fallback activation is deterministic and fully auditable.

### Step 6.3: Cutover and 14-day stability certification
- Task purpose: Transition pilot cohort to Beta 6 promotion authority with proven stability.
- Code/files to be changed:
1. `backend/run_daily_analysis.py`
2. `backend/ml/beta6/serving_loader.py`
3. `docs/planning/roadmap_ml_stabilization.md`
4. `docs/planning/beta6_pilot_stability_report.md` (new)
- Testing procedures:
1. Run daily certification checks for 14 consecutive days.
2. Verify nightly SLA, gate outcomes, and incident logs.
3. Perform one controlled rollback drill during stability window.
- Pass condition:
1. 14 consecutive stable days achieved.
2. No unresolved P0/P1 incidents.
3. Product, ops, and clinical sign-off completed.

## 9. Rollout Ladder and Auto-Rollback Rules

## Ladder
1. Rung 1: 3 users, 7 days.
2. Rung 2: 20 users, 7 days.
3. Rung 3: 50 users, 7 days.
4. Rung 4: 100 users, 14 days.

## Progression criteria at each rung
1. All mandatory metric floors passed.
2. No open P0 incidents.
3. Nightly pipeline success >= 99%.
4. Drift alerts within budget.
5. Timeline hard gates (MAE + fragmentation) passed for all active rooms.
6. Rung 2+ requires completed Phase 5 Step 5.1 + 5.2 acceptance tests (adapter lifecycle + CRF/HMM A/B gates).

## Automatic rollback triggers
1. > 10% MAE regression in any room for 2 consecutive nights.
2. Alert precision below threshold for 2 consecutive nights.
3. Worst-room F1 below floor for 2 consecutive nights.
4. Pipeline reliability below 97% daily success.

## Operator-safe fallback mode
1. If sequence model fails gate, serving auto-falls back to approved baseline profile (`rule` or `HMM`) per room.
2. Fallback must emit deterministic reason code and event artifact; no silent fallback.
3. Fallback activation/deactivation requires:
   - Alert to operations channel.
   - Pointer/flag audit event with timestamp and actor/process id.
4. Recovery criterion:
   - Room must pass hard gates for configured consecutive nights before exiting fallback.

## 10. Operating Model and Ownership

| Workstream | Accountable | Responsible | Consulted |
|---|---|---|---|
| Contracts/Registry | ML Platform Lead | Backend ML Engineer | QA Lead |
| Backbone/Tuning | Modeling Lead | ML Engineers | Clinical Scientist |
| Sequence Layer | Modeling Lead | ML Engineers | Platform Lead |
| Active Learning Ops | Clinical Ops Lead | Labeling Operators | Modeling Lead |
| Gating/Promotion | MLOps Lead | Platform Engineer | ML Lead |
| Shadow/Cutover | Release Manager | MLOps + Backend | Product + Clinical |

## 11. Required Artifacts per Phase

1. `run_spec.yaml` and schema hash.
2. Label-pack intake bundle (`validate + diff + smoke` reports).
3. Data corpus manifest and fingerprint.
4. Backbone training report and embedding quality report.
5. Candidate evaluation report with room-level reason codes.
6. Promotion decision artifact with rollback pointer.
7. Shadow divergence report.
8. Weekly pilot status report (metrics, incidents, decisions).

## 11.1 Model behavior SLOs (daily)
1. Unknown/abstain rate by room and by uncertainty taxonomy state.
2. Room occupancy drift vs trailing baseline.
3. Gate reason-code distribution (room-level and run-level).
4. SLO breach handling:
   - Daily alert for threshold breach.
   - Escalation owner and remediation ETA logged.

## 12. Day-0 to Day-10 Kickoff Checklist

1. Finalize this plan and assign named owners for each workstream.
2. Lock metric thresholds and rollback policy in a single signed policy doc.
3. Complete Phase 0 label-pack intake gate (`validate + diff + smoke`) and publish approved intake bundle.
4. Freeze YAML registry schema and publish examples for 2-room and 5-room flats.
5. Lock uncertainty taxonomy semantics (`low_confidence`, `unknown`, `outside_sensed_space`) and reason-code mapping.
6. Publish sensor onboarding protocol and complete installer dry-run for at least one 2-room and one 5-room flat.
7. Implement leakage test harness and fail CI on violations.
8. Lock runtime/eval parity tests and fail CI on mismatch.
9. Build unknown/abstain path in scoring outputs before pilot runs.
10. Stand up active-learning triage queue with reviewer staffing schedule.
11. Run tabletop rollback + fallback drill and confirm pointer swap + baseline auto-fallback behavior.
12. Validate Beta 5.5 champion models are stable, loadable, and available as rollback targets.
13. Prepare shadow dashboard with Beta 5.5 vs Beta 6 comparisons.
   - Include non-ML Ops "ML Health Snapshot" panel (macro-F1, transition F1, drift threshold, threshold source, status badge).
   - Include persistent active-system banner (`Beta 5.5 currently active (Beta 6 in Shadow)`) until cutover.
14. Pre-approve rollout ladder gates with product, clinical, and ops.
   - Explicitly sign off default thresholds and override ownership for `WF_DRIFT_THRESHOLD`, `WF_MIN_TRANSITION_F1`, and `WF_MIN_STABILITY_ACCURACY`.
   - Sign off `Today` alert-routing rule (clinical-actionable alerts only; routine uncertainty to Review Queue).
15. Start Week 1 with a daily 15-minute cross-functional status checkpoint.

## 13. Risks and Mitigations

| Risk | Impact | Mitigation | Owner |
|---|---|---|---|
| Labeling backlog grows | Model stagnates, delayed gates | Throughput SLA, triage prioritization, temporary reviewer surge | Clinical Ops Lead |
| Leakage bug in evaluation | False confidence, unsafe promotion | CI leakage tests, resident/time disjoint enforcement, audit trails | QA Lead |
| Sensor placement/calibration faults | Quality ceiling, false regressions, delayed rollout | Sensor onboarding protocol, nightly health gates, field-ops recalibration loop | Field Ops Lead |
| Runtime/eval semantic drift | Passes in backtest, fails in production behavior | Locked parity contract, fixed-trace replay tests, CI parity gate | ML Platform Lead |
| CRF instability on sparse labels | Regressions at rollout | Keep HMM fallback flag, rung-gated rollout | Modeling Lead |
| Drift from sensor firmware changes | Silent degradation | Drift detector with release hooks and block-on-critical shift policy | MLOps Lead |
| Adapter sprawl | Storage/ops overhead | Adapter retention policy and max active versions | Platform Lead |
| Compliance gap on unlabeled corpus | Launch block | Early privacy gate and written sign-off in Phase 1 | Product/Compliance |

## 14. Definition of Pilot Success (100 Users)

Pilot is considered successful only when all are true:
1. 14-day stability window passed at 100 users.
2. No unresolved P0/P1 safety incidents.
3. Metric floors and divergence budgets met.
4. Rollback path verified by at least one controlled drill during pilot.
5. Team agrees architecture can proceed to next scaling rung without re-architecture.
