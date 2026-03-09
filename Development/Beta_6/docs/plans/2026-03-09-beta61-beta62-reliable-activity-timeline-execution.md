# Beta 6.1 / Beta 6.2 Reliable Activity Timeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute Beta 6.1 as the productionization track for reliable authority-grade activity timelines, while executing Beta 6.2 as the isolated correction-reduction and learning-efficiency track.

**Architecture:** Use `codex/pilot-bootstrap-gates` at `94524af` as the new source-of-truth baseline. Treat `origin/codex/livingroom-fast-diagnosis` at `6c4128b` as reviewed evidence that some rooms remain policy-sensitive, not as a replacement baseline. Beta 6.1 finishes production contracts, observability, certification, and replayable room-policy diagnostics around the current good model stack. Beta 6.2 builds a shared `20x14` corpus program, correction-aware learning loop, timeline-native supervision, context-conditioned modeling, and faster experiment infrastructure, without changing Beta 6.1 outputs when Beta 6.2 flags are off.

**Tech Stack:** Python 3.12, pytest, PostgreSQL, JSON artifact signing, YAML policy/config, Beta 6 registry v2, timeline evaluation modules, active-learning triage, correction/review services.

---

## Baseline Assumptions

1. `codex/pilot-bootstrap-gates` at `94524af` is the planning baseline.
2. LivingRoom reliability and review-loop work have already landed.
3. Beta 6.1 is no longer primarily blocked by Bedroom/Entrance model failures.
4. Beta 6.2 must maximize model-first improvement before escalating to human correction.
5. `origin/codex/livingroom-fast-diagnosis` adds a reviewed LivingRoom downsample retune candidate, which should be handled through replay diagnostics before any default adoption.

## Beta 6.1 Tasks

### Task 1: Rebaseline the Beta 6.1 source of truth

**Files:**
- Modify: `docs/plans/2026-03-09-beta61-beta62-reliable-activity-timeline-design.md`
- Modify: `docs/plans/2026-03-09-beta61-beta62-reliable-activity-timeline-execution.md`
- Read/verify: `backend/models_beta6_registry_v2/HK0011_jessica/_run/events.jsonl`
- Read/verify: `backend/models/HK0011_jessica/*_decision_trace.json`

**Step 1: Write the failing documentation assertion**

Record the March 8 assumptions that are no longer true:
1. older baseline SHA
2. unresolved LivingRoom status
3. missing production-profile startup changes

**Step 2: Verify the current baseline evidence**

Run:

```bash
git rev-parse --short HEAD
git log --oneline --decorate -n 8
tail -n 1 backend/models_beta6_registry_v2/HK0011_jessica/_run/events.jsonl
```

Expected:
1. baseline is current `codex/pilot-bootstrap-gates`
2. latest commits include authority hardening, prod-profile prep, and LivingRoom reliability work

**Step 3: Write the minimal documentation update**

Document:
1. the new baseline
2. what already landed
3. what remains open

**Step 4: Re-read the updated docs**

Run:

```bash
sed -n '1,220p' docs/plans/2026-03-09-beta61-beta62-reliable-activity-timeline-design.md
sed -n '1,220p' docs/plans/2026-03-09-beta61-beta62-reliable-activity-timeline-execution.md
```

Expected: docs reflect `94524af` as source of truth.

**Step 5: Commit**

```bash
git add docs/plans/2026-03-09-beta61-beta62-reliable-activity-timeline-design.md docs/plans/2026-03-09-beta61-beta62-reliable-activity-timeline-execution.md
git commit -m "docs: rebaseline beta61 beta62 roadmap to current pilot bootstrap"
```

### Task 2: Make Beta 6.1 authority preflight explicit and production-safe

**Files:**
- Modify: `backend/run_daily_analysis.py`
- Modify: `backend/.env.example`
- Modify: `start.sh`
- Modify: `ops_start.sh`
- Test: `backend/tests/test_run_daily_analysis_beta6_authority.py`
- Test: `backend/tests/test_health_server.py`

**Step 1: Write the failing tests**

Add tests for:

```python
def test_live_authority_requires_explicit_signing_key(): ...
def test_live_authority_requires_explicit_evidence_profile(): ...
def test_live_authority_preflight_reports_postgres_unavailable(): ...
```

**Step 2: Run the focused tests to verify failure**

Run:

```bash
pytest backend/tests/test_run_daily_analysis_beta6_authority.py backend/tests/test_health_server.py -q
```

Expected: failure because current startup/runtime paths still allow implicit defaults in some cases.

**Step 3: Write the minimal implementation**

Implement a single preflight path that:
1. requires explicit `BETA6_GATE_SIGNING_KEY` for live authority
2. requires explicit `RELEASE_GATE_EVIDENCE_PROFILE`
3. checks PostgreSQL reachability before claiming production readiness
4. reports deterministic failure reasons

**Step 4: Run focused tests**

Run:

```bash
pytest backend/tests/test_run_daily_analysis_beta6_authority.py backend/tests/test_health_server.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/run_daily_analysis.py backend/.env.example start.sh ops_start.sh backend/tests/test_run_daily_analysis_beta6_authority.py backend/tests/test_health_server.py
git commit -m "feat: require explicit beta61 authority preflight contracts"
```

### Task 3: Make fallback and rollback deterministic for every room

**Files:**
- Modify: `backend/ml/beta6/registry/registry_v2.py`
- Modify: `backend/ml/t80_rollout_manager.py`
- Modify: `backend/ml/beta6/serving/serving_loader.py`
- Test: `backend/tests/test_beta6_registry_v2.py`
- Test: `backend/tests/test_t80_rollout.py`
- Test: `backend/tests/test_beta6_serving_loader.py`

**Step 1: Write the failing tests**

Add tests for:

```python
def test_registry_resolves_fallback_from_previous_known_good_pointer(): ...
def test_rollout_auto_rollback_does_not_crash_on_missing_materialized_pointer(): ...
def test_serving_loader_uses_deterministic_fallback_resolution(): ...
```

**Step 2: Run the focused tests to verify failure**

Run:

```bash
pytest backend/tests/test_beta6_registry_v2.py backend/tests/test_t80_rollout.py backend/tests/test_beta6_serving_loader.py -q
```

Expected: failure or missing coverage for `livingroom`-style fallback gaps.

**Step 3: Write the minimal implementation**

Implement:
1. deterministic fallback resolution order
2. explicit “no safe fallback exists” terminal reason
3. non-crashing rollback event behavior

**Step 4: Run focused tests**

Run:

```bash
pytest backend/tests/test_beta6_registry_v2.py backend/tests/test_t80_rollout.py backend/tests/test_beta6_serving_loader.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/ml/beta6/registry/registry_v2.py backend/ml/t80_rollout_manager.py backend/ml/beta6/serving/serving_loader.py backend/tests/test_beta6_registry_v2.py backend/tests/test_t80_rollout.py backend/tests/test_beta6_serving_loader.py
git commit -m "feat: make beta61 rollback fallback deterministic"
```

### Task 4: Publish product-facing timeline reliability and correction-load scorecards

**Files:**
- Modify: `backend/export_dashboard.py`
- Modify: `backend/services/ops_service.py`
- Modify: `backend/services/correction_service.py`
- Modify: `backend/health_server.py`
- Modify: `backend/app/pages/4_ops_dashboard.py`
- Modify: `backend/app/pages/1_correction_studio.py`
- Test: `backend/tests/test_ui_services.py`
- Test: `backend/tests/test_health_server.py`

**Step 1: Write the failing tests**

Add tests for scorecard fields such as:

```python
def test_ops_scorecard_includes_correction_volume_and_backlog(): ...
def test_health_snapshot_includes_timeline_reliability_metrics(): ...
def test_review_ui_exposes_manual_review_rate(): ...
```

**Step 2: Run the focused tests to verify failure**

Run:

```bash
pytest backend/tests/test_ui_services.py backend/tests/test_health_server.py -q
```

Expected: failure because product-level correction metrics are incomplete or absent.

**Step 3: Write the minimal implementation**

Expose:
1. correction count
2. review backlog
3. contradiction rate
4. fragmentation rate
5. unknown/abstain trend
6. active-system and authority-state clarity
7. room-policy sensitivity indicators for rooms under active tuning

**Step 4: Run focused tests**

Run:

```bash
pytest backend/tests/test_ui_services.py backend/tests/test_health_server.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/export_dashboard.py backend/services/ops_service.py backend/services/correction_service.py backend/health_server.py backend/app/pages/4_ops_dashboard.py backend/app/pages/1_correction_studio.py backend/tests/test_ui_services.py backend/tests/test_health_server.py
git commit -m "feat: publish beta61 timeline reliability and correction scorecards"
```

### Task 4A: Add replayable room-policy diagnostics before default retunes

**Files:**
- Modify: `backend/ml/room_experiments.py`
- Modify: `backend/scripts/run_room_experiments.py`
- Modify: `backend/config/beta6_policy_defaults.yaml`
- Modify: `backend/ml/policy_config.py`
- Modify: `backend/ml/policy_defaults.py`
- Test: `backend/tests/test_run_room_experiments.py`
- Test: `backend/tests/test_policy_config.py`

**Step 1: Write the failing tests**

Add tests for:

```python
def test_room_experiments_can_replay_policy_sweep_candidates(): ...
def test_policy_defaults_can_emit_named_room_diagnostic_profiles(): ...
def test_livingroom_replay_candidate_is_traceable_to_typed_policy_fields(): ...
```

**Step 2: Run the focused tests to verify failure**

Run:

```bash
pytest backend/tests/test_run_room_experiments.py backend/tests/test_policy_config.py -q
```

Expected: failure or missing coverage for named replayable policy-sweep candidates.

**Step 3: Write the minimal implementation**

Implement:
1. named room-diagnostic policy profiles for replay sweeps
2. report output that records the exact typed-policy values under test
3. lightweight replay support for branches like `livingroom-fast-diagnosis`

**Step 4: Run focused tests**

Run:

```bash
pytest backend/tests/test_run_room_experiments.py backend/tests/test_policy_config.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/ml/room_experiments.py backend/scripts/run_room_experiments.py backend/config/beta6_policy_defaults.yaml backend/ml/policy_config.py backend/ml/policy_defaults.py backend/tests/test_run_room_experiments.py backend/tests/test_policy_config.py
git commit -m "feat: add beta61 replayable room policy diagnostics"
```

### Task 5: Add the Beta 6.1 resident/home context contract

**Files:**
- Modify: `backend/db/schema.sql`
- Modify: `backend/processors/profile_processor.py`
- Modify: `backend/ml/household_analyzer.py`
- Modify: `backend/ml/home_empty_fusion.py`
- Modify: `backend/ml/beta6/sequence/transition_builder.py`
- Modify: `backend/export_dashboard.py`
- Test: `backend/tests/test_home_empty_fusion.py`
- Test: `backend/tests/test_health_server.py`

**Step 1: Write the failing tests**

Add tests for:

```python
def test_single_vs_multi_resident_context_is_typed_and_explicit(): ...
def test_layout_context_reaches_transition_builder_without_env_reads(): ...
def test_missing_required_context_is_reported_clearly(): ...
```

**Step 2: Run the focused tests to verify failure**

Run:

```bash
pytest backend/tests/test_home_empty_fusion.py backend/tests/test_health_server.py -q
```

Expected: failure because context is incomplete or not explicitly typed end-to-end.

**Step 3: Write the minimal implementation**

Implement typed context fields for:
1. household type
2. helper presence
3. layout/adjacency/topology

Use them for routing, cohorting, arbitration, and reliability reporting only.

**Step 4: Run focused tests**

Run:

```bash
pytest backend/tests/test_home_empty_fusion.py backend/tests/test_health_server.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/db/schema.sql backend/processors/profile_processor.py backend/ml/household_analyzer.py backend/ml/home_empty_fusion.py backend/ml/beta6/sequence/transition_builder.py backend/export_dashboard.py backend/tests/test_home_empty_fusion.py backend/tests/test_health_server.py
git commit -m "feat: add beta61 resident home context contract"
```

### Task 6: Run the real-environment Beta 6.1 certification entry pass

**Files:**
- Modify: `backend/run_daily_analysis.py`
- Modify: `docs/plans/2026-03-09-beta61-beta62-reliable-activity-timeline-execution.md`
- Test: `backend/tests/test_run_daily_analysis_beta6_authority.py`

**Step 1: Write the failing certification checklist**

Define a checklist that fails if:
1. signing is not explicit
2. evidence profile is not explicit
3. PostgreSQL is unavailable
4. rollback fallback state is incomplete

**Step 2: Run the targeted tests**

Run:

```bash
pytest backend/tests/test_run_daily_analysis_beta6_authority.py -q
```

Expected: PASS

**Step 3: Run the real-environment rerun**

Run:

```bash
python3 backend/run_daily_analysis.py
```

Expected:
1. all room gates pass
2. authority artifacts are signed
3. no fallback-target crash
4. `phase6_stability_report.json` shows non-zero pipeline success in the real environment

**Step 4: Verify the run artifacts**

Run:

```bash
tail -n 1 backend/models_beta6_registry_v2/HK0011_jessica/_run/events.jsonl
```

Expected: authority/runtime blockers are cleared or reduced to clearly actionable residuals.

**Step 5: Commit**

```bash
git add backend/run_daily_analysis.py docs/plans/2026-03-09-beta61-beta62-reliable-activity-timeline-execution.md
git commit -m "docs: record beta61 certification entry checks"
```

## Beta 6.2 Tasks

### Task 7: Establish a Beta 6.2 SSOT namespace before adding more model work

**Files:**
- Modify: `backend/ml/beta6/beta6_trainer.py`
- Modify: `backend/ml/beta6/training/beta6_trainer.py`
- Modify: `backend/ml/beta6/registry_v2.py`
- Modify: `backend/ml/beta6/registry/registry_v2.py`
- Modify: `backend/ml/beta6/timeline_hard_gates.py`
- Modify: `backend/ml/beta6/gates/timeline_hard_gates.py`
- Test: `backend/tests/test_beta6_import_boundaries.py`

**Step 1: Write the failing import-boundary tests**

Add tests asserting:

```python
def test_beta62_training_namespace_has_single_authoritative_entrypoint(): ...
def test_duplicate_registry_and_gate_paths_are_explicit_shims_or_blocked(): ...
```

**Step 2: Run the focused tests to verify failure**

Run:

```bash
pytest backend/tests/test_beta6_import_boundaries.py -q
```

Expected: failure or missing coverage because duplicate module surfaces still exist.

**Step 3: Write the minimal implementation**

Choose authoritative Beta 6.2 paths and make other paths:
1. explicit shims
2. deprecated wrappers
3. or blocked imports

**Step 4: Run focused tests**

Run:

```bash
pytest backend/tests/test_beta6_import_boundaries.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/ml/beta6/beta6_trainer.py backend/ml/beta6/training/beta6_trainer.py backend/ml/beta6/registry_v2.py backend/ml/beta6/registry/registry_v2.py backend/ml/beta6/timeline_hard_gates.py backend/ml/beta6/gates/timeline_hard_gates.py backend/tests/test_beta6_import_boundaries.py
git commit -m "refactor: define beta62 ssot module surfaces"
```

### Task 8: Build the shared `20x14` corpus contract

**Files:**
- Modify: `backend/scripts/build_pretrain_corpus_manifest.py`
- Modify: `backend/ml/beta6/data_manifest.py`
- Modify: `backend/ml/beta6/feature_store.py`
- Modify: `docs/planning/beta6_pretrain_corpus_manifest.md`
- Test: `backend/tests/test_build_pretrain_corpus_manifest_script.py`
- Test: `backend/tests/test_beta6_data_manifest.py`
- Test: `backend/tests/test_beta6_trainer_intake_gate.py`

**Step 1: Write the failing tests**

Add tests for:

```python
def test_manifest_contains_shadow_pretrain_and_labeled_views(): ...
def test_manifest_tracks_context_completeness_and_label_quality(): ...
def test_intake_gate_rejects_incomplete_20x14_resident_windows(): ...
```

**Step 2: Run the focused tests to verify failure**

Run:

```bash
pytest backend/tests/test_build_pretrain_corpus_manifest_script.py backend/tests/test_beta6_data_manifest.py backend/tests/test_beta6_trainer_intake_gate.py -q
```

Expected: failure because the current corpus contract is not rich enough.

**Step 3: Write the minimal implementation**

Add:
1. shadow cohort view
2. unlabeled pretrain view
3. labeled high-trust fine-tune/eval view
4. resident/home context completeness
5. label quality metadata

**Step 4: Run focused tests**

Run:

```bash
pytest backend/tests/test_build_pretrain_corpus_manifest_script.py backend/tests/test_beta6_data_manifest.py backend/tests/test_beta6_trainer_intake_gate.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/scripts/build_pretrain_corpus_manifest.py backend/ml/beta6/data_manifest.py backend/ml/beta6/feature_store.py docs/planning/beta6_pretrain_corpus_manifest.md backend/tests/test_build_pretrain_corpus_manifest_script.py backend/tests/test_beta6_data_manifest.py backend/tests/test_beta6_trainer_intake_gate.py
git commit -m "feat: define shared beta6 20x14 corpus contract"
```

### Task 9: Turn corrections into structured training signal

**Files:**
- Modify: `backend/services/correction_service.py`
- Modify: `backend/ml/beta6/active_learning.py`
- Modify: `backend/ml/beta6/training/active_learning.py`
- Modify: `backend/scripts/run_active_learning_triage.py`
- Modify: `backend/config/beta6_active_learning_policy.yaml`
- Test: `backend/tests/test_beta6_active_learning.py`
- Test: `backend/tests/test_run_active_learning_triage_script.py`
- Test: `backend/tests/test_ui_services.py`

**Step 1: Write the failing tests**

Add tests for:

```python
def test_accepted_corrections_emit_boundary_and_hard_negative_payloads(): ...
def test_active_learning_triage_prioritizes_high_yield_segments(): ...
def test_correction_queue_outputs_training_ready_records(): ...
```

**Step 2: Run the focused tests to verify failure**

Run:

```bash
pytest backend/tests/test_beta6_active_learning.py backend/tests/test_run_active_learning_triage_script.py backend/tests/test_ui_services.py -q
```

Expected: failure because corrections are not yet fully normalized into learning-ready signals.

**Step 3: Write the minimal implementation**

Emit correction-derived:
1. corrected events
2. onset/offset targets
3. hard negatives
4. residual review packs
5. triage priority scores

**Step 4: Run focused tests**

Run:

```bash
pytest backend/tests/test_beta6_active_learning.py backend/tests/test_run_active_learning_triage_script.py backend/tests/test_ui_services.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/services/correction_service.py backend/ml/beta6/active_learning.py backend/ml/beta6/training/active_learning.py backend/scripts/run_active_learning_triage.py backend/config/beta6_active_learning_policy.yaml backend/tests/test_beta6_active_learning.py backend/tests/test_run_active_learning_triage_script.py backend/tests/test_ui_services.py
git commit -m "feat: convert corrections into beta62 learning signals"
```

### Task 10: Add timeline-native targets and heads

**Files:**
- Modify: `backend/ml/transformer_timeline_heads.py`
- Modify: `backend/ml/timeline_targets.py`
- Modify: `backend/ml/timeline_metrics.py`
- Modify: `backend/ml/beta6/training/beta6_trainer.py`
- Modify: `backend/ml/beta6/gates/timeline_hard_gates.py`
- Test: `backend/tests/test_transformer_timeline_heads.py`
- Test: `backend/tests/test_timeline_targets.py`
- Test: `backend/tests/test_timeline_metrics.py`
- Test: `backend/tests/test_timeline_gates.py`

**Step 1: Write the failing tests**

Add tests for:

```python
def test_timeline_heads_emit_onset_offset_duration_outputs(): ...
def test_timeline_targets_generate_event_native_supervision(): ...
def test_timeline_metrics_include_event_level_quality_fields(): ...
```

**Step 2: Run the focused tests to verify failure**

Run:

```bash
pytest backend/tests/test_transformer_timeline_heads.py backend/tests/test_timeline_targets.py backend/tests/test_timeline_metrics.py backend/tests/test_timeline_gates.py -q
```

Expected: failure or missing coverage for event-native outputs.

**Step 3: Write the minimal implementation**

Add offline-only:
1. onset head
2. offset head
3. duration/continuity targets
4. event-native evaluation summaries

**Step 4: Run focused tests**

Run:

```bash
pytest backend/tests/test_transformer_timeline_heads.py backend/tests/test_timeline_targets.py backend/tests/test_timeline_metrics.py backend/tests/test_timeline_gates.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/ml/transformer_timeline_heads.py backend/ml/timeline_targets.py backend/ml/timeline_metrics.py backend/ml/beta6/training/beta6_trainer.py backend/ml/beta6/gates/timeline_hard_gates.py backend/tests/test_transformer_timeline_heads.py backend/tests/test_timeline_targets.py backend/tests/test_timeline_metrics.py backend/tests/test_timeline_gates.py
git commit -m "feat: add beta62 timeline native supervision"
```

### Task 11: Add context-conditioned modeling before demographic modeling

**Files:**
- Modify: `backend/ml/beta6/training/beta6_trainer.py`
- Modify: `backend/ml/beta6/sequence/transition_builder.py`
- Modify: `backend/ml/home_empty_fusion.py`
- Modify: `backend/ml/household_analyzer.py`
- Test: `backend/tests/test_home_empty_fusion.py`
- Test: `backend/tests/test_transformer_timeline_heads.py`

**Step 1: Write the failing tests**

Add tests for:

```python
def test_layout_topology_context_reaches_beta62_model_path(): ...
def test_household_helper_context_changes_decoder_constraints_offline_only(): ...
def test_demographic_context_is_not_default_training_input(): ...
```

**Step 2: Run the focused tests to verify failure**

Run:

```bash
pytest backend/tests/test_home_empty_fusion.py backend/tests/test_transformer_timeline_heads.py -q
```

Expected: failure or missing coverage for context-conditioned behavior.

**Step 3: Write the minimal implementation**

Implement offline-only conditioning for:
1. layout/adjacency/topology
2. household/helper context

Do not make age/sex default model inputs.

**Step 4: Run focused tests**

Run:

```bash
pytest backend/tests/test_home_empty_fusion.py backend/tests/test_transformer_timeline_heads.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/ml/beta6/training/beta6_trainer.py backend/ml/beta6/sequence/transition_builder.py backend/ml/home_empty_fusion.py backend/ml/household_analyzer.py backend/tests/test_home_empty_fusion.py backend/tests/test_transformer_timeline_heads.py
git commit -m "feat: add beta62 context conditioned modeling"
```

### Task 12: Improve learning efficiency and experiment throughput

**Files:**
- Modify: `backend/ml/beta6/feature_store.py`
- Modify: `backend/ml/beta6/training/self_supervised_pretrain.py`
- Modify: `backend/ml/beta6/training/beta6_trainer.py`
- Modify: `backend/scripts/build_pretrain_corpus_manifest.py`
- Modify: `backend/ml/room_experiments.py`
- Modify: `backend/scripts/run_room_experiments.py`
- Test: `backend/tests/test_beta6_self_supervised_pretrain.py`
- Test: `backend/tests/test_beta6_data_manifest.py`
- Test: `backend/tests/test_beta6_trainer_intake_gate.py`
- Test: `backend/tests/test_run_room_experiments.py`

**Step 1: Write the failing tests**

Add tests for:

```python
def test_feature_sequence_cache_respects_manifest_and_policy_fingerprint(): ...
def test_bad_candidates_can_be_eliminated_early_without_affecting_good_runs(): ...
def test_tensor_reuse_is_disabled_when_inputs_change(): ...
def test_room_policy_sweeps_can_run_as_fast_replay_diagnostics(): ...
```

**Step 2: Run the focused tests to verify failure**

Run:

```bash
pytest backend/tests/test_beta6_self_supervised_pretrain.py backend/tests/test_beta6_data_manifest.py backend/tests/test_beta6_trainer_intake_gate.py backend/tests/test_run_room_experiments.py -q
```

Expected: failure because caching and candidate triage are not yet fully implemented.

**Step 3: Write the minimal implementation**

Add:
1. canonical cache keys
2. cached feature/sequence reuse
3. early bad-candidate elimination
4. safe invalidation on manifest/policy change
5. fast replay mode for room-policy sweeps before full retrain

**Step 4: Run focused tests**

Run:

```bash
pytest backend/tests/test_beta6_self_supervised_pretrain.py backend/tests/test_beta6_data_manifest.py backend/tests/test_beta6_trainer_intake_gate.py backend/tests/test_run_room_experiments.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/ml/beta6/feature_store.py backend/ml/beta6/training/self_supervised_pretrain.py backend/ml/beta6/training/beta6_trainer.py backend/scripts/build_pretrain_corpus_manifest.py backend/ml/room_experiments.py backend/scripts/run_room_experiments.py backend/tests/test_beta6_self_supervised_pretrain.py backend/tests/test_beta6_data_manifest.py backend/tests/test_beta6_trainer_intake_gate.py backend/tests/test_run_room_experiments.py
git commit -m "feat: improve beta62 learning efficiency infrastructure"
```

## Pass Requirements for the Whole Program

### Beta 6.1

1. authority preflight is explicit and production-safe
2. rollback/fallback is deterministic and tested
3. product timeline reliability and correction-load scorecards are visible
4. resident/home context contract exists and is typed
5. real-environment rerun succeeds with signed artifacts and real runtime dependencies

### Beta 6.2

1. `20x14` is a shared Beta 6 asset, not a Beta 6.2 silo
2. corrections flow into structured training assets
3. timeline-native targets and context-conditioned modeling are offline and isolated
4. manual correction demand is measurable and expected to decline over time
5. experiment throughput improves through caching and candidate triage
6. room-policy regressions can be diagnosed by replay sweeps before expensive full retrains

## Recommended Execution Order

1. Task 1 to Task 6: Beta 6.1 productionization and certification
2. Task 7 to Task 9: Beta 6.2 namespace, corpus, and correction-learning loop
3. Task 10 to Task 12: Beta 6.2 timeline-native/context/efficiency upgrades
