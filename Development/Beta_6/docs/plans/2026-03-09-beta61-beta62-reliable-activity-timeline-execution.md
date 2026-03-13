# Beta 6.1 / Beta 6.2 Reliable Activity Timeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute Beta 6.1 as the productionization track for reliable authority-grade activity timelines, while executing Beta 6.2 as the isolated correction-reduction and learning-efficiency track.

**Architecture:** Use `codex/pilot-bootstrap-gates` at `94524af` as the new source-of-truth baseline. Treat `origin/codex/livingroom-fast-diagnosis` at `6c4128b` as reviewed evidence that some rooms remain policy-sensitive, not as a replacement baseline. Beta 6.1 finishes production contracts, observability, certification, fragile-room operational contracts, and replayable room-policy diagnostics around the current good model stack. Beta 6.2 builds an offline governed training-file intake subsystem, grouped-regime robustness gates, a shared `20x14` corpus program, correction-aware learning loop, timeline-native supervision, context-conditioned modeling, and faster experiment infrastructure, without changing Beta 6.1 outputs when Beta 6.2 flags are off.

**Tech Stack:** Python 3.12, pytest, PostgreSQL, JSON artifact signing, YAML policy/config, Beta 6 registry v2, timeline evaluation modules, active-learning triage, correction/review services.

---

## Baseline Assumptions

1. `codex/pilot-bootstrap-gates` at `94524af` is the planning baseline.
2. LivingRoom reliability and review-loop work have already landed.
3. Beta 6.1 is no longer primarily blocked by Bedroom/Entrance model failures.
4. Beta 6.2 must maximize model-first improvement before escalating to human correction.
5. `origin/codex/livingroom-fast-diagnosis` adds a reviewed LivingRoom downsample retune candidate, which should be handled through replay diagnostics before any default adoption.
6. Jessica pre-final on `HK0011_jessica_candidate_supportfix_20260310T2312Z` was `GO` with room status:
   - `Bathroom = pass`
   - `Bedroom = conditional`
   - `Kitchen = pass`
   - `LivingRoom = pass`
7. Bedroom is intentionally single-stage fallback when the saved `Bedroom_v38_two_stage_meta.json` says `runtime_enabled=false`; it should not be assumed to appear in `platform.two_stage_core_models`.

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
4. implicit assumption that Bedroom is simply solved or always two-stage at runtime

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
3. fresh evidence confirms Jessica pre-final `GO` with `Bedroom=conditional`

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

### Task 5: Add fragile-room operational contracts and replay diagnostics

**Files:**
- Modify: `backend/ml/training.py`
- Modify: `backend/ml/room_experiments.py`
- Modify: `backend/scripts/run_room_experiments.py`
- Modify: `backend/config/beta6_policy_defaults.yaml`
- Modify: `backend/ml/policy_config.py`
- Modify: `backend/ml/policy_defaults.py`
- Test: `backend/tests/test_training.py`
- Test: `backend/tests/test_run_room_experiments.py`
- Test: `backend/tests/test_policy_config.py`

**Step 1: Write the failing tests**

Add tests for:

```python
def test_fragile_room_status_allows_pass_conditional_block(): ...
def test_bedroom_runtime_expectation_uses_saved_runtime_enabled_flag(): ...
def test_room_experiments_can_replay_policy_sweep_candidates(): ...
def test_policy_defaults_can_emit_named_room_diagnostic_profiles(): ...
def test_livingroom_replay_candidate_is_traceable_to_typed_policy_fields(): ...
def test_bedroom_grouped_date_fragility_is_persisted_in_review_surface(): ...
```

**Step 2: Run the focused tests to verify failure**

Run:

```bash
pytest backend/tests/test_training.py backend/tests/test_run_room_experiments.py backend/tests/test_policy_config.py -q
```

Expected: failure or missing coverage for fragile-room status, saved-topology expectations, and named replayable policy-sweep candidates.

**Step 3: Write the minimal implementation**

Implement:
1. explicit room status semantics: `pass`, `conditional`, `block`
2. runtime expectation checks that derive from saved runtime topology
3. named room-diagnostic policy profiles for replay sweeps
4. report output that records the exact typed-policy values under test
5. grouped-date / lineage fragility visibility for Bedroom
6. lightweight replay support for branches like `livingroom-fast-diagnosis`

**Step 4: Run focused tests**

Run:

```bash
pytest backend/tests/test_training.py backend/tests/test_run_room_experiments.py backend/tests/test_policy_config.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/ml/training.py backend/ml/room_experiments.py backend/scripts/run_room_experiments.py backend/config/beta6_policy_defaults.yaml backend/ml/policy_config.py backend/ml/policy_defaults.py backend/tests/test_training.py backend/tests/test_run_room_experiments.py backend/tests/test_policy_config.py
git commit -m "feat: add beta61 fragile-room contracts and replay diagnostics"
```

### Task 6: Add the Beta 6.1 resident/home context contract

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

### Task 7: Run the real-environment Beta 6.1 certification entry pass

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
5. any room is `block`
6. runtime topology expectations do not match saved artifacts
7. room status is missing/unknown (GO requires explicit `pass|conditional|block`)

Implementation target:
- add `_evaluate_beta61_certification_entry(...)` in `backend/run_daily_analysis.py`
- persist checklist output in training metadata as `beta61_certification_entry`
- fail closed with `rejected_by_beta61_certification` when checklist fails
- always emit tracked certification signals:
  - PostgreSQL preflight details
  - historical-corrections availability/details

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
5. `GO` is allowed only with `pass` and `conditional`, never with `block` (or unknown status)

**Step 4: Verify the run artifacts**

Run:

```bash
tail -n 1 backend/models_beta6_registry_v2/HK0011_jessica/_run/events.jsonl
```

Expected: authority/runtime blockers are cleared or reduced to clearly actionable residuals.
Expected:
1. room verdicts are explicit as `pass`, `conditional`, or `block`
2. PostgreSQL / historical-corrections availability is recorded as a tracked certification signal
3. Bedroom runtime expectations match the saved single-stage fallback topology if `runtime_enabled=false`
4. `training_history.metadata.beta61_certification_entry` includes explicit checklist checks and failure reason (if any)

**Step 5: Commit**

```bash
git add backend/run_daily_analysis.py docs/plans/2026-03-09-beta61-beta62-reliable-activity-timeline-execution.md
git commit -m "docs: record beta61 certification entry checks"
```

## Beta 6.2 Tasks

### Task 8: Establish a Beta 6.2 SSOT namespace before adding more model work

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

### Task 9: Build the offline governed training-file intake and quarantine subsystem

**Files:**
- Modify: `backend/scripts/build_pretrain_corpus_manifest.py`
- Modify: `backend/ml/beta6/data_manifest.py`
- Modify: `backend/ml/beta6/feature_store.py`
- Modify: `backend/ml/beta6/gates/intake_gate.py`
- Modify: `backend/ml/beta6/gates/intake_precheck.py`
- Modify: `docs/planning/beta6_pretrain_corpus_manifest.md`
- Test: `backend/tests/test_build_pretrain_corpus_manifest_script.py`
- Test: `backend/tests/test_beta6_data_manifest.py`
- Test: `backend/tests/test_beta6_trainer_intake_gate.py`
- Test: `backend/tests/test_beta6_intake_gate.py`
- Test: `backend/tests/test_beta6_intake_precheck.py`

**Step 1: Write the failing tests**

Add tests for:

```python
def test_intake_manifest_fingerprints_and_dedupes_sources(): ...
def test_intake_manifest_tracks_user_and_date_tags(): ...
def test_intake_gate_auto_approves_clean_files(): ...
def test_intake_gate_quarantines_red_flag_files_with_explicit_reasons(): ...
def test_intake_summary_includes_per_room_per_date_label_counts(): ...
```

**Step 2: Run the focused tests to verify failure**

Run:

```bash
pytest backend/tests/test_build_pretrain_corpus_manifest_script.py backend/tests/test_beta6_data_manifest.py backend/tests/test_beta6_trainer_intake_gate.py backend/tests/test_beta6_intake_gate.py backend/tests/test_beta6_intake_precheck.py -q
```

Expected: failure because the current intake contract does not yet govern auto-approval vs quarantine end-to-end.

**Step 3: Write the minimal implementation**

Add:
1. source manifesting
2. fingerprint / dedupe
3. user/date tagging
4. schema / quality checks
5. per-room / per-date label summaries
6. auto-approval unless red flags
7. quarantine with explicit reasons
8. downstream readiness for the shared `20x14` corpus views

**Step 4: Run focused tests**

Run:

```bash
pytest backend/tests/test_build_pretrain_corpus_manifest_script.py backend/tests/test_beta6_data_manifest.py backend/tests/test_beta6_trainer_intake_gate.py backend/tests/test_beta6_intake_gate.py backend/tests/test_beta6_intake_precheck.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/scripts/build_pretrain_corpus_manifest.py backend/ml/beta6/data_manifest.py backend/ml/beta6/feature_store.py backend/ml/beta6/gates/intake_gate.py backend/ml/beta6/gates/intake_precheck.py docs/planning/beta6_pretrain_corpus_manifest.md backend/tests/test_build_pretrain_corpus_manifest_script.py backend/tests/test_beta6_data_manifest.py backend/tests/test_beta6_trainer_intake_gate.py backend/tests/test_beta6_intake_gate.py backend/tests/test_beta6_intake_precheck.py
git commit -m "feat: add beta62 intake governance and quarantine"
```

### Task 10: Add grouped-regime robustness gates

**Files:**
- Modify: `backend/ml/training.py`
- Modify: `backend/ml/root_cause_analysis.py`
- Modify: `backend/ml/room_experiments.py`
- Modify: `backend/scripts/run_room_experiments.py`
- Modify: `backend/ml/timeline_metrics.py`
- Test: `backend/tests/test_training.py`
- Test: `backend/tests/test_root_cause_analysis.py`
- Test: `backend/tests/test_run_room_experiments.py`

**Step 1: Write the failing tests**

Add tests for:

```python
def test_grouped_by_date_summary_emits_worst_date_and_range(): ...
def test_grouped_by_user_summary_emits_worst_user_and_range(): ...
def test_fragile_room_gates_can_block_on_worst_slice_instability(): ...
def test_room_experiments_can_report_grouped_regime_stability(): ...
```

**Step 2: Run the focused tests to verify failure**

Run:

```bash
pytest backend/tests/test_training.py backend/tests/test_root_cause_analysis.py backend/tests/test_run_room_experiments.py -q
```

Expected: failure because grouped-by-date / grouped-by-user worst-slice gating is not yet first-class.

**Step 3: Write the minimal implementation**

Add:
1. grouped-by-date evaluation
2. grouped-by-user evaluation
3. worst-slice selection and gating
4. fragile-room stability diagnostics
5. grouped-regime summaries in replay diagnostics

**Step 4: Run focused tests**

Run:

```bash
pytest backend/tests/test_training.py backend/tests/test_root_cause_analysis.py backend/tests/test_run_room_experiments.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/ml/training.py backend/ml/root_cause_analysis.py backend/ml/room_experiments.py backend/scripts/run_room_experiments.py backend/ml/timeline_metrics.py backend/tests/test_training.py backend/tests/test_root_cause_analysis.py backend/tests/test_run_room_experiments.py
git commit -m "feat: add beta62 grouped regime robustness gates"
```

### Task 11: Turn corrections into structured training signal

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

### Task 12: Add timeline-native targets and heads

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

### Task 13: Add context-conditioned modeling before demographic modeling

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

### Task 14: Improve learning efficiency and experiment throughput

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
4. fragile-room status is explicit as `pass`, `conditional`, or `block`
5. `GO` decisions permit `conditional` rooms but never any `block`
6. resident/home context contract exists and is typed
7. runtime topology expectations match saved runtime artifacts
8. PostgreSQL / historical-corrections availability is tracked explicitly in certification
9. real-environment rerun succeeds with signed artifacts and real runtime dependencies

### Beta 6.2

1. intake can separate auto-approved training files from quarantined files with explicit reasons
2. `20x14` is a shared Beta 6 asset, not a Beta 6.2 silo
3. grouped-by-date and grouped-by-user worst-slice stability are visible and governable
4. corrections flow into structured training assets
5. timeline-native targets and context-conditioned modeling are offline and isolated
6. manual correction demand is measurable and expected to decline over time
7. experiment throughput improves through caching and candidate triage
8. room-policy regressions can be diagnosed by replay sweeps before expensive full retrains

## Recommended Execution Order

1. Task 1 to Task 7: Beta 6.1 productionization and certification
2. Task 8 to Task 10: Beta 6.2 namespace, intake governance, and grouped-regime robustness
3. Task 11 to Task 14: Beta 6.2 correction/timeline/context/efficiency upgrades
