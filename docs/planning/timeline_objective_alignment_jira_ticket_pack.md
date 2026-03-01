# Jira Ticket Pack: Timeline Objective Alignment

## Usage
- This pack maps 1:1 to `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/timeline_objective_alignment_execution_plan.md`.
- Create one Epic first, then Stories, then Sub-tasks.
- Replace placeholder keys (for example `TIMELINE-1`) with your actual Jira keys.

## Epic

### `EPIC` Timeline Objective Alignment for Reliable ADL Timeline
- Objective:
  - Move from event-pass capable to timeline-reliable output.
- Scope:
  - Training objective, decoder, metrics/gates, pipeline shadow, and signoff evidence.
- Success criteria:
  - Candidate beats `v31` baseline on timeline metrics across rolling 3-seed evaluation without critical event regressions.

---

## Stories and Sub-Tasks

### `STORY` WS-0 Baseline Freeze + Reproducibility Pointer
- Summary:
  - Freeze baseline artifacts and embed baseline references in aggregate signoff payload.
- Story points: `3`
- Owner lane: `T3`
- Files:
  - `backend/scripts/aggregate_event_first_backtest.py`
  - `backend/tests/test_event_first_backtest_aggregate.py`
  - `backend/logs/eval/hk0011_event_first_signoff_dec4_to10_v31.json`
- Sub-tasks:
  - Add `baseline_version` and `baseline_artifact_hash` to signoff payload.
  - Add aggregate test assertions for baseline fields.
  - Document exact baseline artifact filenames and hashes.
- Acceptance criteria:
  - Signoff JSON contains baseline pointer fields.
  - Aggregate tests pass.

### `STORY` WS-1 Timeline Target Builder
- Summary:
  - Build deterministic boundary and episode-attribute targets from existing labels.
- Story points: `5`
- Owner lane: `T1`
- Files:
  - `backend/ml/timeline_targets.py` (new)
  - `backend/tests/test_timeline_targets.py` (new)
  - `backend/ml/event_labels.py`
  - `backend/scripts/run_event_first_backtest.py`
- Sub-tasks:
  - Implement `build_boundary_targets(labels)`.
  - Implement `build_episode_attribute_targets(timestamps, labels, room_name)`.
  - Emit debug timeline targets in backtest report.
  - Add deterministic synthetic tests.
- Acceptance criteria:
  - Target generation is deterministic.
  - Boundary targets validated on synthetic edge cases.
  - No temporal leakage.

### `STORY` WS-2 Multi-Task Objective in CNN+Transformer
- Summary:
  - Add timeline-aware training losses and outputs while preserving compatibility.
- Story points: `8`
- Owner lane: `T1`
- Files:
  - `backend/ml/transformer_timeline_heads.py` (new)
  - `backend/ml/transformer_backbone.py`
  - `backend/ml/training.py`
  - `backend/ml/transformer_head_ab.py`
  - `backend/tests/test_transformer_timeline_heads.py` (new)
  - `backend/tests/test_training.py`
- Sub-tasks:
  - Add outputs: `boundary_start_logits`, `boundary_end_logits`, `daily_duration_pred`, `daily_count_pred`.
  - Add weighted multi-task loss in training.
  - Add feature flag `ENABLE_TIMELINE_MULTITASK`.
  - Keep old inference path unchanged when flag is off.
- Acceptance criteria:
  - Multi-task model compiles and trains.
  - Existing tests pass with feature flag off.
  - New tests validate output shapes and losses.

### `STORY` WS-3 Timeline Decoder v2
- Summary:
  - Introduce room-aware segment reconstruction with boundary-aware constraints.
- Story points: `8`
- Owner lane: `T2`
- Files:
  - `backend/ml/timeline_decoder_v2.py` (new)
  - `backend/ml/event_decoder.py`
  - `backend/scripts/run_event_first_backtest.py`
  - `backend/tests/test_timeline_decoder_v2.py` (new)
- Sub-tasks:
  - Implement `TimelineDecodePolicy`.
  - Implement `decode_timeline_v2(...)`.
  - Integrate decoder v2 as optional path in backtest.
  - Add deterministic tests for merge/split continuity behavior.
- Acceptance criteria:
  - Fragmentation rate decreases vs baseline in controlled replay.
  - No collapse in critical events.

### `STORY` WS-4 Timeline Metrics + Promotion Gates
- Summary:
  - Add timeline-native metrics and gate checks for release decisions.
- Story points: `5`
- Owner lane: `T3`
- Files:
  - `backend/ml/event_metrics.py`
  - `backend/ml/event_gates.py`
  - `backend/scripts/aggregate_event_first_backtest.py`
  - `backend/tests/test_event_metrics.py`
  - `backend/tests/test_event_gates.py`
  - `backend/tests/test_event_first_backtest_aggregate.py`
- Sub-tasks:
  - Add metrics: start/end MAE, duration MAE, episode count error, fragmentation rate.
  - Add internal/external timeline gate tiers.
  - Add timeline gate checks into signoff payload.
- Acceptance criteria:
  - Timeline checks are present in signoff JSON.
  - Synthetic failing scenarios are blocked.

### `STORY` WS-5 Pipeline + Unified Training Shadow Integration
- Summary:
  - Wire timeline-multitask + decoder-v2 into shadow mode without changing production promotion behavior.
- Story points: `5`
- Owner lane: `T4`
- Files:
  - `backend/ml/pipeline.py`
  - `backend/ml/unified_training.py`
  - `backend/ml/training.py`
  - `backend/tests/test_unified_training_path.py`
  - `backend/tests/test_pipeline_integration.py`
- Sub-tasks:
  - Add flags: `event_first_shadow`, `timeline_multitask`, `timeline_decoder_v2`.
  - Persist shadow artifacts: `timeline_windows.parquet`, `timeline_episodes.parquet`, `timeline_qc.json`.
  - Ensure non-shadow behavior remains unchanged.
- Acceptance criteria:
  - Shadow artifacts generated when enabled.
  - Existing production path unchanged when disabled.

### `STORY` WS-6 Rolling Backtest + Signoff Pack
- Summary:
  - Execute fixed rolling protocol and produce decision-grade signoff artifacts.
- Story points: `5`
- Owner lane: `T3`
- Files:
  - `backend/scripts/run_event_first_backtest.py`
  - `backend/scripts/aggregate_event_first_backtest.py`
  - `backend/scripts/generate_residual_review_pack.py`
  - `backend/tests/test_event_first_backtest_script.py`
  - `backend/tests/test_generate_residual_review_pack.py`
- Sub-tasks:
  - Run seeds `11/22/33` with fixed config hash.
  - Generate rolling, signoff, residual pack, residual windows CSV.
  - Produce baseline-vs-candidate delta summary.
- Acceptance criteria:
  - All required artifacts generated.
  - Candidate passes gates and beats baseline on timeline metrics target.

---

## Dependency Graph
- WS-0 -> WS-4, WS-6
- WS-1 -> WS-2, WS-3
- WS-2 -> WS-5
- WS-3 -> WS-5, WS-6
- WS-4 -> WS-6
- WS-5 -> WS-6

---

## Common Definition of Done
- Code merged with tests.
- Feature flags default-safe.
- No regression in touched baseline suites.
- Artifact schema updated and validated.
- PR notes include executed commands and generated artifact paths.

---

## Standard Verification Commands
- Unit/integration touched suites:
  - `pytest -q tests/test_event_first_backtest_script.py tests/test_event_first_backtest_aggregate.py tests/test_generate_residual_review_pack.py`
  - `pytest -q tests/test_training.py tests/test_unified_training_path.py tests/test_pipeline_integration.py`
- Rolling evaluation:
  - `python3 scripts/run_event_first_backtest.py ...`
  - `python3 scripts/aggregate_event_first_backtest.py ...`
  - `python3 scripts/generate_residual_review_pack.py ...`
