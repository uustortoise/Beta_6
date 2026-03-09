# Beta 6.1 Authority Hardening and Beta 6.2 Execution Plan

> **Superseded:** Use [2026-03-09-beta61-beta62-reliable-activity-timeline-execution.md](/Users/dickson/DT/DT_development/Development/Beta_6/docs/plans/2026-03-09-beta61-beta62-reliable-activity-timeline-execution.md) for the current execution plan based on `codex/pilot-bootstrap-gates` at `94524af`.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute Beta 6.1 as the release track that turns the now-good integrated Jessica model stack into a real authority-capable system, while starting Beta 6.2 as an isolated data/model improvement track that does not perturb Beta 6.1 outputs.

**Architecture:** Beta 6.1 uses the existing integrated room-model stack on `codex/beta6-integration-rerun` as the model baseline and focuses only on authority/runtime completeness: signed artifacts, deterministic fallback/rollback, explicit evidence profiles, production-like environment readiness, and shadow certification. Beta 6.2 is a separate namespace and data program built around the shared `20x14` corpus contract, timeline-native supervision, context-conditioned modeling, and training/runtime efficiency work, with strict isolation from Beta 6.1 behavior when Beta 6.2 flags are off.

**Tech Stack:** Python 3.12, pytest, PostgreSQL, Beta 6 registry v2, signed JSON artifacts, YAML policy/config, pandas/NumPy, timeline evaluation/gates, offline training scripts.

---

## Current Baseline

### Integrated Jessica status on the current good stack
The integrated rerun `beta6_daily_HK0011_jessica_20260308T005915Z` passed all five room release gates on `codex/beta6-integration-rerun`:

1. Bathroom: `macro_f1=0.6377`
2. Bedroom: `macro_f1=0.6456`
3. Entrance: `macro_f1=0.9990`
4. Kitchen: `macro_f1=0.7180`
5. LivingRoom: `macro_f1=0.7030`

### Remaining blockers
The run still failed above the room layer because:

1. `BETA6_GATE_SIGNING_KEY` was missing for live authority signing.
2. `livingroom` had no fallback target in registry/rollback flow.
3. PostgreSQL was unavailable, so `pipeline_success_rate=0.0`.
4. `RELEASE_GATE_EVIDENCE_PROFILE` defaulted silently instead of being explicit.

### Planning consequence
There is no current execution-ready plan that matches this updated state. The older Beta 6.1 executable plan in `docs/planning/beta6_1_executable_plan_2026-03-02.md` is now outdated because it assumes room-model reliability is the main blocker. This document replaces it for active execution.

## Execution Rules

1. Beta 6.1 is the release track. No Beta 6.2 work may change Beta 6.1 outputs while Beta 6.2 flags are off.
2. Every Beta 6.1 task must be test-first where practical and must include a concrete rerun or dry-run check.
3. Every Beta 6.2 task must remain offline-only until Beta 6.1 passes a real-environment authority rerun.
4. The `20x14` corpus is a Beta 6 shared asset, not a Beta 6.2-only asset.
5. Metadata use is split by layer:
   - Beta 6.1: policy/routing/gating/evaluation only
   - Beta 6.2: optional model input only after offline ablation

## Beta 6.1 Execution Plan

### Task 1: Freeze the integrated Beta 6.1 baseline

**Purpose:** Lock the known-good integrated model stack and its evidence before authority/runtime edits.

**Files:**
- Modify: `docs/plans/2026-03-08-beta61-beta62-executable-plan.md`
- Read/verify: `backend/models/HK0011_jessica/*_decision_trace.json`
- Read/verify: `backend/models_beta6_registry_v2/HK0011_jessica/_run/events.jsonl`
- Read/verify: `backend/tmp/beta6_gate_artifacts/HK0011_jessica/beta6_daily_HK0011_jessica_20260308T005915Z/*`

**Implementation steps:**
1. Record the exact integrated branch SHA and run id.
2. Snapshot the five room traces and the latest run event evidence into the execution notes.
3. Confirm the current blocker list is authority/runtime only.

**Testing procedure:**
```bash
git -C .worktrees/beta6-integration-rerun rev-parse --short HEAD
jq '.release_gate.pass, .macro_f1' .worktrees/beta6-integration-rerun/backend/models/HK0011_jessica/Bathroom_v6_decision_trace.json
jq '.release_gate.pass, .macro_f1' .worktrees/beta6-integration-rerun/backend/models/HK0011_jessica/Bedroom_v3_decision_trace.json
jq '.release_gate.pass, .macro_f1' .worktrees/beta6-integration-rerun/backend/models/HK0011_jessica/Entrance_v17_decision_trace.json
jq '.release_gate.pass, .macro_f1' .worktrees/beta6-integration-rerun/backend/models/HK0011_jessica/Kitchen_v2_decision_trace.json
jq '.release_gate.pass, .macro_f1' .worktrees/beta6-integration-rerun/backend/models/HK0011_jessica/LivingRoom_v4_decision_trace.json
tail -n 1 .worktrees/beta6-integration-rerun/backend/models_beta6_registry_v2/HK0011_jessica/_run/events.jsonl
```

**Pass requirements:**
1. All five traces show `release_gate.pass=true`.
2. Latest run event still shows run-level failure driven by authority/runtime blockers.

### Task 2: Harden the signing-key contract

**Purpose:** Make live authority signing fail early, clearly, and deterministically.

**Files:**
- Modify: `backend/run_daily_analysis.py`
- Modify: `backend/.env.example`
- Test: `backend/tests/test_run_daily_analysis_beta6_authority.py`
- Test: `backend/tests/test_beta6_dynamic_gate_artifacts.py`

**Implementation steps:**
1. Add failing tests for missing `BETA6_GATE_SIGNING_KEY` in registry-v2 live authority mode.
2. Move signing-key resolution to a single explicit preflight path.
3. Ensure the failure is surfaced before expensive finalization/rollout steps.
4. Make the evidence profile explicit in the same preflight surface.

**Testing procedure:**
```bash
pytest backend/tests/test_run_daily_analysis_beta6_authority.py backend/tests/test_beta6_dynamic_gate_artifacts.py -q
python3 backend/run_daily_analysis.py --help >/dev/null
```

**Pass requirements:**
1. Missing signing key produces a deterministic testable failure reason.
2. Live authority path cannot silently proceed without the key.
3. Tests covering missing-key and explicit-evidence-profile paths pass.

### Task 3: Fix registry fallback-target completeness

**Purpose:** Ensure rollback/fallback never fails just because a room lacks a materialized fallback pointer.

**Files:**
- Modify: `backend/ml/beta6/registry/registry_v2.py`
- Modify: `backend/ml/t80_rollout_manager.py`
- Modify: `backend/ml/beta6/serving/serving_loader.py`
- Test: `backend/tests/test_beta6_registry_v2.py`
- Test: `backend/tests/test_t80_rollout_manager.py`
- Test: `backend/tests/test_beta6_serving_loader.py`

**Implementation steps:**
1. Add failing tests for `livingroom`-style missing fallback target behavior.
2. Define deterministic fallback resolution order from current champion/history/previous known good record.
3. Make rollback protection consume that deterministic resolver instead of assuming pre-materialized pointers.
4. Ensure failure mode is explicit when no safe fallback truly exists.

**Testing procedure:**
```bash
pytest backend/tests/test_beta6_registry_v2.py backend/tests/test_t80_rollout_manager.py backend/tests/test_beta6_serving_loader.py -q
```

**Pass requirements:**
1. No `No fallback target available ...` exception remains for valid rollback scenarios.
2. Rollback path has a deterministic fallback choice in tests.
3. Truly missing fallback cases fail with explicit, auditable reason codes.

### Task 4: Resolve authority artifact/logging contradictions

**Purpose:** Remove contradictory pass/fail semantics between run events, artifact writing, and deferred promotion logging.

**Files:**
- Modify: `backend/run_daily_analysis.py`
- Modify: `backend/ml/beta6/registry/registry_v2.py`
- Test: `backend/tests/test_run_daily_analysis_beta6_authority.py`
- Test: `backend/tests/test_beta6_dynamic_gate_artifacts.py`

**Implementation steps:**
1. Add failing tests for “run failed but promoted deferred candidates” contradictions.
2. Split artifact/reporting states into clear phases: room pass, authority pass, deferred promotion decision.
3. Ensure event log wording and final artifact states cannot contradict one another.

**Testing procedure:**
```bash
pytest backend/tests/test_run_daily_analysis_beta6_authority.py backend/tests/test_beta6_dynamic_gate_artifacts.py -q -k 'deferred or authority'
```

**Pass requirements:**
1. A failed authority run cannot emit a promotion-success semantic event.
2. Artifact/report/test outputs agree on the same final authority state.

### Task 5: Real-environment preflight and rerun

**Purpose:** Re-run the already-good model stack in a production-like environment and verify authority completion.

**Files:**
- Modify: `backend/run_daily_analysis.py`
- Modify: `backend/.env.example`
- Optional docs: `docs/plans/2026-03-08-beta61-beta62-executable-plan.md`
- Test: `backend/tests/test_run_daily_analysis_beta6_authority.py`
- Test: `backend/tests/test_health_server.py`

**Implementation steps:**
1. Add a preflight helper for:
   - `BETA6_GATE_SIGNING_KEY`
   - PostgreSQL connectivity
   - explicit `RELEASE_GATE_EVIDENCE_PROFILE`
   - registry/rollout fallback readiness
2. Run the integrated branch in a real environment with those prerequisites satisfied.
3. Verify signed evaluation artifacts, phase-6 stability report, and `_run/events.jsonl`.

**Testing procedure:**
```bash
pytest backend/tests/test_run_daily_analysis_beta6_authority.py backend/tests/test_health_server.py -q
python3 backend/run_daily_analysis.py
```

**Pass requirements:**
1. All five rooms pass again on the integrated branch.
2. Authority passes.
3. Signed artifacts are written.
4. `phase6_stability_report.json` no longer shows `pipeline_success_rate=0.0`.
5. No signing-key or fallback-target errors appear in `_run/events.jsonl`.

### Task 6: Beta 6.1 shadow certification and resident/home context contract

**Purpose:** Turn one successful run into an operationally credible release path.

**Files:**
- Modify: `backend/db/schema.sql`
- Modify: `backend/processors/profile_processor.py`
- Modify: `backend/ml/household_analyzer.py`
- Modify: `backend/ml/home_empty_fusion.py`
- Modify: `backend/ml/beta6/sequence/transition_builder.py`
- Modify: `backend/export_dashboard.py`
- Test: `backend/tests/test_home_empty_fusion.py`
- Test: `backend/tests/test_health_server.py`
- Add tests as needed for context parsing/validation

**Implementation steps:**
1. Introduce a typed resident/home context contract:
   - single vs multi resident
   - helper presence
   - layout / adjacency / sensor topology
2. Use it for routing, gating, arbitration, and evaluation cohorting only.
3. Add dashboard/reporting visibility for context completeness.
4. Run shadow certification on Jessica plus at least one additional resident.

**Testing procedure:**
```bash
pytest backend/tests/test_home_empty_fusion.py backend/tests/test_health_server.py -q
```

**Pass requirements:**
1. Context has a typed source of truth.
2. Shadow certification can operate without hidden env assumptions.
3. Jessica plus at least one additional resident reach the planned stable-day window.

## Beta 6.2 Execution Plan

### Task 7: Establish the shared `20x14` corpus contract

**Purpose:** Start Beta 6.2 now without making the data program Beta 6.2-only.

**Files:**
- Modify: `docs/plans/2026-03-08-beta61-beta62-executable-plan.md`
- Modify: `docs/planning/beta6_pretrain_corpus_manifest.md`
- Modify: `backend/scripts/build_pretrain_corpus_manifest.py`
- Add: `backend/tests/test_beta6_trainer_intake_gate.py`
- Add/modify tests for manifest builder if absent

**Implementation steps:**
1. Define the shared corpus contract:
   - authority shadow cohort view
   - unlabeled pretrain view
   - labeled high-trust fine-tune/eval view
2. Add manifest fields for resident/home context completeness and labeling status.
3. Add intake gates for missing days, broken context, and label quality.

**Testing procedure:**
```bash
pytest backend/tests/test_beta6_trainer_intake_gate.py -q
python3 backend/scripts/build_pretrain_corpus_manifest.py --help >/dev/null
```

**Pass requirements:**
1. `20x14` is defined as a Beta 6 shared asset, not a Beta 6.2-only artifact.
2. Manifest builder can encode all three corpus views.
3. Intake gating rejects incomplete or malformed cohorts deterministically.

### Task 8: Create a Beta 6.2 isolated namespace

**Purpose:** Allow offline Beta 6.2 development without changing Beta 6.1 outputs.

**Files:**
- Modify: `backend/ml/beta6/training/beta6_trainer.py`
- Modify: `backend/ml/beta6/training/self_supervised_pretrain.py`
- Modify: `backend/ml/beta6/serving_loader.py`
- Modify: `backend/.env.example`
- Add tests around flag-off parity

**Implementation steps:**
1. Create explicit Beta 6.2 config/module flags and artifact namespace.
2. Guarantee Beta 6.1 parity when Beta 6.2 flags are off.
3. Keep Beta 6.2 outputs out of the authority/serving default path.

**Testing procedure:**
```bash
pytest backend/tests/test_beta6_self_supervised_pretrain.py backend/tests/test_beta6_serving_loader.py -q
```

**Pass requirements:**
1. Beta 6.2 artifacts cannot be loaded by Beta 6.1 accidentally.
2. Flag-off parity is preserved.
3. Beta 6.2 can train/evaluate offline without touching rollout code paths.

### Task 9: Add timeline-native heads and baseline evaluation

**Purpose:** Improve timeline quality rather than only per-window classification.

**Files:**
- Modify: `backend/ml/transformer_timeline_heads.py`
- Modify: `backend/ml/beta6/training/beta6_trainer.py`
- Modify: `backend/ml/beta6/gates/timeline_hard_gates.py`
- Test: `backend/tests/test_transformer_timeline_heads.py`
- Test: `backend/tests/test_timeline_targets.py`
- Test: `backend/tests/test_timeline_metrics.py`
- Test: `backend/tests/test_timeline_gates.py`

**Implementation steps:**
1. Add offline-only timeline heads for onset/offset/duration or segment targets.
2. Define training targets and evaluation summaries for event-native behavior.
3. Benchmark against the Beta 6.1 integrated baseline without changing serving outputs.

**Testing procedure:**
```bash
pytest backend/tests/test_transformer_timeline_heads.py backend/tests/test_timeline_targets.py backend/tests/test_timeline_metrics.py backend/tests/test_timeline_gates.py -q
```

**Pass requirements:**
1. Timeline-native targets and metrics are reproducible offline.
2. Beta 6.2 reports include event-oriented metrics, not only frame/window metrics.
3. No Beta 6.1 default behavior changes when flags are off.

### Task 10: Add context-conditioned modeling

**Purpose:** Use the highest-value metadata where it is most likely to improve timeline quality.

**Files:**
- Modify: `backend/ml/beta6/training/beta6_trainer.py`
- Modify: `backend/ml/beta6/sequence/transition_builder.py`
- Modify: `backend/ml/home_empty_fusion.py`
- Modify: `backend/ml/household_analyzer.py`
- Test: `backend/tests/test_home_empty_fusion.py`
- Add tests for context-conditioned trainer behavior as needed

**Implementation steps:**
1. Start with layout/adjacency/sensor topology context.
2. Add household/helper context second.
3. Keep age/sex out of model inputs until later ablations justify inclusion.

**Testing procedure:**
```bash
pytest backend/tests/test_home_empty_fusion.py backend/tests/test_transformer_timeline_heads.py -q
```

**Pass requirements:**
1. Context-conditioned paths are offline-only.
2. Layout/topology and household context are testable and auditable.
3. No age/sex modeling path is default-on.

### Task 11: Improve training/runtime efficiency

**Purpose:** Reduce retrain cost and make future Beta 6.2 iteration faster.

**Files:**
- Modify: `backend/ml/beta6/training/beta6_trainer.py`
- Modify: `backend/ml/beta6/training/self_supervised_pretrain.py`
- Modify: `backend/scripts/build_pretrain_corpus_manifest.py`
- Add tests/benchmarks around cache keys and candidate pruning

**Implementation steps:**
1. Add canonical cached feature/sequence storage.
2. Reuse tensors across retries/seeds where safe.
3. Add early candidate elimination for obviously bad branches.
4. Keep cache keys tied to typed policy and manifest fingerprints.

**Testing procedure:**
```bash
pytest backend/tests/test_beta6_self_supervised_pretrain.py backend/tests/test_beta6_trainer_intake_gate.py -q
```

**Pass requirements:**
1. Identical inputs/policy reuse cached artifacts safely.
2. Cache invalidates on manifest/policy changes.
3. Beta 6.2 offline experiment wall-clock drops materially versus uncached baseline.

## Exit Criteria

### Beta 6.1 exit criteria
1. A real-environment integrated rerun passes all rooms and authority/runtime gates.
2. Signed artifacts are written successfully.
3. Fallback/rollback paths are deterministic and tested.
4. Jessica plus at least one additional resident complete the shadow-certification window.

### Beta 6.2 start/continuation criteria
1. Namespace isolation is in place.
2. `20x14` corpus contract is defined and executable.
3. Beta 6.2 remains strictly non-blocking to Beta 6.1.

## Recommended Execution Order

1. Task 1 to Task 5: Beta 6.1 authority/runtime hardening and real-environment rerun
2. Task 6: Beta 6.1 shadow certification and context contract
3. Task 7 to Task 8: Beta 6.2 corpus contract and namespace isolation
4. Task 9 to Task 11: Beta 6.2 timeline/context/efficiency work

## Immediate Recommendation

Execute Beta 6.1 Task 2 to Task 5 first. Beta 6.2 can start now, but only with Task 7 and Task 8 until Beta 6.1 completes a real-environment authority rerun.
