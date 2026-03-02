# Beta 6 Canonical Training Store Execution Plan (2026-03-02)

## Objective
Deliver a safe migration from filename-based training aggregation to a versioned canonical training store, with shadow validation before authority cutover.

Primary outcomes:
1. Remove filename fragility from retrain corpus assembly.
2. Provide content-based reproducibility and auditability for retraining.
3. Enable row-level correction merges without destabilizing production training.
4. Preserve one-step rollback and fail-closed operations during rollout.

## Why Now
1. Current dedupe and fingerprinting are file-identity based and brittle for corrected uploads.
2. Team now has a clear phased design and can execute with controlled blast radius.
3. Shadow-first rollout aligns with Beta 6 operating model and existing release-gate discipline.

## Scope
In scope:
1. New `CanonicalTrainingStore` module with versioning, changelog, content hash, rollback.
2. Training-eligible ingest integration in `process_data.py` and `run_daily_analysis.py`.
3. New retrain modes: `canonical_shadow`, `canonical`.
4. Canonical content hash integration into training run fingerprinting.
5. Validation pipeline: strict parity then expected-delta checks.

Out of scope:
1. Immediate default switch of `RETRAIN_INPUT_MODE` to canonical.
2. Training algorithm redesign.
3. Deletion of legacy archive flows.

## Execution Principles
1. Shadow-first, no forced cutover.
2. Fail-closed on evidence and data-integrity checks.
3. Atomic and idempotent writes for canonical state.
4. Keep current training pipeline contract stable in early phases (snapshot adapter).
5. Every phase has explicit exit criteria and rollback path.

## Workstream Structure

### WS1: Canonical Store Core
Objective:
1. Build robust canonical storage primitives before integration.

Code touchpoints:
1. `backend/ml/canonical_store.py` (new)
2. `backend/tests/test_canonical_store.py` (new)

Tasks:
1. Implement per-elder store layout and manifest/changelog schema.
2. Implement `ingest_paths`, `ingest_frames`, `load`, `content_hash`, `rollback`.
3. Add lock + atomic write semantics (`.lock`, temp file + `os.replace`).
4. Add source-hash idempotency to avoid duplicate ingests.

Deliverables:
1. New module + tests.
2. Schema docs in module docstring and tests.

Exit criteria:
1. Unit tests green.
2. Crash/fault-injection test confirms no partial-manifest advance.
3. Idempotent ingest verified by source hash.

Owner:
1. ML Platform / Backend

Estimate:
1. 2.0 to 3.0 engineering days

Dependencies:
1. None

---

### WS2: Archive and Ingest Integration
Objective:
1. Keep canonical store synchronized without data pollution.

Code touchpoints:
1. `backend/process_data.py`
2. `backend/run_daily_analysis.py`

Tasks:
1. Refactor `archive_file(...)` to return `ArchiveResult` with `status`, `source_path`, `archived_path`.
2. Add strict training eligibility guard before canonical ingest:
   - `_is_training_file(path)`
   - contains `activity`
   - elder lineage match
3. Ingest from safe source (`archived_path` or preloaded frames), not moved raw path.
4. Add structured logs for ingest report and skip reasons.

Deliverables:
1. `ArchiveResult` contract + caller updates.
2. Integration tests covering eligibility and duplicate behavior.

Exit criteria:
1. No non-training file enters canonical store in integration tests.
2. Duplicate and missing file states handled without ingest failure.

Owner:
1. Backend Runtime

Estimate:
1. 1.5 to 2.5 engineering days

Dependencies:
1. WS1

---

### WS3: Canonical Read Path and Fingerprint Alignment
Objective:
1. Enable canonical-mode training with minimal contract disruption.

Code touchpoints:
1. `backend/run_daily_analysis.py`
2. `backend/tests/test_run_daily_analysis_thresholds.py`

Tasks:
1. Add modes `canonical_shadow` and `canonical` to mode resolution.
2. In canonical mode, materialize canonical snapshot parquet and pass `[snapshot_path]` into existing `train_from_files(...)`.
3. Update `_compute_training_run_fingerprint(...)`:
   - canonical modes use `store.content_hash()`
   - legacy modes keep manifest path/size/mtime behavior
4. Add training-history metadata fields:
   - `canonical_mode`
   - `canonical_version`
   - `canonical_content_hash`

Deliverables:
1. Mode handling + fingerprint implementation.
2. Targeted tests for mode routing and fingerprint behavior.

Exit criteria:
1. Canonical mode executes full training pipeline without API break.
2. Fingerprint behavior deterministic across reruns.

Owner:
1. ML Runtime + Backend

Estimate:
1. 1.5 to 2.0 engineering days

Dependencies:
1. WS1, WS2

---

### WS4: Validation and Promotion Gates
Objective:
1. Prove correctness before any authority switch.

Code touchpoints:
1. `backend/tests/` (new parity/expected-delta tests)
2. Optional validation scripts under `backend/scripts/`

Tasks:
1. Implement strict parity validator (non-overlap row parity).
2. Implement expected-delta validator for overlap corrections.
3. Emit daily validation summary artifact for shadow runs.
4. Add promotion gate requiring 14 consecutive clean days.

Deliverables:
1. Validation report template + automation path.
2. Go/no-go gate checklist artifact.

Exit criteria:
1. 14/14 consecutive days clean.
2. All diffs explained by changelog overlap semantics.

Owner:
1. QA + ML Platform

Estimate:
1. 2.0 to 3.0 engineering days (plus 14-day run window)

Dependencies:
1. WS3

---

### WS5: Migration Backfill and Operational Cutover
Objective:
1. Backfill existing archive corpus safely and execute controlled cutover.

Code touchpoints:
1. `backend/scripts/migrate_archive_to_canonical.py` (new)
2. rollout config / runbook docs

Tasks:
1. Implement deterministic backfill ordering and lineage grouping.
2. Produce migration report: source files, ingested rows, overwrites, skips, errors.
3. Run pilot backfill on one elder, then cohort.
4. Execute staged cutover plan with explicit rollback triggers.

Deliverables:
1. Migration script + report format.
2. Cutover runbook and rollback drill evidence.

Exit criteria:
1. Pilot backfill clean.
2. Cutover drill passes rollback SLA.
3. Release approval signed by owner triad.

Owner:
1. MLOps + Backend + QA

Estimate:
1. 1.5 to 2.0 engineering days (excluding shadow observation window)

Dependencies:
1. WS4

## Timeline (Calendar)

Assumes kickoff on Monday, 2026-03-09.

1. Week 1 (2026-03-09 to 2026-03-13): WS1 + WS2 complete.
2. Week 2 (2026-03-16 to 2026-03-20): WS3 complete, begin shadow runs.
3. Weeks 3-4 (2026-03-23 to 2026-04-05): WS4 validation window (14 consecutive days).
4. Week 5 start (earliest 2026-04-06): WS5 pilot cutover if all gates pass.

Earliest safe cutover date:
1. 2026-04-06 (only if no active release hardening and all promotion gates pass).

## Milestone Checklist

1. M1 Core ready: canonical store API and tests merged.
2. M2 Ingest safe: archive integration merged, no non-training pollution.
3. M3 Canonical mode runnable: snapshot adapter + fingerprint update merged.
4. M4 Shadow validated: 14 clean days with explained diffs only.
5. M5 Cutover approved: runbook, rollback drill, and signoff complete.

## Roles and Accountability (RACI-lite)

1. Engineering Lead (A): final go/no-go and priority arbitration.
2. ML Platform (R): WS1, WS3 store/fingerprint logic.
3. Backend Runtime (R): WS2 archive/integration path.
4. QA/Data Validation (R): WS4 parity/expected-delta pipeline and evidence.
5. MLOps (R): WS5 migration automation, rollout, rollback drills.
6. Product/Ops (C): acceptance criteria for operational behavior.

## Risk Register and Mitigations

1. Risk: canonical corruption via concurrent writers.
   Mitigation: per-elder lock + atomic replace + failure rollback.
2. Risk: non-training files ingested.
   Mitigation: strict eligibility guard with tests and metrics.
3. Risk: parity confusion from intentional overwrite semantics.
   Mitigation: split strict parity and expected-delta phases.
4. Risk: production disruption at cutover.
   Mitigation: shadow-first, staged enablement, rollback SLA.
5. Risk: team ambiguity during rollout.
   Mitigation: weekly checkpoint + daily shadow summary artifacts.

## Quality Gates

Code/test gates:
1. Unit tests for canonical store and rollback semantics.
2. Integration tests for eligibility, mode routing, and fingerprint mode behavior.
3. No regression in existing retrain-mode tests.

Operational gates:
1. 14 clean shadow days.
2. No unexplained divergence between legacy and canonical outputs.
3. Rollback drill meets SLA:
   - scoped rollback <= 15 minutes
   - full rollback <= 30 minutes

## Communication Plan

Cadence:
1. Daily async status update in team channel (progress + blockers + gate state).
2. Twice-weekly 30-minute execution review during WS1-WS3.
3. Daily validation digest during shadow window.
4. Formal go/no-go review before any cutover.

Artifacts for circulation:
1. This execution plan.
2. Weekly milestone tracker.
3. Shadow validation digest (daily).
4. Go/no-go decision record.

## Go / No-Go Criteria

Go when all are true:
1. M1-M4 completed.
2. 14-day validation clean and signed by QA + ML lead.
3. Rollback drill evidence available and SLA-met.
4. No conflicting major release hardening in same window.

No-Go if any are true:
1. Reproducible unexplained diffs in shadow window.
2. Canonical ingest pollution observed.
3. Rollback drill fails SLA or has unresolved gaps.

## Immediate Next Actions (This Week)

1. Open implementation tickets for WS1-WS3 with owners and estimates.
2. Finalize `ArchiveResult` schema and tests first (blocks WS2).
3. Create `test_canonical_store.py` scaffolding and CI entry.
4. Prepare shadow validation report template before mode enablement.

