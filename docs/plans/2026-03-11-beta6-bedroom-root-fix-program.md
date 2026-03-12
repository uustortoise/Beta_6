# Beta6 Bedroom Root-Fix Program Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prove whether the harmful added Bedroom dates are bad data or valid regime shift, then implement the corresponding root fix without prematurely widening the problem to all rooms.

**Architecture:** First add deterministic source-lineage observability so every Bedroom run records its exact inputs and per-date class mix. Then run a bounded Bedroom date-ablation matrix and harmful-day segment audit to classify the failure as either data-quality or valid-regime. Only after that decision gate should the code branch into a Bedroom data fix or a broader robustness fix, while keeping lightweight observability additions available for other rooms.

**Tech Stack:** Python, pandas, existing Beta 6 training pipeline, JSON/parquet forensic artifacts, pytest, existing room benchmark scripts.

---

### Task 1: Persist Bedroom Source-Lineage Metadata

**Files:**
- Modify: `backend/ml/training.py`
- Modify: `backend/ml/legacy/registry.py`
- Test: `backend/tests/test_training.py`
- Test: `backend/tests/test_registry.py`

**Step 1: Write the failing test**

Add a focused test in `backend/tests/test_training.py` asserting that a saved Bedroom training artifact includes:

- exact source paths
- stable source fingerprint
- per-date / per-label pre-sampling counts

Add a companion regression test in `backend/tests/test_registry.py` asserting the metadata survives save / load / alias promotion.

**Step 2: Run tests to verify they fail**

Run:

```bash
pytest backend/tests/test_training.py -q -k source_lineage_metadata
pytest backend/tests/test_registry.py -q -k source_lineage_metadata
```

Expected:

- both fail because the metadata is not yet persisted end-to-end

**Step 3: Write the minimal implementation**

In `backend/ml/training.py`:

- collect the exact source manifest for the active room run
- derive a stable fingerprint
- record per-date / per-label counts before sampling
- emit the metadata into train metrics and decision trace payloads

In `backend/ml/legacy/registry.py`:

- preserve the new metadata when promoting / materializing aliases

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest backend/tests/test_training.py -q -k source_lineage_metadata
pytest backend/tests/test_registry.py -q -k source_lineage_metadata
```

Expected:

- both pass

**Step 5: Run broader regression coverage**

Run:

```bash
pytest backend/tests/test_training.py backend/tests/test_registry.py -q
```

Expected:

- full suites pass

### Task 2: Build The Bedroom Date-Ablation Runner

**Files:**
- Create: `backend/scripts/run_bedroom_root_cause_matrix.py`
- Test: `backend/tests/test_run_bedroom_root_cause_matrix.py`
- Reference: `backend/scripts/livingroom_seed_forensic.py`
- Reference: `backend/tests/test_livingroom_seed_forensic.py`

**Step 1: Write the failing test**

Create `backend/tests/test_run_bedroom_root_cause_matrix.py` covering:

- anchor manifest creation for `2025-12-10` + `2025-12-17`
- single-day add-back variants for `2025-12-04` to `2025-12-09`
- cumulative add-back variants
- deterministic JSON manifest output without running the full retrain in dry-run mode

**Step 2: Run test to verify it fails**

Run:

```bash
pytest backend/tests/test_run_bedroom_root_cause_matrix.py -q
```

Expected:

- fail because the script does not yet exist

**Step 3: Write minimal implementation**

Create `backend/scripts/run_bedroom_root_cause_matrix.py` that can:

- build the ablation variant list
- write a manifest JSON describing each variant
- optionally execute Bedroom-only retrains against each variant
- record output artifact paths for later analysis

**Step 4: Run test to verify it passes**

Run:

```bash
pytest backend/tests/test_run_bedroom_root_cause_matrix.py -q
```

Expected:

- pass

**Step 5: Dry-run the matrix builder**

Run:

```bash
PYTHONPATH=.:backend python3 backend/scripts/run_bedroom_root_cause_matrix.py --dry-run --output-dir tmp/bedroom_root_cause_matrix_dry_run
```

Expected:

- manifest JSON is written
- no retrain is executed
- anchor plus add-back variants are listed deterministically

### Task 3: Execute The Bedroom Date-Ablation Matrix

**Files:**
- Use: `backend/scripts/run_bedroom_root_cause_matrix.py`
- Use: `tmp/`
- Reference: `backend/models/HK0011_jessica/*`
- Reference: `backend/models/HK0011_jessica_candidate_*/*`

**Step 1: Run the anchor and add-back variants**

Run the matrix runner for the Bedroom-only candidate namespace with the corrected Jessica workbooks.

Expected outputs per variant:

- train metrics JSON
- decision trace JSON
- Dec 17 replay summary JSON
- variant manifest entry with source fingerprint

**Step 2: Build the comparison table**

Summarize all variants into one table with:

- pre-sampling shares
- post-sampling shares
- holdout macro-F1
- holdout `bedroom_normal_use` recall
- Dec 17 final macro-F1
- Dec 17 confusion counts for the three critical error families

**Step 3: Verify the first harmful date is explicit**

Expected:

- either one single date clearly causes the regime break
- or the evidence shows a cumulative multi-day interaction

**Step 4: Stop and record the result before changing any training policy**

Write the intermediate forensic note under:

```text
docs/reviews/2026-03-11-beta6-bedroom-date-ablation-matrix.md
```

Expected:

- one explicit statement naming the first harmful date or the first harmful cumulative set

### Task 4: Build The Harmful-Day Segment Audit Tool

**Files:**
- Create: `backend/scripts/bedroom_day_segment_audit.py`
- Test: `backend/tests/test_bedroom_day_segment_audit.py`
- Reference: `backend/tests/test_check_beta6_training_lineage.py`

**Step 1: Write the failing test**

Create `backend/tests/test_bedroom_day_segment_audit.py` covering:

- splitting a day into bounded time blocks
- computing per-block label counts
- computing transition and run-length summaries
- marking obvious sparsity / missingness anomalies

**Step 2: Run test to verify it fails**

Run:

```bash
pytest backend/tests/test_bedroom_day_segment_audit.py -q
```

Expected:

- fail because the script does not yet exist

**Step 3: Write minimal implementation**

Create `backend/scripts/bedroom_day_segment_audit.py` that:

- reads the harmful-day source workbook or corrected pack slice
- partitions the day into fixed or configurable time windows
- outputs per-block summaries in JSON / parquet form
- supports comparison against the good anchor dates

**Step 4: Run test to verify it passes**

Run:

```bash
pytest backend/tests/test_bedroom_day_segment_audit.py -q
```

Expected:

- pass

**Step 5: Run the audit on the first harmful day**

Run the script against the first harmful day discovered in Task 3.

Expected:

- one or more blocks stand out as the strongest candidate explanation for the regime shift

### Task 5: Classify The Harmful Segments

**Files:**
- Use: `backend/scripts/bedroom_day_segment_audit.py`
- Create: `docs/reviews/2026-03-11-beta6-bedroom-harmful-day-segment-audit.md`

**Step 1: Compare harmful segments to the good anchor**

For each suspicious block, compare:

- class mix
- transition topology
- run-length structure
- sensor sparsity / missingness

**Step 2: Classify each segment**

Assign each segment one status:

- likely bad / inconsistent data
- likely valid alternate regime
- uncertain

**Step 3: Record the classification**

Expected:

- the review note makes the diagnosis explicit and evidence-based

**Step 4: Decision gate**

Only continue after the note states which branch is supported:

- Branch A: bad / inconsistent data
- Branch B: valid alternate regime
- Branch C: unresolved and what specific unknown remains

### Task 6A: Implement The Data-Fix Branch If The Harmful Segments Are Bad Data

**Files:**
- Modify: `backend/scripts/run_bedroom_root_cause_matrix.py`
- Modify: `backend/ml/training.py`
- Test: `backend/tests/test_training.py`
- Create: `docs/reviews/2026-03-11-beta6-bedroom-data-fix-validation.md`

**Step 1: Write the failing regression test**

Add a focused regression proving that the selected exclusion / correction rule is applied only to the identified Bedroom-bad slices and does not alter unrelated rooms or dates.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest backend/tests/test_training.py -q -k bedroom_bad_slice_filter
```

Expected:

- fail before the filter exists

**Step 3: Implement the minimal data fix**

Implement only the explicit correction or exclusion rule supported by Task 5.

**Step 4: Run the regression test and broader suite**

Run:

```bash
pytest backend/tests/test_training.py -q -k bedroom_bad_slice_filter
pytest backend/tests/test_training.py backend/tests/test_registry.py -q
```

Expected:

- pass

**Step 5: Re-run Bedroom validation**

Re-run:

- anchor
- harmful-date variant
- full intended Bedroom pack
- Dec 17 replay benchmark

Expected:

- the harmful failure family disappears without introducing a new dominant error family

### Task 6B: Implement The Robustness Branch If The Harmful Segments Are Valid Regime Shift

**Files:**
- Modify: `backend/ml/training.py`
- Modify: `backend/ml/legacy/registry.py`
- Test: `backend/tests/test_training.py`
- Test: `backend/tests/test_registry.py`
- Create: `docs/reviews/2026-03-11-beta6-bedroom-regime-shift-hardening.md`

**Step 1: Write the failing tests**

Add tests proving that promotion artifacts must include grouped-by-date stability summaries and that a Bedroom candidate with strong pooled metrics but unstable date slices is surfaced as risky.

**Step 2: Run tests to verify they fail**

Run:

```bash
pytest backend/tests/test_training.py -q -k grouped_date_stability
pytest backend/tests/test_registry.py -q -k grouped_date_stability
```

Expected:

- fail because the grouped-by-date summaries do not yet exist

**Step 3: Implement the minimal robustness hardening**

Add:

- grouped-by-date validation summaries for Bedroom
- promotion-time drift summary persistence
- explicit “unstable across date slices” evidence in saved metadata

Do not change non-Bedroom training behavior yet.

**Step 4: Run tests and broader regression coverage**

Run:

```bash
pytest backend/tests/test_training.py -q -k grouped_date_stability
pytest backend/tests/test_registry.py -q -k grouped_date_stability
pytest backend/tests/test_training.py backend/tests/test_registry.py -q
```

Expected:

- pass

**Step 5: Re-run Bedroom validation**

Re-run the candidate that includes the valid alternate regime and confirm the new summaries make the instability explicit and actionable.

### Task 7: Add Lightweight Cross-Room Observability

**Files:**
- Modify: `backend/ml/training.py`
- Test: `backend/tests/test_training.py`
- Create: `docs/reviews/2026-03-11-beta6-room-lineage-observability.md`

**Step 1: Write the failing test**

Add a test that non-Bedroom room traces also carry the lightweight source-manifest and per-date summary fields, without requiring new room-specific gating behavior.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest backend/tests/test_training.py -q -k room_lineage_observability
```

Expected:

- fail before the observability fields are generalized

**Step 3: Implement the minimal generalization**

Persist the observability metadata for all rooms, but keep the new decision logic Bedroom-only unless Bedroom evidence proves wider applicability.

**Step 4: Run the tests**

Run:

```bash
pytest backend/tests/test_training.py -q -k room_lineage_observability
pytest backend/tests/test_training.py backend/tests/test_registry.py -q
```

Expected:

- pass

### Task 8: Final Verification And Decision Note

**Files:**
- Create: `docs/reviews/2026-03-11-beta6-bedroom-root-fix-final.md`
- Modify: `dev_history.log`

**Step 1: Re-read the approved design and checklist the success criteria**

Checklist:

- proved bad data vs valid regime shift
- implemented the matching root fix
- validated Bedroom behavior with fresh evidence
- persisted enough lineage metadata that future drift is auditable

**Step 2: Run the final verification commands**

Run the exact commands used for the chosen branch plus:

```bash
pytest backend/tests/test_training.py backend/tests/test_registry.py -q
```

Expected:

- all touched regression coverage passes

**Step 3: Write the final review note**

The note must state:

- what the root cause actually was
- what was changed
- what remains Bedroom-specific
- what now generalizes across rooms

**Step 4: Update `dev_history.log`**

Add a timestamped entry summarizing:

- actions taken
- files changed
- verification evidence
- final decision on generalization

**Step 5: Commit only after verification evidence exists**

Use small commits aligned to completed tasks, not one large batch.
