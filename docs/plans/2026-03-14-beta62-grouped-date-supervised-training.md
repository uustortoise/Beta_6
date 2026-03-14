# Beta6.2 Grouped-Date Supervised Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a real Beta6.2 grouped-date supervised training/evaluation path that consumes explicit day-segment manifests and never flattens discontinuous Dec+Mar data into one synthetic timeline.

**Architecture:** Implement reusable grouped-date supervised logic in `backend/ml/beta6/grouped_date_supervised.py` and expose it through a thin CLI at `backend/scripts/run_beta62_grouped_date_supervised.py`. Reuse the grouped-date manifest/day-loading patterns from the diagnostic path, but keep supervised split assignment explicit per segment so train/val/calib/eval remain date-safe and interpretable.

**Tech Stack:** Python 3, pandas, numpy, pytest, existing Beta6 evaluation helpers, existing workbook/parquet room loading conventions.

---

### Task 1: Lock the grouped-date supervised contract with failing tests

**Files:**
- Create: `backend/tests/test_beta62_grouped_date_supervised.py`

**Step 1: Write the failing tests**

Cover:

- discontinuous Dec->Mar segments remain separate through supervised prep
- late March candidate segments assigned to `train` remain in train instead of being pushed out by chronology
- mixed `.parquet` baseline and `.xlsx` candidate segments load correctly
- manifest lineage survives into emitted supervised artifacts
- grouped-date evaluation output is emitted after supervised prep/execution

**Step 2: Run the focused test file to verify RED**

Run: `pytest backend/tests/test_beta62_grouped_date_supervised.py -q`

Expected: failures because the supervised module/CLI do not exist yet.

### Task 2: Add the reusable grouped-date supervised module

**Files:**
- Create: `backend/ml/beta6/grouped_date_supervised.py`

**Step 1: Add manifest parser/validator**

Implement:

- schema parsing
- required top-level fields (`schema_version`, `resident_id`, `target_rooms`, `segments`)
- required segment fields (`role`, `date`, `path`, `split`)
- deterministic normalization preserving segment order

**Step 2: Reuse workbook/parquet day loading**

Implement minimal shared loading:

- room-sheet loading for `.xlsx`
- room-filter loading for `.parquet`
- per-segment metadata extraction
- no global concatenation across segment boundaries at the manifest layer

**Step 3: Add grouped supervised prep/eval artifact builder**

Return a JSON-ready payload including:

- manifest lineage
- per-room split summaries by date
- per-room grouped evaluation slices by date
- explicit segment counts per split

### Task 3: Add the thin CLI entrypoint

**Files:**
- Create: `backend/scripts/run_beta62_grouped_date_supervised.py`

**Step 1: Parse manifest/output arguments**

**Step 2: Call the reusable module**

**Step 3: Write the machine-readable supervised artifact**

### Task 4: Re-run tests and fix only failing behavior

**Files:**
- Modify: `backend/tests/test_beta62_grouped_date_supervised.py`
- Modify: `backend/ml/beta6/grouped_date_supervised.py`
- Modify: `backend/scripts/run_beta62_grouped_date_supervised.py`

**Step 1: Run focused tests**

Run: `pytest backend/tests/test_beta62_grouped_date_supervised.py -q`

**Step 2: Tighten implementation only where tests fail**

### Task 5: Add review/log documentation

**Files:**
- Create: `docs/reviews/2026-03-14-beta62-grouped-date-supervised-path.md`
- Modify: `/Users/dickson/DT/DT_development/Development/Beta_6/dev_history.log`

**Step 1: Document the permanent grouped-date supervised path**

Include:

- manifest contract
- explicit split semantics
- why it is date-safe
- what it still does not do yet (no March retrain in this thread)

**Step 2: Append the dev history entry**

### Task 6: Final verification

**Files:**
- No code changes expected

**Step 1: Run compile checks**

Run:

- `python3 -m py_compile backend/ml/beta6/grouped_date_supervised.py`
- `python3 -m py_compile backend/scripts/run_beta62_grouped_date_supervised.py`

**Step 2: Run focused pytest**

Run:

- `pytest backend/tests/test_beta62_grouped_date_supervised.py -q`

**Step 3: Run a broader regression slice**

Run:

- `pytest backend/tests/test_run_room_experiments.py backend/tests/test_beta6_trainer_intake_gate.py -q`

**Step 4: Confirm doc/log references**

Run:

- `rg -n "grouped-date supervised|TASK-|run_beta62_grouped_date_supervised" docs/reviews/2026-03-14-beta62-grouped-date-supervised-path.md /Users/dickson/DT/DT_development/Development/Beta_6/dev_history.log`
