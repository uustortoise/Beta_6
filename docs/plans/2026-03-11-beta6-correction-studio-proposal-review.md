# Correction Studio Proposal Review Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a DB-backed proposal review workflow to Correction Studio so ops can review AI-suggested label corrections at both segment and timestamp granularity, keep a full audit trail, export review/apply-ready artifacts, and apply approved rows through the existing correction pipeline.

**Architecture:** Add a dedicated proposal-review service that owns batch staging tables, proposal rows, decision logs, timestamp-resolution logic, and apply-ready exports. Keep Streamlit focused on ingestion, review, and action controls by calling that service from a new `Proposed Label Review` section inside Correction Studio for training-file workflows.

**Tech Stack:** Streamlit, pandas, SQLite/PostgreSQL-compatible SQL, existing correction services, pytest.

---

### Task 1: Add the failing proposal-review service tests

**Files:**
- Create: `backend/tests/test_label_proposal_service.py`
- Reference: `backend/tests/conftest.py`
- Reference: `backend/tests/test_ui_services.py`

**Step 1: Write the failing tests**

Add tests for:
- creating a proposal batch with mixed segment/timestamp rows and parent-child linking
- approving/rejecting proposals and persisting decision-log history
- resolving approved segment proposals into exact timestamp corrections while honoring timestamp overrides/rejections
- applying approved proposals into `adl_history` and `correction_history`, then marking proposal rows as `applied`

**Step 2: Run the focused test file to verify it fails**

Run: `pytest backend/tests/test_label_proposal_service.py -q`

Expected: import or attribute failures because the proposal-review service does not exist yet.

### Task 2: Implement the proposal-review service

**Files:**
- Create: `backend/services/label_proposal_service.py`
- Modify: `backend/elderlycare_v1_16/models/schema.sql`

**Step 1: Write the minimal service implementation**

Add:
- lazy `CREATE TABLE IF NOT EXISTS` helpers for proposal batches, proposal items, and decision logs
- batch creation/import from a normalized DataFrame
- list/fetch helpers for batches, rows, and decision logs
- decision update helpers for segment/timestamp review actions
- exact timestamp resolution for pending review exports and approved apply-ready exports
- approved-row application that delegates final data mutation to the existing correction service path and records proposal audit state

**Step 2: Run the focused test file to verify it passes**

Run: `pytest backend/tests/test_label_proposal_service.py -q`

Expected: all proposal-review service tests pass.

### Task 3: Wire proposal review into Correction Studio

**Files:**
- Modify: `backend/export_dashboard.py`

**Step 1: Add UI helpers**

Add small helpers for:
- parsing uploaded CSV/JSON proposal packs into normalized rows
- rendering proposal overlays and batch summaries
- packaging review/apply-ready exports for download buttons

**Step 2: Add the review workflow**

Inside the training-file Correction Studio flow:
- add a `Proposed Label Review` section below the existing timeline/manual grid
- allow importing a proposal batch into the DB
- show batch filters, summary metrics, and a proposal overlay on the activity timeline
- show segment review and timestamp review tables with approve/reject actions
- show decision logs and download buttons for the ops review CSV, proposed timestamp-set CSV, and apply-ready JSON
- add `Apply Approved Corrections` that routes through the service and then refreshes the page state

**Step 3: Run targeted UI-adjacent tests**

Run: `pytest backend/tests/test_label_proposal_service.py backend/tests/test_ui_services.py -q`

Expected: all targeted tests pass with no regressions in existing correction-service behavior.

### Task 4: Verify and document the finished workflow

**Files:**
- Modify: `dev_history.log`

**Step 1: Run final verification**

Run: `pytest backend/tests/test_label_proposal_service.py backend/tests/test_ui_services.py -q`

Expected: green test run.

**Step 2: Record development history**

Append start/completion entries to `dev_history.log` with the files changed, verification commands, and the approved architecture decision.
