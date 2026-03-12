# Bedroom Two-Stage Runtime Root Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent Bedroom and other two-stage rooms from activating runtime two-stage bundles when training selected single-stage fallback, and repair legacy metadata that omitted that runtime gate state.

**Architecture:** Persist the training-time two-stage runtime decision into `*_two_stage_meta.json`, then teach registry loading to infer and self-heal missing `runtime_enabled` flags from matching decision traces. Verify with regression tests around artifact writing and registry loading so buggy metadata cannot silently reactivate two-stage runtime.

**Tech Stack:** Python, unittest/pytest, JSON artifact metadata, legacy registry/runtime loading

---

### Task 1: Add failing tests for persisted runtime gate metadata

**Files:**
- Modify: `backend/tests/test_training.py`
- Modify: `backend/ml/training.py`

**Step 1: Write the failing test**

Add a regression test that calls `_write_two_stage_core_artifacts(...)` with a two-stage payload containing:
- `runtime_enabled=False`
- `runtime_gate_source="single_stage_fallback_no_regress"`
- `selected_reliable=True`
- `fail_closed=False`

Assert the written versioned/latest meta JSON includes those fields.

**Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_training.py -q -k two_stage_runtime_metadata`
Expected: FAIL because `runtime_enabled` and related fields are not written today.

**Step 3: Write minimal implementation**

Update the two-stage artifact writer to include runtime gate fields in meta payloads.

**Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_training.py -q -k two_stage_runtime_metadata`
Expected: PASS

### Task 2: Add failing test for legacy metadata self-heal on load

**Files:**
- Modify: `backend/tests/test_registry.py`
- Modify: `backend/ml/legacy/registry.py`

**Step 1: Write the failing test**

Add a registry regression test with:
- latest/versioned two-stage meta missing `runtime_enabled`
- matching decision trace showing `metrics.two_stage_core.runtime_use_two_stage=false`

Assert `load_models_for_elder(...)` does not activate `platform.two_stage_core_models[room]` and that the meta file is repaired with `runtime_enabled=false`.

**Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_registry.py -q -k runtime_disabled_two_stage`
Expected: FAIL because the current loader defaults missing `runtime_enabled` to `True`.

**Step 3: Write minimal implementation**

Add a registry helper that infers runtime gate fields from the matching decision trace when two-stage meta omitted them, then writes the repaired metadata back before load continues.

**Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_registry.py -q -k runtime_disabled_two_stage`
Expected: PASS

### Task 3: Verify end-to-end regression coverage

**Files:**
- Modify: `backend/tests/test_training.py`
- Modify: `backend/tests/test_registry.py`

**Step 1: Run focused suites**

Run:
- `pytest backend/tests/test_training.py -q`
- `pytest backend/tests/test_registry.py -q`

Expected: PASS

**Step 2: Run combined root-fix verification**

Run:
- `pytest backend/tests/test_training.py backend/tests/test_registry.py -q`

Expected: PASS

### Task 4: Document the root fix

**Files:**
- Create: `docs/reviews/2026-03-11-beta6-bedroom-two-stage-runtime-root-fix.md`
- Modify: `dev_history.log`

**Step 1: Write review note**

Document:
- root cause: runtime gate decision not persisted
- evidence: decision trace said single-stage fallback while runtime loaded two-stage
- fix: persist + self-heal metadata
- verification commands and results

**Step 2: Append dev history**

Log the implementation, tests, and decision rationale.
