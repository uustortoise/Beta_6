# Bedroom Separation Retrain Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run one Bedroom-only corrected-pack retrain that neutralizes Bedroom class-0 oversampling, then replay Dec 17 on the same harness shape used by the canonical Bedroom benchmark.

**Architecture:** Clone the current `HK0011_jessica` namespace into a fresh candidate namespace, run `train_from_files(..., rooms={"Bedroom"}, defer_promotion=True)` with Bedroom-only sampling overrides, activate the selected Bedroom version in the candidate namespace, then evaluate Dec 17 by loading the candidate namespace and generating Bedroom raw/final prediction outputs plus comparison metrics.

**Tech Stack:** Python, UnifiedPipeline `train_from_files`, ModelRegistry rollback, pandas, scikit-learn, JSON/parquet artifacts, Markdown review docs

---

### Task 1: Prepare candidate namespace

**Files:**
- Read: `backend/models/HK0011_jessica/*`
- Create: `backend/models/HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z/`

**Step 1: Clone the current namespace**

Use `rsync -a` to copy `backend/models/HK0011_jessica/` into:

- `backend/models/HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z/`

### Task 2: Run Bedroom-only retrain

**Files:**
- Read: `backend/ml/pipeline.py`
- Read: `backend/ml/training.py`
- Create: `tmp/jessica_bedroom_sepfix_20260311T041155Z/train_metrics.json`

**Step 1: Execute `train_from_files` with Bedroom-only oversampling rollback**

Use corrected pack:

- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_4dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_5dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_6dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_7dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_8dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_9dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_10dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_17dec2025.xlsx`

Set environment overrides only for this run:

- `TRANSITION_FOCUS_MAX_MULTIPLIER_BY_ROOM=bedroom:1`
- `MINORITY_MAX_MULTIPLIER_BY_ROOM=bedroom:1`

Call:

- `UnifiedPipeline(enable_denoising=True).train_from_files(..., elder_id=<candidate>, rooms={"Bedroom"}, defer_promotion=True)`

**Step 2: Persist metrics and selected version**

Write returned metrics to:

- `tmp/jessica_bedroom_sepfix_20260311T041155Z/train_metrics.json`

Extract:

- saved Bedroom version
- gate pass
- holdout macro-F1
- post-sampling class support
- class weights

### Task 3: Activate and replay Dec 17

**Files:**
- Read: `backend/models/HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z/*`
- Create: `tmp/jessica_17dec_eval_candidate_bedroom_sepfix_20260311T041155Z/final/*`

**Step 1: Promote the saved Bedroom version inside the candidate namespace**

Use `ModelRegistry.rollback_to_version(...)` so the candidate namespace serves the newly trained Bedroom version as latest.

**Step 2: Run Dec 17 Bedroom replay evaluation**

Load the candidate namespace, run prediction on the corrected Dec 17 workbook, and produce:

- `raw_predictions/Bedroom.parquet`
- `comparison/Bedroom_merged.parquet`
- `comparison/summary.json`

The summary must include at least:

- Bedroom final macro-F1
- Bedroom raw top-1 macro-F1
- Bedroom raw export macro-F1
- rewrite count from raw top-1
- dominant error pairs

### Task 4: Capture outcome

**Files:**
- Create: `docs/reviews/2026-03-11-beta6-bedroom-separation-retrain.md`
- Modify: `dev_history.log`

**Step 1: Write the result note**

Document:

- the candidate namespace
- the exact Bedroom-only overrides
- holdout result
- Dec 17 Bedroom result vs the `0.3511` canonical reference
- whether `unoccupied -> bedroom_normal_use` improved

**Step 2: Log verification**

Append:

- commands run
- produced artifacts
- verified metrics
- recommendation on whether Bedroom should move to another retrain hypothesis
