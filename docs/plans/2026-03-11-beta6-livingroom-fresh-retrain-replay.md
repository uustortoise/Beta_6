# Beta6 LivingRoom Fresh Retrain Replay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run one fresh LivingRoom-only retrain on the corrected Jessica pack in an isolated candidate namespace, then replay Dec 17 to verify the selector now chooses a v46-like seed automatically.

**Architecture:** Clone the current live Jessica namespace into a new candidate namespace under this worktree, retrain only `LivingRoom` via `train_from_files(..., rooms={"LivingRoom"}, defer_promotion=True)` against the corrected Dec 4-10 plus Dec 17 workbook pack, and inspect the saved decision-trace metadata to identify the selected seed/version. Reuse the existing Dec 17 replay output shape so the new candidate can be compared directly to the validated live root-fix benchmark.

**Tech Stack:** Python, UnifiedPipeline `train_from_files`, model registry artifacts, JSON decision traces, parquet replay outputs, Markdown review docs

---

### Task 1: Prepare isolated execution state

**Files:**
- Read: `backend/models/HK0011_jessica/*`
- Create: `backend/models/HK0011_jessica_candidate_livingroom_fresh_20260311T*.*/`
- Create: `dev_history.log`

**Step 1: Clone the live namespace**

Run `rsync -a` from `backend/models/HK0011_jessica/` into a fresh `HK0011_jessica_candidate_livingroom_fresh_<timestamp>/`.

**Step 2: Record the run intent**

Append a timestamped entry to `dev_history.log` describing:
- the branch/worktree
- candidate namespace
- corrected pack scope
- expected verification outputs

### Task 2: Run the fresh LivingRoom-only retrain

**Files:**
- Read: `backend/ml/pipeline.py`
- Read: `backend/ml/training.py`
- Create: `tmp/jessica_livingroom_fresh_<timestamp>/train_metrics.json`

**Step 1: Execute `train_from_files` on the corrected pack**

Use the corrected workbook pack:
- `HK0011_jessica_train_4dec2025.xlsx`
- `HK0011_jessica_train_5dec2025.xlsx`
- `HK0011_jessica_train_6dec2025.xlsx`
- `HK0011_jessica_train_7dec2025.xlsx`
- `HK0011_jessica_train_8dec2025.xlsx`
- `HK0011_jessica_train_9dec2025.xlsx`
- `HK0011_jessica_train_10dec2025.xlsx`
- `HK0011_jessica_train_17dec2025.xlsx`

Call `UnifiedPipeline(enable_denoising=True).train_from_files(...)` with:
- `elder_id=<candidate namespace>`
- `rooms={"LivingRoom"}`
- `defer_promotion=True`

Persist the returned metrics to `tmp/.../train_metrics.json`.

**Step 2: Inspect selector evidence**

From the returned metrics and any new `LivingRoom_v*_decision_trace.json` files, extract:
- selected saved version
- selected seed
- selection mode
- `stage_a_occupancy_saturated`
- `stage_a_occupancy_rate_error`
- holdout macro-F1

### Task 3: Replay Dec 17 on the fresh candidate

**Files:**
- Read: `backend/models/<candidate namespace>/*`
- Read: `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_17dec2025.xlsx`
- Create: `tmp/jessica_17dec_eval_<candidate namespace>/final/*`

**Step 1: Run the Dec 17 replay harness**

Generate:
- `raw_predictions/*.parquet`
- `comparison/*_merged.parquet`
- `comparison/summary.json`

The replay output must be directly comparable to:
- `tmp/jessica_17dec_eval_live_livingroom_rootfix_registryfix_20260311/final/comparison/summary.json`

**Step 2: Compare against the validated live/root-fix result**

Focus on:
- overall final macro-F1
- LivingRoom final macro-F1
- Bedroom final macro-F1
- whether LivingRoom remains near the `0.4468` live/root-fix reference

### Task 4: Capture the outcome

**Files:**
- Modify: `dev_history.log`
- Optionally create: `docs/reviews/2026-03-11-beta6-livingroom-fresh-retrain-replay.md`

**Step 1: Log the results**

Append the commands run, selected candidate details, replay metrics, and any blocker/root-cause notes.

**Step 2: Summarize merge-readiness**

State whether the fresh retrain confirms the selector behavior strongly enough to move from "validated checkpoint" toward merge-ready.
