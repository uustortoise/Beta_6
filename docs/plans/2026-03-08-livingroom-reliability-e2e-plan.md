# Livingroom Reliability End-to-End Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix Livingroom model selection reliability and add truthful Streamlit review/metric surfaces end-to-end.

**Architecture:** Keep model-selection changes inside `backend/ml/training.py` and drive them with focused unit tests in `backend/tests/test_training.py`. Keep UI compare-mode and truthful metric labeling in the existing service/page modules so review operations and dashboard monitoring stay in their current flows.

**Tech Stack:** Python, Streamlit, Altair, pandas, pytest

---

### Task 1: Harden two-stage final path selection

**Files:**
- Modify: `backend/ml/training.py`
- Test: `backend/tests/test_training.py`

**Step 1: Write the failing tests**

Add tests that assert:
- a non-collapsed two-stage candidate beats a higher-score collapsed single-stage candidate
- a path that passes no-regress beats a path that fails it when collapse status ties
- selection metadata records a fail-closed source when both paths are unreliable

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_training.py -q -k "two_stage_gate_source and (collapsed or no_regress or fail_closed)"`
Expected: FAIL because current selection still permits score-led fallback.

**Step 3: Write minimal implementation**

Update `_select_final_two_stage_gate_source` in `backend/ml/training.py` so ranking is:
- non-collapsed
- passes hard reliability checks
- gate-aligned score
- macro F1
- two-stage tiebreak

Record explicit fail-closed metadata if neither path is reliable.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_training.py -q -k "two_stage_gate_source and (collapsed or no_regress or fail_closed)"`
Expected: PASS

### Task 2: Harden multi-seed panel selection and artifact reuse

**Files:**
- Modify: `backend/ml/training.py`
- Test: `backend/tests/test_training.py`

**Step 1: Write the failing tests**

Add tests that assert:
- a gate-passing non-collapsed seed beats a gate-failing seed with higher score
- the multi-seed panel returns the evaluated winning metrics instead of retraining the winner from scratch

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_training.py -q -k "multi_seed and (gate_passing or reuses_selected_candidate)"`
Expected: FAIL because current ranking soft-penalizes gate failure and retrains the winner.

**Step 3: Write minimal implementation**

Update `_select_multi_seed_candidate` and `_train_room_with_multi_seed_panel` so:
- ranking prefers non-collapsed, gate-passing, no-regress-safe candidates
- the selected candidate carries its evaluated metrics and seed-panel debug metadata to the final result

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_training.py -q -k "multi_seed and (gate_passing or reuses_selected_candidate)"`
Expected: PASS

### Task 3: Make the sample collection target configurable

**Files:**
- Modify: `backend/services/ops_service.py`
- Test: `backend/tests/test_ui_services.py`

**Step 1: Write the failing test**

Add a test that sets `UI_SAMPLE_COLLECTION_TARGET_DAYS=14`, seeds enough `adl_history` days, and asserts `get_sample_collection_status(...)[\"target\"] == 14`.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_ui_services.py -q -k sample_collection_target`
Expected: FAIL because the service still hardcodes `21`.

**Step 3: Write minimal implementation**

Update `backend/services/ops_service.py` to read `UI_SAMPLE_COLLECTION_TARGET_DAYS`, defaulting to `14`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_ui_services.py -q -k sample_collection_target`
Expected: PASS

### Task 4: Add training timeline retrieval and compare annotations

**Files:**
- Modify: `backend/services/correction_service.py`
- Test: `backend/tests/test_ui_services.py`

**Step 1: Write the failing tests**

Add tests that assert:
- a training/corrected timeline helper returns normalized blocks for one resident-room-day
- a compare helper annotates prediction blocks with `is_unknown`, `is_low_confidence`, `is_mismatch`, and `is_corrected`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_ui_services.py -q -k "training_timeline or compare_annotations"`
Expected: FAIL because the helpers do not exist.

**Step 3: Write minimal implementation**

Add helpers in `backend/services/correction_service.py` to load the training timeline and annotate prediction blocks by time overlap.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_ui_services.py -q -k "training_timeline or compare_annotations"`
Expected: PASS

### Task 5: Render stacked compare timelines in Correction Studio

**Files:**
- Modify: `backend/app/pages/1_correction_studio.py`
- Test: `backend/tests/test_ui_services.py`

**Step 1: Write the failing test**

Add a lightweight helper/page test that verifies compare-mode data preparation returns two timelines plus explicit uncertainty labels for a selected day.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_ui_services.py -q -k "correction_compare_mode or correction_runtime_labels"`
Expected: FAIL because the page does not expose compare-mode behavior.

**Step 3: Write minimal implementation**

Update `backend/app/pages/1_correction_studio.py` to render:
- `Training / Corrected Timeline`
- `Prediction / Runtime Timeline`
- explicit captions/tooltips for unknown and low-confidence output

Preserve the current queue and correction flow.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_ui_services.py -q -k "correction_compare_mode or correction_runtime_labels"`
Expected: PASS

### Task 6: Make dashboard metrics source-aware and readiness truthful

**Files:**
- Modify: `backend/services/ops_service.py`
- Modify: `backend/app/pages/4_ops_dashboard.py`
- Modify: `backend/export_dashboard.py`
- Test: `backend/tests/test_ui_services.py`

**Step 1: Write the failing tests**

Add tests that assert:
- collection/readiness payloads distinguish labeled days, target days, and ready rooms
- model update monitor payloads preserve walk-forward metric fields instead of collapsing them into generic `accuracy`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_ui_services.py -q -k "sample_collection_status or model_update_monitor"`
Expected: FAIL because labels/source semantics are still ambiguous.

**Step 3: Write minimal implementation**

Update the dashboard/service code so:
- the 14-day target is shown explicitly
- cards and charts say `WF Candidate F1`, `WF Candidate Accuracy`, `Champion WF F1`, `Stability Score`, and `Transition Score`
- captions distinguish walk-forward metrics from raw run scores

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_ui_services.py -q -k "sample_collection_status or model_update_monitor"`
Expected: PASS

### Task 7: Run end-to-end targeted verification

**Files:**
- No code changes required unless verification exposes a missed gap

**Step 1: Run the targeted training suite**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_training.py -q -k "two_stage_gate_source or multi_seed or low_support_reasons_go_to_watch_list or pilot_profile_skips_low_support_room_threshold_block"`
Expected: PASS

**Step 2: Run the UI service suite**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_ui_services.py -q`
Expected: PASS

**Step 3: If both pass, inspect git diff and summarize the work**

Run: `git status --short`
Expected: only intended files are modified.
