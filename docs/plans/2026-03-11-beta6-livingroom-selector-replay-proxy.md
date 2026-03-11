# Beta6 LivingRoom Selector Replay-Proxy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve LivingRoom seed selection so future retrains prefer replay-stable candidates like `v46` over higher-holdout-score seeds that still show stage-A occupancy distortion.

**Architecture:** Extend the multi-seed candidate summary with a LivingRoom-only stage-A occupancy-rate error derived from `metrics["two_stage_core"]["stage_a_calibration"]`, then rank candidates on that signal before raw holdout score. Verify with red-green selector tests and by recomputing the selector over the saved `v44/v45/v46` traces.

**Tech Stack:** Python, pytest, JSON decision traces, training pipeline selector logic, Markdown planning docs

---

### Task 1: Add failing selector tests

**Files:**
- Modify: `backend/tests/test_training.py`
- Read: `backend/ml/training.py`

**Step 1: Add a failing summary test**

Write a unit test that passes LivingRoom metrics with:
- `two_stage_core.stage_a_calibration.true_occupied_rate`
- `two_stage_core.stage_a_calibration.predicted_occupied_rate`
- a non-saturated but nontrivial gap

Assert the summary exposes:
- `stage_a_occupancy_rate_error`
- `stage_a_occupancy_saturated`

**Step 2: Add a failing selector-ranking test**

Create two LivingRoom candidates where both:
- are not collapsed
- pass gates
- pass no-regress
- are not saturated

But one candidate has:
- higher `gate_aligned_score`
- much worse `stage_a_occupancy_rate_error`

Assert the selector chooses the lower-error candidate.

**Step 3: Verify RED**

Run:

```bash
pytest backend/tests/test_training.py -q -k "occupancy_rate_error or non_saturated_stage_a_seed"
```

Expected: FAIL because the current selector ignores the new error signal.

### Task 2: Implement the minimal selector replay-proxy

**Files:**
- Modify: `backend/ml/training.py`
- Modify: `backend/tests/test_training.py`

**Step 1: Add a focused helper**

Extract LivingRoom stage-A occupancy alignment from candidate metrics:
- fallback/saturation status
- absolute error between predicted and true occupied rates

Return `None` when the required calibration fields are absent.

**Step 2: Attach the signal to candidate summaries**

Extend `_build_seed_panel_candidate_summary(...)` to include:
- `stage_a_occupancy_rate_error`

Keep existing fields and semantics unchanged.

**Step 3: Update selector ordering**

Extend `_select_multi_seed_candidate(...)` so after:
- non-collapsed
- gate pass
- no-regress
- non-saturated

it prefers:
- lower `stage_a_occupancy_rate_error`

before falling back to:
- `gate_aligned_score`
- `macro_f1`
- seed order

### Task 3: Verify on saved Jessica artifacts

**Files:**
- Create: `tmp/jessica_livingroom_rootfix_pack_20260311T004042Z/selector_replay_proxy_summary.json`
- Read: `backend/models/HK0011_jessica_candidate_livingroom_rootfix_pack_20260311T004042Z/LivingRoom_v44_decision_trace.json`
- Read: `backend/models/HK0011_jessica_candidate_livingroom_rootfix_pack_20260311T004042Z/LivingRoom_v45_decision_trace.json`
- Read: `backend/models/HK0011_jessica_candidate_livingroom_rootfix_pack_20260311T004042Z/LivingRoom_v46_decision_trace.json`

**Step 1: Run focused tests**

Run:

```bash
pytest backend/tests/test_training.py -q -k "occupancy_rate_error or multi_seed_candidate or seed_panel"
```

Expected: PASS

**Step 2: Recompute selector choice over saved traces**

Build a small one-off artifact using the current training code so the selector re-evaluates the saved `v44/v45/v46` metrics and records:
- summary fields for each candidate
- selected version
- selected seed

Expected: `v46` / seed `42` remains selected.
