# Beta6 Jessica LivingRoom Root Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Permanently reduce LivingRoom seed instability by stabilizing the training path itself and by rejecting LivingRoom multi-seed panels that are still only carried by a single lucky winner.

**Architecture:** Update the default training policy so LivingRoom receives post-split batch shuffling, extend multi-seed training summaries with LivingRoom-specific stage-A collapse and stability accounting, then rerun a LivingRoom-only seed panel to verify the recipe is materially more stable. Keep runtime selection logic reliability-first, but add room-level panel validation so unstable LivingRoom winners are not treated as promotion-grade.

**Tech Stack:** Python, pytest, YAML-backed policy defaults, training pipeline instrumentation, JSON model artifacts, Markdown review docs

---

### Task 1: Record the implementation plan and local history

**Files:**
- Create: `docs/plans/2026-03-11-beta6-jessica-livingroom-root-fix.md`
- Modify: `dev_history.log`

**Step 1: Write the implementation plan**

Capture:
- exact policy and training files to modify
- the failing tests to add first
- the LivingRoom-only rerun and replay commands
- the review/doc outputs needed for merge-ready handoff

**Step 2: Extend local development history**

Record:
- the committed design-doc baseline
- the planned root-fix sequence
- the intended verification criteria

**Step 3: Do not start code changes until the plan is saved**

This preserves the brainstorming hard gate and gives the branch a documented execution contract.

### Task 2: Write the failing tests for the policy and panel-stability contract

**Files:**
- Modify: `backend/tests/test_policy_config.py`
- Modify: `backend/tests/test_training.py`
- Read: `backend/config/beta6_policy_defaults.yaml`
- Read: `backend/ml/training.py`

**Step 1: Add a failing policy-default test**

Extend the default-policy assertions so `policy.training_profile.post_split_shuffle_rooms` is expected to equal:

```python
["entrance", "bedroom", "livingroom"]
```

**Step 2: Add a failing LivingRoom panel-stability test**

Add a focused unit test around multi-seed training that feeds candidate metrics where:
- one seed passes `no_regress_ok`
- multiple other seeds report `stage_a_calibration.status == "fallback_recall_floor"`
- those same seeds report `predicted_occupied_rate == 1.0`

Assert that the resulting `seed_panel_debug` includes:
- the correct `seed_panel_no_regress_pass_count`
- the correct `seed_panel_stage_a_collapse_count`
- `seed_panel_is_stable is False`

**Step 3: Add a failing stability-positive control**

Add a second unit test where at least two LivingRoom seeds pass cleanly without stage-A collapse, and assert:
- the selected winner is still chosen by the existing ranking
- `seed_panel_is_stable is True`

**Step 4: Run the targeted tests to verify RED**

Run:

```bash
pytest backend/tests/test_policy_config.py backend/tests/test_training.py -q -k "post_split_shuffle_rooms or seed_panel"
```

Expected: FAIL because the default policy and seed-panel debug output do not yet expose the new LivingRoom stability behavior.

### Task 3: Implement the minimal policy change

**Files:**
- Modify: `backend/config/beta6_policy_defaults.yaml`
- Modify: `backend/ml/policy_defaults.py` only if required by parsing behavior

**Step 1: Add LivingRoom to the default post-split shuffle list**

Update the YAML defaults:

```yaml
post_split_shuffle_rooms:
  - entrance
  - bedroom
  - livingroom
```

**Step 2: Keep parsing behavior unchanged unless tests force a code change**

The existing list loader should already pick up the new YAML token set. Do not modify Python parsing code unless the tests prove it is necessary.

**Step 3: Re-run the targeted policy test**

Run:

```bash
pytest backend/tests/test_policy_config.py -q -k post_split_shuffle_rooms
```

Expected: PASS

### Task 4: Implement LivingRoom stability accounting in multi-seed training

**Files:**
- Modify: `backend/ml/training.py`
- Modify: `backend/tests/test_training.py`

**Step 1: Add a helper that detects LivingRoom stage-A collapse**

In `backend/ml/training.py`, add a focused helper that inspects candidate metrics and returns `True` when:
- the room is `LivingRoom`
- `metrics["two_stage_core"]["stage_a_calibration"]["status"]` is `fallback_recall_floor` or another explicit fallback status
- `predicted_occupied_rate` is saturated at or near the fully occupied regime

Keep this helper narrow and artifact-backed; do not guess from macro-F1 alone.

**Step 2: Add a helper that summarizes seed-panel stability**

Build a small summary helper that computes:
- `seed_panel_no_regress_pass_count`
- `seed_panel_stage_a_collapse_count`
- `seed_panel_candidate_count`
- `seed_panel_is_stable`

Use a room-sensitive contract:
- for `LivingRoom`, require more than one no-regress-safe seed and no stage-A occupancy-collapse majority
- for other rooms, leave behavior effectively unchanged unless the summary is purely additive metadata

**Step 3: Attach the new summary to `seed_panel_debug`**

Extend `_train_room_with_multi_seed_panel(...)` so the returned metrics include the new fields inside `seed_panel_debug`, while preserving:
- candidate summaries
- selected seed
- existing artifact reuse behavior

**Step 4: Keep winner selection logic intact**

Do not replace `_select_multi_seed_candidate(...)` unless the new tests require it. The goal is to preserve the ranking, then add the missing room-level stability visibility.

**Step 5: Run the targeted training tests**

Run:

```bash
pytest backend/tests/test_training.py -q -k "seed_panel or multi_seed_candidate"
```

Expected: PASS

### Task 5: Verify the recipe path is actually active for LivingRoom

**Files:**
- Modify: `backend/tests/test_training.py`
- Read: `backend/ml/training.py`

**Step 1: Add or extend a focused shuffle-path test if coverage is still missing**

If current tests do not prove it, add a small test that confirms `TrainingProfilePolicy.should_shuffle_post_split("LivingRoom")` is true under default policy and that the training fit path receives `shuffle=True` for a LivingRoom training call where post-split shuffle is enabled.

**Step 2: Run the focused shuffle verification**

Run:

```bash
pytest backend/tests/test_policy_config.py backend/tests/test_training.py -q -k "livingroom and shuffle"
```

Expected: PASS

### Task 6: Run a LivingRoom-only seed-panel retrain with the new recipe

**Files:**
- Read: `backend/models/HK0011_jessica/*`
- Create: `tmp/jessica_livingroom_rootfix_<timestamp>/...`

**Step 1: Execute the LivingRoom-only retrain**

Use the corrected Jessica workbook and keep the run scoped to `LivingRoom`. Preserve the no-downsample / multi-seed context so the comparison stays honest.

**Step 2: Inspect the resulting seed-panel artifacts**

Confirm:
- more than one seed can clear the no-regress floor, or
- the prior failure mode is materially reduced and explicitly observable through the new stability metadata

**Step 3: If instability persists, stop at evidence**

Do not pile on threshold or runtime changes in the same thread. If the rerun still collapses in stage-A, document that the next root-cause layer is inside the two-stage stage-A training path.

### Task 7: Publish the root-fix review note

**Files:**
- Create: `docs/reviews/2026-03-11-beta6-jessica-livingroom-root-fix.md`
- Modify: `dev_history.log`
- Read: `tmp/jessica_livingroom_rootfix_<timestamp>/...`

**Step 1: Write the review note**

Summarize:
- what changed in policy/training
- whether LivingRoom stability improved across the seed panel
- whether the fix is sufficient to promote as the new default recipe
- what remains unresolved, if anything

**Step 2: Update local development history**

Record:
- the test evidence
- the retrain artifact path
- the final decision on whether the root fix is complete or if deeper stage-A work remains

### Task 8: Run final verification and commit

**Files:**
- Modify: any files required by the implementation above

**Step 1: Run focused regression coverage**

Run:

```bash
pytest backend/tests/test_policy_config.py backend/tests/test_training.py backend/tests/test_registry.py backend/tests/test_promote_room_versions_from_namespace.py backend/tests/test_livingroom_seed_forensic.py -q
```

Expected: PASS

**Step 2: Inspect branch status**

Run:

```bash
git status --short
```

Expected: only intended policy, training, test, and doc changes are present.

**Step 3: Commit the merge-ready implementation**

Run:

```bash
git add backend/config/beta6_policy_defaults.yaml backend/ml/training.py backend/tests/test_policy_config.py backend/tests/test_training.py docs/plans/2026-03-11-beta6-jessica-livingroom-root-fix.md docs/reviews/2026-03-11-beta6-jessica-livingroom-root-fix.md
git commit -m "fix: stabilize livingroom multi-seed training"
```
