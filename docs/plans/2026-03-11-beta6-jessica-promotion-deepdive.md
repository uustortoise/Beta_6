# Beta6 Jessica Promotion And Deep Dive Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Promote the validated Jessica support-fix room artifacts into live `HK0011_jessica` without losing rollback history, then produce a narrow post-promotion LivingRoom deep-dive note.

**Architecture:** Add a small room-wise namespace merge helper under `backend/scripts/` and drive it through the existing registry alias-sync path instead of replacing the whole live namespace. After promotion, verify the live namespace with a fresh load and a Dec 17 replay, then use existing saved artifacts to analyze LivingRoom seed instability.

**Tech Stack:** Python, pytest, existing `ModelRegistry`, Jessica benchmark harness, Markdown review docs

---

### Task 1: Add a failing regression test for room-wise namespace promotion

**Files:**
- Modify: `backend/tests/test_registry.py`
- Read: `backend/ml/legacy/registry.py`

**Step 1: Write the failing test**

Add a test that:
- creates a source namespace and a target namespace in a temp backend tree
- gives the target room an existing promoted version and older rollback history
- gives the source room a newer promoted version plus optional threshold / confidence / two-stage artifacts
- expects the promotion helper to merge the new version into target history and then promote it without dropping the old target entry

**Step 2: Run test to verify it fails**

Run:

```bash
pytest backend/tests/test_registry.py -q -k promote_room_versions_from_namespace
```

Expected: FAIL because no helper exists yet.

**Step 3: Commit**

Do not commit yet.

### Task 2: Implement the room-wise promotion helper

**Files:**
- Create: `backend/scripts/promote_room_versions_from_namespace.py`
- Read: `backend/ml/legacy/registry.py`

**Step 1: Write minimal implementation**

Implement a script that:
- accepts `--source-elder-id`, `--target-elder-id`, repeated `--room`, and optional `--version`
- copies required versioned artifacts from source namespace to target namespace
- merges version metadata with duplicate-version conflict checks
- calls `registry.rollback_to_version()` on target to materialize latest aliases
- emits a JSON summary of copied versions and promoted target versions

**Step 2: Run the failing test again**

Run:

```bash
pytest backend/tests/test_registry.py -q -k promote_room_versions_from_namespace
```

Expected: PASS

**Step 3: Run broader coverage**

Run:

```bash
pytest backend/tests/test_registry.py backend/tests/test_training.py -q
```

Expected: PASS

### Task 3: Promote Jessica live rooms

**Files:**
- Read: `backend/models/HK0011_jessica/*`
- Read: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-bedroom-support-fix/backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z/*`
- Create: `tmp/jessica_live_promotion_20260311T*.json`

**Step 1: Execute the promotion helper**

Run the helper for:
- `Bedroom` -> source current `v38`
- `LivingRoom` -> source current `v40`

Expected: live `HK0011_jessica` keeps its old room history and now points to the promoted candidate versions.

**Step 2: Verify target version files**

Check `Bedroom_versions.json` and `LivingRoom_versions.json` under live `HK0011_jessica`.

Expected:
- Bedroom current version `38`
- LivingRoom current version `40`
- older live versions remain present

### Task 4: Verify the live promotion

**Files:**
- Read: `backend/ml/pipeline.py`
- Read: `tmp/jessica_activity_confidence_benchmark.py`
- Create: `tmp/jessica_17dec_eval_live_20260311T*/final/*`

**Step 1: Run fresh-load sanity**

Run a Python entrypoint that:
- loads `HK0011_jessica` through `UnifiedPipeline`
- records loaded rooms
- records room current versions
- records `two_stage_core_models`

Expected: all five rooms load and the promoted rooms report Bedroom `38`, LivingRoom `40`.

**Step 2: Replay the corrected Dec 17 benchmark through live**

Run the Jessica benchmark harness on live `HK0011_jessica` against:

```text
/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_17dec2025.xlsx
```

Expected: the live replay remains aligned with the support-fix candidate benchmark.

### Task 5: Produce the post-promotion deep dive

**Files:**
- Create: `docs/reviews/2026-03-11-beta6-livingroom-seed-instability-deep-dive.md`
- Read: `backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z/*`
- Read: `tmp/jessica_17dec_eval_candidate_supportfix_20260310T2312Z/*`

**Step 1: Compare the winning and neighboring LivingRoom candidate versions**

Use saved metrics / decision traces to compare the promoted winner against the nearby seed versions.

**Step 2: Write the note**

Summarize:
- what appears stable
- what appears fragile
- whether the next model-side thread should target sampling, calibration geometry, or checkpoint selection

**Step 3: Commit**

Commit the implementation, docs, and any test updates with a promotion-focused message.
