# Beta6 Jessica LivingRoom Seed Forensic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible LivingRoom-only forensic that explains why `LivingRoom_v40` alone clears the no-regress checkpoint floor across Jessica seed panel versions `v39..v43`.

**Architecture:** Add a small analysis helper under `backend/scripts/` that reads the saved LivingRoom decision traces, versions metadata, and activity-confidence calibrators, then emits a JSON comparison artifact. Drive it with a focused test on synthetic fixtures and publish the conclusions in a review note.

**Tech Stack:** Python, pytest, JSON artifact analysis, Markdown review docs

---

### Task 1: Record the forensic setup and branch-local history

**Files:**
- Create: `dev_history.log`
- Create: `docs/plans/2026-03-11-beta6-jessica-livingroom-seed-forensic-design.md`
- Create: `docs/plans/2026-03-11-beta6-jessica-livingroom-seed-forensic.md`

**Step 1: Write the design and plan docs**

Capture:
- scope boundaries
- selected approach
- success criteria
- exact files and commands for the forensic

**Step 2: Initialize development history**

Create a branch-local `dev_history.log` and record:
- worktree creation
- live-model namespace sync
- design / plan approval

**Step 3: Do not commit yet**

Keep the branch open until code, docs, and verification are complete.

### Task 2: Write the failing regression test for the forensic helper

**Files:**
- Create: `backend/tests/test_livingroom_seed_forensic.py`
- Read: `backend/models/HK0011_jessica/LivingRoom_v40_decision_trace.json`

**Step 1: Write the failing test**

Add tests that construct synthetic fixture files for a winner and a fallback loser and assert that the helper:
- extracts the random seed from the saved policy
- marks the winner as reaching the no-regress floor
- marks the loser as fallback-selected
- computes winner-relative threshold and calibrator deltas

**Step 2: Run test to verify it fails**

Run:

```bash
pytest backend/tests/test_livingroom_seed_forensic.py -q
```

Expected: FAIL because the forensic helper does not exist yet.

### Task 3: Implement the minimal forensic helper

**Files:**
- Create: `backend/scripts/livingroom_seed_forensic.py`
- Read: `backend/models/HK0011_jessica/LivingRoom_versions.json`

**Step 1: Write minimal implementation**

Implement helpers that:
- load version metadata, decision traces, and calibrators
- normalize per-version summaries
- compare all versions against a declared winner version
- write a JSON artifact from a CLI entrypoint

**Step 2: Run the new test**

Run:

```bash
pytest backend/tests/test_livingroom_seed_forensic.py -q
```

Expected: PASS

### Task 4: Generate the real Jessica LivingRoom forensic artifact

**Files:**
- Read: `backend/models/HK0011_jessica/LivingRoom_versions.json`
- Read: `backend/models/HK0011_jessica/LivingRoom_v39_decision_trace.json`
- Read: `backend/models/HK0011_jessica/LivingRoom_v40_decision_trace.json`
- Read: `backend/models/HK0011_jessica/LivingRoom_v41_decision_trace.json`
- Read: `backend/models/HK0011_jessica/LivingRoom_v42_decision_trace.json`
- Read: `backend/models/HK0011_jessica/LivingRoom_v43_decision_trace.json`
- Create: `tmp/jessica_livingroom_seed_forensic_20260311T*.json`

**Step 1: Run the helper against the synced live namespace**

Run the helper for:
- elder `HK0011_jessica`
- room `LivingRoom`
- versions `39 40 41 42 43`
- winner `40`

**Step 2: Inspect the output**

Confirm the JSON reflects the already-known high-level facts:
- `v40` reaches `no_regress_floor`
- `v39`, `v41`, `v42`, `v43` fall back
- post-sampling prior drift stays approximately constant across all five versions

### Task 5: Publish the review note

**Files:**
- Create: `docs/reviews/2026-03-11-beta6-jessica-livingroom-seed-forensic.md`
- Read: `tmp/jessica_livingroom_seed_forensic_20260311T*.json`

**Step 1: Write the review note**

Summarize:
- what stayed constant across the panel
- what diverged at checkpoint-selection time
- why threshold and confidence artifacts are reflections rather than the root cause
- what the next honest thread should target

**Step 2: Update development history**

Record the artifact path, review conclusion, and recommended next thread.

### Task 6: Run targeted verification

**Files:**
- No additional files required unless verification exposes a gap

**Step 1: Run the new forensic test**

Run:

```bash
pytest backend/tests/test_livingroom_seed_forensic.py -q
```

Expected: PASS

**Step 2: Run the promotion helper regression coverage**

Run:

```bash
pytest backend/tests/test_registry.py backend/tests/test_promote_room_versions_from_namespace.py -q
```

Expected: PASS

**Step 3: Inspect branch status**

Run:

```bash
git status --short
```

Expected: only intended docs, script, test, and log updates are present.
