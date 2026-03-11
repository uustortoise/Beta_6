# Beta6 Promotion Metadata Normalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make room-wise promotion rewrite namespace-bound JSON metadata so promoted artifacts are target-native on first promotion and remain safe to re-promote.

**Architecture:** Keep the current room/version merge flow, but treat copied JSON artifacts as structured payloads instead of opaque blobs. During promotion, normalize source-namespace `elder_id` and model-path references to the target namespace before writing versioned JSON artifacts, and compare JSON artifacts namespace-agnostically so repeat promotions repair older copied payloads instead of flagging false conflicts.

**Tech Stack:** Python, `backend/scripts/promote_room_versions_from_namespace.py`, `ModelRegistry`, `unittest`, JSON artifact normalization

---

### Task 1: Capture the root cause and target behavior

**Files:**
- Create: `docs/plans/2026-03-11-beta6-promotion-metadata-normalization.md`
- Modify: `dev_history.log`
- Read: `backend/scripts/promote_room_versions_from_namespace.py`
- Read: `backend/models/HK0011_jessica/LivingRoom_decision_trace.json`

**Step 1: Record the defect**

Document that the helper copies versioned JSON artifacts verbatim, leaving promoted traces and two-stage metadata bound to the candidate namespace.

**Step 2: Record the required invariant**

Promoted JSON artifacts must refer to the target namespace, and re-running promotion must not fail or preserve stale source metadata.

### Task 2: Write failing tests first

**Files:**
- Modify: `backend/tests/test_promote_room_versions_from_namespace.py`

**Step 1: Add first-promotion normalization coverage**

Create a test where versioned decision-trace and two-stage metadata include:
- `elder_id=<source elder>`
- absolute model paths under the source models dir

Expected after promotion:
- versioned target JSON references `<target elder>`
- latest aliases also reference `<target elder>`
- target model paths point into the target models dir

**Step 2: Add repeat-promotion repair coverage**

Create a test where the target already contains a copied version with stale source metadata.

Expected after re-running promotion:
- helper accepts the artifact as equivalent
- target JSON is rewritten to the normalized target form

### Task 3: Implement normalization in the helper

**Files:**
- Modify: `backend/scripts/promote_room_versions_from_namespace.py`

**Step 1: Add JSON namespace normalization helpers**

Implement helpers that:
- detect JSON artifacts
- recursively rewrite source elder/path references to target elder/path
- canonicalize JSON for namespace-agnostic equivalence checks

**Step 2: Use normalized writes and comparisons**

Update promotion so:
- copied JSON artifacts are written in normalized target form
- existing JSON artifacts are compared canonically, not by raw hash
- equivalent but stale JSON artifacts are rewritten in place to the normalized target form

### Task 4: Verify and re-apply locally

**Files:**
- Modify: `dev_history.log`
- Update existing local artifacts via helper rerun

**Step 1: Run focused tests**

Run:

```bash
pytest backend/tests/test_promote_room_versions_from_namespace.py -q
```

Expected: new normalization tests fail before implementation, then pass after.

**Step 2: Re-promote local LivingRoom_v52**

Run the promotion helper again against the forensic `HK0011_jessica` target.

Expected:
- promoted JSON now embeds `HK0011_jessica`
- decision-trace model paths point at `backend/models/HK0011_jessica/...`
- replay behavior remains unchanged
