# Beta6 LivingRoom v52 Promotion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Promote `LivingRoom_v52` from the fresh Jessica candidate namespace into the forensic `HK0011_jessica` target namespace and confirm the promoted target reproduces the candidate Dec 17 replay result.

**Architecture:** Reuse the dedicated room-wise promotion helper so the target namespace keeps its older rollback history while importing the candidate `LivingRoom` version chain. After promotion, run a fresh load sanity check and a Dec 17 replay directly against the promoted target namespace, then compare its final metrics with the authoritative candidate `final_v52` replay.

**Tech Stack:** Python, `ModelRegistry`, `backend/scripts/promote_room_versions_from_namespace.py`, `UnifiedPipeline`, JSON/Parquet replay artifacts, Markdown review docs

---

### Task 1: Capture the baseline and plan

**Files:**
- Create: `docs/plans/2026-03-11-beta6-livingroom-v52-promotion.md`
- Modify: `dev_history.log`
- Read: `backend/models/HK0011_jessica/LivingRoom_versions.json`
- Read: `backend/models/HK0011_jessica_candidate_livingroom_fresh_20260311T023304Z/LivingRoom_versions.json`

**Step 1: Record baseline registry state**

Capture the target `LivingRoom current_version=46` and candidate `LivingRoom current_version=52` before promotion.

**Step 2: Record execution intent**

Append a UTC timestamped `dev_history.log` entry for the promotion task, including:
- source namespace
- target namespace
- selected room/version
- verification artifacts to produce

### Task 2: Verify the promotion helper and promote the room

**Files:**
- Read: `backend/scripts/promote_room_versions_from_namespace.py`
- Read: `backend/tests/test_promote_room_versions_from_namespace.py`
- Create: `tmp/jessica_live_livingroom_v52_promotion_20260311.json`

**Step 1: Run focused promotion helper tests**

Run:

```bash
pytest backend/tests/test_registry.py backend/tests/test_promote_room_versions_from_namespace.py -q
```

Expected: promotion/rollback helper tests pass before touching the target namespace.

**Step 2: Promote only LivingRoom**

Run:

```bash
python3 backend/scripts/promote_room_versions_from_namespace.py \
  --backend-dir backend \
  --source-elder-id HK0011_jessica_candidate_livingroom_fresh_20260311T023304Z \
  --target-elder-id HK0011_jessica \
  --room LivingRoom \
  --version LivingRoom=52 \
  --summary-out tmp/jessica_live_livingroom_v52_promotion_20260311.json
```

Expected:
- target `LivingRoom_versions.json` now reports `current_version=52`
- target rollback history still includes older live versions
- latest aliases point at `LivingRoom_v52_*`

### Task 3: Re-run live verification

**Files:**
- Create: `tmp/jessica_live_livingroom_v52_load_sanity_20260311.json`
- Create: `tmp/jessica_17dec_eval_live_livingroom_v52_20260311/final/*`
- Read: `tmp/jessica_17dec_eval_candidate_livingroom_fresh_20260311T023304Z/final_v52/comparison/summary.json`

**Step 1: Run fresh-load sanity**

Load `HK0011_jessica` through `UnifiedPipeline`, record loaded rooms, current room versions, and two-stage rooms.

Expected: all five rooms load and `LivingRoom current_version=52`.

**Step 2: Replay Dec 17 on the promoted target**

Run the Dec 17 Jessica replay harness against:

```text
/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_17dec2025.xlsx
```

Expected: fresh `comparison/summary.json` whose final metrics match or nearly match the authoritative candidate `final_v52` replay.

### Task 4: Document the result

**Files:**
- Modify: `dev_history.log`
- Create: `docs/reviews/2026-03-11-beta6-livingroom-v52-promotion.md`

**Step 1: Log promotion and verification outputs**

Append commands run, promotion summary path, load sanity path, and replay deltas vs the candidate `final_v52` summary.

**Step 2: Write the review note**

Summarize:
- whether promotion preserved rollback history
- whether the promoted target replay reproduced the candidate result
- any residual risk before a real production-facing promotion
