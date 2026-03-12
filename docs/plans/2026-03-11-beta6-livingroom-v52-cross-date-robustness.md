# Beta6 LivingRoom v52 Cross-Date Robustness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate whether promoted `LivingRoom_v52` stays better than live `LivingRoom_v46` across the corrected Jessica dates `2025-12-04`, `05`, `06`, `07`, `08`, `09`, `10`, and `17`, then escalate to `v50` and `v51` only if the date sweep shows mixed behavior.

**Architecture:** Reuse the recovered replay harness against the forensic `HK0011_jessica` namespace, running one replay per date for `v52` and `v46` under an isolated output root. Summarize the LivingRoom deltas date-by-date from the produced `comparison/summary.json` files; only run extra comparator sweeps on dates where `v52` is flat, worse, or materially noisy versus `v46`.

**Tech Stack:** Python, `UnifiedPipeline`, recovered benchmark harness, corrected Jessica `.xlsx` packs, JSON/Parquet replay artifacts, Markdown review docs

---

### Task 1: Lock inputs and record execution intent

**Files:**
- Create: `docs/plans/2026-03-11-beta6-livingroom-v52-cross-date-robustness.md`
- Modify: `dev_history.log`
- Read: `backend/models/HK0011_jessica/LivingRoom_versions.json`
- Read: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-confidence-architecture/tmp/jessica_activity_confidence_benchmark.py`

**Step 1: Confirm the active forensic target**

Verify `backend/models/HK0011_jessica/LivingRoom_versions.json` still reports `current_version=52` before the sweep starts.

**Step 2: Record the sweep outputs**

Append a UTC timestamped `dev_history.log` entry naming the output root `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z` and the planned comparator order `v52 -> v46 -> (v50/v51 only if needed)`.

### Task 2: Run the primary cross-date sweep (`v52` vs `v46`)

**Files:**
- Read: `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_4dec2025.xlsx`
- Read: `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_5dec2025.xlsx`
- Read: `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_6dec2025.xlsx`
- Read: `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_7dec2025.xlsx`
- Read: `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_8dec2025.xlsx`
- Read: `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_9dec2025.xlsx`
- Read: `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_10dec2025.xlsx`
- Read: `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_17dec2025.xlsx`
- Create: `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/v52/*`
- Create: `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/v46/*`

**Step 1: Roll to `LivingRoom_v52` and run all corrected-pack dates**

For each corrected-pack date, run the recovered harness with:

```bash
PYTHONPATH='.:backend' python3 /Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-confidence-architecture/tmp/jessica_activity_confidence_benchmark.py \
  --elder-id HK0011_jessica \
  --source-file "/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_<date>.xlsx" \
  --output-dir tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/v52/<date>
```

Expected: one `comparison/summary.json` per date under the `v52` subtree.

**Step 2: Roll back to `LivingRoom_v46` and rerun the same dates**

Repeat the same corrected-pack sweep with `HK0011_jessica` temporarily set to `LivingRoom_v46`, writing outputs under the `v46` subtree.

Expected: one directly comparable `comparison/summary.json` per date under the `v46` subtree.

### Task 3: Decide whether targeted `v50` / `v51` replays are required

**Files:**
- Create: `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/comparison_primary.json`
- Create: `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/v50/*`
- Create: `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/v51/*`

**Step 1: Aggregate the primary sweep**

Build a compact JSON summary containing, for each date:
- `v52` LivingRoom accuracy and macro-F1
- `v46` LivingRoom accuracy and macro-F1
- deltas (`v52 - v46`)
- a flag for any mixed result

Expected: a single artifact that makes the go/no-go for extra comparators explicit.

**Step 2: Only if mixed dates exist, replay `v50` and `v51` there**

For each mixed date only, rerun the harness after activating `LivingRoom_v50` and `LivingRoom_v51`, storing outputs in the `v50` and `v51` subtrees.

Expected: no extra runs if `v52` consistently beats or matches `v46`; otherwise, targeted comparator evidence on the unstable dates only.

### Task 4: Record the verdict

**Files:**
- Modify: `dev_history.log`
- Create: `docs/reviews/2026-03-11-beta6-livingroom-v52-cross-date-robustness.md`

**Step 1: Append the execution log**

Record the versions tested, output roots, any rollback/activation commands used, and whether the targeted comparator branch was needed.

**Step 2: Write the review note**

Summarize:
- date-by-date `v52` vs `v46` outcome
- whether `v52` is promotion-grade across the corrected pack
- whether `v50` or `v51` needed to be checked
- whether a new brittleness forensic is justified
