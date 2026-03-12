# Beta6 LivingRoom Brittleness Forensic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Explain why the fresh LivingRoom seeds (`v50..v52`) improve Dec 17 but lose to the old `v46` anchor on most corrected-pack dates, and determine whether the leading mechanism is occupancy routing/calibration drift or a deeper model-shape regression.

**Architecture:** Reuse the matched cross-date replay artifacts already generated under `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z`, then build a reproducible one-off forensic script that compares `v46`, `v50`, and `v52` on the exact same timestamps. The script should join replay outputs, summarize occupied-rate drift by date and hour, extract changed contiguous segments, and attach saved model metadata (`decision_trace` plus `two_stage_meta`) so the replay reversals can be tied back to threshold/gate geometry rather than guessed from top-line metrics.

**Tech Stack:** Python, pandas, sklearn metrics, JSON, Parquet replay artifacts, LivingRoom decision-trace/two-stage metadata, Markdown review docs

---

### Task 1: Lock scope and record intent

**Files:**
- Create: `docs/plans/2026-03-11-beta6-livingroom-brittleness-forensic.md`
- Modify: `dev_history.log`
- Read: `docs/reviews/2026-03-11-beta6-livingroom-v52-cross-date-robustness.md`
- Read: `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/comparison_panel.json`

**Step 1: Record the forensic target**

Capture that the study compares:
- stable anchor `v46`
- strongest fresh neighbor `v50`
- Dec 17 winner `v52`
- catastrophic control `v51` metadata only

**Step 2: Log execution intent**

Append a UTC `dev_history.log` entry naming:
- output root `tmp/livingroom_brittleness_forensic_20260311T093026Z`
- script path `tmp/livingroom_brittleness_forensic.py`
- review target `docs/reviews/2026-03-11-beta6-livingroom-brittleness-forensic.md`

### Task 2: Implement the forensic analysis script

**Files:**
- Create: `tmp/livingroom_brittleness_forensic.py`

**Step 1: Load replay artifacts and model metadata**

The script should load:
- `LivingRoom_merged.parquet` for available versions per date under `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z`
- `LivingRoom_v46/v50/v51/v52_decision_trace.json`
- `LivingRoom_v46/v50/v51/v52_two_stage_meta.json`

**Step 2: Emit machine-readable evidence**

The script should write:
- `summary.json`
- `date_version_summary.csv`
- `hourly_summary.csv`
- `pair_segment_summary.csv`
- `changed_rows.parquet`

Required content:
- per-date version metrics and occupied-rate drift
- pairwise changed-row counts (`v46_vs_v50`, `v46_vs_v52`)
- dominant truth-label flips and contiguous disagreement segments
- replay-side evidence for whether errors are low-confidence rewrites or confident upstream routing
- saved threshold / calibration metadata needed to interpret the replay behavior

### Task 3: Run the forensic and inspect outputs

**Files:**
- Create: `tmp/livingroom_brittleness_forensic_20260311T093026Z/*`

**Step 1: Execute the script**

Run the script from repo root against the existing sweep root and write the results into the dedicated forensic output folder.

Expected: a complete summary plus parquet/csv evidence without rerunning inference.

**Step 2: Inspect the decisive metrics**

Verify:
- whether the `v52-v46` delta tracks occupied-share regime
- whether changed rows are dominated by `unoccupied <-> livingroom_normal_use`
- whether the disagreement segments are coherent time blocks instead of scattered noise
- whether confidence / threshold evidence points upstream of late low-confidence logic

### Task 4: Write the verdict

**Files:**
- Create: `docs/reviews/2026-03-11-beta6-livingroom-brittleness-forensic.md`
- Modify: `dev_history.log`

**Step 1: Document the root-cause conclusion**

Summarize:
- what the fresh family is doing differently from `v46`
- why Dec 17 benefits while Dec 4-9 regress
- whether `v50` and `v52` reflect the same mechanism at different strengths
- what `v51` proves about fresh-panel instability

**Step 2: Make the next-step recommendation explicit**

If the evidence points to occupancy routing/calibration drift, recommend a narrow fixed-weight threshold/gate experiment before any retrain. If it does not, explicitly hold the line on `v46` and explain why calibration-only changes are unlikely to be enough.
