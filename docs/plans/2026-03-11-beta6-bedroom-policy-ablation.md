# Bedroom Policy Ablation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Quantify how much Bedroom Dec 17 performance can be recovered by policy-only changes using the canonical saved replay outputs for `Bedroom_v38`.

**Architecture:** Reuse the canonical March 11 repaired replay artifacts instead of rerunning inference or retraining. Load the saved Bedroom parquet, derive a few policy-only export variants from the recorded raw and final columns, and compare those variants against the existing benchmark to determine whether runtime policy can materially close the gap.

**Tech Stack:** Python 3, pandas, scikit-learn, saved parquet/json artifacts under `tmp/`

---

### Task 1: Lock canonical inputs

**Files:**
- Read: `tmp/jessica_17dec_eval_live_livingroom_rootfix_registryfix_20260311/final/comparison/summary.json`
- Read: `tmp/jessica_17dec_eval_live_livingroom_rootfix_registryfix_20260311/final/comparison/all_rooms_merged.parquet`
- Read: `tmp/jessica_17dec_eval_live_livingroom_rootfix_registryfix_20260311/final/raw_predictions/Bedroom.parquet`

**Step 1: Verify the canonical benchmark anchor**

Run a one-off script that reads `summary.json` and records:
- Bedroom final macro-F1
- Bedroom raw top-1 macro-F1
- Bedroom raw export macro-F1
- merged parquet path

**Step 2: Verify the Bedroom parquet exposes policy fields**

Confirm the parquet includes:
- `activity`
- `predicted_top1_label_raw`
- `predicted_activity_raw`
- `predicted_activity`
- `activity_acceptance_score`
- `low_confidence_threshold`

### Task 2: Compute policy-only variants

**Files:**
- Read: `tmp/jessica_17dec_eval_live_livingroom_rootfix_registryfix_20260311/final/comparison/all_rooms_merged.parquet`
- Create: `tmp/bedroom_dec17_policy_ablation_20260311/summary.json`

**Step 1: Derive variants from saved Bedroom rows**

Compute at least:
- `raw_top1_direct`: use `predicted_top1_label_raw`
- `raw_export`: use `predicted_activity_raw`
- `final_export`: use `predicted_activity`
- `no_low_confidence`: start from `predicted_activity`, but replace `low_confidence` with the corresponding raw top-1 label

**Step 2: Quantify recoverable policy-only gain**

For each variant, compute:
- accuracy
- macro-F1
- per-label precision/recall/F1 for `bedroom_normal_use`, `sleep`, `unoccupied`
- confusion counts for dominant error pairs

**Step 3: Record how many rows each variant changes**

Measure row deltas relative to the canonical final export so the note can distinguish “tiny policy effect” from “material policy effect.”

### Task 3: Write the forensic follow-up

**Files:**
- Modify: `docs/reviews/2026-03-11-beta6-bedroom-dec17-forensic.md`
- Create: `docs/reviews/2026-03-11-beta6-bedroom-policy-ablation.md`

**Step 1: Add the ablation result**

Document:
- canonical benchmark
- variant metrics
- maximum achievable policy-only gain from saved outputs
- whether the dominant `unoccupied -> bedroom_normal_use` confusion survives the best variant

**Step 2: State the next move**

If policy-only gain is negligible, recommend a Bedroom-only retrain focused on occupied-vs-unoccupied separation.

### Task 4: Verify and log

**Files:**
- Modify: `dev_history.log`

**Step 1: Verify artifacts exist and are internally consistent**

Confirm:
- the new ablation summary file exists
- the written note matches the computed metrics

**Step 2: Append development history**

Log:
- actions taken
- artifacts produced
- verification performed
- final recommendation
