# Comprehensive Review: Beta 5.5 Imbalance Upgrade Execution Plan

**Date:** February 19, 2026 | **Status:** Review Complete

---

## 1. Verdict

**The plan CAN resolve the imbalance issues**, but this deeper analysis identifies **5 issues** the plan must address to succeed. The strategy is correct, but the plan underestimates the severity of two structural problems.

---

## 2. Anchor Run Failure Forensics

Extracted from anchor run `ws6_next_ab_min3_smooth_kitchen_tune`:

### 2.1 Pass/Fail by Room (3 seeds × 4 splits = 12 checks per room)

| Room | Pass | Fail | Pass Rate |
|---|---|---|---|
| **Kitchen** | 12 | 0 | 100% ✅ |
| **Bathroom** | 12 | 0 | 100% ✅ |
| **Entrance** | 12 | 0 | 100% ✅ |
| **Bedroom** | 3 | 9 | **25%** ❌ |
| **LivingRoom** | 2 | 10 | **17%** ❌ |

### 2.2 Failure Reason Breakdown

| Reason | Count | Room |
|---|---|---|
| `occupied_f1_lt_0.580` | 10 | LivingRoom |
| `occupied_recall_lt_0.500` | 9 | LivingRoom |
| `occupied_f1_lt_0.550` | 7 | Bedroom |
| `occupied_recall_lt_0.500` | 6 | Bedroom |
| `fragmentation_score_lt_0.450` | 4 | Bedroom |
| `recall_livingroom_normal_use_lt_0.400` | 3 | LivingRoom |
| `fragmentation_score_lt_0.450` | 2 | LivingRoom |
| `recall_sleep_lt_0.400` | 1 | Bedroom |

### 2.3 Eligibility Problem

> **50% of all checks (30/60) are ineligible** due to `not_eligible_below_min_train_days`. Only 6 out of 12 checks per room are eligible. Phase 1 (data coverage) blocks half the evaluation matrix.

---

## 3. Issues Found

### Issue 1: Segment Features Are Skeletal ⚠️ CRITICAL

`segment_features.py` only extracts 8 trivial features (duration, occ_mean, hour_sin/cos). **Missing:** motion stats, light levels, CO2 slopes, temperature/humidity deltas — the features needed for BL label separability. Phase 3 becomes a no-op without this work.

### Issue 2: Segment Classifier Has No Learned Model ⚠️ SIGNIFICANT

`segment_classifier.py` uses argmax of mean activity probability — not a trained classifier. It has no confidence logic, no low-support fallback, and doesn't use segment-level features at all. Even with richer features from Issue 1, the classifier can't leverage them.

### Issue 3: Model Architecture Is RF-Based Scaffold ⚠️ NOTED

Stage A uses RF/HGB, Stage B always uses RF. The greenfield blueprint flags this as a gap. The plan correctly avoids a full rebuild for the 11-day timeline, but if RF/HGB capacity is the actual ceiling, no amount of feature engineering will reach 30/30.

### Issue 4: Timeline Is Aggressive ⚠️ MODERATE

Phase 3 needs 4-5 days, not 3, given Issues 1-2 require substantial new code. Total should be 13 days with buffer.

### Issue 5: Experiment Matrix Complexity ⚠️ MINOR

9 factors × 3 seeds = ~40 hours of compute. Plan should specify parallelism strategy.

---

## 4. What the Plan Gets Right

1. **Correct root cause identification:** BL occupied recall and F1 are the blocking failures.
2. **Data-first approach:** Phase 1 addresses the 50% ineligibility problem.
3. **Fail-closed discipline and default-off rollout.**
4. **Hard-negative mining logic is sound** (boost FN weights → retrain).
5. **Arbitration is well-wired** with all necessary flags.
6. **Risk register is realistic.**

---

## 5. Recommendations

| Priority | Action | Phase Impact |
|---|---|---|
| **P0** | Enrich `segment_features.py` with sensor stats before Phase 3 | Phase 3 prerequisite |
| **P1** | Upgrade `segment_classifier.py` to use segment-level features | Phase 3 |
| **P2** | Add 2 buffer days to Phase 3 timeline | Timeline |
| **P3** | Define experiment parallelism strategy in Section 5.1 | Phase 2-4 |
| **P4** | Consider HGB as Stage A default for BL rooms | Phase 2 |

---

## 6. Go/No-Go

| Question | Answer |
|---|---|
| Can the plan reach 30/30? | **Yes**, if Issues 1-2 are addressed |
| Is the strategy correct? | **Yes** |
| Is the timeline realistic? | **Tight** — needs 2 extra days for Phase 3 |
| Should we proceed? | **Yes**, with modifications above |
| Is there a showstopper? | **No**, but Phase 3 without segment feature enrichment is a dead end |
