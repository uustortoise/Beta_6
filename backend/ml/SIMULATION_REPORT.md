# ML Stability Simulation Report: Solving Recall Instability & Collapse

**Date:** Feb 16, 2026
**Author**: AntiGravity (Agent)
**Status**: Concept Note (Non-Binding)
**Context**: Investigating theoretical metric behavior for "Model Collapse" scenarios.

---

## 1. Executive Summary

**Concept Note Only**: This report uses simplified IID simulations to demonstrate the *theoretical* utility of MCC as a collapse signal. It is not evidence of production efficacy.

**Key Concepts:**
1.  **Collapse Detection**: The **Matthews Correlation Coefficient (MCC)** correctly identifies both "Lazy" (All-Zero) and "Ghost" (All-One) failure modes as uncorrelated (MCC ≈ 0.0).
2.  **Safety Additive**: MCC measures *overall* correlation but can hide specific failures. It should be used **in addition** to, not instead of, support-aware critical label gates.
3.  **Recommendation**: Fix the downsampling pipeline to enable honest validation, then instrument MCC as a monitoring signal.

---

## 2. The Problem: "The Recall Trap"

In imbalanced datasets (e.g., a room is Empty 98% of the time), widespread metrics fail:

*   **Accuracy is deceptive**: A model that *always* predicts "Empty" is **98% Accurate** but has **0% Activity Recall**.
*   **Recall is deceptive**: A model that *always* predicts "Activity" has **100% Activity Recall** but is functionally hallucinating (Ghost).

We need a metric that punishes *both* extremes.

---

## 3. Simulation Setup (V2 - Robust)

*   **Script**: `backend/scripts/simulate_proposal_v2.py`
*   **Methodology**:
    *   **Structure**: Temporal split (Night/Day drift) with overlapping classes (Harder task).
    *   **Reproducibility**: 10 runs with fixed seeds (42-51).
    *   **Comparison**:
        *   **Scenario A (Bug)**: Downsample validation data (current state).
        *   **Scenario B (Fix)**: Downsample training only + Class Weights.

## 4. Simulation Results (Mean over 10 runs)

### A. The "Downsampling Bug" (Current Baseline)
*   **Setup**: Downsampling applied to *both* Train and Validation (simulating current sequence).
*   **Metrics**:
    *   **Activity Recall**: **0.528** (Poor) - We miss ~47% of activity events.
    *   **Precision**: **0.929** (High) - Conservative model due to data loss.
    *   **MCC**: **0.636**
*   *Verdict*: The current pipeline is overly conservative, sacrificing massive amounts of recall.

### B. The "Pipeline Fix" (Proposed)
*   **Setup**: Downsampling applied *only* to Train. Validation is full/honest. Class Weights used.
*   **Metrics**:
    *   **Activity Recall**: **0.819** (+0.291 gain) - We capture ~30% more activity.
    *   **Precision**: **0.755** (Trade-off) - Expected drop as we become more sensitive.
    *   **MCC**: **0.711** (+0.075 gain) - Overall correlation improves significantly.
*   *Verdict*: The fix provides a **~30% Recall boost** for critical activities, which is the primary goal of this safety initiative.

---

## 5. Technical Recommendation

### Phase 1: Fix Pipeline (Root Cause)
*   **Action**: Move `_downsample_easy_unoccupied` to happen **AFTER** the Train/Validation split.
*   **Why**: Currently, we validate on downsampled data, which hides false positives. We must validate on "Real" (imbalanced) data to detect ghosts.

### Phase 2: Additive Guardrails (MCC)
*   **Action**: Add `MCC` to `backend/ml/evaluation.py` and log it.
*   **Gate**: Use MCC as a "Collapse Check" (e.g., block if MCC < 0.2), while keeping specific Per-Label Recall gates for safety.
*   **Why**: MCC provides a single-metric health check, but specific safety constraints are still needed for critical labels.

### Phase 3: Smart Weighting (Optimization)
*   **Action**: Use `class_weight='balanced'` or `Focal Loss` in `training.py`.
*   **Why**: Allows the model to learn rare events without aggressive downsampling (which destroys data).
