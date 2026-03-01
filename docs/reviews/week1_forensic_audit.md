# Forensic Audit: Week 1 ML Hardening Implementation

**Date:** 2026-02-16
**Status:** ✅ APPROVED
**Scope:** Items 1, 2, 3, 6, 7 (Train-Split Scaling, Coverage Contract, Policy Profiles, Stat Validity, Walk-Forward Robustness).

---

## 1. Executive Summary

The implementation of the **Week 1 ML Hardening** plan is **robust, defensively programmed, and effectively decoupled**. The decision to encapsulate the new leakage-free pipeline logic in `train_split_scaling_pipeline.py` rather than mutating the existing `Pipeline` class significantly reduces regression risk. The strict use of the `ENABLE_TRAIN_SPLIT_SCALING` feature flag ensures safe deployment.

**Verdict:** Ready for Staging/Pilot deployment.

## 2. Code Quality & Architecture Review

### ✅ Strengths
*   **Decoupling (A+)**: The new `train_split_scaling_pipeline.py` module creates a clean "Pipeline within a Pipeline". It handles the complex orchestration of `preprocess_without_scaling` -> `temporal_split` -> `apply_scaling` without cluttering the main `Pipeline` class.
*   **Defensive Design (A)**:
    *   **Leakage Validation**: `validate_no_leakage` checks are explicit (`train_max < val_min`). It doesn't trust the splitter blindly; it verifies the output timestamps.
    *   **Fallback Handling**: `temporal_split_dataframe` has sensible defaults for edge cases (e.g., <10 samples force-reserves 80% for training so the model doesn't crash on empty training sets).
    *   **Statistical Validity**: The `StatisticalValidityGate` explicitly flags fallback metrics (`metric_source == 'fallback'`) as non-promotable. This closes a critical loophole where models could promote based on insufficient evaluation data.
*   **Observability (A)**:
    *   **Metadata**: The capturing of `fit_start_ts`, `fit_end_ts`, and `fit_sample_count` in the scaler metadata is excellent for future audits. We can trace exactly *what data* the scaler saw.
    *   **Explicit Reasons**: Gate failures return structured reasons (e.g., "Insufficient minority class support: class 1 has 5 < 10 samples"), making debugging trivial for Ops.

### ⚠️ Minor Observations (Non-Blocking)
*   **Hardcoded Fallback Ratio**: In `temporal_split_dataframe`, for extremely small datasets (<10 samples), the split overrides the configured `validation_split` to execute an 80/20 split. While pragmatic, this deviation from config should ideally emit a warning log (it currently does log a warning, which is good).
*   **Scalability**: The `validate_no_leakage` check iterates validation logic in Python. For massive datasets, this is negligible since we operate on room-level dataframes (typically <1M rows), so performance impact is minimal.

## 3. Test Coverage Analysis

**Total Tests:** 32 Passed (100%)
**Execution Time:** ~4.4s (Fast)

| Component | Coverage Type | Assessment |
| :--- | :--- | :--- |
| **Train-Split Scaling** | Unit + Integration | ✅ Verified `preprocess_without_scaling` preserves raw values (paramount for correctness). Verified scaler fits *only* on train split. |
| **Leakage Check** | Negative Testing | ✅ Verified it catches overlapping timestamps. |
| **Coverage Gate** | Boundary Testing | ✅ Verified 1-day data fails, 5-day data passes (correctly implementing the `min_train + valid` logic). |
| **Stat Validity** | Logic Testing | ✅ Verified failure on minority class support and class count. |
| **Feature Switch** | Config Testing | ✅ Verified `ENABLE_TRAIN_SPLIT_SCALING` defaults to `False`. |

## 4. Safety & Rollout Strategy

### Feature Flag Safety
The implementation strictly respects `os.environ["ENABLE_TRAIN_SPLIT_SCALING"]`.
*   **Default Behavior:** The code defaults to the *Legacy Path* (`preprocess_with_resampling`).
*   **Risk:** Near Zero for existing production flows unless the flag is explicitly enabled.

### Recommended Rollout
1.  **Deploy to Beta 5.5 Env**: Deploy code with flag **DISABLED** (default). Verify no regression in existing `train_and_predict` flows.
2.  **Enable in Pilot**: Set `ENABLE_TRAIN_SPLIT_SCALING=true` and `TRAINING_PROFILE=pilot` in the `.env` file or run configuration.
3.  **Validate Leakage Metrics**: Run a full retraining cycle on a known "noisy" room (e.g., `Kitchen`). Compare validation metrics.
    *   *Expectation:* Validation F1 scores may **drop** slightly (due to removal of leakage), but they will be **honest**.
    *   *Success Criteria:* Stable training, valid `scaler_metadata` in the output artifacts.

## 5. Conclusion

This implementation meets the "Production Hardening" mandate. It introduces rigorous correctness checks without destabilizing the current system.

**Action:** proceed with deployment.
