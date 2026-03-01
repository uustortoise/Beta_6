# ML Stabilization Roadmap: Phases 2–4

**Owner:** Dickson (Senior Engineer)
**Goal:** Transition from "retrain-all" to stable, incremental learning with rigorous temporal validation.

## Beta 6 Phase 6.3 Cutover Readiness (2026-02-27)
- Added daily stability certification artifact path in `run_daily_analysis.py`.
- Certification now persists rolling consecutive-stable-day state per elder and emits signed artifacts.
- Active authority signal is explicit in each report (`beta5.5_authority` vs `beta6_authority`).
- Default certification requirement is 14 consecutive stable days, aligned with rollout ladder policy.
- Cutover policy remains fail-closed: unstable day resets the consecutive-day counter.

## Phase 2: True Fine-Tuning with Replay Buffer (Week 2)
**Objective:** Replace "full retrain on correction" with "warm-start fine-tuning" to stabilize model behavior and reduce compute waste.

### Current Status (as of 2026-02-13)
- Implemented:
  - Correction retrain path now uses `training_mode="correction_fine_tune"` through `train_from_files(...)`.
  - Warm-start from champion model is enabled when champion artifacts exist.
  - Layer freezing is applied on warm-start fine-tune: lower representation layers frozen; top Transformer block(s) + head remain trainable.
  - Replay sampling is enabled with corrected:uncorrected mixing (target 1:10 from policy).
  - Replay sampling strategy now respects policy-driven stratified replay (`sampling: random_stratified`).
  - Fine-tune uses lower LR, fewer epochs, and tighter early stopping than full retrain.
  - Model version lineage metadata (`parent_version_id`) is persisted for fine-tuned candidates.
  - Release-gate hardening fixes landed:
    - Early-window threshold fallback uses earliest bracket.
    - Gate room-key normalization is consistent with policy keys.
    - Small-dataset bootstrap path no longer fails by construction.
  - Registry retention now preserves champion rollback safety under churn.
- Remaining for full Phase 2 completion:
  - Optional transfer-learning load path without compile for further fine-tune startup optimization.

### 2.1. Model Registry Enhancements
- [ ] **Checkpoint Loading:** Add optional `compile_model=False` behavior (or equivalent) so warm-start paths can avoid unnecessary compile work.
- [x] **Lineage Tracking:** Added `parent_version_id` metadata to track fine-tuning chains (e.g., `v5 (fine-tuned from v4)`).

### 2.2. Training Pipeline Updates (`train_room`)
- [x] **Warm-Start Logic:**
    - If champion exists, warm-start from current champion artifact before fine-tune.
    - Freeze lower layers (CNN), unfreeze Top 2 Transformer blocks + Head.
    - Reduce LR: default `1e-5` for fine-tune; allow policy override per room if needed.
    - Reduce Epochs: small fixed budget (e.g., `3-5`) vs full retrain schedule.
- [x] **Replay Buffer Implementation:**
    - **Problem:** Fine-tuning on 50 corrections overfits and forgets old patterns.
    - **Solution:** `ReplayGenerator`.
        - Inputs: `corrections_df` (new labels), `archive_df` (historical data).
        - Logic: For every 1 correction sample, sample 10 diverse historical samples.
        - Output: Mixed `X_train`, `y_train` respecting temporal order.

### 2.3. Correction Studio Integration
- [x] Updated correction retrain flow in `export_dashboard.py` to call `train_from_files(..., training_mode="correction_fine_tune")`.
- [ ] Ensure `release_gates` still apply to the fine-tuned candidate (no regress rule is critical here).

---

## Phase 3: Leakage-Safe Validation (Week 3)
**Objective:** Replace naive random split with strict walk-forward validation to measure *true* production performance.

### Current Status (as of 2026-02-13)
- Implemented:
  - `TimeCheckpointedSplitter` (calendar-day expanding-window folds) in `backend/ml/evaluation.py`.
  - Standalone frozen-model walk-forward evaluator `evaluate_model(...)` in `backend/ml/evaluation.py`.
  - Fold-level metrics and confusion matrices with aggregate summary metrics.
  - Monitoring export endpoint for model-health metrics is available for scheduler/Prometheus wiring.
- Remaining:
  - Optional: production observability wiring (scrape config / scheduled exporter job).

### 3.1. Temporal Splitter
- [x] Implement `TimeCheckpointedSplitter`:
    - Define folds based on calendar time, not row count.
    - **Fold 1:** Train Dec 1-7, Valid Dec 8.
    - **Fold 2:** Train Dec 1-8, Valid Dec 9.
    - **Fold 3:** Train Dec 1-14, Valid Dec 15-17.

### 3.2. Evaluation Pipeline (`ml/evaluation.py`)
- [x] Create standalone `evaluate_model(model, data_range)`:
    - Loads frozen model.
    - Replays data stream as if live (no future peeking).
    - Computes metrics: Precision/Recall/F1 per class, Confusion Matrix.
    - **Critical:** Ignores training data in metric calculation (unlike current `val_split` which allows random shuffle leakage if not careful).

### 3.3. Dashboard "True Performance" Tab
- [x] New view in Dashboard: `Model Health`.
    - Plots "Holdout F1" over time (not Training Accuracy).
    - Drift detection: Warn if "Daily Holdout F1" drops below 0.60.

---

## Phase 4: Data Audit & Baselines (Week 4)
**Objective:** Quantify data quality issues (e.g., Bathroom accuracy) and validate Transformer complexity.

### 4.1. The "Little Model" Baseline
- [ ] Implement `XGBoostBaseline`:
    - Features: `time_of_day`, `seconds_since_last_event`, `prev_room`, `rolling_sensor_activation`.
    - Train alongside Transformer on 22-day dataset.
    - **Decision Gate:** Align with `release_gates.json` baseline policy (`required_transformer_advantage=0.05` at day 22) unless policy is explicitly revised.

### 4.2. Label Quality Audit
- [ ] **Bathroom Analysis Script:**
    - Filter for `Bathroom` episodes < 30s vs > 2 mins.
    - Correlate with `shower` vs `toilet` labels.
    - Identify "Unoccupied" sequences with high sensor activity (likely missed labels).
- [ ] **Confusion Matrix Heatmap:**
    - Visualize `Predicted: Unoccupied` vs `Actual: Bathroom` (Type II error).

### 4.3. Class Balancing Fixes
- [ ] **Active Learning Sampler:**
    - Instead of random training batches, sampling probability $\propto (1 - \text{class\_accuracy})$.
    - Force model to see `shower` 5x more often than `unoccupied` during training.

---

## Execution Order
1. **Week 2:** Phase 2.1 & 2.2 (Replay Fine-Tuning) — *High Technical Risk*
2. **Week 3:** Phase 3 (Walk-Forward Eval) — *High Value, Low Risk*
3. **Week 4:** Phase 4 (Audit & Baseline) — *Optimization*

**Sign-off:**
- [ ] Senior Engineer (Dickson)
- [ ] Team Lead
