# Beta 6 Safe-Class Fine-Tune Contract (Phase 3.1)

- Date: 2026-02-26
- Status: Active
- Module: `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/fine_tune_safe_classes.py`
- Config: `/Users/dicksonng/DT/Development/Beta_6/backend/config/beta6_golden_safe_finetune.yaml`

## 1. Gate Rules

1. Dataset must contain all configured safe classes.
2. Per-class support must meet `min_samples_per_class`.
3. Resident-disjoint holdout must be feasible.
4. Held-out accuracy must be `>= min_accuracy` (default `0.85`).
5. Leakage report must have no warnings (`resident/time/window overlap`).

## 2. Artifacts

1. `safe_class_head.joblib`
2. `safe_class_finetune_report.json`

## 3. Determinism

1. Split and model training are seeded via `random_seed`.
2. Re-running with the same dataset and seed must produce identical held-out accuracy.
