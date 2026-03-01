# Beta 6 Representation Gate Report Template (Phase 2.3)

- Date: 2026-02-26
- Status: Template
- Runner: `/Users/dicksonng/DT/Development/Beta_6/backend/scripts/run_representation_eval.py`
- Module: `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/representation_eval.py`

## 1. Run Metadata

1. `run_id`:
2. `checkpoint_npz`:
3. `dataset_csv`:
4. `seed`:

## 2. Resident-Disjoint Split Summary

1. `train_residents`:
2. `test_residents`:
3. `train_rows`:
4. `test_rows`:

## 3. Metrics

1. `linear_probe_accuracy`:
2. `random_probe_accuracy`:
3. `improvement_margin`:
4. `knn_purity`:

## 4. Gate Decision

1. Pass if `improvement_margin > 0`.
2. Fail if `improvement_margin <= 0` (representation not better than random baseline).
3. Attach reason code and mitigation owner for any fail.
