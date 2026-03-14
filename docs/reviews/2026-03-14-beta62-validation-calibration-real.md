# Beta6.2 Validation And Calibration Split Consumption

- Date: `2026-03-14`
- Workspace: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/beta62-reliability-followup-execution`
- Branch: `codex/beta62-reliability-followup-execution`

## Task

`Task 2: Make validation and calibration real`

## Root cause

The grouped-date fit/eval path only trained on the grouped `train` split and then evaluated `holdout`.

It did not consume the explicit grouped-date `validation` or `calibration` artifacts:

1. `grouped_date_fit_eval.py` zeroed the generic validation split and never built grouped-date validation/calibration sequences.
2. `TrainingPipeline.train_room(...)` had no way to accept explicit override validation/calibration data even when the grouped-date path had already prepared those splits.
3. As a result, grouped-date reports could not honestly say that explicit validation drove checkpoint selection or that explicit calibration drove thresholding.

## Fix

1. `backend/ml/beta6/grouped_date_fit_eval.py` now builds explicit validation/calibration sequences from the grouped-date split artifacts using the same segment-aware preprocessing and strict sequence alignment used elsewhere on the grouped-date path.
2. `backend/ml/training.py` now accepts optional `explicit_validation_data` and `explicit_calibration_data` overrides in `TrainingPipeline.train_room(...)`.
3. When those overrides are present, training uses them directly for `model.fit(...)` validation and threshold calibration instead of inventing a temporal split from the train data.
4. Fit metrics now record:
   - `validation_source`
   - `calibration_source`
   - `validation_samples`
   - `calibration_samples`

## Verification

- RED:
  - `pytest backend/tests/test_beta62_grouped_date_fit_eval.py -q -k explicit_validation_and_calibration_sequences` -> failed before implementation because grouped-date fit/eval did not pass explicit split data into training.
  - `pytest backend/tests/test_training.py -q -k explicit_validation_and_calibration_overrides` -> failed before implementation because `TrainingPipeline.train_room()` did not accept explicit override inputs.
- GREEN:
  - `pytest backend/tests/test_beta62_grouped_date_fit_eval.py -q -k explicit_validation_and_calibration_sequences` -> `1 passed, 8 deselected`
  - `pytest backend/tests/test_training.py -q -k explicit_validation_and_calibration_overrides` -> `1 passed, 108 deselected`
- Regression slice:
  - `pytest backend/tests/test_beta62_grouped_date_fit_eval.py backend/tests/test_beta62_grouped_date_supervised.py backend/tests/test_training.py -q` -> `123 passed, 3 warnings`
  - `python3 -m py_compile backend/ml/beta6/grouped_date_fit_eval.py backend/ml/training.py` -> exit `0`

## Decision

`Task 2` is complete on this branch.

The next reliability gap is `Task 3: Make timeline quality first-class`.
