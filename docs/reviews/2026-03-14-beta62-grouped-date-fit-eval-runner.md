# Beta6.2 Grouped-Date Fit/Eval Runner

- Date: 2026-03-14
- Workspace: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62`
- Branch: `codex/jessica-march-grouped-supervised-beta62`

## Goal

Add the missing grouped-date candidate-only fit/eval runner so the prepared split artifacts from the Beta6.2 grouped-date supervised path can drive a real exploratory candidate evaluation without falling back to legacy discontinuous-file flattening.

## Permanent path

- Reusable module:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62/backend/ml/beta6/grouped_date_fit_eval.py`
- Thin CLI:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62/backend/scripts/run_beta62_grouped_date_fit_eval.py`

## Input contract

The runner accepts one of:

- grouped-date supervised report JSON plus prepared split artifact directory
- grouped-date supervised manifest JSON plus prepared split artifact directory

Prepared split artifacts are the only training/evaluation data source for this path.

Supported split vocabulary:

- `train`
- optional `validation`
- optional `calibration`
- `holdout`

## Date-safety and no-legacy guarantee

- the runner consumes already materialized room/split parquet artifacts, not raw discontinuous file packs
- fit uses the explicit `train` split artifact only
- holdout evaluation uses the explicit `holdout` split artifact only
- optional `validation` and `calibration` artifacts are accepted and preserved in lineage output
- the runner never delegates to `UnifiedPipeline.train_from_files(...)`
- candidate artifacts are saved under a fresh candidate namespace with deferred promotion only

## Candidate-only behavior

The runner writes candidate room artifacts into the supplied namespace and does not update live/runtime promotion pointers.

Result JSON includes:

- manifest path and digest
- split counts by room
- artifact paths used per room/split
- lineage by room/split (`dates`, `segment_roles`, row counts)
- candidate artifact paths produced
- room-level fit result summary
- room-level holdout evaluation summary

## Verification summary

- focused RED first:
  - `pytest backend/tests/test_beta62_grouped_date_fit_eval.py -q`
  - result before implementation: import error because the fit/eval module and CLI did not exist
- focused GREEN after implementation:
  - `pytest backend/tests/test_beta62_grouped_date_fit_eval.py -q`
  - result: `4 passed, 3 warnings`
- regression slice:
  - `pytest backend/tests/test_beta62_grouped_date_fit_eval.py backend/tests/test_beta62_grouped_date_supervised.py backend/tests/test_training.py -q`
  - result: `116 passed, 3 warnings`

## Scope

- no six-date Jessica exploratory retrain was run in this thread
- no candidate was promoted
- no live/runtime behavior was modified
