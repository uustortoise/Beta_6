# Beta6.2 Jessica Grouped-Date Retrain Blocked

- Date: 2026-03-14
- Workspace: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62`
- Branch: `codex/jessica-march-grouped-supervised-beta62`

## Requested run

Attempt one exploratory Jessica retrain using the new grouped-date supervised Beta6.2 path with:

- accepted Dec baseline `2025-12-04..2025-12-10`
- included March dates:
  - `2026-02-20`
  - `2026-03-02`
  - `2026-03-06`
  - `2026-03-07`
  - `2026-03-08`
  - `2026-03-09`
- excluded March dates as holdout-only context:
  - `2026-02-18`
  - `2026-03-01`
  - `2026-03-03`
  - `2026-03-04`
  - `2026-03-05`

## What actually ran

The grouped-date supervised manifest was built and executed through:

- `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62/backend/scripts/run_beta62_grouped_date_supervised.py`

Artifacts written:

- manifest:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62/tmp/jessica_grouped_date_supervised_attempt_20260314T014529Z/grouped_date_manifest.json`
- grouped-date supervised report:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62/tmp/jessica_grouped_date_supervised_attempt_20260314T014529Z/grouped_date_supervised_report.json`
- prepared split parquets:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62/tmp/jessica_grouped_date_supervised_attempt_20260314T014529Z/prepared_splits/`

Prepared split summary:

- `18` manifest segments total
- `13` train segments
- `5` holdout segments
- per-room prepared train artifacts for `Bathroom`, `Bedroom`, `Kitchen`, `LivingRoom`
- per-room prepared holdout artifacts for `Bathroom`, `Bedroom`, `Kitchen`, `LivingRoom`

## Blocker

No candidate model was trained.

Why:

- `run_beta62_grouped_date_supervised.py` only loads the manifest, runs `run_grouped_date_supervised(...)`, writes a JSON report, and exits
- `run_grouped_date_supervised(...)` only prepares grouped summaries and split parquet artifacts; it does not fit a model, save model artifacts, create a candidate namespace, or run replay evaluation
- therefore no honest room-by-room retrain result exists yet for comparison against the signed-off Jessica baseline

This is a surface mismatch:

- the grouped-date supervised path is valid as a governed split-artifact generator
- it is not yet a full retrain/evaluation runner

## Consequence

This thread did **not** produce:

- a candidate namespace
- holdout/train model metrics
- Dec 17 replay metrics
- room-by-room exploratory retrain results

The signed-off baseline remains unchanged:

- candidate: `HK0011_jessica_candidate_supportfix_20260310T2312Z`
- replay accuracy: `0.8417598841689825`
- replay macro-F1: `0.4663352546942085`

## Recommendation

Do not claim a retrain result from this thread.

Next implementation step is to add a true grouped-date model-fit consumer that takes the emitted room/split artifacts and performs:

1. per-room supervised fit
2. holdout evaluation
3. Dec 17 replay
4. candidate-only artifact save without promotion
