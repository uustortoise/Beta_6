# Beta6.2 Grouped-Date Supervised Path

- Date: 2026-03-14
- Workspace: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62`
- Branch: `codex/jessica-march-grouped-supervised-beta62`

## Goal

Add a reusable Beta6.2 supervised prep/evaluation surface that consumes explicit day segments and never delegates discontinuous Dec+Mar packs to legacy `UnifiedPipeline.train_from_files(...)`.

## Permanent path

- Reusable module:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62/backend/ml/beta6/grouped_date_supervised.py`
- Thin CLI:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62/backend/scripts/run_beta62_grouped_date_supervised.py`

## Manifest contract

Required top-level fields:

- `schema_version`
- `resident_id`
- `target_rooms`
- `segments`

Optional top-level fields:

- `sequence_length_by_room`
- `notes`

Each segment requires:

- `role` (`baseline` or `candidate`)
- `date`
- `split` (`train`, `validation`, `calibration`, `holdout`)
- `path`

## Date-safety rules

- day files are loaded independently per segment
- split assignment is explicit in the manifest, not derived from a single global timestamp cutoff
- grouped-by-date output remains tied to manifest lineage (`role`, `date`, `split`, `path`)
- prepared split artifacts are emitted per room/per split with metadata columns:
  - `__segment_role`
  - `__segment_date`
  - `__segment_split`

This keeps discontinuous December and March data from being silently flattened into one synthetic timeline before downstream supervised training/evaluation.

## Current output surface

The grouped-date supervised path now emits:

- machine-readable supervised report JSON
- grouped-by-date room summaries with row counts, usable rows, sequence counts, and class counts
- prepared split parquet artifacts per room/per split when `--artifact-dir` is provided

These artifacts are the governed supervised input surface for the next six-date Jessica exploratory retrain thread.

## Verification summary

- focused RED first:
  - `pytest backend/tests/test_beta62_grouped_date_supervised.py -q`
  - result before implementation: `ModuleNotFoundError: No module named 'ml.beta6.grouped_date_supervised'`
- focused GREEN after implementation:
  - `pytest backend/tests/test_beta62_grouped_date_supervised.py -q`
  - result: `4 passed, 3 warnings`

## Scope

- no March retrain was run in this thread
- no candidate was promoted
- no live/runtime behavior was modified
