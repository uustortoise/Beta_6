# Beta6.2 Grouped-Date Fit/Eval Runner Implementation Plan

> **For Codex:** Required workflow for this task: `brainstorming` -> `writing-plans` -> `test-driven-development` -> implementation -> `verification-before-completion`.

**Goal:** Add a real Beta6.2 grouped-date candidate-only fit/eval runner that consumes prepared split artifacts from the grouped-date supervised path, fits room models from explicit split artifacts, evaluates explicit holdout artifacts, and never delegates discontinuous Dec+Mar packs to `UnifiedPipeline.train_from_files(...)`.

**Architecture:** Implement reusable fit/eval logic in `backend/ml/beta6/grouped_date_fit_eval.py` and expose it through a thin CLI at `backend/scripts/run_beta62_grouped_date_fit_eval.py`. The runner will consume prepared split parquets, keep split vocabulary open (`train`, optional `validation`, optional `calibration`, `holdout`), save candidate-only artifacts into a fresh namespace, and emit lineage-rich JSON results.

**Tech Stack:** Python 3, pandas, numpy, pytest, existing Beta6 training/evaluation internals, grouped-date supervised report artifacts, model registry candidate versioning.

---

### Task 1: Lock the fit/eval contract with failing tests

**Files:**
- Create: `backend/tests/test_beta62_grouped_date_fit_eval.py`

**Step 1: Write failing tests**

Cover:

- explicit `train` split artifacts are the only split used for fitting
- explicit `holdout` split artifacts are the only split used for holdout evaluation
- optional `validation` and `calibration` splits are accepted and preserved in lineage/result payloads
- mixed `.parquet` baseline plus `.xlsx`-origin prepared artifacts work when consumed as parquets
- result payload includes:
  - candidate namespace
  - manifest path and digest
  - split counts by room
  - artifact paths used per room/split
  - candidate artifact paths produced
- hard-negative contract test: fail immediately if anything reaches `UnifiedPipeline.train_from_files(...)`

**Step 2: Run focused pytest to verify RED**

Run:

- `pytest backend/tests/test_beta62_grouped_date_fit_eval.py -q`

Expected: missing module / missing runner failures.

### Task 2: Add the reusable grouped-date fit/eval module

**Files:**
- Create: `backend/ml/beta6/grouped_date_fit_eval.py`

**Step 1: Add report/manifest input loader**

Support:

- grouped-date supervised report + prepared split directory
- or manifest path + prepared split directory

Normalize:

- resident id
- target rooms
- per-room split artifact paths
- manifest path and digest

**Step 2: Add explicit split artifact reader**

Implement:

- room/split parquet discovery from prepared artifact directory or report lineage
- split counts and artifact lineage capture
- strict failure when required `train` or `holdout` artifacts are missing for a target room

**Step 3: Add candidate-only room fit/eval flow**

Implement a minimal runner that:

- preprocesses/encodes from explicit split artifacts without global chronology flattening
- fits room models from `train` only
- optionally carries `validation` / `calibration` in metadata and future hooks
- evaluates `holdout` only
- saves candidate-only outputs into a fresh namespace
- never promotes or mutates runtime default pointers

### Task 3: Add the thin CLI

**Files:**
- Create: `backend/scripts/run_beta62_grouped_date_fit_eval.py`

**Step 1: Parse report/manifest/artifact arguments**

**Step 2: Call the reusable module**

**Step 3: Write machine-readable fit/eval result JSON**

### Task 4: Re-run tests and tighten only failing behavior

**Files:**
- Modify: `backend/tests/test_beta62_grouped_date_fit_eval.py`
- Modify: `backend/ml/beta6/grouped_date_fit_eval.py`
- Modify: `backend/scripts/run_beta62_grouped_date_fit_eval.py`

**Step 1: Run focused pytest**

Run:

- `pytest backend/tests/test_beta62_grouped_date_fit_eval.py -q`

**Step 2: Fix only what the tests expose**

### Task 5: Document the permanent runner

**Files:**
- Create: `docs/reviews/2026-03-14-beta62-grouped-date-fit-eval-runner.md`
- Modify: `/Users/dickson/DT/DT_development/Development/Beta_6/dev_history.log`

**Step 1: Write the review note**

Include:

- input contract
- split semantics
- candidate-only save behavior
- explicit no-legacy-path guarantee
- what this runner still does not do in this thread (no Jessica exploratory retrain yet)

**Step 2: Append the dev history entry**

### Task 6: Final verification

**Files:**
- No additional changes expected

**Step 1: Compile checks**

Run:

- `python3 -m py_compile backend/ml/beta6/grouped_date_fit_eval.py`
- `python3 -m py_compile backend/scripts/run_beta62_grouped_date_fit_eval.py`

**Step 2: Focused pytest**

Run:

- `pytest backend/tests/test_beta62_grouped_date_fit_eval.py -q`

**Step 3: Regression slice**

Run:

- `pytest backend/tests/test_beta62_grouped_date_fit_eval.py backend/tests/test_beta62_grouped_date_supervised.py backend/tests/test_training.py -q`

**Step 4: Confirm review/log references**

Run:

- `rg -n "grouped-date fit/eval|TASK-|run_beta62_grouped_date_fit_eval" docs/reviews/2026-03-14-beta62-grouped-date-fit-eval-runner.md /Users/dickson/DT/DT_development/Development/Beta_6/dev_history.log`
