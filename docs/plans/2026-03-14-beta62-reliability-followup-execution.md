# Beta6.2 Reliability Follow-Up Execution Plan

**Date:** 2026-03-14  
**Workspace:** `/Users/dickson/DT/DT_development/Development/Beta_6`  
**Planning branch:** `codex/pilot-bootstrap-gates`

## Goal

Turn the current Beta6.2 grouped-date work from "feature-complete but still experimental" into a decision-grade training and evaluation path that can produce reliable activity timelines without hidden legacy fallback behavior.

This document is execution-oriented. Each task is meant to be run in order and closed with explicit verification before moving on.

## Live Execution Ledger

- Status date: `2026-03-14`
- Execution branch: `codex/beta62-reliability-followup-execution`
- Execution workspace: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/beta62-reliability-followup-execution`
- Base commit: `c309523`
- Current task: `Task 2: Make validation and calibration real`
- Baseline verification on this branch:
  - `pytest backend/tests/test_beta62_grouped_date_supervised.py backend/tests/test_beta62_grouped_date_fit_eval.py -q`
  - result: `8 passed, 3 warnings in 1.09s`
- Task 1 rerun evidence:
  - workspace: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62`
  - report: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62/tmp/jessica_grouped_date_hardened_safe6_20260314T153000Z/grouped_date_fit_eval_report_rerun.json`
  - candidate namespace: `HK0011_jessica_candidate_beta62_grouped_safe6_hardened_20260314T154500Z`
  - outcome: runner exited cleanly and wrote the grouped-date fit/eval report without manual recovery
  - caveat: room `holdout_metrics` payloads currently resolve to `status=walk_forward_unavailable` / `reason=insufficient_data_for_walk_forward`, so the end-to-end path is now stable but not yet decision-grade for direct holdout summary reporting

| Task | Status | Exact reason / dependency |
| --- | --- | --- |
| Task 1: Finish the grouped-date path end-to-end | completed | Hardened grouped-date code and tests were ported onto this branch. The safe-6 rerun finished cleanly and wrote `grouped_date_fit_eval_report_rerun.json` without manual recovery. Known caveat carried forward: result `holdout_metrics` still emit `walk_forward_unavailable` summaries instead of direct holdout accuracy/F1. |
| Task 2: Make validation and calibration real | in_progress | Next reliability gap. The rerun still logs `No calibration split available ... using default threshold=0.8`, and the grouped-date result payload is not yet using explicit validation/calibration splits to produce decision-grade holdout summaries. |
| Task 3: Make timeline quality first-class | deferred | Depends on Task 2 so timeline gates attach to the real candidate result payload rather than a still-moving runner contract. |
| Task 4: Standardize room decision outputs | deferred | Depends on Task 3 so statuses can include timeline-quality blockers from the same result surface. |
| Task 5: Keep grouped-date worst-slice gating | deferred | Depends on Task 4 so worst-slice findings land inside the standardized room decision payload. |
| Task 6: Keep data governance strict | deferred | Depends on Task 1 and Task 5 so manifest lineage and grouped-date decision outputs use one settled evidence contract. |
| Task 7: Integrate replay into the same path | deferred | Depends on Task 1 through Task 6 because replay must attach to the same candidate namespace and room-status payload. |
| Task 8: Keep legacy fallback impossible | deferred | Depends on Task 1 and Task 7 so the guarded execution mode reflects the final grouped-date runner path. |
| Task 9: If Dec 4-10 Jessica labels were updated, treat that as a baseline-refresh project | deferred | Depends on Task 6 lineage governance and fresh evidence that an updated Dec batch exists. |
| Task 10: Final trusted-path gate | deferred | Depends on Tasks 1 through 9 closing with explicit completed/blocked decisions. |

## Current reality

What is already true:

1. grouped-date intake, diagnostics, prepared-split generation, and candidate-only fit/eval surfaces now exist
2. grouped-date report-only manifest handling has been hardened
3. deferred candidate artifact resolution has been hardened
4. grouped-date fit/eval now preprocesses split data per source segment instead of flattening discontinuous Dec plus Mar data into one fake continuous raw dataframe
5. strict sequence creation now skips windows that cross large timestamp gaps

What is still not yet proven end-to-end:

1. reliable use of explicit validation/calibration splits for model selection and thresholding
2. a single execution surface that emits direct holdout summary metrics, replay comparison, and room-decision outputs together
3. timeline-quality gating as a first-class candidate decision, not just secondary interpretation

What has now been proven end-to-end:

1. one clean grouped-date Jessica safe-subset run can complete without manual recovery on the hardened path
2. grouped-date preprocessing, fit, candidate save, and result JSON emission now complete in one run on the hardened path

## Reliability definition

Beta6.2 is only "reliable" when one execution path can do all of the following without manual repair:

1. load a manifest of explicit day segments
2. preserve segment lineage through split prep, fit, and evaluation
3. fit only from the intended train segments
4. evaluate only on the intended validation/calibration/holdout segments
5. emit grouped-date room findings
6. emit replay comparison against the accepted Jessica baseline
7. emit explicit room status: `pass`, `conditional`, or `block`
8. refuse silent fallback to legacy discontinuous-file flattening

## Execution order

### Task 1: Finish the grouped-date path end-to-end

**Objective:** Prove the hardened grouped-date path can complete one real candidate run without recovery workarounds.

**Required outcome:**

1. one fresh candidate namespace created through the grouped-date path
2. fit/eval CLI exits cleanly
3. result JSON is written directly by the runner
4. no manual recovery from saved artifacts is needed

**Verification:**

1. grouped-date supervised report exists
2. grouped-date fit/eval report exists
3. candidate artifacts exist under the fresh namespace
4. no runner crash or partial-result recovery note is needed

**Exit criteria:** close only when the path completes a real safe-subset run cleanly.

**Task 1 closure evidence (2026-03-14):**

1. Hardened rerun completed and wrote:
   - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62/tmp/jessica_grouped_date_hardened_safe6_20260314T153000Z/grouped_date_fit_eval_report_rerun.json`
2. Candidate namespace used:
   - `HK0011_jessica_candidate_beta62_grouped_safe6_hardened_20260314T154500Z`
3. No manual artifact-recovery step was needed after runner completion.
4. Known carry-forward caveat:
   - room `holdout_metrics` currently report `walk_forward_unavailable` because the present evaluation surface still expects walk-forward evidence; this is not a Task 1 blocker, but it is direct input to Task 2 and later reporting tasks.

### Task 2: Make validation and calibration real

**Objective:** Stop treating grouped-date `validation` and `calibration` as metadata-only placeholders.

**Required outcome:**

1. validation split is actually used for checkpoint selection / early stopping when present
2. calibration split is actually used for thresholding / confidence behavior when present
3. no misleading `val_loss unavailable` situation remains on the intended grouped-date path

**Verification:**

1. candidate report records whether validation and calibration were consumed
2. threshold/calibration artifact lineage points to explicit split segments
3. test coverage proves split-specific behavior, not just split-specific lineage

**Exit criteria:** close only when grouped-date runs can honestly say how validation and calibration influenced the candidate.

### Task 3: Make timeline quality first-class

**Objective:** Candidate decisions must be based on timeline reliability, not only classifier summary metrics.

**Required outcome:**

1. grouped-date candidate reports include fragmentation and duration behavior
2. grouped-date candidate reports include transition / boundary quality
3. class-collapse or single-label collapse checks are explicit
4. low-confidence / abstain behavior is surfaced in the candidate result

**Verification:**

1. room result payload includes timeline metrics alongside accuracy and macro-F1
2. failure reasons reference timeline quality when timeline behavior is the real blocker
3. candidate reports can explain cases like "high accuracy but unusable timeline"

**Exit criteria:** close only when activity timeline quality can block a candidate even if pooled classification metrics look acceptable.

### Task 4: Standardize room decision outputs

**Objective:** Every candidate must end in explicit room-level status, not informal interpretation.

**Required outcome:**

1. every room result ends in `pass`, `conditional`, or `block`
2. every non-pass room includes explicit reasons
3. fragile rooms such as `Bedroom`, `LivingRoom`, and any newly fragile `Bathroom` path are handled consistently

**Verification:**

1. machine-readable result payload includes room status
2. human review note includes room status with rationale
3. no candidate is described as overall `GO` while hiding a blocked room

**Exit criteria:** close only when room decision semantics are standardized across grouped-date experiments.

### Task 5: Keep grouped-date worst-slice gating

**Objective:** Prevent pooled metrics from hiding harmful dates.

**Required outcome:**

1. worst-date grouped metrics are emitted
2. harmful dates are explicitly listed
3. room-level grouped-date evidence is part of the final candidate decision

**Verification:**

1. grouped-date result payload includes per-date findings
2. harmful-date lists are reproducible from saved artifacts
3. promotion-oriented conclusions cannot ignore worst-slice failures

**Exit criteria:** close only when grouped-date evidence is treated as a hard decision surface, not just diagnostic context.

### Task 6: Keep data governance strict

**Objective:** Prevent future label/source drift from silently invalidating training conclusions.

**Required outcome:**

1. every incoming workbook batch is versioned by manifest and fingerprint
2. relabeled replacements are treated as new batch versions, not silent overwrites
3. accepted vs holdout vs quarantine decisions are recorded by date and room when needed

**Verification:**

1. manifest lineage is present in candidate reports
2. audit trail shows which raw version of each day was used
3. old baseline files remain recoverable after relabel updates

**Exit criteria:** close only when a later relabel event cannot silently rewrite historical training evidence.

### Task 7: Integrate replay into the same path

**Objective:** One grouped-date execution surface should produce candidate fit/eval plus replay comparison to the accepted Jessica baseline.

**Required outcome:**

1. grouped-date runner or its immediate companion emits replay evaluation for the accepted comparator day set
2. Dec 17 comparison is no longer stitched together by a separate manual recovery flow
3. room-by-room replay summaries are written alongside grouped-date holdout summaries

**Verification:**

1. result artifacts include holdout plus replay paths
2. replay summary is traceable to the exact candidate namespace used in grouped-date fit/eval
3. no manual post-hoc reconstruction is required

**Exit criteria:** close only when grouped-date candidate evaluation is complete in one coherent evidence bundle.

### Task 8: Keep legacy fallback impossible

**Objective:** Make it impossible to accidentally run discontinuous grouped-date experiments through the legacy flattening path.

**Required outcome:**

1. grouped-date surfaces fail loudly if they would delegate to legacy discontinuous-file flattening
2. candidate reports record the exact execution mode used
3. any unsupported path is a hard error, not a quiet fallback

**Verification:**

1. tests explicitly sentinel-fail if `UnifiedPipeline.train_from_files(...)` is reached
2. review notes state the execution mode used
3. CLI help / docs clearly separate legacy from grouped-date paths

**Exit criteria:** close only when execution-mode ambiguity is gone.

### Task 9: If Dec 4-10 Jessica labels were updated, treat that as a baseline-refresh project

**Objective:** Protect the accepted Jessica baseline from being silently overwritten.

**Honest view:**

If the team updated the `2025-12-04` to `2025-12-10` Jessica labels, that is not a small follow-up. That is a possible baseline change. Do **not** just swap the files in and continue.

**Required handling:**

1. keep the currently accepted Dec baseline pack as the frozen audit anchor
2. intake the updated Dec files as a new versioned batch with fresh fingerprints
3. compare old Dec vs new Dec by date and room
4. identify which rooms materially changed in label share / episode shape
5. re-run grouped-date diagnostics on the updated Dec pack before mixing it with March or using it as the new anchor

**Decision rule:**

1. if the updated Dec labels are minor and do not materially change room/date topology, they can become a candidate baseline refresh after verification
2. if they materially change `Bathroom`, `LivingRoom`, or `Bedroom`, then prior Jessica comparisons must be treated as anchored to the old Dec baseline and a new baseline-certification cycle is required
3. if the updated Dec pack is not clearly better and more trustworthy, keep the current accepted Dec baseline frozen

**Immediate recommendation if Dec was relabeled:**

1. pause any new March retrain decision until the updated Dec baseline question is resolved
2. validate the updated Dec pack first
3. only then decide whether March experiments should be reinterpreted against the new anchor

**Exit criteria:** close only when the team can explicitly answer which Dec baseline is authoritative.

### Task 10: Final trusted-path gate

**Objective:** Decide whether Beta6.2 is trusted for ongoing activity-timeline experimentation.

**Trusted means:**

1. grouped-date path completes cleanly
2. validation/calibration are real
3. holdout plus replay are integrated
4. room status is explicit
5. timeline metrics can block candidates
6. data lineage is fully recorded
7. no silent legacy fallback exists

**Possible final states:**

1. `trusted_for_grouped_date_experiments`
2. `usable_but_still_experimental`
3. `still_blocked`

**Exit criteria:** close only with one of those three explicit decisions.

## Honest recommendation on Dec 4-10 relabel updates

If the Dec 4-10 pack may have updated labels, I would prioritize that **before** new March retraining decisions.

Why:

1. Dec 4-10 is the accepted Jessica training anchor
2. if that anchor changed, some March conclusions may still be directionally right but no longer strictly comparable
3. baseline drift is more dangerous than one more failed experiment, because it silently changes what "better" means

So my honest order would be:

1. finish the grouped-date path hardening and one clean grouped-date candidate run
2. validate whether Dec 4-10 has changed materially
3. decide which Dec baseline is authoritative
4. only then continue broader Jessica data decisions

## Minimum verification slice for this plan

Before calling Beta6.2 reliable, require at least:

```bash
pytest backend/tests/test_beta62_grouped_date_supervised.py \
       backend/tests/test_beta62_grouped_date_fit_eval.py \
       backend/tests/test_sequence_alignment.py \
       backend/tests/test_training.py -q

python3 -m py_compile \
  backend/ml/beta6/grouped_date_supervised.py \
  backend/ml/beta6/grouped_date_fit_eval.py \
  backend/ml/sequence_alignment.py \
  backend/scripts/run_beta62_grouped_date_supervised.py \
  backend/scripts/run_beta62_grouped_date_fit_eval.py
```

And one clean grouped-date candidate run that writes:

1. grouped-date supervised report
2. grouped-date fit/eval report
3. room-by-room result note
4. replay summary against the accepted baseline
