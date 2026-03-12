# Beta6 Bedroom Upstream Root-Cause Handoff

## Purpose

This handoff is for a fresh thread whose only goal is to dig out the deepest upstream root cause of the repaired Bedroom regression.

Do **not** start from the old pre-fix `v40` conclusions.
Do **not** start with another threshold sweep.
Do **not** start with another retrain until the first upstream divergence is pinned down.

## Current Branch State

- Worktree: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-livingroom-seed-forensic`
- Branch: `codex/jessica-livingroom-seed-forensic`
- Last pushed commit: `2b95cb602f30ea2aa0fae936a2e1bd24fafc9542`
- Pushed fix: permanent two-stage runtime metadata repair

Important local workspace note:

- There are unrelated in-progress local changes in:
  - `backend/export_dashboard.py`
  - `backend/elderlycare_v1_16/models/schema.sql`
  - `backend/services/label_proposal_service.py`
  - `backend/tests/test_label_proposal_service.py`
- Leave those alone unless the new thread is explicitly about Correction Studio proposal review.

## Previous Two Tasks Recap

### TASK-BETA6-JESSICA-BEDROOM-POSTFIX-RUNTIME-REBASELINE

Status: complete

What was proven:

- after the permanent runtime metadata fix, both live `Bedroom_v38` and candidate `Bedroom_v40` resolve to single-stage fallback
- the old `v40` candidate narrative was invalid because it depended on a wrong two-stage runtime path
- corrected Dec 17 Bedroom result:
  - live `v38` final macro-F1: `0.4929066675409425`
  - repaired candidate `v40` final macro-F1: `0.33052062399815013`

Primary artifacts:

- `docs/reviews/2026-03-11-beta6-bedroom-postfix-runtime-rebaseline.md`
- `tmp/bedroom_postfix_rebaseline_20260311T175000Z/comparison.json`
- `tmp/bedroom_postfix_rebaseline_20260311T175000Z/live_v38_load_check.json`
- `tmp/bedroom_postfix_rebaseline_20260311T175000Z/candidate_v40_load_check.json`

### TASK-BETA6-JESSICA-BEDROOM-POSTFIX-SINGLE-STAGE-ROOT-CAUSE

Status: complete

What was proven:

- the remaining Bedroom failure is now a single-stage model/calibration problem, not a runtime bug
- `v40` moved into a more `bedroom_normal_use`-heavy and `sleep`-lighter training regime
- `v40` also accepted much softer class-0 and class-2 thresholds, which made the bad boundary operational on Dec 17

Primary artifacts:

- `docs/reviews/2026-03-11-beta6-bedroom-postfix-single-stage-root-cause-forensic.md`
- `tmp/bedroom_postfix_single_stage_root_cause_20260311/summary.json`
- `tmp/bedroom_postfix_single_stage_root_cause_20260311/changed_rows.parquet`

## Current Decision-Grade Conclusions

These are the facts the next thread should assume unless new evidence disproves them.

### 1. The runtime bug is fixed

Pushed fix commit:

- `2b95cb602f30ea2aa0fae936a2e1bd24fafc9542`

Files in that pushed fix:

- `backend/ml/training.py`
- `backend/ml/legacy/registry.py`
- `backend/tests/test_training.py`
- `backend/tests/test_registry.py`

Verification already run:

- `pytest backend/tests/test_training.py backend/tests/test_registry.py -q`
- result: `145 passed`

### 2. The old pre-fix Bedroom `v40` replay is not valid evidence

The earlier `v40` replay and stage-A threshold-sweep path were contaminated by the runtime metadata bug and should not be used to justify promotion decisions or new calibration tuning.

Treat these as historical only:

- `docs/reviews/2026-03-11-beta6-bedroom-separation-retrain.md`
- `docs/reviews/2026-03-11-beta6-bedroom-v38-v40-root-cause-forensic.md`
- `docs/reviews/2026-03-11-beta6-bedroom-stagea-threshold-sweep.md`

### 3. The repaired Bedroom regression is real and single-stage

Corrected Dec 17 Bedroom metrics:

- truth class share:
  - `bedroom_normal_use`: `0.0715`
  - `sleep`: `0.4013`
  - `unoccupied`: `0.5272`
- live `v38` predicted share:
  - `bedroom_normal_use`: `0.1090`
  - `sleep`: `0.3862`
  - `unoccupied`: `0.4842`
  - `low_confidence`: `0.0207`
- repaired candidate `v40` predicted share:
  - `bedroom_normal_use`: `0.2761`
  - `sleep`: `0.2479`
  - `unoccupied`: `0.4423`
  - `low_confidence`: `0.0337`

Key error deltas:

- `sleep -> unoccupied`: `117 -> 832`
- `unoccupied -> bedroom_normal_use`: `435 -> 1734`
- `bedroom_normal_use -> unoccupied`: `132 -> 233`

### 4. The nearest root cause already identified is a coupled model/calibration shift

Saved single-stage traces show:

- `v38` pre-sampling train share:
  - `bedroom_normal_use`: `0.0904`
  - `sleep`: `0.4074`
  - `unoccupied`: `0.5022`
- `v40` pre-sampling train share:
  - `bedroom_normal_use`: `0.1618`
  - `sleep`: `0.3151`
  - `unoccupied`: `0.5231`

Saved thresholds:

- `v38`
  - class-0 `bedroom_normal_use`: `0.17066459956573105`
  - class-1 `sleep`: `0.8892368714230562`
  - class-2 `unoccupied`: `0.7697282402092486`
- `v40`
  - class-0 `bedroom_normal_use`: `0.0764292632392825`
  - class-1 `sleep`: `0.8249020877936664`
  - class-2 `unoccupied`: `0.5456312662762256`

Counterfactual already measured:

- all `1734` Dec 17 `unoccupied -> bedroom_normal_use` candidate false positives survive under the `v40` class-0 threshold
- only `1028` survive under the live `v38` class-0 threshold
- `0` survive under the older `v39` class-0 threshold

But the sleep collapse is not threshold-only:

- of `832` `sleep -> unoccupied` errors in repaired `v40`, `393` would still remain even if the unoccupied threshold were raised back to the live `v38` value

That means the raw decision surface already moved in the wrong direction before thresholding.

## What Is Still Unknown

This is the actual target for the new thread.

We still do **not** know the first upstream stage where the `v38` regime became the `v40` regime.

Possible upstream breakpoints:

1. raw source label topology
2. merged corrected-pack construction
3. sequence/window generation
4. train/holdout split selection
5. pre-sampling class mix
6. post-sampling / weighting
7. checkpoint selection
8. threshold calibration

Right now, the earliest proven divergence is at least by the pre-sampling class-mix stage.
The new thread should identify whether the first irreversible bad turn happened even earlier.

## Recommended Next Thread Objective

Objective:

- reconstruct the full Bedroom lineage for live `v38` and repaired candidate `v40`
- stop at the **first stage** where the data distribution or selection policy diverges enough to explain the final Dec 17 regression

This is the real upstream-root-cause investigation.

## Recommended Next Thread Plan

### Phase 1: provenance reconstruction

Rebuild the exact Bedroom lineage tables for both `v38` and `v40`:

1. raw workbook rows by day/hour/label
2. merged corrected pack rows by day/hour/label
3. generated Bedroom sequences/windows by day/hour/label
4. selected train/holdout split distributions
5. post-sampling distributions
6. checkpoint-selected holdout outputs
7. final calibrated thresholds

Deliverable:

- one comparative table per stage for `v38` vs `v40`

### Phase 2: first-divergence isolation

For each stage above, compute:

- total count by label
- share by label
- counts by date
- counts by hour
- any dedup / drop / split effects

Stop at the first stage where the shift appears strongly enough to explain:

- `bedroom_normal_use` inflation
- `sleep` erosion
- the later class-0 / class-2 threshold behavior

Deliverable:

- one explicit statement:
  - “the first irreversible divergence appears at stage X because Y”

### Phase 3: only then decide the fix

If the first divergence is:

- raw labels / merged pack:
  - root fix is data-pack or pack-builder logic
- sequence generation:
  - root fix is feature/window construction
- split selection:
  - root fix is split policy or drift objective
- post-sampling:
  - root fix is weighting / sampling policy
- checkpoint / calibration:
  - root fix is release-gate / threshold selection policy

Only after that should the new thread decide whether to:

- change training policy
- add release gates
- or launch a fresh Bedroom retrain

## Concrete Proposed First Step

Start with lineage reconstruction, not code changes.

Specifically:

1. Reconstruct `Bedroom_v38` and `Bedroom_v40` source populations from the corrected pack.
2. Export stage-by-stage comparative counts/shares for:
   - `bedroom_normal_use`
   - `sleep`
   - `unoccupied`
3. Identify the earliest stage where:
   - `bedroom_normal_use` jumps toward the `v40` regime
   - `sleep` drops toward the `v40` regime
4. Write that result before proposing any fix.

## Key Artifacts To Read First In The New Thread

### Root-fix and corrected-runtime artifacts

- `docs/reviews/2026-03-11-beta6-bedroom-two-stage-runtime-root-fix.md`
- `tmp/bedroom_two_stage_runtime_root_fix_load_check_20260311.json`
- `docs/reviews/2026-03-11-beta6-bedroom-postfix-runtime-rebaseline.md`
- `tmp/bedroom_postfix_rebaseline_20260311T175000Z/comparison.json`

### Current best Bedroom forensic

- `docs/reviews/2026-03-11-beta6-bedroom-postfix-single-stage-root-cause-forensic.md`
- `tmp/bedroom_postfix_single_stage_root_cause_20260311/summary.json`
- `tmp/bedroom_postfix_single_stage_root_cause_20260311/changed_rows.parquet`

### Saved model metadata

- `backend/models/HK0011_jessica/Bedroom_v38_decision_trace.json`
- `backend/models/HK0011_jessica/Bedroom_versions.json`
- `backend/models/HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z/Bedroom_v40_decision_trace.json`
- `backend/models/HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z/Bedroom_versions.json`

### Source pack artifact for the Bedroom-only retrain

- `tmp/jessica_bedroom_sepfix_20260311T041856Z/build_combined_corrected_pack.py`
- `tmp/jessica_bedroom_sepfix_20260311T041856Z/combined_corrected_pack.parquet`
- `tmp/jessica_bedroom_sepfix_20260311T041856Z/train_metrics.json`

### Historical context only

Read only for chronology, not for decision-making:

- `docs/reviews/2026-03-11-beta6-bedroom-separation-retrain.md`
- `docs/reviews/2026-03-11-beta6-bedroom-v38-v40-root-cause-forensic.md`
- `docs/reviews/2026-03-11-beta6-bedroom-stagea-threshold-sweep.md`

## Success Criteria For The New Thread

The new thread is successful only if it can answer:

1. What is the earliest stage where `v38` and `v40` Bedroom regimes diverge?
2. Is that divergence caused by data topology, pack construction, sequence construction, split policy, sampling, or calibration?
3. What is the smallest permanent fix that prevents that divergence from recurring?

If those three are not answered, the root cause is not fully dug out yet.

## Do Not Do These First

- do not rerun the old `v40` threshold sweep
- do not start from two-stage Bedroom routing
- do not propose a generic guard before proving the first upstream divergence
- do not launch a fresh Bedroom retrain until lineage reconstruction is complete
