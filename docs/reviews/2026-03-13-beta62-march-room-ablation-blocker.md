# Beta6.2 March Room-Ablation Blocker Review

## Scope

- Branch/worktree: `codex/jessica-march-room-ablation-beta62`
- Base: `codex/beta61-beta62-execution`
- Goal: continue the next recommended March investigation using `Bathroom`-only and `LivingRoom`-only date ablations on top of the accepted Jessica Dec baseline
- Intended path: use the finished Beta6.1/6.2 line, not the signed-off Beta6 pre-final release branch

## What Was Attempted

- Verified the finished Beta6.1/6.2 room-experiment surface:
  - `pytest backend/tests/test_run_room_experiments.py backend/tests/test_beta6_trainer_intake_gate.py backend/tests/test_beta6_timeline_hard_gates.py -q`
  - result: `23 passed, 3 warnings`
- Created an isolated March ablation worktree from `codex/beta61-beta62-execution`
- Began a room-scoped ablation matrix using the legacy `UnifiedPipeline.train_from_files(..., rooms={...}, defer_promotion=True)` path

## Root-Cause Finding

The legacy aggregated training path is not a valid tool for single-date March add-back ablations.

### Finding 1: Single late-date add-backs do not enter the training window

Probe artifact:

- `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-room-ablation-beta62/tmp/beta62_room_ablation_gap_probe_20260313T222230Z.json`

For `Bathroom` with accepted Dec baseline plus only `2026-02-20`:

- `march_rows_total = 8639`
- `march_rows_in_train_window = 0`
- `march_rows_in_val_calib_window = 8639`
- `train_end_timestamp = 2025-12-10 09:36:00`
- `march_start_timestamp = 2026-02-20 00:00:10`

Interpretation:

- the added March day is entirely outside the train split
- a “single-date add-back training ablation” would actually be a validation/calibration-only probe
- any conclusion framed as “training on 2026-02-20 helped or hurt” would be technically wrong

### Finding 2: The combined Dec+March path still spans a multi-month synthetic gap

Probe artifact:

- `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-room-ablation-beta62/tmp/beta62_room_ablation_gap_probe_allcleared7_20260313T222230Z.json`

For `Bathroom` with accepted Dec baseline plus all seven cleared March dates:

- `max_gap_seconds = 6134410.0`
- `first_timestamp = 2025-12-04 00:00:10`
- `last_timestamp = 2026-03-10 00:00:00`
- `march_rows_in_train_window = 36287`
- `march_rows_in_val_calib_window = 24191`

Interpretation:

- the combined safe-subset pack does place some March rows into training
- but it still relies on a global chronological split across a two-month discontinuous span
- that makes single-date or small-group add-back interpretation fragile, because date placement in train vs validation is dominated by chronology rather than controlled grouped-date assignment

## Practical Consequence

Do **not** use legacy `train_from_files` cross-month add-back experiments as the decision-grade path for March data.

This affects the previously planned next step:

- `Bathroom`-only March single-date add-backs are not valid with this path
- `LivingRoom`-only March single-date add-backs are not valid with this path

The issue is methodological, not just performance:

- the experiment does not actually train on the late add-back date in the single-date case
- therefore it cannot answer the question it was supposed to answer

## What Remains Valid

- the Beta6.1/6.2 integration branch is in good shape and already pushed for review:
  - PR: `https://github.com/uustortoise/Beta_6/pull/2`
- the March label-side findings still stand:
  - `LivingRoom` requires more relabel/review on `2026-02-18`, `2026-03-01`, `2026-03-04`, and `2026-03-05`
- the Beta6 release line should remain frozen on the signed-off pre-final candidate while March work continues

## Recommended Next Step

Use a date-safe Beta6.2 path for the next March investigation:

1. consume March files through the governed intake / manifest path
2. build grouped-by-date diagnostics from canonical manifests
3. run `Bathroom` and `LivingRoom` investigations with grouped-date evaluation rather than legacy global chronological add-backs
4. only after that, decide whether any March subset is safe for exploratory retraining

If a legacy-path fallback is required, it needs an explicit date-aware split or per-source-segment preservation before it can support valid add-back experiments.

## Caveat

- PostgreSQL historical-corrections access was still unavailable in this sandbox during the attempted ablation run
- that caveat is secondary here; the main blocker is the split semantics, not the lack of corrections
