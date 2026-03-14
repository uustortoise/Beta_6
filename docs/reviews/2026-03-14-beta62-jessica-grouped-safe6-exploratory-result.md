# Beta6.2 Jessica Grouped Safe-6 Exploratory Result

- Date: 2026-03-14
- Workspace: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62`
- Branch: `codex/jessica-march-grouped-supervised-beta62`

## Candidate

- Candidate namespace:
  - `HK0011_jessica_candidate_beta62_grouped_safe6_20260314T103500Z`
- Safe March subset used:
  - `2026-02-20`
  - `2026-03-02`
  - `2026-03-06`
  - `2026-03-07`
  - `2026-03-08`
  - `2026-03-09`
- Holdout-only March dates:
  - `2026-02-18`
  - `2026-03-01`
  - `2026-03-03`
  - `2026-03-04`
  - `2026-03-05`

## What happened

The grouped-date fit/eval runner trained and saved the candidate namespace, but the CLI did not complete cleanly:

1. `--supervised-report` mode failed because the embedded supervised report manifest did not include `resident_id`.
2. `--manifest` mode trained all four rooms, but final holdout evaluation failed while resolving saved LivingRoom artifacts.

Because the candidate artifacts were already written to disk, the run result was recovered directly from the saved namespace plus replay/holdout evaluation against:

- corrected `2025-12-17`
- the five excluded March holdout dates

Recovery artifacts:

- `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62/tmp/jessica_grouped_fit_eval_recovery_20260314T060119Z/exploratory_summary.json`
- `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62/tmp/jessica_grouped_fit_eval_recovery_20260314T060119Z/dec17/final/comparison/summary.json`
- `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-march-grouped-supervised-beta62/tmp/jessica_grouped_fit_eval_recovery_20260314T060119Z/holdout_summary.json`

## Result

### Dec 17 replay vs signed-off baseline

- grouped safe-6 candidate:
  - accuracy: `0.8012456868822738`
  - macro-F1: `0.3491946334062444`
- signed-off pre-final baseline:
  - accuracy: `0.8417598841689825`
  - macro-F1: `0.4663352546942085`
- delta:
  - accuracy: `-0.040514197286708686`
  - macro-F1: `-0.1171406212879641`

### Excluded-date March holdout

- overall accuracy: `0.7118765970634499`
- overall macro-F1: `0.21324745388665461`

### Dec 17 room highlights

- `Bathroom`:
  - accuracy `0.7693915466790525`
  - macro-F1 `0.3519500176128205`
- `Bedroom`:
  - accuracy `0.8250591016548463`
  - macro-F1 `0.4407023759822503`
- `Kitchen`:
  - accuracy `0.6982517482517483`
  - macro-F1 `0.22532721528509103`
  - low-confidence rate `0.2925407925407925`
- `LivingRoom`:
  - accuracy `0.9131757547390592`
  - macro-F1 `0.3182059123343527`
  - only one predicted final label on replay despite three truth labels

## Decision

Do not continue this safe-6 path as a release-oriented Jessica candidate.

Why:

- it is materially worse than the signed-off pre-final baseline on Dec 17
- `Kitchen` regressed badly
- `LivingRoom` is still structurally weak despite high raw accuracy
- excluded-date March holdout remains poor overall

## Important runner limitations discovered

1. `grouped_date_fit_eval.py` still needs a fix for `--supervised-report` input mode.
2. `grouped_date_fit_eval.py` still needs a robust saved-artifact resolution path when evaluating a reused namespace.
3. the grouped-date fit/eval path is better than legacy cross-month add-back, but the underlying room preprocessing still resamples each split across its whole discontinuous span before unresolved-gap rows are dropped. That approximation should be treated cautiously for future work.

## Recommendation

- keep the signed-off Jessica pre-final candidate frozen
- do not promote or continue this grouped safe-6 candidate
- fix the two grouped-date fit/eval runner bugs before the next experiment
- if March work continues, target room-scoped investigation rather than another broad mixed-pack attempt
