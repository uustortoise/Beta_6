# Bedroom Forensic Handoff - Separation Retrain In Progress

Date: 2026-03-11
Worktree: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-livingroom-seed-forensic`
Branch: `codex/jessica-livingroom-seed-forensic`

## Current status

- Bedroom-only retrain is still running as PID `910`.
- Latest checked process state when this note was written:
  - `etime=43:21`
  - `%cpu=139.4`
  - `%mem=15.1`
  - `stat=R`
- Active command:
  - `python3 tmp/jessica_bedroom_sepfix_20260311T041856Z/run_bedroom_sepfix_retrain.py`
- No new Bedroom version has been saved yet.
- No `status.json` or final exit-code artifact has been written yet for the current combined-pack run.

## Why this retrain was started

Bedroom policy-only ablation already showed runtime policy is not the main fix path:

- canonical final benchmark stayed `0.3511`
- removing `low_confidence` lifted benchmark macro-F1 mechanically but did not improve the substantive 3-label Bedroom macro-F1
- dominant error remained `unoccupied -> bedroom_normal_use`

That pointed to model-side occupied-vs-unoccupied separation as the next narrow experiment.

## Retrain hypothesis being tested

Keep the current good Bedroom support-fix posture, but remove Bedroom-only class-0 oversampling pressure:

- `TRANSITION_FOCUS_MAX_MULTIPLIER_BY_ROOM=bedroom:1`
- `MINORITY_MAX_MULTIPLIER_BY_ROOM=bedroom:1`

This is intended to reduce artificial inflation of `bedroom_normal_use` without reopening:

- LivingRoom work
- the old Bedroom support-gating bug
- all-room retraining

## Important artifacts

Prior forensic outputs:

- `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-livingroom-seed-forensic/docs/reviews/2026-03-11-beta6-bedroom-dec17-forensic.md`
- `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-livingroom-seed-forensic/docs/reviews/2026-03-11-beta6-bedroom-policy-ablation.md`

Retrain design/plan:

- `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-livingroom-seed-forensic/docs/plans/2026-03-11-beta6-bedroom-separation-retrain-design.md`
- `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-livingroom-seed-forensic/docs/plans/2026-03-11-beta6-bedroom-separation-retrain.md`

Run directory for this experiment:

- `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-livingroom-seed-forensic/tmp/jessica_bedroom_sepfix_20260311T041856Z`

Key files in that run directory:

- combined corrected pack:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-livingroom-seed-forensic/tmp/jessica_bedroom_sepfix_20260311T041856Z/combined_corrected_pack.parquet`
- current retrain harness:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-livingroom-seed-forensic/tmp/jessica_bedroom_sepfix_20260311T041856Z/run_bedroom_sepfix_retrain.py`
- combined-pack builder:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-livingroom-seed-forensic/tmp/jessica_bedroom_sepfix_20260311T041856Z/build_combined_corrected_pack.py`
- active live log:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-livingroom-seed-forensic/tmp/jessica_bedroom_sepfix_20260311T041856Z/train_combined.log`

Candidate model namespace:

- `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-livingroom-seed-forensic/backend/models/HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z`

## What has already been proven

1. Raw Bedroom loss is mainly model-side, not runtime-policy-side.
2. Bedroom already had restored near-neutral room-label weighting in `Bedroom_v38`.
3. The remaining suspicious training bias was Bedroom class-0 oversampling pressure.
4. Switching from 8 workbook inputs to 1 combined parquet removed the ingest bottleneck and allowed the retrain to get well into the actual model fit.

## Current run details

The active run now uses the single combined corrected pack instead of reading 8 workbooks directly.

Training posture confirmed in log:

- Bedroom only
- unchanged rooms skipped
- class weight posture still:
  - `bedroom_normal_use`: `2.0602`
  - `sleep`: `1.6927`
  - `unoccupied`: `0.6372`
- oversampling rollback env overrides active

Observed Stage-A training result from the log:

- epoch 1: `accuracy=0.8111`, `val_accuracy=0.7461`, `loss=0.3975`, `val_loss=1.1785`
- epoch 2: `accuracy=0.8834`, `val_accuracy=0.7038`, `loss=0.2520`, `val_loss=1.5064`
- epoch 3: `accuracy=0.9054`, `val_accuracy=0.7402`, `loss=0.2117`, `val_loss=1.6352`
- early stopping triggered at epoch 3
- restored best weights from epoch 1

After that, the run advanced into the downstream two-stage path. Latest concrete downstream log line:

- `Two-stage stage-A occupancy threshold for Bedroom: 0.9500 (target_met, support=1538)`

No final save/decision-trace/version-registration line has appeared yet.

## Environment caveats seen in the log

- PostgreSQL auth for user `postgres` is failing:
  - `FATAL: password authentication failed for user "postgres"`
- Training is continuing despite that.
- As a result, the run is not using historical Golden Samples augmentation.

This is not new model evidence. It is an environment issue.

## Stale-process cleanup already done

- There had been an older stale retrain process from an earlier attempt.
- That stale competing Python process was killed.
- The current live process to watch is PID `910`.

## What to do next

Wait for the current retrain to finish. Then do exactly this:

1. Read the current run result artifacts if they exist:
   - `tmp/jessica_bedroom_sepfix_20260311T041856Z/status.json`
   - `tmp/jessica_bedroom_sepfix_20260311T041856Z/train_metrics.json`
   - `tmp/jessica_bedroom_sepfix_20260311T041856Z/exit_code_combined.txt`

2. Check whether a new Bedroom version was saved in:
   - `backend/models/HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z`

3. If a new Bedroom version exists and metrics are sane:
   - activate that saved Bedroom version in the candidate namespace
   - keep other rooms untouched

4. Run the canonical Dec 17 Bedroom replay/evaluation on the same branch harness.

5. Compare against:
   - canonical final Bedroom `0.3511`
   - prior `Bedroom_v38` behavior
   - dominant `unoccupied -> bedroom_normal_use` error count

6. Write the final forensic note only after that replay.

## Honest recommendation

Do not start a new retrain. The current combined-pack Bedroom-only retrain is the right experiment and it is genuinely running.

The honest next step is:

- let PID `910` finish
- then evaluate only that result on Dec 17

If this retrain still leaves the same dominant `unoccupied -> bedroom_normal_use` confusion, I would stop spending time on runtime calibration and move directly to a Bedroom-only data/model experiment targeted at occupied-vs-unoccupied separation.

## Suggested prompt for the new thread

```text
Continue the Bedroom forensic retrain thread in worktree /Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-livingroom-seed-forensic on branch codex/jessica-livingroom-seed-forensic.

Read handoff note:
- docs/reviews/2026-03-11-beta6-bedroom-retrain-handoff.md

Current state:
- Active Bedroom-only retrain is still running as PID 910
- Live log: tmp/jessica_bedroom_sepfix_20260311T041856Z/train_combined.log
- Run dir: tmp/jessica_bedroom_sepfix_20260311T041856Z
- Candidate namespace: backend/models/HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z

Context:
- This retrain tests Bedroom-only rollback of class-0 oversampling pressure:
  - TRANSITION_FOCUS_MAX_MULTIPLIER_BY_ROOM=bedroom:1
  - MINORITY_MAX_MULTIPLIER_BY_ROOM=bedroom:1
- It uses a single prebuilt corrected-pack parquet, not the 8 xlsx files directly.
- Do not start another retrain unless this one fails conclusively.

Your job:
1. Check whether PID 910 has finished.
2. If finished, inspect status.json / train_metrics.json / exit_code_combined.txt.
3. Determine whether a new Bedroom version was saved.
4. If yes, activate that saved Bedroom version in the candidate namespace.
5. Run the canonical Dec 17 Bedroom replay/evaluation.
6. Write a concise forensic note with outcome, evidence, and recommendation.

Important:
- Do not reopen LivingRoom work.
- Do not rerun all rooms.
- Do not treat runtime policy as the leading fix path unless the new replay contradicts the earlier policy-ablation evidence.
```
