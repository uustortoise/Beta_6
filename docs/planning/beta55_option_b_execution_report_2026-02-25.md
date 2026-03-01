# Beta 5.5 Option B (Cross-Room Presence) Execution Report

Date: 2026-02-25
Owner: Codex execution run
Scope: End-to-end implementation + evaluation of Option B for LivingRoom passive occupancy support

## 1) What Was Implemented

Code changes were applied in `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`:

- Added LivingRoom cross-room presence decoder:
  - `_apply_livingroom_cross_room_presence_decoder(...)`
- Added CLI/config surface for Option B controls:
  - enable/disable flag
  - supporting rooms
  - hold/extension timing
  - entry/refresh thresholds
  - exit thresholds/confirm windows
  - minimum support rooms
  - optional requirement that other room is predicted occupied for exit
- Decoder is wired before arbitration and emits debug telemetry into split payload/room payload.

Matrix profiles were added in `/Users/dicksonng/DT/Development/Beta_5.5/backend/config/event_first_matrix_profiles.yaml`:

- Variants:
  - `lr_cross_room_presence_v1`
  - `lr_cross_room_presence_v2`
  - `lr_cross_room_presence_v3` (conservative tuning)
- Profiles:
  - `lr_cross_room_presence_quick`
  - `lr_cross_room_presence_full`
  - `lr_cross_room_presence_candidate_full`
  - `lr_cross_room_presence_tune_quick`
  - `lr_cross_room_presence_tune_full`

Tests added/updated in `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`:

- cross-room extension when other rooms are quiet
- exit on sustained other-room occupancy evidence

## 2) Validation Performed

Unit/integration tests:

- Command:
  - `pytest -q tests/test_event_first_backtest_script.py tests/test_run_event_first_matrix.py tests/test_run_lr_fragmentation_sweep_clean.py tests/test_run_event_first_variant_backtest.py`
- Result:
  - `86 passed`

Execution matrix (full 3 seeds, fallback clean path):

- Output root:
  - `/tmp/beta55_lr_cross_room_tune_full_timeout60_20260225/lr_cross_room_presence_tune_full/lr_cross_room_presence_tune_full`
- Ranking:
  - `/tmp/beta55_lr_cross_room_tune_full_timeout60_20260225/lr_cross_room_presence_tune_full/lr_cross_room_presence_tune_full/ranking.csv`
- Manifest:
  - `/tmp/beta55_lr_cross_room_tune_full_timeout60_20260225/lr_cross_room_presence_tune_full/lr_cross_room_presence_tune_full/clean_sweep_manifest.json`

## 3) Full-Matrix Results (Anchor vs Option B v3)

From `ranking.csv`:

- `anchor_top2_frag_v3`
  - status: fail
  - eligible: `48/60`
  - LivingRoom eligible passes: `8/12`
  - blockers:
    - `bedroom_max_regression_splits`
    - `day8_bedroom_sleep_recall_min`
    - `day8_livingroom_fragmentation_min`
    - `livingroom_episode_recall_min`
    - `livingroom_episode_f1_min`
    - `day7_livingroom_episode_recall_min`

- `lr_cross_room_presence_v3`
  - status: fail
  - eligible: `46/60`
  - LivingRoom eligible passes: `6/12`
  - blockers:
    - `bedroom_max_regression_splits`
    - `day8_bedroom_sleep_recall_min`
    - `livingroom_episode_recall_min`
    - `livingroom_episode_f1_min`

Interpretation:

- Option B v3 removed two blockers:
  - `day8_livingroom_fragmentation_min`
  - `day7_livingroom_episode_recall_min`
- But it regressed net gate outcome:
  - eligible passes `48 -> 46`
  - LivingRoom eligible passes `8 -> 6`

## 4) Key Metric Deltas (LivingRoom)

From rolling summaries:

- Occupied precision: `0.6394 -> 0.5823` (down)
- Occupied recall: `0.7484 -> 0.8541` (up)
- Occupied F1: `0.6855 -> 0.6863` (flat)
- Accuracy: `0.7163 -> 0.6774` (down)
- Macro F1: `0.7028 -> 0.6626` (down)

Timeline KPI impact:

- LivingRoom active MAE minutes: `131.13 -> 226.07` (worse by `+94.93` min)

Episode-level behavior (seed/day aggregate):

- Day 7 episode recall improved (min `0.40 -> 0.533`)
- Day 8 fragmentation improved (min `0.20 -> 0.50`)
- Episode precision/F1 dropped materially across days (precision collapse from over-hold pattern)

## 5) Honest Technical Conclusion

Option B improves recall and temporal continuity, but degrades precision enough to hurt timeline reliability.

This is the same precision-recall trap seen earlier:

- stronger hold/extension reduces false exits
- but it over-extends occupancy through true absences
- net effect: timeline MAE worsens even when recall-based checks improve

Therefore Option B v3 is **not promotion-safe** for reliable timeline output in current Beta 5.5 configuration.

## 6) Recommendation

Immediate decision:

1. Keep `top2_frag_v3` as anchor for Beta 5.5.
2. Keep Option B code path in repo as default-off experimental feature.
3. Do not promote Option B v3 to anchor.

Next practical steps:

1. If team still wants Option B iteration, gate it with strict MAE guardrail (must not worsen LR active MAE vs anchor).
2. Add an entry/exit calibration dataset focused on passive LR occupancy boundaries before further Option B tuning.
3. Prioritize label-policy/model-objective alignment workstream (episode semantics vs instantaneous window evidence) rather than more hold-threshold sweeps.

## 7) Operational Note

In this environment, matrix orchestration frequently stalls in parent subprocess communication. Reliable completion required the clean-sweep fallback route (`--matrix-timeout-seconds 60` + per-seed fallback execution). This is an execution-environment issue, not a model-logic issue.
