# Beta6 LivingRoom Root-Fix Handoff

## Previous two tasks recap

- `TASK-BETA6-JESSICA-LIVINGROOM-ROOT-FIX` alias repair and live replay revalidation
  - Promotion/rollback alias sync was missing latest `*_decision_trace.json`.
  - Registry and promotion helper were fixed so latest aliases now include `decision_trace` when the versioned source exists.
  - Live `HK0011_jessica` and the forensic worktree were both re-promoted to `LivingRoom_v46`.
  - Fresh Dec 17 replay reproduced the strong synced result: overall final accuracy `0.8138`, overall final macro-F1 `0.4504`, LivingRoom final accuracy `0.9200`, LivingRoom final macro-F1 `0.4468`, Bedroom final macro-F1 `0.3511`.

- `TASK-BETA6-JESSICA-LIVINGROOM-ROOT-FIX` selector replay-proxy scoring
  - Multi-seed candidate summaries now expose a LivingRoom-only `stage_a_occupancy_rate_error` derived from `metrics["two_stage_core"]["stage_a_calibration"]`.
  - Selector ordering now prefers lower stage-A occupancy distortion after the existing non-collapsed / gate-pass / no-regress / non-saturated checks.
  - Saved `v44/v45/v46` selector replay artifact still chooses `v46` / seed `42`.

## Current branch status

- Branch: `codex/jessica-livingroom-seed-forensic`
- Status: implementation checkpoint complete and verified locally; not yet merged
- Intent of this branch:
  - stabilize LivingRoom training selection without reopening the older downsample thread
  - keep `LivingRoom_v46` as the current promotion-grade reference
  - make future LivingRoom retrains select `v46`-like seeds automatically instead of requiring manual forensic review

## What changed

### Policy / training

- `backend/config/beta6_policy_defaults.yaml`
  - adds `livingroom` to default `post_split_shuffle_rooms`
  - adds LivingRoom room-level minimum passing-seed requirement

- `backend/ml/policy_defaults.py`
- `backend/ml/policy_config.py`
  - expose the room-level minimum seed-panel no-regress pass count through policy loading

- `backend/ml/training.py`
  - adds LivingRoom stage-A saturation detection
  - adds LivingRoom stage-A occupancy-rate error extraction
  - records both signals in seed-panel candidate summaries
  - marks unstable LivingRoom panels fail-closed when they do not meet the multi-seed contract
  - selector ranking now prefers lower stage-A occupancy distortion before raw holdout score

### Registry / promotion

- `backend/ml/legacy/registry.py`
  - latest alias sync / cleanup now handles `*_decision_trace.json` as an optional alias when the versioned source exists

- `backend/scripts/promote_room_versions_from_namespace.py`
  - promotion summaries now report `*_decision_trace.json` among latest artifacts

### Tests

- `backend/tests/test_policy_config.py`
  - covers LivingRoom shuffle and room-level seed-panel requirements

- `backend/tests/test_promote_room_versions_from_namespace.py`
  - proves room promotion materializes the latest `*_decision_trace.json` alias

- `backend/tests/test_training.py`
  - covers:
    - LivingRoom post-split shuffle
    - unstable LivingRoom panel rejection
    - stable LivingRoom panel acceptance
    - non-saturated seed preference
    - lower `stage_a_occupancy_rate_error` preference
    - summary emission of `stage_a_occupancy_rate_error`

## Key evidence artifacts

- Live repaired replay summary:
  - `tmp/jessica_17dec_eval_live_livingroom_rootfix_registryfix_20260311/final/comparison/summary.json`

- Saved selector replay summaries:
  - `tmp/jessica_livingroom_rootfix_pack_20260311T004042Z/selector_postfix_summary.json`
  - `tmp/jessica_livingroom_rootfix_pack_20260311T004042Z/selector_replay_proxy_summary.json`

- Live promotion repair summaries:
  - `tmp/jessica_live_livingroom_rootfix_registryfix.json`
  - `tmp/jessica_live_livingroom_rootfix_registryfix_main_workspace.json`

## Verified results

- `pytest backend/tests/test_registry.py backend/tests/test_promote_room_versions_from_namespace.py -q`
  - `31 passed`

- `pytest backend/tests/test_policy_config.py backend/tests/test_training.py -q`
  - `129 passed`

- `pytest backend/tests/test_training.py -q`
  - `113 passed`

- Selector replay over saved `v44/v45/v46`
  - `v44`: `stage_a_occupancy_saturated=true`, `stage_a_occupancy_rate_error=0.8662`
  - `v45`: `stage_a_occupancy_saturated=true`, `stage_a_occupancy_rate_error=0.8662`
  - `v46`: `stage_a_occupancy_saturated=false`, `stage_a_occupancy_rate_error=0.0669`
  - selected candidate remains `v46` / seed `42`

## Live model state

- Main workspace live namespace:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/backend/models/HK0011_jessica`

- LivingRoom latest alias now points at `saved_version: 46` in:
  - `backend/models/HK0011_jessica/LivingRoom_decision_trace.json`

- Fresh replay result on Dec 17:
  - overall final accuracy `0.8138`
  - overall final macro-F1 `0.4504`
  - LivingRoom final accuracy `0.9200`
  - LivingRoom final macro-F1 `0.4468`
  - Bedroom final macro-F1 `0.3511`

## Recommended next step

Do not start with a full all-room rerun.

Run one fresh LivingRoom-only retrain on the corrected Jessica pack using this branch code, then:

1. inspect the selected seed-panel artifact
2. confirm the selector now picks a `v46`-like candidate automatically
3. replay Dec 17 on that new candidate
4. only consider an all-room rerun if the LivingRoom-only retrain stays strong

## Risks / open points

- This branch improves training-time LivingRoom seed selection, but it does not yet prove the selector on a brand-new retrain end-to-end.
- The live namespace is already repaired and pointing at `v46`, so the main remaining uncertainty is future retrain reproducibility, not current deployed behavior.
