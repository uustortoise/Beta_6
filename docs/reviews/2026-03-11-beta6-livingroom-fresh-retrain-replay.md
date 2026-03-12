# Beta6 LivingRoom Fresh Retrain Replay

## Scope

- Run one fresh LivingRoom-only retrain on the corrected Jessica pack using the root-fix branch code.
- Verify which seed the selector chooses on a brand-new multi-seed panel.
- Replay Dec 17 on the selected candidate inside an isolated namespace and compare against the current live namespace.

## Candidate namespace

- Candidate namespace: `backend/models/HK0011_jessica_candidate_livingroom_fresh_20260311T023304Z`
- Base clone: `backend/models/HK0011_jessica`
- Corrected pack:
  - `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_4dec2025.xlsx`
  - `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_5dec2025.xlsx`
  - `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_6dec2025.xlsx`
  - `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_7dec2025.xlsx`
  - `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_8dec2025.xlsx`
  - `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_9dec2025.xlsx`
  - `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_10dec2025.xlsx`
  - `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_17dec2025.xlsx`

## Fresh retrain result

Primary artifact:

- `tmp/jessica_livingroom_fresh_20260311T023304Z/train_metrics.json`

Seed-panel result:

- candidate seeds: `40, 41, 42, 43, 44, 45`
- selector requirement: at least `2` no-regress-safe seeds
- no-regress-safe seeds found: `6`
- selected seed: `45`
- selected saved version: `LivingRoom_v52`
- selection mode: `no_regress_floor`

Key selector evidence:

- seed `40` -> `v47`
  - `gate_aligned_score=0.7377`
  - `stage_a_occupancy_saturated=true`
  - `stage_a_occupancy_rate_error=0.8662`
- seed `41` -> `v48`
  - `gate_aligned_score=0.7137`
  - `stage_a_occupancy_saturated=true`
  - `stage_a_occupancy_rate_error=0.8662`
- seed `42` -> `v49`
  - `gate_aligned_score=0.6729`
  - `stage_a_occupancy_saturated=false`
  - `stage_a_occupancy_rate_error=0.0669`
- seed `43` -> `v50`
  - `gate_aligned_score=0.6964`
  - `stage_a_occupancy_saturated=false`
  - `stage_a_occupancy_rate_error=0.0669`
- seed `44` -> `v51`
  - `gate_aligned_score=0.6944`
  - `stage_a_occupancy_saturated=true`
  - `stage_a_occupancy_rate_error=0.8662`
- seed `45` -> `v52`
  - `gate_aligned_score=0.7230`
  - `stage_a_occupancy_saturated=false`
  - `stage_a_occupancy_rate_error=0.0124`

Interpretation:

- the selector did not need to fall back to the earlier `seed 42` / `v46` shape
- it found a stronger new candidate whose stage-A occupancy distortion was even lower than the earlier `0.0669` anchor
- the replay-proxy discriminator is active on a fresh retrain, not only on saved forensic artifacts

## Important evaluation note

The first Dec 17 replay against the candidate namespace was a false negative.

Why:

- `train_from_files(..., defer_promotion=True)` saved `v47..v52` but left `LivingRoom_versions.json.current_version=46`
- runtime loading still used the alias chain for `LivingRoom_v46`
- the first candidate replay therefore matched live exactly because it was still serving `v46`

Correction applied:

- rolled the isolated candidate namespace forward to `LivingRoom_v52` with `ModelRegistry.rollback_to_version(...)`
- verified `current_version=52` before rerunning replay

## Dec 17 replay

Replay artifacts:

- stale-alias replay:
  - `tmp/jessica_17dec_eval_candidate_livingroom_fresh_20260311T023304Z/final/comparison/summary.json`
- promoted-in-candidate replay:
  - `tmp/jessica_17dec_eval_candidate_livingroom_fresh_20260311T023304Z/final_v52/comparison/summary.json`
- live reference replay on the same local harness:
  - `tmp/jessica_17dec_eval_live_recheck_20260311T023304Z/final/comparison/summary.json`

### Before candidate activation (`current_version=46`)

- overall accuracy `0.8407`
- overall macro-F1 `0.4567`
- LivingRoom macro-F1 `0.4530`

This matched live exactly and should not be used as the candidate verdict.

### After candidate activation (`current_version=52`)

- overall accuracy `0.8411`
- overall macro-F1 `0.4709`
- LivingRoom accuracy `0.9236`
- LivingRoom macro-F1 `0.5004`
- Bedroom macro-F1 `0.3900`

Live reference on the same harness:

- overall accuracy `0.8407`
- overall macro-F1 `0.4567`
- LivingRoom accuracy `0.9217`
- LivingRoom macro-F1 `0.4530`
- Bedroom macro-F1 `0.3900`

Deltas vs live:

- overall accuracy `+0.0004`
- overall macro-F1 `+0.0142`
- LivingRoom accuracy `+0.0019`
- LivingRoom macro-F1 `+0.0475`
- Bedroom macro-F1 `+0.0000`

Interpretation:

- the fresh selector winner improves Dec 17 meaningfully once the candidate namespace is actually switched to the selected version
- the gain is isolated to LivingRoom; other rooms stayed unchanged on this local replay harness

## Conclusion

The fresh LivingRoom-only retrain succeeded at the branch objective:

- it produced a stable multi-seed panel
- the updated selector automatically chose a new replay-stable winner (`seed 45` / `v52`)
- that winner improved Dec 17 over the current live `v46` reference when evaluated as the active candidate

Recommended next step:

- treat this as the missing end-to-end confirmation for the selector logic
- if you want promotion readiness, the next operational move is a controlled room-wise promotion of `LivingRoom_v52` from the candidate namespace into the target namespace, followed by one final live replay confirmation
