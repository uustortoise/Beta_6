# Beta6 Jessica Pre-Final Testing Execution Sheet

Use this sheet while executing the checklist in `docs/plans/2026-03-12-beta6-jessica-pre-final-testing-checklist.md`.

## Run Header

- Branch: `codex/jessica-pre-final-testing`
- Commit: `23e42f0`
- Worktree: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing`
- Tester: `Codex`
- Date: `2026-03-12`
- Candidate namespace: `HK0011_jessica_candidate_supportfix_20260310T2312Z`
- Primary replay workbook: `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_17dec2025.xlsx`
- Optional auxiliary workbooks reviewed: none

## Required File Set

Primary required replay file for pre-final testing:

- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_17dec2025.xlsx`

Optional auxiliary corrected-pack files for provenance or room-specific follow-up only:

- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_4dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_5dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_6dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_7dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_8dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_9dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_10dec2025.xlsx`

Rule:

- Do not expand the pre-final gate into a new full-pack retrain unless a room-level blocker requires it.
- The default pre-final path is candidate validation on the corrected `2025-12-17` workbook plus artifact readback.

## Preflight

- [x] branch confirmed as `codex/jessica-pre-final-testing`
- [x] commit recorded
- [x] candidate namespace confirmed as `HK0011_jessica_candidate_supportfix_20260310T2312Z`
- [x] focused regression suite passed
- [x] syntax sanity passed
- [x] candidate load sanity completed
- [x] fresh Dec 17 replay completed

Artifacts:

- Regression output: `pytest backend/tests/test_run_bedroom_root_cause_matrix.py backend/tests/test_bedroom_day_segment_audit.py backend/tests/test_training.py backend/tests/test_registry.py -q` -> `161 passed, 3 warnings in 3.04s`
- Load sanity output:
  - first load artifact: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/tmp/jessica_prefinal_candidate_supportfix_20260312T_load_sanity/load_sanity.json`
  - recheck artifact: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/tmp/jessica_prefinal_candidate_supportfix_20260312T_load_sanity_recheck/load_sanity.json`
  - result: loaded `Bathroom`, `Bedroom`, `Entrance`, `Kitchen`, `LivingRoom`; versions matched `35/38/26/27/40`; `platform.two_stage_core_models` resolved to `Bathroom` and `LivingRoom`; `Bedroom_v38_two_stage_meta.json` says `runtime_enabled=false`, so Bedroom is intentionally single-stage fallback under the repaired runtime semantics
- Dec 17 replay summary: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/tmp/jessica_17dec_eval_candidate_supportfix_prefinal_20260312T0935Z/final/comparison/summary.json`
- Additional comparison artifacts:
  - prior accepted candidate replay: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-bedroom-support-fix/tmp/jessica_17dec_eval_candidate_supportfix_20260310T2312Z/final/comparison/summary.json`
  - live promoted replay comparator: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-promotion-deepdive/tmp/jessica_17dec_eval_live_promoted_20260310T230119Z/final/comparison/summary.json`
  - Bedroom post-fix rebaseline comparator: `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-livingroom-seed-forensic/tmp/bedroom_postfix_rebaseline_20260311T175000Z/comparison.json`

## Bathroom Sign-Off

- Expected version: `v35`
- Versions file checked: `/Users/dickson/DT/DT_development/Development/Beta_6/backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z/Bathroom_versions.json` -> `current_version=35`
- Load path checked: fresh load sanity used the two-stage runtime path; `platform.two_stage_core_models` included `Bathroom`
- Dec 17 final macro-F1: `0.42003535353043925`
- Low-confidence / rewrite notes: low-confidence rate `0.004412447747329308`; `40` rewrites from raw top-1, mostly `shower -> low_confidence (38)`
- Failure-family notes: main confusion remained `unoccupied -> bathroom_normal_use (1130)` and `bathroom_normal_use -> unoccupied (357)`; metrics and low-confidence exactly matched the previously accepted support-fix/live replay
- Verdict: `pass`
- Blocking issue: none
- Artifact paths:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z/Bathroom_versions.json`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/tmp/jessica_prefinal_candidate_supportfix_20260312T_load_sanity/load_sanity.json`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/tmp/jessica_17dec_eval_candidate_supportfix_prefinal_20260312T0935Z/final/comparison/Bathroom_merged.parquet`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/tmp/jessica_17dec_eval_candidate_supportfix_prefinal_20260312T0935Z/final/comparison/summary.json`

## Bedroom Sign-Off

- Expected version: `v38`
- Versions file checked: `/Users/dickson/DT/DT_development/Development/Beta_6/backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z/Bedroom_versions.json` -> `current_version=38`
- Dec 17 final macro-F1: `0.4449128824184004`
- Low-confidence / rewrite notes: low-confidence rate `0.01773049645390071`; `171` rewrites from raw top-1, dominated by `bedroom_normal_use -> low_confidence (144)`; runtime is single-stage fallback because `/Users/dickson/DT/DT_development/Development/Beta_6/backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z/Bedroom_v38_two_stage_meta.json` says `runtime_enabled=false`
- `unoccupied -> bedroom_normal_use`: `1017`
- `sleep -> unoccupied`: `124`
- `bedroom_normal_use -> unoccupied`: `119`
- Control-plane safety artifacts checked:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/docs/reviews/2026-03-11-beta6-bedroom-root-fix-final.md`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/docs/reviews/2026-03-11-beta6-bedroom-instability-promotion-policy.md`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/docs/reviews/2026-03-11-beta6-bedroom-two-stage-runtime-root-fix.md`
- Verdict: `conditional`
- Blocking issue: no new runtime collapse was observed, but Bedroom remains fail-closed because the structural `unoccupied <-> bedroom_normal_use` regime-instability family is still large and Bedroom is intentionally not running on two-stage runtime
- Artifact paths:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z/Bedroom_versions.json`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z/Bedroom_v38_decision_trace.json`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z/Bedroom_v38_two_stage_meta.json`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/tmp/jessica_17dec_eval_candidate_supportfix_prefinal_20260312T0935Z/final/comparison/Bedroom_merged.parquet`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/tmp/jessica_17dec_eval_candidate_supportfix_prefinal_20260312T0935Z/final/comparison/summary.json`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-livingroom-seed-forensic/tmp/bedroom_postfix_rebaseline_20260311T175000Z/comparison.json`

## Kitchen Sign-Off

- Expected version: `v27`
- Versions file checked: `/Users/dickson/DT/DT_development/Development/Beta_6/backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z/Kitchen_versions.json` -> `current_version=27`
- Dec 17 final macro-F1: `0.40799361200550766`
- Drift notes: unchanged versus the previously accepted support-fix/live replay; low-confidence rate stayed `0.00011655011655011655`; main confusion remained `unoccupied -> kitchen_normal_use (778)` with no new cross-room drift signal
- Verdict: `pass`
- Blocking issue: none
- Artifact paths:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z/Kitchen_versions.json`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/tmp/jessica_17dec_eval_candidate_supportfix_prefinal_20260312T0935Z/final/comparison/Kitchen_merged.parquet`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/tmp/jessica_17dec_eval_candidate_supportfix_prefinal_20260312T0935Z/final/comparison/summary.json`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-bedroom-support-fix/tmp/jessica_17dec_eval_candidate_supportfix_20260310T2312Z/final/comparison/summary.json`

## LivingRoom Sign-Off

- Expected version: `v40`
- Versions file checked: `/Users/dickson/DT/DT_development/Development/Beta_6/backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z/LivingRoom_versions.json` -> `current_version=40`
- Dec 17 final macro-F1: `0.43397967577731184`
- Low-confidence / rewrite notes: low-confidence rate `0.0`; only `2` rewrites from raw top-1 (`livingroom_normal_use -> unoccupied`)
- `livingroom_active` recall notes: `LivingRoom_v40` saved gate evidence still reports `livingroom_active recall = 0.6956521739130435`, far from the earlier direct-retrain collapse shape
- Runtime artifact load notes: fresh load sanity included `LivingRoom` in `platform.two_stage_core_models`; `LivingRoom_two_stage_meta.json` present and runtime loaded through the two-stage core path
- Verdict: `pass`
- Blocking issue: none
- Artifact paths:
  - `/Users/dickson/DT/DT_development/Development/Beta_6/backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z/LivingRoom_versions.json`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z/LivingRoom_v40_decision_trace.json`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/tmp/jessica_prefinal_candidate_supportfix_20260312T_load_sanity/load_sanity.json`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/tmp/jessica_17dec_eval_candidate_supportfix_prefinal_20260312T0935Z/final/comparison/LivingRoom_merged.parquet`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/tmp/jessica_17dec_eval_candidate_supportfix_prefinal_20260312T0935Z/final/comparison/summary.json`
  - `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-bedroom-support-fix/tmp/jessica_17dec_eval_candidate_supportfix_20260310T2312Z/final/comparison/summary.json`

## Final Decision

- Overall verdict: `GO`
- Conditional rooms: `Bedroom`
- Blocking rooms: none
- Exact blocker summary: no room reached `block`; Bathroom, Kitchen, and LivingRoom matched accepted frontiers, while Bedroom remained operationally acceptable on the fresh Dec 17 replay but still carried the already-known regime-instability/confusion risk and intentionally stayed on single-stage fallback
- Recommended next step: carry this candidate into final pre-final sign-off with Bedroom explicitly marked `conditional`, and note that the checklist expectation about Bedroom in `platform.two_stage_core_models` is stale after the 2026-03-11 Bedroom runtime root fix

## Notes To Preserve

- Bedroom is still fail-closed, not fully robust across valid alternate regimes.
- All-room lightweight lineage observability is present, but only Bedroom has the added instability block.
- Local PostgreSQL historical-corrections access remained unavailable during the Bedroom root-fix program; carry that caveat if it affects the final risk call.
