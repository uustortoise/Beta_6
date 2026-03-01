# Beta 5.5 LivingRoom Semantic Alignment Execution (2026-02-24)

## Scope Executed
- Implemented LivingRoom direct-vs-passive supervision weighting for Stage A.
- Added stricter LivingRoom passive-hysteresis entry mode and knobs.
- Added optional episode-based go/no-go checks in matrix evaluator.
- Added semantic-alignment matrix profiles and ran quick A/B sweeps on corrected pack.

## Code Changes
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_matrix.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/config/event_first_matrix_profiles.yaml`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/config/event_first_go_no_go.yaml`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_go_no_go.py`

## Test Status
- `pytest -q tests/test_event_first_backtest_script.py tests/test_event_first_go_no_go.py tests/test_run_event_first_matrix.py tests/test_run_lr_fragmentation_sweep_clean.py tests/test_run_event_first_variant_backtest.py`
- Result: **86 passed**.

## Run: semantic_quick
- Manifest: `/tmp/beta55_lr_semantic_quick_20260224/lr_semantic_alignment_quick/clean_sweep_manifest.json`
- Ranking: `/tmp/beta55_lr_semantic_quick_20260224/lr_semantic_alignment_quick/ranking.csv`
- Matrix execution: `failed` (returncode `2`)

| Variant | Go/No-Go | Eligible | LR Eligible Passed | Blocking Reasons |
|---|---|---:|---:|---|
| anchor_top2_frag_v3 | fail | 16/20 | 0 | overall_eligible_pass_count_min;livingroom_eligible_pass_count_min;livingroom_episode_recall_min;livingroom_episode_f1_min;day7_livingroom_episode_recall_min |
| lr_semantic_align_v1 | fail | 12/20 | 0 | overall_eligible_pass_count_min;livingroom_eligible_pass_count_min;bedroom_max_regression_splits;day8_livingroom_fragmentation_min;livingroom_episode_recall_min;livingroom_episode_f1_min;day7_livingroom_episode_recall_min |
| lr_semantic_align_v2 | fail | 12/20 | 0 | overall_eligible_pass_count_min;livingroom_eligible_pass_count_min;bedroom_max_regression_splits;day8_livingroom_fragmentation_min;livingroom_episode_recall_min;livingroom_episode_f1_min;day7_livingroom_episode_recall_min |
| lr_semantic_align_v3 | fail | 12/20 | 0 | overall_eligible_pass_count_min;livingroom_eligible_pass_count_min;bedroom_max_regression_splits;day8_livingroom_fragmentation_min;livingroom_episode_recall_min;livingroom_episode_f1_min;day7_livingroom_episode_recall_min |

## Run: semantic_rescue_quick
- Manifest: `/tmp/beta55_lr_semantic_rescue_quick_20260224/lr_semantic_alignment_rescue_quick/clean_sweep_manifest.json`
- Ranking: `/tmp/beta55_lr_semantic_rescue_quick_20260224/lr_semantic_alignment_rescue_quick/ranking.csv`
- Matrix execution: `failed` (returncode `2`)

| Variant | Go/No-Go | Eligible | LR Eligible Passed | Blocking Reasons |
|---|---|---:|---:|---|
| anchor_top2_frag_v3 | fail | 16/20 | 0 | overall_eligible_pass_count_min;livingroom_eligible_pass_count_min;livingroom_episode_recall_min;livingroom_episode_f1_min;day7_livingroom_episode_recall_min |
| lr_semantic_align_v4 | fail | 16/20 | 0 | overall_eligible_pass_count_min;livingroom_eligible_pass_count_min;day7_livingroom_recall_min;day8_livingroom_fragmentation_min;livingroom_episode_recall_min;livingroom_episode_f1_min;day7_livingroom_episode_recall_min |
| lr_semantic_align_v5 | fail | 12/20 | 0 | overall_eligible_pass_count_min;livingroom_eligible_pass_count_min;bedroom_max_regression_splits;livingroom_episode_recall_min;day7_livingroom_episode_recall_min |

## Run: semantic_tune_quick
- Manifest: `/tmp/beta55_lr_semantic_tune_quick_20260224/lr_semantic_alignment_tune_quick/clean_sweep_manifest.json`
- Ranking: `/tmp/beta55_lr_semantic_tune_quick_20260224/lr_semantic_alignment_tune_quick/ranking.csv`
- Matrix execution: `failed` (returncode `2`)

| Variant | Go/No-Go | Eligible | LR Eligible Passed | Blocking Reasons |
|---|---|---:|---:|---|
| anchor_top2_frag_v3 | fail | 16/20 | 0 | overall_eligible_pass_count_min;livingroom_eligible_pass_count_min;livingroom_episode_recall_min;livingroom_episode_f1_min;day7_livingroom_episode_recall_min |
| lr_semantic_align_v6 | fail | 12/20 | 0 | overall_eligible_pass_count_min;livingroom_eligible_pass_count_min;bedroom_max_regression_splits;livingroom_episode_recall_min;livingroom_episode_f1_min;day7_livingroom_episode_recall_min |

## Outcome Summary
- No semantic-alignment candidate produced LivingRoom eligible hard-gate passes (`LR eligible passed` remained `0`).
- Aggressive hysteresis variants inflated occupied windows and caused precision collapse / fragmentation regressions.
- Conservative variants improved some soft metrics but still failed hard gates and/or caused Bedroom regression guard failures.

## Honest Recommendation
- Current LR hard-gate target (`occupied_f1 >= 0.58`) remains unattainable under present labels/objective coupling.
- Next step should be policy-level: gate re-baselining around episode reliability for LivingRoom timeline quality, or architecture-level migration to sequence-native state modeling.
- Do not run full 3-seed on these semantic candidates unless gate policy is revised first (quick screens already non-promotable).
