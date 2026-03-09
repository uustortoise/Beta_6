# Beta 6.1 / Beta 6.2 Reliable Activity Timeline Execution

Promoted to the active `docs/plans/` tree on 2026-03-10 after recovery of the frozen `beta6.1/6.2` thread.

Planning baseline:
1. `codex/pilot-bootstrap-gates` at `94524af`

Validated current branch state:
1. local branch `codex/pilot-bootstrap-gates` at `6794c6c`
2. targeted Beta 6.1 / Beta 6.2 suites pass on the recovered working tree

## Goal

Execute Beta 6.1 as the productionization track for reliable authority-grade activity timelines, while executing Beta 6.2 as the isolated correction-reduction and learning-efficiency track.

## Baseline Assumptions

1. LivingRoom reliability and review-loop work have already landed.
2. Beta 6.1 is no longer primarily blocked by Bedroom/Entrance model failures.
3. Beta 6.2 should maximize model-first improvement before escalating to human correction.
4. `origin/codex/livingroom-fast-diagnosis` is evidence for replay diagnostics, not a replacement baseline.

## Status Ledger

### Beta 6.1

1. Task 1: Rebaseline the Beta 6.1 source of truth
Status: complete
Notes: the March 9 roadmap existed only under the nested `Development/Beta_6/docs/plans/` mirror. It is now promoted into the active top-level `docs/plans/` tree.

2. Task 2: Make Beta 6.1 authority preflight explicit and production-safe
Status: implemented locally and validated
Validation:
`pytest -q backend/tests/test_run_daily_analysis_beta6_authority.py backend/tests/test_health_server.py`

3. Task 3: Make fallback and rollback deterministic for every room
Status: implemented locally and validated
Validation:
`pytest -q backend/tests/test_beta6_registry_v2.py backend/tests/test_t80_rollout.py backend/tests/test_beta6_serving_loader.py`

4. Task 4: Publish product-facing timeline reliability and correction-load scorecards
Status: implemented locally and validated
Validation:
`pytest -q backend/tests/test_ui_services.py backend/tests/test_health_server.py`

5. Task 4A: Add replayable room-policy diagnostics before default retunes
Status: implemented locally and validated
Validation:
`pytest -q backend/tests/test_run_room_experiments.py backend/tests/test_policy_config.py`

6. Task 5: Add the Beta 6.1 resident/home context contract
Status: implemented locally and validated
Validation:
`pytest -q backend/tests/test_home_empty_fusion.py backend/tests/test_health_server.py`

7. Task 6: Run the real-environment Beta 6.1 certification entry pass
Status: pending external environment
Blocked by:
1. PostgreSQL-backed runtime execution in the real environment
2. additional resident coverage beyond `HK001_jessica`
3. shadow soak evidence and final promotion artifacts

### Beta 6.2

8. Task 7: Establish a Beta 6.2 SSOT namespace
Status: implemented locally and validated
Validation:
`pytest -q backend/tests/test_beta6_import_boundaries.py`

9. Task 8: Build the shared `20x14` corpus contract
Status: implemented locally and validated
Validation:
`pytest -q backend/tests/test_build_pretrain_corpus_manifest_script.py backend/tests/test_beta6_data_manifest.py backend/tests/test_beta6_trainer_intake_gate.py`

10. Task 9: Turn corrections into structured training signal
Status: implemented locally and validated
Validation:
`pytest -q backend/tests/test_beta6_active_learning.py backend/tests/test_run_active_learning_triage_script.py backend/tests/test_ui_services.py`

11. Task 10: Add timeline-native targets and heads
Status: implemented locally and validated
Validation:
`pytest -q backend/tests/test_transformer_timeline_heads.py backend/tests/test_timeline_targets.py backend/tests/test_timeline_metrics.py backend/tests/test_timeline_gates.py`

12. Task 11: Add context-conditioned modeling before demographic modeling
Status: implemented locally and validated
Validation:
covered by targeted `home_empty_fusion` and timeline-head suites on the recovered branch

13. Task 12: Improve learning efficiency and experiment throughput
Status: implemented locally and validated
Validation:
`pytest -q backend/tests/test_beta6_self_supervised_pretrain.py backend/tests/test_beta6_data_manifest.py backend/tests/test_beta6_trainer_intake_gate.py backend/tests/test_run_room_experiments.py`

## Recovered Validation Snapshot

Validated together on 2026-03-10:
1. `pytest -q backend/tests/test_health_server.py backend/tests/test_ui_services.py backend/tests/test_run_daily_analysis_beta6_authority.py backend/tests/test_t80_rollout.py backend/tests/test_beta6_active_learning.py backend/tests/test_beta6_data_manifest.py backend/tests/test_beta6_registry_v2.py backend/tests/test_beta6_serving_loader.py backend/tests/test_beta6_trainer_intake_gate.py backend/tests/test_beta6_self_supervised_pretrain.py backend/tests/test_build_pretrain_corpus_manifest_script.py backend/tests/test_run_active_learning_triage_script.py backend/tests/test_home_empty_fusion.py backend/tests/test_timeline_gates.py backend/tests/test_timeline_metrics.py backend/tests/test_timeline_targets.py backend/tests/test_transformer_timeline_heads.py`
Result:
`274 passed`

Additional focused validation:
1. `pytest -q backend/tests/test_run_room_experiments.py`
Result:
`3 passed`

2. `pytest -q backend/tests/test_beta6_import_boundaries.py backend/tests/test_policy_config.py`
Result:
`20 passed`

## Next Actions

1. Commit and push the recovered Beta 6.1 / Beta 6.2 branch on a new remote branch, not by mutating the frozen reference branch directly.
2. Decide whether `room_experiments` belongs on that branch or on a dedicated follow-up diagnostics branch.
3. Run the real-environment certification entry pass once PostgreSQL-backed runtime access and the intended evidence profile are available.
4. Add at least one additional resident beyond `HK001_jessica`.
5. Complete the Step 4 shadow soak window and final promotion artifacts.
