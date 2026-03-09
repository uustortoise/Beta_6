# Beta 6.1 / Beta 6.2 Recovery Status

Date:
1. 2026-03-10

Recovery branch:
1. `codex/beta61-beta62-recovery-20260310`

Source branch recovered from:
1. `codex/pilot-bootstrap-gates`

## What is already in the recovered worktree

Validated Beta 6.1 areas:
1. authority preflight hardening
2. deterministic fallback / rollback resolution
3. product-facing timeline reliability and correction scorecards
4. replayable room-policy diagnostics
5. resident / home context contract

Validated Beta 6.2 areas:
1. import-boundary / SSOT guardrails
2. shared `20x14` corpus contract
3. correction-derived learning signals
4. timeline-native targets / metrics / heads
5. learning-efficiency infrastructure

## Validation snapshot

Main targeted bundle run on 2026-03-10:
1. `pytest -q backend/tests/test_health_server.py backend/tests/test_ui_services.py backend/tests/test_run_daily_analysis_beta6_authority.py backend/tests/test_t80_rollout.py backend/tests/test_beta6_active_learning.py backend/tests/test_beta6_data_manifest.py backend/tests/test_beta6_registry_v2.py backend/tests/test_beta6_serving_loader.py backend/tests/test_beta6_trainer_intake_gate.py backend/tests/test_beta6_self_supervised_pretrain.py backend/tests/test_build_pretrain_corpus_manifest_script.py backend/tests/test_run_active_learning_triage_script.py backend/tests/test_home_empty_fusion.py backend/tests/test_timeline_gates.py backend/tests/test_timeline_metrics.py backend/tests/test_timeline_targets.py backend/tests/test_transformer_timeline_heads.py`
Result:
1. `274 passed`

Additional focused checks:
1. `pytest -q backend/tests/test_run_room_experiments.py` -> `3 passed`
2. `pytest -q backend/tests/test_beta6_import_boundaries.py backend/tests/test_policy_config.py` -> `20 passed`

## Team follow-up

1. Decide whether `room_experiments` ships on this branch or moves to a dedicated diagnostics branch.
2. Run the real-environment Beta 6.1 certification entry pass with explicit PostgreSQL-backed authority settings.
3. Add at least one additional resident beyond `HK001_jessica` before promotion decisions.
4. Complete the Step 4 shadow soak window and final promotion artifacts.
5. Use the March 9 roadmap in the active docs tree as the planning baseline:
   - `docs/plans/2026-03-09-beta61-beta62-reliable-activity-timeline-design.md`
   - `docs/plans/2026-03-09-beta61-beta62-reliable-activity-timeline-execution.md`

## Scope note

1. `cloud-app/` is intentionally excluded from this recovery branch because it does not appear to be part of the Beta 6.1 / Beta 6.2 continuation surface.
