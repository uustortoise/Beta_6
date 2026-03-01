# Timeline Objective Alignment Execution Plan (Centralized)

## Distribution-Ready Playbook
- Detailed executable plan for team distribution is maintained at:
  - `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/timeline_reliability_team_execution_playbook.md`

## Document Control
- Date: February 17, 2026
- Scope: `/Users/dicksonng/DT/Development/Beta_5.5`
- Branch context: `beta-5.5-transformer`
- Baseline line: `v31` (frozen for comparative evidence)
- Purpose: single source of truth for timeline-quality upgrade from event-pass to production-ready timeline output.

## 1. Problem Statement and Goal
Current system can pass event-level checks, but timeline quality is not consistently promotable due to:
- boundary timing drift,
- segment fragmentation,
- over/under-extended episodes,
- room-specific unoccupied ambiguity.

Primary objective:
- optimize and gate for timeline correctness (episode start/end/duration/count) while preserving critical event safety behavior.

## 2. Non-Negotiable Engineering Principles
1. No leakage: all fit/transform/statistics must be train-split-only.
2. Reproducible evidence: every signoff run must be fully traceable.
3. Stability over lucky wins: promotion depends on split/seed robustness, not best-case run.
4. Safe rollout: shadow-first deployment before any promotion-path change.
5. Backward compatibility: existing production path remains safe-default unless feature flags are enabled.

## 3. Canonical Evaluation Contract
### 3.1 Fixed rolling protocol
- Splits: `4->5`, `4+5->6`, `4+5+6->7`, `4+5+6+7->8` (extend through available days when present).
- Seeds: `11, 22, 33`.
- Same room set and same manifest definition across baseline and candidate.

### 3.2 Leakage-control appendix (mandatory per run)
For each split:
- normalization/scaler/imputer/feature-stat fit on train only.
- no target-derived stats from validation/holdout.
- no decoder threshold tuning on holdout.
- no calibration fit on holdout; calibration uses train-internal calibration slice only.
- explicit leakage audit artifact required (`leakage_audit.json`).

### 3.3 Required robustness criteria
Promotion checks must include all:
- hard safety gates pass on all seeds.
- at least 80% of split-seed cells pass timeline gates.
- mean metric improvement threshold met versus baseline.
- variance ceiling met (std thresholds; see Section 8).
- no single split catastrophic regression beyond tolerance.

## 4. Reproducibility and Artifact Contract
Every signoff artifact must include:
- `git_sha`
- `config_hash`
- `baseline_version`
- `baseline_artifact_hash`
- `data_version` (manifest hash or snapshot id)
- `feature_schema_hash`
- `model_hash`
- `run_timestamp`
- `seed`
- `split_id`

Required files:
- `..._rolling.json`
- `..._signoff.json`
- `..._residual_pack.json`
- `..._residual_windows.csv`
- `..._leakage_audit.json`

## 5. Workstream Plan

### 5.0 Mandatory Inclusion Scope (Locked)
The following items are mandatory and must be treated as explicit release scope, not optional enhancements:
1. Full multi-task timeline heads integrated into production training path.
2. Calibrated Decoder v2 for robust episode reconstruction.
3. Stronger timeline gates for promotion decisions.
4. Shadow -> Canary -> Full rollout with strict pass criteria and rollback controls.

Implementation anchors:
- Training path integration: `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/training.py`, `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/pipeline.py`, `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/unified_training.py`.
- Timeline heads: `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/transformer_timeline_heads.py`.
- Decoder + calibration: `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/timeline_decoder_v2.py`, `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/calibration.py`.
- Promotion gates/signoff: `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/timeline_gates.py`, `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/aggregate_event_first_backtest.py`, `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/validation_run.py`.
- Rollout controller: `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/t80_rollout_manager.py`.

### WS-0: Baseline Freeze + Reproducibility (1 day)
Objective:
- Freeze comparison baseline and bind every candidate run to it.

Files:
- `backend/scripts/aggregate_event_first_backtest.py`
- `backend/tests/test_event_first_backtest_aggregate.py`

Deliverables:
- baseline fields emitted and included in `config_hash`.
- baseline artifacts immutable and checksum-verified.

Done criteria:
- all aggregate tests pass.
- reruns with same inputs produce byte-stable output payloads except timestamps.

---

### WS-1: Timeline Target Builder (2 days)
Objective:
- generate deterministic, leakage-safe timeline targets.

Files:
- `backend/ml/timeline_targets.py`
- `backend/tests/test_timeline_targets.py`
- `backend/scripts/run_event_first_backtest.py`

Deliverables:
- start/end boundary targets.
- episode duration/count targets excluding `unoccupied` and `unknown` by policy.
- deterministic behavior across repeated runs.

Done criteria:
- deterministic tests pass.
- leakage validation tests pass.
- transition semantics validated: excluded->care (start), care->excluded (end), care->care (end+start).

---

### WS-2: Multi-Task Objective (CNN+Transformer) (3-4 days)
Objective:
- align loss with timeline quality, not only per-window class accuracy.

Files:
- `backend/ml/training.py`
- `backend/ml/transformer_backbone.py`
- `backend/ml/transformer_timeline_heads.py`
- `backend/ml/transformer_head_ab.py`
- `backend/tests/test_training.py`
- `backend/tests/test_transformer_timeline_heads.py`

Deliverables:
- shared encoder + heads: activity, occupancy, boundary-start, boundary-end, daily-duration, daily-count.
- weighted loss with policy-driven defaults.
- safe flag: `ENABLE_TIMELINE_MULTITASK`.

Done criteria:
- feature-flag OFF: zero behavior change.
- feature-flag ON: trains and exports artifacts.
- CPU CI graph compile + shape tests pass.

---

### WS-3: Decoder v2 + Calibration (2-3 days)
Objective:
- reconstruct stable episodes from probabilistic outputs.

Files:
- `backend/ml/timeline_decoder_v2.py`
- `backend/ml/event_decoder.py`
- `backend/ml/calibration.py` (create if needed)
- `backend/tests/test_timeline_decoder_v2.py`
- `backend/tests/test_event_decoder.py`

Deliverables:
- room-aware decode policy (min-duration, gap-fill, hysteresis, priors).
- calibrated probabilities (temperature scaling or isotonic with fixed protocol).
- deterministic decode output for fixed inputs.

Done criteria:
- fragmentation rate improves versus baseline.
- no increase in false-empty breach.
- calibration metrics (ECE/Brier) reported per split/seed.

---

### WS-4: Timeline Metrics + Tiered Gates (2 days)
Objective:
- make release decisions on timeline and care relevance.

Files:
- `backend/ml/event_metrics.py`
- `backend/ml/event_kpi.py`
- `backend/ml/event_gates.py`
- `backend/tests/test_event_metrics.py`
- `backend/tests/test_event_kpi.py`
- `backend/tests/test_event_gates.py`

Deliverables:
- metrics: start/end MAE, duration MAE, episode count error, fragmentation rate.
- tiered critical gates (Tier-1/2/3) with stricter floors than exploratory mode.
- unknown-rate governance per room/label with fail reasons.

Done criteria:
- synthetic collapse and ghost-model tests fail correctly.
- clean scenarios pass with margin.
- gate report includes explicit failed conditions and supports.

---

### WS-5: Pipeline/Unified Integration (Shadow) (2 days)
Objective:
- integrate timeline artifacts into live path with no promotion risk.

Files:
- `backend/ml/pipeline.py`
- `backend/ml/unified_training.py`
- `backend/ml/training.py`
- `backend/tests/test_unified_training_path.py`
- `backend/tests/test_pipeline_integration.py`

Deliverables:
- shadow flags:
  - `event_first_shadow=true`
  - `timeline_decoder_v2=true`
  - `timeline_multitask=true`
- persisted artifacts:
  - `timeline_windows.parquet`
  - `timeline_episodes.parquet`
  - `timeline_qc.json`

Done criteria:
- shadow ON writes artifacts and gates.
- shadow OFF unchanged runtime behavior.
- no regression in production entrypoints.

---

### WS-6: Controlled Validation + Signoff (2 days)
Objective:
- generate promotion-grade decision packet.

Files:
- `backend/scripts/run_event_first_backtest.py`
- `backend/scripts/aggregate_event_first_backtest.py`
- `backend/scripts/generate_residual_review_pack.py`
- `backend/tests/test_event_first_backtest_script.py`
- `backend/tests/test_generate_residual_review_pack.py`

Deliverables:
- complete split-seed matrix report.
- baseline-vs-candidate delta tables.
- top residual clusters by room/label and error mode.

Done criteria:
- all required artifacts produced.
- gate decision is deterministic with fixed seeds/config.
- promotion recommendation includes explicit risk statement.

---

## 5A. Detailed Team Execution Packages (Implementation-Ready)
This section is the execution contract for team members.  
Every package below includes purpose, objective, concrete code changes, expected outcomes, and exact testing.

### EP-1: Full Multi-Task Timeline Heads in Production Training Path
Purpose:
- align training objective with timeline fidelity instead of window-only class accuracy.

Objective:
- integrate multi-task heads in live training path behind safe flags, with no behavior change when flags are OFF.

Files to edit:
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/training.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/transformer_timeline_heads.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/timeline_targets.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/pipeline.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_training.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_transformer_timeline_heads.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_timeline_targets.py`

Code change checklist:
- add model-builder switch in `training.py`:
  - `ENABLE_TIMELINE_MULTITASK=false` -> existing path unchanged.
  - `ENABLE_TIMELINE_MULTITASK=true` and room in `TIMELINE_NATIVE_ROOMS` -> multi-task model path.
- ensure target-builder emits:
  - activity target,
  - occupancy target,
  - boundary start/end targets,
  - daily duration/count targets.
- ensure loss weights are explicit and policy-driven; never hidden constants in multiple files.
- emit training diagnostics into run artifacts:
  - per-head loss values,
  - weight config,
  - enabled/disabled flags per room.
- confirm serialization/loading of model config works with new heads.

Expected outcome:
- flag OFF: identical behavior and metrics to prior baseline path.
- flag ON: multi-task outputs available and trainable, with artifacts proving timeline objectives were optimized.

Testing procedure:
1. Unit tests
- `pytest -q /Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_transformer_timeline_heads.py`
- `pytest -q /Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_timeline_targets.py`
2. Training integration tests
- `pytest -q /Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_training.py -k timeline`
3. Regression safety
- `pytest -q /Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_unified_training_path.py`

Pass criteria:
- all tests pass,
- no OFF-path regression,
- run artifact contains per-head losses and flags.

Common bug traps:
- missing label-space alignment between targets and logits,
- silently training with empty targets,
- producing timeline outputs but never using them in loss.

### EP-2: Calibrated Decoder v2 for Robust Episode Reconstruction
Purpose:
- convert per-window probabilities into stable, realistic episodes for timeline use.

Objective:
- deploy decoder v2 + calibration in evaluation/training output path with leakage-safe fitting.

Files to edit:
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/timeline_decoder_v2.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/calibration.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/event_decoder.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_timeline_decoder_v2.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_calibration.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_decoder.py`

Code change checklist:
- enforce calibrator fit only on train-internal calibration slice.
- ban threshold tuning or calibrator fit on holdout/test split.
- decoder must use policy object (room-aware min-duration, gap-fill, hysteresis).
- record decode diagnostics:
  - episode merges/splits,
  - dropped short episodes,
  - applied thresholds and calibration method.
- include deterministic decode mode for fixed input tests.

Expected outcome:
- lower fragmentation and better duration/start-end error for target rooms without safety regressions.

Testing procedure:
1. Decoder unit tests
- `pytest -q /Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_timeline_decoder_v2.py`
2. Calibration unit tests
- `pytest -q /Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_calibration.py`
3. Backtest integration
- `pytest -q /Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`

Pass criteria:
- deterministic decode tests pass,
- calibration metrics (ECE/Brier) present in artifacts,
- no leakage audit violation from decoder/calibration path.

Common bug traps:
- probability renormalization errors after calibration,
- unstable decoder behavior from implicit defaults,
- timeline improvements caused by leakage from holdout tuning.

### EP-3: Stronger Timeline Gates for Promotion Decisions
Purpose:
- prevent promotion when timeline quality is unstable or incomplete.

Objective:
- make signoff fail-closed for missing/invalid metrics and enforce strict split-seed evidence.

Files to edit:
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/aggregate_event_first_backtest.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/validation_run.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/signoff_pack.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_aggregate.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_d2_strict_splitseed_integration.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_validation_run.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_signoff_pack.py`

Code change checklist:
- enforce exact split-seed matrix coverage (canonical splits and seeds).
- missing KPI/timeline metrics => explicit fail reason.
- leakage audit missing/failed => fail.
- malformed payload schema => fail, not skip.
- signoff output must include:
  - blocking reasons,
  - timeline release stage,
  - reproducibility metadata (sha/hash/version).

Expected outcome:
- no optimistic PASS due to partial evidence or malformed artifacts.

Testing procedure:
1. Aggregator + strict matrix tests
- `pytest -q /Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_aggregate.py`
- `pytest -q /Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_d2_strict_splitseed_integration.py`
2. Validation/signoff tests
- `pytest -q /Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_validation_run.py /Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_signoff_pack.py`

Pass criteria:
- strict matrix failures are caught,
- fail reasons are explicit and stable,
- no fail-open path remains in signoff generation.

Common bug traps:
- using count-based thresholds without cell-identity checks,
- silently ignoring malformed split payloads,
- treating absent audit flags as pass.

### EP-4: Shadow -> Canary -> Full Rollout with Strict Pass Criteria
Purpose:
- operationally safe promotion path with rollback for real-world deployment.

Objective:
- make rollout stage transitions depend on complete, validated signoff evidence.

Files to edit:
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/t80_rollout_manager.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_t80_rollout.py`

Code change checklist:
- enforce stage transitions:
  - `SHADOW -> CANARY` only via explicit API,
  - `CANARY -> FULL` only after observation complete + prior PROMOTE decision.
- fail-closed canary checks:
  - hard gates 100% pass,
  - leakage audit explicit pass only,
  - timeline pass-rate threshold met,
  - full cohort coverage (no missing/extra/duplicate elders).
- clear stale promotion decision when later evidence fails.
- adapter from live artifacts to canary result schema:
  - WS-6 signoff,
  - validation signoff,
  - aggregate signoff (with elder override if needed).

Expected outcome:
- production cannot move to FULL based on stale or incomplete evidence.

Testing procedure:
1. Rollout unit/integration tests
- `pytest -q /Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_t80_rollout.py`
2. Compatibility checks with signoff generation
- `pytest -q /Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_validation_run.py /Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_signoff_pack.py`

Pass criteria:
- all rollout tests pass,
- artifact adapters parse supported schemas,
- promotion blocked whenever any fail-closed condition is unmet.

Common bug traps:
- cached promote state not invalidated on later failures,
- canary evaluated with partial elder results,
- promotion allowed before observation window.

### EP-5: Team PR Handoff Packet (Required for every merge)
Purpose:
- reduce review loops and avoid hidden regressions.

Each PR must include:
- Purpose and scope (what changed, why).
- Files edited (exact paths).
- Tests run (exact commands + pass output summary).
- Artifacts produced (absolute paths).
- Before/after metrics table for touched gates/metrics.
- Known limitations and explicit follow-ups.

Reject PR if any item is missing.

### EP-6: Merge Order to Minimize Rework
1. EP-1 (training objective path)
2. EP-2 (decoder/calibration)
3. EP-3 (gates/signoff strictness)
4. EP-4 (rollout controller and adapters)
5. One shadow run and one canary rehearsal before production toggle

## 6. PR-by-PR Quality Checklist (Mandatory in every PR)
1. Contract and schema:
- API/artifact schema changes documented.
- backward compatibility behavior stated.

2. Leakage and data hygiene:
- train-only fit policy explicitly enforced.
- leakage tests added/updated.

3. Failure tests:
- one test proving expected fail behavior for bad inputs/models.

4. Determinism:
- fixed-seed reproducibility test or assertion.

5. Metrics evidence:
- before/after table for touched metrics.
- confidence/variance stats where applicable.

6. Operational safety:
- feature flag default-safe.
- rollout and rollback documented.

7. Observability:
- logs include run id, gate decisions, and artifact references.

8. Performance guardrail:
- runtime/memory impact measured and reported.

9. Reviewer packet:
- changed files list.
- exact commands run.
- artifact paths.
- known limitations/open risks.

## 7. Team Lane Split (Parallel)
- Lane T1: WS-1 + WS-2 (targets/losses/model heads).
- Lane T2: WS-3 (decoder/calibration).
- Lane T3: WS-4 + WS-6 (metrics/gates/signoff).
- Lane T4: WS-5 (pipeline integration/shadow runtime).

Each lane must publish:
- implementation PR,
- test PR (if split),
- evidence note with artifacts.

## 8. Promotion Gate Thresholds (Initial)
These are initial controlled-validation thresholds; adjust only via reviewed policy change.

Safety gates (must pass all split-seed cells):
- Tier-1 critical recall floor: `>= 0.50`
- Tier-2 critical recall floor: `>= 0.35`
- Tier-3 critical recall floor: `>= 0.20`
- Home-empty precision: `>= 0.95`
- False-empty rate: `<= 0.05`
- No critical collapse (recall near zero with adequate support).

Timeline quality gates:
- fragmentation_rate improvement vs baseline: `>= 20%` relative.
- segment_duration_mae improvement in target rooms: non-negative in at least 80% split-seed cells.
- macro-F1 stability std ceiling: room-wise std `<= 0.05` (or approved override).
- unknown-rate cap: room/label must stay below policy cap; breaches are hard-fail until justified.

## 9. Rollout Path
1. Dev validation.
2. Shadow in live-like run for fixed observation window.
3. Canary promotion for restricted cohort.
4. Full promotion after canary success and unchanged safety metrics.

No direct jump from offline success to full rollout.

## 10. Execution Timeline
- Day 1: WS-0 baseline freeze and reproducibility checks.
- Day 2-3: WS-1 complete.
- Day 4-6: WS-2 and WS-3 parallel.
- Day 7-8: WS-4 and WS-5.
- Day 9-10: WS-6 signoff pack + go/no-go.
- Buffer: add 2 days if integration issues occur.

## 11. Definition of Done (Program-Level)
Program is done only when all are true:
- all WS done criteria met,
- all mandatory checklists satisfied for merged PRs,
- controlled validation pack complete and reproducible,
- shadow/canary evidence acceptable,
- documented go/no-go signed off with explicit residual risks.

## 12. Immediate Next Action
Execute WS-2/WS-3 integration with strict leakage audit artifact generation enabled by default in backtest/signoff scripts.
