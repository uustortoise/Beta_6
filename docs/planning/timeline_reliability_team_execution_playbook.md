# Timeline Reliability Team Execution Playbook

## 0) Document Control
- Date: February 17, 2026
- Workspace: `/Users/dicksonng/DT/Development/Beta_5.5`
- Branch: `beta-5.5-transformer`
- Program owner: ML Platform lead
- Objective: deliver reliable ADL timeline quality (start/end/duration/episode integrity), not only event-pass.

## 1) Background and Problem Statement
Current model path is operationally stable, but Bedroom/LivingRoom timeline quality is not yet promotion-grade.  
Observed pattern:
- window-level accuracy can increase while macro-F1 and minority recall degrade,
- episodes are fragmented or mis-timed,
- model learns occupancy easier than room-specific ADL semantics.

Root issue:
- objective mismatch (per-window classification optimized; timeline integrity weakly supervised).

## 2) Target Outcome (What “Done” Means)
A candidate is promotable only if all are true:
1. Safety gates pass on all split-seed cells.
2. Timeline gates pass in at least 80% of split-seed cells.
3. Bedroom/LivingRoom minority labels do not collapse.
4. Fragmentation improves vs baseline.
5. Results are reproducible (same config/data => same pass/fail outcome).

## 3) Non-Negotiable Engineering Rules
1. Train-only fit for scaler/imputer/calibrator (no holdout leakage).
2. Feature flags default-safe (off by default).
3. No API contract break to existing entrypoints.
4. No promotion from exploratory/scaffold path.
5. Every PR includes evidence package (tests + artifact paths + before/after metrics).

## 4) Fixed Evaluation Contract

### 4.1 Rolling splits and seeds
- Splits: `4->5`, `4+5->6`, `4+5+6->7`, `4+5+6+7->8`, `4+...+8->9`, `4+...+9->10` (when data exists).
- Seeds: `11`, `22`, `33`.
- Data directory for current project: `/Users/dicksonng/DT/Development/New training files`.

### 4.2 Mandatory artifacts
- `..._seedXX_...json` per run
- aggregated rolling summary JSON
- signoff JSON
- leakage audit JSON
- residual pack JSON/CSV

### 4.3 Baseline lock
- Baseline version/hash must be embedded in all aggregate and signoff payloads.
- Candidate comparisons must reference the same baseline payload hash.

## 5) Team Lanes and Responsibilities

### Lane L1: Model Objective and Targets
- Scope: timeline-native training objective integration.

### Lane L2: Decoder and Calibration
- Scope: calibrated inference and stable episode reconstruction.

### Lane L3: Metrics/Gates/Signoff
- Scope: promotion criteria and reproducible decision pack.

### Lane L4: Pipeline and Rollout
- Scope: shadow/canary integration with zero default behavior change.

## 6) Task Breakdown (Executable)

---

## T-00 Baseline Freeze and Reproducibility

### Background
Without baseline/version lock, comparisons are not auditable and review churn is high.

### Objective
Guarantee every candidate run is traceable to immutable baseline artifacts.

### Files to change
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/aggregate_event_first_backtest.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_aggregate.py`

### Procedure
1. Ensure output JSON includes:
   - `baseline_version`
   - `baseline_artifact_hash`
   - `git_sha`
   - `config_hash`
2. Ensure baseline fields are part of config hash payload.
3. Add/extend tests for:
   - fields present,
   - defaults (`None`) when not provided,
   - config hash binding.

### Test commands
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend
pytest -q tests/test_event_first_backtest_aggregate.py
```

### Pass requirements
- All tests pass.
- Re-running aggregator with same inputs yields identical gate outcome and metric deltas.

---

## T-10 Label Governance for Future ADL Expansion

### Background
ADL label set will evolve. Without governance, support fragmentation and alias drift will destabilize training.

### Objective
Support new ADL labels safely without breaking training/gates.

### Files to change
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/config/adl_event_registry.v1.yaml`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/adl_registry.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/event_labels.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_adl_registry_loader.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_labels.py`

### Procedure
1. Enforce canonical alias resolution at ingest before episode/target generation.
2. Implement support-aware activation policy:
   - low-support new labels can map to safe fallback (`occupied_unknown`) in promotion gates.
   - keep raw label stats for observability.
3. Add unknown-budget governance fields to report payload.

### Test commands
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend
pytest -q tests/test_adl_registry_loader.py tests/test_event_labels.py
```

### Pass requirements
- New labels do not crash training/inference.
- Alias mapping deterministic.
- Unknown fallback behavior explicitly covered by tests.

---

## T-20 Timeline-Native Training Path (Live Train Path, Flagged)

### Background
Heuristic decoders alone are insufficient; objective must learn boundaries and durations.

### Objective
Integrate timeline-native supervision into real `train_room` path for Bedroom/LivingRoom.

### Files to change
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/training.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/transformer_timeline_heads.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/transformer_backbone.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_training.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_transformer_timeline_heads.py`

### Procedure
1. Keep default production path unchanged when flags are off.
2. Introduce/keep feature flags:
   - `ENABLE_TIMELINE_MULTITASK`
   - `TIMELINE_NATIVE_ROOMS` (default `bedroom,livingroom`)
3. In flag-on path:
   - generate boundary/attribute targets,
   - train multi-task heads (activity + occupancy + boundary start/end + optional duration/count),
   - emit per-head loss and support diagnostics.
4. Preserve current model artifact contract for downstream compatibility.

### Test commands
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend
pytest -q tests/test_training.py tests/test_transformer_timeline_heads.py
```

### Pass requirements
- Flag OFF: zero behavior change vs existing tests.
- Flag ON: training runs and outputs timeline-native diagnostics.
- No regression in full backend suite.

---

## T-30 Target Builder Integration and Leakage Audit

### Background
Timeline target quality is critical; leakage invalidates all evidence.

### Objective
Use deterministic boundary/attribute targets with explicit leakage audit in run artifacts.

### Files to change
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/timeline_targets.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/leakage_audit.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_timeline_targets.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_leakage_audit.py`

### Procedure
1. Ensure transitions:
   - excluded->care => start
   - care->excluded => end
   - care->care label change => end+start
2. Ensure excluded labels are not counted as care episodes.
3. Emit per-split leakage checklist with pass/fail and reasons.
4. Fail signoff if leakage audit fails in any split.

### Test commands
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend
pytest -q tests/test_timeline_targets.py tests/test_leakage_audit.py
```

### Pass requirements
- Deterministic target tests pass.
- Leakage audit defaults fail-closed and passes only with explicit checks set.

---

## T-40 Decoder v2 + Calibration Integration

### Background
Boundary timing and fragmentation require calibrated probabilities and stable decode policy.

### Objective
Run BR/LR decode with calibrated probabilities and deterministic decoder v2 policy.

### Files to change
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/timeline_decoder_v2.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/calibration.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/training.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_timeline_decoder_v2.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_calibration.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`

### Procedure
1. Fit calibrators only on train/calibration slices.
2. Normalize calibrated activity probabilities to valid distributions.
3. Run decoder v2 with room-aware policy and deterministic thresholds.
4. Emit decoder debug payload per room/split.

### Test commands
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend
pytest -q tests/test_timeline_decoder_v2.py tests/test_calibration.py tests/test_event_first_backtest_script.py
```

### Pass requirements
- No serialization/probability-shape failures.
- Decoder deterministic for fixed inputs.
- Fragmentation metric improves in BR/LR candidate runs vs baseline.

---

## T-50 Timeline Metrics + Promotion Gates

### Background
Promotion must reflect timeline quality, not just pointwise classification.

### Objective
Enforce timeline-centric gates with stability and collapse checks.

### Files to change
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/timeline_metrics.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/timeline_gates.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/signoff_pack.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_timeline_metrics.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_timeline_gates.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_signoff_pack.py`

### Procedure
1. Compute per-room metrics:
   - start/end MAE
   - duration MAE
   - episode count error
   - fragmentation
2. Add promotion gates:
   - minority recall floor by criticality tier,
   - unknown-rate caps,
   - collapse detection.
3. Update signoff logic to require timeline gates in passing-cell criteria.

### Test commands
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend
pytest -q tests/test_timeline_metrics.py tests/test_timeline_gates.py tests/test_signoff_pack.py
```

### Pass requirements
- Synthetic collapse/ghost scenarios fail.
- Gate pass/fail reasons explicit and reproducible.

---

## T-60 Pipeline Integration (Shadow-First)

### Background
Need live-path integration without risking current production behavior.

### Objective
Wire timeline path into pipeline with default-off safety and shadow artifact generation.

### Files to change
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/pipeline.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/unified_training.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/timeline_pipeline_integration.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_pipeline_integration.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_unified_training_path.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_timeline_pipeline_integration.py`

### Procedure
1. Preserve parity between `train_and_predict` and `train_from_files`.
2. Keep all timeline flags default `false`.
3. When shadow enabled, write:
   - `timeline_windows.parquet`
   - `timeline_episodes.parquet`
   - `timeline_qc.json`
4. Ensure no behavior change when shadow is disabled.

### Test commands
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend
pytest -q tests/test_pipeline_integration.py tests/test_unified_training_path.py tests/test_timeline_pipeline_integration.py
```

### Pass requirements
- Entry-point parity holds.
- Shadow mode writes artifacts.
- Non-shadow mode unchanged.

---

## T-70 Controlled Validation and Signoff

### Background
Need a deterministic go/no-go packet instead of ad-hoc judgments.

### Objective
Generate one reproducible signoff pack with explicit decision and residual risks.

### Files to change
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/aggregate_event_first_backtest.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/validation_run.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_validation_run.py`

### Procedure
1. Run baseline and candidate with identical protocol.
2. Aggregate split-seed results and compute pass rates + variances.
3. Build signoff decision:
   - `PASS`, `CONDITIONAL`, `FAIL`
   - primary reasons, blocking issues, residual risks, recommended stage.

### Test commands
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend
pytest -q tests/test_event_first_backtest_script.py tests/test_validation_run.py
```

### Pass requirements
- Signoff output complete and deterministic.
- Decision logic fails closed if required evidence missing.

---

## T-80 Rollout (Shadow -> Canary -> Full)

### Background
Offline improvements do not guarantee production reliability.

### Objective
Promote with controlled exposure and rollback safety.

### Files to change
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/pipeline.py`
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/config/*.json` (feature flag control)
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/validation_run.py`

### Procedure
1. Shadow period:
   - timeline features ON in shadow only.
   - no promotion from shadow outputs.
2. Canary:
   - limited elder cohort + fixed observation window.
3. Full:
   - only if canary meets same safety/timeline criteria.

### Pass requirements
- No safety regressions.
- BR/LR timeline metrics stable in canary.
- Rollback validated.

## 7) Promotion Thresholds (Current Policy)
- Tier-1 recall floor: `>= 0.50`
- Tier-2 recall floor: `>= 0.35`
- Tier-3 recall floor: `>= 0.20`
- Home-empty precision: `>= 0.95`
- False-empty rate: `<= 0.05`
- Timeline gate pass rate: `>= 0.80` (split-seed cells)
- Hard gate pass: all safety gates per split-seed cell
- Macro-F1 stability std ceiling (room-wise): `<= 0.05` unless approved override.

## 8) Required PR Template (Mandatory)
Every PR must include:
1. Background and objective.
2. File list changed.
3. Procedure completed (step-by-step).
4. Tests run (exact commands + results).
5. Before/after metrics table.
6. Leakage/safety statement.
7. Risks and rollback notes.
8. Artifact paths produced.

## 9) Review Fast-Path Checklist (Reviewer Use)
1. Flag OFF path unchanged?
2. Train-only fit enforced?
3. Deterministic test present?
4. Failure-path test present?
5. Artifacts include baseline/version/hash?
6. Gate logic fail-closed?
7. Evidence table included?

## 10) Suggested Execution Sequence (Low Rework Order)
1. `T-00` -> `T-10` -> `T-20` -> `T-30`
2. `T-40` in parallel with `T-50`
3. `T-60`
4. `T-70`
5. `T-80`

## 11) Immediate Next Sprint Scope (Recommended)
Focus only on:
- `T-20` true multi-task integration in live train path,
- `T-30` leakage-safe target plumbing,
- `T-70` controlled Jessica evaluation.

Do not expand to full rollout until these three produce stable pass rates.
