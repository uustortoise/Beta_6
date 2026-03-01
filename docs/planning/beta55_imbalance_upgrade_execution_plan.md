# Beta 5.5 Imbalance Upgrade Execution Plan (Promotion-Path Safe)

- Status: Draft for execution
- Date: February 19, 2026
- Workspace: `/Users/dicksonng/DT/Development/Beta_5.5`
- Scope: Improve Bedroom/LivingRoom hard-gate reliability under strict WS-6 without architecture reset
- Primary objective: convert current `23/30` eligible hard-gate result to `30/30` while preserving non-BL safety and fail-closed contracts

---

## 1. Current State Snapshot

### 1.1 Anchor run
- Anchor family: `ws6_next_ab_min3_smooth_kitchen_tune`
- Key artifacts:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_next_ab_min3_smooth_kitchen_tune/ws6_rolling.json`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_next_ab_min3_smooth_kitchen_tune/ws6_signoff.json`

### 1.2 Observed status (as of 2026-02-19)
- Eligible hard-gate: `23/30`
- Full hard-gate: `41/60`
- Eligibility blocker: `30/60` checks are ineligible due to `not_eligible_below_min_train_days` (50% of matrix blocked)
- Room pass profile (3 seeds x 4 splits = 12 checks/room):
  - Kitchen: `12/12` pass
  - Bathroom: `12/12` pass
  - Entrance: `12/12` pass
  - Bedroom: `3/12` pass
  - LivingRoom: `2/12` pass
- Blocking reasons:
  - `hard_gate_all_seeds_failed`
  - `hard_gate_split_requirement_failed:23<30`
- Dominant BL failure reasons:
  - LivingRoom: `occupied_f1_lt_0.580`, `occupied_recall_lt_0.500`, `recall_livingroom_normal_use_lt_0.400`
  - Bedroom: `occupied_f1_lt_0.550`, `occupied_recall_lt_0.500`, `fragmentation_score_lt_0.450`
- Kitchen MAE is below threshold in anchor, but Bedroom/LivingRoom hard-gates remain bottleneck.

### 1.3 Why this plan
- We already have hardened promotion contracts in Beta 5.5.
- Failures are concentrated in BL imbalance/separability and data support.
- Fastest safe path is to upgrade Beta 5.5 directly, and keep any major redesign shadow-only.

### 1.4 Execution implications from failure forensics
1. Phase 1 is mandatory: we must unblock eligibility before model tuning can be trusted.
2. Phase 3 must include real segment feature expansion and a learned segment classifier path; otherwise segment work is effectively a no-op.
3. Stage A capacity must be tested explicitly (RF vs HGB in BL scope) before declaring architecture ceiling.

---

## 2. Non-Negotiable Constraints

1. Promotion contract stays fail-closed and unchanged:
   - strict split-seed matrix
   - baseline binding and artifact hash verification
   - leakage audit artifact enforcement
2. New behavior remains default-off until strict signoff is clean.
3. No relaxing hard-gate floors to “win metrics.”
4. No destructive changes to existing artifact schemas.
5. Every experiment must be reproducible by command + config + artifact bundle.

---

## 3. Upgrade Strategy

## 3.1 Core approach
1. Fix data support/coverage first.
2. Add imbalance-aware BL controls in existing training path.
3. Make segment mode meaningful: expand segment features, then add learned segment-level classifier + low-support fallback.
4. Add Stage A BL capacity contingency (`RF` baseline vs `HGB` scoped toggle) before Phase 4.
5. Tune arbitration/timeline controls only after BL support and class balance are improved.
6. Attempt strict signoff only with full D2 contract evidence.

## 3.2 Planned timeline
- Phase 0: Day 0 (baseline freeze/repro harness)
- Phase 1: Day 1-2 (coverage + support diagnostics)
- Phase 2: Day 3-5 (imbalance-aware BL training)
- Phase 3A: Day 6-7 (segment feature foundation + instrumentation)
- Phase 3B: Day 8-10 (learned segment classifier + room tuning + buffer)
- Phase 4: Day 11-12 (arbitration + timeline stabilization)
- Phase 5: Day 13 (strict signoff and go/no-go)
- Total planned duration: 13 days (includes 2-day Phase 3 buffer for structural segment work)

---

## 4. Deliverables by Phase

## 4.1 Phase 0 - Baseline Freeze and Repro Harness

### Goal
Create a deterministic baseline and experiment harness so every delta is attributable.

### Task P0.1 - Freeze anchor evidence
- Description:
  - Copy anchor outputs into a dated immutable archive path.
  - Record baseline version/hash for aggregate binding.
- File/code to change:
  - No code change required.
  - Create docs note under `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/`.
- Testing procedure:
  1. Confirm archived files checksum and presence.
  2. Confirm `ws6_signoff.json` has expected fail reasons.
- Pass criteria:
  - Archive exists and can be read by team.
  - Baseline metadata recorded for reuse.

### Task P0.2 - Repro runner templates
- Description:
  - Standardize commands for seed 11/22/33 + aggregate.
- File/code to change:
  - Add command block to this plan only.
- Testing procedure:
  1. Execute all three seed runs.
  2. Execute aggregator once.
  3. Verify deterministic artifact structure.
- Pass criteria:
  - Rerun reproduces same fail profile class (hard-gate seed/split failure).

### Commands (baseline repro template)
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend

OUT_DIR="/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/baseline_repro_2026-02-19"
mkdir -p "$OUT_DIR"

python3 scripts/run_event_first_backtest.py \
  --data-dir "/Users/dicksonng/DT/Development/New training files" \
  --elder-id HK0011_jessica \
  --min-day 4 --max-day 8 \
  --seed 11 \
  --output "$OUT_DIR/seed11.json"

python3 scripts/run_event_first_backtest.py \
  --data-dir "/Users/dicksonng/DT/Development/New training files" \
  --elder-id HK0011_jessica \
  --min-day 4 --max-day 8 \
  --seed 22 \
  --output "$OUT_DIR/seed22.json"

python3 scripts/run_event_first_backtest.py \
  --data-dir "/Users/dicksonng/DT/Development/New training files" \
  --elder-id HK0011_jessica \
  --min-day 4 --max-day 8 \
  --seed 33 \
  --output "$OUT_DIR/seed33.json"

python3 scripts/aggregate_event_first_backtest.py \
  --seed-reports "$OUT_DIR/seed11.json" "$OUT_DIR/seed22.json" "$OUT_DIR/seed33.json" \
  --rolling-output "$OUT_DIR/ws6_rolling.json" \
  --signoff-output "$OUT_DIR/ws6_signoff.json" \
  --comparison-window "ws6_day4_to_day8" \
  --required-split-pass-ratio 1.0 \
  --baseline-version "ws6_next_ab_min3_smooth_kitchen_tune" \
  --baseline-artifact-hash "sha256:<fill_from_anchor>" \
  --baseline-artifact-path "/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_next_ab_min3_smooth_kitchen_tune/ws6_signoff.json" \
  --leakage-audit-paths "$OUT_DIR/seed11_leakage_audit.json" "$OUT_DIR/seed22_leakage_audit.json" "$OUT_DIR/seed33_leakage_audit.json"
```

---

## 4.2 Phase 1 - Coverage and Support Hardening

### Goal
Reduce imbalance caused by sparse/non-contiguous training support before model complexity changes.

### Task P1.1 - Add support diagnostics in backtest payload
- Description:
  - Emit split/room/label/day-support summary in output JSON.
  - Mark where label-recall gate is skipped due to missing support day.
- File/code to change:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`
- Testing procedure:
  1. Unit tests for support payload schema and deterministic formatting.
  2. Backtest run and inspect `label_recall_train_day_support` fields.
- Pass criteria:
  - Each BL split includes day-support map.
  - Skip logic is explicit and tested.

### Task P1.2 - Canonical data continuity audit
- Description:
  - Audit day sequence completeness and flag missing contiguous windows.
  - Enforce canonical training files only.
- File/code to change:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py` (discovery/report section)
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`
- Testing procedure:
  1. Inject synthetic file list with gaps.
  2. Verify emitted continuity warnings and file-selection behavior.
- Pass criteria:
  - Missing-day windows are visible in artifact payload.
  - Derived variants stay excluded.

### Task P1.3 - Data ingest action and strict rerun
- Description:
  - Add missing contiguous canonical training files in raw intake.
  - Re-run strict WS-6 unchanged.
- File/code to change:
  - No code mandatory, data operation + rerun only.
- Testing procedure:
  1. Re-run baseline command template against updated data.
  2. Compare split support and hard-gate deltas vs anchor.
- Pass criteria:
  - BL support profile improved.
  - `not_eligible_below_min_train_days` checks reduced from `30/60` to `<=15/60`.
  - Bedroom/LivingRoom each have at least `9/12` eligible checks.
  - No regression in Kitchen/Bathroom/Entrance KPI checks.

---

## 4.3 Phase 2 - Imbalance-Aware BL Training Controls

### Goal
Improve BL occupied recall/F1 safely without causing duration inflation or false-positive collapse.

### Task P2.1 - BL hard-negative mining and replay tuning
- Description:
  - Tune weighted hard negatives and replay rows under strict caps.
  - Keep default-off and scoped to BL.
- File/code to change:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`
- Relevant flags:
  - `--enable-bedroom-livingroom-hard-negative-mining`
  - `--bedroom-livingroom-hard-negative-weight`
  - `--enable-bedroom-livingroom-failure-replay`
  - `--bedroom-livingroom-failure-replay-weight`
  - `--bedroom-livingroom-max-replay-rows-per-day`
  - `--livingroom-occupied-sample-weight`
- Testing procedure:
  1. Unit tests on replay cap and weight application.
  2. A/B runs per seed with one-flag-family-at-a-time.
- Pass criteria:
  - BL occupied recall increases in majority of seeds.
  - No hard-gate regressions in non-BL rooms.

### Task P2.2 - Hard-gate threshold tuning with duration guardrails
- Description:
  - Use bounded BL threshold tuning for hard-gate recovery.
  - Retain duration guardrails to avoid over-occupancy drift.
- File/code to change:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`
- Relevant controls:
  - `--enable-bedroom-livingroom-hardgate-threshold-tuning`
  - `--hard-gate-room-metric-floors`
  - `--hard-gate-label-recall-floors`
  - `--hard-gate-label-recall-min-supports`
  - `--hard-gate-fragmentation-min-run-windows`
  - `--hard-gate-fragmentation-gap-fill-windows`
  - `--hard-gate-min-train-days`
- Testing procedure:
  1. Existing hard-gate unit tests.
  2. Split-level sanity checks for adopted threshold vs baseline threshold.
- Pass criteria:
  - Recovered recall does not increase duration MAE outside guardrail.
  - Hard-gate reasons shift downward for BL recall/F1 failures.

### Task P2.3 - Temporal and context features controlled rollout
- Description:
  - Evaluate temporal occupancy features and optional cross-room context features.
- File/code to change:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`
- Relevant flags:
  - `--enable-room-temporal-occupancy-features`
  - `--bedroom-livingroom-texture-profile={30m,60m,mixed}`
  - `--enable-bedroom-light-texture-features`
  - `--enable-cross-room-context-features`
  - `--cross-room-context-rooms`
- Testing procedure:
  1. Unit tests for feature generation and profile behavior.
  2. A/B run grid: `30m`, `60m`, `mixed` with and without bedroom light texture.
- Pass criteria:
  - BL metrics improve with no data-leakage style failures.
  - Model behavior remains deterministic across reruns.

### Task P2.4 - Stage A BL capacity contingency (RF vs HGB)
- Description:
  - Run controlled BL-only Stage A model-family ablation before Phase 3B lock:
    - baseline: RF
    - contingency: HGB
  - Promote HGB for BL path only if it improves BL recall/F1 and does not regress non-BL gates.
- File/code to change:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/event_models.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_models.py`
- Relevant flags:
  - `--enable-bedroom-livingroom-stage-a-hgb`
  - `--enable-bedroom-livingroom-stage-a-sequence-model`
- Testing procedure:
  1. Execute RF vs HGB A/B with identical seeds/splits and feature flags.
  2. Compare BL `occupied_recall`, `occupied_f1`, and non-BL gate status.
  3. Verify deterministic serialization of selected `stage_a_model_type`.
- Pass criteria:
  - Capacity contingency decision recorded with evidence.
  - If HGB is adopted for BL, non-BL strict checks remain green.

---

## 4.4 Phase 3 - Segment Mode Strengthening (Existing Path)

### Goal
Reduce fragmentation and boundary noise by improving existing segment modules, not replacing architecture.

### Task P3.1 - Segment proposal robustness
- Description:
  - Improve gap merge/drop-short behavior for BL occupancy episodes.
- File/code to change:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/segment_proposal.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_segment_proposal.py`
- Testing procedure:
  1. Add synthetic square-wave and jittered occupancy tests.
  2. Validate boundary behavior for min duration and small-gap merges.
- Pass criteria:
  - Deterministic boundaries across seeds for same inputs.
  - No regression in existing segment proposal tests.

### Task P3.2 - Segment feature enrichment
- Description:
  - Expand `segment_features.py` from minimal occupancy/time stats to a full segment feature table used by classifier.
  - Required feature families:
    - Occupancy shape: mean/max/min/std, percentiles, slope, rising/falling edge counts.
    - Motion/activity: mean/std/max, active-ratio windows, burst count, run-length summaries.
    - Light: mean/std, start-end delta, linear slope, low-light/high-light ratios.
    - CO2: mean/std, slope, start-end delta, rise/fall counts.
    - Temperature/humidity: mean and delta statistics per segment.
    - Time context: start/end hour encodings, segment duration buckets, night/day indicator.
  - Add schema/version metadata so downstream can assert feature contract.
- File/code to change:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/segment_features.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_segment_features.py`
- Testing procedure:
  1. Unit tests for feature key coverage, deterministic values, and NaN/inf handling.
  2. Unit tests for missing-sensor fallback behavior (feature still emitted with safe defaults).
  3. Integration validation of segment debug payload (feature count, schema version, finite values).
- Pass criteria:
  - Feature output is deterministic, finite, and materially expanded (target: `>=25` numeric feature columns/segment).
  - Segment-mode run completes with expected debug payload fields.

### Task P3.3 - Segment label assignment refinement
- Description:
  - Replace pure per-label mean-probability argmax with a learned segment-level classifier using `segment_features.py` output.
  - Add confidence gating and low-support fallback:
    - use learned classifier when segment support is sufficient
    - fallback to current heuristic path when support below threshold or confidence below floor
    - log fallback reasons for auditability
  - Keep behavior default-off until Gate G3 passes.
- File/code to change:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/segment_classifier.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_segment_classifier.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`
- Testing procedure:
  1. Unit tests for learned-path prediction, confidence thresholding, and fallback branch behavior.
  2. Unit tests proving classifier consumes segment features (not activity mean argmax only).
  3. Integration run with segment mode enabled to validate debug payload:
     - classifier type
     - confidence stats
     - fallback count/rate
- Pass criteria:
  - Learned path is active on high-support segments and fallback path is deterministic on low-support segments.
  - BL fragmentation improves without occupied inflation.
  - Segment path obeys living-room guardrail.

### Task P3.4 - Projection and guardrail preservation
- Description:
  - Keep and verify living-room segment inflation guardrail and projection behavior.
- File/code to change:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/segment_projection.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_segment_projection.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
- Testing procedure:
  1. Unit tests for projection range behavior.
  2. Integration tests that trigger guardrail revert path.
- Pass criteria:
  - Guardrail fires when occupancy inflation exceeds cap.
  - Revert path keeps stable output and logs reason.

---

## 4.5 Phase 4 - Arbitration and Timeline Gate Stabilization

### Goal
Avoid cross-room conflict amplification while preserving genuine BL activity windows.

### Task P4.1 - Single-resident arbitration calibration
- Description:
  - Tune arbitration margins and guard priorities only where conflict suppression harms BL recall.
- File/code to change:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`
- Relevant flags:
  - `--enable-single-resident-arbitration`
  - `--single-resident-rooms`
  - `--single-resident-min-margin`
  - `--single-resident-bedroom-night-hours`
  - `--single-resident-bedroom-night-min-run-windows`
  - `--single-resident-bedroom-night-min-score`
  - `--single-resident-kitchen-min-score`
  - `--single-resident-kitchen-guard-hours`
  - `--single-resident-continuity-min-run-windows`
  - `--single-resident-continuity-min-occ-prob`
- Testing procedure:
  1. Existing arbitration tests.
  2. Add BL recall regression checks for arbitration-on vs arbitration-off.
- Pass criteria:
  - Conflict suppression improves consistency without BL recall collapse.

### Task P4.2 - Timeline gate consistency checks
- Description:
  - Keep timeline gates fail-closed and ensure they reflect actual timeline quality changes.
- File/code to change:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/timeline_gates.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_timeline_gates.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
- Testing procedure:
  1. Timeline gate unit suite.
  2. Integration checks in strict backtest payload.
- Pass criteria:
  - Timeline gate counts are deterministic.
  - No spurious pass from missing metrics.

---

## 4.6 Phase 5 - Strict Signoff Execution

### Goal
Generate promotion-grade evidence and make explicit go/no-go decision.

### Task P5.1 - Final strict run package
- Description:
  - Run 3 seeds, aggregate with full contract enforcement, produce signoff package.
- File/code to change:
  - No required code change.
- Testing procedure:
  1. Execute final run commands.
  2. Verify signoff contains no fail-closed violations.
  3. Compare against anchor deltas.
- Pass criteria:
  - `hard_gate_all_seeds = true`
  - `hard_gate_split_requirement_pass = true`
  - strict matrix pass true
  - no baseline binding violations
  - no leakage artifact violations
  - Kitchen KPI remains under threshold

---

## 5. Experiment Matrix and Run Discipline

## 5.1 One-factor-at-a-time matrix
1. Data coverage only
2. + hard-negative mining
3. + failure replay
4. + livingroom occupied sample weight sweep (`1.0`, `2.0`, `3.0`, `4.0`)
5. + temporal occupancy features (`30m`, `60m`, `mixed`)
6. + bedroom light texture
7. + Stage A BL model family sweep (RF vs HGB)
8. + segment mode baseline (legacy classifier retained for reference)
9. + segment feature enrichment enabled
10. + learned segment classifier + fallback thresholds
11. + segment room-specific tuning
12. + arbitration tuning

## 5.2 Parallel execution strategy (required)
1. Estimated workload:
   - `12` factor steps x `3` seeds = `36` seed runs + `12` aggregate runs.
   - Prior estimate was ~40h serial; target is <=16h wall-clock via parallel seeds.
2. Parallelism model:
   - For each factor step, run seeds `11/22/33` concurrently in separate output dirs.
   - Run aggregate only after all three seed runs complete.
   - Do not overlap different factor steps; keep one-factor-at-a-time discipline.
3. Lane assignment:
   - Lane A: seed 11
   - Lane B: seed 22
   - Lane C: seed 33
4. Failure handling:
   - Fail-fast if non-BL room hard-gate regresses in any lane.
   - Stop step if output schema/contract checks fail in one lane.
5. Artifact hygiene:
   - Output path format: `/backend/tmp/ws6_beta55_upgrade/<step_tag>/<seed>.json`
   - Aggregates stored as `/backend/tmp/ws6_beta55_upgrade/<step_tag>/ws6_signoff.json`

## 5.3 Required logging per run
1. Config flags used
2. `seed11/22/33` artifact paths
3. `ws6_rolling.json` and `ws6_signoff.json` path
4. Hard-gate totals (eligible/full)
5. Top failed reasons histogram
6. BL support diagnostics
7. Segment classifier mode stats (learned-path rate, fallback rate, confidence distribution)
8. Stage A model type actually selected per room

## 5.4 Run ledger location
- `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/beta55_imbalance_upgrade_run_log.md` (create/update during execution)

---

## 6. Detailed Testing Procedures

## 6.1 Mandatory unit/integration suite per code change batch
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend

pytest -q \
  tests/test_segment_proposal.py \
  tests/test_segment_features.py \
  tests/test_segment_classifier.py \
  tests/test_segment_projection.py \
  tests/test_event_models.py \
  tests/test_timeline_gates.py \
  tests/test_event_first_backtest_script.py \
  tests/test_event_first_backtest_aggregate.py \
  tests/test_d2_strict_splitseed_integration.py
```

## 6.2 Fast smoke after each BL tuning iteration
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend

python3 scripts/run_event_first_backtest.py \
  --data-dir "/Users/dicksonng/DT/Development/New training files" \
  --elder-id HK0011_jessica \
  --min-day 4 --max-day 8 \
  --seed 11 \
  --enable-room-temporal-occupancy-features \
  --bedroom-livingroom-texture-profile mixed \
  --enable-bedroom-light-texture-features \
  --enable-bedroom-livingroom-hard-negative-mining \
  --bedroom-livingroom-hard-negative-weight 2.5 \
  --enable-bedroom-livingroom-failure-replay \
  --bedroom-livingroom-failure-replay-weight 3.0 \
  --livingroom-occupied-sample-weight 2.0 \
  --output "/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/smoke_seed11.json"
```

## 6.3 Final strict signoff command template
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend

OUT_DIR="/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/final_candidate_<tag>"
mkdir -p "$OUT_DIR"

# Seed runs
python3 scripts/run_event_first_backtest.py --data-dir "/Users/dicksonng/DT/Development/New training files" --elder-id HK0011_jessica --min-day 4 --max-day 8 --seed 11 --output "$OUT_DIR/seed11.json" <flags>
python3 scripts/run_event_first_backtest.py --data-dir "/Users/dicksonng/DT/Development/New training files" --elder-id HK0011_jessica --min-day 4 --max-day 8 --seed 22 --output "$OUT_DIR/seed22.json" <flags>
python3 scripts/run_event_first_backtest.py --data-dir "/Users/dicksonng/DT/Development/New training files" --elder-id HK0011_jessica --min-day 4 --max-day 8 --seed 33 --output "$OUT_DIR/seed33.json" <flags>

# Aggregate signoff
python3 scripts/aggregate_event_first_backtest.py \
  --seed-reports "$OUT_DIR/seed11.json" "$OUT_DIR/seed22.json" "$OUT_DIR/seed33.json" \
  --rolling-output "$OUT_DIR/ws6_rolling.json" \
  --signoff-output "$OUT_DIR/ws6_signoff.json" \
  --comparison-window "ws6_day4_to_day8" \
  --required-split-pass-ratio 1.0 \
  --baseline-version "ws6_next_ab_min3_smooth_kitchen_tune" \
  --baseline-artifact-hash "sha256:<baseline_hash>" \
  --baseline-artifact-path "/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_next_ab_min3_smooth_kitchen_tune/ws6_signoff.json" \
  --leakage-audit-paths "$OUT_DIR/seed11_leakage_audit.json" "$OUT_DIR/seed22_leakage_audit.json" "$OUT_DIR/seed33_leakage_audit.json"
```

---

## 7. Phase Gates and Numeric Exit Criteria

## Gate G0 - Repro lock complete
- Pass when:
  1. Anchor fail profile reproduced.
  2. D2 contract test suite green.

## Gate G1 - Coverage uplift complete
- Pass when:
  1. BL support diagnostics show improved day support continuity.
  2. `not_eligible_below_min_train_days` ineligibility is `<=15/60`.
  3. Bedroom/LivingRoom each have at least `9/12` eligible checks.
  4. No non-BL KPI regression.

## Gate G2 - Imbalance controls validated
- Pass when:
  1. BL `occupied_recall` and `occupied_f1` improve in at least 2 seeds.
  2. BL failure reasons trend down for `occupied_recall_lt` and `occupied_f1_lt`.
  3. Stage A BL capacity decision (RF keep vs HGB adopt) is documented with A/B evidence.

## Gate G3 - Segment path stable
- Pass when:
  1. Expanded segment features are present and finite (target `>=25` numeric columns/segment).
  2. Learned classifier is active on supported segments with deterministic fallback on low-support segments.
  3. Segment mode improves or preserves BL fragmentation score.
  4. LivingRoom inflation guardrail remains effective.

## Gate G4 - Promotion-grade signoff
- Pass when:
  1. Eligible hard-gate `30/30`.
  2. `hard_gate_all_seeds=true`.
  3. `strict_split_seed_matrix_pass=true`.
  4. `baseline_binding` has no violations.
  5. `leakage_audit` has no violations.

---

## 8. Risk Register and Mitigations

1. Risk: BL recall gain causes duration overprediction.
   - Mitigation: enforce duration guardrails in threshold tuning.
2. Risk: Segment mode inflates LivingRoom occupancy.
   - Mitigation: keep ratio+delta guardrail and revert path.
3. Risk: Segment feature work is under-scoped and does not improve separability.
   - Mitigation: enforce required feature families + minimum feature-count criterion in Gate G3.
4. Risk: Segment classifier remains heuristic despite code changes.
   - Mitigation: require learned-path activation evidence + fallback-rate reporting.
5. Risk: Improvements depend on one seed only.
   - Mitigation: accept only if gains hold across seed 11/22/33.
6. Risk: Data continuity not improved enough.
   - Mitigation: block Phase 2 optimization until coverage audit improves.
7. Risk: RF capacity ceiling blocks BL recovery.
   - Mitigation: mandatory RF vs HGB BL contingency in Phase 2; escalate to shadow greenfield only if both fail.
8. Risk: Contract drift under fast iteration.
   - Mitigation: run D2 strict integration tests every merge candidate.

---

## 9. Rollback Plan

1. Keep all new knobs default-off.
2. If candidate regresses, rerun anchor command set and restore anchor config.
3. Use archived baseline artifacts for immediate reference comparison.
4. No production promotion action unless Gate G4 passes.

---

## 10. Ownership and Tracking

| Workstream | Owner | Reviewer | Start | Target Finish | Status |
|---|---|---|---|---|---|
| Phase 0 baseline freeze |  |  | 2026-02-19 | 2026-02-19 | Pending |
| Phase 1 coverage hardening |  |  | 2026-02-20 | 2026-02-21 | Pending |
| Phase 2 imbalance controls (+Stage A contingency) |  |  | 2026-02-22 | 2026-02-24 | Pending |
| Phase 3A segment feature foundation |  |  | 2026-02-25 | 2026-02-26 | Pending |
| Phase 3B learned segment classifier + tuning + buffer |  |  | 2026-02-27 | 2026-03-01 | Pending |
| Phase 4 arbitration/timeline stabilization |  |  | 2026-03-02 | 2026-03-03 | Pending |
| Phase 5 final strict signoff |  |  | 2026-03-04 | 2026-03-04 | Pending |

---

## 11. Definition of Done

1. Strict WS-6 signoff passes promotion-grade checks.
2. BL hard-gate bottleneck is cleared without non-BL regressions.
3. All changes are reproducible via committed commands and artifact paths.
4. Team has a complete run ledger and failure-to-fix traceability.
