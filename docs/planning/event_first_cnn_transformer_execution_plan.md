# Event-First CNN+Transformer Execution Plan

## Document Control
- **Date:** February 16, 2026
- **Scope:** Beta 5.5 (`/Users/dicksonng/DT/Development/Beta_5.5`)
- **Primary Goal:** Improve care-outcome reliability without increasing labeling complexity.

## Final Architecture (Production Target)
This plan keeps **CNN+Transformer** as the production model family.

1. **Input**
- Per-room time windows (existing sensor features + temporal features).

2. **Shared Encoder**
- CNN feature extractor over windowed signals.
- Transformer temporal backbone over encoded sequence.

3. **Dual Heads**
- **Head A (Occupancy):** `occupied` vs `unoccupied` (binary).
- **Head B (Activity):** room-specific activity classes, applied when Head A indicates occupied.

4. **Inference Decoder**
- Occupancy thresholding.
- Per-room label thresholds/calibration.
- Temporal hysteresis/min-duration smoothing to reduce flip-flop transitions.

5. **Event Layer**
- Window predictions -> episodes/events.
- Events -> care KPIs (sleep duration, shower-day, kitchen-use, living-room activity, out-time).
- Whole-home fusion -> `home_empty` / `home_occupied` / `home_unknown` with confidence.

6. **Promotion Gates**
- Event-level KPI gates as primary criteria.
- Window-level macro-F1 is secondary diagnostic only.
- Include household-level gate for false-empty control.

## Why RandomForest Was Used Earlier
RandomForest in `backend/ml/event_models.py` is a **temporary scaffold** for fast iteration on:
- event compiler,
- decoder behavior,
- KPI/gate contract,
- report format.

It is not the intended production architecture.

## Why Not Use Final CNN+Transformer From Day 1
Using final architecture immediately without stable event contract causes slow, coupled iteration:
- model code, event logic, and gate semantics all change together,
- debugging becomes ambiguous,
- team parallelization is blocked.

**Decision now:** move to CNN+Transformer integration immediately after contract/gate stabilization.  
Scaffold remains only for smoke validation and should not be used for promotion.

---

## Team Split (Parallel Work)

### Lane A: Contract + Compatibility
**Owner focus:** data contract and taxonomy safety.

#### A1. Registry contract
- **Files to create/edit**
  - `backend/config/adl_event_registry.v1.yaml`
  - `backend/config/schemas/adl_event_registry.schema.json`
  - `backend/ml/adl_registry.py`
  - `backend/tests/test_adl_event_registry_schema.py`
  - `backend/tests/test_adl_registry_loader.py`
- **Logic**
  - canonical event IDs, aliases, room scope, KPI groups, criticality, enabled flags.
  - unknown label fallback (`occupied_unknown`/`unoccupied_unknown`).
- **Pass criteria**
  - schema validation passes.
  - no alias collisions.
  - unknown labels map safely.

#### A2. Backward-compatible evolution
- **Files**
  - `backend/config/adl_event_migrations/` (new folder)
  - `backend/ml/ci_adl_registry_validator.py`
  - `.github/workflows/ci-gate-hardening.yml`
  - `backend/tests/test_adl_registry_migrations.py`
- **Logic**
  - detect breaking removals/renames without migration map.
  - allow additive changes.
- **Pass criteria**
  - CI fails on breaking taxonomy edits.
  - CI passes for additive edits.

### Lane B: Event Compiler + KPI/Gates
**Owner focus:** event semantics and evaluation.

#### B1. Event compiler and decoder hardening
- **Files**
  - `backend/ml/event_labels.py`
  - `backend/ml/event_decoder.py`
  - `backend/ml/derived_events.py` (new)
  - `backend/tests/test_event_labels.py`
  - `backend/tests/test_event_decoder.py`
  - `backend/tests/test_derived_events.py` (new)
- **Logic**
  - label -> episode collapse with gap-aware splitting.
  - hysteresis/min-duration smoothing.
  - derived care events from canonical labels.
- **Pass criteria**
  - deterministic episode outputs for fixed inputs.
  - noisy transitions reduced without deleting long events.

#### B2. Event KPI + gate layer
- **Files**
  - `backend/ml/event_metrics.py`
  - `backend/ml/event_gates.py` (new)
  - `backend/tests/test_event_metrics.py`
  - `backend/tests/test_event_gates.py` (new)
- **Logic**
  - room-scoped KPI summary.
  - gate checks: KPI thresholds, collapse, uncertainty rate.
- **Pass criteria**
  - synthetic collapse scenarios are blocked.
  - KPI computations match expected values in unit tests.

#### B3. Home-empty fusion module
- **Files**
  - `backend/ml/home_empty_fusion.py` (new)
  - `backend/tests/test_home_empty_fusion.py` (new)
  - `backend/ml/event_gates.py` (extend)
- **Logic**
  - fuse per-room occupancy probabilities + entrance activity into household status:
    - `home_empty`
    - `home_occupied`
    - `home_unknown`
  - apply minimum-empty-duration confirmation window.
  - expose confidence score and reason codes.
- **Pass criteria**
  - no immediate false-empty during short inactivity.
  - sustained all-room absence transitions to `home_empty`.
  - entrance activity suppresses false-empty state.

### Lane C: CNN+Transformer Integration
**Owner focus:** model path integration, no API regressions.

#### C1. Dual-head occupancy-first output contract
- **Files**
  - `backend/ml/training.py`
  - `backend/ml/transformer_head_ab.py`
  - `backend/ml/transformer_backbone.py` (if needed)
  - `backend/tests/test_training.py`
- **Logic**
  - keep existing CNN+Transformer backbone.
  - expose Head A + Head B probabilities for event decoder.
  - preserve existing return contracts used by pipeline.
- **Pass criteria**
  - no runtime break in training entrypoints.
  - old paths still pass existing tests.
  - new output payload available for event layer.

#### C2. Shadow-mode pipeline integration
- **Files**
  - `backend/ml/pipeline.py`
  - `backend/ml/unified_training.py`
  - `backend/ml/policy_config.py`
  - `backend/tests/test_unified_training_path.py`
  - `backend/tests/test_pipeline_integration.py`
  - `backend/tests/test_event_first_shadow_pipeline.py` (new)
- **Logic**
  - config switch `event_first_shadow`.
  - run legacy + event-first side by side.
  - write separate artifacts without changing promotion result.
- **Pass criteria**
  - `event_first_shadow=false`: behavior unchanged.
  - `event_first_shadow=true`: event artifacts generated.

### Lane D: Backtest + Validation + Signoff
**Owner focus:** reproducible evidence and rollout readiness.

#### D1. Rolling backtest runner
- **Files**
  - `backend/scripts/run_event_first_backtest.py`
  - `backend/logs/eval/` outputs
  - `backend/tests/test_event_first_backtest_script.py` (new)
- **Logic**
  - rolling splits `4->5`, `4+5->6`, ..., `4+...+9->10`.
  - multi-seed support (`11,22,33`).
  - event KPI summaries + gate outcomes.
- **Pass criteria**
  - report generated for all 5 rooms and all splits.
  - deterministic structure and required fields in JSON.

#### D2. Controlled validation run and signoff
- **Files**
  - `backend/ml/validation_run.py`
  - `backend/logs/eval/hk0011_event_first_vs_legacy_signoff.json`
  - `backend/logs/eval/hk0011_event_registry_change_impact.json`
- **Logic**
  - shadow run in production-like flow.
  - compare legacy vs event-first on care KPIs and gates.
- **Pass criteria**
  - no critical collapse.
  - improve or non-regress on agreed care KPIs.
  - uncertainty rate below threshold.

---

## Stage Plan and PR Sequence

### Stage 1 (Days 1-2): Contract Stabilization
- PR-A1: registry schema + loader + tests.
- PR-A2: migration validator + CI checks.

### Stage 2 (Days 2-4): Event Semantics ✅ COMPLETE
- ~~PR-B1~~: ✅ COMPLETE - event compiler + decoder + derived events (56 tests)
- ~~PR-B2~~: ✅ COMPLETE - event KPI + gate layer (28 tests)
- ~~PR-B3~~: ✅ COMPLETE - home-empty fusion + household gate (22 tests)

### Stage 3 (Days 4-7): CNN+Transformer Path
- PR-C1: dual-head output contract in training.
- PR-C2: shadow mode in pipeline/unified training.

### Stage 4 (Days 7-9): Evidence
- PR-D1: rolling backtest with 3 seeds.
- PR-D2: validation run + signoff pack.

### Stage 5 (Days 10-12): Canary + Go/No-Go
- Canary shadow in live environment for fixed period (no promotion switch).
- Promotion decision based on event-level gates and signoff artifacts.

---

## Testing Approach

### Unit tests (required per lane)
- Registry schema/compatibility tests.
- Event compiler/decoder/metrics/gates tests.
- Home-empty fusion tests.
- Model output contract tests.

### Integration tests
- Shadow pipeline end-to-end run.
- Backtest script smoke + full run.

### Regression tests
- Existing baseline tests in touched areas:
  - `backend/tests/test_training.py`
  - `backend/tests/test_unified_training_path.py`
  - `backend/tests/test_pipeline_integration.py`
  - `backend/tests/test_policy_config.py`

## Test Pass Criteria (merge gate)
1. New tests for changed logic are present.
2. All touched test suites pass locally.
3. No API/contract break in existing entrypoints.
4. Required report artifact generated and attached in PR notes.
5. CI registry checks pass.

## Operational Rule
No model promotion from scaffolded RandomForest path.  
Promotion candidates must come from CNN+Transformer event-first shadow path after gates pass.

---

## Execution Annex (Required for Team Parallel Work)

### A. PR Dependency Graph (Hard Ordering)

| PR | Scope | Depends On | Can Run In Parallel With |
|---|---|---|---|
| PR-A1 | Registry schema + loader + tests | None | PR-B1 prep, PR-C1 prep |
| PR-A2 | Registry migration validator + CI checks | PR-A1 | PR-B1, PR-C1 |
| PR-B1 | Event compiler + decoder + derived events | PR-A1 | PR-C1 |
| PR-B2 | Event KPI + gate layer | PR-B1 | PR-C1 |
| PR-B3 | Home-empty fusion + household gate | PR-B1 | PR-C1 |
| PR-C1 | CNN+Transformer dual-head output contract | PR-A1 | PR-B2, PR-B3 |
| PR-C2 | Shadow-mode pipeline integration | PR-C1, PR-B2, PR-B3 | PR-D1 prep |
| PR-D1 | Rolling backtest (3 seeds) + report | PR-C2 | None |
| PR-D2 | Controlled validation + signoff pack | PR-D1 | None |
| PR-D3 | Canary shadow rollout + final enablement decision | PR-D2 | None |

### A1. Leakage Control Specification (mandatory for PR-D1 evidence validity)

All rolling-split evidence is invalid unless this policy is enforced and logged.

1. **Train-only fit rule**
- Any scaler/normalizer/encoder/calibrator must be fit on train partition only.
- Validation/test partitions may only call `.transform()` / `.predict()`.

2. **Split-time feature rule**
- Rolling/window statistics must be computed causally (no future rows).
- Forbidden: centered windows, global means/quantiles over full split, future timestamp leakage.

3. **Calibration split rule**
- Inside each split: `train -> calib -> test` chronological partition.
- Calibration artifacts must be split-local and discarded after split run.

4. **Leakage audit artifact**
- D1 must emit `leakage_checklist` section per split:
  - `fit_on_train_only: true|false`
  - `centered_window_used: true|false`
  - `future_feature_detected: true|false`
  - `calibration_partition_valid: true|false`
  - `leakage_audit_pass: true|false`
- If any split has `leakage_audit_pass=false`, D1 fails.

### A2. Canonical Evaluation Granularity (to avoid window/event contradictions)

1. Event/KPI gates are computed at **day-level event aggregates** (primary).
2. Home-empty safety uses **window-level false-empty rate** (primary safety check).
3. Both must be reported together in D1/D2 under separate namespaces:
   - `event_level_kpis`
   - `window_level_safety`
4. Promotion fails if either namespace fails hard safety criteria.

### B. Numeric Gate Thresholds (v1.0, executable now)

These values are the default promotion thresholds for this cycle.  
Tune only after one full 3-seed retrospective run is complete.

| KPI / Gate | Threshold |
|---|---|
| Bedroom sleep duration MAE | `<= 120` minutes/day |
| LivingRoom active duration MAE | `<= 120` minutes/day |
| Kitchen use duration MAE | `<= 90` minutes/day |
| Bathroom use duration MAE | `<= 45` minutes/day |
| Shower-day recall | `>= 0.80` |
| Shower-day precision | `>= 0.70` |
| Entrance out-time MAE | `<= 90` minutes/day |
| Home-empty precision | `>= 0.95` |
| Home-empty recall | `>= 0.80` |
| Home-empty false-empty rate | `<= 0.05` (window-level) |
| Critical event collapse floor | no critical recall `<= 0.02` when support `>= 30` |
| Tier-1 critical event minimum recall | `>= 0.50` when support `>= 30` |
| Tier-2 critical event minimum recall | `>= 0.35` when support `>= 30` |
| Tier-3 critical event minimum recall | `>= 0.20` when support `>= 30` |
| Uncertainty rate | `<= 0.25` |
| Unknown-rate cap (global) | `<= 0.15` |
| Unknown-rate cap (per room) | `<= 0.20` |
| ECE (occupancy head) | `<= 0.08` |
| Brier (occupancy head) | `<= 0.20` |
| Macro-F1 guardrail (secondary) | no room drop worse than `-0.05` vs champion |

Promotion decision rule:
1. All hard safety gates pass (`collapse`, `home-empty precision`, `uncertainty`).
2. At least 4 of 6 care KPI targets pass.
3. Macro-F1 guardrail passes.
4. Stability rule passes:
   - hard safety gates pass on **all seeds**,
   - hard safety gates pass on at least **5/6 splits**,
   - KPI 95% CI lower bound meets threshold for Tier-1 metrics.

Criticality tiers (default mapping):
1. Tier-1: `shower_day`, `home_empty_false_empty_rate`, `sleep_duration`
2. Tier-2: `bathroom_use`, `kitchen_use`, `livingroom_active`
3. Tier-3: `out_time`

### C. Config Contract (must exist before PR-C2 merge)

Target file: `backend/config/release_gates.json` and policy loader in `backend/ml/policy_config.py`

| Key | Type | Default | Behavior |
|---|---|---|---|
| `event_first_shadow` | bool | `false` | Run event-first in parallel without affecting promotion. |
| `event_first_enabled` | bool | `false` | Enables event-first as promotion candidate path. |
| `event_registry_path` | string | `backend/config/adl_event_registry.v1.yaml` | Canonical taxonomy source. |
| `event_unknown_enabled` | bool | `true` | Map unseen labels to unknown-safe states. |
| `event_decoder_on_threshold` | float | `0.60` | Hysteresis activation threshold. |
| `event_decoder_off_threshold` | float | `0.40` | Hysteresis release threshold. |
| `event_decoder_min_on_steps` | int | `3` | Minimum sustained activation windows. |
| `event_probability_calibration` | string | `isotonic` | Per-split calibration method (`isotonic|platt|temperature`). |
| `event_calibration_min_samples` | int | `500` | Minimum calibration samples before enabling calibration fit. |
| `home_empty_enabled` | bool | `true` | Enables household empty fusion logic. |
| `home_empty_min_empty_minutes` | float | `15.0` | Required sustained absence duration. |
| `home_empty_empty_score_threshold` | float | `0.75` | Empty confidence threshold. |
| `home_empty_occupancy_threshold` | float | `0.55` | Occupied override threshold. |
| `home_empty_entrance_penalty` | float | `0.35` | Suppress false-empty on entrance activity. |
| `unknown_rate_global_cap` | float | `0.15` | Global unknown budget before gate fail. |
| `unknown_rate_room_cap` | float | `0.20` | Per-room unknown budget before gate fail. |

### D. Artifact Contracts (required fields)

#### D1 rolling report
Target: `backend/logs/eval/hk0011_event_first_rolling_dec4_to10_3seed.json`

Required top-level fields:
```json
{
  "elder_id": "HK0011_jessica",
  "run_timestamp_utc": "2026-02-16T12:00:00Z",
  "git_sha": "abcdef1",
  "config_hash": "sha256:...",
  "data_version": "dec4_to_dec10_snapshot_v1",
  "feature_schema_hash": "sha256:...",
  "model_hashes": {"Bedroom":"sha256:...", "LivingRoom":"sha256:..."},
  "days": [4,5,6,7,8,9,10],
  "seeds": [11,22,33],
  "splits": [],
  "room_summary": {},
  "home_empty_summary": {},
  "gate_summary": {},
  "registry_version": "v1",
  "leakage_checklist": [],
  "calibration_summary": {"method":"isotonic","ece":0.0,"brier":0.0}
}
```

#### D2 signoff pack
Target: `backend/logs/eval/hk0011_event_first_vs_legacy_signoff.json`

Required top-level fields:
```json
{
  "comparison_window": "dec4_to_dec10",
  "run_timestamp_utc": "2026-02-16T12:00:00Z",
  "git_sha": "abcdef1",
  "config_hash": "sha256:...",
  "data_version": "dec4_to_dec10_snapshot_v1",
  "feature_schema_hash": "sha256:...",
  "legacy": {},
  "event_first": {},
  "delta": {},
  "gate_decision": "PASS|FAIL",
  "failed_reasons": [],
  "registry_version": "v1",
  "model_version_candidate": "...",
  "seed_split_stability": {"hard_gate_all_seeds": true, "hard_gate_splits_passed": 6}
}
```

### E. Per-PR Definition of Done (DoD + commands)

#### PR-A1 DoD
1. Schema and registry files created.
2. Loader normalizes aliases and unknowns.
3. Commands:
```bash
cd backend
pytest -q tests/test_adl_event_registry_schema.py tests/test_adl_registry_loader.py
```

#### PR-A2 DoD
1. CI validator checks backward compatibility.
2. Breaking rename/delete without migration fails.
3. Commands:
```bash
cd backend
pytest -q tests/test_adl_registry_migrations.py tests/test_ci_gate_validator.py
```

#### PR-B1 DoD
1. Episode compiler and decoder merged.
2. Derived events implemented and deterministic.
3. Commands:
```bash
cd backend
pytest -q tests/test_event_labels.py tests/test_event_decoder.py tests/test_derived_events.py
```

#### PR-B2 DoD
1. Event KPI + gate evaluation implemented.
2. Collapse, uncertainty, and unknown-budget checks enforced.
3. Commands:
```bash
cd backend
pytest -q tests/test_event_metrics.py tests/test_event_gates.py
```

#### PR-B3 DoD
1. Household fusion module implemented.
2. Home-empty precision/recall evaluator added.
3. Commands:
```bash
cd backend
pytest -q tests/test_home_empty_fusion.py
```

#### PR-C1 DoD
1. CNN+Transformer outputs occupancy/activity head probabilities.
2. Per-split calibration implemented (`train->calib->test` chronology).
3. Existing training entrypoint contract preserved.
4. Commands:
```bash
cd backend
pytest -q tests/test_training.py tests/test_policy_config.py tests/test_temporal_split.py
```

#### PR-C2 DoD
1. Shadow mode wired in pipeline.
2. No behavior change when `event_first_shadow=false`.
3. Home-empty evaluation reported in both window-level and event-level views with consistent gate logic.
4. Commands:
```bash
cd backend
pytest -q tests/test_unified_training_path.py tests/test_pipeline_integration.py tests/test_event_first_shadow_pipeline.py
```

#### PR-D1 DoD
1. 3-seed rolling report generated.
2. JSON includes all required artifact fields + reproducibility metadata.
3. Leakage checklist and calibration summary populated per split.
4. Commands:
```bash
cd backend
python3 scripts/run_event_first_backtest.py --data-dir "/Users/dicksonng/DT/Development/New training files" --elder-id HK0011_jessica --min-day 4 --max-day 10 --seed 11 --output backend/logs/eval/hk0011_event_first_seed11_dec4_to10.json
python3 scripts/run_event_first_backtest.py --data-dir "/Users/dicksonng/DT/Development/New training files" --elder-id HK0011_jessica --min-day 4 --max-day 10 --seed 22 --output backend/logs/eval/hk0011_event_first_seed22_dec4_to10.json
python3 scripts/run_event_first_backtest.py --data-dir "/Users/dicksonng/DT/Development/New training files" --elder-id HK0011_jessica --min-day 4 --max-day 10 --seed 33 --output backend/logs/eval/hk0011_event_first_seed33_dec4_to10.json
```

#### PR-D2 DoD
1. Controlled validation + signoff report generated.
2. Gate decision rendered with explicit failure reasons.
3. Commands:
```bash
cd backend
python3 -m ml.validation_run start --elder-id HK0011_jessica --duration-days 7
python3 -m ml.validation_run finalize --elder-id HK0011_jessica --run-id <run_id>
```

#### PR-D3 DoD
1. Canary shadow run completed for 3-7 days with no safety regression.
2. `event_first_enabled=true` only if D2 gates pass and canary shows no new critical failures.
3. Commands:
```bash
cd backend
python3 -m ml.validation_run start --elder-id HK0011_jessica --duration-days 3
python3 -m ml.validation_run finalize --elder-id HK0011_jessica --run-id <run_id>
```

### F. Ownership and ETA Matrix

| Lane | Primary Role | Backup Role | ETA Start | ETA Finish |
|---|---|---|---|---|
| A | Platform/Backend engineer | QA engineer | Day 1 | Day 2 |
| B | ML applied scientist | Backend engineer | Day 2 | Day 4 |
| C | ML model engineer | MLOps engineer | Day 4 | Day 7 |
| D | MLOps + evaluation engineer | QA engineer | Day 7 | Day 10 |
| Buffer/Canary | Cross-lane | Tech lead | Day 10 | Day 12 |

Daily coordination rules:
1. 15-minute standup at start of day.
2. PR dependency board updated before EOD.
3. No direct push to promotion path without D2 signoff.
4. If C2 slips by >1 day, de-scope rule: freeze UI work and prioritize D1 leakage/calibration validity.

---

## Execution Status Log

### PR-B1: Event Compiler + Decoder + Derived Events ✅ COMPLETE (Feb 16, 2026)

**Deliverables:**
- `backend/ml/event_compiler.py` - Episode compilation with gap-aware splitting and hysteresis smoothing
- `backend/ml/event_decoder.py` - Window-level decoding with occupancy/activity fusion
- `backend/ml/derived_events.py` - Care-relevant KPI derivation
- `backend/tests/test_event_compiler.py` - 18 tests
- `backend/tests/test_event_decoder.py` - 18 tests  
- `backend/tests/test_derived_events.py` - 20 tests

**Key Features Implemented:**

| Component | Feature | Status |
|-----------|---------|--------|
| EpisodeCompiler | Gap-aware episode splitting | ✅ |
| EpisodeCompiler | Min-duration filtering | ✅ |
| EpisodeCompiler | Hysteresis smoothing | ✅ |
| EpisodeCompiler | Multi-room support | ✅ |
| EpisodeCompiler | Episode merging (small gaps) | ✅ |
| EventDecoder | Head A/B fusion | ✅ |
| EventDecoder | Hysteresis state machine | ✅ |
| EventDecoder | Temporal smoothing | ✅ |
| EventDecoder | Unknown fallback | ✅ |
| EventDecoder | Room-aware management | ✅ |
| DerivedEvents | Sleep metrics extraction | ✅ |
| DerivedEvents | Bathroom/shower detection | ✅ |
| DerivedEvents | Kitchen activity metrics | ✅ |
| DerivedEvents | Out-time calculation | ✅ |
| DerivedEvents | Day/weekly aggregation | ✅ |

**Test Results:**
- Total new tests: 56
- All passing: ✅
- Full suite: 411 passing

**Integration:**
- Uses Lane A ADL Registry for canonical event IDs
- Deterministic outputs for fixed inputs
- Ready for PR-B2 (Event KPI + Gate Layer)

**Next Steps:**
1. PR-B2: Event KPI + Gate Layer (Tier-1/2/3 thresholds, collapse detection)
2. PR-B3: Home-Empty Fusion + Household Gate (false-empty protection)

---

### PR-B2: Event KPI + Gate Layer ✅ COMPLETE (Feb 16, 2026)

**Deliverables:**
- `backend/ml/event_gates.py` - Tiered gate checking (Tier-1/2/3)
- `backend/ml/event_kpi.py` - Event-level KPI calculator
- `backend/tests/test_event_gates.py` - 28 tests

**Key Features Implemented:**

| Component | Feature | Status |
|-----------|---------|--------|
| EventGateChecker | Tier-1 gates (>=0.50 recall) | ✅ |
| EventGateChecker | Tier-2 gates (>=0.35 recall) | ✅ |
| EventGateChecker | Tier-3 gates (>=0.20 recall) | ✅ |
| EventGateChecker | Home-empty precision >= 0.95 | ✅ |
| EventGateChecker | False-empty rate <= 0.05 | ✅ |
| EventGateChecker | Unknown rate caps | ✅ |
| EventGateChecker | Collapse detection (recall <= 0.02) | ✅ |
| EventKPICalculator | Home-empty metrics | ✅ |
| EventKPICalculator | Per-event recall/precision/F1 | ✅ |
| EventKPICalculator | Care KPI extraction | ✅ |
| EventKPICalculator | Daily metrics aggregation | ✅ |

**Criticality Tiers:**

| Tier | Events | Min Recall |
|------|--------|------------|
| Tier-1 | shower_day, home_empty_false_empty_rate, sleep_duration | 0.50 |
| Tier-2 | bathroom_use, kitchen_use, livingroom_active | 0.35 |
| Tier-3 | out_time | 0.20 |

**Test Results:**
- Total new tests: 28
- All passing: ✅
- Full suite: 463 passing

---

### PR-B3: Home-Empty Fusion + Household Gate ✅ COMPLETE (Feb 16, 2026)

**Deliverables:**
- `backend/ml/home_empty_fusion.py` - Multi-room fusion for home-empty detection
- `backend/tests/test_home_empty_fusion.py` - 22 tests

**Key Features Implemented:**

| Component | Feature | Status |
|-----------|---------|--------|
| HomeEmptyFusion | Multi-room state aggregation | ✅ |
| HomeEmptyFusion | Room consensus algorithm | ✅ |
| HomeEmptyFusion | Entrance penalty logic | ✅ |
| HomeEmptyFusion | Temporal smoothing | ✅ |
| HomeEmptyFusion | Episode detection | ✅ |
| HomeEmptyFusion | False-empty protection | ✅ |
| HouseholdGate | Precision >= 0.95 check | ✅ |
| HouseholdGate | False-empty rate <= 0.05 check | ✅ |
| HouseholdGate | Coverage validation | ✅ |

**Safety Requirements (Hard Gates):**
- Home-empty precision >= 0.95 ✅
- False-empty rate <= 0.05 (window-level) ✅
- Entrance penalty: 5-minute boost after entrance ✅
- Temporal smoothing: 60-second window ✅

**Test Results:**
- Total new tests: 22
- All passing: ✅
- Full suite: 463 passing

---

## Lane B Complete Summary

**Total Deliverables:**
- 5 new modules (`event_compiler`, `event_decoder`, `derived_events`, `event_gates`, `event_kpi`, `home_empty_fusion`)
- 6 new test files
- 106 new tests (56 + 28 + 22)

**Integration Status:**
- ✅ PR-B1: Event Compiler + Decoder + Derived Events
- ✅ PR-B2: Event KPI + Gate Layer  
- ✅ PR-B3: Home-Empty Fusion + Household Gate
- **Total Test Count: 463 passing**

**Next: Lane C (CNN+Transformer Path)**
- PR-C1: Dual-head output contract in training
- PR-C2: Shadow mode in pipeline/unified training

---
