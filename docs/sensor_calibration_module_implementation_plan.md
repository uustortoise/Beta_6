# Sensor Calibration Module (Toggle On/Off) - Implementation Plan

## Objective
Build an optional sensor calibration module in Beta_6 that normalizes home-specific sensor baselines to reduce model instability across different home setups, while guaranteeing zero behavior change when calibration is disabled.

## Scope
In scope:
- Add a runtime toggle (`sensor_calibration_enabled`) with safe default `off`.
- Add per-home/per-resident calibration profile persistence.
- Apply calibration in both training and inference preprocessing.
- Add UI control and visibility in existing configuration panel.
- Add tests and rollout gates for no-regression and measurable benefit.

Out of scope for this phase:
- New model architectures.
- Replacing existing class-threshold calibration in training.
- Automatic online learning without explicit retrain/recalibration trigger.

## Current-State Anchors (already in code)
- Existing config toggle pattern: `household_config` and Household Overview form.
- Existing preprocessing insertion point: `preprocess_without_scaling(...)`.
- Existing calibration and threshold artifact infrastructure in training/registry.
- Existing release gates and no-regression checks in `release_gates.json`.

## Proposed Design
1. Toggle behavior:
- `OFF` (default): pipeline is unchanged; no calibration transform is applied.
- `ON`: apply per-sensor normalization before temporal/rolling feature generation.

2. Calibration profile:
- Profile unit: `elder_id + room + sensor`.
- Stored stats (initial version): `mean`, `std`, `p05`, `p95`, `sample_count`, `fit_window_start/end`, `updated_at`.
- Guardrails:
  - Require minimum samples per sensor.
  - Skip sensor calibration for insufficient support and use identity transform.
  - Clamp normalized values to configurable bounds to avoid instability.

3. Config precedence:
- Per-home DB setting (highest).
- Environment override (optional emergency kill switch).
- Default false.

4. Processing order:
- Resample and bounded gap handling.
- Sensor calibration (if enabled).
- Temporal features.
- Rolling features.
- Denoising/scaling as currently implemented.

## Files and Code Changes
1. Database schema and compatibility:
- `backend/db/schema.sql`
  - Add table `sensor_calibration_settings` keyed by `elder_id`.
  - Add table `sensor_calibration_profiles` keyed by (`elder_id`, `room`, `sensor_name`).
- `backend/elderlycare_v1_16/models/schema.sql`
  - Mirror new tables for SQLite/local bootstrap compatibility.
- `backend/db/legacy_adapter.py`
  - Extend `pk_map` for new tables to keep `INSERT OR REPLACE` behavior correct in PostgreSQL.

2. Calibration module:
- `backend/ml/sensor_calibration.py` (new)
  - `compute_profile(df, sensor_columns, config)`.
  - `apply_profile(df, profile, sensor_columns, config)`.
  - `profile_quality(profile)` and fallback logic.
- `backend/ml/sensor_calibration_store.py` (new)
  - Load/save settings and profiles via existing DB adapter.

3. Pipeline integration:
- `backend/elderlycare_v1_16/platform.py`
  - Inject calibration step inside `preprocess_without_scaling(...)` after gap handling and before feature engineering.
  - Keep strict identity behavior when disabled or profile missing.
- `backend/ml/pipeline.py`
  - For training: compute/update calibration profile before model fit.
  - For inference: load and apply latest profile for (`elder_id`, `room`).
- `backend/ml/training.py`
  - Add calibration profile metadata into decision trace payload.
- `backend/ml/registry.py` (optional but recommended)
  - Persist versioned calibration artifact alongside model artifacts for reproducibility.

4. UI and runtime controls:
- `backend/export_dashboard.py`
  - Add `Enable Sensor Calibration` checkbox in Household/Config panel.
  - Add fields for min samples and clipping bounds.
  - Save/read settings via calibration settings store.
- `backend/run_daily_analysis.py`
  - Include calibration summary in run metadata logs for ops visibility.

5. Config defaults:
- `backend/config/release_gates.json` (optional defaults block)
  - Add non-blocking defaults for calibration parameters if team prefers JSON-managed defaults.

## Testing Procedures
1. Unit tests:
- Add `backend/tests/test_sensor_calibration.py`:
  - Profile compute correctness on synthetic offsets/scales.
  - Identity fallback on low support.
  - Clipping bounds and NaN safety.
  - Toggle-off path returns unchanged frame.
- Add/extend:
  - `backend/tests/test_policy_config.py` for env override parsing (if env support added).
  - `backend/tests/test_registry.py` for artifact persistence (if registry integration added).
  - `backend/tests/test_postgres_compatibility_regressions.py` for upsert behavior on new tables.

2. Integration tests:
- Add `backend/tests/test_pipeline_sensor_calibration_integration.py`:
  - Training -> save profile -> inference apply profile.
  - Per-elder isolation (A profile does not affect B).
  - Missing profile path does not fail prediction.

3. Backtest/quality validation:
- Run event-first backtest matrix with calibration off vs on for same data slices.
- Compare:
  - Room F1 by room tier.
  - Global macro F1.
  - Low-confidence rate.
  - Empty-home related false positives (if impacted).

4. Operational checks:
- DB init/migration smoke in SQLite and PostgreSQL modes.
- Runtime latency overhead measurement on representative file.

## Suggested Test Commands
1. Unit and integration:
- `backend/venv/bin/python -m pytest -q backend/tests/test_sensor_calibration.py backend/tests/test_pipeline_sensor_calibration_integration.py backend/tests/test_registry.py backend/tests/test_postgres_compatibility_regressions.py`

2. Existing regression suites most likely impacted:
- `backend/venv/bin/python -m pytest -q backend/tests/test_prediction.py backend/tests/test_training.py backend/tests/test_unified_training_path.py backend/tests/test_pipeline_integration.py`

3. Backtest comparison (same dataset, off vs on):
- `backend/venv/bin/python backend/scripts/run_event_first_backtest.py --data-dir <DATA_DIR> --elder-id <ELDER_ID> --min-day 4 --max-day 10 --seed 42 --calibration-method isotonic`
- Run once with calibration toggle off, once on; compare generated report JSON metrics.

## Passing Requirements (Go/No-Go)
Functional:
- Toggle `OFF` must preserve current behavior:
  - Same predicted labels on fixed seed test fixtures.
  - Confidence deltas only within floating-point tolerance.
- Toggle `ON` must be safe:
  - No crashes when profiles are missing/partial.
  - No NaN/Inf in downstream features.
- Per-home isolation:
  - Calibration settings and profiles must be scoped by `elder_id`.

Quality:
- Must satisfy existing release gates in `backend/config/release_gates.json`.
- Must satisfy existing no-regression rule:
  - `room_f1` drop vs champion <= 0.05 (except already exempt rooms).
- For heterogeneous-home validation set, calibration-on should meet at least one:
  - >= 2 percentage-point macro F1 improvement, or
  - >= 15% reduction in low-confidence windows, or
  - >= 15% reduction in false positives for targeted behaviors.

Performance:
- End-to-end preprocessing overhead increase <= 10% median per file.

Operational:
- New schema applies cleanly in both SQLite and PostgreSQL paths.
- Rollback path available: disable toggle globally/per-home without redeploy.

## Rollout Plan
1. Phase 1 (safe skeleton):
- Implement toggle, identity calibration path, schema, tests.
- Keep default off.

2. Phase 2 (active calibration):
- Enable profile compute/apply in training + inference.
- Add UI controls and observability.

3. Phase 3 (pilot):
- Enable for 1-2 pilot homes only.
- Compare off/on weekly reports and gate metrics.

4. Phase 4 (general availability):
- Enable for new homes by default only after pass criteria are met.

## Risks and Mitigations
- Risk: global config contamination across homes.
- Mitigation: separate settings table keyed by `elder_id`; do not reuse global `household_config` key-only model.

- Risk: over-normalization can suppress true events.
- Mitigation: min-sample guardrails, clipping, and conservative defaults.

- Risk: SQL upsert incompatibility in PostgreSQL shim.
- Mitigation: update `pk_map` and add regression tests.

## Open Decisions for Team Sign-Off
1. Should calibration profile be stored only in DB, or both DB + versioned model artifact?
2. Should default for new homes be `OFF` (safer) or `ON` after pilot?
3. Which KPI is primary for heterogeneous-home success: macro F1, low-confidence rate, or false-positive reduction?
