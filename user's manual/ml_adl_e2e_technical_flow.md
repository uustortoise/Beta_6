# Beta 6 ML + ADL E2E Technical Flow

## 1) Purpose
This is the runtime source-of-truth for Beta 6:
- raw file ingestion and routing,
- resident-batched retraining and model promotion,
- inference and ADL persistence,
- segment/timeline generation,
- correction and label authority behavior.

## 2) Runtime Topology

### Services
- Watcher: `backend/run_daily_analysis.py`
- File processor: `backend/process_data.py`
- ML orchestrator: `backend/ml/pipeline.py` (`UnifiedPipeline`)
- Training core: `backend/ml/training.py`
- Prediction core: `backend/ml/legacy/prediction.py` (via `backend/ml/prediction.py` wrapper)
- Registry/versioning: `backend/ml/legacy/registry.py` (via `backend/ml/registry.py` wrapper)
- Streamlit Studio default: `backend/app/main.py` (legacy fallback `backend/export_dashboard.py`)
- Web UI: `web-ui` (`http://localhost:3002`)

### Data stores
- Primary DB: PostgreSQL/Timescale (default runtime path)
- Core row table: `adl_history`
- Timeline segment table: `activity_segments`
- Sidecar tables: `household_segments`, `context_episodes`, `trajectory_events`, `routine_anomalies`

## 3) Watcher Loop and File Routing
`run_daily_analysis.py` runs `job()` immediately, then every 30 seconds.

### File discovery
- Scans `data/raw` for `.xlsx`, `.xls`, `.parquet`.
- Ignores transient/system files (`~$*`, dotfiles).
- Excludes `_manual_` files from watcher path.

### Routing
- Training files (`*_train_*`): grouped by resident and executed as one batch per resident.
- Non-train files: processed individually through `process_file(...)`.

## 4) Training Path (Watcher Batch Mode)

### 4.1 Resolve retrain set
For each resident batch, watcher resolves training files with `RETRAIN_INPUT_MODE`:
1. `auto_aggregate` (default): incoming files + archived resident history, then deterministic dedupe.
2. `incoming_only`: only the current batch.
3. `manifest_only`: explicit file list from `RETRAIN_MANIFEST_PATH`.

Additional safeguards:
- Pilot evidence profile can enforce `incoming_only`.
- Preflight checks can block runs for lineage/policy inconsistencies.
- Fingerprint checks can skip retrain if `(data_fingerprint, policy_hash, code_version)` is unchanged.

### 4.2 Aggregated room training
Watcher calls:
- `UnifiedPipeline.train_from_files(aggregate_files, elder_id, defer_promotion=...)`

Inside `train_from_files(...)`:
1. Load all files through `load_sensor_data(..., resample=True)`.
2. Merge per-room data and deduplicate by timestamp with deterministic source precedence.
3. Denoise (when enabled).
4. Evaluate pre-training gate stack (`GateIntegrationPipeline`):
   - `CoverageContractGate`
   - `PostGapRetentionGate`
   - `ClassCoverageGate`
5. If pre-training gates pass, call
   - `TrainingPipeline.train_room_with_leakage_free_scaling(...)`
6. Training computes calibration thresholds, release/lane-B checks, and post-training statistical gate info.
7. Save artifacts via registry with versioning (`save_model_artifacts(...)`).
8. `defer_promotion` controls whether new versions are promoted immediately or kept candidate-only.

## 5) Run-Level Promotion and Rollback Gates
After per-room training metrics are produced, watcher applies run-level controls:
1. Decision-trace artifact gate.
2. Optional walk-forward promotion gate.
3. Backbone alignment gate.
4. Beta6 authority gate + fallback sync.
5. Global gate for run-level acceptance.

Outcomes:
- Promote deferred candidates when gate chain passes.
- Roll back or deactivate rooms when gate chain fails (policy dependent).
- Persist run metadata to `training_history`.

## 6) Training Timeline Materialization
`train_from_files(...)` returns metrics, not prediction payload rows.
Watcher then builds timeline rows from training labels via:
- `_build_legacy_training_timeline_results(...)`

Those rows are persisted to `adl_history`, then:
- `activity_segments` regeneration runs by day/room,
- hard-negative mining runs,
- household analysis runs.

## 7) Inference Path (Non-Train Files)
For non-train files:
1. Watcher calls `process_file(file)`.
2. `process_file(...)` runs `UnifiedPipeline.predict(...)`.
3. `predict(...)` loads resampled room data and active room models.
4. `PredictionPipeline.run_prediction(...)` emits:
   - `predicted_activity`
   - confidence
   - `predicted_top1_*` / `predicted_top2_*`
   - low-confidence fields
   - runtime unknown/abstain flags when enabled
5. Golden samples are applied (`apply_golden_samples(...)`).
6. Optional pre-persistence arbitration can fuse cross-room conflicts.
7. Rows are saved to `adl_history`; segments regenerated; downstream analyzers execute.

## 8) Label Authority Chain
Authoritative priority:
1. Manual corrections / golden samples (`is_corrected=1`)
2. Training labels (`activity` in training sheets)
3. Model predictions

Corrected rows are protected in persistence flow from accidental overwrite.

## 9) Data Contracts

### File naming
- Training: contains `_train`
- Inference: not `_train`

### Sheets/rooms
Expected room sheets commonly include:
- `Bedroom`, `LivingRoom`, `Kitchen`, `Bathroom`, `Entrance`

### Canonical timeline
- Loader/resampler normalizes to fixed interval (default 10 seconds).
- Missing/gap behavior follows policy-resolved forward-fill gap limits.

## 10) Event-First Evaluation (Separate from Runtime)
Promotion QA scripts remain separate from watcher runtime generation:
1. `backend/scripts/validate_label_pack.py`
2. `backend/scripts/diff_label_pack.py`
3. `backend/scripts/run_event_first_smoke.py`
4. `backend/scripts/run_event_first_matrix.py`
5. `backend/scripts/run_event_first_backtest.py`

Use this path for experiment/evaluation governance, not daily timeline ingestion.

## 11) Operational Commands

### Start stack
```bash
cd /Users/dicksonng/DT/Development/Beta_6
./ops_doctor.sh preflight
./start-ops.sh --no-scan --skip-pull --skip-install
./ops_doctor.sh running
```

### Stop stack
```bash
cd /Users/dicksonng/DT/Development/Beta_6
./stop.sh
```

## 12) Related docs
- `user's manual/data_flow_logic.md`
- `user's manual/operation_manual.md`
- `user's manual/labeling_guide.md`
- `user's manual/golden_sample_harvesting.md`
