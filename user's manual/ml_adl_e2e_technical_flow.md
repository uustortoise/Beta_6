# Beta 5.5 ML + ADL E2E Technical Flow

## 1) Purpose
This is the source-of-truth technical document for how Beta 5.5 processes data end-to-end:
- raw sensor files -> preprocessing -> model training/inference -> ADL timeline,
- manual corrections/labels -> retraining priority chain,
- event-first evaluation path (smoke/matrix/go-no-go).

Use this document for engineering follow-up and onboarding.

## 2) Runtime Topology

### Services
- Operator entrypoint: `./start-ops.sh` (wrapper) -> `./ops_start.sh` -> `./start.sh`
- Watcher: `backend/run_daily_analysis.py`
- Core processing: `backend/process_data.py`
- ML pipeline: `backend/ml/pipeline.py`
- Web UI: `web-ui` on `http://localhost:3002`
- Streamlit Studio: `backend/export_dashboard.py` on `http://localhost:8503`

### Data Stores
- Primary DB (production mode): PostgreSQL (`elderlycare_timescaledb`, port 5432)
- Key table for predictions/corrections: `adl_history`
- Timeline table: `activity_segments`
- Context/trajectory/anomaly tables: `household_segments`, `context_episodes`, `trajectory_events`, `routine_anomalies`

## 3) Input Contracts

### File types
- Training files: `*_train_*.xlsx` / `.xls` / `.parquet`
- Inference files: `*_input_*.xlsx` / `.xls` / `.parquet`

### Resident identity
- Derived from filename prefix: `HKxxxx_name_...`
- Extractor: `get_elder_id_from_filename(...)` in `backend/process_data.py`

### Room sheets
- Expected sheets: `Bedroom`, `LivingRoom`, `Kitchen`, `Bathroom`, `Entrance`
- Validation tooling: `backend/scripts/validate_label_pack.py`

## 4) Runtime E2E Path (Raw -> Timeline)

### Step A: Watcher detection
1. `backend/run_daily_analysis.py` scans `data/raw`.
2. Classifies training vs input file (`_train` marker).
3. Archives and routes each file for processing.

### Step B: Load + normalize
1. Loader: `backend/utils/data_loader.py` (`load_sensor_data`).
2. Timestamp parsing + optional resampling/cleaning (`clean_and_resample`).
3. Canonical resampling to fixed 10s interval uses:
   `elderlycare_v1_16/preprocessing/resampling.py`.

### Step C: Model execution
1. Entry: `process_file(...)` in `backend/process_data.py`.
2. Pipeline object: `UnifiedPipeline` from `backend/ml/pipeline.py`.
3. If `_train` file:
   - `pipeline.train_and_predict(...)` (training + prediction path).
4. Else input file:
   - `pipeline.predict(...)` (inference only).

### Step D: Prediction persistence
1. Each predicted row is written to `adl_history` via:
   `ADLService.save_adl_event(...)` in `backend/elderlycare_v1_16/services/adl_service.py`.
2. Persisted fields include:
   - `activity_type`, `confidence`, `room`, `timestamp`, `record_date`,
   - `sensor_features` JSON (including top1/top2 and low-confidence metadata).

### Step E: Timeline generation
1. Segment regeneration:
   - `generate_activity_segments(...)` in `backend/process_data.py`,
   - backed by `regenerate_segments(...)` in `backend/utils/segment_utils.py`.
2. Segment logic:
   - activity normalization/validation by room,
   - sleep continuity merge rules,
   - long segment chunking safeguards.

### Step F: Downstream intelligence
Triggered after ADL persistence:
- Sleep analysis (`SleepAnalyzer` + `SleepService`)
- ICOPE scoring (`ICOPEService`)
- Insight/alerts (`InsightService`)
- Household analyzer (`ml/household_analyzer.py`)
- Trajectory + pattern + context sidecars (if enabled)

## 5) Label Priority Chain (Critical)
Label authority order in Beta 5.5:
1. Golden Samples / manual corrections (`is_corrected = 1`)
2. Training file labels (`activity` column in train sheets)
3. Model predictions

Enforcement details:
- `save_adl_event(...)` prevents overwriting corrected rows.
- Correction history supports soft-delete and rollback with segment regeneration.

## 6) Core ML Training/Inference Mechanics

### Training
- Main code: `backend/ml/training.py`
- Architecture path: Transformer backbone (`ml/transformer_backbone.py`) with policy controls.
- Includes:
  - deterministic seeding,
  - class weighting/minority handling,
  - calibration logic,
  - gate integration (`ml/gate_integration.py`).

### Inference
- Main code: `backend/ml/prediction.py`
- Includes:
  - calibrated thresholds,
  - low-confidence labeling (`low_confidence`),
  - optional scoped runtime `unknown` conversion (from low-confidence windows) with room/night filters and unknown-rate caps (`RUNTIME_UNKNOWN_*`),
  - optional inference hysteresis (`ENABLE_INFERENCE_HYSTERESIS`).

## 7) Event-First Evaluation Toolchain (Model QA Path)
This is the controlled evaluation path used for smoke/matrix/go-no-go decisions.

### Tools
- Label-pack validator:
  - `backend/scripts/validate_label_pack.py`
- Baseline-vs-candidate diff:
  - `backend/scripts/diff_label_pack.py`
- Single-day smoke:
  - `backend/scripts/run_event_first_smoke.py`
- Full matrix runner:
  - `backend/scripts/run_event_first_matrix.py`
- Backtest core:
  - `backend/scripts/run_event_first_backtest.py`

### Typical run order
1. Validate incoming label pack.
2. Diff against baseline pack (confirm actual label deltas).
3. Run smoke on target day/seed.
4. Run matrix profile.
5. Aggregate + go/no-go using configured gates.

## 8) Ops Team: Adding New ADL Labels
Yes, ops can add labels without core code changes, via registry/config workflow.

### Required updates
1. Update ADL registry:
   - `backend/config/adl_event_registry.v1.yaml`
2. Keep room validity mapping aligned when needed:
   - `backend/utils/segment_utils.py` (`ROOM_ACTIVITY_VALIDATION`)
3. Re-run label pack validator to catch unknown labels.

### Safety check
- Unknown labels are surfaced by `validate_label_pack.py` and should be resolved before matrix/signoff.

## 9) Exact Command Blocks (Copy/Paste)

### Platform start + health
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5
./ops_doctor.sh preflight
./start-ops.sh --no-scan --skip-pull --skip-install
./ops_doctor.sh running
```

### Event-first intake/matrix
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5

python3 backend/scripts/validate_label_pack.py \
  --pack-dir "/Users/dicksonng/DT/Development/New training files" \
  --elder-id HK0011_jessica \
  --min-day 4 --max-day 10 \
  --output /tmp/beta55_label_pack_validation.json

python3 backend/scripts/diff_label_pack.py \
  --baseline-dir "/Users/dicksonng/DT/Development/New training files/corrected_clones" \
  --candidate-dir "/Users/dicksonng/DT/Development/New training files" \
  --elder-id HK0011_jessica \
  --min-day 4 --max-day 10 \
  --json-output /tmp/beta55_label_pack_diff.json \
  --csv-output /tmp/beta55_label_pack_diff.csv

python3 backend/scripts/run_event_first_smoke.py \
  --data-dir "/Users/dicksonng/DT/Development/New training files" \
  --elder-id HK0011_jessica \
  --day 7 --seed 11 \
  --expectation-config backend/config/event_first_go_no_go.yaml \
  --diff-report /tmp/beta55_label_pack_diff.json \
  --output /tmp/beta55_smoke.json

python3 backend/scripts/run_event_first_matrix.py \
  --profiles-yaml backend/config/event_first_matrix_profiles.yaml \
  --profile anchor_top2_frag_v3 \
  --data-dir "/Users/dicksonng/DT/Development/New training files" \
  --elder-id HK0011_jessica \
  --output-dir /tmp/beta55_matrix \
  --max-workers 3
```

### Golden sample harvest
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5
python3 backend/scripts/harvest_gold_samples.py --dry-run
python3 backend/scripts/harvest_gold_samples.py \
  --filter-safe-only \
  --output /Users/dicksonng/DT/Development/Beta_5.5/data/golden_samples
```

## 10) Known Constraints and Practical Notes
1. `POSTGRES_ONLY=true` environments require DB availability; several UI/API flows will fail if PostgreSQL is down.
2. LivingRoom passive occupancy remains the hardest reliability zone; release policy handles this via configured go/no-go logic.
3. Keep evaluation and runtime paths distinct:
   - runtime watcher drives product timeline,
   - event-first matrix drives model promotion decisions.

## 11) Related Docs
- Ops SOP:
  - `docs/planning/golden_sample_ops_sop_2026-02-25.md`
- Labeling guide:
  - `user's manual/labeling_guide.md`
- Golden sample guide:
  - `user's manual/golden_sample_harvesting.md`
- Non-ML handbook:
  - `user's manual/ml_module_handbook_non_ml.md`
