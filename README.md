# Beta 6

Beta 6 is the active ADL timeline platform branch (runtime + evaluation toolchain).

## Quick start

```bash
cd /Users/dicksonng/DT/Development/Beta_6
./ops_doctor.sh preflight
./start-ops.sh --no-scan --skip-pull --skip-install
./ops_doctor.sh running
```

Endpoints:
- Web UI: `http://localhost:3002`
- Streamlit Studio: `http://localhost:8503`

Stop:
```bash
./stop.sh
```

## Operator entrypoints

- `./start-ops.sh`: team entrypoint wrapper (includes optional non-destructive artifact scan)
- `./ops_start.sh`: legacy operator wrapper (`pull -> preflight -> install -> smoke -> start`)
- `./ops_doctor.sh`: preflight/runtime checks
- `./start.sh`: core service bootstrap script

## Current technical scope

1. Runtime watcher pipeline (`backend/run_daily_analysis.py`) for ingestion, aggregated retraining, gated promotion, inference, and timeline regeneration.
2. Streamlit workflow (`backend/app/main.py`) for export, labeling, corrections, and model controls (legacy fallback: `backend/export_dashboard.py`).
3. Web API/UI (`web-ui`) for resident dashboards and operations.
4. Event-first evaluation toolchain (`backend/scripts/run_event_first_*.py`) for smoke/matrix/go-no-go.

## Runtime flow (actual code path)

1. Watcher scans `data/raw` every 30 seconds.
2. Files are split into:
   - Training batch per resident (`*_train_*`): one aggregated retrain run per resident.
   - Inference files (non-train): per-file prediction path.
3. Training batch path:
   - Training set is resolved by `RETRAIN_INPUT_MODE` (`auto_aggregate` default, or `incoming_only`, `manifest_only`).
   - `UnifiedPipeline.train_from_files(...)` trains per-room candidates with pre-training gates + post-training checks.
   - Models are saved versioned via registry; promotion may be deferred until run-level gates pass.
   - Run-level checks in watcher include decision-trace gate, optional walk-forward gate, backbone alignment gate, Beta6 authority gate, and global gate.
4. Inference file path:
   - `process_file(...)` -> `UnifiedPipeline.predict(...)` -> per-room model inference.
   - Prediction outputs include `predicted_activity`, confidence, top1/top2 metadata, low-confidence flags, and optional runtime unknown/abstain signals.
5. Persistence + downstream:
   - Rows are written to `adl_history`.
   - Segments are regenerated into `activity_segments`.
   - Downstream services run sleep, ICOPE, insight, household, and optional trajectory/pattern/context analysis.

## Documentation map

Primary docs:
- Technical E2E flow:
  - `user's manual/ml_adl_e2e_technical_flow.md`
- Golden sample ops SOP:
  - `docs/planning/golden_sample_ops_sop_2026-02-25.md`
- User manual index:
  - `user's manual/readme.md`

ML/labelling docs:
- Labeling guide:
  - `user's manual/labeling_guide.md`
- Golden sample harvesting:
  - `user's manual/golden_sample_harvesting.md`
- Non-ML handbook:
  - `user's manual/ml_module_handbook_non_ml.md`
- Data flow overview:
  - `user's manual/data_flow_logic.md`

## Notes

1. PostgreSQL is required in current default runtime (`POSTGRES_ONLY=true` path).
2. `adl_history` is the core source table for predictions/corrections.
3. `activity_segments` is the final room timeline table consumed by UI flows.
4. Scoped runtime `unknown` emission is controlled by `RUNTIME_UNKNOWN_*`; runtime is fail-closed enabled by default and scoped to configured rooms (default policy includes `livingroom`).
