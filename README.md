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

1. Runtime watcher pipeline (`backend/run_daily_analysis.py`) for ingestion, training, inference, and timeline generation.
2. Streamlit workflow (`backend/app/main.py`) for export, labeling, corrections, and model controls (legacy fallback: `backend/export_dashboard.py`).
3. Web API/UI (`web-ui`) for resident dashboards and operations.
4. Event-first evaluation toolchain (`backend/scripts/run_event_first_*.py`) for smoke/matrix/go-no-go.

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
4. Scoped runtime `unknown` emission is available behind flags (`RUNTIME_UNKNOWN_*`) and is default-off.
