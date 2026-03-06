# Beta 6 Operation Manual

## 1. Purpose
This manual is for engineering and ops execution of Beta 6 runtime and model operations.

## 2. Runtime Prerequisites
- macOS/Linux shell with `bash`/`zsh`
- Python environment for `backend`
- Node/npm for `web-ui`
- PostgreSQL/Timescale available (default runtime is PostgreSQL-first)

## 3. Startup and Shutdown

### Recommended team entrypoint
```bash
cd /Users/dicksonng/DT/Development/Beta_6
./ops_doctor.sh preflight
./start-ops.sh --no-scan --skip-pull --skip-install
./ops_doctor.sh running
```

### Stop services
```bash
cd /Users/dicksonng/DT/Development/Beta_6
./stop.sh
```

### Script roles
- `start-ops.sh`: team-safe wrapper; standard way for daily operations.
- `ops_start.sh`: legacy wrapper that chains pull/install/smoke/start.
- `start.sh`: core low-level bootstrap for services.
- `ops_doctor.sh`: health and preflight diagnostics.

## 4. Service Endpoints
- Web UI: `http://localhost:3002`
- Streamlit Studio: `http://localhost:8503`

## 5. Daily Ops Workflow
1. Run preflight (`ops_doctor.sh preflight`).
2. Start stack (`start-ops.sh ...`).
3. Confirm health (`ops_doctor.sh running`).
4. Ingest/process files through standard watcher flow.
5. Review timeline/alerts in UI.
6. Stop stack when done.

## 6. Label Pack Intake Workflow (Event-First)
Use this workflow when a new corrected training pack arrives.

### 6.1 Validate pack schema
```bash
cd /Users/dicksonng/DT/Development/Beta_6
python3 backend/scripts/validate_label_pack.py \
  --pack-dir "/Users/dicksonng/DT/Development/New training files" \
  --elder-id HK0011_jessica \
  --min-day 4 --max-day 10 \
  --output /tmp/beta6_label_pack_validation.json
```

### 6.2 Diff old vs new labels
```bash
python3 backend/scripts/diff_label_pack.py \
  --baseline-dir "/Users/dicksonng/DT/Development/New training files/corrected_clones" \
  --candidate-dir "/Users/dicksonng/DT/Development/New training files" \
  --elder-id HK0011_jessica \
  --min-day 4 --max-day 10 \
  --json-output /tmp/beta6_label_pack_diff.json \
  --csv-output /tmp/beta6_label_pack_diff.csv
```

### 6.3 Run smoke gate
```bash
python3 backend/scripts/run_event_first_smoke.py \
  --data-dir "/Users/dicksonng/DT/Development/New training files" \
  --elder-id HK0011_jessica \
  --day 7 --seed 11 \
  --expectation-config backend/config/event_first_go_no_go.yaml \
  --diff-report /tmp/beta6_label_pack_diff.json \
  --output /tmp/beta6_smoke.json
```

### 6.4 Run matrix profile
```bash
python3 backend/scripts/run_event_first_matrix.py \
  --profiles-yaml backend/config/event_first_matrix_profiles.yaml \
  --profile anchor_top2_frag_v3 \
  --data-dir "/Users/dicksonng/DT/Development/New training files" \
  --elder-id HK0011_jessica \
  --output-dir /tmp/beta6_matrix \
  --max-workers 3
```

## 7. Release Policy Notes (Current)
- Release policy is configured in `backend/config/event_first_go_no_go.yaml`.
- LivingRoom episode metrics are currently treated as informational for release (Option Y policy), while core room quality and MAE non-regression guards remain enforced.

## 8. Scoped Runtime `unknown` (Fail-Closed Enabled)
Runtime scoped `unknown` handling is enabled by default in current Beta 6 prediction code and is room-scoped by policy (default includes `livingroom`).
Use these settings for targeted ambiguity handling windows (for example LivingRoom at night).

Flags:
- `RUNTIME_UNKNOWN_ENABLED` (should remain `true` in fail-closed runtime)
- `RUNTIME_UNKNOWN_ROOMS` (comma-separated normalized rooms, example `livingroom`)
- `RUNTIME_UNKNOWN_NIGHT_ONLY` (`true|false`)
- `RUNTIME_UNKNOWN_NIGHT_HOURS` (`22-6` format)
- `RUNTIME_UNKNOWN_MIN_CONF` (max confidence for candidate windows, default `0.55`)
- `RUNTIME_UNKNOWN_RATE_GLOBAL_CAP` (default `0.12`)
- `RUNTIME_UNKNOWN_RATE_ROOM_CAP` (default `0.25`)

Example:
```bash
export RUNTIME_UNKNOWN_ENABLED=true
export RUNTIME_UNKNOWN_ROOMS=livingroom
export RUNTIME_UNKNOWN_NIGHT_ONLY=true
export RUNTIME_UNKNOWN_NIGHT_HOURS=22-6
export RUNTIME_UNKNOWN_MIN_CONF=0.55
export RUNTIME_UNKNOWN_RATE_GLOBAL_CAP=0.12
export RUNTIME_UNKNOWN_RATE_ROOM_CAP=0.25
```

## 9. Troubleshooting
- If startup fails, run:
```bash
./ops_doctor.sh preflight
./ops_doctor.sh running
```
- If APIs are unhealthy, check backend logs and DB connectivity first.
- If matrix/smoke fails, inspect JSON outputs in `/tmp` and `blocking_reasons` fields.

## 10. Related Docs
- Technical flow: `ml_adl_e2e_technical_flow.md`
- Labeling rules: `labeling_guide.md`
- Golden sample SOP: `golden_sample_harvesting.md`
- Planning SOP: `/Users/dicksonng/DT/Development/Beta_6/docs/planning/golden_sample_ops_sop_2026-02-25.md`
