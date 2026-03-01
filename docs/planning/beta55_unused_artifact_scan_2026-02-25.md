# Beta 5.5 Unused Artifact Scan (2026-02-25)

## Scope
- Repository root: `/Users/dicksonng/DT/Development/Beta_5.5`
- Purpose: identify non-core runtime artifacts likely safe to prune, and folders that are not part of `start.sh` runtime path.

## Core Runtime Path (from `start.sh`)
- Uses: `backend/`, `web-ui/`, `data/`, `ops_doctor.sh`, `stop.sh`
- Does not reference in startup flow: `health_advisory_chatbot/`, `Vital sign addon/`, `Beta6/`

## Likely Safe To Prune (generated/runtime artifacts)
- `web-ui/node_modules` (~626M)
- `backend/tmp` (~91M)
- `backend/validation_runs_canary` (~11M)
- `.pytest_cache` (~116K)
- `logs` (~16K)
- `automation.log`
- `export_ui.log`
- `.DS_Store`
- `backend/export_ui.log`
- `backend/logs/pipeline.log*`

## Review Before Prune (not startup-critical but may be intentionally kept)
- `health_advisory_chatbot` (~6.3M)
- `Vital sign addon` (~600K)
- `Beta6` (sandbox folder)

## Not flagged as unused
- `archive` (~736M) is used by training/evaluation/archive workflows and should not be removed as part of this cleanup.

## Team command
- Run non-destructive scan + startup through new operator entrypoint:
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5
./start-ops.sh
```

- Scan only:
```bash
./start-ops.sh --scan-only
```
