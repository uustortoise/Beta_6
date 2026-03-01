# Golden Sample Ops SOP (Beta_5.5)

## Purpose
Collect Golden Samples (target `N=20`) with a deterministic operator flow: preflight -> start stack -> ingest files -> verify UIs/APIs -> export/harvest -> shutdown.

## 0) Set Workspace
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5
```

## 1) Preflight
```bash
./ops_doctor.sh preflight
```

## 2) Start Platform (Operator Mode)
```bash
./start-ops.sh --no-scan --skip-pull --skip-install
```

Expected:
- Web UI: `http://localhost:3002`
- Streamlit Export UI: `http://localhost:8503`

## 3) Runtime Health Check
```bash
./ops_doctor.sh running
curl -sS -o /tmp/beta55_health.json -w "HTTP %{http_code}\n" http://localhost:3002/api/health
curl -sS -o /tmp/beta55_status.json -w "HTTP %{http_code}\n" http://localhost:3002/api/status
```

Pass criteria:
- `ops_doctor.sh running` shows both UIs reachable.
- `/api/health` and `/api/status` return `HTTP 200`.

## 4) Ingest New Raw Files
Drop files into:
```bash
/Users/dicksonng/DT/Development/Beta_5.5/data/raw
```

Watch ingestion logs:
```bash
tail -f /Users/dicksonng/DT/Development/Beta_5.5/backend/logs/pipeline.log
```

Watcher process check:
```bash
ps aux | grep run_daily_analysis.py | grep -v grep
```

## 5) Streamlit Export/Prediction Check
Open Streamlit:
```bash
open http://localhost:8503
```

In UI:
1. Go to `📤 Data Export` and run one export for target resident.
2. Go to `🏷️ Labeling Studio`, load a file/day, confirm prediction pre-population appears in `activity`.
3. Download output (Excel/CSV) once to confirm export button path works.

API sanity for training review queue:
```bash
curl -sS -o /tmp/beta55_candidates.json -w "HTTP %{http_code}\n" http://localhost:3002/api/candidates
```

Pass criteria:
- `HTTP 200` on `/api/candidates`.
- Labeling Studio shows prefilled predictions (if matching `adl_history` exists for selected resident/room/day).

## 6) Harvest Golden Samples JSON
Dry-run first:
```bash
python3 /Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/harvest_gold_samples.py --dry-run
```

Safe-only harvest (recommended):
```bash
python3 /Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/harvest_gold_samples.py \
  --filter-safe-only \
  --output /Users/dicksonng/DT/Development/Beta_5.5/data/golden_samples
```

Single-resident harvest (optional):
```bash
python3 /Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/harvest_gold_samples.py \
  --elder-id HK0011_jessica \
  --filter-safe-only \
  --output /Users/dicksonng/DT/Development/Beta_5.5/data/golden_samples
```

## 7) Operator Knobs To Track During Collection
1. `--filter-safe-only`: keep enabled for safer mass-label quality.
2. Resident scope: use `--elder-id` for controlled batches.
3. Target count: stop batch when exported valid samples reaches planned milestone (`N=20`).

Quick count latest export:
```bash
latest=$(ls -t /Users/dicksonng/DT/Development/Beta_5.5/data/golden_samples/golden_dataset_*.json | head -n 1)
python3 - <<'PY'
import json, os
p = os.popen("ls -t /Users/dicksonng/DT/Development/Beta_5.5/data/golden_samples/golden_dataset_*.json | head -n 1").read().strip()
obj = json.load(open(p))
print("file:", p)
print("sample_count:", obj.get("metadata", {}).get("sample_count"))
print("activities:", obj.get("metadata", {}).get("activities"))
PY
```

## 8) Shutdown
```bash
./stop.sh
```

Confirm clean stop:
```bash
ps aux | grep -E "run_daily_analysis.py|streamlit run|next start|next dev" | grep -v grep
```

