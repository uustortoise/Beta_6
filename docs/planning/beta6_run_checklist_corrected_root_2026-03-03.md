# Beta6 Run Checklist (Corrected Repo Root)

- Date: 2026-03-03
- Canonical repo root: `/Users/dickson/DT/DT_development`
- Active branch target: `codex/pilot-bootstrap-gates`

## 1. Branch and Remote Sanity

```bash
cd /Users/dickson/DT/DT_development
git remote -v
git fetch --prune origin
git checkout codex/pilot-bootstrap-gates
git pull --ff-only origin codex/pilot-bootstrap-gates
```

Expected:
1. `origin` points to `https://github.com/uustortoise/Beta_6.git`
2. Branch is `codex/pilot-bootstrap-gates`

## 2. Required Files Presence

```bash
for p in \
  backend/ml/utils.py \
  backend/config/release_gates.json \
  backend/utils/health_check.py \
  backend/utils/time_utils.py \
  backend/process_data.py \
  backend/run_daily_analysis.py \
  backend/health_server.py
do
  [ -f "$p" ] && echo "FOUND $p" || echo "MISSING $p"
done
```

## 3. Runtime Paths (Training Intake)

Training file intake path:
1. `/Users/dickson/DT/DT_development/data/raw`

Ensure runtime folders exist:

```bash
mkdir -p data/raw data/processed data/archive data/models backend/models backend/logs
```

## 4. Backend Environment

```bash
cd /Users/dickson/DT/DT_development/backend
[ -f .env ] || cp .env.example .env
python3 scripts/init_db.py
```

## 5. Import Smoke Gates

```bash
cd /Users/dickson/DT/DT_development
PYTHONPATH=/Users/dickson/DT/DT_development:/Users/dickson/DT/DT_development/backend \
python3 -c "import backend.process_data; print('OK backend.process_data')"

PYTHONPATH=/Users/dickson/DT/DT_development:/Users/dickson/DT/DT_development/backend \
python3 -c "import backend.run_daily_analysis; print('OK backend.run_daily_analysis')"

PYTHONPATH=/Users/dickson/DT/DT_development:/Users/dickson/DT/DT_development/backend \
python3 -c "import backend.health_server; print('OK backend.health_server')"
```

## 6. Launch

```bash
cd /Users/dickson/DT/DT_development
./start.sh
```

## 7. Notes

1. `Development/` under repo root is currently untracked on this branch and should not be used as runtime root.
2. If import checks fail with permission errors for logs, ensure `backend/logs` exists and is writable.
