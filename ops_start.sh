#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$PROJECT_ROOT/backend"

SKIP_PULL=0
SKIP_INSTALL=0
SKIP_SMOKE=0

usage() {
    cat <<'EOF'
Usage: ./ops_start.sh [options]

Options:
  --skip-pull      Skip git pull step
  --skip-install   Skip python package install step
  --skip-smoke     Skip smoke test step
  -h, --help       Show this help

Notes:
- This script is opt-in and does not change default start.sh behavior.
- It runs preflight checks, then hands off to ./start.sh.
- When running from a feature or integration worktree, pass --skip-pull to avoid switching back to main.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-pull) SKIP_PULL=1 ;;
        --skip-install) SKIP_INSTALL=1 ;;
        --skip-smoke) SKIP_SMOKE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
    shift
done

echo "----------------------------------------"
echo "Beta_6 Operator Start"
echo "----------------------------------------"
echo "Project: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

if [ "$SKIP_PULL" -eq 0 ]; then
    echo "[1/6] Pulling latest code..."
    CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo DETACHED)"
    if [ "$CURRENT_BRANCH" != "main" ]; then
        if ! git diff --quiet || ! git diff --cached --quiet; then
            echo "ERROR: Current branch is '$CURRENT_BRANCH' with uncommitted changes."
            echo "Operator start requires syncing from 'main'."
            echo "Please commit/stash your changes, then run:"
            echo "  git checkout main"
            echo "  git branch --set-upstream-to=origin/main main"
            echo "  ./ops_start.sh"
            exit 1
        fi
        echo "Switching branch '$CURRENT_BRANCH' -> 'main' for operator run..."
        git checkout main
    fi
    git fetch --prune origin main
    git branch --set-upstream-to=origin/main main >/dev/null 2>&1 || true
    git pull --ff-only origin main
else
    echo "[1/6] Pull step skipped."
fi

echo "[2/6] Ensuring backend env file exists..."
if [ ! -f "$BACKEND_DIR/.env" ] && [ -f "$BACKEND_DIR/.env.example" ]; then
    cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
    echo "Created $BACKEND_DIR/.env from .env.example"
fi

if [ -f "$BACKEND_DIR/.env" ]; then
    set -a
    source "$BACKEND_DIR/.env"
    set +a
fi
ENABLE_BETA6_AUTHORITY="${ENABLE_BETA6_AUTHORITY:-true}"
if [ "$ENABLE_BETA6_AUTHORITY" = "true" ]; then
    if [ -z "${BETA6_GATE_SIGNING_KEY:-}" ]; then
        echo "ERROR: Beta 6 live authority requires explicit BETA6_GATE_SIGNING_KEY in backend/.env"
        exit 1
    fi
    if [ -z "${RELEASE_GATE_EVIDENCE_PROFILE:-}" ]; then
        echo "ERROR: Beta 6 live authority requires explicit RELEASE_GATE_EVIDENCE_PROFILE in backend/.env"
        exit 1
    fi
fi

echo "[3/6] Running preflight doctor..."
"$PROJECT_ROOT/ops_doctor.sh" preflight

PYTHON_CMD="python3"
if [ -x "$BACKEND_DIR/venv/bin/python" ]; then
    PYTHON_CMD="$BACKEND_DIR/venv/bin/python"
fi
echo "Using Python: $PYTHON_CMD"

if [ "$SKIP_INSTALL" -eq 0 ]; then
    echo "[4/6] Installing backend Python dependencies..."
    "$PYTHON_CMD" -m pip install -r "$BACKEND_DIR/requirements.txt"
else
    echo "[4/6] Install step skipped."
fi

if [ "$SKIP_SMOKE" -eq 0 ]; then
    echo "[5/6] Running smoke test..."
    cd "$PROJECT_ROOT"
    SMOKE_TESTS=(
        "backend/tests/test_run_daily_analysis_beta6_authority.py"
        "backend/tests/test_health_server.py"
        "backend/tests/test_t80_rollout.py"
        "backend/tests/test_prediction_beta6_runtime_hook_parity.py"
        "backend/tests/test_beta6_orchestrator.py"
    )
    EXISTING_SMOKE_TESTS=()
    for test_file in "${SMOKE_TESTS[@]}"; do
        if [ -f "$test_file" ]; then
            EXISTING_SMOKE_TESTS+=("$test_file")
        fi
    done

    if [ "${#EXISTING_SMOKE_TESTS[@]}" -eq 0 ]; then
        echo "No curated Beta6 smoke tests found under $BACKEND_DIR/tests"
        exit 1
    fi

    "$PYTHON_CMD" -m pytest -q "${EXISTING_SMOKE_TESTS[@]}"
else
    echo "[5/6] Smoke test skipped."
fi

echo "[6/6] Starting platform via existing start.sh..."
exec "$PROJECT_ROOT/start.sh"
