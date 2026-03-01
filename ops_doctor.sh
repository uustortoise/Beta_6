#!/bin/bash
set -u

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$PROJECT_ROOT/backend"
WEB_DIR="$PROJECT_ROOT/web-ui"
MODE="${1:-preflight}" # preflight | running

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

pass() { echo "PASS: $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
warn() { echo "WARN: $1"; WARN_COUNT=$((WARN_COUNT + 1)); }
fail() { echo "FAIL: $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

check_path() {
    local label="$1"
    local path="$2"
    if [ -e "$path" ]; then
        pass "$label exists ($path)"
    else
        fail "$label missing ($path)"
    fi
}

check_cmd() {
    local cmd="$1"
    local label="$2"
    if command -v "$cmd" >/dev/null 2>&1; then
        pass "$label available ($cmd)"
    else
        fail "$label not found ($cmd)"
    fi
}

check_optional_cmd() {
    local cmd="$1"
    local label="$2"
    if command -v "$cmd" >/dev/null 2>&1; then
        pass "$label available ($cmd)"
    else
        warn "$label not found ($cmd)"
    fi
}

echo "----------------------------------------"
echo "Beta_6 Operator Doctor ($MODE)"
echo "----------------------------------------"

check_path "Project root" "$PROJECT_ROOT"
check_path "Backend directory" "$BACKEND_DIR"
check_path "Web UI directory" "$WEB_DIR"
check_path "Start script" "$PROJECT_ROOT/start.sh"
check_path "Stop script" "$PROJECT_ROOT/stop.sh"
check_path "Backend requirements" "$BACKEND_DIR/requirements.txt"
check_path "Backend env example" "$BACKEND_DIR/.env.example"
check_path "Backend run_daily_analysis" "$BACKEND_DIR/run_daily_analysis.py"
check_path "Streamlit app shell" "$BACKEND_DIR/app/main.py"
if [ -f "$BACKEND_DIR/export_dashboard.py" ]; then
    pass "Legacy export dashboard present ($BACKEND_DIR/export_dashboard.py)"
else
    warn "Legacy export dashboard missing ($BACKEND_DIR/export_dashboard.py)"
fi
check_path "Docker compose file" "$BACKEND_DIR/docker-compose.yml"

if [ -f "$BACKEND_DIR/.env" ]; then
    pass "Backend .env present"
else
    warn "Backend .env missing (will be auto-created from .env.example by start.sh)"
fi

if [ -x "$BACKEND_DIR/venv/bin/python" ]; then
    pass "Virtualenv python present ($BACKEND_DIR/venv/bin/python)"
else
    warn "Virtualenv python missing ($BACKEND_DIR/venv/bin/python)"
fi

check_cmd "python3" "System Python"
check_optional_cmd "docker" "Docker"
check_optional_cmd "docker-compose" "Docker Compose"
check_optional_cmd "npm" "Node package manager"
check_optional_cmd "curl" "HTTP probe tool"

if command -v docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then
        pass "Docker daemon is running"
    else
        warn "Docker command found but daemon is not running"
    fi
fi

if [ "$MODE" = "running" ]; then
    if command -v curl >/dev/null 2>&1; then
        if curl -fsS "http://localhost:${WEB_PORT:-3002}/" >/dev/null 2>&1; then
            pass "Web UI is reachable on http://localhost:${WEB_PORT:-3002}/"
        else
            warn "Web UI is not reachable on http://localhost:${WEB_PORT:-3002}/"
        fi
        if curl -fsS "http://localhost:${EXPORT_PORT:-8503}/" >/dev/null 2>&1; then
            pass "Export UI is reachable on http://localhost:${EXPORT_PORT:-8503}/"
        else
            warn "Export UI is not reachable on http://localhost:${EXPORT_PORT:-8503}/"
        fi
    else
        warn "Skipping runtime HTTP probes (curl not installed)"
    fi
fi

echo "----------------------------------------"
echo "Doctor summary: PASS=$PASS_COUNT WARN=$WARN_COUNT FAIL=$FAIL_COUNT"
echo "----------------------------------------"

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
exit 0
