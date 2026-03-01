#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LEGACY_OPS_START="$PROJECT_ROOT/ops_start.sh"

SCAN_UNUSED=1
SCAN_ONLY=0
PASSTHRU_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./start-ops.sh [options]

Operator wrapper for Beta_6 startup.
- Runs a non-destructive "unused artifact" scan (default on)
- Then executes ./ops_start.sh with the same options

Options:
  --no-scan       Skip unused-artifact scan
  --scan-only     Run scan and exit (do not start services)
  -h, --help      Show this help

All other options are forwarded to ./ops_start.sh
Examples:
  ./start-ops.sh
  ./start-ops.sh --skip-pull --skip-smoke
  ./start-ops.sh --scan-only
EOF
}

scan_unused_artifacts() {
    echo "----------------------------------------"
    echo "Beta_6 Unused Artifact Scan (non-destructive)"
    echo "----------------------------------------"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Project:   $PROJECT_ROOT"
    echo ""
    echo "Likely safe to prune (runtime/generated):"

    local safe_paths=(
        ".pytest_cache"
        "web-ui/node_modules"
        "backend/tmp"
        "backend/validation_runs_canary"
        "logs"
        "export_ui.log"
        "automation.log"
        ".DS_Store"
        "backend/export_ui.log"
        "backend/logs/pipeline.log"
        "backend/logs/pipeline.log.1"
        "backend/logs/pipeline.log.2"
    )

    local found_safe=0
    for rel in "${safe_paths[@]}"; do
        local full="$PROJECT_ROOT/$rel"
        if [ -e "$full" ]; then
            found_safe=1
            local size
            size="$(du -sh "$full" 2>/dev/null | awk '{print $1}')"
            [ -z "$size" ] && size="(size unavailable)"
            echo "  - $rel  [$size]"
        fi
    done
    if [ "$found_safe" -eq 0 ]; then
        echo "  - None detected."
    fi

    echo ""
    echo "Review-before-prune (not used by start.sh path, but may be intentional):"
    local review_paths=(
        "health_advisory_chatbot"
        "Vital sign addon"
        "Beta6"
    )
    local found_review=0
    for rel in "${review_paths[@]}"; do
        local full="$PROJECT_ROOT/$rel"
        if [ -e "$full" ]; then
            found_review=1
            local size
            size="$(du -sh "$full" 2>/dev/null | awk '{print $1}')"
            [ -z "$size" ] && size="(size unavailable)"
            echo "  - $rel  [$size]"
        fi
    done
    if [ "$found_review" -eq 0 ]; then
        echo "  - None detected."
    fi

    echo ""
    echo "Note: No files are deleted by this script."
    echo "----------------------------------------"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --no-scan)
            SCAN_UNUSED=0
            ;;
        --scan-only)
            SCAN_ONLY=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            PASSTHRU_ARGS+=("$1")
            ;;
    esac
    shift
done

if [ "$SCAN_UNUSED" -eq 1 ] || [ "$SCAN_ONLY" -eq 1 ]; then
    scan_unused_artifacts
fi

if [ "$SCAN_ONLY" -eq 1 ]; then
    exit 0
fi

if [ ! -x "$LEGACY_OPS_START" ]; then
    echo "Error: missing executable $LEGACY_OPS_START"
    exit 1
fi

if [ "${#PASSTHRU_ARGS[@]}" -gt 0 ]; then
    exec "$LEGACY_OPS_START" "${PASSTHRU_ARGS[@]}"
else
    exec "$LEGACY_OPS_START"
fi
