#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/Users/dicksonng/DT/Development/Beta_5.5}"
DATA_DIR="${DATA_DIR:-/Users/dicksonng/DT/Development/New training files}"
ELDER_ID="${ELDER_ID:-HK0011_jessica}"
PROFILE="${PROFILE:-lr_fragmentation_sweep}"
PROFILES_YAML="${PROFILES_YAML:-$REPO_DIR/backend/config/event_first_matrix_profiles.yaml}"
GO_NO_GO_CONFIG="${GO_NO_GO_CONFIG:-$REPO_DIR/backend/config/event_first_go_no_go.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/beta55_lr_frag_sweep_clean_$(date +%Y%m%d_%H%M%S)}"
MAX_WORKERS="${MAX_WORKERS:-1}"
SEED_TIMEOUT_SECONDS="${SEED_TIMEOUT_SECONDS:-300}"
SEED_RETRIES="${SEED_RETRIES:-1}"
MATRIX_TIMEOUT_SECONDS="${MATRIX_TIMEOUT_SECONDS:-1200}"

cd "$REPO_DIR"

python3 backend/scripts/run_lr_fragmentation_sweep_clean.py \
  --profiles-yaml "$PROFILES_YAML" \
  --profile "$PROFILE" \
  --data-dir "$DATA_DIR" \
  --elder-id "$ELDER_ID" \
  --output-dir "$OUTPUT_DIR" \
  --go-no-go-config "$GO_NO_GO_CONFIG" \
  --max-workers "$MAX_WORKERS" \
  --seed-timeout-seconds "$SEED_TIMEOUT_SECONDS" \
  --seed-retries "$SEED_RETRIES" \
  --matrix-timeout-seconds "$MATRIX_TIMEOUT_SECONDS" \
  --cleanup-resource-trackers

echo "Sweep output: $OUTPUT_DIR/$PROFILE"
echo "Manifest: $OUTPUT_DIR/$PROFILE/clean_sweep_manifest.json"
echo "Ranking:  $OUTPUT_DIR/$PROFILE/ranking.csv"
