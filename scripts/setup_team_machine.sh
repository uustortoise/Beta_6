#!/usr/bin/env bash
set -euo pipefail

REPO_URL_DEFAULT="https://github.com/uustortoise/Beta_6.git"

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_team_machine.sh [target_dir] [branch_name]

Examples:
  scripts/setup_team_machine.sh
  scripts/setup_team_machine.sh ~/work/Beta_6
  scripts/setup_team_machine.sh ~/work/Beta_6 codex/hk001-gate-fix

Behavior:
  1) Clone Beta_6 repo (if target dir does not exist)
  2) Configure local hooks path (.githooks)
  3) Run repository hygiene check
  4) Optionally create/switch to branch_name
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TARGET_DIR="${1:-$HOME/Beta_6}"
BRANCH_NAME="${2:-}"

for cmd in git python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

if [[ -d "$TARGET_DIR/.git" ]]; then
  echo "Repo already exists at $TARGET_DIR"
else
  if [[ -e "$TARGET_DIR" ]]; then
    echo "Target path exists but is not a git repo: $TARGET_DIR" >&2
    exit 1
  fi
  echo "Cloning $REPO_URL_DEFAULT into $TARGET_DIR"
  git clone "$REPO_URL_DEFAULT" "$TARGET_DIR"
fi

cd "$TARGET_DIR"

echo "Setting local hooks path to .githooks"
git config core.hooksPath .githooks

echo "Running repo hygiene check"
python3 scripts/check_repo_hygiene.py

if [[ -n "$BRANCH_NAME" ]]; then
  echo "Switching to branch: $BRANCH_NAME"
  git checkout -B "$BRANCH_NAME"
else
  echo "No branch provided. Staying on $(git branch --show-current)."
fi

echo
echo "Setup complete."
echo "Repo: $TARGET_DIR"
echo "Current branch: $(git branch --show-current)"
echo "Next: copy backend/.env.example to backend/.env locally (do not commit .env)."
