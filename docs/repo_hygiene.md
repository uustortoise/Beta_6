# Beta_6 Repo Hygiene

This repository is standalone. Keep commits source-only and reproducible.

## Required Local Setup

```bash
git config core.hooksPath .githooks
python3 scripts/check_repo_hygiene.py
```

## What Is Blocked

- Secret/local env files:
  - `.env`
  - `backend/.env`
  - `web-ui/.env`
  - `web-ui/.env.local`
- Runtime artifact trees:
  - `logs/`
  - `tmp/`
  - `archive/`
  - `data/`
  - `backend/data/`
  - `backend/tmp/`
  - `backend/logs/`
  - `backend/models/`
  - `backend/models_beta6_registry_v2/`
  - `backend/validation_runs_canary/`
- Local DB/secret key suffixes:
  - `*.db`, `*.db-shm`, `*.db-wal`, `*.pem`, `*.key`

## Required Review Checks

Before pushing:

```bash
git status --short
python3 scripts/check_repo_hygiene.py
```

CI enforces the same check via `.github/workflows/repo-hygiene.yml`.

