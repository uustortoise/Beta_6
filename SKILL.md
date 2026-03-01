# Beta 5.5 Engineering Skill: Production-Ready, Audit-Ready Changes

This repository is approaching production. When making changes, assume they may impact PHI, clinical safety, and long-term maintainability.

## Non-Negotiables

- **No silent fallbacks** for critical infrastructure (e.g., database) in production paths.
- **No time-series leakage** in ML training/validation (no shuffle, no future→past fill, no non-temporal validation splits).
- **Schema-first discipline:** update `schema.sql` and migrations before app code depends on it.
- **Observable by default:** errors are logged with context; avoid `except: pass` and “print debugging”.
- **Small, reviewable diffs:** avoid mixed refactors + feature work; isolate risk.

## Release Gates (Must Pass Before Shipping)

- **Security:** authentication + authorization + audit logging for any PHI access/mutation.
- **Clinical safety:** emergency-symptom triage in any health-advice flow; no prescribing/dosing instructions in consumer mode.
- **DB mode consistency:** `POSTGRES_ONLY` and `USE_POSTGRESQL` are consistent across services and enforced at startup.
- **ML credibility:**
  - Temporal holdout evaluation (train on past, validate on future).
  - Training/inference window alignment per room (same `seq_length`).
  - Causal preprocessing (no `bfill` across real gaps; interpolation is non-causal unless explicitly allowed in offline analytics).

## ML Rules of Thumb (ADL Learning)

- Treat sensor streams as **time series**: any validation must respect chronology.
- Any resampling must be **causal** for production inference. If leading NaNs exist, fill only the prefix.
- If corrections (“goldens”) are merged, ensure they do not break temporal splits (sort sequences by timestamp before splitting).
- **Cross-Room Isolation**: Scope retraining and historical reprediction to corrected rooms ONLY. Retraining unchanged rooms introduces non-deterministic noise (different weights/seeds) that corrupts verified historical data.
- **Identity Binding**: Always derive `elder_id` from the correction source (queue/file) rather than live UI state during writes to prevent identity collision.
- Prefer **room-specific configuration** for window and interval; avoid hardcoded 10s assumptions.
- Keep **policy-key normalization** centralized and identical wherever release gates are resolved (no ad-hoc `lower().replace(...)` variants).
- For **scheduled thresholds**, treat out-of-range early windows as "use earliest bracket", not strictest fallback.
- For low-volume bootstrap data, avoid "missing metric => automatic fail" when validation split is intentionally disabled.
- Registry cleanup must never remove the active rollback target; preserve `current_version` artifacts/metadata and heal stale pointers.
- **True Pre-Promotion Gating**: All validation gates (Walk-forward, Global, Integrity) must pass against *candidate artifacts* BEFORE switching the champion pointer. Do not use "promote-then-rollback" patterns.
- **Beta 6 Architecture (Backbone/Adapter)**: Shared backbone weights must be treated as immutable "Core" assets. Custom adapter weights/scalers must be versioned alongside the backbone identity they were trained against.
- **Canary & Migration**: Use a "Canary List" for controlled rollout of the new loading mode. Measure "Outcome Deltas" (manual corrections vs total windows) to prove quality parity before full migration.
- **Data Quality Gates**: Enforce hard checks for label coverage and timestamp gaps *pre-training*. Rejection in the quality gate is a success for system reliability.
- **Lineage Integrity**: Every model run must persist the exact file hashes, code version, and feature config used to produce the artifacts.

## ML Troubleshooting & Interpretation (Feb 15, 2026)

### 1. The "Null Model" Paradox (Accuracy=1.0, F1=0.5)
- **Symptom:** Your model has perfect accuracy on the holdout set, but the Macro-F1 score is exactly 0.5.
- **Cause:** The holdout set contains **only negative/background samples** (e.g., empty room), and the model predicts "Background" for everything.
- **Meaning:** The model is "Lazy" but correct. It hasn't learned the target activity (e.g., Fall), it just knows "most things are nothing".
- **Fix:** You need **positive support**. Add at least 5 examples of the target activity to the validation window.

### 2. Smart Downsampling (Do NOT Cut Files)
- **Rule:** Never manually split or truncate 24h files. The pipeline handles thinning automatically.
- **Mechanism:** `UNOCCUPIED_DOWNSAMPLE_STRIDE=4` (keeps 1 sample every 40s) + `BOUNDARY_KEEP=6` (preserves transitions).
- **Why:** Manual cutting destroys the context required for "Smart Downsampling" to work. Trust the code.

### 3. Walk-Forward Feasibility Math
- **Formula:** `Required Days = Min Train Days + Valid Days`.
- **Example:** If `Min Train = 3` and `Valid = 1`, you need **4 full days** of data.
- **Error:** If you have 3 days, you get `Expected Folds: 0`. The system cannot invent the 4th day (the "Exam").

### 4. Auto-Tuner & Environment Persistence
- **Config Strategy:** Do not hardcode tuning values. Use `_env_int` / `_env_float` and persist changes to `.env`.
- **Reason:** Allows the Auto-Tuner to modify system behavior without code changes, enabling "Self-Healing" via operator button clicks.

## Review Checklist (Before You Say “Done”)

- Does the change introduce a new default password, token, or secret in code or config?
- Does the change affect timestamps, resampling, sequence windows, or validation splits?
- Are tests updated/added for the regression you fixed?
- Are the production and dev run paths explicit and consistent?
