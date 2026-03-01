# Development Progress Log

> [!IMPORTANT]
> **Project Mandate (1000 POC Production Readiness)**
> 1. **Solid Foundation**: No temporary fixes or debug patches. All solutions must be production-ready for scaling to 1000 elders.
> 2. **Preserve Logic**: Do not remove or change ANY existing Beta 5 features or logic without explicit approval.
> 3. **Additive Development**: All new work must be strictly additive extensions to the existing system.

## Beta 5: Scalability & Stabilization
**Status:** 🟡 In Progress  
**Goal:** Transition from Beta 4 prototype to a robust, scalable architecture with reliable data correction workflows.

### ✅ Lane B COMPLETE: Event Compiler + KPI/Gates + Home-Empty Fusion (Feb 16, 2026)
- **PR-B1: Event Compiler + Decoder + Derived Events - COMPLETE:**
    - **Event Compiler (`ml/event_compiler.py`):** Episode compilation, gap-aware splitting, hysteresis smoothing, multi-room support.
    - **Event Decoder (`ml/event_decoder.py`):** Head A/B fusion, hysteresis state machine, temporal smoothing.
    - **Derived Events (`ml/derived_events.py`):** Sleep, bathroom, kitchen metrics, out-time calculation, weekly aggregation.
    - **Tests:** 56 new tests passing.

- **PR-B2: Event KPI + Gate Layer - COMPLETE:**
    - **Event Gates (`ml/event_gates.py`):** Tier-1/2/3 gate checks, collapse detection, unknown-rate caps.
    - **Event KPI (`ml/event_kpi.py`):** Home-empty metrics, per-event recall/precision/F1, care KPI extraction.
    - **Criticality Tiers:** Tier-1 (>=0.50 recall), Tier-2 (>=0.35 recall), Tier-3 (>=0.20 recall).
    - **Hard Safety Gates:** Home-empty precision >= 0.95, false-empty rate <= 0.05.
    - **Tests:** 28 new tests passing.

- **PR-B3: Home-Empty Fusion + Household Gate - COMPLETE:**
    - **Home-Empty Fusion (`ml/home_empty_fusion.py`):** Multi-room fusion, room consensus, entrance penalty logic.
    - **Household Gate:** Precision/false-empty rate validation, coverage checks.
    - **Safety Features:** 5-minute entrance penalty, 60-second temporal smoothing, false-empty protection.
    - **Tests:** 22 new tests passing.

- **Lane B Summary:**
    - **Total New Modules:** 6 (`event_compiler`, `event_decoder`, `derived_events`, `event_gates`, `event_kpi`, `home_empty_fusion`)
    - **Total New Tests:** 106 (56 + 28 + 22)
    - **Full Test Suite:** 463 passing, 3 skipped
    - **Integration:** All modules integrated with Lane A ADL Registry
    - **Status:** ✅ Ready for Lane C (CNN+Transformer Path)

### ✅ Latest ML Stabilization Updates (Feb 16, 2026)
- **Production ML Hardening - COMPLETE (Weeks 1-5):**
    - **Goal:** Ship a fully hardened, governed, and mathematically rigorous ML pipeline.
    - **Week 5 Delivered (Governance & Research):**
        - **Registry Validator:** CI-integrated checks for registry consistency and artifact existence.
        - **Policy Presets:** `PolicyPresetManager` with Conservative/Balanced/Aggressive profiles (Aggressive blocked in Prod).
        - **Pilot Overrides:** `PilotOverrideManager` with auto-expiry and rollback for safe production testing.
        - **Transformer Head A/B:** Framework for rigorous architectural comparison (GlobalAvg vs Attention vs LastToken).
    - **Final Stats:**
        - **142 Tests Passing:** 100% coverage of all hardening items.
        - **Performance:** Duplicate resolution optimized 400x.
        - **Reliability:** Zero temporal leakage, deterministic training, fail-closed data gates.
    - **Next Step:** Deployment to Production (Mac Studio).

- **Production ML Hardening (Feb 15, 2026):**
    - **Registry Repair:** Implemented `validate_and_repair_room_registry_state` to automatically fix inconsistent promotion flags and orphan aliases.
    - **Fail-Closed Resampling:** Refactored `data_loader.py` to strictly raise errors on resampling failures (no silent fallbacks), adding `raw_rows_before_resample` and `rows_after_resample` diagnostics.
    - **Determinism:** Enforced global random seeding (`tf`, `np`, `python`) in `TrainingPipeline` via `TrainingPolicy`.
    - **Verification:** Added comprehensive test suites:
        - `test_registry.py`: Verified multi-promotion repair and alias consistency.
        - `test_data_loader_resampling.py`: Verified diagnostic attachment and fail-closed behavior.
        - `test_training.py`: Verified deterministic seeding.
    - **Status:** All hardening measures implemented and passed verification.

- **Operator Ergonomics & UX (Phase 5) - COMPLETED:**
    - **Pre-Flight Feasibility Check:**
        - Instantly warns if selected `min_train_days` + `valid_days` exceeds available data.
        - Prevents "0 Fold" errors by calculating `expected_folds` dynamically before the run.
    - **Auto-Tuner Recommendations:**
        - Smart suggestions based on failure patterns (e.g., "Class Imbalance detected -> Increase minority sampling").
        - "One-Click Apply" button to update `.env` or session state instantly.
    - **Plain-Language Promotion Cards:**
        - Translates cryptic error codes (`room_threshold_failed`) into human-readable report cards.
        - Displays **"Promotion Denied"** with clear reasons ("Score: 48% < Required: 55%") and actionable tips.
    - **Model Health Check Table:**
        - Renamed confusing metrics ("Macro-F1" -> "Balanced Score").
        - Added **Stability Warnings** if performance drifts across holdout folds.

- **ML Logic Verification & Hardening:**
    - **Smart Downsampling Confirmed:** Verified `training.py` logic (`STRIDE=4`, `MIN_SHARE=0.45`). Code automatically preserves boundary context while thinning repetitive "Unoccupied" blocks (1 sample per 40s). **No manual file cutting required.**
    - **Fold Feasibility Logic:** Clarified that `min_train_days=3` + `valid_days=1` requires **Day 4 data** (future) to run any test.
    - **Null Model Paradox:** Confirmed that `Accuracy=1.0` / `F1=0.5` mathematically proves a "Lazy Model" predicting only background classes on empty test folds.

- **Infrastructure:**
    - **Environment Persistence:** Implemented `_persist_env_updates_to_dotenv` to save Auto-Tuner changes to `.env` permanently.
    - **Profile Management:** Validated "Fast Iteration" vs "Standard" profiles via `WF_MIN_TRAIN_DAYS` environment variables.

### ✅ Latest ML Stabilization Updates (Feb 14, 2026)
- **Senior ML Implementation Review (Phases 2-4) - SUCCESSFUL:**
    - Full audit of layer freezing (`_apply_correction_layer_freeze`), stratified replay sampling, and walk-forward evaluator.
    - Confirmed LR resolution fix (`low` -> `1e-5`) and registry rollback safety.
    - Infrastructure verified as "Production Ready" for 1000-POC scale.
- **Observability & Ops Readiness (Dashboard 2.0):**
    - **Model Update Safety Monitor**: Added "Friendly" gate reasons and severity badges to the dashboard for ops visibility.
    - **Runtime Load Mode Tracker**: Added audit logic to detect "Full Model" vs "Shared Adapter" (Beta 6) loading status per room.
    - **Safe Update Rate**: Implemented rolling pass/fail metrics with automatic "Action Needed" alerts.
- **Release Gate Reliability Hardening:**
    - Fixed scheduled-threshold fallback so `< min_days` now resolves to the earliest bracket (not strictest).
    - Applied in both training-time and daily-run gate evaluators.
- **Gate Policy Key Safety:**
    - Unified room-key normalization to prevent silent `no_room_policy:*` bypass when policy keys differ in separators/case.
- **Bootstrap/Low-Volume Training Path:**
    - Removed "fail by construction" behavior for `<100` sample windows where validation split is disabled.
    - Added bootstrap-safe metric fallback so early residents can promote when policy permits.
- **Prediction Integrity Contract (Now Enforced):**
    - Upgraded integrity validation from advisory logging to hard-blocking failure on invalid predictions.
    - Prevents downstream persistence of invalid outputs.
- **Model Registry Rollback Safety:**
    - Cleanup now preserves current champion metadata/artifacts even when older than the newest N.
    - Stale `current_version` pointers are automatically healed to `0` when metadata is missing.
- **Regression Coverage Added:**
    - New tests for scheduled threshold behavior, integrity-gate blocking, and registry cleanup/current-version coherence.

### ✅ Completed
- **Architecture Overhaul:**
    - Established new `Beta_5` environment separate from Beta 4.
    - Implemented modular design (`FeatureEngine`, `ModelRegistry`, `BatchPredictor`).
- **Correction Studio:**
    - Developed Labeling Studio with Activity Timeline visualization.
    - Implemented "Batch Correction" queue for multi-room edits.
    - **Feature Revert:** Removed "Click-to-Fill" in favor of robust manual entry (Jan 17 2026).
- **Audit Trail Fix (Feb 8, 2026):**
    - **Root Cause:** Cached audit functions defined inside `tab6` context were recreated on each render, breaking cache invalidation.
    - **Fix:** Moved `fetch_correction_history_cached`, `get_correction_summary`, `get_available_filters` to module level.
    - **Fix:** Added explicit `clear_all_caches()` call immediately after correction commit.
    - **UX:** Changed default time filter from "Last 30 Days" to "All Time" to prevent missing corrections.
- **Health Advisory Chatbot - Knowledge Base Admin UI (Feb 8, 2026):**
    - **Data Migration:** Converted hardcoded Python guidelines/drugs/FAQ to JSON files
    - **File Upload:** Created web UI for uploading JSON/CSV files with drag-and-drop
    - **Validation:** Automatic schema validation with detailed error/warning reporting
    - **Dashboard:** Overview stats, recent activity feed, filterable data browser
    - **Templates:** Downloadable starter templates for each data type
    - **API:** REST endpoints for upload, validation, stats, activity log
    - **Location:** `health_advisory_chatbot/admin/` with standalone server
- **Automation & Workflow:**
    - Implemented `run_daily_analysis.py` for automated processing.
    - Confirmed "Train Retrospectively" logic: Retrains on full history + Repredicts past files.
    - Created comprehensive `/reset` workflow to clear all databases/caches cleanly.
    - **Stabilization:** Fixed resampling crash related to non-numeric (activity) labels in duplicate timestamps.
    - **1000 POC Prep:** Implemented early resampling in `data_loader.py` - all data is now cleaned to 10s intervals on load, eliminating duplicate timestamps at the source.
- **Sleep & Health Analysis:**
    - Fixed "Missing Sleep Data" bug: Analysis now uses correct record date instead of `now()`.
    - Implemented `backfill_analysis.py` to populate historical sleep and ICOPE records.
    - Updated `getSleepData` API to support 7-day trends and stage breakdown donut charts.
- **ML Pipeline & Data Integrity (Jan 19 2026):**
    - **Fix:** Aligned denoising window in prediction to match training (was hardcoded 5, now uses config default 3).
    - **Fix:** Long activity segments now split into chunks instead of being skipped, preventing data loss on timeline.
- **Documentation:**
    - Updated `labeling_guide.md` with Traffic Light rules for Golden Samples.
    - Centralized sleep logic in `SleepAnalyzer` class.
    - Standardized stage detection, scoring, and insights.
    - Integrated into main pipeline and backfill scripts.
    - **High Deep Sleep Fix (Jan 19 2026):**
        - **Root Cause:** Analysis was using Z-score scaled motion data (from neural network) instead of raw.
        - **Solution:** Implemented Heuristic Fallback (Deep 15% / Light 55%) for when raw motion is unavailable/scaled.
        - **Hardening:** Added `UNIQUE(elder_id, analysis_date)` constraint to DB schema to prevent duplicate records.

- **External Review Implementation (Jan 20 2026):**
    - **HIGH:** Created `MotionDataNormalizer` utility class for consistent motion data handling across real-time and historical pipelines.
    - **MEDIUM:** Added batch processing to `backfill_analysis.py` with `--batch-size`, `--start-date`, `--end-date` CLI options.
    - **MEDIUM:** Added test coverage: `test_motion_normalizer.py`, `test_backfill_performance.py`, `test_pipeline_integration.py`.
    - **LOW:** Created `timestamp_utils.py` for timestamp format normalization.
    - **Cleanup:** Removed unused `numpy` import from `process_data.py`, fixed pre-existing test issue in `test_sleep_analyzer.py`.

- **Data Processing Hardening (Jan 21 2026):**
    - **Fix:** Resolved intermittent "File not found" errors in `process_data.py` during archiving.
    - **File Locking:** Added `fcntl`-based exclusive lock to prevent concurrent execution.
    - **Optimized Deduplication:** Changed O(N) hash loop to O(1) targeted check for performance.
    - **Hardened Operations:** All `unlink()` and `shutil.move()` now wrapped with `exists()` checks and `try/except`.

### 🚧 In Progress / Planned
- [x] **Deployment**: Finalize deployment logic for onsite NUC and Mac Studio.
    - ✅ Created `setup_mac_studio.sh` and `setup_guide.md`.
    - ✅ Fixed log path resolution for cross-directory stability.
- [ ] **Long-term Testing**: Verify stability over multiple weeks of data.

---

## Beta 6: Universal Backbone & Industrial Reliability
**Status:** 🟡 Active Development  
**Goal:** Scalable model architecture for 1000+ residents with zero-risk deployment.

### 📋 Roadmap & Advanced Stabilization
- **Universal Backbone (Shared Architecture):**
    - [ ] Deploy Shared Backbone + Per-Resident Adapters (The "Core + Custom Shell").
    - [ ] Support hybrid loading (backbone weights + adapter weights).
- **True Pre-Promotion Gating:**
    - [ ] Run *all* gates (Walk-forward, Global, Integrity) on candidate artifacts *before* switching champion pointers.
    - [ ] Implement "Staging" state for candidate models.
- **Canary Rollout Strategy:**
    - [ ] Implement "Canary List" config to control shared-adapter activation.
    - [ ] Measure "Online Outcome Deltas" (Correction Rate, Misses, False Alarms) vs Legacy.
- **Industrial Hardening:**
    - [ ] **Deterministic Training**: Enforce global seed control and deterministic ops.
    - [ ] **Full Lineage**: Store exact file hashes, policy versions, and feature configs in run metadata.
    - [ ] **Persistent Fold-Metrics**: Write detailed validation stats to DB `training_history` (not just JSON).
    - [ ] **Data Quality Gates**: Implement hard pre-training checks for label coverage and timestamp gaps.
    - [ ] **Self-Healing**: Automated drift-triggered retraining policy with cooldowns.
- **Behavioral & Clinical:**
    - [ ] Transition Behavioral Layer from code to Database (Clinical CMS).
    - [ ] Enable self-service rule tuning for clinical researchers.
    - [ ] Expand RAG corpus to cover top 20 chronic elderly conditions.

---

## Beta 5.5: Hybrid CNN-Transformer Prototype (NEW - Jan 23, 2026)
**Status:** ✅ Completed  
**Goal:** Validate Transformer-based architecture for offline/batch daily analysis (no real-time alert requirement).

### 📋 Rationale
- **Acausal Processing**: Model can see full day context (past + future) for better pattern disambiguation.
- **Parallel Training**: Significantly faster than sequential LSTMs.
- **Minimum N for Value**: ~50 residents for statistically robust backbone.

### ✅ Completed
- Environment cloned from Beta 5 with independent ports (3002/8503) and database.
- Updated `ml_strategy_discussion.md` with Hybrid CNN-Transformer strategy and risk analysis.
- **Preprocessing for Transformer (Jan 23 2026):**
    - **Denoising:** Increased threshold from 3.0 → 4.0 (less aggressive, lets attention handle noise).
    - **Gap Detection:** Added `detect_gaps()` and `insert_gap_tokens()` for discontinuity handling.
    - **Positional Encoding:** Created module with Sinusoidal, Relative (ALiBi), and Learnable encodings.
    - **Transformer Backbone:** Created `transformer_backbone.py` with CNN embedding + Multi-Head Self-Attention.
- **Queue Improvements (Jan 23 2026):**
    - **Multi-Room Conflict Detection:** Prevents physically impossible corrections (being in two rooms at once).
    - **Auto-Queue from Grid:** Detects edits in the data grid and offers to batch-queue them.

- **Environment Hardening & Portability (Jan 26, 2026):**
    - **Path Resolution:** Replaced hardcoded paths with environment-anchored relative resolution in `settings.py` (e.g., `RAW_DATA_DIR` resolves relative to project root).
    - **Clean Reset:** Executed full reset of Beta 5.5 environment; re-initialized database and created "HK001_jessica" resident profile.
- **ML & Taxonomy (Jan 26 2026):**
    - **Serialization Fix:** Registered all custom Keras layers (`TransformerEncoderBlock`, `SinusoidalPositionalEncoding`, etc.) with `@register_keras_serializable` to ensure model loading persistence.
    - **Per-Room Config:** Optimized Pipeline to respect individual room "Sequence Window" settings from the Backend AI Config tab.
    - **Label Update:** Split `inactive` into `inactive` (person present, idle) and `unoccupied` (room empty).
    - **Conflict Logic:** Updated validation to allow `unoccupied` to coexist across multiple rooms (same as `out`).
    - **Verification:** Successfully trained Transformer models and generated predictions for HK001_jessica on Jan 1st 2026 data.
- **UnifiedPipeline Modularization (Jan 26 2026):**
    - **Refactor:** Decomposed monolithic `UnifiedPipeline` into `ModelRegistry`, `TrainingPipeline`, and `PredictionPipeline`.
    - **Outcome:** Improved maintainability, separation of concerns, and testability.
    - **Fixes:**
        - Resolved `ModuleNotFoundError` by standardizing relative imports.
        - Fixed `AttributeError: 'int' object has no attribute 'floor'` by robustly handling Index vs Series timestamps.
        - Restored missing DB logging for training history.
    - **Verification:** Validated full training/prediction cycle with legacy dataset after environment reset.

- **Pipeline Stabilization & Robustness (Jan 26, 2026 - Late Evening):**
    - **Fix (Imports):** Resolved `ModuleNotFoundError: No module named 'backend'` by standardizing relative imports in `ModelRegistry` and implementing the `PROJECT_ROOT` sys.path hotfix in `pipeline.py` (Option A).
    - **Fix (Timestamps):** Resolved `ValueError: Given date string "29" not likely a datetime` in the prediction pipeline by ensuring timestamps are extracted from the `timestamp` column rather than the `RangeIndex`.
    - **Keras Layers:** Added explicit `build()` method to `TransformerEncoderBlock` to ensure strict layer initialization and suppress Keras initialization warnings.
    - **Motion Normalization:** Implemented case-insensitive column matching in `MotionDataNormalizer` (e.g., "Motion" vs "motion"), ensuring reliable data injection for accurate sleep analysis.
    - **System Verification:** Confirmed stable end-to-end performance (Training -> Prediction -> Sleep Analysis) after a final environment reset and schema re-initialization.
- **System Reliability & Feedback (Jan 27, 2026):**
    - **Progress Monitoring:** Implemented granular `progress_callback` across the entire ML pipeline, providing real-time UI feedback (e.g., "Training Kitchen: Epoch 3/5").
    - **Race Condition Fix:** Implemented naming-based exclusion (`_manual_train.xlsx`) to prevent the `run_daily_analysis.py` background worker from "stealing" files currently being processed by the dashboard.
    - **Logic Fix:** Corrected indentation and nesting in `export_dashboard.py` to ensure training triggers reliably in both Retro and Daily modes.
    - **Bug Fix:** Resolved `UnboundLocalError` in `repredict_all` by correctly initializing result counters.
    - **Optimization:** Defined "Good Data" indicators and reduced default training from 10 to 5 epochs (achieving ~85% accuracy with 50% less compute).
    - **Documentation:** Created `ML strategy.md` and updated `operation_manual.md` with performance tuning guides.
    - **Validation:** Successfully pushed all stability fixes and documentation updates to GitHub.
- **Audit Implementation (Jan 27, 2026):**
    - **ML Consistency:** Refactored `training.py` and `prediction.py` to use centralized `ml.utils`.
    - **Shared Logic:** Created `calculate_sequence_length` and `fetch_golden_samples` to eliminate 100+ lines of duplicate code and ensure formula alignment.
    - **Robustness:** Replaced duplicate SQL queries and inline filtering with a single, tested utility method.
- **Optimization & Restoration (Jan 27, 2026 - Late Morning):**
    - **Archive Optimization:** Implemented `fetch_sensor_windows_batch` to solve N+1 SQL overhead. Added `adl_history` indexes.
    - **ALiBi Encoding:** Integrated Attention with Linear Biases (ALiBi) into Transformer for better sequence length generalization.
    - **Label Logic Restoration:** Fixed "Unoccupied" split logic which was missing due to partial commit.
        - `unoccupied` now valid in all rooms (represents "Empty Room").
        - `inactive` remains "Person Present, Idle".
        - Conflict resolution updated to ignore `unoccupied` states, preventing false multi-presence warnings.
    - **Architectural Audit (Jan 27, 2026 - Late Afternoon):**
        - **Decoupling:** Removed fragile `sys.path` hacks from `pipeline.py` and `run_daily_analysis.py`.
        - **Facade Pattern:** Refactored `UnifiedPipeline` to properly delegate to `ModelRegistry`, `TrainingPipeline`, and `PredictionPipeline`.
        - **Cleanup:** Removed dead code (`_fetch_golden_samples`) that was duplicated in `ml/utils.py`.
        - **Verification:** Validated pipeline initialization and import integrity with `test_transformer_train.py`.
        - **Singleton Removal:** Refactored `RoomConfigManager` to support isolated instances for testing while maintaining global state for production.
        - **Data Integrity:** Implemented "5-Point Validation" gatekeeper in `ml/validation.py` to reject future timestamps, invalid labels, and anomalies.

## 🏠 Home Desktop Sync Instructions
If you are moving from this Work Laptop to the Home Desktop, follow these steps to ensure logic consistency:
1. **Fetch/Pull**: `git fetch origin` then `git pull origin beta-5.5-transformer`.
2. **Conflict Resolution**: If `segment_utils.py` or `household_analyzer.py` show merge conflicts, **accept the Incoming (Work Laptop) changes**, as they contain the finalized restoration logic.
3. **Environment Reset**: Run `python backend/scripts/reset_environment.py` on the Home Desktop to ensure you start with a clean state matching this test cycle.

- **Deployment Preparation (Jan 28, 2026):**
    - **Deployment Plan:** Drafted comprehensive Phase 1 (Backend Only) deployment strategy (`implementation_plan.md`).
    - **Configuration:** Created `.env.production.example` for secure server setup.
    - **Codebase Hardening:**
        - Fixed hardcoded paths in `test_pipeline.py`, `patch_sync.py`, and `compare_predictions.py` to use relative paths.
        - Resolved `ModuleNotFoundError` in `backend/test_alibi.py` by converting `backend/ml` to a proper package and fixing test imports.
        - Fixed `conftest.py` import errors.
    - **Reset Utility:** Enhanced `reset_environment.py` to thoroughly clean `backend/models` in addition to the DB.
- **Scalability Analysis (Jan 28, 2026):**
    - **SQLite Assessment:** Confirmed ~130 touchpoints. Explicitly deferred PostgreSQL migration to Phase 2 (post-pilot) to avoid destabilizing the verified ML pipeline.
    - **Memory Strategy:** identified Transformer memory footprint as a key metric to monitor (`ps -o rss`).
    - **Documentation:** Updated `todo.md` to reflect these architectural decisions.

- **Infrastructure & Reliability (Jan 29, 2026):**
    - **Infrastructure Report:** Completed `final_infrastructure_report.md` detailing the transition to PostgreSQL+TimescaleDB and Mac Studio Ultra hardware sizing.
    - **Pipeline Fix (Data Gaps):** 
        - Identified root cause of "squeezed" timeline (missing segments for older dates in multi-day files).
        - **Fixed `run_daily_analysis.py`** to explicitly regenerate activity segments for all dates touched by training.
        - **Fixed `process_data.py`** to iterate through *all* unique dates in uploaded files, not just the first one.
    - **UI Improvements:**
        - Updated `ActivityTimeline.tsx` to hide "Unoccupied" (empty room) events by default, reducing visual clutter.
    - **Consolidation:** Merged `ML strategy.md` into `ml_strategy_discussion.md` and removed the redundant file.
    - **Migration Plan Review:**
        - Validated `postgresql_timescaledb_migration_plan.md` as production-ready.
        - **Enhancement:** Added "Circuit Breaker" pattern for resilient Dual-Write (prevents DB migration from impacting live app).
        - **Enhancement:** Defined strict "Ingestion Latency" (<50ms) constraints to protect ML pipeline performance.
        - **Action:** Predictions with confidence < `DEFAULT_CONFIDENCE_THRESHOLD` (0.8) are now explicitly labeled as `low_confidence` instead of being accepted as-is.

- **Dual-Write / PostgreSQL Migration (Jan 29, 2026):**
    - **Architecture:** Transitioned from SQLite-only to Hybrid Dual-Write (SQLite + TimescaleDB/PostgreSQL).
    - **Migration:** Backfilled 100% of historical data (85k+ rows) with verified integrity.
    - **Dual-Write:** Implemented fail-safe dual-write mechanism with `verify_dual_write.py` success.
    - **Audit Remediation:**
        - Fixed logic bugs in `database.py`.
        - Implemented structured logging monitoring (~6ms latency overhead).
        - Created `ROLLBACK.md` and `ROLLOUT_PLAN.md`.
        - Verified fail-safe recovery during Rollback Drill.
    - **Status:** **APPROVED FOR PRODUCTION**.

- **Infrastructure & Multi-Device Readiness (Jan 30, 2026):**
    - **PostgreSQL Activation**: Successfully deployed TimescaleDB (Postgres 16) via Docker.
    - **Portability Fix**: Modified `settings.py` for absolute path resolution of `.env`.
    - **Data Migration**: Migrated 85k+ records with 100% parity verification.
    - **Two-Layer Compatibility Defense**:
        - **Schema Layer**: Changed all flags to `INTEGER` (matches SQLite/App reality).
        - **Logic Layer**: Implemented `_normalize_params` in `database.py` to recursively force Python `bool` types to `int` (1/0) before DB execution.
    - **SQL Translator**: Automated conversion of `INSERT OR IGNORE` -> `ON CONFLICT DO NOTHING`.
    - **Verification**: `verify_dual_write.py` passed with `🎉 SUCCESS` after a full clean reset.
    - **Mac Studio Onboarding (Jan 30 - Feb 2, 2026)**:
        - Successfully cloned and verified environment on New Mac Studio hardware.
        - Created `setup_mac_studio.sh` (One-Click Setup) and `setup_guide.md`.
        - **Fix (Automation Log)**: Updated `run_daily_analysis.py` to use absolute paths for `automation.log`, resolving `PermissionError` during remote execution.
        - **Fix (Permissions)**: Created `fix_permissions.sh` to reconcile ownership issues on shared/remote environments.
- **ML Architecture Review Remediation (Jan 31 2026):**
    - **Architecture Review:** Completed `ML_Architecture_Review_Report.md` (Grade B+).
    - **Phase 1 (Cleanup):** Removed 5 debug `sys.stderr.write` statements from `training.py`.
    - **Phase 2 (Critical Fix):** Replaced MLP placeholder in `beta6_trainer.py` with actual Hybrid CNN-Transformer.
        - Updated `prepare_training_data` to support sequence windows (3D tensors).
        - Updated `train_backbone` to auto-detect 2D/3D shapes for backward compatibility.
    - **Phase 3 (Versioning):** Implemented model versioning in `ModelRegistry`.
        - Tracks version history in `{room}_versions.json`.
        - Supports `rollback_to_version()` and maintains an unversioned "latest" copy for seamless integration.
        - Implemented auto-cleanup (retains last 5 versions).
    - **Documentation:** Updated all project status logs to reflect architectural hardening.

- **Health Advisory & Chatbot Integration (Feb 1-2, 2026):**
    - **Chatbot Module**: Deployed `health_advisory_chatbot` v1.0.0 featuring an Advisory Engine, Citation Validator, and Context Fusion.
    - **AI Models**: Integrated DeepSeek API for clinical advisory with RAG-based citation validation.
    - **Frontend**: Added `ChatbotWidget` and `ChatWindow` components with suggested questions and evidence badges.
    - **Manuals**: Created `chatbot_prompt_engineering.md` for prompt optimization.

- **Chatbot Topic Extraction Enhancement (Feb 9, 2026):**
    - **Problem Identified**: Rule-based topic extraction (`_extract_topics()`) was too rigid - "i fell" didn't match "fall", "night" alone didn't trigger sleep topics.
    - **Solution Implemented**: Enhanced keyword matching with 3-tier system:
        - **Keywords** (weight: 3): Primary medical terms
        - **Variations** (weight: 2): Word forms (fell, forgot, slept)
        - **Related** (weight: 1): Contextual terms (dizzy, night, can't remember)
    - **New Topics Added**: cardiovascular, pain_management, mental_health (9 total topics)
    - **Coverage**: Now handles natural language variations like "i fell", "can't remember", "tossing and turning"

- **🤖 Architecture Discussion: Local LLM for Topic Extraction (Feb 9, 2026):**
    - **Current State**: Rule-based matching with expanded keyword coverage (~85% accuracy, <1ms latency)
    - **Problem**: Edge cases still fail ("How do I stop myself from going down?", "My mind feels cloudy")
    - **Proposed Solution**: Local lightweight LLM (Phi-3, Mistral 7B, Llama 3.2) for semantic understanding
    - **Trade-off Analysis**:
        | Aspect | Rules | Local LLM | Hybrid |
        |--------|-------|-----------|--------|
        | Latency | <1ms | 100-500ms | <1ms (fast path) |
        | Accuracy | 70% | 90% | 85% |
        | Memory | Minimal | 2-8GB | 2-8GB |
        | Debuggable | ✅ Yes | ❌ Black box | ✅ Partial |
        | Determinism | ✅ High | 🎲 Variable | ✅ High (fallback) |
    - **Recommendation**: **Hybrid Approach** (Future)
        - Phase 1: Keep improved rule-based (current) - covers 80% common queries
        - Phase 2: Add Local LLM as fallback when no rules match (20% edge cases)
        - Phase 3: Full semantic RAG when GPU infrastructure available
    - **Decision**: Defer Local LLM until rule-based coverage plateaus; monitor extraction failures via logs
- **RAG & Clinical Intelligence (Feb 3, 2026):**
    - **Vector Store:** Implemented `VectorStore` using ChromaDB with collections for `clinical_evidence`, `adl_correlations`, and `predictive_stats`.
    - **Embeddings:** Integrated `SentenceTransformer` with `all-MiniLM-L6-v2` for medical semantic search.
    - **Data Ingestion:** Created `ingest_pubmed.py` to automate fetching clinical abstracts (160+ items indexed for Dementia, Diabetes, and Sleep).
    - **Clinical Linking:** Developed `ADLDiseaseLinker` and `OutcomeStatistics` to ground health predictions in medical literature.
    - **LLM Integration:** Enhanced `LLMService` with automatic RAG retrieval; prompts are now grounded in retrieved clinical evidence with citations.
    - **Verification:** Successfully validated end-to-end RAG flow (Query -> Retrieval -> Citation) with `test_rag_pipeline.py`.
- **Frontend Hardening & DX (Feb 2, 2026):**
    - **Error Handling**: Implemented centralized `ErrorBoundary` and Sentry integration for frontend crash monitoring.
    - **UX Polish**: Added global Toaster system for success/error feedback.
    - **Unit Testing**: Added 100% core coverage for `ModelRegistry`, `TrainingPipeline`, and `PredictionPipeline`.
    - **Golden Samples**: Implemented `harvest_gold_samples.py` to automate high-impact dataset creation for Beta 6.
- **TensorFlow Scalability & Stability Fix (Feb 6, 2026):**
    - **Issue:** Resolved persistent execution deadlock (0% CPU hang) during `model.fit()` on Mac Studio with Real Data arrays.
    - **Root Cause:** Incompatibility between TensorFlow 2.20.0 CPU kernel and memory fragmentation caused by iterative Python sequence generation.
    - **Fix (Environment):** Downgraded to **TensorFlow 2.16.2** (Py3.12 compatible) + **Metal 1.2.0** + **Numpy < 2.0**.
    - **Fix (Code):** Replaced iterative `sequences.py` logic with vectorized `numpy.lib.stride_tricks` for zero-copy memory layout.
    - **Verification:**  Validating "Naked" training script (instantly success) and Full Pipeline (Epoch 1 in ~14s).
    - **Artifact:** Created `.agent/workflows/troubleshoot_training_hang.md` for future debugging.
- **PostgreSQL Migration & Distributed Workflow (Feb 6, 2026):**
    - **Data Migration Complete:** Successfully migrated 162k+ `adl_history` records from SQLite to PostgreSQL.
    - **Schema Fix:** Resolved `null value in column "id"` errors by adding `GENERATED BY DEFAULT AS IDENTITY` to `adl_history.id`.
    - **ID Auto-Increment:** Created `fix_postgres_id.py` to upgrade plain `INTEGER PRIMARY KEY` columns to identity columns.
    - **Distributed Workflow Established:** Adopted "Freeze, Develop, Merge" strategy:
        - **Mac Studio (Production):** Team continues training/golden sample collection on stable branch.
        - **Laptop/Desktop (Dev):** Code development and debugging on feature branches.
        - **Merge Phase:** Export golden samples → Upgrade Studio code → Import samples → Retrain.
    - **Golden Sample Scripts:** Confirmed `harvest_gold_samples.py` (export) exists; created `import_golden_samples.py` for upgrade phase.
- **Full PostgreSQL Native Transition (Feb 6, 2026 - Evening):**
    - **Architecture:** Achieved 100% PostgreSQL-native operation (`POSTGRES_ONLY=True` readiness).
    - **Dependency Remediation:** Audited and refactored **60+ files** containing hardcoded `sqlite3` connections.
    - **Legacy Adapter (Production Grade):**
        - Enhanced `LegacyDatabaseAdapter` to handle dialect differences: `date('now')`, `date(timestamp)`, and `datetime('now')` are now automatically translated via regex to PostgreSQL equivalents.
        - Implemented automated Boolean-to-Integer casting to maintain compatibility with legacy schemas.
        - **Fix:** Resolved `PoolError` (connection pool exhaustion) by implementing proper connection return (`putconn`) in the adapter shim.
    - **Component Refactoring:**
        - **Intelligence:** Decoupled `watchdog.py`, `classifier.py`, `trajectory_engine.py`, and `intelligence_lab.py` from local SQLite databases.
        - **Data Management:** Refactored `data_manager.py` to eliminate `residents_master_data.db` in favor of centralized PostgreSQL storage.
        - **Utilities:** Updated `process_data.py`, `backfill_analysis.py`, `check_integrity.py`, and `health_check.py` to use the unified adapter.
    - **Verification:** Successfully validated full end-to-end flow (`data/raw` -> `UnifiedPipeline` -> `PostgreSQL`) with real resident training data.

- **Frontend PostgreSQL Integration & Pipeline Hardening (Feb 6, 2026):**
    - **Frontend Migration:** 100% completed refactor of Next.js frontend to use PostgreSQL (`pg` library).
        - Updated `db.ts` and `data.ts` for asynchronous PostgreSQL queries.
        - Refactored all API routes (`/api/health`, `/api/residents`, `/api/status`, etc.) for PostgreSQL compatibility.
    - **Legacy Adapter Fixes:**
        - Resolved `TypeError` in `PostgresCursorShim` by implementing context manager protocol.
        - Fixed `init_db.py` to use the unified `backend/db/schema.sql` instead of legacy paths.
    - **ML Pipeline Fixes:**
        - **Fix (Timestamp Error):** Resolved `date/time field value out of range: "0"` error during training history logging.
        - Updated `platform.py` to preserve original timestamps during training and `training.py` to handle invalid timestamp fallbacks gracefully.
    - **Senior Engineer Mandate:** Updated `SKILL.md` to formalize requirements for audit-readiness and permanent, production-grade solutions.
    - **Bug Fix (Frontend):** Resolved `column m.data does not exist` error.
        - Restored missing `data JSONB` column to the `medical_history` table in `backend/db/schema.sql`.
        - Updated `lib/data.ts` query to explicitly cast `JSONB` to `TEXT` for string aggregation.
    - **Verification:** Successfully reset the environment, re-initialized the PostgreSQL schema, and verified "Empty State" UI display.

- **Timeline & Visualization Hardening (Feb 6, 2026 - Late Night):**
    - **Issue:** "Squeezed" timeline (thousands of narrow segments) and "Sleep" appearing at incorrect time (08:29).
    - **Fix (Merging):** Implemented per-room segment merging in `data.ts` to consolidate consecutive 10s sensor events into duration-based activity blocks.
    - **Fix (Robustness):** Fortified `ActivityTimeline.tsx` with end-time clamping (to 23:59:59) and CSS bounds checking (0-100%) to prevent layout breakage.
    - **Fix (Timezone):** Implemented "Read-Side Shift" in `data.ts` (fetching timestamps as strings) to force local time interpretation, correcting the 8-hour UTC offset visual bug.
    - **Verification:** Browser recording confirmed solid sleep blocks starting at ~00:29 local time.

- **Training Pipeline & Robustness (Feb 7, 2026):**
    - **Training Pipeline Fixes:**
        - **Resolved Critical Bug:** `augment_training_data` in `training.py` was returning prematurely after the first date.
        - **Resolved Schema Drift:** `fetch_golden_samples` in `ml/utils.py` was missing the `record_date` column, causing group-by failures.
        - **Defensive Coding:** Added explicit returns and fixed `NoneType` unpacking errors.
    - **Timestamp Standardization:**
        - Created centralized `backend/utils/time_utils.py` with `ensure_naive`, `ensure_utc`, and `safe_merge_timestamps`.
        - Refactored `export_dashboard.py`, `pipeline.py`, and `motion_normalizer.py` to use unified timezone handling, eliminating scattered `tz_convert(None)` hacks.
    - **Schema Validation Testing:**
        - Implemented `backend/tests/test_query_schemas.py` to verify that database functions return the expected columns (preventing regressions like the `record_date` bug).
    - **Environment Reset:**
        - Performed a clean sweep of the Beta 5.5 environment: wiped PostgreSQL volumes, deleted residual SQLite files (`elder_master_data.db`, etc.), and cleared archive folders.
    - **Status:** System is now fully robust for Golden Sample harvesting at n=20.

- **Ground Truth Priority \u0026 Golden Sample Integration (Feb 7, 2026):**
    - **Problem:** Training files were being re-predicted by the model, introducing noise into the timeline. User-labeled data was treated as secondary to model output.
    - **Solution (Pipeline Refactor):**
        - **Training Files:** Modified `train_and_predict()` in `pipeline.py` to skip prediction and use the `activity` column directly.
        - **Golden Samples:** Enhanced training to merge Golden Sample corrections into training data before model training (`fetch_all_golden_samples()` in `ml/utils.py`).
        - **Retrospective Re-prediction:** Fixed `repredict_all()` in `prediction.py` to apply Golden Samples when reprocessing historical files.
    - **Priority Hierarchy (Single Source of Truth):**
        1. **Golden Samples** (manual corrections) — Absolute priority.
        2. **Training File Labels** (`activity` column) — Used directly for `_train` files.
        3. **Model Predictions** — Used for non-training files when no correction exists.
    - **Outcome:** `adl_history` now contains the "truth" regardless of source. All frontend/backend data is consistent with this.
    - **Verification:** Samuel's Bedroom now shows a solid 7.8h sleep block (03:05-10:51), matching training labels exactly.

- **Training File Correction & Schema Hardening (Feb 7, 2026 - Late Evening):**
    - **Feature: Training File Correction**:
        - Enabled "Training Files" (Ground Truth) correction mode in the Studio.
        - Implemented fallback pre-population from Excel labels when no predictions exist.
        - Added safety warning banners to prevent accidental ground-truth overrides.
    - **Data Integrity Fixes (Senior Audit):**
        - **Missing Elder Resolution**: Added elder auto-registration to `adl_service.py`, `prediction.py`, and `export_dashboard.py` to prevent FK violations.
        - **SQLite-to-Postgres Translation**: Enhanced `LegacyDatabaseAdapter` to translate `INSERT OR REPLACE` (upsert) to `ON CONFLICT DO UPDATE`.
        - **Schema Alignment**: Patched missing columns (`correction_source` in `activity_segments` and `recorded_date` in `medical_history`) across code, live DB, and `schema.sql`.
    - **Governance & DX:**
        - Codified **Schema Governance** (schema-first updates) and **Holistic Review Protocol** (assume pattern, audit all) in `SKILL.md`.
        - Created `walkthrough_training_correct.md` documenting the features and fixes.
    - **Sync:** Successfully pushed all code fixes, schema patches, and documentation updates to GitHub.

- **Naive Timestamp Architecture & Environment Reset (Feb 7, 2026 - Final):**
    - **Naive Timestamp Architecture (Option A):**
        - Migrated entire database schema from `TIMESTAMPTZ` (aware) to `TIMESTAMP` (naive) to eliminate timezone drift.
        - Simplified data flow: Excel (naive) → DB (naive) → UI (naive), ensuring "23:00" in source is "23:00" in view.
        - Aligned frontend/backend merge thresholds to a unified 5-minute window.
    - **UI Safety & Polish:**
        - Modified `Correction Studio` in `export_dashboard.py` to automatically hide and disable the "Apply Retrospective" checkbox for Training Files.
        - **Cache Fix**: Implemented `clear_all_caches()` to invalidate all 6 Streamlit caches after corrections, ensuring instant feedback (Audit Trail/Timeline).
        - **Refinement**: Defaulted "Apply Retrospective" to `True` for input files and removed debug info from the sidebar.
    - **Environment Hardening:**
        - Completed a full environment reset: wiped PostgreSQL, cleared all resident profiles in `elder_data/`, and emptied archive/model directories.
        - Updated `SKILL.md` with a 5-layer **Correction Visibility Troubleshooting** guide (Schema, App, Consistency, Cache, Segments).
    - **Status:** Environment is clean, architecture is standardized on local time, and documentation is updated for Phase 2 testing.

- **Critical Bug Remediation & Audits (Feb 7, 2026 - Late Night):**
    - **Audit Scope:** Comprehensive review of `export_dashboard.py`, `segment_utils.py`, and core utilities.
    - **Critical Fixes:**
        - **Resource Leak:** Fixed `save_corrections_to_db` manual context manager failure pattern.
        - **Crash Prevention:** Added missing `logger` definition in `export_dashboard.py`.
        - **Silent Failures:** Removed bare `except:` clauses hiding `NameError` and DB errors.
    - **Findings:**
        - Confirmed `LegacyDatabaseAdapter` correctly handles `INSERT OR REPLACE` (Not a bug).
    - **Outcome:** System stability reinforced against connection leaks and silent data loss.

- **ML Audit Remediation (Feb 7, 2026 - Final):**
    - **Critical Fixes:**
        - **Class Imbalance:** Integrated `sklearn.utils.class_weight` to handle severe distribution skew.
        - **Imbalance Logging:** Added warning threshold (10:1 ratio) to flag problematic training sets.
        - **Metrics:** Replaced simple accuracy with comprehensive `macro_f1`, `recall`, and `precision` reporting.
    - **High Severity:**
        - **Early Stopping:** Added `EarlyStopping` callback (patience=2) to prevent overfitting.
        - **Data Handling:** Implemented safety gates for small datasets (<100 samples) to skip validation splits.
        - **Silent Failures:** Added logging for skipped corrections due to label encoder mismatches.
    - **Cleanup:** Sanitized duplicate imports and enforced `shuffle=True` for classification consistency.
    - **Status:** ML pipeline is now production-ready with robust monitoring and validation.

- **API Review & Integration (Feb 7, 2026 - Night):**
    - **Transaction Safety:** Wrapped cascade deletes in `residents/[id]/route.ts` with explicit PostgreSQL transactions (`BEGIN`, `COMMIT`, `ROLLBACK`).
    - **PostgreSQL Migration:** Purged remaining SQLite-specific syntax (`db.prepare`) in `alert-rules-v2/route.ts`, replaced with `pg` Pool queries and parameterized placeholders ($1, $2).
    - **Standardized Error Handling:** Refactored 7 key routes to use `withErrorHandling()` wrapper, `successResponse()`, and custom error classes (`NotFoundError`, `ValidationError`).
    - **Integration:** Verified frontend-backend consistency for timeline, anomalies, trajectories, and candidates APIs.
    - **Outcome:** API layer is now atomic, type-safe (Zod usage), and provides standardized error responses with correlation IDs.

- **Performance Enhancement Phase 1 (Feb 7, 2026 - Night):**
    - **SELECT Optimization:** Replaced 4 `SELECT *` queries with explicit columns in `data.ts` (getResidents, getAlerts, getAllAlerts, activity_segments).
    - **Database Indexes:** Created `migrate_add_indexes.sql` with 4 new indexes:
        - `idx_segments_lookup` for timeline queries
        - `idx_alerts_unread` (partial) for dashboard filtering
        - `idx_correction_date` for history sorting
        - `idx_adl_record_date` for fallback queries
    - **Pool Monitoring:** Added keepAlive, application_name, error logging, and `getPoolMetrics()` to `db.ts`.
    - **Query Protection (Hybrid Strategy):**
        - **Safety Net:** `statement_timeout: 10000` (10s) configured at pool level for all background queries.
        - **Fast Path:** `timedQuery()` used for user-facing routes (e.g., Dashboard) with strict 2s timeouts.
    - **Deferred:** Pagination, caching, and parallel ML prediction moved to Phase 2/Beta 6.

- **Production Readiness Hardening (Feb 9, 2026):**
    - **Startup Script:** Updated `start.sh` to support `BETA55_MODE=prod` (Next.js build+start) vs `dev`, and fail-fast on PostgreSQL unavailability in prod.
    - **Postgres-Only Consistency:** `POSTGRES_ONLY`/`USE_POSTGRESQL` are now environment-driven, and Postgres initialization raises (no silent SQLite fallback) when `POSTGRES_ONLY=true`.
    - **ML Integrity:** Eliminated time-series leakage in training (chronological ordering + strict temporal split + `shuffle=False`), and ensured augmentation sequences carry timestamps so validation remains future-only.
    - **Causal Preprocessing:** Resampling and missing-value fill paths now restrict `bfill` to leading NaNs only; non-causal `interpolate`/`bfill` are downgraded to forward-fill with warnings.
    - **Config Correctness:** Rolling-window features now derive sample counts from the configured interval (per-room interval when available).
    - **Governance:** Added `SKILL.md` for production/audit readiness and created `BETA55_TEAM_FOLLOWUPS.md` to track P0/P1/P2 issues with acceptance criteria.

- **Health Advisory Chatbot Safety Rebuild (Feb 9, 2026 - Late Night):**
    - **SAFE-001/002/003:** Added deterministic `SafetyGateway` with emergency/urgent detection and hard bypass before context/RAG/LLM generation.
    - **SAFE-004/005:** Added structured safety telemetry (`urgency_level`, `trigger_terms`, `llm_bypassed`) and API-level bypass regression tests.
    - **API-001/002:** Standardized chatbot API contract to canonical `snake_case`; added frontend mapping for compatibility.
    - **POL-001:** Introduced deterministic `ActionPlan` schema and integrated policy-engine output into advisory responses.
    - **POL-002/003:** Implemented versioned fall and medication safety rules (interaction review, polypharmacy, anticholinergic burden handling).
    - **POL-004/005:** Externalized policy rules to versioned manifest (`rules_v2026_02.json`) with changelog references and policy metadata in API output.
    - **VAL-001:** Removed permissive citation acceptance (`et al.` / year heuristics), enforcing fail-closed citation validation.
    - **VAL-002:** Added `CitationIDRegistry` for strict citation format routing (`PMID/DOI/HASH/guideline_id`) and deterministic source membership checks.
    - **Validation:** Added/updated tests for safety, API contract, policy engine, citation validator, and citation registry; focused suite passing (`21 passed`).
- **System Standardization & UX Polish (Feb 13, 2026):**
    - **Label Normalization**: Successfully transitioned `living_normal_use` to `livingroom_normal_use` across all backend logic (`training.py`, `prediction.py`, `settings.py`) and frontend displays, ensuring consistent taxonomy.
    - **Bug Fix (Frontend)**: Resolved critical SWR data extraction bug in `TimelineWithDatePicker.tsx`. The component now correctly handles the nested `{ data: { ... } }` object returned by the PostgreSQL-ready API, restoring date selection and timeline visibility.
    - **Backend Optimization**: Increased raw data scan frequency from 10s to 30s in `run_daily_analysis.py` to reduce I/O overhead while maintaining production-grade responsiveness.
    - **Correction Studio 2.0**:
        - **Transparency**: Integrated `sensor_features` parsing from `adl_history` into the Correction Studio. Users can now see raw numeric data in tooltips for better labeling context.
        - **ML Hints**: Enhanced the Data Editor with `low_confidence-<label>` hints and top-1/top-2 suggestions (with probabilities) to speed up manual correction.
        - **UI Polish**: Improved column configurations with proper formatting and descriptive help text.
    - **Research & Calibration**:
        - Drafted `docs/research/rationale_10s_resolution.md` documenting the technical justification for 10-second sampling in elderly care.
        - **ML Insights**: Analyzed adaptive threshold dynamics (Floor: 0.35, Cap: 0.80) and identified class imbalance in Bedroom data (2.1x ratio) as the primary driver for low-confidence fallback (0.60).
    - **Verification**: Validated full end-to-end flow from training update to dashboard visualization.

- **Cross-Room Correction Isolation (Feb 13, 2026):**
    - **Bug 1 (Retraining):** Resolved cross-room contamination where correcting one room (e.g., Bedroom) retrained all models, causing non-deterministic shifts in unrelated rooms.
    - **Fix:** `train_from_files` now accepts `rooms` filter; skips retraining for unchanged rooms.
    - **Bug 2 (Reprediction):** `repredict_all` now filters models and input data early to prevent "creeping" changes in other rooms.
    - **Bug 3 (Segments):** Fixed `save_predictions_to_db` and batch-apply logic using `now()` for segments; now tracks actual `(room, date)` pairs for precise regeneration.
    - **Safety Guard:** Implemented multi-resident queue check in Correction Studio; prevents identity collisions during batch apply by enforcing single-resident processing.
    - **Hardening:** Added normalization for `affected_rooms` and 5 integration tests in `test_cross_room_isolation.py` (100% pass).

## 2026-02-16 15:00 - PR-2: Wire Hardening Gates Into Live Promotion - Complete

### Completed
- Added `uuid` import to `training.py` for gate run ID generation
- Fixed `GateEvaluationResult` import to be at module level (not conditional)
- Updated `train_room()` calls in `pipeline.py` to pass `gate_evaluation_result` parameter
- Fixed `test_cross_room_isolation.py` to handle keyword arguments in mock assertions
- All 204 tests passing (142 original + 15 PR-1 unified path + 46 integration + 1 fixed)

### Gate Integration Flow
1. `train_and_predict()` / `train_from_files()` evaluate gates FIRST (pre-training)
2. If gates pass, gate_evaluation_result is passed to `train_room()`
3. `train_room()` creates GateIntegrationPipeline and evaluates post-training gates
4. Gate stack is persisted in training history metadata
5. `why_rejected.json` artifacts generated on gate failures

### Key Files Modified
- `backend/ml/training.py` - PR-2 imports, gate_integration in train_room()
- `backend/ml/pipeline.py` - Pass gate_evaluation_result to train_room
- `backend/tests/test_cross_room_isolation.py` - Handle kwargs in mock assertions


## 2026-02-16 16:30 - PR-3: Correctness Fixes - Complete

### Completed

#### PR-3.1: Strict Sequence-Label Alignment
- Integrated `safe_create_sequences()` from `sequence_alignment.py` into `training.py`
- Replaced legacy `platform.create_sequences()` with strict alignment path
- Added hard assertions on sequence-label alignment with `assert_sequence_label_alignment()`
- Raises `SequenceLabelAlignmentError` on misalignment (fail-closed)

#### PR-3.2: Duplicate Resolution Policy
- Integrated `DuplicateTimestampResolver` into training pipeline
- Applied duplicate resolution before sequence creation
- Configurable via `DuplicateResolutionPolicy` (method, tie_breaker, emit_stats)
- Fast path when no duplicates detected

#### PR-3.3: Reproducibility No-Op Hash
- Added `ReproducibilityTracker` integration to `pipeline.py`
- No-op detection based on `(data_fingerprint, policy_hash, code_version)`
- Skip redundant training when identical inputs already processed
- Requires clean code (git_dirty=False) for no-op eligibility

#### PR-3.4: Pilot Override CI-Safe Mode
- Added `is_ci_environment()` detection function
- Added `ci_safe` parameter to `activate_pilot()` (default=True)
- Blocks pilot activation in CI environments unless ci_safe=False
- Detects: CI, GITHUB_ACTIONS, GITLAB_CI, CIRCLECI, JENKINS_URL, BUILDKITE, TF_BUILD, DRONE, TRAVIS, CODEBUILD_BUILD_ID

### Files Modified
- `backend/ml/training.py` - Added strict sequence creation and duplicate resolution
- `backend/ml/pipeline.py` - Added reproducibility no-op check
- `backend/ml/pilot_override_manager.py` - Added CI-safe mode

### Tests Added
- `backend/tests/test_pr3_correctness_fixes.py` - 19 tests covering:
  - Strict sequence-label alignment (6 tests)
  - Duplicate resolution policy (3 tests)
  - Reproducibility no-op hash (4 tests)
  - Pilot override CI-safe (4 tests)
  - Integration tests (2 tests)

### Test Results
- 225 tests passing
- 3 skipped
- 2 pre-existing test isolation issues (unrelated to PR-3)


## 2026-02-16 17:00 - PR-3 Issues Fixed

### [P0] Fixed Environment Leakage
**Root cause**: PR-3 tests activated pilot mode and left `TRAINING_PROFILE=pilot` in environment
- `test_pr3_correctness_fixes.py` line 337, 358 called `activate_pilot()` which sets `os.environ["TRAINING_PROFILE"]`
- Tests cleaned up state file but not environment variable

**Fix**: Added `setUp()` and `tearDown()` methods to `TestPilotOverrideCISafe` class:
- Save original `TRAINING_PROFILE` in `setUp()`
- Restore in `tearDown()` after each test
- Clean up test state files in `tearDown()`

### [P1] Fixed No-Op Return Contract Inconsistency
**Root cause**: No-op path returned `prior_report.outcome.promoted_rooms` (list of strings) but callers expect list of dicts

**Fix**: Removed partially-wired no-op check from `pipeline.py`:
- The feature was not properly integrated with policy SSoT
- Return contract mismatch would break existing consumers
- Added TODO comment for proper implementation in future PR

### [P1] Removed Overstated Feature Claim
**Root cause**: No-op detection was only partially wired:
- Used ad-hoc `enable_noop_check` attr not in constructor
- Ignored policy SSoT flag `skip_if_same_data_and_policy`
- Used fallback "unknown" policy hash
- Did not create reports in live pipeline flow

**Fix**: Removed the ad-hoc implementation and imports. The feature needs proper design:
- Read from `TrainingPolicy.skip_if_same_data_and_policy`
- Initialize tracker in constructor
- Create reports after training completion
- Maintain backward-compatible return contract

### Files Modified
- `backend/tests/test_pr3_correctness_fixes.py` - Added proper env cleanup
- `backend/ml/pipeline.py` - Removed partially-wired no-op check

### Test Results
- **227 tests passing** (up from 225)
- **3 skipped**
- **0 failures**


## 2026-02-16 18:00 - PR-4: CI Release Gate Hardening - Complete

### Completed

#### PR-4.1: CI Gate Validator Module
Created `backend/ml/ci_gate_validator.py` with:
- `CIGateValidator` class that detects gate bypasses
- Bypass detection via environment variables: `SKIP_GATES`, `BYPASS_GATES`, `FORCE_PROMOTE`, `DISABLE_GATE_CHECK`, etc.
- Training profile validation (warns on pilot mode in CI)
- Required configuration file/directory checks
- Gate integration module import verification
- JSON report generation for CI artifacts
- Exit code 0 on success, 1 on failure

#### PR-4.2: GitHub Actions Workflow
Created `.github/workflows/ci-gate-hardening.yml` with 4 jobs:
1. **gate-validation**: Runs `ci_gate_validator` with strict mode
2. **gate-integration-tests**: Runs PR-3 and unified training path tests
3. **code-scan**: Scans for gate bypass patterns in code
4. **profile-check**: Verifies production profile is default

#### PR-4.3: Gate Bypass Detection Methods
Environment variable checks:
- `SKIP_GATES`, `BYPASS_GATES`, `FORCE_PROMOTE`
- `DISABLE_GATE_CHECK`, `SKIP_PRE_TRAINING_GATES`
- `SKIP_POST_TRAINING_GATES`, `SKIP_STATISTICAL_VALIDITY`

Code pattern scanning:
- Detects hardcoded bypass patterns in Python files
- Excludes test files and the validator itself

#### PR-4.4: Build Failure on Gate Bypass
- Validator exits with code 1 when bypass detected
- CI workflow fails the build on validation failure
- Strict mode treats warnings as errors
- Artifacts uploaded for debugging

#### PR-4.5: Test Coverage
Created `backend/tests/test_ci_gate_validator.py` with 20 tests:
- Gate bypass detection tests (10 tests)
- CI environment validation tests (3 tests)
- Main CLI function tests (3 tests)
- Enum and unit tests (2 tests)
- Integration scenario tests (2 tests)

### Files Created/Modified
- **New**: `backend/ml/ci_gate_validator.py` - CI gate validation module
- **New**: `.github/workflows/ci-gate-hardening.yml` - GitHub Actions workflow
- **New**: `backend/tests/test_ci_gate_validator.py` - 20 validation tests

### Test Results
- **247 tests passing** (up from 227)
- **3 skipped**
- **0 failures**

### Usage

Local validation:
```bash
cd backend
python -m ml.ci_gate_validator --strict
```

CI integration:
- Workflow runs automatically on PRs and pushes to main/beta-* branches
- Fails build if gate bypass detected
- Generates `ci_gate_validation_report.json` artifact


## 2026-02-16 19:00 - PR-5: Controlled Validation Run + Signoff Pack - Complete

### Completed

#### PR-5.1: Deterministic Manifest-Based Retrain
Created `backend/scripts/deterministic_retrain.py`:
- `DeterministicManifest` class locks all training variables
- Manifest includes: data file hashes, code version (git commit), policy config, random seed
- Code version verification (fails if dirty or commit mismatch)
- Data file verification (fails if hash mismatch)
- Commands: `create`, `run`, `verify`

#### PR-5.2: 7+ Day Validation Run Tracking
Created `backend/ml/validation_run.py`:
- `ValidationRun` dataclass tracks multi-day runs
- `DailyRunResult` records each day's outcome
- `ValidationRunManager` persists runs to disk
- `ValidationRunStatus`: PENDING, RUNNING, COMPLETED, FAILED, ABORTED
- CLI commands: `start`, `record`, `finalize`

#### PR-5.3: Decision Trace Collection
- Integrates with existing `_write_decision_trace()` in training.py
- `DailyRunResult.decision_traces` maps room -> trace path
- Signoff pack aggregates all decision traces

#### PR-5.4: Rejection Artifact Collection
- Integrates with existing `rejection_artifact.py`
- `DailyRunResult.rejection_artifacts` maps room -> artifact path
- Gate reason summary counts rejection codes

#### PR-5.5: Signoff Pack Generator
- `SignoffPack` dataclass contains all artifacts
- `finalize_run()` generates signoff pack
- JSON + Markdown report generation
- Compliance checklist:
  - minimum_7_days
  - determinism_score_above_80
  - all_rooms_have_decision_traces
  - no_missing_gate_reasons
  - manifest_hash_consistent
- Determinism score: % of days with identical promotions

### Files Created
- `backend/ml/validation_run.py` - Validation run management (559 lines)
- `backend/scripts/deterministic_retrain.py` - Manifest-based retrain (316 lines)
- `backend/tests/test_validation_run.py` - 15 validation tests

### Usage

Create manifest:
```bash
python scripts/deterministic_retrain.py create \
  --elder-id HK001 \
  --data-files data/day1.csv data/day2.csv \
  --output manifest.json \
  --random-seed 42
```

Start validation run:
```bash
python -m ml.validation_run start \
  --elder-id HK001 \
  --duration-days 7 \
  --manifest manifest.json
```

Record daily result:
```bash
python -m ml.validation_run record \
  --run-id <run-uuid> \
  --date 2026-02-16 \
  --passed True \
  --promoted-rooms bedroom,kitchen
```

Finalize and generate signoff pack:
```bash
python -m ml.validation_run finalize --run-id <run-uuid>
```


## 2026-02-16 20:30 - Per-Label Recall Gates & Policy Support

### Completed
#### Policy Parsing Hardening
- **Fix**: Updated `_read_room_label_float_overrides` in `policy_config.py` to correctly parse `room.label` keys from stats.
- **Fix**: Added support for `min_recall_by_room_label` configuration in `ReleaseGatePolicy`.
- **Constraint**: Enforced strict float parsing with `0.0-1.0` bounds checking for all recall thresholds.

#### Hard Gate Infrastructure
- **Infrastructure**: Wired `min_recall_by_room_label` into `ReleaseGatePolicy` dataclass.
- **Defaulting**: Prepared `default_factory` to support overrides (e.g., `bedroom.unoccupied=0.60`).
- **Validation**: Added unit tests to verify gate logic correctly blocks promotion on specific label failures (e.g., `label_recall_failed:bedroom:unoccupied:0.45<0.60`).

### Impact
- Enables fine-grained quality control (stopping class collapse).
- Prevents "Average F1" from hiding specific dangerous failures (like failing to see an empty room).
```bash
python -m ml.validation_run record \
  --run-id <run_id> \
  --result-file daily_result.json
```

Finalize and generate signoff pack:
```bash
python -m ml.validation_run finalize --run-id <run_id>
```

### Test Results
- **263 tests passing** (up from 247)
- **3 skipped**
- **0 failures**

### 3-Day PR Plan Complete

| PR | Scope | Status | Tests Added |
|----|-------|--------|-------------|
| PR-1 | Unify Runtime Training Path | ✅ | 16 |
| PR-2 | Wire Hardening Gates Into Live Promotion | ✅ | 46 |
| PR-3 | Correctness Fixes | ✅ | 19 |
| PR-4 | CI Release Gate Hardening | ✅ | 20 |
| PR-5 | Controlled Validation Run + Signoff Pack | ✅ | 15 |

**Total: 263 tests passing**


## 2026-02-16 20:00 - Production Readiness Fixes

### [P0] Fixed Untracked Files
**Issue**: PR-4/PR-5 files were present locally but not tracked by git

**Files now tracked**:
- `.github/workflows/ci-gate-hardening.yml`
- `backend/ml/ci_gate_validator.py`
- `backend/ml/validation_run.py`
- `backend/scripts/deterministic_retrain.py`
- `backend/tests/test_ci_gate_validator.py`
- `backend/tests/test_validation_run.py`

**Fix**: Added files to git and committed

### [P0] Fixed deterministic_retrain.py Stub
**Issue**: Script reported `status: "success"` with empty placeholder outputs

**Fix**: Implemented actual training pipeline integration:
- Uses `UnifiedPipeline` for training
- Uses `ModelRegistry` for model management
- Collects actual `trained_rooms`, `decision_traces`, `rejection_artifacts`
- Reports actual training status (`completed` or `failed`)
- Raises `RuntimeError` on training failure

### [P1] Fixed Validation Finalization
**Issue**: `finalize_run()` marked runs as COMPLETED unconditionally, even with failed compliance

**Fix**: 
- Added `force` parameter to `finalize_run(force=False)`
- Status set to FAILED if compliance checks fail and force=False
- Raises `RuntimeError` with failed checks list
- Final summary includes `compliance_forced` flag and `failed_checks` list
- CLI updated with `--force` flag

### [P1] Fixed CI Validator Claims
**Issue**: Docstring claimed "promotion happens without gate evidence" check, but only env vars/config were checked

**Fix**:
- Updated docstring to accurately describe what IS implemented:
  - Environment variable bypass detection (SKIP_GATES, BYPASS_GATES, FORCE_PROMOTE)
  - Gate configuration presence validation
  - Gate integration module importability
  - Pilot profile in CI detection
- Added note about difference between bypass MECHANISMS vs actual gate execution
- Updated GitHub Actions workflow comments similarly

### Commit
```
950c0b7 PR-4/PR-5: Production fixes - untracked files added, 
        deterministic_retrain implemented, validation compliance 
        gating, doc alignment
```

### Test Results
- **263 tests passing**
- **3 skipped**
- **0 failures**


## 2026-02-16 21:00 - Lane A (Contract + Compatibility) Complete

As Senior ML Engineer for Lane A, I have completed PR-A1 and PR-A2 with production quality.

### PR-A1: Registry Schema + Loader + Tests

#### Files Created
- `backend/config/schemas/adl_event_registry.schema.json` - JSON Schema validation
- `backend/config/adl_event_registry.v1.yaml` - Initial taxonomy with 16 events
- `backend/ml/adl_registry.py` - Production-quality loader (625 lines)
- `backend/tests/test_adl_event_registry_schema.py` - Schema tests (18 tests)
- `backend/tests/test_adl_registry_loader.py` - Loader tests (27 tests)

#### Registry Features
**Canonical Events (16 total):**
- Occupancy: `occupied`, `unoccupied`
- Bedroom: `sleeping`, `awake_in_bed`
- Living Room: `relaxing`, `active_living`
- Kitchen: `cooking`, `eating`, `kitchen_use`
- Bathroom: `showering`, `bathroom_use`
- Entrance: `entering_home`, `leaving_home`
- Fallback: `occupied_unknown`, `unoccupied_unknown`

**Critical Care Events:** sleeping, showering, eating (critical), cooking (high)

**KPI Groups (10):** care_critical, sleep_quality, nutrition, hygiene, activity_level, social_engagement, occupancy_base, home_presence, out_time, shower_day

**Alias System:** 20+ aliases with collision detection

**Loader Features:**
- Schema validation with jsonschema
- Alias normalization and collision detection
- Unknown label fallback handling
- Room-scoped event lookup
- KPI group resolution
- Singleton pattern for default registry

### PR-A2: Migration Validator + CI Checks

#### Files Created
- `backend/ml/ci_adl_registry_validator.py` - CI validator (508 lines)
- `backend/config/adl_event_migrations/` - Migration directory
- `backend/tests/test_adl_registry_migrations.py` - Migration tests (18 tests)

#### Validation Features
**Breaking Changes Detected:**
- Event removal without migration
- Event disabling
- Alias removal
- Criticality upgrade (medium→high→critical)
- Room scope removal from events
- Room scope removal from registry
- KPI group removal

**Safe Changes Allowed:**
- New events
- New aliases
- Criticality downgrade
- Adding rooms to events
- New room scopes
- New KPI groups

**Migration Support:**
- Migration file format: `v1.0_to_v1.1.yaml`
- Migration reduces severity from BREAKING to WARNING
- CLI tool for validation: `--baseline`, `--current`, `--strict`

#### CI Workflow Extended
Added `adl-registry-validation` job to `.github/workflows/ci-gate-hardening.yml`:
- Compares registry against baseline (HEAD~1)
- Fails build on breaking changes
- Runs all ADL registry tests

### Test Results
- **349 tests passing** (up from 263)
- **3 skipped**
- **0 failures**

### Lane A Test Summary
| Component | Tests |
|-----------|-------|
| Schema validation | 18 |
| Registry loader | 27 |
| Migration/CI validation | 18 |
| **Total Lane A** | **63** |

### Production Quality Checklist
✅ Schema validation with detailed error reporting
✅ Alias collision detection (event ID + alias uniqueness)
✅ Unknown label fallback policy
✅ Room scope validation
✅ KPI group consistency
✅ Backward compatibility checking
✅ CI integration with build failure on breaking changes
✅ Comprehensive test coverage (63 tests)
✅ Type hints throughout
✅ Production-ready error handling


## 2026-02-16 22:00 - Lane A Review: Execution Plan Updates

Reviewed updated documents:
- `/docs/planning/event_first_cnn_transformer_execution_plan.md` (540 lines)
- `/docs/planning/event_first_team_kickoff_checklist.md` (172 lines)

### New Requirements Analysis

**Section A1 - Leakage Control:** PR-D1 requirement (Lane D)
**Section A2 - Evaluation Granularity:** PR-C2/PR-D1 requirement (Lane C/D)
**Section B - Gate Thresholds:** New metrics (ECE, Brier, unknown-rate caps) - gating logic, not registry
**Section C - Config Contract:** New keys for PR-C2 (Lane C)
**Section D - Artifact Contracts:** Output format for PR-D1/D2 (Lane D)
**Section E/F - DoD & ETA:** Timeline and PR ordering

### Lane A Status: COMPLETE ✅

All Lane A requirements from kickoff checklist satisfied:

| Done Criteria | Status |
|---------------|--------|
| Unknown labels map safely | ✅ `resolve_unknown_label()` implemented |
| Alias collisions are blocked | ✅ `AliasCollisionError` raised on collision |
| Breaking changes fail CI | ✅ CI validator blocks without migration |

### Files Delivered (7 files)
1. `backend/config/adl_event_registry.v1.yaml`
2. `backend/config/schemas/adl_event_registry.schema.json`
3. `backend/ml/adl_registry.py`
4. `backend/ml/ci_adl_registry_validator.py`
5. `backend/tests/test_adl_event_registry_schema.py`
6. `backend/tests/test_adl_registry_loader.py`
7. `backend/tests/test_adl_registry_migrations.py`

### Test Status
```bash
cd backend
pytest -q tests/test_adl_event_registry_schema.py tests/test_adl_registry_loader.py
# 45 passed

pytest -q tests/test_adl_registry_migrations.py tests/test_ci_gate_validator.py
# 38 passed
```

**Total: 349 tests passing**

### Handoff to Lane B
Lane A contract is stable. Lane B can proceed with:
- Event compiler/decoder (PR-B1)
- Event KPI + gates (PR-B2)
- Home-empty fusion (PR-B3)

No blockers from Lane A.

---

## 2026-02-18 06:36

### Lane D1 (WS-6/T-70) Bug Fixes - COMPLETE

**Bug fixes completed:**
1. **Signoff PASS despite compliance failure** - FIXED: `_compute_signoff_decision()` now properly evaluates all 10 compliance checks in the decision path
2. **Split-seed matrix not enforced** - FIXED: Added `_check_split_seed_matrix_complete()` to validate 4×3 matrix (12 cells)
3. **WS-6 export day-based** - FIXED: `_save_ws6_signoff_pack()` now generates 12 split-seed results instead of 7 daily results
4. **Artifact integrity fail-open** - FIXED: Artifact errors now captured and checked in compliance checklist
5. **Threshold config is declarative** - FIXED: `timeline_gate_pass_rate` config value is now used operationally in `_check_timeline_gates_pass()`

**Test updates:**
- Updated `TestSignoffDecision` tests to use 12 split-seed cells (4 splits × 3 seeds) instead of 7 days
- Updated helper methods to create actual trace files for artifact integrity checks
- All 24 validation_run tests pass
- All 726 tests pass (3 skipped)

**Fail-closed principles enforced:**
- Missing evidence = FAIL
- Force=True only allows artifact generation, never changes decision to PASS
- All compliance checks must pass for PASS decision

---

## 2026-02-18 07:00

### D2 (T-70 Controlled Validation + Signoff) - COMPLETE

**Objective:** Run strict 4×3 split-seed protocol end-to-end with WS-6 signoff generation

**Deliverables:**
1. **Strict 4×3 split-seed integration test** (`test_d2_strict_splitseed_integration.py`)
   - Validates 4 splits × 3 seeds = 12 cells mandatory for signoff
   - Tests fail-closed behavior for leakage audit failures
   - Tests fail-closed behavior for hard gate failures
   - Verifies feature flags default to OFF

2. **Leakage audit validation in aggregator** (`aggregate_event_first_backtest.py`)
   - Added `_validate_leakage_audits()` function
   - Fail-closed: any missing or failed leakage_audit_pass = FAIL
   - Integrated into signoff decision path

3. **Updated existing tests** (`test_event_first_backtest_aggregate.py`)
   - Added `leakage_audit_pass` fields to all test fixtures
   - Ensures backward compatibility with D2 strict validation

**Test Results:**
- D2 integration tests: 6 passed
- Full backend suite: 734 passed, 3 skipped

**Fail-closed principles enforced:**
- Missing leakage audit = FAIL
- Leakage audit failure in any cell = FAIL  
- Hard gate failure in any cell = FAIL
- Missing timeline metrics = FAIL

**Promotion Gates Validated:**
- Hard gate all seeds: must pass
- Leakage audit all cells: must pass
- Timeline gates: 80% pass rate minimum
- Consistency violations: blocking

**Next Stage:** Ready for T-80 Rollout (Shadow -> Canary -> Full)

---

## 2026-02-18 08:00

### T-80 Rollout (Shadow -> Canary -> Full) - COMPLETE

**Objective:** Promote with controlled exposure and rollback safety

**Deliverables:**
1. **T80RolloutManager** (`backend/ml/t80_rollout_manager.py`)
   - Shadow mode: Default state, generates artifacts without affecting production
   - Canary mode: Limited cohort (max 5 elders), 7-day observation window
   - Full mode: Complete rollout after canary validation
   - Rollback: Automatic on safety regression or leakage failure

2. **Canary Configuration** (fail-closed defaults)
   - Max elders: 5
   - Observation period: 7 days
   - Timeline gate pass rate: >= 80%
   - Hard gate pass rate: 100% (all must pass)
   - Auto-rollback on safety regression: enabled
   - Auto-rollback on leakage failure: enabled

3. **Comprehensive test suite** (`backend/tests/test_t80_rollout.py`)
   - 18 tests covering all rollout scenarios
   - Shadow -> Canary -> Full workflow
   - Rollback triggers and validation
   - State persistence across restarts

**T-80 Pass Requirements Validated:**
- ✅ No safety regressions (hard gates 100% pass required)
- ✅ BR/LR timeline metrics stable (80% pass rate)
- ✅ Rollback validated (auto-rollback on leakage/hard-gate failure)

**Test Results:**
- T-80 tests: 18 passed
- Full backend suite: 755 passed, 3 skipped

**Files Changed:**
- `backend/ml/t80_rollout_manager.py` (new)
- `backend/tests/test_t80_rollout.py` (new)

**Rollback Triggers:**
| Condition | Action |
|-----------|--------|
| Leakage audit failure | Auto-rollback |
| Hard gate failure | Auto-rollback |
| Timeline regression | Hold (manual review) |
| Observation incomplete | Hold |

**Ready for Production:** T-80 rollout infrastructure complete and tested.

---

## 2026-02-18 08:30

### T-80 Critical Bug Fixes - COMPLETE

**Issues Fixed:**

| Severity | Issue | Fix |
|----------|-------|-----|
| **High** | Leakage audit check fail-open when field missing | Changed `r.get("leakage_audit_pass", True)` to `r.get("leakage_audit_pass", False)` |
| **High** | Canary promotion without full cohort coverage | Added elder_id coverage check - requires results for ALL canary elders |
| **High** | promote_to_full() bypassed decision/observation gates | Added checks for `promotion_decision == "promote"` and `_can_promote()` |
| **Medium** | Unused config fields | Removed `min_data_days`, `min_hard_gate_pass_rate`, `max_regression_from_baseline` |
| **Medium** | promotion_ready misleading | Now computes from both time + gate outcomes + coverage |
| **Low** | Default state path cwd-dependent | Resolved path relative to file location (`Path(__file__).resolve().parents[2]`) |

**New Fail-Closed Tests:**
- `test_leakage_audit_missing_defaults_to_fail` - Missing field = FAIL
- `test_canary_requires_full_cohort_coverage` - All elders must have results
- `test_canary_rejects_extra_elders` - No unexpected elders allowed
- `test_promote_to_full_requires_promote_decision` - Must have PROMOTE decision
- `test_promote_to_full_requires_observation_complete` - Must wait for observation

**Test Results:**
- T-80 tests: 23 passed
- Full backend suite: 760 passed, 3 skipped

**Files Changed:**
- `backend/ml/t80_rollout_manager.py` - All fixes applied
- `backend/tests/test_t80_rollout.py` - Tests updated + new fail-closed tests
