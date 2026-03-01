# Beta Development History 📜

## Jan 29, 2026: Hybrid Transformer Prototype & PostgreSQL Migration
The platform reached a major milestone with a new high-performance ML architecture and a scalable database backbone.

### 🚀 Major Improvements
- **Transformer Backbone**: Implemented Hybrid CNN-Transformer model for offline/batch daily analysis, utilizing multi-head self-attention for better pattern disambiguation.
- **PostgreSQL + TimescaleDB Migration**:
    - Transitioned from SQLite-only to **Hybrid Dual-Write** architecture (SQLite + TimescaleDB).
    - Successfully backfilled **100%** of historical data (85k+ rows) with verified integrity.
    - Implemented **TimescaleDB Hypertables** for optimized time-series performance.
- **Production Readiness Audit**:
    - Passed three rounds of independent audits.
    - Implemented structured logging for monitoring (verified ~6ms latency overhead).
    - Created robust **ROLLBACK.md** and **ROLLOUT_PLAN.md** procedures.
    - Verified **Fail-Safe Recovery**: Application gracefully survives database downtime.
- **Automated Environment Management**:
    - Enhanced `./start.sh` with Docker automation (auto-detects and boots DB container).
    - Updated `reset_environment.py` for full multi-db data clearing.

### 🐛 Critical Bug Fixes
- **Verification False Negatives**: Fixed timestamp rounding bugs in `verify_dual_write.py` using tolerant matching (±500ms).
- **Dual-Write Logic Race Conditions**: Refactored `database.py` to support correct `RETURNING id` handling for PostgreSQL.
- **Fail-Safe Startup**: Patched `database.py` to prevent application crashes when PostgreSQL is unreachable.

---
## Jan 2, 2026: Production Hardening & Parquet Migration
The platform was upgraded to handle production-scale data and fix critical UI sync issues.

### 🚀 Major Improvements
- **Data Format Revolution**: Migrated archive system from `.xlsx` to `.parquet`. Reduced file sizes by ~90% and improved ML load speeds significantly.
- **Unified Data Loader**: Implemented `backend/utils/data_loader.py` to seamlessly handle legacy Excel and new Parquet formats.
- **Correction Studio UI Enhancements**:
    - Added **Bulk Labeling** (Apply to Range) for faster data curation.
    - Added **Retrospective Training** toggle to retrain models on full historical datasets.
    - Improved **Pre-population**: Predictions are now merged into the labeling editor using `merge_asof` for fuzzy timestamp matching.

### 🐛 Critical Bug Fixes
- **Label Sync Case-Sensitivity**: Fixed a recurring bug where labels saved as "Bathroom" (Title Case) failed to match "bathroom" (lowercase) in the database.
- **Segment Regeneration**: Fixed logic to properly `DELETE` and `REGENERATE` timeline segments, preventing overlapping or fragmented UI blocks.
- **Streamlit Cache Management**: Reduced cache TTL and added versioning to ensure UI reflects real-time database and code changes.

### 🏗️ Technical Debt & Refactoring
- Standardized ML pipeline return signatures for robustness.
- Improved error handling in `process_data.py` for training failures.

---
## Dec 2025: Beta 5 Intelligence Launch
- Initial implementation of Trajectory Tracking.
- Household Context Classification (Empty/Home/Guest).
- Household Analyzer logic for cross-room conflict resolution.
