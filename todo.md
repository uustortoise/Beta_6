# Beta 5.5 Project TODO 📋

> [!IMPORTANT]
> **Beta 5.5 Mandate (1000 POC Readiness)**
> 1. **Solid Foundation**: No temporary fixes or debug patches. Production-ready solutions for scaling to 1000 elders.
> 2. **Preserve Logic**: Do not remove or change existing Beta 5 features without explicit approval.
> 3. **Additive Development**: All new work must be strictly additive extensions.

---

## ✅ Completed (Feb 2026)

### Core Infrastructure
- [x] PostgreSQL Migration (162k+ rows, ID auto-increment fix)
- [x] Path Standardization & Config Validation (Pydantic)
- [x] Standardized Error Handling (`ml/exceptions.py`, API `withErrorHandling`)
- [x] UnifiedPipeline Refactor (ModelRegistry, TrainingPipeline, PredictionPipeline)
- [x] Mac Studio Deployment (setup scripts, PostgreSQL connection)

### ML Pipeline (Stabilization Phase)
- [x] **Smart Downsampling**: Confirmed logic for thinning "Unoccupied" windows (1 per 40s).
- [x] **Walk-Forward Validation**: "Model Insights" tab with holdout F1/Accuracy metrics.
- [x] **Operator Ergonomics (Phase 5)**:
    - [x] Fold Feasibility Check (Data vs Requirement).
    - [x] Auto-Tuner Suggestions ("One-Click Apply").
    - [x] Plain-Language Promotion Cards ("Score 48% < 55%").
- [x] CNN-Transformer Backbone (`transformer_backbone.py`)
- [x] ML Audit Remediation: Class weights, comprehensive metrics, early stopping
- [x] Golden Sample Harvesting (`harvest_gold_samples.py`)
- [x] Sleep Analyzer Consolidation (`SleepAnalyzer` class)
- [x] Unit Testing (15+ test cases for Registry, Training, Prediction)

### API & Frontend
- [x] API Review: Transaction safety, error standardization, PostgreSQL dialect
- [x] RAG Pipeline (ChromaDB + 160+ clinical abstracts)
- [x] Frontend Polish: Global error handling, toast notifications, API client
- [x] Chatbot API Contract Standardization (`snake_case` + frontend mapper)

### Health Advisory Chatbot Safety Rebuild
- [x] SAFE-001: Deterministic SafetyGateway module
- [x] SAFE-002: Emergency/urgent hard bypass before RAG/LLM
- [x] SAFE-003: Deterministic escalation templates (EN/ZH)
- [x] SAFE-004: Structured safety telemetry logging
- [x] SAFE-005: Engine/API safety bypass tests
- [x] POL-001: ActionPlan schema + policy engine integration
- [x] POL-002: Fall policy rules (high/moderate/night safety)
- [x] POL-003: Medication policy rules (major interactions, polypharmacy, ACB)
- [x] POL-004: Cognitive + sleep deterministic policy rules
- [x] POL-005: Externalized policy manifest with changelog metadata
- [x] VAL-001: Fail-closed citation validation; removed permissive external patterns
- [x] VAL-002: Citation ID registry with strict source routing

### Performance Phase 1 (Feb 7)
- [x] SELECT * → Explicit columns (4 queries)
- [x] Index migration script (`migrate_add_indexes.sql`)
- [x] Pool monitoring (keepAlive, metrics export)
- [x] Query timeout protection (`queryWithTimeout`)

### Bug Fixes & Audits
- [x] Bug Audit Remediation (6 critical/high severity issues)
- [x] ML Audit Remediation (6 critical/high severity issues)
- [x] API Polish (boolean syntax, withErrorHandling overloads)

### Production ML Hardening (18-Item Execution Plan) - COMPLETE ✅
- [x] **Week 1**: Train-Split Scaling, Coverage Contract, Policy Profiles, Statistical Validity, Walk-Forward Robustness (32 tests)
- [x] **Week 2**: Post-Gap Retention, Sequence-Label Alignment, Duplicate Resolution (optimized), Class Coverage (24 tests)
- [x] **Week 3**: Calibration Semantics, Rejection Artifacts, Reproducibility Report (25 tests)
- [x] **Week 4**: Room Calibration Diagnostics, Data Quality Contract (21 tests)
- [x] **Week 5**: Registry Validator, Policy Presets, Pilot Override Rollback, Transformer Head A/B (40 tests)
- [x] **Performance Fix**: DuplicateTimestampResolver optimized (O(N) → O(duplicates only), 400x speedup)
- **Total**: 142 tests passing, all 18 items complete, exit criteria satisfied

---

## 🔄 In Progress

### Performance Phase 2 (Deferred to next sprint)
- [ ] Implement pagination for alerts and timeline
- [ ] Add LRU caching layer with TTL strategy
- [ ] Materialized view for available dates (if verified bottleneck)

### Health Advisory Chatbot (Current Sprint)
- [ ] VAL-003: Claim extraction and claim-to-evidence linkage enforcement
- [ ] VAL-004: Response policy for unverified claims/citations (deterministic fallback wording)
- [ ] UX-001: Replace hardcoded risk/evidence UI badges with API-backed values
- [ ] API-003: Surface full urgency assessment object in API response contract

---

## 📋 Backlog (Beta 6 Roadmap)

### Golden Sample Integration
- [ ] **Data Archive N=20**: Accumulate 20+ residents with verified Golden Samples
- [ ] **Import Script**: Bulk importer for Golden Samples on Mac Studio
- [ ] **Universal Backbone Training**: Primary training run on combined dataset
- [ ] **Performance Benchmarking**: Compare Backbone vs. Resident-specific LSTMs

### Intelligence Roadmap
- [ ] Clinical CMS Transition: Move behavioral rules to PostgreSQL
- [ ] Clinical Rule Editor UI: Admin interface for gerontologists
- [ ] RAG Feedback Loop: "Helpful/Not Helpful" ratings
- [ ] Expanded Disease Portfolio: Scale to 20+ conditions
- [ ] **Local LLM for Topic Extraction** (Research Phase):
  - **Problem**: Rule-based topic extraction misses natural language variations (e.g., "i fell" vs "fall")
  - **Options**: 
    - Option A: Continue improving keyword rules (current)
    - Option B: Local lightweight LLM (Phi-3, Mistral 7B, Llama 3.2) for semantic understanding
    - Option C: Hybrid approach (rules for speed + LLM fallback for complex queries)
  - **Considerations**: Latency (100-500ms), memory (2-8GB), determinism vs flexibility trade-off
  - **Decision**: Monitor rule-based coverage via logs; consider LLM fallback when coverage plateaus

### Infrastructure Scale
- [ ] S3/Cloud Sync: Automate Parquet sync to Alibaba Cloud OSS
- [ ] Retire SQLite: Switch read path to PostgreSQL only
- [ ] Performance Monitoring: Track latency/memory/throughput per elder
- [ ] **Timeout Enforcement**: Migrate all queries to `timedQuery` with explicit timeouts (remove pool default reliance)

### ML Evolution
- [ ] Confidence-Based CRF: Replace rule-based segment consolidation
- [ ] Multi-Label Classification: Foreground vs. Background activities
- [ ] Sleep Boundary Refinement: Train ML for morning transition periods
- [ ] ⚠️ Parallel ML Prediction: **DEFERRED** (TensorFlow Metal stability issues)

---

## 🚫 Deferred / On Hold

### Two-Stage Correction Workflow
- Draft corrections (`is_draft=1`) → Finalize button commits
- **Status**: Deferred per user request (Jan 26, 2026)

### Event-Driven Pipeline
- Correction → Event → Segment Regen → Cache Invalidation
- **Status**: Architectural improvement, low priority

---

## 📊 Targets

| Metric | Target | Status |
|--------|--------|--------|
| API Response Time (p95) | < 200ms | ✅ (Phase 1 optimizations applied) |
| Dashboard Load Time | < 2s | ✅ |
| DB Query Time (adl_history) | < 100ms | ✅ (Indexes added) |
| ML Prediction (per room) | < 500ms | ✅ |
| PostgreSQL Rows | 162k+ | ✅ |

---

*Updated: Feb 16, 2026 (Production ML Hardening Complete: All 18 items implemented, 142 tests passing. Documentation updated.)*
