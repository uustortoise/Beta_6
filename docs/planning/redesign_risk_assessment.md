# Architecture Redesign: Risk & Timeline Assessment

## The Move: Monolith (Beta 5.5) -> Functional Pipeline (Beta 6)

You asked: **How risky is this move? How much time is needed?**

### Quick Summary
*   **Risk Level:** **High (Operational)** / **Medium (Technical)**
*   **Time Estimate:** **2-3 Weeks (Dedicated)**
*   **Recommendation:** Do **NOT** do this now. Finish the current Hardening Plan (1 week stability). Schedule this redesign for the "Beta 6" cycle.

---

### Detailed Risk Assessment

#### 1. Regression Risk (High)
*   **Why:** We are ripping out the "nervous system" of the application (`Pipeline.py`). Every single feature (Correction Studio, Active Learning, Model Promotion) currently depends on `Pipeline` behaving exactly as it does now.
*   **Danger Zone:**
    *   **Data Consistency:** The new "Feature Store" must produce *bit-identical* data to the old `preprocess_data`. If it differs by even 0.001%, all existing models become invalid.
    *   **Registry State:** Migrating the "stateful JSON registry" to the new "event-sourced registry" requires a perfect migration script. If we lose model history, we lose trust.

#### 2. Integration Risk (High)
*   **Why:** The UI and API endpoints are tightly coupled to the `Pipeline` class.
*   **Example:** The Correction Studio calls `pipeline.retrain_room(...)` and expects a synchronous update. The new architecture might be async (Materialize -> Train -> Promote).
*   **Danger Zone:** We would need to rewrite significant parts of the API layer (`backend/routes/`) to talk to the new components.

#### 3. Technical Risk (Medium)
*   **Why:** The code itself (pure functions) is simpler and easier to write.
*   **Danger Zone:** The complexity shifts to **Orchestration**. Instead of `self.do_step_1(); self.do_step_2()`, we need a way to chain these independent artifacts together (e.g., a lightweight DAG runner). Building a bad orchestrator is a common trap.

---

### Timeline Estimate (Conservative)

Assuming 1 Senior Engineer (You + Me):

#### Week 1: Foundation & Feature Store
*   Define `Manifest` schema (YAML).
*   Extract `feature_engineering.py` into a standalone materializer.
*   **Milestone:** Generate a Parquet file from raw data that matches `df` from `pipeline.py` exactly.

#### Week 2: Trainer & Registry
*   Extract `TrainingPipeline` into a pure stateless function `train(parquet, config) -> model`.
*   Build the new `RegistryObserver` logic.
*   **Milestone:** Train a model from a Parquet file and manually promote it.

#### Week 3: Integration & Migration
*   Wire up the API endpoints to use the new runners.
*   Write migration scripts for existing registry JSONs.
*   **Milestone:** "Correction Studio" works with the new backend.

#### Week 4: Stabilization (The "Hidden" Week)
*   Fixing edge cases (what happens if disk is full during materialization?).
*   Performance tuning (is passing Parquet files slower than in-memory DFs?).
*   **Milestone:** Feature Parity with Beta 5.5.

**Total:** 3-4 Weeks to get back to *where we are today*, but with better architecture.

---

### Strategic Recommendation

**Do NOT STOP the current Hardening Plan.**

1.  **Sunk Cost & Momentum:** We are 80% of the way to a stable Beta 5.5. The current "Hardening" steps (Coverage Gate, Registry Repair) are effectively "patching the roof." It’s raining outside. We need a dry roof *now*.
2.  **Safety First:** The current system, while messy, is *working*. The Hardening Plan makes it *safe*.
3.  **Beta 6 Goal:** Make this Redesign the mandate for Beta 6.
    *   **Benefit:** We can build Beta 6 in parallel (e.g., in a separate module `backend/ml_v2/`) while Beta 5.5 keeps the lights on.
    *   **Plan:** Once Beta 5.5 is hardened and deployed (1 week), we can start the `ml_v2` experiment without pressure.

### Verdict
*   **Risk:** Too high for a "quick fix".
*   **Time:** 1 month detour.
*   **Decision:** Execute the **Beta 5.5 Hardening Plan** immediately (1 week). Schedule **Redesign** for next quart/milestone.
