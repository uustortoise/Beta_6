# 1000-Elder POC Roadmap (Beta 6 Architecture)

**Status:** Draft / Strategic Planning
**Date:** 2026-02-08
**Architecture:** Universal Backbone + Transfer Learning

## 1. The Scaling Challenge
*   **Target:** Scale from N=10 to **N=1000 residents**.
*   **Bottleneck (Beta 5):** Supervised learning requires **5 minutes per resident** for retraining.
    *   1000 residents = 83 hours of compute time (unfeasible for daily updates).
*   **Solution (Beta 6):** Transfer Learning with a **Universal Backbone**.

## 2. The Solution: "Professor + Apprentice" Model

### A. Universal Backbone ("The Professor")
*   **Concept:** A single, massive Transformer model pre-trained on N=50+ diverse residents.
*   **Function:** Learns the universal "physics of a home" (e.g., bathroom usage patterns, sleep cycles, sensor noise).
*   **Training Frequency:** Monthly (Heavy compute).
*   **Status:** **Frozen** in production.

### B. Resident Head ("The Apprentice")
*   **Concept:** A lightweight neural network layer specific to *one* resident.
*   **Function:** Adapts the Professor's knowledge to a specific home layout and routine.
*   **Training Frequency:** Daily (Light compute).
*   **Performance:**
    *   **Training Time:** **< 30 seconds** (vs. 5 minutes).
    *   **Total for 1000 Residents:** ~8 hours (manageable within nightly batch window).

## 3. Infrastructure & Technology Stack

### Current Stack (Beta 5.5+)
*   **Database:** **PostgreSQL + TimescaleDB** (Scalable, partitioned time-series storage).
*   **Data Format:** **Parquet** (High-performance columnar storage for raw sensor data).
*   **Compute:** **Dockerized Python 3.12** environment.
*   **Model:** **Hybrid CNN-Transformer** (Acausal, global context awareness).

### Required for 1000-POC (Phase 4)
*   **Job Queue:** Async task distribution (likely internal Python queue or lightweight Redis) to manage 1000 concurrent "Head" training jobs.
*   **Storage Strategy:**
    *   **Hot Data (Last 30 days):** PostgreSQL (TimescaleDB) for instant dashboard access.
    *   **Cold Data (Archive):** S3/Object Storage in Parquet format for model retraining.

## 4. Execution Roadmap

### Phase 1: The Foundation (Current)
- [x] **PostgreSQL Migration:** Move from SQLite to handle concurrent writes.
- [x] **Transformer Prototype:** Validate accuracy of the backbone architecture (Beta 5.5).
- [ ] **Golden Sample Harvesting:** Collect "perfect" labeled data (N=20) to train the first Backbone.

### Phase 2: The Backbone (Next Month)
- [ ] **Train Universal Backbone:** Using N=50 diverse datasets.
- [ ] **Validate Transfer Learning:** Prove that "Backbone + 30s Head Training" achieves >90% accuracy.

### Phase 3: The Scale-Up (1000-POC)
- [ ] **Parallel Processing:** Implement worker pool to train 50-100 "Heads" in parallel.
- [ ] **Automated QC:** "Traffic Light" system to auto-flag failing models (red/yellow/green) without human review.
- [ ] **Shadow Mode:** Run 1000-POC simulation on synthetic/replayed data.

## 5. Comparison: Old vs. New Plan

| Feature | Old Plan (Legacy) | **New Plan (Beta 6)** |
| :--- | :--- | :--- |
| **scaling Strategy** | Brute-force Parallelism | **Algorithmic Efficiency (Transfer Learning)** |
| **Compute Cost** | High (50+ Workers) | **Low (Single Machine / Small Cluster)** |
| **Training Time** | 5 mins / resident | **30 secs / resident** |
| **Core Tech** | Celery + Redis | **Transformer Backbone** |
| **Database** | SQLite Sharding | **PostgreSQL (TimescaleDB)** |

---
*Reference: `Software/Beta_5.5/ml_strategy_discussion.md`*
