# 🧠 ML Strategy Discussion: Current Beta 5 & Future Beta 6 Architecture

This document outlines the evolutionary pathways for our Machine Learning foundation, with **Beta 5 (current supervised system)** and **Beta 6 (scalable backbone architecture)**, focusing on balancing **labeling efficiency** against **system accuracy** and **computational cost**.

---

## 1. Supervised Learning (Current Beta 5)
The AI learns from explicit "Ground Truth" provided by humans.

*   **Workflow:** Human corrects labels in Correction Studio → System retrains entire model (5 minutes/resident).
*   **Efficiency:** **Low.** Requires manual entry for multiple days; training time increases as history grows.
*   **Accuracy:** **High.** Model reflects specific human nuances and environmental physics.
*   **Cost:** Low engineering complexity; high user "time tax."
*   **Scalability Limit:** ~10 residents/practitioner due to 5-minute training bottleneck.
*   **Current Status:** ✅ **PRODUCTION** - All features functional (Correction Studio, ICOPE, Multi-Room Logic).

## 2. Transfer Learning (Beta 6 Universal Backbone) - **SCALABILITY BREAKTHROUGH**
The **Backbone + Head** architecture that enables scaling to 1000+ residents:

### **Core Concept: "Professor + Apprentice" Model**
*   **Backbone (90% - "Professor"):** Learns universal "Physics of a Home" from N=50 diverse residents. **Frozen** after training.
*   **Head (10% - "Apprentice"):** Learns resident-specific habits. Trained in **30 seconds** instead of 5 minutes.

### **Key Technical Insights (Updated)**
1. **Single Universal Backbone:** One backbone handles ALL home variations:
   - ✅ Different room layouts (2-room to 5+ room apartments)
   - ✅ Household sizes (singleton vs. couple)
   - ✅ Sensor distances (motion patterns encode room size naturally)
   - ⚠️ Pets/unusual lifestyles require sufficient examples in training data

2. **Training Time Transformation:**
   ```
   Beta 5: 5 minutes × 1000 residents = 83 hours of training
   Beta 6: 30 seconds × 1000 residents = 8.3 hours (10x faster)
   ```

3. **Data Diversity & Class Balance Requirements:** Backbone robustness depends on training data variety:
   - **Community-Wide Balance:** Class balance (e.g., getting N=50 shower samples) is prioritized across the **entire dataset**, not per individual resident.
   - **Missing Activities:** It is acceptable if some residents are missing specific activities (e.g. no shower data). The backbone learns those patterns from other residents in the community.
   - **Diversity:** Include different home layouts, household types, and routines to ensure universal physics.

### **Strategic Milestones**
| N (Residents) | Accuracy | Capabilities | Status |
| :--- | :--- | :--- | :--- |
| **N=20** | ~82-85% | Good standard physics | Starting point for Beta 6 deployment |
| **N=50** | **~89-93%** | **Statistically robust** | Production freeze target |
| **N=1000** | >95% | Universal black-box reliability | 1000-POC goal |

### **Implementation Progress**
- ✅ **Backbone Module:** `backbone.py` with Attention, BackboneModel, builder functions
- ✅ **Data Harvester:** `scripts/harvest_gold_samples.py` - CLI for extracting Golden Samples with Traffic Light filtering
- ✅ **Beta 6 Trainer:** `ml/beta6/beta6_trainer.py` - Training pipeline skeleton for Universal Backbone
- ⚠️ **Integration:** Pending full Transformer integration (currently MLP placeholder)
- ❌ **Validation:** Shadow mode validation framework needed

> [!TIP]
> **Execution Steps**: See the [Golden Sample Harvesting Guide](file:///Users/dicksonng/DT/Development/Beta_5.5/user's%20manual/golden_sample_harvesting.md) for detailed executable commands.


## 3. Supporting Strategies for Labeling Efficiency

### **Active Learning (Uncertainty Sampling)**
*   **Concept:** AI only asks you to label what it's "confused" about (40-60% confidence).
*   **Benefit:** Reduces manual review by ~90%.
*   **Status:** Planned for Beta 6.x

### **Heuristic Pre-Labeling (Expert Rules)**
*   **Concept:** Hard-coded rules (Humidity > 80% = Shower) auto-approve obvious labels.
*   **Benefit:** Focuses human effort on complex cases.
*   **Status:** Quick win, can be added anytime

### **Incremental Fine-Tuning**
*   **Concept:** "Nudge" weights with new corrections instead of full retraining.
*   **Benefit:** "Save & Train" becomes near-instantaneous.
*   **Status:** Planned post-backbone integration

## 4. Deployment Strategy: "Shadow Mode" Transition

To move from Beta 5 → Beta 6 without service interruption:

1.  **Harvesting Phase:** Continue Beta 5, collect Golden Labels from all residents.
2.  **Shadow Validation:** Train backbone in background, compare vs. Beta 5 daily.
3.  **Switchover:** When backbone matches Beta 5 accuracy (N=30), onboard new residents to backbone.
4.  **Feature Flags:** Controlled rollout using `ENABLE_BACKBONE_MODE` flag.

## 📊 Updated Strategy Matrix

| Strategy | Efficiency | Speed to Train | Reliability | Engineering Effort | Scalability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Beta 5 (Current)** | Low | Slow (5 min) | High | Low | ❌ ~10 residents |
| **Beta 6 (Backbone)** | High | **Fast (30 sec)** | High | Medium | ✅ **1000+ residents** |
| **Active Learning** | **High** | Medium | High | **Low** | ✅ 200+ residents |
| **Combined Approach**| **Very High**| **Fast** | **High** | High | ✅ **1000+ residents** |

## 🚀 Beta 6 Development Roadmap

### **Phase 1: Architecture (COMPLETE)**
- ✅ Data Access Layer (DAL) with sharding support
- ✅ EnvironmentConfig (DEV/SHADOW/PROD)
- ✅ Feature Flags for A/B testing

### **Phase 2: Backbone Core (IN PROGRESS)**
- ✅ `backbone.py` module built
- ✅ `harvest_gold_samples.py` ready
- ⚠️ **Next:** Pipeline integration, N=20 dataset generation

### **Phase 3: Integration & Validation**
- ✅ Integrate backbone into `pipeline.py` (Completed Jan 26 via Modularization)
- ❌ Generate initial universal dataset
- ❌ Shadow validation framework
- ❌ Feature flag rollout

### **Phase 4: Scalability & Production**
- ❌ Async job queue for 1000-resident scale
- ❌ Performance optimization (caching, pagination)
- ❌ Security audit, data lineage tracking

## 🎯 Current Recommendations

### **Immediate (Next Week)**
1. **Integrate backbone into pipeline** - Critical path for Beta 6
2. **Run harvester** to create N=20 universal dataset
3. **Train initial backbone** and validate accuracy
4. **Deploy via feature flags** to beta users

### **Medium-term (Next Month)**
1. **Reach N=50** dataset for production-ready backbone
2. **Implement Active Learning** for 90% labeling efficiency
3. **Performance optimization** for 1000-resident scale

### **Long-term (1000-POC Goal)**
1. **Combine Backbone + Active Learning** for maximum efficiency
2. **Full production hardening** (security, audit trails, monitoring)
3. **Continuous learning** system that improves with each correction

**Bottom Line:** Beta 5 provides solid foundation. Beta 6's backbone architecture is the **scalability breakthrough** needed for 1000-POC. Focus now on integration to realize 10x training speed improvement.

---

## 🧪 Beta 5.5: Hybrid CNN-Transformer Prototype (NEW)

Beta 5.5 is an experimental branch to validate the **Hybrid CNN-Transformer** architecture for **offline/batch** daily analysis scenarios where real-time alerts are **not required**.

### **Rationale: Why Transformers for Offline ADL?**
Since we are not targeting real-time fall detection or immediate alerts, the "causality" constraint of LSTMs/TCNs (which can only look at past data) is relaxed. A **Transformer Encoder** can use **global context** (looking at the entire day's data at once), providing:

1.  **Acausal Processing**: The model can "see" both past and future relative to any timestamp, improving disambiguation.
2.  **Superior Routine Capture**: Global Self-Attention naturally learns long-range dependencies (e.g., correlating morning routines with evening patterns).
3.  **Parallel Training**: Significantly faster than sequential LSTMs.

| Feature | CNN-LSTM (Beta 5) | Hybrid CNN-Transformer (Beta 5.5) |
| :--- | :--- | :--- |
| Context | Local/Sequential | **Global** |
| Causality | Causal (Past only) | **Acausal (Full day)** |
| Training | Sequential $O(T)$ | **Parallel $O(1)$** |
| Accuracy | High | **Very High** |
| Min N for Value| ~10 | ~50 |

### **Minimum N Requirements for Transformer Value**
Due to the lack of inductive biases versus CNNs/RNNs, Transformers require more diverse data:

| Phase | Target N | Strategy | Expected Outcome |
| :--- | :--- | :--- | :--- |
| **Experimental** | N=10-20 | Hybrid CNN-Transformer | Matches CNN-LSTM accuracy, reduces "flicker". |
| **Production Ready**| **N=50** | **Pre-trained Backbone** | **Scalability Breakthrough.** Training <30s. |
| **Statistical Gold** | N=100+ | Full Sensor Transformer | >95% accuracy. |

### **Risk Mitigation**
1.  **Data Scarcity**: Use **Hybrid CNN-Transformer**. CNNs handle low-level sensor physics; Transformer handles high-level routine logic.
2.  **Complexity**: Implement **Attention Visualization** in Shadow Lab for debugging.
3.  **Hardware**: Use quantization (INT8) for CPU deployment if needed.
4.  **Interpretability**: Map attention weights to ICOPE domains for narrative explanations.

### **Beta 5.5 Implementation Status (Updated Jan 27, 2026)**
- ✅ **Environment Setup:** Cloned from Beta 5 with independent ports (3002/8503) and database.
- ✅ **transformer_backbone.py:** CNN embedding + Multi-Head Self-Attention + Positional Encoding.
- ✅ **Preprocessing Improvements:**
    - Denoising threshold: 3.0 → 4.0 (less aggressive for attention-based models)
    - Gap detection: `detect_gaps()` marks discontinuities >5min with segment IDs
    - GAP tokens: `insert_gap_tokens()` inserts explicit boundary markers
    - Positional encoding: Sinusoidal, Relative (ALiBi), and Learnable options
- ✅ **Pipeline Modularization:** Refactored into `ModelRegistry`, `TrainingPipeline`, `PredictionPipeline`.
- ✅ **System Hardening:** 
    - 5-Point Data Validation Gatekeeper.
    - Race condition fixes for background workers.
    - Robust relative import system (no more sys.path hacks).
- ❌ **Benchmarking:** Compare accuracy/throughput vs Beta 5 CNN-LSTM

### **🔑 Technical Design: 10s Temporal Resolution**
To ensure stability for the Transformer's attention mechanism and scalability for the 1000-POC, the system uses a fixed **10-second floor** for all sensor data.
- **Rationale:** Balanced against sensor debounce jitter and computational cost.
- **Full Analysis:** See [docs/research/rationale_10s_resolution.md](file:///Users/dickson/DT/DT_development/Development/Beta_5.5/docs/research/rationale_10s_resolution.md).

---

## 🔄 Beta 6 Integration: CNN-Transformer Learnings (NEW)

The following changes from Beta 5.5 will be integrated into Beta 6 after validation:

### **Preprocessing Changes for Universal Backbone**
| Component | Beta 5 | Beta 6 (Post-Validation) |
|-----------|--------|--------------------------|
| **Denoising** | Hampel σ=3.0 | Hampel σ=4.0 (Transformer-compatible) |
| **Gap Handling** | Forward-fill only | GAP token insertion + segment IDs |
| **Positional Encoding** | N/A | Relative (ALiBi) for variable sequences |

### **Architecture Evolution**
```
Beta 5:   Input → CNN → LSTM → Attention → Output
Beta 6:   Input → CNN Embedding → Positional Encoding → Transformer Encoder → Output
                                                       ↳ Multi-Head Self-Attention
```

### **Labeling Practice Implications**
> [!IMPORTANT]
> The Transformer's acausal processing changes how corrections influence predictions:

1. **Segment-Level Corrections**: Since the model sees full-day context, correcting a single timestamp now influences predictions both **before and after** that point (unlike LSTM which only affects future predictions).

2. **GAP Token Awareness**: When labeling data with significant gaps (>5min), the Transformer treats each segment independently. Corrections in one segment do not "bleed" into adjacent segments.

3. **Minimum Correction Density**: For Transformer training to be effective, corrections should cover at least **3-5 examples per activity class per room**. Sparse corrections (<3 per class) may cause the attention to overfit.

4. **Batch Correction Preferred**: The Transformer benefits from batch corrections (multiple rooms/timestamps) applied together before retraining, rather than incremental single-point edits.

---

## 5. Optimal Training Configuration

For most daily analysis scenarios, the following configuration provides the best balance between time and accuracy:

- **Target Accuracy**: 85% (Optimal for routine detection without overfitting).
- **Default Epochs**: **5** (Reduced from 10). 
    *   *Insight*: Most models achieve ~80-85% accuracy within the first 3-5 epochs. Further training often leads to diminishing returns and increases the risk of overfitting.
- **Early Stopping**: Enabled (Patience: 2).
    *   *Benefit*: Stops training automatically if accuracy plateaus, saving significant time during large batch corrections.

## 6. Defining "Good Data"

Accuracy is driven by data quality ("Garbage In, Garbage Out"). High-quality data ensures the model learns real patterns.

### Key Quality Indicators:
1.  **Label Accuracy**: Manual corrections must be precise. Incorrectly labeling a "Nap" as "Shower" confuses the model's understanding of sensor patterns.
2.  **Signal-to-Noise Ratio**: Clean data with minimal sensor "spikes." The Hampel Filter is used to maintain this quality.
3.  **Class Separability**: Activities should have distinct sensor signatures. If "Cooking" and "Washing Dishes" look identical to sensors, accuracy will naturally be lower.
4.  **Data Continuity**: Avoid large gaps (>1 min) in sensor reporting. Continuous data flow allows the Transformer architecture to capture long-range habits effectively.
