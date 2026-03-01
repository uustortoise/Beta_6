# Segment-First Greenfield ML Architecture: Detailed Execution Plan

**Version 3.0** — Expanded with Ops UI & Extensibility.  
**Objective**: Build a production-ready occupancy timeline system reliable with **3-5 days** of data.  
**Core Strategy**: 3-Tier Progressive Architecture (Rule Engine → Segment ML → Neural).

---

## 1. Architecture & Data Structures

### 1.1 Core Data Structures (`ml/segment_types.py`)

Create this file first. All modules will depend on it.

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import pandas as pd

@dataclass
class Segment:
    room: str
    start_idx: int
    end_idx: int
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    features: Dict[str, float] = field(default_factory=dict)
    
    # Tier 1 outputs
    rule_label: str = "unknown"
    rule_confidence: float = 0.0
    
    # Tier 2 outputs
    ml_label: str = "unknown"
    ml_probs: Dict[str, float] = field(default_factory=dict)
    
    # Final Arbitrated output
    final_label: str = "unknown"
    source: str = "unknown"  # "rule", "ml", "arbitration"

@dataclass
class SensorBaselines:
    room: str
    light_off_threshold: float  # e.g., 5.0
    light_on_threshold: float   # e.g., 20.0
    background_co2: float       # e.g., 450.0
    motion_noise_floor: float   # e.g., 0.01
```

---

## 2. Phase 1: Tier 1 Rule Engine (Week 1-2)

**Goal**: Produce a high-precision timeline from Day 0 without training.

### 2.1 Auto-Calibration (`ml/tier1_rules.py`)

**Functionality**: Inspect the first 24h of data to set sensor thresholds. This solves the "Samuel CO2" problem.

```python
def auto_calibrate_sensor_baselines(df: pd.DataFrame, room: str) -> SensorBaselines:
    # 1. Light: Find the "dark" mode
    light = df['light'].fillna(0)
    light_off = float(light.quantile(0.10)) + 2.0  # 10th percentile + buffer
    
    # 2. CO2: Find the "ventilated" mode
    co2 = df['co2'].fillna(400)
    bg_co2 = float(co2.quantile(0.05))
    
    return SensorBaselines(
        room=room,
        light_off_threshold=light_off,
        light_on_threshold=max(light_off * 3, 20.0),
        background_co2=bg_co2,
        motion_noise_floor=0.01
    )
```

### 2.2 Change-Point Detection (`ml/change_point.py`)

**Functionality**: Identify *candidate* segment boundaries from raw sensors.

**Algorithm**:
1.  **Triggers (Start)**: `motion > 0` after silence, `light` jumps, `door` open.
2.  **Triggers (End)**: `motion = 0` for > `X` min, `light` switch.

**Implementation Task**:
```python
def propose_segments_from_sensors(
    df: pd.DataFrame, 
    baselines: SensorBaselines
) -> List[Segment]:
    # ... implementation of triggers ...
    # Return list of Segments with start/end indices
```

### 2.3 Rule Logic (Extensible Design)

Move hardcoded logic to `config/tier1_rules.yaml`.

```yaml
bedroom:
  sleep:
    conditions:
      - sensor: light
        operator: "<"
        ref: light_off_threshold
      - sensor: motion_30m_mean
        operator: "<"
        value: 0.5
    time_window: [21, 9]
    confidence: 0.9
  napping:
    conditions:
      - sensor: motion_30m_mean
        operator: "<"
        value: 0.5
    time_window: [12, 17]
    confidence: 0.6
```

---

## 3. Phase 2: Tier 2 Segment ML (Week 3-4)

**Goal**: Train a model to correct the Tier 1 timeline using user feedback.

### 3.1 Feature Extraction (`ml/segment_features.py`)

**Features**: `duration_log`, `motion_density`, `motion_intensity`, `light_mean`, `co2_slope`, `time_cos/sin`, `preceding_label`.

### 3.2 Segment Classifier (`ml/segment_classifier.py`)

**Model**: `HistGradientBoostingClassifier`. Weights samples by `np.log(duration)`.

### 3.3 Arbitration (Extensible Priority)

Move to `config/arbitration.yaml`.

```yaml
priorities:
  sleep: 90
  shower: 85
  exercise: 70
  kitchen_normal_use: 60
  livingroom_normal_use: 50
  unoccupied: 0
```

---

## 4. Phase 3: Ops Automation & UI (Week 5)

**Goal**: Enable non-ML Ops team to manage the daily workflow.

### 4.1 Streamlit App (`backend/segment_ops_lab.py`)

Structure the app into 3 tabs matching the daily workflow:

#### Tab 1: "Inbox (Day 1-3)"
**Purpose**: Review Tier 1 output for new users and create initial training data.
*   **Input**: Select `Elder ID`.
*   **View**: Plot sensor streams (Light, Motion) + Tier 1 Timeline.
*   **Action**: "Accept Timeline" vs "Edit Segment" (Drag & Drop or form).
*   **Output**: Saves corrected timeline as `ground_truth.json`.

#### Tab 2: "Training & Calibration"
**Purpose**: Train Tier 2 model once enough data exists.
*   **View**: Data sufficiency check ("User has 3 days labeled").
*   **Action**: "Train Segment Model".
*   **Result**: Shows validation metrics (IoU, Accuracy) immediately.
*   **Config**: Sliders to adjust Tier 1 thresholds if baselines are wrong.

#### Tab 3: "Quality Dashboard"
**Purpose**: Monitor ongoing performance.
*   **Metrics**: Daily fragmentation count, Sleep recall, Missing data alerts.
*   **Alerts**: "Samuel's CO2 sensor is drifting (+500ppm vs baseline)".

### 4.2 Extensibility Design (Ops-Managed)

The Ops team manages these files in the repo (no code changes):
1.  `config/tier1_rules.yaml`: Add new rules (e.g., `exercise`).
2.  `config/arbitration.yaml`: Set priority for new labels.
3.  `config/gates.yaml`: Define pass criteria for new labels.

---

## 5. Phase 4: Integration & Strict Evaluation (Week 6)

### 5.1 CLI Integration (`run_segment_backtest.py`)

Create a **new** entry point script to avoid breaking the existing pipeline.

```python
def main():
    # 1. Auto-Calibrate (Day 0)
    baselines = auto_calibrate(data)
    
    # 2. Tier 1 (Real-time)
    segments = propose_segments(data, baselines)
    segments = apply_rules(segments, baselines)
    
    # 3. Tier 2 (Day 3+)
    features = extract_features(segments, data)
    model = train_segment_classifier(features, labels)
    final_segments = predict_segments(model, features)
    
    # 4. Evaluate
    iou_scores = evaluate_segment_iou(final_segments, gt_labels)
```

### 5.2 Strict Evaluation
Run on **WS-6 strict splits**. Pass Requirements:
1.  **Segment IoU > 0.70** (All rooms).
2.  **Fragmentation < 5/day**.
3.  **Kitchen Duration Error < 60 min**.

---

## 6. Daily Ops Workflow (SOP)

1.  **Day 0**: Onboard user. Check Tab 1 to confirm auto-calibration looks sane.
2.  **Day 1-3**: Ops reviews Day 1 timeline in Tab 1. Corrects obvious errors (e.g., missed sleep start). Saves as GT.
3.  **Day 4**: Go to Tab 2. Click "Broad-Phase Train". System builds Tier 2 model.
4.  **Day 5+**: System runs Tier 2 auto-pilot. Ops only checks Tab 3 alerts.

---

## 7. Development Schedule

| Week | Team A (Tier 1 + Ops UI) | Team B (Tier 2 ML) |
|---|---|---|
| **W1** | `ml/tier1_rules.py` + Calibration | `ml/change_point.py` + Tests |
| **W2** | `backend/segment_ops_lab.py` (UI) | `ml/segment_features.py` |
| **W3** | Ops Config Parsers (YAML) | `ml/segment_classifier.py` |
| **W4** | `ml/arbitration.py` | `run_segment_backtest.py` |
| **W5** | End-to-End Testing | Strict Matrix Eval |

This plan is ready to execute. Start with **Phase 1** immediately.
