# Lane B Summary: Event Compiler + KPI/Gates + Home-Empty Fusion

**Status:** ✅ COMPLETE (Feb 16, 2026)  
**Test Count:** 106 new tests (463 total)  
**Modules:** 6 new Python modules

---

## Overview

Lane B implements the event semantics layer for the Event-First CNN-Transformer architecture. It converts raw window-level predictions into canonical event episodes, computes care-relevant KPIs, enforces tiered quality gates, and provides multi-room fusion for reliable home-empty detection.

---

## PR-B1: Event Compiler + Decoder + Derived Events

### Files
- `backend/ml/event_compiler.py` (487 lines)
- `backend/ml/event_decoder.py` (351 lines)
- `backend/ml/derived_events.py` (516 lines)
- `backend/tests/test_event_compiler.py` (399 lines)
- `backend/tests/test_event_decoder.py` (346 lines)
- `backend/tests/test_derived_events.py` (414 lines)

### Key Classes

#### EpisodeCompiler
Groups consecutive window predictions into canonical episodes with:
- **Gap-aware splitting:** Merge episodes with gaps <= threshold
- **Min-duration filtering:** Remove noise (default: 30s minimum)
- **Hysteresis smoothing:** Reduce flip-flop transitions
- **Multi-room support:** Household-level episode compilation

```python
compiler = EpisodeCompiler(EpisodeCompilerConfig(
    min_duration_seconds=30.0,
    merge_gap_seconds=30.0,
    use_hysteresis=True,
))
episodes = compiler.compile(predictions)  # List[Episode]
```

#### EventDecoder
Converts raw model probabilities to discrete predictions:
- **Head A/B fusion:** Combines occupancy + activity probabilities
- **Hysteresis state machine:** Stable transitions with configurable thresholds
- **Temporal smoothing:** Noise reduction via windowed averaging
- **Unknown fallback:** Low-confidence predictions marked as "unknown"

```python
decoder = EventDecoder(DecoderConfig(
    occupancy_on_threshold=0.60,
    occupancy_off_threshold=0.40,
    use_hysteresis=True,
))
predictions = decoder.decode(occupancy_probs, activity_probs, timestamps, room_name)
```

#### DerivedEventCalculator
Extracts care-relevant KPIs from episodes:
- **Sleep metrics:** Duration, efficiency, interruptions, quality score
- **Bathroom metrics:** Visits, shower detection, duration
- **Kitchen metrics:** Cooking/eating counts, meal detection
- **Out-time:** Home-empty periods, duration statistics
- **Weekly aggregation:** Trend analysis

```python
extractor = CareKPIExtractor()
kpis = extractor.extract_day_kpis(room_episodes, target_date)
```

### Test Coverage: 56 tests

---

## PR-B2: Event KPI + Gate Layer

### Files
- `backend/ml/event_gates.py` (481 lines)
- `backend/ml/event_kpi.py` (407 lines)
- `backend/tests/test_event_gates.py` (396 lines)

### Key Classes

#### EventGateChecker
Tiered gate checking for event-level metrics:

| Tier | Events | Min Recall | Critical |
|------|--------|------------|----------|
| Tier-1 | shower_day, home_empty_false_empty_rate, sleep_duration | 0.50 | Yes |
| Tier-2 | bathroom_use, kitchen_use, livingroom_active | 0.35 | No |
| Tier-3 | out_time | 0.20 | No |

**Hard Safety Gates:**
- Home-empty precision >= 0.95
- False-empty rate <= 0.05
- Global unknown rate <= 0.15
- Per-room unknown rate <= 0.20

**Collapse Detection:**
- Recall <= 0.02 with support >= 30 triggers critical failure

```python
checker = EventGateChecker()
report = checker.check_all_gates(metrics, target_date)
if report.is_promotable:
    # Model eligible for promotion
    pass
```

#### EventKPICalculator
Computes event-level KPIs for gate checking:
- Home-empty precision/recall
- Per-event recall/precision/F1
- Unknown rate calculations
- Care KPI extraction

### Test Coverage: 28 tests

---

## PR-B3: Home-Empty Fusion + Household Gate

### Files
- `backend/ml/home_empty_fusion.py` (576 lines)
- `backend/tests/test_home_empty_fusion.py` (509 lines)

### Key Classes

#### HomeEmptyFusion
Multi-room fusion for reliable home-empty detection:

**Algorithm:**
1. Aggregate room-level occupancy states
2. Apply room consensus (all rooms must agree on empty)
3. Apply entrance penalty (5-minute boost after entrance)
4. Temporal smoothing (60-second window)
5. Episode detection with minimum duration

**Safety Features:**
- Entrance penalty prevents rapid false-empty after entrance
- Temporal smoothing reduces flickering
- Room consensus requires all rooms to agree

```python
fusion = HomeEmptyFusion(HomeEmptyConfig(
    min_precision=0.95,
    max_false_empty_rate=0.05,
    entrance_penalty_duration_seconds=300.0,
))
predictions = fusion.fuse(room_predictions, timestamps)
episodes = fusion.detect_episodes(predictions)
```

#### HouseholdGate
Validates household-level gate requirements:
- Precision >= 0.95 check
- False-empty rate <= 0.05 check
- Coverage validation (< 20% uncertain)

### Test Coverage: 22 tests

---

## Integration Points

### Lane A (ADL Registry)
All modules use canonical event IDs from `ADLEventRegistry`:
```python
from ml.adl_registry import ADLEventRegistry
registry = ADLEventRegistry.load()
event_id = registry.resolve_event_id("sleeping")  # "sleeping"
```

### Lane C (CNN+Transformer)
Event decoder consumes Head A/B outputs:
```python
# Head A: Occupancy probabilities
occupancy_probs = model.head_a(features)  # [batch, seq_len, 1]

# Head B: Activity probabilities  
activity_probs = model.head_b(features)   # [batch, seq_len, num_activities]

# Decoder fusion
predictions = decoder.decode(occupancy_probs, activity_probs, timestamps, room)
```

### Validation Framework
Gate reports integrate with validation run manager:
```python
from ml.validation_run import ValidationRunManager
manager = ValidationRunManager()
report = checker.check_all_gates(metrics, date)
manager.record_daily_result(run_id, date, report.to_dict())
```

---

## Usage Examples

### Complete Pipeline
```python
from ml.event_compiler import EpisodeCompiler
from ml.event_decoder import EventDecoder
from ml.derived_events import CareKPIExtractor
from ml.event_gates import EventGateChecker

# 1. Decode window predictions
decoder = EventDecoder()
window_preds = decoder.decode(occ_probs, act_probs, timestamps, "bedroom")

# 2. Compile episodes
compiler = EpisodeCompiler()
episodes = compiler.compile(window_preds)

# 3. Extract care KPIs
extractor = CareKPIExtractor()
kpis = extractor.extract_day_kpis({"bedroom": episodes}, target_date)

# 4. Check gates
checker = EventGateChecker()
metrics = {
    "home_empty_precision": 0.96,
    "home_empty_false_empty_rate": 0.04,
    "event_recalls": {"sleep_duration": 0.55},
}
report = checker.check_all_gates(metrics, target_date)
```

### Multi-Room Home-Empty Detection
```python
from ml.home_empty_fusion import HomeEmptyFusion

fusion = HomeEmptyFusion()
room_predictions = {
    "bedroom": bedroom_df,
    "kitchen": kitchen_df,
    "bathroom": bathroom_df,
}

predictions = fusion.fuse(room_predictions, timestamps)
episodes = fusion.detect_episodes(predictions)

# Check household gates
gate = HouseholdGate()
results = gate.check_household_gate(predictions, ground_truth)
```

---

## Test Summary

| Module | Tests | Status |
|--------|-------|--------|
| test_event_compiler.py | 18 | ✅ Pass |
| test_event_decoder.py | 18 | ✅ Pass |
| test_derived_events.py | 20 | ✅ Pass |
| test_event_gates.py | 28 | ✅ Pass |
| test_home_empty_fusion.py | 22 | ✅ Pass |
| **Lane B Total** | **106** | ✅ **Pass** |
| Full Suite | 463 | ✅ Pass |

---

## Next Steps: Lane C

1. **PR-C1: Dual-Head Output Contract**
   - Define Head A (occupancy) + Head B (activity) output shapes
   - Implement loss functions for dual-head training
   - Add gradient balancing between heads

2. **PR-C2: Shadow Mode Pipeline Integration**
   - Integrate event-first path into unified training
   - Shadow mode for A/B comparison with legacy path
   - Gate evaluation during validation runs

---

## Dependencies

```
Lane B depends on:
├── Lane A (ADL Registry)
│   └── backend/ml/adl_registry.py
├── backend/ml/event_compiler.py
├── backend/ml/event_decoder.py
├── backend/ml/derived_events.py
├── backend/ml/event_gates.py
├── backend/ml/event_kpi.py
└── backend/ml/home_empty_fusion.py
```

Lane C will depend on all Lane B modules.
