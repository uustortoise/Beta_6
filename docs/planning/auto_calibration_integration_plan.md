# Auto-Calibration Module: Integration Plan

**Status**: Proposed — for team review before implementation.
**Module**: `backend/ml/sensor_calibration.py` (new file)
**Toggle**: `--enable-sensor-auto-calibration` CLI flag (default OFF)

---

## Problem Statement

Sensor baselines differ across homes. Jessica's "empty room" CO2 is ~2,500 ppm; Samuel's is ~430 ppm. Our current pipeline uses hardcoded thresholds (e.g., `co2 > 3100`), which means:

- Models trained on Jessica break on Samuel
- Cross-user training data is incompatible
- New installations require manual threshold tuning
- Seasonal drift (winter windows closed → CO2 rises) causes silent degradation

## Proposed Solution

A standalone, toggleable module that inspects the first 2 nights of raw sensor data to automatically determine per-room occupied/unoccupied thresholds.

---

## Module Design

### New File: `backend/ml/sensor_calibration.py`

```python
# Public API (3 functions):

calibrate_room_baselines(df, room) -> SensorBaselines
  # Uses 2AM-5AM nighttime windows to establish "empty room" readings
  # Returns: co2_empty, light_off, motion_floor, sound_floor, humidity_base

compute_occupied_thresholds(baselines) -> OccupiedThresholds
  # Derives "occupied" triggers as relative offsets from baselines
  # Returns: co2_occupied, light_on, motion_active, sound_active

normalize_sensor_features(df, baselines) -> DataFrame
  # Converts absolute sensor values to relative features
  # e.g., co2_relative = (current - baseline) / baseline
  # Makes features comparable across homes
```

### Data Structures

```python
@dataclass
class SensorBaselines:
    room: str
    co2_empty: float        # Median nighttime CO2
    light_off: float        # 90th percentile nighttime light
    motion_floor: float     # 95th percentile nighttime motion
    sound_floor: float      # 90th percentile nighttime sound
    humidity_base: float    # Median nighttime humidity
    calibration_nights: int # Number of nights used
    calibrated_at: str      # ISO timestamp

@dataclass
class OccupiedThresholds:
    room: str
    co2_occupied: float     # baselines.co2_empty * 1.20
    light_on: float         # max(baselines.light_off * 5.0, 20.0)
    motion_active: float    # max(baselines.motion_floor * 3.0, 0.5)
    sound_active: float     # baselines.sound_floor * 1.15
```

---

## Integration Points

### 1. Backtest Pipeline (`run_event_first_backtest.py`)

```diff
  # New CLI flag
+ --enable-sensor-auto-calibration   (default: off)

  # In run_backtest(), after data load, before split generation:
+ if enable_sensor_auto_calibration:
+     baselines = calibrate_room_baselines(room_df, room_name)
+     thresholds = compute_occupied_thresholds(baselines)
+     room_df = normalize_sensor_features(room_df, baselines)
+     # Store in output JSON for traceability
+     report['sensor_calibration'][room] = asdict(baselines)
```

### 2. Label Audit Script (`audit_livingroom_ui.py`)

```diff
  # Replace hardcoded thresholds:
- lr_unocc['has_co2'] = (lr_unocc['co2'] > 3100).astype(int)
+ lr_unocc['has_co2'] = (lr_unocc['co2'] > thresholds.co2_occupied).astype(int)
```

### 3. Tier 1 Rule Engine (future)

The rule engine would consume `OccupiedThresholds` directly instead of hardcoded values. This module becomes the data source for Tier 1 rules.

---

## Toggle Behavior

| Flag State | Behavior |
|---|---|
| **OFF** (default) | No change. Existing hardcoded thresholds used. Zero risk. |
| **ON** | Calibration runs at pipeline start. Thresholds logged in output JSON. Normalized features fed to model. |

When OFF, the module is never imported. No performance cost, no side effects.

---

## Testing Plan

### Unit Tests (`test_sensor_calibration.py`)

1. **Night extraction**: Verify 2AM-5AM filter works across timezone edge cases
2. **Baseline stability**: Two nights of identical data → stable baselines
3. **Threshold derivation**: Known baselines → known thresholds
4. **Edge cases**: Missing sensors, all-zero data, single night only
5. **Normalization**: Verify relative features are bounded and finite

### Integration Validation

1. Run Candidate D on Jessica data WITH calibration ON → confirm results don't regress (thresholds should auto-discover values close to current hardcoded ones)
2. Run on Samuel data WITH calibration ON → confirm it produces sensible thresholds (~516 CO2 instead of 3100)

---

## Estimated Effort

| Task | Time |
|---|---|
| `sensor_calibration.py` + dataclasses | 2 hours |
| Unit tests | 2 hours |
| Backtest CLI flag wiring | 1 hour |
| Audit script integration | 30 min |
| Validation run (Jessica + Samuel) | 1 hour |
| **Total** | **~1 day** |

---

## Open Questions for Team

1. **Night window**: Is 2AM-5AM reliable for all elders? Some may have insomnia or nighttime bathroom habits that create noise in the "empty room" window. Should we use a wider window (1AM-6AM) with outlier rejection?

2. **Minimum nights**: Plan requires 2 nights for stability. Is 1 night acceptable as a degraded fallback?

3. **Storage**: Should calibrated baselines be persisted to DB (for the Ops UI to display) or kept ephemeral (recalculated each run)?

4. **Bedroom special case**: The Bedroom is occupied at night (sleep), so its "empty room" baseline must come from daytime unoccupied periods instead. This requires a different calibration strategy. Should we handle this in v1 or defer?
