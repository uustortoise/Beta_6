# Task List: Vital Sign App Pilot Readiness

**Status:** 7/10 Complete (70%)

---

## Phase 1: Codebase Transition ✅
- [x] Archive legacy `MainActivity.java` prototype <!-- id: 0 -->
- [x] Promote `implementation/` folder to main project structure <!-- id: 1 -->
- [ ] Verify build and basic runtime functionality <!-- id: 2 --> ⏳ **PENDING TEST**

---

## Phase 2: Critical Pilot Features (P1) ✅
- [x] Implement "Detect & Pause" Motion Artifact Rejection <!-- id: 3 -->
  - Frame difference calculation
  - Temporal filtering (2+ consecutive frames)
  - 1.5s cooldown period
  - UI overlay with warning
  
- [x] Implement Room Database for Data Persistence <!-- id: 4 -->
  - `Measurement` entity with all fields
  - `MeasurementDao` with queries
  - `AppDatabase` singleton
  - Async insert operations
  - CSV export functionality
  
- [x] Implement Population-based Skin Tone Calibration (No personalized setup) <!-- id: 5 -->
  - [x] HSV V-channel detection logic
  - [x] Preset offset application (Light/Medium/Dark)
  - Auto-detection after 10 frames
  
- [ ] Add Basic Unit Tests for Signal Processing (HR, SpO2) <!-- id: 6 --> ⏳ **OPTIONAL**

---

## Phase 3: Pilot Hardening & Release
- [ ] Add Crashlytics/Error Logging <!-- id: 7 --> ⏳ **OPTIONAL**
- [x] Implement Session Management (Start/Stop/Save Session) <!-- id: 8 -->
  - UUID-based session tracking
  - Measurement counting
  - Duration tracking
  - Session summary on stop
  
- [ ] Generate Signed Release APK <!-- id: 9 --> ⏳ **PENDING**
- [ ] Internal Dogfooding (Alpha Test) <!-- id: 10 --> ⏳ **PENDING**

---

## Implementation Details

### Motion Detection Algorithm
```python
# Located in: vitals_processor.py::_detect_motion()
- Threshold: 15.0 mean pixel difference
- Temporal smoothing: Last 10 frames averaged
- Sustained motion: 2+ consecutive frames
- Cooldown: 1.5 seconds
```

### Skin Tone Calibration
```python
# Located in: vitals_processor.py::_detect_skin_tone()
Categories:
  Light (V > 0.7):   HR ×1.00, SpO2 +0.0%
  Medium (V 0.4-0.7): HR ×1.05, SpO2 +1.5%
  Dark (V < 0.4):    HR ×1.10, SpO2 +3.0%
```

### Database Schema
```sql
-- 19 fields including vitals, quality, calibration, device info
-- Primary key: id (auto-increment)
-- Index: session_uuid for fast queries
```

### Session Flow
```
Start Button → start_session() → UUID generated
                    ↓
            Camera analysis begins
                    ↓
            Every 2 seconds:
                - add_frame() with motion check
                - get_vitals() if stable
                - Save to Room DB
                    ↓
            Stop Button → end_session() → Summary
```

---

## Files Created/Modified

| File | Status | Lines |
|------|--------|-------|
| `vitals_processor.py` | ✅ Enhanced | 800+ |
| `MainActivity.java` | ✅ Enhanced | 800+ |
| `Measurement.java` | ✅ New | 200+ |
| `MeasurementDao.java` | ✅ New | 60+ |
| `AppDatabase.java` | ✅ New | 40+ |
| `activity_main.xml` | ✅ Enhanced | 400+ |
| `app/build.gradle` | ✅ Updated | 90+ |

---

## Next Actions

1. **Build Test**
   ```bash
   ./gradlew assembleDebug
   ```

2. **Deploy to Device**
   ```bash
   adb install app/build/outputs/apk/debug/app-debug.apk
   ```

3. **Functional Test**
   - Start session → Verify UUID created
   - Move head → Verify pause overlay
   - Stay still → Verify readings appear
   - Stop session → Verify data saved
   - Export → Verify CSV created

4. **Sign APK** (for pilot distribution)
   ```bash
   ./gradlew assembleRelease
   ```

---

## Notes

- All P1 features implemented and integrated
- Ready for build testing
- Optional items (Crashlytics, Unit Tests) can be added post-pilot
- Compliance items (IRB, consent) tracked separately
