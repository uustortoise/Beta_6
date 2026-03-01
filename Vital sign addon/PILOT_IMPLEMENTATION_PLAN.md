# Implementation Plan: Vital Sign App Pilot Readiness

**Goal:** Prepare the Vital Sign Android App for **Pilot Testing**.
**Constraint:** No personalized user calibration required (Population-based approach).
**Base:** Refactored `pyVHR` implementation (Java + Chaquopy).

## 1. Codebase Transition (Week 1)

**Objective:** Move from the deprecated prototype to the production-ready structure.

### 1.1 Archive Legacy Code
- **Action:** Move `MainActivity.java` (root) and `VITAL_SIGN_ADDON_CODE_REVIEW.md` to `legacy/` folder.
- **Why:** Prevent confusion between the prototype and the actual product code.

### 1.2 Promote Implementation
- **Action:** Move contents of `implementation/` to the project root (or appropriate module location `Development/Beta_5.5/Vital sign addon/`).
- **Dependencies:** Ensure `build.gradle` paths for Chaquopy are correct after control moves.

## 2. P1 Feature Implementation (Week 2-3)

**Objective:** Implement critical features required for valid data collection during pilot.

### 2.1 Motion Artifact Rejection (Detect & Pause)
**Approach:** 
- Instead of complex filtering, simply pause analysis when the user moves.
- **Algorithm:**
  1. Calculate pixel intensity difference between consecutive frames in the Forehead ROI.
  2. If `diff > THRESHOLD`, set state to `UNSTABLE`.
  3. UI Feedback: Show "Moved too much - Pausing" overlay.
  4. Discard buffered frames during unstable period.

```python
# Pseudo-code for vitals_processor.py
def check_motion(current_frame, prev_frame):
    diff = np.mean(np.abs(current_frame - prev_frame))
    if diff > MOTION_THRESHOLD:
        return True # Motion detected
    return False
```

### 2.2 Data Persistence (Room Database)
**Approach:** 
- Store every valid reading to local SQLite DB to prevent data loss.
- **Schema:** `Measurement(id, session_uuid, timestamp, heart_rate, spo2, quality_score, skin_tone_category)`
- **Export:** Add functionality to export specific sessions to CSV/JSON from the DB.

### 2.3 Population-Based Skin Tone Calibration
**Approach:** Global adjustments based on detected skin lightness (No user setup).
- **Logic:**
  1. Calculate Mean Value (V in HSV) of Forehead ROI.
  2. Categorize: `Light (>0.7)`, `Medium (0.4-0.7)`, `Dark (<0.4)`.
  3. Apply Offset:
     - **Light:** No offset.
     - **Medium:** SpO2 +1.5%, HR Gain 1.05.
     - **Dark:** SpO2 +3.0%, HR Gain 1.10.
- **Note:** These values are starting points and should be refined during the pilot.

## 3. Pilot Hardening (Week 4)

**Objective:** Ensure the app doesn't crash and provides debug info.

### 3.1 Error Reporting
- Integrate a crash reporting tool (e.g., Firebase Crashlytics) or write logs to a local file (`app_log.txt`) that can be exported.

### 3.2 Session Management
- Ensure `Start` generates a new Session ID.
- Ensure `Stop` finalizes the session and saves summary stats.

## 4. Verification Plan

### 4.1 Automated Tests
- **Unit:** Test `vitals_processor.py` with synthetic sine wave signals (70, 90, 120 BPM) to verify HR calculation logic.

### 4.2 Manual Pilot Verification Steps
1. **Install:** Deploy APK to 3 different Android devices (Low/Mid/High end).
2. **Motion Test:** Shake head while measuring -> Confirm app pauses and resumes.
3. **Data Test:** Complete 1-minute session -> Restart App -> Confirm data exists in History/Export.
4. **Calibration Test:** Test on subjects with different skin tones -> Verify `skin_tone_category` is logged correctly in DB.

## 5. Timeline

| Week | Activity | Deliverable |
|------|----------|-------------|
| **1** | Migration & Cleanup | Clean Project Structure |
| **2** | Motion & Database | App with History & Stability Check |
| **3** | Calibration & Testing | "Smart" App that adjusts to user |
| **4** | Hardening & Release | **Pilot-Ready APK** |
