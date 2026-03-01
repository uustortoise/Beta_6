# Pilot Implementation Plan & Task List - Code Review

**Review Date:** 2026-02-03  
**Documents Reviewed:**
- `PILOT_IMPLEMENTATION_PLAN.md`
- `PILOT_TASK_LIST.md`

---

## Executive Summary

Both documents are **well-structured and practical**. The 4-week timeline is realistic for the MVP scope. However, there are **critical gaps** in clinical validation, regulatory compliance, and risk mitigation that need addressing before any pilot deployment.

| Aspect | Assessment | Priority |
|--------|------------|----------|
| Technical Planning | ✅ Good | - |
| Timeline | ✅ Realistic | - |
| Clinical Safety | ⚠️ Needs Improvement | P0 |
| Data Privacy | ⚠️ Missing | P0 |
| Risk Mitigation | ⚠️ Incomplete | P1 |
| Testing Coverage | ⚠️ Light | P1 |

---

## Detailed Review: PILOT_IMPLEMENTATION_PLAN.md

### ✅ Strengths

#### 1. Clear Phased Approach
The 4-week structure is logical:
- Week 1: Cleanup & Migration
- Week 2-3: Core Features (Motion, DB, Calibration)
- Week 4: Hardening & Release

**Verdict:** Appropriate for the scope.

#### 2. Motion Artifact Rejection Strategy
```
"Instead of complex filtering, simply pause analysis when user moves"
```

**Verdict:** Smart MVP approach. Reduces complexity while ensuring data quality.

#### 3. Population-Based Calibration
Acknowledging no personalized calibration keeps scope manageable:
```
Light: No offset
Medium: SpO2 +1.5%, HR Gain 1.05
Dark: SpO2 +3.0%, HR Gain 1.10
```

**Verdict:** Reasonable starting point with clear caveat that values need refinement.

#### 4. Practical Database Schema
```
Measurement(id, session_uuid, timestamp, heart_rate, spo2, quality_score, skin_tone_category)
```

**Verdict:** Captures essential data for pilot analysis.

---

### ⚠️ Critical Gaps (Must Address)

#### 1. [P0] Missing Clinical Validation Plan

**Issue:** No plan for comparing against gold-standard devices.

**Why It Matters:** Without validation, pilot data is uninterpretable.

**Required Addition:**
```markdown
## Clinical Validation Protocol

### Reference Devices
- Pulse Oximeter: [Model - e.g., Masimo Rad-97]
- ECG: [Model for HRV ground truth]
- Manual Respiratory Count (for RPM)

### Validation Procedure
1. Simultaneous recording with reference device
2. 5-minute sessions per subject
3. Record: Device readings, app readings, time offset
4. Calculate: MAE, RMSE, Pearson correlation

### Success Criteria
- HR: MAE < 5 BPM, r > 0.9
- SpO2: MAE < 3%, r > 0.85
- RPM: MAE < 3 RPM, r > 0.8
```

---

#### 2. [P0] Missing Informed Consent & Ethics

**Issue:** No mention of IRB approval or participant consent.

**Required Addition:**
```markdown
## Ethics & Compliance

- [ ] IRB approval obtained (or exempt status confirmed)
- [ ] Informed consent form drafted
- [ ] Data use agreement signed by participants
- [ ] Right to withdraw explained
- [ ] "Not a medical device" disclaimer in app and consent
```

---

#### 3. [P0] Data Privacy Plan Missing

**Issue:** No data protection measures specified.

**Required Addition:**
```markdown
## Data Privacy & Security

### Data Classification
- Health data (PHI) - High sensitivity

### Protection Measures
- [ ] Local encryption (SQLCipher for Room DB)
- [ ] No cloud transmission in pilot
- [ ] Device passcode required
- [ ] Data deletion procedure post-pilot
- [ ] Anonymization before analysis

### Compliance
- [ ] GDPR compliance (if EU subjects)
- [ ] HIPAA checklist (if US healthcare context)
```

---

#### 4. [P1] Incomplete Risk Mitigation

**Issue:** No "what if things go wrong" planning.

**Required Addition:**
```markdown
## Risk Mitigation

### Technical Risks
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| App crashes during session | Medium | High | Auto-save every 10s; graceful recovery |
| Motion rejection too aggressive | Medium | Medium | Adjustable threshold; user feedback |
| Skin tone detection fails | Low | High | Manual override option; log failures |
| Battery drain >10%/hour | Low | Medium | Background limits; user warning |

### Clinical Risks
| Risk | Mitigation |
|------|------------|
| User trusts app over medical advice | Clear "Research Only" warnings; consent reinforces |
| False reassurance from normal readings | Display confidence intervals; uncertainty communication |
| Anxiety from abnormal readings | Contact information for healthcare provider |
```

---

#### 5. [P1] Light Testing Coverage

**Issue:** Only 3 manual tests specified.

**Required Expansion:**
```markdown
## Expanded Verification Plan

### Automated Tests (Unit)
- [ ] HR calculation: 60 BPM sine wave → expect 60±2
- [ ] HR calculation: 100 BPM sine wave → expect 100±2
- [ ] HR calculation: 120 BPM sine wave → expect 120±2
- [ ] Motion detection: Sudden change → expect pause trigger
- [ ] Skin tone: Light image → expect "Light" category
- [ ] Skin tone: Dark image → expect "Dark" category
- [ ] Database: Insert → Query → expect match

### Device Compatibility Tests
- [ ] Low-end: Android 7, 2GB RAM, 720p camera
- [ ] Mid-range: Android 10, 4GB RAM, 1080p camera
- [ ] High-end: Android 14, 8GB RAM, 4K camera

### Environmental Tests
- [ ] Bright sunlight (>1000 lux)
- [ ] Indoor office lighting (300-500 lux)
- [ ] Dim room (<100 lux)
- [ ] Backlight (window behind subject)

### User Scenario Tests
- [ ] 5-minute continuous session
- [ ] Multiple start/stop cycles
- [ ] App background/foreground switch
- [ ] Phone call interruption
- [ ] Low battery (<20%) behavior
```

---

### 📝 Minor Improvements

#### 1. Motion Algorithm Specification

Current pseudo-code is too simple:
```python
def check_motion(current_frame, prev_frame):
    diff = np.mean(np.abs(current_frame - prev_frame))
    if diff > MOTION_THRESHOLD:
        return True
```

**Recommended Enhancement:**
```python
def check_motion(current_frame, prev_frame, roi_coords):
    """
    Motion detection with ROI isolation and temporal filtering.
    """
    # Extract ROI only
    cx, cy, cw, ch = roi_coords
    curr_roi = current_frame[cy:cy+ch, cx:cx+cw]
    prev_roi = prev_frame[cy:cy+ch, cx:cx+cw]
    
    # Calculate frame difference
    diff = np.mean(np.abs(curr_roi.astype(float) - prev_roi.astype(float)))
    
    # Temporal filtering (require 2 consecutive motion frames)
    if not hasattr(check_motion, 'prev_diff'):
        check_motion.prev_diff = 0
    
    sustained_motion = diff > MOTION_THRESHOLD and check_motion.prev_diff > MOTION_THRESHOLD * 0.8
    check_motion.prev_diff = diff
    
    return sustained_motion, diff  # Return diff for UI feedback
```

---

#### 2. Database Schema Enhancement

Current schema misses important fields:

**Recommended:**
```sql
CREATE TABLE measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_uuid TEXT NOT NULL,
    timestamp INTEGER NOT NULL,  -- Unix timestamp (ms)
    
    -- Vital signs
    heart_rate REAL,
    heart_rate_confidence REAL,
    spo2 REAL,
    spo2_confidence REAL,
    respiration_rate REAL,
    hrv_sdnn REAL,
    
    -- Quality metrics
    signal_quality REAL,
    motion_detected BOOLEAN,
    
    -- Calibration data
    skin_tone_category TEXT,  -- 'light', 'medium', 'dark'
    skin_tone_value REAL,     -- HSV V-channel value
    calibration_offset_applied REAL,
    
    -- Device info
    device_model TEXT,
    android_version TEXT,
    app_version TEXT,
    
    -- Environmental (if available)
    ambient_light_estimate REAL,  -- From camera exposure
    
    -- Raw data reference (optional)
    has_raw_data BOOLEAN DEFAULT FALSE,
    raw_data_file_path TEXT
);

-- Indexes for queries
CREATE INDEX idx_session ON measurements(session_uuid);
CREATE INDEX idx_timestamp ON measurements(timestamp);
```

---

#### 3. Success Criteria Definition

Missing: What defines a "successful" pilot?

**Recommended Addition:**
```markdown
## Pilot Success Criteria

### Technical Success
- [ ] App crash rate < 1%
- [ ] Data capture success > 90% (valid readings)
- [ ] Motion pause < 30% of session time
- [ ] Battery usage < 8% per hour

### Clinical Success
- [ ] HR MAE < 5 BPM vs reference
- [ ] SpO2 MAE < 4% vs reference
- [ ] Valid readings on all 3 skin tone categories

### User Experience Success
- [ ] 80%+ users complete 5-minute session
- [ ] < 20% report "confusing" or "frustrating"
- [ ] Average signal quality > 0.6

### Go/No-Go Decision
- **Go:** 3/3 technical + 2/3 clinical criteria met
- **No-Go:** Major rework required
```

---

## Detailed Review: PILOT_TASK_LIST.md

### ✅ Strengths

1. **Checkbox Format:** Easy to track progress
2. **IDs for Reference:** Good for ticketing system
3. **Hierarchical Structure:** Phases → Tasks → Subtasks
4. **Clear Completion Criteria:** Each item is actionable

---

### ⚠️ Gaps & Recommendations

#### 1. [P0] Missing Dependencies Between Tasks

**Issue:** No indication of task dependencies.

**Recommended Addition:**
```markdown
## Task Dependencies

graph TD
    A[Archive Legacy] --> B[Promote Implementation]
    B --> C[Verify Build]
    C --> D[Motion Detection]
    C --> E[Room Database]
    D --> F[Calibration]
    E --> F
    F --> G[Unit Tests]
    G --> H[Crashlytics]
    H --> I[Session Management]
    I --> J[Signed APK]
    J --> K[Alpha Test]
```

Or in table format:
| ID | Task | Depends On | Blocks |
|----|------|------------|--------|
| 0 | Archive Legacy | - | 1 |
| 1 | Promote Impl | 0 | 2 |
| 2 | Verify Build | 1 | 3,4,5 |
| 3 | Motion | 2 | 6 |
| 4 | Database | 2 | 8 |
| 5 | Calibration | 2,3 | 6 |

---

#### 2. [P1] Missing Owner Assignment

**Issue:** No clear ownership for tasks.

**Recommended Format:**
```markdown
## Phase 1: Codebase Transition

| ID | Task | Owner | Est. Hours | Status |
|----|------|-------|------------|--------|
| 0 | Archive legacy | @android-dev | 2 | ⬜ |
| 1 | Promote impl | @android-dev | 4 | ⬜ |
| 2 | Verify build | @qa-engineer | 4 | ⬜ |
```

---

#### 3. [P1] Missing Definition of Done

**Issue:** What constitutes "complete"?

**Recommended Addition per Task:**
```markdown
### Task 3: Motion Artifact Rejection

**Definition of Done:**
- [ ] Algorithm implemented in vitals_processor.py
- [ ] UI overlay "Pausing - Please hold still" implemented
- [ ] Unit test: Motion simulation triggers pause
- [ ] Manual test: 3/3 head shakes detected and paused
- [ ] Code reviewed and merged
- [ ] Documented in README
```

---

#### 4. [P1] Risk Column Missing

Some tasks are riskier than others.

**Recommended Addition:**
```markdown
## Risk-Adjusted Task List

| ID | Task | Risk Level | Contingency |
|----|------|------------|-------------|
| 3 | Motion Detection | Low | Increase threshold if too sensitive |
| 4 | Room Database | Low | Standard Android pattern |
| 5 | Skin Tone Calibration | **High** | Manual override if auto-detection fails |
| 6 | Unit Tests | Medium | Reduce coverage if time-constrained |
```

**Specific Risk for Task 5:**
Population-based calibration is experimental. Risk of:
- Inaccurate categorization
- Insufficient pilot data to validate offsets
- Need for personalized calibration anyway

**Mitigation:**
- Add manual skin tone selection as backup
- Log raw values alongside calibrated values
- Plan for personalized calibration in Phase 2

---

## Consolidated Recommendations

### Immediate Actions (Before Sprint Start)

1. **Add Clinical Validation Protocol** (P0)
   - Reference device specification
   - Comparison methodology
   - Success criteria

2. **Add Ethics & Compliance Section** (P0)
   - IRB status
   - Consent form
   - Privacy measures

3. **Expand Testing Plan** (P1)
   - Automated unit tests
   - Device matrix
   - Environmental scenarios

4. **Assign Task Owners** (P1)
   - Clear ownership
   - Time estimates
   - Dependencies

### Timeline Adjustment Suggestion

Current: 4 weeks  
Recommended: **5 weeks** with following adjustment:

| Week | Original | Adjusted |
|------|----------|----------|
| 1 | Migration | Migration + Validation Protocol |
| 2 | Motion + DB | Motion + DB |
| 3 | Calibration + Testing | Calibration + Testing + Ethics |
| 4 | Hardening | Hardening |
| 5 | - | **Validation Study** |

**Rationale:** Week 5 is essential for actually validating the app works before calling it "pilot-ready."

---

## Revised Task List (Recommended)

```markdown
# Revised Task List: Pilot Readiness

## Pre-Flight (Before Coding)
- [ ] [P0] Finalize clinical validation protocol
- [ ] [P0] Obtain IRB approval / exempt status
- [ ] [P0] Draft informed consent form
- [ ] [P0] Secure reference devices (pulse oximeter, ECG)

## Phase 1: Foundation (Week 1)
- [ ] Archive legacy code (@android-dev, 2h)
- [ ] Promote implementation (@android-dev, 4h)
- [ ] Verify build on 3 devices (@qa, 4h)

## Phase 2: Core Features (Week 2-3)
- [ ] Motion detection with UI feedback (@android-dev, 8h)
- [ ] Room DB with encryption (@android-dev, 8h)
- [ ] Population skin tone calibration (@ml-dev, 12h)
  - [ ] Subtask: HSV detection
  - [ ] Subtask: Offset application
  - [ ] Subtask: Manual override fallback
- [ ] Unit tests for signal processing (@qa, 8h)

## Phase 3: Hardening (Week 4)
- [ ] Crashlytics integration (@android-dev, 4h)
- [ ] Session management (@android-dev, 6h)
- [ ] Privacy compliance audit (@pm, 4h)
- [ ] Generate signed APK (@devops, 2h)

## Phase 4: Validation (Week 5)
- [ ] Internal dogfooding (Team, 2 days)
- [ ] Validation study (10 subjects) (@clinical, 3 days)
- [ ] Data analysis & go/no-go decision (@pm + @clinical, 2 days)
```

---

## Final Verdict

| Document | Rating | Key Action |
|----------|--------|------------|
| **PILOT_IMPLEMENTATION_PLAN.md** | B+ | Add clinical validation, ethics, privacy sections |
| **PILOT_TASK_LIST.md** | B | Add owners, dependencies, definition of done |

**Overall Assessment:** Solid foundation with **critical gaps in clinical safety and compliance** that must be addressed before pilot deployment.

---

*Review completed: 2026-02-03*
