-- 1. Elder Master Table
CREATE TABLE IF NOT EXISTS elders (
    elder_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    date_of_birth DATE,
    gender TEXT,
    blood_type TEXT,
    profile_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP
);

-- 2. Medical History (Flexible JSON for complex data)
CREATE TABLE IF NOT EXISTS medical_history (
    id INTEGER PRIMARY KEY,
    elder_id TEXT REFERENCES elders(elder_id),
    category TEXT, -- 'chronic', 'medication', 'allergy', 'surgery', 'vaccination'
    data JSON NOT NULL, -- Structured medical data
    recorded_date DATE,
    recorded_by TEXT,
    version INTEGER DEFAULT 1
);

-- Index for querying history by category
CREATE INDEX IF NOT EXISTS idx_med_history_cat ON medical_history(elder_id, category);

-- 3. ICOPE Assessments
CREATE TABLE IF NOT EXISTS icope_assessments (
    id INTEGER PRIMARY KEY,
    elder_id TEXT REFERENCES elders(elder_id),
    assessment_date DATE NOT NULL,
    locomotion_score REAL,
    cognition_score REAL,
    psychological_score REAL,
    sensory_score REAL,
    vitality_score REAL,
    overall_score REAL,
    recommendations JSON,
    trend TEXT -- 'improving', 'stable', 'declining'
);

CREATE INDEX IF NOT EXISTS idx_icope_date ON icope_assessments(elder_id, assessment_date);

-- 4. Sleep Analysis
CREATE TABLE IF NOT EXISTS sleep_analysis (
    id INTEGER PRIMARY KEY,
    elder_id TEXT REFERENCES elders(elder_id),
    analysis_date DATE NOT NULL,
    duration_hours REAL,
    efficiency_percent REAL,
    sleep_stages JSON, -- Breakdown of Light/Deep/REM
    quality_score REAL,
    insights TEXT,
    UNIQUE(elder_id, analysis_date)
);

CREATE INDEX IF NOT EXISTS idx_sleep_date ON sleep_analysis(elder_id, analysis_date);

-- 5. ADL (Activities of Daily Living) History
CREATE TABLE IF NOT EXISTS adl_history (
    id INTEGER PRIMARY KEY,
    elder_id TEXT REFERENCES elders(elder_id),
    record_date DATE NOT NULL,
    timestamp DATETIME, -- Exact time (normalized to 10-second intervals)
    activity_type TEXT, -- 'toileting', 'sleeping', 'wandering'
    duration_minutes INTEGER,
    confidence REAL,
    room TEXT,
    is_anomaly INTEGER DEFAULT 0,
    is_corrected INTEGER DEFAULT 0,  -- Flag for manually corrected labels (Golden Samples)
    sensor_features JSON  -- Stores {motion, temp, light, sound, co2, humidity} for training
);


-- Critical Index for the Insights Engine (Query: "How many toilet visits between 12AM-6AM?")
CREATE INDEX IF NOT EXISTS idx_adl_type_time ON adl_history(elder_id, activity_type, timestamp);
CREATE INDEX IF NOT EXISTS idx_adl_anomaly ON adl_history(elder_id, is_anomaly);
CREATE INDEX IF NOT EXISTS idx_adl_date ON adl_history(elder_id, record_date);

-- UNIQUE constraint to prevent duplicate events when re-uploading same file
CREATE UNIQUE INDEX IF NOT EXISTS idx_adl_unique ON adl_history(elder_id, timestamp, room);

-- Performance index for correction-aware queries
CREATE INDEX IF NOT EXISTS idx_adl_corrected ON adl_history(elder_id, record_date, room, is_corrected);

-- 5b. Correction History (Audit Trail)
-- Tracks all corrections made to adl_history for debugging and accountability
-- Supports soft-delete: ML forgets (via cascade reset), Audit remembers (via is_deleted flag)
CREATE TABLE IF NOT EXISTS correction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    adl_history_id INTEGER,  -- Reference to corrected row (if available)
    elder_id TEXT NOT NULL,
    room TEXT NOT NULL,
    timestamp_start DATETIME,  -- Start of correction range
    timestamp_end DATETIME,    -- End of correction range
    old_activity TEXT,         -- Previous activity label
    new_activity TEXT NOT NULL, -- New corrected label
    rows_affected INTEGER,     -- Number of rows updated
    corrected_by TEXT DEFAULT 'user',
    corrected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Soft-delete support (1000 POC: ML forgets, Audit remembers)
    is_deleted INTEGER DEFAULT 0,  -- 0=active, 1=soft-deleted
    deleted_at TIMESTAMP,          -- When the correction was deleted
    deleted_by TEXT                -- Who deleted it (for accountability)
);

CREATE INDEX IF NOT EXISTS idx_correction_hist_elder ON correction_history(elder_id, corrected_at DESC);
CREATE INDEX IF NOT EXISTS idx_correction_active ON correction_history(elder_id, is_deleted);

-- Performance indexes for medical history and contacts
CREATE INDEX IF NOT EXISTS idx_medical_history_elder ON medical_history(elder_id, category);

CREATE INDEX IF NOT EXISTS idx_icope_latest ON icope_assessments(elder_id, assessment_date DESC);
CREATE INDEX IF NOT EXISTS idx_sleep_latest ON sleep_analysis(elder_id, analysis_date DESC);

-- 6. Emergency Contacts
CREATE TABLE IF NOT EXISTS emergency_contacts (
    id INTEGER PRIMARY KEY,
    elder_id TEXT REFERENCES elders(elder_id),
    name TEXT NOT NULL,
    relationship TEXT,
    phone TEXT NOT NULL,
    email TEXT,
    is_primary INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_contacts_elder ON emergency_contacts(elder_id);

-- 7. Predictions (Enhancing existing table if needed, or ensuring compatibility)
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resident_id TEXT, -- Legacy name for elder_id, kept for compatibility or migrated
    timestamp DATETIME,
    room TEXT,
    activity TEXT,
    confidence REAL,
    is_anomaly INTEGER,
    UNIQUE(resident_id, timestamp, room)
);

CREATE INDEX IF NOT EXISTS idx_resident_ts ON predictions (resident_id, timestamp);

-- 8. Generated Alerts
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    elder_id TEXT REFERENCES elders(elder_id),
    alert_date DATE DEFAULT CURRENT_DATE,
    alert_type TEXT, -- 'health_risk', 'maintenance', 'behavior'
    severity TEXT, -- 'low', 'medium', 'high', 'critical'
    title TEXT,
    message TEXT,
    recommendations JSON,
    is_read INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_elder ON alerts(elder_id, is_read);

-- Configurable Alert Rules (The "Brain" Configuration)
CREATE TABLE IF NOT EXISTS alert_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_name TEXT NOT NULL UNIQUE,
    rule_type TEXT NOT NULL,  -- 'sleep', 'adl', 'vital', 'composite'
    enabled INTEGER DEFAULT 1,
    description TEXT,
    thresholds TEXT,  -- JSON: {"deep_sleep_min": 15, "night_toilet_max": 3}
    conditions TEXT,  -- JSON: {"requires_condition": "hypertension"}
    alert_severity TEXT DEFAULT 'medium',
    alert_message_template TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 9. Alert Rules V2 (Disease Driven)
CREATE TABLE IF NOT EXISTS alert_rules_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_name TEXT NOT NULL,
    required_condition TEXT, -- e.g. 'hypertension'
    conditions TEXT,         -- JSON: { logic: 'AND', rules: [...] }
    alert_severity TEXT,     -- 'low', 'medium', 'high'
    alert_message TEXT,
    enabled INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rules_v2_enabled ON alert_rules_v2(enabled);

-- 10. Training History
CREATE TABLE IF NOT EXISTS training_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    elder_id TEXT,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_type TEXT,
    epochs INTEGER,
    accuracy REAL,
    status TEXT,
    metadata JSON
);

CREATE INDEX IF NOT EXISTS idx_training_hist_elder ON training_history(elder_id);

-- 10b. Hard-Negative Review Queue (active-learning triage)
CREATE TABLE IF NOT EXISTS hard_negative_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    elder_id TEXT NOT NULL,
    room TEXT NOT NULL,
    record_date DATE NOT NULL,
    timestamp_start DATETIME NOT NULL,
    timestamp_end DATETIME NOT NULL,
    duration_minutes REAL,
    reason TEXT NOT NULL,
    score REAL NOT NULL,
    suggested_label TEXT,
    source TEXT DEFAULT 'hard_negative_miner_v1',
    status TEXT DEFAULT 'open', -- open, queued, reviewed, applied, dismissed
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_hnq_unique
ON hard_negative_queue(elder_id, room, timestamp_start, timestamp_end, reason);
CREATE INDEX IF NOT EXISTS idx_hnq_lookup
ON hard_negative_queue(elder_id, status, created_at);

-- 11. Activity Segments (Consolidated view of adl_history for efficient UI rendering)
-- Generated by backend after prediction, aggregates consecutive same-activity events into blocks
CREATE TABLE IF NOT EXISTS activity_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    elder_id TEXT REFERENCES elders(elder_id),
    room TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    duration_minutes REAL,
    avg_confidence REAL,
    event_count INTEGER,  -- Number of raw events consolidated
    record_date DATE NOT NULL,
    is_corrected INTEGER DEFAULT 0,  -- True if segment contains corrected data
    correction_source TEXT,  -- 'manual', 'golden_sample', NULL
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_segments_date ON activity_segments(elder_id, record_date);
CREATE INDEX IF NOT EXISTS idx_segments_time ON activity_segments(elder_id, start_time);
CREATE INDEX IF NOT EXISTS idx_segments_unique ON activity_segments(elder_id, room, start_time);

-- 12. Household Behavior (Global State)
-- Stores the high-level state of the home at specific timestamps (e.g., every minute)
CREATE TABLE IF NOT EXISTS household_behavior (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    elder_id TEXT, -- Logical ID for the household (usually the primary elder ID)
    timestamp DATETIME NOT NULL,
    state TEXT NOT NULL, -- 'empty_home', 'home_active', 'home_quiet', 'social_interaction'
    confidence REAL,
    supporting_evidence JSON, -- e.g. {"entrance": "out", "silent_rooms": ["bedroom", "kitchen"]}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_household_ts ON household_behavior(elder_id, timestamp);

-- 13. Household Segments (Consolidated UI Blocks)
-- Just like activity_segments, but for the global timeline
CREATE TABLE IF NOT EXISTS household_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    elder_id TEXT,
    state TEXT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    duration_minutes REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_household_seg_unique ON household_segments(elder_id, start_time);

-- 14. Household Configuration (Rule Panel)
-- Stores user-tunable parameters for behavior logic
CREATE TABLE IF NOT EXISTS household_config (
    key TEXT PRIMARY KEY, -- e.g. 'empty_home_silence_threshold'
    value TEXT, -- Stored as string, parsed by backend
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Seed default configuration
INSERT OR IGNORE INTO household_config (key, value, description) VALUES 
('empty_home_silence_threshold_min', '15', 'Minutes of silence required to confirm Empty Home'),
('empty_home_ignore_rooms', '[]', 'JSON list of rooms to ignore for silence check'),
('enable_empty_home_detection', 'true', 'Master switch for empty home logic'),
('household_type', 'single', 'Type of household: single or double (enables conflict resolution)');

-- ============================================================
-- INTELLIGENCE PHASE TABLES (Beta 5 Add-Ons)
-- ============================================================

-- 15. Trajectory Events (Cross-Room Movement Tracking)
-- Generated by TrajectoryEngine, links room activities into movement paths
CREATE TABLE IF NOT EXISTS trajectory_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    elder_id TEXT REFERENCES elders(elder_id),
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    path TEXT NOT NULL, -- e.g. "Bedroom->Hallway->Kitchen"
    primary_activity TEXT, -- Main activity at destination
    room_sequence JSON, -- Detailed: [{"room": "Bedroom", "ts": "...", "activity": "..."},...]
    duration_minutes REAL,
    confidence REAL,
    record_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trajectory_date ON trajectory_events(elder_id, record_date);
CREATE INDEX IF NOT EXISTS idx_trajectory_time ON trajectory_events(elder_id, start_time);

-- 16. Context Episodes (ML-Based Household State Classification)
-- Parallel to household_segments but from ML classifier
CREATE TABLE IF NOT EXISTS context_episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    elder_id TEXT,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    context_label TEXT NOT NULL, -- 'Home_Alone', 'Home_Double', 'Guest_Present', 'Empty_Home', 'Emergency'
    confidence REAL,
    model_version TEXT,
    features_used JSON, -- For explainability
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_context_time ON context_episodes(elder_id, start_time);

-- 17. Routine Anomalies (Pattern Learning Detections)
-- Generated by PatternWatchdog when deviations from learned routine are detected
CREATE TABLE IF NOT EXISTS routine_anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    elder_id TEXT REFERENCES elders(elder_id),
    detection_date DATE NOT NULL,
    anomaly_type TEXT, -- 'late_wakeup', 'skipped_meal', 'unusual_night_activity', 'routine_shift'
    anomaly_score REAL, -- 0-1, higher = more significant deviation
    description TEXT,
    baseline_value TEXT, -- What the "normal" value was
    observed_value TEXT, -- What was observed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_anomaly_date ON routine_anomalies(elder_id, detection_date);

-- 18. Model Training History
CREATE TABLE IF NOT EXISTS model_training_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    elder_id TEXT NOT NULL,
    room TEXT NOT NULL,
    model_type TEXT,
    accuracy REAL,
    samples_count INTEGER,
    epochs INTEGER,
    status TEXT,
    error_message TEXT
);
