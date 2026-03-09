-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 1. Elders Metadata (Core)
CREATE TABLE IF NOT EXISTS elders (
    elder_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL, -- Matched to SQLite
    nickname TEXT,
    date_of_birth DATE,      -- Matched to SQLite
    gender TEXT,             -- Matched to SQLite
    blood_type TEXT,         -- Matched to SQLite
    profile_data JSONB,      -- Includes Beta 6.1 typed resident_home_context contract
    created_at TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP -- Matched to SQLite
);

-- 2. ADL History (Hypertable - Time Series)
CREATE TABLE IF NOT EXISTS adl_history (
    id SERIAL, -- Helper ID
    elder_id TEXT NOT NULL REFERENCES elders(elder_id) ON DELETE CASCADE,
    timestamp TIMESTAMP NOT NULL,
    room TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    duration_minutes INTEGER, -- Matched to SQLite (Critical for inserts)
    source TEXT,
    sensor_features JSONB, 
    record_date DATE,
    is_corrected INTEGER DEFAULT 0,
    is_anomaly INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Composite PK for TimescaleDB compatibility
    CONSTRAINT pk_adl_history PRIMARY KEY (elder_id, timestamp, room)
);

-- Convert to Hypertable
SELECT create_hypertable('adl_history', 'timestamp', 
    chunk_time_interval => INTERVAL '7 days', 
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes for ADL History
CREATE INDEX IF NOT EXISTS idx_adl_elder_ts ON adl_history (elder_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_adl_room_ts ON adl_history (room, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_adl_sensor_features ON adl_history USING GIN (sensor_features);
CREATE INDEX IF NOT EXISTS idx_adl_anomaly ON adl_history(elder_id, is_anomaly);
CREATE INDEX IF NOT EXISTS idx_adl_corrected ON adl_history(elder_id, record_date, room, is_corrected);

-- 3. Correction History (Audit Trail)
CREATE TABLE IF NOT EXISTS correction_history (
    id SERIAL PRIMARY KEY,
    adl_history_id INTEGER,
    elder_id TEXT NOT NULL REFERENCES elders(elder_id),
    room TEXT NOT NULL,
    timestamp_start TIMESTAMP,
    timestamp_end TIMESTAMP,
    old_activity TEXT,
    new_activity TEXT NOT NULL,
    rows_affected INTEGER,
    corrected_by TEXT DEFAULT 'user',
    corrected_at TIMESTAMP DEFAULT NOW(),
    comments TEXT,
    
    is_deleted INTEGER DEFAULT 0,
    deleted_at TIMESTAMP,
    deleted_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_correction_elder ON correction_history(elder_id);
CREATE INDEX IF NOT EXISTS idx_correction_active ON correction_history(elder_id, is_deleted);

-- 4. Sensor Data (Raw - Hypertable)
CREATE TABLE IF NOT EXISTS sensor_data (
    timestamp TIMESTAMP NOT NULL,
    elder_id TEXT NOT NULL,
    room TEXT,
    sensor_type TEXT,
    value DOUBLE PRECISION,
    metadata JSONB
);
SELECT create_hypertable('sensor_data', 'timestamp', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_sensor_lookup ON sensor_data (elder_id, sensor_type, timestamp DESC);

-- 5. Medical History
CREATE TABLE IF NOT EXISTS medical_history (
    id SERIAL PRIMARY KEY,
    elder_id TEXT NOT NULL REFERENCES elders(elder_id),
    category TEXT,
    condition TEXT,
    notes TEXT,
    data JSONB,
    diagnosed_date DATE,
    recorded_date DATE,  -- Used by profile_service.py INSERT
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_medical_elder ON medical_history(elder_id);

-- 6. ICOPE Assessments
CREATE TABLE IF NOT EXISTS icope_assessments (
    id SERIAL PRIMARY KEY,
    elder_id TEXT NOT NULL REFERENCES elders(elder_id),
    assessment_date DATE NOT NULL,
    locomotion_score REAL,
    cognition_score REAL,
    psychological_score REAL,
    sensory_score REAL,
    vitality_score REAL,
    overall_score REAL,
    recommendations JSONB,
    trend TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_icope_latest ON icope_assessments(elder_id, assessment_date DESC);

-- 7. Sleep Analysis
CREATE TABLE IF NOT EXISTS sleep_analysis (
    id SERIAL PRIMARY KEY,
    elder_id TEXT NOT NULL REFERENCES elders(elder_id),
    analysis_date DATE NOT NULL,
    duration_hours REAL,
    efficiency_percent REAL,
    sleep_stages JSONB,
    quality_score REAL,
    insights TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(elder_id, analysis_date)
);
CREATE INDEX IF NOT EXISTS idx_sleep_latest ON sleep_analysis(elder_id, analysis_date DESC);

-- 8. Emergency Contacts
CREATE TABLE IF NOT EXISTS emergency_contacts (
    id SERIAL PRIMARY KEY,
    elder_id TEXT NOT NULL REFERENCES elders(elder_id),
    name TEXT NOT NULL,
    relationship TEXT,
    phone TEXT NOT NULL,
    email TEXT,
    is_primary INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_contacts_elder ON emergency_contacts(elder_id);

-- 9. Predictions (Legacy/Cache)
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    resident_id TEXT, -- Legacy name
    timestamp TIMESTAMP,
    room TEXT,
    activity TEXT,
    confidence REAL,
    is_anomaly INTEGER,
    UNIQUE(resident_id, timestamp, room)
);
CREATE INDEX IF NOT EXISTS idx_resident_ts ON predictions (resident_id, timestamp);

-- 10. Alerts
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    elder_id TEXT REFERENCES elders(elder_id),
    alert_date DATE DEFAULT CURRENT_DATE,
    alert_type TEXT,
    severity TEXT,
    title TEXT,
    message TEXT,
    recommendations JSONB,
    is_read INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_alerts_elder ON alerts(elder_id, is_read);

-- 11. Alert Rules
CREATE TABLE IF NOT EXISTS alert_rules (
    id SERIAL PRIMARY KEY,
    rule_name TEXT NOT NULL UNIQUE,
    rule_type TEXT NOT NULL,
    enabled INTEGER DEFAULT 1,
    description TEXT,
    thresholds JSONB,
    conditions JSONB,
    alert_severity TEXT DEFAULT 'medium',
    alert_message_template TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 12. Alert Rules V2
CREATE TABLE IF NOT EXISTS alert_rules_v2 (
    id SERIAL PRIMARY KEY,
    rule_name TEXT NOT NULL,
    required_condition TEXT,
    conditions JSONB,
    alert_severity TEXT,
    alert_message TEXT,
    enabled INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

-- 13. Training History
CREATE TABLE IF NOT EXISTS training_history (
    id SERIAL PRIMARY KEY,
    elder_id TEXT,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_type TEXT,
    epochs INTEGER,
    accuracy REAL,
    status TEXT,
    metadata JSONB
);

-- 14b. Hard-Negative Review Queue (active-learning triage)
CREATE TABLE IF NOT EXISTS hard_negative_queue (
    id SERIAL PRIMARY KEY,
    elder_id TEXT NOT NULL,
    room TEXT NOT NULL,
    record_date DATE NOT NULL,
    timestamp_start TIMESTAMP NOT NULL,
    timestamp_end TIMESTAMP NOT NULL,
    duration_minutes REAL,
    reason TEXT NOT NULL,
    score REAL NOT NULL,
    suggested_label TEXT,
    source TEXT DEFAULT 'hard_negative_miner_v1',
    status TEXT DEFAULT 'open',
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_hnq_unique
ON hard_negative_queue(elder_id, room, timestamp_start, timestamp_end, reason);
CREATE INDEX IF NOT EXISTS idx_hnq_lookup
ON hard_negative_queue(elder_id, status, created_at);

-- 14. Activity Segments (Aggregated)
CREATE TABLE IF NOT EXISTS activity_segments (
    id SERIAL PRIMARY KEY,
    elder_id TEXT REFERENCES elders(elder_id),
    room TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    duration_minutes REAL,
    avg_confidence REAL,
    event_count INTEGER,
    record_date DATE NOT NULL,
    is_corrected INTEGER DEFAULT 0,
    correction_source TEXT,
    is_anomaly INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_segments_date ON activity_segments(elder_id, record_date);
CREATE INDEX IF NOT EXISTS idx_segments_unique ON activity_segments(elder_id, room, start_time);

-- 15. Household Behavior
CREATE TABLE IF NOT EXISTS household_behavior (
    id SERIAL, -- Removed PRIMARY KEY from here
    elder_id TEXT,
    timestamp TIMESTAMP NOT NULL,
    state TEXT NOT NULL,
    confidence REAL,
    supporting_evidence JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (id, timestamp) -- Composite PK required for Hypertable
);
SELECT create_hypertable('household_behavior', 'timestamp', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

-- 16. Household Segments
CREATE TABLE IF NOT EXISTS household_segments (
    id SERIAL PRIMARY KEY,
    elder_id TEXT,
    state TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    duration_minutes REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 17. Household Config
CREATE TABLE IF NOT EXISTS household_config (
    key TEXT PRIMARY KEY,
    value TEXT,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 18. Trajectory Events
CREATE TABLE IF NOT EXISTS trajectory_events (
    id SERIAL PRIMARY KEY,
    elder_id TEXT REFERENCES elders(elder_id),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    path TEXT NOT NULL,
    primary_activity TEXT,
    room_sequence JSONB,
    duration_minutes REAL,
    confidence REAL,
    record_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_trajectory_date ON trajectory_events(elder_id, record_date);

-- 19. Context Episodes
CREATE TABLE IF NOT EXISTS context_episodes (
    id SERIAL PRIMARY KEY,
    elder_id TEXT,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    context_label TEXT NOT NULL,
    confidence REAL,
    model_version TEXT,
    features_used JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 20. Routine Anomalies
CREATE TABLE IF NOT EXISTS routine_anomalies (
    id SERIAL PRIMARY KEY,
    elder_id TEXT REFERENCES elders(elder_id),
    detection_date DATE NOT NULL,
    anomaly_type TEXT,
    anomaly_score REAL,
    description TEXT,
    baseline_value TEXT,
    observed_value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_anomaly_date ON routine_anomalies(elder_id, detection_date);

-- 21. Model Training History (Detailed)
CREATE TABLE IF NOT EXISTS model_training_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    elder_id TEXT NOT NULL,
    room TEXT NOT NULL,
    model_type TEXT,
    accuracy REAL,
    samples_count INTEGER,
    epochs INTEGER,
    status TEXT,
    error_message TEXT,
    data_start_time TIMESTAMP,
    data_end_time TIMESTAMP
);

-- Critical Fix for Application Compatibility: Implicit Boolean Casting
-- Allows the application to send Integers (0/1) to Boolean columns without code changes.
CREATE OR REPLACE FUNCTION int_to_bool(int) RETURNS boolean AS $$
BEGIN
    IF $1 = 0 THEN
        RETURN false;
    ELSE
        RETURN true;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

DO $$ BEGIN
    CREATE CAST (integer AS boolean) WITH FUNCTION int_to_bool(int) AS IMPLICIT;
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;
