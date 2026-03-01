
-- Indexes for Dashboard Performance (Timeline & History)
CREATE INDEX IF NOT EXISTS idx_adl_history_elder_date ON adl_history(elder_id, record_date);
CREATE INDEX IF NOT EXISTS idx_adl_history_elder_room_corrected ON adl_history(elder_id, room, is_corrected);

-- Indexes for Trajectory Analysis
CREATE INDEX IF NOT EXISTS idx_trajectory_events_elder_date ON trajectory_events(elder_id, record_date);

-- Indexes for Household Context
CREATE INDEX IF NOT EXISTS idx_household_segments_elder_time ON household_segments(elder_id, start_time);

-- Optimization for Golden Samples query
CREATE INDEX IF NOT EXISTS idx_adl_is_corrected_timestamp ON adl_history(elder_id, room, is_corrected, timestamp);
