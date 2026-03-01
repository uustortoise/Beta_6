-- Performance Enhancement Phase 1: Database Indexes
-- Generated: Feb 7, 2026
-- Run with: psql -f migrate_add_indexes.sql

-- For getActivityTimelineForDate() frequent queries
CREATE INDEX IF NOT EXISTS idx_segments_lookup 
ON activity_segments(elder_id, record_date, start_time);

-- For alert filtering on dashboard (partial index for unread only)
CREATE INDEX IF NOT EXISTS idx_alerts_unread 
ON alerts(elder_id, is_read, created_at DESC) 
WHERE is_read = 0;

-- For correction history filtering by date
CREATE INDEX IF NOT EXISTS idx_correction_date 
ON correction_history(corrected_at DESC);

-- For adl_history queries by record_date (fallback queries)
CREATE INDEX IF NOT EXISTS idx_adl_record_date 
ON adl_history(elder_id, record_date, timestamp);

-- Verify indexes created
SELECT indexname, tablename 
FROM pg_indexes 
WHERE schemaname = 'public' 
  AND indexname IN (
    'idx_segments_lookup', 
    'idx_alerts_unread', 
    'idx_correction_date', 
    'idx_adl_record_date'
  );
