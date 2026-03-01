================================================================================
CORRECTION PIPELINE DIAGNOSTIC SQL SCRIPTS
================================================================================
Run these queries in your PostgreSQL database to diagnose correction issues.

USAGE:
    psql -d your_database -f scripts/diagnostic_correction_pipeline.sql
    
Or run individual queries in your SQL client (pgAdmin, DBeaver, etc.)
================================================================================

-- =============================================================================
-- CHECK 1: ADL_HISTORY CORRECTIONS
-- Verify corrections are being saved to adl_history
-- =============================================================================

-- 1.1 Count total corrections by elder and date
SELECT 
    elder_id,
    record_date,
    room,
    COUNT(*) as corrected_row_count,
    MIN(timestamp) as earliest_correction,
    MAX(timestamp) as latest_correction,
    ARRAY_AGG(DISTINCT activity_type) as activities
FROM adl_history
WHERE is_corrected = 1
GROUP BY elder_id, record_date, room
ORDER BY latest_correction DESC
LIMIT 10;

-- 1.2 Recent corrections (last 24 hours)
SELECT 
    elder_id,
    room,
    timestamp,
    activity_type,
    confidence,
    record_date
FROM adl_history
WHERE is_corrected = 1
    AND created_at > NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC
LIMIT 20;


-- =============================================================================
-- CHECK 2: CORRECTION_HISTORY AUDIT TRAIL
-- Verify audit entries are being created
-- =============================================================================

-- 2.1 Recent audit entries
SELECT 
    id,
    elder_id,
    room,
    timestamp_start,
    timestamp_end,
    old_activity,
    new_activity,
    rows_affected,
    corrected_at,
    is_deleted
FROM correction_history
ORDER BY corrected_at DESC
LIMIT 20;

-- 2.2 Summary by date
SELECT 
    DATE(corrected_at) as correction_date,
    COUNT(*) as total_corrections,
    SUM(rows_affected) as total_rows_affected,
    COUNT(DISTINCT elder_id) as elders_affected
FROM correction_history
WHERE is_deleted = 0
GROUP BY DATE(corrected_at)
ORDER BY correction_date DESC
LIMIT 10;


-- =============================================================================
-- CHECK 3: ACTIVITY_SEGMENTS CORRECTION STATUS
-- Verify segments reflect corrections
-- =============================================================================

-- 3.1 Count corrected segments
SELECT 
    elder_id,
    record_date,
    room,
    COUNT(*) as corrected_segment_count,
    SUM(duration_minutes) as total_duration_minutes
FROM activity_segments
WHERE is_corrected = 1
GROUP BY elder_id, record_date, room
ORDER BY record_date DESC
LIMIT 20;

-- 3.2 Detailed corrected segments (last 10)
SELECT 
    elder_id,
    record_date,
    room,
    activity_type,
    start_time,
    end_time,
    is_corrected,
    correction_source,
    duration_minutes,
    created_at
FROM activity_segments
WHERE is_corrected = 1
ORDER BY created_at DESC
LIMIT 20;


-- =============================================================================
-- CHECK 4: ORPHANED CORRECTIONS
-- Find corrections in adl_history that don't have matching segments
-- =============================================================================

-- 4.1 Find adl_history corrections with NO matching segment
SELECT 
    ah.elder_id,
    ah.record_date,
    ah.room,
    ah.activity_type as adl_activity,
    ah.timestamp as adl_timestamp,
    ah.is_corrected as adl_is_corrected,
    seg.id as segment_id,
    seg.activity_type as seg_activity
FROM adl_history ah
LEFT JOIN activity_segments seg ON 
    ah.elder_id = seg.elder_id 
    AND ah.record_date = seg.record_date
    AND LOWER(REPLACE(REPLACE(ah.room, ' ', ''), '_', '')) = LOWER(REPLACE(REPLACE(seg.room, ' ', ''), '_', ''))
    AND ah.timestamp BETWEEN seg.start_time AND seg.end_time
WHERE ah.is_corrected = 1
    AND seg.id IS NULL
ORDER BY ah.timestamp DESC
LIMIT 20;

-- 4.2 Count orphaned corrections by date
SELECT 
    DATE(ah.timestamp) as correction_date,
    COUNT(*) as orphaned_count,
    COUNT(DISTINCT ah.elder_id) as elders_affected
FROM adl_history ah
LEFT JOIN activity_segments seg ON 
    ah.elder_id = seg.elder_id 
    AND ah.record_date = seg.record_date
    AND LOWER(REPLACE(REPLACE(ah.room, ' ', ''), '_', '')) = LOWER(REPLACE(REPLACE(seg.room, ' ', ''), '_', ''))
    AND ah.timestamp BETWEEN seg.start_time AND seg.end_time
WHERE ah.is_corrected = 1
    AND seg.id IS NULL
GROUP BY DATE(ah.timestamp)
ORDER BY correction_date DESC
LIMIT 10;


-- =============================================================================
-- CHECK 5: ACTIVITY MISMATCHES
-- Find cases where segment activity doesn't match adl_history
-- =============================================================================

SELECT 
    ah.elder_id,
    ah.record_date,
    ah.room,
    ah.timestamp as adl_timestamp,
    ah.activity_type as adl_activity,
    seg.activity_type as seg_activity,
    seg.start_time,
    seg.end_time,
    seg.is_corrected as seg_is_corrected
FROM adl_history ah
JOIN activity_segments seg ON 
    ah.elder_id = seg.elder_id 
    AND ah.record_date = seg.record_date
    AND LOWER(REPLACE(REPLACE(ah.room, ' ', ''), '_', '')) = LOWER(REPLACE(REPLACE(seg.room, ' ', ''), '_', ''))
    AND ah.timestamp BETWEEN seg.start_time AND seg.end_time
WHERE ah.is_corrected = 1
    AND ah.activity_type != seg.activity_type
ORDER BY ah.timestamp DESC
LIMIT 20;


-- =============================================================================
-- CHECK 6: TIMESTAMP DRIFT ANALYSIS
-- Check for timezone/rounding issues causing merge_asof to fail
-- =============================================================================

-- 6.1 Compare timestamps between adl_history and activity_segments
SELECT 
    ah.elder_id,
    ah.record_date,
    ah.room,
    ah.timestamp as adl_timestamp,
    ah.activity_type as adl_activity,
    seg.start_time as seg_start_time,
    seg.activity_type as seg_activity,
    EXTRACT(EPOCH FROM (ah.timestamp - seg.start_time)) as time_diff_seconds,
    CASE 
        WHEN ABS(EXTRACT(EPOCH FROM (ah.timestamp - seg.start_time))) > 10 THEN 'DRIFT DETECTED'
        ELSE 'OK'
    END as drift_status
FROM adl_history ah
JOIN activity_segments seg ON 
    ah.elder_id = seg.elder_id 
    AND ah.record_date = seg.record_date
    AND ah.timestamp BETWEEN seg.start_time AND seg.end_time
WHERE ah.is_corrected = 1
ORDER BY ABS(EXTRACT(EPOCH FROM (ah.timestamp - seg.start_time))) DESC
LIMIT 20;

-- 6.2 Check for timezone issues (timestamps should be in same timezone)
SELECT 
    ah.elder_id,
    ah.timestamp as adl_timestamp,
    TO_CHAR(ah.timestamp, 'TZ') as adl_timezone,
    seg.start_time as seg_timestamp,
    TO_CHAR(seg.start_time, 'TZ') as seg_timezone
FROM adl_history ah
JOIN activity_segments seg ON ah.elder_id = seg.elder_id
WHERE ah.is_corrected = 1
LIMIT 5;


-- =============================================================================
-- CHECK 7: PIPELINE HEALTH SUMMARY
-- Overall statistics to assess pipeline health
-- =============================================================================

WITH stats AS (
    SELECT 
        (SELECT COUNT(*) FROM adl_history WHERE is_corrected = 1) as adl_corrections,
        (SELECT COUNT(*) FROM correction_history WHERE is_deleted = 0) as audit_entries,
        (SELECT COUNT(*) FROM activity_segments WHERE is_corrected = 1) as segment_corrections,
        (SELECT COUNT(*) FROM adl_history ah
         LEFT JOIN activity_segments seg ON ah.elder_id = seg.elder_id 
             AND ah.record_date = seg.record_date
             AND LOWER(REPLACE(REPLACE(ah.room, ' ', ''), '_', '')) = LOWER(REPLACE(REPLACE(seg.room, ' ', ''), '_', ''))
             AND ah.timestamp BETWEEN seg.start_time AND seg.end_time
         WHERE ah.is_corrected = 1 AND seg.id IS NULL) as orphaned_count
)
SELECT 
    adl_corrections,
    audit_entries,
    segment_corrections,
    orphaned_count,
    CASE 
        WHEN adl_corrections = 0 THEN 'NO_DATA: No corrections to analyze'
        WHEN orphaned_count > 0 THEN 'CRITICAL: Orphaned corrections detected'
        WHEN ABS(adl_corrections - segment_corrections) > adl_corrections * 0.1 THEN 'WARNING: Count mismatch'
        ELSE 'HEALTHY: Pipeline functioning correctly'
    END as pipeline_status
FROM stats;


-- =============================================================================
-- CHECK 8: SPECIFIC ELDER/ROOM/DATE INVESTIGATION
-- Use these queries to investigate specific cases
-- =============================================================================

-- Uncomment and modify these queries to investigate specific issues:

/*
-- Replace 'ELDER_ID' with actual elder ID
-- Replace '2026-02-06' with actual date

-- Check all adl_history for a specific elder/date
SELECT * FROM adl_history 
WHERE elder_id = 'ELDER_ID' 
    AND record_date = '2026-02-06'
ORDER BY timestamp;

-- Check all segments for a specific elder/date
SELECT * FROM activity_segments 
WHERE elder_id = 'ELDER_ID' 
    AND record_date = '2026-02-06'
ORDER BY start_time;

-- Check correction history for specific elder
SELECT * FROM correction_history 
WHERE elder_id = 'ELDER_ID'
ORDER BY corrected_at DESC;
*/


-- =============================================================================
-- END OF DIAGNOSTIC SCRIPTS
-- =============================================================================
