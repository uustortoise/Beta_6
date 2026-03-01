import { cache } from 'react';
import { getDb, timedQuery, QUERY_TIMEOUTS } from './db';

// Interfaces (Matches Schema roughly)
export interface ResidentProfile {
    id: string;
    risk_level: 'low' | 'medium' | 'high' | 'unknown';
    name?: string;
    age?: number;
    last_updated: string;
    personal_info: {
        full_name: string;
        date_of_birth?: string;
        age?: number;
        gender?: string;
        blood_type?: string;
    };
    medical_history: {
        chronic_conditions: string[];
        medications: any[];
        allergies: string[];
        surgeries: string[];
    };
    emergency_contacts: {
        id: number;
        name: string;
        relationship: string;
        phone: string;
        email?: string;
        is_primary: boolean;
    }[];
}

export async function getResidents(): Promise<ResidentProfile[]> {
    const db = getDb();
    const result = await db.query('SELECT elder_id, full_name, date_of_birth, gender, blood_type, last_updated FROM elders');
    const elders = result.rows;

    return elders.map(elder => {
        // Calculate Age
        const dob = new Date(elder.date_of_birth);
        const ageDifMs = Date.now() - dob.getTime();
        const ageDate = new Date(ageDifMs);
        const age = Math.abs(ageDate.getUTCFullYear() - 1970);

        return {
            id: elder.elder_id,
            name: elder.full_name,
            age: isNaN(age) ? 0 : age,
            gender: elder.gender,
            risk_level: 'unknown',
            last_updated: elder.last_updated,
            personal_info: {
                full_name: elder.full_name,
                date_of_birth: elder.date_of_birth,
                gender: elder.gender,
                blood_type: elder.blood_type
            },
            medical_history: { chronic_conditions: [], medications: [], allergies: [], surgeries: [] },
            emergency_contacts: []
        };
    });
}

export const getResidentById = cache(async (id: string): Promise<ResidentProfile | null> => {
    const db = getDb();

    // Single optimized query with JOINs
    // Note: GROUP_CONCAT -> STRING_AGG in Postgres
    const result = await db.query(`
        SELECT 
            e.*,
            STRING_AGG(m.category || ':' || m.data::text, '|||') as medical_data,
            STRING_AGG(c.id || '|' || c.name || '|' || c.relationship || '|' || c.phone || '|' || COALESCE(c.email, '') || '|' || c.is_primary, '|||') as contacts_data,
            (SELECT COUNT(*) FROM alerts WHERE elder_id = e.elder_id AND severity IN ('high', 'critical') AND is_read = 0) as alert_count
        FROM elders e
        LEFT JOIN medical_history m ON e.elder_id = m.elder_id
        LEFT JOIN emergency_contacts c ON e.elder_id = c.elder_id
        WHERE e.elder_id = $1
        GROUP BY e.elder_id
    `, [id]);

    const row = result.rows[0];
    if (!row) return null;

    // Calculate Age
    const dob = new Date(row.date_of_birth);
    const ageDifMs = Date.now() - dob.getTime();
    const ageDate = new Date(ageDifMs);
    const age = Math.abs(ageDate.getUTCFullYear() - 1970);

    // Parse Medical History from concatenated data
    const medical_history: any = {
        chronic_conditions: [], medications: [], allergies: [], surgeries: []
    };

    if (row.medical_data) {
        const medicalEntries = row.medical_data.split('|||');
        medicalEntries.forEach((entry: string) => {
            const [category, dataStr] = entry.split(':');
            try {
                const data = JSON.parse(dataStr);
                if (category === 'chronic') medical_history.chronic_conditions = data;
                if (category === 'medication') medical_history.medications = data;
                if (category === 'allergy') medical_history.allergies = data;
                if (category === 'surgery') medical_history.surgeries = data;
            } catch (e) { }
        });
    }

    // Parse Contacts from concatenated data
    const emergency_contacts = [];
    if (row.contacts_data) {
        const contactEntries = row.contacts_data.split('|||');
        for (const entry of contactEntries) {
            const [id, name, relationship, phone, email, is_primary] = entry.split('|');
            emergency_contacts.push({
                id: parseInt(id),
                name,
                relationship,
                phone,
                email: email || undefined,
                is_primary: is_primary === '1'
            });
        }
    }

    const risk_level = row.alert_count > 0 ? 'high' : 'low';

    return {
        id: row.elder_id,
        risk_level: risk_level,
        last_updated: row.last_updated,
        personal_info: {
            full_name: row.full_name,
            date_of_birth: row.date_of_birth,
            age: isNaN(age) ? 0 : age,
            gender: row.gender,
            blood_type: row.blood_type
        },
        medical_history,
        emergency_contacts
    };
});

export interface Alert {
    id: number;
    alert_date: string;
    alert_type: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    title: string;
    message: string;
    recommendations: string[];
    is_read: boolean;
}

export async function getAlerts(id: string): Promise<Alert[]> {
    const db = getDb();
    const result = await db.query('SELECT id, alert_date, alert_type, severity, title, message, recommendations, is_read, created_at FROM alerts WHERE elder_id = $1 ORDER BY created_at DESC', [id]);
    const rows = result.rows;

    return rows.map(row => ({
        id: row.id,
        alert_date: row.alert_date,
        alert_type: row.alert_type,
        severity: row.severity,
        title: row.title,
        message: row.message,
        recommendations: typeof row.recommendations === 'string' ? JSON.parse(row.recommendations || '[]') : row.recommendations,
        is_read: !!row.is_read
    }));
}

export async function getAllAlerts(): Promise<Alert[]> {
    const db = getDb();
    const result = await db.query('SELECT id, elder_id, alert_date, alert_type, severity, title, message, recommendations, is_read, created_at FROM alerts ORDER BY created_at DESC LIMIT 50');
    const rows = result.rows;

    return rows.map(row => ({
        id: row.id,
        alert_date: row.alert_date,
        alert_type: row.alert_type,
        severity: row.severity,
        title: row.title,
        message: row.message,
        recommendations: typeof row.recommendations === 'string' ? JSON.parse(row.recommendations || '[]') : row.recommendations,
        is_read: !!row.is_read
    }));
}

export interface SleepData {
    analysis_date: string;
    duration_hours: number;
    efficiency_percent: number;
    quality_score: number;
    stage_breakdown: { Light: number, Deep: number, REM: number, Awake: number };
    sleep_periods: any[];
    insights: string[];
    // Extended fields for SleepChart
    seven_day_average?: {
        duration_hours: number;
        efficiency: number;
        score: number;
    };
    daily_history?: { date: string; duration_hours: number; quality_score: number }[];
    grade?: string;
}

export const getSleepData = cache(async (id: string): Promise<SleepData | null> => {
    const db = getDb();

    // Get last 7 days of sleep data for trend chart
    const result = await db.query(`
        SELECT analysis_date, duration_hours, efficiency_percent, 
               quality_score, sleep_stages, insights 
        FROM sleep_analysis 
        WHERE elder_id = $1 
        ORDER BY analysis_date DESC 
        LIMIT 7
    `, [id]);
    const rows = result.rows;

    if (!rows || rows.length === 0) {
        return null;
    }

    // Latest record for primary display
    const latest = rows[0];
    const stages = typeof latest.sleep_stages === 'string' ? JSON.parse(latest.sleep_stages || '{}') : latest.sleep_stages;

    // Build daily history for trend chart (oldest first for proper chart ordering)
    const daily_history = [...rows].reverse().map(r => ({
        date: r.analysis_date,
        duration_hours: r.duration_hours || 0,
        quality_score: r.quality_score || 0
    }));

    // Calculate 7-day average
    const avgDuration = rows.reduce((sum, r) => sum + (r.duration_hours || 0), 0) / rows.length;
    const avgEfficiency = rows.reduce((sum, r) => sum + (r.efficiency_percent || 0), 0) / rows.length;
    const avgScore = rows.reduce((sum, r) => sum + (r.quality_score || 0), 0) / rows.length;

    // Calculate grade based on quality score
    let grade = 'C';
    if (avgScore >= 90) grade = 'A+';
    else if (avgScore >= 80) grade = 'A';
    else if (avgScore >= 70) grade = 'B';
    else if (avgScore >= 60) grade = 'C';
    else grade = 'D';

    // Mock periods for timeline chart until we have granular data
    const sleep_periods = [
        { state: 'Awake', start: '22:00', end: '22:30' },
        { state: 'Light', start: '22:30', end: '23:30' },
        { state: 'Deep', start: '23:30', end: '01:00' },
        { state: 'Light', start: '01:00', end: '05:00' },
        { state: 'Awake', start: '05:00', end: '06:00' }
    ];

    return {
        analysis_date: latest.analysis_date,
        duration_hours: latest.duration_hours,
        efficiency_percent: latest.efficiency_percent,
        quality_score: latest.quality_score,
        stage_breakdown: stages,
        sleep_periods: sleep_periods,
        insights: JSON.parse(latest.insights || '[]'),
        seven_day_average: {
            duration_hours: Math.round(avgDuration * 10) / 10,
            efficiency: Math.round(avgEfficiency),
            score: Math.round(avgScore)
        },
        daily_history: daily_history,
        grade: grade
    };
});



// Create new resident profile
export async function createResident(id: string, initialData: Partial<ResidentProfile>): Promise<boolean> {
    const db = getDb();
    const dateOfBirth = (initialData.personal_info?.date_of_birth || new Date().toISOString()).split('T')[0];
    const profileData = {
        age: initialData.personal_info?.age ?? null
    };

    try {
        await db.query(`
            INSERT INTO elders (elder_id, full_name, gender, date_of_birth, profile_data, last_updated)
            VALUES ($1, $2, $3, $4, $5::jsonb, CURRENT_TIMESTAMP)
        `, [
            id,
            initialData.personal_info?.full_name || id,
            initialData.personal_info?.gender || 'Unknown',
            dateOfBirth,
            JSON.stringify(profileData)
        ]);
        return true;
    } catch (e: any) {
        console.error("Create failed", e);
        if (e?.code === '23505') {
            throw new Error(`Resident ${id} already exists`);
        }
        throw e;
    }
}

export async function updateResidentProfile(id: string, updates: Partial<ResidentProfile>): Promise<boolean> {
    const db = getDb();
    try {
        if (updates.personal_info?.full_name) {
            await db.query('UPDATE elders SET full_name = $1, last_updated = CURRENT_TIMESTAMP WHERE elder_id = $2',
                [updates.personal_info.full_name, id]);
        }
        return true;
    } catch (e) {
        console.error("Update failed", e);
        return false;
    }
}

// Stats for dashboard
export interface DashboardStats {
    totalResidents: number;
    activeAlerts: number;
    avgVitality: number;
    avgVitalityTrend: 'up' | 'down' | 'stable';
    complianceRate: number;
}

export async function getDashboardStats(): Promise<DashboardStats> {
    // Dashboard is user-facing - use strict 2s timeout for fast response
    const totalResult = await timedQuery(
        'SELECT COUNT(*) as c FROM elders',
        [],
        QUERY_TIMEOUTS.SIMPLE_LOOKUP  // 2s
    );
    const alertsResult = await timedQuery(
        'SELECT COUNT(*) as c FROM alerts WHERE is_read = 0',
        [],
        QUERY_TIMEOUTS.SIMPLE_LOOKUP  // 2s
    );

    return {
        totalResidents: parseInt(totalResult.rows[0].c),
        activeAlerts: parseInt(alertsResult.rows[0].c),
        avgVitality: 85, // Placeholder
        avgVitalityTrend: 'stable',
        complianceRate: 98
    };
}

// --- Missing Fetchers Added for Beta_3 ---

export interface ADLRecord {
    date: string;
    mobility_index: number;
    diversity_score: number;
    night_activity_count: number;
    bathroom_night_visits: number;
}
export interface ADLHistory {
    last_updated: string;
    daily_records: ADLRecord[];
    anomalies?: string[];
    trends?: { mobility_slope?: number };
}
export const getADLHistory = cache(async (id: string): Promise<ADLHistory | null> => {
    const db = getDb();
    // Count actual activity types from the database
    const result = await db.query(`
        SELECT record_date, 
               COUNT(*) as total_events,
               COUNT(DISTINCT room) as unique_rooms,
               COUNT(DISTINCT activity_type) as unique_activities,
               SUM(CASE WHEN activity_type IN ('sleep', 'nap') THEN 1 ELSE 0 END) as count_sleep,
               SUM(CASE WHEN activity_type IN ('room_normal_use', 'livingroom_normal_use', 'bathroom_normal_use') THEN 1 ELSE 0 END) as count_normal,
               SUM(CASE WHEN activity_type = 'kitchen normal use' THEN 1 ELSE 0 END) as count_cooking,
               SUM(CASE WHEN activity_type IN ('bathroom_normal_use', 'Bath') THEN 1 ELSE 0 END) as count_bath
        FROM adl_history
        WHERE elder_id = $1
        GROUP BY record_date
        ORDER BY record_date ASC
    `, [id]);
    const rows = result.rows;

    if (rows.length === 0) return null;

    const daily_records = rows.map(r => ({
        date: r.record_date,
        // Use actual counts for the stacked bar chart
        count_sleep: r.count_sleep || 0,
        count_normal: r.count_normal || 0,
        count_cooking: r.count_cooking || 0,
        count_bath: r.count_bath || 0,
        // Calculate derived metrics (normalized 0-1)
        mobility_index: Math.min(1.0, (r.unique_rooms || 0) / 5.0),
        diversity_score: Math.min(1.0, (r.unique_activities || 0) / 8.0),
        night_activity_count: r.total_events,
        bathroom_night_visits: r.count_bath
    }));

    return {
        last_updated: new Date().toISOString(),
        daily_records,
        anomalies: [] // Placeholder
    };
});

export interface ICOPEStatus {
    last_updated: string;
    overall_score: number;
    trend: 'improving' | 'stable' | 'declining';
    assessment_summary: string;
    priority_actions: string[];
    domains: { [key: string]: { score: number, rating: string } };
}
export const getICOPEStatus = cache(async (id: string): Promise<ICOPEStatus | null> => {
    const db = getDb();
    const result = await db.query(`
        SELECT assessment_date, locomotion_score, cognition_score,
               psychological_score, sensory_score, vitality_score,
               overall_score, recommendations, trend
        FROM icope_assessments 
        WHERE elder_id = $1 
        ORDER BY assessment_date DESC 
        LIMIT 1
    `, [id]);
    const row = result.rows[0];

    if (!row) return null;

    return {
        last_updated: row.assessment_date,
        overall_score: row.overall_score,
        trend: row.trend as any,
        assessment_summary: "Assessment based on recent data.",
        priority_actions: typeof row.recommendations === 'string' ? JSON.parse(row.recommendations || '[]') : row.recommendations,
        domains: {
            locomotion: { score: row.locomotion_score, rating: 'Satisfactory' },
            cognition: { score: row.cognition_score, rating: 'Satisfactory' },
            vitality: { score: row.vitality_score, rating: 'Satisfactory' },
            psychological: { score: row.psychological_score, rating: 'Satisfactory' },
            sensory: { score: row.sensory_score, rating: 'Satisfactory' }
        }
    };
});

export interface TimelineEvent {
    timestamp: string;
    room: string;
    activity: string;
    confidence: number;
    is_anomaly: boolean;
}

// Get available dates for activity timeline
// IMPORTANT: Merges dates from BOTH sources to ensure complete visibility
// regardless of whether data has been processed into optimized segments
export async function getAvailableTimelineDates(id: string): Promise<string[]> {
    const db = getDb();

    // Collect dates from optimized activity_segments
    // Collect dates from optimized activity_segments
    const segmentResult = await db.query(`
        SELECT DISTINCT TO_CHAR(record_date, 'YYYY-MM-DD') as record_date 
        FROM activity_segments 
        WHERE elder_id = $1
    `, [id]);

    // Collect dates from raw adl_history
    const adlResult = await db.query(`
        SELECT DISTINCT TO_CHAR(record_date, 'YYYY-MM-DD') as record_date 
        FROM adl_history 
        WHERE elder_id = $1
    `, [id]);

    // Merge into Set for deduplication
    const allDates = new Set<string>();
    segmentResult.rows.forEach(r => allDates.add(r.record_date));
    adlResult.rows.forEach(r => allDates.add(r.record_date));

    // Return sorted descending (newest first)
    return Array.from(allDates).sort((a, b) => b.localeCompare(a));
}

// Get activity timeline for a specific date (or latest if not specified)
export async function getActivityTimelineForDate(id: string, date?: string): Promise<TimelineEvent[]> {
    const db = getDb();

    // Determine which date to use
    let targetDate = date;

    if (!targetDate) {
        // Get the most recent record date
        const latestResult = await db.query(`
            SELECT MAX(record_date) as latest FROM activity_segments WHERE elder_id = $1
        `, [id]);
        const latestDate = latestResult.rows[0];

        if (!latestDate?.latest) {
            // Try adl_history
            const rawLatestResult = await db.query(`
                SELECT MAX(record_date) as latest FROM adl_history WHERE elder_id = $1
            `, [id]);
            const rawLatest = rawLatestResult.rows[0];

            if (!rawLatest?.latest) return [];
            targetDate = rawLatest.latest;
        } else {
            targetDate = latestDate.latest;
        }
    }

    // Try activity_segments first
    const result = await db.query(`
        SELECT start_time, end_time, room, activity_type, avg_confidence, duration_minutes, event_count 
        FROM activity_segments 
        WHERE elder_id = $1 
          AND record_date = $2
        ORDER BY start_time ASC
    `, [id, targetDate]);
    const segmentRows = result.rows;

    if (segmentRows.length > 0) {
        return segmentRows.map(s => ({
            timestamp: s.start_time,
            end_time: s.end_time,
            room: s.room || 'Unknown',
            activity: s.activity_type,
            confidence: Number(s.avg_confidence) || 1.0,
            duration_minutes: Number(s.duration_minutes) || 0,
            event_count: Number(s.event_count) || 1,
            is_anomaly: false
        }));
    }

    // Fallback to adl_history
    // Remove LIMIT to see full day, but be mindful of payload size
    // FIX: Fetch timestamp as string to avoid automatic UTC conversion by pg driver
    const adlResult = await db.query(`
        SELECT 
            *, 
            TO_CHAR(timestamp, 'YYYY-MM-DD"T"HH24:MI:SS.MS') as timestamp_str 
        FROM adl_history 
        WHERE elder_id = $1 
          AND record_date = $2
        ORDER BY timestamp ASC
    `, [id, targetDate]);
    const adlRows = adlResult.rows;

    // Use a Map to track active segments per room to handle interleaved data
    // e.g. Sleep in Bedroom (00:00) -> Unoccupied in Kitchen (00:05) -> Sleep in Bedroom (00:10)
    const mergedSegments: TimelineEvent[] = [];
    if (adlRows.length === 0) return [];

    // Should merge the Sleep records despite the Kitchen record in between.
    const activeSegmentsByRoom = new Map<string, any>();
    const MERGE_THRESHOLD_MS = 5 * 60 * 1000; // 5 minute gap allowed (matches backend)

    for (const row of adlRows) {
        // Parse 'timestamp_str' as LOCAL time (browser will assume local if no Z provided)
        // This fixes the 08:29 sleep bug (db has 00:29, parsed as UTC -> +8h display)
        const rowTimeStr = row.timestamp_str || row.timestamp;
        const rowTime = new Date(rowTimeStr).getTime();
        const room = row.room || 'Unknown';

        let currentSegment = activeSegmentsByRoom.get(room);

        if (currentSegment) {
            const lastTime = new Date(currentSegment.end_time || currentSegment.timestamp).getTime();
            const timeDiff = rowTime - lastTime;

            // Check merge conditions (Room matches by definition of Map key)
            const sameActivity = row.activity_type === currentSegment.activity;
            const withinTime = timeDiff <= MERGE_THRESHOLD_MS;

            if (sameActivity && withinTime) {
                // Extend current segment
                currentSegment.end_time = rowTimeStr;
                currentSegment.event_count = (currentSegment.event_count || 1) + 1;
                continue;
            } else {
                // Finalize current segment and push
                const start = new Date(currentSegment.timestamp).getTime();
                const end = new Date(currentSegment.end_time || currentSegment.timestamp).getTime();
                currentSegment.duration_minutes = Math.max(1, Math.round((end - start) / 60000));

                mergedSegments.push(currentSegment);

                // Remove from map, will start new below
                activeSegmentsByRoom.delete(room);
            }
        }

        // Start new segment if not merged
        const newSegment = {
            timestamp: rowTimeStr, // Use string
            end_time: rowTimeStr,
            room: room,
            activity: row.activity_type,
            confidence: Number(row.confidence) || 1.0,
            is_anomaly: !!row.is_anomaly,
            event_count: 1
        };
        activeSegmentsByRoom.set(room, newSegment);
    }

    // Finalize all remaining open segments
    activeSegmentsByRoom.forEach((segment) => {
        const start = new Date(segment.timestamp).getTime();
        const end = new Date(segment.end_time || segment.timestamp).getTime();
        segment.duration_minutes = Math.max(1, Math.round((end - start) / 60000));
        mergedSegments.push(segment);
    });

    return mergedSegments;
}

// Legacy function - still works for backwards compatibility
export const getActivityTimeline = cache(async (id: string): Promise<TimelineEvent[]> => {
    return getActivityTimelineForDate(id);
});

export interface ActivityLogItem {
    id: string; // resident ID
    timestamp: string;
    description: string;
    room: string;
    confidence: number;
}

export async function getRecentSystemActivities(): Promise<ActivityLogItem[]> {
    const db = getDb();
    const result = await db.query(`
        SELECT id, elder_id, alert_date, alert_type, severity, 
               title, message, is_read, created_at
        FROM alerts 
        ORDER BY created_at DESC 
        LIMIT 10
    `);
    const alerts = result.rows;

    return alerts.map(a => ({
        id: a.elder_id,
        timestamp: a.created_at,
        description: `Alert: ${a.title} (${a.severity})`,
        room: 'System',
        confidence: 1.0
    }));
}
