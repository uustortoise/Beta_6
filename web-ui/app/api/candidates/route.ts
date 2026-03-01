import { getDb } from '../../lib/db';
import { withErrorHandling, successResponse } from '../../lib/errors';

export const GET = withErrorHandling(async () => {
    const db = getDb();

    // Query for anomalies or low confidence predictions
    // We look for explicit 'low_confidence' activity OR is_anomaly flag
    // Also assuming 'low_confidence' threshold might be implicit if activity is labeled as such
    const result = await db.query(`
        SELECT 
            elder_id,
            timestamp,
            room,
            activity_type,
            confidence,
            is_anomaly
        FROM adl_history 
        WHERE 
            activity_type = 'low_confidence' 
            OR COALESCE(is_anomaly, 0) = 1
            OR confidence < 0.6
        ORDER BY timestamp DESC
        LIMIT 1000
    `);
    const rows = result.rows;

    const candidates = rows.map((r: any) => ({
        resident_id: r.elder_id, // Map database column to frontend expectation
        timestamp: r.timestamp,
        room: r.room,
        activity: r.activity_type,
        confidence: r.confidence,
        reason: r.is_anomaly ? 'Anomaly' : 'Low Confidence'
    }));

    return successResponse({ candidates });
});
