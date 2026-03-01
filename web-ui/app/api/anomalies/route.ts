import { NextResponse } from 'next/server';
import { getDb } from '../../lib/db';
import { withErrorHandling, ValidationError, successResponse } from '../../lib/errors';

export const dynamic = 'force-dynamic';

export const GET = withErrorHandling(async (request: Request) => {
  const { searchParams } = new URL(request.url);
  const elderId = searchParams.get('elderId');
  const days = parseInt(searchParams.get('days') || '7');

  if (!elderId) {
    throw new ValidationError('elderId is required');
  }

  const db = getDb();

  // Get recent anomalies
  const query = `
    SELECT id, detection_date, anomaly_type, anomaly_score, 
           description, baseline_value, observed_value, created_at
    FROM routine_anomalies
    WHERE elder_id = $1
    ORDER BY detection_date DESC, anomaly_score DESC
    LIMIT $2
  `;

  const anomaliesResult = await db.query(query, [elderId, days * 5]);
  const anomalies = anomaliesResult.rows;

  // Also get the routine profile summary (latest day's available dates)
  const datesQuery = `
    SELECT DISTINCT record_date 
    FROM adl_history 
    WHERE elder_id = $1
    ORDER BY record_date DESC
    LIMIT 30
  `;
  const datesResult = await db.query(datesQuery, [elderId]);
  const dates = datesResult.rows;

  return successResponse({
    anomalies,
    history_days: dates.length,
    message: dates.length < 7
      ? `Need ${7 - dates.length} more days of data for reliable pattern detection`
      : 'Pattern profile ready'
  });
});
