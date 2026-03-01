import { NextResponse } from 'next/server';
import { getDb } from '../../lib/db';
import { withErrorHandling, ValidationError, successResponse } from '../../lib/errors';

export const dynamic = 'force-dynamic';

export const GET = withErrorHandling(async (request: Request) => {
  const { searchParams } = new URL(request.url);
  const elderId = searchParams.get('elderId');
  const date = searchParams.get('date');

  if (!elderId) {
    throw new ValidationError('elderId is required');
  }

  const db = getDb();

  let query: string;
  let params: any[];

  if (date) {
    // Get trajectories for specific date
    query = `
      SELECT id, start_time, end_time, path, primary_activity, 
             room_sequence, duration_minutes, confidence, record_date
      FROM trajectory_events
      WHERE elder_id = $1 AND record_date = $2
      ORDER BY start_time
    `;
    params = [elderId, date];
  } else {
    // Get latest day's trajectories
    query = `
      SELECT id, start_time, end_time, path, primary_activity, 
             room_sequence, duration_minutes, confidence, record_date
      FROM trajectory_events
      WHERE elder_id = $1 
      AND record_date = (SELECT MAX(record_date) FROM trajectory_events WHERE elder_id = $2)
      ORDER BY start_time
    `;
    params = [elderId, elderId];
  }

  const result = await db.query(query, params);
  const trajectories = result.rows;

  // Parse room_sequence JSON
  const parsed = trajectories.map((t: any) => ({
    ...t,
    room_sequence: t.room_sequence ? JSON.parse(t.room_sequence) : []
  }));

  return successResponse({
    trajectories: parsed,
    date: parsed.length > 0 ? parsed[0].record_date : date
  });
});
