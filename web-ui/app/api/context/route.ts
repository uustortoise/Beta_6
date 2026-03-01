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

  try {
    let query_date = date;
    if (!query_date) {
      // Default to latest date with context data
      const dateRes = await db.query(`
            SELECT TO_CHAR(MAX(start_time), 'YYYY-MM-DD') as date
            FROM context_episodes
            WHERE elder_id = $1
        `, [elderId]);

      const row = dateRes.rows[0];
      if (row) query_date = row.date;
    }

    if (!query_date) {
      return successResponse({ episodes: [], date: null });
    }

    // Get episodes for this date
    // We filter by overlap with the day
    const episodesResult = await db.query(`
        SELECT start_time, end_time, context_label, confidence, features_used
        FROM context_episodes
        WHERE elder_id = $1 
          AND start_time::date = $2::date
        ORDER BY start_time
    `, [elderId, query_date]);

    const episodes = episodesResult.rows;

    return successResponse({
      episodes,
      date: query_date
    });
  } catch (error: unknown) {
    // If context table is not present yet in this environment, return empty data
    // so dashboard can still render instead of failing hard.
    const pgError = error as { code?: string };
    if (pgError?.code === '42P01') {
      return successResponse({ episodes: [], date: null });
    }
    throw error;
  }
});
