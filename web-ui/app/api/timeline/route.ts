import { NextRequest, NextResponse } from 'next/server';
import { getAvailableTimelineDates, getActivityTimelineForDate } from '../../lib/data';
import { withErrorHandling, ValidationError, successResponse } from '../../lib/errors';

export const GET = withErrorHandling(async (request: NextRequest) => {
    const searchParams = request.nextUrl.searchParams;
    const elderId = searchParams.get('elderId');
    const date = searchParams.get('date');

    if (!elderId) {
        throw new ValidationError('elderId is required');
    }

    // If date is requested, return timeline for that date
    if (date) {
        const timeline = await getActivityTimelineForDate(elderId, date);
        return successResponse({ timeline });
    }

    // Otherwise return available dates
    const dates = await getAvailableTimelineDates(elderId);
    return successResponse({ dates });
});
