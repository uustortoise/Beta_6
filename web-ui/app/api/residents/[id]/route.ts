import { NextResponse } from 'next/server';
import { getResidentById, updateResidentProfile, ResidentProfile } from '../../../lib/data';
import { getDb } from '../../../lib/db';
import { withErrorHandling, errorResponse, NotFoundError, InternalError, successResponse } from '../../../lib/errors';

export const GET = withErrorHandling(async (
    request: Request,
    context: { params: Promise<{ id: string }> }
) => {
    const { id } = await context.params;
    const profile = await getResidentById(id);

    if (!profile) {
        throw new NotFoundError('Resident', id);
    }

    return NextResponse.json(profile);
});

export const POST = withErrorHandling(async (
    request: Request,
    context: { params: Promise<{ id: string }> }
) => {
    const { id } = await context.params;
    const body = await request.json();
    const success = await updateResidentProfile(id, body as Partial<ResidentProfile>);

    if (!success) {
        throw new InternalError('Failed to update profile');
    }

    return successResponse({ success: true });
});

export const DELETE = withErrorHandling(async (
    request: Request,
    context: { params: Promise<{ id: string }> }
) => {
    const { id } = await context.params;
    const db = getDb();
    const client = await db.connect();

    try {
        await client.query('BEGIN');

        // Check if resident exists
        const residentResult = await client.query('SELECT elder_id FROM elders WHERE elder_id = $1', [id]);
        if (residentResult.rowCount === 0) {
            throw new NotFoundError('Resident', id);
        }

        // Cascade delete all related data
        await client.query('DELETE FROM medical_history WHERE elder_id = $1', [id]);
        await client.query('DELETE FROM emergency_contacts WHERE elder_id = $1', [id]);
        await client.query('DELETE FROM alerts WHERE elder_id = $1', [id]);
        await client.query('DELETE FROM sleep_analysis WHERE elder_id = $1', [id]);
        await client.query('DELETE FROM adl_history WHERE elder_id = $1', [id]);
        await client.query('DELETE FROM icope_assessments WHERE elder_id = $1', [id]);

        // Finally delete the elder record
        await client.query('DELETE FROM elders WHERE elder_id = $1', [id]);

        await client.query('COMMIT');

        return successResponse({ success: true, message: 'Profile deleted successfully' });
    } catch (error) {
        await client.query('ROLLBACK');
        throw error;
    } finally {
        client.release();
    }
});
