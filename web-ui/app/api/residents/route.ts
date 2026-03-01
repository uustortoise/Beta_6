import { NextResponse } from 'next/server';
import { createResident } from '../../lib/data';
import { residentSchema, parseRequestBody } from '../../lib/schemas';
import { withErrorHandling, successResponse, ConflictError } from '../../lib/errors';

export const POST = withErrorHandling(async (request: Request) => {
    // 1. Validate input body against Zod schema
    const body = await parseRequestBody(request, residentSchema);

    // 2. Determine ID (either provided or generated)
    // Note: Zod schema enforces letters/numbers/underscores if ID provided
    const newId = body.id;

    // 3. Create resident
    try {
        const created = await createResident(newId, {
            personal_info: {
                full_name: body.name,
                age: body.age,
                // Default others for now
                gender: 'Unknown',
                date_of_birth: new Date().toISOString()
            }
        });

        if (!created) {
            throw new Error('Database insert failed');
        }
    } catch (e) {
        if (e instanceof Error && e.message.includes('exists')) {
            throw new ConflictError(`Resident ID '${newId}' already exists`);
        }
        throw e;
    }

    return successResponse({
        id: newId,
        message: 'Resident created successfully'
    }, 201);
});
