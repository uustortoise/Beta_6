/**
 * API Request/Response Validation Schemas (Zod)
 * 
 * REQUIREMENTS: Install zod before using:
 *   npm install zod
 * 
 * Usage:
 *   import { residentSchema, trainRequestSchema } from '../lib/schemas';
 *   const validatedData = residentSchema.parse(requestBody);
 */

import { z } from 'zod';

// ============================================
// Resident Schemas
// ============================================

/**
 * Resident creation/update schema
 */
export const residentSchema = z.object({
    id: z.string()
        .min(1, 'Resident ID is required')
        .max(50, 'ID too long')
        .regex(/^[a-zA-Z0-9_]+$/, 'ID can only contain letters, numbers, and underscores'),
    name: z.string()
        .min(2, 'Name must be at least 2 characters')
        .max(100, 'Name too long'),
    age: z.number()
        .int('Age must be a whole number')
        .min(0, 'Age cannot be negative')
        .max(150, 'Age seems unrealistic')
        .optional(),
    room_id: z.string().optional(),
    medical_conditions: z.array(z.string()).optional(),
    emergency_contacts: z.array(z.object({
        name: z.string(),
        phone: z.string(),
        relationship: z.string().optional()
    })).optional()
});

export type ResidentInput = z.infer<typeof residentSchema>;

// ============================================
// Training Schemas
// ============================================

/**
 * Training request schema
 */
export const trainRequestSchema = z.object({
    elderId: z.string()
        .min(1, 'Elder ID is required'),
    room: z.string()
        .min(1, 'Room is required'),
    epochs: z.number()
        .int('Epochs must be a whole number')
        .min(1, 'At least 1 epoch required')
        .max(100, 'Too many epochs (max 100)')
        .default(5),
    retrospective: z.boolean()
        .default(false),
    learningRate: z.number()
        .min(0.0001, 'Learning rate too small')
        .max(0.1, 'Learning rate too large')
        .optional()
});

export type TrainRequest = z.infer<typeof trainRequestSchema>;

// ============================================
// Query Parameter Schemas
// ============================================

/**
 * Date range query parameters
 */
export const dateRangeSchema = z.object({
    startDate: z.string()
        .regex(/^\d{4}-\d{2}-\d{2}$/, 'Date must be YYYY-MM-DD format')
        .optional(),
    endDate: z.string()
        .regex(/^\d{4}-\d{2}-\d{2}$/, 'Date must be YYYY-MM-DD format')
        .optional()
}).refine(
    data => {
        if (data.startDate && data.endDate) {
            return new Date(data.startDate) <= new Date(data.endDate);
        }
        return true;
    },
    { message: 'Start date must be before or equal to end date' }
);

/**
 * Pagination query parameters
 */
export const paginationSchema = z.object({
    page: z.coerce.number()
        .int()
        .min(1)
        .default(1),
    limit: z.coerce.number()
        .int()
        .min(1)
        .max(100, 'Maximum 100 items per page')
        .default(20)
});

export type PaginationParams = z.infer<typeof paginationSchema>;

// ============================================
// Timeline/Activity Schemas
// ============================================

/**
 * Activity correction schema
 */
export const correctionSchema = z.object({
    elderId: z.string().min(1),
    room: z.string().min(1),
    startTime: z.string()
        .datetime({ message: 'Invalid datetime format' }),
    endTime: z.string()
        .datetime({ message: 'Invalid datetime format' }),
    activity: z.string()
        .min(1, 'Activity label is required')
        .max(50, 'Activity label too long')
});

export type CorrectionInput = z.infer<typeof correctionSchema>;

// ============================================
// Configuration Schemas
// ============================================

/**
 * Room configuration schema
 */
export const roomConfigSchema = z.object({
    roomId: z.string().min(1),
    sequenceTimeWindow: z.number()
        .int('Must be a whole number')
        .min(60, 'Minimum 60 seconds')
        .max(7200, 'Maximum 2 hours')
        .refine(v => v % 10 === 0, 'Must be a multiple of 10 seconds'),
    alertThresholds: z.object({
        inactivityMinutes: z.number().min(1).max(1440).optional(),
        lowConfidence: z.number().min(0).max(1).optional()
    }).optional()
});

export type RoomConfig = z.infer<typeof roomConfigSchema>;

// ============================================
// Helper Functions
// ============================================

/**
 * Safely parse request body with a Zod schema
 * Returns validated data or throws with user-friendly error
 */
export async function parseRequestBody<T>(
    request: Request,
    schema: z.ZodSchema<T>
): Promise<T> {
    const body = await request.json();
    return schema.parse(body);
}

/**
 * Safely parse query parameters with a Zod schema
 */
export function parseQueryParams<T>(
    url: string,
    schema: z.ZodSchema<T>
): T {
    const { searchParams } = new URL(url);
    const params = Object.fromEntries(searchParams.entries());
    return schema.parse(params);
}

/**
 * Format Zod validation errors for API response
 */
export function formatZodError(error: z.ZodError): {
    message: string;
    errors: { field: string; message: string }[];
} {
    return {
        message: 'Validation failed',
        errors: error.issues.map(e => ({
            field: e.path.join('.'),
            message: e.message
        }))
    };
}
