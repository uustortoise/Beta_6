/**
 * Standardized Error Handling Utilities
 * 
 * Provides consistent error responses across all API endpoints.
 * 
 * Features:
 * - Structured error format
 * - No stack traces in production
 * - Correlation ID support
 * - Type-safe error responses
 */

import { NextResponse } from 'next/server';
import { ZodError } from 'zod';

// ============================================
// Error Types
// ============================================

export interface ApiError {
    error: {
        code: string;
        message: string;
        details?: Record<string, unknown>;
        correlationId?: string;
    };
}

export class AppError extends Error {
    constructor(
        message: string,
        public readonly code: string,
        public readonly statusCode: number = 500,
        public readonly details?: Record<string, unknown>
    ) {
        super(message);
        this.name = 'AppError';
    }
}

// ============================================
// Common Error Classes
// ============================================

export class NotFoundError extends AppError {
    constructor(resource: string, id?: string) {
        const message = id
            ? `${resource} with ID '${id}' not found`
            : `${resource} not found`;
        super(message, 'NOT_FOUND', 404);
    }
}

export class ValidationError extends AppError {
    constructor(message: string, details?: Record<string, unknown>) {
        super(message, 'VALIDATION_ERROR', 400, details);
    }
}

export class UnauthorizedError extends AppError {
    constructor(message: string = 'Authentication required') {
        super(message, 'UNAUTHORIZED', 401);
    }
}

export class ForbiddenError extends AppError {
    constructor(message: string = 'Access denied') {
        super(message, 'FORBIDDEN', 403);
    }
}

export class ConflictError extends AppError {
    constructor(message: string) {
        super(message, 'CONFLICT', 409);
    }
}

export class RateLimitError extends AppError {
    constructor(retryAfter?: number) {
        super('Too many requests', 'RATE_LIMITED', 429, { retryAfter });
    }
}

export class InternalError extends AppError {
    constructor(message: string = 'An unexpected error occurred') {
        super(message, 'INTERNAL_ERROR', 500);
    }
}

// ============================================
// Correlation ID Generation
// ============================================

let requestCounter = 0;

export function generateCorrelationId(): string {
    requestCounter = (requestCounter + 1) % 100000;
    const timestamp = Date.now().toString(36);
    const counter = requestCounter.toString(36).padStart(3, '0');
    return `${timestamp}-${counter}`;
}

// ============================================
// Error Response Helpers
// ============================================

const IS_PRODUCTION = process.env.NODE_ENV === 'production';

/**
 * Create standardized error response
 */
export function errorResponse(
    error: AppError | Error | unknown,
    correlationId?: string
): NextResponse<ApiError> {
    const cid = correlationId || generateCorrelationId();

    // Handle AppError (our custom errors)
    if (error instanceof AppError) {
        return NextResponse.json(
            {
                error: {
                    code: error.code,
                    message: error.message,
                    details: error.details,
                    correlationId: cid
                }
            },
            { status: error.statusCode }
        );
    }

    // Handle Zod validation errors
    if (error instanceof ZodError) {
        return NextResponse.json(
            {
                error: {
                    code: 'VALIDATION_ERROR',
                    message: 'Request validation failed',
                    details: {
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        errors: error.issues.map((e: any) => ({
                            field: e.path.join('.'),
                            message: e.message
                        }))
                    },
                    correlationId: cid
                }
            },
            { status: 400 }
        );
    }

    // Handle generic errors
    if (error instanceof Error) {
        // Log full error server-side
        console.error(`[${cid}] Unhandled error:`, error);

        return NextResponse.json(
            {
                error: {
                    code: 'INTERNAL_ERROR',
                    // Don't expose error details in production
                    message: IS_PRODUCTION
                        ? 'An unexpected error occurred'
                        : error.message,
                    correlationId: cid
                }
            },
            { status: 500 }
        );
    }

    // Unknown error type
    console.error(`[${cid}] Unknown error type:`, error);
    return NextResponse.json(
        {
            error: {
                code: 'INTERNAL_ERROR',
                message: 'An unexpected error occurred',
                correlationId: cid
            }
        },
        { status: 500 }
    );
}

/**
 * Wrap async route handler with error handling
 * 
 * Usage:
 *   export const GET = withErrorHandling(async (request) => {
 *       const data = await fetchData();
 *       return NextResponse.json(data);
 *   });
 */
// Overload for handlers without context (most routes)
export function withErrorHandling<T extends Request>(
    handler: (request: T) => Promise<NextResponse>
): (request: T) => Promise<NextResponse>;

// Overload for handlers with context (dynamic routes like [id])
export function withErrorHandling<T extends Request, C>(
    handler: (request: T, context: C) => Promise<NextResponse>
): (request: T, context: C) => Promise<NextResponse>;

// Implementation
export function withErrorHandling<T extends Request, C = void>(
    handler: (request: T, context?: C) => Promise<NextResponse>
) {
    return async (request: T, context?: C): Promise<NextResponse> => {
        const correlationId = generateCorrelationId();

        try {
            return await handler(request, context);
        } catch (error) {
            return errorResponse(error, correlationId);
        }
    };
}

/**
 * Success response helper with standard format
 */
export function successResponse<T>(
    data: T,
    status: number = 200
): NextResponse<{ data: T }> {
    return NextResponse.json({ data }, { status });
}

/**
 * Paginated response helper
 */
export function paginatedResponse<T>(
    items: T[],
    total: number,
    page: number,
    limit: number
): NextResponse {
    return NextResponse.json({
        data: items,
        pagination: {
            total,
            page,
            limit,
            totalPages: Math.ceil(total / limit),
            hasMore: page * limit < total
        }
    });
}
