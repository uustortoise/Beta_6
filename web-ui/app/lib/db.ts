import { Pool } from 'pg';

function requiredEnv(name: string): string {
    const value = process.env[name];
    if (!value || value.trim() === '') {
        throw new Error(`[DB Config] Missing required environment variable: ${name}`);
    }
    return value;
}

const pool = new Pool({
    host: process.env.POSTGRES_HOST || 'localhost',
    port: parseInt(process.env.POSTGRES_PORT || '5432', 10),
    user: requiredEnv('POSTGRES_USER'),
    password: requiredEnv('POSTGRES_PASSWORD'),
    database: requiredEnv('POSTGRES_DB'),
    max: 20,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000,
    // Performance monitoring
    application_name: 'eldercare_web_ui',
    keepAlive: true,
    keepAliveInitialDelayMillis: 10000,
    // Safety net: 10s default timeout for all queries (prevents runaway queries)
    statement_timeout: 10000,
});

// Monitor pool errors for debugging
pool.on('error', (err) => {
    console.error('[DB Pool] Unexpected error on idle client:', err);
});

export function getDb() {
    return pool;
}

// Export pool metrics for health checks and monitoring
export function getPoolMetrics() {
    return {
        total: pool.totalCount,
        idle: pool.idleCount,
        waiting: pool.waitingCount
    };
}

// Query with timeout protection
export async function queryWithTimeout<T>(
    queryFn: () => Promise<T>,
    timeoutMs: number = 5000,
    context: string = 'query'
): Promise<T> {
    const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new QueryTimeoutError(context, timeoutMs)), timeoutMs)
    );

    return Promise.race([queryFn(), timeoutPromise]);
}

// Custom error for query timeouts
export class QueryTimeoutError extends Error {
    public readonly code = 'QUERY_TIMEOUT';
    public readonly statusCode = 504;

    constructor(context: string, timeoutMs: number) {
        super(`[${context}] Query timeout after ${timeoutMs}ms`);
        this.name = 'QueryTimeoutError';
    }
}

// Timeout recommendations by query type (ms)
export const QUERY_TIMEOUTS = {
    SIMPLE_LOOKUP: 2000,   // PK lookups
    LIST_QUERY: 5000,      // List queries with LIMIT
    AGGREGATION: 10000,    // COUNT, SUM, GROUP BY
    TIMELINE: 10000,       // Date range scans
    EXPORT: 60000          // Full table exports
} as const;

// Convenience wrapper: timedQuery for direct SQL execution with timeout
export async function timedQuery<T extends Record<string, any> = Record<string, any>>(
    sql: string,
    params?: any[],
    timeoutMs: number = QUERY_TIMEOUTS.LIST_QUERY
): Promise<{ rows: T[]; rowCount: number | null }> {
    return queryWithTimeout(
        () => pool.query<T>(sql, params),
        timeoutMs,
        `SQL: ${sql.substring(0, 50)}...`
    );
}
