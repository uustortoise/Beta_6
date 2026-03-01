import { NextResponse } from 'next/server';
import { getDb, getPoolMetrics } from '../../lib/db';
import fs from 'fs';
import path from 'path';

const VERSION = '5.5.1';

interface ComponentHealth {
    status: 'healthy' | 'degraded' | 'unhealthy';
    message: string;
    latency_ms?: number;
}

interface HealthResponse {
    status: 'healthy' | 'degraded' | 'unhealthy';
    timestamp: string;
    version: string;
    components?: Record<string, ComponentHealth>;
}

/**
 * Check PostgreSQL database connectivity and pool health
 */
async function checkDatabase(): Promise<ComponentHealth> {
    const start = Date.now();
    try {
        const db = getDb();
        await db.query('SELECT 1');

        // Check pool health
        const metrics = getPoolMetrics();
        if (metrics.waiting > 5) {
            return {
                status: 'degraded',
                message: `Pool near capacity: ${metrics.waiting} waiting, ${metrics.idle}/${metrics.total} idle`,
                latency_ms: Date.now() - start
            };
        }

        return {
            status: 'healthy',
            message: `PostgreSQL Connected (pool: ${metrics.idle}/${metrics.total} idle)`,
            latency_ms: Date.now() - start
        };
    } catch (error) {
        return {
            status: 'unhealthy',
            message: error instanceof Error ? error.message : 'Connection failed'
        };
    }
}

/**
 * Check if ML models directory exists and has models
 */
function checkModels(): ComponentHealth {
    const modelsDir = path.resolve(process.cwd(), '../backend/models');

    try {
        if (!fs.existsSync(modelsDir)) {
            return {
                status: 'degraded',
                message: 'Models directory not found'
            };
        }

        // Check for any .keras files
        const files = fs.readdirSync(modelsDir, { recursive: true });
        const modelCount = (files as string[]).filter(f =>
            typeof f === 'string' && f.endsWith('.keras')
        ).length;

        if (modelCount === 0) {
            return {
                status: 'degraded',
                message: 'No models found'
            };
        }

        return {
            status: 'healthy',
            message: `${modelCount} model(s) available`
        };
    } catch (error) {
        return {
            status: 'unhealthy',
            message: error instanceof Error ? error.message : 'Check failed'
        };
    }
}

/**
 * Check Python backend health server if running
 */
async function checkBackend(): Promise<ComponentHealth> {
    const start = Date.now();
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000);

        const response = await fetch('http://localhost:8504/health', {
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (response.ok) {
            return {
                status: 'healthy',
                message: 'Backend responding',
                latency_ms: Date.now() - start
            };
        }
        return {
            status: 'degraded',
            message: `Backend returned ${response.status}`
        };
    } catch {
        return {
            status: 'degraded',
            message: 'Backend health server not reachable (optional)'
        };
    }
}

export async function GET(request: Request) {
    const { searchParams } = new URL(request.url);
    const checkType = searchParams.get('check') || 'liveness';

    const timestamp = new Date().toISOString();

    // Simple liveness check
    if (checkType === 'liveness') {
        return NextResponse.json({
            status: 'ok',
            timestamp
        });
    }

    // Readiness check - can we serve requests?
    if (checkType === 'ready') {
        const dbHealth = await checkDatabase();
        const ready = dbHealth.status !== 'unhealthy';

        const response: HealthResponse = {
            status: ready ? 'healthy' : 'unhealthy',
            timestamp,
            version: VERSION,
            components: {
                database: dbHealth
            }
        };

        return NextResponse.json(response, {
            status: ready ? 200 : 503
        });
    }

    // Deep health check - all components
    if (checkType === 'deep') {
        const dbHealth = await checkDatabase();
        const modelsHealth = checkModels();
        const backendHealth = await checkBackend();

        const components = {
            database: dbHealth,
            models: modelsHealth,
            backend: backendHealth
        };

        // Determine overall status
        let overallStatus: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
        const statuses = Object.values(components).map(c => c.status);

        if (statuses.includes('unhealthy')) {
            overallStatus = 'unhealthy';
        } else if (statuses.includes('degraded')) {
            overallStatus = 'degraded';
        }

        const response: HealthResponse = {
            status: overallStatus,
            timestamp,
            version: VERSION,
            components
        };

        return NextResponse.json(response, {
            status: overallStatus === 'unhealthy' ? 503 : 200
        });
    }

    // Default: simple OK response
    return NextResponse.json({
        status: 'healthy',
        timestamp,
        version: VERSION
    });
}
