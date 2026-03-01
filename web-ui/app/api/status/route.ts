import { NextResponse } from 'next/server';
import { getDb } from '../../lib/db';
import fs from 'fs';
import path from 'path';
import { withErrorHandling, successResponse } from '../../lib/errors';

const DATA_DIR = path.resolve(process.cwd(), '../data');
const STATUS_FILE = path.join(DATA_DIR, 'training_status.json');

export const GET = withErrorHandling(async () => {
    let status = { status: 'idle', progress: 0, message: 'Ready' };
    let history: any[] = [];

    // 1. Get Current Status (File based for ephemeral progress often easier, or could use DB)
    if (fs.existsSync(STATUS_FILE)) {
        try {
            const statusData = fs.readFileSync(STATUS_FILE, 'utf-8');
            status = JSON.parse(statusData);
        } catch (e) {
            console.error("Error reading status file", e);
            // Don't fail the request, just return default status
        }
    }

    // 2. Get History from DB
    try {
        const db = getDb();
        const result = await db.query(`
            SELECT training_date as timestamp, model_type, epochs, accuracy, status
            FROM training_history
            ORDER BY training_date DESC
            LIMIT 50
        `);
        history = result.rows;
    } catch (e) {
        console.error("Error fetching history from DB:", e);
        // Continue with empty history rather than fail
    }

    return successResponse({
        current: status,
        history: history
    });
});
