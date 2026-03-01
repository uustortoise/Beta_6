import { NextRequest, NextResponse } from 'next/server';
import { getDb } from '../../lib/db';
import { withErrorHandling, successResponse, ValidationError, NotFoundError } from '../../lib/errors';

// GET /api/alert-rules-v2 - List all v2 rules
export const GET = withErrorHandling(async () => {
    const db = getDb();
    const result = await db.query('SELECT * FROM alert_rules_v2 ORDER BY created_at DESC');
    const rules = result.rows;

    // Parse JSON fields
    const parsed = rules.map((rule: any) => ({
        ...rule,
        conditions: JSON.parse(rule.conditions || '{}'),
        enabled: Boolean(rule.enabled)
    }));

    return NextResponse.json(parsed);
});

// POST /api/alert-rules-v2 - Create new rule
export const POST = withErrorHandling(async (request: NextRequest) => {
    const db = getDb();
    const body = await request.json();
    const { rule_name, required_condition, conditions, alert_severity, alert_message } = body;

    if (!rule_name || !conditions) {
        throw new ValidationError('Rule name and conditions are required');
    }

    const result = await db.query(`
        INSERT INTO alert_rules_v2 (rule_name, required_condition, conditions, alert_severity, alert_message, enabled)
        VALUES ($1, $2, $3, $4, $5, TRUE)
        RETURNING id
    `, [
        rule_name,
        required_condition || null,
        JSON.stringify(conditions),
        alert_severity || 'medium',
        alert_message || ''
    ]);

    return successResponse({ success: true, id: result.rows[0].id });
});

// PUT /api/alert-rules-v2 - Update rule
export const PUT = withErrorHandling(async (request: NextRequest) => {
    const db = getDb();
    const body = await request.json();
    const { id, enabled, conditions, alert_severity, required_condition } = body;

    if (!id) {
        throw new ValidationError('Rule ID is required');
    }

    const result = await db.query(`
        UPDATE alert_rules_v2 
        SET enabled = $1,
            conditions = $2,
            alert_severity = $3,
            required_condition = $4,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $5
        RETURNING id
    `, [
        enabled ? 1 : 0,
        JSON.stringify(conditions || {}),
        alert_severity || 'medium',
        required_condition || null,
        id
    ]);

    if (result.rowCount === 0) {
        throw new NotFoundError('Alert rule', String(id));
    }

    return successResponse({ success: true });
});

// DELETE /api/alert-rules-v2 - Delete rule
export const DELETE = withErrorHandling(async (request: NextRequest) => {
    const db = getDb();
    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');

    if (!id) {
        throw new ValidationError('Rule ID is required');
    }

    const result = await db.query('DELETE FROM alert_rules_v2 WHERE id = $1 RETURNING id', [id]);

    if (result.rowCount === 0) {
        throw new NotFoundError('Alert rule', id);
    }

    return successResponse({ success: true });
});
