import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import { trainRequestSchema, parseRequestBody } from '../../lib/schemas';
import { withErrorHandling, successResponse, ConflictError } from '../../lib/errors';

// Lock to prevent multiple concurrent training sessions
let isTraining = false;

export const POST = withErrorHandling(async (request: Request) => {
    if (isTraining) {
        throw new ConflictError("Training is already in progress");
    }

    // 1. Zod Validation
    const body = await parseRequestBody(request, trainRequestSchema);

    // 2. Training Logic
    const epochs = body.epochs;
    const scriptPath = path.resolve(process.cwd(), '../backend/mock_train.py');

    console.log(`Starting training for ${body.elderId} (${body.room}) with script: ${scriptPath}`);

    // Spawn python process
    const pythonProcess = spawn('python3', [
        scriptPath,
        '--epochs', String(epochs),
        '--elder', body.elderId,
        '--room', body.room
    ], {
        detached: true,
        stdio: 'ignore'
    });

    pythonProcess.unref();
    isTraining = true;

    // Reset lock logic (improved for reliability?)
    setTimeout(() => { isTraining = false; }, 2000);

    return successResponse({
        message: "Training started successfully",
        jobId: `train_${Date.now()}`
    });
});
