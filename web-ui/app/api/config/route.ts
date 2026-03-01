import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

// Path to settings.py
const SETTINGS_PATH = path.resolve(process.cwd(), '../backend/elderlycare_v1_16/config/settings.py');

const DEFAULT_SETTINGS = {
    DEFAULT_CONV_FILTERS_1: 64,
    DEFAULT_CONV_FILTERS_2: 32,
    DEFAULT_LSTM_UNITS: 64,
    DEFAULT_DROPOUT_RATE: 0.3,
    DEFAULT_EPOCHS: 5,
    DEFAULT_VALIDATION_SPLIT: 0.2,
    DEFAULT_CONFIDENCE_THRESHOLD: 0.8,
    DEFAULT_DATA_INTERVAL: '10s',
    DEFAULT_SEQUENCE_WINDOW: '10min',
    DEFAULT_DENOISING_WINDOW: 3,
    DEFAULT_DENOISING_THRESHOLD: 4,
    DEFAULT_DENOISING_METHOD: 'hampel',
    SLEEP_STAGE_RATIOS: {
        Light: 0.55,
        Deep: 0.15,
        REM: 0.20,
        Awake: 0.10
    },
    ADL_CONFIDENCE_THRESHOLD: 0.8
};

// Settings that are allowed to be modified (Security)
const ALLOWED_Settings = [
    // Model Hyperparameters
    'DEFAULT_CONV_FILTERS_1',
    'DEFAULT_CONV_FILTERS_2',
    'DEFAULT_LSTM_UNITS',
    'DEFAULT_DROPOUT_RATE',
    'DEFAULT_EPOCHS',
    'DEFAULT_VALIDATION_SPLIT',
    'DEFAULT_CONFIDENCE_THRESHOLD',

    // Data Processing
    'DEFAULT_DATA_INTERVAL',
    'DEFAULT_SEQUENCE_WINDOW',
    'DEFAULT_DENOISING_WINDOW',
    'DEFAULT_DENOISING_THRESHOLD',
    'DEFAULT_DENOISING_METHOD',

    // Sleep Analysis
    'SLEEP_STAGE_RATIOS',

    // Activity Analysis
    'ADL_CONFIDENCE_THRESHOLD'
];

export async function GET() {
    try {
        if (!fs.existsSync(SETTINGS_PATH)) {
            return NextResponse.json({ error: 'Settings file not found' }, { status: 404 });
        }

        const content = fs.readFileSync(SETTINGS_PATH, 'utf-8');
        const settings: any = {};

        // Simple regex parser for Python variables
        // Matches: VARIABLE = value
        // Also captures dictionaries roughly

        const lines = content.split('\n');
        let currentDictKey = null;
        let currentDictContent = "";

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || trimmed.startsWith('#')) continue;

            // Handle Dictionary Start (e.g. SLEEP_STAGE_RATIOS = {)
            if (trimmed.includes(' = {')) {
                const [key, _] = trimmed.split(' = {');
                if (ALLOWED_Settings.includes(key)) {
                    currentDictKey = key;
                    currentDictContent = "{";
                }
                continue;
            }

            // Handle Dictionary End
            if (currentDictKey && trimmed === '}') {
                currentDictContent += "}";
                // Parse the JSON-like python dict
                try {
                    // Python dicts often use single quotes, JSON needs double.
                    // This is a naive converter, perfectly handling the implementation plan's dicts
                    const jsonStr = currentDictContent
                        .replace(/'/g, '"')
                        .replace(/,\s*}/, '}'); // Remove trailing comma if any
                    settings[currentDictKey] = JSON.parse(jsonStr);
                } catch (e) {
                    console.error(`Failed to parse dictionary ${currentDictKey}`, e);
                }
                currentDictKey = null;
                continue;
            }

            // Handle Dictionary Content
            if (currentDictKey) {
                currentDictContent += line + "\n";
                continue;
            }

            // Handle Simple Values
            if (trimmed.includes(' = ')) {
                const [key, rawValue] = trimmed.split(' = ');
                if (ALLOWED_Settings.includes(key)) {
                    // Parse values
                    let value: any = rawValue.split('#')[0].trim(); // Remove comments

                    // Number
                    if (!isNaN(Number(value))) {
                        settings[key] = Number(value);
                    }
                    // String
                    else if (value.startsWith("'") || value.startsWith('"')) {
                        settings[key] = value.replace(/['"]/g, '');
                    }
                    // Boolean (Python to JS)
                    else if (value === 'True') settings[key] = true;
                    else if (value === 'False') settings[key] = false;
                    // Lists (Arrays)
                    else if (value.startsWith('[')) {
                        try {
                            // Simple array parse, assume single line for now as per default settings
                            settings[key] = JSON.parse(value.replace(/'/g, '"'));
                        } catch (e) { }
                    }
                }
            }
        }

        const merged = {
            ...DEFAULT_SETTINGS,
            ...settings,
            SLEEP_STAGE_RATIOS: {
                ...DEFAULT_SETTINGS.SLEEP_STAGE_RATIOS,
                ...(settings.SLEEP_STAGE_RATIOS || {})
            }
        };

        return NextResponse.json(merged);
    } catch (error) {
        console.error('Error reading settings:', error);
        return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
}

export async function POST(request: Request) {
    try {
        const updates = await request.json();

        if (!fs.existsSync(SETTINGS_PATH)) {
            return NextResponse.json({ error: 'Settings file not found' }, { status: 404 });
        }

        let content = fs.readFileSync(SETTINGS_PATH, 'utf-8');
        let lines = content.split('\n');

        // Naive update strategy: Line-by-line regex replacement
        // This preserves comments and structure
        for (const [key, value] of Object.entries(updates)) {
            if (!ALLOWED_Settings.includes(key)) continue;

            // Handle Dictionary Updates (Complex)
            if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                // Find start line
                const startIdx = lines.findIndex(l => l.trim().startsWith(`${key} = {`));
                if (startIdx !== -1) {
                    // We rewrite the dictionary entirely
                    // 1. Find end line
                    let endIdx = -1;
                    for (let i = startIdx; i < lines.length; i++) {
                        if (lines[i].trim() === '}') {
                            endIdx = i;
                            break;
                        }
                    }

                    if (endIdx !== -1) {
                        // Create new lines
                        const newLines = [`${key} = {`];
                        for (const [k, v] of Object.entries(value)) {
                            newLines.push(`    '${k}': ${v},`);
                        }
                        newLines.push('}');

                        // Replace lines
                        lines.splice(startIdx, endIdx - startIdx + 1, ...newLines);
                    }
                }
            }
            // Handle Simple Value Updates
            else {
                const regex = new RegExp(`^${key}\\s*=\\s*(.*)`);
                let found = false;

                // Try to find and replace existing line
                for (let i = 0; i < lines.length; i++) {
                    if (regex.test(lines[i])) {
                        // Preserve comments if existing
                        const match = lines[i].match(regex);
                        const existingComment = match && match[1].includes('#') ? ' #' + match[1].split('#')[1] : '';

                        // Format Value
                        let finalValue = JSON.stringify(value);
                        if (typeof value === 'string') finalValue = `'${value}'`; // Use single quotes for python style preference
                        if (typeof value === 'boolean') finalValue = value ? 'True' : 'False';
                        if (Array.isArray(value)) finalValue = JSON.stringify(value).replace(/"/g, "'");

                        lines[i] = `${key} = ${finalValue}${existingComment}`;
                        found = true;
                        break;
                    }
                }

                // If not found in file but allowed, append it (unlikely for settings.py but good safety)
                if (!found) {
                    // Determine section based on key prefix (heuristic)
                    lines.push(`${key} = ${typeof value === 'string' ? `'${value}'` : value}`);
                }
            }
        }

        const newContent = lines.join('\n');
        fs.writeFileSync(SETTINGS_PATH, newContent);

        return NextResponse.json({ success: true });
    } catch (error) {
        console.error('Error saving settings:', error);
        return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
}
