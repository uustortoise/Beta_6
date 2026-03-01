#!/bin/bash

# Configuration - Uses relative path from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
BACKEND_DIR="$PROJECT_ROOT/backend"
DATA_DIR="$PROJECT_ROOT/data"
WEB_DIR="$PROJECT_ROOT/web-ui"

echo "--------------------------------------------------"
echo " Elderly Care Platform - Beta_5 (Prototype Mode) "
echo "--------------------------------------------------"

# 1. Environment Check
if [ ! -d "$BACKEND_DIR" ]; then
    echo "Error: Backend directory not found at $BACKEND_DIR"
    exit 1
fi

# 2. Setup Directories
mkdir -p "$DATA_DIR/raw" "$DATA_DIR/processed" "$DATA_DIR/archive"

# 3. Start Data Processing Service (in background)
echo "👁️  Starting Daily Analysis Watcher..."
python3 "$BACKEND_DIR/run_daily_analysis.py" &
PROCESS_PID=$!

# 4. Start Streamlit Export Dashboard (New)
echo "📊 Starting Export Dashboard on http://localhost:8501..."
# Use absolute path for streamlit script
streamlit run "$BACKEND_DIR/export_dashboard.py" --server.port 8501 --server.headings.visible false > "$PROJECT_ROOT/export_ui.log" 2>&1 &
PID_UI=$!

# 5. Start Next.js Web-UI (Resident Dashboard)
echo "🌐 Starting Resident Dashboard on http://localhost:3000..."
cd "$WEB_DIR"
# Ensure dependencies are installed (first run only)
if [ ! -d "node_modules" ]; then
    echo "Installing Web-UI dependencies (this may take a minute)..."
    npm install > install.log 2>&1
fi
npm run dev > "$PROJECT_ROOT/web_ui.log" 2>&1 &
PID_WEB=$!

echo "✅ Beta_5 Prototype Environment Started!"
echo "   - Backend Process: $PROCESS_PID"
echo "   - Export Tool:     $PID_UI (http://localhost:8501)"
echo "   - Web Dashboard:   $PID_WEB (http://localhost:3000)"
echo "   - Drop files in: $DATA_DIR/raw"
echo "   - Logs: $PROJECT_ROOT/automation.log"
echo "--------------------------------------------------"
echo "Press Ctrl+C to stop services."

# Wait for all background processes
wait $PROCESS_PID $PID_UI $PID_WEB
