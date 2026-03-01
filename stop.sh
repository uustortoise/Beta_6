#!/bin/bash
# stop.sh - Kill all Beta_6 related processes
# Kill automation watcher, streamlit dashboard, and Next.js dev server
#
# Usage: ./stop.sh

echo "🛑 Stopping all Beta_6 processes..."

# Kill Python watchers
pkill -f "run_daily_analysis.py" 2>/dev/null && echo "  ✓ Killed analysis watcher(s)" || echo "  - No analysis watchers running"

# Kill Streamlit
pkill -f "streamlit run" 2>/dev/null && echo "  ✓ Killed Streamlit" || echo "  - No Streamlit running"

# Kill Next.js dev server
pkill -f "next dev" 2>/dev/null && echo "  ✓ Killed Next.js dev server" || echo "  - No Next.js dev server running"
pkill -f "next-server" 2>/dev/null

echo "✅ Cleanup complete."
