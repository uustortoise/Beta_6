#!/bin/bash

# Health Advisory Chatbot - Demo Startup Script

echo "=================================="
echo "🚀 Chatbot Demo Dashboard"
echo "=================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed."
    exit 1
fi

# Set up Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../backend"
export LLM_PROVIDER="mock"

echo "📦 Setting up environment..."
cd backend

# Check if dependencies are installed
python3 -c "import pydantic" 2>/dev/null || {
    echo "⚠️  Warning: pydantic not installed. Install with: pip install pydantic"
}

echo ""
echo "🎭 Available Mock Elders:"
echo "   1. Margaret - High fall risk scenario"
echo "   2. Robert  - Cognitive concerns scenario"
echo "   3. Helen   - Generally healthy scenario"
echo ""

echo "🌐 Starting server on http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop"
echo "=================================="
echo ""

# Start the server
python3 demo_server.py "$@"
