#!/bin/bash

# 🐍 Beta 5.5 - Virtual Environment Setup Script
# ------------------------------------------------
# Creates a Python virtual environment and installs dependencies.
# Can be run standalone or called from setup_mac_studio.sh.
#
# Usage: ./create_venv.sh [--force]
#   --force: Recreate venv even if it exists

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$SCRIPT_DIR/backend"
VENV_PATH="$BACKEND_DIR/venv"
REQUIREMENTS="$BACKEND_DIR/requirements.txt"

# Parse arguments
FORCE=false
if [ "$1" == "--force" ]; then
    FORCE=true
fi

echo "🐍 Beta 5.5 - Virtual Environment Setup"
echo "----------------------------------------"

# Find best available Python (prefer 3.11, fallback to 3.12, then python3)
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ No suitable Python 3 found. Please install Python 3.11 or 3.12."
    exit 1
fi

echo "📌 Using: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Check if venv exists
if [ -d "$VENV_PATH" ]; then
    if [ "$FORCE" == true ]; then
        echo "🗑️  Removing existing venv (--force)..."
        rm -rf "$VENV_PATH"
    else
        echo "✅ Virtual environment already exists at $VENV_PATH"
        echo "   Use --force to recreate."
        exit 0
    fi
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
$PYTHON_CMD -m venv "$VENV_PATH"

# Activate and install
echo "📥 Installing dependencies from requirements.txt..."
source "$VENV_PATH/bin/activate"
pip install --upgrade pip --quiet
pip install -r "$REQUIREMENTS"

echo ""
echo "----------------------------------------"
echo "✅ Virtual environment created successfully!"
echo "   Location: $VENV_PATH"
echo ""
echo "To activate manually:"
echo "   source $VENV_PATH/bin/activate"
echo "----------------------------------------"
