#!/bin/bash
# Activation script for Beta_5.5 backend Python 3.12 virtual environment
# Usage: source activate_venv.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$SCRIPT_DIR/venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Please run setup first."
    return 1
fi

source "$VENV_PATH/bin/activate"

echo "✅ Python virtual environment activated!"
python --version
echo ""
echo "Available commands:"
echo "  python     - Python 3.12 interpreter"
echo "  pip        - Package manager"
echo "  pytest     - Test runner"
echo "  black      - Code formatter"
echo "  mypy       - Type checker"
echo "  streamlit  - Streamlit server"
echo ""
echo "To deactivate, run: deactivate"
