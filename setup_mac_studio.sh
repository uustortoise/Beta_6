#!/bin/bash

# 🚀 Beta 5.5 - Mac Studio One-Click Setup Script
# ------------------------------------------------
# This script automates the setup of the DT_development (Beta 5.5) environment.
# Run this on your NEW Mac Studio.

# Exit on error
set -e

echo "--------------------------------------------------"
echo "🍏 Mac Studio Setup - Beta 5.5"
echo "--------------------------------------------------"

# 1. Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew is NOT installed."
    echo "👉 Please run this command first, then re-run this script:"
    echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi

# 2. Install System Dependencies
echo "📦 Installing System Dependencies (Python 3.11, Node 22, Git)..."
brew install python@3.11 node@22 git

# 3. Clone Repository
echo "⬇️ Cloning Repository..."
# Check if directory already exists
if [ -d "DT_development" ]; then
    echo "⚠️  'DT_development' directory already exists. Updating instead..."
    cd DT_development
    git fetch origin
    git checkout beta-5.5-transformer
    git pull origin beta-5.5-transformer
else
    echo "🔑 You will be asked for your GitHub credentials."
    echo "   - Username: uustortoise"
    echo "   - Password: <your-github-personal-access-token>" 
    git clone -b beta-5.5-transformer https://github.com/uustortoise/DT_development.git
    cd DT_development
fi

# 4. Navigate to Project Root
# The actual Beta 5.5 code is nested
TARGET_DIR="Development/Beta_5.5"
if [ ! -d "$TARGET_DIR" ]; then
    echo "❌ Error: Could not find '$TARGET_DIR' inside the repo."
    exit 1
fi
cd "$TARGET_DIR"
echo "📂 Working in: $(pwd)"

# 5. Backend Setup
echo "🔧 Setting up Backend..."
cd backend
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "   - Created .env file"
else
    echo "   - .env already exists"
fi

echo "   - Setting up Python virtual environment..."
cd ..
# Force venv rebuild to ensure TF 2.16 downgrade is applied cleanly
echo "   - Rebuilding venv to apply critical TensorFlow fixes..."
rm -rf backend/venv
./create_venv.sh
cd backend

# 6. Database Setup
echo "🐳 Starting Docker Services..."
if docker info > /dev/null 2>&1; then
    docker-compose up -d
    echo "   - Waiting for Database..."
    sleep 10
else
    echo "⚠️  Docker is NOT running. Please start Docker Desktop manually."
    read -p "Press [Enter] once Docker is running to continue..."
fi

echo "📦 Initializing Database Schema..."
python3.11 scripts/init_db.py

# 7. Frontend Setup
echo "🎨 Setting up Frontend..."
cd ../web-ui
echo "   - Installing Node packages..."
npm install

echo "--------------------------------------------------"
echo "✅ SETUP COMPLETE!"
echo "--------------------------------------------------"
echo "To start the system:"
echo "  1. cd $(pwd)/.."   # Points to Beta_5.5 dir
echo "  2. ./start.sh"
echo "--------------------------------------------------"
