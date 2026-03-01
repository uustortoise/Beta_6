#!/bin/bash

# 🔧 Mac Studio Permission Fix Script
# -----------------------------------
# Fixes "Permission denied" errors for automation.log and other files.
# Run this from the project root (where start.sh is located).

echo "🔧 Fixing Ownership and Permissions..."

# 1. Get current user and group
USER=$(whoami)
GROUP=$(id -g -n)

echo "   - Setting owner to: $USER:$GROUP"

# 2. Fix ownership recursively for the entire project
# This ensures you own all files, even if created by sudo/docker
sudo chown -R "$USER:$GROUP" .

# 3. Fix permissions for scripts to be executable
echo "   - Making scripts executable..."
chmod +x start.sh stop.sh backend/scripts/*.py

# 4. Ensure log files are writable
echo "   - Ensuring log files are writable..."
touch automation.log
chmod 664 automation.log

echo "✅ Permissions Fixed! Try running ./start.sh again."
