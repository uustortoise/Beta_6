#!/bin/bash

# Configuration
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
USER_DESKTOP="$HOME/Desktop"
BACKUP_DIR="$USER_DESKTOP/Beta5_Backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
HOSTNAME=$(hostname)

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure Backup Dir exists
mkdir -p "$BACKUP_DIR"

function show_help() {
    echo -e "${BLUE}Beta 5 Data Sync Tool${NC}"
    echo "Usage:"
    echo "  ./sync_data.sh backup   -> Save current state to Desktop/iCloud"
    echo "  ./sync_data.sh restore  -> Load latest state from Desktop/iCloud"
    echo ""
}

function do_backup() {
    BACKUP_NAME="beta5_state_${HOSTNAME}_${TIMESTAMP}.tar.gz"
    DEST_FILE="$BACKUP_DIR/$BACKUP_NAME"

    echo -e "${YELLOW}📤 STARTING BACKUP${NC} (Home -> Cloud)"
    echo "Target: $DEST_FILE"

    # Check services
    if pgrep -f "run_daily_analysis.py" > /dev/null; then
        echo -e "${RED}⚠️  Services are running!${NC}"
        echo "   Backing up a live DB might corrupt it."
        read -p "   Stop services automatically? (y/N) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping services..."
            "$PROJECT_ROOT/stop.sh"
            sleep 2
        else
            echo "Proceeding with RISKY live backup..."
        fi
    fi

    cd "$PROJECT_ROOT" || exit
    
    # Create the Archive
    tar --exclude='*.DS_Store' \
        -czf "$DEST_FILE" \
        data/processed \
        data/archive \
        backend/models \
        backend/.env

    if [ -f "$DEST_FILE" ]; then
        SIZE=$(du -h "$DEST_FILE" | cut -f1)
        echo -e "${GREEN}✅ Backup Complete!${NC} ($SIZE)"
        echo "   File is on your Desktop. Wait for iCloud to sync it."
    else
        echo -e "${RED}❌ Backup Failed.${NC}"
    fi
}

function do_restore() {
    echo -e "${BLUE}📥 STARTING RESTORE${NC} (Cloud -> Work)"
    
    # 1. Find latest backup
    # Sort by time, take the last one
    LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/beta5_state_*.tar.gz 2>/dev/null | head -n 1)

    if [ -z "$LATEST_BACKUP" ]; then
        echo -e "${RED}❌ No backups found in $BACKUP_DIR${NC}"
        exit 1
    fi

    echo "Found Backup: $(basename "$LATEST_BACKUP")"
    read -p "❓ Restore this state? This will OVERWRITE local data. (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi

    # 2. Stop services forcefully
    echo "🛑 Stopping local services..."
    "$PROJECT_ROOT/stop.sh"

    # 3. Local Safety Backup (Just in case)
    echo "🛡️  Creating safety backup of current local state..."
    mkdir -p "$PROJECT_ROOT/data/safety_backup"
    cp -r "$PROJECT_ROOT/data/processed" "$PROJECT_ROOT/data/safety_backup/processed_$(date +%s)"
    
    # 4. Extract
    cd "$PROJECT_ROOT" || exit
    echo "📦 Extracting..."
    tar -xzf "$LATEST_BACKUP"
    
    echo -e "${GREEN}✅ Restore Complete!${NC}"
    echo "   You can now run ./start.sh"
}

# Main Dispatch
case "$1" in
    backup)
        do_backup
        ;;
    restore)
        do_restore
        ;;
    *)
        show_help
        ;;
esac
