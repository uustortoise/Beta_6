#!/usr/bin/env python3
"""
Migration Script: Convert existing xlsx archive files to Parquet format.
Run this once to compress historical archive files and reclaim ~90% storage.

Usage:
    python migrate_archive.py [--dry-run]
"""

import sys
import os
from pathlib import Path

# Setup path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from elderlycare_v1_16.config.settings import ARCHIVE_DATA_DIR
from utils.data_loader import convert_xlsx_to_parquet


def migrate_archive(dry_run=False):
    """Convert all xlsx files in archive to parquet."""
    print(f"📂 Scanning archive directory: {ARCHIVE_DATA_DIR}")
    
    if not ARCHIVE_DATA_DIR.exists():
        print("❌ Archive directory does not exist.")
        return
    
    xlsx_files = list(ARCHIVE_DATA_DIR.rglob("*.xlsx"))
    
    if not xlsx_files:
        print("✅ No xlsx files found. Archive is already optimized.")
        return
    
    print(f"📊 Found {len(xlsx_files)} xlsx files to convert.")
    
    # Calculate current size
    total_size_before = sum(f.stat().st_size for f in xlsx_files)
    print(f"   Current size: {total_size_before / 1024 / 1024:.1f} MB")
    
    if dry_run:
        print("\n🔍 DRY RUN - No files will be modified.\n")
        for f in xlsx_files:
            print(f"   Would convert: {f.name}")
        return
    
    print("\n🔄 Converting files...\n")
    
    converted = 0
    errors = 0
    
    for xlsx_file in xlsx_files:
        try:
            parquet_path = convert_xlsx_to_parquet(xlsx_file, delete_original=True)
            print(f"   ✅ {xlsx_file.name} → {parquet_path.name}")
            converted += 1
        except Exception as e:
            print(f"   ❌ {xlsx_file.name}: {e}")
            errors += 1
    
    # Calculate new size
    parquet_files = list(ARCHIVE_DATA_DIR.rglob("*.parquet"))
    total_size_after = sum(f.stat().st_size for f in parquet_files)
    
    print(f"\n✨ Migration complete!")
    print(f"   Converted: {converted} files")
    print(f"   Errors: {errors}")
    print(f"   Size before: {total_size_before / 1024 / 1024:.1f} MB")
    print(f"   Size after: {total_size_after / 1024 / 1024:.1f} MB")
    print(f"   Savings: {(1 - total_size_after / total_size_before) * 100:.0f}%")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    migrate_archive(dry_run=dry_run)
