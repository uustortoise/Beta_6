#!/usr/bin/env python3
"""
Deduplicate Archive Script
Scans the archive directory and removes duplicate files based on content hash.
Keeps the file without timestamp suffix (the "original").

Usage:
    python deduplicate_archive.py [--dry-run]
"""

import sys
import hashlib
from pathlib import Path

# Setup path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from elderlycare_v1_16.config.settings import ARCHIVE_DATA_DIR


def get_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def deduplicate_archive(dry_run=False):
    """Remove duplicate files from archive, keeping originals."""
    print(f"📂 Scanning archive directory: {ARCHIVE_DATA_DIR}")
    
    if not ARCHIVE_DATA_DIR.exists():
        print("❌ Archive directory does not exist.")
        return
    
    # Track files by hash
    hash_to_files = {}
    
    for date_dir in ARCHIVE_DATA_DIR.iterdir():
        if not date_dir.is_dir():
            continue
            
        for f in date_dir.glob("*.parquet"):
            file_hash = get_file_hash(f)
            if file_hash not in hash_to_files:
                hash_to_files[file_hash] = []
            hash_to_files[file_hash].append(f)
    
    # Find duplicates
    duplicates_to_remove = []
    
    for file_hash, files in hash_to_files.items():
        if len(files) > 1:
            # Sort: prefer files WITHOUT timestamp suffix (e.g., no _102530)
            # Original files have shorter names typically
            files_sorted = sorted(files, key=lambda x: len(x.stem))
            
            # Keep the first (shortest name = original), mark rest for deletion
            original = files_sorted[0]
            for dup in files_sorted[1:]:
                duplicates_to_remove.append((dup, original))
    
    if not duplicates_to_remove:
        print("✅ No duplicates found. Archive is clean.")
        return
    
    print(f"📊 Found {len(duplicates_to_remove)} duplicate files.")
    
    if dry_run:
        print("\n🔍 DRY RUN - No files will be deleted.\n")
        for dup, orig in duplicates_to_remove:
            print(f"   Would delete: {dup.name}")
            print(f"        (same as: {orig.name})")
        return
    
    print("\n🗑️ Removing duplicates...\n")
    
    removed = 0
    freed_bytes = 0
    
    for dup, orig in duplicates_to_remove:
        try:
            size = dup.stat().st_size
            dup.unlink()
            print(f"   ✅ Deleted: {dup.name}")
            removed += 1
            freed_bytes += size
        except Exception as e:
            print(f"   ❌ Failed to delete {dup.name}: {e}")
    
    print(f"\n✨ Cleanup complete!")
    print(f"   Removed: {removed} files")
    print(f"   Freed: {freed_bytes / 1024:.1f} KB")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    deduplicate_archive(dry_run=dry_run)
