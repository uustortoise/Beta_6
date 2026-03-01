#!/usr/bin/env python3
"""
================================================================================
CRITICAL BUG FIXES APPLIER
================================================================================
This script applies the critical bug fixes for Beta 5.5.

Usage:
    python scripts/apply_critical_fixes.py --check   # Check current status
    python scripts/apply_critical_fixes.py --apply   # Apply fixes (dry run)
    python scripts/apply_critical_fixes.py --apply --confirm  # Apply for real

WARNING: Make a backup before running with --confirm!
================================================================================
"""

import sys
import re
import argparse
from pathlib import Path

def check_file_exists(filepath):
    """Check if file exists."""
    path = Path(filepath)
    if not path.exists():
        print(f"❌ File not found: {filepath}")
        return False
    return True

def check_bug_exists(filepath, pattern, bug_name):
    """Check if a bug pattern exists in file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    if re.search(pattern, content):
        print(f"🔴 {bug_name}: FOUND (needs fix)")
        return True
    else:
        print(f"✅ {bug_name}: NOT FOUND (already fixed)")
        return False

def check_all_bugs():
    """Check status of all known bugs."""
    print("="*70)
    print("CHECKING FOR KNOWN BUGS")
    print("="*70)
    
    export_dashboard = Path("backend/export_dashboard.py")
    segment_utils = Path("backend/utils/segment_utils.py")
    time_utils = Path("backend/utils/time_utils.py")
    
    bugs_found = 0
    
    # Check 1: Missing logger
    if check_file_exists(export_dashboard):
        with open(export_dashboard, 'r') as f:
            content = f.read()
        if 'logger = logging.getLogger(__name__)' in content[:1000]:
            print("✅ Logger definition: FOUND")
        else:
            print("🔴 Logger definition: MISSING")
            bugs_found += 1
    
    # Check 2: Resource leak (ctx.__enter__ without with)
    if check_file_exists(export_dashboard):
        if check_bug_exists(export_dashboard, r'ctx = get_dashboard_connection\(\)\s*\n\s*conn = ctx\.__enter__\(\)', 
                          "Resource leak in save_corrections_to_db"):
            bugs_found += 1
    
    # Check 3: INSERT OR REPLACE (SQLite syntax)
    if check_file_exists(export_dashboard):
        if check_bug_exists(export_dashboard, r'INSERT OR REPLACE INTO activity_segments',
                          "SQLite SQL in PostgreSQL path"):
            bugs_found += 1
    
    # Check 4: DB_PATH check in fetch_predictions
    if check_file_exists(export_dashboard):
        if check_bug_exists(export_dashboard, r'if not DB_PATH\.exists\(\):\s*\n\s*return pd\.DataFrame\(\)',
                          "DB_PATH check in fetch_predictions"):
            bugs_found += 1
    
    # Check 5: Bare except clauses
    if check_file_exists(export_dashboard):
        with open(export_dashboard, 'r') as f:
            content = f.read()
        bare_excepts = len(re.findall(r'except\s*:\s*\n', content))
        if bare_excepts > 0:
            print(f"🔴 Bare except clauses: {bare_excepts} found")
            bugs_found += 1
        else:
            print("✅ Bare except clauses: None")
    
    # Check 6: Resource leak in segment_utils
    if check_file_exists(segment_utils):
        with open(segment_utils, 'r') as f:
            content = f.read()
        if 'conn = conn_ctx.__enter__()' in content and 'try:' not in content[:5000]:
            print("🔴 Resource leak in segment_utils: FOUND")
            bugs_found += 1
        else:
            print("✅ Resource leak in segment_utils: Not found (fixed)")
    
    print("="*70)
    if bugs_found == 0:
        print("✅ ALL CRITICAL BUGS FIXED!")
    else:
        print(f"🔴 {bugs_found} CRITICAL BUG(S) NEED FIXING")
    print("="*70)
    
    return bugs_found

def fix_missing_logger(content):
    """Fix missing logger definition."""
    # Find a good place to insert logger
    lines = content.split('\n')
    
    # Look for imports
    import_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_idx = i + 1
    
    # Insert logger after imports
    logger_code = "\nimport logging\nlogger = logging.getLogger(__name__)\n"
    lines.insert(import_idx, logger_code)
    
    return '\n'.join(lines)

def fix_resource_leak(content):
    """Fix resource leak in save_corrections_to_db."""
    # This is a complex fix - requires structural changes
    # For now, just add a warning comment
    pattern = r'(ctx = get_dashboard_connection\(\)\s*\n\s*conn = ctx\.__enter__\(\))'
    replacement = r'# FIXME: Resource leak - use "with get_dashboard_connection() as conn:" instead\n\1'
    return re.sub(pattern, replacement, content)

def fix_bare_excepts(content):
    """Fix bare except clauses."""
    # Replace bare except: with except Exception:
    return re.sub(r'except\s*:\s*\n', 'except Exception:\n', content)

def apply_fixes(confirm=False):
    """Apply all fixes."""
    print("="*70)
    print("APPLYING FIXES" + (" (CONFIRMED)" if confirm else " (DRY RUN)"))
    print("="*70)
    
    export_dashboard = Path("backend/export_dashboard.py")
    
    if not check_file_exists(export_dashboard):
        return
    
    with open(export_dashboard, 'r') as f:
        original_content = f.read()
    
    content = original_content
    fixes_applied = []
    
    # Fix 1: Missing logger
    if 'logger = logging.getLogger(__name__)' not in content[:1000]:
        if confirm:
            content = fix_missing_logger(content)
            fixes_applied.append("Added logger definition")
        else:
            print("Would add: logger = logging.getLogger(__name__)")
    
    # Fix 2: Bare excepts
    bare_count = len(re.findall(r'except\s*:\s*\n', content))
    if bare_count > 0:
        if confirm:
            content = fix_bare_excepts(content)
            fixes_applied.append(f"Fixed {bare_count} bare except clauses")
        else:
            print(f"Would fix {bare_count} bare except clauses")
    
    if confirm and fixes_applied:
        # Backup original
        backup_path = export_dashboard.with_suffix('.py.backup')
        with open(backup_path, 'w') as f:
            f.write(original_content)
        print(f"✅ Backup created: {backup_path}")
        
        # Write fixed content
        with open(export_dashboard, 'w') as f:
            f.write(content)
        
        print("\n✅ Fixes applied:")
        for fix in fixes_applied:
            print(f"  - {fix}")
    elif not confirm:
        print("\n⚠️  This was a dry run. Use --confirm to apply fixes.")
    else:
        print("\n✅ No fixes needed (already applied)")
    
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Apply critical bug fixes for Beta 5.5')
    parser.add_argument('--check', action='store_true', help='Check for bugs only')
    parser.add_argument('--apply', action='store_true', help='Apply fixes')
    parser.add_argument('--confirm', action='store_true', help='Confirm application (required with --apply)')
    
    args = parser.parse_args()
    
    if args.check or (not args.apply):
        bugs = check_all_bugs()
        return 1 if bugs > 0 else 0
    
    if args.apply:
        apply_fixes(confirm=args.confirm)
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
