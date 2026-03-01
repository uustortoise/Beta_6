#!/usr/bin/env python3
"""
Data Integrity Check System
===========================
Validates database consistency to catch anomalies before they reach production UI.

Usage:
    python scripts/check_integrity.py                    # Check all
    python scripts/check_integrity.py --elder HK001_jessica  # Check specific elder
    python scripts/check_integrity.py --date 2025-12-18      # Check specific date

Exit codes:
    0 = All checks passed
    1 = One or more checks failed
"""

import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path
from backend.db.legacy_adapter import LegacyDatabaseAdapter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from elderlycare_v1_16.database import DatabaseManager

# ANSI colors for CLI output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def log_pass(check_name: str, details: str = ""):
    print(f"  {Colors.GREEN}✓{Colors.RESET} {check_name} {Colors.BLUE}{details}{Colors.RESET}")

def log_fail(check_name: str, count: int, details: str = ""):
    print(f"  {Colors.RED}✗{Colors.RESET} {check_name} - {Colors.RED}{count} issues{Colors.RESET} {details}")

def log_warn(check_name: str, details: str = ""):
    print(f"  {Colors.YELLOW}⚠{Colors.RESET} {check_name} {details}")


class IntegrityChecker:
    """Production-grade data integrity validator."""
    
    def __init__(self, db_path: str = None):
        # We ignore db_path in Postgres mode as adapter handles connection
        self.adapter = LegacyDatabaseAdapter()
        self.db_path = "PostgreSQL"
        
        self.issues = []
        self.checks_passed = 0
        self.checks_failed = 0
    
    def run_all_checks(self, elder_id: str = None, record_date: str = None) -> bool:
        """Run all integrity checks. Returns True if all pass."""
        print(f"\n{Colors.BOLD}═══════════════════════════════════════════════════════════{Colors.RESET}")
        print(f"{Colors.BOLD}  DATA INTEGRITY CHECK{Colors.RESET}")
        print(f"{Colors.BOLD}═══════════════════════════════════════════════════════════{Colors.RESET}")
        print(f"  Database: {self.db_path}")
        if elder_id:
            print(f"  Elder: {elder_id}")
        if record_date:
            print(f"  Date: {record_date}")
        print(f"{Colors.BOLD}───────────────────────────────────────────────────────────{Colors.RESET}\n")
        
        # Run individual checks
        self.check_overlapping_segments(elder_id, record_date)
        self.check_orphan_records(elder_id)
        self.check_date_mismatches(elder_id, record_date)
        self.check_future_timestamps(elder_id)
        self.check_confidence_bounds(elder_id, record_date)
        
        # Summary
        print(f"\n{Colors.BOLD}───────────────────────────────────────────────────────────{Colors.RESET}")
        total = self.checks_passed + self.checks_failed
        if self.checks_failed == 0:
            print(f"  {Colors.GREEN}{Colors.BOLD}ALL {total} CHECKS PASSED ✓{Colors.RESET}")
        else:
            print(f"  {Colors.RED}{Colors.BOLD}{self.checks_failed}/{total} CHECKS FAILED ✗{Colors.RESET}")
        print(f"{Colors.BOLD}═══════════════════════════════════════════════════════════{Colors.RESET}\n")
        
        return self.checks_failed == 0
    
    def check_overlapping_segments(self, elder_id: str = None, record_date: str = None):
        """Check for segments with overlapping time ranges in the same room."""
        query = """
            SELECT a.elder_id, a.room, a.record_date, 
                   a.start_time as seg1_start, a.end_time as seg1_end, a.activity_type as seg1_activity,
                   b.start_time as seg2_start, b.end_time as seg2_end, b.activity_type as seg2_activity
            FROM activity_segments a
            JOIN activity_segments b ON a.elder_id = b.elder_id 
                AND a.room = b.room 
                AND a.record_date = b.record_date
                AND a.id < b.id
            WHERE (a.start_time < b.end_time AND a.end_time > b.start_time)
        """
        params = []
        
        if elder_id:
            query += " AND a.elder_id = ?"
            params.append(elder_id)
        if record_date:
            query += " AND a.record_date = ?"
            params.append(record_date)
        
        if record_date:
            query += " AND a.record_date = ?"
            params.append(record_date)
        
        with self.adapter.get_connection() as conn:
            cursor = conn.execute(query, params)
            overlaps = cursor.fetchall()
        
        if not overlaps:
            log_pass("Overlapping Segments", "(none found)")
            self.checks_passed += 1
        else:
            log_fail("Overlapping Segments", len(overlaps))
            for o in overlaps[:3]:  # Show first 3
                self.issues.append({
                    "type": "overlapping_segment",
                    "elder": o[0], "room": o[1], "date": o[2],
                    "seg1": f"{o[5]} ({o[3]} to {o[4]})",
                    "seg2": f"{o[8]} ({o[6]} to {o[7]})"
                })
                print(f"      → {o[1]}: '{o[5]}' overlaps with '{o[8]}' on {o[2]}")
            self.checks_failed += 1
    
    def check_orphan_records(self, elder_id: str = None):
        """Check for records referencing non-existent elders."""
        query = """
            SELECT DISTINCT a.elder_id 
            FROM adl_history a
            LEFT JOIN elders e ON a.elder_id = e.elder_id
            WHERE e.elder_id IS NULL
        """
        params = []
        if elder_id:
            query += " AND a.elder_id = ?"
            params.append(elder_id)
        
        with self.adapter.get_connection() as conn:
            cursor = conn.execute(query, params)
            orphans = cursor.fetchall()
        
        if not orphans:
            log_pass("Orphan Records", "(all records have valid elder refs)")
            self.checks_passed += 1
        else:
            log_fail("Orphan Records", len(orphans))
            for o in orphans:
                self.issues.append({"type": "orphan", "elder_id": o[0]})
                print(f"      → Elder '{o[0]}' referenced but not in elders table")
            self.checks_failed += 1
    
    def check_date_mismatches(self, elder_id: str = None, record_date: str = None):
        """Check that record_date matches the date portion of start_time."""
        query = """
            SELECT id, elder_id, room, record_date, start_time
            FROM activity_segments
            WHERE date(start_time) != record_date
        """
        params = []
        if elder_id:
            query += " AND elder_id = ?"
            params.append(elder_id)
        if record_date:
            query += " AND record_date = ?"
            params.append(record_date)
        query += " LIMIT 10"
        
        with self.adapter.get_connection() as conn:
            cursor = conn.execute(query, params)
            mismatches = cursor.fetchall()
        
        if not mismatches:
            log_pass("Date Mismatches", "(record_date matches timestamp)")
            self.checks_passed += 1
        else:
            log_fail("Date Mismatches", len(mismatches))
            for m in mismatches[:3]:
                print(f"      → ID {m[0]}: record_date={m[3]} but start_time={m[4]}")
            self.checks_failed += 1
    
    def check_future_timestamps(self, elder_id: str = None):
        """Check for timestamps in the future."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query = f"""
            SELECT COUNT(*) FROM adl_history
            WHERE timestamp > '{now}'
        """
        params = []
        if elder_id:
            query = query.replace("WHERE", f"WHERE elder_id = '{elder_id}' AND")
        
        with self.adapter.get_connection() as conn:
            cursor = conn.execute(query)
            count = cursor.fetchone()[0]
        
        if count == 0:
            log_pass("Future Timestamps", "(none found)")
            self.checks_passed += 1
        else:
            log_fail("Future Timestamps", count)
            self.issues.append({"type": "future_timestamp", "count": count})
            self.checks_failed += 1
    
    def check_confidence_bounds(self, elder_id: str = None, record_date: str = None):
        """Check that confidence values are within valid range [0.0, 1.0]."""
        query = """
            SELECT COUNT(*) FROM adl_history
            WHERE confidence < 0.0 OR confidence > 1.0
        """
        params = []
        conditions = []
        if elder_id:
            conditions.append(f"elder_id = '{elder_id}'")
        if record_date:
            conditions.append(f"record_date = '{record_date}'")
        if conditions:
            query = query.replace("WHERE", "WHERE " + " AND ".join(conditions) + " AND")
        
        with self.adapter.get_connection() as conn:
            cursor = conn.execute(query)
            count = cursor.fetchone()[0]
        
        if count == 0:
            log_pass("Confidence Bounds", "(all values in [0.0, 1.0])")
            self.checks_passed += 1
        else:
            log_fail("Confidence Bounds", count, "values outside [0.0, 1.0]")
            self.checks_failed += 1
    
    def get_issues_json(self) -> str:
        """Return issues as JSON for logging."""
        return json.dumps(self.issues, indent=2)


def run_integrity_check(elder_id: str = None, record_date: str = None) -> bool:
    """
    Run integrity check. Called by process_data.py after file processing.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    checker = IntegrityChecker()
    return checker.run_all_checks(elder_id, record_date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Integrity Check")
    parser.add_argument("--elder", help="Check specific elder ID")
    parser.add_argument("--date", help="Check specific date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    success = run_integrity_check(args.elder, args.date)
    sys.exit(0 if success else 1)
