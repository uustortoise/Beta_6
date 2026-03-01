"""
Reset Beta 5.5 development environment in one script.

Actions:
1) Truncate all PostgreSQL tables in public schema (dynamic discovery).
2) Clear data/archive and backend/logs contents.
3) Delete web-ui/.next/cache.
4) Remove __pycache__ directories (excluding venv/node_modules).
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import psycopg2
from psycopg2 import sql

# Add project root to path to import settings.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.elderlycare_v1_16.config.settings import POSTGRES_CONFIG

BASE_DIR = Path(__file__).resolve().parents[2]

DIRS_TO_CLEAR = [
    BASE_DIR / "data" / "archive",
    BASE_DIR / "data" / "raw",
    BASE_DIR / "backend" / "models",
    BASE_DIR / "backend" / "logs",
]

DIRS_TO_DELETE = [
    BASE_DIR / "web-ui" / ".next" / "cache",
]

VERIFY_TABLES = [
    "elders",
    "adl_history",
    "alerts",
    "activity_segments",
    "sleep_analysis",
    "model_training_history",
]


def _sanitize_conn_params(params):
    return {k: v if k != "password" else "***" for k, v in params.items()}


def _discover_public_tables(cur):
    cur.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
    )
    return [row[0] for row in cur.fetchall()]


def _truncate_all_tables(cur, table_names):
    if not table_names:
        print("No tables found in public schema. Database is already empty.")
        return
    print(f"Found {len(table_names)} tables: {', '.join(table_names)}")
    table_list = sql.SQL(", ").join(sql.Identifier(t) for t in table_names)
    stmt = sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE;").format(table_list)
    cur.execute(stmt)
    print("All discovered tables truncated and sequences restarted.")


def _verify_counts(cur, existing_tables):
    existing_set = set(existing_tables)
    targets = [t for t in VERIFY_TABLES if t in existing_set]
    if not targets:
        print("No verification targets present; skipping count verification.")
        return
    print("Verification counts (should be 0):")
    for table_name in targets:
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name)))
        count = cur.fetchone()[0]
        print(f"  - {table_name}: {count}")


def _clear_directory_contents(dir_path):
    if not dir_path.exists():
        print(f"Skipping missing directory: {dir_path}")
        return
    removed = 0
    for item in dir_path.iterdir():
        if item.name == ".DS_Store":
            continue
        if item.is_file():
            item.unlink()
            removed += 1
        elif item.is_dir():
            shutil.rmtree(item)
            removed += 1
    print(f"Cleared {removed} item(s) from {dir_path}")


def _delete_directory(dir_path):
    if dir_path.exists():
        shutil.rmtree(dir_path)
        print(f"Deleted directory: {dir_path}")
    else:
        print(f"Already clean: {dir_path}")


def _clear_pycache_dirs(base_dir):
    removed = 0
    for pycache in base_dir.rglob("__pycache__"):
        parts = pycache.parts
        if "venv" in parts or "node_modules" in parts:
            continue
        shutil.rmtree(pycache)
        removed += 1
    print(f"Removed {removed} __pycache__ directorie(s).")


def run_reset(force=False):
    if not force:
        confirm = input(
            "This will delete all Beta 5.5 dev data (DB tables, archives, caches). Continue? (y/n): "
        )
        if confirm.strip().lower() != "y":
            print("Reset aborted.")
            return 0

    conn_params = {k: v for k, v in POSTGRES_CONFIG.items() if k not in ["minconn", "maxconn"]}

    print(f"Starting reset from base dir: {BASE_DIR}")
    print("Step 1/4: Resetting PostgreSQL tables...")
    try:
        with psycopg2.connect(**conn_params) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                tables = _discover_public_tables(cur)
                _truncate_all_tables(cur, tables)
                _verify_counts(cur, tables)
    except Exception as exc:
        print(f"Database reset failed: {exc}")
        print(f"Connection params: {_sanitize_conn_params(conn_params)}")
        return 1

    print("Step 2/4: Clearing archive and logs...")
    for path in DIRS_TO_CLEAR:
        _clear_directory_contents(path)

    print("Step 3/4: Clearing Next.js cache...")
    for path in DIRS_TO_DELETE:
        _delete_directory(path)

    print("Step 4/4: Clearing Python cache...")
    _clear_pycache_dirs(BASE_DIR)

    # Recreate placeholder directories expected by dev workflows.
    (BASE_DIR / "data" / "archive").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "backend" / "models").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "backend" / "logs").mkdir(parents=True, exist_ok=True)

    print("Reset complete.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Reset Beta 5.5 dev environment.")
    parser.add_argument("-f", "--force", action="store_true", help="Skip confirmation prompt.")
    args = parser.parse_args()
    raise SystemExit(run_reset(force=args.force))


if __name__ == "__main__":
    main()
