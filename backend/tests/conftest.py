
import pytest
import os
import sys
import tempfile
from pathlib import Path

# Ensure project root and backend/ are importable.
# Many modules use `import backend...` (namespace package), which requires project root on sys.path.
backend_path = Path(__file__).resolve().parent.parent
project_root = backend_path.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_path))

from elderlycare_v1_16.database import DatabaseManager


@pytest.fixture(scope="function")
def test_db(monkeypatch):
    """
    Creates a temporary database for each test.
    """
    # Create a temporary file
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    
    import backend.db.database as db_module
    import backend.db.legacy_adapter as legacy_adapter_module

    # Force test runtime onto isolated SQLite DB.
    monkeypatch.setattr(db_module, "POSTGRES_ONLY", False)
    monkeypatch.setattr(db_module, "USE_POSTGRESQL", False)
    monkeypatch.setattr(legacy_adapter_module, "POSTGRES_ONLY", False)

    # Reset lazy DB proxy and point SQLite backend to the test file.
    test_dual_db = db_module.DualWriteDatabase()
    test_dual_db._sqlite_db = db_module.SQLiteDatabase(db_path=db_path)
    test_dual_db._pg_db = None
    monkeypatch.setattr(db_module.db, "_db", test_dual_db)

    # Reset adapter singleton to ensure clean test state.
    DatabaseManager._instance = None
    db_mgr = DatabaseManager()
    
    # Initialize Schema
    with open(backend_path / "elderlycare_v1_16/models/schema.sql", "r") as f:
        schema_sql = f.read()
        
    with db_mgr.get_connection() as conn:
        conn.executescript(schema_sql)
        
    yield db_mgr
    
    # Teardown
    # No need to close explicitly if using LegacyAdapter, it returns to pool
    if hasattr(db_mgr, 'close'):
        db_mgr.close()
    
    os.close(db_fd)
    os.unlink(db_path)
    DatabaseManager._instance = None

@pytest.fixture
def clean_env(monkeypatch):
    """
    Ensures environment variables don't leak between tests.
    """
    monkeypatch.setenv("ENV", "testing")
