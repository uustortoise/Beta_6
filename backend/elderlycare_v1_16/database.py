"""
This module has been replaced by the Dual-Write Adapter for the PostgreSQL Migration.
It redirects all calls to backend.db.legacy_adapter.LegacyDatabaseAdapter.
"""
from backend.db.legacy_adapter import LegacyDatabaseAdapter

# Expose the singleton instance as `db`
db = LegacyDatabaseAdapter()

# Re-export DatabaseManager class for type-hint compatibility if needed
# (Though users should rely on the `db` instance)
DatabaseManager = LegacyDatabaseAdapter
