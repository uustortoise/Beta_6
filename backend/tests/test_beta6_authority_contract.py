from types import SimpleNamespace

from backend.utils import beta6_authority_contract as authority_contract


def test_check_postgresql_preflight_fails_when_use_postgresql_disabled(monkeypatch):
    monkeypatch.setattr(authority_contract, "USE_POSTGRESQL", False)

    ok, details = authority_contract.check_postgresql_preflight()

    assert ok is False
    assert "USE_POSTGRESQL=false" in str(details.get("error", ""))


def test_check_postgresql_preflight_fails_when_pg_db_missing(monkeypatch):
    monkeypatch.setattr(authority_contract, "USE_POSTGRESQL", True)
    monkeypatch.setattr(authority_contract, "dual_write_db", SimpleNamespace(pg_db=None))

    ok, details = authority_contract.check_postgresql_preflight()

    assert ok is False
    assert "PostgreSQL unavailable" in str(details.get("error", ""))


def test_check_postgresql_preflight_passes_and_returns_connection(monkeypatch):
    monkeypatch.setattr(authority_contract, "USE_POSTGRESQL", True)

    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query):
            self.query = query

        def fetchone(self):
            return (1,)

    class _Conn:
        def cursor(self):
            return _Cursor()

    returned = {"called": False}

    class _Pg:
        def get_raw_connection(self):
            return _Conn()

        def return_connection(self, conn):  # noqa: ARG002
            returned["called"] = True

    monkeypatch.setattr(authority_contract, "dual_write_db", SimpleNamespace(pg_db=_Pg()))

    ok, details = authority_contract.check_postgresql_preflight()

    assert ok is True
    assert details.get("status") == "ok"
    assert returned["called"] is True
