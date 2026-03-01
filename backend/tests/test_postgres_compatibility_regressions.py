from backend.db.legacy_adapter import PostgresConnectionShim
from elderlycare_v1_16.services.adl_service import ADLService
from elderlycare_v1_16.services.icope_service import ICOPEService
from elderlycare_v1_16.services.insight_service import InsightService
from elderlycare_v1_16.services.sleep_service import SleepService


class _StubPsycopgCursor:
    def __init__(self):
        self.last_execute = None
        self.last_executemany = None

    def execute(self, sql, params=()):
        self.last_execute = (sql, params)

    def executemany(self, sql, params_list):
        self.last_executemany = (sql, params_list)


class _StubPsycopgConnection:
    def __init__(self):
        self.cursors = []

    def cursor(self):
        cursor = _StubPsycopgCursor()
        self.cursors.append(cursor)
        return cursor


class _FakeQueryResult:
    def __init__(self, one=None, all_rows=None):
        self._one = one
        self._all_rows = [] if all_rows is None else all_rows

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._all_rows


class _FakeConnection:
    def __init__(self, results):
        self._results = list(results)
        self.queries = []

    def execute(self, query, params=()):
        self.queries.append((query, params))
        return self._results.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class _FakeDB:
    def __init__(self, connection):
        self.connection = connection

    def get_connection(self):
        return self.connection


def test_postgres_connection_shim_executemany_delegates_and_translates():
    pg_conn = _StubPsycopgConnection()
    shim = PostgresConnectionShim(pg_conn, db_proxy=None)

    cursor = shim.executemany(
        "INSERT OR IGNORE INTO household_behavior (elder_id, confidence) VALUES (?, ?)",
        [("HK001", True), ("HK002", False)],
    )

    assert cursor.pg_cursor is pg_conn.cursors[-1]
    sql, params = cursor.pg_cursor.last_executemany
    assert "%s" in sql
    assert "ON CONFLICT DO NOTHING" in sql.upper()
    assert params == [("HK001", 1), ("HK002", 0)]


def test_insight_count_adl_event_handles_tuple_rows():
    connection = _FakeConnection([_FakeQueryResult(one=(3,))])
    service = InsightService()
    service.db = _FakeDB(connection)

    count = service._count_adl_event("HK001", "2026-02-09", "toileting")

    assert count == 3
    assert "activity_type IN ('toileting', 'toilet')" in connection.queries[0][0]


def test_insight_fetch_enabled_rules_handles_tuple_rows():
    tuple_row = (
        1,
        "Hypertension Risk Rule",
        "hypertension",
        '{"logic":"AND","rules":[]}',
        "Risk detected",
        "high",
        1,
        "2026-02-09 00:00:00",
    )
    connection = _FakeConnection([_FakeQueryResult(all_rows=[tuple_row])])
    service = InsightService()
    service.db = _FakeDB(connection)

    rules = service._fetch_enabled_rules()

    assert len(rules) == 1
    assert rules[0]["rule_name"] == "Hypertension Risk Rule"
    assert rules[0]["required_condition"] == "hypertension"
    assert rules[0]["enabled"] == 1


def test_adl_get_todays_events_handles_tuple_rows():
    tuple_row = (
        101,
        "HK001",
        "2026-02-09",
        "2026-02-09 08:00:00",
        "toileting",
        3,
        0.96,
        "Bathroom",
        0,
        1,
        '{"motion": 0.8}',
    )
    connection = _FakeConnection([_FakeQueryResult(all_rows=[tuple_row])])
    service = ADLService()
    service.db = _FakeDB(connection)

    rows = service.get_todays_events("HK001")

    assert len(rows) == 1
    assert rows[0]["id"] == 101
    assert rows[0]["activity_type"] == "toileting"
    assert rows[0]["room"] == "Bathroom"


def test_sleep_get_latest_handles_tuple_rows():
    tuple_row = (
        9,
        "HK001",
        "2026-02-09",
        7.5,
        88.2,
        '{"Deep":15,"REM":20,"Light":55,"Awake":10}',
        82.0,
        "Stable sleep",
    )
    connection = _FakeConnection([_FakeQueryResult(one=tuple_row)])
    service = SleepService()
    service.db = _FakeDB(connection)

    row = service.get_latest_sleep("HK001")

    assert row is not None
    assert row["analysis_date"] == "2026-02-09"
    assert row["quality_score"] == 82.0


def test_icope_get_latest_handles_tuple_rows():
    tuple_row = (
        33,
        "HK001",
        "2026-02-09",
        78.0,
        74.0,
        70.0,
        90.0,
        72.0,
        77.0,
        '["keep walking"]',
        "stable",
    )
    connection = _FakeConnection([_FakeQueryResult(one=tuple_row)])
    service = ICOPEService()
    service.db = _FakeDB(connection)

    row = service.get_latest_assessment("HK001")

    assert row is not None
    assert row["id"] == 33
    assert row["overall_score"] == 77.0
    assert row["trend"] == "stable"
