import sqlite3

import pandas as pd

from backend.utils.intelligence_db import coerce_timestamp_column, query_to_dataframe


def test_query_to_dataframe_with_sqlite_connection():
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("CREATE TABLE adl_history (elder_id TEXT, timestamp TEXT, activity_type TEXT)")
    cur.executemany(
        "INSERT INTO adl_history (elder_id, timestamp, activity_type) VALUES (?, ?, ?)",
        [
            ("HK001", "2026-02-09 08:00:00", "toilet"),
            ("HK001", "2026-02-09 08:05:00", "kitchen_normal_use"),
        ],
    )
    conn.commit()

    df = query_to_dataframe(
        conn,
        "SELECT elder_id, timestamp, activity_type FROM adl_history WHERE elder_id = ? ORDER BY timestamp",
        ("HK001",),
    )

    assert list(df.columns) == ["elder_id", "timestamp", "activity_type"]
    assert len(df) == 2
    assert df.iloc[0]["activity_type"] == "toilet"


def test_coerce_timestamp_column_drops_invalid_rows():
    df = pd.DataFrame(
        {
            "timestamp": ["2026-02-09 08:00:00", "timestamp", None],
            "value": [1, 2, 3],
        }
    )

    cleaned = coerce_timestamp_column(df, "timestamp", "unit-test")

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["value"] == 1
    assert str(cleaned.iloc[0]["timestamp"]) == "2026-02-09 08:00:00"
