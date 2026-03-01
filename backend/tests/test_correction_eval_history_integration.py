import sqlite3
from datetime import datetime, timedelta

import pandas as pd

from utils.correction_eval_history import fetch_and_enrich_correction_evaluations


def _seed_training_history(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            elder_id TEXT,
            training_date TIMESTAMP,
            model_type TEXT,
            epochs INTEGER,
            accuracy REAL,
            status TEXT,
            metadata TEXT
        )
        """
    )

    now = datetime(2026, 2, 13, 12, 0, 0)
    rows = [
        (
            "HK001",
            (now - timedelta(days=1)).isoformat(),
            "Correction Retrain",
            5,
            0.77,
            "success",
            '{"decision":"PASS","artifact_path":"/tmp/pass.json","corrected_window_report":{"local_gain":0.11,"global_drop":0.01}}',
        ),
        (
            "HK001",
            (now - timedelta(days=2)).isoformat(),
            "Correction Retrain",
            5,
            0.71,
            "rejected_by_corrected_window_policy",
            '{"decision":"FAIL","artifact_path":"/tmp/fail.json","corrected_window_report":{"local_gain":-0.03,"global_drop":0.05}}',
        ),
        (
            "HK002",
            (now - timedelta(days=1)).isoformat(),
            "Unified Transformer (Auto-Aggregate)",
            5,
            0.80,
            "success",
            '{"note":"non-correction row should be excluded"}',
        ),
        (
            "HK003",
            (now - timedelta(days=40)).isoformat(),
            "Correction Retrain",
            5,
            0.69,
            "pass_with_flag",
            '{"decision":"PASS_WITH_FLAG","artifact_path":"/tmp/flag.json","corrected_window_report":{"local_gain":0.02,"global_drop":0.00}}',
        ),
    ]
    conn.executemany(
        """
        INSERT INTO training_history
        (elder_id, training_date, model_type, epochs, accuracy, status, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def test_fetch_and_enrich_correction_evaluations_filters_and_maps_fields():
    conn = sqlite3.connect(":memory:")
    try:
        _seed_training_history(conn)
        now = datetime(2026, 2, 13, 12, 0, 0)

        def query_fn(query, params):
            return pd.read_sql_query(query, conn, params=params)

        df = fetch_and_enrich_correction_evaluations(
            query_fn=query_fn,
            elder_filter="HK001",
            days=7,
            now=now,
        )

        # Only HK001 correction retrain rows in last 7 days
        assert len(df) == 2
        assert set(df["elder_id"]) == {"HK001"}
        assert set(df["model_type"]) == {"Correction Retrain"}
        assert set(df["decision"]) == {"PASS", "FAIL"}

        # Field extraction checks
        pass_row = df[df["decision"] == "PASS"].iloc[0]
        fail_row = df[df["decision"] == "FAIL"].iloc[0]
        assert pass_row["artifact_path"] == "/tmp/pass.json"
        assert abs(float(pass_row["local_gain"]) - 0.11) < 1e-9
        assert abs(float(fail_row["global_drop"]) - 0.05) < 1e-9
    finally:
        conn.close()
