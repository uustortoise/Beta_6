import json
from datetime import datetime, timedelta

from ml.hard_negative_mining import (
    ensure_hard_negative_table,
    fetch_hard_negative_queue,
    mark_hard_negative_applied,
    mark_hard_negative_status,
    mine_hard_negative_windows,
)


def _insert_elder(conn, elder_id: str):
    conn.execute(
        """
        INSERT INTO elders (elder_id, full_name)
        VALUES (?, ?)
        ON CONFLICT (elder_id) DO NOTHING
        """,
        (elder_id, elder_id),
    )


def test_mine_hard_negative_windows_inserts_candidates(test_db):
    elder_id = "HKHNM001"
    record_date = "2026-02-15"
    with test_db.get_connection() as conn:
        _insert_elder(conn, elder_id)
        start_ts = datetime.strptime(f"{record_date} 10:00:00", "%Y-%m-%d %H:%M:%S")
        for i in range(12):  # 2 minutes in 10s intervals
            ts = start_ts + timedelta(seconds=i * 10)
            sf = {
                "is_low_confidence": True,
                "low_confidence_hint_label": "kitchen_normal_use",
                "predicted_top1_label": "kitchen_normal_use",
            }
            conn.execute(
                """
                INSERT INTO adl_history
                (elder_id, record_date, timestamp, activity_type, confidence, room, is_anomaly, is_corrected, sensor_features)
                VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?)
                """,
                (
                    elder_id,
                    record_date,
                    ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "low_confidence",
                    0.30,
                    "Kitchen",
                    json.dumps(sf),
                ),
            )
        conn.commit()

        stats = mine_hard_negative_windows(
            conn=conn,
            elder_id=elder_id,
            record_date=record_date,
            risky_rooms=["kitchen"],
            top_k_per_room=3,
            min_block_rows=3,
            source="test",
        )
        conn.commit()

        assert int(stats["inserted"]) >= 1
        queue_df = fetch_hard_negative_queue(
            conn=conn,
            elder_id=elder_id,
            room="kitchen",
            days=7,
            status="open",
            limit=20,
        )
        assert not queue_df.empty
        assert "kitchen_normal_use" in queue_df["suggested_label"].astype(str).tolist()


def test_mark_hard_negative_status_and_applied(test_db):
    elder_id = "HKHNM002"
    record_date = "2026-02-15"
    with test_db.get_connection() as conn:
        ensure_hard_negative_table(conn)
        conn.execute(
            """
            INSERT INTO hard_negative_queue
            (elder_id, room, record_date, timestamp_start, timestamp_end, duration_minutes, reason, score, suggested_label, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                elder_id,
                "bathroom",
                record_date,
                f"{record_date} 03:00:00",
                f"{record_date} 03:05:00",
                5.0,
                "low_confidence_cluster",
                0.90,
                "bathroom_normal_use",
                "open",
            ),
        )
        conn.commit()

        qdf = fetch_hard_negative_queue(conn, elder_id=elder_id, room="bathroom", days=7, status="open")
        assert len(qdf) == 1
        row_id = int(qdf.iloc[0]["id"])
        changed = mark_hard_negative_status(conn, [row_id], status="queued")
        assert changed == 1
        mark_hard_negative_applied(
            conn=conn,
            elder_id=elder_id,
            room="Bathroom",
            timestamp_start=f"{record_date} 03:01:00",
            timestamp_end=f"{record_date} 03:04:00",
        )
        conn.commit()

        applied = fetch_hard_negative_queue(conn, elder_id=elder_id, room="bathroom", days=7, status="applied")
        assert len(applied) == 1
