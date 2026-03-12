import pytest
from datetime import date, datetime
import pandas as pd
import json

from services.export_service import get_residents, export_predicted_results
from services.audit_service import fetch_correction_trail, rollback_correction
from services.correction_service import (
    apply_batch_corrections,
    build_compare_timeline_payload,
    find_nearest_activity_date,
    get_activity_timeline,
    get_activity_timeline_any_room,
    get_low_confidence_queue,
    get_sensor_timeseries,
    get_training_timeline,
    preview_batch_corrections,
    save_correction,
)
import services.ops_service as ops_service
from services.ops_service import get_daily_ops_summary, get_model_status


def test_export_service_predicted_results(test_db):
    """Test predicted results extraction includes the confidence score."""
    start_date = date(2026, 1, 1)
    end_date = date(2026, 1, 31)
    
    # 1. Manually seed test database
    with test_db.get_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions 
            (resident_id, timestamp, room, activity, confidence, is_anomaly)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("HK001", "2026-01-15 10:00:00", "bathroom", "shower", 0.95, 0)
        )
        conn.commit()

    # 2. Call Service
    df = export_predicted_results("HK001", start_date, end_date)
    
    # 3. Assert
    assert not df.empty
    assert len(df) == 1
    assert "confidence" in df.columns
    assert df.iloc[0]["predicted_activity"] == "shower"
    assert df.iloc[0]["confidence"] == 0.95


def test_export_service_get_residents_collects_multiple_sources(test_db):
    """Resident discovery should include IDs from multiple operational tables."""
    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", ("R_ELDER", "Resident Elder"))
        conn.execute(
            "INSERT INTO training_history (elder_id, model_type, status) VALUES (?, ?, ?)",
            ("R_TRAIN", "test", "ok"),
        )
        conn.execute(
            """
            INSERT INTO correction_history
            (elder_id, room, timestamp_start, timestamp_end, old_activity, new_activity, rows_affected, corrected_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("R_CORR", "bedroom", "2026-03-01 09:00:00", "2026-03-01 09:10:00", "sleep", "nap", 1, "tester"),
        )
        conn.execute(
            """
            INSERT INTO predictions (resident_id, timestamp, room, activity, confidence, is_anomaly)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("R_PRED", "2026-03-01 10:00:00", "kitchen", "cooking", 0.88, 0),
        )
        conn.commit()

    residents = get_residents()
    assert "R_ELDER" in residents
    assert "R_TRAIN" in residents
    assert "R_CORR" in residents
    assert "R_PRED" in residents


def test_correction_service_save_and_undo(test_db):
    """Test atomic saving and undo operations for corrections."""
    elder = "HK002"
    room = "livingroom"
    
    # 1. Seed ADL history
    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Test"))
        conn.execute(
            """
            INSERT INTO adl_history (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-02-01 12:00:00", room, "low_confidence", 0.45, 0, "2026-02-01", 10)
        )
        conn.commit()

    # 2. Assert initial queue
    queue = get_low_confidence_queue(elder, room, date(2026, 2, 1), 0.50)
    assert len(queue) == 1
    
    # 3. Save Correction
    rows = [{"timestamp": "2026-02-01 12:00:00", "activity_type": "low_confidence"}]
    success, msg, count = save_correction(elder, room, rows, "watch_tv", "tester")
    assert success
    assert count == 1
    
    # 4. Verify ADL history was updated
    with test_db.get_connection() as conn:
        updated = conn.execute("SELECT activity_type, is_corrected FROM adl_history").fetchone()
        assert updated["activity_type"] == "watch_tv"
        assert updated["is_corrected"] == 1
        
    # 5. Verify Audit Trail was written
    audit = fetch_correction_trail(elder, room, "tester")
    assert len(audit) == 1
    assert audit.iloc[0]["new_activity"] == "watch_tv"
    correction_id = int(audit.iloc[0]["id"])
    
    # 6. Test Rollback
    rb_success, rb_msg = rollback_correction(correction_id)
    assert rb_success
    
    # 7. Verify soft delete and restore
    audit_after = fetch_correction_trail(elder, room, "tester")
    assert audit_after.empty # Default excludes deleted
    
    with test_db.get_connection() as conn:
        restored = conn.execute("SELECT activity_type, is_corrected FROM adl_history").fetchone()
        assert restored["activity_type"] == "low_confidence"
        assert restored["is_corrected"] == 0


def test_correction_rollback_restores_only_selected_rows(test_db):
    """Rollback should restore only corrected timestamps, not unrelated rows in the time span."""
    elder = "HK002B"
    room = "livingroom"

    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Rollback Precision"))
        conn.execute(
            """
            INSERT INTO adl_history (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-02-02 12:00:00", room, "low_confidence", 0.35, 0, "2026-02-02", 10),
        )
        conn.execute(
            """
            INSERT INTO adl_history (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-02-02 12:00:20", room, "cooking", 0.91, 0, "2026-02-02", 10),
        )
        conn.execute(
            """
            INSERT INTO adl_history (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-02-02 12:00:40", room, "low_confidence", 0.38, 0, "2026-02-02", 10),
        )
        conn.commit()

    selected_rows = [
        {"timestamp": "2026-02-02 12:00:00", "activity_type": "low_confidence"},
        {"timestamp": "2026-02-02 12:00:40", "activity_type": "low_confidence"},
    ]
    success, msg, count = save_correction(elder, room, selected_rows, "watch_tv", "tester")
    assert success
    assert count == 2

    audit = fetch_correction_trail(elder, room, "tester")
    assert len(audit) == 1
    correction_id = int(audit.iloc[0]["id"])
    rb_success, _ = rollback_correction(correction_id)
    assert rb_success

    with test_db.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT timestamp, activity_type
            FROM adl_history
            WHERE elder_id = ? AND room = ?
            ORDER BY timestamp ASC
            """,
            (elder, room),
        ).fetchall()
        values = [(str(r["timestamp"]), str(r["activity_type"])) for r in rows]

    assert values[0][1] == "low_confidence"
    assert values[1][1] == "cooking"
    assert values[2][1] == "low_confidence"


def test_audit_service_fetch_trail_filter(test_db):
    """Verify audit trail filters properly."""
    
    # 1. Seed Correction History
    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", ("E1", "Test1"))
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", ("E2", "Test2"))
        
        # Base query to insert
        q = """
        INSERT INTO correction_history 
        (elder_id, room, new_activity, corrected_by, is_deleted, corrected_at, timestamp_start, timestamp_end, rows_affected) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        now = datetime.now()
        ts_start = datetime(2026, 2, 28, 10, 0, 0)
        ts_end = datetime(2026, 2, 28, 10, 30, 0)
        
        conn.execute(q, ("E1", "bathroom", "shower", "op1", 0, now, ts_start, ts_end, 1))
        conn.execute(q, ("E1", "bedroom", "sleep", "op2", 0, now, ts_start, ts_end, 1))
        conn.execute(q, ("E2", "bathroom", "toilet", "op1", 0, now, ts_start, ts_end, 1))
        conn.commit()
        
    # Filters
    assert len(fetch_correction_trail("All", "All", "All")) == 3
    assert len(fetch_correction_trail("E1", "All", "All")) == 2
    assert len(fetch_correction_trail("All", "bathroom", "All")) == 2
    assert len(fetch_correction_trail("All", "All", "op2")) == 1


def test_ops_service_daily_summary(test_db):
    """Test ops service parses schema outputs."""
    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", ("OP1", "OpTest"))
        ts = (datetime.now() - pd.Timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
        conn.execute("INSERT INTO adl_history (elder_id, timestamp, room, activity_type, record_date, duration_minutes) VALUES (?, ?, ?, ?, ?, ?)", 
                    ("OP1", ts, "bed", "sleep", "2026-02-28", 60))
        conn.execute("INSERT INTO alerts (elder_id, is_read, alert_type) VALUES (?, ?, ?)", ("OP1", 0, "test"))
        conn.commit()

    summary = get_daily_ops_summary("OP1")
    assert summary["open_alerts"] == 1
    assert round(summary["hours_since_last_ingestion"]) == 2
    assert not summary["is_stale"] # 2 < 6 limit


def test_ops_service_sample_collection_uses_adl_day_coverage(test_db):
    elder = "OPS_SAMPLE_1"
    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Ops Sample"))
        rows = [
            (elder, "2025-12-04 00:00:10", "Bedroom", "sleep", "2025-12-04", 10),
            (elder, "2025-12-05 00:00:10", "Bedroom", "sleep", "2025-12-05", 10),
            (elder, "2025-12-04 00:00:10", "LivingRoom", "watch_tv", "2025-12-04", 10),
        ]
        for row in rows:
            conn.execute(
                """
                INSERT INTO adl_history
                (elder_id, timestamp, room, activity_type, record_date, duration_minutes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                row,
            )
        conn.commit()

    samples = ops_service.get_sample_collection_status(elder)
    counts = samples.get("counts", {})
    assert counts.get("bedroom") == 2
    assert counts.get("living_room") == 1
    assert samples.get("target") == 14


def test_ops_service_sample_collection_target_respects_env_override(test_db, monkeypatch):
    elder = "OPS_SAMPLE_2"
    monkeypatch.setenv("UI_SAMPLE_COLLECTION_TARGET_DAYS", "9")
    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Ops Sample Target"))
        conn.execute(
            """
            INSERT INTO adl_history
            (elder_id, timestamp, room, activity_type, record_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (elder, "2025-12-04 00:00:10", "Bedroom", "sleep", "2025-12-04", 10),
        )
        conn.commit()

    samples = ops_service.get_sample_collection_status(elder)
    assert samples.get("target") == 9


def test_ops_service_hard_negative_last_run_falls_back_to_training(test_db):
    elder = "OPS_HN_1"
    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Ops HN"))
        conn.execute(
            """
            INSERT INTO training_history (elder_id, training_date, model_type, status, accuracy, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-02-28 09:30:00", "baseline", "ok", 0.8, "{}"),
        )
        conn.commit()

    summary = ops_service.get_hard_negative_summary(elder, days=30)
    assert summary["open_count"] == 0
    assert summary["recent_resolved"] == 0
    assert summary["last_run"] is not None


def test_ops_service_model_status_fallback_rooms(monkeypatch):
    """All-rooms-only snapshots should expand to room-specific rows when artifacts exist."""
    monkeypatch.setattr(
        ops_service,
        "build_ml_snapshot_report",
        lambda **kwargs: (
            {
                "status": {"overall": "watch"},
                "rooms": [{"room": "All_Rooms", "status": "watch", "metrics": {}}],
            },
            200,
        ),
    )
    monkeypatch.setattr(
        ops_service,
        "_fallback_model_rooms_from_artifacts",
        lambda elder_id: ["bedroom", "livingroom"],
    )

    status = get_model_status("HK0011_jessica")
    rooms = status.get("rooms", [])
    names = [str(r.get("room")) for r in rooms]
    assert names == ["bedroom", "livingroom"]
    assert all(r.get("status") == "watch" for r in rooms)


def test_ops_service_model_status_keeps_real_rooms(monkeypatch):
    """Room-specific snapshots should pass through unchanged."""
    monkeypatch.setattr(
        ops_service,
        "build_ml_snapshot_report",
        lambda **kwargs: (
            {
                "status": {"overall": "healthy"},
                "rooms": [
                    {"room": "bedroom", "status": "healthy", "metrics": {"candidate_macro_f1_mean": 0.9}},
                    {"room": "livingroom", "status": "watch", "metrics": {"candidate_macro_f1_mean": 0.8}},
                ],
            },
            200,
        ),
    )
    monkeypatch.setattr(
        ops_service,
        "_fallback_model_rooms_from_artifacts",
        lambda elder_id: ["bathroom"],
    )

    status = get_model_status("HK0011_jessica")
    rooms = status.get("rooms", [])
    names = [str(r.get("room")) for r in rooms]
    assert names == ["bedroom", "livingroom"]


def test_correction_service_get_activity_labels(test_db):
    """Test that get_activity_labels returns default labels when DB has nothing."""
    from services.correction_service import get_activity_labels, DEFAULT_ACTIVITY_LABELS
    
    labels = get_activity_labels()
    assert isinstance(labels, list)
    assert len(labels) == len(DEFAULT_ACTIVITY_LABELS)
    assert "sleep" in labels
    assert "watch_tv" in labels


def test_correction_service_get_activity_timeline(test_db):
    """Test activity timeline returns full-day data with expected columns."""
    from services.correction_service import get_activity_timeline
    
    elder = "TL001"
    room = "bedroom"
    
    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Timeline"))
        conn.execute(
            "INSERT INTO adl_history (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (elder, "2026-03-01 08:00:00", room, "sleep", 0.95, 0, "2026-03-01", 30)
        )
        conn.execute(
            "INSERT INTO adl_history (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (elder, "2026-03-01 12:00:00", room, "nap", 0.40, 0, "2026-03-01", 15)
        )
        conn.commit()
    
    df = get_activity_timeline(elder, room, date(2026, 3, 1))
    assert len(df) == 2
    assert "end_time" in df.columns
    assert "time" in df.columns
    assert df.iloc[0]["activity_type"] == "sleep"


def test_correction_service_get_training_timeline_marks_corrected_source(test_db):
    elder = "TRAIN_TL001"
    room = "livingroom"

    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Training Timeline"))
        conn.execute(
            """
            INSERT INTO adl_history
            (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-03-05 08:00:00", room, "watch_tv", 1.0, 1, "2026-03-05", 30),
        )
        conn.execute(
            """
            INSERT INTO adl_history
            (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-03-05 08:30:00", room, "nap", 0.95, 0, "2026-03-05", 15),
        )
        conn.commit()

    df = get_training_timeline(elder, room, date(2026, 3, 5))
    assert len(df) == 2
    assert "source_label" in df.columns
    assert "end_time" in df.columns
    assert df.iloc[0]["source_label"] == "corrected"
    assert df.iloc[1]["source_label"] == "recorded"


def test_correction_service_build_compare_timeline_payload_marks_review_states(test_db):
    elder = "COMPARE001"
    room = "livingroom"
    record_day = date(2026, 3, 6)

    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Compare Timeline"))
        conn.execute(
            """
            INSERT INTO adl_history
            (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-03-06 10:00:00", room, "watch_tv", 1.0, 1, "2026-03-06", 10),
        )
        conn.execute(
            """
            INSERT INTO adl_history
            (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-03-06 10:10:00", room, "watch_tv", 1.0, 1, "2026-03-06", 10),
        )
        conn.execute(
            """
            INSERT INTO predictions
            (resident_id, timestamp, room, activity, confidence, is_anomaly)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-03-06 10:00:00", room, "low_confidence", 0.35, 0),
        )
        conn.execute(
            """
            INSERT INTO predictions
            (resident_id, timestamp, room, activity, confidence, is_anomaly)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-03-06 10:10:00", room, "nap", 0.92, 0),
        )
        conn.commit()

    payload = build_compare_timeline_payload(elder, room, record_day, confidence_threshold=0.60)
    training_df = payload.get("training_timeline")
    prediction_df = payload.get("prediction_timeline")

    assert training_df is not None and len(training_df) == 2
    assert prediction_df is not None and len(prediction_df) == 2

    first = prediction_df.iloc[0]
    second = prediction_df.iloc[1]
    assert bool(first["is_unknown"])
    assert bool(first["is_low_confidence"])
    assert bool(first["is_mismatch"])
    assert bool(first["is_corrected"])
    assert first["training_activity_type"] == "watch_tv"
    assert first["prediction_state_label"] == "unknown / review needed"

    assert not bool(second["is_unknown"])
    assert not bool(second["is_low_confidence"])
    assert bool(second["is_mismatch"])
    assert second["training_activity_type"] == "watch_tv"


def test_review_ui_exposes_manual_review_rate_in_compare_payload(test_db):
    elder = "COMPARE002"
    room = "bedroom"
    record_day = date(2026, 3, 7)

    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Compare Timeline 2"))
        conn.execute(
            """
            INSERT INTO adl_history
            (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-03-07 09:00:00", room, "sleep", 1.0, 1, "2026-03-07", 10),
        )
        conn.execute(
            """
            INSERT INTO predictions
            (resident_id, timestamp, room, activity, confidence, is_anomaly)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-03-07 09:00:00", room, "unknown", 0.22, 0),
        )
        conn.execute(
            """
            INSERT INTO predictions
            (resident_id, timestamp, room, activity, confidence, is_anomaly)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-03-07 09:10:00", room, "sleep", 0.93, 0),
        )
        conn.commit()

    payload = build_compare_timeline_payload(elder, room, record_day, confidence_threshold=0.60)
    summary = payload.get("summary", {})
    assert summary.get("prediction_blocks") == 2
    assert summary.get("review_needed_blocks") == 1
    assert summary.get("manual_review_rate") == pytest.approx(0.5)


def test_ops_scorecard_includes_correction_volume_and_backlog(test_db, monkeypatch):
    elder = "OPS_SCORECARD_1"
    room = "livingroom"
    monkeypatch.setattr(
        ops_service,
        "get_model_status",
        lambda _elder: {
            "status": {"overall": "watch"},
            "rooms": [
                {"room": "bedroom", "status": "watch"},
                {"room": "kitchen", "status": "healthy"},
            ],
        },
    )

    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Ops Scorecard"))
        conn.execute(
            """
            INSERT INTO correction_history
            (elder_id, room, timestamp_start, timestamp_end, old_activity, new_activity, rows_affected, corrected_by, corrected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                elder,
                room,
                "2026-03-07 09:00:00",
                "2026-03-07 09:10:00",
                "unknown",
                "watch_tv",
                1,
                "tester",
                "2026-03-07 12:00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO adl_history
            (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-03-07 09:00:00", room, "watch_tv", 1.0, 1, "2026-03-07", 10),
        )
        conn.execute(
            """
            INSERT INTO adl_history
            (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-03-07 10:00:00", room, "low_confidence", 0.21, 0, "2026-03-07", 10),
        )
        conn.execute(
            """
            INSERT INTO predictions
            (resident_id, timestamp, room, activity, confidence, is_anomaly)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-03-07 09:00:00", room, "nap", 0.31, 0),
        )
        conn.execute(
            """
            INSERT INTO predictions
            (resident_id, timestamp, room, activity, confidence, is_anomaly)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-03-07 09:10:00", room, "unknown", 0.20, 0),
        )
        conn.execute(
            """
            INSERT INTO predictions
            (resident_id, timestamp, room, activity, confidence, is_anomaly)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (elder, "2026-03-07 09:20:00", room, "watch_tv", 0.95, 0),
        )
        conn.commit()

    scorecard = ops_service.get_timeline_reliability_scorecard(elder_id=elder, days=30, confidence_threshold=0.60)
    assert scorecard.get("correction_volume") == 1
    assert scorecard.get("review_backlog") >= 1
    assert scorecard.get("manual_review_rate") is not None
    assert scorecard.get("unknown_abstain_rate") is not None
    assert scorecard.get("contradiction_rate") is not None
    assert scorecard.get("fragmentation_rate") is not None
    assert scorecard.get("authority_state") == "watch"
    assert "bedroom" in scorecard.get("policy_sensitive_rooms", [])


def test_ops_service_model_update_monitor_exposes_metric_source_labels(test_db):
    elder = "OPS_MONITOR_1"
    metadata = {
        "walk_forward_gate": {
            "reason": "evaluated",
            "pass": True,
            "room_reports": [
                {
                    "room": "livingroom",
                    "pass": True,
                    "candidate_summary": {
                        "macro_f1_mean": 0.72,
                        "accuracy_mean": 0.81,
                    },
                    "candidate_stability_accuracy_mean": 0.77,
                    "candidate_transition_macro_f1_mean": 0.61,
                    "champion_macro_f1_mean": 0.68,
                    "candidate_threshold_required": 0.50,
                    "training_days": 14.0,
                    "reasons": [],
                }
            ],
        },
        "metrics": [],
    }
    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Ops Monitor"))
        conn.execute(
            """
            INSERT INTO training_history (elder_id, training_date, model_type, status, accuracy, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                elder,
                "2026-03-06 09:00:00",
                "baseline",
                "SUCCESS",
                0.91,
                json.dumps(metadata),
            ),
        )
        conn.commit()

    monitor = ops_service.get_model_update_monitor(elder, days=30, limit=10)
    labels = monitor.get("metric_labels", {})
    assert labels.get("latest_candidate_macro_f1_mean") == "WF Candidate F1"
    assert labels.get("latest_candidate_accuracy_mean") == "WF Candidate Accuracy"
    assert labels.get("latest_run_accuracy") == "Raw Training Run Accuracy"


def test_correction_service_activity_timeline_room_alias_match(test_db):
    """Timeline should support room aliases (living_room vs livingroom)."""
    elder = "ALIAS_TL001"
    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Alias Timeline"))
        conn.execute(
            """
            INSERT INTO adl_history
            (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, "2025-12-04 09:00:00", "livingroom", "watch_tv", 0.88, 0, "2025-12-04", 10),
        )
        conn.commit()

    df = get_activity_timeline(elder, "living_room", date(2025, 12, 4))
    assert len(df) == 1
    assert df.iloc[0]["activity_type"] == "watch_tv"


def test_correction_service_activity_timeline_segments_fallback(test_db):
    """Timeline should fall back to activity_segments when adl_history has no rows."""
    elder = "SEGMENT_TL001"
    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Segment Timeline"))
        conn.execute(
            """
            INSERT INTO activity_segments
            (elder_id, room, activity_type, start_time, end_time, duration_minutes, avg_confidence, event_count, record_date, is_corrected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                elder,
                "livingroom",
                "watch_tv",
                "2025-12-04 18:00:00",
                "2025-12-04 18:20:00",
                20.0,
                0.86,
                12,
                "2025-12-04",
                0,
            ),
        )
        conn.commit()

    df = get_activity_timeline(elder, "living_room", date(2025, 12, 4))
    assert len(df) == 1
    assert df.iloc[0]["activity_type"] == "watch_tv"
    assert pd.to_datetime(df.iloc[0]["end_time"]) > pd.to_datetime(df.iloc[0]["timestamp"])


def test_correction_service_low_confidence_room_alias_match(test_db):
    """Low-confidence queue should support room aliases."""
    elder = "ALIAS_LQ001"
    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Alias Queue"))
        conn.execute(
            """
            INSERT INTO adl_history
            (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, "2025-12-04 10:10:00", "livingroom", "low_confidence", 0.22, 0, "2025-12-04", 10),
        )
        conn.commit()

    df = get_low_confidence_queue(elder, "living_room", date(2025, 12, 4), 0.50)
    assert len(df) == 1
    assert df.iloc[0]["activity_type"] == "low_confidence"


def test_correction_service_activity_timeline_any_room_fallback(test_db):
    """All-room timeline fallback should return rows when room-specific view is empty."""
    elder = "ANYROOM001"
    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Any Room"))
        conn.execute(
            """
            INSERT INTO adl_history
            (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, "2025-12-04 06:30:00", "kitchen", "cooking", 0.91, 0, "2025-12-04", 10),
        )
        conn.commit()

    df = get_activity_timeline_any_room(elder, date(2025, 12, 4))
    assert len(df) == 1
    assert str(df.iloc[0]["room"]) == "kitchen"


def test_correction_service_find_nearest_activity_date(test_db):
    """Nearest-date lookup should find closest day with activity rows."""
    elder = "NEAREST001"
    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Nearest Date"))
        conn.execute(
            """
            INSERT INTO adl_history
            (elder_id, timestamp, room, activity_type, confidence, is_corrected, record_date, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, "2025-12-07 08:00:00", "bedroom", "sleep", 0.95, 0, "2025-12-07", 10),
        )
        conn.commit()

    nearest = find_nearest_activity_date(elder, date(2025, 12, 4), room="bedroom")
    assert nearest is not None
    assert nearest["date"] == date(2025, 12, 7)
    assert nearest["rows"] >= 1


def test_audit_service_rollback_nonexistent(test_db):
    """Test rollback returns failure for nonexistent correction ID."""
    success, msg = rollback_correction(99999)
    assert not success
    assert "not found" in msg.lower()


def test_correction_service_get_sensor_timeseries(test_db):
    """Ensure sensor timeseries is parsed from sensor_features payload."""
    elder = "SENSOR001"
    room = "kitchen"

    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Sensor Test"))
        conn.execute(
            """
            INSERT INTO adl_history
            (elder_id, record_date, timestamp, room, activity_type, confidence, sensor_features)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                elder,
                "2026-03-02",
                "2026-03-02 09:15:00",
                room,
                "cooking",
                0.82,
                json.dumps({
                    "co2": 640,
                    "humidity": 55,
                    "motion": 1,
                    "temperature": 24.1,
                    "sound": 36.5,
                    "light": 210,
                }),
            ),
        )
        conn.commit()

    df = get_sensor_timeseries(elder, room, date(2026, 3, 2))
    assert len(df) == 1
    assert df.iloc[0]["activity_type"] == "cooking"
    assert df.iloc[0]["co2"] == 640
    assert df.iloc[0]["motion"] == 1


def test_correction_service_apply_batch_corrections(test_db):
    """Batch apply should update rows and write correction history."""
    elder = "BATCH001"
    room = "livingroom"
    day = date(2026, 3, 3)

    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Batch Test"))
        conn.execute(
            """
            INSERT INTO adl_history
            (elder_id, record_date, timestamp, room, activity_type, confidence, is_corrected, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, day.isoformat(), "2026-03-03 10:00:00", room, "low_confidence", 0.44, 0, 10),
        )
        conn.execute(
            """
            INSERT INTO adl_history
            (elder_id, record_date, timestamp, room, activity_type, confidence, is_corrected, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, day.isoformat(), "2026-03-03 10:10:00", room, "low_confidence", 0.41, 0, 10),
        )
        conn.commit()

    summary = apply_batch_corrections(
        elder_id=elder,
        room=room,
        record_date=day,
        corrections=[
            {
                "start_time": datetime.strptime("10:00:00", "%H:%M:%S").time(),
                "end_time": datetime.strptime("10:10:00", "%H:%M:%S").time(),
                "new_label": "watch_tv",
            }
        ],
        corrected_by="tester",
    )
    assert summary["ok"]
    assert summary["ranges_applied"] == 1
    assert summary["rows_updated"] >= 2

    with test_db.get_connection() as conn:
        updated = conn.execute(
            "SELECT COUNT(*) AS cnt FROM adl_history WHERE elder_id = ? AND room = ? AND activity_type = ?",
            (elder, room, "watch_tv"),
        ).fetchone()
        assert int(updated["cnt"]) == 2
        audit = conn.execute(
            "SELECT COUNT(*) AS cnt FROM correction_history WHERE elder_id = ? AND room = ?",
            (elder, room),
        ).fetchone()
        assert int(audit["cnt"]) == 1


def test_correction_service_preview_batch_detects_overlap(test_db):
    """Dry-run preview should flag overlapping queued ranges."""
    elder = "PREVIEW001"
    room = "bedroom"
    day = date(2026, 3, 4)

    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Preview Test"))
        conn.execute(
            """
            INSERT INTO adl_history
            (elder_id, record_date, timestamp, room, activity_type, confidence, is_corrected, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (elder, day.isoformat(), "2026-03-04 08:00:00", room, "sleep", 0.92, 0, 10),
        )
        conn.commit()

    preview = preview_batch_corrections(
        elder_id=elder,
        room=room,
        record_date=day,
        corrections=[
            {
                "start_time": datetime.strptime("08:00:00", "%H:%M:%S").time(),
                "end_time": datetime.strptime("08:20:00", "%H:%M:%S").time(),
                "new_label": "sleep",
            },
            {
                "start_time": datetime.strptime("08:10:00", "%H:%M:%S").time(),
                "end_time": datetime.strptime("08:30:00", "%H:%M:%S").time(),
                "new_label": "nap",
            },
        ],
    )
    assert not preview["ok"]
    assert preview["has_invalid"]
    assert preview["has_overlap"]
