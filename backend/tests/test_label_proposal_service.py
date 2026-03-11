from datetime import datetime

import pandas as pd

from services.label_proposal_service import (
    apply_approved_label_review_batch,
    build_apply_ready_payload,
    build_proposed_timestamp_set,
    create_label_review_batch,
    get_label_review_audit_log,
    get_label_review_items,
    list_label_review_batches,
    load_label_review_snapshot,
    update_label_review_statuses,
)


def _seed_livingroom_rows(test_db, elder: str, room: str, record_date: str) -> None:
    rows = [
        (elder, record_date, f"{record_date} 10:00:00", room, "unoccupied", 0.95, 0, 10),
        (elder, record_date, f"{record_date} 10:00:10", room, "unoccupied", 0.95, 0, 10),
        (elder, record_date, f"{record_date} 10:00:20", room, "unoccupied", 0.95, 0, 10),
    ]
    with test_db.get_connection() as conn:
        conn.execute("INSERT INTO elders (elder_id, full_name) VALUES (?, ?)", (elder, "Proposal Review"))
        for row in rows:
            conn.execute(
                """
                INSERT INTO adl_history
                (elder_id, record_date, timestamp, room, activity_type, confidence, is_corrected, duration_minutes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
        conn.commit()


def _sample_proposals(elder: str, room: str, record_date: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "external_id": "seg-1",
                "elder_id": elder,
                "room": room,
                "record_date": record_date,
                "granularity": "segment",
                "timestamp_start": f"{record_date} 10:00:00",
                "timestamp_end": f"{record_date} 10:00:20",
                "current_label": "unoccupied",
                "proposed_label": "livingroom_normal_use",
                "confidence_tier": "high",
                "proposal_score": 0.93,
                "reason_codes": ["anchor_conflict", "run_consensus"],
                "rationale": {"kind": "segment_consensus"},
            },
            {
                "external_id": "ts-1",
                "parent_external_id": "seg-1",
                "elder_id": elder,
                "room": room,
                "record_date": record_date,
                "granularity": "timestamp",
                "timestamp_start": f"{record_date} 10:00:20",
                "timestamp_end": f"{record_date} 10:00:20",
                "current_label": "unoccupied",
                "proposed_label": "watch_tv",
                "confidence_tier": "medium",
                "proposal_score": 0.78,
                "reason_codes": ["edge_override"],
                "rationale": {"kind": "timestamp_override"},
            },
            {
                "external_id": "ts-2",
                "parent_external_id": "seg-1",
                "elder_id": elder,
                "room": room,
                "record_date": record_date,
                "granularity": "timestamp",
                "timestamp_start": f"{record_date} 10:00:10",
                "timestamp_end": f"{record_date} 10:00:10",
                "current_label": "unoccupied",
                "proposed_label": "livingroom_normal_use",
                "confidence_tier": "low",
                "proposal_score": 0.51,
                "reason_codes": ["edge_review"],
                "rationale": {"kind": "edge_check"},
            },
        ]
    )


def test_label_proposal_service_creates_batch_and_resolves_timestamp_set(test_db):
    elder = "LABEL_REVIEW_1"
    room = "livingroom"
    record_date = "2026-03-08"
    _seed_livingroom_rows(test_db, elder, room, record_date)

    batch_id = create_label_review_batch(
        batch_name="LivingRoom suspicious block",
        proposals_df=_sample_proposals(elder, room, record_date),
        created_by="forensic",
        source_kind="forensic",
        source_ref="tmp/forensic.json",
        notes="precision-first candidates",
    )

    batches = list_label_review_batches(elder_id=elder, room=room, record_date=record_date)
    assert len(batches) == 1
    assert int(batches.iloc[0]["proposal_count"]) == 3
    assert batches.iloc[0]["batch_status"] == "proposed"

    items = get_label_review_items(batch_id)
    assert len(items) == 3

    segment_id = int(items.loc[items["external_id"] == "seg-1", "id"].iloc[0])
    ts_parent_ids = items.loc[items["granularity"] == "timestamp", "parent_id"].dropna().astype(int).tolist()
    assert ts_parent_ids == [segment_id, segment_id]

    proposed = build_proposed_timestamp_set(batch_id)
    assert list(proposed["timestamp"].astype(str)) == [
        f"{record_date} 10:00:00",
        f"{record_date} 10:00:10",
        f"{record_date} 10:00:20",
    ]
    assert list(proposed["proposed_label"]) == [
        "livingroom_normal_use",
        "livingroom_normal_use",
        "watch_tv",
    ]
    assert list(proposed["source_granularity"]) == ["segment", "timestamp", "timestamp"]


def test_label_proposal_service_logs_review_decisions(test_db):
    elder = "LABEL_REVIEW_2"
    room = "livingroom"
    record_date = "2026-03-09"
    _seed_livingroom_rows(test_db, elder, room, record_date)

    batch_id = create_label_review_batch(
        batch_name="LivingRoom review actions",
        proposals_df=_sample_proposals(elder, room, record_date),
        created_by="forensic",
    )

    items = get_label_review_items(batch_id)
    segment_id = int(items.loc[items["external_id"] == "seg-1", "id"].iloc[0])
    ts1_id = int(items.loc[items["external_id"] == "ts-1", "id"].iloc[0])
    ts2_id = int(items.loc[items["external_id"] == "ts-2", "id"].iloc[0])

    update_label_review_statuses(batch_id, [segment_id], "approved", actor="ops_a", note="coherent run")
    update_label_review_statuses(batch_id, [ts1_id], "approved", actor="ops_a", note="TV edge confirmed")
    update_label_review_statuses(batch_id, [ts2_id], "rejected", actor="ops_a", note="leave midpoint alone")

    updated = get_label_review_items(batch_id).set_index("external_id")
    assert updated.loc["seg-1", "review_status"] == "approved"
    assert updated.loc["ts-1", "review_status"] == "approved"
    assert updated.loc["ts-2", "review_status"] == "rejected"
    assert updated.loc["ts-2", "review_note"] == "leave midpoint alone"

    audit = get_label_review_audit_log(batch_id)
    actions = audit["action"].tolist()
    assert "batch_created" in actions
    assert actions.count("status_changed") == 3


def test_label_proposal_service_apply_ready_payload_honors_timestamp_overrides(test_db):
    elder = "LABEL_REVIEW_3"
    room = "livingroom"
    record_date = "2026-03-10"
    _seed_livingroom_rows(test_db, elder, room, record_date)

    batch_id = create_label_review_batch(
        batch_name="LivingRoom apply-ready",
        proposals_df=_sample_proposals(elder, room, record_date),
        created_by="forensic",
    )

    items = get_label_review_items(batch_id)
    segment_id = int(items.loc[items["external_id"] == "seg-1", "id"].iloc[0])
    ts1_id = int(items.loc[items["external_id"] == "ts-1", "id"].iloc[0])
    ts2_id = int(items.loc[items["external_id"] == "ts-2", "id"].iloc[0])

    update_label_review_statuses(batch_id, [segment_id], "approved", actor="ops_b")
    update_label_review_statuses(batch_id, [ts1_id], "approved", actor="ops_b")
    update_label_review_statuses(batch_id, [ts2_id], "rejected", actor="ops_b")

    payload = build_apply_ready_payload(batch_id)
    rows = payload["rows"]

    assert [row["timestamp"] for row in rows] == [
        f"{record_date} 10:00:00",
        f"{record_date} 10:00:20",
    ]
    assert [row["proposed_label"] for row in rows] == [
        "livingroom_normal_use",
        "watch_tv",
    ]


def test_label_proposal_service_applies_approved_rows_and_marks_items_applied(test_db):
    elder = "LABEL_REVIEW_4"
    room = "livingroom"
    record_date = "2026-03-11"
    _seed_livingroom_rows(test_db, elder, room, record_date)

    batch_id = create_label_review_batch(
        batch_name="LivingRoom apply batch",
        proposals_df=_sample_proposals(elder, room, record_date),
        created_by="forensic",
    )

    items = get_label_review_items(batch_id)
    segment_id = int(items.loc[items["external_id"] == "seg-1", "id"].iloc[0])
    ts1_id = int(items.loc[items["external_id"] == "ts-1", "id"].iloc[0])
    ts2_id = int(items.loc[items["external_id"] == "ts-2", "id"].iloc[0])

    update_label_review_statuses(batch_id, [segment_id], "approved", actor="ops_c")
    update_label_review_statuses(batch_id, [ts1_id], "approved", actor="ops_c")
    update_label_review_statuses(batch_id, [ts2_id], "rejected", actor="ops_c")

    summary = apply_approved_label_review_batch(batch_id, applied_by="ops_c")
    assert summary["ok"]
    assert summary["applied_rows"] == 2
    assert summary["applied_ranges"] == 2

    with test_db.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT timestamp, activity_type, is_corrected
            FROM adl_history
            WHERE elder_id = ? AND room = ?
            ORDER BY timestamp ASC
            """,
            (elder, room),
        ).fetchall()
        values = [(str(r["timestamp"]), str(r["activity_type"]), int(r["is_corrected"])) for r in rows]
        history_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM correction_history WHERE elder_id = ? AND room = ?",
            (elder, room),
        ).fetchone()["cnt"]

    assert values == [
        (f"{record_date} 10:00:00", "livingroom_normal_use", 1),
        (f"{record_date} 10:00:10", "unoccupied", 0),
        (f"{record_date} 10:00:20", "watch_tv", 1),
    ]
    assert int(history_count) == 2

    final_items = get_label_review_items(batch_id).set_index("external_id")
    assert final_items.loc["seg-1", "review_status"] == "applied"
    assert final_items.loc["ts-1", "review_status"] == "applied"
    assert final_items.loc["ts-2", "review_status"] == "rejected"


def test_label_proposal_service_snapshot_degrades_gracefully_on_db_errors(monkeypatch):
    import services.label_proposal_service as svc

    def _boom(**kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(svc, "list_label_review_batches", _boom)

    snapshot = load_label_review_snapshot(elder_id="HK0011_jessica", room="livingroom", record_date="2025-12-17")
    assert not snapshot["ok"]
    assert "db unavailable" in snapshot["error"]
    assert snapshot["batches_df"].empty
    assert snapshot["items_df"].empty
    assert snapshot["proposed_timestamp_df"].empty
    assert snapshot["audit_df"].empty
