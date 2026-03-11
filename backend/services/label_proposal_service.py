import json
import logging
from datetime import date as dt_date

import pandas as pd

from services.correction_service import apply_batch_corrections
from services.db_utils import get_dashboard_connection, parse_json_object

logger = logging.getLogger(__name__)

ROOM_MATCH_SQL = "LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = LOWER(REPLACE(REPLACE(?, ' ', ''), '_', ''))"
VALID_REVIEW_STATUSES = {"proposed", "approved", "rejected", "applied"}
VALID_GRANULARITIES = {"segment", "timestamp"}
ACTIVE_REVIEW_STATUSES = {"proposed", "approved"}
APPLY_READY_STATUSES = {"approved"}
BLOCKING_STATUSES = {"rejected", "applied"}


def _fetch_dataframe(conn, query: str, params: tuple | list = ()) -> pd.DataFrame:
    cursor = conn.execute(query, params)
    rows = cursor.fetchall() or []
    columns = [col[0] for col in (cursor.description or [])]
    return pd.DataFrame(rows, columns=columns)


def _ensure_label_review_tables(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS label_review_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_name TEXT NOT NULL,
            elder_id TEXT,
            room TEXT,
            record_date DATE,
            source_kind TEXT DEFAULT 'import',
            source_ref TEXT,
            notes TEXT,
            created_by TEXT DEFAULT 'system',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            batch_status TEXT DEFAULT 'proposed',
            proposal_count INTEGER DEFAULT 0,
            approved_count INTEGER DEFAULT 0,
            rejected_count INTEGER DEFAULT 0,
            applied_count INTEGER DEFAULT 0,
            last_action_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS label_review_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL,
            parent_id INTEGER,
            external_id TEXT,
            parent_external_id TEXT,
            elder_id TEXT NOT NULL,
            room TEXT NOT NULL,
            record_date DATE NOT NULL,
            granularity TEXT NOT NULL,
            timestamp_start TIMESTAMP NOT NULL,
            timestamp_end TIMESTAMP NOT NULL,
            current_label TEXT,
            proposed_label TEXT NOT NULL,
            confidence_tier TEXT,
            proposal_score REAL,
            reason_codes_json TEXT,
            rationale_json TEXT,
            source_model TEXT,
            review_status TEXT DEFAULT 'proposed',
            review_note TEXT,
            reviewed_by TEXT,
            reviewed_at TIMESTAMP,
            applied_at TIMESTAMP,
            applied_by TEXT,
            display_order INTEGER DEFAULT 0,
            created_by TEXT DEFAULT 'system',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS label_review_decision_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL,
            proposal_id INTEGER,
            action TEXT NOT NULL,
            from_status TEXT,
            to_status TEXT,
            actor TEXT DEFAULT 'system',
            note TEXT,
            payload_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_label_review_batches_scope ON label_review_batches(elder_id, room, record_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_label_review_items_batch ON label_review_items(batch_id, display_order)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_label_review_items_parent ON label_review_items(parent_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_label_review_items_scope ON label_review_items(elder_id, room, record_date, review_status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_label_review_log_batch ON label_review_decision_log(batch_id, created_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_label_review_log_item ON label_review_decision_log(proposal_id, created_at DESC)")


def _coerce_record_date(value) -> str | None:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _json_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
            return json.dumps(parsed, sort_keys=True)
        except Exception:
            return json.dumps([part.strip() for part in stripped.split(",") if part.strip()])
    if isinstance(value, (list, tuple, dict, bool, int, float)):
        return json.dumps(value, sort_keys=True)
    return json.dumps(str(value))


def _display_reason_codes(raw_value) -> str:
    payload = parse_json_object(raw_value)
    if isinstance(payload, dict):
        return ", ".join(f"{k}={v}" for k, v in payload.items())
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except Exception:
            return raw_value
        if isinstance(parsed, list):
            return ", ".join(str(v) for v in parsed)
        if isinstance(parsed, dict):
            return ", ".join(f"{k}={v}" for k, v in parsed.items())
        return str(parsed)
    return ""


def _normalize_proposals_df(
    proposals_df: pd.DataFrame,
    *,
    default_elder_id: str | None = None,
    default_room: str | None = None,
    default_record_date: str | dt_date | None = None,
) -> pd.DataFrame:
    if proposals_df is None or proposals_df.empty:
        raise ValueError("Proposal batch is empty.")

    out = proposals_df.copy()
    rename_map = {
        "timestamp": "timestamp_start",
        "start_time": "timestamp_start",
        "end_time": "timestamp_end",
        "activity": "current_label",
        "activity_type": "current_label",
        "suggested_label": "proposed_label",
        "new_activity": "proposed_label",
        "status": "review_status",
        "score": "proposal_score",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    if "timestamp_start" not in out.columns:
        raise ValueError("Proposal batch must include a timestamp_start column.")
    if "proposed_label" not in out.columns:
        raise ValueError("Proposal batch must include a proposed_label column.")

    if "external_id" not in out.columns:
        out["external_id"] = [f"proposal-{i + 1}" for i in range(len(out))]
    out["external_id"] = out["external_id"].fillna("").astype(str).str.strip()
    out.loc[out["external_id"] == "", "external_id"] = [f"proposal-{i + 1}" for i in range(len(out))]

    if "parent_external_id" not in out.columns:
        out["parent_external_id"] = None
    out["parent_external_id"] = out["parent_external_id"].apply(
        lambda value: None
        if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""
        else str(value).strip()
    )

    out["timestamp_start"] = pd.to_datetime(out["timestamp_start"], errors="coerce")
    if "timestamp_end" not in out.columns:
        out["timestamp_end"] = out["timestamp_start"]
    out["timestamp_end"] = pd.to_datetime(out["timestamp_end"], errors="coerce")
    out["timestamp_end"] = out["timestamp_end"].fillna(out["timestamp_start"])

    if out["timestamp_start"].isna().any() or out["timestamp_end"].isna().any():
        raise ValueError("Proposal batch contains invalid timestamps.")
    if (out["timestamp_end"] < out["timestamp_start"]).any():
        raise ValueError("Proposal batch contains timestamp_end earlier than timestamp_start.")

    if "granularity" not in out.columns:
        out["granularity"] = out.apply(
            lambda row: "timestamp" if row["timestamp_start"] == row["timestamp_end"] else "segment",
            axis=1,
        )
    out["granularity"] = out["granularity"].fillna("timestamp").astype(str).str.strip().str.lower()
    invalid_granularity = sorted(set(out["granularity"]) - VALID_GRANULARITIES)
    if invalid_granularity:
        raise ValueError(f"Unsupported proposal granularity values: {invalid_granularity}")

    if "review_status" not in out.columns:
        out["review_status"] = "proposed"
    out["review_status"] = out["review_status"].fillna("proposed").astype(str).str.strip().str.lower()
    invalid_status = sorted(set(out["review_status"]) - VALID_REVIEW_STATUSES)
    if invalid_status:
        raise ValueError(f"Unsupported proposal review statuses: {invalid_status}")

    default_record_date_str = _coerce_record_date(default_record_date)
    if "record_date" not in out.columns:
        out["record_date"] = default_record_date_str
    out["record_date"] = out["record_date"].apply(_coerce_record_date)
    out["record_date"] = out["record_date"].fillna(out["timestamp_start"].dt.strftime("%Y-%m-%d"))

    if "elder_id" not in out.columns:
        out["elder_id"] = default_elder_id
    if "room" not in out.columns:
        out["room"] = default_room
    out["elder_id"] = out["elder_id"].apply(
        lambda value: default_elder_id
        if value is None or (isinstance(value, float) and pd.isna(value))
        else value
    )
    out["room"] = out["room"].apply(
        lambda value: default_room
        if value is None or (isinstance(value, float) and pd.isna(value))
        else value
    )
    out["elder_id"] = out["elder_id"].apply(
        lambda value: "" if value is None else str(value).strip()
    )
    out["room"] = out["room"].apply(
        lambda value: "" if value is None else str(value).strip()
    )
    if (out["elder_id"] == "").any() or (out["room"] == "").any():
        raise ValueError("Proposal batch must include elder_id and room values.")

    if "current_label" not in out.columns:
        out["current_label"] = None
    out["current_label"] = out["current_label"].where(out["current_label"].notna(), None)

    out["proposed_label"] = out["proposed_label"].fillna("").astype(str).str.strip()
    if (out["proposed_label"] == "").any():
        raise ValueError("Proposal batch contains blank proposed_label values.")

    if "confidence_tier" not in out.columns:
        out["confidence_tier"] = None
    if "proposal_score" not in out.columns:
        out["proposal_score"] = None
    out["proposal_score"] = pd.to_numeric(out["proposal_score"], errors="coerce")

    if "reason_codes" not in out.columns:
        out["reason_codes"] = None
    if "rationale" not in out.columns:
        out["rationale"] = None
    if "source_model" not in out.columns:
        out["source_model"] = None

    out["reason_codes_json"] = out["reason_codes"].apply(_json_text)
    out["rationale_json"] = out["rationale"].apply(_json_text)
    out["display_order"] = list(range(len(out)))
    return out[
        [
            "external_id",
            "parent_external_id",
            "elder_id",
            "room",
            "record_date",
            "granularity",
            "timestamp_start",
            "timestamp_end",
            "current_label",
            "proposed_label",
            "confidence_tier",
            "proposal_score",
            "reason_codes_json",
            "rationale_json",
            "source_model",
            "review_status",
            "display_order",
        ]
    ].copy()


def _insert_log(
    conn,
    *,
    batch_id: int,
    action: str,
    actor: str,
    proposal_id: int | None = None,
    from_status: str | None = None,
    to_status: str | None = None,
    note: str | None = None,
    payload: dict | list | str | None = None,
) -> None:
    payload_json = None
    if payload is not None:
        if isinstance(payload, str):
            payload_json = payload
        else:
            payload_json = json.dumps(payload, sort_keys=True)
    conn.execute(
        """
        INSERT INTO label_review_decision_log
        (batch_id, proposal_id, action, from_status, to_status, actor, note, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (batch_id, proposal_id, action, from_status, to_status, actor, note, payload_json),
    )


def _refresh_batch_summary(conn, batch_id: int) -> None:
    summary_row = conn.execute(
        """
        SELECT
            COUNT(*) AS proposal_count,
            SUM(CASE WHEN review_status = 'approved' THEN 1 ELSE 0 END) AS approved_count,
            SUM(CASE WHEN review_status = 'rejected' THEN 1 ELSE 0 END) AS rejected_count,
            SUM(CASE WHEN review_status = 'applied' THEN 1 ELSE 0 END) AS applied_count
        FROM label_review_items
        WHERE batch_id = ?
        """,
        (batch_id,),
    ).fetchone()
    proposal_count = int(summary_row[0] or 0)
    approved_count = int(summary_row[1] or 0)
    rejected_count = int(summary_row[2] or 0)
    applied_count = int(summary_row[3] or 0)

    if proposal_count > 0 and applied_count == proposal_count:
        batch_status = "applied"
    elif approved_count > 0 or rejected_count > 0:
        batch_status = "in_review"
    else:
        batch_status = "proposed"

    conn.execute(
        """
        UPDATE label_review_batches
        SET proposal_count = ?,
            approved_count = ?,
            rejected_count = ?,
            applied_count = ?,
            batch_status = ?,
            last_action_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (proposal_count, approved_count, rejected_count, applied_count, batch_status, batch_id),
    )


def create_label_review_batch(
    *,
    batch_name: str,
    proposals_df: pd.DataFrame,
    created_by: str = "system",
    source_kind: str = "import",
    source_ref: str | None = None,
    notes: str | None = None,
    default_elder_id: str | None = None,
    default_room: str | None = None,
    default_record_date: str | dt_date | None = None,
) -> int:
    normalized = _normalize_proposals_df(
        proposals_df,
        default_elder_id=default_elder_id,
        default_room=default_room,
        default_record_date=default_record_date,
    )

    elder_values = sorted({str(v).strip() for v in normalized["elder_id"] if str(v).strip()})
    room_values = sorted({str(v).strip() for v in normalized["room"] if str(v).strip()})
    date_values = sorted({str(v).strip() for v in normalized["record_date"] if str(v).strip()})
    batch_elder = elder_values[0] if len(elder_values) == 1 else None
    batch_room = room_values[0] if len(room_values) == 1 else None
    batch_date = date_values[0] if len(date_values) == 1 else None

    with get_dashboard_connection() as conn:
        _ensure_label_review_tables(conn)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO label_review_batches
            (batch_name, elder_id, room, record_date, source_kind, source_ref, notes, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (batch_name, batch_elder, batch_room, batch_date, source_kind, source_ref, notes, created_by),
        )
        batch_id = int(cursor.lastrowid)

        item_ids_by_external: dict[str, int] = {}
        parent_links: list[tuple[int, str]] = []
        for row in normalized.to_dict("records"):
            cursor.execute(
                """
                INSERT INTO label_review_items
                (
                    batch_id, external_id, parent_external_id, elder_id, room, record_date, granularity,
                    timestamp_start, timestamp_end, current_label, proposed_label, confidence_tier,
                    proposal_score, reason_codes_json, rationale_json, source_model, review_status,
                    display_order, created_by
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    row["external_id"],
                    row["parent_external_id"],
                    row["elder_id"],
                    row["room"],
                    row["record_date"],
                    row["granularity"],
                    row["timestamp_start"].strftime("%Y-%m-%d %H:%M:%S"),
                    row["timestamp_end"].strftime("%Y-%m-%d %H:%M:%S"),
                    row["current_label"],
                    row["proposed_label"],
                    row["confidence_tier"],
                    None if pd.isna(row["proposal_score"]) else float(row["proposal_score"]),
                    row["reason_codes_json"],
                    row["rationale_json"],
                    row["source_model"],
                    row["review_status"],
                    int(row["display_order"]),
                    created_by,
                ),
            )
            item_id = int(cursor.lastrowid)
            item_ids_by_external[str(row["external_id"])] = item_id
            parent_external_id = row.get("parent_external_id")
            if parent_external_id:
                parent_links.append((item_id, str(parent_external_id)))

        for item_id, parent_external_id in parent_links:
            parent_id = item_ids_by_external.get(parent_external_id)
            if parent_id:
                cursor.execute(
                    "UPDATE label_review_items SET parent_id = ? WHERE id = ?",
                    (parent_id, item_id),
                )

        _refresh_batch_summary(conn, batch_id)
        _insert_log(
            conn,
            batch_id=batch_id,
            action="batch_created",
            actor=created_by,
            note=notes,
            payload={
                "proposal_count": len(normalized),
                "source_kind": source_kind,
                "source_ref": source_ref,
            },
        )
        conn.commit()
        return batch_id


def list_label_review_batches(
    *,
    elder_id: str | None = None,
    room: str | None = None,
    record_date: str | dt_date | None = None,
) -> pd.DataFrame:
    with get_dashboard_connection() as conn:
        _ensure_label_review_tables(conn)
        query = """
            SELECT
                id, batch_name, elder_id, room, record_date, source_kind, source_ref, notes,
                created_by, created_at, batch_status, proposal_count, approved_count,
                rejected_count, applied_count, last_action_at
            FROM label_review_batches
            WHERE 1 = 1
        """
        params: list = []
        if elder_id:
            query += " AND elder_id = ?"
            params.append(elder_id)
        if room:
            query += " AND " + ROOM_MATCH_SQL
            params.append(room)
        record_date_str = _coerce_record_date(record_date)
        if record_date_str:
            query += " AND record_date = ?"
            params.append(record_date_str)
        query += " ORDER BY last_action_at DESC, id DESC"
        return _fetch_dataframe(conn, query, tuple(params))


def get_label_review_items(batch_id: int, *, granularity: str | None = None) -> pd.DataFrame:
    with get_dashboard_connection() as conn:
        _ensure_label_review_tables(conn)
        query = """
            SELECT
                id, batch_id, parent_id, external_id, parent_external_id, elder_id, room, record_date,
                granularity, timestamp_start, timestamp_end, current_label, proposed_label,
                confidence_tier, proposal_score, reason_codes_json, rationale_json, source_model,
                review_status, review_note, reviewed_by, reviewed_at, applied_at, applied_by,
                display_order, created_by, created_at
            FROM label_review_items
            WHERE batch_id = ?
        """
        params: list = [batch_id]
        if granularity:
            query += " AND granularity = ?"
            params.append(str(granularity).strip().lower())
        query += " ORDER BY display_order ASC, id ASC"
        return _fetch_dataframe(conn, query, tuple(params))


def get_label_review_audit_log(batch_id: int) -> pd.DataFrame:
    with get_dashboard_connection() as conn:
        _ensure_label_review_tables(conn)
        return _fetch_dataframe(
            conn,
            """
            SELECT
                id, batch_id, proposal_id, action, from_status, to_status, actor, note, payload_json, created_at
            FROM label_review_decision_log
            WHERE batch_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (batch_id,),
        )


def update_label_review_statuses(
    batch_id: int,
    proposal_ids: list[int],
    new_status: str,
    *,
    actor: str,
    note: str | None = None,
) -> dict:
    if not proposal_ids:
        return {"ok": False, "updated": 0, "error": "No proposal_ids provided."}
    if new_status not in VALID_REVIEW_STATUSES:
        return {"ok": False, "updated": 0, "error": f"Unsupported status: {new_status}"}

    placeholders = ",".join(["?"] * len(proposal_ids))
    with get_dashboard_connection() as conn:
        _ensure_label_review_tables(conn)
        rows = _fetch_dataframe(
            conn,
            f"""
            SELECT id, review_status
            FROM label_review_items
            WHERE batch_id = ? AND id IN ({placeholders})
            ORDER BY id ASC
            """,
            [batch_id, *proposal_ids],
        )
        if rows.empty:
            return {"ok": False, "updated": 0, "error": "No matching proposal rows found."}

        updated = 0
        for item in rows.to_dict("records"):
            old_status = str(item["review_status"] or "proposed")
            conn.execute(
                """
                UPDATE label_review_items
                SET review_status = ?, review_note = ?, reviewed_by = ?, reviewed_at = CURRENT_TIMESTAMP
                WHERE id = ? AND batch_id = ?
                """,
                (new_status, note, actor, int(item["id"]), batch_id),
            )
            _insert_log(
                conn,
                batch_id=batch_id,
                proposal_id=int(item["id"]),
                action="status_changed",
                from_status=old_status,
                to_status=new_status,
                actor=actor,
                note=note,
            )
            updated += 1

        _refresh_batch_summary(conn, batch_id)
        conn.commit()
        return {"ok": True, "updated": updated}


def _load_exact_rows_for_item(conn, item: dict, *, allow_synthetic: bool = False) -> pd.DataFrame:
    start_ts = pd.to_datetime(item["timestamp_start"], errors="coerce")
    end_ts = pd.to_datetime(item["timestamp_end"], errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return pd.DataFrame(columns=["elder_id", "room", "record_date", "timestamp", "current_label"])
    df = _fetch_dataframe(
        conn,
        """
        SELECT elder_id, room, record_date, timestamp, activity_type AS current_label
        FROM adl_history
        WHERE elder_id = ? AND """ + ROOM_MATCH_SQL + """ AND record_date = ?
          AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp ASC
        """,
        (
            item["elder_id"],
            item["room"],
            item["record_date"],
            start_ts.strftime("%Y-%m-%d %H:%M:%S"),
            end_ts.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    if df.empty and allow_synthetic:
        return pd.DataFrame(
            [
                {
                    "elder_id": item["elder_id"],
                    "room": item["room"],
                    "record_date": item["record_date"],
                    "timestamp": item["timestamp_start"],
                    "current_label": item.get("current_label"),
                }
            ]
        )
    return df


def _resolve_exact_timestamp_rows(
    conn,
    batch_id: int,
    *,
    base_statuses: set[str],
    active_explicit_statuses: set[str],
    blocking_statuses: set[str],
    allow_synthetic: bool = False,
) -> pd.DataFrame:
    items = _fetch_dataframe(
        conn,
        """
        SELECT
            id, batch_id, parent_id, external_id, parent_external_id, elder_id, room, record_date,
            granularity, timestamp_start, timestamp_end, current_label, proposed_label,
            confidence_tier, proposal_score, reason_codes_json, rationale_json, source_model,
            review_status, review_note, reviewed_by, reviewed_at, applied_at, applied_by,
            display_order, created_by, created_at
        FROM label_review_items
        WHERE batch_id = ?
        ORDER BY display_order ASC, id ASC
        """,
        (batch_id,),
    )
    if items.empty:
        return pd.DataFrame(
            columns=[
                "batch_id",
                "timestamp",
                "record_date",
                "elder_id",
                "room",
                "current_label",
                "proposed_label",
                "review_status",
                "source_proposal_id",
                "source_parent_id",
                "source_granularity",
                "confidence_tier",
                "proposal_score",
                "reason_codes",
            ]
        )

    items = items.copy()
    items["timestamp_start"] = pd.to_datetime(items["timestamp_start"], errors="coerce")
    items["timestamp_end"] = pd.to_datetime(items["timestamp_end"], errors="coerce")
    items = items.dropna(subset=["timestamp_start", "timestamp_end"])
    items = items.sort_values(["display_order", "id"]).reset_index(drop=True)

    explicit = items[items["granularity"] == "timestamp"].copy()
    explicit["ts_key"] = explicit["timestamp_start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    explicit = explicit.sort_values(["ts_key", "id"])
    latest_explicit = explicit.drop_duplicates(subset=["elder_id", "room", "record_date", "ts_key"], keep="last")

    active_explicit_map: dict[str, dict] = {}
    blocked_ts_keys: set[str] = set()
    for row in latest_explicit.to_dict("records"):
        status = str(row["review_status"] or "proposed")
        ts_key = str(row["ts_key"])
        if status in active_explicit_statuses:
            active_explicit_map[ts_key] = row
        elif status in blocking_statuses:
            blocked_ts_keys.add(ts_key)

    records: list[dict] = []

    segments = items[(items["granularity"] == "segment") & (items["review_status"].isin(base_statuses))].copy()
    for segment in segments.to_dict("records"):
        matched = _load_exact_rows_for_item(conn, segment, allow_synthetic=allow_synthetic)
        if matched.empty:
            continue
        matched["timestamp"] = pd.to_datetime(matched["timestamp"], errors="coerce")
        matched = matched.dropna(subset=["timestamp"])
        for row in matched.to_dict("records"):
            ts_key = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            if ts_key in blocked_ts_keys or ts_key in active_explicit_map:
                continue
            records.append(
                {
                    "batch_id": batch_id,
                    "timestamp": ts_key,
                    "record_date": str(row.get("record_date") or segment["record_date"]),
                    "elder_id": str(row.get("elder_id") or segment["elder_id"]),
                    "room": str(row.get("room") or segment["room"]),
                    "current_label": row.get("current_label") or segment.get("current_label"),
                    "proposed_label": segment["proposed_label"],
                    "review_status": segment["review_status"],
                    "source_proposal_id": int(segment["id"]),
                    "source_parent_id": int(segment["parent_id"]) if pd.notna(segment["parent_id"]) else None,
                    "source_granularity": "segment",
                    "confidence_tier": segment.get("confidence_tier"),
                    "proposal_score": segment.get("proposal_score"),
                    "reason_codes": _display_reason_codes(segment.get("reason_codes_json")),
                }
            )

    for explicit_row in active_explicit_map.values():
        matched = _load_exact_rows_for_item(conn, explicit_row, allow_synthetic=allow_synthetic)
        if matched.empty:
            continue
        matched["timestamp"] = pd.to_datetime(matched["timestamp"], errors="coerce")
        matched = matched.dropna(subset=["timestamp"])
        for row in matched.to_dict("records"):
            ts_key = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            records.append(
                {
                    "batch_id": batch_id,
                    "timestamp": ts_key,
                    "record_date": str(row.get("record_date") or explicit_row["record_date"]),
                    "elder_id": str(row.get("elder_id") or explicit_row["elder_id"]),
                    "room": str(row.get("room") or explicit_row["room"]),
                    "current_label": row.get("current_label") or explicit_row.get("current_label"),
                    "proposed_label": explicit_row["proposed_label"],
                    "review_status": explicit_row["review_status"],
                    "source_proposal_id": int(explicit_row["id"]),
                    "source_parent_id": int(explicit_row["parent_id"]) if pd.notna(explicit_row["parent_id"]) else None,
                    "source_granularity": "timestamp",
                    "confidence_tier": explicit_row.get("confidence_tier"),
                    "proposal_score": explicit_row.get("proposal_score"),
                    "reason_codes": _display_reason_codes(explicit_row.get("reason_codes_json")),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "batch_id",
                "timestamp",
                "record_date",
                "elder_id",
                "room",
                "current_label",
                "proposed_label",
                "review_status",
                "source_proposal_id",
                "source_parent_id",
                "source_granularity",
                "confidence_tier",
                "proposal_score",
                "reason_codes",
            ]
        )

    resolved = pd.DataFrame(records)
    resolved["timestamp"] = pd.to_datetime(resolved["timestamp"], errors="coerce")
    resolved = resolved.dropna(subset=["timestamp"])
    resolved = resolved.sort_values(["timestamp", "source_granularity", "source_proposal_id"]).drop_duplicates(
        subset=["elder_id", "room", "record_date", "timestamp"], keep="last"
    )
    resolved["timestamp"] = resolved["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    resolved = resolved.sort_values(["timestamp", "source_granularity", "source_proposal_id"]).reset_index(drop=True)
    return resolved


def build_proposed_timestamp_set(batch_id: int) -> pd.DataFrame:
    with get_dashboard_connection() as conn:
        _ensure_label_review_tables(conn)
        return _resolve_exact_timestamp_rows(
            conn,
            batch_id,
            base_statuses=ACTIVE_REVIEW_STATUSES,
            active_explicit_statuses=ACTIVE_REVIEW_STATUSES,
            blocking_statuses=BLOCKING_STATUSES,
        )


def build_apply_ready_payload(batch_id: int) -> dict:
    with get_dashboard_connection() as conn:
        _ensure_label_review_tables(conn)
        batch_df = _fetch_dataframe(
            conn,
            """
            SELECT id, batch_name, elder_id, room, record_date, batch_status
            FROM label_review_batches
            WHERE id = ?
            """,
            (batch_id,),
        )
        if batch_df.empty:
            return {"batch_id": batch_id, "rows": [], "error": "Batch not found."}

        resolved = _resolve_exact_timestamp_rows(
            conn,
            batch_id,
            base_statuses=APPLY_READY_STATUSES,
            active_explicit_statuses=APPLY_READY_STATUSES,
            blocking_statuses=BLOCKING_STATUSES,
        )
        rows = []
        for row in resolved.to_dict("records"):
            rows.append(
                {
                    "timestamp": str(row["timestamp"]),
                    "record_date": str(row["record_date"]),
                    "elder_id": str(row["elder_id"]),
                    "room": str(row["room"]),
                    "current_label": row.get("current_label"),
                    "proposed_label": str(row["proposed_label"]),
                    "source_proposal_id": int(row["source_proposal_id"]),
                    "source_granularity": str(row["source_granularity"]),
                }
            )
        batch_row = batch_df.iloc[0].to_dict()
        return {
            "batch_id": int(batch_row["id"]),
            "batch_name": str(batch_row["batch_name"]),
            "batch_status": str(batch_row["batch_status"]),
            "rows": rows,
        }


def load_label_review_snapshot(
    *,
    elder_id: str | None = None,
    room: str | None = None,
    record_date: str | dt_date | None = None,
    batch_id: int | None = None,
) -> dict:
    """Load proposal-review state for the UI without raising on DB failures."""
    empty_items = pd.DataFrame()
    empty_batches = pd.DataFrame()
    empty_audit = pd.DataFrame()
    empty_proposed = pd.DataFrame()
    try:
        batch_filters = {}
        if elder_id:
            batch_filters["elder_id"] = elder_id
        if room:
            batch_filters["room"] = room
        if record_date:
            batch_filters["record_date"] = record_date
        batches_df = list_label_review_batches(**batch_filters)
        selected_batch_id = batch_id
        if selected_batch_id is None and not batches_df.empty:
            selected_batch_id = int(batches_df.iloc[0]["id"])

        items_df = empty_items
        proposed_timestamp_df = empty_proposed
        audit_df = empty_audit
        apply_ready_payload = {"batch_id": selected_batch_id, "rows": []}
        selected_batch_row = None
        if selected_batch_id is not None and not batches_df.empty:
            match = batches_df.loc[batches_df["id"].astype(int) == int(selected_batch_id)]
            if not match.empty:
                selected_batch_row = match.iloc[0]
            items_df = get_label_review_items(int(selected_batch_id))
            proposed_timestamp_df = build_proposed_timestamp_set(int(selected_batch_id))
            audit_df = get_label_review_audit_log(int(selected_batch_id))
            apply_ready_payload = build_apply_ready_payload(int(selected_batch_id))

        return {
            "ok": True,
            "error": None,
            "batches_df": batches_df,
            "selected_batch_id": selected_batch_id,
            "selected_batch_row": selected_batch_row,
            "items_df": items_df,
            "proposed_timestamp_df": proposed_timestamp_df,
            "audit_df": audit_df,
            "apply_ready_payload": apply_ready_payload,
        }
    except Exception as e:
        logger.warning("Failed to load label review snapshot: %s", e)
        return {
            "ok": False,
            "error": str(e),
            "batches_df": empty_batches,
            "selected_batch_id": batch_id,
            "selected_batch_row": None,
            "items_df": empty_items,
            "proposed_timestamp_df": empty_proposed,
            "audit_df": empty_audit,
            "apply_ready_payload": {"batch_id": batch_id, "rows": []},
        }


def _group_apply_ready_rows(rows_df: pd.DataFrame) -> list[dict]:
    if rows_df.empty:
        return []
    out = rows_df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if out.empty:
        return []

    prev_ts = out["timestamp"].shift()
    gap_break = out["timestamp"].sub(prev_ts).dt.total_seconds().fillna(10).ne(10)
    label_break = out["proposed_label"].ne(out["proposed_label"].shift())
    out["group_id"] = (gap_break | label_break).cumsum()

    groups: list[dict] = []
    for _, block in out.groupby("group_id"):
        start_ts = pd.to_datetime(block["timestamp"].iloc[0])
        end_ts = pd.to_datetime(block["timestamp"].iloc[-1])
        # Existing correction batch APIs require a non-zero window.
        # Using +9s keeps the update inside the last 10-second bin instead of spilling
        # into the next row.
        safe_end_ts = end_ts + pd.Timedelta(seconds=9)
        groups.append(
            {
                "start_time": start_ts.time(),
                "end_time": safe_end_ts.time(),
                "new_label": str(block["proposed_label"].iloc[0]),
            }
        )
    return groups


def apply_approved_label_review_batch(batch_id: int, *, applied_by: str = "ops_ui") -> dict:
    payload = build_apply_ready_payload(batch_id)
    rows = payload.get("rows", [])
    if not rows:
        return {"ok": False, "applied_rows": 0, "applied_ranges": 0, "error": "No approved proposal rows to apply."}

    rows_df = pd.DataFrame(rows)
    rows_df["timestamp"] = pd.to_datetime(rows_df["timestamp"], errors="coerce")
    rows_df = rows_df.dropna(subset=["timestamp"])
    if rows_df.empty:
        return {"ok": False, "applied_rows": 0, "applied_ranges": 0, "error": "No valid timestamps in approved payload."}

    applied_rows = 0
    applied_ranges = 0
    applied_source_ids: set[int] = set()
    details: list[dict] = []

    for (elder_id, room, record_date), group in rows_df.groupby(["elder_id", "room", "record_date"], dropna=False):
        corrections = _group_apply_ready_rows(group)
        if not corrections:
            continue
        summary = apply_batch_corrections(
            elder_id=str(elder_id),
            room=str(room),
            record_date=pd.to_datetime(record_date).date(),
            corrections=corrections,
            corrected_by=applied_by,
        )
        details.append({"elder_id": elder_id, "room": room, "record_date": record_date, **summary})
        if summary.get("ok"):
            applied_rows += int(summary.get("rows_updated") or 0)
            applied_ranges += int(summary.get("ranges_applied") or 0)
            applied_source_ids.update(int(v) for v in group["source_proposal_id"].dropna().astype(int).tolist())

    if not applied_source_ids:
        return {
            "ok": False,
            "applied_rows": applied_rows,
            "applied_ranges": applied_ranges,
            "details": details,
            "error": "Approved proposals did not update any rows.",
        }

    placeholders = ",".join(["?"] * len(applied_source_ids))
    with get_dashboard_connection() as conn:
        _ensure_label_review_tables(conn)
        previous = _fetch_dataframe(
            conn,
            f"""
            SELECT id, review_status
            FROM label_review_items
            WHERE batch_id = ? AND id IN ({placeholders})
            """,
            [batch_id, *sorted(applied_source_ids)],
        )
        conn.execute(
            f"""
            UPDATE label_review_items
            SET review_status = 'applied', applied_by = ?, applied_at = CURRENT_TIMESTAMP
            WHERE batch_id = ? AND id IN ({placeholders})
            """,
            [applied_by, batch_id, *sorted(applied_source_ids)],
        )
        for row in previous.to_dict("records"):
            _insert_log(
                conn,
                batch_id=batch_id,
                proposal_id=int(row["id"]),
                action="applied",
                from_status=str(row["review_status"] or "approved"),
                to_status="applied",
                actor=applied_by,
                payload={"applied_rows": applied_rows, "applied_ranges": applied_ranges},
            )
        _refresh_batch_summary(conn, batch_id)
        conn.commit()

    return {
        "ok": True,
        "applied_rows": applied_rows,
        "applied_ranges": applied_ranges,
        "applied_proposals": len(applied_source_ids),
        "details": details,
    }
