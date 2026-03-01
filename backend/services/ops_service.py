import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config import get_room_config
from elderlycare_v1_16.config.settings import ARCHIVE_DATA_DIR
from utils.room_utils import normalize_room_name

# Ensure backend and project root are in sys.path for health_server's backend.* imports
backend_root = str(Path(__file__).resolve().parent.parent)
project_root = str(Path(backend_root).parent)

if backend_root not in sys.path:
    sys.path.append(backend_root)
if project_root not in sys.path:
    sys.path.append(project_root)

from health_server import build_ml_snapshot_report
from services.db_utils import get_dashboard_connection, parse_json_object, query_df
from utils.elder_id_utils import apply_canonical_alias_map, parse_elder_id_from_filename

logger = logging.getLogger(__name__)


def _elder_id_from_training_filename(filename: str) -> str:
    parsed = parse_elder_id_from_filename(filename)
    return apply_canonical_alias_map(parsed)


def _fallback_model_rooms_from_artifacts(elder_id: str) -> list[str]:
    """Derive model rooms from local model registry artifacts."""
    try:
        from ml.registry import ModelRegistry

        backend_dir = Path(__file__).resolve().parent.parent
        registry = ModelRegistry(str(backend_dir))
        models_dir = registry.get_models_dir(elder_id)
        if not models_dir.exists():
            return []
        return sorted({p.name.replace("_versions.json", "") for p in models_dir.glob("*_versions.json")})
    except Exception:
        return []


def get_model_status(elder_id: str) -> dict:
    """Fetch the latest ML health snapshot for all rooms for a resident."""
    if not elder_id or elder_id == "All":
         return {"status": {"overall": "not_available", "reason": "No resident selected"}}

    try:
        report, status_code = build_ml_snapshot_report(
            elder_id=elder_id,
            room="",
            lookback_runs=20,
            include_raw=False,
        )
        if status_code >= 500:
            logger.warning(f"ML snapshot returned {status_code} for {elder_id}")
            
        report = report if isinstance(report, dict) else {}
        room_rows = report.get("rooms")
        room_rows = room_rows if isinstance(room_rows, list) else []
        only_global_room = (
            len(room_rows) == 1
            and str(room_rows[0].get("room", "")).strip().lower().replace("_", "")
            in {"allrooms", "allroom"}
        )
        if not room_rows or only_global_room:
            fallback_rooms = _fallback_model_rooms_from_artifacts(elder_id)
            if fallback_rooms:
                overall = str(report.get("status", {}).get("overall", "not_available"))
                report["rooms"] = [
                    {"room": room_name, "status": overall if overall else "not_available", "metrics": {}}
                    for room_name in fallback_rooms
                ]
        return report
    except Exception as e:
        logger.error(f"Failed to fetch model status for {elder_id}: {e}")
        return {"status": {"overall": "error", "reason": str(e)}}


def get_sample_collection_status(elder_id: str) -> dict:
    """
    Analyze the archives to determine how many days of training data exist per room.
    Target is usually 21 days for Phase 1 stability.
    """
    if not elder_id or elder_id == "All":
        return {}
        
    try:
        target_days = 21
        config_rooms = [room for room in get_room_config().get_all_rooms().keys() if str(room).lower() != "default"]
        normalized_to_display = {
            normalize_room_name(room): room
            for room in config_rooms
            if normalize_room_name(room)
        }
        room_counts = {room: 0 for room in config_rooms}

        # Primary source: adl_history day coverage by room (works for parquet/xlsx archive layouts).
        df_days = query_df(
            """
            SELECT room, DATE(timestamp) AS day_key
            FROM adl_history
            WHERE elder_id = ?
            """,
            (elder_id,),
        )
        if not df_days.empty and {"room", "day_key"}.issubset(df_days.columns):
            day_sets: dict[str, set[str]] = {room: set() for room in room_counts}
            for _, row in df_days.iterrows():
                room_key = normalize_room_name(row.get("room"))
                room_display = normalized_to_display.get(room_key)
                if room_display in day_sets:
                    day_val = row.get("day_key")
                    if pd.notna(day_val):
                        day_sets[room_display].add(str(day_val))
            for room_display, values in day_sets.items():
                room_counts[room_display] = len(values)
        else:
            # Fallback: infer from archived file names (recursive + modern parquet layout).
            fallback_day_sets: dict[str, set[str]] = {room: set() for room in room_counts}
            for ext in (".parquet", ".xlsx", ".xls", ".csv"):
                for file_path in ARCHIVE_DATA_DIR.rglob(f"*train*{ext}"):
                    if not file_path.is_file():
                        continue
                    if file_path.name.startswith("~$") or file_path.name.startswith("."):
                        continue
                    if _elder_id_from_training_filename(file_path.name) != elder_id:
                        continue
                    match = re.search(r"train[_-](\d{1,2}[a-z]{3}\d{4})", file_path.stem.lower())
                    if not match:
                        continue
                    day_key = str(match.group(1))
                    # Legacy filenames can include room token before "_train".
                    prefix = file_path.stem.lower().split("_train", 1)[0]
                    room_tokens = prefix.split("_")[2:]
                    room_key = normalize_room_name("_".join(room_tokens)) if room_tokens else None
                    room_display = normalized_to_display.get(room_key)
                    if room_display:
                        fallback_day_sets[room_display].add(day_key)
            for room_display, values in fallback_day_sets.items():
                room_counts[room_display] = len(values)

        return {
            "counts": room_counts,
            "target": target_days,
            "ready_rooms": [room for room, count in room_counts.items() if count >= target_days],
        }
    except Exception as e:
        logger.error(f"Failed to calculate sample collection for {elder_id}: {e}")
        return {}


def get_hard_negative_summary(elder_id: str, days: int = 30) -> dict:
    """Summarize the active learning hard-negative queue."""
    if not elder_id or elder_id == "All":
        return {"open_count": 0, "recent_resolved": 0, "last_run": None}
        
    try:
        cutoff = datetime.now() - timedelta(days=days)
        open_df = query_df(
            """
            SELECT COUNT(*) AS count
            FROM hard_negative_queue
            WHERE elder_id = ? AND status = 'open'
            """,
            (elder_id,),
        )
        resolved_df = query_df(
            """
            SELECT COUNT(*) AS count
            FROM hard_negative_queue
            WHERE elder_id = ?
              AND status IN ('applied', 'dismissed')
              AND updated_at >= ?
            """,
            (elder_id, cutoff),
        )
        last_df = query_df(
            """
            SELECT MAX(updated_at) AS last_run
            FROM hard_negative_queue
            WHERE elder_id = ?
            """,
            (elder_id,),
        )

        open_count = int(open_df["count"].iloc[0]) if not open_df.empty else 0
        resolved_count = int(resolved_df["count"].iloc[0]) if not resolved_df.empty else 0
        last_run = last_df["last_run"].iloc[0] if not last_df.empty else None

        # If queue has no rows, still expose miner heartbeat from latest training run.
        if pd.isna(last_run) or last_run is None:
            train_df = query_df(
                """
                SELECT MAX(training_date) AS last_train
                FROM training_history
                WHERE elder_id = ?
                """,
                (elder_id,),
            )
            if not train_df.empty:
                candidate = train_df["last_train"].iloc[0]
                if pd.notna(candidate):
                    last_run = candidate

        return {
            "open_count": open_count,
            "recent_resolved": resolved_count,
            "last_run": last_run,
        }
    except Exception as e:
        logger.error(f"Failed to fetch hard negative summary for {elder_id}: {e}")
        return {"open_count": 0, "recent_resolved": 0, "last_run": None}


def get_daily_ops_summary(elder_id: str) -> dict:
    """Top-level health checks for daily non-ML operations (data flow)."""
    if not elder_id or elder_id == "All":
        return {"freshness": "unknown", "open_alerts": 0}
        
    try:
        adl_df = query_df("SELECT MAX(timestamp) AS last_ts FROM adl_history WHERE elder_id = ?", (elder_id,))
        try:
            sensor_df = query_df("SELECT MAX(timestamp) AS last_ts FROM sensor_data WHERE elder_id = ?", (elder_id,))
        except Exception:
            sensor_df = pd.DataFrame(columns=["last_ts"])

        adl_ts = pd.to_datetime(adl_df["last_ts"].iloc[0]) if not adl_df.empty and pd.notna(adl_df["last_ts"].iloc[0]) else None
        sensor_ts = (
            pd.to_datetime(sensor_df["last_ts"].iloc[0])
            if not sensor_df.empty and pd.notna(sensor_df["last_ts"].iloc[0])
            else None
        )
        candidates = [ts for ts in [adl_ts, sensor_ts] if ts is not None]
        last_ts = max(candidates) if candidates else None

        hours_ago = None
        if last_ts is not None:
            if isinstance(last_ts, str):
                last_ts = datetime.fromisoformat(last_ts[:19].replace("Z", ""))
            hours_ago = (datetime.now() - last_ts).total_seconds() / 3600.0

        alerts_df = query_df("SELECT COUNT(*) AS count FROM alerts WHERE elder_id = ? AND is_read = 0", (elder_id,))
        open_alerts = int(alerts_df["count"].iloc[0]) if not alerts_df.empty else 0

        return {
            "last_ingestion_time": last_ts,
            "last_adl_time": adl_ts,
            "last_sensor_time": sensor_ts,
            "hours_since_last_ingestion": hours_ago,
            "is_stale": (hours_ago is None) or (hours_ago > 6.0),
            "open_alerts": open_alerts,
        }
    except Exception as e:
        logger.error(f"Failed to fetch daily ops summary for {elder_id}: {e}")
        return {"freshness": "error", "open_alerts": 0}
