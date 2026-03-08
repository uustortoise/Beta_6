import json
import logging
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config import get_release_gates_config, get_room_config
from elderlycare_v1_16.config.settings import ARCHIVE_DATA_DIR
from ml.release_gates import resolve_scheduled_threshold
from utils.room_utils import normalize_room_name

# Ensure backend and project root are in sys.path for health_server's backend.* imports
backend_root = str(Path(__file__).resolve().parent.parent)
project_root = str(Path(backend_root).parent)

if backend_root not in sys.path:
    sys.path.append(backend_root)
if project_root not in sys.path:
    sys.path.append(project_root)

from health_server import build_ml_snapshot_report
from services.db_utils import coerce_bool, get_dashboard_connection, parse_json_object, query_df
from utils.elder_id_utils import apply_canonical_alias_map, parse_elder_id_from_filename

logger = logging.getLogger(__name__)


def _to_float_or_none(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return float(parsed)


def _normalize_room_key(value: str) -> str:
    normalized = normalize_room_name(value)
    return normalized if isinstance(normalized, str) else ""


def _parse_manifest_timestamp(token: str) -> datetime | None:
    value = str(token or "").strip()
    if not value:
        return None
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return datetime.strptime(value, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}[T_ ]\d{2}:\d{2}:\d{2}", value):
            return datetime.fromisoformat(value.replace("_", " ").replace("T", " "))
        if re.fullmatch(r"\d{8}", value):
            return datetime.strptime(value, "%Y%m%d").replace(hour=23, minute=59, second=59)
        if re.fullmatch(r"(?i)\d{1,2}[a-z]{3}\d{4}", value):
            return datetime.strptime(value.lower(), "%d%b%Y").replace(hour=23, minute=59, second=59)
    except ValueError:
        return None
    return None


def _resolve_sensor_event_time(metadata: dict) -> str | None:
    manifest = metadata.get("training_manifest") if isinstance(metadata, dict) else None
    if not isinstance(manifest, list) or not manifest:
        return None

    latest_ts: datetime | None = None
    for entry in manifest:
        text = str(entry or "")
        if not text:
            continue
        candidates: list[datetime] = []
        for match in re.findall(r"\d{4}-\d{2}-\d{2}[T_ ]\d{2}:\d{2}:\d{2}", text):
            parsed = _parse_manifest_timestamp(match)
            if parsed is not None:
                candidates.append(parsed)
        for match in re.findall(r"\d{4}-\d{2}-\d{2}", text):
            parsed = _parse_manifest_timestamp(match)
            if parsed is not None:
                candidates.append(parsed)
        for match in re.findall(r"(?<!\d)(\d{8})(?!\d)", text):
            parsed = _parse_manifest_timestamp(match)
            if parsed is not None:
                candidates.append(parsed)
        for match in re.findall(r"(?i)(\d{1,2}[a-z]{3}\d{4})", text):
            parsed = _parse_manifest_timestamp(match)
            if parsed is not None:
                candidates.append(parsed)
        if not candidates:
            continue
        entry_latest = max(candidates)
        if latest_ts is None or entry_latest > latest_ts:
            latest_ts = entry_latest
    return latest_ts.isoformat(sep=" ") if latest_ts is not None else None


def _resolve_schedule_day_bucket(schedule: list, training_days: float | None) -> str | None:
    if training_days is None:
        return None
    for entry in schedule if isinstance(schedule, list) else []:
        if not isinstance(entry, dict):
            continue
        min_days = _to_float_or_none(entry.get("min_days"))
        max_days = _to_float_or_none(entry.get("max_days"))
        if min_days is None:
            continue
        in_bucket = float(training_days) >= float(min_days) and (
            max_days is None or float(training_days) <= float(max_days)
        )
        if not in_bucket:
            continue
        if max_days is None:
            return f"day_{int(min_days)}+"
        return f"day_{int(min_days)}-{int(max_days)}"
    return None


def _derive_room_reports_from_metrics(metadata: dict, room_policy_map: dict) -> list[dict]:
    metrics = metadata.get("metrics") if isinstance(metadata, dict) else None
    if not isinstance(metrics, list):
        return []

    derived: list[dict] = []
    for item in metrics:
        if not isinstance(item, dict):
            continue
        room_name = str(item.get("room") or "").strip()
        room_key = _normalize_room_key(room_name)
        if not room_name or not room_key:
            continue

        candidate_macro = _to_float_or_none(item.get("macro_f1"))
        candidate_acc = _to_float_or_none(item.get("accuracy"))
        training_days = _to_float_or_none(item.get("training_days"))
        room_schedule = (
            room_policy_map.get(room_key, {}).get("schedule", [])
            if isinstance(room_policy_map.get(room_key, {}), dict)
            else []
        )
        threshold_required = (
            resolve_scheduled_threshold(room_schedule, float(training_days))
            if training_days is not None and isinstance(room_schedule, list)
            else None
        )

        derived.append(
            {
                "room": room_name,
                "pass": bool(item.get("gate_pass", False)),
                "reasons": item.get("gate_reasons", []) if isinstance(item.get("gate_reasons"), list) else [],
                "candidate_threshold_required": _to_float_or_none(threshold_required),
                "training_days": training_days,
                "candidate_summary": {
                    "macro_f1_mean": candidate_macro,
                    "accuracy_mean": candidate_acc,
                    "stability_accuracy_mean": _to_float_or_none(item.get("candidate_stability_accuracy_mean")),
                    "transition_macro_f1_mean": _to_float_or_none(item.get("candidate_transition_macro_f1_mean")),
                },
                "candidate_stability_accuracy_mean": _to_float_or_none(item.get("candidate_stability_accuracy_mean")),
                "candidate_transition_macro_f1_mean": _to_float_or_none(item.get("candidate_transition_macro_f1_mean")),
                "champion_macro_f1_mean": _to_float_or_none(item.get("champion_macro_f1_mean")),
                "candidate_wf_config": item.get("candidate_wf_config", {})
                if isinstance(item.get("candidate_wf_config"), dict)
                else {},
            }
        )
    return derived


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return int(default)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return float(default)
    try:
        parsed = float(str(raw).strip())
        return float(default) if math.isnan(parsed) else float(parsed)
    except (TypeError, ValueError):
        return float(default)


def _empty_model_update_monitor(error: str | None = None) -> dict:
    payload = {
        "total_runs": 0,
        "wf_enabled_runs": 0,
        "wf_pass_runs": 0,
        "wf_fail_runs": 0,
        "wf_pass_rate_pct": None,
        "top_failure_reason": None,
        "latest_delta": None,
        "latest_run": None,
        "room_trends": [],
        "room_history_points": [],
        "release_gate_profile": {},
        "metric_labels": {
            "latest_candidate_macro_f1_mean": "WF Candidate F1",
            "latest_candidate_accuracy_mean": "WF Candidate Accuracy",
            "latest_candidate_stability_accuracy_mean": "Stability Score",
            "latest_candidate_transition_macro_f1_mean": "Transition Score",
            "latest_champion_macro_f1_mean": "Champion WF F1",
            "latest_run_accuracy": "Raw Training Run Accuracy",
        },
        "production_counters": {
            "preprocessing_fail": 0,
            "viability_fail": 0,
            "statistical_validity_fail": 0,
            "gate_fail": 0,
            "promoted": 0,
            "no_op_retrain_skip": 0,
        },
    }
    if error:
        payload["error"] = str(error)
    return payload


def get_model_update_monitor(elder_id: str, days: int = 30, limit: int = 60) -> dict:
    """
    Summarize promotion-gate outcomes and trend metrics for active Ops Dashboard monitoring.
    """
    elder_id = str(elder_id or "").strip()
    if not elder_id:
        return _empty_model_update_monitor()

    days = max(1, int(days))
    limit = max(1, min(int(limit), 200))
    cutoff_date = datetime.now() - timedelta(days=days)

    try:
        history_df = query_df(
            """
            SELECT id, training_date, status, accuracy, metadata
            FROM training_history
            WHERE elder_id = ?
              AND training_date >= ?
            ORDER BY training_date DESC
            LIMIT ?
            """,
            (elder_id, cutoff_date, limit),
        )
    except Exception as e:
        logger.error(f"Failed to fetch model update monitor data for {elder_id}: {e}")
        return _empty_model_update_monitor(error=str(e))

    if history_df is None or history_df.empty:
        return _empty_model_update_monitor()

    total_runs = 0
    wf_enabled_runs = 0
    wf_pass_runs = 0
    wf_fail_runs = 0
    top_failure_counter: dict[str, int] = {}
    latest_run = None
    latest_run_id = None
    room_history: dict[str, list[dict]] = {}
    production_counters = {
        "preprocessing_fail": 0,
        "viability_fail": 0,
        "statistical_validity_fail": 0,
        "gate_fail": 0,
        "promoted": 0,
        "no_op_retrain_skip": 0,
    }
    latest_metadata: dict = {}
    latest_training_days_by_room: dict[str, float] = {}
    latest_training_days_global = None
    try:
        release_cfg = get_release_gates_config()
    except Exception:
        release_cfg = {}
    release_gates_cfg = release_cfg.get("release_gates", {}) if isinstance(release_cfg, dict) else {}
    room_policy_map = release_gates_cfg.get("rooms", {}) if isinstance(release_gates_cfg.get("rooms"), dict) else {}
    global_schedule = (
        release_gates_cfg.get("global", {}).get("schedule", [])
        if isinstance(release_gates_cfg.get("global"), dict)
        else []
    )

    for _, row in history_df.iterrows():
        total_runs += 1
        run_id = int(row.get("id")) if row.get("id") is not None else None
        training_date = row.get("training_date")
        run_status = str(row.get("status") or "")
        run_accuracy = row.get("accuracy")
        metadata = parse_json_object(row.get("metadata"))
        gate_failed = False
        if isinstance(metadata, dict):
            run_failure_stage = str(metadata.get("run_failure_stage") or "").strip().lower()
            if run_failure_stage == "preprocessing_contract_failed":
                production_counters["preprocessing_fail"] += 1
            elif run_failure_stage == "data_viability_failed":
                production_counters["viability_fail"] += 1
            elif run_failure_stage == "statistical_validity_failed":
                production_counters["statistical_validity_fail"] += 1
            elif run_failure_stage in {"global_gate_failed", "walk_forward_failed"}:
                gate_failed = True
        status_lower = run_status.strip().lower()
        if status_lower in {"rejected_by_global_gate", "rejected_by_walk_forward_gate"}:
            gate_failed = True
        if gate_failed:
            production_counters["gate_fail"] += 1
        if status_lower == "no_op_same_fingerprint":
            production_counters["no_op_retrain_skip"] += 1

        metrics_list = metadata.get("metrics") if isinstance(metadata, dict) else None
        run_sensor_event_time = _resolve_sensor_event_time(metadata if isinstance(metadata, dict) else {})
        if isinstance(metrics_list, list):
            production_counters["promoted"] += int(
                sum(1 for metric in metrics_list if bool((metric or {}).get("gate_pass", False)))
            )

        if latest_run is None:
            latest_run_id = run_id
            latest_metadata = metadata if isinstance(metadata, dict) else {}
            latest_run = {
                "run_id": run_id,
                "training_date": str(training_date) if pd.notna(training_date) else None,
                "status": run_status,
                "accuracy": float(run_accuracy) if pd.notna(run_accuracy) else None,
            }
            if isinstance(metrics_list, list):
                for metric in metrics_list:
                    if not isinstance(metric, dict):
                        continue
                    room_name = str(metric.get("room") or "").strip()
                    room_key = _normalize_room_key(room_name)
                    training_days = _to_float_or_none(metric.get("training_days"))
                    if room_key and training_days is not None:
                        latest_training_days_by_room[room_key] = float(training_days)
                if latest_training_days_by_room:
                    latest_training_days_global = max(latest_training_days_by_room.values())

        wf = metadata.get("walk_forward_gate", {}) if isinstance(metadata, dict) else {}
        if not isinstance(wf, dict):
            continue
        if wf.get("reason") in (None, "disabled"):
            continue

        wf_enabled_runs += 1
        if bool(wf.get("pass", False)):
            wf_pass_runs += 1
        else:
            wf_fail_runs += 1

        room_reports = wf.get("room_reports", [])
        if not isinstance(room_reports, list) or len(room_reports) == 0:
            room_reports = _derive_room_reports_from_metrics(metadata, room_policy_map)
        if not isinstance(room_reports, list):
            room_reports = []

        for rr in room_reports:
            if not isinstance(rr, dict):
                continue
            room_name = str(rr.get("room") or "").strip()
            if not room_name:
                continue
            room_key = _normalize_room_key(room_name)

            reasons = rr.get("reasons", [])
            if isinstance(reasons, list):
                for reason in reasons:
                    reason_txt = str(reason or "").strip()
                    if not reason_txt:
                        continue
                    reason_key = reason_txt.split(":")[0]
                    top_failure_counter[reason_key] = top_failure_counter.get(reason_key, 0) + 1

            candidate_summary = rr.get("candidate_summary", {})
            candidate_f1 = None
            candidate_accuracy = None
            if isinstance(candidate_summary, dict):
                candidate_f1 = candidate_summary.get("macro_f1_mean")
                candidate_accuracy = candidate_summary.get("accuracy_mean")

            threshold_required = _to_float_or_none(rr.get("candidate_threshold_required"))
            training_days = _to_float_or_none(rr.get("training_days"))
            room_schedule = (
                room_policy_map.get(room_key, {}).get("schedule", [])
                if isinstance(room_policy_map.get(room_key, {}), dict)
                else []
            )
            if threshold_required is None and training_days is not None and isinstance(room_schedule, list):
                threshold_required = _to_float_or_none(
                    resolve_scheduled_threshold(room_schedule, float(training_days))
                )
            threshold_bucket = (
                _resolve_schedule_day_bucket(room_schedule, training_days)
                if isinstance(room_schedule, list)
                else None
            )

            room_history.setdefault(room_name, []).append(
                {
                    "run_id": run_id,
                    "training_date": str(training_date) if pd.notna(training_date) else None,
                    "sensor_event_time": run_sensor_event_time,
                    "pass": bool(rr.get("pass", False)),
                    "champion_version": int(rr.get("champion_version")) if rr.get("champion_version") is not None else None,
                    "candidate_macro_f1_mean": float(candidate_f1) if candidate_f1 is not None else None,
                    "candidate_accuracy_mean": float(candidate_accuracy) if candidate_accuracy is not None else None,
                    "candidate_stability_accuracy_mean": (
                        float(rr.get("candidate_stability_accuracy_mean"))
                        if rr.get("candidate_stability_accuracy_mean") is not None
                        else None
                    ),
                    "candidate_transition_macro_f1_mean": (
                        float(rr.get("candidate_transition_macro_f1_mean"))
                        if rr.get("candidate_transition_macro_f1_mean") is not None
                        else None
                    ),
                    "champion_macro_f1_mean": (
                        float(rr.get("champion_macro_f1_mean"))
                        if rr.get("champion_macro_f1_mean") is not None
                        else None
                    ),
                    "training_days": float(training_days) if training_days is not None else None,
                    "required_threshold": float(threshold_required) if threshold_required is not None else None,
                    "threshold_day_bucket": threshold_bucket,
                    "reasons": reasons if isinstance(reasons, list) else [],
                }
            )

    room_trends = []
    room_history_points = []
    for room_name, entries in room_history.items():
        if not entries:
            continue
        for point in entries:
            room_history_points.append(
                {
                    "room": room_name,
                    "run_id": point.get("run_id"),
                    "training_date": point.get("training_date"),
                    "sensor_event_time": point.get("sensor_event_time"),
                    "candidate_macro_f1_mean": point.get("candidate_macro_f1_mean"),
                    "candidate_accuracy_mean": point.get("candidate_accuracy_mean"),
                    "required_threshold": point.get("required_threshold"),
                    "training_days": point.get("training_days"),
                    "pass": bool(point.get("pass", False)),
                }
            )
        latest = entries[0]
        previous = entries[1] if len(entries) > 1 else None
        latest_candidate = latest.get("candidate_macro_f1_mean")
        previous_candidate = previous.get("candidate_macro_f1_mean") if previous else None
        latest_candidate_accuracy = latest.get("candidate_accuracy_mean")
        previous_candidate_accuracy = previous.get("candidate_accuracy_mean") if previous else None
        delta = None
        if latest_candidate is not None and previous_candidate is not None:
            delta = float(latest_candidate) - float(previous_candidate)

        evaluated_in_latest_run = bool(
            latest_run_id is not None and latest.get("run_id") is not None and int(latest.get("run_id")) == int(latest_run_id)
        )

        room_trends.append(
            {
                "room": room_name,
                "latest_run_id": latest.get("run_id"),
                "latest_training_date": latest.get("training_date"),
                "evaluated_in_latest_run": evaluated_in_latest_run,
                "latest_pass": bool(latest.get("pass", False)),
                "latest_candidate_macro_f1_mean": latest_candidate,
                "latest_candidate_accuracy_mean": latest_candidate_accuracy,
                "latest_candidate_stability_accuracy_mean": latest.get("candidate_stability_accuracy_mean"),
                "latest_candidate_transition_macro_f1_mean": latest.get("candidate_transition_macro_f1_mean"),
                "previous_candidate_macro_f1_mean": previous_candidate,
                "previous_candidate_accuracy_mean": previous_candidate_accuracy,
                "delta_vs_previous": delta,
                "latest_training_days": latest.get("training_days"),
                "latest_required_threshold": latest.get("required_threshold"),
                "latest_threshold_day_bucket": latest.get("threshold_day_bucket"),
                "latest_champion_version": latest.get("champion_version"),
                "latest_champion_macro_f1_mean": latest.get("champion_macro_f1_mean"),
                "latest_reasons": latest.get("reasons", []),
            }
        )

    room_trends = sorted(room_trends, key=lambda x: x["room"])
    room_history_points = sorted(
        room_history_points,
        key=lambda x: (
            str(x.get("training_date") or ""),
            int(x.get("run_id") or 0),
            str(x.get("room") or ""),
        ),
    )
    latest_delta = next((r for r in room_trends if r.get("delta_vs_previous") is not None), None)
    top_failure_reason = None
    if top_failure_counter:
        top_failure_reason = sorted(top_failure_counter.items(), key=lambda x: (-x[1], x[0]))[0][0]

    wf_pass_rate_pct = None
    if wf_enabled_runs > 0:
        wf_pass_rate_pct = (wf_pass_runs / wf_enabled_runs) * 100.0

    global_required = None
    global_bucket = None
    latest_global_gate = (
        latest_metadata.get("global_gate", {})
        if isinstance(latest_metadata.get("global_gate"), dict)
        else {}
    )
    if isinstance(latest_global_gate, dict):
        global_required = _to_float_or_none(latest_global_gate.get("required"))
    if global_required is None and latest_training_days_global is not None and isinstance(global_schedule, list):
        global_required = _to_float_or_none(
            resolve_scheduled_threshold(global_schedule, float(latest_training_days_global))
        )
    if isinstance(global_schedule, list):
        global_bucket = _resolve_schedule_day_bucket(global_schedule, latest_training_days_global)

    release_tracker_rows = []
    release_tracker_rows.append(
        {
            "Scope": "Global",
            "Training Days": latest_training_days_global,
            "Required Threshold": global_required,
            "Day Bucket": global_bucket,
        }
    )
    for room_key in sorted(room_policy_map.keys()):
        if not isinstance(room_policy_map.get(room_key), dict):
            continue
        room_schedule = room_policy_map.get(room_key, {}).get("schedule", [])
        room_days = latest_training_days_by_room.get(_normalize_room_key(room_key))
        room_required = (
            _to_float_or_none(resolve_scheduled_threshold(room_schedule, float(room_days)))
            if room_days is not None and isinstance(room_schedule, list)
            else None
        )
        room_bucket = (
            _resolve_schedule_day_bucket(room_schedule, room_days)
            if isinstance(room_schedule, list)
            else None
        )
        release_tracker_rows.append(
            {
                "Scope": str(room_key).title(),
                "Training Days": room_days,
                "Required Threshold": room_required,
                "Day Bucket": room_bucket,
            }
        )

    return {
        "total_runs": int(total_runs),
        "wf_enabled_runs": int(wf_enabled_runs),
        "wf_pass_runs": int(wf_pass_runs),
        "wf_fail_runs": int(wf_fail_runs),
        "wf_pass_rate_pct": wf_pass_rate_pct,
        "top_failure_reason": top_failure_reason,
        "latest_delta": latest_delta,
        "latest_run": latest_run,
        "room_trends": room_trends,
        "room_history_points": room_history_points,
        "metric_labels": {
            "latest_candidate_macro_f1_mean": "WF Candidate F1",
            "latest_candidate_accuracy_mean": "WF Candidate Accuracy",
            "latest_candidate_stability_accuracy_mean": "Stability Score",
            "latest_candidate_transition_macro_f1_mean": "Transition Score",
            "latest_champion_macro_f1_mean": "Champion WF F1",
            "latest_run_accuracy": "Raw Training Run Accuracy",
        },
        "release_gate_profile": {
            "evidence_profile": str(os.getenv("RELEASE_GATE_EVIDENCE_PROFILE", "production")).strip() or "production",
            "bootstrap_enabled": coerce_bool(os.getenv("RELEASE_GATE_BOOTSTRAP_ENABLED", False)),
            "bootstrap_phase1_max_days": _env_int("RELEASE_GATE_BOOTSTRAP_PHASE1_MAX_DAYS", 7),
            "bootstrap_max_days": _env_int("RELEASE_GATE_BOOTSTRAP_MAX_TRAINING_DAYS", 14),
            "strict_prior_drift_max": _env_float("RELEASE_GATE_MAX_CLASS_PRIOR_DRIFT", 0.10),
            "wf_min_train_days": _env_int("WF_MIN_TRAIN_DAYS", 7),
            "wf_valid_days": _env_int("WF_VALID_DAYS", 1),
            "release_threshold_tracker": release_tracker_rows,
        },
        "production_counters": production_counters,
    }


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
        target_days = _env_int("UI_SAMPLE_COLLECTION_TARGET_DAYS", 14)
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
            "count_label": "Days Recorded",
            "target_label": "Labeled Day Target",
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
