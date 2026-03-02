#!/usr/bin/env python3
"""
Simple Health API Server for ElderlyCarePlatform.

This provides REST endpoints for health checks that can be consumed by:
- Load balancers
- Container orchestrators (Docker, Kubernetes)
- Monitoring systems (Prometheus, DataDog)

Usage:
    python health_server.py  # Runs on port 8504 by default
    
    # Or specify port:
    python health_server.py --port 8550
    
Endpoints:
    GET /health          - Liveness check (is process running?)
    GET /health/ready    - Readiness check (is system ready to serve?)
    GET /health/deep     - Deep health check (all components)
"""

import argparse
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import sys
from urllib.parse import urlparse, parse_qs
import os
import numpy as np
import pandas as pd

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from elderlycare_v1_16.config.settings import DB_PATH, BACKEND_DIR as SETTINGS_BACKEND_DIR, ARCHIVE_DATA_DIR
from ml.pipeline import UnifiedPipeline
from ml.evaluation import TimeCheckpointedSplitter, evaluate_model, load_room_training_dataframe
from config import get_release_gates_config
from ml.beta6.evaluation.evaluation_metrics import (
    build_room_status,
    derive_room_confidence,
    parse_room_override_map,
    resolve_float_threshold_with_source,
    resolve_int_threshold_with_source,
)
from ml.policy_defaults import (
    get_runtime_wf_min_minority_support_by_room,
    get_runtime_wf_min_minority_support_default,
)
try:
    from backend.utils.health_check import HealthChecker
    from backend.utils.data_loader import load_sensor_data
    from backend.utils.room_utils import normalize_room_name
except Exception:
    from utils.health_check import HealthChecker
    from utils.data_loader import load_sensor_data
    from utils.room_utils import normalize_room_name
try:
    from ml.beta6.registry.registry_v2 import RegistryV2
except Exception:  # pragma: no cover - optional runtime dependency in health checks
    RegistryV2 = None
try:
    from backend.db.legacy_adapter import LegacyDatabaseAdapter
except ImportError:
    from elderlycare_v1_16.database import db as adapter
else:
    adapter = LegacyDatabaseAdapter()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize health checker
MODELS_DIR = SETTINGS_BACKEND_DIR / "models"
health_checker = HealthChecker(
    db_path=DB_PATH,
    models_dir=MODELS_DIR,
    check_postgresql=True
)


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health endpoints."""
    
    def _send_json(self, data: dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def _send_text(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8"):
        """Send plain-text response."""
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == '/health':
            # Liveness check
            self._send_json(health_checker.check_liveness())
            
        elif path == '/health/ready':
            # Readiness check
            result = health_checker.check_readiness()
            status = 200 if result['ready'] else 503
            self._send_json(result, status)
            
        elif path == '/health/deep':
            # Deep health check
            result = health_checker.check_all().to_dict()
            status = 200 if result['status'] == 'healthy' else 503
            self._send_json(result, status)

        elif path == '/health/model/walk-forward':
            result, status = self._evaluate_walk_forward(query)
            self._send_json(result, status)

        elif path == '/metrics/model/walk-forward':
            result, status = self._evaluate_walk_forward(query)
            body = render_walk_forward_prometheus_metrics(result)
            # Expose metrics even for degraded/unhealthy report so scrapers retain signal.
            self._send_text(body, 200, "text/plain; version=0.0.4; charset=utf-8")

        elif path == '/health/model/promotion-gates':
            result, status = self._evaluate_promotion_gates(query)
            self._send_json(result, status)

        elif path == '/health/model/ml-snapshot':
            result, status = self._evaluate_ml_snapshot(query)
            self._send_json(result, status)

        elif path == '/metrics/model/promotion-gates':
            result, status = self._evaluate_promotion_gates(query)
            body = render_promotion_gate_prometheus_metrics(result)
            self._send_text(body, 200, "text/plain; version=0.0.4; charset=utf-8")
            
        elif path == '/':
            # Root - show available endpoints
            self._send_json({
                "service": "ElderlyCarePlatform Health API",
                "version": HealthChecker.VERSION,
                "endpoints": {
                    "/health": "Liveness check",
                    "/health/ready": "Readiness check",
                    "/health/deep": "Deep health check (all components)",
                    "/health/model/walk-forward": "Walk-forward model health report (JSON)",
                    "/metrics/model/walk-forward": "Walk-forward model health report (Prometheus text)",
                    "/health/model/promotion-gates": "Recent training gate outcomes summary (JSON)",
                    "/health/model/ml-snapshot": "Aggregate ML health snapshot for Ops and Admin (JSON)",
                    "/metrics/model/promotion-gates": "Recent training gate outcomes summary (Prometheus text)",
                }
            })
        else:
            self.send_error(404, "Not Found")

    def _evaluate_walk_forward(self, query: dict) -> tuple[dict, int]:
        def _get_str(name: str, default: str = "") -> str:
            vals = query.get(name)
            return vals[0] if vals else default

        def _get_int(name: str, default: int) -> int:
            raw = _get_str(name, str(default))
            try:
                return int(raw)
            except ValueError:
                return int(default)

        elder_id = _get_str("elder_id", "").strip()
        room_param = _get_str("room", "").strip()
        if not elder_id:
            return {
                "status": "error",
                "error": "Missing required query parameter: elder_id",
                "example": "/health/model/walk-forward?elder_id=<id>&room=<optional>",
            }, 400

        lookback_days = max(7, _get_int("lookback_days", 90))
        min_train_days = max(2, _get_int("min_train_days", 7))
        valid_days = max(1, _get_int("valid_days", 1))
        step_days = max(1, _get_int("step_days", 1))
        max_folds = max(1, _get_int("max_folds", 30))
        try:
            drift_threshold = float(_get_str("drift_threshold", "0.60"))
        except ValueError:
            drift_threshold = 0.60

        return build_walk_forward_report(
            elder_id=elder_id,
            room_param=room_param,
            lookback_days=lookback_days,
            min_train_days=min_train_days,
            valid_days=valid_days,
            step_days=step_days,
            max_folds=max_folds,
            drift_threshold=drift_threshold,
        )

    def _evaluate_promotion_gates(self, query: dict) -> tuple[dict, int]:
        vals = query.get("elder_id")
        elder_id = vals[0].strip() if vals else ""
        if not elder_id:
            return {
                "status": "error",
                "error": "Missing required query parameter: elder_id",
                "example": "/health/model/promotion-gates?elder_id=<id>&limit=<optional>",
            }, 400

        limit_vals = query.get("limit")
        if limit_vals:
            try:
                limit = int(limit_vals[0])
            except ValueError:
                limit = 20
        else:
            limit = 20
        limit = max(1, min(limit, 200))
        return build_promotion_gate_report(elder_id=elder_id, limit=limit)

    def _evaluate_ml_snapshot(self, query: dict) -> tuple[dict, int]:
        def _get_str(name: str, default: str = "") -> str:
            vals = query.get(name)
            return vals[0] if vals else default

        def _get_int(name: str, default: int) -> int:
            raw = _get_str(name, str(default))
            try:
                return int(raw)
            except ValueError:
                return int(default)

        elder_id = _get_str("elder_id", "").strip()
        if not elder_id:
            return {
                "status": "error",
                "error": "Missing required query parameter: elder_id",
                "example": "/health/model/ml-snapshot?elder_id=<id>&room=<optional>&lookback_runs=<optional>",
            }, 400

        room = _get_str("room", "").strip()
        lookback_runs = max(1, min(_get_int("lookback_runs", 20), 200))
        include_raw_val = _get_str("include_raw", "false").strip().lower()
        include_raw = include_raw_val in {"1", "true", "yes", "y"}
        return build_ml_snapshot_report(
            elder_id=elder_id,
            room=room,
            lookback_runs=lookback_runs,
            include_raw=include_raw,
        )

    def log_message(self, format, *args):
        """Override to use Python logging."""
        logger.info("%s - %s", self.address_string(), format % args)


def build_walk_forward_report(
    elder_id: str,
    room_param: str,
    lookback_days: int,
    min_train_days: int,
    valid_days: int,
    step_days: int,
    max_folds: int,
    drift_threshold: float,
) -> tuple[dict, int]:
    try:
        pipeline = UnifiedPipeline(enable_denoising=True)
        loaded_rooms = pipeline.registry.load_models_for_elder(elder_id, pipeline.platform)
        if not loaded_rooms:
            return {
                "status": "error",
                "elder_id": elder_id,
                "error": "No models could be loaded for elder_id.",
            }, 503

        if room_param:
            target_rooms = [
                r for r in loaded_rooms if normalize_room_name(r) == normalize_room_name(room_param)
            ]
        else:
            target_rooms = list(loaded_rooms)

        if not target_rooms:
            return {
                "status": "error",
                "elder_id": elder_id,
                "error": f"Requested room not found among loaded models: {room_param}",
            }, 404

        room_reports = {}
        monitoring_metrics = []
        any_drift = False
        any_error = False

        for room_name in target_rooms:
            room_df, data_err = load_room_training_dataframe(
                elder_id=elder_id,
                room_name=room_name,
                archive_dir=ARCHIVE_DATA_DIR,
                load_sensor_data_fn=load_sensor_data,
                normalize_room_name_fn=normalize_room_name,
                lookback_days=lookback_days,
            )
            if data_err:
                any_error = True
                room_reports[room_name] = {"status": "error", "error": data_err}
                continue

            processed = pipeline.platform.preprocess_with_resampling(
                room_df,
                room_name,
                # Health check evaluates already-trained models; do not run training-mode preprocessing.
                is_training=False,
                apply_denoising=False,
            )

            encoder = pipeline.platform.label_encoders.get(room_name)
            if "activity_encoded" in processed.columns:
                labels = processed["activity_encoded"].values.astype(np.int32)
            else:
                if encoder is None:
                    any_error = True
                    room_reports[room_name] = {
                        "status": "error",
                        "error": "Missing label encoder for room during walk-forward evaluation.",
                    }
                    continue
                if "activity" not in processed.columns:
                    any_error = True
                    room_reports[room_name] = {
                        "status": "error",
                        "error": "Missing activity labels after preprocessing.",
                    }
                    continue
                classes = [str(c).strip().lower() for c in getattr(encoder, "classes_", [])]
                if not classes:
                    any_error = True
                    room_reports[room_name] = {
                        "status": "error",
                        "error": "Label encoder has no classes for room.",
                    }
                    continue
                label_map = {label: idx for idx, label in enumerate(classes)}
                mapped = (
                    processed["activity"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map(label_map)
                    .fillna(-1)
                    .astype(int)
                )
                valid_mask = mapped >= 0
                if not bool(valid_mask.any()):
                    any_error = True
                    room_reports[room_name] = {
                        "status": "error",
                        "error": "No labels in evaluation data overlap model label encoder.",
                    }
                    continue
                if not bool(valid_mask.all()):
                    processed = processed.loc[valid_mask].reset_index(drop=True)
                    mapped = mapped.loc[valid_mask].reset_index(drop=True)
                labels = mapped.to_numpy(dtype=np.int32)

            seq_length = pipeline.room_config.calculate_seq_length(room_name)
            if len(processed) <= seq_length:
                any_error = True
                room_reports[room_name] = {
                    "status": "error",
                    "error": f"Insufficient rows for sequenceing: {len(processed)} <= {seq_length}",
                }
                continue

            sensor_data = np.asarray(processed[pipeline.platform.sensor_columns].values, dtype=np.float32)
            created_sequences = pipeline.platform.create_sequences(sensor_data, seq_length)
            if isinstance(created_sequences, tuple):
                X_seq = np.asarray(created_sequences[0])
                if len(created_sequences) > 1 and created_sequences[1] is not None:
                    y_seq = np.asarray(created_sequences[1], dtype=np.int32)
                else:
                    y_seq = labels[seq_length - 1:].astype(np.int32)
            else:
                X_seq = np.asarray(created_sequences)
                y_seq = labels[seq_length - 1:].astype(np.int32)
            if len(y_seq) != len(X_seq):
                y_seq = labels[seq_length - 1:].astype(np.int32)[:len(X_seq)]

            seq_timestamps = np.asarray(
                pd.to_datetime(processed["timestamp"]).iloc[seq_length - 1:],
                dtype="datetime64[ns]",
            )
            if len(seq_timestamps) > len(X_seq):
                seq_timestamps = seq_timestamps[:len(X_seq)]

            splitter = TimeCheckpointedSplitter(
                min_train_days=min_train_days,
                valid_days=valid_days,
                step_days=step_days,
                max_folds=max_folds,
            )
            label_ids = (
                list(range(len(getattr(encoder, "classes_", []))))
                if encoder is not None
                else None
            )
            report = evaluate_model(
                model=pipeline.platform.room_models[room_name],
                X_seq=X_seq,
                y_seq=y_seq,
                seq_timestamps=seq_timestamps,
                splitter=splitter,
                class_thresholds=pipeline.platform.class_thresholds.get(room_name, {}),
                labels=label_ids,
            )
            summary = report.get("summary", {})
            folds = report.get("folds", [])
            low_folds = [f for f in folds if float(f.get("macro_f1", 0.0)) < drift_threshold]
            room_drift = len(low_folds) > 0
            any_drift = any_drift or room_drift

            room_reports[room_name] = {
                "status": "ok",
                "summary": summary,
                "drift": {
                    "threshold": drift_threshold,
                    "num_low_folds": len(low_folds),
                    "drift_detected": room_drift,
                },
                "folds": folds,
            }
            monitoring_metrics.append(
                {
                    "room": room_name,
                    "macro_f1_mean": float(summary.get("macro_f1_mean") or 0.0),
                    "accuracy_mean": float(summary.get("accuracy_mean") or 0.0),
                    "drift_detected": room_drift,
                    "num_folds": int(summary.get("num_folds") or 0),
                }
            )

        overall_status = "healthy"
        http_status = 200
        if any_error:
            overall_status = "unhealthy"
            http_status = 503
        elif any_drift:
            overall_status = "degraded"
            http_status = 503

        return (
            {
                "status": overall_status,
                "elder_id": elder_id,
                "rooms_evaluated": target_rooms,
                "config": {
                    "lookback_days": lookback_days,
                    "min_train_days": min_train_days,
                    "valid_days": valid_days,
                    "step_days": step_days,
                    "max_folds": max_folds,
                    "drift_threshold": drift_threshold,
                },
                "rooms": room_reports,
                "monitoring_metrics": {
                    "rooms": monitoring_metrics,
                    "rooms_with_drift": sum(1 for m in monitoring_metrics if m["drift_detected"]),
                    "total_rooms": len(monitoring_metrics),
                },
                "timestamp": pd.Timestamp.utcnow().isoformat(),
            },
            http_status,
        )
    except Exception as e:
        logger.exception("Walk-forward model health evaluation failed.")
        return {
            "status": "error",
            "elder_id": elder_id,
            "error": str(e),
        }, 503


def render_walk_forward_prometheus_metrics(report: dict) -> str:
    """
    Render walk-forward report JSON as Prometheus exposition format (0.0.4).
    """
    def _esc(value: str) -> str:
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    status_map = {
        "healthy": 0,
        "degraded": 1,
        "unhealthy": 2,
        "error": 3,
    }
    elder_id = str(report.get("elder_id", "unknown"))
    overall_status = str(report.get("status", "error")).lower()
    monitoring = report.get("monitoring_metrics", {}) or {}
    rooms = monitoring.get("rooms", []) or []
    rooms_with_drift = int(monitoring.get("rooms_with_drift", 0) or 0)
    total_rooms = int(monitoring.get("total_rooms", len(rooms)) or 0)

    lines = [
        "# HELP beta_walk_forward_status Overall walk-forward health status (0=healthy,1=degraded,2=unhealthy,3=error)",
        "# TYPE beta_walk_forward_status gauge",
        f'beta_walk_forward_status{{elder_id="{_esc(elder_id)}"}} {status_map.get(overall_status, 3)}',
        "# HELP beta_walk_forward_rooms_total Number of rooms evaluated in latest walk-forward run",
        "# TYPE beta_walk_forward_rooms_total gauge",
        f'beta_walk_forward_rooms_total{{elder_id="{_esc(elder_id)}"}} {total_rooms}',
        "# HELP beta_walk_forward_rooms_with_drift Number of rooms with drift detected",
        "# TYPE beta_walk_forward_rooms_with_drift gauge",
        f'beta_walk_forward_rooms_with_drift{{elder_id="{_esc(elder_id)}"}} {rooms_with_drift}',
        "# HELP beta_walk_forward_room_macro_f1_mean Mean holdout macro-F1 per room",
        "# TYPE beta_walk_forward_room_macro_f1_mean gauge",
        "# HELP beta_walk_forward_room_accuracy_mean Mean holdout accuracy per room",
        "# TYPE beta_walk_forward_room_accuracy_mean gauge",
        "# HELP beta_walk_forward_room_drift_detected Drift detected flag per room (0/1)",
        "# TYPE beta_walk_forward_room_drift_detected gauge",
        "# HELP beta_walk_forward_room_num_folds Number of folds evaluated per room",
        "# TYPE beta_walk_forward_room_num_folds gauge",
    ]

    for row in rooms:
        room = _esc(row.get("room", "unknown"))
        macro_f1 = float(row.get("macro_f1_mean", 0.0) or 0.0)
        accuracy = float(row.get("accuracy_mean", 0.0) or 0.0)
        drift = 1 if bool(row.get("drift_detected", False)) else 0
        num_folds = int(row.get("num_folds", 0) or 0)
        labels = f'elder_id="{_esc(elder_id)}",room="{room}"'
        lines.append(f"beta_walk_forward_room_macro_f1_mean{{{labels}}} {macro_f1}")
        lines.append(f"beta_walk_forward_room_accuracy_mean{{{labels}}} {accuracy}")
        lines.append(f"beta_walk_forward_room_drift_detected{{{labels}}} {drift}")
        lines.append(f"beta_walk_forward_room_num_folds{{{labels}}} {num_folds}")

    return "\n".join(lines) + "\n"


def _safe_json_load(value: object) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _row_get(row: object, key: str, default: object = None) -> object:
    # sqlite rows can be tuple or mapping; psycopg rows are often tuples.
    if isinstance(row, dict):
        return row.get(key, default)
    if hasattr(row, "keys") and callable(getattr(row, "keys", None)):
        try:
            return row[key]
        except Exception:
            pass
    return default


def _normalize_fetched_rows(rows: list, cursor: object) -> list:
    """
    Normalize DB result rows to dicts when cursor returns tuples.
    """
    if not rows:
        return rows
    first = rows[0]
    if isinstance(first, dict):
        return rows
    if hasattr(first, "keys") and callable(getattr(first, "keys", None)):
        return rows

    description = getattr(cursor, "description", None)
    if not description:
        return rows
    col_names = [str(col[0]) for col in description if col and len(col) > 0]
    if not col_names:
        return rows

    normalized = []
    for row in rows:
        if isinstance(row, (tuple, list)) and len(row) == len(col_names):
            normalized.append({col_names[i]: row[i] for i in range(len(col_names))})
        else:
            normalized.append(row)
    return normalized


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _utc_iso(value: object) -> str | None:
    if value in (None, ""):
        return None
    try:
        ts = pd.to_datetime(value, utc=True)
        if pd.isna(ts):
            return None
        return ts.isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def _hours_since(value: object) -> float | None:
    iso = _utc_iso(value)
    if iso is None:
        return None
    try:
        then = pd.Timestamp(iso)
        now = pd.Timestamp.utcnow()
        now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
        return float((now - then).total_seconds() / 3600.0)
    except Exception:
        return None


def _parse_room_override_map(raw: str) -> dict[str, str]:
    return parse_room_override_map(raw, normalize_room_name_fn=normalize_room_name)


def _resolve_float_threshold_with_source(
    var_name: str,
    room_name: str,
    fallback_default: float,
    explicit_value: object | None = None,
) -> dict:
    return resolve_float_threshold_with_source(
        var_name=var_name,
        room_name=room_name,
        fallback_default=float(fallback_default),
        normalize_room_name_fn=normalize_room_name,
        explicit_value=explicit_value,
        env=os.environ,
    )


def _resolve_int_threshold_with_source(
    var_name: str,
    room_name: str,
    fallback_default: int,
    explicit_value: object | None = None,
) -> dict:
    return resolve_int_threshold_with_source(
        var_name=var_name,
        room_name=room_name,
        fallback_default=int(fallback_default),
        normalize_room_name_fn=normalize_room_name,
        explicit_value=explicit_value,
        env=os.environ,
    )


def _resolve_global_release_threshold(latest_metadata: dict) -> dict:
    training_days = _safe_float((latest_metadata or {}).get("global_gate", {}).get("training_days"))
    required = _safe_float((latest_metadata or {}).get("global_gate", {}).get("required"))
    bucket = None

    try:
        policy = get_release_gates_config()
    except Exception:
        return {
            "value": required,
            "source": "unknown",
            "policy_file": "backend/config/release_gates.json",
            "effective_day_bucket": None,
        }

    schedule = (
        policy.get("release_gates", {})
        .get("global", {})
        .get("schedule", [])
    )
    if required is None and training_days is not None:
        for entry in schedule:
            if not isinstance(entry, dict):
                continue
            min_days = _safe_float(entry.get("min_days")) or 0.0
            max_days_raw = entry.get("max_days")
            max_days = _safe_float(max_days_raw)
            in_bucket = training_days >= min_days and (max_days is None or training_days <= max_days)
            if in_bucket:
                required = _safe_float(entry.get("min_value"))
                if max_days is None:
                    bucket = f"day_{int(min_days)}_plus"
                else:
                    bucket = f"day_{int(min_days)}_{int(max_days)}"
                break
    elif training_days is not None:
        for entry in schedule:
            if not isinstance(entry, dict):
                continue
            min_days = _safe_float(entry.get("min_days")) or 0.0
            max_days_raw = entry.get("max_days")
            max_days = _safe_float(max_days_raw)
            in_bucket = training_days >= min_days and (max_days is None or training_days <= max_days)
            if in_bucket:
                if max_days is None:
                    bucket = f"day_{int(min_days)}_plus"
                else:
                    bucket = f"day_{int(min_days)}_{int(max_days)}"
                break

    return {
        "value": required,
        "source": "policy" if required is not None else "unknown",
        "policy_file": "backend/config/release_gates.json",
        "effective_day_bucket": bucket,
    }


def _derive_room_confidence(fold_count: int | None, low_transition_folds: int | None) -> str:
    return derive_room_confidence(fold_count, low_transition_folds)


def _build_room_status(room_payload: dict, global_release_threshold: float | None) -> tuple[str, str, str]:
    return build_room_status(
        room_payload=room_payload,
        global_release_threshold=global_release_threshold,
        hours_since_fn=_hours_since,
    )


def _is_insufficient_evidence_reason_code(reason_code: str | None) -> bool:
    code = str(reason_code or "").strip().lower()
    return code.startswith("insufficient_evidence")


def _mask_insufficient_evidence_metrics(room_payload: dict) -> None:
    """
    Prevent low-support runs from being interpreted as trustworthy quality metrics.
    """
    metrics = room_payload.get("metrics")
    if not isinstance(metrics, dict):
        return
    for key in (
        "candidate_macro_f1_mean",
        "candidate_transition_macro_f1_mean",
        "candidate_stability_accuracy_mean",
    ):
        if metrics.get(key) is None:
            continue
        metrics[f"{key}_raw"] = metrics.get(key)
        metrics[key] = None


def _resolve_beta6_registry_v2_root() -> Path:
    raw = str(os.getenv("BETA6_REGISTRY_V2_ROOT", "")).strip()
    if raw:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (BACKEND_DIR.parent / candidate).resolve()
        return candidate
    return (BACKEND_DIR / "models_beta6_registry_v2").resolve()


def _read_runtime_fallback_by_room(
    *,
    elder_id: str,
    room_candidates: dict[str, set[str]],
) -> dict[str, bool]:
    """
    Read runtime fallback state from RegistryV2 and return normalized-room fallback flags.
    Runtime registry state takes precedence over metadata-derived fallback summaries.
    """
    if RegistryV2 is None:
        return {}
    elder_id = str(elder_id or "").strip()
    if not elder_id:
        return {}

    candidates: dict[str, set[str]] = {}
    for key, values in (room_candidates or {}).items():
        room_key = normalize_room_name(str(key or ""))
        if not room_key:
            continue
        bucket = candidates.setdefault(room_key, set())
        bucket.add(room_key)
        if isinstance(values, set):
            iter_values = values
        elif isinstance(values, (list, tuple)):
            iter_values = set(values)
        else:
            iter_values = {values} if values is not None else set()
        for alias in iter_values:
            alias_txt = str(alias or "").strip()
            if alias_txt:
                bucket.add(alias_txt)

    root = _resolve_beta6_registry_v2_root()
    elder_dir = root / elder_id
    if elder_dir.exists() and elder_dir.is_dir():
        try:
            for child in elder_dir.iterdir():
                if not child.is_dir() or child.name == "_runtime":
                    continue
                room_key = normalize_room_name(child.name)
                if not room_key:
                    continue
                bucket = candidates.setdefault(room_key, set())
                bucket.add(room_key)
                bucket.add(child.name)
        except Exception as e:
            logger.warning("ML snapshot runtime fallback room discovery failed for %s: %s", elder_id, e)

    if not candidates:
        return {}

    try:
        registry_v2 = RegistryV2(root=root)
    except Exception as e:
        logger.warning("ML snapshot runtime fallback registry init failed: %s", e)
        return {}

    fallback_by_room: dict[str, bool] = {}
    for room_key, aliases in candidates.items():
        ordered_aliases: list[str] = []
        seen_aliases: set[str] = set()
        for alias in sorted(aliases, key=lambda v: str(v)):
            alias_txt = str(alias or "").strip()
            if not alias_txt or alias_txt in seen_aliases:
                continue
            ordered_aliases.append(alias_txt)
            seen_aliases.add(alias_txt)
        if room_key not in seen_aliases:
            ordered_aliases.append(room_key)

        state_found = False
        active = False
        for alias in ordered_aliases:
            try:
                state = registry_v2.read_fallback_state(elder_id=elder_id, room=alias)
            except Exception:
                state = None
            if not isinstance(state, dict):
                continue
            state_found = True
            if bool(state.get("active", False)):
                active = True
                break
        if state_found:
            fallback_by_room[room_key] = bool(active)

    return fallback_by_room


def _derive_room_reports_from_metadata_metrics(metadata: dict) -> list[dict]:
    """
    Fallback when walk_forward_gate.room_reports is missing.

    Use metadata.metrics (written by run_daily_analysis) to recover per-room
    candidate signals so Ops Dashboard does not degrade to N/A for every room.
    """
    if not isinstance(metadata, dict):
        return []

    metric_rows = metadata.get("metrics")
    if not isinstance(metric_rows, list):
        return []

    derived: list[dict] = []
    for item in metric_rows:
        if not isinstance(item, dict):
            continue
        room_name = str(item.get("room") or "").strip()
        if not room_name:
            continue
        macro_f1 = _safe_float(item.get("macro_f1"))
        # Fallback to accuracy only when macro_f1 is unavailable.
        if macro_f1 is None:
            macro_f1 = _safe_float(item.get("accuracy"))
        candidate_summary = {
            "macro_f1_mean": macro_f1,
            "accuracy_mean": _safe_float(item.get("accuracy")),
            "num_folds": _safe_int(item.get("num_folds")) or 0,
            "stability_accuracy_mean": _safe_float(item.get("candidate_stability_accuracy_mean")),
            "transition_macro_f1_mean": _safe_float(item.get("candidate_transition_macro_f1_mean")),
        }
        derived.append(
            {
                "room": room_name,
                "pass": _safe_bool(item.get("gate_pass", False)),
                "reasons": item.get("gate_reasons", []) if isinstance(item.get("gate_reasons"), list) else [],
                "candidate_summary": candidate_summary,
                "candidate_low_folds": _safe_int(item.get("candidate_low_folds")) or 0,
                "candidate_low_support_folds": _safe_int(item.get("candidate_low_support_folds")) or 0,
                "candidate_low_transition_folds": _safe_int(item.get("candidate_low_transition_folds")) or 0,
                "candidate_transition_supported_folds": _safe_int(item.get("candidate_transition_supported_folds")),
                "candidate_stability_accuracy_mean": _safe_float(item.get("candidate_stability_accuracy_mean")),
                "candidate_transition_macro_f1_mean": _safe_float(item.get("candidate_transition_macro_f1_mean")),
                "champion_macro_f1_mean": _safe_float(item.get("champion_macro_f1_mean")),
                "candidate_wf_config": item.get("candidate_wf_config", {})
                if isinstance(item.get("candidate_wf_config"), dict)
                else {},
            }
        )
    return derived


def _extract_room_reports_from_metadata(metadata: dict) -> list[dict]:
    if not isinstance(metadata, dict):
        return []
    walk_forward_gate = metadata.get("walk_forward_gate", {})
    walk_forward_gate = walk_forward_gate if isinstance(walk_forward_gate, dict) else {}
    room_reports = walk_forward_gate.get("room_reports", [])
    if (not isinstance(room_reports, list) or len(room_reports) == 0) and isinstance(metadata, dict):
        room_reports = _derive_room_reports_from_metadata_metrics(metadata)
    return room_reports if isinstance(room_reports, list) else []


def build_ml_snapshot_report(
    elder_id: str,
    room: str = "",
    lookback_runs: int = 20,
    include_raw: bool = False,
) -> tuple[dict, int]:
    """
    Build a read-only aggregate ML health snapshot for dashboard panels.
    """
    elder_id = str(elder_id or "").strip()
    room_param = normalize_room_name(room or "")
    lookback_runs = max(1, min(int(lookback_runs), 200))
    if not elder_id:
        return {
            "status": "error",
            "error": "Missing elder_id",
        }, 400

    try:
        with adapter.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT training_date, status, accuracy, metadata
                FROM training_history
                WHERE elder_id = ?
                ORDER BY training_date DESC
                LIMIT ?
                """,
                (elder_id, lookback_runs),
            )
            rows = cursor.fetchall()
            rows = _normalize_fetched_rows(rows, cursor)
    except Exception as e:
        logger.exception("ML snapshot query failed.")
        return {
            "status": "error",
            "elder_id": elder_id,
            "error": str(e),
        }, 503

    generated_at = _utc_iso(pd.Timestamp.utcnow())
    if not rows:
        return {
            "elder_id": elder_id,
            "generated_at": generated_at,
            "status": {
                "overall": "not_available",
                "reason": "No recent training history found.",
                "reason_code": "no_training_history",
                "data_freshness_hours": None,
            },
            "thresholds": {
                "global_release_macro_f1_threshold": {
                    "value": None,
                    "source": "unknown",
                    "policy_file": "backend/config/release_gates.json",
                    "effective_day_bucket": None,
                }
            },
            "window": {
                "lookback_runs": int(lookback_runs),
                "lookback_days": None,
            },
            "rooms": [],
            "raw": [] if include_raw else None,
        }, 200

    latest_metadata = _safe_json_load(_row_get(rows[0], "metadata"))
    latest_wf_gate = (
        latest_metadata.get("walk_forward_gate", {})
        if isinstance(latest_metadata.get("walk_forward_gate", {}), dict)
        else {}
    )
    latest_wf_config = (
        latest_wf_gate.get("config", {})
        if isinstance(latest_wf_gate.get("config", {}), dict)
        else {}
    )
    global_threshold = _resolve_global_release_threshold(latest_metadata)
    global_threshold_value = _safe_float(global_threshold.get("value"))

    fallback_by_room: dict[str, bool] = {}
    raw_items: list[dict] = []
    room_payloads: dict[str, dict] = {}
    runtime_room_candidates: dict[str, set[str]] = {}
    previous_macro_by_room: dict[str, float] = {}
    wf_minority_support_default = max(0, int(get_runtime_wf_min_minority_support_default()))
    wf_minority_support_by_room = get_runtime_wf_min_minority_support_by_room()

    def _register_runtime_room_candidate(raw_room: object) -> None:
        raw_text = str(raw_room or "").strip()
        room_key = normalize_room_name(raw_text)
        if not room_key:
            return
        bucket = runtime_room_candidates.setdefault(room_key, set())
        bucket.add(room_key)
        bucket.add(raw_text)

    if room_param:
        _register_runtime_room_candidate(room_param)

    # Recover previous-run macro F1 as a baseline when champion metric is unavailable.
    for row in rows[1:]:
        metadata = _safe_json_load(_row_get(row, "metadata"))
        room_reports = _extract_room_reports_from_metadata(metadata)
        for rr in room_reports:
            if not isinstance(rr, dict):
                continue
            room_name = str(rr.get("room") or "").strip()
            room_key = normalize_room_name(room_name)
            if not room_key:
                continue
            if room_key in previous_macro_by_room:
                continue
            candidate_summary = rr.get("candidate_summary", {})
            candidate_summary = candidate_summary if isinstance(candidate_summary, dict) else {}
            macro_f1 = _safe_float(candidate_summary.get("macro_f1_mean"))
            if macro_f1 is None:
                macro_f1 = _safe_float(rr.get("candidate_macro_f1_mean"))
            if macro_f1 is not None:
                previous_macro_by_room[room_key] = macro_f1

    for row in rows:
        training_date = _row_get(row, "training_date")
        row_status = str(_row_get(row, "status", "") or "")
        metadata = _safe_json_load(_row_get(row, "metadata"))
        walk_forward_gate = metadata.get("walk_forward_gate", {}) if isinstance(metadata, dict) else {}
        room_reports = _extract_room_reports_from_metadata(metadata)
        fallback = metadata.get("beta6_fallback", {}) if isinstance(metadata, dict) else {}
        activated_list = fallback.get("activated", []) if isinstance(fallback, dict) else []
        for item in activated_list if isinstance(activated_list, list) else []:
            if isinstance(item, str):
                room_key = normalize_room_name(item)
                if room_key:
                    fallback_by_room[room_key] = True
                    _register_runtime_room_candidate(item)
            elif isinstance(item, dict):
                room_name = item.get("room") or item.get("room_name") or ""
                room_key = normalize_room_name(room_name)
                if room_key:
                    fallback_by_room[room_key] = True
                    _register_runtime_room_candidate(room_name)

        if include_raw:
            raw_items.append(
                {
                    "training_date": _utc_iso(training_date),
                    "status": row_status,
                    "walk_forward_gate": walk_forward_gate if isinstance(walk_forward_gate, dict) else {},
                    "global_gate": metadata.get("global_gate", {}) if isinstance(metadata, dict) else {},
                    "beta6_fallback": fallback if isinstance(fallback, dict) else {},
                }
            )

        if not isinstance(room_reports, list):
            continue
        for rr in room_reports:
            if not isinstance(rr, dict):
                continue
            room_name = str(rr.get("room") or "").strip()
            if not room_name:
                continue
            room_key = normalize_room_name(room_name)
            _register_runtime_room_candidate(room_name)
            if room_param and room_key != room_param:
                continue
            if room_key in room_payloads:
                continue

            candidate_summary = rr.get("candidate_summary", {})
            candidate_summary = candidate_summary if isinstance(candidate_summary, dict) else {}
            wf_cfg = rr.get("candidate_wf_config", {})
            wf_cfg = wf_cfg if isinstance(wf_cfg, dict) else {}

            drift_threshold = _resolve_float_threshold_with_source(
                "WF_DRIFT_THRESHOLD",
                room_name,
                fallback_default=0.60,
                explicit_value=wf_cfg.get("drift_threshold"),
            )
            min_transition = _resolve_float_threshold_with_source(
                "WF_MIN_TRANSITION_F1",
                room_name,
                fallback_default=0.80,
                explicit_value=wf_cfg.get("min_transition_f1"),
            )
            min_stability = _resolve_float_threshold_with_source(
                "WF_MIN_STABILITY_ACCURACY",
                room_name,
                fallback_default=0.99,
                explicit_value=wf_cfg.get("min_stability_accuracy"),
            )
            max_transition_low = _resolve_int_threshold_with_source(
                "WF_MAX_TRANSITION_LOW_FOLDS",
                room_name,
                fallback_default=1,
                explicit_value=wf_cfg.get("max_transition_low_folds"),
            )
            policy_minority_support = int(wf_minority_support_by_room.get(room_key, wf_minority_support_default))
            min_minority_support = _resolve_int_threshold_with_source(
                "WF_MIN_MINORITY_SUPPORT",
                room_name,
                fallback_default=policy_minority_support,
                explicit_value=wf_cfg.get("min_minority_support"),
            )

            candidate_macro = _safe_float(candidate_summary.get("macro_f1_mean"))
            champion_macro = _safe_float(rr.get("champion_macro_f1_mean"))
            prev_run_macro = previous_macro_by_room.get(room_key)
            delta_vs_champion = None
            if candidate_macro is not None and champion_macro is not None:
                delta_vs_champion = candidate_macro - champion_macro

            fold_count = _safe_int(candidate_summary.get("num_folds"))
            low_transition_folds = _safe_int(rr.get("candidate_low_transition_folds"))

            payload = {
                "room": room_key,
                "status": "not_available",
                "last_check_time": _utc_iso(training_date),
                "fallback_active": bool(fallback_by_room.get(room_key, False)),
                "confidence": _derive_room_confidence(fold_count, low_transition_folds),
                "metrics": {
                    "candidate_macro_f1_mean": candidate_macro,
                    "champion_macro_f1_mean": champion_macro,
                    "delta_vs_champion_macro_f1": delta_vs_champion,
                    "candidate_transition_macro_f1_mean": (
                        _safe_float(rr.get("candidate_transition_macro_f1_mean"))
                        if rr.get("candidate_transition_macro_f1_mean") is not None
                        else _safe_float(candidate_summary.get("transition_macro_f1_mean"))
                    ),
                    "previous_run_macro_f1_mean": prev_run_macro,
                    "candidate_stability_accuracy_mean": (
                        _safe_float(rr.get("candidate_stability_accuracy_mean"))
                        if rr.get("candidate_stability_accuracy_mean") is not None
                        else _safe_float(candidate_summary.get("stability_accuracy_mean"))
                    ),
                    "candidate_accuracy_mean": _safe_float(candidate_summary.get("accuracy_mean")),
                },
                "thresholds": {
                    "drift_threshold": drift_threshold,
                    "min_transition_f1": min_transition,
                    "min_stability_accuracy": min_stability,
                    "max_transition_low_folds": max_transition_low,
                    "min_minority_support": min_minority_support,
                },
                "support": {
                    "fold_count": fold_count,
                    "transition_supported_folds": _safe_int(rr.get("candidate_transition_supported_folds")),
                    "candidate_low_folds": _safe_int(rr.get("candidate_low_folds")),
                    "candidate_low_support_folds": _safe_int(rr.get("candidate_low_support_folds")),
                    "candidate_low_transition_folds": low_transition_folds,
                    "lookback_days": _safe_int(wf_cfg.get("lookback_days")),
                },
                "gate": {
                    "pass": _safe_bool(rr.get("pass", False)),
                    "reasons": rr.get("reasons", []) if isinstance(rr.get("reasons"), list) else [],
                },
                "_run_status": row_status,
            }
            room_status, room_reason, room_reason_code = _build_room_status(payload, global_threshold_value)
            payload["status"] = room_status
            payload["status_reason"] = room_reason
            payload["status_reason_code"] = room_reason_code
            payload["review_queue_recommended"] = room_reason_code == "routed_review_queue_uncertainty"
            if _is_insufficient_evidence_reason_code(room_reason_code):
                _mask_insufficient_evidence_metrics(payload)
            room_payloads[room_key] = payload

    runtime_fallback_by_room = _read_runtime_fallback_by_room(
        elder_id=elder_id,
        room_candidates=runtime_room_candidates,
    )
    for room_key, active in runtime_fallback_by_room.items():
        fallback_by_room[room_key] = bool(active)
    any_fallback_active = any(bool(v) for v in fallback_by_room.values())

    for room_key, payload in room_payloads.items():
        payload["fallback_active"] = bool(fallback_by_room.get(room_key, False))
        room_status, room_reason, room_reason_code = _build_room_status(payload, global_threshold_value)
        payload["status"] = room_status
        payload["status_reason"] = room_reason
        payload["status_reason_code"] = room_reason_code
        payload["review_queue_recommended"] = room_reason_code == "routed_review_queue_uncertainty"
        if _is_insufficient_evidence_reason_code(room_reason_code):
            _mask_insufficient_evidence_metrics(payload)

    if not room_payloads:
        promotion_report, promotion_status = build_promotion_gate_report(elder_id=elder_id, limit=lookback_runs)
        if promotion_status == 200 and isinstance(promotion_report, dict):
            promo_summary = promotion_report.get("summary", {})
            promo_summary = promo_summary if isinstance(promo_summary, dict) else {}
            total_runs = int(promo_summary.get("total_runs", 0) or 0)
            if total_runs > 0:
                latest_global_gate = (
                    latest_metadata.get("global_gate", {})
                    if isinstance(latest_metadata.get("global_gate", {}), dict)
                    else {}
                )
                promo_latest = promotion_report.get("latest_run", {})
                promo_latest = promo_latest if isinstance(promo_latest, dict) else {}
                wf_enabled_runs = int(promo_summary.get("walk_forward_enabled_runs", 0) or 0)
                wf_fail_runs = int(promo_summary.get("walk_forward_fail_runs", 0) or 0)

                fallback_status = "watch"
                fallback_reason = "Recent model runs exist, but room-level walk-forward details are unavailable."
                fallback_reason_code = "missing_room_reports"
                fallback_gate_pass = True

                if any_fallback_active:
                    fallback_status = "action_needed"
                    fallback_reason = "Fallback mode is active for one or more rooms."
                    fallback_reason_code = "fallback_active"
                    fallback_gate_pass = False
                elif latest_global_gate:
                    global_pass = latest_global_gate.get("pass")
                    if global_pass is False:
                        fallback_status = "action_needed"
                        fallback_reason = "Global model quality checks paused updates."
                        fallback_reason_code = "global_gate_failed"
                        fallback_gate_pass = False
                    elif global_pass is True:
                        fallback_status = "watch"
                        fallback_reason = "Global quality checks passed, but room-level details are unavailable."
                        fallback_reason_code = "global_only_signal"
                        fallback_gate_pass = True
                elif wf_enabled_runs > 0:
                    if wf_fail_runs > 0:
                        fallback_status = "action_needed"
                        fallback_reason = "Some recent safety checks failed, but room-level details are unavailable."
                        fallback_reason_code = "wf_fail_without_room_details"
                        fallback_gate_pass = False
                    else:
                        fallback_status = "watch"
                        fallback_reason = "Safety checks passed recently, but room-level details are unavailable."
                        fallback_reason_code = "wf_pass_without_room_details"
                        fallback_gate_pass = True

                synthetic_payload = {
                    "room": "all_rooms",
                    "status": fallback_status,
                    "last_check_time": _utc_iso(_row_get(rows[0], "training_date")),
                    "fallback_active": bool(any_fallback_active),
                    "confidence": "Low",
                    "metrics": {
                        "candidate_macro_f1_mean": _safe_float(latest_global_gate.get("actual_global_macro_f1")),
                        "champion_macro_f1_mean": None,
                        "delta_vs_champion_macro_f1": None,
                        "candidate_transition_macro_f1_mean": None,
                        "candidate_stability_accuracy_mean": None,
                        "candidate_accuracy_mean": _safe_float(promo_latest.get("accuracy")),
                    },
                    "thresholds": {
                        "drift_threshold": _resolve_float_threshold_with_source(
                            "WF_DRIFT_THRESHOLD",
                            "all_rooms",
                            fallback_default=0.60,
                        ),
                        "min_transition_f1": _resolve_float_threshold_with_source(
                            "WF_MIN_TRANSITION_F1",
                            "all_rooms",
                            fallback_default=0.80,
                        ),
                        "min_stability_accuracy": _resolve_float_threshold_with_source(
                            "WF_MIN_STABILITY_ACCURACY",
                            "all_rooms",
                            fallback_default=0.99,
                        ),
                        "max_transition_low_folds": _resolve_int_threshold_with_source(
                            "WF_MAX_TRANSITION_LOW_FOLDS",
                            "all_rooms",
                            fallback_default=1,
                        ),
                        "min_minority_support": _resolve_int_threshold_with_source(
                            "WF_MIN_MINORITY_SUPPORT",
                            "all_rooms",
                            fallback_default=wf_minority_support_default,
                        ),
                    },
                    "support": {
                        "fold_count": None,
                        "transition_supported_folds": None,
                        "candidate_low_folds": None,
                        "candidate_low_support_folds": None,
                        "candidate_low_transition_folds": None,
                        "lookback_days": _safe_int(latest_wf_config.get("lookback_days")),
                    },
                    "gate": {
                        "pass": bool(fallback_gate_pass),
                        "reasons": [],
                    },
                    "_run_status": str(promo_latest.get("status") or ""),
                    "status_reason": fallback_reason,
                    "status_reason_code": fallback_reason_code,
                    "review_queue_recommended": False,
                }
                room_payloads["all_rooms"] = synthetic_payload

    rooms = sorted(
        room_payloads.values(),
        key=lambda x: ({"action_needed": 0, "watch": 1, "healthy": 2, "not_available": 3}.get(x.get("status"), 9), x.get("room", "")),
    )

    freshness_values = [_hours_since(r.get("last_check_time")) for r in rooms]
    freshness_values = [v for v in freshness_values if v is not None]
    freshness_hours = min(freshness_values) if freshness_values else _hours_since(_row_get(rows[0], "training_date"))

    if not rooms:
        overall_status = "not_available"
        overall_reason = "No walk-forward model-health report is available yet."
        overall_reason_code = "no_walk_forward_report"
    else:
        precedence = {"action_needed": 0, "watch": 1, "healthy": 2, "not_available": 3}
        rooms_by_priority = sorted(rooms, key=lambda item: precedence.get(item.get("status"), 9))
        top = rooms_by_priority[0]
        overall_status = top.get("status", "not_available")
        overall_reason = top.get("status_reason", "Model health summary unavailable.")
        overall_reason_code = top.get("status_reason_code", "unknown")

    normalized_rooms = []
    for room_item in rooms:
        cleaned = dict(room_item)
        cleaned.pop("_run_status", None)
        cleaned.pop("status_reason", None)
        cleaned.pop("status_reason_code", None)
        normalized_rooms.append(cleaned)

    return {
        "elder_id": elder_id,
        "generated_at": generated_at,
        "status": {
            "overall": overall_status,
            "reason": overall_reason,
            "reason_code": overall_reason_code,
            "data_freshness_hours": freshness_hours,
        },
        "thresholds": {
            "global_release_macro_f1_threshold": global_threshold,
        },
        "window": {
            "lookback_runs": int(lookback_runs),
            "lookback_days": _safe_int(latest_wf_config.get("lookback_days")),
        },
        "rooms": normalized_rooms,
        "raw": raw_items if include_raw else None,
    }, 200


def build_promotion_gate_report(elder_id: str, limit: int = 20) -> tuple[dict, int]:
    """
    Summarize recent training gate outcomes for team monitoring.
    """
    try:
        with adapter.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT training_date, status, accuracy, metadata
                FROM training_history
                WHERE elder_id = ?
                ORDER BY training_date DESC
                LIMIT ?
                """,
                (elder_id, int(limit)),
            )
            rows = cursor.fetchall()
            rows = _normalize_fetched_rows(rows, cursor)

        if not rows:
            return {
                "status": "error",
                "elder_id": elder_id,
                "error": "No training history found for elder_id.",
                "limit": int(limit),
            }, 404

        total_runs = 0
        wf_enabled_runs = 0
        wf_pass_runs = 0
        wf_fail_runs = 0
        global_gate_fail_runs = 0
        success_runs = 0
        rejected_walk_forward_runs = 0
        accuracies: list[float] = []
        latest_run = None
        room_history: dict[str, list[dict]] = {}

        for row in rows:
            total_runs += 1
            training_date = _row_get(row, "training_date")
            run_status = str(_row_get(row, "status", "") or "")
            accuracy_raw = _row_get(row, "accuracy")
            metadata = _safe_json_load(_row_get(row, "metadata"))

            if latest_run is None:
                latest_run = {
                    "training_date": str(training_date) if training_date is not None else None,
                    "status": run_status,
                    "accuracy": float(accuracy_raw) if accuracy_raw is not None else None,
                    "walk_forward_gate": metadata.get("walk_forward_gate", {}),
                    "global_gate": metadata.get("global_gate", {}),
                }

            if accuracy_raw is not None:
                try:
                    accuracies.append(float(accuracy_raw))
                except (TypeError, ValueError):
                    pass

            if run_status == "success":
                success_runs += 1
            if run_status == "rejected_by_walk_forward_gate":
                rejected_walk_forward_runs += 1
            if run_status == "rejected_by_global_gate":
                global_gate_fail_runs += 1

            wf = metadata.get("walk_forward_gate", {}) if isinstance(metadata, dict) else {}
            if isinstance(wf, dict) and wf.get("reason") not in (None, "disabled"):
                wf_enabled_runs += 1
                if bool(wf.get("pass", False)):
                    wf_pass_runs += 1
                else:
                    wf_fail_runs += 1

            room_reports = wf.get("room_reports", []) if isinstance(wf, dict) else []
            if isinstance(room_reports, list):
                for rr in room_reports:
                    if not isinstance(rr, dict):
                        continue
                    room_name = str(rr.get("room") or "").strip()
                    if not room_name:
                        continue
                    entry = {
                        "training_date": str(training_date) if training_date is not None else None,
                        "pass": bool(rr.get("pass", False)),
                        "candidate_macro_f1_mean": rr.get("candidate_summary", {}).get("macro_f1_mean")
                        if isinstance(rr.get("candidate_summary"), dict)
                        else None,
                        "champion_macro_f1_mean": rr.get("champion_macro_f1_mean"),
                        "reasons": rr.get("reasons", []),
                    }
                    room_history.setdefault(room_name, []).append(entry)

        room_trends = []
        for room_name, entries in room_history.items():
            if not entries:
                continue
            latest = entries[0]
            previous = entries[1] if len(entries) > 1 else None
            latest_candidate = latest.get("candidate_macro_f1_mean")
            previous_candidate = previous.get("candidate_macro_f1_mean") if previous else None
            delta = None
            if latest_candidate is not None and previous_candidate is not None:
                try:
                    delta = float(latest_candidate) - float(previous_candidate)
                except (TypeError, ValueError):
                    delta = None
            room_trends.append(
                {
                    "room": room_name,
                    "latest_training_date": latest.get("training_date"),
                    "latest_pass": bool(latest.get("pass", False)),
                    "latest_candidate_macro_f1_mean": float(latest_candidate) if latest_candidate is not None else None,
                    "previous_candidate_macro_f1_mean": float(previous_candidate) if previous_candidate is not None else None,
                    "delta_vs_previous": delta,
                    "latest_champion_macro_f1_mean": (
                        float(latest.get("champion_macro_f1_mean"))
                        if latest.get("champion_macro_f1_mean") is not None
                        else None
                    ),
                    "latest_reasons": latest.get("reasons", []),
                }
            )

        room_trends = sorted(room_trends, key=lambda x: x["room"])

        overall_status = "healthy"
        if wf_enabled_runs > 0 and wf_fail_runs > 0:
            overall_status = "degraded"

        return {
            "status": overall_status,
            "elder_id": elder_id,
            "limit": int(limit),
            "summary": {
                "total_runs": total_runs,
                "success_runs": success_runs,
                "rejected_by_walk_forward_gate_runs": rejected_walk_forward_runs,
                "rejected_by_global_gate_runs": global_gate_fail_runs,
                "walk_forward_enabled_runs": wf_enabled_runs,
                "walk_forward_pass_runs": wf_pass_runs,
                "walk_forward_fail_runs": wf_fail_runs,
                "average_accuracy": float(np.mean(accuracies)) if accuracies else None,
            },
            "latest_run": latest_run,
            "room_trends": room_trends,
            "timestamp": pd.Timestamp.utcnow().isoformat(),
        }, 200
    except Exception as e:
        logger.exception("Promotion gate report generation failed.")
        return {
            "status": "error",
            "elder_id": elder_id,
            "error": str(e),
        }, 503


def render_promotion_gate_prometheus_metrics(report: dict) -> str:
    """
    Render promotion gate summary as Prometheus exposition format (0.0.4).
    """
    def _esc(value: str) -> str:
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    status_map = {
        "healthy": 0,
        "degraded": 1,
        "error": 2,
    }
    elder_id = str(report.get("elder_id", "unknown"))
    overall_status = str(report.get("status", "error")).lower()
    summary = report.get("summary", {}) or {}
    room_trends = report.get("room_trends", []) or []

    lines = [
        "# HELP beta_promotion_gate_status Overall promotion gate status (0=healthy,1=degraded,2=error)",
        "# TYPE beta_promotion_gate_status gauge",
        f'beta_promotion_gate_status{{elder_id="{_esc(elder_id)}"}} {status_map.get(overall_status, 2)}',
        "# HELP beta_promotion_gate_runs_total Recent training runs considered",
        "# TYPE beta_promotion_gate_runs_total gauge",
        f'beta_promotion_gate_runs_total{{elder_id="{_esc(elder_id)}"}} {int(summary.get("total_runs", 0) or 0)}',
        "# HELP beta_promotion_gate_wf_enabled_runs Walk-forward-enabled runs in window",
        "# TYPE beta_promotion_gate_wf_enabled_runs gauge",
        f'beta_promotion_gate_wf_enabled_runs{{elder_id="{_esc(elder_id)}"}} {int(summary.get("walk_forward_enabled_runs", 0) or 0)}',
        "# HELP beta_promotion_gate_wf_pass_runs Walk-forward pass runs in window",
        "# TYPE beta_promotion_gate_wf_pass_runs gauge",
        f'beta_promotion_gate_wf_pass_runs{{elder_id="{_esc(elder_id)}"}} {int(summary.get("walk_forward_pass_runs", 0) or 0)}',
        "# HELP beta_promotion_gate_wf_fail_runs Walk-forward fail runs in window",
        "# TYPE beta_promotion_gate_wf_fail_runs gauge",
        f'beta_promotion_gate_wf_fail_runs{{elder_id="{_esc(elder_id)}"}} {int(summary.get("walk_forward_fail_runs", 0) or 0)}',
        "# HELP beta_promotion_gate_rejected_runs Rejected runs by walk-forward gate in window",
        "# TYPE beta_promotion_gate_rejected_runs gauge",
        (
            f'beta_promotion_gate_rejected_runs{{elder_id="{_esc(elder_id)}"}} '
            f'{int(summary.get("rejected_by_walk_forward_gate_runs", 0) or 0)}'
        ),
        "# HELP beta_promotion_gate_room_latest_candidate_f1 Latest candidate macro-F1 mean per room",
        "# TYPE beta_promotion_gate_room_latest_candidate_f1 gauge",
        "# HELP beta_promotion_gate_room_candidate_f1_delta Delta vs previous candidate macro-F1 mean per room",
        "# TYPE beta_promotion_gate_room_candidate_f1_delta gauge",
        "# HELP beta_promotion_gate_room_latest_pass Latest gate pass flag per room (0/1)",
        "# TYPE beta_promotion_gate_room_latest_pass gauge",
    ]

    for row in room_trends:
        room = _esc(row.get("room", "unknown"))
        candidate_f1 = float(row.get("latest_candidate_macro_f1_mean", 0.0) or 0.0)
        delta = row.get("delta_vs_previous")
        delta_val = float(delta) if delta is not None else 0.0
        latest_pass = 1 if bool(row.get("latest_pass", False)) else 0
        labels = f'elder_id="{_esc(elder_id)}",room="{room}"'
        lines.append(f"beta_promotion_gate_room_latest_candidate_f1{{{labels}}} {candidate_f1}")
        lines.append(f"beta_promotion_gate_room_candidate_f1_delta{{{labels}}} {delta_val}")
        lines.append(f"beta_promotion_gate_room_latest_pass{{{labels}}} {latest_pass}")

    return "\n".join(lines) + "\n"


def run_server(port: int = 8504):
    """Run the health check server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, HealthHandler)
    logger.info(f"🏥 Health API Server running on http://localhost:{port}")
    logger.info(f"   Endpoints: /health, /health/ready, /health/deep")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down health server...")
        httpd.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Health Check API Server')
    parser.add_argument('--port', type=int, default=8504, help='Port to run on (default: 8504)')
    args = parser.parse_args()
    
    run_server(port=args.port)
