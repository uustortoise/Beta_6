
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import logging
import re
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO
from elderlycare_v1_16.preprocessing.noise import hampel_filter
from elderlycare_v1_16.config.settings import DEFAULT_SENSOR_COLUMNS
from ml.pipeline import UnifiedPipeline
from ml.registry import ModelRegistry
from ml.evaluation import TimeCheckpointedSplitter, evaluate_model, load_room_training_dataframe
from ml.policy_config import load_policy_from_env
from ml.release_gates import resolve_scheduled_threshold
from ml.household_analyzer import HouseholdAnalyzer
from ml.hard_negative_mining import (
    ensure_hard_negative_table,
    fetch_hard_negative_queue,
    mark_hard_negative_applied,
    mark_hard_negative_status,
    mine_hard_negative_windows,
)
import altair as alt
import glob
from config import get_room_config, get_release_gates_config
from elderlycare_v1_16.config.settings import DATA_ROOT, DB_PATH, RAW_DATA_DIR, ARCHIVE_DATA_DIR
from utils.intelligence_db import query_to_dataframe
from utils.correction_eval_history import (
    fetch_and_enrich_correction_evaluations,
    summarize_correction_evaluation_decisions,
)
from sklearn.metrics import f1_score
from dotenv import load_dotenv

# Module logger
logger = logging.getLogger(__name__)

# Constants
ALL_TIME_DAYS = 3650  # ~10 years for 'All Time' filter
RECOMMENDED_TUNING_ENV_DEFAULTS = {
    "ENABLE_MINORITY_CLASS_SAMPLING": "true",
    "MINORITY_TARGET_SHARE": "0.14",
    "MINORITY_MAX_MULTIPLIER": "3",
    "UNOCCUPIED_DOWNSAMPLE_STRIDE": "10",
    "MINORITY_TARGET_SHARE_BY_ROOM": "bathroom:0.18,bedroom:0.25,entrance:0.22,kitchen:0.12,livingroom:0.15",
    "MINORITY_MAX_MULTIPLIER_BY_ROOM": "bathroom:4,bedroom:6,entrance:5,kitchen:3,livingroom:3",
    "UNOCCUPIED_DOWNSAMPLE_STRIDE_BY_ROOM": "bathroom:12,bedroom:4,entrance:14,kitchen:10,livingroom:12",
}
RECOMMENDED_THRESHOLD_DEFAULTS_BY_ROOM = {
    "bathroom": {0: 0.35, 1: 0.65},
    "bedroom": {0: 0.35, 1: 0.60},
    "entrance": {0: 0.35, 1: 0.62},
    "kitchen": {0: 0.35, 1: 0.38},
    "livingroom": {0: 0.35, 1: 0.65},
}

# Ensure backend path is set
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))
load_dotenv()

# Config
st.set_page_config(page_title="Beta_5 Model Studio", layout="wide")

# Paths (Imported from centralized settings)
ARCHIVE_DIR = ARCHIVE_DATA_DIR
RAW_DIR = RAW_DATA_DIR

# --- Database Connection (PostgreSQL compatible) ---
try:
    from backend.db.legacy_adapter import LegacyDatabaseAdapter
    _dashboard_adapter = LegacyDatabaseAdapter()
except ImportError:
    import sqlite3
    _dashboard_adapter = None

def get_dashboard_connection():
    """Get database connection compatible with PostgreSQL or SQLite."""
    if _dashboard_adapter:
        return _dashboard_adapter.get_connection()
    else:
        # Fallback for standalone usage
        import sqlite3
        return sqlite3.connect(DB_PATH)

# --- Utils ---
@st.cache_resource
def get_db_connection():
    # Deprecated: Use get_dashboard_connection() instead
    if _dashboard_adapter:
        return _dashboard_adapter.get_connection().__enter__()
    if not DB_PATH.exists():
        return None
    import sqlite3
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def _parse_sensor_features(raw_value):
    """Parse sensor_features payload from SQLite/PG rows into a dict."""
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return {}
        try:
            parsed = json.loads(raw_value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def _parse_json_object(raw_value):
    """Parse JSON payload from DB rows into a dict."""
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return {}
        try:
            parsed = json.loads(raw_value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def _coerce_bool(raw_value) -> bool:
    """Coerce serialized bool-like values from DB JSON."""
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return False
    if isinstance(raw_value, (int, float)):
        return raw_value != 0
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {'1', 'true', 'yes', 'y'}
    return bool(raw_value)


def _friendly_gate_reason(reason_code: str) -> str:
    """Convert internal gate reason codes into operator-friendly text."""
    code = str(reason_code or "").strip()
    key = code.split(":")[0] if code else ""
    reason_map = {
        "room_threshold_failed": "Quality score is below the safe target",
        "drift_stability_failed": "Results were unstable across checks",
        "stability_guard_failed": "Night-time stability was below target",
        "transition_guard_failed": "Transition detection was below target",
        "transition_coverage_missing": "Not enough transitions to validate safely",
        "wf_no_folds": "Not enough distinct days to run safety checks",
        "baseline_advantage_failed": "Improvement over baseline was too small",
        "no_regress_failed": "New model was worse than the live model",
        "candidate_eval_error": "Could not check the new model",
        "candidate_load_error": "New model files are incomplete",
        "dataset_error": "Training data could not be read",
        "baseline_eval_error": "Could not run baseline check",
        "champion_eval_error": "Could not check the live model",
        "champion_load_error": "Live model files are incomplete",
        "gate_config_unavailable": "Safety settings are unavailable",
        "walk_forward_gate_failed": "Future-data safety check failed",
        "promotion_apply_failed": "Model passed checks but could not go live",
        "routed_review_queue_uncertainty": "Routine uncertainty routed to Review Queue",
    }
    return reason_map.get(key, key.replace("_", " ").title() if key else "Unknown")


def _friendly_run_status(status: str) -> str:
    """Convert run status values into plain language for ops users."""
    key = str(status or "").strip().lower()
    status_map = {
        "success": "Completed",
        "rejected_by_walk_forward_gate": "Paused by safety checks",
        "rejected_by_global_gate": "Paused by overall quality check",
        "failed": "Run failed",
    }
    return status_map.get(key, status or "Unknown")


def _friendly_reasons_join(reasons: list) -> str:
    if not isinstance(reasons, list) or not reasons:
        return "None"
    return ", ".join(_friendly_gate_reason(r) for r in reasons)


def _parse_gate_reason_detail(reason_code: str) -> dict:
    """Parse common gate reason payloads into structured values."""
    text = str(reason_code or "").strip()
    if not text:
        return {"kind": "unknown"}

    m = re.match(r"^room_threshold_failed:([^:]+):f1=([0-9.]+)<required=([0-9.]+)$", text)
    if m:
        return {
            "kind": "room_threshold_failed",
            "room": m.group(1),
            "f1": float(m.group(2)),
            "required": float(m.group(3)),
        }

    m = re.match(r"^fold_support_failed:([^:]+):low_support_folds=([0-9]+)<min_support=([0-9]+)$", text)
    if m:
        return {
            "kind": "fold_support_failed",
            "room": m.group(1),
            "low_folds": int(m.group(2)),
            "min_support": int(m.group(3)),
        }

    m = re.match(r"^drift_stability_failed:([^:]+):low_folds=([0-9]+)>max_low_folds=([0-9]+)$", text)
    if m:
        return {
            "kind": "drift_stability_failed",
            "room": m.group(1),
            "low_folds": int(m.group(2)),
            "max_low_folds": int(m.group(3)),
        }

    m = re.match(r"^stability_guard_failed:([^:]+):low_folds=([0-9]+)>max_low_folds=([0-9]+)$", text)
    if m:
        return {
            "kind": "stability_guard_failed",
            "room": m.group(1),
            "low_folds": int(m.group(2)),
            "max_low_folds": int(m.group(3)),
        }

    m = re.match(r"^transition_guard_failed:([^:]+):low_folds=([0-9]+)>max_low_folds=([0-9]+)$", text)
    if m:
        return {
            "kind": "transition_guard_failed",
            "room": m.group(1),
            "low_folds": int(m.group(2)),
            "max_low_folds": int(m.group(3)),
        }

    m = re.match(r"^transition_coverage_missing:([^:]+):no_transition_windows$", text)
    if m:
        return {
            "kind": "transition_coverage_missing",
            "room": m.group(1),
        }

    m = re.match(r"^wf_no_folds:([^:]+):observed_days=([0-9]+)<required_days=([0-9]+)$", text)
    if m:
        return {
            "kind": "wf_no_folds",
            "room": m.group(1),
            "observed_days": int(m.group(2)),
            "required_days": int(m.group(3)),
        }

    return {"kind": text.split(":")[0] if ":" in text else text}


def _estimate_fold_feasibility(
    observed_days: int,
    min_train_days: int,
    valid_days: int,
    step_days: int,
    max_folds: int,
) -> dict:
    """Estimate walk-forward feasibility before running model evaluation."""
    observed_days = max(0, int(observed_days))
    min_train_days = max(1, int(min_train_days))
    valid_days = max(1, int(valid_days))
    step_days = max(1, int(step_days))
    max_folds = max(1, int(max_folds))

    required_days = min_train_days + valid_days
    if observed_days < required_days:
        return {
            "ready": False,
            "required_days": required_days,
            "observed_days": observed_days,
            "expected_folds": 0,
            "level": "bad",
            "message": (
                f"{observed_days} day(s) available, but {required_days} day(s) are required. "
                "No fold can be created with this setup."
            ),
        }

    raw_folds = 1 + ((observed_days - required_days) // step_days)
    expected_folds = int(min(max_folds, max(0, raw_folds)))
    return {
        "ready": expected_folds > 0,
        "required_days": required_days,
        "observed_days": observed_days,
        "expected_folds": expected_folds,
        "level": "good" if expected_folds > 0 else "bad",
        "message": (
            f"{observed_days} day(s) available, {required_days} day(s) required. "
            f"Expected folds: {expected_folds}."
        ),
    }


def _derive_auto_tuner_suggestions(
    report: dict,
    min_train_days: int,
    step_days: int,
    lookback_days: int,
) -> list[dict]:
    """Generate operator-friendly tuning suggestions from failure patterns."""
    suggestions: list[dict] = []
    summary = (report or {}).get("summary", {}) if isinstance(report, dict) else {}
    folds_df = pd.DataFrame((report or {}).get("folds", [])) if isinstance(report, dict) else pd.DataFrame()
    num_folds = int(summary.get("num_folds", 0) or 0)
    macro_f1 = summary.get("macro_f1_mean")
    macro_precision = summary.get("macro_precision_mean")
    accuracy_mean = summary.get("accuracy_mean")

    if num_folds == 0:
        suggestions.append(
            {
                "title": "Need More Fold Coverage",
                "diagnostic": "Current setup cannot create enough validation checks.",
                "recommendation": "Use a shorter training window or include more days of data.",
                "apply_ui": {
                    "mh_min_train_days": max(2, int(min_train_days) - 1),
                    "model_health_lookback_days": min(3650, int(lookback_days) + 14),
                },
            }
        )

    min_support = None
    if not folds_df.empty and "minority_support" in folds_df.columns:
        min_support = int(folds_df["minority_support"].min())

    if (
        macro_f1 is not None
        and accuracy_mean is not None
        and float(macro_f1) <= 0.55
        and float(accuracy_mean) >= 0.90
    ) or (min_support is not None and min_support <= 1):
        suggestions.append(
            {
                "title": "High Label Imbalance",
                "diagnostic": "Model is strong on common labels but weak on rare labels.",
                "recommendation": "Increase minority sampling so rare activities are learned better.",
                "apply_env": {
                    "ENABLE_MINORITY_CLASS_SAMPLING": "true",
                    "MINORITY_TARGET_SHARE": "0.20",
                    "MINORITY_MAX_MULTIPLIER": "4",
                },
            }
        )

    if (
        macro_precision is not None
        and float(macro_precision) < 0.20
        and macro_f1 is not None
        and float(macro_f1) < 0.60
    ):
        try:
            active_stride = int(_active_training_policy().unoccupied_downsample.stride)
        except Exception:
            active_stride = int(RECOMMENDED_TUNING_ENV_DEFAULTS["UNOCCUPIED_DOWNSAMPLE_STRIDE"])
        suggestions.append(
            {
                "title": "Prediction Noise Too High",
                "diagnostic": "Too many false alarms were detected.",
                "recommendation": "Increase step size to reduce noisy repeated windows.",
                "apply_ui": {"mh_step_days": min(30, int(step_days) + 2)},
                "apply_env": {
                    "UNOCCUPIED_DOWNSAMPLE_STRIDE": str(
                        max(4, int(active_stride) + 2)
                    )
                },
            }
        )

    return suggestions


def _build_plain_promotion_card(room_report: dict) -> dict:
    """Build plain-language promotion summary card for non-ML operators."""
    room = str(room_report.get("room") or "Room")
    reasons = room_report.get("latest_reasons", [])
    reasons = reasons if isinstance(reasons, list) else []
    candidate = room_report.get("latest_candidate_macro_f1_mean")
    champion_version = room_report.get("latest_champion_version")
    title = "Promotion Denied"
    reason = "The new model did not pass safety checks."
    score = None
    for raw in reasons:
        detail = _parse_gate_reason_detail(raw)
        if detail.get("kind") == "room_threshold_failed":
            reason = f"The new {room} model is not confident enough yet."
            score = f"It scored {float(detail['f1']) * 100:.0f}%, but needs {float(detail['required']) * 100:.0f}%."
            break
        if detail.get("kind") == "fold_support_failed":
            reason = f"The {room} data is too thin across validation checks."
            score = f"{int(detail['low_folds'])} check(s) had too few rare examples (minimum: {int(detail['min_support'])})."
            break
        if detail.get("kind") == "drift_stability_failed":
            reason = f"The new {room} model was unstable across future checks."
            score = f"{int(detail['low_folds'])} check(s) were below the allowed limit of {int(detail['max_low_folds'])}."
            break
        if detail.get("kind") == "stability_guard_failed":
            reason = f"The new {room} model was not stable enough during long same-state periods."
            score = f"{int(detail['low_folds'])} check(s) failed the stability guard."
            break
        if detail.get("kind") == "transition_guard_failed":
            reason = f"The new {room} model missed too many activity-change moments."
            score = f"{int(detail['low_folds'])} check(s) failed transition quality requirements."
            break
        if detail.get("kind") == "transition_coverage_missing":
            reason = f"The new {room} model could not be validated on transitions yet."
            score = "Need more varied labels that include activity changes."
            break
        if detail.get("kind") == "wf_no_folds":
            reason = f"The new {room} model could not be validated yet."
            score = (
                f"Need more days of data: {int(detail['observed_days'])} observed day(s), "
                f"{int(detail['required_days'])} required."
            )
            break
    if score is None and candidate is not None:
        score = f"Latest new-model score: {float(candidate) * 100:.0f}%."
    incumbent = (
        f"Current live model stays active (v{int(champion_version)})."
        if champion_version is not None
        else "Current live model stays active."
    )
    return {"title": title, "room": room, "reason": reason, "score": score, "incumbent": incumbent}


def _severity_badge(text: str, level: str = "neutral") -> str:
    palette = {
        "good": {"bg": "#e8f7ee", "fg": "#1b5e20", "border": "#a5d6a7"},
        "warn": {"bg": "#fff8e1", "fg": "#7a4f01", "border": "#ffe082"},
        "bad": {"bg": "#ffebee", "fg": "#8e1b1b", "border": "#ef9a9a"},
        "neutral": {"bg": "#f5f5f5", "fg": "#424242", "border": "#e0e0e0"},
    }
    c = palette.get(level, palette["neutral"])
    return (
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:{c['bg']};color:{c['fg']};border:1px solid {c['border']};"
        "font-size:0.85rem;font-weight:600;margin-right:8px;'>"
        f"{text}</span>"
    )


def _overall_monitor_status(pass_rate_pct: float | None, blocked_runs: int) -> tuple[str, str]:
    if pass_rate_pct is None:
        return "Limited Data", "warn"
    if pass_rate_pct >= 85.0 and int(blocked_runs) == 0:
        return "Healthy", "good"
    if pass_rate_pct >= 60.0:
        return "Watch", "warn"
    return "Action Needed", "bad"


def _ml_snapshot_level(status: str) -> str:
    mapping = {
        "healthy": "good",
        "watch": "warn",
        "action_needed": "bad",
        "not_available": "neutral",
    }
    return mapping.get(str(status or "").strip().lower(), "neutral")


def _ml_snapshot_label(status: str) -> str:
    mapping = {
        "healthy": "Healthy",
        "watch": "Watch",
        "action_needed": "Action Needed",
        "not_available": "Not Available",
    }
    return mapping.get(str(status or "").strip().lower(), "Not Available")


def _ml_snapshot_metric(value: object, digits: int = 3) -> str:
    try:
        if value is None:
            return "N/A"
        return f"{float(value):.{int(digits)}f}"
    except (TypeError, ValueError):
        return "N/A"


def _ml_snapshot_source_label(source: object) -> str:
    token = str(source or "").strip().lower().replace("_", " ")
    if token in {"default", "env override", "room override", "policy", "unknown"}:
        return token
    return "unknown"


def _ml_snapshot_room_priority(status: str) -> int:
    return {
        "action_needed": 0,
        "watch": 1,
        "healthy": 2,
        "not_available": 3,
    }.get(str(status or "").strip().lower(), 9)


def _ml_snapshot_primary_room(rooms: list[dict]) -> dict | None:
    if not rooms:
        return None
    ordered = sorted(
        [r for r in rooms if isinstance(r, dict)],
        key=lambda row: (_ml_snapshot_room_priority(row.get("status")), str(row.get("room") or "")),
    )
    return ordered[0] if ordered else None


@st.cache_data(ttl=60)
def get_ml_snapshot_residents() -> list[str]:
    try:
        with get_dashboard_connection() as conn:
            df = query_to_dataframe(
                conn,
                """
                SELECT DISTINCT elder_id
                FROM training_history
                WHERE elder_id IS NOT NULL AND elder_id <> ''
                ORDER BY elder_id
                """,
                (),
            )
        if df is None or df.empty or "elder_id" not in df.columns:
            return []
        residents = [str(v).strip() for v in df["elder_id"].tolist() if str(v).strip()]
        return sorted(set(residents))
    except Exception as e:
        logger.warning(f"Failed to load ML snapshot residents: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_ml_health_snapshot(
    elder_id: str,
    room: str = "",
    lookback_runs: int = 20,
    include_raw: bool = False,
) -> dict:
    resident = str(elder_id or "").strip()
    if not resident:
        return {
            "elder_id": "",
            "generated_at": None,
            "status": {
                "overall": "not_available",
                "reason": "Resident is required.",
                "reason_code": "missing_resident",
                "data_freshness_hours": None,
            },
            "thresholds": {"global_release_macro_f1_threshold": {"value": None, "source": "unknown"}},
            "rooms": [],
            "raw": None,
        }

    try:
        from health_server import build_ml_snapshot_report

        report, status = build_ml_snapshot_report(
            elder_id=resident,
            room=str(room or ""),
            lookback_runs=int(lookback_runs),
            include_raw=bool(include_raw),
        )
        if status >= 500:
            logger.warning("ML snapshot endpoint returned %s for resident %s", status, resident)
        if not isinstance(report, dict):
            raise ValueError("Invalid snapshot payload type")
        return report
    except Exception as e:
        logger.error(f"Failed to fetch ML snapshot for {resident}: {e}")
        return {
            "elder_id": resident,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "status": {
                "overall": "not_available",
                "reason": "ML snapshot service is temporarily unavailable.",
                "reason_code": "snapshot_unavailable",
                "data_freshness_hours": None,
            },
            "thresholds": {"global_release_macro_f1_threshold": {"value": None, "source": "unknown"}},
            "rooms": [],
            "raw": None,
        }


@st.cache_data(ttl=30)
def fetch_active_system_banner() -> dict:
    """
    Resolve active-system banner from rollout state.
    """
    try:
        from ml.t80_rollout_manager import T80RolloutManager

        manager = T80RolloutManager()
        state = manager.get_state()
        if state is None:
            return {
                "visible": False,
                "stage": "unknown",
                "text": "",
                "level": "neutral",
                "started_at": None,
            }
        stage = str(getattr(state.stage, "value", "unknown"))
        started_at = getattr(state, "started_at", None)
        if stage in {"shadow", "canary"}:
            return {
                "visible": True,
                "stage": stage,
                "text": "Beta 5.5 currently active (Beta 6 in Shadow)",
                "level": "warn",
                "started_at": started_at,
            }
        if stage == "full":
            return {
                "visible": True,
                "stage": stage,
                "text": "Beta 6 currently active",
                "level": "good",
                "started_at": started_at,
            }
        if stage == "rolled_back":
            return {
                "visible": True,
                "stage": stage,
                "text": "Beta 5.5 active (Beta 6 rolled back)",
                "level": "bad",
                "started_at": started_at,
            }
    except Exception:
        pass
    return {
        "visible": False,
        "stage": "unknown",
        "text": "",
        "level": "neutral",
        "started_at": None,
    }


def render_active_system_banner(*, context_label: str = "", show_stage_details: bool = False) -> None:
    """
    Render active-system authority banner for Ops visibility.
    """
    banner = fetch_active_system_banner()
    if not banner.get("visible", False):
        return
    banner_text = str(banner.get("text") or "").strip()
    banner_level = str(banner.get("level") or "neutral")
    started_at = banner.get("started_at")
    if not banner_text:
        return
    context_prefix = f"{context_label} · " if str(context_label or "").strip() else ""
    st.markdown(
        _severity_badge(f"{context_prefix}Active System: {banner_text}", banner_level),
        unsafe_allow_html=True,
    )
    if show_stage_details and started_at:
        st.caption(f"Rollout stage: {banner.get('stage', 'unknown')} | Started: {started_at}")


@st.cache_data(ttl=60)
def fetch_shadow_divergence_monitor(elder_id: str, days: int = 30, limit: int = 60) -> dict:
    """
    Fetch latest Phase 6 shadow divergence summary from training_history metadata.
    """
    elder_id = str(elder_id or "").strip()
    if not elder_id:
        return {"status": "not_available", "reason": "missing_elder_id"}

    days = max(1, int(days))
    limit = max(1, min(int(limit), 200))
    cutoff_date = datetime.now() - timedelta(days=days)
    try:
        with get_dashboard_connection() as conn:
            history_df = query_to_dataframe(
                conn,
                """
                SELECT id, training_date, metadata
                FROM training_history
                WHERE elder_id = ?
                  AND training_date >= ?
                ORDER BY training_date DESC
                LIMIT ?
                """,
                (elder_id, cutoff_date, limit),
            )
    except Exception as e:
        return {"status": "error", "reason": str(e)}

    if history_df is None or history_df.empty:
        return {"status": "not_available", "reason": "no_training_history"}

    for _, row in history_df.iterrows():
        metadata = _parse_json_object(row.get("metadata"))
        beta6_gate = metadata.get("beta6_gate", {}) if isinstance(metadata, dict) else {}
        phase6 = beta6_gate.get("phase6_shadow_compare", {}) if isinstance(beta6_gate, dict) else {}
        if not isinstance(phase6, dict):
            continue
        summary = phase6.get("summary", {}) if isinstance(phase6.get("summary"), dict) else {}
        if not summary:
            continue
        return {
            "status": "ok",
            "run_id": metadata.get("beta6_run_id"),
            "training_date": str(row.get("training_date")) if pd.notna(row.get("training_date")) else None,
            "phase6_status": str(summary.get("status", "unknown")),
            "divergence_count": int(summary.get("divergence_count", 0) or 0),
            "unexplained_divergence_count": int(summary.get("unexplained_divergence_count", 0) or 0),
            "divergence_rate": float(summary.get("divergence_rate", 0.0) or 0.0),
            "unexplained_divergence_rate": float(summary.get("unexplained_divergence_rate", 0.0) or 0.0),
            "unexplained_divergence_rate_max": float(summary.get("unexplained_divergence_rate_max", 0.05) or 0.05),
            "report_path": summary.get("report_path"),
            "badges": summary.get("badges", []) if isinstance(summary.get("badges"), list) else [],
            "error": str(phase6.get("error", "")),
        }

    return {"status": "not_available", "reason": "phase6_shadow_compare_missing"}


def render_ml_health_snapshot_panel(snapshot: dict, panel_scope: str = "weekly", compact: bool = False) -> None:
    report = snapshot if isinstance(snapshot, dict) else {}
    status = report.get("status", {}) if isinstance(report.get("status"), dict) else {}
    rooms = report.get("rooms", []) if isinstance(report.get("rooms"), list) else []
    resident_home_context = (
        report.get("resident_home_context", {})
        if isinstance(report.get("resident_home_context"), dict)
        else {}
    )
    overall_status = str(status.get("overall") or "not_available").lower()
    overall_reason = str(status.get("reason") or "No model-health details available.")
    freshness_hours = status.get("data_freshness_hours")
    generated_at = report.get("generated_at")
    primary = _ml_snapshot_primary_room(rooms)

    status_badges = (
        _severity_badge(f"Safety Status: {_ml_snapshot_label(overall_status)}", _ml_snapshot_level(overall_status))
        + _severity_badge(f"Reason: {overall_reason}", _ml_snapshot_level(overall_status))
    )
    if str(status.get("reason_code") or "").strip().lower() == "routed_review_queue_uncertainty":
        status_badges += _severity_badge(
            "Routing: Review Queue (Not Today Clinical Alerts)",
            "warn",
        )
    st.markdown(status_badges, unsafe_allow_html=True)
    if generated_at or freshness_hours is not None:
        freshness_text = "N/A" if freshness_hours is None else f"{float(freshness_hours):.1f}h"
        st.caption(f"Generated: {generated_at or 'N/A'} | Data freshness: {freshness_text}")

    if not primary:
        st.info("ML Health Snapshot is not available yet for this resident.")
        return

    metrics = primary.get("metrics", {}) if isinstance(primary.get("metrics"), dict) else {}
    thresholds = primary.get("thresholds", {}) if isinstance(primary.get("thresholds"), dict) else {}
    support = primary.get("support", {}) if isinstance(primary.get("support"), dict) else {}
    window = report.get("window", {}) if isinstance(report.get("window"), dict) else {}

    drift = (thresholds.get("drift_threshold") or {}).get("value") if isinstance(thresholds.get("drift_threshold"), dict) else None
    threshold_source_raw = (
        (thresholds.get("drift_threshold") or {}).get("source")
        if isinstance(thresholds.get("drift_threshold"), dict)
        else "unknown"
    )
    threshold_source = _ml_snapshot_source_label(threshold_source_raw)
    fold_count = support.get("fold_count")
    lookback_days = support.get("lookback_days")
    if lookback_days is None:
        lookback_days = window.get("lookback_days")
    lookback_label = f"{int(lookback_days)}d" if isinstance(lookback_days, (int, float)) else "N/A"
    fold_label = "N/A" if fold_count is None else str(fold_count)

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("WF Model Quality (F1)", _ml_snapshot_metric(metrics.get("candidate_macro_f1_mean")))
    top2.metric("Transition Quality", _ml_snapshot_metric(metrics.get("candidate_transition_macro_f1_mean")))
    top3.metric("Safety Drift Threshold", _ml_snapshot_metric(drift, digits=2))
    top4.metric("Safety Status", _ml_snapshot_label(primary.get("status")))

    if resident_home_context:
        ctx1, ctx2, ctx3, ctx4 = st.columns(4)
        ctx1.metric(
            "Household Context",
            str(resident_home_context.get("household_type") or "missing").replace("_", " ").title(),
        )
        ctx2.metric(
            "Helper Presence",
            str(resident_home_context.get("helper_presence") or "missing").replace("_", " ").title(),
        )
        layout = resident_home_context.get("layout", {}) if isinstance(resident_home_context.get("layout"), dict) else {}
        ctx3.metric(
            "Layout Topology",
            str(layout.get("topology") or "missing").replace("_", " ").title(),
        )
        ctx4.metric(
            "Context Contract",
            str(resident_home_context.get("status") or "unknown").replace("_", " ").title(),
        )
        missing_fields = resident_home_context.get("missing_fields", [])
        if isinstance(missing_fields, list) and missing_fields:
            st.warning(
                "Resident/home context is incomplete: "
                + ", ".join(str(item) for item in missing_fields)
            )

    if compact:
        sec1, sec2, sec3, sec4 = st.columns(4)
        sec1.metric("Champion WF F1", _ml_snapshot_metric(metrics.get("champion_macro_f1_mean")))
        sec2.metric("Candidate vs Champion Delta", _ml_snapshot_metric(metrics.get("delta_vs_champion_macro_f1")))
        sec3.metric("Stability Score", _ml_snapshot_metric(metrics.get("candidate_stability_accuracy_mean")))
        sec4.metric("Last Check Time", str(primary.get("last_check_time") or "N/A"))
        st.caption(
            f"Room: {str(primary.get('room') or 'N/A').title()} | "
            f"Threshold Source: {threshold_source} | "
            f"Folds: {fold_label} | Lookback: {lookback_label} | "
            f"Confidence: {primary.get('confidence', 'Low')}"
        )
        return

    sec1, sec2, sec3, sec4 = st.columns(4)
    sec1.metric("Champion WF F1", _ml_snapshot_metric(metrics.get("champion_macro_f1_mean")))
    sec2.metric("Candidate vs Champion Delta", _ml_snapshot_metric(metrics.get("delta_vs_champion_macro_f1")))
    sec3.metric("Stability Score", _ml_snapshot_metric(metrics.get("candidate_stability_accuracy_mean")))
    sec4.metric("Last Check Time", str(primary.get("last_check_time") or "N/A"))
    st.markdown(
        _severity_badge(f"Threshold Source: {threshold_source}", "neutral")
        + _severity_badge(f"Folds: {fold_label}", "neutral")
        + _severity_badge(f"Lookback: {lookback_label}", "neutral")
        + _severity_badge(f"Confidence: {primary.get('confidence', 'Low')}", "neutral")
        + (_severity_badge("Fallback Active", "bad") if bool(primary.get("fallback_active", False)) else ""),
        unsafe_allow_html=True,
    )

    if rooms:
        room_rows = []
        for room_item in rooms:
            if not isinstance(room_item, dict):
                continue
            room_thresholds = room_item.get("thresholds", {}) if isinstance(room_item.get("thresholds"), dict) else {}
            room_metrics = room_item.get("metrics", {}) if isinstance(room_item.get("metrics"), dict) else {}
            room_rows.append(
                {
                    "Room": str(room_item.get("room") or "").title(),
                    "Status": _ml_snapshot_label(room_item.get("status")),
                    "WF Candidate F1": room_metrics.get("candidate_macro_f1_mean"),
                    "Transition Quality": room_metrics.get("candidate_transition_macro_f1_mean"),
                    "Champion WF F1": room_metrics.get("champion_macro_f1_mean"),
                    "Delta vs Champion": room_metrics.get("delta_vs_champion_macro_f1"),
                    "Drift Threshold": (room_thresholds.get("drift_threshold") or {}).get("value")
                    if isinstance(room_thresholds.get("drift_threshold"), dict)
                    else None,
                    "Threshold Source": _ml_snapshot_source_label(
                        (room_thresholds.get("drift_threshold") or {}).get("source")
                        if isinstance(room_thresholds.get("drift_threshold"), dict)
                        else "unknown"
                    ),
                    "Fold Count": (room_item.get("support") or {}).get("fold_count")
                    if isinstance(room_item.get("support"), dict)
                    else None,
                    "Lookback Days": (room_item.get("support") or {}).get("lookback_days")
                    if isinstance(room_item.get("support"), dict)
                    else None,
                    "Confidence": room_item.get("confidence"),
                    "Fallback Active": bool(room_item.get("fallback_active", False)),
                }
            )
        if room_rows:
            st.dataframe(pd.DataFrame(room_rows), hide_index=True, use_container_width=True)

    with st.expander("Technical Details", expanded=False):
        st.markdown(
            f"Scope: `{panel_scope}` | Room focus: `{str(primary.get('room') or 'n/a')}` | "
            f"fold_count={support.get('fold_count')} | "
            f"candidate_low_transition_folds={support.get('candidate_low_transition_folds')}"
        )
        st.json(
            {
                "status": status,
                "thresholds": report.get("thresholds", {}),
                "primary_room": primary,
            }
        )
        if panel_scope == "admin":
            prov_rows = []
            for room_item in rooms:
                room_name = str(room_item.get("room") or "").title()
                room_thresholds = room_item.get("thresholds", {})
                if not isinstance(room_thresholds, dict):
                    continue
                for key, payload in room_thresholds.items():
                    if not isinstance(payload, dict):
                        continue
                    prov_rows.append(
                        {
                            "Room": room_name,
                            "Threshold": key,
                            "Value": payload.get("value"),
                            "Source": _ml_snapshot_source_label(payload.get("source")),
                            "Resolved Key": payload.get("resolved_key"),
                            "Source File": payload.get("source_file"),
                            "Editor Target": payload.get("editor_target"),
                            "Owner": payload.get("owner"),
                        }
                    )
            if prov_rows:
                st.markdown("##### Threshold Provenance")
                st.dataframe(pd.DataFrame(prov_rows), hide_index=True, use_container_width=True)


def _env_int(var_name: str, default: int) -> int:
    raw = os.getenv(var_name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _env_float(var_name: str, default: float) -> float:
    raw = os.getenv(var_name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _active_training_policy():
    return load_policy_from_env(os.environ)


def _policy_room_override_csv(mapping: dict[str, object] | None) -> str:
    if not mapping:
        return ""
    normalized = {_normalize_room_key(k): str(v) for k, v in mapping.items() if _normalize_room_key(k)}
    return _format_room_override_csv(normalized)


def _normalize_room_key(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "").replace("_", "")


def _parse_room_override_csv(raw: str) -> dict[str, str]:
    """
    Parse room override CSV in the form: room:value,other_room=value
    """
    result: dict[str, str] = {}
    txt = str(raw or "").strip()
    if not txt:
        return result
    for token in txt.split(","):
        item = str(token).strip()
        if not item:
            continue
        if ":" in item:
            room_raw, value_raw = item.split(":", 1)
        elif "=" in item:
            room_raw, value_raw = item.split("=", 1)
        else:
            continue
        room_key = _normalize_room_key(room_raw)
        value_txt = str(value_raw).strip()
        if room_key and value_txt:
            result[room_key] = value_txt
    return result


def _format_room_override_csv(mapping: dict[str, str]) -> str:
    if not mapping:
        return ""
    return ",".join(f"{k}:{v}" for k, v in sorted(mapping.items()))


def _with_room_override(base_value: str, room_name: str, room_value: str) -> str:
    mapping = _parse_room_override_csv(base_value)
    key = _normalize_room_key(room_name)
    if key:
        mapping[key] = str(room_value)
    return _format_room_override_csv(mapping)


def _apply_room_aware_suggestion_env(
    env_updates: dict[str, str],
    suggestion: dict,
    target_room: str,
) -> dict[str, str]:
    """
    Ensure suggestion env updates are effective even when *_BY_ROOM overrides exist.
    """
    updates = {str(k): str(v) for k, v in (env_updates or {}).items()}
    room_key = _normalize_room_key(target_room)
    if not room_key:
        return updates

    title = str((suggestion or {}).get("title") or "").strip().lower()
    policy = _active_training_policy()
    minority_policy = policy.minority_sampling
    unocc_policy = policy.unoccupied_downsample

    if "label imbalance" in title:
        target_share = updates.get(
            "MINORITY_TARGET_SHARE",
            str(minority_policy.target_share),
        )
        max_mult = updates.get(
            "MINORITY_MAX_MULTIPLIER",
            str(minority_policy.max_multiplier),
        )
        existing_share = updates.get(
            "MINORITY_TARGET_SHARE_BY_ROOM",
            _policy_room_override_csv(minority_policy.target_share_by_room),
        )
        existing_mult = updates.get(
            "MINORITY_MAX_MULTIPLIER_BY_ROOM",
            _policy_room_override_csv(minority_policy.max_multiplier_by_room),
        )
        updates["MINORITY_TARGET_SHARE_BY_ROOM"] = _with_room_override(existing_share, room_key, target_share)
        updates["MINORITY_MAX_MULTIPLIER_BY_ROOM"] = _with_room_override(existing_mult, room_key, max_mult)

    if "prediction noise" in title and "UNOCCUPIED_DOWNSAMPLE_STRIDE" in updates:
        stride = updates.get(
            "UNOCCUPIED_DOWNSAMPLE_STRIDE",
            str(unocc_policy.stride),
        )
        existing = updates.get(
            "UNOCCUPIED_DOWNSAMPLE_STRIDE_BY_ROOM",
            _policy_room_override_csv(unocc_policy.stride_by_room),
        )
        updates["UNOCCUPIED_DOWNSAMPLE_STRIDE_BY_ROOM"] = _with_room_override(existing, room_key, stride)

    return updates


def _persist_env_updates_to_dotenv(env_updates: dict, dotenv_path: Path | None = None) -> tuple[bool, str]:
    """
    Persist env updates to .env so settings survive Streamlit restarts/scheduled runs.
    """
    if not isinstance(env_updates, dict) or not env_updates:
        return True, "no_updates"

    target_path = Path(dotenv_path) if dotenv_path is not None else (current_dir / ".env")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        lines = target_path.read_text().splitlines() if target_path.exists() else []
        key_to_idx: dict[str, int] = {}
        for idx, raw in enumerate(lines):
            line = str(raw or "").strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key = line.split("=", 1)[0].strip()
            if key:
                key_to_idx[key] = idx

        for raw_key, raw_value in env_updates.items():
            key = str(raw_key or "").strip()
            if not key:
                continue
            value = str(raw_value or "").replace("\n", " ").strip()
            record = f"{key}={value}"
            if key in key_to_idx:
                lines[key_to_idx[key]] = record
            else:
                lines.append(record)

        target_path.write_text("\n".join(lines) + "\n")
        return True, str(target_path)
    except Exception as e:
        return False, str(e)


def _summarize_runtime_load_modes(elder_id: str) -> dict:
    """
    Summarize per-room runtime loading mode for ops migration tracking.
    """
    registry = ModelRegistry(str(current_dir))
    models_dir = registry.get_models_dir(elder_id)
    if not models_dir.exists():
        return {"rows": [], "summary": {"total_rooms": 0, "shared_adapter_rooms": 0, "full_model_rooms": 0}}

    room_names = set()
    for path in models_dir.glob("*_versions.json"):
        room_names.add(path.name.replace("_versions.json", ""))
    for path in models_dir.glob("*_model.keras"):
        if "_v" not in path.name:
            room_names.add(path.name.replace("_model.keras", ""))

    rows = []
    for room in sorted(room_names):
        versions_path = models_dir / f"{room}_versions.json"
        model_path = models_dir / f"{room}_model.keras"
        scaler_path = models_dir / f"{room}_scaler.pkl"
        encoder_path = models_dir / f"{room}_label_encoder.pkl"
        adapter_path = models_dir / f"{room}_adapter_weights.pkl"

        status = "Not Ready"
        level = "bad"
        mode = "Unknown"
        blocker = "Missing model artifacts"
        backbone_id = None

        meta = registry.get_current_version_metadata(elder_id, room) if versions_path.exists() else None
        identity = meta.get("model_identity", {}) if isinstance(meta, dict) else {}
        family = str(identity.get("family", "")).strip().lower()
        backbone_id = identity.get("backbone_id") if isinstance(identity, dict) else None
        if backbone_id is not None:
            backbone_id = str(backbone_id)

        has_scaler_encoder = scaler_path.exists() and encoder_path.exists()
        has_full_model = model_path.exists()

        if family == "shared_backbone_adapter" and backbone_id:
            backbone_path = registry.get_backbone_weights_path(elder_id, room, backbone_id)
            has_backbone = backbone_path.exists()
            has_adapter = adapter_path.exists()
            mode = "Shared Adapter"
            if has_scaler_encoder and has_backbone and has_adapter:
                status = "Ready"
                level = "good"
                blocker = "None"
            else:
                missing = []
                if not has_scaler_encoder:
                    missing.append("scaler/label encoder")
                if not has_backbone:
                    missing.append("shared backbone weights")
                if not has_adapter:
                    missing.append("adapter weights")
                blocker = "Missing: " + ", ".join(missing)
                if has_full_model and has_scaler_encoder:
                    status = "Fallback Ready"
                    level = "warn"
                else:
                    status = "Not Ready"
                    level = "bad"
        elif has_full_model and has_scaler_encoder:
            mode = "Full Model"
            status = "Ready"
            level = "warn"
            blocker = "Legacy mode"
        else:
            mode = "Unknown"
            status = "Not Ready"
            level = "bad"
            missing = []
            if not has_full_model:
                missing.append("full model")
            if not has_scaler_encoder:
                missing.append("scaler/label encoder")
            blocker = "Missing: " + ", ".join(missing) if missing else blocker

        rows.append(
            {
                "Room": room,
                "Load Mode": mode,
                "Status": status,
                "Backbone ID": backbone_id or "N/A",
                "Blocker": blocker,
                "level": level,
            }
        )

    summary = {
        "total_rooms": len(rows),
        "shared_adapter_rooms": sum(1 for r in rows if r["Load Mode"] == "Shared Adapter" and r["Status"] == "Ready"),
        "full_model_rooms": sum(1 for r in rows if r["Load Mode"] == "Full Model" and r["Status"] == "Ready"),
        "fallback_ready_rooms": sum(1 for r in rows if r["Status"] == "Fallback Ready"),
        "not_ready_rooms": sum(1 for r in rows if r["Status"] == "Not Ready"),
    }
    return {"rows": rows, "summary": summary}


def _load_room_threshold_rows(elder_id: str) -> list[dict]:
    """Load latest alias thresholds + class labels for all rooms."""
    rows: list[dict] = []
    registry = ModelRegistry(str(current_dir))
    models_dir = registry.get_models_dir(elder_id)
    if not models_dir.exists():
        return rows

    for thresholds_path in sorted(models_dir.glob("*_thresholds.json")):
        room_name = thresholds_path.name.replace("_thresholds.json", "")
        if "_v" in room_name:
            continue
        try:
            thresholds_obj = json.loads(thresholds_path.read_text())
            if not isinstance(thresholds_obj, dict):
                continue
        except Exception:
            continue

        classes: list[str] = []
        encoder_path = models_dir / f"{room_name}_label_encoder.pkl"
        if encoder_path.exists():
            try:
                encoder = joblib.load(encoder_path)
                classes = [str(x) for x in getattr(encoder, "classes_", [])]
            except Exception:
                classes = []

        for raw_class_id, raw_threshold in thresholds_obj.items():
            try:
                class_id = int(raw_class_id)
                threshold_val = float(raw_threshold)
            except Exception:
                continue
            label_name = classes[class_id] if 0 <= class_id < len(classes) else f"class_{class_id}"
            rows.append(
                {
                    "room": room_name,
                    "class_id": class_id,
                    "label": label_name,
                    "threshold": float(threshold_val),
                }
            )
    return rows


def _save_room_threshold_rows(elder_id: str, rows: list[dict]) -> tuple[bool, str]:
    """Persist edited thresholds into latest alias *_thresholds.json files."""
    if not rows:
        return False, "No threshold rows to save."
    registry = ModelRegistry(str(current_dir))
    models_dir = registry.get_models_dir(elder_id)
    grouped: dict[str, dict[str, float]] = {}

    for row in rows:
        room = str(row.get("room") or "").strip()
        if not room:
            continue
        try:
            class_id = int(row.get("class_id"))
            threshold_val = float(row.get("threshold"))
        except Exception:
            continue
        threshold_val = float(min(0.99, max(0.01, threshold_val)))
        grouped.setdefault(room, {})[str(class_id)] = threshold_val

    if not grouped:
        return False, "No valid threshold rows after parsing."

    try:
        saved = 0
        for room, threshold_map in grouped.items():
            out_path = models_dir / f"{room}_thresholds.json"
            out_path.write_text(json.dumps(threshold_map, indent=2, sort_keys=True) + "\n")
            saved += 1
        return True, f"Saved thresholds for {saved} room(s)."
    except Exception as e:
        return False, str(e)


def _recommended_tuning_env_defaults() -> dict[str, str]:
    return {str(k): str(v) for k, v in RECOMMENDED_TUNING_ENV_DEFAULTS.items()}


def _apply_recommended_threshold_defaults(elder_id: str) -> tuple[bool, str]:
    rows = _load_room_threshold_rows(elder_id)
    if not rows:
        return False, "No threshold files found for this resident."

    updated_rows: list[dict] = []
    touched_rooms: set[str] = set()
    for row in rows:
        next_row = dict(row)
        room_key = _normalize_room_key(next_row.get("room", ""))
        try:
            class_id = int(next_row.get("class_id"))
        except Exception:
            updated_rows.append(next_row)
            continue
        target = RECOMMENDED_THRESHOLD_DEFAULTS_BY_ROOM.get(room_key, {}).get(class_id)
        if target is not None:
            next_row["threshold"] = float(target)
            touched_rooms.add(str(next_row.get("room")))
        updated_rows.append(next_row)

    if not touched_rooms:
        return False, "No matching room thresholds found for recommended defaults."

    ok, msg = _save_room_threshold_rows(elder_id=elder_id, rows=updated_rows)
    if not ok:
        return False, msg
    rooms_txt = ", ".join(sorted(touched_rooms))
    return True, f"{msg} Recommended thresholds applied for: {rooms_txt}."


def _room_key(room_name: str) -> str:
    return str(room_name).strip().lower().replace(" ", "").replace("_", "")


def _snapshot_champions(registry: ModelRegistry, elder_id: str) -> dict:
    """
    Snapshot champion versions and metrics keyed by canonical room name.
    """
    snapshot = {}
    model_dir = registry.get_models_dir(elder_id)
    for versions_file in model_dir.glob("*_versions.json"):
        room_name = versions_file.name.replace("_versions.json", "")
        key = _room_key(room_name)
        version = int(registry.get_current_version(elder_id, room_name) or 0)
        meta = registry.get_current_version_metadata(elder_id, room_name) if version > 0 else None
        snapshot[key] = {
            "room_name": room_name,
            "version": version,
            "meta": meta or {},
        }
    return snapshot


def _compute_global_macro_f1(snapshot: dict, room_keys: set[str]) -> float | None:
    vals = []
    for key in room_keys:
        item = snapshot.get(key)
        if not item:
            continue
        metrics = (item.get("meta") or {}).get("metrics") or {}
        macro_f1 = metrics.get("macro_f1")
        if macro_f1 is not None:
            vals.append(float(macro_f1))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _compute_macro_f1_safe(y_true: list[str], y_pred: list[str]) -> float | None:
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return None
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _evaluate_corrected_window_policy(
    local_before_f1: float | None,
    local_after_f1: float | None,
    global_before_f1: float | None,
    global_after_f1: float | None,
) -> dict:
    policy = get_release_gates_config().get("corrected_window_policy", {})
    local_gain_min = float(policy.get("local_gain_min", 0.10))
    global_drop_max = float(policy.get("global_drop_max", 0.02))

    local_gain = None
    if local_before_f1 is not None and local_after_f1 is not None:
        local_gain = float(local_after_f1) - float(local_before_f1)

    global_drop = None
    if global_before_f1 is not None and global_after_f1 is not None:
        global_drop = float(global_before_f1) - float(global_after_f1)

    if global_drop is not None and global_drop > global_drop_max:
        decision = "FAIL"
    elif local_gain is not None and local_gain >= local_gain_min:
        decision = "PASS"
    else:
        decision = "PASS_WITH_FLAG"

    return {
        "decision": decision,
        "local_before_f1": local_before_f1,
        "local_after_f1": local_after_f1,
        "local_gain": local_gain,
        "local_gain_min": local_gain_min,
        "global_before_f1": global_before_f1,
        "global_after_f1": global_after_f1,
        "global_drop": global_drop,
        "global_drop_max": global_drop_max,
    }


def _rollback_to_snapshot(
    registry: ModelRegistry,
    elder_id: str,
    room_keys: set[str],
    before_snapshot: dict,
) -> tuple[list[str], list[str]]:
    rolled_back = []
    deactivated = []

    current_snapshot = _snapshot_champions(registry, elder_id)
    for key in room_keys:
        current_room = (current_snapshot.get(key) or {}).get("room_name")
        before = before_snapshot.get(key)
        if not current_room:
            continue
        target_version = int((before or {}).get("version", 0))
        if target_version > 0:
            if registry.rollback_to_version(elder_id, current_room, target_version):
                rolled_back.append(current_room)
        else:
            if registry.deactivate_current_version(elder_id, current_room):
                deactivated.append(current_room)
    return rolled_back, deactivated


def _persist_correction_evaluation_artifact(
    elder_id: str,
    decision: str,
    avg_acc: float,
    metrics: list[dict] | None,
    corrected_window_report: dict,
    rolled_back: list[str],
    deactivated: list[str],
    ranges_applied: int,
    total_rows_updated: int,
    affected_rooms: set[str],
) -> str:
    """
    Persist correction evaluation artifact to disk and training_history table.

    Returns:
        Artifact file path as string.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = DATA_ROOT / "processed" / "correction_evaluations"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{elder_id}_correction_eval_{ts}.json"

    payload = {
        "artifact_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "elder_id": elder_id,
        "decision": decision,
        "avg_accuracy": float(avg_acc),
        "ranges_applied": int(ranges_applied),
        "rows_updated": int(total_rows_updated),
        "affected_rooms": sorted([str(r) for r in affected_rooms]),
        "corrected_window_report": corrected_window_report,
        "metrics": metrics or [],
        "rollback": {
            "rolled_back_rooms": rolled_back,
            "deactivated_rooms": deactivated,
        },
    }

    with open(artifact_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    status_map = {
        "PASS": "success",
        "PASS_WITH_FLAG": "pass_with_flag",
        "FAIL": "rejected_by_corrected_window_policy",
    }
    status = status_map.get(decision, "unknown")

    try:
        with get_dashboard_connection() as conn_hist:
            conn_hist.execute(
                '''
                INSERT INTO training_history (elder_id, model_type, epochs, accuracy, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (
                    elder_id,
                    "Correction Retrain",
                    int(metrics[0]["epochs"]) if metrics else 0,
                    float(avg_acc),
                    status,
                    json.dumps(
                        {
                            "artifact_path": str(artifact_path),
                            "decision": decision,
                            "corrected_window_report": corrected_window_report,
                            "rollback": {
                                "rolled_back_rooms": rolled_back,
                                "deactivated_rooms": deactivated,
                            },
                        }
                    ),
                ),
            )
            conn_hist.commit()
    except Exception as e:
        logger.warning(f"Failed to persist correction evaluation history row: {e}")

    return str(artifact_path)


@st.cache_data(ttl=5)  # Short TTL for near real-time updates (cache cleared after corrections)
def fetch_predictions(elder_id, room, record_date, _version="v4"):
    """Cached fetching of predictions to speed up UI responsiveness."""
    # Note: DB_PATH check removed - not needed for PostgreSQL-only mode
    try:
        with get_dashboard_connection() as conn:
            # Normalize room names: remove spaces/underscores and lowercase
            # This handles: 'Living Room', 'living_room', 'LivingRoom', 'livingroom' all matching
            q_pred = """
                SELECT timestamp, activity_type as predicted_activity, confidence, sensor_features
                FROM adl_history
                WHERE elder_id=? 
                  AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = LOWER(REPLACE(REPLACE(?, ' ', ''), '_', ''))
                  AND record_date=?
            """
            pred_df = query_to_dataframe(conn, q_pred, (elder_id, room, record_date))
            if pred_df.empty:
                return pred_df

            parsed_features = pred_df['sensor_features'].apply(_parse_sensor_features)

            pred_df['predicted_top1_label'] = parsed_features.apply(
                lambda x: x.get('predicted_top1_label') or x.get('low_confidence_hint_label')
            )
            pred_df['predicted_top1_prob'] = pd.to_numeric(
                parsed_features.apply(lambda x: x.get('predicted_top1_prob')),
                errors='coerce'
            )
            pred_df['predicted_top2_label'] = parsed_features.apply(lambda x: x.get('predicted_top2_label'))
            pred_df['predicted_top2_prob'] = pd.to_numeric(
                parsed_features.apply(lambda x: x.get('predicted_top2_prob')),
                errors='coerce'
            )
            pred_df['low_confidence_threshold'] = pd.to_numeric(
                parsed_features.apply(lambda x: x.get('low_confidence_threshold')),
                errors='coerce'
            )
            pred_df['is_low_confidence'] = parsed_features.apply(
                lambda x: _coerce_bool(x.get('is_low_confidence', False))
            )
            pred_df['low_confidence_hint_label'] = parsed_features.apply(lambda x: x.get('low_confidence_hint_label'))

            hint_series = pred_df['low_confidence_hint_label'].fillna(pred_df['predicted_top1_label'])
            low_mask = pred_df['predicted_activity'].astype(str).str.lower().eq('low_confidence')
            pred_df['display_activity'] = pred_df['predicted_activity']
            pred_df.loc[low_mask & hint_series.notna(), 'display_activity'] = (
                'low_confidence-' + hint_series[low_mask & hint_series.notna()].astype(str)
            )
            return pred_df
    except Exception as e:
        logger.error(f"Failed to fetch predictions: {e}")
        return pd.DataFrame()


# =============================================================================
# AUDIT TRAIL CACHED FUNCTIONS (Moved outside tab context for proper caching)
# =============================================================================

@st.cache_data(ttl=60)
def fetch_correction_history_cached(elder_filter=None, room_filter=None, days=ALL_TIME_DAYS, include_deleted=False):
    """Fetch correction history with optional filters."""
    try:
        with get_dashboard_connection() as conn:
            query = '''
                SELECT
                    id,
                    corrected_at,
                    elder_id,
                    room,
                    timestamp_start,
                    timestamp_end,
                    old_activity,
                    new_activity,
                    rows_affected,
                    corrected_by,
                    is_deleted,
                    deleted_at,
                    deleted_by
                FROM correction_history
                WHERE corrected_at >= ?
            '''
            cutoff_date = datetime.now() - timedelta(days=days)
            params = [cutoff_date]

            # Filter deleted unless showing all
            if not include_deleted:
                query += ' AND (is_deleted = 0 OR is_deleted IS NULL)'

            if elder_filter and elder_filter != 'All':
                query += ' AND elder_id = ?'
                params.append(elder_filter)

            if room_filter and room_filter != 'All':
                # Normalize room name for matching
                clean_room = room_filter.replace(" ", "").replace("_", "").lower()
                query += ' AND LOWER(REPLACE(REPLACE(room, " ", ""), "_", "")) = ?'
                params.append(clean_room)

            query += ' ORDER BY corrected_at DESC'

            # Use cursor directly instead of pd.read_sql
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=columns)
            else:
                df = pd.DataFrame()
                
            return df
    except Exception as e:
        logger.error(f"Failed to load correction history: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_correction_evaluation_history(elder_filter=None, days=ALL_TIME_DAYS):
    """Fetch correction retrain evaluation rows from training_history."""
    try:
        with get_dashboard_connection() as conn:
            return fetch_and_enrich_correction_evaluations(
                query_fn=lambda query, params: query_to_dataframe(conn, query, params),
                elder_filter=elder_filter,
                days=days,
            )
    except Exception as e:
        logger.error(f"Failed to load correction evaluation history: {e}")
        return pd.DataFrame()


def _to_float_or_none(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(parsed):
        return None
    return float(parsed)


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


@st.cache_data(ttl=60)
def fetch_promotion_gate_monitor(elder_id: str, days: int = 30, limit: int = 60) -> dict:
    """
    Summarize promotion gate outcomes from training_history metadata for dashboard monitoring.
    """
    elder_id = str(elder_id or "").strip()
    if not elder_id:
        return {
            "total_runs": 0,
            "wf_enabled_runs": 0,
            "wf_pass_runs": 0,
            "wf_fail_runs": 0,
            "wf_pass_rate_pct": None,
            "top_failure_reason": None,
            "latest_delta": None,
            "latest_run": None,
            "room_trends": [],
            "production_counters": {
                "preprocessing_fail": 0,
                "viability_fail": 0,
                "statistical_validity_fail": 0,
                "gate_fail": 0,
                "promoted": 0,
                "no_op_retrain_skip": 0,
            },
        }

    days = max(1, int(days))
    limit = max(1, min(int(limit), 200))
    cutoff_date = datetime.now() - timedelta(days=days)

    try:
        with get_dashboard_connection() as conn:
            history_df = query_to_dataframe(
                conn,
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
        logger.error(f"Failed to fetch promotion gate monitor data: {e}")
        return {
            "total_runs": 0,
            "wf_enabled_runs": 0,
            "wf_pass_runs": 0,
            "wf_fail_runs": 0,
            "wf_pass_rate_pct": None,
            "top_failure_reason": None,
            "latest_delta": None,
            "latest_run": None,
            "room_trends": [],
            "production_counters": {
                "preprocessing_fail": 0,
                "viability_fail": 0,
                "statistical_validity_fail": 0,
                "gate_fail": 0,
                "promoted": 0,
                "no_op_retrain_skip": 0,
            },
            "error": str(e),
        }

    if history_df is None or history_df.empty:
        return {
            "total_runs": 0,
            "wf_enabled_runs": 0,
            "wf_pass_runs": 0,
            "wf_fail_runs": 0,
            "wf_pass_rate_pct": None,
            "top_failure_reason": None,
            "latest_delta": None,
            "latest_run": None,
            "room_trends": [],
            "production_counters": {
                "preprocessing_fail": 0,
                "viability_fail": 0,
                "statistical_validity_fail": 0,
                "gate_fail": 0,
                "promoted": 0,
                "no_op_retrain_skip": 0,
            },
        }

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
        metadata = _parse_json_object(row.get("metadata"))
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
        "release_gate_profile": {
            "evidence_profile": str(os.getenv("RELEASE_GATE_EVIDENCE_PROFILE", "production")).strip() or "production",
            "bootstrap_enabled": _coerce_bool(os.getenv("RELEASE_GATE_BOOTSTRAP_ENABLED", False)),
            "bootstrap_phase1_max_days": _env_int("RELEASE_GATE_BOOTSTRAP_PHASE1_MAX_DAYS", 7),
            "bootstrap_max_days": _env_int("RELEASE_GATE_BOOTSTRAP_MAX_TRAINING_DAYS", 14),
            "strict_prior_drift_max": _env_float("RELEASE_GATE_MAX_CLASS_PRIOR_DRIFT", 0.10),
            "wf_min_train_days": _env_int("WF_MIN_TRAIN_DAYS", 7),
            "wf_valid_days": _env_int("WF_VALID_DAYS", 1),
            "release_threshold_tracker": release_tracker_rows,
        },
        "production_counters": production_counters,
    }


@st.cache_data(ttl=300)
def build_walk_forward_dataset(elder_id: str, room_name: str, lookback_days: int = 90):
    """
    Build sequence dataset for walk-forward evaluation from archived training files.
    """
    try:
        from utils.data_loader import load_sensor_data
        from utils.room_utils import normalize_room_name
    except Exception as e:
        logger.error(f"Failed to import data loaders for walk-forward dataset: {e}")
        return None, f"Import error: {e}"
    return load_room_training_dataframe(
        elder_id=elder_id,
        room_name=room_name,
        archive_dir=ARCHIVE_DIR,
        load_sensor_data_fn=load_sensor_data,
        normalize_room_name_fn=normalize_room_name,
        lookback_days=int(lookback_days),
    )


@st.cache_data(ttl=60)
def get_correction_summary():
    """Get summary statistics for dashboard."""
    try:
        with get_dashboard_connection() as conn:
            cursor = conn.cursor()
            
            # Total corrections
            cursor.execute('SELECT COUNT(*) FROM correction_history')
            total = cursor.fetchone()[0]
            
            # This week
            seven_days_ago = datetime.now() - timedelta(days=7)
            cursor.execute("SELECT COUNT(*) FROM correction_history WHERE corrected_at >= ?", (seven_days_ago,))
            this_week = cursor.fetchone()[0]
            
            # Today
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cursor.execute("SELECT COUNT(*) FROM correction_history WHERE corrected_at >= ?", (today_start,))
            today = cursor.fetchone()[0]
            
            # Most corrected elder
            cursor.execute('''
                SELECT elder_id, COUNT(*) as cnt 
                FROM correction_history 
                GROUP BY elder_id 
                ORDER BY cnt DESC 
                LIMIT 1
            ''')
            top_elder_row = cursor.fetchone()
            top_elder = f"{top_elder_row[0]} ({top_elder_row[1]})" if top_elder_row else "N/A"
            
            # Most corrected activity (new_activity)
            cursor.execute('''
                SELECT new_activity, COUNT(*) as cnt 
                FROM correction_history 
                GROUP BY new_activity 
                ORDER BY cnt DESC 
                LIMIT 1
            ''')
            top_activity_row = cursor.fetchone()
            top_activity = f"{top_activity_row[0]} ({top_activity_row[1]})" if top_activity_row else "N/A"
            
            # Total rows affected
            cursor.execute('SELECT SUM(rows_affected) FROM correction_history')
            total_rows = cursor.fetchone()[0] or 0
            
            return {
                'total': total,
                'this_week': this_week,
                'today': today,
                'top_elder': top_elder,
                'top_activity': top_activity,
                'total_rows': total_rows
            }
    except Exception as e:
        return {'error': str(e)}


@st.cache_data(ttl=20)
def fetch_hard_negative_queue_cached(
    elder_id: str,
    room_filter: str = "",
    days: int = 30,
    status: str = "open",
    limit: int = 200,
) -> pd.DataFrame:
    elder_id = str(elder_id or "").strip()
    if not elder_id:
        return pd.DataFrame()
    room_filter = str(room_filter or "").strip()
    try:
        with get_dashboard_connection() as conn:
            ensure_hard_negative_table(conn)
            return fetch_hard_negative_queue(
                conn=conn,
                elder_id=elder_id,
                room=room_filter or None,
                days=int(days),
                status=str(status or "open"),
                limit=int(limit),
            )
    except Exception as e:
        logger.warning(f"Failed to fetch hard-negative queue: {e}")
        return pd.DataFrame()


def run_hard_negative_mining_now(
    elder_id: str,
    record_date: str,
    risky_rooms_csv: str = "bathroom,entrance,kitchen",
    top_k_per_room: int = 3,
    min_block_rows: int = 6,
) -> dict:
    elder_id = str(elder_id or "").strip()
    if not elder_id:
        return {"candidates": 0, "inserted": 0, "updated": 0, "error": "missing_elder_id"}
    risky_rooms = [r.strip() for r in str(risky_rooms_csv or "").split(",") if r.strip()]
    try:
        with get_dashboard_connection() as conn:
            ensure_hard_negative_table(conn)
            stats = mine_hard_negative_windows(
                conn=conn,
                elder_id=elder_id,
                record_date=str(record_date or "").strip() or None,
                risky_rooms=risky_rooms,
                top_k_per_room=max(1, int(top_k_per_room)),
                min_block_rows=max(3, int(min_block_rows)),
                source="streamlit_manual",
            )
            conn.commit()
            try:
                fetch_hard_negative_queue_cached.clear()
            except Exception:
                pass
            try:
                fetch_hard_negative_summary_cached.clear()
            except Exception:
                pass
            return stats
    except Exception as e:
        return {"candidates": 0, "inserted": 0, "updated": 0, "error": str(e)}


@st.cache_data(ttl=20)
def fetch_hard_negative_summary_cached(
    elder_id: str,
    days: int = 30,
) -> dict:
    elder_id = str(elder_id or "").strip()
    if not elder_id:
        return {
            "open": 0,
            "queued": 0,
            "reviewed": 0,
            "applied_week": 0,
            "dismissed": 0,
            "top_rooms": [],
        }
    days = max(1, int(days))
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    week_cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        with get_dashboard_connection() as conn:
            ensure_hard_negative_table(conn)
            status_rows = conn.execute(
                """
                SELECT status, COUNT(*) AS cnt
                FROM hard_negative_queue
                WHERE elder_id = ? AND created_at >= ?
                GROUP BY status
                """,
                (elder_id, cutoff),
            ).fetchall()
            room_rows = conn.execute(
                """
                SELECT room, reason, COUNT(*) AS cnt
                FROM hard_negative_queue
                WHERE elder_id = ? AND created_at >= ? AND status IN ('open', 'queued')
                GROUP BY room, reason
                ORDER BY cnt DESC, room ASC
                LIMIT 10
                """,
                (elder_id, cutoff),
            ).fetchall()
            applied_row = conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM hard_negative_queue
                WHERE elder_id = ? AND status = 'applied' AND updated_at >= ?
                """,
                (elder_id, week_cutoff),
            ).fetchone()

        status_counts = {"open": 0, "queued": 0, "reviewed": 0, "dismissed": 0}
        for row in status_rows:
            try:
                key = str(row["status"] or "").strip().lower()
                cnt = int(row["cnt"] or 0)
            except Exception:
                key = str(row[0] or "").strip().lower()
                cnt = int(row[1] or 0)
            if key in status_counts:
                status_counts[key] = cnt

        top_rooms = []
        for row in room_rows:
            try:
                room = row["room"]
                reason = row["reason"]
                cnt = int(row["cnt"] or 0)
            except Exception:
                room, reason, cnt = row[0], row[1], int(row[2] or 0)
            top_rooms.append({"room": room, "reason": reason, "count": cnt})

        try:
            applied_week = int(applied_row["cnt"] or 0) if applied_row is not None else 0
        except Exception:
            applied_week = int(applied_row[0] or 0) if applied_row is not None else 0

        return {
            "open": int(status_counts["open"]),
            "queued": int(status_counts["queued"]),
            "reviewed": int(status_counts["reviewed"]),
            "dismissed": int(status_counts["dismissed"]),
            "applied_week": int(applied_week),
            "top_rooms": top_rooms,
        }
    except Exception as e:
        logger.warning(f"Failed to fetch hard-negative summary: {e}")
        return {
            "open": 0,
            "queued": 0,
            "reviewed": 0,
            "dismissed": 0,
            "applied_week": 0,
            "top_rooms": [],
            "error": str(e),
        }


@st.cache_data(ttl=120)
def get_available_filters():
    """Get unique elders and rooms for filter dropdowns."""
    try:
        with get_dashboard_connection() as conn:
            elders_df = query_to_dataframe(conn, 'SELECT DISTINCT elder_id FROM correction_history ORDER BY elder_id')
            rooms_df = query_to_dataframe(conn, 'SELECT DISTINCT room FROM correction_history ORDER BY room')
            elders = elders_df['elder_id'].tolist() if 'elder_id' in elders_df.columns else []
            rooms = rooms_df['room'].tolist() if 'room' in rooms_df.columns else []
            return elders, rooms
    except:
        return [], []


def clear_all_caches():
    """Clear all Streamlit caches after corrections to ensure fresh data is shown."""
    # Activity Timeline / Predictions
    fetch_predictions.clear()
    
    # Export bounds
    try:
        get_db_bounds.clear()
    except Exception as e:
        logger.warning(f"Failed to clear get_db_bounds cache: {e}")
    
    # Audit Trail
    try:
        fetch_correction_history_cached.clear()
    except Exception as e:
        logger.warning(f"Failed to clear fetch_correction_history_cached cache: {e}")
    
    # Dashboard stats
    try:
        get_correction_summary.clear()
    except Exception as e:
        logger.warning(f"Failed to clear get_correction_summary cache: {e}")

    # Hard-negative review queue
    try:
        fetch_hard_negative_queue_cached.clear()
    except Exception as e:
        logger.warning(f"Failed to clear fetch_hard_negative_queue_cached cache: {e}")
    try:
        fetch_hard_negative_summary_cached.clear()
    except Exception as e:
        logger.warning(f"Failed to clear fetch_hard_negative_summary_cached cache: {e}")
    
    # Filters
    try:
        get_available_filters.clear()
    except Exception as e:
        logger.warning(f"Failed to clear get_available_filters cache: {e}")
    
    # Full history export
    try:
        get_full_history.clear()
    except Exception as e:
        logger.warning(f"Failed to clear get_full_history cache: {e}")
    
    logger.info("All caches cleared after correction")



# --- Activity Label Management ---
DEFAULT_ACTIVITY_LABELS = [
    "inactive", "unoccupied", "low_confidence",
    "bathroom_normal_use", "shower", "toilet",
    "kitchen_normal_use", "cooking", "washing_dishes",
    "livingroom_normal_use", "watch_tv", "nap",
    "bedroom_normal_use", "sleep", "change_clothes",
    "out", "fall"
]

def get_activity_labels():
    """Get activity labels from database config or return defaults."""
    try:
        with get_dashboard_connection() as conn:
            result = conn.execute(
                "SELECT value FROM household_config WHERE key = 'custom_activity_labels'"
            ).fetchone()
            if result and result[0]:
                return json.loads(result[0])
    except Exception as e:
        logger.warning(f"Failed to get activity labels from DB: {e}")
    return DEFAULT_ACTIVITY_LABELS

def save_activity_labels(labels: list):
    """Save activity labels to database config."""
    with get_dashboard_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO household_config (key, value) VALUES (?, ?)",
            ('custom_activity_labels', json.dumps(labels))
        )
        conn.commit()


def get_archive_files(file_type=None, resident_id=None):
    """
    Get archived files.
    Args:
        file_type: 'input' for prediction files, 'train' for training files, None for all
        resident_id: Filter by resident ID (optional)
    """
    files = []
    if ARCHIVE_DIR.exists():
        for date_dir in ARCHIVE_DIR.iterdir():
            if date_dir.is_dir():
                # Support both xlsx and parquet formats
                for pattern in ["*.xlsx", "*.parquet"]:
                    for f in date_dir.glob(pattern):
                        fname_lower = f.name.lower()
                        
                        # Filter by file type
                        if file_type == 'input':
                            if '_train' in fname_lower:
                                continue  # Skip training files
                        elif file_type == 'train':
                            if '_input' in fname_lower or '_train' not in fname_lower:
                                continue  # Skip non-training files
                        
                        # Filter by resident
                        if resident_id and resident_id.lower() not in fname_lower:
                            continue
                        
                        files.append({
                            "date": date_dir.name,
                            "filename": f.name,
                            "path": str(f)
                        })
    # Sort by date desc
    files.sort(key=lambda x: x['date'], reverse=True)
    return files


@st.cache_data(ttl=60)
def build_label_quality_preflight(elder_id: str, room_filter_csv: str = "") -> dict:
    """
    Build a non-blocking label quality preflight report from archived training files.

    The report is intended for operator warning/visibility before retraining.
    """
    elder_id = str(elder_id or "").strip()
    if not elder_id:
        return {"status": "error", "score": None, "warnings": ["Missing resident ID."], "rooms": []}

    target_rooms = {
        x.strip().lower().replace(" ", "").replace("_", "")
        for x in str(room_filter_csv or "").split(",")
        if str(x).strip()
    }

    train_files = get_archive_files(file_type="train", resident_id=elder_id)
    if not train_files:
        return {
            "status": "warn",
            "score": None,
            "warnings": [f"No archived training files found for {elder_id}."],
            "rooms": [],
        }

    try:
        from utils.data_loader import load_sensor_data
        from utils.room_utils import normalize_room_name
        from utils.segment_utils import normalize_activity_name, validate_activity_for_room
    except Exception as e:
        return {"status": "error", "score": None, "warnings": [f"Preflight import failed: {e}"], "rooms": []}

    rows = []
    for f in train_files:
        path = Path(f["path"])
        try:
            loaded = load_sensor_data(path, resample=False)
        except Exception:
            continue
        if not isinstance(loaded, dict):
            continue
        for room_name, df in loaded.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if "timestamp" not in df.columns or "activity" not in df.columns:
                continue
            room_norm = normalize_room_name(str(room_name))
            if target_rooms and room_norm not in target_rooms:
                continue
            local = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(df["timestamp"], errors="coerce"),
                    "room": room_norm,
                    "activity": df["activity"].astype(str).str.strip().str.lower(),
                }
            ).dropna(subset=["timestamp"])
            if local.empty:
                continue
            local["activity"] = local["activity"].map(normalize_activity_name)
            local["activity"] = local.apply(
                lambda r: validate_activity_for_room(str(r["activity"]), str(r["room"])),
                axis=1,
            )
            local["day"] = local["timestamp"].dt.floor("D")
            rows.append(local[["timestamp", "day", "room", "activity"]])

    if not rows:
        return {
            "status": "warn",
            "score": None,
            "warnings": ["No usable timestamp/activity rows found in archived training files."],
            "rooms": [],
        }

    all_df = pd.concat(rows, ignore_index=True)
    inactive_family = {"inactive", "unoccupied", "low_confidence"}
    room_reports = []
    warnings = []

    for room, room_df in all_df.groupby("room"):
        total = int(len(room_df))
        uniq = int(room_df["activity"].nunique())
        class_counts = room_df["activity"].value_counts()
        dominant_share = float(class_counts.iloc[0] / total) if total > 0 else 1.0
        minority_share = float(1.0 - dominant_share)
        inactive_ratio = float(room_df["activity"].isin(inactive_family).mean())
        day_nunique = room_df.groupby("day")["activity"].nunique()
        single_class_days = int((day_nunique <= 1).sum())
        total_days = int(day_nunique.shape[0])

        score = 100.0
        if uniq < 2:
            score -= 35.0
        if inactive_ratio > 0.90:
            score -= 25.0
        elif inactive_ratio > 0.80:
            score -= 15.0
        elif inactive_ratio > 0.70:
            score -= 5.0
        if minority_share < 0.10:
            score -= 25.0
        elif minority_share < 0.20:
            score -= 10.0
        if total_days > 0 and single_class_days > 0:
            score -= min(20.0, (single_class_days / total_days) * 20.0)
        score = max(0.0, min(100.0, score))

        room_reports.append(
            {
                "room": room,
                "score": float(round(score, 1)),
                "rows": total,
                "days": total_days,
                "unique_labels": uniq,
                "inactive_family_ratio": float(round(inactive_ratio, 3)),
                "minority_share": float(round(minority_share, 3)),
                "single_class_days": single_class_days,
            }
        )

        if uniq < 2:
            warnings.append(f"{room}: only one label class found.")
        if single_class_days > 0:
            warnings.append(f"{room}: {single_class_days}/{total_days} day(s) are single-class.")
        if inactive_ratio > 0.90:
            warnings.append(f"{room}: inactive/unoccupied dominates ({inactive_ratio:.1%}).")

    room_reports.sort(key=lambda x: x["room"])
    if room_reports:
        weighted = np.average(
            np.array([r["score"] for r in room_reports], dtype=float),
            weights=np.array([max(1, int(r["rows"])) for r in room_reports], dtype=float),
        )
        overall = float(round(weighted, 1))
    else:
        overall = None

    status = "ok"
    if overall is None or len(warnings) > 0:
        status = "warn"
    if overall is not None and overall < 60:
        status = "warn"

    return {
        "status": status,
        "score": overall,
        "warnings": warnings[:8],
        "rooms": room_reports,
        "rows_total": int(len(all_df)),
        "files_total": int(len(train_files)),
    }

def save_corrections_to_db(elder_id: str, room: str, corrections_df: pd.DataFrame):
    """
    Save label corrections directly to the database.
    This updates adl_history and regenerates activity_segments.
    
    Args:
        elder_id: The resident ID
        room: The room name (e.g., 'bathroom')
        corrections_df: DataFrame with 'timestamp' and 'activity' columns
    
    Returns:
        tuple: (success: bool, message: str, count: int)
    """
    if 'timestamp' not in corrections_df.columns or 'activity' not in corrections_df.columns:
        return False, "DataFrame must have 'timestamp' and 'activity' columns", 0
    
    try:
        with get_dashboard_connection() as conn:
            cursor = conn.cursor()
            
            updated_count = 0
            record_dates = set()
            
            # Group corrections by contiguous same-activity blocks for bulk updates
            # This ensures we update ALL DB rows in the range, not just Excel-matched ones
            corrections_df = corrections_df.copy()
            corrections_df['timestamp'] = pd.to_datetime(corrections_df['timestamp'], errors='coerce')
            corrections_df = corrections_df.dropna(subset=['timestamp'])
            if corrections_df.empty:
                return False, "No valid correction timestamps after parsing.", 0
            corrections_df = corrections_df.sort_values('timestamp')
            
            # Identify contiguous blocks of same activity
            corrections_df['activity_changed'] = corrections_df['activity'] != corrections_df['activity'].shift()
            corrections_df['block'] = corrections_df['activity_changed'].cumsum()
            
            for block_id, block_df in corrections_df.groupby('block'):
                activity = block_df['activity'].iloc[0]
                block_start = block_df['timestamp'].min()
                block_end = block_df['timestamp'].max()
                
                record_date = block_start.strftime('%Y-%m-%d')
                record_dates.add(record_date)
                
                # Expand window slightly to catch edge rows
                ts_min = (block_start - pd.Timedelta(seconds=5)).strftime('%Y-%m-%d %H:%M:%S')
                ts_max = (block_end + pd.Timedelta(seconds=5)).strftime('%Y-%m-%d %H:%M:%S')
                
                # Bulk update ALL DB rows in this time range (with room normalization)
                cursor.execute('''
                    UPDATE adl_history 
                    SET activity_type = ?, is_corrected = 1
                    WHERE elder_id = ? 
                      AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = LOWER(REPLACE(REPLACE(?, ' ', ''), '_', ''))
                      AND timestamp BETWEEN ? AND ?
                ''', (activity, elder_id, room, ts_min, ts_max))
                
                updated_count += cursor.rowcount
            
            conn.commit()
            
            # Regenerate activity_segments for affected dates
            for record_date in record_dates:
                # Delete old segments for this date/room (with room normalization)
                cursor.execute('''
                    DELETE FROM activity_segments 
                    WHERE elder_id = ? 
                      AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = LOWER(REPLACE(REPLACE(?, ' ', ''), '_', ''))
                      AND record_date = ?
                ''', (elder_id, room, record_date))
                
                # Fetch updated events and regenerate segments (with room normalization)
                events_df = query_to_dataframe(conn, '''
                    SELECT timestamp, activity_type as predicted_activity, confidence, is_corrected
                    FROM adl_history 
                    WHERE elder_id = ? 
                      AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = LOWER(REPLACE(REPLACE(?, ' ', ''), '_', ''))
                      AND record_date = ?
                    ORDER BY timestamp
                ''', (elder_id, room, record_date))
                
                if not events_df.empty:
                    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], errors='coerce')
                    events_df = events_df.dropna(subset=['timestamp'])
                    if events_df.empty:
                        continue
                    events_df = events_df.sort_values('timestamp')
                    
                    # Improved grouping: break on activity change OR time gap > 5 minutes
                    events_df['time_diff'] = events_df['timestamp'].diff().dt.total_seconds().fillna(0)
                    events_df['activity_changed'] = events_df['predicted_activity'] != events_df['predicted_activity'].shift()
                    events_df['time_gap'] = events_df['time_diff'] > 300  # 5 minute gap
                    events_df['group'] = (events_df['activity_changed'] | events_df['time_gap']).cumsum()
                    
                    for group_id, group_df in events_df.groupby('group'):
                        activity = group_df['predicted_activity'].iloc[0]
                        
                        # Skip NULL/empty activities
                        if pd.isna(activity) or activity is None or str(activity).strip() == '':
                            continue
                        
                        start_time = group_df['timestamp'].min()
                        end_time = group_df['timestamp'].max() + pd.Timedelta(seconds=10)
                        duration_minutes = (end_time - start_time).total_seconds() / 60.0
                        
                        # Safeguard: Skip unreasonably long segments (> 60 min) likely due to data gaps
                        if duration_minutes > 60:
                            continue

                        avg_confidence = group_df['confidence'].mean() if 'confidence' in group_df.columns else 1.0
                        event_count = len(group_df)
                        
                        # Check if this segment contains any corrected rows
                        has_correction = group_df['is_corrected'].any() if 'is_corrected' in group_df.columns else False
                        
                        cursor.execute('''
                            INSERT OR REPLACE INTO activity_segments 
                            (elder_id, room, activity_type, start_time, end_time, 
                             duration_minutes, avg_confidence, event_count, record_date,
                             is_corrected, correction_source)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            elder_id, room, str(activity),
                            start_time.strftime('%Y-%m-%d %H:%M:%S'),
                            end_time.strftime('%Y-%m-%d %H:%M:%S'),
                            round(duration_minutes, 2),
                            round(avg_confidence, 3) if not pd.isna(avg_confidence) else 1.0,
                            event_count,
                            record_date,
                            1 if has_correction else 0,
                            'manual' if has_correction else None
                        ))
            
            conn.commit()
            return True, f"Updated {updated_count} records and regenerated segments for {len(record_dates)} date(s)", updated_count
        
    except Exception as e:
        logger.error(f"Database error saving corrections: {e}")
        return False, f"Database error: {str(e)}", 0


ADL_CORRELATION_SENSOR_COLUMNS = ["co2", "humidity", "temperature", "motion", "sound", "light"]


def _normalize_room_token(value: str) -> str:
    return re.sub(r"[\s_]+", "", str(value or "").strip().lower())


@st.cache_data(ttl=300)
def list_adl_correlation_residents() -> list[str]:
    """Residents with either trained models or archived training files."""
    residents: set[str] = set()

    models_dir = current_dir / "models"
    if models_dir.exists():
        for entry in models_dir.iterdir():
            if entry.is_dir():
                token = str(entry.name or "").strip()
                if token:
                    residents.add(token)

    parse_elder_id = None
    try:
        from process_data import get_elder_id_from_filename as parse_elder_id  # type: ignore
    except Exception:
        parse_elder_id = None

    if ARCHIVE_DIR.exists():
        for file_path in ARCHIVE_DIR.rglob("*"):
            if not file_path.is_file():
                continue
            name = file_path.name
            lower = name.lower()
            if "train" not in lower:
                continue
            if file_path.suffix.lower() not in {".parquet", ".xlsx", ".xls"}:
                continue
            inferred = ""
            if parse_elder_id is not None:
                try:
                    inferred = str(parse_elder_id(name) or "").strip()
                except Exception:
                    inferred = ""
            if inferred:
                residents.add(inferred)

    return sorted(residents)


@st.cache_data(ttl=300)
def load_adl_sensor_training_dataset(resident_id: str) -> pd.DataFrame:
    """
    Aggregate all archived *train* files for one resident and return activity + sensor rows.
    """
    resident = str(resident_id or "").strip()
    if not resident or not ARCHIVE_DIR.exists():
        return pd.DataFrame()

    try:
        from utils.data_loader import load_sensor_data
    except Exception:
        return pd.DataFrame()

    combined_frames: list[pd.DataFrame] = []
    resident_key = resident.lower()
    allowed_ext = {".parquet", ".xlsx", ".xls"}

    for file_path in ARCHIVE_DIR.rglob("*"):
        if not file_path.is_file():
            continue
        lower = file_path.name.lower()
        if "train" not in lower:
            continue
        if resident_key not in lower:
            continue
        if file_path.suffix.lower() not in allowed_ext:
            continue

        try:
            room_data = load_sensor_data(file_path)
        except Exception:
            continue
        if not isinstance(room_data, dict):
            continue

        for room_name, room_df in room_data.items():
            if not isinstance(room_df, pd.DataFrame) or room_df.empty:
                continue
            if "activity" not in room_df.columns:
                continue
            sensors_present = [c for c in ADL_CORRELATION_SENSOR_COLUMNS if c in room_df.columns]
            if not sensors_present:
                continue

            local = room_df[["activity"] + sensors_present].copy()
            local["activity"] = local["activity"].astype(str).str.strip().str.lower()
            local = local[local["activity"] != ""]
            if local.empty:
                continue

            for sensor_col in sensors_present:
                local[sensor_col] = pd.to_numeric(local[sensor_col], errors="coerce")
            local["room"] = _normalize_room_token(room_name)
            local["source_file"] = str(file_path.name)
            combined_frames.append(local)

    if not combined_frames:
        return pd.DataFrame()

    dataset = pd.concat(combined_frames, ignore_index=True)
    for sensor_col in ADL_CORRELATION_SENSOR_COLUMNS:
        if sensor_col not in dataset.columns:
            dataset[sensor_col] = np.nan

    valid_sensor_row = dataset[ADL_CORRELATION_SENSOR_COLUMNS].notna().any(axis=1)
    dataset = dataset.loc[valid_sensor_row].copy()
    return dataset

# --- UI ---
st.title("🧠 Beta_5 Model Studio & Data Center")
render_active_system_banner(show_stage_details=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📤 Data Export",
        "🏷️ Labeling Studio",
        "📊 Model Insights",
        "🏠 Household Overview",
        "⚙️ AI Configuration",
        "🔗 ADL Correlation",
        "📝 Audit Trail",
    ]
)

# ==========================================
# TAB 1: DATA EXPORT
# ==========================================
with tab1:
    st.header("On-Demand Data Export")
    
    # Resident Selection
    def get_residents():
        queries = [
            "SELECT elder_id FROM elders",
            "SELECT DISTINCT elder_id FROM adl_history",
            "SELECT DISTINCT elder_id FROM correction_history",
            "SELECT DISTINCT elder_id FROM training_history",
            "SELECT DISTINCT elder_id FROM model_training_history",
        ]
        resident_ids: set[str] = set()

        for query in queries:
            try:
                # Use a fresh connection for each query to avoid transaction-abort
                # cascades when one source table is missing.
                with get_dashboard_connection() as conn:
                    df = query_to_dataframe(conn, query)
                if 'elder_id' not in df.columns:
                    continue
                for elder in df['elder_id'].dropna().astype(str).tolist():
                    elder = elder.strip()
                    if elder:
                        resident_ids.add(elder)
            except Exception:
                continue

        # Fallback 1: model directories still indicate known residents.
        models_root = current_dir / "models"
        if models_root.exists():
            for resident_dir in models_root.iterdir():
                if resident_dir.is_dir():
                    resident_ids.add(resident_dir.name.strip())

        # Fallback 2: infer resident IDs from raw/archive filenames.
        try:
            from process_data import get_elder_id_from_filename
        except Exception:
            get_elder_id_from_filename = None

        if get_elder_id_from_filename is not None:
            for base_dir in (RAW_DIR, ARCHIVE_DIR):
                if not base_dir.exists():
                    continue
                for p in base_dir.rglob("*"):
                    if not p.is_file():
                        continue
                    if p.suffix.lower() not in {".parquet", ".xlsx", ".xls", ".csv"}:
                        continue
                    try:
                        inferred = str(get_elder_id_from_filename(p.name)).strip()
                    except Exception:
                        inferred = ""
                    if inferred:
                        resident_ids.add(inferred)

        return sorted(resident_ids)

    residents = get_residents()
    if not residents:
        st.warning("No residents found.")
    else:
        selected_resident = st.selectbox("Select Resident (Export)", residents)
        
        def _run_export_query(conn, sql: str, params: tuple):
            """Execute query with portable parameter styles across SQLite/PostgreSQL."""
            try:
                return query_to_dataframe(conn, sql, params)
            except Exception:
                if "?" in sql:
                    return query_to_dataframe(conn, sql.replace("?", "%s"), params)
                raise

        # Date Range
        @st.cache_data(ttl=60)
        def get_db_bounds(resident_id):
            try:
                with get_dashboard_connection() as conn:
                    q = "SELECT min(timestamp) AS min_ts, max(timestamp) AS max_ts FROM adl_history WHERE elder_id = ?"
                    bounds_df = _run_export_query(conn, q, (resident_id,))
                    if bounds_df.empty:
                        return None, None
                    min_raw = bounds_df.iloc[0].get("min_ts")
                    max_raw = bounds_df.iloc[0].get("max_ts")
                    if min_raw is not None and max_raw is not None:
                        min_ts = pd.to_datetime(min_raw, errors='coerce')
                        max_ts = pd.to_datetime(max_raw, errors='coerce')
                        if pd.isna(min_ts) or pd.isna(max_ts):
                            return None, None
                        return min_ts, max_ts
            except Exception as e:
                logger.warning(f"Export bounds lookup failed for {resident_id}: {e}")
            return None, None

        min_date, max_date = get_db_bounds(selected_resident)
        if min_date is not None and max_date is not None:
            st.info(f"Data Available: {min_date.date()} to {max_date.date()}")
            default_start = min_date.date()
            default_end = max_date.date()
        else:
            default_end = datetime.now().date()
            default_start = default_end - timedelta(days=7)
            st.warning(
                "Could not detect date bounds from `adl_history` for this resident. "
                "You can still export by selecting a manual date range."
            )

        d_start = st.date_input("Start", default_start)
        d_end = st.date_input("End", default_end)
        if d_start > d_end:
            st.error("Start date must be on or before end date.")
            d_start, d_end = d_end, d_start

        ex_type = st.radio("Type", ["Raw Activity Logs (Excel)", "Review Candidates (CSV)"], horizontal=True)

        if st.button("Generate Export", type="primary"):
            with st.spinner("Processing..."):
                try:
                    with get_dashboard_connection() as conn:
                        start_ts = datetime.combine(d_start, datetime.min.time())
                        end_ts = datetime.combine(d_end, datetime.max.time())

                        if ex_type == "Raw Activity Logs (Excel)":
                            q = "SELECT * FROM adl_history WHERE elder_id=? AND timestamp BETWEEN ? AND ? ORDER BY timestamp ASC"
                            df = _run_export_query(conn, q, (selected_resident, start_ts, end_ts))
                            if not df.empty:
                                buf = BytesIO()
                                with pd.ExcelWriter(buf) as writer:
                                    df.to_excel(writer, index=False)
                                st.download_button("Download Excel", buf.getvalue(), f"{selected_resident}_log.xlsx")
                                st.success(f"Generated {len(df)} records")
                            else:
                                st.warning("No data found for the selected resident/date range.")

                        else:
                            q = (
                                "SELECT * FROM adl_history WHERE elder_id=? AND timestamp BETWEEN ? AND ? "
                                "AND (activity_type='low_confidence' OR is_anomaly=1 OR confidence<0.6) "
                                "ORDER BY timestamp DESC"
                            )
                            df = _run_export_query(conn, q, (selected_resident, start_ts, end_ts))
                            if not df.empty:
                                st.download_button(
                                    "Download CSV",
                                    df.to_csv(index=False).encode(),
                                    f"{selected_resident}_review.csv",
                                )
                                st.success(f"Found {len(df)} candidates")
                            else:
                                st.info("No review candidates found for the selected range.")
                except Exception as e:
                    st.error(f"Export failed: {e}")
                    logger.exception("On-demand export failed.")

        # Danger Zone
        with st.expander("⚠️ Danger Zone: Manage Resident"):
            st.warning(f"Actions here perform PERMANENT DELETION for {selected_resident}.")
            
            # Option 1: Delete Specific Date
            st.markdown("#### 🗓️ Delete Specific Date")
            delete_date = st.date_input("Select Date to Delete", value=max_date.date() if max_date else datetime.now().date(), key="delete_date_input")
            
            if st.button(f"🗑️ Delete {delete_date} Data Only", key="delete_date_btn"):
                try:
                    with get_dashboard_connection() as conn:
                        cursor = conn.cursor()
                        date_str = delete_date.strftime('%Y-%m-%d')
                        
                        # Delete from all relevant tables for this specific date
                        tables = [
                            ('adl_history', 'record_date'),
                            ('activity_segments', 'record_date'),
                            ('household_behavior', "date(timestamp)"),
                            ('household_segments', "date(start_time)"),
                            ('sleep_analysis', 'record_date')
                        ]
                        
                        total_deleted = 0
                        for table, date_col in tables:
                            try:
                                if 'date(' in date_col:
                                    cursor.execute(f"DELETE FROM {table} WHERE elder_id = ? AND {date_col} = ?", (selected_resident, date_str))
                                else:
                                    cursor.execute(f"DELETE FROM {table} WHERE elder_id = ? AND {date_col} = ?", (selected_resident, date_str))
                                total_deleted += cursor.rowcount
                            except Exception:
                                pass  # Table might not exist
                        
                        conn.commit()
                        st.success(f"✅ Deleted {total_deleted} records for {date_str}. You can now re-upload data for this date.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Deletion failed: {e}")
            
            st.divider()
            
            # Option 2: Delete ALL Resident Data
            st.markdown("#### 🚨 Delete Entire Resident")
            if st.button(f"🗑️ Delete ALL {selected_resident} Data", type="secondary", key="delete_all_btn"):
                from elderlycare_v1_16.services.profile_service import ProfileService
                
                try:
                    svc = ProfileService()
                    if svc.delete_elder(selected_resident):
                        st.success(f"Deleted {selected_resident}. Please refresh.")
                        st.rerun()
                    else:
                        st.error("Deletion failed.")
                except Exception as e:
                    st.error(f"Error: {e}")


# ==========================================
# TAB 2: LABELING STUDIO
# ==========================================
with tab2:
    st.header("Correction Studio (Active Learning)")
    st.markdown("Load **prediction files**, review model outputs, correct mistakes, and retrain. Labels are pre-populated from model predictions.")
    
    # 1. File Selector - Show ALL files (input + training)
    # Training files can now be corrected to create Golden Samples
    file_type_filter = st.radio(
        "File Type", 
        ["📥 Input Files (Predictions)", "📚 Training Files (Ground Truth)"],
        horizontal=True,
        help="Select 'Training Files' to correct labels in your ground-truth data."
    )
    
    target_type = 'input' if "Input" in file_type_filter else 'train'
    archive_files = get_archive_files(file_type=target_type)
    
    if not archive_files:
        st.warning(f"No {target_type} files found in archive.")
    else:
        file_options = [f"{f['date']} | {f['filename']}" for f in archive_files]
        selected_file_idx = st.selectbox("Select Archived File", range(len(file_options)), format_func=lambda x: file_options[x], key="file_selector_idx")
        selected_file_info = archive_files[selected_file_idx]
        
        if target_type == 'train':
            st.warning("⚠️ **Correcting Ground Truth**: Changes here will override original training labels on next retrain.")
        
        st.info(f"Selected: {selected_file_info['path']}")
        
        # 2. Load Data Button
        if st.button("Load File Data"):
            try:
                from utils.data_loader import load_sensor_data
                # Load all data into memory (efficient for both Excel and Parquet)
                data_dict = load_sensor_data(selected_file_info['path'])
                
                st.session_state['loaded_data'] = data_dict
                st.session_state['loaded_file'] = selected_file_info
                st.session_state['sheet_names'] = list(data_dict.keys())
                
            except Exception as e:
                st.error(f"Failed to load file: {e}")

        # 3. Sheet & Data Editor
        if 'loaded_file' in st.session_state and st.session_state['loaded_file']['path'] == selected_file_info['path']:
            sheet = st.selectbox("Select Room/Sheet", st.session_state['sheet_names'])
            
            # Get dataframe from memory
            if 'loaded_data' in st.session_state:
                df = st.session_state['loaded_data'][sheet]
            else:
                # Fallback reload if session lost but 'loaded_file' persists (rare)
                from utils.data_loader import load_sensor_data
                data_dict = load_sensor_data(selected_file_info['path'])
                df = data_dict[sheet]
            
            # Check essential columns
            essential_cols = ['timestamp', 'activity_type'] if 'activity_type' in df.columns else []
            # If input file, usually has 'activity' column if it was used for training previously.
            # If it was an INPUT file, it might not have 'activity' filled (or empty).
            # But process_data adds 'predicted_activity'? No, process_data saves to DB. 
            # The ARCHIVE file is the ORIGINAL input.
            # So it might NOT have predictions in it unless we enabled saving predictions back to file (we didn't).
            # WAIT. If archive file is just raw sensor data, user needs to see PREDICTIONS to correct them.
            # Where are the predictions? In DB!
            
            # MERGE STRATEGY:
            # 1. Load Raw Data (DF)
            # 2. Load Predictions from DB for this time range
            # 3. Merge or Display side-by-side
            
            st.divider()
            st.subheader(f"Labeling: {sheet}")
            
            # Ensure timestamp conversion and normalize to 10-second intervals
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                if df.empty:
                    st.warning("No valid timestamps found in selected data.")
                    st.stop()
                # Normalize timestamps to 10-second intervals (match DB structure)
                df['timestamp'] = df['timestamp'].dt.floor('10s')
                
                # Extract record_date from filename (e.g. HK001_jessica_input_18dec2025.xlsx -> 2025-12-18)
                fname = Path(selected_file_info['path']).name
                elder_id_guess = fname.split('_')[0] + "_" + fname.split('_')[1]
                
                # Parse date from filename (e.g. "1jan2026", "18dec2025", "18dec 2025")
                import re
                date_match = re.search(r'(\d{1,2})([a-z]{3})\s?(\d{4})', fname.lower())
                if date_match:
                    day, mon, year = date_match.groups()
                    month_map = {'jan':'01','feb':'02','mar':'03','apr':'04','may':'05','jun':'06',
                                 'jul':'07','aug':'08','sep':'09','oct':'10','nov':'11','dec':'12'}
                    record_date = f"{year}-{month_map.get(mon,'01')}-{int(day):02d}"
                else:
                    # Fallback to Excel timestamp range
                    record_date = df['timestamp'].min().strftime('%Y-%m-%d')
                
                st.info(f"📅 Querying predictions for **{record_date}** (Elder: {elder_id_guess}, Room: {sheet.title()})")
                
                # Fetch Predictions using cached function
                preds = fetch_predictions(elder_id_guess, sheet.lower(), record_date)

                
                if not preds.empty:
                    preds['timestamp'] = pd.to_datetime(preds['timestamp'], errors='coerce')
                    preds = preds.dropna(subset=['timestamp'])
                    if preds.empty:
                        st.warning("Predictions had no valid timestamps after parsing.")
                        st.stop()
                    
                    # *** TIMEZONE NORMALIZATION (Using centralized utility) ***
                    from utils.time_utils import ensure_naive
                    df['timestamp'] = ensure_naive(df['timestamp'])
                    preds['timestamp'] = ensure_naive(preds['timestamp'])
                    
                    # *** ACTIVE LEARNING: Pre-populate 'activity' column from predictions ***
                    # Use merge_asof for fuzzy timestamp matching (parquet and DB timestamps differ slightly)
                    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
                    preds_sorted = preds.sort_values('timestamp').reset_index(drop=True)
                    merge_cols = [
                        'timestamp', 'predicted_activity', 'confidence', 'display_activity',
                        'predicted_top1_label', 'predicted_top1_prob',
                        'predicted_top2_label', 'predicted_top2_prob',
                        'low_confidence_threshold', 'is_low_confidence', 'low_confidence_hint_label'
                    ]
                    merge_cols = [c for c in merge_cols if c in preds_sorted.columns]
                    
                    merged = pd.merge_asof(
                        df_sorted,
                        preds_sorted[merge_cols],
                        on='timestamp',
                        tolerance=pd.Timedelta('30s'),
                        direction='nearest'
                    )
                    df = merged
                    df['activity'] = df['predicted_activity'].fillna('inactive')
                    df['ml_status'] = df.get('display_activity', df.get('predicted_activity')).fillna('')
                    df['ml_hint_top1'] = df.get('predicted_top1_label').fillna('')
                    df['ml_hint_top1_prob'] = df.get('predicted_top1_prob')
                    df['ml_hint_top2'] = df.get('predicted_top2_label').fillna('')
                    df['ml_hint_top2_prob'] = df.get('predicted_top2_prob')
                    df = df.drop(columns=[
                        'predicted_activity', 'display_activity',
                        'predicted_top1_label', 'predicted_top1_prob',
                        'predicted_top2_label', 'predicted_top2_prob',
                        'low_confidence_threshold', 'is_low_confidence',
                        'low_confidence_hint_label'
                    ], errors='ignore')
                    
                    match_count = (df['activity'] != 'inactive').sum()
                    st.success(f"✅ Pre-populated {match_count} predictions into 'activity' column. Just correct any mistakes!")
                    
                    # Visualization (only when predictions available)
                    st.subheader("1. Activity Timeline (Current Model Views)")
                    st.info("This chart shows what the AI currently predicts. Use it to spot gaps or 'Low Confidence' areas.")
                    
                    import altair as alt
                    
                    # Predictions Gantt
                    preds['end_time'] = preds['timestamp'] + pd.Timedelta(seconds=10)
                    preds['time'] = preds['timestamp'].dt.strftime('%H:%M:%S')
                    date_str = preds['timestamp'].min().strftime('%Y-%m-%d')
                    
                    # Filter out null/empty activities for cleaner display
                    preds_clean = preds[preds['predicted_activity'].notna() & (preds['predicted_activity'] != '')].copy()
                    
                    # Simple chart without selection
                    c = alt.Chart(preds_clean).mark_bar().encode(
                        x=alt.X('timestamp:T', axis=alt.Axis(format='%H:%M', title='Time')),
                        x2='end_time:T',
                        y=alt.Y('display_activity:N', title='Predicted Activity', sort='-x'),
                        color=alt.Color('display_activity:N', scale=alt.Scale(scheme='tableau20'), legend=None),
                        opacity=alt.value(0.9),
                        tooltip=[
                            alt.Tooltip('time:N', title='Time'),
                            alt.Tooltip('display_activity:N', title='Shown Label'),
                            alt.Tooltip('predicted_activity:N', title='Stored Label'),
                            alt.Tooltip('predicted_top1_label:N', title='Top-1 Suggestion'),
                            alt.Tooltip('predicted_top1_prob:Q', title='Top-1 Prob', format='.2f'),
                            alt.Tooltip('predicted_top2_label:N', title='Top-2 Suggestion'),
                            alt.Tooltip('predicted_top2_prob:Q', title='Top-2 Prob', format='.2f'),
                            alt.Tooltip('low_confidence_threshold:Q', title='Threshold', format='.2f'),
                            alt.Tooltip('confidence:Q', title='Confidence', format='.2f'),
                        ]
                    ).properties(
                        title=f"Activity Patterns - {date_str}",
                        height=250,
                        width='container'
                    ).configure_view(
                        strokeWidth=0
                    ).configure_axis(
                        labelFontSize=11,
                        titleFontSize=12
                    ).interactive()
                    
                    st.altair_chart(c, use_container_width=True)
                    st.caption("`low_confidence-<label>` means the model is uncertain, but `<label>` is its top suggestion for faster correction.")
                    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
                    
                elif 'activity' in df.columns:
                    # Training file or re-upload: use existing labels as base if no DB predictions exist
                    df['activity'] = df['activity'].fillna('inactive')
                    st.success("📚 Pre-populated labels from the original file. Just correct any mistakes below.")
                    
                    # Visualization for Training Files (Ground Truth Labels)
                    st.subheader("1. Activity Timeline (Ground Truth Labels)")
                    st.info("This chart shows the **original labels** from the training file. Correct any mistakes below.")
                    
                    import altair as alt
                    
                    # Prepare visualization data from df (training file has 'activity', not 'predicted_activity')
                    viz_df = df[['timestamp', 'activity']].copy()
                    viz_df['end_time'] = viz_df['timestamp'] + pd.Timedelta(seconds=10)
                    viz_df['time'] = viz_df['timestamp'].dt.strftime('%H:%M:%S')
                    date_str = viz_df['timestamp'].min().strftime('%Y-%m-%d') if not viz_df.empty else "Unknown Date"
                    
                    # Filter out null/empty activities for cleaner display
                    viz_clean = viz_df[viz_df['activity'].notna() & (viz_df['activity'] != '')].copy()
                    
                    # Simple chart for training file labels
                    c = alt.Chart(viz_clean).mark_bar().encode(
                        x=alt.X('timestamp:T', axis=alt.Axis(format='%H:%M', title='Time')),
                        x2='end_time:T',
                        y=alt.Y('activity:N', title='Activity Label', sort='-x'),
                        color=alt.Color('activity:N', scale=alt.Scale(scheme='tableau20'), legend=None),
                        opacity=alt.value(0.9),
                        tooltip=[alt.Tooltip('time:N', title='Time'), 'activity']
                    ).properties(
                        title=f"Ground Truth Labels - {date_str}",
                        height=250,
                        width='container'
                    ).configure_view(
                        strokeWidth=0
                    ).configure_axis(
                        labelFontSize=11,
                        titleFontSize=12
                    ).interactive()
                    
                    st.altair_chart(c, use_container_width=True)
                    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
                
                # Interactive Sensor Chart
                sensor_cols = [c for c in df.columns if c in ['co2', 'humidity', 'motion', 'temperature', 'sound', 'light']]
                if sensor_cols:
                    st.subheader("2. Sensor Data Context (Interactive)")
                
                    # 1. Select Sensors
                    selected_sensors = st.multiselect(
                        "Select Sensors to Visualize", 
                        sensor_cols, 
                        default=sensor_cols[:2] # Default to first 2 to save space/noise
                    )
                    
                    # Option to Show Denoised with adjustable parameters
                    show_denoised = st.checkbox("Apply Denoising (Preview Model Input)", value=True)
                    
                    if show_denoised:
                        from elderlycare_v1_16.config.settings import DEFAULT_DENOISING_WINDOW, DEFAULT_DENOISING_THRESHOLD
                        col_w, col_s = st.columns(2)
                        with col_w:
                            preview_window = st.slider("Window", 2, 7, DEFAULT_DENOISING_WINDOW, 
                                                       help="Samples in context window. Lower = preserves more detail.")
                        with col_s:
                            preview_sigma = st.slider("Sigma", 2.0, 5.0, DEFAULT_DENOISING_THRESHOLD, step=0.5,
                                                       help="Outlier threshold. Higher = less filtering.")
                    
                    if selected_sensors:
                        plot_df = df.copy()
                        if show_denoised:
                            # Apply filter to copy
                            try:
                                # Only denoise selected numeric sensors
                                denoise_targets = [c for c in selected_sensors if c in DEFAULT_SENSOR_COLUMNS]
                                if denoise_targets:
                                    from elderlycare_v1_16.preprocessing.noise import hampel_filter
                                    hampel_filter(plot_df, denoise_targets, window=preview_window, n_sigmas=preview_sigma, inplace=True)
                                    st.caption(f"✅ Applied Hampel Filter (Window={preview_window}, Sigma={preview_sigma}) to preview.")
                            except Exception as e:
                                st.warning(f"Denoising preview failed: {e}")

                        # Melt
                        # s_melt = plot_df.melt('timestamp', value_vars=selected_sensors, var_name='sensor', value_name='value')
                        
                        # Individual chart per sensor (for proper width alignment)
                        for sensor in selected_sensors:
                            sensor_data = plot_df[['timestamp', sensor]].copy()
                            sensor_data['time'] = sensor_data['timestamp'].dt.strftime('%H:%M:%S')
                            
                            chart = alt.Chart(sensor_data).mark_line(color='steelblue').encode(
                                x=alt.X('timestamp:T', axis=alt.Axis(format='%H:%M', title=None)),
                                y=alt.Y(f'{sensor}:Q', axis=alt.Axis(title=sensor.capitalize()), scale=alt.Scale(zero=False)),
                                tooltip=[alt.Tooltip('time:N', title='Time'), alt.Tooltip(f'{sensor}:Q', format='.2f')]
                            ).properties(
                                height=100
                            ).interactive()
                            
                            st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("Select sensors to view data.")

                # Editor
                st.divider()
                st.subheader("3. Correction Studio")
                
                # --- INITIALIZE WIDGET KEYS ---
                if 'batch_start' not in st.session_state:
                    st.session_state['batch_start'] = df['timestamp'].min().time()
                if 'batch_end' not in st.session_state:
                    st.session_state['batch_end'] = df['timestamp'].max().time()
                if 'batch_label' not in st.session_state:
                    st.session_state['batch_label'] = get_activity_labels()[3] if len(get_activity_labels()) > 3 else get_activity_labels()[0]

                # --- PROCESS TIMELINE SELECTION (Click-to-Fill) ---
                # REMOVED due to user request (complexity). Manual entry preferred.
                
                # --- BATCH CORRECTION MODE (Multi-Room) ---
                # Initialize correction queue in session state
                if 'correction_queue' not in st.session_state:
                    st.session_state.correction_queue = []
                
                with st.expander("⚡ Batch Labeling (Queue Multiple Corrections)", expanded=True):
                    st.markdown("**Add corrections to queue, then apply all at once.** Supports multiple rooms. Training runs only once at the end.")
                    
                    # Show current room context
                    st.caption(f"📍 Current Room: **{sheet.title()}**")

                    c1, c2, c3 = st.columns([2, 2, 2])
                    with c1:
                        # Widgets now use 'key' and we updated session_state directly above
                        b_start = st.time_input("Start Time", step=60, key="batch_start")
                    with c2:
                        b_end = st.time_input("End Time", step=60, key="batch_end")
                    with c3:
                        ACTIVITY_OPTIONS = get_activity_labels()
                        b_label = st.selectbox("Activity to Apply", ACTIVITY_OPTIONS, index=min(3, len(ACTIVITY_OPTIONS)-1), key="batch_label")
                    
                    # Queue buttons
                    col_queue, col_apply = st.columns(2)
                    
                    with col_queue:
                        if st.button("➕ Add to Queue", help="Add this correction to the queue. Training will happen when you Apply All."):
                            if b_start >= b_end:
                                st.error("❌ Start time must be before end time!")
                            else:
                                # === ENHANCED CONFLICT DETECTION (Beta 5.5) ===
                                has_conflict = False
                                conflict_msg = ""
                                
                                for existing in st.session_state.correction_queue:
                                    ex_room = existing.get('room', '').lower()
                                    ex_start = existing['start']
                                    ex_end = existing['end']
                                    ex_label = existing['label']
                                    
                                    # Check for time overlap
                                    if b_start < ex_end and ex_start < b_end:
                                        if ex_room == sheet.lower():
                                            # Same room overlap - ERROR
                                            has_conflict = True
                                            conflict_msg = f"❌ Same-room overlap in {sheet}: {ex_start.strftime('%H:%M')}-{ex_end.strftime('%H:%M')} ({ex_label})"
                                            break
                                        else:
                                            # DIFFERENT room overlap - WARNING (physical conflict)
                                            # Allow 'out' and 'unoccupied' to coexist across rooms
                                            NON_EXCLUSIVE_LABELS = {'out', 'unoccupied'}
                                            if b_label.lower() not in NON_EXCLUSIVE_LABELS and ex_label.lower() not in NON_EXCLUSIVE_LABELS:
                                                has_conflict = True
                                                conflict_msg = f"⚠️ Physical conflict: You cannot be in '{sheet.title()}' AND '{ex_room.title()}' at the same time ({ex_start.strftime('%H:%M')}-{ex_end.strftime('%H:%M')})"
                                                break
                                
                                if has_conflict:
                                    st.error(conflict_msg)
                                else:
                                    # Store as dict with room metadata
                                    new_correction = {
                                        'room': sheet,
                                        'start': b_start,
                                        'end': b_end,
                                        'label': b_label,
                                        'record_date': record_date,
                                        'elder_id': elder_id_guess
                                    }
                                    
                                    # Capture old activity for audit trail
                                    # We look at the dataframe to see what we are overwriting
                                    try:
                                        # Filter DF for this time range (using string comparison for safety)
                                        t_start_ts = pd.Timestamp.combine(datetime.strptime(record_date, "%Y-%m-%d").date(), b_start)
                                        t_end_ts = pd.Timestamp.combine(datetime.strptime(record_date, "%Y-%m-%d").date(), b_end)
                                        
                                        mask = (df['timestamp'] >= t_start_ts) & (df['timestamp'] <= t_end_ts)
                                        if mask.any() and 'activity' in df.columns:
                                            # Use the most frequent activity in this range as the "old" value
                                            old_acts = df.loc[mask, 'activity'].mode()
                                            if not old_acts.empty:
                                                new_correction['old_activity'] = old_acts[0]
                                    except Exception as e:
                                        print(f"Error capturing old activity: {e}")
                                    st.session_state.correction_queue.append(new_correction)
                                    st.success(f"✅ Added: [{sheet.title()}] {b_start.strftime('%H:%M')}-{b_end.strftime('%H:%M')} → {b_label}")
                                    st.rerun()
                    
                    with col_apply:
                        queue_count = len(st.session_state.correction_queue)
                        if st.button(f"⚡ Apply All & Train ({queue_count})", type="primary", 
                                    disabled=(queue_count == 0),
                                    help="Apply all queued corrections to DB and retrain model ONCE."):
                            st.session_state['apply_batch'] = True
                            st.rerun()

                    # Non-blocking preflight quality check for queued rooms.
                    if st.session_state.correction_queue:
                        queue_room_keys = sorted(
                            {
                                str(item.get("room", "")).strip().lower().replace(" ", "").replace("_", "")
                                for item in st.session_state.correction_queue
                                if str(item.get("room", "")).strip()
                            }
                        )
                        preflight = build_label_quality_preflight(
                            elder_id=str(elder_id_guess or ""),
                            room_filter_csv=",".join(queue_room_keys),
                        )
                        st.markdown("#### 🧪 Label Quality Preflight (Non-Blocking)")
                        score = preflight.get("score")
                        if score is None:
                            st.info("Label quality score unavailable for current queue.")
                        else:
                            lvl = "good" if score >= 80 else "warn" if score >= 60 else "bad"
                            st.markdown(
                                _severity_badge(f"Label Quality Score: {score:.1f}/100", lvl),
                                unsafe_allow_html=True,
                            )
                        st.caption(
                            f"Based on archived training data for queued room(s). "
                            f"Rows: {int(preflight.get('rows_total', 0) or 0)} | "
                            f"Files: {int(preflight.get('files_total', 0) or 0)}"
                        )
                        for msg in preflight.get("warnings", [])[:5]:
                            st.warning(f"Preflight warning: {msg}")
                        room_rows = preflight.get("rooms", [])
                        if room_rows:
                            room_df = pd.DataFrame(room_rows).rename(
                                columns={
                                    "room": "Room",
                                    "score": "Score",
                                    "rows": "Rows",
                                    "days": "Days",
                                    "unique_labels": "Unique Labels",
                                    "inactive_family_ratio": "Inactive Ratio",
                                    "minority_share": "Minority Share",
                                    "single_class_days": "Single-Class Days",
                                }
                            )
                            with st.expander("View Label Quality Details", expanded=False):
                                st.dataframe(room_df, hide_index=True, use_container_width=True)

                    st.markdown("#### 🎯 High-Impact Review Queue")
                    st.caption(
                        "Automatically mined hard-negative windows for risky rooms. "
                        "Use this to prioritize corrections that improve model learning fastest."
                    )
                    hn_summary = fetch_hard_negative_summary_cached(
                        elder_id=str(elder_id_guess or ""),
                        days=30,
                    )
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Open Suggestions", int(hn_summary.get("open", 0) or 0))
                    k2.metric("Queued for Review", int(hn_summary.get("queued", 0) or 0))
                    k3.metric("Reviewed", int(hn_summary.get("reviewed", 0) or 0))
                    k4.metric("Applied (7d)", int(hn_summary.get("applied_week", 0) or 0))
                    top_room_rows = hn_summary.get("top_rooms", []) or []
                    if top_room_rows:
                        top_df = pd.DataFrame(top_room_rows).rename(
                            columns={"room": "Room", "reason": "Blocker Type", "count": "Count"}
                        )
                        with st.expander("View Room-Wise Top Blockers", expanded=False):
                            st.dataframe(top_df, hide_index=True, use_container_width=True)
                    mine_col1, mine_col2, mine_col3 = st.columns([1, 1, 1])
                    with mine_col1:
                        if st.button(
                            "🔍 Refresh Suggestions",
                            key=f"refresh_hnq_{elder_id_guess}_{sheet}_{record_date}",
                        ):
                            stats = run_hard_negative_mining_now(
                                elder_id=str(elder_id_guess or ""),
                                record_date=str(record_date or ""),
                                risky_rooms_csv=os.getenv("HARD_NEGATIVE_RISKY_ROOMS", "bathroom,entrance,kitchen"),
                                top_k_per_room=max(1, _env_int("HARD_NEGATIVE_TOP_K_PER_ROOM", 3)),
                                min_block_rows=max(3, _env_int("HARD_NEGATIVE_MIN_BLOCK_ROWS", 6)),
                            )
                            if stats.get("error"):
                                st.error(f"Hard-negative mining failed: {stats['error']}")
                            else:
                                st.success(
                                    f"Suggestions updated. candidates={int(stats.get('candidates', 0))}, "
                                    f"inserted={int(stats.get('inserted', 0))}, updated={int(stats.get('updated', 0))}"
                                )
                            st.rerun()
                    with mine_col2:
                        auto_queue_n = st.number_input(
                            "Auto-Queue Top N",
                            min_value=1,
                            max_value=10,
                            value=3,
                            step=1,
                            key=f"hnq_top_n_{elder_id_guess}_{sheet}",
                        )
                    with mine_col3:
                        room_filter_norm = str(sheet or "").strip().lower().replace(" ", "").replace("_", "")
                        hnq_df = fetch_hard_negative_queue_cached(
                            elder_id=str(elder_id_guess or ""),
                            room_filter=room_filter_norm,
                            days=30,
                            status="open",
                            limit=30,
                        )
                        if st.button(
                            "➕ Auto-Queue Suggestions",
                            key=f"auto_queue_hnq_{elder_id_guess}_{sheet}_{record_date}",
                            disabled=hnq_df.empty,
                        ):
                            added = 0
                            queued_ids = []
                            view_df = hnq_df.sort_values(["score", "duration_minutes"], ascending=[False, False]).head(int(auto_queue_n))
                            for _, row in view_df.iterrows():
                                ts_start = pd.to_datetime(row["timestamp_start"], errors="coerce")
                                ts_end = pd.to_datetime(row["timestamp_end"], errors="coerce")
                                if pd.isna(ts_start) or pd.isna(ts_end):
                                    continue
                                label = str(row.get("suggested_label") or "").strip()
                                if not label:
                                    continue
                                candidate = {
                                    "room": sheet,
                                    "start": ts_start.time(),
                                    "end": ts_end.time(),
                                    "label": label,
                                    "record_date": str(row.get("record_date") or record_date),
                                    "elder_id": str(elder_id_guess or ""),
                                    "source": "hard_negative_queue",
                                    "hard_negative_id": int(row.get("id")),
                                    "old_activity": "hard_negative_suggestion",
                                }
                                duplicate = any(
                                    (
                                        str(x.get("room")) == str(candidate["room"])
                                        and str(x.get("record_date")) == str(candidate["record_date"])
                                        and getattr(x.get("start"), "strftime", lambda *_: str(x.get("start")))("%H:%M:%S")
                                        == candidate["start"].strftime("%H:%M:%S")
                                        and getattr(x.get("end"), "strftime", lambda *_: str(x.get("end")))("%H:%M:%S")
                                        == candidate["end"].strftime("%H:%M:%S")
                                        and str(x.get("label")) == str(candidate["label"])
                                    )
                                    for x in st.session_state.correction_queue
                                )
                                if duplicate:
                                    continue
                                st.session_state.correction_queue.append(candidate)
                                added += 1
                                queued_ids.append(int(row.get("id")))

                            if queued_ids:
                                try:
                                    with get_dashboard_connection() as conn:
                                        ensure_hard_negative_table(conn)
                                        mark_hard_negative_status(conn, queued_ids, status="queued")
                                        conn.commit()
                                    fetch_hard_negative_queue_cached.clear()
                                    fetch_hard_negative_summary_cached.clear()
                                except Exception as e:
                                    logger.warning(f"Failed to mark hard-negative rows queued: {e}")
                            st.success(f"Added {added} hard-negative suggestion(s) to queue.")
                            st.rerun()

                    if hnq_df.empty:
                        st.info("No open high-impact suggestions for this room yet.")
                    else:
                        preview_df = hnq_df.rename(
                            columns={
                                "timestamp_start": "Start",
                                "timestamp_end": "End",
                                "duration_minutes": "Duration (min)",
                                "reason": "Reason",
                                "score": "Priority Score",
                                "suggested_label": "Suggested Label",
                                "status": "Status",
                            }
                        )
                        show_cols = [c for c in ["id", "Start", "End", "Duration (min)", "Reason", "Priority Score", "Suggested Label", "Status"] if c in preview_df.columns]
                        st.dataframe(preview_df[show_cols], hide_index=True, use_container_width=True)
                    
                    # Display queue grouped by room
                    if st.session_state.correction_queue:
                        st.markdown("---")
                        st.markdown("**📋 Queued Corrections:**")
                        
                        # Group by room for display
                        from collections import defaultdict
                        queue_by_room = defaultdict(list)
                        for idx, item in enumerate(st.session_state.correction_queue):
                            queue_by_room[item.get('room', 'Unknown')].append((idx, item))
                        
                        for room_name, items in queue_by_room.items():
                            st.markdown(f"**🚪 {room_name.title()}**")
                            for idx, item in items:
                                col_item, col_delete = st.columns([4, 1])
                                with col_item:
                                    st.text(f"  • {item['start'].strftime('%H:%M')} - {item['end'].strftime('%H:%M')} → {item['label']}")
                                with col_delete:
                                    if st.button("🗑️", key=f"del_{idx}", help="Remove from queue"):
                                        st.session_state.correction_queue.pop(idx)
                                        st.rerun()
                        
                        if st.button("🗑️ Clear All", help="Remove all queued corrections"):
                            st.session_state.correction_queue = []
                            st.rerun()
                    else:
                        st.info("No corrections queued. Add time ranges above.")
                    
                    st.markdown("---")
                    # Option to train on ALL history
                    # Retrospective training is DANGEROUS for Training Files (overwrites ground truth with predictions)
                    is_training_file = (target_type == 'train')
                    if is_training_file:
                        st.warning("⚠️ **Retrospective Training Disabled**: Training files are ground truth. Re-predicting them would overwrite your manual labels.")
                        train_retro = False
                    else:
                        train_retro = st.checkbox("Train Retrospectively (All History)", value=True, help="If checked, model will be retrained on ALL available historical data for this resident.")
                    
                # Processing Logic for Batch Apply
                if st.session_state.get('apply_batch', False):
                    st.session_state['apply_batch'] = False # Reset flag
                    
                    if not st.session_state.correction_queue:
                        st.warning("Queue is empty!")
                    else:
                        # Safety: queue must contain corrections for exactly one resident.
                        # Prevents partial updates where DB writes use per-item elder_id but
                        # regeneration/retraining accidentally uses the current UI resident.
                        queue_elder_ids = {
                            str(item.get('elder_id', elder_id_guess)).strip()
                            for item in st.session_state.correction_queue
                        }
                        if len(queue_elder_ids) != 1:
                            st.error("❌ Queue contains multiple residents. Please clear the queue and apply one resident at a time.")
                            st.info(f"Detected residents: {', '.join(sorted(queue_elder_ids))}")
                            st.stop()

                        apply_elder_id = next(iter(queue_elder_ids))
                        try:
                            # Setup logging
                            logger = logging.getLogger('CorrectionStudio')
                            logger.setLevel(logging.DEBUG)
                            registry = ModelRegistry(str(current_dir))
                            champion_before = _snapshot_champions(registry, apply_elder_id)

                            # Run preflight once more at execution time and surface warnings (non-blocking).
                            queue_room_keys = sorted(
                                {
                                    str(item.get("room", "")).strip().lower().replace(" ", "").replace("_", "")
                                    for item in st.session_state.correction_queue
                                    if str(item.get("room", "")).strip()
                                }
                            )
                            run_preflight = build_label_quality_preflight(
                                elder_id=apply_elder_id,
                                room_filter_csv=",".join(queue_room_keys),
                            )
                            run_score = run_preflight.get("score")
                            if run_score is not None and float(run_score) < 60:
                                st.warning(
                                    f"⚠️ Label quality preflight is low ({float(run_score):.1f}/100). "
                                    "Training will continue, but model stability risk is high."
                                )
                            
                            progress_bar = st.progress(0, text="Starting batch workflow...")
                            
                            # Ensure timestamp column in current df
                            if df['timestamp'].dtype == 'object':
                                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                                df = df.dropna(subset=['timestamp'])
                                if df.empty:
                                    st.error("No valid timestamps found in current working data.")
                                    st.stop()
                            
                            # Track stats
                            total_rows_updated = 0
                            ranges_applied = 0
                            affected_rooms = set()
                            room_dates = {}  # normalized_room -> set of date strings
                            local_true_before = []
                            local_pred_before = []
                            correction_windows = []
                            
                            # Open DB connection ONCE for all updates
                            progress_bar.progress(10, text="Applying corrections to database...")
                            
                            with get_dashboard_connection() as conn:
                                from utils.room_utils import normalize_timestamp, normalize_room_name
                                import json
                                cursor = conn.cursor()
                                
                                # *** AUTO-REGISTER ELDER IF NOT EXISTS ***
                                # This prevents FK violation when inserting into correction_history
                                # Extract name from elder_id (e.g. "HK002_samuel" -> "Samuel")
                                elder_name = apply_elder_id.split('_')[1].title() if '_' in apply_elder_id else apply_elder_id
                                cursor.execute('''
                                    INSERT INTO elders (elder_id, full_name, created_at)
                                    VALUES (?, ?, CURRENT_TIMESTAMP)
                                    ON CONFLICT (elder_id) DO NOTHING
                                ''', (apply_elder_id, elder_name))

                                
                                # Process Queue
                                queue_len = len(st.session_state.correction_queue)
                                
                                for i, item in enumerate(st.session_state.correction_queue):
                                    # Unpack item
                                    q_room = item.get('room', sheet) # Default to current if missing
                                    q_start = item['start']
                                    q_end = item['end']
                                    q_label = item['label']
                                    q_date = item.get('record_date', df['timestamp'].iloc[0].date().isoformat())
                                    q_elder = item.get('elder_id', apply_elder_id)
                                    
                                    affected_rooms.add(normalize_room_name(q_room))
                                    normalized_room = normalize_room_name(q_room)
                                    
                                    # Track per-room dates for correct segment regeneration
                                    room_dates.setdefault(normalized_room, set()).add(q_date if isinstance(q_date, str) else q_date.isoformat())
                                    
                                    # Update Progress
                                    p = 5 + int(15 * (i / queue_len))
                                    progress_bar.progress(p, text=f"Applying {i+1}/{queue_len}: [{q_room}] {q_label}")
                                    
                                    # Prepare timings
                                    # Note: q_date is a string or date object
                                    if isinstance(q_date, str):
                                        base_date = datetime.strptime(q_date, "%Y-%m-%d").date()
                                    else:
                                        base_date = q_date
                                        
                                    t_start = datetime.combine(base_date, q_start)
                                    t_end = datetime.combine(base_date, q_end)
                                    t_start_str = normalize_timestamp(t_start)
                                    t_end_str = normalize_timestamp(t_end)

                                    # Snapshot pre-correction labels for corrected-window before/after policy evaluation.
                                    cursor.execute('''
                                        SELECT activity_type
                                        FROM adl_history
                                        WHERE elder_id = ?
                                          AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = ?
                                          AND timestamp BETWEEN ? AND ?
                                    ''', (q_elder, normalized_room, t_start_str, t_end_str))
                                    before_rows = cursor.fetchall()
                                    local_true_before.extend([str(q_label)] * len(before_rows))
                                    local_pred_before.extend([str(r[0]) for r in before_rows])
                                    correction_windows.append((q_elder, normalized_room, t_start_str, t_end_str, str(q_label)))
                                    
                                    # 1. Update In-Memory DataFrame (ONLY if this correction is for the CURRENT view)
                                    if q_room == sheet:
                                        mask = (df['timestamp'] >= t_start) & (df['timestamp'] <= t_end)
                                        if mask.any():
                                            df.loc[mask, 'activity'] = q_label
                                            
                                            # Update sensor features in DB based on CURRENT df data
                                            # This only works if we are looking at the room's data. 
                                            # For other rooms, we skip sensor feature update and rely on existing DB values.
                                            sensor_cols = ['motion', 'temperature', 'light', 'sound', 'co2', 'humidity']
                                            rows_to_update = df[mask]
                                            for idx, row in rows_to_update.iterrows():
                                                row_ts = normalize_timestamp(row['timestamp'])
                                                sensor_data = {}
                                                for col in sensor_cols:
                                                    if col in row and pd.notna(row[col]):
                                                        sensor_data[col] = float(row[col])
                                                if sensor_data:
                                                    conn.execute('''
                                                        UPDATE adl_history 
                                                        SET sensor_features = ?
                                                        WHERE elder_id = ? AND timestamp = ? 
                                                          AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = ?
                                                    ''', (json.dumps(sensor_data), q_elder, row_ts, normalized_room))
                                    
                                    # 2. Update DB Activity (Primary Correction)
                                    cursor.execute('''
                                        UPDATE adl_history 
                                        SET activity_type = ?, is_corrected = 1
                                        WHERE elder_id = ? 
                                          AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = ?
                                          AND timestamp BETWEEN ? AND ?
                                    ''', (q_label, q_elder, normalized_room, t_start_str, t_end_str))
                                    rows = cursor.rowcount
                                    total_rows_updated += rows
                                    
                                    # 3. Audit Trail
                                    cursor.execute('''
                                        INSERT INTO correction_history 
                                        (elder_id, room, timestamp_start, timestamp_end, old_activity, new_activity, rows_affected)
                                        VALUES (?, ?, ?, ?, ?, ?, ?)
                                    ''', (q_elder, q_room, t_start_str, t_end_str, item.get('old_activity', 'batch'), q_label, rows))

                                    # Close any overlapping hard-negative suggestions after correction is applied.
                                    try:
                                        ensure_hard_negative_table(conn)
                                        mark_hard_negative_applied(
                                            conn=conn,
                                            elder_id=str(q_elder),
                                            room=str(q_room),
                                            timestamp_start=str(t_start_str),
                                            timestamp_end=str(t_end_str),
                                        )
                                    except Exception as e:
                                        logger.debug(f"Could not mark hard-negative suggestions as applied: {e}")
                                    
                                    ranges_applied += 1
                                
                                conn.commit()
                                
                                # Regenerate segments for ALL affected rooms using correct per-room dates
                                progress_bar.progress(20, text=f"Regenerating segments for {len(affected_rooms)} rooms...")
                                from utils.segment_utils import regenerate_segments
                                
                                for room_to_regen in affected_rooms:
                                    # Use actual dates collected from queue items for this room
                                    dates_for_room = room_dates.get(room_to_regen, 
                                        room_dates.get(normalize_room_name(room_to_regen),
                                            {df['timestamp'].iloc[0].date().isoformat()}))
                                    for rec_date in dates_for_room:
                                        regenerate_segments(apply_elder_id, room_to_regen, rec_date, conn)
                                
                                conn.commit()
                                
                                # FIX: Clear all caches so new corrections appear immediately in Audit Trail
                                clear_all_caches()
                            
                            # Update Buffer (for current room)
                            st.session_state['df_buffer'] = df
                            
                            # Save File (Current Room Only)
                            # Using '_manual_' suffix to prevent background automation from picking it up
                            progress_bar.progress(25, text="Saving corrected file (current room)...")
                            new_filename = f"{apply_elder_id}_manual_train.xlsx"
                            save_path = RAW_DIR / new_filename
                            save_path.parent.mkdir(parents=True, exist_ok=True)
                            with pd.ExcelWriter(save_path) as writer:
                                df.to_excel(writer, sheet_name=sheet, index=False)
                            
                            # Train
                            pipeline = UnifiedPipeline(enable_denoising=True)
                            
                            # Define grandular progress callback (maps 0-100% of pipeline to 30-95% of UI bar)
                            def ui_progress_callback(percent, msg):
                                segment_start = 30
                                segment_end = 95
                                total_p = segment_start + int((segment_end - segment_start) * (percent / 100))
                                progress_bar.progress(total_p, text=f"🧠 {msg}")

                            # Safety check: Ensure the loaded UnifiedPipeline class supports the new progress_callback argument
                            import inspect
                            sig_files = inspect.signature(pipeline.train_from_files)
                            supports_callback = 'progress_callback' in sig_files.parameters
                            
                            if train_retro:
                                all_files = get_archive_files(resident_id=apply_elder_id)
                                # Filter for archived training/input files
                                train_files = [f['path'] for f in all_files if '_train' in f['filename'].lower() or '_input' in f['filename'].lower()]
                                # Add the fresh manual correction file if not already included
                                if str(save_path) not in train_files:
                                    train_files.append(str(save_path))
                                
                                if supports_callback:
                                    results, metrics = pipeline.train_from_files(
                                        train_files,
                                        apply_elder_id,
                                        progress_callback=ui_progress_callback,
                                        rooms=affected_rooms,
                                        training_mode="correction_fine_tune",
                                    )
                                    pipeline.repredict_all(apply_elder_id, ARCHIVE_DIR, progress_callback=ui_progress_callback, rooms=affected_rooms)
                                else:
                                    st.warning("🔄 Note: New progress bar features detected. **Please restart your Streamlit server** to enable granular feedback.")
                                    results, metrics = pipeline.train_from_files(
                                        train_files,
                                        apply_elder_id,
                                        rooms=affected_rooms,
                                        training_mode="correction_fine_tune",
                                    )
                                    pipeline.repredict_all(apply_elder_id, ARCHIVE_DIR, rooms=affected_rooms)
                            else:
                                if supports_callback:
                                    results, metrics = pipeline.train_from_files(
                                        [str(save_path)],
                                        apply_elder_id,
                                        progress_callback=ui_progress_callback,
                                        rooms=affected_rooms,
                                        training_mode="correction_fine_tune",
                                    )
                                    pipeline.repredict_all(
                                        apply_elder_id,
                                        ARCHIVE_DIR,
                                        progress_callback=ui_progress_callback,
                                        rooms=affected_rooms,
                                    )
                                else:
                                    st.warning("🔄 Note: New progress bar features detected. **Please restart your Streamlit server** to enable granular feedback.")
                                    results, metrics = pipeline.train_from_files(
                                        [str(save_path)],
                                        apply_elder_id,
                                        rooms=affected_rooms,
                                        training_mode="correction_fine_tune",
                                    )
                                    pipeline.repredict_all(apply_elder_id, ARCHIVE_DIR, rooms=affected_rooms)

                            # Evaluate corrected-window policy (local gain + global drop) for correction-triggered retrain.
                            local_true_after = []
                            local_pred_after = []
                            with get_dashboard_connection() as conn_eval:
                                eval_cursor = conn_eval.cursor()
                                for q_elder, normalized_room, t_start_str, t_end_str, q_label in correction_windows:
                                    eval_cursor.execute('''
                                        SELECT activity_type
                                        FROM adl_history
                                        WHERE elder_id = ?
                                          AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = ?
                                          AND timestamp BETWEEN ? AND ?
                                    ''', (q_elder, normalized_room, t_start_str, t_end_str))
                                    after_rows = eval_cursor.fetchall()
                                    local_true_after.extend([q_label] * len(after_rows))
                                    local_pred_after.extend([str(r[0]) for r in after_rows])

                            local_before_f1 = _compute_macro_f1_safe(local_true_before, local_pred_before)
                            local_after_f1 = _compute_macro_f1_safe(local_true_after, local_pred_after)
                            affected_room_keys = {_room_key(r) for r in affected_rooms}
                            global_before_f1 = _compute_global_macro_f1(champion_before, affected_room_keys)
                            champion_after = _snapshot_champions(registry, apply_elder_id)
                            global_after_f1 = _compute_global_macro_f1(champion_after, affected_room_keys)
                            corrected_window_report = _evaluate_corrected_window_policy(
                                local_before_f1=local_before_f1,
                                local_after_f1=local_after_f1,
                                global_before_f1=global_before_f1,
                                global_after_f1=global_after_f1,
                            )

                            rolled_back = []
                            deactivated = []
                            if corrected_window_report["decision"] == "FAIL":
                                rolled_back, deactivated = _rollback_to_snapshot(
                                    registry=registry,
                                    elder_id=apply_elder_id,
                                    room_keys=affected_room_keys,
                                    before_snapshot=champion_before,
                                )
                                logger.warning(
                                    f"Correction candidate rejected by corrected-window policy for {apply_elder_id}. "
                                    f"rollback={rolled_back}, deactivated={deactivated}, report={corrected_window_report}"
                                )
                            
                            # Done
                            avg_acc = 0
                            if metrics:
                                avg_acc = sum([m['accuracy'] for m in metrics]) / len(metrics)

                            decision = corrected_window_report["decision"]
                            artifact_path = _persist_correction_evaluation_artifact(
                                elder_id=apply_elder_id,
                                decision=decision,
                                avg_acc=avg_acc,
                                metrics=metrics,
                                corrected_window_report=corrected_window_report,
                                rolled_back=rolled_back,
                                deactivated=deactivated,
                                ranges_applied=ranges_applied,
                                total_rows_updated=total_rows_updated,
                                affected_rooms=affected_rooms,
                            )
                            
                            progress_bar.progress(100, text="✅ Batch complete!")
                            if decision == "FAIL":
                                st.error(
                                    f"❌ Retrain candidate rejected by corrected-window policy. "
                                    f"Local gain={corrected_window_report.get('local_gain')}, "
                                    f"Global drop={corrected_window_report.get('global_drop')}. "
                                    f"Artifact: {artifact_path}"
                                )
                            else:
                                st.balloons()
                                st.success(
                                    f"🎉 Success! Applied {ranges_applied} corrections across "
                                    f"{len(affected_rooms)} rooms ({total_rows_updated} rows). "
                                    f"Models retrained. Artifact: {artifact_path}"
                                )
                                if decision == "PASS_WITH_FLAG":
                                    st.warning(
                                        "⚠️ Correction accepted with flag: local gain below target "
                                        "but no harmful global regression detected."
                                    )
                            
                            # Clear queue and ALL caches to force fresh data on rerun
                            st.session_state.correction_queue = []
                            clear_all_caches()
                            
                            # Clear session state data so Activity Timeline re-fetches corrected data
                            if 'loaded_data' in st.session_state:
                                del st.session_state['loaded_data']
                            if 'df_buffer' in st.session_state:
                                del st.session_state['df_buffer']
                            if 'loaded_file' in st.session_state:
                                del st.session_state['loaded_file']
                            
                            st.rerun()

                        except Exception as e:
                            import traceback
                            st.error(f"Batch failed: {e}")
                            st.code(traceback.format_exc())
                            clear_all_caches()

                # Initialize session buffer if not present
                if 'df_buffer' not in st.session_state:
                     st.session_state['df_buffer'] = df
                else:
                     # Check if we switched files/sheets, reset buffer if needed?
                     # Simple check: timestamp match or size match
                     if len(st.session_state['df_buffer']) != len(df):
                          st.session_state['df_buffer'] = df
                
                # Use buffer for editor
                # Capture result back to buffer? 
                # st.data_editor returns the state WITH edits.
                
                editor_column_config = {
                    "activity": st.column_config.SelectboxColumn(
                        "Activity Label",
                        help="Select the correct activity",
                        width="medium",
                        options=ACTIVITY_OPTIONS,
                        required=False
                    )
                }
                editor_disabled_cols = []

                if 'ml_status' in st.session_state['df_buffer'].columns:
                    editor_column_config['ml_status'] = st.column_config.TextColumn(
                        "Model Output",
                        help="`low_confidence-<label>` means uncertain with top suggestion."
                    )
                    editor_disabled_cols.append('ml_status')

                if 'ml_hint_top1' in st.session_state['df_buffer'].columns:
                    editor_column_config['ml_hint_top1'] = st.column_config.TextColumn("Top-1 Suggestion")
                    editor_disabled_cols.append('ml_hint_top1')

                if 'ml_hint_top1_prob' in st.session_state['df_buffer'].columns:
                    editor_column_config['ml_hint_top1_prob'] = st.column_config.NumberColumn(
                        "Top-1 Prob",
                        format="%.2f"
                    )
                    editor_disabled_cols.append('ml_hint_top1_prob')

                if 'ml_hint_top2' in st.session_state['df_buffer'].columns:
                    editor_column_config['ml_hint_top2'] = st.column_config.TextColumn("Top-2 Suggestion")
                    editor_disabled_cols.append('ml_hint_top2')

                if 'ml_hint_top2_prob' in st.session_state['df_buffer'].columns:
                    editor_column_config['ml_hint_top2_prob'] = st.column_config.NumberColumn(
                        "Top-2 Prob",
                        format="%.2f"
                    )
                    editor_disabled_cols.append('ml_hint_top2_prob')

                edited_df = st.data_editor(
                    st.session_state['df_buffer'], 
                    num_rows="dynamic", 
                    use_container_width=True,
                    key="editor_key", # Key essential for state
                    column_config=editor_column_config,
                    disabled=editor_disabled_cols
                )
                
                # === AUTO-QUEUE FROM GRID EDITS (Beta 5.5) ===
                # Detect changes between buffer and edited_df
                if 'activity' in edited_df.columns and 'activity' in st.session_state['df_buffer'].columns:
                    buffer_activities = st.session_state['df_buffer']['activity'].tolist()
                    edited_activities = edited_df['activity'].tolist()
                    
                    # Find changed rows
                    changed_indices = []
                    for i, (old, new) in enumerate(zip(buffer_activities, edited_activities)):
                        if old != new and new is not None:
                            changed_indices.append((i, old, new))
                    
                    if changed_indices:
                        st.info(f"📝 **{len(changed_indices)} grid edit(s) detected.** Click below to add to queue.")
                        
                        if st.button("➕ Auto-Queue Grid Edits", key="auto_queue_btn"):
                            # Group consecutive edits with same new activity into ranges
                            edited_df['timestamp'] = pd.to_datetime(edited_df['timestamp'], errors='coerce')
                            edited_df = edited_df.dropna(subset=['timestamp'])
                            if edited_df.empty:
                                st.warning("No valid timestamps available in grid edits.")
                                st.stop()
                            added_count = 0
                            
                            # Simple approach: add each edit as individual queue item
                            # Group by activity to create ranges
                            current_range = None
                            
                            for idx, old_act, new_act in changed_indices:
                                ts = edited_df.iloc[idx]['timestamp']
                                
                                if current_range is None or current_range['label'] != new_act:
                                    # Start new range
                                    if current_range is not None:
                                        # Save previous range
                                        st.session_state.correction_queue.append(current_range)
                                        added_count += 1
                                    
                                    current_range = {
                                        'room': sheet,
                                        'start': ts.time(),
                                        'end': (ts + pd.Timedelta(seconds=10)).time(),
                                        'label': new_act,
                                        'old_activity': old_act, # Capture what we are replacing
                                        'record_date': record_date,
                                        'elder_id': elder_id_guess
                                    }
                                else:
                                    # Extend current range
                                    current_range['end'] = (ts + pd.Timedelta(seconds=10)).time()
                            
                            # Don't forget the last range
                            if current_range is not None:
                                st.session_state.correction_queue.append(current_range)
                                added_count += 1
                            
                            # Update buffer with edits
                            st.session_state['df_buffer'] = edited_df.copy()
                            
                            st.success(f"✅ Added {added_count} correction range(s) to queue from grid edits.")
                            st.rerun()
                
                # Sync buffer (without auto-queue, just preserve manual edits)
                st.session_state['df_buffer'] = edited_df.copy()

            else:
                st.warning("No timestamp column found.")

# ==========================================
# TAB 3: MODEL INSIGHTS
# ==========================================
with tab3:
    st.header("Model Intelligence: Learned Sensor Patterns")
    st.markdown("Aggregated insights from **all training sessions**. This shows what patterns the deployed models have learned.")
    
    # Get residents with trained models
    models_dir = current_dir / "models"
    if not models_dir.exists():
        st.warning("No trained models found. Train models first to see insights.")
    else:
        available_residents = [d.name for d in models_dir.iterdir() if d.is_dir()]
        
        if not available_residents:
            st.warning("No trained models found.")
        else:
            selected_resident = st.selectbox("Select Resident", available_residents)
            
            if st.button("🔍 Show Model Insights", type="primary"):
                with st.spinner(f"Analyzing all training data for {selected_resident}..."):
                    try:
                        # Find ALL training files for this resident across all dates
                        all_training_data = {}  # {room: [dataframes]}
                        
                        if ARCHIVE_DIR.exists():
                            for date_dir in ARCHIVE_DIR.iterdir():
                                if not date_dir.is_dir():
                                    continue
                                    
                                # Search for both parquet and xlsx files (support migration)
                                found_files = list(date_dir.rglob("*train*.xlsx")) + list(date_dir.rglob("*train*.parquet"))
                                
                                for f in found_files:
                                    # Check if file belongs to this resident
                                    if selected_resident.lower() in f.name.lower():
                                        try:
                                            from utils.data_loader import load_sensor_data
                                            data_dict = load_sensor_data(f)
                                            
                                            for room_name, df in data_dict.items():
                                                if 'activity' not in df.columns:
                                                    continue
                                                
                                                # Store for aggregation
                                                if room_name not in all_training_data:
                                                    all_training_data[room_name] = []
                                                all_training_data[room_name].append(df)
                                                
                                        except Exception as e:
                                            st.warning(f"Skipped {f.name}: {e}")
                        
                        if not all_training_data:
                            st.error(f"No training data found for {selected_resident}. Upload training files to see insights.")
                        else:
                            st.success(f"📚 Aggregated {sum(len(dfs) for dfs in all_training_data.values())} training files across {len(all_training_data)} rooms")
                            
                            all_insights = []
                            
                            # Analyze each room
                            for room, dataframes in all_training_data.items():
                                # Concatenate all training data for this room
                                combined_df = pd.concat(dataframes, ignore_index=True)
                                
                                # Define sensor columns (exclude temporal features from baseline comparison)
                                # Temporal features (hour_sin, hour_cos, day_period) are used by model but not for pattern display
                                sensor_cols = ['motion', 'temperature', 'light', 'sound', 'co2', 'humidity']
                                available_sensors = [c for c in sensor_cols if c in combined_df.columns]
                                
                                if not available_sensors:
                                    continue
                                
                                # Group by activity and compute AGGREGATED statistics
                                activity_stats = combined_df.groupby('activity')[available_sensors].agg(['mean', 'std']).round(2)
                                
                                # Get overall baseline (inactive mean)
                                if 'inactive' in combined_df['activity'].values:
                                    baseline = combined_df[combined_df['activity'] == 'inactive'][available_sensors].mean()
                                else:
                                    baseline = combined_df[available_sensors].mean()
                                
                                # For each activity, determine which sensors are elevated
                                for activity in activity_stats.index:
                                    if activity == 'inactive':
                                        continue  # Skip inactive baseline
                                    
                                    activity_row = {
                                        'Room': room.title(),
                                        'Activity': activity,
                                        'Total Samples': len(combined_df[combined_df['activity'] == activity])
                                    }
                                    
                                    # Analyze each sensor
                                    elevated_sensors = []
                                    for sensor in available_sensors:
                                        mean_val = activity_stats.loc[activity, (sensor, 'mean')]
                                        baseline_val = baseline[sensor]
                                        
                                        # Check if sensor is significantly elevated (>20% increase)
                                        if mean_val > baseline_val * 1.2:
                                            pct_increase = ((mean_val - baseline_val) / baseline_val) * 100
                                            elevated_sensors.append(f"{sensor.upper()}↑{pct_increase:.0f}%")
                                        elif mean_val < baseline_val * 0.8:
                                            pct_decrease = ((baseline_val - mean_val) / baseline_val) * 100
                                            elevated_sensors.append(f"{sensor.upper()}↓{pct_decrease:.0f}%")
                                    
                                    activity_row['Learned Pattern'] = ', '.join(elevated_sensors) if elevated_sensors else 'No significant changes'
                                    all_insights.append(activity_row)
                            
                            if all_insights:
                                insights_df = pd.DataFrame(all_insights)
                                
                                st.markdown("---")
                                st.subheader("📊 Aggregated Model Knowledge")
                                
                                # Display as grouped table
                                for room in sorted(insights_df['Room'].unique()):
                                    with st.expander(f"🚪 {room}", expanded=True):
                                        room_data = insights_df[insights_df['Room'] == room][['Activity', 'Total Samples', 'Learned Pattern']]
                                        st.dataframe(room_data, use_container_width=True, hide_index=True)
                                        
                                        # Show sample count summary
                                        total_samples = room_data['Total Samples'].sum()
                                        st.caption(f"💡 This room's model trained on **{total_samples:,} total labeled examples** across all sessions")
                            else:
                                st.info("No activity patterns found. Ensure training files have 'activity' labels.")
                                
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # --- Training History Sub-Panel ---
    # st.sidebar.markdown("---")
    # st.sidebar.header("🛠️ Debug Info")
    
    # DB Connection Check
    db_type = "Unknown"
    if '_dashboard_adapter' in globals() and _dashboard_adapter: # Check if _dashboard_adapter is defined and not None
        db_type = "PostgreSQL (LegacyAdapter)"
    else:
        db_type = "SQLite (Fallback)"
    
    # st.sidebar.info(f"DB Connection: **{db_type}**")
    
    import sys
    # st.sidebar.text(f"Path: {sys.path[0]}")
    
    st.title("🏡 Elderly Care Dashboard") # This line seems misplaced if it's meant to be a global title. Assuming it's part of the user's intended insertion.
    st.markdown("---")
    st.subheader("🛠️ Model Training History")
    
    def get_training_history(elder_id):
        """Fetch training logs for an elder."""
        try:
            with get_dashboard_connection() as conn:
                query = """
                    SELECT timestamp, room, model_type, accuracy, samples_count, epochs, status, error_message,
                           data_start_time, data_end_time
                    FROM model_training_history
                    WHERE elder_id = ?
                    ORDER BY timestamp DESC
                """
                df = query_to_dataframe(conn, query, (elder_id,))
                if df.empty:
                    return df

                # Backward-compatible UI dedup: keep latest per logical run key.
                dedup_cols = ['room', 'model_type', 'status', 'data_start_time', 'data_end_time']
                available_cols = [c for c in dedup_cols if c in df.columns]
                if available_cols:
                    df = df.sort_values('timestamp', ascending=False).drop_duplicates(
                        subset=available_cols, keep='first'
                    )
                return df
        except Exception:
            return pd.DataFrame()
    
    # Use selected resident from above or default
    if 'available_residents' in dir() and available_residents:
        training_df = get_training_history(selected_resident)
        
        if training_df.empty:
            st.info("No training history found. Train models to see logs here.")
        else:
            with st.expander("View Training Logs", expanded=True):
                # Optional Filters
                rooms = training_df['room'].unique().tolist()
                selected_rooms = st.multiselect("Filter by Room", rooms, default=rooms, key="training_hist_rooms")
                
                filtered_df = training_df[training_df['room'].isin(selected_rooms)]
                
                # Create a formatted data range column if timestamps exist
                if 'data_start_time' in filtered_df.columns and 'data_end_time' in filtered_df.columns:
                    def format_range(row):
                        if pd.isna(row['data_start_time']) or pd.isna(row['data_end_time']):
                            return 'N/A'
                        try:
                            start = pd.to_datetime(row['data_start_time'])
                            end = pd.to_datetime(row['data_end_time'])
                            return f"{start.strftime('%d %b %H:%M')} - {end.strftime('%H:%M')}"
                        except:
                            return 'N/A'
                    filtered_df = filtered_df.copy()
                    filtered_df['data_range'] = filtered_df.apply(format_range, axis=1)
                    display_cols = ['timestamp', 'room', 'model_type', 'accuracy', 'samples_count', 'data_range', 'status']
                else:
                    display_cols = ['timestamp', 'room', 'model_type', 'accuracy', 'samples_count', 'status']
                
                st.dataframe(
                    filtered_df[display_cols] if all(c in filtered_df.columns for c in display_cols) else filtered_df,
                    column_config={
                        "timestamp": st.column_config.DatetimeColumn("Run Time", format="D MMM, HH:mm:ss"),
                        "accuracy": st.column_config.ProgressColumn(
                            "Raw Run Accuracy",
                            format="%.2f",
                            min_value=0,
                            max_value=1
                        ),
                        "samples_count": st.column_config.NumberColumn("Samples"),
                        "data_range": st.column_config.TextColumn("Training Data Range"),
                        "status": st.column_config.TextColumn("Status"),
                    },
                    hide_index=True,
                    use_container_width=True
                )
    else:
        st.info("Select a resident above to view training history.")

    st.markdown("---")
    st.subheader("📈 Model Health Check")
    st.caption("Checks live models on future data windows so you can verify update safety.")

    if 'available_residents' in dir() and available_residents:
        health_col1, health_col2, health_col3 = st.columns([2, 1, 1])
        selected_health_resident = health_col1.selectbox(
            "Resident",
            available_residents,
            index=available_residents.index(selected_resident) if 'selected_resident' in locals() and selected_resident in available_residents else 0,
            key="model_health_resident",
        )
        render_active_system_banner(context_label="Weekly View", show_stage_details=False)
        st.markdown("#### 📋 Weekly Report: ML Health Snapshot")
        st.caption(
            "Read-only summary of balanced score, transition quality, safety thresholds, and confidence."
        )
        weekly_snapshot = fetch_ml_health_snapshot(
            elder_id=selected_health_resident,
            lookback_runs=20,
            include_raw=False,
        )
        render_ml_health_snapshot_panel(weekly_snapshot, panel_scope="weekly", compact=False)
        st.markdown("---")

        render_active_system_banner(context_label="Shadow Comparison", show_stage_details=False)
        st.markdown("#### 🔍 Shadow Divergence Diagnostics")
        st.caption(
            "Compares legacy vs Beta 6 room-gate outcomes. Reason text is shown first; technical trace is expandable."
        )
        shadow_diag = fetch_shadow_divergence_monitor(
            elder_id=selected_health_resident,
            days=30,
            limit=60,
        )
        if shadow_diag.get("status") != "ok":
            st.info("No Phase 6 shadow divergence report is available yet for this resident.")
        else:
            diag_status = str(shadow_diag.get("phase6_status", "unknown"))
            diag_level = "good" if diag_status == "ok" else "warn" if diag_status == "watch" else "bad"
            st.markdown(
                _severity_badge(f"Shadow Compare: {diag_status.title()}", diag_level)
                + _severity_badge(
                    f"Divergence Rate: {float(shadow_diag.get('divergence_rate', 0.0)):.1%}",
                    diag_level,
                )
                + _severity_badge(
                    f"Unexplained Rate: {float(shadow_diag.get('unexplained_divergence_rate', 0.0)):.1%}",
                    "bad" if float(shadow_diag.get("unexplained_divergence_rate", 0.0)) > float(shadow_diag.get("unexplained_divergence_rate_max", 0.05)) else "neutral",
                ),
                unsafe_allow_html=True,
            )
            if shadow_diag.get("training_date"):
                st.caption(
                    f"Latest shadow report: {shadow_diag.get('training_date')} | "
                    f"Run: {shadow_diag.get('run_id') or 'N/A'}"
                )
            badges = shadow_diag.get("badges", [])
            if isinstance(badges, list) and badges:
                for badge in badges[:5]:
                    room_txt = str(badge.get("room") or "room").title()
                    reason_text = str(badge.get("reason_text") or "Divergence requires investigation.")
                    st.warning(f"{room_txt}: {reason_text}")
                    with st.expander(f"Technical trace: {room_txt}", expanded=False):
                        st.json(badge.get("technical_trace", {}))
            else:
                st.success("No room-level divergence badges in the latest shadow report.")

        st.markdown("---")

        st.markdown("#### ✅ Model Update Safety")
        st.caption(
            "Quick view of whether recent model updates were safe to go live. "
            "Use this section to spot blocked updates and common blockers."
        )
        monitor_cfg1, monitor_cfg2 = st.columns([1, 1])
        monitor_window_label = monitor_cfg1.selectbox(
            "Time Window",
            options=["Last 7 Days", "Last 30 Days", "Last 90 Days"],
            index=1,
            key="promotion_gate_window",
        )
        monitor_days_map = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90}
        monitor_days = monitor_days_map.get(monitor_window_label, 30)
        monitor_limit = monitor_cfg2.number_input(
            "Recent Runs to Show",
            min_value=10,
            max_value=200,
            value=60,
            step=10,
            key="promotion_gate_max_runs",
        )

        monitor = fetch_promotion_gate_monitor(
            elder_id=selected_health_resident,
            days=int(monitor_days),
            limit=int(monitor_limit),
        )

        if monitor.get("total_runs", 0) == 0:
            st.info("No recent training records found for this resident.")
        else:
            latest_delta = monitor.get("latest_delta")
            latest_delta_text = "No comparison yet"
            if latest_delta and latest_delta.get("delta_vs_previous") is not None:
                delta_value = float(latest_delta["delta_vs_previous"])
                delta_room = latest_delta.get("room", "unknown")
                if delta_value > 0:
                    latest_delta_text = f"Improved ({delta_room})"
                elif delta_value < 0:
                    latest_delta_text = f"Dropped ({delta_room})"
                else:
                    latest_delta_text = f"No change ({delta_room})"

            pass_rate_pct = monitor.get("wf_pass_rate_pct")
            pass_rate_text = f"{pass_rate_pct:.1f}%" if pass_rate_pct is not None else "N/A"
            top_reason = _friendly_gate_reason(monitor.get("top_failure_reason") or "")
            blocked_runs = int(monitor.get("wf_fail_runs", 0))
            overall_label, overall_level = _overall_monitor_status(pass_rate_pct, blocked_runs)

            latest_change_level = "neutral"
            if latest_delta and latest_delta.get("delta_vs_previous") is not None:
                delta_value = float(latest_delta["delta_vs_previous"])
                if delta_value > 0:
                    latest_change_level = "good"
                elif delta_value < 0:
                    latest_change_level = "bad"
                else:
                    latest_change_level = "warn"

            st.markdown(
                (
                    _severity_badge(f"Overall: {overall_label}", overall_level)
                    + _severity_badge(f"Safe Update Rate: {pass_rate_text}", overall_level if pass_rate_pct is not None else "warn")
                    + _severity_badge(f"Latest Change: {latest_delta_text}", latest_change_level)
                ),
                unsafe_allow_html=True,
            )

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Safe Update Rate", pass_rate_text)
            p2.metric("Latest Change", latest_delta_text)
            p3.metric("Most Common Blocker", top_reason)
            p4.metric("Blocked Runs", blocked_runs)

            latest_run = monitor.get("latest_run") or {}
            if latest_run:
                latest_accuracy = latest_run.get("accuracy")
                latest_accuracy_txt = f"{float(latest_accuracy):.3f}" if latest_accuracy is not None else "N/A"
                latest_run_id_txt = latest_run.get("run_id")
                st.caption(
                    f"Latest Run ID: {latest_run_id_txt if latest_run_id_txt is not None else 'N/A'} | "
                    f"Latest run: {latest_run.get('training_date', 'N/A')} | "
                    f"Result: {_friendly_run_status(latest_run.get('status', 'N/A'))} | "
                    f"Raw Training Run Accuracy: {latest_accuracy_txt}"
                )

            release_profile = monitor.get("release_gate_profile", {})
            if isinstance(release_profile, dict):
                st.markdown("#### 🧭 Pilot Gate Profile")
                profile_level = "good" if str(release_profile.get("evidence_profile", "")).startswith("pilot") else "warn"
                st.markdown(
                    _severity_badge(f"Evidence Profile: {release_profile.get('evidence_profile', 'unknown')}", profile_level)
                    + _severity_badge(
                        f"Bootstrap: {'ON' if bool(release_profile.get('bootstrap_enabled', False)) else 'OFF'}",
                        "good" if bool(release_profile.get("bootstrap_enabled", False)) else "warn",
                    )
                    + _severity_badge(
                        f"Strict Prior Drift Max: {float(release_profile.get('strict_prior_drift_max', 0.10)):.2f}",
                        "neutral",
                    ),
                    unsafe_allow_html=True,
                )
                rp1, rp2, rp3, rp4 = st.columns(4)
                rp1.metric("WF Min Train Days", int(release_profile.get("wf_min_train_days", 7) or 7))
                rp2.metric("WF Validation Days", int(release_profile.get("wf_valid_days", 1) or 1))
                rp3.metric("Bootstrap Phase-1 Max Day", int(release_profile.get("bootstrap_phase1_max_days", 7) or 7))
                rp4.metric("Bootstrap Max Day", int(release_profile.get("bootstrap_max_days", 14) or 14))

                tracker_rows = release_profile.get("release_threshold_tracker", [])
                if isinstance(tracker_rows, list) and tracker_rows:
                    with st.expander("View Effective Release Thresholds", expanded=False):
                        st.caption(
                            "Computed from backend/config/release_gates.json using latest available training-days evidence."
                        )
                        st.dataframe(pd.DataFrame(tracker_rows), hide_index=True, use_container_width=True)

            room_trends = monitor.get("room_trends", [])
            blocked_cards_latest = [
                row
                for row in room_trends
                if bool(row.get("evaluated_in_latest_run", False)) and not bool(row.get("latest_pass", False))
            ]
            blocked_cards_historical = [
                row
                for row in room_trends
                if not bool(row.get("evaluated_in_latest_run", False)) and not bool(row.get("latest_pass", False))
            ]
            if blocked_cards_latest:
                st.markdown("#### 🚦 Promotion Report Cards")
                st.caption("Simple summary of why a room update was paused in the latest run.")
                for item in blocked_cards_latest[:3]:
                    card = _build_plain_promotion_card(item)
                    st.error(f"❌ {card['room']}: {card['title']}")
                    st.write(card["reason"])
                    if card.get("score"):
                        st.caption(card["score"])
                    st.caption(card["incumbent"])
            if blocked_cards_historical:
                st.markdown("#### 🕘 Historical Blockers")
                st.caption(
                    "These rooms were not evaluated in the latest run, so their pause reason is from an older run."
                )
                for item in blocked_cards_historical[:3]:
                    st.warning(
                        f"{item.get('room', 'Unknown')}: last seen in run "
                        f"{item.get('latest_run_id', 'N/A')} ({item.get('latest_training_date', 'N/A')})"
                    )

            if room_trends:
                trend_df = pd.DataFrame(room_trends)
                trend_df["status_text"] = trend_df["latest_pass"].map(lambda v: "Passed" if bool(v) else "Blocked")
                trend_df["evaluated_in_latest_run"] = trend_df["evaluated_in_latest_run"].map(
                    lambda v: "Yes" if bool(v) else "No"
                )
                trend_df["latest_pass"] = trend_df["latest_pass"].map(lambda v: "Yes" if bool(v) else "No")
                trend_df["latest_reasons"] = trend_df["latest_reasons"].map(_friendly_reasons_join)
                trend_df = trend_df.rename(
                    columns={
                        "room": "Room",
                        "latest_run_id": "Run ID (Last Seen)",
                        "evaluated_in_latest_run": "Evaluated In Latest Run",
                        "status_text": "Status",
                        "latest_training_date": "Latest Training Time",
                        "latest_pass": "Passed Safety Check",
                        "latest_training_days": "Latest Training Days",
                        "latest_required_threshold": "Required Threshold",
                        "latest_threshold_day_bucket": "Threshold Day Bucket",
                        "latest_candidate_macro_f1_mean": "Latest WF Candidate F1",
                        "latest_candidate_accuracy_mean": "Latest WF Candidate Accuracy",
                        "latest_candidate_stability_accuracy_mean": "Latest Stability Score",
                        "latest_candidate_transition_macro_f1_mean": "Latest Transition Score",
                        "previous_candidate_macro_f1_mean": "Previous WF Candidate F1",
                        "previous_candidate_accuracy_mean": "Previous WF Candidate Accuracy",
                        "delta_vs_previous": "Change vs Previous",
                        "latest_champion_macro_f1_mean": "Current Champion WF F1",
                        "latest_reasons": "Pause Reason",
                    }
                )
                with st.expander("View Room-by-Room Details", expanded=False):
                    st.markdown(
                        "<small><b>Note</b>: WF = walk-forward future-window evaluation. "
                        "This is different from raw training accuracy/F1 shown in training logs. "
                        "<b>Evaluated In Latest Run = No</b> means the shown blocker is historical.</small>",
                        unsafe_allow_html=True,
                    )
                    st.dataframe(trend_df, hide_index=True, use_container_width=True)

            room_history_points = monitor.get("room_history_points", [])
            if isinstance(room_history_points, list) and room_history_points:
                history_df = pd.DataFrame(room_history_points)
                history_df["training_time"] = pd.to_datetime(history_df.get("training_date"), errors="coerce")
                history_df["sensor_event_time"] = pd.to_datetime(
                    history_df.get("sensor_event_time"), errors="coerce"
                )
                history_df = history_df.dropna(subset=["training_time"])
                if not history_df.empty:
                    st.markdown("#### 📈 F1/Accuracy Over Time")
                    st.caption("Hover any point to see exact run/time/metric values.")
                    time_axis_mode = st.radio(
                        "X-axis Time Source",
                        options=["Sensor Event Time (Chronology)", "Training Time (Run Chronology)"],
                        index=0,
                        horizontal=True,
                        key="promotion_trend_time_axis",
                    )
                    prefer_sensor_time = time_axis_mode.startswith("Sensor Event Time")
                    if prefer_sensor_time and int(history_df["sensor_event_time"].notna().sum()) == 0:
                        st.info("No sensor event timestamps found in training manifests yet; using training run time.")
                    axis_col = "sensor_event_time" if prefer_sensor_time else "training_time"
                    axis_title = "Sensor Event Time" if prefer_sensor_time else "Training Time"
                    history_df["plot_time"] = history_df[axis_col].where(
                        history_df[axis_col].notna(),
                        history_df["training_time"],
                    )
                    history_df = history_df.dropna(subset=["plot_time"])
                    room_options = sorted(history_df["room"].dropna().astype(str).unique().tolist())
                    default_rooms = room_options[: min(len(room_options), 6)]
                    selected_trend_rooms = st.multiselect(
                        "Rooms to Plot",
                        options=room_options,
                        default=default_rooms,
                        key="promotion_trend_rooms",
                    )
                    if selected_trend_rooms:
                        history_df = history_df[history_df["room"].isin(selected_trend_rooms)].copy()
                    history_df = history_df.sort_values(["plot_time", "run_id", "room"])

                    f1_df = history_df[history_df["candidate_macro_f1_mean"].notna()].copy()
                    acc_df = history_df[history_df["candidate_accuracy_mean"].notna()].copy()
                    threshold_df = history_df[history_df["required_threshold"].notna()].copy()

                    if not f1_df.empty:
                        f1_tooltips = [
                            alt.Tooltip("room:N", title="Room"),
                            alt.Tooltip("run_id:Q", title="Run ID"),
                            alt.Tooltip("sensor_event_time:T", title="Sensor Event Time"),
                            alt.Tooltip("training_time:T", title="Training Time"),
                            alt.Tooltip("candidate_macro_f1_mean:Q", title="WF Candidate F1", format=".6f"),
                            alt.Tooltip("required_threshold:Q", title="Required Threshold", format=".6f"),
                            alt.Tooltip("training_days:Q", title="Training Days", format=".1f"),
                            alt.Tooltip("pass:N", title="Passed"),
                        ]
                        f1_base = alt.Chart(f1_df).encode(
                            x=alt.X("plot_time:T", title=axis_title),
                            y=alt.Y("candidate_macro_f1_mean:Q", title="WF Candidate F1"),
                            color=alt.Color("room:N", title="Room"),
                        )
                        f1_chart = (
                            f1_base.mark_line()
                            + f1_base.mark_circle(size=70, filled=True)
                            + f1_base.mark_circle(size=220, opacity=0).encode(tooltip=f1_tooltips)
                        )
                        f1_chart = f1_chart.properties(height=260)
                        if not threshold_df.empty:
                            threshold_tooltips = [
                                alt.Tooltip("room:N", title="Room"),
                                alt.Tooltip("run_id:Q", title="Run ID"),
                                alt.Tooltip("sensor_event_time:T", title="Sensor Event Time"),
                                alt.Tooltip("training_time:T", title="Training Time"),
                                alt.Tooltip("required_threshold:Q", title="Required Threshold", format=".6f"),
                                alt.Tooltip("training_days:Q", title="Training Days", format=".1f"),
                            ]
                            threshold_base = alt.Chart(threshold_df).encode(
                                x=alt.X("plot_time:T", title=axis_title),
                                y=alt.Y("required_threshold:Q", title="WF Candidate F1"),
                                color=alt.Color("room:N", title="Room"),
                            )
                            threshold_chart = (
                                threshold_base.mark_line(point=False, strokeDash=[4, 4], opacity=0.55)
                                + threshold_base.mark_circle(size=64, filled=True, opacity=0.35)
                                + threshold_base.mark_circle(size=220, opacity=0).encode(tooltip=threshold_tooltips)
                            .properties(height=260)
                            )
                            st.altair_chart((f1_chart + threshold_chart).interactive(), use_container_width=True)
                        else:
                            st.altair_chart(f1_chart.interactive(), use_container_width=True)

                    if not acc_df.empty:
                        acc_tooltips = [
                            alt.Tooltip("room:N", title="Room"),
                            alt.Tooltip("run_id:Q", title="Run ID"),
                            alt.Tooltip("sensor_event_time:T", title="Sensor Event Time"),
                            alt.Tooltip("training_time:T", title="Training Time"),
                            alt.Tooltip("candidate_accuracy_mean:Q", title="WF Candidate Accuracy", format=".6f"),
                            alt.Tooltip("training_days:Q", title="Training Days", format=".1f"),
                            alt.Tooltip("pass:N", title="Passed"),
                        ]
                        acc_base = alt.Chart(acc_df).encode(
                            x=alt.X("plot_time:T", title=axis_title),
                            y=alt.Y("candidate_accuracy_mean:Q", title="WF Candidate Accuracy"),
                            color=alt.Color("room:N", title="Room"),
                        )
                        acc_chart = (
                            acc_base.mark_line()
                            + acc_base.mark_circle(size=70, filled=True)
                            + acc_base.mark_circle(size=220, opacity=0).encode(tooltip=acc_tooltips)
                        )
                        acc_chart = acc_chart.properties(height=260)
                        st.altair_chart(acc_chart.interactive(), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔧 Runtime Load Mode (Migration Coverage)")
        st.caption(
            "Shows how each room model is loaded in production. "
            "Target state is Shared Adapter for all active rooms."
        )
        runtime_modes = _summarize_runtime_load_modes(selected_health_resident)
        runtime_summary = runtime_modes.get("summary", {})
        rt_total = int(runtime_summary.get("total_rooms", 0))
        rt_shared = int(runtime_summary.get("shared_adapter_rooms", 0))
        rt_full = int(runtime_summary.get("full_model_rooms", 0))
        rt_fallback = int(runtime_summary.get("fallback_ready_rooms", 0))
        rt_not_ready = int(runtime_summary.get("not_ready_rooms", 0))

        coverage_pct = (100.0 * rt_shared / rt_total) if rt_total > 0 else 0.0
        coverage_level = "good" if coverage_pct >= 80 else "warn" if coverage_pct >= 40 else "bad"
        st.markdown(
            _severity_badge(f"Shared Adapter Coverage: {coverage_pct:.1f}%", coverage_level)
            + _severity_badge(f"Rooms Not Ready: {rt_not_ready}", "bad" if rt_not_ready > 0 else "good"),
            unsafe_allow_html=True,
        )
        rt1, rt2, rt3, rt4 = st.columns(4)
        rt1.metric("Total Rooms", rt_total)
        rt2.metric("Shared Adapter Ready", rt_shared)
        rt3.metric("Full Model Ready", rt_full)
        rt4.metric("Fallback Ready", rt_fallback)

        runtime_rows = runtime_modes.get("rows", [])
        if runtime_rows:
            runtime_df = pd.DataFrame(runtime_rows)
            runtime_df = runtime_df.drop(columns=["level"], errors="ignore")
            with st.expander("View Runtime Load Details", expanded=False):
                st.dataframe(runtime_df, hide_index=True, use_container_width=True)
        else:
            st.info("No room model artifacts found for runtime load mode check.")

        st.markdown("---")

        resident_model_dir = current_dir / "models" / selected_health_resident
        resident_model_rooms = []
        if resident_model_dir.exists():
            resident_model_rooms = sorted(
                [f.name.replace("_model.keras", "") for f in resident_model_dir.glob("*_model.keras") if "_v" not in f.name]
            )

        selected_health_room = health_col2.selectbox(
            "Room",
            resident_model_rooms if resident_model_rooms else ["(no_model)"],
            key="model_health_room",
        )
        lookback_days = health_col3.number_input(
            "Days to Include",
            min_value=7,
            max_value=3650,
            value=90,
            step=7,
            key="model_health_lookback_days",
        )

        default_min_train_days = max(2, _env_int("WF_MIN_TRAIN_DAYS", 7))
        default_valid_days = max(1, _env_int("WF_VALID_DAYS", 1))
        default_step_days = max(1, _env_int("WF_STEP_DAYS", 1))
        default_max_folds = max(1, _env_int("WF_MAX_FOLDS", 30))
        default_drift_threshold = _env_float("WF_DRIFT_THRESHOLD", 0.60)

        cfg1, cfg2, cfg3, cfg4 = st.columns(4)
        min_train_days = cfg1.number_input("Training Days", min_value=2, max_value=120, value=default_min_train_days, step=1, key="mh_min_train_days")
        valid_days = cfg2.number_input("Validation Days", min_value=1, max_value=30, value=default_valid_days, step=1, key="mh_valid_days")
        step_days = cfg3.number_input("Step Size (Days)", min_value=1, max_value=30, value=default_step_days, step=1, key="mh_step_days")
        max_folds = cfg4.number_input("Max Checks", min_value=1, max_value=200, value=default_max_folds, step=1, key="mh_max_folds")

        st.markdown("#### 🧭 Pre-Flight Feasibility")
        if selected_health_room == "(no_model)":
            st.info("Pick a room model to preview feasibility.")
        else:
            room_df_preview, room_df_preview_err = build_walk_forward_dataset(
                elder_id=selected_health_resident,
                room_name=selected_health_room,
                lookback_days=int(lookback_days),
            )
            if room_df_preview_err:
                st.warning(f"Could not preview feasibility: {room_df_preview_err}")
            else:
                observed_days = 0
                if room_df_preview is not None and not room_df_preview.empty and "timestamp" in room_df_preview.columns:
                    observed_days = int(
                        pd.to_datetime(room_df_preview["timestamp"], errors="coerce")
                        .dropna()
                        .dt.floor("D")
                        .nunique()
                    )
                feasibility = _estimate_fold_feasibility(
                    observed_days=observed_days,
                    min_train_days=int(min_train_days),
                    valid_days=int(valid_days),
                    step_days=int(step_days),
                    max_folds=int(max_folds),
                )
                badge = (
                    _severity_badge(
                        f"Ready: {feasibility['observed_days']} days >={feasibility['required_days']} required "
                        f"(expected folds: {feasibility['expected_folds']})",
                        "good",
                    )
                    if feasibility["ready"]
                    else _severity_badge(
                        f"Not ready: {feasibility['observed_days']} days < {feasibility['required_days']} required "
                        f"(expected folds: 0)",
                        "bad",
                    )
                )
                st.markdown(badge, unsafe_allow_html=True)
                st.caption(feasibility["message"])

        with st.expander("⚙️ Manual Tuning Knobs", expanded=False):
            st.caption(
                "Directly adjust key imbalance/noise knobs here. "
                "Use room scope to override only the selected room."
            )
            tuning_room_options = resident_model_rooms if resident_model_rooms else ["(no_model)"]
            default_tuning_idx = (
                tuning_room_options.index(selected_health_room)
                if selected_health_room in tuning_room_options
                else 0
            )
            selected_tuning_room = st.selectbox(
                "Tuning Room (for Selected Room scope)",
                options=tuning_room_options,
                index=default_tuning_idx,
                key="mh_tuning_room",
            )
            room_key = _normalize_room_key(selected_tuning_room) if selected_tuning_room != "(no_model)" else ""
            if selected_tuning_room != "(no_model)":
                st.caption(f"Current room for room-scoped tuning: `{selected_tuning_room}`")
            scope = st.radio(
                "Apply Scope",
                options=["Selected Room", "Global"],
                horizontal=True,
                index=0,
                key="mh_tune_scope",
            )

            st.markdown("##### Recommended Defaults")
            st.caption(
                "One-click profile for current pilot data. "
                "It keeps rare events learnable and reduces repetitive idle noise."
            )
            if st.button("⭐ Apply Recommended Defaults", key="mh_apply_recommended_defaults"):
                env_updates = _recommended_tuning_env_defaults()
                for k, v in env_updates.items():
                    os.environ[str(k)] = str(v)
                env_ok, env_info = _persist_env_updates_to_dotenv(
                    env_updates=env_updates,
                    dotenv_path=current_dir / ".env",
                )
                threshold_ok, threshold_msg = _apply_recommended_threshold_defaults(selected_health_resident)
                if env_ok:
                    st.success(f"Saved recommended tuning defaults to .env ({env_info}).")
                else:
                    st.warning(f"Applied defaults for this session only; .env save failed: {env_info}")
                if threshold_ok:
                    st.success(threshold_msg)
                else:
                    st.warning(f"Threshold defaults not fully applied: {threshold_msg}")
                st.rerun()

            active_policy = _active_training_policy()
            enable_minority_default = bool(active_policy.minority_sampling.enabled)

            minority_target_global = float(active_policy.minority_sampling.target_share)
            minority_target_room = minority_target_global
            if room_key:
                room_map = dict(active_policy.minority_sampling.target_share_by_room or {})
                if room_key in room_map:
                    try:
                        minority_target_room = float(room_map[room_key])
                    except Exception:
                        minority_target_room = minority_target_global

            minority_mult_global = int(active_policy.minority_sampling.max_multiplier)
            minority_mult_room = minority_mult_global
            if room_key:
                room_map = dict(active_policy.minority_sampling.max_multiplier_by_room or {})
                if room_key in room_map:
                    try:
                        minority_mult_room = int(float(room_map[room_key]))
                    except Exception:
                        minority_mult_room = minority_mult_global

            stride_global = int(active_policy.unoccupied_downsample.stride)
            stride_room = stride_global
            if room_key:
                room_map = dict(active_policy.unoccupied_downsample.stride_by_room or {})
                if room_key in room_map:
                    try:
                        stride_room = int(float(room_map[room_key]))
                    except Exception:
                        stride_room = stride_global

            t1, t2, t3, t4 = st.columns(4)
            tune_enable_minority = t1.checkbox(
                "Enable Minority Sampling",
                value=enable_minority_default,
                key="mh_tune_enable_minority",
            )
            tune_target_share = t2.slider(
                "Minority Target Share",
                min_value=0.05,
                max_value=0.40,
                value=float(minority_target_room if scope == "Selected Room" else minority_target_global),
                step=0.01,
                key="mh_tune_target_share",
            )
            tune_max_multiplier = t3.number_input(
                "Minority Max Multiplier",
                min_value=1,
                max_value=10,
                value=int(minority_mult_room if scope == "Selected Room" else minority_mult_global),
                step=1,
                key="mh_tune_max_multiplier",
            )
            tune_unocc_stride = t4.number_input(
                "Unoccupied Stride",
                min_value=1,
                max_value=20,
                value=int(stride_room if scope == "Selected Room" else stride_global),
                step=1,
                key="mh_tune_unocc_stride",
            )

            if st.button("💾 Save Tuning", key="mh_save_tuning"):
                if scope == "Selected Room" and not room_key:
                    st.warning("Pick a valid room in 'Tuning Room' before saving room-scoped tuning.")
                    st.stop()
                env_updates = {
                    "ENABLE_MINORITY_CLASS_SAMPLING": "true" if tune_enable_minority else "false",
                    "MINORITY_TARGET_SHARE": f"{float(tune_target_share):.2f}",
                    "MINORITY_MAX_MULTIPLIER": str(int(tune_max_multiplier)),
                    "UNOCCUPIED_DOWNSAMPLE_STRIDE": str(int(tune_unocc_stride)),
                }
                if scope == "Selected Room" and room_key:
                    existing_target_share = _policy_room_override_csv(
                        active_policy.minority_sampling.target_share_by_room
                    )
                    existing_max_multiplier = _policy_room_override_csv(
                        active_policy.minority_sampling.max_multiplier_by_room
                    )
                    existing_stride = _policy_room_override_csv(
                        active_policy.unoccupied_downsample.stride_by_room
                    )
                    env_updates["MINORITY_TARGET_SHARE_BY_ROOM"] = _with_room_override(
                        existing_target_share,
                        room_key,
                        f"{float(tune_target_share):.2f}",
                    )
                    env_updates["MINORITY_MAX_MULTIPLIER_BY_ROOM"] = _with_room_override(
                        existing_max_multiplier,
                        room_key,
                        str(int(tune_max_multiplier)),
                    )
                    env_updates["UNOCCUPIED_DOWNSAMPLE_STRIDE_BY_ROOM"] = _with_room_override(
                        existing_stride,
                        room_key,
                        str(int(tune_unocc_stride)),
                    )

                for k, v in env_updates.items():
                    os.environ[str(k)] = str(v)
                ok, info = _persist_env_updates_to_dotenv(
                    env_updates=env_updates,
                    dotenv_path=current_dir / ".env",
                )
                if ok:
                    st.success(f"Tuning saved to .env ({info}).")
                else:
                    st.warning(f"Tuning applied to session only; .env save failed: {info}")
                st.rerun()

            st.markdown("---")
            st.markdown("##### Decision Thresholds (All Rooms)")
            st.caption(
                "Edit inference thresholds for each room/class. "
                "Lower threshold = easier to trigger that class."
            )
            threshold_rows = _load_room_threshold_rows(selected_health_resident)
            if not threshold_rows:
                st.info("No threshold files found for this resident yet.")
            else:
                threshold_df = pd.DataFrame(threshold_rows).sort_values(["room", "class_id"]).reset_index(drop=True)
                edited_threshold_df = st.data_editor(
                    threshold_df,
                    use_container_width=True,
                    hide_index=True,
                    key="mh_threshold_editor",
                    disabled=["room", "class_id", "label"],
                    column_config={
                        "room": st.column_config.TextColumn("Room"),
                        "class_id": st.column_config.NumberColumn("Class ID"),
                        "label": st.column_config.TextColumn("Label"),
                        "threshold": st.column_config.NumberColumn(
                            "Threshold",
                            min_value=0.01,
                            max_value=0.99,
                            step=0.01,
                            format="%.2f",
                        ),
                    },
                )
                if st.button("💾 Save Thresholds (All Rooms)", key="mh_save_thresholds_all"):
                    ok, msg = _save_room_threshold_rows(
                        elder_id=selected_health_resident,
                        rows=edited_threshold_df.to_dict(orient="records"),
                    )
                    if ok:
                        st.success(msg)
                    else:
                        st.warning(f"Threshold save failed: {msg}")
                    st.rerun()

        if st.button("🧪 Run Health Check", key="run_model_health_eval", type="primary"):
            if selected_health_room == "(no_model)":
                st.warning("No deployed model found for selected resident.")
            else:
                with st.spinner("Running model health check..."):
                    try:
                        pipeline = UnifiedPipeline(enable_denoising=True)
                        loaded_rooms = pipeline.registry.load_models_for_elder(selected_health_resident, pipeline.platform)
                        if not loaded_rooms:
                            st.error("No models could be loaded for the selected resident.")
                        else:
                            from utils.room_utils import normalize_room_name

                            target_room = next(
                                (r for r in loaded_rooms if normalize_room_name(r) == normalize_room_name(selected_health_room)),
                                None
                            )
                            if target_room is None:
                                st.error(f"Selected room model '{selected_health_room}' not loaded.")
                            else:
                                room_df, data_err = build_walk_forward_dataset(
                                    elder_id=selected_health_resident,
                                    room_name=target_room,
                                    lookback_days=int(lookback_days),
                                )
                                if data_err:
                                    st.error(data_err)
                                else:
                                    processed = pipeline.platform.preprocess_with_resampling(
                                        room_df,
                                        target_room,
                                        is_training=True,
                                        apply_denoising=False,
                                    )

                                    if "activity_encoded" not in processed.columns:
                                        st.error("Preprocessing did not produce encoded labels for evaluation.")
                                    else:
                                        seq_length = get_room_config().calculate_seq_length(target_room)
                                        if len(processed) <= seq_length:
                                            st.error(
                                                f"Insufficient processed rows for sequenceing: {len(processed)} <= seq_length {seq_length}"
                                            )
                                        else:
                                            sensor_data = np.asarray(
                                                processed[pipeline.platform.sensor_columns].values,
                                                dtype=np.float32,
                                            )
                                            labels = processed["activity_encoded"].values.astype(np.int32)
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

                                            if "timestamp" in processed.columns:
                                                seq_timestamps = np.asarray(
                                                    pd.to_datetime(processed["timestamp"]).iloc[seq_length - 1:],
                                                    dtype="datetime64[ns]",
                                                )
                                            else:
                                                seq_timestamps = np.asarray(
                                                    pd.to_datetime(processed.index).to_series().iloc[seq_length - 1:],
                                                    dtype="datetime64[ns]",
                                                )
                                            if len(seq_timestamps) > len(X_seq):
                                                seq_timestamps = seq_timestamps[:len(X_seq)]

                                            splitter = TimeCheckpointedSplitter(
                                                min_train_days=int(min_train_days),
                                                valid_days=int(valid_days),
                                                step_days=int(step_days),
                                                max_folds=int(max_folds),
                                            )
                                            encoder = pipeline.platform.label_encoders.get(target_room)
                                            label_ids = list(range(len(getattr(encoder, "classes_", [])))) if encoder is not None else None
                                            report = evaluate_model(
                                                model=pipeline.platform.room_models[target_room],
                                                X_seq=X_seq,
                                                y_seq=y_seq,
                                                seq_timestamps=seq_timestamps,
                                                splitter=splitter,
                                                class_thresholds=pipeline.platform.class_thresholds.get(target_room, {}),
                                                labels=label_ids,
                                            )
                                            st.session_state["mh_last_eval"] = {
                                                "resident": selected_health_resident,
                                                "room": target_room,
                                                "lookback_days": int(lookback_days),
                                                "report": report,
                                            }

                                            summary = report.get("summary", {})
                                            num_folds = int(summary.get("num_folds", 0) or 0)
                                            if num_folds == 0:
                                                st.warning("No valid checks could be generated with current settings.")
                                            else:
                                                m1, m2, m3, m4, m5, m6 = st.columns(6)
                                                m1.metric("Checks", num_folds)
                                                m2.metric("WF Candidate F1", f"{float(summary.get('macro_f1_mean', 0.0)):.3f}")
                                                m3.metric("WF Candidate Accuracy", f"{float(summary.get('accuracy_mean', 0.0)):.3f}")
                                                m4.metric("Recall", f"{float(summary.get('macro_recall_mean', 0.0)):.3f}")
                                                st_mean = summary.get("stability_accuracy_mean")
                                                tr_mean = summary.get("transition_macro_f1_mean")
                                                m5.metric("Stability Score", "N/A" if st_mean is None else f"{float(st_mean):.3f}")
                                                m6.metric("Transition Score", "N/A" if tr_mean is None else f"{float(tr_mean):.3f}")

                                                folds_df = pd.DataFrame(report.get("folds", []))
                                                if not folds_df.empty:
                                                    folds_df["valid_end"] = pd.to_datetime(folds_df["valid_end"], errors="coerce")
                                                    trend_df = folds_df[
                                                        [
                                                            "fold_id",
                                                            "valid_end",
                                                            "macro_f1",
                                                            "accuracy",
                                                            "stability_accuracy",
                                                            "transition_macro_f1",
                                                        ]
                                                    ].sort_values("valid_end")

                                                    chart = (
                                                        alt.Chart(trend_df)
                                                        .mark_line(point=True)
                                                        .encode(
                                                            x=alt.X("valid_end:T", title="Validation Window End"),
                                                            y=alt.Y("macro_f1:Q", title="WF Candidate F1", scale=alt.Scale(domain=[0, 1])),
                                                            tooltip=["fold_id", "valid_end", "macro_f1", "accuracy"],
                                                        )
                                                        .properties(height=280)
                                                    )
                                                    st.altair_chart(chart, use_container_width=True)

                                                    drift_threshold = float(default_drift_threshold)
                                                    low_rows = trend_df[trend_df["macro_f1"] < drift_threshold]
                                                    if not low_rows.empty:
                                                        st.warning(
                                                            f"Stability warning: {len(low_rows)} check(s) are below target score {drift_threshold:.2f}."
                                                        )
                                                    else:
                                                        st.success(f"No stability alert: all checks are >= {drift_threshold:.2f}.")

                                                    st_min_threshold = _env_float("WF_MIN_STABILITY_ACCURACY", 0.99)
                                                    tr_min_threshold = _env_float("WF_MIN_TRANSITION_F1", 0.80)
                                                    st_low_rows = trend_df[
                                                        trend_df["stability_accuracy"].notna()
                                                        & (trend_df["stability_accuracy"] < float(st_min_threshold))
                                                    ]
                                                    tr_low_rows = trend_df[
                                                        trend_df["transition_macro_f1"].notna()
                                                        & (trend_df["transition_macro_f1"] < float(tr_min_threshold))
                                                    ]
                                                    if not st_low_rows.empty:
                                                        st.warning(
                                                            f"Stability guard warning: {len(st_low_rows)} check(s) below {float(st_min_threshold):.2f}."
                                                        )
                                                    if not tr_low_rows.empty:
                                                        st.warning(
                                                            f"Transition guard warning: {len(tr_low_rows)} check(s) below {float(tr_min_threshold):.2f}."
                                                        )

                                                    st.dataframe(
                                                        folds_df[
                                                            [
                                                                "fold_id",
                                                                "train_start",
                                                                "train_end",
                                                                "valid_start",
                                                                "valid_end",
                                                                "n_train",
                                                                "n_valid",
                                                                "macro_precision",
                                                                "macro_recall",
                                                                "macro_f1",
                                                                "accuracy",
                                                                "stability_accuracy",
                                                                "transition_macro_f1",
                                                                "stability_support",
                                                                "transition_support",
                                                            ]
                                                        ],
                                                        hide_index=True,
                                                        use_container_width=True,
                                                    )

                                            auto_suggestions = _derive_auto_tuner_suggestions(
                                                report=report,
                                                min_train_days=int(min_train_days),
                                                step_days=int(step_days),
                                                lookback_days=int(lookback_days),
                                            )
                                            st.markdown("#### 🛠️ Smart Suggestions")
                                            if not auto_suggestions:
                                                st.success("No urgent tuning changes detected from this run.")
                                            else:
                                                for idx, suggestion in enumerate(auto_suggestions):
                                                    st.warning(f"⚠️ {suggestion['title']}")
                                                    st.write(f"Diagnostic: {suggestion['diagnostic']}")
                                                    st.write(f"Suggestion: {suggestion['recommendation']}")
                                                    suggestion_env_base = {
                                                        str(k): str(v)
                                                        for k, v in (suggestion.get("apply_env") or {}).items()
                                                    }
                                                    suggestion_env = _apply_room_aware_suggestion_env(
                                                        env_updates=suggestion_env_base,
                                                        suggestion=suggestion,
                                                        target_room=target_room,
                                                    )
                                                    apply_cols = st.columns([1, 2])
                                                    with apply_cols[0]:
                                                        if st.button(
                                                            "Apply Fix",
                                                            key=f"mh_apply_fix_{idx}_{selected_health_resident}_{target_room}",
                                                        ):
                                                            for k, v in (suggestion.get("apply_ui") or {}).items():
                                                                st.session_state[k] = v
                                                            env_updates = {}
                                                            for env_key in suggestion_env.keys():
                                                                state_key = f"mh_sugg_env_{idx}_{selected_health_resident}_{target_room}_{env_key}"
                                                                if state_key in st.session_state:
                                                                    env_updates[str(env_key)] = str(st.session_state[state_key]).strip()
                                                                else:
                                                                    env_updates[str(env_key)] = str(suggestion_env[env_key]).strip()
                                                            for k, v in env_updates.items():
                                                                os.environ[str(k)] = str(v)
                                                            persisted_ok, persisted_info = _persist_env_updates_to_dotenv(
                                                                env_updates=env_updates,
                                                                dotenv_path=current_dir / ".env",
                                                            )
                                                            if env_updates and persisted_ok:
                                                                st.success(
                                                                    f"Applied fix and saved to .env ({persisted_info}) for future runs."
                                                                )
                                                            elif env_updates:
                                                                st.warning(
                                                                    "Applied only to current session. Could not save to .env: "
                                                                    f"{persisted_info}"
                                                                )
                                                            else:
                                                                st.success("Applied fix for this session.")
                                                            st.rerun()
                                                    with apply_cols[1]:
                                                        if suggestion_env:
                                                            st.caption("Environment updates (editable, saved to .env when possible):")
                                                            for env_key, env_val in suggestion_env.items():
                                                                st.text_input(
                                                                    env_key,
                                                                    value=str(env_val),
                                                                    key=f"mh_sugg_env_{idx}_{selected_health_resident}_{target_room}_{env_key}",
                                                                )
                    except Exception as e:
                        st.error(f"Model health check failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    else:
        st.info("No residents with trained models available for walk-forward evaluation.")

# ==========================================
# TAB 4: HOUSEHOLD OVERVIEW
# ==========================================
with tab4:
    st.header("🏠 Household Overview (Global Behavior)")
    
    st.info("This view aggregates data from ALL rooms to determine the overall state of the home (e.g. Empty, Active, Resting).")
    render_active_system_banner(context_label="Today View", show_stage_details=False)
    st.markdown("#### 🩺 Today: ML Health Snapshot")
    today_snapshot_residents = get_ml_snapshot_residents()
    if today_snapshot_residents:
        today_snapshot_resident = st.selectbox(
            "Resident (Today Snapshot)",
            today_snapshot_residents,
            key="ml_snapshot_today_resident",
        )
        today_snapshot = fetch_ml_health_snapshot(
            elder_id=today_snapshot_resident,
            lookback_runs=20,
            include_raw=False,
        )
        render_ml_health_snapshot_panel(today_snapshot, panel_scope="today", compact=True)
    else:
        st.info("No training history is available yet for Today snapshot.")
    st.markdown("---")
    
    # Grid Layout
    col_main, col_config = st.columns([2, 1])
    
    # Init Analyzer
    h_analyzer = HouseholdAnalyzer()
    
    with col_config:
        st.subheader("⚙️ Rule Configuration")
        with st.form("household_config_form"):
            current_config = h_analyzer.get_config()
            
            # Household Type Toggle (NEW)
            household_type = st.radio(
                "Household Type",
                options=['single', 'double'],
                index=0 if current_config.get('household_type', 'single') == 'single' else 1,
                help="Single = 1 elder. Double = 2 people (enables Conflict Resolution to filter out secondary person)."
            )
            
            enable_detection = st.checkbox("Enable Empty Home Detection", value=current_config.get('enable_empty_home_detection', True))
            
            silence_threshold = st.slider(
                "Silence Threshold (Minutes)", 
                min_value=5, 
                max_value=60, 
                value=current_config.get('empty_home_silence_threshold_min', 15),
                help="How long must the house be silent (ignoring ignored rooms) to confirm it is empty?"
            )
            
            # Fetch all rooms for multiselect
            all_rooms = []
            if DB_PATH.exists():
                with get_dashboard_connection() as conn:
                    rooms_df = query_to_dataframe(conn, "SELECT DISTINCT room FROM adl_history")
                    all_rooms = rooms_df['room'].tolist()
            
            # Filter out Entrance from options if desired, or keep it. 
            # Usually Entrance cannot be ignored as it's the trigger.
            # Convert config list to actual list
            default_ignored = current_config.get('empty_home_ignore_rooms', [])
            if not isinstance(default_ignored, list): default_ignored = []
            
            ignored_rooms = st.multiselect(
                "Ignore Rooms for Silence Check",
                options=[r for r in all_rooms if r != 'entrance'],
                default=[r for r in default_ignored if r in all_rooms],
                help="Rooms where activity should NOT prevent 'Empty Home' status (e.g. Basement, Garage)"
            )
            
            save_config = st.form_submit_button("💾 Save Configuration")
    
    if save_config:
        # Save to DB
        try:
            with get_dashboard_connection() as conn:
                conn.execute("INSERT OR REPLACE INTO household_config (key, value) VALUES (?, ?)", 
                             ('household_type', household_type))
                conn.execute("INSERT OR REPLACE INTO household_config (key, value) VALUES (?, ?)", 
                             ('enable_empty_home_detection', str(enable_detection).lower()))
                conn.execute("INSERT OR REPLACE INTO household_config (key, value) VALUES (?, ?)", 
                             ('empty_home_silence_threshold_min', str(silence_threshold)))
                conn.execute("INSERT OR REPLACE INTO household_config (key, value) VALUES (?, ?)", 
                             ('empty_home_ignore_rooms', json.dumps(ignored_rooms)))
                conn.commit()
            st.success("Configuration saved!")
        except Exception as e:
            st.error(f"Failed to save config: {e}")
            
    with col_main:
        st.subheader("Global History Timeline")
        
        # Date Selection (Synced with Session State or Local)
        # Using a fresh date picker for this tab 
        view_date = st.date_input("Select Date", value=datetime.today(), key="global_date_picker")
        
        if save_config:
            # Re-run analysis for this date
            with st.spinner("Recalculating Global State..."):
                # Need resident ID. Assuming single resident or first one.
                res_id = 'resident_01'
                if get_residents():
                    res_id = get_residents()[0]
                    
                h_analyzer.analyze_day(res_id, view_date.strftime('%Y-%m-%d'))
                # Force reload by rerun
                st.rerun()

        # Load Segments
        try:
             with get_dashboard_connection() as conn:
                 # Need resident ID
                 res_id = 'resident_01'
                 if get_residents():
                    res_id = get_residents()[0]
                    
                 query = """
                     SELECT state, start_time, end_time, duration_minutes 
                     FROM household_segments 
                     WHERE elder_id = ? AND date(start_time) = ?
                     ORDER BY start_time
                 """
                 segments_df = query_to_dataframe(conn, query, (res_id, view_date.strftime('%Y-%m-%d')))
                 
                 if not segments_df.empty:
                     # Color Mapping
                     domain = ['empty_home', 'home_active', 'home_quiet', 'social_interaction']
                     range_ = ['#d3d3d3', '#4daf4a', '#377eb8', '#e41a1c'] # Grey, Green, Blue, Red
                     
                     # Convert timestamps for human-readable tooltips
                     segments_df['start_time'] = pd.to_datetime(segments_df['start_time'], errors='coerce')
                     segments_df['end_time'] = pd.to_datetime(segments_df['end_time'], errors='coerce')
                     segments_df = segments_df.dropna(subset=['start_time', 'end_time'])
                     if segments_df.empty:
                         st.warning("No valid household segment timestamps for selected date.")
                         st.stop()
                     segments_df['start_display'] = segments_df['start_time'].dt.strftime('%H:%M')
                     segments_df['end_display'] = segments_df['end_time'].dt.strftime('%H:%M')
                     
                     chart = alt.Chart(segments_df).mark_bar().encode(
                         x=alt.X('start_time:T', title='Time', axis=alt.Axis(format='%H:%M')),
                         x2='end_time:T',
                         y=alt.Y('state:N', title=None, sort=['home_active', 'home_quiet', 'empty_home', 'social_interaction'], axis=alt.Axis(labelFontSize=11, labelLimit=150)),
                         color=alt.Color('state:N', scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(orient='top')),
                         tooltip=[
                              alt.Tooltip('state:N', title='State'),
                              alt.Tooltip('start_display:N', title='Start'),
                              alt.Tooltip('end_display:N', title='End'),
                              alt.Tooltip('duration_minutes:Q', title='Duration (min)', format='.1f')
                          ]
                     ).properties(
                         width='container',
                         height=250
                     ).configure_axis(
                         labelFontSize=11
                     ).interactive()
                     
                     st.altair_chart(chart, use_container_width=True)
                     
                     # Simple Stats
                     total_mins = 24 * 60
                     empty_mins = segments_df[segments_df['state'] == 'empty_home']['duration_minutes'].sum()
                     occupancy_rate = 100 - (empty_mins / total_mins * 100)
                     
                     m1, m2, m3 = st.columns(3)
                     m1.metric("Home Occupancy", f"{occupancy_rate:.1f}%")
                     m2.metric("Time Empty", f"{empty_mins/60:.1f} hrs")
                     m3.metric("Status Count", len(segments_df))
                     
                 else:
                     st.info(f"No global history found for {view_date}. Run analysis or load data.")
                     if st.button("Run Analysis Now"):
                         with st.spinner("Analyzing..."):
                             res_id = get_residents()[0] if get_residents() else 'resident_01'
                             h_analyzer.analyze_day(res_id, view_date.strftime('%Y-%m-%d'))
                             st.rerun()
                             
        except Exception as e:
            st.error(f"Error loading global history: {e}")


# ==========================================
# TAB 5: AI CONFIGURATION
# ==========================================
with tab5:
    st.header("⚙️ AI Configuration & Room Window Tuning")
    
    st.write("""
    Configure the **Sequence Context Window** for each room. 
    This determines how many samples the AI looks at to make a prediction.
    """)
    st.markdown("#### 🛡️ Admin & Audit: ML Health Snapshot")
    admin_snapshot_residents = get_ml_snapshot_residents()
    if admin_snapshot_residents:
        admin_snapshot_resident = st.selectbox(
            "Resident (Admin Snapshot)",
            admin_snapshot_residents,
            key="ml_snapshot_admin_resident",
        )
        admin_snapshot = fetch_ml_health_snapshot(
            elder_id=admin_snapshot_resident,
            lookback_runs=40,
            include_raw=False,
        )
        render_ml_health_snapshot_panel(admin_snapshot, panel_scope="admin", compact=False)
    else:
        st.info("No training history is available yet for Admin snapshot.")
    st.markdown("---")
    
    room_mgr = get_room_config()
    defaults = room_mgr._config.get("defaults", {})
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🏠 Room-Specific Settings")
        with st.form("room_config_form"):
            updated_windows = {}
            for room_key, settings in room_mgr.get_all_rooms().items():
                current_window = settings.get("sequence_time_window")
                note = settings.get("note", "")
                
                new_window = st.number_input(
                    f"Sequence Window: {room_key.replace('_', ' ').title()} (seconds)",
                    min_value=60,
                    max_value=7200,
                    value=current_window,
                    step=60,
                    help=f"{note} | Standard interval is 10s."
                )
                updated_windows[room_key] = new_window
            
            st.divider()
            
            # Retrain Option
            should_retrain = st.checkbox(
                "✅ Apply & Retrain Immediately", 
                value=False,
                help="Finds the most recent training file and retrains all models with new window sizes. Takes ~3 mins."
            )

            save_room_config = st.form_submit_button("💾 Save & Apply")
            
            if save_room_config:
                try:
                    # 1. Save Config
                    for r_key, r_window in updated_windows.items():
                        room_mgr.update_room_config(r_key, r_window)
                    st.success("Configuration saved!")
                    
                    # 2. Retrain if checked
                    if should_retrain:
                        with st.status("🔄 Retraining Intelligence Models...", expanded=True) as status:
                            st.write("🔍 Searching for latest training file...")
                            
                            # Find all files in archive/raw matching *train*.xlsx
                            search_paths = [
                                str(RAW_DIR / "*train*.xlsx"),
                                str(ARCHIVE_DIR / "**" / "*train*.xlsx")
                            ]
                            
                            found_files = []
                            for p in search_paths:
                                found_files.extend(glob.glob(p, recursive=True))
                                
                            if not found_files:
                                status.update(label="❌ No training files found!", state="error")
                                st.error("Please drop a file named '*train*' into data/raw/ manually.")
                            else:
                                # Get latest by modification time
                                latest_file = max(found_files, key=os.path.getmtime)
                                st.write(f"📂 Found: `{os.path.basename(latest_file)}`")
                                
                                st.write("🏗️ Initializing Pipeline...")
                                pipeline = UnifiedPipeline()
                                
                                # Extract Elder ID from filename
                                folder_name = os.path.basename(latest_file).split("_train")[0]
                                elder_id = folder_name if folder_name else "HK001_jessica"
                                st.write(f"👤 Elder ID: `{elder_id}`")
                                
                                st.write("🚀 Starting Training (this may take ~3 mins)...")
                                try:
                                    pipeline.train_and_predict(latest_file, elder_id)
                                    status.update(label="✅ Retraining Complete!", state="complete")
                                    st.success("Models for all rooms have been updated with new Sequence Windows.")
                                    st.balloons()
                                except Exception as e:
                                    status.update(label="❌ Training Failed", state="error")
                                    st.error(f"Error during training: {e}")
                                    st.code(str(e))

                except Exception as e:
                    st.error(f"Failed to process: {e}")

    with col2:
        st.subheader("ℹ️ System Info")
        st.info(f"""
        **Current Defaults:**
        - Interval: `{defaults.get('data_interval')}s`
        - Window: `{defaults.get('sequence_time_window')}s`
        """)
        
        st.warning("""
        **⚠️ ALERT on Retraining**
        
        Changing the sequence window size will make existing models **incompatible**.
        
        The **Apply & Retrain** checkbox will handle this for you automatically using your last uploaded dataset.
        """)
        
        if st.button("🔄 Reload Config"):
            room_mgr._load_config()
            st.rerun()

    # --- Activity Label Management ---
    st.divider()
    st.subheader("🏷️ Activity Label Management")
    st.caption("Configure the activity labels available in the Correction Studio dropdown.")
    
    current_labels = get_activity_labels()
    
    with st.expander("📝 Current Labels", expanded=False):
        st.code("\n".join(current_labels))
    
    col_add, col_remove = st.columns(2)
    
    with col_add:
        new_label = st.text_input("Add New Label", placeholder="e.g., reading")
        if st.button("➕ Add Label"):
            if new_label and new_label.strip():
                label = new_label.strip().lower().replace(" ", "_")
                if label not in current_labels:
                    current_labels.append(label)
                    save_activity_labels(current_labels)
                    st.success(f"Added: {label}")
                    st.rerun()
                else:
                    st.warning("Label already exists.")
            else:
                st.warning("Enter a label name.")
    
    with col_remove:
        if current_labels:
            label_to_remove = st.selectbox("Remove Label", current_labels, key="remove_label_select")
            if st.button("🗑️ Remove Label"):
                if label_to_remove in current_labels:
                    current_labels.remove(label_to_remove)
                    save_activity_labels(current_labels)
                    st.success(f"Removed: {label_to_remove}")
                    st.rerun()
    
    if st.button("🔄 Reset to Defaults"):
        save_activity_labels(DEFAULT_ACTIVITY_LABELS)
        st.success("Labels reset to defaults.")
        st.rerun()

# ==========================================
# TAB 6: ADL CORRELATION
# ==========================================
with tab6:
    st.header("🔗 ADL ↔ Sensor Correlation")
    st.markdown(
        "View how learned ADL labels align with sensor behavior "
        "(example: `shower -> humidity/temp/co2 up`)."
    )

    correlation_residents = list_adl_correlation_residents()
    if not correlation_residents:
        st.info("No residents with training files or trained models were found.")
    else:
        corr_ctrl1, corr_ctrl2 = st.columns([2, 1])
        selected_corr_resident = corr_ctrl1.selectbox(
            "Resident",
            correlation_residents,
            key="adl_corr_resident",
        )
        min_samples = int(
            corr_ctrl2.number_input(
                "Min Samples / Activity",
                min_value=10,
                max_value=5000,
                value=50,
                step=10,
                key="adl_corr_min_samples",
            )
        )

        correlation_dataset = load_adl_sensor_training_dataset(selected_corr_resident)
        if correlation_dataset.empty:
            st.info("No usable training rows with activity + sensor values were found for this resident.")
        else:
            room_values = sorted(
                [str(v) for v in correlation_dataset.get("room", pd.Series(dtype=str)).dropna().unique().tolist() if str(v).strip()]
            )
            selected_room = st.selectbox(
                "Room Filter",
                options=["All"] + room_values,
                key="adl_corr_room",
                format_func=lambda x: "All Rooms" if x == "All" else str(x).title(),
            )
            if selected_room == "All":
                scoped = correlation_dataset.copy()
            else:
                scoped = correlation_dataset[
                    correlation_dataset["room"] == _normalize_room_token(selected_room)
                ].copy()

            if scoped.empty:
                st.warning("No rows matched the selected room filter.")
            else:
                sensor_columns = [
                    c for c in ADL_CORRELATION_SENSOR_COLUMNS
                    if c in scoped.columns and scoped[c].notna().any()
                ]
                if not sensor_columns:
                    st.warning("No supported sensor columns found (co2/humidity/temperature/motion/sound/light).")
                else:
                    activity_counts = scoped["activity"].value_counts()
                    eligible = activity_counts[activity_counts >= min_samples]
                    if eligible.empty:
                        st.warning(
                            "No activities met the minimum sample threshold. "
                            "Lower the threshold to include more labels."
                        )
                        st.dataframe(
                            activity_counts.reset_index().rename(
                                columns={"index": "activity", "activity": "samples"}
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        activities = sorted(eligible.index.tolist())
                        if "inactive" in activities:
                            default_baseline = "inactive"
                        elif "unoccupied" in activities:
                            default_baseline = "unoccupied"
                        else:
                            default_baseline = activities[0]
                        baseline_activity = st.selectbox(
                            "Baseline Activity",
                            options=activities,
                            index=activities.index(default_baseline),
                            key="adl_corr_baseline",
                        )

                        scoped = scoped[scoped["activity"].isin(activities)].copy()
                        baseline_slice = scoped[scoped["activity"] == baseline_activity]
                        baseline_means = baseline_slice[sensor_columns].mean(numeric_only=True)
                        if baseline_means.isna().all():
                            baseline_means = scoped[sensor_columns].mean(numeric_only=True)

                        correlation_rows = []
                        for activity in activities:
                            activity_mask = scoped["activity"].eq(activity)
                            for sensor in sensor_columns:
                                sensor_series = pd.to_numeric(scoped[sensor], errors="coerce").replace(
                                    [np.inf, -np.inf],
                                    np.nan,
                                )
                                valid = sensor_series.notna()
                                if int(valid.sum()) < 10:
                                    continue

                                sensor_valid = sensor_series.loc[valid]
                                mask_valid = activity_mask.loc[valid]
                                corr_value = np.nan
                                if mask_valid.nunique() > 1 and sensor_valid.nunique() > 1:
                                    corr_value = float(mask_valid.astype(int).corr(sensor_valid))
                                    if not np.isfinite(corr_value):
                                        corr_value = np.nan

                                activity_values = sensor_valid.loc[mask_valid]
                                activity_mean = float(activity_values.mean()) if not activity_values.empty else np.nan
                                baseline_mean = float(baseline_means.get(sensor, np.nan))
                                if not np.isfinite(activity_mean):
                                    activity_mean = np.nan
                                if not np.isfinite(baseline_mean):
                                    baseline_mean = np.nan

                                delta = np.nan
                                pct_change = np.nan
                                if not np.isnan(activity_mean) and not np.isnan(baseline_mean):
                                    delta = activity_mean - baseline_mean
                                    if abs(baseline_mean) > 1e-9:
                                        pct_change = (delta / abs(baseline_mean)) * 100.0
                                if not np.isfinite(delta):
                                    delta = np.nan
                                if not np.isfinite(pct_change):
                                    pct_change = np.nan

                                correlation_rows.append(
                                    {
                                        "activity": activity,
                                        "sensor": sensor,
                                        "correlation": corr_value,
                                        "activity_mean": activity_mean,
                                        "baseline_mean": baseline_mean,
                                        "delta_vs_baseline": delta,
                                        "pct_change_vs_baseline": pct_change,
                                        "sample_count": int(activity_mask.sum()),
                                    }
                                )

                        correlation_df = pd.DataFrame(correlation_rows)
                        if correlation_df.empty:
                            st.info("Not enough numeric sensor coverage to compute correlations.")
                        else:
                            correlation_df = correlation_df.replace([np.inf, -np.inf], np.nan)
                            correlation_df["abs_correlation"] = correlation_df["correlation"].abs()

                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Rows Analyzed", f"{len(scoped):,}")
                            m2.metric("Activities", len(activities))
                            m3.metric("Sensors", len(sensor_columns))
                            m4.metric(
                                "Strong Links (|r| >= 0.35)",
                                int((correlation_df["abs_correlation"] >= 0.35).sum()),
                            )

                            st.caption(
                                "Correlation (r) compares a one-vs-rest activity indicator against sensor value. "
                                "Positive values mean higher sensor levels when the activity is present."
                            )

                            inspect_default = next((a for a in activities if a != baseline_activity), baseline_activity)
                            inspect_activity = st.selectbox(
                                "Inspect Activity Signature",
                                options=activities,
                                index=activities.index(inspect_default),
                                key="adl_corr_activity",
                            )
                            activity_view = correlation_df[
                                correlation_df["activity"] == inspect_activity
                            ].sort_values("abs_correlation", ascending=False)
                            activity_plot = activity_view.copy()
                            activity_plot["pct_change_plot"] = (
                                pd.to_numeric(activity_plot["pct_change_vs_baseline"], errors="coerce")
                                .replace([np.inf, -np.inf], np.nan)
                                .fillna(0.0)
                            )

                            positive_signature = activity_view[
                                (activity_view["pct_change_vs_baseline"] >= 10.0)
                                & (activity_view["correlation"] >= 0.15)
                            ].head(3)
                            if not positive_signature.empty:
                                signature_text = ", ".join(
                                    f"{row['sensor'].upper()} +{row['pct_change_vs_baseline']:.0f}%"
                                    for _, row in positive_signature.iterrows()
                                )
                                st.success(
                                    f"Learned signature for `{inspect_activity}`: {signature_text} "
                                    f"(vs `{baseline_activity}` baseline)."
                                )
                            else:
                                top_rows = activity_view.head(3)
                                if not top_rows.empty:
                                    signature_text = ", ".join(
                                        f"{row['sensor'].upper()} {row['pct_change_vs_baseline']:+.0f}%"
                                        for _, row in top_rows.iterrows()
                                        if not np.isnan(row["pct_change_vs_baseline"])
                                    )
                                    if signature_text:
                                        st.info(
                                            f"Top sensor shifts for `{inspect_activity}`: {signature_text} "
                                            f"(vs `{baseline_activity}` baseline)."
                                        )

                            heatmap = alt.Chart(correlation_df).mark_rect().encode(
                                x=alt.X("sensor:N", title="Sensor"),
                                y=alt.Y("activity:N", title="Activity"),
                                color=alt.Color(
                                    "correlation:Q",
                                    title="Correlation (r)",
                                    scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
                                ),
                                tooltip=[
                                    alt.Tooltip("activity:N", title="Activity"),
                                    alt.Tooltip("sensor:N", title="Sensor"),
                                    alt.Tooltip("correlation:Q", title="r", format=".3f"),
                                    alt.Tooltip("sample_count:Q", title="Samples"),
                                ],
                            )
                            st.altair_chart(heatmap, use_container_width=True)

                            delta_bar = alt.Chart(activity_plot).mark_bar().encode(
                                x=alt.X("sensor:N", title="Sensor"),
                                y=alt.Y(
                                    "pct_change_plot:Q",
                                    title=f"% Change vs {baseline_activity}",
                                ),
                                color=alt.condition(
                                    alt.datum.pct_change_plot >= 0,
                                    alt.value("#2ca02c"),
                                    alt.value("#d62728"),
                                ),
                                tooltip=[
                                    alt.Tooltip("sensor:N", title="Sensor"),
                                    alt.Tooltip("pct_change_plot:Q", title="% vs baseline", format=".1f"),
                                    alt.Tooltip("correlation:Q", title="r", format=".3f"),
                                    alt.Tooltip("sample_count:Q", title="Samples"),
                                ],
                            )
                            st.altair_chart(delta_bar, use_container_width=True)

                            top_links = correlation_df.sort_values(
                                ["abs_correlation", "sample_count"],
                                ascending=[False, False],
                            ).head(25)
                            st.markdown("**Top ADL ↔ Sensor Links**")
                            st.dataframe(
                                top_links[
                                    [
                                        "activity",
                                        "sensor",
                                        "correlation",
                                        "pct_change_vs_baseline",
                                        "sample_count",
                                    ]
                                ].rename(
                                    columns={
                                        "activity": "Activity",
                                        "sensor": "Sensor",
                                        "correlation": "Correlation (r)",
                                        "pct_change_vs_baseline": "% vs Baseline",
                                        "sample_count": "Samples",
                                    }
                                ),
                                hide_index=True,
                                use_container_width=True,
                            )

# ==========================================
# TAB 7: AUDIT TRAIL
# ==========================================
with tab7:
    st.header("📝 Correction Audit Trail")
    st.markdown("Track and review all manual corrections made to activity predictions. Supports 100+ elders with advanced filtering.")
    
    # --- Debug Info (removed from inside cached functions) ---
    with st.expander("🕵️‍♂️ Deep Debug", expanded=False):
        try:
            with get_dashboard_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT count(*) FROM correction_history")
                count = cursor.fetchone()[0]
                st.write(f"**Total Rows in 'correction_history' table**: {count}")
        except Exception as e:
            st.error(f"Database check failed: {e}")
    
    # --- Summary Dashboard ---
    st.subheader("📊 Summary Dashboard")
    summary = get_correction_summary()
    
    if 'error' in summary:
        st.warning(f"Could not load summary: {summary['error']}")
    elif summary.get('total', 0) == 0:
        st.info("No corrections recorded yet. Corrections made in Labeling Studio will appear here.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Corrections", summary.get('total', 0))
        with col2:
            st.metric("This Week", summary.get('this_week', 0))
        with col3:
            st.metric("Today", summary.get('today', 0))
        with col4:
            st.metric("Rows Modified", f"{summary.get('total_rows', 0):,}")
        
        col5, col6 = st.columns(2)
        with col5:
            st.info(f"**Most Corrected Elder:** {summary.get('top_elder', 'N/A')}")
        with col6:
            st.info(f"**Most Applied Activity:** {summary.get('top_activity', 'N/A')}")

    st.divider()
    st.subheader("🧪 Correction Retrain Evaluations")
    st.caption("Review PASS / PASS_WITH_FLAG / FAIL decisions and inspect persisted evaluation artifacts.")

    eval_col1, eval_col2 = st.columns([1, 1])
    with eval_col1:
        eval_days_label = st.selectbox(
            "Evaluation Window",
            options=["Today", "Last 7 Days", "Last 30 Days", "All Time"],
            index=1,
            key="eval_window",
        )
    with eval_col2:
        eval_elder = st.selectbox(
            "Resident (Evaluations)",
            options=["All"] + (get_available_filters()[0] or []),
            index=0,
            key="eval_elder",
        )

    eval_days_map = {"Today": 1, "Last 7 Days": 7, "Last 30 Days": 30, "All Time": ALL_TIME_DAYS}
    eval_days = eval_days_map.get(eval_days_label, 7)
    eval_elder_filter = None if eval_elder == "All" else eval_elder
    eval_df = fetch_correction_evaluation_history(elder_filter=eval_elder_filter, days=eval_days)

    if eval_df.empty:
        st.info("No correction retrain evaluations found for the selected filters.")
    else:
        eval_summary = summarize_correction_evaluation_decisions(eval_df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Runs", eval_summary["total"])
        c2.metric("PASS", eval_summary["PASS"])
        c3.metric("PASS_WITH_FLAG", eval_summary["PASS_WITH_FLAG"])
        c4.metric("FAIL", eval_summary["FAIL"])

        display_df = eval_df[['training_date', 'elder_id', 'decision', 'status', 'accuracy', 'local_gain', 'global_drop']].copy()
        display_df = display_df.rename(
            columns={
                'training_date': 'Training Time',
                'elder_id': 'Elder',
                'decision': 'Decision',
                'status': 'Status',
                'accuracy': 'Accuracy',
                'local_gain': 'Local Gain',
                'global_drop': 'Global Drop',
            }
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        export_df = eval_df[
            ['id', 'training_date', 'elder_id', 'decision', 'status', 'accuracy', 'local_gain', 'global_drop', 'artifact_path', 'metadata_obj']
        ].copy()
        export_df['metadata_json'] = export_df['metadata_obj'].apply(lambda x: json.dumps(x, default=str))
        export_df = export_df.drop(columns=['metadata_obj'])

        export_csv = export_df.to_csv(index=False)
        export_json = export_df.to_json(orient='records', date_format='iso', force_ascii=False, indent=2)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_prefix = f"correction_retrain_evals_{'all' if not eval_elder_filter else eval_elder_filter}_{eval_days}d_{ts}"

        artifact_bundle = []
        for _, e_row in export_df.iterrows():
            artifact_path = e_row.get('artifact_path')
            item = {
                "id": int(e_row.get('id')) if pd.notna(e_row.get('id')) else None,
                "elder_id": e_row.get('elder_id'),
                "training_date": str(e_row.get('training_date')),
                "decision": e_row.get('decision'),
                "status": e_row.get('status'),
                "artifact_path": artifact_path,
            }
            if artifact_path:
                p = Path(str(artifact_path))
                if p.exists():
                    try:
                        item["artifact"] = json.loads(p.read_text())
                        item["artifact_load_status"] = "loaded"
                    except Exception as e:
                        item["artifact"] = None
                        item["artifact_load_status"] = f"parse_error:{e}"
                else:
                    item["artifact"] = None
                    item["artifact_load_status"] = "missing_file"
            else:
                item["artifact"] = None
                item["artifact_load_status"] = "no_path"
            artifact_bundle.append(item)
        artifact_bundle_json = json.dumps(artifact_bundle, indent=2, default=str)

        dl_col1, dl_col2, dl_col3 = st.columns(3)
        with dl_col1:
            st.download_button(
                "⬇️ Download Eval CSV",
                data=export_csv,
                file_name=f"{export_prefix}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_eval_csv",
            )
        with dl_col2:
            st.download_button(
                "⬇️ Download Eval JSON",
                data=export_json,
                file_name=f"{export_prefix}.json",
                mime="application/json",
                use_container_width=True,
                key="download_eval_json",
            )
        with dl_col3:
            st.download_button(
                "⬇️ Download Artifact Bundle",
                data=artifact_bundle_json,
                file_name=f"{export_prefix}_artifact_bundle.json",
                mime="application/json",
                use_container_width=True,
                key="download_eval_artifact_bundle",
            )

        st.markdown("**Run Details**")
        for _, row in eval_df.head(20).iterrows():
            run_id = row.get('id')
            elder_id = row.get('elder_id')
            decision = row.get('decision')
            train_ts = row.get('training_date')
            with st.expander(f"Run #{run_id} | {elder_id} | {decision} | {train_ts}"):
                meta = row.get('metadata_obj') or {}
                corrected_report = meta.get('corrected_window_report', {})
                rollback = meta.get('rollback', {})
                artifact_path = row.get('artifact_path')

                st.write({
                    "decision": decision,
                    "status": row.get('status'),
                    "accuracy": row.get('accuracy'),
                    "local_gain": corrected_report.get('local_gain'),
                    "global_drop": corrected_report.get('global_drop'),
                    "artifact_path": artifact_path,
                    "rolled_back_rooms": rollback.get('rolled_back_rooms'),
                    "deactivated_rooms": rollback.get('deactivated_rooms'),
                })

                if artifact_path:
                    p = Path(str(artifact_path))
                    if p.exists():
                        try:
                            artifact_obj = json.loads(p.read_text())
                            st.json(artifact_obj)
                        except Exception as e:
                            st.warning(f"Artifact exists but could not be parsed: {e}")
                    else:
                        st.warning("Artifact file path recorded but file is missing.")

    st.divider()
    
    # --- Filters ---
    st.subheader("🔍 Filter Corrections")
    
    elders_list, rooms_list = get_available_filters()
    
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1, 1])
    
    with filter_col1:
        elder_options = ['All'] + elders_list
        selected_elder = st.selectbox("Elder", elder_options, key="audit_elder_filter")
    
    with filter_col2:
        room_options = ['All'] + rooms_list
        selected_room = st.selectbox("Room", room_options, key="audit_room_filter")
    
    with filter_col3:
        days_options = {'Last 7 Days': 7, 'Last 30 Days': 30, 'Last 90 Days': 90, 'All Time': 3650}
        # FIX: Default to 'All Time' (index 3) to show all corrections regardless of date
        selected_period = st.selectbox("Time Period", list(days_options.keys()), 
                                       index=3,  # Default to 'All Time'
                                       key="audit_period_filter")
        days = days_options[selected_period]
    
    with filter_col4:
        include_deleted = st.checkbox("Show Deleted", value=False, help="Include soft-deleted corrections in the list")
    
    # Apply filters button
    if st.button("🔄 Apply Filters", key="apply_audit_filters"):
        st.cache_data.clear()  # Clear cache to reload with new filters
    
    # --- Data Table ---
    st.subheader("📋 Correction History")
    
    history_df = fetch_correction_history_cached(
        elder_filter=selected_elder if selected_elder != 'All' else None,
        room_filter=selected_room if selected_room != 'All' else None,
        days=days,
        include_deleted=include_deleted
    )
    
    if history_df.empty:
        st.info("No correction records found for the selected filters.")
    else:
        # Add status column for display
        history_df['status'] = history_df['is_deleted'].apply(lambda x: '🗑️ Deleted' if x == 1 else '✅ Active')
        
        # Add selection checkbox column (only for active corrections)
        history_df['select'] = False
        
        # Format for display - include 'select' column and 'id' for deletion
        display_df = history_df[['select', 'id', 'corrected_at', 'elder_id', 'room', 'timestamp_start', 'timestamp_end', 
                                  'old_activity', 'new_activity', 'rows_affected', 'status']].copy()
        display_df.columns = ['Select', 'ID', 'Corrected At', 'Elder', 'Room', 'Start', 'End', 'From', 'To', 'Rows', 'Status']
        
        # Pagination for large datasets
        ROWS_PER_PAGE = 50
        total_rows = len(display_df)
        total_pages = max(1, (total_rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)
        
        if total_rows > ROWS_PER_PAGE:
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="audit_page")
            st.caption(f"Showing page {page} of {total_pages} ({total_rows} total records)")
            start_idx = (page - 1) * ROWS_PER_PAGE
            end_idx = start_idx + ROWS_PER_PAGE
            paginated_df = display_df.iloc[start_idx:end_idx].copy()
        else:
            paginated_df = display_df.copy()
            st.caption(f"Showing {total_rows} records")
        
        # Disable checkbox for already deleted items
        disabled_rows = paginated_df['Status'] == '🗑️ Deleted'
        
        # Display editable table with checkbox column
        edited_df = st.data_editor(
            paginated_df,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "🗑️",
                    help="Select to delete",
                    default=False,
                    width="small"
                ),
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Corrected At": st.column_config.DatetimeColumn("Corrected At", format="D MMM, HH:mm"),
                "Start": st.column_config.TextColumn("Start", width="medium"),
                "End": st.column_config.TextColumn("End", width="medium"),
                "Rows": st.column_config.NumberColumn("Rows", width="small"),
                "Status": st.column_config.TextColumn("Status", width="small"),
            },
            disabled=["ID", "Corrected At", "Elder", "Room", "Start", "End", "From", "To", "Rows", "Status"],
            hide_index=True,
            use_container_width=True,
            key="audit_table_editor"
        )
        
        # Get selected rows for deletion
        selected_ids = edited_df[edited_df['Select'] == True]['ID'].tolist()
        
        # Filter out already deleted ones
        active_selected = [id for id in selected_ids if history_df[history_df['id'] == id]['is_deleted'].values[0] != 1]
        
        # Buttons for Revert and Delete
        st.divider()
        col_revert, col_del, col_info = st.columns([1, 1, 2])
        
        with col_revert:
            action_disabled = len(active_selected) == 0
            if st.button(
                f"↩️ Revert Selected ({len(active_selected)})",
                type="primary",
                disabled=action_disabled,
                key="revert_selected_btn",
                help="Restore original activity and remove from training."
            ):
                try:
                    from elderlycare_v1_16.services.adl_service import ADLService
                    from utils.segment_utils import regenerate_segments
                    import sqlite3
                    
                    # We can reuse ADLService.delete_correction for soft delete, 
                    # but we also need to RESTORE old_activity manually.
                    # Or implement specific revert logic here.
                    
                    revert_count = 0
                    total_rows_reverted = 0
                    
                    with get_dashboard_connection() as conn:
                        cursor = conn.cursor()
                        
                        for correction_id in active_selected:
                            # 1. Get Correction Info
                            c_row = history_df[history_df['id'] == correction_id].iloc[0]
                            c_start = c_row['timestamp_start']
                            c_end = c_row['timestamp_end']
                            c_room = c_row['room']
                            c_old_act = c_row['old_activity']
                            c_elder = c_row['elder_id']
                            
                            # 2. Restore Old Activity
                            # Clean room name for matching
                            clean_room = c_room.lower().replace(" ", "").replace("_", "")
                            
                            cursor.execute('''
                                UPDATE adl_history
                                SET activity_type = ?, is_corrected = 0
                                WHERE elder_id = ?
                                  AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = ?
                                  AND timestamp BETWEEN ? AND ?
                            ''', (c_old_act, c_elder, clean_room, c_start, c_end))
                            
                            rows = cursor.rowcount
                            total_rows_reverted += rows
                            
                            # 3. Mark as Deleted
                            cursor.execute('''
                                UPDATE correction_history
                                SET is_deleted = 1, deleted_at = CURRENT_TIMESTAMP, deleted_by = 'revert_btn'
                                WHERE id = ?
                            ''', (correction_id,))
                            
                            conn.commit()
                            
                            # 4. Regenerate Segments
                            # Use date from start timestamp
                            if isinstance(c_start, str):
                                rec_date = c_start.split(" ")[0]
                            else:
                                rec_date = c_start.strftime("%Y-%m-%d")
                                
                            regenerate_segments(c_elder, c_room, rec_date, conn)
                            conn.commit()
                            
                            revert_count += 1
                            
                    st.success(f"✅ Reverted {revert_count} correction(s). Restored {total_rows_reverted} rows to original values.")
                    st.cache_data.clear()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Revert Failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        with col_del:
            if st.button(
                f"🗑️ Delete (Soft) ({len(active_selected)})", 
                type="secondary",
                disabled=action_disabled,
                key="delete_selected_btn",
                help="Remove from audit view and training, but keeps current DB data (doesn't revert)."
            ):
                try:
                    from elderlycare_v1_16.services.adl_service import ADLService
                    adl_svc = ADLService()
                    
                    total_reset = 0
                    success_count = 0
                    
                    for correction_id in active_selected:
                        # Soft delete only marks correction as deleted. 
                        # It usually RESTORES rows to is_corrected=0 but keeps valid activity? 
                        # Let's check adl_svc logic. usually it sets is_corrected=0.
                        result = adl_svc.delete_correction(int(correction_id), deleted_by='dashboard_user')
                        if result['success']:
                            total_reset += result['adl_rows_reset']
                            success_count += 1
                    
                    st.success(f"✅ Deleted {success_count} correction(s). Reset {total_reset} rows.")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        with col_info:
            if len(active_selected) > 0:
                st.info(f"**Revert**: Restores logic to BEFORE correction.\n**Delete**: Removes record of correction but keeps data.")
        
        # --- Export ---
        st.divider()
        st.subheader("📥 Export")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # Export filtered data
            csv_filtered = history_df.to_csv(index=False)
            st.download_button(
                "📄 Download Filtered (CSV)",
                data=csv_filtered,
                file_name=f"correction_history_filtered_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="export_filtered_audit"
            )
        
        with export_col2:
            # Export all data
            @st.cache_data(ttl=300)
            def get_full_history():
                if not DB_PATH.exists():
                    return pd.DataFrame()
                try:
                    with get_dashboard_connection() as conn:
                        return query_to_dataframe(conn, 'SELECT * FROM correction_history ORDER BY corrected_at DESC')
                except:
                    return pd.DataFrame()
            
            full_df = get_full_history()
            if not full_df.empty:
                csv_full = full_df.to_csv(index=False)
                st.download_button(
                    "📊 Download All History (CSV)",
                    data=csv_full,
                    file_name=f"correction_history_full_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="export_full_audit"
                )
