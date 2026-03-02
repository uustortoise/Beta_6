"""Shared Beta 6 evaluation metric helpers for Ops-facing health snapshots."""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def parse_room_override_map(
    raw: str,
    *,
    normalize_room_name_fn: Callable[[str], str],
) -> dict[str, str]:
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
        room_key = normalize_room_name_fn(room_raw)
        value_txt = str(value_raw).strip()
        if room_key and value_txt:
            result[room_key] = value_txt
    return result


def resolve_float_threshold_with_source(
    var_name: str,
    room_name: str,
    fallback_default: float,
    *,
    normalize_room_name_fn: Callable[[str], str],
    explicit_value: Any = None,
    env: Mapping[str, str] | None = None,
) -> dict:
    env_map = env if env is not None else os.environ
    room_key = normalize_room_name_fn(room_name)
    room_map = parse_room_override_map(
        str(env_map.get(f"{var_name}_BY_ROOM", "")),
        normalize_room_name_fn=normalize_room_name_fn,
    )

    source = "default"
    resolved_key = var_name
    value = _safe_float(explicit_value)
    if value is not None:
        source = "policy"

    if room_key in room_map:
        parsed = _safe_float(room_map.get(room_key))
        if parsed is not None:
            value = parsed
            source = "room override"
            resolved_key = f"{var_name}_{room_key.upper()}"
    elif env_map.get(var_name) is not None:
        parsed_env = _safe_float(env_map.get(var_name))
        if parsed_env is not None:
            value = parsed_env
            source = "env override"

    if value is None:
        value = float(fallback_default)
        source = "default"

    if source == "room override":
        source_file = "env:WF_*_BY_ROOM"
        editor_target = f"admin.env.{var_name}_BY_ROOM[{room_key}]"
    elif source == "env override":
        source_file = "env:WF_*"
        editor_target = f"admin.env.{var_name}"
    elif source == "policy":
        source_file = "backend/config/release_gates.json"
        editor_target = "admin.thresholds.walk_forward"
    else:
        source_file = "backend/ml/beta6/evaluation/evaluation_metrics.py"
        editor_target = "admin.thresholds.walk_forward_defaults"
    return {
        "value": float(value),
        "source": source,
        "resolved_key": resolved_key,
        "source_file": source_file,
        "editor_target": editor_target,
        "owner": "MLOps",
    }


def resolve_int_threshold_with_source(
    var_name: str,
    room_name: str,
    fallback_default: int,
    *,
    normalize_room_name_fn: Callable[[str], str],
    explicit_value: Any = None,
    env: Mapping[str, str] | None = None,
) -> dict:
    env_map = env if env is not None else os.environ
    room_key = normalize_room_name_fn(room_name)
    room_map = parse_room_override_map(
        str(env_map.get(f"{var_name}_BY_ROOM", "")),
        normalize_room_name_fn=normalize_room_name_fn,
    )

    source = "default"
    resolved_key = var_name
    value = _safe_int(explicit_value)
    if value is not None:
        source = "policy"

    if room_key in room_map:
        parsed = _safe_int(room_map.get(room_key))
        if parsed is not None:
            value = parsed
            source = "room override"
            resolved_key = f"{var_name}_{room_key.upper()}"
    elif env_map.get(var_name) is not None:
        parsed_env = _safe_int(env_map.get(var_name))
        if parsed_env is not None:
            value = parsed_env
            source = "env override"

    if value is None:
        value = int(fallback_default)
        source = "default"

    if source == "room override":
        source_file = "env:WF_*_BY_ROOM"
        editor_target = f"admin.env.{var_name}_BY_ROOM[{room_key}]"
    elif source == "env override":
        source_file = "env:WF_*"
        editor_target = f"admin.env.{var_name}"
    elif source == "policy":
        source_file = "backend/config/release_gates.json"
        editor_target = "admin.thresholds.walk_forward"
    else:
        source_file = "backend/ml/beta6/evaluation/evaluation_metrics.py"
        editor_target = "admin.thresholds.walk_forward_defaults"
    return {
        "value": int(value),
        "source": source,
        "resolved_key": resolved_key,
        "source_file": source_file,
        "editor_target": editor_target,
        "owner": "MLOps",
    }


def derive_room_confidence(fold_count: int | None, low_transition_folds: int | None) -> str:
    fc = int(fold_count or 0)
    ltf = int(low_transition_folds or 0)
    if fc >= 5 and ltf == 0:
        return "High"
    if fc >= 3:
        return "Medium"
    return "Low"


def build_room_status(
    room_payload: Mapping[str, Any],
    *,
    global_release_threshold: float | None,
    hours_since_fn: Callable[[Any], float | None],
) -> tuple[str, str, str]:
    metrics = room_payload.get("metrics", {}) if isinstance(room_payload.get("metrics"), dict) else {}
    support = room_payload.get("support", {}) if isinstance(room_payload.get("support"), dict) else {}
    gate = room_payload.get("gate", {}) if isinstance(room_payload.get("gate"), dict) else {}
    thresholds = room_payload.get("thresholds", {}) if isinstance(room_payload.get("thresholds"), dict) else {}
    fallback_active = bool(room_payload.get("fallback_active", False))
    room_name = str(room_payload.get("room") or "room")
    gate_reasons = gate.get("reasons", []) if isinstance(gate.get("reasons"), list) else []

    fold_count = _safe_int(support.get("fold_count")) or 0
    candidate_macro = _safe_float(metrics.get("candidate_macro_f1_mean"))
    transition_f1 = _safe_float(metrics.get("candidate_transition_macro_f1_mean"))
    stability_acc = _safe_float(metrics.get("candidate_stability_accuracy_mean"))
    min_transition = _safe_float((thresholds.get("min_transition_f1") or {}).get("value"))
    min_stability = _safe_float((thresholds.get("min_stability_accuracy") or {}).get("value"))
    max_transition_low = _safe_int((thresholds.get("max_transition_low_folds") or {}).get("value"))
    min_minority_support = _safe_int((thresholds.get("min_minority_support") or {}).get("value"))
    low_support_folds = _safe_int(support.get("candidate_low_support_folds")) or 0
    low_transition_folds = _safe_int(support.get("candidate_low_transition_folds")) or 0
    freshness_hours = hours_since_fn(room_payload.get("last_check_time"))

    evidence_reason_prefixes = (
        "wf_no_folds:",
        "fold_support_failed:",
        "insufficient_validation_support:",
    )
    normalized_gate_reason_codes: set[str] = set()
    for reason in gate_reasons:
        token = str(reason).strip().lower()
        if not token:
            continue
        if token.startswith("beta6_reason:"):
            token = token.split(":", 1)[1]
        normalized_gate_reason_codes.add(token)
    has_explicit_evidence_failure = any(
        str(reason).strip().lower().startswith(evidence_reason_prefixes)
        for reason in gate_reasons
    )
    routine_uncertainty_codes = {
        "fail_uncertainty_low_confidence",
        "fail_uncertainty_unknown",
        "fail_uncertainty_outside_sensed_space",
    }

    if candidate_macro is None and transition_f1 is None and stability_acc is None and fold_count == 0:
        return "not_available", f"No recent model-health checks for {room_name}.", "no_recent_data"

    if fallback_active:
        return "action_needed", f"Fallback mode is active for {room_name}.", "fallback_active"

    if min_minority_support is not None and int(min_minority_support) > 0 and low_support_folds > 0:
        return (
            "action_needed",
            f"Model quality checks for {room_name} lack enough minority-label evidence.",
            "insufficient_evidence_support",
        )

    if has_explicit_evidence_failure:
        return (
            "action_needed",
            f"Model quality checks for {room_name} have insufficient validation evidence.",
            "insufficient_evidence_gate",
        )

    if (
        not _safe_bool(gate.get("pass", True))
        and normalized_gate_reason_codes
        and normalized_gate_reason_codes.issubset(routine_uncertainty_codes)
    ):
        return (
            "watch",
            f"Routine uncertainty in {room_name} is routed to Review Queue.",
            "routed_review_queue_uncertainty",
        )

    if not _safe_bool(gate.get("pass", True)):
        return "action_needed", f"Safety checks paused model updates for {room_name}.", "gate_failed"

    metric_below = False
    if global_release_threshold is not None and candidate_macro is not None and fold_count > 0:
        metric_below = metric_below or (candidate_macro < float(global_release_threshold))
    if min_transition is not None and transition_f1 is not None and fold_count > 0:
        metric_below = metric_below or (transition_f1 < min_transition)
    if min_stability is not None and stability_acc is not None and fold_count > 0:
        metric_below = metric_below or (stability_acc < min_stability)
    if max_transition_low is not None and low_transition_folds > max_transition_low:
        metric_below = metric_below or True
    if metric_below:
        return "action_needed", f"Model quality for {room_name} is below safety thresholds.", "metric_below_threshold"

    macro_near = (
        global_release_threshold is not None
        and candidate_macro is not None
        and candidate_macro >= float(global_release_threshold)
        and (candidate_macro - float(global_release_threshold)) <= 0.03
    )
    transition_near = (
        min_transition is not None
        and transition_f1 is not None
        and transition_f1 >= min_transition
        and (transition_f1 - min_transition) <= 0.05
    )
    low_support = fold_count < 3
    stale = freshness_hours is not None and freshness_hours > 24.0
    if macro_near or transition_near:
        return "watch", f"Model quality for {room_name} is near threshold.", "near_threshold"
    if low_support:
        return "watch", f"Model confidence is low for {room_name} due to limited checks.", "low_support"
    if stale:
        return "watch", f"Model checks for {room_name} are stale.", "data_stale"

    return "healthy", f"Model health is within thresholds for {room_name}.", "healthy"


__all__ = [
    "build_room_status",
    "derive_room_confidence",
    "parse_room_override_map",
    "resolve_float_threshold_with_source",
    "resolve_int_threshold_with_source",
]
