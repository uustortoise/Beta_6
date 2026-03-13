from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from ml.policy_config import TrainingPolicy
from ml.timeline_metrics import summarize_grouped_metric_slices
from utils.room_utils import normalize_room_name


def _as_mapping(payload: Any) -> Mapping[str, Any]:
    return payload if isinstance(payload, Mapping) else {}


def _lookup_typed_policy_value(policy_payload: Mapping[str, Any], dotted_path: str) -> Any:
    node: Any = policy_payload
    for token in str(dotted_path).split("."):
        key = str(token).strip()
        if not key or not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def resolve_typed_policy_values(
    *,
    policy: TrainingPolicy,
    typed_policy_fields: list[str],
) -> dict[str, Any]:
    policy_payload = policy.to_dict()
    resolved: dict[str, Any] = {}
    for field in typed_policy_fields:
        key = str(field).strip()
        if not key:
            continue
        resolved[key] = _lookup_typed_policy_value(policy_payload, key)
    return resolved


def build_room_diagnostic_report(
    *,
    room_name: str,
    profile_name: str,
    profile_payload: Mapping[str, Any],
    typed_policy_values: Mapping[str, Any],
    grouped_fragility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    profile = _as_mapping(profile_payload)
    report_room = normalize_room_name(room_name)
    profile_room = normalize_room_name(profile.get("room") or report_room)

    return {
        "schema_version": "beta6.room_diagnostic_report.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "room": report_room,
        "profile_name": str(profile_name).strip().lower(),
        "profile_room": profile_room,
        "grouped_regime": str(profile.get("grouped_regime", "")).strip().lower(),
        "typed_policy_fields": [
            str(field).strip()
            for field in profile.get("typed_policy_fields", [])
            if str(field).strip()
        ],
        "typed_policy_values": dict(typed_policy_values),
        "env_overrides": {
            str(k).strip(): str(v).strip()
            for k, v in _as_mapping(profile.get("env_overrides")).items()
            if str(k).strip()
        },
        "fragility": dict(grouped_fragility or {}),
    }


def build_grouped_regime_fragility_report(
    *,
    grouped_by_date_slices: list[dict[str, Any]] | None = None,
    grouped_by_user_slices: list[dict[str, Any]] | None = None,
    fragile_room_floor: float | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {}

    grouped_by_date = summarize_grouped_metric_slices(
        grouped_by_date_slices,
        slice_key="date",
        metric_key="macro_f1",
    )
    if int(grouped_by_date.get("slice_count", 0)) > 0:
        report["grouped_by_date"] = grouped_by_date

    grouped_by_user = summarize_grouped_metric_slices(
        grouped_by_user_slices,
        slice_key="user",
        metric_key="macro_f1",
    )
    if int(grouped_by_user.get("slice_count", 0)) > 0:
        report["grouped_by_user"] = grouped_by_user

    if fragile_room_floor is None:
        return report

    floor = round(float(fragile_room_floor), 4)
    failures: list[dict[str, Any]] = []
    for regime_name in ("grouped_by_date", "grouped_by_user"):
        regime_summary = _as_mapping(report.get(regime_name))
        if not regime_summary:
            continue
        worst_macro_f1 = regime_summary.get("worst_slice_macro_f1")
        try:
            worst_macro_f1_float = float(worst_macro_f1)
        except (TypeError, ValueError):
            continue
        if worst_macro_f1_float >= floor:
            continue
        failures.append(
            {
                "regime": regime_name,
                "worst_slice": str(regime_summary.get("worst_slice") or "").strip(),
                "worst_slice_macro_f1": round(worst_macro_f1_float, 4),
                "floor": floor,
            }
        )

    report["stability_gate"] = {
        "fragile_floor": floor,
        "pass": not bool(failures),
        "failures": failures,
    }
    return report
