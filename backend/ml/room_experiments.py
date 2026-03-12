from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from ml.policy_config import TrainingPolicy
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

