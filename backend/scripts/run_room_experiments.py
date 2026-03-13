#!/usr/bin/env python3
"""Generate replayable room-diagnostic artifacts with typed policy traceability."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ml.policy_config import load_policy_from_env
from ml.room_experiments import (
    build_grouped_regime_fragility_report,
    build_room_diagnostic_report,
    resolve_typed_policy_values,
)
from utils.room_utils import normalize_room_name


def _as_mapping(payload: Any) -> Mapping[str, Any]:
    return payload if isinstance(payload, Mapping) else {}


def _parse_grouped_fragility(raw_value: str) -> dict[str, Any]:
    txt = str(raw_value or "").strip()
    if not txt:
        return {}
    try:
        payload = json.loads(txt)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --grouped-fragility JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Invalid --grouped-fragility payload: expected JSON object")
    return payload


def _parse_grouped_slices(raw_value: str, *, label: str) -> list[dict[str, Any]]:
    txt = str(raw_value or "").strip()
    if not txt:
        return []
    try:
        payload = json.loads(txt)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --{label} JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"Invalid --{label} payload: expected JSON array")
    normalized: list[dict[str, Any]] = []
    for row in payload:
        if isinstance(row, dict):
            normalized.append(dict(row))
    return normalized


def _resolve_grouped_fragility_payload(
    *,
    raw_grouped_fragility: str,
    raw_grouped_date_slices: str,
    raw_grouped_user_slices: str,
    fragile_room_floor: float | None,
) -> dict[str, Any]:
    payload = _parse_grouped_fragility(raw_grouped_fragility)
    derived_payload = build_grouped_regime_fragility_report(
        grouped_by_date_slices=_parse_grouped_slices(
            raw_grouped_date_slices,
            label="grouped-date-slices",
        ),
        grouped_by_user_slices=_parse_grouped_slices(
            raw_grouped_user_slices,
            label="grouped-user-slices",
        ),
        fragile_room_floor=fragile_room_floor,
    )
    if derived_payload:
        payload.update(derived_payload)
    return payload


def _build_manifest_payload(
    *,
    profile_name: str,
    room_name: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "beta6.room_experiments_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile_name": str(profile_name).strip().lower(),
        "room": normalize_room_name(room_name),
        "report": dict(report),
    }


def _resolve_profile_typed_policy_values(profile: Mapping[str, Any]) -> dict[str, Any]:
    resolved_env = dict(os.environ)
    env_overrides = _as_mapping(profile.get("env_overrides"))
    for key, value in env_overrides.items():
        token = str(key).strip()
        if token:
            resolved_env[token] = str(value).strip()
    policy = load_policy_from_env(resolved_env)
    typed_fields = [
        str(field).strip()
        for field in profile.get("typed_policy_fields", [])
        if str(field).strip()
    ]
    return resolve_typed_policy_values(policy=policy, typed_policy_fields=typed_fields)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run room replay diagnostics")
    parser.add_argument("--room", required=True, help="Room name (e.g. Bedroom)")
    parser.add_argument("--profile", required=True, help="Diagnostic profile name")
    parser.add_argument("--output", required=True, help="Output manifest json path")
    parser.add_argument(
        "--grouped-fragility",
        default="",
        help="Optional grouped-fragility JSON payload for review surface",
    )
    parser.add_argument(
        "--grouped-date-slices",
        default="",
        help="Optional grouped-by-date slice metrics JSON array",
    )
    parser.add_argument(
        "--grouped-user-slices",
        default="",
        help="Optional grouped-by-user slice metrics JSON array",
    )
    parser.add_argument(
        "--fragile-room-floor",
        type=float,
        default=None,
        help="Optional macro-F1 floor for grouped fragile-room stability gating",
    )
    args = parser.parse_args()

    room = normalize_room_name(args.room)
    profile_name = str(args.profile).strip().lower()
    output_path = Path(args.output).expanduser()

    policy = load_policy_from_env()
    profile = policy.training_profile.get_room_diagnostic_profile(profile_name)
    if not profile:
        raise ValueError(f"Unknown room diagnostic profile: {profile_name}")
    if room and str(profile.get("room") or room) != room:
        raise ValueError(
            f"Profile room mismatch: profile={profile.get('room')} requested={room}"
        )

    typed_values = _resolve_profile_typed_policy_values(profile)
    grouped_fragility = _resolve_grouped_fragility_payload(
        raw_grouped_fragility=args.grouped_fragility,
        raw_grouped_date_slices=args.grouped_date_slices,
        raw_grouped_user_slices=args.grouped_user_slices,
        fragile_room_floor=args.fragile_room_floor,
    )
    report = build_room_diagnostic_report(
        room_name=room,
        profile_name=profile_name,
        profile_payload=profile,
        typed_policy_values=typed_values,
        grouped_fragility=grouped_fragility,
    )
    payload = _build_manifest_payload(
        profile_name=profile_name,
        room_name=room,
        report=report,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
