"""Room capability-aware gate profiles for Beta 6."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
from typing import Dict, Mapping

from ..beta6_schema import load_validated_beta6_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CapabilityProfile:
    """Deterministic room capability profile used by gate evaluation."""

    profile_id: str
    room_type: str
    min_expected_signal_auc: float
    min_f1: float
    max_timeline_mae_minutes: int
    max_fragmentation_rate: float


DEFAULT_PROFILE = CapabilityProfile(
    profile_id="cap_profile_generic_fallback",
    room_type="generic",
    min_expected_signal_auc=0.50,
    min_f1=0.50,
    max_timeline_mae_minutes=15,
    max_fragmentation_rate=0.30,
)

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "beta6_room_capability_gate_profiles.yaml"


def _build_profile(room_type: str, payload: Mapping[str, object]) -> CapabilityProfile:
    return CapabilityProfile(
        profile_id=str(payload.get("profile_id") or f"cap_profile_{room_type}_v1"),
        room_type=str(payload.get("room_type") or room_type),
        min_expected_signal_auc=float(payload.get("min_expected_signal_auc", DEFAULT_PROFILE.min_expected_signal_auc)),
        min_f1=float(payload.get("min_f1", DEFAULT_PROFILE.min_f1)),
        max_timeline_mae_minutes=int(payload.get("max_timeline_mae_minutes", DEFAULT_PROFILE.max_timeline_mae_minutes)),
        max_fragmentation_rate=float(
            payload.get("max_fragmentation_rate", DEFAULT_PROFILE.max_fragmentation_rate)
        ),
    )


@lru_cache(maxsize=1)
def _load_config() -> tuple[Dict[str, CapabilityProfile], str, Dict[str, tuple[str, ...]]]:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Capability profile config not found: {_CONFIG_PATH}")

    raw = load_validated_beta6_config(
        _CONFIG_PATH,
        expected_filename="beta6_room_capability_gate_profiles.yaml",
    )

    default_room_type = str(raw.get("default_room_type", "generic")).strip().lower() or "generic"
    profiles_raw = raw.get("profiles") or {}
    if not isinstance(profiles_raw, Mapping) or not profiles_raw:
        raise ValueError(f"Capability profile config has no profiles: {_CONFIG_PATH}")

    profiles: Dict[str, CapabilityProfile] = {}
    for room_type, payload in profiles_raw.items():
        key = str(room_type).strip().lower()
        if not key:
            continue
        if not isinstance(payload, Mapping):
            continue
        profiles[key] = _build_profile(key, payload)

    if default_room_type not in profiles:
        raise ValueError(
            f"default_room_type={default_room_type!r} not present in profiles at {_CONFIG_PATH}"
        )

    hints_raw = raw.get("room_type_hints") or {}
    hints: Dict[str, tuple[str, ...]] = {}
    if isinstance(hints_raw, Mapping):
        for room_type, values in hints_raw.items():
            key = str(room_type).strip().lower()
            if not key:
                continue
            if isinstance(values, (list, tuple)):
                hints[key] = tuple(str(v).strip().lower() for v in values if str(v).strip())
    return profiles, default_room_type, hints


CAPABILITY_PROFILES, _DEFAULT_ROOM_TYPE, _ROOM_TYPE_HINTS = _load_config()
DEFAULT_PROFILE = CAPABILITY_PROFILES.get(_DEFAULT_ROOM_TYPE, DEFAULT_PROFILE)


def infer_room_type(room: str) -> str:
    normalized = str(room).strip().lower()
    if not normalized:
        return _DEFAULT_ROOM_TYPE
    for room_type, hints in _ROOM_TYPE_HINTS.items():
        if any(hint in normalized for hint in hints):
            return room_type
    return _DEFAULT_ROOM_TYPE


def select_capability_profile(room: str, room_type: str | None = None) -> CapabilityProfile:
    selected_room_type = str(room_type).strip().lower() if room_type else infer_room_type(room)
    return CAPABILITY_PROFILES.get(selected_room_type, DEFAULT_PROFILE)
