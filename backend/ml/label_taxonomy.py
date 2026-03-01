"""Config and registry-backed label taxonomy helpers for Beta 6."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Set

from ml.beta6.beta6_schema import load_validated_beta6_config
from utils.room_utils import normalize_room_name

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_CRITICAL_LABELS_PATH = _CONFIG_DIR / "beta6_critical_labels.yaml"
_LANE_B_LABELS_PATH = _CONFIG_DIR / "beta6_lane_b_event_labels.yaml"
_FALLBACK_UNCERTAINTY_LABELS = {"low_confidence", "unknown", "outside_sensed_space"}


def _norm_label(raw: Any) -> str:
    txt = str(raw or "").strip().lower()
    return txt


def _as_mapping(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    return {}


@lru_cache(maxsize=1)
def _load_yaml(path: str) -> Mapping[str, Any]:
    file_path = Path(path)
    expected_filename = file_path.name
    payload = load_validated_beta6_config(file_path, expected_filename=expected_filename)
    return _as_mapping(payload)


def _load_critical_cfg() -> Mapping[str, Any]:
    return _load_yaml(str(_CRITICAL_LABELS_PATH))


def _load_lane_b_cfg() -> Mapping[str, Any]:
    return _load_yaml(str(_LANE_B_LABELS_PATH))


@lru_cache(maxsize=1)
def get_label_alias_to_canonical_map() -> Dict[str, str]:
    """
    Resolve explicit alias->canonical mapping for training/runtime vocabulary.
    """
    critical_cfg = _load_critical_cfg()
    defaults = _as_mapping(critical_cfg.get("defaults"))
    raw_map = _as_mapping(defaults.get("alias_to_canonical"))
    alias_map: Dict[str, str] = {}
    for alias_raw, canonical_raw in raw_map.items():
        alias = _norm_label(alias_raw)
        canonical = _norm_label(canonical_raw)
        if alias and canonical:
            alias_map[alias] = canonical
    return alias_map


def canonicalize_label(label: Any) -> str:
    """
    Canonicalize label token using explicit alias map.
    """
    token = _norm_label(label)
    if not token:
        return ""
    alias_map = get_label_alias_to_canonical_map()
    seen: Set[str] = set()
    current = token
    while current in alias_map and current not in seen:
        seen.add(current)
        current = _norm_label(alias_map.get(current))
        if not current:
            return token
    return current or token


@lru_cache(maxsize=1)
def get_label_alias_equivalents() -> Dict[str, list[str]]:
    """
    Build bidirectional equivalence classes from alias map.
    """
    alias_map = get_label_alias_to_canonical_map()
    grouped: Dict[str, Set[str]] = {}
    for alias, canonical in alias_map.items():
        canon = canonicalize_label(canonical)
        if not canon:
            continue
        bucket = grouped.setdefault(canon, set())
        bucket.add(canon)
        bucket.add(alias)

    resolved: Dict[str, list[str]] = {}
    for _, tokens in grouped.items():
        normalized = sorted({t for t in tokens if t})
        for token in normalized:
            resolved[token] = list(normalized)
    return resolved


@lru_cache(maxsize=1)
def _registry_aliases() -> Dict[str, str]:
    try:
        from ml.adl_registry import get_default_registry

        registry = get_default_registry(skip_validation=False)
        aliases = registry.list_all_aliases()
    except Exception as exc:
        logger.warning(f"Failed to load ADL registry aliases: {exc}")
        return {}
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in aliases.items()}


def _collect_registry_labels() -> Set[str]:
    labels: Set[str] = set()
    try:
        from ml.adl_registry import get_default_registry

        registry = get_default_registry(skip_validation=False)
        labels.update(_norm_label(item) for item in registry.list_all_events())
        labels.update(_norm_label(item) for item in registry.list_all_aliases().keys())
        for room_name in registry.list_rooms():
            room_scope = registry.get_room_scope(room_name)
            if room_scope is None:
                continue
            for event_id in room_scope.valid_events:
                token = _norm_label(event_id)
                if token:
                    labels.add(token)
                try:
                    event_def = registry.get_event(event_id)
                except Exception:
                    continue
                labels.update(_norm_label(alias) for alias in event_def.aliases if _norm_label(alias))
    except Exception as exc:
        logger.warning(f"Failed to resolve registry label taxonomy: {exc}")
    return labels


def _collect_labels_from_mapping(raw: Mapping[str, Any]) -> Set[str]:
    labels: Set[str] = set()
    for value in raw.values():
        if isinstance(value, Mapping):
            labels.update(_collect_labels_from_mapping(value))
            continue
        if isinstance(value, (list, tuple, set)):
            for item in value:
                token = _norm_label(item)
                if token:
                    labels.add(token)
                    canonical = canonicalize_label(token)
                    if canonical:
                        labels.add(canonical)
    return labels


@lru_cache(maxsize=1)
def get_valid_prediction_labels() -> Set[str]:
    """
    Return allowed labels for runtime prediction integrity checks.

    Sources:
    - ADL registry event ids + aliases.
    - Beta 6 critical-label config.
    - Beta 6 Lane-B event-label config.
    - Explicit uncertainty taxonomy labels.
    """
    labels: Set[str] = set()
    labels.update(_collect_registry_labels())

    critical_cfg = _load_critical_cfg()
    labels.update(_collect_labels_from_mapping(_as_mapping(critical_cfg.get("critical_labels_by_room"))))

    lane_b_cfg = _load_lane_b_cfg()
    labels.update(_collect_labels_from_mapping(_as_mapping(lane_b_cfg.get("lane_b_event_labels_by_room"))))

    defaults = _as_mapping(critical_cfg.get("defaults"))
    include_uncertainty = bool(defaults.get("include_uncertainty_labels", True))
    if include_uncertainty:
        uncertainty = defaults.get("uncertainty_labels")
        if isinstance(uncertainty, (list, tuple, set)) and len(uncertainty) > 0:
            labels.update(_norm_label(item) for item in uncertainty if _norm_label(item))
        else:
            labels.update(_FALLBACK_UNCERTAINTY_LABELS)
    else:
        labels.update(_FALLBACK_UNCERTAINTY_LABELS)

    alias_map = get_label_alias_to_canonical_map()
    expanded: Set[str] = set()
    for label in labels:
        token = _norm_label(label)
        if not token:
            continue
        expanded.add(token)
        canonical = canonicalize_label(token)
        if canonical:
            expanded.add(canonical)
    for alias, canonical in alias_map.items():
        if alias:
            expanded.add(alias)
        if canonical:
            expanded.add(canonical)

    return {label for label in expanded if label}


@lru_cache(maxsize=64)
def get_critical_labels_for_room(room_name: str) -> Set[str]:
    """
    Resolve room-specific critical labels from config + registry aliases.
    """
    room_key = normalize_room_name(room_name)
    critical_cfg = _load_critical_cfg()
    labels_by_room = _as_mapping(critical_cfg.get("critical_labels_by_room"))
    raw_labels = labels_by_room.get(room_key) or []
    labels = {canonicalize_label(item) for item in raw_labels if _norm_label(item)}

    aliases = _registry_aliases()
    expanded = set(labels)
    for label in list(labels):
        if not label:
            continue
        canonical = aliases.get(label)
        if canonical:
            expanded.add(canonicalize_label(canonical))
    for raw_label in raw_labels:
        token = _norm_label(raw_label)
        if not token:
            continue
        expanded.add(canonicalize_label(token))
        canonical = aliases.get(token)
        if canonical:
            expanded.add(canonicalize_label(canonical))
    return {label for label in expanded if label}


@lru_cache(maxsize=64)
def get_lane_b_event_labels_for_room(room_name: str) -> Dict[str, list[str]]:
    """
    Resolve Lane-B event-to-label mapping from config.
    """
    room_key = normalize_room_name(room_name)
    lane_b_cfg = _load_lane_b_cfg()
    labels_by_room = _as_mapping(lane_b_cfg.get("lane_b_event_labels_by_room"))
    room_map = _as_mapping(labels_by_room.get(room_key))

    aliases = _registry_aliases()
    resolved: Dict[str, list[str]] = {}
    for event_name, labels in room_map.items():
        if not isinstance(labels, (list, tuple, set)):
            continue
        bucket = set()
        for item in labels:
            label = _norm_label(item)
            if not label:
                continue
            bucket.add(label)
            bucket.add(canonicalize_label(label))
            canonical = aliases.get(label)
            if canonical:
                bucket.add(canonicalize_label(canonical))
        if bucket:
            resolved[_norm_label(event_name)] = sorted(bucket)
    return resolved
