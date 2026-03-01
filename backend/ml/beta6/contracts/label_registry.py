"""Label-registry contract for Beta 6 dynamic head generation and decoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ml.adl_registry import get_default_registry
from utils.room_utils import normalize_room_name


DEFAULT_MANDATORY_CLASSES: Tuple[str, ...] = ("active_use", "unoccupied")
_ACTIVE_USE_ALIASES: Tuple[str, ...] = (
    "active_living",
    "kitchen_use",
    "bathroom_use",
    "relaxing",
    "cooking",
    "eating",
    "occupied",
)
_UNOCCUPIED_ALIASES: Tuple[str, ...] = ("vacant", "room_unoccupied")


def _norm_label(value: Any) -> str:
    return str(value or "").strip().lower()


def _norm_room(value: Any) -> str:
    return normalize_room_name(value)


def _collect_backend_labels_by_room() -> Dict[str, set[str]]:
    registry = get_default_registry(skip_validation=False)
    out: Dict[str, set[str]] = {}
    for room in registry.list_rooms():
        room_key = _norm_room(room)
        scope = registry.get_room_scope(room)
        if scope is None:
            continue
        labels = out.setdefault(room_key, set())
        for event_id in scope.valid_events:
            token = _norm_label(event_id)
            if token:
                labels.add(token)
            try:
                event_def = registry.get_event(event_id)
            except Exception:
                continue
            for alias in event_def.aliases:
                alias_token = _norm_label(alias)
                if alias_token:
                    labels.add(alias_token)
    return out


def _normalize_labels(items: Iterable[Any]) -> set[str]:
    return {token for token in (_norm_label(item) for item in items) if token}


@dataclass(frozen=True)
class LabelRegistry:
    version: str
    room_to_labels: Dict[str, Tuple[str, ...]]
    mandatory_classes: Tuple[str, ...]
    sources: Dict[str, Dict[str, int]]

    def labels_for_room(self, room: str) -> Tuple[str, ...]:
        room_key = _norm_room(room)
        return self.room_to_labels.get(room_key, tuple())

    def index_map_for_room(self, room: str) -> Dict[str, int]:
        labels = self.labels_for_room(room)
        return {label: idx for idx, label in enumerate(labels)}

    def output_dim_for_room(self, room: str) -> int:
        return len(self.labels_for_room(room))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "mandatory_classes": list(self.mandatory_classes),
            "room_to_labels": {room: list(labels) for room, labels in self.room_to_labels.items()},
            "sources": self.sources,
        }


def _inject_mandatory_classes(labels: set[str], mandatory: Sequence[str]) -> set[str]:
    label_set = set(labels)
    mandatory_set = {_norm_label(item) for item in mandatory if _norm_label(item)}
    if "unoccupied" in mandatory_set and "unoccupied" not in label_set:
        if any(alias in label_set for alias in _UNOCCUPIED_ALIASES):
            label_set.add("unoccupied")
    if "active_use" in mandatory_set and "active_use" not in label_set:
        if any(alias in label_set for alias in _ACTIVE_USE_ALIASES):
            label_set.add("active_use")
    label_set.update(mandatory_set.intersection({"unoccupied", "active_use"}))
    return label_set


def build_label_registry(
    *,
    training_labels_by_room: Optional[Mapping[str, Iterable[Any]]] = None,
    manual_additions_by_room: Optional[Mapping[str, Iterable[Any]]] = None,
    mandatory_classes: Sequence[str] = DEFAULT_MANDATORY_CLASSES,
    include_backend_registry: bool = True,
) -> LabelRegistry:
    backend_raw = _collect_backend_labels_by_room() if include_backend_registry else {}
    backend: Dict[str, set[str]] = {}
    for room, labels in backend_raw.items():
        room_key = _norm_room(room)
        if not room_key:
            continue
        bucket = backend.setdefault(room_key, set())
        bucket.update(_normalize_labels(labels))

    training: Dict[str, set[str]] = {}
    for room, labels in (training_labels_by_room or {}).items():
        room_key = _norm_room(room)
        if not room_key:
            continue
        bucket = training.setdefault(room_key, set())
        bucket.update(_normalize_labels(labels))

    manual: Dict[str, set[str]] = {}
    for room, labels in (manual_additions_by_room or {}).items():
        room_key = _norm_room(room)
        if not room_key:
            continue
        bucket = manual.setdefault(room_key, set())
        bucket.update(_normalize_labels(labels))

    rooms = {
        *(_norm_room(room) for room in training.keys()),
        *(_norm_room(room) for room in manual.keys()),
        *(_norm_room(room) for room in backend.keys()),
    }
    rooms = {room for room in rooms if room}
    if not rooms:
        raise ValueError("no rooms available to build label registry")

    room_to_labels: Dict[str, Tuple[str, ...]] = {}
    source_stats: Dict[str, Dict[str, int]] = {}
    for room in sorted(rooms):
        labels = set()
        labels.update(backend.get(room, set()))
        labels.update(training.get(room, set()))
        labels.update(manual.get(room, set()))
        labels = _inject_mandatory_classes(labels, mandatory_classes)
        if not labels:
            raise ValueError(f"empty label set for room '{room}'")
        room_to_labels[room] = tuple(sorted(labels))
        source_stats[room] = {
            "backend_count": len(backend.get(room, set())),
            "training_count": len(training.get(room, set())),
            "manual_count": len(manual.get(room, set())),
            "final_count": len(room_to_labels[room]),
        }

    return LabelRegistry(
        version="label_registry_v1",
        room_to_labels=room_to_labels,
        mandatory_classes=tuple(_norm_label(v) for v in mandatory_classes if _norm_label(v)),
        sources=source_stats,
    )


def build_label_registry_from_training_frame(
    frame: Any,
    *,
    room_col: str = "room",
    activity_col: str = "activity",
    manual_additions_by_room: Optional[Mapping[str, Iterable[Any]]] = None,
    mandatory_classes: Sequence[str] = DEFAULT_MANDATORY_CLASSES,
    include_backend_registry: bool = True,
) -> LabelRegistry:
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if room_col not in frame.columns or activity_col not in frame.columns:
        raise ValueError(f"frame must contain '{room_col}' and '{activity_col}' columns")
    by_room: Dict[str, list[str]] = {}
    for room, group in frame.groupby(room_col):
        room_key = _norm_room(room)
        labels = [_norm_label(v) for v in group[activity_col].tolist()]
        by_room[room_key] = [v for v in labels if v]
    return build_label_registry(
        training_labels_by_room=by_room,
        manual_additions_by_room=manual_additions_by_room,
        mandatory_classes=mandatory_classes,
        include_backend_registry=include_backend_registry,
    )


__all__ = [
    "DEFAULT_MANDATORY_CLASSES",
    "LabelRegistry",
    "build_label_registry",
    "build_label_registry_from_training_frame",
]
