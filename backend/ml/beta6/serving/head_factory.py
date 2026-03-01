"""Dynamic output-head generator driven by Beta 6 label registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from ..contracts.label_registry import DEFAULT_MANDATORY_CLASSES, LabelRegistry


@dataclass(frozen=True)
class HeadSpec:
    room: str
    labels: tuple[str, ...]
    output_dim: int
    index_map: Dict[str, int]
    mandatory_present: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room": self.room,
            "labels": list(self.labels),
            "output_dim": int(self.output_dim),
            "index_map": dict(self.index_map),
            "mandatory_present": dict(self.mandatory_present),
        }


def build_dynamic_head_specs(
    *,
    label_registry: LabelRegistry,
    rooms: Optional[Sequence[str]] = None,
    mandatory_classes: Sequence[str] = DEFAULT_MANDATORY_CLASSES,
) -> Dict[str, HeadSpec]:
    target_rooms = sorted(set(rooms)) if rooms is not None else sorted(label_registry.room_to_labels.keys())
    if not target_rooms:
        raise ValueError("no rooms provided for head generation")

    specs: Dict[str, HeadSpec] = {}
    for room in target_rooms:
        labels = label_registry.labels_for_room(room)
        if not labels:
            raise ValueError(f"label registry has no labels for room '{room}'")
        index_map = {label: idx for idx, label in enumerate(labels)}
        mandatory_present = {str(token): str(token) in index_map for token in mandatory_classes}
        specs[room] = HeadSpec(
            room=room,
            labels=labels,
            output_dim=len(labels),
            index_map=index_map,
            mandatory_present=mandatory_present,
        )
    return specs


def summarize_head_specs(specs: Mapping[str, HeadSpec]) -> Dict[str, Any]:
    room_output_dims = {room: int(spec.output_dim) for room, spec in specs.items()}
    total_outputs = int(sum(room_output_dims.values()))
    return {
        "rooms": sorted(specs.keys()),
        "room_output_dims": room_output_dims,
        "total_output_dim": total_outputs,
    }


__all__ = [
    "HeadSpec",
    "build_dynamic_head_specs",
    "summarize_head_specs",
]
