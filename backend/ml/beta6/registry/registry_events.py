"""Event helpers for Beta 6 registry persistence."""

from __future__ import annotations

from typing import Dict, Iterable

from ..contracts.events import DecisionEvent, EventType


def event_to_record(event: DecisionEvent) -> Dict[str, object]:
    """Convert a typed event to a JSON-serializable dictionary."""
    return {
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "run_id": event.run_id,
        "elder_id": event.elder_id,
        "room": event.room,
        "reason_code": event.reason_code,
        "payload": event.payload,
        "created_at": event.created_at,
    }


def allowed_event_types() -> Iterable[str]:
    """Return the canonical set of registry event type names."""
    return (event_type.value for event_type in EventType)
