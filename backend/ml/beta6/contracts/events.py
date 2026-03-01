"""Event contracts for Beta 6 registry and decision traces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class EventType(str, Enum):
    """Canonical event types for registry and decision traces."""

    CANDIDATE_SAVED = "candidate_saved"
    ROOM_GATE_PASSED = "room_gate_passed"
    ROOM_GATE_FAILED = "room_gate_failed"
    RUN_GATE_PASSED = "run_gate_passed"
    RUN_GATE_FAILED = "run_gate_failed"
    CANDIDATE_PROMOTED = "candidate_promoted"
    PROMOTION_ROLLBACK = "promotion_rollback"
    FALLBACK_MODE_ACTIVATED = "fallback_mode_activated"
    FALLBACK_MODE_CLEARED = "fallback_mode_cleared"


@dataclass(frozen=True)
class DecisionEvent:
    """Append-only event record emitted during a Beta 6 run."""

    event_type: EventType
    run_id: str
    elder_id: str
    room: Optional[str] = None
    reason_code: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def __post_init__(self) -> None:
        if self.event_id:
            return
        material = {
            "event_type": self.event_type.value,
            "run_id": self.run_id,
            "elder_id": self.elder_id,
            "room": self.room,
            "reason_code": self.reason_code,
            "payload": self.payload,
        }
        encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        object.__setattr__(self, "event_id", f"sha256:{digest}")
