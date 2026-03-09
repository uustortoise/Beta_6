"""Registry v2: append-only events with atomic champion pointers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..contracts.decisions import ReasonCode
from ..contracts.events import DecisionEvent, EventType
from .registry_events import event_to_record

BETA62_AUTHORITATIVE_MODULE = "ml.beta6.registry.registry_v2"
BETA62_MODULE_SURFACE = "registry"


class RegistryV2:
    """Minimal registry API used in Beta 6 Phase 1 scaffolding."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _room_dir(self, elder_id: str, room: str) -> Path:
        return self.root / elder_id / room

    def _events_path(self, elder_id: str, room: str) -> Path:
        return self._room_dir(elder_id, room) / "events.jsonl"

    def _pointer_path(self, elder_id: str, room: str) -> Path:
        return self._room_dir(elder_id, room) / "champion_pointer.json"

    def _history_path(self, elder_id: str, room: str) -> Path:
        return self._room_dir(elder_id, room) / "champion_history.jsonl"

    def _fallback_state_path(self, elder_id: str, room: str) -> Path:
        return self._room_dir(elder_id, room) / "fallback_state.json"

    def append_event(self, event: DecisionEvent) -> bool:
        """Append a registry event as a single JSON line with idempotency."""
        room = event.room or "_run"
        path = self._events_path(event.elder_id, room)
        path.parent.mkdir(parents=True, exist_ok=True)

        record = event_to_record(event)
        event_id = str(record["event_id"])
        existing_ids = set()
        if path.exists():
            with path.open("r", encoding="utf-8") as read_handle:
                for line in read_handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    parsed_event_id = parsed.get("event_id")
                    if parsed_event_id:
                        existing_ids.add(str(parsed_event_id))

        if event_id in existing_ids:
            return False

        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
        return True

    def read_events(self, elder_id: str, room: str) -> List[Dict[str, object]]:
        """Read room event history in append order."""
        path = self._events_path(elder_id, room)
        if not path.exists():
            return []
        records: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def update_champion_pointer(
        self, elder_id: str, room: str, pointer: Dict[str, object]
    ) -> None:
        """Write champion pointer using atomic replace."""
        pointer_path = self._pointer_path(elder_id, room)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = pointer_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(pointer, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(pointer_path)
        self._append_history(elder_id=elder_id, room=room, pointer=pointer)

    def read_champion_pointer(self, elder_id: str, room: str) -> Optional[Dict[str, object]]:
        """Load champion pointer if available."""
        pointer_path = self._pointer_path(elder_id, room)
        if not pointer_path.exists():
            return None
        with pointer_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _append_history(self, elder_id: str, room: str, pointer: Dict[str, object]) -> None:
        history_path = self._history_path(elder_id, room)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pointer": pointer,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True))
            handle.write("\n")

    def read_pointer_history(self, elder_id: str, room: str) -> List[Dict[str, object]]:
        """Read pointer history in append order."""
        history_path = self._history_path(elder_id, room)
        if not history_path.exists():
            return []
        history: List[Dict[str, object]] = []
        with history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    history.append(json.loads(line))
        return history

    def _write_fallback_state(self, elder_id: str, room: str, state: Dict[str, object]) -> None:
        fallback_state_path = self._fallback_state_path(elder_id, room)
        fallback_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = fallback_state_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(fallback_state_path)

    def read_fallback_state(self, elder_id: str, room: str) -> Optional[Dict[str, object]]:
        path = self._fallback_state_path(elder_id, room)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _resolve_fallback_target(
        self,
        *,
        current_pointer: Optional[Dict[str, object]],
        history: List[Dict[str, object]],
        fallback_candidate_id: Optional[str],
        fallback_state: Optional[Mapping[str, object]] = None,
    ) -> Optional[Dict[str, object]]:
        current_signature = self._pointer_signature(current_pointer)

        def _coerce_pointer(raw: object, *, allow_current: bool) -> Optional[Dict[str, object]]:
            if not isinstance(raw, Mapping):
                return None
            pointer = dict(raw)
            if not allow_current and self._pointer_signature(pointer) == current_signature:
                return None
            return pointer

        if fallback_candidate_id:
            if (
                isinstance(current_pointer, Mapping)
                and str(current_pointer.get("candidate_id")) == str(fallback_candidate_id)
            ):
                return dict(current_pointer)
            for candidate in (
                (fallback_state or {}).get("active_pointer"),
                (fallback_state or {}).get("previous_pointer"),
            ):
                pointer = _coerce_pointer(candidate, allow_current=True)
                if isinstance(pointer, Mapping) and str(pointer.get("candidate_id")) == str(
                    fallback_candidate_id
                ):
                    return dict(pointer)
            for entry in reversed(history):
                pointer = entry.get("pointer")
                if isinstance(pointer, Mapping) and str(pointer.get("candidate_id")) == str(
                    fallback_candidate_id
                ):
                    return dict(pointer)
            return None

        previous_pointer = _coerce_pointer((fallback_state or {}).get("previous_pointer"), allow_current=False)
        if previous_pointer is not None:
            return previous_pointer

        for entry in reversed(history):
            pointer = _coerce_pointer(entry.get("pointer"), allow_current=False)
            if pointer is not None:
                return pointer

        if history:
            pointer = _coerce_pointer(history[-1].get("pointer"), allow_current=True)
            if pointer is not None and current_pointer is None:
                return pointer
        return None

    def resolve_fallback_target(
        self,
        *,
        elder_id: str,
        room: str,
        fallback_candidate_id: Optional[str] = None,
    ) -> Optional[Dict[str, object]]:
        return self._resolve_fallback_target(
            current_pointer=self.read_champion_pointer(elder_id, room),
            history=self.read_pointer_history(elder_id, room),
            fallback_candidate_id=fallback_candidate_id,
            fallback_state=self.read_fallback_state(elder_id, room),
        )

    @staticmethod
    def _pointer_signature(pointer: Optional[Mapping[str, object]]) -> tuple[object, object]:
        if not isinstance(pointer, Mapping):
            return (None, None)
        return (pointer.get("candidate_id"), pointer.get("run_id"))

    def activate_fallback_mode(
        self,
        *,
        elder_id: str,
        room: str,
        run_id: str,
        trigger_reason_code: str,
        fallback_candidate_id: Optional[str] = None,
        fallback_flags: Optional[Dict[str, object]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Activate deterministic operator-safe fallback mode with full audit trail."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        existing_state = self.read_fallback_state(elder_id, room)
        if isinstance(existing_state, Mapping) and bool(existing_state.get("active", False)):
            state = dict(existing_state)
            self.append_event(
                DecisionEvent(
                    event_type=EventType.FALLBACK_MODE_ACTIVATED,
                    run_id=run_id,
                    elder_id=elder_id,
                    room=room,
                    reason_code=ReasonCode.FALLBACK_ALREADY_ACTIVE.value,
                    payload={
                        "fallback_state": state,
                        "trigger_reason_code": trigger_reason_code,
                    },
                )
            )
            return state

        current_pointer = self.read_champion_pointer(elder_id, room)
        history = self.read_pointer_history(elder_id, room)
        fallback_state = self.read_fallback_state(elder_id, room)
        target_pointer = self._resolve_fallback_target(
            current_pointer=current_pointer,
            history=history,
            fallback_candidate_id=fallback_candidate_id,
            fallback_state=fallback_state,
        )
        if target_pointer is None:
            self.append_event(
                DecisionEvent(
                    event_type=EventType.FALLBACK_MODE_ACTIVATED,
                    run_id=run_id,
                    elder_id=elder_id,
                    room=room,
                    reason_code=ReasonCode.FALLBACK_MISSING_TARGET.value,
                    payload={
                        "trigger_reason_code": trigger_reason_code,
                        "requested_candidate_id": fallback_candidate_id,
                        "history_length": len(history),
                    },
                )
            )
            raise ValueError(f"No fallback target available for {elder_id}/{room}")

        switched_pointer = False
        if self._pointer_signature(current_pointer) != self._pointer_signature(target_pointer):
            self.update_champion_pointer(elder_id=elder_id, room=room, pointer=target_pointer)
            switched_pointer = True
        active_pointer = self.read_champion_pointer(elder_id, room)

        normalized_flags: Dict[str, object] = {
            "operator_safe_mode": True,
            "serving_mode": "rule_hmm_baseline",
        }
        if fallback_flags:
            normalized_flags.update(dict(fallback_flags))

        state = {
            "state_version": "v1",
            "active": True,
            "activated_at": now,
            "activated_run_id": run_id,
            "trigger_reason_code": str(trigger_reason_code),
            "fallback_reason_code": ReasonCode.FALLBACK_ACTIVATED.value,
            "fallback_candidate_id": target_pointer.get("candidate_id"),
            "fallback_flags": normalized_flags,
            "previous_pointer": current_pointer,
            "active_pointer": active_pointer,
            "metadata": dict(metadata or {}),
        }
        self._write_fallback_state(elder_id, room, state)
        self.append_event(
            DecisionEvent(
                event_type=EventType.FALLBACK_MODE_ACTIVATED,
                run_id=run_id,
                elder_id=elder_id,
                room=room,
                reason_code=ReasonCode.FALLBACK_ACTIVATED.value,
                payload={
                    "trigger_reason_code": trigger_reason_code,
                    "requested_candidate_id": fallback_candidate_id,
                    "switched_pointer": switched_pointer,
                    "from_pointer": current_pointer,
                    "to_pointer": active_pointer,
                    "fallback_flags": normalized_flags,
                    "fallback_state_path": str(self._fallback_state_path(elder_id, room)),
                },
            )
        )
        return state

    def clear_fallback_mode(
        self,
        *,
        elder_id: str,
        room: str,
        run_id: str,
        clear_reason_code: Optional[str] = None,
        restore_previous_pointer: bool = True,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Clear fallback mode and optionally restore pre-fallback champion pointer."""
        def _normalize_reason(raw: Optional[str], default: ReasonCode) -> str:
            if raw is None:
                return default.value
            token = str(raw).strip()
            if not token:
                return default.value
            try:
                return ReasonCode(token).value
            except ValueError:
                return default.value

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        state = self.read_fallback_state(elder_id, room)
        current_pointer = self.read_champion_pointer(elder_id, room)
        if not isinstance(state, Mapping) or not bool(state.get("active", False)):
            reason_code = _normalize_reason(clear_reason_code, ReasonCode.FALLBACK_NOT_ACTIVE)
            cleared_state = {
                "state_version": "v1",
                "active": False,
                "cleared_at": now,
                "cleared_run_id": run_id,
                "clear_reason_code": reason_code,
                "previous_state": state,
                "active_pointer": current_pointer,
                "metadata": dict(metadata or {}),
            }
            self._write_fallback_state(elder_id, room, cleared_state)
            self.append_event(
                DecisionEvent(
                    event_type=EventType.FALLBACK_MODE_CLEARED,
                    run_id=run_id,
                    elder_id=elder_id,
                    room=room,
                    reason_code=reason_code,
                    payload={
                        "restore_previous_pointer": restore_previous_pointer,
                        "clear_reason_code": reason_code,
                    },
                )
            )
            return cleared_state

        previous_pointer = state.get("previous_pointer")
        restored_previous_pointer = False
        if restore_previous_pointer and isinstance(previous_pointer, Mapping):
            if self._pointer_signature(current_pointer) != self._pointer_signature(previous_pointer):
                self.update_champion_pointer(elder_id=elder_id, room=room, pointer=dict(previous_pointer))
                restored_previous_pointer = True
        pointer_after_clear = self.read_champion_pointer(elder_id, room)
        reason_code = _normalize_reason(clear_reason_code, ReasonCode.FALLBACK_CLEARED)
        cleared_state = {
            "state_version": "v1",
            "active": False,
            "cleared_at": now,
            "cleared_run_id": run_id,
            "clear_reason_code": reason_code,
            "restored_previous_pointer": restored_previous_pointer,
            "restore_previous_pointer_requested": bool(restore_previous_pointer),
            "previous_state": dict(state),
            "active_pointer": pointer_after_clear,
            "metadata": dict(metadata or {}),
        }
        self._write_fallback_state(elder_id, room, cleared_state)
        self.append_event(
            DecisionEvent(
                event_type=EventType.FALLBACK_MODE_CLEARED,
                run_id=run_id,
                elder_id=elder_id,
                room=room,
                reason_code=reason_code,
                payload={
                    "restore_previous_pointer": bool(restore_previous_pointer),
                    "restored_previous_pointer": restored_previous_pointer,
                    "from_pointer": current_pointer,
                    "to_pointer": pointer_after_clear,
                    "clear_reason_code": reason_code,
                    "fallback_state_path": str(self._fallback_state_path(elder_id, room)),
                },
            )
        )
        return cleared_state

    def rollback_and_activate_fallback(
        self,
        *,
        elder_id: str,
        room: str,
        run_id: str,
        trigger_reason_code: str,
        fallback_candidate_id: Optional[str] = None,
        fallback_flags: Optional[Dict[str, object]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """
        Execute rollback + fallback activation as a single deterministic action.

        Rollback restores the previous champion pointer when available; fallback then
        activates operator-safe baseline serving for the room.
        """
        result: Dict[str, object] = {
            "elder_id": elder_id,
            "room": room,
            "run_id": run_id,
            "rollback_applied": False,
            "rollback_error": None,
            "rollback_pointer": None,
            "fallback_state": None,
            "error": None,
        }

        candidate_hint = fallback_candidate_id
        rollback_reason = ReasonCode.ROLLBACK_TRIGGERED.value
        try:
            rollback_pointer = self.rollback_to_previous(elder_id=elder_id, room=room, run_id=run_id)
            result["rollback_applied"] = True
            result["rollback_pointer"] = rollback_pointer
            candidate_hint = str(rollback_pointer.get("candidate_id") or "") or candidate_hint
        except ValueError as exc:
            result["rollback_error"] = str(exc)
            rollback_reason = ReasonCode.ROLLBACK_MISSING_TARGET.value
            safe_pointer = self.resolve_fallback_target(elder_id=elder_id, room=room)
            if isinstance(safe_pointer, Mapping):
                candidate_hint = str(safe_pointer.get("candidate_id") or "") or candidate_hint

        fallback_reason = str(trigger_reason_code or "").strip() or rollback_reason
        merged_flags = dict(fallback_flags or {})
        merged_flags.setdefault("operator_safe_mode", True)
        merged_flags.setdefault("serving_mode", "rule_hmm_baseline")

        try:
            fallback_state = self.activate_fallback_mode(
                elder_id=elder_id,
                room=room,
                run_id=run_id,
                trigger_reason_code=fallback_reason,
                fallback_candidate_id=candidate_hint,
                fallback_flags=merged_flags,
                metadata={
                    "auto_rollback_reason": rollback_reason,
                    "trigger_reason_code": fallback_reason,
                    **dict(metadata or {}),
                },
            )
        except ValueError as exc:
            result["error"] = str(exc)
            return result
        result["fallback_state"] = fallback_state
        return result

    def promote_candidate(
        self,
        elder_id: str,
        room: str,
        run_id: str,
        candidate_id: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """
        Atomically update champion pointer and append promotion event.

        Returns the new champion pointer.
        """
        previous = self.read_champion_pointer(elder_id, room)
        pointer = {
            "elder_id": elder_id,
            "room": room,
            "candidate_id": candidate_id,
            "run_id": run_id,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metadata": metadata or {},
        }
        self.update_champion_pointer(elder_id=elder_id, room=room, pointer=pointer)
        self.append_event(
            DecisionEvent(
                event_type=EventType.CANDIDATE_PROMOTED,
                run_id=run_id,
                elder_id=elder_id,
                room=room,
                reason_code=ReasonCode.PASS.value,
                payload={"previous_pointer": previous, "new_pointer": pointer},
            )
        )
        return pointer

    def rollback_to_previous(self, elder_id: str, room: str, run_id: str) -> Dict[str, object]:
        """
        Restore previous champion pointer from history.

        Raises ValueError when no rollback target exists.
        """
        history = self.read_pointer_history(elder_id, room)
        if len(history) < 2:
            self.append_event(
                DecisionEvent(
                    event_type=EventType.PROMOTION_ROLLBACK,
                    run_id=run_id,
                    elder_id=elder_id,
                    room=room,
                    reason_code=ReasonCode.ROLLBACK_MISSING_TARGET.value,
                    payload={"history_length": len(history)},
                )
            )
            raise ValueError(f"No rollback target available for {elder_id}/{room}")

        current = self.read_champion_pointer(elder_id, room)
        target = dict(history[-2]["pointer"])
        self.update_champion_pointer(elder_id=elder_id, room=room, pointer=target)
        self.append_event(
            DecisionEvent(
                event_type=EventType.PROMOTION_ROLLBACK,
                run_id=run_id,
                elder_id=elder_id,
                room=room,
                reason_code=ReasonCode.ROLLBACK_TRIGGERED.value,
                payload={"from_pointer": current, "to_pointer": target},
            )
        )
        return target
