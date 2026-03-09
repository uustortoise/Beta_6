from pathlib import Path

import pytest

from ml.beta6.contracts.decisions import ReasonCode
from ml.beta6.contracts.events import DecisionEvent, EventType
from ml.beta6.registry_v2 import RegistryV2


def test_append_event_is_idempotent(tmp_path: Path):
    registry = RegistryV2(root=tmp_path)
    event = DecisionEvent(
        event_type=EventType.ROOM_GATE_FAILED,
        run_id="run-1",
        elder_id="HK001",
        room="bedroom",
        reason_code=ReasonCode.FAIL_GATE_POLICY.value,
        payload={"x": 1},
    )
    assert registry.append_event(event) is True
    # Same logical event should dedupe by event_id.
    duplicate = DecisionEvent(
        event_type=EventType.ROOM_GATE_FAILED,
        run_id="run-1",
        elder_id="HK001",
        room="bedroom",
        reason_code=ReasonCode.FAIL_GATE_POLICY.value,
        payload={"x": 1},
    )
    assert registry.append_event(duplicate) is False
    records = registry.read_events("HK001", "bedroom")
    assert len(records) == 1


def test_event_order_is_append_order(tmp_path: Path):
    registry = RegistryV2(root=tmp_path)
    events = [
        DecisionEvent(EventType.CANDIDATE_SAVED, "run-1", "HK001", "livingroom", payload={"seq": 1}),
        DecisionEvent(EventType.ROOM_GATE_PASSED, "run-1", "HK001", "livingroom", payload={"seq": 2}),
        DecisionEvent(EventType.CANDIDATE_PROMOTED, "run-1", "HK001", "livingroom", payload={"seq": 3}),
    ]
    for event in events:
        assert registry.append_event(event) is True

    records = registry.read_events("HK001", "livingroom")
    assert [record["payload"]["seq"] for record in records] == [1, 2, 3]


def test_promote_then_rollback_restores_previous_pointer(tmp_path: Path):
    registry = RegistryV2(root=tmp_path)
    elder_id = "HK001"
    room = "bedroom"

    pointer_1 = registry.promote_candidate(
        elder_id=elder_id,
        room=room,
        run_id="run-1",
        candidate_id="cand-v1",
        metadata={"score": 0.81},
    )
    pointer_2 = registry.promote_candidate(
        elder_id=elder_id,
        room=room,
        run_id="run-2",
        candidate_id="cand-v2",
        metadata={"score": 0.85},
    )
    assert pointer_2["candidate_id"] == "cand-v2"
    assert registry.read_champion_pointer(elder_id, room)["candidate_id"] == "cand-v2"

    restored = registry.rollback_to_previous(elder_id=elder_id, room=room, run_id="run-3")
    assert restored["candidate_id"] == pointer_1["candidate_id"]
    assert registry.read_champion_pointer(elder_id, room)["candidate_id"] == "cand-v1"

    events = registry.read_events(elder_id, room)
    assert events[-1]["event_type"] == EventType.PROMOTION_ROLLBACK.value
    assert events[-1]["reason_code"] == ReasonCode.ROLLBACK_TRIGGERED.value


def test_rollback_without_target_raises_and_logs_reason(tmp_path: Path):
    registry = RegistryV2(root=tmp_path)
    elder_id = "HK001"
    room = "livingroom"

    # Only one pointer in history -> no rollback target.
    registry.promote_candidate(
        elder_id=elder_id,
        room=room,
        run_id="run-1",
        candidate_id="cand-v1",
    )

    with pytest.raises(ValueError, match="No rollback target available"):
        registry.rollback_to_previous(elder_id=elder_id, room=room, run_id="run-2")

    events = registry.read_events(elder_id, room)
    assert events[-1]["event_type"] == EventType.PROMOTION_ROLLBACK.value
    assert events[-1]["reason_code"] == ReasonCode.ROLLBACK_MISSING_TARGET.value


def test_activate_fallback_mode_switches_pointer_and_logs_event(tmp_path: Path):
    registry = RegistryV2(root=tmp_path)
    elder_id = "HK001"
    room = "bedroom"

    registry.promote_candidate(
        elder_id=elder_id,
        room=room,
        run_id="run-1",
        candidate_id="cand-v1",
        metadata={"score": 0.81},
    )
    registry.promote_candidate(
        elder_id=elder_id,
        room=room,
        run_id="run-2",
        candidate_id="cand-v2",
        metadata={"score": 0.88},
    )

    state = registry.activate_fallback_mode(
        elder_id=elder_id,
        room=room,
        run_id="run-3",
        trigger_reason_code=ReasonCode.FAIL_GATE_POLICY.value,
        fallback_candidate_id="cand-v1",
        fallback_flags={"manual_override": True},
        metadata={"operator": "qa"},
    )
    assert state["active"] is True
    assert state["fallback_candidate_id"] == "cand-v1"
    assert state["fallback_flags"]["operator_safe_mode"] is True
    assert state["fallback_flags"]["manual_override"] is True
    assert registry.read_champion_pointer(elder_id, room)["candidate_id"] == "cand-v1"

    events = registry.read_events(elder_id, room)
    assert events[-1]["event_type"] == EventType.FALLBACK_MODE_ACTIVATED.value
    assert events[-1]["reason_code"] == ReasonCode.FALLBACK_ACTIVATED.value
    assert events[-1]["payload"]["switched_pointer"] is True


def test_registry_resolves_fallback_from_previous_known_good_pointer(tmp_path: Path):
    registry = RegistryV2(root=tmp_path)
    elder_id = "HK001"
    room = "livingroom"

    registry.promote_candidate(
        elder_id=elder_id,
        room=room,
        run_id="run-1",
        candidate_id="cand-v1",
    )
    registry.promote_candidate(
        elder_id=elder_id,
        room=room,
        run_id="run-2",
        candidate_id="cand-v2",
    )

    state = registry.activate_fallback_mode(
        elder_id=elder_id,
        room=room,
        run_id="run-3",
        trigger_reason_code=ReasonCode.FAIL_GATE_POLICY.value,
    )

    assert state["active"] is True
    assert state["fallback_candidate_id"] == "cand-v1"
    assert registry.read_champion_pointer(elder_id, room)["candidate_id"] == "cand-v1"


def test_activate_fallback_mode_without_target_raises_and_logs_reason(tmp_path: Path):
    registry = RegistryV2(root=tmp_path)
    elder_id = "HK001"
    room = "livingroom"

    with pytest.raises(ValueError, match="No fallback target available"):
        registry.activate_fallback_mode(
            elder_id=elder_id,
            room=room,
            run_id="run-1",
            trigger_reason_code=ReasonCode.FAIL_GATE_POLICY.value,
            fallback_candidate_id="missing-candidate",
        )

    events = registry.read_events(elder_id, room)
    assert events[-1]["event_type"] == EventType.FALLBACK_MODE_ACTIVATED.value
    assert events[-1]["reason_code"] == ReasonCode.FALLBACK_MISSING_TARGET.value


def test_clear_fallback_mode_restores_previous_pointer_and_logs_event(tmp_path: Path):
    registry = RegistryV2(root=tmp_path)
    elder_id = "HK001"
    room = "bathroom"

    registry.promote_candidate(
        elder_id=elder_id,
        room=room,
        run_id="run-1",
        candidate_id="cand-v1",
    )
    registry.promote_candidate(
        elder_id=elder_id,
        room=room,
        run_id="run-2",
        candidate_id="cand-v2",
    )
    registry.activate_fallback_mode(
        elder_id=elder_id,
        room=room,
        run_id="run-3",
        trigger_reason_code=ReasonCode.FAIL_GATE_POLICY.value,
        fallback_candidate_id="cand-v1",
    )
    assert registry.read_champion_pointer(elder_id, room)["candidate_id"] == "cand-v1"

    cleared = registry.clear_fallback_mode(
        elder_id=elder_id,
        room=room,
        run_id="run-4",
        restore_previous_pointer=True,
    )
    assert cleared["active"] is False
    assert cleared["restored_previous_pointer"] is True
    assert registry.read_champion_pointer(elder_id, room)["candidate_id"] == "cand-v2"

    events = registry.read_events(elder_id, room)
    assert events[-1]["event_type"] == EventType.FALLBACK_MODE_CLEARED.value
    assert events[-1]["reason_code"] == ReasonCode.FALLBACK_CLEARED.value


def test_clear_fallback_mode_accepts_explicit_reason_code(tmp_path: Path):
    registry = RegistryV2(root=tmp_path)
    elder_id = "HK001"
    room = "livingroom"

    registry.promote_candidate(
        elder_id=elder_id,
        room=room,
        run_id="run-1",
        candidate_id="cand-v1",
    )
    registry.activate_fallback_mode(
        elder_id=elder_id,
        room=room,
        run_id="run-2",
        trigger_reason_code=ReasonCode.FAIL_GATE_POLICY.value,
        fallback_candidate_id="cand-v1",
    )

    state = registry.clear_fallback_mode(
        elder_id=elder_id,
        room=room,
        run_id="run-3",
        clear_reason_code=ReasonCode.PASS.value,
        restore_previous_pointer=False,
    )

    assert state["clear_reason_code"] == ReasonCode.PASS.value
    events = registry.read_events(elder_id, room)
    assert events[-1]["event_type"] == EventType.FALLBACK_MODE_CLEARED.value
    assert events[-1]["reason_code"] == ReasonCode.PASS.value


def test_clear_fallback_mode_when_not_active_logs_noop_reason(tmp_path: Path):
    registry = RegistryV2(root=tmp_path)
    elder_id = "HK001"
    room = "kitchen"

    state = registry.clear_fallback_mode(
        elder_id=elder_id,
        room=room,
        run_id="run-1",
        restore_previous_pointer=True,
    )
    assert state["active"] is False
    assert state["clear_reason_code"] == ReasonCode.FALLBACK_NOT_ACTIVE.value

    events = registry.read_events(elder_id, room)
    assert events[-1]["event_type"] == EventType.FALLBACK_MODE_CLEARED.value
    assert events[-1]["reason_code"] == ReasonCode.FALLBACK_NOT_ACTIVE.value


def test_rollback_and_activate_fallback_applies_both_actions(tmp_path: Path):
    registry = RegistryV2(root=tmp_path)
    elder_id = "HK001"
    room = "livingroom"

    registry.promote_candidate(
        elder_id=elder_id,
        room=room,
        run_id="run-1",
        candidate_id="cand-v1",
    )
    registry.promote_candidate(
        elder_id=elder_id,
        room=room,
        run_id="run-2",
        candidate_id="cand-v2",
    )

    result = registry.rollback_and_activate_fallback(
        elder_id=elder_id,
        room=room,
        run_id="run-3",
        trigger_reason_code="pipeline_reliability_breach",
    )

    assert result["rollback_applied"] is True
    assert result["fallback_state"]["active"] is True
    assert result["fallback_state"]["fallback_candidate_id"] == "cand-v1"
    assert registry.read_champion_pointer(elder_id, room)["candidate_id"] == "cand-v1"


def test_rollback_and_activate_fallback_handles_missing_rollback_target(tmp_path: Path):
    registry = RegistryV2(root=tmp_path)
    elder_id = "HK009"
    room = "bedroom"
    registry.promote_candidate(
        elder_id=elder_id,
        room=room,
        run_id="run-1",
        candidate_id="cand-v1",
    )

    result = registry.rollback_and_activate_fallback(
        elder_id=elder_id,
        room=room,
        run_id="run-2",
        trigger_reason_code="mae_regression_breach",
    )

    assert result["rollback_applied"] is False
    assert "No rollback target available" in str(result["rollback_error"])
    assert result["fallback_state"] is None
    assert "No fallback target available" in str(result["error"])
