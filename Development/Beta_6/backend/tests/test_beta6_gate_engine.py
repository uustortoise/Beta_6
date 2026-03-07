from ml.beta6.contracts.decisions import ReasonCode, RoomDecision
from ml.beta6.gate_engine import GateEngine


def test_decide_room_pass_forces_pass_reason():
    decision = GateEngine().decide_room(
        room="livingroom",
        report={"passed": True, "reason_code": "fail_gate_policy"},
    )
    assert decision.passed is True
    assert decision.reason_code == ReasonCode.PASS


def test_decide_room_uses_explicit_reason_code_when_valid():
    decision = GateEngine().decide_room(
        room="bedroom",
        report={"passed": False, "reason_code": ReasonCode.FAIL_LEAKAGE_TIME.value},
    )
    assert decision.passed is False
    assert decision.reason_code == ReasonCode.FAIL_LEAKAGE_TIME
    assert decision.details["reason_source"] == "explicit"


def test_decide_room_derives_reason_from_report_when_missing():
    decision = GateEngine().decide_room(
        room="bathroom",
        report={"passed": False, "data_viable": False, "details": {"k": "v"}},
    )
    assert decision.reason_code == ReasonCode.FAIL_DATA_VIABILITY
    assert decision.details["reason_source"] == "derived"
    assert decision.details["k"] == "v"
    assert decision.details["capability_profile_id"] == "cap_profile_bathroom_v1"


def test_decide_room_emits_capability_profile_for_unknown_room():
    decision = GateEngine().decide_room(
        room="balcony",
        report={"passed": False, "metrics_passed": False},
    )
    assert decision.details["room_type"] == "generic"
    assert decision.details["capability_profile_id"] == "cap_profile_generic_v1"


def test_decide_room_fallback_for_invalid_reason_uses_leakage_precedence():
    decision = GateEngine().decide_room(
        room="kitchen",
        report={
            "passed": False,
            "reason_code": "nonsense",
            "leakage": {"resident_overlap": True, "time_overlap": True},
        },
    )
    assert decision.reason_code == ReasonCode.FAIL_LEAKAGE_RESIDENT


def test_decide_room_derives_uncertainty_reason_from_state():
    decision = GateEngine().decide_room(
        room="livingroom",
        report={"passed": False, "uncertainty_state": "low_confidence"},
    )
    assert decision.reason_code == ReasonCode.FAIL_UNCERTAINTY_LOW_CONFIDENCE
    assert decision.details["uncertainty_state"] == "low_confidence"
    assert decision.details["reason_source"] == "derived"


def test_decide_room_blocks_conflicting_uncertainty_flags():
    decision = GateEngine().decide_room(
        room="livingroom",
        report={
            "passed": False,
            "uncertainty": {"low_confidence": True, "unknown": True},
        },
    )
    assert decision.reason_code == ReasonCode.FAIL_UNCERTAINTY_CONFLICT
    assert "conflict_multiple_flags" in decision.details["uncertainty_error"]


def test_decide_room_timeline_mae_blocks_even_when_report_passed_true():
    decision = GateEngine().decide_room(
        room="bedroom",
        report={
            "passed": True,
            "metrics_passed": True,
            "timeline_metrics": {"duration_mae_minutes": 12.0, "fragmentation_rate": 0.1},
        },
    )
    assert decision.passed is False
    assert decision.reason_code == ReasonCode.FAIL_TIMELINE_MAE
    assert decision.details["pass_overridden_by_timeline_hard_gate"] is True


def test_decide_room_timeline_fragmentation_blocks():
    decision = GateEngine().decide_room(
        room="livingroom",
        report={
            "passed": False,
            "metrics_passed": True,
            "timeline_metrics": {"duration_mae_minutes": 8.0, "fragmentation_rate": 0.5},
        },
    )
    assert decision.passed is False
    assert decision.reason_code == ReasonCode.FAIL_TIMELINE_FRAGMENTATION


def test_decide_room_missing_required_timeline_metrics_blocks():
    decision = GateEngine().decide_room(
        room="bedroom",
        report={
            "passed": True,
            "metrics_passed": True,
            "details": {"require_timeline_metrics": True},
        },
    )
    assert decision.passed is False
    assert decision.reason_code == ReasonCode.FAIL_TIMELINE_METRICS_MISSING


def test_decide_run_empty_is_incomplete():
    run_decision = GateEngine().decide_run([])
    assert run_decision.passed is False
    assert run_decision.reason_code == ReasonCode.FAIL_RUN_INCOMPLETE
    assert run_decision.details["summary"] == "no_room_decisions"


def test_decide_run_uses_reason_precedence_not_input_order():
    room_decisions = [
        RoomDecision("r1", False, ReasonCode.FAIL_GATE_POLICY, {}),
        RoomDecision("r2", False, ReasonCode.FAIL_LEAKAGE_WINDOW, {}),
        RoomDecision("r3", False, ReasonCode.FAIL_DATA_VIABILITY, {}),
    ]
    run_decision = GateEngine().decide_run(room_decisions)
    assert run_decision.passed is False
    assert run_decision.reason_code == ReasonCode.FAIL_DATA_VIABILITY
    assert run_decision.details["failed_rooms"] == ["r1", "r2", "r3"]


def test_decide_run_pass_when_all_rooms_pass():
    room_decisions = [
        RoomDecision("r1", True, ReasonCode.PASS, {}),
        RoomDecision("r2", True, ReasonCode.PASS, {}),
    ]
    run_decision = GateEngine().decide_run(room_decisions)
    assert run_decision.passed is True
    assert run_decision.reason_code == ReasonCode.PASS
