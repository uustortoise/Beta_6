from ml.beta6.capability_profiles import select_capability_profile
from ml.beta6.contracts.decisions import ReasonCode
from ml.beta6.timeline_hard_gates import evaluate_timeline_hard_gates


def test_timeline_hard_gates_not_checked_when_metrics_absent():
    profile = select_capability_profile("bedroom")
    result = evaluate_timeline_hard_gates({}, profile)
    assert result.checked is False
    assert result.passed is True
    assert result.reason_code is None


def test_timeline_hard_gates_fail_when_metrics_missing_fields():
    profile = select_capability_profile("bedroom")
    result = evaluate_timeline_hard_gates({"timeline_metrics": {"duration_mae_minutes": 5.0}}, profile)
    assert result.checked is True
    assert result.passed is False
    assert result.reason_code == ReasonCode.FAIL_TIMELINE_METRICS_MISSING


def test_timeline_hard_gates_fail_mae_before_fragmentation():
    profile = select_capability_profile("bedroom")
    result = evaluate_timeline_hard_gates(
        {"timeline_metrics": {"duration_mae_minutes": 11.0, "fragmentation_rate": 0.4}},
        profile,
    )
    assert result.checked is True
    assert result.passed is False
    assert result.reason_code == ReasonCode.FAIL_TIMELINE_MAE


def test_timeline_hard_gates_fail_fragmentation():
    profile = select_capability_profile("livingroom")
    result = evaluate_timeline_hard_gates(
        {"timeline_metrics": {"duration_mae_minutes": 9.0, "fragmentation_rate": 0.5}},
        profile,
    )
    assert result.checked is True
    assert result.passed is False
    assert result.reason_code == ReasonCode.FAIL_TIMELINE_FRAGMENTATION


def test_timeline_hard_gates_pass_when_below_thresholds():
    profile = select_capability_profile("bathroom")
    result = evaluate_timeline_hard_gates(
        {"timeline_metrics": {"duration_mae_minutes": 8.0, "fragmentation_rate": 0.2}},
        profile,
    )
    assert result.checked is True
    assert result.passed is True
    assert result.reason_code is None
