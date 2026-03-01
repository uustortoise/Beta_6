from ml.beta6.contracts.decisions import (
    ReasonCode,
    UncertaintyClass,
    resolve_uncertainty,
)


def test_resolve_uncertainty_from_explicit_top_level_token():
    resolved = resolve_uncertainty({"uncertainty": "unknown"})
    assert resolved.state == UncertaintyClass.UNKNOWN
    assert resolved.source == "report.uncertainty"
    assert resolved.reason_code == ReasonCode.FAIL_UNCERTAINTY_UNKNOWN
    assert resolved.error is None


def test_resolve_uncertainty_from_nested_boolean_flag():
    resolved = resolve_uncertainty({"uncertainty": {"outside_sensed_space": True}})
    assert resolved.state == UncertaintyClass.OUTSIDE_SENSED_SPACE
    assert resolved.reason_code == ReasonCode.FAIL_UNCERTAINTY_OUTSIDE_SENSED_SPACE
    assert resolved.error is None


def test_resolve_uncertainty_conflict_when_multiple_flags_enabled():
    resolved = resolve_uncertainty({"low_confidence": True, "unknown": True})
    assert resolved.state is None
    assert resolved.reason_code == ReasonCode.FAIL_UNCERTAINTY_CONFLICT
    assert "conflict_multiple_flags" in str(resolved.error)


def test_resolve_uncertainty_invalid_for_unknown_token():
    resolved = resolve_uncertainty({"uncertainty_state": "maybe"})
    assert resolved.state is None
    assert resolved.reason_code == ReasonCode.FAIL_UNCERTAINTY_INVALID
    assert "unsupported uncertainty_state" in str(resolved.error)


def test_resolve_uncertainty_conflict_between_explicit_and_flag():
    resolved = resolve_uncertainty(
        {
            "uncertainty_state": "low_confidence",
            "uncertainty": {"unknown": True},
        }
    )
    assert resolved.state is None
    assert resolved.reason_code == ReasonCode.FAIL_UNCERTAINTY_CONFLICT
    assert "conflict_explicit_vs_flag" in str(resolved.error)


def test_resolve_uncertainty_accepts_consistent_explicit_and_flag():
    resolved = resolve_uncertainty(
        {
            "uncertainty_state": "low_confidence",
            "uncertainty": {"low_confidence": True},
        }
    )
    assert resolved.state == UncertaintyClass.LOW_CONFIDENCE
    assert resolved.reason_code == ReasonCode.FAIL_UNCERTAINTY_LOW_CONFIDENCE
    assert resolved.error is None
