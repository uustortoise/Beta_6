import pytest

from ml.beta6.runtime_eval_parity import (
    DecoderPolicy,
    assert_fixed_trace_parity,
    run_fixed_trace_parity,
)


def _trace():
    return [
        {"label": "occupied"},
        {"label": "unoccupied"},
        {"label": "occupied"},
        {"label": "occupied", "uncertainty_state": "low_confidence"},
    ]


def test_fixed_trace_parity_passes_when_mapping_and_decoder_match():
    report = run_fixed_trace_parity(_trace())
    assert report.passed is True
    assert report.mismatch_count == 0


def test_fixed_trace_parity_detects_label_mapping_mismatch():
    runtime_map = {
        "occupied": "occupied",
        "unoccupied": "unoccupied",
    }
    eval_map = {
        "occupied": "occupied",
        "unoccupied": "occupied",
    }
    report = run_fixed_trace_parity(
        _trace(),
        runtime_label_map=runtime_map,
        eval_label_map=eval_map,
    )
    assert report.passed is False
    assert any(m.field in {"decoded_label", "source_label"} for m in report.mismatches)


def test_fixed_trace_parity_detects_decoder_semantics_mismatch():
    report = run_fixed_trace_parity(
        _trace(),
        runtime_policy=DecoderPolicy(spike_suppression=True),
        eval_policy=DecoderPolicy(spike_suppression=False),
    )
    assert report.passed is False
    assert any(m.field == "decoded_label" for m in report.mismatches)


def test_assert_fixed_trace_parity_raises_on_mismatch():
    with pytest.raises(ValueError, match="runtime_eval_parity_failed"):
        assert_fixed_trace_parity(
            _trace(),
            runtime_policy=DecoderPolicy(spike_suppression=True),
            eval_policy=DecoderPolicy(spike_suppression=False),
        )
