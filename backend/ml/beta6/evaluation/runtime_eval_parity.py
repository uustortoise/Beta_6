"""Fixed-trace runtime/eval parity harness for Beta 6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..sequence.hmm_decoder import (
    DEFAULT_LABEL_MAP,
    DecodedTraceStep,
    DecoderPolicy,
    decode_sequence,
)


@dataclass(frozen=True)
class ParityMismatch:
    """Runtime/eval mismatch detail at one trace step."""

    step_index: int
    field: str
    runtime_value: Any
    eval_value: Any


@dataclass(frozen=True)
class ParityReport:
    """Parity result for one fixed trace."""

    passed: bool
    mismatch_count: int
    mismatches: List[ParityMismatch] = field(default_factory=list)


def decode_runtime_trace(
    trace_steps: Sequence[Mapping[str, Any]],
    *,
    label_map: Optional[Mapping[str, str]] = None,
    policy: DecoderPolicy = DecoderPolicy(),
) -> List[DecodedTraceStep]:
    return decode_sequence(trace_steps, label_map=label_map or DEFAULT_LABEL_MAP, policy=policy)


def decode_eval_trace(
    trace_steps: Sequence[Mapping[str, Any]],
    *,
    label_map: Optional[Mapping[str, str]] = None,
    policy: DecoderPolicy = DecoderPolicy(),
) -> List[DecodedTraceStep]:
    return decode_sequence(trace_steps, label_map=label_map or DEFAULT_LABEL_MAP, policy=policy)


def compare_runtime_eval_steps(
    runtime_steps: Sequence[DecodedTraceStep],
    eval_steps: Sequence[DecodedTraceStep],
) -> List[ParityMismatch]:
    mismatches: List[ParityMismatch] = []
    max_len = max(len(runtime_steps), len(eval_steps))
    for i in range(max_len):
        runtime_step = runtime_steps[i] if i < len(runtime_steps) else None
        eval_step = eval_steps[i] if i < len(eval_steps) else None
        if runtime_step is None or eval_step is None:
            mismatches.append(
                ParityMismatch(
                    step_index=i,
                    field="step_presence",
                    runtime_value=runtime_step,
                    eval_value=eval_step,
                )
            )
            continue
        if runtime_step.label != eval_step.label:
            mismatches.append(
                ParityMismatch(
                    step_index=i,
                    field="decoded_label",
                    runtime_value=runtime_step.label,
                    eval_value=eval_step.label,
                )
            )
        if runtime_step.source_label != eval_step.source_label:
            mismatches.append(
                ParityMismatch(
                    step_index=i,
                    field="source_label",
                    runtime_value=runtime_step.source_label,
                    eval_value=eval_step.source_label,
                )
            )
        if runtime_step.uncertainty_state != eval_step.uncertainty_state:
            mismatches.append(
                ParityMismatch(
                    step_index=i,
                    field="uncertainty_state",
                    runtime_value=runtime_step.uncertainty_state,
                    eval_value=eval_step.uncertainty_state,
                )
            )
    return mismatches


def run_fixed_trace_parity(
    trace_steps: Sequence[Mapping[str, Any]],
    *,
    runtime_label_map: Optional[Mapping[str, str]] = None,
    eval_label_map: Optional[Mapping[str, str]] = None,
    runtime_policy: DecoderPolicy = DecoderPolicy(),
    eval_policy: DecoderPolicy = DecoderPolicy(),
) -> ParityReport:
    """
    Evaluate runtime/eval parity on a fixed trace.

    CI should fail closed when `passed` is false.
    """
    runtime_steps = decode_runtime_trace(
        trace_steps,
        label_map=runtime_label_map,
        policy=runtime_policy,
    )
    eval_steps = decode_eval_trace(
        trace_steps,
        label_map=eval_label_map,
        policy=eval_policy,
    )
    mismatches = compare_runtime_eval_steps(runtime_steps, eval_steps)
    return ParityReport(
        passed=len(mismatches) == 0,
        mismatch_count=len(mismatches),
        mismatches=mismatches,
    )


def assert_fixed_trace_parity(
    trace_steps: Sequence[Mapping[str, Any]],
    *,
    runtime_label_map: Optional[Mapping[str, str]] = None,
    eval_label_map: Optional[Mapping[str, str]] = None,
    runtime_policy: DecoderPolicy = DecoderPolicy(),
    eval_policy: DecoderPolicy = DecoderPolicy(),
) -> ParityReport:
    report = run_fixed_trace_parity(
        trace_steps,
        runtime_label_map=runtime_label_map,
        eval_label_map=eval_label_map,
        runtime_policy=runtime_policy,
        eval_policy=eval_policy,
    )
    if not report.passed:
        mismatch_preview = [
            f"{m.step_index}:{m.field}:{m.runtime_value}!={m.eval_value}" for m in report.mismatches[:5]
        ]
        raise ValueError(f"runtime_eval_parity_failed ({'; '.join(mismatch_preview)})")
    return report
