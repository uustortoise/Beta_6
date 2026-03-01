"""Decision contracts and reason codes for Beta 6 gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class ReasonCode(str, Enum):
    """Deterministic reason codes for pass/fail decisions."""

    PASS = "pass"
    FAIL_GATE_POLICY = "fail_gate_policy"
    FAIL_DATA_VIABILITY = "fail_data_viability"
    FAIL_LEAKAGE_RESIDENT = "fail_leakage_resident"
    FAIL_LEAKAGE_TIME = "fail_leakage_time"
    FAIL_LEAKAGE_WINDOW = "fail_leakage_window"
    FAIL_UNCERTAINTY_LOW_CONFIDENCE = "fail_uncertainty_low_confidence"
    FAIL_UNCERTAINTY_UNKNOWN = "fail_uncertainty_unknown"
    FAIL_UNCERTAINTY_OUTSIDE_SENSED_SPACE = "fail_uncertainty_outside_sensed_space"
    FAIL_UNCERTAINTY_CONFLICT = "fail_uncertainty_conflict"
    FAIL_UNCERTAINTY_INVALID = "fail_uncertainty_invalid"
    FAIL_RUNTIME_EVAL_PARITY = "fail_runtime_eval_parity"
    FAIL_TIMELINE_METRICS_MISSING = "fail_timeline_metrics_missing"
    FAIL_TIMELINE_MAE = "fail_timeline_mae"
    FAIL_TIMELINE_FRAGMENTATION = "fail_timeline_fragmentation"
    FAIL_RUN_INCOMPLETE = "fail_run_incomplete"
    FAIL_UNKNOWN_REASON = "fail_unknown_reason"
    ROLLBACK_TRIGGERED = "rollback_triggered"
    ROLLBACK_MISSING_TARGET = "rollback_missing_target"
    FALLBACK_ACTIVATED = "fallback_activated"
    FALLBACK_CLEARED = "fallback_cleared"
    FALLBACK_ALREADY_ACTIVE = "fallback_already_active"
    FALLBACK_NOT_ACTIVE = "fallback_not_active"
    FALLBACK_MISSING_TARGET = "fallback_missing_target"


class UncertaintyClass(str, Enum):
    """Canonical uncertainty taxonomy (must never be merged)."""

    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN = "unknown"
    OUTSIDE_SENSED_SPACE = "outside_sensed_space"


UNCERTAINTY_REASON_CODE_MAP: Dict[UncertaintyClass, ReasonCode] = {
    UncertaintyClass.LOW_CONFIDENCE: ReasonCode.FAIL_UNCERTAINTY_LOW_CONFIDENCE,
    UncertaintyClass.UNKNOWN: ReasonCode.FAIL_UNCERTAINTY_UNKNOWN,
    UncertaintyClass.OUTSIDE_SENSED_SPACE: ReasonCode.FAIL_UNCERTAINTY_OUTSIDE_SENSED_SPACE,
}

_FLAG_TO_UNCERTAINTY: Dict[str, UncertaintyClass] = {
    "low_confidence": UncertaintyClass.LOW_CONFIDENCE,
    "unknown": UncertaintyClass.UNKNOWN,
    "outside_sensed_space": UncertaintyClass.OUTSIDE_SENSED_SPACE,
}


@dataclass(frozen=True)
class UncertaintyResolution:
    """
    Resolved uncertainty class and deterministic failure reason.

    `state` is mutually exclusive; `error` is set when contract is violated.
    """

    state: Optional[UncertaintyClass]
    source: Optional[str]
    reason_code: Optional[ReasonCode]
    error: Optional[str] = None


def _parse_uncertainty_state_token(raw_state: Any) -> Optional[UncertaintyClass]:
    if raw_state is None:
        return None
    if not isinstance(raw_state, str):
        raise ValueError("uncertainty_state must be a string")
    token = raw_state.strip()
    if not token:
        raise ValueError("uncertainty_state must be non-empty when present")
    try:
        return UncertaintyClass(token)
    except ValueError as exc:
        raise ValueError(f"unsupported uncertainty_state: {token}") from exc


def resolve_uncertainty(report: Mapping[str, Any]) -> UncertaintyResolution:
    """
    Resolve uncertainty taxonomy from report payload.

    Accepted shapes:
    1. `uncertainty: "<state>"`.
    2. `uncertainty_state: "<state>"`.
    3. `uncertainty: {state: "<state>"}`.
    4. Boolean flags at top-level or under `uncertainty` mapping.
    """
    uncertainty_payload = report.get("uncertainty")
    uncertainty_map = uncertainty_payload if isinstance(uncertainty_payload, Mapping) else {}

    explicit_tokens: List[tuple[str, UncertaintyClass]] = []
    parse_errors: List[str] = []

    explicit_candidates = [
        ("report.uncertainty", uncertainty_payload if isinstance(uncertainty_payload, str) else None),
        (
            "report.uncertainty.state",
            uncertainty_map.get("state") if isinstance(uncertainty_payload, Mapping) else None,
        ),
        ("report.uncertainty_state", report.get("uncertainty_state")),
    ]

    for source, candidate in explicit_candidates:
        if candidate is None:
            continue
        try:
            parsed = _parse_uncertainty_state_token(candidate)
        except ValueError as exc:
            parse_errors.append(f"{source}:{exc}")
            continue
        if parsed is not None:
            explicit_tokens.append((source, parsed))

    if parse_errors:
        return UncertaintyResolution(
            state=None,
            source=None,
            reason_code=ReasonCode.FAIL_UNCERTAINTY_INVALID,
            error="invalid_token|" + "|".join(parse_errors),
        )

    explicit_state: Optional[UncertaintyClass] = None
    explicit_source: Optional[str] = None
    for source, state in explicit_tokens:
        if explicit_state is None:
            explicit_state = state
            explicit_source = source
            continue
        if state != explicit_state:
            return UncertaintyResolution(
                state=None,
                source=None,
                reason_code=ReasonCode.FAIL_UNCERTAINTY_CONFLICT,
                error=(
                    "conflict_explicit_states|"
                    f"{explicit_source}:{explicit_state.value}|{source}:{state.value}"
                ),
            )

    flag_values: Dict[UncertaintyClass, List[tuple[str, bool]]] = {state: [] for state in UncertaintyClass}
    for flag_name, state in _FLAG_TO_UNCERTAINTY.items():
        top_value = report.get(flag_name)
        if top_value is not None:
            if not isinstance(top_value, bool):
                return UncertaintyResolution(
                    state=None,
                    source=None,
                    reason_code=ReasonCode.FAIL_UNCERTAINTY_INVALID,
                    error=f"invalid_flag_type|report.{flag_name}",
                )
            flag_values[state].append((f"report.{flag_name}", top_value))

        if isinstance(uncertainty_payload, Mapping) and flag_name in uncertainty_map:
            nested_value = uncertainty_map[flag_name]
            if not isinstance(nested_value, bool):
                return UncertaintyResolution(
                    state=None,
                    source=None,
                    reason_code=ReasonCode.FAIL_UNCERTAINTY_INVALID,
                    error=f"invalid_flag_type|report.uncertainty.{flag_name}",
                )
            flag_values[state].append((f"report.uncertainty.{flag_name}", nested_value))

    positive_flags: List[tuple[UncertaintyClass, str]] = []
    for state, values in flag_values.items():
        if not values:
            continue
        bool_values = {value for _, value in values}
        if len(bool_values) > 1:
            return UncertaintyResolution(
                state=None,
                source=None,
                reason_code=ReasonCode.FAIL_UNCERTAINTY_CONFLICT,
                error="conflict_flag_values|" + "|".join(f"{src}:{val}" for src, val in values),
            )
        if True in bool_values:
            positive_flags.append((state, values[0][0]))

    if len(positive_flags) > 1:
        return UncertaintyResolution(
            state=None,
            source=None,
            reason_code=ReasonCode.FAIL_UNCERTAINTY_CONFLICT,
            error="conflict_multiple_flags|" + "|".join(
                f"{source}:{state.value}" for state, source in positive_flags
            ),
        )

    if explicit_state is not None and positive_flags:
        flag_state, flag_source = positive_flags[0]
        if flag_state != explicit_state:
            return UncertaintyResolution(
                state=None,
                source=None,
                reason_code=ReasonCode.FAIL_UNCERTAINTY_CONFLICT,
                error=(
                    "conflict_explicit_vs_flag|"
                    f"{explicit_source}:{explicit_state.value}|{flag_source}:{flag_state.value}"
                ),
            )

    if explicit_state is not None:
        return UncertaintyResolution(
            state=explicit_state,
            source=explicit_source,
            reason_code=UNCERTAINTY_REASON_CODE_MAP[explicit_state],
        )

    if positive_flags:
        flag_state, flag_source = positive_flags[0]
        return UncertaintyResolution(
            state=flag_state,
            source=flag_source,
            reason_code=UNCERTAINTY_REASON_CODE_MAP[flag_state],
        )

    return UncertaintyResolution(state=None, source=None, reason_code=None, error=None)


@dataclass(frozen=True)
class RoomDecision:
    """Room-level pass/fail decision."""

    room: str
    passed: bool
    reason_code: ReasonCode
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunDecision:
    """Run-level pass/fail decision rolled up from room decisions."""

    passed: bool
    reason_code: ReasonCode
    room_decisions: List[RoomDecision] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
