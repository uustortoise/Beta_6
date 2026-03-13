"""Transition graph and duration-prior helpers for Beta 6 HMM/CRF decoders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..beta6_schema import load_validated_beta6_config


@dataclass(frozen=True)
class DurationPrior:
    min_minutes: float
    target_minutes: float
    max_minutes: float
    penalty_weight: float


@dataclass(frozen=True)
class TransitionPolicy:
    switch_penalty: float = 0.15
    impossible_transition_penalty: float = 1_000_000.0
    self_transition_bias: float = 0.05
    step_minutes: float = 1.0


@dataclass(frozen=True)
class DurationPriorPolicy:
    priors_by_label: Dict[str, DurationPrior]
    default_prior: DurationPrior
    transition: TransitionPolicy


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalize_room_key(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _resolve_transition_policy_for_context(
    *,
    policy: TransitionPolicy,
    room_name: str | None = None,
    resident_home_context: Optional[Mapping[str, Any]] = None,
) -> TransitionPolicy:
    """
    Apply resident/home context as deterministic policy adjustments.
    This path must depend only on explicit payload, never environment reads.
    """
    context = _as_mapping(resident_home_context)
    if not context:
        return policy

    switch_penalty = float(policy.switch_penalty)
    self_transition_bias = float(policy.self_transition_bias)
    impossible_transition_penalty = float(policy.impossible_transition_penalty)
    step_minutes = float(policy.step_minutes)

    household_type = _norm(context.get("household_type"))
    if household_type in {"multi", "multiple", "double"}:
        switch_penalty *= 1.10

    helper_presence = _norm(context.get("helper_presence"))
    if helper_presence in {"present", "yes", "true", "1", "part_time", "full_time"}:
        switch_penalty *= 1.08

    layout_topology = _as_mapping(context.get("layout_topology"))
    room_key = _normalize_room_key(room_name)
    neighbours_raw = layout_topology.get(room_key)
    if isinstance(neighbours_raw, str):
        neighbours = [_normalize_room_key(neighbours_raw)]
    elif isinstance(neighbours_raw, Sequence):
        neighbours = [_normalize_room_key(item) for item in neighbours_raw]
    else:
        neighbours = []
    neighbours = [item for item in neighbours if item]
    if neighbours:
        connectivity = min(len(set(neighbours)), 4)
        switch_penalty *= max(0.85, 1.0 - (0.03 * float(connectivity)))
        self_transition_bias += min(0.04, 0.01 * float(connectivity))

    return TransitionPolicy(
        switch_penalty=switch_penalty,
        impossible_transition_penalty=impossible_transition_penalty,
        self_transition_bias=self_transition_bias,
        step_minutes=step_minutes,
    )


def load_duration_prior_policy(path: str | Path | None) -> DurationPriorPolicy:
    policy_path = (
        Path(path).resolve()
        if path is not None
        else Path(__file__).resolve().parents[3] / "config" / "beta6_duration_prior_policy.yaml"
    )
    raw = load_validated_beta6_config(
        policy_path,
        expected_filename="beta6_duration_prior_policy.yaml",
    )
    duration_cfg = _as_mapping(raw).get("duration_priors")
    duration_cfg = _as_mapping(duration_cfg)
    default_cfg = _as_mapping(duration_cfg.get("default"))

    default_prior = DurationPrior(
        min_minutes=float(default_cfg.get("min_minutes", 1.0)),
        target_minutes=float(default_cfg.get("target_minutes", 12.0)),
        max_minutes=float(default_cfg.get("max_minutes", 90.0)),
        penalty_weight=float(default_cfg.get("penalty_weight", 0.8)),
    )

    priors_by_label: Dict[str, DurationPrior] = {}
    for label, cfg in _as_mapping(duration_cfg.get("by_label")).items():
        section = _as_mapping(cfg)
        priors_by_label[_norm(label)] = DurationPrior(
            min_minutes=float(section.get("min_minutes", default_prior.min_minutes)),
            target_minutes=float(section.get("target_minutes", default_prior.target_minutes)),
            max_minutes=float(section.get("max_minutes", default_prior.max_minutes)),
            penalty_weight=float(section.get("penalty_weight", default_prior.penalty_weight)),
        )

    transition_cfg = _as_mapping(raw).get("transition")
    transition_cfg = _as_mapping(transition_cfg)
    transition = TransitionPolicy(
        switch_penalty=float(transition_cfg.get("switch_penalty", 0.15)),
        impossible_transition_penalty=float(transition_cfg.get("impossible_transition_penalty", 1_000_000.0)),
        self_transition_bias=float(transition_cfg.get("self_transition_bias", 0.05)),
        step_minutes=float(transition_cfg.get("step_minutes", 1.0)),
    )
    return DurationPriorPolicy(
        priors_by_label=priors_by_label,
        default_prior=default_prior,
        transition=transition,
    )


def build_allowed_transition_map(
    labels: Sequence[str],
    *,
    disallowed_pairs: Optional[Iterable[Tuple[str, str]]] = None,
) -> Dict[Tuple[str, str], bool]:
    normalized = [_norm(label) for label in labels]
    disallowed = {(_norm(a), _norm(b)) for a, b in (disallowed_pairs or [])}
    allowed: Dict[Tuple[str, str], bool] = {}
    for src in normalized:
        for dst in normalized:
            allowed[(src, dst)] = (src, dst) not in disallowed
    return allowed


def build_transition_log_matrix(
    labels: Sequence[str],
    *,
    allowed_map: Optional[Mapping[Tuple[str, str], bool]] = None,
    policy: TransitionPolicy = TransitionPolicy(),
    room_name: str | None = None,
    resident_home_context: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    normalized = [_norm(label) for label in labels]
    n = len(normalized)
    if n == 0:
        raise ValueError("labels must not be empty")
    effective_policy = _resolve_transition_policy_for_context(
        policy=policy,
        room_name=room_name,
        resident_home_context=resident_home_context,
    )
    matrix = np.full((n, n), -float(effective_policy.switch_penalty), dtype=np.float64)
    for i in range(n):
        matrix[i, i] += float(effective_policy.self_transition_bias)

    if allowed_map:
        impossible = -float(effective_policy.impossible_transition_penalty)
        for i, src in enumerate(normalized):
            for j, dst in enumerate(normalized):
                if not bool(allowed_map.get((src, dst), True)):
                    matrix[i, j] = impossible
    return matrix


def duration_log_penalty(
    *,
    prev_label: str,
    next_label: str,
    run_length_steps: int,
    priors: Mapping[str, DurationPrior],
    default_prior: DurationPrior,
    step_minutes: float,
) -> float:
    """
    Soft duration penalty (option b): log-linear cost term, not hard constraints.
    """
    label = _norm(prev_label if next_label == prev_label else next_label)
    prior = priors.get(label, default_prior)
    minutes = max(float(run_length_steps), 1.0) * max(float(step_minutes), 1e-6)
    w = float(prior.penalty_weight)

    if next_label != prev_label:
        if minutes < float(prior.min_minutes):
            ratio = (float(prior.min_minutes) - minutes) / max(float(prior.min_minutes), 1e-6)
            return -w * ratio
        target_ratio = abs(minutes - float(prior.target_minutes)) / max(float(prior.target_minutes), 1e-6)
        return -0.25 * w * target_ratio

    if minutes > float(prior.max_minutes):
        ratio = (minutes - float(prior.max_minutes)) / max(float(prior.max_minutes), 1e-6)
        return -w * ratio
    target_ratio = abs(minutes - float(prior.target_minutes)) / max(float(prior.target_minutes), 1e-6)
    return -0.15 * w * target_ratio


__all__ = [
    "DurationPrior",
    "DurationPriorPolicy",
    "TransitionPolicy",
    "build_allowed_transition_map",
    "build_transition_log_matrix",
    "duration_log_penalty",
    "load_duration_prior_policy",
    "_resolve_transition_policy_for_context",
]
