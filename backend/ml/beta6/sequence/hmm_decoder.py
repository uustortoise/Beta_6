"""Shared deterministic decoder semantics for runtime and evaluation paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ml.beta6.contracts.decisions import resolve_uncertainty

from ..beta6_schema import load_validated_beta6_config

from .transition_builder import (
    DurationPrior,
    DurationPriorPolicy,
    build_transition_log_matrix,
    duration_log_penalty,
    load_duration_prior_policy,
)


ABSTAIN_LABEL = "abstain"
_PARITY_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "beta6_runtime_eval_parity.yaml"


@dataclass(frozen=True)
class DecodedTraceStep:
    step_index: int
    label: str
    uncertainty_state: Optional[str]
    source_label: str


@dataclass(frozen=True)
class DecoderPolicy:
    spike_suppression: bool = True


def load_runtime_eval_parity_config(
    config_path: Path | None = None,
) -> Tuple[Dict[str, str], DecoderPolicy]:
    path = config_path or _PARITY_CONFIG_PATH
    payload = load_validated_beta6_config(path, expected_filename="beta6_runtime_eval_parity.yaml")

    raw_map = payload.get("label_map")
    label_map = {}
    if isinstance(raw_map, Mapping):
        for key, value in raw_map.items():
            normalized_key = str(key).strip().lower()
            normalized_value = str(value).strip().lower()
            if normalized_key and normalized_value:
                label_map[normalized_key] = normalized_value

    raw_policy = payload.get("decoder_policy")
    policy = DecoderPolicy()
    if isinstance(raw_policy, Mapping):
        policy = DecoderPolicy(
            spike_suppression=bool(raw_policy.get("spike_suppression", policy.spike_suppression)),
        )
    return label_map, policy


DEFAULT_LABEL_MAP, DEFAULT_DECODER_POLICY = load_runtime_eval_parity_config()


def _canonicalize_label(raw_label: Any, label_map: Mapping[str, str]) -> str:
    if not isinstance(raw_label, str):
        raise ValueError(f"label must be a string, got {type(raw_label).__name__}")
    token = raw_label.strip().lower()
    if not token:
        raise ValueError("label must be non-empty")
    mapped = label_map.get(token)
    if mapped is None:
        raise ValueError(f"unsupported label token: {raw_label}")
    return str(mapped).strip().lower()


def decode_sequence(
    trace_steps: Sequence[Mapping[str, Any]],
    *,
    label_map: Optional[Mapping[str, str]] = None,
    policy: DecoderPolicy = DEFAULT_DECODER_POLICY,
) -> List[DecodedTraceStep]:
    decoder_map = dict(label_map) if label_map is not None else dict(DEFAULT_LABEL_MAP)
    decoded: List[DecodedTraceStep] = []
    for index, step in enumerate(trace_steps):
        source_label = _canonicalize_label(step.get("label"), decoder_map)
        uncertainty = resolve_uncertainty(step)
        uncertainty_state = uncertainty.state.value if uncertainty.state is not None else None
        decoded_label = ABSTAIN_LABEL if uncertainty_state is not None else source_label
        decoded.append(
            DecodedTraceStep(
                step_index=index,
                label=decoded_label,
                uncertainty_state=uncertainty_state,
                source_label=source_label,
            )
        )

    if policy.spike_suppression and len(decoded) >= 3:
        mutable = list(decoded)
        for i in range(1, len(mutable) - 1):
            left = mutable[i - 1]
            center = mutable[i]
            right = mutable[i + 1]
            if center.uncertainty_state is not None:
                continue
            if left.label == right.label and center.label != left.label:
                mutable[i] = DecodedTraceStep(
                    step_index=center.step_index,
                    label=left.label,
                    uncertainty_state=center.uncertainty_state,
                    source_label=center.source_label,
                )
        decoded = mutable
    return decoded


@dataclass(frozen=True)
class HMMDecodeResult:
    labels: List[str]
    state_indices: List[int]
    score: float
    ping_pong_rate: float


def _normalize_observation_log_probs(observation_log_probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(observation_log_probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"observation_log_probs must be 2D, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("observation_log_probs contains non-finite values")
    return arr


def _compute_ping_pong_rate(path: Sequence[int]) -> float:
    if len(path) < 3:
        return 0.0
    flips = 0
    for i in range(1, len(path) - 1):
        if path[i - 1] == path[i + 1] and path[i] != path[i - 1]:
            flips += 1
    return float(flips) / float(max(len(path) - 2, 1))


def decode_hmm_with_duration_priors(
    *,
    observation_log_probs: np.ndarray,
    labels: Sequence[str],
    transition_log_matrix: Optional[np.ndarray] = None,
    duration_policy: Optional[DurationPriorPolicy] = None,
    room_name: str | None = None,
    resident_home_context: Optional[Mapping[str, Any]] = None,
) -> HMMDecodeResult:
    """
    Decode sequence with Viterbi + soft duration-prior penalties (option b).
    """
    observations = _normalize_observation_log_probs(observation_log_probs)
    states = [str(label).strip().lower() for label in labels]
    if not states:
        raise ValueError("labels must not be empty")
    t_steps, n_states = observations.shape
    if n_states != len(states):
        raise ValueError("observation_log_probs width must equal labels length")

    policy = duration_policy or load_duration_prior_policy(
        Path(__file__).resolve().parents[3] / "config" / "beta6_duration_prior_policy.yaml"
    )
    transition = (
        np.asarray(transition_log_matrix, dtype=np.float64)
        if transition_log_matrix is not None
        else build_transition_log_matrix(
            states,
            policy=policy.transition,
            room_name=room_name,
            resident_home_context=resident_home_context,
        )
    )
    if transition.shape != (n_states, n_states):
        raise ValueError(f"transition_log_matrix must be {(n_states, n_states)}, got {transition.shape}")

    dp = np.full((t_steps, n_states), -np.inf, dtype=np.float64)
    parent = np.full((t_steps, n_states), -1, dtype=np.int32)
    runlen = np.ones((t_steps, n_states), dtype=np.int32)
    dp[0, :] = observations[0, :]

    for t in range(1, t_steps):
        for j in range(n_states):
            best_score = -np.inf
            best_parent = -1
            best_run = 1
            for i in range(n_states):
                prev_score = dp[t - 1, i]
                if prev_score == -np.inf:
                    continue
                prev_run = int(runlen[t - 1, i])
                next_run = prev_run + 1 if i == j else 1
                penalty = duration_log_penalty(
                    prev_label=states[i],
                    next_label=states[j],
                    run_length_steps=prev_run,
                    priors=policy.priors_by_label,
                    default_prior=policy.default_prior,
                    step_minutes=policy.transition.step_minutes,
                )
                score = prev_score + transition[i, j] + penalty + observations[t, j]
                if score > best_score:
                    best_score = score
                    best_parent = i
                    best_run = next_run
            dp[t, j] = best_score
            parent[t, j] = best_parent
            runlen[t, j] = best_run

    last_state = int(np.argmax(dp[-1, :]))
    best_score = float(dp[-1, last_state])
    path = [last_state]
    for t in range(t_steps - 1, 0, -1):
        prev = int(parent[t, path[-1]])
        if prev < 0:
            prev = path[-1]
        path.append(prev)
    path.reverse()
    decoded_labels = [states[idx] for idx in path]
    return HMMDecodeResult(
        labels=decoded_labels,
        state_indices=path,
        score=best_score,
        ping_pong_rate=_compute_ping_pong_rate(path),
    )


__all__ = [
    "ABSTAIN_LABEL",
    "DEFAULT_DECODER_POLICY",
    "DEFAULT_LABEL_MAP",
    "DecodedTraceStep",
    "DecoderPolicy",
    "HMMDecodeResult",
    "decode_hmm_with_duration_priors",
    "decode_sequence",
    "load_runtime_eval_parity_config",
]
