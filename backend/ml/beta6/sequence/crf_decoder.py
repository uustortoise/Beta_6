"""Phase 5 CRF decoder with adjacency constraints and duration-prior penalties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .hmm_decoder import _compute_ping_pong_rate
from .transition_builder import (
    DurationPriorPolicy,
    TransitionPolicy,
    build_allowed_transition_map,
    build_transition_log_matrix,
    duration_log_penalty,
    load_duration_prior_policy,
)


@dataclass(frozen=True)
class CRFDecodeResult:
    labels: List[str]
    state_indices: List[int]
    score: float
    ping_pong_rate: float
    transition_log_matrix: np.ndarray


def _normalize_observation_log_probs(observation_log_probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(observation_log_probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"observation_log_probs must be 2D, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("observation_log_probs contains non-finite values")
    return arr


def fit_transition_log_matrix_from_sequences(
    *,
    label_sequences: Sequence[Sequence[str]],
    labels: Sequence[str],
    smoothing: float = 1.0,
    allowed_map: Optional[Mapping[Tuple[str, str], bool]] = None,
    disallowed_pairs: Optional[Iterable[Tuple[str, str]]] = None,
    policy: TransitionPolicy = TransitionPolicy(),
    room_name: str | None = None,
    resident_home_context: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Fit log transition potentials from labeled sequences with smoothing."""
    states = [str(label).strip().lower() for label in labels]
    if not states:
        raise ValueError("labels must not be empty")
    resolved_allowed_map = (
        dict(allowed_map)
        if allowed_map is not None
        else build_allowed_transition_map(states, disallowed_pairs=disallowed_pairs)
    )
    index = {label: i for i, label in enumerate(states)}
    n = len(states)
    counts = np.full((n, n), float(max(smoothing, 1e-9)), dtype=np.float64)
    for seq in label_sequences:
        if not seq:
            continue
        normalized = [str(token).strip().lower() for token in seq]
        for prev, nxt in zip(normalized[:-1], normalized[1:]):
            if prev not in index or nxt not in index:
                continue
            counts[index[prev], index[nxt]] += 1.0

    probs = counts / np.clip(np.sum(counts, axis=1, keepdims=True), 1e-9, None)
    learned = np.log(np.clip(probs, 1e-12, 1.0))
    base = build_transition_log_matrix(
        states,
        allowed_map=resolved_allowed_map,
        policy=policy,
        room_name=room_name,
        resident_home_context=resident_home_context,
    )
    transition = learned + base
    if resolved_allowed_map:
        impossible = -float(policy.impossible_transition_penalty)
        for src_idx, src in enumerate(states):
            for dst_idx, dst in enumerate(states):
                if not bool(resolved_allowed_map.get((src, dst), True)):
                    transition[src_idx, dst_idx] = impossible
    return transition


def decode_crf_with_duration_priors(
    *,
    observation_log_probs: np.ndarray,
    labels: Sequence[str],
    duration_policy: Optional[DurationPriorPolicy] = None,
    transition_log_matrix: Optional[np.ndarray] = None,
    label_sequences_for_fit: Optional[Sequence[Sequence[str]]] = None,
    disallowed_pairs: Optional[Iterable[Tuple[str, str]]] = None,
    smoothing: float = 1.0,
    room_name: str | None = None,
    resident_home_context: Optional[Mapping[str, Any]] = None,
) -> CRFDecodeResult:
    """
    Decode with linear-chain CRF-style transition potentials + duration penalties.

    The decode uses Viterbi over:
    observation + transition potentials + soft duration-prior penalties.
    """
    observations = _normalize_observation_log_probs(observation_log_probs)
    states = [str(label).strip().lower() for label in labels]
    if not states:
        raise ValueError("labels must not be empty")
    t_steps, n_states = observations.shape
    if n_states != len(states):
        raise ValueError("observation_log_probs width must equal labels length")

    policy = duration_policy or load_duration_prior_policy(None)
    allowed_map = build_allowed_transition_map(states, disallowed_pairs=disallowed_pairs)
    if transition_log_matrix is not None:
        transition = np.asarray(transition_log_matrix, dtype=np.float64)
    elif label_sequences_for_fit:
        transition = fit_transition_log_matrix_from_sequences(
            label_sequences=label_sequences_for_fit,
            labels=states,
            smoothing=smoothing,
            allowed_map=allowed_map,
            policy=policy.transition,
            room_name=room_name,
            resident_home_context=resident_home_context,
        )
    else:
        transition = build_transition_log_matrix(
            states,
            allowed_map=allowed_map,
            policy=policy.transition,
            room_name=room_name,
            resident_home_context=resident_home_context,
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
    return CRFDecodeResult(
        labels=decoded_labels,
        state_indices=path,
        score=best_score,
        ping_pong_rate=_compute_ping_pong_rate(path),
        transition_log_matrix=transition,
    )


__all__ = [
    "CRFDecodeResult",
    "decode_crf_with_duration_priors",
    "fit_transition_log_matrix_from_sequences",
]
