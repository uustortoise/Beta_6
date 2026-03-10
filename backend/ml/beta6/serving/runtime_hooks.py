"""Beta 6 runtime hook logic for legacy prediction bridge."""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import numpy as np

from ml.exceptions import PredictionError
from utils.room_utils import normalize_room_name

from ..sequence import decode_hmm_with_duration_priors, load_duration_prior_policy
from ..sequence.crf_decoder import decode_crf_with_duration_priors
from .prediction import infer_with_unknown_path, load_unknown_policy


def _env_enabled(var_name: str, default: bool = False) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _env_enabled_with_master(
    var_name: str,
    *,
    master_var: str = "BETA6_PHASE4_RUNTIME_ENABLED",
    default: bool = False,
) -> bool:
    raw = os.getenv(var_name)
    if raw is not None:
        return _env_enabled(var_name, default=default)
    return _env_enabled(master_var, default=default)


def _env_path(var_name: str) -> str | None:
    raw = os.getenv(var_name)
    if raw is None:
        return None
    txt = str(raw).strip()
    return txt or None


def _resolve_sequence_mode() -> str:
    raw = os.getenv("BETA6_SEQUENCE_RUNTIME_MODE")
    if raw is not None:
        token = str(raw).strip().lower()
        if token in {"hmm", "crf"}:
            return token
    if _env_enabled_with_master("ENABLE_BETA6_CRF_RUNTIME", default=False):
        return "crf"
    return "hmm"


def _parse_room_set_override(raw: str) -> set[str]:
    rooms: set[str] = set()
    txt = str(raw or "").strip()
    if not txt:
        return rooms
    for token in txt.split(","):
        item = str(token).strip()
        if not item:
            continue
        rooms.add(normalize_room_name(item))
    return rooms


def _phase4_runtime_room_enabled(room_name: str) -> bool:
    scope_raw = os.getenv("BETA6_PHASE4_RUNTIME_ROOMS")
    if scope_raw is None:
        return True
    room_scope = _parse_room_set_override(scope_raw)
    if not room_scope:
        return True
    return normalize_room_name(room_name) in room_scope


def apply_beta6_unknown_abstain_runtime(
    *,
    room_name: str,
    y_pred_probs: np.ndarray,
    label_classes: list[str],
    final_labels: np.ndarray,
    low_conf_flags: list[bool],
    low_conf_hints: list[Optional[str]],
    confidence_scores: Optional[np.ndarray] = None,
    confidence_source: Optional[str] = None,
    apply_entropy_gate: bool = True,
    load_unknown_policy_fn: Callable[[str | None], Any] | None = None,
    infer_with_unknown_path_fn: Callable[..., dict[str, Any]] | None = None,
) -> tuple[np.ndarray, Optional[list[Optional[str]]], Optional[list[bool]], list[bool], list[Optional[str]]]:
    """
    Optional Beta 6 uncertainty path integration.

    Uses Beta 6 unknown policy to classify uncertainty semantics and maps them into
    active runtime tokens:
    - `low_confidence` stays `low_confidence`
    - `unknown` / `outside_sensed_space` map to `unknown`
    """
    if not _env_enabled_with_master("ENABLE_BETA6_UNKNOWN_ABSTAIN_RUNTIME", default=False):
        return (
            final_labels,
            None,
            None,
            low_conf_flags,
            low_conf_hints,
        )
    if not _phase4_runtime_room_enabled(room_name):
        return (
            final_labels,
            None,
            None,
            low_conf_flags,
            low_conf_hints,
        )

    try:
        load_policy = load_unknown_policy_fn or load_unknown_policy
        infer_unknown = infer_with_unknown_path_fn or infer_with_unknown_path
        policy_path = _env_path("BETA6_UNKNOWN_POLICY_PATH")
        policy = load_policy(policy_path)
        inference = infer_unknown(
            probabilities=np.asarray(y_pred_probs, dtype=float),
            labels=[str(lbl).strip().lower() for lbl in label_classes],
            policy=policy,
            outside_sensed_space_scores=None,
            confidence_scores=confidence_scores,
            confidence_source=confidence_source,
            preexisting_low_conf_flags=list(low_conf_flags),
            apply_confidence_gate=False,
            apply_entropy_gate=apply_entropy_gate,
        )
    except Exception as exc:
        raise PredictionError(
            f"Beta6 unknown/abstain runtime path failed for {room_name}: {type(exc).__name__}: {exc}"
        ) from exc

    out_labels = np.asarray(final_labels, dtype=object).copy()
    uncertainty_states = list(inference.get("uncertainty_states", []))
    if len(uncertainty_states) != len(out_labels):
        raise PredictionError(
            f"Beta6 unknown/abstain runtime path length mismatch for {room_name}: "
            f"{len(uncertainty_states)} vs {len(out_labels)}"
        )
    beta6_abstain_flags: list[bool] = [False] * len(out_labels)

    for idx, raw_state in enumerate(uncertainty_states):
        state = None if raw_state is None else str(raw_state).strip().lower()
        if state is None:
            continue
        beta6_abstain_flags[idx] = True
        if state == "low_confidence":
            out_labels[idx] = "low_confidence"
            if idx < len(low_conf_flags):
                low_conf_flags[idx] = True
            if idx < len(low_conf_hints) and not low_conf_hints[idx]:
                low_conf_hints[idx] = str(out_labels[idx])
        elif state in {"unknown", "outside_sensed_space"}:
            out_labels[idx] = "unknown"
        else:
            raise PredictionError(
                f"unsupported Beta6 uncertainty state for {room_name}: {state}"
            )

    return out_labels, uncertainty_states, beta6_abstain_flags, low_conf_flags, low_conf_hints


def apply_beta6_hmm_runtime(
    *,
    room_name: str,
    y_pred_probs: np.ndarray,
    label_classes: list[str],
    final_labels: np.ndarray,
    low_conf_flags: list[bool],
    load_duration_prior_policy_fn: Callable[[str | None], Any] | None = None,
    decode_hmm_with_duration_priors_fn: Callable[..., Any] | None = None,
) -> np.ndarray:
    """
    Optional Beta 6 sequence smoothing with soft duration-prior penalties.

    Supports `hmm` (default) and `crf` modes.
    Applied after thresholding/hysteresis and before unknown-abstain routing.
    Low-confidence windows are preserved as-is.
    """
    if not _env_enabled_with_master("ENABLE_BETA6_HMM_RUNTIME", default=False):
        return final_labels
    if not _phase4_runtime_room_enabled(room_name):
        return final_labels

    out_labels = np.asarray(final_labels, dtype=object).copy()
    if len(out_labels) == 0:
        return out_labels

    probs = np.asarray(y_pred_probs, dtype=float)
    if probs.ndim != 2 or probs.shape[0] != len(out_labels):
        raise PredictionError(
            f"Beta6 HMM runtime path shape mismatch for {room_name}: probs={probs.shape} labels={len(out_labels)}"
        )
    if probs.shape[1] != len(label_classes):
        raise PredictionError(
            f"Beta6 HMM runtime class mismatch for {room_name}: probs_width={probs.shape[1]} classes={len(label_classes)}"
        )

    sequence_mode = _resolve_sequence_mode()
    try:
        load_duration_policy = load_duration_prior_policy_fn or load_duration_prior_policy
        duration_policy_path = _env_path("BETA6_HMM_DURATION_POLICY_PATH")
        duration_policy = load_duration_policy(duration_policy_path)
        observation_log_probs = np.log(np.clip(probs, 1e-9, 1.0))
        normalized_labels = [str(lbl).strip().lower() for lbl in label_classes]
        if sequence_mode == "crf":
            hmm_result = decode_crf_with_duration_priors(
                observation_log_probs=observation_log_probs,
                labels=normalized_labels,
                duration_policy=duration_policy,
            )
        else:
            decode_hmm = decode_hmm_with_duration_priors_fn or decode_hmm_with_duration_priors
            hmm_result = decode_hmm(
                observation_log_probs=observation_log_probs,
                labels=normalized_labels,
                duration_policy=duration_policy,
            )
    except Exception as exc:
        raise PredictionError(
            f"Beta6 sequence runtime path failed for {room_name} mode={sequence_mode}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    decoded = [str(lbl).strip().lower() for lbl in hmm_result.labels]
    if len(decoded) != len(out_labels):
        raise PredictionError(
            f"Beta6 HMM runtime path length mismatch for {room_name}: {len(decoded)} vs {len(out_labels)}"
        )

    low_conf_arr = np.asarray(low_conf_flags, dtype=bool)
    if len(low_conf_arr) != len(out_labels):
        raise PredictionError(
            f"Beta6 HMM runtime low_conf length mismatch for {room_name}: {len(low_conf_arr)} vs {len(out_labels)}"
        )

    for idx, hmm_label in enumerate(decoded):
        current = str(out_labels[idx]).strip().lower()
        if low_conf_arr[idx] or current in {"low_confidence", "unknown"}:
            continue
        if hmm_label:
            out_labels[idx] = hmm_label
    return out_labels


def apply_beta6_sequence_runtime(**kwargs) -> np.ndarray:
    """Named alias for future call sites using explicit sequence terminology."""
    return apply_beta6_hmm_runtime(**kwargs)


__all__ = [
    "apply_beta6_hmm_runtime",
    "apply_beta6_sequence_runtime",
    "apply_beta6_unknown_abstain_runtime",
]
