import numpy as np

from ml.beta6.sequence.hmm_decoder import decode_hmm_with_duration_priors
from ml.beta6.sequence.transition_builder import (
    DurationPrior,
    DurationPriorPolicy,
    TransitionPolicy,
    build_allowed_transition_map,
    build_transition_log_matrix,
)


def test_hmm_respects_impossible_transition_mask():
    labels = ["a", "b"]
    allowed = build_allowed_transition_map(labels, disallowed_pairs=[("a", "b")])
    transition = build_transition_log_matrix(
        labels,
        allowed_map=allowed,
        policy=TransitionPolicy(switch_penalty=0.0, self_transition_bias=0.0),
    )
    observation_log_probs = np.array(
        [
            [0.0, -5.0],
            [-5.0, 0.0],
            [-5.0, 0.0],
        ],
        dtype=np.float64,
    )

    result = decode_hmm_with_duration_priors(
        observation_log_probs=observation_log_probs,
        labels=labels,
        transition_log_matrix=transition,
    )
    forbidden = [pair for pair in zip(result.labels[:-1], result.labels[1:]) if pair == ("a", "b")]
    assert forbidden == []


def test_hmm_allows_observation_override_of_soft_duration_prior():
    labels = ["a", "b"]
    policy = DurationPriorPolicy(
        priors_by_label={
            "a": DurationPrior(min_minutes=5.0, target_minutes=10.0, max_minutes=30.0, penalty_weight=1.0),
            "b": DurationPrior(min_minutes=1.0, target_minutes=5.0, max_minutes=20.0, penalty_weight=1.0),
        },
        default_prior=DurationPrior(min_minutes=1.0, target_minutes=5.0, max_minutes=20.0, penalty_weight=1.0),
        transition=TransitionPolicy(switch_penalty=0.0, self_transition_bias=0.0, step_minutes=1.0),
    )
    transition = build_transition_log_matrix(
        labels,
        policy=policy.transition,
    )
    # Step 1 strongly supports state b; soft priors should not hard-block the switch.
    observation_log_probs = np.array(
        [
            [0.0, -8.0],
            [-8.0, 0.0],
            [-8.0, 0.0],
        ],
        dtype=np.float64,
    )
    result = decode_hmm_with_duration_priors(
        observation_log_probs=observation_log_probs,
        labels=labels,
        transition_log_matrix=transition,
        duration_policy=policy,
    )
    assert result.labels[1] == "b"
    assert result.labels[2] == "b"


def test_hmm_duration_prior_reduces_ping_pong():
    labels = ["a", "b"]
    observations = np.array(
        [
            [0.0, -0.1],
            [-0.1, 0.0],
            [0.0, -0.1],
            [-0.1, 0.0],
            [0.0, -0.1],
            [-0.1, 0.0],
        ],
        dtype=np.float64,
    )

    baseline_policy = DurationPriorPolicy(
        priors_by_label={},
        default_prior=DurationPrior(min_minutes=1.0, target_minutes=1.0, max_minutes=120.0, penalty_weight=0.0),
        transition=TransitionPolicy(switch_penalty=0.0, self_transition_bias=0.0, step_minutes=1.0),
    )
    smoother_policy = DurationPriorPolicy(
        priors_by_label={
            "a": DurationPrior(min_minutes=3.0, target_minutes=5.0, max_minutes=120.0, penalty_weight=1.5),
            "b": DurationPrior(min_minutes=3.0, target_minutes=5.0, max_minutes=120.0, penalty_weight=1.5),
        },
        default_prior=DurationPrior(min_minutes=3.0, target_minutes=5.0, max_minutes=120.0, penalty_weight=1.5),
        transition=TransitionPolicy(switch_penalty=0.0, self_transition_bias=0.0, step_minutes=1.0),
    )

    baseline = decode_hmm_with_duration_priors(
        observation_log_probs=observations,
        labels=labels,
        duration_policy=baseline_policy,
    )
    smoother = decode_hmm_with_duration_priors(
        observation_log_probs=observations,
        labels=labels,
        duration_policy=smoother_policy,
    )

    assert smoother.ping_pong_rate <= baseline.ping_pong_rate


def test_hmm_continuity_penalty_preserves_sleep_runs_on_weak_switch_evidence():
    labels = ["sleep", "awake"]
    observations = np.array(
        [
            [0.0, -3.0],
            [-0.30, 0.0],
            [0.0, -0.20],
            [0.0, -0.20],
        ],
        dtype=np.float64,
    )
    base_policy = DurationPriorPolicy(
        priors_by_label={},
        default_prior=DurationPrior(min_minutes=1.0, target_minutes=1.0, max_minutes=120.0, penalty_weight=0.0),
        transition=TransitionPolicy(switch_penalty=0.0, self_transition_bias=0.0, step_minutes=1.0),
    )
    continuity_policy = DurationPriorPolicy(
        priors_by_label={},
        default_prior=DurationPrior(min_minutes=1.0, target_minutes=1.0, max_minutes=120.0, penalty_weight=0.0),
        transition=TransitionPolicy(
            switch_penalty=0.0,
            self_transition_bias=0.0,
            step_minutes=1.0,
            continuity_labels=("sleep",),
            continuity_min_minutes=3.0,
            continuity_break_penalty=2.0,
            continuity_escape_margin=0.50,
        ),
    )

    baseline = decode_hmm_with_duration_priors(
        observation_log_probs=observations,
        labels=labels,
        duration_policy=base_policy,
    )
    with_continuity = decode_hmm_with_duration_priors(
        observation_log_probs=observations,
        labels=labels,
        duration_policy=continuity_policy,
    )

    assert baseline.labels[1] == "awake"
    assert with_continuity.labels[1] == "sleep"
