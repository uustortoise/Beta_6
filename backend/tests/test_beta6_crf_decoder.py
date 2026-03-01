import numpy as np

from ml.beta6.sequence.crf_decoder import (
    decode_crf_with_duration_priors,
    fit_transition_log_matrix_from_sequences,
)


def test_fit_transition_log_matrix_respects_disallowed_pairs():
    labels = ["sleep", "out"]
    matrix = fit_transition_log_matrix_from_sequences(
        label_sequences=[["sleep", "sleep", "out", "out"]],
        labels=labels,
        disallowed_pairs=[("sleep", "out")],
    )
    # Disallowed transition should receive an effectively impossible score.
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] < -1e5


def test_crf_decode_with_fit_reduces_ping_pong_on_ambiguous_trace():
    labels = ["sleep", "out"]
    probs = np.asarray(
        [
            [0.51, 0.49],
            [0.49, 0.51],
            [0.52, 0.48],
            [0.48, 0.52],
            [0.51, 0.49],
            [0.49, 0.51],
        ],
        dtype=np.float64,
    )
    log_probs = np.log(np.clip(probs, 1e-9, 1.0))
    baseline = decode_crf_with_duration_priors(
        observation_log_probs=log_probs,
        labels=labels,
    )
    fitted = decode_crf_with_duration_priors(
        observation_log_probs=log_probs,
        labels=labels,
        label_sequences_for_fit=[["sleep", "sleep", "sleep", "sleep", "sleep", "sleep"]],
    )
    assert len(fitted.labels) == probs.shape[0]
    assert fitted.ping_pong_rate <= baseline.ping_pong_rate
