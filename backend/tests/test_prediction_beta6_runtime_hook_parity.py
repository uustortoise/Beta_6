import numpy as np

from ml.beta6.serving.runtime_hooks import (
    apply_beta6_hmm_runtime,
    apply_beta6_unknown_abstain_runtime,
)
from ml.legacy.prediction import PredictionPipeline


def _config_path(name: str) -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parent.parent / "config" / name)


def test_unknown_runtime_hook_legacy_delegator_matches_beta6_serving(monkeypatch):
    monkeypatch.setenv("ENABLE_BETA6_UNKNOWN_ABSTAIN_RUNTIME", "true")
    monkeypatch.setenv("BETA6_PHASE4_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("BETA6_PHASE4_RUNTIME_ROOMS", "bedroom")
    monkeypatch.setenv("BETA6_UNKNOWN_POLICY_PATH", _config_path("beta6_unknown_policy.yaml"))

    y_pred_probs = np.asarray(
        [
            [0.85, 0.10, 0.05],
            [0.34, 0.33, 0.33],
            [0.40, 0.35, 0.25],
        ],
        dtype=float,
    )
    label_classes = ["sleep", "out", "unoccupied"]
    final_labels = np.asarray(["sleep", "out", "unoccupied"], dtype=object)
    low_conf_flags = [False, False, False]
    low_conf_hints = [None, None, None]

    legacy = PredictionPipeline.__new__(PredictionPipeline)
    legacy_out = legacy._apply_beta6_unknown_abstain_runtime(
        room_name="bedroom",
        y_pred_probs=y_pred_probs,
        label_classes=label_classes,
        final_labels=final_labels.copy(),
        low_conf_flags=list(low_conf_flags),
        low_conf_hints=list(low_conf_hints),
    )
    new_out = apply_beta6_unknown_abstain_runtime(
        room_name="bedroom",
        y_pred_probs=y_pred_probs,
        label_classes=label_classes,
        final_labels=final_labels.copy(),
        low_conf_flags=list(low_conf_flags),
        low_conf_hints=list(low_conf_hints),
    )

    assert np.array_equal(legacy_out[0], new_out[0])
    assert legacy_out[1] == new_out[1]
    assert legacy_out[2] == new_out[2]
    assert legacy_out[3] == new_out[3]
    assert legacy_out[4] == new_out[4]


def test_hmm_runtime_hook_legacy_delegator_matches_beta6_serving(monkeypatch):
    monkeypatch.setenv("ENABLE_BETA6_HMM_RUNTIME", "true")
    monkeypatch.setenv("BETA6_PHASE4_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("BETA6_PHASE4_RUNTIME_ROOMS", "bedroom")
    monkeypatch.setenv("BETA6_HMM_DURATION_POLICY_PATH", _config_path("beta6_duration_prior_policy.yaml"))

    y_pred_probs = np.asarray(
        [
            [0.60, 0.20, 0.20],
            [0.55, 0.25, 0.20],
            [0.15, 0.70, 0.15],
            [0.10, 0.80, 0.10],
        ],
        dtype=float,
    )
    label_classes = ["sleep", "out", "unoccupied"]
    final_labels = np.asarray(["sleep", "sleep", "out", "out"], dtype=object)
    low_conf_flags = [False, True, False, False]

    legacy = PredictionPipeline.__new__(PredictionPipeline)
    legacy_out = legacy._apply_beta6_hmm_runtime(
        room_name="bedroom",
        y_pred_probs=y_pred_probs,
        label_classes=label_classes,
        final_labels=final_labels.copy(),
        low_conf_flags=list(low_conf_flags),
    )
    new_out = apply_beta6_hmm_runtime(
        room_name="bedroom",
        y_pred_probs=y_pred_probs,
        label_classes=label_classes,
        final_labels=final_labels.copy(),
        low_conf_flags=list(low_conf_flags),
    )

    assert np.array_equal(legacy_out, new_out)


def test_sequence_runtime_hook_can_select_crf_mode(monkeypatch):
    monkeypatch.setenv("ENABLE_BETA6_HMM_RUNTIME", "true")
    monkeypatch.setenv("BETA6_PHASE4_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("BETA6_PHASE4_RUNTIME_ROOMS", "bedroom")
    monkeypatch.setenv("BETA6_SEQUENCE_RUNTIME_MODE", "crf")
    monkeypatch.setenv("BETA6_HMM_DURATION_POLICY_PATH", _config_path("beta6_duration_prior_policy.yaml"))

    called = {"crf": False}

    class _FakeCRFResult:
        def __init__(self):
            self.labels = ["sleep", "sleep", "out", "out"]
            self.state_indices = [0, 0, 1, 1]
            self.score = 0.0
            self.ping_pong_rate = 0.0
            self.transition_log_matrix = np.zeros((3, 3), dtype=float)

    def _fake_decode_crf(**kwargs):
        called["crf"] = True
        return _FakeCRFResult()

    monkeypatch.setattr(
        "ml.beta6.serving.runtime_hooks.decode_crf_with_duration_priors",
        _fake_decode_crf,
    )

    y_pred_probs = np.asarray(
        [
            [0.60, 0.20, 0.20],
            [0.55, 0.25, 0.20],
            [0.15, 0.70, 0.15],
            [0.10, 0.80, 0.10],
        ],
        dtype=float,
    )
    label_classes = ["sleep", "out", "unoccupied"]
    final_labels = np.asarray(["sleep", "sleep", "out", "out"], dtype=object)
    low_conf_flags = [False, False, False, False]

    out = apply_beta6_hmm_runtime(
        room_name="bedroom",
        y_pred_probs=y_pred_probs,
        label_classes=label_classes,
        final_labels=final_labels.copy(),
        low_conf_flags=list(low_conf_flags),
    )

    assert called["crf"] is True
    assert np.array_equal(out, np.asarray(["sleep", "sleep", "out", "out"], dtype=object))
