import numpy as np
import pytest

from ml.event_models import EventFirstConfig, EventFirstTwoStageModel


def test_event_first_two_stage_fit_predict():
    # Two features, simple separable pattern
    x = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.1],
            [0.9, 0.8],
            [0.8, 0.9],
            [0.75, 0.7],
            [0.7, 0.75],
        ]
    )
    y = np.array(
        [
            "unoccupied",
            "unoccupied",
            "sleep",
            "sleep",
            "bedroom_normal_use",
            "bedroom_normal_use",
        ]
    )

    model = EventFirstTwoStageModel(EventFirstConfig(random_state=7, n_estimators_stage_a=50, n_estimators_stage_b=50))
    model.fit(x, y)

    pred = model.predict(x, occupancy_threshold=0.4)
    assert pred.shape[0] == x.shape[0]

    # Ensure model predicts at least one occupied class and one unoccupied class.
    assert any(p == "unoccupied" for p in pred)
    assert any(p != "unoccupied" for p in pred)


def test_event_first_tune_operating_points_updates_thresholds():
    x = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.1],
            [0.3, 0.2],
            [0.9, 0.8],
            [0.8, 0.9],
            [0.75, 0.7],
            [0.7, 0.75],
            [0.85, 0.65],
        ]
    )
    y = np.array(
        [
            "unoccupied",
            "unoccupied",
            "unoccupied",
            "sleep",
            "sleep",
            "bedroom_normal_use",
            "bedroom_normal_use",
            "bedroom_normal_use",
        ]
    )

    model = EventFirstTwoStageModel(EventFirstConfig(random_state=7, n_estimators_stage_a=40, n_estimators_stage_b=40))
    model.fit(x, y)
    tuning = model.tune_operating_points(
        x,
        y,
        calibration_method="isotonic",
        min_samples=4,
        min_label_support=1,
    )
    assert tuning["used"] is True
    assert 0.0 <= float(tuning["occupancy_threshold"]) <= 1.0
    assert "activity_thresholds" in tuning
    assert model.get_operating_points()["calibration_method"] in {"isotonic", "none"}

    pred = model.predict(x)
    assert pred.shape[0] == x.shape[0]


def test_event_first_fit_accepts_stage_a_sample_weight():
    x = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.1],
            [0.3, 0.2],
            [0.9, 0.8],
            [0.8, 0.9],
            [0.75, 0.7],
        ]
    )
    y = np.array(
        [
            "unoccupied",
            "unoccupied",
            "unoccupied",
            "sleep",
            "sleep",
            "bedroom_normal_use",
        ]
    )
    w = np.array([0.7, 0.7, 1.0, 2.0, 2.0, 2.0], dtype=float)
    model = EventFirstTwoStageModel(EventFirstConfig(random_state=7, n_estimators_stage_a=30, n_estimators_stage_b=30))
    model.fit(x, y, stage_a_sample_weight=w)
    pred = model.predict(x, occupancy_threshold=0.4)
    assert pred.shape[0] == x.shape[0]


class _DummyEstimator:
    def __init__(self, classes_, probs):
        self.classes_ = np.array(classes_)
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, x):
        n = int(len(x))
        if n <= len(self._probs):
            return self._probs[:n]
        tail = np.repeat(self._probs[-1:], repeats=(n - len(self._probs)), axis=0)
        return np.vstack([self._probs, tail])


def test_event_first_predict_uses_unknown_for_low_confidence():
    cfg = EventFirstConfig(
        unknown_label="unknown",
        default_activity_threshold=0.80,
        use_unknown_for_low_confidence=True,
    )
    model = EventFirstTwoStageModel(cfg)
    model.stage_a = _DummyEstimator(["occupied", "unoccupied"], [[0.9, 0.1], [0.88, 0.12]])
    model.stage_b = _DummyEstimator(["sleep", "bedroom_normal_use"], [[0.60, 0.40], [0.55, 0.45]])

    pred = model.predict(np.zeros((2, 3), dtype=float), occupancy_threshold=0.5)
    assert list(pred) == ["unknown", "unknown"]


def test_event_first_predict_label_threshold_can_bias_argmax():
    cfg = EventFirstConfig(use_unknown_for_low_confidence=False)
    model = EventFirstTwoStageModel(cfg)
    model.stage_a = _DummyEstimator(["occupied", "unoccupied"], [[0.95, 0.05]])
    model.stage_b = _DummyEstimator(["sleep", "bedroom_normal_use"], [[0.45, 0.55]])

    base_pred = model.predict(np.zeros((1, 3), dtype=float), occupancy_threshold=0.5)
    assert list(base_pred) == ["bedroom_normal_use"]

    boosted_pred = model.predict(
        np.zeros((1, 3), dtype=float),
        occupancy_threshold=0.5,
        label_thresholds={"sleep": 0.35, "bedroom_normal_use": 0.55},
    )
    assert list(boosted_pred) == ["sleep"]


def test_temporal_stage_a_matrix_is_causal():
    x = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=float,
    )
    out = EventFirstTwoStageModel._build_temporal_stage_a_matrix(
        x,
        lag_windows=2,
        include_delta=True,
    )
    assert out.shape == (3, 8)
    np.testing.assert_allclose(out[0, :6], np.array([1.0, 10.0, 1.0, 10.0, 1.0, 10.0], dtype=float))
    np.testing.assert_allclose(out[1, :6], np.array([2.0, 20.0, 1.0, 10.0, 1.0, 10.0], dtype=float))
    np.testing.assert_allclose(out[2, :6], np.array([3.0, 30.0, 2.0, 20.0, 1.0, 10.0], dtype=float))
    np.testing.assert_allclose(out[0, 6:], np.array([0.0, 0.0], dtype=float))
    np.testing.assert_allclose(out[1, 6:], np.array([1.0, 10.0], dtype=float))
    np.testing.assert_allclose(out[2, 6:], np.array([1.0, 10.0], dtype=float))


def test_event_first_temporal_stage_a_fit_predict():
    x = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.1],
            [0.9, 0.8],
            [0.8, 0.9],
            [0.75, 0.7],
            [0.7, 0.75],
        ],
        dtype=float,
    )
    y = np.array(
        [
            "unoccupied",
            "unoccupied",
            "sleep",
            "sleep",
            "bedroom_normal_use",
            "bedroom_normal_use",
        ],
        dtype=object,
    )

    model = EventFirstTwoStageModel(
        EventFirstConfig(
            random_state=7,
            n_estimators_stage_a=50,
            n_estimators_stage_b=50,
            stage_a_model_type="temporal_rf",
            stage_a_temporal_lag_windows=3,
        )
    )
    model.fit(x, y)
    pred = model.predict(x, occupancy_threshold=0.4)
    assert pred.shape[0] == x.shape[0]
    ops = model.get_operating_points()
    assert ops["stage_a_model_type"] == "temporal_rf"
    assert int(ops["stage_a_temporal_lag_windows"]) == 3


def test_event_first_hgb_stage_a_fit_predict():
    x = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.1],
            [0.9, 0.8],
            [0.8, 0.9],
            [0.75, 0.7],
            [0.7, 0.75],
        ],
        dtype=float,
    )
    y = np.array(
        [
            "unoccupied",
            "unoccupied",
            "sleep",
            "sleep",
            "bedroom_normal_use",
            "bedroom_normal_use",
        ],
        dtype=object,
    )

    model = EventFirstTwoStageModel(
        EventFirstConfig(
            random_state=7,
            n_estimators_stage_a=60,
            n_estimators_stage_b=50,
            stage_a_model_type="hgb",
        )
    )
    model.fit(x, y)
    pred = model.predict(x, occupancy_threshold=0.4)
    assert pred.shape[0] == x.shape[0]
    ops = model.get_operating_points()
    assert ops["stage_a_model_type"] == "hgb"


def test_event_first_sequence_stage_a_enables_markov_filter():
    x = np.array(
        [
            [0.10, 0.10],
            [0.12, 0.10],
            [0.90, 0.85],
            [0.88, 0.86],
            [0.11, 0.10],
            [0.90, 0.90],
            [0.92, 0.92],
            [0.10, 0.12],
        ],
        dtype=float,
    )
    y = np.array(
        [
            "unoccupied",
            "unoccupied",
            "sleep",
            "sleep",
            "unoccupied",
            "bedroom_normal_use",
            "bedroom_normal_use",
            "unoccupied",
        ],
        dtype=object,
    )

    model = EventFirstTwoStageModel(
        EventFirstConfig(
            random_state=7,
            n_estimators_stage_a=60,
            n_estimators_stage_b=40,
            stage_a_model_type="sequence_rf",
            stage_a_temporal_lag_windows=3,
        )
    )
    model.fit(x, y)
    p_occ = model.predict_occupancy_proba(x)
    assert p_occ.shape[0] == x.shape[0]
    assert np.all((p_occ >= 0.0) & (p_occ <= 1.0))

    ops = model.get_operating_points()
    assert ops["stage_a_model_type"] == "sequence_rf"
    assert bool(ops["stage_a_sequence_filter_enabled"]) is True
    assert len(ops["stage_a_sequence_transition_matrix"]) == 2
    assert len(ops["stage_a_sequence_initial_state"]) == 2
    row0 = np.asarray(ops["stage_a_sequence_transition_matrix"][0], dtype=float)
    row1 = np.asarray(ops["stage_a_sequence_transition_matrix"][1], dtype=float)
    assert np.isclose(np.sum(row0), 1.0)
    assert np.isclose(np.sum(row1), 1.0)


def test_event_first_sequence_transformer_stage_a_fit_predict():
    pytest.importorskip("tensorflow")
    x = np.array(
        [
            [0.10, 0.10],
            [0.12, 0.10],
            [0.90, 0.85],
            [0.88, 0.86],
            [0.11, 0.10],
            [0.90, 0.90],
            [0.92, 0.92],
            [0.10, 0.12],
        ],
        dtype=float,
    )
    y = np.array(
        [
            "unoccupied",
            "unoccupied",
            "sleep",
            "sleep",
            "unoccupied",
            "bedroom_normal_use",
            "bedroom_normal_use",
            "unoccupied",
        ],
        dtype=object,
    )
    model = EventFirstTwoStageModel(
        EventFirstConfig(
            random_state=7,
            n_estimators_stage_b=40,
            stage_a_model_type="sequence_transformer",
            stage_a_temporal_lag_windows=3,
            stage_a_transformer_epochs=2,
            stage_a_transformer_batch_size=8,
            stage_a_transformer_hidden_dim=16,
            stage_a_transformer_num_heads=2,
            stage_a_transformer_dropout=0.0,
        )
    )
    model.fit(x, y)
    p_occ = model.predict_occupancy_proba(x)
    assert p_occ.shape[0] == x.shape[0]
    assert np.all((p_occ >= 0.0) & (p_occ <= 1.0))
    pred = model.predict(x, occupancy_threshold=0.4)
    assert pred.shape[0] == x.shape[0]
    ops = model.get_operating_points()
    assert ops["stage_a_model_type"] == "sequence_transformer"
    assert bool(ops["stage_a_sequence_filter_enabled"]) is True


def test_event_first_sequence_transformer_can_disable_sequence_filter():
    pytest.importorskip("tensorflow")
    x = np.array(
        [
            [0.10, 0.10],
            [0.12, 0.10],
            [0.90, 0.85],
            [0.88, 0.86],
            [0.11, 0.10],
            [0.90, 0.90],
            [0.92, 0.92],
            [0.10, 0.12],
        ],
        dtype=float,
    )
    y = np.array(
        [
            "unoccupied",
            "unoccupied",
            "sleep",
            "sleep",
            "unoccupied",
            "bedroom_normal_use",
            "bedroom_normal_use",
            "unoccupied",
        ],
        dtype=object,
    )
    model = EventFirstTwoStageModel(
        EventFirstConfig(
            random_state=9,
            n_estimators_stage_b=40,
            stage_a_model_type="sequence_transformer",
            stage_a_temporal_lag_windows=3,
            stage_a_transformer_epochs=2,
            stage_a_transformer_batch_size=8,
            stage_a_transformer_hidden_dim=16,
            stage_a_transformer_num_heads=2,
            stage_a_transformer_dropout=0.0,
            stage_a_transformer_use_sequence_filter=False,
        )
    )
    model.fit(x, y)
    ops = model.get_operating_points()
    assert ops["stage_a_model_type"] == "sequence_transformer"
    assert bool(ops["stage_a_sequence_filter_enabled"]) is False


def test_event_first_stage_a_grouped_fit_predict_maps_probs_back_to_window_grid():
    x = np.array(
        [
            [0.10, 0.10],
            [0.12, 0.09],
            [0.88, 0.85],
            [0.90, 0.86],
            [0.11, 0.12],
            [0.10, 0.11],
        ],
        dtype=float,
    )
    y = np.array(
        [
            "unoccupied",
            "unoccupied",
            "sleep",
            "sleep",
            "unoccupied",
            "unoccupied",
        ],
        dtype=object,
    )
    group_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    model = EventFirstTwoStageModel(
        EventFirstConfig(
            random_state=17,
            n_estimators_stage_a=60,
            n_estimators_stage_b=50,
            stage_a_model_type="hgb",
        )
    )
    model.fit(x, y, stage_a_group_ids=group_ids)

    p_occ = model.predict_occupancy_proba(x, stage_a_group_ids=group_ids)
    assert p_occ.shape[0] == x.shape[0]
    assert np.all((p_occ >= 0.0) & (p_occ <= 1.0))
    assert float(abs(p_occ[0] - p_occ[1])) < 1e-9
    assert float(abs(p_occ[2] - p_occ[3])) < 1e-9
    assert float(abs(p_occ[4] - p_occ[5])) < 1e-9

    pred = model.predict(x, occupancy_threshold=0.45, stage_a_group_ids=group_ids)
    assert pred.shape[0] == x.shape[0]


def test_event_first_stage_a_grouped_fit_rejects_bad_group_id_length():
    x = np.array([[0.1, 0.1], [0.9, 0.8]], dtype=float)
    y = np.array(["unoccupied", "sleep"], dtype=object)
    model = EventFirstTwoStageModel(EventFirstConfig(random_state=3, n_estimators_stage_a=20, n_estimators_stage_b=20))
    with pytest.raises(ValueError, match="stage_a_group_ids"):
        model.fit(x, y, stage_a_group_ids=np.array([0], dtype=np.int64))
