import numpy as np
import pytest

from ml.transformer_head_ab import derive_dual_head_probabilities


def test_derive_dual_head_probabilities_basic_conversion():
    probs = np.array(
        [
            [0.20, 0.70, 0.10],
            [0.60, 0.20, 0.20],
        ],
        dtype=np.float32,
    )
    classes = ["sleep", "unoccupied", "bedroom_normal_use"]

    dual = derive_dual_head_probabilities(probs, classes, occupancy_label="unoccupied")

    np.testing.assert_allclose(dual.occupancy_prob, np.array([0.30, 0.80], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(
        dual.activity_probs["sleep"],
        np.array([0.20 / 0.30, 0.60 / 0.80], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        dual.activity_probs["bedroom_normal_use"],
        np.array([0.10 / 0.30, 0.20 / 0.80], dtype=np.float32),
        atol=1e-6,
    )


def test_derive_dual_head_probabilities_zeroes_activity_when_unoccupied_is_certain():
    probs = np.array([[0.0, 1.0]], dtype=np.float32)
    classes = ["sleep", "unoccupied"]

    dual = derive_dual_head_probabilities(probs, classes, occupancy_label="unoccupied")

    np.testing.assert_allclose(dual.occupancy_prob, np.array([0.0], dtype=np.float32), atol=1e-8)
    np.testing.assert_allclose(dual.activity_probs["sleep"], np.array([0.0], dtype=np.float32), atol=1e-8)


def test_derive_dual_head_probabilities_requires_occupancy_label():
    probs = np.array([[0.5, 0.5]], dtype=np.float32)
    classes = ["sleep", "awake"]

    with pytest.raises(ValueError, match="occupancy label"):
        derive_dual_head_probabilities(probs, classes, occupancy_label="unoccupied")

