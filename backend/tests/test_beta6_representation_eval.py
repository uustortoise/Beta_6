import numpy as np
import pytest

from ml.beta6.representation_eval import evaluate_representation_quality


def test_representation_eval_beats_random_baseline_on_separable_embeddings():
    rng = np.random.default_rng(42)
    n = 80
    features = rng.normal(size=(n, 6)).astype(np.float32)
    labels = np.array(["rest"] * (n // 2) + ["walk"] * (n // 2), dtype=object)
    features[n // 2 :, 0] += 2.5
    resident_ids = np.array([f"R{i % 8}" for i in range(n)], dtype=object)

    result = evaluate_representation_quality(
        embeddings=features,
        labels=labels,
        resident_ids=resident_ids,
        seed=11,
    )
    assert result.linear_probe_accuracy >= 0.7
    assert result.improvement_margin > 0.0
    assert result.knn_purity >= 0.65


def test_representation_eval_requires_multiple_residents():
    features = np.ones((20, 4), dtype=np.float32)
    labels = np.array(["a"] * 10 + ["b"] * 10, dtype=object)
    resident_ids = np.array(["R0"] * 20, dtype=object)
    with pytest.raises(ValueError, match="at least 2 unique residents"):
        evaluate_representation_quality(
            embeddings=features,
            labels=labels,
            resident_ids=resident_ids,
            seed=1,
        )
