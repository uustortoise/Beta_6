import numpy as np

from ml.beta6.serving.activity_confidence import (
    choose_activity_confidence_threshold,
    fit_activity_confidence_calibrator,
    score_activity_confidence,
)


def test_activity_confidence_scoring_prefers_reliable_rows():
    probs = np.array(
        [
            [0.95, 0.05],
            [0.55, 0.45],
            [0.90, 0.10],
            [0.60, 0.40],
            [0.10, 0.90],
            [0.45, 0.55],
            [0.05, 0.95],
            [0.40, 0.60],
        ],
        dtype=np.float64,
    )
    true_indices = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int64)

    artifact = fit_activity_confidence_calibrator(
        probabilities=probs,
        true_indices=true_indices,
        labels=["sleep", "unoccupied"],
        min_samples=6,
        min_positive=2,
        min_negative=2,
    )
    assert artifact["schema_version"] == "activity_acceptance_score_v1"

    scored = score_activity_confidence(
        probabilities=probs,
        labels=["sleep", "unoccupied"],
        artifact=artifact,
    )
    assert scored["confidence_source"] == "activity_acceptance_score_v1"
    assert float(scored["scores"][0]) > float(scored["scores"][1])
    assert float(scored["scores"][4]) > float(scored["scores"][5])


def test_activity_confidence_scoring_falls_back_to_raw_top1_without_artifact():
    probs = np.array([[0.8, 0.2], [0.52, 0.48]], dtype=np.float64)
    scored = score_activity_confidence(
        probabilities=probs,
        labels=["active_use", "unoccupied"],
        artifact=None,
    )
    assert scored["confidence_source"] == "raw_top1_confidence"
    assert np.allclose(scored["scores"], np.array([0.8, 0.52], dtype=np.float64))


def test_choose_activity_confidence_threshold_flags_dense_band_cliff():
    scores = np.array(
        ([0.56] * 18) + ([0.57] * 18) + ([0.82] * 4) + ([0.22] * 6),
        dtype=np.float64,
    )
    outcomes = np.array(
        ([1] * 28) + ([0] * 8) + ([1] * 4) + ([0] * 6),
        dtype=np.int64,
    )

    result = choose_activity_confidence_threshold(
        scores=scores,
        outcomes=outcomes,
        target_precision=0.70,
        recall_floor=0.60,
        threshold_floor=0.35,
        threshold_cap=0.80,
        stability_window=0.015,
        max_near_threshold_share=0.20,
    )

    assert "stability" in str(result["status"])
    assert float(result["threshold"]) <= 0.55
    assert float(result["near_threshold_share"]) >= 0.20


def test_choose_activity_confidence_threshold_escapes_dense_band_when_no_stable_candidate_exists():
    scores = np.array(
        ([0.2345] * 18) + ([0.2350] * 18) + ([0.2360] * 18) + ([0.2600] * 4),
        dtype=np.float64,
    )
    outcomes = np.array(
        ([1] * 20) + ([0] * 30) + ([1] * 8),
        dtype=np.int64,
    )

    result = choose_activity_confidence_threshold(
        scores=scores,
        outcomes=outcomes,
        target_precision=0.70,
        recall_floor=0.60,
        threshold_floor=0.0,
        threshold_cap=1.0,
        stability_window=0.03,
        max_near_threshold_share=0.20,
    )

    assert "stability" in str(result["status"])
    assert float(result["near_threshold_share"]) > 0.20
    assert float(result["threshold"]) <= 0.206
    assert float(result["selected_threshold_before_fallback"]) >= 0.2345
