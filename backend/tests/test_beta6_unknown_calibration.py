import numpy as np

from ml.beta6.calibration import evaluate_calibration
from ml.beta6.prediction import (
    UnknownPolicy,
    build_triage_candidates_from_inference,
    infer_with_unknown_path,
)


def test_infer_with_unknown_path_routes_abstain_states():
    probs = np.array(
        [
            [0.98, 0.02],  # confident
            [0.52, 0.48],  # low confidence
            [0.56, 0.44],  # high entropy but above confidence floor
            [0.90, 0.10],  # outside sensed space override
        ],
        dtype=np.float64,
    )
    outside_scores = np.array([0.0, 0.0, 0.0, 0.95], dtype=np.float64)
    policy = UnknownPolicy(
        min_confidence=0.55,
        max_entropy=0.68,
        outside_sensed_space_threshold=0.8,
        abstain_rate_min=0.2,
        abstain_rate_max=0.9,
    )

    result = infer_with_unknown_path(
        probabilities=probs,
        labels=["active_use", "unoccupied"],
        policy=policy,
        outside_sensed_space_scores=outside_scores,
    )
    assert result["labels"][0] == "active_use"
    assert result["labels"][1] == policy.abstain_label
    assert result["labels"][2] == policy.abstain_label
    assert result["labels"][3] == policy.abstain_label
    assert result["uncertainty_states"][1] == policy.low_confidence_state
    assert result["uncertainty_states"][2] == policy.unknown_state
    assert result["uncertainty_states"][3] == policy.outside_sensed_space_state
    assert result["abstain_rate"] == 0.75


def test_triage_candidates_include_only_uncertain_rows():
    inference = {
        "labels": ["active_use", "abstain", "abstain"],
        "confidence": [0.9, 0.4, 0.3],
        "uncertainty_states": [None, "low_confidence", "unknown"],
    }
    rows = build_triage_candidates_from_inference(
        inference=inference,
        room="Bedroom",
        activity_hint="sleep",
    )
    assert len(rows) == 2
    assert rows[0]["room"] == "bedroom"
    assert rows[0]["baseline_label"] == "sleep"


def test_evaluate_calibration_reports_expected_fields():
    probs = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.4, 0.6],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )
    report = evaluate_calibration(probabilities=probs, true_indices=[0, 0, 1, 1])
    payload = report.to_dict()
    assert 0.0 <= payload["accuracy"] <= 1.0
    assert payload["brier_score"] >= 0.0
    assert payload["expected_calibration_error"] >= 0.0
