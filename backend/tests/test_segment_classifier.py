import numpy as np

from ml.segment_classifier import classify_occupied_segments


def test_classify_occupied_segments_picks_highest_mean_label():
    segments = [{"start_idx": 2, "end_idx": 6}]
    activity_probs = {
        "sleep": np.asarray([0.1, 0.1, 0.8, 0.8, 0.7, 0.7, 0.1], dtype=float),
        "bedroom_normal_use": np.asarray([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.1], dtype=float),
    }
    labeled = classify_occupied_segments(
        segments=segments,
        activity_probs=activity_probs,
        min_activity_prob=0.35,
    )
    assert len(labeled) == 1
    assert labeled[0]["label"] == "sleep"
    assert labeled[0]["classifier_mode"] == "heuristic"


def test_classify_occupied_segments_sets_unoccupied_below_threshold():
    segments = [{"start_idx": 1, "end_idx": 4}]
    activity_probs = {
        "livingroom_normal_use": np.asarray([0.1, 0.2, 0.2, 0.2, 0.1], dtype=float),
    }
    labeled = classify_occupied_segments(
        segments=segments,
        activity_probs=activity_probs,
        min_activity_prob=0.5,
    )
    assert labeled[0]["label"] == "unoccupied"


def test_classify_occupied_segments_can_use_learned_path_when_enabled():
    segments = [
        {"start_idx": 0, "end_idx": 5},
        {"start_idx": 5, "end_idx": 10},
        {"start_idx": 10, "end_idx": 15},
        {"start_idx": 15, "end_idx": 20},
    ]
    activity_probs = {
        "sleep": np.asarray([0.9] * 10 + [0.1] * 10, dtype=float),
        "bedroom_normal_use": np.asarray([0.1] * 10 + [0.9] * 10, dtype=float),
    }
    segment_features = [
        {"duration_windows": 5.0, "occ_mean": 0.8, "motion_mean": 0.2},
        {"duration_windows": 5.0, "occ_mean": 0.75, "motion_mean": 0.3},
        {"duration_windows": 5.0, "occ_mean": 0.85, "motion_mean": 0.8},
        {"duration_windows": 5.0, "occ_mean": 0.82, "motion_mean": 0.9},
    ]
    labeled = classify_occupied_segments(
        segments=segments,
        activity_probs=activity_probs,
        segment_features=segment_features,
        min_activity_prob=0.35,
        enable_learned_classifier=True,
        learned_classifier_min_segments=2,
        learned_classifier_confidence_floor=0.0,
        learned_classifier_min_windows=1,
        random_state=7,
    )
    assert len(labeled) == 4
    assert all(row["classifier_mode"] == "learned" for row in labeled)
    assert any(bool(row["classifier_selected"]) for row in labeled)
    assert {row["label"] for row in labeled} == {"sleep", "bedroom_normal_use"}


def test_classify_occupied_segments_falls_back_on_low_support_even_with_model():
    segments = [
        {"start_idx": 0, "end_idx": 3},
        {"start_idx": 3, "end_idx": 6},
        {"start_idx": 6, "end_idx": 9},
        {"start_idx": 9, "end_idx": 12},
    ]
    activity_probs = {
        "sleep": np.asarray([0.8] * 6 + [0.2] * 6, dtype=float),
        "bedroom_normal_use": np.asarray([0.2] * 6 + [0.8] * 6, dtype=float),
    }
    segment_features = [
        {"duration_windows": 3.0, "occ_mean": 0.7},
        {"duration_windows": 3.0, "occ_mean": 0.72},
        {"duration_windows": 3.0, "occ_mean": 0.74},
        {"duration_windows": 3.0, "occ_mean": 0.76},
    ]
    labeled = classify_occupied_segments(
        segments=segments,
        activity_probs=activity_probs,
        segment_features=segment_features,
        min_activity_prob=0.35,
        enable_learned_classifier=True,
        learned_classifier_min_segments=2,
        learned_classifier_confidence_floor=0.0,
        learned_classifier_min_windows=5,
        random_state=11,
    )
    assert len(labeled) == 4
    assert all(row["classifier_mode"] == "learned" for row in labeled)
    assert all(bool(row["fallback_reason"]) for row in labeled)
    assert all(not bool(row["classifier_selected"]) for row in labeled)
