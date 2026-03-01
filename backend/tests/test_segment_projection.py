import numpy as np

from ml.segment_projection import project_segments_to_window_labels


def test_project_segments_to_window_labels_projects_half_open_ranges():
    labeled = [
        {"start_idx": 1, "end_idx": 3, "label": "sleep"},
        {"start_idx": 4, "end_idx": 6, "label": "bedroom_normal_use"},
    ]
    out = project_segments_to_window_labels(
        n_windows=7,
        labeled_segments=labeled,
        default_label="unoccupied",
    )
    assert out.tolist() == [
        "unoccupied",
        "sleep",
        "sleep",
        "unoccupied",
        "bedroom_normal_use",
        "bedroom_normal_use",
        "unoccupied",
    ]
    assert np.asarray(out, dtype=object).shape[0] == 7
