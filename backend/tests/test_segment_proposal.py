import numpy as np

from ml.segment_proposal import propose_occupancy_segments


def test_propose_occupancy_segments_merges_small_gap():
    probs = np.asarray([0.1, 0.8, 0.8, 0.2, 0.8, 0.8, 0.1], dtype=float)
    segs = propose_occupancy_segments(
        probs,
        threshold=0.5,
        min_duration_windows=2,
        gap_merge_windows=1,
    )
    assert len(segs) == 1
    assert segs[0]["start_idx"] == 1
    assert segs[0]["end_idx"] == 6


def test_propose_occupancy_segments_drops_short_run():
    probs = np.asarray([0.1, 0.8, 0.1, 0.8, 0.8, 0.1], dtype=float)
    segs = propose_occupancy_segments(
        probs,
        threshold=0.5,
        min_duration_windows=2,
        gap_merge_windows=0,
    )
    assert len(segs) == 1
    assert segs[0]["start_idx"] == 3
    assert segs[0]["end_idx"] == 5
