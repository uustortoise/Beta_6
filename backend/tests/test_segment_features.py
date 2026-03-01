import numpy as np
import pandas as pd

from ml.segment_features import build_segment_features


def test_build_segment_features_computes_duration_and_occupancy_stats():
    ts = pd.date_range("2025-12-06 22:00:00", periods=10, freq="10s")
    occ = np.asarray([0.1, 0.2, 0.8, 0.9, 0.85, 0.8, 0.2, 0.1, 0.1, 0.1], dtype=float)
    segs = [{"start_idx": 2, "end_idx": 6}]
    rows = build_segment_features(
        segments=segs,
        timestamps=ts,
        occupancy_probs=occ,
    )
    assert len(rows) == 1
    row = rows[0]
    assert int(row["duration_windows"]) == 4
    assert row["duration_minutes"] > 0.0
    assert 0.8 <= row["occ_mean"] <= 0.9


def test_build_segment_features_enriches_sensor_and_activity_stats_with_finite_values():
    ts = pd.date_range("2025-12-06 22:00:00", periods=24, freq="10s")
    occ = np.linspace(0.2, 0.9, num=24, dtype=float)
    segs = [{"start_idx": 4, "end_idx": 18}]
    sensor_series = {
        "motion": np.linspace(0.0, 1.0, num=24, dtype=float),
        "light": np.linspace(10.0, 200.0, num=24, dtype=float),
        "co2": np.linspace(500.0, 620.0, num=24, dtype=float),
        "temperature": np.linspace(23.0, 24.5, num=24, dtype=float),
        "humidity": np.linspace(45.0, 52.0, num=24, dtype=float),
    }
    activity_probs = {
        "sleep": np.linspace(0.7, 0.2, num=24, dtype=float),
        "bedroom_normal_use": np.linspace(0.2, 0.8, num=24, dtype=float),
    }
    rows = build_segment_features(
        segments=segs,
        timestamps=ts,
        occupancy_probs=occ,
        sensor_series=sensor_series,
        activity_probs=activity_probs,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["feature_version"] == 2.0
    assert row["motion_active_ratio"] >= 0.0
    assert row["light_low_ratio"] >= 0.0
    assert row["co2_rise_count"] >= 0.0
    assert row["act_sleep_mean"] > 0.0
    assert row["act_bedroom_normal_use_mean"] > 0.0
    numeric_keys = [k for k in row.keys() if k not in {"start_idx", "end_idx"}]
    assert len(numeric_keys) >= 25
    assert np.isfinite(np.asarray([float(row[k]) for k in numeric_keys], dtype=float)).all()
