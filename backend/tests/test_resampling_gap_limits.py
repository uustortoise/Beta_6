from datetime import datetime

import pandas as pd

from elderlycare_v1_16.platform import _resolve_ffill_limit
from elderlycare_v1_16.preprocessing.resampling import resample_to_fixed_interval


def test_resample_bounded_ffill_does_not_bridge_long_gap():
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 1, 0, 0, 0),
                datetime(2026, 1, 1, 0, 0, 10),
                datetime(2026, 1, 1, 0, 0, 30),
                datetime(2026, 1, 1, 0, 2, 0),
            ],
            "sensor": [1.0, 2.0, 3.0, 9.0],
        }
    )

    out = resample_to_fixed_interval(
        df,
        interval="10s",
        timestamp_col="timestamp",
        fill_method="ffill",
        max_ffill_gap_seconds=60.0,
    )

    ts_to_sensor = out.set_index("timestamp")["sensor"]
    assert ts_to_sensor.loc[datetime(2026, 1, 1, 0, 1, 30)] == 3.0
    assert pd.isna(ts_to_sensor.loc[datetime(2026, 1, 1, 0, 1, 40)])
    assert pd.isna(ts_to_sensor.loc[datetime(2026, 1, 1, 0, 1, 50)])
    assert ts_to_sensor.loc[datetime(2026, 1, 1, 0, 2, 0)] == 9.0


def test_resample_unbounded_fill_is_still_supported():
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 1, 0, 0, 0),
                datetime(2026, 1, 1, 0, 0, 10),
                datetime(2026, 1, 1, 0, 0, 30),
                datetime(2026, 1, 1, 0, 2, 0),
            ],
            "sensor": [1.0, 2.0, 3.0, 9.0],
        }
    )

    out = resample_to_fixed_interval(
        df,
        interval="10s",
        timestamp_col="timestamp",
        fill_method="ffill",
        max_ffill_gap_seconds=None,
    )

    ts_to_sensor = out.set_index("timestamp")["sensor"]
    assert ts_to_sensor.loc[datetime(2026, 1, 1, 0, 1, 40)] == 3.0
    assert ts_to_sensor.loc[datetime(2026, 1, 1, 0, 1, 50)] == 3.0


def test_resolve_ffill_limit_from_interval():
    assert _resolve_ffill_limit("10s", 60.0) == 6
    assert _resolve_ffill_limit("1min", 60.0) == 1
    assert _resolve_ffill_limit("10s", None) is None


def test_resample_normalizes_jittered_timestamps_to_interval_grid():
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 1, 0, 0, 7),
                datetime(2026, 1, 1, 0, 0, 16),
                datetime(2026, 1, 1, 0, 0, 24),
                datetime(2026, 1, 1, 0, 0, 33),
            ],
            "sensor": [1.0, 2.0, 3.0, 4.0],
        }
    )

    out = resample_to_fixed_interval(
        df,
        interval="10s",
        timestamp_col="timestamp",
        fill_method="ffill",
        max_ffill_gap_seconds=60.0,
        normalize_timestamps_to_interval=True,
    )

    ts_to_sensor = out.set_index("timestamp")["sensor"]
    assert list(ts_to_sensor.index) == [
        datetime(2026, 1, 1, 0, 0, 10),
        datetime(2026, 1, 1, 0, 0, 20),
        datetime(2026, 1, 1, 0, 0, 30),
    ]
    assert ts_to_sensor.loc[datetime(2026, 1, 1, 0, 0, 10)] == 1.0
    assert ts_to_sensor.loc[datetime(2026, 1, 1, 0, 0, 20)] == 2.5
    assert ts_to_sensor.loc[datetime(2026, 1, 1, 0, 0, 30)] == 4.0


def test_resample_without_normalization_keeps_start_anchored_grid():
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 1, 0, 0, 7),
                datetime(2026, 1, 1, 0, 0, 16),
                datetime(2026, 1, 1, 0, 0, 24),
                datetime(2026, 1, 1, 0, 0, 33),
            ],
            "sensor": [1.0, 2.0, 3.0, 4.0],
        }
    )

    out = resample_to_fixed_interval(
        df,
        interval="10s",
        timestamp_col="timestamp",
        fill_method="ffill",
        max_ffill_gap_seconds=60.0,
        normalize_timestamps_to_interval=False,
    )

    ts_to_sensor = out.set_index("timestamp")["sensor"]
    assert list(ts_to_sensor.index) == [
        datetime(2026, 1, 1, 0, 0, 7),
        datetime(2026, 1, 1, 0, 0, 17),
        datetime(2026, 1, 1, 0, 0, 27),
    ]
