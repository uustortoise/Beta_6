import numpy as np
import pandas as pd

from ml.sequence_alignment import safe_create_sequences


class _DummyPlatform:
    data_interval = "10s"

    def create_sequences(self, sensor_data, seq_length):
        raise AssertionError("legacy create_sequences should not be used in strict mode")


def test_safe_create_sequences_skips_windows_crossing_large_timestamp_gaps():
    platform = _DummyPlatform()
    sensor_data = np.arange(16, dtype=np.float32).reshape(8, 2)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    timestamps = pd.to_datetime(
        [
            "2025-12-04 08:00:00",
            "2025-12-04 08:00:10",
            "2025-12-04 08:00:20",
            "2025-12-04 08:00:30",
            "2026-03-09 08:00:00",
            "2026-03-09 08:00:10",
            "2026-03-09 08:00:20",
            "2026-03-09 08:00:30",
        ]
    ).to_numpy()

    X_seq, y_seq, seq_ts = safe_create_sequences(
        platform=platform,
        sensor_data=sensor_data,
        labels=labels,
        seq_length=3,
        room_name="Bathroom",
        timestamps=timestamps,
        strict=True,
    )

    assert len(X_seq) == 4
    assert y_seq.tolist() == [0, 0, 1, 1]
    assert pd.to_datetime(seq_ts).strftime("%Y-%m-%d %H:%M:%S").tolist() == [
        "2025-12-04 08:00:20",
        "2025-12-04 08:00:30",
        "2026-03-09 08:00:20",
        "2026-03-09 08:00:30",
    ]
