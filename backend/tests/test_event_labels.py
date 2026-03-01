import pandas as pd

from ml.event_labels import EventEpisode, build_daily_event_targets, labels_to_episodes


def _df(rows):
    return pd.DataFrame(rows, columns=["timestamp", "activity"])


def test_labels_to_episodes_merges_and_splits_on_gap():
    df = _df(
        [
            ("2025-12-04 00:00:00", "unoccupied"),
            ("2025-12-04 00:00:10", "unoccupied"),
            ("2025-12-04 00:00:20", "sleep"),
            ("2025-12-04 00:00:30", "sleep"),
            ("2025-12-04 00:10:50", "sleep"),
        ]
    )

    episodes = labels_to_episodes(df, max_gap_seconds=300, sample_interval_seconds=10)

    assert [e.label for e in episodes] == ["unoccupied", "sleep", "sleep"]
    assert episodes[0].sample_count == 2
    assert episodes[1].sample_count == 2
    assert episodes[2].sample_count == 1


def test_labels_to_episodes_respects_min_duration():
    df = _df(
        [
            ("2025-12-04 00:00:00", "sleep"),
            ("2025-12-04 00:00:10", "sleep"),
            ("2025-12-04 00:00:20", "bathroom_normal_use"),
        ]
    )

    episodes = labels_to_episodes(
        df,
        min_duration_seconds=15,
        sample_interval_seconds=10,
    )

    assert [e.label for e in episodes] == ["sleep"]


def test_build_daily_event_targets_core_fields():
    episodes = [
        EventEpisode(
            label="sleep",
            start_time=pd.Timestamp("2025-12-04 00:00:00"),
            end_time=pd.Timestamp("2025-12-04 01:00:00"),
            duration_seconds=3600,
            sample_count=360,
        ),
        EventEpisode(
            label="shower",
            start_time=pd.Timestamp("2025-12-04 07:00:00"),
            end_time=pd.Timestamp("2025-12-04 07:10:00"),
            duration_seconds=600,
            sample_count=60,
        ),
        EventEpisode(
            label="unoccupied",
            start_time=pd.Timestamp("2025-12-04 12:00:00"),
            end_time=pd.Timestamp("2025-12-04 13:00:00"),
            duration_seconds=3600,
            sample_count=360,
        ),
    ]

    out = build_daily_event_targets(episodes)

    assert out["sleep_minutes"] == 60.0
    assert out["shower_day"] == 1.0
    assert out["shower_events"] == 1.0
    assert out["occupied_minutes"] == 70.0
    assert out["unoccupied_minutes"] == 60.0
