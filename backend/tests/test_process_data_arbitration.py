import pandas as pd

import process_data


def _is_exclusive_occupied(label: str) -> bool:
    token = str(label or "").strip().lower()
    return token not in {"inactive", "unoccupied", "out", "unknown", "low_confidence"}


def test_pre_persistence_arbitration_resolves_cross_room_contradictions(monkeypatch):
    monkeypatch.setenv("ENABLE_PRE_PERSISTENCE_EVENT_DECODER", "false")

    timestamps = pd.date_range("2026-03-01 10:00:00", periods=2, freq="10s")
    prediction_results = {
        "bedroom": pd.DataFrame(
            {
                "timestamp": timestamps,
                "predicted_activity": ["sleep", "sleep"],
                "confidence": [0.90, 0.40],
            }
        ),
        "kitchen": pd.DataFrame(
            {
                "timestamp": timestamps,
                "predicted_activity": ["cooking", "cooking"],
                "confidence": [0.80, 0.95],
            }
        ),
    }

    adjusted, report = process_data._apply_pre_persistence_arbitration(prediction_results)

    assert report["status"] == "ok"
    assert report["contradiction_timestamps"] == 2
    assert report["adjustments"] == 2

    for ts in timestamps:
        occupied_count = 0
        for room_df in adjusted.values():
            row = room_df.loc[room_df["timestamp"] == ts]
            if row.empty:
                continue
            if _is_exclusive_occupied(str(row.iloc[0]["predicted_activity"])):
                occupied_count += 1
        assert occupied_count <= 1


def test_pre_persistence_arbitration_skips_when_single_room_input(monkeypatch):
    monkeypatch.setenv("ENABLE_PRE_PERSISTENCE_EVENT_DECODER", "false")

    prediction_results = {
        "bedroom": pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-03-01 10:00:00", periods=2, freq="10s"),
                "predicted_activity": ["sleep", "sleep"],
                "confidence": [0.90, 0.92],
            }
        )
    }

    adjusted, report = process_data._apply_pre_persistence_arbitration(prediction_results)

    assert report["status"] == "skipped"
    assert report["reason"] == "insufficient_room_inputs"
    assert list(adjusted.keys()) == ["bedroom"]
