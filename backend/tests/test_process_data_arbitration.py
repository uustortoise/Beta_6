import pandas as pd

import process_data
from ml.home_empty_fusion import HomeEmptyPrediction, HomeEmptyState, RoomState


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


def test_pre_persistence_arbitration_passes_resident_context_to_fusion(monkeypatch):
    monkeypatch.setenv("ENABLE_PRE_PERSISTENCE_EVENT_DECODER", "false")

    captured = {}

    class _FakeAnalyzer:
        def get_resident_home_context_contract(self, elder_id=None):
            captured["elder_id"] = elder_id
            return {
                "status": "ready",
                "household_type": "multi",
                "helper_presence": "present",
                "layout_topology": {"bedroom": ["entrance"]},
                "missing_required_fields": [],
                "message": "Resident/home context contract is complete.",
            }

    class _FakeFusion:
        def __init__(self, config=None):
            captured["resident_home_context"] = getattr(config, "resident_home_context", None)

        def fuse(self, room_predictions, timestamps):
            ts = timestamps[0]
            return [
                HomeEmptyPrediction(
                    timestamp=ts,
                    state=HomeEmptyState.EMPTY,
                    confidence=0.9,
                    room_states=[
                        RoomState("bedroom", ts, 0.1, False),
                        RoomState("kitchen", ts, 0.1, False),
                    ],
                    unoccupied_room_count=2,
                    total_room_count=2,
                )
            ]

    monkeypatch.setattr(process_data, "HouseholdAnalyzer", lambda: _FakeAnalyzer())
    monkeypatch.setattr("ml.home_empty_fusion.HomeEmptyFusion", _FakeFusion)

    prediction_results = {
        "bedroom": pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-03-01 10:00:00", periods=1, freq="10s"),
                "predicted_activity": ["sleep"],
                "confidence": [0.90],
            }
        ),
        "kitchen": pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-03-01 10:00:00", periods=1, freq="10s"),
                "predicted_activity": ["cooking"],
                "confidence": [0.85],
            }
        ),
    }

    _, report = process_data._apply_pre_persistence_arbitration(
        prediction_results,
        elder_id="elder_123",
    )

    assert report["status"] == "ok"
    assert captured["elder_id"] == "elder_123"
    assert captured["resident_home_context"]["household_type"] == "multi"
    assert captured["resident_home_context"]["helper_presence"] == "present"
