from datetime import datetime
import pandas as pd
import pytest
from utils.data_loader import clean_and_resample

def test_clean_and_resample_attaches_diagnostics():
    data = {
        "LivingRoom": pd.DataFrame({
            "timestamp": [
                datetime(2026, 1, 1, 0, 0, 0),
                datetime(2026, 1, 1, 0, 0, 30), # Gap of 30s
            ],
            "motion": [1.0, 0.0],
            "activity": ["idle", "idle"]
        })
    }
    
    # 10s interval -> should produce timestamps at :00, :10, :20, :30 (4 rows)
    cleaned = clean_and_resample(data, interval_seconds=10)
    df = cleaned["LivingRoom"]
    
    assert "raw_rows_before_resample" in df.attrs
    assert df.attrs["raw_rows_before_resample"] == 2
    
    # Resample should add rows for :10 and :20
    assert "rows_after_resample" in df.attrs
    assert df.attrs["rows_after_resample"] == 4
    assert len(df) == 4

def test_clean_and_resample_uses_bounded_ffill_policy(monkeypatch):
    monkeypatch.setenv("MAX_RESAMPLE_FFILL_GAP_SECONDS", "60")
    data = {
        "Bedroom": pd.DataFrame(
            {
                "timestamp": [
                    datetime(2026, 1, 1, 0, 0, 0),
                    datetime(2026, 1, 1, 0, 0, 10),
                    datetime(2026, 1, 1, 0, 2, 0),
                ],
                "motion": [1.0, 2.0, 9.0],
                "activity": ["sleep", "sleep", "wake"],
            }
        )
    }

    out = clean_and_resample(data, interval_seconds=10)["Bedroom"]
    series = out.set_index("timestamp")["motion"]

    assert series.loc[datetime(2026, 1, 1, 0, 1, 10)] == 2.0
    assert pd.isna(series.loc[datetime(2026, 1, 1, 0, 1, 20)])
    assert series.loc[datetime(2026, 1, 1, 0, 2, 0)] == 9.0


def test_loader_and_platform_gap_tokens_match(monkeypatch):
    data = {
        "Bedroom": pd.DataFrame(
            {
                "timestamp": [
                    datetime(2026, 1, 1, 0, 0, 0),
                    datetime(2026, 1, 1, 0, 0, 10),
                    datetime(2026, 1, 1, 0, 2, 0),
                ],
                "motion": [1.0, 2.0, 9.0],
                "activity": ["sleep", "sleep", "wake"],
            }
        )
    }

    # Default behavior is bounded to 60s without env dependency.
    bounded = clean_and_resample(data, interval_seconds=10)["Bedroom"].set_index("timestamp")["motion"]
    assert pd.isna(bounded.loc[datetime(2026, 1, 1, 0, 1, 20)])

    # Explicit unbounded behavior must be injected.
    unbounded = clean_and_resample(
        data,
        interval_seconds=10,
        max_ffill_gap_seconds=None,
    )["Bedroom"].set_index("timestamp")["motion"]
    assert unbounded.loc[datetime(2026, 1, 1, 0, 1, 20)] == 2.0


def test_clean_and_resample_fails_closed_when_resampler_errors(monkeypatch):
    data = {
        "Bedroom": pd.DataFrame(
            {
                "timestamp": [datetime(2026, 1, 1, 0, 0, 0)],
                "motion": [1.0],
                "activity": ["sleep"],
            }
        )
    }
    monkeypatch.setattr(
        "utils.data_loader.resample_to_fixed_interval",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )
    with pytest.raises(ValueError, match="Resampling failed for Bedroom"):
        clean_and_resample(data, interval_seconds=10)

def test_clean_and_resample_fails_on_missing_timestamp():
    """
    P1 Hardening: Data loader must fail closed if timestamp column is missing.
    """
    data = {
        "Bedroom": pd.DataFrame({
            "motion": [1.0, 0.0],
            "activity": ["sleep", "sleep"]
            # timestamp column missing
        })
    }
    
    with pytest.raises(ValueError, match="Missing required 'timestamp' column"):
        clean_and_resample(data, interval_seconds=10)
