"""
Schema Validation Tests

These tests verify that database query functions return DataFrames
with the expected columns. Prevents bugs like missing 'record_date'
in fetch_golden_samples.
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestQuerySchemas:
    """Verify expected columns are returned by query functions."""
    
    def test_fetch_golden_samples_schema(self):
        """fetch_golden_samples must return timestamp, activity, record_date."""
        from ml.utils import fetch_golden_samples
        
        # This test uses a known elder/room combo; adjust as needed
        df = fetch_golden_samples('HK002_samuel', 'Bathroom')
        
        if df is None or df.empty:
            pytest.skip("No golden samples available for test")
        
        required_columns = ['timestamp', 'activity', 'record_date']
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Verify types
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp']), \
            "timestamp should be datetime type"
    
    def test_fetch_all_golden_samples_schema(self):
        """fetch_all_golden_samples must return room, timestamp, activity."""
        from ml.utils import fetch_all_golden_samples
        
        df = fetch_all_golden_samples('HK002_samuel')
        
        if df is None or df.empty:
            pytest.skip("No golden samples available for test")
        
        required_columns = ['room', 'timestamp', 'activity']
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_fetch_predictions_returns_timestamp(self):
        """fetch_predictions must return timestamp column."""
        # Import here to ensure DB connection works
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from export_dashboard import fetch_predictions
        
        df = fetch_predictions('HK002_samuel', 'bathroom', '2026-01-26')
        
        if df.empty:
            pytest.skip("No predictions available for test")
        
        assert 'timestamp' in df.columns, "Missing timestamp column"
        assert 'predicted_activity' in df.columns, "Missing predicted_activity column"


class TestTimestampUtils:
    """Verify timezone utility functions work correctly."""
    
    def test_ensure_naive_with_aware_series(self):
        """ensure_naive should strip timezone from aware Series."""
        from utils.time_utils import ensure_naive
        
        # Create UTC-aware Series
        ts = pd.to_datetime(['2026-01-01 00:00:00', '2026-01-01 01:00:00'])
        ts_utc = ts.tz_localize('UTC')
        series = pd.Series(ts_utc)
        
        result = ensure_naive(series)
        
        assert result.dt.tz is None, "Timezone should be stripped"
        assert len(result) == 2, "Should preserve length"
    
    def test_ensure_naive_with_naive_series(self):
        """ensure_naive should return naive Series unchanged."""
        from utils.time_utils import ensure_naive
        
        ts = pd.to_datetime(['2026-01-01 00:00:00', '2026-01-01 01:00:00'])
        series = pd.Series(ts)
        
        result = ensure_naive(series)
        
        assert result.dt.tz is None, "Should remain naive"
        assert list(result) == list(series), "Values should be unchanged"
    
    def test_safe_merge_timestamps_mixed_tz(self):
        """safe_merge_timestamps should handle mixed timezone DataFrames."""
        from utils.time_utils import safe_merge_timestamps
        
        # Left: naive
        left = pd.DataFrame({
            'timestamp': pd.to_datetime(['2026-01-01 00:00:00', '2026-01-01 01:00:00']),
            'value_a': [1, 2]
        })
        
        # Right: UTC-aware
        right_ts = pd.to_datetime(['2026-01-01 00:00:00', '2026-01-01 01:00:00'])
        right = pd.DataFrame({
            'timestamp': right_ts.tz_localize('UTC'),
            'value_b': ['x', 'y']
        })
        
        # Should not raise MergeError
        result = safe_merge_timestamps(left, right)
        
        assert 'value_a' in result.columns
        assert 'value_b' in result.columns
        assert len(result) == 2
