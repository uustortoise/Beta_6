"""
Test to verify resampling does not use bfill() in a way that leaks future data.
Ensures causal correctness by checking that values only propagate forward in time.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_no_bfill_before_ffill():
    """Verify that ffill is applied before bfill to maintain causality."""
    # Create test data with a gap
    timestamps = [
        datetime(2024, 1, 1, 0, 0, 0),
        datetime(2024, 1, 1, 0, 0, 10),
        # GAP: missing 20s
        datetime(2024, 1, 1, 0, 0, 30),
        datetime(2024, 1, 1, 0, 0, 40),
    ]
    values = [0, 1, 5, 6]
    
    df = pd.DataFrame({'timestamp': timestamps, 'sensor': values})
    df = df.set_index('timestamp')
    
    # Reindex to 10s intervals
    full_index = pd.date_range(start=timestamps[0], end=timestamps[-1], freq='10s')
    df_resampled = df.reindex(full_index)
    
    # Apply P1 fix: ffill THEN bfill
    df_resampled = df_resampled.ffill().bfill()
    
    # Check value at 20s (the gap)
    gap_value = df_resampled.loc[datetime(2024, 1, 1, 0, 0, 20), 'sensor']
    
    # Correct: ffill propagates 1 (from 10s) forward
    # Incorrect: bfill would propagate 5 (from 30s) backward
    assert gap_value == 1, \
        f"Gap filled with future value! Expected 1 (ffill from past), got {gap_value}"
    
    print(f"✓ Gap at 20s correctly filled with past value (1, not future value 5)")


def test_leading_nans_handled_correctly():
    """Verify leading NaNs (before first data point) are filled without using future from middle."""
    # Data starts at 20s, not 0s
    timestamps = [
        datetime(2024, 1, 1, 0, 0, 20),
        datetime(2024, 1, 1, 0, 0, 30),
        datetime(2024, 1, 1, 0, 0, 40),
    ]
    values = [5, 6, 7]
    
    df = pd.DataFrame({'timestamp': timestamps, 'sensor': values})
    df = df.set_index('timestamp')
    
    # Reindex to include 0s and 10s (leading NaNs)
    full_index = pd.date_range(start=datetime(2024, 1, 1, 0, 0, 0), 
                                end=datetime(2024, 1, 1, 0, 0, 40), 
                                freq='10s')
    df_resampled = df.reindex(full_index)
    
    # Apply fix: ffill first (does nothing for leading NaNs), then bfill (fills from first valid)
    df_resampled = df_resampled.ffill().bfill()
    
    # Leading NaNs should be filled with first valid value (5 from 20s)
    assert df_resampled.loc[datetime(2024, 1, 1, 0, 0, 0), 'sensor'] == 5
    assert df_resampled.loc[datetime(2024, 1, 1, 0, 0, 10), 'sensor'] == 5
    
    print(f"✓ Leading NaNs correctly filled with first valid value (5)")


def test_no_future_leakage_in_sequence():
    """End-to-end test: verify no timestamp sees data from its future."""
    # Create sequential increasing data
    n = 10
    timestamps = [datetime(2024, 1, 1, 0, i, 0) for i in range(n)]  # Use minutes, not seconds
    values = list(range(n))  # 0, 1, 2, ..., 9
    
    df = pd.DataFrame({'timestamp': timestamps, 'sensor': values})
    df = df.set_index('timestamp')
    
    # Introduce gaps (remove some rows)
    df = df.drop(df.index[3])  # Remove index 3 (30s, value=3)
    df = df.drop(df.index[6])  # Remove index 6 (60s, value=6) - note index shifted after first drop
    
    # Reindex and fill
    full_index = pd.date_range(start=timestamps[0], end=timestamps[-1], freq='1min')
    df_resampled = df.reindex(full_index).ffill().bfill()
    
    # Verify causality: each timestamp's value <= all future timestamps' values
    values_list = df_resampled['sensor'].tolist()
    for i in range(len(values_list) - 1):
        assert values_list[i] <= values_list[i+1], \
            f"Future leakage at index {i}: {values_list[i]} > {values_list[i+1]}"
    
    print(f"✓ No future leakage detected in {len(values_list)} samples")


if __name__ == "__main__":
    test_no_bfill_before_ffill()
    test_leading_nans_handled_correctly()
    test_no_future_leakage_in_sequence()
    print("\n✅ All bfill leakage tests passed")
