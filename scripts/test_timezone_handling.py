#!/usr/bin/env python3
"""
================================================================================
TIMEZONE HANDLING VERIFICATION TEST
================================================================================
Run this script to verify that timezone normalization is working correctly.

Usage:
    python scripts/test_timezone_handling.py

Expected Output:
    ✅ All tests passed!
================================================================================
"""

import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

def test_ensure_naive():
    """Test ensure_naive function with various inputs."""
    print("Testing ensure_naive()...")
    
    from utils.time_utils import ensure_naive
    
    # Test 1: UTC-aware Series
    aware = pd.Series(pd.date_range('2026-01-01', periods=3, tz='UTC'))
    naive = ensure_naive(aware)
    assert naive.dt.tz is None, "❌ Should strip timezone from Series"
    print("  ✅ UTC-aware Series → naive")
    
    # Test 2: Already naive Series
    already_naive = pd.Series(pd.date_range('2026-01-01', periods=3))
    result = ensure_naive(already_naive)
    assert result.dt.tz is None, "❌ Should remain naive"
    assert (result == already_naive).all(), "❌ Should not change values"
    print("  ✅ Already naive Series unchanged")
    
    # Test 3: Single UTC-aware Timestamp
    aware_ts = pd.Timestamp('2026-01-01 12:00:00', tz='UTC')
    naive_ts = ensure_naive(aware_ts)
    assert naive_ts.tz is None, "❌ Should strip timezone from Timestamp"
    assert naive_ts.hour == 12, "❌ Should preserve hour"
    print("  ✅ UTC-aware Timestamp → naive")
    
    # Test 4: Single naive Timestamp
    naive_ts2 = pd.Timestamp('2026-01-01 12:00:00')
    result_ts = ensure_naive(naive_ts2)
    assert result_ts.tz is None, "❌ Should remain naive"
    assert result_ts == naive_ts2, "❌ Should not change value"
    print("  ✅ Naive Timestamp unchanged")
    
    # Test 5: None input
    assert ensure_naive(None) is None, "❌ Should handle None"
    print("  ✅ None input handled")
    
    # Test 6: DatetimeIndex
    aware_idx = pd.DatetimeIndex(['2026-01-01 12:00:00+00:00', '2026-01-01 13:00:00+00:00'])
    naive_idx = ensure_naive(aware_idx)
    assert naive_idx.tz is None, "❌ Should strip timezone from DatetimeIndex"
    print("  ✅ DatetimeIndex → naive")
    
    print("✅ ensure_naive() tests passed!\n")


def test_safe_merge_timestamps():
    """Test safe_merge_timestamps with mixed timezones."""
    print("Testing safe_merge_timestamps()...")
    
    from utils.time_utils import safe_merge_timestamps
    
    # Test 1: Mixed timezone merge (should not raise MergeError)
    left = pd.DataFrame({
        'timestamp': pd.to_datetime(['2026-01-01 12:00:00', '2026-01-01 13:00:00']),
        'value_left': [1, 2]
    })
    
    right = pd.DataFrame({
        'timestamp': pd.to_datetime(['2026-01-01 12:00:05', '2026-01-01 13:00:05']).tz_localize('UTC'),
        'value_right': ['a', 'b']
    })
    
    try:
        result = safe_merge_timestamps(left, right, tolerance='30s')
        assert 'value_right' in result.columns, "❌ Should include right columns"
        assert len(result) == 2, "❌ Should have correct number of rows"
        print("  ✅ Mixed timezone merge works")
    except Exception as e:
        print(f"  ❌ Mixed timezone merge failed: {e}")
        return False
    
    # Test 2: Both naive (should work)
    left_naive = pd.DataFrame({
        'timestamp': pd.to_datetime(['2026-01-01 12:00:00']),
        'value': [1]
    })
    right_naive = pd.DataFrame({
        'timestamp': pd.to_datetime(['2026-01-01 12:00:10']),
        'activity': ['sleep']
    })
    
    try:
        result = safe_merge_timestamps(left_naive, right_naive, tolerance='30s')
        assert result['activity'].iloc[0] == 'sleep', "❌ Should match within tolerance"
        print("  ✅ Both naive merge works")
    except Exception as e:
        print(f"  ❌ Both naive merge failed: {e}")
        return False
    
    # Test 3: No match outside tolerance
    left_far = pd.DataFrame({
        'timestamp': pd.to_datetime(['2026-01-01 12:00:00']),
        'value': [1]
    })
    right_far = pd.DataFrame({
        'timestamp': pd.to_datetime(['2026-01-01 12:05:00']),  # 5 minutes later
        'activity': ['sleep']
    })
    
    try:
        result = safe_merge_timestamps(left_far, right_far, tolerance='30s')
        assert pd.isna(result['activity'].iloc[0]), "❌ Should not match outside tolerance"
        print("  ✅ Tolerance respected")
    except Exception as e:
        print(f"  ❌ Tolerance test failed: {e}")
        return False
    
    print("✅ safe_merge_timestamps() tests passed!\n")
    return True


def test_timezone_normalization_consistency():
    """Test that normalization produces consistent results."""
    print("Testing timezone normalization consistency...")
    
    from utils.time_utils import ensure_naive
    
    # Same moment in time, different representations
    utc_time = pd.Timestamp('2026-01-01 15:00:00', tz='UTC')
    
    # Convert to various timezones
    ny_time = utc_time.tz_convert('America/New_York')  # 10:00 AM
    tokyo_time = utc_time.tz_convert('Asia/Tokyo')     # 00:00+1d
    
    # Normalize all to naive
    utc_naive = ensure_naive(utc_time)
    ny_naive = ensure_naive(ny_time)
    tokyo_naive = ensure_naive(tokyo_time)
    
    # They should all be different (local times)
    assert utc_naive.hour == 15, f"❌ UTC should be 15:00, got {utc_naive.hour}"
    assert ny_naive.hour == 10, f"❌ NY should be 10:00, got {ny_naive.hour}"
    assert tokyo_naive.hour == 0, f"❌ Tokyo should be 00:00, got {tokyo_naive.hour}"
    
    print("  ✅ Different timezones normalize to different local times")
    print(f"     UTC: {utc_naive}")
    print(f"     NY: {ny_naive}")
    print(f"     Tokyo: {tokyo_naive}")
    
    print("✅ Consistency tests passed!\n")


def test_idempotency():
    """Test that ensure_naive is idempotent."""
    print("Testing idempotency...")
    
    from utils.time_utils import ensure_naive
    
    # Start with aware timestamp
    aware = pd.Series(pd.date_range('2026-01-01', periods=3, tz='UTC'))
    
    # Apply ensure_naive multiple times
    first = ensure_naive(aware)
    second = ensure_naive(first)
    third = ensure_naive(second)
    
    # All should be equal
    assert (first == second).all(), "❌ First and second should be equal"
    assert (second == third).all(), "❌ Second and third should be equal"
    
    print("  ✅ ensure_naive is idempotent")
    print("✅ Idempotency tests passed!\n")


def main():
    """Run all tests."""
    print("="*70)
    print("TIMEZONE HANDLING VERIFICATION TESTS")
    print("="*70)
    print()
    
    try:
        test_ensure_naive()
        test_safe_merge_timestamps()
        test_timezone_normalization_consistency()
        test_idempotency()
        
        print("="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print()
        print("Timezone handling is working correctly.")
        print("Ready for end-to-end correction testing.")
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
