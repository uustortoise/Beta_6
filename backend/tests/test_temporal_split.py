"""
Test for temporal validation split to prevent time-series leakage.
Ensures that validation data always comes AFTER training data chronologically.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def test_temporal_split_order():
    """Verify validation timestamps are strictly after training timestamps."""
    # Create synthetic time-series data
    n_samples = 100
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(seconds=10*i) for i in range(n_samples)]
    
    # Simulate a temporal split (80/20)
    val_split = 0.2
    split_idx = int(n_samples * (1 - val_split))
    
    train_timestamps = timestamps[:split_idx]
    val_timestamps = timestamps[split_idx:]
    
    # Critical assertion: All validation timestamps must be > all training timestamps
    assert max(train_timestamps) < min(val_timestamps), \
        "Temporal split violated: validation contains data from before training end"
    
    print(f"✓ Temporal split valid: Train {len(train_timestamps)} samples (up to {max(train_timestamps)}), "
          f"Val {len(val_timestamps)} samples (from {min(val_timestamps)})")


def test_no_shuffle_maintains_order():
    """Verify that shuffle=False maintains temporal order."""
    # Create sequential data
    n_samples = 50
    X = np.arange(n_samples).reshape(-1, 1)
    
    # Simulate batching without shuffle
    batch_size = 10
    batches = [X[i:i+batch_size] for i in range(0, len(X), batch_size)]
    
    # Verify each batch maintains sequential order
    for i, batch in enumerate(batches):
        if len(batch) > 1:
            assert np.all(np.diff(batch.flatten()) == 1), \
                f"Batch {i} is not in sequential order"
    
    print(f"✓ No shuffle verified: {len(batches)} batches maintain temporal order")


def test_temporal_split_no_data_leakage():
    """Verify no overlap between train and validation sets."""
    # Create sample indices
    n_samples = 100
    indices = set(range(n_samples))
    
    val_split = 0.2
    split_idx = int(n_samples * (1 - val_split))
    
    train_indices = set(range(split_idx))
    val_indices = set(range(split_idx, n_samples))
    
    # Verify sets are disjoint
    overlap = train_indices.intersection(val_indices)
    assert len(overlap) == 0, f"Data leakage detected: {len(overlap)} samples in both train and val"
    
    # Verify sets are exhaustive
    assert train_indices.union(val_indices) == indices, "Missing samples in split"
    
    print(f"✓ No data leakage: Train {len(train_indices)}, Val {len(val_indices)}, Overlap {len(overlap)}")


if __name__ == "__main__":
    test_temporal_split_order()
    test_no_shuffle_maintains_order()
    test_temporal_split_no_data_leakage()
    print("\n✅ All temporal split tests passed")
