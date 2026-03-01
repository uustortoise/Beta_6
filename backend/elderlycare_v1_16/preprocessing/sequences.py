"""
Sequence creation utilities for time series data.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

def create_sequences(data: np.ndarray, seq_length: int = 50, 
                    stride: int = 1, padding: str = 'pre') -> np.ndarray:
    """
    Create sequences from time series data.
    
    Parameters:
    - data: Input data of shape (n_samples, n_features)
    - seq_length: Length of each sequence
    - stride: Step size between sequences
    - padding: 'pre', 'post', or 'none' for handling short sequences
    
    Returns:
    - sequences: Array of shape (n_sequences, seq_length, n_features)
    """
    if len(data) < seq_length:
        if padding == 'none':
            return np.array([]).reshape(0, seq_length, data.shape[1])
        elif padding == 'pre':
            # Pad with first value
            pad_width = seq_length - len(data)
            pad_values = np.tile(data[0:1], (pad_width, 1))
            data = np.vstack([pad_values, data])
        elif padding == 'post':
            # Pad with last value
            pad_width = seq_length - len(data)
            pad_values = np.tile(data[-1:], (pad_width, 1))
            data = np.vstack([data, pad_values])
    
    # Vectorized sequence creation - significantly faster and prevents memory fragmentation
    # which can cause deadlocks in TensorFlow on macOS
    num_samples, num_features = data.shape
    num_sequences = (num_samples - seq_length) // stride + 1
    
    # Use stride tricks for zero-copy view, then force copy to contiguous array
    strides = (stride * data.strides[0], data.strides[0], data.strides[1])
    sequences = np.lib.stride_tricks.as_strided(
        data, 
        shape=(num_sequences, seq_length, num_features),
        strides=strides
    )
    
    return np.ascontiguousarray(sequences)

def create_labeled_sequences(data: np.ndarray, labels: np.ndarray, 
                           seq_length: int = 50, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences with corresponding labels.
    
    Parameters:
    - data: Input data of shape (n_samples, n_features)
    - labels: Labels of shape (n_samples,)
    - seq_length: Length of each sequence
    - stride: Step size between sequences
    
    Returns:
    - sequences: Array of shape (n_sequences, seq_length, n_features)
    - sequence_labels: Array of shape (n_sequences,)
    """
    sequences = create_sequences(data, seq_length, stride, padding='none')
    
    if len(sequences) == 0:
        return np.array([]), np.array([])
    
    # Use the label of the last element in each sequence
    sequence_labels = labels[seq_length - 1::stride][:len(sequences)]
    
    return sequences, sequence_labels

def create_multi_step_sequences(data: np.ndarray, seq_length: int = 50, 
                              forecast_horizon: int = 1, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for multi-step forecasting.
    
    Parameters:
    - data: Input data of shape (n_samples, n_features)
    - seq_length: Length of input sequence
    - forecast_horizon: Number of steps to forecast
    - stride: Step size between sequences
    
    Returns:
    - X: Input sequences of shape (n_sequences, seq_length, n_features)
    - y: Target sequences of shape (n_sequences, forecast_horizon, n_features)
    """
    total_length = seq_length + forecast_horizon
    
    if len(data) < total_length:
        return np.array([]), np.array([])
    
    X, y = [], []
    for i in range(0, len(data) - total_length + 1, stride):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + total_length])
    
    return np.array(X), np.array(y)

def create_overlapping_sequences(data: np.ndarray, seq_length: int = 50, 
                               overlap: float = 0.5) -> np.ndarray:
    """
    Create sequences with specified overlap.
    
    Parameters:
    - data: Input data
    - seq_length: Length of each sequence
    - overlap: Overlap ratio between sequences (0 to 1)
    
    Returns:
    - sequences: Array of sequences
    """
    stride = int(seq_length * (1 - overlap))
    if stride < 1:
        stride = 1
    
    return create_sequences(data, seq_length, stride)

def split_sequences_by_time(data: pd.DataFrame, time_col: str = 'timestamp',
                          seq_length: int = 50, time_window: str = '1H') -> List[np.ndarray]:
    """
    Create sequences within specified time windows.
    
    Parameters:
    - data: DataFrame with time index
    - time_col: Name of timestamp column
    - seq_length: Minimum sequence length
    - time_window: Pandas time window string (e.g., '1H', '30T')
    
    Returns:
    - sequences: List of sequences for each time window
    """
    if time_col not in data.columns:
        raise ValueError(f"Time column '{time_col}' not found in data")
    
    # Set timestamp as index for resampling
    data = data.set_index(time_col)
    
    # Resample by time window
    resampled = data.resample(time_window)
    
    sequences = []
    for _, window_data in resampled:
        if len(window_data) >= seq_length:
            window_seq = create_sequences(window_data.values, seq_length, stride=1)
            sequences.extend(window_seq)
    
    return sequences
