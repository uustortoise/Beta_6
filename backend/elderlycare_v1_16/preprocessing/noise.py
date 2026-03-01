"""
Noise treatment functions for sensor data.

Revised and production-ready implementation.

Features:
- Window parity handling with warnings
- Safer integer casting with proper rounding and nullable integer support
- All-NaN column skipping
- Comprehensive input validation
- Proper inplace behavior (returns None when inplace=True)
- Detailed diagnostics via return_stats parameter
- Vectorized operations for performance (index-aware)
- Preserved index and timezone
- Non-numeric column handling
"""

from typing import Union, List, Tuple, Dict, Any, Optional
import warnings
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _vectorized_rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling median for 2D array using a simple vectorized approach.

    Note: For very large arrays consider using optimized libraries (bottleneck, numba).
    """
    n_samples, n_features = arr.shape
    result = np.full_like(arr, np.nan, dtype=float)

    half = window // 2
    for i in range(n_samples):
        start = max(0, i - half)
        end = min(n_samples, i + half + 1)
        if end - start >= 1:
            result[i, :] = np.nanmedian(arr[start:end, :], axis=0)

    return result


def _vectorized_rolling_quantile(arr: np.ndarray, window: int, q: float) -> np.ndarray:
    """
    Compute rolling quantile for 2D array using a simple vectorized approach.
    """
    n_samples, n_features = arr.shape
    result = np.full_like(arr, np.nan, dtype=float)

    half = window // 2
    for i in range(n_samples):
        start = max(0, i - half)
        end = min(n_samples, i + half + 1)
        if end - start >= 1:
            result[i, :] = np.nanquantile(arr[start:end, :], q, axis=0)

    return result


def _to_index_positions(index: pd.Index, boolean_mask: np.ndarray) -> pd.Index:
    """
    Convert a boolean numpy mask aligned to the index positions into an Index of labels.
    """
    if boolean_mask.dtype != bool:
        boolean_mask = boolean_mask.astype(bool)
    return index[boolean_mask]


def _assign_values_with_index(target_df: pd.DataFrame, idx_positions: np.ndarray, col: str, values: np.ndarray):
    """
    Assign values to target_df[col] at positions specified by idx_positions (boolean mask or index labels).
    Handles pandas nullable integer dtypes safely.
    """
    # idx_positions may be boolean mask or index labels
    if isinstance(idx_positions, np.ndarray) and idx_positions.dtype == bool:
        labels = target_df.index[idx_positions]
    else:
        labels = idx_positions

    # Create a Series with the same index to preserve alignment and dtype handling
    s = pd.Series(values, index=labels)

    # If target column is a pandas nullable integer dtype, cast via Series.astype to preserve NA
    orig_dtype = target_df[col].dtype
    if pd.api.types.is_integer_dtype(orig_dtype) and not isinstance(orig_dtype, np.dtype):
        # pandas nullable integer (e.g., Int64) - not a numpy dtype
        s = s.round().astype(orig_dtype)
        target_df.loc[labels, col] = s
    else:
        # For numpy integer or float types
        try:
            if pd.api.types.is_integer_dtype(orig_dtype):
                target_df.loc[labels, col] = np.rint(s.values).astype(orig_dtype)
            else:
                target_df.loc[labels, col] = s.values.astype(orig_dtype)
        except Exception:
            # Fallback: assign as-is (may upcast)
            target_df.loc[labels, col] = s.values


def hampel_filter(
    df: pd.DataFrame,
    columns: Union[List[str], Tuple[str], np.ndarray],
    window: int = 5,
    n_sigmas: float = 3.0,
    inplace: bool = False,
    return_stats: bool = False,
    use_vectorized: bool = False
) -> Union[None, pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]]:
    """
    Detect and replace isolated spikes using rolling median + MAD (Hampel filter).

    Returns:
      - None if inplace=True and return_stats=False
      - (None, stats) if inplace=True and return_stats=True
      - df if inplace=False and return_stats=False
      - (df, stats) if inplace=False and return_stats=True
    """
    # Input validation
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if n_sigmas <= 0:
        raise ValueError("n_sigmas must be positive")
    if isinstance(columns, str):
        raise ValueError("columns must be an iterable of column names, not a string")
    if not hasattr(columns, "__iter__"):
        raise ValueError("columns must be an iterable of column names")

    # Window parity handling
    original_window = window
    if window % 2 == 0:
        window += 1
        warnings.warn(
            f"Window size {original_window} is even. Using {window} for symmetric window.",
            UserWarning,
            stacklevel=2,
        )

    # Prepare target DataFrame
    if inplace:
        target_df = df
    else:
        target_df = df.copy(deep=True)

    # Constants
    MAD_TO_STD = 1.4826

    stats: Dict[str, Dict[str, Any]] = {}
    columns_list = list(columns)

    # Filter numeric, existing, non-all-NaN columns
    numeric_columns: List[str] = []
    for col in columns_list:
        if col not in target_df.columns:
            logger.debug("Column '%s' not found in DataFrame, skipping", col)
            continue
        if not pd.api.types.is_numeric_dtype(target_df[col].dtype):
            logger.debug("Column '%s' is non-numeric (dtype: %s), skipping", col, target_df[col].dtype)
            continue
        if target_df[col].isna().all():
            logger.debug("Column '%s' is all NaN, skipping", col)
            continue
        numeric_columns.append(col)

    if not numeric_columns:
        if inplace:
            return (None, stats) if return_stats else None
        else:
            return (target_df, stats) if return_stats else target_df

    # Vectorized path
    if use_vectorized and len(numeric_columns) > 1:
        try:
            data = target_df[numeric_columns].to_numpy(dtype=float, copy=True)
            n_samples, n_features = data.shape

            # Preserve index for alignment
            idx = target_df.index

            # Handle NaNs with index-aware fill
            nan_mask = np.isnan(data)
            if nan_mask.any():
                temp_df = pd.DataFrame(data, index=idx)
                data_filled = temp_df.ffill().bfill().to_numpy(dtype=float)
            else:
                data_filled = data

            rolling_median = _vectorized_rolling_median(data_filled, window)
            abs_diff = np.abs(data_filled - rolling_median)
            rolling_mad = _vectorized_rolling_median(abs_diff, window)
            threshold = n_sigmas * MAD_TO_STD * rolling_mad
            spike_mask = abs_diff > threshold

            for i, col in enumerate(numeric_columns):
                col_spike_mask = spike_mask[:, i]
                col_valid_median = ~np.isnan(rolling_median[:, i])
                col_replace_mask = col_spike_mask & col_valid_median

                if col_replace_mask.any():
                    median_vals = rolling_median[col_replace_mask, i]
                    # Convert boolean mask to index labels
                    labels = _to_index_positions(idx, col_replace_mask)
                    _assign_values_with_index(target_df, labels, col, median_vals)

                    n_corrections = int(col_replace_mask.sum())
                    logger.info("Hampel filter corrected %d spikes in column '%s'", n_corrections, col)

                    if return_stats:
                        stats[col] = {
                            "n_corrections": n_corrections,
                            "original_dtype": str(target_df[col].dtype),
                            "window_used": window,
                            "threshold_used": float(np.nanmean(threshold[:, i]) if not np.all(np.isnan(threshold[:, i])) else 0.0),
                        }

                # Restore original NaNs
                if nan_mask[:, i].any():
                    nan_labels = _to_index_positions(idx, nan_mask[:, i])
                    target_df.loc[nan_labels, col] = np.nan

        except Exception as e:
            logger.warning("Vectorized Hampel filter failed: %s. Falling back to per-column processing.", e)
            use_vectorized = False

    # Per-column path
    if not use_vectorized or len(numeric_columns) == 1:
        for col in numeric_columns:
            series = target_df[col]
            original_dtype = series.dtype
            series_copy = series.copy()

            nan_mask = series_copy.isna()
            if nan_mask.any():
                series_filled = series_copy.ffill().bfill()
            else:
                series_filled = series_copy

            # If still all NaN after fill, skip
            if series_filled.isna().all():
                logger.debug("Column '%s' remains all NaN after fill, skipping", col)
                continue

            rolling_median = series_filled.rolling(window=window, center=True, min_periods=1).median()
            abs_diff = (series_filled - rolling_median).abs()
            rolling_mad = abs_diff.rolling(window=window, center=True, min_periods=1).median()
            threshold = n_sigmas * MAD_TO_STD * rolling_mad
            spike_mask = abs_diff > threshold

            if spike_mask.any():
                valid_median_mask = ~rolling_median.isna()
                replace_mask = spike_mask & valid_median_mask

                if replace_mask.any():
                    median_vals = rolling_median[replace_mask].to_numpy(dtype=float)
                    labels = target_df.index[replace_mask.to_numpy(dtype=bool)]
                    _assign_values_with_index(target_df, labels, col, median_vals)

                    n_corrections = int(replace_mask.sum())
                    logger.info("Hampel filter corrected %d spikes in column '%s'", n_corrections, col)

                    if return_stats:
                        stats[col] = {
                            "n_corrections": n_corrections,
                            "original_dtype": str(original_dtype),
                            "window_used": window,
                            "threshold_used": float(threshold.mean() if not threshold.isna().all() else 0.0),
                        }

            # Restore original NaNs
            if nan_mask.any():
                nan_labels = target_df.index[nan_mask.to_numpy(dtype=bool)]
                target_df.loc[nan_labels, col] = np.nan

    # Return according to inplace/return_stats
    if inplace:
        return (None, stats) if return_stats else None
    else:
        return (target_df, stats) if return_stats else target_df


def clip_outliers(
    df: pd.DataFrame,
    columns: Union[List[str], Tuple[str], np.ndarray],
    method: str = "mad",
    factor: float = 3.0,
    window: int = 5,
    inplace: bool = False,
    return_stats: bool = False,
    use_vectorized: bool = False,
) -> Union[None, pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]]:
    """
    Clip extreme values using robust thresholds (MAD or IQR).

    Returns same shapes/semantics as hampel_filter regarding inplace/return_stats.
    """
    # Input validation
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if factor <= 0:
        raise ValueError("factor must be positive")
    if isinstance(columns, str):
        raise ValueError("columns must be an iterable of column names, not a string")
    if not hasattr(columns, "__iter__"):
        raise ValueError("columns must be an iterable of column names")
    if method.lower() not in ["mad", "iqr"]:
        raise ValueError(f"Unknown method: {method}. Use 'mad' or 'iqr'.")

    original_window = window
    if window % 2 == 0:
        window += 1
        warnings.warn(
            f"Window size {original_window} is even. Using {window} for symmetric window.",
            UserWarning,
            stacklevel=2,
        )

    if inplace:
        target_df = df
    else:
        target_df = df.copy(deep=True)

    stats: Dict[str, Dict[str, Any]] = {}
    columns_list = list(columns)

    numeric_columns: List[str] = []
    for col in columns_list:
        if col not in target_df.columns:
            logger.debug("Column '%s' not found in DataFrame, skipping", col)
            continue
        if not pd.api.types.is_numeric_dtype(target_df[col].dtype):
            logger.debug("Column '%s' is non-numeric (dtype: %s), skipping", col, target_df[col].dtype)
            continue
        if target_df[col].isna().all():
            logger.debug("Column '%s' is all NaN, skipping", col)
            continue
        numeric_columns.append(col)

    if not numeric_columns:
        if inplace:
            return (None, stats) if return_stats else None
        else:
            return (target_df, stats) if return_stats else target_df

    # Vectorized path
    if use_vectorized and len(numeric_columns) > 1:
        try:
            data = target_df[numeric_columns].to_numpy(dtype=float, copy=True)
            n_samples, n_features = data.shape
            idx = target_df.index

            nan_mask = np.isnan(data)
            if nan_mask.any():
                temp_df = pd.DataFrame(data, index=idx)
                data_filled = temp_df.ffill().bfill().to_numpy(dtype=float)
            else:
                data_filled = data

            if method.lower() == "mad":
                rolling_median = _vectorized_rolling_median(data_filled, window)
                abs_diff = np.abs(data_filled - rolling_median)
                rolling_mad = _vectorized_rolling_median(abs_diff, window)
                MAD_TO_STD = 1.4826
                threshold = factor * MAD_TO_STD * rolling_mad
                lower_bound = rolling_median - threshold
                upper_bound = rolling_median + threshold
            else:  # iqr
                rolling_q1 = _vectorized_rolling_quantile(data_filled, window, 0.25)
                rolling_q3 = _vectorized_rolling_quantile(data_filled, window, 0.75)
                rolling_iqr = rolling_q3 - rolling_q1
                lower_bound = rolling_q1 - factor * rolling_iqr
                upper_bound = rolling_q3 + factor * rolling_iqr

            clipped_data = np.clip(data_filled, lower_bound, upper_bound)

            for i, col in enumerate(numeric_columns):
                clipped_mask = (data_filled[:, i] < lower_bound[:, i]) | (data_filled[:, i] > upper_bound[:, i])
                n_clipped = int(clipped_mask.sum())

                if n_clipped > 0:
                    logger.info("Clipped %d outliers in column '%s' using %s method", n_clipped, col, method.upper())

                original_dtype = target_df[col].dtype

                # Assign back with index-aware helper
                labels = _to_index_positions(idx, clipped_mask)
                col_values = clipped_data[:, i]
                if pd.api.types.is_integer_dtype(original_dtype) and not isinstance(original_dtype, np.dtype):
                    # pandas nullable integer
                    s = pd.Series(np.rint(col_values), index=idx).astype(original_dtype)
                    target_df[col] = s
                else:
                    if pd.api.types.is_integer_dtype(original_dtype):
                        target_df[col] = np.rint(col_values).astype(original_dtype)
                    else:
                        target_df[col] = col_values.astype(original_dtype)

                # Restore original NaNs
                if nan_mask[:, i].any():
                    nan_labels = _to_index_positions(idx, nan_mask[:, i])
                    target_df.loc[nan_labels, col] = np.nan

                if return_stats and n_clipped > 0:
                    stats[col] = {
                        "n_clipped": n_clipped,
                        "original_dtype": str(original_dtype),
                        "window_used": window,
                        "method_used": method.lower(),
                        "factor_used": factor,
                    }

        except Exception as e:
            logger.warning("Vectorized clip_outliers failed: %s. Falling back to per-column processing.", e)
            use_vectorized = False

    # Per-column path
    if not use_vectorized or len(numeric_columns) == 1:
        for col in numeric_columns:
            series = target_df[col]
            original_dtype = series.dtype
            series_copy = series.copy()

            nan_mask = series_copy.isna()
            if nan_mask.any():
                series_filled = series_copy.ffill().bfill()
            else:
                series_filled = series_copy

            if series_filled.isna().all():
                logger.debug("Column '%s' remains all NaN after fill, skipping", col)
                continue

            if method.lower() == "mad":
                rolling_median = series_filled.rolling(window=window, center=True, min_periods=1).median()
                abs_diff = (series_filled - rolling_median).abs()
                rolling_mad = abs_diff.rolling(window=window, center=True, min_periods=1).median()
                MAD_TO_STD = 1.4826
                threshold = factor * MAD_TO_STD * rolling_mad
                lower_bound = rolling_median - threshold
                upper_bound = rolling_median + threshold
            else:
                rolling_q1 = series_filled.rolling(window=window, center=True, min_periods=1).quantile(0.25)
                rolling_q3 = series_filled.rolling(window=window, center=True, min_periods=1).quantile(0.75)
                rolling_iqr = rolling_q3 - rolling_q1
                lower_bound = rolling_q1 - factor * rolling_iqr
                upper_bound = rolling_q3 + factor * rolling_iqr

            clipped_series = series_filled.clip(lower=lower_bound, upper=upper_bound)
            clipped_mask = (series_filled < lower_bound) | (series_filled > upper_bound)
            n_clipped = int(clipped_mask.sum())

            if n_clipped > 0:
                logger.info("Clipped %d outliers in column '%s' using %s method", n_clipped, col, method.upper())

            if pd.api.types.is_integer_dtype(original_dtype) and not isinstance(original_dtype, np.dtype):
                # pandas nullable integer
                target_df[col] = pd.Series(np.rint(clipped_series), index=target_df.index).astype(original_dtype)
            else:
                if pd.api.types.is_integer_dtype(original_dtype):
                    target_df[col] = np.rint(clipped_series).astype(original_dtype)
                else:
                    target_df[col] = clipped_series.astype(original_dtype)

            if nan_mask.any():
                nan_labels = target_df.index[nan_mask.to_numpy(dtype=bool)]
                target_df.loc[nan_labels, col] = np.nan

            if return_stats and n_clipped > 0:
                stats[col] = {
                    "n_clipped": n_clipped,
                    "original_dtype": str(original_dtype),
                    "window_used": window,
                    "method_used": method.lower(),
                    "factor_used": factor,
                }

    if inplace:
        return (None, stats) if return_stats else None
    else:
        return (target_df, stats) if return_stats else target_df
