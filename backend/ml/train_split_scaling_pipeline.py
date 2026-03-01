"""
Item 1 Phase B+D: Train-Split Scaling Pipeline

This module implements the leakage-free preprocessing flow:
1. preprocess_without_scaling() - no scaler fitting
2. temporal_split() - split by time
3. apply_scaling() - fit on train, transform all

Feature flag: ENABLE_TRAIN_SPLIT_SCALING=true
"""

import logging
import os
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


def is_train_split_scaling_enabled() -> bool:
    """Check if train-split scaling feature is enabled."""
    # Beta 6 policy: leakage guard is mandatory and fail-closed.
    return os.getenv("ENABLE_TRAIN_SPLIT_SCALING", "true").lower() in ("true", "1", "yes", "on")


def _attach_activity_encoded_with_train_encoder(
    platform,
    room_name: str,
    df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """
    Encode activity labels for non-train splits using the train-fitted label encoder.

    Fail-closed behavior:
    - Missing activity column -> error
    - Missing room label encoder -> error
    - Unknown labels not present in train classes -> error
    """
    if df is None:
        return None
    if df.empty:
        out = df.copy()
        if "activity_encoded" not in out.columns:
            out["activity_encoded"] = pd.Series(dtype="int32")
        return out

    out = df.copy()

    if "activity" not in out.columns:
        raise ValueError(
            f"[{room_name}] Missing 'activity' column while deriving activity_encoded for non-train split"
        )

    label_encoder = getattr(platform, "label_encoders", {}).get(room_name)
    if label_encoder is None or not hasattr(label_encoder, "classes_"):
        raise ValueError(
            f"[{room_name}] Missing train label encoder; cannot encode non-train split labels"
        )

    classes = [str(item).strip().lower() for item in label_encoder.classes_]
    known = set(classes)
    labels = out["activity"].astype(str).str.strip().str.lower()
    unknown = sorted({label for label in labels.unique().tolist() if label not in known})
    if unknown:
        raise ValueError(
            f"[{room_name}] Non-train split contains labels absent from train encoder: {unknown}"
        )

    encoded = label_encoder.transform(labels).astype(np.int32, copy=False)

    # If an encoded column already exists, enforce consistency with canonical
    # train-encoder mapping instead of silently trusting pre-existing values.
    if "activity_encoded" in out.columns:
        prior = pd.to_numeric(out["activity_encoded"], errors="coerce")
        invalid_prior = int(prior.isna().sum())
        if invalid_prior > 0:
            raise ValueError(
                f"[{room_name}] Existing activity_encoded has {invalid_prior} invalid rows in non-train split"
            )
        mismatch = int((prior.astype(np.int32, copy=False).to_numpy() != encoded).sum())
        if mismatch > 0:
            raise ValueError(
                f"[{room_name}] Existing activity_encoded mismatches canonical encoding for {mismatch} rows"
            )

    out["activity_encoded"] = encoded
    if out["activity_encoded"].isna().any():
        raise ValueError(f"[{room_name}] activity_encoded contains NaN after non-train split encoding")
    return out


def temporal_split_dataframe(
    df: pd.DataFrame,
    validation_split: float = 0.2,
    calibration_fraction: float = 0.3,
    min_calibration_samples: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Split DataFrame temporally into train/validation/calibration.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame with 'timestamp' column (unscaled)
    validation_split : float
        Fraction of data to use for validation + calibration
    calibration_fraction : float
        Fraction of validation data to use for calibration
    min_calibration_samples : int
        Minimum samples required for separate calibration split
        
    Returns:
    --------
    (train_df, val_df, calib_df, metadata) : Tuple
        train_df: Training split (to fit scaler on)
        val_df: Validation split
        calib_df: Calibration split (may be None if insufficient data)
        metadata: Dict with split timestamps and sample counts
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column for temporal split")
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    n_total = len(df)
    if n_total < 10:
        raise ValueError(f"Insufficient data for temporal split: {n_total} samples")
    
    # Calculate split indices
    n_holdout = int(n_total * validation_split)
    n_holdout = max(1, min(n_total - 1, n_holdout))  # Ensure at least 1 sample for train
    n_train = n_total - n_holdout
    
    # Ensure minimum training samples
    if n_train < 10:
        logger.warning(f"Very few training samples ({n_train}), adjusting split")
        n_train = max(10, int(n_total * 0.8))
        n_holdout = n_total - n_train
    
    # Temporal split: train on past, validate on future
    train_df = df.iloc[:n_train].copy()
    holdout_df = df.iloc[n_train:].copy()
    
    # Split holdout into validation and calibration
    n_calib = int(len(holdout_df) * calibration_fraction)
    n_calib = max(0, min(len(holdout_df) - min_calibration_samples, n_calib))
    
    if n_calib >= min_calibration_samples:
        val_df = holdout_df.iloc[:-n_calib].copy()
        calib_df = holdout_df.iloc[-n_calib:].copy()
    else:
        val_df = holdout_df.copy()
        calib_df = None
    
    # Build metadata
    metadata = {
        "total_samples": n_total,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "calib_samples": len(calib_df) if calib_df is not None else 0,
        "train_start_ts": train_df['timestamp'].min().isoformat(),
        "train_end_ts": train_df['timestamp'].max().isoformat(),
        "val_start_ts": val_df['timestamp'].min().isoformat(),
        "val_end_ts": val_df['timestamp'].max().isoformat(),
        "calib_start_ts": calib_df['timestamp'].min().isoformat() if calib_df is not None else None,
        "calib_end_ts": calib_df['timestamp'].max().isoformat() if calib_df is not None else None,
    }
    
    logger.info(
        f"Temporal split: Train={len(train_df)} ({metadata['train_start_ts']}..{metadata['train_end_ts']}), "
        f"Val={len(val_df)}, Calib={len(calib_df) if calib_df is not None else 0}"
    )
    
    return train_df, val_df, calib_df, metadata


def prepare_training_data_with_train_split_scaling(
    platform,
    room_name: str,
    raw_df: pd.DataFrame,
    validation_split: float = 0.2,
    calibration_fraction: float = 0.3,
    min_calibration_samples: int = 50,
    apply_denoising: bool = False,
    max_ffill_gap_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Complete leakage-free training data preparation.
    
    This is the main entry point for Item 1 Phase B+C.
    
    Parameters:
    -----------
    platform : ElderlyCarePlatform
        Platform instance
    room_name : str
        Room name
    raw_df : pd.DataFrame
        Raw input data
    validation_split : float
        Fraction for validation + calibration
    calibration_fraction : float
        Fraction of holdout for calibration
    min_calibration_samples : int
        Minimum samples for calibration split
    apply_denoising : bool
        Whether to apply denoising
    max_ffill_gap_seconds : float, optional
        Max forward-fill gap
        
    Returns:
    --------
    Dict with:
        - train_scaled: Scaled training DataFrame
        - val_scaled: Scaled validation DataFrame  
        - calib_scaled: Scaled calibration DataFrame (may be None)
        - split_metadata: Temporal split information
        - scaler_metadata: Scaler fit range information
    """
    # Phase A: Preprocess WITHOUT scaling
    logger.info(f"[{room_name}] Phase A: Preprocessing without scaling")
    preprocessed = platform.preprocess_without_scaling(
        df=raw_df,
        room_name=room_name,
        is_training=True,
        apply_denoising=apply_denoising,
        max_ffill_gap_seconds=max_ffill_gap_seconds,
    )
    
    if preprocessed.empty:
        raise ValueError(f"Preprocessing returned empty DataFrame for {room_name}")
    
    # Phase B: Temporal split
    logger.info(f"[{room_name}] Phase B: Temporal split")
    train_df, val_df, calib_df, split_metadata = temporal_split_dataframe(
        df=preprocessed,
        validation_split=validation_split,
        calibration_fraction=calibration_fraction,
        min_calibration_samples=min_calibration_samples,
    )
    
    # Phase C: Fit scaler on train, transform all
    logger.info(f"[{room_name}] Phase C: Fitting scaler on train split only")
    scaler_fit_range = {
        "fit_start_ts": split_metadata['train_start_ts'],
        "fit_end_ts": split_metadata['train_end_ts'],
        "fit_sample_count": split_metadata['train_samples'],
    }
    
    train_scaled = platform.apply_scaling(
        train_df, room_name, is_training=True, scaler_fit_range=scaler_fit_range
    )

    train_nulls = int(pd.to_numeric(train_scaled.get("activity_encoded"), errors="coerce").isna().sum()) \
        if "activity_encoded" in train_scaled.columns else len(train_scaled)
    if train_nulls > 0:
        raise ValueError(
            f"[{room_name}] Train split has {train_nulls} rows without valid activity_encoded after scaling"
        )
    
    val_scaled = platform.apply_scaling(
        val_df, room_name, is_training=False
    )
    val_scaled = _attach_activity_encoded_with_train_encoder(platform, room_name, val_scaled)
    
    calib_scaled = None
    if calib_df is not None and not calib_df.empty:
        calib_scaled = platform.apply_scaling(
            calib_df, room_name, is_training=False
        )
        calib_scaled = _attach_activity_encoded_with_train_encoder(platform, room_name, calib_scaled)

    return {
        "train_scaled": train_scaled,
        "val_scaled": val_scaled,
        "calib_scaled": calib_scaled,
        "split_metadata": split_metadata,
        "scaler_metadata": scaler_fit_range,
        "status": "success",
    }


def validate_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    calib_df: Optional[pd.DataFrame] = None,
    timestamp_col: str = 'timestamp'
) -> Tuple[bool, List[str]]:
    """
    Validate that there is no temporal leakage between splits.
    
    Returns:
    --------
    (passes, violations) : Tuple[bool, List[str]]
        passes: True if no leakage detected
        violations: List of violation descriptions
    """
    violations = []
    
    train_max = pd.to_datetime(train_df[timestamp_col]).max()
    val_min = pd.to_datetime(val_df[timestamp_col]).min()
    val_max = pd.to_datetime(val_df[timestamp_col]).max()
    
    # Check train/val overlap
    if train_max >= val_min:
        violations.append(
            f"TRAIN-VAL LEAKAGE: train_max ({train_max}) >= val_min ({val_min})"
        )
    
    # Check val/calib overlap if calib exists
    if calib_df is not None and not calib_df.empty:
        calib_min = pd.to_datetime(calib_df[timestamp_col]).min()
        
        if val_max >= calib_min:
            violations.append(
                f"VAL-CALIB LEAKAGE: val_max ({val_max}) >= calib_min ({calib_min})"
            )
    
    return len(violations) == 0, violations
