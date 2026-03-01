"""
Week 1 Integration: Foundation Items

This module integrates:
- Item 1 Phase A: preprocess_without_scaling()
- Item 2: CoverageContractGate
- Item 3: TRAINING_PROFILE support
- Item 7: Walk-Forward Robustness

Usage:
    from ml.week1_integration import prepare_training_data_with_leakage_free_scaling
    
    result = prepare_training_data_with_leakage_free_scaling(
        platform=platform,
        room_name="bedroom",
        df=raw_df,
        elder_id="resident_001",
        training_profile="production"
    )
"""

import logging
import os
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from ml.coverage_contract import CoverageContractGate, WalkForwardConfig
from ml.policy_config import load_policy_from_env, TrainingPolicy

logger = logging.getLogger(__name__)


def check_coverage_contract_for_room(
    room_name: str,
    df: pd.DataFrame,
    policy: TrainingPolicy
) -> Tuple[bool, Dict[str, Any]]:
    """
    Item 2: Check coverage contract before training.
    
    Returns:
    --------
    (passes, details) : Tuple[bool, Dict]
        passes: True if coverage contract passes
        details: Dict with observed_days, required_days, reason
    """
    if 'timestamp' not in df.columns:
        return False, {"reason": "missing_timestamp_column"}
    
    # Extract observed days
    timestamps = pd.to_datetime(df['timestamp'])
    observed_days = timestamps.dt.floor('D').nunique()
    
    # Get walk-forward config from policy
    wf_config = WalkForwardConfig(
        min_train_days=policy.data_viability.min_observed_days,
        valid_days=1,  # Standard validation window
        step_days=1
    )
    
    gate = CoverageContractGate(wf_config)
    result = gate.evaluate(room_name, observed_days)
    
    details = {
        "observed_days": result.observed_days,
        "required_days": result.required_days,
        "passes": result.passes,
        "reason": result.reason,
        "walk_forward_config": {
            "min_train_days": wf_config.min_train_days,
            "valid_days": wf_config.valid_days,
        }
    }
    
    return result.passes, details


def prepare_training_data_with_leakage_free_scaling(
    platform,
    room_name: str,
    df: pd.DataFrame,
    elder_id: str,
    training_profile: Optional[str] = None,
    apply_denoising: bool = False,
    max_ffill_gap_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Week 1 Integration: Complete leakage-free training data preparation.
    
    This function implements:
    1. Coverage contract check (Item 2)
    2. Profile-based policy loading (Item 3)
    3. Preprocessing WITHOUT scaling (Item 1 Phase A)
    4. Temporal split (to be implemented in training pipeline)
    5. Scaling on train split only (Item 1 Phase C)
    
    Parameters:
    -----------
    platform : ElderlyCarePlatform
        The platform instance
    room_name : str
        Room being processed
    df : pd.DataFrame
        Raw input data
    elder_id : str
        Resident ID for logging
    training_profile : str, optional
        "pilot" or "production" (overrides env var)
    apply_denoising : bool, default=False
        Whether to apply denoising
    max_ffill_gap_seconds : float, optional
        Max forward-fill gap
    
    Returns:
    --------
    Dict[str, Any]
        Result with status, preprocessed data, and metadata
    """
    # Set up profile if provided
    if training_profile:
        os.environ['TRAINING_PROFILE'] = training_profile
        logger.info(f"Using TRAINING_PROFILE={training_profile}")
    
    # Load policy (respects TRAINING_PROFILE env var)
    policy = load_policy_from_env()
    logger.info(f"Active training profile: {policy.get_profile_name()}")
    
    # Item 2: Check coverage contract
    coverage_passes, coverage_details = check_coverage_contract_for_room(
        room_name, df, policy
    )
    
    if not coverage_passes:
        logger.error(f"Coverage contract failed for {room_name}: {coverage_details['reason']}")
        return {
            "status": "coverage_contract_failed",
            "room": room_name,
            "elder_id": elder_id,
            "coverage_details": coverage_details,
            "preprocessed_df": None,
        }
    
    logger.info(f"Coverage contract passed for {room_name}: {coverage_details['observed_days']} days observed")
    
    # Item 1 Phase A: Preprocess WITHOUT scaling
    try:
        preprocessed_df = platform.preprocess_without_scaling(
            df=df,
            room_name=room_name,
            is_training=True,
            apply_denoising=apply_denoising,
            max_ffill_gap_seconds=max_ffill_gap_seconds,
        )
        
        if preprocessed_df.empty:
            return {
                "status": "preprocessing_failed",
                "room": room_name,
                "elder_id": elder_id,
                "reason": "empty_preprocessed_data",
                "preprocessed_df": None,
            }
        
        logger.info(f"Preprocessing complete for {room_name}: {len(preprocessed_df)} rows")
        
        return {
            "status": "preprocessing_complete",
            "room": room_name,
            "elder_id": elder_id,
            "coverage_details": coverage_details,
            "preprocessed_df": preprocessed_df,
            "policy_profile": policy.get_profile_name(),
            "policy_hash": policy_hash(policy),
        }
        
    except Exception as e:
        logger.error(f"Preprocessing failed for {room_name}: {e}")
        return {
            "status": "preprocessing_failed",
            "room": room_name,
            "elder_id": elder_id,
            "reason": str(e),
            "preprocessed_df": None,
        }


def policy_hash(policy: TrainingPolicy) -> str:
    """Generate stable hash for policy."""
    import hashlib
    import json
    
    payload = json.dumps(policy.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def apply_train_split_scaling(
    platform,
    room_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    calib_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Item 1 Phase C: Apply scaling fitted on train split only.
    
    Parameters:
    -----------
    platform : ElderlyCarePlatform
        Platform instance
    room_name : str
        Room name
    train_df : pd.DataFrame
        Training split (unscaled)
    val_df : pd.DataFrame
        Validation split (unscaled)
    calib_df : pd.DataFrame, optional
        Calibration split (unscaled)
    
    Returns:
    --------
    Dict with scaled dataframes and scaler metadata
    """
    # Record fit range metadata
    scaler_fit_range = {
        "fit_start_ts": train_df['timestamp'].min().isoformat() if 'timestamp' in train_df.columns else None,
        "fit_end_ts": train_df['timestamp'].max().isoformat() if 'timestamp' in train_df.columns else None,
        "fit_sample_count": len(train_df),
    }
    
    # Fit scaler on train and transform all splits
    train_scaled = platform.apply_scaling(
        train_df, room_name, is_training=True, scaler_fit_range=scaler_fit_range
    )
    
    val_scaled = platform.apply_scaling(
        val_df, room_name, is_training=False
    )
    
    calib_scaled = None
    if calib_df is not None and not calib_df.empty:
        calib_scaled = platform.apply_scaling(
            calib_df, room_name, is_training=False
        )
    
    return {
        "train_scaled": train_scaled,
        "val_scaled": val_scaled,
        "calib_scaled": calib_scaled,
        "scaler_metadata": scaler_fit_range,
    }


def check_walk_forward_status(evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Item 7: Check walk-forward evaluation status.
    
    Returns standardized status for downstream gating.
    """
    status = evaluation_result.get("status", "unknown")
    
    if status == "walk_forward_unavailable":
        summary = evaluation_result.get("summary", {})
        return {
            "walk_forward_available": False,
            "status": "walk_forward_unavailable",
            "reason": summary.get("reason", "unknown"),
            "promotable": False,  # Cannot promote without walk-forward validation
        }
    
    elif status == "completed":
        num_folds = evaluation_result.get("summary", {}).get("num_folds", 0)
        return {
            "walk_forward_available": True,
            "status": "completed",
            "num_folds": num_folds,
            "promotable": num_folds > 0,
        }
    
    else:
        return {
            "walk_forward_available": False,
            "status": status,
            "reason": "unexpected_evaluation_status",
            "promotable": False,
        }
