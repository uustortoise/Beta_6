"""
Timeline Pipeline Integration Module

Shadow mode integration for timeline-aware training and inference.

Part of WS-5: Pipeline/Unified Integration (Shadow)

Usage:
    # Check if timeline mode is enabled
    if is_timeline_multitask_enabled():
        # Use timeline heads and decoder
        outputs = timeline_forward_pass(...)
    else:
        # Use existing production path (zero behavior change)
        outputs = standard_forward_pass(...)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# Optional imports - may not be available in all environments
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# Feature flag constants
ENABLE_TIMELINE_MULTITASK = "ENABLE_TIMELINE_MULTITASK"
ENABLE_TIMELINE_DECODER_V2 = "ENABLE_TIMELINE_DECODER_V2"
ENABLE_TIMELINE_SHADOW_MODE = "ENABLE_TIMELINE_SHADOW_MODE"
# Backward compatibility aliases (legacy names still used by some callers/tests).
TIMELINE_DECODER_V2 = "TIMELINE_DECODER_V2"
TIMELINE_SHADOW_MODE = "TIMELINE_SHADOW_MODE"


def is_timeline_multitask_enabled() -> bool:
    """
    Check if timeline multi-task is enabled.
    
    Returns:
        True if ENABLE_TIMELINE_MULTITASK environment variable is set to "true" or "1"
    """
    value = os.environ.get(ENABLE_TIMELINE_MULTITASK, "false").lower()
    return value in ("true", "1", "yes", "on")


def is_timeline_decoder_v2_enabled() -> bool:
    """
    Check if timeline decoder v2 is enabled.
    
    Returns:
        True if ENABLE_TIMELINE_DECODER_V2 environment variable is set to "true" or "1"
    """
    value = os.environ.get(ENABLE_TIMELINE_DECODER_V2, os.environ.get(TIMELINE_DECODER_V2, "false")).lower()
    return value in ("true", "1", "yes", "on")


def is_timeline_shadow_mode() -> bool:
    """
    Check if timeline shadow mode is enabled.
    
    Shadow mode generates timeline artifacts without affecting production decisions.
    
    Returns:
        True if ENABLE_TIMELINE_SHADOW_MODE environment variable is set to "true" or "1"
    """
    value = os.environ.get(ENABLE_TIMELINE_SHADOW_MODE, os.environ.get(TIMELINE_SHADOW_MODE, "false")).lower()
    return value in ("true", "1", "yes", "on")


@dataclass
class TimelineShadowArtifacts:
    """Container for timeline shadow artifacts."""
    
    windows_df: Optional[pd.DataFrame] = None
    episodes_df: Optional[pd.DataFrame] = None
    qc_report: Optional[Dict[str, Any]] = None
    
    def save(self, output_dir: Path) -> Dict[str, Path]:
        """
        Save artifacts to directory.
        
        Args:
            output_dir: Directory to save artifacts
            
        Returns:
            Dictionary mapping artifact names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        if self.windows_df is not None:
            windows_path = output_dir / "timeline_windows.parquet"
            self.windows_df.to_parquet(windows_path)
            paths['timeline_windows'] = windows_path
        
        if self.episodes_df is not None:
            episodes_path = output_dir / "timeline_episodes.parquet"
            self.episodes_df.to_parquet(episodes_path)
            paths['timeline_episodes'] = episodes_path
        
        if self.qc_report is not None:
            import json
            qc_path = output_dir / "timeline_qc.json"
            qc_path.write_text(json.dumps(self.qc_report, indent=2))
            paths['timeline_qc'] = qc_path
        
        return paths


class TimelineShadowPipeline:
    """
    Shadow pipeline for timeline-aware processing.
    
    This class provides timeline-aware training and inference paths
    that run alongside (but don't interfere with) the production pipeline.
    """
    
    def __init__(
        self,
        adl_registry,
        timeline_heads=None,
        timeline_decoder=None,
        calibrator=None,
    ):
        """
        Initialize shadow pipeline.
        
        Args:
            adl_registry: ADL event registry
            timeline_heads: Optional pre-built timeline heads
            timeline_decoder: Optional pre-built timeline decoder
            calibrator: Optional calibrator
        """
        self.adl_registry = adl_registry
        self.timeline_heads = timeline_heads
        self.timeline_decoder = timeline_decoder
        self.calibrator = calibrator
        
        self._artifacts = TimelineShadowArtifacts()
    
    def process_split(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
        split_id: str,
    ) -> TimelineShadowArtifacts:
        """
        Process a train/val split in shadow mode.
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            split_id: Split identifier (e.g., "4->5")
            
        Returns:
            TimelineShadowArtifacts with results
        """
        artifacts = TimelineShadowArtifacts()
        artifacts.qc_report = {
            'split_id': split_id,
            'train_windows': int(len(train_data.get('labels', []))) if isinstance(train_data, dict) else 0,
        }

        labels = np.asarray(train_data.get('labels', []), dtype=object) if isinstance(train_data, dict) else np.array([], dtype=object)
        timestamps = np.asarray(train_data.get('timestamps', []), dtype=object) if isinstance(train_data, dict) else np.array([], dtype=object)

        # 1. Generate timeline targets from labels (if available)
        if labels.size > 0:
            from ml.timeline_targets import build_boundary_targets

            targets = build_boundary_targets(labels)
            artifacts.qc_report['boundary_starts'] = int(np.sum(targets.start_flags))
            artifacts.qc_report['boundary_ends'] = int(np.sum(targets.end_flags))

            if timestamps.size == 0:
                timestamps = np.arange(labels.size, dtype=np.int64)
            elif timestamps.size != labels.size:
                # Keep this fail-closed in QC while still producing deterministic artifacts.
                artifacts.qc_report['timestamp_mismatch'] = {
                    'labels': int(labels.size),
                    'timestamps': int(timestamps.size),
                }
                timestamps = np.arange(labels.size, dtype=np.int64)

            windows_df = pd.DataFrame(
                {
                    "index": np.arange(labels.size, dtype=np.int64),
                    "timestamp": timestamps,
                    "label": labels.astype(str),
                    "start_flag": targets.start_flags.astype(np.int32),
                    "end_flag": targets.end_flags.astype(np.int32),
                }
            )
            windows_df["is_care_label"] = ~windows_df["label"].isin({"unoccupied", "unknown"})
            artifacts.windows_df = windows_df

            artifacts.episodes_df = self._labels_to_episodes_df(labels=labels, timestamps=timestamps)
            artifacts.qc_report['episode_count'] = int(len(artifacts.episodes_df))
        else:
            artifacts.qc_report['labels_available'] = False

        # 2. If model available, record safe forward-pass readiness.
        if self.timeline_heads is not None and TF_AVAILABLE:
            artifacts.qc_report['model_forward_pass'] = 'ready'
        elif self.timeline_heads is not None:
            artifacts.qc_report['model_forward_pass'] = 'skipped_tf_unavailable'

        # 3. If decoder available, record decoder readiness.
        if self.timeline_decoder is not None:
            artifacts.qc_report['decoder_v2'] = 'ready'
        
        self._artifacts = artifacts
        return artifacts
    
    def save_artifacts(self, output_dir: Path) -> Dict[str, Path]:
        """Save artifacts to output directory."""
        return self._artifacts.save(output_dir)

    @staticmethod
    def _labels_to_episodes_df(labels: np.ndarray, timestamps: np.ndarray) -> pd.DataFrame:
        """
        Convert contiguous care-label runs into episode rows.
        Excludes unoccupied/unknown to keep timeline artifacts care-focused.
        """
        episodes: List[Dict[str, Any]] = []
        excluded = {"unoccupied", "unknown"}
        n = int(len(labels))
        if n == 0:
            return pd.DataFrame(columns=["start_idx", "end_idx", "start_time", "end_time", "label", "window_count"])

        start_idx: Optional[int] = None
        current_label: Optional[str] = None
        for i in range(n):
            label = str(labels[i])
            is_care = label not in excluded
            if not is_care:
                if current_label is not None and start_idx is not None:
                    end_idx = i - 1
                    episodes.append(
                        {
                            "start_idx": int(start_idx),
                            "end_idx": int(end_idx),
                            "start_time": timestamps[start_idx],
                            "end_time": timestamps[end_idx],
                            "label": str(current_label),
                            "window_count": int(end_idx - start_idx + 1),
                        }
                    )
                start_idx = None
                current_label = None
                continue

            if current_label is None:
                current_label = label
                start_idx = i
                continue

            if label != current_label:
                end_idx = i - 1
                episodes.append(
                    {
                        "start_idx": int(start_idx),
                        "end_idx": int(end_idx),
                        "start_time": timestamps[start_idx],
                        "end_time": timestamps[end_idx],
                        "label": str(current_label),
                        "window_count": int(end_idx - start_idx + 1),
                    }
                )
                current_label = label
                start_idx = i

        if current_label is not None and start_idx is not None:
            end_idx = n - 1
            episodes.append(
                {
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "start_time": timestamps[start_idx],
                    "end_time": timestamps[end_idx],
                    "label": str(current_label),
                    "window_count": int(end_idx - start_idx + 1),
                }
            )

        return pd.DataFrame(episodes)


def get_timeline_feature_flags() -> Dict[str, bool]:
    """
    Get all timeline feature flags.
    
    Returns:
        Dictionary mapping flag names to boolean values
    """
    multitask_enabled = is_timeline_multitask_enabled()
    decoder_enabled = is_timeline_decoder_v2_enabled()
    shadow_enabled = is_timeline_shadow_mode()
    return {
        "ENABLE_TIMELINE_MULTITASK": multitask_enabled,
        "ENABLE_TIMELINE_DECODER_V2": decoder_enabled,
        "ENABLE_TIMELINE_SHADOW_MODE": shadow_enabled,
        # Legacy keys kept for compatibility with older integration code/tests.
        "TIMELINE_DECODER_V2": decoder_enabled,
        "TIMELINE_SHADOW_MODE": shadow_enabled,
    }


def validate_timeline_safety() -> List[str]:
    """
    Validate timeline feature flag safety.
    
    Ensures that:
    - When flags are OFF, behavior is unchanged
    - Shadow mode is recommended before full enablement
    
    Returns:
        List of safety warnings (empty if safe)
    """
    warnings = []
    
    flags = get_timeline_feature_flags()
    
    # Check if multitask is enabled without shadow mode
    if flags['ENABLE_TIMELINE_MULTITASK'] and not flags['ENABLE_TIMELINE_SHADOW_MODE']:
        warnings.append(
            "ENABLE_TIMELINE_MULTITASK is ON but ENABLE_TIMELINE_SHADOW_MODE/TIMELINE_SHADOW_MODE is OFF. "
            "Recommend shadow mode for initial validation."
        )
    
    # Check if decoder v2 is enabled without shadow mode
    if flags['ENABLE_TIMELINE_DECODER_V2'] and not flags['ENABLE_TIMELINE_SHADOW_MODE']:
        warnings.append(
            "ENABLE_TIMELINE_DECODER_V2/TIMELINE_DECODER_V2 is ON but ENABLE_TIMELINE_SHADOW_MODE/TIMELINE_SHADOW_MODE is OFF. "
            "Recommend shadow mode for initial validation."
        )
    
    return warnings


def create_timeline_qc_report(
    predictions_df: Optional[pd.DataFrame] = None,
    episodes_df: Optional[pd.DataFrame] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create quality control report for timeline artifacts.
    
    Args:
        predictions_df: Window-level predictions DataFrame
        episodes_df: Episode-level DataFrame
        metrics: Timeline metrics dictionary
        
    Returns:
        QC report dictionary
    """
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'feature_flags': get_timeline_feature_flags(),
        'safety_warnings': validate_timeline_safety(),
    }
    
    if predictions_df is not None:
        report['predictions'] = {
            'num_windows': len(predictions_df),
            'columns': list(predictions_df.columns),
        }
    
    if episodes_df is not None:
        report['episodes'] = {
            'num_episodes': len(episodes_df),
            'columns': list(episodes_df.columns),
        }
    
    if metrics is not None:
        report['metrics'] = metrics
    
    return report


def should_run_timeline_path() -> bool:
    """
    Determine if timeline path should be run.
    
    This is the main entry point for deciding whether to use
    timeline-aware processing.
    
    Returns:
        True if timeline path should be run
    """
    # Run if multitask is enabled OR decoder v2 is enabled OR shadow mode is enabled
    return (
        is_timeline_multitask_enabled() or
        is_timeline_decoder_v2_enabled() or
        is_timeline_shadow_mode()
    )


def log_feature_flag_status():
    """Log current feature flag status for observability."""
    flags = get_timeline_feature_flags()
    warnings = validate_timeline_safety()
    
    # This would be replaced with proper logging in production
    status_msg = "Timeline Feature Flags: " + ", ".join(
        f"{k}={v}" for k, v in flags.items()
    )
    
    if warnings:
        status_msg += f" | Warnings: {'; '.join(warnings)}"
    
    return status_msg
