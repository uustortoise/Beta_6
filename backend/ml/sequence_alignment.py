"""
Item 5: Sequence-Label Alignment Contract Hardening

Ensures strict alignment between sequences and labels with explicit
contract enforcement and hard assertions.
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SequenceLabelAlignmentError(Exception):
    """Raised when sequence-label alignment fails."""
    pass


def _resolve_max_sequence_gap_seconds(platform, room_name: str) -> Optional[float]:
    interval_token = getattr(platform, "data_interval", None)
    try:
        from config import get_room_config

        room_interval = get_room_config().get_data_interval(room_name)
        if room_interval:
            interval_token = room_interval
    except Exception:
        pass

    if not interval_token:
        return None
    try:
        if isinstance(interval_token, (int, float)):
            seconds = float(interval_token)
        else:
            token_str = str(interval_token).strip()
            if token_str.replace(".", "", 1).isdigit():
                seconds = float(token_str)
            else:
                seconds = float(pd.to_timedelta(token_str).total_seconds())
    except Exception:
        return None
    if seconds <= 0:
        return None
    return max(seconds * 1.5, seconds + 1.0)


def create_labeled_sequences_strict(
    sensor_data: np.ndarray,
    labels: np.ndarray,
    seq_length: int,
    stride: int = 1,
    timestamps: Optional[np.ndarray] = None,
    max_gap_seconds: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences with strict label alignment.
    
    This is the explicit labeled sequence creator that replaces the
    brittle `labels[seq_length-1:]` fallback.
    
    Parameters:
    -----------
    sensor_data : np.ndarray
        Sensor data array of shape (n_samples, n_features)
    labels : np.ndarray
        Label array of shape (n_samples,)
    seq_length : int
        Length of each sequence
    stride : int
        Stride between consecutive sequences
    timestamps : np.ndarray, optional
        Timestamp array for tracking
        
    Returns:
    --------
    (X_seq, y_seq, seq_timestamps) : Tuple
        X_seq: Sequences of shape (n_sequences, seq_length, n_features)
        y_seq: Labels of shape (n_sequences,)
        seq_timestamps: Timestamps for each sequence (if provided)
        
    Raises:
    -------
    SequenceLabelAlignmentError
        If alignment cannot be guaranteed
    """
    n_samples = len(sensor_data)
    
    if n_samples != len(labels):
        raise SequenceLabelAlignmentError(
            f"Sample count mismatch: sensor_data={n_samples}, labels={len(labels)}"
        )
    
    if n_samples < seq_length:
        raise SequenceLabelAlignmentError(
            f"Insufficient samples for sequence length: {n_samples} < {seq_length}"
        )
    
    if timestamps is not None and len(timestamps) != n_samples:
        raise SequenceLabelAlignmentError(
            f"Timestamp count mismatch: timestamps={len(timestamps)}, samples={n_samples}"
        )
    
    # Calculate number of sequences
    n_sequences = (n_samples - seq_length) // stride + 1
    
    if n_sequences <= 0:
        raise SequenceLabelAlignmentError(
            f"No sequences can be created: n_samples={n_samples}, seq_length={seq_length}, stride={stride}"
        )
    
    # Create sequences
    X_seq = []
    y_seq = []
    seq_ts = []
    
    max_gap_ns = None
    if max_gap_seconds is not None:
        max_gap_ns = int(float(max_gap_seconds) * 1_000_000_000)

    skipped_gap_windows = 0
    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + seq_length

        if timestamps is not None and max_gap_ns is not None:
            window_ts = pd.to_datetime(timestamps[start_idx:end_idx], errors="coerce")
            if window_ts.isna().any():
                skipped_gap_windows += 1
                continue
            ts_ns = window_ts.view("int64")
            diffs = np.diff(ts_ns)
            if np.any(diffs <= 0) or np.any(diffs > max_gap_ns):
                skipped_gap_windows += 1
                continue
        
        # Extract sequence
        seq = sensor_data[start_idx:end_idx]
        # Label is from the LAST timestep of the sequence
        label = labels[end_idx - 1]
        
        X_seq.append(seq)
        y_seq.append(label)
        
        if timestamps is not None:
            seq_ts.append(timestamps[end_idx - 1])
    
    if not X_seq:
        raise SequenceLabelAlignmentError(
            f"No sequences remain after timestamp-gap filtering: "
            f"n_samples={n_samples}, seq_length={seq_length}, skipped_windows={skipped_gap_windows}"
        )

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    seq_ts = np.array(seq_ts) if timestamps is not None else np.arange(len(y_seq))
    
    return X_seq, y_seq, seq_ts


def assert_sequence_label_alignment(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    seq_timestamps: np.ndarray,
    context: str = "",
) -> None:
    """
    Hard assertion on sequence-label alignment.
    
    Parameters:
    -----------
    X_seq : np.ndarray
        Sequences array
    y_seq : np.ndarray
        Labels array
    seq_timestamps : np.ndarray
        Sequence timestamps
    context : str
        Context for error messages
        
    Raises:
    -------
    SequenceLabelAlignmentError
        If lengths don't match
    """
    errors = []
    
    if len(X_seq) != len(y_seq):
        errors.append(
            f"X_seq/y_seq length mismatch: {len(X_seq)} vs {len(y_seq)}"
        )
    
    if len(X_seq) != len(seq_timestamps):
        errors.append(
            f"X_seq/seq_timestamps length mismatch: {len(X_seq)} vs {len(seq_timestamps)}"
        )
    
    if len(y_seq) != len(seq_timestamps):
        errors.append(
            f"y_seq/seq_timestamps length mismatch: {len(y_seq)} vs {len(seq_timestamps)}"
        )
    
    if errors:
        prefix = f"[{context}] " if context else ""
        raise SequenceLabelAlignmentError(f"{prefix}Alignment errors: {'; '.join(errors)}")
    
    logger.debug(f"[{context}] Sequence-label alignment verified: {len(X_seq)} sequences")


def safe_create_sequences(
    platform,
    sensor_data: np.ndarray,
    labels: np.ndarray,
    seq_length: int,
    room_name: str,
    timestamps: Optional[np.ndarray] = None,
    strict: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Safe sequence creation with alignment guarantees.
    
    This replaces the legacy platform.create_sequences() for training paths.
    
    Parameters:
    -----------
    platform : ElderlyCarePlatform
        Platform instance (for legacy compatibility)
    sensor_data : np.ndarray
        Sensor data
    labels : np.ndarray
        Labels
    seq_length : int
        Sequence length
    room_name : str
        Room name for logging
    timestamps : np.ndarray, optional
        Timestamps
    strict : bool
        If True, use strict alignment; if False, fall back to legacy
        
    Returns:
    --------
    (X_seq, y_seq, seq_timestamps) : Tuple
        Aligned sequences and labels
        
    Raises:
    -------
    SequenceLabelAlignmentError
        If alignment fails and strict=True
    """
    if not strict:
        # Legacy path - use platform's create_sequences
        logger.warning(f"[{room_name}] Using legacy sequence creation (non-strict)")
        created = platform.create_sequences(sensor_data, seq_length)
        
        if isinstance(created, tuple):
            X_seq = np.asarray(created[0])
            y_seq = np.asarray(created[1]) if len(created) > 1 and created[1] is not None else labels[seq_length-1:]
        else:
            X_seq = np.asarray(created)
            y_seq = labels[seq_length-1:]
        
        # Generate timestamps if not provided
        if timestamps is not None:
            seq_ts = timestamps[seq_length-1:seq_length-1+len(X_seq)]
        else:
            seq_ts = np.arange(len(X_seq))
        
        return X_seq, y_seq, seq_ts
    
    # Strict path - use explicit alignment
    try:
        max_gap_seconds = _resolve_max_sequence_gap_seconds(platform, room_name)
        X_seq, y_seq, seq_ts = create_labeled_sequences_strict(
            sensor_data=sensor_data,
            labels=labels,
            seq_length=seq_length,
            stride=1,
            timestamps=timestamps,
            max_gap_seconds=max_gap_seconds,
        )
        
        # Hard assertion
        assert_sequence_label_alignment(X_seq, y_seq, seq_ts, context=room_name)
        
        logger.info(
            f"[{room_name}] Strict sequence creation: {len(X_seq)} sequences, "
            f"alignment verified"
        )
        
        return X_seq, y_seq, seq_ts
        
    except SequenceLabelAlignmentError as e:
        logger.error(f"[{room_name}] Sequence alignment failed: {e}")
        raise


class SequenceAlignmentValidator:
    """
    Validator for sequence-label alignment throughout the pipeline.
    """
    
    def __init__(self, room_name: str):
        self.room_name = room_name
        self.validation_points: List[Dict[str, Any]] = []
    
    def validate(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        seq_timestamps: np.ndarray,
        stage: str,
    ) -> bool:
        """
        Validate alignment at a specific pipeline stage.
        
        Parameters:
        -----------
        X_seq : np.ndarray
            Sequences
        y_seq : np.ndarray
            Labels
        seq_timestamps : np.ndarray
            Timestamps
        stage : str
            Pipeline stage name (e.g., "after_augmentation")
            
        Returns:
        --------
        bool
            True if validation passes
            
        Raises:
        -------
        SequenceLabelAlignmentError
            If validation fails
        """
        try:
            assert_sequence_label_alignment(X_seq, y_seq, seq_timestamps, context=self.room_name)
            
            self.validation_points.append({
                "stage": stage,
                "n_sequences": len(X_seq),
                "passed": True,
            })
            
            return True
            
        except SequenceLabelAlignmentError as e:
            self.validation_points.append({
                "stage": stage,
                "n_sequences": len(X_seq) if X_seq is not None else 0,
                "passed": False,
                "error": str(e),
            })
            raise
    
    def get_report(self) -> Dict[str, Any]:
        """Get validation report."""
        return {
            "room": self.room_name,
            "validation_points": self.validation_points,
            "all_passed": all(p.get("passed", False) for p in self.validation_points),
        }


def validate_stride_safety(
    original_length: int,
    seq_length: int,
    stride: int,
) -> Dict[str, Any]:
    """
    Validate that stride configuration maintains alignment.
    
    Parameters:
    -----------
    original_length : int
        Original number of samples
    seq_length : int
        Sequence length
    stride : int
        Stride value
        
    Returns:
    --------
    Dict with safety assessment
    """
    if stride < 1:
        return {
            "safe": False,
            "reason": f"Invalid stride: {stride} < 1",
            "expected_sequences": 0,
        }
    
    if original_length < seq_length:
        return {
            "safe": False,
            "reason": f"Insufficient data: {original_length} < {seq_length}",
            "expected_sequences": 0,
        }
    
    n_sequences = (original_length - seq_length) // stride + 1
    
    # Calculate data utilization
    last_sequence_end = (n_sequences - 1) * stride + seq_length
    utilization = last_sequence_end / original_length
    
    return {
        "safe": True,
        "expected_sequences": n_sequences,
        "utilization_ratio": utilization,
        "stride": stride,
        "seq_length": seq_length,
    }
