"""
Timeline Target Builder Module

Generates training/evaluation targets for segment boundaries and episode attributes.
Part of WS-1: Timeline Target Builder.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class BoundaryTargets:
    """Container for boundary detection targets."""
    
    start_flags: np.ndarray  # 1 at episode start positions
    end_flags: np.ndarray    # 1 at episode end positions
    timestamps: np.ndarray   # Corresponding timestamps
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_flags": self.start_flags.tolist(),
            "end_flags": self.end_flags.tolist(),
            "timestamps": [ts.isoformat() for ts in self.timestamps] if len(self.timestamps) > 0 and isinstance(self.timestamps[0], datetime) else self.timestamps.tolist(),
        }


@dataclass
class EpisodeAttributeTargets:
    """Container for episode attribute targets (day-level)."""
    
    # Episode counts per label
    episode_counts: Dict[str, int]
    
    # Total duration per label (minutes)
    episode_durations: Dict[str, float]
    
    # Episode boundaries (for debugging)
    episode_starts: List[datetime]
    episode_ends: List[datetime]
    episode_labels: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Normalize timestamps via pd.Timestamp to handle numpy.datetime64, strings, etc.
        def _to_iso(ts):
            try:
                return pd.Timestamp(ts).isoformat()
            except Exception:
                return str(ts)
        
        return {
            "episode_counts": self.episode_counts,
            "episode_durations": {k: round(v, 2) for k, v in self.episode_durations.items()},
            "episode_starts": [_to_iso(ts) for ts in self.episode_starts],
            "episode_ends": [_to_iso(ts) for ts in self.episode_ends],
            "episode_labels": self.episode_labels,
        }


@dataclass
class EventNativeTargets:
    """Container for offline event-native supervision targets."""

    boundary_targets: BoundaryTargets
    continuity_flags: np.ndarray
    duration_targets_minutes: np.ndarray
    timestamps: np.ndarray
    daily_duration_minutes: float
    daily_episode_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boundary_targets": self.boundary_targets.to_dict(),
            "continuity_flags": self.continuity_flags.tolist(),
            "duration_targets_minutes": [round(float(v), 4) for v in self.duration_targets_minutes.tolist()],
            "timestamps": [pd.Timestamp(ts).isoformat() for ts in self.timestamps],
            "daily_duration_minutes": round(float(self.daily_duration_minutes), 4),
            "daily_episode_count": int(self.daily_episode_count),
        }


def _resolve_window_duration_seconds(timestamps: np.ndarray) -> float:
    if len(timestamps) > 1:
        ts1 = pd.Timestamp(timestamps[1])
        ts0 = pd.Timestamp(timestamps[0])
        delta = (ts1 - ts0).total_seconds()
        if delta > 0:
            return float(delta)
    return 10.0


def build_boundary_targets(
    labels: np.ndarray,
    window_duration_seconds: float = 10.0,
    excluded_labels: frozenset = frozenset({'unoccupied', 'unknown'}),
) -> BoundaryTargets:
    """
    Build boundary targets from label sequence for ADL episode detection.
    
    Detects episode start and end boundaries by finding transitions
    between labels while excluding non-care labels (for example:
    unoccupied, unknown) as episodes.
    
    Args:
        labels: Array of string labels (one per window)
        window_duration_seconds: Duration of each window in seconds
        excluded_labels: Labels to exclude from boundary detection (default: unoccupied, unknown)
        
    Returns:
        BoundaryTargets with start/end flags
        
    Raises:
        ValueError: If labels contain fewer than 2 elements
        
    Example:
        >>> labels = np.array(['unoccupied', 'sleeping', 'sleeping', 'unoccupied'])
        >>> targets = build_boundary_targets(labels)
        >>> targets.start_flags
        array([0, 1, 0, 0])  # Start at index 1 (sleeping episode)
        >>> targets.end_flags
        array([0, 0, 1, 0])  # End at index 2 (sleeping episode ends)
    """
    if len(labels) < 2:
        raise ValueError("labels must contain at least 2 elements")
    
    n = len(labels)
    start_flags = np.zeros(n, dtype=np.int32)
    end_flags = np.zeros(n, dtype=np.int32)
    
    # First window: start if it's a care activity (not excluded)
    if labels[0] not in excluded_labels:
        start_flags[0] = 1
    
    # Detect episode boundaries
    for i in range(1, n):
        prev_label = labels[i - 1]
        curr_label = labels[i]
        
        prev_is_care = prev_label not in excluded_labels
        curr_is_care = curr_label not in excluded_labels
        
        # Case 1: Transition from excluded to care -> episode START
        if not prev_is_care and curr_is_care:
            start_flags[i] = 1
        
        # Case 2: Transition from care to excluded -> episode END
        elif prev_is_care and not curr_is_care:
            end_flags[i - 1] = 1
        
        # Case 3: Transition between different care activities -> boundary (end + start)
        elif prev_is_care and curr_is_care and curr_label != prev_label:
            end_flags[i - 1] = 1  # Previous episode ends
            start_flags[i] = 1    # New episode starts
        
        # Case 4: Transition between excluded labels -> no boundary
    
    # Last window: end if it's a care activity (not excluded)
    if labels[-1] not in excluded_labels:
        end_flags[-1] = 1
    
    # Generate timestamps (relative to 0 for training targets)
    timestamps = np.arange(n) * window_duration_seconds
    
    return BoundaryTargets(
        start_flags=start_flags,
        end_flags=end_flags,
        timestamps=timestamps,
    )


def build_episode_attribute_targets(
    timestamps: np.ndarray,
    labels: np.ndarray,
    room_name: str,
    min_episode_duration_seconds: float = 30.0,
    excluded_labels: frozenset = frozenset({'unoccupied', 'unknown'}),
) -> EpisodeAttributeTargets:
    """
    Build episode attribute targets from timestamps and labels.
    
    Aggregates episodes and computes day-level duration/count targets
    used by auxiliary heads during training. Excluded labels (unoccupied,
    unknown) are filtered out and do not contribute to targets.
    
    Args:
        timestamps: Array of datetime objects
        labels: Array of string labels
        room_name: Name of the room (for room-specific logic - reserved for future use)
        min_episode_duration_seconds: Minimum duration to count as episode
        excluded_labels: Labels to exclude from target computation (default: unoccupied, unknown)
        
    Returns:
        EpisodeAttributeTargets with aggregated metrics
        
    Raises:
        ValueError: If timestamps and labels have different lengths
        
    Note:
        This function is deterministic and has no temporal leakage.
        It only uses current and past windows to compute targets.
    """
    if len(timestamps) != len(labels):
        raise ValueError("timestamps and labels must have same length")
    
    if len(labels) == 0:
        return EpisodeAttributeTargets(
            episode_counts={},
            episode_durations={},
            episode_starts=[],
            episode_ends=[],
            episode_labels=[],
        )
    
    # Convert timestamps to datetime if needed
    if isinstance(timestamps[0], str):
        timestamps = pd.to_datetime(timestamps).to_numpy()
    
    episode_counts: Dict[str, int] = {}
    episode_durations: Dict[str, float] = {}
    episode_starts: List[datetime] = []
    episode_ends: List[datetime] = []
    episode_labels_list: List[str] = []
    
    # Calculate window duration, handling both datetime and string timestamps
    if len(timestamps) > 1:
        if hasattr(timestamps[1], 'total_seconds'):
            # Already a timedelta
            window_duration = timestamps[1].total_seconds()
        elif isinstance(timestamps[1], (datetime, pd.Timestamp)):
            window_duration = (timestamps[1] - timestamps[0]).total_seconds()
        else:
            # numpy datetime64 - convert to pandas Timestamp
            ts1 = pd.Timestamp(timestamps[1])
            ts0 = pd.Timestamp(timestamps[0])
            window_duration = (ts1 - ts0).total_seconds()
    else:
        window_duration = 10.0

    current_label = None
    current_start = None
    current_end = None

    def _finalize_episode(label, start_ts, end_ts) -> None:
        if label is None or start_ts is None or end_ts is None:
            return
        if isinstance(end_ts, (datetime, pd.Timestamp)) and isinstance(start_ts, (datetime, pd.Timestamp)):
            duration_seconds = (end_ts - start_ts).total_seconds() + window_duration
        else:
            duration_seconds = (pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).total_seconds() + window_duration
        if duration_seconds < min_episode_duration_seconds:
            return
        episode_counts[label] = episode_counts.get(label, 0) + 1
        duration_minutes = duration_seconds / 60.0
        episode_durations[label] = episode_durations.get(label, 0.0) + duration_minutes
        episode_starts.append(start_ts)
        episode_ends.append(end_ts)
        episode_labels_list.append(label)

    for i in range(len(labels)):
        label = labels[i]
        ts = timestamps[i]

        if label in excluded_labels:
            # Excluded labels terminate current care episode (do not bridge across gaps).
            _finalize_episode(current_label, current_start, current_end)
            current_label = None
            current_start = None
            current_end = None
            continue

        if current_label is None:
            current_label = label
            current_start = ts
            current_end = ts
            continue

        if label != current_label:
            _finalize_episode(current_label, current_start, current_end)
            current_label = label
            current_start = ts
            current_end = ts
        else:
            current_end = ts

    _finalize_episode(current_label, current_start, current_end)
    
    return EpisodeAttributeTargets(
        episode_counts=episode_counts,
        episode_durations=episode_durations,
        episode_starts=episode_starts,
        episode_ends=episode_ends,
        episode_labels=episode_labels_list,
    )


def build_event_native_targets(
    timestamps: np.ndarray,
    labels: np.ndarray,
    room_name: str,
    min_episode_duration_seconds: float = 30.0,
    excluded_labels: frozenset = frozenset({'unoccupied', 'unknown'}),
) -> EventNativeTargets:
    """
    Build offline event-native supervision targets for Beta 6.2.
    """
    if len(timestamps) != len(labels):
        raise ValueError("timestamps and labels must have same length")

    if len(labels) == 0:
        empty_boundary = BoundaryTargets(
            start_flags=np.array([], dtype=np.int32),
            end_flags=np.array([], dtype=np.int32),
            timestamps=np.array([], dtype=object),
        )
        return EventNativeTargets(
            boundary_targets=empty_boundary,
            continuity_flags=np.array([], dtype=np.int32),
            duration_targets_minutes=np.array([], dtype=np.float32),
            timestamps=np.array([], dtype=object),
            daily_duration_minutes=0.0,
            daily_episode_count=0,
        )

    normalized_timestamps = pd.to_datetime(timestamps).to_numpy()
    window_duration_seconds = _resolve_window_duration_seconds(normalized_timestamps)
    if len(labels) >= 2:
        boundary_targets = build_boundary_targets(
            labels,
            window_duration_seconds=window_duration_seconds,
            excluded_labels=excluded_labels,
        )
    else:
        is_care = labels[0] not in excluded_labels
        boundary_targets = BoundaryTargets(
            start_flags=np.array([1 if is_care else 0], dtype=np.int32),
            end_flags=np.array([1 if is_care else 0], dtype=np.int32),
            timestamps=np.array([0.0], dtype=np.float32),
        )

    continuity_flags = np.zeros(len(labels), dtype=np.int32)
    duration_targets_minutes = np.zeros(len(labels), dtype=np.float32)
    daily_duration_minutes = 0.0
    daily_episode_count = 0

    current_label: Optional[str] = None
    current_start_idx: Optional[int] = None
    current_end_idx: Optional[int] = None

    def _finalize_episode(label: Optional[str], start_idx: Optional[int], end_idx: Optional[int]) -> None:
        nonlocal daily_duration_minutes, daily_episode_count
        if label is None or start_idx is None or end_idx is None:
            return
        start_ts = pd.Timestamp(normalized_timestamps[start_idx])
        end_ts = pd.Timestamp(normalized_timestamps[end_idx])
        duration_seconds = (end_ts - start_ts).total_seconds() + window_duration_seconds
        if duration_seconds < float(min_episode_duration_seconds):
            return
        duration_minutes = float(duration_seconds / 60.0)
        duration_targets_minutes[start_idx : end_idx + 1] = duration_minutes
        if end_idx > start_idx:
            continuity_flags[start_idx:end_idx] = 1
        daily_duration_minutes += duration_minutes
        daily_episode_count += 1

    for idx, label in enumerate(labels):
        label_token = str(label)
        if label_token in excluded_labels:
            _finalize_episode(current_label, current_start_idx, current_end_idx)
            current_label = None
            current_start_idx = None
            current_end_idx = None
            continue
        if current_label is None:
            current_label = label_token
            current_start_idx = idx
            current_end_idx = idx
            continue
        if label_token != current_label:
            _finalize_episode(current_label, current_start_idx, current_end_idx)
            current_label = label_token
            current_start_idx = idx
            current_end_idx = idx
        else:
            current_end_idx = idx

    _finalize_episode(current_label, current_start_idx, current_end_idx)

    return EventNativeTargets(
        boundary_targets=boundary_targets,
        continuity_flags=continuity_flags,
        duration_targets_minutes=duration_targets_minutes,
        timestamps=normalized_timestamps,
        daily_duration_minutes=float(daily_duration_minutes),
        daily_episode_count=int(daily_episode_count),
    )


def build_boundary_targets_from_episodes(
    episodes: List[Dict[str, Any]],
    total_windows: int,
    window_duration_seconds: float = 10.0,
) -> BoundaryTargets:
    """
    Build boundary targets from episode list.
    
    Alternative entry point when episodes are already compiled.
    
    Args:
        episodes: List of episode dictionaries with 'start_idx', 'end_idx', 'label'
        total_windows: Total number of windows in the sequence
        window_duration_seconds: Duration of each window
        
    Returns:
        BoundaryTargets with start/end flags
    """
    start_flags = np.zeros(total_windows, dtype=np.int32)
    end_flags = np.zeros(total_windows, dtype=np.int32)
    
    for ep in episodes:
        start_idx = ep.get('start_idx', 0)
        end_idx = ep.get('end_idx', 0)
        
        if 0 <= start_idx < total_windows:
            start_flags[start_idx] = 1
        if 0 <= end_idx < total_windows:
            end_flags[end_idx] = 1
    
    timestamps = np.arange(total_windows) * window_duration_seconds
    
    return BoundaryTargets(
        start_flags=start_flags,
        end_flags=end_flags,
        timestamps=timestamps,
    )


def validate_no_temporal_leakage(
    labels: np.ndarray,
    targets: BoundaryTargets,
    excluded_labels: frozenset = frozenset({'unoccupied', 'unknown'}),
) -> bool:
    """
    Validate that boundary targets have no temporal leakage.
    
    A boundary target at position i should only depend on:
    - labels[i-1] and labels[i] for transitions
    - Care activity episodes (not involving excluded labels)
    
    This validation is independent of the build function to catch logic errors.
    
    Args:
        labels: Original label sequence
        targets: Computed boundary targets
        excluded_labels: Labels excluded from boundary detection
        
    Returns:
        True if no leakage detected
    """
    if len(labels) != len(targets.start_flags) or len(labels) != len(targets.end_flags):
        return False
    
    n = len(labels)
    
    # Handle empty labels - no leakage possible with no data
    if n == 0:
        return True
    
    # Independent validation: expected flags use only local transition rules.
    expected_start = np.zeros(n, dtype=np.int32)
    expected_end = np.zeros(n, dtype=np.int32)
    
    # First window: start if it's a care activity
    if labels[0] not in excluded_labels:
        expected_start[0] = 1
    
    # Detect episode boundaries
    for i in range(1, n):
        prev_label = labels[i - 1]
        curr_label = labels[i]
        
        prev_is_care = prev_label not in excluded_labels
        curr_is_care = curr_label not in excluded_labels
        
        # Transition from excluded to care -> episode START
        if not prev_is_care and curr_is_care:
            expected_start[i] = 1
        
        # Transition from care to excluded -> episode END
        elif prev_is_care and not curr_is_care:
            expected_end[i - 1] = 1
        
        # Transition between different care activities -> boundary (end + start)
        elif prev_is_care and curr_is_care and curr_label != prev_label:
            expected_end[i - 1] = 1
            expected_start[i] = 1
    
    # Last window: end if it's a care activity
    if labels[-1] not in excluded_labels:
        expected_end[-1] = 1
    
    # Compare with actual targets
    if not np.array_equal(targets.start_flags, expected_start):
        return False
    if not np.array_equal(targets.end_flags, expected_end):
        return False

    # Causality sanity check: start at i cannot depend on labels after i,
    # end at i cannot depend on labels after i+1.
    for i in range(1, n):
        prev_label = labels[i - 1]
        curr_label = labels[i]
        expected_start_i = int(
            (prev_label in excluded_labels and curr_label not in excluded_labels)
            or (
                prev_label not in excluded_labels
                and curr_label not in excluded_labels
                and curr_label != prev_label
            )
        )
        if int(targets.start_flags[i]) != expected_start_i:
            return False
    for i in range(0, n - 1):
        curr_label = labels[i]
        next_label = labels[i + 1]
        expected_end_i = int(
            (curr_label not in excluded_labels and next_label in excluded_labels)
            or (
                curr_label not in excluded_labels
                and next_label not in excluded_labels
                and curr_label != next_label
            )
        )
        if int(targets.end_flags[i]) != expected_end_i:
            return False
    
    return True


def build_multi_room_targets(
    room_data: Dict[str, Dict[str, np.ndarray]],
    window_duration_seconds: float = 10.0,
) -> Dict[str, BoundaryTargets]:
    """
    Build boundary targets for multiple rooms.
    
    Args:
        room_data: Dict mapping room names to {'timestamps': ..., 'labels': ...}
        window_duration_seconds: Duration of each window
        
    Returns:
        Dict mapping room names to BoundaryTargets
    """
    targets = {}
    
    for room_name, data in room_data.items():
        labels = data.get('labels', np.array([]))
        if len(labels) > 0:
            targets[room_name] = build_boundary_targets(labels, window_duration_seconds)
    
    return targets


def targets_to_dataframe(targets: BoundaryTargets) -> pd.DataFrame:
    """Convert BoundaryTargets to DataFrame for inspection."""
    return pd.DataFrame({
        'timestamp': targets.timestamps,
        'start_flag': targets.start_flags,
        'end_flag': targets.end_flags,
        'is_boundary': targets.start_flags | targets.end_flags,
    })
