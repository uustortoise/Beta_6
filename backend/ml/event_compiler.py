"""
Event Compiler Module

Converts window-level predictions into canonical event episodes.
Part of PR-B1: Event Compiler + KPI/Gates.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class EpisodeStatus(Enum):
    """Status of an episode compilation."""
    ACTIVE = "active"
    MERGED = "merged"
    DELETED = "deleted"
    COMPLETED = "completed"


@dataclass
class Episode:
    """A canonical event episode."""
    
    event_label: str
    start_time: datetime
    end_time: datetime
    confidence: float
    window_count: int
    episode_id: str = field(default_factory=lambda: f"ep_{uuid.uuid4().hex[:12]}")
    room: str = "unknown"
    status: EpisodeStatus = EpisodeStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate episode data."""
        if self.start_time > self.end_time:
            raise ValueError("start_time must be <= end_time")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("confidence must be in [0, 1]")
    
    @property
    def duration_seconds(self) -> float:
        """Episode duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def duration_minutes(self) -> float:
        """Episode duration in minutes."""
        return self.duration_seconds / 60.0
    
    @property
    def duration_hours(self) -> float:
        """Episode duration in hours."""
        return self.duration_seconds / 3600.0
    
    def overlaps(self, other: "Episode") -> bool:
        """Check if this episode overlaps with another."""
        return not (self.end_time < other.start_time or self.start_time > other.end_time)
    
    def can_merge(self, other: "Episode", max_gap_seconds: float) -> bool:
        """Check if this episode can merge with another."""
        if self.event_label != other.event_label:
            return False
        if self.room != other.room:
            return False
        
        gap = other.start_time - self.end_time
        gap_seconds = gap.total_seconds()
        
        # Check if episodes are adjacent or have small gap
        return -1.0 <= gap_seconds <= max_gap_seconds
    
    def merge(self, other: "Episode") -> "Episode":
        """Merge this episode with another (same label)."""
        if not self.can_merge(other, float('inf')):
            raise ValueError("Cannot merge episodes with different labels or rooms")
        
        total_windows = self.window_count + other.window_count
        
        # Weighted average confidence
        merged_confidence = (
            (self.confidence * self.window_count + other.confidence * other.window_count)
            / total_windows
        ) if total_windows > 0 else 0.0
        
        merged_metadata = {**self.metadata, **other.metadata}
        merged_metadata["merged_episodes"] = [
            self.episode_id, other.episode_id
        ]
        
        return Episode(
            episode_id=self.episode_id,  # Keep first episode ID
            event_label=self.event_label,
            start_time=min(self.start_time, other.start_time),
            end_time=max(self.end_time, other.end_time),
            confidence=merged_confidence,
            window_count=total_windows,
            room=self.room,
            status=EpisodeStatus.ACTIVE,
            metadata=merged_metadata,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary."""
        return {
            "episode_id": self.episode_id,
            "event_label": self.event_label,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "duration_minutes": self.duration_minutes,
            "confidence": round(self.confidence, 4),
            "window_count": self.window_count,
            "room": self.room,
            "status": self.status.value,
            "metadata": self.metadata,
        }
    
    def to_feature_dict(self) -> Dict[str, Any]:
        """Convert episode to ML feature dictionary."""
        return {
            "event_label": self.event_label,
            "duration_minutes": self.duration_minutes,
            "hour_of_day_start": self.start_time.hour,
            "hour_of_day_end": self.end_time.hour,
            "day_of_week": self.start_time.weekday(),
            "confidence": self.confidence,
            "window_count": self.window_count,
        }


@dataclass
class EpisodeCompilerConfig:
    """Configuration for episode compilation."""
    
    # Duration filtering
    min_duration_seconds: float = 30.0  # Minimum episode duration
    max_duration_seconds: Optional[float] = None  # Maximum episode duration
    
    # Gap handling
    merge_gap_seconds: float = 30.0  # Merge episodes with gap <= this
    
    # Hysteresis parameters
    use_hysteresis: bool = True
    hysteresis_on_threshold: float = 0.60
    hysteresis_off_threshold: float = 0.40
    hysteresis_min_windows: int = 2  # Min consecutive windows for state change
    
    # Confidence weighting
    use_confidence_weighting: bool = True
    min_confidence_for_episode: float = 0.0
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.min_duration_seconds < 0:
            raise ValueError("min_duration_seconds must be >= 0")
        if self.max_duration_seconds is not None:
            if self.max_duration_seconds <= self.min_duration_seconds:
                raise ValueError("max_duration_seconds must be > min_duration_seconds")
        if self.hysteresis_on_threshold <= self.hysteresis_off_threshold:
            raise ValueError("hysteresis_on_threshold must be > hysteresis_off_threshold")
        if self.hysteresis_min_windows < 1:
            raise ValueError("hysteresis_min_windows must be >= 1")


class HysteresisState(Enum):
    """Internal hysteresis state."""
    OFF = "off"
    RISING = "rising"
    ON = "on"
    FALLING = "falling"


class EpisodeCompiler:
    """Compiles window predictions into episodes."""
    
    def __init__(self, config: Optional[EpisodeCompilerConfig] = None):
        self.config = config or EpisodeCompilerConfig()
        self.config.validate()
        
        # Hysteresis state tracking
        self._current_state: Optional[str] = None
        self._state_history: List[Tuple[str, int]] = []  # (label, count)
        self._pending_windows: List[Tuple[datetime, str, float]] = []
    
    def reset(self) -> None:
        """Reset compiler state."""
        self._current_state = None
        self._state_history = []
        self._pending_windows = []
    
    def compile(
        self,
        predictions: List[Tuple[datetime, str, float]],
    ) -> List[Episode]:
        """
        Compile predictions into episodes.
        
        Args:
            predictions: List of (timestamp, predicted_label, confidence) tuples
            
        Returns:
            List of compiled episodes
        """
        if not predictions:
            return []
        
        # Sort by timestamp
        sorted_preds = sorted(predictions, key=lambda x: x[0])
        
        if self.config.use_hysteresis:
            sorted_preds = self._apply_hysteresis(sorted_preds)
        
        episodes = self._group_into_episodes(sorted_preds)
        episodes = self._merge_adjacent_episodes(episodes)
        episodes = self._filter_by_duration(episodes)
        
        return episodes
    
    def compile_to_dataframe(
        self,
        predictions: List[Tuple[datetime, str, float]],
        room_name: str = "unknown",
    ) -> pd.DataFrame:
        """Compile predictions to episode DataFrame."""
        episodes = self.compile(predictions)
        
        if not episodes:
            return pd.DataFrame()
        
        records = []
        for ep in episodes:
            record = ep.to_dict()
            record["metadata"]["room"] = room_name
            records.append(record)
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df["start_time"] = pd.to_datetime(df["start_time"])
            df["end_time"] = pd.to_datetime(df["end_time"])
        
        return df
    
    def _apply_hysteresis(
        self,
        predictions: List[Tuple[datetime, str, float]],
    ) -> List[Tuple[datetime, str, float]]:
        """Apply hysteresis smoothing to predictions."""
        if not predictions:
            return predictions
        
        current_label = predictions[0][1]
        smoothed: List[Tuple[datetime, str, float]] = [
            (predictions[0][0], current_label, predictions[0][2])
        ]
        
        candidate_label: Optional[str] = None
        candidate_indices: List[int] = []
        
        for ts, label, conf in predictions[1:]:
            # No change requested.
            if label == current_label:
                candidate_label = None
                candidate_indices = []
                smoothed.append((ts, current_label, conf))
                continue
            
            # Confidence below off-threshold cannot trigger label change.
            if conf < self.config.hysteresis_off_threshold:
                candidate_label = None
                candidate_indices = []
                smoothed.append((ts, current_label, conf))
                continue
            
            # Only high-confidence windows can accumulate transition evidence.
            if conf >= self.config.hysteresis_on_threshold:
                if candidate_label != label:
                    candidate_label = label
                    candidate_indices = [len(smoothed)]
                else:
                    candidate_indices.append(len(smoothed))
                
                # Keep prior label until transition is confirmed.
                smoothed.append((ts, current_label, conf))
                
                if len(candidate_indices) >= self.config.hysteresis_min_windows:
                    for idx in candidate_indices:
                        old_ts, _, old_conf = smoothed[idx]
                        smoothed[idx] = (old_ts, candidate_label, old_conf)
                    current_label = candidate_label
                    candidate_label = None
                    candidate_indices = []
                continue
            
            # Between off and on thresholds: remain in current label.
            candidate_label = None
            candidate_indices = []
            smoothed.append((ts, current_label, conf))
        
        return smoothed
    
    def _group_into_episodes(
        self,
        predictions: List[Tuple[datetime, str, float]],
    ) -> List[Episode]:
        """Group consecutive same-label predictions into episodes."""
        if not predictions:
            return []
        
        episodes = []
        current_label = predictions[0][1]
        current_windows = [predictions[0]]
        
        for i in range(1, len(predictions)):
            ts, label, conf = predictions[i]
            
            if label == current_label:
                current_windows.append((ts, label, conf))
            else:
                # Create episode for current group
                episode = self._create_episode(current_windows)
                if episode:
                    episodes.append(episode)
                
                # Start new group
                current_label = label
                current_windows = [(ts, label, conf)]
        
        # Handle final group
        if current_windows:
            episode = self._create_episode(current_windows)
            if episode:
                episodes.append(episode)
        
        return episodes
    
    def _create_episode(
        self,
        windows: List[Tuple[datetime, str, float]],
    ) -> Optional[Episode]:
        """Create an episode from a list of windows."""
        if not windows:
            return None
        
        label = windows[0][1]
        start_time = windows[0][0]
        end_time = windows[-1][0]
        
        confidences = [w[2] for w in windows]
        avg_confidence = float(np.mean(confidences))
        
        # Filter by minimum confidence
        if avg_confidence < self.config.min_confidence_for_episode:
            return None
        
        return Episode(
            event_label=label,
            start_time=start_time,
            end_time=end_time,
            confidence=avg_confidence,
            window_count=len(windows),
        )
    
    def _merge_adjacent_episodes(self, episodes: List[Episode]) -> List[Episode]:
        """Merge adjacent episodes of same label with small gaps."""
        if not episodes:
            return episodes
        
        merged = [episodes[0]]
        
        for current in episodes[1:]:
            previous = merged[-1]
            
            if previous.can_merge(current, self.config.merge_gap_seconds):
                # Merge with previous
                merged[-1] = previous.merge(current)
            else:
                merged.append(current)
        
        return merged
    
    def _filter_by_duration(self, episodes: List[Episode]) -> List[Episode]:
        """Filter episodes by duration constraints."""
        filtered = []
        
        for ep in episodes:
            duration = ep.duration_seconds
            
            # Check minimum duration
            if duration < self.config.min_duration_seconds:
                continue
            
            # Check maximum duration
            if self.config.max_duration_seconds is not None:
                if duration > self.config.max_duration_seconds:
                    continue
            
            filtered.append(ep)
        
        return filtered


class MultiRoomEpisodeCompiler:
    """Compiles episodes for multiple rooms."""
    
    def __init__(self, config: Optional[EpisodeCompilerConfig] = None):
        self.config = config
        self._compilers: Dict[str, EpisodeCompiler] = {}
    
    def get_compiler(self, room_name: str) -> EpisodeCompiler:
        """Get or create compiler for a room."""
        if room_name not in self._compilers:
            self._compilers[room_name] = EpisodeCompiler(self.config)
        return self._compilers[room_name]
    
    def compile_all(
        self,
        room_predictions: Dict[str, List[Tuple[datetime, str, float]]],
    ) -> Dict[str, List[Episode]]:
        """
        Compile predictions for multiple rooms.
        
        Args:
            room_predictions: Dict mapping room names to prediction lists
            
        Returns:
            Dict mapping room names to episode lists
        """
        results = {}
        
        for room_name, predictions in room_predictions.items():
            compiler = self.get_compiler(room_name)
            episodes = compiler.compile(predictions)
            
            # Set room name on each episode
            for ep in episodes:
                ep.room = room_name
            
            results[room_name] = episodes
        
        return results
    
    def compile_all_to_dataframe(
        self,
        room_predictions: Dict[str, List[Tuple[datetime, str, float]]],
    ) -> pd.DataFrame:
        """Compile all rooms to a single DataFrame."""
        results = self.compile_all(room_predictions)
        
        all_records = []
        for room_name, episodes in results.items():
            for ep in episodes:
                record = ep.to_dict()
                record["room"] = room_name
                all_records.append(record)
        
        df = pd.DataFrame(all_records)
        
        if not df.empty:
            df["start_time"] = pd.to_datetime(df["start_time"])
            df["end_time"] = pd.to_datetime(df["end_time"])
        
        return df
    
    def reset_all(self) -> None:
        """Reset all compilers."""
        for compiler in self._compilers.values():
            compiler.reset()
    
    def reset_room(self, room_name: str) -> None:
        """Reset compiler for a specific room."""
        if room_name in self._compilers:
            self._compilers[room_name].reset()


def compile_day_episodes(
    predictions_df: pd.DataFrame,
    room_col: Optional[str] = None,
    timestamp_col: str = "timestamp",
    label_col: str = "predicted_label",
    confidence_col: str = "confidence",
    config: Optional[EpisodeCompilerConfig] = None,
) -> pd.DataFrame:
    """
    Convenience function to compile episodes from a DataFrame.
    
    Args:
        predictions_df: DataFrame with predictions
        room_col: Column name for room (if multi-room)
        timestamp_col: Column name for timestamps
        label_col: Column name for predicted labels
        confidence_col: Column name for confidence scores
        config: Optional compiler configuration
        
    Returns:
        DataFrame of compiled episodes
    """
    if predictions_df.empty:
        return pd.DataFrame()
    
    # Ensure timestamp is datetime
    df = predictions_df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    if room_col and room_col in df.columns:
        # Multi-room mode
        multi_compiler = MultiRoomEpisodeCompiler(config)
        
        room_predictions = {}
        for room_name in df[room_col].unique():
            room_df = df[df[room_col] == room_name]
            predictions = [
                (row[timestamp_col], row[label_col], row[confidence_col])
                for _, row in room_df.iterrows()
            ]
            room_predictions[room_name] = predictions
        
        return multi_compiler.compile_all_to_dataframe(room_predictions)
    else:
        # Single room mode
        compiler = EpisodeCompiler(config)
        predictions = [
            (row[timestamp_col], row[label_col], row[confidence_col])
            for _, row in df.iterrows()
        ]
        return compiler.compile_to_dataframe(predictions)
