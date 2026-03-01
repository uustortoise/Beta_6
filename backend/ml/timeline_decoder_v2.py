"""
Timeline Decoder v2 Module

Segment reconstruction from probabilistic outputs with:
- Room-aware decode policy
- Boundary-aware constraints
- Hysteresis and gap-filling

Part of WS-3: Decoder v2 + Calibration
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


class DecodeState(Enum):
    """Internal decoder state."""
    UNCERTAIN = "uncertain"
    IN_EPISODE = "in_episode"
    BOUNDARY_CANDIDATE = "boundary_candidate"


@dataclass
class TimelineDecodePolicy:
    """
    Room-aware decoding policy for timeline reconstruction.
    
    These parameters control how probabilistic outputs are converted
    into discrete episode segments.
    """
    
    # Episode constraints
    min_episode_windows: int = 3  # Minimum windows to form episode
    max_gap_fill_windows: int = 2  # Fill gaps up to this length
    
    # Boundary thresholds
    boundary_on_threshold: float = 0.5  # Probability threshold for boundary
    boundary_off_threshold: float = 0.3  # Hysteresis off threshold
    
    # Hysteresis for state stability
    hysteresis_windows: int = 2  # Minimum windows to change state
    
    # Time-of-day priors (room_name -> (start_hour, end_hour) of expected activity)
    time_of_day_priors: Optional[Dict[str, Tuple[int, int]]] = None
    
    # Room-specific overrides
    room_name: str = "default"
    
    def validate(self) -> None:
        """Validate policy configuration."""
        assert self.min_episode_windows >= 1
        assert self.max_gap_fill_windows >= 0
        assert 0 < self.boundary_off_threshold < self.boundary_on_threshold < 1
        assert self.hysteresis_windows >= 1


@dataclass
class DecodedEpisode:
    """A decoded episode from timeline v2."""
    
    label: str
    start_idx: int
    end_idx: int
    start_time: datetime
    end_time: datetime
    confidence: float
    boundary_confidence: float
    
    @property
    def duration_seconds(self) -> float:
        """Episode duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def duration_minutes(self) -> float:
        """Episode duration in minutes."""
        return self.duration_seconds / 60.0
    
    @property
    def window_count(self) -> int:
        """Number of windows in episode."""
        return self.end_idx - self.start_idx + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'label': self.label,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': self.duration_seconds,
            'duration_minutes': self.duration_minutes,
            'window_count': self.window_count,
            'confidence': round(self.confidence, 4),
            'boundary_confidence': round(self.boundary_confidence, 4),
        }


class TimelineDecoderV2:
    """
    Decoder v2 for segment reconstruction from probabilistic outputs.
    
    Uses boundary signals and activity probabilities with hysteresis
    and room-aware policies to produce stable episode segments.
    """
    
    def __init__(self, policy: TimelineDecodePolicy):
        self.policy = policy
        policy.validate()
        
        # State tracking
        self._reset_state()
    
    def _reset_state(self) -> None:
        """Reset decoder state."""
        self.state = DecodeState.UNCERTAIN
        self.current_label: Optional[str] = None
        self.episode_start_idx: int = 0
        self.episode_windows: int = 0
        self.hysteresis_count: int = 0
        self.gap_windows: int = 0
    
    def decode(
        self,
        timestamps: np.ndarray,
        activity_probs: np.ndarray,  # [seq_len, num_classes]
        activity_labels: List[str],
        occupancy_probs: np.ndarray,  # [seq_len]
        boundary_start_probs: np.ndarray,  # [seq_len]
        boundary_end_probs: np.ndarray,  # [seq_len]
    ) -> List[DecodedEpisode]:
        """
        Decode timeline from probabilistic outputs.
        
        Args:
            timestamps: Array of datetime objects
            activity_probs: Per-window activity probabilities
            activity_labels: List of activity label strings
            occupancy_probs: Per-window occupancy probabilities
            boundary_start_probs: Per-window start boundary probabilities
            boundary_end_probs: Per-window end boundary probabilities
            
        Returns:
            List of decoded episodes
        """
        self._reset_state()
        episodes = []
        
        seq_len = len(timestamps)
        
        for i in range(seq_len):
            # Determine most likely activity
            if occupancy_probs[i] < 0.5:
                current_label = 'unoccupied'
                confidence = 1.0 - occupancy_probs[i]
            else:
                activity_idx = np.argmax(activity_probs[i])
                current_label = activity_labels[activity_idx]
                confidence = activity_probs[i][activity_idx]
            
            # State machine
            if self.state == DecodeState.UNCERTAIN:
                episodes.extend(self._handle_uncertain_state(
                    i, timestamps, current_label, confidence,
                    boundary_start_probs[i], boundary_end_probs[i]
                ))
            elif self.state == DecodeState.IN_EPISODE:
                episodes.extend(self._handle_in_episode_state(
                    i, timestamps, current_label, confidence,
                    boundary_start_probs[i], boundary_end_probs[i]
                ))
            elif self.state == DecodeState.BOUNDARY_CANDIDATE:
                episodes.extend(self._handle_boundary_candidate_state(
                    i, timestamps, current_label, confidence,
                    boundary_start_probs[i], boundary_end_probs[i]
                ))
        
        # Close final episode if exists
        if self.state == DecodeState.IN_EPISODE and self.current_label is not None:
            episode = self._create_episode(
                self.current_label,
                self.episode_start_idx,
                seq_len - 1,
                timestamps,
                confidence=0.5,  # Default for trailing episode
                boundary_confidence=0.5,
            )
            if episode.window_count >= self.policy.min_episode_windows:
                episodes.append(episode)
        
        # Post-process: merge short gaps
        episodes = self._merge_gaps(episodes, timestamps)
        
        return episodes
    
    def _handle_uncertain_state(
        self,
        idx: int,
        timestamps: np.ndarray,
        label: str,
        confidence: float,
        start_prob: float,
        end_prob: float,
    ) -> List[DecodedEpisode]:
        """Handle UNCERTAIN state."""
        episodes = []
        
        if label == 'unoccupied':
            return episodes
        
        # Check for episode start
        if start_prob >= self.policy.boundary_on_threshold:
            self.state = DecodeState.IN_EPISODE
            self.current_label = label
            self.episode_start_idx = idx
            self.episode_windows = 1
            self.hysteresis_count = 0
        elif confidence > 0.7:  # High confidence without explicit start
            self.state = DecodeState.IN_EPISODE
            self.current_label = label
            self.episode_start_idx = idx
            self.episode_windows = 1
            self.hysteresis_count = 0
        
        return episodes
    
    def _handle_in_episode_state(
        self,
        idx: int,
        timestamps: np.ndarray,
        label: str,
        confidence: float,
        start_prob: float,
        end_prob: float,
    ) -> List[DecodedEpisode]:
        """Handle IN_EPISODE state."""
        episodes = []
        
        # Check for episode end
        if end_prob >= self.policy.boundary_on_threshold:
            # Potential end - require hysteresis
            self.hysteresis_count += 1
            if self.hysteresis_count >= self.policy.hysteresis_windows:
                # End episode
                episode = self._create_episode(
                    self.current_label,
                    self.episode_start_idx,
                    idx - 1,
                    timestamps,
                    confidence,
                    end_prob,
                )
                if episode.window_count >= self.policy.min_episode_windows:
                    episodes.append(episode)
                
                # Transition to uncertain
                self.state = DecodeState.UNCERTAIN
                self.current_label = None
                self.hysteresis_count = 0
        elif label != self.current_label and label != 'unoccupied':
            # Label changed without boundary signal - potential transition
            self.hysteresis_count += 1
            if self.hysteresis_count >= self.policy.hysteresis_windows:
                # End current episode
                episode = self._create_episode(
                    self.current_label,
                    self.episode_start_idx,
                    idx - 1,
                    timestamps,
                    confidence,
                    start_prob,  # Use start prob as proxy
                )
                if episode.window_count >= self.policy.min_episode_windows:
                    episodes.append(episode)
                
                # Start new episode
                self.current_label = label
                self.episode_start_idx = idx
                self.episode_windows = 1
                self.hysteresis_count = 0
        elif label == 'unoccupied':
            # Gap detected
            self.gap_windows += 1
            if self.gap_windows > self.policy.max_gap_fill_windows:
                # End episode at start of gap
                end_idx = idx - self.gap_windows
                if end_idx >= self.episode_start_idx:
                    episode = self._create_episode(
                        self.current_label,
                        self.episode_start_idx,
                        end_idx,
                        timestamps,
                        confidence,
                        0.5,
                    )
                    if episode.window_count >= self.policy.min_episode_windows:
                        episodes.append(episode)
                
                self.state = DecodeState.UNCERTAIN
                self.current_label = None
                self.gap_windows = 0
                self.hysteresis_count = 0
        else:
            # Continue episode
            self.episode_windows += 1
            self.hysteresis_count = max(0, self.hysteresis_count - 1)
            self.gap_windows = 0
        
        return episodes
    
    def _handle_boundary_candidate_state(
        self,
        idx: int,
        timestamps: np.ndarray,
        label: str,
        confidence: float,
        start_prob: float,
        end_prob: float,
    ) -> List[DecodedEpisode]:
        """Handle BOUNDARY_CANDIDATE state."""
        # Simplified: treat as uncertain for now
        self.state = DecodeState.UNCERTAIN
        return []
    
    def _create_episode(
        self,
        label: str,
        start_idx: int,
        end_idx: int,
        timestamps: np.ndarray,
        confidence: float,
        boundary_confidence: float,
    ) -> DecodedEpisode:
        """Create a DecodedEpisode from indices."""
        return DecodedEpisode(
            label=label,
            start_idx=start_idx,
            end_idx=end_idx,
            start_time=pd.to_datetime(timestamps[start_idx]),
            end_time=pd.to_datetime(timestamps[end_idx]),
            confidence=confidence,
            boundary_confidence=boundary_confidence,
        )
    
    def _merge_gaps(
        self,
        episodes: List[DecodedEpisode],
        timestamps: np.ndarray,
    ) -> List[DecodedEpisode]:
        """
        Merge episodes of same label with small gaps.
        
        Args:
            episodes: List of decoded episodes
            timestamps: Array of timestamps
            
        Returns:
            Merged episode list
        """
        if len(episodes) < 2:
            return episodes
        
        merged = [episodes[0]]
        
        for current in episodes[1:]:
            previous = merged[-1]
            
            # Check if same label and small gap
            if current.label == previous.label:
                gap_windows = current.start_idx - previous.end_idx - 1
                if gap_windows <= self.policy.max_gap_fill_windows:
                    # Merge with previous
                    merged[-1] = DecodedEpisode(
                        label=previous.label,
                        start_idx=previous.start_idx,
                        end_idx=current.end_idx,
                        start_time=previous.start_time,
                        end_time=current.end_time,
                        confidence=min(previous.confidence, current.confidence),
                        boundary_confidence=min(previous.boundary_confidence, current.boundary_confidence),
                    )
                    continue
            
            merged.append(current)
        
        return merged
    
    def decode_to_dataframe(
        self,
        timestamps: np.ndarray,
        activity_probs: np.ndarray,
        activity_labels: List[str],
        occupancy_probs: np.ndarray,
        boundary_start_probs: np.ndarray,
        boundary_end_probs: np.ndarray,
    ) -> pd.DataFrame:
        """
        Decode and return as DataFrame.
        
        Returns:
            DataFrame with episode information
        """
        episodes = self.decode(
            timestamps, activity_probs, activity_labels,
            occupancy_probs, boundary_start_probs, boundary_end_probs
        )
        
        if not episodes:
            return pd.DataFrame()
        
        records = [ep.to_dict() for ep in episodes]
        return pd.DataFrame(records)


def create_default_policy(room_name: str = "default") -> TimelineDecodePolicy:
    """Create a default decode policy."""
    return TimelineDecodePolicy(room_name=room_name)


def decode_timeline_v2(
    timestamps: np.ndarray,
    activity_probs: np.ndarray,
    activity_labels: List[str],
    occupancy_probs: np.ndarray,
    boundary_start_probs: np.ndarray,
    boundary_end_probs: np.ndarray,
    room_name: str = "default",
    policy: Optional[TimelineDecodePolicy] = None,
) -> List[DecodedEpisode]:
    """
    Convenience function for timeline decoding.
    
    Args:
        timestamps: Array of timestamps
        activity_probs: Activity probabilities [seq_len, num_classes]
        activity_labels: List of activity label strings
        occupancy_probs: Occupancy probabilities [seq_len]
        boundary_start_probs: Start boundary probabilities [seq_len]
        boundary_end_probs: End boundary probabilities [seq_len]
        room_name: Room name for policy selection
        policy: Optional custom policy
        
    Returns:
        List of decoded episodes
    """
    if policy is None:
        policy = create_default_policy(room_name)
    
    decoder = TimelineDecoderV2(policy)
    return decoder.decode(
        timestamps, activity_probs, activity_labels,
        occupancy_probs, boundary_start_probs, boundary_end_probs
    )
