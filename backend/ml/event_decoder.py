"""
Event Decoder Module

Converts raw model probabilities into discrete event predictions.
Part of PR-B1: Event Compiler + KPI/Gates.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DecoderState(Enum):
    """Internal decoder state."""
    UNOCCUPIED = "unoccupied"
    OCCUPIED_TRANSITION = "occupied_transition"
    OCCUPIED = "occupied"
    UNOCCUPIED_TRANSITION = "unoccupied_transition"


@dataclass
class DecoderConfig:
    """Configuration for event decoder."""
    
    # Occupancy thresholds
    occupancy_on_threshold: float = 0.60
    occupancy_off_threshold: float = 0.40
    
    # Hysteresis parameters
    use_hysteresis: bool = True
    hysteresis_min_windows: int = 2
    
    # Temporal smoothing
    temporal_smoothing_window: int = 0  # 0 = disabled
    
    # Activity selection
    activity_threshold: float = 0.0  # Min activity prob to consider
    use_softmax_for_activity: bool = False
    
    # Unknown handling
    unknown_threshold: float = 0.50  # Below this -> unknown label
    use_unknown_fallback: bool = True
    unknown_label: str = "unknown"
    
    def validate(self) -> None:
        """Validate configuration."""
        assert 0 <= self.occupancy_on_threshold <= 1, "occupancy_on_threshold must be in [0, 1]"
        assert 0 <= self.occupancy_off_threshold <= 1, "occupancy_off_threshold must be in [0, 1]"
        assert self.occupancy_on_threshold > self.occupancy_off_threshold, \
            "occupancy_on_threshold must be > occupancy_off_threshold"
        assert self.hysteresis_min_windows >= 1, "hysteresis_min_windows must be >= 1"
        assert self.temporal_smoothing_window >= 0, "temporal_smoothing_window must be >= 0"


@dataclass
class WindowPrediction:
    """A single window prediction result."""
    
    timestamp: datetime
    room_name: str
    occupancy_prob: float
    activity_probs: Dict[str, float]
    predicted_label: str
    confidence: float
    is_unknown: bool = False
    is_occupied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "room_name": self.room_name,
            "occupancy_prob": round(self.occupancy_prob, 4),
            "predicted_label": self.predicted_label,
            "confidence": round(self.confidence, 4),
            "is_unknown": self.is_unknown,
            "is_occupied": self.is_occupied,
            "activity_probs": {k: round(v, 4) for k, v in self.activity_probs.items()},
            "metadata": self.metadata,
        }


class EventDecoder:
    """Decodes model probabilities to discrete predictions."""
    
    def __init__(self, config: Optional[DecoderConfig] = None):
        self.config = config or DecoderConfig()
        self.config.validate()
        
        # Hysteresis state
        self._state = DecoderState.UNOCCUPIED
        self._state_window_count = 0
        self._current_label: Optional[str] = None
        
        # Temporal smoothing buffer
        self._smoothing_buffer: List[Dict[str, np.ndarray]] = []
    
    def reset(self) -> None:
        """Reset decoder state."""
        self._state = DecoderState.UNOCCUPIED
        self._state_window_count = 0
        self._current_label = None
        self._smoothing_buffer = []
    
    def decode(
        self,
        occupancy_probs: np.ndarray,
        activity_probs: Dict[str, np.ndarray],
        timestamps: List[datetime],
        room_name: str,
    ) -> List[WindowPrediction]:
        """
        Decode model outputs to predictions.
        
        Args:
            occupancy_probs: Array of occupancy probabilities (Head A)
            activity_probs: Dict of activity probabilities by label (Head B)
            timestamps: List of timestamps
            room_name: Name of the room
            
        Returns:
            List of WindowPrediction objects
        """
        # Validate inputs
        n_windows = len(occupancy_probs)
        if n_windows != len(timestamps):
            raise ValueError("occupancy_probs and timestamps must have same length")
        
        for label, probs in activity_probs.items():
            if len(probs) != n_windows:
                raise ValueError(f"activity_probs['{label}'] must have same length as occupancy_probs")
        
        # Apply temporal smoothing if enabled
        if self.config.temporal_smoothing_window > 0:
            occupancy_probs = self._apply_temporal_smoothing(occupancy_probs)
            activity_probs = {
                label: self._apply_temporal_smoothing(probs)
                for label, probs in activity_probs.items()
            }
        
        predictions = []
        
        for i in range(n_windows):
            occ_prob = float(occupancy_probs[i])
            act_probs = {label: float(probs[i]) for label, probs in activity_probs.items()}
            
            prediction = self._decode_single_window(
                occ_prob, act_probs, timestamps[i], room_name
            )
            predictions.append(prediction)
        
        return predictions
    
    def decode_to_dataframe(
        self,
        occupancy_probs: np.ndarray,
        activity_probs: Dict[str, np.ndarray],
        timestamps: List[datetime],
        room_name: str,
    ) -> pd.DataFrame:
        """Decode to DataFrame format."""
        predictions = self.decode(occupancy_probs, activity_probs, timestamps, room_name)
        
        if not predictions:
            return pd.DataFrame()
        
        records = [p.to_dict() for p in predictions]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        return df
    
    def _decode_single_window(
        self,
        occupancy_prob: float,
        activity_probs: Dict[str, float],
        timestamp: datetime,
        room_name: str,
    ) -> WindowPrediction:
        """Decode a single window."""
        # Update hysteresis state
        occupied = self._update_hysteresis_state(occupancy_prob)
        
        # Select activity label
        if occupied:
            predicted_label, max_activity_prob = self._select_activity_label(activity_probs)
            is_unknown = (
                predicted_label == self.config.unknown_label
                or self._is_unknown_prediction(max_activity_prob)
            )
            confidence = max_activity_prob
        else:
            predicted_label = "unoccupied"
            is_unknown = False
            confidence = 1.0 - occupancy_prob
        
        return WindowPrediction(
            timestamp=timestamp,
            room_name=room_name,
            occupancy_prob=occupancy_prob,
            activity_probs=activity_probs,
            predicted_label=predicted_label if not is_unknown else self.config.unknown_label,
            confidence=confidence,
            is_unknown=is_unknown,
            is_occupied=occupied,
        )
    
    def _update_hysteresis_state(self, occupancy_prob: float) -> bool:
        """Update hysteresis state and return current occupied status."""
        if not self.config.use_hysteresis:
            return occupancy_prob >= self.config.occupancy_on_threshold
        
        if self._state == DecoderState.UNOCCUPIED:
            if occupancy_prob >= self.config.occupancy_on_threshold:
                self._state = DecoderState.OCCUPIED_TRANSITION
                self._state_window_count = 1
            return False
        
        elif self._state == DecoderState.OCCUPIED_TRANSITION:
            if occupancy_prob >= self.config.occupancy_on_threshold:
                self._state_window_count += 1
                if self._state_window_count >= self.config.hysteresis_min_windows:
                    self._state = DecoderState.OCCUPIED
            else:
                # Fell back below threshold - return to unoccupied
                self._state = DecoderState.UNOCCUPIED
                self._state_window_count = 0
            return self._state == DecoderState.OCCUPIED
        
        elif self._state == DecoderState.OCCUPIED:
            if occupancy_prob < self.config.occupancy_off_threshold:
                self._state = DecoderState.UNOCCUPIED_TRANSITION
                self._state_window_count = 1
            return True
        
        elif self._state == DecoderState.UNOCCUPIED_TRANSITION:
            if occupancy_prob < self.config.occupancy_off_threshold:
                self._state_window_count += 1
                if self._state_window_count >= self.config.hysteresis_min_windows:
                    self._state = DecoderState.UNOCCUPIED
            else:
                # Rose above threshold - return to occupied
                self._state = DecoderState.OCCUPIED
                self._state_window_count = 0
            return self._state != DecoderState.UNOCCUPIED
        
        return False
    
    def _select_activity_label(
        self,
        activity_probs: Dict[str, float],
    ) -> Tuple[str, float]:
        """Select the activity label with highest probability."""
        if not activity_probs:
            return "unknown", 0.0
        
        if self.config.use_softmax_for_activity:
            # Apply softmax
            probs_array = np.array(list(activity_probs.values()))
            exp_probs = np.exp(probs_array - np.max(probs_array))
            softmax_probs = exp_probs / np.sum(exp_probs)
            
            max_idx = np.argmax(softmax_probs)
            labels = list(activity_probs.keys())
            max_label, max_prob = labels[max_idx], float(softmax_probs[max_idx])
        else:
            # Direct argmax
            max_item = max(activity_probs.items(), key=lambda x: x[1])
            max_label, max_prob = max_item[0], float(max_item[1])
        
        # Enforce minimum confidence required for activity selection.
        if max_prob < self.config.activity_threshold:
            return self.config.unknown_label, max_prob
        
        return max_label, max_prob
    
    def _is_unknown_prediction(self, max_activity_prob: float) -> bool:
        """Check if prediction should be marked as unknown."""
        if not self.config.use_unknown_fallback:
            return False
        return max_activity_prob < self.config.unknown_threshold
    
    def _apply_temporal_smoothing(self, probs: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to probability array."""
        if self.config.temporal_smoothing_window <= 1:
            return probs
        
        window = min(self.config.temporal_smoothing_window, len(probs))
        kernel = np.ones(window) / window
        smoothed = np.convolve(probs, kernel, mode='same')
        return smoothed


class RoomAwareDecoder:
    """Manages decoders for multiple rooms."""
    
    def __init__(self, config: Optional[DecoderConfig] = None):
        self.config = config
        self._decoders: Dict[str, EventDecoder] = {}
    
    def get_decoder(self, room_name: str) -> EventDecoder:
        """Get or create decoder for a room."""
        if room_name not in self._decoders:
            self._decoders[room_name] = EventDecoder(self.config)
        return self._decoders[room_name]
    
    def decode_room(
        self,
        room_name: str,
        occupancy_probs: np.ndarray,
        activity_probs: Dict[str, np.ndarray],
        timestamps: List[datetime],
    ) -> List[WindowPrediction]:
        """Decode predictions for a specific room."""
        decoder = self.get_decoder(room_name)
        return decoder.decode(occupancy_probs, activity_probs, timestamps, room_name)
    
    def reset_room(self, room_name: str) -> None:
        """Reset decoder for a specific room."""
        if room_name in self._decoders:
            self._decoders[room_name].reset()
    
    def reset_all(self) -> None:
        """Reset all decoders."""
        for decoder in self._decoders.values():
            decoder.reset()


def apply_decoder_to_predictions(
    predictions_df: pd.DataFrame,
    room_col: str = "room",
    timestamp_col: str = "timestamp",
    occupancy_prob_col: str = "prob_occupied",
    activity_prob_prefix: str = "prob_",
    config: Optional[DecoderConfig] = None,
) -> pd.DataFrame:
    """
    Apply decoder to a DataFrame of predictions.
    
    Args:
        predictions_df: DataFrame with raw probability columns
        room_col: Column name for room
        timestamp_col: Column name for timestamp
        occupancy_prob_col: Column name for occupancy probability
        activity_prob_prefix: Prefix for activity probability columns
        config: Optional decoder configuration
        
    Returns:
        DataFrame with decoded predictions
    """
    if predictions_df.empty:
        return pd.DataFrame()
    
    df = predictions_df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    decoder = RoomAwareDecoder(config)
    results = []
    
    for room_name in df[room_col].unique():
        room_df = df[df[room_col] == room_name].sort_values(timestamp_col)
        
        timestamps = room_df[timestamp_col].tolist()
        occupancy_probs = room_df[occupancy_prob_col].values
        
        # Extract activity probabilities
        activity_probs = {}
        for col in room_df.columns:
            if col.startswith(activity_prob_prefix) and col != occupancy_prob_col:
                label = col[len(activity_prob_prefix):]
                activity_probs[label] = room_df[col].values
        
        predictions = decoder.decode_room(
            room_name, occupancy_probs, activity_probs, timestamps
        )
        
        for pred in predictions:
            results.append(pred.to_dict())
    
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df["timestamp"] = pd.to_datetime(result_df["timestamp"])
    
    return result_df
