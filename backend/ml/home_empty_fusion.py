"""
Home-Empty Fusion Module

Multi-room fusion for home-empty detection with false-empty protection.
Part of PR-B3: Home-Empty Fusion + Household Gate.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from processors.profile_processor import normalize_resident_home_context

logger = logging.getLogger(__name__)

class HomeEmptyState(Enum):
    """Home-empty detection state."""
    OCCUPIED = "occupied"  # At least one room occupied
    EMPTY = "empty"  # All rooms unoccupied
    UNCERTAIN = "uncertain"  # Insufficient data


class EntrancePenaltyStatus(Enum):
    """Status of entrance penalty logic."""
    NO_PENALTY = "no_penalty"
    RECENT_ENTRANCE = "recent_entrance"
    PENALTY_ACTIVE = "penalty_active"


@dataclass(frozen=True)
class ResidentHomeContext:
    """Typed resident/home context used by household routing and reporting."""

    household_type: Optional[str] = None
    helper_presence: Optional[str] = None
    layout_topology: Optional[str] = None
    adjacency: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    cohort_key: str = "unknown_household:unknown_helper:unknown_topology"
    status: str = "missing_required_context"
    missing_fields: Tuple[str, ...] = ()

    @property
    def requires_conservative_empty_gate(self) -> bool:
        return self.household_type == "multi_resident" or self.helper_presence in {"scheduled", "live_in"}


def build_resident_home_context(payload: Optional[Dict[str, Any]]) -> ResidentHomeContext:
    normalized = normalize_resident_home_context(payload or {})
    layout = normalized.get("layout", {}) if isinstance(normalized.get("layout"), dict) else {}
    adjacency_payload = layout.get("adjacency", {}) if isinstance(layout, dict) else {}
    adjacency: Dict[str, Tuple[str, ...]] = {}
    if isinstance(adjacency_payload, dict):
        for room, neighbors in adjacency_payload.items():
            if isinstance(neighbors, list):
                adjacency[str(room)] = tuple(str(item) for item in neighbors)
    return ResidentHomeContext(
        household_type=normalized.get("household_type"),
        helper_presence=normalized.get("helper_presence"),
        layout_topology=layout.get("topology") if isinstance(layout, dict) else None,
        adjacency=adjacency,
        cohort_key=str(normalized.get("cohort_key") or "unknown_household:unknown_helper:unknown_topology"),
        status=str(normalized.get("status") or "missing_required_context"),
        missing_fields=tuple(str(item) for item in normalized.get("missing_fields") or []),
    )


@dataclass
class HomeEmptyConfig:
    """Configuration for home-empty fusion."""
    
    # Precision requirements (hard safety constraints)
    min_precision: float = 0.95
    max_false_empty_rate: float = 0.05
    
    # Entrance penalty parameters
    entrance_penalty_duration_seconds: float = 300.0  # 5 minutes
    entrance_penalty_prob_boost: float = 0.20  # Boost occupancy prob after entrance
    
    # Room consensus parameters
    room_consensus_threshold: float = 0.5  # Fraction of rooms that must agree
    min_rooms_for_consensus: int = 2  # Minimum rooms needed for empty detection
    
    # Temporal smoothing
    temporal_window_seconds: float = 60.0  # Smoothing window
    min_empty_duration_seconds: float = 30.0  # Minimum duration to count as empty
    expected_interval_seconds: Optional[float] = None  # Auto-infer when None
    
    # Uncertainty handling
    max_unknown_rate_for_empty: float = 0.3  # Max unknown rate to declare empty
    max_uncertain_rate: float = 0.20  # Household coverage gate threshold
    max_prediction_age_seconds: float = 90.0  # Max age for nearest prediction matching
    alignment_tolerance_seconds: float = 15.0  # GT alignment tolerance for gate checks
    
    def validate(self) -> None:
        """Validate configuration."""
        if not (0 <= self.min_precision <= 1):
            raise ValueError("min_precision must be in [0, 1]")
        if not (0 <= self.max_false_empty_rate <= 1):
            raise ValueError("max_false_empty_rate must be in [0, 1]")
        if self.entrance_penalty_duration_seconds < 0:
            raise ValueError("entrance_penalty_duration_seconds must be >= 0")
        if not (0 <= self.entrance_penalty_prob_boost <= 1):
            raise ValueError("entrance_penalty_prob_boost must be in [0, 1]")
        if not (0 <= self.room_consensus_threshold <= 1):
            raise ValueError("room_consensus_threshold must be in [0, 1]")
        if self.min_rooms_for_consensus < 1:
            raise ValueError("min_rooms_for_consensus must be >= 1")
        if self.max_prediction_age_seconds < 0:
            raise ValueError("max_prediction_age_seconds must be >= 0")
        if self.alignment_tolerance_seconds < 0:
            raise ValueError("alignment_tolerance_seconds must be >= 0")
        if self.expected_interval_seconds is not None and self.expected_interval_seconds <= 0:
            raise ValueError("expected_interval_seconds must be > 0 when provided")


@dataclass
class RoomState:
    """State of a single room at a timestamp."""
    room_name: str
    timestamp: datetime
    occupancy_prob: float
    is_occupied: bool
    activity_label: Optional[str] = None
    confidence: float = 0.0


@dataclass
class HomeEmptyPrediction:
    """Home-empty prediction for a timestamp."""
    timestamp: datetime
    state: HomeEmptyState
    confidence: float
    room_states: List[RoomState]
    
    # Fusion metadata
    occupied_room_count: int = 0
    unoccupied_room_count: int = 0
    unknown_room_count: int = 0
    total_room_count: int = 0
    
    # Entrance penalty info
    entrance_penalty_status: EntrancePenaltyStatus = EntrancePenaltyStatus.NO_PENALTY
    seconds_since_entrance: Optional[float] = None
    context_contract_status: str = "not_provided"
    context_cohort: Optional[str] = None
    context_missing_fields: Tuple[str, ...] = ()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "state": self.state.value,
            "confidence": round(self.confidence, 4),
            "occupied_room_count": self.occupied_room_count,
            "unoccupied_room_count": self.unoccupied_room_count,
            "unknown_room_count": self.unknown_room_count,
            "total_room_count": self.total_room_count,
            "entrance_penalty_status": self.entrance_penalty_status.value,
            "seconds_since_entrance": self.seconds_since_entrance,
            "context_contract_status": self.context_contract_status,
            "context_cohort": self.context_cohort,
            "context_missing_fields": list(self.context_missing_fields),
        }


@dataclass
class HomeEmptyEpisode:
    """An episode of home-empty state."""
    start_time: datetime
    end_time: datetime
    confidence: float
    room_participation: Set[str]
    
    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def duration_minutes(self) -> float:
        """Duration in minutes."""
        return self.duration_seconds / 60.0


class HomeEmptyFusion:
    """Multi-room fusion for home-empty detection."""
    
    def __init__(self, config: Optional[HomeEmptyConfig] = None):
        self.config = config or HomeEmptyConfig()
        self.config.validate()
        
        # State tracking
        self._last_entrance_time: Optional[datetime] = None
        self._previous_state: HomeEmptyState = HomeEmptyState.OCCUPIED
        self._empty_start_time: Optional[datetime] = None
    
    def reset(self) -> None:
        """Reset fusion state."""
        self._last_entrance_time = None
        self._previous_state = HomeEmptyState.OCCUPIED
        self._empty_start_time = None
    
    def fuse(
        self,
        room_predictions: Dict[str, pd.DataFrame],
        timestamps: List[datetime],
        resident_home_context: Optional[ResidentHomeContext | Dict[str, Any]] = None,
    ) -> List[HomeEmptyPrediction]:
        """
        Fuse multi-room predictions into home-empty predictions.
        
        Args:
            room_predictions: Dict mapping room names to prediction DataFrames
            timestamps: List of timestamps to evaluate
            
        Returns:
            List of HomeEmptyPrediction objects
        """
        predictions = []
        context = (
            resident_home_context
            if isinstance(resident_home_context, ResidentHomeContext)
            else build_resident_home_context(resident_home_context)
            if resident_home_context is not None
            else None
        )
        
        for ts in timestamps:
            prediction = self._fuse_single_timestamp(
                room_predictions,
                ts,
                resident_home_context=context,
            )
            predictions.append(prediction)
        
        # Apply temporal smoothing
        predictions = self._apply_temporal_smoothing(predictions)
        
        return predictions
    
    def fuse_to_dataframe(
        self,
        room_predictions: Dict[str, pd.DataFrame],
        timestamps: List[datetime],
        resident_home_context: Optional[ResidentHomeContext | Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Fuse predictions to DataFrame format."""
        predictions = self.fuse(
            room_predictions,
            timestamps,
            resident_home_context=resident_home_context,
        )
        
        if not predictions:
            return pd.DataFrame()
        
        records = [p.to_dict() for p in predictions]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        return df
    
    def detect_episodes(
        self,
        predictions: List[HomeEmptyPrediction],
    ) -> List[HomeEmptyEpisode]:
        """
        Detect home-empty episodes from predictions.
        
        Args:
            predictions: List of HomeEmptyPrediction objects
            
        Returns:
            List of HomeEmptyEpisode objects
        """
        episodes = []
        current_episode: Optional[HomeEmptyEpisode] = None
        
        for pred in predictions:
            if pred.state == HomeEmptyState.EMPTY:
                if current_episode is None:
                    # Start new episode
                    current_episode = HomeEmptyEpisode(
                        start_time=pred.timestamp,
                        end_time=pred.timestamp,
                        confidence=pred.confidence,
                        room_participation=set(),
                    )
                else:
                    # Continue episode
                    current_episode.end_time = pred.timestamp
                    current_episode.confidence = min(current_episode.confidence, pred.confidence)
                for state in pred.room_states:
                    if not state.is_occupied:
                        current_episode.room_participation.add(state.room_name)
            else:
                if current_episode is not None:
                    # End episode
                    if current_episode.duration_seconds >= self.config.min_empty_duration_seconds:
                        episodes.append(current_episode)
                    current_episode = None
        
        # Handle final episode
        if current_episode is not None:
            if current_episode.duration_seconds >= self.config.min_empty_duration_seconds:
                episodes.append(current_episode)
        
        return episodes
    
    def _fuse_single_timestamp(
        self,
        room_predictions: Dict[str, pd.DataFrame],
        timestamp: datetime,
        resident_home_context: Optional[ResidentHomeContext] = None,
    ) -> HomeEmptyPrediction:
        """Fuse predictions for a single timestamp."""
        room_states = []
        occupied_count = 0
        unoccupied_count = 0
        unknown_count = 0
        entrance_signal = False
        
        for room_name, pred_df in room_predictions.items():
            # Find prediction closest to timestamp
            row = self._find_nearest_prediction(pred_df, timestamp)
            
            if row is None:
                unknown_count += 1
                continue
            
            occupancy_prob = row.get("occupancy_prob", 0.5)
            is_occupied = row.get("predicted_label", "unknown") != "unoccupied"
            
            # Apply entrance penalty if active
            penalty_status, seconds_since = self._check_entrance_penalty(timestamp)
            if penalty_status == EntrancePenaltyStatus.PENALTY_ACTIVE:
                occupancy_prob = min(1.0, occupancy_prob + self.config.entrance_penalty_prob_boost)
                is_occupied = True  # Force occupied during penalty
            
            room_state = RoomState(
                room_name=room_name,
                timestamp=timestamp,
                occupancy_prob=occupancy_prob,
                is_occupied=is_occupied,
                activity_label=row.get("predicted_label"),
                confidence=row.get("confidence", 0.0),
            )
            
            room_states.append(room_state)

            room_key = str(room_name).strip().lower()
            if "entrance" in room_key and is_occupied:
                entrance_signal = True
            
            if is_occupied:
                occupied_count += 1
            else:
                unoccupied_count += 1
        
        total_rooms = len(room_predictions)
        
        # Determine home state
        state, confidence = self._determine_home_state(
            occupied_count,
            unoccupied_count,
            unknown_count,
            total_rooms,
            resident_home_context=resident_home_context,
        )
        
        # Update entrance tracking
        if entrance_signal:
            self._last_entrance_time = timestamp
        elif state == HomeEmptyState.OCCUPIED and self._previous_state == HomeEmptyState.EMPTY:
            self._last_entrance_time = timestamp
        
        self._previous_state = state
        
        # Check entrance penalty status
        penalty_status, seconds_since = self._check_entrance_penalty(timestamp)
        
        return HomeEmptyPrediction(
            timestamp=timestamp,
            state=state,
            confidence=confidence,
            room_states=room_states,
            occupied_room_count=occupied_count,
            unoccupied_room_count=unoccupied_count,
            unknown_room_count=unknown_count,
            total_room_count=total_rooms,
            entrance_penalty_status=penalty_status,
            seconds_since_entrance=seconds_since,
            context_contract_status=(
                resident_home_context.status if resident_home_context is not None else "not_provided"
            ),
            context_cohort=resident_home_context.cohort_key if resident_home_context is not None else None,
            context_missing_fields=(
                resident_home_context.missing_fields if resident_home_context is not None else ()
            ),
        )
    
    def _find_nearest_prediction(
        self,
        pred_df: pd.DataFrame,
        timestamp: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Find the prediction row nearest to the given timestamp."""
        if pred_df.empty:
            return None
        
        if "timestamp" not in pred_df.columns:
            return None
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(pred_df["timestamp"]):
            pred_df = pred_df.copy()
            pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
        
        # Find nearest
        time_diffs = abs(pred_df["timestamp"] - timestamp)
        nearest_idx = time_diffs.idxmin()
        nearest_delta_seconds = float(time_diffs.loc[nearest_idx].total_seconds())
        if nearest_delta_seconds > float(self.config.max_prediction_age_seconds):
            return None

        return pred_df.loc[nearest_idx].to_dict()
    
    def _check_entrance_penalty(
        self,
        timestamp: datetime,
    ) -> Tuple[EntrancePenaltyStatus, Optional[float]]:
        """Check if entrance penalty is active."""
        if self._last_entrance_time is None:
            return EntrancePenaltyStatus.NO_PENALTY, None
        
        seconds_since = (timestamp - self._last_entrance_time).total_seconds()
        
        if seconds_since < self.config.entrance_penalty_duration_seconds:
            return EntrancePenaltyStatus.PENALTY_ACTIVE, seconds_since
        
        return EntrancePenaltyStatus.NO_PENALTY, seconds_since
    
    def _determine_home_state(
        self,
        occupied_count: int,
        unoccupied_count: int,
        unknown_count: int,
        total_rooms: int,
        resident_home_context: Optional[ResidentHomeContext] = None,
    ) -> Tuple[HomeEmptyState, float]:
        """
        Determine home state based on room consensus.
        
        Returns:
            Tuple of (state, confidence)
        """
        if total_rooms < self.config.min_rooms_for_consensus:
            return HomeEmptyState.UNCERTAIN, 0.0
        
        # Calculate unknown rate
        unknown_rate = unknown_count / total_rooms if total_rooms > 0 else 0.0
        
        if unknown_rate > self.config.max_unknown_rate_for_empty:
            return HomeEmptyState.UNCERTAIN, 1.0 - unknown_rate
        
        known_rooms = occupied_count + unoccupied_count
        if known_rooms < self.config.min_rooms_for_consensus:
            return HomeEmptyState.UNCERTAIN, 0.0

        occupied_ratio = occupied_count / total_rooms
        empty_ratio = unoccupied_count / total_rooms
        threshold = float(self.config.room_consensus_threshold)
        if resident_home_context is not None and resident_home_context.requires_conservative_empty_gate:
            threshold = max(threshold, 1.0)

        # Safety-first: any occupied evidence keeps household occupied.
        if occupied_count > 0:
            return HomeEmptyState.OCCUPIED, occupied_ratio

        if resident_home_context is not None and resident_home_context.status != "ready":
            return HomeEmptyState.UNCERTAIN, max(occupied_ratio, empty_ratio)

        # Empty only when enough room consensus exists.
        if empty_ratio >= threshold:
            return HomeEmptyState.EMPTY, empty_ratio

        return HomeEmptyState.UNCERTAIN, max(occupied_ratio, empty_ratio)
    
    def _apply_temporal_smoothing(
        self,
        predictions: List[HomeEmptyPrediction],
    ) -> List[HomeEmptyPrediction]:
        """Apply temporal smoothing to predictions."""
        if len(predictions) < 2:
            return predictions
        
        smoothed = []
        interval_seconds = self._infer_interval_seconds(predictions)
        window_size = int(round(self.config.temporal_window_seconds / max(interval_seconds, 1e-6)))
        window_size = max(1, window_size)
        
        for i, pred in enumerate(predictions):
            # Get window
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(predictions), i + window_size // 2 + 1)
            window = predictions[start_idx:end_idx]
            
            # Majority vote
            empty_count = sum(1 for p in window if p.state == HomeEmptyState.EMPTY)
            occupied_count = sum(1 for p in window if p.state == HomeEmptyState.OCCUPIED)
            uncertain_count = sum(1 for p in window if p.state == HomeEmptyState.UNCERTAIN)
            
            # Create smoothed prediction
            if empty_count > occupied_count:
                new_state = HomeEmptyState.EMPTY
            elif uncertain_count > max(empty_count, occupied_count):
                new_state = HomeEmptyState.UNCERTAIN
            else:
                new_state = HomeEmptyState.OCCUPIED
            
            smoothed_pred = HomeEmptyPrediction(
                timestamp=pred.timestamp,
                state=new_state,
                confidence=pred.confidence,
                room_states=pred.room_states,
                occupied_room_count=pred.occupied_room_count,
                unoccupied_room_count=pred.unoccupied_room_count,
                unknown_room_count=pred.unknown_room_count,
                total_room_count=pred.total_room_count,
                entrance_penalty_status=pred.entrance_penalty_status,
                seconds_since_entrance=pred.seconds_since_entrance,
            )
            
            smoothed.append(smoothed_pred)
        
        return smoothed

    def _infer_interval_seconds(self, predictions: List[HomeEmptyPrediction]) -> float:
        """Infer sampling interval for smoothing window sizing."""
        if self.config.expected_interval_seconds is not None:
            return float(self.config.expected_interval_seconds)
        if len(predictions) < 2:
            return 10.0
        timestamps = pd.to_datetime([p.timestamp for p in predictions])
        diffs = timestamps.to_series().diff().dt.total_seconds().dropna()
        if diffs.empty:
            return 10.0
        median_diff = float(diffs.median())
        if not np.isfinite(median_diff) or median_diff <= 0:
            return 10.0
        return median_diff


class HouseholdGate:
    """Household-level gate checking for home-empty fusion."""
    
    def __init__(self, config: Optional[HomeEmptyConfig] = None):
        self.config = config or HomeEmptyConfig()
        self.config.validate()
    
    def check_household_gate(
        self,
        predictions: List[HomeEmptyPrediction],
        ground_truth: Optional[List[Tuple[datetime, bool]]] = None,
    ) -> Dict[str, Any]:
        """
        Check household-level gates.
        
        Args:
            predictions: List of home-empty predictions
            ground_truth: Optional list of (timestamp, is_empty) tuples
            
        Returns:
            Dictionary with gate results
        """
        results = {
            "precision_check": self._check_precision(predictions, ground_truth),
            "false_empty_rate_check": self._check_false_empty_rate(predictions, ground_truth),
            "coverage_check": self._check_coverage(predictions),
        }
        
        # Overall status
        all_pass = all(r["passed"] for r in results.values())
        results["overall_passed"] = all_pass
        
        return results
    
    def _check_precision(
        self,
        predictions: List[HomeEmptyPrediction],
        ground_truth: Optional[List[Tuple[datetime, bool]]],
    ) -> Dict[str, Any]:
        """Check precision requirement."""
        if ground_truth is None:
            return {"passed": True, "message": "No ground truth available"}

        y_true, y_pred = self._align_for_binary_metrics(predictions, ground_truth)
        if len(y_true) == 0:
            return {"passed": True, "message": "No matching ground truth"}
        
        from sklearn.metrics import precision_score
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
        except Exception:
            precision = 0.0
        
        passed = precision >= self.config.min_precision
        
        return {
            "passed": passed,
            "precision": precision,
            "threshold": self.config.min_precision,
            "message": f"Precision: {precision:.4f} (threshold: {self.config.min_precision})",
        }
    
    def _check_false_empty_rate(
        self,
        predictions: List[HomeEmptyPrediction],
        ground_truth: Optional[List[Tuple[datetime, bool]]],
    ) -> Dict[str, Any]:
        """Check false-empty rate requirement."""
        if ground_truth is None:
            return {"passed": True, "message": "No ground truth available"}

        y_true, y_pred = self._align_for_binary_metrics(predictions, ground_truth)
        if len(y_true) == 0:
            return {"passed": True, "message": "No matching ground truth"}

        occupied_mask = (y_true == 0)
        false_positives = int(np.logical_and(occupied_mask, y_pred == 1).sum())
        true_negatives = int(np.logical_and(occupied_mask, y_pred == 0).sum())
        total_occupied = int(occupied_mask.sum())
        false_empty_rate = false_positives / total_occupied if total_occupied > 0 else 0.0
        
        passed = false_empty_rate <= self.config.max_false_empty_rate
        
        return {
            "passed": passed,
            "false_empty_rate": false_empty_rate,
            "threshold": self.config.max_false_empty_rate,
            "message": f"False-empty rate: {false_empty_rate:.4f} (threshold: {self.config.max_false_empty_rate})",
        }
    
    def _check_coverage(
        self,
        predictions: List[HomeEmptyPrediction],
    ) -> Dict[str, Any]:
        """Check that we have sufficient coverage."""
        empty_predictions = [p for p in predictions if p.state == HomeEmptyState.EMPTY]
        uncertain_predictions = [p for p in predictions if p.state == HomeEmptyState.UNCERTAIN]
        
        total = len(predictions)
        if total == 0:
            return {"passed": False, "message": "No predictions"}
        
        uncertain_rate = len(uncertain_predictions) / total
        
        passed = uncertain_rate <= float(self.config.max_uncertain_rate)
        
        return {
            "passed": passed,
            "uncertain_rate": uncertain_rate,
            "empty_episodes": len(empty_predictions),
            "threshold": float(self.config.max_uncertain_rate),
            "message": f"Uncertain rate: {uncertain_rate:.4f} (threshold: {self.config.max_uncertain_rate})",
        }

    def _align_for_binary_metrics(
        self,
        predictions: List[HomeEmptyPrediction],
        ground_truth: List[Tuple[datetime, bool]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align predictions to ground truth using timestamp tolerance."""
        if not predictions or not ground_truth:
            return np.array([], dtype=int), np.array([], dtype=int)

        pred_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([p.timestamp for p in predictions]),
                "pred_empty": [1 if p.state == HomeEmptyState.EMPTY else 0 for p in predictions],
            }
        ).sort_values("timestamp")
        gt_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([g[0] for g in ground_truth]),
                "gt_empty": [1 if g[1] else 0 for g in ground_truth],
            }
        ).sort_values("timestamp")

        tolerance = pd.Timedelta(seconds=float(self.config.alignment_tolerance_seconds))
        merged = pd.merge_asof(
            gt_df,
            pred_df,
            on="timestamp",
            direction="nearest",
            tolerance=tolerance,
        )
        merged = merged.dropna(subset=["pred_empty"])
        if merged.empty:
            return np.array([], dtype=int), np.array([], dtype=int)
        return merged["gt_empty"].astype(int).to_numpy(), merged["pred_empty"].astype(int).to_numpy()


def fuse_home_empty_predictions(
    room_predictions: Dict[str, pd.DataFrame],
    timestamps: Optional[List[datetime]] = None,
    config: Optional[HomeEmptyConfig] = None,
    resident_home_context: Optional[ResidentHomeContext | Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Convenience function to fuse home-empty predictions.
    
    Args:
        room_predictions: Dict mapping room names to prediction DataFrames
        timestamps: Optional list of timestamps (auto-generated if None)
        config: Optional configuration
        
    Returns:
        DataFrame with home-empty predictions
    """
    fusion = HomeEmptyFusion(config)
    
    # Generate timestamps if not provided
    if timestamps is None:
        # Use union of all timestamps from room predictions
        all_timestamps = set()
        for pred_df in room_predictions.values():
            if "timestamp" in pred_df.columns:
                all_timestamps.update(pd.to_datetime(pred_df["timestamp"]).tolist())
        timestamps = sorted(all_timestamps)
    
    return fusion.fuse_to_dataframe(
        room_predictions,
        timestamps,
        resident_home_context=resident_home_context,
    )
