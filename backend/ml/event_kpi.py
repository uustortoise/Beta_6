"""
Event KPI Calculator Module

Computes event-level KPIs for gate checking.
Part of PR-B2: Event KPI + Gate Layer.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from ml.derived_events import CareKPIExtractor

logger = logging.getLogger(__name__)


@dataclass
class EventKPIConfig:
    """Configuration for KPI alignment and robustness."""
    timestamp_tolerance_seconds: float = 15.0


@dataclass
class EventKPIMetrics:
    """Container for event-level KPI metrics."""
    
    # Home-empty safety metrics
    home_empty_precision: float = 0.0
    home_empty_recall: float = 0.0
    home_empty_false_empty_rate: float = 0.0
    
    # Unknown rates
    unknown_rate_global: float = 0.0
    unknown_rate_per_room: Dict[str, float] = field(default_factory=dict)
    
    # Event recalls by name
    event_recalls: Dict[str, float] = field(default_factory=dict)
    event_precisions: Dict[str, float] = field(default_factory=dict)
    event_f1s: Dict[str, float] = field(default_factory=dict)
    event_supports: Dict[str, int] = field(default_factory=dict)
    
    # Care KPIs
    sleep_hours: float = 0.0
    sleep_efficiency: float = 0.0
    bathroom_visits: int = 0
    shower_detected: bool = False
    meal_count: int = 0
    out_time_minutes: float = 0.0
    
    def to_gate_metrics(self) -> Dict[str, Any]:
        """Convert to format expected by gate checker."""
        return {
            "home_empty_precision": self.home_empty_precision,
            "home_empty_false_empty_rate": self.home_empty_false_empty_rate,
            "unknown_rate_global": self.unknown_rate_global,
            "unknown_rate_per_room": self.unknown_rate_per_room,
            "event_recalls": self.event_recalls,
            "event_supports": self.event_supports,
        }


class EventKPICalculator:
    """Calculates event-level KPIs from predictions and ground truth."""
    
    def __init__(self, config: Optional[EventKPIConfig] = None):
        self.config = config or EventKPIConfig()
        self.care_extractor = CareKPIExtractor()

    def _resolve_label_column(self, df: pd.DataFrame) -> str:
        """Resolve label column for either prediction or ground-truth frame."""
        if "predicted_label" in df.columns:
            return "predicted_label"
        if "label" in df.columns:
            return "label"
        raise KeyError("Expected one of label/predicted_label columns")

    def _asof_align(
        self,
        ground_truth_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        *,
        gt_label_col: str,
        pred_label_col: str,
    ) -> pd.DataFrame:
        """Align predictions to GT timestamps with optional room-aware matching."""
        tolerance = pd.Timedelta(seconds=float(self.config.timestamp_tolerance_seconds))

        gt_cols = ["timestamp", gt_label_col]
        pred_cols = ["timestamp", pred_label_col]
        use_room = "room" in ground_truth_df.columns and "room" in predictions_df.columns
        if use_room:
            gt_cols.append("room")
            pred_cols.append("room")

        gt = ground_truth_df[gt_cols].copy()
        pred = predictions_df[pred_cols].copy()
        gt["timestamp"] = pd.to_datetime(gt["timestamp"], errors="coerce")
        pred["timestamp"] = pd.to_datetime(pred["timestamp"], errors="coerce")
        gt = gt.dropna(subset=["timestamp"]).sort_values("timestamp")
        pred = pred.dropna(subset=["timestamp"]).sort_values("timestamp")

        merge_kwargs: Dict[str, Any] = {
            "on": "timestamp",
            "direction": "nearest",
            "tolerance": tolerance,
            "suffixes": ("_gt", "_pred"),
        }
        if use_room:
            merge_kwargs["by"] = "room"

        aligned = pd.merge_asof(gt, pred, **merge_kwargs)
        aligned = aligned.dropna(subset=[pred_label_col])
        return aligned
    
    def calculate_metrics(
        self,
        predictions_df: pd.DataFrame,
        ground_truth_df: Optional[pd.DataFrame] = None,
    ) -> EventKPIMetrics:
        """
        Calculate all event-level KPI metrics.
        
        Args:
            predictions_df: DataFrame with predicted events
            ground_truth_df: Optional DataFrame with ground truth labels
            
        Returns:
            EventKPIMetrics with all computed metrics
        """
        metrics = EventKPIMetrics()
        
        # Calculate unknown rates
        metrics.unknown_rate_global = self._calculate_unknown_rate(predictions_df)
        metrics.unknown_rate_per_room = self._calculate_unknown_rate_per_room(predictions_df)
        
        # Calculate home-empty metrics if ground truth available
        if ground_truth_df is not None:
            home_empty_metrics = self._calculate_home_empty_metrics(predictions_df, ground_truth_df)
            metrics.home_empty_precision = home_empty_metrics.get("precision", 0.0)
            metrics.home_empty_recall = home_empty_metrics.get("recall", 0.0)
            metrics.home_empty_false_empty_rate = home_empty_metrics.get("false_empty_rate", 0.0)
            
            # Calculate per-event metrics
            event_metrics = self._calculate_per_event_metrics(predictions_df, ground_truth_df)
            metrics.event_recalls = event_metrics.get("recalls", {})
            metrics.event_precisions = event_metrics.get("precisions", {})
            metrics.event_f1s = event_metrics.get("f1s", {})
            metrics.event_supports = event_metrics.get("supports", {})
        
        # Calculate care KPIs
        care_kpis = self._calculate_care_kpis(predictions_df)
        metrics.sleep_hours = care_kpis.get("sleep_hours", 0.0)
        metrics.sleep_efficiency = care_kpis.get("sleep_efficiency", 0.0)
        metrics.bathroom_visits = care_kpis.get("bathroom_visits", 0)
        metrics.shower_detected = care_kpis.get("shower_detected", False)
        metrics.meal_count = care_kpis.get("meal_count", 0)
        metrics.out_time_minutes = care_kpis.get("out_time_minutes", 0.0)
        
        return metrics
    
    def calculate_daily_metrics(
        self,
        predictions_df: pd.DataFrame,
        target_date: date,
        ground_truth_df: Optional[pd.DataFrame] = None,
    ) -> EventKPIMetrics:
        """
        Calculate metrics for a specific day.
        
        Args:
            predictions_df: DataFrame with predicted events
            target_date: Date to filter for
            ground_truth_df: Optional ground truth DataFrame
            
        Returns:
            EventKPIMetrics for the day
        """
        # Filter predictions to target date
        pred_df = predictions_df.copy()
        pred_df["date"] = pd.to_datetime(pred_df["timestamp"]).dt.date
        day_pred_df = pred_df[pred_df["date"] == target_date]
        
        # Filter ground truth if provided
        day_gt_df = None
        if ground_truth_df is not None:
            gt_df = ground_truth_df.copy()
            gt_df["date"] = pd.to_datetime(gt_df["timestamp"]).dt.date
            day_gt_df = gt_df[gt_df["date"] == target_date]
        
        return self.calculate_metrics(day_pred_df, day_gt_df)
    
    def _calculate_unknown_rate(self, predictions_df: pd.DataFrame) -> float:
        """Calculate global unknown rate."""
        if predictions_df.empty:
            return 0.0
        label_col = self._resolve_label_column(predictions_df)
        unknown_count = (predictions_df[label_col] == "unknown").sum()
        return unknown_count / len(predictions_df)
    
    def _calculate_unknown_rate_per_room(self, predictions_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate unknown rate per room."""
        if predictions_df.empty or "room" not in predictions_df.columns:
            return {}
        label_col = self._resolve_label_column(predictions_df)
        
        rates = {}
        for room in predictions_df["room"].unique():
            room_df = predictions_df[predictions_df["room"] == room]
            unknown_count = (room_df[label_col] == "unknown").sum()
            rates[room] = unknown_count / len(room_df) if len(room_df) > 0 else 0.0
        
        return rates
    
    def _calculate_home_empty_metrics(
        self,
        predictions_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Calculate home-empty specific metrics.
        
        Home-empty is defined as all rooms being unoccupied simultaneously.
        """
        metrics = {}
        
        # Check if we have room information
        if "room" not in predictions_df.columns:
            return metrics
        pred_label_col = self._resolve_label_column(predictions_df)
        gt_label_col = self._resolve_label_column(ground_truth_df)

        # Aggregate by timestamp to get home-level occupancy
        pred_home_empty = self._get_home_empty_status(predictions_df, pred_label_col)
        gt_home_empty = self._get_home_empty_status(ground_truth_df, gt_label_col)
        
        if not pred_home_empty or not gt_home_empty:
            return metrics
        
        pred_df = pd.DataFrame(list(pred_home_empty.items()), columns=["timestamp", "pred_empty"])
        gt_df = pd.DataFrame(list(gt_home_empty.items()), columns=["timestamp", "gt_empty"])
        pred_df = pred_df.sort_values("timestamp")
        gt_df = gt_df.sort_values("timestamp")
        tolerance = pd.Timedelta(seconds=float(self.config.timestamp_tolerance_seconds))
        merged = pd.merge_asof(
            gt_df,
            pred_df,
            on="timestamp",
            direction="nearest",
            tolerance=tolerance,
        ).dropna(subset=["pred_empty"])
        
        if merged.empty:
            return metrics
        
        y_true = merged["gt_empty"].astype(int)
        y_pred = merged["pred_empty"].astype(int)
        
        # Calculate metrics
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)

        # False empty rate: predicted empty when actually occupied
        false_empty = ((y_pred == 1) & (y_true == 0)).sum()
        total_occupied = (y_true == 0).sum()
        metrics["false_empty_rate"] = false_empty / total_occupied if total_occupied > 0 else 0.0
        
        return metrics
    
    def _get_home_empty_status(self, df: pd.DataFrame, label_col: str) -> Dict[datetime, bool]:
        """
        Determine home-empty status for each timestamp.
        
        Home is empty when ALL rooms report unoccupied.
        """
        if df.empty or "room" not in df.columns:
            return {}
        
        # Group by timestamp
        status = {}
        for timestamp, group in df.groupby("timestamp"):
            # Home is empty if all rooms are unoccupied
            rooms = group["room"].unique()
            unoccupied_rooms = group[group[label_col] == "unoccupied"]["room"].unique()
            status[timestamp] = len(unoccupied_rooms) == len(rooms) and len(rooms) > 0
        
        return status
    
    def _calculate_per_event_metrics(
        self,
        predictions_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate precision/recall/F1 for each event type."""
        metrics = {
            "recalls": {},
            "precisions": {},
            "f1s": {},
            "supports": {},
        }
        
        if predictions_df.empty or ground_truth_df.empty:
            return metrics
        pred_label_col = self._resolve_label_column(predictions_df)
        gt_label_col = self._resolve_label_column(ground_truth_df)

        # Get all unique labels
        all_labels = set(predictions_df[pred_label_col].unique()) | set(ground_truth_df[gt_label_col].unique())
        all_labels.discard("unknown")
        all_labels.discard("unoccupied")

        aligned = self._asof_align(
            ground_truth_df,
            predictions_df,
            gt_label_col=gt_label_col,
            pred_label_col=pred_label_col,
        )
        if aligned.empty:
            return metrics

        y_true = aligned[gt_label_col]
        y_pred = aligned[pred_label_col]
        
        # Calculate metrics for each label
        for label in all_labels:
            # Binary classification for this label
            y_true_binary = (y_true == label).astype(int)
            y_pred_binary = (y_pred == label).astype(int)
            
            support = y_true_binary.sum()
            
            if support == 0:
                continue
            
            metrics["recalls"][label] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            metrics["precisions"][label] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            metrics["f1s"][label] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            metrics["supports"][label] = int(support)
        
        return metrics
    
    def _calculate_care_kpis(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate care-relevant KPIs from predictions."""
        if predictions_df.empty:
            return {}
        
        kpis = {}
        
        # Convert to episode format
        episodes = self._predictions_to_episodes(predictions_df)
        
        # Extract room-specific episodes
        room_episodes = {}
        if "room" in predictions_df.columns:
            for room in predictions_df["room"].unique():
                room_df = predictions_df[predictions_df["room"] == room]
                room_episodes[room] = self._predictions_to_episodes(room_df)
        else:
            room_episodes["unknown"] = episodes
        
        # Calculate care KPIs using DerivedEventCalculator
        from ml.derived_events import DerivedEventConfig
        config = DerivedEventConfig()
        extractor = CareKPIExtractor(config)
        
        # Get the date from predictions
        if "timestamp" in predictions_df.columns:
            sample_date = pd.to_datetime(predictions_df["timestamp"].iloc[0]).date()
        else:
            sample_date = date.today()
        
        try:
            care_data = extractor.extract_day_kpis(room_episodes, sample_date)
        except Exception:
            logger.exception("Failed to extract care KPIs from episode predictions")
            return kpis

        kpis["sleep_hours"] = care_data.sleep_hours
        kpis["sleep_efficiency"] = care_data.sleep_efficiency or 0.0
        kpis["bathroom_visits"] = care_data.bathroom_visit_count
        kpis["shower_detected"] = care_data.shower_detected
        kpis["meal_count"] = care_data.meal_count
        kpis["out_time_minutes"] = care_data.out_time_minutes
        
        return kpis
    
    def _predictions_to_episodes(self, predictions_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert prediction DataFrame to episode list format."""
        if predictions_df.empty:
            return []
        
        episodes = []
        label_col = self._resolve_label_column(predictions_df)
        
        # Group consecutive same-label predictions
        df = predictions_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        if df.empty:
            return []
        deltas = df["timestamp"].diff().dt.total_seconds().dropna()
        step_seconds = float(deltas.median()) if not deltas.empty else 10.0
        if not np.isfinite(step_seconds) or step_seconds <= 0:
            step_seconds = 10.0
        current_label = None
        current_start = None
        current_end = None
        
        for _, row in df.iterrows():
            label = row[label_col]
            ts = pd.to_datetime(row["timestamp"])
            
            if label != current_label:
                # Save previous episode
                if current_label is not None:
                    duration_seconds = max(
                        step_seconds,
                        (current_end - current_start).total_seconds() + step_seconds,
                    )
                    episodes.append({
                        "event_label": current_label,
                        "start_time": current_start,
                        "end_time": current_end,
                        "duration_seconds": duration_seconds,
                    })
                
                current_label = label
                current_start = ts
                current_end = ts
            else:
                current_end = ts
        
        # Save final episode
        if current_label is not None:
            duration_seconds = max(
                step_seconds,
                (current_end - current_start).total_seconds() + step_seconds,
            )
            episodes.append({
                "event_label": current_label,
                "start_time": current_start,
                "end_time": current_end,
                "duration_seconds": duration_seconds,
            })
        
        return episodes


def calculate_event_kpis_from_episodes(
    episodes_df: pd.DataFrame,
    ground_truth_df: Optional[pd.DataFrame] = None,
) -> EventKPIMetrics:
    """
    Convenience function to calculate KPIs from episodes.
    
    Args:
        episodes_df: DataFrame with compiled episodes
        ground_truth_df: Optional ground truth for accuracy metrics
        
    Returns:
        EventKPIMetrics
    """
    calculator = EventKPICalculator()
    return calculator.calculate_metrics(episodes_df, ground_truth_df)
