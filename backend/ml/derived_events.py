"""
Derived Events Module

Computes care-relevant KPIs and metrics from canonical event episodes.
Part of PR-B1: Event Compiler + KPI/Gates.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

CANONICAL_ROOM_LIVING = "livingroom"
LEGACY_ROOM_LIVING = "living_room"

CANONICAL_SLEEP_EVENTS = {"sleeping"}
CANONICAL_BATHROOM_VISIT_EVENTS = {"bathroom_use", "toileting", "bathing", "grooming"}
CANONICAL_SHOWER_EVENTS = {"showering", "shower", "bath"}
CANONICAL_COOKING_EVENTS = {"cooking"}
CANONICAL_EATING_EVENTS = {"eating"}
CANONICAL_OUT_EVENTS = {"unoccupied"}


@dataclass
class DerivedEventConfig:
    """Configuration for derived event calculation."""
    
    # Sleep detection parameters
    min_sleep_duration_hours: float = 2.0
    max_sleep_duration_hours: float = 14.0
    max_sleep_start_hour: int = 20  # 8 PM
    min_sleep_end_hour: int = 4  # 4 AM
    
    # Activity thresholds
    min_activity_duration_minutes: float = 5.0
    max_gap_within_activity_minutes: float = 10.0
    
    # Shower detection
    shower_min_duration_minutes: float = 5.0
    shower_max_duration_minutes: float = 45.0
    
    # Out time calculation
    min_out_time_minutes: float = 10.0
    
    # Confidence thresholds
    min_confidence_for_kpi: float = 0.5
    
    def validate(self) -> None:
        """Validate configuration."""
        assert self.min_sleep_duration_hours < self.max_sleep_duration_hours, \
            "min_sleep_duration_hours must be < max_sleep_duration_hours"
        assert 0 <= self.max_sleep_start_hour <= 23, \
            "max_sleep_start_hour must be in [0, 23]"
        assert 0 <= self.min_sleep_end_hour <= 23, \
            "min_sleep_end_hour must be in [0, 23]"


@dataclass
class SleepMetrics:
    """Sleep-related metrics."""
    
    total_sleep_minutes: float
    sleep_start_time: Optional[datetime] = None
    sleep_end_time: Optional[datetime] = None
    bed_entry_time: Optional[datetime] = None
    bed_exit_time: Optional[datetime] = None
    sleep_efficiency: Optional[float] = None
    interruption_count: int = 0
    sleep_quality_score: Optional[float] = None
    
    @property
    def sleep_hours(self) -> float:
        """Sleep duration in hours."""
        return self.total_sleep_minutes / 60.0
    
    @property
    def time_in_bed_minutes(self) -> float:
        """Time in bed (from entry to exit)."""
        if self.bed_entry_time and self.bed_exit_time:
            return (self.bed_exit_time - self.bed_entry_time).total_seconds() / 60.0
        return self.total_sleep_minutes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_sleep_minutes": self.total_sleep_minutes,
            "sleep_hours": self.sleep_hours,
            "sleep_start_time": self.sleep_start_time.isoformat() if self.sleep_start_time else None,
            "sleep_end_time": self.sleep_end_time.isoformat() if self.sleep_end_time else None,
            "bed_entry_time": self.bed_entry_time.isoformat() if self.bed_entry_time else None,
            "bed_exit_time": self.bed_exit_time.isoformat() if self.bed_exit_time else None,
            "sleep_efficiency": self.sleep_efficiency,
            "interruption_count": self.interruption_count,
            "sleep_quality_score": self.sleep_quality_score,
        }


@dataclass
class RoomUsageProfile:
    """Profile of room usage patterns."""
    
    room_name: str
    total_time_minutes: float
    episode_count: int
    unique_events: Set[str] = field(default_factory=set)
    hourly_distribution: Dict[int, float] = field(default_factory=dict)
    
    @property
    def total_time_hours(self) -> float:
        """Total time in hours."""
        return self.total_time_minutes / 60.0
    
    @property
    def event_frequency(self) -> float:
        """Events per hour."""
        if self.total_time_hours > 0:
            return self.episode_count / self.total_time_hours
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "room_name": self.room_name,
            "total_time_minutes": self.total_time_minutes,
            "total_time_hours": self.total_time_hours,
            "episode_count": self.episode_count,
            "unique_events": list(self.unique_events),
            "event_frequency": self.event_frequency,
            "hourly_distribution": self.hourly_distribution,
        }


@dataclass
class CareKPIData:
    """Care-relevant KPIs for a single day."""
    
    date: date
    
    # Sleep metrics
    total_sleep_minutes: float = 0.0
    sleep_start_time: Optional[datetime] = None
    sleep_end_time: Optional[datetime] = None
    bed_entry_time: Optional[datetime] = None
    bed_exit_time: Optional[datetime] = None
    sleep_efficiency: Optional[float] = None
    sleep_interruptions: int = 0
    
    # Bathroom metrics
    bathroom_visit_count: int = 0
    bathroom_total_minutes: float = 0.0
    shower_detected: bool = False
    shower_duration_minutes: Optional[float] = None
    
    # Kitchen metrics
    meal_count: int = 0
    cooking_count: int = 0
    eating_count: int = 0
    kitchen_active_minutes: float = 0.0
    
    # Living room metrics
    livingroom_active_minutes: float = 0.0
    sedentary_minutes: float = 0.0
    
    # Out time
    out_time_minutes: float = 0.0
    out_episode_count: int = 0
    
    # Activity diversity
    unique_activities: int = 0
    activity_transitions: int = 0
    
    # Quality metrics
    unknown_ratio: float = 0.0
    confidence_mean: float = 0.0
    confidence_std: float = 0.0
    
    # Per-room KPIs
    room_kpis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @property
    def sleep_hours(self) -> float:
        """Sleep duration in hours."""
        return self.total_sleep_minutes / 60.0
    
    @property
    def has_showered(self) -> bool:
        """Whether a shower was detected."""
        return self.shower_detected
    
    @property
    def is_day_valid(self) -> bool:
        """Check if day has valid data (low unknown ratio)."""
        return self.unknown_ratio < 0.15
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat(),
            "sleep_hours": self.sleep_hours,
            "sleep_start_time": self.sleep_start_time.isoformat() if self.sleep_start_time else None,
            "sleep_end_time": self.sleep_end_time.isoformat() if self.sleep_end_time else None,
            "bathroom_visit_count": self.bathroom_visit_count,
            "shower_detected": self.shower_detected,
            "meal_count": self.meal_count,
            "out_time_minutes": self.out_time_minutes,
            "unknown_ratio": self.unknown_ratio,
            "is_day_valid": self.is_day_valid,
        }


@dataclass
class DayStatistics:
    """Statistics aggregated over multiple days."""
    
    daily_kpis: List[CareKPIData]
    
    @property
    def total_days(self) -> int:
        """Total number of days."""
        return len(self.daily_kpis)
    
    @property
    def valid_days(self) -> int:
        """Number of valid days."""
        return sum(1 for kpi in self.daily_kpis if kpi.is_day_valid)
    
    @property
    def average_sleep_hours(self) -> float:
        """Average sleep hours across all days."""
        if not self.daily_kpis:
            return 0.0
        return np.mean([kpi.sleep_hours for kpi in self.daily_kpis])
    
    @property
    def sleep_hours_std(self) -> float:
        """Standard deviation of sleep hours."""
        if len(self.daily_kpis) < 2:
            return 0.0
        return np.std([kpi.sleep_hours for kpi in self.daily_kpis])
    
    @property
    def shower_days(self) -> int:
        """Number of days with shower detected."""
        return sum(1 for kpi in self.daily_kpis if kpi.has_showered)
    
    @property
    def average_meals_per_day(self) -> float:
        """Average meals per day."""
        if not self.daily_kpis:
            return 0.0
        return np.mean([kpi.meal_count for kpi in self.daily_kpis])
    
    def get_trend(self, metric: str, window_days: int = 7) -> List[float]:
        """Get trend for a specific metric."""
        sorted_kpis = sorted(self.daily_kpis, key=lambda k: k.date)
        values = []
        
        for kpi in sorted_kpis:
            if hasattr(kpi, metric):
                values.append(getattr(kpi, metric))
            elif metric in kpi.room_kpis:
                values.append(kpi.room_kpis[metric])
            else:
                values.append(0.0)
        
        return values
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_days": self.total_days,
            "valid_days": self.valid_days,
            "average_sleep_hours": self.average_sleep_hours,
            "sleep_hours_std": self.sleep_hours_std,
            "shower_days": self.shower_days,
            "average_meals_per_day": self.average_meals_per_day,
        }


class CareKPIExtractor:
    """Extracts care-relevant KPIs from episode data."""
    
    def __init__(self, config: Optional[DerivedEventConfig] = None):
        self.config = config or DerivedEventConfig()
        self.config.validate()
    
    def extract_sleep_metrics(
        self,
        episodes: List[Dict[str, Any]],
        target_date: date,
    ) -> SleepMetrics:
        """Extract sleep metrics from bedroom episodes."""
        sleep_episodes = [
            ep for ep in episodes
            if ep.get("event_label") in CANONICAL_SLEEP_EVENTS
        ]
        
        if not sleep_episodes:
            return SleepMetrics(total_sleep_minutes=0.0)
        
        # Sort by start time
        sleep_episodes.sort(key=lambda x: x.get("start_time", datetime.min))
        
        total_minutes = sum(
            ep.get("duration_seconds", 0) / 60.0
            for ep in sleep_episodes
        )
        
        # Get sleep start/end times
        sleep_start = sleep_episodes[0].get("start_time")
        sleep_end = sleep_episodes[-1].get("end_time")
        
        # Calculate sleep efficiency
        if len(sleep_episodes) == 1:
            sleep_efficiency = 1.0
        else:
            time_in_bed = (sleep_end - sleep_start).total_seconds() / 60.0 if sleep_start and sleep_end else total_minutes
            sleep_efficiency = total_minutes / time_in_bed if time_in_bed > 0 else 1.0
        
        # Count interruptions (gaps between sleep episodes)
        interruption_count = max(0, len(sleep_episodes) - 1)
        
        return SleepMetrics(
            total_sleep_minutes=total_minutes,
            sleep_start_time=sleep_start,
            sleep_end_time=sleep_end,
            sleep_efficiency=sleep_efficiency,
            interruption_count=interruption_count,
        )
    
    def extract_bathroom_metrics(
        self,
        episodes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract bathroom usage metrics."""
        # Count visits
        visits = [
            ep for ep in episodes
            if ep.get("event_label") in CANONICAL_BATHROOM_VISIT_EVENTS
        ]
        
        # Detect shower
        showers = [
            ep for ep in episodes
            if ep.get("event_label") in CANONICAL_SHOWER_EVENTS
        ]
        
        shower_duration = None
        if showers:
            shower_duration = sum(
                ep.get("duration_seconds", 0) / 60.0
                for ep in showers
            )
        
        total_minutes = sum(
            ep.get("duration_seconds", 0) / 60.0
            for ep in visits
        )
        
        return {
            "visit_count": len(visits),
            "total_minutes": total_minutes,
            "has_showered": len(showers) > 0,
            "shower_count": len(showers),
            "shower_duration_minutes": shower_duration,
        }
    
    def extract_kitchen_metrics(
        self,
        episodes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract kitchen activity metrics."""
        cooking = [ep for ep in episodes if ep.get("event_label") in CANONICAL_COOKING_EVENTS]
        eating = [ep for ep in episodes if ep.get("event_label") in CANONICAL_EATING_EVENTS]
        
        total_active = sum(
            ep.get("duration_seconds", 0) / 60.0
            for ep in cooking + eating
        )
        
        return {
            "cooking_count": len(cooking),
            "eating_count": len(eating),
            "meal_count": len(cooking) + len(eating),
            "active_minutes": total_active,
        }
    
    def extract_out_time(
        self,
        episodes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract out time metrics."""
        unoccupied = [
            ep for ep in episodes
            if ep.get("event_label") in CANONICAL_OUT_EVENTS
        ]
        
        total_minutes = sum(
            ep.get("duration_seconds", 0) / 60.0
            for ep in unoccupied
        )
        
        return {
            "total_minutes": total_minutes,
            "episode_count": len(unoccupied),
        }
    
    def extract_room_usage_profile(
        self,
        room_name: str,
        episodes: List[Dict[str, Any]],
    ) -> RoomUsageProfile:
        """Extract room usage profile."""
        total_minutes = sum(
            ep.get("duration_seconds", 0) / 60.0
            for ep in episodes
        )
        
        unique_events = set(
            ep.get("event_label") for ep in episodes
        )
        
        # Calculate hourly distribution
        hourly = {h: 0.0 for h in range(24)}
        for ep in episodes:
            start = ep.get("start_time")
            end = ep.get("end_time")
            duration = ep.get("duration_seconds", 0) / 60.0
            
            if start and end:
                hour = start.hour
                hourly[hour] = hourly.get(hour, 0.0) + duration
        
        return RoomUsageProfile(
            room_name=room_name,
            total_time_minutes=total_minutes,
            episode_count=len(episodes),
            unique_events=unique_events,
            hourly_distribution=hourly,
        )
    
    def extract_day_kpis(
        self,
        room_episodes: Dict[str, List[Dict[str, Any]]],
        target_date: date,
    ) -> CareKPIData:
        """Extract all KPIs for a single day."""
        kpi = CareKPIData(date=target_date)
        
        # Bedroom / Sleep metrics
        bedroom_eps = room_episodes.get("bedroom", [])
        sleep_metrics = self.extract_sleep_metrics(bedroom_eps, target_date)
        kpi.total_sleep_minutes = sleep_metrics.total_sleep_minutes
        kpi.sleep_start_time = sleep_metrics.sleep_start_time
        kpi.sleep_end_time = sleep_metrics.sleep_end_time
        kpi.sleep_efficiency = sleep_metrics.sleep_efficiency
        kpi.sleep_interruptions = sleep_metrics.interruption_count
        
        # Bathroom metrics
        bathroom_eps = room_episodes.get("bathroom", [])
        bath_metrics = self.extract_bathroom_metrics(bathroom_eps)
        kpi.bathroom_visit_count = bath_metrics["visit_count"]
        kpi.bathroom_total_minutes = bath_metrics["total_minutes"]
        kpi.shower_detected = bath_metrics["has_showered"]
        kpi.shower_duration_minutes = bath_metrics["shower_duration_minutes"]
        
        # Kitchen metrics
        kitchen_eps = room_episodes.get("kitchen", [])
        kit_metrics = self.extract_kitchen_metrics(kitchen_eps)
        kpi.meal_count = kit_metrics["meal_count"]
        kpi.cooking_count = kit_metrics["cooking_count"]
        kpi.eating_count = kit_metrics["eating_count"]
        kpi.kitchen_active_minutes = kit_metrics["active_minutes"]
        
        # Living room metrics
        living_eps = self._collect_room_episodes(
            room_episodes, CANONICAL_ROOM_LIVING, LEGACY_ROOM_LIVING
        )
        kpi.livingroom_active_minutes = sum(
            ep.get("duration_seconds", 0) / 60.0
            for ep in living_eps
            if ep.get("event_label") in ("watching_tv", "reading", "relaxing")
        )
        
        # Out time from all rooms (home-empty periods)
        all_eps = []
        for eps in room_episodes.values():
            all_eps.extend(eps)
        
        out_metrics = self.extract_out_time(all_eps)
        kpi.out_time_minutes = out_metrics["total_minutes"]
        kpi.out_episode_count = out_metrics["episode_count"]
        
        # Activity diversity
        all_labels = [ep.get("event_label") for ep in all_eps]
        kpi.unique_activities = len(set(all_labels))
        
        # Calculate transitions
        kpi.activity_transitions = self._count_transitions(all_eps)
        
        # Unknown ratio
        unknown_count = sum(1 for label in all_labels if label == "unknown")
        kpi.unknown_ratio = unknown_count / len(all_labels) if all_labels else 0.0
        
        # Room-specific KPIs
        for room_name, eps in room_episodes.items():
            kpi.room_kpis[room_name] = self.extract_room_usage_profile(room_name, eps).to_dict()
        
        return kpi
    
    def calculate_statistics(
        self,
        all_episodes: Dict[date, Dict[str, List[Dict[str, Any]]]],
    ) -> DayStatistics:
        """Calculate statistics over multiple days."""
        daily_kpis = []
        
        for target_date, room_episodes in all_episodes.items():
            kpi = self.extract_day_kpis(room_episodes, target_date)
            daily_kpis.append(kpi)
        
        return DayStatistics(daily_kpis=daily_kpis)
    
    def _count_transitions(self, episodes: List[Dict[str, Any]]) -> int:
        """Count activity transitions in episode list."""
        if len(episodes) < 2:
            return 0
        
        sorted_episodes = sorted(episodes, key=self._episode_sort_key)
        transitions = 0
        prev_label = sorted_episodes[0].get("event_label")
        
        for ep in sorted_episodes[1:]:
            label = ep.get("event_label")
            if label != prev_label:
                transitions += 1
                prev_label = label
        
        return transitions
    
    @staticmethod
    def _collect_room_episodes(
        room_episodes: Dict[str, List[Dict[str, Any]]],
        canonical_room: str,
        legacy_room: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Collect episodes for canonical room and optional legacy alias."""
        episodes = list(room_episodes.get(canonical_room, []))
        if legacy_room and legacy_room != canonical_room:
            episodes.extend(room_episodes.get(legacy_room, []))
        return episodes
    
    @staticmethod
    def _episode_sort_key(episode: Dict[str, Any]) -> float:
        """Sort key using episode start time, robust to string/datetime inputs."""
        start_time = episode.get("start_time")
        if start_time is None:
            return float("-inf")
        
        parsed = pd.to_datetime(start_time, utc=True, errors="coerce")
        if pd.isna(parsed):
            return float("-inf")
        return float(parsed.value)


def calculate_sleep_efficiency(
    total_sleep_minutes: float,
    time_in_bed_minutes: float,
) -> float:
    """Calculate sleep efficiency ratio."""
    if time_in_bed_minutes <= 0:
        return 0.0
    efficiency = total_sleep_minutes / time_in_bed_minutes
    return min(1.0, max(0.0, efficiency))


def aggregate_weekly_stats(
    daily_kpis: List[CareKPIData],
) -> List[Dict[str, Any]]:
    """Aggregate daily KPIs into weekly statistics."""
    if not daily_kpis:
        return []
    
    # Sort by date
    sorted_kpis = sorted(daily_kpis, key=lambda k: k.date)
    
    weekly_stats = []
    current_week = []
    
    for kpi in sorted_kpis:
        if not current_week:
            current_week.append(kpi)
        else:
            # Check if same week
            days_diff = (kpi.date - current_week[0].date).days
            if days_diff < 7:
                current_week.append(kpi)
            else:
                # Aggregate current week
                weekly_stats.append(_aggregate_week(current_week))
                current_week = [kpi]
    
    # Handle final week
    if current_week:
        weekly_stats.append(_aggregate_week(current_week))
    
    return weekly_stats


def _aggregate_week(kpis: List[CareKPIData]) -> Dict[str, Any]:
    """Aggregate a list of daily KPIs into weekly stats."""
    if not kpis:
        return {}
    
    start_date = min(k.date for k in kpis)
    end_date = max(k.date for k in kpis)
    
    sleep_hours = [k.sleep_hours for k in kpis]
    meals = [k.meal_count for k in kpis]
    showers = [1 if k.has_showered else 0 for k in kpis]
    
    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "days_count": len(kpis),
        "avg_sleep_hours": np.mean(sleep_hours) if sleep_hours else 0.0,
        "sleep_hours_std": np.std(sleep_hours) if len(sleep_hours) > 1 else 0.0,
        "avg_meals_per_day": np.mean(meals) if meals else 0.0,
        "shower_days": sum(showers),
        "total_out_time_hours": sum(k.out_time_minutes for k in kpis) / 60.0,
    }
