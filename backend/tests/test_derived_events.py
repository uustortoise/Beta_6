"""
Tests for PR-B1: Derived Events

Tests care-relevant event derivation from canonical episodes.
"""

import sys
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.derived_events import (
    CareKPIData,
    CareKPIExtractor,
    DayStatistics,
    DerivedEventConfig,
    RoomUsageProfile,
    SleepMetrics,
    aggregate_weekly_stats,
    calculate_sleep_efficiency,
)


class TestDerivedEventConfig(unittest.TestCase):
    """Tests for DerivedEventConfig."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = DerivedEventConfig(
            min_sleep_duration_hours=2.0,
            max_sleep_duration_hours=14.0,
            max_sleep_start_hour=20,
            min_sleep_end_hour=4,
        )
        
        config.validate()  # Should not raise
    
    def test_invalid_sleep_duration_range(self):
        """Test invalid sleep duration range."""
        config = DerivedEventConfig(
            min_sleep_duration_hours=10.0,
            max_sleep_duration_hours=8.0,  # Should be > min
        )
        
        with self.assertRaises(AssertionError):
            config.validate()


class TestCareKPIData(unittest.TestCase):
    """Tests for CareKPIData dataclass."""
    
    def test_kpi_data_creation(self):
        """Test creating KPI data."""
        kpi = CareKPIData(
            date=date(2026, 2, 1),
            total_sleep_minutes=420,
            bed_entry_time=datetime(2026, 2, 1, 22, 0, 0),
            bed_exit_time=datetime(2026, 2, 2, 5, 0, 0),
            bathroom_visit_count=3,
            bathroom_total_minutes=15,
            shower_detected=True,
            meal_count=3,
            out_time_minutes=120,
            unknown_ratio=0.05,
        )
        
        self.assertEqual(kpi.date, date(2026, 2, 1))
        self.assertEqual(kpi.sleep_hours, 7.0)
        self.assertTrue(kpi.has_showered)


class TestCareKPIExtractor(unittest.TestCase):
    """Tests for CareKPIExtractor."""
    
    def test_extract_sleep_metrics(self):
        """Test extracting sleep metrics."""
        extractor = CareKPIExtractor()
        
        episodes = [
            {
                "event_label": "sleeping",
                "start_time": datetime(2026, 2, 1, 22, 0, 0),
                "end_time": datetime(2026, 2, 2, 5, 0, 0),
                "duration_seconds": 25200,  # 7 hours
            }
        ]
        
        metrics = extractor.extract_sleep_metrics(episodes, date(2026, 2, 1))
        
        self.assertIsInstance(metrics, SleepMetrics)
        self.assertEqual(metrics.total_sleep_minutes, 420)
        self.assertEqual(metrics.sleep_hours, 7.0)
    
    def test_extract_bathroom_metrics(self):
        """Test extracting bathroom metrics."""
        extractor = CareKPIExtractor()
        
        episodes = [
            {
                "event_label": "toileting",
                "start_time": datetime(2026, 2, 1, 8, 0, 0),
                "end_time": datetime(2026, 2, 1, 8, 5, 0),
                "duration_seconds": 300,
            },
            {
                "event_label": "showering",
                "start_time": datetime(2026, 2, 1, 9, 0, 0),
                "end_time": datetime(2026, 2, 1, 9, 15, 0),
                "duration_seconds": 900,
            },
        ]
        
        metrics = extractor.extract_bathroom_metrics(episodes)
        
        self.assertEqual(metrics["visit_count"], 1)  # toileting only, showering is separate
        self.assertEqual(metrics["total_minutes"], 5)  # only toileting duration
        self.assertTrue(metrics["has_showered"])
    
    def test_extract_bathroom_metrics_with_canonical_label(self):
        """Canonical bathroom_use label must count as a bathroom visit."""
        extractor = CareKPIExtractor()
        
        episodes = [
            {
                "event_label": "bathroom_use",
                "start_time": datetime(2026, 2, 1, 8, 0, 0),
                "end_time": datetime(2026, 2, 1, 8, 4, 0),
                "duration_seconds": 240,
            }
        ]
        
        metrics = extractor.extract_bathroom_metrics(episodes)
        
        self.assertEqual(metrics["visit_count"], 1)
        self.assertEqual(metrics["total_minutes"], 4)
    
    def test_extract_kitchen_metrics(self):
        """Test extracting kitchen metrics."""
        extractor = CareKPIExtractor()
        
        episodes = [
            {
                "event_label": "cooking",
                "start_time": datetime(2026, 2, 1, 7, 0, 0),
                "end_time": datetime(2026, 2, 1, 7, 30, 0),
                "duration_seconds": 1800,
            },
            {
                "event_label": "eating",
                "start_time": datetime(2026, 2, 1, 12, 0, 0),
                "end_time": datetime(2026, 2, 1, 12, 30, 0),
                "duration_seconds": 1800,
            },
        ]
        
        metrics = extractor.extract_kitchen_metrics(episodes)
        
        self.assertEqual(metrics["cooking_count"], 1)
        self.assertEqual(metrics["eating_count"], 1)
        self.assertEqual(metrics["meal_count"], 2)
    
    def test_extract_out_time(self):
        """Test extracting out time metrics."""
        extractor = CareKPIExtractor()
        
        episodes = [
            {
                "event_label": "unoccupied",
                "start_time": datetime(2026, 2, 1, 10, 0, 0),
                "end_time": datetime(2026, 2, 1, 12, 0, 0),
                "duration_seconds": 7200,
            }
        ]
        
        metrics = extractor.extract_out_time(episodes)
        
        self.assertEqual(metrics["total_minutes"], 120)
        self.assertEqual(metrics["episode_count"], 1)
    
    def test_extract_full_day_kpis(self):
        """Test extracting full day KPIs."""
        extractor = CareKPIExtractor()
        
        room_episodes = {
            "bedroom": [
                {
                    "event_label": "sleeping",
                    "start_time": datetime(2026, 2, 1, 22, 0, 0),
                    "end_time": datetime(2026, 2, 2, 5, 0, 0),
                    "duration_seconds": 25200,
                }
            ],
            "bathroom": [
                {
                    "event_label": "showering",
                    "start_time": datetime(2026, 2, 1, 9, 0, 0),
                    "end_time": datetime(2026, 2, 1, 9, 15, 0),
                    "duration_seconds": 900,
                }
            ],
            "kitchen": [
                {
                    "event_label": "cooking",
                    "start_time": datetime(2026, 2, 1, 7, 0, 0),
                    "end_time": datetime(2026, 2, 1, 7, 30, 0),
                    "duration_seconds": 1800,
                }
            ],
        }
        
        kpis = extractor.extract_day_kpis(room_episodes, date(2026, 2, 1))
        
        self.assertIsInstance(kpis, CareKPIData)
        self.assertEqual(kpis.sleep_hours, 7.0)
        self.assertTrue(kpis.shower_detected)
        self.assertEqual(kpis.meal_count, 1)
    
    def test_extract_full_day_kpis_accepts_canonical_livingroom_key(self):
        """livingroom canonical room key must be consumed for activity KPIs."""
        extractor = CareKPIExtractor()
        
        room_episodes = {
            "livingroom": [
                {
                    "event_label": "relaxing",
                    "start_time": datetime(2026, 2, 1, 20, 0, 0),
                    "end_time": datetime(2026, 2, 1, 20, 30, 0),
                    "duration_seconds": 1800,
                }
            ]
        }
        
        kpis = extractor.extract_day_kpis(room_episodes, date(2026, 2, 1))
        
        self.assertEqual(kpis.livingroom_active_minutes, 30)
    
    def test_extract_room_usage_profile(self):
        """Test extracting room usage profile."""
        extractor = CareKPIExtractor()
        
        episodes = [
            {
                "event_label": "sleeping",
                "start_time": datetime(2026, 2, 1, 22, 0, 0),
                "end_time": datetime(2026, 2, 2, 5, 0, 0),
                "duration_seconds": 25200,
            }
        ]
        
        profile = extractor.extract_room_usage_profile("bedroom", episodes)
        
        self.assertIsInstance(profile, RoomUsageProfile)
        self.assertEqual(profile.room_name, "bedroom")
        self.assertGreater(profile.total_time_minutes, 0)
    
    def test_calculate_day_statistics(self):
        """Test calculating day statistics."""
        extractor = CareKPIExtractor()
        
        # Multiple days of episodes
        all_episodes = {
            date(2026, 2, 1): {
                "bedroom": [
                    {
                        "event_label": "sleeping",
                        "start_time": datetime(2026, 2, 1, 22, 0, 0),
                        "end_time": datetime(2026, 2, 2, 5, 0, 0),
                        "duration_seconds": 25200,
                    }
                ]
            },
            date(2026, 2, 2): {
                "bedroom": [
                    {
                        "event_label": "sleeping",
                        "start_time": datetime(2026, 2, 2, 22, 0, 0),
                        "end_time": datetime(2026, 2, 3, 6, 0, 0),
                        "duration_seconds": 28800,
                    }
                ]
            }
        }
        
        stats = extractor.calculate_statistics(all_episodes)
        
        self.assertIsInstance(stats, DayStatistics)
        self.assertEqual(len(stats.daily_kpis), 2)
    
    def test_count_transitions_is_time_sorted(self):
        """Transition counting should use chronological order, not list order."""
        extractor = CareKPIExtractor()
        
        episodes = [
            {"event_label": "cooking", "start_time": datetime(2026, 2, 1, 11, 0, 0)},
            {"event_label": "sleeping", "start_time": datetime(2026, 2, 1, 10, 0, 0)},
            {"event_label": "sleeping", "start_time": datetime(2026, 2, 1, 12, 0, 0)},
        ]
        
        transitions = extractor._count_transitions(episodes)
        
        # Sorted order: sleeping -> cooking -> sleeping => 2 transitions
        self.assertEqual(transitions, 2)


class TestSleepMetrics(unittest.TestCase):
    """Tests for SleepMetrics."""
    
    def test_sleep_metrics_creation(self):
        """Test creating sleep metrics."""
        metrics = SleepMetrics(
            total_sleep_minutes=420,
            sleep_start_time=datetime(2026, 2, 1, 22, 0, 0),
            sleep_end_time=datetime(2026, 2, 2, 5, 0, 0),
            bed_entry_time=datetime(2026, 2, 1, 21, 45, 0),
            bed_exit_time=datetime(2026, 2, 2, 5, 15, 0),
            sleep_efficiency=0.95,
            interruption_count=1,
        )
        
        self.assertEqual(metrics.sleep_hours, 7.0)
        self.assertEqual(metrics.time_in_bed_minutes, 450)
    
    def test_sleep_efficiency_calculation(self):
        """Test sleep efficiency calculation."""
        efficiency = calculate_sleep_efficiency(
            total_sleep_minutes=420,
            time_in_bed_minutes=450,
        )
        
        self.assertAlmostEqual(efficiency, 0.9333, places=3)


class TestRoomUsageProfile(unittest.TestCase):
    """Tests for RoomUsageProfile."""
    
    def test_room_usage_creation(self):
        """Test creating room usage profile."""
        profile = RoomUsageProfile(
            room_name="bedroom",
            total_time_minutes=480,
            episode_count=2,
            unique_events={"sleeping", "dressing"},
        )
        
        self.assertEqual(profile.room_name, "bedroom")
        self.assertEqual(profile.total_time_hours, 8.0)
    
    def test_event_frequency(self):
        """Test event frequency calculation."""
        profile = RoomUsageProfile(
            room_name="bedroom",
            total_time_minutes=480,
            episode_count=3,
            unique_events={"sleeping", "dressing", "resting"},
        )
        
        freq = profile.event_frequency
        
        self.assertEqual(freq, 0.375)  # 3 events / 8 hours


class TestDayStatistics(unittest.TestCase):
    """Tests for DayStatistics."""
    
    def test_statistics_calculation(self):
        """Test statistics calculation."""
        from ml.derived_events import CareKPIData
        
        kpis = [
            CareKPIData(
                date=date(2026, 2, 1),
                total_sleep_minutes=420,
            ),
            CareKPIData(
                date=date(2026, 2, 2),
                total_sleep_minutes=480,
            ),
        ]
        
        stats = DayStatistics(daily_kpis=kpis)
        
        avg_sleep = stats.average_sleep_hours
        
        self.assertAlmostEqual(avg_sleep, 7.5)
    
    def test_empty_statistics(self):
        """Test statistics with no data."""
        stats = DayStatistics(daily_kpis=[])
        
        self.assertEqual(stats.average_sleep_hours, 0.0)
        self.assertEqual(stats.total_days, 0)


class TestAggregateWeeklyStats(unittest.TestCase):
    """Tests for aggregate_weekly_stats function."""
    
    def test_weekly_aggregation(self):
        """Test aggregating weekly statistics."""
        from ml.derived_events import CareKPIData
        
        # 7 days of data
        daily_kpis = [
            CareKPIData(
                date=date(2026, 2, i),
                total_sleep_minutes=400 + i * 10,
            )
            for i in range(1, 8)
        ]
        
        weekly = aggregate_weekly_stats(daily_kpis)
        
        self.assertEqual(len(weekly), 1)
        self.assertIn("avg_sleep_hours", weekly[0])
    
    def test_multiple_weeks(self):
        """Test aggregation across multiple weeks."""
        from ml.derived_events import CareKPIData
        
        # 14 days (2 weeks) of data
        daily_kpis = [
            CareKPIData(
                date=date(2026, 2, i),
                total_sleep_minutes=420,
            )
            for i in range(1, 15)
        ]
        
        weekly = aggregate_weekly_stats(daily_kpis)
        
        self.assertEqual(len(weekly), 2)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""
    
    def test_overlapping_episodes(self):
        """Test handling overlapping episodes."""
        extractor = CareKPIExtractor()
        
        # Overlapping sleep episodes (edge case)
        episodes = [
            {
                "event_label": "sleeping",
                "start_time": datetime(2026, 2, 1, 22, 0, 0),
                "end_time": datetime(2026, 2, 2, 5, 0, 0),
                "duration_seconds": 25200,
            },
            {
                "event_label": "sleeping",
                "start_time": datetime(2026, 2, 1, 23, 0, 0),
                "end_time": datetime(2026, 2, 2, 6, 0, 0),
                "duration_seconds": 25200,
            },
        ]
        
        metrics = extractor.extract_sleep_metrics(episodes, date(2026, 2, 1))
        
        # Should handle gracefully
        self.assertGreaterEqual(metrics.total_sleep_minutes, 0)
    
    def test_empty_episodes(self):
        """Test handling empty episodes."""
        extractor = CareKPIExtractor()
        
        metrics = extractor.extract_sleep_metrics([], date(2026, 2, 1))
        
        self.assertEqual(metrics.total_sleep_minutes, 0)
    
    def test_unknown_events_filtered(self):
        """Test that unknown events are filtered from KPIs."""
        extractor = CareKPIExtractor()
        
        episodes = [
            {
                "event_label": "unknown",
                "start_time": datetime(2026, 2, 1, 10, 0, 0),
                "end_time": datetime(2026, 2, 1, 10, 30, 0),
                "duration_seconds": 1800,
            },
            {
                "event_label": "sleeping",
                "start_time": datetime(2026, 2, 1, 22, 0, 0),
                "end_time": datetime(2026, 2, 2, 5, 0, 0),
                "duration_seconds": 25200,
            }
        ]
        
        kpis = extractor.extract_day_kpis({"bedroom": episodes}, date(2026, 2, 1))
        
        # Unknown should be counted separately
        self.assertGreater(kpis.unknown_ratio, 0)
        self.assertEqual(kpis.sleep_hours, 7.0)


if __name__ == '__main__':
    unittest.main()
