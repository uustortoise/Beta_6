#!/usr/bin/env python3
"""
Unit tests for SleepAnalyzer module.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.sleep_analyzer import SleepAnalyzer


class TestSleepAnalyzer(unittest.TestCase):
    """Tests for the SleepAnalyzer class."""
    
    def setUp(self):
        self.analyzer = SleepAnalyzer()
    
    def test_ideal_sleep_scenario(self):
        """Test 8 hours of perfect, uninterrupted deep sleep."""
        # Generate 8 hours of data at 10-second intervals
        records = 8 * 60 * 6  # 8 hours * 60 mins * 6 records/min
        base_time = datetime(2026, 1, 1, 22, 0, 0)
        
        df = pd.DataFrame({
            'timestamp': [base_time + timedelta(seconds=10*i) for i in range(records)],
            'predicted_activity': ['sleep'] * records,
            # Use varying low motion values to avoid triggering heuristic fallback (nunique <= 1 check)
            'motion': [0.01 + (0.01 * (i % 3)) for i in range(records)]  # 0.01, 0.02, 0.03, repeating
        })
        
        results = self.analyzer.analyze_day("test_elder", "2026-01-01", df)
        
        self.assertIsNotNone(results)
        self.assertEqual(results['total_duration_hours'], 8.0)
        self.assertEqual(results['sleep_efficiency'], 1.0)
        self.assertEqual(results['quality_score'], 100)
        self.assertEqual(results['stages_summary']['Deep'], 100.0)

    def test_short_sleep_penalty(self):
        """Test 4-hour sleep (Severe Duration Penalty)."""
        records = 4 * 60 * 6
        base_time = datetime(2026, 1, 1, 23, 0, 0)
        
        df = pd.DataFrame({
            'timestamp': [base_time + timedelta(seconds=10*i) for i in range(records)],
            'predicted_activity': ['sleep'] * records,
            'motion': [0.01] * records
        })
        
        results = self.analyzer.analyze_day("test_elder", "2026-01-01", df)
        
        self.assertIsNotNone(results)
        self.assertEqual(results['total_duration_hours'], 4.0)
        # 100% efficiency * 0.7 (short_severe penalty) = 70
        self.assertEqual(results['quality_score'], 70)

    def test_fragmented_sleep(self):
        """Test sleep with awake periods (reduced efficiency)."""
        records = 3 * 60 * 6  # 3 hours
        base_time = datetime(2026, 1, 1, 22, 0, 0)
        
        # 2 hours deep sleep, 1 hour awake
        motion_vals = [0.01] * (2 * 60 * 6) + [0.6] * (1 * 60 * 6)
        
        df = pd.DataFrame({
            'timestamp': [base_time + timedelta(seconds=10*i) for i in range(records)],
            'predicted_activity': ['sleep'] * records,
            'motion': motion_vals
        })
        
        results = self.analyzer.analyze_day("test_elder", "2026-01-01", df)
        
        self.assertEqual(results['total_duration_hours'], 3.0)
        # 2 hours sleep, 1 hour awake -> Efficiency should be roughly ~0.666
        self.assertAlmostEqual(results['sleep_efficiency'], 0.67, places=2)

    def test_no_sleep_data(self):
        """Test handling of no sleep data."""
        df = pd.DataFrame({
            'timestamp': [datetime.now()],
            'predicted_activity': ['cooking'],
            'motion': [0.5]
        })
        
        results = self.analyzer.analyze_day("test_elder", "2026-01-01", df)
        self.assertIsNone(results)


if __name__ == "__main__":
    unittest.main()
