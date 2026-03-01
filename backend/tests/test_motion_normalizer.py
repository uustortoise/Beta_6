#!/usr/bin/env python3
"""
Unit tests for MotionDataNormalizer utility.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.motion_normalizer import MotionDataNormalizer, MotionDataStatus, MotionDataQuality


class TestMotionDataNormalizer(unittest.TestCase):
    """Tests for the MotionDataNormalizer class."""
    
    def test_valid_raw_motion(self):
        """Test detection of valid raw sensor motion data."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=100, freq='10s'),
            'predicted_activity': ['sleep'] * 100,
            'motion': np.random.uniform(0.0, 0.8, 100)  # Valid raw range
        })
        
        quality = MotionDataNormalizer.validate(df)
        
        self.assertEqual(quality.status, MotionDataStatus.VALID_RAW)
        self.assertFalse(quality.use_heuristics)
    
    def test_zscore_detection_with_negatives(self):
        """Test Z-score detection when negative values present."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=100, freq='10s'),
            'predicted_activity': ['sleep'] * 100,
            'motion': np.random.normal(0, 1, 100)  # Z-score normalized
        })
        
        quality = MotionDataNormalizer.validate(df)
        
        self.assertEqual(quality.status, MotionDataStatus.SCALED_ZSCORE)
        self.assertTrue(quality.use_heuristics)
    
    def test_missing_motion_column(self):
        """Test handling of missing motion column."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=10, freq='10s'),
            'predicted_activity': ['sleep'] * 10
        })
        
        quality = MotionDataNormalizer.validate(df)
        
        self.assertEqual(quality.status, MotionDataStatus.MISSING)
        self.assertTrue(quality.use_heuristics)
    
    def test_constant_motion_value(self):
        """Test detection of constant motion values."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=100, freq='10s'),
            'predicted_activity': ['sleep'] * 100,
            'motion': [0.1] * 100  # All same value
        })
        
        quality = MotionDataNormalizer.validate(df)
        
        self.assertEqual(quality.status, MotionDataStatus.CONSTANT)
        self.assertTrue(quality.use_heuristics)
    
    def test_normalize_preserves_valid_data(self):
        """Test that valid raw data is preserved after normalization."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=10, freq='10s'),
            'predicted_activity': ['sleep'] * 10,
            'motion': [0.02, 0.03, 0.05, 0.1, 0.15, 0.08, 0.04, 0.02, 0.03, 0.06]
        })
        
        normalized, quality = MotionDataNormalizer.normalize_for_sleep_analysis(
            df, source="test"
        )
        
        self.assertEqual(quality.status, MotionDataStatus.VALID_RAW)
        self.assertIn('motion', normalized.columns)
        self.assertEqual(len(normalized), 10)
    
    def test_normalize_removes_zscore_data(self):
        """Test that Z-score data is removed to force heuristic fallback."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=100, freq='10s'),
            'predicted_activity': ['sleep'] * 100,
            'motion': np.random.normal(0, 1, 100)  # Z-score normalized
        })
        
        normalized, quality = MotionDataNormalizer.normalize_for_sleep_analysis(
            df, source="test"
        )
        
        self.assertEqual(quality.status, MotionDataStatus.SCALED_ZSCORE)
        self.assertNotIn('motion', normalized.columns)  # Should be removed
    
    def test_normalize_with_raw_source_injection(self):
        """Test injection of raw motion from external source."""
        # Prediction data with scaled/invalid motion
        pred_df = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=10, freq='10s'),
            'predicted_activity': ['sleep'] * 10,
            'motion': np.random.normal(0, 1, 10)  # Z-score
        })
        
        # Raw source with valid motion
        raw_df = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=10, freq='10s'),
            'motion': np.random.uniform(0.01, 0.3, 10)  # Valid raw
        })
        
        normalized, quality = MotionDataNormalizer.normalize_for_sleep_analysis(
            pred_df,
            source="test",
            raw_data_source=raw_df
        )
        
        # Should have injected raw motion
        self.assertIn('motion', normalized.columns)
        # Motion should now be in valid range
        self.assertTrue((normalized['motion'] >= 0).all())
        self.assertTrue((normalized['motion'] <= 1).all())


if __name__ == "__main__":
    unittest.main()
