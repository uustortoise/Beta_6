#!/usr/bin/env python3
"""
Performance tests for backfill analysis with mock 10k record dataset.

Tests the batch processing performance under simulated 1000 POC load.
"""

import sys
import os
import unittest
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backfill_analysis import (
    process_elder_batch,
    backfill_elder,
    get_dates_with_adl_data,
    get_all_elder_ids
)


class TestBackfillPerformance(unittest.TestCase):
    """Performance tests for backfill batch processing."""
    
    @classmethod
    def setUpClass(cls):
        """Create a temporary database with mock 10k records."""
        cls.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        cls.db_path = cls.temp_db.name
        cls.temp_db.close()
        
        # Create schema and populate with mock data
        conn = sqlite3.connect(cls.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS elders (
                elder_id TEXT PRIMARY KEY,
                profile_data TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS adl_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                elder_id TEXT,
                record_date TEXT,
                timestamp TEXT,
                room TEXT,
                activity_type TEXT,
                confidence REAL,
                sensor_features TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sleep_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                elder_id TEXT,
                analysis_date TEXT,
                UNIQUE(elder_id, analysis_date)
            )
        ''')
        
        # Create indices for performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_adl_elder_date ON adl_history(elder_id, record_date)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_sleep_elder ON sleep_analysis(elder_id)')
        
        # Insert mock elders (100 for faster testing, simulates 10% of 1000 POC)
        num_elders = 100
        for i in range(num_elders):
            conn.execute('INSERT INTO elders VALUES (?, ?)', 
                        (f'ELDER_{i:04d}', '{}'))
        
        # Insert mock ADL records (100 per elder per day, 10 days = 100k records simulated)
        base_date = datetime(2026, 1, 1)
        num_days = 10
        records_per_day = 100
        
        batch = []
        for elder_idx in range(num_elders):
            elder_id = f'ELDER_{elder_idx:04d}'
            for day_offset in range(num_days):
                record_date = (base_date + timedelta(days=day_offset)).strftime('%Y-%m-%d')
                for rec_idx in range(records_per_day):
                    timestamp = f"{record_date} {rec_idx//6:02d}:{(rec_idx%6)*10:02d}:00"
                    batch.append((
                        elder_id,
                        record_date,
                        timestamp,
                        'bedroom' if rec_idx % 2 == 0 else 'living_room',
                        'sleep' if rec_idx < 50 else 'inactive',
                        0.85,
                        '{}'
                    ))
        
        conn.executemany('''
            INSERT INTO adl_history 
            (elder_id, record_date, timestamp, room, activity_type, confidence, sensor_features)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', batch)
        
        conn.commit()
        conn.close()
        
        print(f"Created test database with {num_elders} elders, {num_elders * num_days} elder-days, "
              f"{len(batch)} total ADL records")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary database."""
        try:
            os.unlink(cls.db_path)
        except:
            pass
    
    def get_connection(self):
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def test_batch_size_scaling(self):
        """Test that batch processing scales with batch size."""
        conn = self.get_connection()
        elder_ids = get_all_elder_ids(conn)[:20]  # Use subset for speed
        
        # Test different batch sizes
        batch_sizes = [5, 10, 20]
        times = {}
        
        for batch_size in batch_sizes:
            # Mock the downstream analysis to measure batch overhead only
            with patch('scripts.backfill_analysis.run_downstream_analysis'):
                start = time.time()
                stats = process_elder_batch(
                    conn, elder_ids, 
                    batch_size=batch_size,
                    force=True
                )
                elapsed = time.time() - start
                times[batch_size] = elapsed
        
        conn.close()
        
        # Verify we got stats back
        self.assertIn('total_elders', stats)
        self.assertEqual(stats['total_elders'], 20)
        
        print(f"Batch timing: {times}")
    
    def test_date_range_filtering(self):
        """Test that date range filtering reduces processing scope."""
        conn = self.get_connection()
        elder_id = 'ELDER_0000'
        
        # Get all dates
        all_dates = get_dates_with_adl_data(conn, elder_id)
        self.assertEqual(len(all_dates), 10)  # We inserted 10 days
        
        # Test with date range filter (should only get subset)
        with patch('scripts.backfill_analysis.run_downstream_analysis'):
            processed = backfill_elder(
                conn, elder_id, 
                force=True,
                date_range=('2026-01-03', '2026-01-05')
            )
        
        # Should only process 3 days: Jan 3, 4, 5
        self.assertEqual(processed, 3)
        
        conn.close()
    
    def test_throughput_measurement(self):
        """Test that throughput stats are calculated correctly."""
        conn = self.get_connection()
        elder_ids = get_all_elder_ids(conn)[:10]
        
        with patch('scripts.backfill_analysis.run_downstream_analysis'):
            stats = process_elder_batch(
                conn, elder_ids,
                batch_size=5,
                force=True
            )
        
        conn.close()
        
        self.assertIn('elders_per_second', stats)
        self.assertIn('total_time_seconds', stats)
        self.assertGreater(stats['elders_per_second'], 0)
        print(f"Throughput: {stats['elders_per_second']:.2f} elders/sec")


class TestDataMigrationValidation(unittest.TestCase):
    """Tests to validate data migration integrity."""
    
    def test_motion_data_preservation(self):
        """Validate that motion data handling doesn't corrupt records."""
        import pandas as pd
        from utils.motion_normalizer import MotionDataNormalizer
        
        # Simulate raw motion data
        original_df = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=100, freq='10s'),
            'predicted_activity': ['sleep'] * 100,
            'motion': [0.02 + (i % 5) * 0.01 for i in range(100)]
        })
        
        # Process through normalizer
        processed_df, quality = MotionDataNormalizer.normalize_for_sleep_analysis(
            original_df.copy(),
            source="migration_test"
        )
        
        # Validate: row count preserved
        self.assertEqual(len(processed_df), len(original_df))
        
        # Validate: timestamp and activity preserved
        self.assertTrue((processed_df['timestamp'] == original_df['timestamp']).all())
        self.assertTrue((processed_df['predicted_activity'] == original_df['predicted_activity']).all())
    
    def test_scaled_data_detection(self):
        """Validate Z-score data is correctly identified and handled."""
        import pandas as pd
        import numpy as np
        from utils.motion_normalizer import MotionDataNormalizer, MotionDataStatus
        
        # Z-score normalized data (mean ≈ 0, std ≈ 1, has negatives)
        zscore_df = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=100, freq='10s'),
            'predicted_activity': ['sleep'] * 100,
            'motion': np.random.normal(0, 1, 100)
        })
        
        quality = MotionDataNormalizer.validate(zscore_df)
        
        # Should detect as Z-score
        self.assertEqual(quality.status, MotionDataStatus.SCALED_ZSCORE)
        self.assertTrue(quality.use_heuristics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
