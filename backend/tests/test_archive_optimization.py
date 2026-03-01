
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'Development/Beta_5.5/backend'))

from ml.utils import fetch_sensor_windows_batch
from ml.training import TrainingPipeline

class TestArchiveOptimization(unittest.TestCase):
    def setUp(self):
        self.mock_platform = MagicMock()
        self.mock_registry = MagicMock()
        self.pipeline = TrainingPipeline(self.mock_platform, self.mock_registry)
        
    @patch('ml.utils.query_to_dataframe')
    @patch('ml.utils.adapter')
    def test_fetch_sensor_windows_batch_success(self, mock_adapter, mock_query_to_df):
        """Test that batch data is correctly fetched and parsed from DB."""
        mock_conn = MagicMock()
        mock_adapter.get_connection.return_value.__enter__.return_value = mock_conn
        
        sensor_json_1 = '{"motion": 1.0, "co2": 400}'
        sensor_json_2 = '{"motion": 0.0, "co2": 405}'
        
        # Returned DF (simulating multiple windows in one result)
        mock_df = pd.DataFrame({
            'timestamp': ['2026-01-27 10:00:00', '2026-01-27 10:10:00'],
            'sensor_features': [sensor_json_1, sensor_json_2]
        })
        mock_query_to_df.return_value = mock_df
        
        # Run
        w1_start = pd.Timestamp('2026-01-27 10:00:00')
        w1_end = pd.Timestamp('2026-01-27 10:00:10')
        w2_start = pd.Timestamp('2026-01-27 10:10:00')
        w2_end = pd.Timestamp('2026-01-27 10:10:10')
        
        windows = [(w1_start, w1_end), (w2_start, w2_end)]
        
        result = fetch_sensor_windows_batch('test_elder', 'living_room', windows)
        
        # Verify
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertTrue('motion' in result.columns)
        self.assertEqual(result.iloc[0]['motion'], 1.0)
        self.assertEqual(result.iloc[1]['motion'], 0.0)

    @patch('ml.training.fetch_sensor_windows_batch')
    @patch('ml.training.TrainingPipeline._find_archive_for_date')
    @patch('ml.training.load_sensor_data')
    @patch('ml.training.calculate_sequence_length') 
    def test_augment_training_data_batch_hit(self, mock_calc_seq, mock_load, mock_find_archive, mock_fetch_batch):
        """Test that archive loading is SKIPPED when Batch DB returns valid data."""
        
        # Setup valid Batch DB return covering the requested window (Backwards from correction)
        start_ts = pd.Timestamp('2026-01-27 10:00:00')
        # 60 samples * 10s = 600s coverage (10 min window)
        timestamps = [start_ts - timedelta(seconds=i*10) for i in range(60)]
        timestamps.sort()
        
        mock_df = pd.DataFrame({
            'timestamp': timestamps,
            'motion': [1.0]*60, 
            'co2': [400]*60,
            'temperature': [20]*60, 'light': [0]*60, 'sound': [0]*60, 'humidity': [50]*60
        })
        # Explicit datetime conversion to be safe
        mock_df['timestamp'] = pd.to_datetime(mock_df['timestamp'])
        mock_fetch_batch.return_value = mock_df
        
        # Mock sequence length
        mock_calc_seq.return_value = 60
        
        # Setup inputs
        self.pipeline.room_config.get_data_interval = MagicMock(return_value=10)
        self.pipeline.platform.sensor_columns = ['motion', 'co2', 'temperature', 'light', 'sound', 'humidity']
        # Mock label encoder
        mock_encoder = MagicMock()
        mock_encoder.transform.return_value = [0]
        self.pipeline.platform.label_encoders = {'kitchen': mock_encoder}
        
        # Mock fetch_golden_samples to return 1 correction
        with patch('ml.training.fetch_golden_samples') as mock_fetch_gold:
            mock_fetch_gold.return_value = pd.DataFrame({
                'timestamp': [start_ts],
                'activity': ['kitchen_normal_use'],
                'record_date': ['2026-01-27']
            })
            
            # Pass window_seconds=600 to get a 600s (60 sample) slice
            # Pass correctly shaped empty arrays for concatenation to work
            empty_X = np.empty((0, 60, 6))
            empty_y = np.empty((0,))
            
            X, y = self.pipeline.augment_training_data(
                'kitchen', 'test_elder', [], empty_X, empty_y, 600, 10
            )
            
            # CRITICAL ASSERTION:
            # fetch_sensor_windows_batch SHOULD have been called
            mock_fetch_batch.assert_called()
            # _find_archive_for_date SHOULD NOT have been called (Optimization Success)
            mock_find_archive.assert_not_called()
            
            # Verify we got data back
            self.assertEqual(len(X), 1) # 1 sequence added

    @patch('ml.training.fetch_sensor_windows_batch')
    @patch('ml.training.TrainingPipeline._find_archive_for_date')
    @patch('ml.training.load_sensor_data')
    @patch('ml.training.calculate_sequence_length') 
    def test_augment_training_data_batch_incomplete_fallback(self, mock_calc_seq, mock_load, mock_find_archive, mock_fetch_batch):
        """Test that archive loading is TRIGGERED when Batch DB is incomplete (Strict Fallback)."""
        
        # Setup INCOMPLETE Batch DB return (only 30 samples instead of 60)
        start_ts = pd.Timestamp('2026-01-27 10:00:00')
        timestamps = [start_ts - timedelta(seconds=i*10) for i in range(30)] # Too short!
        timestamps.sort()
        
        mock_df = pd.DataFrame({
            'timestamp': timestamps,
            'motion': [1.0]*30, 
            'co2': [400]*30,
            'temperature': [20]*30, 'light': [0]*30, 'sound': [0]*30, 'humidity': [50]*30
        })
        mock_df['timestamp'] = pd.to_datetime(mock_df['timestamp'])
        mock_fetch_batch.return_value = mock_df
        
        # Mock sequence length
        mock_calc_seq.return_value = 60
        
        # Mock ARCHIVE loading success
        # Ensure archive timestamps cover the window using robust date_range
        # periods=60, freq='10s', end=start_ts ensures we hit the end time exactly
        archive_timestamps = pd.date_range(end=start_ts, periods=60, freq='10s')

        mock_archive = {'kitchen': pd.DataFrame({
            'timestamp': archive_timestamps,
            'motion': [1.0]*60, 'co2': [400]*60,
             'temperature': [20]*60, 'light': [0]*60, 'sound': [0]*60, 'humidity': [50]*60
        })}
        # Ensure timestamp type
        mock_archive['kitchen']['timestamp'] = pd.to_datetime(mock_archive['kitchen']['timestamp'])
        mock_load.return_value = mock_archive
        mock_find_archive.return_value = "path/to/archive.parquet"
        
        # Setup inputs
        self.pipeline.room_config.get_data_interval = MagicMock(return_value=10)
        self.pipeline.platform.sensor_columns = ['motion', 'co2', 'temperature', 'light', 'sound', 'humidity']
        # Mock label encoder
        mock_encoder = MagicMock()
        mock_encoder.transform.return_value = [0]
        self.pipeline.platform.label_encoders = {'kitchen': mock_encoder}
        
        # Mock fetch_golden_samples to return 1 correction
        with patch('ml.training.fetch_golden_samples') as mock_fetch_gold:
            mock_fetch_gold.return_value = pd.DataFrame({
                'timestamp': [start_ts],
                'activity': ['kitchen_normal_use'],
                'record_date': ['2026-01-27']
            })
            
            empty_X = np.empty((0, 60, 6))
            empty_y = np.empty((0,))
            
            X, y = self.pipeline.augment_training_data(
                'kitchen', 'test_elder', [], empty_X, empty_y, 600, 10
            )
            
            # CRITICAL ASSERTION:
            # fetch_sensor_windows_batch SHOULD have been called
            mock_fetch_batch.assert_called()
            
            # _find_archive_for_date SHOULD HAVE BEEN CALLED because batch was short
            mock_find_archive.assert_called()
            
            # Verify we got data back (from archive)
            self.assertEqual(len(X), 1)

if __name__ == '__main__':
    unittest.main()
