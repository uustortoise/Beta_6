"""
Integration tests for cross-room correction isolation.

Validates that corrections applied to one room do NOT affect other rooms.
Covers:
    - train_from_files respects rooms filter
    - repredict_all respects rooms filter with early model/data filtering
    - save_predictions_to_db uses actual dates (not datetime.now())
    - save_predictions_to_db calls regenerate_segments for correct (room, date) pairs
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from ml.prediction import PredictionPipeline


class TestCrossRoomIsolation(unittest.TestCase):
    """Ensure corrections to one room don't contaminate other rooms."""

    def setUp(self):
        self.mock_platform = MagicMock()
        self.mock_registry = MagicMock()
        self.mock_platform.sensor_columns = ['s1', 's2']
        self.mock_platform.label_encoders = {}
        self.mock_platform.room_models = {}

        self.pipeline = PredictionPipeline(
            self.mock_platform, self.mock_registry, enable_denoising=False
        )

    # ─── save_predictions_to_db date tracking ────────────────────────

    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x.lower().replace(' ', ''))
    @patch('ml.prediction.adapter')
    @patch('ml.prediction.regenerate_segments')
    @patch('ml.prediction.validate_activity_for_room', side_effect=lambda a, r: a)
    @patch('ml.prediction.normalize_timestamp', side_effect=str)
    def test_save_predictions_uses_actual_dates(
        self, mock_norm_ts, mock_validate, mock_regen, mock_adapter, mock_norm_room
    ):
        """regenerate_segments must be called with the ACTUAL prediction date,
        not datetime.now()."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_adapter.get_connection.return_value.__enter__.return_value = mock_conn
        mock_cursor.fetchone.return_value = None  # No existing corrected row

        # Predictions for TWO different dates in the same room
        pred_df = pd.DataFrame({
            'timestamp': [
                pd.Timestamp('2025-12-01 08:00:00'),
                pd.Timestamp('2025-12-02 09:00:00'),
            ],
            'predicted_activity': ['walking', 'sitting'],
            'confidence': [0.9, 0.85],
        })

        self.pipeline.save_predictions_to_db({'Bedroom': pred_df}, 'elder_samuel')

        # regenerate_segments should be called for BOTH actual dates, not today
        regen_calls = mock_regen.call_args_list
        called_dates = {c[0][2] for c in regen_calls}  # 3rd positional arg is record_date
        self.assertIn('2025-12-01', called_dates, "Missing segment regen for 2025-12-01")
        self.assertIn('2025-12-02', called_dates, "Missing segment regen for 2025-12-02")
        # Must NOT contain today's date (the old bug)
        import datetime
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        if today not in ('2025-12-01', '2025-12-02'):
            self.assertNotIn(today, called_dates, "Should not regenerate for today's date")

    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x.lower().replace(' ', ''))
    @patch('ml.prediction.adapter')
    @patch('ml.prediction.regenerate_segments')
    @patch('ml.prediction.validate_activity_for_room', side_effect=lambda a, r: a)
    @patch('ml.prediction.normalize_timestamp', side_effect=str)
    def test_save_predictions_regenerates_only_touched_rooms(
        self, mock_norm_ts, mock_validate, mock_regen, mock_adapter, mock_norm_room
    ):
        """regenerate_segments must only be called for rooms that had data inserted."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_adapter.get_connection.return_value.__enter__.return_value = mock_conn
        mock_cursor.fetchone.return_value = None

        bedroom_df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2025-12-01 08:00:00')],
            'predicted_activity': ['sleeping'],
            'confidence': [0.95],
        })

        self.pipeline.save_predictions_to_db({'Bedroom': bedroom_df}, 'elder_samuel')

        # Only Bedroom should get segment regeneration
        regen_calls = mock_regen.call_args_list
        called_rooms = {c[0][1] for c in regen_calls}  # 2nd arg is room
        self.assertEqual(called_rooms, {'Bedroom'})

    # ─── repredict_all room scoping ──────────────────────────────────

    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x.lower().replace(' ', ''))
    @patch('ml.prediction.get_archive_files')
    @patch('ml.prediction.load_sensor_data')
    def test_repredict_all_filters_rooms_early(
        self, mock_load, mock_archive, mock_norm
    ):
        """When rooms filter is provided, repredict_all should only load models
        and data for those rooms, not all rooms."""
        # Setup: 3 archived files, data has 3 rooms
        mock_archive.return_value = [
            {'path': '/archive/file1.xlsx', 'filename': 'file1.xlsx'}
        ]
        mock_load.return_value = {
            'Bedroom': pd.DataFrame({'timestamp': [1], 's1': [0.1], 's2': [0.2]}),
            'Kitchen': pd.DataFrame({'timestamp': [1], 's1': [0.3], 's2': [0.4]}),
            'Bathroom': pd.DataFrame({'timestamp': [1], 's1': [0.5], 's2': [0.6]}),
        }

        # Models exist for all rooms
        self.mock_registry.load_models_for_elder.return_value = ['Bedroom', 'Kitchen', 'Bathroom']

        # Mock run_prediction to capture what rooms it receives
        captured_rooms = []
        def mock_run_pred(data, rooms, seq_length):
            captured_rooms.extend(rooms)
            return {}
        self.pipeline.run_prediction = mock_run_pred
        self.pipeline.apply_golden_samples = lambda p, e: p
        self.pipeline.save_predictions_to_db = MagicMock()

        # Only correct Bedroom
        self.pipeline.repredict_all(
            'elder_samuel', Path('/archive'),
            rooms={'Bedroom'}
        )

        # run_prediction should only receive Bedroom model
        self.assertEqual(captured_rooms, ['Bedroom'])

    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x.lower().replace(' ', ''))
    @patch('ml.prediction.get_archive_files')
    @patch('ml.prediction.load_sensor_data')
    def test_repredict_all_without_filter_processes_all(
        self, mock_load, mock_archive, mock_norm
    ):
        """Without rooms filter, repredict_all should process all rooms."""
        mock_archive.return_value = [
            {'path': '/archive/file1.xlsx', 'filename': 'file1.xlsx'}
        ]
        mock_load.return_value = {
            'Bedroom': pd.DataFrame({'timestamp': [1], 's1': [0.1], 's2': [0.2]}),
            'Kitchen': pd.DataFrame({'timestamp': [1], 's1': [0.3], 's2': [0.4]}),
        }
        self.mock_registry.load_models_for_elder.return_value = ['Bedroom', 'Kitchen']

        captured_rooms = []
        def mock_run_pred(data, rooms, seq_length):
            captured_rooms.extend(rooms)
            return {}
        self.pipeline.run_prediction = mock_run_pred
        self.pipeline.apply_golden_samples = lambda p, e: p
        self.pipeline.save_predictions_to_db = MagicMock()

        self.pipeline.repredict_all('elder_samuel', Path('/archive'))

        self.assertEqual(set(captured_rooms), {'Bedroom', 'Kitchen'})


class TestTrainFromFilesRoomScoping(unittest.TestCase):
    """Ensure train_from_files only trains corrected rooms."""

    @patch('ml.pipeline.get_room_config')
    def test_train_from_files_skips_uncorrected_rooms(self, mock_room_config):
        """When rooms filter is provided, only those rooms should be trained."""
        from ml.pipeline import UnifiedPipeline

        with patch.object(UnifiedPipeline, '__init__', lambda self, **kw: None):
            pipeline = UnifiedPipeline.__new__(UnifiedPipeline)

            # Minimal setup
            pipeline.platform = MagicMock()
            pipeline.platform.sensor_columns = ['s1']
            pipeline.trainer = MagicMock()
            pipeline.trainer.train_room.return_value = {'room': 'Bedroom', 'accuracy': 0.9}
            pipeline.enable_denoising = False
            pipeline.denoising_method = 'hampel'
            pipeline.denoising_window = 5
            pipeline.denoising_threshold = 3
            pipeline.registry = MagicMock()
            pipeline.registry.get_models_dir.return_value = Path('/tmp/models')
            pipeline.feature_extractors = {'Bedroom': MagicMock(), 'Kitchen': MagicMock()}
            
            mock_room_config.return_value.calculate_seq_length.return_value = 5

            # Mock data loading: file has Bedroom + Kitchen
            with patch('utils.data_loader.load_sensor_data') as mock_load:
                mock_load.return_value = {
                    'Bedroom': pd.DataFrame({
                        'timestamp': pd.date_range('2025-01-01', periods=100, freq='10s'),
                        's1': np.random.randn(100),
                        'activity': ['sleeping'] * 100,
                    }),
                    'Kitchen': pd.DataFrame({
                        'timestamp': pd.date_range('2025-01-01', periods=100, freq='10s'),
                        's1': np.random.randn(100),
                        'activity': ['cooking'] * 100,
                    }),
                }

                pipeline.platform.preprocess_with_resampling.return_value = mock_load.return_value['Bedroom']
            
                # Mock gate evaluation to pass (test is about room filtering, not gates)
                pipeline._evaluate_gates_unified = MagicMock(return_value={
                    'gate_pass': True,
                    'gate_stack': [],
                    'gate_reasons': [],
                })
                pipeline._evaluate_post_training_gates = MagicMock(return_value=(True, []))

                # Only train Bedroom
                _, metrics = pipeline.train_from_files(
                    ['/data/file.xlsx'], 'elder_samuel', rooms={'Bedroom'}
                )

                # train_room should only be called for Bedroom, NOT Kitchen
                train_calls = pipeline.trainer.train_room.call_args_list
                trained_rooms = [c.kwargs.get('room_name') or c.args[0] for c in train_calls]
                self.assertIn('Bedroom', trained_rooms)
                self.assertNotIn('Kitchen', trained_rooms)


if __name__ == '__main__':
    unittest.main()
