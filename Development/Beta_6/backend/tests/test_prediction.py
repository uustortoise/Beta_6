
import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, ANY
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.prediction import PredictionPipeline
from ml.exceptions import PredictionError, DatabaseError

class TestPredictionPipeline(unittest.TestCase):
    def setUp(self):
        self._orig_enable_beta6_unknown_runtime = os.environ.get("ENABLE_BETA6_UNKNOWN_ABSTAIN_RUNTIME")
        self._orig_enable_beta6_hmm_runtime = os.environ.get("ENABLE_BETA6_HMM_RUNTIME")
        self._orig_beta6_phase4_runtime_enabled = os.environ.get("BETA6_PHASE4_RUNTIME_ENABLED")
        self._orig_beta6_phase4_runtime_rooms = os.environ.get("BETA6_PHASE4_RUNTIME_ROOMS")
        os.environ["ENABLE_BETA6_UNKNOWN_ABSTAIN_RUNTIME"] = "false"
        os.environ["ENABLE_BETA6_HMM_RUNTIME"] = "false"
        os.environ["BETA6_PHASE4_RUNTIME_ENABLED"] = "false"
        os.environ.pop("BETA6_PHASE4_RUNTIME_ROOMS", None)
        # Mock dependencies
        self.mock_platform = MagicMock()
        self.mock_registry = MagicMock()
        
        # Setup mock platform
        self.mock_platform.sensor_columns = ['s1', 's2']
        self.mock_platform.label_encoders = {'room1': MagicMock()}
        self.mock_platform.label_encoders['room1'].classes_ = np.array(['walking', 'sitting'])
        self.mock_platform.label_encoders['room1'].inverse_transform.return_value = ['walking', 'sitting']
        self.mock_platform.room_models = {'room1': MagicMock()}
        # Keep thresholds deterministic; MagicMock auto-attrs can create false calibrated paths.
        self.mock_platform.class_thresholds = {}
        
        # Mock create_sequences
        self.mock_platform.create_sequences.return_value = np.zeros((2, 5, 2))
        
        # Mock preprocess to return something
        self.mock_platform.preprocess_with_resampling.return_value = pd.DataFrame() 
        
        self.pipeline = PredictionPipeline(self.mock_platform, self.mock_registry)

    def tearDown(self):
        if self._orig_enable_beta6_unknown_runtime is None:
            os.environ.pop("ENABLE_BETA6_UNKNOWN_ABSTAIN_RUNTIME", None)
        else:
            os.environ["ENABLE_BETA6_UNKNOWN_ABSTAIN_RUNTIME"] = self._orig_enable_beta6_unknown_runtime
        if self._orig_enable_beta6_hmm_runtime is None:
            os.environ.pop("ENABLE_BETA6_HMM_RUNTIME", None)
        else:
            os.environ["ENABLE_BETA6_HMM_RUNTIME"] = self._orig_enable_beta6_hmm_runtime
        if self._orig_beta6_phase4_runtime_enabled is None:
            os.environ.pop("BETA6_PHASE4_RUNTIME_ENABLED", None)
        else:
            os.environ["BETA6_PHASE4_RUNTIME_ENABLED"] = self._orig_beta6_phase4_runtime_enabled
        if self._orig_beta6_phase4_runtime_rooms is None:
            os.environ.pop("BETA6_PHASE4_RUNTIME_ROOMS", None)
        else:
            os.environ["BETA6_PHASE4_RUNTIME_ROOMS"] = self._orig_beta6_phase4_runtime_rooms

    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x)
    @patch('ml.prediction.calculate_sequence_length')
    @patch('ml.prediction.validate_activity_for_room')
    @patch('ml.prediction.sqlite3.connect')
    @patch('ml.prediction.regenerate_segments')
    def test_run_prediction_success(self, mock_regen, mock_sqlite, mock_validate, mock_calc_seq, mock_norm):
        """Test successful prediction flow."""
        # Setup mocks
        mock_calc_seq.return_value = 5
        
        # We need consistent dimensions:
        # Data len = 10 (min requirement)
        # Seq len = 5
        # Expected sequences = 10 - 5 + 1 = 6
        
        # Configure model to return 6 predictions
        # Shape (6, 2) for 2 classes
        # Set max probability > default threshold (0.6)
        probs = np.array([[0.9, 0.1]] * 6)
        probs[1] = [0.2, 0.8] # 2nd prediction is class 1 (sitting)
        self.mock_platform.room_models['room1'].predict.return_value = probs
        
        # Ensure decoded labels match prediction count (6)
        self.mock_platform.label_encoders['room1'].inverse_transform.return_value = \
            ['walking', 'sitting', 'walking', 'walking', 'walking', 'walking']
        
        self.mock_platform.create_sequences.return_value = np.zeros((6, 5, 2))
        
        # Input data (10 rows)
        sensor_data = {
            'room1': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='10s'),
                's1': np.random.randn(10),
                's2': np.random.randn(10)
            })
        }
        
        # Update preprocess mock to return data with timestamp
        self.mock_platform.preprocess_with_resampling.return_value = sensor_data['room1']
        
        # Run
        predictions = self.pipeline.run_prediction(
            sensor_data=sensor_data,
            loaded_rooms=['room1'],
            seq_length=5
        )
        
        # Verify
        self.assertIn('room1', predictions)
        pred_df = predictions['room1']
        self.assertEqual(len(pred_df), 6) # 6 sequences
        self.assertTrue('confidence' in pred_df.columns)
        self.assertTrue('predicted_activity' in pred_df.columns)
        self.assertTrue('predicted_activity_raw' in pred_df.columns)
        self.assertTrue('predicted_top2_label_raw' in pred_df.columns)
        self.assertTrue('predicted_top2_prob_raw' in pred_df.columns)
        
        # Check values
        self.assertEqual(pred_df.iloc[0]['predicted_activity'], 'walking')
        self.assertEqual(pred_df.iloc[1]['predicted_activity'], 'sitting')
        self.assertNotIn('low_confidence', set(pred_df['predicted_activity'].tolist()))

    @patch.dict('os.environ', {'ENABLE_INFERENCE_HYSTERESIS': 'true', 'INFERENCE_HYSTERESIS_STEPS': '2'}, clear=False)
    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x)
    @patch('ml.prediction.calculate_sequence_length')
    def test_run_prediction_applies_hysteresis(self, mock_calc_seq, mock_norm):
        mock_calc_seq.return_value = 5

        # Raw predictions flip for one step only; hysteresis should suppress this.
        probs = np.array(
            [
                [0.9, 0.1],  # walking
                [0.1, 0.9],  # sitting (single spike)
                [0.9, 0.1],  # walking
                [0.9, 0.1],  # walking
                [0.9, 0.1],  # walking
                [0.9, 0.1],  # walking
            ]
        )
        self.mock_platform.room_models['room1'].predict.return_value = probs
        self.mock_platform.label_encoders['room1'].inverse_transform.return_value = \
            ['walking', 'sitting', 'walking', 'walking', 'walking', 'walking']
        self.mock_platform.create_sequences.return_value = np.zeros((6, 5, 2))

        sensor_data = {
            'room1': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='10s'),
                's1': np.random.randn(10),
                's2': np.random.randn(10)
            })
        }
        self.mock_platform.preprocess_with_resampling.return_value = sensor_data['room1']

        predictions = self.pipeline.run_prediction(sensor_data=sensor_data, loaded_rooms=['room1'], seq_length=5)
        pred_df = predictions['room1']

        self.assertEqual(pred_df.iloc[1]['predicted_activity_raw'], 'sitting')
        self.assertEqual(pred_df.iloc[1]['predicted_activity'], 'walking')
        self.assertEqual(pred_df.iloc[1]['predicted_top1_label'], 'walking')
        self.assertEqual(pred_df.iloc[1]['predicted_top1_label_raw'], 'sitting')
        self.assertEqual(pred_df.iloc[1]['predicted_top2_label'], 'sitting')
        self.assertEqual(pred_df.iloc[1]['predicted_top2_label_raw'], 'sitting')
        self.assertAlmostEqual(float(pred_df.iloc[1]['predicted_top2_prob']), 0.9, places=6)
        self.assertAlmostEqual(float(pred_df.iloc[1]['predicted_top2_prob_raw']), 0.1, places=6)
        self.assertAlmostEqual(float(pred_df.iloc[1]['confidence']), 0.1, places=6)
        self.assertAlmostEqual(float(pred_df.iloc[1]['confidence_raw']), 0.9, places=6)

    @patch.dict(
        'os.environ',
        {
            'ENABLE_TWO_STAGE_CORE_RUNTIME': 'true',
            'TWO_STAGE_CORE_RUNTIME_ROOMS': 'room1',
        },
        clear=False,
    )
    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x)
    @patch('ml.prediction.calculate_sequence_length')
    def test_run_prediction_uses_two_stage_runtime_path_when_available(self, mock_calc_seq, mock_norm):
        mock_calc_seq.return_value = 5
        self.mock_platform.label_encoders['room1'].classes_ = np.array(['unoccupied', 'walking'])
        self.mock_platform.label_encoders['room1'].inverse_transform.side_effect = (
            lambda arr: np.array(['unoccupied' if int(v) == 0 else 'walking' for v in arr])
        )
        self.mock_platform.room_models['room1'].predict.return_value = np.array([[0.99, 0.01]] * 6)

        stage_a_model = MagicMock()
        stage_a_model.predict.return_value = np.array([[0.1, 0.9]] * 6)
        self.mock_platform.two_stage_core_models = {
            'room1': {
                'stage_a_model': stage_a_model,
                'stage_b_model': None,
                'num_classes': 2,
                'excluded_class_ids': [0],
                'occupied_class_ids': [1],
                'primary_occupied_class_id': 1,
            }
        }
        self.mock_platform.create_sequences.return_value = np.zeros((6, 5, 2))
        sensor_data = {
            'room1': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='10s'),
                's1': np.random.randn(10),
                's2': np.random.randn(10),
            })
        }
        self.mock_platform.preprocess_with_resampling.return_value = sensor_data['room1']

        predictions = self.pipeline.run_prediction(sensor_data=sensor_data, loaded_rooms=['room1'], seq_length=5)
        pred_df = predictions['room1']

        self.assertEqual(len(pred_df), 6)
        self.assertTrue((pred_df['predicted_activity'] == 'walking').all())
        stage_a_model.predict.assert_called_once()
        self.mock_platform.room_models['room1'].predict.assert_not_called()

    @patch.dict(
        'os.environ',
        {
            'ENABLE_TWO_STAGE_CORE_RUNTIME': 'true',
            'TWO_STAGE_CORE_RUNTIME_ROOMS': 'room1',
        },
        clear=False,
    )
    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x)
    @patch('ml.prediction.calculate_sequence_length')
    def test_run_prediction_two_stage_respects_stage_a_occupied_threshold(self, mock_calc_seq, mock_norm):
        mock_calc_seq.return_value = 5
        self.mock_platform.label_encoders['room1'].classes_ = np.array(['unoccupied', 'walking'])
        self.mock_platform.label_encoders['room1'].inverse_transform.side_effect = (
            lambda arr: np.array(['unoccupied' if int(v) == 0 else 'walking' for v in arr])
        )
        self.mock_platform.room_models['room1'].predict.return_value = np.array([[0.01, 0.99]] * 6)

        stage_a_model = MagicMock()
        # Occupancy probability is high but below configured threshold.
        stage_a_model.predict.return_value = np.array([[0.10, 0.90]] * 6)
        self.mock_platform.two_stage_core_models = {
            'room1': {
                'stage_a_model': stage_a_model,
                'stage_b_model': None,
                'num_classes': 2,
                'excluded_class_ids': [0],
                'occupied_class_ids': [1],
                'primary_occupied_class_id': 1,
                'stage_a_occupied_threshold': 0.95,
            }
        }
        self.mock_platform.create_sequences.return_value = np.zeros((6, 5, 2))
        sensor_data = {
            'room1': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='10s'),
                's1': np.random.randn(10),
                's2': np.random.randn(10),
            })
        }
        self.mock_platform.preprocess_with_resampling.return_value = sensor_data['room1']

        predictions = self.pipeline.run_prediction(sensor_data=sensor_data, loaded_rooms=['room1'], seq_length=5)
        pred_df = predictions['room1']

        self.assertEqual(len(pred_df), 6)
        self.assertTrue((pred_df['predicted_activity'] == 'unoccupied').all())
        stage_a_model.predict.assert_called_once()
        self.mock_platform.room_models['room1'].predict.assert_not_called()

    @patch.dict(
        'os.environ',
        {
            'ENABLE_TWO_STAGE_CORE_RUNTIME': 'true',
            'TWO_STAGE_CORE_RUNTIME_ROOMS': 'room1',
            'TWO_STAGE_CORE_STRICT_ROUTING': 'true',
        },
        clear=False,
    )
    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x)
    @patch('ml.prediction.calculate_sequence_length')
    def test_run_prediction_two_stage_strict_routing_blocks_excluded_argmax(self, mock_calc_seq, mock_norm):
        mock_calc_seq.return_value = 5
        self.mock_platform.label_encoders['room1'].classes_ = np.array(
            ['unoccupied', 'sleep', 'bedroom_normal_use']
        )
        self.mock_platform.label_encoders['room1'].inverse_transform.side_effect = (
            lambda arr: np.array(
                [
                    'unoccupied' if int(v) == 0 else 'sleep' if int(v) == 1 else 'bedroom_normal_use'
                    for v in arr
                ]
            )
        )
        self.mock_platform.class_thresholds = {'room1': {'0': 0.0, '1': 0.0, '2': 0.0}}
        self.mock_platform.room_models['room1'].predict.return_value = np.array([[0.99, 0.005, 0.005]] * 6)

        stage_a_model = MagicMock()
        stage_a_model.predict.return_value = np.array([[0.48, 0.52]] * 6)
        stage_b_model = MagicMock()
        stage_b_model.predict.return_value = np.array([[0.50, 0.50]] * 6)
        self.mock_platform.two_stage_core_models = {
            'room1': {
                'stage_a_model': stage_a_model,
                'stage_b_model': stage_b_model,
                'num_classes': 3,
                'excluded_class_ids': [0],
                'occupied_class_ids': [1, 2],
                'primary_occupied_class_id': 1,
                'stage_a_occupied_threshold': 0.5,
            }
        }
        self.mock_platform.create_sequences.return_value = np.zeros((6, 5, 2))
        sensor_data = {
            'room1': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='10s'),
                's1': np.random.randn(10),
                's2': np.random.randn(10),
            })
        }
        self.mock_platform.preprocess_with_resampling.return_value = sensor_data['room1']

        predictions = self.pipeline.run_prediction(sensor_data=sensor_data, loaded_rooms=['room1'], seq_length=5)
        pred_df = predictions['room1']

        self.assertEqual(len(pred_df), 6)
        self.assertTrue((pred_df['predicted_activity'] != 'unoccupied').all())
        stage_a_model.predict.assert_called_once()
        stage_b_model.predict.assert_called_once()
        self.mock_platform.room_models['room1'].predict.assert_not_called()

    @patch.dict(
        'os.environ',
        {
            'ENABLE_TWO_STAGE_CORE_RUNTIME': 'true',
            'TWO_STAGE_CORE_RUNTIME_ROOMS': 'room1',
            'TWO_STAGE_CORE_STRICT_ROUTING': 'false',
        },
        clear=False,
    )
    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x)
    @patch('ml.prediction.calculate_sequence_length')
    def test_run_prediction_two_stage_soft_routing_can_pick_excluded(self, mock_calc_seq, mock_norm):
        mock_calc_seq.return_value = 5
        self.mock_platform.label_encoders['room1'].classes_ = np.array(
            ['unoccupied', 'sleep', 'bedroom_normal_use']
        )
        self.mock_platform.label_encoders['room1'].inverse_transform.side_effect = (
            lambda arr: np.array(
                [
                    'unoccupied' if int(v) == 0 else 'sleep' if int(v) == 1 else 'bedroom_normal_use'
                    for v in arr
                ]
            )
        )
        self.mock_platform.class_thresholds = {'room1': {'0': 0.0, '1': 0.0, '2': 0.0}}
        self.mock_platform.room_models['room1'].predict.return_value = np.array([[0.99, 0.005, 0.005]] * 6)

        stage_a_model = MagicMock()
        stage_a_model.predict.return_value = np.array([[0.48, 0.52]] * 6)
        stage_b_model = MagicMock()
        stage_b_model.predict.return_value = np.array([[0.50, 0.50]] * 6)
        self.mock_platform.two_stage_core_models = {
            'room1': {
                'stage_a_model': stage_a_model,
                'stage_b_model': stage_b_model,
                'num_classes': 3,
                'excluded_class_ids': [0],
                'occupied_class_ids': [1, 2],
                'primary_occupied_class_id': 1,
                'stage_a_occupied_threshold': 0.5,
            }
        }
        self.mock_platform.create_sequences.return_value = np.zeros((6, 5, 2))
        sensor_data = {
            'room1': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='10s'),
                's1': np.random.randn(10),
                's2': np.random.randn(10),
            })
        }
        self.mock_platform.preprocess_with_resampling.return_value = sensor_data['room1']

        predictions = self.pipeline.run_prediction(sensor_data=sensor_data, loaded_rooms=['room1'], seq_length=5)
        pred_df = predictions['room1']

        self.assertEqual(len(pred_df), 6)
        self.assertTrue((pred_df['predicted_activity'] == 'unoccupied').all())
        stage_a_model.predict.assert_called_once()
        stage_b_model.predict.assert_called_once()
        self.mock_platform.room_models['room1'].predict.assert_not_called()

    @patch.dict(
        'os.environ',
        {
            'ENABLE_TWO_STAGE_CORE_RUNTIME': 'true',
            'TWO_STAGE_CORE_RUNTIME_ROOMS': 'bedroom',
            'ENABLE_BEDROOM_SLEEP_CONTINUITY': 'true',
            'BEDROOM_SLEEP_BRIDGE_MAX_STEPS': '3',
            'BEDROOM_SLEEP_BRIDGE_MIN_OCC_PROB': '0.30',
        },
        clear=False,
    )
    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: str(x).strip().lower())
    @patch('ml.prediction.calculate_sequence_length')
    def test_run_prediction_two_stage_bridges_short_bedroom_sleep_gaps(self, mock_calc_seq, mock_norm):
        mock_calc_seq.return_value = 5
        self.mock_platform.label_encoders['bedroom'] = MagicMock()
        self.mock_platform.room_models['bedroom'] = MagicMock()
        self.mock_platform.label_encoders['bedroom'].classes_ = np.array(
            ['sleep', 'bedroom_normal_use', 'unoccupied']
        )
        self.mock_platform.label_encoders['bedroom'].inverse_transform.side_effect = (
            lambda arr: np.array(
                [
                    'sleep' if int(v) == 0 else 'bedroom_normal_use' if int(v) == 1 else 'unoccupied'
                    for v in arr
                ]
            )
        )
        self.mock_platform.class_thresholds = {'bedroom': {'0': 0.0, '1': 0.0, '2': 0.0}}
        self.mock_platform.room_models['bedroom'].predict.return_value = np.array([[0.99, 0.005, 0.005]] * 6)

        stage_a_model = MagicMock()
        stage_a_model.predict.return_value = np.array(
            [
                [0.05, 0.95],
                [0.05, 0.95],
                [0.60, 0.40],
                [0.60, 0.40],
                [0.05, 0.95],
                [0.05, 0.95],
            ]
        )
        stage_b_model = MagicMock()
        stage_b_model.predict.return_value = np.array(
            [
                [0.90, 0.10],
                [0.90, 0.10],
                [0.10, 0.90],
                [0.10, 0.90],
                [0.90, 0.10],
                [0.90, 0.10],
            ]
        )
        self.mock_platform.two_stage_core_models = {
            'bedroom': {
                'stage_a_model': stage_a_model,
                'stage_b_model': stage_b_model,
                'num_classes': 3,
                'excluded_class_ids': [2],
                'occupied_class_ids': [0, 1],
                'primary_occupied_class_id': 0,
                'stage_a_occupied_threshold': 0.5,
            }
        }
        self.mock_platform.create_sequences.return_value = np.zeros((6, 5, 2))
        sensor_data = {
            'bedroom': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='10s'),
                's1': np.random.randn(10),
                's2': np.random.randn(10),
            })
        }
        self.mock_platform.preprocess_with_resampling.return_value = sensor_data['bedroom']

        predictions = self.pipeline.run_prediction(sensor_data=sensor_data, loaded_rooms=['bedroom'], seq_length=5)
        pred_df = predictions['bedroom']

        self.assertEqual(len(pred_df), 6)
        self.assertTrue((pred_df['predicted_activity'] == 'sleep').all())
        stage_a_model.predict.assert_called_once()
        stage_b_model.predict.assert_called_once()

    @patch.dict(
        'os.environ',
        {
            'RUNTIME_UNKNOWN_ENABLED': 'true',
            'RUNTIME_UNKNOWN_ROOMS': 'room1',
            'RUNTIME_UNKNOWN_NIGHT_ONLY': 'true',
            'RUNTIME_UNKNOWN_NIGHT_HOURS': '22-6',
            'RUNTIME_UNKNOWN_MIN_CONF': '0.90',
            'RUNTIME_UNKNOWN_RATE_ROOM_CAP': '0.50',
            'RUNTIME_UNKNOWN_RATE_GLOBAL_CAP': '0.50',
        },
        clear=False,
    )
    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x)
    @patch('ml.prediction.calculate_sequence_length')
    def test_run_prediction_applies_scoped_runtime_unknown_with_caps(self, mock_calc_seq, mock_norm):
        mock_calc_seq.return_value = 5
        self.mock_platform.class_thresholds = {'room1': {'0': 0.99, '1': 0.99}}
        probs = np.array([[0.8, 0.2]] * 6)  # Always low confidence vs threshold=0.99
        self.mock_platform.room_models['room1'].predict.return_value = probs
        self.mock_platform.label_encoders['room1'].inverse_transform.return_value = ['walking'] * 6
        self.mock_platform.create_sequences.return_value = np.zeros((6, 5, 2))

        sensor_data = {
            'room1': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01 23:00:00', periods=10, freq='10s'),
                's1': np.random.randn(10),
                's2': np.random.randn(10)
            })
        }
        self.mock_platform.preprocess_with_resampling.return_value = sensor_data['room1']

        predictions = self.pipeline.run_prediction(sensor_data=sensor_data, loaded_rooms=['room1'], seq_length=5)
        pred_df = predictions['room1']
        # 6 windows total, room/global cap 50% => at most 3 unknown
        self.assertEqual(int((pred_df['predicted_activity'] == 'unknown').sum()), 3)
        self.assertEqual(int(pred_df['is_runtime_unknown'].sum()), 3)
        self.assertEqual(int((pred_df['predicted_activity'] == 'low_confidence').sum()), 3)

    @patch.dict(
        'os.environ',
        {
            'RUNTIME_UNKNOWN_ENABLED': 'true',
            'RUNTIME_UNKNOWN_ROOMS': 'room1',
            'RUNTIME_UNKNOWN_NIGHT_ONLY': 'true',
            'RUNTIME_UNKNOWN_NIGHT_HOURS': '22-6',
            'RUNTIME_UNKNOWN_MIN_CONF': '0.90',
            'RUNTIME_UNKNOWN_RATE_ROOM_CAP': '1.00',
            'RUNTIME_UNKNOWN_RATE_GLOBAL_CAP': '1.00',
        },
        clear=False,
    )
    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x)
    @patch('ml.prediction.calculate_sequence_length')
    def test_run_prediction_scoped_runtime_unknown_skips_daytime(self, mock_calc_seq, mock_norm):
        mock_calc_seq.return_value = 5
        self.mock_platform.class_thresholds = {'room1': {'0': 0.99, '1': 0.99}}
        probs = np.array([[0.8, 0.2]] * 6)
        self.mock_platform.room_models['room1'].predict.return_value = probs
        self.mock_platform.label_encoders['room1'].inverse_transform.return_value = ['walking'] * 6
        self.mock_platform.create_sequences.return_value = np.zeros((6, 5, 2))

        sensor_data = {
            'room1': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01 12:00:00', periods=10, freq='10s'),
                's1': np.random.randn(10),
                's2': np.random.randn(10)
            })
        }
        self.mock_platform.preprocess_with_resampling.return_value = sensor_data['room1']

        predictions = self.pipeline.run_prediction(sensor_data=sensor_data, loaded_rooms=['room1'], seq_length=5)
        pred_df = predictions['room1']
        self.assertEqual(int((pred_df['predicted_activity'] == 'unknown').sum()), 0)
        self.assertEqual(int(pred_df['is_runtime_unknown'].sum()), 0)

    @patch.dict(
        'os.environ',
        {
            'BETA6_PHASE4_RUNTIME_ENABLED': 'true',
            'RUNTIME_UNKNOWN_ENABLED': 'true',
            'RUNTIME_UNKNOWN_ROOMS': 'other_room',  # skip legacy scoped-unknown conversion branch
        },
        clear=False,
    )
    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x)
    @patch('ml.prediction.calculate_sequence_length')
    @patch('ml.prediction.load_unknown_policy')
    @patch('ml.prediction.infer_with_unknown_path')
    def test_run_prediction_applies_beta6_unknown_abstain_runtime_hook(
        self, mock_infer_unknown, mock_load_policy, mock_calc_seq, mock_norm
    ):
        mock_calc_seq.return_value = 5
        mock_load_policy.return_value = MagicMock()
        probs = np.array([[0.9, 0.1]] * 6)
        self.mock_platform.room_models['room1'].predict.return_value = probs
        self.mock_platform.label_encoders['room1'].inverse_transform.return_value = ['walking'] * 6
        self.mock_platform.create_sequences.return_value = np.zeros((6, 5, 2))
        mock_infer_unknown.return_value = {
            "labels": ["walking"] * 6,
            "uncertainty_states": [None, "unknown", "outside_sensed_space", "low_confidence", None, None],
            "confidence": [0.9] * 6,
            "entropy": [0.1] * 6,
            "abstain_rate": 0.5,
        }

        sensor_data = {
            'room1': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='10s'),
                's1': np.random.randn(10),
                's2': np.random.randn(10)
            })
        }
        self.mock_platform.preprocess_with_resampling.return_value = sensor_data['room1']
        os.environ.pop("ENABLE_BETA6_UNKNOWN_ABSTAIN_RUNTIME", None)
        predictions = self.pipeline.run_prediction(sensor_data=sensor_data, loaded_rooms=['room1'], seq_length=5)
        pred_df = predictions['room1']

        self.assertIn('beta6_uncertainty_state', pred_df.columns)
        self.assertIn('is_beta6_abstain', pred_df.columns)
        self.assertEqual(pred_df.iloc[1]['predicted_activity'], 'unknown')
        self.assertEqual(pred_df.iloc[2]['predicted_activity'], 'unknown')
        self.assertEqual(pred_df.iloc[3]['predicted_activity'], 'low_confidence')
        self.assertEqual(int(pred_df['is_beta6_abstain'].sum()), 3)

    @patch.dict('os.environ', {'ENABLE_BETA6_HMM_RUNTIME': 'true'}, clear=False)
    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x)
    @patch('ml.prediction.calculate_sequence_length')
    @patch('ml.prediction.load_duration_prior_policy')
    @patch('ml.prediction.decode_hmm_with_duration_priors')
    def test_run_prediction_applies_beta6_hmm_runtime_hook(
        self, mock_decode_hmm, mock_load_duration_policy, mock_calc_seq, mock_norm
    ):
        mock_calc_seq.return_value = 5
        mock_load_duration_policy.return_value = MagicMock()
        probs = np.array([
            [0.9, 0.1],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
        ])
        self.mock_platform.room_models['room1'].predict.return_value = probs
        self.mock_platform.label_encoders['room1'].inverse_transform.return_value = [
            'walking', 'walking', 'sitting', 'walking', 'walking', 'walking'
        ]
        self.mock_platform.create_sequences.return_value = np.zeros((6, 5, 2))
        mock_decode_hmm.return_value = MagicMock(
            labels=['walking', 'sitting', 'sitting', 'walking', 'walking', 'walking'],
            ping_pong_rate=0.0,
        )

        sensor_data = {
            'room1': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='10s'),
                's1': np.random.randn(10),
                's2': np.random.randn(10)
            })
        }
        self.mock_platform.preprocess_with_resampling.return_value = sensor_data['room1']

        predictions = self.pipeline.run_prediction(sensor_data=sensor_data, loaded_rooms=['room1'], seq_length=5)
        pred_df = predictions['room1']

        self.assertEqual(pred_df.iloc[1]['predicted_activity'], 'sitting')
        self.assertEqual(pred_df.iloc[1]['predicted_top1_label'], 'sitting')
        self.assertEqual(pred_df.iloc[1]['predicted_activity_raw'], 'walking')
        mock_decode_hmm.assert_called_once()

    @patch.dict('os.environ', {'ENABLE_BETA6_HMM_RUNTIME': 'true'}, clear=False)
    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x)
    @patch('ml.prediction.calculate_sequence_length')
    @patch('ml.prediction.load_duration_prior_policy')
    @patch('ml.prediction.decode_hmm_with_duration_priors')
    def test_run_prediction_beta6_hmm_runtime_fails_closed(
        self, mock_decode_hmm, mock_load_duration_policy, mock_calc_seq, mock_norm
    ):
        mock_calc_seq.return_value = 5
        mock_load_duration_policy.return_value = MagicMock()
        mock_decode_hmm.side_effect = RuntimeError("decoder boom")
        probs = np.array([[0.9, 0.1]] * 6)
        self.mock_platform.room_models['room1'].predict.return_value = probs
        self.mock_platform.label_encoders['room1'].inverse_transform.return_value = ['walking'] * 6
        self.mock_platform.create_sequences.return_value = np.zeros((6, 5, 2))
        sensor_data = {
            'room1': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='10s'),
                's1': np.random.randn(10),
                's2': np.random.randn(10)
            })
        }
        self.mock_platform.preprocess_with_resampling.return_value = sensor_data['room1']

        with self.assertRaises(PredictionError):
            self.pipeline.run_prediction(sensor_data=sensor_data, loaded_rooms=['room1'], seq_length=5)

    @patch.dict('os.environ', {'ENABLE_BETA6_UNKNOWN_ABSTAIN_RUNTIME': 'true'}, clear=False)
    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x)
    @patch('ml.prediction.calculate_sequence_length')
    @patch('ml.prediction.load_unknown_policy')
    def test_run_prediction_beta6_unknown_abstain_runtime_fails_closed(
        self, mock_load_policy, mock_calc_seq, mock_norm
    ):
        mock_calc_seq.return_value = 5
        mock_load_policy.side_effect = RuntimeError("bad policy")
        probs = np.array([[0.9, 0.1]] * 6)
        self.mock_platform.room_models['room1'].predict.return_value = probs
        self.mock_platform.label_encoders['room1'].inverse_transform.return_value = ['walking'] * 6
        self.mock_platform.create_sequences.return_value = np.zeros((6, 5, 2))
        sensor_data = {
            'room1': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='10s'),
                's1': np.random.randn(10),
                's2': np.random.randn(10)
            })
        }
        self.mock_platform.preprocess_with_resampling.return_value = sensor_data['room1']

        with self.assertRaises(PredictionError):
            self.pipeline.run_prediction(sensor_data=sensor_data, loaded_rooms=['room1'], seq_length=5)

    @patch.dict(
        'os.environ',
        {
            'BETA6_PHASE4_RUNTIME_ENABLED': 'true',
            'BETA6_PHASE4_RUNTIME_ROOMS': 'other_room',
            'RUNTIME_UNKNOWN_ENABLED': 'true',
            'RUNTIME_UNKNOWN_ROOMS': 'other_room',
        },
        clear=False,
    )
    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x)
    @patch('ml.prediction.calculate_sequence_length')
    @patch('ml.prediction.load_unknown_policy')
    @patch('ml.prediction.infer_with_unknown_path')
    def test_run_prediction_beta6_runtime_room_scope_blocks_hook_for_other_rooms(
        self, mock_infer_unknown, mock_load_policy, mock_calc_seq, mock_norm
    ):
        mock_calc_seq.return_value = 5
        mock_load_policy.return_value = MagicMock()
        probs = np.array([[0.9, 0.1]] * 6)
        self.mock_platform.room_models['room1'].predict.return_value = probs
        self.mock_platform.label_encoders['room1'].inverse_transform.return_value = ['walking'] * 6
        self.mock_platform.create_sequences.return_value = np.zeros((6, 5, 2))
        sensor_data = {
            'room1': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='10s'),
                's1': np.random.randn(10),
                's2': np.random.randn(10)
            })
        }
        self.mock_platform.preprocess_with_resampling.return_value = sensor_data['room1']

        predictions = self.pipeline.run_prediction(sensor_data=sensor_data, loaded_rooms=['room1'], seq_length=5)
        pred_df = predictions['room1']
        self.assertNotIn('beta6_uncertainty_state', pred_df.columns)
        self.assertNotIn('is_beta6_abstain', pred_df.columns)
        self.assertEqual(int((pred_df['predicted_activity'] == 'unknown').sum()), 0)
        mock_infer_unknown.assert_not_called()

    @patch('ml.prediction.fetch_golden_samples')
    def test_apply_golden_samples(self, mock_fetch_golden):
        """Test applying corrections from golden samples."""
        timestamp = pd.Timestamp('2023-01-01 10:00:00')
        
        # Predictions df
        pred_df = pd.DataFrame({
            'timestamp': [timestamp, timestamp + pd.Timedelta('10s')],
            'predicted_activity': ['walking', 'walking'],
            'confidence': [0.8, 0.8]
        })
        
        predictions = {'room1': pred_df}
        
        # Mock golden samples to override first row to 'falling'
        mock_fetch_golden.return_value = pd.DataFrame({
            'timestamp': [timestamp],
            'activity': ['falling']
        })
        
        # Apply
        corrected = self.pipeline.apply_golden_samples(predictions, 'elder1')
        
        # Verify
        result_df = corrected['room1']
        self.assertEqual(result_df.iloc[0]['predicted_activity'], 'falling') # Overridden
        self.assertEqual(result_df.iloc[0]['confidence'], 1.0) # Corrected confidence
        self.assertEqual(result_df.iloc[1]['predicted_activity'], 'walking') # Unchanged
        self.assertEqual(result_df.iloc[1]['confidence'], 0.8)

    @patch('ml.prediction.normalize_room_name', side_effect=lambda x: x)
    @patch('ml.prediction.adapter')
    @patch('ml.prediction.regenerate_segments')
    @patch('ml.prediction.validate_activity_for_room')
    def test_save_predictions_to_db(self, mock_validate, mock_regen, mock_adapter, mock_norm):
        """Test saving predictions to DB."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_adapter.get_connection.return_value.__enter__.return_value = mock_conn
        
        # IMPORTANT: Mock fetchone to return None (no existing corrected row)
        mock_cursor.fetchone.return_value = None
        
        mock_validate.return_value = 'walking' 
        
        pred_df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2023-01-01 10:00:00')],
            'predicted_activity': ['walking'],
            'confidence': [0.9]
        })
        
        self.pipeline.save_predictions_to_db({'room1': pred_df}, 'elder1')
        
        # Verify DB calls
        calls = mock_cursor.execute.call_args_list
        insert_calls = [c for c in calls if 'INSERT INTO adl_history' in c[0][0]]
        self.assertTrue(len(insert_calls) > 0)
        
        # Verify regen called
        mock_regen.assert_called_once()

if __name__ == '__main__':
    unittest.main()
