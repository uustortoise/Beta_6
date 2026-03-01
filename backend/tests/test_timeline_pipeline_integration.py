"""
Tests for WS-5: Timeline Pipeline Integration

Tests shadow mode integration and feature flag safety.
"""

import sys
import os
import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.timeline_pipeline_integration import (
    is_timeline_multitask_enabled,
    is_timeline_decoder_v2_enabled,
    is_timeline_shadow_mode,
    get_timeline_feature_flags,
    validate_timeline_safety,
    should_run_timeline_path,
    create_timeline_qc_report,
    TimelineShadowArtifacts,
    TimelineShadowPipeline,
    ENABLE_TIMELINE_MULTITASK,
    TIMELINE_DECODER_V2,
    TIMELINE_SHADOW_MODE,
)


class TestFeatureFlags(unittest.TestCase):
    """Tests for feature flag detection."""
    
    def setUp(self):
        """Clear environment variables before each test."""
        for var in [ENABLE_TIMELINE_MULTITASK, TIMELINE_DECODER_V2, TIMELINE_SHADOW_MODE]:
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clear environment variables after each test."""
        for var in [ENABLE_TIMELINE_MULTITASK, TIMELINE_DECODER_V2, TIMELINE_SHADOW_MODE]:
            if var in os.environ:
                del os.environ[var]
    
    def test_multitask_disabled_by_default(self):
        """Test that multitask is disabled by default."""
        self.assertFalse(is_timeline_multitask_enabled())
    
    def test_multitask_enabled_true(self):
        """Test enabling multitask with 'true'."""
        os.environ[ENABLE_TIMELINE_MULTITASK] = 'true'
        self.assertTrue(is_timeline_multitask_enabled())
    
    def test_multitask_enabled_1(self):
        """Test enabling multitask with '1'."""
        os.environ[ENABLE_TIMELINE_MULTITASK] = '1'
        self.assertTrue(is_timeline_multitask_enabled())
    
    def test_multitask_case_insensitive(self):
        """Test that flag is case insensitive."""
        os.environ[ENABLE_TIMELINE_MULTITASK] = 'TRUE'
        self.assertTrue(is_timeline_multitask_enabled())
    
    def test_decoder_v2_disabled_by_default(self):
        """Test that decoder v2 is disabled by default."""
        self.assertFalse(is_timeline_decoder_v2_enabled())
    
    def test_decoder_v2_enabled(self):
        """Test enabling decoder v2."""
        os.environ[TIMELINE_DECODER_V2] = 'true'
        self.assertTrue(is_timeline_decoder_v2_enabled())
    
    def test_shadow_mode_disabled_by_default(self):
        """Test that shadow mode is disabled by default."""
        self.assertFalse(is_timeline_shadow_mode())
    
    def test_shadow_mode_enabled(self):
        """Test enabling shadow mode."""
        os.environ[TIMELINE_SHADOW_MODE] = 'true'
        self.assertTrue(is_timeline_shadow_mode())


class TestFeatureFlagQueries(unittest.TestCase):
    """Tests for feature flag queries."""
    
    def setUp(self):
        """Clear environment variables."""
        for var in [ENABLE_TIMELINE_MULTITASK, TIMELINE_DECODER_V2, TIMELINE_SHADOW_MODE]:
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clear environment variables."""
        for var in [ENABLE_TIMELINE_MULTITASK, TIMELINE_DECODER_V2, TIMELINE_SHADOW_MODE]:
            if var in os.environ:
                del os.environ[var]
    
    def test_get_all_flags_default(self):
        """Test getting all flags when disabled."""
        flags = get_timeline_feature_flags()
        
        self.assertEqual(flags['ENABLE_TIMELINE_MULTITASK'], False)
        self.assertEqual(flags['TIMELINE_DECODER_V2'], False)
        self.assertEqual(flags['TIMELINE_SHADOW_MODE'], False)
    
    def test_get_all_flags_enabled(self):
        """Test getting all flags when enabled."""
        os.environ[ENABLE_TIMELINE_MULTITASK] = 'true'
        os.environ[TIMELINE_DECODER_V2] = 'true'
        os.environ[TIMELINE_SHADOW_MODE] = 'true'
        
        flags = get_timeline_feature_flags()
        
        self.assertEqual(flags['ENABLE_TIMELINE_MULTITASK'], True)
        self.assertEqual(flags['TIMELINE_DECODER_V2'], True)
        self.assertEqual(flags['TIMELINE_SHADOW_MODE'], True)


class TestSafetyValidation(unittest.TestCase):
    """Tests for safety validation."""
    
    def setUp(self):
        """Clear environment variables."""
        for var in [ENABLE_TIMELINE_MULTITASK, TIMELINE_DECODER_V2, TIMELINE_SHADOW_MODE]:
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clear environment variables."""
        for var in [ENABLE_TIMELINE_MULTITASK, TIMELINE_DECODER_V2, TIMELINE_SHADOW_MODE]:
            if var in os.environ:
                del os.environ[var]
    
    def test_no_warnings_when_all_off(self):
        """Test no warnings when all flags are off."""
        warnings = validate_timeline_safety()
        
        self.assertEqual(len(warnings), 0)
    
    def test_warning_multitask_without_shadow(self):
        """Test warning when multitask enabled without shadow mode."""
        os.environ[ENABLE_TIMELINE_MULTITASK] = 'true'
        os.environ[TIMELINE_SHADOW_MODE] = 'false'
        
        warnings = validate_timeline_safety()
        
        self.assertEqual(len(warnings), 1)
        self.assertIn('ENABLE_TIMELINE_MULTITASK', warnings[0])
        self.assertIn('shadow mode', warnings[0].lower())
    
    def test_warning_decoder_without_shadow(self):
        """Test warning when decoder v2 enabled without shadow mode."""
        os.environ[TIMELINE_DECODER_V2] = 'true'
        os.environ[TIMELINE_SHADOW_MODE] = 'false'
        
        warnings = validate_timeline_safety()
        
        self.assertEqual(len(warnings), 1)
        self.assertIn('TIMELINE_DECODER_V2', warnings[0])
    
    def test_no_warnings_with_shadow_mode(self):
        """Test no warnings when shadow mode is enabled."""
        os.environ[ENABLE_TIMELINE_MULTITASK] = 'true'
        os.environ[TIMELINE_DECODER_V2] = 'true'
        os.environ[TIMELINE_SHADOW_MODE] = 'true'
        
        warnings = validate_timeline_safety()
        
        self.assertEqual(len(warnings), 0)


class TestShouldRunTimelinePath(unittest.TestCase):
    """Tests for timeline path decision."""
    
    def setUp(self):
        """Clear environment variables."""
        for var in [ENABLE_TIMELINE_MULTITASK, TIMELINE_DECODER_V2, TIMELINE_SHADOW_MODE]:
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clear environment variables."""
        for var in [ENABLE_TIMELINE_MULTITASK, TIMELINE_DECODER_V2, TIMELINE_SHADOW_MODE]:
            if var in os.environ:
                del os.environ[var]
    
    def test_not_run_when_all_disabled(self):
        """Test timeline path not run when all flags disabled."""
        self.assertFalse(should_run_timeline_path())
    
    def test_run_when_multitask_enabled(self):
        """Test timeline path runs when multitask enabled."""
        os.environ[ENABLE_TIMELINE_MULTITASK] = 'true'
        self.assertTrue(should_run_timeline_path())
    
    def test_run_when_decoder_enabled(self):
        """Test timeline path runs when decoder v2 enabled."""
        os.environ[TIMELINE_DECODER_V2] = 'true'
        self.assertTrue(should_run_timeline_path())
    
    def test_run_when_shadow_enabled(self):
        """Test timeline path runs when shadow mode enabled."""
        os.environ[TIMELINE_SHADOW_MODE] = 'true'
        self.assertTrue(should_run_timeline_path())


class TestTimelineShadowArtifacts(unittest.TestCase):
    """Tests for TimelineShadowArtifacts."""
    
    def test_creation(self):
        """Test creating artifacts container."""
        artifacts = TimelineShadowArtifacts()
        
        self.assertIsNone(artifacts.windows_df)
        self.assertIsNone(artifacts.episodes_df)
        self.assertIsNone(artifacts.qc_report)
    
    def test_save_windows(self):
        """Test saving windows DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = TimelineShadowArtifacts(
                windows_df=pd.DataFrame({'a': [1, 2, 3]}),
            )
            
            paths = artifacts.save(Path(tmpdir))
            
            self.assertIn('timeline_windows', paths)
            self.assertTrue(paths['timeline_windows'].exists())
    
    def test_save_episodes(self):
        """Test saving episodes DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = TimelineShadowArtifacts(
                episodes_df=pd.DataFrame({'b': [4, 5, 6]}),
            )
            
            paths = artifacts.save(Path(tmpdir))
            
            self.assertIn('timeline_episodes', paths)
            self.assertTrue(paths['timeline_episodes'].exists())
    
    def test_save_qc_report(self):
        """Test saving QC report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = TimelineShadowArtifacts(
                qc_report={'test': 'data'},
            )
            
            paths = artifacts.save(Path(tmpdir))
            
            self.assertIn('timeline_qc', paths)
            self.assertTrue(paths['timeline_qc'].exists())


class TestTimelineShadowPipeline(unittest.TestCase):
    """Tests for TimelineShadowPipeline processing."""

    def test_process_split_generates_windows_episodes_and_qc(self):
        pipeline = TimelineShadowPipeline(adl_registry=None)
        labels = np.array(
            ["unoccupied", "sleeping", "sleeping", "unoccupied", "reading", "reading", "unknown"],
            dtype=object,
        )
        timestamps = np.arange(len(labels))

        artifacts = pipeline.process_split(
            train_data={"labels": labels, "timestamps": timestamps},
            val_data={},
            split_id="4->5",
        )

        self.assertIsNotNone(artifacts.qc_report)
        self.assertIsNotNone(artifacts.windows_df)
        self.assertIsNotNone(artifacts.episodes_df)
        self.assertEqual(len(artifacts.windows_df), len(labels))
        self.assertGreaterEqual(len(artifacts.episodes_df), 1)

    def test_process_split_does_not_crash_without_labels(self):
        pipeline = TimelineShadowPipeline(
            adl_registry=None,
            timeline_heads=object(),  # Triggers readiness branch without requiring TF
            timeline_decoder=object(),
        )

        artifacts = pipeline.process_split(
            train_data={},  # no labels
            val_data={},
            split_id="4->5",
        )

        self.assertIsNotNone(artifacts.qc_report)
        self.assertIn("labels_available", artifacts.qc_report)
        self.assertEqual(artifacts.qc_report["labels_available"], False)


class TestQCReport(unittest.TestCase):
    """Tests for QC report generation."""
    
    def setUp(self):
        """Clear environment variables."""
        for var in [ENABLE_TIMELINE_MULTITASK, TIMELINE_DECODER_V2, TIMELINE_SHADOW_MODE]:
            if var in os.environ:
                del os.environ[var]
    
    def test_basic_qc_report(self):
        """Test basic QC report generation."""
        report = create_timeline_qc_report()
        
        self.assertIn('timestamp', report)
        self.assertIn('feature_flags', report)
        self.assertIn('safety_warnings', report)
    
    def test_qc_report_with_predictions(self):
        """Test QC report with predictions DataFrame."""
        predictions = pd.DataFrame({'label': ['a', 'b', 'c']})
        
        report = create_timeline_qc_report(predictions_df=predictions)
        
        self.assertIn('predictions', report)
        self.assertEqual(report['predictions']['num_windows'], 3)
    
    def test_qc_report_with_episodes(self):
        """Test QC report with episodes DataFrame."""
        episodes = pd.DataFrame({'duration': [10, 20, 30]})
        
        report = create_timeline_qc_report(episodes_df=episodes)
        
        self.assertIn('episodes', report)
        self.assertEqual(report['episodes']['num_episodes'], 3)
    
    def test_qc_report_with_metrics(self):
        """Test QC report with metrics."""
        metrics = {'accuracy': 0.95, 'f1': 0.92}
        
        report = create_timeline_qc_report(metrics=metrics)
        
        self.assertIn('metrics', report)
        self.assertEqual(report['metrics']['accuracy'], 0.95)


class TestZeroBehaviorChange(unittest.TestCase):
    """
    Tests that verify zero behavior change when flags are OFF.
    
    This is the critical safety requirement from WS-5.
    """
    
    def setUp(self):
        """Ensure all flags are off."""
        for var in [ENABLE_TIMELINE_MULTITASK, TIMELINE_DECODER_V2, TIMELINE_SHADOW_MODE]:
            if var in os.environ:
                del os.environ[var]
    
    def test_feature_flags_all_false_by_default(self):
        """Verify all feature flags default to False."""
        flags = get_timeline_feature_flags()
        
        for flag_name, flag_value in flags.items():
            self.assertFalse(
                flag_value,
                f"{flag_name} should be False by default for zero behavior change"
            )
    
    def test_timeline_path_not_run_by_default(self):
        """Verify timeline path does not run by default."""
        self.assertFalse(
            should_run_timeline_path(),
            "Timeline path should not run when all flags are off"
        )
    
    def test_no_safety_warnings_by_default(self):
        """Verify no safety warnings when all flags are off."""
        warnings = validate_timeline_safety()
        
        self.assertEqual(
            len(warnings), 0,
            "No warnings should be generated when all flags are off"
        )


if __name__ == '__main__':
    unittest.main()
