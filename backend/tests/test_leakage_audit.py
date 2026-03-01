"""
Tests for Leakage Audit Module

Tests mandatory leakage audit artifact generation.
"""

import sys
import unittest
from pathlib import Path
import tempfile
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.leakage_audit import (
    SplitLeakageAudit,
    LeakageAuditReport,
    LeakageAuditor,
    create_leakage_audit_report,
)


class TestSplitLeakageAudit(unittest.TestCase):
    """Tests for SplitLeakageAudit."""
    
    def test_audit_creation(self):
        """Test creating a split audit."""
        audit = SplitLeakageAudit(
            split_id="4->5",
            train_days=[4],
            val_days=[5],
        )
        
        self.assertEqual(audit.split_id, "4->5")
        self.assertEqual(audit.train_days, [4])
        self.assertEqual(audit.val_days, [5])
    
    def test_validate_all_pass(self):
        """Test validation when all checks explicitly pass."""
        audit = SplitLeakageAudit(
            split_id="4->5",
            train_days=[4],
            val_days=[5],
            scaler_fit_on_train_only=True,
            imputer_fit_on_train_only=True,
            feature_stats_fit_on_train_only=True,
            calibrator_fit_on_train_only=True,
            temporal_window_causal_only=True,
            no_centered_windows=True,
            no_future_derived_features=True,
            no_target_stats_from_val=True,
            no_decoder_threshold_tuning_on_val=True,
            calibration_uses_train_slice_only=True,
        )
        
        violations = audit.validate()
        
        self.assertEqual(len(violations), 0)
    
    def test_validate_detects_violations(self):
        """Test validation detects violations (default is fail-closed)."""
        audit = SplitLeakageAudit(
            split_id="4->5",
            train_days=[4],
            val_days=[5],
            # All checks default to False (fail), so this should have many violations
        )
        
        violations = audit.validate()
        
        # Should detect all unmarked checks as violations
        self.assertGreater(len(violations), 0)
        self.assertIn('Scaler fit on validation data', violations)
        self.assertIn('Centered window (looks forward) used', violations)
        self.assertIn('Imputer fit on validation data', violations)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        audit = SplitLeakageAudit(
            split_id="4->5",
            train_days=[4],
            val_days=[5],
            train_sample_count=1000,
            val_sample_count=200,
        )
        
        d = audit.to_dict()
        
        self.assertEqual(d['split_id'], "4->5")
        self.assertEqual(d['train_sample_count'], 1000)
        self.assertEqual(d['val_sample_count'], 200)


class TestLeakageAuditReport(unittest.TestCase):
    """Tests for LeakageAuditReport."""
    
    def test_report_creation(self):
        """Test creating a report."""
        report = LeakageAuditReport(
            run_id="run_001",
            elder_id="HK0011",
            seed=11,
            timestamp="2026-02-17T10:00:00Z",
            git_sha="abc123",
        )
        
        self.assertEqual(report.run_id, "run_001")
        self.assertEqual(report.elder_id, "HK0011")
        self.assertEqual(report.seed, 11)
        self.assertTrue(report.all_splits_pass)
    
    def test_add_split(self):
        """Test adding splits to report."""
        report = LeakageAuditReport(
            run_id="run_001",
            elder_id="HK0011",
            seed=11,
            timestamp="2026-02-17T10:00:00Z",
            git_sha="abc123",
        )
        
        audit = SplitLeakageAudit(
            split_id="4->5",
            train_days=[4],
            val_days=[5],
        )
        
        report.add_split(audit)
        
        self.assertEqual(len(report.splits), 1)
    
    def test_add_split_detects_violations(self):
        """Test that adding split with violations updates report."""
        report = LeakageAuditReport(
            run_id="run_001",
            elder_id="HK0011",
            seed=11,
            timestamp="2026-02-17T10:00:00Z",
            git_sha="abc123",
        )
        
        audit = SplitLeakageAudit(
            split_id="4->5",
            train_days=[4],
            val_days=[5],
            scaler_fit_on_train_only=False,  # Violation
        )
        
        report.add_split(audit)
        
        self.assertFalse(report.all_splits_pass)
        self.assertGreater(len(report.violations), 0)
        self.assertIn("[4->5] Scaler fit on validation data", report.violations)
    
    def test_save_and_load(self):
        """Test saving and loading report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "leakage_audit.json"
            
            report = LeakageAuditReport(
                run_id="run_001",
                elder_id="HK0011",
                seed=11,
                timestamp="2026-02-17T10:00:00Z",
                git_sha="abc123",
            )
            
            audit = SplitLeakageAudit(
                split_id="4->5",
                train_days=[4],
                val_days=[5],
            )
            report.add_split(audit)
            
            # Save
            report.save(path)
            
            # Load
            loaded = LeakageAuditReport.load(path)
            
            self.assertEqual(loaded.run_id, report.run_id)
            self.assertEqual(loaded.elder_id, report.elder_id)
            self.assertEqual(loaded.seed, report.seed)
            self.assertEqual(len(loaded.splits), 1)


class TestLeakageAuditor(unittest.TestCase):
    """Tests for LeakageAuditor."""
    
    def test_auditor_creation(self):
        """Test creating auditor."""
        auditor = LeakageAuditor(
            run_id="run_001",
            elder_id="HK0011",
            seed=11,
            git_sha="abc123",
        )
        
        self.assertEqual(auditor.run_id, "run_001")
    
    def test_create_split_audit(self):
        """Test creating split audit."""
        auditor = LeakageAuditor("run_001", "HK0011", 11)
        
        audit = auditor.create_split_audit(
            split_id="4->5",
            train_days=[4],
            val_days=[5],
        )
        
        self.assertEqual(audit.split_id, "4->5")
        self.assertEqual(audit.train_days, [4])
    
    def test_mark_checks(self):
        """Test marking various checks."""
        auditor = LeakageAuditor("run_001", "HK0011", 11)
        
        audit = auditor.create_split_audit("4->5", [4], [5])
        
        # Mark checks
        auditor.mark_scaler_fit_on_train_only(True)
        auditor.mark_imputer_fit_on_train_only(True)
        auditor.mark_feature_stats_fit_on_train_only(True)
        auditor.mark_calibrator_fit_on_train_only(True)
        auditor.mark_temporal_window_causal_only(True)
        auditor.mark_no_centered_windows(True)
        auditor.mark_no_future_derived_features(True)
        auditor.mark_no_target_stats_from_val(True)
        auditor.mark_no_decoder_threshold_tuning_on_val(True)
        auditor.mark_calibration_uses_train_slice_only(True)
        
        self.assertTrue(audit.scaler_fit_on_train_only)
        self.assertTrue(audit.no_centered_windows)
    
    def test_mark_checks_false(self):
        """Test marking checks as failed."""
        auditor = LeakageAuditor("run_001", "HK0011", 11)
        audit = auditor.create_split_audit("4->5", [4], [5])
        
        auditor.mark_scaler_fit_on_train_only(False)
        
        self.assertFalse(audit.scaler_fit_on_train_only)
    
    def test_set_sample_counts(self):
        """Test setting sample counts."""
        auditor = LeakageAuditor("run_001", "HK0011", 11)
        audit = auditor.create_split_audit("4->5", [4], [5])
        
        auditor.set_sample_counts(train_count=1000, val_count=200)
        
        self.assertEqual(audit.train_sample_count, 1000)
        self.assertEqual(audit.val_sample_count, 200)
    
    def test_set_feature_names(self):
        """Test setting feature names."""
        auditor = LeakageAuditor("run_001", "HK0011", 11)
        audit = auditor.create_split_audit("4->5", [4], [5])
        
        features = ['feature_a', 'feature_b', 'feature_c']
        auditor.set_feature_names(features)
        
        self.assertEqual(audit.feature_names, features)
    
    def test_finalize_split(self):
        """Test finalizing split."""
        auditor = LeakageAuditor("run_001", "HK0011", 11)
        audit = auditor.create_split_audit("4->5", [4], [5])
        
        auditor.finalize_split()
        
        report = auditor.generate_report()
        
        self.assertEqual(len(report.splits), 1)
    
    def test_generate_report_all_pass(self):
        """Test report generation when all checks explicitly pass."""
        auditor = LeakageAuditor("run_001", "HK0011", 11, git_sha="abc123")
        
        audit = auditor.create_split_audit("4->5", [4], [5])
        # Mark ALL checks as passed (fail-closed: must explicitly mark each)
        auditor.mark_scaler_fit_on_train_only(True)
        auditor.mark_imputer_fit_on_train_only(True)
        auditor.mark_feature_stats_fit_on_train_only(True)
        auditor.mark_calibrator_fit_on_train_only(True)
        auditor.mark_temporal_window_causal_only(True)
        auditor.mark_no_centered_windows(True)
        auditor.mark_no_future_derived_features(True)
        auditor.mark_no_target_stats_from_val(True)
        auditor.mark_no_decoder_threshold_tuning_on_val(True)
        auditor.mark_calibration_uses_train_slice_only(True)
        auditor.finalize_split()
        
        report = auditor.generate_report()
        
        self.assertTrue(report.all_splits_pass)
        self.assertEqual(len(report.violations), 0)
    
    def test_generate_report_with_violations(self):
        """Test report generation with violations (fail-closed defaults)."""
        auditor = LeakageAuditor("run_001", "HK0011", 11, git_sha="abc123")
        
        audit = auditor.create_split_audit("4->5", [4], [5])
        # Only mark 2 checks as passed, leave rest as False (violations)
        auditor.mark_scaler_fit_on_train_only(True)
        auditor.mark_no_centered_windows(True)
        # All other checks remain False (violations in fail-closed mode)
        auditor.finalize_split()
        
        report = auditor.generate_report()
        
        self.assertFalse(report.all_splits_pass)
        # Should have 8 violations (10 total checks - 2 passed)
        self.assertEqual(len(report.violations), 8)


class TestCreateLeakageAuditReport(unittest.TestCase):
    """Tests for convenience function."""
    
    def test_create_report(self):
        """Test creating report with convenience function - all checks must be explicit."""
        splits_data = [
            {
                'split_id': '4->5',
                'train_days': [4],
                'val_days': [5],
                'scaler_fit_on_train_only': True,
                'imputer_fit_on_train_only': True,
                'feature_stats_fit_on_train_only': True,
                'calibrator_fit_on_train_only': True,
                'temporal_window_causal_only': True,
                'no_centered_windows': True,
                'no_future_derived_features': True,
                'no_target_stats_from_val': True,
                'no_decoder_threshold_tuning_on_val': True,
                'calibration_uses_train_slice_only': True,
            },
            {
                'split_id': '4+5->6',
                'train_days': [4, 5],
                'val_days': [6],
                'scaler_fit_on_train_only': True,
                'imputer_fit_on_train_only': True,
                'feature_stats_fit_on_train_only': True,
                'calibrator_fit_on_train_only': True,
                'temporal_window_causal_only': True,
                'no_centered_windows': True,
                'no_future_derived_features': True,
                'no_target_stats_from_val': True,
                'no_decoder_threshold_tuning_on_val': True,
                'calibration_uses_train_slice_only': True,
            },
        ]
        
        report = create_leakage_audit_report(
            run_id="run_001",
            elder_id="HK0011",
            seed=11,
            splits_data=splits_data,
            git_sha="abc123",
        )
        
        self.assertEqual(report.run_id, "run_001")
        self.assertEqual(len(report.splits), 2)
        self.assertTrue(report.all_splits_pass)
    
    def test_create_report_with_save(self):
        """Test creating and saving report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.json"
            
            splits_data = [
                {
                    'split_id': '4->5',
                    'train_days': [4],
                    'val_days': [5],
                },
            ]
            
            report = create_leakage_audit_report(
                run_id="run_001",
                elder_id="HK0011",
                seed=11,
                splits_data=splits_data,
                output_path=path,
            )
            
            # Verify file was created
            self.assertTrue(path.exists())
            
            # Verify content
            data = json.loads(path.read_text())
            self.assertEqual(data['run_id'], "run_001")


    def test_generate_report_fails_closed_with_zero_splits(self):
        """Regression test: Empty splits should fail closed, not pass by default."""
        auditor = LeakageAuditor("run_001", "HK0011", 11, git_sha="abc123")
        # Don't add any splits
        
        report = auditor.generate_report()
        
        # Should fail closed when no splits audited
        self.assertFalse(report.all_splits_pass)
        self.assertTrue(any("No splits audited" in v for v in report.violations))


if __name__ == '__main__':
    unittest.main()
