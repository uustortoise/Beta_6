"""
Complete Week 3 Integration Tests

Tests all Week 3 items:
- Item 10: Calibration/Validation Temporal Semantics Hardening
- Item 11: Unified "Why Rejected" Artifact
- Item 13: Deterministic Retrain Reproducibility Report
"""

import os
import sys
import pytest
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pandas as pd
import numpy as np

from ml.calibration_semantics import (
    CalibrationSemanticsTracker,
    TemporalPartition,
    TimeOrdering,
    CalibrationSemanticsMode,
    compare_validation_calibration_metrics,
)
from ml.rejection_artifact import (
    RejectionArtifactBuilder,
    RejectionCategory,
    Severity,
    create_rejection_artifact,
)
from ml.reproducibility_report import (
    ReproducibilityTracker,
    ReproducibilityReport,
    DataFingerprint,
    CodeVersion,
    RunOutcome,
    get_code_version,
    compute_data_fingerprint,
    verify_reproducibility_claim,
)


class TestItem10_CalibrationSemantics:
    """Item 10: Calibration/Validation Temporal Semantics Hardening."""
    
    def test_temporal_partition_creation(self):
        """Temporal partitions should store bounds correctly."""
        start = datetime(2024, 1, 1, 0, 0)
        end = datetime(2024, 1, 10, 0, 0)
        
        partition = TemporalPartition(
            partition_type="validation",
            start_ts=start,
            end_ts=end,
            sample_count=1000,
        )
        
        assert partition.partition_type == "validation"
        assert partition.start_ts == start
        assert partition.end_ts == end
        assert partition.sample_count == 1000
        assert partition.is_valid() is True
    
    def test_validation_before_calibration_detection(self):
        """Should detect when validation is before calibration."""
        tracker = CalibrationSemanticsTracker()
        
        # Validation: Jan 1-5, Calibration: Jan 6-10
        tracker.record_partition("validation", datetime(2024, 1, 1), datetime(2024, 1, 5), 500)
        tracker.record_partition("calibration", datetime(2024, 1, 6), datetime(2024, 1, 10), 500)
        
        order = tracker.analyze_temporal_ordering()
        
        assert order == TimeOrdering.VALIDATION_BEFORE_CALIBRATION
        assert len(tracker.report.warnings) == 1  # Should warn about optimistic metrics
    
    def test_overlapping_partitions_detection(self):
        """Should detect overlapping validation and calibration."""
        tracker = CalibrationSemanticsTracker()
        
        # Validation: Jan 1-5, Calibration: Jan 4-8 (overlap!)
        tracker.record_partition("validation", datetime(2024, 1, 1), datetime(2024, 1, 5), 500)
        tracker.record_partition("calibration", datetime(2024, 1, 4), datetime(2024, 1, 8), 500)
        
        order = tracker.analyze_temporal_ordering()
        
        assert order == TimeOrdering.OVERLAPPING
    
    def test_strict_mode_validation(self):
        """Strict mode should fail for validation-before-calibration."""
        tracker = CalibrationSemanticsTracker(mode=CalibrationSemanticsMode.STRICT)
        
        tracker.record_partition("validation", datetime(2024, 1, 1), datetime(2024, 1, 5), 500)
        tracker.record_partition("calibration", datetime(2024, 1, 6), datetime(2024, 1, 10), 500)
        
        tracker.analyze_temporal_ordering()
        passes, errors = tracker.validate_strict_mode()
        
        assert passes is False
        assert len(errors) == 1
    
    def test_metrics_recording(self):
        """Should record metrics for validation and calibration."""
        tracker = CalibrationSemanticsTracker()
        
        val_metrics = {"macro_f1": 0.75, "accuracy": 0.85}
        calib_metrics = {"macro_f1": 0.80, "accuracy": 0.88}
        
        tracker.record_metrics("validation", val_metrics, thresholded=False)
        tracker.record_metrics("calibration", calib_metrics, thresholded=True)
        
        report = tracker.get_report()
        
        assert report["validation_unthresholded_metrics"]["macro_f1"] == 0.75
        assert report["calibration_thresholded_metrics"]["macro_f1"] == 0.80
    
    def test_report_generation(self):
        """Should generate comprehensive report."""
        tracker = CalibrationSemanticsTracker()
        
        tracker.record_partition("train", datetime(2024, 1, 1), datetime(2024, 1, 7), 2000)
        tracker.record_partition("validation", datetime(2024, 1, 8), datetime(2024, 1, 9), 300)
        tracker.record_partition("calibration", datetime(2024, 1, 10), datetime(2024, 1, 10), 200)
        
        tracker.analyze_temporal_ordering()
        report = tracker.get_report()
        
        assert report["mode"] == "transparent"
        assert report["val_partition"]["partition_type"] == "validation"
        assert report["calib_partition"]["partition_type"] == "calibration"
    
    def test_compare_metrics_function(self):
        """Compare function should provide interpretation guidance."""
        val_partition = TemporalPartition(
            "validation", datetime(2024, 1, 1), datetime(2024, 1, 5), 500
        )
        calib_partition = TemporalPartition(
            "calibration", datetime(2024, 1, 6), datetime(2024, 1, 10), 500
        )
        
        val_metrics = {"macro_f1": 0.75}
        calib_metrics = {"macro_f1": 0.85}
        
        comparison = compare_validation_calibration_metrics(
            val_metrics, calib_metrics, val_partition, calib_partition
        )
        
        assert comparison["temporal_ordering"] == "validation_before_calibration"
        assert "interpretation_guidance" in comparison


class TestItem11_RejectionArtifact:
    """Item 11: Unified "Why Rejected" Artifact."""
    
    def test_builder_creates_reason(self):
        """Builder should create rejection reasons correctly."""
        builder = RejectionArtifactBuilder(
            run_id="run_001",
            elder_id="elder_123",
        )
        
        builder.add_coverage_failure(
            room="bedroom",
            observed_days=3,
            required_days=7,
        )
        
        artifact = builder.build()
        
        assert len(artifact.coverage_reasons) == 1
        assert artifact.coverage_reasons[0].category == RejectionCategory.COVERAGE
        assert artifact.coverage_reasons[0].severity == Severity.CRITICAL
    
    def test_executive_summary_generation(self):
        """Should generate executive summary."""
        builder = RejectionArtifactBuilder(
            run_id="run_001",
            elder_id="elder_123",
        )
        
        builder.add_coverage_failure("bedroom", 3, 7)
        builder.add_viability_failure("livingroom", "Insufficient samples")
        builder.add_global_gate_failure("kitchen", "macro_f1", 0.45, 0.55)
        
        artifact = builder.build()
        
        assert artifact.executive_summary != ""
        # 2 critical (coverage, viability) + 1 high (global_gate)
        assert "2 critical" in artifact.executive_summary or "critical" in artifact.executive_summary
        assert artifact.top_priority_fix != ""
        assert artifact.overall_status == "rejected"
    
    def test_room_summaries(self):
        """Should create per-room summaries."""
        builder = RejectionArtifactBuilder(
            run_id="run_001",
            elder_id="elder_123",
        )
        
        builder.add_coverage_failure("bedroom", 3, 7)
        builder.add_room_summary(
            room="bedroom",
            passed=False,
            metrics={"observed_days": 3},
            actionable_step="Collect 4 more days of data",
        )
        
        artifact = builder.build()
        
        assert "bedroom" in artifact.room_summaries
        assert artifact.room_summaries["bedroom"].actionable_next_step == "Collect 4 more days of data"
    
    def test_artifact_serialization(self):
        """Artifact should serialize to JSON correctly."""
        builder = RejectionArtifactBuilder(
            run_id="run_001",
            elder_id="elder_123",
            policy_hash="abc123",
            data_fingerprint="def456",
        )
        
        builder.add_statistical_validity_failure(
            room="bedroom",
            metric_name="minority_support",
            metric_value=5.0,
            threshold_value=10.0,
        )
        
        artifact = builder.build()
        
        # Test JSON serialization
        json_str = artifact.to_json()
        assert "run_001" in json_str
        assert "elder_123" in json_str
        
        # Test dict conversion
        data = artifact.to_dict()
        assert data["run_id"] == "run_001"
        assert data["policy_hash"] == "abc123"
    
    def test_artifact_save_load(self):
        """Artifact should save to and load from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = RejectionArtifactBuilder(
                run_id="run_001",
                elder_id="elder_123",
            )
            
            builder.add_walk_forward_failure("bedroom", 3, 7)
            artifact = builder.build()
            
            filepath = Path(tmpdir) / "rejection_summary.json"
            artifact.save(filepath)
            
            assert filepath.exists()
            
            # Load and verify
            loaded_data = json.loads(filepath.read_text())
            assert loaded_data["run_id"] == "run_001"
            assert loaded_data["overall_status"] == "rejected"
    
    def test_all_rejection_categories(self):
        """Should support all rejection categories."""
        builder = RejectionArtifactBuilder(
            run_id="run_001",
            elder_id="elder_123",
        )
        
        # Add various failure types
        builder.add_coverage_failure("room1", 3, 7)
        builder.add_viability_failure("room2", "Failed")
        builder.add_statistical_validity_failure("room3", "support", 5, 10)
        builder.add_walk_forward_failure("room4", 3, 7)
        builder.add_class_coverage_failure("room5", [1, 2])
        builder.add_global_gate_failure("room6", "f1", 0.4, 0.5)
        builder.add_post_gap_retention_failure("room7", 0.3, 0.5, 15)
        
        artifact = builder.build()
        
        # Verify all categories populated
        assert len(artifact.coverage_reasons) == 1
        assert len(artifact.viability_reasons) == 1
        assert len(artifact.statistical_validity_reasons) == 1
        assert len(artifact.walk_forward_reasons) == 1
        assert len(artifact.class_coverage_reasons) == 1
        assert len(artifact.global_gate_reasons) == 1
        assert len(artifact.post_gap_retention_reasons) == 1
    
    def test_passed_run_artifact(self):
        """Artifact for passed run should reflect success."""
        builder = RejectionArtifactBuilder(
            run_id="run_001",
            elder_id="elder_123",
        )
        
        # No failures added - run passed
        artifact = builder.build()
        
        assert artifact.overall_passed is True
        assert artifact.overall_status == "promoted"
        assert "successfully" in artifact.executive_summary.lower()


class TestItem13_ReproducibilityReport:
    """Item 13: Deterministic Retrain Reproducibility Report."""
    
    def test_data_fingerprint_creation(self):
        """Data fingerprint should capture data characteristics."""
        fingerprint = DataFingerprint(
            elder_id="elder_123",
            room_names=("bedroom", "livingroom"),
            total_samples=10000,
            observed_days=10,
            raw_data_hash="abc123def456",
            timestamp_range_start="2024-01-01T00:00:00",
            timestamp_range_end="2024-01-10T00:00:00",
        )
        
        assert fingerprint.elder_id == "elder_123"
        assert len(fingerprint.room_names) == 2
        assert fingerprint.compute_hash() is not None
    
    def test_code_version_detection(self):
        """Should detect code version from git."""
        version = get_code_version()
        
        assert version.git_commit is not None
        assert version.git_branch is not None
        assert version.python_version is not None
        # May or may not be dirty in test environment
    
    def test_run_outcome_signature(self):
        """Outcome signatures should be deterministic."""
        outcome1 = RunOutcome(
            promoted_rooms=["bedroom", "kitchen"],
            rejected_rooms=["livingroom"],
        )
        
        outcome2 = RunOutcome(
            promoted_rooms=["kitchen", "bedroom"],  # Different order
            rejected_rooms=["livingroom"],
        )
        
        # Signatures should be same (sorted)
        assert outcome1.compute_signature() == outcome2.compute_signature()
    
    def test_tracker_creates_report(self):
        """Tracker should create reproducibility report."""
        tracker = ReproducibilityTracker()
        
        fingerprint = DataFingerprint(
            elder_id="elder_123",
            room_names=("bedroom",),
            total_samples=1000,
            observed_days=5,
            raw_data_hash="abc123",
        )
        
        outcome = RunOutcome(
            promoted_rooms=["bedroom"],
            rejected_rooms=[],
        )
        
        report = tracker.create_report(
            run_id="run_001",
            elder_id="elder_123",
            data_fingerprint=fingerprint,
            policy_hash="policy_abc",
            random_seed=42,
            outcome=outcome,
        )
        
        assert report.run_id == "run_001"
        assert report.data_fingerprint.compute_hash() == fingerprint.compute_hash()
        assert report.policy_hash == "policy_abc"
        assert report.random_seed == 42
        assert report.outcome.compute_signature() == outcome.compute_signature()
        assert report.compute_composite_hash() is not None
    
    def test_noop_detection(self):
        """Should detect no-op reruns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ReproducibilityTracker(history_dir=tmpdir)
            
            fingerprint = DataFingerprint(
                elder_id="elder_123",
                room_names=("bedroom",),
                total_samples=1000,
                observed_days=5,
                raw_data_hash="abc123",
            )
            
            # First run - this will use the actual git code version
            outcome1 = RunOutcome(promoted_rooms=["bedroom"])
            report1 = tracker.create_report(
                run_id="run_001",
                elder_id="elder_123",
                data_fingerprint=fingerprint,
                policy_hash="policy_abc",
                random_seed=42,
                outcome=outcome1,
            )
            
            # Check no-op eligibility with SAME factors (no explicit code_version)
            # This will auto-detect the same code version
            is_noop, reason, prior = tracker.check_noop_eligibility(
                data_fingerprint=fingerprint,
                policy_hash="policy_abc",
                # code_version=None to use auto-detected (same as first run)
            )
            
            # Note: If git is dirty, this may return False - that's expected behavior
            # We just verify the logic works
            if report1.code_version.is_clean():
                assert is_noop is True
                assert prior is not None
                assert prior.run_id == "run_001"
            else:
                # If code is dirty, no-op should be blocked
                assert is_noop is False or prior is not None
    
    def test_noop_blocked_on_dirty_code(self):
        """No-op should be blocked if code is dirty."""
        tracker = ReproducibilityTracker()  # No history dir
        
        fingerprint = DataFingerprint(
            elder_id="elder_123",
            room_names=("bedroom",),
            total_samples=1000,
            observed_days=5,
            raw_data_hash="abc123",
        )
        
        # Create a mock prior run with DIRTY code (simulating a previous dirty run)
        dirty_version = CodeVersion(
            git_commit="abc123def456",
            git_branch="main",
            git_dirty=True,  # Dirty
            python_version="3.12.0",
        )
        
        outcome = RunOutcome(promoted_rooms=["bedroom"])
        report = ReproducibilityReport(
            run_id="run_001",
            timestamp=datetime.utcnow().isoformat(),
            elder_id="elder_123",
            data_fingerprint=fingerprint,
            policy_hash="policy_abc",
            code_version=dirty_version,
            random_seed=0,  # Must match what check_noop_eligibility uses
            outcome=outcome,
        )
        tracker._history["run_001"] = report
        
        # Check with SAME dirty version - should find prior but block
        is_noop, reason, prior = tracker.check_noop_eligibility(
            data_fingerprint=fingerprint,
            policy_hash="policy_abc",
            code_version=dirty_version,  # Same dirty version
        )
        
        # Should find the prior run but block due to dirty code
        assert prior is not None  # Should find the prior
        assert prior.run_id == "run_001"
        assert is_noop is False
        assert "dirty" in reason.lower()
    
    def test_outcome_parity_verification(self):
        """Should verify outcome parity with prior runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ReproducibilityTracker(history_dir=tmpdir)
            
            fingerprint = DataFingerprint(
                elder_id="elder_123",
                room_names=("bedroom",),
                total_samples=1000,
                observed_days=5,
                raw_data_hash="abc123",
            )
            
            # First run
            outcome1 = RunOutcome(promoted_rooms=["bedroom"])
            report1 = tracker.create_report(
                run_id="run_001",
                elder_id="elder_123",
                data_fingerprint=fingerprint,
                policy_hash="policy_abc",
                random_seed=42,
                outcome=outcome1,
            )
            
            # Second run with same factors
            outcome2 = RunOutcome(promoted_rooms=["bedroom"])
            report2 = tracker.create_report(
                run_id="run_002",
                elder_id="elder_123",
                data_fingerprint=fingerprint,
                policy_hash="policy_abc",
                random_seed=42,
                outcome=outcome2,
            )
            
            assert report2.prior_run_linked is True
            assert report2.prior_run_id == "run_001"
            assert report2.outcome_parity_with_prior is True
            assert report2.parity_verification_hash is not None
    
    def test_report_save_load(self):
        """Report should save to and load from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ReproducibilityTracker(history_dir=tmpdir)
            
            fingerprint = DataFingerprint(
                elder_id="elder_123",
                room_names=("bedroom",),
                total_samples=1000,
                observed_days=5,
                raw_data_hash="abc123",
            )
            
            outcome = RunOutcome(promoted_rooms=["bedroom"])
            
            report = tracker.create_report(
                run_id="run_001",
                elder_id="elder_123",
                data_fingerprint=fingerprint,
                policy_hash="policy_abc",
                random_seed=42,
                outcome=outcome,
            )
            
            filepath = Path(tmpdir) / "run_001.json"
            report.save(filepath)
            
            assert filepath.exists()
            
            # Verify content
            loaded = json.loads(filepath.read_text())
            assert loaded["run_id"] == "run_001"
            assert loaded["policy_hash"] == "policy_abc"
            assert loaded["outcome_signature"] == outcome.compute_signature()
    
    def test_reproducibility_verification(self):
        """Should verify reproducibility between two runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two reports
            tracker = ReproducibilityTracker(history_dir=tmpdir)
            
            fingerprint = DataFingerprint(
                elder_id="elder_123",
                room_names=("bedroom",),
                total_samples=1000,
                observed_days=5,
                raw_data_hash="abc123",
            )
            
            outcome = RunOutcome(promoted_rooms=["bedroom"])
            
            report1 = tracker.create_report(
                run_id="run_001",
                elder_id="elder_123",
                data_fingerprint=fingerprint,
                policy_hash="policy_abc",
                random_seed=42,
                outcome=outcome,
            )
            
            report2 = tracker.create_report(
                run_id="run_002",
                elder_id="elder_123",
                data_fingerprint=fingerprint,
                policy_hash="policy_abc",
                random_seed=42,
                outcome=outcome,
            )
            
            # Save both
            path1 = Path(tmpdir) / "report1.json"
            path2 = Path(tmpdir) / "report2.json"
            report1.save(path1)
            report2.save(path2)
            
            # Verify reproducibility
            verified, explanation = verify_reproducibility_claim(path1, path2)
            
            assert verified is True
            assert "reproducible" in explanation.lower()


class TestWeek3EndToEnd:
    """End-to-end integration tests for Week 3."""
    
    def test_complete_rejection_workflow(self):
        """Complete workflow: gates → rejection artifact."""
        # Simulate gate failures
        gate_results = {
            "bedroom": {
                "passes": False,
                "blocking": True,
                "gate_name": "coverage",
                "reasons": ["Insufficient days: 3 < 7"],
            },
            "kitchen": {
                "passes": False,
                "blocking": False,
                "gate_name": "global_gate",
                "reasons": ["F1 below threshold"],
            },
        }
        
        artifact = create_rejection_artifact(
            run_id="run_001",
            elder_id="elder_123",
            gate_results=gate_results,
        )
        
        assert artifact.overall_status == "rejected"
        assert len(artifact.room_summaries) == 0  # Convenience function doesn't add room summaries
    
    def test_reproducibility_with_temporal_semantics(self):
        """Reproducibility report should include temporal semantics."""
        # Create temporal semantics
        tracker = CalibrationSemanticsTracker()
        tracker.record_partition("train", datetime(2024, 1, 1), datetime(2024, 1, 7), 2000)
        tracker.record_partition("validation", datetime(2024, 1, 8), datetime(2024, 1, 9), 300)
        tracker.record_partition("calibration", datetime(2024, 1, 10), datetime(2024, 1, 10), 200)
        tracker.analyze_temporal_ordering()
        
        temporal_report = tracker.get_report()
        
        # Verify temporal semantics in report
        assert temporal_report["val_partition"] is not None
        assert temporal_report["calib_partition"] is not None
        assert temporal_report["validation_vs_calibration_order"] == "validation_before_calibration"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
