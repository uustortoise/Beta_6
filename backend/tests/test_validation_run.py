"""
Tests for PR-5: Controlled Validation Run + Signoff Pack
"""

import json
import os
import sys
import unittest
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.validation_run import (
    ValidationRun,
    ValidationRunStatus,
    ValidationRunManager,
    DailyRunResult,
    SignoffPack,
    SignoffDecision,
    StrictEvaluationConfig,
)


def _utc_now_iso() -> str:
    """Return timezone-aware UTC timestamp as ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


class TestValidationRun(unittest.TestCase):
    """Tests for ValidationRun dataclass."""
    
    def test_run_creation(self):
        """Test creating a validation run."""
        run = ValidationRun(
            run_id="test_run_001",
            elder_id="HK001",
            start_date=_utc_now_iso(),
            duration_days=7,
            manifest_path="/path/to/manifest.json",
            manifest_hash="abc123",
            status=ValidationRunStatus.PENDING,
        )
        
        self.assertEqual(run.run_id, "test_run_001")
        self.assertEqual(run.elder_id, "HK001")
        self.assertEqual(run.duration_days, 7)
        self.assertEqual(run.status, ValidationRunStatus.PENDING)
    
    def test_run_serialization(self):
        """Test run serialization to dict."""
        run = ValidationRun(
            run_id="test_run_001",
            elder_id="HK001",
            start_date="2026-02-01T00:00:00",
            duration_days=7,
            manifest_path="/path/to/manifest.json",
            manifest_hash="abc123",
            status=ValidationRunStatus.COMPLETED,
        )
        
        data = run.to_dict()
        
        self.assertEqual(data["run_id"], "test_run_001")
        self.assertEqual(data["elder_id"], "HK001")
        self.assertEqual(data["status"], "completed")
    
    def test_determinism_score_perfect(self):
        """Test determinism score when all days match."""
        run = ValidationRun(
            run_id="test_run",
            elder_id="HK001",
            start_date="2026-02-01T00:00:00",
            duration_days=7,
            manifest_path="/path/manifest.json",
            manifest_hash="abc123",
            status=ValidationRunStatus.COMPLETED,
        )
        
        # Add 3 days with identical promotions
        for i in range(3):
            run.daily_results.append(DailyRunResult(
                date=f"2026-02-0{i+1}",
                run_timestamp=_utc_now_iso(),
                manifest_hash="abc123",
                rooms_trained=["bedroom", "kitchen"],
                rooms_promoted=["bedroom"],
                rooms_rejected=["kitchen"],
                decision_traces={"bedroom": "/path/trace.json"},
                rejection_artifacts={"kitchen": "/path/reject.json"},
                gate_stack_summary={},
            ))
        
        score, issues = run.compute_determinism_score()
        
        self.assertEqual(score, 1.0)
        self.assertEqual(len(issues), 0)
    
    def test_determinism_score_imperfect(self):
        """Test determinism score when days don't match."""
        run = ValidationRun(
            run_id="test_run",
            elder_id="HK001",
            start_date="2026-02-01T00:00:00",
            duration_days=7,
            manifest_path="/path/manifest.json",
            manifest_hash="abc123",
            status=ValidationRunStatus.COMPLETED,
        )
        
        # Day 1: bedroom promoted
        run.daily_results.append(DailyRunResult(
            date="2026-02-01",
            run_timestamp=_utc_now_iso(),
            manifest_hash="abc123",
            rooms_trained=["bedroom", "kitchen"],
            rooms_promoted=["bedroom"],
            rooms_rejected=["kitchen"],
            decision_traces={},
            rejection_artifacts={},
            gate_stack_summary={},
        ))
        
        # Day 2: different promotions
        run.daily_results.append(DailyRunResult(
            date="2026-02-02",
            run_timestamp=_utc_now_iso(),
            manifest_hash="abc123",
            rooms_trained=["bedroom", "kitchen"],
            rooms_promoted=["bedroom", "kitchen"],  # Different!
            rooms_rejected=[],
            decision_traces={},
            rejection_artifacts={},
            gate_stack_summary={},
        ))
        
        score, issues = run.compute_determinism_score()
        
        self.assertEqual(score, 0.0)
        self.assertGreater(len(issues), 0)
    
    def test_determinism_score_single_day(self):
        """Test determinism score with only one day."""
        run = ValidationRun(
            run_id="test_run",
            elder_id="HK001",
            start_date="2026-02-01T00:00:00",
            duration_days=7,
            manifest_path="/path/manifest.json",
            manifest_hash="abc123",
            status=ValidationRunStatus.COMPLETED,
        )
        
        run.daily_results.append(DailyRunResult(
            date="2026-02-01",
            run_timestamp=_utc_now_iso(),
            manifest_hash="abc123",
            rooms_trained=["bedroom"],
            rooms_promoted=["bedroom"],
            rooms_rejected=[],
            decision_traces={},
            rejection_artifacts={},
            gate_stack_summary={},
        ))
        
        score, issues = run.compute_determinism_score()
        
        self.assertEqual(score, 1.0)
        self.assertEqual(len(issues), 0)


class TestDailyRunResult(unittest.TestCase):
    """Tests for DailyRunResult dataclass."""
    
    def test_daily_result_creation(self):
        """Test creating a daily result."""
        result = DailyRunResult(
            date="2026-02-01",
            run_timestamp=_utc_now_iso(),
            manifest_hash="abc123",
            rooms_trained=["bedroom", "kitchen"],
            rooms_promoted=["bedroom"],
            rooms_rejected=["kitchen"],
            decision_traces={"bedroom": "/path/trace.json"},
            rejection_artifacts={"kitchen": "/path/reject.json"},
            gate_stack_summary={"gates_passed": 3},
        )
        
        self.assertEqual(result.date, "2026-02-01")
        self.assertEqual(result.rooms_trained, ["bedroom", "kitchen"])
        self.assertEqual(result.rooms_promoted, ["bedroom"])


class TestValidationRunManager(unittest.TestCase):
    """Tests for ValidationRunManager."""
    
    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ValidationRunManager(runs_dir=Path(self.temp_dir))
    
    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_run(self):
        """Test creating a validation run."""
        # Create a dummy manifest
        manifest_path = Path(self.temp_dir) / "manifest.json"
        manifest_path.write_text(json.dumps({
            "elder_id": "HK001",
            "data_files": [],
        }))
        
        run = self.manager.create_run(
            elder_id="HK001",
            duration_days=7,
            manifest_path=str(manifest_path),
        )
        
        self.assertEqual(run.elder_id, "HK001")
        self.assertEqual(run.duration_days, 7)
        self.assertEqual(run.status, ValidationRunStatus.PENDING)
        self.assertTrue(run.manifest_hash)
        
        # Verify file was created
        run_file = Path(self.temp_dir) / f"{run.run_id}.json"
        self.assertTrue(run_file.exists())
    
    def test_load_run(self):
        """Test loading a saved run."""
        # Create a dummy manifest
        manifest_path = Path(self.temp_dir) / "manifest.json"
        manifest_path.write_text(json.dumps({
            "elder_id": "HK001",
            "data_files": [],
        }))
        
        run = self.manager.create_run(
            elder_id="HK001",
            duration_days=7,
            manifest_path=str(manifest_path),
        )
        
        # Load it back
        loaded = self.manager._load_run(run.run_id)
        
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id, run.run_id)
        self.assertEqual(loaded.elder_id, run.elder_id)
    
    def test_load_nonexistent_run(self):
        """Test loading a run that doesn't exist."""
        loaded = self.manager._load_run("nonexistent_run")
        self.assertIsNone(loaded)
    
    def test_record_daily_result(self):
        """Test recording a daily result."""
        # Create a dummy manifest
        manifest_path = Path(self.temp_dir) / "manifest.json"
        manifest_path.write_text(json.dumps({
            "elder_id": "HK001",
            "data_files": [],
        }))
        
        run = self.manager.create_run(
            elder_id="HK001",
            duration_days=7,
            manifest_path=str(manifest_path),
        )
        
        result = DailyRunResult(
            date="2026-02-01",
            run_timestamp=_utc_now_iso(),
            manifest_hash=run.manifest_hash,
            rooms_trained=["bedroom"],
            rooms_promoted=["bedroom"],
            rooms_rejected=[],
            decision_traces={},
            rejection_artifacts={},
            gate_stack_summary={},
        )
        
        self.manager.record_daily_result(run.run_id, result)
        
        # Verify it was recorded
        loaded = self.manager._load_run(run.run_id)
        self.assertEqual(len(loaded.daily_results), 1)
        self.assertEqual(loaded.status, ValidationRunStatus.RUNNING)
    
    def test_build_compliance_checklist_pass(self):
        """Test compliance checklist when all checks pass."""
        run = ValidationRun(
            run_id="test_run",
            elder_id="HK001",
            start_date="2026-02-01T00:00:00",
            duration_days=7,
            manifest_path="/path/manifest.json",
            manifest_hash="abc123",
            status=ValidationRunStatus.COMPLETED,
        )
        
        # Add 7 days with consistent data
        for i in range(7):
            run.daily_results.append(DailyRunResult(
                date=f"2026-02-0{i+1}",
                run_timestamp=_utc_now_iso(),
                manifest_hash="abc123",
                rooms_trained=["bedroom"],
                rooms_promoted=["bedroom"],
                rooms_rejected=[],
                decision_traces={"bedroom": "/path/trace.json"},
                rejection_artifacts={},
                gate_stack_summary={},
            ))
        
        checklist = self.manager._build_compliance_checklist(run)
        
        self.assertTrue(checklist["minimum_7_days"])
        self.assertTrue(checklist["determinism_score_above_80"])
        self.assertTrue(checklist["all_rooms_have_decision_traces"])
        self.assertTrue(checklist["no_missing_gate_reasons"])
        self.assertTrue(checklist["manifest_hash_consistent"])
    
    def test_build_compliance_checklist_fail_7_days(self):
        """Test compliance checklist with less than 7 days."""
        run = ValidationRun(
            run_id="test_run",
            elder_id="HK001",
            start_date="2026-02-01T00:00:00",
            duration_days=7,
            manifest_path="/path/manifest.json",
            manifest_hash="abc123",
            status=ValidationRunStatus.COMPLETED,
        )
        
        # Add only 5 days
        for i in range(5):
            run.daily_results.append(DailyRunResult(
                date=f"2026-02-0{i+1}",
                run_timestamp=_utc_now_iso(),
                manifest_hash="abc123",
                rooms_trained=["bedroom"],
                rooms_promoted=["bedroom"],
                rooms_rejected=[],
                decision_traces={"bedroom": "/path/trace.json"},
                rejection_artifacts={},
                gate_stack_summary={},
            ))
        
        checklist = self.manager._build_compliance_checklist(run)
        
        self.assertFalse(checklist["minimum_7_days"])

    def test_finalize_run_generates_ws6_signoff_artifacts(self):
        """Finalize run writes both legacy and WS-6 signoff artifacts."""
        manifest_path = Path(self.temp_dir) / "manifest.json"
        manifest_path.write_text(json.dumps({
            "elder_id": "HK001",
            "data_files": [],
        }))

        run = self.manager.create_run(
            elder_id="HK001",
            duration_days=7,
            manifest_path=str(manifest_path),
            config={
                "baseline_version": "v31",
                "baseline_artifact_hash": "sha256:baseline123",
                "git_sha": "abc123",
            },
        )

        # Lane D1: Generate 12 split-seed cells (4 splits × 3 seeds)
        splits = [
            ([4], 5),
            ([4, 5], 6),
            ([4, 5, 6], 7),
            ([4, 5, 6, 7], 8),
        ]
        seeds = [11, 22, 33]
        
        idx = 0
        for train_days, val_day in splits:
            split_id = f"{','.join(map(str, train_days))}->{val_day}"
            for seed in seeds:
                trace_path = Path(self.temp_dir) / f"trace_{idx}.json"
                trace_path.write_text(json.dumps({"room": "bedroom", "split": split_id, "seed": seed}))
                result = DailyRunResult(
                    date=f"2026-02-{idx+1:02d}",
                    run_timestamp=_utc_now_iso(),
                    manifest_hash=run.manifest_hash,
                    rooms_trained=["bedroom"],
                    rooms_promoted=["bedroom"],
                    rooms_rejected=[],
                    decision_traces={"bedroom": str(trace_path)},
                    rejection_artifacts={},
                    gate_stack_summary={
                        "seed": seed,
                        "split_id": split_id,
                        "train_days": train_days,
                        "val_days": [val_day],
                        "hard_gates_passed": 1,
                        "hard_gates_total": 1,
                        "timeline_gates_passed": 1,
                        "timeline_gates_total": 1,
                        "all_hard_gates_pass": True,
                        "leakage_audit_pass": True,
                        "leakage_audit": {
                            "pass": True,
                            "reasons": [],
                        },
                        "timeline_metrics": {
                            "bedroom": {
                                "segment_start_mae_minutes": 4.0,
                                "segment_duration_mae_minutes": 18.0,
                                "fragmentation_rate": 0.2,
                            }
                        },
                    },
                )
                self.manager.record_daily_result(run.run_id, result)
                idx += 1

        pack = self.manager.finalize_run(run.run_id)
        self.assertEqual(pack.run_id, run.run_id)

        output_dir = Path(self.temp_dir) / run.run_id
        self.assertTrue((output_dir / "signoff_pack.json").exists())
        self.assertTrue((output_dir / "signoff_pack_ws6.json").exists())
        self.assertTrue((output_dir / "signoff_report_ws6.md").exists())

        ws6_data = json.loads((output_dir / "signoff_pack_ws6.json").read_text())
        self.assertEqual(ws6_data["baseline_version"], "v31")
        self.assertEqual(ws6_data["baseline_artifact_hash"], "sha256:baseline123")
        # Lane D1: Should have 12 split-seed results
        self.assertEqual(len(ws6_data["split_seed_results"]), 12)

    def test_finalize_run_ws6_handles_invalid_seed_without_crashing(self):
        """WS-6 export should fail safely when a split-seed row has an invalid seed."""
        manifest_path = Path(self.temp_dir) / "manifest.json"
        manifest_path.write_text(json.dumps({
            "elder_id": "HK001",
            "data_files": [],
        }))

        run = self.manager.create_run(
            elder_id="HK001",
            duration_days=7,
            manifest_path=str(manifest_path),
            config={"baseline_version": "v31"},
        )

        splits = [
            ([4], 5),
            ([4, 5], 6),
            ([4, 5, 6], 7),
            ([4, 5, 6, 7], 8),
        ]
        seeds = [11, 22, 33]

        idx = 0
        for train_days, val_day in splits:
            split_id = f"{','.join(map(str, train_days))}->{val_day}"
            for seed in seeds:
                trace_path = Path(self.temp_dir) / f"trace_invalid_seed_{idx}.json"
                trace_path.write_text(json.dumps({"room": "bedroom", "split": split_id, "seed": seed}))
                seed_value = "bad_seed" if (split_id == "4,5,6,7->8" and seed == 33) else seed
                result = DailyRunResult(
                    date=f"2026-02-{idx+1:02d}",
                    run_timestamp=_utc_now_iso(),
                    manifest_hash=run.manifest_hash,
                    rooms_trained=["bedroom"],
                    rooms_promoted=["bedroom"],
                    rooms_rejected=[],
                    decision_traces={"bedroom": str(trace_path)},
                    rejection_artifacts={},
                    gate_stack_summary={
                        "seed": seed_value,
                        "split_id": split_id,
                        "train_days": train_days,
                        "val_days": [val_day],
                        "hard_gates_passed": 1,
                        "hard_gates_total": 1,
                        "timeline_gates_passed": 1,
                        "timeline_gates_total": 1,
                        "all_hard_gates_pass": True,
                        "leakage_audit_pass": True,
                        "leakage_audit": {"pass": True, "reasons": []},
                    },
                )
                self.manager.record_daily_result(run.run_id, result)
                idx += 1

        # Force finalization: invalid seed should not crash WS-6 export.
        pack = self.manager.finalize_run(run.run_id, force=True)
        self.assertEqual(pack.signoff_decision, SignoffDecision.FAIL)

        output_dir = Path(self.temp_dir) / run.run_id
        ws6_path = output_dir / "signoff_pack_ws6.json"
        self.assertTrue(ws6_path.exists())
        ws6_data = json.loads(ws6_path.read_text())
        self.assertEqual(len(ws6_data["split_seed_results"]), 12)

        missing_cell_rows = [
            row for row in ws6_data["split_seed_results"]
            if "missing_split_seed_cell" in row.get("leakage_violations", [])
        ]
        self.assertEqual(len(missing_cell_rows), 1)


class TestSignoffPack(unittest.TestCase):
    """Tests for SignoffPack."""
    
    def test_signoff_pack_creation(self):
        """Test creating a signoff pack."""
        run = ValidationRun(
            run_id="test_run",
            elder_id="HK001",
            start_date="2026-02-01T00:00:00",
            duration_days=7,
            manifest_path="/path/manifest.json",
            manifest_hash="abc123",
            status=ValidationRunStatus.COMPLETED,
        )
        
        pack = SignoffPack(
            run_id="test_run",
            generated_at=_utc_now_iso(),
            validation_run=run,
            all_decision_traces=[],
            all_rejection_artifacts=[],
            gate_reason_summary={},
            determinism_score=1.0,
            determinism_issues=[],
            compliance_checklist={"test": True},
        )
        
        self.assertEqual(pack.run_id, "test_run")
        self.assertEqual(pack.determinism_score, 1.0)
    
    def test_signoff_pack_markdown_report(self):
        """Test markdown report generation."""
        run = ValidationRun(
            run_id="test_run",
            elder_id="HK001",
            start_date="2026-02-01T00:00:00",
            duration_days=7,
            manifest_path="/path/manifest.json",
            manifest_hash="abc123",
            status=ValidationRunStatus.COMPLETED,
        )
        
        pack = SignoffPack(
            run_id="test_run",
            generated_at="2026-02-08T00:00:00",
            validation_run=run,
            all_decision_traces=[],
            all_rejection_artifacts=[],
            gate_reason_summary={"coverage": 3, "validity": 1},
            determinism_score=0.95,
            determinism_issues=["Day 2 had extra promotion"],
            compliance_checklist={
                "minimum_7_days": True,
                "determinism_score_above_80": True,
            },
        )
        
        report = pack.generate_markdown_report()
        
        self.assertIn("Validation Run Signoff Pack", report)
        self.assertIn("test_run", report)
        self.assertIn("95.0%", report)  # Determinism score
        self.assertIn("minimum_7_days", report)
        self.assertIn("Day 2 had extra promotion", report)
    
    def test_signoff_pack_no_issues(self):
        """Test markdown report with no issues."""
        run = ValidationRun(
            run_id="test_run",
            elder_id="HK001",
            start_date="2026-02-01T00:00:00",
            duration_days=7,
            manifest_path="/path/manifest.json",
            manifest_hash="abc123",
            status=ValidationRunStatus.COMPLETED,
        )
        
        pack = SignoffPack(
            run_id="test_run",
            generated_at="2026-02-08T00:00:00",
            validation_run=run,
            all_decision_traces=[],
            all_rejection_artifacts=[],
            gate_reason_summary={},
            determinism_score=1.0,
            determinism_issues=[],
            compliance_checklist={"test": True},
        )
        
        report = pack.generate_markdown_report()
        
        self.assertIn("No determinism issues detected", report)


class TestStrictEvaluationConfig(unittest.TestCase):
    """Tests for Lane D1 StrictEvaluationConfig."""
    
    def test_default_splits_and_seeds(self):
        """Test default rolling splits and seeds."""
        config = StrictEvaluationConfig()
        
        # Should have 4 splits
        self.assertEqual(len(config.splits), 4)
        
        # Should have 3 seeds
        self.assertEqual(len(config.seeds), 3)
        self.assertEqual(config.seeds, [11, 22, 33])
        
        # Matrix should be 4 * 3 = 12
        matrix = config.get_split_seed_matrix()
        self.assertEqual(len(matrix), 12)
    
    def test_custom_splits_and_seeds(self):
        """Test custom configuration."""
        config = StrictEvaluationConfig(
            splits=[([1], 2), ([1, 2], 3)],
            seeds=[42],
        )
        matrix = config.get_split_seed_matrix()
        self.assertEqual(len(matrix), 2)
        self.assertEqual(matrix[0], ("1->2", 42))
    
    def test_thresholds(self):
        """Test default promotion thresholds."""
        config = StrictEvaluationConfig()
        
        self.assertEqual(config.tier1_recall_floor, 0.50)
        self.assertEqual(config.tier2_recall_floor, 0.35)
        self.assertEqual(config.tier3_recall_floor, 0.20)
        self.assertEqual(config.home_empty_precision_floor, 0.95)
        self.assertEqual(config.false_empty_rate_ceiling, 0.05)
        self.assertEqual(config.timeline_gate_pass_rate, 0.80)


class TestSignoffDecision(unittest.TestCase):
    """Tests for Lane D1 signoff decision logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ValidationRunManager(runs_dir=Path(self.temp_dir))
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_split_seed_result(self, split_id, seed, manifest_hash, leakage_pass=True, hard_pass=True, temp_dir=None):
        """Helper to create split-seed result with proper gate_stack_summary.
        
        Args:
            temp_dir: If provided, creates actual trace files in this directory
        """
        # Create trace file if temp_dir provided
        if temp_dir:
            trace_dir = Path(temp_dir) / "traces"
            trace_dir.mkdir(exist_ok=True)
            trace_path = trace_dir / f"trace_{split_id.replace(',', '_').replace('->', '_')}_seed{seed}.json"
            trace_path.write_text(json.dumps({
                "split_id": split_id,
                "seed": seed,
                "decision": "promoted"
            }))
            trace_path_str = str(trace_path)
        else:
            trace_path_str = f"/tmp/trace_{split_id}_seed{seed}.json"
        
        return DailyRunResult(
            date=f"2026-02-{split_id.replace(',', '_').replace('->', '_')}_seed{seed}",
            run_timestamp=_utc_now_iso(),
            manifest_hash=manifest_hash,
            rooms_trained=["bedroom"],
            rooms_promoted=["bedroom"],
            rooms_rejected=[],
            decision_traces={"bedroom": trace_path_str},
            rejection_artifacts={},
            gate_stack_summary={
                "seed": seed,
                "split_id": split_id,
                "all_hard_gates_pass": hard_pass,
                "timeline_gates_passed": 1,
                "timeline_gates_total": 1,
                "leakage_audit_pass": leakage_pass,
                "leakage_audit": {
                    "pass": leakage_pass,
                    "reasons": [] if leakage_pass else ["leakage_detected"],
                },
            },
        )
    
    def _add_full_split_seed_matrix(self, run, leakage_pass=True, hard_pass=True, temp_dir=None):
        """Add all 12 split-seed cells (4 splits × 3 seeds)."""
        splits = [
            ([4], 5),
            ([4, 5], 6),
            ([4, 5, 6], 7),
            ([4, 5, 6, 7], 8),
        ]
        seeds = [11, 22, 33]
        
        for train_days, val_day in splits:
            split_id = f"{','.join(map(str, train_days))}->{val_day}"
            for seed in seeds:
                result = self._create_split_seed_result(
                    split_id=split_id,
                    seed=seed,
                    manifest_hash=run.manifest_hash,
                    leakage_pass=leakage_pass,
                    hard_pass=hard_pass,
                    temp_dir=temp_dir,
                )
                self.manager.record_daily_result(run.run_id, result)
    
    def test_signoff_pass_all_checks(self):
        """Test PASS decision when all checks pass."""
        manifest_path = Path(self.temp_dir) / "manifest.json"
        manifest_path.write_text(json.dumps({"elder_id": "HK001"}))
        
        run = self.manager.create_run(
            elder_id="HK001",
            duration_days=7,
            manifest_path=str(manifest_path),
        )
        
        # Add all 12 split-seed cells with all checks passing
        self._add_full_split_seed_matrix(run, leakage_pass=True, hard_pass=True, temp_dir=self.temp_dir)
        
        pack = self.manager.finalize_run(run.run_id)
        
        self.assertEqual(pack.signoff_decision, SignoffDecision.PASS)
        self.assertEqual(pack.recommended_stage, "external_ready")
        self.assertEqual(len(pack.blocking_issues), 0)

    def test_ws6_unavailable_blocks_and_does_not_write_signoff_pack(self):
        """Promotion mode must hard-fail without WS-6 and avoid partial signoff artifacts."""
        manifest_path = Path(self.temp_dir) / "manifest.json"
        manifest_path.write_text(json.dumps({"elder_id": "HK001"}))

        run = self.manager.create_run(
            elder_id="HK001",
            duration_days=7,
            manifest_path=str(manifest_path),
        )
        self._add_full_split_seed_matrix(run, leakage_pass=True, hard_pass=True, temp_dir=self.temp_dir)

        import ml.validation_run as validation_run_module

        old_generator = validation_run_module.Ws6SignoffPackGenerator
        old_split_result = validation_run_module.Ws6SplitSeedResult
        validation_run_module.Ws6SignoffPackGenerator = None
        validation_run_module.Ws6SplitSeedResult = None
        try:
            with self.assertRaises(RuntimeError):
                self.manager.finalize_run(run.run_id)
        finally:
            validation_run_module.Ws6SignoffPackGenerator = old_generator
            validation_run_module.Ws6SplitSeedResult = old_split_result

        signoff_pack_path = Path(self.temp_dir) / run.run_id / "signoff_pack.json"
        self.assertFalse(signoff_pack_path.exists())
    
    def test_signoff_fail_leakage_audit(self):
        """Test FAIL decision when leakage audit fails."""
        manifest_path = Path(self.temp_dir) / "manifest.json"
        manifest_path.write_text(json.dumps({"elder_id": "HK001"}))
        
        run = self.manager.create_run(
            elder_id="HK001",
            duration_days=7,
            manifest_path=str(manifest_path),
        )
        
        # Add all 12 split-seed cells with leakage audit failing
        self._add_full_split_seed_matrix(run, leakage_pass=False, hard_pass=True, temp_dir=self.temp_dir)
        
        # Use force=True to bypass compliance exception for testing
        pack = self.manager.finalize_run(run.run_id, force=True)
        
        self.assertEqual(pack.signoff_decision, SignoffDecision.FAIL)
        self.assertEqual(pack.recommended_stage, "not_ready")
        self.assertTrue(len(pack.blocking_issues) > 0)
        self.assertTrue(any("leakage" in issue.lower() for issue in pack.blocking_issues))
    
    def test_signoff_fail_hard_gates(self):
        """Test FAIL decision when hard gates fail."""
        manifest_path = Path(self.temp_dir) / "manifest.json"
        manifest_path.write_text(json.dumps({"elder_id": "HK001"}))
        
        run = self.manager.create_run(
            elder_id="HK001",
            duration_days=7,
            manifest_path=str(manifest_path),
        )
        
        # Add all 12 split-seed cells with hard gates failing
        self._add_full_split_seed_matrix(run, leakage_pass=True, hard_pass=False, temp_dir=self.temp_dir)
        
        # Use force=True to bypass compliance exception for testing
        pack = self.manager.finalize_run(run.run_id, force=True)
        
        self.assertEqual(pack.signoff_decision, SignoffDecision.FAIL)
        self.assertEqual(pack.recommended_stage, "not_ready")
        self.assertTrue(len(pack.blocking_issues) > 0)
        self.assertTrue(any("hard" in issue.lower() for issue in pack.blocking_issues))

    def test_split_seed_matrix_duplicates_fail_compliance(self):
        """Duplicate split-seed rows must not satisfy matrix completeness."""
        manifest_path = Path(self.temp_dir) / "manifest.json"
        manifest_path.write_text(json.dumps({"elder_id": "HK001"}))
        run = self.manager.create_run(
            elder_id="HK001",
            duration_days=7,
            manifest_path=str(manifest_path),
        )

        # Add 12 rows but all with same split+seed (duplicate cells).
        for _ in range(12):
            result = self._create_split_seed_result(
                split_id="4->5",
                seed=11,
                manifest_hash=run.manifest_hash,
                leakage_pass=True,
                hard_pass=True,
                temp_dir=self.temp_dir,
            )
            self.manager.record_daily_result(run.run_id, result)

        loaded = self.manager._load_run(run.run_id)
        checklist = self.manager._build_compliance_checklist(loaded, StrictEvaluationConfig())
        self.assertFalse(checklist["split_seed_matrix_complete"])

        pack = self.manager.finalize_run(run.run_id, force=True)
        self.assertEqual(pack.signoff_decision, SignoffDecision.FAIL)
        self.assertTrue(any("split-seed matrix" in msg.lower() for msg in pack.blocking_issues))
    
    def test_compliance_checklist_includes_leakage(self):
        """Test that compliance checklist includes leakage audit checks."""
        run = ValidationRun(
            run_id="test_run",
            elder_id="HK001",
            start_date="2026-02-01T00:00:00",
            duration_days=7,
            manifest_path="/path/manifest.json",
            manifest_hash="abc123",
            status=ValidationRunStatus.COMPLETED,
        )
        
        # Add all 12 split-seed cells
        splits = [
            ([4], 5),
            ([4, 5], 6),
            ([4, 5, 6], 7),
            ([4, 5, 6, 7], 8),
        ]
        seeds = [11, 22, 33]
        
        for train_days, val_day in splits:
            split_id = f"{','.join(map(str, train_days))}->{val_day}"
            for seed in seeds:
                run.daily_results.append(DailyRunResult(
                    date=f"2026-02-{split_id}_seed{seed}",
                    run_timestamp=_utc_now_iso(),
                    manifest_hash="abc123",
                    rooms_trained=["bedroom"],
                    rooms_promoted=["bedroom"],
                    rooms_rejected=[],
                    decision_traces={"bedroom": f"/tmp/trace_{split_id}_seed{seed}.json"},
                    rejection_artifacts={},
                    gate_stack_summary={
                        "seed": seed,
                        "split_id": split_id,
                        "all_hard_gates_pass": True,
                        "timeline_gates_passed": 1,
                        "timeline_gates_total": 1,
                        "leakage_audit_pass": True,
                        "leakage_audit": {"pass": True, "reasons": []},
                    },
                ))
        
        checklist = self.manager._build_compliance_checklist(run)
        
        # Should include Lane D1 checks
        self.assertIn("leakage_audit_present", checklist)
        self.assertIn("leakage_audit_pass", checklist)
        self.assertIn("all_hard_gates_pass", checklist)
        self.assertIn("timeline_gates_pass", checklist)
        self.assertIn("split_seed_matrix_complete", checklist)
        
        # All should pass
        self.assertTrue(checklist["leakage_audit_present"])
        self.assertTrue(checklist["leakage_audit_pass"])
        self.assertTrue(checklist["all_hard_gates_pass"])
        self.assertTrue(checklist["timeline_gates_pass"])
        self.assertTrue(checklist["split_seed_matrix_complete"])
    
    def test_compliance_missing_leakage_audit(self):
        """Test compliance fails when leakage audit is missing."""
        run = ValidationRun(
            run_id="test_run",
            elder_id="HK001",
            start_date="2026-02-01T00:00:00",
            duration_days=7,
            manifest_path="/path/manifest.json",
            manifest_hash="abc123",
            status=ValidationRunStatus.COMPLETED,
        )
        
        # Add split-seed results with missing leakage audit
        splits = [
            ([4], 5),
            ([4, 5], 6),
            ([4, 5, 6], 7),
            ([4, 5, 6, 7], 8),
        ]
        seeds = [11, 22, 33]
        
        for train_days, val_day in splits:
            split_id = f"{','.join(map(str, train_days))}->{val_day}"
            for seed in seeds:
                run.daily_results.append(DailyRunResult(
                    date=f"2026-02-{split_id}_seed{seed}",
                    run_timestamp=_utc_now_iso(),
                    manifest_hash="abc123",
                    rooms_trained=["bedroom"],
                    rooms_promoted=["bedroom"],
                    rooms_rejected=[],
                    decision_traces={"bedroom": f"/tmp/trace_{split_id}_seed{seed}.json"},
                    rejection_artifacts={},
                    gate_stack_summary={
                        "seed": seed,
                        "split_id": split_id,
                        "all_hard_gates_pass": True,
                        "timeline_gates_passed": 1,
                        "timeline_gates_total": 1,
                        # Missing leakage_audit and leakage_audit_pass
                    },
                ))
        
        checklist = self.manager._build_compliance_checklist(run)
        
        self.assertFalse(checklist["leakage_audit_present"])
        self.assertFalse(checklist["leakage_audit_pass"])


if __name__ == '__main__':
    unittest.main()
