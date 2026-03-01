"""
Tests for WS-6: Signoff Pack Generator

Tests signoff pack generation and decision logic.
"""

import sys
import unittest
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.signoff_pack import (
    SplitSeedResult,
    BaselineComparison,
    SignoffDecision,
    SignoffPack,
    SignoffPackGenerator,
    create_signoff_pack,
)


class TestSplitSeedResult(unittest.TestCase):
    """Tests for SplitSeedResult."""
    
    def test_creation(self):
        """Test creating a split-seed result."""
        result = SplitSeedResult(
            split_id="4->5",
            seed=11,
            train_days=[4],
            val_days=[5],
            hard_gates_passed=10,
            hard_gates_total=10,
            leakage_audit_pass=True,
        )
        
        self.assertEqual(result.split_id, "4->5")
        self.assertEqual(result.seed, 11)
        self.assertTrue(result.leakage_audit_pass)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SplitSeedResult(
            split_id="4->5",
            seed=11,
            train_days=[4],
            val_days=[5],
        )
        
        d = result.to_dict()
        
        self.assertEqual(d['split_id'], "4->5")
        self.assertEqual(d['seed'], 11)


class TestBaselineComparison(unittest.TestCase):
    """Tests for BaselineComparison."""
    
    def test_creation(self):
        """Test creating comparison."""
        comp = BaselineComparison(
            baseline_version="v31",
            candidate_version="v32",
            fragmentation_improved=True,
            duration_mae_improved=True,
            overall_improved=True,
        )
        
        self.assertEqual(comp.baseline_version, "v31")
        self.assertTrue(comp.overall_improved)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        comp = BaselineComparison(
            baseline_version="v31",
            candidate_version="v32",
        )
        
        d = comp.to_dict()
        
        self.assertEqual(d['baseline_version'], "v31")
        self.assertIn('metric_deltas', d)


class TestSignoffDecision(unittest.TestCase):
    """Tests for SignoffDecision."""
    
    def test_creation(self):
        """Test creating decision."""
        decision = SignoffDecision(
            decision="PASS",
            confidence="HIGH",
            recommended_stage="canary",
        )
        
        self.assertEqual(decision.decision, "PASS")
        self.assertEqual(decision.confidence, "HIGH")
    
    def test_to_markdown(self):
        """Test markdown report generation."""
        decision = SignoffDecision(
            decision="PASS",
            confidence="HIGH",
            recommended_stage="canary",
            primary_reasons=["All gates pass", "Metrics improved"],
            residual_risks=["Risk 1", "Risk 2"],
        )
        
        md = decision.to_markdown()
        
        self.assertIn("PASS", md)
        self.assertIn("All gates pass", md)
        self.assertIn("Residual Risks", md)


class TestSignoffPack(unittest.TestCase):
    """Tests for SignoffPack."""
    
    def test_creation(self):
        """Test creating signoff pack."""
        pack = SignoffPack(
            run_id="run_001",
            elder_id="HK0011",
            timestamp="2026-02-17T10:00:00Z",
            git_sha="abc123",
            config_hash="sha256:def456",
            baseline_version="v31",
            baseline_artifact_hash="sha256:baseline789",
        )
        
        self.assertEqual(pack.run_id, "run_001")
        self.assertEqual(pack.baseline_version, "v31")
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        pack = SignoffPack(
            run_id="run_001",
            elder_id="HK0011",
            timestamp="2026-02-17T10:00:00Z",
            git_sha="abc123",
            config_hash="sha256:def456",
            baseline_version="v31",
            baseline_artifact_hash="sha256:baseline789",
        )
        
        d = pack.to_dict()
        
        self.assertEqual(d['run_id'], "run_001")
        self.assertIn('decision', d)
    
    def test_save(self):
        """Test saving artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = SignoffPack(
                run_id="run_001",
                elder_id="HK0011",
                timestamp="2026-02-17T10:00:00Z",
                git_sha="abc123",
                config_hash="sha256:def456",
                baseline_version="v31",
                baseline_artifact_hash="sha256:baseline789",
                decision=SignoffDecision(decision="PASS", confidence="HIGH"),
            )
            
            paths = pack.save(Path(tmpdir))
            
            self.assertIn('signoff', paths)
            self.assertIn('markdown_report', paths)
            self.assertTrue(paths['signoff'].exists())
            self.assertTrue(paths['markdown_report'].exists())


class TestSignoffPackGenerator(unittest.TestCase):
    """Tests for SignoffPackGenerator."""
    
    def test_creation(self):
        """Test creating generator."""
        gen = SignoffPackGenerator(
            run_id="run_001",
            elder_id="HK0011",
            baseline_version="v31",
        )
        
        self.assertEqual(gen.run_id, "run_001")
    
    def test_add_split_seed_result(self):
        """Test adding split-seed results."""
        gen = SignoffPackGenerator("run_001", "HK0011", "v31")
        
        result = SplitSeedResult(
            split_id="4->5",
            seed=11,
            train_days=[4],
            val_days=[5],
        )
        
        gen.add_split_seed_result(result)
        
        self.assertEqual(len(gen.split_seed_results), 1)
    
    def test_compute_aggregates(self):
        """Test computing aggregate metrics."""
        gen = SignoffPackGenerator("run_001", "HK0011", "v31")
        
        gen.add_split_seed_result(SplitSeedResult(
            split_id="4->5",
            seed=11,
            train_days=[4],
            val_days=[5],
            timeline_metrics={
                'bedroom': {'segment_start_mae_minutes': 5.0, 'fragmentation_rate': 0.2},
            },
        ))
        
        gen.add_split_seed_result(SplitSeedResult(
            split_id="4->5",
            seed=22,
            train_days=[4],
            val_days=[5],
            timeline_metrics={
                'bedroom': {'segment_start_mae_minutes': 7.0, 'fragmentation_rate': 0.3},
            },
        ))
        
        aggregates = gen.compute_aggregates()
        
        self.assertIn('bedroom', aggregates)
        # Mean of 5.0 and 7.0 = 6.0
        self.assertAlmostEqual(aggregates['bedroom']['segment_start_mae_mean'], 6.0, places=1)
    
    def test_compare_to_baseline(self):
        """Test baseline comparison."""
        gen = SignoffPackGenerator("run_001", "HK0011", "v31")
        
        gen.add_split_seed_result(SplitSeedResult(
            split_id="4->5",
            seed=11,
            train_days=[4],
            val_days=[5],
            timeline_metrics={
                'bedroom': {'fragmentation_rate': 0.3},
            },
        ))
        
        baseline_metrics = {
            'bedroom': {'fragmentation_rate_mean': 0.5},
        }
        
        comparison = gen.compare_to_baseline(baseline_metrics)
        
        self.assertTrue(comparison.fragmentation_improved)  # 0.3 < 0.5
        self.assertAlmostEqual(comparison.metric_deltas['bedroom']['fragmentation_delta'], -0.2)

    def test_compare_to_baseline_uses_separate_denominators(self):
        """Duration threshold must be evaluated on duration-support rooms only."""
        gen = SignoffPackGenerator("run_001", "HK0011", "v31")
        gen.add_split_seed_result(SplitSeedResult(
            split_id="4->5",
            seed=11,
            train_days=[4],
            val_days=[5],
            timeline_metrics={
                "bedroom": {
                    "fragmentation_rate": 0.30,
                    "segment_duration_mae_minutes": 10.0,
                },
                "kitchen": {
                    "segment_duration_mae_minutes": 20.0,
                },
            },
        ))

        baseline_metrics = {
            "bedroom": {
                "fragmentation_rate_mean": 0.50,
                "segment_duration_mae_mean": 15.0,
            },
            "kitchen": {
                "segment_duration_mae_mean": 15.0,
            },
        }

        comparison = gen.compare_to_baseline(baseline_metrics)
        self.assertTrue(comparison.fragmentation_improved)  # 1/1 rooms improved
        self.assertFalse(comparison.duration_mae_improved)  # 1/2 rooms improved
        self.assertFalse(comparison.overall_improved)
    
    def test_generate_decision_pass(self):
        """Test decision generation - PASS case."""
        gen = SignoffPackGenerator("run_001", "HK0011", "v31")
        
        # Add all passing results (3 seeds x 5 splits = 15 cells)
        for seed in [11, 22, 33]:
            for split in ["4->5", "4+5->6", "4+5+6->7", "4+5+6+7->8", "4+5+6+7+8->9"]:
                gen.add_split_seed_result(SplitSeedResult(
                    split_id=split,
                    seed=seed,
                    train_days=[4],
                    val_days=[5],
                    hard_gates_passed=10,
                    hard_gates_total=10,
                    leakage_audit_pass=True,
                ))
        
        decision = gen.generate_decision()
        
        self.assertEqual(decision.decision, "PASS")
        self.assertEqual(decision.recommended_stage, "canary")
        self.assertGreater(len(decision.residual_risks), 0)  # Mandatory
    
    def test_generate_decision_fail_low_pass_rate(self):
        """Test decision generation - FAIL due to low pass rate."""
        gen = SignoffPackGenerator("run_001", "HK0011", "v31")
        
        # Add mixed results (only 50% pass)
        for i in range(10):
            gen.add_split_seed_result(SplitSeedResult(
                split_id=f"split_{i}",
                seed=11,
                train_days=[4],
                val_days=[5],
                hard_gates_passed=5 if i < 5 else 10,  # Only 50% fully pass
                hard_gates_total=10,
                leakage_audit_pass=i >= 5,
            ))
        
        decision = gen.generate_decision()
        
        self.assertEqual(decision.decision, "FAIL")
        self.assertIn("50.0%", decision.blocking_issues[0])
    
    def test_generate_decision_conditional(self):
        """Test decision generation - CONDITIONAL case."""
        gen = SignoffPackGenerator("run_001", "HK0011", "v31")
        
        # Add mostly passing results (1 failure out of 10)
        for i in range(10):
            gen.add_split_seed_result(SplitSeedResult(
                split_id=f"split_{i}",
                seed=11,
                train_days=[4],
                val_days=[5],
                hard_gates_passed=10 if i > 0 else 8,
                hard_gates_total=10,
                leakage_audit_pass=i > 0,
            ))
        
        decision = gen.generate_decision()
        
        self.assertEqual(decision.decision, "CONDITIONAL")
        self.assertEqual(decision.recommended_stage, "canary")

    def test_generate_decision_fails_when_hard_gates_not_evaluated(self):
        """Cells with zero hard-gate totals must not be counted as passing."""
        gen = SignoffPackGenerator("run_001", "HK0011", "v31")
        for i in range(5):
            gen.add_split_seed_result(SplitSeedResult(
                split_id=f"split_{i}",
                seed=11,
                train_days=[4],
                val_days=[5],
                hard_gates_passed=0,
                hard_gates_total=0,
                leakage_audit_pass=True,
            ))

        decision = gen.generate_decision()
        self.assertEqual(decision.decision, "FAIL")
    
    def test_generate_full_pack(self):
        """Test generating complete signoff pack."""
        gen = SignoffPackGenerator("run_001", "HK0011", "v31", git_sha="abc123")
        
        # Add multiple split-seed results for robust statistics
        for seed in [11, 22, 33]:
            gen.add_split_seed_result(SplitSeedResult(
                split_id="4->5",
                seed=seed,
                train_days=[4],
                val_days=[5],
                timeline_metrics={'bedroom': {
                    'fragmentation_rate': 0.3,  # Improved from 0.5 (40% improvement)
                    'segment_start_mae_minutes': 5.0,
                    'segment_duration_mae_minutes': 45.0,  # Improved from 60.0
                }},
                hard_gates_passed=10,
                hard_gates_total=10,
                leakage_audit_pass=True,
            ))
        
        pack = gen.generate(
            baseline_metrics={'bedroom': {
                'fragmentation_rate_mean': 0.5,
                'segment_start_mae_mean': 6.0,
                'segment_duration_mae_mean': 60.0,
            }},
            data_version="dec4_to_dec10",
        )
        
        self.assertEqual(pack.run_id, "run_001")
        self.assertEqual(pack.git_sha, "abc123")
        self.assertIsNotNone(pack.baseline_comparison)
        # With 100% pass rate and improvement in both metrics, should be PASS
        self.assertEqual(pack.decision.decision, "PASS")


class TestCreateSignoffPack(unittest.TestCase):
    """Tests for convenience function."""
    
    def test_create_and_save(self):
        """Test creating and saving signoff pack."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_results = [
                {
                    'split_id': '4->5',
                    'seed': 11,
                    'train_days': [4],
                    'val_days': [5],
                    'hard_gates_passed': 10,
                    'hard_gates_total': 10,
                    'leakage_audit_pass': True,
                },
            ]
            
            pack = create_signoff_pack(
                run_id="run_001",
                elder_id="HK0011",
                baseline_version="v31",
                split_seed_results=split_results,
                git_sha="abc123",
                output_dir=Path(tmpdir),
            )
            
            self.assertEqual(pack.run_id, "run_001")
            # Should have saved files
            self.assertTrue((Path(tmpdir) / "run_001_signoff.json").exists())


if __name__ == '__main__':
    unittest.main()
