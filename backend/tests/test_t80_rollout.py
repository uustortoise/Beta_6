"""
T-80 Rollout Manager Tests

Validates Shadow -> Canary -> Full rollout with safety gates.
"""

import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from ml.t80_rollout_manager import (
    T80RolloutManager,
    RolloutStage,
    PromotionDecision,
    CanaryConfig,
)
from ml.beta6.registry.registry_v2 import RegistryV2


class TestT80RolloutManager(unittest.TestCase):
    """Tests for T-80 rollout manager."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = T80RolloutManager(state_dir=Path(self.temp_dir))
    
    def test_start_shadow(self):
        """T-80: Shadow mode is the starting point."""
        state = self.manager.start_shadow()
        
        self.assertEqual(state.stage, RolloutStage.SHADOW)
        self.assertIsNotNone(state.started_at)
    
    def test_promote_to_canary(self):
        """T-80: Can promote from shadow to canary with limited cohort."""
        self.manager.start_shadow()
        
        canary_elders = ["HK001", "HK002", "HK003"]
        state = self.manager.promote_to_canary(elder_ids=canary_elders)
        
        self.assertEqual(state.stage, RolloutStage.CANARY)
        self.assertEqual(state.canary_elders, canary_elders)
        self.assertIsNotNone(state.observation_end)
    
    def test_canary_max_elders_enforced(self):
        """T-80: Canary cohort limited to max 5 elders."""
        self.manager.start_shadow()
        
        too_many_elders = ["HK001", "HK002", "HK003", "HK004", "HK005", "HK006"]
        
        with self.assertRaises(ValueError) as ctx:
            self.manager.promote_to_canary(elder_ids=too_many_elders)
        
        self.assertIn("max 5 elders", str(ctx.exception))
    
    def test_canary_requires_shadow_first(self):
        """T-80: Cannot promote to canary without starting shadow."""
        with self.assertRaises(ValueError) as ctx:
            self.manager.promote_to_canary(elder_ids=["HK001"])
        
        self.assertIn("Must start shadow", str(ctx.exception))
    
    def test_evaluate_canary_hold_before_observation_complete(self):
        """T-80: Canary evaluation returns HOLD before observation period ends."""
        self.manager.start_shadow()
        self.manager.promote_to_canary(elder_ids=["HK001"])
        
        # Immediately evaluate (before 7 days)
        decision = self.manager.evaluate_canary([])
        
        self.assertEqual(decision, PromotionDecision.HOLD)
    
    def test_evaluate_canary_promote_when_all_checks_pass(self):
        """T-80: Promote when all canary checks pass."""
        config = CanaryConfig(observation_days=0)  # No wait
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])
        
        # Fix: Include elder_id for coverage check
        signoff_results = [
            {
                "elder_id": "HK001",
                "timeline_gates_pass": True,
                "hard_gates_pass": True,
                "leakage_audit_pass": True,
                "real_data_evidence_pass": True,
            }
        ]
        
        decision = manager.evaluate_canary(signoff_results)
        
        self.assertEqual(decision, PromotionDecision.PROMOTE)
    
    def test_evaluate_canary_rollback_on_leakage_failure(self):
        """T-80: Auto-rollback on leakage audit failure."""
        config = CanaryConfig(
            observation_days=0,
            auto_rollback_on_leakage_failure=True,
        )
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])
        
        # Fix: Include elder_id for coverage check
        signoff_results = [
            {
                "elder_id": "HK001",
                "timeline_gates_pass": True,
                "hard_gates_pass": True,
                "leakage_audit_pass": False,  # Leakage failure
            }
        ]
        
        decision = manager.evaluate_canary(signoff_results)
        
        self.assertEqual(decision, PromotionDecision.ROLLBACK)
    
    def test_evaluate_canary_hold_on_timeline_regression(self):
        """T-80: Hold (not rollback) on timeline quality regression."""
        config = CanaryConfig(observation_days=0)
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])
        
        # Fix: Include elder_id for coverage check
        signoff_results = [
            {
                "elder_id": "HK001",
                "timeline_gates_pass": False,  # Timeline regression
                "hard_gates_pass": True,
                "leakage_audit_pass": True,
            }
        ]
        
        decision = manager.evaluate_canary(signoff_results)
        
        self.assertEqual(decision, PromotionDecision.HOLD)
    
    def test_evaluate_canary_rollback_on_hard_gate_failure(self):
        """T-80: Rollback on hard gate failure (safety regression)."""
        config = CanaryConfig(
            observation_days=0,
            auto_rollback_on_safety_regression=True,
        )
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])
        
        # Fix: Include elder_id for coverage check
        signoff_results = [
            {
                "elder_id": "HK001",
                "timeline_gates_pass": True,
                "hard_gates_pass": False,  # Safety regression
                "leakage_audit_pass": True,
            }
        ]
        
        decision = manager.evaluate_canary(signoff_results)
        
        self.assertEqual(decision, PromotionDecision.ROLLBACK)
    
    def test_promote_to_full(self):
        """T-80: Can promote from canary to full after PROMOTE decision."""
        config = CanaryConfig(observation_days=0)
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])
        
        # First evaluate and get PROMOTE decision
        signoff_results = [
            {
                "elder_id": "HK001",
                "timeline_gates_pass": True,
                "hard_gates_pass": True,
                "leakage_audit_pass": True,
                "real_data_evidence_pass": True,
            }
        ]
        decision = manager.evaluate_canary(signoff_results)
        self.assertEqual(decision, PromotionDecision.PROMOTE)
        
        # Now can promote to full
        state = manager.promote_to_full()
        
        self.assertEqual(state.stage, RolloutStage.FULL)
        self.assertEqual(state.promotion_decision, "promoted")
    
    def test_promote_to_full_requires_canary(self):
        """T-80: Cannot promote to full without canary stage."""
        self.manager.start_shadow()
        
        with self.assertRaises(ValueError) as ctx:
            self.manager.promote_to_full()
        
        self.assertIn("Must be in canary stage", str(ctx.exception))
    
    def test_rollback(self):
        """T-80: Rollback transitions to rolled_back state."""
        self.manager.start_shadow()
        self.manager.promote_to_canary(elder_ids=["HK001"])
        
        state = self.manager.rollback(reason="Safety regression detected")
        
        self.assertEqual(state.stage, RolloutStage.ROLLED_BACK)
        self.assertIn("Safety regression", state.rollback_reason)
    
    def test_state_persistence(self):
        """T-80: Rollout state persists across manager instances."""
        # First manager instance
        self.manager.start_shadow()
        self.manager.promote_to_canary(elder_ids=["HK001", "HK002"])
        
        # Second manager instance (same state dir)
        manager2 = T80RolloutManager(state_dir=Path(self.temp_dir))
        state = manager2.get_state()
        
        self.assertIsNotNone(state)
        self.assertEqual(state.stage, RolloutStage.CANARY)
        self.assertEqual(state.canary_elders, ["HK001", "HK002"])
    
    def test_get_summary_shadow(self):
        """T-80: Summary reports correct shadow status."""
        self.manager.start_shadow()
        
        summary = self.manager.get_summary()
        
        self.assertEqual(summary.current_stage, "shadow")
        self.assertIsNone(summary.canary_progress)
        self.assertFalse(summary.promotion_ready)
    
    def test_get_summary_canary(self):
        """T-80: Summary reports canary progress."""
        config = CanaryConfig(observation_days=7)
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])
        
        summary = manager.get_summary()
        
        self.assertEqual(summary.current_stage, "canary")
        self.assertIsNotNone(summary.canary_progress)
        self.assertEqual(summary.canary_progress["elders"], ["HK001"])
        self.assertEqual(summary.canary_progress["observation_days_total"], 7)
    
    def test_can_promote_check(self):
        """T-80: Cannot promote before observation period ends."""
        config = CanaryConfig(observation_days=7)
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])
        
        summary = manager.get_summary()
        
        # Should not be ready to promote (7 days remaining)
        self.assertFalse(summary.promotion_ready)


class TestT80PromotionThresholds(unittest.TestCase):
    """Tests for T-80 promotion thresholds."""
    
    def test_timeline_gate_pass_rate_threshold(self):
        """T-80: Timeline gate pass rate must be >= 80%."""
        config = CanaryConfig(
            observation_days=0,
            min_timeline_gate_pass_rate=0.80,
        )
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001", "HK002", "HK003", "HK004", "HK005"])
        
        # 60% pass rate (below 80% threshold) - need 2 failures to be below 80%
        # Fix: Include elder_id for each result
        signoff_results = [
            {"elder_id": "HK001", "timeline_gates_pass": True, "hard_gates_pass": True, "leakage_audit_pass": True},
            {"elder_id": "HK002", "timeline_gates_pass": True, "hard_gates_pass": True, "leakage_audit_pass": True},
            {"elder_id": "HK003", "timeline_gates_pass": True, "hard_gates_pass": True, "leakage_audit_pass": True},
            {"elder_id": "HK004", "timeline_gates_pass": False, "hard_gates_pass": True, "leakage_audit_pass": True},  # Failed
            {"elder_id": "HK005", "timeline_gates_pass": False, "hard_gates_pass": True, "leakage_audit_pass": True},  # Failed
        ]
        
        decision = manager.evaluate_canary(signoff_results)
        
        self.assertEqual(decision, PromotionDecision.HOLD)
    
    def test_hard_gate_100_percent_required(self):
        """T-80: Hard gates must pass 100% (no failures allowed)."""
        config = CanaryConfig(observation_days=0)
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001", "HK002"])
        
        # One hard gate failure
        # Fix: Include elder_id for coverage check
        signoff_results = [
            {"elder_id": "HK001", "timeline_gates_pass": True, "hard_gates_pass": True, "leakage_audit_pass": True},
            {"elder_id": "HK002", "timeline_gates_pass": True, "hard_gates_pass": False, "leakage_audit_pass": True},  # Failed
        ]
        
        decision = manager.evaluate_canary(signoff_results)
        
        self.assertEqual(decision, PromotionDecision.ROLLBACK)


if __name__ == "__main__":
    unittest.main()


class TestT80FailClosed(unittest.TestCase):
    """Tests for fail-closed behavior."""
    
    def test_leakage_audit_missing_defaults_to_fail(self):
        """T-80: Missing leakage_audit_pass field defaults to FAIL (not pass)."""
        config = CanaryConfig(observation_days=0)
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])
        
        # No leakage_audit_pass field - should fail closed
        signoff_results = [
            {
                "elder_id": "HK001",
                "timeline_gates_pass": True,
                "hard_gates_pass": True,
                # leakage_audit_pass is MISSING - should fail
            }
        ]
        
        decision = manager.evaluate_canary(signoff_results)
        
        # Should rollback because leakage audit is not explicitly passed
        self.assertEqual(decision, PromotionDecision.ROLLBACK)
    
    def test_canary_requires_full_cohort_coverage(self):
        """T-80: Canary evaluation requires results for ALL elders in cohort."""
        config = CanaryConfig(observation_days=0)
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001", "HK002", "HK003"])
        
        # Only provide results for 1 of 3 elders
        signoff_results = [
            {
                "elder_id": "HK001",
                "timeline_gates_pass": True,
                "hard_gates_pass": True,
                "leakage_audit_pass": True,
            }
        ]
        
        decision = manager.evaluate_canary(signoff_results)
        
        # Should hold because coverage is incomplete
        self.assertEqual(decision, PromotionDecision.HOLD)
    
    def test_canary_rejects_extra_elders(self):
        """T-80: Canary evaluation rejects unexpected elder results."""
        config = CanaryConfig(observation_days=0)
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])
        
        # Provide results for HK001 AND unexpected HK999
        signoff_results = [
            {
                "elder_id": "HK001",
                "timeline_gates_pass": True,
                "hard_gates_pass": True,
                "leakage_audit_pass": True,
            },
            {
                "elder_id": "HK999",  # Unexpected
                "timeline_gates_pass": True,
                "hard_gates_pass": True,
                "leakage_audit_pass": True,
            }
        ]
        
        decision = manager.evaluate_canary(signoff_results)
        
        # Should hold because of unexpected elder
        self.assertEqual(decision, PromotionDecision.HOLD)
    
    def test_promote_to_full_requires_promote_decision(self):
        """T-80: Cannot promote to full without prior PROMOTE decision."""
        config = CanaryConfig(observation_days=0)
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])
        
        # Try to promote without calling evaluate_canary first
        with self.assertRaises(ValueError) as ctx:
            manager.promote_to_full()
        
        self.assertIn("no prior PROMOTE decision", str(ctx.exception))
    
    def test_promote_to_full_requires_observation_complete(self):
        """T-80: Cannot promote to full before observation period ends."""
        config = CanaryConfig(observation_days=7)  # 7 day observation
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])
        
        # Get PROMOTE decision (bypass time check in evaluate)
        signoff_results = [
            {
                "elder_id": "HK001",
                "timeline_gates_pass": True,
                "hard_gates_pass": True,
                "leakage_audit_pass": True,
            }
        ]
        # Use zero observation config just for evaluation
        config_zero = CanaryConfig(observation_days=0)
        manager_eval = T80RolloutManager(
            state_dir=manager.state_dir,
            canary_config=config_zero,
        )
        manager_eval._state = manager._state
        manager_eval.evaluate_canary(signoff_results)
        
        # Try to promote with original manager (still has 7 days)
        with self.assertRaises(ValueError) as ctx:
            manager.promote_to_full()
        
        self.assertIn("observation period not complete", str(ctx.exception))

    def test_promotion_decision_clears_after_later_failure(self):
        """T-80: Prior PROMOTE must be cleared if later canary evidence fails."""
        config = CanaryConfig(observation_days=0)
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])

        pass_result = [{
            "elder_id": "HK001",
            "timeline_gates_pass": True,
            "hard_gates_pass": True,
            "leakage_audit_pass": True,
            "real_data_evidence_pass": True,
        }]
        fail_result = [{
            "elder_id": "HK001",
            "timeline_gates_pass": False,
            "hard_gates_pass": True,
            "leakage_audit_pass": True,
        }]

        self.assertEqual(manager.evaluate_canary(pass_result), PromotionDecision.PROMOTE)
        self.assertEqual(manager.get_state().promotion_decision, "promote")
        self.assertEqual(manager.evaluate_canary(fail_result), PromotionDecision.HOLD)
        self.assertIsNone(manager.get_state().promotion_decision)

        with self.assertRaises(ValueError):
            manager.promote_to_full()

    def test_duplicate_elder_results_block_evaluation(self):
        """T-80: Duplicate elder records in canary results must fail closed."""
        config = CanaryConfig(observation_days=0)
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])

        duplicate_results = [
            {
                "elder_id": "HK001",
                "timeline_gates_pass": True,
                "hard_gates_pass": True,
                "leakage_audit_pass": True,
            },
            {
                "elder_id": "HK001",
                "timeline_gates_pass": True,
                "hard_gates_pass": True,
                "leakage_audit_pass": True,
            },
        ]

        self.assertEqual(manager.evaluate_canary(duplicate_results), PromotionDecision.HOLD)

    def test_invalid_payload_shape_blocks_evaluation(self):
        """T-80: Non-dict signoff payload entries must fail closed."""
        config = CanaryConfig(observation_days=0)
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])

        invalid_results = ["not_a_dict"]
        self.assertEqual(manager.evaluate_canary(invalid_results), PromotionDecision.HOLD)

    def test_direct_evaluate_canary_blocks_without_real_data_evidence_by_default(self):
        """Direct evaluate_canary() must enforce policy-driven real-data evidence by default."""
        config = CanaryConfig(observation_days=0)
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=config,
        )
        self.assertTrue(manager.real_data_policy.require_real_data_evidence)
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])

        # No real_data_evidence_pass field provided.
        signoff_results = [
            {
                "elder_id": "HK001",
                "timeline_gates_pass": True,
                "hard_gates_pass": True,
                "leakage_audit_pass": True,
            }
        ]

        decision = manager.evaluate_canary(signoff_results)
        self.assertEqual(decision, PromotionDecision.HOLD)

    def test_direct_evaluate_canary_no_evidence_path_holds(self):
        """Direct evaluate_canary() no-evidence path must fail closed when policy requires evidence."""
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=CanaryConfig(observation_days=0),
        )
        self.assertTrue(manager.real_data_policy.require_real_data_evidence)
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])

        # This is the direct evaluate_canary() path (not evaluate_canary_artifacts()).
        decision = manager.evaluate_canary(
            [
                {
                    "elder_id": "HK001",
                    "timeline_gates_pass": True,
                    "hard_gates_pass": True,
                    "leakage_audit_pass": True,
                }
            ]
        )
        self.assertEqual(decision, PromotionDecision.HOLD)

    def test_direct_evaluate_canary_override_cannot_disable_policy_gate(self):
        """Explicit override must not relax policy-required evidence enforcement."""
        manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=CanaryConfig(observation_days=0),
        )
        self.assertTrue(manager.real_data_policy.require_real_data_evidence)
        manager.start_shadow()
        manager.promote_to_canary(elder_ids=["HK001"])

        decision = manager.evaluate_canary(
            [
                {
                    "elder_id": "HK001",
                    "timeline_gates_pass": True,
                    "hard_gates_pass": True,
                    "leakage_audit_pass": True,
                }
            ],
            require_real_data_evidence=False,
        )
        self.assertEqual(decision, PromotionDecision.HOLD)


class TestT80ArtifactAdapter(unittest.TestCase):
    """Tests for live signoff artifact -> canary result adapter."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = T80RolloutManager(
            state_dir=Path(self.temp_dir) / "state",
            canary_config=CanaryConfig(observation_days=0),
        )
        self.artifacts_dir = Path(self.temp_dir) / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _write_json(self, name: str, payload: dict) -> Path:
        path = self.artifacts_dir / name
        path.write_text(json.dumps(payload))
        return path

    def _real_data_evidence(self, *, residents_covered: int = 2, resident_days_total: int = 14) -> dict:
        return {
            "is_real_data": True,
            "data_source": "production_shadow",
            "residents_covered": residents_covered,
            "resident_days_total": resident_days_total,
        }

    def test_build_canary_result_from_ws6_signoff(self):
        """WS-6 artifact should map to canary result with strict gate checks."""
        ws6 = {
            "run_id": "run_1",
            "elder_id": "HK001",
            "canary_real_data_evidence": self._real_data_evidence(),
            "split_seed_results": [
                {
                    "hard_gates_passed": 5,
                    "hard_gates_total": 5,
                    "timeline_gates_passed": 8,
                    "timeline_gates_total": 10,
                    "leakage_audit_pass": True,
                }
            ],
            "decision": {"decision": "PASS"},
        }
        path = self._write_json("ws6.json", ws6)
        results = self.manager.build_canary_results_from_artifacts([path])
        self.assertEqual(results[0]["elder_id"], "HK001")
        self.assertTrue(results[0]["hard_gates_pass"])
        self.assertTrue(results[0]["leakage_audit_pass"])
        self.assertTrue(results[0]["timeline_gates_pass"])  # 0.8 threshold

    def test_build_canary_result_from_validation_signoff_pack(self):
        """Validation signoff pack compliance checklist should map directly."""
        signoff_pack = {
            "run_id": "run_2",
            "canary_real_data_evidence": self._real_data_evidence(),
            "validation_run": {"elder_id": "HK002", "daily_results": []},
            "compliance_checklist": {
                "all_hard_gates_pass": True,
                "leakage_audit_present": True,
                "leakage_audit_pass": True,
                "timeline_gates_pass": True,
            },
        }
        path = self._write_json("signoff_pack.json", signoff_pack)
        results = self.manager.build_canary_results_from_artifacts([path])
        self.assertEqual(results[0]["elder_id"], "HK002")
        self.assertTrue(results[0]["hard_gates_pass"])
        self.assertTrue(results[0]["leakage_audit_pass"])
        self.assertTrue(results[0]["timeline_gates_pass"])

    def test_build_canary_result_from_aggregate_signoff_with_override(self):
        """Aggregate signoff requires elder_id override when elder_id is absent."""
        agg = {
            "gate_decision": "PASS",
            "failed_reasons": [],
            "seed_split_stability": {"hard_gate_all_seeds": True},
            "timeline_release": {"internal_ready": True, "external_ready": False},
            "canary_real_data_evidence": self._real_data_evidence(),
        }
        path = self._write_json("agg.json", agg)
        results = self.manager.build_canary_results_from_artifacts(
            [{"path": str(path), "elder_id": "HK003"}]
        )
        self.assertEqual(results[0]["elder_id"], "HK003")
        self.assertTrue(results[0]["hard_gates_pass"])
        self.assertTrue(results[0]["leakage_audit_pass"])
        self.assertTrue(results[0]["timeline_gates_pass"])

    def test_aggregate_signoff_without_elder_id_fails_closed(self):
        """Aggregate signoff without elder_id override should raise error."""
        agg = {
            "gate_decision": "PASS",
            "failed_reasons": [],
            "seed_split_stability": {"hard_gate_all_seeds": True},
            "timeline_release": {"internal_ready": True},
            "canary_real_data_evidence": self._real_data_evidence(),
        }
        path = self._write_json("agg_missing_elder.json", agg)
        with self.assertRaises(ValueError):
            self.manager.build_canary_results_from_artifacts([path])

    def test_evaluate_canary_artifacts_end_to_end(self):
        """End-to-end: evaluate canary from artifact paths with mixed schemas."""
        self.manager.start_shadow()
        self.manager.promote_to_canary(elder_ids=["HK001", "HK002"])

        ws6 = {
            "run_id": "run_1",
            "elder_id": "HK001",
            "canary_real_data_evidence": self._real_data_evidence(),
            "split_seed_results": [
                {
                    "hard_gates_passed": 5,
                    "hard_gates_total": 5,
                    "timeline_gates_passed": 9,
                    "timeline_gates_total": 10,
                    "leakage_audit_pass": True,
                }
            ],
        }
        agg = {
            "gate_decision": "PASS",
            "failed_reasons": [],
            "seed_split_stability": {"hard_gate_all_seeds": True},
            "timeline_release": {"internal_ready": True, "external_ready": False},
            "canary_real_data_evidence": self._real_data_evidence(),
        }
        ws6_path = self._write_json("ws6_hk001.json", ws6)
        agg_path = self._write_json("agg_hk002.json", agg)

        decision = self.manager.evaluate_canary_artifacts(
            [
                ws6_path,
                {"path": str(agg_path), "elder_id": "HK002"},
            ]
        )
        self.assertEqual(decision, PromotionDecision.PROMOTE)

    def test_evaluate_canary_artifacts_fails_without_real_data_evidence(self):
        """Artifact path must fail closed when real-data evidence is missing."""
        self.manager.start_shadow()
        self.manager.promote_to_canary(elder_ids=["HK001"])

        ws6_missing_evidence = {
            "run_id": "run_1",
            "elder_id": "HK001",
            "split_seed_results": [
                {
                    "hard_gates_passed": 5,
                    "hard_gates_total": 5,
                    "timeline_gates_passed": 9,
                    "timeline_gates_total": 10,
                    "leakage_audit_pass": True,
                }
            ],
        }
        path = self._write_json("ws6_missing_evidence.json", ws6_missing_evidence)
        decision = self.manager.evaluate_canary_artifacts([path])
        self.assertEqual(decision, PromotionDecision.HOLD)


class TestT80LadderAndAutoRollback(unittest.TestCase):
    """Phase 6.2 ladder progression and auto-protection tests."""

    def setUp(self):
        self.manager = T80RolloutManager(
            state_dir=Path(tempfile.mkdtemp()),
            canary_config=CanaryConfig(observation_days=0),
        )
        self.manager.start_shadow()

    def test_ladder_progression_blocks_rung2_without_phase5_acceptance(self):
        result = self.manager.evaluate_ladder_progression(
            current_rung=1,
            gate_summary={
                "mandatory_metric_floors_pass": True,
                "open_p0_incidents": 0,
                "nightly_pipeline_success_rate": 0.995,
                "drift_alerts_within_budget": True,
                "timeline_hard_gates_all_rooms_pass": True,
                "phase5_acceptance_pass": False,
            },
        )
        self.assertFalse(result.can_advance)
        self.assertIn("phase5_acceptance_required_for_rung2_plus", list(result.blockers))

    def test_ladder_progression_passes_when_all_criteria_pass(self):
        result = self.manager.evaluate_ladder_progression(
            current_rung=1,
            gate_summary={
                "mandatory_metric_floors_pass": True,
                "open_p0_incidents": 0,
                "nightly_pipeline_success_rate": 0.997,
                "drift_alerts_within_budget": True,
                "timeline_hard_gates_all_rooms_pass": True,
                "phase5_acceptance_pass": True,
            },
        )
        self.assertTrue(result.can_advance)
        self.assertEqual(result.target_rung, 2)

    def test_auto_rollback_detects_consecutive_mae_room_regression(self):
        assessment = self.manager.evaluate_auto_rollback(
            [
                {
                    "pipeline_success_rate": 0.99,
                    "mae_regression_pct_by_room": {"bedroom": 0.12},
                },
                {
                    "pipeline_success_rate": 0.99,
                    "mae_regression_pct_by_room": {"bedroom": 0.11},
                },
            ]
        )
        self.assertTrue(assessment.should_rollback)
        self.assertIn("mae_regression_breach", list(assessment.reason_codes))

    def test_auto_rollback_keeps_zero_pipeline_success_values(self):
        assessment = self.manager.evaluate_auto_rollback(
            [
                {"pipeline_success_rate": 0.0},
                {"pipeline_success_rate": 0.0},
            ]
        )
        self.assertTrue(assessment.should_rollback)
        self.assertIn("pipeline_reliability_breach", list(assessment.reason_codes))
        self.assertEqual(
            list((assessment.details or {}).get("pipeline_success_rate_window", [])),
            [0.0, 0.0],
        )

    def test_apply_auto_rollback_protection_calls_registry_and_baseline_fallback(self):
        class _StubRegistry:
            def __init__(self):
                self.calls = []

            def rollback_and_activate_fallback(self, **kwargs):
                self.calls.append(kwargs)
                return {"ok": True, "kwargs": kwargs}

        class _StubOverrideManager:
            def __init__(self):
                self.calls = []

            def activate_baseline_fallback(self, *, reason: str):
                self.calls.append(reason)
                return True, {"reason": reason, "profile": "production"}

        stub_registry = _StubRegistry()
        stub_override = _StubOverrideManager()
        response = self.manager.apply_auto_rollback_protection(
            nightly_metrics=[
                {"pipeline_success_rate": 0.95},
                {"pipeline_success_rate": 0.96},
            ],
            elder_id="HK001",
            room="livingroom",
            run_id="run-1",
            registry_v2=stub_registry,
            override_manager=stub_override,
        )
        self.assertEqual(response["status"], "rollback_applied")
        self.assertEqual(len(stub_registry.calls), 1)
        self.assertEqual(len(stub_override.calls), 1)
        self.assertEqual(self.manager.get_state().stage, RolloutStage.ROLLED_BACK)

    def test_apply_auto_rollback_protection_multi_room_fallback_calls_registry_per_room(self):
        class _StubRegistry:
            def __init__(self):
                self.calls = []

            def rollback_and_activate_fallback(self, **kwargs):
                self.calls.append(kwargs)
                return {"ok": True, "kwargs": kwargs}

        class _StubOverrideManager:
            def __init__(self):
                self.calls = []

            def activate_baseline_fallback(self, *, reason: str):
                self.calls.append(reason)
                return True, {"reason": reason, "profile": "production"}

        stub_registry = _StubRegistry()
        stub_override = _StubOverrideManager()
        response = self.manager.apply_auto_rollback_protection(
            nightly_metrics=[
                {"pipeline_success_rate": 0.95},
                {"pipeline_success_rate": 0.96},
            ],
            elder_id="HK001",
            rooms=["livingroom", "bedroom", "livingroom", ""],
            run_id="run-2",
            registry_v2=stub_registry,
            override_manager=stub_override,
        )
        self.assertEqual(response["status"], "rollback_applied")
        self.assertEqual(len(stub_registry.calls), 2)
        self.assertEqual(
            sorted({call["room"] for call in stub_registry.calls}),
            ["bedroom", "livingroom"],
        )
        self.assertEqual(len(response["registry_action"]), 2)
        self.assertEqual(len(stub_override.calls), 1)
        self.assertEqual(self.manager.get_state().stage, RolloutStage.ROLLED_BACK)

    def test_apply_auto_rollback_protection_records_missing_fallback_target(self):
        class _StubOverrideManager:
            def __init__(self):
                self.calls = []

            def activate_baseline_fallback(self, *, reason: str):
                self.calls.append(reason)
                return True, {"reason": reason, "profile": "production"}

        registry = RegistryV2(root=self.manager.state_dir / "registry_v2")
        stub_override = _StubOverrideManager()

        response = self.manager.apply_auto_rollback_protection(
            nightly_metrics=[
                {"pipeline_success_rate": 0.95},
                {"pipeline_success_rate": 0.96},
            ],
            elder_id="HK001",
            room="livingroom",
            run_id="run-missing-target",
            registry_v2=registry,
            override_manager=stub_override,
        )

        self.assertEqual(response["status"], "rollback_applied")
        self.assertEqual(len(response["registry_action"]), 1)
        self.assertEqual(response["registry_action"][0]["room"], "livingroom")
        self.assertIn("No fallback target available", response["registry_action"][0]["error"])
        self.assertIsNone(response["registry_action"][0]["fallback_state"])
        self.assertEqual(len(stub_override.calls), 1)
        self.assertEqual(self.manager.get_state().stage, RolloutStage.ROLLED_BACK)
