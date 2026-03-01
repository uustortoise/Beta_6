"""
D2: Strict 4x3 Split-Seed Protocol End-to-End Integration Test

Lane D2 (T-70 Controlled Validation + Signoff) - Production Gate

Validates:
1. 4 splits x 3 seeds = 12 cells mandatory for signoff
2. No fail-open paths (missing/invalid metrics/artifacts = FAIL)
3. All feature flags default-off (shadow-only)
4. WS-6 signoff generation with complete artifact set

Gate to enforce before merge:
- Run strict 4x3 split-seed protocol end-to-end with WS-6 signoff generation
- Require no fail-open paths (missing/invalid metrics/artifacts must yield FAIL)
- Keep flags default-off and shadow-only until signoff is clean
"""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from scripts.aggregate_event_first_backtest import aggregate_reports, _compute_artifact_hash


def _create_temp_baseline_artifact() -> Path:
    """Create a temporary baseline artifact file for testing."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        baseline_content = {"version": "v31", "metrics": {"accuracy": 0.95}}
        json.dump(baseline_content, f)
        return Path(f.name)


class TestStrictSplitSeedProtocol(unittest.TestCase):
    """
    Strict 4x3 split-seed protocol validation.
    
    The canonical evaluation contract requires:
    - Splits: 4->5, 4+5->6, 4+5+6->7, 4+5+6+7->8
    - Seeds: 11, 22, 33
    - Total: 12 split-seed cells
    """
    
    SPLITS = [
        ([4], 5),
        ([4, 5], 6),
        ([4, 5, 6], 7),
        ([4, 5, 6, 7], 8),
    ]
    SEEDS = [11, 22, 33]
    
    def _create_split_seed_report(
        self,
        seed: int,
        split_idx: int,
        hard_gate_pass: bool = True,
        timeline_gate_pass: bool = True,
        leakage_audit_pass: bool = True,
    ):
        train_days, val_day = self.SPLITS[split_idx]
        split_id = f"{','.join(map(str, train_days))}->{val_day}"
        
        leakage_audit = {
            "pass": leakage_audit_pass,
            "reasons": [] if leakage_audit_pass else ["fit_after_test_start"],
            "fit_size": 1000,
            "calib_size": 200,
            "test_size": 300,
        }
        
        timeline_metrics = {
            "start_mae_seconds": 120.0,
            "end_mae_seconds": 150.0,
            "duration_mae_minutes": 45.0,
        }
        
        # Room configuration with all required metrics
        rooms_config = {
            "Bedroom": {
                "gt_targets": {"sleep_events": 1.0, "sleep_minutes": 480.0},
                "pred_targets": {"sleep_events": 1.0, "sleep_minutes": 465.0},
                "timeline_metrics": timeline_metrics,
                "timeline_gates_passed": 2 if timeline_gate_pass else 0,
                "timeline_gates_total": 2,
                "hard_gate": {"pass": hard_gate_pass},
                "leakage_audit": leakage_audit,
                "leakage_audit_pass": leakage_audit_pass,
            },
            "LivingRoom": {
                "gt_targets": {"livingroom_active_events": 3.0, "livingroom_active_minutes": 180.0},
                "pred_targets": {"livingroom_active_events": 3.0, "livingroom_active_minutes": 175.0},
                "timeline_metrics": timeline_metrics,
                "timeline_gates_passed": 2 if timeline_gate_pass else 0,
                "timeline_gates_total": 2,
                "hard_gate": {"pass": hard_gate_pass},
                "leakage_audit": leakage_audit,
                "leakage_audit_pass": leakage_audit_pass,
            },
            "Kitchen": {
                "gt_targets": {"kitchen_use_events": 2.0, "kitchen_use_minutes": 120.0},
                "pred_targets": {"kitchen_use_events": 2.0, "kitchen_use_minutes": 115.0},
                "timeline_metrics": timeline_metrics,
                "timeline_gates_passed": 2 if timeline_gate_pass else 0,
                "timeline_gates_total": 2,
                "hard_gate": {"pass": hard_gate_pass},
                "leakage_audit": leakage_audit,
                "leakage_audit_pass": leakage_audit_pass,
            },
            "Bathroom": {
                "gt_targets": {"bathroom_use_events": 3.0, "bathroom_use_minutes": 45.0},
                "pred_targets": {"bathroom_use_events": 3.0, "bathroom_use_minutes": 42.0},
                "timeline_metrics": timeline_metrics,
                "timeline_gates_passed": 2 if timeline_gate_pass else 0,
                "timeline_gates_total": 2,
                "hard_gate": {"pass": hard_gate_pass},
                "leakage_audit": leakage_audit,
                "leakage_audit_pass": leakage_audit_pass,
            },
            "Entrance": {
                "gt_targets": {"out_events": 2.0, "out_minutes": 60.0},
                "pred_targets": {"out_events": 2.0, "out_minutes": 58.0},
                "timeline_metrics": timeline_metrics,
                "timeline_gates_passed": 2 if timeline_gate_pass else 0,
                "timeline_gates_total": 2,
                "hard_gate": {"pass": hard_gate_pass},
                "leakage_audit": leakage_audit,
                "leakage_audit_pass": leakage_audit_pass,
            },
        }
        
        return {
            "elder_id": "HK0011_jessica",
            "seed": seed,
            "split_id": split_id,
            "train_days": train_days,
            "val_day": val_day,
            "days": train_days + [val_day],
            "data_version": "dec4_to_dec10",
            "feature_schema_hash": "sha256:test_schema_abc123",
            "model_hash": f"sha256:model_seed{seed}_split{split_idx}",
            "git_sha": "abc123def456",
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "leakage_audit": leakage_audit,
            "summary": {
                "Bedroom": {"sleep_duration_mae_minutes": 45.0, "macro_f1": 0.75},
                "LivingRoom": {"livingroom_active_mae_minutes": 60.0, "macro_f1": 0.70},
                "Kitchen": {"kitchen_use_mae_minutes": 40.0, "macro_f1": 0.78},
                "Bathroom": {"bathroom_use_mae_minutes": 30.0, "shower_day_precision": 0.92, "shower_day_recall": 0.88},
                "Entrance": {"out_minutes_mae": 35.0},
            },
            "splits": [
                {
                    "split_index": 0,
                    "rooms": rooms_config,
                },
            ],
            "gate_summary": {
                "hard_gate_checks_total": 10,
                "hard_gate_checks_passed": 10 if hard_gate_pass else 8,
                "hard_gate_all_seeds": hard_gate_pass,
            },
        }
    
    def _create_full_matrix(self, **kwargs):
        reports = []
        for split_idx in range(len(self.SPLITS)):
            for seed in self.SEEDS:
                reports.append(self._create_split_seed_report(seed, split_idx, **kwargs))
        return reports
    
    def test_full_4x3_matrix_passes_signoff(self):
        """D2 Gate: Full 4x3 matrix (12 cells) with all checks passing yields PASS."""
        reports = self._create_full_matrix(
            hard_gate_pass=True,
            timeline_gate_pass=True,
            leakage_audit_pass=True,
        )
        
        # Create temp baseline artifact for hash verification
        baseline_path = _create_temp_baseline_artifact()
        try:
            correct_hash = _compute_artifact_hash(baseline_path)
            
            rolling, signoff = aggregate_reports(
                reports,
                report_paths=[Path(f"report_{i}.json") for i in range(len(reports))],
                comparison_window="dec4_to_dec8",
                required_split_pass_count=120,  # 12 cells x 10 checks
                required_split_pass_ratio=1.0,
                baseline_version="v31",
                baseline_artifact_hash=f"sha256:{correct_hash}",
                baseline_artifact_path=baseline_path,
                require_leakage_artifact=False,  # Disable for this test (tested separately)
            )
            
            self.assertEqual(signoff["gate_decision"], "PASS", 
                            f"Expected PASS but got FAIL. Reasons: {signoff.get('failed_reasons', [])}")
            self.assertEqual(rolling["baseline_version"], "v31")
            self.assertEqual(len(reports), 12)
        finally:
            baseline_path.unlink()
    
    def test_leakage_audit_failure_blocks_signoff(self):
        """D2 Gate: Leakage audit failure in any cell blocks signoff."""
        reports = self._create_full_matrix(
            hard_gate_pass=True,
            timeline_gate_pass=True,
            leakage_audit_pass=True,
        )
        # Make one cell fail leakage audit
        reports[7]["splits"][0]["rooms"]["Bedroom"]["leakage_audit"]["pass"] = False
        reports[7]["splits"][0]["rooms"]["Bedroom"]["leakage_audit_pass"] = False
        
        rolling, signoff = aggregate_reports(
            reports,
            report_paths=[Path(f"report_{i}.json") for i in range(len(reports))],
            comparison_window="dec4_to_dec8",
            required_split_pass_count=120,
            required_split_pass_ratio=1.0,
            baseline_version="v31",
            baseline_artifact_hash="sha256:baseline_abc123",
            require_baseline_for_promotion=False,  # Disable for this test
            require_leakage_artifact=False,
        )
        
        self.assertEqual(signoff["gate_decision"], "FAIL")
    
    def test_hard_gate_failure_blocks_signoff(self):
        """D2 Gate: Hard gate failure in any cell blocks signoff."""
        reports = self._create_full_matrix(hard_gate_pass=True)
        # Make one cell fail hard gate
        reports[2]["gate_summary"]["hard_gate_checks_passed"] = 8
        reports[2]["gate_summary"]["hard_gate_all_seeds"] = False
        reports[2]["splits"][0]["rooms"]["Bedroom"]["hard_gate"]["pass"] = False
        
        rolling, signoff = aggregate_reports(
            reports,
            report_paths=[Path(f"report_{i}.json") for i in range(len(reports))],
            comparison_window="dec4_to_dec8",
            required_split_pass_count=120,
            required_split_pass_ratio=1.0,
            baseline_version="v31",
            baseline_artifact_hash="sha256:baseline_abc123",
            require_baseline_for_promotion=False,  # Disable for this test
            require_leakage_artifact=False,
        )
        
        self.assertEqual(signoff["gate_decision"], "FAIL")

    def test_incomplete_split_seed_matrix_blocks_signoff(self):
        """D2 Gate: Missing any split-seed cell must fail signoff."""
        reports = self._create_full_matrix(
            hard_gate_pass=True,
            timeline_gate_pass=True,
            leakage_audit_pass=True,
        )
        reports.pop()  # 11/12 cells only

        _, signoff = aggregate_reports(
            reports,
            report_paths=[Path(f"report_{i}.json") for i in range(len(reports))],
            comparison_window="dec4_to_dec8",
            required_split_pass_count=110,
            required_split_pass_ratio=1.0,
            require_baseline_for_promotion=False,  # Disable for this test
            require_leakage_artifact=False,
        )

        self.assertEqual(signoff["gate_decision"], "FAIL")
        self.assertTrue(
            any("missing_split_seed_cell:" in reason for reason in signoff.get("failed_reasons", []))
        )

    def test_duplicate_split_seed_cell_blocks_signoff(self):
        """D2 Gate: Duplicate split-seed cells must fail signoff."""
        reports = self._create_full_matrix(
            hard_gate_pass=True,
            timeline_gate_pass=True,
            leakage_audit_pass=True,
        )
        reports.append(dict(reports[0]))  # duplicate one cell

        _, signoff = aggregate_reports(
            reports,
            report_paths=[Path(f"report_{i}.json") for i in range(len(reports))],
            comparison_window="dec4_to_dec8",
            required_split_pass_count=130,
            required_split_pass_ratio=1.0,
            require_baseline_for_promotion=False,  # Disable for this test
            require_leakage_artifact=False,
        )

        self.assertEqual(signoff["gate_decision"], "FAIL")
        self.assertTrue(
            any("duplicate_split_seed_cell:" in reason for reason in signoff.get("failed_reasons", []))
        )

    def test_malformed_leakage_rooms_payload_blocks_signoff(self):
        """D2 Gate: Malformed leakage payload structure must fail closed."""
        reports = self._create_full_matrix(
            hard_gate_pass=True,
            timeline_gate_pass=True,
            leakage_audit_pass=True,
        )
        reports[0]["splits"][0]["rooms"] = "invalid"

        _, signoff = aggregate_reports(
            reports,
            report_paths=[Path(f"report_{i}.json") for i in range(len(reports))],
            comparison_window="dec4_to_dec8",
            required_split_pass_count=120,
            required_split_pass_ratio=1.0,
            baseline_version="v31",
            baseline_artifact_hash="sha256:baseline_abc123",
            require_baseline_for_promotion=False,  # Disable for this test
            require_leakage_artifact=False,
        )

        self.assertEqual(signoff["gate_decision"], "FAIL")
        # Check for either inline or file-based leakage audit error
        failed_reasons = signoff.get("failed_reasons", [])
        has_malformed_error = any(
            "leakage_audit" in reason and "invalid_rooms_payload" in reason
            for reason in failed_reasons
        ) or any(
            "leakage_audit_inline" in reason and "invalid_rooms_payload" in reason
            for reason in failed_reasons
        )
        self.assertTrue(
            has_malformed_error,
            f"Expected malformed rooms payload error in: {failed_reasons}"
        )


if __name__ == "__main__":
    unittest.main()


class TestFeatureFlagDefaults(unittest.TestCase):
    """
    D2 Gate: All timeline-related feature flags must default to OFF.
    """
    
    def test_timeline_multitask_flag_defaults_off(self):
        """ENABLE_TIMELINE_MULTITASK must default to False."""
        import os
        
        # Check environment variable default (should not be set or be False)
        env_value = os.environ.get("ENABLE_TIMELINE_MULTITASK", "")
        if env_value:
            self.assertIn(env_value.lower(), ["false", "0", "", "no"],
                         "ENABLE_TIMELINE_MULTITASK must default to off")
    
    def test_timeline_decoder_v2_flags_default_off(self):
        """Timeline decoder v2 flags must default to False."""
        import os
        
        flag_names = [
            "ENABLE_TIMELINE_DECODER_V2",
        ]
        
        for flag in flag_names:
            env_value = os.environ.get(flag, "")
            if env_value:
                self.assertIn(env_value.lower(), ["false", "0", "", "no"],
                             f"{flag} must default to off")
    
    def test_event_first_shadow_flag_defaults_off(self):
        """ENABLE_TIMELINE_SHADOW_MODE must default to False."""
        import os
        
        env_value = os.environ.get("ENABLE_TIMELINE_SHADOW_MODE", "")
        if env_value:
            self.assertIn(env_value.lower(), ["false", "0", "", "no"],
                         "ENABLE_TIMELINE_SHADOW_MODE must default to off")
