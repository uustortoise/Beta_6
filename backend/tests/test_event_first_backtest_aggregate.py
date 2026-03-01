from pathlib import Path
import json
import tempfile

from scripts.aggregate_event_first_backtest import aggregate_reports, _compute_artifact_hash


def test_baseline_fields_included_in_outputs():
    """Test that baseline_version and baseline_artifact_hash are emitted in both rolling and signoff."""
    reports = [
        _seed_report(11, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6),
    ]
    rolling, signoff = aggregate_reports(
        reports,
        report_paths=[Path("seed11.json")],
        comparison_window="dec4_to_dec10",
        required_split_pass_count=6,
        required_split_pass_ratio=1.0,
        baseline_version="v31",
        baseline_artifact_hash="sha256:abc123def456",
        enforce_strict_split_seed_matrix=False,
        require_leakage_artifact=False,
    )
    
    # Rolling payload should have baseline fields
    assert rolling.get("baseline_version") == "v31"
    assert rolling.get("baseline_artifact_hash") == "sha256:abc123def456"
    
    # Signoff payload should have baseline fields
    assert signoff.get("baseline_version") == "v31"
    assert signoff.get("baseline_artifact_hash") == "sha256:abc123def456"


def test_baseline_fields_included_in_config_hash():
    """Test that baseline fields are included in config_hash for run identity binding."""
    reports = [
        _seed_report(11, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6),
    ]
    
    # With baseline fields
    rolling1, _ = aggregate_reports(
        reports,
        report_paths=[Path("seed11.json")],
        comparison_window="dec4_to_dec10",
        required_split_pass_count=6,
        required_split_pass_ratio=1.0,
        baseline_version="v31",
        baseline_artifact_hash="sha256:abc123",
        enforce_strict_split_seed_matrix=False,
        require_leakage_artifact=False,
    )
    
    # Without baseline fields (but with requirement disabled)
    rolling2, _ = aggregate_reports(
        reports,
        report_paths=[Path("seed11.json")],
        comparison_window="dec4_to_dec10",
        required_split_pass_count=6,
        required_split_pass_ratio=1.0,
        baseline_version=None,
        baseline_artifact_hash=None,
        enforce_strict_split_seed_matrix=False,
        require_baseline_for_promotion=False,  # Disable requirement
        require_leakage_artifact=False,
    )
    
    # Config hashes should differ
    assert rolling1["config_hash"] != rolling2["config_hash"]


def test_baseline_fields_mandatory_for_promotion():
    """D2 Gate: Missing baseline fields must cause FAIL for promotion-grade signoff."""
    reports = [
        _seed_report(11, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6),
    ]
    rolling, signoff = aggregate_reports(
        reports,
        report_paths=[Path("seed11.json")],
        comparison_window="dec4_to_dec10",
        required_split_pass_count=6,
        required_split_pass_ratio=1.0,
        enforce_strict_split_seed_matrix=False,
        require_leakage_artifact=False,  # Disable for this test
        # Intentionally NOT providing baseline_version or baseline_artifact_hash
    )
    
    # Should FAIL due to missing baseline binding
    assert signoff["gate_decision"] == "FAIL"
    assert any("baseline_binding:missing_baseline_version" in r for r in signoff["failed_reasons"])
    assert any("baseline_binding:missing_baseline_artifact_hash" in r for r in signoff["failed_reasons"])


def test_baseline_binding_can_be_disabled():
    """Test that baseline binding requirement can be disabled for exploratory runs."""
    reports = [
        _seed_report(11, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6),
    ]
    rolling, signoff = aggregate_reports(
        reports,
        report_paths=[Path("seed11.json")],
        comparison_window="dec4_to_dec10",
        required_split_pass_count=6,
        required_split_pass_ratio=1.0,
        enforce_strict_split_seed_matrix=False,
        require_baseline_for_promotion=False,  # Disable baseline requirement
        require_leakage_artifact=False,  # Also disable for this test
    )
    
    # Should PASS when baseline requirement is disabled
    assert signoff["gate_decision"] == "PASS"
    # Fields should still be None when not provided
    assert rolling.get("baseline_version") is None
    assert rolling.get("baseline_artifact_hash") is None


def test_baseline_artifact_hash_verification():
    """D2 Gate: Baseline artifact hash must be verified when path is provided."""
    reports = [
        _seed_report(11, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6),
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        # Write a baseline artifact
        baseline_content = {"version": "v31", "metrics": {"accuracy": 0.95}}
        json.dump(baseline_content, f)
        baseline_path = Path(f.name)
    
    try:
        # Compute correct hash
        correct_hash = _compute_artifact_hash(baseline_path)
        
        # Test with correct hash - should PASS
        _, signoff_correct = aggregate_reports(
            reports,
            report_paths=[Path("seed11.json")],
            comparison_window="dec4_to_dec10",
            required_split_pass_count=6,
            required_split_pass_ratio=1.0,
            baseline_version="v31",
            baseline_artifact_hash=f"sha256:{correct_hash}",
            baseline_artifact_path=baseline_path,
            enforce_strict_split_seed_matrix=False,
            require_leakage_artifact=False,
        )
        # Should not have hash mismatch error
        assert not any("baseline_binding:hash_mismatch" in r for r in signoff_correct["failed_reasons"])
        
        # Test with incorrect hash - should FAIL
        _, signoff_incorrect = aggregate_reports(
            reports,
            report_paths=[Path("seed11.json")],
            comparison_window="dec4_to_dec10",
            required_split_pass_count=6,
            required_split_pass_ratio=1.0,
            baseline_version="v31",
            baseline_artifact_hash="sha256:wronghash123456",
            baseline_artifact_path=baseline_path,
            enforce_strict_split_seed_matrix=False,
            require_leakage_artifact=False,
        )
        # Should have hash mismatch error
        assert signoff_incorrect["gate_decision"] == "FAIL"
        assert any("baseline_binding:hash_mismatch" in r for r in signoff_incorrect["failed_reasons"])
        
    finally:
        baseline_path.unlink()


def test_baseline_artifact_path_mandatory_in_promotion_mode():
    """D2 Gate: baseline_artifact_path is mandatory when require_baseline_for_promotion=True."""
    reports = [
        _seed_report(11, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6),
    ]
    
    # Provide version and hash but NOT path - should FAIL
    _, signoff = aggregate_reports(
        reports,
        report_paths=[Path("seed11.json")],
        comparison_window="dec4_to_dec10",
        required_split_pass_count=6,
        required_split_pass_ratio=1.0,
        baseline_version="v31",
        baseline_artifact_hash="sha256:abc123",
        baseline_artifact_path=None,  # Intentionally not provided
        enforce_strict_split_seed_matrix=False,
        require_leakage_artifact=False,
    )
    
    assert signoff["gate_decision"] == "FAIL"
    assert any("baseline_binding:missing_artifact_path" in r for r in signoff["failed_reasons"])


def test_baseline_artifact_not_found():
    """D2 Gate: Missing baseline artifact file must cause FAIL."""
    reports = [
        _seed_report(11, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6),
    ]
    
    nonexistent_path = Path("/nonexistent/baseline_artifact.json")
    
    _, signoff = aggregate_reports(
        reports,
        report_paths=[Path("seed11.json")],
        comparison_window="dec4_to_dec10",
        required_split_pass_count=6,
        required_split_pass_ratio=1.0,
        baseline_version="v31",
        baseline_artifact_hash="sha256:abc123",
        baseline_artifact_path=nonexistent_path,
        enforce_strict_split_seed_matrix=False,
        require_leakage_artifact=False,
    )
    
    assert signoff["gate_decision"] == "FAIL"
    assert any("baseline_binding:artifact_not_found" in r for r in signoff["failed_reasons"])


def test_leakage_artifact_mandatory_for_promotion():
    """D2 Gate: Missing leakage audit artifact must cause FAIL for promotion-grade signoff."""
    reports = [
        _seed_report(11, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6),
    ]
    
    _, signoff = aggregate_reports(
        reports,
        report_paths=[Path("seed11.json")],
        comparison_window="dec4_to_dec10",
        required_split_pass_count=6,
        required_split_pass_ratio=1.0,
        baseline_version="v31",
        baseline_artifact_hash="sha256:abc123",
        enforce_strict_split_seed_matrix=False,
        require_leakage_artifact=True,  # Require leakage artifact
        leakage_audit_paths=None,  # But don't provide it
    )
    
    assert signoff["gate_decision"] == "FAIL"
    assert any("leakage_audit:missing_artifact_requirement" in r for r in signoff["failed_reasons"])


def test_leakage_artifact_validation():
    """D2 Gate: Leakage audit artifact must be valid JSON with correct schema."""
    reports = [
        _seed_report(11, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6),
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        # Write a valid leakage audit artifact
        audit_content = {
            "audit_pass": True,
            "split_audits": [
                {"split_id": "4->5", "pass": True, "checks": []}
            ]
        }
        json.dump(audit_content, f)
        audit_path = Path(f.name)
    
    try:
        _, signoff = aggregate_reports(
            reports,
            report_paths=[Path("seed11.json")],
            comparison_window="dec4_to_dec10",
            required_split_pass_count=6,
            required_split_pass_ratio=1.0,
            baseline_version="v31",
            baseline_artifact_hash="sha256:abc123",
            enforce_strict_split_seed_matrix=False,
            require_leakage_artifact=True,
            leakage_audit_paths=[audit_path],
        )
        
        # Should not have leakage artifact errors
        assert not any("leakage_audit:missing_artifact_requirement" in r for r in signoff["failed_reasons"])
        assert not any("leakage_audit:invalid_json" in r for r in signoff["failed_reasons"])
        
    finally:
        audit_path.unlink()


def test_leakage_artifact_invalid_json():
    """D2 Gate: Invalid JSON in leakage audit artifact must cause FAIL."""
    reports = [
        _seed_report(11, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6),
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("invalid json content")
        audit_path = Path(f.name)
    
    try:
        _, signoff = aggregate_reports(
            reports,
            report_paths=[Path("seed11.json")],
            comparison_window="dec4_to_dec10",
            required_split_pass_count=6,
            required_split_pass_ratio=1.0,
            baseline_version="v31",
            baseline_artifact_hash="sha256:abc123",
            enforce_strict_split_seed_matrix=False,
            require_leakage_artifact=True,
            leakage_audit_paths=[audit_path],
        )
        
        assert signoff["gate_decision"] == "FAIL"
        # Error message includes the JSON parse error details
        assert any("leakage_audit:seed11:invalid_json:" in r for r in signoff["failed_reasons"])
        
    finally:
        audit_path.unlink()


def test_leakage_artifact_audit_pass_false():
    """D2 Gate: Leakage audit with audit_pass=False must cause FAIL."""
    reports = [
        _seed_report(11, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6),
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        audit_content = {
            "audit_pass": False,
            "split_audits": [
                {"split_id": "4->5", "pass": False, "checks": ["temporal_leakage_detected"]}
            ]
        }
        json.dump(audit_content, f)
        audit_path = Path(f.name)
    
    try:
        _, signoff = aggregate_reports(
            reports,
            report_paths=[Path("seed11.json")],
            comparison_window="dec4_to_dec10",
            required_split_pass_count=6,
            required_split_pass_ratio=1.0,
            baseline_version="v31",
            baseline_artifact_hash="sha256:abc123",
            enforce_strict_split_seed_matrix=False,
            require_leakage_artifact=True,
            leakage_audit_paths=[audit_path],
        )
        
        assert signoff["gate_decision"] == "FAIL"
        assert any("leakage_audit:seed11:audit_pass_false" in r for r in signoff["failed_reasons"])
        
    finally:
        audit_path.unlink()


def test_inline_leakage_fallback_to_legacy_nested_payload():
    """Backward compatibility: accept legacy rooms[*].leakage_audit.pass when top-level flag is absent."""
    reports = [
        _seed_report(11, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6, leakage_pass=True),
    ]
    # Simulate legacy producer shape: no top-level leakage_audit_pass, only nested leakage_audit.pass
    for split in reports[0]["splits"]:
        for room_payload in split.get("rooms", {}).values():
            room_payload["leakage_audit"] = {"pass": True, "reasons": []}
            room_payload.pop("leakage_audit_pass", None)

    _, signoff = aggregate_reports(
        reports,
        report_paths=[Path("seed11.json")],
        comparison_window="dec4_to_dec10",
        required_split_pass_count=6,
        required_split_pass_ratio=1.0,
        baseline_version="v31",
        baseline_artifact_hash=f"sha256:{_compute_artifact_hash(Path(__file__))}",
        baseline_artifact_path=Path(__file__),
        enforce_strict_split_seed_matrix=False,
        require_leakage_artifact=False,
    )

    assert signoff["gate_decision"] == "PASS"
    assert not any("leakage_audit_inline_failed:" in r for r in signoff["failed_reasons"])


def test_aggregate_reports_marks_fail_on_kpi_and_hard_gate():
    reports = [
        _seed_report(11, sleep_mae=210.0, hard_gate_passed=5, hard_gate_total=6),
        _seed_report(22, sleep_mae=200.0, hard_gate_passed=6, hard_gate_total=6),
        _seed_report(33, sleep_mae=190.0, hard_gate_passed=6, hard_gate_total=6),
    ]
    rolling, signoff = aggregate_reports(
        reports,
        report_paths=[Path("seed11.json"), Path("seed22.json"), Path("seed33.json")],
        comparison_window="dec4_to_dec10",
        required_split_pass_count=18,
        required_split_pass_ratio=1.0,
        enforce_strict_split_seed_matrix=False,
        require_baseline_for_promotion=False,
        require_leakage_artifact=False,
    )
    assert rolling["gate_summary"]["hard_gate_all_seeds"] is False
    assert signoff["gate_decision"] == "FAIL"
    assert any("kpi_failed:Bedroom:sleep_duration_mae_minutes" in r for r in signoff["failed_reasons"])
    assert any("hard_gate_all_seeds_failed" in r for r in signoff["failed_reasons"])
    assert "timeline_release" in signoff
    assert "recommended_stage" in signoff["timeline_release"]


def test_aggregate_reports_passes_when_all_rules_met():
    reports = [
        _seed_report(11, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6),
        _seed_report(22, sleep_mae=105.0, hard_gate_passed=6, hard_gate_total=6),
        _seed_report(33, sleep_mae=98.0, hard_gate_passed=6, hard_gate_total=6),
    ]
    
    # Create a temporary baseline artifact for hash verification
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        baseline_content = {"version": "v31", "metrics": {"accuracy": 0.95}}
        json.dump(baseline_content, f)
        baseline_path = Path(f.name)
    
    try:
        correct_hash = _compute_artifact_hash(baseline_path)
        
        rolling, signoff = aggregate_reports(
            reports,
            report_paths=[Path("seed11.json"), Path("seed22.json"), Path("seed33.json")],
            comparison_window="dec4_to_dec10",
            required_split_pass_count=18,
            required_split_pass_ratio=1.0,
            baseline_version="v31",
            baseline_artifact_hash=f"sha256:{correct_hash}",
            baseline_artifact_path=baseline_path,
            enforce_strict_split_seed_matrix=False,
            require_leakage_artifact=False,
        )
        assert rolling["gate_summary"]["hard_gate_all_seeds"] is True
        assert signoff["gate_decision"] == "PASS"
        assert signoff["failed_reasons"] == []
        assert "timeline_summary" in rolling
        assert "timeline_checks" in signoff["event_first"]
    finally:
        baseline_path.unlink()


def test_fail_closed_on_missing_kpi_metric():
    """Regression test: Missing KPI metrics should cause FAIL, not default to 0.0 pass."""
    reports = [
        _seed_report_missing_bedroom(11, hard_gate_passed=6, hard_gate_total=6),
    ]
    rolling, signoff = aggregate_reports(
        reports,
        report_paths=[Path("seed11.json")],
        comparison_window="dec4_to_dec10",
        required_split_pass_count=6,
        required_split_pass_ratio=1.0,
        enforce_strict_split_seed_matrix=False,
        require_baseline_for_promotion=False,
        require_leakage_artifact=False,
    )
    # Should FAIL due to missing metric
    assert signoff["gate_decision"] == "FAIL"
    assert any("missing_metric:Bedroom:sleep_duration_mae_minutes" in r for r in signoff["failed_reasons"])


def test_fail_closed_on_missing_timeline_metric():
    """Regression test: Missing timeline metrics should cause FAIL."""
    reports = [
        _seed_report_no_timeline(11, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6),
    ]
    rolling, signoff = aggregate_reports(
        reports,
        report_paths=[Path("seed11.json")],
        comparison_window="dec4_to_dec10",
        required_split_pass_count=6,
        required_split_pass_ratio=1.0,
        enforce_strict_split_seed_matrix=False,
        require_baseline_for_promotion=False,
        require_leakage_artifact=False,
    )
    # Timeline metrics are empty, so should fail
    assert any("missing_timeline_metric:" in r for r in signoff["failed_reasons"])


def test_consistency_violation_blocks_signoff():
    """Regression test: Inconsistent elder_id across seeds should cause FAIL."""
    reports = [
        _seed_report(11, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6),
        _seed_report_inconsistent(22, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6, elder_id="DIFFERENT"),
    ]
    rolling, signoff = aggregate_reports(
        reports,
        report_paths=[Path("seed11.json"), Path("seed22.json")],
        comparison_window="dec4_to_dec10",
        required_split_pass_count=12,
        required_split_pass_ratio=1.0,
        enforce_strict_split_seed_matrix=False,
        require_baseline_for_promotion=False,
        require_leakage_artifact=False,
    )
    # Should FAIL due to consistency violation
    assert signoff["gate_decision"] == "FAIL"
    assert any("consistency_mismatch:elder_id" in r for r in signoff["failed_reasons"])


def test_missing_required_metadata_blocks_signoff():
    """Regression test: Missing required metadata should fail closed."""
    reports = [
        _seed_report(11, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6),
        _seed_report_missing_data_version(22, sleep_mae=100.0, hard_gate_passed=6, hard_gate_total=6),
    ]
    _, signoff = aggregate_reports(
        reports,
        report_paths=[Path("seed11.json"), Path("seed22.json")],
        comparison_window="dec4_to_dec10",
        required_split_pass_count=12,
        required_split_pass_ratio=1.0,
        enforce_strict_split_seed_matrix=False,
        require_baseline_for_promotion=False,
        require_leakage_artifact=False,
    )
    assert signoff["gate_decision"] == "FAIL"
    assert any("missing_metadata:data_version:seed1" in r for r in signoff["failed_reasons"])


def _seed_report(seed: int, *, sleep_mae: float, hard_gate_passed: int, hard_gate_total: int, leakage_pass: bool = True):
    return {
        "elder_id": "HK0011_jessica",
        "seed": seed,
        "days": [4, 5, 6, 7, 8, 9, 10],
        "data_version": "dec4_to_dec10",
        "feature_schema_hash": "sha256:test",
        "leakage_checklist": ["temporal_only"],
        "calibration_summary": {"method": "isotonic"},
        "summary": {
            "Bedroom": {"sleep_duration_mae_minutes": sleep_mae},
            "Bathroom": {
                "bathroom_use_mae_minutes": 30.0,
                "shower_day_precision": 0.9,
                "shower_day_recall": 0.9,
            },
            "LivingRoom": {"livingroom_active_mae_minutes": 100.0},
            "Kitchen": {"kitchen_use_mae_minutes": 80.0},
            "Entrance": {"out_minutes_mae": 80.0},
        },
        "classification_summary": {
            "Bedroom": {"macro_f1": {"mean": 0.5}},
        },
        "splits": [
            {
                "rooms": {
                    "Bedroom": {
                        "gt_targets": {"sleep_events": 1.0},
                        "pred_targets": {"sleep_events": 1.0},
                        "leakage_audit_pass": leakage_pass,
                    },
                    "LivingRoom": {
                        "gt_targets": {"livingroom_active_events": 1.0},
                        "pred_targets": {"livingroom_active_events": 1.0},
                        "leakage_audit_pass": leakage_pass,
                    },
                    "Kitchen": {
                        "gt_targets": {"kitchen_use_events": 2.0},
                        "pred_targets": {"kitchen_use_events": 2.0},
                        "leakage_audit_pass": leakage_pass,
                    },
                    "Bathroom": {
                        "gt_targets": {"bathroom_use_events": 3.0},
                        "pred_targets": {"bathroom_use_events": 3.0},
                        "leakage_audit_pass": leakage_pass,
                    },
                    "Entrance": {
                        "gt_targets": {"out_events": 2.0},
                        "pred_targets": {"out_events": 2.0},
                        "leakage_audit_pass": leakage_pass,
                    },
                },
            },
        ],
        "gate_summary": {
            "hard_gate_checks_total": hard_gate_total,
            "hard_gate_checks_passed": hard_gate_passed,
        },
    }


def _seed_report_missing_bedroom(seed: int, *, hard_gate_passed: int, hard_gate_total: int):
    """Seed report with missing Bedroom metrics (fail-closed test)."""
    return {
        "elder_id": "HK0011_jessica",
        "seed": seed,
        "days": [4, 5, 6, 7, 8, 9, 10],
        "data_version": "dec4_to_dec10",
        "feature_schema_hash": "sha256:test",
        "leakage_checklist": ["temporal_only"],
        "calibration_summary": {"method": "isotonic"},
        "summary": {
            # Missing "Bedroom" entry entirely
            "Bathroom": {
                "bathroom_use_mae_minutes": 30.0,
                "shower_day_precision": 0.9,
                "shower_day_recall": 0.9,
            },
            "LivingRoom": {"livingroom_active_mae_minutes": 100.0},
            "Kitchen": {"kitchen_use_mae_minutes": 80.0},
            "Entrance": {"out_minutes_mae": 80.0},
        },
        "classification_summary": {
            "Bedroom": {"macro_f1": {"mean": 0.5}},
        },
        "splits": [
            {
                "rooms": {
                    # Missing Bedroom timeline data
                    "LivingRoom": {
                        "gt_targets": {"livingroom_active_events": 1.0},
                        "pred_targets": {"livingroom_active_events": 1.0},
                        "leakage_audit_pass": True,
                    },
                    "Kitchen": {
                        "gt_targets": {"kitchen_use_events": 2.0},
                        "pred_targets": {"kitchen_use_events": 2.0},
                        "leakage_audit_pass": True,
                    },
                    "Bathroom": {
                        "gt_targets": {"bathroom_use_events": 3.0},
                        "pred_targets": {"bathroom_use_events": 3.0},
                        "leakage_audit_pass": True,
                    },
                    "Entrance": {
                        "gt_targets": {"out_events": 2.0},
                        "pred_targets": {"out_events": 2.0},
                        "leakage_audit_pass": True,
                    },
                },
            },
        ],
        "gate_summary": {
            "hard_gate_checks_total": hard_gate_total,
            "hard_gate_checks_passed": hard_gate_passed,
        },
    }


def _seed_report_missing_data_version(seed: int, *, sleep_mae: float, hard_gate_passed: int, hard_gate_total: int):
    report = _seed_report(seed, sleep_mae=sleep_mae, hard_gate_passed=hard_gate_passed, hard_gate_total=hard_gate_total)
    report["data_version"] = None
    return report


def _seed_report_inconsistent(seed: int, *, sleep_mae: float, hard_gate_passed: int, hard_gate_total: int, elder_id: str):
    """Seed report with inconsistent metadata for consistency guard testing."""
    return {
        "elder_id": elder_id,
        "seed": seed,
        "days": [4, 5, 6, 7, 8, 9, 10],
        "data_version": "dec4_to_dec10",
        "feature_schema_hash": "sha256:test",
        "leakage_checklist": ["temporal_only"],
        "calibration_summary": {"method": "isotonic"},
        "summary": {
            "Bedroom": {"sleep_duration_mae_minutes": sleep_mae},
            "Bathroom": {
                "bathroom_use_mae_minutes": 30.0,
                "shower_day_precision": 0.9,
                "shower_day_recall": 0.9,
            },
            "LivingRoom": {"livingroom_active_mae_minutes": 100.0},
            "Kitchen": {"kitchen_use_mae_minutes": 80.0},
            "Entrance": {"out_minutes_mae": 80.0},
        },
        "classification_summary": {
            "Bedroom": {"macro_f1": {"mean": 0.5}},
        },
        "splits": [
            {
                "rooms": {
                    "Bedroom": {
                        "gt_targets": {"sleep_events": 1.0},
                        "pred_targets": {"sleep_events": 1.0},
                        "leakage_audit_pass": True,
                    },
                    "LivingRoom": {
                        "gt_targets": {"livingroom_active_events": 1.0},
                        "pred_targets": {"livingroom_active_events": 1.0},
                        "leakage_audit_pass": True,
                    },
                    "Kitchen": {
                        "gt_targets": {"kitchen_use_events": 2.0},
                        "pred_targets": {"kitchen_use_events": 2.0},
                        "leakage_audit_pass": True,
                    },
                    "Bathroom": {
                        "gt_targets": {"bathroom_use_events": 3.0},
                        "pred_targets": {"bathroom_use_events": 3.0},
                        "leakage_audit_pass": True,
                    },
                    "Entrance": {
                        "gt_targets": {"out_events": 2.0},
                        "pred_targets": {"out_events": 2.0},
                        "leakage_audit_pass": True,
                    },
                },
            },
        ],
        "gate_summary": {
            "hard_gate_checks_total": hard_gate_total,
            "hard_gate_checks_passed": hard_gate_passed,
        },
    }


def _seed_report_no_timeline(seed: int, *, sleep_mae: float, hard_gate_passed: int, hard_gate_total: int):
    """Seed report without timeline metrics (fail-closed test)."""
    return {
        "elder_id": "HK0011_jessica",
        "seed": seed,
        "days": [4, 5, 6, 7, 8, 9, 10],
        "data_version": "dec4_to_dec10",
        "feature_schema_hash": "sha256:test",
        "leakage_checklist": ["temporal_only"],
        "calibration_summary": {"method": "isotonic"},
        "summary": {
            "Bedroom": {"sleep_duration_mae_minutes": sleep_mae},
            "Bathroom": {
                "bathroom_use_mae_minutes": 30.0,
                "shower_day_precision": 0.9,
                "shower_day_recall": 0.9,
            },
            "LivingRoom": {"livingroom_active_mae_minutes": 100.0},
            "Kitchen": {"kitchen_use_mae_minutes": 80.0},
            "Entrance": {"out_minutes_mae": 80.0},
        },
        "classification_summary": {
            "Bedroom": {"macro_f1": {"mean": 0.5}},
        },
        "splits": [
            {
                "rooms": {
                    # No timeline data (gt_targets/pred_targets)
                    # But need leakage_audit_pass for D2 validation
                },
            },
        ],
        "gate_summary": {
            "hard_gate_checks_total": hard_gate_total,
            "hard_gate_checks_passed": hard_gate_passed,
        },
    }
