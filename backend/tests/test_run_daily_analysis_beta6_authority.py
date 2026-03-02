from pathlib import Path
import json
from types import SimpleNamespace

import run_daily_analysis


class _RegistryV2Stub:
    def __init__(self) -> None:
        self.events = []

    def append_event(self, event) -> None:  # noqa: ANN001
        self.events.append(event)


def test_beta6_authority_parity_mismatch_blocks_room(monkeypatch):
    monkeypatch.setattr("run_daily_analysis._is_beta6_authority_enabled", lambda: True)
    metrics = [
        {
            "room": "Bedroom",
            "gate_pass": True,
            "gate_reasons": [],
            "beta6_parity_trace": [
                {"label": "occupied"},
                {"label": "unoccupied"},
                {"label": "occupied"},
            ],
            "beta6_runtime_policy": {"spike_suppression": True},
            "beta6_eval_policy": {"spike_suppression": False},
        }
    ]

    gate_pass, report = run_daily_analysis._apply_beta6_gate_authority(
        metrics=metrics,
        elder_id="HK001",
        run_id="r1",
        registry_v2=None,
    )

    assert gate_pass is False
    assert report["pass"] is False
    assert report["reason_code"] == "fail_runtime_eval_parity"
    assert report["phase4_dynamic_gate"]["status"] == "ok"
    assert report["phase4_dynamic_gate"]["evaluation_report_signature"].startswith("sha256:")
    assert report["phase6_shadow_compare"]["status"] == "ok"
    assert report["phase6_shadow_compare"]["summary"]["signature"].startswith("sha256:")
    assert metrics[0]["gate_pass"] is False
    assert "beta6_reason:fail_runtime_eval_parity" in metrics[0]["gate_reasons"]
    assert metrics[0]["beta6_room_decision"]["reason_code"] == "fail_runtime_eval_parity"


def test_beta6_authority_marks_parity_unchecked_when_trace_absent(monkeypatch):
    monkeypatch.setattr("run_daily_analysis._is_beta6_authority_enabled", lambda: True)
    metrics = [{"room": "Bedroom", "gate_pass": True, "gate_reasons": []}]

    gate_pass, report = run_daily_analysis._apply_beta6_gate_authority(
        metrics=metrics,
        elder_id="HK001",
        run_id="r2",
        registry_v2=None,
    )

    assert gate_pass is True
    assert report["pass"] is True
    assert report["phase4_dynamic_gate"]["status"] == "ok"
    assert report["phase4_dynamic_gate"]["evaluation_report_path"] is None
    assert metrics[0]["beta6_room_decision"]["details"]["runtime_eval_parity_checked"] is False
    assert report["phase6_shadow_compare"]["status"] == "ok"
    assert report["phase6_shadow_compare"]["summary"]["report_path"] is None


def test_beta6_authority_phase62_controls_are_evaluated(monkeypatch):
    monkeypatch.setattr("run_daily_analysis._is_beta6_authority_enabled", lambda: True)
    calls = {"ladder": 0, "auto": 0}

    def _fake_ladder(self, *, current_rung, gate_summary):  # noqa: ANN001
        calls["ladder"] += 1
        assert isinstance(current_rung, int)
        assert "nightly_pipeline_success_rate" in gate_summary
        return SimpleNamespace(
            can_advance=False,
            target_rung=current_rung + 1,
            blockers=("phase5_acceptance_required_for_rung2_plus",),
        )

    def _fake_auto(self, *, nightly_metrics, elder_id=None, rooms=None, run_id=None, registry_v2=None, override_manager=None, room=None):  # noqa: ANN001,E501
        calls["auto"] += 1
        assert isinstance(nightly_metrics, list)
        assert nightly_metrics
        return {
            "status": "no_action",
            "assessment": {"should_rollback": False, "reason_codes": [], "details": {}},
            "registry_action": [],
            "baseline_fallback": None,
        }

    monkeypatch.setattr(run_daily_analysis.T80RolloutManager, "evaluate_ladder_progression", _fake_ladder)
    monkeypatch.setattr(run_daily_analysis.T80RolloutManager, "apply_auto_rollback_protection", _fake_auto)

    gate_pass, report = run_daily_analysis._apply_beta6_gate_authority(
        metrics=[{"room": "Bedroom", "gate_pass": True, "gate_reasons": []}],
        elder_id="HK001",
        run_id="r_phase62",
        registry_v2=None,
    )

    assert gate_pass is True
    assert calls["ladder"] == 1
    assert calls["auto"] == 1
    assert report["phase6_rollout_ladder"]["status"] == "ok"
    assert report["phase6_auto_rollback"]["status"] == "no_action"


def test_phase62_pipeline_success_rate_tracks_execution_not_model_gate(monkeypatch):
    monkeypatch.setattr("run_daily_analysis._is_beta6_authority_enabled", lambda: True)
    captured = {"gate_summary": None, "nightly_metrics": None}

    def _capture_ladder(self, *, current_rung, gate_summary):  # noqa: ANN001, ARG001
        captured["gate_summary"] = dict(gate_summary)
        return SimpleNamespace(can_advance=False, target_rung=2, blockers=())

    def _capture_auto(self, *, nightly_metrics, **kwargs):  # noqa: ANN001, ARG001
        captured["nightly_metrics"] = list(nightly_metrics)
        return {
            "status": "no_action",
            "assessment": {"should_rollback": False, "reason_codes": [], "details": {}},
            "registry_action": [],
            "baseline_fallback": None,
        }

    monkeypatch.setattr(run_daily_analysis.T80RolloutManager, "evaluate_ladder_progression", _capture_ladder)
    monkeypatch.setattr(run_daily_analysis.T80RolloutManager, "apply_auto_rollback_protection", _capture_auto)

    gate_pass, report = run_daily_analysis._apply_beta6_gate_authority(
        metrics=[{"room": "Bedroom", "gate_pass": False, "gate_reasons": ["f1_below_floor"]}],
        elder_id="HK001",
        run_id="r_phase62_quality_reject",
        registry_v2=None,
    )

    assert gate_pass is False
    assert report["pass"] is False
    assert captured["gate_summary"] is not None
    assert captured["gate_summary"]["mandatory_metric_floors_pass"] is False
    assert captured["gate_summary"]["nightly_pipeline_success_rate"] == 1.0
    assert captured["nightly_metrics"] is not None
    assert captured["nightly_metrics"][-1]["pipeline_success_rate"] == 1.0
    assert captured["nightly_metrics"][-1]["pipeline_success_source"] == "beta6_authority_execution"


def test_phase62_worst_room_f1_ignores_low_evidence_failures(monkeypatch):
    monkeypatch.setattr("run_daily_analysis._is_beta6_authority_enabled", lambda: True)
    captured = {"nightly_metrics": None}

    monkeypatch.setattr(
        run_daily_analysis.T80RolloutManager,
        "evaluate_ladder_progression",
        lambda self, **kwargs: SimpleNamespace(can_advance=False, target_rung=2, blockers=()),
    )

    def _capture_auto(self, *, nightly_metrics, **kwargs):  # noqa: ANN001, ARG001
        captured["nightly_metrics"] = list(nightly_metrics)
        return {
            "status": "no_action",
            "assessment": {"should_rollback": False, "reason_codes": [], "details": {}},
            "registry_action": [],
            "baseline_fallback": None,
        }

    monkeypatch.setattr(run_daily_analysis.T80RolloutManager, "apply_auto_rollback_protection", _capture_auto)

    gate_pass, report = run_daily_analysis._apply_beta6_gate_authority(
        metrics=[
            {
                "room": "Kitchen",
                "gate_pass": False,
                "gate_reasons": [
                    "room_threshold_failed:kitchen:f1=0.000<required=0.500",
                    "insufficient_validation_support:kitchen:5<20",
                ],
                "metric_source": "holdout_validation",
                "validation_min_class_support": 5,
                "required_minority_support": 20,
            }
        ],
        elder_id="HK001",
        run_id="r_phase62_low_support_f1",
        registry_v2=None,
    )

    assert gate_pass is False
    assert report["pass"] is False
    assert captured["nightly_metrics"] is not None
    assert captured["nightly_metrics"][-1]["worst_room_f1_below_floor"] is False


def test_beta6_authority_phase62_auto_rollback_forces_gate_fail(monkeypatch):
    monkeypatch.setattr("run_daily_analysis._is_beta6_authority_enabled", lambda: True)

    monkeypatch.setattr(
        run_daily_analysis.T80RolloutManager,
        "evaluate_ladder_progression",
        lambda self, **kwargs: SimpleNamespace(can_advance=True, target_rung=2, blockers=()),
    )
    monkeypatch.setattr(
        run_daily_analysis.T80RolloutManager,
        "apply_auto_rollback_protection",
        lambda self, **kwargs: {
            "status": "rollback_applied",
            "assessment": {"should_rollback": True, "reason_codes": ["pipeline_reliability_breach"]},
            "registry_action": [],
            "baseline_fallback": None,
        },
    )

    gate_pass, report = run_daily_analysis._apply_beta6_gate_authority(
        metrics=[{"room": "Bedroom", "gate_pass": True, "gate_reasons": []}],
        elder_id="HK001",
        run_id="r_phase62_rollback",
        registry_v2=None,
    )

    assert gate_pass is False
    assert report["pass"] is False
    assert report["reason_code"] == "rollback_triggered"
    assert report["phase6_auto_rollback"]["status"] == "rollback_applied"


def test_phase62_exception_forces_gate_fail(monkeypatch):
    monkeypatch.setattr("run_daily_analysis._is_beta6_authority_enabled", lambda: True)

    def _boom(self, *, current_rung, gate_summary):  # noqa: ANN001
        raise RuntimeError("phase62 ladder failure")

    monkeypatch.setattr(run_daily_analysis.T80RolloutManager, "evaluate_ladder_progression", _boom)

    gate_pass, report = run_daily_analysis._apply_beta6_gate_authority(
        metrics=[{"room": "Bedroom", "gate_pass": True, "gate_reasons": []}],
        elder_id="HK001",
        run_id="r_phase62_exception",
        registry_v2=None,
    )

    assert gate_pass is False
    assert report["pass"] is False
    assert report["reason_code"] == "fail_gate_policy"
    assert report["details"]["phase6_step6_2_failed"] is True
    assert "phase62 ladder failure" in report["details"]["phase6_step6_2_error"]
    assert report["phase6_rollout_ladder"]["status"] == "error"
    assert report["phase6_auto_rollback"]["status"] == "error"


def test_phase62_auto_rollback_exception_forces_gate_fail(monkeypatch):
    monkeypatch.setattr("run_daily_analysis._is_beta6_authority_enabled", lambda: True)

    monkeypatch.setattr(
        run_daily_analysis.T80RolloutManager,
        "evaluate_ladder_progression",
        lambda self, **kwargs: SimpleNamespace(can_advance=True, target_rung=2, blockers=()),
    )

    def _auto_boom(self, **kwargs):  # noqa: ANN001, ARG001
        raise RuntimeError("phase62 auto rollback failure")

    monkeypatch.setattr(run_daily_analysis.T80RolloutManager, "apply_auto_rollback_protection", _auto_boom)

    gate_pass, report = run_daily_analysis._apply_beta6_gate_authority(
        metrics=[{"room": "Bedroom", "gate_pass": True, "gate_reasons": []}],
        elder_id="HK001",
        run_id="r_phase62_auto_exception",
        registry_v2=None,
    )

    assert gate_pass is False
    assert report["pass"] is False
    assert report["reason_code"] == "fail_gate_policy"
    assert report["details"]["phase6_step6_2_failed"] is True
    assert "phase62 auto rollback failure" in report["details"]["phase6_step6_2_error"]
    assert report["phase6_rollout_ladder"]["status"] == "error"
    assert report["phase6_auto_rollback"]["status"] == "error"


def test_shadow_compare_error_blocks_ladder_progression(monkeypatch):
    monkeypatch.setattr("run_daily_analysis._is_beta6_authority_enabled", lambda: True)
    monkeypatch.setenv("BETA6_PHASE5_ACCEPTANCE_PASS", "1")

    captured = {}

    def _shadow_boom(self, **kwargs):  # noqa: ANN001, ARG001
        raise RuntimeError("shadow compare failed")

    def _capture_ladder(self, *, current_rung, gate_summary):  # noqa: ANN001
        captured["gate_summary"] = dict(gate_summary)
        return SimpleNamespace(can_advance=False, target_rung=current_rung + 1, blockers=("drift_alerts_over_budget",))

    monkeypatch.setattr(run_daily_analysis.Beta6Orchestrator, "run_phase6_shadow_compare", _shadow_boom)
    monkeypatch.setattr(run_daily_analysis.T80RolloutManager, "evaluate_ladder_progression", _capture_ladder)
    monkeypatch.setattr(
        run_daily_analysis.T80RolloutManager,
        "apply_auto_rollback_protection",
        lambda self, **kwargs: {
            "status": "no_action",
            "assessment": {"should_rollback": False, "reason_codes": [], "details": {}},
            "registry_action": [],
            "baseline_fallback": None,
        },
    )

    gate_pass, report = run_daily_analysis._apply_beta6_gate_authority(
        metrics=[{"room": "Bedroom", "gate_pass": True, "gate_reasons": []}],
        elder_id="HK001",
        run_id="r_shadow_error",
        registry_v2=None,
    )

    assert gate_pass is True
    assert report["phase6_shadow_compare"]["status"] == "error"
    assert captured["gate_summary"]["drift_alerts_within_budget"] is False
    assert report["phase6_rollout_ladder"]["gate_summary"]["drift_alerts_within_budget"] is False


def test_beta6_authority_live_run_fails_closed_when_signing_key_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("run_daily_analysis._is_beta6_authority_enabled", lambda: True)
    monkeypatch.delenv("BETA6_GATE_SIGNING_KEY", raising=False)
    monkeypatch.setattr(
        "run_daily_analysis._beta6_gate_artifact_output_dir",
        lambda elder_id, run_id, registry_v2: tmp_path,  # noqa: ARG005
    )
    metrics = [{"room": "Bedroom", "gate_pass": True, "gate_reasons": []}]
    registry = _RegistryV2Stub()

    gate_pass, report = run_daily_analysis._apply_beta6_gate_authority(
        metrics=metrics,
        elder_id="HK001",
        run_id="r_live_missing_key",
        registry_v2=registry,
    )

    assert gate_pass is False
    assert report["pass"] is False
    assert report["reason_code"] == "fail_gate_policy"
    assert report["phase4_dynamic_gate"]["status"] == "error"
    assert "BETA6_GATE_SIGNING_KEY" in str(report["phase4_dynamic_gate"]["error"])
    assert report["phase4_dynamic_gate"]["evaluation_report_path"] is None
    assert report["phase4_dynamic_gate"]["rejection_artifact_path"] is None


def test_beta6_authority_live_run_persists_phase4_artifacts(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("run_daily_analysis._is_beta6_authority_enabled", lambda: True)
    monkeypatch.setenv("BETA6_GATE_SIGNING_KEY", "test-live-key")
    monkeypatch.setattr(
        "run_daily_analysis._beta6_gate_artifact_output_dir",
        lambda elder_id, run_id, registry_v2: tmp_path,  # noqa: ARG005
    )
    metrics = [{"room": "Bedroom", "gate_pass": True, "gate_reasons": []}]
    registry = _RegistryV2Stub()

    gate_pass, report = run_daily_analysis._apply_beta6_gate_authority(
        metrics=metrics,
        elder_id="HK001",
        run_id="r_live_ok",
        registry_v2=registry,
    )

    assert gate_pass is True
    assert report["phase4_dynamic_gate"]["status"] == "ok"
    eval_path = report["phase4_dynamic_gate"]["evaluation_report_path"]
    assert isinstance(eval_path, str) and eval_path
    assert Path(eval_path).exists()
    assert report["phase4_dynamic_gate"]["evaluation_report_signature"].startswith("sha256:")
    assert report["phase4_dynamic_gate"]["rejection_artifact_path"] is None
    assert report["phase6_shadow_compare"]["status"] == "ok"
    shadow_path = report["phase6_shadow_compare"]["summary"]["report_path"]
    assert isinstance(shadow_path, str) and shadow_path
    assert Path(shadow_path).exists()
    assert report["phase6_shadow_compare"]["summary"]["signature"].startswith("sha256:")


def test_beta6_authority_stage4_error_does_not_report_nonexistent_paths(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("run_daily_analysis._is_beta6_authority_enabled", lambda: True)
    monkeypatch.setenv("BETA6_GATE_SIGNING_KEY", "test-live-key")
    monkeypatch.setattr(
        "run_daily_analysis._beta6_gate_artifact_output_dir",
        lambda elder_id, run_id, registry_v2: tmp_path,  # noqa: ARG005
    )

    def _boom(self, **kwargs):  # noqa: ANN001, ARG001
        raise RuntimeError("forced stage4 failure")

    monkeypatch.setattr(run_daily_analysis.Beta6Orchestrator, "run_phase4_dynamic_gate", _boom)
    metrics = [{"room": "Bedroom", "gate_pass": True, "gate_reasons": []}]
    registry = _RegistryV2Stub()

    gate_pass, report = run_daily_analysis._apply_beta6_gate_authority(
        metrics=metrics,
        elder_id="HK001",
        run_id="r_live_error",
        registry_v2=registry,
    )

    assert gate_pass is False
    assert report["phase4_dynamic_gate"]["status"] == "error"
    assert report["phase4_dynamic_gate"]["evaluation_report_path"] is None
    assert report["phase4_dynamic_gate"]["rejection_artifact_path"] is None


def test_beta6_authority_stage4_invalid_payload_fails_closed(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("run_daily_analysis._is_beta6_authority_enabled", lambda: True)
    monkeypatch.setenv("BETA6_GATE_SIGNING_KEY", "test-live-key")
    monkeypatch.setattr(
        "run_daily_analysis._beta6_gate_artifact_output_dir",
        lambda elder_id, run_id, registry_v2: tmp_path,  # noqa: ARG005
    )

    def _invalid_payload(self, **kwargs):  # noqa: ANN001, ARG001
        return {"unexpected": "payload"}

    monkeypatch.setattr(run_daily_analysis.Beta6Orchestrator, "run_phase4_dynamic_gate", _invalid_payload)
    metrics = [{"room": "Bedroom", "gate_pass": True, "gate_reasons": []}]
    registry = _RegistryV2Stub()

    gate_pass, report = run_daily_analysis._apply_beta6_gate_authority(
        metrics=metrics,
        elder_id="HK001",
        run_id="r_live_invalid_payload",
        registry_v2=registry,
    )

    assert gate_pass is False
    assert report["pass"] is False
    assert report["reason_code"] == "fail_gate_policy"
    assert report["phase4_dynamic_gate"]["status"] == "error"
    assert "invalid_phase4_dynamic_gate_artifacts_payload" in str(report["phase4_dynamic_gate"]["error"])
    assert report["phase4_dynamic_gate"]["evaluation_report_path"] is None
    assert report["phase4_dynamic_gate"]["rejection_artifact_path"] is None


def test_publish_runtime_policy_artifact_writes_room_activation(monkeypatch, tmp_path: Path):
    registry = run_daily_analysis.RegistryV2(root=tmp_path)
    elder_id = "HK001"
    run_id = "r_policy_1"
    metrics = [
        {
            "room": "Bedroom",
            "gate_pass": True,
            "gate_reasons": [],
            "timeline_metrics": {"duration_mae_minutes": 2.0, "fragmentation_rate": 0.05},
            "unknown_rate": 0.03,
            "abstain_rate": 0.04,
            "beta6_room_decision": {"reason_code": "pass", "details": {}},
        },
        {
            "room": "LivingRoom",
            "gate_pass": False,
            "gate_reasons": ["beta6_reason:fail_timeline_mae"],
            "timeline_metrics": {"duration_mae_minutes": 12.0, "fragmentation_rate": 0.15},
            "unknown_rate": 0.08,
            "abstain_rate": 0.11,
            "beta6_room_decision": {"reason_code": "fail_timeline_mae", "details": {}},
        },
    ]

    summary = run_daily_analysis._publish_beta6_phase4_runtime_policy(
        metrics=metrics,
        elder_id=elder_id,
        run_id=run_id,
        registry_v2=registry,
        beta6_gate_report={"pass": False, "reason_code": "fail_timeline_mae"},
        beta6_fallback_summary={"activated": ["livingroom"], "cleared": [], "errors": []},
    )

    assert summary["status"] == "ok"
    policy_path = Path(summary["path"])
    assert policy_path.exists()
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "beta6.phase4.runtime_policy.v1"
    assert payload["master_enabled"] is True
    assert payload["room_runtime"]["bedroom"]["enable_phase4_runtime"] is True
    assert payload["room_runtime"]["livingroom"]["enable_phase4_runtime"] is False


def test_publish_runtime_policy_artifact_emits_critical_shadow_drift_alert(monkeypatch, tmp_path: Path):
    registry = run_daily_analysis.RegistryV2(root=tmp_path)
    elder_id = "HK001"
    policy_path = run_daily_analysis._beta6_phase4_runtime_policy_path(elder_id, registry)
    assert policy_path is not None
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    prior = {
        "schema_version": "beta6.phase4.runtime_policy.v1",
        "history": [
            {
                "run_id": f"prior_{idx}",
                "timestamp": f"2026-02-{10+idx:02d}T00:00:00+00:00",
                "unknown_rate": 0.02,
                "abstain_rate": 0.03,
                "duration_mae_minutes": 2.0,
                "fragmentation_rate": 0.04,
            }
            for idx in range(7)
        ],
    }
    policy_path.write_text(json.dumps(prior), encoding="utf-8")

    metrics = [
        {
            "room": "Bedroom",
            "gate_pass": True,
            "gate_reasons": [],
            "timeline_metrics": {"duration_mae_minutes": 8.0, "fragmentation_rate": 0.20},
            "unknown_rate": 0.20,
            "abstain_rate": 0.22,
            "beta6_room_decision": {"reason_code": "pass", "details": {}},
        }
    ]

    summary = run_daily_analysis._publish_beta6_phase4_runtime_policy(
        metrics=metrics,
        elder_id=elder_id,
        run_id="r_policy_critical",
        registry_v2=registry,
        beta6_gate_report={"pass": True, "reason_code": "pass"},
        beta6_fallback_summary={"activated": [], "cleared": [], "errors": []},
    )

    assert summary["status"] == "ok"
    assert summary["shadow_parity_status"] == "critical"
    alert_metrics = {str(alert.get("metric")) for alert in summary["shadow_parity_alerts"]}
    assert {"unknown_rate", "abstain_rate", "duration_mae_minutes", "fragmentation_rate"} <= alert_metrics
