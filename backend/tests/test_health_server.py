import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from backend.health_server import (
    _read_runtime_fallback_by_room,
    build_ml_snapshot_report,
    build_promotion_gate_report,
    build_walk_forward_report,
    render_promotion_gate_prometheus_metrics,
    render_walk_forward_prometheus_metrics,
)
from backend.utils.health_check import ComponentHealth, HealthChecker, HealthStatus


@patch("backend.health_server.load_room_training_dataframe")
@patch("backend.health_server.evaluate_model")
@patch("backend.health_server.UnifiedPipeline")
def test_build_walk_forward_report_includes_monitoring_metrics(
    mock_pipeline_cls, mock_evaluate_model, mock_load_df
):
    mock_pipeline = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline

    mock_pipeline.registry.load_models_for_elder.return_value = ["living room"]
    mock_pipeline.platform.sensor_columns = ["motion", "temperature"]
    mock_pipeline.platform.label_encoders = {
        "living room": MagicMock(classes_=np.array(["inactive", "watch_tv"]))
    }
    mock_pipeline.platform.class_thresholds = {"living room": {}}
    model = MagicMock()
    model.predict.return_value = np.tile(np.array([0.7, 0.3]), (10, 1))
    mock_pipeline.platform.room_models = {"living room": model}
    mock_pipeline.room_config.calculate_seq_length.return_value = 3

    def fake_preprocess(df, room, is_training, apply_denoising):
        return df

    mock_pipeline.platform.preprocess_with_resampling.side_effect = fake_preprocess
    mock_pipeline.platform.create_sequences.return_value = (np.zeros((10, 3, 2)), np.zeros(10))

    mock_load_df.return_value = (
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-02-01", periods=10, freq="h"),
                "activity": ["inactive"] * 10,
                "motion": np.random.rand(10),
                "temperature": np.random.rand(10),
                "activity_encoded": np.zeros(10, dtype=int),
            }
        ),
        None,
    )

    mock_evaluate_model.return_value = {
        "summary": {"num_folds": 1, "macro_f1_mean": 0.85, "accuracy_mean": 0.9},
        "folds": [{"macro_f1": 0.85, "accuracy": 0.9}],
    }

    report, status = build_walk_forward_report(
        elder_id="elder1",
        room_param="",
        lookback_days=30,
        min_train_days=2,
        valid_days=1,
        step_days=1,
        max_folds=3,
        drift_threshold=0.5,
    )

    assert status == 200
    metrics = report["monitoring_metrics"]
    assert metrics["total_rooms"] == 1
    assert metrics["rooms_with_drift"] == 0
    assert metrics["rooms"][0]["macro_f1_mean"] == 0.85


@patch("backend.health_server.load_room_training_dataframe")
@patch("backend.health_server.evaluate_model")
@patch("backend.health_server.UnifiedPipeline")
def test_build_walk_forward_report_encodes_labels_when_activity_encoded_missing(
    mock_pipeline_cls, mock_evaluate_model, mock_load_df
):
    mock_pipeline = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline

    mock_pipeline.registry.load_models_for_elder.return_value = ["living room"]
    mock_pipeline.platform.sensor_columns = ["motion", "temperature"]
    mock_pipeline.platform.label_encoders = {
        "living room": MagicMock(classes_=np.array(["inactive", "watch_tv"]))
    }
    mock_pipeline.platform.class_thresholds = {"living room": {}}
    model = MagicMock()
    model.predict.return_value = np.tile(np.array([0.7, 0.3]), (10, 1))
    mock_pipeline.platform.room_models = {"living room": model}
    mock_pipeline.room_config.calculate_seq_length.return_value = 3

    def fake_preprocess(df, room, is_training, apply_denoising):
        # Simulate inference-preprocessing output without activity_encoded.
        return df[["timestamp", "motion", "temperature", "activity"]].copy()

    mock_pipeline.platform.preprocess_with_resampling.side_effect = fake_preprocess
    mock_pipeline.platform.create_sequences.return_value = (np.zeros((10, 3, 2)), np.zeros(10))

    mock_load_df.return_value = (
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-02-01", periods=10, freq="h"),
                "activity": ["inactive"] * 10,
                "motion": np.random.rand(10),
                "temperature": np.random.rand(10),
            }
        ),
        None,
    )

    mock_evaluate_model.return_value = {
        "summary": {"num_folds": 1, "macro_f1_mean": 0.80, "accuracy_mean": 0.88},
        "folds": [{"macro_f1": 0.80, "accuracy": 0.88}],
    }

    report, status = build_walk_forward_report(
        elder_id="elder1",
        room_param="",
        lookback_days=30,
        min_train_days=2,
        valid_days=1,
        step_days=1,
        max_folds=3,
        drift_threshold=0.5,
    )

    assert status == 200
    assert report["status"] == "healthy"
    assert report["monitoring_metrics"]["total_rooms"] == 1


def test_render_walk_forward_prometheus_metrics():
    report = {
        "status": "degraded",
        "elder_id": "elder_123",
        "monitoring_metrics": {
            "total_rooms": 2,
            "rooms_with_drift": 1,
            "rooms": [
                {
                    "room": "bedroom",
                    "macro_f1_mean": 0.72,
                    "accuracy_mean": 0.81,
                    "drift_detected": True,
                    "num_folds": 8,
                },
                {
                    "room": "kitchen",
                    "macro_f1_mean": 0.88,
                    "accuracy_mean": 0.9,
                    "drift_detected": False,
                    "num_folds": 8,
                },
            ],
        },
    }

    text = render_walk_forward_prometheus_metrics(report)
    assert 'beta_walk_forward_status{elder_id="elder_123"} 1' in text
    assert 'beta_walk_forward_rooms_total{elder_id="elder_123"} 2' in text
    assert 'beta_walk_forward_rooms_with_drift{elder_id="elder_123"} 1' in text
    assert 'beta_walk_forward_room_macro_f1_mean{elder_id="elder_123",room="bedroom"} 0.72' in text
    assert 'beta_walk_forward_room_drift_detected{elder_id="elder_123",room="bedroom"} 1' in text


def test_health_readiness_requires_authority_contract_when_enabled(monkeypatch):
    checker = HealthChecker(check_postgresql=True)
    monkeypatch.setenv("ENABLE_BETA6_AUTHORITY", "1")
    monkeypatch.delenv("RELEASE_GATE_EVIDENCE_PROFILE", raising=False)
    monkeypatch.delenv("BETA6_GATE_SIGNING_KEY", raising=False)
    monkeypatch.setattr(
        checker,
        "check_database",
        lambda: ComponentHealth("database", HealthStatus.HEALTHY, "ok", 1.0),
    )
    monkeypatch.setattr(
        checker,
        "check_models",
        lambda: ComponentHealth("ml_models", HealthStatus.HEALTHY, "ok", 1.0),
    )
    monkeypatch.setattr(
        checker,
        "check_postgresql_preflight",
        lambda: (True, {"status": "ok"}),
    )

    out = checker.check_readiness()
    assert out["ready"] is False
    assert out["components"]["authority_contract"]["status"] == "unhealthy"


def test_health_readiness_requires_postgres_when_authority_enabled(monkeypatch):
    checker = HealthChecker(check_postgresql=True)
    monkeypatch.setenv("ENABLE_BETA6_AUTHORITY", "yes")
    monkeypatch.setenv("RELEASE_GATE_EVIDENCE_PROFILE", "production")
    monkeypatch.setenv("BETA6_GATE_SIGNING_KEY", "live-key")
    monkeypatch.setattr(
        checker,
        "check_database",
        lambda: ComponentHealth("database", HealthStatus.HEALTHY, "ok", 1.0),
    )
    monkeypatch.setattr(
        checker,
        "check_models",
        lambda: ComponentHealth("ml_models", HealthStatus.HEALTHY, "ok", 1.0),
    )
    monkeypatch.setattr(
        checker,
        "check_postgresql_preflight",
        lambda: (False, {"error": "connection refused"}),
    )

    out = checker.check_readiness()
    assert out["ready"] is False
    assert out["components"]["authority_contract"]["status"] == "unhealthy"
    assert "connection refused" in out["components"]["authority_contract"]["message"]


def test_health_checker_postgres_preflight_delegates_to_shared_helper(monkeypatch):
    from backend.utils import beta6_authority_contract as authority_contract

    checker = HealthChecker(check_postgresql=True)
    monkeypatch.setattr(
        authority_contract,
        "check_postgresql_preflight",
        lambda: (False, {"error": "shared-helper"}),
    )

    ok, details = checker.check_postgresql_preflight()

    assert ok is False
    assert details == {"error": "shared-helper"}


def test_read_runtime_fallback_by_room_reads_registry_state(tmp_path, monkeypatch):
    root = tmp_path / "models_beta6_registry_v2"
    elder_id = "elder_123"

    living_room_state = root / elder_id / "Living Room" / "fallback_state.json"
    living_room_state.parent.mkdir(parents=True, exist_ok=True)
    living_room_state.write_text('{"active": true}', encoding="utf-8")

    bedroom_state = root / elder_id / "Bedroom" / "fallback_state.json"
    bedroom_state.parent.mkdir(parents=True, exist_ok=True)
    bedroom_state.write_text('{"active": false}', encoding="utf-8")

    monkeypatch.setenv("BETA6_REGISTRY_V2_ROOT", str(root))

    out = _read_runtime_fallback_by_room(
        elder_id=elder_id,
        room_candidates={
            "livingroom": {"Living Room"},
            "bedroom": {"Bedroom"},
        },
    )
    assert out["livingroom"] is True
    assert out["bedroom"] is False


def test_build_promotion_gate_report_summarizes_recent_runs(monkeypatch):
    class _FakeCursor:
        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchall(self):
            return [
                {
                    "training_date": "2026-02-13 10:00:00",
                    "status": "success",
                    "accuracy": 0.91,
                    "metadata": {
                        "walk_forward_gate": {
                            "pass": True,
                            "reason": "all_rooms_passed",
                            "room_reports": [
                                {
                                    "room": "Bedroom",
                                    "pass": True,
                                    "candidate_summary": {"macro_f1_mean": 0.80},
                                    "champion_macro_f1_mean": 0.77,
                                    "reasons": [],
                                }
                            ],
                        }
                    },
                },
                {
                    "training_date": "2026-02-12 10:00:00",
                    "status": "rejected_by_walk_forward_gate",
                    "accuracy": 0.89,
                    "metadata": {
                        "walk_forward_gate": {
                            "pass": False,
                            "reason": "room_failures",
                            "room_reports": [
                                {
                                    "room": "Bedroom",
                                    "pass": False,
                                    "candidate_summary": {"macro_f1_mean": 0.74},
                                    "champion_macro_f1_mean": 0.78,
                                    "reasons": ["no_regress_failed:bedroom:drop=0.040>max_drop=0.030"],
                                }
                            ],
                        }
                    },
                },
            ]

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeAdapter:
        def get_connection(self):
            return _FakeConn()

    monkeypatch.setattr("backend.health_server.adapter", _FakeAdapter())

    report, status = build_promotion_gate_report(elder_id="elder_123", limit=20)
    assert status == 200
    assert report["status"] == "degraded"
    assert report["summary"]["total_runs"] == 2
    assert report["summary"]["walk_forward_enabled_runs"] == 2
    assert report["summary"]["walk_forward_pass_runs"] == 1
    assert report["summary"]["walk_forward_fail_runs"] == 1
    assert report["room_trends"][0]["room"] == "Bedroom"
    assert report["room_trends"][0]["delta_vs_previous"] == pytest.approx(0.06, abs=1e-6)


def test_render_promotion_gate_prometheus_metrics():
    report = {
        "status": "degraded",
        "elder_id": "elder_123",
        "summary": {
            "total_runs": 3,
            "walk_forward_enabled_runs": 2,
            "walk_forward_pass_runs": 1,
            "walk_forward_fail_runs": 1,
            "rejected_by_walk_forward_gate_runs": 1,
        },
        "room_trends": [
            {
                "room": "Bedroom",
                "latest_candidate_macro_f1_mean": 0.81,
                "delta_vs_previous": 0.04,
                "latest_pass": True,
            }
        ],
    }
    text = render_promotion_gate_prometheus_metrics(report)
    assert 'beta_promotion_gate_status{elder_id="elder_123"} 1' in text
    assert 'beta_promotion_gate_runs_total{elder_id="elder_123"} 3' in text
    assert 'beta_promotion_gate_wf_fail_runs{elder_id="elder_123"} 1' in text
    assert 'beta_promotion_gate_room_latest_candidate_f1{elder_id="elder_123",room="Bedroom"} 0.81' in text


def test_build_ml_snapshot_report_handles_missing_history(monkeypatch):
    class _FakeCursor:
        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchall(self):
            return []

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeAdapter:
        def get_connection(self):
            return _FakeConn()

    monkeypatch.setattr("backend.health_server.adapter", _FakeAdapter())

    report, status = build_ml_snapshot_report(elder_id="elder_123")
    assert status == 200
    assert report["status"]["overall"] == "not_available"
    assert report["window"]["lookback_runs"] == 20
    assert report["window"]["lookback_days"] is None
    assert report["rooms"] == []


def test_health_snapshot_includes_timeline_reliability_metrics(monkeypatch):
    class _FakeCursor:
        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchall(self):
            return [
                {
                    "training_date": "2026-03-07T10:00:00Z",
                    "status": "success",
                    "accuracy": 0.92,
                    "metadata": {
                        "global_gate": {"training_days": 12, "required": 0.65, "actual_global_macro_f1": 0.70},
                        "walk_forward_gate": {
                            "pass": True,
                            "reason": "all_rooms_passed",
                            "room_reports": [
                                {
                                    "room": "Kitchen",
                                    "pass": True,
                                    "reasons": [],
                                    "candidate_summary": {"macro_f1_mean": 0.81, "accuracy_mean": 0.93, "num_folds": 5},
                                    "candidate_wf_config": {"lookback_days": 90},
                                }
                            ],
                        },
                    },
                }
            ]

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeAdapter:
        def get_connection(self):
            return _FakeConn()

    monkeypatch.setattr("backend.health_server.adapter", _FakeAdapter())
    monkeypatch.setattr(
        "backend.health_server.get_timeline_reliability_metrics",
        lambda elder_id, days=30, confidence_threshold=0.60: {
            "correction_volume": 3,
            "review_backlog": 2,
            "manual_review_rate": 0.4,
            "unknown_abstain_rate": 0.2,
            "contradiction_rate": 0.1,
            "fragmentation_rate": 0.3,
            "unknown_abstain_trend": [],
        },
        raising=False,
    )

    report, status = build_ml_snapshot_report(elder_id="elder_123", lookback_runs=10)
    assert status == 200
    assert "timeline_reliability" in report
    assert report["timeline_reliability"]["correction_volume"] == 3
    assert report["timeline_reliability"]["manual_review_rate"] == pytest.approx(0.4, abs=1e-9)


def test_build_ml_snapshot_report_maps_room_status_and_thresholds(monkeypatch):
    monkeypatch.delenv("WF_DRIFT_THRESHOLD", raising=False)
    monkeypatch.delenv("WF_MIN_TRANSITION_F1", raising=False)
    monkeypatch.delenv("WF_MIN_STABILITY_ACCURACY", raising=False)
    monkeypatch.delenv("WF_MAX_TRANSITION_LOW_FOLDS", raising=False)
    monkeypatch.delenv("WF_DRIFT_THRESHOLD_BY_ROOM", raising=False)
    monkeypatch.delenv("WF_MIN_TRANSITION_F1_BY_ROOM", raising=False)
    monkeypatch.delenv("WF_MIN_STABILITY_ACCURACY_BY_ROOM", raising=False)
    monkeypatch.delenv("WF_MAX_TRANSITION_LOW_FOLDS_BY_ROOM", raising=False)

    class _FakeCursor:
        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchall(self):
            return [
                {
                    "training_date": "2026-02-26T09:12:00Z",
                    "status": "rejected_by_walk_forward_gate",
                    "accuracy": 0.9,
                    "metadata": {
                        "global_gate": {
                            "training_days": 12,
                            "required": 0.65,
                            "actual_global_macro_f1": 0.62,
                        },
                        "beta6_fallback": {"activated": ["bedroom"]},
                        "walk_forward_gate": {
                            "pass": False,
                            "reason": "room_failures",
                            "room_reports": [
                                {
                                    "room": "Bedroom",
                                    "pass": False,
                                    "reasons": ["transition_guard_failed:bedroom"],
                                    "candidate_summary": {
                                        "macro_f1_mean": 0.62,
                                        "accuracy_mean": 0.91,
                                        "num_folds": 5,
                                        "stability_accuracy_mean": 0.995,
                                        "transition_macro_f1_mean": 0.79,
                                    },
                                    "candidate_low_folds": 1,
                                    "candidate_low_transition_folds": 2,
                                    "candidate_transition_supported_folds": 4,
                                    "candidate_stability_accuracy_mean": 0.995,
                                    "candidate_transition_macro_f1_mean": 0.79,
                                    "champion_macro_f1_mean": 0.74,
                                    "candidate_wf_config": {
                                        "lookback_days": 90,
                                        "drift_threshold": 0.60,
                                        "min_transition_f1": 0.80,
                                        "min_stability_accuracy": 0.99,
                                        "max_transition_low_folds": 1,
                                    },
                                }
                            ],
                        },
                    },
                }
            ]

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeAdapter:
        def get_connection(self):
            return _FakeConn()

    monkeypatch.setattr("backend.health_server.adapter", _FakeAdapter())

    report, status = build_ml_snapshot_report(elder_id="elder_123", lookback_runs=10, include_raw=True)
    assert status == 200
    assert report["status"]["overall"] == "action_needed"
    assert report["thresholds"]["global_release_macro_f1_threshold"]["value"] == pytest.approx(0.65)
    assert len(report["rooms"]) == 1
    room = report["rooms"][0]
    assert room["room"] == "bedroom"
    assert room["status"] == "action_needed"
    assert room["fallback_active"] is True
    assert room["metrics"]["candidate_macro_f1_mean"] == pytest.approx(0.62)
    assert room["metrics"]["champion_macro_f1_mean"] == pytest.approx(0.74)
    assert room["thresholds"]["drift_threshold"]["value"] == pytest.approx(0.60)
    assert room["thresholds"]["drift_threshold"]["source"] == "policy"
    assert room["support"]["fold_count"] == 5
    assert room["support"]["transition_supported_folds"] == 4
    assert room["support"]["candidate_low_transition_folds"] == 2
    assert room["support"]["lookback_days"] == 90
    assert report["window"]["lookback_runs"] == 10
    assert report["window"]["lookback_days"] is None
    assert report["raw"] and isinstance(report["raw"], list)


def test_build_ml_snapshot_report_routes_routine_uncertainty_to_review_queue(monkeypatch):
    class _FakeCursor:
        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchall(self):
            return [
                {
                    "training_date": "2026-02-27T09:12:00Z",
                    "status": "rejected_by_walk_forward_gate",
                    "accuracy": 0.90,
                    "metadata": {
                        "global_gate": {
                            "training_days": 12,
                            "required": 0.65,
                            "actual_global_macro_f1": 0.70,
                        },
                        "walk_forward_gate": {
                            "pass": False,
                            "reason": "room_failures",
                            "room_reports": [
                                {
                                    "room": "Bedroom",
                                    "pass": False,
                                    "reasons": ["beta6_reason:fail_uncertainty_low_confidence"],
                                    "candidate_summary": {
                                        "macro_f1_mean": 0.72,
                                        "accuracy_mean": 0.91,
                                        "num_folds": 5,
                                        "stability_accuracy_mean": 0.995,
                                        "transition_macro_f1_mean": 0.89,
                                    },
                                    "candidate_low_folds": 0,
                                    "candidate_low_transition_folds": 0,
                                    "candidate_transition_supported_folds": 5,
                                    "candidate_stability_accuracy_mean": 0.995,
                                    "candidate_transition_macro_f1_mean": 0.89,
                                    "champion_macro_f1_mean": 0.70,
                                    "candidate_wf_config": {
                                        "lookback_days": 90,
                                        "drift_threshold": 0.60,
                                        "min_transition_f1": 0.80,
                                        "min_stability_accuracy": 0.99,
                                        "max_transition_low_folds": 1,
                                    },
                                }
                            ],
                        },
                    },
                }
            ]

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeAdapter:
        def get_connection(self):
            return _FakeConn()

    monkeypatch.setattr("backend.health_server.adapter", _FakeAdapter())

    report, status = build_ml_snapshot_report(elder_id="elder_123", lookback_runs=10, include_raw=False)
    assert status == 200
    assert report["status"]["overall"] == "watch"
    assert report["status"]["reason_code"] == "routed_review_queue_uncertainty"
    assert len(report["rooms"]) == 1
    room = report["rooms"][0]
    assert room["room"] == "bedroom"
    assert room["status"] == "watch"
    assert room["review_queue_recommended"] is True


def test_build_ml_snapshot_report_runtime_fallback_overrides_metadata(monkeypatch):
    class _FakeCursor:
        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchall(self):
            return [
                {
                    "training_date": "2026-02-27T09:00:00Z",
                    "status": "success",
                    "accuracy": 0.95,
                    "metadata": {
                        "global_gate": {"training_days": 12, "required": 0.65},
                        "walk_forward_gate": {
                            "pass": True,
                            "reason": "all_rooms_passed",
                            "room_reports": [
                                {
                                    "room": "Bedroom",
                                    "pass": True,
                                    "reasons": [],
                                    "candidate_summary": {
                                        "macro_f1_mean": 0.90,
                                        "accuracy_mean": 0.95,
                                        "num_folds": 6,
                                        "stability_accuracy_mean": 0.995,
                                        "transition_macro_f1_mean": 0.92,
                                    },
                                    "candidate_low_folds": 0,
                                    "candidate_low_transition_folds": 0,
                                    "candidate_transition_supported_folds": 6,
                                    "candidate_wf_config": {
                                        "lookback_days": 90,
                                        "drift_threshold": 0.60,
                                        "min_transition_f1": 0.80,
                                        "min_stability_accuracy": 0.99,
                                        "max_transition_low_folds": 1,
                                    },
                                }
                            ],
                        },
                    },
                }
            ]

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeAdapter:
        def get_connection(self):
            return _FakeConn()

    monkeypatch.setattr("backend.health_server.adapter", _FakeAdapter())
    monkeypatch.setattr(
        "backend.health_server._read_runtime_fallback_by_room",
        lambda **kwargs: {"bedroom": True},
    )

    report, status = build_ml_snapshot_report(elder_id="elder_123", lookback_runs=20)
    assert status == 200
    assert report["status"]["overall"] == "action_needed"
    assert report["status"]["reason_code"] == "fallback_active"
    assert len(report["rooms"]) == 1
    room = report["rooms"][0]
    assert room["room"] == "bedroom"
    assert room["fallback_active"] is True
    assert room["status"] == "action_needed"


def test_build_ml_snapshot_report_runtime_fallback_clears_stale_metadata(monkeypatch):
    class _FakeCursor:
        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchall(self):
            return [
                {
                    "training_date": "2026-02-27T09:00:00Z",
                    "status": "success",
                    "accuracy": 0.95,
                    "metadata": {
                        "global_gate": {"training_days": 12, "required": 0.65},
                        "beta6_fallback": {"activated": ["bedroom"]},
                        "walk_forward_gate": {
                            "pass": True,
                            "reason": "all_rooms_passed",
                            "room_reports": [
                                {
                                    "room": "Bedroom",
                                    "pass": True,
                                    "reasons": [],
                                    "candidate_summary": {
                                        "macro_f1_mean": 0.90,
                                        "accuracy_mean": 0.95,
                                        "num_folds": 6,
                                        "stability_accuracy_mean": 0.995,
                                        "transition_macro_f1_mean": 0.92,
                                    },
                                    "candidate_low_folds": 0,
                                    "candidate_low_transition_folds": 0,
                                    "candidate_transition_supported_folds": 6,
                                    "candidate_wf_config": {
                                        "lookback_days": 90,
                                        "drift_threshold": 0.60,
                                        "min_transition_f1": 0.80,
                                        "min_stability_accuracy": 0.99,
                                        "max_transition_low_folds": 1,
                                    },
                                }
                            ],
                        },
                    },
                }
            ]

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeAdapter:
        def get_connection(self):
            return _FakeConn()

    monkeypatch.setattr("backend.health_server.adapter", _FakeAdapter())
    monkeypatch.setattr(
        "backend.health_server._read_runtime_fallback_by_room",
        lambda **kwargs: {"bedroom": False},
    )
    monkeypatch.setattr("backend.health_server._hours_since", lambda _value: 1.0)

    report, status = build_ml_snapshot_report(elder_id="elder_123", lookback_runs=20)
    assert status == 200
    assert report["status"]["overall"] == "healthy"
    assert report["status"]["reason_code"] == "healthy"
    assert len(report["rooms"]) == 1
    room = report["rooms"][0]
    assert room["room"] == "bedroom"
    assert room["fallback_active"] is False
    assert room["status"] == "healthy"


def test_build_ml_snapshot_report_room_filter(monkeypatch):
    class _FakeCursor:
        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchall(self):
            return [
                {
                    "training_date": "2026-02-26T10:00:00Z",
                    "status": "success",
                    "accuracy": 0.95,
                    "metadata": {
                        "global_gate": {"training_days": 12, "required": 0.65},
                        "walk_forward_gate": {
                            "pass": True,
                            "reason": "all_rooms_passed",
                            "room_reports": [
                                {
                                    "room": "Bedroom",
                                    "pass": True,
                                    "reasons": [],
                                    "candidate_summary": {"macro_f1_mean": 0.80, "accuracy_mean": 0.95, "num_folds": 5},
                                    "candidate_wf_config": {"drift_threshold": 0.60},
                                },
                                {
                                    "room": "Kitchen",
                                    "pass": True,
                                    "reasons": [],
                                    "candidate_summary": {"macro_f1_mean": 0.78, "accuracy_mean": 0.94, "num_folds": 4},
                                    "candidate_wf_config": {"drift_threshold": 0.60},
                                },
                            ],
                        },
                    },
                }
            ]

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeAdapter:
        def get_connection(self):
            return _FakeConn()

    monkeypatch.setattr("backend.health_server.adapter", _FakeAdapter())

    report, status = build_ml_snapshot_report(elder_id="elder_123", room="Kitchen", lookback_runs=20)
    assert status == 200
    assert len(report["rooms"]) == 1
    assert report["rooms"][0]["room"] == "kitchen"


def test_build_ml_snapshot_report_partial_room_payload_degrades_gracefully(monkeypatch):
    class _FakeCursor:
        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchall(self):
            return [
                {
                    "training_date": "2026-02-27T10:00:00Z",
                    "status": "success",
                    "accuracy": 0.93,
                    "metadata": {
                        "global_gate": {"training_days": 12, "required": 0.65},
                        "walk_forward_gate": {
                            "pass": True,
                            "reason": "all_rooms_passed",
                            "room_reports": [
                                {
                                    "room": "Kitchen",
                                    "pass": True,
                                    "reasons": [],
                                    # Intentionally partial: no candidate_summary, no transition/stability.
                                    "candidate_wf_config": {"drift_threshold": 0.60},
                                }
                            ],
                        },
                    },
                }
            ]

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeAdapter:
        def get_connection(self):
            return _FakeConn()

    monkeypatch.setattr("backend.health_server.adapter", _FakeAdapter())

    report, status = build_ml_snapshot_report(elder_id="elder_123", lookback_runs=20)
    assert status == 200
    assert report["status"]["overall"] in {"watch", "not_available", "healthy"}
    assert len(report["rooms"]) == 1
    room = report["rooms"][0]
    assert room["room"] == "kitchen"
    assert room["metrics"]["candidate_macro_f1_mean"] is None
    assert room["metrics"]["candidate_transition_macro_f1_mean"] is None
    assert room["metrics"]["candidate_stability_accuracy_mean"] is None


def test_build_ml_snapshot_report_falls_back_to_global_signal_when_room_reports_missing(monkeypatch):
    class _FakeCursor:
        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchall(self):
            return [
                {
                    "training_date": "2026-02-27T10:00:00Z",
                    "status": "rejected_by_global_gate",
                    "accuracy": 0.90,
                    "metadata": {
                        "global_gate": {
                            "pass": False,
                            "training_days": 12,
                            "required": 0.65,
                            "actual_global_macro_f1": 0.61,
                        },
                        "walk_forward_gate": {
                            "pass": True,
                            "reason": "all_rooms_passed",
                            # Missing room_reports on purpose.
                        },
                    },
                }
            ]

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeAdapter:
        def get_connection(self):
            return _FakeConn()

    monkeypatch.setattr("backend.health_server.adapter", _FakeAdapter())

    report, status = build_ml_snapshot_report(elder_id="elder_123", lookback_runs=20)
    assert status == 200
    assert report["status"]["overall"] == "action_needed"
    assert report["status"]["reason_code"] == "global_gate_failed"
    assert len(report["rooms"]) == 1
    room = report["rooms"][0]
    assert room["room"] == "all_rooms"
    assert room["status"] == "action_needed"
    assert room["metrics"]["candidate_macro_f1_mean"] == pytest.approx(0.61)


def test_build_ml_snapshot_report_uses_metadata_metrics_when_room_reports_empty(monkeypatch):
    class _FakeCursor:
        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchall(self):
            return [
                {
                    "training_date": "2026-02-28T08:41:48Z",
                    "status": "rejected_by_global_gate",
                    "accuracy": 0.47,
                    "metadata": {
                        "global_gate": {
                            "pass": False,
                            "training_days": 7,
                            "required": 0.55,
                        },
                        "walk_forward_gate": {
                            "pass": False,
                            "reason": "room_failures",
                            "room_reports": [],
                        },
                        "metrics": [
                            {
                                "room": "Bedroom",
                                "macro_f1": 0.29,
                                "accuracy": 0.38,
                                "gate_pass": False,
                                "gate_reasons": ["room_threshold_failed:bedroom:f1=0.290<required=0.600"],
                            },
                            {
                                "room": "Kitchen",
                                "macro_f1": 0.11,
                                "accuracy": 0.52,
                                "gate_pass": False,
                                "gate_reasons": ["room_threshold_failed:kitchen:f1=0.110<required=0.550"],
                            },
                        ],
                    },
                }
            ]

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeAdapter:
        def get_connection(self):
            return _FakeConn()

    monkeypatch.setattr("backend.health_server.adapter", _FakeAdapter())

    report, status = build_ml_snapshot_report(elder_id="elder_123", lookback_runs=20)
    assert status == 200
    assert len(report["rooms"]) == 2
    rooms = {r["room"]: r for r in report["rooms"]}
    assert rooms["bedroom"]["metrics"]["candidate_macro_f1_mean"] == pytest.approx(0.29)
    assert rooms["kitchen"]["metrics"]["candidate_macro_f1_mean"] == pytest.approx(0.11)
    assert rooms["bedroom"]["status"] == "action_needed"


def test_build_ml_snapshot_report_uses_previous_run_when_champion_missing(monkeypatch):
    class _FakeCursor:
        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchall(self):
            return [
                {
                    "training_date": "2026-02-28T08:41:48Z",
                    "status": "rejected_by_global_gate",
                    "accuracy": 0.47,
                    "metadata": {
                        "walk_forward_gate": {"room_reports": []},
                        "metrics": [
                            {"room": "Bedroom", "macro_f1": 0.29, "accuracy": 0.38, "gate_pass": False},
                        ],
                    },
                },
                {
                    "training_date": "2026-02-28T07:41:48Z",
                    "status": "rejected_by_global_gate",
                    "accuracy": 0.44,
                    "metadata": {
                        "walk_forward_gate": {"room_reports": []},
                        "metrics": [
                            {"room": "Bedroom", "macro_f1": 0.41, "accuracy": 0.51, "gate_pass": False},
                        ],
                    },
                },
            ]

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeAdapter:
        def get_connection(self):
            return _FakeConn()

    monkeypatch.setattr("backend.health_server.adapter", _FakeAdapter())

    report, status = build_ml_snapshot_report(elder_id="elder_123", lookback_runs=20)
    assert status == 200
    assert len(report["rooms"]) == 1
    room = report["rooms"][0]
    assert room["room"] == "bedroom"
    assert room["metrics"]["candidate_macro_f1_mean"] == pytest.approx(0.29)
    assert room["metrics"]["champion_macro_f1_mean"] is None
    assert room["metrics"]["previous_run_macro_f1_mean"] == pytest.approx(0.41)


def test_build_ml_snapshot_report_masks_metrics_when_evidence_is_insufficient(monkeypatch):
    monkeypatch.delenv("WF_MIN_MINORITY_SUPPORT", raising=False)
    monkeypatch.delenv("WF_MIN_MINORITY_SUPPORT_BY_ROOM", raising=False)

    class _FakeCursor:
        def execute(self, query, params):
            self.query = query
            self.params = params

        def fetchall(self):
            return [
                {
                    "training_date": "2026-02-28T08:41:48Z",
                    "status": "rejected_by_walk_forward_gate",
                    "accuracy": 0.47,
                    "metadata": {
                        "walk_forward_gate": {
                            "pass": False,
                            "reason": "room_failures",
                            "room_reports": [
                                {
                                    "room": "Kitchen",
                                    "pass": False,
                                    "reasons": ["fold_support_failed:kitchen:low_support_folds=1<min_support=5"],
                                    "candidate_summary": {
                                        "macro_f1_mean": 0.11,
                                        "accuracy_mean": 0.52,
                                        "num_folds": 2,
                                        "transition_macro_f1_mean": 0.30,
                                    },
                                    "candidate_low_folds": 1,
                                    "candidate_low_support_folds": 1,
                                    "candidate_low_transition_folds": 1,
                                    "candidate_transition_supported_folds": 1,
                                    "candidate_wf_config": {
                                        "lookback_days": 90,
                                        "min_minority_support": 5,
                                    },
                                }
                            ],
                        },
                    },
                }
            ]

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeAdapter:
        def get_connection(self):
            return _FakeConn()

    monkeypatch.setattr("backend.health_server.adapter", _FakeAdapter())

    report, status = build_ml_snapshot_report(elder_id="elder_123", lookback_runs=20)
    assert status == 200
    assert report["status"]["overall"] == "action_needed"
    assert report["status"]["reason_code"] in {"insufficient_evidence_support", "insufficient_evidence_gate"}
    room = report["rooms"][0]
    assert room["room"] == "kitchen"
    assert room["status"] == "action_needed"
    assert room["thresholds"]["min_minority_support"]["value"] == 5
    assert room["support"]["candidate_low_support_folds"] == 1
    assert room["metrics"]["candidate_macro_f1_mean"] is None
    assert room["metrics"]["candidate_macro_f1_mean_raw"] == pytest.approx(0.11)
