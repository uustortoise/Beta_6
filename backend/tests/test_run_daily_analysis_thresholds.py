import run_daily_analysis
from run_daily_analysis import _resolve_scheduled_threshold
from pathlib import Path
from unittest.mock import MagicMock
import json

import pandas as pd
import pytest

from run_daily_analysis import (
    _evaluate_backbone_alignment_gate,
    _evaluate_walk_forward_promotion_gate,
    _evaluate_global_gate,
    _is_non_destructive_global_gate_failure,
    _resolve_retrain_input_mode,
    _validate_decision_trace_artifacts,
    _is_env_enabled,
    _derive_run_failure_stage,
    _emit_registry_integrity_summary,
)


def test_global_schedule_threshold_uses_earliest_for_early_days():
    schedule = [
        {"min_days": 2, "max_days": 9, "min_value": 0.55},
        {"min_days": 10, "max_days": 21, "min_value": 0.65},
        {"min_days": 22, "max_days": None, "min_value": 0.75},
    ]
    assert _resolve_scheduled_threshold(schedule, training_days=0.0) == 0.55


def test_global_schedule_threshold_handles_unsorted_schedule():
    schedule = [
        {"min_days": 22, "max_days": None, "min_value": 0.75},
        {"min_days": 2, "max_days": 9, "min_value": 0.55},
        {"min_days": 10, "max_days": 21, "min_value": 0.65},
    ]
    assert _resolve_scheduled_threshold(schedule, training_days=1.0) == 0.55


def test_global_gate_fails_when_no_promoted_metrics(monkeypatch):
    monkeypatch.setattr(
        "run_daily_analysis.get_release_gates_config",
        lambda: {
            "release_gates": {
                "global": {
                    "schedule": [{"min_days": 2, "max_days": None, "min_value": 0.55}]
                }
            }
        },
    )

    gate_pass, report = _evaluate_global_gate(
        [
            {"room": "Bedroom", "gate_pass": False, "training_days": 10.0, "macro_f1": 0.90},
            {"room": "Kitchen", "gate_pass": False, "training_days": 10.0, "macro_f1": 0.80},
        ]
    )

    assert gate_pass is False
    assert report["pass"] is False
    assert report["reason"] == "no_promoted_metrics_for_global_gate"
    assert report["required"] == 0.55


def test_global_gate_schedule_unresolved_is_non_destructive(monkeypatch):
    monkeypatch.setattr(
        "run_daily_analysis.get_release_gates_config",
        lambda: {"release_gates": {"global": {"schedule": []}}},
    )
    gate_pass, report = _evaluate_global_gate(
        [{"room": "Bedroom", "gate_pass": True, "training_days": 10.0, "macro_f1": 0.9}]
    )
    assert gate_pass is False
    assert report["reason"] == "global_schedule_unresolved"
    assert _is_non_destructive_global_gate_failure(report) is True


def test_global_gate_config_unavailable_is_non_destructive():
    report = {"reason": "gate_config_unavailable:file-not-found"}
    assert _is_non_destructive_global_gate_failure(report) is True


def test_global_gate_metric_failure_is_destructive():
    report = {"reason": "no_promoted_metrics_for_global_gate"}
    assert _is_non_destructive_global_gate_failure(report) is False


def test_derive_run_failure_stage_distinguishes_data_vs_statistical_failures():
    viability_stage = _derive_run_failure_stage(
        global_gate_pass=True,
        decision_trace_gate_pass=True,
        walk_forward_stage_failed=False,
        metrics=[{"gate_reasons": ["insufficient_training_windows:bedroom:100<3000"]}],
    )
    assert viability_stage == "data_viability_failed"

    stat_stage = _derive_run_failure_stage(
        global_gate_pass=True,
        decision_trace_gate_pass=True,
        walk_forward_stage_failed=False,
        metrics=[{"gate_reasons": ["insufficient_validation_support:bedroom:10<30"]}],
    )
    assert stat_stage == "statistical_validity_failed"


def test_emit_registry_integrity_summary_writes_artifact(tmp_path):
    models_root = tmp_path / "models" / "elder_1"
    models_root.mkdir(parents=True, exist_ok=True)
    (models_root / "bedroom_versions.json").write_text("{}")

    class _DummyRegistry:
        backend_dir = str(tmp_path)

        def get_models_dir(self, elder_id):
            return Path(self.backend_dir) / "models" / elder_id

        def validate_and_repair_room_registry_state(self, elder_id, room_name):
            return {"valid": True, "repaired": True, "issues": ["synced_aliases"]}

    out = _emit_registry_integrity_summary(_DummyRegistry(), elder_ids=["elder_1"])
    assert out is not None
    payload = json.loads(Path(out).read_text())
    assert payload["elders_total"] == 1
    assert payload["rooms_total"] == 1
    assert payload["rooms_repaired"] == 1


def test_resolve_retrain_input_mode_defaults_on_invalid(monkeypatch):
    monkeypatch.setenv("RETRAIN_INPUT_MODE", "weird_mode")
    assert _resolve_retrain_input_mode() == "auto_aggregate"


def test_resolve_training_files_requires_incoming_only_for_pilot_profile(monkeypatch, tmp_path):
    incoming = tmp_path / "elder_1_train_5dec2025.xlsx"
    incoming.write_text("x")
    monkeypatch.setenv("RELEASE_GATE_EVIDENCE_PROFILE", "pilot_stage_a")
    monkeypatch.setenv("RETRAIN_INPUT_MODE", "auto_aggregate")

    with pytest.raises(ValueError, match="incoming_only"):
        run_daily_analysis._resolve_training_files_for_run("elder_1", [incoming])


def test_is_training_file_ignores_temp_and_dotfiles(tmp_path):
    temp_lock = tmp_path / "~$HK001_jessica_train_4dec2025.xlsx"
    temp_lock.write_text("x")
    dot_file = tmp_path / ".DS_Store"
    dot_file.write_text("x")
    real_file = tmp_path / "HK001_jessica_train_4dec2025.xlsx"
    real_file.write_text("x")

    assert run_daily_analysis._is_training_file(temp_lock) is False
    assert run_daily_analysis._is_training_file(dot_file) is False
    assert run_daily_analysis._is_training_file(real_file) is True


def test_validate_decision_trace_artifacts_detects_missing_paths(tmp_path):
    ok_versioned = tmp_path / "Bedroom_v1_decision_trace.json"
    ok_latest = tmp_path / "Bedroom_decision_trace.json"
    ok_versioned.write_text("{}")
    ok_latest.write_text("{}")
    metrics = [
        {
            "room": "Bedroom",
            "saved_version": 1,
            "decision_trace": {"versioned": str(ok_versioned), "latest": str(ok_latest)},
            "gate_pass": True,
        },
        {
            "room": "Kitchen",
            "saved_version": 2,
            "decision_trace": {"versioned": str(tmp_path / "missing_v2.json"), "latest": str(tmp_path / "missing.json")},
            "gate_pass": True,
        },
    ]
    passed, report = _validate_decision_trace_artifacts(metrics)
    assert passed is False
    assert report["reason"] == "missing_decision_trace_artifacts"
    assert "Kitchen" in report["failed_rooms"]


def test_global_gate_config_unavailable_fails_non_destructively(monkeypatch):
    def _raise():
        raise RuntimeError("missing release_gates.json")

    monkeypatch.setattr("run_daily_analysis.get_release_gates_config", _raise)

    gate_pass, report = _evaluate_global_gate(
        [{"room": "Bedroom", "gate_pass": True, "training_days": 10.0, "macro_f1": 0.9}]
    )

    assert gate_pass is False
    assert report["pass"] is False
    assert str(report["reason"]).startswith("gate_config_unavailable")
    assert _is_non_destructive_global_gate_failure(report) is True


def test_build_aggregate_training_set_dedupes_extension_variants(monkeypatch, tmp_path):
    incoming = tmp_path / "HK0011_jessica_train_4dec2025.xlsx"
    archived_dup = tmp_path / "HK0011_jessica_train_4dec2025.parquet"
    archived_other = tmp_path / "HK0011_jessica_train_5dec2025.parquet"
    incoming.write_text("x")
    archived_dup.write_text("y")
    archived_other.write_text("z")

    monkeypatch.setattr(
        "run_daily_analysis._collect_archived_training_files",
        lambda elder_id: [archived_dup, archived_other],
    )

    out = run_daily_analysis._build_aggregate_training_set("HK0011_jessica", [incoming])
    names = [p.name for p in out]

    # Keep incoming variant for duplicate identity, and keep distinct other day.
    assert "HK0011_jessica_train_4dec2025.xlsx" in names
    assert "HK0011_jessica_train_4dec2025.parquet" not in names
    assert "HK0011_jessica_train_5dec2025.parquet" in names


def test_build_aggregate_training_set_prefers_parquet_for_archived_duplicates(monkeypatch, tmp_path):
    archived_xlsx = tmp_path / "HK0011_jessica_train_6dec2025.xlsx"
    archived_parquet = tmp_path / "HK0011_jessica_train_6dec2025.parquet"
    archived_xlsx.write_text("x")
    archived_parquet.write_text("y")

    monkeypatch.setattr(
        "run_daily_analysis._collect_archived_training_files",
        lambda elder_id: [archived_xlsx, archived_parquet],
    )

    out = run_daily_analysis._build_aggregate_training_set("HK0011_jessica", [])
    assert [p.name for p in out] == ["HK0011_jessica_train_6dec2025.parquet"]


def test_elder_id_lineage_match_handles_numeric_suffix_drift():
    assert run_daily_analysis._elder_id_lineage_matches("HK001_jessica", "HK0011_jessica") is True
    assert run_daily_analysis._elder_id_lineage_matches("HK0011_jessica", "HK001_jessica") is True
    assert run_daily_analysis._elder_id_lineage_matches("HK001_jessica", "HK002_jessica") is False


def test_choose_canonical_elder_id_prefers_shorter_numeric_alias():
    out = run_daily_analysis._choose_canonical_elder_id(["HK0011_jessica", "HK001_jessica"])
    assert out == "HK001_jessica"


def test_dedupe_training_files_collapses_alias_prefix_duplicate_day(tmp_path):
    incoming = tmp_path / "HK0011_jessica_train_4dec2025.xlsx"
    archived_alias = tmp_path / "HK001_jessica_train_4dec2025.parquet"
    incoming.write_text("x")
    archived_alias.write_text("y")

    out = run_daily_analysis._dedupe_training_files([incoming, archived_alias], incoming_files=[incoming])
    assert [p.name for p in out] == ["HK0011_jessica_train_4dec2025.xlsx"]


def test_beta6_authority_sets_default_evidence_profile(monkeypatch):
    monkeypatch.delenv("RELEASE_GATE_EVIDENCE_PROFILE", raising=False)
    monkeypatch.setattr("run_daily_analysis._is_beta6_authority_enabled", lambda: True)

    run_daily_analysis._ensure_beta6_authority_evidence_profile_default()

    assert run_daily_analysis.os.getenv("RELEASE_GATE_EVIDENCE_PROFILE") == "pilot_stage_a"


def test_beta6_authority_default_evidence_profile_respects_explicit_env(monkeypatch):
    monkeypatch.setenv("RELEASE_GATE_EVIDENCE_PROFILE", "production")
    monkeypatch.setattr("run_daily_analysis._is_beta6_authority_enabled", lambda: True)

    run_daily_analysis._ensure_beta6_authority_evidence_profile_default()

    assert run_daily_analysis.os.getenv("RELEASE_GATE_EVIDENCE_PROFILE") == "production"


def test_validate_beta6_training_preflight_fails_when_label_policy_check_fails(monkeypatch, tmp_path):
    class _Issue:
        def __init__(self):
            self.code = "x"
            self.message = "y"
            self.context = {}

    class _Report:
        status = "fail"
        errors = [_Issue()]
        warnings = []

    monkeypatch.setattr("run_daily_analysis.validate_label_policy_consistency", lambda **kwargs: _Report())
    monkeypatch.setattr("run_daily_analysis.RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr("run_daily_analysis.ARCHIVE_DATA_DIR", tmp_path / "archive")
    agg = [tmp_path / "HK0011_jessica_train_4dec2025.xlsx"]
    agg[0].write_text("x")

    ok, report = run_daily_analysis._validate_beta6_training_preflight(
        elder_id="HK0011_jessica",
        aggregate_files=agg,
    )
    assert ok is False
    assert report["reason"] == "label_policy_consistency_failed"


def test_validate_beta6_training_preflight_passes_for_consistent_inputs(monkeypatch, tmp_path):
    class _Report:
        status = "pass"
        errors = []
        warnings = []

    raw = tmp_path / "raw"
    archive = tmp_path / "archive"
    raw.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    (raw / "HK0011_jessica_train_4dec2025.xlsx").write_text("x")
    (archive / "HK001_jessica_train_5dec2025.parquet").write_text("x")

    monkeypatch.setattr("run_daily_analysis.validate_label_policy_consistency", lambda **kwargs: _Report())
    monkeypatch.setattr("run_daily_analysis.RAW_DATA_DIR", raw)
    monkeypatch.setattr("run_daily_analysis.ARCHIVE_DATA_DIR", archive)
    agg = [raw / "HK0011_jessica_train_4dec2025.xlsx", archive / "HK001_jessica_train_5dec2025.parquet"]

    ok, report = run_daily_analysis._validate_beta6_training_preflight(
        elder_id="HK0011_jessica",
        aggregate_files=agg,
    )
    assert ok is True
    assert report["reason"] == "ok"


def test_build_legacy_training_timeline_results_uses_activity_labels(monkeypatch, tmp_path):
    f1 = tmp_path / "elder_1_train_4dec2025.parquet"
    f2 = tmp_path / "elder_1_train_5dec2025.parquet"
    f1.write_text("x")
    f2.write_text("y")

    df_first = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-12-04 00:00:10", "2025-12-04 00:00:20"]),
            "activity": ["sleep", "nap"],
            "motion": [0.1, 0.2],
        }
    )
    df_second = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-12-04 00:00:20", "2025-12-04 00:00:30"]),
            "activity": ["walk", "inactive"],
            "motion": [0.3, 0.4],
        }
    )

    def _load_stub(file_path, resample=True):
        if Path(file_path) == f1:
            return {"Bedroom": df_first}
        return {"Bedroom": df_second}

    monkeypatch.setattr("run_daily_analysis.load_sensor_data", _load_stub)

    out = run_daily_analysis._build_legacy_training_timeline_results([f1, f2])
    assert "Bedroom" in out
    room_df = out["Bedroom"].sort_values("timestamp").reset_index(drop=True)

    assert len(room_df) == 3
    assert list(room_df["predicted_activity"]) == ["sleep", "nap", "inactive"]
    assert list(room_df["confidence"]) == [1.0, 1.0, 1.0]


def test_is_env_enabled_parses_truthy_and_falsy(monkeypatch):
    monkeypatch.setenv("ENABLE_WALK_FORWARD_PROMOTION_GATE", "true")
    assert _is_env_enabled("ENABLE_WALK_FORWARD_PROMOTION_GATE", default=False) is True

    monkeypatch.setenv("ENABLE_WALK_FORWARD_PROMOTION_GATE", "0")
    assert _is_env_enabled("ENABLE_WALK_FORWARD_PROMOTION_GATE", default=True) is False


def test_train_files_skips_when_fingerprint_matches(monkeypatch, tmp_path):
    incoming = tmp_path / "HK0011_jessica_train_17dec2025.parquet"
    incoming.write_text("x")

    monkeypatch.setattr(
        "run_daily_analysis._resolve_training_files_for_run",
        lambda elder_id, incoming_files: ("incoming_only", [incoming], {"manifest_path": None}),
    )
    monkeypatch.setattr(
        "run_daily_analysis._compute_training_run_fingerprint",
        lambda **kwargs: {
            "fingerprint": "same-hash",
            "policy_hash": "policy-1",
            "code_version": "code-1",
        },
    )
    monkeypatch.setattr(
        "run_daily_analysis._load_last_run_fingerprint",
        lambda registry, elder_id: {
            "fingerprint": "same-hash",
            "policy_hash": "policy-1",
            "code_version": "code-1",
            "outcome": "success",
        },
    )
    monkeypatch.setattr("run_daily_analysis._snapshot_current_versions", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "run_daily_analysis.load_policy_from_env",
        lambda: type("P", (), {"reproducibility": type("R", (), {"skip_if_same_data_and_policy": True})()})(),
    )

    archived = []
    monkeypatch.setattr("run_daily_analysis.archive_file", lambda file_path, archive_dir: archived.append(file_path.name))

    class _DummyPipeline:
        def __init__(self, enable_denoising=True):
            self.was_trained = False

        def train_from_files(self, *args, **kwargs):
            raise AssertionError("train_from_files should not be called when fingerprint matches")

    class _DummyRegistry:
        def __init__(self, backend_dir):
            self._dir = tmp_path

        def get_models_dir(self, elder_id):
            return self._dir

    monkeypatch.setattr("run_daily_analysis.UnifiedPipeline", _DummyPipeline)
    monkeypatch.setattr("run_daily_analysis.ModelRegistry", _DummyRegistry)

    run_daily_analysis.train_files([incoming])
    assert archived == [incoming.name]


def test_backbone_alignment_gate_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_SHARED_BACKBONE_ADAPTERS", raising=False)
    monkeypatch.delenv("ENFORCE_BACKBONE_ALIGNMENT_GATE", raising=False)
    gate_pass, report = _evaluate_backbone_alignment_gate(
        [{"room": "Bedroom", "gate_pass": True, "model_identity": {"backbone_id": "b1"}}]
    )
    assert gate_pass is True
    assert report["reason"] == "disabled"


def test_backbone_alignment_gate_rejects_mismatch(monkeypatch):
    monkeypatch.setenv("ENABLE_SHARED_BACKBONE_ADAPTERS", "true")
    monkeypatch.setenv("ACTIVE_SHARED_BACKBONE_ID", "shared-bb-v2")

    gate_pass, report = _evaluate_backbone_alignment_gate(
        [
            {"room": "Bedroom", "gate_pass": True, "model_identity": {"backbone_id": "shared-bb-v2"}},
            {"room": "Kitchen", "gate_pass": True, "model_identity": {"backbone_id": "shared-bb-v1"}},
        ]
    )
    assert gate_pass is False
    assert report["reason"] == "room_backbone_mismatch"
    assert report["failed_rooms"] == ["Kitchen"]


def test_walk_forward_gate_rejects_no_regress_drop(monkeypatch):
    # Policy: room threshold passes, but no-regress fails.
    monkeypatch.setattr(
        "run_daily_analysis.get_release_gates_config",
        lambda: {
            "release_gates": {
                "rooms": {
                    "bedroom": {"schedule": [{"min_days": 2, "max_days": None, "min_value": 0.55}]}
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            }
        },
    )

    room_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="1h"),
            "activity": ["sleep"] * 20,
            "motion": [0.1] * 20,
        }
    )
    monkeypatch.setattr(
        "run_daily_analysis.load_room_training_dataframe",
        lambda **kwargs: (room_df, None),
    )

    # Candidate then champion load.
    monkeypatch.setattr(
        "run_daily_analysis._load_version_artifacts",
        lambda **kwargs: (
            {
                "model": MagicMock(),
                "scaler": MagicMock(),
                "label_encoder": MagicMock(classes_=["sleep", "inactive"]),
                "class_thresholds": {},
            },
            None,
        ),
    )

    # Candidate report then champion report.
    eval_calls = iter(
        [
            ({"summary": {"macro_f1_mean": 0.70}, "folds": [{"macro_f1": 0.70}]}, None),
            ({"summary": {"macro_f1_mean": 0.80}, "folds": [{"macro_f1": 0.80}]}, None),
        ]
    )
    monkeypatch.setattr("run_daily_analysis.evaluate_model_version", lambda **kwargs: next(eval_calls))

    registry = MagicMock()
    registry.get_current_version.return_value = 2

    pipeline = MagicMock()
    pipeline.room_config.calculate_seq_length.return_value = 5
    pipeline.platform = MagicMock()

    gate_pass, report = _evaluate_walk_forward_promotion_gate(
        pipeline=pipeline,
        registry=registry,
        elder_id="elder_1",
        metrics=[{"room": "Bedroom", "gate_pass": True, "training_days": 10.0}],
        previous_versions={"Bedroom": 1},
    )

    assert gate_pass is False
    assert "Bedroom" in report["failed_rooms"]
    room_report = next(r for r in report["room_reports"] if r["room"] == "Bedroom")
    assert any("no_regress_failed" in reason for reason in room_report["reasons"])


def test_walk_forward_gate_rejects_when_baseline_advantage_too_small(monkeypatch):
    monkeypatch.setattr(
        "run_daily_analysis.get_release_gates_config",
        lambda: {
            "release_gates": {
                "rooms": {
                    "bedroom": {"schedule": [{"min_days": 2, "max_days": None, "min_value": 0.55}]}
                },
                "no_regress": {"max_drop_from_champion": 0.05, "exempt_rooms": []},
            },
            "baseline_check": {
                "run_at_day": 22,
                "baseline_model": "xgboost",
                "required_transformer_advantage": 0.05,
            },
        },
    )

    room_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="1h"),
            "activity": ["sleep"] * 20,
            "motion": [0.1] * 20,
        }
    )
    monkeypatch.setattr(
        "run_daily_analysis.load_room_training_dataframe",
        lambda **kwargs: (room_df, None),
    )
    monkeypatch.setattr(
        "run_daily_analysis._load_version_artifacts",
        lambda **kwargs: (
            {
                "model": MagicMock(),
                "scaler": MagicMock(),
                "label_encoder": MagicMock(classes_=["sleep", "inactive"]),
                "class_thresholds": {},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "run_daily_analysis.evaluate_model_version",
        lambda **kwargs: ({"summary": {"macro_f1_mean": 0.70}, "folds": [{"macro_f1": 0.70}]}, None),
    )
    monkeypatch.setattr(
        "run_daily_analysis.evaluate_baseline_version",
        lambda **kwargs: (
            {"engine": "xgboost", "summary": {"macro_f1_mean": 0.68}, "folds": [{"macro_f1": 0.68}]},
            None,
        ),
    )

    registry = MagicMock()
    registry.get_current_version.return_value = 2

    pipeline = MagicMock()
    pipeline.room_config.calculate_seq_length.return_value = 5
    pipeline.platform = MagicMock()

    gate_pass, report = _evaluate_walk_forward_promotion_gate(
        pipeline=pipeline,
        registry=registry,
        elder_id="elder_1",
        metrics=[{"room": "Bedroom", "gate_pass": True, "training_days": 30.0}],
        previous_versions={"Bedroom": 0},
    )

    assert gate_pass is False
    assert "Bedroom" in report["failed_rooms"]
    room_report = next(r for r in report["room_reports"] if r["room"] == "Bedroom")
    assert any("baseline_advantage_failed" in reason for reason in room_report["reasons"])


def test_walk_forward_gate_config_unavailable_marks_promoted_rooms_failed(monkeypatch):
    def _raise_config_error():
        raise RuntimeError("config missing")

    monkeypatch.setattr("run_daily_analysis.get_release_gates_config", _raise_config_error)

    gate_pass, report = _evaluate_walk_forward_promotion_gate(
        pipeline=MagicMock(),
        registry=MagicMock(),
        elder_id="elder_1",
        metrics=[
            {"room": "Bedroom", "gate_pass": True, "training_days": 10.0},
            {"room": "Kitchen", "gate_pass": False, "training_days": 10.0},
        ],
        previous_versions={"Bedroom": 1, "Kitchen": 3},
    )

    assert gate_pass is False
    assert report["failed_rooms"] == ["Bedroom"]
    assert len(report["room_reports"]) == 1
    assert report["room_reports"][0]["room"] == "Bedroom"
    assert any("gate_config_unavailable" in r for r in report["room_reports"][0]["reasons"])


def test_walk_forward_gate_rejects_low_fold_minority_support_and_applies_room_overrides(monkeypatch):
    monkeypatch.setenv("WF_STEP_DAYS", "2")
    monkeypatch.setenv("WF_STEP_DAYS_BY_ROOM", "bedroom:1")
    monkeypatch.setenv("WF_DRIFT_THRESHOLD", "0.60")
    monkeypatch.setenv("WF_DRIFT_THRESHOLD_BY_ROOM", "bedroom:0.80")
    monkeypatch.setenv("WF_MIN_MINORITY_SUPPORT", "0")
    monkeypatch.setenv("WF_MIN_MINORITY_SUPPORT_BY_ROOM", "bedroom:5")

    monkeypatch.setattr(
        "run_daily_analysis.get_release_gates_config",
        lambda: {
            "release_gates": {
                "rooms": {"bedroom": {"schedule": [{"min_days": 2, "max_days": None, "min_value": 0.50}]}},
                "no_regress": {"max_drop_from_champion": 0.20, "exempt_rooms": []},
            }
        },
    )

    room_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=24, freq="1h"),
            "activity": ["sleep"] * 24,
            "motion": [0.1] * 24,
        }
    )
    monkeypatch.setattr("run_daily_analysis.load_room_training_dataframe", lambda **kwargs: (room_df, None))
    monkeypatch.setattr(
        "run_daily_analysis._load_version_artifacts",
        lambda **kwargs: (
            {
                "model": MagicMock(),
                "scaler": MagicMock(),
                "label_encoder": MagicMock(classes_=["sleep", "inactive"]),
                "class_thresholds": {},
            },
            None,
        ),
    )

    monkeypatch.setattr(
        "run_daily_analysis.evaluate_model_version",
        lambda **kwargs: (
            {
                "summary": {"macro_f1_mean": 0.75},
                "folds": [
                    {"macro_f1": 0.79, "minority_support": 2},
                    {"macro_f1": 0.81, "minority_support": 7},
                ],
            },
            None,
        ),
    )

    registry = MagicMock()
    registry.get_current_version.return_value = 2
    pipeline = MagicMock()
    pipeline.room_config.calculate_seq_length.return_value = 5
    pipeline.platform = MagicMock()

    gate_pass, report = _evaluate_walk_forward_promotion_gate(
        pipeline=pipeline,
        registry=registry,
        elder_id="elder_1",
        metrics=[{"room": "Bedroom", "gate_pass": True, "training_days": 10.0}],
        previous_versions={"Bedroom": 0},
    )

    assert gate_pass is False
    room_report = next(r for r in report["room_reports"] if r["room"] == "Bedroom")
    assert room_report["candidate_wf_config"]["lookback_days"] == report["config"]["lookback_days"]
    assert room_report["candidate_wf_config"]["step_days"] == 1
    assert room_report["candidate_wf_config"]["drift_threshold"] == 0.8
    assert room_report["candidate_wf_config"]["min_minority_support"] == 5
    assert room_report["candidate_transition_supported_folds"] == 0
    assert any("fold_support_failed:bedroom" in reason for reason in room_report["reasons"])


def test_walk_forward_gate_uses_policy_default_minority_support(monkeypatch):
    monkeypatch.delenv("WF_MIN_MINORITY_SUPPORT", raising=False)
    monkeypatch.delenv("WF_MIN_MINORITY_SUPPORT_BY_ROOM", raising=False)
    monkeypatch.setattr("run_daily_analysis.get_runtime_wf_min_minority_support_default", lambda: 3)
    monkeypatch.setattr(
        "run_daily_analysis.get_runtime_wf_min_minority_support_by_room",
        lambda: {"bedroom": 5},
    )

    monkeypatch.setattr(
        "run_daily_analysis.get_release_gates_config",
        lambda: {
            "release_gates": {
                "rooms": {"bedroom": {"schedule": [{"min_days": 2, "max_days": None, "min_value": 0.50}]}},
                "no_regress": {"max_drop_from_champion": 0.20, "exempt_rooms": []},
            }
        },
    )

    room_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=24, freq="1h"),
            "activity": ["sleep"] * 24,
            "motion": [0.1] * 24,
        }
    )
    monkeypatch.setattr("run_daily_analysis.load_room_training_dataframe", lambda **kwargs: (room_df, None))
    monkeypatch.setattr(
        "run_daily_analysis._load_version_artifacts",
        lambda **kwargs: (
            {
                "model": MagicMock(),
                "scaler": MagicMock(),
                "label_encoder": MagicMock(classes_=["sleep", "inactive"]),
                "class_thresholds": {},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "run_daily_analysis.evaluate_model_version",
        lambda **kwargs: (
            {
                "summary": {"macro_f1_mean": 0.75},
                "folds": [
                    {"macro_f1": 0.79, "minority_support": 2},
                    {"macro_f1": 0.81, "minority_support": 7},
                ],
            },
            None,
        ),
    )

    registry = MagicMock()
    registry.get_current_version.return_value = 2
    pipeline = MagicMock()
    pipeline.room_config.calculate_seq_length.return_value = 5
    pipeline.platform = MagicMock()

    gate_pass, report = _evaluate_walk_forward_promotion_gate(
        pipeline=pipeline,
        registry=registry,
        elder_id="elder_1",
        metrics=[{"room": "Bedroom", "gate_pass": True, "training_days": 10.0}],
        previous_versions={"Bedroom": 0},
    )

    assert gate_pass is False
    room_report = next(r for r in report["room_reports"] if r["room"] == "Bedroom")
    assert room_report["candidate_wf_config"]["min_minority_support"] == 5
    assert any("fold_support_failed:bedroom" in reason for reason in room_report["reasons"])


def test_walk_forward_gate_rejects_split_metric_failures(monkeypatch):
    monkeypatch.setenv("WF_MIN_STABILITY_ACCURACY", "0.99")
    monkeypatch.setenv("WF_MAX_STABILITY_LOW_FOLDS", "0")
    monkeypatch.setenv("WF_MIN_TRANSITION_F1", "0.80")
    monkeypatch.setenv("WF_MAX_TRANSITION_LOW_FOLDS", "0")

    monkeypatch.setattr(
        "run_daily_analysis.get_release_gates_config",
        lambda: {
            "release_gates": {
                "rooms": {"bedroom": {"schedule": [{"min_days": 2, "max_days": None, "min_value": 0.50}]}},
                "no_regress": {"max_drop_from_champion": 0.20, "exempt_rooms": []},
            }
        },
    )

    room_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=24, freq="1h"),
            "activity": ["sleep"] * 24,
            "motion": [0.1] * 24,
        }
    )
    monkeypatch.setattr("run_daily_analysis.load_room_training_dataframe", lambda **kwargs: (room_df, None))
    monkeypatch.setattr(
        "run_daily_analysis._load_version_artifacts",
        lambda **kwargs: (
            {
                "model": MagicMock(),
                "scaler": MagicMock(),
                "label_encoder": MagicMock(classes_=["sleep", "inactive"]),
                "class_thresholds": {},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "run_daily_analysis.evaluate_model_version",
        lambda **kwargs: (
            {
                "summary": {
                    "macro_f1_mean": 0.78,
                    "stability_accuracy_mean": 0.95,
                    "transition_macro_f1_mean": 0.55,
                },
                "folds": [
                    {
                        "macro_f1": 0.78,
                        "minority_support": 6,
                        "stability_accuracy": 0.95,
                        "transition_macro_f1": 0.55,
                        "transition_support": 12,
                    }
                ],
            },
            None,
        ),
    )

    registry = MagicMock()
    registry.get_current_version.return_value = 2
    pipeline = MagicMock()
    pipeline.room_config.calculate_seq_length.return_value = 5
    pipeline.platform = MagicMock()

    gate_pass, report = _evaluate_walk_forward_promotion_gate(
        pipeline=pipeline,
        registry=registry,
        elder_id="elder_1",
        metrics=[{"room": "Bedroom", "gate_pass": True, "training_days": 10.0}],
        previous_versions={"Bedroom": 0},
    )

    assert gate_pass is False
    room_report = next(r for r in report["room_reports"] if r["room"] == "Bedroom")
    assert room_report["candidate_wf_config"]["min_stability_accuracy"] == 0.99
    assert room_report["candidate_wf_config"]["min_transition_f1"] == 0.8
    assert any("stability_guard_failed:bedroom" in reason for reason in room_report["reasons"])
    assert any("transition_guard_failed:bedroom" in reason for reason in room_report["reasons"])


def test_walk_forward_gate_reports_no_folds_with_feasibility_details(monkeypatch):
    monkeypatch.setenv("WF_MIN_TRAIN_DAYS", "3")
    monkeypatch.setenv("WF_VALID_DAYS", "1")
    monkeypatch.setenv("WF_MAX_FOLDS", "8")
    monkeypatch.setenv("WF_STEP_DAYS", "1")

    monkeypatch.setattr(
        "run_daily_analysis.get_release_gates_config",
        lambda: {
            "release_gates": {
                "rooms": {"bedroom": {"schedule": [{"min_days": 2, "max_days": None, "min_value": 0.50}]}},
                "no_regress": {"max_drop_from_champion": 0.20, "exempt_rooms": []},
            }
        },
    )

    # 3 observed days only -> with min_train=3, valid=1 this is impossible (needs 4).
    room_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=72, freq="1h"),
            "activity": ["sleep"] * 72,
            "motion": [0.1] * 72,
        }
    )
    monkeypatch.setattr("run_daily_analysis.load_room_training_dataframe", lambda **kwargs: (room_df, None))
    monkeypatch.setattr(
        "run_daily_analysis._load_version_artifacts",
        lambda **kwargs: (
            {
                "model": MagicMock(),
                "scaler": MagicMock(),
                "label_encoder": MagicMock(classes_=["sleep", "inactive"]),
                "class_thresholds": {},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "run_daily_analysis.evaluate_model_version",
        lambda **kwargs: ({"summary": {"num_folds": 0}, "folds": []}, None),
    )

    registry = MagicMock()
    registry.get_current_version.return_value = 2
    pipeline = MagicMock()
    pipeline.room_config.calculate_seq_length.return_value = 5
    pipeline.platform = MagicMock()

    gate_pass, report = _evaluate_walk_forward_promotion_gate(
        pipeline=pipeline,
        registry=registry,
        elder_id="elder_1",
        metrics=[{"room": "Bedroom", "gate_pass": True, "training_days": 10.0}],
        previous_versions={"Bedroom": 0},
    )

    assert gate_pass is False
    room_report = next(r for r in report["room_reports"] if r["room"] == "Bedroom")
    assert any("wf_no_folds:bedroom:observed_days=3<required_days=4" in reason for reason in room_report["reasons"])
    assert room_report["candidate_wf_config"]["observed_days"] == 3
    assert room_report["candidate_wf_config"]["required_days"] == 4
    assert room_report["candidate_wf_config"]["expected_folds"] == 0


def test_walk_forward_gate_uses_pending_files_for_same_run_evaluation(monkeypatch, tmp_path):
    monkeypatch.setenv("WF_MIN_MINORITY_SUPPORT", "0")
    monkeypatch.delenv("WF_MIN_MINORITY_SUPPORT_BY_ROOM", raising=False)
    monkeypatch.setattr(
        "run_daily_analysis.get_release_gates_config",
        lambda: {
            "release_gates": {
                "rooms": {"bedroom": {"schedule": [{"min_days": 1, "max_days": None, "min_value": 0.50}]}},
                "no_regress": {"max_drop_from_champion": 0.20, "exempt_rooms": []},
            }
        },
    )

    room_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=96, freq="1h"),
            "activity": ["sleep"] * 96,
            "motion": [0.1] * 96,
        }
    )
    captured = {}

    def _load_df_stub(**kwargs):
        captured["include_files"] = kwargs.get("include_files", [])
        return room_df, None

    monkeypatch.setattr("run_daily_analysis.load_room_training_dataframe", _load_df_stub)
    monkeypatch.setattr(
        "run_daily_analysis._load_version_artifacts",
        lambda **kwargs: (
            {
                "model": MagicMock(),
                "scaler": MagicMock(),
                "label_encoder": MagicMock(classes_=["sleep", "inactive"]),
                "class_thresholds": {},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "run_daily_analysis.evaluate_model_version",
        lambda **kwargs: (
            {
                "summary": {
                    "num_folds": 1,
                    "macro_f1_mean": 0.80,
                    "accuracy_mean": 0.95,
                    "stability_accuracy_mean": 0.995,
                    "transition_macro_f1_mean": 0.85,
                },
                "folds": [
                    {
                        "macro_f1": 0.80,
                        "minority_support": 5,
                        "transition_support": 10,
                        "transition_macro_f1": 0.85,
                        "stability_accuracy": 0.995,
                    }
                ],
            },
            None,
        ),
    )

    registry = MagicMock()
    registry.get_current_version.return_value = 2
    pipeline = MagicMock()
    pipeline.room_config.calculate_seq_length.return_value = 5
    pipeline.platform = MagicMock()
    pending_file = Path(tmp_path / "elder_1_train_pending.xlsx")
    pending_file.write_text("pending")

    gate_pass, report = _evaluate_walk_forward_promotion_gate(
        pipeline=pipeline,
        registry=registry,
        elder_id="elder_1",
        metrics=[{"room": "Bedroom", "gate_pass": True, "training_days": 10.0}],
        previous_versions={"Bedroom": 0},
        pending_files=[pending_file],
    )

    assert gate_pass is True
    assert report["failed_rooms"] == []
    assert captured["include_files"] == [pending_file]


def test_train_files_rolls_back_promoted_rooms_when_wf_report_has_no_failed_rooms(monkeypatch, tmp_path):
    metrics = [
        {
            "room": "Bedroom",
            "gate_pass": True,
            "training_days": 10.0,
            "accuracy": 0.9,
            "epochs": 1,
            "macro_f1": 0.9,
        }
    ]
    rollback_args = {}

    class _PipelineStub:
        def __init__(self, enable_denoising=True):
            self.enable_denoising = enable_denoising
            self.room_config = MagicMock()
            self.room_config.calculate_seq_length.return_value = 5
            self.platform = MagicMock()

        def train_from_files(self, aggregate_files, elder_id, defer_promotion=False):
            return {}, metrics

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *args, **kwargs):
            return None

        def commit(self):
            return None

    class _DB:
        def get_connection(self):
            return _Conn()

    class _ADLServiceStub:
        def __init__(self):
            self.db = _DB()

    def _rollback_stub(registry, elder_id, room_names, previous_versions):
        rollback_args["room_names"] = list(room_names)
        return room_names, []

    monkeypatch.setattr("run_daily_analysis.get_elder_id_from_filename", lambda _: "elder_1")
    monkeypatch.setattr("run_daily_analysis._build_aggregate_training_set", lambda elder_id, files: files)
    monkeypatch.setattr("run_daily_analysis.UnifiedPipeline", _PipelineStub)
    monkeypatch.setattr("run_daily_analysis.ModelRegistry", lambda *_args, **_kwargs: MagicMock())
    monkeypatch.setattr("run_daily_analysis._snapshot_current_versions", lambda *_args, **_kwargs: {"Bedroom": 1})
    monkeypatch.setattr(
        "run_daily_analysis._is_env_enabled",
        lambda name, default=False: False if name == "ENABLE_PRE_PROMOTION_GATING" else True,
    )
    monkeypatch.setattr(
        "run_daily_analysis._evaluate_walk_forward_promotion_gate",
        lambda **kwargs: (False, {"pass": False, "reason": "gate_config_unavailable:test"}),
    )
    monkeypatch.setattr("run_daily_analysis._rollback_rooms_by_name", _rollback_stub)
    monkeypatch.setattr("run_daily_analysis._evaluate_global_gate", lambda *_args, **_kwargs: (True, {"pass": True}))
    monkeypatch.setattr("run_daily_analysis.ADLService", _ADLServiceStub)
    monkeypatch.setattr("run_daily_analysis.archive_file", lambda *_args, **_kwargs: None)

    run_daily_analysis.train_files([tmp_path / "elder_1_train.parquet"])

    assert rollback_args["room_names"] == ["Bedroom"]
    assert metrics[0]["gate_pass"] is False
    assert "walk_forward_gate_failed" in metrics[0]["gate_reasons"]


def test_train_files_incoming_only_mode_skips_auto_aggregate(monkeypatch, tmp_path):
    metrics = [
        {
            "room": "Bedroom",
            "gate_pass": True,
            "training_days": 10.0,
            "accuracy": 0.9,
            "epochs": 1,
            "macro_f1": 0.9,
            "saved_version": 1,
            "decision_trace": {},
        }
    ]
    captured = {}

    class _PipelineStub:
        def __init__(self, enable_denoising=True):
            self.enable_denoising = enable_denoising
            self.room_config = MagicMock()
            self.room_config.calculate_seq_length.return_value = 5
            self.platform = MagicMock()

        def train_from_files(self, aggregate_files, elder_id, defer_promotion=False):
            captured["aggregate_files"] = list(aggregate_files)
            return {}, metrics

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            captured["metadata"] = params[5]
            return None

        def commit(self):
            return None

    class _DB:
        def get_connection(self):
            return _Conn()

    class _ADLServiceStub:
        def __init__(self):
            self.db = _DB()

    f1 = tmp_path / "elder_1_train_5dec2025.xlsx"
    f2 = tmp_path / "elder_1_train_6dec2025.xlsx"
    f1.write_text("x")
    f2.write_text("y")

    monkeypatch.setenv("RETRAIN_INPUT_MODE", "incoming_only")
    monkeypatch.setattr("run_daily_analysis.get_elder_id_from_filename", lambda _: "elder_1")
    monkeypatch.setattr(
        "run_daily_analysis._build_aggregate_training_set",
        lambda elder_id, files: (_ for _ in ()).throw(RuntimeError("should not call auto aggregate")),
    )
    monkeypatch.setattr("run_daily_analysis.UnifiedPipeline", _PipelineStub)
    monkeypatch.setattr("run_daily_analysis.ModelRegistry", lambda *_args, **_kwargs: MagicMock())
    monkeypatch.setattr("run_daily_analysis._snapshot_current_versions", lambda *_args, **_kwargs: {"Bedroom": 1})
    monkeypatch.setattr("run_daily_analysis._is_env_enabled", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("run_daily_analysis._evaluate_global_gate", lambda *_args, **_kwargs: (True, {"pass": True}))
    monkeypatch.setattr("run_daily_analysis.ADLService", _ADLServiceStub)
    monkeypatch.setattr("run_daily_analysis.archive_file", lambda *_args, **_kwargs: None)

    run_daily_analysis.train_files([f1, f2])

    assert captured["aggregate_files"] == [f1, f2]
    assert '"mode": "incoming_only"' in captured["metadata"]


def test_train_files_promotes_deferred_candidates_after_wf_pass(monkeypatch, tmp_path):
    trace_v = tmp_path / "Bedroom_v7_decision_trace.json"
    trace_l = tmp_path / "Bedroom_decision_trace.json"
    trace_v.write_text("{}")
    trace_l.write_text("{}")
    metrics = [
        {
            "room": "Bedroom",
            "gate_pass": True,
            "saved_version": 7,
            "training_days": 10.0,
            "accuracy": 0.9,
            "epochs": 1,
            "macro_f1": 0.9,
            "promoted_to_latest": False,
            "decision_trace": {"versioned": str(trace_v), "latest": str(trace_l)},
        }
    ]
    registry = MagicMock()

    class _PipelineStub:
        def __init__(self, enable_denoising=True):
            self.enable_denoising = enable_denoising
            self.room_config = MagicMock()
            self.room_config.calculate_seq_length.return_value = 5
            self.platform = MagicMock()

        def train_from_files(self, aggregate_files, elder_id, defer_promotion=False):
            assert defer_promotion is True
            return {}, metrics

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *args, **kwargs):
            return None

        def commit(self):
            return None

    class _DB:
        def get_connection(self):
            return _Conn()

    class _ADLServiceStub:
        def __init__(self):
            self.db = _DB()

    monkeypatch.setattr("run_daily_analysis.get_elder_id_from_filename", lambda _: "elder_1")
    monkeypatch.setattr("run_daily_analysis._build_aggregate_training_set", lambda elder_id, files: files)
    monkeypatch.setattr("run_daily_analysis.UnifiedPipeline", _PipelineStub)
    monkeypatch.setattr("run_daily_analysis.ModelRegistry", lambda *_args, **_kwargs: registry)
    monkeypatch.setattr("run_daily_analysis._snapshot_current_versions", lambda *_args, **_kwargs: {"Bedroom": 1})
    monkeypatch.setattr("run_daily_analysis._is_env_enabled", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "run_daily_analysis._evaluate_walk_forward_promotion_gate",
        lambda **kwargs: (True, {"pass": True, "reason": "all_rooms_passed"}),
    )
    monkeypatch.setattr("run_daily_analysis._evaluate_global_gate", lambda *_args, **_kwargs: (True, {"pass": True}))
    monkeypatch.setattr("run_daily_analysis.ADLService", _ADLServiceStub)
    monkeypatch.setattr("run_daily_analysis.archive_file", lambda *_args, **_kwargs: None)

    run_daily_analysis.train_files([tmp_path / "elder_1_train.parquet"])

    registry.rollback_to_version.assert_called_once_with("elder_1", "Bedroom", 7)
    assert metrics[0]["promoted_to_latest"] is True


def test_train_files_marks_rejected_when_deferred_promotion_apply_fails(monkeypatch, tmp_path):
    trace_v = tmp_path / "Bedroom_v7_decision_trace.json"
    trace_l = tmp_path / "Bedroom_decision_trace.json"
    trace_v.write_text("{}")
    trace_l.write_text("{}")
    metrics = [
        {
            "room": "Bedroom",
            "gate_pass": True,
            "saved_version": 7,
            "training_days": 10.0,
            "accuracy": 0.9,
            "epochs": 1,
            "macro_f1": 0.9,
            "promoted_to_latest": False,
            "decision_trace": {"versioned": str(trace_v), "latest": str(trace_l)},
        }
    ]
    captured = {}
    registry = MagicMock()
    registry.rollback_to_version.return_value = False

    class _PipelineStub:
        def __init__(self, enable_denoising=True):
            self.enable_denoising = enable_denoising
            self.room_config = MagicMock()
            self.room_config.calculate_seq_length.return_value = 5
            self.platform = MagicMock()

        def train_from_files(self, aggregate_files, elder_id, defer_promotion=False):
            assert defer_promotion is True
            return {}, metrics

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            captured["status"] = params[4]
            captured["metadata"] = params[5]
            return None

        def commit(self):
            return None

    class _DB:
        def get_connection(self):
            return _Conn()

    class _ADLServiceStub:
        def __init__(self):
            self.db = _DB()

    monkeypatch.setattr("run_daily_analysis.get_elder_id_from_filename", lambda _: "elder_1")
    monkeypatch.setattr("run_daily_analysis._build_aggregate_training_set", lambda elder_id, files: files)
    monkeypatch.setattr("run_daily_analysis.UnifiedPipeline", _PipelineStub)
    monkeypatch.setattr("run_daily_analysis.ModelRegistry", lambda *_args, **_kwargs: registry)
    monkeypatch.setattr("run_daily_analysis._snapshot_current_versions", lambda *_args, **_kwargs: {"Bedroom": 1})
    monkeypatch.setattr("run_daily_analysis._is_env_enabled", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "run_daily_analysis._evaluate_walk_forward_promotion_gate",
        lambda **kwargs: (True, {"pass": True, "reason": "all_rooms_passed"}),
    )
    monkeypatch.setattr("run_daily_analysis._evaluate_global_gate", lambda *_args, **_kwargs: (True, {"pass": True}))
    monkeypatch.setattr("run_daily_analysis.ADLService", _ADLServiceStub)
    monkeypatch.setattr("run_daily_analysis.archive_file", lambda *_args, **_kwargs: None)

    run_daily_analysis.train_files([tmp_path / "elder_1_train.parquet"])

    assert captured["status"] == "rejected_by_walk_forward_gate"
    assert metrics[0]["gate_pass"] is False
    assert "promotion_apply_failed" in metrics[0]["gate_reasons"]


def test_train_files_rejects_when_decision_trace_missing(monkeypatch, tmp_path):
    metrics = [
        {
            "room": "Bedroom",
            "gate_pass": True,
            "saved_version": 3,
            "training_days": 10.0,
            "accuracy": 0.9,
            "epochs": 1,
            "macro_f1": 0.9,
            # missing decision_trace on purpose
        }
    ]
    captured = {}

    class _PipelineStub:
        def __init__(self, enable_denoising=True):
            self.enable_denoising = enable_denoising
            self.room_config = MagicMock()
            self.room_config.calculate_seq_length.return_value = 5
            self.platform = MagicMock()

        def train_from_files(self, aggregate_files, elder_id, defer_promotion=False):
            return {}, metrics

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            captured["status"] = params[4]
            captured["metadata"] = params[5]
            return None

        def commit(self):
            return None

    class _DB:
        def get_connection(self):
            return _Conn()

    class _ADLServiceStub:
        def __init__(self):
            self.db = _DB()

    monkeypatch.setattr("run_daily_analysis.get_elder_id_from_filename", lambda _: "elder_1")
    monkeypatch.setattr("run_daily_analysis._build_aggregate_training_set", lambda elder_id, files: files)
    monkeypatch.setattr("run_daily_analysis.UnifiedPipeline", _PipelineStub)
    monkeypatch.setattr("run_daily_analysis.ModelRegistry", lambda *_args, **_kwargs: MagicMock())
    monkeypatch.setattr("run_daily_analysis._snapshot_current_versions", lambda *_args, **_kwargs: {"Bedroom": 1})
    monkeypatch.setattr("run_daily_analysis._is_env_enabled", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("run_daily_analysis._evaluate_global_gate", lambda *_args, **_kwargs: (True, {"pass": True}))
    monkeypatch.setattr("run_daily_analysis.ADLService", _ADLServiceStub)
    monkeypatch.setattr("run_daily_analysis.archive_file", lambda *_args, **_kwargs: None)

    run_daily_analysis.train_files([tmp_path / "elder_1_train.parquet"])

    assert captured["status"] == "rejected_by_artifact_gate"
    assert metrics[0]["gate_pass"] is False
    assert "decision_trace_missing" in metrics[0]["gate_reasons"]
