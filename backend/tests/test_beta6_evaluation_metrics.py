from ml.beta6.evaluation.evaluation_metrics import (
    build_room_status,
    derive_room_confidence,
    resolve_float_threshold_with_source,
    resolve_int_threshold_with_source,
)


def _norm(name: str) -> str:
    return str(name or "").strip().lower().replace(" ", "").replace("_", "")


def test_resolve_threshold_source_precedence_room_override():
    env = {
        "WF_DRIFT_THRESHOLD": "0.71",
        "WF_DRIFT_THRESHOLD_BY_ROOM": "bedroom:0.83",
    }
    resolved = resolve_float_threshold_with_source(
        "WF_DRIFT_THRESHOLD",
        "Bedroom",
        fallback_default=0.60,
        explicit_value=0.75,
        normalize_room_name_fn=_norm,
        env=env,
    )
    assert resolved["value"] == 0.83
    assert resolved["source"] == "room override"
    assert resolved["source_file"] == "env:WF_*_BY_ROOM"
    assert resolved["editor_target"] == "admin.env.WF_DRIFT_THRESHOLD_BY_ROOM[bedroom]"


def test_resolve_threshold_source_precedence_env_override():
    env = {
        "WF_MIN_TRANSITION_F1": "0.88",
    }
    resolved = resolve_float_threshold_with_source(
        "WF_MIN_TRANSITION_F1",
        "Bedroom",
        fallback_default=0.80,
        explicit_value=0.81,
        normalize_room_name_fn=_norm,
        env=env,
    )
    assert resolved["value"] == 0.88
    assert resolved["source"] == "env override"
    assert resolved["source_file"] == "env:WF_*"
    assert resolved["editor_target"] == "admin.env.WF_MIN_TRANSITION_F1"


def test_resolve_threshold_source_policy_then_default():
    policy_resolved = resolve_int_threshold_with_source(
        "WF_MAX_TRANSITION_LOW_FOLDS",
        "Bedroom",
        fallback_default=1,
        explicit_value=2,
        normalize_room_name_fn=_norm,
        env={},
    )
    assert policy_resolved["value"] == 2
    assert policy_resolved["source"] == "policy"
    assert policy_resolved["source_file"] == "backend/config/release_gates.json"
    assert policy_resolved["editor_target"] == "admin.thresholds.walk_forward"

    default_resolved = resolve_int_threshold_with_source(
        "WF_MAX_TRANSITION_LOW_FOLDS",
        "Bedroom",
        fallback_default=1,
        explicit_value=None,
        normalize_room_name_fn=_norm,
        env={},
    )
    assert default_resolved["value"] == 1
    assert default_resolved["source"] == "default"
    assert default_resolved["source_file"] == "backend/ml/beta6/evaluation/evaluation_metrics.py"
    assert default_resolved["editor_target"] == "admin.thresholds.walk_forward_defaults"


def test_resolve_int_threshold_source_precedence_room_override():
    env = {
        "WF_MAX_TRANSITION_LOW_FOLDS": "3",
        "WF_MAX_TRANSITION_LOW_FOLDS_BY_ROOM": "bedroom:2",
    }
    resolved = resolve_int_threshold_with_source(
        "WF_MAX_TRANSITION_LOW_FOLDS",
        "Bedroom",
        fallback_default=1,
        explicit_value=4,
        normalize_room_name_fn=_norm,
        env=env,
    )
    assert resolved["value"] == 2
    assert resolved["source"] == "room override"
    assert resolved["source_file"] == "env:WF_*_BY_ROOM"
    assert resolved["editor_target"] == "admin.env.WF_MAX_TRANSITION_LOW_FOLDS_BY_ROOM[bedroom]"


def test_resolve_int_threshold_source_precedence_env_override():
    env = {
        "WF_MAX_TRANSITION_LOW_FOLDS": "5",
    }
    resolved = resolve_int_threshold_with_source(
        "WF_MAX_TRANSITION_LOW_FOLDS",
        "Bedroom",
        fallback_default=1,
        explicit_value=4,
        normalize_room_name_fn=_norm,
        env=env,
    )
    assert resolved["value"] == 5
    assert resolved["source"] == "env override"
    assert resolved["source_file"] == "env:WF_*"
    assert resolved["editor_target"] == "admin.env.WF_MAX_TRANSITION_LOW_FOLDS"


def test_build_room_status_and_confidence():
    payload = {
        "room": "bedroom",
        "last_check_time": "2026-02-27T12:00:00Z",
        "fallback_active": False,
        "metrics": {
            "candidate_macro_f1_mean": 0.72,
            "candidate_transition_macro_f1_mean": 0.90,
            "candidate_stability_accuracy_mean": 0.995,
        },
        "thresholds": {
            "min_transition_f1": {"value": 0.80},
            "min_stability_accuracy": {"value": 0.99},
            "max_transition_low_folds": {"value": 1},
        },
        "support": {
            "fold_count": 5,
            "candidate_low_transition_folds": 0,
        },
        "gate": {"pass": True},
    }
    status, reason, reason_code = build_room_status(
        payload,
        global_release_threshold=0.65,
        hours_since_fn=lambda _v: 1.0,
    )
    assert status == "healthy"
    assert reason_code == "healthy"
    assert "within thresholds" in reason
    assert derive_room_confidence(5, 0) == "High"


def test_build_room_status_insufficient_evidence_from_support_floor():
    payload = {
        "room": "kitchen",
        "last_check_time": "2026-02-27T12:00:00Z",
        "fallback_active": False,
        "metrics": {
            "candidate_macro_f1_mean": 0.75,
            "candidate_transition_macro_f1_mean": 0.85,
            "candidate_stability_accuracy_mean": 0.995,
        },
        "thresholds": {
            "min_transition_f1": {"value": 0.80},
            "min_stability_accuracy": {"value": 0.99},
            "max_transition_low_folds": {"value": 1},
            "min_minority_support": {"value": 5},
        },
        "support": {
            "fold_count": 5,
            "candidate_low_support_folds": 2,
            "candidate_low_transition_folds": 0,
        },
        "gate": {"pass": True, "reasons": []},
    }
    status, _, reason_code = build_room_status(
        payload,
        global_release_threshold=0.65,
        hours_since_fn=lambda _v: 1.0,
    )
    assert status == "action_needed"
    assert reason_code == "insufficient_evidence_support"


def test_build_room_status_insufficient_evidence_from_gate_reason():
    payload = {
        "room": "livingroom",
        "last_check_time": "2026-02-27T12:00:00Z",
        "fallback_active": False,
        "metrics": {
            "candidate_macro_f1_mean": 0.72,
            "candidate_transition_macro_f1_mean": 0.82,
            "candidate_stability_accuracy_mean": 0.995,
        },
        "thresholds": {
            "min_transition_f1": {"value": 0.80},
            "min_stability_accuracy": {"value": 0.99},
            "max_transition_low_folds": {"value": 1},
            "min_minority_support": {"value": 3},
        },
        "support": {
            "fold_count": 2,
            "candidate_low_support_folds": 0,
            "candidate_low_transition_folds": 0,
        },
        "gate": {"pass": False, "reasons": ["wf_no_folds:livingroom:observed_days=3<required_days=8"]},
    }
    status, _, reason_code = build_room_status(
        payload,
        global_release_threshold=0.65,
        hours_since_fn=lambda _v: 1.0,
    )
    assert status == "action_needed"
    assert reason_code == "insufficient_evidence_gate"
