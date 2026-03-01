import copy

import pytest

from ml.beta6.contracts.run_spec import (
    HASH_POLICY_VERSION,
    RUN_SPEC_VERSION,
    RunSpec,
    schema_metadata,
)

EXPECTED_RUN_SPEC_V1_SCHEMA_HASH = (
    "sha256:677b27b537d9fcff202a84659fc6031101597dcd3ae3b2bf1f21e088e33208f6"
)


@pytest.fixture
def valid_run_spec_payload():
    return {
        "run_spec_version": RUN_SPEC_VERSION,
        "run_id": "2026-02-25T12:00:00Z_elder_HK001",
        "elder_id": "HK001",
        "mode": "auto_aggregate",
        "data": {
            "manifest_paths": ["/tmp/a.csv", "/tmp/b.csv"],
            "time_zone": "Asia/Hong_Kong",
            "max_ffill_gap_seconds": 60,
            "duplicate_resolution_policy": "majority_vote_latest_tiebreak",
        },
        "features": {
            "sequence_window_seconds": 600,
            "stride_seconds": 10,
            "feature_version": "feature_schema_v3",
        },
        "training": {
            "architecture_family": "transformer",
            "random_seed": 42,
            "profile": "production",
            "optimizer": "adam",
            "learning_rate": 1.0e-4,
            "epochs": 20,
        },
        "evaluation": {
            "walk_forward": {
                "lookback_days": 90,
                "min_train_days": 7,
                "valid_days": 1,
                "step_days": 1,
                "max_folds": 30,
            }
        },
        "gating": {
            "room_policy_ref": "release_policy_v5",
            "run_policy_ref": "global_gate_policy_v3",
        },
    }


def test_run_spec_parses_valid_payload(valid_run_spec_payload):
    run_spec = RunSpec.from_dict(valid_run_spec_payload)
    assert run_spec.run_spec_version == RUN_SPEC_VERSION
    assert run_spec.data.manifest_paths == ("/tmp/a.csv", "/tmp/b.csv")
    assert run_spec.features.sequence_window_seconds == 600
    assert run_spec.training.learning_rate == pytest.approx(1.0e-4, abs=1e-12)
    assert run_spec.evaluation.walk_forward.max_folds == 30


def test_run_spec_rejects_missing_top_level_field(valid_run_spec_payload):
    payload = copy.deepcopy(valid_run_spec_payload)
    del payload["gating"]
    with pytest.raises(ValueError, match="missing"):
        RunSpec.from_dict(payload)


def test_run_spec_rejects_unknown_top_level_field(valid_run_spec_payload):
    payload = copy.deepcopy(valid_run_spec_payload)
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        RunSpec.from_dict(payload)


def test_run_spec_rejects_unknown_nested_field(valid_run_spec_payload):
    payload = copy.deepcopy(valid_run_spec_payload)
    payload["data"]["extra"] = "nope"
    with pytest.raises(ValueError, match="unknown"):
        RunSpec.from_dict(payload)


def test_run_spec_rejects_stride_greater_than_window(valid_run_spec_payload):
    payload = copy.deepcopy(valid_run_spec_payload)
    payload["features"]["stride_seconds"] = payload["features"]["sequence_window_seconds"] + 1
    with pytest.raises(ValueError, match="stride_seconds"):
        RunSpec.from_dict(payload)


def test_run_spec_rejects_wrong_version(valid_run_spec_payload):
    payload = copy.deepcopy(valid_run_spec_payload)
    payload["run_spec_version"] = "v0"
    with pytest.raises(ValueError, match="Unsupported run_spec_version"):
        RunSpec.from_dict(payload)


def test_run_spec_hash_ignores_run_id(valid_run_spec_payload):
    payload_a = copy.deepcopy(valid_run_spec_payload)
    payload_b = copy.deepcopy(valid_run_spec_payload)
    payload_b["run_id"] = "2026-02-26T00:00:00Z_elder_HK001"

    hash_a = RunSpec.from_dict(payload_a).run_spec_hash()
    hash_b = RunSpec.from_dict(payload_b).run_spec_hash()
    assert hash_a == hash_b


def test_run_spec_hash_is_order_invariant(valid_run_spec_payload):
    payload_a = copy.deepcopy(valid_run_spec_payload)
    payload_b = {
        "gating": valid_run_spec_payload["gating"],
        "evaluation": valid_run_spec_payload["evaluation"],
        "training": valid_run_spec_payload["training"],
        "features": valid_run_spec_payload["features"],
        "data": valid_run_spec_payload["data"],
        "mode": valid_run_spec_payload["mode"],
        "elder_id": valid_run_spec_payload["elder_id"],
        "run_id": valid_run_spec_payload["run_id"],
        "run_spec_version": valid_run_spec_payload["run_spec_version"],
    }

    hash_a = RunSpec.from_dict(payload_a).run_spec_hash()
    hash_b = RunSpec.from_dict(payload_b).run_spec_hash()
    assert hash_a == hash_b


def test_run_spec_round_trip_preserves_payload_and_hash(valid_run_spec_payload):
    run_spec = RunSpec.from_dict(valid_run_spec_payload)
    round_tripped = RunSpec.from_dict(run_spec.to_dict())
    assert round_tripped.to_dict() == run_spec.to_dict()
    assert round_tripped.run_spec_hash() == run_spec.run_spec_hash()


def test_schema_metadata_is_published_and_versioned():
    metadata = schema_metadata()
    assert metadata["schema_name"] == "RunSpec"
    assert metadata["schema_version"] == RUN_SPEC_VERSION
    assert metadata["hash_policy_version"] == HASH_POLICY_VERSION
    assert metadata["schema_hash"] == EXPECTED_RUN_SPEC_V1_SCHEMA_HASH
