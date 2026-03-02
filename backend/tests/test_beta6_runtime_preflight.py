import json
from pathlib import Path

from ml.beta6.serving.runtime_preflight import (
    validate_beta6_phase4_runtime_preflight,
    validate_beta6_phase4_runtime_preflight_cohort,
)


def _write_runtime_policy(
    *,
    root: Path,
    elder_id: str,
    enabled_rooms: list[str],
    duration_policy_path: Path,
) -> Path:
    path = root / elder_id / "_runtime" / "phase4_runtime_policy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "beta6.phase4.runtime_policy.v1",
        "room_runtime": {
            room: {
                "enable_phase4_runtime": True,
                "gate_pass": True,
                "reason_code": "pass",
            }
            for room in enabled_rooms
        },
        "enabled_rooms": list(enabled_rooms),
        "policy_paths": {
            "hmm_duration_policy_path": str(duration_policy_path),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_duration_policy(config_dir: Path) -> Path:
    path = config_dir / "beta6_duration_prior_policy.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "version: v1",
                "",
                "duration_priors:",
                "  default:",
                "    min_minutes: 1.0",
                "    target_minutes: 12.0",
                "    max_minutes: 90.0",
                "    penalty_weight: 0.8",
                "  by_label:",
                "    sleep:",
                "      min_minutes: 20.0",
                "      target_minutes: 420.0",
                "      max_minutes: 720.0",
                "      penalty_weight: 1.2",
                "",
                "transition:",
                "  switch_penalty: 0.15",
                "  impossible_transition_penalty: 1000000.0",
                "  self_transition_bias: 0.05",
                "  step_minutes: 1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_runtime_preflight_passes_for_valid_hmm_shadow_setup(tmp_path: Path):
    registry_root = tmp_path / "registry_v2"
    elder_id = "HK0011_jessica"
    duration_policy = _write_duration_policy(tmp_path / "config")
    _write_runtime_policy(
        root=registry_root,
        elder_id=elder_id,
        enabled_rooms=["bedroom"],
        duration_policy_path=duration_policy,
    )
    env = {
        "BETA6_PHASE4_RUNTIME_ENABLED": "true",
        "ENABLE_BETA6_HMM_RUNTIME": "true",
        "BETA6_PHASE4_RUNTIME_ROOMS": "bedroom",
        "BETA6_SEQUENCE_RUNTIME_MODE": "hmm",
    }

    ok, report = validate_beta6_phase4_runtime_preflight(
        elder_id=elder_id,
        target_cohort=[elder_id],
        registry_root=registry_root,
        env=env,
    )

    assert ok is True
    assert report["reason"] == "ok"
    assert report["enabled_rooms"] == ["bedroom"]


def test_runtime_preflight_fails_when_runtime_policy_missing(tmp_path: Path):
    ok, report = validate_beta6_phase4_runtime_preflight(
        elder_id="HK0011_jessica",
        target_cohort=["HK0011_jessica"],
        registry_root=tmp_path / "registry_v2",
        env={},
    )

    assert ok is False
    assert report["reason"] == "runtime_policy_missing"


def test_runtime_preflight_fails_when_duration_policy_missing(tmp_path: Path):
    registry_root = tmp_path / "registry_v2"
    elder_id = "HK0011_jessica"
    missing_policy = tmp_path / "config" / "beta6_duration_prior_policy.yaml"
    _write_runtime_policy(
        root=registry_root,
        elder_id=elder_id,
        enabled_rooms=["bedroom"],
        duration_policy_path=missing_policy,
    )

    ok, report = validate_beta6_phase4_runtime_preflight(
        elder_id=elder_id,
        target_cohort=[elder_id],
        registry_root=registry_root,
        env={},
    )

    assert ok is False
    assert report["reason"] == "duration_policy_missing"


def test_runtime_preflight_blocks_runtime_flags_outside_target_cohort(tmp_path: Path):
    registry_root = tmp_path / "registry_v2"
    elder_id = "HK0011_jessica"
    duration_policy = _write_duration_policy(tmp_path / "config")
    _write_runtime_policy(
        root=registry_root,
        elder_id=elder_id,
        enabled_rooms=["bedroom"],
        duration_policy_path=duration_policy,
    )
    env = {
        "BETA6_PHASE4_RUNTIME_ENABLED": "true",
        "ENABLE_BETA6_HMM_RUNTIME": "true",
    }

    ok, report = validate_beta6_phase4_runtime_preflight(
        elder_id=elder_id,
        target_cohort=["HK0099_other"],
        registry_root=registry_root,
        env=env,
    )

    assert ok is False
    assert report["reason"] == "runtime_flags_not_cohort_scoped"


def test_runtime_preflight_accepts_lineage_equivalent_target_cohort(tmp_path: Path):
    registry_root = tmp_path / "registry_v2"
    elder_id = "HK001_jessica"
    duration_policy = _write_duration_policy(tmp_path / "config")
    _write_runtime_policy(
        root=registry_root,
        elder_id=elder_id,
        enabled_rooms=["bedroom"],
        duration_policy_path=duration_policy,
    )
    env = {
        "BETA6_PHASE4_RUNTIME_ENABLED": "true",
        "ENABLE_BETA6_HMM_RUNTIME": "true",
    }

    ok, report = validate_beta6_phase4_runtime_preflight(
        elder_id=elder_id,
        target_cohort=["HK0011_jessica"],
        registry_root=registry_root,
        env=env,
    )

    assert ok is True
    assert report["reason"] == "ok"


def test_runtime_preflight_fails_when_room_scope_not_enabled(tmp_path: Path):
    registry_root = tmp_path / "registry_v2"
    elder_id = "HK0011_jessica"
    duration_policy = _write_duration_policy(tmp_path / "config")
    _write_runtime_policy(
        root=registry_root,
        elder_id=elder_id,
        enabled_rooms=["bedroom"],
        duration_policy_path=duration_policy,
    )
    env = {
        "BETA6_PHASE4_RUNTIME_ROOMS": "livingroom",
    }

    ok, report = validate_beta6_phase4_runtime_preflight(
        elder_id=elder_id,
        target_cohort=[elder_id],
        registry_root=registry_root,
        env=env,
    )

    assert ok is False
    assert report["reason"] == "runtime_room_scope_mismatch"


def test_runtime_preflight_blocks_crf_without_canary_override(tmp_path: Path):
    registry_root = tmp_path / "registry_v2"
    elder_id = "HK0011_jessica"
    duration_policy = _write_duration_policy(tmp_path / "config")
    _write_runtime_policy(
        root=registry_root,
        elder_id=elder_id,
        enabled_rooms=["bedroom"],
        duration_policy_path=duration_policy,
    )
    env = {
        "BETA6_SEQUENCE_RUNTIME_MODE": "crf",
    }

    ok, report = validate_beta6_phase4_runtime_preflight(
        elder_id=elder_id,
        target_cohort=[elder_id],
        registry_root=registry_root,
        env=env,
    )
    assert ok is False
    assert report["reason"] == "crf_runtime_not_allowed"

    ok_canary, report_canary = validate_beta6_phase4_runtime_preflight(
        elder_id=elder_id,
        target_cohort=[elder_id],
        registry_root=registry_root,
        env=env,
        allow_crf_canary=True,
    )
    assert ok_canary is True
    assert report_canary["reason"] == "ok"


def test_runtime_preflight_cohort_fails_when_target_cohort_not_fully_checked(tmp_path: Path):
    registry_root = tmp_path / "registry_v2"
    duration_policy = _write_duration_policy(tmp_path / "config")
    _write_runtime_policy(
        root=registry_root,
        elder_id="HK0011_jessica",
        enabled_rooms=["bedroom"],
        duration_policy_path=duration_policy,
    )

    ok, report = validate_beta6_phase4_runtime_preflight_cohort(
        elder_ids=["HK0011_jessica"],
        target_cohort=["HK0011_jessica", "HK0012_mary"],
        registry_root=registry_root,
        env={},
    )

    assert ok is False
    assert report["reason"] == "target_cohort_not_fully_checked"


def test_runtime_preflight_cohort_accepts_lineage_equivalent_target_coverage(tmp_path: Path):
    registry_root = tmp_path / "registry_v2"
    duration_policy = _write_duration_policy(tmp_path / "config")
    _write_runtime_policy(
        root=registry_root,
        elder_id="HK001_jessica",
        enabled_rooms=["bedroom"],
        duration_policy_path=duration_policy,
    )

    ok, report = validate_beta6_phase4_runtime_preflight_cohort(
        elder_ids=["HK001_jessica"],
        target_cohort=["HK0011_jessica"],
        registry_root=registry_root,
        env={},
    )

    assert ok is True
    assert report["reason"] == "ok"
