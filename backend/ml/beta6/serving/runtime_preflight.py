"""Fail-closed preflight checks for Beta 6 Phase-4 runtime activation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from ml.beta6.sequence import load_duration_prior_policy
from utils.room_utils import normalize_room_name
from utils.elder_id_utils import apply_canonical_alias_map, elder_id_lineage_matches


def _env_enabled(var_name: str, default: bool = False, *, env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    raw = source.get(var_name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _env_path(var_name: str, *, env: Mapping[str, str] | None = None) -> str | None:
    source = os.environ if env is None else env
    raw = source.get(var_name)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    tokens: list[str] = []
    for raw in str(value).split(","):
        item = str(raw).strip()
        if item:
            tokens.append(item)
    return tokens


def _runtime_registry_root(
    *,
    registry_root: str | Path | None,
    env: Mapping[str, str] | None = None,
) -> Path:
    if registry_root is not None:
        return Path(registry_root).expanduser().resolve()
    root_from_env = _env_path("BETA6_REGISTRY_V2_ROOT", env=env)
    if root_from_env:
        return Path(root_from_env).expanduser().resolve()
    return (Path(__file__).resolve().parents[3] / "models_beta6_registry_v2").resolve()


def _runtime_policy_path(
    *,
    elder_id: str,
    registry_root: Path,
) -> Path:
    safe_elder = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(elder_id or "").strip())
    return registry_root / safe_elder / "_runtime" / "phase4_runtime_policy.json"


def _default_duration_policy_path() -> Path:
    return (Path(__file__).resolve().parents[3] / "config" / "beta6_duration_prior_policy.yaml").resolve()


def _resolve_target_cohort(
    *,
    target_cohort: Sequence[str] | None,
    env: Mapping[str, str] | None = None,
) -> set[str]:
    if target_cohort is not None:
        return {str(item).strip() for item in target_cohort if str(item).strip()}
    raw = _env_path("BETA6_RUNTIME_TARGET_COHORT", env=env)
    return {token for token in _parse_csv(raw)}


def _canonicalize_elder_id(value: str) -> str:
    txt = str(value or "").strip()
    if not txt:
        return ""
    return str(apply_canonical_alias_map(txt)).strip()


def _elder_id_matches_target(elder_id: str, target_elder_id: str) -> bool:
    elder = _canonicalize_elder_id(elder_id)
    target = _canonicalize_elder_id(target_elder_id)
    if not elder or not target:
        return False
    if elder == target:
        return True
    return bool(elder_id_lineage_matches(elder, target))


def _cohort_contains_elder(target_set: Sequence[str], elder_id: str) -> bool:
    return any(_elder_id_matches_target(elder_id, target) for target in target_set)


def _target_cohort_missing(requested_elders: Sequence[str], target_set: Sequence[str]) -> list[str]:
    missing: list[str] = []
    for target in target_set:
        if not _cohort_contains_elder(requested_elders, str(target)):
            missing.append(str(target))
    return sorted(missing)


def _resolve_enabled_rooms(runtime_policy: Mapping[str, Any]) -> list[str]:
    room_runtime = runtime_policy.get("room_runtime")
    room_runtime = room_runtime if isinstance(room_runtime, Mapping) else {}
    enabled_rooms: set[str] = set()
    for room_raw, entry in room_runtime.items():
        entry = entry if isinstance(entry, Mapping) else {}
        if bool(entry.get("enable_phase4_runtime", False)):
            enabled_rooms.add(normalize_room_name(str(room_raw)))
    for item in runtime_policy.get("enabled_rooms") or []:
        room = normalize_room_name(str(item))
        if room:
            enabled_rooms.add(room)
    return sorted(enabled_rooms)


def _resolve_duration_policy_path(
    *,
    runtime_policy: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
) -> Path:
    env_path = _env_path("BETA6_HMM_DURATION_POLICY_PATH", env=env)
    if env_path:
        return Path(env_path).expanduser().resolve()

    policy_paths = runtime_policy.get("policy_paths")
    policy_paths = policy_paths if isinstance(policy_paths, Mapping) else {}
    artifact_path = str(policy_paths.get("hmm_duration_policy_path") or "").strip()
    if artifact_path:
        return Path(artifact_path).expanduser().resolve()
    return _default_duration_policy_path()


def _resolve_runtime_mode(*, env: Mapping[str, str] | None = None) -> str:
    raw_mode = _env_path("BETA6_SEQUENCE_RUNTIME_MODE", env=env)
    if raw_mode:
        return str(raw_mode).strip().lower()
    if _env_enabled("ENABLE_BETA6_CRF_RUNTIME", default=False, env=env):
        return "crf"
    return "hmm"


def validate_beta6_phase4_runtime_preflight(
    *,
    elder_id: str,
    target_cohort: Sequence[str] | None = None,
    registry_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    require_target_cohort_for_runtime_flags: bool = True,
    allow_crf_canary: bool = False,
) -> tuple[bool, dict]:
    """
    Validate Beta 6 runtime artifact and flag safety for one elder.

    Returns
    -------
    (ok, report):
      ok=True only when all checks pass.
      report is JSON-serializable and includes per-check results.
    """
    canonical_elder_id = _canonicalize_elder_id(elder_id)
    report: dict[str, Any] = {
        "elder_id": str(elder_id),
        "elder_id_canonical": canonical_elder_id or str(elder_id),
        "checks": [],
    }

    def _push(check_name: str, passed: bool, details: Mapping[str, Any] | None = None) -> None:
        entry: dict[str, Any] = {"check": str(check_name), "pass": bool(passed)}
        if details:
            entry["details"] = dict(details)
        report["checks"].append(entry)

    root = _runtime_registry_root(registry_root=registry_root, env=env)
    report["registry_root"] = str(root)
    runtime_policy_path = _runtime_policy_path(elder_id=elder_id, registry_root=root)
    report["runtime_policy_path"] = str(runtime_policy_path)

    if not runtime_policy_path.exists():
        _push("runtime_policy_artifact_exists", False, {"path": str(runtime_policy_path)})
        report["reason"] = "runtime_policy_missing"
        return False, report
    _push("runtime_policy_artifact_exists", True, {"path": str(runtime_policy_path)})

    try:
        payload = json.loads(runtime_policy_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _push(
            "runtime_policy_json_parse",
            False,
            {"path": str(runtime_policy_path), "error": f"{type(exc).__name__}: {exc}"},
        )
        report["reason"] = "runtime_policy_unreadable"
        return False, report
    if not isinstance(payload, Mapping):
        _push(
            "runtime_policy_json_parse",
            False,
            {"path": str(runtime_policy_path), "error": "policy payload must be a JSON object"},
        )
        report["reason"] = "runtime_policy_invalid_payload"
        return False, report
    _push("runtime_policy_json_parse", True, {"path": str(runtime_policy_path)})

    schema_version = str(payload.get("schema_version") or "").strip()
    schema_ok = schema_version == "beta6.phase4.runtime_policy.v1"
    _push(
        "runtime_policy_schema_version",
        schema_ok,
        {"schema_version": schema_version or None},
    )
    if not schema_ok:
        report["reason"] = "runtime_policy_schema_mismatch"
        return False, report

    enabled_rooms = _resolve_enabled_rooms(payload)
    report["enabled_rooms"] = list(enabled_rooms)
    has_enabled_rooms = len(enabled_rooms) > 0
    _push(
        "runtime_policy_enabled_rooms_non_empty",
        has_enabled_rooms,
        {"enabled_rooms": list(enabled_rooms)},
    )
    if not has_enabled_rooms:
        report["reason"] = "runtime_policy_has_no_enabled_rooms"
        return False, report

    duration_policy_path = _resolve_duration_policy_path(runtime_policy=payload, env=env)
    report["duration_policy_path"] = str(duration_policy_path)
    if not duration_policy_path.exists():
        _push(
            "duration_prior_policy_exists",
            False,
            {"path": str(duration_policy_path)},
        )
        report["reason"] = "duration_policy_missing"
        return False, report
    _push("duration_prior_policy_exists", True, {"path": str(duration_policy_path)})

    try:
        load_duration_prior_policy(duration_policy_path)
    except Exception as exc:
        _push(
            "duration_prior_policy_loadable",
            False,
            {"path": str(duration_policy_path), "error": f"{type(exc).__name__}: {exc}"},
        )
        report["reason"] = "duration_policy_unloadable"
        return False, report
    _push("duration_prior_policy_loadable", True, {"path": str(duration_policy_path)})

    runtime_mode = _resolve_runtime_mode(env=env)
    runtime_mode_ok = runtime_mode in {"hmm", "crf"}
    _push("runtime_mode_supported", runtime_mode_ok, {"mode": runtime_mode})
    if not runtime_mode_ok:
        report["reason"] = "runtime_mode_invalid"
        return False, report

    crf_enabled = runtime_mode == "crf" or _env_enabled("ENABLE_BETA6_CRF_RUNTIME", default=False, env=env)
    crf_guard_ok = (not crf_enabled) or bool(allow_crf_canary)
    _push(
        "runtime_mode_crf_canary_guard",
        crf_guard_ok,
        {
            "mode": runtime_mode,
            "enable_beta6_crf_runtime": _env_enabled("ENABLE_BETA6_CRF_RUNTIME", default=False, env=env),
            "allow_crf_canary": bool(allow_crf_canary),
        },
    )
    if not crf_guard_ok:
        report["reason"] = "crf_runtime_not_allowed"
        return False, report

    target_set = _resolve_target_cohort(target_cohort=target_cohort, env=env)
    report["target_cohort"] = sorted(target_set)
    report["target_cohort_canonical"] = sorted(
        {_canonicalize_elder_id(item) or str(item).strip() for item in target_set}
    )
    runtime_flags_enabled = any(
        [
            _env_enabled("BETA6_PHASE4_RUNTIME_ENABLED", default=False, env=env),
            _env_enabled("ENABLE_BETA6_HMM_RUNTIME", default=False, env=env),
            _env_enabled("ENABLE_BETA6_UNKNOWN_ABSTAIN_RUNTIME", default=False, env=env),
        ]
    )
    cohort_scope_ok = True
    cohort_scope_details = {
        "runtime_flags_enabled": bool(runtime_flags_enabled),
        "require_target_cohort_for_runtime_flags": bool(require_target_cohort_for_runtime_flags),
    }
    if runtime_flags_enabled:
        if require_target_cohort_for_runtime_flags and not target_set:
            cohort_scope_ok = False
            cohort_scope_details["error"] = "target cohort is required when runtime flags are enabled"
        elif target_set and not _cohort_contains_elder(target_set, str(elder_id)):
            cohort_scope_ok = False
            cohort_scope_details["error"] = "elder not in target runtime cohort"
    _push("runtime_flags_cohort_scoped", cohort_scope_ok, cohort_scope_details)
    if not cohort_scope_ok:
        report["reason"] = "runtime_flags_not_cohort_scoped"
        return False, report

    room_scope_raw = _env_path("BETA6_PHASE4_RUNTIME_ROOMS", env=env)
    room_scope = {normalize_room_name(room) for room in _parse_csv(room_scope_raw)}
    room_scope_ok = True
    room_scope_details = {
        "runtime_room_scope": sorted(room_scope),
        "enabled_rooms": list(enabled_rooms),
    }
    if room_scope and not room_scope.issubset(set(enabled_rooms)):
        room_scope_ok = False
        room_scope_details["unknown_scope_rooms"] = sorted(room_scope - set(enabled_rooms))
    _push("runtime_room_scope_subset_of_enabled_rooms", room_scope_ok, room_scope_details)
    if not room_scope_ok:
        report["reason"] = "runtime_room_scope_mismatch"
        return False, report

    report["reason"] = "ok"
    return True, report


def validate_beta6_phase4_runtime_preflight_cohort(
    *,
    elder_ids: Sequence[str],
    target_cohort: Sequence[str] | None = None,
    registry_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    require_target_cohort_for_runtime_flags: bool = True,
    allow_crf_canary: bool = False,
) -> tuple[bool, dict]:
    """
    Validate runtime preflight for all elders in a shadow/canary cohort.
    """
    requested_elders = [str(elder).strip() for elder in elder_ids if str(elder).strip()]
    report: dict[str, Any] = {
        "requested_elders": list(requested_elders),
        "checks": [],
    }
    if not requested_elders:
        report["reason"] = "empty_elder_ids"
        return False, report

    cohort_set = _resolve_target_cohort(target_cohort=target_cohort, env=env)
    report["target_cohort"] = sorted(cohort_set)
    reports: dict[str, Any] = {}
    failures: list[str] = []
    for elder_id in requested_elders:
        ok, elder_report = validate_beta6_phase4_runtime_preflight(
            elder_id=elder_id,
            target_cohort=sorted(cohort_set) if cohort_set else None,
            registry_root=registry_root,
            env=env,
            require_target_cohort_for_runtime_flags=require_target_cohort_for_runtime_flags,
            allow_crf_canary=allow_crf_canary,
        )
        reports[elder_id] = elder_report
        if not ok:
            failures.append(elder_id)

    missing_target_elders = _target_cohort_missing(requested_elders, sorted(cohort_set))
    coverage_ok = len(missing_target_elders) == 0
    report["checks"].append(
        {
            "check": "target_cohort_coverage",
            "pass": bool(coverage_ok),
            "details": {
                "missing_target_elders": missing_target_elders,
            },
        }
    )
    if not coverage_ok:
        report["reason"] = "target_cohort_not_fully_checked"
        report["elder_reports"] = reports
        report["failed_elders"] = failures
        return False, report

    report["elder_reports"] = reports
    report["failed_elders"] = failures
    report["reason"] = "ok" if not failures else "elder_preflight_failed"
    return len(failures) == 0, report


__all__ = [
    "validate_beta6_phase4_runtime_preflight",
    "validate_beta6_phase4_runtime_preflight_cohort",
]
