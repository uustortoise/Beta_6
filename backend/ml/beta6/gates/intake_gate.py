"""Phase 0 intake-gate contract for Beta 6 label-pack onboarding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


INTAKE_ARTIFACT_VERSION = "v1"
APPROVED_STATUS = "approved"
REJECTED_STATUS = "rejected"

_REQUIRED_TOP_LEVEL = (
    "artifact_version",
    "generated_at",
    "status",
    "pack",
    "steps",
    "reports",
    "gate",
)

_REQUIRED_PACK_FIELDS = (
    "candidate_dir",
    "baseline_dir",
    "elder_id",
    "min_day",
    "max_day",
    "smoke_day",
    "seed",
)

_REQUIRED_STEP_NAMES = ("validate", "diff", "smoke")
_VALID_STEP_STATUSES = {"pass", "fail", "skipped"}
_REQUIRED_REPORT_FIELDS = (
    "validate_json",
    "diff_json",
    "diff_csv",
    "smoke_json",
)
_REQUIRED_GATE_FIELDS = ("approved", "blocking_reasons")
_PRETRAIN_REQUIRED_TOP_LEVEL = (
    "manifest_version",
    "generated_at",
    "policy",
    "corpus_roots",
    "entries",
    "duplicates",
    "quarantine",
    "violations",
    "summary",
    "gate",
    "stats",
    "fingerprint",
)
_PRETRAIN_REQUIRED_SUMMARY_FIELDS = (
    "user_tags",
    "date_tags",
    "per_room_per_date_label_counts",
)
_PRETRAIN_REQUIRED_ENTRY_FIELDS = (
    "path",
    "extension",
    "content_hash",
    "row_count",
    "feature_count",
    "missing_ratio",
    "status",
    "source_tags",
    "label_summary",
)
_PRETRAIN_REQUIRED_SOURCE_TAG_FIELDS = ("user_tags", "date_tags")
_PRETRAIN_REQUIRED_LABEL_SUMMARY_FIELDS = ("per_room_per_date_label_counts",)


def _is_pretrain_manifest(payload: Mapping[str, Any]) -> bool:
    version = payload.get("manifest_version")
    return isinstance(version, str) and version.startswith("beta6_pretrain_manifest_")


def _ensure_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _ensure_exact_keys(name: str, payload: Mapping[str, Any], required: Iterable[str]) -> None:
    expected = set(required)
    provided = set(payload.keys())
    missing = sorted(expected - provided)
    extra = sorted(provided - expected)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing={','.join(missing)}")
        if extra:
            parts.append(f"extra={','.join(extra)}")
        raise ValueError(f"{name} has invalid fields ({'; '.join(parts)})")


def _ensure_nonempty_str(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _resolve_path(raw: str, *, artifact_path: Path | None) -> Path:
    path = Path(raw)
    if path.is_absolute() or artifact_path is None:
        return path
    return (artifact_path.parent / path).resolve()


def validate_intake_artifact(
    artifact: Mapping[str, Any],
    *,
    artifact_path: Path | None = None,
    require_report_files: bool = True,
) -> Dict[str, Any]:
    """
    Validate Beta 6 intake artifact schema and policy invariants.

    Returns a normalized dictionary on success; raises on violations.
    """
    top = _ensure_mapping("intake_artifact", artifact)
    if _is_pretrain_manifest(top):
        return _validate_pretrain_manifest(top)
    _ensure_exact_keys("intake_artifact", top, _REQUIRED_TOP_LEVEL)

    version = _ensure_nonempty_str("artifact_version", top["artifact_version"])
    if version != INTAKE_ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported intake artifact version: {version}; expected {INTAKE_ARTIFACT_VERSION}"
        )
    _ensure_nonempty_str("generated_at", top["generated_at"])

    status = _ensure_nonempty_str("status", top["status"])
    if status not in {APPROVED_STATUS, REJECTED_STATUS}:
        raise ValueError(f"status must be '{APPROVED_STATUS}' or '{REJECTED_STATUS}'")

    pack = _ensure_mapping("pack", top["pack"])
    _ensure_exact_keys("pack", pack, _REQUIRED_PACK_FIELDS)
    _ensure_nonempty_str("pack.candidate_dir", pack["candidate_dir"])
    _ensure_nonempty_str("pack.baseline_dir", pack["baseline_dir"])
    _ensure_nonempty_str("pack.elder_id", pack["elder_id"])
    for int_field in ("min_day", "max_day", "smoke_day", "seed"):
        val = pack[int_field]
        if not isinstance(val, int) or isinstance(val, bool):
            raise TypeError(f"pack.{int_field} must be an integer")
    if int(pack["max_day"]) < int(pack["min_day"]):
        raise ValueError("pack.max_day must be >= pack.min_day")
    if not (int(pack["min_day"]) <= int(pack["smoke_day"]) <= int(pack["max_day"])):
        raise ValueError("pack.smoke_day must be within [min_day, max_day]")

    steps = _ensure_mapping("steps", top["steps"])
    _ensure_exact_keys("steps", steps, _REQUIRED_STEP_NAMES)
    for step_name in _REQUIRED_STEP_NAMES:
        step_payload = _ensure_mapping(f"steps.{step_name}", steps[step_name])
        step_status = _ensure_nonempty_str(f"steps.{step_name}.status", step_payload.get("status"))
        if step_status not in _VALID_STEP_STATUSES:
            raise ValueError(
                f"steps.{step_name}.status must be one of {sorted(_VALID_STEP_STATUSES)}"
            )

    reports = _ensure_mapping("reports", top["reports"])
    _ensure_exact_keys("reports", reports, _REQUIRED_REPORT_FIELDS)
    resolved_reports: Dict[str, str] = {}
    for field_name in _REQUIRED_REPORT_FIELDS:
        report_raw = _ensure_nonempty_str(f"reports.{field_name}", reports[field_name])
        report_path = _resolve_path(report_raw, artifact_path=artifact_path)
        if require_report_files and not report_path.exists():
            raise ValueError(f"Missing required report file: {report_path}")
        resolved_reports[field_name] = str(report_path)

    gate = _ensure_mapping("gate", top["gate"])
    _ensure_exact_keys("gate", gate, _REQUIRED_GATE_FIELDS)
    approved = gate["approved"]
    if not isinstance(approved, bool):
        raise TypeError("gate.approved must be a boolean")

    blocking_reasons = gate["blocking_reasons"]
    if not isinstance(blocking_reasons, list) or any(
        not isinstance(reason, str) or not reason.strip() for reason in blocking_reasons
    ):
        raise TypeError("gate.blocking_reasons must be a list of non-empty strings")

    step_statuses = {step: str(steps[step]["status"]).strip() for step in _REQUIRED_STEP_NAMES}
    all_steps_pass = all(status == "pass" for status in step_statuses.values())

    if approved and status != APPROVED_STATUS:
        raise ValueError("status must be 'approved' when gate.approved is true")
    if approved and not all_steps_pass:
        raise ValueError("gate.approved cannot be true unless all intake steps pass")
    if approved and len(blocking_reasons) != 0:
        raise ValueError("gate.approved cannot be true when blocking reasons exist")
    if (not approved) and status != REJECTED_STATUS:
        raise ValueError("status must be 'rejected' when gate.approved is false")
    if (not approved) and len(blocking_reasons) == 0:
        raise ValueError("gate.blocking_reasons must be non-empty when gate.approved is false")

    normalized = dict(top)
    normalized["reports"] = resolved_reports
    return normalized


def _validate_pretrain_manifest(top: Mapping[str, Any]) -> Dict[str, Any]:
    _ensure_exact_keys("intake_artifact", top, _PRETRAIN_REQUIRED_TOP_LEVEL)
    _ensure_nonempty_str("manifest_version", top["manifest_version"])
    _ensure_nonempty_str("generated_at", top["generated_at"])

    policy = _ensure_mapping("policy", top["policy"])
    if "include_extensions" not in policy:
        raise ValueError("policy.include_extensions is required")

    corpus_roots = top["corpus_roots"]
    if not isinstance(corpus_roots, list):
        raise TypeError("corpus_roots must be a list")

    summary = _ensure_mapping("summary", top["summary"])
    _ensure_exact_keys("summary", summary, _PRETRAIN_REQUIRED_SUMMARY_FIELDS)

    gate = _ensure_mapping("gate", top["gate"])
    _ensure_exact_keys("gate", gate, _REQUIRED_GATE_FIELDS)
    approved = gate["approved"]
    if not isinstance(approved, bool):
        raise TypeError("gate.approved must be a boolean")
    blocking_reasons = gate["blocking_reasons"]
    if not isinstance(blocking_reasons, list) or any(
        not isinstance(reason, str) or not reason.strip() for reason in blocking_reasons
    ):
        raise TypeError("gate.blocking_reasons must be a list of non-empty strings")

    entries = top["entries"]
    quarantine = top["quarantine"]
    duplicates = top["duplicates"]
    violations = top["violations"]
    for field_name, payload in (
        ("entries", entries),
        ("quarantine", quarantine),
        ("duplicates", duplicates),
        ("violations", violations),
    ):
        if not isinstance(payload, list):
            raise TypeError(f"{field_name} must be a list")

    for item in entries:
        entry = _ensure_mapping("entries[]", item)
        _ensure_exact_keys("entries[]", entry, _PRETRAIN_REQUIRED_ENTRY_FIELDS)
        _ensure_nonempty_str("entries[].path", entry["path"])
        _ensure_nonempty_str("entries[].content_hash", entry["content_hash"])
        if str(entry["status"]).strip() != "auto_approved":
            raise ValueError("entries[].status must be 'auto_approved'")
        source_tags = _ensure_mapping("entries[].source_tags", entry["source_tags"])
        _ensure_exact_keys("entries[].source_tags", source_tags, _PRETRAIN_REQUIRED_SOURCE_TAG_FIELDS)
        label_summary = _ensure_mapping("entries[].label_summary", entry["label_summary"])
        _ensure_exact_keys(
            "entries[].label_summary",
            label_summary,
            _PRETRAIN_REQUIRED_LABEL_SUMMARY_FIELDS,
        )

    for item in quarantine:
        entry = _ensure_mapping("quarantine[]", item)
        if "quarantine_reasons" not in entry:
            raise ValueError("quarantine[].quarantine_reasons is required")
        reasons = entry["quarantine_reasons"]
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) or not reason.strip() for reason in reasons
        ):
            raise TypeError("quarantine[].quarantine_reasons must be a list of non-empty strings")

    stats = _ensure_mapping("stats", top["stats"])
    records_kept = int(stats.get("records_kept", len(entries)) or 0)
    if approved and records_kept <= 0:
        raise ValueError("gate.approved cannot be true when stats.records_kept <= 0")
    if approved and len(blocking_reasons) != 0:
        raise ValueError("gate.approved cannot be true when blocking reasons exist")
    if (not approved) and len(blocking_reasons) == 0:
        raise ValueError("gate.blocking_reasons must be non-empty when gate.approved is false")

    return dict(top)


def load_intake_artifact(path: Path | str, *, require_report_files: bool = True) -> Dict[str, Any]:
    artifact_path = Path(path).resolve()
    with artifact_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return validate_intake_artifact(
        payload,
        artifact_path=artifact_path,
        require_report_files=require_report_files,
    )


def assert_intake_artifact_approved(path: Path | str, *, require_report_files: bool = True) -> Dict[str, Any]:
    artifact = load_intake_artifact(path, require_report_files=require_report_files)
    if not bool(artifact["gate"]["approved"]):
        reasons = list(artifact["gate"]["blocking_reasons"])
        raise ValueError(f"Intake gate is not approved: {reasons}")
    return artifact
