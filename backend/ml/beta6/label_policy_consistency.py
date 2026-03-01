"""Fail-closed consistency checks for Beta 6 label policy configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from ml.yaml_compat import load_yaml_file
from utils.room_utils import normalize_room_name


@dataclass(frozen=True)
class LabelPolicyIssue:
    """Single consistency finding."""

    severity: str  # "error" | "warning"
    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "severity": str(self.severity),
            "code": str(self.code),
            "message": str(self.message),
        }
        if self.context:
            payload["context"] = dict(self.context)
        return payload


@dataclass(frozen=True)
class LabelPolicyConsistencyReport:
    """Validation report for Beta 6 label policy consistency."""

    status: str
    checked_rooms: int
    observed_rooms: int
    errors: Sequence[LabelPolicyIssue] = field(default_factory=tuple)
    warnings: Sequence[LabelPolicyIssue] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": str(self.status),
            "checked_rooms": int(self.checked_rooms),
            "observed_rooms": int(self.observed_rooms),
            "error_count": int(len(self.errors)),
            "warning_count": int(len(self.warnings)),
            "errors": [item.to_dict() for item in self.errors],
            "warnings": [item.to_dict() for item in self.warnings],
        }


def _as_mapping(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    return {}


def _normalize_label(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _canonicalize_label(label: str, alias_map: Mapping[str, str]) -> str:
    token = _normalize_label(label)
    if not token:
        return ""
    seen: set[str] = set()
    current = token
    while current in alias_map and current not in seen:
        seen.add(current)
        current = _normalize_label(alias_map.get(current))
        if not current:
            return token
    return current or token


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    payload = load_yaml_file(path) or {}
    if isinstance(payload, Mapping):
        return payload
    return {}


def _build_alias_map(critical_cfg: Mapping[str, Any]) -> Dict[str, str]:
    defaults = _as_mapping(critical_cfg.get("defaults"))
    raw_alias_map = _as_mapping(defaults.get("alias_to_canonical"))
    alias_map: Dict[str, str] = {}
    for raw_alias, raw_canonical in raw_alias_map.items():
        alias = _normalize_label(raw_alias)
        canonical = _normalize_label(raw_canonical)
        if alias and canonical and alias != canonical:
            alias_map[alias] = canonical
    return alias_map


def _collect_observed_labels_by_room(models_dir: Path, alias_map: Mapping[str, str]) -> Dict[str, set[str]]:
    observed: Dict[str, set[str]] = {}
    if not models_dir.exists():
        return observed
    for path in models_dir.glob("*/*_v*_decision_trace.json"):
        room_raw = path.name.split("_v", 1)[0]
        room = normalize_room_name(room_raw)
        if not room:
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        metrics = payload.get("metrics", {})
        metrics = metrics if isinstance(metrics, Mapping) else {}
        per_label_support = metrics.get("per_label_support", {})
        per_label_support = per_label_support if isinstance(per_label_support, Mapping) else {}
        bucket = observed.setdefault(room, set())
        for raw_label in per_label_support.keys():
            label = _normalize_label(raw_label)
            if not label:
                continue
            bucket.add(_canonicalize_label(label, alias_map))
    return observed


def validate_label_policy_consistency(
    *,
    config_dir: Path | str,
    models_dir: Path | str | None = None,
    fail_on_warnings: bool = False,
) -> LabelPolicyConsistencyReport:
    """
    Validate consistency between critical labels, Lane-B labels, and alias policy.

    Fail-closed errors:
    - non-canonical labels in critical/Lane-B policy (alias token instead of canonical token)
    - malformed room/event mappings
    - Lane-B labels not present in room critical labels
    """
    cfg_root = Path(config_dir).resolve()
    critical_cfg = _load_yaml_mapping(cfg_root / "beta6_critical_labels.yaml")
    lane_b_cfg = _load_yaml_mapping(cfg_root / "beta6_lane_b_event_labels.yaml")

    errors: list[LabelPolicyIssue] = []
    warnings: list[LabelPolicyIssue] = []

    alias_map = _build_alias_map(critical_cfg)
    critical_by_room_raw = _as_mapping(critical_cfg.get("critical_labels_by_room"))
    lane_b_by_room_raw = _as_mapping(lane_b_cfg.get("lane_b_event_labels_by_room"))

    critical_by_room: Dict[str, set[str]] = {}
    for raw_room, raw_labels in critical_by_room_raw.items():
        room = normalize_room_name(raw_room)
        if not room:
            errors.append(
                LabelPolicyIssue(
                    severity="error",
                    code="invalid_room_name",
                    message="critical_labels_by_room has an invalid room name.",
                    context={"room": str(raw_room)},
                )
            )
            continue
        if not isinstance(raw_labels, (list, tuple, set)):
            errors.append(
                LabelPolicyIssue(
                    severity="error",
                    code="critical_labels_not_list",
                    message="critical_labels_by_room entries must be a list.",
                    context={"room": room, "type": type(raw_labels).__name__},
                )
            )
            continue

        canonical_labels: set[str] = set()
        for raw_label in raw_labels:
            label = _normalize_label(raw_label)
            if not label:
                errors.append(
                    LabelPolicyIssue(
                        severity="error",
                        code="empty_critical_label",
                        message="critical_labels_by_room contains an empty label.",
                        context={"room": room},
                    )
                )
                continue
            canonical = _canonicalize_label(label, alias_map)
            if label != canonical:
                errors.append(
                    LabelPolicyIssue(
                        severity="error",
                        code="non_canonical_critical_label",
                        message="Critical labels must use canonical vocabulary.",
                        context={"room": room, "label": label, "canonical": canonical},
                    )
                )
            if canonical in canonical_labels:
                errors.append(
                    LabelPolicyIssue(
                        severity="error",
                        code="duplicate_critical_label",
                        message="Duplicate canonical label in critical_labels_by_room.",
                        context={"room": room, "label": canonical},
                    )
                )
            canonical_labels.add(canonical)
        critical_by_room[room] = canonical_labels

    for raw_room, raw_event_map in lane_b_by_room_raw.items():
        room = normalize_room_name(raw_room)
        if not room:
            errors.append(
                LabelPolicyIssue(
                    severity="error",
                    code="invalid_room_name",
                    message="lane_b_event_labels_by_room has an invalid room name.",
                    context={"room": str(raw_room)},
                )
            )
            continue
        if room not in critical_by_room:
            errors.append(
                LabelPolicyIssue(
                    severity="error",
                    code="lane_b_room_missing_critical_mapping",
                    message="Lane-B room mapping must exist in critical_labels_by_room.",
                    context={"room": room},
                )
            )
            continue
        if not isinstance(raw_event_map, Mapping):
            errors.append(
                LabelPolicyIssue(
                    severity="error",
                    code="lane_b_event_map_not_mapping",
                    message="lane_b_event_labels_by_room entries must be a mapping.",
                    context={"room": room, "type": type(raw_event_map).__name__},
                )
            )
            continue

        room_critical = critical_by_room.get(room, set())
        for raw_event, raw_labels in raw_event_map.items():
            event = _normalize_label(raw_event)
            if not event:
                errors.append(
                    LabelPolicyIssue(
                        severity="error",
                        code="empty_lane_b_event_name",
                        message="Lane-B event name cannot be empty.",
                        context={"room": room},
                    )
                )
                continue
            if not isinstance(raw_labels, (list, tuple, set)):
                errors.append(
                    LabelPolicyIssue(
                        severity="error",
                        code="lane_b_event_labels_not_list",
                        message="Lane-B event labels must be a list.",
                        context={"room": room, "event": event, "type": type(raw_labels).__name__},
                    )
                )
                continue

            event_labels: set[str] = set()
            for raw_label in raw_labels:
                label = _normalize_label(raw_label)
                if not label:
                    errors.append(
                        LabelPolicyIssue(
                            severity="error",
                            code="empty_lane_b_label",
                            message="Lane-B event labels contain an empty label.",
                            context={"room": room, "event": event},
                        )
                    )
                    continue
                canonical = _canonicalize_label(label, alias_map)
                if label != canonical:
                    errors.append(
                        LabelPolicyIssue(
                            severity="error",
                            code="non_canonical_lane_b_label",
                            message="Lane-B labels must use canonical vocabulary.",
                            context={
                                "room": room,
                                "event": event,
                                "label": label,
                                "canonical": canonical,
                            },
                        )
                    )
                if canonical in event_labels:
                    errors.append(
                        LabelPolicyIssue(
                            severity="error",
                            code="duplicate_lane_b_label",
                            message="Duplicate canonical label in Lane-B event mapping.",
                            context={"room": room, "event": event, "label": canonical},
                        )
                    )
                event_labels.add(canonical)
                if canonical not in room_critical:
                    errors.append(
                        LabelPolicyIssue(
                            severity="error",
                            code="lane_b_label_not_in_critical",
                            message="Lane-B labels must be included in room critical labels.",
                            context={"room": room, "event": event, "label": canonical},
                        )
                    )

    observed_by_room: Dict[str, set[str]] = {}
    if models_dir is not None:
        observed_by_room = _collect_observed_labels_by_room(Path(models_dir).resolve(), alias_map)
        for room, critical_labels in critical_by_room.items():
            observed = observed_by_room.get(room, set())
            missing = sorted(label for label in critical_labels if label not in observed)
            if missing:
                warnings.append(
                    LabelPolicyIssue(
                        severity="warning",
                        code="critical_labels_missing_in_observed_data",
                        message="Configured critical labels were not observed in saved decision traces.",
                        context={"room": room, "missing_labels": missing},
                    )
                )

    status = "pass"
    if errors:
        status = "fail"
    elif fail_on_warnings and warnings:
        status = "fail"

    return LabelPolicyConsistencyReport(
        status=status,
        checked_rooms=len(critical_by_room),
        observed_rooms=len(observed_by_room),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


__all__ = [
    "LabelPolicyIssue",
    "LabelPolicyConsistencyReport",
    "validate_label_policy_consistency",
]

