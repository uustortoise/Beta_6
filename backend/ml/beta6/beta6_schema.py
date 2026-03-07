"""Beta 6 YAML schema validation and strict config loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from ml.yaml_compat import load_yaml_file


EXPECTED_BETA6_CONFIG_FILES: tuple[str, ...] = (
    "beta6_active_learning_policy.yaml",
    "beta6_adapter_policy.yaml",
    "beta6_canary_gate.yaml",
    "beta6_critical_labels.yaml",
    "beta6_duration_prior_policy.yaml",
    "beta6_fallback_mode_policy.yaml",
    "beta6_golden_safe_finetune.yaml",
    "beta6_lane_b_event_labels.yaml",
    "beta6_model_behavior_slo.yaml",
    "beta6_policy_defaults.yaml",
    "beta6_pretrain.yaml",
    "beta6_rollout_ladder.yaml",
    "beta6_room_capability_gate_profiles.yaml",
    "beta6_runtime_eval_parity.yaml",
    "beta6_timeline_hard_gates.yaml",
    "beta6_uncertainty_policy.yaml",
    "beta6_unknown_policy.yaml",
)

RUNTIME_CRITICAL_BETA6_CONFIG_FILES: tuple[str, ...] = (
    "beta6_adapter_policy.yaml",
    "beta6_canary_gate.yaml",
    "beta6_unknown_policy.yaml",
    "beta6_duration_prior_policy.yaml",
    "beta6_rollout_ladder.yaml",
    "beta6_runtime_eval_parity.yaml",
    "beta6_room_capability_gate_profiles.yaml",
    "beta6_policy_defaults.yaml",
    "beta6_critical_labels.yaml",
    "beta6_lane_b_event_labels.yaml",
)

_SCHEMA_RULES: dict[str, dict[str, str]] = {
    "beta6_active_learning_policy.yaml": {
        "version": "scalar",
        "active_learning": "mapping",
    },
    "beta6_adapter_policy.yaml": {
        "version": "scalar",
        "adapter": "mapping",
    },
    "beta6_canary_gate.yaml": {
        "version": "scalar",
        "canary": "mapping",
    },
    "beta6_critical_labels.yaml": {
        "version": "scalar",
        "critical_labels_by_room": "mapping",
        "defaults": "mapping",
    },
    "beta6_duration_prior_policy.yaml": {
        "version": "scalar",
        "duration_priors": "mapping",
        "transition": "mapping",
    },
    "beta6_fallback_mode_policy.yaml": {
        "version": "scalar",
        "state_file": "scalar",
        "defaults": "mapping",
        "event_contract": "mapping",
        "required_payload_fields": "mapping",
    },
    "beta6_golden_safe_finetune.yaml": {
        "version": "scalar",
        "fine_tune": "mapping",
    },
    "beta6_lane_b_event_labels.yaml": {
        "version": "scalar",
        "lane_b_event_labels_by_room": "mapping",
    },
    "beta6_model_behavior_slo.yaml": {
        "version": "scalar",
        "thresholds": "mapping",
        "reason_code_ratio_caps": "mapping",
        "routing": "mapping",
    },
    "beta6_policy_defaults.yaml": {
        "version": "scalar",
        "clinical_priority": "mapping",
        "calibration": "mapping",
        "unoccupied_downsample": "mapping",
        "minority_sampling": "mapping",
        "data_viability": "mapping",
        "training": "mapping",
        "runtime": "mapping",
    },
    "beta6_pretrain.yaml": {
        "version": "scalar",
        "pretrain": "mapping",
    },
    "beta6_rollout_ladder.yaml": {
        "version": "scalar",
        "ladder": "mapping",
        "progression_criteria": "mapping",
        "auto_rollback_triggers": "mapping",
        "fallback": "mapping",
    },
    "beta6_room_capability_gate_profiles.yaml": {
        "version": "scalar",
        "default_room_type": "scalar",
        "profiles": "mapping",
        "room_type_hints": "mapping",
    },
    "beta6_runtime_eval_parity.yaml": {
        "version": "scalar",
        "label_map": "mapping",
        "decoder_policy": "mapping",
        "required_parity_fields": "sequence",
    },
    "beta6_timeline_hard_gates.yaml": {
        "version": "scalar",
        "metrics": "mapping",
        "reason_code_precedence": "sequence",
        "threshold_source": "mapping",
    },
    "beta6_uncertainty_policy.yaml": {
        "version": "scalar",
        "strict_taxonomy": "scalar",
        "allow_combined_states": "scalar",
        "accepted_input_shapes": "sequence",
        "taxonomy": "mapping",
        "contract_violations": "mapping",
    },
    "beta6_unknown_policy.yaml": {
        "version": "scalar",
        "unknown_policy": "mapping",
        "targets": "mapping",
    },
}

_NESTED_SCHEMA_RULES: dict[str, dict[str, str]] = {
    "beta6_unknown_policy.yaml": {
        "unknown_policy.min_confidence": "scalar",
        "unknown_policy.max_entropy": "scalar",
        "unknown_policy.outside_sensed_space_threshold": "scalar",
        "targets.abstain_rate_min": "scalar",
        "targets.abstain_rate_max": "scalar",
        "targets.unknown_recall_min": "scalar",
    },
    "beta6_duration_prior_policy.yaml": {
        "duration_priors.default.min_minutes": "scalar",
        "duration_priors.default.target_minutes": "scalar",
        "duration_priors.default.max_minutes": "scalar",
        "duration_priors.default.penalty_weight": "scalar",
        "duration_priors.by_label": "mapping",
        "transition.switch_penalty": "scalar",
        "transition.impossible_transition_penalty": "scalar",
        "transition.self_transition_bias": "scalar",
        "transition.step_minutes": "scalar",
    },
    "beta6_canary_gate.yaml": {
        "canary.require_real_data_evidence": "boolean",
        "canary.min_residents_covered": "scalar",
        "canary.min_resident_days_total": "scalar",
        "canary.allowed_data_sources": "sequence",
    },
    "beta6_adapter_policy.yaml": {
        "adapter.rank": "scalar",
        "adapter.alpha": "scalar",
        "adapter.l2_reg": "scalar",
        "adapter.random_seed": "scalar",
        "adapter.min_rows": "scalar",
        "adapter.max_versions_per_resident": "scalar",
        "adapter.min_warmup_accuracy": "scalar",
        "adapter.retirement_inactive_days": "scalar",
        "adapter.enable_auto_retire": "boolean",
    },
    "beta6_runtime_eval_parity.yaml": {
        "label_map.occupied": "scalar",
        "label_map.unoccupied": "scalar",
        "decoder_policy.spike_suppression": "scalar",
    },
    "beta6_rollout_ladder.yaml": {
        "ladder.rungs": "sequence",
        "progression_criteria.all_mandatory_metric_floors_pass": "boolean",
        "progression_criteria.block_on_open_p0_incidents": "boolean",
        "progression_criteria.min_nightly_pipeline_success_rate": "scalar",
        "progression_criteria.require_drift_alerts_within_budget": "boolean",
        "progression_criteria.require_timeline_hard_gates_all_rooms": "boolean",
        "auto_rollback_triggers.consecutive_nights": "scalar",
        "auto_rollback_triggers.mae_regression_pct_gt": "scalar",
        "auto_rollback_triggers.alert_precision_below_threshold": "boolean",
        "auto_rollback_triggers.worst_room_f1_below_floor": "boolean",
        "auto_rollback_triggers.pipeline_success_rate_lt": "scalar",
        "fallback.baseline_profile": "scalar",
        "fallback.serving_mode": "scalar",
        "fallback.require_registry_event": "boolean",
        "fallback.require_ops_alert": "boolean",
        "fallback.recovery_consecutive_pass_nights": "scalar",
    },
    "beta6_room_capability_gate_profiles.yaml": {
        "profiles": "mapping",
        "room_type_hints": "mapping",
    },
    "beta6_critical_labels.yaml": {
        "critical_labels_by_room": "mapping",
        "defaults.include_uncertainty_labels": "scalar",
        "defaults.uncertainty_labels": "sequence",
    },
    "beta6_lane_b_event_labels.yaml": {
        "lane_b_event_labels_by_room": "mapping",
    },
    "beta6_policy_defaults.yaml": {
        "clinical_priority.multipliers_by_label": "mapping",
        "calibration.precision_targets_by_label": "mapping",
        "calibration.recall_floors_by_label": "mapping",
        "unoccupied_downsample.min_share_by_room": "mapping",
        "unoccupied_downsample.stride_by_room": "mapping",
        "minority_sampling.target_share_by_room": "mapping",
        "minority_sampling.max_multiplier_by_room": "mapping",
        "data_viability.min_observed_days_by_room": "mapping",
        "data_viability.min_post_gap_rows_by_room": "mapping",
        "data_viability.max_unresolved_drop_ratio_by_room": "mapping",
        "data_viability.min_training_windows_by_room": "mapping",
        "training.timeline_native_rooms": "sequence",
        "training.two_stage_core": "mapping",
        "training.two_stage_core.enabled": "boolean",
        "training.two_stage_core.rooms": "sequence",
        "training.two_stage_core.gate_mode": "scalar",
        "training.two_stage_core.stage_a_occupied_threshold": "scalar",
        "training.two_stage_core.stage_a_target_precision": "scalar",
        "training.two_stage_core.stage_a_recall_floor": "scalar",
        "training.two_stage_core.stage_a_threshold_min": "scalar",
        "training.two_stage_core.stage_a_threshold_max": "scalar",
        "training.two_stage_core.stage_a_min_predicted_occupied_ratio": "scalar",
        "training.two_stage_core.stage_a_min_predicted_occupied_abs": "scalar",
        "training.min_holdout_support_default": "scalar",
        "training.min_holdout_support_by_room": "mapping",
        "runtime.hard_negative_risky_rooms": "sequence",
        "runtime.runtime_unknown_rooms": "sequence",
        "runtime.wf_min_minority_support_default": "scalar",
        "runtime.wf_min_minority_support_by_room": "mapping",
    },
}


@dataclass(frozen=True)
class Beta6SchemaValidationReport:
    status: str
    checked_files: int
    expected_files: int
    errors: list[str]
    config_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checked_files": self.checked_files,
            "expected_files": self.expected_files,
            "errors": list(self.errors),
            "config_dir": self.config_dir,
        }


def _as_mapping(payload: Any) -> Mapping[str, Any]:
    return payload if isinstance(payload, Mapping) else {}


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, (list, tuple)):
        return "sequence"
    return "scalar"


def _matches_expected_type(value: Any, expected_type: str) -> bool:
    if expected_type == "mapping":
        return isinstance(value, Mapping)
    if expected_type == "sequence":
        return isinstance(value, (list, tuple))
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "scalar":
        return not isinstance(value, Mapping) and not isinstance(value, (list, tuple))
    return False


def _validate_payload(filename: str, payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    rules = _SCHEMA_RULES.get(filename)
    if rules is None:
        return [f"{filename}: no schema rules registered"]

    version = str(payload.get("version", "")).strip()
    if version != "v1":
        errors.append(f"{filename}: invalid version={version!r} (expected 'v1')")

    for key, expected_type in rules.items():
        if key not in payload:
            errors.append(f"{filename}: missing key '{key}'")
            continue
        observed_type = _type_name(payload[key])
        if not _matches_expected_type(payload[key], expected_type):
            errors.append(
                f"{filename}: key '{key}' expected {expected_type}, got {observed_type}"
            )

    nested_rules = _NESTED_SCHEMA_RULES.get(filename, {})
    for dotted_key, expected_type in nested_rules.items():
        node: Any = payload
        missing = False
        for token in dotted_key.split("."):
            if not isinstance(node, Mapping) or token not in node:
                missing = True
                break
            node = node[token]
        if missing:
            errors.append(f"{filename}: missing nested key '{dotted_key}'")
            continue
        observed_type = _type_name(node)
        if not _matches_expected_type(node, expected_type):
            errors.append(
                f"{filename}: nested key '{dotted_key}' expected {expected_type}, got {observed_type}"
            )
    return errors


def load_validated_beta6_config(
    path: str | Path,
    *,
    expected_filename: str | None = None,
) -> Mapping[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Beta6 config missing: {config_path}")
    filename = expected_filename or config_path.name
    payload = _as_mapping(load_yaml_file(config_path))
    if not payload:
        raise ValueError(f"Beta6 config payload must be mapping: {config_path}")
    errors = _validate_payload(filename, payload)
    if errors:
        raise ValueError(" ; ".join(errors))
    return payload


def validate_all_beta6_configs(config_dir: str | Path) -> Beta6SchemaValidationReport:
    root = Path(config_dir).resolve()
    errors: list[str] = []
    checked = 0

    for filename in EXPECTED_BETA6_CONFIG_FILES:
        file_path = root / filename
        if not file_path.exists():
            errors.append(f"{filename}: file missing")
            continue
        try:
            payload = _as_mapping(load_yaml_file(file_path))
        except Exception as exc:
            errors.append(f"{filename}: yaml load failed ({type(exc).__name__}: {exc})")
            continue
        if not payload:
            errors.append(f"{filename}: payload must be mapping")
            continue
        errors.extend(_validate_payload(filename, payload))
        checked += 1

    status = "pass" if not errors else "fail"
    return Beta6SchemaValidationReport(
        status=status,
        checked_files=checked,
        expected_files=len(EXPECTED_BETA6_CONFIG_FILES),
        errors=errors,
        config_dir=str(root),
    )


def is_beta6_runtime_critical_config(path: str | Path) -> bool:
    return Path(path).name in RUNTIME_CRITICAL_BETA6_CONFIG_FILES


__all__ = [
    "EXPECTED_BETA6_CONFIG_FILES",
    "RUNTIME_CRITICAL_BETA6_CONFIG_FILES",
    "Beta6SchemaValidationReport",
    "is_beta6_runtime_critical_config",
    "load_validated_beta6_config",
    "validate_all_beta6_configs",
]
