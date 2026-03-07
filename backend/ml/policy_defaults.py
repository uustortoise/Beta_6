"""Config-backed defaults used by active Beta 6 policy/runtime paths."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from ml.beta6.beta6_schema import load_validated_beta6_config

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "beta6_policy_defaults.yaml"


def _as_mapping(raw: Any) -> Mapping[str, Any]:
    return raw if isinstance(raw, Mapping) else {}


@lru_cache(maxsize=1)
def _load_config() -> Mapping[str, Any]:
    payload = load_validated_beta6_config(
        _CONFIG_PATH,
        expected_filename="beta6_policy_defaults.yaml",
    )
    return _as_mapping(payload)


def _get_map(*path: str) -> dict[str, Any]:
    node: Any = _load_config()
    for token in path:
        node = _as_mapping(node).get(token)
    return {str(k): v for k, v in _as_mapping(node).items()}


def _get_str_list(*path: str) -> list[str]:
    node: Any = _load_config()
    for token in path:
        node = _as_mapping(node).get(token)
    if not isinstance(node, (list, tuple)):
        return []
    return [str(item).strip().lower() for item in node if str(item).strip()]


def get_clinical_priority_multipliers_by_label() -> dict[str, float]:
    return {
        str(k).strip().lower(): float(v)
        for k, v in _get_map("clinical_priority", "multipliers_by_label").items()
        if str(k).strip()
    }


def get_calibration_precision_targets_by_label() -> dict[str, float]:
    return {
        str(k).strip().lower(): float(v)
        for k, v in _get_map("calibration", "precision_targets_by_label").items()
        if str(k).strip()
    }


def get_calibration_recall_floors_by_label() -> dict[str, float]:
    return {
        str(k).strip().lower(): float(v)
        for k, v in _get_map("calibration", "recall_floors_by_label").items()
        if str(k).strip()
    }


def get_unoccupied_downsample_min_share_by_room() -> dict[str, float]:
    return {
        str(k).strip().lower(): float(v)
        for k, v in _get_map("unoccupied_downsample", "min_share_by_room").items()
        if str(k).strip()
    }


def get_unoccupied_downsample_stride_by_room() -> dict[str, int]:
    return {
        str(k).strip().lower(): int(v)
        for k, v in _get_map("unoccupied_downsample", "stride_by_room").items()
        if str(k).strip()
    }


def get_data_viability_min_observed_days_by_room() -> dict[str, int]:
    return {
        str(k).strip().lower(): int(v)
        for k, v in _get_map("data_viability", "min_observed_days_by_room").items()
        if str(k).strip()
    }


def get_data_viability_min_post_gap_rows_by_room() -> dict[str, int]:
    return {
        str(k).strip().lower(): int(v)
        for k, v in _get_map("data_viability", "min_post_gap_rows_by_room").items()
        if str(k).strip()
    }


def get_data_viability_max_unresolved_drop_ratio_by_room() -> dict[str, float]:
    return {
        str(k).strip().lower(): float(v)
        for k, v in _get_map("data_viability", "max_unresolved_drop_ratio_by_room").items()
        if str(k).strip()
    }


def get_data_viability_min_training_windows_by_room() -> dict[str, int]:
    return {
        str(k).strip().lower(): int(v)
        for k, v in _get_map("data_viability", "min_training_windows_by_room").items()
        if str(k).strip()
    }


def get_timeline_native_rooms_default() -> list[str]:
    return _get_str_list("training", "timeline_native_rooms")


def get_training_min_holdout_support_default() -> int:
    value = _as_mapping(_load_config()).get("training", {})
    if isinstance(value, Mapping):
        raw = value.get("min_holdout_support_default")
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return 3
    return 3


def get_training_min_holdout_support_by_room() -> dict[str, int]:
    return {
        str(k).strip().lower(): int(v)
        for k, v in _get_map("training", "min_holdout_support_by_room").items()
        if str(k).strip()
    }


def get_training_two_stage_core_enabled_default() -> bool:
    value = _as_mapping(_load_config()).get("training", {})
    if not isinstance(value, Mapping):
        return False
    raw = value.get("two_stage_core", {})
    if not isinstance(raw, Mapping):
        return False
    return bool(raw.get("enabled", False))


def get_training_two_stage_core_rooms_default() -> list[str]:
    return _get_str_list("training", "two_stage_core", "rooms")


def get_training_two_stage_core_gate_mode_default() -> str:
    value = _as_mapping(_load_config()).get("training", {})
    if not isinstance(value, Mapping):
        return "shadow"
    raw = value.get("two_stage_core", {})
    if not isinstance(raw, Mapping):
        return "shadow"
    mode = str(raw.get("gate_mode", "shadow")).strip().lower()
    return mode if mode in {"primary", "shadow"} else "shadow"


def _get_training_two_stage_core_float_default(key: str, fallback: float) -> float:
    value = _as_mapping(_load_config()).get("training", {})
    if not isinstance(value, Mapping):
        return fallback
    raw = value.get("two_stage_core", {})
    if not isinstance(raw, Mapping):
        return fallback
    try:
        return float(raw.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def get_training_two_stage_core_stage_a_occupied_threshold_default() -> float:
    return _get_training_two_stage_core_float_default("stage_a_occupied_threshold", 0.5)


def get_training_two_stage_core_stage_a_target_precision_default() -> float:
    return _get_training_two_stage_core_float_default("stage_a_target_precision", 0.70)


def get_training_two_stage_core_stage_a_recall_floor_default() -> float:
    return _get_training_two_stage_core_float_default("stage_a_recall_floor", 0.20)


def get_training_two_stage_core_stage_a_threshold_min_default() -> float:
    return _get_training_two_stage_core_float_default("stage_a_threshold_min", 0.00)


def get_training_two_stage_core_stage_a_threshold_max_default() -> float:
    return _get_training_two_stage_core_float_default("stage_a_threshold_max", 0.95)


def get_training_two_stage_core_stage_a_min_predicted_occupied_ratio_default() -> float:
    return _get_training_two_stage_core_float_default("stage_a_min_predicted_occupied_ratio", 0.50)


def get_training_two_stage_core_stage_a_min_predicted_occupied_abs_default() -> float:
    return _get_training_two_stage_core_float_default("stage_a_min_predicted_occupied_abs", 0.05)


def get_hard_negative_risky_rooms_default() -> list[str]:
    return _get_str_list("runtime", "hard_negative_risky_rooms")


def get_runtime_unknown_rooms_default() -> list[str]:
    return _get_str_list("runtime", "runtime_unknown_rooms")


def get_runtime_wf_min_minority_support_default() -> int:
    value = _as_mapping(_load_config()).get("runtime", {})
    if isinstance(value, Mapping):
        raw = value.get("wf_min_minority_support_default")
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 3
    return 3


def get_runtime_wf_min_minority_support_by_room() -> dict[str, int]:
    return {
        str(k).strip().lower(): int(v)
        for k, v in _get_map("runtime", "wf_min_minority_support_by_room").items()
        if str(k).strip()
    }


def get_minority_sampling_target_share_by_room() -> dict[str, float]:
    return {
        str(k).strip().lower(): float(v)
        for k, v in _get_map("minority_sampling", "target_share_by_room").items()
        if str(k).strip()
    }


def get_minority_sampling_max_multiplier_by_room() -> dict[str, int]:
    return {
        str(k).strip().lower(): int(v)
        for k, v in _get_map("minority_sampling", "max_multiplier_by_room").items()
        if str(k).strip()
    }
