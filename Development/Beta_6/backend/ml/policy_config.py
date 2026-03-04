from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
import os
from typing import Any, Mapping

from elderlycare_v1_16.preprocessing.gap_policy import resolve_max_ffill_gap_seconds
from ml.policy_defaults import (
    get_calibration_precision_targets_by_label,
    get_calibration_recall_floors_by_label,
    get_clinical_priority_multipliers_by_label,
    get_data_viability_max_unresolved_drop_ratio_by_room,
    get_data_viability_min_observed_days_by_room,
    get_data_viability_min_post_gap_rows_by_room,
    get_data_viability_min_training_windows_by_room,
    get_minority_sampling_max_multiplier_by_room,
    get_minority_sampling_target_share_by_room,
    get_unoccupied_downsample_min_share_by_room,
    get_unoccupied_downsample_stride_by_room,
)
from utils.room_utils import normalize_room_name

logger = logging.getLogger(__name__)

_RELEASE_GATE_EVIDENCE_PROFILES: dict[str, dict[str, int]] = {
    # Keep existing behavior as default.
    "production": {
        "min_validation_class_support": 20,
        "min_recall_support": 30,
    },
    # Pilot profiles for short-window timeline validation.
    "pilot_stage_a": {
        "min_validation_class_support": 10,
        "min_recall_support": 20,
    },
    "pilot_stage_b": {
        "min_validation_class_support": 8,
        "min_recall_support": 8,
    },
}


def _is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _resolve_release_gate_profile(env: Mapping[str, str], default_profile: str) -> str:
    profile = str(env.get("RELEASE_GATE_EVIDENCE_PROFILE", default_profile)).strip().lower()
    if not profile:
        profile = "production"
    if profile not in _RELEASE_GATE_EVIDENCE_PROFILES:
        logger.warning(
            "Unknown RELEASE_GATE_EVIDENCE_PROFILE=%s; defaulting to production.",
            profile,
        )
        profile = "production"
    return profile


def _read_int_env(
    env: Mapping[str, str],
    name: str,
    default: int,
    minimum: int = 0,
) -> int:
    raw = env.get(name)
    if raw is None:
        return max(int(default), int(minimum))
    try:
        return max(int(raw), int(minimum))
    except (TypeError, ValueError):
        return max(int(default), int(minimum))


def _read_float_env(
    env: Mapping[str, str],
    name: str,
    default: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    raw = env.get(name)
    if raw is None:
        return float(min(max(float(default), minimum), maximum))
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        parsed = float(default)
    return float(min(max(parsed, minimum), maximum))


def _parse_override_map(raw_value: str) -> dict[str, str]:
    """
    Parse CSV-like room/label map values:
    - "bathroom:0.2,bedroom=0.3"
    """
    result: dict[str, str] = {}
    txt = str(raw_value or "").strip()
    if not txt:
        return result

    for token in txt.split(","):
        item = str(token).strip()
        if not item:
            continue
        if ":" in item:
            key_raw, value_raw = item.split(":", 1)
        elif "=" in item:
            key_raw, value_raw = item.split("=", 1)
        else:
            continue
        key = normalize_room_name(key_raw)
        value = str(value_raw).strip()
        if key and value:
            result[key] = value
    return result


def _read_room_int_overrides(
    env: Mapping[str, str],
    var_name: str,
    fallback: Mapping[str, int],
    minimum: int = 0,
) -> dict[str, int]:
    raw_map = env.get(var_name)
    if raw_map is None:
        return {k: int(v) for k, v in fallback.items()}
    out: dict[str, int] = {}
    parsed = _parse_override_map(raw_map)
    for room, value in parsed.items():
        try:
            out[room] = max(int(minimum), int(value))
        except (TypeError, ValueError):
            continue
    return out


def _read_room_float_overrides(
    env: Mapping[str, str],
    var_name: str,
    fallback: Mapping[str, float],
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> dict[str, float]:
    raw_map = env.get(var_name)
    if raw_map is None:
        return {k: float(v) for k, v in fallback.items()}
    out: dict[str, float] = {}
    parsed = _parse_override_map(raw_map)
    for room, value in parsed.items():
        try:
            out[room] = float(min(max(float(value), minimum), maximum))
        except (TypeError, ValueError):
            continue
    return out


def _read_room_label_float_overrides(
    env: Mapping[str, str],
    var_name: str,
    fallback: Mapping[str, float],
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> dict[str, float]:
    """
    Parse room+label keyed float map.
    Expected keys: "room.label" (e.g., "bedroom.unoccupied:0.6")
    """
    raw_map = env.get(var_name)
    if raw_map is None:
        return {str(k): float(v) for k, v in fallback.items()}
    out: dict[str, float] = {}
    for token in str(raw_map).split(","):
        item = str(token).strip()
        if not item:
            continue
        if ":" in item:
            key_raw, value_raw = item.split(":", 1)
        elif "=" in item:
            key_raw, value_raw = item.split("=", 1)
        else:
            continue
        key_txt = str(key_raw).strip().lower()
        if "." not in key_txt:
            continue
        room_part, label_part = key_txt.split(".", 1)
        room = normalize_room_name(room_part)
        label = str(label_part).strip().lower()
        if not room or not label:
            continue
        try:
            out[f"{room}.{label}"] = float(min(max(float(value_raw), minimum), maximum))
        except (TypeError, ValueError):
            continue
    return out


def _parse_label_override_map(raw_value: str) -> dict[str, str]:
    """
    Parse label override map from JSON or CSV-like syntax.
    Supports:
    - '{"sleep":0.75,"toilet":0.8}'
    - "sleep:0.75,toilet=0.8"
    """
    txt = str(raw_value or "").strip()
    if not txt:
        return {}
    if txt.startswith("{") and txt.endswith("}"):
        try:
            parsed = json.loads(txt)
            if isinstance(parsed, dict):
                out: dict[str, str] = {}
                for k, v in parsed.items():
                    key = str(k).strip().lower()
                    val = str(v).strip()
                    if key and val:
                        out[key] = val
                return out
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}

    out: dict[str, str] = {}
    for token in txt.split(","):
        item = str(token).strip()
        if not item:
            continue
        if ":" in item:
            key_raw, value_raw = item.split(":", 1)
        elif "=" in item:
            key_raw, value_raw = item.split("=", 1)
        else:
            continue
        key = str(key_raw).strip().lower()
        value = str(value_raw).strip()
        if key and value:
            out[key] = value
    return out


def _parse_lower_token_set(raw_value: str | None, default_csv: str = "") -> set[str]:
    txt = str(raw_value).strip() if raw_value is not None else str(default_csv).strip()
    if not txt:
        return set()
    return {
        str(token).strip().lower()
        for token in txt.split(",")
        if str(token).strip()
    }


@dataclass
class ClinicalPriorityPolicy:
    multipliers: dict[str, float] = field(
        default_factory=get_clinical_priority_multipliers_by_label
    )
    class_weight_cap: float = 8.0
    class_weight_floor: float = 0.25
    multipliers_by_room_label: dict[str, float] = field(default_factory=dict)

    def get_multiplier(self, label_name: str | None) -> float:
        if not label_name:
            return 1.0
        return float(self.multipliers.get(str(label_name).strip().lower(), 1.0))

    def get_room_label_multiplier(self, room_name: str | None, label_name: str | None) -> float:
        if not room_name or not label_name:
            return self.get_multiplier(label_name)
        room_key = normalize_room_name(room_name)
        label_key = str(label_name).strip().lower()
        key = f"{room_key}.{label_key}"
        if key in self.multipliers_by_room_label:
            return float(self.multipliers_by_room_label[key])
        return self.get_multiplier(label_name)


@dataclass
class UnoccupiedDownsamplePolicy:
    min_share: float = 0.45
    stride: int = 10
    boundary_keep: int = 6
    min_run_length: int = 24
    min_share_by_room: dict[str, float] = field(
        default_factory=get_unoccupied_downsample_min_share_by_room
    )
    stride_by_room: dict[str, int] = field(
        default_factory=get_unoccupied_downsample_stride_by_room
    )
    boundary_keep_by_room: dict[str, int] = field(default_factory=dict)
    min_run_length_by_room: dict[str, int] = field(default_factory=dict)

    def resolve(self, room_name: str) -> dict[str, float | int]:
        room_key = normalize_room_name(room_name)
        min_share = float(self.min_share_by_room.get(room_key, self.min_share))
        stride = int(self.stride_by_room.get(room_key, self.stride))
        boundary_keep = int(self.boundary_keep_by_room.get(room_key, self.boundary_keep))
        min_run_length = int(self.min_run_length_by_room.get(room_key, self.min_run_length))
        return {
            "min_share": float(min(max(min_share, 0.0), 1.0)),
            "stride": max(1, stride),
            "boundary_keep": max(0, boundary_keep),
            "min_run_length": max(1, min_run_length),
        }


@dataclass
class MinoritySamplingPolicy:
    enabled: bool = True
    target_share: float = 0.14
    max_multiplier: int = 3
    target_share_by_room: dict[str, float] = field(
        default_factory=get_minority_sampling_target_share_by_room
    )
    max_multiplier_by_room: dict[str, int] = field(
        default_factory=get_minority_sampling_max_multiplier_by_room
    )

    def resolve(self, room_name: str) -> dict[str, float | int | bool]:
        room_key = normalize_room_name(room_name)
        target_share = float(self.target_share_by_room.get(room_key, self.target_share))
        max_multiplier = int(self.max_multiplier_by_room.get(room_key, self.max_multiplier))
        return {
            "enabled": bool(self.enabled),
            "target_share": float(min(max(target_share, 0.0), 0.49)),
            "max_multiplier": max(1, max_multiplier),
        }


@dataclass
class CalibrationPolicy:
    fraction_of_holdout: float = 0.4
    min_samples: int = 80
    separate_calibration_min_holdout: int = 160
    min_support_per_class: int = 30
    threshold_floor: float = 0.35
    threshold_cap: float = 0.80
    default_precision_target: float = 0.70
    precision_targets_by_label: dict[str, float] = field(
        default_factory=get_calibration_precision_targets_by_label
    )
    default_recall_floor: float = 0.08
    recall_floors_by_label: dict[str, float] = field(
        default_factory=get_calibration_recall_floors_by_label
    )

    def get_precision_target(self, label_name: str | None) -> float:
        if not label_name:
            return float(self.default_precision_target)
        return float(self.precision_targets_by_label.get(str(label_name), self.default_precision_target))

    def get_recall_floor(self, label_name: str | None) -> float:
        if not label_name:
            return float(self.default_recall_floor)
        return float(self.recall_floors_by_label.get(str(label_name), self.default_recall_floor))


@dataclass
class ReleaseGatePolicy:
    evidence_profile: str = "production"
    allow_gate_config_fallback_pass: bool = True
    max_drop_from_champion_default: float = 0.05
    min_training_days: float = 2.0
    min_observed_days: int = 2
    min_samples: int = 120
    min_calibration_support: int = 30
    min_validation_class_support: int = 20
    min_retained_sample_ratio: float = 0.10
    max_dropped_ratio: float = 0.90
    block_on_low_support_fallback: bool = True
    block_on_train_fallback_metrics: bool = True
    min_recall_support: int = 30
    min_recall_by_room_label: dict[str, float] = field(default_factory=dict)


@dataclass
class ReproducibilityPolicy:
    random_seed: int = 42
    skip_if_same_data_and_policy: bool = True


@dataclass
class PromotionEligibilityPolicy:
    min_training_days_with_champion: float = 7.0


@dataclass
class ResamplingPolicy:
    max_ffill_gap_seconds: float | None = 60.0

    def resolve(self):
        return self.max_ffill_gap_seconds


@dataclass
class DataViabilityPolicy:
    # Global fallback defaults.
    min_observed_days: int = 7
    min_post_gap_rows: int = 10000
    max_unresolved_drop_ratio: float = 0.85
    min_training_windows: int = 2500
    # RFC table defaults by room.
    min_observed_days_by_room: dict[str, int] = field(
        default_factory=get_data_viability_min_observed_days_by_room
    )
    min_post_gap_rows_by_room: dict[str, int] = field(
        default_factory=get_data_viability_min_post_gap_rows_by_room
    )
    max_unresolved_drop_ratio_by_room: dict[str, float] = field(
        default_factory=get_data_viability_max_unresolved_drop_ratio_by_room
    )
    min_training_windows_by_room: dict[str, int] = field(
        default_factory=get_data_viability_min_training_windows_by_room
    )

    def resolve(self, room_name: str) -> dict[str, int | float]:
        room_key = normalize_room_name(room_name)
        return {
            "min_observed_days": max(
                1, int(self.min_observed_days_by_room.get(room_key, self.min_observed_days))
            ),
            "min_post_gap_rows": max(
                1, int(self.min_post_gap_rows_by_room.get(room_key, self.min_post_gap_rows))
            ),
            "max_unresolved_drop_ratio": float(
                min(
                    max(
                        float(
                            self.max_unresolved_drop_ratio_by_room.get(
                                room_key, self.max_unresolved_drop_ratio
                            )
                        ),
                        0.0,
                    ),
                    1.0,
                )
            ),
            "min_training_windows": max(
                1, int(self.min_training_windows_by_room.get(room_key, self.min_training_windows))
            ),
        }


@dataclass
class TrainingProfilePolicy:
    """Training profile selection: pilot vs production."""
    profile: str = "production"  # "pilot" or "production"
    
    def is_pilot(self) -> bool:
        return self.profile.lower() == "pilot"
    
    def is_production(self) -> bool:
        return self.profile.lower() == "production"


@dataclass
class EventFirstPolicy:
    """
    Event-first shadow/runtime controls (Lane C).
    """
    shadow: bool = False
    enabled: bool = False
    event_registry_path: str = "backend/config/adl_event_registry.v1.yaml"
    unknown_enabled: bool = True
    decoder_on_threshold: float = 0.60
    decoder_off_threshold: float = 0.40
    decoder_min_on_steps: int = 3
    probability_calibration: str = "isotonic"
    calibration_min_samples: int = 500
    home_empty_enabled: bool = True
    home_empty_min_empty_minutes: float = 15.0
    home_empty_empty_score_threshold: float = 0.75
    home_empty_occupancy_threshold: float = 0.55
    home_empty_entrance_penalty: float = 0.35
    unknown_rate_global_cap: float = 0.15
    unknown_rate_room_cap: float = 0.20


@dataclass
class TrainingPolicy:
    clinical_priority: ClinicalPriorityPolicy = field(default_factory=ClinicalPriorityPolicy)
    unoccupied_downsample: UnoccupiedDownsamplePolicy = field(default_factory=UnoccupiedDownsamplePolicy)
    minority_sampling: MinoritySamplingPolicy = field(default_factory=MinoritySamplingPolicy)
    calibration: CalibrationPolicy = field(default_factory=CalibrationPolicy)
    release_gate: ReleaseGatePolicy = field(default_factory=ReleaseGatePolicy)
    resampling: ResamplingPolicy = field(default_factory=ResamplingPolicy)
    data_viability: DataViabilityPolicy = field(default_factory=DataViabilityPolicy)
    reproducibility: ReproducibilityPolicy = field(default_factory=ReproducibilityPolicy)
    promotion_eligibility: PromotionEligibilityPolicy = field(default_factory=PromotionEligibilityPolicy)
    training_profile: TrainingProfilePolicy = field(default_factory=TrainingProfilePolicy)
    event_first: EventFirstPolicy = field(default_factory=EventFirstPolicy)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    def get_profile_name(self) -> str:
        """Return the active training profile name."""
        return self.training_profile.profile


def load_policy_from_env(environ: Mapping[str, str] | None = None) -> TrainingPolicy:
    """
    Build TrainingPolicy from environment variables with backward-compatible names.
    """
    env = os.environ if environ is None else environ
    policy = TrainingPolicy()
    release_gate_profile = _resolve_release_gate_profile(env, policy.release_gate.evidence_profile)
    pilot_relaxed_evidence = release_gate_profile in {"pilot_stage_a", "pilot_stage_b"}

    # Clinical priority + class weight clipping.
    policy.clinical_priority.class_weight_cap = _read_float_env(
        env, "CLASS_WEIGHT_CAP", policy.clinical_priority.class_weight_cap, minimum=0.01, maximum=100.0
    )
    policy.clinical_priority.class_weight_floor = _read_float_env(
        env, "CLASS_WEIGHT_FLOOR", policy.clinical_priority.class_weight_floor, minimum=0.0, maximum=100.0
    )
    raw_clinical_map = env.get("CLINICAL_PRIORITY_MULTIPLIERS")
    if raw_clinical_map:
        parsed = _parse_label_override_map(raw_clinical_map)
        for label, value in parsed.items():
            try:
                policy.clinical_priority.multipliers[label] = float(value)
            except (TypeError, ValueError):
                continue
    policy.clinical_priority.multipliers_by_room_label = _read_room_label_float_overrides(
        env,
        "CLINICAL_PRIORITY_MULTIPLIERS_BY_ROOM_LABEL",
        policy.clinical_priority.multipliers_by_room_label,
        minimum=0.0,
        maximum=100.0,
    )
    # Pilot profiles run on short-window evidence. Keep weights neutral by
    # default, but preserve bedroom sleep-critical labels so checkpoint/gate
    # selection can still optimize sleep-duration recall.
    if pilot_relaxed_evidence:
        preserved_labels = _parse_lower_token_set(
            env.get("PILOT_STAGE_CLINICAL_PRIORITY_LABELS"),
            default_csv="sleep,sleep_duration",
        )
        preserved_room_labels = _parse_lower_token_set(
            env.get("PILOT_STAGE_CLINICAL_PRIORITY_ROOM_LABELS"),
            default_csv="bedroom.sleep,bedroom.sleep_duration",
        )
        policy.clinical_priority.multipliers = {
            str(label).strip().lower(): (
                float(value)
                if str(label).strip().lower() in preserved_labels
                else 1.0
            )
            for label, value in policy.clinical_priority.multipliers.items()
            if str(label).strip()
        }
        policy.clinical_priority.multipliers_by_room_label = {
            str(key).strip().lower(): (
                float(value)
                if str(key).strip().lower() in preserved_room_labels
                else 1.0
            )
            for key, value in policy.clinical_priority.multipliers_by_room_label.items()
            if str(key).strip()
        }

    # Unoccupied downsampling.
    unocc = policy.unoccupied_downsample
    unocc.min_share = _read_float_env(env, "UNOCCUPIED_DOWNSAMPLE_MIN_SHARE", unocc.min_share, 0.0, 1.0)
    unocc.stride = _read_int_env(env, "UNOCCUPIED_DOWNSAMPLE_STRIDE", unocc.stride, minimum=1)
    unocc.boundary_keep = _read_int_env(env, "UNOCCUPIED_BOUNDARY_KEEP", unocc.boundary_keep, minimum=0)
    unocc.min_run_length = _read_int_env(env, "UNOCCUPIED_MIN_RUN_LENGTH", unocc.min_run_length, minimum=1)

    raw_min_share_map = env.get("UNOCCUPIED_DOWNSAMPLE_MIN_SHARE_BY_ROOM")
    if raw_min_share_map is not None:
        unocc.min_share_by_room = {}
        parsed = _parse_override_map(raw_min_share_map)
        for room, value in parsed.items():
            try:
                unocc.min_share_by_room[room] = float(min(max(float(value), 0.0), 1.0))
            except (TypeError, ValueError):
                continue
    raw_stride_map = env.get("UNOCCUPIED_DOWNSAMPLE_STRIDE_BY_ROOM")
    if raw_stride_map is not None:
        unocc.stride_by_room = {}
        parsed = _parse_override_map(raw_stride_map)
        for room, value in parsed.items():
            try:
                unocc.stride_by_room[room] = max(1, int(value))
            except (TypeError, ValueError):
                continue
    raw_boundary_map = env.get("UNOCCUPIED_BOUNDARY_KEEP_BY_ROOM")
    if raw_boundary_map is not None:
        unocc.boundary_keep_by_room = {}
        parsed = _parse_override_map(raw_boundary_map)
        for room, value in parsed.items():
            try:
                unocc.boundary_keep_by_room[room] = max(0, int(value))
            except (TypeError, ValueError):
                continue
    raw_min_run_map = env.get("UNOCCUPIED_MIN_RUN_LENGTH_BY_ROOM")
    if raw_min_run_map is not None:
        unocc.min_run_length_by_room = {}
        parsed = _parse_override_map(raw_min_run_map)
        for room, value in parsed.items():
            try:
                unocc.min_run_length_by_room[room] = max(1, int(value))
            except (TypeError, ValueError):
                continue

    # Minority sampling.
    minority = policy.minority_sampling
    minority.enabled = _is_truthy(env.get("ENABLE_MINORITY_CLASS_SAMPLING", str(minority.enabled)))
    minority.target_share = _read_float_env(env, "MINORITY_TARGET_SHARE", minority.target_share, 0.0, 0.49)
    minority.max_multiplier = _read_int_env(env, "MINORITY_MAX_MULTIPLIER", minority.max_multiplier, minimum=1)

    raw_target_share_map = env.get("MINORITY_TARGET_SHARE_BY_ROOM")
    if raw_target_share_map is not None:
        minority.target_share_by_room = {}
        parsed = _parse_override_map(raw_target_share_map)
        for room, value in parsed.items():
            try:
                minority.target_share_by_room[room] = float(min(max(float(value), 0.0), 0.49))
            except (TypeError, ValueError):
                continue
    raw_max_multiplier_map = env.get("MINORITY_MAX_MULTIPLIER_BY_ROOM")
    if raw_max_multiplier_map is not None:
        minority.max_multiplier_by_room = {}
        parsed = _parse_override_map(raw_max_multiplier_map)
        for room, value in parsed.items():
            try:
                minority.max_multiplier_by_room[room] = max(1, int(value))
            except (TypeError, ValueError):
                continue

    # Calibration / thresholding.
    calib = policy.calibration
    calib.fraction_of_holdout = _read_float_env(
        env,
        "CALIBRATION_FRACTION_OF_HOLDOUT",
        calib.fraction_of_holdout,
        minimum=0.05,
        maximum=0.95,
    )
    calib.min_samples = _read_int_env(env, "CALIBRATION_MIN_SAMPLES", calib.min_samples, minimum=1)
    calib.separate_calibration_min_holdout = _read_int_env(
        env,
        "SEPARATE_CALIBRATION_MIN_HOLDOUT",
        calib.separate_calibration_min_holdout,
        minimum=2,
    )
    calib.min_support_per_class = _read_int_env(
        env,
        "CALIBRATION_MIN_SUPPORT_PER_CLASS",
        calib.min_support_per_class,
        minimum=1,
    )
    calib.threshold_floor = _read_float_env(env, "THRESHOLD_FLOOR", calib.threshold_floor, minimum=0.0, maximum=1.0)
    calib.threshold_cap = _read_float_env(env, "THRESHOLD_CAP", calib.threshold_cap, minimum=0.0, maximum=1.0)
    if calib.threshold_floor > calib.threshold_cap:
        calib.threshold_floor, calib.threshold_cap = calib.threshold_cap, calib.threshold_floor

    calib.default_precision_target = _read_float_env(
        env,
        "DEFAULT_PRECISION_TARGET",
        calib.default_precision_target,
        minimum=0.0,
        maximum=1.0,
    )
    calib.default_recall_floor = _read_float_env(
        env,
        "DEFAULT_RECALL_FLOOR",
        calib.default_recall_floor,
        minimum=0.0,
        maximum=1.0,
    )

    raw_precision_map = env.get("PRECISION_TARGETS_BY_LABEL")
    if raw_precision_map:
        parsed = _parse_label_override_map(raw_precision_map)
        for label, value in parsed.items():
            try:
                calib.precision_targets_by_label[label] = float(min(max(float(value), 0.0), 1.0))
            except (TypeError, ValueError):
                continue

    raw_recall_floor_map = env.get("RECALL_FLOOR_BY_LABEL")
    if raw_recall_floor_map:
        parsed = _parse_label_override_map(raw_recall_floor_map)
        for label, value in parsed.items():
            try:
                calib.recall_floors_by_label[label] = float(min(max(float(value), 0.0), 1.0))
            except (TypeError, ValueError):
                continue

    # Resampling policy.
    default_gap = policy.resampling.max_ffill_gap_seconds
    raw_gap = env.get("MAX_RESAMPLE_FFILL_GAP_SECONDS", default_gap)
    policy.resampling.max_ffill_gap_seconds = resolve_max_ffill_gap_seconds(
        raw_value=raw_gap,
        default_seconds=60.0 if default_gap is None else float(default_gap),
    )

    # Release-gate hard evidence floors.
    release_gate = policy.release_gate
    profile_defaults = _RELEASE_GATE_EVIDENCE_PROFILES[release_gate_profile]
    release_gate.evidence_profile = release_gate_profile
    release_gate.min_validation_class_support = int(
        profile_defaults.get("min_validation_class_support", release_gate.min_validation_class_support)
    )
    release_gate.min_recall_support = int(
        profile_defaults.get("min_recall_support", release_gate.min_recall_support)
    )

    release_gate.min_training_days = _read_float_env(
        env,
        "RELEASE_GATE_MIN_TRAINING_DAYS",
        release_gate.min_training_days,
        minimum=0.0,
        maximum=3650.0,
    )
    release_gate.min_samples = _read_int_env(
        env,
        "RELEASE_GATE_MIN_SAMPLES",
        release_gate.min_samples,
        minimum=1,
    )
    release_gate.min_calibration_support = _read_int_env(
        env,
        "RELEASE_GATE_MIN_CALIBRATION_SUPPORT",
        release_gate.min_calibration_support,
        minimum=1,
    )
    release_gate.min_validation_class_support = _read_int_env(
        env,
        "RELEASE_GATE_MIN_VALIDATION_CLASS_SUPPORT",
        release_gate.min_validation_class_support,
        minimum=1,
    )
    release_gate.min_observed_days = _read_int_env(
        env,
        "RELEASE_GATE_MIN_OBSERVED_DAYS",
        release_gate.min_observed_days,
        minimum=1,
    )
    release_gate.min_retained_sample_ratio = _read_float_env(
        env,
        "RELEASE_GATE_MIN_RETAINED_SAMPLE_RATIO",
        release_gate.min_retained_sample_ratio,
        minimum=0.0,
        maximum=1.0,
    )
    release_gate.max_dropped_ratio = _read_float_env(
        env,
        "RELEASE_GATE_MAX_DROPPED_RATIO",
        release_gate.max_dropped_ratio,
        minimum=0.0,
        maximum=1.0,
    )
    release_gate.allow_gate_config_fallback_pass = _is_truthy(
        env.get(
            "ALLOW_GATE_CONFIG_FALLBACK_PASS",
            str(release_gate.allow_gate_config_fallback_pass),
        )
    )
    release_gate.block_on_low_support_fallback = _is_truthy(
        env.get(
            "RELEASE_GATE_BLOCK_ON_LOW_SUPPORT_FALLBACK",
            str(release_gate.block_on_low_support_fallback),
        )
    )
    release_gate.block_on_train_fallback_metrics = _is_truthy(
        env.get(
            "RELEASE_GATE_BLOCK_ON_TRAIN_FALLBACK_METRICS",
            str(release_gate.block_on_train_fallback_metrics),
        )
    )
    release_gate.min_recall_support = _read_int_env(
        env,
        "RELEASE_GATE_MIN_RECALL_SUPPORT",
        release_gate.min_recall_support,
        minimum=1,
    )
    release_gate.min_recall_by_room_label = _read_room_label_float_overrides(
        env,
        "RELEASE_GATE_MIN_RECALL_BY_ROOM_LABEL",
        release_gate.min_recall_by_room_label,
        minimum=0.0,
        maximum=1.0,
    )

    # Data viability gate (room-specific RFC thresholds).
    viability = policy.data_viability
    viability.min_observed_days = _read_int_env(
        env,
        "DATA_VIABILITY_MIN_OBSERVED_DAYS",
        viability.min_observed_days,
        minimum=1,
    )
    viability.min_post_gap_rows = _read_int_env(
        env,
        "DATA_VIABILITY_MIN_POST_GAP_ROWS",
        viability.min_post_gap_rows,
        minimum=1,
    )
    viability.max_unresolved_drop_ratio = _read_float_env(
        env,
        "DATA_VIABILITY_MAX_UNRESOLVED_DROP_RATIO",
        viability.max_unresolved_drop_ratio,
        minimum=0.0,
        maximum=1.0,
    )
    viability.min_training_windows = _read_int_env(
        env,
        "DATA_VIABILITY_MIN_TRAINING_WINDOWS",
        viability.min_training_windows,
        minimum=1,
    )
    viability.min_observed_days_by_room = _read_room_int_overrides(
        env,
        "DATA_VIABILITY_MIN_OBSERVED_DAYS_BY_ROOM",
        viability.min_observed_days_by_room,
        minimum=1,
    )
    viability.min_post_gap_rows_by_room = _read_room_int_overrides(
        env,
        "DATA_VIABILITY_MIN_POST_GAP_ROWS_BY_ROOM",
        viability.min_post_gap_rows_by_room,
        minimum=1,
    )
    viability.max_unresolved_drop_ratio_by_room = _read_room_float_overrides(
        env,
        "DATA_VIABILITY_MAX_UNRESOLVED_DROP_RATIO_BY_ROOM",
        viability.max_unresolved_drop_ratio_by_room,
        minimum=0.0,
        maximum=1.0,
    )
    viability.min_training_windows_by_room = _read_room_int_overrides(
        env,
        "DATA_VIABILITY_MIN_TRAINING_WINDOWS_BY_ROOM",
        viability.min_training_windows_by_room,
        minimum=1,
    )

    # Reproducibility controls.
    repro = policy.reproducibility
    repro.random_seed = _read_int_env(
        env,
        "TRAINING_RANDOM_SEED",
        repro.random_seed,
        minimum=0,
    )
    repro.skip_if_same_data_and_policy = _is_truthy(
        env.get(
            "SKIP_RETRAIN_IF_SAME_DATA_AND_POLICY",
            str(repro.skip_if_same_data_and_policy),
        )
    )

    # Promotion-eligibility controls.
    promo = policy.promotion_eligibility
    promo.min_training_days_with_champion = _read_float_env(
        env,
        "PROMOTION_MIN_TRAINING_DAYS_WITH_CHAMPION",
        promo.min_training_days_with_champion,
        minimum=0.0,
        maximum=3650.0,
    )

    # Event-first controls (Lane C).
    event_first = policy.event_first
    event_first.shadow = _is_truthy(env.get("EVENT_FIRST_SHADOW", str(event_first.shadow)))
    event_first.enabled = _is_truthy(env.get("EVENT_FIRST_ENABLED", str(event_first.enabled)))
    event_first.event_registry_path = str(
        env.get("EVENT_REGISTRY_PATH", event_first.event_registry_path)
    ).strip() or event_first.event_registry_path
    event_first.unknown_enabled = _is_truthy(
        env.get("EVENT_UNKNOWN_ENABLED", str(event_first.unknown_enabled))
    )
    event_first.decoder_on_threshold = _read_float_env(
        env,
        "EVENT_DECODER_ON_THRESHOLD",
        event_first.decoder_on_threshold,
        minimum=0.0,
        maximum=1.0,
    )
    event_first.decoder_off_threshold = _read_float_env(
        env,
        "EVENT_DECODER_OFF_THRESHOLD",
        event_first.decoder_off_threshold,
        minimum=0.0,
        maximum=1.0,
    )
    if event_first.decoder_off_threshold >= event_first.decoder_on_threshold:
        event_first.decoder_off_threshold = max(0.0, event_first.decoder_on_threshold - 0.05)
    event_first.decoder_min_on_steps = _read_int_env(
        env,
        "EVENT_DECODER_MIN_ON_STEPS",
        event_first.decoder_min_on_steps,
        minimum=1,
    )
    calibration_mode = str(
        env.get("EVENT_PROBABILITY_CALIBRATION", event_first.probability_calibration)
    ).strip().lower()
    if calibration_mode not in {"isotonic", "platt", "temperature"}:
        calibration_mode = "isotonic"
    event_first.probability_calibration = calibration_mode
    event_first.calibration_min_samples = _read_int_env(
        env,
        "EVENT_CALIBRATION_MIN_SAMPLES",
        event_first.calibration_min_samples,
        minimum=1,
    )
    event_first.home_empty_enabled = _is_truthy(
        env.get("HOME_EMPTY_ENABLED", str(event_first.home_empty_enabled))
    )
    event_first.home_empty_min_empty_minutes = _read_float_env(
        env,
        "HOME_EMPTY_MIN_EMPTY_MINUTES",
        event_first.home_empty_min_empty_minutes,
        minimum=0.0,
        maximum=1440.0,
    )
    event_first.home_empty_empty_score_threshold = _read_float_env(
        env,
        "HOME_EMPTY_EMPTY_SCORE_THRESHOLD",
        event_first.home_empty_empty_score_threshold,
        minimum=0.0,
        maximum=1.0,
    )
    event_first.home_empty_occupancy_threshold = _read_float_env(
        env,
        "HOME_EMPTY_OCCUPANCY_THRESHOLD",
        event_first.home_empty_occupancy_threshold,
        minimum=0.0,
        maximum=1.0,
    )
    event_first.home_empty_entrance_penalty = _read_float_env(
        env,
        "HOME_EMPTY_ENTRANCE_PENALTY",
        event_first.home_empty_entrance_penalty,
        minimum=0.0,
        maximum=1.0,
    )
    event_first.unknown_rate_global_cap = _read_float_env(
        env,
        "UNKNOWN_RATE_GLOBAL_CAP",
        event_first.unknown_rate_global_cap,
        minimum=0.0,
        maximum=1.0,
    )
    event_first.unknown_rate_room_cap = _read_float_env(
        env,
        "UNKNOWN_RATE_ROOM_CAP",
        event_first.unknown_rate_room_cap,
        minimum=0.0,
        maximum=1.0,
    )
    # Merge config-file defaults for event-first (env still has priority).
    try:
        from config import get_release_gates_config

        event_cfg = (get_release_gates_config() or {}).get("event_first", {})
        if isinstance(event_cfg, dict):
            if env.get("EVENT_FIRST_SHADOW") is None and "event_first_shadow" in event_cfg:
                event_first.shadow = bool(event_cfg.get("event_first_shadow"))
            if env.get("EVENT_FIRST_ENABLED") is None and "event_first_enabled" in event_cfg:
                event_first.enabled = bool(event_cfg.get("event_first_enabled"))
            if env.get("EVENT_REGISTRY_PATH") is None and "event_registry_path" in event_cfg:
                event_first.event_registry_path = str(
                    event_cfg.get("event_registry_path", event_first.event_registry_path)
                ).strip() or event_first.event_registry_path
            if env.get("EVENT_UNKNOWN_ENABLED") is None and "event_unknown_enabled" in event_cfg:
                event_first.unknown_enabled = bool(event_cfg.get("event_unknown_enabled"))
            if env.get("EVENT_DECODER_ON_THRESHOLD") is None and "event_decoder_on_threshold" in event_cfg:
                event_first.decoder_on_threshold = float(
                    min(max(float(event_cfg.get("event_decoder_on_threshold")), 0.0), 1.0)
                )
            if env.get("EVENT_DECODER_OFF_THRESHOLD") is None and "event_decoder_off_threshold" in event_cfg:
                event_first.decoder_off_threshold = float(
                    min(max(float(event_cfg.get("event_decoder_off_threshold")), 0.0), 1.0)
                )
            if event_first.decoder_off_threshold >= event_first.decoder_on_threshold:
                event_first.decoder_off_threshold = max(0.0, event_first.decoder_on_threshold - 0.05)
            if env.get("EVENT_DECODER_MIN_ON_STEPS") is None and "event_decoder_min_on_steps" in event_cfg:
                event_first.decoder_min_on_steps = max(
                    1, int(event_cfg.get("event_decoder_min_on_steps"))
                )
            if (
                env.get("EVENT_PROBABILITY_CALIBRATION") is None
                and "event_probability_calibration" in event_cfg
            ):
                mode = str(event_cfg.get("event_probability_calibration")).strip().lower()
                if mode in {"isotonic", "platt", "temperature"}:
                    event_first.probability_calibration = mode
            if env.get("EVENT_CALIBRATION_MIN_SAMPLES") is None and "event_calibration_min_samples" in event_cfg:
                event_first.calibration_min_samples = max(
                    1, int(event_cfg.get("event_calibration_min_samples"))
                )
            if env.get("HOME_EMPTY_ENABLED") is None and "home_empty_enabled" in event_cfg:
                event_first.home_empty_enabled = bool(event_cfg.get("home_empty_enabled"))
            if env.get("HOME_EMPTY_MIN_EMPTY_MINUTES") is None and "home_empty_min_empty_minutes" in event_cfg:
                event_first.home_empty_min_empty_minutes = float(
                    max(float(event_cfg.get("home_empty_min_empty_minutes")), 0.0)
                )
            if env.get("HOME_EMPTY_EMPTY_SCORE_THRESHOLD") is None and "home_empty_empty_score_threshold" in event_cfg:
                event_first.home_empty_empty_score_threshold = float(
                    min(max(float(event_cfg.get("home_empty_empty_score_threshold")), 0.0), 1.0)
                )
            if env.get("HOME_EMPTY_OCCUPANCY_THRESHOLD") is None and "home_empty_occupancy_threshold" in event_cfg:
                event_first.home_empty_occupancy_threshold = float(
                    min(max(float(event_cfg.get("home_empty_occupancy_threshold")), 0.0), 1.0)
                )
            if env.get("HOME_EMPTY_ENTRANCE_PENALTY") is None and "home_empty_entrance_penalty" in event_cfg:
                event_first.home_empty_entrance_penalty = float(
                    min(max(float(event_cfg.get("home_empty_entrance_penalty")), 0.0), 1.0)
                )
            if env.get("UNKNOWN_RATE_GLOBAL_CAP") is None and "unknown_rate_global_cap" in event_cfg:
                event_first.unknown_rate_global_cap = float(
                    min(max(float(event_cfg.get("unknown_rate_global_cap")), 0.0), 1.0)
                )
            if env.get("UNKNOWN_RATE_ROOM_CAP") is None and "unknown_rate_room_cap" in event_cfg:
                event_first.unknown_rate_room_cap = float(
                    min(max(float(event_cfg.get("unknown_rate_room_cap")), 0.0), 1.0)
                )
    except Exception:
        # Release-gate config may be unavailable in isolated tests.
        pass

    # Training profile (pilot vs production).
    profile_policy = policy.training_profile
    profile_value = env.get("TRAINING_PROFILE", profile_policy.profile).strip().lower()
    if profile_value in ("pilot", "production"):
        profile_policy.profile = profile_value
    else:
        logger.warning(f"Invalid TRAINING_PROFILE='{profile_value}', defaulting to 'production'")
        profile_policy.profile = "production"
    
    # Apply profile-specific overrides.
    if profile_policy.is_pilot():
        # Pilot mode: relaxed thresholds for rapid iteration.
        _apply_pilot_thresholds(policy)
    
    return policy


def _apply_pilot_thresholds(policy: TrainingPolicy):
    """
    Apply relaxed thresholds for pilot/training mode.
    
    This allows faster iteration with lower data requirements,
    but models trained in pilot mode should NOT be promoted to production.
    """
    logger.info("Applying PILOT mode thresholds (relaxed for rapid iteration)")
    
    # Data viability: lower minimums.
    viability = policy.data_viability
    viability.min_observed_days = min(viability.min_observed_days, 3)
    viability.min_post_gap_rows = min(viability.min_post_gap_rows, 5000)
    viability.min_training_windows = min(viability.min_training_windows, 1000)
    
    # Release gates: lower evidence floors.
    gate = policy.release_gate
    gate.min_training_days = min(gate.min_training_days, 1.0)
    gate.min_samples = min(gate.min_samples, 50)
    gate.min_observed_days = min(gate.min_observed_days, 3)
    
    # Promotion: still require 7 days with champion (safety).
    # But allow faster promotion for new rooms.
    promo = policy.promotion_eligibility
    promo.min_training_days_with_champion = min(promo.min_training_days_with_champion, 7.0)
