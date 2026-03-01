"""
Item 16: Policy Complexity Governance

Provides bounded presets (conservative, balanced, aggressive) as top-level
policy modes to reduce heuristic overload risk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Mapping, Tuple
import os

from ml.policy_config import TrainingPolicy

logger = logging.getLogger(__name__)


class PolicyPreset(Enum):
    """Pre-defined policy presets."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class PresetDefinition:
    """Definition of a policy preset."""
    name: str
    description: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    allowed_overrides: Set[str] = field(default_factory=set)
    blocked_overrides: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "overrides": self.overrides,
            "allowed_overrides": sorted(list(self.allowed_overrides)),
            "blocked_overrides": sorted(list(self.blocked_overrides)),
        }


@dataclass(frozen=True)
class RolloutLadderRung:
    """Single rollout ladder rung policy."""

    rung: int
    users: int
    min_days: int
    requires_phase5_acceptance: bool = False


@dataclass(frozen=True)
class RolloutProgressionCriteria:
    """Rung progression gate policy."""

    all_mandatory_metric_floors_pass: bool = True
    block_on_open_p0_incidents: bool = True
    min_nightly_pipeline_success_rate: float = 0.99
    require_drift_alerts_within_budget: bool = True
    require_timeline_hard_gates_all_rooms: bool = True


@dataclass(frozen=True)
class RolloutAutoRollbackPolicy:
    """Automatic rollback trigger policy."""

    consecutive_nights: int = 2
    mae_regression_pct_gt: float = 0.10
    alert_precision_below_threshold: bool = True
    worst_room_f1_below_floor: bool = True
    pipeline_success_rate_lt: float = 0.97


@dataclass(frozen=True)
class RolloutFallbackPolicy:
    """Operator-safe fallback behavior policy."""

    baseline_profile: str = "production"
    serving_mode: str = "rule_hmm_baseline"
    require_registry_event: bool = True
    require_ops_alert: bool = True
    recovery_consecutive_pass_nights: int = 2


@dataclass(frozen=True)
class RolloutLadderPolicy:
    """Full rollout ladder policy loaded from config."""

    rungs: tuple[RolloutLadderRung, ...]
    progression: RolloutProgressionCriteria
    auto_rollback: RolloutAutoRollbackPolicy
    fallback: RolloutFallbackPolicy

    def get_rung(self, rung: int) -> Optional[RolloutLadderRung]:
        for entry in self.rungs:
            if int(entry.rung) == int(rung):
                return entry
        return None


class PolicyPresetManager:
    """
    Manager for policy presets.
    
    Reduces heuristic overload by providing bounded presets
    and controlling which knobs can be manually adjusted.
    """
    
    # Preset definitions
    PRESETS = {
        PolicyPreset.CONSERVATIVE: PresetDefinition(
            name="conservative",
            description=(
                "Conservative preset prioritizes stability and safety. "
                "Higher thresholds, more data requirements, stricter gates. "
                "Recommended for production environments."
            ),
            overrides={
                # Data viability
                "data_viability.min_observed_days": 7,
                "data_viability.min_post_gap_rows": 10000,
                "data_viability.min_training_windows": 5000,
                
                # Release gates
                "release_gate.min_training_days": 3.0,
                "release_gate.min_samples": 500,
                "release_gate.min_retained_sample_ratio": 0.7,
                
                # Calibration
                "calibration.min_samples": 100,
                "calibration.min_support_per_class": 20,
                
                # Class weights
                "clinical_priority.class_weight_cap": 5.0,
                "clinical_priority.class_weight_floor": 0.2,
                
                # Training
                "reproducibility.random_seed": 42,
            },
            allowed_overrides={
                # Production-safe overrides
                "TRAINING_PROFILE",
                "CLASS_WEIGHT_CAP",
                "RANDOM_SEED",
                "ENABLE_TRAIN_SPLIT_SCALING",
            },
            blocked_overrides={
                # Never allow these in production
                "MIN_OBSERVED_DAYS",
                "MIN_TRAINING_DAYS",
                "MIN_SAMPLES",
            },
        ),
        
        PolicyPreset.BALANCED: PresetDefinition(
            name="balanced",
            description=(
                "Balanced preset provides reasonable defaults for most scenarios. "
                "Moderate thresholds and requirements. "
                "Recommended for general use."
            ),
            overrides={
                # Data viability
                "data_viability.min_observed_days": 5,
                "data_viability.min_post_gap_rows": 5000,
                "data_viability.min_training_windows": 2000,
                
                # Release gates
                "release_gate.min_training_days": 2.0,
                "release_gate.min_samples": 200,
                "release_gate.min_retained_sample_ratio": 0.5,
                
                # Calibration
                "calibration.min_samples": 50,
                "calibration.min_support_per_class": 10,
                
                # Class weights
                "clinical_priority.class_weight_cap": 10.0,
                "clinical_priority.class_weight_floor": 0.1,
                
                # Training
                "reproducibility.random_seed": 42,
            },
            allowed_overrides={
                # More flexibility in balanced mode
                "TRAINING_PROFILE",
                "CLASS_WEIGHT_CAP",
                "CLASS_WEIGHT_FLOOR",
                "RANDOM_SEED",
                "ENABLE_TRAIN_SPLIT_SCALING",
                "MIN_OBSERVED_DAYS",
                "MIN_TRAINING_DAYS",
            },
            blocked_overrides={
                # Still block critical safety params
                "MIN_CALIBRATION_SUPPORT",
            },
        ),
        
        PolicyPreset.AGGRESSIVE: PresetDefinition(
            name="aggressive",
            description=(
                "Aggressive preset prioritizes iteration speed over safety. "
                "Lower thresholds, fewer requirements, permissive gates. "
                "Recommended for rapid prototyping only. "
                "NOT FOR PRODUCTION USE."
            ),
            overrides={
                # Data viability
                "data_viability.min_observed_days": 2,
                "data_viability.min_post_gap_rows": 1000,
                "data_viability.min_training_windows": 500,
                
                # Release gates
                "release_gate.min_training_days": 1.0,
                "release_gate.min_samples": 50,
                "release_gate.min_retained_sample_ratio": 0.3,
                
                # Calibration
                "calibration.min_samples": 20,
                "calibration.min_support_per_class": 5,
                
                # Class weights
                "clinical_priority.class_weight_cap": 20.0,
                "clinical_priority.class_weight_floor": 0.05,
                
                # Training
                "reproducibility.random_seed": 42,
            },
            allowed_overrides={
                # Maximum flexibility (but still logged)
                "TRAINING_PROFILE",
                "CLASS_WEIGHT_CAP",
                "CLASS_WEIGHT_FLOOR",
                "RANDOM_SEED",
                "ENABLE_TRAIN_SPLIT_SCALING",
                "MIN_OBSERVED_DAYS",
                "MIN_TRAINING_DAYS",
                "MIN_SAMPLES",
                "MIN_CALIBRATION_SUPPORT",
            },
            blocked_overrides=set(),  # Allow everything (with warnings)
        ),
    }
    
    def __init__(self, environment: str = "production"):
        """
        Initialize preset manager.
        
        Parameters:
        -----------
        environment : str
            'production', 'staging', or 'development'
        """
        self.environment = environment
        
        # In production, restrict to conservative/balanced only
        if environment == "production":
            self.allowed_presets = {
                PolicyPreset.CONSERVATIVE,
                PolicyPreset.BALANCED,
            }
        else:
            self.allowed_presets = set(PolicyPreset)
    
    def apply_preset(
        self,
        policy: TrainingPolicy,
        preset: PolicyPreset,
        custom_overrides: Optional[Dict[str, Any]] = None,
    ) -> TrainingPolicy:
        """
        Apply a preset to a policy.
        
        Parameters:
        -----------
        policy : TrainingPolicy
            Base policy to modify
        preset : PolicyPreset
            Preset to apply
        custom_overrides : Dict, optional
            Custom environment variable overrides
            
        Returns:
        --------
        TrainingPolicy
            Modified policy
        """
        if preset not in self.allowed_presets:
            raise ValueError(
                f"Preset '{preset.value}' not allowed in {self.environment} environment. "
                f"Allowed: {[p.value for p in self.allowed_presets]}"
            )
        
        definition = self.PRESETS[preset]
        
        logger.info(f"Applying preset: {preset.value}")
        logger.info(f"  Description: {definition.description}")
        
        # Apply preset overrides
        for key, value in definition.overrides.items():
            self._set_nested_attr(policy, key, value)
            logger.debug(f"  Set {key} = {value}")
        
        # Validate and apply custom overrides
        if custom_overrides:
            self._validate_custom_overrides(definition, custom_overrides)
            for key, value in custom_overrides.items():
                self._set_nested_attr(policy, key, value)
                logger.info(f"  Custom override: {key} = {value}")
        
        # Mark preset in policy
        policy._applied_preset = preset.value
        
        return policy
    
    def validate_environment_overrides(
        self,
        preset: PolicyPreset,
        environ: Dict[str, str],
    ) -> Tuple[bool, List[str]]:
        """
        Validate environment variable overrides against preset constraints.
        
        Parameters:
        -----------
        preset : PolicyPreset
            Applied preset
        environ : Dict[str, str]
            Environment variables
            
        Returns:
        --------
        (valid, violations) : Tuple
            valid: True if all overrides allowed
            violations: List of violation messages
        """
        definition = self.PRESETS[preset]
        violations = []
        
        for env_var in environ.keys():
            # Check if this is a policy-related env var
            if not env_var.startswith(("MIN_", "MAX_", "CLASS_WEIGHT_", "CALIBRATION_")):
                continue
            
            # Check if blocked
            if env_var in definition.blocked_overrides:
                violations.append(
                    f"Environment variable '{env_var}' is blocked in '{preset.value}' preset"
                )
            
            # Check if not in allowed list (warning only in non-production)
            elif env_var not in definition.allowed_overrides:
                if self.environment == "production":
                    violations.append(
                        f"Environment variable '{env_var}' not in allowed list for '{preset.value}' preset"
                    )
                else:
                    logger.warning(
                        f"Non-standard override '{env_var}' used with '{preset.value}' preset"
                    )
        
        return len(violations) == 0, violations
    
    def get_preset_info(self, preset: PolicyPreset) -> Dict[str, Any]:
        """Get information about a preset."""
        definition = self.PRESETS[preset]
        return {
            "preset": preset.value,
            "allowed_in_environment": preset in self.allowed_presets,
            **definition.to_dict(),
        }
    
    def list_presets(self) -> List[Dict[str, Any]]:
        """List all available presets."""
        return [
            self.get_preset_info(preset)
            for preset in PolicyPreset
        ]
    
    def _set_nested_attr(self, obj: Any, key: str, value: Any) -> None:
        """Set a nested attribute by dot-notation key."""
        parts = key.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    
    def _validate_custom_overrides(
        self,
        definition: PresetDefinition,
        custom_overrides: Dict[str, Any],
    ) -> None:
        """Validate custom overrides against preset constraints."""
        for key in custom_overrides.keys():
            env_var = key.upper().replace(".", "_")
            
            if env_var in definition.blocked_overrides:
                raise ValueError(
                    f"Override '{key}' is blocked in '{definition.name}' preset"
                )


def load_policy_with_preset(
    preset_name: Optional[str] = None,
    environment: Optional[str] = None,
) -> TrainingPolicy:
    """
    Load policy with preset applied.
    
    Parameters:
    -----------
    preset_name : str, optional
        Preset name ('conservative', 'balanced', 'aggressive')
        Defaults to POLICY_PRESET env var or 'balanced'
    environment : str, optional
        Environment name ('production', 'staging', 'development')
        Defaults to DEPLOYMENT_ENV env var or 'production'
        
    Returns:
    --------
    TrainingPolicy
        Policy with preset applied
    """
    from ml.policy_config import load_policy_from_env
    
    # Determine environment
    if environment is None:
        environment = os.getenv("DEPLOYMENT_ENV", "production").lower()
    
    # Determine preset
    if preset_name is None:
        preset_name = os.getenv("POLICY_PRESET", "balanced").lower()
    
    try:
        preset = PolicyPreset(preset_name)
    except ValueError:
        logger.warning(f"Unknown preset '{preset_name}', using 'balanced'")
        preset = PolicyPreset.BALANCED
    
    # Load base policy
    policy = load_policy_from_env()
    
    # Apply preset
    manager = PolicyPresetManager(environment)
    policy = manager.apply_preset(policy, preset)
    
    # Validate environment overrides
    valid, violations = manager.validate_environment_overrides(preset, dict(os.environ))
    if not valid:
        for v in violations:
            logger.error(f"Policy override violation: {v}")
        if environment == "production":
            raise ValueError("Invalid policy overrides in production environment")
    
    return policy


# Convenience functions for ops
def get_current_preset_info() -> Dict[str, Any]:
    """Get information about the currently configured preset."""
    preset_name = os.getenv("POLICY_PRESET", "balanced")
    environment = os.getenv("DEPLOYMENT_ENV", "production")
    
    try:
        preset = PolicyPreset(preset_name)
    except ValueError:
        return {"error": f"Unknown preset: {preset_name}"}
    
    manager = PolicyPresetManager(environment)
    return manager.get_preset_info(preset)


def validate_policy_overrides() -> Tuple[bool, List[str]]:
    """Validate current environment overrides against preset."""
    preset_name = os.getenv("POLICY_PRESET", "balanced")
    environment = os.getenv("DEPLOYMENT_ENV", "production")
    
    try:
        preset = PolicyPreset(preset_name)
    except ValueError:
        return False, [f"Unknown preset: {preset_name}"]
    
    manager = PolicyPresetManager(environment)
    return manager.validate_environment_overrides(preset, dict(os.environ))


def _coerce_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be boolean, got {type(value).__name__}")


def _coerce_int(value: Any, *, key: str, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be integer, got {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {parsed}")
    return parsed


def _coerce_float(value: Any, *, key: str, minimum: float = 0.0, maximum: Optional[float] = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be float, got {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {parsed}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{key} must be <= {maximum}, got {parsed}")
    return parsed


def _resolve_rollout_ladder_config_path(path: Optional[str | Path]) -> Path:
    if path is not None:
        return Path(path).resolve()
    return Path(__file__).resolve().parents[1] / "config" / "beta6_rollout_ladder.yaml"


def load_rollout_ladder_policy(path: Optional[str | Path] = None) -> RolloutLadderPolicy:
    """Load strict rollout ladder policy for Phase 6.2 controls."""
    config_path = _resolve_rollout_ladder_config_path(path)
    from ml.beta6.beta6_schema import load_validated_beta6_config

    payload = load_validated_beta6_config(
        config_path,
        expected_filename="beta6_rollout_ladder.yaml",
    )

    ladder = payload.get("ladder")
    if not isinstance(ladder, Mapping):
        raise ValueError("beta6_rollout_ladder.yaml: ladder must be a mapping")
    rung_entries = ladder.get("rungs")
    if not isinstance(rung_entries, list) or not rung_entries:
        raise ValueError("beta6_rollout_ladder.yaml: ladder.rungs must be a non-empty sequence")

    parsed_rungs: list[RolloutLadderRung] = []
    seen_rungs: set[int] = set()
    for idx, raw in enumerate(rung_entries):
        if not isinstance(raw, Mapping):
            raise ValueError(f"beta6_rollout_ladder.yaml: ladder.rungs[{idx}] must be mapping")
        rung_value = _coerce_int(raw.get("rung"), key=f"ladder.rungs[{idx}].rung", minimum=1)
        if rung_value in seen_rungs:
            raise ValueError(f"beta6_rollout_ladder.yaml: duplicate rung={rung_value}")
        seen_rungs.add(rung_value)
        parsed_rungs.append(
            RolloutLadderRung(
                rung=rung_value,
                users=_coerce_int(raw.get("users"), key=f"ladder.rungs[{idx}].users", minimum=1),
                min_days=_coerce_int(raw.get("min_days"), key=f"ladder.rungs[{idx}].min_days", minimum=1),
                requires_phase5_acceptance=_coerce_bool(
                    raw.get("requires_phase5_acceptance", False),
                    key=f"ladder.rungs[{idx}].requires_phase5_acceptance",
                ),
            )
        )

    parsed_rungs.sort(key=lambda entry: entry.rung)
    expected_order = list(range(1, len(parsed_rungs) + 1))
    if [entry.rung for entry in parsed_rungs] != expected_order:
        raise ValueError("beta6_rollout_ladder.yaml: rung numbering must be contiguous starting at 1")

    progression_raw = payload.get("progression_criteria")
    if not isinstance(progression_raw, Mapping):
        raise ValueError("beta6_rollout_ladder.yaml: progression_criteria must be mapping")
    progression = RolloutProgressionCriteria(
        all_mandatory_metric_floors_pass=_coerce_bool(
            progression_raw.get("all_mandatory_metric_floors_pass", True),
            key="progression_criteria.all_mandatory_metric_floors_pass",
        ),
        block_on_open_p0_incidents=_coerce_bool(
            progression_raw.get("block_on_open_p0_incidents", True),
            key="progression_criteria.block_on_open_p0_incidents",
        ),
        min_nightly_pipeline_success_rate=_coerce_float(
            progression_raw.get("min_nightly_pipeline_success_rate", 0.99),
            key="progression_criteria.min_nightly_pipeline_success_rate",
            minimum=0.0,
            maximum=1.0,
        ),
        require_drift_alerts_within_budget=_coerce_bool(
            progression_raw.get("require_drift_alerts_within_budget", True),
            key="progression_criteria.require_drift_alerts_within_budget",
        ),
        require_timeline_hard_gates_all_rooms=_coerce_bool(
            progression_raw.get("require_timeline_hard_gates_all_rooms", True),
            key="progression_criteria.require_timeline_hard_gates_all_rooms",
        ),
    )

    rollback_raw = payload.get("auto_rollback_triggers")
    if not isinstance(rollback_raw, Mapping):
        raise ValueError("beta6_rollout_ladder.yaml: auto_rollback_triggers must be mapping")
    auto_rollback = RolloutAutoRollbackPolicy(
        consecutive_nights=_coerce_int(
            rollback_raw.get("consecutive_nights", 2),
            key="auto_rollback_triggers.consecutive_nights",
            minimum=1,
        ),
        mae_regression_pct_gt=_coerce_float(
            rollback_raw.get("mae_regression_pct_gt", 0.10),
            key="auto_rollback_triggers.mae_regression_pct_gt",
            minimum=0.0,
        ),
        alert_precision_below_threshold=_coerce_bool(
            rollback_raw.get("alert_precision_below_threshold", True),
            key="auto_rollback_triggers.alert_precision_below_threshold",
        ),
        worst_room_f1_below_floor=_coerce_bool(
            rollback_raw.get("worst_room_f1_below_floor", True),
            key="auto_rollback_triggers.worst_room_f1_below_floor",
        ),
        pipeline_success_rate_lt=_coerce_float(
            rollback_raw.get("pipeline_success_rate_lt", 0.97),
            key="auto_rollback_triggers.pipeline_success_rate_lt",
            minimum=0.0,
            maximum=1.0,
        ),
    )

    fallback_raw = payload.get("fallback")
    if not isinstance(fallback_raw, Mapping):
        raise ValueError("beta6_rollout_ladder.yaml: fallback must be mapping")
    baseline_profile = str(fallback_raw.get("baseline_profile", "production")).strip().lower()
    if baseline_profile not in {"pilot", "production"}:
        raise ValueError(
            "beta6_rollout_ladder.yaml: fallback.baseline_profile must be 'pilot' or 'production'"
        )
    fallback = RolloutFallbackPolicy(
        baseline_profile=baseline_profile,
        serving_mode=str(fallback_raw.get("serving_mode", "rule_hmm_baseline")).strip() or "rule_hmm_baseline",
        require_registry_event=_coerce_bool(
            fallback_raw.get("require_registry_event", True),
            key="fallback.require_registry_event",
        ),
        require_ops_alert=_coerce_bool(
            fallback_raw.get("require_ops_alert", True),
            key="fallback.require_ops_alert",
        ),
        recovery_consecutive_pass_nights=_coerce_int(
            fallback_raw.get("recovery_consecutive_pass_nights", 2),
            key="fallback.recovery_consecutive_pass_nights",
            minimum=1,
        ),
    )

    return RolloutLadderPolicy(
        rungs=tuple(parsed_rungs),
        progression=progression,
        auto_rollback=auto_rollback,
        fallback=fallback,
    )
