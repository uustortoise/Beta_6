from pathlib import Path

from ml.beta6.sequence.hmm_decoder import DecoderPolicy, load_runtime_eval_parity_config
from ml.beta6.sequence.transition_builder import load_duration_prior_policy
from ml.beta6.serving.prediction import load_unknown_policy
from ml.beta6.training.active_learning import load_active_learning_policy
from ml.beta6.training.fine_tune_safe_classes import load_safe_finetune_config
from ml.beta6.training.self_supervised_pretrain import load_pretrain_config
from ml.policy_presets import load_rollout_ladder_policy
from ml.yaml_compat import load_yaml_file


BACKEND_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BACKEND_DIR / "config"


def _load_yaml(path: Path) -> dict:
    payload = load_yaml_file(path) or {}
    if not isinstance(payload, dict):
        raise AssertionError(f"expected mapping payload in {path}, got {type(payload).__name__}")
    return payload


def _assert_version_v1(payload: dict, path: Path) -> None:
    assert str(payload.get("version", "")).strip() == "v1", f"missing/invalid version in {path}"


def test_beta6_unknown_policy_yaml_schema():
    path = CONFIG_DIR / "beta6_unknown_policy.yaml"
    payload = _load_yaml(path)
    _assert_version_v1(payload, path)

    section = payload.get("unknown_policy")
    assert isinstance(section, dict), "unknown_policy section must be a mapping"
    for key in (
        "min_confidence",
        "max_entropy",
        "outside_sensed_space_threshold",
        "abstain_label",
        "unknown_label",
        "low_confidence_state",
        "unknown_state",
        "outside_sensed_space_state",
    ):
        assert key in section, f"missing unknown_policy.{key}"

    targets = payload.get("targets")
    assert isinstance(targets, dict), "targets section must be a mapping"
    for key in ("abstain_rate_min", "abstain_rate_max", "unknown_recall_min"):
        assert key in targets, f"missing targets.{key}"

    policy = load_unknown_policy(path)
    assert 0.0 <= policy.min_confidence <= 1.0
    assert policy.max_entropy >= 0.0
    assert 0.0 <= policy.outside_sensed_space_threshold <= 1.0
    assert policy.abstain_label
    assert policy.unknown_label
    assert policy.low_confidence_state
    assert policy.unknown_state
    assert policy.outside_sensed_space_state
    assert 0.0 <= policy.abstain_rate_min <= policy.abstain_rate_max <= 1.0
    assert 0.0 <= policy.unknown_recall_min <= 1.0


def test_beta6_duration_prior_policy_yaml_schema():
    path = CONFIG_DIR / "beta6_duration_prior_policy.yaml"
    payload = _load_yaml(path)
    _assert_version_v1(payload, path)

    duration_priors = payload.get("duration_priors")
    assert isinstance(duration_priors, dict), "duration_priors section must be a mapping"
    assert isinstance(duration_priors.get("default"), dict), "duration_priors.default is required"
    assert isinstance(duration_priors.get("by_label"), dict), "duration_priors.by_label is required"
    assert isinstance(payload.get("transition"), dict), "transition section must be a mapping"

    policy = load_duration_prior_policy(path)
    default = policy.default_prior
    assert 0.0 <= default.min_minutes <= default.target_minutes <= default.max_minutes
    assert default.penalty_weight >= 0.0
    assert policy.transition.step_minutes > 0.0
    assert policy.transition.impossible_transition_penalty > 0.0
    assert policy.transition.switch_penalty >= 0.0
    assert len(policy.priors_by_label) > 0
    for prior in policy.priors_by_label.values():
        assert 0.0 <= prior.min_minutes <= prior.target_minutes <= prior.max_minutes
        assert prior.penalty_weight >= 0.0


def test_beta6_pretrain_yaml_schema():
    path = CONFIG_DIR / "beta6_pretrain.yaml"
    payload = _load_yaml(path)
    _assert_version_v1(payload, path)
    section = payload.get("pretrain")
    assert isinstance(section, dict), "pretrain section must be a mapping"
    for key in ("embedding_dim", "mask_ratio", "epochs", "random_seed", "min_total_rows"):
        assert key in section, f"missing pretrain.{key}"

    cfg = load_pretrain_config(path)
    assert cfg.embedding_dim >= 2
    assert 0.0 <= cfg.mask_ratio <= 0.9
    assert cfg.epochs >= 1
    assert cfg.min_total_rows >= 1


def test_beta6_safe_finetune_yaml_schema():
    path = CONFIG_DIR / "beta6_golden_safe_finetune.yaml"
    payload = _load_yaml(path)
    _assert_version_v1(payload, path)
    section = payload.get("fine_tune")
    assert isinstance(section, dict), "fine_tune section must be a mapping"
    for key in (
        "safe_classes",
        "min_samples_per_class",
        "min_unique_residents",
        "holdout_fraction",
        "random_seed",
        "min_accuracy",
        "max_split_attempts",
    ):
        assert key in section, f"missing fine_tune.{key}"

    cfg = load_safe_finetune_config(path)
    assert len(cfg.safe_classes) > 0
    assert cfg.min_samples_per_class >= 1
    assert cfg.min_unique_residents >= 2
    assert 0.05 <= cfg.holdout_fraction <= 0.5
    assert 0.0 <= cfg.min_accuracy <= 1.0
    assert cfg.max_split_attempts >= 1


def test_beta6_active_learning_yaml_schema():
    path = CONFIG_DIR / "beta6_active_learning_policy.yaml"
    payload = _load_yaml(path)
    _assert_version_v1(payload, path)
    section = payload.get("active_learning")
    assert isinstance(section, dict), "active_learning section must be a mapping"
    for key in (
        "queue_size",
        "uncertainty_fraction",
        "disagreement_fraction",
        "diversity_fraction",
        "uncertainty_percentile",
        "max_share_per_room",
        "max_share_per_class",
        "random_seed",
    ):
        assert key in section, f"missing active_learning.{key}"

    cfg = load_active_learning_policy(path)
    assert cfg.queue_size >= 1
    assert 0.0 <= cfg.uncertainty_fraction <= 1.0
    assert 0.0 <= cfg.disagreement_fraction <= 1.0
    assert 0.0 <= cfg.diversity_fraction <= 1.0
    assert (cfg.uncertainty_fraction + cfg.disagreement_fraction + cfg.diversity_fraction) <= 1.0
    assert 0.0 <= cfg.uncertainty_percentile <= 100.0
    assert 0.1 <= cfg.max_share_per_room <= 1.0
    assert 0.1 <= cfg.max_share_per_class <= 1.0


def test_beta6_runtime_eval_parity_yaml_schema():
    path = CONFIG_DIR / "beta6_runtime_eval_parity.yaml"
    payload = _load_yaml(path)
    _assert_version_v1(payload, path)
    label_map = payload.get("label_map")
    assert isinstance(label_map, dict), "label_map section must be a mapping"
    assert isinstance(payload.get("decoder_policy"), dict), "decoder_policy section must be a mapping"
    required_fields = payload.get("required_parity_fields")
    assert isinstance(required_fields, list), "required_parity_fields must be a list"
    assert set(required_fields) >= {"decoded_label", "source_label", "uncertainty_state"}

    resolved_map, policy = load_runtime_eval_parity_config(path)
    assert isinstance(resolved_map, dict) and resolved_map
    assert "occupied" in resolved_map
    assert "unoccupied" in resolved_map
    assert isinstance(policy, DecoderPolicy)


def test_beta6_policy_defaults_yaml_includes_two_stage_core_training_contract():
    path = CONFIG_DIR / "beta6_policy_defaults.yaml"
    payload = _load_yaml(path)
    _assert_version_v1(payload, path)

    training = payload.get("training")
    assert isinstance(training, dict), "training section must be a mapping"
    two_stage = training.get("two_stage_core")
    assert isinstance(two_stage, dict), "training.two_stage_core must be a mapping"
    for key in (
        "enabled",
        "rooms",
        "gate_mode",
        "stage_a_occupied_threshold",
        "stage_a_target_precision",
        "stage_a_recall_floor",
        "stage_a_threshold_min",
        "stage_a_threshold_max",
        "stage_a_min_predicted_occupied_ratio",
        "stage_a_min_predicted_occupied_abs",
    ):
        assert key in two_stage, f"missing training.two_stage_core.{key}"


def test_beta6_canary_gate_yaml_schema():
    path = CONFIG_DIR / "beta6_canary_gate.yaml"
    payload = _load_yaml(path)
    _assert_version_v1(payload, path)
    canary = payload.get("canary")
    assert isinstance(canary, dict), "canary section must be a mapping"
    for key in (
        "require_real_data_evidence",
        "min_residents_covered",
        "min_resident_days_total",
        "allowed_data_sources",
    ):
        assert key in canary, f"missing canary.{key}"
    assert isinstance(canary["require_real_data_evidence"], bool)
    assert isinstance(canary["allowed_data_sources"], list)
    assert len(canary["allowed_data_sources"]) >= 1


def test_beta6_adapter_policy_yaml_schema():
    path = CONFIG_DIR / "beta6_adapter_policy.yaml"
    payload = _load_yaml(path)
    _assert_version_v1(payload, path)
    adapter = payload.get("adapter")
    assert isinstance(adapter, dict), "adapter section must be a mapping"
    for key in (
        "rank",
        "alpha",
        "l2_reg",
        "random_seed",
        "min_rows",
        "max_versions_per_resident",
        "min_warmup_accuracy",
        "retirement_inactive_days",
        "enable_auto_retire",
    ):
        assert key in adapter, f"missing adapter.{key}"
    assert isinstance(adapter["enable_auto_retire"], bool)


def test_beta6_rollout_ladder_yaml_schema():
    path = CONFIG_DIR / "beta6_rollout_ladder.yaml"
    payload = _load_yaml(path)
    _assert_version_v1(payload, path)

    ladder = payload.get("ladder")
    assert isinstance(ladder, dict), "ladder section must be a mapping"
    assert isinstance(ladder.get("rungs"), list), "ladder.rungs must be a list"
    assert len(ladder["rungs"]) >= 1
    for idx, rung in enumerate(ladder["rungs"]):
        assert isinstance(rung, dict), f"ladder.rungs[{idx}] must be a mapping"
        for key in ("rung", "users", "min_days", "requires_phase5_acceptance"):
            assert key in rung, f"missing ladder.rungs[{idx}].{key}"
        assert isinstance(rung["requires_phase5_acceptance"], bool)

    for section in ("progression_criteria", "auto_rollback_triggers", "fallback"):
        assert isinstance(payload.get(section), dict), f"{section} section must be a mapping"

    policy = load_rollout_ladder_policy(path)
    assert len(policy.rungs) >= 1
    assert policy.progression.min_nightly_pipeline_success_rate >= 0.0
    assert policy.auto_rollback.consecutive_nights >= 1
    assert policy.fallback.baseline_profile in {"pilot", "production"}


def test_beta6_policy_defaults_yaml_supports_bedroom_entrance_fix_controls():
    path = CONFIG_DIR / "beta6_policy_defaults.yaml"
    payload = _load_yaml(path)
    _assert_version_v1(payload, path)

    unoccupied = payload.get("unoccupied_downsample")
    assert isinstance(unoccupied, dict), "unoccupied_downsample section must be a mapping"
    assert isinstance(
        unoccupied.get("max_post_downsample_prior_drift_by_room"),
        dict,
    ), "unoccupied_downsample.max_post_downsample_prior_drift_by_room must be a mapping"
    assert isinstance(
        unoccupied.get("prior_drift_guard_rooms"),
        list,
    ), "unoccupied_downsample.prior_drift_guard_rooms must be a sequence"

    minority = payload.get("minority_sampling")
    assert isinstance(minority, dict), "minority_sampling section must be a mapping"
    assert isinstance(
        minority.get("max_post_sampling_prior_drift_by_room"),
        dict,
    ), "minority_sampling.max_post_sampling_prior_drift_by_room must be a mapping"
    assert isinstance(
        minority.get("prior_drift_guard_rooms"),
        list,
    ), "minority_sampling.prior_drift_guard_rooms must be a sequence"

    training = payload.get("training")
    assert isinstance(training, dict), "training section must be a mapping"
    assert isinstance(training.get("factorized_primary_rooms"), list)
    assert isinstance(training.get("transition_focus_room_labels"), dict)
    assert isinstance(training.get("transition_focus_radius_steps_by_room"), dict)
    assert isinstance(training.get("transition_focus_max_multiplier_by_room"), dict)
    assert isinstance(training.get("transition_focus_max_post_sampling_prior_drift_by_room"), dict)
    assert isinstance(training.get("transition_focus_prior_drift_guard_rooms"), list)

    reproducibility = payload.get("reproducibility")
    assert isinstance(reproducibility, dict), "reproducibility section must be a mapping"
    assert isinstance(reproducibility.get("multi_seed_rooms"), list)
    assert isinstance(reproducibility.get("multi_seed_candidate_seeds"), list)
