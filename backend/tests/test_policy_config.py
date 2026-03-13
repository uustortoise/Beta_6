from ml.policy_config import load_policy_from_env


def test_load_policy_defaults_match_legacy_knobs():
    policy = load_policy_from_env({})
    assert policy.unoccupied_downsample.min_share == 0.45
    assert policy.unoccupied_downsample.stride == 10
    livingroom_downsample = policy.unoccupied_downsample.resolve("LivingRoom")
    assert livingroom_downsample["min_share"] == 0.30
    assert livingroom_downsample["stride"] == 4
    assert policy.minority_sampling.enabled is True
    assert policy.minority_sampling.target_share == 0.14
    assert policy.calibration.threshold_floor == 0.35
    assert policy.calibration.threshold_cap == 0.80
    assert policy.clinical_priority.multipliers["sleep"] == 1.6
    assert policy.resampling.max_ffill_gap_seconds == 60.0
    assert policy.release_gate.min_training_days == 2.0
    assert policy.release_gate.min_observed_days == 2
    assert policy.release_gate.min_retained_sample_ratio == 0.10
    assert policy.release_gate.max_dropped_ratio == 0.90
    assert policy.release_gate.min_validation_class_support == 20
    bathroom_viability = policy.data_viability.resolve("bathroom")
    assert bathroom_viability["min_observed_days"] == 7
    assert bathroom_viability["min_post_gap_rows"] == 8000
    assert bathroom_viability["max_unresolved_drop_ratio"] == 0.90
    assert bathroom_viability["min_training_windows"] == 2000
    assert policy.reproducibility.random_seed == 42
    assert policy.reproducibility.skip_if_same_data_and_policy is True
    assert policy.reproducibility.multi_seed_rooms == ["entrance", "livingroom"]
    assert policy.reproducibility.multi_seed_candidate_seeds == [40, 41, 42, 43, 44, 45]
    assert policy.promotion_eligibility.min_training_days_with_champion == 7.0
    assert policy.training_profile.post_split_shuffle_rooms == ["entrance", "bedroom"]
    assert policy.training_profile.transition_focus_prior_drift_guard_rooms == ["bedroom"]
    assert policy.training_profile.transition_focus_max_post_sampling_prior_drift_by_room["bedroom"] == 0.10
    assert policy.two_stage_core.enabled is True
    assert policy.two_stage_core.rooms == ["bedroom", "livingroom", "kitchen", "bathroom"]
    assert policy.two_stage_core.gate_mode == "primary"
    assert policy.two_stage_core.stage_a_occupied_threshold == 0.50
    assert policy.two_stage_core.stage_a_target_precision == 0.70
    assert policy.two_stage_core.stage_a_recall_floor == 0.20
    assert policy.two_stage_core.stage_a_threshold_min == 0.00
    assert policy.two_stage_core.stage_a_threshold_max == 0.95
    assert policy.two_stage_core.stage_a_min_predicted_occupied_ratio == 0.50
    assert policy.two_stage_core.stage_a_min_predicted_occupied_abs == 0.05


def test_load_policy_env_overrides_unoccupied_and_minority():
    env = {
        "UNOCCUPIED_DOWNSAMPLE_MIN_SHARE": "0.2",
        "UNOCCUPIED_DOWNSAMPLE_STRIDE": "7",
        "UNOCCUPIED_DOWNSAMPLE_MIN_SHARE_BY_ROOM": "bathroom:0.3",
        "UNOCCUPIED_DOWNSAMPLE_STRIDE_BY_ROOM": "bathroom:2",
        "ENABLE_MINORITY_CLASS_SAMPLING": "false",
        "MINORITY_TARGET_SHARE": "0.22",
        "MINORITY_MAX_MULTIPLIER": "9",
        "MINORITY_TARGET_SHARE_BY_ROOM": "bathroom:0.31",
        "MINORITY_MAX_MULTIPLIER_BY_ROOM": "bathroom:4",
    }
    policy = load_policy_from_env(env)

    unocc_room = policy.unoccupied_downsample.resolve("Bathroom")
    assert unocc_room["min_share"] == 0.3
    assert unocc_room["stride"] == 2

    minority_room = policy.minority_sampling.resolve("Bathroom")
    assert minority_room["enabled"] is False
    assert minority_room["target_share"] == 0.31
    assert minority_room["max_multiplier"] == 4


def test_load_policy_defaults_include_livingroom_downsample_override():
    policy = load_policy_from_env({})
    livingroom_cfg = policy.unoccupied_downsample.resolve("LivingRoom")
    assert livingroom_cfg["min_share"] == 0.30
    assert livingroom_cfg["stride"] == 4


def test_empty_room_override_env_disables_default_room_map():
    env = {
        "MINORITY_TARGET_SHARE_BY_ROOM": "",
        "MINORITY_MAX_MULTIPLIER_BY_ROOM": "",
    }
    policy = load_policy_from_env(env)
    # Bathroom room-specific default is 0.18, but explicit empty env should disable the map.
    bathroom_cfg = policy.minority_sampling.resolve("Bathroom")
    assert bathroom_cfg["target_share"] == policy.minority_sampling.target_share
    assert bathroom_cfg["max_multiplier"] == policy.minority_sampling.max_multiplier


def test_load_policy_supports_room_fix_controls():
    env = {
        "UNOCCUPIED_MAX_POST_DOWNSAMPLE_PRIOR_DRIFT_BY_ROOM": "entrance:0.03",
        "UNOCCUPIED_PRIOR_DRIFT_GUARD_ROOMS": "entrance",
        "MINORITY_MAX_POST_SAMPLING_PRIOR_DRIFT_BY_ROOM": "entrance:0.03",
        "MINORITY_PRIOR_DRIFT_GUARD_ROOMS": "entrance",
        "MULTI_SEED_ROOMS": "entrance",
        "MULTI_SEED_CANDIDATE_SEEDS": "40,41,42,43,44,45",
        "FACTORIZED_PRIMARY_ROOMS": "bedroom",
        "POST_SPLIT_SHUFFLE_ROOMS": "entrance,kitchen",
        "TRANSITION_FOCUS_ROOM_LABELS": "bedroom:bedroom_normal_use",
        "TRANSITION_FOCUS_RADIUS_STEPS_BY_ROOM": "bedroom:12",
        "TRANSITION_FOCUS_MAX_MULTIPLIER_BY_ROOM": "bedroom:3",
        "TRANSITION_FOCUS_MAX_POST_SAMPLING_PRIOR_DRIFT_BY_ROOM": "bedroom:0.08",
        "TRANSITION_FOCUS_PRIOR_DRIFT_GUARD_ROOMS": "bedroom",
        "ENABLE_TWO_STAGE_CORE_MODELING": "false",
        "TWO_STAGE_CORE_ROOMS": "bathroom",
        "TWO_STAGE_CORE_GATE_MODE": "shadow",
        "TWO_STAGE_CORE_STAGE_A_OCCUPIED_THRESHOLD": "0.42",
        "TWO_STAGE_CORE_STAGE_A_TARGET_PRECISION": "0.76",
        "TWO_STAGE_CORE_STAGE_A_RECALL_FLOOR": "0.33",
        "TWO_STAGE_CORE_STAGE_A_THRESHOLD_MIN": "0.10",
        "TWO_STAGE_CORE_STAGE_A_THRESHOLD_MAX": "0.88",
        "TWO_STAGE_CORE_STAGE_A_MIN_PRED_OCCUPIED_RATIO": "0.61",
        "TWO_STAGE_CORE_STAGE_A_MIN_PRED_OCCUPIED_ABS": "0.14",
    }
    policy = load_policy_from_env(env)

    entrance_downsample_cfg = policy.unoccupied_downsample.resolve("Entrance")
    assert entrance_downsample_cfg["prior_drift_guard_enabled"] is True
    assert entrance_downsample_cfg["max_post_downsample_prior_drift"] == 0.03

    entrance_cfg = policy.minority_sampling.resolve("Entrance")
    assert entrance_cfg["prior_drift_guard_enabled"] is True
    assert entrance_cfg["max_post_sampling_prior_drift"] == 0.03

    assert policy.reproducibility.multi_seed_rooms == ["entrance"]
    assert policy.reproducibility.multi_seed_candidate_seeds == [40, 41, 42, 43, 44, 45]

    assert policy.training_profile.factorized_primary_rooms == ["bedroom"]
    assert policy.training_profile.post_split_shuffle_rooms == ["entrance", "kitchen"]
    assert policy.training_profile.transition_focus_room_labels["bedroom"] == "bedroom_normal_use"
    assert policy.training_profile.transition_focus_radius_steps_by_room["bedroom"] == 12
    assert policy.training_profile.transition_focus_max_multiplier_by_room["bedroom"] == 3
    assert policy.training_profile.transition_focus_max_post_sampling_prior_drift_by_room["bedroom"] == 0.08
    assert policy.training_profile.transition_focus_prior_drift_guard_rooms == ["bedroom"]
    assert policy.two_stage_core.enabled is False
    assert policy.two_stage_core.rooms == ["bathroom"]
    assert policy.two_stage_core.gate_mode == "shadow"
    assert policy.two_stage_core.stage_a_occupied_threshold == 0.42
    assert policy.two_stage_core.stage_a_target_precision == 0.76
    assert policy.two_stage_core.stage_a_recall_floor == 0.33
    assert policy.two_stage_core.stage_a_threshold_min == 0.10
    assert policy.two_stage_core.stage_a_threshold_max == 0.88
    assert policy.two_stage_core.stage_a_min_predicted_occupied_ratio == 0.61
    assert policy.two_stage_core.stage_a_min_predicted_occupied_abs == 0.14


def test_label_map_env_parsing_for_calibration_and_clinical_priority():
    env = {
        "CLINICAL_PRIORITY_MULTIPLIERS": '{"sleep":2.2,"inactive":0.5}',
        "PRECISION_TARGETS_BY_LABEL": "sleep:0.81,toilet:0.73",
        "RECALL_FLOOR_BY_LABEL": '{"sleep":0.25}',
        "THRESHOLD_FLOOR": "0.2",
        "THRESHOLD_CAP": "0.9",
    }
    policy = load_policy_from_env(env)

    assert policy.clinical_priority.get_multiplier("sleep") == 2.2
    assert policy.clinical_priority.get_multiplier("inactive") == 0.5
    assert policy.calibration.get_precision_target("sleep") == 0.81
    assert policy.calibration.get_precision_target("toilet") == 0.73
    assert policy.calibration.get_recall_floor("sleep") == 0.25
    assert policy.calibration.threshold_floor == 0.2
    assert policy.calibration.threshold_cap == 0.9


def test_room_label_clinical_priority_overrides_global_multiplier():
    env = {
        "CLINICAL_PRIORITY_MULTIPLIERS": "sleep:1.2,unoccupied:0.9",
        "CLINICAL_PRIORITY_MULTIPLIERS_BY_ROOM_LABEL": "bedroom.unoccupied:2.4,bedroom.sleep:0.7,bathroom.bathroom_normal_use:0.8",
    }
    policy = load_policy_from_env(env)

    # Room-specific override should take precedence.
    assert policy.clinical_priority.get_room_label_multiplier("Bedroom", "unoccupied") == 2.4
    assert policy.clinical_priority.get_room_label_multiplier("Bedroom", "sleep") == 0.7
    assert policy.clinical_priority.get_room_label_multiplier("Bathroom", "bathroom_normal_use") == 0.8
    # Fall back to global label multiplier when room-specific entry does not exist.
    assert policy.clinical_priority.get_room_label_multiplier("Kitchen", "sleep") == 1.2
    assert policy.clinical_priority.get_room_label_multiplier("Kitchen", "unoccupied") == 0.9


def test_production_defaults_neutralize_livingroom_room_label_clinical_priority():
    policy = load_policy_from_env({})

    assert policy.release_gate.evidence_profile == "production"
    assert (
        policy.clinical_priority.get_room_label_multiplier("LivingRoom", "livingroom_normal_use") == 1.0
    )
    assert policy.clinical_priority.get_room_label_multiplier("LivingRoom", "unoccupied") == 1.0
    assert policy.clinical_priority.get_room_label_multiplier("Kitchen", "kitchen_normal_use") == 1.2
    assert policy.clinical_priority.get_room_label_multiplier("Kitchen", "unoccupied") == 0.75


def test_resampling_policy_parses_env_tokens():
    policy = load_policy_from_env({"MAX_RESAMPLE_FFILL_GAP_SECONDS": "unbounded"})
    assert policy.resampling.max_ffill_gap_seconds is None

    policy = load_policy_from_env({"MAX_RESAMPLE_FFILL_GAP_SECONDS": "45"})
    assert policy.resampling.max_ffill_gap_seconds == 45.0


def test_release_gate_and_reproducibility_env_overrides():
    policy = load_policy_from_env(
        {
            "RELEASE_GATE_MIN_TRAINING_DAYS": "3.5",
            "RELEASE_GATE_MIN_SAMPLES": "200",
            "RELEASE_GATE_MIN_CALIBRATION_SUPPORT": "50",
            "RELEASE_GATE_MIN_VALIDATION_CLASS_SUPPORT": "40",
            "RELEASE_GATE_MIN_OBSERVED_DAYS": "4",
            "RELEASE_GATE_MIN_RETAINED_SAMPLE_RATIO": "0.25",
            "RELEASE_GATE_MAX_DROPPED_RATIO": "0.75",
            "RELEASE_GATE_BLOCK_ON_LOW_SUPPORT_FALLBACK": "true",
            "RELEASE_GATE_BLOCK_ON_TRAIN_FALLBACK_METRICS": "false",
            "RELEASE_GATE_MIN_RECALL_SUPPORT": "35",
            "RELEASE_GATE_MIN_RECALL_BY_ROOM_LABEL": "bedroom.unoccupied:0.60,livingroom.unoccupied:0.55",
            "TRAINING_RANDOM_SEED": "123",
            "SKIP_RETRAIN_IF_SAME_DATA_AND_POLICY": "false",
            "ALLOW_GATE_CONFIG_FALLBACK_PASS": "false",
            "PROMOTION_MIN_TRAINING_DAYS_WITH_CHAMPION": "9",
        }
    )
    assert policy.release_gate.min_training_days == 3.5
    assert policy.release_gate.min_samples == 200
    assert policy.release_gate.min_calibration_support == 50
    assert policy.release_gate.min_validation_class_support == 40
    assert policy.release_gate.min_observed_days == 4
    assert policy.release_gate.min_retained_sample_ratio == 0.25
    assert policy.release_gate.max_dropped_ratio == 0.75
    assert policy.release_gate.block_on_low_support_fallback is True
    assert policy.release_gate.block_on_train_fallback_metrics is False
    assert policy.release_gate.min_recall_support == 35
    assert policy.release_gate.min_recall_by_room_label["bedroom.unoccupied"] == 0.60
    assert policy.release_gate.min_recall_by_room_label["livingroom.unoccupied"] == 0.55
    assert policy.reproducibility.random_seed == 123
    assert policy.reproducibility.skip_if_same_data_and_policy is False
    assert policy.release_gate.allow_gate_config_fallback_pass is False
    assert policy.promotion_eligibility.min_training_days_with_champion == 9.0


def test_release_gate_evidence_profile_pilot_stage_a_defaults():
    policy = load_policy_from_env({"RELEASE_GATE_EVIDENCE_PROFILE": "pilot_stage_a"})
    assert policy.release_gate.evidence_profile == "pilot_stage_a"
    assert policy.release_gate.min_validation_class_support == 10
    assert policy.release_gate.min_recall_support == 20


def test_release_gate_evidence_profile_respects_explicit_floor_overrides():
    policy = load_policy_from_env(
        {
            "RELEASE_GATE_EVIDENCE_PROFILE": "pilot_stage_b",
            "RELEASE_GATE_MIN_VALIDATION_CLASS_SUPPORT": "12",
            "RELEASE_GATE_MIN_RECALL_SUPPORT": "17",
        }
    )
    assert policy.release_gate.evidence_profile == "pilot_stage_b"
    assert policy.release_gate.min_validation_class_support == 12
    assert policy.release_gate.min_recall_support == 17


def test_pilot_profile_forces_neutral_clinical_priority_multipliers():
    policy = load_policy_from_env(
        {
            "RELEASE_GATE_EVIDENCE_PROFILE": "pilot_stage_a",
            "CLINICAL_PRIORITY_MULTIPLIERS": "sleep:2.2,inactive:0.5",
            "CLINICAL_PRIORITY_MULTIPLIERS_BY_ROOM_LABEL": "bedroom.sleep:0.7,bathroom.shower:2.5",
        }
    )

    assert policy.release_gate.evidence_profile == "pilot_stage_a"
    assert policy.clinical_priority.get_multiplier("sleep") == 1.0
    assert policy.clinical_priority.get_multiplier("inactive") == 1.0
    assert policy.clinical_priority.get_multiplier("shower") == 1.0
    assert all(value == 1.0 for value in policy.clinical_priority.multipliers.values())
    assert all(value == 1.0 for value in policy.clinical_priority.multipliers_by_room_label.values())


def test_data_viability_env_overrides():
    policy = load_policy_from_env(
        {
            "DATA_VIABILITY_MIN_OBSERVED_DAYS": "9",
            "DATA_VIABILITY_MIN_POST_GAP_ROWS": "14000",
            "DATA_VIABILITY_MAX_UNRESOLVED_DROP_RATIO": "0.7",
            "DATA_VIABILITY_MIN_TRAINING_WINDOWS": "3200",
            "DATA_VIABILITY_MIN_OBSERVED_DAYS_BY_ROOM": "bathroom:5",
            "DATA_VIABILITY_MIN_POST_GAP_ROWS_BY_ROOM": "bathroom:6000",
            "DATA_VIABILITY_MAX_UNRESOLVED_DROP_RATIO_BY_ROOM": "bathroom:0.95",
            "DATA_VIABILITY_MIN_TRAINING_WINDOWS_BY_ROOM": "bathroom:1500",
        }
    )
    bathroom = policy.data_viability.resolve("bathroom")
    bedroom = policy.data_viability.resolve("bedroom")
    assert bathroom["min_observed_days"] == 5
    assert bathroom["min_post_gap_rows"] == 6000
    assert bathroom["max_unresolved_drop_ratio"] == 0.95
    assert bathroom["min_training_windows"] == 1500
    assert bedroom["min_observed_days"] == 9
    assert bedroom["min_post_gap_rows"] == 14000
    assert bedroom["max_unresolved_drop_ratio"] == 0.7
    assert bedroom["min_training_windows"] == 3200


def test_event_first_defaults():
    policy = load_policy_from_env({})
    assert policy.event_first.shadow is False
    assert policy.event_first.enabled is False
    assert policy.event_first.decoder_on_threshold == 0.60
    assert policy.event_first.decoder_off_threshold == 0.40
    assert policy.event_first.decoder_min_on_steps == 3
    assert policy.event_first.unknown_rate_global_cap == 0.15
    assert policy.event_first.unknown_rate_room_cap == 0.20


def test_event_first_env_overrides():
    policy = load_policy_from_env(
        {
            "EVENT_FIRST_SHADOW": "true",
            "EVENT_FIRST_ENABLED": "true",
            "EVENT_DECODER_ON_THRESHOLD": "0.72",
            "EVENT_DECODER_OFF_THRESHOLD": "0.55",
            "EVENT_DECODER_MIN_ON_STEPS": "5",
            "EVENT_PROBABILITY_CALIBRATION": "platt",
            "EVENT_CALIBRATION_MIN_SAMPLES": "800",
            "UNKNOWN_RATE_GLOBAL_CAP": "0.11",
            "UNKNOWN_RATE_ROOM_CAP": "0.16",
        }
    )
    assert policy.event_first.shadow is True
    assert policy.event_first.enabled is True
    assert policy.event_first.decoder_on_threshold == 0.72
    assert policy.event_first.decoder_off_threshold == 0.55
    assert policy.event_first.decoder_min_on_steps == 5
    assert policy.event_first.probability_calibration == "platt"
    assert policy.event_first.calibration_min_samples == 800
    assert policy.event_first.unknown_rate_global_cap == 0.11
    assert policy.event_first.unknown_rate_room_cap == 0.16


def test_policy_defaults_can_emit_named_room_diagnostic_profiles():
    policy = load_policy_from_env({})
    profiles = policy.training_profile.room_diagnostic_profiles

    assert isinstance(profiles, dict)
    assert "livingroom_policy_sensitivity" in profiles
    assert "bedroom_grouped_fragility" in profiles

    livingroom_profile = profiles["livingroom_policy_sensitivity"]
    assert livingroom_profile["room"] == "livingroom"
    assert livingroom_profile["training_gate"] is False
    assert "typed_policy_fields" in livingroom_profile
    assert "two_stage_core.gate_mode" in livingroom_profile["typed_policy_fields"]

    bedroom_profile = profiles["bedroom_grouped_fragility"]
    assert bedroom_profile["room"] == "bedroom"
    assert bedroom_profile["grouped_regime"] == "grouped_by_date"
    assert bedroom_profile["training_gate"] is True
