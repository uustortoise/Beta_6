import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.run_event_first_backtest import (
    _add_cross_room_context_features_for_day,
    _add_room_temporal_occupancy_features,
    _adaptive_room_occupancy_threshold,
    _apply_activity_label_corrections,
    _classification_metrics,
    _care_fragmentation_score,
    _apply_room_prediction_occupancy_smoothing,
    _apply_room_passive_occupancy_hysteresis,
    _calibrate_duration_from_train,
    _compute_binary_episode_metrics,
    _compute_home_empty_metrics,
    _daily_target_series,
    _duration_mae_from_labels,
    _extract_error_episodes,
    _extract_day,
    _compute_day_continuity,
    _build_data_continuity_audit,
    _build_stage_a_group_ids_from_timestamps,
    _apply_single_resident_arbitration,
    _apply_livingroom_cross_room_presence_decoder,
    _apply_room_timeline_decoder_v2,
    _mean_abs_error,
    _tune_kitchen_occupancy_threshold_by_mae,
    _build_kitchen_stage_a_sample_weights,
    _build_livingroom_passive_alignment_sample_weights,
    _build_room_boundary_sample_weights,
    _apply_kitchen_temporal_decoder,
    _apply_duration_calibration,
    _apply_bathroom_shower_fallback,
    _apply_critical_label_rescue,
    _apply_room_occupancy_temporal_decoder,
    _fit_affine_duration_calibrator,
    _load_activity_label_corrections,
    _parse_hour_range,
    _parse_hour_ranges,
    _parse_room_label_thresholds,
    _parse_room_thresholds,
    _parse_tune_rooms,
    _rolling_linear_slope,
    _build_room_data_diagnostics,
    _build_home_empty_status,
    _label_correction_reference_for_room_day,
    _build_room_model_config,
    _room_mae_tuning_flag_enabled,
    _split_hard_gate,
    _tune_room_occupancy_threshold_for_hard_gate,
)


def test_parse_room_thresholds_parses_and_clamps():
    out = _parse_room_thresholds("entrance=0.55, bedroom=1.5, kitchen=-0.2, bad")
    assert out["entrance"] == 0.55
    assert out["bedroom"] == 1.0
    assert out["kitchen"] == 0.0
    assert "bad" not in out


def test_extract_day_accepts_non_december_train_filename():
    assert _extract_day(Path("HK0011_jessica_train_12jan2026.xlsx")) == 12


def test_extract_day_rejects_derived_train_variants():
    assert _extract_day(Path("HK0011_jessica_train_8dec2025_occupied_only.xlsx")) is None


def test_compute_day_continuity_reports_missing_internal_days():
    out = _compute_day_continuity([4, 5, 7, 8, 8])
    assert out["days"] == [4, 5, 7, 8]
    assert out["count"] == 4
    assert out["min_day"] == 4
    assert out["max_day"] == 8
    assert out["missing_days_between_min_max"] == [6]
    assert out["is_contiguous"] is False


def test_build_data_continuity_audit_includes_gap_and_noncanonical_exclusions():
    out = _build_data_continuity_audit(
        elder_id="HK0011_jessica",
        min_day=4,
        max_day=8,
        candidate_files=[
            Path("HK0011_jessica_train_4dec2025.xlsx"),
            Path("HK0011_jessica_train_6dec2025.xlsx"),
            Path("HK0011_jessica_train_8dec2025.xlsx"),
            Path("HK0011_jessica_train_8dec2025_occupied_only.xlsx"),
        ],
        canonical_files=[
            Path("HK0011_jessica_train_4dec2025.xlsx"),
            Path("HK0011_jessica_train_6dec2025.xlsx"),
            Path("HK0011_jessica_train_8dec2025.xlsx"),
        ],
        files_by_day={
            4: Path("HK0011_jessica_train_4dec2025.xlsx"),
            6: Path("HK0011_jessica_train_6dec2025.xlsx"),
            8: Path("HK0011_jessica_train_8dec2025.xlsx"),
        },
        excluded_non_canonical=[
            {"file": "HK0011_jessica_train_8dec2025_occupied_only.xlsx", "reason": "derived_or_noncanonical_suffix"}
        ],
        invalid_day_token_files=[],
    )
    assert out["candidate_file_count"] == 4
    assert out["canonical_file_count"] == 3
    assert out["selected_file_count"] == 3
    assert out["selected_days"] == [4, 6, 8]
    assert out["missing_days_in_requested_window"] == [5, 7]
    assert out["missing_days_between_selected_min_max"] == [5, 7]
    assert out["selected_day_continuity"]["is_contiguous"] is False
    assert len(out["excluded_non_canonical_files"]) == 1
    assert out["excluded_non_canonical_files"][0]["reason"] == "derived_or_noncanonical_suffix"


def test_load_activity_label_corrections_defaults_disabled_when_path_missing():
    corrections, summary = _load_activity_label_corrections(None)
    assert corrections == []
    assert summary["enabled"] is False
    assert summary["rows_loaded"] == 0


def test_load_activity_label_corrections_parses_valid_rows_and_fallback_day(tmp_path: Path):
    csv_path = tmp_path / "corrections.csv"
    pd.DataFrame(
        [
            {
                "room": "LivingRoom",
                "label": "livingroom_normal_use",
                "start_time": "2025-12-06 09:00:00",
                "end_time": "2025-12-06 09:00:20",
            },
            {
                "room": "LivingRoom",
                "label": "",
                "start_time": "2025-12-06 09:01:00",
                "end_time": "2025-12-06 09:01:20",
            },
            {
                "room": "Bedroom",
                "label": "sleep",
                "start_time": "2025-12-06 22:10:00",
                "end_time": "2025-12-06 22:09:50",
            },
        ]
    ).to_csv(csv_path, index=False)

    corrections, summary = _load_activity_label_corrections(csv_path)
    assert summary["enabled"] is True
    assert summary["rows_loaded"] == 3
    assert summary["rows_valid"] == 1
    assert summary["rows_invalid"] == 2
    assert corrections[0]["room"] == "livingroom"
    assert corrections[0]["label"] == "livingroom_normal_use"
    assert corrections[0]["day"] == 6


def test_load_activity_label_corrections_rejects_missing_required_columns(tmp_path: Path):
    csv_path = tmp_path / "bad_corrections.csv"
    pd.DataFrame([{"room": "Bedroom", "label": "sleep", "start_time": "2025-12-06 22:00:00"}]).to_csv(
        csv_path, index=False
    )
    with pytest.raises(ValueError, match="requires columns"):
        _load_activity_label_corrections(csv_path)


def test_build_stage_a_group_ids_from_timestamps_groups_within_minute():
    ts = pd.to_datetime(
        [
            "2025-12-07 10:00:01",
            "2025-12-07 10:00:39",
            "2025-12-07 10:01:05",
            "2025-12-07 10:01:59",
            "2025-12-07 10:02:00",
        ]
    )
    gids = _build_stage_a_group_ids_from_timestamps(timestamps=ts, resolution_seconds=60)
    assert len(gids) == len(ts)
    assert int(gids[0]) == int(gids[1])
    assert int(gids[2]) == int(gids[3])
    assert int(gids[1]) != int(gids[2])
    assert int(gids[3]) != int(gids[4])


def test_apply_activity_label_corrections_updates_windows_and_reports_skips():
    ts = pd.date_range("2025-12-07 10:00:00", periods=6, freq="10s")
    room_day_data = {
        "LivingRoom": {
            7: pd.DataFrame(
                {
                    "timestamp": ts,
                    "activity": ["unoccupied"] * len(ts),
                }
            )
        }
    }
    corrections = [
        {
            "room": "livingroom",
            "label": "livingroom_normal_use",
            "start_time": "2025-12-07 10:00:10",
            "end_time": "2025-12-07 10:00:30",
            "day": 7,
        },
        {
            "room": "kitchen",
            "label": "kitchen_normal_use",
            "start_time": "2025-12-07 10:00:10",
            "end_time": "2025-12-07 10:00:30",
            "day": 7,
        },
        {
            "room": "livingroom",
            "label": "livingroom_normal_use",
            "start_time": "2025-12-07 12:00:10",
            "end_time": "2025-12-07 12:00:30",
            "day": 7,
        },
    ]
    summary = _apply_activity_label_corrections(room_day_data=room_day_data, corrections=corrections)
    out = room_day_data["LivingRoom"][7]["activity"].tolist()
    assert out == [
        "unoccupied",
        "livingroom_normal_use",
        "livingroom_normal_use",
        "livingroom_normal_use",
        "unoccupied",
        "unoccupied",
    ]
    assert summary["enabled"] is True
    assert summary["requested_windows"] == 3
    assert summary["applied_windows"] == 1
    assert summary["applied_rows"] == 3
    assert summary["skipped_by_reason"]["room_not_found"] == 1
    assert summary["skipped_by_reason"]["no_rows_in_window"] == 1
    assert summary["by_room_day"]["LivingRoom:day7"]["windows_applied"] == 1
    assert summary["by_room_day"]["LivingRoom:day7"]["rows_updated"] == 3


def test_build_room_data_diagnostics_produces_snapshot_payloads():
    out = _build_room_data_diagnostics(
        y_fit=np.asarray(["unoccupied", "sleep", "sleep"], dtype=object),
        y_calib=np.asarray(["unoccupied", "livingroom_normal_use"], dtype=object),
        y_test=np.asarray(["unoccupied", "sleep", "unknown", "unoccupied"], dtype=object),
    )
    assert out["window_seconds"] == 10
    assert out["occupied_rate_snapshot"]["fit"]["occupied_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert out["occupied_rate_snapshot"]["calib"]["occupied_rate"] == pytest.approx(0.5, abs=1e-6)
    assert out["label_minutes_snapshot"]["test"]["unoccupied"] == pytest.approx(2 / 6, abs=1e-6)
    assert out["label_minutes_snapshot"]["test"]["occupied_total"] == pytest.approx(2 / 6, abs=1e-6)
    assert "sleep" in out["key_labels"]


def test_label_correction_reference_for_room_day_extracts_room_day_details():
    summary = {
        "load": {"enabled": True, "rows_loaded": 6, "rows_valid": 4},
        "apply": {
            "requested_windows": 3,
            "applied_windows": 2,
            "applied_rows": 18,
            "by_room_day": {"LivingRoom:day7": {"windows_applied": 2, "rows_updated": 18}},
        },
    }
    out = _label_correction_reference_for_room_day(
        label_corrections_summary=summary,
        room="LivingRoom",
        day=7,
    )
    assert out["enabled"] is True
    assert out["requested_windows"] == 3
    assert out["applied_windows"] == 2
    assert out["applied_rows"] == 18
    assert out["room_day"]["room_day_key"] == "LivingRoom:day7"
    assert out["room_day"]["windows_applied"] == 2
    assert out["room_day"]["rows_updated"] == 18


def test_room_mae_tuning_flag_enabled_decouples_kitchen_from_global_flag():
    enabled, reason = _room_mae_tuning_flag_enabled(
        room_key="kitchen",
        enable_room_mae_threshold_tuning=False,
        enable_kitchen_mae_threshold_tuning=True,
    )
    assert enabled is True
    assert reason == "kitchen_mae_tuning_disabled"


def test_room_mae_tuning_flag_enabled_uses_global_for_non_kitchen():
    enabled, reason = _room_mae_tuning_flag_enabled(
        room_key="bedroom",
        enable_room_mae_threshold_tuning=False,
        enable_kitchen_mae_threshold_tuning=True,
    )
    assert enabled is False
    assert reason == "room_mae_tuning_disabled"


def test_cross_room_context_features_aggregate_other_rooms_same_timestamp():
    ts = pd.date_range("2025-12-06 09:00:00", periods=3, freq="10s")
    day_frames = {
        "Bedroom": pd.DataFrame(
            {
                "timestamp": ts,
                "motion": [0.0, 0.2, 0.0],
                "co2": [500.0, 510.0, 520.0],
            }
        ),
        "LivingRoom": pd.DataFrame(
            {
                "timestamp": ts,
                "motion": [1.0, 0.0, 0.8],
                "co2": [600.0, 620.0, 640.0],
            }
        ),
        "Kitchen": pd.DataFrame(
            {
                "timestamp": ts,
                "motion": [0.0, 1.0, 0.9],
                "co2": [550.0, 560.0, 570.0],
            }
        ),
    }

    _add_cross_room_context_features_for_day(
        day_room_frames=day_frames,
        rooms=["Bedroom", "LivingRoom", "Kitchen"],
    )

    bedroom = day_frames["Bedroom"]
    np.testing.assert_allclose(
        bedroom["ctx_other_co2_mean"].to_numpy(dtype=float),
        np.asarray([575.0, 590.0, 605.0], dtype=float),
    )
    np.testing.assert_allclose(
        bedroom["ctx_other_co2_max"].to_numpy(dtype=float),
        np.asarray([600.0, 620.0, 640.0], dtype=float),
    )
    np.testing.assert_allclose(
        bedroom["ctx_other_motion_active_count"].to_numpy(dtype=float),
        np.asarray([1.0, 1.0, 2.0], dtype=float),
    )
    np.testing.assert_allclose(
        bedroom["ctx_other_motion_any"].to_numpy(dtype=float),
        np.asarray([1.0, 1.0, 1.0], dtype=float),
    )
    np.testing.assert_allclose(
        bedroom["ctx_other_rooms_reporting"].to_numpy(dtype=float),
        np.asarray([2.0, 2.0, 2.0], dtype=float),
    )


def test_cross_room_context_features_zero_when_no_other_rooms():
    ts = pd.date_range("2025-12-06 09:00:00", periods=2, freq="10s")
    day_frames = {
        "Bedroom": pd.DataFrame(
            {
                "timestamp": ts,
                "motion": [0.0, 0.1],
                "co2": [500.0, 510.0],
            }
        ),
    }
    _add_cross_room_context_features_for_day(
        day_room_frames=day_frames,
        rooms=["Bedroom"],
    )
    bedroom = day_frames["Bedroom"]
    assert np.allclose(bedroom["ctx_other_co2_mean"].to_numpy(dtype=float), 0.0)
    assert np.allclose(bedroom["ctx_other_motion_active_count"].to_numpy(dtype=float), 0.0)
    assert np.allclose(bedroom["ctx_other_motion_any"].to_numpy(dtype=float), 0.0)
    assert np.allclose(bedroom["ctx_other_rooms_reporting"].to_numpy(dtype=float), 0.0)


def test_cross_room_context_features_respect_room_scope():
    ts = pd.date_range("2025-12-06 09:00:00", periods=2, freq="10s")
    day_frames = {
        "Bedroom": pd.DataFrame({"timestamp": ts, "motion": [0.0, 0.0], "co2": [500.0, 510.0]}),
        "LivingRoom": pd.DataFrame({"timestamp": ts, "motion": [1.0, 1.0], "co2": [700.0, 710.0]}),
    }
    _add_cross_room_context_features_for_day(
        day_room_frames=day_frames,
        rooms=["Bedroom", "LivingRoom"],
        context_rooms=["bedroom"],
    )
    assert "ctx_other_co2_mean" in day_frames["Bedroom"].columns
    assert "ctx_other_co2_mean" not in day_frames["LivingRoom"].columns


def test_add_room_temporal_occupancy_features_creates_causal_columns():
    ts = pd.date_range("2025-12-06 09:00:00", periods=20, freq="10s")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "motion": np.linspace(0.0, 1.0, num=20),
            "sound": np.linspace(10.0, 12.0, num=20),
            "light": np.linspace(200.0, 180.0, num=20),
            "co2": np.linspace(500.0, 530.0, num=20),
            "humidity": np.linspace(45.0, 50.0, num=20),
            "temperature": np.linspace(23.0, 24.0, num=20),
            "vibration": np.linspace(0.1, 0.3, num=20),
        }
    )
    cols = _add_room_temporal_occupancy_features(df)
    assert len(cols) > 0
    assert "occ_motion_d1m" in cols
    assert "occ_motion_active_2m" in cols
    assert "occ_motion_activity_ratio_15m" in cols
    assert "occ_motion_inactivity_ratio_30m" in cols
    assert "occ_co2_slope_30m" in cols
    assert "occ_co2_slope_15m" in cols
    assert "occ_light_roll_mean_10m" in cols
    assert "occ_light_roll_std_10m" in cols
    assert "occ_time_since_motion_active_minutes" in cols
    assert "occ_temp_humidity_interaction" in cols
    assert all(col in df.columns for col in cols)
    assert np.isfinite(df[cols].to_numpy(dtype=float)).all()


def test_add_room_temporal_occupancy_features_adds_bedroom_light_regime_columns():
    ts = pd.date_range("2025-12-06 20:00:00", periods=16, freq="10s")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "motion": [0.0] * 8 + [0.2] * 8,
            "sound": [10.0] * 16,
            "light": [15.0] * 4 + [0.0] * 8 + [20.0] * 4,
            "co2": np.linspace(500.0, 520.0, num=16),
            "humidity": np.linspace(45.0, 47.0, num=16),
            "temperature": np.linspace(23.0, 24.0, num=16),
            "vibration": np.linspace(0.1, 0.2, num=16),
        }
    )
    cols = _add_room_temporal_occupancy_features(
        df,
        room_key="bedroom",
        enable_bedroom_light_texture_features=True,
    )
    assert "occ_light_off_streak_30m" in cols
    assert "occ_light_regime_switch" in cols
    assert float(df["occ_light_off_streak_30m"].max()) > 0.0
    assert int(np.sum(df["occ_light_regime_switch"].to_numpy(dtype=float))) >= 1


def test_add_room_temporal_occupancy_features_does_not_add_bedroom_light_columns_without_flag():
    ts = pd.date_range("2025-12-06 20:00:00", periods=16, freq="10s")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "motion": [0.0] * 8 + [0.2] * 8,
            "sound": [10.0] * 16,
            "light": [15.0] * 4 + [0.0] * 8 + [20.0] * 4,
            "co2": np.linspace(500.0, 520.0, num=16),
            "humidity": np.linspace(45.0, 47.0, num=16),
            "temperature": np.linspace(23.0, 24.0, num=16),
            "vibration": np.linspace(0.1, 0.2, num=16),
        }
    )
    cols = _add_room_temporal_occupancy_features(df, room_key="bedroom")
    assert "occ_light_off_streak_30m" not in cols
    assert "occ_light_regime_switch" not in cols


def test_add_room_temporal_occupancy_features_night_window_starts_at_22():
    ts = pd.to_datetime(
        [
            "2025-12-06 21:50:00",
            "2025-12-06 22:00:00",
            "2025-12-07 06:50:00",
            "2025-12-07 07:00:00",
        ]
    )
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "motion": [0.0, 0.0, 0.0, 0.0],
            "sound": [10.0, 10.0, 10.0, 10.0],
            "light": [50.0, 50.0, 50.0, 50.0],
            "co2": [500.0, 500.0, 500.0, 500.0],
            "humidity": [45.0, 45.0, 45.0, 45.0],
            "temperature": [23.0, 23.0, 23.0, 23.0],
            "vibration": [0.1, 0.1, 0.1, 0.1],
        }
    )
    _add_room_temporal_occupancy_features(df)
    np.testing.assert_allclose(df["occ_is_night"].to_numpy(dtype=float), np.asarray([0.0, 1.0, 1.0, 0.0], dtype=float))


def test_add_room_temporal_occupancy_features_supports_30m_texture_profile():
    ts = pd.date_range("2025-12-06 09:00:00", periods=240, freq="10s")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "motion": np.linspace(0.0, 1.0, num=240),
            "sound": np.linspace(10.0, 12.0, num=240),
            "light": np.linspace(200.0, 180.0, num=240),
            "co2": np.linspace(500.0, 530.0, num=240),
            "humidity": np.linspace(45.0, 50.0, num=240),
            "temperature": np.linspace(23.0, 24.0, num=240),
            "vibration": np.linspace(0.1, 0.3, num=240),
        }
    )
    cols = _add_room_temporal_occupancy_features(df, bedroom_livingroom_texture_profile="30m")
    assert "occ_motion_inactivity_ratio_30m" in cols
    assert "occ_co2_slope_30m" in cols
    assert "occ_motion_inactivity_ratio_60m" not in cols
    assert "occ_co2_slope_60m" not in cols


def test_add_room_temporal_occupancy_features_supports_60m_texture_profile():
    ts = pd.date_range("2025-12-06 09:00:00", periods=480, freq="10s")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "motion": np.linspace(0.0, 1.0, num=480),
            "sound": np.linspace(10.0, 12.0, num=480),
            "light": np.linspace(200.0, 180.0, num=480),
            "co2": np.linspace(500.0, 530.0, num=480),
            "humidity": np.linspace(45.0, 50.0, num=480),
            "temperature": np.linspace(23.0, 24.0, num=480),
            "vibration": np.linspace(0.1, 0.3, num=480),
        }
    )
    cols = _add_room_temporal_occupancy_features(df, bedroom_livingroom_texture_profile="60m")
    assert "occ_motion_inactivity_ratio_60m" in cols
    assert "occ_co2_slope_60m" in cols
    assert "occ_motion_inactivity_ratio_30m" not in cols
    assert "occ_co2_slope_30m" not in cols


def test_rolling_linear_slope_matches_linear_trend():
    n = 240
    x = np.arange(n, dtype=float)
    y = 100.0 + (0.25 * x)
    s = _rolling_linear_slope(pd.Series(y), window=180)
    assert float(s.iloc[-1]) == pytest.approx(0.25, abs=1e-6)


def test_parse_tune_rooms_normalizes_case_and_whitespace():
    rooms = _parse_tune_rooms(" Entrance,bedRoom , ,LIVINGROOM")
    assert rooms == {"entrance", "bedroom", "livingroom"}


def test_build_room_model_config_enables_sequence_stage_a_for_bedroom_livingroom():
    bedroom_cfg = _build_room_model_config(
        seed=11,
        room_key="bedroom",
        enable_bedroom_livingroom_stage_a_sequence_model=True,
        bedroom_livingroom_stage_a_sequence_lag_windows=9,
    )
    livingroom_cfg = _build_room_model_config(
        seed=11,
        room_key="livingroom",
        enable_bedroom_livingroom_stage_a_sequence_model=True,
        bedroom_livingroom_stage_a_sequence_lag_windows=9,
    )
    assert bedroom_cfg.stage_a_model_type == "sequence_rf"
    assert bedroom_cfg.stage_a_temporal_lag_windows == 9
    assert livingroom_cfg.stage_a_model_type == "sequence_rf"
    assert livingroom_cfg.stage_a_temporal_lag_windows == 9


def test_build_room_model_config_keeps_default_for_other_rooms():
    kitchen_cfg = _build_room_model_config(
        seed=11,
        room_key="kitchen",
        enable_bedroom_livingroom_stage_a_sequence_model=True,
        bedroom_livingroom_stage_a_sequence_lag_windows=9,
    )
    assert kitchen_cfg.stage_a_model_type == "rf"
    assert kitchen_cfg.stage_a_temporal_lag_windows == 0


def test_build_room_model_config_supports_hgb_for_bedroom_livingroom():
    bedroom_cfg = _build_room_model_config(
        seed=11,
        room_key="bedroom",
        enable_bedroom_livingroom_stage_a_hgb=True,
    )
    livingroom_cfg = _build_room_model_config(
        seed=11,
        room_key="livingroom",
        enable_bedroom_livingroom_stage_a_hgb=True,
    )
    assert bedroom_cfg.stage_a_model_type == "hgb"
    assert livingroom_cfg.stage_a_model_type == "hgb"


def test_build_room_model_config_supports_sequence_hgb_for_bedroom_livingroom():
    bedroom_cfg = _build_room_model_config(
        seed=11,
        room_key="bedroom",
        enable_bedroom_livingroom_stage_a_hgb=True,
        enable_bedroom_livingroom_stage_a_sequence_model=True,
        bedroom_livingroom_stage_a_sequence_lag_windows=8,
    )
    livingroom_cfg = _build_room_model_config(
        seed=11,
        room_key="livingroom",
        enable_bedroom_livingroom_stage_a_hgb=True,
        enable_bedroom_livingroom_stage_a_sequence_model=True,
        bedroom_livingroom_stage_a_sequence_lag_windows=8,
    )
    assert bedroom_cfg.stage_a_model_type == "sequence_hgb"
    assert livingroom_cfg.stage_a_model_type == "sequence_hgb"
    assert bedroom_cfg.stage_a_temporal_lag_windows == 8
    assert livingroom_cfg.stage_a_temporal_lag_windows == 8


def test_build_room_model_config_supports_sequence_transformer_for_bedroom_livingroom():
    bedroom_cfg = _build_room_model_config(
        seed=11,
        room_key="bedroom",
        enable_bedroom_livingroom_stage_a_transformer=True,
        bedroom_livingroom_stage_a_sequence_lag_windows=10,
        bedroom_livingroom_stage_a_transformer_epochs=3,
        bedroom_livingroom_stage_a_transformer_batch_size=64,
        bedroom_livingroom_stage_a_transformer_learning_rate=5e-4,
        bedroom_livingroom_stage_a_transformer_hidden_dim=56,
        bedroom_livingroom_stage_a_transformer_num_heads=4,
        bedroom_livingroom_stage_a_transformer_dropout=0.2,
        bedroom_livingroom_stage_a_transformer_class_weight_power=0.6,
        bedroom_livingroom_stage_a_transformer_conv_kernel_size=5,
        bedroom_livingroom_stage_a_transformer_conv_blocks=3,
        enable_bedroom_livingroom_stage_a_transformer_sequence_filter=False,
    )
    livingroom_cfg = _build_room_model_config(
        seed=11,
        room_key="livingroom",
        enable_bedroom_livingroom_stage_a_transformer=True,
        bedroom_livingroom_stage_a_sequence_lag_windows=10,
        bedroom_livingroom_stage_a_transformer_epochs=3,
        bedroom_livingroom_stage_a_transformer_batch_size=64,
        bedroom_livingroom_stage_a_transformer_learning_rate=5e-4,
        bedroom_livingroom_stage_a_transformer_hidden_dim=56,
        bedroom_livingroom_stage_a_transformer_num_heads=4,
        bedroom_livingroom_stage_a_transformer_dropout=0.2,
        bedroom_livingroom_stage_a_transformer_class_weight_power=0.6,
        bedroom_livingroom_stage_a_transformer_conv_kernel_size=5,
        bedroom_livingroom_stage_a_transformer_conv_blocks=3,
        enable_bedroom_livingroom_stage_a_transformer_sequence_filter=False,
    )
    assert bedroom_cfg.stage_a_model_type == "sequence_transformer"
    assert livingroom_cfg.stage_a_model_type == "sequence_transformer"
    assert bedroom_cfg.stage_a_temporal_lag_windows == 10
    assert livingroom_cfg.stage_a_temporal_lag_windows == 10
    assert bedroom_cfg.stage_a_transformer_epochs == 3
    assert livingroom_cfg.stage_a_transformer_batch_size == 64
    assert bedroom_cfg.stage_a_class_weight == "balanced_sqrt"
    assert bedroom_cfg.stage_a_transformer_learning_rate == pytest.approx(5e-4)
    assert bedroom_cfg.stage_a_transformer_hidden_dim == 56
    assert livingroom_cfg.stage_a_transformer_num_heads == 4
    assert bedroom_cfg.stage_a_transformer_dropout == pytest.approx(0.2)
    assert livingroom_cfg.stage_a_transformer_class_weight_power == pytest.approx(0.6)
    assert bedroom_cfg.stage_a_transformer_conv_kernel_size == 5
    assert livingroom_cfg.stage_a_transformer_conv_blocks == 3
    assert bedroom_cfg.stage_a_transformer_use_sequence_filter is False


def test_build_room_model_config_regime_routing_raises_stage_a_capacity_for_low_occupancy():
    low_cfg = _build_room_model_config(
        seed=11,
        room_key="livingroom",
        bedroom_livingroom_regime="low_occupancy",
        enable_bedroom_livingroom_regime_routing=True,
    )
    normal_cfg = _build_room_model_config(
        seed=11,
        room_key="livingroom",
        bedroom_livingroom_regime="normal",
        enable_bedroom_livingroom_regime_routing=True,
    )
    assert int(low_cfg.n_estimators_stage_a) > int(normal_cfg.n_estimators_stage_a)


def test_parse_room_label_thresholds_parses_valid_pairs():
    out = _parse_room_label_thresholds(
        "bedroom.sleep=0.35,bathroom.shower=0.40,bad,entrance=0.5, kitchen.cooking=1.2"
    )
    assert out["bedroom"]["sleep"] == 0.35
    assert out["bathroom"]["shower"] == 0.4
    assert out["kitchen"]["cooking"] == 1.0
    assert "entrance" not in out


def test_parse_hour_range_parses_and_falls_back():
    assert _parse_hour_range("22-7", default_start=8, default_end=20) == (22, 7)
    assert _parse_hour_range("bad", default_start=8, default_end=20) == (8, 20)


def test_parse_hour_ranges_parses_multiple_ranges():
    out = _parse_hour_ranges("6-10,11-14,17-20", default_ranges=[(0, 1)])
    assert out == [(6, 10), (11, 14), (17, 20)]
    assert _parse_hour_ranges("bad", default_ranges=[(1, 2)]) == [(1, 2)]


def test_apply_critical_label_rescue_inserts_label_when_score_high():
    y_pred = ["unoccupied", "bathroom_normal_use", "unoccupied"]
    activity_probs = {
        "shower": [0.01, 0.20, 0.03],
        "bathroom_normal_use": [0.2, 0.6, 0.3],
    }
    rescued, debug = _apply_critical_label_rescue(
        room_key="bathroom",
        y_pred=y_pred,
        activity_probs=activity_probs,
        rescue_min_scores={"bathroom": {"shower": 0.15}},
    )
    assert rescued[1] == "shower"
    assert debug["applied"] is True


def test_apply_bathroom_shower_fallback_marks_long_humid_run():
    ts = pd.date_range("2025-12-08 08:00:00", periods=120, freq="10s")
    y_pred = ["unoccupied"] * 120
    # Occupied run from 30..90 without shower label.
    for i in range(30, 90):
        y_pred[i] = "bathroom_normal_use"
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "humidity": [45.0] * 30 + [45.0 + (i - 30) * 0.08 for i in range(30, 90)] + [45.0] * 30,
            "motion": [0.0] * 30 + [0.8] * 60 + [0.0] * 30,
        }
    )
    activity_probs = {"shower": [0.01] * 120}
    out, debug = _apply_bathroom_shower_fallback(
        room_key="bathroom",
        y_pred=y_pred,
        timestamps=df["timestamp"],
        test_df=df,
        activity_probs=activity_probs,
        min_duration_minutes=8.0,
    )
    assert debug["applied"] is True
    assert any(label == "shower" for label in out[30:90])


def test_apply_bathroom_shower_fallback_skips_when_shower_exists():
    ts = pd.date_range("2025-12-08 08:00:00", periods=20, freq="10s")
    y_pred = ["unoccupied"] * 10 + ["shower"] * 10
    df = pd.DataFrame({"timestamp": ts, "humidity": [50.0] * 20, "motion": [0.1] * 20})
    out, debug = _apply_bathroom_shower_fallback(
        room_key="bathroom",
        y_pred=y_pred,
        timestamps=df["timestamp"],
        test_df=df,
        activity_probs={"shower": [0.2] * 20},
    )
    assert list(out) == y_pred
    assert debug["applied"] is False


def test_apply_bathroom_shower_fallback_uses_humidity_peak_when_no_occupied_run():
    ts = pd.date_range("2025-12-08 08:00:00", periods=180, freq="10s")
    y_pred = ["unoccupied"] * 180
    # Create a humidity bump around the middle without occupied labels.
    humidity = [45.0] * 70 + [45.0 + (i - 70) * 0.1 for i in range(70, 100)] + [48.0] * 10 + [48.0 - (i - 110) * 0.1 for i in range(110, 140)] + [45.0] * 40
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "humidity": humidity,
            "motion": [0.2] * 180,
        }
    )
    out, debug = _apply_bathroom_shower_fallback(
        room_key="bathroom",
        y_pred=y_pred,
        timestamps=df["timestamp"],
        test_df=df,
        activity_probs={"shower": [0.01] * 180},
    )
    assert debug["applied"] is True
    assert debug["reason"] in {"humidity_peak_fallback", "fallback_run_selected"}
    assert any(label == "shower" for label in out)


def test_fit_affine_duration_calibrator_returns_enabled_with_pairs():
    cal = _fit_affine_duration_calibrator([100.0, 200.0, 300.0], [120.0, 220.0, 320.0])
    assert cal["enabled"] is True
    assert 0.25 <= float(cal["slope"]) <= 1.75


def test_fit_affine_duration_calibrator_single_pair_uses_ratio_correction():
    cal = _fit_affine_duration_calibrator([200.0], [100.0])
    assert cal["enabled"] is True
    assert float(cal["intercept"]) == 0.0
    assert float(cal["slope"]) == 0.5


def test_care_fragmentation_score_is_symmetric_and_not_overly_punitive():
    y_true = np.asarray(["sleep", "sleep", "unoccupied", "sleep", "sleep"], dtype=object)
    y_pred = np.asarray(["sleep", "unoccupied", "sleep", "unoccupied", "sleep"], dtype=object)
    score = _care_fragmentation_score(y_true, y_pred)
    assert 0.0 <= float(score) <= 1.0
    # Pred has 3 care runs vs 2 in truth: non-zero penalty, but should not collapse to 0.
    assert float(score) > 0.0


def test_care_fragmentation_score_can_skip_prediction_resmoothing():
    y_true = np.asarray(["sleep", "sleep", "unoccupied", "sleep", "sleep"], dtype=object)
    y_pred = np.asarray(["sleep", "sleep", "sleep", "sleep", "sleep"], dtype=object)
    smoothed = _care_fragmentation_score(
        y_true,
        y_pred,
        min_run_windows=3,
        gap_fill_windows=2,
        smooth_pred_mask=True,
    )
    unsmoothed = _care_fragmentation_score(
        y_true,
        y_pred,
        min_run_windows=3,
        gap_fill_windows=2,
        smooth_pred_mask=False,
    )
    assert 0.0 <= float(smoothed) <= 1.0
    assert 0.0 <= float(unsmoothed) <= 1.0


def test_apply_room_prediction_occupancy_smoothing_updates_final_labels():
    y_pred = np.asarray(
        [
            "sleep",
            "sleep",
            "unoccupied",
            "sleep",
            "sleep",
            "unoccupied",
            "unoccupied",
        ],
        dtype=object,
    )
    activity_probs = {
        "sleep": np.asarray([0.8, 0.8, 0.7, 0.8, 0.8, 0.1, 0.1], dtype=float),
        "bedroom_normal_use": np.asarray([0.2, 0.2, 0.3, 0.2, 0.2, 0.9, 0.9], dtype=float),
    }
    out, debug = _apply_room_prediction_occupancy_smoothing(
        room_key="bedroom",
        y_pred=y_pred,
        activity_probs=activity_probs,
        min_run_windows=2,
        gap_fill_windows=1,
    )
    assert debug["applied"] is True
    assert int(debug["changed_windows"]) >= 1
    # short 1-window internal gap is filled
    assert str(out[2]) == "sleep"
    # trailing 2-window run remains unoccupied
    assert str(out[-1]) == "unoccupied"


def test_apply_room_passive_occupancy_hysteresis_extends_low_motion_passive_occupancy():
    y_pred = np.asarray(
        [
            "livingroom_normal_use",
            "livingroom_normal_use",
            "unoccupied",
            "unoccupied",
            "unoccupied",
            "unoccupied",
        ],
        dtype=object,
    )
    occupancy_probs = np.asarray([0.80, 0.74, 0.28, 0.24, 0.20, 0.18], dtype=float)
    activity_probs = {
        "livingroom_normal_use": np.asarray([0.70, 0.68, 0.25, 0.24, 0.23, 0.20], dtype=float),
    }
    motion = np.asarray([1.2, 0.8, 0.05, 0.05, 0.04, 0.04], dtype=float)
    out, debug = _apply_room_passive_occupancy_hysteresis(
        room_key="livingroom",
        y_pred=y_pred,
        occupancy_probs=occupancy_probs,
        activity_probs=activity_probs,
        motion_values=motion,
        hold_minutes=1.0,
        exit_min_consecutive_windows=3,
    )
    assert debug["applied"] is True
    assert int(debug["after_occupied_windows"]) > int(debug["before_occupied_windows"])
    assert str(out[2]) == "livingroom_normal_use"
    assert str(out[3]) == "livingroom_normal_use"


def test_apply_room_passive_occupancy_hysteresis_noop_for_other_rooms():
    out, debug = _apply_room_passive_occupancy_hysteresis(
        room_key="kitchen",
        y_pred=np.asarray(["kitchen_normal_use", "unoccupied"], dtype=object),
        occupancy_probs=np.asarray([0.9, 0.1], dtype=float),
        activity_probs={"kitchen_normal_use": np.asarray([0.9, 0.1], dtype=float)},
    )
    assert list(out) == ["kitchen_normal_use", "unoccupied"]
    assert debug["applied"] is False
    assert debug["reason"] == "not_applicable"


def test_apply_room_passive_occupancy_hysteresis_hold_minutes_zero_disables_room():
    y_pred = np.asarray(
        ["sleep", "sleep", "unoccupied", "sleep"],
        dtype=object,
    )
    out, debug = _apply_room_passive_occupancy_hysteresis(
        room_key="bedroom",
        y_pred=y_pred,
        occupancy_probs=np.asarray([0.9, 0.8, 0.2, 0.7], dtype=float),
        activity_probs={"sleep": np.asarray([0.9, 0.8, 0.1, 0.6], dtype=float)},
        hold_minutes=0.0,
    )
    assert list(out) == list(y_pred)
    assert debug["applied"] is False
    assert debug["reason"] == "disabled_by_hold_minutes"


def test_apply_room_passive_occupancy_hysteresis_strict_entry_blocks_weak_base_entry():
    y_pred = np.asarray(
        [
            "unoccupied",
            "livingroom_normal_use",
            "unoccupied",
            "unoccupied",
        ],
        dtype=object,
    )
    occupancy_probs = np.asarray([0.10, 0.30, 0.24, 0.20], dtype=float)
    activity_probs = {
        "livingroom_normal_use": np.asarray([0.08, 0.32, 0.18, 0.10], dtype=float),
    }
    motion = np.asarray([0.02, 0.12, 0.06, 0.04], dtype=float)
    out, debug = _apply_room_passive_occupancy_hysteresis(
        room_key="livingroom",
        y_pred=y_pred,
        occupancy_probs=occupancy_probs,
        activity_probs=activity_probs,
        motion_values=motion,
        hold_minutes=2.0,
        livingroom_strict_entry_requires_strong_signal=True,
        livingroom_entry_motion_threshold=0.70,
    )
    assert debug["applied"] is True
    assert bool(debug["config"]["livingroom_strict_entry_requires_strong_signal"]) is True
    assert list(out) == ["unoccupied", "unoccupied", "unoccupied", "unoccupied"]


def test_compute_binary_episode_metrics_reports_episode_and_window_scores():
    ts = pd.date_range("2025-12-07 10:00:00", periods=8, freq="10s")
    y_true = np.asarray(
        ["unoccupied", "sleep", "sleep", "sleep", "unoccupied", "sleep", "sleep", "unoccupied"],
        dtype=object,
    )
    y_pred = np.asarray(
        ["unoccupied", "sleep", "sleep", "unoccupied", "unoccupied", "sleep", "unoccupied", "unoccupied"],
        dtype=object,
    )
    out = _compute_binary_episode_metrics(
        timestamps=pd.Series(ts),
        y_true=y_true,
        y_pred=y_pred,
    )
    assert out["windows_total"] == 8
    assert out["occupied_windows_true"] == 5
    assert out["occupied_windows_pred"] == 3
    assert 0.0 <= float(out["episode_precision"]) <= 1.0
    assert 0.0 <= float(out["episode_recall"]) <= 1.0
    assert "timeline_metrics_binary" in out


def test_build_home_empty_status_requires_all_rooms_unoccupied():
    ts = pd.to_datetime(["2025-12-07 10:00:00", "2025-12-07 10:00:00", "2025-12-07 10:00:10", "2025-12-07 10:00:10"])
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "room": ["bedroom", "livingroom", "bedroom", "livingroom"],
            "label": ["unoccupied", "unoccupied", "sleep", "unoccupied"],
        }
    )
    status = _build_home_empty_status(
        df,
        label_col="label",
        min_empty_duration_seconds=0.0,
    )
    assert status[pd.Timestamp("2025-12-07 10:00:00")] is True
    assert status[pd.Timestamp("2025-12-07 10:00:10")] is False


def test_build_home_empty_status_aligns_misaligned_room_timestamps():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-12-07 10:00:00",
                    "2025-12-07 10:00:10",
                    "2025-12-07 10:00:02",
                    "2025-12-07 10:00:12",
                ]
            ),
            "room": ["bedroom", "bedroom", "livingroom", "livingroom"],
            "label": ["unoccupied", "sleep", "unoccupied", "unoccupied"],
        }
    )
    status = _build_home_empty_status(
        df,
        label_col="label",
        rooms=["bedroom", "livingroom"],
        reference_timestamps=pd.Series(pd.to_datetime(["2025-12-07 10:00:00", "2025-12-07 10:00:10"])),
        alignment_tolerance_seconds=5.0,
        require_full_coverage=True,
        min_empty_duration_seconds=0.0,
    )
    assert status[pd.Timestamp("2025-12-07 10:00:00")] is True
    assert status[pd.Timestamp("2025-12-07 10:00:10")] is False


def test_build_home_empty_status_requires_full_coverage_when_enabled():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-12-07 10:00:00", "2025-12-07 10:00:02"]),
            "room": ["bedroom", "livingroom"],
            "label": ["unoccupied", "unoccupied"],
        }
    )
    status = _build_home_empty_status(
        df,
        label_col="label",
        rooms=["bedroom", "livingroom"],
        reference_timestamps=pd.Series(pd.to_datetime(["2025-12-07 10:00:00"])),
        alignment_tolerance_seconds=0.0,
        require_full_coverage=True,
        min_empty_duration_seconds=0.0,
    )
    assert status[pd.Timestamp("2025-12-07 10:00:00")] is False


def test_compute_home_empty_metrics_reports_false_empty_rate():
    ts = pd.to_datetime(["2025-12-07 10:00:00", "2025-12-07 10:00:10"])
    gt = pd.DataFrame(
        {
            "timestamp": [ts[0], ts[0], ts[1], ts[1]],
            "room": ["bedroom", "livingroom", "bedroom", "livingroom"],
            "label": ["unoccupied", "unoccupied", "sleep", "unoccupied"],
        }
    )
    pred = pd.DataFrame(
        {
            "timestamp": [ts[0], ts[0], ts[1], ts[1]],
            "room": ["bedroom", "livingroom", "bedroom", "livingroom"],
            "label": ["unoccupied", "unoccupied", "unoccupied", "unoccupied"],
        }
    )
    out = _compute_home_empty_metrics(
        pred_df=pred,
        gt_df=gt,
        tolerance_seconds=1.0,
        min_empty_duration_seconds=0.0,
    )
    assert out["evaluated"] is True
    assert out["tp"] == 1
    assert out["fp"] == 1
    assert out["fn"] == 0
    assert out["tn"] == 0
    assert out["home_empty_precision"] == pytest.approx(0.5, abs=1e-6)
    assert out["home_empty_recall"] == pytest.approx(1.0, abs=1e-6)
    assert out["home_empty_false_empty_rate"] == pytest.approx(1.0, abs=1e-6)


def test_classification_metrics_include_occupied_binary_metrics():
    y_true = np.asarray(["sleep", "sleep", "unoccupied", "unoccupied"], dtype=object)
    y_pred = np.asarray(["sleep", "unoccupied", "sleep", "unoccupied"], dtype=object)
    m = _classification_metrics(y_true, y_pred)
    assert "occupied_precision" in m
    assert "occupied_recall" in m
    assert "occupied_f1" in m
    assert 0.0 <= float(m["occupied_f1"]) <= 1.0


def test_adaptive_room_occupancy_threshold_respects_room_scope():
    thr, dbg = _adaptive_room_occupancy_threshold(
        room_key="kitchen",
        base_threshold=0.35,
        occupancy_probs=np.asarray([0.2, 0.4, 0.6], dtype=float),
        y_fit_labels=np.asarray(["unoccupied", "kitchen_normal_use"], dtype=object),
    )
    assert thr == 0.35
    assert dbg["applied"] is False


def test_adaptive_room_occupancy_threshold_applies_for_livingroom():
    probs = np.asarray([0.05, 0.10, 0.20, 0.40, 0.70, 0.90], dtype=float)
    y_fit = np.asarray(["unoccupied", "livingroom_normal_use", "livingroom_normal_use"], dtype=object)
    thr, dbg = _adaptive_room_occupancy_threshold(
        room_key="livingroom",
        base_threshold=0.35,
        occupancy_probs=probs,
        y_fit_labels=y_fit,
    )
    assert dbg["applied"] is True
    assert 0.20 <= float(thr) <= 0.50


def test_apply_duration_calibration_updates_target_key():
    pred_daily = {"sleep_minutes": 100.0, "sleep_day": 1.0}
    out = _apply_duration_calibration(
        room_key="bedroom",
        pred_daily=pred_daily,
        calibrator={"enabled": True, "slope": 1.2, "intercept": 10.0},
    )
    assert out["sleep_minutes"] == 130.0
    assert out["sleep_day"] == 1.0


def test_calibrate_duration_from_train_prefers_isotonic_with_enough_points():
    res = _calibrate_duration_from_train(
        pred_values=[50.0, 100.0, 200.0, 300.0, 400.0],
        gt_values=[60.0, 110.0, 210.0, 260.0, 280.0],
        raw_value=250.0,
    )
    assert res["enabled"] is True
    assert res["method"] in {"isotonic", "affine"}
    assert 0.0 <= float(res["corrected_value"]) <= 1440.0


def test_calibrate_duration_from_train_keeps_affine_when_raw_is_best():
    res = _calibrate_duration_from_train(
        pred_values=[100.0, 200.0, 300.0, 400.0],
        gt_values=[100.0, 200.0, 300.0, 400.0],
        raw_value=250.0,
    )
    assert res["enabled"] is True
    assert res["method"] == "affine"
    assert float(res["corrected_value"]) == 250.0


def test_calibrate_duration_from_train_kitchen_uses_shrunk_affine():
    res = _calibrate_duration_from_train(
        pred_values=[120.0, 180.0, 300.0, 360.0, 280.0],
        gt_values=[140.0, 220.0, 260.0, 320.0, 250.0],
        raw_value=340.0,
        room_key="kitchen",
        enable_kitchen_robust_duration_calibration=True,
    )
    assert res["enabled"] is True
    assert res["method"] == "kitchen_affine_shrunk"
    assert 0.0 <= float(res["corrected_value"]) <= 1440.0


class _DummyKitchenModel:
    def __init__(self, baseline_threshold: float = 0.55) -> None:
        self._baseline_threshold = float(baseline_threshold)

    def get_operating_points(self):
        return {"occupancy_threshold": self._baseline_threshold}

    def predict(self, x, *, occupancy_threshold=None, label_thresholds=None):
        thr = float(self._baseline_threshold if occupancy_threshold is None else occupancy_threshold)
        # x[:,0] encodes occupancy strength.
        occupied = np.asarray(x[:, 0], dtype=float) >= thr
        out = np.full(shape=(x.shape[0],), fill_value="unoccupied", dtype=object)
        out[occupied] = "kitchen_normal_use"
        return out

    def predict_activity_proba(self, x):
        n = int(x.shape[0])
        return {"kitchen_normal_use": np.ones(shape=(n,), dtype=float)}


class _DummyRoomHardGateModel:
    def __init__(self, baseline_threshold: float = 0.55, active_label: str = "sleep") -> None:
        self._baseline_threshold = float(baseline_threshold)
        self._active_label = str(active_label)

    def get_operating_points(self):
        return {"occupancy_threshold": self._baseline_threshold}

    def predict(self, x, *, occupancy_threshold=None, label_thresholds=None):
        thr = float(self._baseline_threshold if occupancy_threshold is None else occupancy_threshold)
        occupied = np.asarray(x[:, 0], dtype=float) >= thr
        out = np.full(shape=(x.shape[0],), fill_value="unoccupied", dtype=object)
        out[occupied] = self._active_label
        return out

    def predict_activity_proba(self, x):
        n = int(x.shape[0])
        return {self._active_label: np.ones(shape=(n,), dtype=float)}

    def predict_occupancy_proba(self, x):
        return np.asarray(x[:, 0], dtype=float)


def test_daily_target_series_extracts_per_day_minutes():
    ts = pd.to_datetime(
        [
            "2025-12-04 08:00:00",
            "2025-12-04 08:00:10",
            "2025-12-05 09:00:00",
            "2025-12-05 09:00:10",
        ]
    )
    df = pd.DataFrame({"timestamp": ts, "activity": ["kitchen_normal_use", "unoccupied", "unoccupied", "unoccupied"]})
    vals = _daily_target_series(df=df, label_col="activity", target_key="kitchen_use_minutes")
    assert len(vals) == 2
    assert vals[0] > 0.0
    assert vals[1] == 0.0


def test_mean_abs_error_handles_empty_sequences():
    assert _mean_abs_error([], [1.0]) == float("inf")
    assert _mean_abs_error([1.0], []) == float("inf")


def test_duration_mae_from_labels_returns_finite_value():
    ts = pd.date_range("2025-12-04 08:00:00", periods=12, freq="10s")
    calib_df = pd.DataFrame(
        {
            "timestamp": ts,
            "activity": ["sleep"] * 6 + ["unoccupied"] * 6,
        }
    )
    y_pred = np.asarray(["sleep"] * 3 + ["unoccupied"] * 9, dtype=object)
    mae = _duration_mae_from_labels(
        calib_df=calib_df,
        pred_labels=y_pred,
        target_key="sleep_minutes",
    )
    assert np.isfinite(mae)
    assert mae >= 0.0


def test_tune_kitchen_occupancy_threshold_by_mae_selects_lower_threshold():
    ts = pd.date_range("2025-12-04 08:00:00", periods=12, freq="10s")
    # First 6 rows are true kitchen use but moderate score (~0.4), next 6 unoccupied (~0.1).
    x0 = [0.40] * 6 + [0.10] * 6
    calib_df = pd.DataFrame(
        {
            "timestamp": ts,
            "activity": ["kitchen_normal_use"] * 6 + ["unoccupied"] * 6,
            "f0": x0,
        }
    )
    model = _DummyKitchenModel(baseline_threshold=0.55)
    payload = _tune_kitchen_occupancy_threshold_by_mae(
        model=model,
        calib_df=calib_df,
        feat_cols=["f0"],
        room_label_threshold={},
        critical_label_rescue_min_scores={},
        threshold_grid=np.asarray([0.35, 0.55], dtype=float),
        max_threshold_delta=0.50,
        min_required_mae_improvement=0.0,
        stability_penalty_weight=0.0,
        threshold_delta_penalty_weight=0.0,
    )
    assert payload["used"] is True
    assert payload["selected_occupancy_threshold"] == 0.35
    assert payload["selected_daily_mae_minutes"] <= payload["baseline_daily_mae_minutes"]


def test_tune_kitchen_occupancy_threshold_by_mae_can_keep_baseline():
    ts = pd.date_range("2025-12-04 08:00:00", periods=12, freq="10s")
    x0 = [0.60] * 6 + [0.05] * 6
    calib_df = pd.DataFrame(
        {
            "timestamp": ts,
            "activity": ["kitchen_normal_use"] * 6 + ["unoccupied"] * 6,
            "f0": x0,
        }
    )
    model = _DummyKitchenModel(baseline_threshold=0.55)
    payload = _tune_kitchen_occupancy_threshold_by_mae(
        model=model,
        calib_df=calib_df,
        feat_cols=["f0"],
        room_label_threshold={},
        critical_label_rescue_min_scores={},
        threshold_grid=np.asarray([0.35, 0.55], dtype=float),
        min_required_mae_improvement=10.0,
    )
    assert payload["used"] is True
    assert payload["adopted"] is False
    assert payload["selected_occupancy_threshold"] == 0.55


def test_tune_room_occupancy_threshold_for_hard_gate_selects_recall_recovery_threshold():
    ts = pd.date_range("2025-12-04 22:00:00", periods=20, freq="10s")
    # Positive windows have moderate occupancy score; baseline threshold misses them.
    x0 = [0.42] * 10 + [0.12] * 10
    calib_df = pd.DataFrame(
        {
            "timestamp": ts,
            "activity": ["sleep"] * 10 + ["unoccupied"] * 10,
            "f0": x0,
        }
    )
    model = _DummyRoomHardGateModel(baseline_threshold=0.55, active_label="sleep")
    payload = _tune_room_occupancy_threshold_for_hard_gate(
        model=model,
        room_key="bedroom",
        calib_df=calib_df,
        feat_cols=["f0"],
        room_label_threshold={},
        critical_label_rescue_min_scores={},
        threshold_grid=np.asarray([0.35, 0.55], dtype=float),
        max_threshold_delta=0.30,
        recall_floor=0.50,
        enable_room_occupancy_decoder=False,
    )
    assert payload["used"] is True
    assert payload["adopted"] is True
    assert payload["selected_occupancy_threshold"] == 0.35
    assert payload["selected_occupied_recall"] >= payload["baseline_occupied_recall"]


def test_tune_room_occupancy_threshold_for_hard_gate_rejects_unsupported_room():
    payload = _tune_room_occupancy_threshold_for_hard_gate(
        model=_DummyRoomHardGateModel(),
        room_key="kitchen",
        calib_df=pd.DataFrame({"timestamp": [], "activity": [], "f0": []}),
        feat_cols=["f0"],
        room_label_threshold={},
        critical_label_rescue_min_scores={},
    )
    assert payload["used"] is False
    assert payload["reason"] == "room_not_supported"


def test_tune_room_occupancy_threshold_for_hard_gate_applies_duration_guardrail():
    ts = pd.date_range("2025-12-04 22:00:00", periods=120, freq="10s")
    # Candidate low threshold recovers recall but wildly over-predicts occupied duration.
    x0 = [0.42] * 10 + [0.38] * 110
    calib_df = pd.DataFrame(
        {
            "timestamp": ts,
            "activity": ["sleep"] * 10 + ["unoccupied"] * 110,
            "f0": x0,
        }
    )
    model = _DummyRoomHardGateModel(baseline_threshold=0.55, active_label="sleep")
    payload = _tune_room_occupancy_threshold_for_hard_gate(
        model=model,
        room_key="bedroom",
        calib_df=calib_df,
        feat_cols=["f0"],
        room_label_threshold={},
        critical_label_rescue_min_scores={},
        threshold_grid=np.asarray([0.35, 0.55], dtype=float),
        max_threshold_delta=0.30,
        recall_floor=0.50,
        max_allowed_duration_mae_increase_minutes=0.2,
        max_allowed_duration_mae_ratio=1.05,
        enable_room_occupancy_decoder=False,
    )
    assert payload["used"] is True
    assert payload["adopted"] is False
    assert payload["selected_occupancy_threshold"] == 0.55
    assert payload["duration_guardrail_pass"] is True
    assert payload["baseline_duration_mae_minutes"] is not None


def test_split_hard_gate_skips_label_recall_floor_when_train_day_support_is_missing():
    gate = _split_hard_gate(
        "livingroom",
        gt_daily={},
        pred_daily={},
        cls={
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "occupied_precision": 0.90,
            "occupied_recall": 0.90,
            "occupied_f1": 0.90,
        },
        label_recall_summary={"livingroom_normal_use": {"support": 1000.0, "recall": 0.10}},
        fragmentation_score=0.90,
        unknown_rate=0.0,
        unknown_rate_cap=0.20,
        room_metric_floors={"livingroom": {"occupied_f1": 0.50, "occupied_recall": 0.50, "fragmentation_score": 0.40}},
        room_label_recall_floors={"livingroom": {"livingroom_normal_use": 0.80}},
        room_label_recall_min_supports={"livingroom": 100},
        label_recall_train_day_support={
            "livingroom_normal_use": {
                "all_days_supported": False,
                "zero_support_days": [17],
                "day_support": {"4": 120, "5": 220, "6": 160, "17": 0},
            }
        },
    )
    assert gate["pass"] is True


def test_split_hard_gate_enforces_label_recall_floor_when_train_day_support_exists():
    gate = _split_hard_gate(
        "livingroom",
        gt_daily={},
        pred_daily={},
        cls={
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "occupied_precision": 0.90,
            "occupied_recall": 0.90,
            "occupied_f1": 0.90,
        },
        label_recall_summary={"livingroom_normal_use": {"support": 1000.0, "recall": 0.10}},
        fragmentation_score=0.90,
        unknown_rate=0.0,
        unknown_rate_cap=0.20,
        room_metric_floors={"livingroom": {"occupied_f1": 0.50, "occupied_recall": 0.50, "fragmentation_score": 0.40}},
        room_label_recall_floors={"livingroom": {"livingroom_normal_use": 0.80}},
        room_label_recall_min_supports={"livingroom": 100},
        label_recall_train_day_support={
            "livingroom_normal_use": {
                "all_days_supported": True,
                "zero_support_days": [],
                "day_support": {"4": 120, "5": 220, "6": 160, "7": 80},
            }
        },
    )
    assert gate["pass"] is False
    assert any("recall_livingroom_normal_use_lt" in str(reason) for reason in gate["reasons"])


def test_split_hard_gate_marks_ineligible_when_train_days_below_minimum():
    gate = _split_hard_gate(
        "bedroom",
        gt_daily={"sleep_day": 1.0},
        pred_daily={"sleep_day": 1.0},
        cls={
            "accuracy": 0.9,
            "macro_f1": 0.8,
            "macro_precision": 0.8,
            "macro_recall": 0.8,
            "occupied_precision": 0.8,
            "occupied_recall": 0.8,
            "occupied_f1": 0.8,
        },
        label_recall_summary={"sleep": {"support": 1000.0, "recall": 0.8}},
        fragmentation_score=0.9,
        unknown_rate=0.0,
        unknown_rate_cap=0.2,
        room_metric_floors={"bedroom": {"occupied_f1": 0.5, "occupied_recall": 0.5, "fragmentation_score": 0.4}},
        room_label_recall_floors={"bedroom": {"sleep": 0.4}},
        room_label_recall_min_supports={"bedroom": 100},
        n_train_days=2,
        hard_gate_min_train_days=3,
    )
    assert gate["eligible"] is False
    assert gate["pass"] is True
    assert "not_eligible_below_min_train_days" in gate["reasons"]


def test_apply_kitchen_temporal_decoder_fills_short_internal_gap():
    y_pred = np.array(
        ["unoccupied"] * 2
        + ["kitchen_normal_use"] * 6
        + ["unoccupied"] * 3
        + ["kitchen_normal_use"] * 6
        + ["unoccupied"] * 2,
        dtype=object,
    )
    n = len(y_pred)
    occ = np.array([0.1] * 2 + [0.7] * 6 + [0.2] * 3 + [0.7] * 6 + [0.1] * 2, dtype=float)
    act = {"kitchen_normal_use": np.array([0.05] * 2 + [0.65] * 6 + [0.08] * 3 + [0.62] * 6 + [0.05] * 2)}
    out, debug = _apply_kitchen_temporal_decoder(
        y_pred=y_pred,
        occupancy_probs=occ,
        activity_probs=act,
        gap_fill_windows=6,
        min_run_windows=3,
    )
    # Internal 3-window gap should be filled as kitchen.
    assert all(label == "kitchen_normal_use" for label in out[8:11])
    assert debug["applied"] is True


def test_apply_kitchen_temporal_decoder_drops_short_low_confidence_burst():
    y_pred = np.array(["unoccupied"] * 5 + ["kitchen_normal_use"] * 2 + ["unoccupied"] * 5, dtype=object)
    occ = np.array([0.1] * len(y_pred), dtype=float)
    act = {"kitchen_normal_use": np.array([0.05] * 5 + [0.2, 0.22] + [0.05] * 5)}
    out, debug = _apply_kitchen_temporal_decoder(
        y_pred=y_pred,
        occupancy_probs=occ,
        activity_probs=act,
        min_run_windows=4,
        keep_short_run_if_prob_ge=0.8,
    )
    assert all(label == "unoccupied" for label in out)
    assert debug["applied"] is True


def test_build_kitchen_stage_a_sample_weights_emphasizes_occupied_and_transitions():
    labels = (
        ["unoccupied"] * 260
        + ["kitchen_normal_use"] * 20
        + ["unoccupied"] * 260
    )
    w = _build_kitchen_stage_a_sample_weights(labels)
    assert len(w) == len(labels)
    occ = np.asarray(labels, dtype=object) != "unoccupied"
    assert float(np.mean(w[occ])) > float(np.mean(w[~occ]))
    # Long unoccupied mid-run should be downweighted.
    assert float(np.min(w)) <= 0.7


def test_extract_error_episodes_returns_top_fn_fp():
    ts = pd.date_range("2025-12-06 09:00:00", periods=12, freq="10s")
    y_true = np.array(
        ["kitchen_normal_use"] * 4 + ["unoccupied"] * 4 + ["kitchen_normal_use"] * 4,
        dtype=object,
    )
    y_pred = np.array(
        ["unoccupied"] * 3 + ["kitchen_normal_use"] * 5 + ["unoccupied"] * 4,
        dtype=object,
    )
    out = _extract_error_episodes(
        timestamps=pd.Series(ts),
        y_true=y_true,
        y_pred=y_pred,
        positive_label="kitchen_normal_use",
        top_k=2,
    )
    assert out["positive_label"] == "kitchen_normal_use"
    assert out["fn_minutes_total"] > 0.0
    assert out["fp_minutes_total"] > 0.0
    assert len(out["fn_top"]) >= 1
    assert len(out["fp_top"]) >= 1


def test_single_resident_arbitration_suppresses_lower_conflict_room():
    ts = pd.date_range("2025-12-06 09:00:00", periods=3, freq="10s")
    split_room_outputs = {
        "Bedroom": {
            "test_df": pd.DataFrame({"timestamp": ts}),
            "y_pred": np.array(["sleep", "sleep", "unoccupied"], dtype=object),
            "occupancy_probs": np.array([0.61, 0.58, 0.10], dtype=float),
            "activity_probs": {"sleep": np.array([0.45, 0.50, 0.02], dtype=float)},
        },
        "LivingRoom": {
            "test_df": pd.DataFrame({"timestamp": ts}),
            "y_pred": np.array(["livingroom_normal_use", "unoccupied", "unoccupied"], dtype=object),
            "occupancy_probs": np.array([0.54, 0.07, 0.05], dtype=float),
            "activity_probs": {"livingroom_normal_use": np.array([0.40, 0.02, 0.01], dtype=float)},
        },
    }
    debug = _apply_single_resident_arbitration(
        split_room_outputs=split_room_outputs,
        arbitration_rooms=["bedroom", "livingroom", "kitchen"],
        min_confidence_margin=0.02,
    )
    assert debug["enabled"] is True
    assert debug["conflicts_total"] == 1
    assert debug["adjustments_total"] == 1
    assert split_room_outputs["Bedroom"]["y_pred"][0] == "sleep"
    assert split_room_outputs["LivingRoom"]["y_pred"][0] == "unoccupied"


def test_single_resident_arbitration_requires_at_least_two_candidate_rooms():
    ts = pd.date_range("2025-12-06 09:00:00", periods=1, freq="10s")
    split_room_outputs = {
        "Bedroom": {
            "test_df": pd.DataFrame({"timestamp": ts}),
            "y_pred": np.array(["sleep"], dtype=object),
            "occupancy_probs": np.array([0.52], dtype=float),
            "activity_probs": {"sleep": np.array([0.20], dtype=float)},
        },
    }
    debug = _apply_single_resident_arbitration(
        split_room_outputs=split_room_outputs,
        arbitration_rooms=["bedroom", "livingroom"],
        min_confidence_margin=0.03,
    )
    assert debug["conflicts_total"] == 0
    assert debug["adjustments_total"] == 0
    assert debug["reason"] == "insufficient_candidate_rooms"
    assert split_room_outputs["Bedroom"]["y_pred"][0] == "sleep"


def test_single_resident_arbitration_prefers_bedroom_sleep_at_night():
    ts = pd.date_range("2025-12-06 23:00:00", periods=24, freq="10s")
    split_room_outputs = {
        "Bedroom": {
            "test_df": pd.DataFrame({"timestamp": ts}),
            "y_pred": np.array(["sleep"] * len(ts), dtype=object),
            "occupancy_probs": np.array([0.51] * len(ts), dtype=float),
            "activity_probs": {"sleep": np.array([0.40] * len(ts), dtype=float)},
        },
        "LivingRoom": {
            "test_df": pd.DataFrame({"timestamp": ts}),
            "y_pred": np.array(["livingroom_normal_use"] * len(ts), dtype=object),
            "occupancy_probs": np.array([0.64] * len(ts), dtype=float),
            "activity_probs": {"livingroom_normal_use": np.array([0.80] * len(ts), dtype=float)},
        },
    }
    debug = _apply_single_resident_arbitration(
        split_room_outputs=split_room_outputs,
        arbitration_rooms=["bedroom", "livingroom"],
    )
    assert int(debug["adjustments_total"]) == len(ts)
    assert int(debug["guard_wins"]["bedroom_night_sleep"]) >= 1
    assert all(label == "sleep" for label in split_room_outputs["Bedroom"]["y_pred"])
    assert all(label == "unoccupied" for label in split_room_outputs["LivingRoom"]["y_pred"])


def test_single_resident_arbitration_prefers_kitchen_high_confidence():
    ts = pd.date_range("2025-12-06 13:00:00", periods=6, freq="10s")
    split_room_outputs = {
        "Kitchen": {
            "test_df": pd.DataFrame({"timestamp": ts}),
            "y_pred": np.array(["kitchen_normal_use"] * len(ts), dtype=object),
            "occupancy_probs": np.array([0.52] * len(ts), dtype=float),
            "activity_probs": {"kitchen_normal_use": np.array([0.90] * len(ts), dtype=float)},
        },
        "LivingRoom": {
            "test_df": pd.DataFrame({"timestamp": ts}),
            "y_pred": np.array(["livingroom_normal_use"] * len(ts), dtype=object),
            "occupancy_probs": np.array([0.60] * len(ts), dtype=float),
            "activity_probs": {"livingroom_normal_use": np.array([0.35] * len(ts), dtype=float)},
        },
    }
    debug = _apply_single_resident_arbitration(
        split_room_outputs=split_room_outputs,
        arbitration_rooms=["kitchen", "livingroom"],
    )
    assert int(debug["adjustments_total"]) == len(ts)
    assert int(debug["guard_wins"]["kitchen_high_confidence"]) >= 1
    assert all(label == "kitchen_normal_use" for label in split_room_outputs["Kitchen"]["y_pred"])
    assert all(label == "unoccupied" for label in split_room_outputs["LivingRoom"]["y_pred"])


def test_livingroom_cross_room_presence_decoder_extends_when_other_rooms_quiet():
    ts = pd.date_range("2025-12-06 10:00:00", periods=8, freq="10s")
    split_room_outputs = {
        "LivingRoom": {
            "test_df": pd.DataFrame({"timestamp": ts, "motion": [1.0] + [0.0] * 7}),
            "y_pred": np.array(
                ["livingroom_normal_use"] + ["unoccupied"] * 7,
                dtype=object,
            ),
            "occupancy_probs": np.array([0.80] + [0.18] * 7, dtype=float),
            "activity_probs": {"livingroom_normal_use": np.array([0.82] + [0.12] * 7, dtype=float)},
        },
        "Bedroom": {
            "test_df": pd.DataFrame({"timestamp": ts}),
            "y_pred": np.array(["unoccupied"] * len(ts), dtype=object),
            "occupancy_probs": np.array([0.08] * len(ts), dtype=float),
            "activity_probs": {"sleep": np.array([0.05] * len(ts), dtype=float)},
        },
        "Kitchen": {
            "test_df": pd.DataFrame({"timestamp": ts}),
            "y_pred": np.array(["unoccupied"] * len(ts), dtype=object),
            "occupancy_probs": np.array([0.10] * len(ts), dtype=float),
            "activity_probs": {"kitchen_normal_use": np.array([0.05] * len(ts), dtype=float)},
        },
    }
    debug = _apply_livingroom_cross_room_presence_decoder(
        split_room_outputs=split_room_outputs,
        supporting_rooms=["bedroom", "kitchen"],
        hold_minutes=0.5,
        max_extension_minutes=2.0,
        entry_occ_threshold=0.70,
        entry_room_prob_threshold=0.40,
        entry_motion_threshold=0.70,
        refresh_occ_threshold=0.30,
        refresh_room_prob_threshold=0.10,
        other_room_exit_occ_threshold=0.70,
        other_room_exit_confirm_windows=2,
        other_room_unoccupied_max_occ_prob=0.25,
        min_support_rooms=2,
    )
    assert debug["enabled"] is True
    assert debug["reason"] == "applied"
    assert int(debug["added_occupied_windows"]) >= 3
    assert int(debug["exit_events"]) == 0
    assert split_room_outputs["LivingRoom"]["y_pred"][1] == "livingroom_normal_use"
    assert split_room_outputs["LivingRoom"]["y_pred"][2] == "livingroom_normal_use"


def test_livingroom_cross_room_presence_decoder_exits_on_other_room_evidence():
    ts = pd.date_range("2025-12-06 18:00:00", periods=10, freq="10s")
    split_room_outputs = {
        "LivingRoom": {
            "test_df": pd.DataFrame({"timestamp": ts, "motion": [0.9] + [0.0] * 9}),
            "y_pred": np.array(["livingroom_normal_use"] + ["unoccupied"] * 9, dtype=object),
            "occupancy_probs": np.array([0.82] + [0.18] * 9, dtype=float),
            "activity_probs": {"livingroom_normal_use": np.array([0.84] + [0.10] * 9, dtype=float)},
        },
        "Bedroom": {
            "test_df": pd.DataFrame({"timestamp": ts}),
            "y_pred": np.array(["unoccupied"] * len(ts), dtype=object),
            "occupancy_probs": np.array([0.08] * len(ts), dtype=float),
            "activity_probs": {"sleep": np.array([0.05] * len(ts), dtype=float)},
        },
        "Kitchen": {
            "test_df": pd.DataFrame({"timestamp": ts}),
            "y_pred": np.array(
                ["unoccupied", "unoccupied", "unoccupied", "kitchen_normal_use", "kitchen_normal_use"]
                + ["kitchen_normal_use"] * 5,
                dtype=object,
            ),
            "occupancy_probs": np.array([0.10, 0.12, 0.15, 0.78, 0.80] + [0.82] * 5, dtype=float),
            "activity_probs": {"kitchen_normal_use": np.array([0.04, 0.05, 0.08, 0.82, 0.84] + [0.85] * 5, dtype=float)},
        },
    }
    debug = _apply_livingroom_cross_room_presence_decoder(
        split_room_outputs=split_room_outputs,
        supporting_rooms=["bedroom", "kitchen"],
        hold_minutes=1.0,
        max_extension_minutes=3.0,
        entry_occ_threshold=0.70,
        entry_room_prob_threshold=0.40,
        entry_motion_threshold=0.70,
        refresh_occ_threshold=0.35,
        refresh_room_prob_threshold=0.10,
        other_room_exit_occ_threshold=0.70,
        other_room_exit_confirm_windows=2,
        other_room_unoccupied_max_occ_prob=0.25,
        min_support_rooms=2,
    )
    assert debug["enabled"] is True
    assert int(debug["added_occupied_windows"]) >= 1
    assert int(debug["exit_events"]) >= 1
    # Early passive windows are filled, but later windows remain unoccupied after kitchen evidence.
    assert split_room_outputs["LivingRoom"]["y_pred"][1] == "livingroom_normal_use"
    assert split_room_outputs["LivingRoom"]["y_pred"][-1] == "unoccupied"


def test_livingroom_cross_room_presence_decoder_night_bedroom_guard_suppresses_overnight_fp():
    ts = pd.date_range("2025-12-07 03:00:00", periods=12, freq="10s")
    split_room_outputs = {
        "LivingRoom": {
            "test_df": pd.DataFrame({"timestamp": ts, "motion": [0.12] * len(ts)}),
            "y_pred": np.array(["livingroom_normal_use"] * len(ts), dtype=object),
            "occupancy_probs": np.array([0.52] * len(ts), dtype=float),
            "activity_probs": {"livingroom_normal_use": np.array([0.35] * len(ts), dtype=float)},
        },
        "Bedroom": {
            "test_df": pd.DataFrame({"timestamp": ts, "motion": [0.05] * len(ts)}),
            "y_pred": np.array(["sleep"] * len(ts), dtype=object),
            "occupancy_probs": np.array([0.90] * len(ts), dtype=float),
            "activity_probs": {"sleep": np.array([0.92] * len(ts), dtype=float)},
        },
        "Kitchen": {
            "test_df": pd.DataFrame({"timestamp": ts}),
            "y_pred": np.array(["unoccupied"] * len(ts), dtype=object),
            "occupancy_probs": np.array([0.06] * len(ts), dtype=float),
            "activity_probs": {"kitchen_normal_use": np.array([0.03] * len(ts), dtype=float)},
        },
    }
    debug = _apply_livingroom_cross_room_presence_decoder(
        split_room_outputs=split_room_outputs,
        supporting_rooms=["bedroom", "kitchen"],
        hold_minutes=0.5,
        max_extension_minutes=1.0,
        entry_occ_threshold=0.70,
        entry_room_prob_threshold=0.40,
        entry_motion_threshold=0.70,
        refresh_occ_threshold=0.35,
        refresh_room_prob_threshold=0.10,
        other_room_exit_occ_threshold=0.70,
        other_room_exit_confirm_windows=2,
        other_room_unoccupied_max_occ_prob=0.25,
        min_support_rooms=2,
        enable_bedroom_sleep_night_guard=True,
        night_bedroom_guard_start_hour=22,
        night_bedroom_guard_end_hour=6,
        night_bedroom_sleep_occ_threshold=0.66,
        night_bedroom_sleep_prob_threshold=0.55,
        night_bedroom_exit_occ_threshold=0.35,
        night_bedroom_exit_motion_threshold=0.75,
        night_entry_occ_threshold=0.66,
        night_entry_motion_threshold=0.75,
        night_entry_confirm_windows=2,
        night_bedroom_suppression_label="unknown",
    )
    assert debug["enabled"] is True
    assert debug["night_bedroom_guard_applied"] is True
    assert int(debug["night_bedroom_guard_suppressed_windows"]) >= 1
    assert int(debug["night_bedroom_guard_unknown_windows"]) >= 1
    assert all(label == "unknown" for label in split_room_outputs["LivingRoom"]["y_pred"])


def test_livingroom_cross_room_presence_decoder_night_bedroom_guard_allows_strong_lr_entry():
    ts = pd.date_range("2025-12-07 23:00:00", periods=10, freq="10s")
    split_room_outputs = {
        "LivingRoom": {
            "test_df": pd.DataFrame({"timestamp": ts, "motion": [0.82, 0.83] + [0.08] * 8}),
            "y_pred": np.array(["unoccupied"] * len(ts), dtype=object),
            "occupancy_probs": np.array([0.72, 0.75] + [0.18] * 8, dtype=float),
            "activity_probs": {"livingroom_normal_use": np.array([0.80, 0.78] + [0.08] * 8, dtype=float)},
        },
        "Bedroom": {
            "test_df": pd.DataFrame({"timestamp": ts, "motion": [0.05] * len(ts)}),
            "y_pred": np.array(["sleep"] * len(ts), dtype=object),
            "occupancy_probs": np.array([0.92] * len(ts), dtype=float),
            "activity_probs": {"sleep": np.array([0.93] * len(ts), dtype=float)},
        },
        "Kitchen": {
            "test_df": pd.DataFrame({"timestamp": ts}),
            "y_pred": np.array(["unoccupied"] * len(ts), dtype=object),
            "occupancy_probs": np.array([0.08] * len(ts), dtype=float),
            "activity_probs": {"kitchen_normal_use": np.array([0.04] * len(ts), dtype=float)},
        },
    }
    debug = _apply_livingroom_cross_room_presence_decoder(
        split_room_outputs=split_room_outputs,
        supporting_rooms=["bedroom", "kitchen"],
        hold_minutes=0.6,
        max_extension_minutes=1.0,
        entry_occ_threshold=0.70,
        entry_room_prob_threshold=0.40,
        entry_motion_threshold=0.70,
        refresh_occ_threshold=0.35,
        refresh_room_prob_threshold=0.10,
        other_room_exit_occ_threshold=0.70,
        other_room_exit_confirm_windows=2,
        other_room_unoccupied_max_occ_prob=0.25,
        min_support_rooms=2,
        enable_bedroom_sleep_night_guard=True,
        night_bedroom_guard_start_hour=22,
        night_bedroom_guard_end_hour=6,
        night_bedroom_sleep_occ_threshold=0.66,
        night_bedroom_sleep_prob_threshold=0.55,
        night_bedroom_exit_occ_threshold=0.35,
        night_bedroom_exit_motion_threshold=0.75,
        night_entry_occ_threshold=0.66,
        night_entry_motion_threshold=0.75,
        night_entry_confirm_windows=2,
        night_bedroom_suppression_label="unknown",
    )
    assert debug["enabled"] is True
    assert debug["night_bedroom_guard_applied"] is True
    assert int(debug["night_bedroom_guard_blocked_entries"]) >= 1
    assert int(debug["added_occupied_windows"]) >= 1
    assert split_room_outputs["LivingRoom"]["y_pred"][0] == "unoccupied"
    assert split_room_outputs["LivingRoom"]["y_pred"][1] == "livingroom_normal_use"


def test_build_room_boundary_sample_weights_emphasizes_boundary_band():
    labels = (
        ["unoccupied"] * 120
        + ["sleep"] * 40
        + ["unoccupied"] * 120
    )
    w = _build_room_boundary_sample_weights(labels, boundary_band_windows=5)
    assert len(w) == len(labels)
    # Weights around care transition should be higher than deep easy-negative windows.
    boundary_slice = w[115:126]
    deep_easy_slice = w[20:80]
    assert float(np.mean(boundary_slice)) > float(np.mean(deep_easy_slice))


def test_build_livingroom_passive_alignment_sample_weights_marks_direct_and_passive_windows():
    labels = ["unoccupied"] * 12 + ["livingroom_normal_use"] * 24 + ["unoccupied"] * 8
    motion = [0.0] * 12 + [0.9] + [0.05] * 22 + [0.0] * 8
    w, payload = _build_livingroom_passive_alignment_sample_weights(
        labels,
        motion_values=motion,
        entry_exit_band_windows=4,
        direct_positive_weight=1.0,
        passive_positive_weight=0.2,
        unoccupied_weight=1.0,
        motion_direct_threshold=0.8,
    )
    assert len(w) == len(labels)
    assert payload["applied"] is True
    assert int(payload["direct_windows"]) > 0
    assert int(payload["passive_windows"]) > 0
    pos_slice = np.asarray(labels, dtype=object) != "unoccupied"
    # Passive interior windows should be lighter than direct windows.
    assert float(np.min(w[pos_slice])) <= 0.2 + 1e-9
    assert float(np.max(w[pos_slice])) >= 1.0 - 1e-9


def test_apply_room_occupancy_temporal_decoder_supports_dynamic_label_space():
    y_pred = np.array(["unoccupied", "unoccupied", "unoccupied", "unoccupied"], dtype=object)
    occ = np.array([0.15, 0.72, 0.75, 0.20], dtype=float)
    activity_probs = {
        "reading": np.array([0.10, 0.80, 0.82, 0.12], dtype=float),
        "tv": np.array([0.05, 0.10, 0.09, 0.04], dtype=float),
    }
    out, debug = _apply_room_occupancy_temporal_decoder(
        room_key="livingroom",
        y_pred=y_pred,
        occupancy_probs=occ,
        activity_probs=activity_probs,
        min_on_windows=1,
        min_off_windows=1,
        gap_fill_windows=0,
        min_run_windows=1,
    )
    assert debug["applied"] is True
    assert any(label == "reading" for label in out)


def test_apply_room_timeline_decoder_v2_outputs_episode_labels():
    ts = pd.date_range("2025-12-08 22:00:00", periods=24, freq="10s")
    y_pred = np.array(["unoccupied"] * 6 + ["sleep"] * 12 + ["unoccupied"] * 6, dtype=object)
    occ = np.array([0.10] * 6 + [0.90] * 12 + [0.12] * 6, dtype=float)
    activity_probs = {
        "sleep": np.array([0.05] * 6 + [0.92] * 12 + [0.04] * 6, dtype=float),
        "bedroom_normal_use": np.array([0.03] * 24, dtype=float),
    }
    decoded, debug = _apply_room_timeline_decoder_v2(
        room_key="bedroom",
        timestamps=pd.Series(ts),
        y_pred=y_pred,
        occupancy_probs=occ,
        activity_probs=activity_probs,
    )
    assert debug["applied"] is True
    assert int(debug["episodes_decoded"]) >= 1
    assert any(label == "sleep" for label in decoded)
