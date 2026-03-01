from unittest.mock import MagicMock, patch

import pandas as pd

from ml.pipeline import UnifiedPipeline


def _make_pipeline_stub() -> UnifiedPipeline:
    pipeline = UnifiedPipeline.__new__(UnifiedPipeline)
    pipeline.registry = MagicMock()
    pipeline.platform = MagicMock()
    pipeline.predictor = MagicMock()
    pipeline.denoising_window = 3
    pipeline.denoising_threshold = 4.0
    pipeline.logger = MagicMock()
    return pipeline


def _sensor_input():
    return {
        "room1": pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=5, freq="10s"),
                "motion": [0.1] * 5,
                "temperature": [24.0] * 5,
            }
        )
    }


def _prediction_output():
    return {
        "room1": pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=2, freq="10s"),
                "predicted_activity": ["inactive", "inactive"],
                "confidence": [0.95, 0.95],
            }
        )
    }


def _runtime_policy_for_room(room: str = "room1") -> dict:
    return {
        "schema_version": "beta6.phase4.runtime_policy.v1",
        "source_run_id": "run_shadow_001",
        "master_enabled": True,
        "room_runtime": {
            room: {
                "enable_phase4_runtime": True,
            }
        },
        "policy_paths": {},
    }


@patch("ml.validation.run_validation", return_value=True)
@patch("utils.data_loader.load_sensor_data")
@patch.dict("os.environ", {"ENABLE_BETA6_AUTHORITY": "true"}, clear=True)
def test_predict_auto_enables_beta6_phase4_runtime_bridge(mock_load_sensor_data, _mock_validate):
    pipeline = _make_pipeline_stub()
    mock_load_sensor_data.return_value = _sensor_input()
    pipeline.registry.load_models_for_elder.return_value = ["room1"]
    pipeline._load_beta6_phase4_runtime_policy = MagicMock(return_value=_runtime_policy_for_room("room1"))
    captured = {}

    def _run_prediction_side_effect(*args, **kwargs):
        import os

        captured["master"] = os.getenv("BETA6_PHASE4_RUNTIME_ENABLED")
        captured["rooms"] = os.getenv("BETA6_PHASE4_RUNTIME_ROOMS")
        captured["unknown_policy"] = os.getenv("BETA6_UNKNOWN_POLICY_PATH")
        captured["hmm_policy"] = os.getenv("BETA6_HMM_DURATION_POLICY_PATH")
        return _prediction_output()

    pipeline.predictor.run_prediction.side_effect = _run_prediction_side_effect
    pipeline.predictor.apply_golden_samples.side_effect = lambda raw, elder_id: raw

    result = pipeline.predict("dummy.parquet", "elder_1")

    assert "room1" in result
    assert captured["master"] == "true"
    assert captured["rooms"] == "room1"
    assert str(captured["unknown_policy"]).endswith("backend/config/beta6_unknown_policy.yaml")
    assert str(captured["hmm_policy"]).endswith("backend/config/beta6_duration_prior_policy.yaml")

    import os

    assert os.getenv("BETA6_PHASE4_RUNTIME_ENABLED") is None
    assert os.getenv("BETA6_PHASE4_RUNTIME_ROOMS") is None
    assert os.getenv("BETA6_UNKNOWN_POLICY_PATH") is None
    assert os.getenv("BETA6_HMM_DURATION_POLICY_PATH") is None


@patch("ml.validation.run_validation", return_value=True)
@patch("utils.data_loader.load_sensor_data")
@patch.dict(
    "os.environ",
    {
        "ENABLE_BETA6_AUTHORITY": "true",
        "ENABLE_BETA6_UNKNOWN_ABSTAIN_RUNTIME": "false",
    },
    clear=True,
)
def test_predict_bridge_respects_explicit_runtime_flag(mock_load_sensor_data, _mock_validate):
    pipeline = _make_pipeline_stub()
    mock_load_sensor_data.return_value = _sensor_input()
    pipeline.registry.load_models_for_elder.return_value = ["room1"]
    pipeline._load_beta6_phase4_runtime_policy = MagicMock(return_value=_runtime_policy_for_room("room1"))
    captured = {}

    def _run_prediction_side_effect(*args, **kwargs):
        import os

        captured["master"] = os.getenv("BETA6_PHASE4_RUNTIME_ENABLED")
        captured["unknown_flag"] = os.getenv("ENABLE_BETA6_UNKNOWN_ABSTAIN_RUNTIME")
        return _prediction_output()

    pipeline.predictor.run_prediction.side_effect = _run_prediction_side_effect
    pipeline.predictor.apply_golden_samples.side_effect = lambda raw, elder_id: raw

    pipeline.predict("dummy.parquet", "elder_1")

    assert captured["unknown_flag"] == "false"
    assert captured["master"] is None


@patch("ml.validation.run_validation", return_value=True)
@patch("utils.data_loader.load_sensor_data")
@patch.dict("os.environ", {"ENABLE_BETA6_AUTHORITY": "true"}, clear=True)
def test_predict_bridge_skips_auto_enable_when_fallback_active(mock_load_sensor_data, _mock_validate):
    pipeline = _make_pipeline_stub()
    mock_load_sensor_data.return_value = _sensor_input()
    pipeline.registry.load_models_for_elder.return_value = ["room1"]
    pipeline._load_beta6_phase4_runtime_policy = MagicMock(return_value=_runtime_policy_for_room("room1"))
    pipeline._beta6_any_room_fallback_active = MagicMock(return_value=True)
    captured = {}

    def _run_prediction_side_effect(*args, **kwargs):
        import os

        captured["master"] = os.getenv("BETA6_PHASE4_RUNTIME_ENABLED")
        return _prediction_output()

    pipeline.predictor.run_prediction.side_effect = _run_prediction_side_effect
    pipeline.predictor.apply_golden_samples.side_effect = lambda raw, elder_id: raw

    pipeline.predict("dummy.parquet", "elder_1")

    assert captured["master"] is None
    pipeline._beta6_any_room_fallback_active.assert_called_once()


@patch("ml.validation.run_validation", return_value=True)
@patch("utils.data_loader.load_sensor_data")
@patch.dict("os.environ", {"ENABLE_BETA6_AUTHORITY": "true"}, clear=True)
def test_predict_bridge_does_not_auto_enable_without_runtime_policy_artifact(
    mock_load_sensor_data, _mock_validate
):
    pipeline = _make_pipeline_stub()
    mock_load_sensor_data.return_value = _sensor_input()
    pipeline.registry.load_models_for_elder.return_value = ["room1"]
    pipeline._load_beta6_phase4_runtime_policy = MagicMock(return_value=None)
    captured = {}

    def _run_prediction_side_effect(*args, **kwargs):
        import os

        captured["master"] = os.getenv("BETA6_PHASE4_RUNTIME_ENABLED")
        captured["rooms"] = os.getenv("BETA6_PHASE4_RUNTIME_ROOMS")
        return _prediction_output()

    pipeline.predictor.run_prediction.side_effect = _run_prediction_side_effect
    pipeline.predictor.apply_golden_samples.side_effect = lambda raw, elder_id: raw

    pipeline.predict("dummy.parquet", "elder_1")

    assert captured["master"] is None
    assert captured["rooms"] is None
