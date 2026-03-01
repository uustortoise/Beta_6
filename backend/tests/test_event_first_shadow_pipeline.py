from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from ml.pipeline import UnifiedPipeline


def _build_pipeline_stub(shadow_enabled: bool) -> UnifiedPipeline:
    pipeline = UnifiedPipeline.__new__(UnifiedPipeline)
    pipeline.enable_denoising = False
    pipeline.denoising_method = "hampel"
    pipeline.denoising_window = 3
    pipeline.denoising_threshold = 3
    pipeline.platform = MagicMock()
    pipeline.platform.preprocess_with_resampling.side_effect = (
        lambda df, room_name, is_training, apply_denoising, **kwargs: df.copy()
    )
    pipeline.trainer = MagicMock()
    pipeline.registry = MagicMock()
    pipeline.registry.get_models_dir.return_value = Path(".")
    pipeline.predictor = MagicMock()
    pipeline.predictor.apply_golden_samples.side_effect = lambda results, elder_id: results
    pipeline._evaluate_gates_unified = MagicMock(
        return_value={
            "pre_training_pass": True,
            "gate_pass": True,
            "gate_stack": [],
            "gate_reasons": [],
        }
    )
    pipeline._is_event_first_shadow_enabled = MagicMock(return_value=shadow_enabled)
    pipeline._resolve_resample_gap_policy = MagicMock(return_value=60.0)
    return pipeline


def test_train_from_files_passes_event_first_shadow_flag(monkeypatch):
    pipeline = _build_pipeline_stub(shadow_enabled=True)
    pipeline.trainer.train_room.return_value = {"room": "Bedroom", "accuracy": 0.9}

    mock_room_cfg = MagicMock()
    mock_room_cfg.calculate_seq_length.return_value = 5
    monkeypatch.setattr("ml.pipeline.get_room_config", lambda: mock_room_cfg)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=120, freq="10s"),
            "motion": [0.1] * 120,
            "activity": ["sleep"] * 120,
        }
    )
    monkeypatch.setattr(
        "utils.data_loader.load_sensor_data",
        lambda path, resample=True, **kwargs: {"Bedroom": df.copy()},
    )

    pipeline.train_from_files(file_paths=["/tmp/file1.xlsx"], elder_id="elder_1")

    assert pipeline.trainer.train_room.call_args.kwargs["event_first_shadow"] is True


def test_train_and_predict_passes_event_first_shadow_flag(monkeypatch):
    pipeline = _build_pipeline_stub(shadow_enabled=True)
    pipeline.trainer.train_room.return_value = {"room": "Bedroom", "accuracy": 0.9}

    mock_room_cfg = MagicMock()
    mock_room_cfg.calculate_seq_length.return_value = 5
    monkeypatch.setattr("ml.pipeline.get_room_config", lambda: mock_room_cfg)

    monkeypatch.setattr(
        "ml.utils.fetch_all_golden_samples",
        lambda elder_id: pd.DataFrame(columns=["timestamp", "activity", "room"]),
    )

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=120, freq="10s"),
            "motion": [0.1] * 120,
            "activity": ["sleep"] * 120,
        }
    )
    monkeypatch.setattr(
        "utils.data_loader.load_sensor_data",
        lambda path, resample=True, **kwargs: {"Bedroom": df.copy()},
    )

    pipeline.train_and_predict(file_path="/tmp/file1.xlsx", elder_id="elder_1")

    assert pipeline.trainer.train_room.call_args.kwargs["event_first_shadow"] is True


def test_train_from_files_shadow_false_remains_legacy(monkeypatch):
    pipeline = _build_pipeline_stub(shadow_enabled=False)
    pipeline.trainer.train_room.return_value = {"room": "Bedroom", "accuracy": 0.9}

    mock_room_cfg = MagicMock()
    mock_room_cfg.calculate_seq_length.return_value = 5
    monkeypatch.setattr("ml.pipeline.get_room_config", lambda: mock_room_cfg)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=120, freq="10s"),
            "motion": [0.1] * 120,
            "activity": ["sleep"] * 120,
        }
    )
    monkeypatch.setattr(
        "utils.data_loader.load_sensor_data",
        lambda path, resample=True, **kwargs: {"Bedroom": df.copy()},
    )

    pipeline.train_from_files(file_paths=["/tmp/file1.xlsx"], elder_id="elder_1")

    assert pipeline.trainer.train_room.call_args.kwargs["event_first_shadow"] is False

