from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from ml.pipeline import UnifiedPipeline


def test_train_from_files_uses_explicit_source_precedence(monkeypatch):
    pipeline = UnifiedPipeline.__new__(UnifiedPipeline)
    pipeline.enable_denoising = False
    pipeline.registry = MagicMock()
    pipeline.registry.get_models_dir.return_value = Path(".")
    pipeline.trainer = MagicMock()
    pipeline.platform = MagicMock()
    pipeline.platform.preprocess_with_resampling.side_effect = (
        lambda df, room_name, is_training, apply_denoising, **kwargs: df.copy()
    )

    captured = {}

    def _capture_train_room(room_name, processed_df, seq_length, elder_id, **kwargs):
        captured["df"] = processed_df.copy()
        return {"room": room_name, "accuracy": 0.9}

    pipeline.trainer.train_room.side_effect = _capture_train_room
    
    # Mock gate evaluation to pass (test is about duplicate precedence, not gates)
    pipeline._evaluate_gates_unified = MagicMock(return_value={
        'gate_pass': True,
        'gate_stack': [],
        'gate_reasons': [],
    })
    pipeline._evaluate_post_training_gates = MagicMock(return_value=(True, []))

    mock_room_cfg = MagicMock()
    mock_room_cfg.calculate_seq_length.return_value = 60
    monkeypatch.setattr("ml.pipeline.get_room_config", lambda: mock_room_cfg)

    t0 = pd.Timestamp("2026-01-01 00:00:00")
    file1_df = pd.DataFrame(
        {
            "timestamp": [t0, t0 + pd.Timedelta(seconds=10)],
            "motion": [1.0, 1.1],
            "activity": ["from_file_1", "from_file_1"],
        }
    )
    file2_df = pd.DataFrame(
        {
            "timestamp": [t0, t0 + pd.Timedelta(seconds=20)],
            "motion": [9.0, 9.1],
            "activity": ["from_file_2", "from_file_2"],
        }
    )

    def _load_sensor_data(path, resample=True, **kwargs):
        if str(path).endswith("file1.xlsx"):
            return {"Bedroom": file1_df.copy()}
        if str(path).endswith("file2.xlsx"):
            return {"Bedroom": file2_df.copy()}
        return {}

    monkeypatch.setattr("utils.data_loader.load_sensor_data", _load_sensor_data)

    pipeline.train_from_files(
        file_paths=["/tmp/file1.xlsx", "/tmp/file2.xlsx"],
        elder_id="elder_1",
    )

    out = captured["df"].sort_values("timestamp").reset_index(drop=True)
    assert len(out) == 3
    # Duplicate timestamp should keep file1 value because file1 appears first in file_paths.
    assert out.loc[out["timestamp"] == t0, "activity"].iloc[0] == "from_file_1"


def test_train_from_files_preserves_trainer_gate_outcome(monkeypatch):
    pipeline = UnifiedPipeline.__new__(UnifiedPipeline)
    pipeline.enable_denoising = False
    pipeline.registry = MagicMock()
    pipeline.registry.get_models_dir.return_value = Path(".")
    pipeline.trainer = MagicMock()
    pipeline.platform = MagicMock()
    pipeline.platform.preprocess_with_resampling.side_effect = (
        lambda df, room_name, is_training, apply_denoising, **kwargs: df.copy()
    )

    pipeline.trainer.train_room.return_value = {
        "room": "Bedroom",
        "accuracy": 0.7,
        "gate_pass": False,
        "gate_reasons": ["statistical_validity_failed:Insufficient calibration support"],
        "gate_stack": [
            {"gate_name": "CoverageContractGate", "passed": True},
            {"gate_name": "StatisticalValidityGate", "passed": False},
        ],
    }

    pipeline._evaluate_gates_unified = MagicMock(return_value={
        "pre_training_pass": True,
        "gate_pass": True,
        "gate_stack": [{"gate_name": "CoverageContractGate", "passed": True}],
        "gate_reasons": [],
    })

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

    _, metrics = pipeline.train_from_files(
        file_paths=["/tmp/file1.xlsx"],
        elder_id="elder_1",
    )

    assert len(metrics) == 1
    assert metrics[0]["gate_pass"] is False
    assert metrics[0]["gate_reasons"] == ["statistical_validity_failed:Insufficient calibration support"]
