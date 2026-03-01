import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

from ml.evaluation import (
    TimeCheckpointedSplitter,
    evaluate_model,
    evaluate_model_version,
    load_room_training_dataframe,
)


class _DummyModel:
    def __init__(self, y_scores: np.ndarray):
        self._scores = y_scores
        self._cursor = 0

    def predict(self, X, verbose=0):
        n = len(X)
        out = self._scores[self._cursor:self._cursor + n]
        self._cursor += n
        return out


class _VersionEvalPlatformStub:
    def __init__(self):
        self.scalers = {}
        self.sensor_columns = ["motion", "temperature"]

    def preprocess_with_resampling(self, room_df, room_name, is_training=False, apply_denoising=False):
        return room_df.copy()

    def create_sequences(self, sensor_data, seq_length):
        total = int(len(sensor_data))
        if total < int(seq_length):
            return np.empty((0, int(seq_length), sensor_data.shape[1]), dtype=np.float32)
        return np.asarray(
            [sensor_data[i : i + int(seq_length)] for i in range(total - int(seq_length) + 1)],
            dtype=np.float32,
        )


def test_time_checkpointed_splitter_has_strict_temporal_boundaries():
    ts = pd.date_range("2026-01-01 00:00:00", periods=24 * 5, freq="1h")
    splitter = TimeCheckpointedSplitter(min_train_days=2, valid_days=1, step_days=1)
    folds = splitter.split(ts)

    assert len(folds) >= 2
    for fold in folds:
        assert fold.train_end < fold.valid_start
        assert len(set(fold.train_idx).intersection(set(fold.valid_idx))) == 0


def test_evaluate_model_walk_forward_outputs_fold_metrics():
    n = 60
    X = np.zeros((n, 5, 3), dtype=np.float32)
    y = np.array(([0, 1] * (n // 2))[:n], dtype=np.int64)
    ts = np.array(pd.date_range("2026-01-01 00:00:00", periods=n, freq="2h"), dtype="datetime64[ns]")

    # Perfect alternating predictions for deterministic high metric.
    y_scores = np.zeros((n, 2), dtype=np.float32)
    y_scores[np.arange(n), y] = 0.9
    y_scores[np.arange(n), 1 - y] = 0.1
    model = _DummyModel(y_scores=y_scores)

    splitter = TimeCheckpointedSplitter(min_train_days=2, valid_days=1, step_days=1, max_folds=3)
    report = evaluate_model(
        model=model,
        X_seq=X,
        y_seq=y,
        seq_timestamps=ts,
        splitter=splitter,
        labels=[0, 1],
    )

    assert report["summary"]["num_folds"] == 3
    assert report["summary"]["macro_f1_mean"] is not None
    assert report["summary"]["macro_f1_mean"] > 0.95
    assert len(report["folds"]) == 3
    assert "confusion_matrix" in report["folds"][0]
    assert "minority_support" in report["folds"][0]
    assert report["folds"][0]["minority_support"] >= 1
    assert "stability_accuracy" in report["folds"][0]
    assert "transition_macro_f1" in report["folds"][0]
    assert "stability_accuracy_mean" in report["summary"]
    assert "transition_macro_f1_mean" in report["summary"]


def test_evaluate_model_thresholds_can_change_predicted_class():
    n = 48
    X = np.zeros((n, 5, 3), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    ts = np.array(pd.date_range("2026-01-01 00:00:00", periods=n, freq="3h"), dtype="datetime64[ns]")

    # Model prefers class 0 but below threshold for class 0.
    y_scores = np.tile(np.array([0.55, 0.45], dtype=np.float32), (n, 1))
    model = _DummyModel(y_scores=y_scores)

    splitter = TimeCheckpointedSplitter(min_train_days=1, valid_days=1, step_days=1, max_folds=1)
    report = evaluate_model(
        model=model,
        X_seq=X,
        y_seq=y,
        seq_timestamps=ts,
        splitter=splitter,
        class_thresholds={0: 0.8},
        labels=[0, 1],
    )

    assert report["summary"]["num_folds"] == 1
    # With threshold, predictions flip to class 1 -> low accuracy for all-zero truth.
    assert report["summary"]["accuracy_mean"] < 0.2


def test_evaluate_model_macro_metrics_use_full_label_space():
    n = 48
    X = np.zeros((n, 5, 3), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)  # Single class in this fold
    ts = np.array(pd.date_range("2026-01-01 00:00:00", periods=n, freq="1h"), dtype="datetime64[ns]")

    # Perfect predictions for present class only.
    y_scores = np.tile(np.array([0.95, 0.05], dtype=np.float32), (n, 1))
    model = _DummyModel(y_scores=y_scores)

    splitter = TimeCheckpointedSplitter(min_train_days=1, valid_days=1, step_days=1, max_folds=1)
    report = evaluate_model(
        model=model,
        X_seq=X,
        y_seq=y,
        seq_timestamps=ts,
        splitter=splitter,
        labels=[0, 1],  # Explicit full label space
    )

    assert report["summary"]["num_folds"] == 1
    fold = report["folds"][0]
    assert fold["accuracy"] == 1.0
    # Macro over [0,1] should not be 1.0 when class 1 has zero support/predictions.
    assert abs(float(fold["macro_f1"]) - 0.5) < 1e-9
    assert abs(float(fold["macro_recall"]) - 0.5) < 1e-9
    assert abs(float(fold["macro_precision"]) - 0.5) < 1e-9
    assert int(fold["minority_support"]) == 0


def test_evaluate_model_split_metrics_penalize_transition_errors():
    n = 96
    X = np.zeros((n, 5, 3), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    y[30:60] = 1
    y[60:96] = 0
    ts = np.array(pd.date_range("2026-01-01 00:00:00", periods=n, freq="1h"), dtype="datetime64[ns]")

    # Near transition boundaries, force wrong class to reduce transition metric.
    y_pred = y.copy()
    for idx in [28, 29, 30, 31, 58, 59, 60, 61]:
        y_pred[idx] = 1 - y_pred[idx]

    y_scores = np.zeros((n, 2), dtype=np.float32)
    y_scores[np.arange(n), y_pred] = 0.9
    y_scores[np.arange(n), 1 - y_pred] = 0.1
    model = _DummyModel(y_scores=y_scores)

    splitter = TimeCheckpointedSplitter(min_train_days=1, valid_days=1, step_days=1, max_folds=1)
    report = evaluate_model(
        model=model,
        X_seq=X,
        y_seq=y,
        seq_timestamps=ts,
        splitter=splitter,
        labels=[0, 1],
    )

    fold = report["folds"][0]
    assert fold["transition_support"] > 0
    assert fold["transition_macro_f1"] is not None
    assert fold["stability_accuracy"] is not None
    assert 0.0 <= float(fold["transition_macro_f1"]) <= 1.0
    assert 0.0 <= float(fold["stability_accuracy"]) <= 1.0
    assert float(fold["transition_macro_f1"]) < 0.8


def test_evaluate_model_transition_metric_not_capped_by_full_label_space():
    n = 120
    X = np.zeros((n, 5, 3), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    y[40:80] = 1
    X[:, 0, 0] = y.astype(np.float32)
    ts = np.array(pd.date_range("2026-01-01 00:00:00", periods=n, freq="1h"), dtype="datetime64[ns]")

    class _FeaturePerfectModel:
        def predict(self, x_in, verbose=0):
            cls = x_in[:, 0, 0].astype(int)
            probs = np.full((len(x_in), 6), 0.01, dtype=np.float32)
            probs[np.arange(len(x_in)), cls] = 0.95
            probs[np.arange(len(x_in)), 1 - cls] = 0.05
            return probs

    model = _FeaturePerfectModel()

    splitter = TimeCheckpointedSplitter(min_train_days=1, valid_days=1, step_days=1, max_folds=1)
    report = evaluate_model(
        model=model,
        X_seq=X,
        y_seq=y,
        seq_timestamps=ts,
        splitter=splitter,
        labels=[0, 1, 2, 3, 4, 5],
    )

    fold = report["folds"][0]
    assert fold["transition_support"] > 0
    assert fold["transition_macro_f1"] is not None
    assert abs(float(fold["transition_macro_f1"]) - 1.0) < 1e-9


def test_evaluate_model_deep_stability_excludes_single_point_runs():
    n = 96
    X = np.zeros((n, 5, 3), dtype=np.float32)
    y = np.array(([0, 1] * (n // 2))[:n], dtype=np.int64)  # alternating runs of len=1
    ts = np.array(pd.date_range("2026-01-01 00:00:00", periods=n, freq="1h"), dtype="datetime64[ns]")

    y_scores = np.zeros((n, 2), dtype=np.float32)
    y_scores[np.arange(n), y] = 0.9
    y_scores[np.arange(n), 1 - y] = 0.1
    model = _DummyModel(y_scores=y_scores)

    splitter = TimeCheckpointedSplitter(min_train_days=2, valid_days=1, step_days=1, max_folds=1)
    report = evaluate_model(
        model=model,
        X_seq=X,
        y_seq=y,
        seq_timestamps=ts,
        splitter=splitter,
        labels=[0, 1],
    )

    fold = report["folds"][0]
    assert int(fold["stability_support"]) == 0


def test_load_room_training_dataframe_merges_and_filters(tmp_path):
    archive_dir = tmp_path / "archive"
    day_dir = archive_dir / "2026-01-01"
    day_dir.mkdir(parents=True)
    fake_file = day_dir / "elder_a_train.parquet"
    fake_file.write_text("placeholder")

    now = pd.Timestamp.utcnow().tz_localize(None)
    old_ts = now - pd.Timedelta(days=120)
    fresh_ts = now - pd.Timedelta(days=3)

    def _load_sensor_data(_path, resample=True):
        return {
            "Living Room": pd.DataFrame({
                "timestamp": [old_ts, fresh_ts, fresh_ts],  # includes duplicate
                "activity": ["inactive", "watch_tv", "watch_tv"],
                "motion": [0.1, 0.2, 0.2],
            })
        }

    def _normalize_room(name: str) -> str:
        return str(name).strip().lower().replace(" ", "")

    df, err = load_room_training_dataframe(
        elder_id="elder_a",
        room_name="livingroom",
        archive_dir=Path(archive_dir),
        load_sensor_data_fn=_load_sensor_data,
        normalize_room_name_fn=_normalize_room,
        lookback_days=90,
    )

    assert err is None
    assert df is not None
    # old row should be filtered out and duplicate timestamp de-duped.
    assert len(df) == 1
    assert df.iloc[0]["activity"] == "watch_tv"


def test_load_room_training_dataframe_lookback_anchors_to_dataset_recency(tmp_path):
    archive_dir = tmp_path / "archive"
    day_dir = archive_dir / "2026-01-01"
    day_dir.mkdir(parents=True)
    fake_file = day_dir / "elder_b_train.parquet"
    fake_file.write_text("placeholder")

    # Historical data can be far from wall-clock now after clean reset/backfill.
    # Loader should anchor lookback to latest timestamp in the dataset.
    historical_old = pd.Timestamp("2025-12-01 09:00:00")
    historical_new = pd.Timestamp("2025-12-20 09:00:00")

    def _load_sensor_data(_path, resample=True):
        return {
            "Bedroom": pd.DataFrame(
                {
                    "timestamp": [historical_old, historical_new],
                    "activity": ["inactive", "reading"],
                    "motion": [0.1, 0.6],
                }
            )
        }

    def _normalize_room(name: str) -> str:
        return str(name).strip().lower().replace(" ", "")

    df, err = load_room_training_dataframe(
        elder_id="elder_b",
        room_name="bedroom",
        archive_dir=Path(archive_dir),
        load_sensor_data_fn=_load_sensor_data,
        normalize_room_name_fn=_normalize_room,
        lookback_days=10,
    )

    assert err is None
    assert df is not None
    assert len(df) == 1
    assert str(df.iloc[0]["activity"]) == "reading"


def test_load_room_training_dataframe_includes_pending_files(tmp_path):
    archive_dir = tmp_path / "archive_missing"
    pending_file = tmp_path / "elder_c_train_2026-01-01.parquet"
    pending_file.write_text("placeholder")

    def _load_sensor_data(path, resample=True):
        if Path(path).name != pending_file.name:
            return {}
        return {
            "Kitchen": pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2026-01-01 00:00:00")],
                    "activity": ["cook"],
                    "motion": [0.4],
                }
            )
        }

    def _normalize_room(name: str) -> str:
        return str(name).strip().lower().replace(" ", "")

    df, err = load_room_training_dataframe(
        elder_id="elder_c",
        room_name="kitchen",
        archive_dir=Path(archive_dir),
        load_sensor_data_fn=_load_sensor_data,
        normalize_room_name_fn=_normalize_room,
        lookback_days=90,
        include_files=[pending_file],
    )

    assert err is None
    assert df is not None
    assert len(df) == 1
    assert str(df.iloc[0]["activity"]) == "cook"


def test_evaluate_model_version_allows_exact_seq_length_rows(monkeypatch):
    seq_length = 5
    captured = {}
    platform = _VersionEvalPlatformStub()

    monkeypatch.setattr("utils.segment_utils.normalize_activity_name", lambda x: x)
    monkeypatch.setattr("utils.segment_utils.validate_activity_for_room", lambda x, _room: x)
    monkeypatch.setattr(
        "ml.evaluation.evaluate_model",
        lambda **kwargs: captured.update(
            {
                "x_len": int(len(kwargs["X_seq"])),
                "y_len": int(len(kwargs["y_seq"])),
                "ts_len": int(len(kwargs["seq_timestamps"])),
            }
        )
        or {"folds": [], "summary": {"num_folds": 0}},
    )

    room_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=seq_length, freq="1h"),
            "activity": ["sleep"] * seq_length,
            "motion": np.linspace(0.1, 0.5, seq_length),
            "temperature": np.linspace(22.0, 23.0, seq_length),
        }
    )

    report, err = evaluate_model_version(
        model=MagicMock(),
        platform=platform,
        room_name="Bedroom",
        room_df=room_df,
        seq_length=seq_length,
        scaler=MagicMock(),
        label_encoder=MagicMock(classes_=["sleep"]),
        splitter=TimeCheckpointedSplitter(min_train_days=1, valid_days=1, step_days=1, max_folds=1),
    )

    assert err is None
    assert report is not None
    assert captured["x_len"] == 1
    assert captured["y_len"] == 1
    assert captured["ts_len"] == 1


def test_evaluate_model_version_allows_exact_seq_length_after_label_filter(monkeypatch):
    seq_length = 5
    captured = {}
    platform = _VersionEvalPlatformStub()

    monkeypatch.setattr("utils.segment_utils.normalize_activity_name", lambda x: x)
    monkeypatch.setattr("utils.segment_utils.validate_activity_for_room", lambda x, _room: x)
    monkeypatch.setattr(
        "ml.evaluation.evaluate_model",
        lambda **kwargs: captured.update({"x_len": int(len(kwargs["X_seq"]))})
        or {"folds": [], "summary": {"num_folds": 0}},
    )

    room_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=seq_length + 1, freq="1h"),
            "activity": ["sleep"] * seq_length + ["unknown"],
            "motion": np.linspace(0.1, 0.6, seq_length + 1),
            "temperature": np.linspace(22.0, 23.2, seq_length + 1),
        }
    )

    report, err = evaluate_model_version(
        model=MagicMock(),
        platform=platform,
        room_name="Bedroom",
        room_df=room_df,
        seq_length=seq_length,
        scaler=MagicMock(),
        label_encoder=MagicMock(classes_=["sleep"]),
        splitter=TimeCheckpointedSplitter(min_train_days=1, valid_days=1, step_days=1, max_folds=1),
    )

    assert err is None
    assert report is not None
    assert captured["x_len"] == 1


def test_evaluate_model_version_handles_numpy_label_classes_without_truthiness_error(monkeypatch):
    seq_length = 5
    platform = _VersionEvalPlatformStub()
    captured = {}

    monkeypatch.setattr("utils.segment_utils.normalize_activity_name", lambda x: x)
    monkeypatch.setattr("utils.segment_utils.validate_activity_for_room", lambda x, _room: x)
    monkeypatch.setattr(
        "ml.evaluation.evaluate_model",
        lambda **kwargs: captured.update(
            {
                "x_len": int(len(kwargs["X_seq"])),
                "labels_len": int(len(kwargs["labels"])) if kwargs.get("labels") is not None else -1,
            }
        )
        or {"folds": [], "summary": {"num_folds": 0}},
    )

    room_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=seq_length + 2, freq="1h"),
            "activity": ["sleep"] * (seq_length + 2),
            "motion": np.linspace(0.1, 0.7, seq_length + 2),
            "temperature": np.linspace(22.0, 23.4, seq_length + 2),
        }
    )
    encoder = MagicMock()
    encoder.classes_ = np.array(["sleep", "inactive"], dtype=object)

    report, err = evaluate_model_version(
        model=MagicMock(),
        platform=platform,
        room_name="Bedroom",
        room_df=room_df,
        seq_length=seq_length,
        scaler=MagicMock(),
        label_encoder=encoder,
        splitter=TimeCheckpointedSplitter(min_train_days=1, valid_days=1, step_days=1, max_folds=1),
    )

    assert err is None
    assert report is not None
    assert captured["x_len"] >= 1
    assert captured["labels_len"] == 2


def test_evaluate_model_version_excludes_synthetic_gap_days_from_folds(monkeypatch):
    seq_length = 2
    captured = {}

    class _GapDayPlatform:
        def __init__(self):
            self.scalers = {}
            self.sensor_columns = ["motion", "temperature"]

        def preprocess_with_resampling(self, room_df, room_name, is_training=False, apply_denoising=False):
            # Includes synthetic middle day (2026-01-02) that does not exist in raw room_df.
            return pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        [
                            "2026-01-01 00:00:00",
                            "2026-01-01 01:00:00",
                            "2026-01-02 00:00:00",
                            "2026-01-02 01:00:00",
                            "2026-01-03 00:00:00",
                            "2026-01-03 01:00:00",
                        ]
                    ),
                    "activity": ["sleep"] * 6,
                    "motion": np.linspace(0.1, 0.6, 6),
                    "temperature": np.linspace(22.0, 23.0, 6),
                }
            )

        def create_sequences(self, sensor_data, seq_length):
            total = int(len(sensor_data))
            if total < int(seq_length):
                return np.empty((0, int(seq_length), sensor_data.shape[1]), dtype=np.float32)
            return np.asarray(
                [sensor_data[i : i + int(seq_length)] for i in range(total - int(seq_length) + 1)],
                dtype=np.float32,
            )

    platform = _GapDayPlatform()

    monkeypatch.setattr("utils.segment_utils.normalize_activity_name", lambda x: x)
    monkeypatch.setattr("utils.segment_utils.validate_activity_for_room", lambda x, _room: x)
    monkeypatch.setattr(
        "ml.evaluation.evaluate_model",
        lambda **kwargs: captured.update(
            {
                "seq_days": sorted(
                    pd.to_datetime(pd.Series(kwargs["seq_timestamps"])).dt.floor("D").astype(str).unique().tolist()
                )
            }
        )
        or {"folds": [], "summary": {"num_folds": 0}},
    )

    # Raw observed data contains only Jan 1 and Jan 3 (gap day Jan 2 should be excluded).
    room_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-03 00:00:00",
                    "2026-01-03 01:00:00",
                ]
            ),
            "activity": ["sleep"] * 4,
            "motion": [0.1, 0.2, 0.3, 0.4],
            "temperature": [22.0, 22.2, 22.4, 22.6],
        }
    )

    report, err = evaluate_model_version(
        model=MagicMock(),
        platform=platform,
        room_name="Bedroom",
        room_df=room_df,
        seq_length=seq_length,
        scaler=MagicMock(),
        label_encoder=MagicMock(classes_=["sleep"]),
        splitter=TimeCheckpointedSplitter(min_train_days=1, valid_days=1, step_days=1, max_folds=1),
    )

    assert err is None
    assert report is not None
    assert captured["seq_days"] == ["2026-01-01", "2026-01-03"]
