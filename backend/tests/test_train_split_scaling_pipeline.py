import numpy as np
import pandas as pd
import pytest

from ml.train_split_scaling_pipeline import prepare_training_data_with_train_split_scaling


class _DummyLabelEncoder:
    def __init__(self) -> None:
        self.classes_ = np.array([], dtype=object)
        self._mapping = {}

    def fit(self, labels):
        normalized = [str(v).strip().lower() for v in labels]
        self.classes_ = np.array(sorted(set(normalized)), dtype=object)
        self._mapping = {label: idx for idx, label in enumerate(self.classes_)}
        return self

    def transform(self, labels):
        normalized = [str(v).strip().lower() for v in labels]
        return np.array([self._mapping[label] for label in normalized], dtype=np.int32)


class _DummyPlatform:
    def __init__(self) -> None:
        self.label_encoders = {}

    def preprocess_without_scaling(self, df, **_kwargs):
        return df.copy()

    def apply_scaling(self, df, room_name, is_training, scaler_fit_range=None):
        _ = scaler_fit_range  # unused in test double
        out = df.copy()
        if is_training:
            encoder = _DummyLabelEncoder().fit(out["activity"].tolist())
            self.label_encoders[room_name] = encoder
            out["activity_encoded"] = encoder.transform(out["activity"].tolist())
        else:
            out = out.drop(columns=["activity_encoded"], errors="ignore")
        return out


def _build_room_df(total_rows: int, labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=total_rows, freq="10s"),
            "s1": np.random.randn(total_rows),
            "s2": np.random.randn(total_rows),
            "s3": np.random.randn(total_rows),
            "activity": labels,
        }
    )


def test_prepare_train_split_scaling_populates_activity_encoded_for_non_train_splits():
    platform = _DummyPlatform()
    labels = (["unoccupied", "bathroom_normal_use"] * 40)
    raw_df = _build_room_df(80, labels[:80])

    result = prepare_training_data_with_train_split_scaling(
        platform=platform,
        room_name="bathroom",
        raw_df=raw_df,
        validation_split=0.25,
        calibration_fraction=0.5,
        min_calibration_samples=5,
    )

    assert "activity_encoded" in result["val_scaled"].columns
    assert result["val_scaled"]["activity_encoded"].isna().sum() == 0
    assert result["val_scaled"]["activity_encoded"].dtype == np.int32
    assert result["calib_scaled"] is not None
    assert "activity_encoded" in result["calib_scaled"].columns
    assert result["calib_scaled"]["activity_encoded"].isna().sum() == 0


def test_prepare_train_split_scaling_fails_closed_on_unknown_non_train_label():
    platform = _DummyPlatform()
    labels = (["unoccupied", "bathroom_normal_use"] * 15) + ["shower"] * 10
    raw_df = _build_room_df(40, labels[:40])

    with pytest.raises(ValueError, match="absent from train encoder"):
        prepare_training_data_with_train_split_scaling(
            platform=platform,
            room_name="bathroom",
            raw_df=raw_df,
            validation_split=0.25,
            calibration_fraction=0.0,
            min_calibration_samples=5,
        )


def test_prepare_train_split_scaling_fails_closed_on_mismatched_existing_activity_encoded():
    class _MismatchPlatform(_DummyPlatform):
        def apply_scaling(self, df, room_name, is_training, scaler_fit_range=None):
            out = super().apply_scaling(df, room_name, is_training, scaler_fit_range=scaler_fit_range)
            if not is_training and not out.empty:
                # Inject wrong-but-numeric encodings to validate mismatch guard.
                out["activity_encoded"] = np.zeros(len(out), dtype=np.int32)
            return out

    platform = _MismatchPlatform()
    labels = (["unoccupied", "bathroom_normal_use"] * 40)
    raw_df = _build_room_df(80, labels[:80])

    with pytest.raises(ValueError, match="mismatches canonical encoding"):
        prepare_training_data_with_train_split_scaling(
            platform=platform,
            room_name="bathroom",
            raw_df=raw_df,
            validation_split=0.25,
            calibration_fraction=0.5,
            min_calibration_samples=5,
        )
