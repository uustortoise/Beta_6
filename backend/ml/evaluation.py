import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

logger = logging.getLogger(__name__)

def _fold_support_stats(y_true: np.ndarray, label_space: list[int]) -> tuple[dict[int, int], int]:
    """
    Return per-label support and minimum support across full label space.

    Important: this includes zero-support labels so minority-starved folds
    are represented as minority_support=0.
    """
    support_by_label: dict[int, int] = {}
    for label in label_space:
        support_by_label[int(label)] = int(np.sum(y_true == int(label)))
    minority_support = int(min(support_by_label.values())) if support_by_label else 0
    return support_by_label, minority_support


def _compute_stability_transition_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_space: list[int],
    transition_window_steps: int = 12,
    stability_edge_trim_steps: int = 12,
) -> dict[str, Any]:
    """
    Compute split metrics:
    - stability_accuracy: accuracy on interior of stable runs
    - transition_macro_f1: macro-F1 around label transitions
    """
    n = int(len(y_true))
    if n == 0 or len(y_pred) != n:
        return {
            "stability_accuracy": None,
            "transition_macro_f1": None,
            "stability_support": 0,
            "transition_support": 0,
            "transition_events": 0,
        }

    transition_window_steps = max(0, int(transition_window_steps))
    stability_edge_trim_steps = max(0, int(stability_edge_trim_steps))
    change_idx = np.where(y_true[1:] != y_true[:-1])[0] + 1

    transition_mask = np.zeros(n, dtype=bool)
    for idx in change_idx:
        start = max(0, int(idx) - transition_window_steps)
        end = min(n, int(idx) + transition_window_steps + 1)
        transition_mask[start:end] = True

    deep_stability_mask = np.zeros(n, dtype=bool)
    run_start = 0
    while run_start < n:
        run_end = run_start + 1
        while run_end < n and y_true[run_end] == y_true[run_start]:
            run_end += 1
        run_len = int(run_end - run_start)
        # Single-point runs have no interior and should not count as "deep stability".
        if run_len <= 1:
            run_start = run_end
            continue
        trim = min(stability_edge_trim_steps, run_len // 2)
        inner_start = run_start + trim
        inner_end = run_end - trim
        if inner_end > inner_start:
            deep_stability_mask[inner_start:inner_end] = True
        run_start = run_end

    stability_mask = deep_stability_mask if bool(np.any(deep_stability_mask)) else ~transition_mask
    stability_support = int(np.sum(stability_mask))
    transition_support = int(np.sum(transition_mask))

    stability_accuracy = None
    if stability_support > 0:
        stability_accuracy = float(accuracy_score(y_true[stability_mask], y_pred[stability_mask]))

    transition_macro_f1 = None
    if transition_support > 0:
        # Use labels observed in transition windows to avoid capping perfect scores
        # by unrelated encoder classes with zero support in this fold.
        eval_labels = sorted(
            np.unique(
                np.concatenate([y_true[transition_mask], y_pred[transition_mask]])
            ).tolist()
        )
        if not eval_labels and label_space:
            eval_labels = list(label_space)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true[transition_mask],
            y_pred[transition_mask],
            labels=eval_labels,
            average="macro",
            zero_division=0,
        )
        transition_macro_f1 = float(f1)

    return {
        "stability_accuracy": stability_accuracy,
        "transition_macro_f1": transition_macro_f1,
        "stability_support": stability_support,
        "transition_support": transition_support,
        "transition_events": int(len(change_idx)),
    }


def _encoder_classes(label_encoder: Any) -> list:
    """Return label encoder classes as a plain list (safe for numpy arrays)."""
    classes = getattr(label_encoder, "classes_", None)
    if classes is None:
        return []
    try:
        return list(classes)
    except Exception:
        return []


def _extract_observed_day_set(df: pd.DataFrame) -> set[pd.Timestamp]:
    """Return normalized day set observed in the original (non-resampled) data."""
    if df is None or df.empty or "timestamp" not in df.columns:
        return set()
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if ts.empty:
        return set()
    return set(ts.dt.floor("D").tolist())


def _filter_to_observed_days(processed_df: pd.DataFrame, observed_days: set[pd.Timestamp]) -> pd.DataFrame:
    """
    Keep only rows on days that actually existed in raw observed data.

    This prevents walk-forward from creating folds over synthetic gap days
    introduced by dense resampling/forward-fill.
    """
    if (
        processed_df is None
        or processed_df.empty
        or "timestamp" not in processed_df.columns
        or not observed_days
    ):
        return processed_df
    ts = pd.to_datetime(processed_df["timestamp"], errors="coerce")
    keep_mask = ts.dt.floor("D").isin(list(observed_days))
    filtered = processed_df.loc[keep_mask].copy().reset_index(drop=True)
    dropped = int(len(processed_df) - len(filtered))
    if dropped > 0:
        logger.info(f"Filtered {dropped} synthetic gap-day rows from walk-forward evaluation input.")
    return filtered


def load_room_training_dataframe(
    elder_id: str,
    room_name: str,
    archive_dir: Path,
    load_sensor_data_fn: Any,
    normalize_room_name_fn: Any,
    lookback_days: int = 90,
    include_files: Optional[Iterable[Path]] = None,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load and merge archived training data for one elder/room.
    """
    room_key = normalize_room_name_fn(room_name)
    room_dfs: List[pd.DataFrame] = []
    allowed_suffixes = {".xlsx", ".xls", ".parquet"}
    include_candidates = [Path(p) for p in (include_files or []) if p is not None]

    if (not archive_dir.exists()) and (not include_candidates):
        return None, "Archive directory not found."

    try:
        candidate_files: List[Path] = []
        if archive_dir.exists():
            for date_dir in archive_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                candidate_files.extend(date_dir.rglob("*train*.xlsx"))
                candidate_files.extend(date_dir.rglob("*train*.parquet"))

        # Include current incoming training files (e.g., raw folder) so
        # same-run walk-forward evaluation does not miss not-yet-archived files.
        for candidate in include_candidates:
            if not candidate.exists() or not candidate.is_file():
                continue
            name = candidate.name.lower()
            if "_train" not in name:
                continue
            if candidate.suffix.lower() not in allowed_suffixes:
                continue
            candidate_files.append(candidate)

        unique_candidates = []
        seen = set()
        for candidate in candidate_files:
            try:
                key = str(candidate.resolve())
            except Exception:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)

        for f in unique_candidates:
            if elder_id.lower() not in f.name.lower():
                continue
            try:
                data_dict = load_sensor_data_fn(f, resample=True)
                for loaded_room, df in data_dict.items():
                    if normalize_room_name_fn(loaded_room) != room_key:
                        continue
                    if "activity" not in df.columns:
                        continue
                    local_df = df.copy()
                    if "timestamp" in local_df.columns:
                        local_df["timestamp"] = pd.to_datetime(local_df["timestamp"], errors="coerce")
                        local_df = local_df.dropna(subset=["timestamp"])
                    if not local_df.empty:
                        room_dfs.append(local_df)
            except Exception:
                continue

        if not room_dfs:
            return None, f"No archived training data found for {elder_id}/{room_name}."

        combined = pd.concat(room_dfs, ignore_index=True)
        if "timestamp" in combined.columns:
            combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
            combined = combined.dropna(subset=["timestamp"])
            # Anchor lookback to dataset recency, not wall-clock time.
            # This keeps walk-forward evaluation stable for historical replay/backfill runs.
            latest_ts = combined["timestamp"].max()
            if pd.notna(latest_ts):
                cutoff_ts = latest_ts - pd.Timedelta(days=int(lookback_days))
                combined = combined[combined["timestamp"] >= cutoff_ts]
            combined = combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

        if combined.empty:
            return None, f"No valid timestamped training data found for {elder_id}/{room_name}."

        return combined, None
    except Exception as e:
        logger.error(f"Failed to load room training dataframe for {elder_id}/{room_name}: {e}")
        return None, str(e)


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding-window walk-forward fold."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    train_idx: np.ndarray
    valid_idx: np.ndarray


class TimeCheckpointedSplitter:
    """
    Build walk-forward folds on calendar day boundaries.

    Fold pattern:
    - Train: from first available day through day K
    - Valid: next `valid_days` days
    - Step: advance by `step_days`
    """

    def __init__(
        self,
        min_train_days: int = 7,
        valid_days: int = 1,
        step_days: int = 1,
        max_folds: Optional[int] = None,
    ):
        if min_train_days < 1:
            raise ValueError("min_train_days must be >= 1")
        if valid_days < 1:
            raise ValueError("valid_days must be >= 1")
        if step_days < 1:
            raise ValueError("step_days must be >= 1")
        self.min_train_days = int(min_train_days)
        self.valid_days = int(valid_days)
        self.step_days = int(step_days)
        self.max_folds = max_folds

    def split(self, timestamps: Iterable[Any]) -> List[WalkForwardFold]:
        ts = pd.to_datetime(pd.Series(list(timestamps)), errors="coerce")
        valid_mask = ~ts.isna()
        if not bool(valid_mask.any()):
            return []

        ts_valid = ts[valid_mask].sort_values().reset_index()
        original_idx = ts_valid["index"].to_numpy(dtype=np.int64)
        ts_values = ts_valid[0]

        day_series = ts_values.dt.floor("D")
        unique_days = day_series.drop_duplicates().to_list()
        if len(unique_days) < (self.min_train_days + self.valid_days):
            return []

        folds: List[WalkForwardFold] = []
        fold_id = 1
        for train_end_pos in range(self.min_train_days - 1, len(unique_days) - self.valid_days, self.step_days):
            train_start_day = unique_days[0]
            train_end_day = unique_days[train_end_pos]
            valid_start_day = unique_days[train_end_pos + 1]
            valid_end_pos = min(train_end_pos + self.valid_days, len(unique_days) - 1)
            valid_end_day = unique_days[valid_end_pos]

            train_mask = (day_series >= train_start_day) & (day_series <= train_end_day)
            valid_mask = (day_series >= valid_start_day) & (day_series <= valid_end_day)
            train_idx = original_idx[train_mask.to_numpy()]
            valid_idx = original_idx[valid_mask.to_numpy()]
            if len(train_idx) == 0 or len(valid_idx) == 0:
                continue

            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_start=pd.Timestamp(train_start_day),
                    train_end=pd.Timestamp(train_end_day),
                    valid_start=pd.Timestamp(valid_start_day),
                    valid_end=pd.Timestamp(valid_end_day),
                    train_idx=np.asarray(train_idx, dtype=np.int64),
                    valid_idx=np.asarray(valid_idx, dtype=np.int64),
                )
            )
            fold_id += 1
            if self.max_folds is not None and len(folds) >= int(self.max_folds):
                break

        return folds


def _predict_with_optional_thresholds(
    model: Any,
    X: np.ndarray,
    class_thresholds: Optional[Dict[int | str, float]] = None,
) -> np.ndarray:
    """Predict classes, applying optional per-class thresholds."""
    y_scores = model.predict(X, verbose=0)
    if y_scores is None or len(y_scores) == 0:
        return np.array([], dtype=np.int64)

    pred = np.argmax(y_scores, axis=1).astype(np.int64)
    if not class_thresholds:
        return pred

    thresholds: Dict[int, float] = {}
    for k, v in class_thresholds.items():
        try:
            thresholds[int(k)] = float(v)
        except (TypeError, ValueError):
            continue

    adjusted = pred.copy()
    for i, cls in enumerate(pred):
        thr = thresholds.get(int(cls))
        if thr is None:
            continue
        conf = float(y_scores[i, int(cls)])
        if conf < thr:
            # fallback to runner-up class to simulate production thresholding behavior
            ranked = np.argsort(y_scores[i])[::-1]
            for candidate in ranked:
                if int(candidate) != int(cls):
                    adjusted[i] = int(candidate)
                    break
    return adjusted


def evaluate_model(
    model: Any,
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    seq_timestamps: np.ndarray,
    splitter: Optional[TimeCheckpointedSplitter] = None,
    class_thresholds: Optional[Dict[int | str, float]] = None,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a frozen model with expanding-window walk-forward validation.
    
    Returns:
    --------
    Dict[str, Any]
        Evaluation results with folds and summary.
        If walk-forward is unavailable (insufficient data), returns:
        {
            "status": "walk_forward_unavailable",
            "folds": [],
            "summary": {"num_folds": 0, "reason": "..."}
        }
    """
    if len(X_seq) != len(y_seq) or len(y_seq) != len(seq_timestamps):
        raise ValueError("X_seq, y_seq, and seq_timestamps must have identical length.")
    if len(X_seq) == 0:
        return {
            "status": "walk_forward_unavailable",
            "folds": [],
            "summary": {"num_folds": 0, "reason": "empty_sequence_data"}
        }

    splitter = splitter or TimeCheckpointedSplitter()
    folds = splitter.split(seq_timestamps)
    if not folds:
        # Explicit status for when walk-forward cannot be performed.
        return {
            "status": "walk_forward_unavailable",
            "folds": [],
            "summary": {
                "num_folds": 0,
                "reason": "insufficient_data_for_walk_forward",
                "min_train_days": splitter.min_train_days,
                "valid_days": splitter.valid_days,
                "available_samples": len(seq_timestamps)
            }
        }

    fold_rows: List[Dict[str, Any]] = []
    f1_values: List[float] = []
    precision_values: List[float] = []
    recall_values: List[float] = []
    acc_values: List[float] = []
    stability_values: List[float] = []
    transition_values: List[float] = []

    for fold in folds:
        X_val = X_seq[fold.valid_idx]
        y_val = y_seq[fold.valid_idx]
        y_pred = _predict_with_optional_thresholds(
            model=model,
            X=X_val,
            class_thresholds=class_thresholds,
        )
        if len(y_pred) != len(y_val):
            logger.warning(f"Skipping fold {fold.fold_id}: prediction length mismatch.")
            continue

        label_space = labels if labels is not None else sorted(np.unique(np.concatenate([y_val, y_pred])).tolist())
        p, r, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, labels=label_space, average="macro", zero_division=0
        )
        acc = accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred, labels=label_space)
        support_by_label, minority_support = _fold_support_stats(y_val, label_space)
        split_metrics = _compute_stability_transition_metrics(
            y_true=y_val,
            y_pred=y_pred,
            label_space=label_space,
        )

        fold_rows.append({
            "fold_id": fold.fold_id,
            "train_start": fold.train_start.isoformat(),
            "train_end": fold.train_end.isoformat(),
            "valid_start": fold.valid_start.isoformat(),
            "valid_end": fold.valid_end.isoformat(),
            "n_train": int(len(fold.train_idx)),
            "n_valid": int(len(fold.valid_idx)),
            "macro_precision": float(p),
            "macro_recall": float(r),
            "macro_f1": float(f1),
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "labels": [int(x) for x in label_space],
            "support_by_label": {str(k): int(v) for k, v in support_by_label.items()},
            "minority_support": int(minority_support),
            "stability_accuracy": split_metrics["stability_accuracy"],
            "transition_macro_f1": split_metrics["transition_macro_f1"],
            "stability_support": int(split_metrics["stability_support"]),
            "transition_support": int(split_metrics["transition_support"]),
            "transition_events": int(split_metrics["transition_events"]),
        })
        f1_values.append(float(f1))
        precision_values.append(float(p))
        recall_values.append(float(r))
        acc_values.append(float(acc))
        if split_metrics["stability_accuracy"] is not None:
            stability_values.append(float(split_metrics["stability_accuracy"]))
        if split_metrics["transition_macro_f1"] is not None:
            transition_values.append(float(split_metrics["transition_macro_f1"]))

    summary = {
        "num_folds": int(len(fold_rows)),
        "macro_f1_mean": float(np.mean(f1_values)) if f1_values else None,
        "macro_precision_mean": float(np.mean(precision_values)) if precision_values else None,
        "macro_recall_mean": float(np.mean(recall_values)) if recall_values else None,
        "accuracy_mean": float(np.mean(acc_values)) if acc_values else None,
        "stability_accuracy_mean": float(np.mean(stability_values)) if stability_values else None,
        "transition_macro_f1_mean": float(np.mean(transition_values)) if transition_values else None,
    }
    return {
        "status": "completed",
        "folds": fold_rows,
        "summary": summary
    }


def evaluate_model_version(
    model: Any,
    platform: Any,
    room_name: str,
    room_df: pd.DataFrame,
    seq_length: int,
    scaler: Any,
    label_encoder: Any,
    splitter: Optional[TimeCheckpointedSplitter] = None,
    class_thresholds: Optional[Dict[int | str, float]] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Evaluate one concrete model version on room data using walk-forward folds.

    This helper is used by orchestration code that compares candidate vs champion
    before final promotion.
    """
    try:
        if room_df is None or room_df.empty:
            return None, f"No evaluation rows for room={room_name}."

        if scaler is None or label_encoder is None:
            return None, f"Missing scaler/label_encoder for room={room_name}."

        observed_days = _extract_observed_day_set(room_df)

        # Inference preprocessing path requires room scaler to be preloaded.
        platform.scalers[room_name] = scaler
        processed = platform.preprocess_with_resampling(
            room_df,
            room_name,
            is_training=False,
            apply_denoising=False,
        )
        processed = _filter_to_observed_days(processed, observed_days)
        if processed is None or processed.empty:
            return None, f"Preprocessing produced empty frame for room={room_name}."
        if "activity" not in processed.columns:
            return None, f"Missing 'activity' column after preprocessing for room={room_name}."
        if len(processed) < int(seq_length):
            return None, (
                f"Insufficient rows for sequenceing in room={room_name}: "
                f"{len(processed)} < {int(seq_length)}"
            )

        # Align labels to this model version's label space.
        try:
            from utils.segment_utils import normalize_activity_name, validate_activity_for_room
            canonical_activity = (
                processed["activity"]
                .astype(str)
                .str.strip()
                .str.lower()
                .apply(lambda x: validate_activity_for_room(normalize_activity_name(x), room_name))
            )
        except Exception:
            canonical_activity = processed["activity"].astype(str).str.strip().str.lower()

        classes = _encoder_classes(label_encoder)
        label_to_id = {str(label): int(idx) for idx, label in enumerate(classes)}
        if not label_to_id:
            return None, f"Label encoder has no classes for room={room_name}."

        row_labels = canonical_activity.map(label_to_id).fillna(-1).astype(int).to_numpy(dtype=np.int32)
        valid_mask = row_labels >= 0
        if not bool(np.any(valid_mask)):
            return None, f"No overlapping labels for room={room_name} against model encoder."

        if not bool(np.all(valid_mask)):
            kept = int(np.sum(valid_mask))
            dropped = int(len(valid_mask) - kept)
            logger.warning(
                f"Dropping {dropped} rows with labels absent from model encoder for {room_name}; "
                f"kept={kept}"
            )

        processed_valid = processed.loc[valid_mask].reset_index(drop=True)
        labels_valid = row_labels[valid_mask]
        if len(processed_valid) < int(seq_length):
            return None, (
                f"Insufficient valid rows for sequenceing in room={room_name}: "
                f"{len(processed_valid)} < {int(seq_length)}"
            )

        sensor_data = np.asarray(processed_valid[platform.sensor_columns].values, dtype=np.float32)
        created_sequences = platform.create_sequences(sensor_data, int(seq_length))
        if isinstance(created_sequences, tuple):
            X_seq = np.asarray(created_sequences[0])
            if len(created_sequences) > 1 and created_sequences[1] is not None:
                y_seq = np.asarray(created_sequences[1], dtype=np.int32)
            else:
                y_seq = labels_valid[int(seq_length) - 1:].astype(np.int32)
        else:
            X_seq = np.asarray(created_sequences)
            y_seq = labels_valid[int(seq_length) - 1:].astype(np.int32)

        if len(X_seq) == 0:
            return None, f"No sequences generated for room={room_name}."

        if len(y_seq) != len(X_seq):
            y_seq = labels_valid[int(seq_length) - 1:].astype(np.int32)[:len(X_seq)]

        seq_timestamps = np.asarray(
            pd.to_datetime(processed_valid["timestamp"]).iloc[int(seq_length) - 1:],
            dtype="datetime64[ns]",
        )
        if len(seq_timestamps) > len(X_seq):
            seq_timestamps = seq_timestamps[:len(X_seq)]
        elif len(seq_timestamps) < len(X_seq):
            if len(seq_timestamps) == 0:
                seq_timestamps = np.full(len(X_seq), np.datetime64("NaT"), dtype="datetime64[ns]")
            else:
                pad = np.full(len(X_seq) - len(seq_timestamps), seq_timestamps[-1], dtype="datetime64[ns]")
                seq_timestamps = np.concatenate([seq_timestamps, pad], axis=0)

        label_ids = list(range(len(classes)))
        report = evaluate_model(
            model=model,
            X_seq=X_seq,
            y_seq=y_seq,
            seq_timestamps=seq_timestamps,
            splitter=splitter,
            class_thresholds=class_thresholds,
            labels=label_ids,
        )
        return report, None
    except Exception as e:
        logger.error(f"evaluate_model_version failed for room={room_name}: {e}")
        return None, str(e)


def evaluate_baseline_version(
    platform: Any,
    room_name: str,
    room_df: pd.DataFrame,
    seq_length: int,
    scaler: Any,
    label_encoder: Any,
    splitter: Optional[TimeCheckpointedSplitter] = None,
    baseline_model: str = "xgboost",
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Evaluate a tabular baseline on walk-forward folds.

    The baseline is trained per fold on historical windows and validated on the
    next fold window, mirroring walk-forward temporal constraints.
    """
    try:
        if room_df is None or room_df.empty:
            return None, f"No evaluation rows for room={room_name}."

        if scaler is None or label_encoder is None:
            return None, f"Missing scaler/label_encoder for room={room_name}."

        observed_days = _extract_observed_day_set(room_df)

        platform.scalers[room_name] = scaler
        processed = platform.preprocess_with_resampling(
            room_df,
            room_name,
            is_training=False,
            apply_denoising=False,
        )
        processed = _filter_to_observed_days(processed, observed_days)
        if processed is None or processed.empty:
            return None, f"Preprocessing produced empty frame for room={room_name}."
        if "activity" not in processed.columns:
            return None, f"Missing 'activity' column after preprocessing for room={room_name}."
        if len(processed) < int(seq_length):
            return None, (
                f"Insufficient rows for sequenceing in room={room_name}: "
                f"{len(processed)} < {int(seq_length)}"
            )

        try:
            from utils.segment_utils import normalize_activity_name, validate_activity_for_room
            canonical_activity = (
                processed["activity"]
                .astype(str)
                .str.strip()
                .str.lower()
                .apply(lambda x: validate_activity_for_room(normalize_activity_name(x), room_name))
            )
        except Exception:
            canonical_activity = processed["activity"].astype(str).str.strip().str.lower()

        classes = _encoder_classes(label_encoder)
        label_to_id = {str(label): int(idx) for idx, label in enumerate(classes)}
        if not label_to_id:
            return None, f"Label encoder has no classes for room={room_name}."

        row_labels = canonical_activity.map(label_to_id).fillna(-1).astype(int).to_numpy(dtype=np.int32)
        valid_mask = row_labels >= 0
        if not bool(np.any(valid_mask)):
            return None, f"No overlapping labels for room={room_name} against model encoder."

        processed_valid = processed.loc[valid_mask].reset_index(drop=True)
        labels_valid = row_labels[valid_mask]
        if len(processed_valid) < int(seq_length):
            return None, (
                f"Insufficient valid rows for sequenceing in room={room_name}: "
                f"{len(processed_valid)} < {int(seq_length)}"
            )

        sensor_data = np.asarray(processed_valid[platform.sensor_columns].values, dtype=np.float32)
        created_sequences = platform.create_sequences(sensor_data, int(seq_length))
        if isinstance(created_sequences, tuple):
            X_seq = np.asarray(created_sequences[0])
            if len(created_sequences) > 1 and created_sequences[1] is not None:
                y_seq = np.asarray(created_sequences[1], dtype=np.int32)
            else:
                y_seq = labels_valid[int(seq_length) - 1:].astype(np.int32)
        else:
            X_seq = np.asarray(created_sequences)
            y_seq = labels_valid[int(seq_length) - 1:].astype(np.int32)

        if len(X_seq) == 0:
            return None, f"No sequences generated for room={room_name}."

        if len(y_seq) != len(X_seq):
            y_seq = labels_valid[int(seq_length) - 1:].astype(np.int32)[:len(X_seq)]

        seq_timestamps = np.asarray(
            pd.to_datetime(processed_valid["timestamp"]).iloc[int(seq_length) - 1:],
            dtype="datetime64[ns]",
        )
        if len(seq_timestamps) > len(X_seq):
            seq_timestamps = seq_timestamps[:len(X_seq)]
        elif len(seq_timestamps) < len(X_seq):
            if len(seq_timestamps) == 0:
                seq_timestamps = np.full(len(X_seq), np.datetime64("NaT"), dtype="datetime64[ns]")
            else:
                pad = np.full(len(X_seq) - len(seq_timestamps), seq_timestamps[-1], dtype="datetime64[ns]")
                seq_timestamps = np.concatenate([seq_timestamps, pad], axis=0)

        splitter = splitter or TimeCheckpointedSplitter()
        folds = splitter.split(seq_timestamps)
        if not folds:
            return {"engine": "none", "folds": [], "summary": {"num_folds": 0}}, None

        baseline_model = str(baseline_model or "xgboost").strip().lower()
        engine_used = "random_forest"
        fold_rows: List[Dict[str, Any]] = []
        f1_values: List[float] = []
        precision_values: List[float] = []
        recall_values: List[float] = []
        acc_values: List[float] = []
        stability_values: List[float] = []
        transition_values: List[float] = []
        label_ids = list(range(len(classes)))

        for fold in folds:
            X_train = X_seq[fold.train_idx].reshape(len(fold.train_idx), -1)
            y_train = y_seq[fold.train_idx]
            X_val = X_seq[fold.valid_idx].reshape(len(fold.valid_idx), -1)
            y_val = y_seq[fold.valid_idx]
            if len(X_train) == 0 or len(X_val) == 0:
                continue

            clf = None
            if baseline_model == "xgboost":
                try:
                    from xgboost import XGBClassifier
                    unique_train = np.unique(y_train)
                    n_classes = int(len(unique_train))
                    xgb_kwargs: Dict[str, Any] = {
                        "n_estimators": 120,
                        "max_depth": 5,
                        "learning_rate": 0.08,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                        "random_state": 42,
                        "n_jobs": 1,
                        "eval_metric": "mlogloss",
                    }
                    if n_classes > 2:
                        xgb_kwargs["objective"] = "multi:softprob"
                        xgb_kwargs["num_class"] = n_classes
                    else:
                        xgb_kwargs["objective"] = "binary:logistic"
                    clf = XGBClassifier(**xgb_kwargs)
                    engine_used = "xgboost"
                except Exception:
                    clf = None

            if clf is None:
                clf = RandomForestClassifier(
                    n_estimators=220,
                    max_depth=10,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                )
                if baseline_model == "xgboost":
                    engine_used = "xgboost_fallback_random_forest"
                else:
                    engine_used = "random_forest"

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            p, r, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, labels=label_ids, average="macro", zero_division=0
            )
            acc = accuracy_score(y_val, y_pred)
            cm = confusion_matrix(y_val, y_pred, labels=label_ids)
            support_by_label, minority_support = _fold_support_stats(y_val, label_ids)
            split_metrics = _compute_stability_transition_metrics(
                y_true=y_val,
                y_pred=y_pred,
                label_space=label_ids,
            )
            fold_rows.append(
                {
                    "fold_id": int(fold.fold_id),
                    "train_start": fold.train_start.isoformat(),
                    "train_end": fold.train_end.isoformat(),
                    "valid_start": fold.valid_start.isoformat(),
                    "valid_end": fold.valid_end.isoformat(),
                    "n_train": int(len(fold.train_idx)),
                    "n_valid": int(len(fold.valid_idx)),
                    "macro_precision": float(p),
                    "macro_recall": float(r),
                    "macro_f1": float(f1),
                    "accuracy": float(acc),
                    "confusion_matrix": cm.tolist(),
                    "labels": [int(x) for x in label_ids],
                    "support_by_label": {str(k): int(v) for k, v in support_by_label.items()},
                    "minority_support": int(minority_support),
                    "stability_accuracy": split_metrics["stability_accuracy"],
                    "transition_macro_f1": split_metrics["transition_macro_f1"],
                    "stability_support": int(split_metrics["stability_support"]),
                    "transition_support": int(split_metrics["transition_support"]),
                    "transition_events": int(split_metrics["transition_events"]),
                }
            )
            f1_values.append(float(f1))
            precision_values.append(float(p))
            recall_values.append(float(r))
            acc_values.append(float(acc))
            if split_metrics["stability_accuracy"] is not None:
                stability_values.append(float(split_metrics["stability_accuracy"]))
            if split_metrics["transition_macro_f1"] is not None:
                transition_values.append(float(split_metrics["transition_macro_f1"]))

        summary = {
            "num_folds": int(len(fold_rows)),
            "macro_f1_mean": float(np.mean(f1_values)) if f1_values else None,
            "macro_precision_mean": float(np.mean(precision_values)) if precision_values else None,
            "macro_recall_mean": float(np.mean(recall_values)) if recall_values else None,
            "accuracy_mean": float(np.mean(acc_values)) if acc_values else None,
            "stability_accuracy_mean": float(np.mean(stability_values)) if stability_values else None,
            "transition_macro_f1_mean": float(np.mean(transition_values)) if transition_values else None,
        }
        return {"engine": engine_used, "folds": fold_rows, "summary": summary}, None
    except Exception as e:
        logger.error(f"evaluate_baseline_version failed for room={room_name}: {e}")
        return None, str(e)
