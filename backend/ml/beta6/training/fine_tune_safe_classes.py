"""Phase 3.1 safe-class fine-tuning on Golden samples."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from ml.yaml_compat import load_yaml_file

from ..evaluation.evaluation_engine import evaluate_leakage
from .self_supervised_pretrain import encode_with_checkpoint, load_pretrain_checkpoint


FINE_TUNE_VERSION = "v1"


@dataclass(frozen=True)
class SafeFineTuneConfig:
    safe_classes: Tuple[str, ...] = ("sleep", "nap", "shower", "out")
    min_samples_per_class: int = 8
    min_unique_residents: int = 2
    holdout_fraction: float = 0.2
    random_seed: int = 42
    min_accuracy: float = 0.85
    max_split_attempts: int = 40


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_mapping(payload: Any) -> Mapping[str, Any]:
    return payload if isinstance(payload, Mapping) else {}


def _normalize_label(value: Any) -> str:
    return str(value).strip().lower()


def load_safe_finetune_config(path: str | Path | None) -> SafeFineTuneConfig:
    if path is None:
        return SafeFineTuneConfig()
    raw = load_yaml_file(Path(path).resolve()) or {}
    section = _as_mapping(raw).get("fine_tune")
    if not isinstance(section, Mapping):
        section = raw if isinstance(raw, Mapping) else {}
    safe_raw = section.get("safe_classes", ["sleep", "nap", "shower", "out"])
    safe_classes = tuple(
        sorted(
            {
                _normalize_label(token)
                for token in (safe_raw if isinstance(safe_raw, (list, tuple)) else [safe_raw])
                if _normalize_label(token)
            }
        )
    )
    if not safe_classes:
        raise ValueError("fine_tune.safe_classes must not be empty")
    return SafeFineTuneConfig(
        safe_classes=safe_classes,
        min_samples_per_class=max(int(section.get("min_samples_per_class", 8)), 1),
        min_unique_residents=max(int(section.get("min_unique_residents", 2)), 2),
        holdout_fraction=min(max(float(section.get("holdout_fraction", 0.2)), 0.05), 0.5),
        random_seed=int(section.get("random_seed", 42)),
        min_accuracy=min(max(float(section.get("min_accuracy", 0.85)), 0.0), 1.0),
        max_split_attempts=max(int(section.get("max_split_attempts", 40)), 1),
    )


def _load_dataset_file(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"dataset payload must be object: {path}")
    return payload


def load_golden_samples(dataset_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(dataset_path).resolve()
    files = [path] if path.is_file() else sorted(path.glob("golden_dataset_*.json"))
    if not files:
        raise ValueError(f"no golden dataset files found at: {path}")
    samples: List[Dict[str, Any]] = []
    for file_path in files:
        payload = _load_dataset_file(file_path)
        rows = payload.get("samples", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping):
                samples.append(dict(row))
    return samples


def _sensor_vector_from_sample(
    sample: Mapping[str, Any],
    feature_keys: Sequence[str],
) -> np.ndarray:
    sensor_features = sample.get("sensor_features")
    if isinstance(sensor_features, Mapping):
        values = [float(sensor_features.get(key, 0.0) or 0.0) for key in feature_keys]
        return np.asarray(values, dtype=np.float32)

    sensor_window = sample.get("sensor_window")
    if isinstance(sensor_window, list) and sensor_window:
        stacked: List[List[float]] = []
        for row in sensor_window:
            if not isinstance(row, Mapping):
                continue
            stacked.append([float(row.get(key, 0.0) or 0.0) for key in feature_keys])
        if stacked:
            return np.asarray(stacked, dtype=np.float32).mean(axis=0)
    raise ValueError("sample missing valid sensor features")


def _infer_feature_keys(samples: Sequence[Mapping[str, Any]]) -> List[str]:
    keys = set()
    for sample in samples:
        sensor_features = sample.get("sensor_features")
        if isinstance(sensor_features, Mapping):
            for key, value in sensor_features.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    keys.add(str(key))
    if not keys:
        raise ValueError("unable to infer numeric sensor feature keys from samples")
    return sorted(keys)


def _build_finetune_rows(
    samples: Sequence[Mapping[str, Any]],
    *,
    safe_classes: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    safe_set = {str(label).strip().lower() for label in safe_classes}
    if not safe_set:
        raise ValueError("safe_classes must not be empty")
    filtered = [sample for sample in samples if _normalize_label(sample.get("activity", "")) in safe_set]
    if not filtered:
        raise ValueError("no safe-class samples available after filtering")

    feature_keys = _infer_feature_keys(filtered)
    x_rows: List[np.ndarray] = []
    y_rows: List[str] = []
    residents: List[str] = []
    timestamps: List[float] = []
    for sample in filtered:
        try:
            vec = _sensor_vector_from_sample(sample, feature_keys)
        except Exception:
            continue
        if not np.isfinite(vec).all():
            continue
        x_rows.append(vec)
        y_rows.append(_normalize_label(sample.get("activity", "")))
        residents.append(str(sample.get("elder_id", "")).strip() or "unknown")
        ts = sample.get("timestamp")
        if ts is None:
            timestamps.append(0.0)
        else:
            try:
                timestamps.append(float(np.datetime64(str(ts)).astype("datetime64[s]").astype(np.int64)))
            except Exception:
                timestamps.append(0.0)

    if not x_rows:
        raise ValueError("no usable safe-class samples with numeric features")
    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=object),
        np.asarray(residents, dtype=object),
        np.asarray(timestamps, dtype=np.float64),
        feature_keys,
    )


def _class_counts(labels: Sequence[str] | np.ndarray) -> Dict[str, int]:
    values = [str(item) for item in labels]
    return {label: values.count(label) for label in sorted(set(values))}


def _resident_disjoint_split(
    resident_ids: np.ndarray,
    labels: np.ndarray,
    *,
    holdout_fraction: float,
    seed: int,
    max_split_attempts: int,
) -> Tuple[np.ndarray, np.ndarray]:
    residents = sorted(set(str(v) for v in resident_ids))
    if len(residents) < 2:
        raise ValueError("resident-disjoint split requires at least 2 unique residents")

    rng = np.random.default_rng(seed)
    classes = sorted(set(str(v) for v in labels))
    for _ in range(max_split_attempts):
        shuffled = list(residents)
        rng.shuffle(shuffled)
        split_idx = max(1, int(round(len(shuffled) * (1.0 - holdout_fraction))))
        split_idx = min(split_idx, len(shuffled) - 1)
        train_residents = set(shuffled[:split_idx])
        train_mask = np.array([str(v) in train_residents for v in resident_ids], dtype=bool)
        test_mask = ~train_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        train_labels = {str(v) for v in labels[train_mask]}
        test_labels = {str(v) for v in labels[test_mask]}
        if all(label in train_labels and label in test_labels for label in classes):
            return train_mask, test_mask
    raise ValueError("unable to form resident-disjoint split with class coverage")


def run_safe_class_finetune(
    *,
    dataset_path: str | Path,
    config_path: str | Path | None,
    output_dir: str | Path,
    pretrain_checkpoint: str | Path | None = None,
) -> Dict[str, Any]:
    config = load_safe_finetune_config(config_path)
    samples = load_golden_samples(dataset_path)
    x, y, residents, timestamps, feature_keys = _build_finetune_rows(
        samples,
        safe_classes=config.safe_classes,
    )

    class_counts = _class_counts(y)
    missing_required = [label for label in config.safe_classes if label not in class_counts]
    low_support = [label for label, count in class_counts.items() if count < config.min_samples_per_class]
    if missing_required:
        raise ValueError(f"missing safe classes in dataset: {missing_required}")
    if low_support:
        raise ValueError(f"insufficient support for safe classes: {low_support}")
    if len(set(str(r) for r in residents)) < config.min_unique_residents:
        raise ValueError("insufficient unique residents for resident-disjoint fine-tune")

    train_mask, test_mask = _resident_disjoint_split(
        residents,
        y,
        holdout_fraction=config.holdout_fraction,
        seed=config.random_seed,
        max_split_attempts=config.max_split_attempts,
    )

    x_train = x[train_mask]
    x_test = x[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    embedding_dim = x.shape[1]
    checkpoint_path = None
    if pretrain_checkpoint is not None:
        checkpoint_path = Path(pretrain_checkpoint).resolve()
        checkpoint = load_pretrain_checkpoint(checkpoint_path)
        x_train = encode_with_checkpoint(x_train, checkpoint)
        x_test = encode_with_checkpoint(x_test, checkpoint)
        embedding_dim = x_train.shape[1]

    classifier = LogisticRegression(max_iter=400, random_state=config.random_seed)
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    accuracy = float(accuracy_score(y_test, pred))

    train_windows = [
        (str(residents[idx]), float(timestamps[idx]), float(timestamps[idx]))
        for idx, keep in enumerate(train_mask)
        if keep
    ]
    test_windows = [
        (str(residents[idx]), float(timestamps[idx]), float(timestamps[idx]))
        for idx, keep in enumerate(test_mask)
        if keep
    ]
    leakage = evaluate_leakage(
        train_resident_ids=[str(v) for v in residents[train_mask]],
        validation_resident_ids=[str(v) for v in residents[test_mask]],
        train_windows=train_windows,
        validation_windows=test_windows,
        gap_seconds=0.0,
    )
    leakage_warnings = []
    if leakage.resident_overlap:
        leakage_warnings.append("resident_overlap")
    if leakage.time_overlap:
        leakage_warnings.append("time_overlap")
    if leakage.window_overlap:
        leakage_warnings.append("window_overlap")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "safe_class_head.joblib"
    report_path = out_dir / "safe_class_finetune_report.json"
    joblib.dump(
        {
            "version": FINE_TUNE_VERSION,
            "model": classifier,
            "feature_keys": feature_keys,
            "safe_classes": list(config.safe_classes),
        },
        model_path,
    )

    status = "pass" if accuracy >= config.min_accuracy and not leakage_warnings else "fail"
    report = {
        "version": FINE_TUNE_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "dataset_path": str(Path(dataset_path).resolve()),
        "pretrain_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "config": {
            "safe_classes": list(config.safe_classes),
            "min_samples_per_class": config.min_samples_per_class,
            "min_unique_residents": config.min_unique_residents,
            "holdout_fraction": config.holdout_fraction,
            "random_seed": config.random_seed,
            "min_accuracy": config.min_accuracy,
        },
        "metrics": {
            "heldout_accuracy": accuracy,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "train_residents": int(len(set(str(v) for v in residents[train_mask]))),
            "test_residents": int(len(set(str(v) for v in residents[test_mask]))),
            "embedding_dim": int(embedding_dim),
        },
        "class_balance": {
            "counts": class_counts,
            "min_count": int(min(class_counts.values())),
            "max_count": int(max(class_counts.values())),
        },
        "leakage": {
            "resident_overlap": bool(leakage.resident_overlap),
            "time_overlap": bool(leakage.time_overlap),
            "window_overlap": bool(leakage.window_overlap),
            "warnings": leakage_warnings,
        },
        "artifacts": {
            "model_joblib": str(model_path),
            "report_json": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = [
    "FINE_TUNE_VERSION",
    "SafeFineTuneConfig",
    "load_golden_samples",
    "load_safe_finetune_config",
    "run_safe_class_finetune",
]
