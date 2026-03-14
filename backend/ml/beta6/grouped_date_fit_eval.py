"""Reusable Beta6.2 grouped-date candidate fit/eval runner."""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from elderlycare_v1_16.platform import ElderlyCarePlatform
from ml.evaluation import evaluate_model
from ml.registry import ModelRegistry
from ml.sequence_alignment import safe_create_sequences
from ml.training import TrainingPipeline
from ml.utils import calculate_sequence_length

from .grouped_date_supervised import (
    ROOMS,
    VALID_SPLITS,
    _ensure_mapping,
    _ensure_nonempty_str,
    _normalize_room_name,
    load_grouped_date_supervised_manifest,
)


REPORT_SCHEMA_VERSION = "beta62.grouped_date_fit_eval_report.v1"
SUPPORTED_INPUT_REPORT_SCHEMA = "beta62.grouped_date_supervised_report.v1"
REQUIRED_CANDIDATE_ARTIFACT_KEYS = ("model", "scaler", "label_encoder")
SEGMENT_LINEAGE_COLUMNS = ("__source_path", "__segment_date", "__segment_role", "__segment_split")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _normalize_manifest_reference(
    *,
    supervised_report: Mapping[str, Any] | None,
    supervised_report_path: str | Path | None,
    manifest: Mapping[str, Any] | None,
    manifest_path: str | Path | None,
) -> dict[str, Any]:
    manifest_payload = manifest
    manifest_path_value: str | None = str(Path(manifest_path).resolve()) if manifest_path else None

    if supervised_report is not None:
        manifest_payload = _ensure_mapping("supervised_report.manifest", supervised_report.get("manifest") or {})
        if manifest_path_value is None:
            raw_manifest_path = supervised_report.get("manifest_path")
            if raw_manifest_path:
                manifest_path_value = str(Path(str(raw_manifest_path)).resolve())

    if manifest_payload is None:
        raise ValueError("manifest payload is required")

    digest = None
    if manifest_path_value:
        candidate = Path(manifest_path_value)
        if candidate.exists():
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if digest is None:
        digest = _json_digest(manifest_payload)

    resident_id = _ensure_nonempty_str("manifest.resident_id", manifest_payload.get("resident_id"))
    raw_target_rooms = manifest_payload.get("target_rooms") or []
    target_rooms = [_normalize_room_name(str(room)) for room in raw_target_rooms]

    sequence_length_by_room_raw = manifest_payload.get("sequence_length_by_room") or {}
    sequence_length_by_room = {
        _normalize_room_name(str(room)): max(int(value), 1)
        for room, value in _ensure_mapping("manifest.sequence_length_by_room", sequence_length_by_room_raw).items()
    } if sequence_length_by_room_raw else {}

    return {
        "payload": dict(manifest_payload),
        "path": manifest_path_value,
        "sha256": digest,
        "resident_id": resident_id,
        "target_rooms": target_rooms,
        "sequence_length_by_room": sequence_length_by_room,
    }


def _load_supervised_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("grouped-date supervised report must decode to an object")
    schema_version = _ensure_nonempty_str("supervised_report.schema_version", payload.get("schema_version"))
    if schema_version != SUPPORTED_INPUT_REPORT_SCHEMA:
        raise ValueError(f"unsupported grouped-date supervised report schema: {schema_version}")
    return payload


def _resolve_artifact_paths(
    *,
    artifact_dir: Path,
    resident_id: str,
    room_key: str,
    report_room: Mapping[str, Any] | None,
) -> dict[str, str]:
    split_paths: dict[str, str] = {}
    if report_room is not None:
        split_summary = _ensure_mapping("room_report.split_summary", report_room.get("split_summary") or {})
        for split, summary in split_summary.items():
            split_key = _ensure_nonempty_str("split_summary split", split).lower()
            if split_key not in VALID_SPLITS:
                continue
            summary_map = _ensure_mapping(f"split_summary[{split_key}]", summary)
            artifact_path = summary_map.get("artifact_path")
            if artifact_path:
                split_paths[split_key] = str(Path(str(artifact_path)).resolve())

    for split in VALID_SPLITS:
        split_paths.setdefault(
            split,
            str((artifact_dir / f"{resident_id}_{room_key}_{split}.parquet").resolve()),
        )
    return split_paths


def _load_split_frames(
    *,
    artifact_dir: Path,
    manifest_reference: Mapping[str, Any],
    supervised_report: Mapping[str, Any] | None,
    target_rooms: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    resident_id = str(manifest_reference["resident_id"])
    room_reports = _ensure_mapping("supervised_report.room_reports", (supervised_report or {}).get("room_reports") or {})

    room_inputs: dict[str, dict[str, dict[str, Any]]] = {}
    for room_key in target_rooms:
        report_room = room_reports.get(room_key)
        split_paths = _resolve_artifact_paths(
            artifact_dir=artifact_dir,
            resident_id=resident_id,
            room_key=room_key,
            report_room=report_room if isinstance(report_room, Mapping) else None,
        )
        split_payloads: dict[str, dict[str, Any]] = {}
        for split, raw_path in split_paths.items():
            path = Path(raw_path)
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            split_payloads[split] = {
                "path": str(path),
                "frame": frame,
            }
        if "train" not in split_payloads:
            raise FileNotFoundError(f"missing required train artifact for room={room_key}")
        if "holdout" not in split_payloads:
            raise FileNotFoundError(f"missing required holdout artifact for room={room_key}")
        room_inputs[room_key] = split_payloads
    return room_inputs


def _canonical_room_name(room_key: str) -> str:
    token = _normalize_room_name(room_key)
    for room in ROOMS:
        if room.lower() == token:
            return room
    return str(room_key)


def _summarize_split_frame(frame: pd.DataFrame) -> dict[str, Any]:
    dates = []
    roles = []
    if "__segment_date" in frame.columns:
        dates = sorted(frame["__segment_date"].astype(str).unique().tolist())
    if "__segment_role" in frame.columns:
        roles = sorted(frame["__segment_role"].astype(str).unique().tolist())
    summary = {
        "row_count": int(len(frame)),
        "dates": dates,
        "segment_roles": roles,
    }
    if "__segment_split" in frame.columns:
        summary["segment_splits"] = sorted(frame["__segment_split"].astype(str).unique().tolist())
    return summary


def _split_frame_into_segments(frame: pd.DataFrame) -> list[pd.DataFrame]:
    if frame is None or frame.empty:
        return []
    available_keys = [column for column in SEGMENT_LINEAGE_COLUMNS if column in frame.columns]
    if not available_keys:
        return [frame.sort_values("timestamp").reset_index(drop=True)]

    segments: list[pd.DataFrame] = []
    grouped = frame.groupby(available_keys, sort=False, dropna=False)
    for _, group in grouped:
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        segments.append(ordered)

    def _segment_key(segment_df: pd.DataFrame) -> tuple[str, str]:
        date_token = ""
        if "__segment_date" in segment_df.columns and not segment_df["__segment_date"].empty:
            date_token = str(segment_df["__segment_date"].iloc[0])
        start_ts = ""
        if "timestamp" in segment_df.columns and not segment_df["timestamp"].empty:
            start_ts = pd.to_datetime(segment_df["timestamp"], errors="coerce").min().isoformat()
        return (date_token, start_ts)

    return sorted(segments, key=_segment_key)


def _room_raw_columns(
    platform: ElderlyCarePlatform,
    frame: pd.DataFrame,
    *,
    include_activity: bool,
) -> list[str]:
    columns = ["timestamp", *list(platform.sensor_columns)]
    if include_activity:
        columns.append("activity")
    return [column for column in columns if column in frame.columns]


def _prepare_segmented_training_frame(
    *,
    platform: ElderlyCarePlatform,
    room_name: str,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    preprocessed_segments: list[pd.DataFrame] = []
    for segment_df in _split_frame_into_segments(train_df):
        raw_segment = segment_df[_room_raw_columns(platform, segment_df, include_activity=True)].copy()
        preprocessed = platform.preprocess_without_scaling(
            raw_segment,
            room_name,
            is_training=True,
            apply_denoising=False,
        )
        if preprocessed is None or preprocessed.empty:
            continue
        preprocessed_segments.append(preprocessed.reset_index(drop=True))

    if not preprocessed_segments:
        raise ValueError(f"train preprocessing produced no rows for room={room_name}")

    preprocessed = pd.concat(preprocessed_segments, ignore_index=True)
    fit_range = {
        "fit_start_ts": preprocessed["timestamp"].min().isoformat() if "timestamp" in preprocessed.columns else None,
        "fit_end_ts": preprocessed["timestamp"].max().isoformat() if "timestamp" in preprocessed.columns else None,
        "fit_sample_count": int(len(preprocessed)),
    }
    processed = platform.apply_scaling(
        preprocessed,
        room_name,
        is_training=True,
        scaler_fit_range=fit_range,
    )
    if processed is None or processed.empty:
        raise ValueError(f"train scaling produced no rows for room={room_name}")
    return processed


def _prepare_segmented_holdout_frame(
    *,
    platform: ElderlyCarePlatform,
    room_name: str,
    holdout_df: pd.DataFrame,
) -> pd.DataFrame:
    processed_segments: list[pd.DataFrame] = []
    for segment_df in _split_frame_into_segments(holdout_df):
        raw_segment = segment_df[_room_raw_columns(platform, segment_df, include_activity=True)].copy()
        preprocessed = platform.preprocess_without_scaling(
            raw_segment,
            room_name,
            is_training=False,
            apply_denoising=False,
        )
        if preprocessed is None or preprocessed.empty:
            continue
        scaled = platform.apply_scaling(
            preprocessed,
            room_name,
            is_training=False,
        )
        if scaled is None or scaled.empty:
            continue
        processed_segments.append(scaled.reset_index(drop=True))

    if not processed_segments:
        raise ValueError(f"holdout preprocessing produced no rows for room={room_name}")

    return pd.concat(processed_segments, ignore_index=True)


def _build_explicit_split_sequence_data(
    *,
    platform: ElderlyCarePlatform,
    room_name: str,
    split_df: pd.DataFrame | None,
    seq_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if split_df is None or split_df.empty:
        return None

    processed = _prepare_segmented_holdout_frame(
        platform=platform,
        room_name=room_name,
        holdout_df=split_df,
    )
    if "activity_encoded" not in processed.columns:
        raise ValueError(f"explicit split preprocessing missing activity_encoded for room={room_name}")

    sensor_data = np.asarray(processed[platform.sensor_columns].values, dtype=np.float32)
    labels = np.asarray(processed["activity_encoded"].values, dtype=np.int32)
    timestamps = np.asarray(pd.to_datetime(processed["timestamp"]), dtype="datetime64[ns]")
    X_seq, y_seq, seq_timestamps = safe_create_sequences(
        platform=platform,
        sensor_data=sensor_data,
        labels=labels,
        seq_length=int(seq_length),
        room_name=room_name,
        timestamps=timestamps,
        strict=True,
    )
    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.int32),
        np.asarray(seq_timestamps, dtype="datetime64[ns]"),
    )


def _encoder_classes(label_encoder: Any) -> list[str]:
    classes = getattr(label_encoder, "classes_", None)
    if classes is None:
        return []
    try:
        return [str(item) for item in list(classes)]
    except Exception:
        return []


@contextlib.contextmanager
def _zero_validation_split():
    import ml.training as training_module

    original = float(training_module.DEFAULT_VALIDATION_SPLIT)
    training_module.DEFAULT_VALIDATION_SPLIT = 0.0
    try:
        yield
    finally:
        training_module.DEFAULT_VALIDATION_SPLIT = original


def _artifact_paths_for_version(
    *,
    registry: ModelRegistry,
    candidate_namespace: str,
    room_name: str,
    version: int,
) -> dict[str, str]:
    models_dir = registry.get_models_dir(candidate_namespace)
    paths = {
        "model": models_dir / f"{room_name}_v{int(version)}_model.keras",
        "scaler": models_dir / f"{room_name}_v{int(version)}_scaler.pkl",
        "label_encoder": models_dir / f"{room_name}_v{int(version)}_label_encoder.pkl",
        "thresholds": models_dir / f"{room_name}_v{int(version)}_thresholds.json",
        "versions": models_dir / f"{room_name}_versions.json",
    }
    return {name: str(path) for name, path in paths.items() if path.exists()}


def _has_required_candidate_artifacts(paths: Mapping[str, str]) -> bool:
    return all(paths.get(key) for key in REQUIRED_CANDIDATE_ARTIFACT_KEYS)


def _candidate_versions_from_namespace(models_dir: Path, room_name: str) -> list[int]:
    discovered: set[int] = set()
    versions_path = models_dir / f"{room_name}_versions.json"
    if versions_path.exists():
        try:
            payload = json.loads(versions_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            for item in payload.get("versions", []) or []:
                if isinstance(item, Mapping):
                    with contextlib.suppress(Exception):
                        discovered.add(int(item.get("version", 0) or 0))

    pattern = re.compile(rf"^{re.escape(room_name)}_v(\d+)_")
    for path in models_dir.glob(f"{room_name}_v*_*"):
        match = pattern.match(path.name)
        if not match:
            continue
        with contextlib.suppress(Exception):
            discovered.add(int(match.group(1)))

    return sorted((version for version in discovered if version > 0), reverse=True)


def _resolve_saved_candidate_artifacts(
    *,
    registry: ModelRegistry,
    candidate_namespace: str,
    room_name: str,
    requested_saved_version: int,
) -> dict[str, Any]:
    models_dir = registry.get_models_dir(candidate_namespace)
    candidate_versions = [int(requested_saved_version)]
    candidate_versions.extend(
        version
        for version in _candidate_versions_from_namespace(models_dir, room_name)
        if int(version) != int(requested_saved_version)
    )

    for version in candidate_versions:
        artifact_paths = _artifact_paths_for_version(
            registry=registry,
            candidate_namespace=candidate_namespace,
            room_name=room_name,
            version=version,
        )
        if _has_required_candidate_artifacts(artifact_paths):
            return {
                "requested_saved_version": int(requested_saved_version),
                "resolved_saved_version": int(version),
                "artifact_paths": artifact_paths,
            }

    raise FileNotFoundError(
        f"candidate artifacts incomplete for room={room_name} "
        f"(requested_saved_version={int(requested_saved_version)})"
    )


def _fit_room_candidate(
    *,
    room_name: str,
    split_frames: Mapping[str, pd.DataFrame],
    candidate_namespace: str,
    backend_dir: Path,
    seq_length: int | None = None,
) -> dict[str, Any]:
    platform = ElderlyCarePlatform()
    registry = ModelRegistry(str(backend_dir))
    pipeline = TrainingPipeline(platform=platform, registry=registry)
    pipeline.augment_training_data = (
        lambda room_name, elder_id, existing_ts, X_seq, y_seq, window_sec, interval, seq_timestamps:
        (X_seq, y_seq, seq_timestamps)
    )

    train_df = split_frames["train"]
    processed_train = _prepare_segmented_training_frame(
        platform=platform,
        room_name=room_name,
        train_df=train_df,
    )
    resolved_seq_length = int(seq_length or calculate_sequence_length(platform, room_name))
    explicit_validation_data = _build_explicit_split_sequence_data(
        platform=platform,
        room_name=room_name,
        split_df=split_frames.get("validation"),
        seq_length=resolved_seq_length,
    )
    explicit_calibration_sequences = _build_explicit_split_sequence_data(
        platform=platform,
        room_name=room_name,
        split_df=split_frames.get("calibration"),
        seq_length=resolved_seq_length,
    )
    explicit_calibration_data = (
        None
        if explicit_calibration_sequences is None
        else (
            explicit_calibration_sequences[0],
            explicit_calibration_sequences[1],
        )
    )

    train_room_kwargs = {
        "room_name": room_name,
        "processed_df": processed_train,
        "seq_length": resolved_seq_length,
        "elder_id": candidate_namespace,
        "training_mode": "full_retrain",
        "defer_promotion": True,
    }
    if explicit_validation_data is not None:
        train_room_kwargs["explicit_validation_data"] = explicit_validation_data
    if explicit_calibration_data is not None:
        train_room_kwargs["explicit_calibration_data"] = explicit_calibration_data

    if explicit_validation_data is not None or explicit_calibration_data is not None:
        fit_metrics = pipeline.train_room(**train_room_kwargs)
    else:
        with _zero_validation_split():
            fit_metrics = pipeline.train_room(**train_room_kwargs)
    if not isinstance(fit_metrics, dict):
        raise ValueError(f"fit did not return metrics for room={room_name}")
    saved_version = int(fit_metrics.get("saved_version", 0) or 0)
    if saved_version <= 0:
        raise ValueError(f"candidate save did not produce a version for room={room_name}")
    artifact_resolution = _resolve_saved_candidate_artifacts(
        registry=registry,
        candidate_namespace=candidate_namespace,
        room_name=room_name,
        requested_saved_version=saved_version,
    )
    return {
        "saved_version": int(artifact_resolution["resolved_saved_version"]),
        "requested_saved_version": saved_version,
        "fit_metrics": fit_metrics,
        "candidate_artifact_paths": dict(artifact_resolution["artifact_paths"]),
        "artifact_resolution": {
            "requested_saved_version": int(artifact_resolution["requested_saved_version"]),
            "resolved_saved_version": int(artifact_resolution["resolved_saved_version"]),
            "fallback_used": int(artifact_resolution["resolved_saved_version"]) != int(saved_version),
        },
        "platform": platform,
        "registry": registry,
        "seq_length": resolved_seq_length,
    }


def _load_thresholds(path: str | None) -> dict[str, float] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return {str(key): float(value) for key, value in payload.items()}


def _evaluate_room_candidate(
    *,
    room_name: str,
    holdout_df: pd.DataFrame,
    fit_result: Mapping[str, Any],
) -> dict[str, Any]:
    registry = fit_result["registry"]
    platform = fit_result["platform"]
    candidate_artifact_paths = fit_result.get("candidate_artifact_paths") or {}
    model_path = candidate_artifact_paths.get("model")
    scaler_path = candidate_artifact_paths.get("scaler")
    label_encoder_path = candidate_artifact_paths.get("label_encoder")
    if not model_path or not scaler_path or not label_encoder_path:
        raise FileNotFoundError(f"candidate artifacts incomplete for room={room_name}")

    model = registry.load_room_model(str(model_path), room_name, compile_model=False)
    scaler = joblib.load(str(scaler_path))
    label_encoder = joblib.load(str(label_encoder_path))
    thresholds = _load_thresholds(candidate_artifact_paths.get("thresholds"))
    platform.scalers[room_name] = scaler
    processed = _prepare_segmented_holdout_frame(
        platform=platform,
        room_name=room_name,
        holdout_df=holdout_df,
    )
    if "activity" not in processed.columns:
        raise ValueError(f"holdout preprocessing missing activity labels for room={room_name}")

    classes = _encoder_classes(label_encoder)
    if not classes:
        raise ValueError(f"label encoder has no classes for room={room_name}")
    label_to_id = {str(label): int(idx) for idx, label in enumerate(classes)}
    canonical_activity = processed["activity"].astype(str).str.strip().str.lower()
    row_labels = canonical_activity.map(label_to_id).fillna(-1).astype(int).to_numpy(dtype="int32")
    valid_mask = row_labels >= 0
    if not bool(valid_mask.any()):
        raise ValueError(f"holdout evaluation has no overlapping labels for room={room_name}")

    processed_valid = processed.loc[valid_mask].reset_index(drop=True)
    labels_valid = row_labels[valid_mask]
    sensor_data = np.asarray(processed_valid[platform.sensor_columns].values, dtype=np.float32)
    timestamps = np.asarray(pd.to_datetime(processed_valid["timestamp"]), dtype="datetime64[ns]")
    X_seq, y_seq, seq_timestamps = safe_create_sequences(
        platform=platform,
        sensor_data=sensor_data,
        labels=labels_valid,
        seq_length=int(fit_result["seq_length"]),
        room_name=room_name,
        timestamps=timestamps,
        strict=True,
    )
    report = evaluate_model(
        model=model,
        X_seq=X_seq,
        y_seq=y_seq,
        seq_timestamps=seq_timestamps,
        class_thresholds=thresholds,
        labels=list(range(len(classes))),
    )
    return report or {}


def _clean_fit_result(fit_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "saved_version": int(fit_result.get("saved_version", 0) or 0),
        "requested_saved_version": int(fit_result.get("requested_saved_version", 0) or 0),
        "fit_metrics": dict(fit_result.get("fit_metrics") or {}),
        "artifact_resolution": dict(fit_result.get("artifact_resolution") or {}),
    }


def run_grouped_date_fit_eval(
    *,
    supervised_report: Mapping[str, Any] | None = None,
    supervised_report_path: str | Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    manifest_path: str | Path | None = None,
    artifact_dir: str | Path,
    candidate_namespace: str,
    backend_dir: str | Path | None = None,
) -> dict[str, Any]:
    if supervised_report is None and supervised_report_path is not None:
        supervised_report = _load_supervised_report(supervised_report_path)
    elif supervised_report is not None:
        schema_version = _ensure_nonempty_str("supervised_report.schema_version", supervised_report.get("schema_version"))
        if schema_version != SUPPORTED_INPUT_REPORT_SCHEMA:
            raise ValueError(f"unsupported grouped-date supervised report schema: {schema_version}")

    if manifest is None and manifest_path is not None:
        manifest = load_grouped_date_supervised_manifest(manifest_path)

    artifact_root = Path(artifact_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    candidate_namespace = _ensure_nonempty_str("candidate_namespace", candidate_namespace)
    resolved_backend_dir = Path(backend_dir).resolve() if backend_dir is not None else Path(__file__).resolve().parents[2]

    manifest_reference = _normalize_manifest_reference(
        supervised_report=supervised_report,
        supervised_report_path=supervised_report_path,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    target_rooms = list(manifest_reference["target_rooms"])
    room_inputs = _load_split_frames(
        artifact_dir=artifact_root,
        manifest_reference=manifest_reference,
        supervised_report=supervised_report,
        target_rooms=target_rooms,
    )

    room_results: dict[str, dict[str, Any]] = {}
    for room_key in target_rooms:
        split_payloads = room_inputs.get(room_key)
        if not split_payloads:
            continue
        split_frames = {split: payload["frame"] for split, payload in split_payloads.items()}
        room_title = _canonical_room_name(room_key)
        seq_length = manifest_reference["sequence_length_by_room"].get(room_key)
        fit_result = _fit_room_candidate(
            room_name=room_title,
            split_frames=split_frames,
            candidate_namespace=candidate_namespace,
            backend_dir=resolved_backend_dir,
            seq_length=seq_length,
        )
        holdout_metrics = _evaluate_room_candidate(
            room_name=room_title,
            holdout_df=split_frames["holdout"],
            fit_result=fit_result,
        )
        room_results[room_key] = {
            "split_counts": {
                split: int(len(payload["frame"]))
                for split, payload in sorted(split_payloads.items())
            },
            "artifact_paths": {
                split: str(payload["path"])
                for split, payload in sorted(split_payloads.items())
            },
            "lineage": {
                split: _summarize_split_frame(payload["frame"])
                for split, payload in sorted(split_payloads.items())
            },
            "fit_result": _clean_fit_result(fit_result),
            "candidate_artifact_paths": dict(fit_result.get("candidate_artifact_paths") or {}),
            "holdout_metrics": holdout_metrics,
        }

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "candidate_namespace": candidate_namespace,
        "artifact_dir": str(artifact_root),
        "manifest": {
            "path": manifest_reference["path"],
            "sha256": manifest_reference["sha256"],
            "resident_id": manifest_reference["resident_id"],
            "target_rooms": target_rooms,
        },
        "room_results": room_results,
    }


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "run_grouped_date_fit_eval",
]
