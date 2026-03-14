"""Reusable Beta6.2 grouped-date candidate fit/eval runner."""

from __future__ import annotations

import contextlib
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from elderlycare_v1_16.platform import ElderlyCarePlatform
from ml.evaluation import evaluate_model_version
from ml.registry import ModelRegistry
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


def _fit_scaling_for_train(platform: ElderlyCarePlatform, room_name: str, train_df: pd.DataFrame) -> pd.DataFrame:
    preprocessed = platform.preprocess_without_scaling(
        train_df,
        room_name,
        is_training=True,
        apply_denoising=False,
    )
    if preprocessed is None or preprocessed.empty:
        raise ValueError(f"train preprocessing produced no rows for room={room_name}")
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


@contextlib.contextmanager
def _zero_validation_split():
    import ml.training as training_module

    original = float(training_module.DEFAULT_VALIDATION_SPLIT)
    training_module.DEFAULT_VALIDATION_SPLIT = 0.0
    try:
        yield
    finally:
        training_module.DEFAULT_VALIDATION_SPLIT = original


def _artifact_paths_for_saved_candidate(
    *,
    registry: ModelRegistry,
    candidate_namespace: str,
    room_name: str,
    saved_version: int,
) -> dict[str, str]:
    models_dir = registry.get_models_dir(candidate_namespace)
    paths = {
        "model": models_dir / f"{room_name}_v{int(saved_version)}_model.keras",
        "scaler": models_dir / f"{room_name}_v{int(saved_version)}_scaler.pkl",
        "label_encoder": models_dir / f"{room_name}_v{int(saved_version)}_label_encoder.pkl",
        "thresholds": models_dir / f"{room_name}_v{int(saved_version)}_thresholds.json",
        "versions": models_dir / f"{room_name}_versions.json",
    }
    return {name: str(path) for name, path in paths.items() if path.exists()}


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
    processed_train = _fit_scaling_for_train(platform, room_name, train_df)
    resolved_seq_length = int(seq_length or calculate_sequence_length(platform, room_name))
    with _zero_validation_split():
        fit_metrics = pipeline.train_room(
            room_name=room_name,
            processed_df=processed_train,
            seq_length=resolved_seq_length,
            elder_id=candidate_namespace,
            training_mode="full_retrain",
            defer_promotion=True,
        )
    if not isinstance(fit_metrics, dict):
        raise ValueError(f"fit did not return metrics for room={room_name}")
    saved_version = int(fit_metrics.get("saved_version", 0) or 0)
    if saved_version <= 0:
        raise ValueError(f"candidate save did not produce a version for room={room_name}")
    return {
        "saved_version": saved_version,
        "fit_metrics": fit_metrics,
        "candidate_artifact_paths": _artifact_paths_for_saved_candidate(
            registry=registry,
            candidate_namespace=candidate_namespace,
            room_name=room_name,
            saved_version=saved_version,
        ),
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
    report, err = evaluate_model_version(
        model=model,
        platform=platform,
        room_name=room_name,
        room_df=holdout_df,
        seq_length=int(fit_result["seq_length"]),
        scaler=scaler,
        label_encoder=label_encoder,
        class_thresholds=thresholds,
    )
    if err:
        raise ValueError(f"holdout evaluation failed for room={room_name}: {err}")
    return report or {}


def _clean_fit_result(fit_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "saved_version": int(fit_result.get("saved_version", 0) or 0),
        "fit_metrics": dict(fit_result.get("fit_metrics") or {}),
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
