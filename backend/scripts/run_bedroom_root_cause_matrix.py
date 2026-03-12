#!/usr/bin/env python3
"""Build and optionally execute the Bedroom-first root-cause date-ablation matrix."""

from __future__ import annotations

import argparse
import json
import shutil
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"

import sys

for path in (ROOT, BACKEND):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ml.pipeline import UnifiedPipeline
from ml.registry import ModelRegistry
from utils.data_loader import load_sensor_data


ROOM = "Bedroom"
SOURCE_NAMESPACE = "HK0011_jessica"
EVAL_DATE = "2025-12-17"
ANCHOR_DATES = ("2025-12-10", "2025-12-17")
ADDED_DATES = (
    "2025-12-04",
    "2025-12-05",
    "2025-12-06",
    "2025-12-07",
    "2025-12-08",
    "2025-12-09",
)
DEFAULT_SOURCE_FILES_BY_DATE = {
    "2025-12-04": "/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_4dec2025.xlsx",
    "2025-12-05": "/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_5dec2025.xlsx",
    "2025-12-06": "/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_6dec2025.xlsx",
    "2025-12-07": "/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_7dec2025.xlsx",
    "2025-12-08": "/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_8dec2025.xlsx",
    "2025-12-09": "/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_9dec2025.xlsx",
    "2025-12-10": "/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_10dec2025.xlsx",
    "2025-12-17": "/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_17dec2025.xlsx",
}
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _sanitize_token(raw: str) -> str:
    text = str(raw or "").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text) or "unknown"


def _variant_slug(name: str) -> str:
    return _sanitize_token(str(name).lower())


def build_variant_specs(
    *,
    source_files_by_date: Mapping[str, str] | None = None,
    anchor_dates: Sequence[str] = ANCHOR_DATES,
    added_dates: Sequence[str] = ADDED_DATES,
) -> list[dict[str, Any]]:
    files_by_date = dict(source_files_by_date or DEFAULT_SOURCE_FILES_BY_DATE)
    variants: list[dict[str, Any]] = []

    def _files_for_dates(dates: Sequence[str]) -> list[str]:
        return [str(files_by_date[date]) for date in dates]

    anchor_variant = {
        "name": "anchor",
        "mode": "anchor",
        "description": "Known-good Bedroom anchor using Dec 10 + Dec 17 only.",
        "included_dates": list(anchor_dates),
        "source_files": _files_for_dates(anchor_dates),
    }
    variants.append(anchor_variant)

    base_dates = list(anchor_dates)
    cumulative_added: list[str] = []
    for date in added_dates:
        single_dates = sorted([*base_dates, str(date)])
        variants.append(
            {
                "name": f"add_{date}",
                "mode": "single_add_back",
                "description": f"Anchor plus {date}.",
                "included_dates": single_dates,
                "source_files": _files_for_dates(single_dates),
                "added_date": str(date),
            }
        )
        cumulative_added.append(str(date))
        cumulative_dates = sorted([*base_dates, *cumulative_added])
        variants.append(
            {
                "name": f"cumulative_through_{date}",
                "mode": "cumulative_add_back",
                "description": f"Anchor plus cumulative add-back through {date}.",
                "included_dates": cumulative_dates,
                "source_files": _files_for_dates(cumulative_dates),
                "added_dates": list(cumulative_added),
            }
        )

    return variants


def _clone_bedroom_namespace(
    *,
    models_root: Path,
    source_namespace: str,
    candidate_namespace: str,
) -> Path:
    source_dir = models_root / source_namespace
    candidate_dir = models_root / candidate_namespace
    if candidate_dir.exists():
        raise FileExistsError(f"Candidate namespace already exists: {candidate_dir}")
    candidate_dir.mkdir(parents=True, exist_ok=False)
    copied = 0
    for source_path in source_dir.glob(f"{ROOM}*"):
        if source_path.is_file():
            shutil.copy2(source_path, candidate_dir / source_path.name)
            copied += 1
    if copied <= 0:
        raise FileNotFoundError(f"No {ROOM} artifacts found under {source_dir}")
    return candidate_dir


def _current_versions(models_dir: Path) -> dict[str, int]:
    versions: dict[str, int] = {}
    for path in models_dir.glob("*_versions.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        versions[path.name.replace("_versions.json", "")] = int(payload.get("current_version", 0) or 0)
    return dict(sorted(versions.items()))


def _truth_labels(merged: pd.DataFrame) -> list[str]:
    labels = sorted(str(value) for value in merged["truth_activity"].dropna().astype(str).unique())
    return [label for label in labels if label]


def _compute_confusion_pairs(
    truth: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for truth_label, pred_label in zip(truth, predicted):
        if truth_label == pred_label:
            continue
        key = f"{truth_label} -> {pred_label}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-int(item[1]), item[0])))


def _evaluate_bedroom_replay(
    *,
    elder_id: str,
    eval_file: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_predictions"
    comparison_dir = output_dir / "comparison"
    raw_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    pipeline = UnifiedPipeline(enable_denoising=True)
    loaded_rooms = pipeline.registry.load_models_for_elder(elder_id, pipeline.platform)
    if ROOM not in loaded_rooms:
        raise RuntimeError(f"{ROOM} not loadable for {elder_id}")

    sensor_data = load_sensor_data(Path(eval_file), resample=True)
    room_sensor_data = {ROOM: sensor_data[ROOM]}
    predictions = pipeline.predictor.run_prediction(room_sensor_data, [ROOM], seq_length=0)
    if ROOM not in predictions:
        raise RuntimeError(f"No {ROOM} predictions produced for {elder_id}")

    pred_df = predictions[ROOM].copy()
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], errors="coerce").dt.floor("10s")
    raw_path = raw_dir / f"{ROOM}.parquet"
    pred_df.to_parquet(raw_path, index=False)

    truth_df = room_sensor_data[ROOM].copy()
    truth_df["timestamp"] = pd.to_datetime(truth_df["timestamp"], errors="coerce").dt.floor("10s")
    merged = pred_df.merge(
        truth_df[["timestamp", "activity"]],
        on="timestamp",
        how="left",
    ).rename(columns={"activity": "truth_activity"})
    merged_path = comparison_dir / f"{ROOM}_merged.parquet"
    merged.to_parquet(merged_path, index=False)

    eval_rows = merged[merged["truth_activity"].notna()].copy()
    labels = _truth_labels(eval_rows)
    y_true = eval_rows["truth_activity"].astype(str)
    y_final = eval_rows["predicted_activity"].astype(str)
    y_raw_top1 = eval_rows["predicted_top1_label_raw"].astype(str)
    y_raw_export = eval_rows["predicted_activity_raw"].astype(str)

    final_report = classification_report(y_true, y_final, labels=labels, output_dict=True, zero_division=0)
    raw_report = classification_report(y_true, y_raw_top1, labels=labels, output_dict=True, zero_division=0)
    raw_export_report = classification_report(y_true, y_raw_export, labels=labels, output_dict=True, zero_division=0)

    raw_summary = {
        "rooms": [
            {
                "room": ROOM,
                "rows": int(len(pred_df)),
                "path": str(raw_path),
            }
        ]
    }
    (raw_dir / "summary.json").write_text(json.dumps(raw_summary, indent=2, default=str), encoding="utf-8")

    summary = {
        "elder_id": str(elder_id),
        "room": ROOM,
        "eval_file": str(eval_file),
        "raw_predictions": {
            "room_path": str(raw_path),
            "summary_path": str(raw_dir / "summary.json"),
        },
        "comparison": {
            "merged_path": str(merged_path),
        },
        "room_metrics": {
            ROOM: {
                "rows_evaluated": int(len(eval_rows)),
                "final_accuracy": float(accuracy_score(y_true, y_final)),
                "final_macro_f1": float(f1_score(y_true, y_final, labels=labels, average="macro", zero_division=0)),
                "raw_top1_macro_f1": float(
                    f1_score(y_true, y_raw_top1, labels=labels, average="macro", zero_division=0)
                ),
                "raw_export_macro_f1": float(
                    f1_score(y_true, y_raw_export, labels=labels, average="macro", zero_division=0)
                ),
                "rewrite_count_from_raw_top1": int((y_raw_top1 != y_final).sum()),
                "truth_class_share": {
                    str(label): float((y_true == label).mean())
                    for label in labels
                },
                "predicted_class_share": {
                    str(label): float((y_final == label).mean())
                    for label in sorted(set(y_final.astype(str)))
                },
                "dominant_error_pairs": [
                    {"pair": key, "count": int(value)}
                    for key, value in list(_compute_confusion_pairs(y_true, y_final).items())[:10]
                ],
                "critical_error_families": {
                    "sleep -> unoccupied": int(((y_true == "sleep") & (y_final == "unoccupied")).sum()),
                    "unoccupied -> bedroom_normal_use": int(
                        ((y_true == "unoccupied") & (y_final == "bedroom_normal_use")).sum()
                    ),
                    "bedroom_normal_use -> unoccupied": int(
                        ((y_true == "bedroom_normal_use") & (y_final == "unoccupied")).sum()
                    ),
                },
                "final_classification_report": final_report,
                "raw_top1_classification_report": raw_report,
                "raw_export_classification_report": raw_export_report,
            }
        },
    }
    summary_path = comparison_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _run_variant(
    *,
    variant: Mapping[str, Any],
    output_dir: Path,
    models_root: Path,
    source_namespace: str,
    candidate_namespace: str,
    eval_file: str,
) -> dict[str, Any]:
    variant_dir = output_dir / _variant_slug(str(variant["name"]))
    variant_dir.mkdir(parents=True, exist_ok=True)
    train_metrics_path = variant_dir / "train_metrics.json"
    load_sanity_path = variant_dir / "load_sanity.json"
    dec17_dir = variant_dir / "dec17_replay"

    _clone_bedroom_namespace(
        models_root=models_root,
        source_namespace=source_namespace,
        candidate_namespace=candidate_namespace,
    )

    pipeline = UnifiedPipeline(enable_denoising=True)
    results, metrics = pipeline.train_from_files(
        file_paths=list(variant["source_files"]),
        elder_id=candidate_namespace,
        rooms={ROOM},
        defer_promotion=True,
    )
    payload = {
        "status": "ok",
        "variant": str(variant["name"]),
        "candidate_namespace": str(candidate_namespace),
        "source_files": list(variant["source_files"]),
        "results": to_jsonable(results),
        "trained_rooms": to_jsonable(metrics),
    }
    train_metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    trained_room = next((entry for entry in metrics if str(entry.get("room")) == ROOM), None)
    if not isinstance(trained_room, dict):
        raise RuntimeError(f"No {ROOM} metrics found for variant {variant['name']}")
    saved_version = int(trained_room.get("saved_version", 0) or 0)
    if saved_version <= 0:
        raise RuntimeError(f"No saved {ROOM} version found for variant {variant['name']}")

    registry = ModelRegistry(str(BACKEND))
    models_dir = registry.get_models_dir(candidate_namespace)
    versions_before = _current_versions(models_dir)
    rollback_ok = registry.rollback_to_version(candidate_namespace, ROOM, saved_version)
    versions_after = _current_versions(models_dir)
    load_summary = {
        "candidate_namespace": str(candidate_namespace),
        "room": ROOM,
        "saved_version": saved_version,
        "rollback_ok": bool(rollback_ok),
        "versions_before": versions_before,
        "versions_after": versions_after,
    }
    load_sanity_path.write_text(json.dumps(load_summary, indent=2), encoding="utf-8")

    replay_summary = _evaluate_bedroom_replay(
        elder_id=candidate_namespace,
        eval_file=eval_file,
        output_dir=dec17_dir,
    )
    replay_room = replay_summary["room_metrics"][ROOM]
    decision_trace = trained_room.get("decision_trace") or {}
    two_stage_core = trained_room.get("two_stage_core") or {}
    review_surface = _extract_variant_review_surface(trained_room)
    return {
        "status": "ok",
        "variant": str(variant["name"]),
        "mode": str(variant["mode"]),
        "candidate_namespace": str(candidate_namespace),
        "included_dates": list(variant["included_dates"]),
        "source_files": list(variant["source_files"]),
        "train_metrics_path": str(train_metrics_path),
        "load_sanity_path": str(load_sanity_path),
        "dec17_summary_path": str(dec17_dir / "comparison" / "summary.json"),
        "saved_version": saved_version,
        "gate_pass": bool(trained_room.get("gate_pass", False)),
        "training_strategy": trained_room.get("training_strategy"),
        "holdout_macro_f1": trained_room.get("macro_f1"),
        "holdout_bedroom_normal_use_recall": _extract_holdout_bedroom_normal_use_recall(trained_room),
        "source_fingerprint": ((trained_room.get("source_lineage") or {}).get("source_fingerprint")),
        "two_stage_training_enabled": bool(two_stage_core.get("enabled", False)),
        "two_stage_runtime_use_two_stage": two_stage_core.get("runtime_use_two_stage"),
        "two_stage_gate_source": two_stage_core.get("gate_source"),
        "decision_trace": to_jsonable(decision_trace),
        "promotion_risk_flags": review_surface["promotion_risk_flags"],
        "gate_watch_reasons": review_surface["gate_watch_reasons"],
        "gate_reasons": review_surface["gate_reasons"],
        "grouped_date_stability": review_surface["grouped_date_stability"],
        "promotion_time_drift_summary": review_surface["promotion_time_drift_summary"],
        "pre_sampling_counts": to_jsonable(trained_room.get("train_class_support_pre_sampling")),
        "post_sampling_counts": to_jsonable(trained_room.get("train_class_support_post_minority_sampling")),
        "dec17_final_macro_f1": replay_room.get("final_macro_f1"),
        "dec17_raw_top1_macro_f1": replay_room.get("raw_top1_macro_f1"),
        "dec17_predicted_class_share": to_jsonable(replay_room.get("predicted_class_share")),
        "dec17_critical_error_families": to_jsonable(replay_room.get("critical_error_families")),
    }


def _planned_variant_entry(
    *,
    variant: Mapping[str, Any],
    output_dir: Path,
    candidate_namespace: str,
) -> dict[str, Any]:
    variant_dir = output_dir / _variant_slug(str(variant["name"]))
    return {
        "status": "planned",
        "variant": str(variant["name"]),
        "mode": str(variant["mode"]),
        "candidate_namespace": str(candidate_namespace),
        "included_dates": list(variant["included_dates"]),
        "source_files": list(variant["source_files"]),
        "expected_artifacts": {
            "train_metrics_path": str(variant_dir / "train_metrics.json"),
            "load_sanity_path": str(variant_dir / "load_sanity.json"),
            "dec17_summary_path": str(variant_dir / "dec17_replay" / "comparison" / "summary.json"),
        },
    }


def _extract_holdout_bedroom_normal_use_recall(trained_room: Mapping[str, Any]) -> float | None:
    checkpoint_selection = trained_room.get("checkpoint_selection") or {}
    best_summary = checkpoint_selection.get("best_summary") or {}
    best_summary_recall = ((best_summary.get("per_label_recall") or {}).get("bedroom_normal_use"))
    if best_summary_recall is not None:
        return float(best_summary_recall)

    report = trained_room.get("classification_report") or {}
    if isinstance(report, Mapping):
        recall = ((report.get("bedroom_normal_use") or {}).get("recall"))
        if recall is not None:
            return float(recall)

    per_label_recall = trained_room.get("per_label_recall")
    if isinstance(per_label_recall, Mapping):
        recall = per_label_recall.get("bedroom_normal_use")
        if recall is not None:
            return float(recall)
    return None


def _extract_variant_review_surface(trained_room: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "promotion_risk_flags": to_jsonable(list(trained_room.get("promotion_risk_flags") or [])),
        "gate_watch_reasons": to_jsonable(list(trained_room.get("gate_watch_reasons") or [])),
        "gate_reasons": to_jsonable(list(trained_room.get("gate_reasons") or [])),
        "grouped_date_stability": to_jsonable(trained_room.get("grouped_date_stability")),
        "promotion_time_drift_summary": to_jsonable(trained_room.get("promotion_time_drift_summary")),
    }


def _summarize_manifest_status(
    variant_results: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool,
) -> tuple[str, str, str, dict[str, int], dict[str, int]]:
    if dry_run:
        return "dry_run", "dry_run", "not_run", {}, {}

    execution_counts: dict[str, int] = {}
    gate_counts: dict[str, int] = {}
    execution_failed = False

    for result in variant_results:
        execution_status = str(result.get("status", "error"))
        execution_counts[execution_status] = execution_counts.get(execution_status, 0) + 1
        if execution_status != "ok":
            execution_failed = True
            continue

        gate_pass = result.get("gate_pass")
        if gate_pass is True:
            gate_key = "pass"
        elif gate_pass is False:
            gate_key = "fail"
        else:
            gate_key = "unknown"
        gate_counts[gate_key] = gate_counts.get(gate_key, 0) + 1

    overall_execution_status = "error" if execution_failed else "ok"
    if execution_failed:
        overall_gate_status = "incomplete"
        overall_status = "execution_error"
    elif gate_counts and gate_counts.get("fail", 0) > 0:
        overall_gate_status = "fail"
        overall_status = "gate_fail"
    elif gate_counts and gate_counts.get("unknown", 0) > 0:
        overall_gate_status = "unknown"
        overall_status = "gate_unknown"
    else:
        overall_gate_status = "pass"
        overall_status = "gate_pass"

    return (
        overall_status,
        overall_execution_status,
        overall_gate_status,
        dict(sorted(execution_counts.items())),
        dict(sorted(gate_counts.items())),
    )


def run_matrix(
    *,
    output_dir: Path,
    dry_run: bool,
    source_namespace: str = SOURCE_NAMESPACE,
    source_files_by_date: Mapping[str, str] | None = None,
    eval_file: str | None = None,
    candidate_prefix: str | None = None,
    variant_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = build_variant_specs(source_files_by_date=source_files_by_date)
    if variant_names:
        allowed = {str(name) for name in variant_names}
        variants = [variant for variant in variants if str(variant["name"]) in allowed]
    files_by_date = dict(source_files_by_date or DEFAULT_SOURCE_FILES_BY_DATE)
    eval_source_file = str(eval_file or files_by_date[EVAL_DATE])
    prefix = str(candidate_prefix or f"{source_namespace}_candidate_bedroom_rootmatrix_{_sanitize_token(output_dir.name)}")
    models_root = BACKEND / "models"

    manifest: dict[str, Any] = {
        "schema_version": "beta6.bedroom_root_cause_matrix.v1",
        "status": "dry_run" if dry_run else "running",
        "execution_status": "dry_run" if dry_run else "running",
        "gate_status": "not_run" if dry_run else "pending",
        "dry_run": bool(dry_run),
        "room": ROOM,
        "source_namespace": str(source_namespace),
        "candidate_prefix": prefix,
        "execution_notes": [
            (
                "Current Bedroom factorized-primary training still fits two-stage core submodels; "
                "this runner records actual saved/runtime posture from train metrics instead of "
                "claiming env-based training overrides."
            )
        ],
        "eval_date": EVAL_DATE,
        "eval_source_file": eval_source_file,
        "anchor_dates": list(ANCHOR_DATES),
        "added_dates": list(ADDED_DATES),
        "selected_variants": [str(variant["name"]) for variant in variants],
        "variant_execution_counts": {},
        "variant_gate_counts": {},
        "variants": {},
    }

    results: list[dict[str, Any]] = []
    for index, variant in enumerate(variants, start=1):
        candidate_namespace = f"{prefix}_{index:02d}_{_variant_slug(str(variant['name']))}"
        try:
            if dry_run:
                result = _planned_variant_entry(
                    variant=variant,
                    output_dir=output_dir,
                    candidate_namespace=candidate_namespace,
                )
            else:
                result = _run_variant(
                    variant=variant,
                    output_dir=output_dir,
                    models_root=models_root,
                    source_namespace=source_namespace,
                    candidate_namespace=candidate_namespace,
                    eval_file=eval_source_file,
                )
        except Exception as exc:
            result = {
                "status": "error",
                "variant": str(variant["name"]),
                "mode": str(variant["mode"]),
                "candidate_namespace": str(candidate_namespace),
                "included_dates": list(variant["included_dates"]),
                "source_files": list(variant["source_files"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        manifest["variants"][str(variant["name"])] = to_jsonable(result)
        results.append(result)

    (
        manifest["status"],
        manifest["execution_status"],
        manifest["gate_status"],
        manifest["variant_execution_counts"],
        manifest["variant_gate_counts"],
    ) = _summarize_manifest_status(results, dry_run=dry_run)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bedroom-first date-ablation matrix runner")
    parser.add_argument("--output-dir", required=True, help="Directory for manifest and per-variant artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest only; do not retrain")
    parser.add_argument("--source-namespace", default=SOURCE_NAMESPACE)
    parser.add_argument("--candidate-prefix", default=None)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--variant-name", action="append", default=None, help="Limit execution to named variants")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = run_matrix(
        output_dir=Path(args.output_dir),
        dry_run=bool(args.dry_run),
        source_namespace=str(args.source_namespace),
        eval_file=args.eval_file,
        candidate_prefix=args.candidate_prefix,
        variant_names=args.variant_name,
    )
    manifest_path = Path(args.output_dir).expanduser().resolve() / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Wrote matrix manifest: {manifest_path}")
    return 0 if str(manifest.get("execution_status")) in {"ok", "dry_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
