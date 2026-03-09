"""Deterministic Beta 6 pretraining corpus manifest builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .feature_fingerprint import hash_file, hash_json_payload


MANIFEST_VERSION = "beta6_pretrain_manifest_v1"


@dataclass(frozen=True)
class CorpusManifestPolicy:
    include_extensions: tuple[str, ...] = (".csv", ".parquet", ".npy")
    max_missing_ratio: float = 0.4
    min_rows: int = 8
    min_features: int = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_extensions(values: Sequence[str]) -> tuple[str, ...]:
    normalized = []
    for value in values:
        token = str(value).strip().lower()
        if not token:
            continue
        if not token.startswith("."):
            token = f".{token}"
        normalized.append(token)
    return tuple(sorted(set(normalized)))


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _iter_candidates(corpus_roots: Iterable[str | Path], *, include_extensions: set[str]) -> list[Path]:
    candidates: list[Path] = []
    for root in corpus_roots:
        path = Path(root).resolve()
        if path.is_file():
            if path.suffix.lower() in include_extensions:
                candidates.append(path)
            continue
        if not path.exists() or not path.is_dir():
            continue
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in include_extensions:
                candidates.append(child.resolve())
    return sorted(candidates, key=lambda item: str(item))


def load_manifest(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest JSON must decode to object")
    return payload


def _load_sidecar_metadata(path: Path) -> Dict[str, Any]:
    sidecars = [
        path.with_suffix(path.suffix + ".meta.json"),
        path.with_suffix(".meta.json"),
    ]
    for sidecar in sidecars:
        if not sidecar.exists():
            continue
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _infer_views(path: Path, metadata: Mapping[str, Any]) -> list[str]:
    explicit = metadata.get("views")
    if isinstance(explicit, list):
        values = [str(item).strip().lower() for item in explicit if str(item).strip()]
        return sorted(set(values))

    token = path.stem.lower()
    views = ["unlabeled_pretrain"]
    if "shadow" in token:
        views.append("shadow_cohort")
    if any(marker in token for marker in ("golden", "labeled", "labelled", "trusted")):
        views.append("labeled_high_trust_finetune_eval")
    return sorted(set(views))


def _infer_resident_id(path: Path, metadata: Mapping[str, Any]) -> str | None:
    explicit = str(metadata.get("resident_id") or "").strip()
    if explicit:
        return explicit
    parent = str(path.parent.name).strip()
    return parent or None


def _normalize_context_completeness(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    context = metadata.get("resident_home_context")
    if not isinstance(context, Mapping):
        context = {}
    status = str(context.get("status") or metadata.get("context_status") or "missing_required_context").strip()
    missing_fields = context.get("missing_fields") or metadata.get("context_missing_fields") or []
    if not isinstance(missing_fields, list):
        missing_fields = [str(missing_fields)]
    return {
        "status": status,
        "missing_fields": [str(item) for item in missing_fields if str(item).strip()],
    }


def _normalize_label_quality(metadata: Mapping[str, Any], *, views: Sequence[str]) -> Dict[str, Any]:
    raw_quality = metadata.get("label_quality")
    if isinstance(raw_quality, Mapping):
        trust_tier = str(raw_quality.get("trust_tier") or raw_quality.get("status") or "unknown").strip().lower()
        reviewed_fraction = _safe_float(raw_quality.get("reviewed_fraction"), default=0.0)
        source = str(raw_quality.get("source") or "sidecar").strip()
    else:
        trust_tier = "high" if "labeled_high_trust_finetune_eval" in views else "unknown"
        reviewed_fraction = 1.0 if trust_tier == "high" else 0.0
        source = "inferred"
    return {
        "source": source,
        "trust_tier": trust_tier,
        "reviewed_fraction": float(max(min(reviewed_fraction, 1.0), 0.0)),
    }


def _summarize_corpus_views(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for view_name in ("shadow_cohort", "unlabeled_pretrain", "labeled_high_trust_finetune_eval"):
        selected = [entry for entry in entries if view_name in (entry.get("views") or [])]
        resident_ids = {
            str(entry.get("resident_id") or "").strip()
            for entry in selected
            if str(entry.get("resident_id") or "").strip()
        }
        days_covered = [int(entry.get("days_covered") or 0) for entry in selected]
        out[view_name] = {
            "entry_count": len(selected),
            "resident_count": len(resident_ids),
            "resident_ids": sorted(resident_ids),
            "min_days_covered": min(days_covered) if days_covered else 0,
            "max_days_covered": max(days_covered) if days_covered else 0,
        }
    return out


def evaluate_beta62_corpus_contract(
    manifest: Mapping[str, Any],
    *,
    required_residents: int = 20,
    required_days: int = 14,
) -> Dict[str, Any]:
    corpus_views = manifest.get("corpus_views", {}) if isinstance(manifest.get("corpus_views"), Mapping) else {}
    shadow = corpus_views.get("shadow_cohort", {}) if isinstance(corpus_views.get("shadow_cohort"), Mapping) else {}
    labeled = (
        corpus_views.get("labeled_high_trust_finetune_eval", {})
        if isinstance(corpus_views.get("labeled_high_trust_finetune_eval"), Mapping)
        else {}
    )
    context_summary = manifest.get("context_summary", {}) if isinstance(manifest.get("context_summary"), Mapping) else {}
    stats = manifest.get("stats", {}) if isinstance(manifest.get("stats"), Mapping) else {}
    reason_codes: list[str] = []
    if int(shadow.get("resident_count", 0) or 0) < int(required_residents):
        reason_codes.append("shadow_resident_coverage_below_contract")
    if int(shadow.get("min_days_covered", 0) or 0) < int(required_days):
        reason_codes.append("shadow_days_coverage_below_contract")
    if int(labeled.get("min_days_covered", 0) or 0) < int(required_days):
        reason_codes.append("labeled_days_coverage_below_contract")
    if int(context_summary.get("ready_entries", 0) or 0) < int(stats.get("records_kept", 0) or 0):
        reason_codes.append("resident_home_context_incomplete")
    return {
        "pass": not reason_codes,
        "required_residents": int(required_residents),
        "required_days": int(required_days),
        "reason_codes": reason_codes,
    }


def _frame_to_matrix(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    matrix = numeric.to_numpy(dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return matrix


def load_feature_matrix(path: str | Path) -> np.ndarray:
    p = Path(path).resolve()
    suffix = p.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(p, allow_pickle=False)
    elif suffix == ".csv":
        matrix = _frame_to_matrix(pd.read_csv(p))
    elif suffix == ".parquet":
        matrix = _frame_to_matrix(pd.read_parquet(p))
    else:
        raise ValueError(f"unsupported feature file extension: {suffix}")

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"feature matrix must be 2D, got shape={matrix.shape}")
    return matrix


def _missing_ratio(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 1.0
    return float(np.isnan(matrix).sum() / matrix.size)


def build_pretrain_corpus_manifest(
    *,
    corpus_roots: Sequence[str | Path],
    policy: CorpusManifestPolicy,
) -> Dict[str, Any]:
    include_extensions = _normalize_extensions(policy.include_extensions)
    include_set = set(include_extensions)

    entries: list[Dict[str, Any]] = []
    duplicates: list[Dict[str, Any]] = []
    violations: list[Dict[str, Any]] = []

    seen_hashes: dict[str, str] = {}
    files_scanned = 0

    for candidate in _iter_candidates(corpus_roots, include_extensions=include_set):
        files_scanned += 1
        path_str = str(candidate)

        try:
            content_hash = hash_file(candidate)
        except Exception as exc:  # pragma: no cover - defensive fail-closed path
            violations.append(
                {
                    "path": path_str,
                    "reason_code": "manifest_read_error",
                    "detail": str(exc),
                }
            )
            continue

        original = seen_hashes.get(content_hash)
        if original is not None:
            duplicates.append(
                {
                    "path": path_str,
                    "duplicate_of": original,
                    "content_hash": content_hash,
                }
            )
            continue
        seen_hashes[content_hash] = path_str

        try:
            matrix = load_feature_matrix(candidate)
        except Exception as exc:
            violations.append(
                {
                    "path": path_str,
                    "reason_code": "manifest_read_error",
                    "detail": str(exc),
                }
            )
            continue

        row_count = int(matrix.shape[0])
        feature_count = int(matrix.shape[1])
        missing_ratio = _missing_ratio(matrix)
        metadata = _load_sidecar_metadata(candidate)
        views = _infer_views(candidate, metadata)
        context_completeness = _normalize_context_completeness(metadata)
        label_quality = _normalize_label_quality(metadata, views=views)

        if row_count < int(policy.min_rows):
            violations.append(
                {
                    "path": path_str,
                    "reason_code": "manifest_row_count_violation",
                    "row_count": row_count,
                    "min_rows": int(policy.min_rows),
                }
            )
            continue

        if feature_count < int(policy.min_features):
            violations.append(
                {
                    "path": path_str,
                    "reason_code": "manifest_feature_count_violation",
                    "feature_count": feature_count,
                    "min_features": int(policy.min_features),
                }
            )
            continue

        if missing_ratio > float(policy.max_missing_ratio):
            violations.append(
                {
                    "path": path_str,
                    "reason_code": "manifest_missing_ratio_violation",
                    "missing_ratio": float(missing_ratio),
                    "max_missing_ratio": float(policy.max_missing_ratio),
                }
            )
            continue

        entries.append(
            {
                "path": path_str,
                "extension": candidate.suffix.lower(),
                "content_hash": content_hash,
                "row_count": row_count,
                "feature_count": feature_count,
                "missing_ratio": float(round(missing_ratio, 8)),
                "resident_id": _infer_resident_id(candidate, metadata),
                "days_covered": _safe_int(metadata.get("days_covered"), default=0),
                "views": views,
                "resident_home_context": metadata.get("resident_home_context", {}),
                "context_completeness": context_completeness,
                "label_quality": label_quality,
            }
        )

    if not entries:
        violations.append(
            {
                "path": None,
                "reason_code": "manifest_empty_corpus",
                "detail": "No usable files met corpus policy",
            }
        )

    normalized_entries = [
        {
            "path": entry["path"],
            "content_hash": entry["content_hash"],
            "row_count": entry["row_count"],
            "feature_count": entry["feature_count"],
            "missing_ratio": entry["missing_ratio"],
            "resident_id": entry.get("resident_id"),
            "days_covered": entry.get("days_covered"),
            "views": entry.get("views", []),
            "context_status": (entry.get("context_completeness") or {}).get("status"),
            "label_trust_tier": (entry.get("label_quality") or {}).get("trust_tier"),
        }
        for entry in sorted(entries, key=lambda item: (item["content_hash"], item["path"]))
    ]

    fingerprint_value = hash_json_payload({"entries": normalized_entries})
    p0_violations = len(violations)
    corpus_views = _summarize_corpus_views(entries)
    ready_entries = sum(
        1
        for entry in entries
        if str((entry.get("context_completeness") or {}).get("status") or "") == "ready"
    )
    high_trust_entries = sum(
        1
        for entry in entries
        if str((entry.get("label_quality") or {}).get("trust_tier") or "") == "high"
    )

    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": _utc_now(),
        "policy": {
            "include_extensions": list(include_extensions),
            "max_missing_ratio": float(_safe_float(policy.max_missing_ratio, default=0.4)),
            "min_rows": int(_safe_int(policy.min_rows, default=8)),
            "min_features": int(_safe_int(policy.min_features, default=2)),
        },
        "corpus_roots": [str(Path(root).resolve()) for root in corpus_roots],
        "entries": entries,
        "duplicates": duplicates,
        "violations": violations,
        "corpus_views": corpus_views,
        "context_summary": {
            "ready_entries": int(ready_entries),
            "incomplete_entries": int(max(len(entries) - ready_entries, 0)),
        },
        "label_quality_summary": {
            "high_trust_entries": int(high_trust_entries),
            "unknown_trust_entries": int(max(len(entries) - high_trust_entries, 0)),
        },
        "stats": {
            "files_scanned": int(files_scanned),
            "records_kept": int(len(entries)),
            "duplicates_removed": int(len(duplicates)),
            "p0_violations": int(p0_violations),
        },
        "fingerprint": {
            "algorithm": "sha256",
            "value": fingerprint_value,
        },
    }


__all__ = [
    "MANIFEST_VERSION",
    "CorpusManifestPolicy",
    "build_pretrain_corpus_manifest",
    "evaluate_beta62_corpus_contract",
    "load_feature_matrix",
    "load_manifest",
]
