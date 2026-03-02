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
        }
        for entry in sorted(entries, key=lambda item: (item["content_hash"], item["path"]))
    ]

    fingerprint_value = hash_json_payload({"entries": normalized_entries})
    p0_violations = len(violations)

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
    "load_feature_matrix",
    "load_manifest",
]
