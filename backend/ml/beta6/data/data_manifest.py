"""Deterministic Beta 6 pretraining corpus manifest builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .feature_fingerprint import hash_file, hash_json_payload


MANIFEST_VERSION = "beta6_pretrain_manifest_v2"
AUTO_APPROVED_STATUS = "auto_approved"
QUARANTINED_STATUS = "quarantined"

_TABULAR_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls"}
_USER_TAG_COLUMNS = ("elder_id", "resident_id", "user_id", "subject_id")
_DATE_COLUMNS = ("date", "day_date", "event_date")
_TIMESTAMP_COLUMNS = ("timestamp", "start_ts", "start_time", "datetime")
_ROOM_COLUMNS = ("room", "room_name")
_ACTIVITY_COLUMNS = ("activity", "label", "adl")
_DATE_TEXT_RE = re.compile(r"(?P<year>\d{4})[-_](?P<month>\d{2})[-_](?P<day>\d{2})")
_DATE_MON_RE = re.compile(
    r"(?P<day>\d{1,2})(?P<month>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?P<year>\d{4})",
    re.IGNORECASE,
)
_MONTH_LOOKUP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_USER_PATH_RE = re.compile(r"\b([A-Z]{2}\d{4}(?:_[A-Za-z0-9]+)?)\b")


@dataclass(frozen=True)
class CorpusManifestPolicy:
    include_extensions: tuple[str, ...] = (".csv", ".parquet", ".npy")
    max_missing_ratio: float = 0.4
    min_rows: int = 8
    min_features: int = 2
    require_user_date_tags_for_labeled_sources: bool = True


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


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"unsupported feature file extension: {suffix}")


def _frame_to_matrix(frame: pd.DataFrame) -> np.ndarray:
    metadata_cols = {
        name
        for name in frame.columns
        if str(name).strip().lower()
        in {
            *_USER_TAG_COLUMNS,
            *_DATE_COLUMNS,
            *_TIMESTAMP_COLUMNS,
            *_ROOM_COLUMNS,
            *_ACTIVITY_COLUMNS,
        }
    }
    feature_frame = frame.drop(columns=list(metadata_cols), errors="ignore")
    numeric = feature_frame.apply(pd.to_numeric, errors="coerce")
    matrix = numeric.to_numpy(dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim == 0:
        matrix = np.zeros((int(len(frame.index)), 0), dtype=np.float32)
    return matrix


def load_feature_matrix(path: str | Path) -> np.ndarray:
    p = Path(path).resolve()
    suffix = p.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(p, allow_pickle=False)
    elif suffix in _TABULAR_EXTENSIONS:
        matrix = _frame_to_matrix(_read_table(p))
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


def _normalize_token(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if pd.isna(value):
        return None
    token = str(value).strip()
    if not token or token.lower() in {"nan", "nat", "none"}:
        return None
    return token


def _normalize_date_token(value: Any) -> str | None:
    token = _normalize_token(value)
    if token is None:
        return None
    parsed = pd.to_datetime(token, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lowered = {str(col).strip().lower(): str(col) for col in frame.columns}
    for candidate in candidates:
        match = lowered.get(candidate)
        if match is not None:
            return match
    return None


def _extract_path_user_tags(path: Path) -> list[str]:
    tokens = [_normalize_token(match.group(1)) for match in _USER_PATH_RE.finditer(path.stem)]
    return sorted({token for token in tokens if token})


def _extract_path_date_tags(path: Path) -> list[str]:
    stem = path.stem.lower()
    tags: set[str] = set()
    for match in _DATE_TEXT_RE.finditer(stem):
        tags.add(f"{match.group('year')}-{match.group('month')}-{match.group('day')}")
    for match in _DATE_MON_RE.finditer(stem):
        month = _MONTH_LOOKUP[match.group("month").lower()]
        tags.add(f"{int(match.group('year')):04d}-{month:02d}-{int(match.group('day')):02d}")
    return sorted(tags)


def _extract_user_tags(frame: pd.DataFrame | None, path: Path) -> list[str]:
    tags: set[str] = set(_extract_path_user_tags(path))
    if frame is not None:
        for column in _USER_TAG_COLUMNS:
            actual = _find_column(frame, (column,))
            if actual is None:
                continue
            for value in frame[actual].tolist():
                token = _normalize_token(value)
                if token:
                    tags.add(token)
    return sorted(tags)


def _extract_row_dates(frame: pd.DataFrame) -> list[str | None]:
    date_col = _find_column(frame, _DATE_COLUMNS)
    if date_col is not None:
        return [_normalize_date_token(value) for value in frame[date_col].tolist()]
    ts_col = _find_column(frame, _TIMESTAMP_COLUMNS)
    if ts_col is not None:
        return [_normalize_date_token(value) for value in frame[ts_col].tolist()]
    return [None] * int(len(frame.index))


def _extract_date_tags(frame: pd.DataFrame | None, path: Path) -> list[str]:
    tags: set[str] = set(_extract_path_date_tags(path))
    if frame is not None:
        for value in _extract_row_dates(frame):
            if value:
                tags.add(value)
    return sorted(tags)


def _label_summary(frame: pd.DataFrame | None) -> dict[str, list[dict[str, Any]]]:
    if frame is None or frame.empty:
        return {"per_room_per_date_label_counts": []}
    room_col = _find_column(frame, _ROOM_COLUMNS)
    activity_col = _find_column(frame, _ACTIVITY_COLUMNS)
    if room_col is None or activity_col is None:
        return {"per_room_per_date_label_counts": []}

    row_dates = _extract_row_dates(frame)
    counts: dict[tuple[str, str, str], int] = {}
    for idx, raw_date in enumerate(row_dates):
        if not raw_date:
            continue
        room = _normalize_token(frame.iloc[idx][room_col])
        activity = _normalize_token(frame.iloc[idx][activity_col])
        if not room or not activity:
            continue
        key = (room.lower(), raw_date, activity.lower())
        counts[key] = counts.get(key, 0) + 1

    return {
        "per_room_per_date_label_counts": [
            {
                "room": room,
                "date": date,
                "activity": activity,
                "count": int(count),
            }
            for room, date, activity, count in sorted(
                ((room, date, activity, count) for (room, date, activity), count in counts.items()),
                key=lambda item: (item[1], item[0], item[2]),
            )
        ]
    }


def _has_labeled_axes(frame: pd.DataFrame | None) -> bool:
    if frame is None or frame.empty:
        return False
    return _find_column(frame, _ROOM_COLUMNS) is not None and _find_column(frame, _ACTIVITY_COLUMNS) is not None


def _build_entry(
    *,
    candidate: Path,
    content_hash: str,
    row_count: int,
    feature_count: int,
    missing_ratio: float,
    user_tags: list[str],
    date_tags: list[str],
    label_summary: Mapping[str, Any],
    status: str,
    quarantine_reasons: Sequence[str],
) -> Dict[str, Any]:
    entry = {
        "path": str(candidate),
        "extension": candidate.suffix.lower(),
        "content_hash": content_hash,
        "row_count": int(row_count),
        "feature_count": int(feature_count),
        "missing_ratio": float(round(float(missing_ratio), 8)),
        "status": str(status),
        "source_tags": {
            "user_tags": list(user_tags),
            "date_tags": list(date_tags),
        },
        "label_summary": {
            "per_room_per_date_label_counts": list(
                label_summary.get("per_room_per_date_label_counts", [])
            )
        },
    }
    if quarantine_reasons:
        normalized_reasons = sorted({str(reason) for reason in quarantine_reasons if str(reason).strip()})
        entry["quarantine_reasons"] = normalized_reasons
        entry["reason_code"] = normalized_reasons[0]
    return entry


def _aggregate_manifest_summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    user_tags: set[str] = set()
    date_tags: set[str] = set()
    label_counts: dict[tuple[str, str, str], int] = {}

    for entry in entries:
        source_tags = entry.get("source_tags") if isinstance(entry, Mapping) else {}
        if isinstance(source_tags, Mapping):
            for user_tag in source_tags.get("user_tags", []):
                token = _normalize_token(user_tag)
                if token:
                    user_tags.add(token)
            for date_tag in source_tags.get("date_tags", []):
                token = _normalize_token(date_tag)
                if token:
                    date_tags.add(token)
        label_summary = entry.get("label_summary") if isinstance(entry, Mapping) else {}
        if not isinstance(label_summary, Mapping):
            continue
        for row in label_summary.get("per_room_per_date_label_counts", []):
            if not isinstance(row, Mapping):
                continue
            room = _normalize_token(row.get("room"))
            date = _normalize_token(row.get("date"))
            activity = _normalize_token(row.get("activity"))
            count = int(row.get("count", 0) or 0)
            if not room or not date or not activity or count <= 0:
                continue
            key = (room.lower(), date, activity.lower())
            label_counts[key] = label_counts.get(key, 0) + count

    return {
        "user_tags": sorted(user_tags),
        "date_tags": sorted(date_tags),
        "per_room_per_date_label_counts": [
            {
                "room": room,
                "date": date,
                "activity": activity,
                "count": int(count),
            }
            for room, date, activity, count in sorted(
                ((room, date, activity, count) for (room, date, activity), count in label_counts.items()),
                key=lambda item: (item[1], item[0], item[2]),
            )
        ],
    }


def build_pretrain_corpus_manifest(
    *,
    corpus_roots: Sequence[str | Path],
    policy: CorpusManifestPolicy,
) -> Dict[str, Any]:
    include_extensions = _normalize_extensions(policy.include_extensions)
    include_set = set(include_extensions)

    entries: list[Dict[str, Any]] = []
    duplicates: list[Dict[str, Any]] = []
    quarantine: list[Dict[str, Any]] = []

    seen_hashes: dict[str, str] = {}
    files_scanned = 0

    for candidate in _iter_candidates(corpus_roots, include_extensions=include_set):
        files_scanned += 1
        path_str = str(candidate)

        try:
            content_hash = hash_file(candidate)
        except Exception as exc:  # pragma: no cover - defensive fail-closed path
            quarantine.append(
                {
                    "path": path_str,
                    "reason_code": "manifest_read_error",
                    "quarantine_reasons": ["manifest_read_error"],
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

        raw_frame: pd.DataFrame | None = None
        try:
            matrix = load_feature_matrix(candidate)
            if candidate.suffix.lower() in _TABULAR_EXTENSIONS:
                raw_frame = _read_table(candidate)
        except Exception as exc:
            quarantine.append(
                {
                    "path": path_str,
                    "reason_code": "manifest_read_error",
                    "quarantine_reasons": ["manifest_read_error"],
                    "detail": str(exc),
                }
            )
            continue

        row_count = int(raw_frame.shape[0] if raw_frame is not None else matrix.shape[0])
        feature_count = int(matrix.shape[1])
        missing_ratio = _missing_ratio(matrix)
        user_tags = _extract_user_tags(raw_frame, candidate)
        date_tags = _extract_date_tags(raw_frame, candidate)
        label_summary = _label_summary(raw_frame)

        quarantine_reasons: list[str] = []
        if row_count < int(policy.min_rows):
            quarantine_reasons.append("manifest_row_count_violation")
        if feature_count < int(policy.min_features):
            quarantine_reasons.append("manifest_feature_count_violation")
        if missing_ratio > float(policy.max_missing_ratio):
            quarantine_reasons.append("manifest_missing_ratio_violation")
        if bool(policy.require_user_date_tags_for_labeled_sources) and _has_labeled_axes(raw_frame):
            if not user_tags:
                quarantine_reasons.append("manifest_missing_user_tag")
            if not date_tags:
                quarantine_reasons.append("manifest_missing_date_tag")

        if quarantine_reasons:
            quarantine.append(
                _build_entry(
                    candidate=candidate,
                    content_hash=content_hash,
                    row_count=row_count,
                    feature_count=feature_count,
                    missing_ratio=missing_ratio,
                    user_tags=user_tags,
                    date_tags=date_tags,
                    label_summary=label_summary,
                    status=QUARANTINED_STATUS,
                    quarantine_reasons=quarantine_reasons,
                )
            )
            continue

        entries.append(
            _build_entry(
                candidate=candidate,
                content_hash=content_hash,
                row_count=row_count,
                feature_count=feature_count,
                missing_ratio=missing_ratio,
                user_tags=user_tags,
                date_tags=date_tags,
                label_summary=label_summary,
                status=AUTO_APPROVED_STATUS,
                quarantine_reasons=(),
            )
        )

    entries = sorted(entries, key=lambda item: (item["content_hash"], item["path"]))
    quarantine = sorted(quarantine, key=lambda item: str(item.get("path") or ""))
    summary = _aggregate_manifest_summary(entries)

    normalized_entries = [
        {
            "path": entry["path"],
            "content_hash": entry["content_hash"],
            "row_count": entry["row_count"],
            "feature_count": entry["feature_count"],
            "missing_ratio": entry["missing_ratio"],
            "source_tags": entry["source_tags"],
        }
        for entry in entries
    ]

    fingerprint_value = hash_json_payload({"entries": normalized_entries})
    gate_approved = len(entries) > 0
    blocking_reasons = (
        []
        if gate_approved
        else sorted(
            {
                str(reason)
                for item in quarantine
                for reason in item.get("quarantine_reasons", [])
                if str(reason).strip()
            }
        )
    )
    if not gate_approved and not blocking_reasons:
        blocking_reasons = ["manifest_empty_corpus"]

    violations = list(quarantine)
    if not gate_approved:
        violations.append(
            {
                "path": None,
                "reason_code": "manifest_empty_corpus",
                "quarantine_reasons": list(blocking_reasons),
                "detail": "No auto-approved files met corpus policy",
            }
        )

    p0_violations = 0 if gate_approved else int(max(1, len(violations)))

    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": _utc_now(),
        "policy": {
            "include_extensions": list(include_extensions),
            "max_missing_ratio": float(_safe_float(policy.max_missing_ratio, default=0.4)),
            "min_rows": int(_safe_int(policy.min_rows, default=8)),
            "min_features": int(_safe_int(policy.min_features, default=2)),
            "require_user_date_tags_for_labeled_sources": bool(
                policy.require_user_date_tags_for_labeled_sources
            ),
        },
        "corpus_roots": [str(Path(root).resolve()) for root in corpus_roots],
        "entries": entries,
        "duplicates": duplicates,
        "quarantine": quarantine,
        "violations": violations,
        "summary": summary,
        "gate": {
            "approved": bool(gate_approved),
            "blocking_reasons": list(blocking_reasons),
        },
        "stats": {
            "files_scanned": int(files_scanned),
            "records_kept": int(len(entries)),
            "auto_approved": int(len(entries)),
            "quarantined": int(len(quarantine)),
            "duplicates_removed": int(len(duplicates)),
            "p0_violations": int(p0_violations),
        },
        "fingerprint": {
            "algorithm": "sha256",
            "value": fingerprint_value,
        },
    }


__all__ = [
    "AUTO_APPROVED_STATUS",
    "MANIFEST_VERSION",
    "QUARANTINED_STATUS",
    "CorpusManifestPolicy",
    "build_pretrain_corpus_manifest",
    "load_feature_matrix",
    "load_manifest",
]
