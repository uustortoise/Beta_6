from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

_ELDER_ID_CODE_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def _normalize_id_token(value: str) -> str:
    return str(value or "").strip().lower()


def _load_canonical_alias_map() -> dict[str, str]:
    """
    Parse canonical elder-id alias map from env.

    Format:
    ELDER_ID_CANONICAL_MAP=alias1=canonical1,alias2:canonical2
    """
    raw = str(os.getenv("ELDER_ID_CANONICAL_MAP", "") or "").strip()
    if not raw:
        return {}

    out: dict[str, str] = {}
    for token in raw.split(","):
        item = str(token).strip()
        if not item:
            continue
        if "=" in item:
            alias_raw, canonical_raw = item.split("=", 1)
        elif ":" in item:
            alias_raw, canonical_raw = item.split(":", 1)
        else:
            continue
        alias = _normalize_id_token(alias_raw)
        canonical = str(canonical_raw or "").strip()
        if not alias or not canonical:
            continue
        out[alias] = canonical
    return out


def apply_canonical_alias_map(elder_id: str) -> str:
    """
    Apply explicit canonical alias mapping if configured.
    """
    txt = str(elder_id or "").strip()
    if not txt:
        return ""
    alias_map = _load_canonical_alias_map()
    if not alias_map:
        return txt
    return str(alias_map.get(_normalize_id_token(txt), txt))


def parse_elder_id_from_filename(filename: str) -> str:
    stem = Path(str(filename or "")).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return "resident_01"


def split_elder_id_code_and_name(elder_id: str) -> tuple[str, str]:
    txt = str(elder_id or "").strip()
    if "_" not in txt:
        return "", ""
    code, name = txt.split("_", 1)
    return code.strip(), name.strip().lower()


def elder_id_lineage_matches(expected_elder_id: str, candidate_elder_id: str) -> bool:
    """
    Determine whether two elder IDs belong to the same resident lineage.

    Exact match is preferred; a narrow numeric-suffix drift fallback is allowed,
    for example HK001_jessica vs HK0011_jessica.
    """
    expected = str(expected_elder_id or "").strip()
    candidate = str(candidate_elder_id or "").strip()
    if not expected or not candidate:
        return False
    if expected == candidate:
        return True

    expected_code, expected_name = split_elder_id_code_and_name(expected)
    candidate_code, candidate_name = split_elder_id_code_and_name(candidate)
    if not expected_code or not candidate_code or expected_name != candidate_name:
        return False

    em = _ELDER_ID_CODE_RE.match(expected_code)
    cm = _ELDER_ID_CODE_RE.match(candidate_code)
    if em is None or cm is None:
        return False

    expected_prefix, expected_digits = em.group(1).upper(), em.group(2)
    candidate_prefix, candidate_digits = cm.group(1).upper(), cm.group(2)
    if expected_prefix != candidate_prefix:
        return False
    if expected_digits == candidate_digits:
        return True
    if expected_digits.lstrip("0") == candidate_digits.lstrip("0"):
        return True

    shorter, longer = (
        (expected_digits, candidate_digits)
        if len(expected_digits) <= len(candidate_digits)
        else (candidate_digits, expected_digits)
    )
    return len(longer) - len(shorter) <= 1 and longer.startswith(shorter)


def _canonical_rank(elder_id: str) -> tuple[int, int, str]:
    """
    Rank elder IDs for canonical selection.

    Prefer the shortest numeric token first. This keeps baseline canonical IDs
    stable when typo drift introduces a duplicated suffix digit.
    """
    code, _ = split_elder_id_code_and_name(elder_id)
    digits = ""
    m = _ELDER_ID_CODE_RE.match(code)
    if m is not None:
        digits = str(m.group(2))
    return (len(digits), len(elder_id), elder_id)


def choose_canonical_elder_id(elder_ids: Iterable[str]) -> str:
    cleaned = [str(item).strip() for item in elder_ids if str(item).strip()]
    if not cleaned:
        return ""
    canonical = sorted(cleaned, key=_canonical_rank)[0]
    return apply_canonical_alias_map(canonical)
