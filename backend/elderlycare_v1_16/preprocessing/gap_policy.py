"""
Shared gap-filling policy helpers used across ingestion and platform paths.
"""

from __future__ import annotations


UNBOUNDED_FFILL_TOKENS = {
    "none",
    "null",
    "unbounded",
    "off",
    "disable",
    "disabled",
    "inf",
    "infinite",
}


def resolve_max_ffill_gap_seconds(raw_value, default_seconds: float = 60.0):
    """
    Resolve MAX_RESAMPLE_FFILL_GAP_SECONDS style value to float seconds or None.

    Accepted unbounded tokens are shared across all call sites.
    """
    if raw_value is None:
        return float(default_seconds)
    txt = str(raw_value).strip()
    if txt == "":
        return float(default_seconds)
    lowered = txt.lower()
    if lowered in UNBOUNDED_FFILL_TOKENS:
        return None
    try:
        value = float(lowered)
    except (TypeError, ValueError):
        return float(default_seconds)
    if value <= 0:
        return None
    return float(value)
