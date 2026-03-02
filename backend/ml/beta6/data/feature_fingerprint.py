"""Deterministic hashing helpers for Beta 6 data artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def hash_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def hash_json_payload(payload: Mapping[str, Any]) -> str:
    return hash_bytes(_canonical_json(payload).encode("utf-8"))


__all__ = [
    "hash_bytes",
    "hash_file",
    "hash_json_payload",
]
