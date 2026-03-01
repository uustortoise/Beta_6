"""Signed rejection artifact builder for Beta 6 dynamic gating."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sign_payload(payload: Mapping[str, Any], *, signing_key: str) -> str:
    body = _canonical(payload).encode("utf-8")
    digest = hashlib.sha256(signing_key.encode("utf-8") + b"|" + body).hexdigest()
    return f"sha256:{digest}"


def create_signed_rejection_artifact(
    *,
    run_id: str,
    elder_id: str,
    reason_code: str,
    room_reports: Sequence[Mapping[str, Any]],
    signing_key: str,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    payload = {
        "version": "v1",
        "generated_at": _utc_now(),
        "run_id": str(run_id),
        "elder_id": str(elder_id),
        "reason_code": str(reason_code),
        "room_reports": [dict(report) for report in room_reports],
    }
    signature = _sign_payload(payload, signing_key=signing_key)
    artifact = {**payload, "signature": signature}
    if output_path is not None:
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def verify_rejection_artifact_signature(artifact: Mapping[str, Any], *, signing_key: str) -> bool:
    if "signature" not in artifact:
        return False
    signature = str(artifact.get("signature", ""))
    payload = {k: v for k, v in artifact.items() if k != "signature"}
    return signature == _sign_payload(payload, signing_key=signing_key)


__all__ = [
    "create_signed_rejection_artifact",
    "verify_rejection_artifact_signature",
]
