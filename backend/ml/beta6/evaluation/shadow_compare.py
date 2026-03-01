"""Phase 6 shadow-mode divergence diagnostics and signed artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sign_payload(payload: Mapping[str, Any], *, signing_key: str) -> str:
    body = _canonical(payload).encode("utf-8")
    digest = hashlib.sha256(signing_key.encode("utf-8") + b"|" + body).hexdigest()
    return f"sha256:{digest}"


_REASON_TEXT = {
    "fail_runtime_eval_parity": "Runtime behavior diverged from evaluation contract.",
    "fail_timeline_mae": "Timeline duration error exceeded the allowed threshold.",
    "fail_timeline_fragmentation": "Timeline fragmentation exceeded the allowed threshold.",
    "fail_timeline_metrics_missing": "Timeline metrics were missing for this room.",
    "fail_uncertainty_low_confidence": "Uncertainty policy blocked this room due to low confidence.",
    "fail_uncertainty_unknown": "Uncertainty policy blocked this room due to unknown behavior.",
    "fail_uncertainty_outside_sensed_space": "Sensor coverage suggests outside-sensed-space behavior.",
    "fail_leakage_resident": "Resident leakage guard detected an invalid split.",
    "fail_leakage_time": "Temporal leakage guard detected an invalid split.",
    "fail_leakage_window": "Window-overlap leakage guard detected an invalid split.",
    "fail_data_viability": "Data viability checks failed for this room.",
    "fail_gate_policy": "Policy gate blocked this room.",
}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _describe_divergence(row: Mapping[str, Any]) -> tuple[str, bool]:
    reason_code = str(row.get("beta6_reason_code") or "").strip()
    reason = _REASON_TEXT.get(reason_code)
    if reason:
        return reason, True
    legacy_reasons = row.get("legacy_gate_reasons")
    if isinstance(legacy_reasons, Sequence):
        for token in legacy_reasons:
            txt = str(token or "").strip()
            if txt:
                return (
                    "Legacy and Beta 6 gate outcomes diverged; legacy reason was used for triage context.",
                    True,
                )
    return "Divergence requires investigation (no mapped reason code).", False


def build_shadow_compare_report(
    *,
    run_id: str,
    elder_id: str,
    room_rows: Sequence[Mapping[str, Any]],
    unexplained_divergence_rate_max: float = 0.05,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    divergence_count = 0
    unexplained_count = 0

    for raw in room_rows:
        room = str(raw.get("room") or "").strip().lower()
        if not room:
            continue
        legacy_pass = _to_bool(raw.get("legacy_gate_pass", False))
        beta6_pass = _to_bool(raw.get("beta6_gate_pass", False))
        divergent = bool(legacy_pass != beta6_pass)
        reason_text, explained = _describe_divergence(raw)
        if divergent:
            divergence_count += 1
            if not explained:
                unexplained_count += 1

        row = {
            "room": room,
            "legacy_gate_pass": legacy_pass,
            "beta6_gate_pass": beta6_pass,
            "divergent": divergent,
            "beta6_reason_code": str(raw.get("beta6_reason_code") or ""),
            "reason_text": reason_text,
            "explained": bool(explained),
            "technical_trace": {
                "legacy_gate_reasons": list(raw.get("legacy_gate_reasons") or []),
                "beta6_details": dict(raw.get("beta6_details") or {}),
            },
        }
        rows.append(row)

    total = len(rows)
    divergence_rate = (float(divergence_count) / float(total)) if total > 0 else 0.0
    unexplained_rate = (float(unexplained_count) / float(total)) if total > 0 else 0.0

    if total == 0:
        status = "insufficient_data"
    elif unexplained_rate > float(unexplained_divergence_rate_max):
        status = "critical"
    elif divergence_count > 0:
        status = "watch"
    else:
        status = "ok"

    badges = [
        {
            "room": row["room"],
            "severity": "critical" if not row["explained"] else "watch",
            "reason_text": row["reason_text"],
            "technical_anchor": f"shadow_trace_{row['room']}",
            "technical_trace": row["technical_trace"],
        }
        for row in rows
        if bool(row.get("divergent", False))
    ]

    return {
        "version": "beta6_shadow_compare_v1",
        "generated_at": _utc_now(),
        "run_id": str(run_id),
        "elder_id": str(elder_id),
        "summary": {
            "status": status,
            "total_rooms": int(total),
            "divergence_count": int(divergence_count),
            "unexplained_divergence_count": int(unexplained_count),
            "divergence_rate": float(divergence_rate),
            "unexplained_divergence_rate": float(unexplained_rate),
            "unexplained_divergence_rate_max": float(unexplained_divergence_rate_max),
        },
        "badges": badges,
        "room_rows": rows,
        "metadata": dict(metadata or {}),
    }


def create_signed_shadow_compare_report(
    *,
    run_id: str,
    elder_id: str,
    room_rows: Sequence[Mapping[str, Any]],
    signing_key: str,
    output_path: str | Path | None = None,
    unexplained_divergence_rate_max: float = 0.05,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = build_shadow_compare_report(
        run_id=run_id,
        elder_id=elder_id,
        room_rows=room_rows,
        unexplained_divergence_rate_max=unexplained_divergence_rate_max,
        metadata=metadata,
    )
    signature = _sign_payload(payload, signing_key=signing_key)
    artifact = {**payload, "signature": signature}
    if output_path is not None:
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def verify_shadow_compare_signature(
    artifact: Mapping[str, Any],
    *,
    signing_key: str,
) -> bool:
    if "signature" not in artifact:
        return False
    payload = {k: v for k, v in artifact.items() if k != "signature"}
    return str(artifact.get("signature", "")) == _sign_payload(payload, signing_key=signing_key)


__all__ = [
    "build_shadow_compare_report",
    "create_signed_shadow_compare_report",
    "verify_shadow_compare_signature",
]
