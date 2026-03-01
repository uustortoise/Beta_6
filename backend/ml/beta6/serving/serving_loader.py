"""Beta 6 serving authority + stability certification helpers (Phase 6.3)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ml.policy_presets import load_rollout_ladder_policy
from ml.t80_rollout_manager import RolloutStage, T80RolloutManager


@dataclass(frozen=True)
class StabilityCertificationConfig:
    required_consecutive_days: int = 14
    min_pipeline_success_rate: float = 0.99
    max_open_p0_incidents: int = 0


@dataclass(frozen=True)
class StabilityCertificationResult:
    status: str
    elder_id: str
    run_id: str
    certification_date: str
    rollout_stage: str
    active_system: str
    stable_today: bool
    consecutive_stable_days: int
    required_consecutive_days: int
    certification_ready: bool
    blockers: tuple[str, ...] = ()
    signature: str = ""
    artifact_path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "elder_id": self.elder_id,
            "run_id": self.run_id,
            "certification_date": self.certification_date,
            "rollout_stage": self.rollout_stage,
            "active_system": self.active_system,
            "stable_today": self.stable_today,
            "consecutive_stable_days": self.consecutive_stable_days,
            "required_consecutive_days": self.required_consecutive_days,
            "certification_ready": self.certification_ready,
            "blockers": list(self.blockers),
            "signature": self.signature,
            "artifact_path": self.artifact_path,
            "details": dict(self.details),
        }


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_date(value: Optional[str]) -> str:
    if value:
        token = str(value).strip()
        if token:
            return token
    return _now_utc().date().isoformat()


def _state_path(state_root: Path, elder_id: str) -> Path:
    safe_elder = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in elder_id)
    return state_root / safe_elder / "phase6_stability_state.json"


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "consecutive_stable_days": 0,
            "last_date": None,
            "history": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "consecutive_stable_days": 0,
            "last_date": None,
            "history": [],
        }
    if not isinstance(payload, dict):
        return {
            "consecutive_stable_days": 0,
            "last_date": None,
            "history": [],
        }
    payload.setdefault("consecutive_stable_days", 0)
    payload.setdefault("last_date", None)
    payload.setdefault("history", [])
    return payload


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def _active_system_for_stage(stage: str) -> str:
    token = str(stage).strip().lower()
    if token == RolloutStage.FULL.value:
        return "beta6_authority"
    return "beta5.5_authority"


def _resolve_stability_config() -> StabilityCertificationConfig:
    policy = load_rollout_ladder_policy()
    required_days = max((int(r.min_days) for r in policy.rungs), default=14)
    return StabilityCertificationConfig(
        required_consecutive_days=max(1, required_days),
        min_pipeline_success_rate=float(policy.progression.min_nightly_pipeline_success_rate),
        max_open_p0_incidents=0,
    )


def _hash_report_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def run_daily_stability_certification(
    *,
    elder_id: str,
    run_id: str,
    beta6_gate_pass: bool,
    beta6_gate_report: Optional[Mapping[str, Any]] = None,
    beta6_fallback_summary: Optional[Mapping[str, Any]] = None,
    pipeline_success_rate: float = 1.0,
    open_p0_incidents: int = 0,
    certification_date: Optional[str] = None,
    state_root: Optional[Path] = None,
    artifact_path: Optional[Path] = None,
) -> StabilityCertificationResult:
    """Evaluate one-day Phase 6.3 stability and persist rolling certification state."""
    manager = T80RolloutManager()
    rollout_state = manager.get_state()
    stage = rollout_state.stage.value if rollout_state is not None else RolloutStage.SHADOW.value

    config = _resolve_stability_config()
    blockers: list[str] = []
    if not bool(beta6_gate_pass):
        blockers.append("beta6_authority_gate_failed")

    if int(open_p0_incidents) > int(config.max_open_p0_incidents):
        blockers.append(f"open_p0_incidents={int(open_p0_incidents)}")

    try:
        pipeline_success = float(pipeline_success_rate)
    except (TypeError, ValueError):
        pipeline_success = 0.0
    if pipeline_success < float(config.min_pipeline_success_rate):
        blockers.append(
            "pipeline_success_rate_below_floor"
            f"({pipeline_success:.4f}<{config.min_pipeline_success_rate:.4f})"
        )

    shadow_compare = {}
    if isinstance(beta6_gate_report, Mapping):
        shadow_compare = beta6_gate_report.get("phase6_shadow_compare") or {}
    if isinstance(shadow_compare, Mapping):
        summary = shadow_compare.get("summary")
        if isinstance(summary, Mapping):
            shadow_status = str(summary.get("status") or "").strip().lower()
            if shadow_status == "critical":
                blockers.append("phase6_shadow_compare_critical")

    fallback_active = False
    if isinstance(beta6_fallback_summary, Mapping):
        activated = beta6_fallback_summary.get("activated")
        if isinstance(activated, list) and activated:
            fallback_active = True
    if fallback_active:
        blockers.append("beta6_fallback_active")

    stable_today = len(blockers) == 0
    report_date = _normalize_date(certification_date)
    if state_root is None:
        state_root = Path(__file__).resolve().parents[3] / "validation_runs_canary" / "phase6_stability"
    state_path = _state_path(Path(state_root), elder_id)
    state = _load_state(state_path)

    previous_days = int(state.get("consecutive_stable_days", 0) or 0)
    previous_date = str(state.get("last_date") or "")
    if previous_date == report_date:
        consecutive_days = previous_days if stable_today else 0
    else:
        consecutive_days = previous_days + 1 if stable_today else 0

    certification_ready = stable_today and consecutive_days >= int(config.required_consecutive_days)
    history = list(state.get("history") or [])
    history.append(
        {
            "date": report_date,
            "stable_today": stable_today,
            "blockers": list(blockers),
            "consecutive_stable_days": consecutive_days,
            "rollout_stage": stage,
            "pipeline_success_rate": pipeline_success,
        }
    )
    if len(history) > 30:
        history = history[-30:]

    persisted_state = {
        "last_date": report_date,
        "consecutive_stable_days": consecutive_days,
        "certification_ready": certification_ready,
        "required_consecutive_days": int(config.required_consecutive_days),
        "history": history,
    }
    _write_state(state_path, persisted_state)

    details = {
        "state_path": str(state_path),
        "pipeline_success_rate": pipeline_success,
        "open_p0_incidents": int(open_p0_incidents),
        "beta6_gate_reason_code": str((beta6_gate_report or {}).get("reason_code") or ""),
    }
    base_payload = {
        "status": "ok",
        "elder_id": elder_id,
        "run_id": run_id,
        "certification_date": report_date,
        "rollout_stage": stage,
        "active_system": _active_system_for_stage(stage),
        "stable_today": stable_today,
        "consecutive_stable_days": consecutive_days,
        "required_consecutive_days": int(config.required_consecutive_days),
        "certification_ready": certification_ready,
        "blockers": list(blockers),
        "details": details,
    }
    signature = _hash_report_payload(base_payload)

    output_path: Optional[str] = None
    if artifact_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_payload = {**base_payload, "signature": signature}
        artifact_path.write_text(json.dumps(artifact_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output_path = str(artifact_path)

    return StabilityCertificationResult(
        status="ok",
        elder_id=elder_id,
        run_id=run_id,
        certification_date=report_date,
        rollout_stage=stage,
        active_system=_active_system_for_stage(stage),
        stable_today=stable_today,
        consecutive_stable_days=consecutive_days,
        required_consecutive_days=int(config.required_consecutive_days),
        certification_ready=certification_ready,
        blockers=tuple(blockers),
        signature=signature,
        artifact_path=output_path,
        details=details,
    )


__all__ = [
    "StabilityCertificationConfig",
    "StabilityCertificationResult",
    "run_daily_stability_certification",
]
