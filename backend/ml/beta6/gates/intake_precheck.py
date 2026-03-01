"""Fail-closed intake artifact precheck for Beta 6 Phase 1+ jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .intake_gate import assert_intake_artifact_approved

REASON_INTAKE_MISSING_ARTIFACT = "intake_gate_missing_artifact"
REASON_INTAKE_INVALID_ARTIFACT = "intake_gate_invalid_artifact"
REASON_INTAKE_NOT_APPROVED = "intake_gate_not_approved"


@dataclass(frozen=True)
class IntakeGateBlockedError(RuntimeError):
    """Deterministic fail-closed error for blocked Beta 6 model runs."""

    reason_code: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"{self.reason_code}: {self.detail}"


def enforce_approved_intake_artifact(
    artifact_path: Path | str,
    *,
    require_report_files: bool = True,
) -> Dict[str, Any]:
    """
    Validate and enforce approved intake artifact for Phase 1+ execution.

    Raises IntakeGateBlockedError with deterministic reason_code on failure.
    """
    path = Path(artifact_path).resolve()
    if not path.exists():
        raise IntakeGateBlockedError(
            reason_code=REASON_INTAKE_MISSING_ARTIFACT,
            detail=f"intake artifact not found: {path}",
        )

    try:
        return assert_intake_artifact_approved(
            path,
            require_report_files=require_report_files,
        )
    except ValueError as exc:
        message = str(exc)
        if "Intake gate is not approved" in message:
            raise IntakeGateBlockedError(
                reason_code=REASON_INTAKE_NOT_APPROVED,
                detail=message,
            ) from exc
        raise IntakeGateBlockedError(
            reason_code=REASON_INTAKE_INVALID_ARTIFACT,
            detail=message,
        ) from exc
    except (TypeError, json.JSONDecodeError, OSError) as exc:
        raise IntakeGateBlockedError(
            reason_code=REASON_INTAKE_INVALID_ARTIFACT,
            detail=str(exc),
        ) from exc
