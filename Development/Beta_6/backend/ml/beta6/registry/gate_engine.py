"""Gate engine scaffold for Beta 6 room/run decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..serving.capability_profiles import select_capability_profile
from ..contracts.decisions import ReasonCode, RoomDecision, RunDecision, resolve_uncertainty
from ..evaluation.evaluation_engine import create_signed_evaluation_report
from .rejection_artifact import create_signed_rejection_artifact
from ..gates.timeline_hard_gates import TimelineHardGateResult, evaluate_timeline_hard_gates


class GateEngine:
    """Deterministic mapping from reports to room/run decisions."""

    _RUN_FAILURE_PRECEDENCE: List[ReasonCode] = [
        ReasonCode.FAIL_DATA_VIABILITY,
        ReasonCode.FAIL_LEAKAGE_RESIDENT,
        ReasonCode.FAIL_LEAKAGE_TIME,
        ReasonCode.FAIL_LEAKAGE_WINDOW,
        ReasonCode.FAIL_UNCERTAINTY_CONFLICT,
        ReasonCode.FAIL_UNCERTAINTY_INVALID,
        ReasonCode.FAIL_UNCERTAINTY_OUTSIDE_SENSED_SPACE,
        ReasonCode.FAIL_UNCERTAINTY_UNKNOWN,
        ReasonCode.FAIL_UNCERTAINTY_LOW_CONFIDENCE,
        ReasonCode.FAIL_RUNTIME_EVAL_PARITY,
        ReasonCode.FAIL_RUN_INCOMPLETE,
        ReasonCode.FAIL_TIMELINE_METRICS_MISSING,
        ReasonCode.FAIL_TIMELINE_MAE,
        ReasonCode.FAIL_TIMELINE_FRAGMENTATION,
        ReasonCode.FAIL_GATE_POLICY,
        ReasonCode.FAIL_UNKNOWN_REASON,
    ]

    def _fallback_reason(
        self,
        report: Dict[str, object],
        timeline_gate: TimelineHardGateResult,
    ) -> ReasonCode:
        leakage = report.get("leakage")
        leakage_map = leakage if isinstance(leakage, dict) else {}

        if not bool(report.get("data_viable", True)):
            return ReasonCode.FAIL_DATA_VIABILITY
        if bool(leakage_map.get("resident_overlap", False)):
            return ReasonCode.FAIL_LEAKAGE_RESIDENT
        if bool(leakage_map.get("time_overlap", False)):
            return ReasonCode.FAIL_LEAKAGE_TIME
        if bool(leakage_map.get("window_overlap", False)):
            return ReasonCode.FAIL_LEAKAGE_WINDOW
        uncertainty = resolve_uncertainty(report)
        if uncertainty.reason_code is not None:
            return uncertainty.reason_code
        if bool(report.get("run_incomplete", False)):
            return ReasonCode.FAIL_RUN_INCOMPLETE
        if timeline_gate.reason_code is not None:
            return timeline_gate.reason_code
        if bool(report.get("metrics_passed", False)) is False:
            return ReasonCode.FAIL_GATE_POLICY
        return ReasonCode.FAIL_UNKNOWN_REASON

    def decide_room(self, room: str, report: Dict[str, object]) -> RoomDecision:
        """Map a room evaluation report to a room decision."""
        reported_passed = bool(report.get("passed", False))
        report_room_type = report.get("room_type")
        room_type = str(report_room_type) if isinstance(report_room_type, str) else None
        capability_profile = select_capability_profile(room=room, room_type=room_type)
        timeline_gate = evaluate_timeline_hard_gates(report, capability_profile)

        # Timeline hard gates are primary and can override nominal pass signals.
        timeline_forced_block = reported_passed and timeline_gate.checked and not timeline_gate.passed
        passed = bool(reported_passed and not timeline_forced_block)
        raw_reason = report.get("reason_code")
        reason: ReasonCode
        if passed:
            reason = ReasonCode.PASS
        elif raw_reason is None:
            reason = self._fallback_reason(report, timeline_gate)
        else:
            try:
                reason = ReasonCode(str(raw_reason))
            except ValueError:
                reason = self._fallback_reason(report, timeline_gate)

        details = dict(report.get("details", {}))
        for detail_key in (
            "require_timeline_metrics",
            "runtime_eval_parity_checked",
            "runtime_eval_parity_passed",
            "runtime_eval_parity_required",
        ):
            if detail_key in report:
                details.setdefault(detail_key, report.get(detail_key))
        details.setdefault("room_type", capability_profile.room_type)
        details.setdefault("capability_profile_id", capability_profile.profile_id)
        details.setdefault("timeline_hard_gate_checked", timeline_gate.checked)
        details.setdefault("timeline_hard_gate_passed", timeline_gate.passed)
        if timeline_gate.details:
            details.setdefault("timeline_hard_gate", timeline_gate.details)
        if timeline_forced_block:
            details.setdefault("pass_overridden_by_timeline_hard_gate", True)
        uncertainty = resolve_uncertainty(report)
        if uncertainty.state is not None:
            details.setdefault("uncertainty_state", uncertainty.state.value)
            details.setdefault("uncertainty_source", uncertainty.source)
        if uncertainty.error is not None:
            details.setdefault("uncertainty_error", uncertainty.error)
        if not passed:
            details = {
                **details,
                "reason_source": "explicit" if raw_reason is not None else "derived",
            }
        return RoomDecision(room=room, passed=passed, reason_code=reason, details=details)

    def decide_run(self, room_decisions: Iterable[RoomDecision]) -> RunDecision:
        """Roll up room decisions into one run decision."""
        decisions = list(room_decisions)
        if not decisions:
            return RunDecision(
                passed=False,
                reason_code=ReasonCode.FAIL_RUN_INCOMPLETE,
                room_decisions=[],
                details={"failed_rooms": [], "summary": "no_room_decisions"},
            )

        failed = [decision for decision in decisions if not decision.passed]
        if failed:
            first_reason = next(
                (
                    reason
                    for reason in self._RUN_FAILURE_PRECEDENCE
                    if any(decision.reason_code == reason for decision in failed)
                ),
                failed[0].reason_code,
            )
            return RunDecision(
                passed=False,
                reason_code=first_reason,
                room_decisions=decisions,
                details={
                    "failed_rooms": [decision.room for decision in failed],
                    "failed_reason_codes": [decision.reason_code.value for decision in failed],
                },
            )
        return RunDecision(
            passed=True,
            reason_code=ReasonCode.PASS,
            room_decisions=decisions,
            details={},
        )

    def decide_run_from_reports(
        self,
        *,
        room_reports: Sequence[Mapping[str, Any]],
        run_id: str,
        elder_id: str,
        signing_key: str,
        output_dir: str | Path | None = None,
    ) -> Dict[str, Any]:
        """
        Evaluate room reports, roll up run decision, and persist signed artifacts.
        """
        room_decisions = []
        serialized_room_decisions = []
        for report in room_reports:
            room = str(report.get("room") or "").strip()
            if not room:
                continue
            decision = self.decide_room(room=room, report=dict(report))
            room_decisions.append(decision)
            serialized_room_decisions.append(
                {
                    "room": decision.room,
                    "passed": bool(decision.passed),
                    "reason_code": decision.reason_code.value,
                    "details": dict(decision.details),
                }
            )

        run_decision = self.decide_run(room_decisions)
        run_payload = {
            "passed": bool(run_decision.passed),
            "reason_code": run_decision.reason_code.value,
            "details": dict(run_decision.details),
            "room_decisions": serialized_room_decisions,
        }

        eval_output = None
        reject_output = None
        if output_dir is not None:
            out = Path(output_dir).resolve()
            out.mkdir(parents=True, exist_ok=True)
            eval_output = out / f"{run_id}_evaluation_report.json"
            reject_output = out / f"{run_id}_rejection_artifact.json"

        evaluation_report = create_signed_evaluation_report(
            run_id=run_id,
            elder_id=elder_id,
            room_reports=room_reports,
            signing_key=signing_key,
            output_path=eval_output,
            metadata=run_payload,
        )

        rejection_artifact: Optional[Dict[str, Any]] = None
        if not run_decision.passed:
            rejection_artifact = create_signed_rejection_artifact(
                run_id=run_id,
                elder_id=elder_id,
                reason_code=run_decision.reason_code.value,
                room_reports=serialized_room_decisions,
                signing_key=signing_key,
                output_path=reject_output,
            )

        return {
            "run_decision": run_payload,
            "evaluation_report": evaluation_report,
            "rejection_artifact": rejection_artifact,
        }
