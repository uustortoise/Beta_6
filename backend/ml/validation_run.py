"""
PR-5: Controlled Validation Run + Signoff Pack

Manages deterministic validation runs over 7+ days with:
- Deterministic manifest-based retraining
- Decision trace collection
- Rejection artifact aggregation
- Signoff pack generation

Usage:
    # Start a validation run
    python -m ml.validation_run start --elder-id HK001 --duration-days 7
    
    # Collect results and generate signoff pack
    python -m ml.validation_run finalize --elder-id HK001 --run-id <run_id>
"""

from __future__ import annotations

import json
import hashlib
import logging
import argparse
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import os

logger = logging.getLogger(__name__)

try:
    from ml.signoff_pack import SignoffPackGenerator as Ws6SignoffPackGenerator
    from ml.signoff_pack import SplitSeedResult as Ws6SplitSeedResult
except Exception:  # pragma: no cover - fallback for partial deployments
    Ws6SignoffPackGenerator = None
    Ws6SplitSeedResult = None

# Lane D1: Strict evaluation imports
try:
    from ml.leakage_audit import LeakageAuditor, LeakageAuditReport
except Exception:
    LeakageAuditor = None
    LeakageAuditReport = None


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class ValidationRunStatus(Enum):
    """Status of a validation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class DailyRunResult:
    """Result from a single day's training run."""
    date: str  # ISO date string
    run_timestamp: str
    manifest_hash: str
    rooms_trained: List[str]
    rooms_promoted: List[str]
    rooms_rejected: List[str]
    decision_traces: Dict[str, str]  # room -> path to decision trace
    rejection_artifacts: Dict[str, str]  # room -> path to rejection artifact
    gate_stack_summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "run_timestamp": self.run_timestamp,
            "manifest_hash": self.manifest_hash,
            "rooms_trained": self.rooms_trained,
            "rooms_promoted": self.rooms_promoted,
            "rooms_rejected": self.rooms_rejected,
            "decision_traces": self.decision_traces,
            "rejection_artifacts": self.rejection_artifacts,
            "gate_stack_summary": self.gate_stack_summary,
        }


@dataclass
class ValidationRun:
    """
    A controlled validation run over multiple days.
    """
    run_id: str
    elder_id: str
    start_date: str
    duration_days: int
    manifest_path: str
    manifest_hash: str
    status: ValidationRunStatus
    
    # Run configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Daily results
    daily_results: List[DailyRunResult] = field(default_factory=list)
    
    # Final summary (populated on completion)
    final_summary: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "elder_id": self.elder_id,
            "start_date": self.start_date,
            "duration_days": self.duration_days,
            "manifest_path": self.manifest_path,
            "manifest_hash": self.manifest_hash,
            "status": self.status.value,
            "config": self.config,
            "daily_results": [r.to_dict() for r in self.daily_results],
            "final_summary": self.final_summary,
        }
    
    def compute_determinism_score(self) -> Tuple[float, List[str]]:
        """
        Compute how deterministic the run was across days.
        
        Returns:
            (score, issues): Score 0-1, list of non-determinism issues
        """
        if len(self.daily_results) < 2:
            return 1.0, []
        
        issues = []
        consistent_promotions = 0
        total_comparisons = 0
        
        # Compare each day's promoted rooms
        base_result = self.daily_results[0]
        base_promoted = set(base_result.rooms_promoted)
        
        for result in self.daily_results[1:]:
            promoted = set(result.rooms_promoted)
            total_comparisons += 1
            
            if promoted == base_promoted:
                consistent_promotions += 1
            else:
                missing = base_promoted - promoted
                extra = promoted - base_promoted
                if missing:
                    issues.append(f"{result.date}: Missing promotions: {missing}")
                if extra:
                    issues.append(f"{result.date}: Extra promotions: {extra}")
        
        score = consistent_promotions / max(1, total_comparisons)
        return score, issues


class SignoffDecision(Enum):
    """Strict evaluation signoff decision."""
    PASS = "PASS"
    CONDITIONAL = "CONDITIONAL"
    FAIL = "FAIL"


@dataclass
class SignoffPack:
    """
    Complete signoff package for a validation run.
    
    Contains all artifacts needed for human review and signoff.
    """
    run_id: str
    generated_at: str
    
    # Run metadata
    validation_run: ValidationRun
    
    # Aggregated artifacts
    all_decision_traces: List[Dict[str, Any]]
    all_rejection_artifacts: List[Dict[str, Any]]
    gate_reason_summary: Dict[str, int]  # reason_code -> count
    
    # Determinism report
    determinism_score: float
    determinism_issues: List[str]
    
    # Compliance checklist
    compliance_checklist: Dict[str, bool]
    
    # Lane D1: Strict evaluation signoff
    signoff_decision: SignoffDecision = field(default=SignoffDecision.FAIL)
    signoff_reasons: List[str] = field(default_factory=list)
    blocking_issues: List[str] = field(default_factory=list)
    residual_risks: List[str] = field(default_factory=list)
    recommended_stage: str = field(default="not_ready")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "validation_run": self.validation_run.to_dict(),
            "all_decision_traces": self.all_decision_traces,
            "all_rejection_artifacts": self.all_rejection_artifacts,
            "gate_reason_summary": self.gate_reason_summary,
            "determinism_score": self.determinism_score,
            "determinism_issues": self.determinism_issues,
            "compliance_checklist": self.compliance_checklist,
            "signoff": {
                "decision": self.signoff_decision.value,
                "reasons": self.signoff_reasons,
                "blocking_issues": self.blocking_issues,
                "residual_risks": self.residual_risks,
                "recommended_stage": self.recommended_stage,
            },
        }
    
    def generate_markdown_report(self) -> str:
        """Generate human-readable markdown report."""
        lines = [
            "# Validation Run Signoff Pack",
            "",
            f"**Run ID:** {self.run_id}",
            f"**Generated:** {self.generated_at}",
            f"**Elder ID:** {self.validation_run.elder_id}",
            f"**Duration:** {self.validation_run.duration_days} days",
            f"**Status:** {self.validation_run.status.value}",
            "",
            "## Determinism Report",
            "",
            f"**Score:** {self.determinism_score:.1%}",
            "",
        ]
        
        if self.determinism_issues:
            lines.append("### Issues")
            for issue in self.determinism_issues:
                lines.append(f"- ⚠️ {issue}")
        else:
            lines.append("✅ No determinism issues detected")
        
        # Lane D1: Signoff Decision
        lines.extend([
            "",
            "## Signoff Decision (Lane D1 Strict Evaluation)",
            "",
            f"**Decision:** {self.signoff_decision.value}",
            f"**Recommended Stage:** {self.recommended_stage}",
        ])
        
        if self.blocking_issues:
            lines.extend(["### Blocking Issues", ""])
            for issue in self.blocking_issues:
                lines.append(f"- 🚫 {issue}")
            lines.append("")
        
        if self.signoff_reasons:
            lines.extend(["### Reasons", ""])
            for reason in self.signoff_reasons:
                lines.append(f"- ℹ️ {reason}")
            lines.append("")
        
        if self.residual_risks:
            lines.extend(["### Residual Risks", ""])
            for risk in self.residual_risks[:5]:  # Limit to top 5
                lines.append(f"- ⚠️ {risk}")
            lines.append("")
        
        lines.extend([
            "",
            "## Compliance Checklist",
            "",
        ])
        
        for check, passed in self.compliance_checklist.items():
            status = "✅" if passed else "❌"
            lines.append(f"- {status} {check}")
        
        lines.extend([
            "",
            "## Gate Reason Summary",
            "",
        ])
        
        if self.gate_reason_summary:
            for reason, count in sorted(self.gate_reason_summary.items()):
                lines.append(f"- {reason}: {count}")
        else:
            lines.append("No gate rejections recorded")
        
        lines.extend([
            "",
            "## Daily Results",
            "",
        ])
        
        for daily in self.validation_run.daily_results:
            lines.append(f"### {daily.date}")
            lines.append(f"- Trained: {len(daily.rooms_trained)} rooms")
            lines.append(f"- Promoted: {len(daily.rooms_promoted)} rooms")
            lines.append(f"- Rejected: {len(daily.rooms_rejected)} rooms")
            lines.append("")
        
        return "\n".join(lines)


@dataclass
class StrictEvaluationConfig:
    """Configuration for Lane D1 strict evaluation protocol."""
    
    # Rolling splits configuration
    splits: List[Tuple[List[int], int]] = field(default_factory=lambda: [
        ([4], 5),
        ([4, 5], 6),
        ([4, 5, 6], 7),
        ([4, 5, 6, 7], 8),
    ])
    
    # Seeds for reproducibility
    seeds: List[int] = field(default_factory=lambda: [11, 22, 33])
    
    # Promotion thresholds
    tier1_recall_floor: float = 0.50
    tier2_recall_floor: float = 0.35
    tier3_recall_floor: float = 0.20
    home_empty_precision_floor: float = 0.95
    false_empty_rate_ceiling: float = 0.05
    timeline_gate_pass_rate: float = 0.80
    macro_f1_stability_std_ceiling: float = 0.05
    
    # Compliance requirements
    require_leakage_audit: bool = True
    require_all_hard_gates: bool = True
    require_timeline_gates_80pct: bool = True
    
    def get_split_seed_matrix(self) -> List[Tuple[str, int]]:
        """Get all split-seed combinations for evaluation."""
        matrix = []
        for train_days, val_day in self.splits:
            split_id = f"{','.join(map(str, train_days))}->{val_day}"
            for seed in self.seeds:
                matrix.append((split_id, seed))
        return matrix


class ValidationRunManager:
    """
    Manages validation runs and signoff pack generation.
    
    Lane D1: Supports strict evaluation protocol with:
    - Rolling split-seed matrix execution
    - Mandatory leakage artifacts
    - Signoff decision (PASS/CONDITIONAL/FAIL)
    """
    
    DEFAULT_RUNS_DIR = Path("validation_runs")
    
    def __init__(self, runs_dir: Optional[Path] = None):
        """
        Initialize manager.
        
        Args:
            runs_dir: Directory to store validation run data
        """
        self.runs_dir = runs_dir or self.DEFAULT_RUNS_DIR
        self.runs_dir.mkdir(parents=True, exist_ok=True)
    
    def create_run(
        self,
        elder_id: str,
        duration_days: int,
        manifest_path: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> ValidationRun:
        """
        Create a new validation run.
        
        Args:
            elder_id: Elder/resident ID
            duration_days: Number of days to run
            manifest_path: Path to deterministic manifest file
            config: Additional run configuration
            
        Returns:
            New ValidationRun instance
        """
        # Compute manifest hash
        manifest_content = Path(manifest_path).read_bytes()
        manifest_hash = hashlib.sha256(manifest_content).hexdigest()[:16]
        
        # Generate run ID
        timestamp = _utc_now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{elder_id}_{timestamp}_{manifest_hash}"
        
        run = ValidationRun(
            run_id=run_id,
            elder_id=elder_id,
            start_date=_utc_now().isoformat(),
            duration_days=duration_days,
            manifest_path=manifest_path,
            manifest_hash=manifest_hash,
            status=ValidationRunStatus.PENDING,
            config=config or {},
        )
        
        self._save_run(run)
        logger.info(f"Created validation run: {run_id}")
        
        return run
    
    def record_daily_result(
        self,
        run_id: str,
        result: DailyRunResult,
    ) -> None:
        """Record a daily result for a validation run."""
        run = self._load_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        
        run.daily_results.append(result)
        run.status = ValidationRunStatus.RUNNING
        
        self._save_run(run)
        logger.info(f"Recorded daily result for {run_id}: {result.date}")
    
    def finalize_run(
        self, 
        run_id: str,
        force: bool = False,
        strict_config: Optional[StrictEvaluationConfig] = None,
    ) -> SignoffPack:
        """
        Finalize a validation run and generate signoff pack.
        
        Lane D1: Strict evaluation with mandatory compliance checks.
        force=True allows artifact generation but NEVER changes PASS decision.
        
        Args:
            run_id: Validation run ID
            force: If True, allow finalization even if compliance checks fail
            strict_config: Strict evaluation configuration (uses default if None)
            
        Returns:
            Complete SignoffPack
            
        Raises:
            RuntimeError: If compliance checks fail and force=False
        """
        config = strict_config or StrictEvaluationConfig()
        run = self._load_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        
        # Load all artifacts (mandatory - failures are tracked)
        all_decision_traces = []
        all_rejection_artifacts = []
        gate_reason_summary = {}
        artifact_errors: List[str] = []
        
        for daily in run.daily_results:
            # Load decision traces
            for room, trace_path in daily.decision_traces.items():
                try:
                    path = Path(trace_path)
                    if not path.exists():
                        raise FileNotFoundError(f"Decision trace not found: {trace_path}")
                    trace_data = json.loads(path.read_text())
                    all_decision_traces.append(trace_data)
                except Exception as e:
                    error_msg = f"Decision trace {trace_path}: {e}"
                    artifact_errors.append(error_msg)
                    logger.error(error_msg)
            
            # Load rejection artifacts
            for room, artifact_path in daily.rejection_artifacts.items():
                try:
                    path = Path(artifact_path)
                    if not path.exists():
                        raise FileNotFoundError(f"Rejection artifact not found: {artifact_path}")
                    artifact_data = json.loads(path.read_text())
                    all_rejection_artifacts.append(artifact_data)
                    
                    # Count gate reasons
                    for reason in artifact_data.get("reasons", []):
                        code = reason.get("code", "unknown")
                        gate_reason_summary[code] = gate_reason_summary.get(code, 0) + 1
                except Exception as e:
                    error_msg = f"Rejection artifact {artifact_path}: {e}"
                    artifact_errors.append(error_msg)
                    logger.error(error_msg)
        
        # Compute determinism score
        determinism_score, determinism_issues = run.compute_determinism_score()
        
        # Build compliance checklist (Lane D1: strict evaluation)
        compliance_checklist = self._build_compliance_checklist(run, config)
        compliance_passed = all(compliance_checklist.values())
        
        # Determine run status based on compliance
        if compliance_passed:
            run.status = ValidationRunStatus.COMPLETED
            logger.info(f"Validation run {run_id} passed all compliance checks")
        elif force:
            run.status = ValidationRunStatus.COMPLETED
            logger.warning(
                f"Validation run {run_id} finalized with FAILED compliance checks "
                f"(force=True). Signoff decision will still be FAIL if checks failed."
            )
        else:
            run.status = ValidationRunStatus.FAILED
            failed_checks = [k for k, v in compliance_checklist.items() if not v]
            self._save_run(run)
            raise RuntimeError(
                f"Validation run {run_id} failed compliance checks: {failed_checks}. "
                f"Use force=True to override."
            )
        
        # Lane D1: Compute strict signoff decision (force never changes decision to PASS)
        signoff_decision, signoff_reasons, blocking_issues, residual_risks, recommended_stage = (
            self._compute_signoff_decision(
                run=run,
                compliance_checklist=compliance_checklist,
                determinism_score=determinism_score,
                determinism_issues=determinism_issues,
                gate_reason_summary=gate_reason_summary,
                strict_config=config,
            )
        )
        
        # Create signoff pack with Lane D1 strict evaluation
        pack = SignoffPack(
            run_id=run_id,
            generated_at=_utc_now().isoformat(),
            validation_run=run,
            all_decision_traces=all_decision_traces,
            all_rejection_artifacts=all_rejection_artifacts,
            gate_reason_summary=gate_reason_summary,
            determinism_score=determinism_score,
            determinism_issues=determinism_issues,
            compliance_checklist=compliance_checklist,
            signoff_decision=signoff_decision,
            signoff_reasons=signoff_reasons,
            blocking_issues=blocking_issues,
            residual_risks=residual_risks,
            recommended_stage=recommended_stage,
        )
        
        # WS-6 artifact generation is mandatory for promotion (fail-closed).
        # Generate WS-6 first so we never persist a partial signoff pack when
        # promotion-critical WS-6 evidence is unavailable.
        ws6_pack_path = self._save_ws6_signoff_pack(run, strict_config=config, promotion_mode=True)
        self._save_signoff_pack(pack)

        # Update run final summary
        run.final_summary = {
            "determinism_score": determinism_score,
            "total_decision_traces": len(all_decision_traces),
            "total_rejection_artifacts": len(all_rejection_artifacts),
            "compliance_passed": compliance_passed,
            "compliance_forced": force and not compliance_passed,
            "failed_checks": [k for k, v in compliance_checklist.items() if not v],
            "ws6_signoff_pack_generated": ws6_pack_path is not None,
            "ws6_signoff_pack_path": str(ws6_pack_path) if ws6_pack_path else None,
        }
        self._save_run(run)
        
        logger.info(f"Finalized validation run: {run_id}")
        
        return pack

    def _save_ws6_signoff_pack(
        self, 
        run: ValidationRun,
        strict_config: Optional[StrictEvaluationConfig] = None,
        promotion_mode: bool = True,
    ) -> Optional[Path]:
        """
        Generate and save WS-6 signoff artifacts.
        
        In promotion mode, hard-fail if WS-6 module is unavailable.
        In non-promotion mode, skip gracefully with warning.
        """
        if Ws6SignoffPackGenerator is None or Ws6SplitSeedResult is None:
            if promotion_mode:
                logger.error("WS-6 signoff pack module unavailable; blocking promotion.")
                raise RuntimeError(
                    "WS-6 signoff pack module is required for promotion. "
                    "Ensure ml.signoff_pack is available and properly imported."
                )
            logger.warning("WS-6 signoff pack module unavailable; skipping WS-6 artifact generation.")
            return None

        config = strict_config or StrictEvaluationConfig()
        baseline_version = str(run.config.get("baseline_version", "unknown"))
        baseline_artifact_hash = str(run.config.get("baseline_artifact_hash", ""))
        git_sha = str(run.config.get("git_sha", os.environ.get("GIT_SHA", "unknown")))

        generator = Ws6SignoffPackGenerator(
            run_id=run.run_id,
            elder_id=run.elder_id,
            baseline_version=baseline_version,
            git_sha=git_sha,
        )

        # Lane D1: Build split-seed matrix from expected splits/seeds
        expected_matrix = config.get_split_seed_matrix()  # List of (split_id, seed)
        
        # Map daily results by (split_id, seed) for lookup.
        # Invalid/malformed cells are skipped so WS-6 artifact generation
        # never crashes and remains fail-closed via missing-cell placeholders.
        daily_by_split_seed: Dict[Tuple[str, int], DailyRunResult] = {}
        for daily in run.daily_results:
            gate_summary = daily.gate_stack_summary if isinstance(daily.gate_stack_summary, dict) else {}
            split_id = str(gate_summary.get("split_id", daily.date))
            seed_raw = gate_summary.get("seed", None)
            try:
                seed = int(seed_raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping WS-6 split-seed mapping for invalid seed '%s' in daily result '%s'",
                    seed_raw,
                    daily.date,
                )
                continue
            key = (split_id, seed)
            if key in daily_by_split_seed:
                logger.warning(
                    "Duplicate WS-6 split-seed cell encountered (%s, %s); keeping first occurrence.",
                    split_id,
                    seed,
                )
                continue
            daily_by_split_seed[key] = daily

        # Add results for each expected split-seed cell
        for split_id, seed in expected_matrix:
            daily = daily_by_split_seed.get((split_id, seed))
            if daily:
                generator.add_split_seed_result(self._build_ws6_split_seed_result(daily, seed, split_id))
            else:
                # Missing cell - add placeholder with failure
                logger.warning(f"Missing split-seed cell: {split_id} seed {seed}")
                generator.add_split_seed_result(self._build_missing_split_seed_result(split_id, seed))

        baseline_metrics = run.config.get("baseline_metrics")
        if not isinstance(baseline_metrics, dict):
            baseline_metrics = None
        kpi_checks = run.config.get("kpi_checks")
        if not isinstance(kpi_checks, list):
            kpi_checks = None
        timeline_checks = run.config.get("timeline_checks")
        if not isinstance(timeline_checks, list):
            timeline_checks = None

        pack = generator.generate(
            baseline_metrics=baseline_metrics,
            baseline_artifact_hash=baseline_artifact_hash,
            data_version=str(run.config.get("data_version", "")),
            feature_schema_hash=str(run.config.get("feature_schema_hash", "")),
            kpi_checks=kpi_checks,
            timeline_checks=timeline_checks,
        )

        pack_dir = self.runs_dir / run.run_id
        pack_dir.mkdir(exist_ok=True)

        json_path = pack_dir / "signoff_pack_ws6.json"
        json_path.write_text(json.dumps(pack.to_dict(), indent=2))

        md_path = pack_dir / "signoff_report_ws6.md"
        md_path.write_text(pack.decision.to_markdown())

        logger.info(f"Saved WS-6 signoff pack to {json_path}")
        return json_path

    def _build_ws6_split_seed_result(
        self, 
        daily: DailyRunResult, 
        expected_seed: int,
        expected_split_id: str,
    ) -> Any:
        """Map PR-5 daily result shape into WS-6 split/seed result shape."""
        gate_summary = daily.gate_stack_summary if isinstance(daily.gate_stack_summary, dict) else {}

        hard_passed, hard_total = self._extract_gate_counts(gate_summary, gate_type="hard")
        if hard_total == 0:
            hard_total = len(daily.rooms_trained)
            hard_passed = len(daily.rooms_promoted)

        timeline_passed, timeline_total = self._extract_gate_counts(gate_summary, gate_type="timeline")

        leakage_audit = gate_summary.get("leakage_audit", {})
        leakage_pass = bool(gate_summary.get("leakage_audit_pass", gate_summary.get("leakage_pass", False)))
        leakage_violations: List[str] = []
        if isinstance(leakage_audit, dict):
            leakage_pass = bool(leakage_audit.get("pass", leakage_pass))
            reasons = leakage_audit.get("reasons", [])
            if isinstance(reasons, list):
                leakage_violations = [str(reason) for reason in reasons]

        # Validate seed matches expected
        seed_raw = gate_summary.get("seed", expected_seed)
        try:
            seed = int(seed_raw)
        except (TypeError, ValueError):
            seed = expected_seed

        # Use expected split_id from matrix (not daily.date)
        split_id = expected_split_id
        train_days = gate_summary.get("train_days", [])
        val_days = gate_summary.get("val_days", [])
        if not isinstance(train_days, list):
            train_days = []
        if not isinstance(val_days, list):
            val_days = []

        return Ws6SplitSeedResult(
            split_id=split_id,
            seed=seed,
            train_days=train_days,
            val_days=val_days,
            timeline_metrics=self._extract_timeline_metrics(gate_summary),
            hard_gates_passed=hard_passed,
            hard_gates_total=hard_total,
            timeline_gates_passed=timeline_passed,
            timeline_gates_total=timeline_total,
            leakage_audit_pass=leakage_pass,
            leakage_violations=leakage_violations,
        )
    
    def _build_missing_split_seed_result(self, split_id: str, seed: int) -> Any:
        """Build a placeholder SplitSeedResult for missing cells."""
        return Ws6SplitSeedResult(
            split_id=split_id,
            seed=seed,
            train_days=[],
            val_days=[],
            timeline_metrics={},
            hard_gates_passed=0,
            hard_gates_total=1,
            timeline_gates_passed=0,
            timeline_gates_total=1,
            leakage_audit_pass=False,
            leakage_violations=["missing_split_seed_cell"],
        )

    @staticmethod
    def _extract_gate_counts(gate_summary: Dict[str, Any], gate_type: str) -> Tuple[int, int]:
        """Extract passed/total gate counts from varying payload shapes."""
        passed_key = f"{gate_type}_gates_passed"
        total_key = f"{gate_type}_gates_total"

        passed = gate_summary.get(passed_key)
        total = gate_summary.get(total_key)
        if isinstance(passed, int) and isinstance(total, int) and total >= 0:
            return max(0, min(passed, total)), total

        list_keys = [f"{gate_type}_gates"]
        if gate_type == "hard":
            list_keys.append("hard_checks")
        if gate_type == "timeline":
            list_keys.append("timeline_checks")

        for key in list_keys:
            maybe_list = gate_summary.get(key)
            if isinstance(maybe_list, list) and maybe_list:
                total_from_list = len(maybe_list)
                passed_from_list = sum(
                    1
                    for item in maybe_list
                    if isinstance(item, dict) and bool(item.get("pass", False))
                )
                return passed_from_list, total_from_list

        if gate_type == "hard" and isinstance(gate_summary.get("all_hard_gates_pass"), bool):
            total_checks = gate_summary.get("total_checks")
            pass_checks = gate_summary.get("pass_checks")
            if isinstance(total_checks, int) and total_checks > 0:
                if isinstance(pass_checks, int):
                    return max(0, min(pass_checks, total_checks)), total_checks
                if gate_summary["all_hard_gates_pass"]:
                    return total_checks, total_checks
                return 0, total_checks

        return 0, 0

    @staticmethod
    def _extract_timeline_metrics(gate_summary: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract per-room timeline metrics and keep only numeric values."""
        metrics_by_room = gate_summary.get("timeline_metrics", {})
        if not isinstance(metrics_by_room, dict):
            metrics_by_room = {}

        normalized: Dict[str, Dict[str, float]] = {}
        for room, metrics in metrics_by_room.items():
            if not isinstance(metrics, dict):
                continue
            numeric = {
                str(key): float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if numeric:
                normalized[str(room)] = numeric

        if normalized:
            return normalized

        rooms_payload = gate_summary.get("rooms", {})
        if not isinstance(rooms_payload, dict):
            return {}

        for room, room_payload in rooms_payload.items():
            if not isinstance(room_payload, dict):
                continue
            room_metrics = room_payload.get("timeline_metrics")
            if not isinstance(room_metrics, dict):
                continue
            numeric = {
                str(key): float(value)
                for key, value in room_metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if numeric:
                normalized[str(room)] = numeric

        return normalized
    
    def _build_compliance_checklist(
        self, 
        run: ValidationRun,
        strict_config: Optional[StrictEvaluationConfig] = None,
    ) -> Dict[str, bool]:
        """Build compliance checklist for a run with strict evaluation."""
        config = strict_config or StrictEvaluationConfig()
        
        # Core compliance checks (all mandatory)
        checklist: Dict[str, bool] = {}
        
        # 1. Split-seed matrix coverage (Lane D1: 4 splits × 3 seeds = 12 cells)
        checklist["split_seed_matrix_complete"] = self._check_split_seed_matrix_complete(run, config)
        checklist["minimum_7_days"] = len(run.daily_results) >= 7
        
        # 2. Determinism check
        determinism_score, _ = run.compute_determinism_score()
        checklist["determinism_score_above_80"] = determinism_score >= 0.8
        
        # 3. Artifact integrity checks (fail-closed: require readable files)
        artifact_checks = self._check_artifact_integrity(run)
        checklist["all_decision_traces_readable"] = artifact_checks["decision_traces_readable"]
        checklist["all_rejection_artifacts_readable"] = artifact_checks["rejection_artifacts_readable"]
        checklist["all_rooms_have_decision_traces"] = all(
            len(d.decision_traces) == len(d.rooms_trained)
            for d in run.daily_results
        )
        checklist["no_missing_gate_reasons"] = all(
            len(d.rooms_rejected) == len(d.rejection_artifacts)
            for d in run.daily_results
        )
        
        # 4. Data integrity
        checklist["manifest_hash_consistent"] = all(
            d.manifest_hash == run.manifest_hash
            for d in run.daily_results
        )
        
        # 5. Lane D1: Mandatory leakage audit check
        checklist["leakage_audit_present"] = self._check_leakage_audit_present(run)
        checklist["leakage_audit_pass"] = self._check_leakage_audit_pass(run)
        
        # 6. Lane D1: Hard gates check (all must pass)
        checklist["all_hard_gates_pass"] = self._check_all_hard_gates_pass(run)
        
        # 7. Lane D1: Timeline gates check (configured threshold, default 80%)
        checklist["timeline_gates_pass"] = self._check_timeline_gates_pass(run, config.timeline_gate_pass_rate)
        
        return checklist

    def _check_split_seed_matrix_complete(
        self,
        run: ValidationRun,
        config: StrictEvaluationConfig,
    ) -> bool:
        """
        Validate exact split-seed matrix completeness.

        Passes only when:
        - every expected split-seed cell exists
        - no duplicates
        - no extra cells
        """
        expected_cells = set(config.get_split_seed_matrix())
        observed_cells: List[Tuple[str, int]] = []

        for daily in run.daily_results:
            gate_summary = daily.gate_stack_summary if isinstance(daily.gate_stack_summary, dict) else {}
            split_id = str(gate_summary.get("split_id", daily.date))
            seed_raw = gate_summary.get("seed", None)
            try:
                seed = int(seed_raw)
            except (TypeError, ValueError):
                return False
            observed_cells.append((split_id, seed))

        observed_set = set(observed_cells)
        if len(observed_cells) != len(expected_cells):
            return False
        if len(observed_set) != len(observed_cells):
            return False
        return observed_set == expected_cells
    
    def _check_artifact_integrity(self, run: ValidationRun) -> Dict[str, bool]:
        """Check that all referenced artifacts are readable."""
        results = {
            "decision_traces_readable": True,
            "rejection_artifacts_readable": True,
        }
        
        for daily in run.daily_results:
            # Check decision traces
            for room, trace_path in daily.decision_traces.items():
                path = Path(trace_path)
                if not path.exists() or not path.is_file():
                    results["decision_traces_readable"] = False
                else:
                    try:
                        json.loads(path.read_text())
                    except Exception:
                        results["decision_traces_readable"] = False
            
            # Check rejection artifacts
            for room, artifact_path in daily.rejection_artifacts.items():
                path = Path(artifact_path)
                if not path.exists() or not path.is_file():
                    results["rejection_artifacts_readable"] = False
                else:
                    try:
                        json.loads(path.read_text())
                    except Exception:
                        results["rejection_artifacts_readable"] = False
        
        return results
    
    def _check_leakage_audit_present(self, run: ValidationRun) -> bool:
        """Check if leakage audit artifact is present for all daily results."""
        for daily in run.daily_results:
            gate_summary = daily.gate_stack_summary if isinstance(daily.gate_stack_summary, dict) else {}
            leakage_audit = gate_summary.get("leakage_audit", {})
            if not isinstance(leakage_audit, dict) or not leakage_audit:
                return False
        return True
    
    def _check_leakage_audit_pass(self, run: ValidationRun) -> bool:
        """Check if leakage audit passes for all daily results."""
        for daily in run.daily_results:
            gate_summary = daily.gate_stack_summary if isinstance(daily.gate_stack_summary, dict) else {}
            leakage_pass = bool(gate_summary.get("leakage_audit_pass", gate_summary.get("leakage_pass", False)))
            if not leakage_pass:
                return False
        return True
    
    def _check_all_hard_gates_pass(self, run: ValidationRun) -> bool:
        """Check if all hard gates pass across all daily results."""
        for daily in run.daily_results:
            gate_summary = daily.gate_stack_summary if isinstance(daily.gate_stack_summary, dict) else {}
            hard_pass = bool(gate_summary.get("all_hard_gates_pass", False))
            if not hard_pass:
                return False
        return True
    
    def _check_timeline_gates_pass(self, run: ValidationRun, threshold: float = 0.80) -> bool:
        """Check if timeline gates pass in at least threshold % of split-seed cells."""
        total = 0
        passed = 0
        for daily in run.daily_results:
            gate_summary = daily.gate_stack_summary if isinstance(daily.gate_stack_summary, dict) else {}
            timeline_passed = int(gate_summary.get("timeline_gates_passed", 0))
            timeline_total = int(gate_summary.get("timeline_gates_total", 0))
            if timeline_total > 0:
                total += timeline_total
                passed += timeline_passed
        if total == 0:
            return False
        return (passed / total) >= threshold
    
    def _compute_signoff_decision(
        self,
        run: ValidationRun,
        compliance_checklist: Dict[str, bool],
        determinism_score: float,
        determinism_issues: List[str],
        gate_reason_summary: Dict[str, int],
        strict_config: Optional[StrictEvaluationConfig] = None,
    ) -> Tuple[SignoffDecision, List[str], List[str], List[str], str]:
        """
        Compute strict signoff decision (Lane D1).
        
        ALL compliance checks must pass for PASS decision.
        force=True allows artifact generation but never changes PASS decision.
        
        Returns:
            (decision, reasons, blocking_issues, residual_risks, recommended_stage)
        """
        config = strict_config or StrictEvaluationConfig()
        reasons: List[str] = []
        blocking: List[str] = []
        risks: List[str] = []
        
        # === BLOCKING ISSUES (hard fails - always result in FAIL) ===
        
        # 1. Split-seed matrix coverage (must have 4 splits × 3 seeds = 12 cells)
        expected_cells = len(config.splits) * len(config.seeds)
        if not compliance_checklist.get("split_seed_matrix_complete", False):
            blocking.append(f"Incomplete split-seed matrix: expected {expected_cells} cells (4 splits × 3 seeds)")
        
        # 2. Leakage audit (mandatory)
        if not compliance_checklist.get("leakage_audit_present", False):
            blocking.append("Missing mandatory leakage audit artifact")
        if not compliance_checklist.get("leakage_audit_pass", False):
            blocking.append("Leakage audit failed - potential data leakage detected")
        
        # 3. Hard gates (safety critical)
        if not compliance_checklist.get("all_hard_gates_pass", False):
            blocking.append("Hard safety gates failed - safety criteria not met")
        
        # 4. Data integrity
        if not compliance_checklist.get("manifest_hash_consistent", False):
            blocking.append("Manifest hash inconsistency - data integrity issue")
        
        # 5. Artifact integrity (files must be readable)
        if not compliance_checklist.get("all_decision_traces_readable", False):
            blocking.append("Decision trace files missing or unreadable")
        if not compliance_checklist.get("all_rejection_artifacts_readable", False):
            blocking.append("Rejection artifact files missing or unreadable")
        
        # === NON-BLOCKING REASONS (may result in CONDITIONAL) ===
        
        # Timeline gates pass rate
        if not compliance_checklist.get("timeline_gates_pass", False):
            reasons.append(f"Timeline gates pass rate below {config.timeline_gate_pass_rate:.0%} threshold")
        
        # Determinism score
        if not compliance_checklist.get("determinism_score_above_80", False):
            reasons.append(f"Determinism score {determinism_score:.1%} below 80% threshold")
            if determinism_issues:
                risks.extend([f"Determinism: {issue}" for issue in determinism_issues[:3]])
        
        # Minimum days
        if not compliance_checklist.get("minimum_7_days", False):
            reasons.append("Less than 7 days of results")
        
        # Trace/gate coverage
        if not compliance_checklist.get("all_rooms_have_decision_traces", False):
            reasons.append("Some rooms missing decision traces")
        if not compliance_checklist.get("no_missing_gate_reasons", False):
            reasons.append("Some rejected rooms missing rejection artifacts")
        
        # Collect residual risks from gate reasons
        for reason_code, count in gate_reason_summary.items():
            if count > 0:
                risks.append(f"Gate reason '{reason_code}' occurred {count} times")
        
        # === DECISION LOGIC (strict fail-closed) ===
        # PASS requires: no blocking issues AND all compliance checks pass
        all_checks_pass = all(compliance_checklist.values())
        
        if blocking or not all_checks_pass:
            decision = SignoffDecision.FAIL
            recommended_stage = "not_ready"
        elif reasons:
            decision = SignoffDecision.CONDITIONAL
            recommended_stage = "internal_ready"
        else:
            decision = SignoffDecision.PASS
            recommended_stage = "external_ready"
        
        return decision, reasons, blocking, risks, recommended_stage
    
    def _save_run(self, run: ValidationRun) -> None:
        """Save validation run to disk."""
        run_path = self.runs_dir / f"{run.run_id}.json"
        run_path.write_text(json.dumps(run.to_dict(), indent=2))
    
    def _load_run(self, run_id: str) -> Optional[ValidationRun]:
        """Load validation run from disk."""
        run_path = self.runs_dir / f"{run_id}.json"
        if not run_path.exists():
            return None
        
        data = json.loads(run_path.read_text())
        
        # Reconstruct ValidationRun
        run = ValidationRun(
            run_id=data["run_id"],
            elder_id=data["elder_id"],
            start_date=data["start_date"],
            duration_days=data["duration_days"],
            manifest_path=data["manifest_path"],
            manifest_hash=data["manifest_hash"],
            status=ValidationRunStatus(data["status"]),
            config=data.get("config", {}),
            final_summary=data.get("final_summary"),
        )
        
        # Reconstruct daily results
        for daily_data in data.get("daily_results", []):
            daily = DailyRunResult(
                date=daily_data["date"],
                run_timestamp=daily_data["run_timestamp"],
                manifest_hash=daily_data["manifest_hash"],
                rooms_trained=daily_data["rooms_trained"],
                rooms_promoted=daily_data["rooms_promoted"],
                rooms_rejected=daily_data["rooms_rejected"],
                decision_traces=daily_data.get("decision_traces", {}),
                rejection_artifacts=daily_data.get("rejection_artifacts", {}),
                gate_stack_summary=daily_data.get("gate_stack_summary", {}),
            )
            run.daily_results.append(daily)
        
        return run
    
    def _save_signoff_pack(self, pack: SignoffPack) -> None:
        """Save signoff pack to disk."""
        pack_dir = self.runs_dir / pack.run_id
        pack_dir.mkdir(exist_ok=True)
        
        # Save JSON
        json_path = pack_dir / "signoff_pack.json"
        json_path.write_text(json.dumps(pack.to_dict(), indent=2))
        
        # Save markdown report
        md_path = pack_dir / "signoff_report.md"
        md_path.write_text(pack.generate_markdown_report())
        
        logger.info(f"Saved signoff pack to {pack_dir}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="PR-5: Controlled Validation Run + Signoff Pack"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start a new validation run")
    start_parser.add_argument("--elder-id", required=True, help="Elder/resident ID")
    start_parser.add_argument("--duration-days", type=int, default=7, help="Run duration")
    start_parser.add_argument("--manifest", required=True, help="Path to manifest file")
    start_parser.add_argument("--runs-dir", default="validation_runs", help="Runs directory")
    
    # Record command
    record_parser = subparsers.add_parser("record", help="Record daily result")
    record_parser.add_argument("--run-id", required=True, help="Run ID")
    record_parser.add_argument("--result-file", required=True, help="Daily result JSON file")
    record_parser.add_argument("--runs-dir", default="validation_runs", help="Runs directory")
    
    # Finalize command
    finalize_parser = subparsers.add_parser("finalize", help="Finalize run and generate signoff pack")
    finalize_parser.add_argument("--run-id", required=True, help="Run ID")
    finalize_parser.add_argument("--runs-dir", default="validation_runs", help="Runs directory")
    finalize_parser.add_argument("--force", action="store_true", help="Allow finalization even if compliance checks fail")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    logging.basicConfig(level=logging.INFO)
    
    manager = ValidationRunManager(runs_dir=Path(args.runs_dir))
    
    if args.command == "start":
        run = manager.create_run(
            elder_id=args.elder_id,
            duration_days=args.duration_days,
            manifest_path=args.manifest,
        )
        print(f"Created validation run: {run.run_id}")
        print(f"Manifest hash: {run.manifest_hash}")
        return 0
    
    elif args.command == "record":
        result_data = json.loads(Path(args.result_file).read_text())
        result = DailyRunResult(**result_data)
        manager.record_daily_result(args.run_id, result)
        print(f"Recorded daily result for {args.run_id}")
        return 0
    
    elif args.command == "finalize":
        try:
            pack = manager.finalize_run(args.run_id, force=args.force)
            print(f"Finalized validation run: {args.run_id}")
            print(f"Determinism score: {pack.determinism_score:.1%}")
            print(f"Compliance passed: {all(pack.compliance_checklist.values())}")
            return 0
        except RuntimeError as e:
            print(f"❌ {e}")
            print("Use --force to override compliance checks (not recommended for production)")
            return 1
    
    return 1


if __name__ == "__main__":
    exit(main())
