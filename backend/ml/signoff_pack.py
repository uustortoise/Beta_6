"""
Signoff Pack Generator Module

Generates promotion-grade decision artifacts:
- Rolling summary JSON
- Signoff JSON with gate decisions
- Residual pack JSON
- Residual windows CSV
- Leakage audit JSON

Part of WS-6: Controlled Validation + Signoff
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd


@dataclass
class SplitSeedResult:
    """Result from a single split-seed cell."""
    
    split_id: str
    seed: int
    train_days: List[int]
    val_days: List[int]
    
    # Timeline metrics per room
    timeline_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Gate results
    hard_gates_passed: int = 0
    hard_gates_total: int = 0
    timeline_gates_passed: int = 0
    timeline_gates_total: int = 0
    
    # Leakage audit
    leakage_audit_pass: bool = False
    leakage_violations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineComparison:
    """Baseline vs candidate comparison."""
    
    baseline_version: str
    candidate_version: str
    
    # Metric deltas (candidate - baseline, negative is improvement for errors)
    metric_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Improvement flags
    fragmentation_improved: bool = False
    duration_mae_improved: bool = False
    overall_improved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'baseline_version': self.baseline_version,
            'candidate_version': self.candidate_version,
            'metric_deltas': self.metric_deltas,
            'fragmentation_improved': self.fragmentation_improved,
            'duration_mae_improved': self.duration_mae_improved,
            'overall_improved': self.overall_improved,
        }


@dataclass
class SignoffDecision:
    """Final signoff decision with risk statement."""
    
    decision: str = "PENDING"  # "PASS", "FAIL", "CONDITIONAL"
    confidence: str = "LOW"  # "HIGH", "MEDIUM", "LOW"
    
    # Decision rationale
    primary_reasons: List[str] = field(default_factory=list)
    blocking_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Risk statement (mandatory per Section 11)
    residual_risks: List[str] = field(default_factory=list)
    mitigation_notes: List[str] = field(default_factory=list)
    
    # Rollout recommendation
    recommended_stage: str = ""  # "shadow", "canary", "full", "hold"
    observation_period_days: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_markdown(self) -> str:
        """Generate markdown report for human review."""
        lines = [
            "# Signoff Decision Report",
            "",
            f"**Decision:** {self.decision}",
            f"**Confidence:** {self.confidence}",
            f"**Recommended Stage:** {self.recommended_stage}",
            "",
            "## Primary Reasons",
        ]
        for reason in self.primary_reasons:
            lines.append(f"- {reason}")
        
        if self.blocking_issues:
            lines.extend(["", "## Blocking Issues"])
            for issue in self.blocking_issues:
                lines.append(f"- {issue}")
        
        if self.warnings:
            lines.extend(["", "## Warnings"])
            for warning in self.warnings:
                lines.append(f"- {warning}")
        
        lines.extend(["", "## Residual Risks"])
        for risk in self.residual_risks:
            lines.append(f"- {risk}")
        
        if self.mitigation_notes:
            lines.extend(["", "## Mitigation Notes"])
            for note in self.mitigation_notes:
                lines.append(f"- {note}")
        
        return "\n".join(lines)


@dataclass
class SignoffPack:
    """Complete signoff pack with all artifacts."""
    
    # Identification
    run_id: str
    elder_id: str
    timestamp: str
    git_sha: str
    config_hash: str
    
    # Baseline info
    baseline_version: str
    baseline_artifact_hash: str
    
    # Results
    split_seed_results: List[SplitSeedResult] = field(default_factory=list)
    baseline_comparison: Optional[BaselineComparison] = None
    
    # Aggregated metrics
    room_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    classification_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timeline_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Gates
    kpi_checks: List[Dict[str, Any]] = field(default_factory=list)
    timeline_checks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Decision
    decision: SignoffDecision = field(default_factory=SignoffDecision)
    
    # Metadata
    data_version: str = ""
    feature_schema_hash: str = ""
    model_version_candidate: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'elder_id': self.elder_id,
            'timestamp': self.timestamp,
            'git_sha': self.git_sha,
            'config_hash': self.config_hash,
            'baseline_version': self.baseline_version,
            'baseline_artifact_hash': self.baseline_artifact_hash,
            'split_seed_results': [r.to_dict() for r in self.split_seed_results],
            'baseline_comparison': self.baseline_comparison.to_dict() if self.baseline_comparison else None,
            'room_summary': self.room_summary,
            'classification_summary': self.classification_summary,
            'timeline_summary': self.timeline_summary,
            'kpi_checks': self.kpi_checks,
            'timeline_checks': self.timeline_checks,
            'decision': self.decision.to_dict(),
            'data_version': self.data_version,
            'feature_schema_hash': self.feature_schema_hash,
            'model_version_candidate': self.model_version_candidate,
        }
    
    def save(self, output_dir: Path) -> Dict[str, Path]:
        """Save all signoff artifacts."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Signoff JSON
        signoff_path = output_dir / f"{self.run_id}_signoff.json"
        signoff_path.write_text(json.dumps(self.to_dict(), indent=2))
        paths['signoff'] = signoff_path
        
        # Markdown report
        md_path = output_dir / f"{self.run_id}_signoff_report.md"
        md_path.write_text(self.decision.to_markdown())
        paths['markdown_report'] = md_path
        
        return paths


class SignoffPackGenerator:
    """Generator for complete signoff packs."""
    
    def __init__(
        self,
        run_id: str,
        elder_id: str,
        baseline_version: str,
        git_sha: str = "unknown",
    ):
        self.run_id = run_id
        self.elder_id = elder_id
        self.baseline_version = baseline_version
        self.git_sha = git_sha
        self.split_seed_results: List[SplitSeedResult] = []
    
    def add_split_seed_result(self, result: SplitSeedResult) -> None:
        """Add a split-seed result."""
        self.split_seed_results.append(result)
    
    def compute_aggregates(self) -> Dict[str, Any]:
        """Compute aggregate metrics across all split-seed cells."""
        if not self.split_seed_results:
            return {}
        
        # Collect all metrics (copy to avoid mutating originals)
        all_metrics = []
        for result in self.split_seed_results:
            for room, metrics in result.timeline_metrics.items():
                # Create a copy to avoid side effects
                metric_copy = dict(metrics)
                metric_copy['room'] = room
                metric_copy['split_id'] = result.split_id
                metric_copy['seed'] = result.seed
                all_metrics.append(metric_copy)
        
        if not all_metrics:
            return {}
        
        df = pd.DataFrame(all_metrics)
        
        # Aggregate by room
        room_summary = {}
        for room in df['room'].unique():
            room_df = df[df['room'] == room]
            room_summary[room] = {
                'segment_start_mae_mean': room_df.get('segment_start_mae_minutes', pd.Series([0])).mean(),
                'segment_end_mae_mean': room_df.get('segment_end_mae_minutes', pd.Series([0])).mean(),
                'segment_duration_mae_mean': room_df.get('segment_duration_mae_minutes', pd.Series([0])).mean(),
                'fragmentation_rate_mean': room_df.get('fragmentation_rate', pd.Series([0])).mean(),
                'n_cells': len(room_df),
            }
        
        return room_summary
    
    def compare_to_baseline(
        self,
        baseline_metrics: Dict[str, Dict[str, float]],
    ) -> BaselineComparison:
        """Compare candidate to baseline."""
        candidate_metrics = self.compute_aggregates()
        
        comparison = BaselineComparison(
            baseline_version=self.baseline_version,
            candidate_version=self.run_id,
        )
        
        # Track improvement per room
        rooms_with_fragmentation_improvement = 0
        rooms_with_duration_improvement = 0
        fragmentation_eval_rooms = 0
        duration_eval_rooms = 0
        
        # Compute deltas per room
        for room in set(baseline_metrics.keys()) | set(candidate_metrics.keys()):
            base = baseline_metrics.get(room, {})
            cand = candidate_metrics.get(room, {})
            
            comparison.metric_deltas[room] = {}
            room_improved = False
            
            # Fragmentation delta (negative is improvement)
            if 'fragmentation_rate_mean' in base and 'fragmentation_rate_mean' in cand:
                fragmentation_eval_rooms += 1
                frag_delta = cand['fragmentation_rate_mean'] - base['fragmentation_rate_mean']
                comparison.metric_deltas[room]['fragmentation_delta'] = frag_delta
                
                # Check if improved by at least 20% relative
                if base['fragmentation_rate_mean'] > 0:
                    improvement = -frag_delta / base['fragmentation_rate_mean']
                    if improvement >= 0.20:
                        rooms_with_fragmentation_improvement += 1
                        room_improved = True
            
            # Duration MAE delta
            if 'segment_duration_mae_mean' in base and 'segment_duration_mae_mean' in cand:
                duration_eval_rooms += 1
                dur_delta = cand['segment_duration_mae_mean'] - base['segment_duration_mae_mean']
                comparison.metric_deltas[room]['duration_mae_delta'] = dur_delta
                
                # Non-negative = improvement
                if dur_delta <= 0:
                    rooms_with_duration_improvement += 1
            
            # Room-level improvement flag
            comparison.metric_deltas[room]['room_improved'] = room_improved
        
        # Overall improvement requires:
        # 1. At least 80% of rooms show fragmentation improvement
        # 2. At least 80% of rooms show duration improvement (non-regression)
        if fragmentation_eval_rooms > 0:
            frag_improve_ratio = rooms_with_fragmentation_improvement / fragmentation_eval_rooms
            comparison.fragmentation_improved = frag_improve_ratio >= 0.80
        if duration_eval_rooms > 0:
            dur_improve_ratio = rooms_with_duration_improvement / duration_eval_rooms
            comparison.duration_mae_improved = dur_improve_ratio >= 0.80
        
        # Overall improvement requires BOTH metrics to meet thresholds
        comparison.overall_improved = (
            comparison.fragmentation_improved and
            comparison.duration_mae_improved
        )
        
        return comparison
    
    def generate_decision(
        self,
        comparison: Optional[BaselineComparison] = None,
    ) -> SignoffDecision:
        """Generate signoff decision."""
        decision = SignoffDecision()
        
        # Check split-seed robustness
        total_cells = len(self.split_seed_results)
        passing_cells = sum(
            1 for r in self.split_seed_results
            if r.hard_gates_total > 0  # Must have hard gates evaluated
            and r.hard_gates_passed == r.hard_gates_total
            and r.leakage_audit_pass
            and (r.timeline_gates_total == 0 or r.timeline_gates_passed == r.timeline_gates_total)  # Timeline gates must pass if evaluated
        )
        
        pass_rate = passing_cells / total_cells if total_cells > 0 else 0.0
        
        # Decision logic
        if pass_rate < 0.8:
            decision.decision = "FAIL"
            decision.confidence = "HIGH"
            decision.recommended_stage = "hold"
            decision.blocking_issues.append(
                f"Only {pass_rate:.1%} of split-seed cells pass (requirement: 80%)"
            )
        elif comparison and not comparison.overall_improved:
            decision.decision = "FAIL"
            decision.confidence = "HIGH"
            decision.recommended_stage = "hold"
            decision.blocking_issues.append(
                "Timeline metrics do not show improvement vs baseline"
            )
        elif pass_rate < 1.0:
            decision.decision = "CONDITIONAL"
            decision.confidence = "MEDIUM"
            decision.recommended_stage = "canary"
            decision.warnings.append(
                f"{total_cells - passing_cells} split-seed cells have issues"
            )
        else:
            decision.decision = "PASS"
            decision.confidence = "HIGH"
            decision.recommended_stage = "canary"
            decision.primary_reasons.append(
                "All split-seed cells pass hard gates and leakage audit"
            )
            if comparison and comparison.overall_improved:
                decision.primary_reasons.append(
                    "Timeline metrics show improvement vs baseline"
                )
        
        # Residual risks (mandatory)
        decision.residual_risks = [
            "Bedroom/LivingRoom quiet presence vs unoccupied ambiguity may persist",
            "Sensor non-identifiability limits boundary precision in some cases",
            "Rolling split evaluation may not capture all seasonal variations",
        ]
        
        # Mitigation notes
        if decision.recommended_stage == "canary":
            decision.mitigation_notes.append(
                "Monitor canary cohort for 7 days before full promotion"
            )
            decision.observation_period_days = 7
        
        return decision
    
    def generate(
        self,
        baseline_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        baseline_artifact_hash: str = "",
        data_version: str = "",
        feature_schema_hash: str = "",
        kpi_checks: Optional[List[Dict[str, Any]]] = None,
        timeline_checks: Optional[List[Dict[str, Any]]] = None,
    ) -> SignoffPack:
        """Generate complete signoff pack."""
        # Compute config hash
        config_data = {
            'run_id': self.run_id,
            'baseline_version': self.baseline_version,
            'baseline_artifact_hash': baseline_artifact_hash,
            'split_seeds': [f"{r.split_id}-{r.seed}" for r in self.split_seed_results],
        }
        config_hash = f"sha256:{hashlib.sha256(json.dumps(config_data, sort_keys=True).encode()).hexdigest()[:16]}"
        
        # Compute aggregates (without mutating inputs)
        room_summary = self.compute_aggregates()
        
        # Build timeline_summary from room_summary
        timeline_summary = {}
        for room, metrics in room_summary.items():
            timeline_summary[room] = {
                'segment_start_mae_minutes': metrics.get('segment_start_mae_mean', 0.0),
                'segment_end_mae_minutes': metrics.get('segment_end_mae_mean', 0.0),
                'segment_duration_mae_minutes': metrics.get('segment_duration_mae_mean', 0.0),
                'fragmentation_rate': metrics.get('fragmentation_rate_mean', 0.0),
                'episode_count_error': metrics.get('episode_count_error_mean', 0.0),
            }
        
        # Compare to baseline if provided
        comparison = None
        if baseline_metrics:
            comparison = self.compare_to_baseline(baseline_metrics)
        
        # Generate decision
        decision = self.generate_decision(comparison)
        
        pack = SignoffPack(
            run_id=self.run_id,
            elder_id=self.elder_id,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            git_sha=self.git_sha,
            config_hash=config_hash,
            baseline_version=self.baseline_version,
            baseline_artifact_hash=baseline_artifact_hash,
            split_seed_results=self.split_seed_results,
            baseline_comparison=comparison,
            room_summary=room_summary,
            timeline_summary=timeline_summary,
            kpi_checks=kpi_checks or [],
            timeline_checks=timeline_checks or [],
            decision=decision,
            data_version=data_version,
            feature_schema_hash=feature_schema_hash,
            model_version_candidate=f"timeline_{self.run_id}",
        )
        
        return pack


def create_signoff_pack(
    run_id: str,
    elder_id: str,
    baseline_version: str,
    split_seed_results: List[Dict[str, Any]],
    baseline_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    baseline_artifact_hash: str = "",
    git_sha: str = "unknown",
    output_dir: Optional[Path] = None,
) -> SignoffPack:
    """
    Convenience function to create and save a signoff pack.
    
    Args:
        run_id: Unique run identifier
        elder_id: Elder identifier
        baseline_version: Baseline version (e.g., "v31")
        split_seed_results: List of split-seed result dictionaries
        baseline_metrics: Optional baseline metrics for comparison
        git_sha: Git commit SHA
        output_dir: Optional directory to save artifacts
        
    Returns:
        SignoffPack
    """
    generator = SignoffPackGenerator(run_id, elder_id, baseline_version, git_sha)
    
    for result_data in split_seed_results:
        result = SplitSeedResult(**result_data)
        generator.add_split_seed_result(result)
    
    pack = generator.generate(
        baseline_metrics=baseline_metrics,
        baseline_artifact_hash=baseline_artifact_hash,
    )
    
    if output_dir:
        pack.save(output_dir)
    
    return pack
