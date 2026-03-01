"""
Leakage Audit Module

Generates mandatory leakage audit artifacts for each split/seed run.
Ensures train-only fit policy and no target-derived leakage.

Part of WS-2/WS-3/WS-6: Canonical Evaluation Contract
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import numpy as np


@dataclass
class SplitLeakageAudit:
    """Leakage audit record for a single train/val split."""
    
    split_id: str
    train_days: List[int]
    val_days: List[int]
    
    # Fit policy checks - default FAIL (must be explicitly marked as passed)
    scaler_fit_on_train_only: bool = False
    imputer_fit_on_train_only: bool = False
    feature_stats_fit_on_train_only: bool = False
    calibrator_fit_on_train_only: bool = False
    
    # Window/statistics policy - default FAIL
    temporal_window_causal_only: bool = False
    no_centered_windows: bool = False
    no_future_derived_features: bool = False
    
    # Target leakage - default FAIL
    no_target_stats_from_val: bool = False
    no_decoder_threshold_tuning_on_val: bool = False
    
    # Calibration policy - default FAIL
    calibration_uses_train_slice_only: bool = False
    
    # Evidence
    train_sample_count: int = 0
    val_sample_count: int = 0
    feature_names: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate audit and return any violations."""
        violations = []
        
        checks = [
            ('scaler_fit_on_train_only', 'Scaler fit on validation data'),
            ('imputer_fit_on_train_only', 'Imputer fit on validation data'),
            ('feature_stats_fit_on_train_only', 'Feature statistics fit on validation data'),
            ('calibrator_fit_on_train_only', 'Calibrator fit on validation data'),
            ('temporal_window_causal_only', 'Non-causal temporal window used'),
            ('no_centered_windows', 'Centered window (looks forward) used'),
            ('no_future_derived_features', 'Future-derived features detected'),
            ('no_target_stats_from_val', 'Target statistics from validation used'),
            ('no_decoder_threshold_tuning_on_val', 'Decoder tuned on validation'),
            ('calibration_uses_train_slice_only', 'Calibration uses validation data'),
        ]
        
        for field_name, violation_msg in checks:
            if not getattr(self, field_name):
                violations.append(violation_msg)
        
        return violations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LeakageAuditReport:
    """
    Complete leakage audit report for a run.
    
    This artifact is mandatory per the Canonical Evaluation Contract.
    """
    
    # Run identification
    run_id: str
    elder_id: str
    seed: int
    timestamp: str
    git_sha: str
    
    # Overall audit results
    splits: List[SplitLeakageAudit] = field(default_factory=list)
    all_splits_pass: bool = True
    violations: List[str] = field(default_factory=list)
    
    # Evidence hashes
    train_manifest_hash: str = ""
    val_manifest_hash: str = ""
    code_version_hash: str = ""
    
    # Auditor metadata
    auditor_version: str = "1.0.0"
    audit_tool: str = "ml.leakage_audit"
    
    def add_split(self, audit: SplitLeakageAudit) -> None:
        """Add a split audit and update overall status."""
        self.splits.append(audit)
        split_violations = audit.validate()
        
        if split_violations:
            self.all_splits_pass = False
            self.violations.extend([
                f"[{audit.split_id}] {v}" for v in split_violations
            ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'run_id': self.run_id,
            'elder_id': self.elder_id,
            'seed': self.seed,
            'timestamp': self.timestamp,
            'git_sha': self.git_sha,
            'splits': [s.to_dict() for s in self.splits],
            'all_splits_pass': self.all_splits_pass,
            'violations': self.violations,
            'train_manifest_hash': self.train_manifest_hash,
            'val_manifest_hash': self.val_manifest_hash,
            'code_version_hash': self.code_version_hash,
            'auditor_version': self.auditor_version,
            'audit_tool': self.audit_tool,
        }
    
    def save(self, path: Path) -> None:
        """Save audit report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
    
    @classmethod
    def load(cls, path: Path) -> "LeakageAuditReport":
        """Load audit report from JSON file."""
        data = json.loads(Path(path).read_text())
        splits = [SplitLeakageAudit(**s) for s in data.pop('splits', [])]
        report = cls(**data)
        report.splits = splits
        return report


class LeakageAuditor:
    """
    Auditor that tracks and verifies leakage control during training/evaluation.
    
    Usage:
        auditor = LeakageAuditor(run_id="run_001", elder_id="HK0011", seed=11)
        
        # For each split
        split_audit = auditor.create_split_audit(split_id="4->5", train_days=[4], val_days=[5])
        
        # Mark checks as you verify them
        auditor.mark_scaler_fit_on_train_only()
        auditor.mark_no_centered_windows()
        ...
        
        # Finalize split
        auditor.finalize_split(split_audit)
        
        # Save report
        report = auditor.generate_report()
        report.save("leakage_audit.json")
    """
    
    def __init__(
        self,
        run_id: str,
        elder_id: str,
        seed: int,
        git_sha: str = "unknown",
    ):
        self.run_id = run_id
        self.elder_id = elder_id
        self.seed = seed
        self.git_sha = git_sha
        self._current_split: Optional[SplitLeakageAudit] = None
        self._splits: List[SplitLeakageAudit] = []
    
    def create_split_audit(
        self,
        split_id: str,
        train_days: List[int],
        val_days: List[int],
    ) -> SplitLeakageAudit:
        """Create a new split audit context."""
        audit = SplitLeakageAudit(
            split_id=split_id,
            train_days=train_days,
            val_days=val_days,
        )
        self._current_split = audit
        return audit
    
    def mark_scaler_fit_on_train_only(self, passed: bool = True) -> None:
        """Mark scaler fit policy check."""
        if self._current_split:
            self._current_split.scaler_fit_on_train_only = passed
    
    def mark_imputer_fit_on_train_only(self, passed: bool = True) -> None:
        """Mark imputer fit policy check."""
        if self._current_split:
            self._current_split.imputer_fit_on_train_only = passed
    
    def mark_feature_stats_fit_on_train_only(self, passed: bool = True) -> None:
        """Mark feature statistics fit policy check."""
        if self._current_split:
            self._current_split.feature_stats_fit_on_train_only = passed
    
    def mark_calibrator_fit_on_train_only(self, passed: bool = True) -> None:
        """Mark calibrator fit policy check."""
        if self._current_split:
            self._current_split.calibrator_fit_on_train_only = passed
    
    def mark_temporal_window_causal_only(self, passed: bool = True) -> None:
        """Mark temporal window causality check."""
        if self._current_split:
            self._current_split.temporal_window_causal_only = passed
    
    def mark_no_centered_windows(self, passed: bool = True) -> None:
        """Mark no centered windows check."""
        if self._current_split:
            self._current_split.no_centered_windows = passed
    
    def mark_no_future_derived_features(self, passed: bool = True) -> None:
        """Mark no future-derived features check."""
        if self._current_split:
            self._current_split.no_future_derived_features = passed
    
    def mark_no_target_stats_from_val(self, passed: bool = True) -> None:
        """Mark no target statistics from validation check."""
        if self._current_split:
            self._current_split.no_target_stats_from_val = passed
    
    def mark_no_decoder_threshold_tuning_on_val(self, passed: bool = True) -> None:
        """Mark no decoder threshold tuning on validation check."""
        if self._current_split:
            self._current_split.no_decoder_threshold_tuning_on_val = passed
    
    def mark_calibration_uses_train_slice_only(self, passed: bool = True) -> None:
        """Mark calibration uses train slice only check."""
        if self._current_split:
            self._current_split.calibration_uses_train_slice_only = passed
    
    def set_sample_counts(self, train_count: int, val_count: int) -> None:
        """Set sample counts for evidence."""
        if self._current_split:
            self._current_split.train_sample_count = train_count
            self._current_split.val_sample_count = val_count
    
    def set_feature_names(self, feature_names: List[str]) -> None:
        """Set feature names for evidence."""
        if self._current_split:
            self._current_split.feature_names = feature_names
    
    def finalize_split(self, audit: Optional[SplitLeakageAudit] = None) -> None:
        """Finalize current split audit."""
        split = audit or self._current_split
        if split:
            self._splits.append(split)
            self._current_split = None
    
    def generate_report(
        self,
        train_manifest_hash: str = "",
        val_manifest_hash: str = "",
    ) -> LeakageAuditReport:
        """Generate complete audit report."""
        # Compute code version hash from key leakage-related files
        code_hash = self._compute_code_hash()
        
        report = LeakageAuditReport(
            run_id=self.run_id,
            elder_id=self.elder_id,
            seed=self.seed,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            git_sha=self.git_sha,
            splits=self._splits,
            train_manifest_hash=train_manifest_hash,
            val_manifest_hash=val_manifest_hash,
            code_version_hash=code_hash,
        )
        
        # Fail-closed: require at least one audited split
        if not self._splits:
            report.all_splits_pass = False
            report.violations.append("[report] No splits audited")
        
        # Validate all splits
        for split in self._splits:
            violations = split.validate()
            if violations:
                report.all_splits_pass = False
                report.violations.extend([f"[{split.split_id}] {v}" for v in violations])
        
        return report
    
    def _compute_code_hash(self) -> str:
        """Compute hash of leakage-critical code files."""
        try:
            import inspect
            from . import timeline_targets, transformer_timeline_heads, calibration
            
            files_to_hash = [
                inspect.getfile(timeline_targets),
                inspect.getfile(transformer_timeline_heads),
                inspect.getfile(calibration),
            ]
            
            hasher = hashlib.sha256()
            for filepath in files_to_hash:
                content = Path(filepath).read_bytes()
                hasher.update(content)
            
            return f"sha256:{hasher.hexdigest()[:16]}"
        except Exception:
            return "unknown"


def create_leakage_audit_report(
    run_id: str,
    elder_id: str,
    seed: int,
    splits_data: List[Dict[str, Any]],
    git_sha: str = "unknown",
    output_path: Optional[Path] = None,
) -> LeakageAuditReport:
    """
    Convenience function to create a leakage audit report.
    
    Args:
        run_id: Unique run identifier
        elder_id: Elder identifier
        seed: Random seed
        splits_data: List of split audit data dictionaries
        git_sha: Git commit SHA
        output_path: Optional path to save report
        
    Returns:
        LeakageAuditReport
    """
    auditor = LeakageAuditor(run_id, elder_id, seed, git_sha)
    
    for split_data in splits_data:
        audit = SplitLeakageAudit(**split_data)
        auditor._splits.append(audit)
    
    report = auditor.generate_report()
    
    if output_path:
        report.save(output_path)
    
    return report
