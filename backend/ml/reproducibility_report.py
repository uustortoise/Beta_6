"""
Item 13: Deterministic Retrain Reproducibility Report

Provides evidence of reproducibility by tracking fingerprints, policy hashes,
code versions, and outcome parity across reruns.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import sys

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    """Return timezone-aware UTC timestamp as ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DataFingerprint:
    """Immutable fingerprint of input data."""
    elder_id: str
    room_names: Tuple[str, ...]
    total_samples: int
    observed_days: int
    raw_data_hash: str  # Hash of raw data file contents
    timestamp_range_start: Optional[str] = None
    timestamp_range_end: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "elder_id": self.elder_id,
            "room_names": list(self.room_names),
            "total_samples": self.total_samples,
            "observed_days": self.observed_days,
            "raw_data_hash": self.raw_data_hash,
            "timestamp_range_start": self.timestamp_range_start,
            "timestamp_range_end": self.timestamp_range_end,
        }
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of this fingerprint."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CodeVersion:
    """Code version information."""
    git_commit: str
    git_branch: str
    git_dirty: bool  # True if uncommitted changes exist
    python_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "git_dirty": self.git_dirty,
            "python_version": self.python_version,
        }
    
    def is_clean(self) -> bool:
        """Check if code is at a clean, reproducible state."""
        return not self.git_dirty


@dataclass
class RunOutcome:
    """Outcome of a training run."""
    promoted_rooms: List[str] = field(default_factory=list)
    rejected_rooms: List[str] = field(default_factory=list)
    gate_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "promoted_rooms": self.promoted_rooms,
            "rejected_rooms": self.rejected_rooms,
            "gate_results": self.gate_results,
            "metrics_summary": self.metrics_summary,
        }
    
    def compute_signature(self) -> str:
        """Compute deterministic signature of outcome."""
        # Sort for determinism
        promoted = sorted(self.promoted_rooms)
        rejected = sorted(self.rejected_rooms)
        
        payload = json.dumps({
            "promoted": promoted,
            "rejected": rejected,
        }, sort_keys=True, separators=(",", ":"))
        
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class ReproducibilityReport:
    """
    Comprehensive reproducibility report.
    
    This artifact proves reproducibility claims and enables
    outcome parity verification across reruns.
    """
    # Run identification
    run_id: str
    timestamp: str
    elder_id: str
    
    # Reproducibility factors
    data_fingerprint: DataFingerprint
    policy_hash: str
    code_version: CodeVersion
    random_seed: int
    
    # Outcome
    outcome: RunOutcome
    
    # Parity tracking
    prior_run_id: Optional[str] = None
    prior_run_linked: bool = False
    outcome_parity_with_prior: Optional[bool] = None
    parity_verification_hash: Optional[str] = None
    
    # Rerun detection
    is_noop_rerun: bool = False
    noop_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "elder_id": self.elder_id,
            "data_fingerprint": self.data_fingerprint.to_dict(),
            "data_fingerprint_hash": self.data_fingerprint.compute_hash(),
            "policy_hash": self.policy_hash,
            "code_version": self.code_version.to_dict(),
            "random_seed": self.random_seed,
            "outcome": self.outcome.to_dict(),
            "outcome_signature": self.outcome.compute_signature(),
            "prior_run_id": self.prior_run_id,
            "prior_run_linked": self.prior_run_linked,
            "outcome_parity_with_prior": self.outcome_parity_with_prior,
            "parity_verification_hash": self.parity_verification_hash,
            "is_noop_rerun": self.is_noop_rerun,
            "noop_reason": self.noop_reason,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save report to file."""
        filepath = Path(filepath)
        filepath.write_text(self.to_json())
        logger.info(f"Reproducibility report saved to {filepath}")
    
    def compute_composite_hash(self) -> str:
        """
        Compute composite hash of all reproducibility factors.
        
        This hash uniquely identifies a deterministic rerun configuration.
        """
        payload = json.dumps({
            "data_fingerprint": self.data_fingerprint.compute_hash(),
            "policy_hash": self.policy_hash,
            "code_version": self.code_version.to_dict(),
            "random_seed": self.random_seed,
        }, sort_keys=True, separators=(",", ":"))
        
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class ReproducibilityTracker:
    """
    Tracks reproducibility across training runs.
    
    Enables:
    - No-op detection (same inputs -> skip training)
    - Outcome parity verification
    - Prior run linkage
    """
    
    def __init__(self, history_dir: Optional[Union[str, Path]] = None):
        """
        Initialize tracker.
        
        Parameters:
        -----------
        history_dir : Path, optional
            Directory to store/load reproducibility history
        """
        self.history_dir = Path(history_dir) if history_dir else None
        self._history: Dict[str, ReproducibilityReport] = {}
        
        if self.history_dir:
            self._load_history()
    
    def create_report(
        self,
        run_id: str,
        elder_id: str,
        data_fingerprint: DataFingerprint,
        policy_hash: str,
        random_seed: int,
        outcome: RunOutcome,
    ) -> ReproducibilityReport:
        """
        Create a new reproducibility report.
        
        Parameters:
        -----------
        run_id : str
            Unique run identifier
        elder_id : str
            Elder/resident ID
        data_fingerprint : DataFingerprint
            Data fingerprint
        policy_hash : str
            Training policy hash
        random_seed : int
            Random seed used
        outcome : RunOutcome
            Run outcome
            
        Returns:
        --------
        ReproducibilityReport
        """
        code_version = get_code_version()
        
        report = ReproducibilityReport(
            run_id=run_id,
            timestamp=_utc_now_iso(),
            elder_id=elder_id,
            data_fingerprint=data_fingerprint,
            policy_hash=policy_hash,
            code_version=code_version,
            random_seed=random_seed,
            outcome=outcome,
        )
        
        # Check for prior run with same factors
        prior_run = self._find_equivalent_run(report)
        if prior_run:
            report.prior_run_id = prior_run.run_id
            report.prior_run_linked = True
            report.outcome_parity_with_prior = self._verify_outcome_parity(
                report.outcome, prior_run.outcome
            )
            report.parity_verification_hash = self._compute_parity_hash(
                report, prior_run
            )
        
        # Store in history
        self._history[run_id] = report
        if self.history_dir:
            self._save_report(report)
        
        return report
    
    def check_noop_eligibility(
        self,
        data_fingerprint: DataFingerprint,
        policy_hash: str,
        code_version: Optional[CodeVersion] = None,
    ) -> Tuple[bool, Optional[str], Optional[ReproducibilityReport]]:
        """
        Check if this configuration would be a no-op rerun.
        
        Returns:
        --------
        (is_noop, reason, prior_report) : Tuple
            is_noop: True if eligible for no-op skip
            reason: Explanation if not eligible
            prior_report: The equivalent prior run if found
        """
        if code_version is None:
            code_version = get_code_version()
        
        # Build temporary report for comparison
        temp_report = ReproducibilityReport(
            run_id="temp",
            timestamp=_utc_now_iso(),
            elder_id=data_fingerprint.elder_id,
            data_fingerprint=data_fingerprint,
            policy_hash=policy_hash,
            code_version=code_version,
            random_seed=0,  # Not relevant for comparison
            outcome=RunOutcome(),
        )
        
        prior_run = self._find_equivalent_run(temp_report)
        
        if prior_run is None:
            return False, "No equivalent prior run found", None
        
        # Check code cleanliness
        if not code_version.is_clean():
            return False, "Code has uncommitted changes (git dirty)", prior_run
        
        # All factors match - eligible for no-op
        return True, None, prior_run
    
    def _find_equivalent_run(
        self, report: ReproducibilityReport
    ) -> Optional[ReproducibilityReport]:
        """Find a prior run with identical reproducibility factors."""
        target_hash = report.compute_composite_hash()
        
        # Search in reverse chronological order (most recent first)
        sorted_runs = sorted(
            self._history.values(),
            key=lambda r: r.timestamp,
            reverse=True
        )
        
        for prior in sorted_runs:
            if prior.run_id == report.run_id:
                continue  # Skip self
            
            if prior.compute_composite_hash() == target_hash:
                return prior
        
        return None
    
    def _verify_outcome_parity(
        self, outcome1: RunOutcome, outcome2: RunOutcome
    ) -> bool:
        """Verify that two outcomes are equivalent."""
        return outcome1.compute_signature() == outcome2.compute_signature()
    
    def _compute_parity_hash(
        self, report1: ReproducibilityReport, report2: ReproducibilityReport
    ) -> str:
        """Compute verification hash for outcome parity."""
        sig1 = report1.outcome.compute_signature()
        sig2 = report2.outcome.compute_signature()
        
        payload = json.dumps({
            "run1": report1.run_id,
            "sig1": sig1,
            "run2": report2.run_id,
            "sig2": sig2,
            "parity": sig1 == sig2,
        }, sort_keys=True, separators=(",", ":"))
        
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    
    def _load_history(self) -> None:
        """Load reproducibility history from disk."""
        if not self.history_dir or not self.history_dir.exists():
            return
        
        for filepath in self.history_dir.glob("*.json"):
            try:
                data = json.loads(filepath.read_text())
                # Parse into ReproducibilityReport
                report = self._parse_report(data)
                self._history[report.run_id] = report
            except Exception as e:
                logger.warning(f"Failed to load history from {filepath}: {e}")
    
    def _save_report(self, report: ReproducibilityReport) -> None:
        """Save report to disk."""
        if not self.history_dir:
            return
        
        self.history_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.history_dir / f"{report.run_id}.json"
        report.save(filepath)
    
    def _parse_report(self, data: Dict[str, Any]) -> ReproducibilityReport:
        """Parse report from dictionary."""
        # Simplified parsing - in production would validate schema
        fp_data = data.get("data_fingerprint", {})
        data_fingerprint = DataFingerprint(
            elder_id=fp_data.get("elder_id", ""),
            room_names=tuple(fp_data.get("room_names", [])),
            total_samples=fp_data.get("total_samples", 0),
            observed_days=fp_data.get("observed_days", 0),
            raw_data_hash=fp_data.get("raw_data_hash", ""),
            timestamp_range_start=fp_data.get("timestamp_range_start"),
            timestamp_range_end=fp_data.get("timestamp_range_end"),
        )
        
        cv_data = data.get("code_version", {})
        code_version = CodeVersion(
            git_commit=cv_data.get("git_commit", "unknown"),
            git_branch=cv_data.get("git_branch", "unknown"),
            git_dirty=cv_data.get("git_dirty", True),
            python_version=cv_data.get("python_version", sys.version),
        )
        
        outcome_data = data.get("outcome", {})
        outcome = RunOutcome(
            promoted_rooms=outcome_data.get("promoted_rooms", []),
            rejected_rooms=outcome_data.get("rejected_rooms", []),
            gate_results=outcome_data.get("gate_results", {}),
            metrics_summary=outcome_data.get("metrics_summary", {}),
        )
        
        return ReproducibilityReport(
            run_id=data.get("run_id", ""),
            timestamp=data.get("timestamp", ""),
            elder_id=data.get("elder_id", ""),
            data_fingerprint=data_fingerprint,
            policy_hash=data.get("policy_hash", ""),
            code_version=code_version,
            random_seed=data.get("random_seed", 0),
            outcome=outcome,
            prior_run_id=data.get("prior_run_id"),
            prior_run_linked=data.get("prior_run_linked", False),
            outcome_parity_with_prior=data.get("outcome_parity_with_prior"),
            parity_verification_hash=data.get("parity_verification_hash"),
            is_noop_rerun=data.get("is_noop_rerun", False),
            noop_reason=data.get("noop_reason"),
        )


def get_code_version() -> CodeVersion:
    """
    Detect current code version from git.
    
    Returns:
    --------
    CodeVersion
    """
    try:
        # Get git commit
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        
        # Get branch
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        
        # Check for uncommitted changes
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = len(status) > 0
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"
        branch = "unknown"
        dirty = True
    
    return CodeVersion(
        git_commit=commit,
        git_branch=branch,
        git_dirty=dirty,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )


def compute_data_fingerprint(
    elder_id: str,
    training_data: Dict[str, Any],
    raw_data_path: Optional[str] = None,
) -> DataFingerprint:
    """
    Compute data fingerprint from training data.
    
    Parameters:
    -----------
    elder_id : str
        Elder ID
    training_data : Dict
        Dictionary of room_name -> DataFrame
    raw_data_path : str, optional
        Path to raw data file for hash computation
        
    Returns:
    --------
    DataFingerprint
    """
    import pandas as pd
    
    room_names = tuple(sorted(training_data.keys()))
    total_samples = sum(len(df) for df in training_data.values())
    
    # Compute observed days
    all_timestamps = []
    for df in training_data.values():
        if 'timestamp' in df.columns:
            all_timestamps.extend(pd.to_datetime(df['timestamp']).tolist())
    
    if all_timestamps:
        observed_days = len(set(ts.date() for ts in all_timestamps))
        timestamp_start = min(all_timestamps).isoformat()
        timestamp_end = max(all_timestamps).isoformat()
    else:
        observed_days = 0
        timestamp_start = None
        timestamp_end = None
    
    # Compute raw data hash if path provided
    if raw_data_path and Path(raw_data_path).exists():
        raw_hash = hashlib.sha256(
            Path(raw_data_path).read_bytes()
        ).hexdigest()[:16]
    else:
        # Fallback: hash of concatenated data hashes
        hashes = []
        for room, df in sorted(training_data.items()):
            hash_str = str(hash(tuple(df.values.tobytes())))
            hashes.append(f"{room}:{hash_str}")
        raw_hash = hashlib.sha256(
            "|".join(hashes).encode()
        ).hexdigest()[:16]
    
    return DataFingerprint(
        elder_id=elder_id,
        room_names=room_names,
        total_samples=total_samples,
        observed_days=observed_days,
        raw_data_hash=raw_hash,
        timestamp_range_start=timestamp_start,
        timestamp_range_end=timestamp_end,
    )


def verify_reproducibility_claim(
    report1_path: Union[str, Path],
    report2_path: Union[str, Path],
) -> Tuple[bool, str]:
    """
    Verify that two runs are reproducible (same inputs, same outcomes).
    
    Parameters:
    -----------
    report1_path : Path
        Path to first reproducibility report
    report2_path : Path
        Path to second reproducibility report
        
    Returns:
    --------
    (verified, explanation) : Tuple
        verified: True if runs are reproducible
        explanation: Human-readable explanation
    """
    try:
        report1_data = json.loads(Path(report1_path).read_text())
        report2_data = json.loads(Path(report2_path).read_text())
    except Exception as e:
        return False, f"Failed to load reports: {e}"
    
    # Check data fingerprint
    if report1_data.get("data_fingerprint_hash") != report2_data.get("data_fingerprint_hash"):
        return False, "Data fingerprints do not match"
    
    # Check policy hash
    if report1_data.get("policy_hash") != report2_data.get("policy_hash"):
        return False, "Policy hashes do not match"
    
    # Check code version
    cv1 = report1_data.get("code_version", {})
    cv2 = report2_data.get("code_version", {})
    if cv1.get("git_commit") != cv2.get("git_commit"):
        return False, f"Code versions differ: {cv1.get('git_commit')[:8]} vs {cv2.get('git_commit')[:8]}"
    
    # Check outcome parity
    sig1 = report1_data.get("outcome_signature")
    sig2 = report2_data.get("outcome_signature")
    if sig1 != sig2:
        return False, f"Outcomes differ: {sig1} vs {sig2}"
    
    return True, "Runs are reproducible - all factors match and outcomes are equivalent"
