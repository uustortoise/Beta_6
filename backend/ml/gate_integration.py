"""
PR-2: Gate Integration Module

Integrates all hardening gates into the live promotion flow:
- Pre-training gates: CoverageContract, PostGapRetention, ClassCoverage
- Post-training gates: StatisticalValidity

Generates why_rejected.json artifacts and persists gate reason codes.
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

import numpy as np
import pandas as pd

from ml.coverage_contract import CoverageContractGate, WalkForwardConfig
from ml.post_gap_retention_gate import PostGapRetentionGate
from ml.class_coverage_gate import ClassCoverageGate
from ml.statistical_validity_gate import StatisticalValidityGate
from ml.rejection_artifact import (
    RejectionArtifactBuilder, 
    RejectionCategory, 
    Severity,
)
from ml.sequence_alignment import create_labeled_sequences_strict
from ml.policy_config import TrainingPolicy

logger = logging.getLogger(__name__)


def _utc_now_iso_z() -> str:
    """Return timezone-aware UTC timestamp in ISO-8601 with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class GateEvaluationContext:
    """Context for gate evaluation."""
    room_name: str
    elder_id: str
    run_id: str
    df: pd.DataFrame
    processed_df: pd.DataFrame
    observed_days: Set[pd.Timestamp]
    seq_length: int
    policy: TrainingPolicy
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_name": self.room_name,
            "elder_id": self.elder_id,
            "run_id": self.run_id,
            "observed_day_count": len(self.observed_days),
            "raw_samples": len(self.df),
            "processed_samples": len(self.processed_df),
            "seq_length": self.seq_length,
        }


@dataclass
class PreTrainingGateResult:
    """Result from pre-training gate evaluation."""
    passes: bool
    gate_name: str
    timestamp: str
    details: Dict[str, Any]
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passes,
            "timestamp": self.timestamp,
            "details": self.details,
            "reason": self.reason,
        }


@dataclass
class GateEvaluationResult:
    """Complete gate evaluation result."""
    run_id: str
    elder_id: str
    room_name: str
    pre_training_pass: bool
    post_training_pass: bool
    overall_pass: bool
    gate_stack: List[PreTrainingGateResult]
    rejection_artifact: Optional[Dict[str, Any]] = None
    failure_stage: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        gate_reasons = [
            g.reason for g in self.gate_stack
            if g.reason
        ]
        return {
            "run_id": self.run_id,
            "elder_id": self.elder_id,
            "room_name": self.room_name,
            "room": self.room_name,  # Backward compatibility
            "pre_training_pass": self.pre_training_pass,
            "post_training_pass": self.post_training_pass,
            "overall_pass": self.overall_pass,
            "gate_pass": self.pre_training_pass,  # Legacy pre-training contract
            "gate_reasons": gate_reasons,
            "gate_stack": [g.to_dict() for g in self.gate_stack],
            "rejection_artifact": self.rejection_artifact,
            "failure_stage": self.failure_stage,
        }


class GateIntegrationPipeline:
    """
    Production-grade gate integration for training/promotion flow.
    
    Combines all hardening gates into a unified evaluation pipeline
    with proper artifact generation and logging.
    """
    
    def __init__(self, policy: Optional[TrainingPolicy] = None):
        """Initialize with training policy."""
        self.policy = policy or self._load_policy()
        self._init_gates()
    
    def _load_policy(self) -> TrainingPolicy:
        """Load policy from environment."""
        from ml.policy_config import load_policy_from_env
        return load_policy_from_env()
    
    def _init_gates(self):
        """Initialize all gates from policy."""
        evidence_profile = str(
            getattr(self.policy.release_gate, "evidence_profile", "production") or "production"
        ).strip().lower()
        pilot_relaxed_evidence = evidence_profile in {"pilot_stage_a", "pilot_stage_b"}

        # Coverage Contract Gate
        self.coverage_gate = CoverageContractGate(
            WalkForwardConfig(
                min_train_days=int(self.policy.release_gate.min_observed_days),
                valid_days=1,
                step_days=1,
                min_folds=1,
            )
        )
        
        # Post-Gap Retention Gate
        self.retention_gate = PostGapRetentionGate(
            min_retained_ratio=float(self.policy.release_gate.min_retained_sample_ratio),
            max_contiguous_segments=10,
            min_max_segment_length=100,
            min_median_segment_length=50,
        )
        
        # Class Coverage Gate
        self.class_coverage_gate = ClassCoverageGate(
            min_train_support=int(self.policy.release_gate.min_calibration_support),
            min_val_support=1 if pilot_relaxed_evidence else 5,
            min_calib_support=1 if pilot_relaxed_evidence else int(self.policy.release_gate.min_calibration_support),
            min_class_coverage_ratio=0.8,
            block_on_val_support=not pilot_relaxed_evidence,
            block_on_calib_support=not pilot_relaxed_evidence,
        )
        
        # Statistical Validity Gate
        self.statistical_gate = StatisticalValidityGate(
            min_calibration_support=int(self.policy.calibration.min_support_per_class),
            min_validation_support=10,
            min_minority_support=5,
            pilot_relaxed_evidence=pilot_relaxed_evidence,
        )

    @staticmethod
    def _parse_class_id(class_key: Any) -> Optional[int]:
        """Normalize class identifiers like 1, '1', or 'class_1'."""
        if isinstance(class_key, (int, np.integer)):
            return int(class_key)
        if class_key is None:
            return None
        text = str(class_key).strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            match = re.search(r"(-?\d+)$", text)
            if not match:
                return None
            return int(match.group(1))
    
    def evaluate_pre_training_gates(
        self,
        room_name: str,
        elder_id: str,
        df: pd.DataFrame,
        processed_df: pd.DataFrame,
        observed_days: Set[pd.Timestamp],
        seq_length: int,
        run_id: Optional[str] = None,
    ) -> GateEvaluationResult:
        """
        Evaluate all pre-training gates.
        
        Gates (in order):
        1. CoverageContractGate - sufficient observed days
        2. PostGapRetentionGate - data continuity
        3. ClassCoverageGate - class distribution across splits
        
        Returns:
            GateEvaluationResult with full gate stack and rejection artifact if failed
        """
        run_id = run_id or str(uuid.uuid4())
        gate_stack: List[PreTrainingGateResult] = []
        
        context = GateEvaluationContext(
            room_name=room_name,
            elder_id=elder_id,
            run_id=run_id,
            df=df,
            processed_df=processed_df,
            observed_days=observed_days,
            seq_length=seq_length,
            policy=self.policy,
        )
        
        # --- Gate 1: Coverage Contract ---
        observed_day_count = len(observed_days)
        coverage_result = self.coverage_gate.evaluate(room_name, observed_day_count)
        
        gate_stack.append(PreTrainingGateResult(
            passes=coverage_result.passes,
            gate_name="CoverageContractGate",
            timestamp=_utc_now_iso_z(),
            details={
                "observed_days": observed_day_count,
                "required_days": coverage_result.required_days,
                "estimated_max_folds": coverage_result.details.get("estimated_max_folds", 0),
            },
            reason=coverage_result.reason if not coverage_result.passes else None,
        ))
        
        if not coverage_result.passes:
            return self._build_failure_result(
                context=context,
                gate_stack=gate_stack,
                failure_stage="coverage_contract",
            )
        
        # --- Gate 2: Post-Gap Retention ---
        retention_result = self.retention_gate.evaluate(
            raw_df=df,
            post_gap_df=processed_df,
            room_name=room_name,
        )
        
        retention_pass = retention_result.get("passes", False)
        retention_metrics = retention_result.get("metrics", {})
        
        gate_stack.append(PreTrainingGateResult(
            passes=retention_pass,
            gate_name="PostGapRetentionGate",
            timestamp=_utc_now_iso_z(),
            details=retention_metrics,
            reason=retention_result.get("reasons", [None])[0] if not retention_pass else None,
        ))
        
        if not retention_pass:
            return self._build_failure_result(
                context=context,
                gate_stack=gate_stack,
                failure_stage="post_gap_retention",
            )
        
        # --- Gate 3: Class Coverage ---
        # Create sequences for class coverage check
        try:
            X, y, _ = self._create_sequences(processed_df, room_name, seq_length)
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No sequences could be created")
            
            # Split 80/20 for coverage check
            split_idx = int(len(y) * 0.8)
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            class_coverage_result = self.class_coverage_gate.evaluate(
                y_train=y_train,
                y_val=y_val,
                y_calib=None,
                room_name=room_name,
            )
            
            class_pass = class_coverage_result.get("passes", False)
            
            gate_stack.append(PreTrainingGateResult(
                passes=class_pass,
                gate_name="ClassCoverageGate",
                timestamp=_utc_now_iso_z(),
                details={
                    "train_classes": len(class_coverage_result.get("analysis", {}).get("train_coverage", {})),
                    "val_classes": len(class_coverage_result.get("analysis", {}).get("val_coverage", {})),
                    "absent_critical": class_coverage_result.get("analysis", {}).get("absent_from_train", []),
                    "low_support_train": class_coverage_result.get("analysis", {}).get("train_insufficient_support", []),
                },
                reason="class_coverage_failed" if not class_pass else None,
            ))
            
            if not class_pass:
                return self._build_failure_result(
                    context=context,
                    gate_stack=gate_stack,
                    failure_stage="class_coverage",
                )
                
        except Exception as e:
            logger.error(f"Class coverage evaluation failed for {room_name}: {e}")
            gate_stack.append(PreTrainingGateResult(
                passes=False,
                gate_name="ClassCoverageGate",
                timestamp=_utc_now_iso_z(),
                details={"error": str(e)},
                reason=f"class_coverage_error:{e}",
            ))
            return self._build_failure_result(
                context=context,
                gate_stack=gate_stack,
                failure_stage="class_coverage",
            )
        
        # All pre-training gates passed
        return GateEvaluationResult(
            run_id=run_id,
            elder_id=elder_id,
            room_name=room_name,
            pre_training_pass=True,
            post_training_pass=True,  # Will be updated after training
            overall_pass=True,
            gate_stack=gate_stack,
            failure_stage=None,
        )
    
    def evaluate_post_training_gates(
        self,
        room_name: str,
        elder_id: str,
        run_id: str,
        calibration_support: Dict[str, int],
        existing_gate_result: GateEvaluationResult,
        validation_class_support: Optional[Dict[str, int]] = None,
    ) -> GateEvaluationResult:
        """
        Evaluate post-training gates (StatisticalValidityGate).
        
        Updates the existing gate result with post-training evaluation.
        """
        # Build y_calib array from calibration support
        y_calib_list = []
        for class_id, count in calibration_support.items():
            class_id_int = self._parse_class_id(class_id)
            try:
                count_int = int(count)
            except (ValueError, TypeError):
                continue
            if class_id_int is None or count_int <= 0:
                continue
            y_calib_list.extend([class_id_int] * count_int)
        
        if not y_calib_list:
            # No calibration data - pass by default
            existing_gate_result.gate_stack.append(PreTrainingGateResult(
                passes=True,
                gate_name="StatisticalValidityGate",
                timestamp=_utc_now_iso_z(),
                details={"note": "no_calibration_data"},
                reason=None,
            ))
            existing_gate_result.post_training_pass = True
            existing_gate_result.overall_pass = existing_gate_result.pre_training_pass
            return existing_gate_result
        
        y_val_list: List[int] = []
        val_support = validation_class_support or {}
        for class_id, count in val_support.items():
            class_id_int = self._parse_class_id(class_id)
            try:
                count_int = int(count)
            except (ValueError, TypeError):
                continue
            if class_id_int is None or count_int <= 0:
                continue
            y_val_list.extend([class_id_int] * count_int)

        y_calib = np.array(y_calib_list, dtype=np.int32)
        y_val = np.array(y_val_list, dtype=np.int32) if y_val_list else None
        
        # Evaluate statistical validity
        stat_result = self.statistical_gate.evaluate(
            y_calib=y_calib,
            y_val=y_val,
            room_name=room_name,
        )
        
        stat_blocking = bool(stat_result.get("blocking", not stat_result.get("passes", False)))
        stat_pass = not stat_blocking
        
        existing_gate_result.gate_stack.append(PreTrainingGateResult(
            passes=stat_pass,
            gate_name="StatisticalValidityGate",
            timestamp=_utc_now_iso_z(),
            details={
                "calibration_support": calibration_support,
                "validation_class_support": val_support,
                "metrics": stat_result.get("metrics", {}),
                "evaluation_status": stat_result.get("evaluation_status"),
                "blocking_reasons": stat_result.get("blocking_reasons", []),
                "non_blocking_reasons": stat_result.get("non_blocking_reasons", []),
                "reason_codes": stat_result.get("reason_codes", []),
            },
            reason=(
                (stat_result.get("blocking_reasons") or stat_result.get("reasons") or [None])[0]
                if not stat_pass
                else None
            ),
        ))
        
        existing_gate_result.post_training_pass = stat_pass
        existing_gate_result.overall_pass = existing_gate_result.pre_training_pass and stat_pass
        
        if not stat_pass:
            existing_gate_result.failure_stage = "statistical_validity"
            existing_gate_result.rejection_artifact = self._build_rejection_artifact(
                context=GateEvaluationContext(
                    room_name=room_name,
                    elder_id=elder_id,
                    run_id=run_id,
                    df=pd.DataFrame(),
                    processed_df=pd.DataFrame(),
                    observed_days=set(),
                    seq_length=0,
                    policy=self.policy,
                ),
                gate_stack=existing_gate_result.gate_stack,
                failure_stage="statistical_validity",
            )
        
        return existing_gate_result
    
    def _create_sequences(
        self,
        df: pd.DataFrame,
        room_name: str,
        seq_length: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences for class coverage check."""
        # Infer sensor columns
        sensor_cols = [c for c in df.columns if c not in ['timestamp', 'activity', 'activity_encoded', 'label']]
        
        sensor_data = df[sensor_cols].values.astype(np.float32)
        if 'activity_encoded' in df.columns:
            labels = df['activity_encoded'].values.astype(np.int32)
        elif 'activity' in df.columns:
            labels, _ = pd.factorize(df['activity'], sort=False)
            labels = labels.astype(np.int32)
        else:
            raise ValueError("Missing activity/activity_encoded column for class coverage evaluation")
        timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df))
        
        return create_labeled_sequences_strict(
            sensor_data=sensor_data,
            labels=labels,
            seq_length=seq_length,
            stride=1,
            timestamps=timestamps,
        )
    
    def _build_failure_result(
        self,
        context: GateEvaluationContext,
        gate_stack: List[PreTrainingGateResult],
        failure_stage: str,
    ) -> GateEvaluationResult:
        """Build a failure result with rejection artifact."""
        rejection_artifact = self._build_rejection_artifact(
            context=context,
            gate_stack=gate_stack,
            failure_stage=failure_stage,
        )
        
        return GateEvaluationResult(
            run_id=context.run_id,
            elder_id=context.elder_id,
            room_name=context.room_name,
            pre_training_pass=False,
            post_training_pass=False,
            overall_pass=False,
            gate_stack=gate_stack,
            rejection_artifact=rejection_artifact,
            failure_stage=failure_stage,
        )
    
    def _build_rejection_artifact(
        self,
        context: GateEvaluationContext,
        gate_stack: List[PreTrainingGateResult],
        failure_stage: str,
    ) -> Dict[str, Any]:
        """Build rejection artifact for gate failure."""
        category_map = {
            "coverage_contract": RejectionCategory.COVERAGE,
            "post_gap_retention": RejectionCategory.VIABILITY,
            "class_coverage": RejectionCategory.CLASS_COVERAGE,
            "statistical_validity": RejectionCategory.STATISTICAL_VALIDITY,
            "sequence_creation": RejectionCategory.SEQUENCE_ALIGNMENT,
            "preprocessing": RejectionCategory.GLOBAL_GATE,
            "training": RejectionCategory.GLOBAL_GATE,
        }
        
        category = category_map.get(failure_stage, RejectionCategory.GLOBAL_GATE)
        
        builder = RejectionArtifactBuilder(
            run_id=context.run_id,
            elder_id=context.elder_id,
            policy_hash=self._policy_hash(),
        )
        
        builder.add_reason(
            category=category,
            severity=Severity.CRITICAL,
            code=f"{failure_stage}_failed",
            message=(
                f"Training blocked at {failure_stage} gate "
                f"(observed_days={len(context.observed_days)}, "
                f"raw_samples={len(context.df)}, "
                f"processed_samples={len(context.processed_df)})"
            ),
            room=context.room_name,
            recommendation="Inspect gate stack in training metadata and adjust data quality or policy thresholds.",
        )
        
        return builder.build().to_dict()
    
    def _policy_hash(self) -> str:
        """Generate hash of current policy."""
        import hashlib
        policy_dict = self.policy.to_dict()
        policy_str = json.dumps(policy_dict, sort_keys=True, default=str)
        return hashlib.sha256(policy_str.encode()).hexdigest()[:16]
    
    def save_rejection_artifact(
        self,
        result: GateEvaluationResult,
        output_dir: Path,
    ) -> Path:
        """
        Save rejection artifact to disk.
        
        Returns:
            Path to saved why_rejected.json file
        """
        if result.rejection_artifact is None:
            raise ValueError("No rejection artifact to save")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{result.room_name}_{result.run_id}_why_rejected.json"
        output_path = output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(result.rejection_artifact, f, indent=2, default=str)
        
        logger.info(f"Saved rejection artifact to {output_path}")
        return output_path


def integrate_gates_into_training(
    room_name: str,
    elder_id: str,
    df: pd.DataFrame,
    processed_df: pd.DataFrame,
    observed_days: Set[pd.Timestamp],
    seq_length: int,
    policy: Optional[TrainingPolicy] = None,
    models_dir: Optional[Path] = None,
) -> Tuple[bool, Dict[str, Any], Optional[Path]]:
    """
    Convenience function to evaluate pre-training gates.
    
    Returns:
        (should_train, gate_result_dict, rejection_artifact_path)
    """
    pipeline = GateIntegrationPipeline(policy=policy)
    
    result = pipeline.evaluate_pre_training_gates(
        room_name=room_name,
        elder_id=elder_id,
        df=df,
        processed_df=processed_df,
        observed_days=observed_days,
        seq_length=seq_length,
    )
    
    rejection_path = None
    if not result.pre_training_pass and models_dir:
        rejection_path = pipeline.save_rejection_artifact(result, models_dir)
    
    return result.pre_training_pass, result.to_dict(), rejection_path
