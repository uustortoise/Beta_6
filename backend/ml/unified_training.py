"""
PR-1: Unified Training Path

Routes all training (watcher and manual) through a single hardened gate pipeline.
Ensures consistent gate execution, evidence artifacts, and fail-closed behavior.
"""

import logging
import time
import uuid
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from ml.coverage_contract import CoverageContractGate, WalkForwardConfig
from ml.post_gap_retention_gate import PostGapRetentionGate
from ml.class_coverage_gate import ClassCoverageGate
from ml.statistical_validity_gate import StatisticalValidityGate
from ml.rejection_artifact import RejectionArtifactBuilder, RejectionCategory, Severity
from ml.sequence_alignment import create_labeled_sequences_strict, SequenceLabelAlignmentError
from ml.policy_config import TrainingPolicy

logger = logging.getLogger(__name__)


def _utc_now_iso_z() -> str:
    """Return timezone-aware UTC timestamp with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class GateResult:
    """Result from a single gate evaluation."""
    gate_name: str
    passed: bool
    timestamp: str
    details: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "timestamp": self.timestamp,
            "details": self.details,
            "reason": self.reason,
        }


@dataclass
class UnifiedTrainingResult:
    """Result from unified training pipeline."""
    room: str
    gate_pass: bool
    gate_reasons: List[str] = field(default_factory=list)
    gate_stack: List[GateResult] = field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = None
    rejection_artifact: Optional[Dict[str, Any]] = None
    training_duration_sec: float = 0.0
    training_executed: bool = False  # [P0] Track if training actually happened
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "room": self.room,
            "gate_pass": self.gate_pass,
            "gate_reasons": self.gate_reasons,
            "gate_stack": [g.to_dict() for g in self.gate_stack],
            "metrics": self.metrics,
            "rejection_artifact": self.rejection_artifact,
            "training_duration_sec": self.training_duration_sec,
            "training_executed": self.training_executed,
        }


class UnifiedTrainingPipeline:
    """
    Hardened training pipeline used by both watcher and manual entrypoints.
    
    Executes a consistent gate stack:
    1. CoverageContractGate - data sufficiency
    2. PostGapRetentionGate - data continuity
    3. ClassCoverageGate - split coverage
    4. Training (if gates pass)
    5. StatisticalValidityGate - metric reliability (post-training)
    6. Rejection artifact (if any gate fails)
    
    [P0] Important: This class only performs GATE EVALUATION, not actual training.
    The caller is responsible for training based on gate results.
    This prevents double-training issues.
    """
    
    def __init__(self, policy: Optional[TrainingPolicy] = None):
        """
        Initialize with training policy.
        
        Parameters:
        -----------
        policy : TrainingPolicy, optional
            Training policy for gate configuration. Uses env policy if None.
        """
        if policy is None:
            from ml.policy_config import load_policy_from_env
            policy = load_policy_from_env()
        self.policy = policy
        
        # [P1] Initialize all gates from policy SSoT with actual field names
        self.coverage_gate = self._init_coverage_gate()
        self.retention_gate = self._init_retention_gate()
        self.class_coverage_gate = self._init_class_coverage_gate()
        self.statistical_gate = self._init_statistical_gate()
    
    def _init_coverage_gate(self) -> CoverageContractGate:
        """[P1] Initialize coverage contract gate from policy SSoT."""
        min_observed = int(self.policy.release_gate.min_observed_days)
        
        # Create a simple config - we only need observed days check
        config = WalkForwardConfig(
            min_train_days=min_observed,  # Use observed days as proxy
            valid_days=1,
            step_days=1,
            min_folds=1,
        )
        return CoverageContractGate(config)
    
    def _init_retention_gate(self) -> PostGapRetentionGate:
        """[P1] Initialize post-gap retention gate from policy SSoT."""
        min_retained = float(self.policy.release_gate.min_retained_sample_ratio)
        
        return PostGapRetentionGate(
            min_retained_ratio=min_retained,
            max_contiguous_segments=10,
            min_max_segment_length=100,
            min_median_segment_length=50,
        )
    
    def _init_class_coverage_gate(self) -> ClassCoverageGate:
        """[P1] Initialize class coverage gate from policy SSoT."""
        evidence_profile = str(
            getattr(self.policy.release_gate, "evidence_profile", "production") or "production"
        ).strip().lower()
        pilot_relaxed_evidence = evidence_profile in {"pilot_stage_a", "pilot_stage_b"}
        min_support = int(self.policy.release_gate.min_calibration_support)
        return ClassCoverageGate(
            min_train_support=min_support,
            min_val_support=1 if pilot_relaxed_evidence else 5,
            min_calib_support=1 if pilot_relaxed_evidence else min_support,  # Use same as train in production
            min_class_coverage_ratio=0.8,
            block_on_val_support=not pilot_relaxed_evidence,
            block_on_calib_support=not pilot_relaxed_evidence,
        )
    
    def _init_statistical_gate(self) -> StatisticalValidityGate:
        """[P1] Initialize statistical validity gate from policy SSoT."""
        evidence_profile = str(
            getattr(self.policy.release_gate, "evidence_profile", "production") or "production"
        ).strip().lower()
        pilot_relaxed_evidence = evidence_profile in {"pilot_stage_a", "pilot_stage_b"}
        min_calib_support = int(self.policy.calibration.min_support_per_class)
        return StatisticalValidityGate(
            min_calibration_support=min_calib_support,
            min_validation_support=10,
            min_minority_support=5,
            pilot_relaxed_evidence=pilot_relaxed_evidence,
        )

    def is_event_first_shadow_enabled(self) -> bool:
        """
        Lane C: expose shadow-mode switch from typed policy.
        """
        try:
            return bool(self.policy.event_first.shadow)
        except Exception:
            return False

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
    
    def evaluate_gates(
        self,
        room_name: str,
        df: pd.DataFrame,
        elder_id: str,
        seq_length: int,
        observed_days: Set[pd.Timestamp],
        platform: Any,
        max_ffill_gap_seconds: float = 60.0,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> UnifiedTrainingResult:
        """
        [P0] Evaluate all pre-training gates WITHOUT executing training.
        
        This method ONLY evaluates gates and returns the result.
        The caller is responsible for training based on gate_pass.
        This prevents double-training issues.
        
        Parameters:
        -----------
        room_name : str
            Room being trained
        df : pd.DataFrame
            Raw training data (before preprocessing)
        elder_id : str
            Resident ID
        seq_length : int
            Sequence length for model
        observed_days : Set[pd.Timestamp]
            Set of days with observed data
        platform : ElderlyCarePlatform
            Platform for preprocessing
        max_ffill_gap_seconds : float
            Max gap for forward fill
        progress_callback : Callable, optional
            Progress callback(percent, message)
            
        Returns:
        --------
        UnifiedTrainingResult
            Gate evaluation result with stack and evidence.
            training_executed will always be False.
            metrics will be None.
        """
        start_time = time.time()
        gate_stack: List[GateResult] = []
        gate_reasons: List[str] = []
        
        # [P2] Helper to record gate result
        def record_gate(name: str, passed: bool, details: Dict, reason: Optional[str] = None):
            gate_stack.append(GateResult(
                gate_name=name,
                passed=passed,
                timestamp=_utc_now_iso_z(),
                details=details,
                reason=reason if not passed else None,
            ))
            return reason if not passed else None
        
        # --- Gate 1: Coverage Contract ---
        observed_day_count = len(observed_days)
        coverage_result = self.coverage_gate.evaluate(room_name, observed_day_count)
        
        reason = record_gate(
            "CoverageContractGate",
            coverage_result.passes,
            {
                "observed_days": observed_day_count,
                "required_days": coverage_result.required_days,
                "estimated_max_folds": coverage_result.details.get("estimated_max_folds", 0),
            },
            f"coverage_contract_failed:{coverage_result.reason}" if not coverage_result.passes else None,
        )
        if reason:
            gate_reasons.append(reason)
        
        if not coverage_result.passes:
            return self._build_failure_result(
                room_name, gate_stack, gate_reasons, "coverage_contract",
                df, observed_day_count, start_time, elder_id=elder_id
            )
        
        if progress_callback:
            progress_callback(10, "Coverage contract passed")
        
        # --- Preprocess for downstream gates ---
        try:
            processed = platform.preprocess_without_scaling(
                df, room_name, is_training=True,
                apply_denoising=False,
                max_ffill_gap_seconds=max_ffill_gap_seconds,
            )
            
            # Filter to observed days
            if 'timestamp' in processed.columns and observed_days:
                ts = pd.to_datetime(processed['timestamp'], errors='coerce')
                keep_mask = ts.dt.floor('D').isin(list(observed_days))
                processed = processed.loc[keep_mask].copy().reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Preprocessing failed for {room_name}: {e}")
            reason = record_gate("PreprocessingGate", False, {"error": str(e)}, f"preprocessing_failed:{e}")
            gate_reasons.append(reason)
            return self._build_failure_result(
                room_name, gate_stack, gate_reasons, "preprocessing",
                df, observed_day_count, start_time, elder_id=elder_id
            )
        
        # --- Gate 2: Post-Gap Retention ---
        retention_result = self.retention_gate.evaluate(
            raw_df=df,
            post_gap_df=processed,
            room_name=room_name,
        )
        retention_pass = retention_result.get("passes", False)
        retention_metrics = retention_result.get("metrics", {})
        
        reason = record_gate(
            "PostGapRetentionGate",
            retention_pass,
            retention_metrics,
            f"post_gap_retention_failed:{retention_result.get('reasons', ['unknown'])[0]}" if not retention_pass and retention_result.get('reasons') else None,
        )
        if reason:
            gate_reasons.append(reason)
        
        if not retention_pass:
            return self._build_failure_result(
                room_name, gate_stack, gate_reasons, "post_gap_retention",
                df, observed_day_count, start_time, processed_df=processed, elder_id=elder_id
            )
        
        if progress_callback:
            progress_callback(20, "Post-gap retention passed")
        
        # --- Create sequences for class coverage check ---
        try:
            X, y, seq_timestamps = self._create_sequences_safe(
                platform, processed, room_name, seq_length
            )
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No sequences could be created")
                
        except Exception as e:
            logger.error(f"Sequence creation failed for {room_name}: {e}")
            reason = record_gate("SequenceCreationGate", False, {"error": str(e)}, f"sequence_creation_failed:{e}")
            gate_reasons.append(reason)
            return self._build_failure_result(
                room_name, gate_stack, gate_reasons, "sequence_creation",
                df, observed_day_count, start_time, processed_df=processed, elder_id=elder_id
            )
        
        # --- Gate 3: Class Coverage (on train/val split) ---
        split_idx = int(len(X) * 0.8)
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        class_coverage_result = self.class_coverage_gate.evaluate(
            y_train=y_train,
            y_val=y_val,
            y_calib=None,
            room_name=room_name,
        )
        
        class_pass = class_coverage_result.get("passes", False)
        
        reason = record_gate(
            "ClassCoverageGate",
            class_pass,
            {
                "train_classes": len(class_coverage_result.get("train_coverage", {})),
                "val_classes": len(class_coverage_result.get("val_coverage", {})),
                "absent_critical": class_coverage_result.get("absent_critical_classes", []),
                "low_support_train": class_coverage_result.get("low_support_train", []),
            },
            "class_coverage_failed:insufficient_class_support" if not class_pass else None,
        )
        if reason:
            gate_reasons.append(reason)
        
        if not class_pass:
            return self._build_failure_result(
                room_name, gate_stack, gate_reasons, "class_coverage",
                df, observed_day_count, start_time, processed_df=processed, elder_id=elder_id
            )
        
        if progress_callback:
            progress_callback(30, "Class coverage passed")
        
        # --- All pre-training gates passed ---
        duration = time.time() - start_time
        
        return UnifiedTrainingResult(
            room=room_name,
            gate_pass=True,
            gate_reasons=gate_reasons,
            gate_stack=gate_stack,
            metrics=None,
            training_duration_sec=duration,
            training_executed=False,
        )
    
    def evaluate_post_training_gates(
        self,
        room_name: str,
        metrics: Dict[str, Any],
        gate_stack: List[GateResult],
        gate_reasons: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        [P1] Evaluate post-training gates (StatisticalValidityGate).
        
        Uses the real StatisticalValidityGate.evaluate() method with correct signature.
        
        Parameters:
        -----------
        room_name : str
            Room name
        metrics : dict
            Training metrics
        gate_stack : list
            Existing gate stack to append to
        gate_reasons : list
            Existing gate reasons to append to
            
        Returns:
        --------
        (gate_pass, updated_gate_reasons) : Tuple
        """
        # [P0] Build calibration label array from calibration_support
        # calibration_support is {class_id: count, ...}
        calib_support = (
            metrics.get("calibration_support")
            or metrics.get("calibration_class_support")
            or {}
        )
        
        # Build y_calib array from support counts
        y_calib_list = []
        if isinstance(calib_support, dict):
            for class_id, count in calib_support.items():
                class_id_int = self._parse_class_id(class_id)
                try:
                    count_int = int(count)
                except (ValueError, TypeError):
                    continue
                if class_id_int is None or count_int <= 0:
                    continue
                y_calib_list.extend([class_id_int] * count_int)

        y_val_list = []
        val_support = metrics.get("validation_class_support", {})
        if isinstance(val_support, dict):
            for class_id, count in val_support.items():
                class_id_int = self._parse_class_id(class_id)
                try:
                    count_int = int(count)
                except (ValueError, TypeError):
                    continue
                if class_id_int is None or count_int <= 0:
                    continue
                y_val_list.extend([class_id_int] * count_int)

        if not y_calib_list:
            # No class support map available from training metrics.
            # Keep non-blocking to preserve backward compatibility.
            gate_stack.append(GateResult(
                gate_name="StatisticalValidityGate",
                passed=True,
                timestamp=_utc_now_iso_z(),
                details={
                    "note": "no_calibration_support_map",
                    "available_metric_keys": sorted(metrics.keys()),
                },
                reason=None,
            ))
            return True, gate_reasons

        y_calib = np.array(y_calib_list, dtype=np.int32)
        y_val = np.array(y_val_list, dtype=np.int32) if y_val_list else None

        # [P0] Use real StatisticalValidityGate.evaluate() with correct signature
        stat_result = self.statistical_gate.evaluate(
            y_calib=y_calib,
            y_val=y_val,
            room_name=room_name,
        )

        stat_blocking = bool(stat_result.get("blocking", not stat_result.get("passes", False)))
        stat_pass = not stat_blocking

        gate_stack.append(GateResult(
            gate_name="StatisticalValidityGate",
            passed=stat_pass,
            timestamp=_utc_now_iso_z(),
            details={
                "calibration_support": calib_support,
                "validation_class_support": val_support,
                "metrics": stat_result.get("metrics", {}),
                "evaluation_status": stat_result.get("evaluation_status"),
                "blocking_reasons": stat_result.get("blocking_reasons", []),
                "non_blocking_reasons": stat_result.get("non_blocking_reasons", []),
                "reason_codes": stat_result.get("reason_codes", []),
            },
            reason=(
                f"statistical_validity_failed:"
                f"{(stat_result.get('blocking_reasons') or stat_result.get('reasons') or ['unknown'])[0]}"
                if not stat_pass
                else None
            ),
        ))

        if not stat_pass:
            reason = (
                f"statistical_validity_failed:"
                f"{(stat_result.get('blocking_reasons') or stat_result.get('reasons') or ['unknown'])[0]}"
            )
            gate_reasons.append(reason)
            return False, gate_reasons

        return True, gate_reasons
    
    def _create_sequences_safe(
        self,
        platform: Any,
        df: pd.DataFrame,
        room_name: str,
        seq_length: int,
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        [P0] Safely create sequences using real platform method.
        
        Returns (X, y, seq_timestamps) or raises exception.
        """
        try:
            # Get sensor columns from platform
            sensor_cols = getattr(platform, 'sensor_columns', None)
            if sensor_cols is None:
                # Fallback: try to infer sensor columns
                sensor_cols = [c for c in df.columns if c not in ['timestamp', 'activity', 'label']]
            
            # Extract sensor data and labels
            sensor_data = df[sensor_cols].values.astype(np.float32)
            labels = df['activity'].values if 'activity' in df.columns else np.zeros(len(df))
            timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df))
            
            # [P0] Use create_labeled_sequences_strict from sequence_alignment module
            X_seq, y_seq, seq_timestamps = create_labeled_sequences_strict(
                sensor_data=sensor_data,
                labels=labels,
                seq_length=seq_length,
                stride=1,
                timestamps=timestamps,
            )
            
            return X_seq, y_seq, seq_timestamps
            
        except SequenceLabelAlignmentError as e:
            logger.error(f"Sequence alignment error: {e}")
            raise
        except Exception as e:
            logger.error(f"Sequence creation error: {e}")
            raise
    
    def _build_failure_result(
        self,
        room_name: str,
        gate_stack: List[GateResult],
        gate_reasons: List[str],
        failure_stage: str,
        raw_df: pd.DataFrame,
        observed_day_count: int,
        start_time: float,
        processed_df: Optional[pd.DataFrame] = None,
        elder_id: Optional[str] = None,
    ) -> UnifiedTrainingResult:
        """Build a failure result with rejection artifact."""
        duration = time.time() - start_time
        
        # Build rejection artifact with required parameters
        run_id = str(uuid.uuid4())
        
        builder = RejectionArtifactBuilder(
            run_id=run_id,
            elder_id=str(elder_id) if elder_id else "unknown",
        )
        
        # Map failure stage to category
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
        
        builder.add_reason(
            category=category,
            severity=Severity.CRITICAL,
            code=f"{failure_stage}_failed",
            message=f"Training blocked at {failure_stage} gate",
            room=room_name if room_name else None,
        )
        
        rejection_artifact = builder.build()
        
        return UnifiedTrainingResult(
            room=room_name,
            gate_pass=False,
            gate_reasons=gate_reasons,
            gate_stack=gate_stack,
            metrics=None,
            rejection_artifact=rejection_artifact.to_dict(),
            training_duration_sec=duration,
            training_executed=False,
        )


def create_unified_training_result(
    room: str,
    gate_pass: bool,
    gate_reasons: List[str],
    metrics: Optional[Dict] = None,
    training_executed: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to create a standardized training result dict.
    
    This ensures both watcher and manual paths return consistent metadata.
    """
    result = {
        "room": room,
        "gate_pass": gate_pass,
        "gate_reasons": gate_reasons,
        "gate_stack": [],
        "metrics": metrics,
        "training_executed": training_executed,
    }
    
    if metrics:
        metrics["gate_pass"] = gate_pass
        metrics["gate_reasons"] = gate_reasons
    
    return result
