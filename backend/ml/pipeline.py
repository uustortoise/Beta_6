import os
import sys
import logging
import sqlite3
import json
import hashlib
import re
from contextlib import contextmanager
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from datetime import datetime

from elderlycare_v1_16.platform import ElderlyCarePlatform, Attention
from elderlycare_v1_16.config.settings import (
    DEFAULT_SENSOR_COLUMNS, DEFAULT_DATA_INTERVAL,
    DEFAULT_SEQUENCE_WINDOW, DEFAULT_EPOCHS,
    DEFAULT_LSTM_UNITS, DEFAULT_DROPOUT_RATE,
    DEFAULT_CONV_FILTERS_1, DEFAULT_CONV_FILTERS_2,
    DEFAULT_VALIDATION_SPLIT, DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DENOISING_METHOD, DEFAULT_DENOISING_WINDOW,
    DEFAULT_DENOISING_THRESHOLD, LOGS_DIR
)

from config import get_room_config, DB_PATH
from ml.utils import calculate_sequence_length
from utils.room_utils import normalize_room_name

# Custom exceptions for structured error handling
from ml.exceptions import (
    ModelLoadError, ModelTrainError, PredictionError, 
    DataValidationError, DatabaseError
)

# Beta 5.5: Import Transformer backbone (default architecture for this environment)
from ml.transformer_backbone import build_transformer_model
# Handle Keras serialization decorator safely
try:
    from tensorflow.keras.saving import register_keras_serializable
except ImportError:
    from tensorflow.keras.utils import register_keras_serializable

# PR-1/PR-2: Unified Training Path and Gate Integration
from ml.unified_training import UnifiedTrainingPipeline, UnifiedTrainingResult
from ml.gate_integration import GateIntegrationPipeline

logger = logging.getLogger(__name__)


def _env_enabled_local(var_name: str, default: bool = False) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}

class UnifiedPipeline:
    def __init__(self, 
                 enable_denoising=True,
                 denoising_method=DEFAULT_DENOISING_METHOD,
                 denoising_window=DEFAULT_DENOISING_WINDOW,
                 denoising_threshold=DEFAULT_DENOISING_THRESHOLD):
        
        # Setup logging first
        from elderlycare_v1_16.config.logging_config import setup_logging
        self.logger = setup_logging(str(LOGS_DIR / "pipeline.log"))
        
        self.room_config = get_room_config()
        self.platform = ElderlyCarePlatform(
            sensor_columns=DEFAULT_SENSOR_COLUMNS,
            data_interval=DEFAULT_DATA_INTERVAL,
            sequence_time_window=DEFAULT_SEQUENCE_WINDOW,
            enable_time_based_processing=True
        )
        
        # Initial stats
        self.logger.info(f"Initialized UnifiedPipeline with Window={self.platform.sequence_time_window}, Interval={self.platform.data_interval}")
            
        self.enable_denoising = enable_denoising
        self.denoising_method = denoising_method
        self.denoising_window = denoising_window
        self.denoising_threshold = denoising_threshold

        # Initialize sub-components using relative imports
        from ml.registry import ModelRegistry
        from ml.training import TrainingPipeline
        from ml.prediction import PredictionPipeline
        
        # Determine backend_dir from current file location to pass to Registry
        current_file = Path(__file__).resolve()
        backend_dir = current_file.parent.parent # backend/ml/pipeline.py -> backend/
        
        self.registry = ModelRegistry(str(backend_dir))
        self.trainer = TrainingPipeline(self.platform, self.registry)
        self.predictor = PredictionPipeline(self.platform, self.registry, 
                                          enable_denoising=self.enable_denoising)
        
        # PR-1: Unified training pipeline for hardened gate execution
        self.unified_training = UnifiedTrainingPipeline()
        try:
            from ml.timeline_pipeline_integration import log_feature_flag_status
            self.logger.info(log_feature_flag_status())
        except Exception:
            # Keep startup behavior unchanged when timeline modules are unavailable.
            pass

    @staticmethod
    def _beta6_runtime_policy_default_paths() -> dict[str, str]:
        backend_dir = Path(__file__).resolve().parent.parent
        cfg = backend_dir / "config"
        return {
            "BETA6_UNKNOWN_POLICY_PATH": str(cfg / "beta6_unknown_policy.yaml"),
            "BETA6_HMM_DURATION_POLICY_PATH": str(cfg / "beta6_duration_prior_policy.yaml"),
        }

    @staticmethod
    def _beta6_registry_v2_root() -> Path | None:
        raw = str(os.getenv("BETA6_REGISTRY_V2_ROOT", "")).strip()
        if raw:
            return Path(raw).expanduser()
        backend_dir = Path(__file__).resolve().parent.parent
        return (backend_dir / "models_beta6_registry_v2").resolve()

    def _beta6_phase4_runtime_policy_path(self, elder_id: str) -> Path | None:
        root = self._beta6_registry_v2_root()
        if root is None:
            return None
        safe_elder = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(elder_id or "").strip())
        if not safe_elder:
            return None
        return root / safe_elder / "_runtime" / "phase4_runtime_policy.json"

    def _load_beta6_phase4_runtime_policy(self, elder_id: str) -> dict | None:
        policy_path = self._beta6_phase4_runtime_policy_path(elder_id)
        if policy_path is None or not policy_path.exists():
            return None
        try:
            payload = json.loads(policy_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if str(payload.get("schema_version", "")).strip() != "beta6.phase4.runtime_policy.v1":
            return None
        return payload

    def _beta6_any_room_fallback_active(self, elder_id: str, rooms: list[str]) -> bool:
        active_logger = getattr(self, "logger", logger)
        root = self._beta6_registry_v2_root()
        if root is None or not rooms:
            return False
        try:
            from ml.beta6.registry.registry_v2 import RegistryV2

            registry_v2 = RegistryV2(root)
            for room in rooms:
                state = registry_v2.read_fallback_state(elder_id=elder_id, room=str(room))
                if isinstance(state, dict) and bool(state.get("active", False)):
                    active_logger.info(
                        "Beta6 Phase4 runtime bridge skipped for %s/%s: fallback mode active",
                        elder_id,
                        room,
                    )
                    return True
        except Exception as e:
            # Fail safe for auto-enable path: preserve legacy runtime behavior when fallback
            # state cannot be inspected. Explicit env flags can still force activation.
            active_logger.warning(f"Beta6 fallback-state read failed; skipping auto runtime bridge: {e}")
            return True
        return False

    @staticmethod
    def _normalize_lineage_label(value) -> str:
        if pd.isna(value):
            return "__missing__"
        text = str(value).strip()
        return text or "__empty__"

    @classmethod
    def _build_pre_sampling_label_counts_by_date(cls, df: pd.DataFrame) -> dict[str, dict[str, int]]:
        if df is None or df.empty or "timestamp" not in df.columns or "activity" not in df.columns:
            return {}
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
        if timestamps.isna().all():
            return {}
        labels = df["activity"].map(cls._normalize_lineage_label)
        summary: dict[str, dict[str, int]] = {}
        summary_df = pd.DataFrame({
            "date": timestamps.dt.strftime("%Y-%m-%d"),
            "label": labels,
        }).dropna(subset=["date"])
        if summary_df.empty:
            return {}
        grouped = (
            summary_df.groupby(["date", "label"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["date", "label"], kind="stable")
        )
        for row in grouped.itertuples(index=False):
            day = str(row.date)
            label = str(row.label)
            summary.setdefault(day, {})[label] = int(row.count)
        return summary

    @classmethod
    def _build_label_counts(cls, df: pd.DataFrame) -> dict[str, int]:
        if df is None or df.empty or "activity" not in df.columns:
            return {}
        counts = (
            df["activity"]
            .map(cls._normalize_lineage_label)
            .value_counts(dropna=False)
            .sort_index()
        )
        return {str(label): int(count) for label, count in counts.items()}

    @classmethod
    def _build_room_source_lineage(
        cls,
        *,
        room_name: str,
        source_entries: list[dict],
        combined_df: pd.DataFrame,
    ) -> dict[str, object]:
        manifest: list[dict[str, object]] = []
        for entry in source_entries:
            path = Path(entry["file_path"]).expanduser().resolve()
            try:
                stats = path.stat()
            except OSError:
                stats = None
            source_df = entry.get("df")
            source_dates: list[str] = []
            if isinstance(source_df, pd.DataFrame) and not source_df.empty and "timestamp" in source_df.columns:
                ts = pd.to_datetime(source_df["timestamp"], errors="coerce").dropna()
                if not ts.empty:
                    source_dates = sorted({str(value.date()) for value in ts})
            manifest.append(
                {
                    "path": str(path),
                    "source_order": int(entry.get("source_order", 0)),
                    "rows_for_room": int(len(source_df)) if isinstance(source_df, pd.DataFrame) else 0,
                    "file_size_bytes": int(stats.st_size) if stats is not None else None,
                    "file_mtime_ns": int(stats.st_mtime_ns) if stats is not None else None,
                    "observed_dates": source_dates,
                    "label_counts": cls._build_label_counts(source_df),
                }
            )
        manifest = sorted(manifest, key=lambda item: (int(item["source_order"]), str(item["path"])))
        fingerprint_basis = {
            "room": str(room_name),
            "source_manifest": [
                {
                    "path": str(entry["path"]),
                    "source_order": int(entry["source_order"]),
                    "rows_for_room": int(entry["rows_for_room"]),
                    "file_size_bytes": entry["file_size_bytes"],
                    "file_mtime_ns": entry["file_mtime_ns"],
                }
                for entry in manifest
            ],
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        pre_sampling_by_date = cls._build_pre_sampling_label_counts_by_date(combined_df)
        observed_dates = sorted(pre_sampling_by_date.keys())
        return {
            "room": str(room_name),
            "source_manifest": manifest,
            "source_fingerprint": fingerprint,
            "pre_sampling_label_counts_by_date": pre_sampling_by_date,
            "pre_sampling_label_counts_total": cls._build_label_counts(combined_df),
            "observed_dates": observed_dates,
        }

    @contextmanager
    def _beta6_phase4_runtime_prediction_context(self, elder_id: str, loaded_rooms: list[str]):
        """
        Authority-aware runtime bridge for Phase 4 prediction hooks.

        Behavior:
        - respects explicit per-hook/master env flags (no auto override)
        - auto-enables master switch from per-elder runtime policy artifact when available
        - injects default Beta 6 policy paths when Phase 4 runtime is enabled and paths unset
        - skips auto-enable when RegistryV2 fallback mode is active for any loaded room
        """
        overrides: dict[str, str] = {}
        original: dict[str, str | None] = {}
        active_logger = getattr(self, "logger", logger)
        explicit_master = os.getenv("BETA6_PHASE4_RUNTIME_ENABLED")
        explicit_hmm = os.getenv("ENABLE_BETA6_HMM_RUNTIME")
        explicit_unknown = os.getenv("ENABLE_BETA6_UNKNOWN_ABSTAIN_RUNTIME")
        has_explicit_runtime_flag = any(v is not None for v in (explicit_master, explicit_hmm, explicit_unknown))

        authority_enabled = _env_enabled_local("ENABLE_BETA6_AUTHORITY", default=True)
        bridge_enabled = _env_enabled_local("ENABLE_BETA6_PHASE4_RUNTIME_BRIDGE", default=True)

        auto_master_enabled = False
        runtime_policy_artifact = None
        auto_enabled_rooms: list[str] = []
        if authority_enabled and bridge_enabled and not has_explicit_runtime_flag:
            runtime_policy_artifact = self._load_beta6_phase4_runtime_policy(elder_id)
            if isinstance(runtime_policy_artifact, dict) and bool(runtime_policy_artifact.get("master_enabled", False)):
                room_runtime = runtime_policy_artifact.get("room_runtime")
                room_runtime = room_runtime if isinstance(room_runtime, dict) else {}
                for room in loaded_rooms:
                    room_key = normalize_room_name(str(room))
                    entry = room_runtime.get(room_key)
                    if isinstance(entry, dict) and bool(entry.get("enable_phase4_runtime", False)):
                        auto_enabled_rooms.append(room_key)
                if auto_enabled_rooms and not self._beta6_any_room_fallback_active(
                    elder_id=elder_id,
                    rooms=list(auto_enabled_rooms),
                ):
                    overrides["BETA6_PHASE4_RUNTIME_ENABLED"] = "true"
                    overrides["BETA6_PHASE4_RUNTIME_ROOMS"] = ",".join(sorted(set(auto_enabled_rooms)))
                    auto_master_enabled = True

        runtime_requested = auto_master_enabled or _env_enabled_local(
            "BETA6_PHASE4_RUNTIME_ENABLED", default=False
        ) or _env_enabled_local("ENABLE_BETA6_HMM_RUNTIME", default=False) or _env_enabled_local(
            "ENABLE_BETA6_UNKNOWN_ABSTAIN_RUNTIME", default=False
        )
        if runtime_requested:
            if isinstance(runtime_policy_artifact, dict):
                policy_paths = runtime_policy_artifact.get("policy_paths")
                policy_paths = policy_paths if isinstance(policy_paths, dict) else {}
                unknown_policy_path = str(policy_paths.get("unknown_policy_path", "")).strip()
                hmm_policy_path = str(policy_paths.get("hmm_duration_policy_path", "")).strip()
                if os.getenv("BETA6_UNKNOWN_POLICY_PATH") is None and unknown_policy_path and Path(
                    unknown_policy_path
                ).exists():
                    overrides["BETA6_UNKNOWN_POLICY_PATH"] = unknown_policy_path
                if os.getenv("BETA6_HMM_DURATION_POLICY_PATH") is None and hmm_policy_path and Path(
                    hmm_policy_path
                ).exists():
                    overrides["BETA6_HMM_DURATION_POLICY_PATH"] = hmm_policy_path
            for key, path in self._beta6_runtime_policy_default_paths().items():
                if key not in overrides and os.getenv(key) is None and Path(path).exists():
                    overrides[key] = path

        try:
            for key, value in overrides.items():
                original[key] = os.getenv(key)
                os.environ[key] = value
            if auto_master_enabled:
                active_logger.info(
                    "Beta6 Phase4 runtime bridge active for %s (rooms=%s, source_run=%s)",
                    elder_id,
                    sorted(set(auto_enabled_rooms)),
                    str((runtime_policy_artifact or {}).get("source_run_id", "")),
                )
            yield
        finally:
            for key, value in overrides.items():
                previous = original.get(key)
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous

    def _get_models_dir(self, elder_id: str) -> Path:
        """Get the directory where models are stored for a specific elder."""
        return self.registry.get_models_dir(elder_id)

    def _resolve_resample_gap_policy(self):
        """
        Resolve active max forward-fill gap from TrainingPolicy.
        Falls back to 60s if policy is unavailable.
        """
        try:
            policy = self.trainer._active_policy()
            return policy.resampling.resolve()
        except Exception:
            return 60.0

    def _resolve_data_viability_policy(self) -> dict:
        """Resolve DataViabilityGate policy."""
        try:
            viability = self.trainer._active_policy().data_viability
            return {
                "resolver": viability,
            }
        except Exception:
            return {
                "resolver": None,
                "defaults": {
                    "min_observed_days": 7,
                    "min_post_gap_rows": 10000,
                    "max_unresolved_drop_ratio": 0.85,
                    "min_training_windows": 2500,
                },
            }

    def _is_event_first_shadow_enabled(self) -> bool:
        """Resolve event-first shadow switch from typed policy."""
        try:
            return bool(self.trainer._active_policy().event_first.shadow)
        except Exception:
            return False

    def _maybe_write_timeline_shadow_artifacts(
        self,
        *,
        elder_id: str,
        room_name: str,
        room_df: pd.DataFrame,
        context: str,
    ):
        """
        Optionally generate timeline shadow artifacts behind WS-5 feature flags.

        Returns a summary dict when timeline path is enabled; returns None when disabled.
        """
        try:
            from ml.timeline_pipeline_integration import (
                should_run_timeline_path,
                is_timeline_shadow_mode,
                validate_timeline_safety,
                TimelineShadowPipeline,
            )
        except Exception:
            return None

        if not should_run_timeline_path():
            return None

        warnings = validate_timeline_safety()
        for warning in warnings:
            logger.warning(f"[TimelineShadow] {warning}")

        summary: dict = {
            "enabled": True,
            "shadow_mode": bool(is_timeline_shadow_mode()),
            "warnings": list(warnings),
        }
        if not summary["shadow_mode"]:
            return summary

        if room_df is None or room_df.empty or "activity" not in room_df.columns:
            summary["error"] = "missing_activity_labels"
            return summary

        try:
            labels = room_df["activity"].astype(str).str.strip().str.lower().to_numpy()
            if "timestamp" in room_df.columns:
                timestamps = pd.to_datetime(room_df["timestamp"], errors="coerce").to_numpy()
            else:
                timestamps = np.arange(len(labels), dtype=np.int64)

            shadow_pipeline = TimelineShadowPipeline(adl_registry=None)
            artifacts = shadow_pipeline.process_split(
                train_data={"labels": labels, "timestamps": timestamps},
                val_data={},
                split_id=f"{context}:{room_name}",
            )
            output_dir = self._get_models_dir(elder_id) / "timeline_shadow" / room_name
            paths = shadow_pipeline.save_artifacts(output_dir)

            summary["artifact_paths"] = {k: str(v) for k, v in paths.items()}
            if isinstance(artifacts.qc_report, dict):
                summary["qc_report"] = artifacts.qc_report
            return summary
        except Exception as e:
            logger.warning(f"[TimelineShadow] Failed artifact generation for {elder_id}/{room_name}: {e}")
            summary["error"] = str(e)
            return summary

    @staticmethod
    def _evaluate_data_viability_gate(
        *,
        room_name: str,
        observed_day_count: int,
        raw_samples: int,
        post_gap_samples: int,
        post_downsample_windows: int,
        policy: dict,
    ) -> dict:
        reasons = []
        resolver = policy.get("resolver") if isinstance(policy, dict) else None
        if resolver is not None:
            resolved = resolver.resolve(room_name)
            min_observed_days = int(resolved.get("min_observed_days", 7))
            min_post_gap_rows = int(resolved.get("min_post_gap_rows", 10000))
            max_unresolved_drop_ratio = float(resolved.get("max_unresolved_drop_ratio", 0.85))
            min_training_windows = int(resolved.get("min_training_windows", 2500))
        else:
            defaults = (policy or {}).get("defaults", {}) if isinstance(policy, dict) else {}
            min_observed_days = int(defaults.get("min_observed_days", 7))
            min_post_gap_rows = int(defaults.get("min_post_gap_rows", 10000))
            max_unresolved_drop_ratio = float(defaults.get("max_unresolved_drop_ratio", 0.85))
            min_training_windows = int(defaults.get("min_training_windows", 2500))
        retained_ratio = (float(post_gap_samples) / float(raw_samples)) if raw_samples > 0 else 0.0
        unresolved_drop_ratio = max(0.0, 1.0 - retained_ratio)
        room_key = normalize_room_name(room_name)
        if observed_day_count < min_observed_days:
            reasons.append(f"insufficient_observed_days:{room_key}:{observed_day_count}<{min_observed_days}")
        if post_gap_samples < min_post_gap_rows:
            reasons.append(f"insufficient_samples:{room_key}:{post_gap_samples}<{min_post_gap_rows}")
        if unresolved_drop_ratio > max_unresolved_drop_ratio:
            reasons.append(
                f"excessive_gap_drop_ratio:{room_key}:{unresolved_drop_ratio:.4f}>{max_unresolved_drop_ratio:.4f}"
            )
        if post_downsample_windows < min_training_windows:
            reasons.append(
                f"insufficient_training_windows:{room_key}:{post_downsample_windows}<{min_training_windows}"
            )
        return {
            "pass": len(reasons) == 0,
            "reasons": reasons,
            "observed_day_count": int(observed_day_count),
            "raw_samples": int(raw_samples),
            "post_gap_samples": int(post_gap_samples),
            "post_downsample_windows": int(post_downsample_windows),
            "retained_ratio": float(retained_ratio),
            "dropped_ratio": float(unresolved_drop_ratio),
            "min_observed_days": int(min_observed_days),
            "min_post_gap_rows": int(min_post_gap_rows),
            "max_unresolved_drop_ratio": float(max_unresolved_drop_ratio),
            "min_training_windows": int(min_training_windows),
        }

    @staticmethod
    def _extract_observed_day_set(df: pd.DataFrame) -> set[pd.Timestamp]:
        """Return day set from observed timestamps before dense resampling."""
        if df is None or df.empty or 'timestamp' not in df.columns:
            return set()
        ts = pd.to_datetime(df['timestamp'], errors='coerce').dropna()
        if ts.empty:
            return set()
        return set(ts.dt.floor('D').tolist())

    @staticmethod
    def _filter_to_observed_days(
        processed_df: pd.DataFrame,
        observed_days: set[pd.Timestamp],
        room_name: str = "",
    ) -> pd.DataFrame:
        """
        Remove synthetic gap-day rows introduced by dense resampling.

        This keeps training distribution aligned with walk-forward evaluation,
        which already excludes non-observed calendar days.
        """
        if (
            processed_df is None
            or processed_df.empty
            or 'timestamp' not in processed_df.columns
            or not observed_days
        ):
            return processed_df

        ts = pd.to_datetime(processed_df['timestamp'], errors='coerce')
        keep_mask = ts.dt.floor('D').isin(list(observed_days))
        filtered = processed_df.loc[keep_mask].copy().reset_index(drop=True)
        dropped = int(len(processed_df) - len(filtered))
        if dropped > 0:
            room_msg = f" for {room_name}" if room_name else ""
            logger.info(f"Filtered {dropped} synthetic gap-day rows from training input{room_msg}.")
        return filtered

    def _evaluate_gates_unified(
        self,
        room_name: str,
        df: pd.DataFrame,
        elder_id: str,
        seq_length: int,
        observed_days: set,
        progress_callback=None,
    ) -> dict:
        """
        PR-2: Evaluate pre-training gates using GateIntegrationPipeline.
        
        [P0] This ONLY evaluates gates - does NOT train.
        Returns standardized result dict with gate_stack markers.
        Caller must train separately if gates pass.
        
        Gate Stack:
        1. CoverageContractGate
        2. PostGapRetentionGate
        3. ClassCoverageGate
        """
        try:
            policy = self.trainer._active_policy()
        except Exception:
            policy = None

        gate_pipeline = GateIntegrationPipeline(policy=policy)

        try:
            # Preprocess for gate evaluation
            max_gap = self._resolve_resample_gap_policy()
            processed = self.platform.preprocess_without_scaling(
                df, room_name, is_training=True,
                apply_denoising=False,
                max_ffill_gap_seconds=max_gap,
            )

            # Filter to observed days
            if 'timestamp' in processed.columns and observed_days:
                ts = pd.to_datetime(processed['timestamp'], errors='coerce')
                keep_mask = ts.dt.floor('D').isin(list(observed_days))
                processed = processed.loc[keep_mask].copy().reset_index(drop=True)

            result = gate_pipeline.evaluate_pre_training_gates(
                room_name=room_name,
                elder_id=elder_id,
                df=df,
                processed_df=processed,
                observed_days=observed_days,
                seq_length=seq_length,
            )
            result_dict = result.to_dict()
            result_dict.setdefault("gate_pass", bool(result_dict.get("pre_training_pass", False)))
            result_dict.setdefault("gate_reasons", [])
            return result_dict
        except Exception as e:
            logger.error(f"Gate evaluation failed for {room_name}: {e}", exc_info=True)
            return {
                "room": room_name,
                "gate_pass": False,
                "pre_training_pass": False,
                "post_training_pass": False,
                "overall_pass": False,
                "gate_reasons": [f"gate_evaluation_error:{e}"],
                "gate_stack": [],
                "rejection_artifact": None,
                "failure_stage": "gate_evaluation",
            }
    
    def _evaluate_post_training_gates(
        self,
        room_name: str,
        metrics: dict,
        gate_stack: list,
        gate_reasons: list,
    ) -> tuple[bool, list]:
        """
        PR-2: Evaluate post-training gates (StatisticalValidityGate).
        
        Returns (gate_pass, updated_gate_reasons).
        """
        # Post-training gating is handled inside TrainingPipeline.train_room().
        # Keep this wrapper for compatibility with legacy callers/tests.
        return bool(metrics.get("gate_pass", True)), list(metrics.get("gate_reasons", gate_reasons))

    def train_and_predict(self, file_path: str, elder_id: str, progress_callback=None) -> tuple:
        """
        Train models on the provided data, then run prediction on it.
        
        Args:
            file_path: Path to training file
            elder_id: Resident ID
            progress_callback: Optional callable(percent, message) for UI updates
            
        Returns:
            tuple: (corrected_results dict, trained_rooms list)
        """
        if progress_callback: progress_callback(0, "Starting training pipeline...")
        logger.info(f"Starting training pipeline for {elder_id} using {file_path}")
        
        # 1. Load Data (with resampling to ensure clean 10s intervals)
        from utils.data_loader import load_sensor_data
        max_gap = self._resolve_resample_gap_policy()
        training_data = load_sensor_data(
            file_path,
            resample=True,
            max_ffill_gap_seconds=max_gap,
        )
        if not training_data:
            raise ValueError("No valid data found in file")

        # 1.5 Merge Golden Samples into Training Data (Beta 5.5 Enhancement)
        # Golden Sample corrections override file labels for training.
        # This allows the model to learn from human-verified labels.
        from ml.utils import fetch_all_golden_samples
        golden_samples = fetch_all_golden_samples(elder_id)
        
        if golden_samples is not None and not golden_samples.empty:
            from utils.room_utils import normalize_room_name
            
            for room_name, df in training_data.items():
                if 'activity' not in df.columns:
                    continue
                    
                # Filter golden samples for this room
                norm_room = normalize_room_name(room_name)
                room_corrections = golden_samples[
                    golden_samples['room'].apply(normalize_room_name) == norm_room
                ]
                
                if room_corrections.empty:
                    continue
                
                # Ensure timestamps are datetime and sorted
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                room_corrections = room_corrections.sort_values('timestamp').reset_index(drop=True)
                
                # *** TIMEZONE NORMALIZATION (Using centralized utility) ***
                from utils.time_utils import ensure_naive
                df['timestamp'] = ensure_naive(df['timestamp'])
                room_corrections = room_corrections.copy()
                room_corrections['timestamp'] = ensure_naive(room_corrections['timestamp'])
                
                # Use merge_asof to align corrections to training timestamps (30s tolerance)
                merged = pd.merge_asof(
                    df[['timestamp']],
                    room_corrections[['timestamp', 'activity']].rename(columns={'activity': 'golden_activity'}),
                    on='timestamp',
                    direction='nearest',
                    tolerance=pd.Timedelta(seconds=30)
                )
                
                # Override 'activity' where golden correction exists
                mask = merged['golden_activity'].notna()
                override_count = mask.sum()
                
                if override_count > 0:
                    df.loc[mask, 'activity'] = merged.loc[mask, 'golden_activity'].values
                    logger.info(f"[Golden Sample] Merged {override_count} corrections into {room_name} training data")
                
                training_data[room_name] = df

        viability_policy = self._resolve_data_viability_policy()
        event_first_shadow = self._is_event_first_shadow_enabled()
        # 2. Setup
        models_dir = self._get_models_dir(elder_id)
        trained_rooms = []
        
        # TODO(PR-3.3): Proper no-op detection implementation
        # - Read policy.skip_if_same_data_and_policy from TrainingPolicy
        # - Initialize ReproducibilityTracker in constructor with proper history_dir
        # - Create reproducibility report after successful training
        # - Return contract must match: list of room metric dicts, not just room names
        # - Must maintain backward compatibility with existing callers
        
        # 3. Training Loop
        total_rooms = len(training_data)
        for i, (room_name, df) in enumerate(training_data.items()):
            if 'activity' not in df.columns:
                logger.warning(f"Room {room_name} missing 'activity' column, skipping training.")
                continue
            observed_days = self._extract_observed_day_set(df)

            if progress_callback:
                p_start = 10 + int(60 * (i / total_rooms))
                p_end = 10 + int(60 * ((i + 1) / total_rooms))
                progress_callback(p_start, f"Training model for {room_name}...")
                
                # Create a sub-callback for the trainer to report epoch progress within its range
                def room_progress(percent, msg):
                    granular_p = p_start + int((p_end - p_start) * (percent / 100))
                    progress_callback(granular_p, f"{room_name}: {msg}")
            else:
                room_progress = None

            logger.info(f"Training model for room: {room_name}")
            
            try:
                # [P1] Apply denoising BEFORE gate evaluation (parity with train_from_files)
                if self.enable_denoising:
                    from elderlycare_v1_16.preprocessing.noise import hampel_filter, clip_outliers
                    if self.denoising_method == 'hampel':
                        hampel_filter(df, self.platform.sensor_columns, 
                                      window=self.denoising_window, 
                                      n_sigmas=self.denoising_threshold,
                                      inplace=True)
                    elif self.denoising_method == 'clip':
                        clip_outliers(df, self.platform.sensor_columns,
                                      method='mad', factor=self.denoising_threshold,
                                      window=self.denoising_window,
                                      inplace=True)

                # PR-1: Evaluate gates first, then train if passed
                seq_length = get_room_config().calculate_seq_length(room_name)
                
                gate_result = self._evaluate_gates_unified(
                    room_name=room_name,
                    df=df,
                    elder_id=elder_id,
                    seq_length=seq_length,
                    observed_days=observed_days,
                    progress_callback=room_progress,
                )
                
                pre_training_pass = bool(
                    gate_result.get("pre_training_pass", gate_result.get("gate_pass", False))
                )
                if not pre_training_pass:
                    # [P0] Gates failed - do not train, record failure
                    logger.warning(
                        f"Pre-training gates failed for {room_name}: {gate_result.get('gate_reasons', [])}"
                    )
                    failed_metrics = {
                        "room": room_name,
                        "gate_pass": False,
                        "gate_reasons": gate_result.get("gate_reasons", []),
                        "gate_stack": gate_result.get("gate_stack", []),
                        "rejection_artifact": gate_result.get("rejection_artifact"),
                        "accuracy": 0.0,
                        "samples": len(df),
                        "training_days": 0.0,
                    }
                    timeline_shadow = self._maybe_write_timeline_shadow_artifacts(
                        elder_id=elder_id,
                        room_name=room_name,
                        room_df=df,
                        context="train_and_predict",
                    )
                    if timeline_shadow is not None:
                        failed_metrics["timeline_shadow"] = timeline_shadow
                    trained_rooms.append(failed_metrics)
                    continue
                
                # Beta 6 policy: leakage-free preprocessing is mandatory.
                metrics = self.trainer.train_room_with_leakage_free_scaling(
                    room_name=room_name,
                    raw_df=df,
                    seq_length=seq_length,
                    elder_id=elder_id,
                    progress_callback=room_progress,
                    gate_evaluation_result=gate_result,
                    event_first_shadow=event_first_shadow,
                    validation_split=DEFAULT_VALIDATION_SPLIT,
                )
                
                if metrics:
                    timeline_shadow = self._maybe_write_timeline_shadow_artifacts(
                        elder_id=elder_id,
                        room_name=room_name,
                        room_df=df,
                        context="train_and_predict",
                    )
                    if timeline_shadow is not None and isinstance(metrics, dict):
                        metrics["timeline_shadow"] = timeline_shadow
                    trained_rooms.append(metrics)
                    
            except Exception as e:
                logger.error(f"Failed to train {room_name}: {e}", exc_info=True)

        logger.info(f"Training completed. Trained rooms: {trained_rooms}")
        
        # 4. Prepare results using Ground Truth (Beta 5.5 Change)
        # For training files, we use the original 'activity' labels directly instead of running predictions.
        # This ensures the timeline reflects exactly what was in the training file.
        if progress_callback: progress_callback(80, "Preparing ground truth results...")
        
        trained_room_names = [r['room'] for r in trained_rooms]
        if not trained_room_names:
            logger.warning("No models were trained. Skipping result preparation.")
            if progress_callback: progress_callback(100, "Done.")
            return {}, []

        # Convert training data to prediction format (activity -> predicted_activity)
        ground_truth_results = {}
        for room_name in trained_room_names:
            if room_name in training_data:
                df = training_data[room_name].copy()
                if 'activity' in df.columns:
                    df['predicted_activity'] = df['activity']
                    df['confidence'] = 1.0  # Ground truth has 100% confidence
                    ground_truth_results[room_name] = df
                    logger.info(f"[Ground Truth] Using training labels for {room_name} ({len(df)} events)")
                else:
                    logger.warning(f"No 'activity' column for {room_name}, falling back to prediction.")
                    # Fallback to prediction for this room only
                    fallback = self.predictor.run_prediction({room_name: training_data[room_name]}, [room_name], seq_length=0)
                    if fallback:
                        ground_truth_results[room_name] = fallback.get(room_name)
        
        # Apply golden samples (corrections take priority over ground truth)
        corrected_results = self.predictor.apply_golden_samples(ground_truth_results, elder_id)
        
        if progress_callback: progress_callback(100, "Done.")
        return corrected_results, trained_rooms

    def predict(self, file_path: str, elder_id: str) -> dict:
        """
        Load existing models and run prediction on new data.
        
        Returns:
            dict: Room name -> DataFrame of predictions
        """
        logger.info(f"Starting prediction pipeline for {elder_id} using {file_path}")
        
        # 1. Load Data (with resampling to ensure clean 10s intervals)
        from utils.data_loader import load_sensor_data
        max_gap = self._resolve_resample_gap_policy()
        pred_data = load_sensor_data(
            file_path,
            resample=True,
            max_ffill_gap_seconds=max_gap,
        )
        if not pred_data:
            raise ValueError("No valid data found in file")
            
        # 1.5 Denoise Data (Apply Hampel Filter to remove spikes)
        # Check if we should apply it (usually yes for sensors)
        try:
            from elderlycare_v1_16.preprocessing.noise import hampel_filter
            logger.info("Applying Hampel Filter for denoising...")
            
            for room, df in pred_data.items():
                # Identify sensor columns present in df
                cols_to_denoise = [c for c in DEFAULT_SENSOR_COLUMNS if c in df.columns]
                if cols_to_denoise:
                    # Apply inplace
                    hampel_filter(df, columns=cols_to_denoise, window=self.denoising_window, n_sigmas=self.denoising_threshold, inplace=True)
                    logger.debug(f"Denoised {room} columns: {cols_to_denoise}")
        except Exception as e:
            logger.warning(f"Denoising failed: {e}. Proceeding with raw data.")

        # 2. Load Models
        loaded_rooms = self.registry.load_models_for_elder(elder_id, self.platform)
        if not loaded_rooms:
            logger.warning("No models could be loaded.")
            return {}
            
        # 3. Predict & Apply Golden Samples
        with self._beta6_phase4_runtime_prediction_context(
            elder_id=elder_id,
            loaded_rooms=[str(r) for r in loaded_rooms],
        ):
            raw_results = self.predictor.run_prediction(pred_data, loaded_rooms, seq_length=0) # seq_length ignored
        final_results = self.predictor.apply_golden_samples(raw_results, elder_id)
        
        # 4. Data Integrity Check (Audit Point 7)
        # Prevents "Garbage In, Garbage Out"
        from ml.validation import run_validation
        is_valid = run_validation(final_results)
        
        if not is_valid:
            logger.error(
                "Data Integrity Validation Failed. Aborting prediction flow to prevent invalid persistence."
            )
            raise DataValidationError(
                "Prediction integrity validation failed; refusing to proceed with invalid results."
            )
            
        return final_results



    def train_from_files(
        self,
        file_paths,
        elder_id,
        progress_callback=None,
        rooms=None,
        training_mode="full_retrain",
        defer_promotion: bool = False,
    ):
        """
        Train models on aggregated data from multiple files.
        This prevents "catastrophic forgetting" by training on ALL available data.
        
        Args:
            file_paths: List of paths to training files (.xlsx or .parquet)
            elder_id: Resident ID
            progress_callback: Optional callable(percent, message)
            rooms: Optional set of room names to scope training.
                   If provided, only these rooms' models are retrained.
                   Prevents cross-room contamination from non-deterministic retraining.
            training_mode: "full_retrain" (default) or "correction_fine_tune".
            defer_promotion: If True, save candidate versions without promoting to latest.
            
        Returns:
            dict with trained_rooms metrics
        """
        from utils.data_loader import load_sensor_data
        
        # Normalize room filter for consistent matching
        normalized_filter = {normalize_room_name(r) for r in rooms} if rooms else None
        scope_msg = f" (scoped to {rooms})" if rooms else " (all rooms)"
        logger.info(f"Starting aggregated training for {elder_id} using {len(file_paths)} files{scope_msg}")
        
        # 1. Load and aggregate data from all files
        aggregated_data = {}  # {room: [dataframes]}
        room_source_entries = {}  # {room: [{file_path, source_order, df}]}
        
        if progress_callback: progress_callback(5, f"Loading {len(file_paths)} files for aggregated training...")
        max_gap = self._resolve_resample_gap_policy()
        viability_policy = self._resolve_data_viability_policy()
        event_first_shadow = self._is_event_first_shadow_enabled()
        for i, file_path in enumerate(file_paths):
            try:
                if progress_callback:
                    progress_callback(5 + int(15 * (i / len(file_paths))), f"Loading {Path(file_path).name}...")
                data = load_sensor_data(
                    file_path,
                    resample=True,
                    max_ffill_gap_seconds=max_gap,
                )
                for room_name, df in data.items():
                    if 'activity' not in df.columns:
                        continue
                    df = df.copy()
                    df["__source_file_order"] = int(i)
                    df["__source_row_order"] = np.arange(len(df), dtype=np.int64)
                    df["__source_file_name"] = Path(file_path).name
                    if room_name not in aggregated_data:
                        aggregated_data[room_name] = []
                    if room_name not in room_source_entries:
                        room_source_entries[room_name] = []
                    aggregated_data[room_name].append(df)
                    room_source_entries[room_name].append(
                        {
                            "file_path": str(Path(file_path).expanduser().resolve()),
                            "source_order": int(i),
                            "df": df.copy(),
                        }
                    )
                    logger.info(f"  Loaded {len(df)} rows from {Path(file_path).name} ({room_name})")
            except Exception as e:
                logger.critical(f"ALERT: failed to load/resample training file {file_path}: {e}", exc_info=True)
                raise ModelTrainError(f"Training aborted due to load/resample failure for {file_path}: {e}") from e
        
        if not aggregated_data:
            raise ValueError("No valid training data found in any file")
        
        # 2. Combine dataframes per room
        combined_data = {}
        room_source_lineage = {}
        for room_name, dfs in aggregated_data.items():
            combined = pd.concat(dfs, ignore_index=True)
            # Ensure timestamps are datetime before sorting (fixes mixed str/Timestamp type error)
            if 'timestamp' in combined.columns:
                combined['timestamp'] = pd.to_datetime(combined['timestamp'])
            # Deterministic precedence for duplicate timestamps:
            # lower source file order wins (explicit file priority from caller), then row order.
            combined = combined.sort_values(
                ['timestamp', '__source_file_order', '__source_row_order'],
                kind='stable',
            )
            before = len(combined)
            combined = combined.drop_duplicates(subset=['timestamp'], keep='first')
            dropped = int(before - len(combined))
            if dropped > 0:
                logger.info(
                    f"  Resolved {dropped} duplicate-timestamp rows in {room_name} "
                    "using explicit source precedence."
                )
            drop_cols = [c for c in ['__source_file_order', '__source_row_order', '__source_file_name'] if c in combined.columns]
            if drop_cols:
                combined = combined.drop(columns=drop_cols)
            combined_data[room_name] = combined
            room_source_lineage[room_name] = self._build_room_source_lineage(
                room_name=room_name,
                source_entries=room_source_entries.get(room_name, []),
                combined_df=combined,
            )
            logger.info(f"  Combined {room_name}: {len(combined)} total rows")
        
        # 3. Setup
        models_dir = self._get_models_dir(elder_id)
        trained_rooms = []
        
        # 4. Training Loop (same as train_and_predict but with combined data)
        # Early filter: only train rooms in the corrected set
        rooms_to_train = {k: v for k, v in combined_data.items() 
                          if not normalized_filter or normalize_room_name(k) in normalized_filter}
        if normalized_filter:
            skipped = set(combined_data.keys()) - set(rooms_to_train.keys())
            if skipped:
                logger.info(f"Skipping unchanged rooms: {skipped}")
        total_rooms = len(rooms_to_train)
        for i, (room_name, df) in enumerate(rooms_to_train.items()):
            observed_days = self._extract_observed_day_set(df)
            if progress_callback:
                p_start = 20 + int(60 * (i / total_rooms))
                p_end = 20 + int(60 * ((i + 1) / total_rooms))
                progress_callback(p_start, f"Training model for {room_name}...")
                
                def room_progress(percent, msg):
                    granular_p = p_start + int((p_end - p_start) * (percent / 100))
                    progress_callback(granular_p, f"{room_name}: {msg}")
            else:
                room_progress = None

            logger.info(f"Training model for room: {room_name} ({len(df)} samples)")
            try:
                # [P1] Apply denoising BEFORE gate evaluation (parity with train_and_predict)
                if self.enable_denoising:
                    from elderlycare_v1_16.preprocessing.noise import hampel_filter, clip_outliers
                    if self.denoising_method == 'hampel':
                        hampel_filter(df, self.platform.sensor_columns, 
                                      window=self.denoising_window, 
                                      n_sigmas=self.denoising_threshold,
                                      inplace=True)
                    elif self.denoising_method == 'clip':
                        clip_outliers(df, self.platform.sensor_columns,
                                      method='mad', factor=self.denoising_threshold,
                                      window=self.denoising_window,
                                      inplace=True)
                
                # PR-1: Evaluate gates first, then train if passed (same as train_and_predict)
                seq_length = get_room_config().calculate_seq_length(room_name)
                
                gate_result = self._evaluate_gates_unified(
                    room_name=room_name,
                    df=df,
                    elder_id=elder_id,
                    seq_length=seq_length,
                    observed_days=observed_days,
                    progress_callback=room_progress,
                )
                
                pre_training_pass = bool(
                    gate_result.get("pre_training_pass", gate_result.get("gate_pass", False))
                )
                if not pre_training_pass:
                    # [P0] Gates failed - do not train, record failure
                    logger.warning(
                        f"Pre-training gates failed for {room_name}: {gate_result.get('gate_reasons', [])}"
                    )
                    training_days = 0.0
                    if 'timestamp' in df.columns and not df.empty:
                        try:
                            ts = pd.to_datetime(df['timestamp'])
                            span_seconds = (ts.max() - ts.min()).total_seconds()
                            training_days = max(0.0, float(span_seconds) / 86400.0)
                        except Exception:
                            training_days = 0.0
                    
                    failed_metrics = {
                        "room": room_name,
                        "gate_pass": False,
                        "gate_reasons": gate_result.get("gate_reasons", []),
                        "gate_stack": gate_result.get("gate_stack", []),
                        "rejection_artifact": gate_result.get("rejection_artifact"),
                        "accuracy": 0.0,
                        "samples": len(df),
                        "training_days": float(training_days),
                        "retained_sample_ratio": 0.0,
                    }
                    timeline_shadow = self._maybe_write_timeline_shadow_artifacts(
                        elder_id=elder_id,
                        room_name=room_name,
                        room_df=df,
                        context="train_from_files",
                    )
                    if timeline_shadow is not None:
                        failed_metrics["timeline_shadow"] = timeline_shadow
                    trained_rooms.append(failed_metrics)
                    continue
                
                # Beta 6 policy: leakage-free preprocessing is mandatory.
                metrics = self.trainer.train_room_with_leakage_free_scaling(
                    room_name=room_name,
                    raw_df=df,
                    seq_length=seq_length,
                    elder_id=elder_id,
                    progress_callback=room_progress,
                    training_mode=training_mode,
                    defer_promotion=defer_promotion,
                    gate_evaluation_result=gate_result,
                    event_first_shadow=event_first_shadow,
                    source_lineage=room_source_lineage.get(room_name),
                    validation_split=DEFAULT_VALIDATION_SPLIT,
                )
                
                if metrics:
                    timeline_shadow = self._maybe_write_timeline_shadow_artifacts(
                        elder_id=elder_id,
                        room_name=room_name,
                        room_df=df,
                        context="train_from_files",
                    )
                    if timeline_shadow is not None and isinstance(metrics, dict):
                        metrics["timeline_shadow"] = timeline_shadow
                    trained_rooms.append(metrics)

            except Exception as e:
                logger.critical(f"ALERT: failed to train {room_name}: {e}", exc_info=True)
                raise ModelTrainError(f"Training aborted for room {room_name}: {e}") from e

        logger.info(f"Aggregated training completed. Trained {len(trained_rooms)} rooms.")
        # Return tuple to match train_and_predict signature (predictions, metrics)
        # We don't return predictions here as we are just training
        return {}, trained_rooms

    def repredict_all(self, elder_id, archive_dir, progress_callback=None, rooms=None):
        """
        Re-run predictions on archived input files using the current model.
        Updates the database WITHOUT creating new files.
        
        Args:
            rooms: Optional set of room names to scope reprediction.
        """
        if progress_callback: progress_callback(85, "Starting historical re-prediction...")
        # Delegate to PredictionPipeline (pass rooms filter through)
        return self.predictor.repredict_all(elder_id, archive_dir, progress_callback=progress_callback, rooms=rooms)
