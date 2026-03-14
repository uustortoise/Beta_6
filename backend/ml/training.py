import logging
import numpy as np
import tensorflow as tf
import os
import json
import hashlib
import shutil
import joblib
import pandas as pd
import re
import random
import uuid
import copy
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Mapping, Sequence

# Import adapter
try:
    from backend.db.legacy_adapter import LegacyDatabaseAdapter
except ImportError:
    from elderlycare_v1_16.database import db as adapter
else:
    adapter = LegacyDatabaseAdapter()
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

# Shared Utilities
from ml.utils import calculate_sequence_length, fetch_golden_samples, fetch_sensor_windows_batch

from elderlycare_v1_16.config.settings import (
    DEFAULT_EPOCHS, DEFAULT_VALIDATION_SPLIT, DEFAULT_DROPOUT_RATE,
    DEFAULT_CONFIDENCE_THRESHOLD, ARCHIVE_DATA_DIR
)
from config import get_room_config, get_release_gates_config, DB_PATH
from ml.exceptions import ModelTrainError
from ml.release_gates import resolve_scheduled_threshold
from ml.registry import ModelRegistry
from ml.transformer_backbone import build_transformer_model
from ml.policy_config import TrainingPolicy, load_policy_from_env
from utils.data_loader import load_sensor_data
from utils.room_utils import normalize_room_name

# PR-2: Gate Integration
from ml.gate_integration import GateIntegrationPipeline, GateEvaluationResult, PreTrainingGateResult

# PR-3: Correctness Fixes
from ml.sequence_alignment import safe_create_sequences, SequenceLabelAlignmentError
from ml.duplicate_resolution import DuplicateTimestampResolver, DuplicateResolutionPolicy
from ml.reproducibility_report import ReproducibilityTracker, compute_data_fingerprint, get_code_version, RunOutcome
from ml.transformer_head_ab import derive_dual_head_probabilities
from ml.event_decoder import EventDecoder, DecoderConfig
from ml.event_kpi import EventKPICalculator, EventKPIConfig
from ml.event_gates import CriticalityTier, EventGateChecker, EventGateThresholds
from ml.label_taxonomy import (
    get_critical_labels_for_room,
    get_label_alias_equivalents,
    get_lane_b_event_labels_for_room,
)
from ml.policy_defaults import (
    get_timeline_native_rooms_default,
    get_training_min_holdout_support_by_room,
    get_training_min_holdout_support_default,
)
from ml.timeline_targets import build_boundary_targets, build_episode_attribute_targets
from ml.transformer_timeline_heads import (
    TransformerTimelineHeads,
    TimelineHeadConfig,
    create_timeline_heads,
)
from ml.beta6.sequence.hmm_decoder import DEFAULT_DECODER_POLICY

logger = logging.getLogger(__name__)

def _utc_now_iso_z() -> str:
    """Return timezone-aware UTC timestamp with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _GateAlignedCheckpointCallback(tf.keras.callbacks.Callback):
    """
    Restore the checkpoint with the strongest gate-aligned validation score.
    """

    def __init__(
        self,
        *,
        pipeline: "TrainingPipeline",
        room_name: str,
        X_val: np.ndarray,
        y_val: np.ndarray,
        timeline_multitask_enabled: bool,
        no_regress_macro_f1_floor: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.room_name = room_name
        self.X_val = np.asarray(X_val, dtype=np.float32)
        self.y_val = np.asarray(y_val, dtype=np.int32)
        self.timeline_multitask_enabled = bool(timeline_multitask_enabled)
        self.no_regress_macro_f1_floor = (
            float(no_regress_macro_f1_floor)
            if no_regress_macro_f1_floor is not None and np.isfinite(float(no_regress_macro_f1_floor))
            else None
        )
        self.best_score = float("-inf")
        self.best_epoch = -1
        self.best_weights = None
        self.best_summary: Dict[str, Any] = {}
        self.best_floor_score = float("-inf")
        self.best_floor_epoch = -1
        self.best_floor_weights = None
        self.best_floor_summary: Dict[str, Any] = {}
        self.best_macro_f1 = float("-inf")
        self.best_macro_epoch = -1
        self.best_macro_weights = None
        self.best_macro_summary: Dict[str, Any] = {}
        self.selection_mode = "overall_best"
        self.last_summary: Dict[str, Any] = {}
        self.last_error: Optional[str] = None

    def on_epoch_end(self, epoch, logs=None):
        if self.X_val.size == 0 or self.y_val.size == 0:
            return
        try:
            raw_pred = self.model.predict(self.X_val, verbose=0)
            y_pred_probs = self.pipeline._extract_activity_probabilities(
                raw_pred,
                timeline_multitask_enabled=self.timeline_multitask_enabled,
            )
            summary = self.pipeline._summarize_gate_aligned_validation(
                room_name=self.room_name,
                y_true=self.y_val,
                y_pred_probs=y_pred_probs,
            )
            score = float(summary.get("gate_aligned_score", float("-inf")))
            macro_f1 = summary.get("macro_f1")
            passes_floor = True
            if self.no_regress_macro_f1_floor is not None:
                if macro_f1 is None:
                    passes_floor = False
                else:
                    passes_floor = float(macro_f1) >= (float(self.no_regress_macro_f1_floor) - 1e-8)
            self.last_summary = {
                "epoch": int(epoch),
                "score": score,
                "passes_no_regress_floor": bool(passes_floor),
                "no_regress_macro_f1_floor": self.no_regress_macro_f1_floor,
                **summary,
            }
            if score > (self.best_score + 1e-8):
                self.best_score = score
                self.best_epoch = int(epoch)
                self.best_weights = self.model.get_weights()
                self.best_summary = dict(self.last_summary)
            if passes_floor and score > (self.best_floor_score + 1e-8):
                self.best_floor_score = score
                self.best_floor_epoch = int(epoch)
                self.best_floor_weights = self.model.get_weights()
                self.best_floor_summary = dict(self.last_summary)
            if macro_f1 is not None:
                macro_f1_value = float(macro_f1)
                better_macro = macro_f1_value > (self.best_macro_f1 + 1e-8)
                tie_macro_better_score = (
                    abs(macro_f1_value - self.best_macro_f1) <= 1e-8
                    and score > float(self.best_macro_summary.get("score", float("-inf")))
                )
                if better_macro or tie_macro_better_score:
                    self.best_macro_f1 = macro_f1_value
                    self.best_macro_epoch = int(epoch)
                    self.best_macro_weights = self.model.get_weights()
                    self.best_macro_summary = dict(self.last_summary)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"

    def on_train_end(self, logs=None):
        if self.best_floor_weights is not None:
            self.model.set_weights(self.best_floor_weights)
            self.best_score = float(self.best_floor_score)
            self.best_epoch = int(self.best_floor_epoch)
            self.best_weights = self.best_floor_weights
            self.best_summary = dict(self.best_floor_summary)
            self.selection_mode = "no_regress_floor"
        elif (
            self.no_regress_macro_f1_floor is not None
            and self.best_macro_weights is not None
            and self.pipeline._resolve_gate_aligned_no_regress_fallback_mode() == "macro_f1"
        ):
            self.model.set_weights(self.best_macro_weights)
            self.best_score = float(self.best_macro_summary.get("score", float("-inf")))
            self.best_epoch = int(self.best_macro_epoch)
            self.best_weights = self.best_macro_weights
            self.best_summary = dict(self.best_macro_summary)
            self.selection_mode = "no_regress_macro_f1_fallback"
        elif self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            self.selection_mode = "overall_best"

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "enabled": True,
            "best_epoch": int(self.best_epoch),
            "best_score": float(self.best_score) if np.isfinite(self.best_score) else None,
            "best_summary": dict(self.best_summary),
            "last_summary": dict(self.last_summary),
            "selection_mode": str(self.selection_mode),
            "no_regress_macro_f1_floor": self.no_regress_macro_f1_floor,
            "best_floor_epoch": int(self.best_floor_epoch) if self.best_floor_epoch >= 0 else None,
            "best_floor_score": float(self.best_floor_score) if np.isfinite(self.best_floor_score) else None,
            "best_floor_summary": dict(self.best_floor_summary),
            "best_macro_epoch": int(self.best_macro_epoch) if self.best_macro_epoch >= 0 else None,
            "best_macro_f1": float(self.best_macro_f1) if np.isfinite(self.best_macro_f1) else None,
            "best_macro_summary": dict(self.best_macro_summary),
        }
        if self.last_error:
            payload["error"] = self.last_error
        return payload

class TrainingPipeline:
    """
    Manages data augmentation, sequence creation, and model training.
    """
    def __init__(self, platform: Any, registry: ModelRegistry, policy: Optional[TrainingPolicy] = None):
        self.platform = platform
        self.registry = registry
        self.room_config = get_room_config()
        self._use_env_policy = policy is None
        self.policy = policy or load_policy_from_env()
        self._policy_snapshot: Optional[TrainingPolicy] = None
        self._last_calibration_debug: List[Dict[str, Any]] = []
        self._last_release_gate_watch_reasons: List[str] = []
        self._last_unoccupied_downsample_debug: Dict[str, Any] = {}
        self._seed_panel_debug_context: Optional[Dict[str, Any]] = None
        
        # macOS Optimization: Threading and Memory Growth
        try:
            # 1. Thread limits to prevent exhaustion
            tf.config.threading.set_intra_op_parallelism_threads(2)
            tf.config.threading.set_inter_op_parallelism_threads(2)
            
            # 2. GPU Memory Growth (Crucial for macOS Metal stability)
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            logger.info("TensorFlow optimized for macOS: Threads=2, MemoryGrowth=True")
        except Exception as e:
            logger.debug(f"Could not apply TF optimizations: {e}")
            
        # P2 Hardening: Enforce deterministic seeding
        self._set_training_random_seed(self.policy.reproducibility.random_seed)

    def _set_training_random_seed(self, seed: int):
        """
        Set global random seeds for reproducibility.
        Enforces AC-7: Determinism.
        """
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logger.info(f"Global random seed set to {seed}")

    def _active_policy(self) -> TrainingPolicy:
        """
        Return currently effective training policy.

        If no explicit policy was injected, refresh from env for backward
        compatibility with existing runtime env toggles.
        """
        if self._policy_snapshot is not None:
            return self._policy_snapshot
        if self._use_env_policy:
            self.policy = load_policy_from_env()
        return self.policy

    @staticmethod
    def _policy_hash(policy: TrainingPolicy) -> str:
        """Stable hash for an effective training policy."""
        payload = json.dumps(policy.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return bool(default)
        return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}

    @staticmethod
    def _env_float(
        name: str,
        default: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        raw = os.getenv(name)
        try:
            value = float(raw) if raw is not None else float(default)
        except (TypeError, ValueError):
            value = float(default)
        if min_value is not None:
            value = max(float(min_value), value)
        if max_value is not None:
            value = min(float(max_value), value)
        return float(value)

    def _resolve_gate_aligned_checkpoint_enabled(self) -> bool:
        return self._env_bool("ENABLE_GATE_ALIGNED_CHECKPOINT", True)

    def _resolve_gate_aligned_critical_support_floor(self) -> int:
        raw = os.getenv("GATE_ALIGNED_CRITICAL_SUPPORT_FLOOR")
        try:
            value = int(raw) if raw is not None else 10
        except (TypeError, ValueError):
            value = 10
        return max(1, int(value))

    def _resolve_gate_aligned_critical_recall_floor(self) -> float:
        return self._env_float(
            "GATE_ALIGNED_CRITICAL_RECALL_FLOOR",
            0.20,
            min_value=0.0,
            max_value=1.0,
        )

    def _resolve_gate_aligned_collapse_ratio(self) -> float:
        return self._env_float(
            "GATE_ALIGNED_COLLAPSE_RATIO",
            0.95,
            min_value=0.50,
            max_value=1.0,
        )

    def _resolve_gate_aligned_critical_weight(self) -> float:
        return self._env_float(
            "GATE_ALIGNED_CRITICAL_WEIGHT",
            0.35,
            min_value=0.0,
            max_value=2.0,
        )

    def _resolve_gate_aligned_collapse_penalty(self) -> float:
        return self._env_float(
            "GATE_ALIGNED_COLLAPSE_PENALTY",
            0.45,
            min_value=0.0,
            max_value=2.0,
        )

    def _resolve_gate_aligned_floor_penalty(self) -> float:
        return self._env_float(
            "GATE_ALIGNED_CRITICAL_FLOOR_PENALTY",
            0.25,
            min_value=0.0,
            max_value=2.0,
        )

    def _resolve_gate_aligned_lane_b_weight(self) -> float:
        return self._env_float(
            "GATE_ALIGNED_LANE_B_WEIGHT",
            0.50,
            min_value=0.0,
            max_value=3.0,
        )

    def _resolve_gate_aligned_lane_b_floor_penalty(self) -> float:
        return self._env_float(
            "GATE_ALIGNED_LANE_B_FLOOR_PENALTY",
            0.35,
            min_value=0.0,
            max_value=3.0,
        )

    @staticmethod
    def _resolve_gate_aligned_no_regress_fallback_mode() -> str:
        raw = str(os.getenv("GATE_ALIGNED_NO_REGRESS_FALLBACK_MODE", "macro_f1")).strip().lower()
        if raw in {"score", "macro_f1"}:
            return raw
        return "macro_f1"

    def _resolve_gate_aligned_precision_floor_penalty(self) -> float:
        return self._env_float(
            "GATE_ALIGNED_PRECISION_FLOOR_PENALTY",
            0.35,
            min_value=0.0,
            max_value=3.0,
        )

    def _resolve_gate_aligned_precision_floors(self, room_name: str) -> Dict[str, float]:
        room_key = normalize_room_name(room_name)
        floors: Dict[str, float] = {"out": 0.20} if room_key == "entrance" else {}
        raw = str(os.getenv("GATE_ALIGNED_PRECISION_FLOOR_BY_ROOM_LABEL", "")).strip()
        if not raw:
            return floors
        for token in raw.split(","):
            part = str(token).strip()
            if not part or ":" not in part or "." not in part:
                continue
            key, value = part.split(":", 1)
            room_label = str(key).strip().lower()
            if "." not in room_label:
                continue
            room_name_part, label_part = room_label.split(".", 1)
            if normalize_room_name(room_name_part) != room_key:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            floors[str(label_part).strip().lower()] = float(np.clip(parsed, 0.0, 1.0))
        return floors

    def _resolve_no_regress_macro_f1_floor(
        self,
        *,
        room_name: str,
        champion_meta: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "enabled": False,
            "target_macro_f1_floor": None,
            "max_drop_from_champion": None,
            "champion_macro_f1": None,
            "exempt": False,
        }
        if not isinstance(champion_meta, Mapping):
            return context

        champion_metrics = champion_meta.get("metrics")
        if not isinstance(champion_metrics, Mapping):
            return context

        champion_f1_raw = champion_metrics.get("macro_f1")
        if champion_f1_raw is None:
            return context

        try:
            champion_f1 = float(champion_f1_raw)
        except (TypeError, ValueError):
            return context

        if not np.isfinite(champion_f1):
            return context

        try:
            policy = get_release_gates_config()
            no_regress_cfg = (policy.get("release_gates", {}) or {}).get("no_regress", {}) or {}
            exempt_rooms = {
                normalize_room_name(room)
                for room in (no_regress_cfg.get("exempt_rooms", []) or [])
            }
            room_key = normalize_room_name(room_name)
            if room_key in exempt_rooms:
                context.update(
                    {
                        "enabled": False,
                        "champion_macro_f1": champion_f1,
                        "exempt": True,
                    }
                )
                return context

            gate_policy = self._active_policy().release_gate
            max_drop = float(
                no_regress_cfg.get(
                    "max_drop_from_champion",
                    gate_policy.max_drop_from_champion_default,
                )
            )
            context.update(
                {
                    "enabled": True,
                    "target_macro_f1_floor": float(champion_f1 - max_drop),
                    "max_drop_from_champion": float(max_drop),
                    "champion_macro_f1": float(champion_f1),
                    "exempt": False,
                }
            )
        except Exception as e:
            context["error"] = f"{type(e).__name__}: {e}"
        return context

    @staticmethod
    def _extract_activity_probabilities(raw_pred: Any, *, timeline_multitask_enabled: bool) -> np.ndarray:
        if isinstance(raw_pred, dict):
            raw_pred = raw_pred.get("activity_logits")
        if raw_pred is None:
            raise ValueError("missing_activity_logits")
        y_pred = np.asarray(raw_pred, dtype=np.float32)
        if y_pred.ndim == 0:
            y_pred = np.zeros((0, 1), dtype=np.float32)
        elif y_pred.ndim == 1:
            y_pred = np.reshape(y_pred, (-1, 1))
        elif y_pred.ndim > 2:
            y_pred = np.reshape(y_pred, (y_pred.shape[0], -1))
        if bool(timeline_multitask_enabled):
            y_pred = tf.nn.softmax(y_pred, axis=1).numpy()
        return np.asarray(y_pred, dtype=np.float32)

    def _get_label_name(self, room_name: str, class_id: int) -> str | None:
        """Resolve encoded class ID to label name for a given room."""
        le = self.platform.label_encoders.get(room_name)
        if le is None:
            return None
        classes = getattr(le, 'classes_', None)
        if classes is None:
            return None
        if class_id < 0 or class_id >= len(classes):
            return None
        return str(classes[class_id])

    def _get_precision_target(self, label_name: str | None) -> float:
        """Return per-label precision target used for threshold calibration."""
        return float(self._active_policy().calibration.get_precision_target(label_name))

    def _get_recall_floor(self, label_name: str | None) -> float:
        """Return per-label minimum recall floor for threshold selection."""
        return float(self._active_policy().calibration.get_recall_floor(label_name))

    def _default_class_thresholds(self, room_name: str) -> Dict[int, float]:
        """Build default thresholds for all classes of a room."""
        le = self.platform.label_encoders.get(room_name)
        classes = getattr(le, 'classes_', []) if le is not None else []
        if classes is None or len(classes) == 0:
            return {}
        return {int(i): float(DEFAULT_CONFIDENCE_THRESHOLD) for i in range(len(classes))}

    def _should_shuffle_post_split_batches(self, room_name: str) -> bool:
        try:
            profile = self._active_policy().training_profile
        except Exception:
            return False
        return bool(profile.should_shuffle_post_split(room_name))

    @staticmethod
    def _resolve_scheduled_threshold(schedule: List[Dict[str, Any]], training_days: float) -> Optional[float]:
        """Resolve threshold from day-based schedule via shared release-gate utility."""
        return resolve_scheduled_threshold(schedule, training_days)

    @staticmethod
    def _resolve_fine_tuning_learning_rate(raw_value: Any) -> float:
        """Map policy learning-rate hints to concrete numeric values."""
        if raw_value is None:
            return 1e-5
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        value = str(raw_value).strip().lower()
        if value == "low":
            return 1e-5
        if value == "very_low":
            return 5e-6
        if value == "medium":
            return 1e-4
        try:
            return float(value)
        except ValueError:
            return 1e-5

    def _get_fine_tuning_params(self) -> Dict[str, Any]:
        """
        Resolve correction fine-tuning parameters from release policy.
        """
        defaults = {
            "warm_start": True,
            "learning_rate": 1e-5,
            "epochs": min(int(DEFAULT_EPOCHS), 4),
            "patience": 1,
            "replay_enabled": True,
            "replay_ratio": 10,
            "replay_sampling": "random_stratified",
        }
        try:
            policy = get_release_gates_config()
            fine_tuning = policy.get("fine_tuning", {})
            replay = fine_tuning.get("replay_buffer", {})
            raw_epochs = fine_tuning.get("epochs", defaults["epochs"])
            if isinstance(raw_epochs, str):
                raw_epochs = raw_epochs.strip().lower()
                if raw_epochs == "few":
                    epochs = defaults["epochs"]
                else:
                    epochs = int(float(raw_epochs))
            else:
                epochs = int(raw_epochs)
            defaults.update({
                "warm_start": bool(fine_tuning.get("warm_start_from_champion", defaults["warm_start"])),
                "learning_rate": self._resolve_fine_tuning_learning_rate(
                    fine_tuning.get("learning_rate", defaults["learning_rate"])
                ),
                "epochs": max(1, epochs),
                "replay_enabled": bool(replay.get("enabled", defaults["replay_enabled"])),
                "replay_ratio": max(1, int(replay.get("uncorrected_to_corrected_ratio", defaults["replay_ratio"]))),
                "replay_sampling": str(replay.get("sampling", defaults["replay_sampling"])).strip().lower(),
            })
        except Exception as e:
            logger.warning(f"Failed to resolve fine-tuning policy; using defaults: {e}")
        return defaults

    def _set_training_random_seed(self, seed: int) -> None:
        """Set deterministic seeds for Python, NumPy, and TensorFlow."""
        random.seed(int(seed))
        np.random.seed(int(seed))
        tf.random.set_seed(int(seed))

    @staticmethod
    def _require_valid_activity_encoded(processed_df: pd.DataFrame, room_name: str) -> pd.DataFrame:
        """
        Fail-closed label guard for sequence training input.

        Prevents silent NaN/object coercion into class IDs by enforcing a fully
        numeric, integral `activity_encoded` column before sequence creation.
        """
        if "activity_encoded" not in processed_df.columns:
            raise ValueError(f"[{room_name}] Missing required activity_encoded column")

        numeric = pd.to_numeric(processed_df["activity_encoded"], errors="coerce")
        invalid_mask = numeric.isna() | ~np.isfinite(numeric)
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            raise ValueError(
                f"[{room_name}] activity_encoded has {invalid_count} invalid rows before sequence creation"
            )

        fractional_mask = ~np.isclose(numeric, np.floor(numeric))
        fractional_count = int(fractional_mask.sum())
        if fractional_count > 0:
            raise ValueError(
                f"[{room_name}] activity_encoded has {fractional_count} non-integer rows before sequence creation"
            )

        out = processed_df.copy()
        out["activity_encoded"] = numeric.astype(np.int32, copy=False)
        return out

    def _match_corrected_sequence_indices(
        self,
        elder_id: str,
        room_name: str,
        seq_timestamps: np.ndarray,
        tolerance_seconds: int = 10,
    ) -> np.ndarray:
        """
        Match corrected timestamps to sequence indices using nearest-neighbor tolerance.
        """
        corrected_df = fetch_golden_samples(elder_id, room_name)
        if corrected_df is None or corrected_df.empty or len(seq_timestamps) == 0:
            return np.array([], dtype=np.int64)

        seq_ts = pd.to_datetime(seq_timestamps, errors='coerce')
        corrected_ts = pd.to_datetime(corrected_df['timestamp'], errors='coerce')
        seq_ns = seq_ts.view('int64')
        corr_ns = corrected_ts.view('int64')
        valid_seq = ~pd.isna(seq_ts)
        if not np.any(valid_seq):
            return np.array([], dtype=np.int64)

        valid_positions = np.where(valid_seq)[0]
        valid_seq_ns = seq_ns[valid_seq]
        tolerance_ns = int(tolerance_seconds) * 1_000_000_000
        matched = set()

        for ts_ns in corr_ns:
            if ts_ns == np.iinfo(np.int64).min:
                continue
            pos = int(np.searchsorted(valid_seq_ns, ts_ns))
            candidates = []
            if pos < len(valid_seq_ns):
                candidates.append(pos)
            if pos > 0:
                candidates.append(pos - 1)
            best_pos = None
            best_delta = None
            for cand in candidates:
                delta = abs(int(valid_seq_ns[cand]) - int(ts_ns))
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_pos = cand
            if best_pos is not None and best_delta is not None and best_delta <= tolerance_ns:
                matched.add(int(valid_positions[best_pos]))

        if not matched:
            return np.array([], dtype=np.int64)
        return np.array(sorted(matched), dtype=np.int64)

    def _apply_replay_sampling(
        self,
        elder_id: str,
        room_name: str,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        seq_timestamps: np.ndarray,
        replay_ratio: int,
        sampling_strategy: str = "random_stratified",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Build replay mix for correction fine-tuning:
        keep corrected windows + sample uncorrected windows at configured ratio.
        """
        corrected_idx = self._match_corrected_sequence_indices(elder_id, room_name, seq_timestamps)
        total = int(len(X_seq))
        if len(corrected_idx) == 0:
            return X_seq, y_seq, seq_timestamps, {
                "total": total,
                "corrected_kept": 0,
                "uncorrected_sampled": total,
                "replay_ratio": int(replay_ratio),
                "sampling_strategy": str(sampling_strategy or "uniform_random"),
            }

        corrected_mask = np.zeros(total, dtype=bool)
        corrected_mask[corrected_idx] = True
        uncorrected_idx = np.where(~corrected_mask)[0]
        target_uncorrected = min(len(uncorrected_idx), len(corrected_idx) * int(replay_ratio))

        if target_uncorrected > 0:
            rng = np.random.default_rng(42)
            strategy = str(sampling_strategy or "").strip().lower()
            if strategy == "random_stratified":
                sampled_uncorrected = self._sample_uncorrected_stratified(
                    y_seq=y_seq,
                    uncorrected_idx=uncorrected_idx,
                    target_uncorrected=target_uncorrected,
                    rng=rng,
                )
            else:
                sampled_uncorrected = np.sort(rng.choice(uncorrected_idx, size=target_uncorrected, replace=False))
            keep_idx = np.sort(np.concatenate([corrected_idx, sampled_uncorrected]))
        else:
            keep_idx = np.sort(corrected_idx)

        replay_X = X_seq[keep_idx]
        replay_y = y_seq[keep_idx]
        replay_ts = seq_timestamps[keep_idx]
        replay_stats = {
            "total": total,
            "corrected_kept": int(len(corrected_idx)),
            "uncorrected_sampled": int(len(keep_idx) - len(corrected_idx)),
            "replay_ratio": int(replay_ratio),
            "sampling_strategy": str(sampling_strategy or "uniform_random"),
        }
        logger.info(
            f"Replay sampling for {room_name}: total={replay_stats['total']}, "
            f"corrected={replay_stats['corrected_kept']}, "
            f"uncorrected={replay_stats['uncorrected_sampled']}, "
            f"ratio={replay_stats['replay_ratio']}, "
            f"sampling={replay_stats['sampling_strategy']}"
        )
        return replay_X, replay_y, replay_ts, replay_stats

    @staticmethod
    def _sample_uncorrected_stratified(
        y_seq: np.ndarray,
        uncorrected_idx: np.ndarray,
        target_uncorrected: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample uncorrected indices with class-balancing to reduce majority-label dominance.
        """
        if target_uncorrected <= 0 or len(uncorrected_idx) == 0:
            return np.array([], dtype=np.int64)

        labels = y_seq[uncorrected_idx]
        classes = np.unique(labels)
        if len(classes) == 0:
            return np.array([], dtype=np.int64)

        per_class_budget = max(1, target_uncorrected // len(classes))
        selected_parts: List[np.ndarray] = []

        for cls in classes:
            cls_idx = uncorrected_idx[labels == cls]
            take = min(len(cls_idx), per_class_budget)
            if take <= 0:
                continue
            selected_parts.append(np.sort(rng.choice(cls_idx, size=take, replace=False)))

        if selected_parts:
            selected = np.unique(np.concatenate(selected_parts))
        else:
            selected = np.array([], dtype=np.int64)

        if len(selected) < target_uncorrected:
            remaining_pool = np.setdiff1d(uncorrected_idx, selected, assume_unique=False)
            fill = min(len(remaining_pool), target_uncorrected - len(selected))
            if fill > 0:
                fill_idx = np.sort(rng.choice(remaining_pool, size=fill, replace=False))
                selected = np.unique(np.concatenate([selected, fill_idx]))

        if len(selected) > target_uncorrected:
            selected = np.sort(rng.choice(selected, size=target_uncorrected, replace=False))

        return np.sort(selected.astype(np.int64))

    def _apply_correction_layer_freeze(
        self,
        model: Any,
        top_transformer_blocks: int = 2,
    ) -> Dict[str, Any]:
        """
        Freeze lower representation layers and unfreeze top Transformer blocks + head.
        """
        for layer in model.layers:
            layer.trainable = False

        transformer_layers: List[Tuple[int, Any]] = []
        for layer in model.layers:
            m = re.match(r"^transformer_block_(\d+)$", str(getattr(layer, "name", "")))
            if m:
                transformer_layers.append((int(m.group(1)), layer))

        transformer_layers = sorted(transformer_layers, key=lambda x: x[0])
        top_n = max(1, int(top_transformer_blocks))
        unfrozen_transformer_ids = {idx for idx, _ in transformer_layers[-top_n:]}

        for idx, layer in transformer_layers:
            layer.trainable = idx in unfrozen_transformer_ids

        if transformer_layers:
            max_block_idx = max(idx for idx, _ in transformer_layers)
            unfreeze_head = False
            for layer in model.layers:
                m = re.match(r"^transformer_block_(\d+)$", str(getattr(layer, "name", "")))
                if m and int(m.group(1)) == max_block_idx:
                    unfreeze_head = True
                    continue
                if unfreeze_head:
                    layer.trainable = True
        else:
            for layer in model.layers[-4:]:
                layer.trainable = True

        summary = {
            "top_transformer_blocks": top_n,
            "transformer_blocks_found": len(transformer_layers),
            "unfrozen_transformer_blocks": sorted(list(unfrozen_transformer_ids)),
            "trainable_layers": [layer.name for layer in model.layers if bool(layer.trainable)],
            "frozen_layers": [layer.name for layer in model.layers if not bool(layer.trainable)],
        }
        logger.info(f"Applied correction fine-tune layer freezing: {summary}")
        return summary

    def _build_model_for_room(
        self,
        room_name: str,
        seq_length: int,
        num_classes: int,
        elder_id: str,
        training_mode: str,
        warm_start: bool,
    ) -> Tuple[Any, bool]:
        """
        Build a fresh model or warm-start from current champion.
        """
        input_shape = (seq_length, len(self.platform.sensor_columns))
        did_warm_start = False

        if training_mode == "correction_fine_tune" and warm_start:
            champion_meta = self.registry.get_current_version_metadata(elder_id, room_name)
            champion_path = self.registry.get_models_dir(elder_id) / f"{room_name}_model.keras"
            if champion_meta and champion_path.exists():
                try:
                    model = self.registry.load_room_model(
                        str(champion_path),
                        room_name,
                        compile_model=False,
                    )
                    self._apply_correction_layer_freeze(model, top_transformer_blocks=2)
                    did_warm_start = True
                    return model, did_warm_start
                except Exception as e:
                    logger.warning(
                        f"Warm-start load failed for {elder_id}/{room_name}; "
                        f"falling back to fresh model: {e}"
                    )

        # Check if timeline multitask is enabled for this room
        room_key = normalize_room_name(room_name)
        use_timeline_heads = (
            self._is_timeline_multitask_enabled() 
            and room_key in self._timeline_native_rooms()
        )
        
        if use_timeline_heads:
            logger.info(f"Building timeline-native model for {room_name} with multi-task heads")
            model = self._build_timeline_model(
                input_shape=input_shape,
                num_classes=num_classes,
                room_name=room_name,
            )
        else:
            model = build_transformer_model(
                input_shape=input_shape,
                num_classes=num_classes,
                d_model=64,
                num_heads=4,
                ff_dim=128,
                num_transformer_blocks=2,
                dropout_rate=DEFAULT_DROPOUT_RATE,
                positional_encoding_type='sinusoidal',
                use_cnn_embedding=True
            )
            model._timeline_multitask_enabled = False
        return model, did_warm_start

    def _build_timeline_model(
        self,
        input_shape: Tuple[int, int],
        num_classes: int,
        room_name: str,
        d_model: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_transformer_blocks: int = 2,
        dropout_rate: float = 0.2,
    ) -> tf.keras.Model:
        """
        Build a timeline-native model with multi-task heads.
        
        This model uses the standard transformer backbone but replaces the
        output head with timeline-aware multi-task heads.
        """
        from ml.transformer_backbone import build_cnn_embedding, TransformerEncoderBlock
        from ml.positional_encoding import get_positional_encoding
        
        inputs = tf.keras.Input(shape=input_shape)
        
        # CNN Embedding (local feature extraction)
        cnn_embed = build_cnn_embedding(input_shape, d_model)
        x = cnn_embed(inputs)
        
        # Positional Encoding
        pos_enc = get_positional_encoding(
            encoding_type='sinusoidal',
            max_seq_length=input_shape[0],
            d_model=d_model
        )
        x = pos_enc(x)
        
        # Transformer Encoder Blocks
        for i in range(num_transformer_blocks):
            x = TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                name=f'transformer_block_{i}'
            )(x)
        
        # Collapse sequence representation to per-sequence embeddings.
        # Current training labels are sequence-level, not per-window.
        pooled = tf.keras.layers.GlobalAveragePooling1D(name="timeline_sequence_pool")(x)
        pooled = tf.keras.layers.Reshape((1, d_model), name="timeline_sequence_reshape")(pooled)

        # Timeline multi-task heads (instead of single classification head)
        timeline_heads = create_timeline_heads(
            num_activity_classes=num_classes,
            hidden_units=128,
            enable_boundary_heads=True,
            enable_daily_auxiliary=False,  # Keep simple for now
        )
        
        outputs = timeline_heads(pooled, training=None)
        if not isinstance(outputs, dict):
            raise ValueError("Timeline heads must return tensor dict in graph mode")

        # Squeeze sequence axis (len=1) to align with sequence-level targets.
        activity_logits = tf.keras.layers.Lambda(
            lambda t: tf.squeeze(t, axis=1), name="activity_logits"
        )(outputs["activity_logits"])
        occupancy_logits = tf.keras.layers.Lambda(
            lambda t: tf.squeeze(t, axis=1), name="occupancy_logits"
        )(outputs["occupancy_logits"])
        boundary_start_logits = tf.keras.layers.Lambda(
            lambda t: tf.squeeze(t, axis=1), name="boundary_start_logits"
        )(outputs["boundary_start_logits"])
        boundary_end_logits = tf.keras.layers.Lambda(
            lambda t: tf.squeeze(t, axis=1), name="boundary_end_logits"
        )(outputs["boundary_end_logits"])

        model = tf.keras.Model(
            inputs=inputs,
            outputs={
                "activity_logits": activity_logits,
                "occupancy_logits": occupancy_logits,
                "boundary_start_logits": boundary_start_logits,
                "boundary_end_logits": boundary_end_logits,
            },
            name=f'timeline_model_{room_name}'
        )
        
        # Store timeline metadata as attributes for training/evaluation path.
        model._timeline_heads = timeline_heads
        model._timeline_config = timeline_heads.config
        model._timeline_multitask_enabled = True
        model._timeline_loss_weights = {
            "activity_logits": float(timeline_heads.config.w_activity),
            "occupancy_logits": float(timeline_heads.config.w_occupancy),
            "boundary_start_logits": float(timeline_heads.config.w_boundary_start),
            "boundary_end_logits": float(timeline_heads.config.w_boundary_end),
        }
        
        logger.info(f"Built timeline-native model for {room_name}: "
                   f"input={input_shape}, classes={num_classes}, "
                   f"heads=activity+occupancy+boundary")
        
        return model

    @staticmethod
    def _is_shared_backbone_enabled() -> bool:
        return str(os.getenv("ENABLE_SHARED_BACKBONE_ADAPTERS", "false")).strip().lower() in {
            "1", "true", "yes", "on", "enabled"
        }

    @staticmethod
    def _get_active_shared_backbone_id(default: str = "") -> str:
        return str(os.getenv("ACTIVE_SHARED_BACKBONE_ID", default)).strip()

    @staticmethod
    def _is_backbone_layer_name(layer_name: str) -> bool:
        name = str(layer_name or "")
        prefixes = (
            "cnn_embedding",
            "transformer_block_",
            "sinusoidal_positional_encoding",
            "relative_positional_encoding",
            "learnable_positional_embedding",
            "alibi_bias_gen",
        )
        return any(name.startswith(prefix) for prefix in prefixes)

    def _apply_shared_adapter_freeze(self, model) -> Dict[str, int]:
        """Freeze backbone layers and keep adapter/head trainable."""
        frozen = 0
        trainable = 0
        for layer in model.layers:
            if self._is_backbone_layer_name(getattr(layer, "name", "")):
                layer.trainable = False
                frozen += 1
            else:
                layer.trainable = True
                trainable += 1
        summary = {"frozen_backbone_layers": frozen, "trainable_adapter_layers": trainable}
        logger.info(f"Applied shared-adapter freeze policy: {summary}")
        return summary

    def _resolve_model_identity(
        self,
        elder_id: str,
        room_name: str,
        training_mode: str,
        did_warm_start: bool,
        champion_meta: Optional[Dict[str, Any]],
        parent_version_id: Optional[int],
        active_backbone_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build model lineage identity for version tracking.

        This enables a smooth migration path from per-resident full retraining
        to shared-backbone + resident-adapter training.
        """
        room_key = normalize_room_name(room_name)
        shared_enabled = str(os.getenv("ENABLE_SHARED_BACKBONE_ADAPTERS", "false")).strip().lower() in {
            "1", "true", "yes", "on", "enabled"
        }

        if shared_enabled:
            champion_identity = champion_meta.get("model_identity", {}) if isinstance(champion_meta, dict) else {}
            champion_backbone_id = (
                champion_identity.get("backbone_id")
                if isinstance(champion_identity, dict)
                else None
            )
            backbone_id = str(active_backbone_id or "").strip()
            if not backbone_id and champion_backbone_id:
                backbone_id = str(champion_backbone_id).strip()

            if not backbone_id:
                return {
                    "family": "per_resident_full_model",
                    "backbone_id": None,
                    "adapter_id": None,
                    "adapter_family": None,
                    "training_mode": str(training_mode),
                    "warm_start_used": bool(did_warm_start),
                    "parent_version_id": int(parent_version_id) if parent_version_id is not None else None,
                }

            adapter_id = f"{elder_id}:{room_key}"
            return {
                "family": "shared_backbone_adapter",
                "backbone_id": backbone_id,
                "adapter_id": adapter_id,
                "adapter_family": "resident_room_adapter",
                "training_mode": str(training_mode),
                "warm_start_used": bool(did_warm_start),
                "parent_version_id": int(parent_version_id) if parent_version_id is not None else None,
                "backbone_changed_from_champion": bool(
                    champion_backbone_id is not None and str(champion_backbone_id) != backbone_id
                ),
            }

        return {
            "family": "per_resident_full_model",
            "backbone_id": None,
            "adapter_id": None,
            "adapter_family": None,
            "training_mode": str(training_mode),
            "warm_start_used": bool(did_warm_start),
            "parent_version_id": int(parent_version_id) if parent_version_id is not None else None,
        }

    @staticmethod
    def _read_float_env(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return float(default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(value):
            return float(default)
        return float(value)

    @staticmethod
    def _normalize_class_count_map(raw: Any) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not isinstance(raw, Mapping):
            return out
        for key, value in raw.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric) or numeric < 0.0:
                continue
            key_txt = str(key).strip()
            if not key_txt:
                continue
            try:
                key_txt = str(int(float(key_txt)))
            except (TypeError, ValueError):
                pass
            out[key_txt] = float(numeric)
        return out

    @staticmethod
    def _build_class_support_map(y_values: Any) -> Dict[str, int]:
        out: Dict[str, int] = {}
        try:
            arr = np.asarray(y_values, dtype=np.int64)
        except Exception:
            return out
        if arr.size == 0:
            return out
        classes, counts = np.unique(arr, return_counts=True)
        for class_id, count in zip(classes, counts):
            out[str(int(class_id))] = int(count)
        return out

    @staticmethod
    def _resolve_debug_room_list_env(name: str) -> set[str]:
        raw = os.getenv(name)
        if raw is None:
            return set()
        rooms: set[str] = set()
        for token in str(raw).split(","):
            item = str(token).strip()
            if not item:
                continue
            rooms.add(normalize_room_name(item))
        return rooms

    def _is_prediction_audit_enabled(self, room_name: str) -> bool:
        rooms = self._resolve_debug_room_list_env("DEBUG_POST_FIT_AUDIT_ROOMS")
        room_key = normalize_room_name(room_name)
        return "*" in rooms or room_key in rooms

    @staticmethod
    def _resolve_reported_accuracy(
        *,
        fit_history_accuracy: float,
        post_fit_train_summary: Optional[Mapping[str, Any]] = None,
    ) -> tuple[float, str]:
        if isinstance(post_fit_train_summary, Mapping) and bool(post_fit_train_summary.get("available")):
            try:
                return float(post_fit_train_summary.get("accuracy", 0.0)), "post_fit_train_predictions"
            except (TypeError, ValueError):
                pass
        return float(fit_history_accuracy), "fit_history"

    def _predict_activity_probabilities(
        self,
        *,
        model: Any,
        X: np.ndarray,
        timeline_multitask_enabled: bool,
    ) -> np.ndarray:
        y_pred = model.predict(X, verbose=0)
        if isinstance(y_pred, dict):
            y_pred = y_pred.get("activity_logits")
        if y_pred is None:
            raise ModelTrainError("Missing activity logits for prediction audit")
        y_pred_arr = np.asarray(y_pred, dtype=np.float32)
        if bool(timeline_multitask_enabled):
            y_pred_arr = tf.nn.softmax(y_pred_arr, axis=1).numpy()
        return np.asarray(y_pred_arr, dtype=np.float32)

    def _summarize_prediction_audit(
        self,
        *,
        room_name: str,
        y_true: Any,
        y_pred_probs: Any,
    ) -> Dict[str, Any]:
        y_true_arr = np.asarray(y_true, dtype=np.int32)
        y_pred_arr = np.asarray(y_pred_probs, dtype=np.float32)
        result: Dict[str, Any] = {
            "available": False,
            "sample_count": int(len(y_true_arr)),
        }
        if y_pred_arr.ndim != 2 or y_pred_arr.shape[0] != len(y_true_arr):
            result["reason"] = f"unexpected_prediction_shape:{getattr(y_pred_arr, 'shape', None)}"
            return result

        label_encoder = self.platform.label_encoders.get(room_name)
        classes = getattr(label_encoder, "classes_", None) if label_encoder is not None else None
        class_ids = sorted(
            set(np.unique(y_true_arr).astype(int)) | set(np.argmax(y_pred_arr, axis=1).astype(int))
        )
        if classes is not None:
            class_ids = [int(cid) for cid in class_ids if 0 <= int(cid) < len(classes)]

        def _label_name(class_id: int) -> str:
            if classes is not None and 0 <= int(class_id) < len(classes):
                return str(classes[int(class_id)]).strip().lower()
            return f"class_{int(class_id)}"

        y_pred_classes = np.argmax(y_pred_arr, axis=1)
        report = classification_report(y_true_arr, y_pred_classes, output_dict=True, zero_division=0)
        pred_dist: Dict[str, int] = {}
        true_dist: Dict[str, int] = {}
        probability_by_true_label: Dict[str, Dict[str, Any]] = {}
        for class_id in class_ids:
            label_name = _label_name(class_id)
            true_mask = y_true_arr == int(class_id)
            true_dist[label_name] = int(np.sum(true_mask))
            pred_dist[label_name] = int(np.sum(y_pred_classes == int(class_id)))
            if bool(np.any(true_mask)):
                probability_by_true_label[label_name] = {
                    "count": int(np.sum(true_mask)),
                    "mean_probability_by_label": {
                        _label_name(other_id): float(np.mean(y_pred_arr[true_mask, int(other_id)]))
                        for other_id in class_ids
                    },
                }
            else:
                probability_by_true_label[label_name] = {
                    "count": 0,
                    "mean_probability_by_label": {
                        _label_name(other_id): 0.0 for other_id in class_ids
                    },
                }

        result.update(
            {
                "available": True,
                "accuracy": float(np.mean(y_pred_classes == y_true_arr)) if len(y_true_arr) else 0.0,
                "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
                "macro_precision": float(report.get("macro avg", {}).get("precision", 0.0)),
                "macro_recall": float(report.get("macro avg", {}).get("recall", 0.0)),
                "true_class_distribution": true_dist,
                "predicted_class_distribution": pred_dist,
                "mean_probability_by_true_label": probability_by_true_label,
            }
        )
        return result

    def _resolve_no_regress_macro_f1_floor(
        self,
        *,
        room_name: str,
        champion_meta: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "enabled": False,
            "target_macro_f1_floor": None,
            "max_drop_from_champion": None,
            "champion_macro_f1": None,
            "exempt": False,
        }
        if not isinstance(champion_meta, Mapping):
            return context

        champion_metrics = champion_meta.get("metrics")
        if not isinstance(champion_metrics, Mapping):
            return context

        champion_f1_raw = champion_metrics.get("macro_f1")
        if champion_f1_raw is None:
            return context

        try:
            champion_f1 = float(champion_f1_raw)
        except (TypeError, ValueError):
            return context

        if not np.isfinite(champion_f1):
            return context

        try:
            policy = get_release_gates_config()
            no_regress_cfg = (policy.get("release_gates", {}) or {}).get("no_regress", {}) or {}
            exempt_rooms = {
                normalize_room_name(room)
                for room in (no_regress_cfg.get("exempt_rooms", []) or [])
            }
            room_key = normalize_room_name(room_name)
            if room_key in exempt_rooms:
                context.update(
                    {
                        "enabled": False,
                        "champion_macro_f1": champion_f1,
                        "exempt": True,
                    }
                )
                return context

            gate_policy = self._active_policy().release_gate
            max_drop = float(
                no_regress_cfg.get(
                    "max_drop_from_champion",
                    gate_policy.max_drop_from_champion_default,
                )
            )
            context.update(
                {
                    "enabled": True,
                    "target_macro_f1_floor": float(champion_f1 - max_drop),
                    "max_drop_from_champion": float(max_drop),
                    "champion_macro_f1": float(champion_f1),
                    "exempt": False,
                }
            )
        except Exception as e:
            context["error"] = f"{type(e).__name__}: {e}"
        return context

    def _summarize_gate_aligned_validation(
        self,
        *,
        room_name: str,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
    ) -> Dict[str, Any]:
        y_true_arr = np.asarray(y_true, dtype=np.int32)
        y_pred_arr = np.asarray(y_pred_probs, dtype=np.float32)
        if y_pred_arr.ndim != 2:
            raise ValueError(f"unexpected_prediction_shape:{getattr(y_pred_arr, 'shape', None)}")
        if int(y_pred_arr.shape[0]) != int(len(y_true_arr)):
            raise ValueError(
                f"prediction_length_mismatch:{int(y_pred_arr.shape[0])}!={int(len(y_true_arr))}"
            )

        y_pred_classes = np.argmax(y_pred_arr, axis=1).astype(np.int32)
        report = classification_report(y_true_arr, y_pred_classes, output_dict=True, zero_division=0)
        macro_f1 = float(report.get("macro avg", {}).get("f1-score", 0.0) or 0.0)

        label_encoder = self.platform.label_encoders.get(room_name)
        classes = getattr(label_encoder, "classes_", None) if label_encoder is not None else None
        per_label_precision: Dict[str, float] = {}
        per_label_recall: Dict[str, float] = {}
        per_label_support: Dict[str, int] = {}
        if classes is not None:
            for class_id in range(len(classes)):
                class_key = str(int(class_id))
                label_name = str(classes[int(class_id)]).strip().lower()
                class_metrics = report.get(class_key, {})
                per_label_precision[label_name] = float(class_metrics.get("precision", 0.0) or 0.0)
                per_label_recall[label_name] = float(class_metrics.get("recall", 0.0) or 0.0)
                per_label_support[label_name] = int(class_metrics.get("support", 0) or 0)

        dominant_class_id = -1
        dominant_class_count = 0
        dominant_class_share = 0.0
        predicted_distribution: Dict[str, int] = {}
        if len(y_pred_classes) > 0:
            pred_ids, pred_counts = np.unique(y_pred_classes, return_counts=True)
            idx = int(np.argmax(pred_counts))
            dominant_class_id = int(pred_ids[idx])
            dominant_class_count = int(pred_counts[idx])
            dominant_class_share = float(dominant_class_count) / float(len(y_pred_classes))
            for class_id, count in zip(pred_ids, pred_counts):
                label_name = self._get_label_name(room_name, int(class_id))
                key = str(label_name).strip().lower() if label_name else f"class_{int(class_id)}"
                predicted_distribution[key] = int(count)

        collapse_ratio = 0.95
        collapsed = bool(dominant_class_share >= collapse_ratio)
        dominant_label = self._get_label_name(room_name, dominant_class_id)
        return {
            "macro_f1": float(macro_f1),
            "per_label_recall": per_label_recall,
            "per_label_support": per_label_support,
            "per_label_precision": per_label_precision,
            "dominant_class_id": int(dominant_class_id),
            "dominant_class_label": (
                str(dominant_label).strip().lower()
                if dominant_label is not None else f"class_{int(dominant_class_id)}"
            ),
            "dominant_class_count": int(dominant_class_count),
            "dominant_class_share": float(dominant_class_share),
            "collapse_ratio": float(collapse_ratio),
            "collapsed": bool(collapsed),
            "predicted_class_distribution": predicted_distribution,
            "gate_aligned_score": float(macro_f1),
        }

    def _select_final_two_stage_gate_source(
        self,
        *,
        room_name: str,
        y_true: np.ndarray,
        single_stage_probs: np.ndarray,
        two_stage_probs: Optional[np.ndarray],
        gate_mode: str,
        champion_meta: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        gate_mode_norm = str(gate_mode or "shadow").strip().lower()
        y_true_arr = np.asarray(y_true, dtype=np.int32)
        single_probs = np.asarray(single_stage_probs, dtype=np.float32)

        no_regress = self._resolve_no_regress_macro_f1_floor(
            room_name=room_name,
            champion_meta=champion_meta,
        )

        def _build_payload(source_name: str, probs: np.ndarray) -> Dict[str, Any]:
            summary = self._summarize_gate_aligned_validation(
                room_name=room_name,
                y_true=y_true_arr,
                y_pred_probs=np.asarray(probs, dtype=np.float32),
            )
            macro_f1 = summary.get("macro_f1")
            macro_f1_value = float(macro_f1) if macro_f1 is not None else None
            no_regress_ok = True
            floor = no_regress.get("target_macro_f1_floor")
            if bool(no_regress.get("enabled")) and floor is not None:
                no_regress_ok = bool(
                    macro_f1_value is not None and macro_f1_value + 1e-9 >= float(floor)
                )
            return {
                "source": str(source_name),
                "summary": dict(summary),
                "collapsed": bool(summary.get("collapsed", False)),
                "macro_f1": macro_f1_value,
                "gate_aligned_score": float(summary.get("gate_aligned_score", float("-inf"))),
                "no_regress_ok": bool(no_regress_ok),
            }

        single_payload = _build_payload("single_stage", single_probs)
        two_payload: Optional[Dict[str, Any]] = None
        if two_stage_probs is not None:
            two_payload = _build_payload("two_stage", np.asarray(two_stage_probs, dtype=np.float32))

        selection: Dict[str, Any] = {
            "gate_mode": gate_mode_norm,
            "no_regress": dict(no_regress),
            "single_stage": single_payload,
            "two_stage": two_payload,
            "selected_source": "single_stage_only",
            "runtime_use_two_stage": False,
            "selected_summary": dict(single_payload.get("summary") or {}),
        }
        if two_payload is None:
            return selection
        if gate_mode_norm != "primary":
            selection["selected_source"] = "single_stage_shadow"
            return selection

        def _rank(payload: Mapping[str, Any]) -> Tuple[int, int, float, float, int]:
            return (
                0 if bool(payload.get("collapsed", False)) else 1,
                1 if bool(payload.get("no_regress_ok", True)) else 0,
                float(payload.get("gate_aligned_score", float("-inf"))),
                float(payload.get("macro_f1", float("-inf"))),
                1 if str(payload.get("source")) == "two_stage" else 0,
            )

        selected = max((single_payload, two_payload), key=_rank)
        if str(selected.get("source")) == "two_stage":
            selection["selected_source"] = "two_stage_primary"
            selection["runtime_use_two_stage"] = True
            selection["selected_summary"] = dict(two_payload.get("summary") or {})
            return selection

        if bool(two_payload.get("collapsed", False)) and not bool(single_payload.get("collapsed", False)):
            source = "single_stage_fallback_collapse"
        elif (
            bool(no_regress.get("enabled"))
            and bool(single_payload.get("no_regress_ok"))
            and not bool(two_payload.get("no_regress_ok"))
        ):
            source = "single_stage_fallback_no_regress"
        elif float(single_payload.get("gate_aligned_score", float("-inf"))) > (
            float(two_payload.get("gate_aligned_score", float("-inf"))) + 1e-8
        ):
            source = "single_stage_fallback_score"
        else:
            source = "single_stage_fallback_macro_f1"

        selection["selected_source"] = source
        selection["selected_summary"] = dict(single_payload.get("summary") or {})
        return selection

    @classmethod
    def _compute_class_share_drift_from_labels(
        cls,
        reference_labels: Any,
        comparison_labels: Any,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "available": False,
            "reference_total": 0.0,
            "comparison_total": 0.0,
            "max_abs_drift": None,
            "max_abs_drift_pp": None,
            "max_drift_class": None,
            "class_drift": {},
        }
        reference_counts = cls._normalize_class_count_map(cls._build_class_support_map(reference_labels))
        comparison_counts = cls._normalize_class_count_map(cls._build_class_support_map(comparison_labels))
        if not reference_counts or not comparison_counts:
            return result

        reference_total = float(sum(reference_counts.values()))
        comparison_total = float(sum(comparison_counts.values()))
        if reference_total <= 0.0 or comparison_total <= 0.0:
            return result

        def _sort_key(token: str) -> Tuple[int, str]:
            try:
                return int(token), token
            except (TypeError, ValueError):
                return 10**9, token

        class_ids = sorted(set(reference_counts.keys()) | set(comparison_counts.keys()), key=_sort_key)
        max_abs = -1.0
        max_class: Optional[str] = None
        class_drift: Dict[str, Dict[str, float]] = {}
        for class_id in class_ids:
            reference_share = float(reference_counts.get(class_id, 0.0)) / reference_total
            comparison_share = float(comparison_counts.get(class_id, 0.0)) / comparison_total
            drift = float(comparison_share - reference_share)
            class_drift[class_id] = {
                "reference_share": float(reference_share),
                "comparison_share": float(comparison_share),
                "drift": float(drift),
                "drift_pp": float(drift * 100.0),
            }
            abs_drift = abs(drift)
            if abs_drift > max_abs:
                max_abs = float(abs_drift)
                max_class = class_id

        result.update(
            {
                "available": True,
                "reference_total": float(reference_total),
                "comparison_total": float(comparison_total),
                "max_abs_drift": float(max_abs if max_abs >= 0.0 else 0.0),
                "max_abs_drift_pp": float((max_abs if max_abs >= 0.0 else 0.0) * 100.0),
                "max_drift_class": max_class,
                "class_drift": class_drift,
            }
        )
        return result

    @staticmethod
    def _resolve_prior_drift_reference_labels(
        reference_labels: Any,
        fallback_labels: Any,
    ) -> Tuple[np.ndarray, str]:
        try:
            fallback = np.asarray(fallback_labels, dtype=np.int64)
        except Exception:
            fallback = np.asarray([], dtype=np.int64)
        try:
            if reference_labels is None:
                raise ValueError("missing_reference")
            reference = np.asarray(reference_labels, dtype=np.int64)
            if reference.size == 0:
                raise ValueError("empty_reference")
            return reference, "holdout"
        except Exception:
            return fallback, "train"

    @staticmethod
    def _select_multi_seed_candidate(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not candidates:
            raise ValueError("multi_seed_candidates_empty")

        def _summary(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
            summary = candidate.get("summary") or {}
            return summary if isinstance(summary, Mapping) else {}

        def _score(candidate: Mapping[str, Any]) -> Tuple[int, int, int, float, float, int]:
            summary = _summary(candidate)
            collapsed = bool(summary.get("collapsed", False))
            gate_pass = bool(summary.get("gate_pass", False))
            no_regress_ok = bool(candidate.get("no_regress_ok", False))
            try:
                gate_aligned_score = float(summary.get("gate_aligned_score", float("-inf")))
            except (TypeError, ValueError):
                gate_aligned_score = float("-inf")
            try:
                macro_f1 = float(summary.get("macro_f1", float("-inf")))
            except (TypeError, ValueError):
                macro_f1 = float("-inf")
            try:
                seed_value = int(candidate.get("seed", 0))
            except (TypeError, ValueError):
                seed_value = 0
            return (
                1 if not collapsed else 0,
                1 if gate_pass else 0,
                1 if no_regress_ok else 0,
                gate_aligned_score,
                macro_f1,
                -seed_value,
            )

        best = max(candidates, key=_score)
        return dict(best)

    @staticmethod
    @contextmanager
    def _temporary_env_overrides(overrides: Mapping[str, Optional[str]]):
        previous: Dict[str, Optional[str]] = {}
        try:
            for key, value in overrides.items():
                previous[str(key)] = os.environ.get(str(key))
                if value is None:
                    os.environ.pop(str(key), None)
                else:
                    os.environ[str(key)] = str(value)
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(str(key), None)
                else:
                    os.environ[str(key)] = value

    def _spawn_candidate_pipeline(self) -> "TrainingPipeline":
        nested_policy = None
        if not self._use_env_policy:
            nested_policy = copy.deepcopy(self._active_policy())
            raw_seed_override = os.getenv("TRAINING_RANDOM_SEED")
            if raw_seed_override is not None:
                try:
                    nested_policy.reproducibility.random_seed = int(raw_seed_override)
                except (TypeError, ValueError):
                    pass
        return TrainingPipeline(self.platform, self.registry, policy=nested_policy)

    def _multi_seed_enabled_for_room(self, room_name: str) -> bool:
        if str(os.getenv("TRAINING_MULTI_SEED_ACTIVE", "")).strip() == "1":
            return False
        room_key = normalize_room_name(room_name)
        enabled_rooms = {
            str(item).strip().lower()
            for item in self._active_policy().reproducibility.multi_seed_rooms
            if str(item).strip()
        }
        return room_key in enabled_rooms

    @staticmethod
    def _build_seed_panel_candidate_summary(metrics: Mapping[str, Any]) -> Tuple[Dict[str, Any], bool]:
        reasons = [str(item) for item in (metrics.get("gate_reasons") or []) if str(item).strip()]
        predicted_distribution = metrics.get("predicted_class_distribution") or {}
        total_predictions = 0.0
        dominant_label = None
        dominant_share = 0.0
        if isinstance(predicted_distribution, Mapping):
            numeric_counts: Dict[str, float] = {}
            for key, value in predicted_distribution.items():
                try:
                    numeric_counts[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            total_predictions = float(sum(numeric_counts.values()))
            if total_predictions > 0.0 and numeric_counts:
                dominant_label, dominant_count = max(
                    numeric_counts.items(),
                    key=lambda item: (float(item[1]), str(item[0])),
                )
                dominant_share = float(dominant_count) / total_predictions

        collapsed = bool(
            dominant_share >= 0.95
            or any("predicted_class_collapse" in reason for reason in reasons)
            or any("critical_label_collapse" in reason for reason in reasons)
        )
        no_regress_ok = not any("no_regress_failed" in reason for reason in reasons)
        macro_f1 = float(metrics.get("macro_f1") or 0.0)
        gate_aligned_score = float(macro_f1)
        if not bool(metrics.get("gate_pass", False)):
            gate_aligned_score -= max(0.5, 0.25 * float(len(reasons) or 1))
        if not no_regress_ok:
            gate_aligned_score -= 1.0
        if collapsed:
            gate_aligned_score -= 5.0

        summary = {
            "collapsed": bool(collapsed),
            "gate_aligned_score": float(gate_aligned_score),
            "macro_f1": float(macro_f1),
            "gate_pass": bool(metrics.get("gate_pass", False)),
            "dominant_label": dominant_label,
            "dominant_share": float(dominant_share),
            "gate_reasons": reasons,
            "saved_version": metrics.get("saved_version"),
        }
        return summary, bool(no_regress_ok)

    def _train_room_with_multi_seed_panel(
        self,
        room_name: str,
        processed_df,
        seq_length: int,
        elder_id: str,
        progress_callback=None,
        training_mode: str = "full_retrain",
        defer_promotion: bool = False,
        gate_evaluation_result: Optional[Dict[str, Any]] = None,
        event_first_shadow: Optional[bool] = None,
    ) -> Dict[str, Any]:
        repro = self._active_policy().reproducibility
        seeds = [int(seed) for seed in repro.multi_seed_candidate_seeds] or [int(repro.random_seed)]
        candidates: List[Dict[str, Any]] = []

        for seed in seeds:
            with self._temporary_env_overrides(
                {
                    "TRAINING_RANDOM_SEED": str(int(seed)),
                    "TRAINING_MULTI_SEED_ACTIVE": "1",
                }
            ):
                candidate_pipeline = self._spawn_candidate_pipeline()
                candidate_metrics = candidate_pipeline.train_room(
                    room_name=room_name,
                    processed_df=processed_df,
                    seq_length=seq_length,
                    elder_id=elder_id,
                    progress_callback=None,
                    training_mode=training_mode,
                    defer_promotion=True,
                    gate_evaluation_result=gate_evaluation_result,
                    event_first_shadow=event_first_shadow,
                )
            summary, no_regress_ok = self._build_seed_panel_candidate_summary(candidate_metrics or {})
            candidates.append(
                {
                    "seed": int(seed),
                    "summary": summary,
                    "no_regress_ok": bool(no_regress_ok),
                    "metrics": copy.deepcopy(candidate_metrics or {}),
                    "artifacts": {
                        "model": self.platform.room_models.get(room_name),
                        "scaler": self.platform.scalers.get(room_name),
                        "label_encoder": self.platform.label_encoders.get(room_name),
                        "class_thresholds": copy.deepcopy(
                            (self.platform.class_thresholds.get(room_name) or {})
                            if hasattr(self.platform, "class_thresholds")
                            else {}
                        ),
                    },
                }
            )

        selected = self._select_multi_seed_candidate(candidates)
        seed_panel_debug = {
            "enabled": True,
            "selection_mode": "deterministic_multi_seed_panel",
            "candidate_seeds": [int(seed) for seed in seeds],
            "selected_seed": int(selected.get("seed", seeds[0])),
            "candidates": [
                {
                    "seed": int(candidate.get("seed", 0)),
                    "summary": dict(candidate.get("summary") or {}),
                    "no_regress_ok": bool(candidate.get("no_regress_ok", False)),
                }
                for candidate in candidates
            ],
        }
        final_metrics = copy.deepcopy(selected.get("metrics") or {})
        selected_artifacts = selected.get("artifacts") or {}
        if selected_artifacts.get("model") is not None:
            self.platform.room_models[room_name] = selected_artifacts.get("model")
        if selected_artifacts.get("scaler") is not None:
            self.platform.scalers[room_name] = selected_artifacts.get("scaler")
        if selected_artifacts.get("label_encoder") is not None:
            self.platform.label_encoders[room_name] = selected_artifacts.get("label_encoder")
        if hasattr(self.platform, "class_thresholds"):
            self.platform.class_thresholds[room_name] = copy.deepcopy(
                selected_artifacts.get("class_thresholds") or {}
            )

        selected_saved_version = final_metrics.get("saved_version")
        selected_gate_pass = bool(final_metrics.get("gate_pass", False))
        if (
            selected_saved_version is not None
            and selected_gate_pass
            and not bool(defer_promotion)
        ):
            try:
                promoted = bool(
                    self.registry.rollback_to_version(
                        elder_id=elder_id,
                        room_name=room_name,
                        version=int(selected_saved_version),
                    )
                )
            except Exception as e:
                promoted = False
                final_metrics["seed_panel_promotion_error"] = f"{type(e).__name__}: {e}"
            final_metrics["promoted_to_latest"] = bool(promoted)
            final_metrics["promotion_deferred"] = False if promoted else bool(
                final_metrics.get("promotion_deferred", False)
            )
        final_metrics["seed_panel_debug"] = dict(seed_panel_debug)
        return final_metrics

    def _compute_class_prior_drift(self, candidate_metrics: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Compare train vs validation class priors for drift diagnostics.

        Primary hard-gate source:
        - pre-sampling train support (temporal train split before train-only rebalancing)

        Secondary diagnostic source:
        - post-sampling train support (after downsampling/minority sampling)
        """
        result: Dict[str, Any] = {
            "available": False,
            "evaluable": False,
            "reason": None,
            "max_abs_drift": None,
            "max_abs_drift_pp": None,
            "max_drift_class": None,
            "class_drift": {},
            "train_total": 0.0,
            "validation_total": 0.0,
            "reference_total": 0.0,
            "required_validation_support": 0,
            "validation_min_support": 0,
            "reference_min_support": 0,
            "reference_source": "validation",
            "drift_source": "pre_sampling_train_vs_validation",
            "sampled_available": False,
            "sampled_max_abs_drift": None,
            "sampled_max_abs_drift_pp": None,
            "sampled_max_drift_class": None,
            "sampled_class_drift": {},
            "sampled_train_total": 0.0,
        }

        train_counts_raw: Any = candidate_metrics.get("train_class_support_pre_sampling") or {}
        minority_stats = candidate_metrics.get("minority_sampling") or {}
        sampled_counts_raw: Any = candidate_metrics.get("train_class_support_post_minority_sampling") or {}
        if isinstance(minority_stats, Mapping):
            sampled_counts_raw = sampled_counts_raw or minority_stats.get("class_counts_after") or {}
        holdout_counts_raw: Any = candidate_metrics.get("holdout_class_support") or {}
        val_counts_raw: Any = candidate_metrics.get("validation_class_support") or {}

        train_counts = self._normalize_class_count_map(train_counts_raw)
        sampled_counts = self._normalize_class_count_map(sampled_counts_raw)
        holdout_counts = self._normalize_class_count_map(holdout_counts_raw)
        val_counts = self._normalize_class_count_map(val_counts_raw)
        reference_counts = dict(holdout_counts or val_counts)
        if holdout_counts:
            result["drift_source"] = "pre_sampling_train_vs_holdout"
            result["reference_source"] = "holdout"
        if not train_counts and sampled_counts:
            # Backward-compatible fallback for historical metrics without pre-sampling support map.
            train_counts = dict(sampled_counts)
            if holdout_counts:
                result["drift_source"] = "post_sampling_train_vs_holdout_fallback"
            else:
                result["drift_source"] = "post_sampling_train_vs_validation_fallback"
        if not train_counts:
            result["reason"] = "missing_train_class_counts"
            return result
        if not reference_counts:
            result["reason"] = "missing_reference_class_counts"
            return result

        train_total = float(sum(train_counts.values()))
        reference_total = float(sum(reference_counts.values()))
        if train_total <= 0.0 or reference_total <= 0.0:
            result["reason"] = "non_positive_class_count_total"
            return result

        def _sort_key(token: str) -> Tuple[int, str]:
            try:
                return int(token), token
            except (TypeError, ValueError):
                return 10**9, token

        class_ids = sorted(set(train_counts.keys()) | set(reference_counts.keys()), key=_sort_key)
        class_drift: Dict[str, Dict[str, float]] = {}
        max_abs = -1.0
        max_class: Optional[str] = None
        for class_id in class_ids:
            train_share = float(train_counts.get(class_id, 0.0)) / train_total
            reference_share = float(reference_counts.get(class_id, 0.0)) / reference_total
            drift = float(train_share - reference_share)
            class_drift[class_id] = {
                "train_share": float(train_share),
                "validation_share": float(reference_share),
                "reference_share": float(reference_share),
                "drift": float(drift),
                "drift_pp": float(drift * 100.0),
            }
            abs_drift = abs(drift)
            if abs_drift > max_abs:
                max_abs = float(abs_drift)
                max_class = class_id

        required_support = int(max(1, candidate_metrics.get("required_minority_support", 1) or 1))
        val_min_support = int(candidate_metrics.get("validation_min_class_support", 0) or 0)
        holdout_min_support = int(candidate_metrics.get("holdout_min_class_support", 0) or 0)
        reference_min_support = int(holdout_min_support or val_min_support)
        insufficient_validation_evidence = bool(candidate_metrics.get("insufficient_validation_evidence", False))
        evaluable = bool(reference_min_support >= required_support and not insufficient_validation_evidence)

        result.update(
            {
                "available": True,
                "evaluable": bool(evaluable),
                "reason": None,
                "max_abs_drift": float(max_abs if max_abs >= 0.0 else 0.0),
                "max_abs_drift_pp": float((max_abs if max_abs >= 0.0 else 0.0) * 100.0),
                "max_drift_class": max_class,
                "class_drift": class_drift,
                "train_total": float(train_total),
                "validation_total": float(reference_total),
                "reference_total": float(reference_total),
                "required_validation_support": int(required_support),
                "validation_min_support": int(val_min_support),
                "reference_min_support": int(reference_min_support),
            }
        )
        if sampled_counts:
            sampled_total = float(sum(sampled_counts.values()))
            if sampled_total > 0.0:
                sampled_class_ids = sorted(set(sampled_counts.keys()) | set(reference_counts.keys()), key=_sort_key)
                sampled_class_drift: Dict[str, Dict[str, float]] = {}
                sampled_max_abs = -1.0
                sampled_max_class: Optional[str] = None
                for class_id in sampled_class_ids:
                    sampled_train_share = float(sampled_counts.get(class_id, 0.0)) / sampled_total
                    reference_share = float(reference_counts.get(class_id, 0.0)) / reference_total
                    drift = float(sampled_train_share - reference_share)
                    sampled_class_drift[class_id] = {
                        "train_share": float(sampled_train_share),
                        "validation_share": float(reference_share),
                        "reference_share": float(reference_share),
                        "drift": float(drift),
                        "drift_pp": float(drift * 100.0),
                    }
                    abs_drift = abs(drift)
                    if abs_drift > sampled_max_abs:
                        sampled_max_abs = float(abs_drift)
                        sampled_max_class = class_id
                result.update(
                    {
                        "sampled_available": True,
                        "sampled_max_abs_drift": float(sampled_max_abs if sampled_max_abs >= 0.0 else 0.0),
                        "sampled_max_abs_drift_pp": float(
                            (sampled_max_abs if sampled_max_abs >= 0.0 else 0.0) * 100.0
                        ),
                        "sampled_max_drift_class": sampled_max_class,
                        "sampled_class_drift": sampled_class_drift,
                        "sampled_train_total": float(sampled_total),
                    }
                )
        return result

    def _is_bootstrap_release_phase(self, candidate_metrics: Mapping[str, Any]) -> bool:
        """
        Return True when pilot evidence profile is active within the bootstrap window.
        """
        bootstrap_enabled = str(os.getenv("RELEASE_GATE_BOOTSTRAP_ENABLED", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "enabled",
        }
        if not bootstrap_enabled:
            return False
        gate_policy = self._active_policy().release_gate
        evidence_profile = str(gate_policy.evidence_profile).strip().lower()
        if evidence_profile not in {"pilot_stage_a", "pilot_stage_b"}:
            return False
        max_days = max(0.0, self._read_float_env("RELEASE_GATE_BOOTSTRAP_MAX_TRAINING_DAYS", 14.0))
        try:
            training_days = float(candidate_metrics.get("training_days", 0.0) or 0.0)
        except Exception:
            training_days = 0.0
        return bool(training_days <= (max_days + 1e-3))

    def _resolve_bootstrap_gate_overrides(
        self,
        candidate_metrics: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve staged pilot gate overrides for days 1-7 and 8-14.
        """
        if not self._is_bootstrap_release_phase(candidate_metrics):
            return {"active": False}
        try:
            training_days = float(candidate_metrics.get("training_days", 0.0) or 0.0)
        except Exception:
            training_days = 0.0
        phase1_max_days = max(1.0, self._read_float_env("RELEASE_GATE_BOOTSTRAP_PHASE1_MAX_DAYS", 7.0))
        phase2_max_days = max(phase1_max_days, self._read_float_env("RELEASE_GATE_BOOTSTRAP_MAX_TRAINING_DAYS", 14.0))

        if training_days <= (phase1_max_days + 1e-3):
            return {
                "active": True,
                "phase": "bootstrap",
                "min_validation_class_support": 5,
                "min_recall_support": 10,
                "label_min_recall_by_room_label": {
                    "bedroom.unoccupied": 0.35,
                    "livingroom.unoccupied": 0.30,
                },
            }
        if training_days <= (phase2_max_days + 1e-3):
            return {
                "active": True,
                "phase": "stabilization",
                "min_validation_class_support": 15,
                "min_recall_support": 15,
                "label_min_recall_by_room_label": {
                    "bedroom.unoccupied": 0.45,
                    "livingroom.unoccupied": 0.40,
                },
            }
        return {"active": False}

    def _evaluate_release_gate(
        self,
        room_name: str,
        candidate_metrics: Dict[str, Any],
        champion_meta: Optional[Dict[str, Any]],
    ) -> Tuple[bool, List[str]]:
        """
        Evaluate room-level release gate checks for a candidate model.
        """
        blocking_reasons: List[str] = []
        watch_reasons: List[str] = []

        def _add_blocking(reason: str) -> None:
            txt = str(reason).strip()
            if txt:
                blocking_reasons.append(txt)

        def _add_watch(reason: str) -> None:
            txt = str(reason).strip()
            if txt:
                watch_reasons.append(txt)

        def _finalize() -> Tuple[bool, List[str]]:
            self._last_release_gate_watch_reasons = list(dict.fromkeys(str(r) for r in watch_reasons if str(r).strip()))
            dedup_blocking = list(dict.fromkeys(str(r) for r in blocking_reasons if str(r).strip()))
            return len(dedup_blocking) == 0, dedup_blocking

        room_key = normalize_room_name(room_name)
        candidate_f1 = candidate_metrics.get("macro_f1")
        training_days = float(candidate_metrics.get("training_days", 0.0))
        samples = int(candidate_metrics.get("samples", 0) or 0)
        observed_day_count = int(candidate_metrics.get("observed_day_count", 0) or 0)
        retained_ratio = candidate_metrics.get("retained_sample_ratio")
        calibration_min_support = int(candidate_metrics.get("calibration_min_support", 0) or 0)
        validation_min_support = int(candidate_metrics.get("validation_min_class_support", 0) or 0)
        calibration_low_support_count = int(candidate_metrics.get("calibration_low_support_count", 0) or 0)
        metric_source = str(candidate_metrics.get("metric_source") or "")
        two_stage_core = candidate_metrics.get("two_stage_core") or {}
        data_viability = candidate_metrics.get("data_viability") or {}
        raw_per_label_recall = candidate_metrics.get("per_label_recall") or {}
        raw_per_label_support = candidate_metrics.get("per_label_support") or {}
        per_label_recall = {
            str(k).strip().lower(): float(v)
            for k, v in raw_per_label_recall.items()
            if v is not None
        } if isinstance(raw_per_label_recall, dict) else {}
        per_label_support = {
            str(k).strip().lower(): int(v)
            for k, v in raw_per_label_support.items()
            if v is not None
        } if isinstance(raw_per_label_support, dict) else {}
        label_equivalents = self._resolve_room_label_equivalents(room_name)
        gate_policy = self._active_policy().release_gate
        bootstrap_overrides = self._resolve_bootstrap_gate_overrides(candidate_metrics)
        bootstrap_release_phase = bool(bootstrap_overrides.get("active", False))
        effective_min_validation_support = int(
            bootstrap_overrides.get("min_validation_class_support", gate_policy.min_validation_class_support)
        )
        effective_min_recall_support = int(
            bootstrap_overrides.get("min_recall_support", gate_policy.min_recall_support)
        )
        effective_label_recall_by_room_label = dict(gate_policy.min_recall_by_room_label or {})
        effective_label_recall_by_room_label.update(
            {
                str(k).strip().lower(): float(v)
                for k, v in (bootstrap_overrides.get("label_min_recall_by_room_label") or {}).items()
                if str(k).strip()
            }
        )
        required_minority_support = int(candidate_metrics.get("required_minority_support", 0) or 0)
        required_validation_support = max(
            int(effective_min_validation_support),
            max(0, int(required_minority_support)),
        )
        pilot_relaxed_evidence = str(gate_policy.evidence_profile).strip().lower() in {
            "pilot_stage_a",
            "pilot_stage_b",
        }
        training_days_epsilon = 1e-3
        insufficient_validation_evidence = bool(candidate_metrics.get("insufficient_validation_evidence", False))
        validation_evaluable = (
            validation_min_support > 0
            and validation_min_support >= int(required_validation_support)
            and not insufficient_validation_evidence
        )
        prior_drift_max = max(0.0, self._read_float_env("RELEASE_GATE_MAX_CLASS_PRIOR_DRIFT", 0.10))
        class_prior_drift = self._compute_class_prior_drift(candidate_metrics)
        candidate_metrics["class_prior_drift"] = class_prior_drift

        if isinstance(data_viability, dict) and not bool(data_viability.get("pass", True)):
            _add_blocking(f"data_viability_failed:{room_key}")
        if training_days < (float(gate_policy.min_training_days) - training_days_epsilon):
            _add_blocking(
                f"insufficient_training_days:{room_key}:{training_days:.4f}<{float(gate_policy.min_training_days):.4f}"
            )
        if observed_day_count > 0 and observed_day_count < int(gate_policy.min_observed_days):
            _add_blocking(
                f"insufficient_observed_days:{room_key}:{observed_day_count}<{int(gate_policy.min_observed_days)}"
            )
        if samples < int(gate_policy.min_samples):
            _add_blocking(f"insufficient_samples:{room_key}:{samples}<{int(gate_policy.min_samples)}")
        if retained_ratio is not None and float(retained_ratio) < float(gate_policy.min_retained_sample_ratio):
            _add_blocking(
                f"insufficient_retained_ratio:{room_key}:{float(retained_ratio):.4f}<"
                f"{float(gate_policy.min_retained_sample_ratio):.4f}"
            )
        if retained_ratio is not None and (1.0 - float(retained_ratio)) > float(gate_policy.max_dropped_ratio):
            _add_blocking(
                f"excessive_gap_drop_ratio:{room_key}:{(1.0 - float(retained_ratio)):.4f}>"
                f"{float(gate_policy.max_dropped_ratio):.4f}"
            )
        if validation_min_support > 0 and validation_min_support < int(required_validation_support):
            _add_watch(
                f"insufficient_validation_support:{room_key}:{validation_min_support}<"
                f"{int(required_validation_support)}"
            )
        if insufficient_validation_evidence and validation_min_support <= 0:
            _add_watch(
                f"insufficient_validation_support:{room_key}:{validation_min_support}<"
                f"{int(required_validation_support)}"
            )
        if calibration_min_support > 0 and calibration_min_support < int(gate_policy.min_calibration_support):
            _add_watch(
                f"insufficient_calibration_support:{room_key}:{calibration_min_support}<"
                f"{int(gate_policy.min_calibration_support)}"
            )
        if bool(gate_policy.block_on_low_support_fallback) and calibration_low_support_count > 0:
            _add_watch(
                f"calibration_low_support_fallback:{room_key}:classes={int(calibration_low_support_count)}"
            )
        if bool(gate_policy.block_on_train_fallback_metrics) and metric_source == "train_fallback_small_dataset":
            _add_watch(f"train_metric_fallback_blocked:{room_key}")
        if bool(class_prior_drift.get("available")):
            if str(class_prior_drift.get("drift_source", "")).strip().lower() == (
                "post_sampling_train_vs_validation_fallback"
            ):
                _add_watch(f"class_prior_drift_source_fallback:{room_key}:post_sampling")
            if bool(class_prior_drift.get("evaluable")):
                max_abs_drift = class_prior_drift.get("max_abs_drift")
                if max_abs_drift is not None and float(max_abs_drift) > float(prior_drift_max):
                    max_class = class_prior_drift.get("max_drift_class") or "unknown"
                    _add_blocking(
                        f"class_prior_drift_failed:{room_key}:{max_class}:{float(max_abs_drift):.3f}>"
                        f"{float(prior_drift_max):.3f}"
                    )
            else:
                _add_watch(
                    f"class_prior_drift_not_evaluable:{room_key}:support={validation_min_support}<"
                    f"required={int(required_validation_support)}"
                )
            sampled_max_abs_drift = class_prior_drift.get("sampled_max_abs_drift")
            if (
                sampled_max_abs_drift is not None
                and float(sampled_max_abs_drift) > float(prior_drift_max)
            ):
                sampled_max_class = class_prior_drift.get("sampled_max_drift_class") or "unknown"
                _add_watch(
                    f"class_prior_drift_sampled_watch:{room_key}:{sampled_max_class}:"
                    f"{float(sampled_max_abs_drift):.3f}>{float(prior_drift_max):.3f}"
                )
        for key, threshold in effective_label_recall_by_room_label.items():
            key_txt = str(key).strip().lower()
            if not key_txt.startswith(f"{room_key}."):
                continue
            label_name = key_txt.split(".", 1)[1]
            _, label_support, label_recall = self._select_best_label_variant(
                label_name,
                per_label_support=per_label_support,
                per_label_recall=per_label_recall,
                equivalents=label_equivalents,
            )
            if label_recall is None:
                _add_watch(f"label_recall_missing:{room_key}:{label_name}")
                continue
            if label_support < int(effective_min_recall_support):
                _add_watch(
                    f"label_recall_insufficient_support:{room_key}:{label_name}:{label_support}"
                    f"<{int(effective_min_recall_support)}"
                )
                continue
            if float(label_recall) < float(threshold):
                _add_blocking(
                    f"label_recall_failed:{room_key}:{label_name}:{float(label_recall):.3f}"
                    f"<{float(threshold):.3f}"
                )
        if metric_source == "holdout_validation":
            critical_labels = self._resolve_critical_labels(room_name)
            collapse_min_support = max(5, int(effective_min_recall_support // 2))
            collapse_recall_floor = 0.02
            for critical_label in critical_labels:
                _, support, recall = self._select_best_label_variant(
                    critical_label,
                    per_label_support=per_label_support,
                    per_label_recall=per_label_recall,
                    equivalents=label_equivalents,
                )
                if support <= 0:
                    _add_watch(f"critical_label_missing_validation:{room_key}:{critical_label}")
                    continue
                if recall is None:
                    _add_watch(f"critical_label_recall_missing:{room_key}:{critical_label}")
                    continue
                if support >= collapse_min_support and float(recall) <= collapse_recall_floor:
                    reason = (
                        f"critical_label_collapse:{room_key}:{critical_label}:{float(recall):.3f}"
                        f"<= {collapse_recall_floor:.3f}"
                    )
                    if pilot_relaxed_evidence and not validation_evaluable:
                        _add_watch(reason)
                    else:
                        _add_blocking(reason)
        promo_policy = self._active_policy().promotion_eligibility
        if (
            champion_meta is not None
            and training_days < (float(promo_policy.min_training_days_with_champion) - training_days_epsilon)
        ):
            _add_blocking(
                f"promotion_ineligible_training_days:{room_key}:{training_days:.4f}<"
                f"{float(promo_policy.min_training_days_with_champion):.4f}"
            )

        try:
            policy = get_release_gates_config()
        except Exception as e:
            allow_fallback_pass = bool(gate_policy.allow_gate_config_fallback_pass)
            logger.warning(
                f"Release gates config unavailable for {room_name}; "
                f"defaulting to {'PASS' if allow_fallback_pass else 'FAIL'}: {e}"
            )
            if blocking_reasons:
                _add_blocking("gate_config_unavailable")
            elif allow_fallback_pass:
                _add_watch("gate_config_unavailable")
            else:
                _add_blocking("gate_config_unavailable")
            return _finalize()

        room_policy = policy.get("release_gates", {}).get("rooms", {}).get(room_key)
        if not room_policy:
            if blocking_reasons:
                _add_blocking(f"no_room_policy:{room_key}")
            else:
                _add_watch(f"no_room_policy:{room_key}")
            return _finalize()

        if isinstance(two_stage_core, Mapping):
            gate_mode = str(two_stage_core.get("gate_mode") or "").strip().lower()
            gate_source = str(two_stage_core.get("gate_source") or "unknown").strip().lower()
            if bool(two_stage_core.get("enabled")) and gate_mode == "primary":
                if bool(two_stage_core.get("fail_closed")) or not bool(
                    two_stage_core.get("selected_reliable", True)
                ):
                    _add_blocking(f"two_stage_primary_unreliable:{room_key}:{gate_source}")

        if candidate_f1 is not None:
            threshold = self._resolve_scheduled_threshold(room_policy.get("schedule", []), training_days)
            if threshold is not None:
                if validation_evaluable:
                    if float(candidate_f1) < float(threshold):
                        _add_blocking(
                            f"room_threshold_failed:{room_key}:f1={float(candidate_f1):.3f}<required={float(threshold):.3f}"
                        )
                else:
                    _add_watch(
                        f"room_threshold_not_evaluable:{room_key}:support={validation_min_support}<"
                        f"required={int(required_validation_support)}"
                    )
        else:
            if metric_source == "insufficient_validation_evidence" or insufficient_validation_evidence:
                _add_watch(
                    f"candidate_metric_not_evaluable:{room_key}:support={validation_min_support}<"
                    f"required={int(required_validation_support)}"
                )
            elif samples < 100:
                bootstrap_reason = f"bootstrap_small_dataset_no_holdout:{room_key}:samples={samples}"
                if bool(gate_policy.block_on_train_fallback_metrics):
                    _add_blocking(bootstrap_reason)
                else:
                    _add_watch(bootstrap_reason)
            else:
                _add_blocking(f"candidate_metric_missing:{room_key}:macro_f1")

        no_regress = policy.get("release_gates", {}).get("no_regress", {})
        exempt_rooms = {normalize_room_name(room) for room in no_regress.get("exempt_rooms", [])}
        max_drop = float(no_regress.get("max_drop_from_champion", gate_policy.max_drop_from_champion_default))
        if (
            room_key not in exempt_rooms
            and champion_meta
            and candidate_f1 is not None
        ):
            champion_metrics = champion_meta.get("metrics") or {}
            champion_f1 = champion_metrics.get("macro_f1")
            if champion_f1 is not None:
                if validation_evaluable:
                    drop = float(champion_f1) - float(candidate_f1)
                    if drop > max_drop:
                        _add_blocking(
                            f"no_regress_failed:{room_key}:drop={drop:.3f}>max_drop={max_drop:.3f}"
                        )
                else:
                    _add_watch(
                        f"no_regress_not_evaluable:{room_key}:support={validation_min_support}<"
                        f"required={int(required_validation_support)}"
                    )

        return _finalize()

    def _derive_lane_b_event_metrics(
        self,
        room_name: str,
        candidate_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Derive Lane B gate metrics from room-level per-label validation metrics."""
        room_key = normalize_room_name(room_name)
        event_map = get_lane_b_event_labels_for_room(room_key)
        if not event_map:
            return {"event_recalls": {}, "event_supports": {}}

        raw_recall = candidate_metrics.get("per_label_recall") or {}
        raw_support = candidate_metrics.get("per_label_support") or {}
        per_label_recall = {
            str(k).strip().lower(): float(v)
            for k, v in raw_recall.items()
            if v is not None
        } if isinstance(raw_recall, dict) else {}
        per_label_support = {
            str(k).strip().lower(): int(v)
            for k, v in raw_support.items()
            if v is not None
        } if isinstance(raw_support, dict) else {}

        event_recalls: Dict[str, float] = {}
        event_supports: Dict[str, int] = {}
        for event_name, labels in event_map.items():
            total_support = 0
            weighted_recall = 0.0
            for label in labels:
                key = str(label).strip().lower()
                support = int(per_label_support.get(key, 0) or 0)
                recall = float(per_label_recall.get(key, 0.0) or 0.0)
                total_support += support
                weighted_recall += recall * float(support)
            event_supports[event_name] = int(total_support)
            event_recalls[event_name] = float(weighted_recall / float(total_support)) if total_support > 0 else 0.0

        total_label_support = int(sum(per_label_support.values()))
        unknown_support = int(per_label_support.get("unknown", 0))
        unknown_rate = float(unknown_support / total_label_support) if total_label_support > 0 else 0.0
        return {
            "event_recalls": event_recalls,
            "event_supports": event_supports,
            "unknown_rate_global": unknown_rate,
            "unknown_rate_per_room": {room_key: unknown_rate},
            "input_support_total": total_label_support,
        }

    def _evaluate_lane_b_event_gates(
        self,
        room_name: str,
        candidate_metrics: Dict[str, Any],
        *,
        target_date: Optional[datetime] = None,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Evaluate Lane B event gates and return hard-block signal for promotion.

        Hard block is triggered only when Lane B report is non-promotable
        (critical failures).
        """
        if not isinstance(candidate_metrics, dict):
            return True, [], {"status": "not_evaluated", "reason": "invalid_candidate_metrics"}
        if "per_label_recall" not in candidate_metrics and "per_label_support" not in candidate_metrics:
            return True, [], {"status": "not_evaluated", "reason": "missing_per_label_metrics"}

        gate_metrics = self._derive_lane_b_event_metrics(room_name, candidate_metrics)
        event_recalls = gate_metrics.get("event_recalls") or {}
        if not event_recalls:
            return True, [], {"status": "not_evaluated", "reason": "no_event_mapping_for_room"}

        try:
            from ml.event_gates import EventGateChecker, EventGateThresholds
        except Exception as exc:
            logger.warning(f"Lane B gate modules unavailable; skipping event gates: {exc}")
            return True, [], {"status": "not_evaluated", "reason": f"lane_b_gate_module_unavailable:{exc}"}

        evidence_profile = str(self._active_policy().release_gate.evidence_profile).strip().lower()
        try:
            training_days = float(candidate_metrics.get("training_days", 0.0) or 0.0)
        except Exception:
            training_days = 0.0
        short_window_max_days = max(1.0, self._read_float_env("RELEASE_GATE_BOOTSTRAP_MAX_TRAINING_DAYS", 14.0))
        is_short_window_pilot = (
            evidence_profile in {"pilot_stage_a", "pilot_stage_b"}
            and training_days <= (short_window_max_days + 1e-3)
        )
        default_min_support = 10 if is_short_window_pilot else 30
        min_support_raw = os.getenv("LANE_B_EVENT_GATE_MIN_SUPPORT")
        try:
            configured_min_support = int(min_support_raw) if min_support_raw is not None else int(default_min_support)
        except (TypeError, ValueError):
            configured_min_support = int(default_min_support)
        configured_min_support = max(1, int(configured_min_support))
        checker = EventGateChecker(
            EventGateThresholds(min_support_for_tier_gates=configured_min_support)
        )

        eval_date = (
            pd.to_datetime(target_date).date()
            if target_date is not None
            else datetime.now(timezone.utc).date()
        )
        report = checker.check_all_gates(gate_metrics, eval_date)
        report_dict = report.to_dict()
        report_dict["derived_gate_metrics"] = gate_metrics
        report_dict["configured_min_support_for_tier_gates"] = int(configured_min_support)
        report_dict["release_gate_evidence_profile"] = evidence_profile

        if report.is_promotable:
            report_dict["enforcement"] = "hard"
            return True, [], report_dict

        room_key = normalize_room_name(room_name)
        failures = list(report.critical_failures or [])
        if not failures:
            failures = ["critical_failure"]
        reasons = [f"lane_b_gate_failed:{room_key}:{name}" for name in failures]
        if self._is_bootstrap_release_phase(candidate_metrics):
            collapse_failures = [name for name in failures if str(name).startswith("collapse_")]
            if collapse_failures:
                report_dict["enforcement"] = "hard_bootstrap_collapse_guard"
                collapse_reasons = [f"lane_b_gate_failed:{room_key}:{name}" for name in collapse_failures]
                return False, collapse_reasons, report_dict
            report_dict["enforcement"] = "watch_only_bootstrap"
            report_dict["soft_failed_critical_failures"] = list(failures)
            report_dict["soft_failed_reasons"] = list(reasons)
            logger.warning(
                "Lane B gate soft-fail during bootstrap for %s/%s; failures=%s",
                room_key,
                str(candidate_metrics.get("elder_id") or "unknown"),
                failures,
            )
            return True, [], report_dict
        report_dict["enforcement"] = "hard"
        return False, reasons, report_dict

    def _decode_label_ids(self, room_name: str, label_ids: np.ndarray) -> List[str]:
        """
        Decode encoded labels to canonical lowercase strings.
        """
        ids = np.asarray(label_ids, dtype=np.int64)
        label_encoder = self.platform.label_encoders.get(room_name)
        classes = getattr(label_encoder, "classes_", None) if label_encoder is not None else None
        decoded: List[str] = []
        for class_id in ids:
            idx = int(class_id)
            if classes is not None and 0 <= idx < len(classes):
                decoded.append(str(classes[idx]).strip().lower())
            else:
                decoded.append(f"class_{idx}")
        return decoded

    def _build_beta6_parity_payload(
        self,
        *,
        room_name: str,
        y_pred_probs: np.ndarray,
        max_steps: int = 128,
    ) -> Dict[str, Any]:
        """
        Build a compact occupancy-level fixed trace for the Beta 6 parity gate.

        The parity contract currently operates on shared occupied/unoccupied
        semantics, so room-specific labels are collapsed to occupancy tokens and
        explicit uncertainty labels are routed through `uncertainty_state`.
        """
        probs = np.asarray(y_pred_probs, dtype=np.float32)
        if probs.ndim != 2 or probs.shape[0] <= 0 or probs.shape[1] <= 0:
            return {}

        label_encoder = self.platform.label_encoders.get(room_name)
        classes = getattr(label_encoder, "classes_", None) if label_encoder is not None else None
        if classes is None or len(classes) <= 0:
            return {}

        predicted_ids = np.argmax(probs, axis=1).astype(np.int32)
        if len(predicted_ids) == 0:
            return {}

        capped_steps = max(1, int(max_steps))
        if len(predicted_ids) > capped_steps:
            step_indices = np.linspace(0, len(predicted_ids) - 1, num=capped_steps, dtype=np.int64)
        else:
            step_indices = np.arange(len(predicted_ids), dtype=np.int64)

        uncertainty_labels = {"low_confidence", "unknown", "outside_sensed_space"}
        unoccupied_labels = {"unoccupied", "vacant", "room_unoccupied"}
        trace_steps: List[Dict[str, Any]] = []
        for idx in step_indices:
            class_id = int(predicted_ids[int(idx)])
            if class_id < 0 or class_id >= len(classes):
                continue
            label_name = str(classes[class_id]).strip().lower()
            if not label_name:
                continue

            if label_name in uncertainty_labels:
                trace_steps.append(
                    {
                        "label": "unoccupied",
                        "uncertainty_state": label_name,
                    }
                )
            elif label_name in unoccupied_labels:
                trace_steps.append({"label": "unoccupied"})
            else:
                trace_steps.append({"label": "occupied"})

        if not trace_steps:
            return {}

        policy_payload = {
            "spike_suppression": bool(DEFAULT_DECODER_POLICY.spike_suppression),
        }
        return {
            "beta6_parity_trace": trace_steps,
            "beta6_runtime_policy": dict(policy_payload),
            "beta6_eval_policy": dict(policy_payload),
        }

    def _evaluate_event_first_shadow(
        self,
        *,
        room_name: str,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        timestamps: np.ndarray,
        target_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate event-first shadow predictions for a single room.
        """
        result: Dict[str, Any] = {"enabled": True, "evaluated": False}
        if y_true is None or y_pred_probs is None or timestamps is None:
            result["reason"] = "missing_holdout_arrays"
            return result

        y_true_arr = np.asarray(y_true, dtype=np.int32)
        probs_arr = np.asarray(y_pred_probs, dtype=np.float32)
        ts_arr = pd.to_datetime(np.asarray(timestamps), errors="coerce")

        n = min(len(y_true_arr), len(probs_arr), len(ts_arr))
        if n <= 0:
            result["reason"] = "empty_holdout_arrays"
            return result

        y_true_arr = y_true_arr[:n]
        probs_arr = probs_arr[:n]
        ts_arr = ts_arr[:n]

        valid_mask = ~pd.isna(ts_arr)
        if not bool(np.any(valid_mask)):
            result["reason"] = "no_valid_timestamps"
            return result

        y_true_arr = y_true_arr[valid_mask]
        probs_arr = probs_arr[valid_mask]
        ts_arr = ts_arr[valid_mask]
        if len(y_true_arr) == 0:
            result["reason"] = "empty_after_timestamp_filter"
            return result

        label_encoder = self.platform.label_encoders.get(room_name)
        classes = getattr(label_encoder, "classes_", None) if label_encoder is not None else None
        if classes is None:
            result["reason"] = "missing_label_encoder_classes"
            return result
        class_names = [str(name).strip().lower() for name in classes]

        try:
            dual = derive_dual_head_probabilities(
                multiclass_probs=probs_arr,
                class_names=class_names,
                occupancy_label="unoccupied",
            )
        except Exception as exc:
            result["reason"] = f"dual_head_conversion_error:{exc}"
            return result

        event_policy = self._active_policy().event_first
        decoder = EventDecoder(
            DecoderConfig(
                occupancy_on_threshold=float(event_policy.decoder_on_threshold),
                occupancy_off_threshold=float(event_policy.decoder_off_threshold),
                hysteresis_min_windows=max(1, int(event_policy.decoder_min_on_steps)),
                use_unknown_fallback=bool(event_policy.unknown_enabled),
            )
        )

        timestamps_list = [pd.Timestamp(ts).to_pydatetime() for ts in ts_arr]
        decoded_df = decoder.decode_to_dataframe(
            occupancy_probs=dual.occupancy_prob,
            activity_probs=dual.activity_probs,
            timestamps=timestamps_list,
            room_name=room_name,
        )
        if decoded_df.empty:
            result["reason"] = "decoder_output_empty"
            return result

        room_key = normalize_room_name(room_name)
        decoded_df["room"] = room_key
        ground_truth_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts_arr),
                "label": self._decode_label_ids(room_name, y_true_arr),
                "room": room_key,
            }
        )

        kpi_metrics = EventKPICalculator(EventKPIConfig(timestamp_tolerance_seconds=15.0)).calculate_metrics(
            decoded_df,
            ground_truth_df,
        )
        thresholds = EventGateThresholds(
            unknown_rate_global_max=float(event_policy.unknown_rate_global_cap),
            unknown_rate_per_room_max=float(event_policy.unknown_rate_room_cap),
        )
        eval_date = (
            pd.to_datetime(target_date).date()
            if target_date is not None
            else pd.to_datetime(ts_arr[-1]).date()
        )
        gate_report = EventGateChecker(thresholds=thresholds).check_all_gates(
            kpi_metrics.to_gate_metrics(),
            eval_date,
        )

        artifact = {
            "run_timestamp_utc": _utc_now_iso_z(),
            "room": room_key,
            "target_date": eval_date.isoformat(),
            "windows": int(len(decoded_df)),
            "decoder": {
                "occupancy_on_threshold": float(event_policy.decoder_on_threshold),
                "occupancy_off_threshold": float(event_policy.decoder_off_threshold),
                "hysteresis_min_windows": int(event_policy.decoder_min_on_steps),
                "unknown_enabled": bool(event_policy.unknown_enabled),
            },
            "contract": {
                "head_a": "occupancy_prob",
                "head_b_labels": sorted(list(dual.activity_probs.keys())),
                "occupancy_mean": float(np.mean(dual.occupancy_prob)),
            },
            "kpi": {
                "home_empty_precision": float(kpi_metrics.home_empty_precision),
                "home_empty_recall": float(kpi_metrics.home_empty_recall),
                "home_empty_false_empty_rate": float(kpi_metrics.home_empty_false_empty_rate),
                "unknown_rate_global": float(kpi_metrics.unknown_rate_global),
                "unknown_rate_per_room": dict(kpi_metrics.unknown_rate_per_room),
                "event_recalls": dict(kpi_metrics.event_recalls),
                "event_supports": dict(kpi_metrics.event_supports),
            },
            "gate_report": gate_report.to_dict(),
        }

        summary = {
            "enabled": True,
            "evaluated": True,
            "overall_status": gate_report.overall_status.value,
            "is_promotable": bool(gate_report.is_promotable),
            "critical_failures": list(gate_report.critical_failures or []),
            "unknown_rate_global": float(kpi_metrics.unknown_rate_global),
            "unknown_rate_room": float(kpi_metrics.unknown_rate_per_room.get(room_key, 0.0)),
        }

        return {
            "enabled": True,
            "evaluated": True,
            "summary": summary,
            "artifact": artifact,
        }

    def _write_event_first_shadow_artifact(
        self,
        *,
        elder_id: str,
        room_name: str,
        saved_version: int,
        payload: Dict[str, Any],
    ) -> Dict[str, str] | None:
        """
        Persist event-first shadow artifact as versioned + latest files.
        """
        try:
            models_dir = self.registry.get_models_dir(elder_id)
            models_dir.mkdir(parents=True, exist_ok=True)
            versioned_path = models_dir / f"{room_name}_v{int(saved_version)}_event_first_shadow.json"
            latest_path = models_dir / f"{room_name}_event_first_shadow.json"
            for path in (versioned_path, latest_path):
                with open(path, "w") as f:
                    json.dump(payload, f, indent=2, default=str)
            return {"versioned": str(versioned_path), "latest": str(latest_path)}
        except Exception as e:
            logger.warning(f"Failed writing event-first shadow artifact for {elder_id}/{room_name}: {e}")
            return None

    def _write_decision_trace(
        self,
        elder_id: str,
        room_name: str,
        saved_version: int,
        payload: Dict[str, Any],
    ) -> Dict[str, str] | None:
        """
        Persist room training decision trace for debugging and reproducibility.
        """
        try:
            models_dir = self.registry.get_models_dir(elder_id)
            models_dir.mkdir(parents=True, exist_ok=True)
            versioned_path = models_dir / f"{room_name}_v{int(saved_version)}_decision_trace.json"
            latest_path = models_dir / f"{room_name}_decision_trace.json"
            for path in (versioned_path, latest_path):
                with open(path, "w") as f:
                    json.dump(payload, f, indent=2, default=str)
            return {
                "versioned": str(versioned_path),
                "latest": str(latest_path),
            }
        except Exception as e:
            logger.warning(f"Failed writing decision trace for {elder_id}/{room_name}: {e}")
            return None

    @staticmethod
    def _resolve_fragile_room_status(
        *,
        gate_pass: bool,
        runtime_topology_matches: bool,
        blocking_reasons: Sequence[str],
    ) -> str:
        if not bool(gate_pass):
            return "block"
        reasons = [str(reason).strip().lower() for reason in (blocking_reasons or []) if str(reason).strip()]
        strict_blockers = [
            reason for reason in reasons if not reason.startswith("runtime_topology_mismatch:")
        ]
        if strict_blockers:
            return "block"
        if reasons:
            return "conditional"
        if not bool(runtime_topology_matches):
            return "conditional"
        return "pass"

    def _evaluate_grouped_regime_gate(
        self,
        *,
        room_name: str,
        grouped_fragility: Mapping[str, Any] | None,
        policy: Optional[TrainingPolicy] = None,
    ) -> Dict[str, Any]:
        gate_report: Dict[str, Any] = {
            "pass": True,
            "blocking_reasons": [],
            "failures": [],
        }
        active_policy = policy if isinstance(policy, TrainingPolicy) else self.policy
        if not active_policy.training_profile.room_uses_grouped_fragility_gate(room_name):
            return gate_report
        payload = grouped_fragility if isinstance(grouped_fragility, Mapping) else {}
        stability_gate = payload.get("stability_gate")
        if not isinstance(stability_gate, Mapping):
            return gate_report

        room_token = normalize_room_name(room_name)
        blocking_reasons: List[str] = []
        failures: List[Dict[str, Any]] = []
        for raw_failure in stability_gate.get("failures") or []:
            if not isinstance(raw_failure, Mapping):
                continue
            regime = str(raw_failure.get("regime") or "unknown").strip().lower()
            worst_slice = str(raw_failure.get("worst_slice") or "unknown").strip() or "unknown"
            try:
                worst_slice_macro_f1 = float(raw_failure.get("worst_slice_macro_f1"))
                floor = float(raw_failure.get("floor"))
            except (TypeError, ValueError):
                continue
            failures.append(
                {
                    "regime": regime,
                    "worst_slice": worst_slice,
                    "worst_slice_macro_f1": round(worst_slice_macro_f1, 4),
                    "floor": round(floor, 4),
                }
            )
            blocking_reasons.append(
                "fragile_grouped_regime_floor_failed:"
                f"{room_token}:{regime}:{worst_slice}:{worst_slice_macro_f1:.3f}<{floor:.3f}"
            )

        explicit_pass = bool(stability_gate.get("pass", not failures))
        if not explicit_pass and not blocking_reasons:
            blocking_reasons.append(f"fragile_grouped_regime_gate_failed:{room_token}")
        gate_report["pass"] = bool(explicit_pass and not failures)
        gate_report["blocking_reasons"] = blocking_reasons
        gate_report["failures"] = failures
        return gate_report

    def _resolve_room_runtime_topology_expectation(
        self,
        *,
        elder_id: str,
        room_name: str,
        saved_version: int,
        runtime_use_two_stage: bool,
    ) -> Dict[str, Any]:
        actual_runtime_mode = "two_stage" if bool(runtime_use_two_stage) else "single_stage"
        expectation: Dict[str, Any] = {
            "expected_runtime_mode": actual_runtime_mode,
            "actual_runtime_mode": actual_runtime_mode,
            "matches": True,
            "source": "runtime_selection_fallback",
        }
        try:
            models_dir = self.registry.get_models_dir(elder_id)
            versioned_meta_path = models_dir / f"{room_name}_v{int(saved_version)}_two_stage_meta.json"
            latest_meta_path = models_dir / f"{room_name}_two_stage_meta.json"
            meta_path: Optional[Path] = None
            source = "saved_two_stage_meta"
            if versioned_meta_path.exists():
                meta_path = versioned_meta_path
            elif latest_meta_path.exists():
                meta_path = latest_meta_path
                source = "saved_two_stage_meta_latest_alias"
            if meta_path is None:
                return expectation

            payload = json.loads(meta_path.read_text())
            if str(payload.get("schema_version", "")).strip() != "beta6.two_stage_core.v1":
                expectation["source"] = "invalid_two_stage_meta_schema"
                return expectation

            if meta_path == latest_meta_path:
                try:
                    payload_saved_version = int(payload.get("saved_version", 0) or 0)
                except (TypeError, ValueError):
                    payload_saved_version = 0
                if int(payload_saved_version) != int(saved_version):
                    expectation.update(
                        {
                            "source": "stale_latest_two_stage_meta",
                            "meta_path": str(meta_path),
                            "stale_saved_version": int(payload_saved_version),
                        }
                    )
                    return expectation

            expected_runtime_mode = (
                "two_stage"
                if bool(payload.get("runtime_enabled", True))
                else "single_stage"
            )
            expectation.update(
                {
                    "expected_runtime_mode": expected_runtime_mode,
                    "actual_runtime_mode": actual_runtime_mode,
                    "matches": expected_runtime_mode == actual_runtime_mode,
                    "source": source,
                    "meta_path": str(meta_path),
                }
            )
            return expectation
        except Exception as e:
            expectation["source"] = "runtime_expectation_error"
            expectation["error"] = f"{type(e).__name__}: {e}"
            return expectation

    def _resolve_unoccupied_downsample_config(self, room_name: str) -> Dict[str, float | int]:
        """Resolve unoccupied downsample config from typed policy."""
        return self._active_policy().unoccupied_downsample.resolve(room_name)

    def _resolve_minority_sampling_config(self, room_name: str) -> Dict[str, float | int | bool]:
        """Resolve minority sampling config from typed policy."""
        return self._active_policy().minority_sampling.resolve(room_name)

    def _resolve_training_strategy(self, room_name: str) -> str:
        profile = self._active_policy().training_profile
        if profile.is_factorized_primary_room(room_name):
            return "factorized_primary"
        return "single_stage"

    def _resolve_transition_focus_sampling_config(
        self,
        room_name: str,
    ) -> Dict[str, str | int | bool]:
        return self._active_policy().training_profile.resolve_transition_focus(room_name)

    @staticmethod
    def _parse_room_override_map(raw: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        txt = str(raw or "").strip()
        if not txt:
            return result
        for token in txt.split(","):
            item = str(token).strip()
            if not item:
                continue
            if ":" in item:
                room_raw, value_raw = item.split(":", 1)
            elif "=" in item:
                room_raw, value_raw = item.split("=", 1)
            else:
                continue
            room_key = normalize_room_name(room_raw)
            value = str(value_raw).strip()
            if room_key and value:
                result[room_key] = value
        return result

    def _resolve_min_holdout_support(self, room_name: str) -> int:
        """
        Resolve evidence floor for critical-label support in holdout/calibration splits.

        Precedence:
        1) TRAIN_MIN_HOLDOUT_SUPPORT_BY_ROOM
        2) TRAIN_MIN_HOLDOUT_SUPPORT
        3) beta6_policy_defaults.yaml training.min_holdout_support_by_room/default
        """
        room_key = normalize_room_name(room_name)
        default_value = int(get_training_min_holdout_support_default())
        by_room_defaults = get_training_min_holdout_support_by_room()
        resolved = int(by_room_defaults.get(room_key, default_value))

        raw_global = os.getenv("TRAIN_MIN_HOLDOUT_SUPPORT")
        if raw_global is not None:
            try:
                resolved = int(raw_global)
            except (TypeError, ValueError):
                pass

        room_map = self._parse_room_override_map(os.getenv("TRAIN_MIN_HOLDOUT_SUPPORT_BY_ROOM", ""))
        if room_key in room_map:
            try:
                resolved = int(room_map[room_key])
            except (TypeError, ValueError):
                pass

        return max(1, int(resolved))

    def _resolve_critical_labels(self, room_name: str) -> List[str]:
        """
        Resolve room-level critical labels used for support/collapse guardrails.
        """
        room_key = normalize_room_name(room_name)
        labels = set(get_critical_labels_for_room(room_key))
        for key in (self._active_policy().release_gate.min_recall_by_room_label or {}).keys():
            key_txt = str(key).strip().lower()
            if key_txt.startswith(f"{room_key}."):
                labels.add(key_txt.split(".", 1)[1])
        return sorted(labels)

    def _resolve_room_label_equivalents(self, room_name: str) -> Dict[str, List[str]]:
        """
        Resolve per-room label equivalence classes from Lane-B event mapping.

        Labels under the same event are treated as semantic equivalents for
        release-gate support/recall checks (e.g. sleep/sleeping).
        """
        room_key = normalize_room_name(room_name)
        event_map = get_lane_b_event_labels_for_room(room_key)
        if not isinstance(event_map, dict):
            return {}

        groups: List[set[str]] = []
        for labels in event_map.values():
            if not isinstance(labels, (list, tuple, set)):
                continue
            tokens = sorted(
                {
                    str(item).strip().lower()
                    for item in labels
                    if str(item).strip()
                }
            )
            if not tokens:
                continue
            groups.append(set(tokens))

        for token, peers in get_label_alias_equivalents().items():
            alias_group = {
                str(item).strip().lower()
                for item in ([token] + list(peers))
                if str(item).strip()
            }
            if alias_group:
                groups.append(alias_group)

        resolved: Dict[str, set[str]] = {}
        for group in groups:
            for token in group:
                bucket = resolved.setdefault(token, set())
                bucket.update(group)

        # Ensure transitive closure across overlapping groups.
        changed = True
        while changed:
            changed = False
            for token, peers in list(resolved.items()):
                expanded = set(peers)
                for peer in list(peers):
                    expanded.update(resolved.get(peer, set()))
                if expanded != peers:
                    resolved[token] = expanded
                    changed = True

        return {
            token: sorted(peers)
            for token, peers in resolved.items()
            if token
        }

    @staticmethod
    def _select_best_label_variant(
        label_name: str,
        *,
        per_label_support: Mapping[str, int],
        per_label_recall: Mapping[str, float],
        equivalents: Mapping[str, Sequence[str]],
    ) -> Tuple[str, int, Optional[float]]:
        """
        Select the strongest-supported equivalent label for gate evaluation.
        """
        label_key = str(label_name).strip().lower()
        candidates = {label_key}
        for item in equivalents.get(label_key, []):
            token = str(item).strip().lower()
            if token:
                candidates.add(token)

        ranked = sorted(candidates)
        selected = max(
            ranked,
            key=lambda item: (
                int(per_label_support.get(item, 0) or 0),
                1 if per_label_recall.get(item) is not None else 0,
                1 if item == label_key else 0,
            ),
        )
        support = int(per_label_support.get(selected, 0) or 0)
        recall_raw = per_label_recall.get(selected)
        recall = float(recall_raw) if recall_raw is not None else None
        return selected, support, recall

    def _summarize_gate_aligned_validation(
        self,
        *,
        room_name: str,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
    ) -> Dict[str, Any]:
        y_true_arr = np.asarray(y_true, dtype=np.int32)
        y_pred_arr = np.asarray(y_pred_probs, dtype=np.float32)
        if y_pred_arr.ndim != 2:
            raise ValueError(f"unexpected_prediction_shape:{getattr(y_pred_arr, 'shape', None)}")
        if int(y_pred_arr.shape[0]) != int(len(y_true_arr)):
            raise ValueError(
                f"prediction_length_mismatch:{int(y_pred_arr.shape[0])}!={int(len(y_true_arr))}"
            )

        y_pred_classes = np.argmax(y_pred_arr, axis=1).astype(np.int32)
        report = classification_report(y_true_arr, y_pred_classes, output_dict=True, zero_division=0)
        macro_f1 = float(report.get("macro avg", {}).get("f1-score", 0.0) or 0.0)

        label_encoder = self.platform.label_encoders.get(room_name)
        classes = getattr(label_encoder, "classes_", None) if label_encoder is not None else None
        per_label_precision: Dict[str, float] = {}
        per_label_recall: Dict[str, float] = {}
        per_label_support: Dict[str, int] = {}
        if classes is not None:
            for class_id in range(len(classes)):
                class_key = str(int(class_id))
                label_name = str(classes[int(class_id)]).strip().lower()
                class_metrics = report.get(class_key, {})
                per_label_precision[label_name] = float(class_metrics.get("precision", 0.0) or 0.0)
                per_label_recall[label_name] = float(class_metrics.get("recall", 0.0) or 0.0)
                per_label_support[label_name] = int(class_metrics.get("support", 0) or 0)

        equivalents = self._resolve_room_label_equivalents(room_name)
        critical_labels = list(self._resolve_critical_labels(room_name))
        critical_support_floor = self._resolve_gate_aligned_critical_support_floor()
        critical_recall_floor = self._resolve_gate_aligned_critical_recall_floor()

        critical_stats: Dict[str, Dict[str, Any]] = {}
        critical_recalls: List[float] = []
        critical_floor_failures: List[str] = []
        for label in critical_labels:
            selected_label, support, recall = self._select_best_label_variant(
                label,
                per_label_support=per_label_support,
                per_label_recall=per_label_recall,
                equivalents=equivalents,
            )
            recall_value = float(recall) if recall is not None else 0.0
            eligible = int(support) >= int(critical_support_floor)
            critical_stats[str(label)] = {
                "selected_label": str(selected_label),
                "support": int(support),
                "recall": float(recall_value),
                "eligible_for_floor": bool(eligible),
            }
            if eligible:
                critical_recalls.append(float(recall_value))
                if float(recall_value) < float(critical_recall_floor):
                    critical_floor_failures.append(str(selected_label))

        critical_recall_mean = float(np.mean(critical_recalls)) if critical_recalls else 0.0
        critical_recall_min = float(np.min(critical_recalls)) if critical_recalls else 0.0

        lane_b_event_map = get_lane_b_event_labels_for_room(normalize_room_name(room_name))
        lane_b_thresholds = EventGateThresholds()
        lane_b_floor_failures: List[str] = []
        lane_b_stats: Dict[str, Dict[str, Any]] = {}
        lane_b_recalls: List[float] = []
        for event_name, labels in lane_b_event_map.items():
            total_support = 0
            weighted_recall = 0.0
            for label in labels:
                key = str(label).strip().lower()
                support = int(per_label_support.get(key, 0) or 0)
                recall = float(per_label_recall.get(key, 0.0) or 0.0)
                total_support += support
                weighted_recall += recall * float(support)
            event_recall = float(weighted_recall / float(total_support)) if total_support > 0 else 0.0
            tier = EventGateChecker.EVENT_TIERS.get(str(event_name), CriticalityTier.TIER_3)
            if tier == CriticalityTier.TIER_1:
                required_recall = float(lane_b_thresholds.tier_1_recall_min)
            elif tier == CriticalityTier.TIER_2:
                required_recall = float(lane_b_thresholds.tier_2_recall_min)
            else:
                required_recall = float(lane_b_thresholds.tier_3_recall_min)
            eligible = int(total_support) >= int(lane_b_thresholds.min_support_for_tier_gates)
            lane_b_stats[str(event_name)] = {
                "labels": [str(item).strip().lower() for item in labels],
                "support": int(total_support),
                "recall": float(event_recall),
                "required_recall": float(required_recall),
                "tier": int(tier.value),
                "eligible_for_floor": bool(eligible),
            }
            if eligible:
                lane_b_recalls.append(float(event_recall))
                if float(event_recall) < float(required_recall):
                    lane_b_floor_failures.append(str(event_name))
        lane_b_recall_mean = float(np.mean(lane_b_recalls)) if lane_b_recalls else 0.0

        precision_floor_by_label = self._resolve_gate_aligned_precision_floors(room_name)
        precision_floor_failures: List[str] = []
        precision_floor_stats: Dict[str, Dict[str, Any]] = {}
        for label, floor in precision_floor_by_label.items():
            selected_label, support, precision = self._select_best_label_variant(
                label,
                per_label_support=per_label_support,
                per_label_recall=per_label_precision,
                equivalents=equivalents,
            )
            precision_value = float(precision) if precision is not None else 0.0
            eligible = int(support) >= int(critical_support_floor)
            precision_floor_stats[str(label)] = {
                "selected_label": str(selected_label),
                "support": int(support),
                "precision": float(precision_value),
                "required_precision": float(floor),
                "eligible_for_floor": bool(eligible),
            }
            if eligible and float(precision_value) < float(floor):
                precision_floor_failures.append(str(selected_label))

        dominant_class_id = -1
        dominant_class_count = 0
        dominant_class_share = 0.0
        predicted_distribution: Dict[str, int] = {}
        if len(y_pred_classes) > 0:
            pred_ids, pred_counts = np.unique(y_pred_classes, return_counts=True)
            idx = int(np.argmax(pred_counts))
            dominant_class_id = int(pred_ids[idx])
            dominant_class_count = int(pred_counts[idx])
            dominant_class_share = float(dominant_class_count) / float(len(y_pred_classes))
            for class_id, count in zip(pred_ids, pred_counts):
                label_name = self._get_label_name(room_name, int(class_id))
                key = str(label_name).strip().lower() if label_name else f"class_{int(class_id)}"
                predicted_distribution[key] = int(count)

        collapse_ratio = self._resolve_gate_aligned_collapse_ratio()
        collapsed = bool(dominant_class_share >= float(collapse_ratio))
        critical_floor_failed = bool(len(critical_floor_failures) > 0)
        lane_b_floor_failed = bool(len(lane_b_floor_failures) > 0)
        precision_floor_failed = bool(len(precision_floor_failures) > 0)

        score = float(macro_f1)
        score += float(self._resolve_gate_aligned_critical_weight()) * float(critical_recall_mean)
        score += float(self._resolve_gate_aligned_lane_b_weight()) * float(lane_b_recall_mean)
        if collapsed:
            score -= float(self._resolve_gate_aligned_collapse_penalty())
        if critical_floor_failed:
            score -= float(self._resolve_gate_aligned_floor_penalty())
        if lane_b_floor_failed:
            score -= float(self._resolve_gate_aligned_lane_b_floor_penalty())
        if precision_floor_failed:
            score -= float(self._resolve_gate_aligned_precision_floor_penalty())

        dominant_label = self._get_label_name(room_name, dominant_class_id)
        return {
            "macro_f1": float(macro_f1),
            "critical_labels": critical_labels,
            "critical_support_floor": int(critical_support_floor),
            "critical_recall_floor": float(critical_recall_floor),
            "critical_recall_mean": float(critical_recall_mean),
            "critical_recall_min": float(critical_recall_min),
            "critical_floor_failures": critical_floor_failures,
            "critical_stats": critical_stats,
            "per_label_recall": per_label_recall,
            "per_label_support": per_label_support,
            "per_label_precision": per_label_precision,
            "precision_floor_by_label": precision_floor_by_label,
            "precision_floor_failures": precision_floor_failures,
            "precision_floor_stats": precision_floor_stats,
            "dominant_class_id": int(dominant_class_id),
            "dominant_class_label": (
                str(dominant_label).strip().lower()
                if dominant_label is not None else f"class_{int(dominant_class_id)}"
            ),
            "dominant_class_count": int(dominant_class_count),
            "dominant_class_share": float(dominant_class_share),
            "collapse_ratio": float(collapse_ratio),
            "collapsed": bool(collapsed),
            "predicted_class_distribution": predicted_distribution,
            "lane_b_event_stats": lane_b_stats,
            "lane_b_recall_mean": float(lane_b_recall_mean),
            "lane_b_floor_failures": lane_b_floor_failures,
            "gate_aligned_score": float(score),
        }

    def _select_final_two_stage_gate_source(
        self,
        *,
        room_name: str,
        y_true: np.ndarray,
        single_stage_probs: np.ndarray,
        two_stage_probs: Optional[np.ndarray],
        gate_mode: str,
        champion_meta: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        gate_mode_norm = str(gate_mode or "shadow").strip().lower()
        y_true_arr = np.asarray(y_true, dtype=np.int32)
        single_probs = np.asarray(single_stage_probs, dtype=np.float32)

        no_regress = self._resolve_no_regress_macro_f1_floor(
            room_name=room_name,
            champion_meta=champion_meta,
        )

        def _build_payload(source_name: str, probs: np.ndarray) -> Dict[str, Any]:
            summary = self._summarize_gate_aligned_validation(
                room_name=room_name,
                y_true=y_true_arr,
                y_pred_probs=np.asarray(probs, dtype=np.float32),
            )
            macro_f1 = summary.get("macro_f1")
            macro_f1_value = float(macro_f1) if macro_f1 is not None else None
            no_regress_ok = True
            floor = no_regress.get("target_macro_f1_floor")
            if bool(no_regress.get("enabled")) and floor is not None:
                no_regress_ok = bool(
                    macro_f1_value is not None and macro_f1_value + 1e-9 >= float(floor)
                )
            return {
                "source": str(source_name),
                "summary": dict(summary),
                "collapsed": bool(summary.get("collapsed", False)),
                "macro_f1": macro_f1_value,
                "gate_aligned_score": float(summary.get("gate_aligned_score", float("-inf"))),
                "no_regress_ok": bool(no_regress_ok),
            }

        single_payload = _build_payload("single_stage", single_probs)
        two_payload: Optional[Dict[str, Any]] = None
        if two_stage_probs is not None:
            two_payload = _build_payload("two_stage", np.asarray(two_stage_probs, dtype=np.float32))

        selection: Dict[str, Any] = {
            "gate_mode": gate_mode_norm,
            "no_regress": dict(no_regress),
            "single_stage": single_payload,
            "two_stage": two_payload,
            "selected_source": "single_stage_only",
            "runtime_use_two_stage": False,
            "selected_summary": dict(single_payload.get("summary") or {}),
            "selected_reliable": bool(
                (not bool(single_payload.get("collapsed", False)))
                and bool(single_payload.get("no_regress_ok", True))
            ),
            "fail_closed": False,
            "fail_closed_reason": None,
        }
        if two_payload is None:
            return selection
        if gate_mode_norm != "primary":
            selection["selected_source"] = "single_stage_shadow"
            return selection

        single_reliable = bool(
            (not bool(single_payload.get("collapsed", False)))
            and bool(single_payload.get("no_regress_ok", True))
        )
        two_reliable = bool(
            (not bool(two_payload.get("collapsed", False)))
            and bool(two_payload.get("no_regress_ok", True))
        )

        if not single_reliable and not two_reliable:
            selection["selected_source"] = "single_stage_fail_closed_unreliable"
            selection["selected_summary"] = dict(single_payload.get("summary") or {})
            selection["selected_reliable"] = False
            selection["fail_closed"] = True
            selection["fail_closed_reason"] = "no_reliable_primary_candidate"
            return selection
        if two_reliable and not single_reliable:
            selection["selected_source"] = "two_stage_primary"
            selection["runtime_use_two_stage"] = True
            selection["selected_summary"] = dict(two_payload.get("summary") or {})
            selection["selected_reliable"] = True
            return selection
        if single_reliable and not two_reliable:
            if bool(two_payload.get("collapsed", False)):
                source = "single_stage_fallback_collapse"
            else:
                source = "single_stage_fallback_no_regress"
            selection["selected_source"] = source
            selection["selected_summary"] = dict(single_payload.get("summary") or {})
            selection["selected_reliable"] = True
            return selection

        def _rank(payload: Mapping[str, Any]) -> Tuple[int, int, float, float, int]:
            return (
                0 if bool(payload.get("collapsed", False)) else 1,
                1 if bool(payload.get("no_regress_ok", True)) else 0,
                float(payload.get("gate_aligned_score", float("-inf"))),
                float(payload.get("macro_f1", float("-inf"))),
                1 if str(payload.get("source")) == "two_stage" else 0,
            )

        selected = max((single_payload, two_payload), key=_rank)
        if str(selected.get("source")) == "two_stage":
            selection["selected_source"] = "two_stage_primary"
            selection["runtime_use_two_stage"] = True
            selection["selected_summary"] = dict(two_payload.get("summary") or {})
            selection["selected_reliable"] = True
            return selection

        if bool(two_payload.get("collapsed", False)) and not bool(single_payload.get("collapsed", False)):
            source = "single_stage_fallback_collapse"
        elif (
            bool(no_regress.get("enabled"))
            and bool(single_payload.get("no_regress_ok"))
            and not bool(two_payload.get("no_regress_ok"))
        ):
            source = "single_stage_fallback_no_regress"
        elif float(single_payload.get("gate_aligned_score", float("-inf"))) > (
            float(two_payload.get("gate_aligned_score", float("-inf"))) + 1e-8
        ):
            source = "single_stage_fallback_score"
        else:
            source = "single_stage_fallback_macro_f1"

        selection["selected_source"] = source
        selection["selected_summary"] = dict(single_payload.get("summary") or {})
        selection["selected_reliable"] = True
        return selection

    def _resolve_critical_class_support(
        self,
        room_name: str,
        y_seq: np.ndarray,
        min_support: int = 1,
    ) -> Dict[str, Any]:
        """
        Map critical labels to encoded class IDs and observed support.
        """
        labels = self._resolve_critical_labels(room_name)
        result: Dict[str, Any] = {
            "labels": labels,
            "class_ids": [],
            "label_to_class_id": {},
            "total_support_by_label": {},
            "missing_labels": [],
            "unsupported_labels": [],
            "min_support": int(max(1, min_support)),
        }
        if len(labels) == 0:
            return result

        le = self.platform.label_encoders.get(room_name)
        classes = getattr(le, "classes_", None) if le is not None else None
        if classes is None:
            result["missing_labels"] = list(labels)
            return result

        label_to_class_id = {
            str(label_name).strip().lower(): int(class_id)
            for class_id, label_name in enumerate(classes)
        }
        unique_classes, unique_counts = np.unique(y_seq, return_counts=True)
        total_by_class = {int(cls): int(cnt) for cls, cnt in zip(unique_classes, unique_counts)}

        required_ids: List[int] = []
        for label in labels:
            class_id = label_to_class_id.get(label)
            if class_id is None:
                result["missing_labels"].append(label)
                continue
            result["label_to_class_id"][label] = int(class_id)
            total_support = int(total_by_class.get(int(class_id), 0))
            result["total_support_by_label"][label] = total_support
            if total_support >= int(max(1, min_support)):
                required_ids.append(int(class_id))
            else:
                result["unsupported_labels"].append(label)

        result["class_ids"] = sorted(set(required_ids))
        return result

    def _resolve_two_stage_core_rooms(self) -> set[str]:
        return self._active_policy().two_stage_core.normalized_rooms()

    def _resolve_two_stage_gate_mode(self) -> str:
        return self._active_policy().two_stage_core.resolved_gate_mode()

    def _resolve_two_stage_stage_a_default_threshold(self) -> float:
        value = self._active_policy().two_stage_core.stage_a_occupied_threshold
        return float(np.clip(value, 0.0, 1.0))

    def _resolve_two_stage_stage_a_threshold_bounds(self) -> Tuple[float, float]:
        """
        Resolve stage-A occupancy threshold bounds.

        Keep these independent from generic per-class calibration floor/cap
        because stage-A (occupied-vs-unoccupied) needs room to lower threshold
        aggressively when collapse pressure is high.
        """
        return self._active_policy().two_stage_core.resolved_threshold_bounds()

    def _resolve_two_stage_stage_a_min_predicted_occupied_floor(self) -> Tuple[float, float]:
        """
        Resolve safeguards for minimum predicted occupied rate on calibration.

        Returns:
            (relative_ratio, absolute_floor)
        where:
            min_pred_rate = max(absolute_floor, true_occupied_rate * relative_ratio)
        """
        two_stage_policy = self._active_policy().two_stage_core
        ratio = float(two_stage_policy.stage_a_min_predicted_occupied_ratio)
        abs_floor = float(two_stage_policy.stage_a_min_predicted_occupied_abs)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        abs_floor = float(np.clip(abs_floor, 0.0, 1.0))
        return ratio, abs_floor

    def _two_stage_core_enabled_for_room(self, room_name: str) -> bool:
        if self._active_policy().training_profile.is_factorized_primary_room(room_name):
            return True
        return self._active_policy().two_stage_core.enabled_for_room(room_name)

    def _resolve_two_stage_class_profile(self, room_name: str) -> Dict[str, Any]:
        """
        Resolve class-role mapping for two-stage occupancy modeling.
        """
        result: Dict[str, Any] = {
            "available": False,
            "reason": "missing_label_encoder_classes",
            "classes": [],
            "excluded_class_ids": [],
            "occupied_class_ids": [],
            "primary_occupied_class_id": None,
        }
        label_encoder = self.platform.label_encoders.get(room_name)
        classes = getattr(label_encoder, "classes_", None) if label_encoder is not None else None
        if classes is None or len(classes) == 0:
            return result
        labels = [str(item).strip().lower() for item in classes]
        excluded_ids = [idx for idx, label in enumerate(labels) if label in {"unoccupied", "unknown"}]
        occupied_ids = [idx for idx in range(len(labels)) if idx not in set(excluded_ids)]
        if len(excluded_ids) == 0:
            result["reason"] = "missing_unoccupied_class"
            return result
        if len(occupied_ids) == 0:
            result["reason"] = "missing_occupied_class"
            return result
        result.update(
            {
                "available": True,
                "reason": "ok",
                "classes": labels,
                "excluded_class_ids": [int(v) for v in excluded_ids],
                "occupied_class_ids": [int(v) for v in occupied_ids],
                "primary_occupied_class_id": int(occupied_ids[0]),
            }
        )
        return result

    @staticmethod
    def _to_probability_matrix(raw: Any) -> np.ndarray:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"expected rank-2 probabilities, got shape={arr.shape}")
        row_sums = np.sum(arr, axis=1, keepdims=True)
        is_prob_like = (
            np.all(np.isfinite(arr))
            and np.min(arr) >= -1e-6
            and np.max(arr) <= 1.0 + 1e-6
            and np.all(np.isfinite(row_sums))
            and float(np.mean(np.abs(row_sums - 1.0))) <= 1e-2
        )
        if is_prob_like:
            return np.clip(arr, 0.0, 1.0)
        logits = arr.astype(np.float64, copy=False)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        denom = np.sum(exp, axis=1, keepdims=True)
        denom = np.clip(denom, 1e-12, None)
        probs = exp / denom
        return probs.astype(np.float32, copy=False)

    @staticmethod
    def _balanced_class_weight_dict(labels: np.ndarray) -> Optional[Dict[int, float]]:
        unique_classes = np.unique(labels)
        if len(unique_classes) <= 1:
            return None
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes.astype(np.int32, copy=False),
            y=labels.astype(np.int32, copy=False),
        )
        return {
            int(cls): float(weight)
            for cls, weight in zip(unique_classes.astype(np.int32, copy=False), class_weights)
        }

    def _train_two_stage_core_models(
        self,
        *,
        room_name: str,
        seq_length: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]],
        max_epochs: int,
    ) -> Dict[str, Any]:
        """
        Train occupancy-first two-stage models for core rooms.
        """
        result: Dict[str, Any] = {
            "enabled": False,
            "reason": "disabled",
            "gate_mode": self._resolve_two_stage_gate_mode(),
            "room": normalize_room_name(room_name),
            "stage_a_occupied_threshold": self._resolve_two_stage_stage_a_default_threshold(),
            "stage_a_threshold_source": "policy_default",
        }
        if not self._two_stage_core_enabled_for_room(room_name):
            return result

        class_profile = self._resolve_two_stage_class_profile(room_name)
        if not bool(class_profile.get("available", False)):
            result["reason"] = str(class_profile.get("reason") or "invalid_class_profile")
            result["class_profile"] = class_profile
            return result

        occupied_class_ids = [int(v) for v in class_profile.get("occupied_class_ids", [])]
        excluded_class_ids = [int(v) for v in class_profile.get("excluded_class_ids", [])]
        y_train_arr = np.asarray(y_train, dtype=np.int32)
        occupied_mask_train = np.isin(y_train_arr, occupied_class_ids)
        y_train_stage_a = occupied_mask_train.astype(np.int32)
        if len(np.unique(y_train_stage_a)) <= 1:
            result["reason"] = "insufficient_stage_a_class_diversity"
            result["class_profile"] = class_profile
            return result

        input_shape = (int(seq_length), int(len(self.platform.sensor_columns)))
        stage_a_model = build_transformer_model(
            input_shape=input_shape,
            num_classes=2,
            dropout_rate=DEFAULT_DROPOUT_RATE,
        )
        stage_a_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            jit_compile=False,
        )

        X_val = None
        y_val = None
        if validation_data is not None:
            X_val = np.asarray(validation_data[0], dtype=np.float32)
            y_val = np.asarray(validation_data[1], dtype=np.int32)
        y_val_stage_a = None
        validation_stage_a = None
        train_batch_shuffle_enabled = bool(self._should_shuffle_post_split_batches(room_name))
        if X_val is not None and y_val is not None and len(y_val) > 0:
            y_val_stage_a = np.isin(y_val, occupied_class_ids).astype(np.int32)
            validation_stage_a = (X_val, y_val_stage_a)

        stage_a_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if validation_stage_a is not None else "loss",
                patience=2,
                restore_best_weights=True,
                verbose=0,
            )
        ]
        stage_a_model.fit(
            np.asarray(X_train, dtype=np.float32),
            y_train_stage_a,
            epochs=max(1, int(max_epochs)),
            batch_size=32,
            validation_data=validation_stage_a,
            shuffle=bool(train_batch_shuffle_enabled),
            verbose=0,
            class_weight=self._balanced_class_weight_dict(y_train_stage_a),
            callbacks=stage_a_callbacks,
        )

        stage_b_model = None
        stage_b_enabled = False
        stage_b_reason = "single_occupied_class"
        occupied_label_to_stage_idx = {int(class_id): idx for idx, class_id in enumerate(occupied_class_ids)}
        primary_occupied_class_id = int(class_profile.get("primary_occupied_class_id", occupied_class_ids[0]))

        if len(occupied_class_ids) >= 2 and int(np.sum(occupied_mask_train)) > 0:
            X_train_stage_b = np.asarray(X_train[occupied_mask_train], dtype=np.float32)
            y_train_stage_b = np.asarray(
                [occupied_label_to_stage_idx[int(item)] for item in y_train_arr[occupied_mask_train]],
                dtype=np.int32,
            )
            if len(np.unique(y_train_stage_b)) > 1:
                stage_b_model = build_transformer_model(
                    input_shape=input_shape,
                    num_classes=len(occupied_class_ids),
                    dropout_rate=DEFAULT_DROPOUT_RATE,
                )
                stage_b_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                    jit_compile=False,
                )
                validation_stage_b = None
                if X_val is not None and y_val is not None and len(y_val) > 0:
                    occupied_mask_val = np.isin(y_val, occupied_class_ids)
                    if int(np.sum(occupied_mask_val)) > 0:
                        y_val_stage_b = np.asarray(
                            [occupied_label_to_stage_idx[int(item)] for item in y_val[occupied_mask_val]],
                            dtype=np.int32,
                        )
                        validation_stage_b = (
                            np.asarray(X_val[occupied_mask_val], dtype=np.float32),
                            y_val_stage_b,
                        )
                stage_b_callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss" if validation_stage_b is not None else "loss",
                        patience=2,
                        restore_best_weights=True,
                        verbose=0,
                    )
                ]
                stage_b_model.fit(
                    X_train_stage_b,
                    y_train_stage_b,
                    epochs=max(1, int(max_epochs)),
                    batch_size=32,
                    validation_data=validation_stage_b,
                    shuffle=bool(train_batch_shuffle_enabled),
                    verbose=0,
                    class_weight=self._balanced_class_weight_dict(y_train_stage_b),
                    callbacks=stage_b_callbacks,
                )
                stage_b_enabled = True
                stage_b_reason = "trained"
                occupied_counts = {
                    int(class_id): int(np.sum(y_train_arr == int(class_id)))
                    for class_id in occupied_class_ids
                }
                primary_occupied_class_id = max(
                    occupied_counts.keys(),
                    key=lambda key: occupied_counts[key],
                )
            else:
                stage_b_reason = "insufficient_stage_b_class_diversity"

        result.update(
            {
                "enabled": True,
                "reason": "ok",
                "class_profile": class_profile,
                "excluded_class_ids": excluded_class_ids,
                "occupied_class_ids": occupied_class_ids,
                "primary_occupied_class_id": int(primary_occupied_class_id),
                "num_classes": int(len(class_profile.get("classes", []))),
                "stage_a_model": stage_a_model,
                "stage_b_model": stage_b_model,
                "stage_b_enabled": bool(stage_b_enabled),
                "stage_b_reason": str(stage_b_reason),
            }
        )
        return result

    def _calibrate_two_stage_stage_a_threshold(
        self,
        *,
        room_name: str,
        two_stage_result: Mapping[str, Any],
        calibration_data: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Any]:
        """
        Calibrate occupancy threshold for stage-A of two-stage core modeling.

        Uses holdout/calibration probabilities for occupied-vs-unoccupied labels.
        """
        default_thr = self._resolve_two_stage_stage_a_default_threshold()
        result: Dict[str, Any] = {
            "threshold": float(default_thr),
            "status": "fallback_default",
            "source": "policy_default",
            "support_occupied": 0,
            "samples": 0,
            "target_precision": None,
            "recall_floor": None,
            "true_occupied_rate": None,
            "predicted_occupied_rate": None,
            "min_predicted_occupied_rate": None,
        }
        if not bool(two_stage_result.get("enabled", False)):
            result["status"] = "disabled"
            return result
        if calibration_data is None:
            result["status"] = "missing_calibration_data"
            return result

        X_calib, y_calib = calibration_data
        if X_calib is None or y_calib is None or len(X_calib) == 0 or len(y_calib) == 0:
            result["status"] = "empty_calibration_data"
            return result

        stage_a_model = two_stage_result.get("stage_a_model")
        if stage_a_model is None:
            result["status"] = "missing_stage_a_model"
            return result

        occupied_ids = [int(v) for v in (two_stage_result.get("occupied_class_ids") or [])]
        if not occupied_ids:
            result["status"] = "invalid_class_profile"
            return result

        y_calib_arr = np.asarray(y_calib, dtype=np.int32)
        y_occ_true = np.isin(y_calib_arr, occupied_ids).astype(np.int32)
        support = int(np.sum(y_occ_true))
        result["support_occupied"] = support
        result["samples"] = int(len(y_occ_true))
        true_occupied_rate = float(np.mean(y_occ_true))
        result["true_occupied_rate"] = true_occupied_rate
        if len(np.unique(y_occ_true)) <= 1:
            result["status"] = "insufficient_stage_a_class_diversity"
            return result

        min_support = int(max(10, self._active_policy().calibration.min_support_per_class))
        if support < min_support:
            result["status"] = f"fallback_low_support:{support}<{min_support}"
            return result

        try:
            raw_stage_a = stage_a_model.predict(np.asarray(X_calib, dtype=np.float32), verbose=0)
            if isinstance(raw_stage_a, dict):
                raw_stage_a = raw_stage_a.get("activity_logits")
            stage_a_probs = self._to_probability_matrix(raw_stage_a)
            if stage_a_probs.shape[1] < 2:
                result["status"] = f"invalid_stage_a_width:{stage_a_probs.shape}"
                return result
            p_occ = np.asarray(stage_a_probs[:, 1], dtype=np.float32)
        except Exception as e:
            result["status"] = f"prediction_error:{type(e).__name__}"
            result["error"] = str(e)
            return result

        precision, recall, thresholds = precision_recall_curve(y_occ_true, p_occ)
        if len(thresholds) == 0:
            result["status"] = "fallback_no_curve"
            return result

        calibration_policy = self._active_policy().calibration
        two_stage_policy = self._active_policy().two_stage_core
        target_precision = float(
            np.clip(
                float(two_stage_policy.stage_a_target_precision),
                0.0,
                1.0,
            )
        )
        recall_floor = float(
            np.clip(
                float(two_stage_policy.stage_a_recall_floor),
                0.0,
                1.0,
            )
        )
        if not np.isfinite(target_precision):
            target_precision = float(calibration_policy.default_precision_target)
        if not np.isfinite(recall_floor):
            recall_floor = 0.20
        result["target_precision"] = target_precision
        result["recall_floor"] = recall_floor

        p = precision[:-1]
        r = recall[:-1]
        valid = np.where((p >= target_precision) & (r >= recall_floor))[0]
        threshold_min, threshold_max = self._resolve_two_stage_stage_a_threshold_bounds()
        if len(valid) > 0:
            raw_threshold = float(np.min(thresholds[valid]))
            base_status = "target_met"
        else:
            # If precision target is unattainable, keep recall-floor first.
            valid_recall = np.where(r >= recall_floor)[0]
            if len(valid_recall) > 0:
                raw_threshold = float(np.min(thresholds[valid_recall]))
                base_status = "fallback_recall_floor"
            else:
                f1 = (2.0 * p * r) / np.maximum(p + r, 1e-9)
                best_idx = int(np.nanargmax(f1))
                raw_threshold = float(thresholds[best_idx])
                base_status = "fallback_best_f1"

        final_threshold = float(np.clip(raw_threshold, threshold_min, threshold_max))

        # Guardrail: prevent stage-A threshold from collapsing to near-zero occupied
        # predictions on calibration when true occupied support is substantial.
        min_ratio, min_abs_floor = self._resolve_two_stage_stage_a_min_predicted_occupied_floor()
        min_pred_rate = float(np.clip(max(min_abs_floor, true_occupied_rate * min_ratio), 0.0, 1.0))
        predicted_occ_rate = float(np.mean(p_occ >= final_threshold))
        adjusted_for_pred_rate = False
        if min_pred_rate > 0.0 and predicted_occ_rate + 1e-9 < min_pred_rate:
            quantile_level = float(np.clip(1.0 - min_pred_rate, 0.0, 1.0))
            candidate_threshold = float(np.quantile(p_occ, quantile_level))
            adjusted_threshold = float(
                np.clip(min(final_threshold, candidate_threshold), threshold_min, threshold_max)
            )
            if adjusted_threshold + 1e-9 < final_threshold:
                final_threshold = adjusted_threshold
                predicted_occ_rate = float(np.mean(p_occ >= final_threshold))
                adjusted_for_pred_rate = True

        status = str(base_status)
        if adjusted_for_pred_rate:
            status = f"{base_status}+pred_occ_floor"

        result.update(
            {
                "threshold": final_threshold,
                "status": status,
                "source": "calibration",
                "raw_threshold": raw_threshold,
                "threshold_bounds": {
                    "min": float(threshold_min),
                    "max": float(threshold_max),
                },
                "predicted_occupied_rate": predicted_occ_rate,
                "min_predicted_occupied_rate": min_pred_rate,
                "predicted_occupied_floor_adjusted": bool(adjusted_for_pred_rate),
            }
        )
        logger.info(
            "Two-stage stage-A occupancy threshold for %s: %.4f (%s, support=%d)",
            room_name,
            final_threshold,
            status,
            support,
        )
        return result

    def _predict_two_stage_core_probabilities(
        self,
        *,
        two_stage_result: Mapping[str, Any],
        X: np.ndarray,
    ) -> np.ndarray:
        stage_a_model = two_stage_result.get("stage_a_model")
        if stage_a_model is None:
            raise ValueError("missing_stage_a_model")
        raw_stage_a = stage_a_model.predict(X, verbose=0)
        if isinstance(raw_stage_a, dict):
            raw_stage_a = raw_stage_a.get("activity_logits")
        stage_a_probs = self._to_probability_matrix(raw_stage_a)
        if stage_a_probs.shape[1] < 2:
            raise ValueError(f"invalid_stage_a_width:{stage_a_probs.shape}")

        n_samples = int(stage_a_probs.shape[0])
        n_classes = int(two_stage_result.get("num_classes", 0) or 0)
        if n_classes <= 1:
            raise ValueError(f"invalid_two_stage_num_classes:{n_classes}")

        excluded_ids = [int(v) for v in (two_stage_result.get("excluded_class_ids") or []) if int(v) < n_classes]
        occupied_ids = [int(v) for v in (two_stage_result.get("occupied_class_ids") or []) if int(v) < n_classes]
        if not excluded_ids or not occupied_ids:
            raise ValueError("invalid_two_stage_class_mapping")

        raw_stage_a_threshold = two_stage_result.get(
            "stage_a_occupied_threshold",
            self._resolve_two_stage_stage_a_default_threshold(),
        )
        try:
            parsed_stage_a_threshold = float(raw_stage_a_threshold)
        except (TypeError, ValueError):
            parsed_stage_a_threshold = self._resolve_two_stage_stage_a_default_threshold()
        stage_a_threshold = float(np.clip(parsed_stage_a_threshold, 0.0, 1.0))
        p_occ = stage_a_probs[:, 1]
        p_unocc = stage_a_probs[:, 0]
        occupied_mask = p_occ >= stage_a_threshold
        out = np.zeros(shape=(n_samples, n_classes), dtype=np.float32)
        non_occ_indices = np.where(~occupied_mask)[0]
        if non_occ_indices.size > 0:
            out[np.ix_(non_occ_indices, np.asarray(excluded_ids, dtype=np.int32))] = (
                1.0 / float(len(excluded_ids))
            )

        if np.any(occupied_mask):
            occ_indices = np.where(occupied_mask)[0]
            p_occ_active = p_occ[occupied_mask]
            p_unocc_active = p_unocc[occupied_mask]
            out[np.ix_(occ_indices, np.asarray(excluded_ids, dtype=np.int32))] = (
                (p_unocc_active / float(len(excluded_ids)))[:, None]
            )

            stage_b_model = two_stage_result.get("stage_b_model")
            if stage_b_model is not None:
                raw_stage_b = stage_b_model.predict(X, verbose=0)
                if isinstance(raw_stage_b, dict):
                    raw_stage_b = raw_stage_b.get("activity_logits")
                stage_b_probs = self._to_probability_matrix(raw_stage_b)
                if stage_b_probs.shape[1] != len(occupied_ids):
                    raise ValueError(
                        f"invalid_stage_b_width:{stage_b_probs.shape[1]}!=expected:{len(occupied_ids)}"
                    )
                stage_b_probs = stage_b_probs[occupied_mask]
                for idx, class_id in enumerate(occupied_ids):
                    out[occ_indices, int(class_id)] = p_occ_active * stage_b_probs[:, idx]
            else:
                primary_occupied_class_id = int(
                    two_stage_result.get("primary_occupied_class_id", occupied_ids[0])
                )
                if primary_occupied_class_id not in occupied_ids:
                    primary_occupied_class_id = occupied_ids[0]
                out[occ_indices, int(primary_occupied_class_id)] = p_occ_active

        row_sums = np.sum(out, axis=1, keepdims=True)
        invalid_rows = row_sums[:, 0] <= 1e-8
        if np.any(invalid_rows):
            out[invalid_rows, :] = 0.0
            out[invalid_rows, excluded_ids[0]] = 1.0
            row_sums = np.sum(out, axis=1, keepdims=True)
        out = out / np.clip(row_sums, 1e-8, None)
        return out.astype(np.float32, copy=False)

    def _write_two_stage_core_artifacts(
        self,
        *,
        elder_id: str,
        room_name: str,
        saved_version: int,
        two_stage_result: Mapping[str, Any],
        promote_to_latest: bool,
    ) -> Dict[str, str]:
        """
        Persist two-stage core-room artifacts as versioned files and latest aliases.
        """
        if not bool(two_stage_result.get("enabled", False)):
            return {}
        stage_a_model = two_stage_result.get("stage_a_model")
        if stage_a_model is None:
            return {}

        models_dir = self.registry.get_models_dir(elder_id)
        stage_a_versioned = models_dir / f"{room_name}_v{int(saved_version)}_two_stage_stage_a_model.keras"
        stage_b_versioned = models_dir / f"{room_name}_v{int(saved_version)}_two_stage_stage_b_model.keras"
        meta_versioned = models_dir / f"{room_name}_v{int(saved_version)}_two_stage_meta.json"

        stage_a_model.save(str(stage_a_versioned))
        stage_b_model = two_stage_result.get("stage_b_model")
        stage_b_enabled = bool(two_stage_result.get("stage_b_enabled", False) and stage_b_model is not None)
        if stage_b_enabled:
            stage_b_model.save(str(stage_b_versioned))

        meta_payload = {
            "schema_version": "beta6.two_stage_core.v1",
            "created_at_utc": _utc_now_iso_z(),
            "elder_id": str(elder_id),
            "room": str(room_name),
            "saved_version": int(saved_version),
            "gate_mode": str(two_stage_result.get("gate_mode") or self._resolve_two_stage_gate_mode()),
            "runtime_enabled": bool(two_stage_result.get("runtime_enabled", True)),
            "runtime_gate_source": str(two_stage_result.get("runtime_gate_source") or ""),
            "stage_b_enabled": bool(stage_b_enabled),
            "stage_b_reason": str(two_stage_result.get("stage_b_reason") or ""),
            "stage_a_occupied_threshold": float(
                two_stage_result.get(
                    "stage_a_occupied_threshold",
                    self._resolve_two_stage_stage_a_default_threshold(),
                )
            ),
            "stage_a_threshold_source": str(two_stage_result.get("stage_a_threshold_source") or "policy_default"),
            "stage_a_calibration": dict(two_stage_result.get("stage_a_calibration") or {}),
            "num_classes": int(two_stage_result.get("num_classes", 0) or 0),
            "classes": list((two_stage_result.get("class_profile") or {}).get("classes", [])),
            "excluded_class_ids": [int(v) for v in (two_stage_result.get("excluded_class_ids") or [])],
            "occupied_class_ids": [int(v) for v in (two_stage_result.get("occupied_class_ids") or [])],
            "primary_occupied_class_id": int(
                two_stage_result.get("primary_occupied_class_id", -1) or -1
            ),
        }
        meta_versioned.write_text(json.dumps(meta_payload, indent=2))

        artifact_paths = {
            "stage_a_model_versioned": str(stage_a_versioned),
            "meta_versioned": str(meta_versioned),
        }
        if stage_b_enabled:
            artifact_paths["stage_b_model_versioned"] = str(stage_b_versioned)

        if promote_to_latest:
            stage_a_latest = models_dir / f"{room_name}_two_stage_stage_a_model.keras"
            stage_b_latest = models_dir / f"{room_name}_two_stage_stage_b_model.keras"
            meta_latest = models_dir / f"{room_name}_two_stage_meta.json"
            shutil.copy2(stage_a_versioned, stage_a_latest)
            if stage_b_enabled:
                shutil.copy2(stage_b_versioned, stage_b_latest)
            else:
                try:
                    stage_b_latest.unlink(missing_ok=True)
                except Exception:
                    pass
            meta_latest.write_text(json.dumps(meta_payload, indent=2))
            artifact_paths.update(
                {
                    "stage_a_model_latest": str(stage_a_latest),
                    "meta_latest": str(meta_latest),
                }
            )
            if stage_b_enabled:
                artifact_paths["stage_b_model_latest"] = str(stage_b_latest)
        return artifact_paths

    @staticmethod
    def _build_class_prefix_counts(y_seq: np.ndarray, class_ids: List[int]) -> Dict[int, np.ndarray]:
        """
        Build cumulative counts per class for O(1) range support queries.
        """
        prefixes: Dict[int, np.ndarray] = {}
        for class_id in sorted(set(int(v) for v in class_ids)):
            mask = (y_seq == int(class_id)).astype(np.int32, copy=False)
            prefix = np.zeros(len(y_seq) + 1, dtype=np.int32)
            prefix[1:] = np.cumsum(mask, dtype=np.int32)
            prefixes[int(class_id)] = prefix
        return prefixes

    def _select_temporal_split_index_with_support(
        self,
        y_seq: np.ndarray,
        default_split_idx: int,
        required_class_ids: List[int],
        min_support: int = 1,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Choose split index nearest default that preserves critical-label support in holdout.
        """
        n = int(len(y_seq))
        default_split_idx = max(1, min(n - 1, int(default_split_idx)))
        debug: Dict[str, Any] = {
            "default_split_idx": int(default_split_idx),
            "selected_split_idx": int(default_split_idx),
            "found": True,
            "required_class_ids": sorted(set(int(v) for v in required_class_ids)),
            "min_support": int(max(1, min_support)),
            "search_radius": 0,
            "drift_optimization_enabled": True,
            "drift_distance_penalty": 0.0,
            "drift_max_shift_fraction": 0.0,
            "min_holdout_samples": 0,
            "min_train_samples": 0,
        }
        required = sorted(set(int(v) for v in required_class_ids))
        if n < 2 or len(required) == 0:
            return default_split_idx, debug

        min_support = int(max(1, min_support))
        prefixes = self._build_class_prefix_counts(y_seq, required)
        totals = {class_id: int(prefixes[class_id][-1]) for class_id in required}

        candidates: List[int] = []
        max_radius = max(default_split_idx - 1, n - default_split_idx - 1)
        for radius in range(0, max_radius + 1):
            left = default_split_idx - radius
            right = default_split_idx + radius
            if left >= 1:
                candidates.append(int(left))
            if radius > 0 and right <= n - 1:
                candidates.append(int(right))

        def _support_ok(split_idx: int) -> bool:
            ok = True
            for class_id in required:
                train_count = int(prefixes[class_id][split_idx])
                holdout_count = int(totals[class_id] - train_count)
                # Require holdout support for validation/calibration evidence.
                if holdout_count < min_support:
                    ok = False
                    break
                # When possible, also keep at least minimal train support.
                if totals[class_id] >= (2 * min_support) and train_count < min_support:
                    ok = False
                    break
            return bool(ok)

        def _holdout_support(split_idx: int) -> Dict[str, int]:
            return {
                str(class_id): int(totals[class_id] - int(prefixes[class_id][split_idx]))
                for class_id in required
            }

        def _drift_summary(split_idx: int) -> Dict[str, Any]:
            train_total = float(max(1, split_idx))
            holdout_total = float(max(1, n - split_idx))
            max_abs = -1.0
            max_class = None
            class_drift: Dict[str, Dict[str, float]] = {}
            for class_id in required:
                train_count = int(prefixes[class_id][split_idx])
                holdout_count = int(totals[class_id] - train_count)
                train_share = float(train_count) / train_total
                holdout_share = float(holdout_count) / holdout_total
                drift = float(train_share - holdout_share)
                class_drift[str(class_id)] = {
                    "train_share": float(train_share),
                    "holdout_share": float(holdout_share),
                    "drift": float(drift),
                    "drift_pp": float(drift * 100.0),
                }
                abs_drift = abs(drift)
                if abs_drift > max_abs:
                    max_abs = float(abs_drift)
                    max_class = int(class_id)
            return {
                "max_abs_drift": float(max_abs if max_abs >= 0.0 else 0.0),
                "max_drift_class": max_class,
                "class_drift": class_drift,
            }

        fallback_split_idx: Optional[int] = None
        fallback_radius: int = 0
        for radius, split_idx in enumerate(candidates):
            if not _support_ok(split_idx):
                continue
            fallback_split_idx = int(split_idx)
            fallback_radius = int(radius)
            break

        if fallback_split_idx is None:
            debug["found"] = False
            debug["holdout_support_by_class"] = _holdout_support(default_split_idx)
            return int(default_split_idx), debug

        optimize_drift = str(os.getenv("TEMPORAL_SPLIT_OPTIMIZE_DRIFT", "true")).strip().lower() in {
            "1", "true", "yes", "on", "enabled",
        }
        max_shift_fraction = float(self._read_float_env("TEMPORAL_SPLIT_MAX_SHIFT_FRACTION", 0.12))
        max_shift_fraction = float(min(max(max_shift_fraction, 0.0), 0.40))
        max_shift_samples = int(max(0, np.floor(float(n) * max_shift_fraction)))
        drift_distance_penalty = float(self._read_float_env("TEMPORAL_SPLIT_DRIFT_DISTANCE_PENALTY", 0.50))
        drift_distance_penalty = float(max(0.0, drift_distance_penalty))
        min_holdout_fraction = float(self._read_float_env("TEMPORAL_SPLIT_MIN_HOLDOUT_FRACTION", 0.10))
        min_holdout_fraction = float(min(max(min_holdout_fraction, 0.02), 0.45))
        min_train_fraction = float(self._read_float_env("TEMPORAL_SPLIT_MIN_TRAIN_FRACTION", 0.60))
        min_train_fraction = float(min(max(min_train_fraction, 0.30), 0.90))
        min_holdout_samples = int(max(min_support, int(np.ceil(float(n) * min_holdout_fraction))))
        min_train_samples = int(max(min_support, int(np.ceil(float(n) * min_train_fraction))))

        debug["drift_optimization_enabled"] = bool(optimize_drift)
        debug["drift_distance_penalty"] = float(drift_distance_penalty)
        debug["drift_max_shift_fraction"] = float(max_shift_fraction)
        debug["min_holdout_samples"] = int(min_holdout_samples)
        debug["min_train_samples"] = int(min_train_samples)

        if not optimize_drift:
            debug["selected_split_idx"] = int(fallback_split_idx)
            debug["search_radius"] = int(fallback_radius)
            debug["holdout_support_by_class"] = _holdout_support(fallback_split_idx)
            return int(fallback_split_idx), debug

        scored: List[Tuple[float, float, float, int, int, Dict[str, Any]]] = []
        for radius, split_idx in enumerate(candidates):
            if not _support_ok(split_idx):
                continue
            if abs(int(split_idx) - int(default_split_idx)) > int(max_shift_samples):
                continue
            if int(split_idx) < int(min_train_samples):
                continue
            holdout_size = int(n - int(split_idx))
            if holdout_size < int(min_holdout_samples):
                continue

            drift = _drift_summary(int(split_idx))
            distance_ratio = float(abs(int(split_idx) - int(default_split_idx))) / float(max(1, n))
            score = float(drift["max_abs_drift"]) + (float(drift_distance_penalty) * distance_ratio)
            scored.append((score, float(drift["max_abs_drift"]), distance_ratio, int(radius), int(split_idx), drift))

        if scored:
            scored.sort(key=lambda item: (item[0], item[2], item[3], item[4]))
            _, _, _, radius, split_idx, drift = scored[0]
            debug["selected_split_idx"] = int(split_idx)
            debug["search_radius"] = int(radius)
            debug["holdout_support_by_class"] = _holdout_support(split_idx)
            debug["drift_objective"] = {
                "score": float(scored[0][0]),
                "max_abs_drift": float(drift["max_abs_drift"]),
                "max_drift_class": drift["max_drift_class"],
                "distance_ratio": float(scored[0][2]),
            }
            return int(split_idx), debug

        debug["drift_optimization_fallback"] = "no_candidate_met_drift_constraints"
        debug["selected_split_idx"] = int(fallback_split_idx)
        debug["search_radius"] = int(fallback_radius)
        debug["holdout_support_by_class"] = _holdout_support(fallback_split_idx)
        return int(fallback_split_idx), debug

    def _select_calibration_size_with_support(
        self,
        y_holdout: np.ndarray,
        default_calib_size: int,
        min_calib_samples: int,
        min_val_samples: int,
        required_class_ids: List[int],
        min_support: int = 1,
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        """
        Choose calibration tail size nearest default with critical-label support in val/calib.
        """
        n = int(len(y_holdout))
        min_calib_samples = int(max(1, min_calib_samples))
        min_val_samples = int(max(1, min_val_samples))
        min_support = int(max(1, min_support))
        required = sorted(set(int(v) for v in required_class_ids))

        min_size = min_calib_samples
        max_size = n - min_val_samples
        debug: Dict[str, Any] = {
            "default_calib_size": int(default_calib_size),
            "selected_calib_size": None,
            "found": False,
            "required_class_ids": required,
            "min_support": min_support,
            "reason": None,
        }
        if n <= 1 or min_size > max_size:
            debug["reason"] = "insufficient_holdout_for_separate_calib"
            return None, debug

        desired = int(max(min_size, min(max_size, int(default_calib_size))))
        if len(required) == 0:
            debug["selected_calib_size"] = desired
            debug["found"] = True
            return desired, debug

        prefixes = self._build_class_prefix_counts(y_holdout, required)
        totals = {class_id: int(prefixes[class_id][-1]) for class_id in required}
        # If a critical class occurs only once in holdout, separate val/calib cannot both contain it.
        unsplittable = [class_id for class_id, total in totals.items() if total < (2 * min_support)]
        if unsplittable:
            debug["reason"] = f"unsplittable_critical_classes:{unsplittable}"
            return None, debug

        candidates: List[int] = []
        max_radius = max(desired - min_size, max_size - desired)
        for radius in range(0, max_radius + 1):
            lower = desired - radius
            upper = desired + radius
            if lower >= min_size:
                candidates.append(int(lower))
            if radius > 0 and upper <= max_size:
                candidates.append(int(upper))

        for calib_size in candidates:
            split_idx = n - int(calib_size)
            ok = True
            for class_id in required:
                val_count = int(prefixes[class_id][split_idx])
                calib_count = int(totals[class_id] - val_count)
                if val_count < min_support or calib_count < min_support:
                    ok = False
                    break
            if not ok:
                continue
            debug["selected_calib_size"] = int(calib_size)
            debug["found"] = True
            debug["val_support_by_class"] = {
                str(class_id): int(prefixes[class_id][split_idx]) for class_id in required
            }
            debug["calib_support_by_class"] = {
                str(class_id): int(totals[class_id] - int(prefixes[class_id][split_idx]))
                for class_id in required
            }
            return int(calib_size), debug

        debug["reason"] = "no_supported_calibration_size"
        return None, debug

    def _apply_minority_class_sampling(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        room_name: str,
        reference_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Oversample minority classes toward a target share with capped multiplier.
        """
        stats: Dict[str, Any] = {
            "enabled": False,
            "target_share": None,
            "max_multiplier": None,
            "added_samples": 0,
            "prior_drift_guard_applied": False,
            "prior_drift_reference": "train",
            "post_sampling_prior_drift": {"available": False},
            "class_counts_before": {},
            "class_counts_after": {},
        }
        if len(y_train) == 0:
            return X_train, y_train, stats

        classes, counts = np.unique(y_train, return_counts=True)
        class_counts_before = {int(cls): int(cnt) for cls, cnt in zip(classes, counts)}
        stats["class_counts_before"] = class_counts_before
        stats["class_counts_after"] = dict(class_counts_before)
        drift_reference_labels, drift_reference = self._resolve_prior_drift_reference_labels(
            reference_labels,
            y_train,
        )
        stats["prior_drift_reference"] = str(drift_reference)

        cfg = self._resolve_minority_sampling_config(room_name)
        stats["enabled"] = bool(cfg["enabled"])
        stats["target_share"] = float(cfg["target_share"])
        stats["max_multiplier"] = int(cfg["max_multiplier"])
        prior_drift_guard_enabled = bool(cfg.get("prior_drift_guard_enabled", False))
        max_post_sampling_prior_drift = float(cfg.get("max_post_sampling_prior_drift", 1.0) or 1.0)
        if not bool(cfg["enabled"]):
            return X_train, y_train, stats

        total = int(len(y_train))
        target_share = float(cfg["target_share"])
        target_count = max(1, int(np.ceil(total * target_share)))
        max_multiplier = int(cfg["max_multiplier"])
        if target_count <= 1 or len(classes) <= 1:
            stats["class_counts_after"] = class_counts_before
            return X_train, y_train, stats

        rng = np.random.default_rng(42)
        sampled_idx_parts_by_class: Dict[int, np.ndarray] = {}
        added_total = 0

        for cls, count in zip(classes, counts):
            cls = int(cls)
            count = int(count)
            if count <= 0 or count >= target_count:
                continue
            cls_positions = np.where(y_train == cls)[0]
            if len(cls_positions) == 0:
                continue
            max_extra = max(0, count * (max_multiplier - 1))
            needed = target_count - count
            add_count = min(max_extra, needed)
            if add_count <= 0:
                continue
            sampled_idx_parts_by_class[cls] = rng.choice(cls_positions, size=add_count, replace=True)
            added_total += int(add_count)

        if added_total == 0:
            stats["post_sampling_prior_drift"] = self._compute_class_share_drift_from_labels(
                drift_reference_labels,
                y_train,
            )
            stats["class_counts_after"] = class_counts_before
            return X_train, y_train, stats

        if prior_drift_guard_enabled and max_post_sampling_prior_drift < 1.0:
            while True:
                sampled_idx_parts = [
                    values
                    for values in sampled_idx_parts_by_class.values()
                    if len(values) > 0
                ]
                if not sampled_idx_parts:
                    break
                candidate_idx = np.concatenate(sampled_idx_parts).astype(np.int64, copy=False)
                candidate_labels = np.concatenate([y_train, y_train[candidate_idx]], axis=0)
                candidate_drift = self._compute_class_share_drift_from_labels(
                    drift_reference_labels,
                    candidate_labels,
                )
                max_abs_drift = float(candidate_drift.get("max_abs_drift") or 0.0)
                if max_abs_drift <= max_post_sampling_prior_drift + 1e-12:
                    stats["post_sampling_prior_drift"] = candidate_drift
                    break
                drift_entries = candidate_drift.get("class_drift") or {}
                trimmed = False
                for class_id, payload in sorted(
                    drift_entries.items(),
                    key=lambda item: (
                        -abs(float((item[1] or {}).get("drift", 0.0) or 0.0)),
                        int(item[0]) if str(item[0]).isdigit() else 10**9,
                    ),
                ):
                    try:
                        class_key = int(class_id)
                    except (TypeError, ValueError):
                        continue
                    if float((payload or {}).get("drift", 0.0) or 0.0) <= 0.0:
                        continue
                    existing = sampled_idx_parts_by_class.get(class_key)
                    if existing is None or len(existing) == 0:
                        continue
                    sampled_idx_parts_by_class[class_key] = existing[:-1]
                    trimmed = True
                    stats["prior_drift_guard_applied"] = True
                    break
                if not trimmed:
                    stats["post_sampling_prior_drift"] = candidate_drift
                    break
        if not isinstance(stats.get("post_sampling_prior_drift"), Mapping) or not bool(
            (stats.get("post_sampling_prior_drift") or {}).get("available")
        ):
            sampled_idx_parts = [
                values for values in sampled_idx_parts_by_class.values() if len(values) > 0
            ]
            candidate_idx = (
                np.concatenate(sampled_idx_parts).astype(np.int64, copy=False)
                if sampled_idx_parts
                else np.asarray([], dtype=np.int64)
            )
            candidate_labels = (
                np.concatenate([y_train, y_train[candidate_idx]], axis=0)
                if len(candidate_idx) > 0
                else y_train
            )
            stats["post_sampling_prior_drift"] = self._compute_class_share_drift_from_labels(
                drift_reference_labels,
                candidate_labels,
            )

        sampled_idx_parts = [values for values in sampled_idx_parts_by_class.values() if len(values) > 0]
        if not sampled_idx_parts:
            stats["added_samples"] = 0
            stats["class_counts_after"] = class_counts_before
            return X_train, y_train, stats

        sampled_idx = np.concatenate(sampled_idx_parts).astype(np.int64, copy=False)
        X_added = X_train[sampled_idx]
        y_added = y_train[sampled_idx]
        X_balanced = np.concatenate([X_train, X_added], axis=0)
        y_balanced = np.concatenate([y_train, y_added], axis=0)

        classes_after, counts_after = np.unique(y_balanced, return_counts=True)
        stats["added_samples"] = int(len(sampled_idx))
        stats["class_counts_after"] = {int(cls): int(cnt) for cls, cnt in zip(classes_after, counts_after)}

        logger.info(
            f"Minority sampling for {room_name}: added={int(len(sampled_idx))}, "
            f"target_share={target_share:.2f}, max_multiplier={max_multiplier}, "
            f"prior_drift_guard_applied={bool(stats.get('prior_drift_guard_applied', False))}, "
            f"before={class_counts_before}, after={stats['class_counts_after']}"
        )
        return X_balanced, y_balanced, stats

    def _apply_transition_focus_sampling(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        room_name: str,
        reference_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        stats: Dict[str, Any] = {
            "enabled": False,
            "focus_label": None,
            "focus_class_id": None,
            "radius_steps": 0,
            "max_multiplier": 1,
            "added_samples": 0,
            "prior_drift_guard_applied": False,
            "prior_drift_reference": "train",
            "post_sampling_prior_drift": {"available": False},
            "transition_window_count": 0,
            "class_counts_before": self._build_class_support_map(y_train),
            "class_counts_after": self._build_class_support_map(y_train),
            "reason": "disabled",
        }
        if len(y_train) == 0:
            stats["reason"] = "empty_training_labels"
            return X_train, y_train, stats

        cfg = self._resolve_transition_focus_sampling_config(room_name)
        focus_label = str(cfg.get("focus_label", "")).strip().lower()
        radius_steps = int(cfg.get("radius_steps", 0) or 0)
        max_multiplier = int(cfg.get("max_multiplier", 1) or 1)
        prior_drift_guard_enabled = bool(cfg.get("prior_drift_guard_enabled", False))
        max_post_sampling_prior_drift = float(cfg.get("max_post_sampling_prior_drift", 1.0) or 1.0)
        stats["focus_label"] = focus_label or None
        stats["radius_steps"] = max(0, radius_steps)
        stats["max_multiplier"] = max(1, max_multiplier)
        drift_reference_labels, drift_reference = self._resolve_prior_drift_reference_labels(
            reference_labels,
            y_train,
        )
        stats["prior_drift_reference"] = str(drift_reference)
        if not bool(cfg.get("enabled", False)):
            stats["reason"] = "room_not_configured"
            return X_train, y_train, stats

        label_encoder = self.platform.label_encoders.get(room_name)
        classes = getattr(label_encoder, "classes_", None) if label_encoder is not None else None
        if classes is None or len(classes) == 0:
            stats["reason"] = "missing_label_encoder"
            return X_train, y_train, stats

        label_to_class_id = {
            str(label).strip().lower(): int(class_id)
            for class_id, label in enumerate(classes)
        }
        focus_class_id = label_to_class_id.get(focus_label)
        if focus_class_id is None:
            stats["reason"] = "focus_label_missing"
            return X_train, y_train, stats
        stats["focus_class_id"] = int(focus_class_id)

        focus_positions = np.where(np.asarray(y_train, dtype=np.int32) == int(focus_class_id))[0]
        if len(focus_positions) == 0:
            stats["reason"] = "focus_label_absent"
            return X_train, y_train, stats

        transition_windows: List[np.ndarray] = []
        start = int(focus_positions[0])
        end = int(focus_positions[0])
        for idx in focus_positions[1:]:
            idx = int(idx)
            if idx == end + 1:
                end = idx
                continue
            window_start = max(0, start - radius_steps)
            window_end = min(len(y_train), end + radius_steps + 1)
            transition_windows.append(np.arange(window_start, window_end, dtype=np.int64))
            start = idx
            end = idx
        window_start = max(0, start - radius_steps)
        window_end = min(len(y_train), end + radius_steps + 1)
        transition_windows.append(np.arange(window_start, window_end, dtype=np.int64))

        unique_indices = np.unique(np.concatenate(transition_windows).astype(np.int64, copy=False))
        stats["transition_window_count"] = int(len(unique_indices))
        if len(unique_indices) == 0 or max_multiplier <= 1:
            stats["reason"] = "no_transition_windows"
            return X_train, y_train, stats

        added_idx = np.tile(unique_indices, max_multiplier - 1).astype(np.int64, copy=False)
        if len(added_idx) == 0:
            stats["reason"] = "no_added_samples"
            return X_train, y_train, stats

        if prior_drift_guard_enabled and max_post_sampling_prior_drift < 1.0:
            while True:
                if len(added_idx) == 0:
                    break
                candidate_labels = np.concatenate([y_train, y_train[added_idx]], axis=0)
                candidate_drift = self._compute_class_share_drift_from_labels(
                    drift_reference_labels,
                    candidate_labels,
                )
                max_abs_drift = float(candidate_drift.get("max_abs_drift") or 0.0)
                if max_abs_drift <= max_post_sampling_prior_drift + 1e-12:
                    stats["post_sampling_prior_drift"] = candidate_drift
                    break
                drift_entries = candidate_drift.get("class_drift") or {}
                trimmed = False
                for class_id, payload in sorted(
                    drift_entries.items(),
                    key=lambda item: (
                        -abs(float((item[1] or {}).get("drift", 0.0) or 0.0)),
                        int(item[0]) if str(item[0]).isdigit() else 10**9,
                    ),
                ):
                    try:
                        class_key = int(class_id)
                    except (TypeError, ValueError):
                        continue
                    if float((payload or {}).get("drift", 0.0) or 0.0) <= 0.0:
                        continue
                    candidate_positions = np.where(y_train[added_idx] == class_key)[0]
                    if len(candidate_positions) == 0:
                        continue
                    added_idx = np.delete(added_idx, int(candidate_positions[-1]))
                    stats["prior_drift_guard_applied"] = True
                    trimmed = True
                    break
                if not trimmed:
                    stats["post_sampling_prior_drift"] = candidate_drift
                    break

        if not isinstance(stats.get("post_sampling_prior_drift"), Mapping) or not bool(
            (stats.get("post_sampling_prior_drift") or {}).get("available")
        ):
            candidate_labels = (
                np.concatenate([y_train, y_train[added_idx]], axis=0)
                if len(added_idx) > 0
                else y_train
            )
            stats["post_sampling_prior_drift"] = self._compute_class_share_drift_from_labels(
                drift_reference_labels,
                candidate_labels,
            )

        if len(added_idx) == 0:
            stats["reason"] = "prior_drift_guard_trimmed_all_samples"
            return X_train, y_train, stats

        X_balanced = np.concatenate([X_train, X_train[added_idx]], axis=0)
        y_balanced = np.concatenate([y_train, y_train[added_idx]], axis=0)
        stats["enabled"] = True
        stats["added_samples"] = int(len(added_idx))
        stats["class_counts_after"] = self._build_class_support_map(y_balanced)
        stats["reason"] = "applied"
        return X_balanced, y_balanced, stats

    def _downsample_easy_unoccupied(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        seq_timestamps: np.ndarray,
        room_name: str,
        resolved_cfg: Optional[Dict[str, float | int]] = None,
        reference_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Downsample long repetitive unoccupied runs while keeping transitions.

        Keeps boundary windows around run edges and samples interior points by stride.
        """
        debug: Dict[str, Any] = {
            "enabled": False,
            "removed": 0,
            "prior_drift_guard_applied": False,
            "prior_drift_reference": "train",
            "post_downsample_prior_drift": {"available": False},
            "class_counts_before": self._build_class_support_map(y_seq),
            "class_counts_after": self._build_class_support_map(y_seq),
        }
        self._last_unoccupied_downsample_debug = dict(debug)

        if len(y_seq) == 0:
            return X_seq, y_seq, seq_timestamps

        le = self.platform.label_encoders.get(room_name)
        if le is None:
            return X_seq, y_seq, seq_timestamps

        classes = getattr(le, 'classes_', None)
        if classes is None:
            return X_seq, y_seq, seq_timestamps

        unoccupied_indices = np.where(np.asarray(classes) == 'unoccupied')[0]
        if len(unoccupied_indices) == 0:
            return X_seq, y_seq, seq_timestamps
        unoccupied_id = int(unoccupied_indices[0])

        cfg = resolved_cfg or self._resolve_unoccupied_downsample_config(room_name)
        min_share = float(cfg["min_share"])
        stride = int(cfg["stride"])
        boundary_keep = int(cfg["boundary_keep"])
        min_run_length = int(cfg["min_run_length"])
        prior_drift_guard_enabled = bool(cfg.get("prior_drift_guard_enabled", False))
        max_post_downsample_prior_drift = float(cfg.get("max_post_downsample_prior_drift", 1.0) or 1.0)
        debug.update(
            {
                "enabled": True,
                "min_share": float(min_share),
                "stride": int(stride),
                "boundary_keep": int(boundary_keep),
                "min_run_length": int(min_run_length),
                "prior_drift_guard_enabled": bool(prior_drift_guard_enabled),
                "max_post_downsample_prior_drift": float(max_post_downsample_prior_drift),
            }
        )

        unoccupied_share = float(np.mean(y_seq == unoccupied_id))
        drift_reference_labels, drift_reference = self._resolve_prior_drift_reference_labels(
            reference_labels,
            y_seq,
        )
        debug["prior_drift_reference"] = str(drift_reference)
        if unoccupied_share < min_share:
            debug["reason"] = "below_min_share"
            self._last_unoccupied_downsample_debug = dict(debug)
            return X_seq, y_seq, seq_timestamps

        keep_mask = np.ones(len(y_seq), dtype=bool)
        i = 0
        removed = 0

        while i < len(y_seq):
            if y_seq[i] != unoccupied_id:
                i += 1
                continue

            run_start = i
            while i + 1 < len(y_seq) and y_seq[i + 1] == unoccupied_id:
                i += 1
            run_end = i
            run_len = run_end - run_start + 1

            if run_len < min_run_length:
                i += 1
                continue

            inner_start = run_start + boundary_keep
            inner_end = run_end - boundary_keep
            if inner_start <= inner_end:
                for idx in range(inner_start, inner_end + 1):
                    if ((idx - inner_start) % stride) != 0:
                        keep_mask[idx] = False
                        removed += 1

            i += 1

        if removed == 0:
            debug["reason"] = "no_removals"
            debug["post_downsample_prior_drift"] = self._compute_class_share_drift_from_labels(
                drift_reference_labels,
                y_seq,
            )
            self._last_unoccupied_downsample_debug = dict(debug)
            return X_seq, y_seq, seq_timestamps

        removed_indices = np.where(~keep_mask)[0]
        if prior_drift_guard_enabled and max_post_downsample_prior_drift < 1.0 and len(removed_indices) > 0:
            low = 0
            high = int(len(removed_indices))
            best_restore: Optional[int] = None
            best_drift: Optional[Dict[str, Any]] = None
            while low <= high:
                restore_count = int((low + high) // 2)
                candidate_mask = keep_mask.copy()
                if restore_count > 0:
                    candidate_mask[removed_indices[:restore_count]] = True
                candidate_drift = self._compute_class_share_drift_from_labels(
                    drift_reference_labels,
                    y_seq[candidate_mask],
                )
                max_abs_drift = float(candidate_drift.get("max_abs_drift") or 0.0)
                if max_abs_drift <= max_post_downsample_prior_drift + 1e-12:
                    best_restore = restore_count
                    best_drift = candidate_drift
                    high = restore_count - 1
                else:
                    low = restore_count + 1
            if best_restore is not None and best_restore > 0:
                keep_mask[removed_indices[:best_restore]] = True
                removed = int(np.sum(~keep_mask))
                debug["prior_drift_guard_applied"] = True
                debug["post_downsample_prior_drift"] = best_drift or {"available": False}
            else:
                debug["post_downsample_prior_drift"] = self._compute_class_share_drift_from_labels(
                    drift_reference_labels,
                    y_seq[keep_mask],
                )
        else:
            debug["post_downsample_prior_drift"] = self._compute_class_share_drift_from_labels(
                drift_reference_labels,
                y_seq[keep_mask],
            )

        logger.info(
            f"Downsampled easy unoccupied windows for {room_name}: "
            f"removed={removed}, kept={int(keep_mask.sum())}, "
            f"unoccupied_share={unoccupied_share:.1%}, "
            f"min_share={min_share:.2f}, stride={stride}, boundary_keep={boundary_keep}, "
            f"min_run_length={min_run_length}"
        )
        debug["removed"] = int(removed)
        debug["class_counts_after"] = self._build_class_support_map(y_seq[keep_mask])
        debug["reason"] = "applied"
        self._last_unoccupied_downsample_debug = dict(debug)
        return X_seq[keep_mask], y_seq[keep_mask], seq_timestamps[keep_mask]

    @staticmethod
    def _is_timeline_multitask_enabled() -> bool:
        """Feature flag: enable timeline-native training objective path."""
        return str(os.getenv("ENABLE_TIMELINE_MULTITASK", "false")).strip().lower() in {
            "1", "true", "yes", "on", "enabled"
        }

    @staticmethod
    def _timeline_native_rooms() -> set[str]:
        """Return rooms eligible for timeline-native weighting."""
        default_rooms = ",".join(get_timeline_native_rooms_default())
        raw = str(os.getenv("TIMELINE_NATIVE_ROOMS", default_rooms))
        return {
            normalize_room_name(token)
            for token in raw.split(",")
            if str(token).strip()
        }

    @staticmethod
    def _timeline_activity_unoccupied_weight_floor() -> float:
        """Optional floor for masked unoccupied/unknown activity-loss windows."""
        raw = os.getenv("TIMELINE_ACTIVITY_UNOCCUPIED_WEIGHT_FLOOR")
        if raw is None:
            return 0.0
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _timeline_activity_reweight_mode() -> str:
        """
        Class-weight mode for timeline activity head.

        - occupied_only (default): recompute balanced class weights on occupied subset
          when support is sufficient.
        - global: keep full-label global class weights.
        """
        raw = str(os.getenv("TIMELINE_ACTIVITY_REWEIGHT_MODE", "occupied_only")).strip().lower()
        return "global" if raw == "global" else "occupied_only"

    def _timeline_activity_masked_class_ids(self, room_name: str) -> set[int]:
        """Class IDs masked for activity-loss weighting (unoccupied + unknown)."""
        label_encoder = self.platform.label_encoders.get(room_name)
        classes = getattr(label_encoder, "classes_", None) if label_encoder is not None else None
        if classes is None:
            return set()
        masked_labels = {"unoccupied", "unknown"}
        return {
            int(class_id)
            for class_id, label in enumerate(classes)
            if str(label).strip().lower() in masked_labels
        }

    def _build_timeline_activity_loss_weights(
        self,
        *,
        room_name: str,
        y_train: np.ndarray,
        global_class_weight_dict: Dict[int, float],
        timeline_native_weights: np.ndarray,
        use_timeline_native_weighting: bool,
        clinical_policy: Any,
    ) -> Tuple[np.ndarray, Dict[int, float], Dict[str, Any]]:
        """
        Build sample weights for timeline activity head with unoccupied/unknown masking.

        Returns:
            - activity_sample_weight: per-sample activity loss weight
            - activity_class_weight_dict: class-weight dictionary used for activity samples
            - diagnostics: structured weighting diagnostics for metrics/trace
        """
        y_arr = np.asarray(y_train, dtype=np.int32)
        n = int(len(y_arr))
        if n == 0:
            diagnostics = {
                "activity_loss_mask_coverage": 0.0,
                "activity_loss_mask_floor": 0.0,
                "activity_class_weight_mode": "global",
                "activity_class_weight_support": 0,
                "activity_weight_mean_occupied": 0.0,
                "activity_weight_mean_masked": 0.0,
                "masked_class_ids": [],
                "reweight_mode_requested": self._timeline_activity_reweight_mode(),
            }
            return np.zeros((0,), dtype=np.float32), dict(global_class_weight_dict), diagnostics

        class_weight_dict: Dict[int, float] = {
            int(class_id): float(weight)
            for class_id, weight in (global_class_weight_dict or {}).items()
        }

        floor = self._timeline_activity_unoccupied_weight_floor()
        reweight_mode = self._timeline_activity_reweight_mode()
        masked_class_ids = sorted(self._timeline_activity_masked_class_ids(room_name))

        if masked_class_ids:
            masked_flags = np.isin(y_arr, np.asarray(masked_class_ids, dtype=np.int32))
        else:
            masked_flags = np.zeros(shape=(n,), dtype=bool)
        occupied_flags = ~masked_flags

        class_weight_mode = "global"
        class_weight_support = int(len(y_arr))
        fallback_reason = None

        if bool(np.any(masked_flags)) and reweight_mode == "occupied_only":
            occupied_labels = np.asarray(y_arr[occupied_flags], dtype=np.int32)
            unique_occupied = np.unique(occupied_labels)
            min_support_for_reweight = max(8, int(len(unique_occupied)))

            if int(len(occupied_labels)) >= min_support_for_reweight and len(unique_occupied) > 0:
                occupied_weights = compute_class_weight(
                    class_weight="balanced",
                    classes=unique_occupied,
                    y=occupied_labels,
                )
                occupied_weight_dict = {
                    int(class_id): float(weight)
                    for class_id, weight in zip(unique_occupied.astype(np.int32, copy=False), occupied_weights)
                }

                label_encoder = self.platform.label_encoders.get(room_name)
                classes = getattr(label_encoder, "classes_", None) if label_encoder is not None else None
                for class_id, base_weight in occupied_weight_dict.items():
                    label_name = None
                    if classes is not None and 0 <= int(class_id) < len(classes):
                        label_name = str(classes[int(class_id)])
                    priority_multiplier = clinical_policy.get_room_label_multiplier(room_name, label_name)
                    adjusted_weight = float(base_weight) * float(priority_multiplier)
                    adjusted_weight = max(
                        float(clinical_policy.class_weight_floor),
                        min(float(clinical_policy.class_weight_cap), adjusted_weight),
                    )
                    class_weight_dict[int(class_id)] = adjusted_weight

                class_weight_mode = "occupied_only"
                class_weight_support = int(len(occupied_labels))
            else:
                fallback_reason = (
                    f"insufficient_occupied_support:{len(occupied_labels)}<required:{min_support_for_reweight}"
                )
                class_weight_support = int(len(occupied_labels))

        weight_by_sample = np.asarray(
            [class_weight_dict.get(int(class_id), 1.0) for class_id in y_arr],
            dtype=np.float32,
        )
        if use_timeline_native_weighting:
            native = np.asarray(timeline_native_weights, dtype=np.float32)
            if native.shape[0] != n:
                native = np.ones(shape=(n,), dtype=np.float32)
            weight_by_sample = np.asarray(weight_by_sample * native, dtype=np.float32)

        activity_mask = np.where(masked_flags, 0.0, 1.0).astype(np.float32)
        effective_mask = np.asarray(activity_mask + (1.0 - activity_mask) * float(floor), dtype=np.float32)
        activity_sample_weight = np.asarray(weight_by_sample * effective_mask, dtype=np.float32)

        diagnostics: Dict[str, Any] = {
            "activity_loss_mask_coverage": float(np.mean(masked_flags.astype(np.float32))),
            "activity_loss_mask_floor": float(floor),
            "activity_class_weight_mode": str(class_weight_mode),
            "activity_class_weight_support": int(class_weight_support),
            "activity_weight_mean_occupied": (
                float(np.mean(activity_sample_weight[occupied_flags]))
                if bool(np.any(occupied_flags))
                else 0.0
            ),
            "activity_weight_mean_masked": (
                float(np.mean(activity_sample_weight[masked_flags]))
                if bool(np.any(masked_flags))
                else 0.0
            ),
            "masked_class_ids": [int(class_id) for class_id in masked_class_ids],
            "reweight_mode_requested": str(reweight_mode),
        }
        if fallback_reason:
            diagnostics["activity_class_weight_fallback_reason"] = fallback_reason

        return activity_sample_weight, class_weight_dict, diagnostics

    def _build_timeline_targets(
        self,
        *,
        room_name: str,
        y_train: np.ndarray,
        seq_timestamps: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Build timeline-native targets for multi-task training.
        
        Generates:
        - activity_labels: Original class labels (sparse)
        - occupancy_labels: Binary occupancy (1=occupied, 0=unoccupied)
        - boundary_start_labels: Episode start boundaries
        - boundary_end_labels: Episode end boundaries
        
        Returns:
            targets_dict: Dictionary of target arrays
            debug_info: Debug information about target generation
        """
        n = int(len(y_train))
        room_key = normalize_room_name(room_name)
        debug: Dict[str, Any] = {
            "enabled": False,
            "room": room_key,
            "n_samples": n,
            "has_activity": False,
            "has_occupancy": False,
            "has_boundaries": False,
        }
        
        if n == 0:
            return {}, debug
        
        # Check if timeline multitask is enabled for this room
        if not self._is_timeline_multitask_enabled() or room_key not in self._timeline_native_rooms():
            debug["reason"] = "feature_flag_or_room_disabled"
            return {}, debug
        
        # Get label encoder to map class IDs to labels
        le = self.platform.label_encoders.get(room_name)
        classes = getattr(le, "classes_", None) if le is not None else None
        if classes is None:
            debug["reason"] = "missing_label_encoder"
            return {}, debug
        
        id_to_label = {int(i): str(lbl).strip().lower() for i, lbl in enumerate(classes)}
        labels = np.asarray([id_to_label.get(int(cid), "unknown") for cid in y_train], dtype=object)
        
        # Activity targets (sparse labels) - already have these as y_train
        targets = {
            "activity_labels": y_train.astype(np.int32),
        }
        debug["has_activity"] = True
        
        # Occupancy targets (binary)
        excluded = {"unoccupied", "unknown"}
        occupancy_labels = np.asarray([0 if lbl in excluded else 1 for lbl in labels], dtype=np.int32)
        targets["occupancy_labels"] = occupancy_labels
        debug["has_occupancy"] = True
        debug["occupancy_rate"] = float(np.mean(occupancy_labels))
        
        # Boundary targets
        if len(labels) >= 2:
            try:
                boundary = build_boundary_targets(labels)
                targets["boundary_start_labels"] = boundary.start_flags.astype(np.int32)
                targets["boundary_end_labels"] = boundary.end_flags.astype(np.int32)
                debug["has_boundaries"] = True
                debug["start_boundaries"] = int(np.sum(boundary.start_flags))
                debug["end_boundaries"] = int(np.sum(boundary.end_flags))
            except Exception as e:
                debug["boundary_error"] = str(e)
                targets["boundary_start_labels"] = np.zeros(n, dtype=np.int32)
                targets["boundary_end_labels"] = np.zeros(n, dtype=np.int32)
        else:
            targets["boundary_start_labels"] = np.zeros(n, dtype=np.int32)
            targets["boundary_end_labels"] = np.zeros(n, dtype=np.int32)
        
        debug["enabled"] = True
        return targets, debug

    def _build_timeline_native_sequence_weights(
        self,
        *,
        room_name: str,
        y_train: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Build timeline-native sample weights for sequence-level classification.

        This is a proxy timeline objective: emphasize care-transition boundaries and
        occupied windows while mildly downweighting long easy-unoccupied runs.
        """
        n = int(len(y_train))
        weights = np.ones(shape=(n,), dtype=np.float32)
        room_key = normalize_room_name(room_name)
        debug: Dict[str, Any] = {
            "enabled": False,
            "room": room_key,
            "n": n,
            "transition_count": 0,
            "occupied_share": 0.0,
            "mean": 1.0,
            "min": 1.0,
            "max": 1.0,
        }
        if n == 0:
            return weights, debug

        if not self._is_timeline_multitask_enabled() or room_key not in self._timeline_native_rooms():
            debug["reason"] = "feature_flag_or_room_disabled"
            return weights, debug

        le = self.platform.label_encoders.get(room_name)
        classes = getattr(le, "classes_", None) if le is not None else None
        if classes is None:
            debug["reason"] = "missing_label_encoder"
            return weights, debug

        id_to_label = {int(i): str(lbl).strip().lower() for i, lbl in enumerate(classes)}
        labels = np.asarray([id_to_label.get(int(cid), "unknown") for cid in y_train], dtype=object)
        excluded = {"unoccupied", "unknown"}
        occupied_mask = np.asarray([lbl not in excluded for lbl in labels], dtype=bool)
        weights[occupied_mask] = np.maximum(weights[occupied_mask], 1.4)

        # Emphasize windows around start/end boundaries.
        if len(labels) >= 2:
            try:
                boundary = build_boundary_targets(labels)
                boundary_idx = np.where((boundary.start_flags + boundary.end_flags) > 0)[0]
            except Exception:
                boundary_idx = np.asarray([], dtype=np.int64)
            band = 6
            for idx in boundary_idx.tolist():
                start = max(0, int(idx) - band)
                end = min(n, int(idx) + band + 1)
                weights[start:end] = np.maximum(weights[start:end], 2.1)
            debug["transition_count"] = int(len(boundary_idx))

        # Mildly downweight deep interior of long unoccupied runs.
        i = 0
        while i < n:
            if labels[i] != "unoccupied":
                i += 1
                continue
            run_start = i
            while i < n and labels[i] == "unoccupied":
                i += 1
            run_end = i
            run_len = int(run_end - run_start)
            if run_len >= 180:
                mid_start = run_start + int(0.25 * run_len)
                mid_end = run_end - int(0.25 * run_len)
                if mid_end > mid_start:
                    weights[mid_start:mid_end] = np.minimum(weights[mid_start:mid_end], 0.85)

        weights = np.clip(weights, 0.25, 5.0).astype(np.float32, copy=False)
        debug.update(
            {
                "enabled": True,
                "occupied_share": float(np.mean(occupied_mask)),
                "mean": float(np.mean(weights)),
                "min": float(np.min(weights)),
                "max": float(np.max(weights)),
            }
        )
        return weights, debug

    def _calibrate_class_thresholds(
        self,
        model: Any,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        room_name: str,
    ) -> Dict[int, float]:
        """
        Learn per-class confidence thresholds from calibration data.

        For each class:
        - find lowest threshold that satisfies precision target and recall floor
        - fallback to best-F1 threshold when constraints are unattainable
        - clip threshold to [policy.threshold_floor, policy.threshold_cap]
        """
        calibration_policy = self._active_policy().calibration
        thresholds = self._default_class_thresholds(room_name)
        if len(X_calib) == 0 or len(y_calib) == 0:
            self._last_calibration_debug = []
            return thresholds

        try:
            y_scores = model.predict(X_calib, verbose=0)
        except TypeError:
            y_scores = model.predict(X_calib)

        if isinstance(y_scores, dict):
            y_scores = y_scores.get("activity_logits")
        if y_scores is not None and not isinstance(y_scores, np.ndarray):
            y_scores = np.asarray(y_scores)
        if y_scores is None or len(y_scores) == 0:
            self._last_calibration_debug = []
            return thresholds
        # Timeline multitask path emits logits.
        if bool(getattr(model, "_timeline_multitask_enabled", False)):
            y_scores = tf.nn.softmax(y_scores, axis=1).numpy()

        n_classes = y_scores.shape[1]
        calibration_debug = []

        for class_id in range(n_classes):
            label_name = self._get_label_name(room_name, class_id)
            default_thr = float(DEFAULT_CONFIDENCE_THRESHOLD)
            y_true = (y_calib == class_id).astype(int)
            support = int(y_true.sum())

            if support < int(calibration_policy.min_support_per_class):
                thresholds[class_id] = float(
                    np.clip(default_thr, calibration_policy.threshold_floor, calibration_policy.threshold_cap)
                )
                calibration_debug.append({
                    'class_id': class_id,
                    'label': label_name or 'unknown',
                    'support': support,
                    'status': 'fallback_low_support',
                    'threshold': thresholds[class_id],
                })
                continue

            scores = y_scores[:, class_id]
            precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
            if len(pr_thresholds) == 0:
                thresholds[class_id] = float(
                    np.clip(default_thr, calibration_policy.threshold_floor, calibration_policy.threshold_cap)
                )
                calibration_debug.append({
                    'class_id': class_id,
                    'label': label_name or 'unknown',
                    'support': support,
                    'status': 'fallback_no_curve',
                    'threshold': thresholds[class_id],
                })
                continue

            p = precision[:-1]
            r = recall[:-1]
            target_precision = float(calibration_policy.get_precision_target(label_name))
            recall_floor = float(calibration_policy.get_recall_floor(label_name))

            valid = np.where((p >= target_precision) & (r >= recall_floor))[0]
            if len(valid) > 0:
                # Lowest threshold among valid points -> highest recall satisfying precision target.
                raw_threshold = float(np.min(pr_thresholds[valid]))
                status = 'target_met'
            else:
                # Fallback: choose best F1 operating point.
                f1 = (2.0 * p * r) / np.maximum(p + r, 1e-9)
                best_idx = int(np.nanargmax(f1))
                raw_threshold = float(pr_thresholds[best_idx])
                status = 'fallback_best_f1'

            final_threshold = float(
                np.clip(raw_threshold, calibration_policy.threshold_floor, calibration_policy.threshold_cap)
            )
            thresholds[class_id] = final_threshold

            calibration_debug.append({
                'class_id': class_id,
                'label': label_name or 'unknown',
                'support': support,
                'target_precision': float(target_precision),
                'recall_floor': float(recall_floor),
                'status': status,
                'threshold': final_threshold,
            })

        self._last_calibration_debug = calibration_debug
        logger.info(f"Calibration thresholds for {room_name}: {thresholds}")
        logger.info(f"Calibration breakdown for {room_name}: {calibration_debug}")
        return thresholds

    def train_room_with_leakage_free_scaling(
        self,
        room_name: str,
        raw_df: pd.DataFrame,
        seq_length: int,
        elder_id: str,
        progress_callback=None,
        training_mode: str = "full_retrain",
        defer_promotion: bool = False,
        gate_evaluation_result: Optional[Dict[str, Any]] = None,
        event_first_shadow: Optional[bool] = None,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Item 1 Phase B+C: Train with leakage-free preprocessing (train-split scaling).
        
        This method implements the proper temporal-split-first flow:
        1. preprocess_without_scaling() - no scaler fitting
        2. temporal_split_dataframe() - split by time
        3. apply_scaling() - fit on train, transform all
        
        Parameters:
        -----------
        room_name : str
            Room to train
        raw_df : pd.DataFrame
            Raw input DataFrame (before preprocessing)
        seq_length : int
            Sequence length
        elder_id : str
            Resident ID
        progress_callback : callable, optional
            Progress callback(percent, message)
        training_mode : str
            "full_retrain" or "correction_fine_tune"
        defer_promotion : bool
            Whether to defer promotion
        validation_split : float
            Fraction for validation + calibration
            
        Returns:
        --------
        Dict with training metrics and leakage-free metadata
        """
        from ml.train_split_scaling_pipeline import (
            prepare_training_data_with_train_split_scaling,
            validate_no_leakage,
            is_train_split_scaling_enabled,
        )
        from ml.statistical_validity_gate import (
            evaluate_promotion_with_statistical_validity,
            create_statistical_validity_gate_from_policy,
        )
        
        if not is_train_split_scaling_enabled():
            raise ModelTrainError(
                "Leakage guard disabled: ENABLE_TRAIN_SPLIT_SCALING must remain enabled in Beta 6"
            )
        
        active_policy = self._active_policy()
        calib_policy = active_policy.calibration
        
        # Phase A+B+C: Preprocess → Split → Scale
        logger.info(f"[{room_name}] Using leakage-free train-split scaling")
        prep_result = prepare_training_data_with_train_split_scaling(
            platform=self.platform,
            room_name=room_name,
            raw_df=raw_df,
            validation_split=validation_split,
            calibration_fraction=float(calib_policy.fraction_of_holdout),
            min_calibration_samples=int(calib_policy.min_samples),
            apply_denoising=False,  # Can be parameterized
        )
        
        # Validate no leakage
        train_scaled = prep_result['train_scaled']
        val_scaled = prep_result['val_scaled']
        calib_scaled = prep_result['calib_scaled']
        
        no_leakage, violations = validate_no_leakage(
            train_scaled, val_scaled, calib_scaled
        )
        if not no_leakage:
            raise ValueError(f"Temporal leakage detected: {violations}")
        
        # Combine train+val for sequence creation (sequences will be split later)
        # Note: For proper walk-forward, we'd use timestamps, but for now combine
        processed_df = pd.concat([train_scaled, val_scaled], ignore_index=True)
        if calib_scaled is not None:
            processed_df = pd.concat([processed_df, calib_scaled], ignore_index=True)
        
        # Store split info for metadata
        processed_df.attrs['train_split_scaling'] = {
            'enabled': True,
            'split_metadata': prep_result['split_metadata'],
            'scaler_metadata': prep_result['scaler_metadata'],
        }
        
        # Call original train_room with processed data
        metrics = self.train_room(
            room_name=room_name,
            processed_df=processed_df,
            seq_length=seq_length,
            elder_id=elder_id,
            progress_callback=progress_callback,
            training_mode=training_mode,
            defer_promotion=defer_promotion,
            gate_evaluation_result=gate_evaluation_result,
            event_first_shadow=event_first_shadow,
        )
        
        if metrics is None:
            return None
        
        # Add leakage-free metadata
        metrics['train_split_scaling'] = processed_df.attrs['train_split_scaling']
        
        # Item 6: Statistical Validity Gate
        # Evaluate on calibration data
        if calib_scaled is not None and 'activity_encoded' in calib_scaled.columns:
            y_calib = calib_scaled['activity_encoded'].values
            y_val = val_scaled['activity_encoded'].values if val_scaled is not None else None
            
            calib_metrics = {
                'is_fallback': metrics.get('metric_source') == 'train_fallback_small_dataset',
                'metric_source': metrics.get('metric_source', 'unknown'),
            }
            
            promotable, sv_reasons, sv_result = evaluate_promotion_with_statistical_validity(
                calibration_metrics=calib_metrics,
                y_calib=y_calib,
                y_val=y_val,
                room_name=room_name,
                policy=active_policy,
            )
            
            metrics['statistical_validity_gate'] = sv_result
            
            # If statistical validity fails, override gate_pass
            if not promotable and metrics.get('gate_pass', False):
                logger.warning(
                    f"[{room_name}] Overriding gate_pass due to statistical validity failure"
                )
                metrics['gate_pass'] = False
                metrics['gate_reasons'] = metrics.get('gate_reasons', []) + sv_reasons
        
        return metrics

    def train_room(self, 
                   room_name: str, 
                   processed_df, 
                   seq_length: int, 
                   elder_id: str,
                   progress_callback=None,
                   training_mode: str = "full_retrain",
                   defer_promotion: bool = False,
                   gate_evaluation_result: Optional[Dict[str, Any]] = None,
                   event_first_shadow: Optional[bool] = None,
                   explicit_validation_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                   explicit_calibration_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Train a model for a specific room.
        
        Args:
            room_name: Room to train
            processed_df: Pre-processed DataFrame (resampled, encoded)
            seq_length: Sequence length for this room
            elder_id: Resident ID for saving model
            progress_callback: Optional callable(percent, message)
            training_mode: "full_retrain" or "correction_fine_tune"
            defer_promotion: Whether to defer promotion to latest
            gate_evaluation_result: Optional pre-training gate evaluation result from GateIntegrationPipeline
            event_first_shadow: Optional override for event-first shadow execution.
            explicit_validation_data: Optional explicit validation split override as
                `(X_val, y_val, seq_timestamps)` for grouped-date flows.
            explicit_calibration_data: Optional explicit calibration split override as
                `(X_calib, y_calib)` for grouped-date flows.
            
        Returns:
            Dict containing training metrics with gate_pass and gate_reasons
        """
        try:
            if self._multi_seed_enabled_for_room(room_name):
                return self._train_room_with_multi_seed_panel(
                    room_name=room_name,
                    processed_df=processed_df,
                    seq_length=seq_length,
                    elder_id=elder_id,
                    progress_callback=progress_callback,
                    training_mode=training_mode,
                    defer_promotion=defer_promotion,
                    gate_evaluation_result=gate_evaluation_result,
                    event_first_shadow=event_first_shadow,
                )
            training_mode = (training_mode or "full_retrain").strip().lower()
            is_correction_fine_tune = training_mode == "correction_fine_tune"
            active_policy = self._active_policy()
            self._policy_snapshot = active_policy
            policy_hash = self._policy_hash(active_policy)
            calibration_policy = active_policy.calibration
            training_strategy = self._resolve_training_strategy(room_name)
            self._set_training_random_seed(active_policy.reproducibility.random_seed)
            fine_tune_params = self._get_fine_tuning_params() if is_correction_fine_tune else {}
            event_first_shadow_enabled = bool(
                active_policy.event_first.shadow if event_first_shadow is None else event_first_shadow
            )

            # PR-2: Check pre-training gate results if provided
            pre_training_pass = True
            if gate_evaluation_result:
                pre_training_pass = bool(
                    gate_evaluation_result.get(
                        "pre_training_pass",
                        gate_evaluation_result.get("gate_pass", True),
                    )
                )
            if gate_evaluation_result and not pre_training_pass:
                logger.warning(f"Pre-training gates failed for {room_name}: {gate_evaluation_result.get('failure_stage')}")
                
                # Build minimal metrics with gate failure info
                metrics = {
                    "room": room_name,
                    "gate_pass": False,
                    "gate_reasons": [f"{gate_evaluation_result.get('failure_stage')}_failed"],
                    "gate_stack": gate_evaluation_result.get("gate_stack", []),
                    "accuracy": 0.0,
                    "samples": len(processed_df),
                    "training_days": 0.0,
                }
                
                # Save rejection artifact if available
                if gate_evaluation_result.get("rejection_artifact"):
                    models_dir = self.registry.get_models_dir(elder_id)
                    rejection_path = models_dir / f"{room_name}_{gate_evaluation_result.get('run_id', 'unknown')}_why_rejected.json"
                    try:
                        with open(rejection_path, 'w') as f:
                            json.dump(gate_evaluation_result["rejection_artifact"], f, indent=2, default=str)
                        logger.info(f"Saved rejection artifact to {rejection_path}")
                        metrics["rejection_artifact_path"] = str(rejection_path)
                    except Exception as e:
                        logger.error(f"Failed to save rejection artifact: {e}")
                
                # Log training history with gate failure
                self._log_training_history(
                    elder_id=elder_id,
                    room_name=room_name,
                    accuracy=0.0,
                    samples=len(processed_df),
                    epochs=0,
                    status="REJECTED_BY_GATE",
                    error_msg=f"Pre-training gate failed: {gate_evaluation_result.get('failure_stage')}",
                )
                
                return metrics

            if len(processed_df) < seq_length:
                logger.warning(f"Insufficient data for {room_name}: {len(processed_df)} < {seq_length}")
                return None

            # Fail-closed guard: prevent silent label corruption in sequence path.
            processed_df = self._require_valid_activity_encoded(processed_df, room_name)

            # PR-3.2: Apply duplicate resolution policy before sequence creation
            duplicate_policy = getattr(self.policy, 'duplicate_resolution', None) or DuplicateResolutionPolicy()
            if duplicate_policy.method != "first":  # Only resolve if not using legacy 'first' method
                # Ensure duplicate resolver always gets a concrete timestamp column.
                if 'timestamp' not in processed_df.columns:
                    idx_col = processed_df.index.name or 'index'
                    processed_df = (
                        processed_df.reset_index()
                        .rename(columns={idx_col: 'timestamp'})
                    )

                resolver = DuplicateTimestampResolver(policy=duplicate_policy)
                processed_df = resolver.resolve(
                    processed_df,
                    timestamp_col='timestamp',
                    label_col='activity' if 'activity' in processed_df.columns else 'activity_encoded'
                )
                # Re-extract after deduplication
                sensor_data = np.asarray(processed_df[self.platform.sensor_columns].values, dtype=np.float32)
                labels = processed_df['activity_encoded'].values
            else:
                # Force float32 at the source to save memory and align with Metal
                sensor_data = np.asarray(processed_df[self.platform.sensor_columns].values, dtype=np.float32)
                labels = processed_df['activity_encoded'].values

            # PR-3.1: Strict sequence-label alignment
            # Extract timestamps for alignment
            if 'timestamp' in processed_df.columns:
                all_ts = pd.to_datetime(processed_df['timestamp']).values
            else:
                all_ts = pd.to_datetime(processed_df.index).values
            
            try:
                X_seq, y_seq, seq_timestamps = safe_create_sequences(
                    platform=self.platform,
                    sensor_data=sensor_data,
                    labels=labels,
                    seq_length=seq_length,
                    room_name=room_name,
                    timestamps=all_ts,
                    strict=True,  # PR-3: Enforce strict alignment
                )
                # Convert to expected dtypes
                X_seq = np.asarray(X_seq, dtype=np.float32)
                y_seq = np.asarray(y_seq, dtype=np.int32)
                seq_timestamps = np.asarray(seq_timestamps, dtype="datetime64[ns]")
            except SequenceLabelAlignmentError as e:
                logger.error(f"[{room_name}] Sequence alignment failed: {e}")
                raise ModelTrainError(f"Sequence alignment failed for {room_name}: {e}") from e
            
            # Augment with Golden Samples
            if 'timestamp' in processed_df.columns:
                 existing_ts = processed_df['timestamp']
            else:
                 # Fallback: if timestamp is index
                 existing_ts = processed_df.index
                 
            window_sec = self.room_config.get_sequence_window(room_name)
            interval = self.room_config.get_data_interval(room_name)
            
            augmented = self.augment_training_data(
                room_name, elder_id, existing_ts, X_seq, y_seq, 
                window_sec, interval, seq_timestamps
            )
            if isinstance(augmented, tuple) and len(augmented) == 3:
                X_seq, y_seq, seq_timestamps = augmented
            elif isinstance(augmented, tuple) and len(augmented) == 2:
                X_seq, y_seq = augmented
                if len(seq_timestamps) > len(X_seq):
                    seq_timestamps = seq_timestamps[:len(X_seq)]
                elif len(seq_timestamps) < len(X_seq):
                    if len(seq_timestamps) == 0:
                        seq_timestamps = np.full(len(X_seq), np.datetime64("NaT"), dtype="datetime64[ns]")
                    else:
                        pad = np.full(len(X_seq) - len(seq_timestamps), seq_timestamps[-1], dtype="datetime64[ns]")
                        seq_timestamps = np.concatenate([seq_timestamps, pad], axis=0)
            else:
                raise ValueError("augment_training_data returned unexpected output")
            
            if len(X_seq) == 0:
                logger.warning(f"No sequences created for {room_name}")
                return None

            # Ensure chronological ordering before any temporal split/metrics.
            order = np.argsort(seq_timestamps)
            X_seq = X_seq[order]
            y_seq = y_seq[order]
            seq_timestamps = seq_timestamps[order]

            validation_override = None
            if explicit_validation_data is not None:
                X_val_override = np.asarray(explicit_validation_data[0], dtype=np.float32)
                y_val_override = np.asarray(explicit_validation_data[1], dtype=np.int32)
                ts_val_override = np.asarray(explicit_validation_data[2], dtype="datetime64[ns]")
                if not (
                    len(X_val_override) == len(y_val_override) == len(ts_val_override)
                ):
                    raise ValueError(
                        f"explicit validation data mismatch for {room_name}: "
                        f"X={len(X_val_override)}, y={len(y_val_override)}, ts={len(ts_val_override)}"
                    )
                validation_override = (X_val_override, y_val_override, ts_val_override)

            calibration_override = None
            if explicit_calibration_data is not None:
                X_calib_override = np.asarray(explicit_calibration_data[0], dtype=np.float32)
                y_calib_override = np.asarray(explicit_calibration_data[1], dtype=np.int32)
                if len(X_calib_override) != len(y_calib_override):
                    raise ValueError(
                        f"explicit calibration data mismatch for {room_name}: "
                        f"X={len(X_calib_override)}, y={len(y_calib_override)}"
                    )
                calibration_override = (X_calib_override, y_calib_override)

            unoccupied_cfg = self._resolve_unoccupied_downsample_config(room_name)
            total_sequences_pre_split = int(len(X_seq))
            holdout_support_floor = self._resolve_min_holdout_support(room_name)
            critical_support_info = self._resolve_critical_class_support(
                room_name=room_name,
                y_seq=y_seq,
                min_support=holdout_support_floor,
            )

            # Build Model - Transformer default, with optional correction warm-start
            num_classes = len(np.unique(labels))
            
            # Helper to handle missing labels in small datasets (e.g. only 0 and 2 present)
            # Ensure num_classes covers the max index found in label encoder
            if hasattr(self.platform, 'label_encoders') and room_name in self.platform.label_encoders:
                 max_class = len(self.platform.label_encoders[room_name].classes_)
                 if max_class > num_classes:
                     num_classes = max_class

            champion_meta = self.registry.get_current_version_metadata(elder_id, room_name)
            if not isinstance(champion_meta, dict):
                champion_meta = None

            model, did_warm_start = self._build_model_for_room(
                room_name=room_name,
                seq_length=seq_length,
                num_classes=num_classes,
                elder_id=elder_id,
                training_mode=training_mode,
                warm_start=bool(fine_tune_params.get("warm_start", True)),
            )
            shared_enabled = self._is_shared_backbone_enabled()
            champion_identity = champion_meta.get("model_identity", {}) if champion_meta else {}
            champion_backbone_id = (
                str(champion_identity.get("backbone_id", "")).strip()
                if isinstance(champion_identity, dict)
                else ""
            )
            shared_backbone_id = self._get_active_shared_backbone_id(default="")
            effective_shared_backbone_id = shared_backbone_id or (champion_backbone_id if shared_enabled else "")
            adapter_training_flag = str(os.getenv("ENABLE_ADAPTER_ONLY_TRAINING", "true")).strip().lower() in {
                "1", "true", "yes", "on", "enabled"
            }
            shared_backbone_loaded_layers = 0

            if shared_enabled and effective_shared_backbone_id and not did_warm_start:
                named_backbone_weights = self.registry.load_shared_backbone_weights(
                    elder_id=elder_id,
                    room_name=room_name,
                    backbone_id=effective_shared_backbone_id,
                )
                if named_backbone_weights:
                    load_stats = self.registry.apply_named_layer_weights(model, named_backbone_weights)
                    shared_backbone_loaded_layers = int(load_stats.get("loaded_layers", 0) or 0)
                    logger.info(
                        f"Loaded shared backbone weights for {elder_id}/{room_name} "
                        f"(backbone_id={effective_shared_backbone_id}): {load_stats}"
                    )
                else:
                    logger.info(
                        f"No shared backbone snapshot found for {elder_id}/{room_name} "
                        f"(backbone_id={effective_shared_backbone_id}); training starts from init."
                    )

            adapter_training_requested = bool(shared_enabled and adapter_training_flag)
            backbone_ready_for_adapter_only = bool(did_warm_start or shared_backbone_loaded_layers > 0)
            adapter_training_only = bool(adapter_training_requested and backbone_ready_for_adapter_only)
            if adapter_training_requested and not adapter_training_only:
                if not effective_shared_backbone_id and not did_warm_start:
                    reason = "missing_backbone_id"
                else:
                    reason = "shared_backbone_not_loaded"
                logger.warning(
                    f"Adapter-only training disabled for {elder_id}/{room_name}: {reason}; "
                    "falling back to full-model optimization."
                )

            freeze_summary = None
            if adapter_training_only:
                freeze_summary = self._apply_shared_adapter_freeze(model)

            learning_rate = (
                float(fine_tune_params.get("learning_rate", 1e-4))
                if is_correction_fine_tune else (1e-4 if adapter_training_only else 1e-3)
            )

            timeline_multitask_enabled = bool(getattr(model, "_timeline_multitask_enabled", False))
            train_batch_shuffle_enabled = bool(self._should_shuffle_post_split_batches(room_name))
            if train_batch_shuffle_enabled:
                logger.info(
                    "Post-split train batch shuffling enabled for %s; temporal train/holdout split is preserved.",
                    room_name,
                )

            # Disable jit_compile to avoid compilation hangs on macOS
            if timeline_multitask_enabled:
                timeline_loss_weights = dict(getattr(model, "_timeline_loss_weights", {}) or {})
                if not timeline_loss_weights:
                    timeline_loss_weights = {
                        "activity_logits": 1.0,
                        "occupancy_logits": 1.0,
                        "boundary_start_logits": 0.5,
                        "boundary_end_logits": 0.5,
                    }
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss={
                        "activity_logits": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        "occupancy_logits": tf.keras.losses.BinaryCrossentropy(from_logits=True),
                        "boundary_start_logits": tf.keras.losses.BinaryCrossentropy(from_logits=True),
                        "boundary_end_logits": tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    },
                    loss_weights=timeline_loss_weights,
                    metrics={"activity_logits": ["accuracy"]},
                    jit_compile=False,
                )
            else:
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'],
                    jit_compile=False,
                )
            
            # Train
            logger.info(f"Fitting model for {room_name}...")
            
            # Progress Callback for Keras
            from tensorflow.keras.callbacks import EarlyStopping
            callbacks = []
            
            # Early Stopping to prevent overfitting
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=int(fine_tune_params.get("patience", 1) if is_correction_fine_tune else 2),
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)

            max_epochs = int(fine_tune_params.get("epochs", min(int(DEFAULT_EPOCHS), 4))) if is_correction_fine_tune else int(DEFAULT_EPOCHS)
            
            if progress_callback:
                from tensorflow.keras.callbacks import LambdaCallback
                def on_epoch_end(epoch, logs):
                    percent = int(((epoch + 1) / max_epochs) * 100)
                    logs = logs or {}
                    acc_value = (
                        logs.get("accuracy")
                        if logs.get("accuracy") is not None
                        else logs.get("activity_logits_accuracy")
                    )
                    if acc_value is None:
                        acc_value = 0.0
                    progress_callback(
                        percent,
                        f"Epoch {epoch+1}/{max_epochs} - loss: {float(logs.get('loss', 0.0)):.4f}, "
                        f"acc: {float(acc_value):.4f}",
                    )
                callbacks.append(LambdaCallback(on_epoch_end=on_epoch_end))
            
            # Handle small datasets - skip holdout split if too few samples
            val_split = DEFAULT_VALIDATION_SPLIT
            if len(X_seq) < 100:
                val_split = 0.0
                logger.warning(f"Small dataset ({len(X_seq)} samples) for {room_name}, skipping validation split")

            # P1 Fix: Prevent time-series leakage via shuffle
            # Use strict temporal split: Train on past, Holdout on future
            calibration_data = None
            shadow_eval_timestamps = None
            shadow_eval_y_true = None
            shadow_eval_probs = None
            split_support_debug: Dict[str, Any] = {}
            calib_split_support_debug: Dict[str, Any] = {}
            validation_source = "none"
            calibration_source = "none"
            if validation_override is not None or calibration_override is not None:
                X_train, y_train = X_seq, y_seq
                ts_train = seq_timestamps
                if validation_override is not None:
                    X_val, y_val, ts_val = validation_override
                    validation_data = (X_val, y_val)
                    shadow_eval_timestamps = ts_val
                    validation_source = "explicit_split"
                else:
                    validation_data = None
                    shadow_eval_timestamps = None

                if calibration_override is not None:
                    calibration_data = calibration_override
                    calibration_source = "explicit_split"
                elif validation_data is not None:
                    calibration_data = validation_data
                    calibration_source = "validation_fallback"
                else:
                    calibration_data = None

                split_support_debug = {
                    "mode": "explicit_split",
                    "train_sequences": int(len(X_train)),
                    "validation_sequences": int(len(validation_data[0])) if validation_data is not None else 0,
                }
                calib_split_support_debug = {
                    "mode": calibration_source,
                    "calibration_sequences": int(len(calibration_data[0])) if calibration_data is not None else 0,
                }
                sampling_reference_labels = validation_data[1] if validation_data is not None else None
                logger.info(
                    f"Using explicit grouped-date split for {room_name}: "
                    f"Train={len(X_train)}, "
                    f"Val={len(validation_data[0]) if validation_data is not None else 0}, "
                    f"Calib={len(calibration_data[0]) if calibration_data is not None else 0}"
                )
            elif val_split > 0:
                default_split_idx = int(len(X_seq) * (1 - val_split))
                default_split_idx = max(1, min(len(X_seq) - 1, default_split_idx))
                split_idx, split_support_debug = self._select_temporal_split_index_with_support(
                    y_seq=y_seq,
                    default_split_idx=default_split_idx,
                    required_class_ids=critical_support_info.get("class_ids", []),
                    min_support=holdout_support_floor,
                )
                if not bool(split_support_debug.get("found", True)):
                    logger.warning(
                        f"Critical-label holdout support not fully satisfiable for {room_name}; "
                        f"using default split={split_idx}. Details={split_support_debug}"
                    )

                X_train = X_seq[:split_idx]
                y_train = y_seq[:split_idx]
                ts_train = seq_timestamps[:split_idx]
                X_holdout = X_seq[split_idx:]
                y_holdout = y_seq[split_idx:]
                ts_holdout = seq_timestamps[split_idx:]

                # Separate calibration split from holdout when enough samples exist.
                if len(X_holdout) >= int(calibration_policy.separate_calibration_min_holdout):
                    desired_calib_size = max(
                        int(calibration_policy.min_samples),
                        int(len(X_holdout) * float(calibration_policy.fraction_of_holdout)),
                    )
                    desired_calib_size = min(
                        desired_calib_size,
                        len(X_holdout) - int(calibration_policy.min_samples),
                    )
                    calib_size, calib_split_support_debug = self._select_calibration_size_with_support(
                        y_holdout=y_holdout,
                        default_calib_size=desired_calib_size,
                        min_calib_samples=int(calibration_policy.min_samples),
                        min_val_samples=int(calibration_policy.min_samples),
                        required_class_ids=critical_support_info.get("class_ids", []),
                        min_support=holdout_support_floor,
                    )
                    if (
                        calib_size is not None
                        and int(calib_size) >= int(calibration_policy.min_samples)
                    ):
                        X_val = X_holdout[:-calib_size]
                        y_val = y_holdout[:-calib_size]
                        ts_val = ts_holdout[:-calib_size]
                        X_calib = X_holdout[-calib_size:]
                        y_calib = y_holdout[-calib_size:]

                        validation_data = (X_val, y_val)
                        shadow_eval_timestamps = ts_val
                        calibration_data = (X_calib, y_calib)
                        validation_source = "temporal_holdout_split"
                        calibration_source = "temporal_calibration_split"
                        logger.info(
                            f"Using split for {room_name}: Train={len(X_train)}, "
                            f"Val={len(X_val)}, Calib={len(X_calib)}"
                        )
                    else:
                        validation_data = (X_holdout, y_holdout)
                        shadow_eval_timestamps = ts_holdout
                        calibration_data = (X_holdout, y_holdout)
                        validation_source = "temporal_holdout_split"
                        calibration_source = "validation_fallback"
                        logger.warning(
                            f"Calibration split fallback for {room_name}: "
                            f"holdout support could not satisfy separate val/calib guarantees "
                            f"({len(X_holdout)} samples). Details={calib_split_support_debug}"
                        )
                else:
                    validation_data = (X_holdout, y_holdout)
                    shadow_eval_timestamps = ts_holdout
                    calibration_data = (X_holdout, y_holdout)
                    validation_source = "temporal_holdout_split"
                    calibration_source = "validation_fallback"
                    logger.warning(
                        f"Calibration split fallback for {room_name}: "
                        f"using full holdout for validation+calibration ({len(X_holdout)} samples)"
                    )

                logger.info(
                    f"Using temporal validation split: Train={len(X_train)} "
                    f"({seq_timestamps[0]}..{seq_timestamps[split_idx-1]}), "
                    f"Holdout={len(X_holdout)} ({ts_holdout[0]}..{ts_holdout[-1]})"
                )
                sampling_reference_labels = y_holdout
            else:
                X_train, y_train = X_seq, y_seq
                ts_train = seq_timestamps
                validation_data = None
                validation_source = "none"
                calibration_source = "none"
                sampling_reference_labels = None
                logger.info(f"Training on full dataset ({len(X_seq)} samples)")

            train_class_support_pre_sampling = self._build_class_support_map(y_train)

            # Apply downsampling on train split only (post-temporal split).
            pre_downsample_count = int(len(X_train))
            X_train, y_train, ts_train = self._downsample_easy_unoccupied(
                X_train,
                y_train,
                ts_train,
                room_name,
                resolved_cfg=unoccupied_cfg,
                reference_labels=sampling_reference_labels,
            )
            downsample_removed = int(pre_downsample_count - len(X_train))
            train_class_support_post_downsample = self._build_class_support_map(y_train)

            replay_stats = {
                "total": int(len(X_train)),
                "corrected_kept": 0,
                "uncorrected_sampled": int(len(X_train)),
                "replay_ratio": 0,
            }
            if is_correction_fine_tune and fine_tune_params.get("replay_enabled", True):
                X_train, y_train, ts_train, replay_stats = self._apply_replay_sampling(
                    elder_id=elder_id,
                    room_name=room_name,
                    X_seq=X_train,
                    y_seq=y_train,
                    seq_timestamps=ts_train,
                    replay_ratio=int(fine_tune_params.get("replay_ratio", 10)),
                    sampling_strategy=str(fine_tune_params.get("replay_sampling", "random_stratified")),
                )
            train_samples_for_evidence = int(len(X_train))
            train_class_support_pre_transition_focus = self._build_class_support_map(y_train)
            transition_focus_stats = {
                "enabled": False,
                "added_samples": 0,
                "class_counts_before": dict(train_class_support_pre_transition_focus),
                "class_counts_after": dict(train_class_support_pre_transition_focus),
                "reason": "not_factorized_primary",
            }
            if training_strategy == "factorized_primary":
                X_train, y_train, transition_focus_stats = self._apply_transition_focus_sampling(
                    X_train=X_train,
                    y_train=y_train,
                    room_name=room_name,
                    reference_labels=sampling_reference_labels,
                )
            train_class_support_post_transition_focus = self._build_class_support_map(y_train)
            train_class_support_pre_minority_sampling = self._build_class_support_map(y_train)

            X_train, y_train, minority_sampling_stats = self._apply_minority_class_sampling(
                X_train=X_train,
                y_train=y_train,
                room_name=room_name,
                reference_labels=sampling_reference_labels,
            )
            train_class_support_post_minority_sampling = self._build_class_support_map(y_train)

            # Class Weights for Imbalance Handling (compute on training portion only).
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
            class_weight_dict = dict(zip(unique_classes.astype(int), class_weights))

            # Apply clinical-priority multipliers on top of balanced weights.
            # This improves recall pressure on meaningful ADLs (sleep/shower/toilet/etc.)
            # while preventing unstable extremes via floor/cap clipping.
            clinical_policy = active_policy.clinical_priority
            adjusted_class_weight_dict = {}
            label_encoder = self.platform.label_encoders.get(room_name)
            weight_debug = []

            for class_id, base_weight in class_weight_dict.items():
                label_name = None
                if label_encoder is not None:
                    classes = getattr(label_encoder, 'classes_', None)
                    if classes is not None and int(class_id) < len(classes):
                        label_name = str(classes[int(class_id)])

                priority_multiplier = clinical_policy.get_room_label_multiplier(room_name, label_name)
                adjusted_weight = float(base_weight) * float(priority_multiplier)
                adjusted_weight = max(
                    float(clinical_policy.class_weight_floor),
                    min(float(clinical_policy.class_weight_cap), adjusted_weight),
                )
                adjusted_class_weight_dict[int(class_id)] = adjusted_weight

                weight_debug.append({
                    'class_id': int(class_id),
                    'label': label_name or 'unknown',
                    'base_weight': float(base_weight),
                    'priority_multiplier': float(priority_multiplier),
                    'final_weight': float(adjusted_weight),
                })

            # Log class distribution and imbalance ratio
            logger.info(f"Class distribution for {room_name} (train): {dict(zip(unique_classes, class_counts))}")
            imbalance_ratio = max(class_counts) / min(class_counts) if min(class_counts) > 0 else float('inf')
            if imbalance_ratio > 10:
                logger.warning(f"Severe class imbalance in {room_name}: {imbalance_ratio:.1f}:1")
            logger.info(f"Base class weights: {class_weight_dict}")
            logger.info(f"Adjusted class weights (clinical priority): {adjusted_class_weight_dict}")
            logger.info(f"Class weight breakdown: {weight_debug}")

            timeline_native_weights, timeline_native_weighting = self._build_timeline_native_sequence_weights(
                room_name=room_name,
                y_train=y_train,
            )
            use_sample_weight = bool(timeline_native_weighting.get("enabled", False))
            sample_weight = None
            if use_sample_weight:
                base_weight_by_sample = np.asarray(
                    [adjusted_class_weight_dict.get(int(cls), 1.0) for cls in y_train],
                    dtype=np.float32,
                )
                sample_weight = np.asarray(base_weight_by_sample * timeline_native_weights, dtype=np.float32)
                logger.info(
                    f"Timeline-native weighting active for {room_name}: "
                    f"sample_weight mean={float(np.mean(sample_weight)):.3f}, "
                    f"min={float(np.min(sample_weight)):.3f}, max={float(np.max(sample_weight)):.3f}"
                )
            activity_weighting_diagnostics: Dict[str, Any] = {
                "activity_loss_mask_coverage": 0.0,
                "activity_loss_mask_floor": 0.0,
                "activity_class_weight_mode": "global",
                "activity_class_weight_support": int(len(y_train)),
                "activity_weight_mean_occupied": 0.0,
                "activity_weight_mean_masked": 0.0,
                "masked_class_ids": [],
            }
            activity_class_weight_dict = dict(adjusted_class_weight_dict)

            timeline_target_debug: Dict[str, Any] = {"enabled": False, "reason": "not_timeline_model"}
            timeline_validation_debug: Dict[str, Any] = {}
            timeline_targets_train: Dict[str, np.ndarray] = {}
            timeline_targets_val: Dict[str, np.ndarray] = {}
            checkpoint_selector: Optional[_GateAlignedCheckpointCallback] = None
            checkpoint_no_regress = self._resolve_no_regress_macro_f1_floor(
                room_name=room_name,
                champion_meta=champion_meta,
            )
            checkpoint_no_regress_floor = (
                float(checkpoint_no_regress.get("target_macro_f1_floor"))
                if bool(checkpoint_no_regress.get("enabled", False))
                and checkpoint_no_regress.get("target_macro_f1_floor") is not None
                else None
            )
            if timeline_multitask_enabled:
                timeline_targets_train, timeline_target_debug = self._build_timeline_targets(
                    room_name=room_name,
                    y_train=y_train,
                    seq_timestamps=ts_train,
                )
                if not bool(timeline_target_debug.get("enabled", False)):
                    logger.warning(
                        f"Timeline multitask disabled at runtime for {room_name}: {timeline_target_debug}"
                    )
                    timeline_multitask_enabled = False
                elif validation_data is not None:
                    y_val_for_timeline = np.asarray(validation_data[1], dtype=np.int32)
                    timeline_targets_val, timeline_validation_debug = self._build_timeline_targets(
                        room_name=room_name,
                        y_train=y_val_for_timeline,
                        seq_timestamps=shadow_eval_timestamps,
                    )
                    if not bool(timeline_validation_debug.get("enabled", False)):
                        logger.warning(
                            f"Timeline validation targets unavailable for {room_name}; "
                            f"proceeding without validation targets. Details={timeline_validation_debug}"
                        )
                        timeline_targets_val = {}

            if validation_data is not None and self._resolve_gate_aligned_checkpoint_enabled():
                checkpoint_selector = _GateAlignedCheckpointCallback(
                    pipeline=self,
                    room_name=room_name,
                    X_val=np.asarray(validation_data[0], dtype=np.float32),
                    y_val=np.asarray(validation_data[1], dtype=np.int32),
                    timeline_multitask_enabled=bool(timeline_multitask_enabled),
                    no_regress_macro_f1_floor=checkpoint_no_regress_floor,
                )
                callbacks.append(checkpoint_selector)

            if timeline_multitask_enabled:
                activity_labels_train = np.asarray(timeline_targets_train["activity_labels"], dtype=np.int32)
                occupancy_labels_train = np.asarray(timeline_targets_train["occupancy_labels"], dtype=np.float32).reshape(-1, 1)
                boundary_start_train = np.asarray(timeline_targets_train["boundary_start_labels"], dtype=np.float32).reshape(-1, 1)
                boundary_end_train = np.asarray(timeline_targets_train["boundary_end_labels"], dtype=np.float32).reshape(-1, 1)
                y_train_fit = {
                    "activity_logits": activity_labels_train,
                    "occupancy_logits": occupancy_labels_train,
                    "boundary_start_logits": boundary_start_train,
                    "boundary_end_logits": boundary_end_train,
                }

                activity_sample_weight, activity_class_weight_dict, activity_weighting_diagnostics = (
                    self._build_timeline_activity_loss_weights(
                        room_name=room_name,
                        y_train=y_train,
                        global_class_weight_dict=adjusted_class_weight_dict,
                        timeline_native_weights=timeline_native_weights,
                        use_timeline_native_weighting=use_sample_weight,
                        clinical_policy=clinical_policy,
                    )
                )
                logger.info(
                    "Timeline activity loss weighting for %s: mode=%s, mask_coverage=%.3f, "
                    "mask_floor=%.3f, mean_occupied=%.3f, mean_masked=%.3f",
                    room_name,
                    activity_weighting_diagnostics.get("activity_class_weight_mode", "global"),
                    float(activity_weighting_diagnostics.get("activity_loss_mask_coverage", 0.0)),
                    float(activity_weighting_diagnostics.get("activity_loss_mask_floor", 0.0)),
                    float(activity_weighting_diagnostics.get("activity_weight_mean_occupied", 0.0)),
                    float(activity_weighting_diagnostics.get("activity_weight_mean_masked", 0.0)),
                )
                sample_weight_fit = {
                    "activity_logits": activity_sample_weight,
                    "occupancy_logits": np.ones_like(activity_sample_weight, dtype=np.float32),
                    "boundary_start_logits": np.ones_like(activity_sample_weight, dtype=np.float32),
                    "boundary_end_logits": np.ones_like(activity_sample_weight, dtype=np.float32),
                }

                validation_fit = None
                if validation_data is not None and timeline_targets_val:
                    X_val_fit = validation_data[0]
                    y_val_fit = {
                        "activity_logits": np.asarray(timeline_targets_val["activity_labels"], dtype=np.int32),
                        "occupancy_logits": np.asarray(timeline_targets_val["occupancy_labels"], dtype=np.float32).reshape(-1, 1),
                        "boundary_start_logits": np.asarray(timeline_targets_val["boundary_start_labels"], dtype=np.float32).reshape(-1, 1),
                        "boundary_end_logits": np.asarray(timeline_targets_val["boundary_end_labels"], dtype=np.float32).reshape(-1, 1),
                    }
                    validation_fit = (X_val_fit, y_val_fit)

                history = model.fit(
                    X_train,
                    y_train_fit,
                    epochs=max_epochs,
                    batch_size=32,
                    validation_data=validation_fit,
                    shuffle=bool(train_batch_shuffle_enabled),
                    verbose=2,
                    class_weight=None,
                    sample_weight=sample_weight_fit,
                    callbacks=callbacks,
                )
            elif validation_data is not None:
                history = model.fit(
                    X_train,
                    y_train,
                    epochs=max_epochs,
                    batch_size=32,
                    validation_data=validation_data,
                    shuffle=bool(train_batch_shuffle_enabled),
                    verbose=2,
                    class_weight=None if use_sample_weight else adjusted_class_weight_dict,
                    sample_weight=sample_weight,
                    callbacks=callbacks,
                )
            else:
                history = model.fit(
                    X_train,
                    y_train,
                    epochs=max_epochs,
                    batch_size=32,
                    shuffle=bool(train_batch_shuffle_enabled),
                    verbose=2,
                    class_weight=None if use_sample_weight else adjusted_class_weight_dict,
                    sample_weight=sample_weight,
                    callbacks=callbacks,
                )

            post_fit_train_summary: Optional[Dict[str, Any]] = None
            prediction_audit: Optional[Dict[str, Any]] = None
            prediction_audit_enabled = self._is_prediction_audit_enabled(room_name)
            if train_batch_shuffle_enabled or prediction_audit_enabled:
                if prediction_audit_enabled:
                    prediction_audit = {
                        "enabled": True,
                        "history": {
                            key: [float(value) for value in values]
                            for key, values in (history.history or {}).items()
                            if isinstance(values, list)
                        },
                        "restored_best_epoch": (
                            int(getattr(early_stopping, "best_epoch", -1)) + 1
                            if getattr(early_stopping, "best_epoch", None) is not None
                            else None
                        ),
                    }
                try:
                    train_probs_audit = self._predict_activity_probabilities(
                        model=model,
                        X=np.asarray(X_train, dtype=np.float32),
                        timeline_multitask_enabled=bool(timeline_multitask_enabled),
                    )
                    post_fit_train_summary = self._summarize_prediction_audit(
                        room_name=room_name,
                        y_true=y_train,
                        y_pred_probs=train_probs_audit,
                    )
                    if prediction_audit is not None:
                        prediction_audit["post_fit_in_memory"] = {
                            "train": post_fit_train_summary,
                        }
                    if validation_data is not None:
                        val_probs_audit = self._predict_activity_probabilities(
                            model=model,
                            X=np.asarray(validation_data[0], dtype=np.float32),
                            timeline_multitask_enabled=bool(timeline_multitask_enabled),
                        )
                        if prediction_audit is not None:
                            prediction_audit["post_fit_in_memory"]["validation"] = (
                                self._summarize_prediction_audit(
                                    room_name=room_name,
                                    y_true=validation_data[1],
                                    y_pred_probs=val_probs_audit,
                                )
                            )
                except Exception as e:
                    if prediction_audit is not None:
                        prediction_audit["error"] = f"{type(e).__name__}: {e}"
                    logger.warning(f"Prediction audit failed for {room_name}: {e}")

            two_stage_result = self._train_two_stage_core_models(
                room_name=room_name,
                seq_length=seq_length,
                X_train=np.asarray(X_train, dtype=np.float32),
                y_train=np.asarray(y_train, dtype=np.int32),
                validation_data=(
                    (
                        np.asarray(validation_data[0], dtype=np.float32),
                        np.asarray(validation_data[1], dtype=np.int32),
                    )
                    if validation_data is not None
                    else None
                ),
                max_epochs=max_epochs,
            )
            if bool(two_stage_result.get("enabled", False)):
                if training_strategy == "factorized_primary":
                    two_stage_result["gate_mode"] = "primary"
                    two_stage_result["training_strategy"] = "factorized_primary"
                stage_a_calibration = self._calibrate_two_stage_stage_a_threshold(
                    room_name=room_name,
                    two_stage_result=two_stage_result,
                    calibration_data=calibration_data or validation_data,
                )
                two_stage_result["stage_a_occupied_threshold"] = float(
                    stage_a_calibration.get(
                        "threshold", self._resolve_two_stage_stage_a_default_threshold()
                    )
                )
                two_stage_result["stage_a_threshold_source"] = str(
                    stage_a_calibration.get("source", "policy_default")
                )
                two_stage_result["stage_a_calibration"] = stage_a_calibration
            
            # Capture metrics - comprehensive reporting
            acc_history = history.history.get("accuracy")
            if acc_history is None:
                acc_history = history.history.get("activity_logits_accuracy")
            if acc_history is None:
                acc_history = [0.0]
            final_acc = float(acc_history[-1])
            reported_accuracy, accuracy_source = self._resolve_reported_accuracy(
                fit_history_accuracy=final_acc,
                post_fit_train_summary=post_fit_train_summary,
            )
            
            # Compute validation metrics if we had a validation split
            metrics = {
                'room': room_name,
                'accuracy': float(reported_accuracy),
                'fit_history_accuracy': float(final_acc),
                'accuracy_source': str(accuracy_source),
                'epochs': len(acc_history),  # Actual epochs (may be less with early stopping)
                'samples': int(train_samples_for_evidence),
                'training_strategy': training_strategy,
                'train_batch_shuffle_enabled': bool(train_batch_shuffle_enabled),
                'imbalance_ratio': float(imbalance_ratio) if imbalance_ratio != float('inf') else None,
                'training_mode': training_mode,
                'warm_start_used': bool(did_warm_start),
                'learning_rate': float(learning_rate),
                'shared_backbone_enabled': bool(shared_enabled),
                'shared_backbone_id': effective_shared_backbone_id or None,
                'shared_backbone_loaded_layers': int(shared_backbone_loaded_layers),
                'adapter_training_only': bool(adapter_training_only),
                'adapter_freeze_summary': freeze_summary,
                'replay': replay_stats,
                'unoccupied_downsample': dict(self._last_unoccupied_downsample_debug),
                'transition_focus_sampling': transition_focus_stats,
                'minority_sampling': minority_sampling_stats,
                'train_class_support_pre_sampling': train_class_support_pre_sampling,
                'train_class_support_post_downsample': train_class_support_post_downsample,
                'train_class_support_pre_transition_focus': train_class_support_pre_transition_focus,
                'train_class_support_post_transition_focus': train_class_support_post_transition_focus,
                'train_class_support_pre_minority_sampling': train_class_support_pre_minority_sampling,
                'train_class_support_post_minority_sampling': train_class_support_post_minority_sampling,
                'policy_hash': policy_hash,
                'metric_source': 'holdout_validation',
                'validation_source': validation_source,
                'calibration_source': calibration_source,
                'validation_samples': int(len(validation_data[0])) if validation_data is not None else 0,
                'calibration_samples': int(len(calibration_data[0])) if calibration_data is not None else 0,
                'validation_min_class_support': 0,
                'validation_class_support': {},
                'holdout_min_class_support': 0,
                'holdout_class_support': {},
                'required_minority_support': int(holdout_support_floor),
                'insufficient_validation_evidence': False,
                'timeline_native_weighting': timeline_native_weighting,
                'activity_loss_mask_coverage': float(
                    activity_weighting_diagnostics.get("activity_loss_mask_coverage", 0.0)
                ),
                'activity_loss_mask_floor': float(
                    activity_weighting_diagnostics.get("activity_loss_mask_floor", 0.0)
                ),
                'activity_class_weight_mode': str(
                    activity_weighting_diagnostics.get("activity_class_weight_mode", "global")
                ),
                'activity_class_weight_support': int(
                    activity_weighting_diagnostics.get("activity_class_weight_support", len(y_train))
                ),
                'activity_weight_mean_occupied': float(
                    activity_weighting_diagnostics.get("activity_weight_mean_occupied", 0.0)
                ),
                'activity_weight_mean_masked': float(
                    activity_weighting_diagnostics.get("activity_weight_mean_masked", 0.0)
                ),
                'timeline_multitask': {
                    'enabled': bool(timeline_multitask_enabled),
                    'targets': timeline_target_debug,
                    'validation_targets': timeline_validation_debug,
                    'activity_weighting': activity_weighting_diagnostics,
                },
                'checkpoint_selection': (
                    checkpoint_selector.to_dict()
                    if checkpoint_selector is not None
                    else {"enabled": False}
                ),
                'checkpoint_no_regress': dict(checkpoint_no_regress),
                'two_stage_core': {
                    'enabled': bool(two_stage_result.get('enabled', False)),
                    'reason': str(two_stage_result.get('reason', 'disabled')),
                    'gate_mode': str(two_stage_result.get('gate_mode', self._resolve_two_stage_gate_mode())),
                    'stage_b_enabled': bool(two_stage_result.get('stage_b_enabled', False)),
                    'stage_b_reason': str(two_stage_result.get('stage_b_reason', '')),
                    'excluded_class_ids': list(two_stage_result.get('excluded_class_ids', [])),
                    'occupied_class_ids': list(two_stage_result.get('occupied_class_ids', [])),
                    'primary_occupied_class_id': two_stage_result.get('primary_occupied_class_id'),
                    'stage_a_occupied_threshold': float(
                        two_stage_result.get(
                            'stage_a_occupied_threshold',
                            self._resolve_two_stage_stage_a_default_threshold(),
                        )
                    ),
                    'stage_a_threshold_source': str(
                        two_stage_result.get('stage_a_threshold_source', 'policy_default')
                    ),
                    'stage_a_calibration': dict(two_stage_result.get('stage_a_calibration', {})),
                },
            }
            if isinstance(self._seed_panel_debug_context, dict):
                metrics["seed_panel_debug"] = dict(self._seed_panel_debug_context)
            metrics["split_support"] = split_support_debug
            metrics["calibration_split_support"] = calib_split_support_debug
            metrics["critical_labels"] = list(critical_support_info.get("labels", []))
            metrics["total_sequences_pre_split"] = int(total_sequences_pre_split)
            metrics["train_samples_pre_downsample"] = int(pre_downsample_count)
            metrics["train_samples_post_downsample"] = int(pre_downsample_count - downsample_removed)
            metrics["train_samples_post_transition_focus"] = int(
                len(X_train) - int(minority_sampling_stats.get("added_samples", 0) or 0)
            )
            metrics["train_samples_post_minority_sampling"] = int(len(X_train))
            raw_rows = int(processed_df.attrs.get("raw_rows_before_resample", len(processed_df)))
            rows_after_gap_drop = int(processed_df.attrs.get("rows_after_gap_drop", len(processed_df)))
            retained_ratio = (
                float(rows_after_gap_drop) / float(raw_rows)
                if raw_rows > 0 else 0.0
            )
            metrics["raw_rows_before_resample"] = raw_rows
            metrics["rows_after_gap_drop"] = rows_after_gap_drop
            metrics["retained_sample_ratio"] = float(retained_ratio)
            metrics["observed_day_count"] = int(processed_df.attrs.get("observed_day_count", 0) or 0)
            metrics["data_viability"] = dict(processed_df.attrs.get("data_viability", {}))
            if prediction_audit is not None:
                metrics["prediction_audit"] = prediction_audit
            
            # Add per-class metrics if validation was performed
            if val_split > 0:
                holdout_classes, holdout_counts = np.unique(y_holdout, return_counts=True)
                holdout_support_map = {str(int(c)): int(n) for c, n in zip(holdout_classes, holdout_counts)}
                metrics["holdout_class_support"] = holdout_support_map
                metrics["holdout_min_class_support"] = (
                    min(holdout_support_map.values()) if holdout_support_map else 0
                )
                y_val_eval = validation_data[1]
                val_classes, val_counts = np.unique(y_val_eval, return_counts=True)
                val_support_map = {str(int(c)): int(n) for c, n in zip(val_classes, val_counts)}
                metrics["validation_class_support"] = val_support_map
                metrics["validation_min_class_support"] = min(val_support_map.values()) if val_support_map else 0
                y_pred_single = model.predict(validation_data[0], verbose=0)
                y_pred = self._extract_activity_probabilities(
                    y_pred_single,
                    timeline_multitask_enabled=bool(timeline_multitask_enabled),
                )

                if bool(two_stage_result.get("enabled", False)):
                    gate_mode = str(
                        (metrics.get("two_stage_core") or {}).get("gate_mode", self._resolve_two_stage_gate_mode())
                    ).strip().lower()
                    try:
                        y_pred_two_stage = self._predict_two_stage_core_probabilities(
                            two_stage_result=two_stage_result,
                            X=np.asarray(validation_data[0], dtype=np.float32),
                        )
                        y_pred_two_stage_classes = np.argmax(y_pred_two_stage, axis=1)
                        report_two_stage = classification_report(
                            y_val_eval,
                            y_pred_two_stage_classes,
                            output_dict=True,
                            zero_division=0,
                        )
                        two_stage_payload = {
                            "macro_f1": float(report_two_stage.get("macro avg", {}).get("f1-score", 0.0)),
                            "macro_recall": float(report_two_stage.get("macro avg", {}).get("recall", 0.0)),
                            "macro_precision": float(report_two_stage.get("macro avg", {}).get("precision", 0.0)),
                        }
                        selection = self._select_final_two_stage_gate_source(
                            room_name=room_name,
                            y_true=np.asarray(y_val_eval, dtype=np.int32),
                            single_stage_probs=np.asarray(y_pred, dtype=np.float32),
                            two_stage_probs=np.asarray(y_pred_two_stage, dtype=np.float32),
                            gate_mode=gate_mode,
                            champion_meta=champion_meta,
                        )
                        metrics["two_stage_core"]["single_stage_validation"] = dict(
                            selection.get("single_stage") or {}
                        )
                        metrics["two_stage_core"]["two_stage_validation"] = dict(
                            selection.get("two_stage") or {}
                        )
                        metrics["two_stage_core"]["validation_summary"] = dict(
                            selection.get("selected_summary") or two_stage_payload
                        )
                        metrics["two_stage_core"]["validation_summary_raw_two_stage"] = dict(two_stage_payload)
                        metrics["two_stage_core"]["gate_source"] = str(
                            selection.get("selected_source") or "single_stage_shadow"
                        )
                        metrics["two_stage_core"]["runtime_use_two_stage"] = bool(
                            selection.get("runtime_use_two_stage", False)
                        )
                        two_stage_result["runtime_enabled"] = bool(
                            selection.get("runtime_use_two_stage", False)
                        )
                        if bool(selection.get("runtime_use_two_stage", False)):
                            y_pred = y_pred_two_stage
                            if training_strategy == "factorized_primary":
                                metrics["two_stage_core"]["gate_source"] = "factorized_primary"
                                metrics["metric_source"] = "holdout_validation_factorized_primary"
                                two_stage_result["runtime_gate_source"] = "factorized_primary"
                            else:
                                metrics["two_stage_core"]["gate_source"] = str(
                                    selection.get("selected_source") or "two_stage_primary"
                                )
                                metrics["metric_source"] = "holdout_validation_two_stage_primary"
                                two_stage_result["runtime_gate_source"] = str(
                                    selection.get("selected_source") or "two_stage_primary"
                                )
                        else:
                            metrics["two_stage_core"]["gate_source"] = str(
                                selection.get("selected_source") or "single_stage_shadow"
                            )
                            metrics["metric_source"] = (
                                "holdout_validation_single_stage_fallback"
                                if gate_mode == "primary"
                                else "holdout_validation_single_stage_shadow"
                            )
                            two_stage_result["runtime_gate_source"] = str(
                                selection.get("selected_source") or "single_stage_shadow"
                            )
                    except Exception as e:
                        metrics["two_stage_core"]["error"] = f"{type(e).__name__}: {e}"
                        metrics["two_stage_core"]["gate_source"] = "single_stage_fallback_error"
                        metrics["two_stage_core"]["runtime_use_two_stage"] = False
                        two_stage_result["runtime_enabled"] = False
                        two_stage_result["runtime_gate_source"] = "single_stage_fallback_error"
                        logger.warning(f"Two-stage validation fallback for {room_name}: {e}")

                shadow_eval_y_true = np.asarray(y_val_eval, dtype=np.int32)
                shadow_eval_probs = None
                label_encoder = self.platform.label_encoders.get(room_name)
                classes = getattr(label_encoder, "classes_", None) if label_encoder is not None else None
                
                try:
                    y_pred_arr = np.asarray(y_pred)
                    if y_pred_arr.ndim != 2 or y_pred_arr.shape[0] != len(y_val_eval):
                        raise ValueError(
                            f"unexpected_validation_prediction_shape:{getattr(y_pred_arr, 'shape', None)}"
                        )
                    y_pred_classes = np.argmax(y_pred_arr, axis=1)
                    shadow_eval_probs = np.asarray(y_pred_arr, dtype=np.float32)
                    metrics.update(
                        self._build_beta6_parity_payload(
                            room_name=room_name,
                            y_pred_probs=y_pred_arr,
                        )
                    )

                    class_ids = sorted(
                        set(np.unique(y_val_eval).astype(int)) | set(np.unique(y_pred_classes).astype(int))
                    )
                    if classes is not None:
                        class_ids = [int(cid) for cid in class_ids if 0 <= int(cid) < len(classes)]
                    if class_ids:
                        cm = confusion_matrix(y_val_eval, y_pred_classes, labels=class_ids).astype(int)
                        labels = []
                        true_dist: Dict[str, int] = {}
                        pred_dist: Dict[str, int] = {}
                        for idx, class_id in enumerate(class_ids):
                            label_name = (
                                str(classes[class_id]).strip().lower()
                                if classes is not None and 0 <= int(class_id) < len(classes)
                                else f"class_{int(class_id)}"
                            )
                            labels.append(label_name)
                            true_dist[label_name] = int(cm[idx, :].sum())
                            pred_dist[label_name] = int(cm[:, idx].sum())
                        metrics["confusion_matrix"] = {
                            "class_ids": [int(cid) for cid in class_ids],
                            "labels": labels,
                            "matrix": cm.tolist(),
                        }
                        metrics["true_class_distribution"] = true_dist
                        metrics["predicted_class_distribution"] = pred_dist

                    report = classification_report(y_val_eval, y_pred_classes, output_dict=True, zero_division=0)
                    metrics['macro_f1'] = float(report['macro avg']['f1-score'])
                    metrics['macro_recall'] = float(report['macro avg']['recall'])
                    metrics['macro_precision'] = float(report['macro avg']['precision'])
                    per_label_recall = {}
                    per_label_support = {}
                    if classes is not None:
                        for class_id in sorted(np.unique(y_val_eval).astype(int)):
                            if class_id < 0 or class_id >= len(classes):
                                continue
                            label_name = str(classes[class_id]).strip().lower()
                            class_key = str(int(class_id))
                            label_metrics = report.get(class_key, {})
                            per_label_recall[label_name] = float(label_metrics.get("recall", 0.0))
                            per_label_support[label_name] = int(label_metrics.get("support", 0) or 0)
                    for critical_label in self._resolve_critical_labels(room_name):
                        per_label_support.setdefault(str(critical_label).strip().lower(), 0)
                    metrics["per_label_recall"] = per_label_recall
                    metrics["per_label_support"] = per_label_support
                    logger.info(f"Validation metrics for {room_name}: F1={metrics['macro_f1']:.3f}, Recall={metrics['macro_recall']:.3f}")
                except Exception as e:
                    logger.warning(f"Could not compute detailed metrics: {e}")

                split_found = bool(split_support_debug.get("found", True)) if split_support_debug else True
                validation_min_support = int(metrics.get("validation_min_class_support", 0) or 0)
                insufficient_holdout_support = (
                    validation_min_support <= 0
                    or validation_min_support < int(holdout_support_floor)
                    or not split_found
                )
                if insufficient_holdout_support:
                    metrics["insufficient_validation_evidence"] = True
                    metrics["insufficient_validation_evidence_reason"] = (
                        f"validation_min_support={validation_min_support}<required={int(holdout_support_floor)}"
                    )
                    metrics["metric_source"] = "insufficient_validation_evidence"
                    for key in ("macro_f1", "macro_recall", "macro_precision"):
                        value = metrics.get(key)
                        if value is not None:
                            metrics[f"{key}_raw"] = float(value)
                            metrics[key] = None
                    logger.warning(
                        f"Insufficient validation evidence for {room_name}: "
                        f"support={validation_min_support}, required={int(holdout_support_floor)}, "
                        f"split_found={split_found}; quality metrics masked."
                    )
            else:
                # Bootstrap fallback: when holdout split is disabled for small datasets,
                # compute macro metrics on training split so gating is not blocked by missing macro_f1.
                try:
                    y_train_pred = model.predict(X_train, verbose=0)
                    if isinstance(y_train_pred, dict):
                        y_train_pred = y_train_pred.get("activity_logits")
                    if y_train_pred is None:
                        raise ModelTrainError(f"Missing activity logits for fallback predictions ({room_name})")
                    if bool(timeline_multitask_enabled):
                        y_train_pred = tf.nn.softmax(np.asarray(y_train_pred), axis=1).numpy()
                    y_train_pred_classes = np.argmax(y_train_pred, axis=1)
                    report = classification_report(y_train, y_train_pred_classes, output_dict=True, zero_division=0)
                    metrics['macro_f1'] = float(report['macro avg']['f1-score'])
                    metrics['macro_recall'] = float(report['macro avg']['recall'])
                    metrics['macro_precision'] = float(report['macro avg']['precision'])
                    metrics['metric_source'] = 'train_fallback_small_dataset'
                    logger.info(
                        f"Small-dataset fallback metrics for {room_name}: "
                        f"F1={metrics['macro_f1']:.3f}, Recall={metrics['macro_recall']:.3f}"
                    )
                except Exception as e:
                    logger.warning(f"Could not compute small-dataset fallback metrics: {e}")

            # Calibrate per-class thresholds on calibration split.
            class_thresholds = self._default_class_thresholds(room_name)
            if calibration_data is not None:
                class_thresholds = self._calibrate_class_thresholds(
                    model=model,
                    X_calib=calibration_data[0],
                    y_calib=calibration_data[1],
                    room_name=room_name,
                )
            else:
                self._last_calibration_debug = []
                logger.warning(
                    f"No calibration split available for {room_name}; "
                    f"using default threshold={DEFAULT_CONFIDENCE_THRESHOLD}"
                )
            metrics['class_thresholds'] = class_thresholds
            calib_supports = [
                int(entry.get("support", 0) or 0)
                for entry in self._last_calibration_debug
                if isinstance(entry, dict)
            ]
            metrics["calibration_min_support"] = min(calib_supports) if calib_supports else 0
            low_support_entries = [
                entry for entry in self._last_calibration_debug
                if isinstance(entry, dict) and str(entry.get("status", "")) == "fallback_low_support"
            ]
            metrics["calibration_low_support_count"] = int(len(low_support_entries))
            metrics["calibration_low_support_labels"] = [
                str(entry.get("label", "unknown")) for entry in low_support_entries
            ]
            if not isinstance(getattr(self.platform, 'class_thresholds', None), dict):
                self.platform.class_thresholds = {}
            
            # Capture data timestamp range for logging
            data_start_time = None
            data_end_time = None
            if 'timestamp' in processed_df.columns:
                data_start_time = processed_df['timestamp'].min()
                data_end_time = processed_df['timestamp'].max()
            elif hasattr(processed_df.index, 'min'):
                # Timestamp might be the index
                try:
                    data_start_time = processed_df.index.min()
                    data_end_time = processed_df.index.max()
                except:
                    pass

            training_days = 0.0
            if data_start_time is not None and data_end_time is not None:
                try:
                    span_seconds = (pd.to_datetime(data_end_time) - pd.to_datetime(data_start_time)).total_seconds()
                    training_days = max(0.0, float(span_seconds) / 86400.0)
                except Exception:
                    training_days = 0.0
            metrics['training_days'] = training_days

            # PR-2: Post-training gate evaluation (StatisticalValidityGate)
            calibration_support = {}
            if self._last_calibration_debug:
                for entry in self._last_calibration_debug:
                    if isinstance(entry, dict):
                        class_id = entry.get("class_id")
                        support = entry.get("support", 0)
                        if class_id is not None:
                            calibration_support[str(int(class_id))] = int(support)
            
            validation_class_support = {}
            raw_val_support = metrics.get("validation_class_support", {})
            if isinstance(raw_val_support, dict):
                for class_id, count in raw_val_support.items():
                    try:
                        validation_class_support[str(int(class_id))] = int(count)
                    except (TypeError, ValueError):
                        continue
            
            # Build pre-training gate result if provided
            pre_training_gate_result = None
            if gate_evaluation_result:
                pre_stack = []
                for g in gate_evaluation_result.get("gate_stack", []):
                    if isinstance(g, PreTrainingGateResult):
                        pre_stack.append(g)
                        continue
                    if not isinstance(g, dict):
                        continue
                    pre_stack.append(
                        PreTrainingGateResult(
                            passes=bool(g.get("passes", g.get("passed", False))),
                            gate_name=str(g.get("gate_name", "unknown")),
                            timestamp=str(g.get("timestamp", _utc_now_iso_z())),
                            details=dict(g.get("details", {}) or {}),
                            reason=g.get("reason"),
                        )
                    )
                pre_training_gate_result = GateEvaluationResult(
                    run_id=gate_evaluation_result.get("run_id", str(uuid.uuid4())),
                    elder_id=elder_id,
                    room_name=room_name,
                    pre_training_pass=bool(
                        gate_evaluation_result.get(
                            "pre_training_pass",
                            gate_evaluation_result.get("gate_pass", True),
                        )
                    ),
                    post_training_pass=True,  # Will be updated
                    overall_pass=True,
                    gate_stack=pre_stack,
                )
            
            # Evaluate post-training gates
            gate_pipeline = GateIntegrationPipeline(policy=active_policy)
            post_training_result = gate_pipeline.evaluate_post_training_gates(
                room_name=room_name,
                elder_id=elder_id,
                run_id=pre_training_gate_result.run_id if pre_training_gate_result else str(uuid.uuid4()),
                calibration_support=calibration_support,
                existing_gate_result=pre_training_gate_result or GateEvaluationResult(
                    run_id=str(uuid.uuid4()),
                    elder_id=elder_id,
                    room_name=room_name,
                    pre_training_pass=True,
                    post_training_pass=True,
                    overall_pass=True,
                    gate_stack=[],
                ),
                validation_class_support=validation_class_support,
            )
            
            # Merge post-training gate results with metrics
            metrics['gate_stack'] = [g.to_dict() for g in post_training_result.gate_stack]
            metrics['gate_pre_training_pass'] = post_training_result.pre_training_pass
            metrics['gate_post_training_pass'] = post_training_result.post_training_pass
            
            # Evaluate release gate (legacy)
            gate_pass, gate_reasons = self._evaluate_release_gate(
                room_name=room_name,
                candidate_metrics=metrics,
                champion_meta=champion_meta,
            )
            gate_watch_reasons = list(
                dict.fromkeys(str(r) for r in (self._last_release_gate_watch_reasons or []) if str(r).strip())
            )
            grouped_regime_gate = self._evaluate_grouped_regime_gate(
                room_name=room_name,
                grouped_fragility=metrics.get("grouped_fragility"),
                policy=active_policy,
            )
            metrics["grouped_regime_gate"] = dict(grouped_regime_gate)
            if not bool(grouped_regime_gate.get("pass", True)):
                gate_pass = False
                gate_reasons.extend(
                    [
                        str(reason).strip()
                        for reason in grouped_regime_gate.get("blocking_reasons", [])
                        if str(reason).strip()
                    ]
                )

            # Lane B: Event gate hardening on per-label holdout metrics.
            lane_b_gate_pass, lane_b_reasons, lane_b_gate_report = self._evaluate_lane_b_event_gates(
                room_name=room_name,
                candidate_metrics=metrics,
                target_date=pd.to_datetime(data_end_time) if data_end_time is not None else None,
            )
            metrics["lane_b_gate"] = lane_b_gate_report
            if not lane_b_gate_pass:
                gate_pass = False
                gate_reasons.extend(lane_b_reasons)
            
            # PR-2: If post-training gates failed, override gate_pass
            if not post_training_result.post_training_pass:
                gate_pass = False
                gate_reasons.extend([
                    g.reason for g in post_training_result.gate_stack 
                    if g.gate_name == "StatisticalValidityGate" and g.reason
                ])
                
                # Save rejection artifact
                if post_training_result.rejection_artifact:
                    models_dir = self.registry.get_models_dir(elder_id)
                    rejection_path = models_dir / f"{room_name}_{post_training_result.run_id}_why_rejected.json"
                    try:
                        with open(rejection_path, 'w') as f:
                            json.dump(post_training_result.rejection_artifact, f, indent=2, default=str)
                        logger.info(f"Saved post-training rejection artifact to {rejection_path}")
                        metrics["rejection_artifact_path"] = str(rejection_path)
                    except Exception as e:
                        logger.error(f"Failed to save rejection artifact: {e}")
            if gate_reasons:
                # Keep reason ordering stable and avoid duplicates in persisted artifacts.
                gate_reasons = list(dict.fromkeys(str(r) for r in gate_reasons))
            if gate_watch_reasons:
                gate_watch_reasons = list(dict.fromkeys(str(r) for r in gate_watch_reasons))
            parent_version_id = None
            if is_correction_fine_tune and did_warm_start and champion_meta:
                try:
                    parent_version_id = int(champion_meta.get("version"))
                except (TypeError, ValueError):
                    parent_version_id = None

            metrics['gate_pass'] = gate_pass
            metrics['gate_reasons'] = gate_reasons
            metrics['gate_watch_reasons'] = gate_watch_reasons
            metrics['gate_blocking_reasons'] = list(gate_reasons)
            metrics['parent_version_id'] = parent_version_id
            model_identity = self._resolve_model_identity(
                elder_id=elder_id,
                room_name=room_name,
                training_mode=training_mode,
                did_warm_start=did_warm_start,
                champion_meta=champion_meta,
                parent_version_id=parent_version_id,
                active_backbone_id=effective_shared_backbone_id,
            )
            metrics["model_identity"] = model_identity
            shadow_artifact_payload = None
            if event_first_shadow_enabled:
                try:
                    shadow_result = self._evaluate_event_first_shadow(
                        room_name=room_name,
                        y_true=shadow_eval_y_true,
                        y_pred_probs=shadow_eval_probs,
                        timestamps=shadow_eval_timestamps,
                        target_date=pd.to_datetime(data_end_time) if data_end_time is not None else None,
                    )
                except Exception as e:
                    shadow_result = {
                        "enabled": True,
                        "evaluated": False,
                        "reason": f"shadow_evaluation_error:{e}",
                    }
                if shadow_result:
                    if isinstance(shadow_result.get("summary"), dict):
                        summary_payload = dict(shadow_result["summary"])
                        summary_payload["enabled"] = bool(shadow_result.get("enabled", True))
                        summary_payload["evaluated"] = bool(shadow_result.get("evaluated", False))
                        metrics["event_first_shadow"] = summary_payload
                    else:
                        metrics["event_first_shadow"] = {
                            k: v for k, v in shadow_result.items() if k != "artifact"
                        }
                    shadow_artifact_payload = shadow_result.get("artifact")

            promote_to_latest = bool(gate_pass and not defer_promotion)

            # Save artifacts via registry (with versioning); optionally defer promotion.
            saved_version = self.registry.save_model_artifacts(
                elder_id,
                room_name,
                model,
                self.platform.scalers[room_name],
                self.platform.label_encoders[room_name],
                accuracy=float(reported_accuracy),
                samples=int(train_samples_for_evidence),
                class_thresholds=class_thresholds,
                metrics=metrics,
                model_identity=model_identity,
                promote_to_latest=promote_to_latest,
                parent_version_id=parent_version_id,
            )
            metrics["saved_version"] = int(saved_version)
            metrics["promotion_deferred"] = bool(defer_promotion and gate_pass)
            metrics["promoted_to_latest"] = bool(promote_to_latest)
            if prediction_audit is not None:
                try:
                    reloaded_model_path = (
                        self.registry.get_models_dir(elder_id) / f"{room_name}_v{int(saved_version)}_model.keras"
                    )
                    reloaded_model = self.registry.load_room_model(
                        str(reloaded_model_path),
                        room_name,
                        compile_model=False,
                    )
                    reloaded_payload: Dict[str, Any] = {
                        "train": self._summarize_prediction_audit(
                            room_name=room_name,
                            y_true=y_train,
                            y_pred_probs=self._predict_activity_probabilities(
                                model=reloaded_model,
                                X=np.asarray(X_train, dtype=np.float32),
                                timeline_multitask_enabled=bool(timeline_multitask_enabled),
                            ),
                        )
                    }
                    if validation_data is not None:
                        reloaded_payload["validation"] = self._summarize_prediction_audit(
                            room_name=room_name,
                            y_true=validation_data[1],
                            y_pred_probs=self._predict_activity_probabilities(
                                model=reloaded_model,
                                X=np.asarray(validation_data[0], dtype=np.float32),
                                timeline_multitask_enabled=bool(timeline_multitask_enabled),
                            ),
                        )
                    metrics["prediction_audit"]["reloaded_saved_model"] = reloaded_payload
                except Exception as e:
                    metrics["prediction_audit"]["reload_error"] = f"{type(e).__name__}: {e}"
                    logger.warning(f"Prediction audit reload failed for {room_name}: {e}")
            if bool(two_stage_result.get("enabled", False)):
                try:
                    two_stage_artifacts = self._write_two_stage_core_artifacts(
                        elder_id=elder_id,
                        room_name=room_name,
                        saved_version=int(saved_version),
                        two_stage_result=two_stage_result,
                        promote_to_latest=bool(promote_to_latest),
                    )
                    if two_stage_artifacts:
                        metrics["two_stage_core"]["artifact_paths"] = two_stage_artifacts
                        metrics["two_stage_core"]["saved_version"] = int(saved_version)
                except Exception as e:
                    metrics["two_stage_core"]["artifact_error"] = f"{type(e).__name__}: {e}"
                    logger.warning(
                        f"Failed writing two-stage artifacts for {elder_id}/{room_name} v{saved_version}: {e}"
                    )
            if shadow_artifact_payload is not None:
                shadow_paths = self._write_event_first_shadow_artifact(
                    elder_id=elder_id,
                    room_name=room_name,
                    saved_version=int(saved_version),
                    payload=shadow_artifact_payload,
                )
                if isinstance(metrics.get("event_first_shadow"), dict):
                    metrics["event_first_shadow"]["artifact_paths"] = shadow_paths
            runtime_topology = self._resolve_room_runtime_topology_expectation(
                elder_id=elder_id,
                room_name=room_name,
                saved_version=int(saved_version),
                runtime_use_two_stage=bool(
                    ((metrics.get("two_stage_core") or {}).get("runtime_use_two_stage", False))
                ),
            )
            room_status = self._resolve_fragile_room_status(
                gate_pass=bool(gate_pass),
                runtime_topology_matches=bool(runtime_topology.get("matches", True)),
                blocking_reasons=list(gate_reasons),
            )
            metrics["runtime_topology"] = dict(runtime_topology)
            metrics["room_status"] = str(room_status)
            decision_trace_payload = {
                "created_at_utc": _utc_now_iso_z(),
                "elder_id": elder_id,
                "room": room_name,
                "saved_version": int(saved_version),
                "training_mode": training_mode,
                "training_strategy": training_strategy,
                "policy_hash": policy_hash,
                "policy_source": "env" if self._use_env_policy else "injected",
                "policy": active_policy.to_dict(),
                "resolved": {
                    "unoccupied_downsample": unoccupied_cfg,
                    "unoccupied_downsample_debug": dict(self._last_unoccupied_downsample_debug),
                    "minority_sampling": {
                        "enabled": bool(minority_sampling_stats.get("enabled", False)),
                        "target_share": minority_sampling_stats.get("target_share"),
                        "max_multiplier": minority_sampling_stats.get("max_multiplier"),
                    },
                    "transition_focus_sampling": transition_focus_stats,
                    "calibration": {
                        "fraction_of_holdout": float(calibration_policy.fraction_of_holdout),
                        "min_samples": int(calibration_policy.min_samples),
                        "separate_calibration_min_holdout": int(calibration_policy.separate_calibration_min_holdout),
                        "min_support_per_class": int(calibration_policy.min_support_per_class),
                        "threshold_floor": float(calibration_policy.threshold_floor),
                        "threshold_cap": float(calibration_policy.threshold_cap),
                    },
                },
                "data": {
                    "downsample_scope": "train_only_post_split",
                    "total_sequences_pre_split": int(total_sequences_pre_split),
                    "samples_before_downsample": int(pre_downsample_count),
                    "samples_after_downsample": int(pre_downsample_count - downsample_removed),
                    "downsample_removed": int(downsample_removed),
                    "train_samples_for_evidence": int(train_samples_for_evidence),
                    "train_samples_after_sampling": int(len(X_train)),
                    "validation_samples": int(len(validation_data[0])) if validation_data is not None else 0,
                    "calibration_samples": int(len(calibration_data[0])) if calibration_data is not None else 0,
                    "critical_labels": list(critical_support_info.get("labels", [])),
                    "split_support": split_support_debug,
                    "calibration_split_support": calib_split_support_debug,
                },
                "class_weights": {
                    "base": {str(k): float(v) for k, v in class_weight_dict.items()},
                    "adjusted": {str(k): float(v) for k, v in adjusted_class_weight_dict.items()},
                    "activity_head": {
                        "mode": str(activity_weighting_diagnostics.get("activity_class_weight_mode", "global")),
                        "support": int(
                            activity_weighting_diagnostics.get(
                                "activity_class_weight_support",
                                len(y_train),
                            )
                        ),
                        "mask_coverage": float(
                            activity_weighting_diagnostics.get("activity_loss_mask_coverage", 0.0)
                        ),
                        "mask_floor": float(
                            activity_weighting_diagnostics.get("activity_loss_mask_floor", 0.0)
                        ),
                        "mean_occupied": float(
                            activity_weighting_diagnostics.get("activity_weight_mean_occupied", 0.0)
                        ),
                        "mean_masked": float(
                            activity_weighting_diagnostics.get("activity_weight_mean_masked", 0.0)
                        ),
                        "class_weights": {
                            str(k): float(v)
                            for k, v in activity_class_weight_dict.items()
                        },
                    },
                    "breakdown": weight_debug,
                },
                "calibration_debug": list(self._last_calibration_debug),
                "class_thresholds": {str(k): float(v) for k, v in class_thresholds.items()},
                "release_gate": {
                    "pass": bool(gate_pass),
                    "blocking_reasons": list(gate_reasons),
                    "watch_reasons": list(gate_watch_reasons),
                    "reasons": list(gate_reasons),
                    "room_status": str(room_status),
                },
                "runtime_topology": dict(runtime_topology),
                "room_status": str(room_status),
                "model_identity": model_identity,
                "metrics": metrics,
            }
            decision_trace_paths = self._write_decision_trace(
                elder_id=elder_id,
                room_name=room_name,
                saved_version=int(saved_version),
                payload=decision_trace_payload,
            )
            if decision_trace_paths:
                metrics["decision_trace"] = decision_trace_paths

            # Update in-memory platform only when candidate is promoted.
            if promote_to_latest:
                self.platform.class_thresholds[room_name] = {
                    str(class_id): float(threshold)
                    for class_id, threshold in class_thresholds.items()
                }
                self.platform.room_models[room_name] = model
            elif gate_pass and defer_promotion:
                logger.info(
                    f"Promotion deferred for {elder_id}/{room_name}; "
                    f"candidate saved as v{saved_version} pending run-level gates."
                )
            else:
                logger.warning(
                    f"Gate rejected candidate for {elder_id}/{room_name}; champion remains active. "
                    f"Reasons: {gate_reasons}"
                )
            
            # Log success
            status = "SUCCESS" if gate_pass else "REJECTED_BY_GATE"
            gate_msg = "; ".join(gate_reasons) if gate_reasons else None
            self._log_training_history(elder_id, room_name, final_acc, int(train_samples_for_evidence), max_epochs, status,
                                       error_msg=gate_msg,
                                       data_start_time=data_start_time, data_end_time=data_end_time)
            
            logger.info(
                f"Successfully trained {room_name}: Acc={final_acc:.2%}, "
                f"Gate={'PASS' if gate_pass else 'FAIL'}"
            )
            return metrics

        except Exception as e:
            # Log failure
            self._log_training_history(elder_id, room_name, 0.0, 0, 0, "FAILED", error_msg=str(e))
            raise ModelTrainError(f"Failed to train {room_name}: {e}") from e
        finally:
            self._policy_snapshot = None
            self._seed_panel_debug_context = None

    def _log_training_history(self, elder_id, room_name, accuracy, samples, epochs, status, 
                               error_msg=None, data_start_time=None, data_end_time=None):
        """Log training session to database."""
        try:
            # Format timestamps for SQLite/Postgres
            start_str = None
            if hasattr(data_start_time, 'strftime'):
                start_str = data_start_time.strftime('%Y-%m-%d %H:%M:%S')
            elif data_start_time is not None:
                # Try parsing if it's a string, or just let it fail gracefully to None
                try:
                    start_str = str(data_start_time)
                except:
                    start_str = None

            end_str = None
            if hasattr(data_end_time, 'strftime'):
                end_str = data_end_time.strftime('%Y-%m-%d %H:%M:%S')
            elif data_end_time is not None:
                try:
                    end_str = str(data_end_time)
                except:
                    end_str = None
            
            with adapter.get_connection() as conn:
                # Keep only the latest training history record for the same logical run key.
                # This prevents repeated drops of the same training file from cluttering UI history.
                conn.execute("""
                    DELETE FROM model_training_history
                    WHERE elder_id = ?
                      AND room = ?
                      AND model_type = ?
                      AND status = ?
                      AND ((data_start_time = ?) OR (data_start_time IS NULL AND ? IS NULL))
                      AND ((data_end_time = ?) OR (data_end_time IS NULL AND ? IS NULL))
                """, (
                    elder_id,
                    room_name.lower(),
                    "CNN+Transformer",
                    status,
                    start_str, start_str,
                    end_str, end_str
                ))

                conn.execute("""
                    INSERT INTO model_training_history 
                    (timestamp, elder_id, room, model_type, accuracy, samples_count, epochs, status, error_message, data_start_time, data_end_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    elder_id,
                    room_name.lower(),
                    "CNN+Transformer",
                    float(accuracy),
                    samples,
                    epochs,
                    status,
                    error_msg,
                    start_str,
                    end_str
                ))
        except Exception as e:
            logger.error(f"Failed to log training history: {e}")

    def augment_training_data(self, 
                              room_name: str,
                              elder_id: str, 
                              existing_timestamps: Any,
                              X_seq: np.ndarray,
                              y_seq: np.ndarray,
                              window_seconds: int,
                              interval_seconds: int,
                              seq_timestamps: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray]:
        """
        Augment live training data with historical corrections (Golden Samples).
        """
        room_seq_length = calculate_sequence_length(self.platform, room_name)
        return_with_timestamps = seq_timestamps is not None
        if seq_timestamps is None:
            seq_timestamps = np.array([], dtype="datetime64[ns]")
        
        try:
            # Fetch ALL historical corrections for this elder/room using shared util
            historical_df = fetch_golden_samples(elder_id, room_name)
            
            if historical_df is None or historical_df.empty:
                logger.info(f"No historical corrections found for augmentation ({elder_id}/{room_name})")
                if return_with_timestamps:
                    return X_seq, y_seq, seq_timestamps
                return X_seq, y_seq
                
            # Filter out timestamps already in the current training file
            # Robustly handle timestamps (Series vs Index)
            ts_obj = pd.to_datetime(existing_timestamps)
            if hasattr(ts_obj, 'dt'):
                floored = ts_obj.dt.floor('10s')
            else:
                floored = ts_obj.floor('10s')
            
            # P2 Fix: Move augmentation logic outside the if/else branch
            # This was previously only inside the else, skipping Series inputs
            existing_set = set(floored)
            historical_df['timestamp_key'] = historical_df['timestamp'].dt.floor('10s')
            new_corrections = historical_df[~historical_df['timestamp_key'].isin(existing_set)]
            
            if new_corrections.empty:
                logger.info(f"All corrections already in training data for {room_name}")
                if return_with_timestamps:
                    return X_seq, y_seq, seq_timestamps
                return X_seq, y_seq
            
            logger.info(f"Found {len(new_corrections)} historical corrections to augment for {room_name}")

            # Group corrections by date for efficient archive loading
            corrections_by_date = new_corrections.groupby('record_date')
            
            augmented_X = []
            augmented_y = []
            augmented_ts = []
            
            for record_date, date_corrections in corrections_by_date:
                # BATCH OPTIMIZATION: Fetch all windows for this date/room group in one query
                # 1. Prepare windows
                windows_to_fetch = []
                
                for _, correction in date_corrections.iterrows():
                    corr_ts = correction['timestamp']
                    window_start = corr_ts - timedelta(seconds=window_seconds - interval_seconds)
                    window_end = corr_ts
                    windows_to_fetch.append((window_start, window_end))
                    
                # 2. Batch Fetch from DB
                batch_df = fetch_sensor_windows_batch(elder_id, room_name, windows_to_fetch)
                
                # 3. STRICT VALIDATION: Check if batch is sufficient for ALL windows
                # "All DB or All Archive" strategy to prevent mixed data sources
                archive_needed = False
                
                if batch_df is None or batch_df.empty:
                    archive_needed = True
                else:
                    # Pre-validate all windows in the batch
                    for w_start, w_end in windows_to_fetch:
                         # Efficiently check counts without slicing copies yet
                         # Count rows within this window
                         count = batch_df[
                            (batch_df['timestamp'] >= w_start) & 
                            (batch_df['timestamp'] <= w_end)
                         ].shape[0]
                         
                         if count < room_seq_length * 0.8:
                             logger.debug(f"Batch incomplete for {w_start} ({count}/{room_seq_length}), forcing FULL archive load for {record_date}")
                             archive_needed = True
                             break # Fail fast: one bad window = discard all DB data for this date
                
                # 4. Load Archive if needed (Global fallback for this date)
                archive_df = None
                if archive_needed:
                    logger.info(f"Loading archive for {record_date} (Fallback)")
                    archive_path = self._find_archive_for_date(record_date, elder_id, ARCHIVE_DATA_DIR)
                    if archive_path:
                        try:
                            full_archive = load_sensor_data(
                                archive_path,
                                resample=True,
                                max_ffill_gap_seconds=self._active_policy().resampling.resolve(),
                            )
                            room_key = next((k for k in full_archive.keys() if k.lower() == room_name.lower()), None)
                            if room_key:
                                archive_df = full_archive[room_key].copy()
                                if 'timestamp' in archive_df.columns:
                                    archive_df['timestamp'] = pd.to_datetime(archive_df['timestamp'])
                                    archive_df = archive_df.sort_values('timestamp').reset_index(drop=True)
                        except Exception as e:
                            logger.warning(f"Archive load failed: {e}")
                
                # 5. Process Windows (Source determined by archive_needed flag)
                for i, (_, correction) in enumerate(date_corrections.iterrows()):
                    logger.debug(f"Processing correction {i}")
                    corr_ts = correction['timestamp']
                    corr_label = correction['activity']
                    w_start, w_end = windows_to_fetch[i]
                    
                    window_df = None
                    
                    if not archive_needed and batch_df is not None:
                         # Use DB Batch (Guaranteed valid by step 3)
                         window_df = batch_df[
                            (batch_df['timestamp'] <= w_end)
                         ].copy()
                         logger.debug(f"Batch slice: {len(window_df)} rows")
                    elif archive_df is not None:
                         # Use Archive
                         window_df = archive_df[
                            (archive_df['timestamp'] >= w_start) & 
                            (archive_df['timestamp'] <= w_end)
                         ].copy()
                         logger.debug(f"Archive slice: {len(window_df)} rows")
                    
                    # === Common Processing Logic ===
                    if window_df is None or window_df.empty:
                        continue

                    # Verify we have enough samples
                    logger.debug(f"Length check: {len(window_df)} vs {room_seq_length*0.8}")
                    if len(window_df) < room_seq_length * 0.8:
                        logger.debug("Length check failed, skipping window")
                        continue
                    
                    # Resample to exact sequence length if needed
                    if len(window_df) > room_seq_length:
                        window_df = window_df.tail(room_seq_length)
                    elif len(window_df) < room_seq_length:
                        # Pad with first row
                        pad_count = room_seq_length - len(window_df)
                        pad_df = pd.concat([window_df.head(1)] * pad_count, ignore_index=True)
                        window_df = pd.concat([pad_df, window_df], ignore_index=True)
                    
                    # Extract sensor features (Validation)
                    sensor_cols = [c for c in self.platform.sensor_columns if c in window_df.columns]
                    if len(sensor_cols) < len(self.platform.sensor_columns):
                        # Add temporal features if missing
                        if 'hour_sin' not in window_df.columns and 'timestamp' in window_df.columns:
                            hours = window_df['timestamp'].dt.hour + window_df['timestamp'].dt.minute / 60
                            window_df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
                            window_df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
                            window_df['day_period'] = (hours // 6).astype(int)
                    
                    # Final check
                    sensor_cols = [c for c in self.platform.sensor_columns if c in window_df.columns]
                    if len(sensor_cols) < 3:
                        continue
                    
                    # Pad missing columns
                    for col in self.platform.sensor_columns:
                        if col not in window_df.columns:
                            window_df[col] = 0.0
                    
                    # Create sequence
                    sensor_values = window_df[self.platform.sensor_columns].values.astype(float)
                    
                    # Encode label
                    if room_name in self.platform.label_encoders:
                        try:
                            encoded_label = self.platform.label_encoders[room_name].transform([corr_label])[0]
                        except ValueError:
                            logger.warning(f"Skipping correction: label '{corr_label}' not in encoder for {room_name}")
                            continue
                    else:
                        continue
                    
                    augmented_X.append(sensor_values)
                    augmented_y.append(encoded_label)
                    augmented_ts.append(np.datetime64(pd.to_datetime(corr_ts)))
            
            # Combine with original sequences
            if augmented_X:
                augmented_X = np.array(augmented_X)
                augmented_y = np.array(augmented_y)
                augmented_ts = np.asarray(augmented_ts, dtype="datetime64[ns]")
                
                X_seq = np.concatenate([X_seq, augmented_X], axis=0)
                y_seq = np.concatenate([y_seq, augmented_y], axis=0)
                seq_timestamps = np.concatenate([seq_timestamps, augmented_ts], axis=0)
                
                logger.info(f"✓ Augmented training with {len(augmented_X)} historical correction sequences for {room_name}")
            else:
                logger.info(f"No valid sequences could be extracted from archives for {room_name}")
            
            if return_with_timestamps:
                return X_seq, y_seq, seq_timestamps
            return X_seq, y_seq

                
        except Exception as e:
            logger.warning(f"Failed to augment training data: {e}", exc_info=True)
            if return_with_timestamps:
                return X_seq, y_seq, seq_timestamps
            return X_seq, y_seq

    def _find_archive_for_date(self, record_date, elder_id, archive_dir):
        """Find the archived Parquet file for a specific date and elder."""
        from pathlib import Path
        
        archive_dir = Path(archive_dir)
        
        # Convert date to various search patterns
        if hasattr(record_date, 'strftime'):
            date_obj = record_date
        else:
            from datetime import datetime
            date_obj = datetime.strptime(str(record_date), '%Y-%m-%d')
        
        date_str = date_obj.strftime('%Y-%m-%d')  # 2025-12-18
        date_ddmon = date_obj.strftime('%d%b').lower()  # 18dec
        date_ddmon_year = date_obj.strftime('%d%b%Y').lower()  # 18dec2025
        # Non-zero-padded day variants (e.g., 5dec / 5dec2025)
        date_dmon = f"{date_obj.day}{date_obj.strftime('%b').lower()}"
        date_dmon_year = f"{date_obj.day}{date_obj.strftime('%b').lower()}{date_obj.strftime('%Y')}"
        
        # Search patterns (prioritized)
        patterns_to_try = [
            f'*{date_str}*',      # 2025-12-18
            f'*{date_ddmon_year}*',  # 18dec2025
            f'*{date_ddmon}*',    # 18dec
            f'*{date_dmon_year}*',  # 5dec2025
            f'*{date_dmon}*',     # 5dec
        ]
        
        # Check date-specific subdirectory first
        date_subdir = archive_dir / date_str
        if date_subdir.exists():
            for f in date_subdir.glob('*.parquet'):
                if elder_id.lower() in f.name.lower():
                    return f
        
        # Search all subdirectories for matching files
        for subdir in archive_dir.iterdir():
            if not subdir.is_dir():
                continue
            for pattern in patterns_to_try:
                for f in subdir.glob(f'{pattern}.parquet'):
                    if elder_id.lower() in f.name.lower():
                        return f
        
        # Also check root archive directory
        for pattern in patterns_to_try:
            for f in archive_dir.glob(f'**/{pattern}.parquet'):
                if elder_id.lower() in f.name.lower():
                    return f

        # Fallback: inspect parquet timestamps when filename/date-dir patterns miss.
        # This is robust against naming variations and reset/replay archives.
        target_day = date_obj.date()
        elder_files = sorted(
            f for f in archive_dir.glob("**/*.parquet")
            if elder_id.lower() in f.name.lower()
        )
        for f in elder_files:
            try:
                loaded = load_sensor_data(f, resample=False)
            except Exception:
                continue
            for _, df in loaded.items():
                if df is None or getattr(df, "empty", True) or "timestamp" not in df.columns:
                    continue
                ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
                if ts.empty:
                    continue
                if ts.min().date() <= target_day <= ts.max().date():
                    return f
        
        return None
