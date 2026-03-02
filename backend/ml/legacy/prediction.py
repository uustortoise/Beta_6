import logging
import sqlite3
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

from config import get_room_config, DB_PATH
from ml.beta6.serving.prediction import infer_with_unknown_path, load_unknown_policy
from ml.beta6.serving.runtime_hooks import (
    apply_beta6_hmm_runtime,
    apply_beta6_unknown_abstain_runtime,
)
from ml.beta6.sequence import decode_hmm_with_duration_priors, load_duration_prior_policy
from ml.exceptions import PredictionError, DatabaseError
from ml.policy_defaults import get_runtime_unknown_rooms_default
# Shared Utilities
from ml.utils import calculate_sequence_length, fetch_golden_samples
from ml.registry import ModelRegistry
from utils.room_utils import normalize_timestamp, normalize_room_name
from utils.segment_utils import regenerate_segments, validate_activity_for_room
from utils.data_loader import load_sensor_data, get_archive_files
from elderlycare_v1_16.config.settings import DEFAULT_SENSOR_COLUMNS, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_DENOISING_METHOD, DEFAULT_DENOISING_WINDOW, DEFAULT_DENOISING_THRESHOLD
    
# Import adapter
try:
    from backend.db.legacy_adapter import LegacyDatabaseAdapter
except ImportError:
    from elderlycare_v1_16.database import db as adapter
else:
    adapter = LegacyDatabaseAdapter()

logger = logging.getLogger(__name__)


def _env_enabled(var_name: str, default: bool = False) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _env_int(var_name: str, default: int) -> int:
    raw = os.getenv(var_name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _env_float(var_name: str, default: float) -> float:
    raw = os.getenv(var_name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _parse_room_int_override(raw: str) -> dict[str, int]:
    overrides: dict[str, int] = {}
    txt = str(raw or "").strip()
    if not txt:
        return overrides
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
        try:
            overrides[room_key] = int(str(value_raw).strip())
        except (TypeError, ValueError):
            continue
    return overrides


def _parse_room_set_override(raw: str) -> set[str]:
    rooms: set[str] = set()
    txt = str(raw or "").strip()
    if not txt:
        return rooms
    for token in txt.split(","):
        item = str(token).strip()
        if not item:
            continue
        rooms.add(normalize_room_name(item))
    return rooms


def _parse_hour_window(raw: str, default_start: int = 22, default_end: int = 6) -> tuple[int, int]:
    txt = str(raw or "").strip()
    if not txt:
        return int(default_start), int(default_end)
    sep = ":" if ":" in txt else "-"
    parts = [p.strip() for p in txt.split(sep)]
    if len(parts) != 2:
        return int(default_start), int(default_end)
    try:
        start = int(parts[0]) % 24
        end = int(parts[1]) % 24
    except (TypeError, ValueError):
        return int(default_start), int(default_end)
    return start, end


class PredictionPipeline:
    """
    Manages inference, golden sample application, and result persistence.
    """
    def __init__(self, platform: Any, registry: ModelRegistry, enable_denoising: bool = True):
        self.platform = platform
        self.registry = registry
        self.room_config = get_room_config()
        self.enable_denoising = enable_denoising
        
        # Denoising defaults
        self.denoising_method = DEFAULT_DENOISING_METHOD
        self.denoising_window = DEFAULT_DENOISING_WINDOW
        self.denoising_threshold = DEFAULT_DENOISING_THRESHOLD

    def _resolve_hysteresis_steps(self, room_name: str) -> int:
        base_steps = max(1, _env_int("INFERENCE_HYSTERESIS_STEPS", 3))
        room_map = _parse_room_int_override(os.getenv("INFERENCE_HYSTERESIS_STEPS_BY_ROOM", ""))
        room_key = normalize_room_name(room_name)
        if room_key in room_map:
            return max(1, int(room_map[room_key]))
        return base_steps

    def _apply_hysteresis(self, labels: np.ndarray, room_name: str) -> np.ndarray:
        """
        Damp rapid label flips by requiring N consecutive votes before switching state.
        """
        if labels is None or len(labels) == 0:
            return labels
        if not _env_enabled("ENABLE_INFERENCE_HYSTERESIS", default=False):
            return labels

        steps = self._resolve_hysteresis_steps(room_name)
        if steps <= 1:
            return labels

        out = labels.astype(object).copy()
        current = str(out[0])
        pending_label = None
        pending_count = 0

        for idx in range(1, len(out)):
            observed = str(labels[idx])

            # low_confidence should not force state transitions.
            if observed == "low_confidence":
                out[idx] = current
                continue

            if observed == current:
                pending_label = None
                pending_count = 0
                out[idx] = current
                continue

            if pending_label == observed:
                pending_count += 1
            else:
                pending_label = observed
                pending_count = 1

            if pending_count >= steps:
                current = observed
                pending_label = None
                pending_count = 0

            out[idx] = current

        return out

    @staticmethod
    def _hour_window_mask(
        timestamps: np.ndarray,
        start_hour: int,
        end_hour: int,
    ) -> np.ndarray:
        ts = pd.to_datetime(pd.Series(timestamps), errors="coerce")
        if ts.empty:
            return np.zeros(shape=(0,), dtype=bool)
        hours = ts.dt.hour.to_numpy(dtype=int, na_value=-1)
        valid = hours >= 0
        mask = np.zeros(shape=(len(hours),), dtype=bool)
        if start_hour == end_hour:
            mask[valid] = True
            return mask
        if start_hour < end_hour:
            mask[valid] = (hours[valid] >= start_hour) & (hours[valid] < end_hour)
            return mask
        # Overnight span, e.g. 22-6
        mask[valid] = (hours[valid] >= start_hour) | (hours[valid] < end_hour)
        return mask

    def _apply_scoped_runtime_unknown(
        self,
        *,
        room_name: str,
        labels: np.ndarray,
        confidences: np.ndarray,
        low_conf_flags: np.ndarray,
        timestamps: np.ndarray,
        global_total_windows: int,
        global_unknown_windows: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a capped subset of ambiguous windows to `unknown` in scoped contexts.

        Beta 6 policy is fail-closed: scoped runtime unknown must remain enabled.
        """
        out = labels.astype(object).copy()
        runtime_unknown_flags = np.zeros(shape=(len(out),), dtype=bool)

        if len(out) == 0:
            return out, runtime_unknown_flags
        if not _env_enabled("RUNTIME_UNKNOWN_ENABLED", default=True):
            raise PredictionError(
                "RUNTIME_UNKNOWN_ENABLED=false is disallowed in Beta 6 fail-closed mode."
            )

        default_scope = ",".join(get_runtime_unknown_rooms_default())
        room_scope = _parse_room_set_override(os.getenv("RUNTIME_UNKNOWN_ROOMS", default_scope))
        room_key = normalize_room_name(room_name)
        if room_scope and room_key not in room_scope:
            return out, runtime_unknown_flags

        night_only = _env_enabled("RUNTIME_UNKNOWN_NIGHT_ONLY", default=True)
        night_start, night_end = _parse_hour_window(os.getenv("RUNTIME_UNKNOWN_NIGHT_HOURS", "22-6"))
        min_conf = float(np.clip(_env_float("RUNTIME_UNKNOWN_MIN_CONF", 0.55), 0.0, 1.0))
        room_cap = float(np.clip(_env_float("RUNTIME_UNKNOWN_RATE_ROOM_CAP", 0.25), 0.0, 1.0))
        global_cap = float(np.clip(_env_float("RUNTIME_UNKNOWN_RATE_GLOBAL_CAP", 0.12), 0.0, 1.0))

        room_total = int(len(out))
        if room_total <= 0:
            return out, runtime_unknown_flags

        label_arr = np.asarray([str(v).strip().lower() for v in out], dtype=object)
        conf_arr = np.asarray(confidences, dtype=float)
        low_conf_arr = np.asarray(low_conf_flags, dtype=bool)

        candidate_mask = (label_arr == "low_confidence") & low_conf_arr & (conf_arr <= min_conf)
        if night_only:
            candidate_mask &= self._hour_window_mask(
                timestamps=np.asarray(timestamps),
                start_hour=night_start,
                end_hour=night_end,
            )

        candidate_idx = np.where(candidate_mask)[0]
        if candidate_idx.size == 0:
            return out, runtime_unknown_flags

        room_unknown_existing = int(np.sum(label_arr == "unknown"))
        room_unknown_allowed = int(np.floor(room_cap * room_total))
        room_unknown_slots = max(0, room_unknown_allowed - room_unknown_existing)
        if room_unknown_slots <= 0:
            return out, runtime_unknown_flags

        global_total_after = int(global_total_windows + room_total)
        global_unknown_allowed = int(np.floor(global_cap * global_total_after))
        global_unknown_slots = max(0, global_unknown_allowed - int(global_unknown_windows))
        if global_unknown_slots <= 0:
            return out, runtime_unknown_flags

        allowed = min(int(candidate_idx.size), room_unknown_slots, global_unknown_slots)
        if allowed <= 0:
            return out, runtime_unknown_flags

        # Prefer lowest-confidence candidates first.
        ranked = candidate_idx[np.argsort(conf_arr[candidate_idx], kind="stable")]
        chosen = ranked[:allowed]
        out[chosen] = "unknown"
        runtime_unknown_flags[chosen] = True

        logger.info(
            "Scoped runtime unknown applied for %s: selected=%s candidates=%s room_cap=%.3f global_cap=%.3f night_only=%s",
            room_name,
            int(allowed),
            int(candidate_idx.size),
            room_cap,
            global_cap,
            bool(night_only),
        )
        return out, runtime_unknown_flags

    def _apply_beta6_unknown_abstain_runtime(
        self,
        *,
        room_name: str,
        y_pred_probs: np.ndarray,
        label_classes: list[str],
        final_labels: np.ndarray,
        low_conf_flags: list[bool],
        low_conf_hints: list[Optional[str]],
    ) -> tuple[np.ndarray, Optional[list[Optional[str]]], Optional[list[bool]], list[bool], list[Optional[str]]]:
        return apply_beta6_unknown_abstain_runtime(
            room_name=room_name,
            y_pred_probs=y_pred_probs,
            label_classes=label_classes,
            final_labels=final_labels,
            low_conf_flags=low_conf_flags,
            low_conf_hints=low_conf_hints,
            load_unknown_policy_fn=load_unknown_policy,
            infer_with_unknown_path_fn=infer_with_unknown_path,
        )

    def _apply_beta6_hmm_runtime(
        self,
        *,
        room_name: str,
        y_pred_probs: np.ndarray,
        label_classes: list[str],
        final_labels: np.ndarray,
        low_conf_flags: list[bool],
    ) -> np.ndarray:
        return apply_beta6_hmm_runtime(
            room_name=room_name,
            y_pred_probs=y_pred_probs,
            label_classes=label_classes,
            final_labels=final_labels,
            low_conf_flags=low_conf_flags,
            load_duration_prior_policy_fn=load_duration_prior_policy,
            decode_hmm_with_duration_priors_fn=decode_hmm_with_duration_priors,
        )

    def _two_stage_runtime_enabled_for_room(self, room_name: str) -> bool:
        if not _env_enabled("ENABLE_TWO_STAGE_CORE_RUNTIME", default=False):
            return False
        scoped = _parse_room_set_override(os.getenv("TWO_STAGE_CORE_RUNTIME_ROOMS", ""))
        if not scoped:
            return True
        return normalize_room_name(room_name) in scoped

    @staticmethod
    def _resolve_two_stage_stage_a_default_threshold() -> float:
        raw = os.getenv("TWO_STAGE_CORE_STAGE_A_OCCUPIED_THRESHOLD", "0.5")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 0.5
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _to_probability_matrix(raw: Any) -> np.ndarray:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"expected rank-2 probability matrix, got shape={arr.shape}")
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
        denom = np.clip(np.sum(exp, axis=1, keepdims=True), 1e-12, None)
        return (exp / denom).astype(np.float32, copy=False)

    def _predict_two_stage_core_probabilities(self, room_name: str, X_seq: np.ndarray) -> Optional[np.ndarray]:
        bundle = getattr(self.platform, "two_stage_core_models", {}).get(room_name)
        if not isinstance(bundle, dict):
            return None
        if not self._two_stage_runtime_enabled_for_room(room_name):
            return None

        stage_a_model = bundle.get("stage_a_model")
        if stage_a_model is None:
            return None

        raw_stage_a = stage_a_model.predict(X_seq, verbose=0)
        if isinstance(raw_stage_a, dict):
            raw_stage_a = raw_stage_a.get("activity_logits")
        stage_a_probs = self._to_probability_matrix(raw_stage_a)
        if stage_a_probs.shape[1] < 2:
            raise PredictionError(
                f"Two-stage stage_a width invalid for {room_name}: shape={stage_a_probs.shape}"
            )

        n_samples = int(stage_a_probs.shape[0])
        n_classes = int(bundle.get("num_classes", 0) or 0)
        excluded_ids = [int(v) for v in (bundle.get("excluded_class_ids") or []) if int(v) < n_classes]
        occupied_ids = [int(v) for v in (bundle.get("occupied_class_ids") or []) if int(v) < n_classes]
        if n_classes <= 1 or not excluded_ids or not occupied_ids:
            raise PredictionError(
                f"Two-stage class mapping invalid for {room_name}: "
                f"num_classes={n_classes}, excluded={excluded_ids}, occupied={occupied_ids}"
            )

        raw_stage_a_threshold = bundle.get(
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
        y_pred_probs = np.zeros((n_samples, n_classes), dtype=np.float32)
        non_occ_indices = np.where(~occupied_mask)[0]
        if non_occ_indices.size > 0:
            y_pred_probs[np.ix_(non_occ_indices, np.asarray(excluded_ids, dtype=np.int32))] = (
                1.0 / float(len(excluded_ids))
            )

        if np.any(occupied_mask):
            occ_indices = np.where(occupied_mask)[0]
            p_occ_active = p_occ[occupied_mask]
            p_unocc_active = p_unocc[occupied_mask]
            y_pred_probs[np.ix_(occ_indices, np.asarray(excluded_ids, dtype=np.int32))] = (
                (p_unocc_active / float(len(excluded_ids)))[:, None]
            )

            stage_b_model = bundle.get("stage_b_model")
            if stage_b_model is not None:
                raw_stage_b = stage_b_model.predict(X_seq, verbose=0)
                if isinstance(raw_stage_b, dict):
                    raw_stage_b = raw_stage_b.get("activity_logits")
                stage_b_probs = self._to_probability_matrix(raw_stage_b)
                if stage_b_probs.shape[1] != len(occupied_ids):
                    raise PredictionError(
                        f"Two-stage stage_b width invalid for {room_name}: "
                        f"shape={stage_b_probs.shape}, expected={len(occupied_ids)}"
                    )
                stage_b_probs = stage_b_probs[occupied_mask]
                for idx, class_id in enumerate(occupied_ids):
                    y_pred_probs[occ_indices, class_id] = p_occ_active * stage_b_probs[:, idx]
            else:
                primary_occupied = int(bundle.get("primary_occupied_class_id", occupied_ids[0]) or occupied_ids[0])
                if primary_occupied not in occupied_ids:
                    primary_occupied = occupied_ids[0]
                y_pred_probs[occ_indices, primary_occupied] = p_occ_active

        row_sums = np.sum(y_pred_probs, axis=1, keepdims=True)
        invalid_rows = row_sums[:, 0] <= 1e-8
        if np.any(invalid_rows):
            y_pred_probs[invalid_rows, :] = 0.0
            y_pred_probs[invalid_rows, excluded_ids[0]] = 1.0
            row_sums = np.sum(y_pred_probs, axis=1, keepdims=True)
        y_pred_probs = y_pred_probs / np.clip(row_sums, 1e-8, None)
        logger.info(
            "Using two-stage core runtime path for %s (stage_a_threshold=%.3f)",
            room_name,
            stage_a_threshold,
        )
        return y_pred_probs

    def run_prediction(self, 
                       sensor_data: Dict[str, pd.DataFrame], 
                       loaded_rooms: List[str],
                       seq_length: int,
                       progress_callback=None) -> Dict[str, pd.DataFrame]:
        """
        Run inference on pre-loaded data for specified rooms.
        """
        predictions = {}
        total_rooms = len(loaded_rooms)
        global_total_windows = 0
        global_unknown_windows = 0
        
        for i, room_name in enumerate(loaded_rooms):
            if progress_callback:
                progress_callback(int(100 * (i / total_rooms)), f"Predicting {room_name}...")
            # Match normalized names
            matching_key = None
            if room_name in sensor_data:
                matching_key = room_name
            else:
                 # Check normalized match
                 norm_room = normalize_room_name(room_name)
                 for key in sensor_data.keys():
                     if normalize_room_name(key) == norm_room:
                         matching_key = key
                         break
            
            if not matching_key:
                continue

            try:
                df = sensor_data[matching_key]
                
                # Check timestamps
                if 'timestamp' not in df.columns:
                     # Attempt recovery
                     ts_col = next((c for c in df.columns if c.lower() in ['time', 'date', 'datetime']), None)
                     if ts_col:
                         df = df.rename(columns={ts_col: 'timestamp'})
                
                # Preprocess (Validation/Inference Mode)
                processed = self.platform.preprocess_with_resampling(
                    df, room_name, is_training=False, 
                    apply_denoising=False # Assumes performed externally or raw data passed
                )
                
                if len(processed) < 10: # Minimum data sanity check
                    continue

                # Dynamic Sequence Length
                current_seq_length = calculate_sequence_length(self.platform, room_name)
                
                if len(processed) < current_seq_length:
                    logger.debug(f"Skipping {room_name}: insufficient data ({len(processed)} < {current_seq_length})")
                    continue

                # Create Sequences
                input_data = np.asarray(processed[self.platform.sensor_columns].values, dtype=float)
                X_seq = self.platform.create_sequences(input_data, current_seq_length)
                
                if len(X_seq) == 0:
                    continue

                # Predict
                model = self.platform.room_models[room_name]
                y_pred_probs = None
                try:
                    y_pred_probs = self._predict_two_stage_core_probabilities(room_name, X_seq)
                except Exception as e:
                    logger.warning(f"Two-stage runtime disabled for {room_name} due to error: {e}")
                    y_pred_probs = None
                if y_pred_probs is None:
                    y_pred_probs = model.predict(X_seq, verbose=0)
                    if isinstance(y_pred_probs, dict):
                        y_pred_probs = y_pred_probs.get("activity_logits")
                if y_pred_probs is None or len(y_pred_probs) == 0:
                    continue

                # Top-k model suggestions (used to enrich low-confidence review UI).
                rank_idx = np.argsort(y_pred_probs, axis=1)[:, ::-1]
                top1_idx = rank_idx[:, 0].astype(int)
                top2_idx = rank_idx[:, 1].astype(int) if y_pred_probs.shape[1] > 1 else top1_idx.copy()
                confidences = y_pred_probs[np.arange(len(y_pred_probs)), top1_idx].astype(float)
                top2_confidences = (
                    y_pred_probs[np.arange(len(y_pred_probs)), top2_idx].astype(float)
                    if y_pred_probs.shape[1] > 1 else np.zeros(len(y_pred_probs), dtype=float)
                )

                # Decode labels for top-k suggestions.
                if room_name in self.platform.label_encoders:
                    encoder = self.platform.label_encoders[room_name]
                    decoded_top1 = encoder.inverse_transform(top1_idx)
                    decoded_top2 = encoder.inverse_transform(top2_idx)
                    label_to_idx = {}
                    label_classes: list[str] = []
                    try:
                        classes = list(getattr(encoder, "classes_", []))
                        label_classes = [str(lbl) for lbl in classes]
                        label_to_idx = {str(lbl): int(idx) for idx, lbl in enumerate(classes)}
                    except Exception:
                        label_classes = []
                        label_to_idx = {}
                else:
                    decoded_top1 = np.array([str(l) for l in top1_idx])
                    decoded_top2 = np.array([str(l) for l in top2_idx])
                    label_to_idx = {}
                    label_classes = [str(i) for i in range(y_pred_probs.shape[1])]
                if not label_to_idx:
                    for k, idx in enumerate(top1_idx):
                        try:
                            label_to_idx[str(decoded_top1[k])] = int(idx)
                        except Exception:
                            continue
                if not label_classes:
                    label_classes = [str(i) for i in range(y_pred_probs.shape[1])]

                # Apply class-calibrated confidence thresholds (fallback to global threshold).
                room_thresholds = getattr(self.platform, 'class_thresholds', {}).get(room_name, {}) or {}

                # Optional log for visibility; keep concise to avoid log spam.
                if room_thresholds:
                    logger.info(
                        f"Using calibrated thresholds for {room_name} "
                        f"({len(room_thresholds)} classes)"
                    )

                final_labels = []
                low_conf_count = 0
                top1_labels = []
                top2_labels = []
                top1_probs = []
                top2_probs = []
                low_conf_flags = []
                low_conf_thresholds = []
                low_conf_hints = []

                for i, class_idx in enumerate(top1_idx):
                    class_idx = int(class_idx)
                    prob = float(confidences[i])
                    top1_label = str(decoded_top1[i])
                    top2_label = str(decoded_top2[i])

                    class_threshold = room_thresholds.get(
                        str(class_idx),
                        room_thresholds.get(class_idx, DEFAULT_CONFIDENCE_THRESHOLD),
                    )
                    try:
                        class_threshold = float(class_threshold)
                    except (TypeError, ValueError):
                        class_threshold = float(DEFAULT_CONFIDENCE_THRESHOLD)

                    if class_threshold < 0:
                        class_threshold = float(DEFAULT_CONFIDENCE_THRESHOLD)

                    if prob < class_threshold:
                        final_labels.append('low_confidence')
                        low_conf_count += 1
                        low_conf_flags.append(True)
                        low_conf_hints.append(top1_label)
                    else:
                        final_labels.append(top1_label)
                        low_conf_flags.append(False)
                        low_conf_hints.append(None)

                    top1_labels.append(top1_label)
                    top2_labels.append(top2_label)
                    top1_probs.append(prob)
                    top2_probs.append(float(top2_confidences[i]))
                    low_conf_thresholds.append(class_threshold)

                final_labels = np.asarray(final_labels, dtype=object)
                raw_labels = final_labels.copy()
                final_labels = self._apply_hysteresis(final_labels, room_name)
                final_labels = self._apply_beta6_hmm_runtime(
                    room_name=room_name,
                    y_pred_probs=y_pred_probs,
                    label_classes=label_classes,
                    final_labels=final_labels,
                    low_conf_flags=list(low_conf_flags),
                )
                (
                    final_labels,
                    beta6_uncertainty_states,
                    beta6_abstain_flags,
                    low_conf_flags,
                    low_conf_hints,
                ) = self._apply_beta6_unknown_abstain_runtime(
                    room_name=room_name,
                    y_pred_probs=y_pred_probs,
                    label_classes=label_classes,
                    final_labels=final_labels,
                    low_conf_flags=list(low_conf_flags),
                    low_conf_hints=list(low_conf_hints),
                )

                # Align timestamps (sequences start from seq_length - 1)
                if 'timestamp' in processed.columns:
                    valid_timestamps = processed['timestamp'].iloc[current_seq_length-1:].values
                else:
                    valid_timestamps = processed.index[current_seq_length-1:]

                final_labels, runtime_unknown_flags = self._apply_scoped_runtime_unknown(
                    room_name=room_name,
                    labels=final_labels,
                    confidences=np.asarray(top1_probs, dtype=float),
                    low_conf_flags=np.asarray(low_conf_flags, dtype=bool),
                    timestamps=np.asarray(valid_timestamps),
                    global_total_windows=int(global_total_windows),
                    global_unknown_windows=int(global_unknown_windows),
                )

                raw_top1_labels = np.asarray(top1_labels, dtype=object)
                raw_top1_probs = np.asarray(top1_probs, dtype=float)
                raw_top2_labels = np.asarray(top2_labels, dtype=object)
                raw_top2_probs = np.asarray(top2_probs, dtype=float)
                effective_top1_labels: list[str] = []
                effective_confidences: list[float] = []
                effective_top2_labels: list[str | None] = []
                effective_top2_probs: list[float] = []
                for j, chosen in enumerate(final_labels):
                    chosen_label = str(chosen)
                    cls_idx = label_to_idx.get(chosen_label)
                    if cls_idx is not None and 0 <= int(cls_idx) < y_pred_probs.shape[1]:
                        eff_conf = float(y_pred_probs[j, int(cls_idx)])
                    else:
                        eff_conf = float(raw_top1_probs[j])
                    effective_confidences.append(eff_conf)
                    if chosen_label in {"low_confidence", "unknown"}:
                        eff_top1_label = str(raw_top1_labels[j])
                        effective_top1_labels.append(eff_top1_label)
                    else:
                        eff_top1_label = chosen_label
                        effective_top1_labels.append(eff_top1_label)

                    # Effective top-2 must be consistent with effective top-1.
                    # Rank remaining classes by raw model probabilities.
                    excluded_idx = label_to_idx.get(eff_top1_label)
                    row_probs = y_pred_probs[j]
                    if excluded_idx is None or int(excluded_idx) < 0 or int(excluded_idx) >= row_probs.shape[0]:
                        effective_top2_labels.append(str(raw_top2_labels[j]))
                        effective_top2_probs.append(float(raw_top2_probs[j]))
                    elif row_probs.shape[0] <= 1:
                        effective_top2_labels.append(None)
                        effective_top2_probs.append(float("nan"))
                    else:
                        candidate_indices = [k for k in range(row_probs.shape[0]) if k != int(excluded_idx)]
                        best_idx = max(candidate_indices, key=lambda k: float(row_probs[k]))
                        effective_top2_labels.append(str(label_classes[best_idx]))
                        effective_top2_probs.append(float(row_probs[best_idx]))
                if len(confidences) > 0:
                    low_conf_rate = low_conf_count / len(confidences)
                    if low_conf_rate > 0.3:
                        logger.warning(
                            f"⚠️ High low_confidence rate for {room_name}: {low_conf_rate:.1%} "
                            f"(thresholds: {'calibrated' if room_thresholds else f'global={DEFAULT_CONFIDENCE_THRESHOLD}'})"
                        )
                
                # Create result DataFrame
                pred_df = pd.DataFrame({
                    'timestamp': valid_timestamps,
                    'predicted_activity': final_labels,
                    'predicted_activity_raw': raw_labels,
                    'confidence': np.asarray(effective_confidences, dtype=float),
                    'confidence_raw': raw_top1_probs,
                    'predicted_top1_label': np.asarray(effective_top1_labels, dtype=object),
                    'predicted_top1_label_raw': raw_top1_labels,
                    'predicted_top1_prob': np.asarray(effective_confidences, dtype=float),
                    'predicted_top1_prob_raw': raw_top1_probs,
                    'predicted_top2_label': np.asarray(effective_top2_labels, dtype=object),
                    'predicted_top2_label_raw': raw_top2_labels,
                    'predicted_top2_prob': np.asarray(effective_top2_probs, dtype=float),
                    'predicted_top2_prob_raw': raw_top2_probs,
                    'low_confidence_threshold': low_conf_thresholds,
                    'is_low_confidence': low_conf_flags,
                    'low_confidence_hint_label': low_conf_hints,
                    'is_runtime_unknown': runtime_unknown_flags,
                })
                if beta6_uncertainty_states is not None:
                    pred_df['beta6_uncertainty_state'] = beta6_uncertainty_states
                if beta6_abstain_flags is not None:
                    pred_df['is_beta6_abstain'] = beta6_abstain_flags
                
                predictions[room_name] = pred_df
                global_total_windows += int(len(final_labels))
                global_unknown_windows += int(np.sum(np.asarray(final_labels) == "unknown"))
                
            except PredictionError:
                raise
            except Exception as e:
                logger.exception(f"Prediction failed for {room_name}: {e}")
                
        return predictions

    def apply_golden_samples(self, predictions: Dict[str, pd.DataFrame], elder_id: str) -> Dict[str, pd.DataFrame]:
        """
        Override predictions with manually corrected "Golden Samples" from DB.
        """
        corrected_results = {}
        for room_name, pred_df in predictions.items():
            golden_df = fetch_golden_samples(elder_id, room_name)
            corrected_results[room_name] = self._apply_golden_samples_to_predictions(pred_df, golden_df)
        return corrected_results

    # _fetch_golden_samples removed in favor of ml.utils.fetch_golden_samples

    def _apply_golden_samples_to_predictions(self, pred_df, golden_df):
        """
        Apply corrections to prediction DataFrame using localized Timestamp merge (Optimized).
        """
        if golden_df is None or golden_df.empty:
            return pred_df
            
        try:
            # Ensure timestamps are datetime and floor to 10s
            if not pd.api.types.is_datetime64_any_dtype(pred_df['timestamp']):
                pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
            pred_df['timestamp'] = pred_df['timestamp'].dt.floor('10s')

            # Create a dictionary for O(1) lookup: timestamp -> activity
            correction_map = dict(zip(golden_df['timestamp'], golden_df['activity']))
            
            # Identify which rows need correction (vectorized isin)
            mask = pred_df['timestamp'].isin(correction_map.keys())
            
            if not mask.any():
                return pred_df

            # Apply corrections efficiently
            # We map timestamps to corrections only for the masked rows
            corrections = pred_df.loc[mask, 'timestamp'].map(correction_map)
            pred_df.loc[mask, 'predicted_activity'] = corrections
            
            # Mark confidence as 1.0 for corrected rows (manual truth is absolute)
            pred_df.loc[mask, 'confidence'] = 1.0
            if 'is_low_confidence' in pred_df.columns:
                pred_df.loc[mask, 'is_low_confidence'] = False
            if 'low_confidence_hint_label' in pred_df.columns:
                pred_df.loc[mask, 'low_confidence_hint_label'] = None
            if 'predicted_top1_label' in pred_df.columns:
                pred_df.loc[mask, 'predicted_top1_label'] = corrections.values
            if 'predicted_top1_prob' in pred_df.columns:
                pred_df.loc[mask, 'predicted_top1_prob'] = 1.0
            
            corrections_applied = mask.sum()
            if corrections_applied > 0:
                logger.info(f"Applied {corrections_applied} Golden Sample corrections")
            
            return pred_df
        except Exception as e:
            logger.warning(f"Failed to apply Golden Samples to predictions: {e}")
            return pred_df

    def save_predictions_to_db(self, 
                               predictions: Dict[str, pd.DataFrame], 
                               elder_id: str,
                               confidence_threshold: float = 0.0):
        """
        Persist predictions to `adl_history` table.
        """
        try:
            with adapter.get_connection() as conn:
                cursor = conn.cursor()
                
                # *** AUTO-REGISTER ELDER IF NOT EXISTS ***
                # Prevents FK violation when inserting into adl_history for new elders
                elder_name = elder_id.split('_')[1].title() if '_' in elder_id else elder_id
                cursor.execute('''
                    INSERT INTO elders (elder_id, full_name, created_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT (elder_id) DO NOTHING
                ''', (elder_id, elder_name))
                
                # Track actual (room, date) pairs touched during insert
                # (replaces old datetime.now() which was incorrect for archived data)
                touched_dates = {}  # room_name -> set of date strings

                
                for room_name, pred_df in predictions.items():
                    if 'timestamp' not in pred_df.columns or 'predicted_activity' not in pred_df.columns:
                        continue
                    
                    normalized_room = normalize_room_name(room_name)
                    
                    for _, row in pred_df.iterrows():
                        ts = row['timestamp']
                        timestamp_str = normalize_timestamp(ts)
                        # Extract date from timestamp
                        row_date = ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)[:10]

                        # Persist both sensor context and model hint context in sensor_features JSON.
                        sensor_features_payload = {}
                        for sensor_col in DEFAULT_SENSOR_COLUMNS:
                            val = row.get(sensor_col)
                            if pd.notna(val):
                                try:
                                    sensor_features_payload[sensor_col] = float(val)
                                except (TypeError, ValueError):
                                    sensor_features_payload[sensor_col] = val

                        top1_label = row.get('predicted_top1_label')
                        if pd.notna(top1_label):
                            sensor_features_payload['predicted_top1_label'] = str(top1_label)

                        top1_prob = row.get('predicted_top1_prob')
                        if pd.notna(top1_prob):
                            sensor_features_payload['predicted_top1_prob'] = float(top1_prob)

                        top2_label = row.get('predicted_top2_label')
                        if pd.notna(top2_label):
                            sensor_features_payload['predicted_top2_label'] = str(top2_label)

                        top2_prob = row.get('predicted_top2_prob')
                        if pd.notna(top2_prob):
                            sensor_features_payload['predicted_top2_prob'] = float(top2_prob)

                        top2_label_raw = row.get('predicted_top2_label_raw')
                        if pd.notna(top2_label_raw):
                            sensor_features_payload['predicted_top2_label_raw'] = str(top2_label_raw)

                        top2_prob_raw = row.get('predicted_top2_prob_raw')
                        if pd.notna(top2_prob_raw):
                            sensor_features_payload['predicted_top2_prob_raw'] = float(top2_prob_raw)

                        low_conf_threshold = row.get('low_confidence_threshold')
                        if pd.notna(low_conf_threshold):
                            sensor_features_payload['low_confidence_threshold'] = float(low_conf_threshold)

                        is_low_conf = row.get('is_low_confidence')
                        if pd.notna(is_low_conf):
                            is_low_conf = bool(is_low_conf)
                            sensor_features_payload['is_low_confidence'] = is_low_conf
                            if is_low_conf:
                                low_conf_hint = row.get('low_confidence_hint_label') or row.get('predicted_top1_label')
                                if pd.notna(low_conf_hint):
                                    sensor_features_payload['low_confidence_hint_label'] = str(low_conf_hint)

                        is_runtime_unknown = row.get('is_runtime_unknown')
                        if pd.notna(is_runtime_unknown):
                            sensor_features_payload['is_runtime_unknown'] = bool(is_runtime_unknown)

                        sensor_features_json = json.dumps(sensor_features_payload) if sensor_features_payload else None

                        # Check if a corrected row already exists for this timestamp/room
                        cursor.execute('''
                            SELECT id FROM adl_history 
                            WHERE elder_id = ? 
                              AND timestamp = ? 
                              AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = ?
                              AND is_corrected = 1
                        ''', (elder_id, timestamp_str, normalized_room))
                        
                        if cursor.fetchone():
                            continue
                        
                        # Delete any existing uncorrected row to avoid duplicates
                        cursor.execute('''
                            DELETE FROM adl_history 
                            WHERE elder_id = ? 
                              AND timestamp = ? 
                              AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = ?
                              AND is_corrected = 0
                        ''', (elder_id, timestamp_str, normalized_room))
                        
                        # Validate and Insert
                        validated_activity = validate_activity_for_room(row['predicted_activity'], room_name)
                        
                        cursor.execute('''
                            INSERT INTO adl_history 
                            (elder_id, room, timestamp, activity_type, confidence, record_date, is_corrected, sensor_features)
                            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                        ''', (
                            elder_id, room_name, timestamp_str,
                            validated_activity,
                            row.get('confidence', 1.0),
                            row_date,
                            sensor_features_json
                        ))
                        
                        # Track actual date for this room
                        touched_dates.setdefault(room_name, set()).add(row_date)
                
                conn.commit()
                
                # Regenerate segments for actual (room, date) pairs
                for room_name, dates in touched_dates.items():
                    for d in dates:
                        regenerate_segments(elder_id, room_name, d, conn)

        except Exception as e:
            raise DatabaseError(f"Failed to save predictions: {e}") from e

    def repredict_all(self, elder_id: str, archive_dir: Path, progress_callback=None, rooms=None) -> Dict[str, int]:
        """
        Re-run predictions on archived input files.
        
        Args:
            rooms: Optional set of normalized room names to scope reprediction.
                   If provided, only these rooms are loaded, predicted, and saved.
                   Prevents cross-room contamination from non-deterministic retraining.
        """
        # Normalize room filter for consistent matching across naming conventions
        normalized_filter = {normalize_room_name(r) for r in rooms} if rooms else None
        scope_msg = f" (scoped to {rooms})" if rooms else " (all rooms)"
        logger.info(f"Starting historical re-prediction for {elder_id}{scope_msg}")
        
        archive_files = get_archive_files(Path(archive_dir), file_type='input', resident_id=elder_id)
        if not archive_files:
            logger.warning(f"No archived input files found for {elder_id}")
            return {"repredicted_files": 0}
        
        logger.info(f"Found {len(archive_files)} files to repredict")
        
        # Early filter: only load models for target rooms
        loaded_rooms = self.registry.load_models_for_elder(elder_id, self.platform)
        if normalized_filter:
            loaded_rooms = [r for r in loaded_rooms if normalize_room_name(r) in normalized_filter]
            logger.info(f"Filtered models to {len(loaded_rooms)} rooms: {loaded_rooms}")
        if not loaded_rooms:
             logger.error(f"No models found for {elder_id}")
             return {"repredicted_files": 0}
        
        repredicted_count = 0
        total_files = len(archive_files)
        for i, file_info in enumerate(archive_files):
            file_path = Path(file_info['path'])
            if progress_callback:
                p = 85 + int(15 * (i / total_files))
                progress_callback(p, f"Re-predicting {file_path.name}...")
            try:
                data = load_sensor_data(file_path, resample=True)
                
                # Early filter: only process target rooms from input data
                if normalized_filter:
                    data = {k: v for k, v in data.items() if normalize_room_name(k) in normalized_filter}
                    if not data:
                        continue
                
                # Denoise
                for room, df in data.items():
                    if self.enable_denoising:
                        from elderlycare_v1_16.preprocessing.noise import hampel_filter
                        cols_to_denoise = [c for c in DEFAULT_SENSOR_COLUMNS if c in df.columns]
                        if cols_to_denoise:
                            hampel_filter(df, columns=cols_to_denoise, 
                                          window=self.denoising_window, 
                                          n_sigmas=self.denoising_threshold, inplace=True)
                
                # Run prediction (only for filtered rooms/models)
                raw_predictions = self.run_prediction(data, loaded_rooms, seq_length=0) 
                
                # Apply golden samples (Manual corrections take priority!)
                predictions = self.apply_golden_samples(raw_predictions, elder_id)
                
                # Save
                self.save_predictions_to_db(predictions, elder_id)
                repredicted_count += 1
                
            except Exception as e:
                logger.error(f"Failed to repredict {file_path.name}: {e}")
                
        return {"repredicted_files": repredicted_count}
