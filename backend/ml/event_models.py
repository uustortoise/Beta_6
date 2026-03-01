"""Event-first two-stage model scaffold for occupancy-aware ADL prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class EventFirstConfig:
    occupancy_class: str = "unoccupied"
    n_estimators_stage_a: int = 200
    n_estimators_stage_b: int = 200
    random_state: int = 42
    min_samples_leaf: int = 2
    stage_a_class_weight: Optional[str] = "balanced"
    stage_b_class_weight: Optional[str] = "balanced_subsample"
    stage_a_model_type: str = "rf"
    stage_a_temporal_lag_windows: int = 0
    stage_a_temporal_include_delta: bool = True
    stage_a_transformer_epochs: int = 8
    stage_a_transformer_batch_size: int = 256
    stage_a_transformer_learning_rate: float = 1e-3
    stage_a_transformer_hidden_dim: int = 32
    stage_a_transformer_num_heads: int = 2
    stage_a_transformer_dropout: float = 0.10
    stage_a_transformer_class_weight_power: float = 0.5
    stage_a_transformer_conv_kernel_size: int = 3
    stage_a_transformer_conv_blocks: int = 2
    stage_a_transformer_use_sequence_filter: bool = True
    unknown_label: str = "unknown"
    default_activity_threshold: float = 0.50
    use_unknown_for_low_confidence: bool = False


class _TinySequenceTransformerStageA:
    """
    Lightweight causal Transformer for Stage-A occupancy classification.

    Expects input shape: [n_samples, seq_len, feature_dim].
    Outputs sklearn-style probabilities with classes_ = [unoccupied, occupied].
    """

    def __init__(
        self,
        *,
        random_state: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        class_weight_power: float,
        conv_kernel_size: int,
        conv_blocks: int,
        class_weight_mode: Optional[str] = "balanced",
    ) -> None:
        self.random_state = int(random_state)
        self.epochs = int(max(epochs, 1))
        self.batch_size = int(max(batch_size, 1))
        self.learning_rate = float(max(learning_rate, 1e-5))
        self.hidden_dim = int(max(hidden_dim, 8))
        self.num_heads = int(max(num_heads, 1))
        self.dropout = float(min(max(dropout, 0.0), 0.5))
        self.class_weight_power = float(min(max(class_weight_power, 0.0), 1.0))
        self.conv_kernel_size = int(max(conv_kernel_size, 2))
        self.conv_blocks = int(max(conv_blocks, 1))
        self.class_weight_mode = class_weight_mode
        self.classes_ = np.asarray(["unoccupied", "occupied"], dtype=object)
        self._constant_occupied_prob: Optional[float] = None
        self._model = None

    @staticmethod
    def _import_tensorflow():
        try:
            import tensorflow as tf  # type: ignore

            return tf
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("TensorFlow is required for sequence_transformer Stage-A.") from exc

    def _build_model(self, *, tf, seq_len: int, feature_dim: int, positive_prior: float):
        d_model = int(max(self.hidden_dim, self.num_heads * 4))
        inputs = tf.keras.Input(shape=(int(seq_len), int(feature_dim)), name="stage_a_sequence_input")
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="stage_a_ln_input")(inputs)
        x = tf.keras.layers.Dense(d_model, activation="relu", name="stage_a_dense_proj")(x)
        x = tf.keras.layers.Dropout(float(self.dropout), name="stage_a_proj_dropout")(x)
        for block_idx in range(int(self.conv_blocks)):
            residual = x
            dilation = int(max(1, 2 ** block_idx))
            c = tf.keras.layers.Conv1D(
                filters=d_model,
                kernel_size=int(self.conv_kernel_size),
                padding="causal",
                dilation_rate=dilation,
                activation="relu",
                name=f"stage_a_tcn_conv_{block_idx}_a",
            )(x)
            c = tf.keras.layers.Dropout(float(self.dropout), name=f"stage_a_tcn_dropout_{block_idx}")(c)
            c = tf.keras.layers.Conv1D(
                filters=d_model,
                kernel_size=1,
                padding="same",
                activation=None,
                name=f"stage_a_tcn_conv_{block_idx}_b",
            )(c)
            x = tf.keras.layers.Add(name=f"stage_a_tcn_residual_{block_idx}")([residual, c])
            x = tf.keras.layers.LayerNormalization(
                epsilon=1e-6,
                name=f"stage_a_tcn_ln_{block_idx}",
            )(x)
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=int(self.num_heads),
            key_dim=max(int(d_model // max(self.num_heads, 1)), 4),
            dropout=float(self.dropout),
            name="stage_a_mha",
        )(x, x, use_causal_mask=True)
        x = tf.keras.layers.Add(name="stage_a_attn_residual")([x, attn])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="stage_a_attn_ln")(x)
        ff = tf.keras.layers.Dense(int(max(d_model * 2, 16)), activation="relu", name="stage_a_ffn_1")(x)
        ff = tf.keras.layers.Dropout(float(self.dropout), name="stage_a_ffn_dropout")(ff)
        ff = tf.keras.layers.Dense(d_model, name="stage_a_ffn_2")(ff)
        x = tf.keras.layers.Add(name="stage_a_ffn_residual")([x, ff])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="stage_a_ffn_ln")(x)
        avg_pool = tf.keras.layers.GlobalAveragePooling1D(name="stage_a_avg_pool")(x)
        max_pool = tf.keras.layers.GlobalMaxPooling1D(name="stage_a_max_pool")(x)
        last_step = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="stage_a_last_step")(x)
        pooled = tf.keras.layers.Concatenate(name="stage_a_pooled_features")([avg_pool, max_pool, last_step])
        pooled = tf.keras.layers.Dropout(float(self.dropout), name="stage_a_head_dropout")(pooled)
        prior = float(np.clip(positive_prior, 1e-4, 1.0 - 1e-4))
        logit_bias = float(math.log(prior / (1.0 - prior)))
        outputs = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(logit_bias),
            name="occupied_probability",
        )(pooled)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="stage_a_tiny_transformer")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(self.learning_rate)),
            loss="binary_crossentropy",
        )
        return model

    def fit(self, x, y, sample_weight=None):
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim != 3:
            raise ValueError("sequence_transformer Stage-A expects a 3D tensor [n, t, d].")
        y_arr = np.asarray(y, dtype=object)
        if len(y_arr) != len(x_arr):
            raise ValueError("Stage-A sequence labels length mismatch.")

        y_bin = np.asarray(y_arr == "occupied", dtype=np.float32)
        positive_count = int(np.sum(y_bin))
        if positive_count == 0:
            self._constant_occupied_prob = 0.0
            self._model = None
            return self
        if positive_count == len(y_bin):
            self._constant_occupied_prob = 1.0
            self._model = None
            return self

        tf = self._import_tensorflow()
        tf.keras.backend.clear_session()
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass
        tf.keras.utils.set_random_seed(int(self.random_state))

        self._model = self._build_model(
            tf=tf,
            seq_len=int(x_arr.shape[1]),
            feature_dim=int(x_arr.shape[2]),
            positive_prior=float(np.mean(y_bin)),
        )
        self._constant_occupied_prob = None

        weights = np.ones(shape=(len(y_bin),), dtype=np.float32)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float32)
            if len(sw) == len(weights):
                weights = weights * sw

        class_weight_mode = str(self.class_weight_mode or "").strip().lower()
        if class_weight_mode in {"balanced", "balanced_sqrt"}:
            pos = float(np.sum(y_bin == 1.0))
            neg = float(np.sum(y_bin == 0.0))
            if pos > 0.0 and neg > 0.0:
                pos_w = float((pos + neg) / (2.0 * pos))
                neg_w = float((pos + neg) / (2.0 * neg))
                if class_weight_mode == "balanced_sqrt":
                    power = float(np.clip(self.class_weight_power, 0.0, 1.0))
                    pos_w = float(np.power(pos_w, power))
                    neg_w = float(np.power(neg_w, power))
                weights = weights * np.where(y_bin > 0.5, pos_w, neg_w).astype(np.float32)
        weights = np.clip(weights, 1e-3, 1e3)

        validation_split = 0.0
        callbacks = []
        if len(y_bin) >= 256:
            validation_split = 0.15
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    restore_best_weights=True,
                )
            )

        self._model.fit(
            x_arr,
            y_bin,
            sample_weight=weights,
            epochs=int(self.epochs),
            batch_size=int(min(max(self.batch_size, 1), len(x_arr))),
            verbose=0,
            validation_split=float(validation_split),
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, x):
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim != 3:
            raise ValueError("sequence_transformer Stage-A expects a 3D tensor [n, t, d].")
        n = int(len(x_arr))
        if self._constant_occupied_prob is not None:
            occ = np.full(shape=(n,), fill_value=float(self._constant_occupied_prob), dtype=float)
        else:
            if self._model is None:
                raise RuntimeError("sequence_transformer Stage-A model is not fitted.")
            occ = np.asarray(self._model.predict(x_arr, verbose=0), dtype=float).reshape(-1)
        occ = np.clip(occ, 0.0, 1.0)
        return np.column_stack([1.0 - occ, occ]).astype(float)


class EventFirstTwoStageModel:
    """Two-stage classifier: occupancy first, then occupied activity."""

    def __init__(self, config: Optional[EventFirstConfig] = None) -> None:
        self.config = config or EventFirstConfig()
        self.stage_a: Optional[object] = None
        self.stage_b: Optional[RandomForestClassifier] = None
        self._fallback_label: Optional[str] = None
        self._occupancy_threshold: float = 0.5
        self._activity_thresholds: Dict[str, float] = {}
        self._occupancy_calibration_method: str = "none"
        self._occupancy_isotonic: Optional[IsotonicRegression] = None
        self._occupancy_platt: Optional[LogisticRegression] = None
        self._stage_a_model_type: str = "rf"
        self._stage_a_temporal_lag_windows: int = 0
        self._stage_a_temporal_include_delta: bool = False
        self._stage_a_sequence_filter_enabled: bool = False
        self._stage_a_sequence_transition_matrix: Optional[np.ndarray] = None
        self._stage_a_sequence_initial_state: Optional[np.ndarray] = None

    def fit(
        self,
        x: np.ndarray,
        y: Sequence[str],
        *,
        stage_a_sample_weight: Optional[np.ndarray] = None,
        stage_a_group_ids: Optional[np.ndarray] = None,
        stage_a_group_occupied_ratio_threshold: float = 0.5,
    ) -> "EventFirstTwoStageModel":
        x_stage_a = self._prepare_stage_a_features(x, fit_mode=True)
        y_arr = np.asarray(y)
        y_occ = np.where(y_arr == self.config.occupancy_class, self.config.occupancy_class, "occupied")

        stage_a_sw = None if stage_a_sample_weight is None else np.asarray(stage_a_sample_weight, dtype=float)
        if stage_a_sw is not None and len(stage_a_sw) != len(y_occ):
            stage_a_sw = None
        stage_a_group_arr = self._validate_stage_a_group_ids(stage_a_group_ids, n=len(y_occ))
        x_stage_a_model = x_stage_a
        y_occ_model = np.asarray(y_occ, dtype=object)
        stage_a_sw_model = stage_a_sw
        if stage_a_group_arr is not None:
            x_stage_a_model, y_occ_model, stage_a_sw_model = self._aggregate_stage_a_training_inputs(
                x_stage_a=x_stage_a_model,
                y_occ=y_occ_model,
                stage_a_sample_weight=stage_a_sw_model,
                stage_a_group_ids=stage_a_group_arr,
                occupied_ratio_threshold=float(np.clip(stage_a_group_occupied_ratio_threshold, 0.0, 1.0)),
            )

        self.stage_a = self._build_stage_a_estimator()
        if stage_a_sw_model is not None:
            self.stage_a.fit(x_stage_a_model, y_occ_model, sample_weight=stage_a_sw_model)
        else:
            self.stage_a.fit(x_stage_a_model, y_occ_model)
        self._fit_stage_a_sequence_filter(y_occ_model)

        occupied_mask = y_arr != self.config.occupancy_class
        y_occupied = y_arr[occupied_mask]

        if len(y_occupied) == 0:
            self.stage_b = None
            self._fallback_label = self.config.occupancy_class
            return self

        unique_occupied = np.unique(y_occupied)
        if len(unique_occupied) == 1:
            self.stage_b = None
            self._fallback_label = str(unique_occupied[0])
            return self

        self.stage_b = RandomForestClassifier(
            n_estimators=self.config.n_estimators_stage_b,
            random_state=self.config.random_state + 1337,
            min_samples_leaf=self.config.min_samples_leaf,
            class_weight=self.config.stage_b_class_weight,
            n_jobs=1,
        )
        self.stage_b.fit(x[occupied_mask], y_occupied)
        self._fallback_label = None
        return self

    def predict_occupancy_proba(
        self,
        x: np.ndarray,
        *,
        stage_a_group_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return calibrated occupancy probability P(occupied)."""
        if self.stage_a is None:
            raise RuntimeError("Model not fitted.")
        raw = self._raw_occupancy_probability(x, stage_a_group_ids=stage_a_group_ids)
        return self._apply_occupancy_calibration(raw)

    def predict_activity_proba(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Return activity probabilities for occupied labels."""
        if self.stage_a is None:
            raise RuntimeError("Model not fitted.")
        if self.stage_b is None:
            if self._fallback_label and self._fallback_label != self.config.occupancy_class:
                return {str(self._fallback_label): np.ones(shape=(x.shape[0],), dtype=float)}
            return {}
        p_occ = self.stage_b.predict_proba(x)
        classes = [str(c) for c in self.stage_b.classes_]
        return {label: p_occ[:, idx].astype(float) for idx, label in enumerate(classes)}

    def tune_operating_points(
        self,
        x_calib: np.ndarray,
        y_calib: Sequence[str],
        *,
        occupancy_threshold_grid: Optional[np.ndarray] = None,
        activity_threshold_grid: Optional[np.ndarray] = None,
        calibration_method: str = "none",
        min_samples: int = 80,
        min_label_support: int = 20,
        calib_stage_a_group_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        Tune occupancy threshold and per-label activity thresholds on calibration data.
        """
        if self.stage_a is None:
            raise RuntimeError("Model not fitted.")
        y_arr = np.asarray(y_calib).astype(str)
        if len(y_arr) == 0 or len(x_calib) == 0:
            return {
                "used": False,
                "reason": "empty_calibration_data",
                "occupancy_threshold": self._occupancy_threshold,
                "activity_thresholds": dict(self._activity_thresholds),
                "calibration_method": self._occupancy_calibration_method,
            }

        method = str(calibration_method or "none").strip().lower()
        if method not in {"none", "isotonic", "platt"}:
            method = "none"
        self._reset_occupancy_calibration()

        y_occ_true = (y_arr != self.config.occupancy_class).astype(int)
        raw_occ = self._raw_occupancy_probability(
            x_calib,
            stage_a_group_ids=calib_stage_a_group_ids,
        )

        can_calibrate = len(y_arr) >= int(min_samples) and len(np.unique(y_occ_true)) > 1
        if can_calibrate and method in {"isotonic", "platt"}:
            if method == "isotonic":
                self._occupancy_isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                self._occupancy_isotonic.fit(raw_occ, y_occ_true)
            else:
                self._occupancy_platt = LogisticRegression(max_iter=500)
                self._occupancy_platt.fit(raw_occ.reshape(-1, 1), y_occ_true)
            self._occupancy_calibration_method = method
        else:
            self._occupancy_calibration_method = "none"
        calibrated_occ = self._apply_occupancy_calibration(raw_occ)

        occ_grid = (
            np.asarray(occupancy_threshold_grid, dtype=float)
            if occupancy_threshold_grid is not None
            else np.linspace(0.20, 0.80, num=61, dtype=float)
        )
        occ_threshold = self._select_best_binary_threshold(
            calibrated_occ,
            y_occ_true,
            occ_grid,
            target_positive_rate=float(np.mean(y_occ_true)),
            positive_rate_penalty=0.25,
        )
        self._occupancy_threshold = float(occ_threshold)

        activity_thresholds: Dict[str, float] = {}
        if self.stage_b is not None:
            occupied_mask = y_occ_true.astype(bool)
            if np.any(occupied_mask):
                y_occ_labels = y_arr[occupied_mask]
                p_occ_labels = self.stage_b.predict_proba(x_calib[occupied_mask])
                class_names = [str(c) for c in self.stage_b.classes_]
                act_grid = (
                    np.asarray(activity_threshold_grid, dtype=float)
                    if activity_threshold_grid is not None
                    else np.linspace(0.30, 0.90, num=61, dtype=float)
                )
                for idx, label in enumerate(class_names):
                    y_true_label = (y_occ_labels == label).astype(int)
                    support = int(np.sum(y_true_label))
                    if support < int(min_label_support):
                        activity_thresholds[label] = float(self.config.default_activity_threshold)
                        continue
                    threshold = self._select_best_binary_threshold(
                        p_occ_labels[:, idx].astype(float), y_true_label, act_grid
                    )
                    activity_thresholds[label] = float(threshold)
        self._activity_thresholds = activity_thresholds

        return {
            "used": True,
            "n_calibration": int(len(y_arr)),
            "occupancy_threshold": float(self._occupancy_threshold),
            "activity_thresholds": dict(self._activity_thresholds),
            "calibration_method": self._occupancy_calibration_method,
            "calibration_applied": bool(self._occupancy_calibration_method != "none"),
            "occupancy_positive_rate": float(np.mean(y_occ_true)),
        }

    def get_operating_points(self) -> Dict[str, object]:
        """Return currently active operating points."""
        return {
            "occupancy_threshold": float(self._occupancy_threshold),
            "activity_thresholds": dict(self._activity_thresholds),
            "calibration_method": str(self._occupancy_calibration_method),
            "stage_a_model_type": str(self._stage_a_model_type),
            "stage_a_temporal_lag_windows": int(self._stage_a_temporal_lag_windows),
            "stage_a_sequence_filter_enabled": bool(self._stage_a_sequence_filter_enabled),
            "stage_a_sequence_transition_matrix": (
                np.asarray(self._stage_a_sequence_transition_matrix, dtype=float).tolist()
                if self._stage_a_sequence_transition_matrix is not None
                else []
            ),
            "stage_a_sequence_initial_state": (
                np.asarray(self._stage_a_sequence_initial_state, dtype=float).tolist()
                if self._stage_a_sequence_initial_state is not None
                else []
            ),
        }

    def predict(
        self,
        x: np.ndarray,
        *,
        occupancy_threshold: Optional[float] = None,
        label_thresholds: Optional[Dict[str, float]] = None,
        stage_a_group_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.stage_a is None:
            raise RuntimeError("Model not fitted.")

        out = np.full(shape=(x.shape[0],), fill_value=self.config.occupancy_class, dtype=object)
        effective_occ_threshold = (
            float(self._occupancy_threshold) if occupancy_threshold is None else float(occupancy_threshold)
        )
        p_occupied = self.predict_occupancy_proba(
            x,
            stage_a_group_ids=stage_a_group_ids,
        )
        occupied_mask = p_occupied >= effective_occ_threshold

        if not occupied_mask.any():
            return out

        if self.stage_b is None:
            out[occupied_mask] = self._fallback_label or "occupied"
            return out

        x_occ = x[occupied_mask]
        p_occ = self.stage_b.predict_proba(x_occ)
        classes = list(self.stage_b.classes_)
        resolved_thresholds = dict(self._activity_thresholds)
        if label_thresholds:
            for key, value in label_thresholds.items():
                resolved_thresholds[str(key)] = float(value)
        class_thresholds = np.array(
            [float(resolved_thresholds.get(str(lbl), self.config.default_activity_threshold)) for lbl in classes],
            dtype=float,
        )
        adjusted_scores = p_occ / np.clip(class_thresholds.reshape(1, -1), 1e-6, 1.0)
        idx = np.argmax(adjusted_scores, axis=1)
        chosen_labels = np.array(classes, dtype=object)[idx]
        chosen_probs = p_occ[np.arange(p_occ.shape[0]), idx]
        chosen_thresholds = class_thresholds[idx]

        if self.config.use_unknown_for_low_confidence:
            low_conf_mask = chosen_probs < chosen_thresholds
            chosen_labels = chosen_labels.astype(object)
            chosen_labels[low_conf_mask] = self.config.unknown_label
        out[occupied_mask] = chosen_labels
        return out

    def _raw_occupancy_probability(
        self,
        x: np.ndarray,
        *,
        stage_a_group_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.stage_a is None:
            raise RuntimeError("Model not fitted.")
        x_stage_a = self._prepare_stage_a_features(x, fit_mode=False)
        group_ids = self._validate_stage_a_group_ids(stage_a_group_ids, n=x_stage_a.shape[0])
        if group_ids is None:
            stage_a_probs = self.stage_a.predict_proba(x_stage_a)
            stage_a_classes = list(self.stage_a.classes_)
            if "occupied" not in stage_a_classes:
                return np.zeros(shape=(x.shape[0],), dtype=float)
            occupied_idx = stage_a_classes.index("occupied")
            raw_occ = stage_a_probs[:, occupied_idx].astype(float)
        else:
            unique_group_ids, inverse = np.unique(group_ids, return_inverse=True)
            x_grouped = np.zeros((len(unique_group_ids),) + tuple(x_stage_a.shape[1:]), dtype=float)
            for idx in range(len(unique_group_ids)):
                mask = inverse == idx
                x_grouped[idx] = np.mean(x_stage_a[mask], axis=0)
            stage_a_probs_group = self.stage_a.predict_proba(x_grouped)
            stage_a_classes = list(self.stage_a.classes_)
            if "occupied" not in stage_a_classes:
                return np.zeros(shape=(x.shape[0],), dtype=float)
            occupied_idx = stage_a_classes.index("occupied")
            raw_occ_group = stage_a_probs_group[:, occupied_idx].astype(float)
            raw_occ = np.asarray(raw_occ_group[inverse], dtype=float)
        return self._apply_stage_a_sequence_filter(raw_occ)

    @staticmethod
    def _validate_stage_a_group_ids(
        stage_a_group_ids: Optional[np.ndarray],
        *,
        n: int,
    ) -> Optional[np.ndarray]:
        if stage_a_group_ids is None:
            return None
        arr = np.asarray(stage_a_group_ids)
        if arr.ndim != 1 or len(arr) != int(n):
            raise ValueError("stage_a_group_ids must be a 1D array with length equal to sample count.")
        if len(arr) <= 1:
            return None
        if np.issubdtype(arr.dtype, np.number):
            numeric = np.asarray(arr, dtype=float)
            out = np.zeros(shape=(len(numeric),), dtype=np.int64)
            finite_mask = np.isfinite(numeric)
            if np.any(finite_mask):
                out[finite_mask] = numeric[finite_mask].astype(np.int64)
                next_gid = int(np.max(out[finite_mask])) + 1
            else:
                next_gid = 0
            for idx in np.where(~finite_mask)[0]:
                out[idx] = int(next_gid)
                next_gid += 1
            return out

        normalized = np.asarray([str(v) for v in arr], dtype=object)
        _, inverse = np.unique(normalized, return_inverse=True)
        return inverse.astype(np.int64)

    def _aggregate_stage_a_training_inputs(
        self,
        *,
        x_stage_a: np.ndarray,
        y_occ: np.ndarray,
        stage_a_sample_weight: Optional[np.ndarray],
        stage_a_group_ids: np.ndarray,
        occupied_ratio_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        x_arr = np.asarray(x_stage_a, dtype=float)
        y_arr = np.asarray(y_occ, dtype=object)
        if len(x_arr) <= 0 or len(y_arr) <= 0:
            return x_arr, y_arr, stage_a_sample_weight

        unique_group_ids, inverse = np.unique(np.asarray(stage_a_group_ids, dtype=np.int64), return_inverse=True)
        if len(unique_group_ids) >= len(y_arr):
            return x_arr, y_arr, stage_a_sample_weight

        x_grouped = np.zeros((len(unique_group_ids),) + tuple(x_arr.shape[1:]), dtype=float)
        y_grouped = np.full(shape=(len(unique_group_ids),), fill_value=self.config.occupancy_class, dtype=object)
        sw_grouped = (
            np.zeros(shape=(len(unique_group_ids),), dtype=float)
            if stage_a_sample_weight is not None
            else None
        )
        for idx in range(len(unique_group_ids)):
            mask = inverse == idx
            x_grouped[idx] = np.mean(x_arr[mask], axis=0)
            occ_ratio = float(np.mean(np.asarray(y_arr[mask] == "occupied", dtype=float)))
            y_grouped[idx] = "occupied" if occ_ratio >= float(occupied_ratio_threshold) else self.config.occupancy_class
            if sw_grouped is not None and stage_a_sample_weight is not None:
                sw_grouped[idx] = float(np.mean(np.asarray(stage_a_sample_weight[mask], dtype=float)))

        x_grouped = np.nan_to_num(x_grouped, nan=0.0, posinf=0.0, neginf=0.0)
        if sw_grouped is not None:
            sw_grouped = np.clip(np.asarray(sw_grouped, dtype=float), 1e-4, 1e4)
        return x_grouped, y_grouped, sw_grouped

    @staticmethod
    def _is_sequence_stage_a_mode(mode: str) -> bool:
        txt = str(mode or "").strip().lower()
        return txt in {"sequence_rf", "sequence_hgb", "sequence_transformer"}

    def _fit_stage_a_sequence_filter(self, y_occ: Sequence[str]) -> None:
        self._stage_a_sequence_filter_enabled = False
        self._stage_a_sequence_transition_matrix = None
        self._stage_a_sequence_initial_state = None
        if not self._is_sequence_stage_a_mode(self._stage_a_model_type):
            return
        if str(self._stage_a_model_type or "") == "sequence_transformer":
            if not bool(getattr(self.config, "stage_a_transformer_use_sequence_filter", True)):
                return

        y_arr = np.asarray(y_occ, dtype=object)
        if len(y_arr) <= 0:
            return
        states = np.where(y_arr == "occupied", 1, 0).astype(int)

        transition_counts = np.ones((2, 2), dtype=float)
        if len(states) >= 2:
            for prev_state, curr_state in zip(states[:-1], states[1:]):
                transition_counts[int(prev_state), int(curr_state)] += 1.0
        transition_row_sums = np.clip(np.sum(transition_counts, axis=1, keepdims=True), 1e-9, None)
        transition_matrix = transition_counts / transition_row_sums

        initial_counts = np.ones((2,), dtype=float)
        initial_counts[0] += float(np.sum(states == 0))
        initial_counts[1] += float(np.sum(states == 1))
        initial_state = initial_counts / float(np.sum(initial_counts))

        self._stage_a_sequence_filter_enabled = True
        self._stage_a_sequence_transition_matrix = transition_matrix.astype(float)
        self._stage_a_sequence_initial_state = initial_state.astype(float)

    def _apply_stage_a_sequence_filter(self, probabilities: np.ndarray) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=float)
        if (
            not bool(self._stage_a_sequence_filter_enabled)
            or self._stage_a_sequence_transition_matrix is None
            or self._stage_a_sequence_initial_state is None
            or len(probs) <= 0
        ):
            return np.clip(probs, 0.0, 1.0).astype(float)

        transition = np.asarray(self._stage_a_sequence_transition_matrix, dtype=float)
        posterior = np.asarray(self._stage_a_sequence_initial_state, dtype=float)
        filtered = np.zeros(shape=(len(probs),), dtype=float)
        eps = 1e-4
        for idx, raw_occ in enumerate(probs):
            prior_state = posterior @ transition if idx > 0 else posterior
            occ = float(np.clip(raw_occ, eps, 1.0 - eps))
            emission = np.asarray([1.0 - occ, occ], dtype=float)
            joint = prior_state * emission
            norm = float(np.sum(joint))
            if norm <= 0.0:
                posterior = prior_state
            else:
                posterior = joint / norm
            filtered[idx] = float(posterior[1])
        return np.clip(filtered, 0.0, 1.0).astype(float)

    def _resolve_stage_a_spec(self) -> tuple[str, int, bool]:
        mode = str(getattr(self.config, "stage_a_model_type", "rf") or "rf").strip().lower()
        try:
            lag_windows = int(getattr(self.config, "stage_a_temporal_lag_windows", 0))
        except (TypeError, ValueError):
            lag_windows = 0
        lag_windows = int(max(lag_windows, 0))
        include_delta = bool(getattr(self.config, "stage_a_temporal_include_delta", True))

        if mode in {"sequence_transformer", "tiny_transformer", "causal_sequence_transformer"}:
            return "sequence_transformer", int(max(lag_windows, 1)), False
        if mode in {"sequence_hgb", "causal_sequence_hgb"}:
            return "sequence_hgb", int(max(lag_windows, 1)), bool(include_delta)
        if mode in {"temporal_hgb", "temporal_hist_gb"}:
            return "temporal_hgb", int(max(lag_windows, 1)), bool(include_delta)
        if mode in {"hgb", "hist_gb", "histgradientboosting", "hist_gradient_boosting"}:
            return "hgb", 0, False
        if mode in {"sequence_rf", "causal_sequence_rf"}:
            return "sequence_rf", int(max(lag_windows, 1)), bool(include_delta)
        if mode in {"temporal_rf"}:
            return "temporal_rf", int(max(lag_windows, 1)), bool(include_delta)
        return "rf", 0, False

    def _build_stage_a_estimator(self) -> object:
        mode = str(self._stage_a_model_type or "rf").strip().lower()
        if mode in {"sequence_transformer"}:
            return _TinySequenceTransformerStageA(
                random_state=int(self.config.random_state),
                epochs=int(max(getattr(self.config, "stage_a_transformer_epochs", 8), 1)),
                batch_size=int(max(getattr(self.config, "stage_a_transformer_batch_size", 256), 1)),
                learning_rate=float(max(getattr(self.config, "stage_a_transformer_learning_rate", 1e-3), 1e-5)),
                hidden_dim=int(max(getattr(self.config, "stage_a_transformer_hidden_dim", 32), 8)),
                num_heads=int(max(getattr(self.config, "stage_a_transformer_num_heads", 2), 1)),
                dropout=float(min(max(getattr(self.config, "stage_a_transformer_dropout", 0.1), 0.0), 0.5)),
                class_weight_power=float(
                    min(max(getattr(self.config, "stage_a_transformer_class_weight_power", 0.5), 0.0), 1.0)
                ),
                conv_kernel_size=int(max(getattr(self.config, "stage_a_transformer_conv_kernel_size", 3), 2)),
                conv_blocks=int(max(getattr(self.config, "stage_a_transformer_conv_blocks", 2), 1)),
                class_weight_mode=self.config.stage_a_class_weight,
            )
        if mode in {"hgb", "temporal_hgb", "sequence_hgb"}:
            return HistGradientBoostingClassifier(
                loss="log_loss",
                max_iter=int(max(self.config.n_estimators_stage_a, 50)),
                learning_rate=0.05,
                min_samples_leaf=int(max(self.config.min_samples_leaf, 2)),
                max_depth=None,
                random_state=int(self.config.random_state),
            )
        return RandomForestClassifier(
            n_estimators=self.config.n_estimators_stage_a,
            random_state=self.config.random_state,
            min_samples_leaf=self.config.min_samples_leaf,
            class_weight=self.config.stage_a_class_weight,
            n_jobs=1,
        )

    @staticmethod
    def _build_temporal_stage_a_matrix(
        x: np.ndarray,
        *,
        lag_windows: int,
        include_delta: bool,
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("Stage-A features must be a 2D array.")
        n, feature_dim = x_arr.shape
        lag_windows = int(max(lag_windows, 1))
        extra_dim = int(feature_dim if include_delta and lag_windows > 0 else 0)
        if n <= 0:
            return np.empty((0, (feature_dim * (lag_windows + 1)) + extra_dim), dtype=float)

        lag_blocks: list[np.ndarray] = [x_arr]
        for lag in range(1, lag_windows + 1):
            pad = np.repeat(x_arr[:1, :], repeats=lag, axis=0)
            shifted = np.vstack([pad, x_arr[:-lag, :]])
            lag_blocks.append(shifted)

        out = np.hstack(lag_blocks)
        if include_delta and lag_windows > 0:
            delta = lag_blocks[0] - lag_blocks[1]
            out = np.hstack([out, delta])
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _build_temporal_stage_a_tensor(
        x: np.ndarray,
        *,
        lag_windows: int,
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("Stage-A features must be a 2D array.")
        lag_windows = int(max(lag_windows, 1))
        lag_blocks: list[np.ndarray] = []
        for lag in range(0, lag_windows + 1):
            if lag == 0:
                shifted = x_arr
            else:
                pad = np.repeat(x_arr[:1, :], repeats=lag, axis=0)
                shifted = np.vstack([pad, x_arr[:-lag, :]])
            lag_blocks.append(np.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0))
        return np.stack(lag_blocks, axis=1).astype(float)

    def _prepare_stage_a_features(self, x: np.ndarray, *, fit_mode: bool) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("Stage-A features must be a 2D array.")
        if fit_mode:
            mode, lag_windows, include_delta = self._resolve_stage_a_spec()
            self._stage_a_model_type = str(mode)
            self._stage_a_temporal_lag_windows = int(lag_windows)
            self._stage_a_temporal_include_delta = bool(include_delta)
        else:
            mode = str(self._stage_a_model_type or "rf")
            lag_windows = int(max(self._stage_a_temporal_lag_windows, 0))
            include_delta = bool(self._stage_a_temporal_include_delta)

        if mode == "sequence_transformer" and lag_windows > 0:
            return self._build_temporal_stage_a_tensor(
                x_arr,
                lag_windows=lag_windows,
            )
        if mode not in {"temporal_rf", "temporal_hgb", "sequence_rf", "sequence_hgb"} or lag_windows <= 0:
            return np.nan_to_num(x_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return self._build_temporal_stage_a_matrix(
            x_arr,
            lag_windows=lag_windows,
            include_delta=include_delta,
        )

    def _reset_occupancy_calibration(self) -> None:
        self._occupancy_isotonic = None
        self._occupancy_platt = None
        self._occupancy_calibration_method = "none"

    def _apply_occupancy_calibration(self, raw_scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(raw_scores, dtype=float)
        method = str(self._occupancy_calibration_method)
        if method == "isotonic" and self._occupancy_isotonic is not None:
            return np.clip(self._occupancy_isotonic.predict(scores), 0.0, 1.0).astype(float)
        if method == "platt" and self._occupancy_platt is not None:
            return np.clip(
                self._occupancy_platt.predict_proba(scores.reshape(-1, 1))[:, 1],
                0.0,
                1.0,
            ).astype(float)
        return np.clip(scores, 0.0, 1.0).astype(float)

    @staticmethod
    def _binary_scores_from_threshold(
        probabilities: np.ndarray,
        y_true: np.ndarray,
        threshold: float,
    ) -> tuple[float, float, float]:
        y_pred = (probabilities >= float(threshold)).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    def _select_best_binary_threshold(
        self,
        probabilities: np.ndarray,
        y_true: np.ndarray,
        threshold_grid: np.ndarray,
        target_positive_rate: Optional[float] = None,
        positive_rate_penalty: float = 0.0,
    ) -> float:
        probs = np.asarray(probabilities, dtype=float)
        y = np.asarray(y_true, dtype=int)
        if len(probs) == 0 or len(np.unique(y)) <= 1:
            return 0.5
        best_t = 0.5
        best_score = (-1.0, -1.0, -1.0)
        for t in np.asarray(threshold_grid, dtype=float):
            precision, recall, f1 = self._binary_scores_from_threshold(probs, y, float(t))
            score_primary = float(f1)
            if target_positive_rate is not None and positive_rate_penalty > 0.0:
                pred_rate = float(np.mean(probs >= float(t)))
                score_primary -= float(positive_rate_penalty) * abs(float(target_positive_rate) - pred_rate)
            score = (score_primary, recall, precision)
            if score > best_score:
                best_t = float(t)
                best_score = score
        return float(np.clip(best_t, 0.0, 1.0))
