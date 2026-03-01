"""
Advanced Anomaly Detection Module

Improvements:
- Validates input shapes
- Uses local ensemble model instances for per-call fitting to avoid mutating shared state
- Exposes per-call parameters for thresholds and window sizes
- Normalizes output columns and index alignment
- Adds model persistence helpers for learned patterns and optional ensemble model saving
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import logging
import joblib
import os

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Advanced anomaly detection with multiple methods."""

    def __init__(self, sensor_columns, z_threshold: float = 3.0, contamination: float = 0.1):
        self.sensor_columns = list(sensor_columns)
        self.z_threshold = float(z_threshold)
        self.contamination = float(contamination)

        # Default prototypes (not fit here). Use local instances for per-call fitting.
        self._isolation_forest_prototype = IsolationForest(contamination=self.contamination, random_state=42)
        self._elliptic_envelope_prototype = EllipticEnvelope(contamination=self.contamination, random_state=42)
        self.scaler = StandardScaler()

        # Learning storage
        self.confirmed_anomalies = []
        self.learned_patterns = {}
    
    def detect_z_score(self, data: Any, z_threshold: Optional[float] = None) -> np.ndarray:
        """Z-score based anomaly detection for 1-D numeric arrays."""
        arr = np.asarray(data)
        if arr.ndim != 1:
            raise ValueError("detect_z_score expects a 1-D array or Series")

        zt = self.z_threshold if z_threshold is None else float(z_threshold)
        mean = np.nanmean(arr)
        std = np.nanstd(arr)

        if std == 0 or np.isnan(std):
            return np.zeros_like(arr, dtype=bool)

        z_scores = np.abs((arr - mean) / std)
        z_scores = np.where(np.isnan(z_scores), 0, z_scores)
        return z_scores > zt
    
    def detect_temporal(self, data: Any, window: int = 10, z_threshold: Optional[float] = None) -> np.ndarray:
        """Temporal anomaly detection using rolling statistics."""
        arr = np.asarray(data)
        if len(arr) < window:
            return np.zeros(len(arr), dtype=bool)

        series = pd.Series(arr)
        rolling_mean = series.rolling(window=window, center=True, min_periods=1).mean()
        rolling_std = series.rolling(window=window, center=True, min_periods=1).std().fillna(0)

        # Handle zero standard deviation with epsilon
        eps = 1e-6
        rolling_std = rolling_std.replace(0, eps)

        deviation = np.abs(series - rolling_mean)
        zt = self.z_threshold if z_threshold is None else float(z_threshold)
        threshold = zt * rolling_std

        return (deviation > threshold).values
    
    def detect_ensemble(self, df: pd.DataFrame, min_samples: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Ensemble detection using multiple methods.

        Uses local model instances for fitting to avoid mutating shared state.
        Returns (combined_boolean_array, scores_dict).
        """
        min_samples = min_samples or max(20, len(self.sensor_columns) * 2)
        if len(df) < min_samples:
            logger.debug("Insufficient samples for ensemble detection: %s < %s", len(df), min_samples)
            return np.zeros(len(df), dtype=bool), {}

        sensor_data = df[self.sensor_columns].ffill().bfill()
        if sensor_data.shape[0] == 0:
            return np.zeros(len(df), dtype=bool), {}

        # Scale with error handling and logging
        try:
            scaled = self.scaler.fit_transform(sensor_data)
        except Exception as e:
            logger.warning("Scaler failed, using raw values: %s", e)
            scaled = sensor_data.values.astype(float)

        # Fit local IsolationForest and EllipticEnvelope instances
        try:
            iso = IsolationForest(contamination=self.contamination, random_state=42)
            iso_pred = iso.fit_predict(scaled)
            iso_anomalies = iso_pred == -1
        except Exception as e:
            logger.error("IsolationForest failed: %s", e)
            iso_anomalies = np.zeros(len(df), dtype=bool)

        try:
            env = EllipticEnvelope(contamination=self.contamination, random_state=42)
            env_pred = env.fit_predict(scaled)
            env_anomalies = env_pred == -1
        except Exception as e:
            logger.warning("EllipticEnvelope failed: %s", e)
            env_anomalies = np.zeros(len(df), dtype=bool)

        combined = iso_anomalies | env_anomalies

        scores = {
            'isolation_forest': iso_anomalies.astype(int),
            'elliptic_envelope': env_anomalies.astype(int),
            'combined': combined.astype(int)
        }

        return combined, scores
    
    def detect_combined(self, df: pd.DataFrame, z_threshold: Optional[float] = None, temporal_window: Optional[int] = None, min_samples: Optional[int] = None) -> pd.DataFrame:
        """Combined detection using z-score, temporal, and ensemble methods.

        Returns a DataFrame indexed like `df` with consistent columns:
        - {sensor}_z_score (0/1)
        - {sensor}_temporal (0/1)
        - {sensor}_anomaly (0/1)
        - z_score_combined, temporal_combined, ensemble_anomaly
        - final_anomaly_score, final_anomaly (0/1)
        """
        results = pd.DataFrame(index=df.index)

        # Z-score for each sensor
        z_anomalies = []
        for sensor in self.sensor_columns:
            if sensor in df.columns:
                try:
                    anomalies = self.detect_z_score(df[sensor].values, z_threshold=z_threshold)
                except Exception as e:
                    logger.warning("Z-score detection failed for %s: %s", sensor, e)
                    anomalies = np.zeros(len(df), dtype=bool)
                z_anomalies.append(anomalies)
                results[f'{sensor}_z_score'] = anomalies.astype(int)

        if z_anomalies:
            results['z_score_combined'] = np.any(z_anomalies, axis=0).astype(int)

        # Temporal for each sensor
        temporal_anomalies = []
        tw = temporal_window or 10
        for sensor in self.sensor_columns:
            if sensor in df.columns:
                try:
                    anomalies = self.detect_temporal(df[sensor].values, window=tw, z_threshold=z_threshold)
                except Exception as e:
                    logger.warning("Temporal detection failed for %s: %s", sensor, e)
                    anomalies = np.zeros(len(df), dtype=bool)
                temporal_anomalies.append(anomalies)
                results[f'{sensor}_temporal'] = anomalies.astype(int)

        if temporal_anomalies:
            results['temporal_combined'] = np.any(temporal_anomalies, axis=0).astype(int)

        # Ensemble detection
        ensemble_anomalies, ensemble_scores = self.detect_ensemble(df, min_samples=min_samples)
        results['ensemble_anomaly'] = ensemble_anomalies.astype(int)

        # Normalize results to ensure consistent columns and alignment
        results = self._normalize_results(results, df.index)

        return results
    
    def _normalize_results(self, results: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
        """Ensure consistent column names and index alignment."""
        results = results.reindex(index).fillna(0)

        for sensor in self.sensor_columns:
            anomaly_col = f'{sensor}_anomaly'
            if anomaly_col not in results.columns:
                z_col = f'{sensor}_z_score'
                t_col = f'{sensor}_temporal'

                z_anomaly = (results[z_col] > 0) if z_col in results.columns else pd.Series(False, index=results.index)
                t_anomaly = (results[t_col] > 0) if t_col in results.columns else pd.Series(False, index=results.index)

                results[anomaly_col] = (z_anomaly | t_anomaly).astype(int)

        if 'ensemble_anomaly' not in results.columns:
            results['ensemble_anomaly'] = 0

        if 'final_anomaly_score' not in results.columns:
            anomaly_cols = [c for c in results.columns if c.endswith('_anomaly') or 'combined' in c or 'ensemble' in c]
            if anomaly_cols:
                results['final_anomaly_score'] = results[anomaly_cols].mean(axis=1)
            else:
                results['final_anomaly_score'] = 0.0

        if 'final_anomaly' not in results.columns:
            results['final_anomaly'] = (results['final_anomaly_score'] > 0.5).astype(int)

        return results
    
    def confirm_anomaly(self, anomaly_data: Any, label: Optional[str] = None) -> None:
        """Confirm an anomaly for learning and update learned patterns."""
        anomaly_arr = np.asarray(anomaly_data)
        if anomaly_arr.ndim == 1:
            anomaly_arr = anomaly_arr.reshape(1, -1)

        self.confirmed_anomalies.append({
            'data': anomaly_arr,
            'label': label,
            'timestamp': pd.Timestamp.now()
        })

        self._update_learned_patterns(anomaly_arr, label)
        logger.info("Confirmed anomaly added (total: %s)", len(self.confirmed_anomalies))

    def _update_learned_patterns(self, anomaly_data: Any, label: Optional[str]) -> None:
        """Update learned patterns from confirmed anomalies."""
        anomaly_arr = np.asarray(anomaly_data)
        if anomaly_arr.ndim != 2:
            raise ValueError(f"anomaly_data must be 2-D array, got {anomaly_arr.ndim}-D")

        if label not in self.learned_patterns:
            self.learned_patterns[label] = {'count': 0, 'mean': None, 'std': None, 'examples': []}

        pattern = self.learned_patterns[label]
        pattern['count'] += 1
        pattern['examples'].append(anomaly_arr)

        if len(pattern['examples']) > 100:
            pattern['examples'] = pattern['examples'][-100:]

        if pattern['mean'] is None:
            pattern['mean'] = np.nanmean(anomaly_arr, axis=0)
            pattern['std'] = np.nanstd(anomaly_arr, axis=0)
        else:
            n = pattern['count']
            old_mean = pattern['mean']
            old_std = pattern['std']
            new_mean = old_mean + (np.nanmean(anomaly_arr, axis=0) - old_mean) / n
            new_std = np.sqrt(((n - 1) * old_std ** 2 + np.nanvar(anomaly_arr, axis=0)) / n)
            pattern['mean'] = new_mean
            pattern['std'] = new_std

    def save_learned_patterns(self, path: str = "models/anomaly/patterns.pkl") -> bool:
        """Save learned patterns to disk."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.learned_patterns, path)
            logger.info("Saved learned patterns to %s", path)
            return True
        except Exception as e:
            logger.error("Failed to save learned patterns to %s: %s", path, e)
            return False

    def load_learned_patterns(self, path: str = "models/anomaly/patterns.pkl") -> bool:
        """Load learned patterns from disk."""
        try:
            if os.path.exists(path):
                self.learned_patterns = joblib.load(path)
                logger.info("Loaded learned patterns from %s", path)
                return True
            logger.warning("Patterns file not found: %s", path)
            return False
        except Exception as e:
            logger.error("Failed to load learned patterns from %s: %s", path, e)
            return False

    def save_models(self, path: str = "models/anomaly/models.pkl") -> bool:
        """Persist ensemble model prototypes and scaler (optional)."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {
                'scaler': self.scaler,
                'contamination': self.contamination
            }
            joblib.dump(payload, path)
            logger.info("Saved detector models to %s", path)
            return True
        except Exception as e:
            logger.error("Failed to save detector models to %s: %s", path, e)
            return False

    def load_models(self, path: str = "models/anomaly/models.pkl") -> bool:
        """Load persisted models (if available)."""
        try:
            if os.path.exists(path):
                payload = joblib.load(path)
                self.scaler = payload.get('scaler', self.scaler)
                self.contamination = payload.get('contamination', self.contamination)
                logger.info("Loaded detector models from %s", path)
                return True
            logger.warning("Detector models file not found: %s", path)
            return False
        except Exception as e:
            logger.error("Failed to load detector models from %s: %s", path, e)
            return False
