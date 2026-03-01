"""
Calibration Module

Probability calibration for timeline outputs:
- Temperature scaling for boundary probabilities
- Isotonic regression for activity/occupancy

Part of WS-3: Decoder v2 + Calibration
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibrationConfig:
    """Configuration for probability calibration."""
    
    method: str = "temperature"  # "temperature" or "isotonic"
    temperature_init: float = 1.0
    temperature_lr: float = 0.01
    temperature_epochs: int = 100
    calibration_fraction: float = 0.2  # Fraction of training data for calibration
    
    def validate(self) -> None:
        """Validate configuration."""
        assert self.method in ["temperature", "isotonic"], "method must be 'temperature' or 'isotonic'"
        assert self.temperature_init > 0
        assert 0 < self.calibration_fraction < 1


class TemperatureScaler:
    """
    Temperature scaling for probability calibration.
    
    Learns a single temperature parameter T to scale logits:
    p_calibrated = softmax(logits / T)
    
    Higher T -> softer probabilities (more uncertain)
    Lower T -> harder probabilities (more confident)
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self._fitted = False
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        epochs: int = 100,
    ) -> "TemperatureScaler":
        """
        Fit temperature parameter using NLL loss.
        
        Args:
            logits: Raw model logits [N, C] or [N] for binary
            labels: True labels [N]
            lr: Learning rate for optimization
            epochs: Number of optimization epochs
            
        Returns:
            self
        """
        # Initialize temperature
        temperature = self.temperature
        
        for epoch in range(epochs):
            # Forward pass
            scaled_logits = logits / temperature
            
            if len(logits.shape) == 1 or logits.shape[1] == 1:
                # Binary case
                probs = 1 / (1 + np.exp(-scaled_logits))
                # NLL gradient
                grad = np.mean((probs - labels) * logits) / (temperature ** 2)
            else:
                # Multiclass case
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                # Simplified gradient (one-hot labels)
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(len(labels)), labels] = 1
                grad = np.mean(np.sum((probs - one_hot) * logits, axis=1)) / (temperature ** 2)
            
            # Update temperature (gradient descent on NLL)
            temperature -= lr * grad
            temperature = max(0.1, temperature)  # Keep positive
        
        self.temperature = temperature
        self._fitted = True
        
        return self
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw logits [N, C] or [N]
            
        Returns:
            Calibrated probabilities
        """
        scaled_logits = logits / self.temperature
        
        if len(logits.shape) == 1 or logits.shape[1] == 1:
            # Binary case
            probs = 1 / (1 + np.exp(-scaled_logits))
            return probs
        else:
            # Multiclass case
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return probs
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'temperature': float(self.temperature),
            'fitted': self._fitted,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemperatureScaler":
        """Deserialize from dictionary."""
        scaler = cls(temperature=data.get('temperature', 1.0))
        scaler._fitted = data.get('fitted', False)
        return scaler


class IsotonicCalibrator:
    """
    Isotonic regression calibration for probabilities.
    
    Fits a monotonic transformation to map uncalibrated
    probabilities to calibrated ones.
    """
    
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self._fitted = False
    
    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> "IsotonicCalibrator":
        """
        Fit isotonic regression.
        
        Args:
            probs: Uncalibrated probabilities [N]
            labels: Binary labels [N]
            
        Returns:
            self
        """
        self.calibrator.fit(probs, labels)
        self._fitted = True
        return self
    
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.
        
        Args:
            probs: Uncalibrated probabilities [N]
            
        Returns:
            Calibrated probabilities [N]
        """
        return self.calibrator.predict(probs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        # Note: sklearn IsotonicRegression doesn't have native serialization
        # Store the fitted X and y for reconstruction
        return {
            'X_thresholds': self.calibrator.X_thresholds_.tolist() if hasattr(self.calibrator, 'X_thresholds_') else [],
            'y_thresholds': self.calibrator.y_thresholds_.tolist() if hasattr(self.calibrator, 'y_thresholds_') else [],
            'fitted': self._fitted,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IsotonicCalibrator":
        """Deserialize from dictionary."""
        calibrator = cls()
        if data.get('fitted', False) and 'X_thresholds' in data and 'y_thresholds' in data:
            # Reconstruct the calibrator
            X = np.array(data['X_thresholds'])
            y = np.array(data['y_thresholds'])
            calibrator.calibrator.fit(X, y)
            calibrator._fitted = True
        return calibrator


class MultiTaskCalibrator:
    """
    Calibration manager for all timeline outputs.
    
    Manages separate calibrators for:
    - Activity classification
    - Occupancy detection
    - Boundary start/end detection
    """
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        config.validate()
        
        self.activity_calibrator: Optional[IsotonicCalibrator] = None
        self.occupancy_calibrator: Optional[IsotonicCalibrator] = None
        self.boundary_start_calibrator: Optional[TemperatureScaler] = None
        self.boundary_end_calibrator: Optional[TemperatureScaler] = None
    
    def fit(
        self,
        logits_dict: Dict[str, np.ndarray],
        labels_dict: Dict[str, np.ndarray],
    ) -> "MultiTaskCalibrator":
        """
        Fit all calibrators.
        
        Args:
            logits_dict: Dictionary of logits for each head
            labels_dict: Dictionary of labels for each head
            
        Returns:
            self
        """
        # Activity calibration (isotonic for multi-class)
        if 'activity' in logits_dict and 'activity' in labels_dict:
            # Convert logits to probs first
            logits = logits_dict['activity']
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Use probability of predicted class for calibration
            pred_probs = np.max(probs, axis=1)
            pred_correct = (np.argmax(probs, axis=1) == labels_dict['activity']).astype(float)
            
            self.activity_calibrator = IsotonicCalibrator()
            self.activity_calibrator.fit(pred_probs, pred_correct)
        
        # Occupancy calibration
        if 'occupancy' in logits_dict and 'occupancy' in labels_dict:
            logits = logits_dict['occupancy'].squeeze()
            probs = 1 / (1 + np.exp(-logits))
            
            self.occupancy_calibrator = IsotonicCalibrator()
            self.occupancy_calibrator.fit(probs, labels_dict['occupancy'])
        
        # Boundary calibration (temperature scaling)
        if 'boundary_start' in logits_dict and 'boundary_start' in labels_dict:
            logits = logits_dict['boundary_start'].squeeze()
            labels = labels_dict['boundary_start']
            
            self.boundary_start_calibrator = TemperatureScaler()
            self.boundary_start_calibrator.fit(
                logits, labels,
                lr=self.config.temperature_lr,
                epochs=self.config.temperature_epochs,
            )
        
        if 'boundary_end' in logits_dict and 'boundary_end' in labels_dict:
            logits = logits_dict['boundary_end'].squeeze()
            labels = labels_dict['boundary_end']
            
            self.boundary_end_calibrator = TemperatureScaler()
            self.boundary_end_calibrator.fit(
                logits, labels,
                lr=self.config.temperature_lr,
                epochs=self.config.temperature_epochs,
            )
        
        return self
    
    def calibrate(
        self,
        logits_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Calibrate all outputs.
        
        Args:
            logits_dict: Dictionary of logits for each head
            
        Returns:
            Dictionary of calibrated probabilities
        """
        probs_dict = {}
        
        # Activity
        if 'activity' in logits_dict:
            logits = logits_dict['activity']
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            if self.activity_calibrator is not None:
                pred_probs = np.max(probs, axis=1)
                calibrated = self.activity_calibrator.calibrate(pred_probs)
                # Scale probabilities proportionally and renormalize
                scale_factors = calibrated / (pred_probs + 1e-8)
                probs = probs * scale_factors[:, None]
                probs = np.clip(probs, 1e-10, 1.0)
                # Renormalize to ensure sum = 1
                probs = probs / np.sum(probs, axis=1, keepdims=True)
            
            probs_dict['activity'] = probs
        
        # Occupancy
        if 'occupancy' in logits_dict:
            logits = logits_dict['occupancy'].squeeze()
            probs = 1 / (1 + np.exp(-logits))
            
            if self.occupancy_calibrator is not None:
                probs = self.occupancy_calibrator.calibrate(probs)
            
            probs_dict['occupancy'] = probs
        
        # Boundary start
        if 'boundary_start' in logits_dict:
            logits = logits_dict['boundary_start'].squeeze()
            
            if self.boundary_start_calibrator is not None:
                probs = self.boundary_start_calibrator.calibrate(logits)
            else:
                probs = 1 / (1 + np.exp(-logits))
            
            probs_dict['boundary_start'] = probs
        
        # Boundary end
        if 'boundary_end' in logits_dict:
            logits = logits_dict['boundary_end'].squeeze()
            
            if self.boundary_end_calibrator is not None:
                probs = self.boundary_end_calibrator.calibrate(logits)
            else:
                probs = 1 / (1 + np.exp(-logits))
            
            probs_dict['boundary_end'] = probs
        
        return probs_dict
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'config': {
                'method': self.config.method,
                'temperature_init': self.config.temperature_init,
                'temperature_lr': self.config.temperature_lr,
                'temperature_epochs': self.config.temperature_epochs,
                'calibration_fraction': self.config.calibration_fraction,
            },
            'activity_calibrator': self.activity_calibrator.to_dict() if self.activity_calibrator else None,
            'occupancy_calibrator': self.occupancy_calibrator.to_dict() if self.occupancy_calibrator else None,
            'boundary_start_calibrator': self.boundary_start_calibrator.to_dict() if self.boundary_start_calibrator else None,
            'boundary_end_calibrator': self.boundary_end_calibrator.to_dict() if self.boundary_end_calibrator else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiTaskCalibrator":
        """Deserialize from dictionary."""
        config = CalibrationConfig(**data.get('config', {}))
        calibrator = cls(config)
        
        if data.get('activity_calibrator'):
            calibrator.activity_calibrator = IsotonicCalibrator.from_dict(data['activity_calibrator'])
        if data.get('occupancy_calibrator'):
            calibrator.occupancy_calibrator = IsotonicCalibrator.from_dict(data['occupancy_calibrator'])
        if data.get('boundary_start_calibrator'):
            calibrator.boundary_start_calibrator = TemperatureScaler.from_dict(data['boundary_start_calibrator'])
        if data.get('boundary_end_calibrator'):
            calibrator.boundary_end_calibrator = TemperatureScaler.from_dict(data['boundary_end_calibrator'])
        
        return calibrator


def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute calibration metrics (ECE, Brier score).
    
    Args:
        probs: Predicted probabilities
        labels: True binary labels
        n_bins: Number of bins for ECE
        
    Returns:
        Dictionary with ECE and Brier score
    """
    # Expected Calibration Error
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            in_bin = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    # Brier score
    brier = np.mean((probs - labels) ** 2)
    
    return {
        'ece': float(ece),
        'brier': float(brier),
        'n_samples': int(len(probs)),
    }
