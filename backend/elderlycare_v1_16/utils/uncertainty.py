"""
Uncertainty quantification utilities for model predictions.
"""

import numpy as np

def calculate_entropy(probabilities):
    """Calculate entropy from probability distributions"""
    eps = 1e-9  # Small constant to avoid log(0)
    return -np.sum(probabilities * np.log(probabilities + eps), axis=1)

def softmax_temperature_scaling(probabilities, temperature=1.0):
    """Apply temperature scaling to softmax probabilities"""
    scaled_logits = np.log(probabilities + 1e-9) / temperature
    scaled_probs = np.exp(scaled_logits)
    scaled_probs = scaled_probs / np.sum(scaled_probs, axis=1, keepdims=True)
    return scaled_probs

def calculate_confidence_interval(predictions, confidence_level=0.95):
    """
    Calculate confidence intervals for predictions.
    
    Parameters:
    - predictions: array of shape (n_samples, n_classes)
    - confidence_level: float between 0 and 1
    
    Returns:
    - lower_bounds: array of shape (n_samples,)
    - upper_bounds: array of shape (n_samples,)
    """
    # For classification, we can use the predicted probabilities
    # to estimate uncertainty
    mean_probs = np.mean(predictions, axis=1)
    std_probs = np.std(predictions, axis=1)
    
    # Z-score for confidence level (approximate for normal distribution)
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    lower_bounds = np.maximum(0, mean_probs - z_score * std_probs)
    upper_bounds = np.minimum(1, mean_probs + z_score * std_probs)
    
    return lower_bounds, upper_bounds
