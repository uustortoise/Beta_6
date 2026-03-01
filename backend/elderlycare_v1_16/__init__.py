"""
Elderly Care Monitoring Platform v1.11

A comprehensive system for monitoring elderly behavior patterns,
detecting anomalies, and providing insights for caregivers.
"""

__version__ = "1.11.0"
__author__ = "Elderly Care Team"

# Core exports
from .platform import ElderlyCarePlatform
from .anomaly.detector import AnomalyDetector


# Enhanced Profile Management (v1.11)
try:
    from .profile import (
        EnhancedProfile,
        create_empty_profile,
        ProfileValidator,
        MedicalDataValidator,
        DEFAULT_PROFILE_TEMPLATE,
        COMMON_MEDICAL_CONDITIONS
    )
    PROFILE_AVAILABLE = True
except ImportError:
    PROFILE_AVAILABLE = False
    EnhancedProfile = None
    create_empty_profile = None
    ProfileValidator = None
    MedicalDataValidator = None
    DEFAULT_PROFILE_TEMPLATE = None
    COMMON_MEDICAL_CONDITIONS = None

# Utility exports (only import what exists)
try:
    from .utils.uncertainty import calculate_entropy, softmax_temperature_scaling
    UNCERTAINTY_AVAILABLE = True
except ImportError:
    UNCERTAINTY_AVAILABLE = False
    calculate_entropy = None
    softmax_temperature_scaling = None

try:
    from .preprocessing.noise import hampel_filter, clip_outliers
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False
    hampel_filter = None
    clip_outliers = None

try:
    from .preprocessing.sequences import create_sequences
    SEQUENCES_AVAILABLE = True
except ImportError:
    SEQUENCES_AVAILABLE = False
    create_sequences = None

# Create a simplified inference function if needed
def predict_room_activities(platform, room_name, df, **kwargs):
    """Simplified prediction function."""
    return platform.predict_room_activities(room_name, df, **kwargs)

def build_activity_model(input_shape, num_classes, **kwargs):
    """Simplified model builder."""
    import tensorflow as tf
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(32, 3, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    lstm_out = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    attention = tf.keras.layers.Attention()([lstm_out, lstm_out])
    attended = tf.keras.layers.GlobalMaxPooling1D()(attention)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(attended)
    return tf.keras.Model(inputs, outputs)

__all__ = [
    "ElderlyCarePlatform",
    "AnomalyDetector",
    "predict_room_activities",
    "build_activity_model",
    "calculate_entropy",
    "softmax_temperature_scaling",
    "hampel_filter",
    "clip_outliers",
    "create_sequences",
    "EnhancedProfile",
    "create_empty_profile",
    "ProfileValidator",
    "MedicalDataValidator",
    "DEFAULT_PROFILE_TEMPLATE",
    "COMMON_MEDICAL_CONDITIONS",
]
