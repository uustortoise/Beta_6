"""
Custom exceptions for the ML pipeline.

This module defines a structured exception hierarchy for the UnifiedPipeline,
enabling specific error handling and better debugging in production.

Usage:
    from ml.exceptions import ModelLoadError, PredictionError
    
    try:
        model = load_model(...)
    except ModelLoadError as e:
        logger.error(f"Model loading failed: {e}")
        # Handle gracefully or re-raise
"""


class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    pass


class ModelLoadError(PipelineError):
    """
    Failed to load a trained model.
    
    Raised when:
    - Model file doesn't exist
    - Model file is corrupted
    - Custom layers cannot be deserialized
    - Both Keras and joblib fallback fail
    """
    pass


class ModelTrainError(PipelineError):
    """
    Failed during model training.
    
    Raised when:
    - Insufficient training data
    - Model compilation fails
    - Training diverges (NaN loss)
    - GPU/memory errors during fit()
    """
    pass


class PredictionError(PipelineError):
    """
    Failed during prediction inference.
    
    Raised when:
    - Input data shape mismatch
    - Model not loaded
    - Inference produces invalid output
    """
    pass


class DataValidationError(PipelineError):
    """
    Invalid or missing data.
    
    Raised when:
    - Required columns are missing
    - Data types are incorrect
    - Timestamp parsing fails
    - Sequence length requirements not met
    """
    pass


class DatabaseError(PipelineError):
    """
    Database operation failed.
    
    Raised when:
    - Connection fails
    - Query execution fails
    - Transaction commit fails
    """
    pass
