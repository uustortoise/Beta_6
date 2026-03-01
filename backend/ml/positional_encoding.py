"""
Positional Encoding utilities for Transformer-based models.

This module provides various positional encoding schemes for the
Hybrid CNN-Transformer architecture in Beta 5.5.

Includes:
- Sinusoidal (absolute) positional encoding
- Relative positional encoding (ALiBi-style)
- Learnable positional embeddings
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
# Handle Keras serialization decorator safely across versions
try:
    from tensorflow.keras.saving import register_keras_serializable
except ImportError:
    from tensorflow.keras.utils import register_keras_serializable

import logging

logger = logging.getLogger(__name__)


@register_keras_serializable(package='Beta5.5')
class SinusoidalPositionalEncoding(Layer):
    """
    Classic sinusoidal positional encoding (Vaswani et al., 2017).
    
    Adds absolute position information to embeddings using sine and cosine
    functions of different frequencies.
    
    Note: This is the standard approach but may struggle with varying sequence
    lengths. Consider RelativePositionalEncoding for more robust handling.
    """
    
    def __init__(self, max_seq_length: int = 1024, d_model: int = 64, **kwargs):
        """
        Parameters:
        -----------
        max_seq_length : int
            Maximum sequence length to pre-compute encodings for.
        d_model : int
            Dimension of the model (embedding size).
        """
        super().__init__(**kwargs)
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        
        # Pre-compute positional encodings
        self.pos_encoding = self._create_positional_encoding()
    
    def _create_positional_encoding(self):
        """Create the positional encoding matrix."""
        positions = np.arange(self.max_seq_length)[:, np.newaxis]
        dims = np.arange(self.d_model)[np.newaxis, :]
        
        # Compute angles
        angles = positions / np.power(10000, (2 * (dims // 2)) / self.d_model)
        
        # Apply sin to even indices, cos to odd indices
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        
        # Add batch dimension
        pos_encoding = angles[np.newaxis, :, :]
        
        return tf.constant(pos_encoding, dtype=tf.float32)
    
    def call(self, x):
        """
        Add positional encoding to input.
        
        Parameters:
        -----------
        x : Tensor
            Input tensor of shape (batch_size, seq_length, d_model)
        
        Returns:
        --------
        Tensor
            Input with positional encoding added.
        """
        seq_length = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_length, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_seq_length': self.max_seq_length,
            'd_model': self.d_model
        })
        return config


@register_keras_serializable(package='Beta5.5')
class RelativePositionalEncoding(Layer):
    """
    Relative positional encoding inspired by ALiBi (Press et al., 2021).
    
    Instead of adding absolute position embeddings, this biases attention
    scores based on the relative distance between tokens. This approach:
    - Handles variable sequence lengths more gracefully
    - Generalizes better to sequences longer than seen during training
    - Is more suitable for time-series data with irregular intervals
    
    For sensor data, we use a simple linear distance penalty.
    """
    
    def __init__(self, num_heads: int = 4, max_distance: int = 512, **kwargs):
        """
        Parameters:
        -----------
        num_heads : int
            Number of attention heads.
        max_distance : int
            Maximum relative distance to consider.
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.max_distance = max_distance
        
        # Create slopes for each head (ALiBi-style)
        # Heads have different "locality" preferences
        slopes = self._get_slopes(num_heads)
        self.slopes = tf.constant(slopes, dtype=tf.float32)
    
    def _get_slopes(self, n_heads):
        """
        Get ALiBi slopes for each head.
        
        Different heads focus on different distances:
        - Some heads are "local" (large slope, quick decay)
        - Some heads are "global" (small slope, slow decay)
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(np.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if np.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)
        else:
            # Handle non-power-of-2 heads
            closest_power_of_2 = 2 ** int(np.log2(n_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)
            slopes.extend(extra_slopes[0::2][:n_heads - closest_power_of_2])
            return slopes
    
    def get_relative_positions(self, seq_length):
        """
        Create relative position matrix.
        
        Returns matrix where position[i, j] = i - j (relative distance).
        """
        positions = tf.range(seq_length, dtype=tf.float32)
        relative_positions = positions[:, tf.newaxis] - positions[tf.newaxis, :]
        return relative_positions
    
    def call(self, attention_scores, seq_length=None):
        """
        Apply relative positional bias to attention scores.
        
        Parameters:
        -----------
        attention_scores : Tensor
            Attention scores of shape (batch, heads, seq_len, seq_len)
        seq_length : int, optional
            Sequence length (inferred if not provided)
        
        Returns:
        --------
        Tensor
            Biased attention scores.
        """
        if seq_length is None:
            seq_length = tf.shape(attention_scores)[-1]
        
        # Get relative positions: (seq_len, seq_len)
        relative_positions = self.get_relative_positions(seq_length)
        
        # Clip to max distance
        relative_positions = tf.clip_by_value(
            relative_positions, 
            -self.max_distance, 
            self.max_distance
        )
        
        # Apply slopes to get biases: (heads, seq_len, seq_len)
        # Each head has a different slope (locality preference)
        biases = self.slopes[:, tf.newaxis, tf.newaxis] * tf.abs(relative_positions)
        
        # Subtract bias (penalize distant positions)
        return attention_scores - biases
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'max_distance': self.max_distance
        })
        return config


@register_keras_serializable(package='Beta5.5')
class LearnablePositionalEmbedding(Layer):
    """
    Learnable positional embeddings.
    
    Unlike sinusoidal encoding, these embeddings are learned during training.
    This can be more flexible but requires the sequence length to be fixed
    or padded consistently.
    """
    
    def __init__(self, max_seq_length: int = 512, d_model: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_length = max_seq_length
        self.d_model = d_model
    
    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            name='position_embeddings',
            shape=(self.max_seq_length, self.d_model),
            initializer='uniform',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, x):
        seq_length = tf.shape(x)[1]
        return x + self.position_embeddings[:seq_length, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_seq_length': self.max_seq_length,
            'd_model': self.d_model
        })
        return config


# =============================================================================
# Utility Functions
# =============================================================================

def get_positional_encoding(
    encoding_type: str = 'relative',
    max_seq_length: int = 512,
    d_model: int = 64,
    num_heads: int = 4
) -> Layer:
    """
    Factory function to create a positional encoding layer.
    
    Parameters:
    -----------
    encoding_type : str
        Type of encoding: 'sinusoidal', 'relative', or 'learnable'
    max_seq_length : int
        Maximum sequence length.
    d_model : int
        Model dimension (for sinusoidal and learnable).
    num_heads : int
        Number of attention heads (for relative).
    
    Returns:
    --------
    Layer
        Positional encoding layer.
    """
    if encoding_type == 'sinusoidal':
        return SinusoidalPositionalEncoding(max_seq_length, d_model)
    elif encoding_type == 'relative':
        return RelativePositionalEncoding(num_heads, max_seq_length)
    elif encoding_type == 'learnable':
        return LearnablePositionalEmbedding(max_seq_length, d_model)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
