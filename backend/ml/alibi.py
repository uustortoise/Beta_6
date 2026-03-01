
"""
ALiBi (Attention with Linear Biases) implementation for TensorFlow/Keras.
Provides utilities to generate static attention biases based on token distance.
"""

import tensorflow as tf
import math

def get_slopes(n_heads):
    """
    Get the slopes for ALiBi attention biases.
    Returns a list of slopes in a geometric sequence.
    
    Ref: https://github.com/ofirpress/attention_with_linear_biases
    """
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    else:
        # For non-power-of-2 heads, we find the nearest power of 2 locally
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        return get_slopes_power_of_2(closest_power_of_2) + \
               get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]

def get_alibi_biases(seq_len, num_heads):
    """
    Generate ALiBi bias tensor for a given sequence length and number of heads.
    
    For an Encoder (Bidirectional), we use symmetric distance:
    bias[i, j] = -slope * |i - j|
    
    Args:
        seq_len: Length of the sequence (integer or tf.Tensor scalar)
        num_heads: Number of attention heads
        
    Returns:
        Tensor of shape (1, num_heads, seq_len, seq_len)
    """
    # 1. Generate slopes (m) -> Shape: (num_heads, 1, 1)
    slopes = tf.constant(get_slopes(num_heads), dtype=tf.float32)
    slopes = tf.reshape(slopes, (num_heads, 1, 1))
    
    # 2. Creates indices range [0, 1, ..., seq_len-1]
    # Handle both static int and dynamic tensor seq_len
    if isinstance(seq_len, int):
        context_position = tf.range(seq_len, dtype=tf.float32)
    else:
        context_position = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
        
    # 3. Create distance matrix |i - j| -> Shape: (seq_len, seq_len)
    # matching dimensions for broadcasting: (seq_len, 1) - (1, seq_len)
    positions_i = context_position[:, tf.newaxis]
    positions_j = context_position[tf.newaxis, :]
    
    # Symmetric distance for encoder: |i - j|
    distance_matrix = tf.abs(positions_i - positions_j)
    
    # 4. Multiply by slopes -> Shape: (num_heads, seq_len, seq_len)
    # (num_heads, 1, 1) * (1, seq_len, seq_len)
    alibi_biases = -1.0 * slopes * distance_matrix[tf.newaxis, :, :]
    
    # 5. Add batch dimension -> Shape: (1, num_heads, seq_len, seq_len)
    alibi_biases = alibi_biases[tf.newaxis, :, :, :]
    
    return alibi_biases
