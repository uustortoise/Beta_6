"""
Transformer Backbone for Beta 5.5 Hybrid CNN-Transformer Architecture.

This module provides the core Transformer encoder architecture for
offline/batch ADL analysis. It combines:
- CNN Embedding: Local sensor feature extraction
- Positional Encoding: Temporal context (relative by default)
- Multi-Head Self-Attention: Global context across the sequence
- Feed-Forward Networks: Non-linear transformations

Usage:
    from ml.transformer_backbone import build_transformer_model
    
    model = build_transformer_model(
        input_shape=(60, 10),  # 60 timesteps, 10 features
        num_classes=8,
        num_heads=4,
        d_model=64
    )
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, LayerNormalization,
    GlobalAveragePooling1D, MultiHeadAttention, Add
)
import logging
from .alibi import get_alibi_biases # Import ALiBi utility

from .positional_encoding import (
    SinusoidalPositionalEncoding,
    RelativePositionalEncoding,
    get_positional_encoding
)

# Handle Keras serialization decorator safely
try:
    from tensorflow.keras.saving import register_keras_serializable
except ImportError:
    from tensorflow.keras.utils import register_keras_serializable

logger = logging.getLogger(__name__)


@register_keras_serializable(package='Beta5.5')
class TransformerEncoderBlock(layers.Layer):
    """
    Single Transformer encoder block with:
    - Multi-Head Self-Attention
    - Feed-Forward Network
    - Layer Normalization
    - Residual Connections
    """
    
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Multi-Head Attention
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        
        # Feed-Forward Network
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='gelu'),
            Dropout(dropout_rate),
            Dense(d_model)
        ])
        
        # Layer Normalization
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        
    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, x, training=False, mask=None, alibi_bias=None):
        # Self-Attention with residual connection
        # If alibi_bias is present, we combine it with the mask (Keras MHA supports bias addition via mask arg if properly shaped)
        # Note: Keras MHA 'attention_mask' argument: 
        # - boolean mask (True=attend, False=ignore)
        # - or float bias (added to attention scores)
        
        mha_mask = mask
        if alibi_bias is not None:
             if mask is not None:
                  # If we have both, we need to combine them. 
                  # Mask is usually 1/0 or boolean. ALiBi is float negative values.
                  # Since Keras MHA logic is complex, simpler approach for this version:
                  # Rely on alibi_bias being passed as 'attention_mask' argument if it exists.
                  # For padding: standard practice is to make ALiBi very negative where padded.
                  # Here we will prioritize ALiBi bias if provided.
                  mha_mask = alibi_bias
                  # Ideally: Add padding mask logic here if strictly needed, but batch/offline often uses fixed size.
             else:
                  mha_mask = alibi_bias
        
        attn_output = self.mha(x, x, attention_mask=mha_mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-Forward with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config


def build_cnn_embedding(input_shape, d_model=64, filters=(32, 64)):
    """
    Build CNN embedding layer for local sensor feature extraction.
    
    This replaces the first few layers of a standard Transformer's
    linear embedding, providing translation-invariant local features
    before global attention.
    """
    inputs = Input(shape=input_shape)
    
    # Two Conv1D layers for local feature extraction
    x = Conv1D(filters[0], kernel_size=3, activation='relu', padding='same')(inputs)
    x = Conv1D(filters[1], kernel_size=3, activation='relu', padding='same')(x)
    
    # Project to d_model dimension
    x = Conv1D(d_model, kernel_size=1, padding='same')(x)
    
    return Model(inputs, x, name='cnn_embedding')


def build_transformer_model(
    input_shape: tuple,
    num_classes: int,
    d_model: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_transformer_blocks: int = 2,
    dropout_rate: float = 0.2,
    positional_encoding_type: str = 'sinusoidal',
    use_cnn_embedding: bool = True,
    use_alibi: bool = False
) -> Model:
    """
    Build the full Hybrid CNN-Transformer model.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (seq_length, num_features)
    num_classes : int
        Number of output classes (activities)
    d_model : int
        Dimension of the Transformer model
    num_heads : int
        Number of attention heads
    ff_dim : int
        Dimension of feed-forward network
    num_transformer_blocks : int
        Number of stacked Transformer encoder blocks
    dropout_rate : float
        Dropout rate
    positional_encoding_type : str
        Type of positional encoding: 'sinusoidal', 'relative', 'learnable'
    use_cnn_embedding : bool
        If True, use CNN for initial embedding; else use linear projection
    
    Returns:
    --------
    Model
        Keras Model ready for training
    """
    inputs = Input(shape=input_shape)
    
    # === Embedding Layer ===
    if use_cnn_embedding:
        # CNN Embedding (local feature extraction)
        cnn_embed = build_cnn_embedding(input_shape, d_model)
        x = cnn_embed(inputs)
        logger.info(f"Using CNN embedding with d_model={d_model}")
    else:
        # Linear projection
        x = Dense(d_model)(inputs)
    
    # === Positional Encoding ===
    if use_alibi:
        logger.info("Using ALiBi: Skipping standard positional encoding")
        # Generate ALiBi bias dynamically based on input shape
        # We use a Lambda layer to compute it since seq_len might vary or we just compute once for static graph
        def compute_alibi_bias(tensor):
            seq_len = tf.shape(tensor)[1]
            return get_alibi_biases(seq_len, num_heads)
            
        alibi_bias_layer = layers.Lambda(compute_alibi_bias, name='alibi_bias_gen')
        alibi_bias = alibi_bias_layer(x)
    elif positional_encoding_type != 'relative':
        alibi_bias = None 
        # For sinusoidal and learnable, add to embeddings
        pos_enc = get_positional_encoding(
            encoding_type=positional_encoding_type,
            max_seq_length=input_shape[0],
            d_model=d_model
        )
        x = pos_enc(x)
        logger.info(f"Using {positional_encoding_type} positional encoding")
    else:
        # Relative encoding is applied inside attention (ALiBi-style)
        # For simplicity, we'll use sinusoidal as default in this version
        pos_enc = SinusoidalPositionalEncoding(
            max_seq_length=input_shape[0],
            d_model=d_model
        )
        x = pos_enc(x)
        logger.info("Using sinusoidal positional encoding (relative requires custom attention)")
    
    # === Transformer Encoder Blocks ===
    for i in range(num_transformer_blocks):
        x = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f'transformer_block_{i}'
        )(x, alibi_bias=alibi_bias)
    
    # === Output Head ===
    # Global pooling to aggregate sequence
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(ff_dim, activation='gelu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='hybrid_cnn_transformer')
    
    logger.info(f"Built Hybrid CNN-Transformer: input={input_shape}, classes={num_classes}, "
                f"d_model={d_model}, heads={num_heads}, blocks={num_transformer_blocks}")
    
    return model


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == '__main__':
    # Test model building
    logging.basicConfig(level=logging.INFO)
    
    model = build_transformer_model(
        input_shape=(60, 10),  # 60 timesteps (10 min @ 10s), 10 sensor features
        num_classes=8,
        d_model=64,
        num_heads=4,
        num_transformer_blocks=2
    )
    
    model.summary()
    
    # Test forward pass
    import numpy as np
    test_input = np.random.randn(2, 60, 10).astype(np.float32)
    output = model.predict(test_input, verbose=0)
    print(f"Output shape: {output.shape}")
    print(f"Output sum (should be ~1.0 per sample): {output.sum(axis=1)}")
