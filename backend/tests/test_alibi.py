
import unittest
import tensorflow as tf
import numpy as np
import sys
import os

# Add backend to path dynamically
# This handles running as script (python backend/tests/test_alibi.py)
# or via pytest where backend/ is usually in path via conftest
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(current_file))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from ml.alibi import get_slopes, get_alibi_biases
from ml.transformer_backbone import build_transformer_model

class TestAlibi(unittest.TestCase):
    
    def test_slopes_power_of_2(self):
        """Test slope generation for power-of-2 heads"""
        slopes = get_slopes(4)
        # Expected: 2^-8/4 => 2^-2 = 0.25. M = [0.25, 0.0625, ...] or similar geometric
        # Implementation is base^1, base^2.. where base = 2^(-8/n)
        # For n=4, base = 2^(-2) = 0.25
        # Slopes: [0.25, 0.0625, 0.015625, 0.00390625]
        expected = [0.25, 0.0625, 0.015625, 0.00390625]
        # Approximate check
        self.assertEqual(len(slopes), 4)
        np.testing.assert_allclose(slopes, expected, rtol=1e-5)

    def test_alibi_bias_shape(self):
        """Test shape of generated bias tensor"""
        seq_len = 10
        num_heads = 4
        bias = get_alibi_biases(seq_len, num_heads)
        
        # Expected shape: (1, num_heads, seq_len, seq_len)
        self.assertEqual(bias.shape, (1, num_heads, seq_len, seq_len))

    def test_alibi_bias_symmetry(self):
        """Test that encoder bias is symmetric (|i-j|)"""
        seq_len = 5
        num_heads = 1
        bias = get_alibi_biases(seq_len, num_heads)
        bias = bias.numpy()
        
        # Check symmetry [0, 0, :, :]
        matrix = bias[0, 0, :, :]
        # Diagonal should be 0
        self.assertEqual(matrix[0, 0], 0)
        # pos [0, 1] should equal pos [1, 0]
        self.assertEqual(matrix[0, 1], matrix[1, 0])
        # pos [0, 4] should be further than [0, 1] (more negative)
        self.assertLess(matrix[0, 4], matrix[0, 1])

    def test_model_build_with_alibi(self):
        """Test building and running the full model with ALiBi enabled"""
        model = build_transformer_model(
            input_shape=(20, 5),
            num_classes=2,
            d_model=16,
            num_heads=2,
            use_alibi=True  # ENABLE ALIBI
        )
        
        # Forward pass
        dummy_input = tf.random.normal((1, 20, 5))
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (1, 2))
        
if __name__ == '__main__':
    unittest.main()
