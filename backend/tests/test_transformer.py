
import unittest
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.transformer_backbone import build_transformer_model

class TestTransformerBackbone(unittest.TestCase):
    def setUp(self):
        self.input_shape = (60, 10) # 60 timesteps, 10 features
        self.num_classes = 5
        
    def test_build_model_basic(self):
        """Test building the model with default parameters."""
        model = build_transformer_model(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
        
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.output_shape, (None, self.num_classes))
        
        # Check if layers are created
        layer_names = [l.name for l in model.layers]
        self.assertTrue(any('cnn_embedding' in name for name in layer_names))
        self.assertTrue(any('transformer_block' in name for name in layer_names))

    def test_forward_pass(self):
        """Test a forward pass with random data."""
        model = build_transformer_model(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
        
        batch_size = 4
        dummy_input = np.random.randn(batch_size, *self.input_shape).astype(np.float32)
        
        output = model.predict(dummy_input, verbose=0)
        
        self.assertEqual(output.shape, (batch_size, self.num_classes))
        # Check softmax probabilities sum to 1
        sums = np.sum(output, axis=1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

    def test_custom_parameters(self):
        """Test building with custom hyperparameters."""
        d_model = 32
        num_heads = 2
        blocks = 1
        
        model = build_transformer_model(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            d_model=d_model,
            num_heads=num_heads,
            num_transformer_blocks=blocks,
            use_cnn_embedding=False # Use linear projection
        )
        
        # Verify d_model propagation (hard to check directly in compiled model without inspecting weights)
        # But we can check if it runs
        dummy_input = np.random.randn(1, *self.input_shape).astype(np.float32)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, self.num_classes))

if __name__ == '__main__':
    unittest.main()
