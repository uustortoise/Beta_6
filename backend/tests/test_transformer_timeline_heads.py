"""
Tests for WS-2: Transformer Timeline Heads

Tests multi-task head outputs and loss computation.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ml.transformer_timeline_heads import (
        TimelineHeadConfig,
        TimelineModelOutput,
        TransformerTimelineHeads,
        create_timeline_heads,
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow not available")
class TestTimelineHeadConfig(unittest.TestCase):
    """Tests for TimelineHeadConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TimelineHeadConfig()
        
        self.assertTrue(config.enable_activity)
        self.assertTrue(config.enable_occupancy)
        self.assertTrue(config.enable_boundary_start)
        self.assertTrue(config.enable_boundary_end)
        self.assertFalse(config.enable_daily_duration)
        self.assertFalse(config.enable_daily_count)
    
    def test_validation_passes(self):
        """Test that valid config passes."""
        config = TimelineHeadConfig(hidden_units=256, dropout_rate=0.2)
        config.validate()  # Should not raise
    
    def test_validation_fails_negative_units(self):
        """Test validation fails with negative units."""
        config = TimelineHeadConfig(hidden_units=-1)
        with self.assertRaises(AssertionError):
            config.validate()
    
    def test_validation_fails_invalid_dropout(self):
        """Test validation fails with invalid dropout."""
        config = TimelineHeadConfig(dropout_rate=1.5)
        with self.assertRaises(AssertionError):
            config.validate()


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow not available")
class TestTransformerTimelineHeads(unittest.TestCase):
    """Tests for TransformerTimelineHeads."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_len = 100
        self.hidden_dim = 64
        self.num_classes = 10
    
    def test_model_creation(self):
        """Test model can be created."""
        config = TimelineHeadConfig(num_activity_classes=self.num_classes)
        model = TransformerTimelineHeads(config)
        
        self.assertIsNotNone(model)
    
    def test_forward_pass_all_heads(self):
        """Test forward pass with all heads enabled."""
        config = TimelineHeadConfig(
            num_activity_classes=self.num_classes,
            enable_activity=True,
            enable_occupancy=True,
            enable_boundary_start=True,
            enable_boundary_end=True,
            enable_daily_duration=True,
            enable_daily_count=True,
        )
        model = TransformerTimelineHeads(config)
        
        # Create dummy encoder outputs
        encoder_outputs = tf.random.normal([self.batch_size, self.seq_len, self.hidden_dim])
        
        # Forward pass
        outputs = model(encoder_outputs, training=False)
        
        # Check all outputs exist and have correct shape
        self.assertIn("activity_logits", outputs)
        self.assertEqual(outputs["activity_logits"].shape, (self.batch_size, self.seq_len, self.num_classes))
        
        self.assertIn("occupancy_logits", outputs)
        self.assertEqual(outputs["occupancy_logits"].shape, (self.batch_size, self.seq_len, 1))
        
        self.assertIn("boundary_start_logits", outputs)
        self.assertEqual(outputs["boundary_start_logits"].shape, (self.batch_size, self.seq_len, 1))
        
        self.assertIn("boundary_end_logits", outputs)
        self.assertEqual(outputs["boundary_end_logits"].shape, (self.batch_size, self.seq_len, 1))
        
        self.assertIn("daily_duration_pred", outputs)
        self.assertEqual(outputs["daily_duration_pred"].shape, (self.batch_size, 1))
        
        self.assertIn("daily_count_pred", outputs)
        self.assertEqual(outputs["daily_count_pred"].shape, (self.batch_size, 1))
    
    def test_forward_pass_selective_heads(self):
        """Test forward pass with selective heads."""
        config = TimelineHeadConfig(
            num_activity_classes=self.num_classes,
            enable_activity=True,
            enable_occupancy=False,
            enable_boundary_start=True,
            enable_boundary_end=False,
            enable_daily_duration=False,
            enable_daily_count=False,
        )
        model = TransformerTimelineHeads(config)
        
        encoder_outputs = tf.random.normal([self.batch_size, self.seq_len, self.hidden_dim])
        outputs = model(encoder_outputs, training=False)
        
        # Enabled heads
        self.assertIn("activity_logits", outputs)
        self.assertIn("boundary_start_logits", outputs)
        
        # Disabled heads
        self.assertNotIn("occupancy_logits", outputs)
        self.assertNotIn("boundary_end_logits", outputs)
        self.assertNotIn("daily_duration_pred", outputs)
        self.assertNotIn("daily_count_pred", outputs)

    def test_timeline_heads_emit_onset_offset_duration_outputs(self):
        """Timeline-native offline heads should emit onset/offset, continuity, and duration outputs."""
        config = TimelineHeadConfig(
            num_activity_classes=self.num_classes,
            enable_activity=False,
            enable_occupancy=False,
            enable_boundary_start=True,
            enable_boundary_end=True,
            enable_daily_duration=True,
            enable_daily_count=False,
            enable_continuity=True,
        )
        model = TransformerTimelineHeads(config)

        encoder_outputs = tf.random.normal([self.batch_size, self.seq_len, self.hidden_dim])
        outputs = model(encoder_outputs, training=False)

        self.assertIn("boundary_start_logits", outputs)
        self.assertIn("boundary_end_logits", outputs)
        self.assertIn("continuity_logits", outputs)
        self.assertIn("daily_duration_pred", outputs)
        self.assertEqual(outputs["continuity_logits"].shape, (self.batch_size, self.seq_len, 1))
    
    def test_loss_computation(self):
        """Test loss computation."""
        config = TimelineHeadConfig(
            num_activity_classes=self.num_classes,
            enable_activity=True,
            enable_occupancy=True,
            enable_boundary_start=True,
            enable_boundary_end=True,
        )
        model = TransformerTimelineHeads(config)
        
        encoder_outputs = tf.random.normal([self.batch_size, self.seq_len, self.hidden_dim])
        outputs = model(encoder_outputs, training=True)
        
        # Create dummy targets
        targets = {
            'activity_labels': tf.constant(np.random.randint(0, self.num_classes, (self.batch_size, self.seq_len))),
            'occupancy_labels': tf.constant(np.random.randint(0, 2, (self.batch_size, self.seq_len))),
            'boundary_start_labels': tf.constant(np.random.randint(0, 2, (self.batch_size, self.seq_len))),
            'boundary_end_labels': tf.constant(np.random.randint(0, 2, (self.batch_size, self.seq_len))),
        }
        
        losses = model.compute_loss(outputs, targets)
        
        # Check losses computed
        self.assertIn('activity', losses)
        self.assertIn('occupancy', losses)
        self.assertIn('boundary_start', losses)
        self.assertIn('boundary_end', losses)
        self.assertIn('total', losses)
        
        # Check total is sum of components
        expected_total = (
            config.w_activity * losses['activity'] +
            config.w_occupancy * losses['occupancy'] +
            config.w_boundary_start * losses['boundary_start'] +
            config.w_boundary_end * losses['boundary_end']
        )
        self.assertAlmostEqual(losses['total'].numpy(), expected_total.numpy(), places=5)
    
    def test_loss_with_missing_targets(self):
        """Test loss computation with missing targets."""
        config = TimelineHeadConfig(
            num_activity_classes=self.num_classes,
            enable_activity=True,
            enable_occupancy=True,
        )
        model = TransformerTimelineHeads(config)
        
        encoder_outputs = tf.random.normal([self.batch_size, self.seq_len, self.hidden_dim])
        outputs = model(encoder_outputs, training=True)
        
        # Missing occupancy targets
        targets = {
            'activity_labels': tf.constant(np.random.randint(0, self.num_classes, (self.batch_size, self.seq_len))),
        }
        
        losses = model.compute_loss(outputs, targets)
        
        # Only activity loss computed
        self.assertIn('activity', losses)
        self.assertNotIn('occupancy', losses)
    
    def test_determinism(self):
        """Test deterministic output for fixed input."""
        config = TimelineHeadConfig(num_activity_classes=self.num_classes)
        model = TransformerTimelineHeads(config)
        
        # Fixed seed input
        tf.random.set_seed(42)
        encoder_outputs = tf.random.normal([self.batch_size, self.seq_len, self.hidden_dim])
        
        # Two forward passes
        outputs1 = model(encoder_outputs, training=False)
        outputs2 = model(encoder_outputs, training=False)
        
        # Check equality
        self.assertTrue(
            np.allclose(outputs1["activity_logits"].numpy(), outputs2["activity_logits"].numpy())
        )


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow not available")
class TestFactoryFunction(unittest.TestCase):
    """Tests for factory function."""
    
    def test_create_timeline_heads_default(self):
        """Test factory with default settings."""
        model = create_timeline_heads(num_activity_classes=10)
        
        self.assertIsInstance(model, TransformerTimelineHeads)
        self.assertTrue(model.config.enable_boundary_start)
        self.assertTrue(model.config.enable_boundary_end)
        self.assertFalse(model.config.enable_daily_duration)
    
    def test_create_timeline_heads_with_auxiliary(self):
        """Test factory with auxiliary heads."""
        model = create_timeline_heads(
            num_activity_classes=10,
            enable_daily_auxiliary=True,
        )
        
        self.assertTrue(model.config.enable_daily_duration)
        self.assertTrue(model.config.enable_daily_count)


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow not available")
class TestTimelineModelOutput(unittest.TestCase):
    """Tests for TimelineModelOutput dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        output = TimelineModelOutput(
            activity_logits=tf.constant([[1.0, 2.0]]),
            occupancy_logits=tf.constant([[0.5]]),
        )
        
        d = output.to_dict()
        
        self.assertIn('activity_logits', d)
        self.assertIn('occupancy_logits', d)
        self.assertIn('boundary_start_logits', d)


if __name__ == '__main__':
    unittest.main()
