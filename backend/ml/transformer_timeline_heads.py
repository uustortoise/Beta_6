"""
Transformer Timeline Heads Module

Multi-task heads for timeline-aware training:
- activity_logits: Existing activity classification (Head B)
- occupancy_logits: Occupancy detection (Head A)
- boundary_start_logits: Episode start boundary detection
- boundary_end_logits: Episode end boundary detection
- daily_duration_pred: Episode duration regression (auxiliary)
- daily_count_pred: Episode count regression (auxiliary)

Part of WS-2: Multi-Task Objective (CNN+Transformer)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras


@dataclass
class TimelineHeadConfig:
    """Configuration for timeline multi-task heads."""
    
    # Head enablement
    enable_activity: bool = True
    enable_occupancy: bool = True
    enable_boundary_start: bool = True
    enable_boundary_end: bool = True
    enable_continuity: bool = True
    enable_daily_duration: bool = False  # Optional auxiliary
    enable_daily_count: bool = False  # Optional auxiliary
    
    # Architecture
    hidden_units: int = 128
    dropout_rate: float = 0.1
    num_activity_classes: int = 10  # Will be set from registry
    
    # Loss weights
    w_activity: float = 1.0
    w_occupancy: float = 1.0
    w_boundary_start: float = 0.5
    w_boundary_end: float = 0.5
    w_continuity: float = 0.25
    w_daily_duration: float = 0.1
    w_daily_count: float = 0.1
    
    def validate(self) -> None:
        """Validate configuration."""
        assert self.hidden_units > 0, "hidden_units must be positive"
        assert 0 <= self.dropout_rate < 1, "dropout_rate must be in [0, 1)"
        assert all(w >= 0 for w in [
            self.w_activity, self.w_occupancy,
            self.w_boundary_start, self.w_boundary_end,
            self.w_continuity,
            self.w_daily_duration, self.w_daily_count
        ]), "all loss weights must be non-negative"


@dataclass
class TimelineModelOutput:
    """Output container for all timeline heads."""
    
    activity_logits: Optional[tf.Tensor] = None
    occupancy_logits: Optional[tf.Tensor] = None
    boundary_start_logits: Optional[tf.Tensor] = None
    boundary_end_logits: Optional[tf.Tensor] = None
    onset_logits: Optional[tf.Tensor] = None
    offset_logits: Optional[tf.Tensor] = None
    continuity_logits: Optional[tf.Tensor] = None
    daily_duration_pred: Optional[tf.Tensor] = None
    daily_count_pred: Optional[tf.Tensor] = None
    
    def to_dict(self) -> Dict[str, Optional[tf.Tensor]]:
        """Convert to dictionary."""
        return {
            'activity_logits': self.activity_logits,
            'occupancy_logits': self.occupancy_logits,
            'boundary_start_logits': self.boundary_start_logits,
            'boundary_end_logits': self.boundary_end_logits,
            'onset_logits': self.onset_logits,
            'offset_logits': self.offset_logits,
            'continuity_logits': self.continuity_logits,
            'daily_duration_pred': self.daily_duration_pred,
            'daily_count_pred': self.daily_count_pred,
        }


class TransformerTimelineHeads(keras.Model):
    """
    Multi-task timeline heads for transformer backbone.
    
    Takes encoder outputs and produces:
    - Per-window activity classification
    - Per-window occupancy detection
    - Per-window boundary start/end detection
    - Daily aggregated duration/count predictions (auxiliary)
    """
    
    def __init__(
        self,
        config: TimelineHeadConfig,
        name: str = "timeline_heads",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.config = config
        config.validate()
        
        # Shared head processing
        self.shared_dense = keras.layers.Dense(
            config.hidden_units,
            activation='relu',
            name='shared_dense'
        )
        self.dropout = keras.layers.Dropout(config.dropout_rate)
        
        # Activity head (Head B)
        if config.enable_activity:
            self.activity_head = keras.layers.Dense(
                config.num_activity_classes,
                name='activity_logits'
            )
        
        # Occupancy head (Head A)
        if config.enable_occupancy:
            self.occupancy_head = keras.layers.Dense(
                1,
                name='occupancy_logits'
            )
        
        # Boundary start head
        if config.enable_boundary_start:
            self.boundary_start_head = keras.layers.Dense(
                1,
                name='boundary_start_logits'
            )
        
        # Boundary end head
        if config.enable_boundary_end:
            self.boundary_end_head = keras.layers.Dense(
                1,
                name='boundary_end_logits'
            )
        if config.enable_continuity:
            self.continuity_head = keras.layers.Dense(
                1,
                name='continuity_logits'
            )
        
        # Daily duration regression (auxiliary)
        if config.enable_daily_duration:
            self.duration_pool = keras.layers.GlobalAveragePooling1D()
            self.duration_head = keras.layers.Dense(
                1,
                name='daily_duration_pred'
            )
        
        # Daily count regression (auxiliary)
        if config.enable_daily_count:
            self.count_pool = keras.layers.GlobalAveragePooling1D()
            self.count_head = keras.layers.Dense(
                1,
                name='daily_count_pred'
            )
    
    def call(
        self,
        encoder_outputs: tf.Tensor,
        training: Optional[bool] = None,
    ) -> Dict[str, tf.Tensor]:
        """
        Forward pass through all enabled heads.
        
        Args:
            encoder_outputs: [batch_size, seq_len, hidden_dim] from transformer encoder
            training: Whether in training mode
            
        Returns:
            Dictionary of enabled head logits/predictions.
        """
        # Shared processing
        shared = self.shared_dense(encoder_outputs)
        shared = self.dropout(shared, training=training)
        
        outputs = TimelineModelOutput()
        
        # Activity head (per-window classification)
        if self.config.enable_activity:
            outputs.activity_logits = self.activity_head(shared)
        
        # Occupancy head (per-window binary)
        if self.config.enable_occupancy:
            outputs.occupancy_logits = self.occupancy_head(shared)
        
        # Boundary heads (per-window binary)
        if self.config.enable_boundary_start:
            outputs.boundary_start_logits = self.boundary_start_head(shared)
            outputs.onset_logits = outputs.boundary_start_logits
        
        if self.config.enable_boundary_end:
            outputs.boundary_end_logits = self.boundary_end_head(shared)
            outputs.offset_logits = outputs.boundary_end_logits
        
        if self.config.enable_continuity:
            outputs.continuity_logits = self.continuity_head(shared)
        
        # Daily aggregated heads (sequence-level regression)
        if self.config.enable_daily_duration:
            pooled = self.duration_pool(shared)
            outputs.daily_duration_pred = self.duration_head(pooled)
        
        if self.config.enable_daily_count:
            pooled = self.count_pool(shared)
            outputs.daily_count_pred = self.count_head(pooled)
        
        return {k: v for k, v in outputs.to_dict().items() if v is not None}
    
    def compute_loss(
        self,
        outputs: Any,
        targets: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        """
        Compute weighted multi-task loss.
        
        Args:
            outputs: Model predictions from forward pass
            targets: Dictionary of target tensors
            
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        if isinstance(outputs, TimelineModelOutput):
            output_dict = outputs.to_dict()
        elif isinstance(outputs, dict):
            output_dict = outputs
        else:
            raise TypeError(f"Unsupported outputs type for compute_loss: {type(outputs)}")

        # Activity loss (sparse categorical crossentropy)
        activity_logits = output_dict.get("activity_logits")
        if self.config.enable_activity and activity_logits is not None:
            activity_targets = targets.get('activity_labels')
            if activity_targets is not None:
                losses['activity'] = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=activity_targets,
                        logits=activity_logits,
                    )
                )
        
        # Occupancy loss (binary crossentropy)
        occupancy_logits = output_dict.get("occupancy_logits")
        if self.config.enable_occupancy and occupancy_logits is not None:
            occupancy_targets = targets.get('occupancy_labels')
            if occupancy_targets is not None:
                losses['occupancy'] = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.cast(occupancy_targets, tf.float32),
                        logits=tf.squeeze(occupancy_logits, axis=-1),
                    )
                )
        
        # Boundary start loss (binary crossentropy)
        boundary_start_logits = output_dict.get("boundary_start_logits")
        if self.config.enable_boundary_start and boundary_start_logits is not None:
            start_targets = targets.get('boundary_start_labels')
            if start_targets is not None:
                losses['boundary_start'] = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.cast(start_targets, tf.float32),
                        logits=tf.squeeze(boundary_start_logits, axis=-1),
                    )
                )
        
        # Boundary end loss (binary crossentropy)
        boundary_end_logits = output_dict.get("boundary_end_logits")
        if self.config.enable_boundary_end and boundary_end_logits is not None:
            end_targets = targets.get('boundary_end_labels')
            if end_targets is not None:
                losses['boundary_end'] = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.cast(end_targets, tf.float32),
                        logits=tf.squeeze(boundary_end_logits, axis=-1),
                    )
                )
        continuity_logits = output_dict.get("continuity_logits")
        if self.config.enable_continuity and continuity_logits is not None:
            continuity_targets = targets.get('continuity_labels')
            if continuity_targets is not None:
                losses['continuity'] = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.cast(continuity_targets, tf.float32),
                        logits=tf.squeeze(continuity_logits, axis=-1),
                    )
                )
        
        # Daily duration loss (Huber)
        daily_duration_pred = output_dict.get("daily_duration_pred")
        if self.config.enable_daily_duration and daily_duration_pred is not None:
            duration_targets = targets.get('daily_duration')
            if duration_targets is not None:
                losses['daily_duration'] = tf.reduce_mean(
                    tf.keras.losses.huber(
                        tf.cast(duration_targets, tf.float32),
                        tf.squeeze(daily_duration_pred, axis=-1),
                    )
                )
        
        # Daily count loss (Huber)
        daily_count_pred = output_dict.get("daily_count_pred")
        if self.config.enable_daily_count and daily_count_pred is not None:
            count_targets = targets.get('daily_count')
            if count_targets is not None:
                losses['daily_count'] = tf.reduce_mean(
                    tf.keras.losses.huber(
                        tf.cast(count_targets, tf.float32),
                        tf.squeeze(daily_count_pred, axis=-1),
                    )
                )
        
        # Weighted total loss
        total_loss = tf.constant(0.0, dtype=tf.float32)
        if 'activity' in losses:
            total_loss += self.config.w_activity * losses['activity']
        if 'occupancy' in losses:
            total_loss += self.config.w_occupancy * losses['occupancy']
        if 'boundary_start' in losses:
            total_loss += self.config.w_boundary_start * losses['boundary_start']
        if 'boundary_end' in losses:
            total_loss += self.config.w_boundary_end * losses['boundary_end']
        if 'continuity' in losses:
            total_loss += self.config.w_continuity * losses['continuity']
        if 'daily_duration' in losses:
            total_loss += self.config.w_daily_duration * losses['daily_duration']
        if 'daily_count' in losses:
            total_loss += self.config.w_daily_count * losses['daily_count']
        
        losses['total'] = total_loss
        
        return losses


def create_timeline_heads(
    num_activity_classes: int,
    hidden_units: int = 128,
    enable_boundary_heads: bool = True,
    enable_daily_auxiliary: bool = False,
) -> TransformerTimelineHeads:
    """
    Factory function to create timeline heads with common configurations.
    
    Args:
        num_activity_classes: Number of activity classes from registry
        hidden_units: Hidden dimension for head processing
        enable_boundary_heads: Whether to enable boundary detection heads
        enable_daily_auxiliary: Whether to enable daily aggregated heads
        
    Returns:
        Configured TransformerTimelineHeads instance
    """
    config = TimelineHeadConfig(
        num_activity_classes=num_activity_classes,
        hidden_units=hidden_units,
        enable_boundary_start=enable_boundary_heads,
        enable_boundary_end=enable_boundary_heads,
        enable_daily_duration=enable_daily_auxiliary,
        enable_daily_count=enable_daily_auxiliary,
    )
    
    return TransformerTimelineHeads(config)
