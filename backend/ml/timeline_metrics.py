"""
Timeline Metrics Module

Computes timeline-quality metrics for episode evaluation:
- segment_start/end MAE
- segment_duration_mae_minutes
- episode_count_error
- fragmentation_rate

Part of WS-4: Timeline Metrics + Tiered Gates
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class TimelineMetrics:
    """Container for timeline-quality metrics."""
    
    # Per-room metrics
    segment_start_mae_minutes: float = 0.0
    segment_end_mae_minutes: float = 0.0
    segment_duration_mae_minutes: float = 0.0
    episode_count_error: float = 0.0
    fragmentation_rate: float = 0.0
    event_iou: float = 0.0
    onset_tolerance_rate: float = 0.0
    offset_tolerance_rate: float = 0.0
    
    # Episode-level details for debugging
    num_pred_episodes: int = 0
    num_gt_episodes: int = 0
    matched_episodes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'segment_start_mae_minutes': round(self.segment_start_mae_minutes, 4),
            'segment_end_mae_minutes': round(self.segment_end_mae_minutes, 4),
            'segment_duration_mae_minutes': round(self.segment_duration_mae_minutes, 4),
            'episode_count_error': round(self.episode_count_error, 4),
            'fragmentation_rate': round(self.fragmentation_rate, 4),
            'event_iou': round(self.event_iou, 4),
            'onset_tolerance_rate': round(self.onset_tolerance_rate, 4),
            'offset_tolerance_rate': round(self.offset_tolerance_rate, 4),
            'num_pred_episodes': self.num_pred_episodes,
            'num_gt_episodes': self.num_gt_episodes,
            'matched_episodes': self.matched_episodes,
        }


def compute_event_level_quality(
    pred_episodes: List[Dict[str, Any]],
    gt_episodes: List[Dict[str, Any]],
    matches: List[Tuple[int, int, float]],
    *,
    tolerance_minutes: float,
) -> Tuple[float, float, float]:
    if not matches:
        return 0.0, 0.0, 0.0

    iou_scores = []
    onset_hits = 0
    offset_hits = 0
    tolerance_seconds = float(tolerance_minutes) * 60.0
    for pred_idx, gt_idx, _ in matches:
        pred_ep = pred_episodes[pred_idx]
        gt_ep = gt_episodes[gt_idx]
        pred_start = pd.to_datetime(pred_ep['start_time'])
        pred_end = pd.to_datetime(pred_ep['end_time'])
        gt_start = pd.to_datetime(gt_ep['start_time'])
        gt_end = pd.to_datetime(gt_ep['end_time'])
        intersection = max(0.0, (min(pred_end, gt_end) - max(pred_start, gt_start)).total_seconds())
        union = max((max(pred_end, gt_end) - min(pred_start, gt_start)).total_seconds(), 1e-6)
        iou_scores.append(intersection / union)
        if abs((pred_start - gt_start).total_seconds()) <= tolerance_seconds:
            onset_hits += 1
        if abs((pred_end - gt_end).total_seconds()) <= tolerance_seconds:
            offset_hits += 1
    total = len(matches)
    return (
        float(np.mean(iou_scores)) if iou_scores else 0.0,
        float(onset_hits) / float(total),
        float(offset_hits) / float(total),
    )


def compute_fragmentation_rate(
    pred_episodes: List[Dict[str, Any]],
    gt_episodes: List[Dict[str, Any]],
    tolerance_minutes: float = 5.0,
) -> float:
    """
    Compute fragmentation rate: how many predicted episodes map to one ground truth episode.
    
    Fragmentation rate = (num_pred - num_matched) / num_gt
    where num_matched = number of GT episodes that have at least one matching pred episode.
    
    Args:
        pred_episodes: List of predicted episode dicts
        gt_episodes: List of ground truth episode dicts
        tolerance_minutes: Temporal tolerance for matching
        
    Returns:
        Fragmentation rate (0 = no fragmentation, higher = more fragmented)
    """
    if not gt_episodes:
        return 0.0
    
    if not pred_episodes:
        return 1.0  # Complete fragmentation (no predictions)
    
    # Count how many GT episodes have at least one matching pred episode
    matched_gt = 0
    for gt_ep in gt_episodes:
        gt_start = pd.to_datetime(gt_ep['start_time'])
        gt_end = pd.to_datetime(gt_ep['end_time'])
        gt_label = gt_ep.get('label')
        
        has_match = False
        for pred_ep in pred_episodes:
            pred_start = pd.to_datetime(pred_ep['start_time'])
            pred_end = pd.to_datetime(pred_ep['end_time'])
            pred_label = pred_ep.get('label')

            # Respect label semantics when both labels are available.
            if gt_label is not None and pred_label is not None and str(gt_label) != str(pred_label):
                continue
            
            # Check overlap with tolerance
            overlap_start = max(gt_start, pred_start) - pd.Timedelta(minutes=tolerance_minutes)
            overlap_end = min(gt_end, pred_end) + pd.Timedelta(minutes=tolerance_minutes)
            
            if overlap_start < overlap_end:
                has_match = True
                break
        
        if has_match:
            matched_gt += 1
    
    # Fragmentation rate
    fragmentation = (len(pred_episodes) - matched_gt) / len(gt_episodes)
    return max(0.0, fragmentation)


def match_episodes(
    pred_episodes: List[Dict[str, Any]],
    gt_episodes: List[Dict[str, Any]],
    tolerance_minutes: float = 5.0,
) -> List[Tuple[int, int, float]]:
    """
    Match predicted episodes to ground truth episodes.
    
    Uses greedy matching based on temporal overlap and label agreement.
    
    Args:
        pred_episodes: List of predicted episode dicts
        gt_episodes: List of ground truth episode dicts
        tolerance_minutes: Temporal tolerance for matching
        
    Returns:
        List of (pred_idx, gt_idx, score) tuples
    """
    matches = []
    matched_gt = set()
    
    for pred_idx, pred_ep in enumerate(pred_episodes):
        pred_start = pd.to_datetime(pred_ep['start_time'])
        pred_end = pd.to_datetime(pred_ep['end_time'])
        pred_label = pred_ep.get('label', 'unknown')
        
        best_gt_idx = -1
        best_score = -1.0
        
        for gt_idx, gt_ep in enumerate(gt_episodes):
            if gt_idx in matched_gt:
                continue
            
            gt_start = pd.to_datetime(gt_ep['start_time'])
            gt_end = pd.to_datetime(gt_ep['end_time'])
            gt_label = gt_ep.get('label', 'unknown')
            
            # Check label match
            if pred_label != gt_label:
                continue
            
            # Compute temporal overlap score
            overlap_start = max(pred_start, gt_start) - pd.Timedelta(minutes=tolerance_minutes)
            overlap_end = min(pred_end, gt_end) + pd.Timedelta(minutes=tolerance_minutes)
            
            if overlap_start >= overlap_end:
                continue
            
            # Score = overlap duration
            score = (overlap_end - overlap_start).total_seconds()
            
            if score > best_score:
                best_score = score
                best_gt_idx = gt_idx
        
        if best_gt_idx >= 0:
            matches.append((pred_idx, best_gt_idx, best_score))
            matched_gt.add(best_gt_idx)
    
    return matches


def compute_segment_boundary_mae(
    pred_episodes: List[Dict[str, Any]],
    gt_episodes: List[Dict[str, Any]],
    matches: List[Tuple[int, int, float]],
) -> Tuple[float, float]:
    """
    Compute start and end boundary MAE in minutes.
    
    Args:
        pred_episodes: List of predicted episode dicts
        gt_episodes: List of ground truth episode dicts
        matches: List of (pred_idx, gt_idx, score) matches
        
    Returns:
        Tuple of (start_mae_minutes, end_mae_minutes)
    """
    if not matches:
        return float('inf'), float('inf')
    
    start_errors = []
    end_errors = []
    
    for pred_idx, gt_idx, _ in matches:
        pred_ep = pred_episodes[pred_idx]
        gt_ep = gt_episodes[gt_idx]
        
        pred_start = pd.to_datetime(pred_ep['start_time'])
        pred_end = pd.to_datetime(pred_ep['end_time'])
        gt_start = pd.to_datetime(gt_ep['start_time'])
        gt_end = pd.to_datetime(gt_ep['end_time'])
        
        # Compute errors in minutes
        start_error = abs((pred_start - gt_start).total_seconds()) / 60.0
        end_error = abs((pred_end - gt_end).total_seconds()) / 60.0
        
        start_errors.append(start_error)
        end_errors.append(end_error)
    
    start_mae = np.mean(start_errors) if start_errors else float('inf')
    end_mae = np.mean(end_errors) if end_errors else float('inf')
    
    return start_mae, end_mae


def compute_duration_mae(
    pred_episodes: List[Dict[str, Any]],
    gt_episodes: List[Dict[str, Any]],
    matches: List[Tuple[int, int, float]],
) -> float:
    """
    Compute duration MAE in minutes.
    
    Args:
        pred_episodes: List of predicted episode dicts
        gt_episodes: List of ground truth episode dicts
        matches: List of (pred_idx, gt_idx, score) matches
        
    Returns:
        Duration MAE in minutes
    """
    if not matches:
        return float('inf')
    
    duration_errors = []
    
    for pred_idx, gt_idx, _ in matches:
        pred_ep = pred_episodes[pred_idx]
        gt_ep = gt_episodes[gt_idx]
        
        pred_duration = pred_ep.get('duration_minutes', 0.0)
        gt_duration = gt_ep.get('duration_minutes', 0.0)
        
        if pred_duration == 0.0:
            # Compute from timestamps
            pred_start = pd.to_datetime(pred_ep['start_time'])
            pred_end = pd.to_datetime(pred_ep['end_time'])
            pred_duration = (pred_end - pred_start).total_seconds() / 60.0
        
        if gt_duration == 0.0:
            gt_start = pd.to_datetime(gt_ep['start_time'])
            gt_end = pd.to_datetime(gt_ep['end_time'])
            gt_duration = (gt_end - gt_start).total_seconds() / 60.0
        
        duration_errors.append(abs(pred_duration - gt_duration))
    
    return np.mean(duration_errors) if duration_errors else float('inf')


def compute_timeline_metrics(
    pred_episodes: List[Dict[str, Any]],
    gt_episodes: List[Dict[str, Any]],
    tolerance_minutes: float = 5.0,
) -> TimelineMetrics:
    """
    Compute all timeline metrics.
    
    Args:
        pred_episodes: List of predicted episode dicts
        gt_episodes: List of ground truth episode dicts
        tolerance_minutes: Temporal tolerance for matching
        
    Returns:
        TimelineMetrics with all computed metrics
    """
    # Match episodes
    matches = match_episodes(pred_episodes, gt_episodes, tolerance_minutes)
    
    # Fragmentation rate
    fragmentation = compute_fragmentation_rate(pred_episodes, gt_episodes, tolerance_minutes)
    
    # Boundary MAE
    start_mae, end_mae = compute_segment_boundary_mae(
        pred_episodes, gt_episodes, matches
    )
    
    # Duration MAE
    duration_mae = compute_duration_mae(pred_episodes, gt_episodes, matches)
    
    # Episode count error
    count_error = abs(len(pred_episodes) - len(gt_episodes))
    event_iou, onset_tolerance_rate, offset_tolerance_rate = compute_event_level_quality(
        pred_episodes,
        gt_episodes,
        matches,
        tolerance_minutes=tolerance_minutes,
    )

    return TimelineMetrics(
        segment_start_mae_minutes=start_mae,
        segment_end_mae_minutes=end_mae,
        segment_duration_mae_minutes=duration_mae,
        episode_count_error=count_error,
        fragmentation_rate=fragmentation,
        event_iou=event_iou,
        onset_tolerance_rate=onset_tolerance_rate,
        offset_tolerance_rate=offset_tolerance_rate,
        num_pred_episodes=len(pred_episodes),
        num_gt_episodes=len(gt_episodes),
        matched_episodes=len(matches),
    )


def compute_room_timeline_metrics(
    pred_episodes_by_room: Dict[str, List[Dict[str, Any]]],
    gt_episodes_by_room: Dict[str, List[Dict[str, Any]]],
    tolerance_minutes: float = 5.0,
) -> Dict[str, TimelineMetrics]:
    """
    Compute timeline metrics per room.
    
    Args:
        pred_episodes_by_room: Dict mapping room names to predicted episodes
        gt_episodes_by_room: Dict mapping room names to ground truth episodes
        tolerance_minutes: Temporal tolerance for matching
        
    Returns:
        Dict mapping room names to TimelineMetrics
    """
    metrics = {}
    
    all_rooms = set(pred_episodes_by_room.keys()) | set(gt_episodes_by_room.keys())
    
    for room in all_rooms:
        pred_eps = pred_episodes_by_room.get(room, [])
        gt_eps = gt_episodes_by_room.get(room, [])
        
        metrics[room] = compute_timeline_metrics(pred_eps, gt_eps, tolerance_minutes)
    
    return metrics


def compute_aggregate_timeline_metrics(
    room_metrics: Dict[str, TimelineMetrics],
) -> Dict[str, float]:
    """
    Aggregate timeline metrics across rooms.
    
    Args:
        room_metrics: Dict mapping room names to TimelineMetrics
        
    Returns:
        Dict with aggregate metrics
    """
    if not room_metrics:
        return {
            'mean_start_mae': float('inf'),
            'mean_end_mae': float('inf'),
            'mean_duration_mae': float('inf'),
            'mean_fragmentation': 0.0,
            'total_episode_count_error': 0,
        }
    
    start_maes = [m.segment_start_mae_minutes for m in room_metrics.values() if m.segment_start_mae_minutes != float('inf')]
    end_maes = [m.segment_end_mae_minutes for m in room_metrics.values() if m.segment_end_mae_minutes != float('inf')]
    duration_maes = [m.segment_duration_mae_minutes for m in room_metrics.values() if m.segment_duration_mae_minutes != float('inf')]
    fragmentations = [m.fragmentation_rate for m in room_metrics.values()]
    count_errors = [m.episode_count_error for m in room_metrics.values()]
    
    return {
        'mean_start_mae': np.mean(start_maes) if start_maes else float('inf'),
        'mean_end_mae': np.mean(end_maes) if end_maes else float('inf'),
        'mean_duration_mae': np.mean(duration_maes) if duration_maes else float('inf'),
        'mean_fragmentation': np.mean(fragmentations) if fragmentations else 0.0,
        'total_episode_count_error': sum(count_errors),
        'num_rooms_evaluated': len(room_metrics),
    }
