"""
Item 9: Room Threshold Calibration Review (Kitchen/LivingRoom)

Provides per-room confusion/error analysis and calibration diagnostics
to address repeated low F1 rooms under realistic data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

logger = logging.getLogger(__name__)


class RoomType(Enum):
    """Room types with known calibration challenges."""
    KITCHEN = "kitchen"
    LIVING_ROOM = "livingroom"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    UNKNOWN = "unknown"


@dataclass
class ConfusionAnalysis:
    """Detailed confusion matrix analysis."""
    true_positives: Dict[str, int] = field(default_factory=dict)
    false_positives: Dict[str, int] = field(default_factory=dict)
    false_negatives: Dict[str, int] = field(default_factory=dict)
    true_negatives: Dict[str, int] = field(default_factory=dict)
    
    # Per-class metrics
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    f1_score: Dict[str, float] = field(default_factory=dict)
    support: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "support": self.support,
        }


@dataclass
class ErrorPattern:
    """Identified error pattern in model predictions."""
    pattern_type: str  # 'confusion', 'missed_detection', 'false_alarm', 'class_imbalance'
    description: str
    affected_classes: List[str]
    severity: str  # 'critical', 'high', 'medium', 'low'
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "affected_classes": self.affected_classes,
            "severity": self.severity,
            "recommendation": self.recommendation,
        }


@dataclass
class RoomCalibrationDiagnostics:
    """
    Comprehensive calibration diagnostics for a specific room.
    
    This artifact provides detailed analysis for rooms with repeated
    low F1 scores to guide threshold and feature adjustments.
    """
    room_name: str
    room_type: RoomType
    timestamp: str
    
    # Model performance
    macro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    accuracy: float = 0.0
    
    # Per-class performance
    class_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Confusion analysis
    confusion_analysis: ConfusionAnalysis = field(default_factory=ConfusionAnalysis)
    
    # Error patterns
    error_patterns: List[ErrorPattern] = field(default_factory=list)
    
    # Threshold analysis
    current_thresholds: Dict[str, float] = field(default_factory=dict)
    recommended_thresholds: Dict[str, float] = field(default_factory=dict)
    threshold_sensitivity: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    
    # Feature analysis
    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_windows: Dict[str, int] = field(default_factory=dict)
    
    # Historical context
    previous_run_ids: List[str] = field(default_factory=list)
    previous_f1_scores: List[float] = field(default_factory=list)
    trend: str = "unknown"  # improving, declining, stable
    
    # Recommendations
    primary_recommendation: str = ""
    secondary_recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_name": self.room_name,
            "room_type": self.room_type.value,
            "timestamp": self.timestamp,
            "macro_f1": self.macro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "accuracy": self.accuracy,
            "class_metrics": self.class_metrics,
            "confusion_analysis": self.confusion_analysis.to_dict(),
            "error_patterns": [p.to_dict() for p in self.error_patterns],
            "current_thresholds": self.current_thresholds,
            "recommended_thresholds": self.recommended_thresholds,
            "feature_importance": self.feature_importance,
            "feature_windows": self.feature_windows,
            "previous_run_ids": self.previous_run_ids,
            "previous_f1_scores": self.previous_f1_scores,
            "trend": self.trend,
            "primary_recommendation": self.primary_recommendation,
            "secondary_recommendations": self.secondary_recommendations,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save diagnostics to file."""
        filepath = Path(filepath)
        filepath.write_text(self.to_json())
        logger.info(f"Room calibration diagnostics saved to {filepath}")


class RoomCalibrationAnalyzer:
    """
    Analyzer for room-specific calibration issues.
    
    Focuses on Kitchen and LivingRoom which historically have
    lower F1 scores due to activity complexity.
    """
    
    # Known challenging room types
    CHALLENGING_ROOMS = [RoomType.KITCHEN, RoomType.LIVING_ROOM]
    
    def __init__(self, history_dir: Optional[Path] = None):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        history_dir : Path, optional
            Directory containing historical run data
        """
        self.history_dir = history_dir
        self._history: Dict[str, List[Dict]] = {}
        
        if history_dir:
            self._load_history()
    
    def analyze_room(
        self,
        room_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        current_thresholds: Optional[Dict[str, float]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        run_id: Optional[str] = None,
    ) -> RoomCalibrationDiagnostics:
        """
        Perform comprehensive room calibration analysis.
        
        Parameters:
        -----------
        room_name : str
            Name of the room
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels
        class_names : List[str]
            List of class names
        current_thresholds : Dict, optional
            Current classification thresholds
        feature_importance : Dict, optional
            Feature importance scores
        run_id : str, optional
            Current run ID for history tracking
            
        Returns:
        --------
        RoomCalibrationDiagnostics
        """
        room_type = self._classify_room_type(room_name)
        
        diagnostics = RoomCalibrationDiagnostics(
            room_name=room_name,
            room_type=room_type,
            timestamp=datetime.utcnow().isoformat(),
            current_thresholds=current_thresholds or {},
            feature_importance=feature_importance or {},
        )
        
        # Compute overall metrics
        report = classification_report(
            y_true, y_pred, target_names=class_names,
            output_dict=True, zero_division=0
        )
        
        diagnostics.macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
        diagnostics.macro_precision = report.get("macro avg", {}).get("precision", 0.0)
        diagnostics.macro_recall = report.get("macro avg", {}).get("recall", 0.0)
        diagnostics.accuracy = report.get("accuracy", 0.0)
        
        # Per-class metrics
        for class_name in class_names:
            if class_name in report:
                diagnostics.class_metrics[class_name] = {
                    "precision": report[class_name].get("precision", 0.0),
                    "recall": report[class_name].get("recall", 0.0),
                    "f1_score": report[class_name].get("f1-score", 0.0),
                    "support": report[class_name].get("support", 0),
                }
        
        # Confusion analysis
        diagnostics.confusion_analysis = self._analyze_confusion(
            y_true, y_pred, class_names
        )
        
        # Identify error patterns
        diagnostics.error_patterns = self._identify_error_patterns(
            diagnostics, class_names
        )
        
        # Load historical context
        if room_name in self._history:
            history = self._history[room_name][-3:]  # Last 3 runs
            diagnostics.previous_run_ids = [h.get("run_id") for h in history]
            diagnostics.previous_f1_scores = [h.get("macro_f1", 0.0) for h in history]
            diagnostics.trend = self._compute_trend(diagnostics.previous_f1_scores)
        
        # Generate recommendations
        diagnostics.primary_recommendation, \
        diagnostics.secondary_recommendations = self._generate_recommendations(
            diagnostics, room_type
        )
        
        # Store in history
        if run_id:
            self._store_history(room_name, {
                "run_id": run_id,
                "timestamp": diagnostics.timestamp,
                "macro_f1": diagnostics.macro_f1,
            })
        
        return diagnostics
    
    def _classify_room_type(self, room_name: str) -> RoomType:
        """Classify room name to room type."""
        name_lower = room_name.lower().replace(" ", "").replace("_", "")
        
        if "kitchen" in name_lower:
            return RoomType.KITCHEN
        elif "living" in name_lower or "lounge" in name_lower:
            return RoomType.LIVING_ROOM
        elif "bed" in name_lower:
            return RoomType.BEDROOM
        elif "bath" in name_lower:
            return RoomType.BATHROOM
        else:
            return RoomType.UNKNOWN
    
    def _analyze_confusion(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
    ) -> ConfusionAnalysis:
        """Analyze confusion matrix in detail."""
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        
        analysis = ConfusionAnalysis()
        
        for i, class_name in enumerate(class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            analysis.true_positives[class_name] = int(tp)
            analysis.false_positives[class_name] = int(fp)
            analysis.false_negatives[class_name] = int(fn)
            analysis.true_negatives[class_name] = int(tn)
            
            # Compute metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            analysis.precision[class_name] = float(precision)
            analysis.recall[class_name] = float(recall)
            analysis.f1_score[class_name] = float(f1)
            analysis.support[class_name] = int(cm[i, :].sum())
        
        return analysis
    
    def _identify_error_patterns(
        self,
        diagnostics: RoomCalibrationDiagnostics,
        class_names: List[str],
    ) -> List[ErrorPattern]:
        """Identify specific error patterns."""
        patterns = []
        
        # Check for class confusion patterns
        for class_name in class_names:
            precision = diagnostics.confusion_analysis.precision.get(class_name, 1.0)
            recall = diagnostics.confusion_analysis.recall.get(class_name, 1.0)
            support = diagnostics.confusion_analysis.support.get(class_name, 0)
            
            # Low recall = missed detections
            if recall < 0.5 and support > 10:
                patterns.append(ErrorPattern(
                    pattern_type="missed_detection",
                    description=f"{class_name}: Low recall ({recall:.2f}) indicates frequent missed detections",
                    affected_classes=[class_name],
                    severity="high" if recall < 0.3 else "medium",
                    recommendation=f"Consider lowering threshold for {class_name} or increasing training examples",
                ))
            
            # Low precision = false alarms
            if precision < 0.5 and support > 10:
                patterns.append(ErrorPattern(
                    pattern_type="false_alarm",
                    description=f"{class_name}: Low precision ({precision:.2f}) indicates frequent false alarms",
                    affected_classes=[class_name],
                    severity="high" if precision < 0.3 else "medium",
                    recommendation=f"Consider raising threshold for {class_name} or improving feature discrimination",
                ))
            
            # Class imbalance
            total_support = sum(diagnostics.confusion_analysis.support.values())
            if total_support > 0:
                class_ratio = support / total_support
                if class_ratio < 0.05:
                    patterns.append(ErrorPattern(
                        pattern_type="class_imbalance",
                        description=f"{class_name}: Severe under-representation ({class_ratio:.1%} of data)",
                        affected_classes=[class_name],
                        severity="high",
                        recommendation=f"Collect more {class_name} examples or consider minority sampling",
                    ))
        
        return patterns
    
    def _compute_trend(self, f1_scores: List[float]) -> str:
        """Compute trend from historical F1 scores."""
        if len(f1_scores) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(f1_scores))
        slope = np.polyfit(x, f1_scores, 1)[0]
        
        if slope > 0.02:
            return "improving"
        elif slope < -0.02:
            return "declining"
        else:
            return "stable"
    
    def _generate_recommendations(
        self,
        diagnostics: RoomCalibrationDiagnostics,
        room_type: RoomType,
    ) -> Tuple[str, List[str]]:
        """Generate calibration recommendations."""
        primary = ""
        secondary = []
        
        # Room-specific recommendations
        if room_type in self.CHALLENGING_ROOMS:
            if diagnostics.macro_f1 < 0.55:
                primary = (
                    f"{room_type.value.title()} shows critically low F1 ({diagnostics.macro_f1:.2f}). "
                    "Recommend: (1) Increase sequence window to capture longer activity patterns, "
                    "(2) Enable minority class sampling for rare activities, "
                    "(3) Review and potentially lower per-class thresholds."
                )
            else:
                primary = (
                    f"{room_type.value.title()} performance is marginal. "
                    "Consider reviewing class thresholds and feature windows."
                )
        
        # Add pattern-based recommendations
        for pattern in diagnostics.error_patterns:
            if pattern.severity == "critical":
                secondary.append(f"[CRITICAL] {pattern.recommendation}")
            elif pattern.severity == "high":
                secondary.append(f"[HIGH] {pattern.recommendation}")
        
        # Trend-based recommendations
        if diagnostics.trend == "declining":
            secondary.append("[TREND] Performance declining over last 3 runs - investigate data drift")
        
        return primary, secondary
    
    def _load_history(self) -> None:
        """Load historical diagnostics data."""
        if not self.history_dir or not self.history_dir.exists():
            return
        
        for filepath in self.history_dir.glob("*_diagnostics.json"):
            try:
                data = json.loads(filepath.read_text())
                room_name = data.get("room_name")
                if room_name:
                    if room_name not in self._history:
                        self._history[room_name] = []
                    self._history[room_name].append({
                        "run_id": data.get("timestamp", ""),  # Use timestamp as ID
                        "timestamp": data.get("timestamp", ""),
                        "macro_f1": data.get("macro_f1", 0.0),
                    })
            except Exception as e:
                logger.warning(f"Failed to load history from {filepath}: {e}")
    
    def _store_history(self, room_name: str, data: Dict) -> None:
        """Store historical data point."""
        if room_name not in self._history:
            self._history[room_name] = []
        self._history[room_name].append(data)


def should_generate_diagnostics(room_name: str, macro_f1: float, threshold: float = 0.55) -> bool:
    """
    Determine if a room should have detailed diagnostics generated.
    
    Parameters:
    -----------
    room_name : str
        Room name
    macro_f1 : float
        Current macro F1 score
    threshold : float
        F1 threshold below which diagnostics are generated
        
    Returns:
    --------
    bool
        True if diagnostics should be generated
    """
    room_lower = room_name.lower()
    
    # Always generate for known challenging rooms if below threshold
    is_challenging = any(r in room_lower for r in ["kitchen", "living"])
    
    if is_challenging and macro_f1 < threshold:
        return True
    
    # Generate for any room with critically low F1
    if macro_f1 < 0.50:
        return True
    
    return False
