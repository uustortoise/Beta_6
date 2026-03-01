"""
Coverage Contract Gate - Pre-Train Hard Gate

Ensures sufficient data coverage for walk-forward validation before training starts.
Prevents wasted compute on datasets that cannot satisfy evaluation requirements.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class CoverageContractResult:
    """Result of coverage contract evaluation."""
    room_name: str
    passes: bool
    observed_days: int
    required_days: int
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    min_train_days: int = 7
    valid_days: int = 1
    step_days: int = 1
    min_folds: int = 1
    
    @property
    def min_required_days(self) -> int:
        """Minimum days required to form at least one fold."""
        return self.min_train_days + self.valid_days


class CoverageContractGate:
    """
    Pre-train gate that validates data coverage before model training.
    
    Fails fast if the dataset cannot support the configured walk-forward validation,
    preventing wasted compute on insufficient data.
    """
    
    def __init__(self, walk_forward_config: Optional[WalkForwardConfig] = None):
        self.wf_config = walk_forward_config or WalkForwardConfig()
    
    def evaluate(self, room_name: str, observed_days: int) -> CoverageContractResult:
        """
        Evaluate coverage contract for a room.
        
        Parameters:
        -----------
        room_name : str
            Name of the room being evaluated
        observed_days : int
            Number of unique days with observed data
        
        Returns:
        --------
        CoverageContractResult
            Pass/fail result with detailed reasoning
        """
        required_days = self.wf_config.min_required_days
        
        if observed_days >= required_days:
            return CoverageContractResult(
                room_name=room_name,
                passes=True,
                observed_days=observed_days,
                required_days=required_days,
                details={
                    'min_train_days': self.wf_config.min_train_days,
                    'valid_days': self.wf_config.valid_days,
                    'step_days': self.wf_config.step_days,
                    'estimated_max_folds': max(0, observed_days - required_days + 1)
                }
            )
        else:
            reason = (
                f"Insufficient observed days for walk-forward validation: "
                f"have {observed_days}, need {required_days} "
                f"({self.wf_config.min_train_days} train + {self.wf_config.valid_days} validation)"
            )
            return CoverageContractResult(
                room_name=room_name,
                passes=False,
                observed_days=observed_days,
                required_days=required_days,
                reason=reason,
                details={
                    'shortfall_days': required_days - observed_days,
                    'min_train_days': self.wf_config.min_train_days,
                    'valid_days': self.wf_config.valid_days
                }
            )
    
    def evaluate_all_rooms(self, room_days: Dict[str, int]) -> Dict[str, CoverageContractResult]:
        """
        Evaluate coverage contract for all rooms.
        
        Parameters:
        -----------
        room_days : Dict[str, int]
            Mapping of room_name -> observed_days
        
        Returns:
        --------
        Dict[str, CoverageContractResult]
            Results for each room
        """
        return {
            room: self.evaluate(room, days)
            for room, days in room_days.items()
        }
    
    def generate_report(self, results: Dict[str, CoverageContractResult]) -> Dict[str, Any]:
        """
        Generate coverage contract report for persistence.
        
        Parameters:
        -----------
        results : Dict[str, CoverageContractResult]
            Results from evaluate_all_rooms()
        
        Returns:
        --------
        Dict[str, Any]
            Structured report suitable for JSON serialization
        """
        passing_rooms = [r for r in results.values() if r.passes]
        failing_rooms = [r for r in results.values() if not r.passes]
        
        report = {
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'walk_forward_config': {
                'min_train_days': self.wf_config.min_train_days,
                'valid_days': self.wf_config.valid_days,
                'step_days': self.wf_config.step_days,
                'min_required_days': self.wf_config.min_required_days
            },
            'summary': {
                'total_rooms': len(results),
                'passing': len(passing_rooms),
                'failing': len(failing_rooms)
            },
            'rooms': {}
        }
        
        for room_name, result in results.items():
            report['rooms'][room_name] = {
                'passes': result.passes,
                'observed_days': result.observed_days,
                'required_days': result.required_days,
                'reason': result.reason,
                'details': result.details
            }
        
        return report
    
    def persist_report(self, results: Dict[str, CoverageContractResult], output_path: str):
        """
        Persist coverage contract report to JSON file.
        
        Parameters:
        -----------
        results : Dict[str, CoverageContractResult]
            Results to persist
        output_path : str
            Path to write JSON file
        """
        report = self.generate_report(results)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Coverage contract report written to {output_path}")


def check_coverage_contract(room_name: str, observed_days: int, 
                            min_train_days: int = 7, valid_days: int = 1) -> bool:
    """
    Convenience function for quick coverage check.
    
    Parameters:
    -----------
    room_name : str
        Name of the room
    observed_days : int
        Number of days with observed data
    min_train_days : int, default=7
        Minimum days required for training portion
    valid_days : int, default=1
        Minimum days required for validation portion
    
    Returns:
    --------
    bool
        True if coverage contract passes, False otherwise
    """
    config = WalkForwardConfig(min_train_days=min_train_days, valid_days=valid_days)
    gate = CoverageContractGate(config)
    result = gate.evaluate(room_name, observed_days)
    
    if not result.passes:
        logger.warning(f"Coverage contract failed for {room_name}: {result.reason}")
    
    return result.passes
