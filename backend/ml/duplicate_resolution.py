"""
Item 8: Duplicate-Timestamp Label Aggregation Policy

Deterministic resolution of duplicate timestamps with configurable
tie-breaking strategies and audit trail.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TieBreaker(Enum):
    """Tie-breaking strategies for duplicate resolution."""
    LATEST = "latest"
    HIGHEST_PRIORITY = "highest_priority"
    FIRST = "first"


@dataclass
class DuplicateResolutionPolicy:
    """
    Configuration for duplicate timestamp resolution.
    
    Example:
        policy = DuplicateResolutionPolicy(
            method="majority_vote",
            tie_breaker=TieBreaker.LATEST,
            emit_stats=True,
        )
    """
    method: str = "majority_vote"  # majority_vote, first, random
    tie_breaker: TieBreaker = field(default=TieBreaker.LATEST)
    class_priority_map: Dict[str, int] = field(default_factory=dict)
    emit_stats: bool = True
    fail_on_unresolved: bool = False  # If True, fail on persistent ties
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "tie_breaker": self.tie_breaker.value,
            "class_priority_map": self.class_priority_map,
            "emit_stats": self.emit_stats,
            "fail_on_unresolved": self.fail_on_unresolved,
        }


@dataclass
class DuplicateResolutionStats:
    """Statistics from duplicate resolution."""
    total_timestamps: int = 0
    unique_timestamps: int = 0
    duplicate_count: int = 0
    tie_count: int = 0
    conflict_unresolved_count: int = 0
    resolution_method: str = ""
    per_timestamp_stats: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_timestamps": self.total_timestamps,
            "unique_timestamps": self.unique_timestamps,
            "duplicate_count": self.duplicate_count,
            "tie_count": self.tie_count,
            "conflict_unresolved_count": self.conflict_unresolved_count,
            "resolution_method": self.resolution_method,
            "duplicate_rate": self.duplicate_count / max(1, self.total_timestamps),
        }


class DuplicateTimestampResolver:
    """
    Resolves duplicate timestamps with deterministic aggregation.
    
    Replaces arbitrary 'first' aggregation with explicit policy.
    """
    
    def __init__(self, policy: Optional[DuplicateResolutionPolicy] = None):
        self.policy = policy or DuplicateResolutionPolicy()
        self.stats = DuplicateResolutionStats()
    
    def resolve(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        label_col: str = 'activity',
    ) -> pd.DataFrame:
        """
        Resolve duplicate timestamps in DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        timestamp_col : str
            Name of timestamp column
        label_col : str
            Name of label column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with resolved duplicates
            
        Raises:
        -------
        ValueError
            If unresolved conflicts exist and fail_on_unresolved=True
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found")
        
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found")
        
        # Initialize stats
        self.stats = DuplicateResolutionStats()
        self.stats.total_timestamps = len(df)
        self.stats.resolution_method = self.policy.method
        
        # PERFORMANCE OPTIMIZATION: Identify duplicates using vectorized operation
        # instead of grouping entire dataframe
        dupe_mask = df.duplicated(subset=[timestamp_col], keep=False)
        
        if not dupe_mask.any():
            # No duplicates - return as-is (fast path)
            self.stats.unique_timestamps = len(df)
            logger.debug("No duplicate timestamps found")
            return df.copy()
        
        # Separate non-duplicates (keep as-is) from duplicates (need resolution)
        non_dupes = df[~dupe_mask].copy()
        dupes = df[dupe_mask].copy()
        
        self.stats.duplicate_count = len(dupes)
        n_unique_dupes = dupes[timestamp_col].nunique()
        
        logger.info(
            f"Found {n_unique_dupes} timestamps with duplicates, "
            f"totaling {len(dupes)} duplicate rows out of {len(df)} total"
        )
        
        # Resolve duplicates - ONLY iterate over duplicate timestamps
        resolved_rows = []
        unresolved_conflicts = []
        
        for ts, group in dupes.groupby(timestamp_col, sort=False):
            # All groups here have duplicates by definition
            resolved_row, was_tie, was_unresolved = self._resolve_duplicate_group(
                group, timestamp_col, label_col
            )
            resolved_rows.append(resolved_row)
            
            if was_tie:
                self.stats.tie_count += 1
            if was_unresolved:
                self.stats.conflict_unresolved_count += 1
                unresolved_conflicts.append(ts)
        
        # Combine non-duplicates with resolved duplicates
        if resolved_rows:
            resolved_df = pd.DataFrame(resolved_rows)
            if not non_dupes.empty:
                resolved_df = pd.concat([non_dupes, resolved_df], ignore_index=True)
        else:
            resolved_df = non_dupes.copy()
        
        self.stats.unique_timestamps = len(non_dupes) + len(resolved_rows)
        
        # Check for unresolved conflicts
        if unresolved_conflicts and self.policy.fail_on_unresolved:
            raise ValueError(
                f"Unresolved duplicate conflicts at timestamps: {unresolved_conflicts[:10]}"
            )
        
        # Sort by timestamp to maintain chronological order
        # Convert to datetime first to handle mixed types (str/Timestamp)
        try:
            resolved_df = resolved_df.sort_values(timestamp_col).reset_index(drop=True)
        except TypeError:
            # Mixed types - convert to datetime then sort
            resolved_df[timestamp_col] = pd.to_datetime(resolved_df[timestamp_col])
            resolved_df = resolved_df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Log stats
        if self.policy.emit_stats:
            logger.info(f"Duplicate resolution stats: {self.stats.to_dict()}")
        
        return resolved_df
    
    def _resolve_duplicate_group(
        self,
        group: pd.DataFrame,
        timestamp_col: str,
        label_col: str,
    ) -> tuple:
        """
        Resolve a single group of duplicate timestamps.
        
        Returns:
        --------
        (resolved_row, was_tie, was_unresolved) : tuple
            resolved_row: The chosen row
            was_tie: True if tie-breaking was needed
            was_unresolved: True if conflict couldn't be resolved
        """
        labels = group[label_col].tolist()
        unique_labels = list(dict.fromkeys(labels))  # Preserve order
        
        if len(unique_labels) == 1:
            # All labels are the same - no conflict
            return group.iloc[0], False, False
        
        # Multiple different labels - apply resolution method
        was_tie = True
        
        if self.policy.method == "majority_vote":
            resolved_label = self._majority_vote(labels)
        elif self.policy.method == "first":
            resolved_label = labels[0]
        elif self.policy.method == "random":
            resolved_label = np.random.choice(labels)
        else:
            resolved_label = self._majority_vote(labels)
        
        # Check if still tied after method
        if resolved_label is None:
            # Apply tie-breaker
            resolved_label = self._apply_tie_breaker(group, labels, timestamp_col, label_col)
        
        if resolved_label is None:
            # Still unresolved
            was_unresolved = True
            # Fall back to first
            resolved_label = labels[0]
        else:
            was_unresolved = False
        
        # Find row with resolved label
        matching_rows = group[group[label_col] == resolved_label]
        if len(matching_rows) > 0:
            # If multiple rows have the same label, apply tie-breaker
            if len(matching_rows) > 1:
                resolved_row = self._select_by_tie_breaker(matching_rows, timestamp_col)
            else:
                resolved_row = matching_rows.iloc[0]
        else:
            # Shouldn't happen, but fall back to first row
            resolved_row = group.iloc[0]
        
        return resolved_row, was_tie, was_unresolved
    
    def _majority_vote(self, labels: List[str]) -> Optional[str]:
        """
        Apply majority vote to labels.
        
        Returns:
        --------
        str or None
            Winning label, or None if tie
        """
        from collections import Counter
        counts = Counter(labels)
        max_count = max(counts.values())
        winners = [label for label, count in counts.items() if count == max_count]
        
        if len(winners) == 1:
            return winners[0]
        else:
            return None  # Tie
    
    def _apply_tie_breaker(
        self,
        group: pd.DataFrame,
        labels: List[str],
        timestamp_col: str,
        label_col: str,
    ) -> Optional[str]:
        """Apply configured tie-breaker."""
        if self.policy.tie_breaker == TieBreaker.LATEST:
            # Use label from latest row (by index position)
            return labels[-1]
        
        elif self.policy.tie_breaker == TieBreaker.HIGHEST_PRIORITY:
            # Use class priority map
            if not self.policy.class_priority_map:
                return None
            
            # Find highest priority label
            best_label = None
            best_priority = float('inf')
            
            for label in labels:
                priority = self.policy.class_priority_map.get(label, float('inf'))
                if priority < best_priority:
                    best_priority = priority
                    best_label = label
            
            return best_label
        
        elif self.policy.tie_breaker == TieBreaker.FIRST:
            return labels[0]
        
        return None
    
    def _select_by_tie_breaker(
        self,
        rows: pd.DataFrame,
        timestamp_col: str,
    ) -> pd.Series:
        """Select one row from multiple candidates using tie-breaker."""
        if self.policy.tie_breaker == TieBreaker.LATEST:
            # Return last row (assuming chronological order)
            return rows.iloc[-1]
        else:
            # Default to first
            return rows.iloc[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resolution statistics."""
        return self.stats.to_dict()


def resolve_duplicate_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    label_col: str = 'activity',
    method: str = "majority_vote",
    tie_breaker: str = "latest",
    emit_stats: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function for duplicate timestamp resolution.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    timestamp_col : str
        Timestamp column name
    label_col : str
        Label column name
    method : str
        Resolution method (majority_vote, first, random)
    tie_breaker : str
        Tie-breaking strategy (latest, highest_priority, first)
    emit_stats : bool
        Whether to emit statistics
        
    Returns:
    --------
    (resolved_df, stats) : Tuple
        resolved_df: DataFrame with resolved duplicates
        stats: Resolution statistics
    """
    policy = DuplicateResolutionPolicy(
        method=method,
        tie_breaker=TieBreaker(tie_breaker),
        emit_stats=emit_stats,
    )
    
    resolver = DuplicateTimestampResolver(policy)
    resolved_df = resolver.resolve(df, timestamp_col, label_col)
    
    return resolved_df, resolver.get_stats()


def add_duplicate_resolution_to_policy(policy):
    """
    Add duplicate resolution configuration to TrainingPolicy.
    
    This is a placeholder for integration with policy_config.py.
    """
    # This function should be called during policy initialization
    # to set default duplicate resolution settings
    if not hasattr(policy, 'duplicate_resolution'):
        policy.duplicate_resolution = DuplicateResolutionPolicy()
    return policy
