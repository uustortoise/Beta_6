"""
PR-A2: CI ADL Registry Validator

Validates ADL event registry changes in CI/CD to prevent breaking taxonomy edits.

Checks:
- Breaking removals (events removed without migration)
- Breaking renames (event IDs changed without migration)
- Alias removals (breaking downstream consumers)
- Breaking criticality changes
- Allow additive changes (new events, new aliases)

Usage:
    python -m ml.ci_adl_registry_validator \
        --baseline registry.v1.yaml \
        --current registry.v2.yaml

Exit codes:
    0 - Changes are backward compatible
    1 - Breaking changes detected
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

from ml.adl_registry import ADLEventRegistry, CriticalityLevel

logger = logging.getLogger(__name__)

MIGRATION_DIR = Path(__file__).parent.parent / "config" / "adl_event_migrations"


class ChangeType(Enum):
    """Types of registry changes."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


class ChangeSeverity(Enum):
    """Severity of registry changes."""
    BREAKING = "breaking"
    WARNING = "warning"
    SAFE = "safe"


@dataclass
class RegistryChange:
    """Single registry change record."""
    change_type: ChangeType
    severity: ChangeSeverity
    entity_type: str  # event, alias, kpi_group, room_scope
    entity_id: str
    details: str
    migration_available: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_type": self.change_type.value,
            "severity": self.severity.value,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "details": self.details,
            "migration_available": self.migration_available,
        }


@dataclass
class ValidationResult:
    """Result of registry validation."""
    passed: bool
    changes: List[RegistryChange]
    baseline_version: str
    current_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "baseline_version": self.baseline_version,
            "current_version": self.current_version,
            "change_count": len(self.changes),
            "breaking_count": len([c for c in self.changes if c.severity == ChangeSeverity.BREAKING]),
            "warning_count": len([c for c in self.changes if c.severity == ChangeSeverity.WARNING]),
            "changes": [c.to_dict() for c in self.changes],
        }
    
    def format_report(self) -> str:
        """Format validation result as human-readable report."""
        lines = [
            "=" * 70,
            "ADL REGISTRY CI VALIDATION REPORT",
            "=" * 70,
            f"Baseline: {self.baseline_version}",
            f"Current:  {self.current_version}",
            f"Status:   {'✅ PASSED' if self.passed else '❌ FAILED'}",
            "",
            f"Changes: {len(self.changes)} total",
        ]
        
        breaking = [c for c in self.changes if c.severity == ChangeSeverity.BREAKING]
        warnings = [c for c in self.changes if c.severity == ChangeSeverity.WARNING]
        safe = [c for c in self.changes if c.severity == ChangeSeverity.SAFE]
        
        if breaking:
            lines.append(f"\n❌ BREAKING CHANGES ({len(breaking)}):")
            for change in breaking:
                mig = " [migration available]" if change.migration_available else ""
                lines.append(f"  - [{change.change_type.value.upper()}] {change.entity_type}:{change.entity_id}{mig}")
                lines.append(f"    {change.details}")
        
        if warnings:
            lines.append(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for change in warnings:
                lines.append(f"  - [{change.change_type.value.upper()}] {change.entity_type}:{change.entity_id}")
                lines.append(f"    {change.details}")
        
        if safe:
            lines.append(f"\n✅ SAFE CHANGES ({len(safe)}):")
            for change in safe:
                lines.append(f"  - [{change.change_type.value.upper()}] {change.entity_type}:{change.entity_id}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


class ADLRegistryMigrationLoader:
    """Loads and queries migration files."""
    
    def __init__(self, migrations_dir: Optional[Path] = None):
        self.migrations_dir = migrations_dir or MIGRATION_DIR
        self._migrations: Dict[str, Dict[str, Any]] = {}
    
    def load_migrations(self, from_version: str, to_version: str) -> Dict[str, Any]:
        """
        Load migrations between two versions.
        
        Returns:
            Dict mapping entity IDs to migration info
        """
        migrations = {}
        
        # Look for migration files
        pattern = f"{from_version}_to_{to_version}.yaml"
        migration_file = self.migrations_dir / pattern
        
        if migration_file.exists():
            data = yaml.safe_load(migration_file.read_text())
            for mig in data.get("migrations", []):
                entity_id = mig.get("from")
                migrations[entity_id] = mig
        
        return migrations
    
    def has_migration(self, from_version: str, to_version: str, entity_id: str) -> bool:
        """Check if a migration exists for an entity."""
        migrations = self.load_migrations(from_version, to_version)
        return entity_id in migrations


class ADLRegistryCIValidator:
    """
    CI validator for ADL registry changes.
    
    Detects breaking changes and ensures backward compatibility.
    """
    
    def __init__(
        self,
        migrations_dir: Optional[Path] = None,
        strict: bool = True,
    ):
        """
        Initialize validator.
        
        Args:
            migrations_dir: Directory containing migration files
            strict: If True, treat warnings as breaking
        """
        self.migration_loader = ADLRegistryMigrationLoader(migrations_dir)
        self.strict = strict
        self.changes: List[RegistryChange] = []
    
    def validate(
        self,
        baseline_path: Path,
        current_path: Path,
    ) -> ValidationResult:
        """
        Validate registry changes between baseline and current.
        
        Args:
            baseline_path: Path to baseline registry YAML
            current_path: Path to current registry YAML
            
        Returns:
            ValidationResult with pass/fail status
        """
        self.changes = []
        
        # Load registries
        baseline = ADLEventRegistry(baseline_path, skip_validation=False)
        current = ADLEventRegistry(current_path, skip_validation=False)
        
        # Load migrations
        migrations = self.migration_loader.load_migrations(
            baseline.version, current.version
        )
        
        # Compare events
        self._compare_events(baseline, current, migrations)
        
        # Compare room scopes
        self._compare_room_scopes(baseline, current)
        
        # Compare KPI groups
        self._compare_kpi_groups(baseline, current)
        
        # Determine pass/fail
        breaking = [c for c in self.changes if c.severity == ChangeSeverity.BREAKING]
        passed = len(breaking) == 0
        
        if self.strict:
            warnings = [c for c in self.changes if c.severity == ChangeSeverity.WARNING]
            if warnings:
                passed = False
        
        return ValidationResult(
            passed=passed,
            changes=self.changes,
            baseline_version=baseline.version,
            current_version=current.version,
        )
    
    def _compare_events(
        self,
        baseline: ADLEventRegistry,
        current: ADLEventRegistry,
        migrations: Dict[str, Any],
    ) -> None:
        """Compare event definitions."""
        baseline_events = set(baseline.list_all_events())
        current_events = set(current.list_all_events())
        
        # Find added events (safe)
        added = current_events - baseline_events
        for event_id in added:
            self.changes.append(RegistryChange(
                change_type=ChangeType.ADDED,
                severity=ChangeSeverity.SAFE,
                entity_type="event",
                entity_id=event_id,
                details=f"New event added",
            ))
        
        # Find removed events (breaking unless migrated)
        removed = baseline_events - current_events
        for event_id in removed:
            has_migration = event_id in migrations
            self.changes.append(RegistryChange(
                change_type=ChangeType.REMOVED,
                severity=ChangeSeverity.BREAKING if not has_migration else ChangeSeverity.WARNING,
                entity_type="event",
                entity_id=event_id,
                details=f"Event removed{' without migration' if not has_migration else ' with migration'}",
                migration_available=has_migration,
            ))
        
        # Compare common events
        common = baseline_events & current_events
        for event_id in common:
            baseline_event = baseline.get_event(event_id)
            current_event = current.get_event(event_id)
            
            # Check criticality changes
            if baseline_event.criticality != current_event.criticality:
                # Define criticality order (higher = more strict)
                criticality_order = {
                    "low": 1,
                    "medium": 2,
                    "high": 3,
                    "critical": 4,
                }
                baseline_level = criticality_order.get(baseline_event.criticality.value, 0)
                current_level = criticality_order.get(current_event.criticality.value, 0)
                
                # Upgrading criticality is breaking (more strict)
                # Downgrading is warning
                if current_level > baseline_level:
                    severity = ChangeSeverity.BREAKING
                else:
                    severity = ChangeSeverity.WARNING
                
                self.changes.append(RegistryChange(
                    change_type=ChangeType.MODIFIED,
                    severity=severity,
                    entity_type="event",
                    entity_id=event_id,
                    details=f"Criticality changed from {baseline_event.criticality.value} to {current_event.criticality.value}",
                ))
            
            # Removing KPI groups from an event is a semantic contract break.
            baseline_kpi_groups = set(baseline_event.kpi_groups)
            current_kpi_groups = set(current_event.kpi_groups)
            removed_kpi_groups = baseline_kpi_groups - current_kpi_groups
            if removed_kpi_groups:
                self.changes.append(RegistryChange(
                    change_type=ChangeType.MODIFIED,
                    severity=ChangeSeverity.BREAKING,
                    entity_type="event",
                    entity_id=event_id,
                    details=f"Removed KPI groups: {sorted(removed_kpi_groups)}",
                ))
            added_kpi_groups = current_kpi_groups - baseline_kpi_groups
            if added_kpi_groups:
                self.changes.append(RegistryChange(
                    change_type=ChangeType.MODIFIED,
                    severity=ChangeSeverity.SAFE,
                    entity_type="event",
                    entity_id=event_id,
                    details=f"Added KPI groups: {sorted(added_kpi_groups)}",
                ))
            
            # Check room scope changes (removing rooms is breaking)
            baseline_rooms = set(baseline_event.room_scope)
            current_rooms = set(current_event.room_scope)
            removed_rooms = baseline_rooms - current_rooms
            if removed_rooms:
                self.changes.append(RegistryChange(
                    change_type=ChangeType.MODIFIED,
                    severity=ChangeSeverity.BREAKING,
                    entity_type="event",
                    entity_id=event_id,
                    details=f"Removed from rooms: {sorted(removed_rooms)}",
                ))
            
            added_rooms = current_rooms - baseline_rooms
            if added_rooms:
                self.changes.append(RegistryChange(
                    change_type=ChangeType.MODIFIED,
                    severity=ChangeSeverity.SAFE,
                    entity_type="event",
                    entity_id=event_id,
                    details=f"Added to rooms: {sorted(added_rooms)}",
                ))
            
            # Check alias changes
            baseline_aliases = set(baseline_event.aliases)
            current_aliases = set(current_event.aliases)
            
            removed_aliases = baseline_aliases - current_aliases
            for alias in removed_aliases:
                self.changes.append(RegistryChange(
                    change_type=ChangeType.REMOVED,
                    severity=ChangeSeverity.BREAKING,
                    entity_type="alias",
                    entity_id=f"{event_id}:{alias}",
                    details=f"Alias removed from event '{event_id}'",
                ))
            
            added_aliases = current_aliases - baseline_aliases
            for alias in added_aliases:
                self.changes.append(RegistryChange(
                    change_type=ChangeType.ADDED,
                    severity=ChangeSeverity.SAFE,
                    entity_type="alias",
                    entity_id=f"{event_id}:{alias}",
                    details=f"New alias added to event '{event_id}'",
                ))
            
            # Check disabled events
            if baseline_event.enabled and not current_event.enabled:
                self.changes.append(RegistryChange(
                    change_type=ChangeType.MODIFIED,
                    severity=ChangeSeverity.BREAKING,
                    entity_type="event",
                    entity_id=event_id,
                    details=f"Event disabled",
                ))
    
    def _compare_room_scopes(
        self,
        baseline: ADLEventRegistry,
        current: ADLEventRegistry,
    ) -> None:
        """Compare room scope definitions."""
        baseline_rooms = set(baseline.list_rooms())
        current_rooms = set(current.list_rooms())
        
        # Removing rooms is breaking
        removed = baseline_rooms - current_rooms
        for room in removed:
            self.changes.append(RegistryChange(
                change_type=ChangeType.REMOVED,
                severity=ChangeSeverity.BREAKING,
                entity_type="room_scope",
                entity_id=room,
                details=f"Room scope removed",
            ))
        
        # Adding rooms is safe
        added = current_rooms - baseline_rooms
        for room in added:
            self.changes.append(RegistryChange(
                change_type=ChangeType.ADDED,
                severity=ChangeSeverity.SAFE,
                entity_type="room_scope",
                entity_id=room,
                details=f"New room scope added",
            ))
        
        # Deep comparison for common room scopes.
        common_rooms = baseline_rooms & current_rooms
        for room in common_rooms:
            baseline_scope = baseline.get_room_scope(room)
            current_scope = current.get_room_scope(room)
            if baseline_scope is None or current_scope is None:
                continue
            
            removed_valid_events = set(baseline_scope.valid_events) - set(current_scope.valid_events)
            if removed_valid_events:
                self.changes.append(RegistryChange(
                    change_type=ChangeType.MODIFIED,
                    severity=ChangeSeverity.BREAKING,
                    entity_type="room_scope",
                    entity_id=room,
                    details=f"Removed valid_events: {sorted(removed_valid_events)}",
                ))
            added_valid_events = set(current_scope.valid_events) - set(baseline_scope.valid_events)
            if added_valid_events:
                self.changes.append(RegistryChange(
                    change_type=ChangeType.MODIFIED,
                    severity=ChangeSeverity.SAFE,
                    entity_type="room_scope",
                    entity_id=room,
                    details=f"Added valid_events: {sorted(added_valid_events)}",
                ))
            
            removed_kpi_groups = set(baseline_scope.kpi_groups) - set(current_scope.kpi_groups)
            if removed_kpi_groups:
                self.changes.append(RegistryChange(
                    change_type=ChangeType.MODIFIED,
                    severity=ChangeSeverity.BREAKING,
                    entity_type="room_scope",
                    entity_id=room,
                    details=f"Removed kpi_groups: {sorted(removed_kpi_groups)}",
                ))
            added_kpi_groups = set(current_scope.kpi_groups) - set(baseline_scope.kpi_groups)
            if added_kpi_groups:
                self.changes.append(RegistryChange(
                    change_type=ChangeType.MODIFIED,
                    severity=ChangeSeverity.SAFE,
                    entity_type="room_scope",
                    entity_id=room,
                    details=f"Added kpi_groups: {sorted(added_kpi_groups)}",
                ))
            
            if baseline_scope.is_entrance != current_scope.is_entrance:
                self.changes.append(RegistryChange(
                    change_type=ChangeType.MODIFIED,
                    severity=ChangeSeverity.WARNING,
                    entity_type="room_scope",
                    entity_id=room,
                    details=(
                        f"entrance_room changed from {baseline_scope.is_entrance} "
                        f"to {current_scope.is_entrance}"
                    ),
                ))
            
            if baseline_scope.is_sleep_room != current_scope.is_sleep_room:
                self.changes.append(RegistryChange(
                    change_type=ChangeType.MODIFIED,
                    severity=ChangeSeverity.WARNING,
                    entity_type="room_scope",
                    entity_id=room,
                    details=(
                        f"sleep_room changed from {baseline_scope.is_sleep_room} "
                        f"to {current_scope.is_sleep_room}"
                    ),
                ))
    
    def _compare_kpi_groups(
        self,
        baseline: ADLEventRegistry,
        current: ADLEventRegistry,
    ) -> None:
        """Compare KPI group definitions."""
        baseline_groups = set(baseline.list_kpi_groups())
        current_groups = set(current.list_kpi_groups())
        
        # Removing KPI groups is breaking
        removed = baseline_groups - current_groups
        for group in removed:
            self.changes.append(RegistryChange(
                change_type=ChangeType.REMOVED,
                severity=ChangeSeverity.BREAKING,
                entity_type="kpi_group",
                entity_id=group,
                details=f"KPI group removed",
            ))
        
        # Adding KPI groups is safe
        added = current_groups - baseline_groups
        for group in added:
            self.changes.append(RegistryChange(
                change_type=ChangeType.ADDED,
                severity=ChangeSeverity.SAFE,
                entity_type="kpi_group",
                entity_id=group,
                details=f"New KPI group added",
            ))
        
        # Deep comparison for common KPI groups.
        common_groups = baseline_groups & current_groups
        for group in common_groups:
            baseline_group = baseline.get_kpi_group(group)
            current_group = current.get_kpi_group(group)
            if baseline_group is None or current_group is None:
                continue
            
            baseline_metrics = {metric.name: metric for metric in baseline_group.metrics}
            current_metrics = {metric.name: metric for metric in current_group.metrics}
            
            removed_metrics = set(baseline_metrics) - set(current_metrics)
            for metric_name in sorted(removed_metrics):
                self.changes.append(RegistryChange(
                    change_type=ChangeType.REMOVED,
                    severity=ChangeSeverity.BREAKING,
                    entity_type="kpi_metric",
                    entity_id=f"{group}:{metric_name}",
                    details=f"Metric removed from KPI group '{group}'",
                ))
            
            added_metrics = set(current_metrics) - set(baseline_metrics)
            for metric_name in sorted(added_metrics):
                self.changes.append(RegistryChange(
                    change_type=ChangeType.ADDED,
                    severity=ChangeSeverity.SAFE,
                    entity_type="kpi_metric",
                    entity_id=f"{group}:{metric_name}",
                    details=f"Metric added to KPI group '{group}'",
                ))
            
            common_metrics = set(baseline_metrics) & set(current_metrics)
            for metric_name in sorted(common_metrics):
                baseline_metric = baseline_metrics[metric_name]
                current_metric = current_metrics[metric_name]
                
                if baseline_metric.metric_type != current_metric.metric_type:
                    self.changes.append(RegistryChange(
                        change_type=ChangeType.MODIFIED,
                        severity=ChangeSeverity.BREAKING,
                        entity_type="kpi_metric",
                        entity_id=f"{group}:{metric_name}",
                        details=(
                            f"Metric type changed from {baseline_metric.metric_type} "
                            f"to {current_metric.metric_type}"
                        ),
                    ))
                
                if baseline_metric.unit != current_metric.unit:
                    self.changes.append(RegistryChange(
                        change_type=ChangeType.MODIFIED,
                        severity=ChangeSeverity.WARNING,
                        entity_type="kpi_metric",
                        entity_id=f"{group}:{metric_name}",
                        details=f"Metric unit changed from {baseline_metric.unit} to {current_metric.unit}",
                    ))
                
                baseline_threshold_keys = set(baseline_metric.thresholds.keys())
                current_threshold_keys = set(current_metric.thresholds.keys())
                removed_thresholds = baseline_threshold_keys - current_threshold_keys
                for threshold in sorted(removed_thresholds):
                    self.changes.append(RegistryChange(
                        change_type=ChangeType.REMOVED,
                        severity=ChangeSeverity.BREAKING,
                        entity_type="kpi_threshold",
                        entity_id=f"{group}:{metric_name}:{threshold}",
                        details=f"Threshold '{threshold}' removed",
                    ))
                added_thresholds = current_threshold_keys - baseline_threshold_keys
                for threshold in sorted(added_thresholds):
                    self.changes.append(RegistryChange(
                        change_type=ChangeType.ADDED,
                        severity=ChangeSeverity.SAFE,
                        entity_type="kpi_threshold",
                        entity_id=f"{group}:{metric_name}:{threshold}",
                        details=f"Threshold '{threshold}' added",
                    ))
                for threshold in sorted(baseline_threshold_keys & current_threshold_keys):
                    baseline_value = baseline_metric.thresholds[threshold]
                    current_value = current_metric.thresholds[threshold]
                    if baseline_value != current_value:
                        self.changes.append(RegistryChange(
                            change_type=ChangeType.MODIFIED,
                            severity=ChangeSeverity.WARNING,
                            entity_type="kpi_threshold",
                            entity_id=f"{group}:{metric_name}:{threshold}",
                            details=f"Threshold changed from {baseline_value} to {current_value}",
                        ))


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="PR-A2: CI ADL Registry Validator"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline registry YAML"
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current registry YAML"
    )
    parser.add_argument(
        "--migrations-dir",
        help="Path to migrations directory"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Treat warnings as errors"
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Allow warnings (not recommended for CI)"
    )
    parser.add_argument(
        "--json-output",
        help="Path for JSON report output"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    strict = not args.no_strict
    
    migrations_dir = Path(args.migrations_dir) if args.migrations_dir else None
    
    validator = ADLRegistryCIValidator(
        migrations_dir=migrations_dir,
        strict=strict,
    )
    
    result = validator.validate(
        baseline_path=Path(args.baseline),
        current_path=Path(args.current),
    )
    
    # Print report
    print(result.format_report())
    
    # Write JSON report if requested
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nJSON report written to: {args.json_output}")
    
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
