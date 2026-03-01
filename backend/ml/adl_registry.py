"""
PR-A1: ADL Event Registry Loader

Production-quality loader for Activity of Daily Living (ADL) event taxonomy registry.

Features:
- Schema validation with detailed error reporting
- Alias normalization and collision detection
- Unknown label fallback handling
- Room-scoped event lookup
- KPI group resolution

Usage:
    from ml.adl_registry import ADLEventRegistry, get_default_registry
    
    registry = get_default_registry()
    event = registry.get_event("sleeping")
    event = registry.resolve_alias("asleep")  # Returns sleeping event
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union

from ml.yaml_compat import load_yaml_file

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path(__file__).parent.parent / "config" / "adl_event_registry.v1.yaml"
DEFAULT_SCHEMA_PATH = Path(__file__).parent.parent / "config" / "schemas" / "adl_event_registry.schema.json"


class CriticalityLevel(Enum):
    """Event criticality levels for care outcomes."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RegistryError(Exception):
    """Base exception for registry errors."""
    pass


class SchemaValidationError(RegistryError):
    """Raised when registry fails schema validation."""
    pass


class AliasCollisionError(RegistryError):
    """Raised when alias collisions are detected."""
    pass


class UnknownEventError(RegistryError):
    """Raised when an unknown event is requested."""
    pass


class UnknownLabelPolicyDisabledError(UnknownEventError):
    """Raised when unknown label fallback is requested but disabled by policy."""
    pass


@dataclass(frozen=True)
class EventDefinition:
    """Canonical event definition."""
    event_id: str
    display_name: str
    description: str
    room_scope: Tuple[str, ...]
    kpi_groups: Tuple[str, ...]
    criticality: CriticalityLevel
    enabled: bool
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    parent_event: Optional[str] = None
    derivation_rules: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, bool] = field(default_factory=dict)

    @property
    def is_occupancy_derived(self) -> bool:
        """Check if event is derived from occupancy detection."""
        return self.derivation_rules.get("from_occupancy", False)

    @property
    def min_duration_seconds(self) -> int:
        """Get minimum duration threshold for event detection."""
        return self.derivation_rules.get("min_duration_seconds", 30)

    @property
    def requires_sustained(self) -> bool:
        """Check if event requires sustained activation."""
        return self.derivation_rules.get("requires_sustained", False)


@dataclass(frozen=True)
class RoomScope:
    """Room scope definition."""
    room_name: str
    valid_events: Tuple[str, ...]
    kpi_groups: Tuple[str, ...]
    is_entrance: bool = False
    is_sleep_room: bool = False


@dataclass(frozen=True)
class KPIMetric:
    """KPI metric definition."""
    name: str
    metric_type: str
    unit: Optional[str] = None
    description: Optional[str] = None
    thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class KPIGroup:
    """KPI group definition."""
    name: str
    description: str
    metrics: List[KPIMetric]
    care_outcomes: List[str]


@dataclass
class UnknownLabelPolicy:
    """Policy for handling unknown labels."""
    enabled: bool
    occupied_unknown_event: str
    occupied_unknown_confidence: float
    unoccupied_unknown_event: str
    unoccupied_unknown_confidence: float


class ADLEventRegistry:
    """
    ADL Event Registry loader and query interface.
    
    This class provides production-quality access to the ADL event taxonomy
    with validation, normalization, and error handling.
    """

    def __init__(
        self,
        registry_path: Optional[Union[str, Path]] = None,
        schema_path: Optional[Union[str, Path]] = None,
        skip_validation: bool = False,
    ):
        """
        Initialize registry from YAML file.
        
        Args:
            registry_path: Path to registry YAML file
            schema_path: Path to JSON schema file
            skip_validation: Skip schema validation (not recommended for production)
            
        Raises:
            SchemaValidationError: If registry fails schema validation
            AliasCollisionError: If alias collisions detected
            FileNotFoundError: If registry or schema file not found
        """
        self._registry_path = Path(registry_path or DEFAULT_REGISTRY_PATH)
        self._schema_path = Path(schema_path or DEFAULT_SCHEMA_PATH)
        self._skip_validation = skip_validation
        
        # Raw data storage
        self._raw_data: Dict[str, Any] = {}
        self._version: str = ""
        self._metadata: Dict[str, Any] = {}
        
        # Normalized data structures
        self._events: Dict[str, EventDefinition] = {}
        self._aliases: Dict[str, str] = {}  # alias -> canonical event_id
        self._room_scopes: Dict[str, RoomScope] = {}
        self._kpi_groups: Dict[str, KPIGroup] = {}
        self._unknown_policy: Optional[UnknownLabelPolicy] = None
        self._global_constraints: Dict[str, Any] = {}
        
        # Load and validate
        self._load()
        
        logger.info(f"Loaded ADL registry {self._version} with {len(self._events)} events")

    def _load(self) -> None:
        """Load and validate registry data."""
        # Check files exist
        if not self._registry_path.exists():
            raise FileNotFoundError(f"Registry file not found: {self._registry_path}")
        
        # Load YAML
        self._raw_data = load_yaml_file(self._registry_path)
        
        # Validate schema if requested
        if not self._skip_validation:
            self._validate_schema()
        
        # Extract version and metadata
        self._version = self._raw_data.get("version", "unknown")
        self._metadata = self._raw_data.get("metadata", {})
        
        # Normalize data structures
        self._normalize_events()
        self._normalize_room_scopes()
        self._normalize_kpi_groups()
        self._normalize_unknown_policy()
        self._normalize_global_constraints()
        
        # Validate alias uniqueness
        self._validate_alias_uniqueness()

    def _validate_schema(self) -> None:
        """Validate registry against JSON schema."""
        try:
            import jsonschema
            
            if not self._schema_path.exists():
                logger.warning(f"Schema file not found: {self._schema_path}, skipping validation")
                return
            
            schema = json.loads(self._schema_path.read_text())
            jsonschema.validate(self._raw_data, schema)
            logger.debug("Schema validation passed")
            
        except ImportError:
            logger.warning("jsonschema not installed, skipping schema validation")
        except jsonschema.ValidationError as e:
            raise SchemaValidationError(
                f"Registry schema validation failed: {e.message} "
                f"at path {list(e.path)}"
            ) from e
        except Exception as e:
            raise SchemaValidationError(f"Schema validation error: {e}") from e

    def _normalize_events(self) -> None:
        """Normalize event definitions."""
        registry_data = self._raw_data.get("registry", {})
        
        for event_id, event_data in registry_data.items():
            # Validate event_id matches key
            if event_data.get("event_id") != event_id:
                raise RegistryError(
                    f"Event ID mismatch: key '{event_id}' vs event_id '{event_data.get('event_id')}'"
                )
            
            # Parse criticality
            criticality_str = event_data.get("criticality", "medium")
            try:
                criticality = CriticalityLevel(criticality_str)
            except ValueError:
                raise RegistryError(f"Invalid criticality '{criticality_str}' for event '{event_id}'")
            
            event_def = EventDefinition(
                event_id=event_id,
                display_name=event_data.get("display_name", event_id),
                description=event_data.get("description", ""),
                room_scope=tuple(event_data.get("room_scope", [])),
                kpi_groups=tuple(event_data.get("kpi_groups", [])),
                criticality=criticality,
                enabled=event_data.get("enabled", True),
                aliases=tuple(event_data.get("aliases", [])),
                parent_event=event_data.get("parent_event"),
                derivation_rules=event_data.get("derivation_rules", {}),
                metrics=event_data.get("metrics", {}),
            )
            
            self._events[event_id] = event_def

    def _normalize_room_scopes(self) -> None:
        """Normalize room scope definitions."""
        room_scopes_data = self._raw_data.get("room_scopes", {})
        
        for room_name, scope_data in room_scopes_data.items():
            room_scope = RoomScope(
                room_name=room_name,
                valid_events=tuple(scope_data.get("valid_events", [])),
                kpi_groups=tuple(scope_data.get("kpi_groups", [])),
                is_entrance=scope_data.get("entrance_room", False),
                is_sleep_room=scope_data.get("sleep_room", False),
            )
            self._room_scopes[room_name] = room_scope

    def _normalize_kpi_groups(self) -> None:
        """Normalize KPI group definitions."""
        kpi_groups_data = self._raw_data.get("kpi_groups", {})
        
        for group_name, group_data in kpi_groups_data.items():
            metrics = []
            for metric_data in group_data.get("metrics", []):
                metric = KPIMetric(
                    name=metric_data.get("name", ""),
                    metric_type=metric_data.get("type", "custom"),
                    unit=metric_data.get("unit"),
                    description=metric_data.get("description"),
                    thresholds=metric_data.get("thresholds", {}),
                )
                metrics.append(metric)
            
            kpi_group = KPIGroup(
                name=group_name,
                description=group_data.get("description", ""),
                metrics=metrics,
                care_outcomes=group_data.get("care_outcomes", []),
            )
            self._kpi_groups[group_name] = kpi_group

    def _normalize_unknown_policy(self) -> None:
        """Normalize unknown label policy."""
        policy_data = self._raw_data.get("unknown_label_policy", {})
        
        if not policy_data.get("enabled", True):
            self._unknown_policy = None
            return
        
        fallback = policy_data.get("fallback_mapping", {})
        occupied = fallback.get("occupied_unknown", {})
        unoccupied = fallback.get("unoccupied_unknown", {})
        
        self._unknown_policy = UnknownLabelPolicy(
            enabled=True,
            occupied_unknown_event=occupied.get("event_id", "occupied_unknown"),
            occupied_unknown_confidence=occupied.get("confidence", 0.5),
            unoccupied_unknown_event=unoccupied.get("event_id", "unoccupied_unknown"),
            unoccupied_unknown_confidence=unoccupied.get("confidence", 0.5),
        )
        
        # Validate unknown fallback events exist in registry.
        if self._unknown_policy.occupied_unknown_event not in self._events:
            raise RegistryError(
                f"unknown_label_policy.occupied_unknown event "
                f"'{self._unknown_policy.occupied_unknown_event}' is not defined in registry"
            )
        if self._unknown_policy.unoccupied_unknown_event not in self._events:
            raise RegistryError(
                f"unknown_label_policy.unoccupied_unknown event "
                f"'{self._unknown_policy.unoccupied_unknown_event}' is not defined in registry"
            )

    def _normalize_global_constraints(self) -> None:
        """Normalize global constraints."""
        constraints_data = self._raw_data.get("global_constraints", {})
        self._global_constraints = {
            "max_event_duration_minutes": constraints_data.get("max_event_duration_minutes", 1440),
            "min_event_duration_seconds": constraints_data.get("min_event_duration_seconds", 30),
            "episode_merge_gap_seconds": constraints_data.get("episode_merge_gap_seconds", 60),
        }

    def _validate_alias_uniqueness(self) -> None:
        """Validate that aliases are unique across all events."""
        alias_to_events: Dict[str, List[str]] = {}
        
        for event_id, event_def in self._events.items():
            # Event ID itself is an implicit alias
            if event_id not in alias_to_events:
                alias_to_events[event_id] = []
            alias_to_events[event_id].append(event_id)
            
            # Add explicit aliases
            for alias in event_def.aliases:
                if alias not in alias_to_events:
                    alias_to_events[alias] = []
                alias_to_events[alias].append(event_id)
        
        # Check for collisions
        collisions = []
        for alias, events in alias_to_events.items():
            if len(events) > 1:
                collisions.append(f"'{alias}' -> {events}")
            else:
                # Store valid alias mapping
                self._aliases[alias] = events[0]
        
        if collisions:
            raise AliasCollisionError(
                f"Alias collisions detected: {'; '.join(collisions)}"
            )

    # Public API
    
    @property
    def version(self) -> str:
        """Get registry version."""
        return self._version
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get registry metadata."""
        return self._metadata.copy()
    
    @property
    def global_constraints(self) -> Dict[str, Any]:
        """Get global constraints."""
        return self._global_constraints.copy()
    
    def get_event(self, event_id: str) -> EventDefinition:
        """
        Get event definition by canonical ID.
        
        Args:
            event_id: Canonical event ID
            
        Returns:
            EventDefinition for the event
            
        Raises:
            UnknownEventError: If event not found
        """
        if event_id not in self._events:
            raise UnknownEventError(f"Unknown event: '{event_id}'")
        return self._events[event_id]
    
    def resolve_alias(self, alias: str) -> str:
        """
        Resolve an alias to canonical event ID.
        
        Args:
            alias: Event alias or canonical ID
            
        Returns:
            Canonical event ID
            
        Raises:
            UnknownEventError: If alias not found
        """
        if alias in self._aliases:
            return self._aliases[alias]
        raise UnknownEventError(f"Unknown event or alias: '{alias}'")
    
    def resolve_to_event(self, alias: str) -> EventDefinition:
        """
        Resolve an alias to full EventDefinition.
        
        Args:
            alias: Event alias or canonical ID
            
        Returns:
            EventDefinition for the resolved event
        """
        event_id = self.resolve_alias(alias)
        return self.get_event(event_id)
    
    def get_events_for_room(self, room_name: str) -> List[EventDefinition]:
        """
        Get all valid events for a specific room.
        
        Args:
            room_name: Room name
            
        Returns:
            List of EventDefinitions valid for the room
        """
        room_scope = self._room_scopes.get(room_name)
        if not room_scope:
            return []
        
        events = []
        for event_id in room_scope.valid_events:
            if event_id in self._events:
                events.append(self._events[event_id])
        return events
    
    def is_valid_event_for_room(self, event_id: str, room_name: str) -> bool:
        """
        Check if an event is valid for a specific room.
        
        Args:
            event_id: Event ID or alias
            room_name: Room name
            
        Returns:
            True if event is valid for the room
        """
        try:
            canonical_id = self.resolve_alias(event_id)
            room_scope = self._room_scopes.get(room_name)
            if not room_scope:
                return False
            return canonical_id in room_scope.valid_events
        except UnknownEventError:
            return False
    
    def get_room_scope(self, room_name: str) -> Optional[RoomScope]:
        """
        Get room scope definition.
        
        Args:
            room_name: Room name
            
        Returns:
            RoomScope or None if room not defined
        """
        return self._room_scopes.get(room_name)
    
    def get_kpi_group(self, group_name: str) -> Optional[KPIGroup]:
        """
        Get KPI group definition.
        
        Args:
            group_name: KPI group name
            
        Returns:
            KPIGroup or None if group not defined
        """
        return self._kpi_groups.get(group_name)
    
    def get_events_by_kpi_group(self, group_name: str) -> List[EventDefinition]:
        """
        Get all events belonging to a KPI group.
        
        Args:
            group_name: KPI group name
            
        Returns:
            List of EventDefinitions in the group
        """
        events = []
        for event_def in self._events.values():
            if group_name in event_def.kpi_groups:
                events.append(event_def)
        return events
    
    def get_critical_events(self) -> List[EventDefinition]:
        """
        Get all critical-level events.
        
        Returns:
            List of critical EventDefinitions
        """
        return [
            e for e in self._events.values()
            if e.criticality == CriticalityLevel.CRITICAL
        ]
    
    def get_enabled_events(self) -> List[EventDefinition]:
        """
        Get all enabled events.
        
        Returns:
            List of enabled EventDefinitions
        """
        return [e for e in self._events.values() if e.enabled]
    
    def resolve_unknown_label(
        self,
        label: str,
        occupied: bool = True,
    ) -> Tuple[str, float]:
        """
        Resolve an unknown label to fallback event.
        
        Args:
            label: Unknown label string
            occupied: Whether room is considered occupied
            
        Returns:
            Tuple of (event_id, confidence)
        
        Raises:
            UnknownLabelPolicyDisabledError: If unknown fallback policy is disabled
        """
        if self._unknown_policy is None or not self._unknown_policy.enabled:
            raise UnknownLabelPolicyDisabledError(
                f"Unknown label fallback is disabled. Cannot resolve label '{label}'."
            )
        
        if occupied:
            return (
                self._unknown_policy.occupied_unknown_event,
                self._unknown_policy.occupied_unknown_confidence,
            )
        else:
            return (
                self._unknown_policy.unoccupied_unknown_event,
                self._unknown_policy.unoccupied_unknown_confidence,
            )
    
    def list_all_events(self) -> List[str]:
        """Get list of all canonical event IDs."""
        return list(self._events.keys())
    
    def list_all_aliases(self) -> Dict[str, str]:
        """Get mapping of all aliases to canonical IDs."""
        return self._aliases.copy()
    
    def list_rooms(self) -> List[str]:
        """Get list of all defined rooms."""
        return list(self._room_scopes.keys())
    
    def list_kpi_groups(self) -> List[str]:
        """Get list of all KPI groups."""
        return list(self._kpi_groups.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Export registry to dictionary."""
        return {
            "version": self._version,
            "metadata": self._metadata,
            "events": {k: {
                "event_id": v.event_id,
                "display_name": v.display_name,
                "description": v.description,
                "room_scope": list(v.room_scope),
                "kpi_groups": list(v.kpi_groups),
                "criticality": v.criticality.value,
                "enabled": v.enabled,
                "aliases": list(v.aliases),
                "parent_event": v.parent_event,
            } for k, v in self._events.items()},
            "aliases": self._aliases,
            "rooms": list(self._room_scopes.keys()),
            "kpi_groups": list(self._kpi_groups.keys()),
        }


# Singleton instance for default registry
_default_registry: Optional[ADLEventRegistry] = None


def get_default_registry(skip_validation: bool = False) -> ADLEventRegistry:
    """
    Get singleton instance of default registry.
    
    Args:
        skip_validation: Skip schema validation (not recommended for production)
        
    Returns:
        ADLEventRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ADLEventRegistry(skip_validation=skip_validation)
    return _default_registry


def reload_registry(skip_validation: bool = False) -> ADLEventRegistry:
    """Force reload of default registry.
    
    Args:
        skip_validation: Skip schema validation (not recommended for production)
        
    Returns:
        ADLEventRegistry instance
    """
    global _default_registry
    _default_registry = ADLEventRegistry(skip_validation=skip_validation)
    return _default_registry
