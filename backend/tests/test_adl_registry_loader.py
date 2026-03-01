"""
Tests for PR-A1: ADL Registry Loader

Tests the ADLEventRegistry loader functionality.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from ml.adl_registry import (
    ADLEventRegistry,
    CriticalityLevel,
    AliasCollisionError,
    UnknownEventError,
    UnknownLabelPolicyDisabledError,
    RegistryError,
    get_default_registry,
    reload_registry,
)


class TestADLEventRegistryLoader(unittest.TestCase):
    """Tests for ADLEventRegistry loader."""
    
    def setUp(self):
        """Create test registry files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a valid test registry
        self.valid_registry = {
            "version": "v1.0",
            "metadata": {
                "created_at": "2026-01-01T00:00:00Z",
                "author": "test",
                "description": "Test registry"
            },
            "registry": {
                "sleeping": {
                    "event_id": "sleeping",
                    "display_name": "Sleeping",
                    "description": "Resident is sleeping",
                    "aliases": ["asleep", "rest"],
                    "room_scope": ["bedroom"],
                    "kpi_groups": ["sleep_quality"],
                    "criticality": "critical",
                    "enabled": True,
                    "derivation_rules": {"min_duration_seconds": 300},
                    "metrics": {"track_duration": True}
                },
                "cooking": {
                    "event_id": "cooking",
                    "display_name": "Cooking",
                    "description": "Preparing food",
                    "aliases": ["meal_prep"],
                    "room_scope": ["kitchen"],
                    "kpi_groups": ["nutrition"],
                    "criticality": "high",
                    "enabled": True,
                    "derivation_rules": {},
                    "metrics": {}
                },
                "occupied": {
                    "event_id": "occupied",
                    "display_name": "Occupied",
                    "description": "Room occupied",
                    "aliases": [],
                    "room_scope": ["bedroom", "kitchen"],
                    "kpi_groups": ["occupancy"],
                    "criticality": "medium",
                    "enabled": True,
                    "derivation_rules": {},
                    "metrics": {}
                },
                "occupied_unknown": {
                    "event_id": "occupied_unknown",
                    "display_name": "Occupied Unknown",
                    "description": "Unknown activity",
                    "aliases": [],
                    "room_scope": ["bedroom", "kitchen"],
                    "kpi_groups": [],
                    "criticality": "low",
                    "enabled": True,
                    "derivation_rules": {},
                    "metrics": {}
                },
                "unoccupied_unknown": {
                    "event_id": "unoccupied_unknown",
                    "display_name": "Unoccupied Unknown",
                    "description": "Unknown empty state",
                    "aliases": [],
                    "room_scope": ["bedroom", "kitchen"],
                    "kpi_groups": [],
                    "criticality": "low",
                    "enabled": True,
                    "derivation_rules": {},
                    "metrics": {}
                }
            },
            "room_scopes": {
                "bedroom": {
                    "valid_events": ["sleeping", "occupied", "occupied_unknown", "unoccupied_unknown"],
                    "kpi_groups": ["sleep_quality"],
                    "entrance_room": False,
                    "sleep_room": True
                },
                "kitchen": {
                    "valid_events": ["cooking", "occupied", "occupied_unknown", "unoccupied_unknown"],
                    "kpi_groups": ["nutrition"],
                    "entrance_room": False,
                    "sleep_room": False
                }
            },
            "kpi_groups": {
                "sleep_quality": {
                    "description": "Sleep metrics",
                    "metrics": [{"name": "duration", "type": "duration", "unit": "minutes"}],
                    "care_outcomes": []
                },
                "nutrition": {
                    "description": "Nutrition metrics",
                    "metrics": [{"name": "cooking_time", "type": "duration", "unit": "minutes"}],
                    "care_outcomes": []
                },
                "occupancy": {
                    "description": "Basic occupancy",
                    "metrics": [],
                    "care_outcomes": []
                }
            },
            "unknown_label_policy": {
                "enabled": True,
                "fallback_mapping": {
                    "occupied_unknown": {"event_id": "occupied_unknown", "confidence": 0.5},
                    "unoccupied_unknown": {"event_id": "unoccupied_unknown", "confidence": 0.5}
                }
            },
            "global_constraints": {
                "max_event_duration_minutes": 1440,
                "min_event_duration_seconds": 30,
                "episode_merge_gap_seconds": 60
            }
        }
        
        self.registry_path = Path(self.temp_dir) / "test_registry.yaml"
        self.registry_path.write_text(yaml.dump(self.valid_registry))
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_valid_registry(self):
        """Test loading a valid registry."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        self.assertEqual(registry.version, "v1.0")
        self.assertEqual(len(registry.list_all_events()), 5)
    
    def test_get_event_by_id(self):
        """Test getting event by canonical ID."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        event = registry.get_event("sleeping")
        
        self.assertEqual(event.event_id, "sleeping")
        self.assertEqual(event.display_name, "Sleeping")
        self.assertEqual(event.criticality, CriticalityLevel.CRITICAL)
        self.assertTrue(event.enabled)
        self.assertEqual(event.min_duration_seconds, 300)
    
    def test_get_event_not_found(self):
        """Test getting non-existent event raises error."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        with self.assertRaises(UnknownEventError):
            registry.get_event("nonexistent")
    
    def test_resolve_alias(self):
        """Test resolving alias to canonical ID."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        # Alias should resolve
        self.assertEqual(registry.resolve_alias("asleep"), "sleeping")
        self.assertEqual(registry.resolve_alias("rest"), "sleeping")
        
        # Canonical ID should also resolve
        self.assertEqual(registry.resolve_alias("sleeping"), "sleeping")
    
    def test_resolve_unknown_alias(self):
        """Test resolving unknown alias raises error."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        with self.assertRaises(UnknownEventError):
            registry.resolve_alias("unknown_alias")
    
    def test_resolve_to_event(self):
        """Test resolving alias to full event definition."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        event = registry.resolve_to_event("meal_prep")
        
        self.assertEqual(event.event_id, "cooking")
        self.assertEqual(event.display_name, "Cooking")
    
    def test_get_events_for_room(self):
        """Test getting events for a specific room."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        bedroom_events = registry.get_events_for_room("bedroom")
        event_ids = [e.event_id for e in bedroom_events]
        
        self.assertIn("sleeping", event_ids)
        self.assertIn("occupied", event_ids)
        self.assertNotIn("cooking", event_ids)
    
    def test_is_valid_event_for_room(self):
        """Test checking event validity for room."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        self.assertTrue(registry.is_valid_event_for_room("sleeping", "bedroom"))
        self.assertFalse(registry.is_valid_event_for_room("sleeping", "kitchen"))
        self.assertTrue(registry.is_valid_event_for_room("cooking", "kitchen"))
    
    def test_get_room_scope(self):
        """Test getting room scope definition."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        bedroom_scope = registry.get_room_scope("bedroom")
        
        self.assertIsNotNone(bedroom_scope)
        self.assertTrue(bedroom_scope.is_sleep_room)
        self.assertFalse(bedroom_scope.is_entrance)
    
    def test_get_kpi_group(self):
        """Test getting KPI group."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        group = registry.get_kpi_group("sleep_quality")
        
        self.assertIsNotNone(group)
        self.assertEqual(group.name, "sleep_quality")
        self.assertEqual(len(group.metrics), 1)
    
    def test_get_events_by_kpi_group(self):
        """Test getting events by KPI group."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        sleep_events = registry.get_events_by_kpi_group("sleep_quality")
        event_ids = [e.event_id for e in sleep_events]
        
        self.assertIn("sleeping", event_ids)
    
    def test_get_critical_events(self):
        """Test getting critical events."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        critical = registry.get_critical_events()
        event_ids = [e.event_id for e in critical]
        
        self.assertIn("sleeping", event_ids)
        self.assertNotIn("occupied", event_ids)
    
    def test_get_enabled_events(self):
        """Test getting enabled events."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        enabled = registry.get_enabled_events()
        
        self.assertEqual(len(enabled), 5)
    
    def test_resolve_unknown_label(self):
        """Test resolving unknown labels."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        # Occupied unknown
        event_id, confidence = registry.resolve_unknown_label("weird_label", occupied=True)
        self.assertEqual(event_id, "occupied_unknown")
        self.assertEqual(confidence, 0.5)
        
        # Unoccupied unknown
        event_id, confidence = registry.resolve_unknown_label("weird_label", occupied=False)
        self.assertEqual(event_id, "unoccupied_unknown")
        self.assertEqual(confidence, 0.5)
    
    def test_resolve_unknown_label_raises_when_policy_disabled(self):
        """Unknown fallback must fail explicitly when policy is disabled."""
        disabled_registry = self.valid_registry.copy()
        disabled_registry["unknown_label_policy"] = {
            "enabled": False,
            "fallback_mapping": {},
        }
        disabled_path = Path(self.temp_dir) / "disabled_policy_registry.yaml"
        disabled_path.write_text(yaml.dump(disabled_registry))
        
        registry = ADLEventRegistry(
            registry_path=disabled_path,
            skip_validation=True
        )
        
        with self.assertRaises(UnknownLabelPolicyDisabledError):
            registry.resolve_unknown_label("unseen_label", occupied=True)
    
    def test_invalid_unknown_fallback_mapping_raises(self):
        """Unknown fallback mapping must reference existing events."""
        invalid_registry = self.valid_registry.copy()
        invalid_registry["unknown_label_policy"] = {
            "enabled": True,
            "fallback_mapping": {
                "occupied_unknown": {"event_id": "missing_event", "confidence": 0.5},
                "unoccupied_unknown": {"event_id": "unoccupied_unknown", "confidence": 0.5},
            },
        }
        invalid_path = Path(self.temp_dir) / "invalid_unknown_fallback.yaml"
        invalid_path.write_text(yaml.dump(invalid_registry))
        
        with self.assertRaises(RegistryError):
            ADLEventRegistry(
                registry_path=invalid_path,
                skip_validation=True
            )
    
    def test_list_all_aliases(self):
        """Test listing all aliases."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        aliases = registry.list_all_aliases()
        
        # Should include both canonical IDs and aliases
        self.assertIn("sleeping", aliases)  # canonical
        self.assertIn("asleep", aliases)    # alias
        self.assertIn("rest", aliases)      # alias
        self.assertEqual(aliases["asleep"], "sleeping")
    
    def test_event_is_occupancy_derived(self):
        """Test checking if event is occupancy-derived."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        sleeping = registry.get_event("sleeping")
        # Default is False unless specified in derivation_rules
        self.assertFalse(sleeping.is_occupancy_derived)
    
    def test_event_requires_sustained(self):
        """Test checking if event requires sustained activation."""
        registry = ADLEventRegistry(
            registry_path=self.registry_path,
            skip_validation=True
        )
        
        sleeping = registry.get_event("sleeping")
        # Default is False
        self.assertFalse(sleeping.requires_sustained)


class TestADLRegistryAliasCollisions(unittest.TestCase):
    """Tests for alias collision detection."""
    
    def test_alias_collision_detection(self):
        """Test that alias collisions are detected."""
        temp_dir = tempfile.mkdtemp()
        
        # Create registry with collision
        collision_registry = {
            "version": "v1.0",
            "metadata": {
                "created_at": "2026-01-01T00:00:00Z",
                "author": "test",
                "description": "Test"
            },
            "registry": {
                "event_a": {
                    "event_id": "event_a",
                    "display_name": "Event A",
                    "description": "Test",
                    "aliases": ["shared_alias"],  # Collision!
                    "room_scope": ["bedroom"],
                    "kpi_groups": [],
                    "criticality": "medium",
                    "enabled": True,
                    "derivation_rules": {},
                    "metrics": {}
                },
                "event_b": {
                    "event_id": "event_b",
                    "display_name": "Event B",
                    "description": "Test",
                    "aliases": ["shared_alias"],  # Same alias - collision!
                    "room_scope": ["bedroom"],
                    "kpi_groups": [],
                    "criticality": "medium",
                    "enabled": True,
                    "derivation_rules": {},
                    "metrics": {}
                }
            },
            "room_scopes": {},
            "kpi_groups": {},
            "unknown_label_policy": {"enabled": False, "fallback_mapping": {}},
            "global_constraints": {}
        }
        
        registry_path = Path(temp_dir) / "collision_registry.yaml"
        registry_path.write_text(yaml.dump(collision_registry))
        
        try:
            with self.assertRaises(AliasCollisionError) as ctx:
                ADLEventRegistry(registry_path=registry_path, skip_validation=True)
            
            self.assertIn("shared_alias", str(ctx.exception))
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_event_id_collision_with_alias(self):
        """Test that event ID colliding with another event's alias is detected."""
        temp_dir = tempfile.mkdtemp()
        
        # Create registry where event_id matches another event's alias
        collision_registry = {
            "version": "v1.0",
            "metadata": {
                "created_at": "2026-01-01T00:00:00Z",
                "author": "test",
                "description": "Test"
            },
            "registry": {
                "sleeping": {
                    "event_id": "sleeping",
                    "display_name": "Sleeping",
                    "description": "Test",
                    "aliases": [],
                    "room_scope": ["bedroom"],
                    "kpi_groups": [],
                    "criticality": "medium",
                    "enabled": True,
                    "derivation_rules": {},
                    "metrics": {}
                },
                "resting": {
                    "event_id": "resting",
                    "display_name": "Resting",
                    "description": "Test",
                    "aliases": ["sleeping"],  # Collides with event_id above!
                    "room_scope": ["bedroom"],
                    "kpi_groups": [],
                    "criticality": "medium",
                    "enabled": True,
                    "derivation_rules": {},
                    "metrics": {}
                }
            },
            "room_scopes": {},
            "kpi_groups": {},
            "unknown_label_policy": {"enabled": False, "fallback_mapping": {}},
            "global_constraints": {}
        }
        
        registry_path = Path(temp_dir) / "collision_registry.yaml"
        registry_path.write_text(yaml.dump(collision_registry))
        
        try:
            with self.assertRaises(AliasCollisionError) as ctx:
                ADLEventRegistry(registry_path=registry_path, skip_validation=True)
            
            self.assertIn("sleeping", str(ctx.exception))
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestADLRegistryFileNotFound(unittest.TestCase):
    """Tests for file not found scenarios."""
    
    def test_registry_file_not_found(self):
        """Test that missing registry file raises error."""
        with self.assertRaises(FileNotFoundError):
            ADLEventRegistry(registry_path="/nonexistent/path/registry.yaml")


class TestADLRegistryEventIDMismatch(unittest.TestCase):
    """Tests for event ID mismatch."""
    
    def test_event_id_mismatch(self):
        """Test that event_id/key mismatch raises error."""
        temp_dir = tempfile.mkdtemp()
        
        mismatch_registry = {
            "version": "v1.0",
            "metadata": {
                "created_at": "2026-01-01T00:00:00Z",
                "author": "test",
                "description": "Test"
            },
            "registry": {
                "sleeping": {
                    "event_id": "cooking",  # Mismatch!
                    "display_name": "Sleeping",
                    "description": "Test",
                    "aliases": [],
                    "room_scope": ["bedroom"],
                    "kpi_groups": [],
                    "criticality": "medium",
                    "enabled": True,
                    "derivation_rules": {},
                    "metrics": {}
                }
            },
            "room_scopes": {},
            "kpi_groups": {},
            "unknown_label_policy": {"enabled": False, "fallback_mapping": {}},
            "global_constraints": {}
        }
        
        registry_path = Path(temp_dir) / "mismatch_registry.yaml"
        registry_path.write_text(yaml.dump(mismatch_registry))
        
        try:
            with self.assertRaises(RegistryError) as ctx:
                ADLEventRegistry(registry_path=registry_path, skip_validation=True)
            
            self.assertIn("mismatch", str(ctx.exception).lower())
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestADLRegistrySingleton(unittest.TestCase):
    """Tests for singleton pattern."""
    
    def test_get_default_registry(self):
        """Test getting default registry (if files exist)."""
        default_path = Path(__file__).parent.parent / "config" / "adl_event_registry.v1.yaml"
        
        if not default_path.exists():
            self.skipTest("Default registry not found")
        
        registry1 = get_default_registry(skip_validation=True)
        registry2 = get_default_registry(skip_validation=True)
        
        # Should be same instance
        self.assertIs(registry1, registry2)
    
    def test_reload_registry(self):
        """Test reloading registry."""
        default_path = Path(__file__).parent.parent / "config" / "adl_event_registry.v1.yaml"
        
        if not default_path.exists():
            self.skipTest("Default registry not found")
        
        registry1 = get_default_registry(skip_validation=True)
        registry2 = reload_registry(skip_validation=True)
        
        # Should be different instances
        self.assertIsNot(registry1, registry2)


if __name__ == '__main__':
    unittest.main()
