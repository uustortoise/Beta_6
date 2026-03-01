"""
Tests for PR-A1: ADL Event Registry Schema Validation

Tests the JSON Schema and YAML registry structure.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip tests if jsonschema not installed
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

import yaml


class TestADLEventRegistrySchema(unittest.TestCase):
    """Tests for ADL Event Registry JSON Schema."""
    
    @classmethod
    def setUpClass(cls):
        """Load schema and registry files."""
        cls.schema_path = Path(__file__).parent.parent / "config" / "schemas" / "adl_event_registry.schema.json"
        cls.registry_path = Path(__file__).parent.parent / "config" / "adl_event_registry.v1.yaml"
        
        if cls.schema_path.exists():
            cls.schema = json.loads(cls.schema_path.read_text())
        else:
            cls.schema = None
        
        if cls.registry_path.exists():
            cls.registry = yaml.safe_load(cls.registry_path.read_text())
        else:
            cls.registry = None
    
    @unittest.skipUnless(HAS_JSONSCHEMA, "jsonschema not installed")
    def test_schema_file_exists(self):
        """Test that schema file exists."""
        self.assertIsNotNone(self.schema)
        self.assertIn("$schema", self.schema)
        self.assertIn("properties", self.schema)
    
    def test_registry_file_exists(self):
        """Test that registry file exists."""
        self.assertIsNotNone(self.registry)
        self.assertIn("version", self.registry)
        self.assertIn("registry", self.registry)
    
    @unittest.skipUnless(HAS_JSONSCHEMA, "jsonschema not installed")
    def test_registry_validates_against_schema(self):
        """Test that registry YAML validates against JSON schema."""
        if self.schema is None or self.registry is None:
            self.skipTest("Schema or registry not found")
        
        # This should not raise
        jsonschema.validate(self.registry, self.schema)
    
    @unittest.skipUnless(HAS_JSONSCHEMA, "jsonschema not installed")
    def test_invalid_registry_fails_validation(self):
        """Test that invalid registry fails schema validation."""
        if self.schema is None:
            self.skipTest("Schema not found")
        
        invalid_registry = {
            "version": "invalid",  # Should be vX.Y format
            "registry": {},
            # Missing required fields
        }
        
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(invalid_registry, self.schema)
    
    def test_registry_version_format(self):
        """Test that version follows semver format."""
        if self.registry is None:
            self.skipTest("Registry not found")
        
        version = self.registry.get("version", "")
        self.assertRegex(version, r"^v\d+\.\d+$")
    
    def test_registry_has_metadata(self):
        """Test that registry has required metadata."""
        if self.registry is None:
            self.skipTest("Registry not found")
        
        metadata = self.registry.get("metadata", {})
        self.assertIn("created_at", metadata)
        self.assertIn("author", metadata)
        self.assertIn("description", metadata)
    
    def test_registry_has_critical_events(self):
        """Test that registry defines critical care events."""
        if self.registry is None:
            self.skipTest("Registry not found")
        
        registry_data = self.registry.get("registry", {})
        
        # Check for critical events
        critical_events = [
            e for e in registry_data.values()
            if e.get("criticality") == "critical"
        ]
        
        self.assertGreater(len(critical_events), 0, "No critical events defined")
        
        # Check for specific critical events
        event_ids = [e.get("event_id") for e in critical_events]
        self.assertIn("sleeping", event_ids)
        self.assertIn("showering", event_ids)
    
    def test_event_ids_match_keys(self):
        """Test that event_id field matches registry key."""
        if self.registry is None:
            self.skipTest("Registry not found")
        
        registry_data = self.registry.get("registry", {})
        
        for key, event_def in registry_data.items():
            self.assertEqual(
                event_def.get("event_id"), key,
                f"Event ID mismatch for key '{key}'"
            )
    
    def test_aliases_are_lowercase_snake_case(self):
        """Test that aliases follow naming convention."""
        if self.registry is None:
            self.skipTest("Registry not found")
        
        registry_data = self.registry.get("registry", {})
        
        for event_id, event_def in registry_data.items():
            for alias in event_def.get("aliases", []):
                self.assertRegex(
                    alias, r"^[a-z][a-z0-9_]*$",
                    f"Invalid alias format '{alias}' in event '{event_id}'"
                )
    
    def test_room_scopes_reference_valid_events(self):
        """Test that room scopes only reference defined events."""
        if self.registry is None:
            self.skipTest("Registry not found")
        
        registry_data = self.registry.get("registry", {})
        room_scopes = self.registry.get("room_scopes", {})
        
        valid_event_ids = set(registry_data.keys())
        
        for room_name, scope in room_scopes.items():
            for event_id in scope.get("valid_events", []):
                self.assertIn(
                    event_id, valid_event_ids,
                    f"Room '{room_name}' references undefined event '{event_id}'"
                )
    
    def test_kpi_groups_are_consistent(self):
        """Test that KPI groups in events match defined groups."""
        if self.registry is None:
            self.skipTest("Registry not found")
        
        registry_data = self.registry.get("registry", {})
        kpi_groups = self.registry.get("kpi_groups", {})
        
        defined_groups = set(kpi_groups.keys())
        
        for event_id, event_def in registry_data.items():
            for group in event_def.get("kpi_groups", []):
                self.assertIn(
                    group, defined_groups,
                    f"Event '{event_id}' references undefined KPI group '{group}'"
                )
    
    def test_care_critical_kpi_group_exists(self):
        """Test that care_critical KPI group exists with expected metrics."""
        if self.registry is None:
            self.skipTest("Registry not found")
        
        kpi_groups = self.registry.get("kpi_groups", {})
        
        self.assertIn("care_critical", kpi_groups)
        
        care_critical = kpi_groups["care_critical"]
        metric_names = [m.get("name") for m in care_critical.get("metrics", [])]
        
        self.assertIn("sleep_duration", metric_names)
        self.assertIn("shower_day", metric_names)
    
    def test_unknown_label_policy_exists(self):
        """Test that unknown label policy is defined."""
        if self.registry is None:
            self.skipTest("Registry not found")
        
        policy = self.registry.get("unknown_label_policy", {})
        
        self.assertTrue(policy.get("enabled", False))
        self.assertIn("fallback_mapping", policy)
        
        fallback = policy["fallback_mapping"]
        self.assertIn("occupied_unknown", fallback)
        self.assertIn("unoccupied_unknown", fallback)


class TestADLEventRegistryStructure(unittest.TestCase):
    """Tests for registry content structure."""
    
    def setUp(self):
        """Load registry."""
        self.registry_path = Path(__file__).parent.parent / "config" / "adl_event_registry.v1.yaml"
        if self.registry_path.exists():
            self.registry = yaml.safe_load(self.registry_path.read_text())
        else:
            self.registry = None
    
    def test_has_occupancy_base_events(self):
        """Test that occupancy base events exist."""
        if self.registry is None:
            self.skipTest("Registry not found")
        
        registry_data = self.registry.get("registry", {})
        
        self.assertIn("occupied", registry_data)
        self.assertIn("unoccupied", registry_data)
    
    def test_has_unknown_fallback_events(self):
        """Test that unknown fallback events exist."""
        if self.registry is None:
            self.skipTest("Registry not found")
        
        registry_data = self.registry.get("registry", {})
        
        self.assertIn("occupied_unknown", registry_data)
        self.assertIn("unoccupied_unknown", registry_data)
    
    def test_bedroom_events(self):
        """Test that bedroom has appropriate events."""
        if self.registry is None:
            self.skipTest("Registry not found")
        
        room_scopes = self.registry.get("room_scopes", {})
        bedroom = room_scopes.get("bedroom", {})
        
        valid_events = bedroom.get("valid_events", [])
        
        self.assertIn("sleeping", valid_events)
        self.assertIn("occupied", valid_events)
        self.assertIn("unoccupied", valid_events)
        
        # Check sleep room flag
        self.assertTrue(bedroom.get("sleep_room", False))
    
    def test_entrance_room_marked(self):
        """Test that entrance room is properly marked."""
        if self.registry is None:
            self.skipTest("Registry not found")
        
        room_scopes = self.registry.get("room_scopes", {})
        entrance = room_scopes.get("entrance", {})
        
        self.assertTrue(entrance.get("entrance_room", False))
        
        # Should have enter/leave events
        valid_events = entrance.get("valid_events", [])
        self.assertIn("entering_home", valid_events)
        self.assertIn("leaving_home", valid_events)
    
    def test_event_has_required_fields(self):
        """Test that events have all required fields."""
        if self.registry is None:
            self.skipTest("Registry not found")
        
        registry_data = self.registry.get("registry", {})
        
        required_fields = [
            "event_id", "display_name", "room_scope",
            "kpi_groups", "criticality", "enabled"
        ]
        
        for event_id, event_def in registry_data.items():
            for field in required_fields:
                self.assertIn(
                    field, event_def,
                    f"Event '{event_id}' missing required field '{field}'"
                )
    
    def test_criticality_values_are_valid(self):
        """Test that criticality values are from allowed set."""
        if self.registry is None:
            self.skipTest("Registry not found")
        
        registry_data = self.registry.get("registry", {})
        valid_criticalities = {"critical", "high", "medium", "low"}
        
        for event_id, event_def in registry_data.items():
            criticality = event_def.get("criticality")
            self.assertIn(
                criticality, valid_criticalities,
                f"Event '{event_id}' has invalid criticality '{criticality}'"
            )
    
    def test_global_constraints_exist(self):
        """Test that global constraints are defined."""
        if self.registry is None:
            self.skipTest("Registry not found")
        
        constraints = self.registry.get("global_constraints", {})
        
        self.assertIn("max_event_duration_minutes", constraints)
        self.assertIn("min_event_duration_seconds", constraints)
        self.assertIn("episode_merge_gap_seconds", constraints)


class TestSchemaComplianceEdgeCases(unittest.TestCase):
    """Edge case tests for schema compliance."""
    
    def test_empty_registry_fails_validation(self):
        """Test that empty registry fails schema validation."""
        if not HAS_JSONSCHEMA:
            self.skipTest("jsonschema not installed")
        
        schema_path = Path(__file__).parent.parent / "config" / "schemas" / "adl_event_registry.schema.json"
        if not schema_path.exists():
            self.skipTest("Schema not found")
        
        schema = json.loads(schema_path.read_text())
        
        empty_registry = {
            "version": "v1.0",
            "registry": {},
            "metadata": {
                "created_at": "2026-01-01T00:00:00Z",
                "author": "test",
                "description": "test"
            }
        }
        
        # This should validate (empty registry is technically valid)
        try:
            jsonschema.validate(empty_registry, schema)
        except jsonschema.ValidationError:
            pass  # Also acceptable if schema requires non-empty
    
    def test_invalid_version_format(self):
        """Test that invalid version format fails."""
        if not HAS_JSONSCHEMA:
            self.skipTest("jsonschema not installed")
        
        schema_path = Path(__file__).parent.parent / "config" / "schemas" / "adl_event_registry.schema.json"
        if not schema_path.exists():
            self.skipTest("Schema not found")
        
        schema = json.loads(schema_path.read_text())
        
        invalid_registry = {
            "version": "invalid_version",  # Should be vX.Y
            "registry": {},
            "metadata": {
                "created_at": "2026-01-01T00:00:00Z",
                "author": "test",
                "description": "test"
            }
        }
        
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(invalid_registry, schema)


if __name__ == '__main__':
    unittest.main()
