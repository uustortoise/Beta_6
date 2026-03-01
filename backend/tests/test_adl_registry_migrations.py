"""
Tests for PR-A2: ADL Registry Migrations and CI Validation

Tests the CI validator and migration system.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from ml.adl_registry import SchemaValidationError
from ml.ci_adl_registry_validator import (
    ADLRegistryCIValidator,
    ADLRegistryMigrationLoader,
    ValidationResult,
    RegistryChange,
    ChangeType,
    ChangeSeverity,
)


class TestADLRegistryMigrationLoader(unittest.TestCase):
    """Tests for migration loader."""
    
    def setUp(self):
        """Create temp migration directory."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_migrations(self):
        """Test loading migration files."""
        # Create a migration file
        migration = {
            "from_version": "v1.0",
            "to_version": "v2.0",
            "migrations": [
                {
                    "from": "old_event",
                    "to": "new_event",
                    "type": "rename"
                }
            ]
        }
        
        mig_file = Path(self.temp_dir) / "v1.0_to_v2.0.yaml"
        mig_file.write_text(yaml.dump(migration))
        
        loader = ADLRegistryMigrationLoader(migrations_dir=Path(self.temp_dir))
        migrations = loader.load_migrations("v1.0", "v2.0")
        
        self.assertIn("old_event", migrations)
        self.assertEqual(migrations["old_event"]["to"], "new_event")
    
    def test_has_migration(self):
        """Test checking if migration exists."""
        migration = {
            "from_version": "v1.0",
            "to_version": "v2.0",
            "migrations": [
                {"from": "old_event", "to": "new_event", "type": "rename"}
            ]
        }
        
        mig_file = Path(self.temp_dir) / "v1.0_to_v2.0.yaml"
        mig_file.write_text(yaml.dump(migration))
        
        loader = ADLRegistryMigrationLoader(migrations_dir=Path(self.temp_dir))
        
        self.assertTrue(loader.has_migration("v1.0", "v2.0", "old_event"))
        self.assertFalse(loader.has_migration("v1.0", "v2.0", "nonexistent"))
    
    def test_missing_migration_file(self):
        """Test handling of missing migration file."""
        loader = ADLRegistryMigrationLoader(migrations_dir=Path(self.temp_dir))
        migrations = loader.load_migrations("v1.0", "v2.0")
        
        self.assertEqual(migrations, {})


class TestADLRegistryCIValidator(unittest.TestCase):
    """Tests for CI validator."""
    
    def setUp(self):
        """Create test registries."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create baseline registry
        self.baseline = {
            "version": "v1.0",
            "metadata": {
                "created_at": "2026-01-01T00:00:00Z",
                "author": "test",
                "description": "Baseline"
            },
            "registry": {
                "sleeping": {
                    "event_id": "sleeping",
                    "display_name": "Sleeping",
                    "description": "Sleep",
                    "aliases": ["asleep"],
                    "room_scope": ["bedroom"],
                    "kpi_groups": ["sleep"],
                    "criticality": "critical",
                    "enabled": True,
                    "derivation_rules": {},
                    "metrics": {}
                },
                "cooking": {
                    "event_id": "cooking",
                    "display_name": "Cooking",
                    "description": "Cook",
                    "aliases": [],
                    "room_scope": ["kitchen"],
                    "kpi_groups": ["nutrition"],
                    "criticality": "high",
                    "enabled": True,
                    "derivation_rules": {},
                    "metrics": {}
                }
            },
            "room_scopes": {
                "bedroom": {
                    "valid_events": ["sleeping"],
                    "kpi_groups": ["sleep"],
                    "entrance_room": False,
                    "sleep_room": True
                },
                "kitchen": {
                    "valid_events": ["cooking"],
                    "kpi_groups": ["nutrition"],
                    "entrance_room": False,
                    "sleep_room": False
                }
            },
            "kpi_groups": {
                "sleep": {"description": "Sleep", "metrics": [], "care_outcomes": []},
                "nutrition": {"description": "Nutrition", "metrics": [], "care_outcomes": []}
            },
            "unknown_label_policy": {"enabled": False, "fallback_mapping": {}},
            "global_constraints": {}
        }
        
        self.baseline_path = Path(self.temp_dir) / "baseline.yaml"
        self.baseline_path.write_text(yaml.dump(self.baseline))
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_no_changes_passes(self):
        """Test that identical registries pass validation."""
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(self.baseline))
        
        validator = ADLRegistryCIValidator(strict=True)
        result = validator.validate(self.baseline_path, current_path)
        
        self.assertTrue(result.passed)
        self.assertEqual(len(result.changes), 0)
    
    def test_adding_event_is_safe(self):
        """Test that adding events is safe."""
        current = self.baseline.copy()
        current["registry"]["new_event"] = {
            "event_id": "new_event",
            "display_name": "New Event",
            "description": "New",
            "aliases": [],
            "room_scope": ["bedroom"],
            "kpi_groups": ["sleep"],
            "criticality": "medium",
            "enabled": True,
            "derivation_rules": {},
            "metrics": {}
        }
        
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(strict=True)
        result = validator.validate(self.baseline_path, current_path)
        
        self.assertTrue(result.passed)
        
        added = [c for c in result.changes if c.change_type == ChangeType.ADDED]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].entity_id, "new_event")
        self.assertEqual(added[0].severity, ChangeSeverity.SAFE)
    
    def test_removing_event_is_breaking(self):
        """Test that removing events is breaking."""
        current = self.baseline.copy()
        del current["registry"]["sleeping"]
        
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(strict=True)
        result = validator.validate(self.baseline_path, current_path)
        
        self.assertFalse(result.passed)
        
        removed = [c for c in result.changes if c.change_type == ChangeType.REMOVED]
        sleeping_removed = [c for c in removed if c.entity_id == "sleeping"]
        self.assertEqual(len(sleeping_removed), 1)
        self.assertEqual(sleeping_removed[0].severity, ChangeSeverity.BREAKING)
        self.assertFalse(sleeping_removed[0].migration_available)
    
    def test_removing_event_with_migration_is_warning(self):
        """Test that removed events with migration show as warning."""
        # Create migration file
        migrations_dir = Path(self.temp_dir) / "migrations"
        migrations_dir.mkdir()
        
        migration = {
            "from_version": "v1.0",
            "to_version": "v1.1",
            "migrations": [
                {"from": "sleeping", "to": "resting", "type": "rename"}
            ]
        }
        mig_file = migrations_dir / "v1.0_to_v1.1.yaml"
        mig_file.write_text(yaml.dump(migration))
        
        # Update current version
        current = self.baseline.copy()
        current["version"] = "v1.1"
        del current["registry"]["sleeping"]
        
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(
            migrations_dir=migrations_dir,
            strict=False  # Allow warnings
        )
        result = validator.validate(self.baseline_path, current_path)
        
        # Should pass in non-strict mode
        self.assertTrue(result.passed)
        
        removed = [c for c in result.changes if c.change_type == ChangeType.REMOVED]
        sleeping_removed = [c for c in removed if c.entity_id == "sleeping"]
        self.assertEqual(len(sleeping_removed), 1)
        self.assertEqual(sleeping_removed[0].severity, ChangeSeverity.WARNING)
        self.assertTrue(sleeping_removed[0].migration_available)
    
    def test_adding_alias_is_safe(self):
        """Test that adding aliases is safe."""
        current = self.baseline.copy()
        current["registry"]["sleeping"]["aliases"] = ["asleep", "resting"]
        
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(strict=True)
        result = validator.validate(self.baseline_path, current_path)
        
        self.assertTrue(result.passed)
        
        added = [c for c in result.changes if c.change_type == ChangeType.ADDED]
        alias_added = [c for c in added if c.entity_type == "alias"]
        self.assertEqual(len(alias_added), 1)
    
    def test_removing_alias_is_breaking(self):
        """Test that removing aliases is breaking."""
        # First add an alias to baseline
        self.baseline["registry"]["sleeping"]["aliases"] = ["asleep", "resting"]
        self.baseline_path.write_text(yaml.dump(self.baseline))
        
        # Now remove it in current
        current = self.baseline.copy()
        current["registry"]["sleeping"]["aliases"] = ["asleep"]
        
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(strict=True)
        result = validator.validate(self.baseline_path, current_path)
        
        self.assertFalse(result.passed)
        
        removed = [c for c in result.changes if c.change_type == ChangeType.REMOVED]
        alias_removed = [c for c in removed if c.entity_type == "alias"]
        self.assertEqual(len(alias_removed), 1)
        self.assertEqual(alias_removed[0].severity, ChangeSeverity.BREAKING)
    
    def test_disabling_event_is_breaking(self):
        """Test that disabling events is breaking."""
        current = self.baseline.copy()
        current["registry"]["sleeping"]["enabled"] = False
        
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(strict=True)
        result = validator.validate(self.baseline_path, current_path)
        
        self.assertFalse(result.passed)
        
        modified = [c for c in result.changes if c.change_type == ChangeType.MODIFIED]
        disabled = [c for c in modified if "disabled" in c.details.lower()]
        self.assertEqual(len(disabled), 1)
    
    def test_upgrading_criticality_is_breaking(self):
        """Test that upgrading criticality (to more strict) is breaking."""
        current = self.baseline.copy()
        current["registry"]["cooking"]["criticality"] = "critical"
        
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(strict=True)
        result = validator.validate(self.baseline_path, current_path)
        
        self.assertFalse(result.passed)
    
    def test_downgrading_criticality_is_warning(self):
        """Test that downgrading criticality is warning."""
        current = self.baseline.copy()
        current["registry"]["sleeping"]["criticality"] = "medium"
        
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(strict=False)
        result = validator.validate(self.baseline_path, current_path)
        
        # Should pass in non-strict mode
        self.assertTrue(result.passed)
        
        modified = [c for c in result.changes if c.change_type == ChangeType.MODIFIED]
        criticality_change = [c for c in modified if "criticality" in c.details.lower()]
        self.assertEqual(len(criticality_change), 1)
        self.assertEqual(criticality_change[0].severity, ChangeSeverity.WARNING)
    
    def test_removing_room_from_event_is_breaking(self):
        """Test that removing rooms from events is breaking."""
        # Add a second room to sleeping in baseline
        self.baseline["registry"]["sleeping"]["room_scope"] = ["bedroom", "livingroom"]
        self.baseline_path.write_text(yaml.dump(self.baseline))
        
        # Remove livingroom in current
        current = self.baseline.copy()
        current["registry"]["sleeping"]["room_scope"] = ["bedroom"]
        
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(strict=True)
        result = validator.validate(self.baseline_path, current_path)
        
        self.assertFalse(result.passed)
    
    def test_removing_room_scope_is_breaking(self):
        """Test that removing room scopes is breaking."""
        current = self.baseline.copy()
        del current["room_scopes"]["kitchen"]
        
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(strict=True)
        result = validator.validate(self.baseline_path, current_path)
        
        self.assertFalse(result.passed)
        
        removed = [c for c in result.changes if c.entity_type == "room_scope"]
        self.assertEqual(len(removed), 1)
    
    def test_removing_kpi_group_is_breaking(self):
        """Test that removing KPI groups is breaking."""
        current = self.baseline.copy()
        del current["kpi_groups"]["nutrition"]
        
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(strict=True)
        result = validator.validate(self.baseline_path, current_path)
        
        self.assertFalse(result.passed)
        
        removed = [c for c in result.changes if c.entity_type == "kpi_group"]
        self.assertEqual(len(removed), 1)
    
    def test_removing_event_kpi_group_is_breaking(self):
        """Removing KPI mapping from an event is a contract break."""
        current = self.baseline.copy()
        current["registry"]["sleeping"]["kpi_groups"] = []
        
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(strict=True)
        result = validator.validate(self.baseline_path, current_path)
        
        self.assertFalse(result.passed)
        breaking = [c for c in result.changes if c.severity == ChangeSeverity.BREAKING]
        self.assertTrue(any("Removed KPI groups" in c.details for c in breaking))
    
    def test_removing_room_valid_event_is_breaking(self):
        """Removing valid_events from a room is breaking."""
        current = self.baseline.copy()
        current["room_scopes"]["bedroom"]["valid_events"] = []
        
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(strict=True)
        result = validator.validate(self.baseline_path, current_path)
        
        self.assertFalse(result.passed)
        breaking = [c for c in result.changes if c.severity == ChangeSeverity.BREAKING]
        self.assertTrue(any("Removed valid_events" in c.details for c in breaking))
    
    def test_removing_room_kpi_group_is_breaking(self):
        """Removing KPI groups from room scope is breaking."""
        current = self.baseline.copy()
        current["room_scopes"]["kitchen"]["kpi_groups"] = []
        
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(strict=True)
        result = validator.validate(self.baseline_path, current_path)
        
        self.assertFalse(result.passed)
        breaking = [c for c in result.changes if c.severity == ChangeSeverity.BREAKING]
        self.assertTrue(any("Removed kpi_groups" in c.details for c in breaking))
    
    def test_removing_kpi_metric_is_breaking(self):
        """Removing KPI metric from an existing group is breaking."""
        self.baseline["kpi_groups"]["sleep"]["metrics"] = [{
            "name": "sleep_duration",
            "type": "duration",
            "unit": "minutes",
            "thresholds": {"target": 420, "minimum": 300}
        }]
        self.baseline_path.write_text(yaml.dump(self.baseline))
        
        current = self.baseline.copy()
        current["kpi_groups"]["sleep"]["metrics"] = []
        
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(strict=True)
        result = validator.validate(self.baseline_path, current_path)
        
        self.assertFalse(result.passed)
        self.assertTrue(any(c.entity_type == "kpi_metric" for c in result.changes))
    
    def test_kpi_threshold_change_is_warning(self):
        """Threshold value changes produce warnings (strict mode will fail)."""
        self.baseline["kpi_groups"]["sleep"]["metrics"] = [{
            "name": "sleep_duration",
            "type": "duration",
            "unit": "minutes",
            "thresholds": {"target": 420, "minimum": 300}
        }]
        self.baseline_path.write_text(yaml.dump(self.baseline))
        
        current = self.baseline.copy()
        current["kpi_groups"]["sleep"]["metrics"][0]["thresholds"]["target"] = 400
        
        current_path = Path(self.temp_dir) / "current.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(strict=False)
        result = validator.validate(self.baseline_path, current_path)
        
        self.assertTrue(result.passed)
        warnings = [c for c in result.changes if c.severity == ChangeSeverity.WARNING]
        self.assertTrue(any(c.entity_type == "kpi_threshold" for c in warnings))
    
    def test_validator_enforces_schema_validation(self):
        """Validator should fail early on schema-invalid registry files."""
        current = self.baseline.copy()
        current["version"] = "invalid-version"
        
        current_path = Path(self.temp_dir) / "current_invalid.yaml"
        current_path.write_text(yaml.dump(current))
        
        validator = ADLRegistryCIValidator(strict=True)
        with self.assertRaises(SchemaValidationError):
            validator.validate(self.baseline_path, current_path)


class TestValidationResultFormatting(unittest.TestCase):
    """Tests for validation result formatting."""
    
    def test_format_report_with_no_changes(self):
        """Test report formatting with no changes."""
        result = ValidationResult(
            passed=True,
            changes=[],
            baseline_version="v1.0",
            current_version="v1.0"
        )
        
        report = result.format_report()
        
        self.assertIn("PASSED", report)
        self.assertIn("v1.0", report)
    
    def test_format_report_with_breaking_changes(self):
        """Test report formatting with breaking changes."""
        result = ValidationResult(
            passed=False,
            changes=[
                RegistryChange(
                    change_type=ChangeType.REMOVED,
                    severity=ChangeSeverity.BREAKING,
                    entity_type="event",
                    entity_id="sleeping",
                    details="Event removed"
                ),
                RegistryChange(
                    change_type=ChangeType.ADDED,
                    severity=ChangeSeverity.SAFE,
                    entity_type="event",
                    entity_id="new_event",
                    details="New event"
                )
            ],
            baseline_version="v1.0",
            current_version="v1.1"
        )
        
        report = result.format_report()
        
        self.assertIn("FAILED", report)
        self.assertIn("sleeping", report)
        self.assertIn("new_event", report)
        self.assertIn("BREAKING", report)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ValidationResult(
            passed=True,
            changes=[
                RegistryChange(
                    change_type=ChangeType.ADDED,
                    severity=ChangeSeverity.SAFE,
                    entity_type="event",
                    entity_id="test",
                    details="Test"
                )
            ],
            baseline_version="v1.0",
            current_version="v1.1"
        )
        
        data = result.to_dict()
        
        self.assertTrue(data["passed"])
        self.assertEqual(data["baseline_version"], "v1.0")
        self.assertEqual(data["current_version"], "v1.1")
        self.assertEqual(data["change_count"], 1)
        self.assertEqual(data["breaking_count"], 0)


if __name__ == '__main__':
    unittest.main()
