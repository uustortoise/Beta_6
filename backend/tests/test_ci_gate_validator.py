"""
Tests for PR-4: CI Release Gate Hardening

Tests the CI gate validator to ensure it correctly detects gate bypasses.
"""

import os
import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.ci_gate_validator import (
    CIGateValidator,
    GateBypassMethod,
    GateBypassEvidence,
    CIGateValidationResult,
    validate_in_ci,
    main,
)


class TestCIGateValidator(unittest.TestCase):
    """Tests for the CI gate validator."""
    
    def setUp(self):
        """Save original environment."""
        self._original_env = dict(os.environ)
    
    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self._original_env)
    
    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        validator = CIGateValidator(strict=True)
        self.assertTrue(validator.strict)
        self.assertEqual(validator.bypass_evidence, [])
        self.assertEqual(validator.checks_performed, [])
    
    def test_no_bypasses_clean_environment(self):
        """Test validation passes with clean environment."""
        # Clear any bypass env vars
        for var in CIGateValidator.BYPASS_ENV_VARS:
            if var in os.environ:
                del os.environ[var]
        
        os.environ["TRAINING_PROFILE"] = "production"
        
        validator = CIGateValidator(strict=True)
        result = validator.validate()
        
        self.assertTrue(result.passed)
        self.assertEqual(len(result.bypass_evidence), 0)
        self.assertGreater(len(result.checks_performed), 0)
    
    def test_detect_skip_gates_env_var(self):
        """Test detection of SKIP_GATES environment variable."""
        os.environ["SKIP_GATES"] = "1"
        os.environ["TRAINING_PROFILE"] = "production"
        
        validator = CIGateValidator(strict=True)
        result = validator.validate()
        
        self.assertFalse(result.passed)
        
        # Find the SKIP_GATES evidence
        skip_evidence = [e for e in result.bypass_evidence 
                        if e.source == "SKIP_GATES"]
        self.assertEqual(len(skip_evidence), 1)
        self.assertEqual(skip_evidence[0].method, GateBypassMethod.ENV_VAR)
        self.assertEqual(skip_evidence[0].severity, "error")
    
    def test_detect_bypass_gates_env_var(self):
        """Test detection of BYPASS_GATES environment variable."""
        os.environ["BYPASS_GATES"] = "true"
        os.environ["TRAINING_PROFILE"] = "production"
        
        validator = CIGateValidator(strict=True)
        result = validator.validate()
        
        self.assertFalse(result.passed)
        
        bypass_evidence = [e for e in result.bypass_evidence 
                          if e.source == "BYPASS_GATES"]
        self.assertEqual(len(bypass_evidence), 1)
    
    def test_detect_force_promote_env_var(self):
        """Test detection of FORCE_PROMOTE environment variable."""
        os.environ["FORCE_PROMOTE"] = "yes"
        os.environ["TRAINING_PROFILE"] = "production"
        
        validator = CIGateValidator(strict=True)
        result = validator.validate()
        
        self.assertFalse(result.passed)
    
    def test_no_bypass_with_false_value(self):
        """Test that 'false' values don't trigger bypass detection."""
        os.environ["SKIP_GATES"] = "false"
        os.environ["BYPASS_GATES"] = "0"
        os.environ["FORCE_PROMOTE"] = "no"
        os.environ["TRAINING_PROFILE"] = "production"
        
        validator = CIGateValidator(strict=True)
        result = validator.validate()
        
        # Should pass - falsy values don't trigger bypass
        self.assertTrue(result.passed)
    
    def test_warning_on_unrecognized_value(self):
        """Test warning on unrecognized env var value."""
        os.environ["SKIP_GATES"] = "maybe"
        os.environ["TRAINING_PROFILE"] = "production"
        
        validator = CIGateValidator(strict=True)
        result = validator.validate()
        
        # In strict mode, warnings become errors
        self.assertFalse(result.passed)
        
        # In non-strict mode, should pass with warning
        validator = CIGateValidator(strict=False)
        result = validator.validate()
        self.assertTrue(result.passed)
        
        warning_evidence = [e for e in result.bypass_evidence 
                           if e.severity == "warning"]
        self.assertGreaterEqual(len(warning_evidence), 1)
    
    def test_detect_invalid_training_profile(self):
        """Test detection of invalid training profile."""
        os.environ["TRAINING_PROFILE"] = "invalid_profile"
        
        validator = CIGateValidator(strict=True)
        result = validator.validate()
        
        self.assertFalse(result.passed)
        
        profile_evidence = [e for e in result.bypass_evidence 
                           if e.source == "TRAINING_PROFILE" and 
                           e.method == GateBypassMethod.CONFIG_FLAG]
        self.assertGreaterEqual(len(profile_evidence), 1)
    
    def test_warning_on_pilot_in_ci(self):
        """Test warning when pilot profile is used in CI."""
        os.environ["TRAINING_PROFILE"] = "pilot"
        os.environ["CI"] = "true"
        
        validator = CIGateValidator(strict=True)
        result = validator.validate()
        
        # Should have warning about pilot in CI
        pilot_warnings = [e for e in result.bypass_evidence 
                         if "pilot" in e.details.lower() and 
                         e.severity == "warning"]
        self.assertGreaterEqual(len(pilot_warnings), 1)
        
        # In strict mode, this becomes an error
        self.assertFalse(result.passed)
    
    def test_validation_result_formatting(self):
        """Test validation result report formatting."""
        evidence = GateBypassEvidence(
            method=GateBypassMethod.ENV_VAR,
            source="TEST_VAR",
            details="Test bypass evidence",
            severity="error"
        )
        
        result = CIGateValidationResult(
            passed=False,
            bypass_evidence=[evidence],
            checks_performed=["check1", "check2"]
        )
        
        report = result.format_report()
        
        self.assertIn("CI GATE VALIDATION REPORT", report)
        self.assertIn("FAILED", report)
        self.assertIn("TEST_VAR", report)
        self.assertIn("Test bypass evidence", report)
        self.assertIn("check1", report)
    
    def test_validation_result_to_dict(self):
        """Test validation result serialization."""
        evidence = GateBypassEvidence(
            method=GateBypassMethod.ENV_VAR,
            source="TEST_VAR",
            details="Test",
            severity="error"
        )
        
        result = CIGateValidationResult(
            passed=False,
            bypass_evidence=[evidence],
            checks_performed=["check1"]
        )
        
        data = result.to_dict()
        
        self.assertEqual(data["passed"], False)
        self.assertEqual(data["bypass_count"], 1)
        self.assertEqual(len(data["bypass_evidence"]), 1)
        self.assertEqual(data["checks_performed"], ["check1"])


class TestValidateInCI(unittest.TestCase):
    """Tests for validate_in_ci function."""
    
    def setUp(self):
        """Save original environment."""
        self._original_env = dict(os.environ)
    
    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self._original_env)
        # Clean up test report file
        test_report = Path("ci_gate_validation_report.json")
        if test_report.exists():
            test_report.unlink()
    
    def test_skip_validation_outside_ci(self):
        """Test that validation is skipped outside CI."""
        # Clear CI env vars
        for var in ['CI', 'GITHUB_ACTIONS', 'GITLAB_CI', 'CIRCLECI', 'JENKINS_URL']:
            if var in os.environ:
                del os.environ[var]
        
        result = validate_in_ci(strict=True)
        
        # Should return True (skip) outside CI
        self.assertTrue(result)
    
    def test_validate_in_ci_environment(self):
        """Test validation runs in CI environment."""
        os.environ["CI"] = "true"
        os.environ["TRAINING_PROFILE"] = "production"
        
        # Clear any bypass vars
        for var in CIGateValidator.BYPASS_ENV_VARS:
            if var in os.environ:
                del os.environ[var]
        
        result = validate_in_ci(strict=True)
        
        # Should pass with clean environment
        self.assertTrue(result)
        
        # Should write report file
        self.assertTrue(Path("ci_gate_validation_report.json").exists())
    
    def test_fail_on_bypass_in_ci(self):
        """Test that validation fails when bypass detected in CI."""
        os.environ["CI"] = "true"
        os.environ["SKIP_GATES"] = "1"
        os.environ["TRAINING_PROFILE"] = "production"
        
        result = validate_in_ci(strict=True)
        
        # Should fail with bypass
        self.assertFalse(result)


class TestMainFunction(unittest.TestCase):
    """Tests for main CLI function."""
    
    def setUp(self):
        """Save original environment."""
        self._original_env = dict(os.environ)
    
    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self._original_env)
        # Clean up test files
        for f in ["test_report.json", "ci_gate_validation_report.json"]:
            if Path(f).exists():
                Path(f).unlink()
    
    @patch('sys.argv', ['ci_gate_validator', '--json-output', 'test_report.json'])
    def test_main_clean_environment(self):
        """Test main function with clean environment."""
        os.environ["TRAINING_PROFILE"] = "production"
        
        # Clear any bypass vars
        for var in CIGateValidator.BYPASS_ENV_VARS:
            if var in os.environ:
                del os.environ[var]
        
        exit_code = main()
        
        self.assertEqual(exit_code, 0)
        self.assertTrue(Path("test_report.json").exists())
    
    @patch('sys.argv', ['ci_gate_validator'])
    def test_main_with_bypass(self):
        """Test main function exits with error on bypass."""
        os.environ["SKIP_GATES"] = "1"
        os.environ["TRAINING_PROFILE"] = "production"
        
        exit_code = main()
        
        self.assertEqual(exit_code, 1)
    
    @patch('sys.argv', ['ci_gate_validator', '--no-strict'])
    def test_main_no_strict(self):
        """Test main function with --no-strict flag."""
        os.environ["TRAINING_PROFILE"] = "production"
        
        # Clear any bypass vars
        for var in CIGateValidator.BYPASS_ENV_VARS:
            if var in os.environ:
                del os.environ[var]
        
        exit_code = main()
        
        # Should pass with clean environment even in non-strict mode
        self.assertEqual(exit_code, 0)


class TestGateBypassMethods(unittest.TestCase):
    """Tests for GateBypassMethod enum."""
    
    def test_enum_values(self):
        """Test that enum values are correct."""
        self.assertEqual(GateBypassMethod.ENV_VAR.value, "environment_variable")
        self.assertEqual(GateBypassMethod.CONFIG_FLAG.value, "configuration_flag")
        self.assertEqual(GateBypassMethod.MISSING_GATE_CHECK.value, "missing_gate_check")
        self.assertEqual(GateBypassMethod.FORCE_PROMOTE.value, "force_promote")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios."""
    
    def setUp(self):
        """Save original environment."""
        self._original_env = dict(os.environ)
    
    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self._original_env)
    
    def test_multiple_bypasses_detected(self):
        """Test that multiple bypasses are all detected."""
        os.environ["SKIP_GATES"] = "1"
        os.environ["BYPASS_GATES"] = "true"
        os.environ["FORCE_PROMOTE"] = "yes"
        os.environ["TRAINING_PROFILE"] = "invalid"
        
        validator = CIGateValidator(strict=True)
        result = validator.validate()
        
        self.assertFalse(result.passed)
        self.assertGreaterEqual(len(result.bypass_evidence), 3)
    
    def test_production_profile_clean(self):
        """Test that production profile with no bypasses passes."""
        os.environ["TRAINING_PROFILE"] = "production"
        
        # Clear any CI vars to simulate non-CI
        for var in ['CI', 'GITHUB_ACTIONS', 'GITLAB_CI', 'CIRCLECI', 'JENKINS_URL']:
            if var in os.environ:
                del os.environ[var]
        
        validator = CIGateValidator(strict=True)
        result = validator.validate()
        
        self.assertTrue(result.passed)


if __name__ == '__main__':
    unittest.main()
