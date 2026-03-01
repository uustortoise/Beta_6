"""
PR-4: CI Release Gate Hardening

Validates that ML gates are not bypassed in CI/CD pipelines.
Fails the build if:
- Gates are explicitly bypassed via environment variables (SKIP_GATES, BYPASS_GATES, FORCE_PROMOTE)
- Gate configuration is missing or invalid
- Required gate integration modules are not importable
- Pilot profile is active in CI environment

Note: This validator checks for bypass MECHANISMS (env vars, config), not the actual
gate execution during training. Gate execution is verified by integration tests.

Usage:
    python -m ml.ci_gate_validator [--strict]
    
Exit codes:
    0 - All gate checks passed
    1 - Gate bypass detected or validation failed
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class GateBypassMethod(Enum):
    """Methods by which gates can be bypassed."""
    ENV_VAR = "environment_variable"
    CONFIG_FLAG = "configuration_flag"
    MISSING_GATE_CHECK = "missing_gate_check"
    FORCE_PROMOTE = "force_promote"


@dataclass
class GateBypassEvidence:
    """Evidence of a gate bypass."""
    method: GateBypassMethod
    source: str
    details: str
    severity: str = "error"  # error, warning
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "source": self.source,
            "details": self.details,
            "severity": self.severity,
        }


@dataclass
class CIGateValidationResult:
    """Result of CI gate validation."""
    passed: bool
    bypass_evidence: List[GateBypassEvidence] = field(default_factory=list)
    checks_performed: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "bypass_count": len(self.bypass_evidence),
            "bypass_evidence": [e.to_dict() for e in self.bypass_evidence],
            "checks_performed": self.checks_performed,
        }
    
    def format_report(self) -> str:
        """Format validation result as human-readable report."""
        lines = [
            "=" * 60,
            "CI GATE VALIDATION REPORT",
            "=" * 60,
            f"Status: {'✅ PASSED' if self.passed else '❌ FAILED'}",
            f"Checks performed: {len(self.checks_performed)}",
            "",
        ]
        
        if self.checks_performed:
            lines.append("Checks:")
            for check in self.checks_performed:
                lines.append(f"  ✓ {check}")
            lines.append("")
        
        if self.bypass_evidence:
            lines.append(f"Bypass evidence found ({len(self.bypass_evidence)} items):")
            for evidence in self.bypass_evidence:
                icon = "⚠️ " if evidence.severity == "warning" else "❌ "
                lines.append(f"  {icon}[{evidence.method.value}] {evidence.source}")
                lines.append(f"     {evidence.details}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class CIGateValidator:
    """
    Validates ML gates in CI/CD environment.
    
    Ensures that:
    1. Gates cannot be bypassed via environment variables
    2. Gate configuration is present and valid
    3. Required gate integration modules are importable
    4. Training profile is sane for CI usage
    """
    
    # Environment variables that would bypass gates
    BYPASS_ENV_VARS = [
        "SKIP_GATES",
        "BYPASS_GATES",
        "FORCE_PROMOTE",
        "DISABLE_GATE_CHECK",
        "SKIP_PRE_TRAINING_GATES",
        "SKIP_POST_TRAINING_GATES",
        "SKIP_STATISTICAL_VALIDITY",
    ]
    
    # Required gate configuration files/directories (relative to repo root)
    REQUIRED_CONFIG_PATHS = [
        ("backend/config", True),  # Directory containing room config
        ("backend/ml/policy_config.py", False),  # Training policy file
        ("backend/ml/gate_integration.py", False),  # Gate integration module
    ]
    
    def __init__(self, strict: bool = True):
        """
        Initialize validator.
        
        Args:
            strict: If True, treat warnings as errors
        """
        self.strict = strict
        self.bypass_evidence: List[GateBypassEvidence] = []
        self.checks_performed: List[str] = []
        self._repo_root = self._find_repo_root()
    
    def _find_repo_root(self) -> Path:
        """Find the repository root directory."""
        # Start from current file location
        current = Path(__file__).resolve()
        
        # Walk up until we find a marker file/directory
        for parent in [current] + list(current.parents):
            # Check for common repo markers
            if (parent / ".git").exists() or (parent / "backend").exists():
                return parent
            # Check if we're in backend/ml directory structure
            if parent.name == "backend" and (parent.parent / "backend").exists():
                return parent.parent
        
        # Fallback to current working directory
        return Path.cwd()
    
    def validate(self) -> CIGateValidationResult:
        """
        Run all validation checks.
        
        Returns:
            CIGateValidationResult with pass/fail status and evidence
        """
        self.bypass_evidence = []
        self.checks_performed = []
        
        # Run all checks
        self._check_env_var_bypasses()
        self._check_required_config_files()
        self._check_training_profile()
        self._check_gate_integration_imports()
        
        # Determine pass/fail
        errors = [e for e in self.bypass_evidence if e.severity == "error"]
        warnings = [e for e in self.bypass_evidence if e.severity == "warning"]
        
        passed = len(errors) == 0
        if self.strict and warnings:
            passed = False
        
        return CIGateValidationResult(
            passed=passed,
            bypass_evidence=self.bypass_evidence,
            checks_performed=self.checks_performed,
        )
    
    def _check_env_var_bypasses(self) -> None:
        """Check for environment variables that bypass gates."""
        self.checks_performed.append("environment_variable_bypass_check")
        
        for var_name in self.BYPASS_ENV_VARS:
            value = os.environ.get(var_name)
            if value:
                # Check for truthy values
                if value.lower() in ("1", "true", "yes", "on", "enabled"):
                    self.bypass_evidence.append(GateBypassEvidence(
                        method=GateBypassMethod.ENV_VAR,
                        source=var_name,
                        details=f"Gate bypass env var is set to '{value}'",
                        severity="error",
                    ))
                elif value.lower() not in ("0", "false", "no", "off", "disabled", ""):
                    # Unrecognized value - treat as warning
                    self.bypass_evidence.append(GateBypassEvidence(
                        method=GateBypassMethod.ENV_VAR,
                        source=var_name,
                        details=f"Gate bypass env var has unrecognized value '{value}'",
                        severity="warning",
                    ))
    
    def _check_required_config_files(self) -> None:
        """Check that required gate configuration files/directories exist."""
        self.checks_performed.append("required_config_files_check")
        
        for config_path, is_directory in self.REQUIRED_CONFIG_PATHS:
            full_path = self._repo_root / config_path
            
            # Also try relative to backend subdirectory
            alt_path = self._repo_root / config_path.replace("backend/", "")
            
            found = False
            for path_to_check in [full_path, alt_path]:
                if is_directory:
                    if path_to_check.is_dir():
                        found = True
                        break
                else:
                    if path_to_check.exists():
                        found = True
                        break
            
            if not found:
                path_type = "directory" if is_directory else "file"
                self.bypass_evidence.append(GateBypassEvidence(
                    method=GateBypassMethod.MISSING_GATE_CHECK,
                    source=config_path,
                    details=f"Required configuration {path_type} not found: {config_path}",
                    severity="error" if self.strict else "warning",
                ))
    
    def _check_training_profile(self) -> None:
        """Check that training profile is valid."""
        self.checks_performed.append("training_profile_check")
        
        profile = os.environ.get("TRAINING_PROFILE", "production")
        valid_profiles = ["production", "pilot", "staging", "development"]
        
        if profile not in valid_profiles:
            self.bypass_evidence.append(GateBypassEvidence(
                method=GateBypassMethod.CONFIG_FLAG,
                source="TRAINING_PROFILE",
                details=f"Invalid training profile '{profile}', must be one of {valid_profiles}",
                severity="error",
            ))
        
        # Warn if pilot mode is active in CI
        if profile == "pilot" and os.environ.get("CI"):
            self.bypass_evidence.append(GateBypassEvidence(
                method=GateBypassMethod.CONFIG_FLAG,
                source="TRAINING_PROFILE",
                details="Pilot profile active in CI environment (gates may be relaxed)",
                severity="warning",
            ))
    
    def _check_gate_integration_imports(self) -> None:
        """Check that gate integration module is importable."""
        self.checks_performed.append("gate_integration_import_check")
        
        try:
            from ml.gate_integration import GateIntegrationPipeline
            from ml.unified_training import UnifiedTrainingPipeline
            from ml.coverage_contract import CoverageContractGate
            from ml.statistical_validity_gate import StatisticalValidityGate
        except ImportError as e:
            self.bypass_evidence.append(GateBypassEvidence(
                method=GateBypassMethod.MISSING_GATE_CHECK,
                source="gate_integration_imports",
                details=f"Failed to import gate integration modules: {e}",
                severity="error",
            ))


def validate_in_ci(strict: bool = True) -> bool:
    """
    Validate gates in CI environment.
    
    Args:
        strict: If True, treat warnings as errors
        
    Returns:
        True if validation passed, False otherwise
    """
    # Check if we're in CI
    is_ci = any(os.environ.get(var) for var in [
        'CI', 'GITHUB_ACTIONS', 'GITLAB_CI', 'CIRCLECI', 
        'JENKINS_URL', 'BUILDKITE'
    ])
    
    if not is_ci:
        logger.warning("Not running in CI environment, skipping gate validation")
        return True
    
    validator = CIGateValidator(strict=strict)
    result = validator.validate()
    
    # Print report
    print(result.format_report())
    
    # Write JSON report for CI artifacts
    report_path = Path("ci_gate_validation_report.json")
    report_path.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"\nReport written to: {report_path}")
    
    return result.passed


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="CI Gate Validator - Ensures ML gates are not bypassed"
    )
    parser.add_argument(
        "--strict", 
        action="store_true", 
        default=True,
        help="Treat warnings as errors (default: True)"
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Allow warnings (not recommended for CI)"
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default="ci_gate_validation_report.json",
        help="Path for JSON report output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    strict = not args.no_strict
    
    # Run validation
    validator = CIGateValidator(strict=strict)
    result = validator.validate()
    
    # Print report
    print(result.format_report())
    
    # Write JSON report
    report_path = Path(args.json_output)
    report_path.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"\nReport written to: {report_path}")
    
    # Exit with appropriate code
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
