"""
Item 17: Ops Runbook + Automated Rollback of Pilot Overrides

Manages pilot profile activation with auto-expiry and rollback
to prevent temporary settings from persisting unnoticed.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import subprocess

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


def _parse_iso_datetime(value: str) -> datetime:
    """Parse ISO datetime and normalize naive values to UTC."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_ci_environment() -> bool:
    """
    PR-3.4: Detect if running in CI environment.
    
    Returns:
    --------
    bool
        True if running in a CI environment
    """
    ci_env_vars = [
        'CI',
        'GITHUB_ACTIONS',
        'GITLAB_CI',
        'CIRCLECI',
        'JENKINS_URL',
        'BUILDKITE',
        'TF_BUILD',  # Azure Pipelines
        'DRONE',
        'TRAVIS',
        'CODEBUILD_BUILD_ID',  # AWS CodeBuild
    ]
    return any(os.environ.get(var) for var in ci_env_vars)


class ProfileType(Enum):
    """Training profile types."""
    PILOT = "pilot"
    PRODUCTION = "production"


@dataclass
class OverrideState:
    """Tracks the state of a pilot override."""
    profile: str
    activated_at: str
    expires_at: Optional[str] = None
    activated_by: str = "unknown"
    reason: str = ""
    auto_rollback: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "activated_at": self.activated_at,
            "expires_at": self.expires_at,
            "activated_by": self.activated_by,
            "reason": self.reason,
            "auto_rollback": self.auto_rollback,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OverrideState":
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if override has expired."""
        if self.expires_at is None:
            return False
        return _utc_now() > _parse_iso_datetime(self.expires_at)


class PilotOverrideManager:
    """
    Manages pilot profile activation with auto-expiry.
    
    Prevents temporary settings from persisting unnoticed by:
    1. Tracking activation timestamp and expiry
    2. Providing rollback commands
    3. Showing post-run reminders
    4. Auto-rollback on expiry
    """
    
    DEFAULT_STATE_FILE = ".pilot_override_state.json"
    DEFAULT_FALLBACK_AUDIT_FILE = ".pilot_override_fallback_audit.jsonl"
    DEFAULT_EXPIRY_HOURS = 24  # Auto-expire after 24 hours
    
    def __init__(
        self,
        state_file: Optional[Union[str, Path]] = None,
        default_expiry_hours: int = DEFAULT_EXPIRY_HOURS,
    ):
        """
        Initialize manager.
        
        Parameters:
        -----------
        state_file : Path, optional
            Path to state file (default: ~/.pilot_override_state.json)
        default_expiry_hours : int
            Default expiry time for pilot overrides
        """
        if state_file is None:
            state_file = Path.home() / self.DEFAULT_STATE_FILE
        
        self.state_file = Path(state_file)
        self.fallback_audit_file = self.state_file.with_name(self.DEFAULT_FALLBACK_AUDIT_FILE)
        self.default_expiry_hours = default_expiry_hours
        self._state: Optional[OverrideState] = None
    
    def activate_pilot(
        self,
        reason: str,
        duration_hours: Optional[int] = None,
        auto_rollback: bool = True,
        ci_safe: bool = True,  # PR-3.4: CI-safe mode
    ) -> Tuple[bool, str]:
        """
        Activate pilot profile with tracking.
        
        Parameters:
        -----------
        reason : str
            Reason for pilot activation
        duration_hours : int, optional
            Override duration (default: 24 hours)
        auto_rollback : bool
            Whether to auto-rollback on expiry
        ci_safe : bool
            If True (default), block pilot activation in CI environments
            
        Returns:
        --------
        (success, message) : Tuple
        """
        # PR-3.4: CI-safe check
        if ci_safe and is_ci_environment():
            return False, (
                "❌ Pilot activation blocked in CI environment\n"
                "   CI environments should use production profile only.\n"
                "   To override (not recommended): set ci_safe=False"
            )
        
        duration = duration_hours or self.default_expiry_hours
        
        # Get current user
        try:
            user = subprocess.check_output(["whoami"], text=True).strip()
        except Exception:
            user = "unknown"
        
        # Calculate expiry
        activated_at = _utc_now()
        expires_at = activated_at + timedelta(hours=duration)
        
        # Create state
        self._state = OverrideState(
            profile=ProfileType.PILOT.value,
            activated_at=activated_at.isoformat(),
            expires_at=expires_at.isoformat(),
            activated_by=user,
            reason=reason,
            auto_rollback=auto_rollback,
        )
        
        # Save state
        self._save_state()
        
        # Set environment
        os.environ["TRAINING_PROFILE"] = ProfileType.PILOT.value
        
        message = (
            f"✓ Pilot profile activated\n"
            f"  Reason: {reason}\n"
            f"  Activated by: {user}\n"
            f"  Expires at: {expires_at.isoformat()} ({duration} hours)\n"
            f"  Auto-rollback: {'enabled' if auto_rollback else 'disabled'}\n"
            f"\n"
            f"To manually rollback:\n"
            f"  python -m ml.pilot_override_manager rollback\n"
        )
        
        logger.info(message)
        return True, message
    
    def rollback(self) -> Tuple[bool, str]:
        """
        Rollback to production profile.
        
        Returns:
        --------
        (success, message) : Tuple
        """
        # Clear state
        self._state = None
        if self.state_file.exists():
            self.state_file.unlink()
        
        # Set environment
        os.environ["TRAINING_PROFILE"] = ProfileType.PRODUCTION.value
        
        message = (
            f"✓ Rolled back to production profile\n"
            f"  TRAINING_PROFILE=production\n"
        )
        
        logger.info(message)
        return True, message

    def activate_baseline_fallback(
        self,
        *,
        reason: str,
        actor: str = "auto_rollout_guard",
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Force baseline profile and append an auditable fallback event.
        """
        previous_state = self.get_status()
        success, message = self.rollback()
        event = {
            "timestamp": _utc_now().isoformat(),
            "action": "baseline_fallback_activated",
            "actor": str(actor).strip() or "auto_rollout_guard",
            "reason": str(reason).strip() or "unspecified",
            "success": bool(success),
            "message": message,
            "previous_state": previous_state,
            "active_profile": ProfileType.PRODUCTION.value,
        }
        self.fallback_audit_file.parent.mkdir(parents=True, exist_ok=True)
        with self.fallback_audit_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")
        return success, event

    def read_fallback_audit_events(self) -> List[Dict[str, Any]]:
        """Read baseline fallback audit events in append order."""
        if not self.fallback_audit_file.exists():
            return []
        events: List[Dict[str, Any]] = []
        with self.fallback_audit_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    events.append(payload)
        return events
    
    def check_and_auto_rollback(self) -> Tuple[bool, str]:
        """
        Check for expired overrides and auto-rollback if needed.
        
        Returns:
        --------
        (did_rollback, message) : Tuple
        """
        state = self._load_state()
        
        if state is None:
            return False, "No active pilot override"
        
        if not state.is_expired():
            remaining = _parse_iso_datetime(state.expires_at) - _utc_now()
            return False, f"Pilot override active, expires in {remaining}"
        
        # Expired - check auto-rollback
        if not state.auto_rollback:
            return False, (
                f"⚠️ Pilot override EXPIRED but auto-rollback is disabled!\n"
                f"  Activated: {state.activated_at}\n"
                f"  Expired: {state.expires_at}\n"
                f"  Manual rollback required:\n"
                f"    python -m ml.pilot_override_manager rollback"
            )
        
        # Auto-rollback
        logger.warning(f"Auto-rolling back expired pilot override (activated: {state.activated_at})")
        return self.rollback()
    
    def get_reminder(self) -> Optional[str]:
        """
        Get reminder message if pilot is active.
        
        Returns:
        --------
        str or None
            Reminder message if pilot active, None otherwise
        """
        state = self._load_state()
        
        if state is None:
            return None
        
        if state.is_expired():
            return (
                f"⚠️ PILOT OVERRIDE EXPIRED ⚠️\n"
                f"  Activated: {state.activated_at}\n"
                f"  Expired: {state.expires_at}\n"
                f"  Reason: {state.reason}\n"
                f"\n"
                f"Run rollback command:\n"
                f"  python -m ml.pilot_override_manager rollback"
            )
        
        remaining = _parse_iso_datetime(state.expires_at) - _utc_now()
        
        return (
            f"🔔 PILOT MODE ACTIVE\n"
            f"  Profile: {state.profile}\n"
            f"  Reason: {state.reason}\n"
            f"  Expires in: {remaining}\n"
            f"\n"
            f"To rollback now:\n"
            f"  python -m ml.pilot_override_manager rollback"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current override status."""
        state = self._load_state()
        
        if state is None:
            return {
                "active": False,
                "profile": ProfileType.PRODUCTION.value,
                "message": "No active pilot override",
            }
        
        return {
            "active": True,
            "profile": state.profile,
            "activated_at": state.activated_at,
            "expires_at": state.expires_at,
            "is_expired": state.is_expired(),
            "activated_by": state.activated_by,
            "reason": state.reason,
            "auto_rollback": state.auto_rollback,
        }
    
    def _load_state(self) -> Optional[OverrideState]:
        """Load state from file."""
        if self._state is not None:
            return self._state
        
        if not self.state_file.exists():
            return None
        
        try:
            data = json.loads(self.state_file.read_text())
            self._state = OverrideState.from_dict(data)
            return self._state
        except Exception as e:
            logger.warning(f"Failed to load override state: {e}")
            return None
    
    def _save_state(self) -> None:
        """Save state to file."""
        if self._state is None:
            return
        
        self.state_file.write_text(json.dumps(self._state.to_dict(), indent=2))


# CLI functions
def set_training_profile(
    profile: str,
    reason: Optional[str] = None,
    duration_hours: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    CLI function to set training profile.
    
    Usage:
        python -c "from ml.pilot_override_manager import set_training_profile; set_training_profile('pilot', 'Testing', 4)"
    
    Or as a script:
        python -m ml.pilot_override_manager pilot --reason "Testing" --duration 4
    """
    manager = PilotOverrideManager()
    
    if profile.lower() == ProfileType.PILOT.value:
        if not reason:
            return False, "Reason required for pilot profile activation"
        return manager.activate_pilot(reason, duration_hours)
    elif profile.lower() == ProfileType.PRODUCTION.value:
        return manager.rollback()
    else:
        return False, f"Unknown profile: {profile}"


def show_status() -> str:
    """Show current override status."""
    manager = PilotOverrideManager()
    status = manager.get_status()
    
    if status["active"]:
        return (
            f"Profile: {status['profile'].upper()}\n"
            f"Activated: {status['activated_at']}\n"
            f"Expires: {status['expires_at']}\n"
            f"Expired: {status['is_expired']}\n"
            f"By: {status['activated_by']}\n"
            f"Reason: {status['reason']}\n"
        )
    else:
        return f"Profile: {status['profile'].upper()}\n{status['message']}\n"


def post_run_reminder() -> Optional[str]:
    """
    Show post-run reminder if pilot is active.
    
    Call this after training run completes.
    """
    manager = PilotOverrideManager()
    reminder = manager.get_reminder()
    
    # Also check for auto-rollback
    did_rollback, message = manager.check_and_auto_rollback()
    if did_rollback:
        return f"🔄 {message}"
    
    return reminder


# Shell script generation for ops
def generate_set_training_profile_script() -> str:
    """
    Generate the set_training_profile.sh script content.
    
    Returns:
    --------
    str
        Shell script content
    """
    return '''#!/bin/bash
#
# set_training_profile.sh
# 
# Sets the training profile with proper tracking and auto-expiry.
#
# Usage:
#   ./set_training_profile.sh pilot --reason "Testing new features" --duration 4
#   ./set_training_profile.sh production
#

set -e

PROFILE=${1:-production}
REASON=""
DURATION=24  # Default 24 hours

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --reason)
            REASON="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ "$PROFILE" = "pilot" ]; then
    if [ -z "$REASON" ]; then
        echo "Error: --reason required for pilot profile"
        echo "Usage: $0 pilot --reason 'Testing' --duration 4"
        exit 1
    fi
    
    python3 -c "
from ml.pilot_override_manager import set_training_profile
success, message = set_training_profile('$PROFILE', '$REASON', $DURATION)
print(message)
exit(0 if success else 1)
"
elif [ "$PROFILE" = "production" ]; then
    python3 -c "
from ml.pilot_override_manager import set_training_profile
success, message = set_training_profile('$PROFILE')
print(message)
exit(0 if success else 1)
"
else
    echo "Unknown profile: $PROFILE"
    echo "Usage: $0 {pilot|production}"
    exit 1
fi
'''


def install_scripts(install_dir: Union[str, Path] = "/usr/local/bin") -> None:
    """
    Install helper scripts for ops.
    
    Parameters:
    -----------
    install_dir : Path
        Directory to install scripts
    """
    install_path = Path(install_dir)
    install_path.mkdir(parents=True, exist_ok=True)
    
    # Install set_training_profile.sh
    script_path = install_path / "set_training_profile.sh"
    script_path.write_text(generate_set_training_profile_script())
    script_path.chmod(0o755)
    
    logger.info(f"Installed scripts to {install_dir}:")
    logger.info(f"  - {script_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage training profile overrides")
    parser.add_argument("profile", choices=["pilot", "production"], help="Profile to activate")
    parser.add_argument("--reason", help="Reason for pilot activation")
    parser.add_argument("--duration", type=int, default=24, help="Duration in hours (default: 24)")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--install-scripts", help="Install helper scripts to directory")
    
    args = parser.parse_args()
    
    if args.install_scripts:
        install_scripts(args.install_scripts)
    elif args.status:
        print(show_status())
    else:
        success, message = set_training_profile(
            args.profile, args.reason, args.duration
        )
        print(message)
        exit(0 if success else 1)
