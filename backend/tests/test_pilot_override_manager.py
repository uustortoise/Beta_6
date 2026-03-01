from pathlib import Path

from ml.pilot_override_manager import PilotOverrideManager


def test_activate_baseline_fallback_rolls_back_and_audits(tmp_path: Path):
    state_file = tmp_path / "override_state.json"
    manager = PilotOverrideManager(state_file=state_file)

    success, _ = manager.activate_pilot(reason="canary test", duration_hours=1)
    assert success is True
    status_before = manager.get_status()
    assert status_before["active"] is True
    assert status_before["profile"] == "pilot"

    fallback_ok, event = manager.activate_baseline_fallback(reason="pipeline_reliability_breach")
    assert fallback_ok is True
    assert event["active_profile"] == "production"
    assert event["reason"] == "pipeline_reliability_breach"

    status_after = manager.get_status()
    assert status_after["active"] is False
    assert status_after["profile"] == "production"

    events = manager.read_fallback_audit_events()
    assert len(events) == 1
    assert events[0]["action"] == "baseline_fallback_activated"
