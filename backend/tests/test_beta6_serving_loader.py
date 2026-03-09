from datetime import date, timedelta
from pathlib import Path

from ml.t80_rollout_manager import RolloutStage
from ml.beta6.serving import serving_loader


class _RolloutState:
    def __init__(self, stage: RolloutStage):
        self.stage = stage


class _Manager:
    def __init__(self, stage: RolloutStage):
        self._state = _RolloutState(stage)

    def get_state(self):
        return self._state


def test_stability_certification_reaches_ready_after_required_days(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(serving_loader, "T80RolloutManager", lambda: _Manager(RolloutStage.CANARY))
    start = date(2026, 2, 1)
    last = None
    for offset in range(14):
        run = serving_loader.run_daily_stability_certification(
            elder_id="HK001",
            run_id=f"run-{offset}",
            beta6_gate_pass=True,
            beta6_gate_report={"reason_code": "pass"},
            beta6_fallback_summary={"activated": [], "cleared": [], "errors": []},
            pipeline_success_rate=1.0,
            certification_date=(start + timedelta(days=offset)).isoformat(),
            state_root=tmp_path,
        )
        last = run

    assert last is not None
    assert last.stable_today is True
    assert last.consecutive_stable_days == 14
    assert last.certification_ready is True
    assert last.rollout_stage == "canary"
    assert last.active_system == "beta5.5_authority"


def test_stability_certification_resets_on_blocker(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(serving_loader, "T80RolloutManager", lambda: _Manager(RolloutStage.FULL))

    ok = serving_loader.run_daily_stability_certification(
        elder_id="HK009",
        run_id="run-ok",
        beta6_gate_pass=True,
        beta6_gate_report={"reason_code": "pass"},
        beta6_fallback_summary={"activated": [], "cleared": [], "errors": []},
        pipeline_success_rate=1.0,
        certification_date="2026-02-20",
        state_root=tmp_path,
    )
    assert ok.consecutive_stable_days == 1
    assert ok.active_system == "beta6_authority"

    bad = serving_loader.run_daily_stability_certification(
        elder_id="HK009",
        run_id="run-bad",
        beta6_gate_pass=False,
        beta6_gate_report={"reason_code": "fail_gate_policy"},
        beta6_fallback_summary={"activated": ["bedroom"], "cleared": [], "errors": []},
        pipeline_success_rate=0.5,
        certification_date="2026-02-21",
        state_root=tmp_path,
    )
    assert bad.stable_today is False
    assert bad.consecutive_stable_days == 0
    assert "beta6_authority_gate_failed" in bad.blockers
    assert "beta6_fallback_active" in bad.blockers


def test_serving_loader_uses_deterministic_fallback_resolution(tmp_path: Path):
    registry = serving_loader.RegistryV2(root=tmp_path / "registry")
    elder_id = "HK001"
    room = "livingroom"

    registry.promote_candidate(
        elder_id=elder_id,
        room=room,
        run_id="run-1",
        candidate_id="cand-v1",
    )
    registry.promote_candidate(
        elder_id=elder_id,
        room=room,
        run_id="run-2",
        candidate_id="cand-v2",
    )
    registry.activate_fallback_mode(
        elder_id=elder_id,
        room=room,
        run_id="run-3",
        trigger_reason_code="pipeline_reliability_breach",
    )

    pointer = serving_loader.resolve_serving_pointer_for_room(
        registry_v2=registry,
        elder_id=elder_id,
        room=room,
    )

    assert pointer is not None
    assert pointer["candidate_id"] == "cand-v1"
