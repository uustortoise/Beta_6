import json
import os
from pathlib import Path
from unittest.mock import patch

from ml.room_experiments import (
    build_candidate_execution_plan,
    build_grouped_regime_fragility_report,
    build_room_diagnostic_report,
)
from scripts.run_room_experiments import (
    _build_manifest_payload,
    _resolve_candidate_execution_plan,
    _resolve_profile_typed_policy_values,
)


def test_livingroom_replay_candidate_is_traceable_to_typed_policy_fields():
    report = build_room_diagnostic_report(
        room_name="LivingRoom",
        profile_name="livingroom_policy_sensitivity",
        profile_payload={
            "room": "livingroom",
            "typed_policy_fields": [
                "two_stage_core.gate_mode",
                "training_profile.post_split_shuffle_rooms",
            ],
            "env_overrides": {
                "TWO_STAGE_CORE_GATE_MODE": "shadow",
            },
        },
        typed_policy_values={
            "two_stage_core.gate_mode": "primary",
            "training_profile.post_split_shuffle_rooms": ["entrance", "bedroom"],
        },
    )

    assert report["room"] == "livingroom"
    assert report["profile_name"] == "livingroom_policy_sensitivity"
    assert report["typed_policy_values"]["two_stage_core.gate_mode"] == "primary"
    assert report["env_overrides"]["TWO_STAGE_CORE_GATE_MODE"] == "shadow"


def test_bedroom_grouped_date_fragility_is_persisted_in_review_surface(tmp_path: Path):
    grouped_fragility = {
        "grouped_by_date": {
            "worst_slice": "2026-03-07",
            "worst_slice_macro_f1": 0.12,
            "slice_count": 7,
        },
        "lineage": {
            "run_id": "beta6_daily_HK0011_jessica_20260307T040810Z",
            "candidate_version": 29,
        },
    }

    report = build_room_diagnostic_report(
        room_name="Bedroom",
        profile_name="bedroom_grouped_fragility",
        profile_payload={
            "room": "bedroom",
            "grouped_regime": "grouped_by_date",
            "typed_policy_fields": ["training_profile.transition_focus_room_labels"],
        },
        typed_policy_values={
            "training_profile.transition_focus_room_labels": {"bedroom": "bedroom_normal_use"},
        },
        grouped_fragility=grouped_fragility,
    )

    manifest_payload = _build_manifest_payload(
        profile_name="bedroom_grouped_fragility",
        room_name="Bedroom",
        report=report,
    )

    output = tmp_path / "report.json"
    output.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    reloaded = json.loads(output.read_text(encoding="utf-8"))

    assert reloaded["report"]["fragility"]["grouped_by_date"]["worst_slice"] == "2026-03-07"
    assert reloaded["report"]["fragility"]["lineage"]["candidate_version"] == 29


def test_profile_env_overrides_are_applied_to_typed_policy_values():
    profile = {
        "typed_policy_fields": ["two_stage_core.gate_mode"],
        "env_overrides": {"TWO_STAGE_CORE_GATE_MODE": "shadow"},
    }
    with patch.dict(os.environ, {"TWO_STAGE_CORE_GATE_MODE": "primary"}, clear=False):
        typed_values = _resolve_profile_typed_policy_values(profile)
    assert typed_values["two_stage_core.gate_mode"] == "shadow"


def test_grouped_by_date_summary_emits_worst_date_and_range():
    grouped = build_grouped_regime_fragility_report(
        grouped_by_date_slices=[
            {"date": "2026-03-05", "macro_f1": 0.61},
            {"date": "2026-03-06", "macro_f1": 0.44},
            {"date": "2026-03-07", "macro_f1": 0.12},
        ]
    )

    summary = grouped["grouped_by_date"]
    assert summary["worst_slice"] == "2026-03-07"
    assert summary["worst_slice_macro_f1"] == 0.12
    assert summary["best_slice"] == "2026-03-05"
    assert summary["range_macro_f1"] == 0.49


def test_grouped_by_user_summary_emits_worst_user_and_range():
    grouped = build_grouped_regime_fragility_report(
        grouped_by_user_slices=[
            {"user": "HK0011_jessica", "macro_f1": 0.71},
            {"user": "HK0012_sam", "macro_f1": 0.27},
            {"user": "HK0013_mary", "macro_f1": 0.63},
        ]
    )

    summary = grouped["grouped_by_user"]
    assert summary["worst_slice"] == "HK0012_sam"
    assert summary["worst_slice_macro_f1"] == 0.27
    assert summary["best_slice"] == "HK0011_jessica"
    assert summary["range_macro_f1"] == 0.44


def test_room_experiments_can_report_grouped_regime_stability():
    grouped = build_grouped_regime_fragility_report(
        grouped_by_date_slices=[
            {"date": "2026-03-05", "macro_f1": 0.61},
            {"date": "2026-03-07", "macro_f1": 0.12},
        ],
        grouped_by_user_slices=[
            {"user": "HK0011_jessica", "macro_f1": 0.45},
            {"user": "HK0012_sam", "macro_f1": 0.41},
        ],
        fragile_room_floor=0.20,
    )

    report = build_room_diagnostic_report(
        room_name="Bedroom",
        profile_name="bedroom_grouped_fragility",
        profile_payload={"room": "bedroom", "grouped_regime": "grouped_by_date"},
        typed_policy_values={},
        grouped_fragility=grouped,
    )

    assert report["fragility"]["stability_gate"]["pass"] is False
    assert report["fragility"]["stability_gate"]["failures"][0]["regime"] == "grouped_by_date"
    assert report["fragility"]["stability_gate"]["failures"][0]["worst_slice"] == "2026-03-07"


def test_bad_candidates_can_be_eliminated_early_without_affecting_good_runs():
    plan = build_candidate_execution_plan(
        candidates=[
            {
                "candidate_name": "known_bad",
                "early_blockers": ["collapse_detected"],
            },
            {
                "candidate_name": "good_shadow",
                "typed_policy_values": {"two_stage_core.gate_mode": "shadow"},
            },
        ],
        fast_replay=False,
    )

    assert plan["selected"][0]["candidate_name"] == "good_shadow"
    assert plan["selected"][0]["execution_mode"] == "full_retrain"
    assert plan["rejected"][0]["candidate_name"] == "known_bad"
    assert plan["rejected"][0]["rejection_reasons"] == ["collapse_detected"]


def test_room_policy_sweeps_can_run_as_fast_replay_diagnostics():
    plan = _resolve_candidate_execution_plan(
        raw_candidate_plan=json.dumps(
            [
                {
                    "candidate_name": "shadow_gate",
                    "typed_policy_values": {"two_stage_core.gate_mode": "shadow"},
                }
            ]
        ),
        fast_replay=True,
    )

    assert plan["selected"][0]["candidate_name"] == "shadow_gate"
    assert plan["selected"][0]["execution_mode"] == "fast_replay"
    assert plan["rejected"] == []
