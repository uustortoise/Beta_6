import json
from pathlib import Path

from ml.room_experiments import build_room_diagnostic_report
from scripts.run_room_experiments import _build_manifest_payload


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
