from pathlib import Path

from ml.beta6.evaluation.shadow_compare import (
    build_shadow_compare_report,
    create_signed_shadow_compare_report,
    verify_shadow_compare_signature,
)


def test_shadow_compare_report_counts_explained_divergence():
    report = build_shadow_compare_report(
        run_id="r1",
        elder_id="HK001",
        room_rows=[
            {
                "room": "bedroom",
                "legacy_gate_pass": True,
                "legacy_gate_reasons": [],
                "beta6_gate_pass": False,
                "beta6_reason_code": "fail_timeline_mae",
                "beta6_details": {"timeline_hard_gate_passed": False},
            }
        ],
    )
    summary = report["summary"]
    assert summary["total_rooms"] == 1
    assert summary["divergence_count"] == 1
    assert summary["unexplained_divergence_count"] == 0
    assert summary["status"] == "watch"
    assert report["badges"][0]["reason_text"].startswith("Timeline duration error")


def test_shadow_compare_report_marks_unexplained_divergence_critical():
    report = build_shadow_compare_report(
        run_id="r2",
        elder_id="HK001",
        room_rows=[
            {
                "room": "livingroom",
                "legacy_gate_pass": True,
                "legacy_gate_reasons": [],
                "beta6_gate_pass": False,
                "beta6_reason_code": "",
                "beta6_details": {},
            }
        ],
        unexplained_divergence_rate_max=0.05,
    )
    summary = report["summary"]
    assert summary["divergence_count"] == 1
    assert summary["unexplained_divergence_count"] == 1
    assert summary["status"] == "critical"


def test_signed_shadow_compare_report_round_trip(tmp_path: Path):
    artifact_path = tmp_path / "shadow_compare.json"
    artifact = create_signed_shadow_compare_report(
        run_id="r3",
        elder_id="HK001",
        room_rows=[
            {
                "room": "bedroom",
                "legacy_gate_pass": True,
                "legacy_gate_reasons": [],
                "beta6_gate_pass": True,
                "beta6_reason_code": "pass",
                "beta6_details": {},
            }
        ],
        signing_key="shadow-key",
        output_path=artifact_path,
    )
    assert artifact_path.exists()
    assert artifact["signature"].startswith("sha256:")
    assert verify_shadow_compare_signature(artifact, signing_key="shadow-key") is True
