from pathlib import Path

import scripts.run_bedroom_root_cause_matrix as matrix
from scripts.run_bedroom_root_cause_matrix import build_variant_specs, run_matrix


def _source_files(tmp_path: Path) -> dict[str, str]:
    return {
        "2025-12-04": str(tmp_path / "HK0011_jessica_train_4dec2025.xlsx"),
        "2025-12-05": str(tmp_path / "HK0011_jessica_train_5dec2025.xlsx"),
        "2025-12-06": str(tmp_path / "HK0011_jessica_train_6dec2025.xlsx"),
        "2025-12-07": str(tmp_path / "HK0011_jessica_train_7dec2025.xlsx"),
        "2025-12-08": str(tmp_path / "HK0011_jessica_train_8dec2025.xlsx"),
        "2025-12-09": str(tmp_path / "HK0011_jessica_train_9dec2025.xlsx"),
        "2025-12-10": str(tmp_path / "HK0011_jessica_train_10dec2025.xlsx"),
        "2025-12-17": str(tmp_path / "HK0011_jessica_train_17dec2025.xlsx"),
    }


def test_build_variant_specs_includes_anchor_single_day_and_cumulative_ladders(tmp_path: Path):
    variants = build_variant_specs(source_files_by_date=_source_files(tmp_path))
    by_name = {variant["name"]: variant for variant in variants}

    assert by_name["anchor"]["included_dates"] == ["2025-12-10", "2025-12-17"]
    assert by_name["add_2025-12-04"]["included_dates"] == ["2025-12-04", "2025-12-10", "2025-12-17"]
    assert by_name["cumulative_through_2025-12-06"]["included_dates"] == [
        "2025-12-04",
        "2025-12-05",
        "2025-12-06",
        "2025-12-10",
        "2025-12-17",
    ]
    assert by_name["cumulative_through_2025-12-09"]["source_files"][-1].endswith(
        "HK0011_jessica_train_17dec2025.xlsx"
    )


def test_run_matrix_dry_run_builds_manifest_without_execution(tmp_path: Path):
    manifest = run_matrix(
        output_dir=tmp_path / "matrix",
        dry_run=True,
        source_namespace="HK0011_jessica",
        source_files_by_date=_source_files(tmp_path),
        eval_file=str(tmp_path / "HK0011_jessica_train_17dec2025.xlsx"),
        candidate_prefix="HK0011_jessica_candidate_matrix_test",
    )

    assert manifest["status"] == "dry_run"
    assert manifest["execution_status"] == "dry_run"
    assert manifest["gate_status"] == "not_run"
    assert manifest["dry_run"] is True
    assert manifest["room"] == "Bedroom"
    assert "factorized-primary training still fits two-stage core submodels" in manifest["execution_notes"][0]
    assert "anchor" in manifest["variants"]
    assert "add_2025-12-04" in manifest["variants"]
    assert "cumulative_through_2025-12-09" in manifest["variants"]

    anchor = manifest["variants"]["anchor"]
    assert anchor["status"] == "planned"
    assert anchor["candidate_namespace"].startswith("HK0011_jessica_candidate_matrix_test_01_")
    assert anchor["expected_artifacts"]["train_metrics_path"].endswith("anchor/train_metrics.json")


def test_extract_holdout_bedroom_normal_use_recall_uses_best_summary_fallback():
    recall = matrix._extract_holdout_bedroom_normal_use_recall(
        {
            "per_label_recall": {
                "bedroom_normal_use": 0.5580110497237569,
            },
            "checkpoint_selection": {
                "best_summary": {
                    "per_label_recall": {
                        "bedroom_normal_use": 0.427255985267035,
                    }
                }
            }
        }
    )

    assert recall == 0.427255985267035


def test_run_matrix_reports_gate_failures_without_hiding_execution(monkeypatch, tmp_path: Path):
    def fake_run_variant(**kwargs):
        variant = kwargs["variant"]
        gate_pass = str(variant["name"]) == "anchor"
        return {
            "status": "ok",
            "variant": str(variant["name"]),
            "mode": str(variant["mode"]),
            "candidate_namespace": kwargs["candidate_namespace"],
            "included_dates": list(variant["included_dates"]),
            "source_files": list(variant["source_files"]),
            "gate_pass": gate_pass,
            "holdout_bedroom_normal_use_recall": 0.5,
        }

    monkeypatch.setattr(matrix, "_run_variant", fake_run_variant)

    manifest = run_matrix(
        output_dir=tmp_path / "matrix",
        dry_run=False,
        source_namespace="HK0011_jessica",
        source_files_by_date=_source_files(tmp_path),
        eval_file=str(tmp_path / "HK0011_jessica_train_17dec2025.xlsx"),
        candidate_prefix="HK0011_jessica_candidate_matrix_test",
        variant_names=["anchor", "add_2025-12-05"],
    )

    assert manifest["status"] == "gate_fail"
    assert manifest["execution_status"] == "ok"
    assert manifest["gate_status"] == "fail"
    assert manifest["variant_execution_counts"] == {"ok": 2}
    assert manifest["variant_gate_counts"] == {"fail": 1, "pass": 1}


def test_extract_variant_review_surface_carries_grouped_date_instability():
    trained_room = {
        "room": "Bedroom",
        "gate_pass": False,
        "promotion_risk_flags": ["unstable_date_slices:bedroom"],
        "gate_watch_reasons": [
            "grouped_date_stability_watch:bedroom:worst_date=2025-12-17:macro_range=0.218"
        ],
        "grouped_date_stability": {
            "available": True,
            "unstable_across_dates": True,
            "worst_date": "2025-12-17",
        },
        "promotion_time_drift_summary": {
            "available": True,
            "risk_level": "high",
            "unstable_across_dates": True,
        },
    }

    surface = matrix._extract_variant_review_surface(trained_room)

    assert surface["promotion_risk_flags"] == ["unstable_date_slices:bedroom"]
    assert surface["gate_watch_reasons"] == [
        "grouped_date_stability_watch:bedroom:worst_date=2025-12-17:macro_range=0.218"
    ]
    assert surface["grouped_date_stability"]["unstable_across_dates"] is True
    assert surface["promotion_time_drift_summary"]["risk_level"] == "high"
