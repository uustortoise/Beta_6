#!/usr/bin/env python3
"""Run Phase 0 Beta 6 label-pack intake gate (validate + diff + smoke)."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.beta6.intake_gate import (  # noqa: E402
    APPROVED_STATUS,
    INTAKE_ARTIFACT_VERSION,
    REJECTED_STATUS,
    validate_intake_artifact,
)
from scripts.diff_label_pack import diff_label_pack  # noqa: E402
from scripts.run_event_first_smoke import run_smoke  # noqa: E402
from scripts.validate_label_pack import validate_label_pack  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_intake(
    *,
    candidate_dir: Path,
    baseline_dir: Path,
    elder_id: str,
    min_day: int,
    max_day: int,
    smoke_day: int,
    seed: int,
    expectation_config: Optional[Path],
    registry_path: Optional[Path],
    output_dir: Path,
    train_context_days: int,
) -> Dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_json = (output_dir / "label_pack_validation.json").resolve()
    diff_json = (output_dir / "label_pack_diff.json").resolve()
    diff_csv = (output_dir / "label_pack_diff.csv").resolve()
    smoke_json = (output_dir / "label_pack_smoke.json").resolve()

    blocking_reasons: List[str] = []

    validate_step: Dict[str, Any] = {"status": "fail"}
    diff_step: Dict[str, Any] = {"status": "fail"}
    smoke_step: Dict[str, Any] = {"status": "skipped"}

    # Step 1: Validate label pack
    try:
        validate_report = validate_label_pack(
            pack_dir=candidate_dir,
            elder_id=elder_id,
            min_day=min_day,
            max_day=max_day,
            registry_path=registry_path,
        )
        validate_json.write_text(json.dumps(validate_report, indent=2))
        validate_status = str(validate_report.get("status", "fail")).strip().lower()
        validate_pass = validate_status == "pass"
        validate_step = {
            "status": "pass" if validate_pass else "fail",
            "violations_count": int(len(validate_report.get("violations", []))),
            "warnings_count": int(len(validate_report.get("warnings", []))),
            "report_path": str(validate_json),
        }
        if not validate_pass:
            blocking_reasons.append("validate_failed")
    except Exception as exc:
        validate_step = {
            "status": "fail",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback_tail": traceback.format_exc(limit=3).splitlines()[-1],
            "report_path": str(validate_json),
        }
        blocking_reasons.append(f"validate_exception:{type(exc).__name__}")

    # Step 2: Baseline-vs-candidate diff
    try:
        diff_report = diff_label_pack(
            baseline_dir=baseline_dir,
            candidate_dir=candidate_dir,
            elder_id=elder_id,
            min_day=min_day,
            max_day=max_day,
        )
        diff_json.write_text(json.dumps(diff_report, indent=2))
        rows = list(diff_report.get("rows", []))
        with diff_csv.open("w", encoding="utf-8") as handle:
            if rows:
                headers = list(rows[0].keys())
                handle.write(",".join(headers) + "\n")
                for row in rows:
                    values = []
                    for header in headers:
                        val = row.get(header, "")
                        if isinstance(val, dict):
                            values.append(json.dumps(val, sort_keys=True))
                        else:
                            values.append(str(val))
                    handle.write(",".join(values) + "\n")
            else:
                handle.write("day,room,windows_changed\n")

        summary = diff_report.get("summary", {}) if isinstance(diff_report, dict) else {}
        diff_step = {
            "status": "pass",
            "rows_compared": int(summary.get("rows_compared", 0) or 0),
            "days_compared": int(summary.get("days_compared", 0) or 0),
            "windows_changed_total": int(summary.get("windows_changed_total", 0) or 0),
            "report_path_json": str(diff_json),
            "report_path_csv": str(diff_csv),
        }
    except Exception as exc:
        diff_step = {
            "status": "fail",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback_tail": traceback.format_exc(limit=3).splitlines()[-1],
            "report_path_json": str(diff_json),
            "report_path_csv": str(diff_csv),
        }
        blocking_reasons.append(f"diff_exception:{type(exc).__name__}")

    # Step 3: Smoke check (skip when validation failed)
    if validate_step.get("status") != "pass":
        smoke_step = {
            "status": "skipped",
            "blocking_reasons": ["validation_failed"],
            "report_path": str(smoke_json),
        }
        blocking_reasons.append("smoke_skipped_due_validation_fail")
    else:
        try:
            smoke_report = run_smoke(
                data_dir=candidate_dir,
                elder_id=elder_id,
                day=smoke_day,
                seed=seed,
                expectation_config=expectation_config,
                diff_report=diff_json if diff_json.exists() else None,
                output=smoke_json,
                train_context_days=train_context_days,
            )
            smoke_json.write_text(json.dumps(smoke_report, indent=2))
            smoke_status = str(smoke_report.get("status", "fail")).strip().lower()
            smoke_blocking = [str(v) for v in smoke_report.get("blocking_reasons", [])]
            smoke_step = {
                "status": "pass" if smoke_status == "pass" else "fail",
                "blocking_reasons_count": int(len(smoke_blocking)),
                "blocking_reasons": smoke_blocking,
                "report_path": str(smoke_json),
            }
            if smoke_status != "pass":
                blocking_reasons.append("smoke_failed")
                for reason in smoke_blocking:
                    blocking_reasons.append(f"smoke:{reason}")
        except Exception as exc:
            smoke_step = {
                "status": "fail",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback_tail": traceback.format_exc(limit=3).splitlines()[-1],
                "report_path": str(smoke_json),
            }
            blocking_reasons.append(f"smoke_exception:{type(exc).__name__}")

    approved = len(blocking_reasons) == 0
    artifact: Dict[str, Any] = {
        "artifact_version": INTAKE_ARTIFACT_VERSION,
        "generated_at": _utc_now(),
        "status": APPROVED_STATUS if approved else REJECTED_STATUS,
        "pack": {
            "candidate_dir": str(candidate_dir.resolve()),
            "baseline_dir": str(baseline_dir.resolve()),
            "elder_id": str(elder_id),
            "min_day": int(min_day),
            "max_day": int(max_day),
            "smoke_day": int(smoke_day),
            "seed": int(seed),
        },
        "steps": {
            "validate": validate_step,
            "diff": diff_step,
            "smoke": smoke_step,
        },
        "reports": {
            "validate_json": str(validate_json),
            "diff_json": str(diff_json),
            "diff_csv": str(diff_csv),
            "smoke_json": str(smoke_json),
        },
        "gate": {
            "approved": bool(approved),
            "blocking_reasons": sorted(set(blocking_reasons)),
        },
    }
    return validate_intake_artifact(
        artifact,
        require_report_files=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Beta 6 label-pack intake gate")
    parser.add_argument("--candidate-dir", required=True, help="Candidate label-pack directory")
    parser.add_argument("--baseline-dir", required=True, help="Baseline label-pack directory")
    parser.add_argument("--elder-id", required=True)
    parser.add_argument("--min-day", type=int, required=True)
    parser.add_argument("--max-day", type=int, required=True)
    parser.add_argument("--smoke-day", type=int, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--expectation-config", default="backend/config/event_first_go_no_go.yaml")
    parser.add_argument("--registry-path", default="backend/config/adl_event_registry.v1.yaml")
    parser.add_argument("--train-context-days", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-output", default=None, help="Output path for intake artifact JSON")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_output = (
        Path(args.artifact_output).resolve()
        if args.artifact_output
        else (output_dir / "beta6_label_pack_intake_artifact.json").resolve()
    )

    artifact = run_intake(
        candidate_dir=Path(args.candidate_dir).resolve(),
        baseline_dir=Path(args.baseline_dir).resolve(),
        elder_id=str(args.elder_id),
        min_day=int(args.min_day),
        max_day=int(args.max_day),
        smoke_day=int(args.smoke_day),
        seed=int(args.seed),
        expectation_config=Path(args.expectation_config).resolve() if args.expectation_config else None,
        registry_path=Path(args.registry_path).resolve() if args.registry_path else None,
        output_dir=output_dir,
        train_context_days=max(int(args.train_context_days), 1),
    )

    artifact_output.parent.mkdir(parents=True, exist_ok=True)
    artifact_output.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote Beta 6 intake artifact: {artifact_output}")
    if not bool(artifact.get("gate", {}).get("approved", False)):
        print(f"Intake gate rejected: {artifact.get('gate', {}).get('blocking_reasons', [])}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
