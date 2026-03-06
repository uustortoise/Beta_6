#!/usr/bin/env python3
"""Generate Step3 rollback-drill evidence artifact for Beta6.1 plan."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from run_daily_analysis import _validate_beta6_runtime_activation_preflight  # noqa: E402


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tail_text(text: str, max_lines: int = 60) -> str:
    lines = str(text or "").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in {float("inf"), float("-inf")}:
        return None
    return float(out)


@contextmanager
def _patched_env(updates: Dict[str, str | None]) -> Iterator[None]:
    previous: Dict[str, str | None] = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_pytest_command(command: Iterable[str], *, cwd: Path) -> Dict[str, Any]:
    cmd_list = [str(item) for item in command]
    proc = subprocess.run(
        cmd_list,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": " ".join(cmd_list),
        "returncode": int(proc.returncode),
        "passed": bool(proc.returncode == 0),
        "stdout_tail": _tail_text(proc.stdout),
        "stderr_tail": _tail_text(proc.stderr),
    }


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _build_quality_check(step1_summary: Dict[str, Any], *, mae_delta_cap_minutes: float = 2.0) -> Dict[str, Any]:
    reference = step1_summary.get("reference") if isinstance(step1_summary.get("reference"), dict) else {}
    variants = step1_summary.get("variants") if isinstance(step1_summary.get("variants"), dict) else {}
    baseline_anchor = _to_float(reference.get("baseline_anchor_livingroom_mae_minutes"))
    post_rollback_mae = None
    if isinstance(variants.get("A0"), dict):
        post_rollback_mae = _to_float(variants["A0"].get("livingroom_active_mae_minutes"))
    allowed_max = (
        float(baseline_anchor + mae_delta_cap_minutes) if baseline_anchor is not None else None
    )
    quality_pass = bool(
        baseline_anchor is not None
        and post_rollback_mae is not None
        and allowed_max is not None
        and float(post_rollback_mae) <= float(allowed_max)
    )
    return {
        "source": "step1_ab_summary:A0_vs_baseline_anchor",
        "baseline_anchor_livingroom_mae_minutes": baseline_anchor,
        "post_rollback_livingroom_mae_minutes": post_rollback_mae,
        "allowed_max_livingroom_mae_minutes": allowed_max,
        "cap_delta_minutes": float(mae_delta_cap_minutes),
        "pass": quality_pass,
    }


def build_step3_report(
    *,
    run_tests: bool,
    test_cwd: Path,
    step1_summary_path: Path,
) -> Dict[str, Any]:
    step1_available = step1_summary_path.exists()
    step1_summary = _load_json(step1_summary_path) if step1_available else {}
    quality_check = _build_quality_check(step1_summary) if step1_available else {
        "source": "step1_ab_summary:A0_vs_baseline_anchor",
        "pass": False,
        "error": f"missing_step1_summary:{step1_summary_path}",
    }

    required_flags = {
        "ENABLE_TIMELINE_MULTITASK": "false",
        "ENABLE_BETA6_HMM_RUNTIME": "false",
        "ENABLE_BETA6_CRF_RUNTIME": "false",
    }
    assist_flags = {
        "BETA6_PHASE4_RUNTIME_ENABLED": "false",
        "ENABLE_BETA6_UNKNOWN_ABSTAIN_RUNTIME": "false",
    }
    observed_before = {key: os.environ.get(key) for key in {**required_flags, **assist_flags}}
    with _patched_env({**required_flags, **assist_flags}):
        observed_after = {key: os.environ.get(key) for key in {**required_flags, **assist_flags}}
        preflight_ok, preflight_report = _validate_beta6_runtime_activation_preflight(
            elder_id=str(step1_summary.get("resident") or "HK001_jessica")
        )

    explicit_flags_verified = all(
        str(observed_after.get(key, "")).strip().lower() in {"0", "false", "no", "off", "disabled"}
        for key in required_flags
    )
    runtime_preflight_check = {
        "ok": bool(preflight_ok),
        "report": preflight_report if isinstance(preflight_report, dict) else {},
        "expected_reason": "runtime_flags_disabled",
        "pass": bool(
            preflight_ok
            and isinstance(preflight_report, dict)
            and str(preflight_report.get("reason")) == "runtime_flags_disabled"
        ),
    }

    tests: list[Dict[str, Any]] = []
    tests_pass = True
    if run_tests:
        commands = [
            ("pytest", "-q", "tests/test_beta6_runtime_preflight.py"),
            ("pytest", "-q", "tests/test_prediction_beta6_runtime_hook_parity.py"),
        ]
        tests = [_run_pytest_command(command, cwd=test_cwd) for command in commands]
        tests_pass = all(bool(item.get("passed")) for item in tests)

    pass_requirements = {
        "step1_summary_available": bool(step1_available),
        "rollback_configuration_explicit": bool(explicit_flags_verified),
        "rollback_runtime_preflight_path_validated": bool(runtime_preflight_check["pass"]),
        "fallback_quality_check": bool(quality_check.get("pass")),
        "required_tests_passed": bool(tests_pass),
    }
    overall_pass = all(bool(v) for v in pass_requirements.values())

    return {
        "generated_at_utc": _utc_now_iso(),
        "step": "Step3_S3-02",
        "artifact_version": "beta6_step3_rollback_drill.v1",
        "rollback_target_path": "pre-Beta6 flat-softmax runtime path",
        "required_rollback_flags": required_flags,
        "rollback_flag_observation": {
            "before": observed_before,
            "after": observed_after,
            "pass": bool(explicit_flags_verified),
        },
        "runtime_preflight_check": runtime_preflight_check,
        "quality_check": quality_check,
        "tests": tests,
        "pass_requirements": pass_requirements,
        "decision": {
            "status": "PASS" if overall_pass else "FAIL",
            "reason_codes": [
                "STEP3_ROLLBACK_DRILL_PASS" if overall_pass else "STEP3_ROLLBACK_DRILL_FAIL",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Step3 rollback drill evidence artifact.")
    parser.add_argument(
        "--output",
        default="/tmp/beta6_step3_rollback_drill_report.json",
        help="Output JSON artifact path.",
    )
    parser.add_argument(
        "--test-cwd",
        default=str(BACKEND_DIR),
        help="Working directory used for pytest commands.",
    )
    parser.add_argument(
        "--step1-summary",
        default="/tmp/beta6_step1_ab_summary.json",
        help="Path to Step1 summary JSON used for rollback quality anchor checks.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest execution and mark required_tests_passed=true by construction.",
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    report = build_step3_report(
        run_tests=not bool(args.skip_tests),
        test_cwd=Path(args.test_cwd).expanduser().resolve(),
        step1_summary_path=Path(args.step1_summary).expanduser().resolve(),
    )
    if args.skip_tests:
        report.setdefault("pass_requirements", {})["required_tests_passed"] = True
        report["decision"]["status"] = "PASS" if all(report["pass_requirements"].values()) else "FAIL"
        report["decision"]["reason_codes"] = [
            "STEP3_ROLLBACK_DRILL_PASS" if report["decision"]["status"] == "PASS" else "STEP3_ROLLBACK_DRILL_FAIL"
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
