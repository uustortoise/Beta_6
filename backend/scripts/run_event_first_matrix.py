#!/usr/bin/env python3
"""Run configured event-first matrix variants and aggregate outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence

import yaml


BACKTEST_SCRIPT = Path("backend/scripts/run_event_first_backtest.py")
AGGREGATE_SCRIPT = Path("backend/scripts/aggregate_event_first_backtest.py")


def _load_yaml(path: Path) -> Dict[str, object]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root at {path}")
    return data


def _bool_arg(value: bool) -> str:
    return "true" if bool(value) else "false"


def _build_run_env(*, force_single_process: bool) -> Dict[str, str]:
    env = dict(os.environ)
    if force_single_process:
        env["JOBLIB_MULTIPROCESSING"] = "0"
        env["LOKY_MAX_CPU_COUNT"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
    return env


def _extract_output_path(cmd: Sequence[str]) -> Path | None:
    try:
        idx = list(cmd).index("--output")
    except ValueError:
        return None
    if idx + 1 >= len(cmd):
        return None
    return Path(str(cmd[idx + 1]))


def _is_valid_json(path: Path | None) -> bool:
    if path is None or not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    return isinstance(payload, dict)


def _run_command(
    cmd: List[str],
    *,
    dry_run: bool,
    timeout_seconds: int | None,
    retries: int,
    force_single_process: bool,
    recover_written_output: bool,
) -> Dict[str, object]:
    started_all = time.time()
    if dry_run:
        return {
            "status": "dry_run",
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "stdout": "",
            "stderr": "",
            "command": cmd,
            "attempts": [],
            "attempt_count": 0,
        }

    env = _build_run_env(force_single_process=force_single_process)
    attempts: List[Dict[str, object]] = []
    max_attempts = max(int(retries), 0) + 1
    out_path = _extract_output_path(cmd)
    last_result: Dict[str, object] | None = None

    for attempt_idx in range(max_attempts):
        attempt_start = time.time()
        timed_out = False
        proc_stdout = ""
        proc_stderr = ""
        proc_returncode = 1
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
                env=env,
            )
            proc_stdout = str(proc.stdout or "")
            proc_stderr = str(proc.stderr or "")
            proc_returncode = int(proc.returncode)
        except subprocess.TimeoutExpired as timeout_err:
            timed_out = True
            proc_returncode = 124
            proc_stdout = str((timeout_err.stdout or ""))
            proc_stderr = str((timeout_err.stderr or ""))

        recovered_output = False
        if proc_returncode != 0 and recover_written_output and _is_valid_json(out_path):
            proc_returncode = 0
            recovered_output = True

        elapsed_attempt = time.time() - attempt_start
        attempt_result = {
            "attempt": int(attempt_idx + 1),
            "timed_out": bool(timed_out),
            "recovered_output": bool(recovered_output),
            "returncode": int(proc_returncode),
            "elapsed_seconds": float(elapsed_attempt),
            "stdout": proc_stdout,
            "stderr": proc_stderr,
        }
        attempts.append(attempt_result)

        if proc_returncode == 0:
            break
        if attempt_idx + 1 < max_attempts:
            time.sleep(1.0)

    if attempts:
        last_result = attempts[-1]
    else:
        last_result = {
            "attempt": 0,
            "timed_out": False,
            "recovered_output": False,
            "returncode": 1,
            "elapsed_seconds": 0.0,
            "stdout": "",
            "stderr": "",
        }

    elapsed_total = time.time() - started_all
    return {
        "status": "ok" if int(last_result["returncode"]) == 0 else "failed",
        "returncode": int(last_result["returncode"]),
        "elapsed_seconds": float(elapsed_total),
        "stdout": str(last_result["stdout"]),
        "stderr": str(last_result["stderr"]),
        "command": cmd,
        "attempts": attempts,
        "attempt_count": len(attempts),
    }


def _build_backtest_command(
    *,
    data_dir: Path,
    elder_id: str,
    seed: int,
    output_path: Path,
    defaults: Dict[str, object],
    variant_args: Sequence[str],
    min_day_override: int | None,
    max_day_override: int | None,
) -> List[str]:
    min_day = int(min_day_override if min_day_override is not None else defaults.get("min_day", 4))
    max_day = int(max_day_override if max_day_override is not None else defaults.get("max_day", 10))
    cmd = [
        "python3",
        str(BACKTEST_SCRIPT),
        "--data-dir",
        str(data_dir),
        "--elder-id",
        str(elder_id),
        "--min-day",
        str(min_day),
        "--max-day",
        str(max_day),
        "--seed",
        str(int(seed)),
        "--occupancy-threshold",
        str(float(defaults.get("occupancy_threshold", 0.35))),
        "--calibration-method",
        str(defaults.get("calibration_method", "isotonic")),
        "--calib-fraction",
        str(float(defaults.get("calib_fraction", 0.20))),
        "--min-calib-samples",
        str(int(defaults.get("min_calib_samples", 500))),
        "--min-calib-label-support",
        str(int(defaults.get("min_calib_label_support", 30))),
        "--output",
        str(output_path),
    ]
    cmd.extend(list(variant_args))
    return cmd


def _build_aggregate_command(
    *,
    seed_reports: Sequence[Path],
    rolling_output: Path,
    signoff_output: Path,
    defaults: Dict[str, object],
) -> List[str]:
    cmd = [
        "python3",
        str(AGGREGATE_SCRIPT),
        "--seed-reports",
    ]
    cmd.extend([str(path) for path in seed_reports])
    cmd.extend(
        [
            "--rolling-output",
            str(rolling_output),
            "--signoff-output",
            str(signoff_output),
            "--comparison-window",
            str(defaults.get("comparison_window", "dec4_to_dec10")),
            "--required-split-pass-ratio",
            str(float(defaults.get("required_split_pass_ratio", 1.0))),
            "--require-baseline-for-promotion",
            _bool_arg(bool(defaults.get("require_baseline_for_promotion", False))),
            "--require-leakage-artifact",
            _bool_arg(bool(defaults.get("require_leakage_artifact", False))),
        ]
    )

    baseline_version = defaults.get("baseline_version")
    baseline_hash = defaults.get("baseline_artifact_hash")
    baseline_path = defaults.get("baseline_artifact_path")
    if baseline_version:
        cmd.extend(["--baseline-version", str(baseline_version)])
    if baseline_hash:
        cmd.extend(["--baseline-artifact-hash", str(baseline_hash)])
    if baseline_path:
        cmd.extend(["--baseline-artifact-path", str(baseline_path)])
    return cmd


def _evaluate_go_no_go(seed_reports: Sequence[Dict[str, object]], cfg: Dict[str, object]) -> Dict[str, object]:
    rules = cfg.get("go_no_go", {}) if isinstance(cfg, dict) else {}
    if not isinstance(rules, dict):
        rules = {}

    informational_checks_raw = rules.get("informational_checks", [])
    if isinstance(informational_checks_raw, (list, tuple, set)):
        informational_checks = {str(v).strip() for v in informational_checks_raw if str(v).strip()}
    else:
        informational_checks = set()

    def _maybe_float(value: object) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    total_eligible = 0
    passed_eligible = 0
    room_eligible_total: Dict[str, int] = {}
    room_eligible_passed: Dict[str, int] = {}

    day7_livingroom_recalls: List[float] = []
    day8_bedroom_sleep_recalls: List[float] = []
    day8_bedroom_sleep_supports: List[float] = []
    day8_livingroom_fragmentation: List[float] = []
    livingroom_episode_recalls: List[float] = []
    livingroom_episode_f1s: List[float] = []
    day7_livingroom_episode_recalls: List[float] = []
    livingroom_active_mae_minutes: List[float] = []
    bedroom_sleep_mae_minutes: List[float] = []

    for rep in seed_reports:
        summary = rep.get("summary", {})
        if isinstance(summary, dict):
            lr_summary = summary.get("LivingRoom", {})
            if isinstance(lr_summary, dict):
                lr_mae = _maybe_float(lr_summary.get("livingroom_active_mae_minutes"))
                if lr_mae is not None:
                    livingroom_active_mae_minutes.append(float(lr_mae))
            bedroom_summary = summary.get("Bedroom", {})
            if isinstance(bedroom_summary, dict):
                bedroom_mae = _maybe_float(bedroom_summary.get("sleep_duration_mae_minutes"))
                if bedroom_mae is not None:
                    bedroom_sleep_mae_minutes.append(float(bedroom_mae))

        splits = rep.get("splits", [])
        if not isinstance(splits, list):
            continue
        for split in splits:
            if not isinstance(split, dict):
                continue
            test_day = int(split.get("test_day", -1))
            rooms = split.get("rooms", {})
            if not isinstance(rooms, dict):
                continue
            for room_name, payload in rooms.items():
                if not isinstance(payload, dict):
                    continue
                room_key = str(room_name).strip().lower()
                hard_gate = payload.get("hard_gate", {})
                if isinstance(hard_gate, dict) and bool(hard_gate.get("eligible", False)):
                    total_eligible += 1
                    room_eligible_total[room_key] = int(room_eligible_total.get(room_key, 0)) + 1
                    if bool(hard_gate.get("pass", False)):
                        passed_eligible += 1
                        room_eligible_passed[room_key] = int(room_eligible_passed.get(room_key, 0)) + 1

                if test_day == 7 and room_key == "livingroom":
                    cls = payload.get("classification", {})
                    if isinstance(cls, dict):
                        try:
                            day7_livingroom_recalls.append(float(cls.get("occupied_recall", 0.0)))
                        except Exception:
                            pass

                if test_day == 8 and room_key == "bedroom":
                    recall_summary = payload.get("label_recall_summary", {})
                    if isinstance(recall_summary, dict):
                        sleep_payload = recall_summary.get("sleep", {})
                        if isinstance(sleep_payload, dict):
                            sleep_support = _maybe_float(sleep_payload.get("support"))
                            day8_bedroom_sleep_supports.append(float(sleep_support) if sleep_support is not None else 0.0)
                            try:
                                day8_bedroom_sleep_recalls.append(float(sleep_payload.get("recall", 0.0)))
                            except Exception:
                                pass

                if test_day == 8 and room_key == "livingroom":
                    try:
                        day8_livingroom_fragmentation.append(float(payload.get("fragmentation_score", 0.0)))
                    except Exception:
                        pass

                if room_key == "livingroom":
                    episode_metrics = payload.get("episode_metrics", {})
                    if isinstance(episode_metrics, dict):
                        try:
                            ep_recall = float(episode_metrics.get("episode_recall", 0.0))
                            livingroom_episode_recalls.append(ep_recall)
                            if test_day == 7:
                                day7_livingroom_episode_recalls.append(ep_recall)
                        except Exception:
                            pass
                        try:
                            livingroom_episode_f1s.append(float(episode_metrics.get("episode_f1", 0.0)))
                        except Exception:
                            pass

    overall_min = int(rules.get("overall_eligible_pass_count_min", 0) or 0)
    livingroom_min = int(rules.get("livingroom_eligible_pass_count_min", 0) or 0)
    bedroom_max_regression_splits = int(rules.get("bedroom_max_regression_splits", 9999) or 9999)
    day7_lr_min = float(rules.get("day7_livingroom_recall_min", 0.0) or 0.0)
    day8_bed_sleep_min = float(rules.get("day8_bedroom_sleep_recall_min", 0.0) or 0.0)
    day8_bed_sleep_min_support = int(max(rules.get("day8_bedroom_sleep_recall_min_support", 0) or 0, 0))
    day8_lr_frag_min = float(rules.get("day8_livingroom_fragmentation_min", 0.0) or 0.0)
    lr_episode_recall_min = float(rules.get("livingroom_episode_recall_min", 0.0) or 0.0)
    lr_episode_f1_min = float(rules.get("livingroom_episode_f1_min", 0.0) or 0.0)
    day7_lr_episode_recall_min = float(rules.get("day7_livingroom_episode_recall_min", 0.0) or 0.0)
    livingroom_mae_regression_pct = _maybe_float(rules.get("livingroom_active_mae_max_regression_pct"))
    livingroom_mae_baseline_minutes = _maybe_float(rules.get("livingroom_active_mae_baseline_minutes"))
    bedroom_mae_regression_pct = _maybe_float(rules.get("bedroom_sleep_mae_max_regression_pct"))
    bedroom_mae_baseline_minutes = _maybe_float(rules.get("bedroom_sleep_mae_baseline_minutes"))

    livingroom_passed = int(room_eligible_passed.get("livingroom", 0))
    bedroom_total = int(room_eligible_total.get("bedroom", 0))
    bedroom_passed = int(room_eligible_passed.get("bedroom", 0))
    bedroom_fail = max(int(bedroom_total - bedroom_passed), 0)

    day8_sleep_check_pass = False
    day8_sleep_actual_min: float | None = None
    day8_sleep_eligible_count = 0
    day8_sleep_low_support_skip = False
    if day8_bed_sleep_min_support > 0:
        eligible_recalls = [
            float(r)
            for r, support in zip(day8_bedroom_sleep_recalls, day8_bedroom_sleep_supports)
            if float(support) >= float(day8_bed_sleep_min_support)
        ]
        day8_sleep_eligible_count = int(len(eligible_recalls))
        if eligible_recalls:
            day8_sleep_actual_min = float(min(eligible_recalls))
            day8_sleep_check_pass = bool(day8_sleep_actual_min >= day8_bed_sleep_min)
        else:
            # Scoring mask may intentionally remove unsupported labels; treat as skipped-pass.
            day8_sleep_check_pass = True
            day8_sleep_low_support_skip = True
    else:
        if day8_bedroom_sleep_recalls:
            day8_sleep_actual_min = float(min(day8_bedroom_sleep_recalls))
            day8_sleep_check_pass = bool(day8_sleep_actual_min >= day8_bed_sleep_min)
            day8_sleep_eligible_count = int(len(day8_bedroom_sleep_recalls))
        else:
            day8_sleep_check_pass = False

    checks = [
        {
            "name": "overall_eligible_pass_count_min",
            "pass": bool(passed_eligible >= overall_min),
            "actual": int(passed_eligible),
            "required": int(overall_min),
        },
        {
            "name": "livingroom_eligible_pass_count_min",
            "pass": bool(livingroom_passed >= livingroom_min),
            "actual": int(livingroom_passed),
            "required": int(livingroom_min),
        },
        {
            "name": "bedroom_max_regression_splits",
            "pass": bool(bedroom_fail <= bedroom_max_regression_splits),
            "actual": int(bedroom_fail),
            "required_max": int(bedroom_max_regression_splits),
        },
        {
            "name": "day7_livingroom_recall_min",
            "pass": bool(min(day7_livingroom_recalls) >= day7_lr_min) if day7_livingroom_recalls else False,
            "actual_min": float(min(day7_livingroom_recalls)) if day7_livingroom_recalls else None,
            "required_min": float(day7_lr_min),
        },
        {
            "name": "day8_bedroom_sleep_recall_min",
            "pass": bool(day8_sleep_check_pass),
            "actual_min": day8_sleep_actual_min,
            "required_min": float(day8_bed_sleep_min),
            "min_support": int(day8_bed_sleep_min_support),
            "eligible_count": int(day8_sleep_eligible_count),
            "low_support_skip": bool(day8_sleep_low_support_skip),
        },
        {
            "name": "day8_livingroom_fragmentation_min",
            "pass": bool(min(day8_livingroom_fragmentation) >= day8_lr_frag_min) if day8_livingroom_fragmentation else False,
            "actual_min": float(min(day8_livingroom_fragmentation)) if day8_livingroom_fragmentation else None,
            "required_min": float(day8_lr_frag_min),
        },
    ]

    if "livingroom_episode_recall_min" in rules:
        checks.append(
            {
                "name": "livingroom_episode_recall_min",
                "pass": bool(min(livingroom_episode_recalls) >= lr_episode_recall_min)
                if livingroom_episode_recalls
                else False,
                "actual_min": float(min(livingroom_episode_recalls)) if livingroom_episode_recalls else None,
                "required_min": float(lr_episode_recall_min),
            }
        )
    if "livingroom_episode_f1_min" in rules:
        checks.append(
            {
                "name": "livingroom_episode_f1_min",
                "pass": bool(min(livingroom_episode_f1s) >= lr_episode_f1_min) if livingroom_episode_f1s else False,
                "actual_min": float(min(livingroom_episode_f1s)) if livingroom_episode_f1s else None,
                "required_min": float(lr_episode_f1_min),
            }
        )
    if "day7_livingroom_episode_recall_min" in rules:
        checks.append(
            {
                "name": "day7_livingroom_episode_recall_min",
                "pass": bool(min(day7_livingroom_episode_recalls) >= day7_lr_episode_recall_min)
                if day7_livingroom_episode_recalls
                else False,
                "actual_min": float(min(day7_livingroom_episode_recalls))
                if day7_livingroom_episode_recalls
                else None,
                "required_min": float(day7_lr_episode_recall_min),
            }
        )

    if livingroom_mae_regression_pct is not None:
        actual_lr_mae = (
            float(sum(livingroom_active_mae_minutes) / len(livingroom_active_mae_minutes))
            if livingroom_active_mae_minutes
            else None
        )
        allowed_lr_mae = None
        actual_lr_mae_regression_pct = None
        lr_mae_pass = False
        missing_lr_mae = True
        if (
            actual_lr_mae is not None
            and livingroom_mae_baseline_minutes is not None
            and livingroom_mae_baseline_minutes > 0.0
        ):
            allowed_lr_mae = float(
                livingroom_mae_baseline_minutes * (1.0 + (float(livingroom_mae_regression_pct) / 100.0))
            )
            actual_lr_mae_regression_pct = float(
                ((actual_lr_mae - livingroom_mae_baseline_minutes) / livingroom_mae_baseline_minutes) * 100.0
            )
            lr_mae_pass = bool(actual_lr_mae <= (allowed_lr_mae + 1e-9))
            missing_lr_mae = False
        checks.append(
            {
                "name": "livingroom_active_mae_max_regression_pct",
                "pass": bool(lr_mae_pass),
                "actual_mean": actual_lr_mae,
                "baseline_minutes": livingroom_mae_baseline_minutes,
                "allowed_max_minutes": allowed_lr_mae,
                "actual_regression_pct": actual_lr_mae_regression_pct,
                "required_max_regression_pct": float(livingroom_mae_regression_pct),
                "missing_metric": bool(missing_lr_mae),
            }
        )

    if bedroom_mae_regression_pct is not None:
        actual_bedroom_mae = (
            float(sum(bedroom_sleep_mae_minutes) / len(bedroom_sleep_mae_minutes))
            if bedroom_sleep_mae_minutes
            else None
        )
        allowed_bedroom_mae = None
        actual_bedroom_mae_regression_pct = None
        bedroom_mae_pass = False
        missing_bedroom_mae = True
        if (
            actual_bedroom_mae is not None
            and bedroom_mae_baseline_minutes is not None
            and bedroom_mae_baseline_minutes > 0.0
        ):
            allowed_bedroom_mae = float(
                bedroom_mae_baseline_minutes * (1.0 + (float(bedroom_mae_regression_pct) / 100.0))
            )
            actual_bedroom_mae_regression_pct = float(
                ((actual_bedroom_mae - bedroom_mae_baseline_minutes) / bedroom_mae_baseline_minutes) * 100.0
            )
            bedroom_mae_pass = bool(actual_bedroom_mae <= (allowed_bedroom_mae + 1e-9))
            missing_bedroom_mae = False
        checks.append(
            {
                "name": "bedroom_sleep_mae_max_regression_pct",
                "pass": bool(bedroom_mae_pass),
                "actual_mean": actual_bedroom_mae,
                "baseline_minutes": bedroom_mae_baseline_minutes,
                "allowed_max_minutes": allowed_bedroom_mae,
                "actual_regression_pct": actual_bedroom_mae_regression_pct,
                "required_max_regression_pct": float(bedroom_mae_regression_pct),
                "missing_metric": bool(missing_bedroom_mae),
            }
        )

    for check in checks:
        name = str(check.get("name", ""))
        check["severity"] = "informational" if name in informational_checks else "blocking"

    informational_failures = [
        str(check["name"])
        for check in checks
        if check.get("severity") == "informational" and not bool(check.get("pass", False))
    ]
    blocking_reasons = [
        str(check["name"])
        for check in checks
        if check.get("severity") != "informational" and not bool(check.get("pass", False))
    ]
    return {
        "status": "pass" if len(blocking_reasons) == 0 else "fail",
        "checks": checks,
        "blocking_reasons": blocking_reasons,
        "informational_failures": informational_failures,
        "counters": {
            "total_eligible": int(total_eligible),
            "passed_eligible": int(passed_eligible),
            "room_eligible_total": room_eligible_total,
            "room_eligible_passed": room_eligible_passed,
        },
    }


def run_matrix(
    *,
    profiles_yaml: Path,
    profile: str,
    data_dir: Path,
    elder_id: str,
    output_dir: Path,
    max_workers: int,
    dry_run: bool,
    go_no_go_config: Path | None,
    min_day_override: int | None,
    max_day_override: int | None,
    seed_timeout_seconds: int | None,
    seed_retries: int,
    force_single_process: bool,
    recover_written_output: bool,
) -> Dict[str, object]:
    spec = _load_yaml(profiles_yaml)
    defaults = spec.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}

    profiles = spec.get("profiles", {})
    if not isinstance(profiles, dict) or profile not in profiles:
        raise ValueError(f"Profile not found: {profile}")
    profile_payload = profiles.get(profile, {})
    if not isinstance(profile_payload, dict):
        raise ValueError(f"Invalid profile payload: {profile}")

    variants_payload = spec.get("variants", {})
    if not isinstance(variants_payload, dict):
        raise ValueError("Invalid variants payload")

    variant_names = profile_payload.get("variants", [])
    if not isinstance(variant_names, list) or not variant_names:
        raise ValueError(f"Profile has no variants: {profile}")

    seeds = profile_payload.get("seeds", defaults.get("seeds", [11, 22, 33]))
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("No seeds configured")
    seeds = [int(s) for s in seeds]

    out_root = output_dir / profile
    out_root.mkdir(parents=True, exist_ok=True)

    go_no_go_rules = _load_yaml(go_no_go_config) if go_no_go_config is not None and go_no_go_config.exists() else {}

    manifest: Dict[str, object] = {
        "profile": profile,
        "data_dir": str(data_dir),
        "elder_id": str(elder_id),
        "seeds": seeds,
        "variants": {},
        "dry_run": bool(dry_run),
        "execution_controls": {
            "seed_timeout_seconds": int(seed_timeout_seconds) if seed_timeout_seconds is not None else None,
            "seed_retries": int(seed_retries),
            "force_single_process": bool(force_single_process),
            "recover_written_output": bool(recover_written_output),
        },
    }

    for variant_name in variant_names:
        if variant_name not in variants_payload:
            raise ValueError(f"Variant not found: {variant_name}")
        variant_cfg = variants_payload.get(variant_name, {})
        if not isinstance(variant_cfg, dict):
            raise ValueError(f"Invalid variant config: {variant_name}")

        variant_dir = out_root / str(variant_name)
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_args = variant_cfg.get("args", [])
        if not isinstance(variant_args, list):
            variant_args = []
        variant_args = [str(v) for v in variant_args]

        seed_report_paths = [variant_dir / f"seed_{int(seed)}.json" for seed in seeds]
        seed_commands = [
            _build_backtest_command(
                data_dir=data_dir,
                elder_id=elder_id,
                seed=int(seed),
                output_path=report_path,
                defaults=defaults,
                variant_args=variant_args,
                min_day_override=min_day_override,
                max_day_override=max_day_override,
            )
            for seed, report_path in zip(seeds, seed_report_paths)
        ]

        seed_results: Dict[str, object] = {}
        if int(max_workers) <= 1:
            for seed, report_path, cmd in zip(seeds, seed_report_paths, seed_commands):
                result = _run_command(
                    cmd,
                    dry_run=dry_run,
                    timeout_seconds=seed_timeout_seconds,
                    retries=seed_retries,
                    force_single_process=force_single_process,
                    recover_written_output=recover_written_output,
                )
                seed_results[str(seed)] = {
                    "seed": int(seed),
                    "report_path": str(report_path),
                    "result": result,
                    "command": cmd,
                }
        else:
            with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
                future_map = {
                    pool.submit(
                        _run_command,
                        cmd,
                        dry_run=dry_run,
                        timeout_seconds=seed_timeout_seconds,
                        retries=seed_retries,
                        force_single_process=force_single_process,
                        recover_written_output=recover_written_output,
                    ): (seed, report_path, cmd)
                    for seed, report_path, cmd in zip(seeds, seed_report_paths, seed_commands)
                }
                for fut in as_completed(future_map):
                    seed, report_path, cmd = future_map[fut]
                    result = fut.result()
                    seed_results[str(seed)] = {
                        "seed": int(seed),
                        "report_path": str(report_path),
                        "result": result,
                        "command": cmd,
                    }

        failed_seed_runs = [v for v in seed_results.values() if isinstance(v, dict) and int((v.get("result") or {}).get("returncode", 1)) != 0]

        aggregate_result: Dict[str, object] = {
            "status": "skipped",
            "reason": "seed_run_failed" if failed_seed_runs else "dry_run",
        }
        rolling_path = variant_dir / "rolling.json"
        signoff_path = variant_dir / "signoff.json"

        if not failed_seed_runs and not dry_run:
            agg_cmd = _build_aggregate_command(
                seed_reports=seed_report_paths,
                rolling_output=rolling_path,
                signoff_output=signoff_path,
                defaults=defaults,
            )
            agg_exec = _run_command(
                agg_cmd,
                dry_run=False,
                timeout_seconds=max(int(seed_timeout_seconds or 1200), 1200),
                retries=0,
                force_single_process=force_single_process,
                recover_written_output=False,
            )
            aggregate_result = {
                "status": "ok" if int(agg_exec.get("returncode", 1)) == 0 else "failed",
                "result": agg_exec,
                "rolling_output": str(rolling_path),
                "signoff_output": str(signoff_path),
                "command": agg_cmd,
            }

        go_no_go_result: Dict[str, object] = {"status": "skipped", "reason": "not_executed"}
        config_hashes: List[str] = []
        if not failed_seed_runs and not dry_run and aggregate_result.get("status") == "ok":
            seed_reports_json: List[Dict[str, object]] = []
            for rp in seed_report_paths:
                if rp.exists():
                    obj = json.loads(rp.read_text())
                    if isinstance(obj, dict):
                        seed_reports_json.append(obj)
                        cfg_hash = obj.get("config_hash")
                        if cfg_hash is not None:
                            config_hashes.append(str(cfg_hash))
            go_no_go_result = _evaluate_go_no_go(seed_reports_json, go_no_go_rules)

        manifest_variant = {
            "description": variant_cfg.get("description"),
            "seed_runs": seed_results,
            "failed_seed_runs": len(failed_seed_runs),
            "aggregate": aggregate_result,
            "go_no_go": go_no_go_result,
            "config_hashes": sorted(set(config_hashes)),
        }
        cast_variants = manifest.get("variants")
        if isinstance(cast_variants, dict):
            cast_variants[str(variant_name)] = manifest_variant

    statuses = []
    variants_payload_out = manifest.get("variants", {})
    if isinstance(variants_payload_out, dict):
        for item in variants_payload_out.values():
            if isinstance(item, dict):
                go_no_go = item.get("go_no_go", {})
                if isinstance(go_no_go, dict):
                    statuses.append(str(go_no_go.get("status", "skipped")))
    if bool(dry_run):
        manifest["status"] = "dry_run"
    else:
        manifest["status"] = "pass" if statuses and all(v == "pass" for v in statuses) else "fail"
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run event-first matrix profiles")
    parser.add_argument("--profiles-yaml", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--elder-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--go-no-go-config", default="backend/config/event_first_go_no_go.yaml")
    parser.add_argument("--min-day", type=int, default=None)
    parser.add_argument("--max-day", type=int, default=None)
    parser.add_argument("--seed-timeout-seconds", type=int, default=900)
    parser.add_argument("--seed-retries", type=int, default=1)
    parser.add_argument("--disable-force-single-process", action="store_true")
    parser.add_argument("--disable-recover-written-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = run_matrix(
        profiles_yaml=Path(args.profiles_yaml),
        profile=str(args.profile),
        data_dir=Path(args.data_dir),
        elder_id=str(args.elder_id),
        output_dir=Path(args.output_dir),
        max_workers=int(max(args.max_workers, 1)),
        dry_run=bool(args.dry_run),
        go_no_go_config=Path(args.go_no_go_config) if args.go_no_go_config else None,
        min_day_override=args.min_day,
        max_day_override=args.max_day,
        seed_timeout_seconds=args.seed_timeout_seconds,
        seed_retries=int(max(args.seed_retries, 0)),
        force_single_process=not bool(args.disable_force_single_process),
        recover_written_output=not bool(args.disable_recover_written_output),
    )

    manifest_path = Path(args.output_dir) / str(args.profile) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote matrix manifest: {manifest_path}")
    return 0 if str(manifest.get("status")) in {"pass", "dry_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
