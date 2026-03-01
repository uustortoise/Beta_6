#!/usr/bin/env python3
"""Run LR-fragmentation sweep with clean-worker fallbacks and ranking outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import yaml

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.run_event_first_matrix import _evaluate_go_no_go  # noqa: E402


def _load_yaml(path: Path) -> Dict[str, object]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML root at {path}")
    return payload


def _valid_json(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    return isinstance(payload, dict)


def _run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    timeout_seconds: int | None,
    dry_run: bool,
) -> Dict[str, object]:
    if dry_run:
        return {
            "status": "dry_run",
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "elapsed_seconds": 0.0,
            "command": list(cmd),
            "timed_out": False,
        }

    started = time.time()
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
        return {
            "status": "ok" if int(proc.returncode) == 0 else "failed",
            "returncode": int(proc.returncode),
            "stdout": str(proc.stdout or ""),
            "stderr": str(proc.stderr or ""),
            "elapsed_seconds": float(time.time() - started),
            "command": list(cmd),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as timeout_err:
        return {
            "status": "timeout",
            "returncode": 124,
            "stdout": str(timeout_err.stdout or ""),
            "stderr": str(timeout_err.stderr or ""),
            "elapsed_seconds": float(time.time() - started),
            "command": list(cmd),
            "timed_out": True,
        }


def _single_process_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["JOBLIB_MULTIPROCESSING"] = "0"
    env["LOKY_MAX_CPU_COUNT"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def _write_ranking(
    variant_summaries: Dict[str, Dict[str, object]],
    *,
    csv_path: Path,
    md_path: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    for variant_name, payload in variant_summaries.items():
        go = payload.get("go_no_go", {})
        counters = go.get("counters", {}) if isinstance(go, dict) else {}
        room_pass = counters.get("room_eligible_passed", {}) if isinstance(counters, dict) else {}
        row = {
            "variant": variant_name,
            "status": str(go.get("status", "failed")),
            "eligible_passed": int(counters.get("passed_eligible", 0) or 0),
            "eligible_total": int(counters.get("total_eligible", 0) or 0),
            "livingroom_eligible_passed": int((room_pass or {}).get("livingroom", 0) or 0),
            "blocking_reason_count": len(go.get("blocking_reasons", []) or []),
            "blocking_reasons": ";".join(go.get("blocking_reasons", []) or []),
        }
        rows.append(row)

    rows.sort(
        key=lambda r: (
            0 if r["status"] == "pass" else 1,
            -int(r["eligible_passed"]),
            -int(r["livingroom_eligible_passed"]),
            int(r["blocking_reason_count"]),
            str(r["variant"]),
        )
    )

    header = [
        "variant",
        "status",
        "eligible_passed",
        "eligible_total",
        "livingroom_eligible_passed",
        "blocking_reason_count",
        "blocking_reasons",
    ]
    csv_lines = [",".join(header)]
    for row in rows:
        vals = [str(row[k]) for k in header]
        csv_lines.append(",".join(vals))
    csv_path.write_text("\n".join(csv_lines) + "\n")

    md_lines = [
        "# LR Fragmentation Sweep Ranking",
        "",
        "| variant | status | eligible | livingroom_eligible_passed | blockers |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['variant']} | {row['status']} | {row['eligible_passed']}/{row['eligible_total']} | "
            f"{row['livingroom_eligible_passed']} | {row['blocking_reasons']} |"
        )
    md_lines.append("")
    md_path.write_text("\n".join(md_lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean-worker LR fragmentation sweep")
    parser.add_argument("--profiles-yaml", default="backend/config/event_first_matrix_profiles.yaml")
    parser.add_argument("--profile", default="lr_fragmentation_sweep")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--elder-id", default="HK0011_jessica")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--go-no-go-config", default="backend/config/event_first_go_no_go.yaml")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--seed-timeout-seconds", type=int, default=300)
    parser.add_argument("--seed-retries", type=int, default=1)
    parser.add_argument("--matrix-timeout-seconds", type=int, default=1200)
    parser.add_argument("--cleanup-resource-trackers", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = BACKEND_ROOT.parent
    profiles_yaml = Path(args.profiles_yaml)
    if not profiles_yaml.is_absolute():
        profiles_yaml = repo / profiles_yaml
    go_no_go_cfg = Path(args.go_no_go_config)
    if not go_no_go_cfg.is_absolute():
        go_no_go_cfg = repo / go_no_go_cfg
    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = repo / out_root
    profile_out = out_root / str(args.profile)
    profile_out.mkdir(parents=True, exist_ok=True)

    spec = _load_yaml(profiles_yaml)
    defaults = spec.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}
    profiles = spec.get("profiles", {})
    if not isinstance(profiles, dict) or str(args.profile) not in profiles:
        raise ValueError(f"Profile not found: {args.profile}")
    profile_payload = profiles.get(str(args.profile), {})
    if not isinstance(profile_payload, dict):
        raise ValueError(f"Invalid profile payload: {args.profile}")
    variants = spec.get("variants", {})
    if not isinstance(variants, dict):
        raise ValueError("Invalid variants payload")

    seeds = profile_payload.get("seeds", defaults.get("seeds", [11, 22, 33]))
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("No seeds configured")
    seeds = [int(v) for v in seeds]
    variant_names = profile_payload.get("variants", [])
    if not isinstance(variant_names, list) or not variant_names:
        raise ValueError("Profile has no variants")
    variant_names = [str(v) for v in variant_names]

    env = _single_process_env()

    cleanup_result: Dict[str, object] = {"status": "skipped"}
    if bool(args.cleanup_resource_trackers):
        cleanup_result = _run_cmd(
            ["pkill", "-f", "resource_tracker"],
            cwd=repo,
            env=env,
            timeout_seconds=20,
            dry_run=bool(args.dry_run),
        )

    matrix_cmd = [
        "python3",
        "backend/scripts/run_event_first_matrix.py",
        "--profiles-yaml",
        str(profiles_yaml),
        "--profile",
        str(args.profile),
        "--data-dir",
        str(Path(args.data_dir)),
        "--elder-id",
        str(args.elder_id),
        "--output-dir",
        str(out_root),
        "--max-workers",
        str(max(int(args.max_workers), 1)),
        "--go-no-go-config",
        str(go_no_go_cfg),
        "--seed-timeout-seconds",
        str(max(int(args.seed_timeout_seconds), 1)),
        "--seed-retries",
        str(max(int(args.seed_retries), 0)),
    ]
    matrix_exec = _run_cmd(
        matrix_cmd,
        cwd=repo,
        env=env,
        timeout_seconds=max(int(args.matrix_timeout_seconds), 60),
        dry_run=bool(args.dry_run),
    )

    fallback_runs: List[Dict[str, object]] = []
    variant_summaries: Dict[str, Dict[str, object]] = {}
    go_rules = _load_yaml(go_no_go_cfg)

    for variant in variant_names:
        vdir = profile_out / variant
        vdir.mkdir(parents=True, exist_ok=True)
        missing: List[int] = []
        for seed in seeds:
            rp = vdir / f"seed_{seed}.json"
            if not _valid_json(rp):
                missing.append(seed)

        for seed in missing:
            out = vdir / f"seed_{seed}.json"
            rc = 1
            attempt_logs: List[Dict[str, object]] = []
            for attempt in range(1, max(int(args.seed_retries), 0) + 2):
                cmd = [
                    "python3",
                    "backend/scripts/run_event_first_variant_backtest.py",
                    "--profiles-yaml",
                    str(profiles_yaml),
                    "--variant",
                    str(variant),
                    "--data-dir",
                    str(Path(args.data_dir)),
                    "--elder-id",
                    str(args.elder_id),
                    "--seed",
                    str(seed),
                    "--output",
                    str(out),
                ]
                exec_result = _run_cmd(
                    cmd,
                    cwd=repo,
                    env=env,
                    timeout_seconds=max(int(args.seed_timeout_seconds), 1),
                    dry_run=bool(args.dry_run),
                )
                entry = {
                    "attempt": int(attempt),
                    "result": exec_result,
                }
                if int(exec_result.get("returncode", 1)) != 0 and _valid_json(out):
                    entry["result"]["status"] = "ok_recovered_output"
                    entry["result"]["returncode"] = 0
                attempt_logs.append(entry)
                rc = int(entry["result"].get("returncode", 1))
                if rc == 0:
                    break
            fallback_runs.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "output": str(out),
                    "attempts": attempt_logs,
                }
            )

        seed_reports: List[Path] = [vdir / f"seed_{s}.json" for s in seeds]
        valid_reports = [p for p in seed_reports if _valid_json(p)]
        summary: Dict[str, object] = {
            "seed_reports": [str(p) for p in seed_reports],
            "valid_seed_reports": [str(p) for p in valid_reports],
            "missing_seed_reports": [str(p) for p in seed_reports if p not in valid_reports],
        }

        if len(valid_reports) == len(seed_reports):
            rolling = vdir / "rolling.json"
            signoff = vdir / "signoff.json"
            agg_cmd = [
                "python3",
                "backend/scripts/aggregate_event_first_backtest.py",
                "--seed-reports",
                *[str(p) for p in seed_reports],
                "--rolling-output",
                str(rolling),
                "--signoff-output",
                str(signoff),
                "--comparison-window",
                str(defaults.get("comparison_window", "dec4_to_dec10")),
                "--required-split-pass-ratio",
                str(float(defaults.get("required_split_pass_ratio", 1.0))),
                "--require-baseline-for-promotion",
                "false",
                "--require-leakage-artifact",
                "false",
                "--baseline-version",
                str(defaults.get("baseline_version", "top2_frag_v3")),
            ]
            agg_exec = _run_cmd(
                agg_cmd,
                cwd=repo,
                env=env,
                timeout_seconds=max(int(args.matrix_timeout_seconds), 60),
                dry_run=bool(args.dry_run),
            )
            summary["aggregate"] = agg_exec
            if int(agg_exec.get("returncode", 1)) == 0 and not bool(args.dry_run):
                seed_payloads = [json.loads(path.read_text()) for path in seed_reports]
                summary["go_no_go"] = _evaluate_go_no_go(seed_payloads, go_rules)
            else:
                summary["go_no_go"] = {"status": "failed", "blocking_reasons": ["aggregate_failed"]}
        else:
            summary["aggregate"] = {"status": "skipped", "reason": "missing_seed_reports"}
            summary["go_no_go"] = {"status": "failed", "blocking_reasons": ["missing_seed_reports"]}

        variant_summaries[variant] = summary

    ranking_csv = profile_out / "ranking.csv"
    ranking_md = profile_out / "ranking.md"
    _write_ranking(variant_summaries, csv_path=ranking_csv, md_path=ranking_md)

    manifest = {
        "profile": str(args.profile),
        "data_dir": str(Path(args.data_dir)),
        "elder_id": str(args.elder_id),
        "seeds": seeds,
        "variants": variant_names,
        "cleanup_resource_trackers": cleanup_result,
        "matrix_exec": matrix_exec,
        "fallback_runs": fallback_runs,
        "variant_summaries": variant_summaries,
        "artifacts": {
            "ranking_csv": str(ranking_csv),
            "ranking_md": str(ranking_md),
        },
    }
    manifest_path = profile_out / "clean_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote clean sweep manifest: {manifest_path}")
    print(f"Wrote ranking csv: {ranking_csv}")
    print(f"Wrote ranking md: {ranking_md}")

    any_pass = any(
        str((payload.get("go_no_go") or {}).get("status", "failed")) == "pass"
        for payload in variant_summaries.values()
    )
    return 0 if any_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
