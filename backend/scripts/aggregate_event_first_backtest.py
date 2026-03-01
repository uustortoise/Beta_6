#!/usr/bin/env python3
"""Aggregate seeded event-first backtests into rolling + signoff artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="ignore").strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _safe_mean_std(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(len(arr)),
    }


def _collect_room_metric_values(reports: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for rep in reports:
        summary = rep.get("summary") or {}
        if not isinstance(summary, dict):
            continue
        for room, metrics in summary.items():
            if not isinstance(metrics, dict):
                continue
            room_entry = out.setdefault(str(room), {})
            for metric_name, metric_value in metrics.items():
                try:
                    value = float(metric_value)
                except (TypeError, ValueError):
                    continue
                room_entry.setdefault(str(metric_name), []).append(value)
    return out


def _collect_cls_metric_values(reports: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for rep in reports:
        cls_summary = rep.get("classification_summary") or {}
        if not isinstance(cls_summary, dict):
            continue
        for room, metric_map in cls_summary.items():
            if not isinstance(metric_map, dict):
                continue
            room_entry = out.setdefault(str(room), {})
            for metric_name, payload in metric_map.items():
                if not isinstance(payload, dict):
                    continue
                try:
                    value = float(payload.get("mean", 0.0))
                except (TypeError, ValueError):
                    value = 0.0
                room_entry.setdefault(str(metric_name), []).append(value)
    return out


def _collect_timeline_metric_values(reports: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for rep in reports:
        splits = rep.get("splits", [])
        if not isinstance(splits, list):
            continue
        for split in splits:
            if not isinstance(split, dict):
                continue
            rooms = split.get("rooms", {})
            if not isinstance(rooms, dict):
                continue
            for room, payload in rooms.items():
                if not isinstance(payload, dict):
                    continue
                gt = payload.get("gt_targets", {}) or {}
                pred = payload.get("pred_targets", {}) or {}
                if not isinstance(gt, dict) or not isinstance(pred, dict):
                    continue
                room_entry = out.setdefault(str(room), {})
                for key, gt_val in gt.items():
                    key_txt = str(key)
                    if not key_txt.endswith("_events"):
                        continue
                    try:
                        g = float(gt_val)
                        p = float(pred.get(key_txt, 0.0))
                    except (TypeError, ValueError):
                        continue
                    room_entry.setdefault(f"{key_txt}_mae", []).append(abs(p - g))
    return out


@dataclass(frozen=True)
class KpiRule:
    room: str
    metric: str
    op: str
    threshold: float
    stability_std_limit: float
    tier: str


DEFAULT_KPI_RULES: List[KpiRule] = [
    KpiRule("Bedroom", "sleep_duration_mae_minutes", "<=", 120.0, 30.0, "tier_1"),
    KpiRule("LivingRoom", "livingroom_active_mae_minutes", "<=", 120.0, 30.0, "tier_2"),
    KpiRule("Kitchen", "kitchen_use_mae_minutes", "<=", 90.0, 30.0, "tier_2"),
    KpiRule("Bathroom", "bathroom_use_mae_minutes", "<=", 45.0, 30.0, "tier_2"),
    KpiRule("Bathroom", "shower_day_recall", ">=", 0.80, 0.10, "tier_1"),
    KpiRule("Bathroom", "shower_day_precision", ">=", 0.70, 0.10, "tier_1"),
    KpiRule("Entrance", "out_minutes_mae", "<=", 90.0, 30.0, "tier_3"),
]


@dataclass(frozen=True)
class TimelineRule:
    room: str
    metric: str
    op: str
    threshold: float
    stage: str


DEFAULT_TIMELINE_RULES: List[TimelineRule] = [
    TimelineRule("Bedroom", "sleep_events_mae", "<=", 2.0, "internal"),
    TimelineRule("LivingRoom", "livingroom_active_events_mae", "<=", 2.5, "internal"),
    TimelineRule("Kitchen", "kitchen_use_events_mae", "<=", 2.5, "internal"),
    TimelineRule("Bathroom", "bathroom_use_events_mae", "<=", 2.0, "internal"),
    TimelineRule("Entrance", "out_events_mae", "<=", 2.0, "internal"),
    TimelineRule("Bedroom", "sleep_events_mae", "<=", 1.5, "external"),
    TimelineRule("LivingRoom", "livingroom_active_events_mae", "<=", 2.0, "external"),
    TimelineRule("Kitchen", "kitchen_use_events_mae", "<=", 2.0, "external"),
    TimelineRule("Bathroom", "bathroom_use_events_mae", "<=", 1.5, "external"),
    TimelineRule("Entrance", "out_events_mae", "<=", 1.5, "external"),
]

STRICT_EXPECTED_SPLITS: List[Tuple[List[int], int]] = [
    ([4], 5),
    ([4, 5], 6),
    ([4, 5, 6], 7),
    ([4, 5, 6, 7], 8),
]
STRICT_EXPECTED_SEEDS: List[int] = [11, 22, 33]


def _canonical_split_id(train_days: Sequence[object], test_day: object) -> str:
    train_tokens = ",".join(str(int(day)) for day in train_days)
    return f"{train_tokens}->{int(test_day)}"


def _validate_split_seed_matrix(
    reports: Sequence[Dict[str, object]],
    *,
    expected_splits: Sequence[Tuple[Sequence[int], int]],
    expected_seeds: Sequence[int],
) -> list[str]:
    """
    Validate exact split-seed matrix coverage (fail-closed).

    Requirements:
    - every expected split-seed cell exists
    - no duplicates
    - no malformed split/seed metadata
    - no unexpected extra cells
    """
    violations: List[str] = []
    expected_cells = {
        (_canonical_split_id(train_days, test_day), int(seed))
        for train_days, test_day in expected_splits
        for seed in expected_seeds
    }

    observed_cells: List[Tuple[str, int]] = []
    for rep_idx, rep in enumerate(reports):
        seed_raw = rep.get("seed", rep_idx)
        try:
            seed = int(seed_raw)
        except (TypeError, ValueError):
            violations.append(f"invalid_seed_metadata:report{rep_idx}:{seed_raw}")
            continue

        splits = rep.get("splits", [])
        if not isinstance(splits, list):
            violations.append(f"missing_splits_payload:seed{seed}")
            continue
        if len(splits) == 0:
            violations.append(f"empty_splits_payload:seed{seed}")
            continue

        for split_idx, split in enumerate(splits):
            if not isinstance(split, dict):
                violations.append(f"invalid_split_payload:seed{seed}:split{split_idx}")
                continue

            split_id_raw = split.get("split_id", rep.get("split_id"))
            if split_id_raw is not None:
                split_id = str(split_id_raw)
            else:
                train_days = split.get("train_days", rep.get("train_days"))
                test_day = split.get("test_day", split.get("val_day", rep.get("val_day")))
                if not isinstance(train_days, list) or test_day is None:
                    violations.append(f"missing_split_metadata:seed{seed}:split{split_idx}")
                    continue
                try:
                    split_id = _canonical_split_id(train_days, test_day)
                except (TypeError, ValueError):
                    violations.append(f"invalid_split_metadata:seed{seed}:split{split_idx}")
                    continue

            observed_cells.append((split_id, seed))

    if not observed_cells:
        violations.append("no_split_seed_cells_observed")
        return violations

    observed_set = set(observed_cells)
    if len(observed_set) != len(observed_cells):
        dup_counts: Dict[Tuple[str, int], int] = {}
        for cell in observed_cells:
            dup_counts[cell] = dup_counts.get(cell, 0) + 1
        for (split_id, seed), count in sorted(dup_counts.items()):
            if count > 1:
                violations.append(f"duplicate_split_seed_cell:{split_id}:seed{seed}:count{count}")

    missing_cells = sorted(expected_cells - observed_set)
    extra_cells = sorted(observed_set - expected_cells)
    for split_id, seed in missing_cells:
        violations.append(f"missing_split_seed_cell:{split_id}:seed{seed}")
    for split_id, seed in extra_cells:
        violations.append(f"unexpected_split_seed_cell:{split_id}:seed{seed}")

    return violations


def _ci95(mean: float, std: float, n: int) -> tuple[float, float]:
    if n <= 1:
        return float(mean), float(mean)
    delta = 1.96 * (float(std) / math.sqrt(float(n)))
    return float(mean - delta), float(mean + delta)


def _evaluate_rule(rule: KpiRule, room_summary: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, object]:
    room_payload = room_summary.get(rule.room, {})
    metric_payload = room_payload.get(rule.metric, {})
    
    # Fail-closed: detect missing metric data
    n = int(metric_payload.get("n", 0))
    has_metric = rule.metric in room_payload and n > 0
    
    # Use NaN for missing metrics to distinguish from actual zero values
    mean = float(metric_payload.get("mean", float("nan"))) if has_metric else float("nan")
    std = float(metric_payload.get("std", float("nan"))) if has_metric else float("nan")
    ci_low, ci_high = _ci95(mean, std, n)

    # Fail-closed: missing metrics always fail
    if not has_metric:
        pass_raw = False
        ci_pass = False
    elif rule.op == "<=":
        pass_raw = bool(mean <= rule.threshold)
        ci_pass = bool(ci_high <= rule.threshold)
    else:
        pass_raw = bool(mean >= rule.threshold)
        ci_pass = bool(ci_low >= rule.threshold)
    
    stability_pass = bool(std <= rule.stability_std_limit) if has_metric and not math.isnan(std) else False

    return {
        "room": rule.room,
        "metric": rule.metric,
        "op": rule.op,
        "threshold": float(rule.threshold),
        "mean": mean,
        "std": std,
        "n": n,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "tier": rule.tier,
        "pass": pass_raw,
        "stability_std_limit": float(rule.stability_std_limit),
        "stability_pass": stability_pass,
        "ci_pass": ci_pass,
        "missing_metric": not has_metric,
    }


def _evaluate_timeline_rule(
    rule: TimelineRule,
    timeline_summary: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, object]:
    room_payload = timeline_summary.get(rule.room, {})
    metric_payload = room_payload.get(rule.metric, {})
    
    # Fail-closed: detect missing metric data
    n = int(metric_payload.get("n", 0))
    has_metric = rule.metric in room_payload and n > 0
    
    # Use NaN for missing metrics to distinguish from actual zero values
    mean = float(metric_payload.get("mean", float("nan"))) if has_metric else float("nan")
    std = float(metric_payload.get("std", float("nan"))) if has_metric else float("nan")
    
    # Fail-closed: missing metrics always fail
    if not has_metric:
        passed = False
    elif rule.op == "<=":
        passed = bool(mean <= rule.threshold)
    else:
        passed = bool(mean >= rule.threshold)
    
    return {
        "room": rule.room,
        "metric": rule.metric,
        "op": rule.op,
        "threshold": float(rule.threshold),
        "mean": mean,
        "std": std,
        "n": n,
        "stage": rule.stage,
        "pass": passed,
        "missing_metric": not has_metric,
    }


def _hash_payload(payload: Dict[str, object]) -> str:
    txt = json.dumps(payload, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(txt.encode('utf-8')).hexdigest()}"


def _is_missing_metadata(value: object) -> bool:
    if value is None:
        return True
    txt = str(value).strip()
    return txt == "" or txt.lower() in {"unknown", "none", "nan"}


def _validate_report_consistency(reports: Sequence[Dict[str, object]], report_paths: Sequence[Path]) -> list[str]:
    """Validate that all seed reports have consistent metadata. Returns list of violations."""
    if not reports:
        return ["no_reports_provided"]
    
    violations = []
    if len(report_paths) != len(reports):
        violations.append(f"report_path_count_mismatch:{len(report_paths)}!={len(reports)}")
    first = reports[0]
    
    # Fields that must be consistent across all reports
    consistency_fields = [
        ("elder_id", "elder_id"),
        ("data_version", "data_version"),
        ("feature_schema_hash", "feature_schema_hash"),
    ]
    
    # Required metadata must be present in seed 0 to define canonical run identity.
    for field_name, display_name in consistency_fields:
        first_val = first.get(field_name, None)
        if _is_missing_metadata(first_val):
            violations.append(f"missing_metadata:{display_name}:seed0")
    
    for i, rep in enumerate(reports[1:], start=1):
        for field_name, display_name in consistency_fields:
            first_val_raw = first.get(field_name, None)
            rep_val_raw = rep.get(field_name, None)
            if _is_missing_metadata(rep_val_raw):
                violations.append(f"missing_metadata:{display_name}:seed{i}")
                continue
            first_val = str(first_val_raw)
            rep_val = str(rep_val_raw)
            if first_val != rep_val:
                violations.append(
                    f"consistency_mismatch:{display_name}:seed0={first_val}:seed{i}={rep_val}"
                )
    
    return violations


def _validate_baseline_binding(
    baseline_version: str | None,
    baseline_artifact_hash: str | None,
    baseline_artifact_path: Path | None,
    require_baseline_for_promotion: bool,
) -> list[str]:
    """
    Validate baseline binding for promotion-grade signoff.
    
    D2 Gate: baseline_version and baseline_artifact_hash are mandatory for promotion.
    If baseline_artifact_path is provided, the hash is recomputed and verified.
    
    Returns list of violations (empty if valid).
    """
    violations = []
    
    if not require_baseline_for_promotion:
        return violations
    
    # Mandatory baseline_version
    if _is_missing_metadata(baseline_version):
        violations.append("baseline_binding:missing_baseline_version")
    
    # Mandatory baseline_artifact_hash
    if _is_missing_metadata(baseline_artifact_hash):
        violations.append("baseline_binding:missing_baseline_artifact_hash")
    
    # Hash verification: mandatory in promotion mode
    if require_baseline_for_promotion and baseline_artifact_path is None:
        violations.append("baseline_binding:missing_artifact_path")
    elif baseline_artifact_path is not None:
        if not baseline_artifact_path.exists():
            violations.append(f"baseline_binding:artifact_not_found:{baseline_artifact_path}")
        else:
            computed_hash = _compute_artifact_hash(baseline_artifact_path)
            expected_hash = str(baseline_artifact_hash) if baseline_artifact_hash else ""
            # Handle both "sha256:abc123" and "abc123" formats
            if expected_hash.startswith("sha256:"):
                expected_hash = expected_hash[7:]
            if computed_hash != expected_hash:
                violations.append(
                    f"baseline_binding:hash_mismatch:expected={expected_hash}:computed={computed_hash}"
                )
    
    return violations


def _compute_artifact_hash(artifact_path: Path) -> str:
    """
    Compute SHA256 hash of an artifact file.
    
    Args:
        artifact_path: Path to the artifact file
        
    Returns:
        Hex digest of the file hash
    """
    h = hashlib.sha256()
    with open(artifact_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_leakage_audits(
    reports: Sequence[Dict[str, object]],
    leakage_audit_paths: Sequence[Path] | None = None,
    require_leakage_artifact: bool = True,
) -> list[str]:
    """
    Validate leakage audit pass status across all split-seed cells.
    
    D2 Gate: Any leakage audit failure blocks signoff (fail-closed).
    
    If leakage_audit_paths is provided, validates that:
    1. Each path exists (fail-closed: missing file = fail)
    2. Each file is valid JSON with required schema
    3. Each audit record has pass=True
    
    Args:
        reports: List of seed reports (with inline leakage_audit_pass checks)
        leakage_audit_paths: Optional list of paths to leakage_audit.json files
        require_leakage_artifact: If True, requires leakage_audit_paths to be provided
        
    Returns list of violations.
    """
    violations = []
    
    # D2 Gate: Require separate leakage_audit.json artifact
    if require_leakage_artifact and leakage_audit_paths is None:
        violations.append("leakage_audit:missing_artifact_requirement")
    
    # Validate leakage audit files if provided
    if leakage_audit_paths is not None:
        if len(leakage_audit_paths) != len(reports):
            violations.append(
                f"leakage_audit:path_count_mismatch:{len(leakage_audit_paths)}!={len(reports)}"
            )
        
        for idx, audit_path in enumerate(leakage_audit_paths):
            seed = reports[idx].get("seed", idx) if idx < len(reports) else idx
            
            # Fail-closed: file must exist
            if not audit_path.exists():
                violations.append(f"leakage_audit:seed{seed}:artifact_not_found:{audit_path}")
                continue
            
            # Fail-closed: file must be valid JSON
            try:
                audit_data = json.loads(audit_path.read_text())
            except json.JSONDecodeError as e:
                violations.append(f"leakage_audit:seed{seed}:invalid_json:{e}")
                continue
            
            # Fail-closed: must have required schema
            if not isinstance(audit_data, dict):
                violations.append(f"leakage_audit:seed{seed}:invalid_schema:not_dict")
                continue
            
            # Check overall audit pass status
            audit_pass = audit_data.get("audit_pass")
            if audit_pass is not True:
                violations.append(f"leakage_audit:seed{seed}:audit_pass_false")
            
            # Validate per-split audit records
            split_audits = audit_data.get("split_audits", [])
            if not isinstance(split_audits, list):
                violations.append(f"leakage_audit:seed{seed}:missing_split_audits")
                continue
            
            for split_idx, split_audit in enumerate(split_audits):
                if not isinstance(split_audit, dict):
                    violations.append(f"leakage_audit:seed{seed}:split{split_idx}:invalid_audit")
                    continue
                
                split_pass = split_audit.get("pass")
                if split_pass is not True:
                    violations.append(f"leakage_audit:seed{seed}:split{split_idx}:pass_false")
    
    # Also validate inline leakage_audit_pass for backward compatibility
    for rep_idx, rep in enumerate(reports):
        seed = rep.get("seed", rep_idx)
        splits = rep.get("splits", [])
        
        if not isinstance(splits, list):
            violations.append(f"leakage_audit_inline:seed{seed}:missing_splits")
            continue
        
        for split_idx, split in enumerate(splits):
            if not isinstance(split, dict):
                violations.append(f"leakage_audit_inline:seed{seed}:split{split_idx}:invalid_split_payload")
                continue
            
            rooms = split.get("rooms", {})
            if not isinstance(rooms, dict):
                violations.append(f"leakage_audit_inline:seed{seed}:split{split_idx}:invalid_rooms_payload")
                continue
            if len(rooms) == 0:
                violations.append(f"leakage_audit_inline:seed{seed}:split{split_idx}:empty_rooms_payload")
                continue
            
            for room_name, room_data in rooms.items():
                if not isinstance(room_data, dict):
                    violations.append(
                        f"leakage_audit_inline:seed{seed}:split{split_idx}:room{room_name}:invalid_room_payload"
                    )
                    continue
                
                # Prefer explicit top-level flag; fallback to legacy nested payload.
                leakage_pass = room_data.get("leakage_audit_pass")
                if leakage_pass is None:
                    legacy_audit = room_data.get("leakage_audit")
                    if isinstance(legacy_audit, dict):
                        leakage_pass = legacy_audit.get("pass")
                if leakage_pass is not True:
                    violations.append(
                        f"leakage_audit_inline_failed:seed{seed}:split{split_idx}:room{room_name}"
                    )
    
    return violations


def aggregate_reports(
    reports: Sequence[Dict[str, object]],
    *,
    report_paths: Sequence[Path],
    comparison_window: str,
    required_split_pass_count: int | None,
    required_split_pass_ratio: float,
    baseline_version: str | None = None,
    baseline_artifact_hash: str | None = None,
    baseline_artifact_path: Path | None = None,
    leakage_audit_paths: Sequence[Path] | None = None,
    enforce_strict_split_seed_matrix: bool = True,
    require_baseline_for_promotion: bool = True,
    require_leakage_artifact: bool = True,
) -> tuple[Dict[str, object], Dict[str, object]]:
    if not reports:
        raise ValueError("No reports provided")

    # Validate consistency across all seed reports
    consistency_violations = _validate_report_consistency(reports, report_paths)
    
    # D2 Gate: Validate baseline binding for promotion-grade signoff
    baseline_violations = _validate_baseline_binding(
        baseline_version=baseline_version,
        baseline_artifact_hash=baseline_artifact_hash,
        baseline_artifact_path=baseline_artifact_path,
        require_baseline_for_promotion=require_baseline_for_promotion,
    )
    consistency_violations.extend(baseline_violations)
    
    first = reports[0]
    elder_id = str(first.get("elder_id", "unknown"))
    days = first.get("days", [])
    data_version = str(first.get("data_version", "unknown"))
    feature_schema_hash = str(first.get("feature_schema_hash", "unknown"))
    leakage_checklist = first.get("leakage_checklist", [])
    calibration_summary = first.get("calibration_summary", {})
    run_timestamp_utc = _utc_now_iso_z()

    room_values = _collect_room_metric_values(reports)
    room_summary = {
        room: {metric: _safe_mean_std(values) for metric, values in metric_map.items()}
        for room, metric_map in room_values.items()
    }

    cls_values = _collect_cls_metric_values(reports)
    classification_summary = {
        room: {metric: _safe_mean_std(values) for metric, values in metric_map.items()}
        for room, metric_map in cls_values.items()
    }
    timeline_values = _collect_timeline_metric_values(reports)
    timeline_summary = {
        room: {metric: _safe_mean_std(values) for metric, values in metric_map.items()}
        for room, metric_map in timeline_values.items()
    }

    seeds = [int(rep.get("seed", idx)) for idx, rep in enumerate(reports)]
    split_records = []
    total_split_checks_eligible = 0
    passed_split_checks_eligible = 0
    total_split_checks_full = 0
    passed_split_checks_full = 0
    seed_hard_pass_eligible = {}
    seed_hard_pass_full = {}
    for idx, rep in enumerate(reports):
        gate = rep.get("gate_summary") or {}
        total_eligible = int(
            gate.get(
                "hard_gate_checks_total_eligible",
                gate.get("hard_gate_checks_total", 0),
            )
            or 0
        )
        passed_eligible = int(
            gate.get(
                "hard_gate_checks_passed_eligible",
                gate.get("hard_gate_checks_passed", 0),
            )
            or 0
        )
        total_full = int(gate.get("hard_gate_checks_total_full", total_eligible) or 0)
        passed_full = int(gate.get("hard_gate_checks_passed_full", passed_eligible) or 0)
        split_records.append(
            {
                "seed": int(rep.get("seed", seeds[idx])),
                "count": int(len(rep.get("splits", []))) if isinstance(rep.get("splits"), list) else 0,
                "path": str(report_paths[idx]),
                "hard_gate_checks_total": total_eligible,
                "hard_gate_checks_passed": passed_eligible,
                "hard_gate_checks_total_eligible": total_eligible,
                "hard_gate_checks_passed_eligible": passed_eligible,
                "hard_gate_checks_total_full": total_full,
                "hard_gate_checks_passed_full": passed_full,
            }
        )
        total_split_checks_eligible += total_eligible
        passed_split_checks_eligible += passed_eligible
        total_split_checks_full += total_full
        passed_split_checks_full += passed_full
        seed_key = int(rep.get("seed", seeds[idx]))
        seed_hard_pass_eligible[seed_key] = bool(total_eligible > 0 and passed_eligible == total_eligible)
        seed_hard_pass_full[seed_key] = bool(total_full > 0 and passed_full == total_full)

    if required_split_pass_count is None:
        required_split_pass_count = int(
            math.ceil(float(required_split_pass_ratio) * float(total_split_checks_eligible))
        )
    required_split_pass_count = max(0, int(required_split_pass_count))

    hard_gate_all_seeds = bool(all(seed_hard_pass_eligible.values())) if seed_hard_pass_eligible else False
    hard_gate_all_seeds_full = bool(all(seed_hard_pass_full.values())) if seed_hard_pass_full else False
    hard_gate_split_requirement_pass = bool(passed_split_checks_eligible >= required_split_pass_count)

    rolling_payload = {
        "elder_id": elder_id,
        "run_timestamp_utc": run_timestamp_utc,
        "baseline_version": baseline_version,
        "baseline_artifact_hash": baseline_artifact_hash,
        "git_sha": str(first.get("git_sha") or _safe_git_sha(Path(__file__).resolve().parents[2])),
        "config_hash": _hash_payload(
            {
                "comparison_window": comparison_window,
                "required_split_pass_count": required_split_pass_count,
                "required_split_pass_ratio": required_split_pass_ratio,
                "kpi_rules": [rule.__dict__ for rule in DEFAULT_KPI_RULES],
                "seed_reports": [str(p) for p in report_paths],
                "baseline_version": baseline_version,
                "baseline_artifact_hash": baseline_artifact_hash,
            }
        ),
        "data_version": data_version,
        "feature_schema_hash": feature_schema_hash,
        "model_hashes": {},
        "days": days,
        "seeds": seeds,
        "splits": split_records,
        "room_summary": room_summary,
        "classification_summary": classification_summary,
        "timeline_summary": timeline_summary,
        "home_empty_summary": {},
        "gate_summary": {
            "hard_gate_all_seeds": hard_gate_all_seeds,
            "hard_gate_checks_total": total_split_checks_eligible,
            "hard_gate_checks_passed": passed_split_checks_eligible,
            "hard_gate_checks_total_eligible": total_split_checks_eligible,
            "hard_gate_checks_passed_eligible": passed_split_checks_eligible,
            "hard_gate_all_seeds_full": hard_gate_all_seeds_full,
            "hard_gate_checks_total_full": total_split_checks_full,
            "hard_gate_checks_passed_full": passed_split_checks_full,
            "required_split_pass_count": required_split_pass_count,
            "hard_gate_split_requirement_pass": hard_gate_split_requirement_pass,
            "split_pass_rate": (
                float(passed_split_checks_eligible) / float(total_split_checks_eligible)
            )
            if total_split_checks_eligible > 0
            else 0.0,
            "split_pass_rate_full": (
                float(passed_split_checks_full) / float(total_split_checks_full)
            )
            if total_split_checks_full > 0
            else 0.0,
        },
        "registry_version": "v1",
        "leakage_checklist": leakage_checklist,
        "calibration_summary": calibration_summary,
    }

    kpi_checks = [_evaluate_rule(rule, room_summary) for rule in DEFAULT_KPI_RULES]
    timeline_checks = [_evaluate_timeline_rule(rule, timeline_summary) for rule in DEFAULT_TIMELINE_RULES]
    failed_reasons: List[str] = []
    
    # Add consistency violations first (blocking)
    failed_reasons.extend(consistency_violations)

    # D2 Gate: strict split-seed matrix must be complete and exact (4x3).
    split_seed_violations: List[str] = []
    if enforce_strict_split_seed_matrix:
        split_seed_violations = _validate_split_seed_matrix(
            reports,
            expected_splits=STRICT_EXPECTED_SPLITS,
            expected_seeds=STRICT_EXPECTED_SEEDS,
        )
        failed_reasons.extend(split_seed_violations)
    
    # D2 Gate: Validate leakage audits (fail-closed)
    leakage_violations = _validate_leakage_audits(
        reports,
        leakage_audit_paths=leakage_audit_paths,
        require_leakage_artifact=require_leakage_artifact,
    )
    failed_reasons.extend(leakage_violations)
    
    for check in kpi_checks:
        if bool(check.get("missing_metric", False)):
            failed_reasons.append(f"missing_metric:{check['room']}:{check['metric']}")
        elif not bool(check["pass"]):
            failed_reasons.append(f"kpi_failed:{check['room']}:{check['metric']}:{check['mean']}")
        if not bool(check["stability_pass"]):
            failed_reasons.append(f"kpi_unstable:{check['room']}:{check['metric']}:{check['std']}")
        if str(check.get("tier")) == "tier_1" and not bool(check.get("ci_pass", False)):
            failed_reasons.append(f"kpi_tier1_ci_failed:{check['room']}:{check['metric']}")
    
    for check in timeline_checks:
        if bool(check.get("missing_metric", False)):
            failed_reasons.append(f"missing_timeline_metric:{check['room']}:{check['metric']}")

    if not hard_gate_all_seeds:
        failed_reasons.append("hard_gate_all_seeds_failed")
    if not hard_gate_split_requirement_pass:
        failed_reasons.append(
            f"hard_gate_split_requirement_failed:{passed_split_checks_eligible}<{required_split_pass_count}"
        )

    timeline_internal_checks = [c for c in timeline_checks if c.get("stage") == "internal"]
    timeline_external_checks = [c for c in timeline_checks if c.get("stage") == "external"]
    timeline_internal_pass = bool(all(bool(c.get("pass", False)) for c in timeline_internal_checks))
    timeline_external_pass = bool(all(bool(c.get("pass", False)) for c in timeline_external_checks))
    if timeline_external_pass:
        timeline_stage = "external_ready"
    elif timeline_internal_pass:
        timeline_stage = "internal_ready"
    else:
        timeline_stage = "not_ready"

    signoff_payload = {
        "comparison_window": comparison_window,
        "run_timestamp_utc": run_timestamp_utc,
        "baseline_version": baseline_version,
        "baseline_artifact_hash": baseline_artifact_hash,
        "git_sha": rolling_payload["git_sha"],
        "config_hash": rolling_payload["config_hash"],
        "data_version": data_version,
        "feature_schema_hash": feature_schema_hash,
        "legacy": {},
        "event_first": {
            "room_summary": room_summary,
            "classification_summary": classification_summary,
            "kpi_checks": kpi_checks,
            "timeline_summary": timeline_summary,
            "timeline_checks": timeline_checks,
        },
        "delta": {},
        "gate_decision": "PASS" if len(failed_reasons) == 0 else "FAIL",
        "failed_reasons": failed_reasons,
        "registry_version": "v1",
        "model_version_candidate": "event_first_shadow_candidate",
        "seed_split_stability": {
            "hard_gate_all_seeds": hard_gate_all_seeds,
            "hard_gate_splits_passed": int(passed_split_checks_eligible),
            "hard_gate_splits_total": int(total_split_checks_eligible),
            "hard_gate_all_seeds_full": hard_gate_all_seeds_full,
            "hard_gate_splits_passed_full": int(passed_split_checks_full),
            "hard_gate_splits_total_full": int(total_split_checks_full),
            "required_split_pass_count": int(required_split_pass_count),
            "strict_split_seed_matrix_enforced": bool(enforce_strict_split_seed_matrix),
            "strict_split_seed_matrix_pass": bool(len(split_seed_violations) == 0),
        },
        "timeline_release": {
            "internal_ready": timeline_internal_pass,
            "external_ready": timeline_external_pass,
            "recommended_stage": timeline_stage,
        },
    }
    return rolling_payload, signoff_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate event-first backtests into rolling/signoff artifacts")
    parser.add_argument(
        "--seed-reports",
        nargs="+",
        required=True,
        help="Paths to per-seed JSON reports from run_event_first_backtest.py",
    )
    parser.add_argument("--rolling-output", required=True, help="Output path for rolling summary JSON")
    parser.add_argument("--signoff-output", required=True, help="Output path for signoff JSON")
    parser.add_argument("--comparison-window", default="dec4_to_dec10")
    parser.add_argument("--required-split-pass-count", type=int, default=None)
    parser.add_argument("--required-split-pass-ratio", type=float, default=1.0)
    parser.add_argument("--baseline-version", type=str, default=None, help="Baseline version identifier (e.g., v31)")
    parser.add_argument("--baseline-artifact-hash", type=str, default=None, help="Baseline artifact hash for reproducibility")
    parser.add_argument("--baseline-artifact-path", type=Path, default=None, help="Path to baseline artifact for hash verification")
    parser.add_argument("--require-baseline-for-promotion", type=lambda x: x.lower() in ('true', '1', 'yes'), default=True, help="Require baseline binding for promotion (default: True)")
    parser.add_argument(
        "--leakage-audit-paths",
        nargs="+",
        default=None,
        help="Optional paths to leakage_audit.json artifacts (one per seed report).",
    )
    parser.add_argument(
        "--require-leakage-artifact",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Require leakage audit artifacts for promotion-grade signoff (default: True)",
    )
    parser.set_defaults(enforce_strict_split_seed_matrix=True)
    parser.add_argument(
        "--enforce-strict-split-seed-matrix",
        dest="enforce_strict_split_seed_matrix",
        action="store_true",
        help="Require exact canonical split-seed matrix coverage (default: enabled).",
    )
    parser.add_argument(
        "--no-enforce-strict-split-seed-matrix",
        dest="enforce_strict_split_seed_matrix",
        action="store_false",
        help="Disable strict split-seed matrix validation (debug/testing only).",
    )
    args = parser.parse_args()

    report_paths = [Path(p) for p in args.seed_reports]
    reports = [json.loads(p.read_text()) for p in report_paths]
    leakage_audit_paths = [Path(p) for p in args.leakage_audit_paths] if args.leakage_audit_paths else None
    rolling_payload, signoff_payload = aggregate_reports(
        reports,
        report_paths=report_paths,
        comparison_window=str(args.comparison_window),
        required_split_pass_count=args.required_split_pass_count,
        required_split_pass_ratio=float(args.required_split_pass_ratio),
        baseline_version=args.baseline_version,
        baseline_artifact_hash=args.baseline_artifact_hash,
        baseline_artifact_path=args.baseline_artifact_path,
        leakage_audit_paths=leakage_audit_paths,
        enforce_strict_split_seed_matrix=bool(args.enforce_strict_split_seed_matrix),
        require_baseline_for_promotion=bool(args.require_baseline_for_promotion),
        require_leakage_artifact=bool(args.require_leakage_artifact),
    )

    rolling_path = Path(args.rolling_output)
    signoff_path = Path(args.signoff_output)
    rolling_path.parent.mkdir(parents=True, exist_ok=True)
    signoff_path.parent.mkdir(parents=True, exist_ok=True)
    rolling_path.write_text(json.dumps(rolling_payload, indent=2))
    signoff_path.write_text(json.dumps(signoff_payload, indent=2))
    print(f"Wrote rolling summary: {rolling_path}")
    print(f"Wrote signoff summary: {signoff_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
