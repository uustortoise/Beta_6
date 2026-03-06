#!/usr/bin/env python3
"""Summarize Step-1 A/B backtest artifacts into gate-ready JSON + Markdown."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


CORE_ROOMS = ("bedroom", "livingroom", "bathroom", "kitchen")
MASKED_ACTIVITY_LABELS = {"unoccupied", "unknown"}
ROOM_MIN_ELIGIBLE_LABELS = {
    "bedroom": 2,
    "bathroom": 2,
    "livingroom": 1,
    "kitchen": 1,
}


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_room(room_name: str) -> str:
    return str(room_name or "").strip().lower()


@dataclass
class VariantSummary:
    name: str
    path: str
    livingroom_active_mae_minutes: float | None
    bedroom_sleep_duration_mae_minutes: float | None
    hard_gate_checks_total: int | None
    hard_gate_checks_passed: int | None
    hard_gate_pass_rate: float | None
    lr_mae_improvement_vs_full_bundle_pct: float | None
    bedroom_sleep_mae_delta_vs_a0_minutes: float | None
    occupancy_false_empty_rate: float | None
    occupancy_home_empty_precision: float | None
    occupancy_home_empty_recall: float | None
    occupancy_home_empty_predicted_empty_rate: float | None
    occupancy_home_empty_windows_total: int | None
    occupancy_home_empty_predicted_empty_windows: int | None
    occupancy_metrics_source: str
    occupancy_inference: Dict[str, Any]
    minority_recall_by_room: Dict[str, Any]
    bedroom_sleep_recall: Dict[str, Any]


def _extract_variant_metrics(
    *,
    name: str,
    report_path: Path,
    full_bundle_livingroom_mae: float,
) -> VariantSummary:
    payload = json.loads(report_path.read_text())
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    gate_summary = payload.get("gate_summary", {}) if isinstance(payload.get("gate_summary"), dict) else {}

    livingroom_mae = _to_float(
        (summary.get("LivingRoom", {}) if isinstance(summary.get("LivingRoom"), dict) else {}).get(
            "livingroom_active_mae_minutes"
        )
    )
    bedroom_sleep_mae = _to_float(
        (summary.get("Bedroom", {}) if isinstance(summary.get("Bedroom"), dict) else {}).get(
            "sleep_duration_mae_minutes"
        )
    )

    # Collect support-weighted per-label recall by room from split payload.
    recall_accum: Dict[str, Dict[str, Dict[str, float]]] = {room: {} for room in CORE_ROOMS}

    # Occupancy safety is inferred from occupied precision/recall + occupancy support.
    occ_tp = 0.0
    occ_fn = 0.0
    occ_fp = 0.0
    occ_tn = 0.0
    evaluated_splits = 0

    home_empty_summary = payload.get("home_empty_summary", {})
    if not isinstance(home_empty_summary, dict):
        home_empty_summary = {}
    direct_home_empty_evaluated = bool(home_empty_summary.get("evaluated", False))
    direct_home_empty_precision = _to_float(home_empty_summary.get("home_empty_precision"))
    direct_home_empty_recall = _to_float(home_empty_summary.get("home_empty_recall"))
    direct_home_empty_false_empty_rate = _to_float(home_empty_summary.get("home_empty_false_empty_rate"))
    direct_home_empty_tp = _to_int(home_empty_summary.get("tp"))
    direct_home_empty_fp = _to_int(home_empty_summary.get("fp"))
    direct_home_empty_fn = _to_int(home_empty_summary.get("fn"))
    direct_home_empty_tn = _to_int(home_empty_summary.get("tn"))

    splits = payload.get("splits", [])
    if not isinstance(splits, list):
        splits = []
    for split in splits:
        if not isinstance(split, dict):
            continue
        rooms = split.get("rooms", {})
        if not isinstance(rooms, dict):
            continue
        for room_name, room_payload in rooms.items():
            room_key = _normalize_room(str(room_name))
            if room_key not in CORE_ROOMS or not isinstance(room_payload, dict):
                continue

            recall_summary = room_payload.get("label_recall_summary", {})
            if isinstance(recall_summary, dict):
                for label_name, label_payload in recall_summary.items():
                    if not isinstance(label_payload, dict):
                        continue
                    support = _to_float(label_payload.get("support"))
                    recall = _to_float(label_payload.get("recall"))
                    if support is None or recall is None or support <= 0:
                        continue
                    label_key = str(label_name).strip().lower()
                    rec = recall_accum[room_key].setdefault(
                        label_key,
                        {"support": 0.0, "weighted_recall_sum": 0.0},
                    )
                    rec["support"] += float(support)
                    rec["weighted_recall_sum"] += float(support) * max(0.0, min(1.0, float(recall)))

            classification = room_payload.get("classification", {})
            occupied_snapshot = (
                (room_payload.get("occupied_rate_snapshot", {}) if isinstance(room_payload.get("occupied_rate_snapshot"), dict) else {})
                .get("test", {})
            )
            if not isinstance(classification, dict) or not isinstance(occupied_snapshot, dict):
                continue

            windows = _to_float(occupied_snapshot.get("windows"))
            occupied_windows = _to_float(occupied_snapshot.get("occupied_windows"))
            occupied_recall = _to_float(classification.get("occupied_recall"))
            occupied_precision = _to_float(classification.get("occupied_precision"))
            if (
                windows is None
                or occupied_windows is None
                or occupied_recall is None
                or occupied_precision is None
                or windows <= 0
                or occupied_windows < 0
            ):
                continue

            p = max(0.0, min(float(occupied_windows), float(windows)))
            n = max(0.0, float(windows) - p)
            rec = max(0.0, min(1.0, float(occupied_recall)))
            prec = max(0.0, min(1.0, float(occupied_precision)))
            tp = rec * p
            fn = max(0.0, p - tp)
            if tp <= 0.0:
                fp = 0.0
            elif prec <= 0.0:
                fp = n
            else:
                fp = tp * (1.0 / prec - 1.0)
            fp = max(0.0, min(n, fp))
            tn = max(0.0, n - fp)

            occ_tp += tp
            occ_fn += fn
            occ_fp += fp
            occ_tn += tn
            evaluated_splits += 1

    room_label_rollup: Dict[str, Any] = {}
    for room_key, labels in recall_accum.items():
        room_payload: Dict[str, Any] = {"labels": {}, "eligible_nonmasked_labels": [], "positive_recall_labels": []}
        for label_key, agg in labels.items():
            support = float(agg.get("support", 0.0))
            weighted_sum = float(agg.get("weighted_recall_sum", 0.0))
            recall_value = (weighted_sum / support) if support > 0 else 0.0
            room_payload["labels"][label_key] = {
                "support": support,
                "recall": recall_value,
            }
            if label_key not in MASKED_ACTIVITY_LABELS and support >= 5.0:
                room_payload["eligible_nonmasked_labels"].append(label_key)
                if recall_value > 0.0:
                    room_payload["positive_recall_labels"].append(label_key)
        room_payload["eligible_nonmasked_labels"] = sorted(room_payload["eligible_nonmasked_labels"])
        room_payload["positive_recall_labels"] = sorted(room_payload["positive_recall_labels"])
        room_label_rollup[room_key] = room_payload

    sleep_payload = room_label_rollup.get("bedroom", {}).get("labels", {}).get("sleep", {})
    sleep_support = _to_float(sleep_payload.get("support"))
    sleep_recall = _to_float(sleep_payload.get("recall"))
    bedroom_sleep_recall = {
        "support": float(sleep_support or 0.0),
        "recall": float(sleep_recall or 0.0),
        "status": (
            "pass"
            if (sleep_support is not None and sleep_support >= 5.0 and sleep_recall is not None and sleep_recall > 0.0)
            else (
                "fail"
                if (sleep_support is not None and sleep_support >= 5.0 and (sleep_recall is None or sleep_recall <= 0.0))
                else "not_evaluable"
            )
        ),
    }

    occupancy_metrics_source = "inferred"
    occ_false_empty_rate = None
    home_empty_precision = None
    home_empty_recall = None
    home_empty_predicted_empty_rate = None
    home_empty_windows_total = None
    home_empty_predicted_empty_windows = None
    if (
        direct_home_empty_evaluated
        and direct_home_empty_precision is not None
        and direct_home_empty_false_empty_rate is not None
    ):
        occupancy_metrics_source = "direct_home_empty_summary"
        occ_false_empty_rate = float(direct_home_empty_false_empty_rate)
        home_empty_precision = float(direct_home_empty_precision)
        home_empty_recall = (
            float(direct_home_empty_recall) if direct_home_empty_recall is not None else None
        )
        if None not in (direct_home_empty_tp, direct_home_empty_fp, direct_home_empty_fn, direct_home_empty_tn):
            total_windows = int(direct_home_empty_tp + direct_home_empty_fp + direct_home_empty_fn + direct_home_empty_tn)
            predicted_empty_windows = int(direct_home_empty_tp + direct_home_empty_fp)
            home_empty_windows_total = total_windows
            home_empty_predicted_empty_windows = predicted_empty_windows
            if total_windows > 0:
                home_empty_predicted_empty_rate = float(predicted_empty_windows / total_windows)
    else:
        if (occ_tp + occ_fn) > 0:
            occ_false_empty_rate = float(occ_fn / (occ_tp + occ_fn))
        if (occ_tn + occ_fn) > 0:
            home_empty_precision = float(occ_tn / (occ_tn + occ_fn))

    lr_improve = None
    if livingroom_mae is not None and full_bundle_livingroom_mae > 0:
        lr_improve = float((full_bundle_livingroom_mae - livingroom_mae) / full_bundle_livingroom_mae)

    return VariantSummary(
        name=name,
        path=str(report_path),
        livingroom_active_mae_minutes=livingroom_mae,
        bedroom_sleep_duration_mae_minutes=bedroom_sleep_mae,
        hard_gate_checks_total=_to_int(gate_summary.get("hard_gate_checks_total")),
        hard_gate_checks_passed=_to_int(gate_summary.get("hard_gate_checks_passed")),
        hard_gate_pass_rate=_to_float(gate_summary.get("hard_gate_pass_rate")),
        lr_mae_improvement_vs_full_bundle_pct=lr_improve,
        bedroom_sleep_mae_delta_vs_a0_minutes=None,
        occupancy_false_empty_rate=occ_false_empty_rate,
        occupancy_home_empty_precision=home_empty_precision,
        occupancy_home_empty_recall=home_empty_recall,
        occupancy_home_empty_predicted_empty_rate=home_empty_predicted_empty_rate,
        occupancy_home_empty_windows_total=home_empty_windows_total,
        occupancy_home_empty_predicted_empty_windows=home_empty_predicted_empty_windows,
        occupancy_metrics_source=occupancy_metrics_source,
        occupancy_inference={
            "tp": occ_tp,
            "fn": occ_fn,
            "fp": occ_fp,
            "tn": occ_tn,
            "evaluated_room_splits": int(evaluated_splits),
            "method": (
                "derived_from_occupied_precision_recall_and_occupied_support; "
                "inferred_not_direct_metric"
                if occupancy_metrics_source == "inferred"
                else "direct_home_empty_summary"
            ),
            "direct_home_empty_summary": home_empty_summary if occupancy_metrics_source != "inferred" else {},
        },
        minority_recall_by_room=room_label_rollup,
        bedroom_sleep_recall=bedroom_sleep_recall,
    )


def _variant_to_dict(item: VariantSummary) -> Dict[str, Any]:
    return {
        "path": item.path,
        "livingroom_active_mae_minutes": item.livingroom_active_mae_minutes,
        "bedroom_sleep_duration_mae_minutes": item.bedroom_sleep_duration_mae_minutes,
        "hard_gate_checks_total": item.hard_gate_checks_total,
        "hard_gate_checks_passed": item.hard_gate_checks_passed,
        "hard_gate_pass_rate": item.hard_gate_pass_rate,
        "lr_mae_improvement_vs_full_bundle_pct": item.lr_mae_improvement_vs_full_bundle_pct,
        "bedroom_sleep_mae_delta_vs_a0_minutes": item.bedroom_sleep_mae_delta_vs_a0_minutes,
        "occupancy_false_empty_rate": item.occupancy_false_empty_rate,
        "occupancy_home_empty_precision": item.occupancy_home_empty_precision,
        "occupancy_home_empty_recall": item.occupancy_home_empty_recall,
        "occupancy_home_empty_predicted_empty_rate": item.occupancy_home_empty_predicted_empty_rate,
        "occupancy_home_empty_windows_total": item.occupancy_home_empty_windows_total,
        "occupancy_home_empty_predicted_empty_windows": item.occupancy_home_empty_predicted_empty_windows,
        "occupancy_metrics_source": item.occupancy_metrics_source,
        "occupancy_inference": item.occupancy_inference,
        "minority_recall_by_room": item.minority_recall_by_room,
        "bedroom_sleep_recall": item.bedroom_sleep_recall,
    }


def _build_markdown(summary: Dict[str, Any]) -> str:
    variants = summary.get("variants", {})
    gate = summary.get("gate_evaluation", {})
    operational = summary.get("operational_recommendation", {})
    lines = [
        "# Beta 6.1 Step 1 A/B Report",
        "",
        f"Date: {summary.get('date')}",
        f"Resident: {summary.get('resident')} | Seed: {summary.get('seed')} | Window: day {summary.get('window', {}).get('min_day')}-{summary.get('window', {}).get('max_day')}",
        "",
        "## Variant Results",
        "",
        "| Variant | LR MAE | Improvement vs Full Bundle | Bedroom Sleep MAE | Bedroom Δ vs A0 | Hard Gate Pass | False-Empty Rate* | Home-Empty Precision* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("A0", "A1", "A2"):
        payload = variants.get(name, {})
        lr = payload.get("livingroom_active_mae_minutes")
        imp = payload.get("lr_mae_improvement_vs_full_bundle_pct")
        bd = payload.get("bedroom_sleep_duration_mae_minutes")
        dlt = payload.get("bedroom_sleep_mae_delta_vs_a0_minutes")
        hp = payload.get("hard_gate_checks_passed")
        ht = payload.get("hard_gate_checks_total")
        fe = payload.get("occupancy_false_empty_rate")
        hpn = payload.get("occupancy_home_empty_precision")
        lines.append(
            f"| {name} | {lr:.3f} | {imp*100:.2f}% | {bd:.3f} | {dlt:+.3f} | {hp}/{ht} | {fe:.4f} | {hpn:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Gate Check Against Plan",
            "",
            f"- LivingRoom MAE improve >= 20% vs full bundle: **{gate.get('livingroom_mae_improve_vs_full_bundle_ge_20pct', {}).get('status', 'unknown').upper()}**",
            f"- Bedroom sleep MAE delta <= +2.0 minutes vs A0: **{gate.get('bedroom_sleep_mae_delta_le_2_minutes', {}).get('status', 'unknown').upper()}**",
            f"- Hard-gate pass count non-regression: **{gate.get('hard_gate_pass_count_non_regression', {}).get('status', 'unknown').upper()}**",
            f"- Minority recall anti-collapse: **{gate.get('minority_recall_anti_collapse', {}).get('status', 'unknown').upper()}**",
            f"- Occupancy-head safety non-regression: **{gate.get('occupancy_head_safety_non_regression', {}).get('status', 'unknown').upper()}**",
            f"- Home-empty operational utility: **{gate.get('home_empty_operational_utility', {}).get('status', 'unknown').upper()}**",
            "",
            "## Decision",
            "",
            f"- Recommended variant: **{summary.get('recommended_variant')}**",
            f"- Overall Step-1 promotion decision: **{summary.get('decision', {}).get('status')}**",
            f"- Operational recommendation: **{operational.get('status', 'UNKNOWN')}**",
            "",
            "*Safety metrics use direct home-empty summary when present; otherwise inference fallback is used.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Step-1 A/B artifacts.")
    parser.add_argument("--a0-report", required=True)
    parser.add_argument("--a1-report", required=True)
    parser.add_argument("--a2-report", required=True)
    parser.add_argument("--resident", default="HK001_jessica")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--min-day", type=int, default=7)
    parser.add_argument("--max-day", type=int, default=10)
    parser.add_argument("--full-bundle-lr-mae", type=float, default=242.95)
    parser.add_argument("--baseline-anchor-lr-mae", type=float, default=88.39)
    parser.add_argument(
        "--home-empty-utility-precision-min",
        type=float,
        default=0.25,
        help="Minimum home-empty precision for operational utility pass.",
    )
    parser.add_argument(
        "--home-empty-utility-recall-min",
        type=float,
        default=0.10,
        help="Minimum home-empty recall for operational utility pass.",
    )
    parser.add_argument(
        "--home-empty-utility-predicted-empty-rate-min",
        type=float,
        default=0.01,
        help="Minimum predicted-empty window rate for operational utility pass.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    reports = {
        "A0": Path(args.a0_report),
        "A1": Path(args.a1_report),
        "A2": Path(args.a2_report),
    }
    variants = {
        name: _extract_variant_metrics(
            name=name,
            report_path=path,
            full_bundle_livingroom_mae=float(args.full_bundle_lr_mae),
        )
        for name, path in reports.items()
    }

    a0 = variants["A0"]
    for name, payload in variants.items():
        if payload.bedroom_sleep_duration_mae_minutes is not None and a0.bedroom_sleep_duration_mae_minutes is not None:
            payload.bedroom_sleep_mae_delta_vs_a0_minutes = (
                float(payload.bedroom_sleep_duration_mae_minutes) - float(a0.bedroom_sleep_duration_mae_minutes)
            )

    # Core gates.
    lr_gate = all(
        v.lr_mae_improvement_vs_full_bundle_pct is not None and v.lr_mae_improvement_vs_full_bundle_pct >= 0.20
        for v in variants.values()
    )
    bedroom_gate = all(
        v.bedroom_sleep_mae_delta_vs_a0_minutes is not None and v.bedroom_sleep_mae_delta_vs_a0_minutes <= 2.0
        for v in variants.values()
    )
    hard_gate_non_regression = all(
        (v.hard_gate_checks_passed is not None and a0.hard_gate_checks_passed is not None and v.hard_gate_checks_passed >= a0.hard_gate_checks_passed)
        for v in variants.values()
    )

    # Minority anti-collapse gate (aggregated support-weighted recall over splits).
    room_status: Dict[str, str] = {}
    room_details: Dict[str, Any] = {}
    has_room_failure = False
    has_room_unknown = False
    for room in CORE_ROOMS:
        labels_payload = variants["A1"].minority_recall_by_room.get(room, {})
        eligible = list(labels_payload.get("eligible_nonmasked_labels", []))
        positive = list(labels_payload.get("positive_recall_labels", []))
        required_labels = int(ROOM_MIN_ELIGIBLE_LABELS.get(room, 2))
        if len(eligible) < required_labels:
            status = "not_evaluable"
            has_room_unknown = True
        elif len(positive) >= required_labels:
            status = "pass"
        else:
            status = "fail"
            has_room_failure = True
        room_status[room] = status
        room_details[room] = {
            "eligible_nonmasked_labels": eligible,
            "positive_recall_labels": positive,
            "required_min_eligible_labels": required_labels,
        }

    sleep_gate_status = variants["A1"].bedroom_sleep_recall.get("status", "not_evaluable")
    if sleep_gate_status == "fail":
        has_room_failure = True
    elif sleep_gate_status == "not_evaluable":
        has_room_unknown = True

    if has_room_failure:
        minority_gate_status = "fail"
    elif has_room_unknown:
        minority_gate_status = "unknown"
    else:
        minority_gate_status = "pass"

    # Occupancy safety gate vs A0 baseline.
    occupancy_variant_status: Dict[str, Any] = {}
    occupancy_unknown = False
    occupancy_failure = False
    for name, payload in variants.items():
        fe = payload.occupancy_false_empty_rate
        hp = payload.occupancy_home_empty_precision
        a0_fe = a0.occupancy_false_empty_rate
        a0_hp = a0.occupancy_home_empty_precision
        if fe is None or hp is None or a0_fe is None or a0_hp is None:
            occupancy_variant_status[name] = {"status": "unknown", "reason": "missing_occupancy_metrics"}
            occupancy_unknown = True
            continue
        fe_delta = fe - a0_fe
        hp_drop = a0_hp - hp
        status = (
            "pass"
            if (fe_delta <= 0.01 and fe <= 0.05 and hp_drop <= 0.02)
            else "fail"
        )
        occupancy_variant_status[name] = {
            "status": status,
            "false_empty_rate": fe,
            "false_empty_delta_vs_a0": fe_delta,
            "home_empty_precision": hp,
            "home_empty_precision_drop_vs_a0": hp_drop,
        }
        if status == "fail":
            occupancy_failure = True

    if occupancy_failure:
        occupancy_gate_status = "fail"
    elif occupancy_unknown:
        occupancy_gate_status = "unknown"
    else:
        # Evaluate A1/A2 (A0 is baseline).
        occupancy_gate_status = (
            "pass"
            if all(occupancy_variant_status.get(v, {}).get("status") == "pass" for v in ("A1", "A2"))
            else "fail"
        )

    utility_precision_min = float(max(args.home_empty_utility_precision_min, 0.0))
    utility_recall_min = float(max(args.home_empty_utility_recall_min, 0.0))
    utility_pred_empty_rate_min = float(max(args.home_empty_utility_predicted_empty_rate_min, 0.0))
    selected_variant_name_for_utility = "A1" if "A1" in variants else recommended_variant if "recommended_variant" in locals() else "A1"
    selected_variant_for_utility = variants.get(selected_variant_name_for_utility, variants["A1"])
    utility_precision = selected_variant_for_utility.occupancy_home_empty_precision
    utility_recall = selected_variant_for_utility.occupancy_home_empty_recall
    utility_pred_empty_rate = selected_variant_for_utility.occupancy_home_empty_predicted_empty_rate
    utility_ready = (
        utility_precision is not None
        and utility_recall is not None
        and utility_pred_empty_rate is not None
    )
    if not utility_ready:
        utility_status = "unknown"
        utility_reasons = ["missing_home_empty_utility_metrics"]
    else:
        utility_reasons = []
        if float(utility_precision) < utility_precision_min:
            utility_reasons.append(
                f"home_empty_precision_lt_{utility_precision_min:.2f}"
            )
        if float(utility_recall) < utility_recall_min:
            utility_reasons.append(
                f"home_empty_recall_lt_{utility_recall_min:.2f}"
            )
        if float(utility_pred_empty_rate) < utility_pred_empty_rate_min:
            utility_reasons.append(
                f"predicted_empty_rate_lt_{utility_pred_empty_rate_min:.3f}"
            )
        utility_status = "pass" if not utility_reasons else "fail"

    def _core_variant_pass(name: str) -> bool:
        v = variants[name]
        occ = occupancy_variant_status.get(name, {})
        return bool(
            v.lr_mae_improvement_vs_full_bundle_pct is not None
            and v.lr_mae_improvement_vs_full_bundle_pct >= 0.20
            and v.bedroom_sleep_mae_delta_vs_a0_minutes is not None
            and v.bedroom_sleep_mae_delta_vs_a0_minutes <= 2.0
            and v.hard_gate_checks_passed is not None
            and a0.hard_gate_checks_passed is not None
            and v.hard_gate_checks_passed >= a0.hard_gate_checks_passed
            and occ.get("status") == "pass"
        )

    # Plan rule: if A1 fails and A2 passes, choose A2.
    a1_core = _core_variant_pass("A1")
    a2_core = _core_variant_pass("A2")
    if (not a1_core) and a2_core:
        recommended_variant = "A2"
    elif a1_core:
        recommended_variant = "A1"
    else:
        recommended_variant = min(
            ("A0", "A1", "A2"),
            key=lambda n: (
                float("inf")
                if variants[n].livingroom_active_mae_minutes is None
                else float(variants[n].livingroom_active_mae_minutes)
            ),
        )

    all_core_pass = bool(lr_gate and bedroom_gate and hard_gate_non_regression and occupancy_gate_status == "pass")
    if all_core_pass and minority_gate_status == "pass":
        decision_status = "GO"
        reason_codes = ["STEP1_ALL_GATES_PASS"]
    elif all_core_pass and minority_gate_status == "unknown":
        decision_status = "NO_GO_INCOMPLETE_EVIDENCE"
        reason_codes = [
            "STEP1_CORE_GATES_PASS",
            "MISSING_OR_INSUFFICIENT_MINORITY_RECALL_EVIDENCE",
        ]
    else:
        decision_status = "NO_GO"
        reason_codes = ["STEP1_GATES_FAILED"]

    output_payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "step": "Step1_S1-03",
        "resident": str(args.resident),
        "seed": int(args.seed),
        "window": {"min_day": int(args.min_day), "max_day": int(args.max_day)},
        "reference": {
            "full_promoted_bundle_livingroom_mae_minutes": float(args.full_bundle_lr_mae),
            "baseline_anchor_livingroom_mae_minutes": float(args.baseline_anchor_lr_mae),
        },
        "variants": {name: _variant_to_dict(payload) for name, payload in variants.items()},
        "gate_evaluation": {
            "livingroom_mae_improve_vs_full_bundle_ge_20pct": {
                "status": "pass" if lr_gate else "fail",
                "required_min_improvement_pct": 0.20,
            },
            "bedroom_sleep_mae_delta_le_2_minutes": {
                "status": "pass" if bedroom_gate else "fail",
                "required_max_delta_minutes": 2.0,
            },
            "hard_gate_pass_count_non_regression": {
                "status": "pass" if hard_gate_non_regression else "fail",
                "baseline_a0_passed": a0.hard_gate_checks_passed,
            },
            "minority_recall_anti_collapse": {
                "status": minority_gate_status,
                "room_status": room_status,
                "room_details": room_details,
                "bedroom_sleep_recall": variants["A1"].bedroom_sleep_recall,
            },
            "occupancy_head_safety_non_regression": {
                "status": occupancy_gate_status,
                "variants": occupancy_variant_status,
                "policy": {
                    "false_empty_rate_delta_max": 0.01,
                    "false_empty_rate_cap": 0.05,
                    "home_empty_precision_drop_max": 0.02,
                },
                "note": "uses direct home_empty_summary when available; otherwise inferred occupancy confusion fallback",
            },
            "home_empty_operational_utility": {
                "status": utility_status,
                "evaluated_variant": selected_variant_name_for_utility,
                "metrics": {
                    "home_empty_precision": utility_precision,
                    "home_empty_recall": utility_recall,
                    "predicted_empty_rate": utility_pred_empty_rate,
                    "predicted_empty_windows": selected_variant_for_utility.occupancy_home_empty_predicted_empty_windows,
                    "total_windows": selected_variant_for_utility.occupancy_home_empty_windows_total,
                },
                "policy": {
                    "home_empty_precision_min": utility_precision_min,
                    "home_empty_recall_min": utility_recall_min,
                    "predicted_empty_rate_min": utility_pred_empty_rate_min,
                },
                "reasons": utility_reasons,
            },
        },
        "recommended_variant": recommended_variant,
        "decision": {
            "status": decision_status,
            "reason_codes": reason_codes,
        },
        "operational_recommendation": {
            "status": "CANARY_ELIGIBLE" if utility_status == "pass" else "SHADOW_ONLY",
            "reason": (
                "home_empty_utility_gate_passed"
                if utility_status == "pass"
                else "home_empty_utility_gate_not_passed"
            ),
            "evaluated_variant": selected_variant_name_for_utility,
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output_payload, indent=2))

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_build_markdown(output_payload))

    print(f"Wrote Step1 summary JSON: {output_json}")
    print(f"Wrote Step1 summary Markdown: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
