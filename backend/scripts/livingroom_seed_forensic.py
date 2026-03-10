from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _models_dir(backend_dir: str | Path, elder_id: str) -> Path:
    return Path(backend_dir).expanduser().resolve() / "models" / str(elder_id)


def _policy_without_random_seed(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(policy or {}))
    reproducibility = normalized.get("reproducibility")
    if isinstance(reproducibility, dict):
        reproducibility.pop("random_seed", None)
    return normalized


def _threshold_delta(
    current_thresholds: Mapping[str, Any],
    winner_thresholds: Mapping[str, Any],
) -> dict[str, float]:
    delta: dict[str, float] = {}
    keys = set(current_thresholds.keys()) | set(winner_thresholds.keys())
    for key in sorted(keys):
        current_value = float(current_thresholds.get(key, 0.0))
        winner_value = float(winner_thresholds.get(key, 0.0))
        delta[str(key)] = current_value - winner_value
    return delta


def _distribution_delta(
    current_distribution: Mapping[str, Any],
    winner_distribution: Mapping[str, Any],
) -> dict[str, int]:
    delta: dict[str, int] = {}
    keys = set(current_distribution.keys()) | set(winner_distribution.keys())
    for key in sorted(keys):
        delta[str(key)] = int(current_distribution.get(key, 0)) - int(winner_distribution.get(key, 0))
    return delta


def _version_entry_by_id(versions_payload: Mapping[str, Any], version: int) -> dict[str, Any]:
    for entry in versions_payload.get("versions", []):
        if int(entry.get("version", 0)) == int(version):
            return dict(entry)
    raise ValueError(f"Version {int(version)} not found in versions payload")


def _best_summary_activity_split(best_summary: Mapping[str, Any]) -> tuple[int, int]:
    distribution = best_summary.get("predicted_class_distribution") or {}
    if not isinstance(distribution, dict):
        return 0, 0
    unoccupied = int(distribution.get("unoccupied", 0))
    active = sum(int(value) for key, value in distribution.items() if str(key) != "unoccupied")
    return active, unoccupied


def _summarize_version(
    *,
    models_dir: Path,
    room_name: str,
    version: int,
    winner_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    trace = _load_json(models_dir / f"{room_name}_v{int(version)}_decision_trace.json")
    calibrator = _load_json(models_dir / f"{room_name}_v{int(version)}_activity_confidence_calibrator.json")
    policy = trace.get("policy") or {}

    metrics = trace.get("metrics") or {}
    checkpoint_selection = metrics.get("checkpoint_selection") or {}
    best_summary = checkpoint_selection.get("best_summary") or {}
    last_summary = checkpoint_selection.get("last_summary") or {}
    class_thresholds = {
        str(key): float(value)
        for key, value in (trace.get("class_thresholds") or {}).items()
    }
    active_count, unoccupied_count = _best_summary_activity_split(best_summary)
    reaches_no_regress_floor = bool(
        best_summary.get("passes_no_regress_floor")
        or checkpoint_selection.get("selection_mode") == "no_regress_floor"
        or checkpoint_selection.get("best_floor_epoch") is not None
    )

    summary = {
        "version": int(version),
        "seed": int(((policy.get("reproducibility") or {}).get("random_seed", 0))),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "selection_mode": str(checkpoint_selection.get("selection_mode", "")),
        "reaches_no_regress_floor": reaches_no_regress_floor,
        "fallback_selected": "fallback" in str(checkpoint_selection.get("selection_mode", "")),
        "collapsed_best_epoch": bool(best_summary.get("collapsed", False)),
        "active_heavy_best_epoch": active_count > unoccupied_count,
        "unoccupied_preserving_best_epoch": unoccupied_count >= active_count,
        "best_epoch": checkpoint_selection.get("best_epoch"),
        "best_floor_epoch": checkpoint_selection.get("best_floor_epoch"),
        "best_macro_epoch": checkpoint_selection.get("best_macro_epoch"),
        "no_regress_macro_f1_floor": best_summary.get("no_regress_macro_f1_floor"),
        "best_epoch_distribution": {
            str(key): int(value)
            for key, value in (best_summary.get("predicted_class_distribution") or {}).items()
        },
        "last_epoch_distribution": {
            str(key): int(value)
            for key, value in (last_summary.get("predicted_class_distribution") or {}).items()
        },
        "dominant_class_label": best_summary.get("dominant_class_label"),
        "gate_aligned_score": best_summary.get("gate_aligned_score"),
        "class_thresholds": class_thresholds,
        "validation_class_support": metrics.get("validation_class_support") or {},
        "holdout_class_support": metrics.get("holdout_class_support") or {},
        "train_class_support_post_minority_sampling": metrics.get("train_class_support_post_minority_sampling") or {},
        "post_sampling_prior_drift_pp": (
            ((metrics.get("minority_sampling") or {}).get("post_sampling_prior_drift") or {}).get("max_abs_drift_pp")
        ),
        "post_downsample_prior_drift_pp": (
            ((metrics.get("unoccupied_downsample") or {}).get("post_downsample_prior_drift") or {}).get("max_abs_drift_pp")
        ),
        "activity_confidence": {
            "intercept": float(calibrator.get("intercept", 0.0)),
            "coefficients": [float(value) for value in calibrator.get("coefficients", [])],
            "feature_names": [str(value) for value in calibrator.get("feature_names", [])],
        },
        "policy_signature": _policy_without_random_seed(policy),
    }

    if winner_summary is not None:
        winner_thresholds = winner_summary.get("class_thresholds") or {}
        winner_distribution = winner_summary.get("best_epoch_distribution") or {}
        winner_activity_confidence = winner_summary.get("activity_confidence") or {}
        summary["comparison_to_winner"] = {
            "macro_f1_delta": float(summary["macro_f1"]) - float(winner_summary.get("macro_f1", 0.0)),
            "class_threshold_delta": _threshold_delta(class_thresholds, winner_thresholds),
            "activity_confidence_intercept_delta": (
                float(summary["activity_confidence"]["intercept"])
                - float(winner_activity_confidence.get("intercept", 0.0))
            ),
            "best_epoch_distribution_delta": _distribution_delta(
                summary["best_epoch_distribution"],
                winner_distribution,
            ),
        }
    else:
        summary["comparison_to_winner"] = {
            "macro_f1_delta": 0.0,
            "class_threshold_delta": _threshold_delta(class_thresholds, class_thresholds),
            "activity_confidence_intercept_delta": 0.0,
            "best_epoch_distribution_delta": _distribution_delta(
                summary["best_epoch_distribution"],
                summary["best_epoch_distribution"],
            ),
        }

    return summary


def build_livingroom_seed_forensic(
    *,
    backend_dir: str | Path,
    elder_id: str,
    room_name: str,
    versions: Sequence[int],
    winner_version: int,
) -> dict[str, Any]:
    models_dir = _models_dir(backend_dir, elder_id)
    versions_payload = _load_json(models_dir / f"{room_name}_versions.json")
    winner_entry = _version_entry_by_id(versions_payload, int(winner_version))
    winner_summary = _summarize_version(
        models_dir=models_dir,
        room_name=room_name,
        version=int(winner_version),
        winner_summary=None,
    )

    version_summaries: list[dict[str, Any]] = []
    policy_signatures: list[dict[str, Any]] = []
    post_sampling_drifts: list[float] = []
    for version in versions:
        entry = _version_entry_by_id(versions_payload, int(version))
        summary = _summarize_version(
            models_dir=models_dir,
            room_name=room_name,
            version=int(version),
            winner_summary=winner_summary,
        )
        summary["created_at"] = entry.get("created_at")
        summary["is_winner"] = int(version) == int(winner_version)
        policy_signature = summary.get("policy_signature")
        summary["policy_matches_winner_except_random_seed"] = policy_signature == winner_summary.get(
            "policy_signature"
        )
        summary.pop("policy_signature", None)
        version_summaries.append(summary)
        if isinstance(policy_signature, dict):
            policy_signatures.append(policy_signature)
        drift_value = summary.get("post_sampling_prior_drift_pp")
        if drift_value is not None:
            post_sampling_drifts.append(float(drift_value))

    policies_match_except_random_seed = all(
        signature == policy_signatures[0] for signature in policy_signatures[1:]
    ) if policy_signatures else True

    drift_min = min(post_sampling_drifts) if post_sampling_drifts else None
    drift_max = max(post_sampling_drifts) if post_sampling_drifts else None

    return {
        "elder_id": str(elder_id),
        "room_name": str(room_name),
        "winner_version": int(winner_version),
        "winner_seed": int(winner_summary.get("seed", 0)),
        "winner_created_at": winner_entry.get("created_at"),
        "policies_match_except_random_seed": policies_match_except_random_seed,
        "post_sampling_prior_drift_pp_range": {
            "min": drift_min,
            "max": drift_max,
            "spread": None if drift_min is None or drift_max is None else float(drift_max - drift_min),
        },
        "versions": sorted(version_summaries, key=lambda item: int(item["version"])),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a LivingRoom seed panel from saved decision traces and confidence calibrators."
    )
    parser.add_argument("--backend-dir", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--elder-id", required=True)
    parser.add_argument("--room-name", default="LivingRoom")
    parser.add_argument("--version", action="append", type=int, required=True)
    parser.add_argument("--winner-version", type=int, required=True)
    parser.add_argument("--summary-out")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_livingroom_seed_forensic(
        backend_dir=str(args.backend_dir),
        elder_id=str(args.elder_id),
        room_name=str(args.room_name),
        versions=[int(version) for version in args.version],
        winner_version=int(args.winner_version),
    )
    rendered = json.dumps(payload, indent=2)
    if args.summary_out:
        out_path = Path(args.summary_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
