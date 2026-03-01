#!/usr/bin/env python3
"""
PR-5: Deterministic Manifest-Based Retrain

Performs deterministic retraining based on a manifest file.
Ensures reproducibility by locking:
- Data manifest (file paths, hashes)
- Code version (git commit)
- Policy configuration
- Random seed

Usage:
    python deterministic_retrain.py create --elder-id HK001 --data-files file1.parquet --output manifest.json
    python deterministic_retrain.py run --manifest manifest.json --output-dir results/
"""

import argparse
import json
import hashlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add backend and repo root to import path so both `ml.*` and `backend.*` resolve.
_SCRIPT_PATH = Path(__file__).resolve()
_BACKEND_DIR = _SCRIPT_PATH.parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
for _p in (str(_REPO_ROOT), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml.reproducibility_report import get_code_version

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _build_lane_b_dataframe(
    room_results: Dict[str, Any],
    *,
    prediction: bool,
) -> Optional["pd.DataFrame"]:
    """Normalize room-wise DataFrames into Lane B input schema."""
    import pandas as pd

    records = []
    for room_name, df in (room_results or {}).items():
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if "timestamp" not in df.columns:
            continue

        label_col = None
        if prediction:
            for candidate in ("predicted_label", "predicted_activity", "activity", "label"):
                if candidate in df.columns:
                    label_col = candidate
                    break
        else:
            for candidate in ("label", "activity", "predicted_activity", "predicted_label"):
                if candidate in df.columns:
                    label_col = candidate
                    break
        if label_col is None:
            continue

        part = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df["timestamp"], errors="coerce"),
                "room": room_name,
                ("predicted_label" if prediction else "label"): df[label_col].astype(str),
            }
        ).dropna(subset=["timestamp"])
        if not part.empty:
            records.append(part)

    if not records:
        return None
    return pd.concat(records, ignore_index=True)


def _evaluate_lane_b_outputs(
    prediction_results: Dict[str, Any],
    ground_truth_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate Lane B KPI + gates on runtime outputs.

    Returns a summary dict safe for persistence in deterministic retrain artifacts.
    """
    try:
        import pandas as pd
        from ml.event_gates import EventGateChecker
        from ml.event_kpi import EventKPICalculator
    except Exception as exc:
        return {"status": "not_available", "error": str(exc)}

    pred_df = _build_lane_b_dataframe(prediction_results, prediction=True)
    gt_df = _build_lane_b_dataframe(ground_truth_results, prediction=False)
    if pred_df is None or pred_df.empty:
        return {"status": "not_evaluated", "reason": "no_prediction_rows"}

    calc = EventKPICalculator()
    metrics = calc.calculate_metrics(pred_df, gt_df if gt_df is not None and not gt_df.empty else None)
    target_date = pd.to_datetime(pred_df["timestamp"]).dt.date.iloc[0]
    report = EventGateChecker().check_all_gates(metrics.to_gate_metrics(), target_date)

    return {
        "status": "evaluated",
        "target_date": str(target_date),
        "overall_status": report.overall_status.value,
        "is_promotable": bool(report.is_promotable),
        "critical_failures": list(report.critical_failures),
        "pass_rate": float(report.pass_rate),
        "metrics": metrics.to_gate_metrics(),
    }


class DeterministicManifest:
    """
    Deterministic training manifest.
    
    Locks all variables that could affect training outcome.
    """
    
    def __init__(self, manifest_path: str):
        """Load manifest from file."""
        self.path = Path(manifest_path)
        self.data = json.loads(self.path.read_text())
        
    @property
    def elder_id(self) -> str:
        return self.data["elder_id"]
    
    @property
    def data_files(self) -> List[Dict[str, Any]]:
        return self.data.get("data_files", [])
    
    @property
    def policy_config(self) -> Dict[str, Any]:
        return self.data.get("policy", {})
    
    @property
    def random_seed(self) -> int:
        return self.data.get("random_seed", 42)
    
    @property
    def expected_code_version(self) -> Dict[str, Any]:
        return self.data.get("code_version", {})
    
    def verify_code_version(self) -> tuple[bool, str]:
        """
        Verify current code matches expected version.
        
        Returns:
            (matches, message)
        """
        current = get_code_version()
        expected = self.expected_code_version
        
        if not expected:
            return True, "No expected version specified"
        
        if current.git_dirty:
            return False, f"Code has uncommitted changes"
        
        if expected.get("git_commit") != current.git_commit:
            return False, (
                f"Git commit mismatch: expected {expected.get('git_commit', 'unknown')[:8]}, "
                f"got {current.git_commit[:8]}"
            )
        
        return True, f"Code version matches: {current.git_commit[:8]}"
    
    def verify_data_files(self) -> tuple[bool, List[str]]:
        """
        Verify data files exist and match expected hashes.
        
        Returns:
            (all_valid, issues)
        """
        issues = []
        
        for file_info in self.data_files:
            file_path = Path(file_info["path"])
            expected_hash = file_info.get("hash", "")
            
            if not file_path.exists():
                issues.append(f"Missing: {file_path}")
                continue
            
            if expected_hash:
                actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]
                if actual_hash != expected_hash:
                    issues.append(
                        f"Hash mismatch: {file_path} "
                        f"(expected {expected_hash}, got {actual_hash})"
                    )
        
        return len(issues) == 0, issues
    
    def compute_fingerprint(self) -> str:
        """Compute deterministic fingerprint of manifest."""
        payload = json.dumps(self.data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def create_manifest(
    elder_id: str,
    data_files: List[str],
    output_path: str,
    policy_override: Optional[Dict] = None,
    random_seed: int = 42,
) -> str:
    """
    Create a deterministic training manifest.
    
    Args:
        elder_id: Elder/resident ID
        data_files: List of data file paths
        output_path: Where to write manifest
        policy_override: Optional policy overrides
        random_seed: Random seed for reproducibility
        
    Returns:
        Path to created manifest
    """
    code_version = get_code_version()
    
    # Compute hashes for data files
    file_entries = []
    for file_path in data_files:
        path = Path(file_path)
        if path.exists():
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            file_entries.append({
                "path": str(path.absolute()),
                "hash": file_hash,
                "size_bytes": path.stat().st_size,
            })
        else:
            logger.warning(f"Data file not found: {path}")
            file_entries.append({
                "path": str(path.absolute()),
                "hash": "",
                "size_bytes": 0,
            })
    
    manifest = {
        "version": "1.0",
        "created_at": datetime.utcnow().isoformat(),
        "elder_id": elder_id,
        "code_version": {
            "git_commit": code_version.git_commit,
            "git_branch": code_version.git_branch,
            "git_dirty": code_version.git_dirty,
            "python_version": code_version.python_version,
        },
        "random_seed": random_seed,
        "data_files": file_entries,
        "policy": policy_override or {},
    }
    
    output = Path(output_path)
    output.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Created manifest: {output}")
    
    return str(output)


def run_deterministic_retrain(
    manifest_path: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run deterministic retrain based on manifest.
    
    Args:
        manifest_path: Path to manifest file
        output_dir: Where to store results (default: current dir)
        
    Returns:
        Run results including paths to artifacts
    """
    manifest = DeterministicManifest(manifest_path)
    
    # Verify code version
    code_ok, code_msg = manifest.verify_code_version()
    logger.info(f"Code version check: {code_msg}")
    if not code_ok:
        raise RuntimeError(f"Code version mismatch: {code_msg}")
    
    # Verify data files
    data_ok, data_issues = manifest.verify_data_files()
    if not data_ok:
        for issue in data_issues:
            logger.error(f"Data verification: {issue}")
        raise RuntimeError("Data verification failed")
    logger.info("Data verification passed")
    
    # Set random seed for reproducibility
    import random
    import numpy as np
    
    random.seed(manifest.random_seed)
    np.random.seed(manifest.random_seed)
    
    # Apply policy overrides (manifest lock) before pipeline initialization.
    policy_overrides = manifest.policy_config or {}
    original_env: Dict[str, Optional[str]] = {}
    applied_overrides: Dict[str, str] = {}
    for key, value in policy_overrides.items():
        if isinstance(value, (str, int, float, bool)):
            env_key = str(key)
            original_env[env_key] = os.environ.get(env_key)
            os.environ[env_key] = str(value)
            applied_overrides[env_key] = str(value)
        else:
            logger.warning(
                "Skipping non-scalar policy override %s (type=%s)",
                key,
                type(value).__name__,
            )
    
    # Perform actual training using UnifiedPipeline
    from ml.pipeline import UnifiedPipeline
    
    fingerprint = manifest.compute_fingerprint()
    
    # Collect results from all data files
    all_trained_rooms = []
    all_decision_traces = {}
    all_rejection_artifacts = {}
    lane_b_runtime_reports = {}
    
    try:
        valid_files = [f for f in manifest.data_files if Path(f.get("path", "")).exists()]
        if not valid_files:
            raise RuntimeError("No valid data files available after manifest verification")

        pipeline = UnifiedPipeline(enable_denoising=True)
        
        for file_info in valid_files:
            file_path = file_info["path"]
            logger.info(f"Training on: {file_path}")
            
            # Run training
            corrected_results, trained_rooms = pipeline.train_and_predict(
                file_path=file_path,
                elder_id=manifest.elder_id,
            )
            lane_b_runtime_reports[file_path] = _evaluate_lane_b_outputs(
                corrected_results,
                corrected_results,  # training-mode path uses labels as GT on output frames
            )
            
            # Collect results
            for room_metrics in trained_rooms:
                room_name = room_metrics.get("room", "unknown")
                all_trained_rooms.append(room_metrics)
                
                # Collect decision trace path (prefer latest pointer)
                trace_info = room_metrics.get("decision_trace")
                if isinstance(trace_info, dict):
                    trace_path = trace_info.get("latest") or trace_info.get("versioned")
                    if trace_path:
                        all_decision_traces[room_name] = str(trace_path)
                elif isinstance(trace_info, str):
                    all_decision_traces[room_name] = trace_info
                
                # Collect explicit rejection artifacts emitted by training
                rejection_path = room_metrics.get("rejection_artifact_path")
                if rejection_path and Path(rejection_path).exists():
                    all_rejection_artifacts[room_name] = str(rejection_path)
        
        training_status = "completed"
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        training_status = f"failed: {e}"
        raise RuntimeError(f"Deterministic retrain failed: {e}") from e
    finally:
        for env_key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = original_value
    
    result = {
        "manifest_path": str(manifest_path),
        "manifest_fingerprint": fingerprint,
        "elder_id": manifest.elder_id,
        "timestamp": datetime.utcnow().isoformat(),
        "code_version_verified": code_ok,
        "data_verified": data_ok,
        "random_seed": manifest.random_seed,
        "policy_overrides_applied": applied_overrides,
        "status": training_status,
        "trained_rooms": all_trained_rooms,
        "decision_traces": all_decision_traces,
        "rejection_artifacts": all_rejection_artifacts,
        "lane_b_runtime_reports": lane_b_runtime_reports,
    }
    
    # Save result
    output_dir = Path(output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = output_dir / f"retrain_result_{fingerprint}.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info(f"Saved result: {result_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="PR-5: Deterministic Manifest-Based Retrain"
    )
    subparsers = parser.add_subparsers(dest="command")
    
    # Create manifest command
    create_parser = subparsers.add_parser("create", help="Create training manifest")
    create_parser.add_argument("--elder-id", required=True)
    create_parser.add_argument("--data-files", nargs="+", required=True)
    create_parser.add_argument("--output", required=True)
    create_parser.add_argument("--random-seed", type=int, default=42)
    
    # Run retrain command
    run_parser = subparsers.add_parser("run", help="Run deterministic retrain")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--output-dir", default=".")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify manifest without training")
    verify_parser.add_argument("--manifest", required=True)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == "create":
        create_manifest(
            elder_id=args.elder_id,
            data_files=args.data_files,
            output_path=args.output,
            random_seed=args.random_seed,
        )
        return 0
    
    elif args.command == "run":
        try:
            result = run_deterministic_retrain(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
            )
            print(json.dumps(result, indent=2))
            return 0
        except RuntimeError as e:
            logger.error(f"Retrain failed: {e}")
            return 1
    
    elif args.command == "verify":
        manifest = DeterministicManifest(args.manifest)
        
        code_ok, code_msg = manifest.verify_code_version()
        print(f"Code version: {'✅' if code_ok else '❌'} {code_msg}")
        
        data_ok, data_issues = manifest.verify_data_files()
        print(f"Data files: {'✅' if data_ok else '❌'}")
        for issue in data_issues:
            print(f"  - {issue}")
        
        print(f"\nManifest fingerprint: {manifest.compute_fingerprint()}")
        
        return 0 if (code_ok and data_ok) else 1
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
