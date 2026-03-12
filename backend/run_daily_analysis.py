import time
import schedule
import logging
import sys
import os
import atexit
import json
import hashlib
import subprocess
import joblib
import fcntl
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import re
from dotenv import load_dotenv

# Determine project root (one level up from this script, since script is in backend/)
# Script: Beta_5.5/backend/run_daily_analysis.py
# Root:   Beta_5.5/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "automation.log"
LOCK_FILE = PROJECT_ROOT / "logs" / "run_daily_analysis.lock"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger("DailyAutomation_Beta3")

# Ensure backend is in path (Python does this automatically for script dir, but implicit is better than explicit hacks)
# current_dir = os.path.dirname(os.path.abspath(__file__))

# Load Env
load_dotenv()

from elderlycare_v1_16.config.settings import RAW_DATA_DIR, ARCHIVE_DATA_DIR, USE_POSTGRESQL
from config import get_release_gates_config
from db.database import db as dual_write_db
from ml.pipeline import UnifiedPipeline
from ml.release_gates import resolve_scheduled_threshold
from ml.registry import ModelRegistry
from ml.pilot_override_manager import PilotOverrideManager
from ml.policy_config import load_policy_from_env
from ml.policy_defaults import (
    get_hard_negative_risky_rooms_default,
    get_runtime_wf_min_minority_support_by_room,
    get_runtime_wf_min_minority_support_default,
)
from ml.hard_negative_mining import mine_hard_negative_windows
from ml.t80_rollout_manager import RolloutStage, T80RolloutManager
from ml.beta6.registry.gate_engine import GateEngine
from ml.beta6.registry.registry_v2 import RegistryV2
from ml.beta6.contracts.decisions import ReasonCode
from ml.beta6.contracts.events import DecisionEvent, EventType
from ml.beta6.label_policy_consistency import validate_label_policy_consistency
from ml.beta6.orchestrator import Beta6Orchestrator, PhaseGateError
from ml.beta6.evaluation.runtime_eval_parity import DecoderPolicy
from ml.beta6.serving.runtime_preflight import (
    validate_beta6_phase4_runtime_preflight,
    validate_beta6_phase4_runtime_preflight_cohort,
)
from ml.beta6.serving.serving_loader import run_daily_stability_certification
from ml.evaluation import (
    TimeCheckpointedSplitter,
    evaluate_baseline_version,
    evaluate_model_version,
    load_room_training_dataframe,
)
from elderlycare_v1_16.services.adl_service import ADLService
from elderlycare_v1_16.services.insight_service import InsightService
from process_data import process_file, get_elder_id_from_filename, archive_file
from ml.household_analyzer import HouseholdAnalyzer
from utils.segment_utils import regenerate_segments
from utils.data_loader import load_sensor_data
from utils.room_utils import normalize_room_name
from utils.elder_id_utils import (
    choose_canonical_elder_id as _choose_canonical_elder_id_impl,
    elder_id_lineage_matches as _elder_id_lineage_matches_impl,
    split_elder_id_code_and_name as _split_elder_id_code_and_name_impl,
)
import pandas as pd

TRAINING_EXTENSIONS = {".xlsx", ".xls", ".parquet"}
_LOCK_HANDLE = None
def _acquire_single_instance_lock() -> bool:
    """Prevent duplicate watcher instances from processing the same raw queue."""
    global _LOCK_HANDLE
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = open(LOCK_FILE, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip()
        logger.error(
            "Another watcher instance is already running (lock=%s, owner_pid=%s). Exiting.",
            LOCK_FILE,
            owner or "unknown",
        )
        handle.close()
        return False

    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    _LOCK_HANDLE = handle
    logger.info("Acquired watcher lock: %s", LOCK_FILE)
    return True


def _release_single_instance_lock() -> None:
    global _LOCK_HANDLE
    if _LOCK_HANDLE is None:
        return
    try:
        fcntl.flock(_LOCK_HANDLE.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        _LOCK_HANDLE.close()
    except Exception:
        pass
    _LOCK_HANDLE = None


def _is_ignored_raw_file(file_path: Path) -> bool:
    """Ignore transient/system files that should never enter training/inference."""
    name = file_path.name
    if name.startswith("~$"):  # Office lock/temp files
        return True
    if name.startswith("."):  # .DS_Store, dotfiles
        return True
    return False


def _is_training_file(file_path: Path) -> bool:
    """Return True if file should trigger model training."""
    if _is_ignored_raw_file(file_path):
        return False
    name = file_path.name.lower()
    return (
        "_train" in name
        and "_manual_" not in name
        and file_path.suffix.lower() in TRAINING_EXTENSIONS
    )


def _split_elder_id_code_and_name(elder_id: str) -> tuple[str, str]:
    return _split_elder_id_code_and_name_impl(elder_id)


def _elder_id_lineage_matches(expected_elder_id: str, candidate_elder_id: str) -> bool:
    return _elder_id_lineage_matches_impl(expected_elder_id, candidate_elder_id)


def _choose_canonical_elder_id(elder_ids: list[str]) -> str:
    """
    Pick deterministic canonical elder ID for one lineage.

    Canonical choice intentionally prefers the shortest numeric token to avoid
    creating parallel model namespaces from numeric-suffix drift.
    """
    return _choose_canonical_elder_id_impl(elder_ids)


def _training_file_priority(file_path: Path, incoming_resolved: set[str]) -> tuple[int, int, str, str]:
    """
    Deterministic precedence for duplicate-day aggregate selection.

    Priority (lowest wins):
    1. Incoming files beat archived files.
    2. Parquet beats Excel variants for equivalent identity.
    3. Name/path lexical fallback for deterministic tie-breaking.
    """
    resolved = str(file_path.resolve())
    source_rank = 0 if resolved in incoming_resolved else 1
    ext_rank = {".parquet": 0, ".xlsx": 1, ".xls": 2}.get(file_path.suffix.lower(), 99)
    return (source_rank, ext_rank, file_path.name.lower(), resolved)


def _training_identity_key(file_path: Path) -> str:
    """
    Build de-duplication identity for training files.
    Keeps resident/day identity stable even if elder ID prefix drifts (HK001/HK0011).
    """
    stem = file_path.stem.lower()
    if "_train_" in stem and "_" in stem:
        _, rest = stem.split("_", 1)
        return rest
    return stem


def _dedupe_training_files(candidates: list[Path], incoming_files: list[Path]) -> list[Path]:
    """De-duplicate training files by day identity with explicit deterministic precedence."""
    incoming_resolved = {str(Path(p).resolve()) for p in incoming_files}
    unique_by_identity: dict[str, Path] = {}
    for path in candidates:
        identity = _training_identity_key(path)
        existing = unique_by_identity.get(identity)
        if existing is None:
            unique_by_identity[identity] = path
            continue
        if _training_file_priority(path, incoming_resolved) < _training_file_priority(existing, incoming_resolved):
            unique_by_identity[identity] = path

    selected = list(unique_by_identity.values())
    selected.sort(key=lambda p: _training_file_priority(p, incoming_resolved))
    return selected


def _collect_archived_training_files(elder_id: str) -> list[Path]:
    """
    Collect archived training files for a resident.

    Includes both legacy Excel archives and Parquet archives.
    """
    if not ARCHIVE_DATA_DIR.exists():
        return []

    candidates: list[Path] = []
    for ext in TRAINING_EXTENSIONS:
        for file_path in ARCHIVE_DATA_DIR.rglob(f"*{ext}"):
            if _is_ignored_raw_file(file_path):
                continue
            if not file_path.is_file() or not _is_training_file(file_path):
                continue
            candidate_elder_id = get_elder_id_from_filename(file_path.name)
            if not _elder_id_lineage_matches(elder_id, candidate_elder_id):
                continue
            candidates.append(file_path)

    # De-duplicate on resolved path to avoid accidental duplicates.
    unique = {str(path.resolve()): path for path in candidates}
    return sorted(unique.values(), key=lambda p: str(p))


def _build_aggregate_training_set(elder_id: str, incoming_files: list[Path]) -> list[Path]:
    """
    Build aggregate training file list = archived history + current new files.

    De-duplicate using a logical file identity (name without extension) so the
    same training day is not trained twice when both archived parquet and
    incoming xlsx variants coexist in one run.
    """
    archived_files = _collect_archived_training_files(elder_id)
    merged = list(incoming_files) + archived_files
    return _dedupe_training_files(merged, incoming_files=incoming_files)


def _resolve_retrain_input_mode(default_mode: str = "auto_aggregate") -> str:
    """
    Resolve retrain input mode.

    Supported:
    - auto_aggregate (default): incoming + archive history
    - incoming_only: only current incoming batch
    - manifest_only: explicit list from RETRAIN_MANIFEST_PATH
    """
    allowed = {"auto_aggregate", "incoming_only", "manifest_only"}
    raw = str(os.getenv("RETRAIN_INPUT_MODE", default_mode)).strip().lower()
    if raw in allowed:
        return raw
    logger.warning(f"Invalid RETRAIN_INPUT_MODE={raw!r}; defaulting to {default_mode!r}.")
    return default_mode


def _resolve_release_gate_evidence_profile(default_profile: str = "production") -> str:
    raw = str(os.getenv("RELEASE_GATE_EVIDENCE_PROFILE", default_profile)).strip().lower()
    return raw or default_profile


def _enforce_retrain_mode_for_evidence_profile(retrain_mode: str) -> None:
    """
    Fail closed for pilot evidence profiles: retrain must be incoming-only.
    This keeps short-window evaluations attributable to the uploaded batch.
    """
    evidence_profile = _resolve_release_gate_evidence_profile(default_profile="production")
    if evidence_profile in {"pilot_stage_a", "pilot_stage_b"} and str(retrain_mode) != "incoming_only":
        raise ValueError(
            f"RETRAIN_INPUT_MODE must be incoming_only when "
            f"RELEASE_GATE_EVIDENCE_PROFILE={evidence_profile}; got {retrain_mode!r}"
        )


def _load_manifest_training_files(manifest_path: Path, elder_id: str) -> list[Path]:
    """
    Load explicit retrain manifest file.

    Manifest format:
    - one file path per line
    - blank lines and lines starting with '#' are ignored
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    selected: list[Path] = []
    for line in manifest_path.read_text().splitlines():
        raw = str(line).strip()
        if not raw or raw.startswith("#"):
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.exists() or not path.is_file():
            logger.warning(f"Manifest path missing, skipping: {path}")
            continue
        if not _is_training_file(path):
            logger.warning(f"Manifest path is not a training file, skipping: {path.name}")
            continue
        candidate_elder_id = get_elder_id_from_filename(path.name)
        if not _elder_id_lineage_matches(elder_id, candidate_elder_id):
            logger.warning(f"Manifest file elder mismatch, skipping: {path.name}")
            continue
        selected.append(path)
    return selected


def _resolve_training_files_for_run(elder_id: str, incoming_files: list[Path]) -> tuple[str, list[Path], dict]:
    """
    Resolve training files using configured retrain input mode.
    """
    mode = _resolve_retrain_input_mode(default_mode="auto_aggregate")
    _enforce_retrain_mode_for_evidence_profile(mode)
    if mode == "auto_aggregate":
        selected = _build_aggregate_training_set(elder_id, incoming_files)
        return mode, selected, {"manifest_path": None}

    if mode == "incoming_only":
        selected = _dedupe_training_files(list(incoming_files), incoming_files=list(incoming_files))
        return mode, selected, {"manifest_path": None}

    # mode == "manifest_only"
    manifest_raw = os.getenv("RETRAIN_MANIFEST_PATH", "").strip()
    if not manifest_raw:
        raise ValueError("RETRAIN_INPUT_MODE=manifest_only but RETRAIN_MANIFEST_PATH is empty")
    manifest_path = Path(manifest_raw)
    if not manifest_path.is_absolute():
        manifest_path = (PROJECT_ROOT / manifest_path).resolve()
    selected = _load_manifest_training_files(manifest_path, elder_id)
    selected = _dedupe_training_files(selected, incoming_files=selected)
    if not selected and incoming_files:
        logger.warning(
            "Manifest mode resolved 0 files for %s; falling back to auto_aggregate for this run.",
            elder_id,
        )
        _enforce_retrain_mode_for_evidence_profile("auto_aggregate")
        fallback = _build_aggregate_training_set(elder_id, incoming_files)
        return "auto_aggregate", fallback, {"manifest_path": str(manifest_path), "fallback": "auto_aggregate"}
    return mode, selected, {"manifest_path": str(manifest_path)}


def _ensure_beta6_authority_evidence_profile_default() -> None:
    """
    Deprecated no-op kept for compatibility.

    Beta 6.1 requires explicit authority env configuration and no longer
    injects an implicit evidence profile default at runtime.
    """
    return


def _check_beta6_authority_postgres_preflight() -> tuple[bool, dict]:
    if not bool(USE_POSTGRESQL):
        return False, {"error": "USE_POSTGRESQL=false"}

    pg_db = None
    conn = None
    try:
        pg_db = getattr(dual_write_db, "pg_db", None)
        if pg_db is None:
            return False, {"error": "PostgreSQL unavailable or failed to initialize"}

        conn = pg_db.get_raw_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        return True, {"status": "ok"}
    except Exception as exc:
        return False, {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        if pg_db is not None and conn is not None:
            try:
                pg_db.return_connection(conn)
            except Exception:
                pass


def _validate_beta6_authority_env_preflight() -> tuple[bool, dict]:
    report: dict = {"checks": []}

    def _push(name: str, ok: bool, details: dict | None = None) -> None:
        entry = {"check": str(name), "pass": bool(ok)}
        if details:
            entry["details"] = details
        report["checks"].append(entry)

    if not _is_beta6_authority_enabled():
        report["reason"] = "disabled"
        return True, report

    evidence_profile_raw = str(os.getenv("RELEASE_GATE_EVIDENCE_PROFILE", "")).strip().lower()
    evidence_profile_ok = bool(evidence_profile_raw)
    _push(
        "release_gate_evidence_profile_explicit",
        evidence_profile_ok,
        {"value": evidence_profile_raw or None},
    )
    if not evidence_profile_ok:
        report["reason"] = "release_gate_evidence_profile_missing"
        return False, report

    signing_key_raw = str(os.getenv("BETA6_GATE_SIGNING_KEY", "")).strip()
    signing_key_ok = bool(signing_key_raw)
    _push(
        "beta6_gate_signing_key_explicit",
        signing_key_ok,
        {"configured": signing_key_ok},
    )
    if not signing_key_ok:
        report["reason"] = "beta6_gate_signing_key_missing"
        return False, report

    postgres_ok, postgres_details = _check_beta6_authority_postgres_preflight()
    _push("postgres_reachable", postgres_ok, postgres_details)
    if not postgres_ok:
        report["reason"] = "postgres_unavailable"
        return False, report

    report["reason"] = "ok"
    return True, report


def _validate_beta6_training_preflight(elder_id: str, aggregate_files: list[Path]) -> tuple[bool, dict]:
    """
    Run fail-closed Beta 6 preflight checks for authority-path training.
    """
    report: dict = {"elder_id": str(elder_id), "checks": []}

    def _push(name: str, ok: bool, details: dict | None = None) -> None:
        entry = {"check": str(name), "pass": bool(ok)}
        if details:
            entry["details"] = details
        report["checks"].append(entry)

    env_ok, env_report = _validate_beta6_authority_env_preflight()
    for check in env_report.get("checks", []):
        report["checks"].append(dict(check))
    if not env_ok:
        report["reason"] = str(env_report.get("reason") or "authority_env_preflight_failed")
        return False, report

    # 1) Label policy consistency must pass.
    try:
        consistency = validate_label_policy_consistency(
            config_dir=Path(__file__).resolve().parent / "config",
            models_dir=Path(__file__).resolve().parent / "models",
            fail_on_warnings=False,
        )
        ok = str(consistency.status).lower() == "pass"
        _push(
            "label_policy_consistency",
            ok,
            {
                "status": str(consistency.status),
                "error_count": int(len(consistency.errors)),
                "warning_count": int(len(consistency.warnings)),
            },
        )
        if not ok:
            report["reason"] = "label_policy_consistency_failed"
            return False, report
    except Exception as exc:
        _push("label_policy_consistency", False, {"error": str(exc)})
        report["reason"] = "label_policy_consistency_exception"
        return False, report

    # 2) Aggregate set must not contain off-lineage files.
    mismatched_files: list[str] = []
    for file_path in aggregate_files:
        candidate = get_elder_id_from_filename(file_path.name)
        if not _elder_id_lineage_matches(elder_id, candidate):
            mismatched_files.append(str(file_path))
    lineage_ok = len(mismatched_files) == 0
    _push("aggregate_lineage_consistency", lineage_ok, {"mismatched_files": mismatched_files[:25]})
    if not lineage_ok:
        report["reason"] = "aggregate_lineage_mismatch"
        return False, report

    # 3) Guard against split resident IDs with same suffix in training corpus.
    _, target_name = _split_elder_id_code_and_name(elder_id)
    sibling_ids: set[str] = set()
    if target_name:
        for root in (RAW_DATA_DIR, ARCHIVE_DATA_DIR):
            if not root.exists():
                continue
            for ext in TRAINING_EXTENSIONS:
                for path in root.rglob(f"*{ext}"):
                    if not _is_training_file(path):
                        continue
                    candidate = get_elder_id_from_filename(path.name)
                    _, candidate_name = _split_elder_id_code_and_name(candidate)
                    if candidate_name != target_name:
                        continue
                    if _elder_id_lineage_matches(elder_id, candidate):
                        continue
                    sibling_ids.add(candidate)
    sibling_ok = len(sibling_ids) == 0
    _push("elder_id_suffix_collision", sibling_ok, {"sibling_ids": sorted(sibling_ids)})
    if not sibling_ok:
        report["reason"] = "elder_id_suffix_collision"
        return False, report

    report["reason"] = "ok"
    return True, report


def _validate_beta6_runtime_activation_preflight(
    elder_id: str,
    *,
    registry: ModelRegistry | None = None,
) -> tuple[bool, dict]:
    """
    Run fail-closed runtime activation preflight when Beta 6 runtime flags are active.
    """
    runtime_flags = {
        "BETA6_PHASE4_RUNTIME_ENABLED": _is_env_enabled("BETA6_PHASE4_RUNTIME_ENABLED", default=False),
        "ENABLE_BETA6_HMM_RUNTIME": _is_env_enabled("ENABLE_BETA6_HMM_RUNTIME", default=False),
        "ENABLE_BETA6_UNKNOWN_ABSTAIN_RUNTIME": _is_env_enabled(
            "ENABLE_BETA6_UNKNOWN_ABSTAIN_RUNTIME", default=False
        ),
        "ENABLE_BETA6_CRF_RUNTIME": _is_env_enabled("ENABLE_BETA6_CRF_RUNTIME", default=False),
    }
    runtime_mode = str(os.getenv("BETA6_SEQUENCE_RUNTIME_MODE", "")).strip().lower()
    runtime_flags_enabled = any(runtime_flags.values()) or runtime_mode in {"hmm", "crf"}

    if not runtime_flags_enabled:
        return True, {
            "elder_id": str(elder_id),
            "runtime_flags": runtime_flags,
            "runtime_mode": runtime_mode or None,
            "reason": "runtime_flags_disabled",
        }

    target_cohort = sorted(
        {
            str(item).strip()
            for item in str(os.getenv("BETA6_RUNTIME_TARGET_COHORT", "")).split(",")
            if str(item).strip()
        }
    )

    registry_root = str(os.getenv("BETA6_REGISTRY_V2_ROOT", "")).strip()
    if not registry_root and registry is not None:
        backend_dir = getattr(registry, "backend_dir", None)
        if backend_dir:
            registry_root = str((Path(backend_dir).resolve() / "models_beta6_registry_v2").resolve())

    ok, report = validate_beta6_phase4_runtime_preflight(
        elder_id=str(elder_id),
        target_cohort=target_cohort or None,
        registry_root=registry_root or None,
        require_target_cohort_for_runtime_flags=True,
        allow_crf_canary=_is_env_enabled("BETA6_RUNTIME_ALLOW_CRF_CANARY", default=False),
    )
    if isinstance(report, dict):
        report.setdefault("elder_id", str(elder_id))
        report.setdefault("runtime_flags", runtime_flags)
        report.setdefault("runtime_mode", runtime_mode or None)
    return ok, report


def _missing_activity_mask(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return series.isna() | text.isin({"", "nan", "none"})


def _is_missing_activity_value(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    txt = str(value).strip().lower()
    return txt in {"", "nan", "none"}


def _fill_missing_activity_labels_temporal(
    df: pd.DataFrame,
    *,
    activity_col: str = "activity",
    timestamp_col: str = "timestamp",
    max_gap_seconds: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """
    Fill missing activity labels using short-gap temporal continuity.

    Policy:
    - Prefer forward/backward fill only within `max_gap_seconds`.
    - Remaining unresolved rows become 'unknown' (not 'inactive').
    """
    if df.empty or activity_col not in df.columns or timestamp_col not in df.columns:
        return df, {"ffill": 0, "bfill": 0, "unknown": 0}

    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    out = out[out[timestamp_col].notna()].copy()
    if out.empty:
        return out, {"ffill": 0, "bfill": 0, "unknown": 0}

    out = out.sort_values(timestamp_col, kind="stable").copy()
    filled = out[activity_col].astype(object).copy()
    missing = _missing_activity_mask(filled)

    counts = {"ffill": 0, "bfill": 0, "unknown": 0}
    if not missing.any():
        out[activity_col] = filled.astype(str).str.strip()
        return out, counts

    # Forward fill with gap cap.
    last_label = None
    last_ts = None
    for idx in out.index:
        cur_ts = out.at[idx, timestamp_col]
        cur_val = filled.at[idx]
        if not _is_missing_activity_value(cur_val):
            last_label = str(cur_val).strip()
            last_ts = cur_ts
            continue
        if last_label is None or pd.isna(cur_ts) or pd.isna(last_ts):
            continue
        if float((cur_ts - last_ts).total_seconds()) <= float(max_gap_seconds):
            filled.at[idx] = last_label
            counts["ffill"] += 1

    # Backfill leading short gaps with next known label.
    next_label = None
    next_ts = None
    for idx in reversed(out.index.tolist()):
        cur_ts = out.at[idx, timestamp_col]
        cur_val = filled.at[idx]
        if not _is_missing_activity_value(cur_val):
            next_label = str(cur_val).strip()
            next_ts = cur_ts
            continue
        if next_label is None or pd.isna(cur_ts) or pd.isna(next_ts):
            continue
        if float((next_ts - cur_ts).total_seconds()) <= float(max_gap_seconds):
            filled.at[idx] = next_label
            counts["bfill"] += 1

    unresolved = _missing_activity_mask(filled)
    if unresolved.any():
        counts["unknown"] = int(unresolved.sum())
        filled.loc[unresolved] = "unknown"

    out[activity_col] = filled.astype(str).str.strip()
    return out, counts


def _build_legacy_training_timeline_results(training_files: list[Path]) -> dict[str, pd.DataFrame]:
    """
    Build old-style timeline payload directly from training labels.

    Why:
    - `train_from_files()` may intentionally return empty prediction payloads.
    - Daily ops still needs ADL timeline rows for Correction Studio and legacy timeline UI.
    """
    if not training_files:
        return {}

    grouped: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for source_order, file_path in enumerate(training_files):
        try:
            loaded = load_sensor_data(file_path, resample=True)
        except TypeError:
            loaded = load_sensor_data(file_path)
        except Exception as exc:
            logger.warning(f"Skipping timeline fallback load for {file_path.name}: {exc}")
            continue

        for room_name, room_df in loaded.items():
            if not isinstance(room_df, pd.DataFrame) or room_df.empty:
                continue
            if "timestamp" not in room_df.columns or "activity" not in room_df.columns:
                continue
            copy_df = room_df.copy()
            fill_gap_seconds = max(10, int(os.getenv("LEGACY_ACTIVITY_FILL_MAX_GAP_SECONDS", "30")))
            copy_df, fill_stats = _fill_missing_activity_labels_temporal(
                copy_df,
                activity_col="activity",
                timestamp_col="timestamp",
                max_gap_seconds=fill_gap_seconds,
            )
            if any(int(fill_stats.get(k, 0)) > 0 for k in ("ffill", "bfill", "unknown")):
                logger.info(
                    f"Legacy timeline fallback label fill for {room_name} in {file_path.name}: "
                    f"ffill={int(fill_stats.get('ffill', 0))}, "
                    f"bfill={int(fill_stats.get('bfill', 0))}, "
                    f"unknown={int(fill_stats.get('unknown', 0))}, "
                    f"max_gap_seconds={fill_gap_seconds}"
                )
            copy_df["__source_file_order"] = int(source_order)
            copy_df["__source_row_order"] = list(range(len(copy_df)))
            copy_df["predicted_activity"] = copy_df["activity"]
            if "confidence" not in copy_df.columns:
                copy_df["confidence"] = 1.0
            grouped[room_name].append(copy_df)

    results: dict[str, pd.DataFrame] = {}
    for room_name, chunks in grouped.items():
        combined = pd.concat(chunks, ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
        combined = combined[combined["timestamp"].notna()].copy()
        if combined.empty:
            continue
        combined = combined.sort_values(
            ["timestamp", "__source_file_order", "__source_row_order"],
            kind="stable",
        )
        before = len(combined)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="first")
        dropped = before - len(combined)
        if dropped > 0:
            logger.info(
                f"Resolved {dropped} duplicate timeline rows for {room_name} "
                "using incoming-file precedence."
            )
        if "predicted_activity" not in combined.columns:
            combined["predicted_activity"] = combined["activity"]
        else:
            combined["predicted_activity"] = combined["predicted_activity"].fillna(combined["activity"])
        combined["confidence"] = pd.to_numeric(combined["confidence"], errors="coerce").fillna(1.0)
        combined = combined.drop(columns=["__source_file_order", "__source_row_order"], errors="ignore")
        results[room_name] = combined

    return results


def _resolve_scheduled_threshold(schedule: list[dict], training_days: float) -> float | None:
    """Backwards-compatible wrapper around shared release-gate threshold resolver."""
    return resolve_scheduled_threshold(schedule, training_days)


def _snapshot_current_versions(registry: ModelRegistry, elder_id: str) -> dict[str, int]:
    """Capture champion version per room before a training run."""
    versions: dict[str, int] = {}
    model_dir = registry.get_models_dir(elder_id)
    for versions_file in model_dir.glob("*_versions.json"):
        room_name = versions_file.name.replace("_versions.json", "")
        try:
            versions[room_name] = int(registry.get_current_version(elder_id, room_name))
        except Exception:
            versions[room_name] = 0
    return versions


def _list_registry_rooms(registry: ModelRegistry, elder_id: str) -> list[str]:
    """Discover all room names with any model artifacts or version metadata."""
    models_dir = registry.get_models_dir(elder_id)
    room_names: set[str] = set()
    for file_path in models_dir.iterdir():
        name = file_path.name
        if name.endswith("_versions.json"):
            room_names.add(name.replace("_versions.json", ""))
            continue
        if name.endswith("_model.keras"):
            if "_v" in name:
                m = re.match(r"^(.*)_v\d+_model\.keras$", name)
                if m:
                    room_names.add(m.group(1))
            else:
                room_names.add(name.replace("_model.keras", ""))
    return sorted(room_names)


def _emit_registry_integrity_summary(
    registry: ModelRegistry,
    elder_ids: list[str] | None = None,
) -> str | None:
    """
    Run startup registry integrity audit/repair and persist a daily JSON summary artifact.
    """
    models_root = Path(registry.backend_dir) / "models"
    if elder_ids is None:
        if not models_root.exists():
            return None
        target_elders = sorted([p.name for p in models_root.iterdir() if p.is_dir()])
    else:
        target_elders = sorted({str(e).strip() for e in elder_ids if str(e).strip()})

    summary = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "elders_total": int(len(target_elders)),
        "rooms_total": 0,
        "rooms_invalid": 0,
        "rooms_repaired": 0,
        "elders": [],
    }
    for elder_id in target_elders:
        room_reports = []
        for room_name in _list_registry_rooms(registry, elder_id):
            report = registry.validate_and_repair_room_registry_state(elder_id, room_name)
            room_reports.append(
                {
                    "room": room_name,
                    "valid": bool(report.get("valid", True)),
                    "repaired": bool(report.get("repaired", False)),
                    "issues": list(report.get("issues") or []),
                }
            )
        summary["rooms_total"] += int(len(room_reports))
        summary["rooms_invalid"] += int(sum(1 for r in room_reports if not r["valid"]))
        summary["rooms_repaired"] += int(sum(1 for r in room_reports if r["repaired"]))
        summary["elders"].append(
            {
                "elder_id": elder_id,
                "rooms_total": int(len(room_reports)),
                "rooms_invalid": int(sum(1 for r in room_reports if not r["valid"])),
                "rooms_repaired": int(sum(1 for r in room_reports if r["repaired"])),
                "rooms": room_reports,
            }
        )

    artifact_dir = PROJECT_ROOT / "logs" / "registry_integrity"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"registry_integrity_{pd.Timestamp.utcnow().strftime('%Y%m%d')}.json"
    artifact_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "Registry integrity summary saved: %s (rooms_total=%s, repaired=%s, invalid=%s)",
        artifact_path,
        summary["rooms_total"],
        summary["rooms_repaired"],
        summary["rooms_invalid"],
    )
    return str(artifact_path)


def _derive_run_failure_stage(
    *,
    global_gate_pass: bool,
    decision_trace_gate_pass: bool,
    walk_forward_stage_failed: bool,
    metrics: list[dict],
) -> str:
    """Derive canonical run failure stage from ordered gate precedence."""
    if not global_gate_pass:
        return "global_gate_failed"
    if not decision_trace_gate_pass:
        return "preprocessing_contract_failed"
    if walk_forward_stage_failed:
        return "walk_forward_failed"

    data_viability_failed = any(
        any(
            str(reason).startswith("data_viability_failed")
            or str(reason).startswith("insufficient_observed_days")
            or str(reason).startswith("insufficient_samples")
            or str(reason).startswith("insufficient_retained_ratio")
            or str(reason).startswith("excessive_gap_drop_ratio")
            or str(reason).startswith("insufficient_training_windows")
            for reason in (metric.get("gate_reasons") or [])
        )
        for metric in metrics
    )
    if data_viability_failed:
        return "data_viability_failed"

    statistical_validity_failed = any(
        any(
            str(reason).startswith("insufficient_validation_support")
            or str(reason).startswith("insufficient_calibration_support")
            or str(reason).startswith("calibration_low_support_fallback")
            or str(reason).startswith("train_metric_fallback_blocked")
            for reason in (metric.get("gate_reasons") or [])
        )
        for metric in metrics
    )
    if statistical_validity_failed:
        return "statistical_validity_failed"
    return "none"


def _resolve_code_version() -> str:
    raw = str(os.getenv("CODE_VERSION", "")).strip()
    if raw:
        return raw
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _compute_training_run_fingerprint(
    elder_id: str,
    retrain_mode: str,
    aggregate_files: list[Path],
) -> dict:
    """Stable fingerprint bundle for deterministic retrain idempotency."""
    policy = load_policy_from_env()
    policy_payload = json.dumps(policy.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
    policy_hash = hashlib.sha256(policy_payload.encode("utf-8")).hexdigest()
    code_version = _resolve_code_version()
    manifest = []
    for p in aggregate_files:
        resolved = p.resolve()
        try:
            st = resolved.stat()
            manifest.append(
                {
                    "path": str(resolved),
                    "size": int(st.st_size),
                    "mtime_ns": int(st.st_mtime_ns),
                }
            )
        except OSError:
            manifest.append({"path": str(resolved), "size": -1, "mtime_ns": -1})
    payload = {
        "elder_id": elder_id,
        "retrain_mode": str(retrain_mode),
        "policy_hash": policy_hash,
        "code_version": code_version,
        "manifest": manifest,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "fingerprint": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "policy_hash": policy_hash,
        "code_version": code_version,
    }


def _fingerprint_record_path(registry: ModelRegistry, elder_id: str) -> Path:
    return registry.get_models_dir(elder_id) / "_last_training_run.json"


def _load_last_run_fingerprint(registry: ModelRegistry, elder_id: str) -> dict:
    path = _fingerprint_record_path(registry, elder_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_last_run_fingerprint(
    registry: ModelRegistry,
    elder_id: str,
    *,
    fingerprint: str,
    policy_hash: str,
    code_version: str,
    retrain_mode: str,
    aggregate_files: list[Path],
    outcome: str,
) -> None:
    path = _fingerprint_record_path(registry, elder_id)
    payload = {
        "fingerprint": str(fingerprint),
        "policy_hash": str(policy_hash),
        "code_version": str(code_version),
        "retrain_mode": str(retrain_mode),
        "outcome": str(outcome),
        "manifest": [str(p.resolve()) for p in aggregate_files],
        "updated_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2))


def _evaluate_global_gate(metrics: list[dict]) -> tuple[bool, dict]:
    """
    Evaluate run-level global gate from release policy.

    Uses promoted room candidates only (`gate_pass=True`), then computes
    macro average of room macro_f1 values.
    """
    try:
        policy = get_release_gates_config()
    except Exception as e:
        return False, {
            "pass": False,
            "reason": f"gate_config_unavailable:{e}",
            "training_days": 0.0,
            "required": None,
            "actual_global_macro_f1": None,
        }
    global_policy = policy.get("release_gates", {}).get("global", {})
    schedule = global_policy.get("schedule", [])

    promoted = [m for m in metrics if bool(m.get("gate_pass", True))]
    f1_values = [float(m["macro_f1"]) for m in promoted if m.get("macro_f1") is not None]
    training_days_values = [float(m.get("training_days", 0.0)) for m in metrics]

    training_days = max(training_days_values) if training_days_values else 0.0
    required = _resolve_scheduled_threshold(schedule, training_days)

    if required is None:
        return False, {
            "pass": False,
            "reason": "global_schedule_unresolved",
            "training_days": training_days,
            "required": None,
            "actual_global_macro_f1": None,
        }

    if not f1_values:
        return False, {
            "pass": False,
            "reason": "no_promoted_metrics_for_global_gate",
            "training_days": training_days,
            "required": float(required),
            "actual_global_macro_f1": None,
            "rooms_evaluated": [],
        }

    actual = sum(f1_values) / len(f1_values)
    gate_pass = actual >= float(required)
    return gate_pass, {
        "pass": gate_pass,
        "training_days": training_days,
        "required": float(required),
        "actual_global_macro_f1": float(actual),
        "rooms_evaluated": [m.get("room") for m in promoted if m.get("macro_f1") is not None],
    }


def _is_non_destructive_global_gate_failure(report: dict | None) -> bool:
    """
    Return True when global gate failed for policy/config reasons and should not rollback/deactivate.
    """
    if not isinstance(report, dict):
        return False
    reason = str(report.get("reason") or "").strip().lower()
    return (
        reason.startswith("gate_config_unavailable")
        or reason == "global_schedule_unresolved"
    )


def _validate_decision_trace_artifacts(metrics: list[dict]) -> tuple[bool, dict]:
    """
    Ensure every trained candidate has persisted decision-trace artifacts.
    """
    missing_rooms: list[str] = []
    missing_details: list[dict] = []

    for metric in metrics or []:
        room = str(metric.get("room") or "")
        if not room:
            continue
        # Require traces for every saved room candidate.
        if metric.get("saved_version") is None:
            continue
        trace = metric.get("decision_trace")
        required_paths: list[str] = []
        if isinstance(trace, dict):
            versioned = trace.get("versioned")
            latest = trace.get("latest")
            if versioned:
                required_paths.append(str(versioned))
            if latest:
                required_paths.append(str(latest))
        elif isinstance(trace, str) and trace.strip():
            required_paths.append(trace.strip())

        room_missing: list[str] = []
        if not required_paths:
            room_missing.append("decision_trace_missing_in_metrics")
        else:
            for p in required_paths:
                if not Path(p).exists():
                    room_missing.append(p)

        if room_missing:
            missing_rooms.append(room)
            missing_details.append({"room": room, "missing": room_missing})

    gate_pass = len(missing_rooms) == 0
    return gate_pass, {
        "pass": gate_pass,
        "reason": "all_traces_present" if gate_pass else "missing_decision_trace_artifacts",
        "failed_rooms": missing_rooms,
        "details": missing_details,
    }


def _rollback_promoted_rooms_if_needed(
    registry: ModelRegistry,
    elder_id: str,
    metrics: list[dict],
    previous_versions: dict[str, int],
    registry_v2: RegistryV2 | None = None,
    run_id: str | None = None,
) -> tuple[list[str], list[str]]:
    """Rollback rooms promoted in this run back to their pre-run champions."""
    promoted_rooms = [
        str(metric.get("room"))
        for metric in metrics
        if metric.get("room") and bool(metric.get("gate_pass", False))
    ]
    return _rollback_rooms_by_name(
        registry=registry,
        elder_id=elder_id,
        room_names=promoted_rooms,
        previous_versions=previous_versions,
        registry_v2=registry_v2,
        run_id=run_id,
    )


def _rollback_rooms_by_name(
    registry: ModelRegistry,
    elder_id: str,
    room_names: list[str],
    previous_versions: dict[str, int],
    registry_v2: RegistryV2 | None = None,
    run_id: str | None = None,
) -> tuple[list[str], list[str]]:
    """Rollback explicit rooms to previous champions."""
    rolled_back: list[str] = []
    deactivated: list[str] = []
    effective_run_id = run_id or f"legacy_rollback_{elder_id}"
    for room in room_names:
        if not room:
            continue
        target = int(previous_versions.get(room, 0))
        if target <= 0:
            if registry.deactivate_current_version(elder_id, room):
                deactivated.append(room)
                if registry_v2 is not None:
                    registry_v2.append_event(
                        DecisionEvent(
                            event_type=EventType.PROMOTION_ROLLBACK,
                            run_id=effective_run_id,
                            elder_id=elder_id,
                            room=room,
                            reason_code=ReasonCode.ROLLBACK_MISSING_TARGET.value,
                            payload={"action": "deactivate_current_version"},
                        )
                    )
            else:
                logger.warning(
                    f"Failed to deactivate current champion for {elder_id}/{room} after global gate failure."
                )
            continue
        if registry.rollback_to_version(elder_id, room, target):
            rolled_back.append(room)
            if registry_v2 is not None:
                try:
                    registry_v2.rollback_to_previous(elder_id=elder_id, room=room, run_id=effective_run_id)
                except Exception as exc:
                    logger.warning(
                        f"RegistryV2 rollback mirror failed for {elder_id}/{room}: {exc}"
                    )
    return rolled_back, deactivated


def _promote_candidate_rooms(
    registry: ModelRegistry,
    elder_id: str,
    metrics: list[dict],
    previous_versions: dict[str, int] | None = None,
    registry_v2: RegistryV2 | None = None,
    run_id: str | None = None,
) -> tuple[list[str], list[str]]:
    """Promote saved candidate versions to latest after all run-level gates pass."""
    promoted: list[str] = []
    failed: list[str] = []
    effective_run_id = run_id or f"legacy_promotion_{elder_id}"
    for metric in metrics:
        room = str(metric.get("room") or "")
        if not room or not bool(metric.get("gate_pass", False)):
            continue
        saved_version = metric.get("saved_version")
        try:
            target_version = int(saved_version)
        except (TypeError, ValueError):
            failed.append(room)
            continue
        if target_version <= 0 or not registry.rollback_to_version(elder_id, room, target_version):
            failed.append(room)
            continue
        if registry_v2 is not None:
            try:
                registry_v2.promote_candidate(
                    elder_id=elder_id,
                    room=room,
                    run_id=effective_run_id,
                    candidate_id=f"legacy_v{target_version}",
                    metadata={
                        "legacy_saved_version": int(target_version),
                        "source": "legacy_registry",
                    },
                )
            except Exception as exc:
                logger.error(
                    f"RegistryV2 promotion mirror failed for {elder_id}/{room}: {exc}"
                )
                prev = int((previous_versions or {}).get(room, 0))
                if prev > 0:
                    registry.rollback_to_version(elder_id, room, prev)
                else:
                    registry.deactivate_current_version(elder_id, room)
                failed.append(room)
                continue
        promoted.append(room)
        metric["promoted_to_latest"] = True
    return promoted, failed


def _is_env_enabled(var_name: str, default: bool = False) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _env_int(var_name: str, default: int) -> int:
    raw = os.getenv(var_name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _env_float(var_name: str, default: float) -> float:
    raw = os.getenv(var_name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _is_beta6_authority_enabled() -> bool:
    """Beta 6 gate/registry authority switch (default ON)."""
    return _is_env_enabled("ENABLE_BETA6_AUTHORITY", default=True)


def _build_beta6_run_id(elder_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"beta6_daily_{elder_id}_{ts}"


def _resolve_beta6_registry_v2_root(registry: ModelRegistry) -> Path:
    raw = str(os.getenv("BETA6_REGISTRY_V2_ROOT", "")).strip()
    if raw:
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    return (Path(registry.backend_dir) / "models_beta6_registry_v2").resolve()


def _init_beta6_registry_v2(registry: ModelRegistry) -> RegistryV2 | None:
    if not _is_beta6_authority_enabled():
        return None
    root = _resolve_beta6_registry_v2_root(registry)
    root.mkdir(parents=True, exist_ok=True)
    return RegistryV2(root=root)


def _bootstrap_beta6_registry_v2_from_legacy(
    *,
    registry_v2: RegistryV2 | None,
    elder_id: str,
    previous_versions: dict[str, int],
) -> None:
    """
    Seed RegistryV2 pointers from legacy champions when missing.
    """
    if registry_v2 is None:
        return

    for room, version in previous_versions.items():
        room_txt = str(room).strip()
        if not room_txt:
            continue
        try:
            version_num = int(version)
        except (TypeError, ValueError):
            continue
        if version_num <= 0:
            continue
        existing = registry_v2.read_champion_pointer(elder_id, room_txt)
        if existing:
            continue
        pointer = {
            "elder_id": elder_id,
            "room": room_txt,
            "candidate_id": f"legacy_v{version_num}",
            "run_id": "bootstrap_legacy_registry",
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metadata": {
                "source": "legacy_registry",
                "legacy_version": version_num,
            },
        }
        registry_v2.update_champion_pointer(elder_id=elder_id, room=room_txt, pointer=pointer)


def _extract_beta6_timeline_metrics(metric: dict) -> dict | None:
    """
    Normalize timeline metrics into GateEngine input contract.
    """
    raw = metric.get("timeline_metrics")
    if isinstance(raw, dict):
        if "duration_mae_minutes" in raw and "fragmentation_rate" in raw:
            return {
                "duration_mae_minutes": float(raw["duration_mae_minutes"]),
                "fragmentation_rate": float(raw["fragmentation_rate"]),
            }
    shadow = metric.get("event_first_shadow")
    if isinstance(shadow, dict):
        nested = shadow.get("timeline_metrics")
        if isinstance(nested, dict) and "duration_mae_minutes" in nested and "fragmentation_rate" in nested:
            return {
                "duration_mae_minutes": float(nested["duration_mae_minutes"]),
                "fragmentation_rate": float(nested["fragmentation_rate"]),
            }
        if "duration_mae_minutes" in shadow and "fragmentation_rate" in shadow:
            return {
                "duration_mae_minutes": float(shadow["duration_mae_minutes"]),
                "fragmentation_rate": float(shadow["fragmentation_rate"]),
            }
    return None


def _build_beta6_room_gate_report(metric: dict) -> dict:
    """
    Convert legacy metric payload into GateEngine room report shape.
    """
    gate_reasons = [str(reason) for reason in (metric.get("gate_reasons") or [])]
    gate_watch_reasons = [str(reason) for reason in (metric.get("gate_watch_reasons") or [])]
    report = {
        "passed": bool(metric.get("gate_pass", False)),
        "metrics_passed": bool(metric.get("gate_pass", False)),
        "details": {
            "legacy_gate_reasons": gate_reasons,
            "legacy_gate_blocking_reasons": gate_reasons,
            "legacy_gate_watch_reasons": gate_watch_reasons,
        },
    }
    timeline = _extract_beta6_timeline_metrics(metric)
    if timeline is not None:
        report["timeline_metrics"] = timeline

    reason_blob = "|".join(gate_reasons).lower()
    report["data_viable"] = not any(
        token in reason_blob
        for token in (
            "insufficient_observed_days",
            "insufficient_samples",
            "excessive_gap_drop_ratio",
            "insufficient_training_windows",
            "data_viability",
        )
    )
    report["leakage"] = {
        "resident_overlap": "resident_overlap" in reason_blob,
        "time_overlap": "time_overlap" in reason_blob,
        "window_overlap": "window_overlap" in reason_blob,
    }
    return report


def _beta6_gate_signing_key(*, require_explicit: bool = False) -> str:
    """
    Resolve signing key for Beta 6 gate artifacts.

    Defaults to a deterministic Beta 6 fallback key only for non-live runs.
    Live authority runs must provide an explicit signing key.
    """
    raw = os.getenv("BETA6_GATE_SIGNING_KEY")
    if raw and str(raw).strip():
        return str(raw).strip()
    if require_explicit:
        raise RuntimeError(
            "BETA6_GATE_SIGNING_KEY is required for live Beta 6 authority signing (registry_v2 present)"
        )
    return "beta6-local-dev-signing-key"


def _beta6_gate_artifact_output_dir(elder_id: str, run_id: str, registry_v2: RegistryV2 | None) -> Path | None:
    """
    Persist Phase 4 signed artifacts only on live runs (registry_v2 present).
    Unit tests and dry runs keep artifacts in-memory to avoid filesystem side effects.
    """
    if registry_v2 is None:
        return None
    safe_elder = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(elder_id or "unknown"))
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id or "unknown"))
    return PROJECT_ROOT / "backend" / "tmp" / "beta6_gate_artifacts" / safe_elder / safe_run


def _beta6_phase4_runtime_policy_path(elder_id: str, registry_v2: RegistryV2 | None) -> Path | None:
    if registry_v2 is None:
        return None
    root = getattr(registry_v2, "root", None)
    if root is None:
        return None
    safe_elder = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(elder_id or "unknown"))
    return Path(root).resolve() / safe_elder / "_runtime" / "phase4_runtime_policy.json"


def _beta6_rollout_nightly_metrics_path(elder_id: str, registry_v2: RegistryV2 | None) -> Path | None:
    if registry_v2 is None:
        return None
    root = getattr(registry_v2, "root", None)
    if root is None:
        return None
    safe_elder = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(elder_id or "unknown"))
    return Path(root).resolve() / safe_elder / "_runtime" / "phase6_rollout_nightly_metrics.jsonl"


def _load_beta6_rollout_nightly_metrics(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _append_beta6_rollout_nightly_metric(
    *,
    elder_id: str,
    run_id: str,
    registry_v2: RegistryV2 | None,
    entry: dict,
    keep_last: int = 45,
) -> list[dict]:
    path = _beta6_rollout_nightly_metrics_path(elder_id, registry_v2)
    if path is None:
        return [dict(entry)]
    history = _load_beta6_rollout_nightly_metrics(path)
    filtered = []
    for row in history:
        if str(row.get("run_id") or "") == str(run_id):
            continue
        filtered.append(row)
    filtered.append(dict(entry))
    filtered = filtered[-max(1, int(keep_last)) :]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in filtered:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    tmp.replace(path)
    return filtered


def _to_optional_float(value) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_beta6_unknown_metrics(metric: dict) -> dict[str, float | None]:
    unknown_rate = _to_optional_float(metric.get("unknown_rate"))
    abstain_rate = _to_optional_float(metric.get("abstain_rate"))

    unknown_payload_candidates = []
    metrics_blob = metric.get("metrics")
    if isinstance(metrics_blob, dict):
        unknown_payload_candidates.append(metrics_blob.get("unknown"))
    decision_blob = metric.get("beta6_room_decision")
    if isinstance(decision_blob, dict):
        details = decision_blob.get("details")
        if isinstance(details, dict):
            unknown_payload_candidates.append(details.get("unknown"))
            if unknown_rate is None:
                unknown_rate = _to_optional_float(details.get("unknown_rate"))
            if abstain_rate is None:
                abstain_rate = _to_optional_float(details.get("abstain_rate"))
    unknown_payload_candidates.append(metric.get("unknown_metrics"))

    for payload in unknown_payload_candidates:
        if not isinstance(payload, dict):
            continue
        if unknown_rate is None:
            unknown_rate = _to_optional_float(payload.get("unknown_rate"))
        if abstain_rate is None:
            abstain_rate = _to_optional_float(payload.get("abstain_rate"))

    return {
        "unknown_rate": unknown_rate,
        "abstain_rate": abstain_rate,
    }


def _build_beta6_shadow_parity_snapshot(metrics: list[dict]) -> dict[str, float | None]:
    values = {
        "unknown_rate": [],
        "abstain_rate": [],
        "duration_mae_minutes": [],
        "fragmentation_rate": [],
    }
    for metric in metrics:
        unknown = _extract_beta6_unknown_metrics(metric)
        for key in ("unknown_rate", "abstain_rate"):
            val = _to_optional_float(unknown.get(key))
            if val is not None:
                values[key].append(val)
        timeline = _extract_beta6_timeline_metrics(metric)
        if isinstance(timeline, dict):
            duration = _to_optional_float(timeline.get("duration_mae_minutes"))
            fragmentation = _to_optional_float(timeline.get("fragmentation_rate"))
            if duration is not None:
                values["duration_mae_minutes"].append(duration)
            if fragmentation is not None:
                values["fragmentation_rate"].append(fragmentation)
    snapshot: dict[str, float | None] = {}
    for key, series in values.items():
        if not series:
            snapshot[key] = None
            continue
        snapshot[key] = float(sum(series) / float(len(series)))
    return snapshot


def _compute_beta6_shadow_parity_alerts(
    *,
    current: dict[str, float | None],
    previous_history: list[dict],
) -> dict:
    window_days = max(1, _env_int("BETA6_SHADOW_PARITY_WINDOW_DAYS", 7))
    drift_thresholds = {
        "unknown_rate": _env_float("BETA6_SHADOW_PARITY_UNKNOWN_RATE_DRIFT_MAX", 0.05),
        "abstain_rate": _env_float("BETA6_SHADOW_PARITY_ABSTAIN_RATE_DRIFT_MAX", 0.05),
        "duration_mae_minutes": _env_float(
            "BETA6_SHADOW_PARITY_DURATION_MAE_DRIFT_MAX_MINUTES", 3.0
        ),
        "fragmentation_rate": _env_float("BETA6_SHADOW_PARITY_FRAGMENTATION_DRIFT_MAX", 0.05),
    }
    recent = [row for row in previous_history if isinstance(row, dict)][-window_days:]
    if len(recent) < window_days:
        return {
            "status": "insufficient_history",
            "window_days": window_days,
            "history_points": len(recent),
            "thresholds": dict(drift_thresholds),
            "current": dict(current),
            "baseline": None,
            "drift_abs": {},
            "alerts": [],
        }

    baseline: dict[str, float | None] = {}
    alerts: list[dict] = []
    drift_abs: dict[str, float] = {}
    for key, threshold in drift_thresholds.items():
        series = [_to_optional_float(row.get(key)) for row in recent]
        series = [val for val in series if val is not None]
        if not series:
            baseline[key] = None
            alerts.append(
                {
                    "metric": key,
                    "severity": "critical",
                    "type": "missing_baseline",
                    "message": f"missing {key} baseline for {window_days}-run shadow parity window",
                }
            )
            continue
        baseline_value = float(sum(series) / float(len(series)))
        baseline[key] = baseline_value
        current_value = _to_optional_float(current.get(key))
        if current_value is None:
            alerts.append(
                {
                    "metric": key,
                    "severity": "critical",
                    "type": "missing_current",
                    "message": f"missing current {key} for shadow parity drift check",
                }
            )
            continue
        delta = abs(float(current_value) - float(baseline_value))
        drift_abs[key] = float(delta)
        if delta >= float(threshold):
            alerts.append(
                {
                    "metric": key,
                    "severity": "critical",
                    "type": "drift_exceeded",
                    "delta_abs": float(delta),
                    "threshold": float(threshold),
                    "current": float(current_value),
                    "baseline": float(baseline_value),
                    "message": (
                        f"{key} drift exceeded threshold: abs_delta={delta:.6f} "
                        f"(threshold={float(threshold):.6f})"
                    ),
                }
            )
    status = "critical" if alerts else "ok"
    return {
        "status": status,
        "window_days": window_days,
        "history_points": len(recent),
        "thresholds": dict(drift_thresholds),
        "current": dict(current),
        "baseline": baseline,
        "drift_abs": drift_abs,
        "alerts": alerts,
    }


def _publish_beta6_phase4_runtime_policy(
    *,
    metrics: list[dict],
    elder_id: str,
    run_id: str,
    registry_v2: RegistryV2 | None,
    beta6_gate_report: dict,
    beta6_fallback_summary: dict | None = None,
) -> dict:
    path = _beta6_phase4_runtime_policy_path(elder_id, registry_v2)
    if path is None:
        return {"status": "disabled", "reason": "registry_unavailable"}

    previous_payload = {}
    if path.exists():
        try:
            previous_payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            previous_payload = {}

    room_runtime = {}
    enabled_rooms = []
    for metric in metrics:
        room_raw = str(metric.get("room") or "").strip()
        if not room_raw:
            continue
        room = normalize_room_name(room_raw)
        gate_pass = bool(metric.get("gate_pass", False))
        decision = metric.get("beta6_room_decision")
        reason_code = str((decision or {}).get("reason_code") or (ReasonCode.PASS.value if gate_pass else ""))
        fallback_active = False
        fallback_reason_code = ""
        if registry_v2 is not None:
            try:
                state = registry_v2.read_fallback_state(elder_id, room_raw)
            except Exception:
                state = None
            if isinstance(state, dict):
                fallback_active = bool(state.get("active", False))
                fallback_reason_code = str(state.get("trigger_reason_code") or "")
        enabled = bool(gate_pass and not fallback_active)
        if enabled:
            enabled_rooms.append(room)
        room_runtime[room] = {
            "enable_phase4_runtime": enabled,
            "gate_pass": gate_pass,
            "reason_code": reason_code,
            "fallback_active": fallback_active,
            "fallback_reason_code": fallback_reason_code,
        }

    master_enabled = bool(enabled_rooms)
    current_shadow = _build_beta6_shadow_parity_snapshot(metrics)
    previous_history = previous_payload.get("history") if isinstance(previous_payload, dict) else None
    previous_history = previous_history if isinstance(previous_history, list) else []
    shadow_parity = _compute_beta6_shadow_parity_alerts(
        current=current_shadow,
        previous_history=previous_history,
    )

    history = list(previous_history)
    history.append(
        {
            "run_id": str(run_id),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **dict(current_shadow),
        }
    )
    history = history[-35:]

    policy_paths = {
        "unknown_policy_path": str((PROJECT_ROOT / "backend" / "config" / "beta6_unknown_policy.yaml").resolve()),
        "hmm_duration_policy_path": str(
            (PROJECT_ROOT / "backend" / "config" / "beta6_duration_prior_policy.yaml").resolve()
        ),
    }
    payload = {
        "schema_version": "beta6.phase4.runtime_policy.v1",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elder_id": str(elder_id),
        "source_run_id": str(run_id),
        "master_enabled": master_enabled,
        "room_runtime": room_runtime,
        "enabled_rooms": sorted(set(enabled_rooms)),
        "policy_paths": policy_paths,
        "gate": {
            "pass": bool((beta6_gate_report or {}).get("pass", False)),
            "reason_code": str((beta6_gate_report or {}).get("reason_code") or ""),
            "phase4_dynamic_gate": dict((beta6_gate_report or {}).get("phase4_dynamic_gate") or {}),
        },
        "fallback_summary": dict(beta6_fallback_summary or {}),
        "shadow_parity": shadow_parity,
        "history": history,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)

    if shadow_parity.get("status") == "critical":
        logger.error(
            "Beta 6 shadow parity drift alerts for %s (run=%s): %s",
            elder_id,
            run_id,
            shadow_parity.get("alerts", []),
        )

    return {
        "status": "ok",
        "path": str(path),
        "master_enabled": bool(master_enabled),
        "enabled_rooms": sorted(set(enabled_rooms)),
        "shadow_parity_status": str(shadow_parity.get("status", "")),
        "shadow_parity_alerts": list(shadow_parity.get("alerts", [])),
    }


def _apply_beta6_gate_authority(
    *,
    metrics: list[dict],
    elder_id: str,
    run_id: str,
    registry_v2: RegistryV2 | None,
) -> tuple[bool, dict]:
    """
    Enforce GateEngine decisions as the active room/run authority.
    """
    if not _is_beta6_authority_enabled():
        return True, {"pass": True, "reason": "disabled"}

    engine = GateEngine()
    orchestrator = Beta6Orchestrator(require_intake_artifact=False)
    room_decisions = []
    room_reports: list[dict] = []
    shadow_compare_rows: list[dict] = []
    for metric in metrics:
        room = str(metric.get("room") or "").strip()
        if not room:
            continue
        legacy_gate_pass = bool(metric.get("gate_pass", False))
        legacy_gate_reasons = [str(reason) for reason in (metric.get("gate_reasons") or [])]
        legacy_gate_watch_reasons = [str(reason) for reason in (metric.get("gate_watch_reasons") or [])]
        base_report = _build_beta6_room_gate_report(metric)
        base_report["room"] = normalize_room_name(room)
        parity_trace = metric.get("beta6_parity_trace")
        runtime_policy_cfg = metric.get("beta6_runtime_policy")
        eval_policy_cfg = metric.get("beta6_eval_policy")
        parity_checked = False
        if parity_trace is not None:
            parity_checked = True
            try:
                if not isinstance(parity_trace, list):
                    raise PhaseGateError(
                        reason_code=ReasonCode.FAIL_RUNTIME_EVAL_PARITY.value,
                        detail="beta6_parity_trace must be a list of trace steps",
                    )
                runtime_policy = DecoderPolicy(
                    spike_suppression=bool(
                        (runtime_policy_cfg or {}).get("spike_suppression", True)
                    )
                ) if isinstance(runtime_policy_cfg, dict) else DecoderPolicy()
                eval_policy = DecoderPolicy(
                    spike_suppression=bool(
                        (eval_policy_cfg or {}).get("spike_suppression", True)
                    )
                ) if isinstance(eval_policy_cfg, dict) else DecoderPolicy()
                orchestrator.run_phase1_parity_gate(
                    trace_steps=parity_trace,
                    runtime_policy=runtime_policy,
                    eval_policy=eval_policy,
                )
                base_report.setdefault("details", {})
                if isinstance(base_report["details"], dict):
                    base_report["details"]["runtime_eval_parity_checked"] = True
                    base_report["details"]["runtime_eval_parity_passed"] = True
            except Exception as exc:
                if isinstance(exc, PhaseGateError):
                    parity_reason = ReasonCode.FAIL_RUNTIME_EVAL_PARITY.value
                    parity_detail = str(exc.detail)
                    parity_contract_code = str(exc.reason_code or "")
                else:
                    parity_reason = ReasonCode.FAIL_RUNTIME_EVAL_PARITY.value
                    parity_detail = f"{type(exc).__name__}: {exc}"
                    parity_contract_code = ""
                base_report = {
                    "room": normalize_room_name(room),
                    "passed": False,
                    "metrics_passed": False,
                    "reason_code": parity_reason,
                    "details": {
                        "runtime_eval_parity_checked": True,
                        "runtime_eval_parity_passed": False,
                        "runtime_eval_parity_contract_reason": parity_contract_code,
                        "runtime_eval_parity_error": parity_detail,
                    },
                }
                logger.warning(
                    "Beta 6 parity gate failed for elder=%s room=%s: %s",
                    elder_id,
                    room,
                    parity_detail,
                )
        if not parity_checked:
            base_report.setdefault("details", {})
            if isinstance(base_report["details"], dict):
                base_report["details"].setdefault("runtime_eval_parity_checked", False)
                base_report["details"].setdefault("runtime_eval_parity_passed", None)

        room_reports.append(dict(base_report))
        decision = engine.decide_room(room=room, report=base_report)
        room_decisions.append(decision)
        shadow_compare_rows.append(
            {
                "room": normalize_room_name(room),
                "legacy_gate_pass": legacy_gate_pass,
                "legacy_gate_reasons": legacy_gate_reasons,
                "legacy_gate_watch_reasons": legacy_gate_watch_reasons,
                "beta6_gate_pass": bool(decision.passed),
                "beta6_reason_code": decision.reason_code.value,
                "beta6_details": dict(decision.details),
            }
        )

        metric["gate_pass"] = bool(decision.passed)
        metric["beta6_room_decision"] = {
            "passed": bool(decision.passed),
            "reason_code": decision.reason_code.value,
            "details": dict(decision.details),
        }

        reasons = [str(reason) for reason in (metric.get("gate_reasons") or [])]
        reasons = [reason for reason in reasons if not reason.startswith("beta6_reason:")]
        if not decision.passed:
            reasons.append(f"beta6_reason:{decision.reason_code.value}")
        metric["gate_reasons"] = list(dict.fromkeys(reasons))
        watch_reasons = [str(reason) for reason in (metric.get("gate_watch_reasons") or [])]
        metric["gate_watch_reasons"] = list(dict.fromkeys(watch_reasons))

        if registry_v2 is not None:
            registry_v2.append_event(
                DecisionEvent(
                    event_type=(
                        EventType.ROOM_GATE_PASSED if decision.passed else EventType.ROOM_GATE_FAILED
                    ),
                    run_id=run_id,
                    elder_id=elder_id,
                    room=room,
                    reason_code=decision.reason_code.value,
                    payload={"details": dict(decision.details)},
                )
            )

    run_decision = engine.decide_run(room_decisions)
    phase4_gate_artifacts: dict | None = None
    phase4_gate_error: str | None = None
    phase6_shadow_compare_artifact: dict | None = None
    phase6_shadow_compare_error: str | None = None
    phase6_rollout_ladder_report: dict = {"status": "disabled", "reason": "not_evaluated"}
    phase6_auto_rollback_report: dict = {"status": "disabled", "reason": "not_evaluated"}
    artifact_output_dir = _beta6_gate_artifact_output_dir(elder_id, run_id, registry_v2)
    require_explicit_signing_key = registry_v2 is not None
    try:
        phase4_gate_artifacts = orchestrator.run_phase4_dynamic_gate(
            room_reports=room_reports,
            run_id=run_id,
            elder_id=elder_id,
            signing_key=_beta6_gate_signing_key(require_explicit=require_explicit_signing_key),
            output_dir=artifact_output_dir,
        )
    except Exception as exc:
        phase4_gate_error = f"{type(exc).__name__}: {exc}"
        logger.exception(
            "Beta 6 Phase 4 dynamic gate artifact generation failed for elder=%s run_id=%s",
            elder_id,
            run_id,
        )
        if run_decision.passed:
            run_decision = run_decision.__class__(
                passed=False,
                reason_code=ReasonCode.FAIL_GATE_POLICY,
                room_decisions=list(run_decision.room_decisions),
                details={
                    **dict(run_decision.details),
                    "phase4_dynamic_gate_error": phase4_gate_error,
                    "phase4_dynamic_gate_failed": True,
                },
            )
    else:
        payload = phase4_gate_artifacts.get("run_decision", {}) if isinstance(phase4_gate_artifacts, dict) else {}
        stage4_payload_valid = (
            isinstance(phase4_gate_artifacts, dict)
            and isinstance(payload, dict)
            and isinstance(payload.get("details", {}), dict)
            and "passed" in payload
            and "reason_code" in payload
        )
        if not stage4_payload_valid:
            phase4_gate_error = "RuntimeError: invalid_phase4_dynamic_gate_artifacts_payload"
            logger.error(
                "Beta 6 Phase 4 dynamic gate returned invalid payload for elder=%s run_id=%s: %s",
                elder_id,
                run_id,
                phase4_gate_artifacts,
            )
            if run_decision.passed:
                run_decision = run_decision.__class__(
                    passed=False,
                    reason_code=ReasonCode.FAIL_GATE_POLICY,
                    room_decisions=list(run_decision.room_decisions),
                    details={
                        **dict(run_decision.details),
                        "phase4_dynamic_gate_error": phase4_gate_error,
                        "phase4_dynamic_gate_failed": True,
                    },
                )
            payload = {}
        payload_reason = str(payload.get("reason_code") or run_decision.reason_code.value)
        try:
            resolved_reason = ReasonCode(payload_reason)
        except ValueError:
            resolved_reason = run_decision.reason_code
        run_decision = run_decision.__class__(
            passed=bool(payload.get("passed", run_decision.passed)),
            reason_code=resolved_reason,
            room_decisions=list(run_decision.room_decisions),
            details={
                **dict(run_decision.details),
                **(dict(payload.get("details", {})) if isinstance(payload.get("details"), dict) else {}),
            },
        )

    shadow_compare_output_path = None
    if artifact_output_dir is not None:
        shadow_compare_output_path = artifact_output_dir / f"{run_id}_shadow_compare_report.json"
    unexplained_divergence_rate_max = _env_float(
        "BETA6_SHADOW_UNEXPLAINED_DIVERGENCE_RATE_MAX",
        0.05,
    )
    try:
        shadow_compare_result = orchestrator.run_phase6_shadow_compare(
            room_rows=shadow_compare_rows,
            run_id=run_id,
            elder_id=elder_id,
            signing_key=_beta6_gate_signing_key(require_explicit=require_explicit_signing_key),
            output_path=shadow_compare_output_path,
            unexplained_divergence_rate_max=unexplained_divergence_rate_max,
            metadata={
                "phase": "phase6_step6_1",
                "gate_pass": bool(run_decision.passed),
                "gate_reason_code": run_decision.reason_code.value,
            },
        )
    except Exception as exc:
        phase6_shadow_compare_error = f"{type(exc).__name__}: {exc}"
        logger.exception(
            "Beta 6 Phase 6 shadow compare generation failed for elder=%s run_id=%s",
            elder_id,
            run_id,
        )
    else:
        phase6_shadow_compare_artifact = {
            "status": str(shadow_compare_result.status),
            "divergence_count": int(shadow_compare_result.divergence_count),
            "unexplained_divergence_count": int(shadow_compare_result.unexplained_divergence_count),
            "divergence_rate": float(shadow_compare_result.divergence_rate),
            "unexplained_divergence_rate": float(shadow_compare_result.unexplained_divergence_rate),
            "unexplained_divergence_rate_max": float(unexplained_divergence_rate_max),
            "signature": str(shadow_compare_result.signature),
            "report_path": shadow_compare_result.report_path,
            "badges": list(shadow_compare_result.badges),
        }

    try:
        rollout_manager = T80RolloutManager()
        rollout_state = rollout_manager.get_state()
        rollout_stage = rollout_state.stage.value if rollout_state is not None else RolloutStage.SHADOW.value
        default_current_rung = 4 if rollout_stage == RolloutStage.FULL.value else 1
        current_rung = max(1, _env_int("BETA6_ROLLOUT_CURRENT_RUNG", default_current_rung))
        shadow_compare_status = str((phase6_shadow_compare_artifact or {}).get("status") or "").strip().lower()
        shadow_compare_completed = phase6_shadow_compare_error is None and bool(shadow_compare_status)
        drift_alerts_within_budget = bool(
            shadow_compare_completed and shadow_compare_status in {"ok", "watch"}
        )
        # Pipeline reliability must reflect authority-path execution health, not model-quality gate outcomes.
        pipeline_execution_success = (
            phase4_gate_error is None and phase6_shadow_compare_error is None
        )
        gate_summary = {
            "mandatory_metric_floors_pass": bool(run_decision.passed),
            "open_p0_incidents": max(0, _env_int("BETA6_OPEN_P0_INCIDENTS", 0)),
            "nightly_pipeline_success_rate": 1.0 if pipeline_execution_success else 0.0,
            "drift_alerts_within_budget": drift_alerts_within_budget,
            "timeline_hard_gates_all_rooms_pass": all(
                bool(decision.passed) for decision in run_decision.room_decisions
            ),
            "phase5_acceptance_pass": _is_env_enabled("BETA6_PHASE5_ACCEPTANCE_PASS", default=False),
        }
        ladder_result = rollout_manager.evaluate_ladder_progression(
            current_rung=current_rung,
            gate_summary=gate_summary,
        )
        phase6_rollout_ladder_report = {
            "status": "ok",
            "rollout_stage": rollout_stage,
            "current_rung": int(current_rung),
            "target_rung": int(ladder_result.target_rung),
            "can_advance": bool(ladder_result.can_advance),
            "blockers": list(ladder_result.blockers),
            "gate_summary": gate_summary,
        }

        room_tokens = sorted(
            {
                normalize_room_name(str(metric.get("room") or "").strip())
                for metric in metrics
                if str(metric.get("room") or "").strip()
            }
        )
        mae_regression_by_room: dict[str, float] = {}
        alert_precision_below = False
        worst_room_f1_below_floor = False
        f1_evaluable_floor = max(0, _env_int("BETA6_WORST_ROOM_F1_MIN_EVALUABLE_SUPPORT", 0))
        for metric in metrics:
            room_raw = str(metric.get("room") or "").strip()
            if not room_raw:
                continue
            room = normalize_room_name(room_raw)
            timeline_metrics = _extract_beta6_timeline_metrics(metric) or {}
            regression = _to_optional_float(timeline_metrics.get("mae_regression_pct"))
            if regression is None:
                candidate_mae = _to_optional_float(timeline_metrics.get("duration_mae_minutes"))
                baseline_mae = _to_optional_float(
                    timeline_metrics.get("baseline_duration_mae_minutes")
                    or timeline_metrics.get("champion_duration_mae_minutes")
                )
                if candidate_mae is not None and baseline_mae is not None and baseline_mae > 0.0:
                    regression = max(0.0, float(candidate_mae - baseline_mae) / float(baseline_mae))
            decision_blob = metric.get("beta6_room_decision") if isinstance(metric, dict) else {}
            decision_reason = str((decision_blob or {}).get("reason_code") or "")
            if regression is None and decision_reason == ReasonCode.FAIL_TIMELINE_MAE.value:
                regression = 1.0
            if regression is not None:
                mae_regression_by_room[room] = float(regression)

            reason_tokens = [str(token).strip().lower() for token in (metric.get("gate_reasons") or [])]
            reason_blob = " ".join(reason_tokens)
            if "alert_precision" in reason_blob:
                alert_precision_below = True
            if "worst_room_f1" in reason_blob or "f1_below_floor" in reason_blob:
                metric_source = str(metric.get("metric_source") or "").strip().lower()
                validation_min_support = int(metric.get("validation_min_class_support", 0) or 0)
                required_minority_support = int(metric.get("required_minority_support", 0) or 0)
                evaluable_floor = max(f1_evaluable_floor, required_minority_support)
                evidence_tokens = (
                    "insufficient_validation_support",
                    "insufficient_calibration_support",
                    "calibration_low_support_fallback",
                    "label_recall_insufficient_support",
                    "critical_label_missing_validation",
                    "critical_label_recall_missing",
                )
                has_insufficient_evidence = (
                    bool(metric.get("insufficient_validation_evidence", False))
                    or metric_source != "holdout_validation"
                    or (evaluable_floor > 0 and validation_min_support <= 0)
                    or (evaluable_floor > 0 and validation_min_support < evaluable_floor)
                    or any(token in reason_blob for token in evidence_tokens)
                )
                if not has_insufficient_evidence:
                    worst_room_f1_below_floor = True

        nightly_entry = {
            "run_id": str(run_id),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pipeline_success_rate": float(gate_summary["nightly_pipeline_success_rate"]),
            "pipeline_success_source": "beta6_authority_execution",
            "mae_regression_pct_by_room": mae_regression_by_room,
            "alert_precision_below_threshold": bool(alert_precision_below),
            "worst_room_f1_below_floor": bool(worst_room_f1_below_floor),
        }
        nightly_history = _append_beta6_rollout_nightly_metric(
            elder_id=elder_id,
            run_id=run_id,
            registry_v2=registry_v2,
            entry=nightly_entry,
        )
        override_manager = PilotOverrideManager() if registry_v2 is not None else None
        auto_rollback_result = rollout_manager.apply_auto_rollback_protection(
            nightly_metrics=nightly_history,
            elder_id=elder_id,
            rooms=room_tokens,
            run_id=run_id,
            registry_v2=registry_v2,
            override_manager=override_manager,
        )
        phase6_auto_rollback_report = {
            **dict(auto_rollback_result),
            "history_points": int(len(nightly_history)),
            "rooms": list(room_tokens),
        }
        if str(auto_rollback_result.get("status", "")) == "rollback_applied":
            run_decision = run_decision.__class__(
                passed=False,
                reason_code=ReasonCode.ROLLBACK_TRIGGERED,
                room_decisions=list(run_decision.room_decisions),
                details={
                    **dict(run_decision.details),
                    "phase6_auto_rollback": dict(phase6_auto_rollback_report),
                },
            )
    except Exception as exc:
        error_token = f"{type(exc).__name__}: {exc}"
        phase6_rollout_ladder_report = {
            "status": "error",
            "error": error_token,
        }
        phase6_auto_rollback_report = {
            "status": "error",
            "error": error_token,
        }
        run_decision = run_decision.__class__(
            passed=False,
            reason_code=(
                ReasonCode.FAIL_GATE_POLICY
                if bool(run_decision.passed)
                else run_decision.reason_code
            ),
            room_decisions=list(run_decision.room_decisions),
            details={
                **dict(run_decision.details),
                "phase6_step6_2_failed": True,
                "phase6_step6_2_error": error_token,
            },
        )
        logger.exception(
            "Beta 6 Phase 6.2 rollout ladder/auto-rollback evaluation failed for elder=%s run_id=%s",
            elder_id,
            run_id,
        )

    if registry_v2 is not None:
        registry_v2.append_event(
            DecisionEvent(
                event_type=EventType.RUN_GATE_PASSED if run_decision.passed else EventType.RUN_GATE_FAILED,
                run_id=run_id,
                elder_id=elder_id,
                reason_code=run_decision.reason_code.value,
                payload={
                    "details": dict(run_decision.details),
                    "room_decisions": [
                        {
                            "room": decision.room,
                            "passed": bool(decision.passed),
                            "reason_code": decision.reason_code.value,
                        }
                        for decision in run_decision.room_decisions
                    ],
                },
            )
        )

    evaluation_report_path = None
    rejection_artifact_path = None
    if (
        phase4_gate_error is None
        and artifact_output_dir is not None
        and isinstance(phase4_gate_artifacts, dict)
    ):
        eval_path = artifact_output_dir / f"{run_id}_evaluation_report.json"
        if eval_path.exists():
            evaluation_report_path = str(eval_path)
        rejection_artifact = phase4_gate_artifacts.get("rejection_artifact")
        if rejection_artifact:
            reject_path = artifact_output_dir / f"{run_id}_rejection_artifact.json"
            if reject_path.exists():
                rejection_artifact_path = str(reject_path)
    return run_decision.passed, {
        "pass": bool(run_decision.passed),
        "reason_code": run_decision.reason_code.value,
        "details": dict(run_decision.details),
        "phase4_dynamic_gate": {
            "status": "error" if phase4_gate_error else "ok",
            "error": phase4_gate_error,
            "evaluation_report_signature": (
                str((phase4_gate_artifacts or {}).get("evaluation_report", {}).get("signature", ""))
                if isinstance(phase4_gate_artifacts, dict)
                else ""
            ),
            "rejection_artifact_signature": (
                str(((phase4_gate_artifacts or {}).get("rejection_artifact") or {}).get("signature", ""))
                if isinstance(phase4_gate_artifacts, dict)
                else ""
            ),
            "evaluation_report_path": evaluation_report_path,
            "rejection_artifact_path": rejection_artifact_path,
        },
        "phase6_shadow_compare": {
            "status": "error" if phase6_shadow_compare_error else "ok",
            "error": phase6_shadow_compare_error,
            "summary": dict(phase6_shadow_compare_artifact or {}),
        },
        "phase6_rollout_ladder": dict(phase6_rollout_ladder_report),
        "phase6_auto_rollback": dict(phase6_auto_rollback_report),
    }


def _sync_beta6_fallback_mode(
    *,
    metrics: list[dict],
    elder_id: str,
    run_id: str,
    registry_v2: RegistryV2 | None,
) -> dict:
    """
    Keep operator-safe fallback mode aligned with Beta 6 room decisions.
    """
    summary = {"activated": [], "cleared": [], "errors": []}
    if registry_v2 is None:
        return summary

    for metric in metrics:
        room = str(metric.get("room") or "").strip()
        if not room:
            continue
        gate_pass = bool(metric.get("gate_pass", False))
        decision = metric.get("beta6_room_decision") if isinstance(metric, dict) else {}
        reason_code = str((decision or {}).get("reason_code") or ReasonCode.FAIL_UNKNOWN_REASON.value)
        try:
            if gate_pass:
                state = registry_v2.read_fallback_state(elder_id, room)
                if isinstance(state, dict) and bool(state.get("active", False)):
                    registry_v2.clear_fallback_mode(
                        elder_id=elder_id,
                        room=room,
                        run_id=run_id,
                        clear_reason_code=ReasonCode.PASS.value,
                        restore_previous_pointer=True,
                        metadata={"source": "run_daily_analysis"},
                    )
                    summary["cleared"].append(room)
            else:
                registry_v2.activate_fallback_mode(
                    elder_id=elder_id,
                    room=room,
                    run_id=run_id,
                    trigger_reason_code=reason_code,
                    fallback_flags={
                        "operator_safe_mode": True,
                        "serving_mode": "rule_hmm_baseline",
                        "beta6_authority": True,
                    },
                    metadata={"source": "run_daily_analysis"},
                )
                summary["activated"].append(room)
        except Exception as exc:
            summary["errors"].append({"room": room, "error": str(exc)})
    return summary


def _call_rollback_rooms_by_name(
    *,
    registry: ModelRegistry,
    elder_id: str,
    room_names: list[str],
    previous_versions: dict[str, int],
    registry_v2: RegistryV2 | None,
    run_id: str,
) -> tuple[list[str], list[str]]:
    try:
        return _rollback_rooms_by_name(
            registry=registry,
            elder_id=elder_id,
            room_names=room_names,
            previous_versions=previous_versions,
            registry_v2=registry_v2,
            run_id=run_id,
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return _rollback_rooms_by_name(registry, elder_id, room_names, previous_versions)


def _call_promote_candidate_rooms(
    *,
    registry: ModelRegistry,
    elder_id: str,
    metrics: list[dict],
    previous_versions: dict[str, int],
    registry_v2: RegistryV2 | None,
    run_id: str,
) -> tuple[list[str], list[str]]:
    try:
        return _promote_candidate_rooms(
            registry=registry,
            elder_id=elder_id,
            metrics=metrics,
            previous_versions=previous_versions,
            registry_v2=registry_v2,
            run_id=run_id,
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return _promote_candidate_rooms(registry, elder_id, metrics)


def _call_rollback_promoted_rooms_if_needed(
    *,
    registry: ModelRegistry,
    elder_id: str,
    metrics: list[dict],
    previous_versions: dict[str, int],
    registry_v2: RegistryV2 | None,
    run_id: str,
) -> tuple[list[str], list[str]]:
    try:
        return _rollback_promoted_rooms_if_needed(
            registry=registry,
            elder_id=elder_id,
            metrics=metrics,
            previous_versions=previous_versions,
            registry_v2=registry_v2,
            run_id=run_id,
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return _rollback_promoted_rooms_if_needed(registry, elder_id, metrics, previous_versions)

def _parse_room_override_map(raw: str) -> dict[str, str]:
    """
    Parse CSV room override map supporting `room:value` and `room=value`.
    """
    result: dict[str, str] = {}
    txt = str(raw or "").strip()
    if not txt:
        return result
    for token in txt.split(","):
        item = str(token).strip()
        if not item:
            continue
        if ":" in item:
            room_raw, value_raw = item.split(":", 1)
        elif "=" in item:
            room_raw, value_raw = item.split("=", 1)
        else:
            continue
        room_key = normalize_room_name(room_raw)
        value_txt = str(value_raw).strip()
        if room_key and value_txt:
            result[room_key] = value_txt
    return result


def _resolve_room_int_env(var_name: str, room_name: str, default: int, minimum: int = 0) -> int:
    value = _env_int(var_name, default)
    room_map = _parse_room_override_map(os.getenv(f"{var_name}_BY_ROOM", ""))
    room_key = normalize_room_name(room_name)
    if room_key in room_map:
        try:
            value = int(room_map[room_key])
        except (TypeError, ValueError):
            pass
    return max(int(minimum), int(value))


def _resolve_room_float_env(
    var_name: str,
    room_name: str,
    default: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    value = _env_float(var_name, default)
    room_map = _parse_room_override_map(os.getenv(f"{var_name}_BY_ROOM", ""))
    room_key = normalize_room_name(room_name)
    if room_key in room_map:
        try:
            value = float(room_map[room_key])
        except (TypeError, ValueError):
            pass
    return float(min(max(value, minimum), maximum))


def _estimate_wf_feasibility(
    room_df: pd.DataFrame | None,
    min_train_days: int,
    valid_days: int,
    step_days: int,
    max_folds: int,
) -> dict:
    """Estimate whether walk-forward can form at least one fold."""
    observed_days = 0
    if room_df is not None and not room_df.empty and "timestamp" in room_df.columns:
        ts = pd.to_datetime(room_df["timestamp"], errors="coerce").dropna()
        if not ts.empty:
            observed_days = int(ts.dt.floor("D").nunique())
    required_days = int(max(1, min_train_days) + max(1, valid_days))
    if observed_days < required_days:
        return {
            "observed_days": int(observed_days),
            "required_days": int(required_days),
            "expected_folds": 0,
            "ready": False,
        }
    raw_folds = 1 + ((int(observed_days) - int(required_days)) // max(1, int(step_days)))
    expected_folds = int(min(max(1, int(max_folds)), max(0, raw_folds)))
    return {
        "observed_days": int(observed_days),
        "required_days": int(required_days),
        "expected_folds": int(expected_folds),
        "ready": expected_folds > 0,
    }


def _evaluate_backbone_alignment_gate(metrics: list[dict]) -> tuple[bool, dict]:
    """
    Ensure promoted candidates align with the active shared backbone id.

    This gate is primarily for shared-backbone + resident-adapter rollout safety.
    """
    shared_enabled = _is_env_enabled("ENABLE_SHARED_BACKBONE_ADAPTERS", default=False)
    enforced = _is_env_enabled("ENFORCE_BACKBONE_ALIGNMENT_GATE", default=shared_enabled)
    if not enforced:
        return True, {"pass": True, "reason": "disabled", "failed_rooms": []}

    target_backbone_id = str(os.getenv("ACTIVE_SHARED_BACKBONE_ID", "")).strip()
    promoted_metrics = [m for m in metrics if m.get("room") and bool(m.get("gate_pass", False))]
    if not promoted_metrics:
        return True, {
            "pass": True,
            "reason": "no_promoted_candidates",
            "target_backbone_id": target_backbone_id or None,
            "failed_rooms": [],
            "room_backbone_ids": {},
        }

    room_backbone_ids: dict[str, str | None] = {}
    failed_rooms: list[str] = []
    for metric in promoted_metrics:
        room = str(metric.get("room"))
        model_identity = metric.get("model_identity") or {}
        backbone_id = model_identity.get("backbone_id") if isinstance(model_identity, dict) else None
        backbone_txt = str(backbone_id).strip() if backbone_id is not None else None
        room_backbone_ids[room] = backbone_txt

        if target_backbone_id:
            if backbone_txt != target_backbone_id:
                failed_rooms.append(room)
        else:
            if backbone_txt is None:
                failed_rooms.append(room)

    return len(failed_rooms) == 0, {
        "pass": len(failed_rooms) == 0,
        "reason": "all_rooms_aligned" if len(failed_rooms) == 0 else "room_backbone_mismatch",
        "target_backbone_id": target_backbone_id or None,
        "failed_rooms": failed_rooms,
        "room_backbone_ids": room_backbone_ids,
    }


def _load_version_artifacts(
    registry: ModelRegistry,
    elder_id: str,
    room_name: str,
    version: int,
) -> tuple[dict | None, str | None]:
    """Load one concrete version's model + scaler + encoder + thresholds."""
    if int(version) <= 0:
        return None, f"Invalid version for room={room_name}: {version}"

    models_dir = registry.get_models_dir(elder_id)
    version = int(version)
    model_path = models_dir / f"{room_name}_v{version}_model.keras"
    scaler_path = models_dir / f"{room_name}_v{version}_scaler.pkl"
    encoder_path = models_dir / f"{room_name}_v{version}_label_encoder.pkl"
    thresholds_path = models_dir / f"{room_name}_v{version}_thresholds.json"

    if not model_path.exists():
        return None, f"Missing model artifact: {model_path.name}"
    if not scaler_path.exists():
        return None, f"Missing scaler artifact: {scaler_path.name}"
    if not encoder_path.exists():
        return None, f"Missing encoder artifact: {encoder_path.name}"

    try:
        model = registry.load_room_model(str(model_path), room_name, compile_model=False)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
        class_thresholds = {}
        if thresholds_path.exists():
            with open(thresholds_path, "r") as f:
                loaded = json.load(f)
            class_thresholds = {str(k): float(v) for k, v in loaded.items()}
        return {
            "version": version,
            "model": model,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "class_thresholds": class_thresholds,
        }, None
    except Exception as e:
        return None, f"Failed loading artifacts for {room_name} v{version}: {e}"


def _evaluate_walk_forward_promotion_gate(
    pipeline: UnifiedPipeline,
    registry: ModelRegistry,
    elder_id: str,
    metrics: list[dict],
    previous_versions: dict[str, int],
    pending_files: list[Path] | None = None,
) -> tuple[bool, dict]:
    """
    Compare promoted candidate versions against previous champions using walk-forward.
    """
    promoted_metrics = [m for m in metrics if m.get("room") and bool(m.get("gate_pass", False))]
    promoted_rooms = [str(m.get("room")) for m in promoted_metrics if m.get("room")]
    if not promoted_metrics:
        return True, {"pass": True, "reason": "no_promoted_candidates", "room_reports": []}

    lookback_days = max(7, _env_int("WF_LOOKBACK_DAYS", 90))
    min_train_days = max(2, _env_int("WF_MIN_TRAIN_DAYS", 7))
    valid_days = max(1, _env_int("WF_VALID_DAYS", 1))
    step_days = max(1, _env_int("WF_STEP_DAYS", 1))
    max_folds = max(1, _env_int("WF_MAX_FOLDS", 30))
    drift_threshold = _env_float("WF_DRIFT_THRESHOLD", 0.60)
    max_low_folds = max(0, _env_int("WF_MAX_LOW_FOLDS", 1))
    min_stability_accuracy = _env_float("WF_MIN_STABILITY_ACCURACY", 0.99)
    max_stability_low_folds = max(0, _env_int("WF_MAX_STABILITY_LOW_FOLDS", 0))
    min_transition_f1 = _env_float("WF_MIN_TRANSITION_F1", 0.80)
    max_transition_low_folds = max(0, _env_int("WF_MAX_TRANSITION_LOW_FOLDS", 1))
    wf_minority_support_default = max(0, int(get_runtime_wf_min_minority_support_default()))
    wf_minority_support_by_room = get_runtime_wf_min_minority_support_by_room()
    require_transition_coverage = _is_env_enabled("WF_REQUIRE_TRANSITION_COVERAGE", default=False)
    evidence_profile = _resolve_release_gate_evidence_profile(default_profile="production")
    pilot_relaxed_evidence = evidence_profile in {"pilot_stage_a", "pilot_stage_b"}

    try:
        policy = get_release_gates_config()
    except Exception as e:
        reason = f"gate_config_unavailable:{e}"
        return False, {
            "pass": False,
            "reason": reason,
            "failed_rooms": promoted_rooms,
            "room_reports": [
                {
                    "room": room,
                    "pass": False,
                    "reasons": [reason],
                }
                for room in promoted_rooms
            ],
        }

    no_regress_cfg = policy.get("release_gates", {}).get("no_regress", {})
    max_drop = float(no_regress_cfg.get("max_drop_from_champion", 0.05))
    exempt_rooms = {normalize_room_name(r) for r in no_regress_cfg.get("exempt_rooms", [])}
    room_policy_map = policy.get("release_gates", {}).get("rooms", {})
    baseline_cfg = policy.get("baseline_check", {})
    baseline_run_at_day = float(baseline_cfg.get("run_at_day", 22))
    baseline_model = str(baseline_cfg.get("baseline_model", "xgboost")).strip().lower()
    baseline_required_adv = float(baseline_cfg.get("required_transformer_advantage", 0.05))

    room_reports: list[dict] = []
    failed_rooms: list[str] = []

    for metric in promoted_metrics:
        room = str(metric.get("room"))
        room_key = normalize_room_name(room)
        policy_minority_support = int(wf_minority_support_by_room.get(room_key, wf_minority_support_default))
        room_step_days = _resolve_room_int_env("WF_STEP_DAYS", room, step_days, minimum=1)
        room_drift_threshold = _resolve_room_float_env(
            "WF_DRIFT_THRESHOLD", room, drift_threshold, minimum=0.0, maximum=1.0
        )
        room_max_low_folds = _resolve_room_int_env("WF_MAX_LOW_FOLDS", room, max_low_folds, minimum=0)
        room_min_minority_support = _resolve_room_int_env(
            "WF_MIN_MINORITY_SUPPORT",
            room,
            policy_minority_support,
            minimum=0,
        )
        room_min_stability_accuracy = _resolve_room_float_env(
            "WF_MIN_STABILITY_ACCURACY", room, min_stability_accuracy, minimum=0.0, maximum=1.0
        )
        room_max_stability_low_folds = _resolve_room_int_env(
            "WF_MAX_STABILITY_LOW_FOLDS", room, max_stability_low_folds, minimum=0
        )
        room_min_transition_f1 = _resolve_room_float_env(
            "WF_MIN_TRANSITION_F1", room, min_transition_f1, minimum=0.0, maximum=1.0
        )
        room_max_transition_low_folds = _resolve_room_int_env(
            "WF_MAX_TRANSITION_LOW_FOLDS", room, max_transition_low_folds, minimum=0
        )
        splitter = TimeCheckpointedSplitter(
            min_train_days=min_train_days,
            valid_days=valid_days,
            step_days=room_step_days,
            max_folds=max_folds,
        )
        candidate_version = int(metric.get("saved_version") or registry.get_current_version(elder_id, room))
        previous_version = int(previous_versions.get(room, 0) or 0)
        training_days = float(metric.get("training_days", 0.0) or 0.0)
        reasons: list[str] = []
        watch_reasons: list[str] = []
        metric_gate_watch_reasons = [str(r) for r in (metric.get("gate_watch_reasons") or [])]

        metric_low_support_markers = (
            "insufficient_validation_support:",
            "room_threshold_not_evaluable:",
            "insufficient_calibration_support:",
            "calibration_low_support_fallback:",
        )
        metric_low_support_not_evaluable = any(
            any(reason.startswith(prefix) for prefix in metric_low_support_markers)
            for reason in metric_gate_watch_reasons
        )

        if pilot_relaxed_evidence and metric_low_support_not_evaluable:
            skip_reason = f"wf_skipped_low_support_from_training_gate:{room_key}"
            merged_watch = list(dict.fromkeys([skip_reason] + metric_gate_watch_reasons))
            room_reports.append(
                {
                    "room": room,
                    "candidate_version": candidate_version,
                    "champion_version": previous_version,
                    "candidate_summary": None,
                    "candidate_low_folds": 0,
                    "candidate_low_support_folds": 0,
                    "candidate_low_stability_folds": 0,
                    "candidate_low_transition_folds": 0,
                    "candidate_transition_supported_folds": 0,
                    "candidate_threshold_required": None,
                    "candidate_wf_config": {
                        "lookback_days": int(lookback_days),
                        "step_days": int(room_step_days),
                        "drift_threshold": float(room_drift_threshold),
                        "max_low_folds": int(room_max_low_folds),
                        "min_minority_support": int(room_min_minority_support),
                        "min_stability_accuracy": float(room_min_stability_accuracy),
                        "max_stability_low_folds": int(room_max_stability_low_folds),
                        "min_transition_f1": float(room_min_transition_f1),
                        "max_transition_low_folds": int(room_max_transition_low_folds),
                        "observed_days": None,
                        "required_days": None,
                        "expected_folds": None,
                    },
                    "baseline_summary": None,
                    "baseline_required_advantage": None,
                    "champion_macro_f1_mean": None,
                    "candidate_stability_accuracy_mean": None,
                    "candidate_transition_macro_f1_mean": None,
                    "pass": True,
                    "reasons": list(merged_watch),
                    "blocking_reasons": [],
                    "watch_reasons": list(merged_watch),
                }
            )
            continue

        def _add_reason(message: str, *, low_support: bool = False) -> None:
            if low_support and pilot_relaxed_evidence:
                watch_reasons.append(str(message))
                return
            reasons.append(str(message))

        room_df, room_err = load_room_training_dataframe(
            elder_id=elder_id,
            room_name=room,
            archive_dir=ARCHIVE_DATA_DIR,
            load_sensor_data_fn=load_sensor_data,
            normalize_room_name_fn=normalize_room_name,
            lookback_days=lookback_days,
            include_files=pending_files or [],
        )
        if room_err:
            _add_reason(f"dataset_error:{room_err}")
            room_reports.append(
                {
                    "room": room,
                    "candidate_version": candidate_version,
                    "champion_version": previous_version,
                    "pass": False,
                    "reasons": reasons,
                }
            )
            failed_rooms.append(room)
            continue

        candidate_artifacts, candidate_load_err = _load_version_artifacts(
            registry=registry,
            elder_id=elder_id,
            room_name=room,
            version=candidate_version,
        )
        if candidate_load_err:
            _add_reason(f"candidate_load_error:{candidate_load_err}")
            room_reports.append(
                {
                    "room": room,
                    "candidate_version": candidate_version,
                    "champion_version": previous_version,
                    "pass": False,
                    "reasons": reasons,
                }
            )
            failed_rooms.append(room)
            continue

        seq_length = int(pipeline.room_config.calculate_seq_length(room))
        candidate_report, candidate_eval_err = evaluate_model_version(
            model=candidate_artifacts["model"],
            platform=pipeline.platform,
            room_name=room,
            room_df=room_df,
            seq_length=seq_length,
            scaler=candidate_artifacts["scaler"],
            label_encoder=candidate_artifacts["label_encoder"],
            splitter=splitter,
            class_thresholds=candidate_artifacts.get("class_thresholds", {}),
        )
        if candidate_eval_err or not candidate_report:
            _add_reason(f"candidate_eval_error:{candidate_eval_err or 'unknown'}")
            room_reports.append(
                {
                    "room": room,
                    "candidate_version": candidate_version,
                    "champion_version": previous_version,
                    "pass": False,
                    "reasons": reasons,
                }
            )
            failed_rooms.append(room)
            continue

        candidate_summary = candidate_report.get("summary", {})
        candidate_f1 = candidate_summary.get("macro_f1_mean")
        candidate_num_folds = int(candidate_summary.get("num_folds", 0) or 0)
        room_policy = room_policy_map.get(room_key, {})
        room_threshold = _resolve_scheduled_threshold(room_policy.get("schedule", []), training_days)
        wf_feasibility = _estimate_wf_feasibility(
            room_df=room_df,
            min_train_days=min_train_days,
            valid_days=valid_days,
            step_days=room_step_days,
            max_folds=max_folds,
        )

        if candidate_num_folds <= 0:
            _add_reason(
                f"wf_no_folds:{room_key}:observed_days={int(wf_feasibility['observed_days'])}"
                f"<required_days={int(wf_feasibility['required_days'])}"
            )
        elif candidate_f1 is None:
            _add_reason("candidate_metric_missing:macro_f1_mean")
        elif room_threshold is not None and float(candidate_f1) < float(room_threshold):
            _add_reason(
                f"room_threshold_failed:{room_key}:f1={float(candidate_f1):.3f}<required={float(room_threshold):.3f}"
            )

        low_folds = [
            fold
            for fold in candidate_report.get("folds", [])
            if float(fold.get("macro_f1", 0.0)) < float(room_drift_threshold)
        ]
        if len(low_folds) > int(room_max_low_folds):
            _add_reason(
                f"drift_stability_failed:{room_key}:low_folds={len(low_folds)}>max_low_folds={int(room_max_low_folds)}"
            )
        low_support_folds = [
            fold
            for fold in candidate_report.get("folds", [])
            if int(fold.get("minority_support", 0) or 0) < int(room_min_minority_support)
        ]
        if int(room_min_minority_support) > 0 and low_support_folds:
            _add_reason(
                f"fold_support_failed:{room_key}:"
                f"low_support_folds={len(low_support_folds)}<min_support={int(room_min_minority_support)}",
                low_support=True,
            )
        stability_low_folds = [
            fold
            for fold in candidate_report.get("folds", [])
            if fold.get("stability_accuracy") is not None
            and float(fold.get("stability_accuracy")) < float(room_min_stability_accuracy)
        ]
        if len(stability_low_folds) > int(room_max_stability_low_folds):
            _add_reason(
                f"stability_guard_failed:{room_key}:"
                f"low_folds={len(stability_low_folds)}>max_low_folds={int(room_max_stability_low_folds)}"
            )
        transition_candidate_folds = [
            fold for fold in candidate_report.get("folds", [])
            if int(fold.get("transition_support", 0) or 0) > 0
        ]
        transition_low_folds = [
            fold
            for fold in transition_candidate_folds
            if fold.get("transition_macro_f1") is not None
            and float(fold.get("transition_macro_f1")) < float(room_min_transition_f1)
        ]
        if len(transition_low_folds) > int(room_max_transition_low_folds):
            _add_reason(
                f"transition_guard_failed:{room_key}:"
                f"low_folds={len(transition_low_folds)}>max_low_folds={int(room_max_transition_low_folds)}"
            )
        if require_transition_coverage and len(transition_candidate_folds) == 0:
            _add_reason(f"transition_coverage_missing:{room_key}:no_transition_windows")

        baseline_summary = None
        if training_days >= baseline_run_at_day and candidate_f1 is not None:
            baseline_report, baseline_err = evaluate_baseline_version(
                platform=pipeline.platform,
                room_name=room,
                room_df=room_df,
                seq_length=seq_length,
                scaler=candidate_artifacts["scaler"],
                label_encoder=candidate_artifacts["label_encoder"],
                splitter=splitter,
                baseline_model=baseline_model,
            )
            if baseline_err or not baseline_report:
                _add_reason(f"baseline_eval_error:{baseline_err or 'unknown'}")
            else:
                baseline_summary = baseline_report.get("summary", {})
                baseline_f1 = baseline_summary.get("macro_f1_mean")
                if baseline_f1 is None:
                    _add_reason("baseline_metric_missing:macro_f1_mean")
                else:
                    advantage = float(candidate_f1) - float(baseline_f1)
                    if advantage < baseline_required_adv:
                        _add_reason(
                            f"baseline_advantage_failed:{room_key}:adv={advantage:.3f}<required={baseline_required_adv:.3f}"
                        )

        champion_f1 = None
        if previous_version > 0:
            champion_artifacts, champion_load_err = _load_version_artifacts(
                registry=registry,
                elder_id=elder_id,
                room_name=room,
                version=previous_version,
            )
            if champion_load_err:
                _add_reason(f"champion_load_error:{champion_load_err}")
            else:
                champion_report, champion_eval_err = evaluate_model_version(
                    model=champion_artifacts["model"],
                    platform=pipeline.platform,
                    room_name=room,
                    room_df=room_df,
                    seq_length=seq_length,
                    scaler=champion_artifacts["scaler"],
                    label_encoder=champion_artifacts["label_encoder"],
                    splitter=splitter,
                    class_thresholds=champion_artifacts.get("class_thresholds", {}),
                )
                if champion_eval_err or not champion_report:
                    _add_reason(f"champion_eval_error:{champion_eval_err or 'unknown'}")
                else:
                    champion_f1 = champion_report.get("summary", {}).get("macro_f1_mean")
                    if (
                        room_key not in exempt_rooms
                        and champion_f1 is not None
                        and candidate_f1 is not None
                        and (float(champion_f1) - float(candidate_f1)) > float(max_drop)
                    ):
                        drop = float(champion_f1) - float(candidate_f1)
                        _add_reason(
                            f"no_regress_failed:{room_key}:drop={drop:.3f}>max_drop={float(max_drop):.3f}"
                        )

        room_pass = len(reasons) == 0
        if not room_pass:
            failed_rooms.append(room)
        all_reasons = list(reasons) + list(watch_reasons)

        room_reports.append(
            {
                "room": room,
                "candidate_version": candidate_version,
                "champion_version": previous_version,
                "candidate_summary": candidate_summary,
                "candidate_low_folds": int(len(low_folds)),
                "candidate_low_support_folds": int(len(low_support_folds)),
                "candidate_low_stability_folds": int(len(stability_low_folds)),
                "candidate_low_transition_folds": int(len(transition_low_folds)),
                "candidate_transition_supported_folds": int(len(transition_candidate_folds)),
                "candidate_threshold_required": float(room_threshold) if room_threshold is not None else None,
                "candidate_wf_config": {
                    "lookback_days": int(lookback_days),
                    "step_days": int(room_step_days),
                    "drift_threshold": float(room_drift_threshold),
                    "max_low_folds": int(room_max_low_folds),
                    "min_minority_support": int(room_min_minority_support),
                    "min_stability_accuracy": float(room_min_stability_accuracy),
                    "max_stability_low_folds": int(room_max_stability_low_folds),
                    "min_transition_f1": float(room_min_transition_f1),
                    "max_transition_low_folds": int(room_max_transition_low_folds),
                    "observed_days": int(wf_feasibility["observed_days"]),
                    "required_days": int(wf_feasibility["required_days"]),
                    "expected_folds": int(wf_feasibility["expected_folds"]),
                },
                "baseline_summary": baseline_summary,
                "baseline_required_advantage": baseline_required_adv if training_days >= baseline_run_at_day else None,
                "champion_macro_f1_mean": float(champion_f1) if champion_f1 is not None else None,
                "candidate_stability_accuracy_mean": (
                    float(candidate_summary.get("stability_accuracy_mean"))
                    if candidate_summary.get("stability_accuracy_mean") is not None
                    else None
                ),
                "candidate_transition_macro_f1_mean": (
                    float(candidate_summary.get("transition_macro_f1_mean"))
                    if candidate_summary.get("transition_macro_f1_mean") is not None
                    else None
                ),
                "pass": room_pass,
                "reasons": all_reasons,
                "blocking_reasons": list(reasons),
                "watch_reasons": list(watch_reasons),
            }
        )

    overall_pass = len(failed_rooms) == 0
    return overall_pass, {
        "pass": overall_pass,
        "reason": "all_rooms_passed" if overall_pass else "room_failures",
        "config": {
            "lookback_days": lookback_days,
            "min_train_days": min_train_days,
            "valid_days": valid_days,
            "step_days": step_days,
            "max_folds": max_folds,
            "drift_threshold": drift_threshold,
            "max_low_folds": max_low_folds,
            "min_stability_accuracy": min_stability_accuracy,
            "max_stability_low_folds": max_stability_low_folds,
            "min_transition_f1": min_transition_f1,
            "max_transition_low_folds": max_transition_low_folds,
            "min_minority_support_default": wf_minority_support_default,
            "require_transition_coverage": require_transition_coverage,
            "max_drop_from_champion": max_drop,
            "baseline_check": {
                "run_at_day": baseline_run_at_day,
                "baseline_model": baseline_model,
                "required_transformer_advantage": baseline_required_adv,
            },
        },
        "failed_rooms": failed_rooms,
        "room_reports": room_reports,
    }


def train_files(file_paths: list[Path]):
    """
    Train on ALL archived + incoming training files for one resident.

    This is the auto-aggregate safeguard that prevents catastrophic forgetting.
    """
    if not file_paths:
        return

    parsed_elder_ids = [get_elder_id_from_filename(path.name) for path in file_paths]
    canonical_elder_id = _choose_canonical_elder_id(parsed_elder_ids)
    if not canonical_elder_id:
        logger.error("Unable to resolve canonical elder ID from training batch. Skipping batch.")
        return
    mismatched = sorted(
        {
            str(eid)
            for eid in parsed_elder_ids
            if not _elder_id_lineage_matches(canonical_elder_id, str(eid))
        }
    )
    if mismatched:
        logger.error(
            f"Mixed elder IDs in training batch (canonical={canonical_elder_id}): "
            f"{sorted(set(parsed_elder_ids))}. Skipping batch."
        )
        return

    elder_id = canonical_elder_id
    retrain_mode = "auto_aggregate"
    mode_context: dict = {"manifest_path": None}
    aggregate_files: list[Path] = []
    try:
        retrain_mode, aggregate_files, mode_context = _resolve_training_files_for_run(
            elder_id=elder_id,
            incoming_files=file_paths,
        )
    except Exception as e:
        logger.error(f"Failed resolving training files for {elder_id}: {e}")
        # Fail closed: still archive incoming files to prevent retry loops.
        for file_path in file_paths:
            archive_file(file_path, ARCHIVE_DATA_DIR)
        return

    logger.info(
        f"Starting Training Mode for {elder_id}: "
        f"mode={retrain_mode}, new_files={len(file_paths)}, total_training_files={len(aggregate_files)}"
    )
    if mode_context.get("manifest_path"):
        logger.info(f"Using retrain manifest: {mode_context['manifest_path']}")
    training_fingerprint = ""
    training_policy_hash = ""
    training_code_version = ""
    run_outcome = "failed"

    try:
        # Initialize Pipeline
        pipeline = UnifiedPipeline(enable_denoising=True)
        registry = ModelRegistry(str(Path(__file__).resolve().parent))
        beta6_run_id = _build_beta6_run_id(elder_id)
        registry_v2 = _init_beta6_registry_v2(registry)
        beta6_authority_enabled = _is_beta6_authority_enabled()
        runtime_activation_preflight_report: dict = {"pass": True, "reason": "disabled"}
        if beta6_authority_enabled:
            _ensure_beta6_authority_evidence_profile_default()
        policy = load_policy_from_env()
        if beta6_authority_enabled:
            preflight_ok, preflight_report = _validate_beta6_training_preflight(
                elder_id=elder_id,
                aggregate_files=aggregate_files,
            )
            if not preflight_ok:
                logger.error(
                    "Beta 6 preflight failed for %s (reason=%s): %s",
                    elder_id,
                    preflight_report.get("reason"),
                    preflight_report,
                )
                for file_path in file_paths:
                    archive_file(file_path, ARCHIVE_DATA_DIR)
                return
            runtime_preflight_ok, runtime_preflight_report = _validate_beta6_runtime_activation_preflight(
                elder_id=elder_id,
                registry=registry,
            )
            runtime_activation_preflight_report = {
                "pass": bool(runtime_preflight_ok),
                **(runtime_preflight_report if isinstance(runtime_preflight_report, dict) else {}),
            }
            if not runtime_preflight_ok:
                logger.error(
                    "Beta 6 runtime activation preflight failed for %s (reason=%s): %s",
                    elder_id,
                    runtime_preflight_report.get("reason") if isinstance(runtime_preflight_report, dict) else None,
                    runtime_preflight_report,
                )
                for file_path in file_paths:
                    archive_file(file_path, ARCHIVE_DATA_DIR)
                return
        fp = _compute_training_run_fingerprint(
            elder_id=elder_id,
            retrain_mode=retrain_mode,
            aggregate_files=aggregate_files,
        )
        training_fingerprint = str(fp.get("fingerprint", ""))
        training_policy_hash = str(fp.get("policy_hash", ""))
        training_code_version = str(fp.get("code_version", ""))
        if bool(policy.reproducibility.skip_if_same_data_and_policy):
            previous_run = _load_last_run_fingerprint(registry, elder_id)
            if (
                str(previous_run.get("fingerprint", "")) == training_fingerprint
                and str(previous_run.get("policy_hash", "")) == training_policy_hash
                and str(previous_run.get("code_version", "")) == training_code_version
                and str(previous_run.get("outcome", "")).strip().lower()
                not in {"", "failed"}
            ):
                logger.info(
                    f"Skipping retrain for {elder_id}: identical (data_fingerprint, policy_hash, code_version) "
                    f"({training_fingerprint[:12]}..., {training_policy_hash[:12]}..., {training_code_version[:12]}...)."
                )
                for file_path in file_paths:
                    archive_file(file_path, ARCHIVE_DATA_DIR)
                return
        previous_versions = _snapshot_current_versions(registry, elder_id)
        _bootstrap_beta6_registry_v2_from_legacy(
            registry_v2=registry_v2,
            elder_id=elder_id,
            previous_versions=previous_versions,
        )
        
        walk_forward_gate_enabled = _is_env_enabled("ENABLE_WALK_FORWARD_PROMOTION_GATE", default=False)
        pre_promotion_enabled = walk_forward_gate_enabled and _is_env_enabled(
            "ENABLE_PRE_PROMOTION_GATING", default=True
        )
        defer_promotion_enabled = bool(pre_promotion_enabled or beta6_authority_enabled)

        # Train models on archived + incoming files (no per-file replacement training).
        results, metrics = pipeline.train_from_files(
            aggregate_files,
            elder_id,
            defer_promotion=defer_promotion_enabled,
        )
        walk_forward_gate_pass = True
        walk_forward_gate_report: dict = {"pass": True, "reason": "disabled"}
        wf_rolled_back_rooms: list[str] = []
        wf_deactivated_rooms: list[str] = []
        pre_promoted_rooms: list[str] = []
        pre_promotion_failures: list[str] = []
        decision_trace_gate_pass = True
        decision_trace_gate_report: dict = {"pass": True, "reason": "no_metrics"}
        trace_rolled_back_rooms: list[str] = []
        trace_deactivated_rooms: list[str] = []
        backbone_gate_pass = True
        backbone_gate_report: dict = {"pass": True, "reason": "disabled"}
        backbone_rolled_back_rooms: list[str] = []
        backbone_deactivated_rooms: list[str] = []
        beta6_gate_pass = True
        beta6_gate_report: dict = {"pass": True, "reason": "disabled"}
        beta6_fallback_summary: dict = {"activated": [], "cleared": [], "errors": []}
        beta6_phase4_runtime_policy: dict = {"status": "disabled", "reason": "no_metrics"}
        phase6_stability_report: dict = {"status": "disabled", "reason": "no_metrics"}

        if metrics:
            decision_trace_gate_pass, decision_trace_gate_report = _validate_decision_trace_artifacts(metrics)
            if not decision_trace_gate_pass:
                failed_rooms = decision_trace_gate_report.get("failed_rooms", []) or []
                logger.error(
                    f"Decision-trace artifact gate FAILED for {elder_id}; failed_rooms={failed_rooms}"
                )
                failed_set = {str(r) for r in failed_rooms}
                for metric in metrics:
                    room = str(metric.get("room"))
                    if room in failed_set:
                        metric["gate_pass"] = False
                        existing = metric.get("gate_reasons") or []
                        metric["gate_reasons"] = list(existing) + ["decision_trace_missing"]
                if pre_promotion_enabled:
                    logger.info(
                        f"Pre-promotion gating enabled; skipping rollback for missing traces in {elder_id}."
                    )
                else:
                    trace_rolled_back_rooms, trace_deactivated_rooms = _call_rollback_rooms_by_name(
                        registry=registry,
                        elder_id=elder_id,
                        room_names=[str(r) for r in failed_rooms],
                        previous_versions=previous_versions,
                        registry_v2=registry_v2,
                        run_id=beta6_run_id,
                    )
                    if trace_rolled_back_rooms:
                        logger.warning(
                            f"Rolled back promoted rooms due to missing decision traces: {trace_rolled_back_rooms}"
                        )
                    if trace_deactivated_rooms:
                        logger.warning(
                            f"Deactivated newly promoted rooms due to missing decision traces: {trace_deactivated_rooms}"
                        )

        if metrics and walk_forward_gate_enabled:
            walk_forward_gate_pass, walk_forward_gate_report = _evaluate_walk_forward_promotion_gate(
                pipeline=pipeline,
                registry=registry,
                elder_id=elder_id,
                metrics=metrics,
                previous_versions=previous_versions,
                pending_files=file_paths,
            )
            if not walk_forward_gate_pass:
                failed_rooms = walk_forward_gate_report.get("failed_rooms", []) or []
                if not failed_rooms:
                    # Fail-closed fallback: rollback every promoted room when report omitted room-level failures.
                    failed_rooms = [
                        str(metric.get("room"))
                        for metric in metrics
                        if metric.get("room") and bool(metric.get("gate_pass", False))
                    ]
                    walk_forward_gate_report["failed_rooms"] = failed_rooms
                logger.warning(
                    f"Walk-forward promotion gate FAILED for {elder_id}; "
                    f"failed_rooms={failed_rooms}"
                )
                if pre_promotion_enabled:
                    logger.info(
                        f"Pre-promotion gating enabled; skipping rollback for {elder_id} "
                        f"because candidates were not promoted yet."
                    )
                else:
                    wf_rolled_back_rooms, wf_deactivated_rooms = _call_rollback_rooms_by_name(
                        registry=registry,
                        elder_id=elder_id,
                        room_names=[str(r) for r in failed_rooms],
                        previous_versions=previous_versions,
                        registry_v2=registry_v2,
                        run_id=beta6_run_id,
                    )
                # Ensure failed rooms do not count as promoted in downstream global gate.
                failed_set = {str(r) for r in failed_rooms}
                for metric in metrics:
                    room = str(metric.get("room"))
                    if room in failed_set:
                        metric["gate_pass"] = False
                        existing = metric.get("gate_reasons") or []
                        metric["gate_reasons"] = list(existing) + ["walk_forward_gate_failed"]
                if wf_rolled_back_rooms:
                    logger.warning(
                        f"Rolled back promoted rooms due to walk-forward gate failure: {wf_rolled_back_rooms}"
                    )
                if wf_deactivated_rooms:
                    logger.warning(
                        f"Deactivated newly promoted rooms due to walk-forward gate failure: "
                        f"{wf_deactivated_rooms}"
                    )
            elif pre_promotion_enabled:
                pre_promoted_rooms, pre_promotion_failures = _call_promote_candidate_rooms(
                    registry=registry,
                    elder_id=elder_id,
                    metrics=metrics,
                    previous_versions=previous_versions,
                    registry_v2=registry_v2,
                    run_id=beta6_run_id,
                )
                if pre_promoted_rooms:
                    logger.info(
                        f"Promoted deferred candidates after walk-forward PASS: {pre_promoted_rooms}"
                    )
                if pre_promotion_failures:
                    logger.warning(
                        f"Failed to promote deferred candidates for {elder_id}: {pre_promotion_failures}"
                    )
                    failed_set = {str(r) for r in pre_promotion_failures}
                    for metric in metrics:
                        room = str(metric.get("room"))
                        if room in failed_set:
                            metric["gate_pass"] = False
                            existing = metric.get("gate_reasons") or []
                            metric["gate_reasons"] = list(existing) + ["promotion_apply_failed"]

        if metrics:
            backbone_gate_pass, backbone_gate_report = _evaluate_backbone_alignment_gate(metrics)
            if not backbone_gate_pass:
                failed_rooms = backbone_gate_report.get("failed_rooms", []) or []
                logger.warning(
                    f"Backbone alignment gate FAILED for {elder_id}; failed_rooms={failed_rooms}"
                )
                if pre_promotion_enabled:
                    logger.info(
                        f"Pre-promotion gating enabled; skipping rollback for backbone mismatch in {elder_id}."
                    )
                else:
                    backbone_rolled_back_rooms, backbone_deactivated_rooms = _call_rollback_rooms_by_name(
                        registry=registry,
                        elder_id=elder_id,
                        room_names=[str(r) for r in failed_rooms],
                        previous_versions=previous_versions,
                        registry_v2=registry_v2,
                        run_id=beta6_run_id,
                    )
                failed_set = {str(r) for r in failed_rooms}
                for metric in metrics:
                    room = str(metric.get("room"))
                    if room in failed_set:
                        metric["gate_pass"] = False
                        existing = metric.get("gate_reasons") or []
                        metric["gate_reasons"] = list(existing) + ["backbone_alignment_gate_failed"]

        if metrics:
            beta6_gate_pass, beta6_gate_report = _apply_beta6_gate_authority(
                metrics=metrics,
                elder_id=elder_id,
                run_id=beta6_run_id,
                registry_v2=registry_v2,
            )
            beta6_fallback_summary = _sync_beta6_fallback_mode(
                metrics=metrics,
                elder_id=elder_id,
                run_id=beta6_run_id,
                registry_v2=registry_v2,
            )
            try:
                beta6_phase4_runtime_policy = _publish_beta6_phase4_runtime_policy(
                    metrics=metrics,
                    elder_id=elder_id,
                    run_id=beta6_run_id,
                    registry_v2=registry_v2,
                    beta6_gate_report=beta6_gate_report,
                    beta6_fallback_summary=beta6_fallback_summary,
                )
            except Exception as exc:
                beta6_phase4_runtime_policy = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                logger.exception(
                    "Failed to publish Beta 6 phase4 runtime policy for elder=%s run=%s",
                    elder_id,
                    beta6_run_id,
                )
            if not beta6_gate_pass:
                logger.warning(
                    f"Beta 6 authority gate FAILED for {elder_id}; details={beta6_gate_report}"
                )

        global_gate_pass = True
        global_gate_report: dict = {"pass": True, "reason": "no_metrics"}
        rolled_back_rooms: list[str] = []
        deactivated_rooms: list[str] = []
        if metrics:
            global_gate_pass, global_gate_report = _evaluate_global_gate(metrics)
            if not global_gate_pass:
                logger.warning(
                    f"Global gate FAILED for {elder_id}. "
                    f"Required={global_gate_report.get('required')}, "
                    f"Actual={global_gate_report.get('actual_global_macro_f1')}"
                )
                if _is_non_destructive_global_gate_failure(global_gate_report):
                    global_gate_report["rollback_skipped"] = True
                    logger.error(
                        f"Global gate failure for {elder_id} is non-destructive (policy/config issue). "
                        "Skipping rollback/deactivation to protect active champions."
                    )
                else:
                    rolled_back_rooms, deactivated_rooms = _call_rollback_promoted_rooms_if_needed(
                        registry=registry,
                        elder_id=elder_id,
                        metrics=metrics,
                        previous_versions=previous_versions,
                        registry_v2=registry_v2,
                        run_id=beta6_run_id,
                    )
                    if rolled_back_rooms:
                        logger.warning(f"Rolled back promoted rooms due to global gate failure: {rolled_back_rooms}")
                    if deactivated_rooms:
                        logger.warning(
                            f"Deactivated newly promoted rooms (no prior champion) due to global gate failure: "
                            f"{deactivated_rooms}"
                        )
            else:
                deferred_candidates = [
                    metric for metric in metrics
                    if bool(metric.get("promotion_deferred", False))
                    and not bool(metric.get("promoted_to_latest", False))
                ]
                if deferred_candidates:
                    promoted_after_global, failed_after_global = _call_promote_candidate_rooms(
                        registry=registry,
                        elder_id=elder_id,
                        metrics=deferred_candidates,
                        previous_versions=previous_versions,
                        registry_v2=registry_v2,
                        run_id=beta6_run_id,
                    )
                    if promoted_after_global:
                        pre_promoted_rooms = sorted(
                            set(pre_promoted_rooms).union(set(promoted_after_global))
                        )
                        logger.info(
                            f"Promoted deferred candidates after gate PASS: {promoted_after_global}"
                        )
                    if failed_after_global:
                        pre_promotion_failures = sorted(
                            set(pre_promotion_failures).union(set(failed_after_global))
                        )
                        logger.warning(
                            f"Failed to promote deferred candidates for {elder_id}: {failed_after_global}"
                        )
                        failed_set = {str(room) for room in failed_after_global}
                        for metric in metrics:
                            room = str(metric.get("room"))
                            if room in failed_set:
                                metric["gate_pass"] = False
                                existing = metric.get("gate_reasons") or []
                                metric["gate_reasons"] = list(existing) + ["promotion_apply_failed"]

        if metrics:
            stability_artifact_path = None
            artifact_root = _beta6_gate_artifact_output_dir(elder_id, beta6_run_id, registry_v2)
            if artifact_root is not None:
                stability_artifact_path = artifact_root / f"{beta6_run_id}_phase6_stability_report.json"
            pipeline_success_proxy = 1.0 if (
                decision_trace_gate_pass
                and (not walk_forward_gate_enabled or walk_forward_gate_pass)
                and backbone_gate_pass
                and beta6_gate_pass
                and global_gate_pass
            ) else 0.0
            try:
                phase6_stability = run_daily_stability_certification(
                    elder_id=elder_id,
                    run_id=beta6_run_id,
                    beta6_gate_pass=beta6_gate_pass,
                    beta6_gate_report=beta6_gate_report,
                    beta6_fallback_summary=beta6_fallback_summary,
                    pipeline_success_rate=pipeline_success_proxy,
                    open_p0_incidents=0,
                    artifact_path=stability_artifact_path,
                )
                phase6_stability_report = phase6_stability.to_dict()
            except Exception as exc:
                phase6_stability_report = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                logger.exception(
                    "Phase 6.3 stability certification failed for elder=%s run=%s",
                    elder_id,
                    beta6_run_id,
                )
        
        if metrics:
            # Calculate summary stats
            avg_acc = sum([m['accuracy'] for m in metrics]) / len(metrics) if metrics else 0.0
            decision_trace_stage_failed = not decision_trace_gate_pass
            walk_forward_stage_failed = bool(
                walk_forward_gate_enabled and (
                    not walk_forward_gate_pass or len(pre_promotion_failures) > 0
                )
            )
            promotion_apply_failed = bool(len(pre_promotion_failures) > 0)
            run_failure_stage = _derive_run_failure_stage(
                global_gate_pass=global_gate_pass,
                decision_trace_gate_pass=decision_trace_gate_pass,
                walk_forward_stage_failed=walk_forward_stage_failed,
                metrics=metrics,
            )
            if promotion_apply_failed:
                run_failure_stage = "promotion_apply_failed"
            elif not beta6_gate_pass:
                run_failure_stage = "beta6_authority_failed"
            
            training_status = (
                'rejected_by_global_gate'
                if not global_gate_pass
                else 'rejected_by_artifact_gate'
                if decision_trace_stage_failed
                else 'rejected_by_walk_forward_gate'
                if walk_forward_stage_failed
                else 'rejected_by_promotion_apply'
                if promotion_apply_failed
                else 'rejected_by_beta6_gate'
                if not beta6_gate_pass
                else 'success'
            )
            run_outcome = training_status
            # Save to Training History DB
            adl_svc = ADLService() # Using service to get DB connection
            with adl_svc.db.get_connection() as conn:
                conn.execute('''
                    INSERT INTO training_history (elder_id, model_type, epochs, accuracy, status, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    elder_id, 
                    'Unified Transformer (Auto-Aggregate)', 
                    int((metrics[0] or {}).get('epochs', 0)) if metrics else 0,
                    avg_acc, 
                    training_status,
                    json.dumps({
                        "mode": retrain_mode,
                        "pre_promotion_gating_enabled": pre_promotion_enabled,
                        "new_files": [p.name for p in file_paths],
                        "total_training_files": len(aggregate_files),
                        "training_manifest": [str(p.resolve()) for p in aggregate_files],
                        "data_fingerprint": training_fingerprint,
                        "policy_hash": training_policy_hash,
                        "code_version": training_code_version,
                        "manifest_path": mode_context.get("manifest_path"),
                        "metrics": metrics,
                        "decision_trace_gate": decision_trace_gate_report,
                        "trace_rolled_back_rooms": trace_rolled_back_rooms,
                        "trace_deactivated_rooms": trace_deactivated_rooms,
                        "walk_forward_gate": walk_forward_gate_report,
                        "pre_promoted_rooms": pre_promoted_rooms,
                        "pre_promotion_failures": pre_promotion_failures,
                        "backbone_gate": backbone_gate_report,
                        "backbone_rolled_back_rooms": backbone_rolled_back_rooms,
                        "backbone_deactivated_rooms": backbone_deactivated_rooms,
                        "beta6_run_id": beta6_run_id,
                        "beta6_gate": beta6_gate_report,
                        "beta6_fallback": beta6_fallback_summary,
                        "beta6_phase4_runtime_policy": beta6_phase4_runtime_policy,
                        "beta6_runtime_activation_preflight": runtime_activation_preflight_report,
                        "phase6_stability": phase6_stability_report,
                        "walk_forward_rolled_back_rooms": wf_rolled_back_rooms,
                        "walk_forward_deactivated_rooms": wf_deactivated_rooms,
                        "global_gate": global_gate_report,
                        "rolled_back_rooms": rolled_back_rooms,
                        "deactivated_rooms": deactivated_rooms,
                        "run_failure_stage": run_failure_stage,
                    })
                ))
                conn.commit()
            logger.info(
                f"Saved training history for {elder_id}: Acc={avg_acc:.2f}, "
                f"GlobalGate={'PASS' if global_gate_pass else 'FAIL'}"
            )
        else:
            run_outcome = "no_metrics"

        if not results:
            results = _build_legacy_training_timeline_results(aggregate_files)
            if results:
                logger.info(
                    f"Legacy timeline fallback materialized from training labels: "
                    f"rooms={len(results)}, files={len(aggregate_files)}"
                )
            else:
                logger.warning(
                    f"No timeline payload produced for {elder_id} "
                    f"(metrics={len(metrics) if metrics else 0}, files={len(aggregate_files)})."
                )

        if results:
            logger.info(f"Training complete for {elder_id}. Models updated.")
            
            # Save validation results
            adl_svc = ADLService()
            saved_rooms = 0
            saved_events = 0
            for room, df in results.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                if 'predicted_activity' not in df.columns:
                    logger.warning(f"Skipping {room}: missing predicted_activity column in timeline payload.")
                    continue
                try:
                    adl_svc.save_adl_events(elder_id, room, df)
                    saved_rooms += 1
                    saved_events += int(len(df))
                except Exception as e:
                    logger.error(f"Failed saving ADL events for {elder_id}/{room}: {e}", exc_info=True)
            
            logger.info(
                f"Saved validation results from training data to database: "
                f"rooms={saved_rooms}, events={saved_events}"
            )

            # Mine hard-negative windows for targeted relabel loop in risky rooms.
            try:
                adl_svc = ADLService()
                risky_rooms_default = ",".join(get_hard_negative_risky_rooms_default())
                risky_rooms_raw = os.getenv("HARD_NEGATIVE_RISKY_ROOMS", risky_rooms_default)
                risky_rooms = [r.strip() for r in str(risky_rooms_raw).split(",") if r.strip()]
                top_k_per_room = max(1, _env_int("HARD_NEGATIVE_TOP_K_PER_ROOM", 3))
                min_block_rows = max(3, _env_int("HARD_NEGATIVE_MIN_BLOCK_ROWS", 6))
                unique_dates = set()
                for _, df in results.items():
                    if 'timestamp' in df.columns:
                        dates = pd.to_datetime(df['timestamp']).dt.date.unique()
                        unique_dates.update(dates)
                total_inserted = 0
                total_updated = 0
                with adl_svc.db.get_connection() as conn:
                    for d in unique_dates:
                        d_str = d.strftime('%Y-%m-%d')
                        stats = mine_hard_negative_windows(
                            conn=conn,
                            elder_id=elder_id,
                            record_date=d_str,
                            risky_rooms=risky_rooms,
                            top_k_per_room=top_k_per_room,
                            min_block_rows=min_block_rows,
                            source="run_daily_analysis",
                        )
                        total_inserted += int(stats.get("inserted", 0) or 0)
                        total_updated += int(stats.get("updated", 0) or 0)
                    conn.commit()
                logger.info(
                    f"Hard-negative mining complete for {elder_id}: "
                    f"inserted={total_inserted}, updated={total_updated}"
                )
            except Exception as e:
                logger.warning(f"Hard-negative mining failed for {elder_id}: {e}")
            
            # Run Household Analysis (Empty Home Detection)
            logger.info("Running Household Analysis (Global State) on training data...")
            try:
                h_analyzer = HouseholdAnalyzer()
                # Use today's date? Or extract date from file? 
                # For training files, they usually cover specific days. 
                # Let's try to infer from the training data results.
                # Assuming training data covers one or more days.
                # We need to run analyze_day for EACH day in the results.
                
                unique_dates = set()
                for _, df in results.items():
                    if 'timestamp' in df.columns:
                        dates = pd.to_datetime(df['timestamp']).dt.date.unique()
                        unique_dates.update(dates)
                
                for d in unique_dates:
                    d_str = d.strftime('%Y-%m-%d')
                    h_analyzer.analyze_day(elder_id, d_str)
                    
            except Exception as e:
                 logger.error(f"Error running Household Analysis on training data: {e}", exc_info=True)
            
            # 8. Regenerate Activity Segments for Timeline UI
            logger.info("Regenerating Activity Segments for Timeline UI...")
            try:
                unique_dates = set()
                # Extract dates from prediction results
                for _, df in results.items():
                    if 'timestamp' in df.columns:
                         dates = pd.to_datetime(df['timestamp']).dt.date.unique()
                         unique_dates.update(dates)
                
                # Get list of rooms from results
                rooms = list(results.keys())
                
                for d in unique_dates:
                    d_str = d.strftime('%Y-%m-%d')
                    logger.info(f"Regenerating segments for {d_str}...")
                    for room in rooms:
                        try:
                            count = regenerate_segments(elder_id, room, d_str)
                            # logger.info(f"  -> {room}: {count} segments")
                        except Exception as e:
                            logger.error(f"Failed to regenerate segments for {room} on {d_str}: {e}")
                            
                logger.info("Segment regeneration complete.")
                
            except Exception as e:
                logger.error(f"Error regenerating activity segments: {e}", exc_info=True)
            
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        # If training fails, do NOT archive?? Or archive to failed folder?
        # For now, let's allow archiving so we don't loop forever.
        run_outcome = "failed"
    finally:
        if training_fingerprint:
            try:
                _save_last_run_fingerprint(
                    registry=ModelRegistry(str(Path(__file__).resolve().parent)),
                    elder_id=elder_id,
                    fingerprint=training_fingerprint,
                    policy_hash=training_policy_hash,
                    code_version=training_code_version,
                    retrain_mode=retrain_mode,
                    aggregate_files=aggregate_files,
                    outcome=run_outcome,
                )
            except Exception as e:
                logger.warning(f"Failed saving retrain fingerprint record for {elder_id}: {e}")
        
    # Archive all incoming files (always, even if processing had issues)
    for file_path in file_paths:
        archive_file(file_path, ARCHIVE_DATA_DIR)
    
    # 7. Archive (always archive, even if processing had issues)
    # archive_file(file_path, ARCHIVE_DATA_DIR)


def train_file(file_path: Path):
    """Backward-compatible wrapper for single-file training calls."""
    train_files([file_path])

def job():
    logger.info(f"Checking for new files in {RAW_DATA_DIR}...")
    
    if RAW_DATA_DIR.exists():
        files: list[Path] = []
        for ext in TRAINING_EXTENSIONS:
            files.extend(RAW_DATA_DIR.glob(f"*{ext}"))
        ignored_files = [f for f in files if _is_ignored_raw_file(f)]
        if ignored_files:
            logger.info(
                "Ignoring %d transient/system file(s): %s",
                len(ignored_files),
                ", ".join(f.name for f in ignored_files[:5]),
            )
        files = [f for f in files if not _is_ignored_raw_file(f)]
        # Exclude '_manual_' files which are handled synchronously by the Correction Studio UI
        files = [f for f in files if "_manual_" not in f.name.lower()]
        files.sort(key=lambda f: (0 if _is_training_file(f) else 1, f.name))

        if not files:
            pass # Silent if nothing
        else:
            training_files_by_elder: dict[str, list[Path]] = defaultdict(list)
            inference_files: list[Path] = []

            for file in files:
                if _is_training_file(file):
                    elder_id = get_elder_id_from_filename(file.name)
                    training_files_by_elder[elder_id].append(file)
                else:
                    inference_files.append(file)

            # Batch training files per resident and run exactly one aggregated retrain per batch.
            for elder_id, elder_files in training_files_by_elder.items():
                logger.info(
                    f"Processing {len(elder_files)} training files for {elder_id} "
                    f"in a single auto-aggregate run."
                )
                try:
                    train_files(elder_files)
                except Exception as e:
                    logger.error(f"Failed to process training batch for {elder_id}: {e}")

            for file in inference_files:
                logger.info(f"Processing new file: {file.name}")
                try:
                    process_file(file)
                except Exception as e:
                    logger.error(f"Failed to process {file.name}: {e}")
    else:
        logger.error(f"Raw directory not found: {RAW_DATA_DIR}")

def main():
    logger.info("Beta_3 Automation Service Started")
    if not _acquire_single_instance_lock():
        return
    atexit.register(_release_single_instance_lock)
    try:
        _emit_registry_integrity_summary(
            registry=ModelRegistry(str(Path(__file__).resolve().parent)),
            elder_ids=None,
        )
    except Exception as e:
        logger.warning(f"Startup registry integrity summary failed: {e}")
    job()
    schedule.every(30).seconds.do(job)
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopped.")

if __name__ == "__main__":
    main()
