import os
import logging
import joblib
import json
import shutil
import filecmp
import math
import tempfile
import tensorflow as tf
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from contextlib import contextmanager
from ml.exceptions import ModelLoadError

try:
    import fcntl
except Exception:
    fcntl = None

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Manages loading, saving, and path resolution for ML models.
    
    Supports model versioning with the following features:
    - Automatic version incrementing on save
    - Version history tracking with metadata
    - Rollback to previous versions
    - Backward compatible with unversioned models
    """
    
    # Maximum versions to keep per room (0 = unlimited)
    MAX_VERSIONS_PER_ROOM = 5
    _MANDATORY_LATEST_SUFFIXES = (
        "_model.keras",
        "_scaler.pkl",
        "_label_encoder.pkl",
    )
    _OPTIONAL_LATEST_SUFFIXES = (
        "_thresholds.json",
        "_adapter_weights.pkl",
        "_activity_confidence_calibrator.json",
        "_decision_trace.json",
    )
    _TWO_STAGE_LATEST_SUFFIXES = (
        "_two_stage_meta.json",
        "_two_stage_stage_a_model.keras",
        "_two_stage_stage_b_model.keras",
    )
    _BACKBONE_LAYER_PREFIXES = (
        "cnn_embedding",
        "transformer_block_",
        "sinusoidal_positional_encoding",
        "relative_positional_encoding",
        "learnable_positional_embedding",
        "alibi_bias_gen",
    )
    
    def __init__(self, backend_dir: str):
        self.backend_dir = backend_dir
        
    def get_models_dir(self, elder_id: str) -> Path:
        """Get the directory where models are stored for a specific elder."""
        models_dir = Path(self.backend_dir) / "models" / elder_id
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

    def _get_custom_objects(self) -> Dict[str, Any]:
        """Return custom Keras objects for model deserialization."""
        # Use local imports to avoid circular dependency if these modules import settings
        from ml.transformer_backbone import TransformerEncoderBlock
        from ml.positional_encoding import (
            SinusoidalPositionalEncoding, RelativePositionalEncoding, LearnablePositionalEmbedding
        )
        from elderlycare_v1_16.platform import Attention
        
        return {
            'Attention': Attention,
            'TransformerEncoderBlock': TransformerEncoderBlock,
            'SinusoidalPositionalEncoding': SinusoidalPositionalEncoding,
            'RelativePositionalEncoding': RelativePositionalEncoding,
            'LearnablePositionalEmbedding': LearnablePositionalEmbedding
        }

    # =========================================================================
    # Versioning Methods
    # =========================================================================
    
    def _get_version_info_path(self, elder_id: str, room_name: str) -> Path:
        """Get path to version info JSON file for a room."""
        return self.get_models_dir(elder_id) / f"{room_name}_versions.json"

    def _get_lock_path(self, elder_id: str, room_name: str) -> Path:
        return self.get_models_dir(elder_id) / f"{room_name}.lock"

    @contextmanager
    def _acquire_room_lock(self, elder_id: str, room_name: str):
        lock_path = self._get_lock_path(elder_id, room_name)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    
    def _load_version_info(self, elder_id: str, room_name: str) -> Dict[str, Any]:
        """Load version info from JSON file."""
        path = self._get_version_info_path(elder_id, room_name)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {"versions": [], "current_version": 0}
    
    def _save_version_info(self, elder_id: str, room_name: str, info: Dict[str, Any]):
        """Save version info to JSON file."""
        path = self._get_version_info_path(elder_id, room_name)
        canonical = self._reconcile_promotion_state(info)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{room_name}_versions_",
            suffix=".json.tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(canonical, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    @staticmethod
    def _artifacts_match(src: Path, dst: Path) -> bool:
        """
        Compare artifacts by content.

        JSON aliases can be semantically identical while differing in whitespace or
        key ordering, so compare parsed payloads before falling back to raw bytes.
        """
        if src.suffix == ".json" and dst.suffix == ".json":
            try:
                return json.loads(src.read_text()) == json.loads(dst.read_text())
            except Exception:
                pass
        return filecmp.cmp(src, dst, shallow=False)

    def _repair_two_stage_meta_runtime_fields(
        self,
        *,
        models_dir: Path,
        room_name: str,
        meta_path: Path,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Backfill missing two-stage runtime gate fields from the matching decision trace.

        Historical artifacts omitted `runtime_enabled`, which made runtime loading
        default to `True` and could reactivate a two-stage bundle even when training
        selected a single-stage fallback. When possible, infer the missing fields
        from the corresponding decision trace and persist the repaired metadata.
        """
        if "runtime_enabled" in payload:
            return payload

        try:
            saved_version = int(payload.get("saved_version", 0) or 0)
        except (TypeError, ValueError):
            return payload
        if saved_version <= 0:
            return payload

        inferred: Dict[str, Any] = {}
        trace_candidates = (
            models_dir / f"{room_name}_v{int(saved_version)}_decision_trace.json",
            models_dir / f"{room_name}_decision_trace.json",
        )
        for trace_path in trace_candidates:
            if not trace_path.exists():
                continue
            try:
                trace_payload = json.loads(trace_path.read_text())
            except Exception:
                continue
            metrics = trace_payload.get("metrics") or {}
            two_stage_core = metrics.get("two_stage_core") or {}
            if not isinstance(two_stage_core, dict):
                continue
            if "runtime_use_two_stage" in two_stage_core:
                inferred["runtime_enabled"] = bool(two_stage_core.get("runtime_use_two_stage"))
            gate_source = two_stage_core.get("runtime_gate_source", two_stage_core.get("gate_source"))
            if str(gate_source or "").strip():
                inferred["runtime_gate_source"] = str(gate_source)
            if "selected_reliable" in two_stage_core:
                inferred["selected_reliable"] = bool(two_stage_core.get("selected_reliable"))
            if "fail_closed" in two_stage_core:
                inferred["fail_closed"] = bool(two_stage_core.get("fail_closed"))
            if "fail_closed_reason" in two_stage_core:
                reason = two_stage_core.get("fail_closed_reason")
                inferred["fail_closed_reason"] = None if reason in (None, "") else str(reason)
            if inferred:
                break

        if not inferred:
            return payload

        repaired = dict(payload)
        repaired.update(inferred)

        def _write_if_changed(path: Path) -> None:
            try:
                existing = json.loads(path.read_text()) if path.exists() else {}
            except Exception:
                existing = {}
            updated = dict(existing) if isinstance(existing, dict) else {}
            updated.update(inferred)
            if updated != existing:
                path.write_text(json.dumps(updated, indent=2))

        _write_if_changed(meta_path)
        versioned_meta_path = models_dir / f"{room_name}_v{int(saved_version)}_two_stage_meta.json"
        if versioned_meta_path != meta_path and versioned_meta_path.exists():
            _write_if_changed(versioned_meta_path)

        logger.info(
            "Repaired missing two-stage runtime metadata for %s/%s v%s using decision trace",
            models_dir.name,
            room_name,
            saved_version,
        )
        return repaired

    @staticmethod
    def _reconcile_promotion_state(info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep `current_version` and per-version `promoted` flags consistent.
        
        Strict Bi-directional Sync Rules:
        1. If current_version > 0:
           - That specific version MUST be promoted=True.
           - All other versions MUST be promoted=False.
           - If current_version is invalid (not in list), reset current_version=0.
        2. If current_version == 0:
           - If any version is promoted=True, it becomes current_version.
           - If multiple are promoted, the latest one wins.
           - If none are promoted, current_version remains 0.
        """
        current = int(info.get("current_version", 0) or 0)
        versions = info.get("versions", []) or []
        
        if not versions:
            info["current_version"] = 0
            return info

        if current > 0:
            # Rule 1: current_version defines truth.
            # Verify validity first.
            current_exists = False
            for v in versions:
                if int(v.get("version", 0)) == current:
                    current_exists = True
                    break
            
            if not current_exists:
                # Invalid pointer, reset to safe state
                current = 0
                info["current_version"] = 0
            else:
                # Enforce promotion flag on current and clear others
                for v in versions:
                    v["promoted"] = (int(v.get("version", 0)) == current)
                return info

        # Rule 2 (falls through if current was 0 or reset to 0): 
        # promoted flag defines truth if current is unset.
        promoted_versions = [v for v in versions if v.get("promoted")]
        
        if not promoted_versions:
            # Safe stable state: nothing active.
            return info
            
        if len(promoted_versions) > 1:
            # Conflict: Pick winner (latest version number)
            promoted_versions.sort(key=lambda v: int(v.get("version", 0)), reverse=True)
            winner = promoted_versions[0]
            # Fix flags
            for v in versions:
                v["promoted"] = (v is winner)
            info["current_version"] = int(winner.get("version", 0))
        else:
            # Single promoted version becomes current
            info["current_version"] = int(promoted_versions[0].get("version", 0))
            
        return info
    
    def _get_next_version(self, elder_id: str, room_name: str) -> int:
        """Get the next version number for a model."""
        info = self._load_version_info(elder_id, room_name)
        if not info["versions"]:
            return 1
        return max(int(v["version"]) for v in info["versions"]) + 1
    
    def list_model_versions(self, elder_id: str, room_name: str) -> List[Dict[str, Any]]:
        """
        List all versions for a model.
        
        Returns:
            List of version dicts with: version, created_at, accuracy, samples
        """
        info = self._load_version_info(elder_id, room_name)
        return info.get("versions", [])
    
    def get_current_version(self, elder_id: str, room_name: str) -> int:
        """Get the current (latest) version number."""
        info = self._load_version_info(elder_id, room_name)
        return info.get("current_version", 0)

    def get_current_version_metadata(self, elder_id: str, room_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for the currently promoted (champion) version.

        Returns:
            Dict for current version entry, or None when no champion exists.
        """
        info = self._load_version_info(elder_id, room_name)
        current = info.get("current_version", 0)
        if not current:
            return None
        for version_meta in info.get("versions", []):
            if int(version_meta.get("version", 0)) == int(current):
                return version_meta
        return None
    
    def _cleanup_old_versions(
        self,
        elder_id: str,
        room_name: str,
        preserve_versions: Optional[List[int]] = None,
    ):
        """Remove old versions beyond MAX_VERSIONS_PER_ROOM while pinning explicit versions."""
        if self.MAX_VERSIONS_PER_ROOM <= 0:
            return  # Unlimited
            
        info = self._load_version_info(elder_id, room_name)
        raw_versions = info.get("versions", [])
        versions = sorted(raw_versions, key=lambda v: int(v.get("version", 0)), reverse=True)
        current_version = int(info.get("current_version", 0) or 0)
        pinned_ids = {
            int(version)
            for version in (preserve_versions or [])
            if int(version) > 0
        }
        
        if len(versions) <= self.MAX_VERSIONS_PER_ROOM:
            # Heal stale current_version metadata if it points to a missing version entry.
            if current_version > 0 and not any(int(v.get("version", 0)) == current_version for v in versions):
                info["current_version"] = 0
                self._save_version_info(elder_id, room_name, info)
            return

        keep_versions = versions[:self.MAX_VERSIONS_PER_ROOM]
        keep_ids = {int(v.get("version", 0)) for v in keep_versions}

        # Always retain the current champion plus any explicitly preserved candidate version.
        if current_version > 0:
            pinned_ids.add(current_version)

        for pinned_version in sorted(pinned_ids, reverse=True):
            if pinned_version in keep_ids:
                continue
            pinned_entry = next(
                (v for v in versions if int(v.get("version", 0)) == pinned_version),
                None,
            )
            if pinned_entry is None:
                continue

            replace_idx = None
            for idx in range(len(keep_versions) - 1, -1, -1):
                retained_version = int(keep_versions[idx].get("version", 0))
                if retained_version not in pinned_ids:
                    replace_idx = idx
                    break

            if replace_idx is None:
                keep_versions.append(pinned_entry)
            else:
                keep_versions[replace_idx] = pinned_entry

            keep_versions = sorted(
                {int(v.get("version", 0)): v for v in keep_versions}.values(),
                key=lambda v: int(v.get("version", 0)),
                reverse=True,
            )
            keep_ids = {int(v.get("version", 0)) for v in keep_versions}

        versions_to_remove = [v for v in versions if int(v.get("version", 0)) not in keep_ids]

        models_dir = self.get_models_dir(elder_id)
        
        for v in versions_to_remove:
            ver = int(v.get("version", 0))
            # Remove versioned files
            for suffix in (
                *self._MANDATORY_LATEST_SUFFIXES,
                *self._OPTIONAL_LATEST_SUFFIXES,
                *self._TWO_STAGE_LATEST_SUFFIXES,
            ):
                path = models_dir / f"{room_name}_v{ver}{suffix}"
                if path.exists():
                    path.unlink()
                    logger.debug(f"Removed old version file: {path.name}")
        
        # Update version info
        info["versions"] = keep_versions
        if current_version > 0 and current_version not in {int(v.get("version", 0)) for v in keep_versions}:
            info["current_version"] = 0
        self._save_version_info(elder_id, room_name, info)
        logger.info(f"Cleaned up {len(versions_to_remove)} old version(s) for {room_name}")

    def validate_and_repair_room_registry_state(self, elder_id: str, room_name: str) -> Dict[str, Any]:
        """
        Audit and repair the registry state for a single room.
        
        Enforces AC-6 conformance:
        1. Strict Bi-directional Sync of current/promoted.
        2. If current_version=0, no unversioned aliases exist.
        3. If current_version>0, unversioned aliases match that version (FORCE COPY).
        
        Returns:
            Report dict with 'valid' (bool), 'repaired' (bool), and 'issues' (list).
        """
        report = {"valid": True, "repaired": False, "issues": []}
        
        with self._acquire_room_lock(elder_id, room_name):
            info = self._load_version_info(elder_id, room_name)
            original_info_str = json.dumps(info, sort_keys=True)
            
            # 1. Reconcile promotion flags (Strict Sync)
            # We reuse the logic from _reconcile_promotion_state but instrument it to report issues.
            current = int(info.get("current_version", 0) or 0)
            versions = info.get("versions", [])
            
            # Detect corruption before fix
            promoted_versions = [v for v in versions if v.get("promoted")]
            
            if current > 0:
                 current_entry = next((v for v in versions if int(v.get("version", 0)) == current), None)
                 if not current_entry:
                     report["issues"].append(f"current_version {current} points to non-existent version")
                 elif not current_entry.get("promoted"):
                     report["issues"].append(f"current_version {current} is not marked promoted")
                 
                 if len(promoted_versions) > 1:
                     report["issues"].append("Multiple promoted versions found while current > 0")
            
            elif current == 0 and promoted_versions:
                 report["issues"].append(f"promoted versions found {[v['version'] for v in promoted_versions]} but current_version=0")
            
            # Apply Fix
            reconciled = self._reconcile_promotion_state(info)
            if json.dumps(reconciled, sort_keys=True) != original_info_str:
                 report["repaired"] = True
                 info = reconciled
            
            # Re-read current after reconciliation
            current = int(info.get("current_version", 0) or 0)

            # 2. Alias Consistency
            models_dir = self.get_models_dir(elder_id)
            suffixes = (
                *self._MANDATORY_LATEST_SUFFIXES,
                *self._OPTIONAL_LATEST_SUFFIXES,
                *self._TWO_STAGE_LATEST_SUFFIXES,
            )
            
            if current == 0:
                # Verify no unversioned aliases exist
                for suffix in suffixes:
                    alias_path = models_dir / f"{room_name}{suffix}"
                    if alias_path.exists():
                        report["issues"].append(f"Orphan alias found while current_version=0: {alias_path.name}")
                        alias_path.unlink()
                        report["repaired"] = True
            else:
                # Verify aliases are bit-consistent with current champion artifacts.
                mandatory = self._MANDATORY_LATEST_SUFFIXES
                optional = self._OPTIONAL_LATEST_SUFFIXES
                missing_versioned_mandatory: list[str] = []
                needs_sync = False

                for suffix in mandatory:
                    src = models_dir / f"{room_name}_v{int(current)}{suffix}"
                    dst = models_dir / f"{room_name}{suffix}"
                    if not src.exists():
                        # Backward compatibility: recover versioned artifact from existing alias.
                        if dst.exists():
                            shutil.copy2(dst, src)
                            report["issues"].append(
                                f"Backfilled missing versioned artifact for current_version {current}: {src.name}"
                            )
                            report["repaired"] = True
                        else:
                            missing_versioned_mandatory.append(suffix)
                        continue
                    if not dst.exists():
                        needs_sync = True
                        continue
                    try:
                        if not self._artifacts_match(src, dst):
                            needs_sync = True
                    except OSError:
                        needs_sync = True

                for suffix in optional:
                    src = models_dir / f"{room_name}_v{int(current)}{suffix}"
                    dst = models_dir / f"{room_name}{suffix}"
                    if src.exists():
                        if not dst.exists():
                            needs_sync = True
                            continue
                        try:
                            if not self._artifacts_match(src, dst):
                                needs_sync = True
                        except OSError:
                            needs_sync = True
                    elif dst.exists():
                        # stale optional alias should be removed on sync
                        needs_sync = True

                two_stage_meta_src = models_dir / f"{room_name}_v{int(current)}_two_stage_meta.json"
                two_stage_meta_dst = models_dir / f"{room_name}_two_stage_meta.json"
                two_stage_stage_a_src = models_dir / f"{room_name}_v{int(current)}_two_stage_stage_a_model.keras"
                two_stage_stage_a_dst = models_dir / f"{room_name}_two_stage_stage_a_model.keras"
                two_stage_stage_b_src = models_dir / f"{room_name}_v{int(current)}_two_stage_stage_b_model.keras"
                two_stage_stage_b_dst = models_dir / f"{room_name}_two_stage_stage_b_model.keras"

                if two_stage_meta_src.exists():
                    try:
                        two_stage_payload = json.loads(two_stage_meta_src.read_text())
                        two_stage_payload = self._repair_two_stage_meta_runtime_fields(
                            models_dir=models_dir,
                            room_name=room_name,
                            meta_path=two_stage_meta_src,
                            payload=two_stage_payload,
                        )
                    except Exception as e:
                        report["issues"].append(
                            f"Invalid two-stage metadata for current_version {current}: {two_stage_meta_src.name}: {e}"
                        )
                        report["valid"] = False
                    else:
                        schema_ok = str(two_stage_payload.get("schema_version", "")).strip() == "beta6.two_stage_core.v1"
                        runtime_enabled = bool(two_stage_payload.get("runtime_enabled", True))
                        stage_b_enabled = bool(two_stage_payload.get("stage_b_enabled", False))

                        if not schema_ok:
                            report["issues"].append(
                                f"Invalid two-stage schema for current_version {current}: {two_stage_meta_src.name}"
                            )
                            report["valid"] = False
                        elif runtime_enabled:
                            missing_two_stage: list[str] = []
                            if not two_stage_stage_a_src.exists():
                                missing_two_stage.append(two_stage_stage_a_src.name)
                            if stage_b_enabled and not two_stage_stage_b_src.exists():
                                missing_two_stage.append(two_stage_stage_b_src.name)

                            if missing_two_stage:
                                report["issues"].append(
                                    f"Missing mandatory two-stage artifacts for current_version {current}: {missing_two_stage}"
                                )
                                report["valid"] = False
                            else:
                                two_stage_checks = (
                                    (two_stage_meta_src, two_stage_meta_dst),
                                    (two_stage_stage_a_src, two_stage_stage_a_dst),
                                )
                                for src, dst in two_stage_checks:
                                    if not dst.exists():
                                        needs_sync = True
                                        continue
                                    try:
                                        if not self._artifacts_match(src, dst):
                                            needs_sync = True
                                    except OSError:
                                        needs_sync = True
                                if stage_b_enabled:
                                    if not two_stage_stage_b_dst.exists():
                                        needs_sync = True
                                    else:
                                        try:
                                            if not self._artifacts_match(two_stage_stage_b_src, two_stage_stage_b_dst):
                                                needs_sync = True
                                        except OSError:
                                            needs_sync = True
                                elif two_stage_stage_b_dst.exists():
                                    needs_sync = True
                        elif any(path.exists() for path in (two_stage_meta_dst, two_stage_stage_a_dst, two_stage_stage_b_dst)):
                            needs_sync = True
                elif any(path.exists() for path in (two_stage_meta_dst, two_stage_stage_a_dst, two_stage_stage_b_dst)):
                    needs_sync = True

                if missing_versioned_mandatory:
                    report["issues"].append(
                        f"Missing mandatory versioned artifacts for current_version {current}: "
                        f"{missing_versioned_mandatory}"
                    )
                    report["valid"] = False
                elif needs_sync:
                    report["issues"].append(
                        f"Alias mismatch detected for current_version {current}; syncing aliases"
                    )
                    try:
                        self._ensure_latest_aliases_match_current(
                            elder_id,
                            room_name,
                            current,
                            require_mandatory=True,
                        )
                        report["repaired"] = True
                    except Exception as e:
                        report["issues"].append(f"Alias sync failed for current_version {current}: {e}")
                        report["valid"] = False

                # Post-condition: mandatory aliases must exist when current_version > 0.
                unresolved_aliases = [
                    suffix
                    for suffix in mandatory
                    if not (models_dir / f"{room_name}{suffix}").exists()
                ]
                for suffix in optional:
                    src = models_dir / f"{room_name}_v{int(current)}{suffix}"
                    dst = models_dir / f"{room_name}{suffix}"
                    if src.exists() and not dst.exists():
                        unresolved_aliases.append(suffix)
                if two_stage_meta_src.exists():
                    try:
                        two_stage_payload = json.loads(two_stage_meta_src.read_text())
                        two_stage_payload = self._repair_two_stage_meta_runtime_fields(
                            models_dir=models_dir,
                            room_name=room_name,
                            meta_path=two_stage_meta_src,
                            payload=two_stage_payload,
                        )
                    except Exception:
                        pass
                    else:
                        schema_ok = (
                            str(two_stage_payload.get("schema_version", "")).strip()
                            == "beta6.two_stage_core.v1"
                        )
                        runtime_enabled = bool(two_stage_payload.get("runtime_enabled", True))
                        stage_b_enabled = bool(two_stage_payload.get("stage_b_enabled", False))
                        if schema_ok and runtime_enabled:
                            if not two_stage_meta_dst.exists():
                                unresolved_aliases.append("_two_stage_meta.json")
                            if two_stage_stage_a_src.exists() and not two_stage_stage_a_dst.exists():
                                unresolved_aliases.append("_two_stage_stage_a_model.keras")
                            if (
                                stage_b_enabled
                                and two_stage_stage_b_src.exists()
                                and not two_stage_stage_b_dst.exists()
                            ):
                                unresolved_aliases.append("_two_stage_stage_b_model.keras")
                if unresolved_aliases:
                    report["issues"].append(
                        f"Unresolved mandatory aliases for current_version {current}: {unresolved_aliases}"
                    )
                    report["valid"] = False

            # Save if changed
            new_info_str = json.dumps(info, sort_keys=True)
            if original_info_str != new_info_str:
                self._save_version_info(elder_id, room_name, info)
                report["repaired"] = True
                
            if report["valid"]:
                report["valid"] = not bool(report["issues"]) or report["repaired"]
            return report


    def _ensure_latest_aliases_match_current(
        self,
        elder_id: str,
        room_name: str,
        version: int,
        require_mandatory: bool = False,
    ):
        """
        Force-copy versioned artifacts to unversioned 'latest' alias paths.
        When `require_mandatory=True`, raises if mandatory source artifacts are missing.
        """
        models_dir = self.get_models_dir(elder_id)
        mandatory = self._MANDATORY_LATEST_SUFFIXES
        optional = self._OPTIONAL_LATEST_SUFFIXES

        for suffix in mandatory:
            src = models_dir / f"{room_name}_v{int(version)}{suffix}"
            dst = models_dir / f"{room_name}{suffix}"
            if not src.exists():
                if not require_mandatory:
                    continue
                raise FileNotFoundError(
                    f"Missing mandatory versioned artifact for {room_name} v{int(version)}: {src.name}"
                )
            shutil.copy2(src, dst)

        for suffix in optional:
            src = models_dir / f"{room_name}_v{int(version)}{suffix}"
            dst = models_dir / f"{room_name}{suffix}"
            if src.exists():
                shutil.copy2(src, dst)
            elif suffix in optional and dst.exists():
                # If source optional is missing but destination exists, it's stale/wrong. Remove it.
                dst.unlink()

        two_stage_meta_src = models_dir / f"{room_name}_v{int(version)}_two_stage_meta.json"
        two_stage_stage_a_src = models_dir / f"{room_name}_v{int(version)}_two_stage_stage_a_model.keras"
        two_stage_stage_b_src = models_dir / f"{room_name}_v{int(version)}_two_stage_stage_b_model.keras"
        two_stage_meta_dst = models_dir / f"{room_name}_two_stage_meta.json"
        two_stage_stage_a_dst = models_dir / f"{room_name}_two_stage_stage_a_model.keras"
        two_stage_stage_b_dst = models_dir / f"{room_name}_two_stage_stage_b_model.keras"

        def _clear_two_stage_latest_aliases() -> None:
            for path in (two_stage_meta_dst, two_stage_stage_a_dst, two_stage_stage_b_dst):
                if path.exists():
                    path.unlink()

        if not two_stage_meta_src.exists():
            _clear_two_stage_latest_aliases()
            return

        try:
            two_stage_payload = json.loads(two_stage_meta_src.read_text())
            two_stage_payload = self._repair_two_stage_meta_runtime_fields(
                models_dir=models_dir,
                room_name=room_name,
                meta_path=two_stage_meta_src,
                payload=two_stage_payload,
            )
        except Exception as exc:
            if require_mandatory:
                raise FileNotFoundError(
                    f"Invalid two-stage metadata for {room_name} v{int(version)}: {two_stage_meta_src.name}: {exc}"
                ) from exc
            _clear_two_stage_latest_aliases()
            return

        schema_ok = str(two_stage_payload.get("schema_version", "")).strip() == "beta6.two_stage_core.v1"
        runtime_enabled = bool(two_stage_payload.get("runtime_enabled", True))
        if not schema_ok or not runtime_enabled:
            _clear_two_stage_latest_aliases()
            return

        if not two_stage_stage_a_src.exists():
            if require_mandatory:
                raise FileNotFoundError(
                    f"Missing mandatory two-stage artifact for {room_name} v{int(version)}: {two_stage_stage_a_src.name}"
                )
            _clear_two_stage_latest_aliases()
            return

        shutil.copy2(two_stage_stage_a_src, two_stage_stage_a_dst)
        two_stage_meta_dst.write_text(json.dumps(two_stage_payload, indent=2))

        stage_b_enabled = bool(two_stage_payload.get("stage_b_enabled", False))
        if stage_b_enabled:
            if not two_stage_stage_b_src.exists():
                if require_mandatory:
                    raise FileNotFoundError(
                        f"Missing mandatory two-stage artifact for {room_name} v{int(version)}: {two_stage_stage_b_src.name}"
                    )
                _clear_two_stage_latest_aliases()
                return
            shutil.copy2(two_stage_stage_b_src, two_stage_stage_b_dst)
        elif two_stage_stage_b_dst.exists():
            two_stage_stage_b_dst.unlink()

    def rollback_to_version(self, elder_id: str, room_name: str, version: int) -> bool:
        """
        Roll back to a specific model version.
        
        Args:
            elder_id: Resident ID
            room_name: Room name
            version: Version number to roll back to
            
        Returns:
            True if successful, False if version not found
        """
        with self._acquire_room_lock(elder_id, room_name):
            info = self._load_version_info(elder_id, room_name)
            
            # Check version exists
            version_exists = any(v["version"] == version for v in info["versions"])
            if not version_exists:
                logger.error(f"Version {version} not found for {room_name}")
                return False
            version_meta = next(
                (v for v in info.get("versions", []) if int(v.get("version", 0)) == int(version)),
                None,
            )
            self._ensure_latest_aliases_match_current(elder_id, room_name, int(version))

            # Deferred-promotion flow: materialize shared backbone snapshot only when
            # this version is actually promoted/rolled back as latest.
            self._materialize_shared_backbone_snapshot_from_version(
                elder_id=elder_id,
                room_name=room_name,
                version=int(version),
                version_meta=version_meta,
            )
            
            # Update current version
            info["current_version"] = version
            info = self._reconcile_promotion_state(info)
            self._save_version_info(elder_id, room_name, info)
            
            logger.info(f"✓ Rolled back {room_name} to version {version}")
            return True

    def deactivate_current_version(self, elder_id: str, room_name: str) -> bool:
        """
        Deactivate current champion when no safe rollback target exists.

        This removes unversioned "latest" artifacts and clears current_version.
        Versioned files remain intact for audit/recovery.
        """
        with self._acquire_room_lock(elder_id, room_name):
            info = self._load_version_info(elder_id, room_name)
            current = int(info.get("current_version", 0) or 0)
            if current <= 0:
                return True

            models_dir = self.get_models_dir(elder_id)
            for suffix in (
                *self._MANDATORY_LATEST_SUFFIXES,
                *self._OPTIONAL_LATEST_SUFFIXES,
                *self._TWO_STAGE_LATEST_SUFFIXES,
            ):
                path = models_dir / f"{room_name}{suffix}"
                if path.exists():
                    path.unlink()

            for version_meta in info.get("versions", []):
                if int(version_meta.get("version", 0)) == current:
                    version_meta["promoted"] = False
                    break

            info["current_version"] = 0
            info = self._reconcile_promotion_state(info)
            self._save_version_info(elder_id, room_name, info)
            logger.warning(f"Deactivated current champion for {elder_id}/{room_name}")
            return True

    @staticmethod
    def _sanitize_token(raw: str) -> str:
        value = str(raw or "").strip()
        safe = "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in value)
        return safe or "unknown"

    @classmethod
    def _is_backbone_layer_name(cls, layer_name: str) -> bool:
        name = str(layer_name or "")
        return any(name.startswith(prefix) for prefix in cls._BACKBONE_LAYER_PREFIXES)

    @staticmethod
    def _extract_named_layer_weights(model, layer_selector) -> Dict[str, List[Any]]:
        payload: Dict[str, List[Any]] = {}
        for layer in getattr(model, "layers", []):
            name = str(getattr(layer, "name", ""))
            if not name or not layer_selector(name):
                continue
            if not getattr(layer, "weights", None):
                continue
            try:
                weights = layer.get_weights()
            except Exception:
                continue
            if weights:
                payload[name] = weights
        return payload

    def _get_backbone_store_dir(self, elder_id: str) -> Path:
        path = self.get_models_dir(elder_id) / "_shared_backbones"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_backbone_weights_path(self, elder_id: str, room_name: str, backbone_id: str) -> Path:
        room_safe = self._sanitize_token(room_name)
        backbone_safe = self._sanitize_token(backbone_id)
        return self._get_backbone_store_dir(elder_id) / f"{room_safe}_{backbone_safe}_backbone_weights.pkl"

    def save_shared_backbone_weights_if_needed(
        self,
        elder_id: str,
        room_name: str,
        backbone_id: str,
        model,
    ) -> Optional[Path]:
        if not backbone_id:
            return None
        path = self.get_backbone_weights_path(elder_id, room_name, backbone_id)
        if path.exists():
            return path
        payload = self._extract_named_layer_weights(
            model=model,
            layer_selector=lambda name: self._is_backbone_layer_name(name),
        )
        if not payload:
            return None
        joblib.dump(payload, path)
        logger.info(f"Saved shared backbone weights snapshot: {path.name}")
        return path

    def load_shared_backbone_weights(self, elder_id: str, room_name: str, backbone_id: str) -> Optional[Dict[str, List[Any]]]:
        if not backbone_id:
            return None
        path = self.get_backbone_weights_path(elder_id, room_name, backbone_id)
        if not path.exists():
            return None
        try:
            payload = joblib.load(path)
            return payload if isinstance(payload, dict) else None
        except Exception as e:
            logger.warning(f"Failed to load shared backbone weights from {path.name}: {e}")
            return None

    @staticmethod
    def apply_named_layer_weights(model, named_weights: Dict[str, List[Any]]) -> Dict[str, int]:
        if not named_weights:
            return {"loaded_layers": 0, "skipped_layers": 0, "attempted_layers": 0}
        layer_map = {str(getattr(layer, "name", "")): layer for layer in getattr(model, "layers", [])}
        loaded = 0
        skipped = 0
        attempted = 0
        for layer_name, weights in named_weights.items():
            attempted += 1
            layer = layer_map.get(layer_name)
            if layer is None:
                skipped += 1
                continue
            try:
                layer.set_weights(weights)
                loaded += 1
            except Exception:
                skipped += 1
        return {"loaded_layers": loaded, "skipped_layers": skipped, "attempted_layers": attempted}

    def _get_version_adapter_weights_path(self, models_dir: Path, room_name: str, version: int) -> Path:
        return models_dir / f"{room_name}_v{int(version)}_adapter_weights.pkl"

    def _get_latest_adapter_weights_path(self, models_dir: Path, room_name: str) -> Path:
        return models_dir / f"{room_name}_adapter_weights.pkl"

    def _load_latest_adapter_weights(self, elder_id: str, room_name: str) -> Optional[Dict[str, List[Any]]]:
        models_dir = self.get_models_dir(elder_id)
        path = self._get_latest_adapter_weights_path(models_dir, room_name)
        if not path.exists():
            return None
        try:
            payload = joblib.load(path)
            return payload if isinstance(payload, dict) else None
        except Exception as e:
            logger.warning(f"Failed to load adapter weights from {path.name}: {e}")
            return None

    def _build_runtime_shared_adapter_model(self, platform: Any, room_name: str, num_classes: int):
        from config import get_room_config
        from ml.transformer_backbone import build_transformer_model
        from elderlycare_v1_16.config.settings import DEFAULT_DROPOUT_RATE

        seq_length = int(get_room_config().calculate_seq_length(room_name))
        num_features = len(getattr(platform, "sensor_columns", []) or [])
        if num_features <= 0:
            raise ValueError("Platform sensor columns are unavailable for shared-adapter runtime model")
        input_shape = (seq_length, num_features)
        return build_transformer_model(
            input_shape=input_shape,
            num_classes=int(num_classes),
            d_model=64,
            num_heads=4,
            ff_dim=128,
            num_transformer_blocks=2,
            dropout_rate=DEFAULT_DROPOUT_RATE,
            positional_encoding_type='sinusoidal',
            use_cnn_embedding=True,
        )

    def _try_load_shared_adapter_room_model(
        self,
        elder_id: str,
        room_name: str,
        platform: Any,
    ) -> Optional[Any]:
        """
        Build runtime model from shared backbone + resident adapter artifacts.
        Returns model on success, else None to allow full-model fallback.
        """
        meta = self.get_current_version_metadata(elder_id, room_name)
        if not meta:
            return None
        identity = meta.get("model_identity") or {}
        if str(identity.get("family", "")).strip().lower() != "shared_backbone_adapter":
            return None
        backbone_id = str(identity.get("backbone_id", "")).strip()
        if not backbone_id:
            return None

        encoder = platform.label_encoders.get(room_name)
        classes = getattr(encoder, "classes_", None)
        if classes is None or len(classes) <= 0:
            return None
        num_classes = len(classes)

        backbone_weights = self.load_shared_backbone_weights(elder_id, room_name, backbone_id)
        adapter_weights = self._load_latest_adapter_weights(elder_id, room_name)
        if not backbone_weights or not adapter_weights:
            return None

        model = self._build_runtime_shared_adapter_model(platform, room_name, num_classes)
        backbone_stats = self.apply_named_layer_weights(model, backbone_weights)
        adapter_stats = self.apply_named_layer_weights(model, adapter_weights)
        min_backbone_layers = max(1, int(math.ceil(backbone_stats.get("attempted_layers", 0) * 0.6)))
        min_adapter_layers = max(1, int(math.ceil(adapter_stats.get("attempted_layers", 0) * 0.5)))
        if (
            int(backbone_stats.get("loaded_layers", 0)) < min_backbone_layers
            or int(adapter_stats.get("loaded_layers", 0)) < min_adapter_layers
        ):
            logger.warning(
                f"Shared-adapter compose coverage too low for {elder_id}/{room_name} "
                f"(backbone loaded={backbone_stats.get('loaded_layers', 0)}/"
                f"{backbone_stats.get('attempted_layers', 0)}, "
                f"adapter loaded={adapter_stats.get('loaded_layers', 0)}/"
                f"{adapter_stats.get('attempted_layers', 0)}). Falling back to full model."
            )
            return None
        logger.info(
            f"Loaded shared-adapter runtime model for {elder_id}/{room_name} "
            f"(backbone_id={backbone_id}, backbone_layers={backbone_stats['loaded_layers']}, "
            f"adapter_layers={adapter_stats['loaded_layers']})"
        )
        return model

    def save_adapter_weights(
        self,
        elder_id: str,
        room_name: str,
        version: int,
        model,
        promote_to_latest: bool,
    ) -> tuple[Optional[Path], Optional[Path]]:
        models_dir = self.get_models_dir(elder_id)
        payload = self._extract_named_layer_weights(
            model=model,
            layer_selector=lambda name: not self._is_backbone_layer_name(name),
        )
        if not payload:
            return None, None
        versioned_path = self._get_version_adapter_weights_path(models_dir, room_name, int(version))
        joblib.dump(payload, versioned_path)
        latest_path: Optional[Path] = None
        if promote_to_latest:
            latest_path = self._get_latest_adapter_weights_path(models_dir, room_name)
            shutil.copy2(versioned_path, latest_path)
        return versioned_path, latest_path

    def _materialize_shared_backbone_snapshot_from_version(
        self,
        elder_id: str,
        room_name: str,
        version: int,
        version_meta: Optional[Dict[str, Any]],
    ) -> None:
        if not isinstance(version_meta, dict):
            return
        identity = version_meta.get("model_identity") or {}
        if str(identity.get("family", "")).strip().lower() != "shared_backbone_adapter":
            return
        backbone_id = str(identity.get("backbone_id", "")).strip()
        if not backbone_id:
            return
        if self.load_shared_backbone_weights(elder_id, room_name, backbone_id):
            return

        model_path = self.get_models_dir(elder_id) / f"{room_name}_v{int(version)}_model.keras"
        if not model_path.exists():
            logger.warning(
                f"Cannot materialize shared backbone snapshot for {elder_id}/{room_name} "
                f"v{version}: missing model artifact {model_path.name}"
            )
            return

        try:
            model = self.load_room_model(str(model_path), room_name, compile_model=False)
        except Exception as e:
            logger.warning(
                f"Failed to load model for shared backbone snapshot materialization "
                f"({elder_id}/{room_name} v{version}): {e}"
            )
            return

        saved_path = self.save_shared_backbone_weights_if_needed(
            elder_id=elder_id,
            room_name=room_name,
            backbone_id=backbone_id,
            model=model,
        )
        if saved_path is None:
            logger.warning(
                f"Shared backbone snapshot materialization yielded no payload for "
                f"{elder_id}/{room_name} v{version} (backbone_id={backbone_id})"
            )

    # =========================================================================
    # Core Load/Save Methods
    # =========================================================================

    def load_room_model(self, model_path: str, room_name: str, compile_model: bool = True):
        """
        Load a single room's Keras model with joblib fallback.
        """
        try:
            return tf.keras.models.load_model(
                model_path,
                custom_objects=self._get_custom_objects(),
                compile=bool(compile_model),
            )
        except (tf.errors.OpError, ValueError, OSError, KeyError) as keras_err:
            logger.warning(f"Keras load failed for {room_name}: {keras_err}, trying joblib fallback")
            try:
                return joblib.load(model_path)
            except Exception as joblib_err:
                raise ModelLoadError(
                    f"All loading methods failed for {room_name}: Keras={keras_err}, Joblib={joblib_err}"
                ) from joblib_err

    def _load_two_stage_core_bundle(
        self,
        *,
        elder_id: str,
        room_name: str,
        current_version: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Load latest promoted two-stage core artifacts when they match current champion version.
        """
        models_dir = self.get_models_dir(elder_id)
        meta_path = models_dir / f"{room_name}_two_stage_meta.json"
        stage_a_path = models_dir / f"{room_name}_two_stage_stage_a_model.keras"
        stage_b_path = models_dir / f"{room_name}_two_stage_stage_b_model.keras"
        if not meta_path.exists() or not stage_a_path.exists():
            return None
        try:
            payload = json.loads(meta_path.read_text())
            payload = self._repair_two_stage_meta_runtime_fields(
                models_dir=models_dir,
                room_name=room_name,
                meta_path=meta_path,
                payload=payload,
            )
        except Exception as e:
            logger.warning(f"Failed reading two-stage metadata for {elder_id}/{room_name}: {e}")
            return None
        if str(payload.get("schema_version", "")).strip() != "beta6.two_stage_core.v1":
            return None
        if payload.get("runtime_enabled") is False:
            logger.info(
                "Skipping runtime-disabled two-stage artifacts for %s/%s",
                elder_id,
                room_name,
            )
            return None
        try:
            saved_version = int(payload.get("saved_version", 0) or 0)
        except (TypeError, ValueError):
            return None
        if int(current_version) <= 0 or saved_version != int(current_version):
            logger.info(
                "Skipping stale two-stage artifacts for %s/%s: saved_version=%s current_version=%s",
                elder_id,
                room_name,
                saved_version,
                current_version,
            )
            return None

        stage_a_model = self.load_room_model(
            str(stage_a_path),
            f"{room_name}_two_stage_stage_a",
            compile_model=False,
        )
        stage_b_model = None
        stage_b_enabled = bool(payload.get("stage_b_enabled", False))
        if stage_b_enabled:
            if not stage_b_path.exists():
                logger.warning(
                    "Two-stage metadata expects stage_b model but file missing for %s/%s",
                    elder_id,
                    room_name,
                )
                return None
            stage_b_model = self.load_room_model(
                str(stage_b_path),
                f"{room_name}_two_stage_stage_b",
                compile_model=False,
            )

        raw_stage_a_threshold = payload.get("stage_a_occupied_threshold", 0.5)
        try:
            stage_a_threshold = float(raw_stage_a_threshold)
        except (TypeError, ValueError):
            stage_a_threshold = 0.5

        return {
            "schema_version": str(payload.get("schema_version")),
            "saved_version": int(saved_version),
            "stage_a_model": stage_a_model,
            "stage_b_model": stage_b_model,
            "stage_b_enabled": bool(stage_b_enabled),
            "stage_a_occupied_threshold": stage_a_threshold,
            "stage_a_threshold_source": str(payload.get("stage_a_threshold_source") or "meta"),
            "stage_a_calibration": dict(payload.get("stage_a_calibration") or {}),
            "num_classes": int(payload.get("num_classes", 0) or 0),
            "excluded_class_ids": [int(v) for v in (payload.get("excluded_class_ids") or [])],
            "occupied_class_ids": [int(v) for v in (payload.get("occupied_class_ids") or [])],
            "primary_occupied_class_id": int(payload.get("primary_occupied_class_id", -1) or -1),
            "meta": payload,
        }

    @staticmethod
    def _is_discoverable_room_model_alias(filename: str) -> bool:
        if not filename.endswith("_model.keras") or "_v" in filename:
            return False
        reserved_two_stage_suffixes = (
            "_two_stage_stage_a_model.keras",
            "_two_stage_stage_b_model.keras",
        )
        return not filename.endswith(reserved_two_stage_suffixes)
                
    def load_models_for_elder(self, elder_id: str, platform: Any) -> List[str]:
        """
        Load all available models for an elder into the platform object.
        
        Loads the "latest" (unversioned) model files for each room.
        
        Args:
            elder_id: Resident ID
            platform: ElderlyCarePlatform instance to populate
            
        Returns:
            List of loaded room names
        """
        models_dir = self.get_models_dir(elder_id)
        loaded_rooms = []
        
        if not models_dir.exists():
            logger.warning(f"No models directory found for {elder_id}")
            return []

        # Store calibrated per-class thresholds on the platform object.
        if not isinstance(getattr(platform, 'class_thresholds', None), dict):
            platform.class_thresholds = {}
        if not isinstance(getattr(platform, 'activity_confidence_artifacts', None), dict):
            platform.activity_confidence_artifacts = {}
        if not isinstance(getattr(platform, 'two_stage_core_models', None), dict):
            platform.two_stage_core_models = {}
        
        # Discover candidate rooms from both latest aliases and version metadata.
        room_names = set()
        for f in os.listdir(models_dir):
            if self._is_discoverable_room_model_alias(f):
                room_names.add(f.replace("_model.keras", ""))
            if f.endswith("_versions.json"):
                room_names.add(f.replace("_versions.json", ""))

        for room_name in sorted(room_names):
            try:
                # P0 Hardening: Audit and repair registry state before loading.
                # This ensures we strictly serve the promoted champion and fix any alias drift.
                repair_report = self.validate_and_repair_room_registry_state(elder_id, room_name)
                if repair_report.get("repaired"):
                     logger.info(f"Registry auto-repaired for {room_name}: {repair_report['issues']}")
                if not bool(repair_report.get("valid", True)):
                    logger.error(
                        f"Skipping room due to unresolved registry state for {room_name}: "
                        f"{repair_report.get('issues')}"
                    )
                    continue
                
                model_path = str(models_dir / f"{room_name}_model.keras")
                scaler_path = str(models_dir / f"{room_name}_scaler.pkl")
                encoder_path = str(models_dir / f"{room_name}_label_encoder.pkl")
                thresholds_path = models_dir / f"{room_name}_thresholds.json"
                activity_confidence_path = models_dir / f"{room_name}_activity_confidence_calibrator.json"
                
                if not (os.path.exists(scaler_path) and os.path.exists(encoder_path)):
                    continue

                # Load components needed in both loading modes.
                platform.scalers[room_name] = joblib.load(scaler_path)
                platform.label_encoders[room_name] = joblib.load(encoder_path)

                # Prefer composing model from shared backbone + adapter artifacts.
                composed_model = self._try_load_shared_adapter_room_model(
                    elder_id=elder_id,
                    room_name=room_name,
                    platform=platform,
                )
                if composed_model is not None:
                    platform.room_models[room_name] = composed_model
                else:
                    # Fallback to latest full model alias.
                    platform.room_models[room_name] = self.load_room_model(model_path, room_name)

                # Load calibrated per-class thresholds if available.
                if thresholds_path.exists():
                    try:
                        with open(thresholds_path, 'r') as f:
                            loaded_thresholds = json.load(f)
                        platform.class_thresholds[room_name] = {
                            str(k): float(v) for k, v in loaded_thresholds.items()
                        }
                    except Exception as e:
                        logger.warning(f"Failed to load thresholds for {room_name}: {e}")
                        platform.class_thresholds[room_name] = {}
                else:
                    platform.class_thresholds[room_name] = {}

                if activity_confidence_path.exists():
                    try:
                        with open(activity_confidence_path, 'r') as f:
                            platform.activity_confidence_artifacts[room_name] = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load activity confidence artifact for {room_name}: {e}")
                        platform.activity_confidence_artifacts[room_name] = {}
                else:
                    platform.activity_confidence_artifacts[room_name] = {}

                current_version = int(self.get_current_version(elder_id, room_name) or 0)
                two_stage_bundle = self._load_two_stage_core_bundle(
                    elder_id=elder_id,
                    room_name=room_name,
                    current_version=current_version,
                )
                if two_stage_bundle is not None:
                    platform.two_stage_core_models[room_name] = two_stage_bundle
                else:
                    platform.two_stage_core_models.pop(room_name, None)
                
                loaded_rooms.append(room_name)
            except ModelLoadError as e:
                platform.room_models.pop(room_name, None)
                platform.scalers.pop(room_name, None)
                platform.label_encoders.pop(room_name, None)
                if hasattr(platform, "class_thresholds"):
                    platform.class_thresholds.pop(room_name, None)
                if hasattr(platform, "activity_confidence_artifacts"):
                    platform.activity_confidence_artifacts.pop(room_name, None)
                if hasattr(platform, "two_stage_core_models"):
                    platform.two_stage_core_models.pop(room_name, None)
                logger.error(f"Skipping room due to model load failure for {room_name}: {e}")
            except (tf.errors.OpError, ValueError, OSError, KeyError) as e:
                platform.room_models.pop(room_name, None)
                platform.scalers.pop(room_name, None)
                platform.label_encoders.pop(room_name, None)
                if hasattr(platform, "class_thresholds"):
                    platform.class_thresholds.pop(room_name, None)
                if hasattr(platform, "activity_confidence_artifacts"):
                    platform.activity_confidence_artifacts.pop(room_name, None)
                if hasattr(platform, "two_stage_core_models"):
                    platform.two_stage_core_models.pop(room_name, None)
                logger.error(f"Failed to load model for {room_name}: {type(e).__name__}: {e}")
            except Exception as e:
                platform.room_models.pop(room_name, None)
                platform.scalers.pop(room_name, None)
                platform.label_encoders.pop(room_name, None)
                if hasattr(platform, "class_thresholds"):
                    platform.class_thresholds.pop(room_name, None)
                if hasattr(platform, "activity_confidence_artifacts"):
                    platform.activity_confidence_artifacts.pop(room_name, None)
                if hasattr(platform, "two_stage_core_models"):
                    platform.two_stage_core_models.pop(room_name, None)
                logger.error(f"Unexpected error loading {room_name}: {e}")
                
        return loaded_rooms
        
    def save_model_artifacts(
        self, 
        elder_id: str, 
        room_name: str, 
        model, 
        scaler, 
        encoder,
        accuracy: float = None,
        samples: int = None,
        class_thresholds: Optional[Dict[int, float]] = None,
        activity_confidence_artifact: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        model_identity: Optional[Dict[str, Any]] = None,
        promote_to_latest: bool = True,
        parent_version_id: Optional[int] = None,
        cleanup_old_versions: bool = True,
    ) -> int:
        """
        Save model, scaler, and encoder to disk with versioning.
        
        Args:
            elder_id: Resident ID
            room_name: Room name
            model: Keras model to save
            scaler: Fitted scaler
            encoder: Fitted label encoder
            accuracy: Optional training accuracy for metadata
            samples: Optional sample count for metadata
            metrics: Optional rich metrics payload (e.g., macro_f1)
            model_identity: Optional model lineage payload (e.g., backbone_id/adapter_id).
            promote_to_latest: If True, promotes this version to active "latest".
            parent_version_id: Optional champion/source version used for warm-start lineage.
            cleanup_old_versions: Whether to prune old versioned artifacts after saving.
            
        Returns:
            Version number of the saved model
        """
        with self._acquire_room_lock(elder_id, room_name):
            models_dir = self.get_models_dir(elder_id)
            version = self._get_next_version(elder_id, room_name)
            
            # Save versioned files
            model_path = models_dir / f"{room_name}_v{version}_model.keras"
            scaler_path = models_dir / f"{room_name}_v{version}_scaler.pkl"
            encoder_path = models_dir / f"{room_name}_v{version}_label_encoder.pkl"
            thresholds_path = models_dir / f"{room_name}_v{version}_thresholds.json"
            activity_confidence_path = models_dir / f"{room_name}_v{version}_activity_confidence_calibrator.json"
            
            model.save(str(model_path))
            joblib.dump(scaler, scaler_path)
            joblib.dump(encoder, encoder_path)
            if class_thresholds is not None:
                serializable_thresholds = {str(k): float(v) for k, v in class_thresholds.items()}
                with open(thresholds_path, 'w') as f:
                    json.dump(serializable_thresholds, f, indent=2)
            if activity_confidence_artifact is not None:
                with open(activity_confidence_path, 'w') as f:
                    json.dump(activity_confidence_artifact, f, indent=2)
            
            identity = model_identity or {}
            model_family = str(identity.get("family", "")).strip().lower()
            adapter_saved_path = None
            backbone_saved_path = None
            if model_family == "shared_backbone_adapter":
                adapter_saved_path, _ = self.save_adapter_weights(
                    elder_id=elder_id,
                    room_name=room_name,
                    version=version,
                    model=model,
                    promote_to_latest=promote_to_latest,
                )
                if promote_to_latest:
                    backbone_saved_path = self.save_shared_backbone_weights_if_needed(
                        elder_id=elder_id,
                        room_name=room_name,
                        backbone_id=str(identity.get("backbone_id", "")).strip(),
                        model=model,
                    )

            # Optionally promote this version as "latest" (active champion).
            if promote_to_latest:
                self._ensure_latest_aliases_match_current(elder_id, room_name, int(version))
            
            # Update version info
            info = self._load_version_info(elder_id, room_name)
            grouped_date_stability = None
            promotion_time_drift_summary = None
            if isinstance(metrics, dict):
                if metrics.get("grouped_date_stability") is not None:
                    grouped_date_stability = copy.deepcopy(metrics.get("grouped_date_stability"))
                if metrics.get("promotion_time_drift_summary") is not None:
                    promotion_time_drift_summary = copy.deepcopy(metrics.get("promotion_time_drift_summary"))
            info["versions"].append({
                "version": version,
                "created_at": datetime.now().isoformat(),
                "accuracy": accuracy,
                "samples": samples,
                "metrics": metrics or {},
                "grouped_date_stability": grouped_date_stability,
                "promotion_time_drift_summary": promotion_time_drift_summary,
                "model_identity": model_identity or {},
                "adapter_weights_path": adapter_saved_path.name if adapter_saved_path else None,
                "backbone_weights_path": backbone_saved_path.name if backbone_saved_path else None,
                "promoted": bool(promote_to_latest),
                "parent_version_id": int(parent_version_id) if parent_version_id is not None else None,
            })
            if promote_to_latest:
                info["current_version"] = version
            info = self._reconcile_promotion_state(info)
            self._save_version_info(elder_id, room_name, info)
            
            # Cleanup old versions
            if cleanup_old_versions:
                self._cleanup_old_versions(elder_id, room_name)
            
            if promote_to_latest:
                logger.info(f"✓ Saved and promoted {room_name} model v{version} for {elder_id}")
            else:
                logger.info(f"✓ Saved candidate-only {room_name} model v{version} for {elder_id} (not promoted)")
            return version
