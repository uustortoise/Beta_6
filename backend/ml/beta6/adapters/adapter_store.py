"""Phase 5 adapter lifecycle store with promote/rollback/retire semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Mapping, Optional

from ..beta6_schema import load_validated_beta6_config
from .lora_adapter import LoRAAdapterArtifact, load_adapter_artifact, save_adapter_artifact


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(raw: str) -> datetime:
    dt = datetime.fromisoformat(str(raw))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class AdapterStorePolicy:
    max_versions_per_resident: int = 14
    min_warmup_accuracy: float = 0.55
    retirement_inactive_days: int = 30
    enable_auto_retire: bool = True


def load_adapter_store_policy(path: str | Path | None) -> AdapterStorePolicy:
    policy_path = (
        Path(path).resolve()
        if path is not None
        else Path(__file__).resolve().parents[3] / "config" / "beta6_adapter_policy.yaml"
    )
    payload = load_validated_beta6_config(
        policy_path,
        expected_filename="beta6_adapter_policy.yaml",
    )
    section = payload.get("adapter")
    if not isinstance(section, Mapping):
        section = {}
    return AdapterStorePolicy(
        max_versions_per_resident=max(int(section.get("max_versions_per_resident", 14)), 1),
        min_warmup_accuracy=min(max(float(section.get("min_warmup_accuracy", 0.55)), 0.0), 1.0),
        retirement_inactive_days=max(int(section.get("retirement_inactive_days", 30)), 1),
        enable_auto_retire=bool(section.get("enable_auto_retire", True)),
    )


class AdapterStore:
    """Filesystem-backed adapter store with deterministic lifecycle transitions."""

    def __init__(
        self,
        *,
        root: str | Path,
        policy: Optional[AdapterStorePolicy] = None,
    ):
        self.root = Path(root).resolve()
        self.policy = policy or load_adapter_store_policy(None)
        self.root.mkdir(parents=True, exist_ok=True)

    def _room_dir(self, resident_id: str, room: str) -> Path:
        return self.root / str(resident_id) / str(room).strip().lower()

    def _adapters_dir(self, resident_id: str, room: str) -> Path:
        return self._room_dir(resident_id, room) / "adapters"

    def _adapter_dir(self, resident_id: str, room: str, adapter_id: str) -> Path:
        return self._adapters_dir(resident_id, room) / str(adapter_id)

    def _lifecycle_path(self, resident_id: str, room: str, adapter_id: str) -> Path:
        return self._adapter_dir(resident_id, room, adapter_id) / "lifecycle.json"

    def _pointer_path(self, resident_id: str, room: str) -> Path:
        return self._room_dir(resident_id, room) / "active_adapter.json"

    def _history_path(self, resident_id: str, room: str) -> Path:
        return self._room_dir(resident_id, room) / "adapter_history.jsonl"

    def _rollback_target_eligible(self, lifecycle: Mapping[str, Any]) -> bool:
        status = str(lifecycle.get("status", "")).strip().lower()
        if status == "retired":
            return False
        if bool(lifecycle.get("warmup_pass", False)):
            return True
        # Backward compatibility for previously promoted artifacts missing warmup flag.
        return status == "promoted"

    def _write_json_atomic(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _append_history(self, resident_id: str, room: str, event: Mapping[str, Any]) -> None:
        path = self._history_path(resident_id, room)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(event), sort_keys=True))
            handle.write("\n")

    def list_adapters(self, resident_id: str, room: str) -> List[Dict[str, Any]]:
        base = self._adapters_dir(resident_id, room)
        if not base.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for item in sorted(base.iterdir()):
            if not item.is_dir():
                continue
            lifecycle = self._read_json(item / "lifecycle.json") or {}
            metadata = self._read_json(item / "metadata.json") or {}
            rows.append(
                {
                    "adapter_id": str(item.name),
                    "status": str(lifecycle.get("status", "created")),
                    "created_at": str(metadata.get("created_at") or lifecycle.get("created_at") or ""),
                    "promoted_at": str(lifecycle.get("promoted_at") or ""),
                    "retired_at": str(lifecycle.get("retired_at") or ""),
                    "warmup_accuracy": float(lifecycle.get("warmup_accuracy", metadata.get("warmup_accuracy", 0.0))),
                }
            )
        rows.sort(key=lambda row: row.get("created_at", ""))
        return rows

    def load_adapter(self, resident_id: str, room: str, adapter_id: str) -> LoRAAdapterArtifact:
        return load_adapter_artifact(self._adapter_dir(resident_id, room, adapter_id))

    def get_active_adapter(self, resident_id: str, room: str) -> Optional[Dict[str, Any]]:
        return self._read_json(self._pointer_path(resident_id, room))

    def create_adapter(
        self,
        *,
        artifact: LoRAAdapterArtifact,
        run_id: str,
    ) -> Dict[str, Any]:
        adapter_dir = self._adapter_dir(artifact.resident_id, artifact.room, artifact.adapter_id)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        saved = save_adapter_artifact(artifact, output_dir=adapter_dir)
        lifecycle = {
            "adapter_id": artifact.adapter_id,
            "status": "created",
            "created_at": artifact.created_at,
            "run_id": str(run_id),
            "warmup_accuracy": float(artifact.warmup_accuracy),
            "row_count": int(artifact.row_count),
            "artifacts": saved,
        }
        self._write_json_atomic(self._lifecycle_path(artifact.resident_id, artifact.room, artifact.adapter_id), lifecycle)
        self._append_history(
            artifact.resident_id,
            artifact.room,
            {
                "event": "adapter_created",
                "recorded_at": _utc_now(),
                "adapter_id": artifact.adapter_id,
                "run_id": str(run_id),
            },
        )
        return lifecycle

    def warmup_adapter(
        self,
        *,
        resident_id: str,
        room: str,
        adapter_id: str,
        warmup_accuracy: float,
        run_id: str,
    ) -> Dict[str, Any]:
        path = self._lifecycle_path(resident_id, room, adapter_id)
        lifecycle = self._read_json(path)
        if not isinstance(lifecycle, dict):
            raise ValueError(f"adapter lifecycle not found: {resident_id}/{room}/{adapter_id}")
        lifecycle["status"] = "warm"
        lifecycle["warmed_at"] = _utc_now()
        lifecycle["warmup_accuracy"] = float(warmup_accuracy)
        lifecycle["warmup_pass"] = bool(float(warmup_accuracy) >= float(self.policy.min_warmup_accuracy))
        lifecycle["run_id"] = str(run_id)
        self._write_json_atomic(path, lifecycle)
        self._append_history(
            resident_id,
            room,
            {
                "event": "adapter_warmup",
                "recorded_at": _utc_now(),
                "adapter_id": str(adapter_id),
                "run_id": str(run_id),
                "warmup_accuracy": float(warmup_accuracy),
                "warmup_pass": bool(lifecycle["warmup_pass"]),
            },
        )
        return lifecycle

    def promote_adapter(
        self,
        *,
        resident_id: str,
        room: str,
        adapter_id: str,
        run_id: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        lifecycle_path = self._lifecycle_path(resident_id, room, adapter_id)
        lifecycle = self._read_json(lifecycle_path)
        if not isinstance(lifecycle, dict):
            raise ValueError(f"adapter lifecycle not found: {resident_id}/{room}/{adapter_id}")
        if bool(lifecycle.get("warmup_pass")) is not True:
            raise ValueError(
                f"adapter warm-up gate not passed for promotion: {resident_id}/{room}/{adapter_id}"
            )

        pointer = {
            "adapter_id": str(adapter_id),
            "resident_id": str(resident_id),
            "room": str(room).strip().lower(),
            "run_id": str(run_id),
            "promoted_at": _utc_now(),
            "metadata": dict(metadata or {}),
        }
        self._write_json_atomic(self._pointer_path(resident_id, room), pointer)
        lifecycle["status"] = "promoted"
        lifecycle["promoted_at"] = pointer["promoted_at"]
        lifecycle["run_id"] = str(run_id)
        self._write_json_atomic(lifecycle_path, lifecycle)
        self._append_history(
            resident_id,
            room,
            {
                "event": "adapter_promoted",
                "recorded_at": _utc_now(),
                "adapter_id": str(adapter_id),
                "run_id": str(run_id),
            },
        )
        self.enforce_retention(resident_id=resident_id, room=room)
        return pointer

    def rollback_adapter(
        self,
        *,
        resident_id: str,
        room: str,
        run_id: str,
        target_adapter_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        adapters = self.list_adapters(resident_id, room)
        if not adapters:
            raise ValueError(f"no adapters found for rollback: {resident_id}/{room}")
        current = self.get_active_adapter(resident_id, room)
        current_id = str((current or {}).get("adapter_id", ""))
        target_id = str(target_adapter_id or "").strip()
        if not target_id:
            candidates = []
            for row in adapters:
                adapter_id = str(row.get("adapter_id", ""))
                if not adapter_id or adapter_id == current_id:
                    continue
                lifecycle = self._read_json(self._lifecycle_path(resident_id, room, adapter_id)) or {}
                if self._rollback_target_eligible(lifecycle):
                    candidates.append(row)
            if not candidates:
                raise ValueError(f"no rollback target available: {resident_id}/{room}")
            target_id = str(candidates[-1]["adapter_id"])
        target = next((row for row in adapters if row["adapter_id"] == target_id), None)
        if target is None:
            raise ValueError(f"rollback target adapter not found: {target_id}")
        target_lifecycle = self._read_json(self._lifecycle_path(resident_id, room, target_id)) or {}
        if not self._rollback_target_eligible(target_lifecycle):
            raise ValueError(
                f"rollback target adapter is not rollback-eligible: {resident_id}/{room}/{target_id}"
            )

        pointer = {
            "adapter_id": target_id,
            "resident_id": str(resident_id),
            "room": str(room).strip().lower(),
            "run_id": str(run_id),
            "promoted_at": _utc_now(),
            "metadata": {"rollback": True, "previous_adapter_id": current_id or None},
        }
        self._write_json_atomic(self._pointer_path(resident_id, room), pointer)
        self._append_history(
            resident_id,
            room,
            {
                "event": "adapter_rollback",
                "recorded_at": _utc_now(),
                "adapter_id": target_id,
                "previous_adapter_id": current_id or None,
                "run_id": str(run_id),
            },
        )
        return pointer

    def retire_adapter(
        self,
        *,
        resident_id: str,
        room: str,
        adapter_id: str,
        reason: str,
        run_id: str,
    ) -> Dict[str, Any]:
        lifecycle_path = self._lifecycle_path(resident_id, room, adapter_id)
        lifecycle = self._read_json(lifecycle_path)
        if not isinstance(lifecycle, dict):
            raise ValueError(f"adapter lifecycle not found: {resident_id}/{room}/{adapter_id}")
        lifecycle["status"] = "retired"
        lifecycle["retired_at"] = _utc_now()
        lifecycle["retire_reason"] = str(reason).strip() or "manual"
        lifecycle["run_id"] = str(run_id)
        self._write_json_atomic(lifecycle_path, lifecycle)
        self._append_history(
            resident_id,
            room,
            {
                "event": "adapter_retired",
                "recorded_at": _utc_now(),
                "adapter_id": str(adapter_id),
                "reason": lifecycle["retire_reason"],
                "run_id": str(run_id),
            },
        )
        return lifecycle

    def enforce_retention(self, *, resident_id: str, room: str) -> Dict[str, Any]:
        adapters = self.list_adapters(resident_id, room)
        max_versions = int(self.policy.max_versions_per_resident)
        if len(adapters) <= max_versions:
            return {"removed": [], "kept": [row["adapter_id"] for row in adapters]}

        active_id = str((self.get_active_adapter(resident_id, room) or {}).get("adapter_id", ""))
        removable = [row for row in adapters if row["adapter_id"] != active_id]
        removable.sort(key=lambda row: row.get("created_at", ""))
        to_remove_count = max(0, len(adapters) - max_versions)
        removed: List[str] = []
        for row in removable[:to_remove_count]:
            adapter_id = str(row["adapter_id"])
            adapter_dir = self._adapter_dir(resident_id, room, adapter_id)
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)
            removed.append(adapter_id)
            self._append_history(
                resident_id,
                room,
                {
                    "event": "adapter_pruned_by_retention",
                    "recorded_at": _utc_now(),
                    "adapter_id": adapter_id,
                },
            )
        kept = [row["adapter_id"] for row in self.list_adapters(resident_id, room)]
        return {"removed": removed, "kept": kept}

    def auto_retire_inactive(self, *, resident_id: str, room: str, run_id: str) -> Dict[str, Any]:
        if not self.policy.enable_auto_retire:
            return {"status": "disabled", "retired": []}
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(self.policy.retirement_inactive_days))
        active_id = str((self.get_active_adapter(resident_id, room) or {}).get("adapter_id", ""))
        retired: List[str] = []
        for row in self.list_adapters(resident_id, room):
            adapter_id = str(row["adapter_id"])
            if adapter_id == active_id:
                continue
            created_at = str(row.get("created_at", "")).strip()
            if not created_at:
                continue
            if _parse_iso(created_at) <= cutoff:
                self.retire_adapter(
                    resident_id=resident_id,
                    room=room,
                    adapter_id=adapter_id,
                    reason="inactive_timeout",
                    run_id=run_id,
                )
                retired.append(adapter_id)
        return {"status": "ok", "retired": retired}


__all__ = [
    "AdapterStore",
    "AdapterStorePolicy",
    "load_adapter_store_policy",
]
