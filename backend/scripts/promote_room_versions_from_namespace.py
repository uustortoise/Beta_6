from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ml.registry import ModelRegistry


def _parse_room_versions(raw_rooms: list[str], raw_versions: list[str]) -> dict[str, int]:
    room_versions: dict[str, int | None] = {str(room): None for room in raw_rooms}
    for raw in raw_versions:
        room_name, sep, version_text = str(raw).partition("=")
        if not sep or not room_name.strip() or not version_text.strip():
            raise ValueError(f"Invalid --version entry: {raw!r}; expected ROOM=VERSION")
        room_versions[str(room_name).strip()] = int(version_text)
    resolved: dict[str, int] = {}
    for room_name, version in room_versions.items():
        if version is None:
            continue
        resolved[str(room_name)] = int(version)
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version_entry_by_id(info: dict[str, Any], version: int) -> dict[str, Any] | None:
    for item in info.get("versions", []):
        if int(item.get("version", 0)) == int(version):
            return dict(item)
    return None


def _normalize_version_entry(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(entry))
    normalized["version"] = int(normalized.get("version", 0))
    normalized["promoted"] = False
    return normalized


def _versioned_artifacts(models_dir: Path, room_name: str, version: int) -> dict[str, Path]:
    pattern = f"{room_name}_v{int(version)}_*"
    return {path.name: path for path in models_dir.glob(pattern) if path.is_file()}


def _latest_artifact_names(room_name: str, versioned_names: set[str], version: int) -> list[tuple[Path, str]]:
    del versioned_names, version
    results: list[tuple[Path, str]] = []
    for suffix in (
        "_model.keras",
        "_scaler.pkl",
        "_label_encoder.pkl",
        "_thresholds.json",
        "_adapter_weights.pkl",
        "_activity_confidence_calibrator.json",
        "_two_stage_meta.json",
        "_two_stage_stage_a_model.keras",
        "_two_stage_stage_b_model.keras",
    ):
        results.append((Path(f"{room_name}{suffix}"), f"{room_name}{suffix}"))
    return results


def _assert_compatible_duplicate(
    room_name: str,
    version: int,
    source_entry: dict[str, Any],
    target_entry: dict[str, Any],
    source_artifacts: dict[str, Path],
    target_artifacts: dict[str, Path],
) -> None:
    if _normalize_version_entry(source_entry) != _normalize_version_entry(target_entry):
        raise ValueError(
            f"Conflicting metadata for {room_name} v{int(version)} between source and target namespaces"
        )

    for artifact_name, source_path in source_artifacts.items():
        target_path = target_artifacts.get(artifact_name)
        if target_path is None or not target_path.exists():
            continue
        if _sha256(source_path) != _sha256(target_path):
            raise ValueError(
                f"Conflicting artifact for {room_name} v{int(version)}: {artifact_name}"
            )


def promote_room_versions_from_namespace(
    *,
    backend_dir: str,
    source_elder_id: str,
    target_elder_id: str,
    room_versions: dict[str, int] | None = None,
) -> dict[str, Any]:
    registry = ModelRegistry(backend_dir=backend_dir)
    summary: dict[str, Any] = {
        "source_elder_id": str(source_elder_id),
        "target_elder_id": str(target_elder_id),
        "rooms": [],
    }

    if room_versions:
        requested_rooms = [str(room) for room in room_versions.keys()]
    else:
        source_models_dir = registry.get_models_dir(source_elder_id)
        requested_rooms = sorted(
            path.name.replace("_versions.json", "")
            for path in source_models_dir.glob("*_versions.json")
        )

    for room_name in requested_rooms:
        source_info = registry._load_version_info(source_elder_id, room_name)
        target_info = registry._load_version_info(target_elder_id, room_name)
        source_current = int(source_info.get("current_version", 0) or 0)
        selected_version = int((room_versions or {}).get(room_name) or source_current)
        if selected_version <= 0:
            raise ValueError(f"No selected version for {source_elder_id}/{room_name}")

        source_entry = _version_entry_by_id(source_info, selected_version)
        if source_entry is None:
            raise ValueError(
                f"Requested version {selected_version} not found for {source_elder_id}/{room_name}"
            )

        source_models_dir = registry.get_models_dir(source_elder_id)
        target_models_dir = registry.get_models_dir(target_elder_id)

        target_versions = {
            int(item.get("version", 0)): dict(item)
            for item in target_info.get("versions", [])
            if int(item.get("version", 0)) > 0
        }
        copied_versions: list[int] = []
        reused_versions: list[int] = []

        for entry in source_info.get("versions", []):
            version = int(entry.get("version", 0) or 0)
            if version <= 0:
                continue
            normalized_entry = _normalize_version_entry(dict(entry))
            source_artifacts = _versioned_artifacts(source_models_dir, room_name, version)
            target_artifacts = _versioned_artifacts(target_models_dir, room_name, version)

            existing_entry = target_versions.get(version)
            if existing_entry is not None:
                _assert_compatible_duplicate(
                    room_name,
                    version,
                    normalized_entry,
                    existing_entry,
                    source_artifacts,
                    target_artifacts,
                )
                reused_versions.append(version)
            else:
                target_versions[version] = normalized_entry
                copied_versions.append(version)

            for artifact_name, source_path in source_artifacts.items():
                target_path = target_models_dir / artifact_name
                if target_path.exists():
                    if _sha256(source_path) != _sha256(target_path):
                        raise ValueError(
                            f"Refusing to overwrite conflicting target artifact: {artifact_name}"
                        )
                    continue
                shutil.copy2(source_path, target_path)

        merged_info = {
            "versions": sorted(target_versions.values(), key=lambda item: int(item.get("version", 0)), reverse=True),
            "current_version": int(target_info.get("current_version", 0) or 0),
        }
        merged_info = registry._reconcile_promotion_state(merged_info)
        registry._save_version_info(target_elder_id, room_name, merged_info)

        if not registry.rollback_to_version(target_elder_id, room_name, selected_version):
            raise ValueError(
                f"Failed to promote {target_elder_id}/{room_name} to version {selected_version}"
            )

        promoted_names = {
            name for name in _versioned_artifacts(target_models_dir, room_name, selected_version).keys()
        }
        latest_names = [
            latest_name
            for latest_path, latest_name in _latest_artifact_names(room_name, promoted_names, selected_version)
            if (target_models_dir / latest_path.name).exists()
        ]

        summary["rooms"].append(
            {
                "room": str(room_name),
                "selected_version": int(selected_version),
                "target_current_version": int(registry.get_current_version(target_elder_id, room_name) or 0),
                "copied_versions": sorted(copied_versions),
                "reused_versions": sorted(reused_versions),
                "imported_version_count": len(copied_versions) + len(reused_versions),
                "latest_artifacts": sorted(latest_names),
            }
        )

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote selected room versions from a candidate namespace into a target namespace while preserving target rollback history."
    )
    parser.add_argument("--backend-dir", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--source-elder-id", required=True)
    parser.add_argument("--target-elder-id", required=True)
    parser.add_argument("--room", action="append", required=True)
    parser.add_argument("--version", action="append", default=[])
    parser.add_argument("--summary-out")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    room_versions = _parse_room_versions(args.room, args.version)
    for room_name in args.room:
        room_versions.setdefault(str(room_name), None)  # allow source current version
    resolved_room_versions = {
        room_name: version for room_name, version in room_versions.items() if version is not None
    }
    summary = promote_room_versions_from_namespace(
        backend_dir=str(args.backend_dir),
        source_elder_id=str(args.source_elder_id),
        target_elder_id=str(args.target_elder_id),
        room_versions=resolved_room_versions or {str(room): 0 for room in args.room},
    )
    payload = json.dumps(summary, indent=2)
    if args.summary_out:
        out_path = Path(args.summary_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
