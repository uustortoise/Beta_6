import json
from pathlib import Path

import yaml

from health_server import _parse_room_override_map as parse_health_map
from ml.beta6.evaluation.evaluation_metrics import parse_room_override_map as parse_eval_map
from ml.policy_config import _parse_override_map as parse_policy_map
from ml.training import TrainingPipeline
from run_daily_analysis import _parse_room_override_map as parse_run_map
from utils.room_utils import normalize_room_name


def _collect_room_mappings(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_txt = str(key)
            path = f"{prefix}.{key_txt}" if prefix else key_txt
            if key_txt.endswith("_by_room") and isinstance(value, dict):
                out.append((path, value))
            out.extend(_collect_room_mappings(value, path))
    return out


def _collect_room_lists(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_txt = str(key)
            path = f"{prefix}.{key_txt}" if prefix else key_txt
            if key_txt.endswith("_rooms") and isinstance(value, list):
                out.append((path, value))
            out.extend(_collect_room_lists(value, path))
    return out


def _alias_collision_details(room_tokens):
    canonical_to_raw = {}
    collisions = {}
    for raw in room_tokens:
        canonical = normalize_room_name(str(raw))
        if not canonical:
            continue
        previous = canonical_to_raw.get(canonical)
        if previous is None:
            canonical_to_raw[canonical] = str(raw)
            continue
        if previous != str(raw):
            collisions.setdefault(canonical, set()).update({previous, str(raw)})
    return {k: sorted(v) for k, v in collisions.items()}


def test_room_override_parsers_are_alias_safe():
    raw = "LivingRoom:1,living_room=2,Living Room:3,kitchen=7"
    expected = {"livingroom": "3", "kitchen": "7"}

    assert parse_run_map(raw) == expected
    assert TrainingPipeline._parse_room_override_map(raw) == expected
    assert parse_policy_map(raw) == expected
    assert parse_health_map(raw) == expected
    assert parse_eval_map(raw, normalize_room_name_fn=normalize_room_name) == expected


def test_room_config_and_policy_defaults_have_no_alias_collisions():
    backend_root = Path(__file__).resolve().parents[1]
    room_cfg_path = backend_root / "config" / "room_config.json"
    policy_defaults_path = backend_root / "config" / "beta6_policy_defaults.yaml"

    room_cfg = json.loads(room_cfg_path.read_text(encoding="utf-8"))
    policy_defaults = yaml.safe_load(policy_defaults_path.read_text(encoding="utf-8")) or {}

    # Room config uses raw keys for display/editability but must still avoid alias collisions.
    room_cfg_rooms = room_cfg.get("rooms", {}) if isinstance(room_cfg, dict) else {}
    room_cfg_collisions = _alias_collision_details(room_cfg_rooms.keys())
    assert room_cfg_collisions == {}, f"room_config alias collisions: {room_cfg_collisions}"

    # Policy defaults maps/lists should not carry duplicate room aliases either.
    mapping_collisions = {}
    for path, mapping in _collect_room_mappings(policy_defaults):
        collisions = _alias_collision_details(mapping.keys())
        if collisions:
            mapping_collisions[path] = collisions

    list_collisions = {}
    for path, room_list in _collect_room_lists(policy_defaults):
        collisions = _alias_collision_details(room_list)
        if collisions:
            list_collisions[path] = collisions

    assert mapping_collisions == {}, f"beta6_policy_defaults *_by_room alias collisions: {mapping_collisions}"
    assert list_collisions == {}, f"beta6_policy_defaults *_rooms alias collisions: {list_collisions}"
