import json
from pathlib import Path

from config import RoomConfigManager


def _write_config(path: Path) -> None:
    payload = {
        "defaults": {
            "sequence_time_window": 600,
            "data_interval": 10,
        },
        "rooms": {
            "living_room": {
                "sequence_time_window": 900,
                "data_interval": 10,
            },
            "bedroom": {
                "sequence_time_window": 1800,
                "data_interval": 10,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_room_config_alias_lookup_supports_livingroom_variants(tmp_path: Path) -> None:
    cfg_path = tmp_path / "room_config.json"
    _write_config(cfg_path)
    manager = RoomConfigManager(config_path=cfg_path)

    assert manager.get_sequence_window("living_room") == 900
    assert manager.get_sequence_window("Living Room") == 900
    assert manager.get_sequence_window("LivingRoom") == 900
    assert manager.calculate_seq_length("LivingRoom") == 90


def test_update_room_config_reuses_existing_alias_key(tmp_path: Path) -> None:
    cfg_path = tmp_path / "room_config.json"
    _write_config(cfg_path)
    manager = RoomConfigManager(config_path=cfg_path)

    manager.update_room_config("LivingRoom", 1200)

    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["rooms"]["living_room"]["sequence_time_window"] == 1200
    assert "livingroom" not in saved["rooms"]
