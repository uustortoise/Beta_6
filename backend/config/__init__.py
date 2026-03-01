"""
Configuration module for Beta 5.5.
Provides centralized config loading for paths, room settings, etc.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

# --- Base Paths (Calculated relative to this file for portability) ---
# Path: __init__.py -> config/ -> backend/ -> Beta_5.5/
_config_file = Path(__file__).resolve()
_config_dir = _config_file.parent        # config/
BACKEND_DIR = _config_dir.parent          # backend/
PROJECT_ROOT = BACKEND_DIR.parent         # Beta_5.5/
DATA_DIR = PROJECT_ROOT / "data"

# --- Database Path ---
# Environment variable can override, but defaults to relative path
_db_env = os.getenv("DB_PATH")
if _db_env:
    DB_PATH = Path(_db_env) if Path(_db_env).is_absolute() else (PROJECT_ROOT / _db_env).resolve()
else:
    DB_PATH = (DATA_DIR / "processed" / "residents_master_data.db").resolve()

MODEL_DIR = BACKEND_DIR / "models"
LOG_DIR = BACKEND_DIR / "logs"

# Config file path (always relative to this file)
ROOM_CONFIG_PATH = _config_dir / "room_config.json"
RELEASE_GATES_CONFIG_PATH = _config_dir / "release_gates.json"

# Logger for config module
_logger = logging.getLogger(__name__)


class RoomConfigManager:
    """
    Manages per-room configuration including sequence length settings.
    Uses Pydantic for validation (Beta 5 Mandate: Solid Foundation).
    """
    
    def __init__(self, config_path: Path = None):
        """Initialize with specific config path (defaults to prod config)."""
        self.config_path = config_path or ROOM_CONFIG_PATH
        self._config: Dict = None
        self._validated_config = None
        self._load_config()
    
    def _load_config(self):
        """Load and validate config from JSON file using Pydantic."""
        from .schemas import SystemConfig
        from pydantic import ValidationError
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                raw_data = json.load(f)
            
            # Validate using Pydantic schema
            try:
                self._validated_config = SystemConfig(**raw_data)
                self._config = raw_data
                _logger.info(f"✓ Configuration loaded from {self.config_path}")
            except ValidationError as e:
                _logger.error(f"Configuration validation failed: {e}")
                raise RuntimeError(
                    f"Invalid room_config.json. Fix errors before starting:\n{e}"
                )
        else:
            # Use defaults if no config file exists
            self._config = {
                "defaults": {
                    "sequence_time_window": 600,
                    "data_interval": 10
                },
                "rooms": {}
            }
            self._validated_config = SystemConfig(**self._config)
            _logger.warning(f"Config file not found at {self.config_path}, using defaults")
    
    def get_sequence_window(self, room_name: str) -> int:
        """
        Get the sequence time window for a specific room.
        Returns in seconds.
        """
        room_key = room_name.lower().replace(" ", "_")
        
        if room_key in self._config.get("rooms", {}):
            return self._config["rooms"][room_key].get(
                "sequence_time_window",
                self._config["defaults"]["sequence_time_window"]
            )
        return self._config["defaults"]["sequence_time_window"]
    
    def get_data_interval(self, room_name: str = None) -> int:
        """Get data interval (typically 10 seconds)."""
        if room_name:
            room_key = room_name.lower().replace(" ", "_")
            if room_key in self._config.get("rooms", {}):
                return self._config["rooms"][room_key].get(
                    "data_interval",
                    self._config["defaults"]["data_interval"]
                )
        return self._config["defaults"]["data_interval"]
    
    def calculate_seq_length(self, room_name: str) -> int:
        """
        Calculate sequence length (number of samples) for a room.
        seq_length = window / interval
        """
        window = self.get_sequence_window(room_name)
        interval = self.get_data_interval(room_name)
        return window // interval
    
    def get_all_rooms(self) -> Dict:
        """Get all room configurations."""
        return self._config.get("rooms", {})
    
    def save_config(self):
        """Save current config to file."""
        if self.config_path.exists():
             with open(self.config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
    
    def update_room_config(self, room_name: str, sequence_time_window: int):
        """Update config for a specific room."""
        room_key = room_name.lower().replace(" ", "_")
        if "rooms" not in self._config:
            self._config["rooms"] = {}
        if room_key not in self._config["rooms"]:
            self._config["rooms"][room_key] = {}
        self._config["rooms"][room_key]["sequence_time_window"] = sequence_time_window
        self.save_config()


# Global instance for app-wide use
_global_config_instance = None
_release_gates_config_cache: Optional[Dict] = None

def get_room_config(force_reload: bool = False) -> RoomConfigManager:
    """
    Factory function for RoomConfigManager.
    
    Args:
        force_reload: If True, returns a FRESH instance (useful for tests).
                      If False, returns the shared global instance (prod mode).
    """
    global _global_config_instance
    
    if force_reload:
        return RoomConfigManager()
        
    if _global_config_instance is None:
        _global_config_instance = RoomConfigManager()
        
    return _global_config_instance


def get_release_gates_config(force_reload: bool = False) -> Dict:
    """
    Load machine-readable ML release gate policy.

    Returns:
        Dict loaded from backend/config/release_gates.json.
    """
    global _release_gates_config_cache

    if force_reload or _release_gates_config_cache is None:
        if not RELEASE_GATES_CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Release gates config not found: {RELEASE_GATES_CONFIG_PATH}"
            )

        with open(RELEASE_GATES_CONFIG_PATH, "r") as f:
            _release_gates_config_cache = json.load(f)

        _logger.info(f"✓ Release gates loaded from {RELEASE_GATES_CONFIG_PATH}")

    return _release_gates_config_cache
