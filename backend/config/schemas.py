"""
Pydantic schemas for configuration validation.
Ensures all configuration values are valid at startup, preventing silent failures.

Beta 5 Mandate: Solid Foundation for 1000 POC
"""
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional


class RoomConfig(BaseModel):
    """Configuration for a single room."""
    sequence_time_window: int = Field(
        ..., 
        gt=0, 
        description="Sequence window in seconds. Must be positive."
    )
    data_interval: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional room-specific data interval override."
    )
    note: Optional[str] = Field(
        default=None,
        description="Human-readable note for documentation."
    )

    @field_validator('sequence_time_window')
    @classmethod
    def must_be_multiple_of_ten(cls, v: int) -> int:
        """Ensure window aligns with 10s sensor granularity."""
        if v % 10 != 0:
            raise ValueError(
                f"sequence_time_window ({v}s) must be a multiple of 10 seconds (sensor interval)"
            )
        return v


class ConfigDefaults(BaseModel):
    """Default configuration values applied when room-specific settings are missing."""
    sequence_time_window: int = Field(
        default=600, 
        gt=0,
        description="Default: 10 minutes (600 seconds)"
    )
    data_interval: int = Field(
        default=10, 
        gt=0, 
        le=60,
        description="Sensor data interval in seconds (1-60)"
    )
    note: Optional[str] = None

    @field_validator('sequence_time_window')
    @classmethod
    def window_must_be_multiple_of_interval(cls, v: int) -> int:
        """Ensure window is divisible by standard interval (10s)."""
        if v % 10 != 0:
            raise ValueError(
                f"sequence_time_window ({v}s) must be a multiple of data_interval"
            )
        return v


class SystemConfig(BaseModel):
    """
    Top-level configuration schema for room_config.json.
    
    Example structure:
    {
        "defaults": {"sequence_time_window": 600, "data_interval": 10},
        "rooms": {
            "bedroom": {"sequence_time_window": 1800},
            "bathroom": {"sequence_time_window": 300}
        }
    }
    """
    defaults: ConfigDefaults
    rooms: Dict[str, RoomConfig] = Field(default_factory=dict)

    @field_validator('rooms')
    @classmethod
    def validate_room_keys(cls, v: Dict[str, RoomConfig]) -> Dict[str, RoomConfig]:
        """Ensure room names are lowercase for consistent lookups."""
        for room_name in v.keys():
            if not room_name.islower() or ' ' in room_name:
                raise ValueError(
                    f"Room name '{room_name}' must be lowercase with underscores (e.g., 'living_room')"
                )
        return v

    def get_room_window(self, room_name: str) -> int:
        """Get sequence window for a room, falling back to defaults."""
        room_key = room_name.lower().replace(" ", "_")
        if room_key in self.rooms:
            return self.rooms[room_key].sequence_time_window
        return self.defaults.sequence_time_window

    def get_data_interval(self, room_name: str = None) -> int:
        """Get data interval, with optional room-specific override."""
        if room_name:
            room_key = room_name.lower().replace(" ", "_")
            if room_key in self.rooms and self.rooms[room_key].data_interval:
                return self.rooms[room_key].data_interval
        return self.defaults.data_interval
