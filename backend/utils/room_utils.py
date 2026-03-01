"""
Room name normalization utility.
Provides a single source of truth for room name matching across the codebase.
"""

import re


def normalize_room_name(room: str) -> str:
    """
    Normalize room names for consistent matching.
    
    Removes spaces, underscores, and converts to lowercase.
    Examples:
        'Living Room' -> 'livingroom'
        'living_room' -> 'livingroom'
        'LivingRoom'  -> 'livingroom'
        'Bathroom'    -> 'bathroom'
    
    Args:
        room: Raw room name string
        
    Returns:
        Normalized lowercase room name without spaces or underscores
    """
    if not room:
        return ''
    return re.sub(r'[\s_]', '', room).lower()


def rooms_match(room1: str, room2: str) -> bool:
    """
    Check if two room names refer to the same room.
    
    Args:
        room1: First room name
        room2: Second room name
        
    Returns:
        True if normalized names match
    """
    return normalize_room_name(room1) == normalize_room_name(room2)


def normalize_timestamp(ts, interval_seconds: int = 10) -> str:
    """
    Normalize timestamp to consistent interval boundaries.
    
    Delegates to unified time_utils.
    """
    from .time_utils import normalize_with_interval
    return normalize_with_interval(ts, interval_seconds)
