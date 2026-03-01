from typing import Any, Dict, List, Optional


def resolve_scheduled_threshold(
    schedule: List[Dict[str, Any]],
    training_days: float,
) -> Optional[float]:
    """
    Resolve a day-based threshold schedule.

    Behavior:
    - Empty schedule => None
    - training_days below the first bracket => first bracket min_value
    - gap between brackets => last reached bracket min_value
    """
    if not schedule:
        return None

    ordered = sorted(schedule, key=lambda item: float(item.get("min_days", 0)))
    first_min = float(ordered[0].get("min_days", 0))
    if training_days < first_min:
        return float(ordered[0]["min_value"])

    fallback = None
    for item in ordered:
        min_days = float(item.get("min_days", 0))
        max_days = item.get("max_days")
        max_days = float(max_days) if max_days is not None else None

        if training_days >= min_days:
            fallback = float(item["min_value"])
        if training_days >= min_days and (max_days is None or training_days <= max_days):
            return float(item["min_value"])

    return fallback

