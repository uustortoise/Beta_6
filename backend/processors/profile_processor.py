import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Mapping, Optional
from .templates import DEFAULT_PROFILE_TEMPLATE
import copy

logger = logging.getLogger(__name__)

_HOUSEHOLD_TYPE_MAP = {
    "single": "single_resident",
    "single_resident": "single_resident",
    "solo": "single_resident",
    "double": "multi_resident",
    "multi": "multi_resident",
    "multiple": "multi_resident",
    "multi_resident": "multi_resident",
}
_HELPER_PRESENCE_MAP = {
    "none": "none",
    "no": "none",
    "false": "none",
    "scheduled": "scheduled",
    "part_time": "scheduled",
    "part-time": "scheduled",
    "visiting": "scheduled",
    "live_in": "live_in",
    "live-in": "live_in",
    "resident": "live_in",
    "full_time": "live_in",
    "full-time": "live_in",
    "unknown": "unknown",
}
_LAYOUT_TOPOLOGY_MAP = {
    "open": "open_plan",
    "open_plan": "open_plan",
    "open-plan": "open_plan",
    "corridor": "corridor",
    "hallway": "corridor",
    "segmented": "segmented",
    "clustered": "clustered",
    "multi_zone": "multi_zone",
    "multi-zone": "multi_zone",
    "unknown": "unknown",
}
_REQUIRED_CONTEXT_FIELDS = (
    "household_type",
    "helper_presence",
    "layout.topology",
)


def _token(value: Any) -> str:
    return str(value or "").strip().lower()


def _coerce_choice(value: Any, mapping: Mapping[str, str]) -> Optional[str]:
    token = _token(value)
    if not token:
        return None
    return mapping.get(token)


def _normalize_room_key(value: Any) -> str:
    room = _token(value)
    return room.replace(" ", "_")


def _normalize_layout_adjacency(raw_adjacency: Any) -> Dict[str, List[str]]:
    if not isinstance(raw_adjacency, Mapping):
        return {}

    normalized: Dict[str, List[str]] = {}
    for raw_room, raw_neighbors in raw_adjacency.items():
        room = _normalize_room_key(raw_room)
        if not room:
            continue
        if isinstance(raw_neighbors, str):
            candidates = [raw_neighbors]
        elif isinstance(raw_neighbors, list):
            candidates = raw_neighbors
        else:
            continue
        seen: set[str] = set()
        neighbors: List[str] = []
        for candidate in candidates:
            neighbor = _normalize_room_key(candidate)
            if not neighbor or neighbor == room or neighbor in seen:
                continue
            seen.add(neighbor)
            neighbors.append(neighbor)
        normalized[room] = sorted(neighbors)

    # Make the graph symmetric so transition/arbitration code gets a stable contract.
    for room, neighbors in list(normalized.items()):
        for neighbor in neighbors:
            normalized.setdefault(neighbor, [])
            if room not in normalized[neighbor]:
                normalized[neighbor].append(room)
                normalized[neighbor] = sorted(set(normalized[neighbor]))
    return normalized


def normalize_resident_home_context(profile_data: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    payload = dict(profile_data or {})
    raw_context = payload.get("resident_home_context")
    if not isinstance(raw_context, Mapping):
        raw_context = payload.get("home_context")
    if not isinstance(raw_context, Mapping):
        raw_context = {}

    household_type = _coerce_choice(
        raw_context.get("household_type", payload.get("household_type")),
        _HOUSEHOLD_TYPE_MAP,
    )
    helper_presence = _coerce_choice(
        raw_context.get("helper_presence", payload.get("helper_presence")),
        _HELPER_PRESENCE_MAP,
    )
    raw_layout = raw_context.get("layout")
    if not isinstance(raw_layout, Mapping):
        raw_layout = payload.get("layout") if isinstance(payload.get("layout"), Mapping) else {}
    topology = _coerce_choice(
        raw_layout.get("topology", raw_context.get("layout_topology")),
        _LAYOUT_TOPOLOGY_MAP,
    )
    adjacency = _normalize_layout_adjacency(
        raw_layout.get("adjacency", raw_context.get("layout_adjacency"))
    )

    missing_fields: List[str] = []
    if household_type is None:
        missing_fields.append("household_type")
    if helper_presence is None:
        missing_fields.append("helper_presence")
    if topology is None:
        missing_fields.append("layout.topology")

    status = "ready" if not missing_fields else "missing_required_context"
    context = {
        "contract_version": "beta61",
        "status": status,
        "missing_fields": missing_fields,
        "household_type": household_type,
        "helper_presence": helper_presence,
        "layout": {
            "topology": topology,
            "adjacency": adjacency,
        },
        "cohort_key": ":".join(
            [
                household_type or "unknown_household",
                helper_presence or "unknown_helper",
                topology or "unknown_topology",
            ]
        ),
    }
    return context


def apply_resident_home_context_contract(profile_data: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    payload = copy.deepcopy(dict(profile_data or {}))
    payload["resident_home_context"] = normalize_resident_home_context(payload)
    return payload

class ProfileProcessor:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.processed_dir = data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def get_elder_dir(self, elder_id: str) -> Path:
        """Get directory path for a specific elder"""
        elder_dir = self.processed_dir / elder_id
        elder_dir.mkdir(parents=True, exist_ok=True)
        return elder_dir

    def load_profile(self, elder_id: str) -> Optional[Dict[str, Any]]:
        """Load resident profile from JSON"""
        profile_path = self.get_elder_dir(elder_id) / "profile.json"
        if not profile_path.exists():
            return None
        
        try:
            with open(profile_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load profile for {elder_id}: {e}")
            return None

    def save_profile(self, elder_id: str, profile_data: Dict[str, Any]) -> bool:
        """Save resident profile to JSON"""
        profile_path = self.get_elder_dir(elder_id) / "profile.json"
        normalized_profile = apply_resident_home_context_contract(profile_data)
        
        # Ensure timestamp
        normalized_profile['last_updated'] = datetime.now().isoformat()
        if 'id' not in normalized_profile:
            normalized_profile['id'] = elder_id
            
        try:
            with open(profile_path, 'w') as f:
                json.dump(normalized_profile, f, indent=4)
            logger.info(f"Saved profile for {elder_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save profile for {elder_id}: {e}")
            return False

    def list_residents(self) -> List[Dict[str, Any]]:
        """List all residents with basic info"""
        residents = []
        if not self.processed_dir.exists():
            return []
            
        for elder_dir in self.processed_dir.iterdir():
            if elder_dir.is_dir():
                elder_id = elder_dir.name
                profile = self.load_profile(elder_id)
                
                if profile:
                    residents.append({
                        'id': elder_id,
                        'name': profile.get('name', 'Unknown'),
                        'age': profile.get('age', 'N/A'),
                        'risk_level': profile.get('risk_level', 'low'),
                        'last_updated': profile.get('last_updated')
                    })
                else:
                    # Synthetic profile if file missing but dir exists
                    residents.append({
                        'id': elder_id,
                        'name': f"Resident {elder_id}",
                        'age': 'N/A',
                        'risk_level': 'unknown',
                        'last_updated': None
                    })
        return residents



    def create_default_profile(self, elder_id: str):
        """Create a default profile if one doesn't exist"""
        if self.load_profile(elder_id):
            return
            
        # deepcopy to avoid mutating the template
        profile = copy.deepcopy(DEFAULT_PROFILE_TEMPLATE)
        
        # Populate basic info
        profile["id"] = elder_id
        profile["personal_info"]["full_name"] = f"Resident {elder_id}"
        profile["personal_info"]["age"] = 75
        profile["personal_info"]["gender"] = "Unknown"
        profile["system_metadata"]["created_at"] = datetime.now().isoformat()
        
        # Keep top-level analysis fields that the dashboard might rely on loosely for now
        # (Though we should migrate frontend to read from proper paths eventually)
        profile["risk_level"] = "medium" 
        
        self.save_profile(elder_id, profile)
