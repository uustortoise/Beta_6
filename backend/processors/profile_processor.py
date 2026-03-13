import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from .templates import DEFAULT_PROFILE_TEMPLATE
import copy

logger = logging.getLogger(__name__)

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
        
        # Ensure timestamp
        self._normalize_resident_home_context(profile_data)
        profile_data['last_updated'] = datetime.now().isoformat()
        if 'id' not in profile_data:
            profile_data['id'] = elder_id
            
        try:
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=4)
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

    @staticmethod
    def _normalize_resident_home_context(profile_data: Dict[str, Any]) -> Dict[str, Any]:
        raw = profile_data.get("resident_home_context")
        source = raw if isinstance(raw, dict) else {}

        household_token = str(source.get("household_type", "single") or "").strip().lower()
        household_type = "multi" if household_token in {"multi", "multiple", "double", "couple"} else "single"

        helper_value = source.get("helper_presence", "unknown")
        if isinstance(helper_value, bool):
            helper_presence = "present" if helper_value else "none"
        else:
            helper_token = str(helper_value or "").strip().lower()
            if helper_token in {"present", "yes", "true", "1", "part_time", "full_time", "live_in"}:
                helper_presence = "present"
            elif helper_token in {"none", "no", "false", "0", "absent"}:
                helper_presence = "none"
            else:
                helper_presence = "unknown"

        raw_layout = source.get("layout_topology")
        normalized_layout: Dict[str, List[str]] = {}
        if isinstance(raw_layout, dict):
            for room, neighbors in raw_layout.items():
                room_key = str(room or "").strip().lower().replace(" ", "_")
                if not room_key:
                    continue
                if isinstance(neighbors, str):
                    values = [neighbors]
                elif isinstance(neighbors, list):
                    values = neighbors
                else:
                    values = []
                items: List[str] = []
                for value in values:
                    key = str(value or "").strip().lower().replace(" ", "_")
                    if key:
                        items.append(key)
                normalized_layout[room_key] = sorted(set(items))

        profile_data["resident_home_context"] = {
            "household_type": household_type,
            "helper_presence": helper_presence,
            "layout_topology": normalized_layout,
        }
        return profile_data["resident_home_context"]

    def get_resident_home_context(self, elder_id: str) -> Dict[str, Any]:
        profile = self.load_profile(elder_id)
        if not isinstance(profile, dict):
            return {
                "status": "not_available",
                "household_type": "single",
                "helper_presence": "unknown",
                "layout_topology": {},
                "missing_required_fields": ["helper_presence", "layout_topology"],
                "message": "Profile missing; resident/home context contract unavailable.",
            }

        context = self._normalize_resident_home_context(profile)
        missing: List[str] = []
        if str(context.get("helper_presence", "unknown")).strip().lower() == "unknown":
            missing.append("helper_presence")
        if not isinstance(context.get("layout_topology"), dict) or not context.get("layout_topology"):
            missing.append("layout_topology")

        return {
            "status": "ready" if not missing else "incomplete",
            "household_type": context.get("household_type", "single"),
            "helper_presence": context.get("helper_presence", "unknown"),
            "layout_topology": context.get("layout_topology", {}),
            "missing_required_fields": missing,
            "message": (
                "Resident/home context contract is complete."
                if not missing
                else f"Missing required context fields: {', '.join(missing)}"
            ),
        }



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
        profile["resident_home_context"] = {
            "household_type": "single",
            "helper_presence": "unknown",
            "layout_topology": {},
        }
        
        # Keep top-level analysis fields that the dashboard might rely on loosely for now
        # (Though we should migrate frontend to read from proper paths eventually)
        profile["risk_level"] = "medium" 
        
        self.save_profile(elder_id, profile)
