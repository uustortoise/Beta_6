import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from .base_service import BaseService
from processors.profile_processor import apply_resident_home_context_contract

logger = logging.getLogger(__name__)


def _decode_profile_blob(raw_value: Any) -> Dict[str, Any]:
    if isinstance(raw_value, dict):
        return dict(raw_value)
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _deep_merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged

class ProfileService(BaseService):
    """
    Manages Elder Profiles, Medical History, and Emergency Contacts.
    """

    def create_or_update_elder(self, elder_id: str, profile_data: Dict[str, Any]) -> str:
        """
        Create or update the master elder record.
        """
        with self.db.get_connection() as conn:
            existing_row = conn.execute(
                """
                SELECT full_name, date_of_birth, gender, blood_type, profile_data
                FROM elders
                WHERE elder_id = ?
                """,
                (elder_id,),
            ).fetchone()
            existing_profile = {}
            if existing_row:
                profile_blob = (
                    existing_row["profile_data"]
                    if hasattr(existing_row, "keys")
                    else existing_row[4] if len(existing_row) > 4 else None
                )
                existing_profile = _decode_profile_blob(profile_blob)

            merged_profile = _deep_merge_dicts(existing_profile, dict(profile_data or {}))
            normalized_profile = apply_resident_home_context_contract(merged_profile)
            personal_info = normalized_profile.get("personal_info", {})

            existing_full_name = (
                existing_row["full_name"]
                if existing_row and hasattr(existing_row, "keys")
                else existing_row[0] if existing_row else None
            )
            existing_dob = (
                existing_row["date_of_birth"]
                if existing_row and hasattr(existing_row, "keys")
                else existing_row[1] if existing_row else None
            )
            existing_gender = (
                existing_row["gender"]
                if existing_row and hasattr(existing_row, "keys")
                else existing_row[2] if existing_row else None
            )
            existing_blood_type = (
                existing_row["blood_type"]
                if existing_row and hasattr(existing_row, "keys")
                else existing_row[3] if existing_row else None
            )

            full_name = personal_info.get('full_name') or existing_full_name or f'Elder_{elder_id}'
            dob = personal_info.get('date_of_birth') or existing_dob
            gender = personal_info.get('gender') or existing_gender
            blood_type = personal_info.get('blood_type') or existing_blood_type

            conn.execute('''
                INSERT INTO elders (elder_id, full_name, date_of_birth, gender, blood_type, profile_data, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(elder_id) DO UPDATE SET
                    full_name=excluded.full_name,
                    date_of_birth=excluded.date_of_birth,
                    gender=excluded.gender,
                    blood_type=excluded.blood_type,
                    profile_data=excluded.profile_data,
                    last_updated=CURRENT_TIMESTAMP
            ''', (
                elder_id,
                full_name,
                dob,
                gender,
                blood_type,
                json.dumps(normalized_profile, default=str),
            ))
            
            # Save Medical History Components
            self._save_medical_history(conn, elder_id, normalized_profile.get('medical_history', {}))
            
            # Save Emergency Contacts
            self._save_contacts(conn, elder_id, normalized_profile.get('emergency_contacts', []))
            
            conn.commit()
            
        logger.info(f"Profile saved for {elder_id}")
        return elder_id

    def get_profile(self, elder_id: str) -> Dict[str, Any]:
        """
        Reconstruct the full 'Enhanced Profile' structure from the database
        to maintain compatibility with the UI.
        """
        with self.db.get_connection() as conn:
            # 1. Elder Info
            # Explicit column selection for tuple access
            cur = conn.execute(
                "SELECT full_name, date_of_birth, gender, blood_type, profile_data FROM elders WHERE elder_id = ?",
                (elder_id,),
            )
            elder = cur.fetchone()
            if not elder:
                return {} # Or raise

            # 2. Medical History
            # Columns: category, data
            history_rows = conn.execute("SELECT category, data FROM medical_history WHERE elder_id = ?", (elder_id,)).fetchall()
            
            # 3. Contacts
            # Columns: name, relationship, phone, email, is_primary
            contact_rows = conn.execute("SELECT name, relationship, phone, email, is_primary FROM emergency_contacts WHERE elder_id = ?", (elder_id,)).fetchall()

        # Reconstruct logical structure
        # Tuple indices: 0=full_name, 1=dob, 2=gender, 3=blood_type
        stored_profile = _decode_profile_blob(
            elder["profile_data"] if hasattr(elder, "keys") else elder[4] if len(elder) > 4 else None
        )
        profile = _deep_merge_dicts(stored_profile, {
            "personal_info": {
                "full_name": elder["full_name"] if hasattr(elder, "keys") else elder[0],
                "date_of_birth": elder["date_of_birth"] if hasattr(elder, "keys") else elder[1],
                "gender": elder["gender"] if hasattr(elder, "keys") else elder[2],
                "blood_type": elder["blood_type"] if hasattr(elder, "keys") else elder[3]
            },
            "medical_history": {
                "chronic_conditions": [],
                "medications": [],
                "allergies": [],
                "surgeries": [],
                "vaccinations": []
            },
            "emergency_contacts": []
            })
        profile = apply_resident_home_context_contract(profile)
        
        # Legacy flat keys kept for backward compatibility with older callers/tests.
        profile["full_name"] = profile["personal_info"]["full_name"]
        profile["date_of_birth"] = profile["personal_info"]["date_of_birth"]
        profile["gender"] = profile["personal_info"]["gender"]
        profile["blood_type"] = profile["personal_info"]["blood_type"]

        for row in history_rows:
            # Tuple: 0=category, 1=data
            category = row[0]
            try:
                data = json.loads(row[1]) if isinstance(row[1], str) else row[1]
            except:
                data = [] # Fallback
            
            if category == 'chronic':
                profile['medical_history']['chronic_conditions'] = data
            elif category == 'medication':
                profile['medical_history']['medications'] = data
            elif category == 'allergy':
                profile['medical_history']['allergies'] = data
            # etc...

        for row in contact_rows:
            # Tuple: 0=name, 1=rel, 2=phone, 3=email, 4=is_primary
            contact = {
                "name": row[0],
                "relationship": row[1],
                "phone": row[2],
                "email": row[3],
                "is_primary": bool(row[4])
            }
            profile['emergency_contacts'].append(contact)

        return profile

    def _save_medical_history(self, conn, elder_id: str, medical_history: Dict[str, Any]):
        # We replace specific categories or append? 
        # For this refactor, let's treat the 'save' as a snapshot replace for simplicity 
        # until we have a proper granular UI.
        
        categories = {
            'chronic': medical_history.get('chronic_conditions', []),
            'medication': medical_history.get('medications', []),
            'allergy': medical_history.get('allergies', []),
            # Add others as needed
        }

        for cat, data in categories.items():
            if not data:
                continue
            
            # We are storing the entire list as one JSON block for that category
            # This is a hybrid approach. 
            # Ideally: one row per medication. 
            # But the requirement is "Flexible JSON for complex data". 
            # So storing the list of meds as one JSON blob per category is valid for now.
            
            json_data = json.dumps(data, default=str)
            
            # Check if exists
            conn.execute('''
                INSERT OR REPLACE INTO medical_history (elder_id, category, data, recorded_date)
                VALUES (?, ?, ?, CURRENT_DATE)
            ''', (elder_id, cat, json_data))

    def _save_contacts(self, conn, elder_id: str, contacts: List[Dict]):
        # Replace strategy: Delete all and re-insert
        conn.execute("DELETE FROM emergency_contacts WHERE elder_id = ?", (elder_id,))
        
        for c in contacts:
            conn.execute('''
                INSERT INTO emergency_contacts (elder_id, name, relationship, phone, email, is_primary)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (elder_id, c.get('name'), c.get('relationship'), c.get('phone'), c.get('email'), c.get('is_primary', False)))

    def delete_elder(self, elder_id: str) -> bool:
        """
        Completely remove an elder and ALL associated data from the system.
        """
        try:
            with self.db.get_connection() as conn:
                # 1. Related Tables (Foreign Keys)
                tables = [
                    'medical_history', 
                    'emergency_contacts',
                    'icope_assessments',
                    'sleep_analysis',
                    'adl_history',
                    'activity_segments',
                    'household_behavior',
                    'household_segments',
                    'training_history',
                    'alerts'
                ]
                
                for table in tables:
                    conn.execute(f"DELETE FROM {table} WHERE elder_id = ?", (elder_id,))
                    
                # 2. Master Record
                conn.execute("DELETE FROM elders WHERE elder_id = ?", (elder_id,))
                
                conn.commit()
            
            logger.info(f"Successfully deleted elder: {elder_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete elder {elder_id}: {e}", exc_info=True)
            return False
