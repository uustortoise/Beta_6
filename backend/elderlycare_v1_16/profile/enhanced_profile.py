"""
Enhanced Profile Management for Elderly Care Platform.

This module provides comprehensive profile management with medical history,
health metrics, and care preferences for multi-elderly monitoring.
"""

from datetime import datetime, date
import json
from typing import Dict, List, Optional, Any, Union
import copy
from .templates import DEFAULT_PROFILE_TEMPLATE, COMMON_MEDICAL_CONDITIONS


class EnhancedProfile:
    """Enhanced profile management for elderly individuals."""
    
    def __init__(self, elder_id: str, profile_data: Optional[Dict] = None):
        """
        Initialize an enhanced profile for an elder.
        
        Args:
            elder_id: Unique identifier for the elder
            profile_data: Optional existing profile data
        """
        self.elder_id = elder_id
        self.profile = self._initialize_profile(profile_data)
        self._calculate_age()
        self._update_timestamps()
    
    def _initialize_profile(self, profile_data: Optional[Dict]) -> Dict:
        """Initialize profile with default template or existing data."""
        if profile_data is None:
            return self._create_empty_profile()
        
        # Merge existing data with template to ensure all fields exist
        merged_profile = copy.deepcopy(DEFAULT_PROFILE_TEMPLATE)
        self._deep_update(merged_profile, profile_data)
        
        # Ensure version is set
        merged_profile['system_metadata']['version'] = "1.11.0"
        return merged_profile
    
    def _create_empty_profile(self) -> Dict:
        """Create an empty profile with defaults."""
        profile = copy.deepcopy(DEFAULT_PROFILE_TEMPLATE)
        profile['system_metadata']['created_at'] = datetime.now().isoformat()
        profile['system_metadata']['last_updated'] = datetime.now().isoformat()
        profile['system_metadata']['version'] = "1.11.0"
        return profile
    
    def _deep_update(self, target: Dict, source: Dict) -> None:
        """Recursively update target dictionary with source values."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _calculate_age(self) -> None:
        """Calculate age from date of birth."""
        dob_str = self.profile['personal_info']['date_of_birth']
        if dob_str:
            try:
                dob = datetime.fromisoformat(dob_str).date()
                today = date.today()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                self.profile['personal_info']['age'] = age
            except (ValueError, AttributeError):
                self.profile['personal_info']['age'] = None
    
    def _update_timestamps(self) -> None:
        """Update last_updated timestamp."""
        self.profile['system_metadata']['last_updated'] = datetime.now().isoformat()
    
    def update_personal_info(self, **kwargs) -> None:
        """Update personal information."""
        for key, value in kwargs.items():
            if key in self.profile['personal_info']:
                self.profile['personal_info'][key] = value
        
        if 'date_of_birth' in kwargs:
            self._calculate_age()
        
        self._update_timestamps()
    
    def add_chronic_condition(self, condition: str, notes: str = "") -> None:
        """Add a chronic condition to medical history."""
        if condition not in self.profile['medical_history']['chronic_conditions']:
            self.profile['medical_history']['chronic_conditions'].append(condition)
        
        self._update_timestamps()
    
    def add_medication(self, name: str, dosage: str, frequency: str, 
                      purpose: str = "", prescribing_doctor: str = "",
                      last_refill: str = "", notes: str = "") -> None:
        """Add a medication record."""
        medication = {
            "name": name,
            "dosage": dosage,
            "frequency": frequency,
            "purpose": purpose,
            "prescribing_doctor": prescribing_doctor,
            "last_refill": last_refill,
            "notes": notes
        }
        self.profile['medical_history']['medications'].append(medication)
        self._update_timestamps()
    
    def add_allergy(self, allergen: str, reaction: str, 
                   severity: str = "mild", notes: str = "") -> None:
        """Add an allergy record."""
        allergy = {
            "allergen": allergen,
            "reaction": reaction,
            "severity": severity,  # mild/moderate/severe
            "notes": notes
        }
        self.profile['medical_history']['allergies'].append(allergy)
        self._update_timestamps()
    
    def update_health_metrics(self, **kwargs) -> None:
        """Update health metrics."""
        for key, value in kwargs.items():
            # Handle nested metrics
            if key in self.profile['health_metrics']:
                self.profile['health_metrics'][key] = value
            elif 'baseline_vitals' in key:
                # Handle blood pressure separately
                if key == 'blood_pressure':
                    if isinstance(value, dict):
                        self.profile['health_metrics']['baseline_vitals']['blood_pressure'] = value
                    elif isinstance(value, tuple):
                        systolic, diastolic = value
                        self.profile['health_metrics']['baseline_vitals']['blood_pressure'] = {
                            'systolic': systolic,
                            'diastolic': diastolic
                        }
                else:
                    # Handle other vitals
                    subkey = key.replace('baseline_vitals_', '')
                    if subkey in self.profile['health_metrics']['baseline_vitals']:
                        self.profile['health_metrics']['baseline_vitals'][subkey] = value
        
        self._update_timestamps()
    
    def add_emergency_contact(self, name: str, relationship: str, phone: str,
                             email: str = "", address: str = "", 
                             is_primary: bool = False, notes: str = "") -> None:
        """Add an emergency contact."""
        contact = {
            "name": name,
            "relationship": relationship,
            "phone": phone,
            "email": email,
            "address": address,
            "is_primary": is_primary,
            "notes": notes
        }
        self.profile['emergency_contacts'].append(contact)
        self._update_timestamps()
    
    def update_care_preferences(self, category: str, preferences: Dict) -> None:
        """Update care preferences."""
        if category in self.profile['care_preferences']:
            self.profile['care_preferences'][category] = preferences
            self._update_timestamps()
    
    def to_dict(self) -> Dict:
        """Return profile as dictionary."""
        return copy.deepcopy(self.profile)
    
    def to_json(self, indent: int = 2) -> str:
        """Return profile as JSON string."""
        return json.dumps(self.profile, indent=indent, default=str)
    
    def save_to_file(self, filepath: str) -> None:
        """Save profile to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.profile, f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, elder_id: str, filepath: str) -> 'EnhancedProfile':
        """Load profile from JSON file."""
        with open(filepath, 'r') as f:
            profile_data = json.load(f)
        return cls(elder_id, profile_data)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the profile."""
        return {
            "elder_id": self.elder_id,
            "name": self.profile['personal_info']['full_name'],
            "age": self.profile['personal_info']['age'],
            "gender": self.profile['personal_info']['gender'],
            "chronic_conditions_count": len(self.profile['medical_history']['chronic_conditions']),
            "medications_count": len(self.profile['medical_history']['medications']),
            "allergies_count": len(self.profile['medical_history']['allergies']),
            "emergency_contacts_count": len(self.profile['emergency_contacts']),
            "last_updated": self.profile['system_metadata']['last_updated'],
            "data_privacy_level": self.profile['system_metadata']['data_privacy_level']
        }
    
    def get_medical_alerts(self) -> List[str]:
        """Get medical alerts based on profile data."""
        alerts = []
        
        # Allergy alerts
        for allergy in self.profile['medical_history']['allergies']:
            if allergy['severity'] in ['moderate', 'severe']:
                alerts.append(f"Severe allergy to {allergy['allergen']}")
        
        # Medication alerts for serious conditions
        serious_conditions = ['Diabetes Type 1', 'Diabetes Type 2', 'Heart Failure', 'COPD']
        for condition in self.profile['medical_history']['chronic_conditions']:
            if condition in serious_conditions:
                alerts.append(f"Serious condition: {condition}")
        
        # Mobility alerts
        if self.profile['health_metrics']['mobility_status'] in ['wheelchair', 'bedridden']:
            alerts.append(f"Mobility: {self.profile['health_metrics']['mobility_status']}")
        
        return alerts
    
    def validate(self) -> Dict[str, List[str]]:
        """Validate profile data and return validation errors."""
        errors = {"warnings": [], "errors": []}
        
        # Required fields
        if not self.profile['personal_info']['full_name']:
            errors['errors'].append("Full name is required")
        
        if not self.profile['personal_info']['date_of_birth']:
            errors['warnings'].append("Date of birth is recommended")
        
        # Validate date formats
        dob = self.profile['personal_info']['date_of_birth']
        if dob:
            try:
                datetime.fromisoformat(dob)
            except ValueError:
                errors['errors'].append("Date of birth must be in ISO format (YYYY-MM-DD)")
        
        # Validate emergency contacts
        if len(self.profile['emergency_contacts']) == 0:
            errors['warnings'].append("No emergency contacts provided")
        else:
            primary_exists = any(contact['is_primary'] for contact in self.profile['emergency_contacts'])
            if not primary_exists:
                errors['warnings'].append("No primary emergency contact designated")
        
        return errors


def create_empty_profile(elder_id: str, full_name: str = "") -> EnhancedProfile:
    """Create an empty profile with basic information."""
    profile = EnhancedProfile(elder_id)
    if full_name:
        profile.update_personal_info(full_name=full_name)
    return profile


def migrate_legacy_profile(elder_id: str, legacy_data: Dict) -> EnhancedProfile:
    """Migrate legacy profile data to enhanced format."""
    profile = EnhancedProfile(elder_id)
    
    # Map legacy fields to new structure
    if 'name' in legacy_data:
        profile.update_personal_info(full_name=legacy_data['name'])
    
    if 'age' in legacy_data:
        profile.update_personal_info(age=legacy_data['age'])
    
    if 'gender' in legacy_data:
        profile.update_personal_info(gender=legacy_data['gender'])
    
    # Handle notes field
    if 'notes' in legacy_data and legacy_data['notes']:
        # Try to extract medical information from notes
        profile.update_personal_info(notes=legacy_data['notes'])
    
    return profile
