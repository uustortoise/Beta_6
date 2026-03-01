"""
Profile and medical data validation for enhanced profile management.

This module provides comprehensive validation for:
- Profile data integrity
- Medical data consistency
- Data format compliance
- Business rule enforcement
"""

from datetime import datetime
import re
from typing import Dict, List, Any, Optional, Tuple
from .templates import (
    MOBILITY_STATUS_OPTIONS,
    COGNITIVE_STATUS_OPTIONS,
    BLOOD_TYPE_OPTIONS,
    COMMON_MEDICAL_CONDITIONS,
    COMMON_ALLERGIES,
    DIETARY_RESTRICTIONS,
    RELATIONSHIP_OPTIONS
)


class ProfileValidator:
    """Validator for enhanced profile data."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_profile(self, profile_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate complete profile data.
        
        Args:
            profile_data: Complete profile dictionary
            
        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        self.errors = []
        self.warnings = []
        
        # Validate each section
        self._validate_personal_info(profile_data.get('personal_info', {}))
        self._validate_medical_history(profile_data.get('medical_history', {}))
        self._validate_health_metrics(profile_data.get('health_metrics', {}))
        self._validate_emergency_contacts(profile_data.get('emergency_contacts', []))
        self._validate_care_preferences(profile_data.get('care_preferences', {}))
        self._validate_system_metadata(profile_data.get('system_metadata', {}))
        
        # Cross-field validations
        self._validate_cross_field_consistency(profile_data)
        
        return {
            'errors': self.errors,
            'warnings': self.warnings
        }
    
    def _validate_personal_info(self, personal_info: Dict[str, Any]) -> None:
        """Validate personal information section."""
        # Full name
        full_name = personal_info.get('full_name', '').strip()
        if not full_name:
            self.errors.append("Full name is required")
        elif len(full_name) < 2:
            self.errors.append("Full name must be at least 2 characters")
        
        # Date of birth
        dob = personal_info.get('date_of_birth', '')
        if dob:
            if not self._is_valid_iso_date(dob):
                self.errors.append("Date of birth must be in ISO format (YYYY-MM-DD)")
            else:
                # Validate age range (reasonable for elderly care)
                try:
                    dob_date = datetime.fromisoformat(dob).date()
                    today = datetime.now().date()
                    age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
                    if age < 60:
                        self.warnings.append("Age is under 60 - consider if this is correct for elderly care")
                    if age > 120:
                        self.errors.append("Age appears to be over 120 - please verify date of birth")
                except ValueError:
                    self.errors.append("Invalid date of birth format")
        
        # Gender validation
        gender = personal_info.get('gender', '').lower()
        if gender and gender not in ['male', 'female', 'other', 'prefer_not_to_say']:
            self.warnings.append(f"Gender '{gender}' is not a standard value")
        
        # Contact info validation
        contact_info = personal_info.get('contact_info', {})
        if contact_info.get('phone'):
            if not self._is_valid_phone(contact_info['phone']):
                self.warnings.append("Phone number format may be invalid")
        
        if contact_info.get('email'):
            if not self._is_valid_email(contact_info['email']):
                self.errors.append("Invalid email format")
    
    def _validate_medical_history(self, medical_history: Dict[str, Any]) -> None:
        """Validate medical history section."""
        # Chronic conditions
        conditions = medical_history.get('chronic_conditions', [])
        for condition in conditions:
            if condition not in COMMON_MEDICAL_CONDITIONS:
                self.warnings.append(f"Condition '{condition}' is not in common medical conditions list")
        
        # Medications validation
        medications = medical_history.get('medications', [])
        for med in medications:
            self._validate_medication(med)
        
        # Allergies validation
        allergies = medical_history.get('allergies', [])
        for allergy in allergies:
            self._validate_allergy(allergy)
        
        # Surgeries validation
        surgeries = medical_history.get('surgeries', [])
        for surgery in surgeries:
            self._validate_surgery(surgery)
        
        # Vaccinations validation
        vaccinations = medical_history.get('vaccinations', [])
        for vaccine in vaccinations:
            self._validate_vaccination(vaccine)
    
    def _validate_health_metrics(self, health_metrics: Dict[str, Any]) -> None:
        """Validate health metrics section."""
        # Height validation
        height = health_metrics.get('height_cm')
        if height is not None:
            if not (50 <= height <= 250):  # Reasonable height range in cm
                self.warnings.append(f"Height {height}cm appears outside normal range (50-250cm)")
        
        # Weight validation
        weight = health_metrics.get('weight_kg')
        if weight is not None:
            if not (20 <= weight <= 200):  # Reasonable weight range in kg
                self.warnings.append(f"Weight {weight}kg appears outside normal range (20-200kg)")
        
        # Blood type validation
        blood_type = health_metrics.get('blood_type')
        if blood_type and blood_type not in BLOOD_TYPE_OPTIONS:
            self.errors.append(f"Invalid blood type '{blood_type}'")
        
        # Baseline vitals validation
        vitals = health_metrics.get('baseline_vitals', {})
        self._validate_vitals(vitals)
        
        # Mobility status validation
        mobility = health_metrics.get('mobility_status')
        if mobility and mobility not in MOBILITY_STATUS_OPTIONS:
            self.errors.append(f"Invalid mobility status '{mobility}'")
        
        # Cognitive status validation
        cognitive = health_metrics.get('cognitive_status')
        if cognitive and cognitive not in COGNITIVE_STATUS_OPTIONS:
            self.errors.append(f"Invalid cognitive status '{cognitive}'")
    
    def _validate_emergency_contacts(self, contacts: List[Dict[str, Any]]) -> None:
        """Validate emergency contacts."""
        if not contacts:
            self.warnings.append("No emergency contacts provided")
            return
        
        primary_count = 0
        for i, contact in enumerate(contacts):
            # Name validation
            if not contact.get('name', '').strip():
                self.errors.append(f"Emergency contact {i+1}: Name is required")
            
            # Relationship validation
            relationship = contact.get('relationship', '')
            if relationship and relationship not in RELATIONSHIP_OPTIONS:
                self.warnings.append(f"Emergency contact {i+1}: Relationship '{relationship}' is not standard")
            
            # Phone validation
            phone = contact.get('phone', '')
            if not phone:
                self.errors.append(f"Emergency contact {i+1}: Phone number is required")
            elif not self._is_valid_phone(phone):
                self.warnings.append(f"Emergency contact {i+1}: Phone number format may be invalid")
            
            # Check primary designation
            if contact.get('is_primary', False):
                primary_count += 1
        
        if primary_count == 0:
            self.warnings.append("No primary emergency contact designated")
        elif primary_count > 1:
            self.warnings.append("Multiple primary emergency contacts designated - consider designating only one")
    
    def _validate_care_preferences(self, care_preferences: Dict[str, Any]) -> None:
        """Validate care preferences section."""
        # Daily routine validation
        daily_routine = care_preferences.get('daily_routine', {})
        if daily_routine:
            wake_up = daily_routine.get('wake_up_time')
            bedtime = daily_routine.get('bedtime')
            
            if wake_up and not self._is_valid_time(wake_up):
                self.errors.append("Invalid wake up time format (use HH:MM)")
            
            if bedtime and not self._is_valid_time(bedtime):
                self.errors.append("Invalid bedtime format (use HH:MM)")
            
            # Validate meal times
            meal_times = daily_routine.get('meal_times', [])
            for time_str in meal_times:
                if not self._is_valid_time(time_str):
                    self.errors.append(f"Invalid meal time format: {time_str} (use HH:MM)")
        
        # Dietary restrictions validation
        restrictions = care_preferences.get('dietary_restrictions', [])
        for restriction in restrictions:
            if restriction not in DIETARY_RESTRICTIONS:
                self.warnings.append(f"Dietary restriction '{restriction}' is not in standard list")
    
    def _validate_system_metadata(self, metadata: Dict[str, Any]) -> None:
        """Validate system metadata."""
        # Version validation
        version = metadata.get('version', '')
        if version and not version.startswith('1.11.'):
            self.warnings.append(f"Profile version {version} may not be compatible with v1.11 system")
        
        # Consent validation
        if metadata.get('data_privacy_level') == 'sensitive':
            if not metadata.get('consent_given', False):
                self.errors.append("Consent required for sensitive data privacy level")
            if not metadata.get('consent_date'):
                self.warnings.append("Consent date should be recorded for sensitive data")
    
    def _validate_cross_field_consistency(self, profile_data: Dict[str, Any]) -> None:
        """Validate cross-field consistency rules."""
        personal_info = profile_data.get('personal_info', {})
        health_metrics = profile_data.get('health_metrics', {})
        medical_history = profile_data.get('medical_history', {})
        
        # Age vs mobility/cognitive status consistency
        age = personal_info.get('age')
        mobility = health_metrics.get('mobility_status')
        cognitive = health_metrics.get('cognitive_status')
        
        if age and age > 80:
            if mobility == 'independent':
                self.warnings.append("Age over 80 with independent mobility - please verify")
            if cognitive == 'normal':
                self.warnings.append("Age over 80 with normal cognitive status - please verify")
        
        # Medication alerts for serious conditions
        conditions = medical_history.get('chronic_conditions', [])
        medications = medical_history.get('medications', [])
        
        serious_conditions = ['Diabetes Type 1', 'Diabetes Type 2', 'Heart Failure', 'COPD']
        for condition in conditions:
            if condition in serious_conditions:
                # Check if there are medications for this condition
                has_medication = any(
                    condition.lower() in med.get('purpose', '').lower() or
                    condition.lower() in med.get('name', '').lower()
                    for med in medications
                )
                if not has_medication:
                    self.warnings.append(f"No medications recorded for serious condition: {condition}")
    
    def _validate_medication(self, medication: Dict[str, Any]) -> None:
        """Validate individual medication record."""
        if not medication.get('name', '').strip():
            self.errors.append("Medication name is required")
        
        dosage = medication.get('dosage', '')
        if not dosage:
            self.warnings.append(f"Medication '{medication.get('name', 'Unknown')}': Dosage is recommended")
        
        frequency = medication.get('frequency', '')
        if not frequency:
            self.warnings.append(f"Medication '{medication.get('name', 'Unknown')}': Frequency is recommended")
        
        # Last refill date validation
        last_refill = medication.get('last_refill', '')
        if last_refill and not self._is_valid_iso_date(last_refill):
            self.warnings.append(f"Medication '{medication.get('name', 'Unknown')}': Invalid last refill date format")
    
    def _validate_allergy(self, allergy: Dict[str, Any]) -> None:
        """Validate individual allergy record."""
        if not allergy.get('allergen', '').strip():
            self.errors.append("Allergy allergen is required")
        
        if not allergy.get('reaction', '').strip():
            self.warnings.append(f"Allergy '{allergy.get('allergen', 'Unknown')}': Reaction description is recommended")
        
        severity = allergy.get('severity', '')
        if severity and severity not in ['mild', 'moderate', 'severe']:
            self.warnings.append(f"Allergy '{allergy.get('allergen', 'Unknown')}': Severity should be mild/moderate/severe")
    
    def _validate_surgery(self, surgery: Dict[str, Any]) -> None:
        """Validate individual surgery record."""
        if not surgery.get('procedure', '').strip():
            self.errors.append("Surgery procedure name is required")
        
        date = surgery.get('date', '')
        if date and not self._is_valid_iso_date(date):
            self.warnings.append(f"Surgery '{surgery.get('procedure', 'Unknown')}': Invalid date format")
    
    def _validate_vaccination(self, vaccine: Dict[str, Any]) -> None:
        """Validate individual vaccination record."""
        if not vaccine.get('vaccine', '').strip():
            self.errors.append("Vaccine name is required")
        
        date = vaccine.get('date', '')
        if date and not self._is_valid_iso_date(date):
            self.warnings.append(f"Vaccine '{vaccine.get('vaccine', 'Unknown')}': Invalid date format")
    
    def _validate_vitals(self, vitals: Dict[str, Any]) -> None:
        """Validate baseline vitals."""
        # Resting heart rate
        hr = vitals.get('resting_heart_rate')
        if hr is not None:
            if not (40 <= hr <= 120):  # Reasonable resting heart rate range
                self.warnings.append(f"Resting heart rate {hr} bpm appears outside normal range (40-120 bpm)")
        
        # Blood pressure
        bp = vitals.get('blood_pressure', {})
        systolic = bp.get('systolic')
        diastolic = bp.get('diastolic')
        
        if systolic is not None and diastolic is not None:
            if not (70 <= systolic <= 200):
                self.warnings.append(f"Systolic blood pressure {systolic} mmHg appears outside normal range (70-200 mmHg)")
            if not (40 <= diastolic <= 120):
                self.warnings.append(f"Diastolic blood pressure {diastolic} mmHg appears outside normal range (40-120 mmHg)")
            if systolic < diastolic:
                self.errors.append("Systolic blood pressure must be higher than diastolic")
        
        # Temperature
        temp = vitals.get('temperature_c')
        if temp is not None:
            if not (35.0 <= temp <= 40.0):  # Reasonable body temperature range in Celsius
                self.warnings.append(f"Temperature {temp}°C appears outside normal range (35-40°C)")
        
        # Oxygen saturation
        spo2 = vitals.get('oxygen_saturation')
        if spo2 is not None:
            if not (70 <= spo2 <= 100):  # Reasonable SpO2 range
                self.warnings.append(f"Oxygen saturation {spo2}% appears outside normal range (70-100%)")
    
    # Helper validation methods
    def _is_valid_iso_date(self, date_str: str) -> bool:
        """Check if string is valid ISO date (YYYY-MM-DD)."""
        try:
            datetime.fromisoformat(date_str)
            return True
        except ValueError:
            return False
    
    def _is_valid_time(self, time_str: str) -> bool:
        """Check if string is valid time (HH:MM)."""
        pattern = r'^([0-1][0-9]|2[0-3]):([0-5][0-9])$'
        return bool(re.match(pattern, time_str))
    
    def _is_valid_phone(self, phone: str) -> bool:
        """Simple phone number validation."""
        # Remove common separators
        cleaned = re.sub(r'[+\-\(\)\s]', '', phone)
        # Check if it's all digits and reasonable length
        return cleaned.isdigit() and 7 <= len(cleaned) <= 15
    
    def _is_valid_email(self, email: str) -> bool:
        """Simple email validation."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))


class MedicalDataValidator:
    """Specialized validator for medical data with clinical rules."""
    
    @staticmethod
    def check_drug_interactions(medications: List[Dict[str, Any]]) -> List[str]:
        """
        Check for potential drug interactions.
        
        Args:
            medications: List of medication dictionaries
            
        Returns:
            List of interaction warnings
        """
        warnings = []
        
        # Common interaction patterns (simplified)
        drug_classes = {}
        for med in medications:
            name = med.get('name', '').lower()
            # Simplified classification
            if any(word in name for word in ['warfarin', 'coumadin']):
                drug_classes['anticoagulant'] = drug_classes.get('anticoagulant', 0) + 1
            if any(word in name for word in ['aspirin', 'ibuprofen', 'naproxen']):
                drug_classes['nsaid'] = drug_classes.get('nsaid', 0) + 1
            if any(word in name for word in ['lisinopril', 'enalapril', 'ramipril']):
                drug_classes['ace_inhibitor'] = drug_classes.get('ace_inhibitor', 0) + 1
        
        # Check interactions
        if 'anticoagulant' in drug_classes and 'nsaid' in drug_classes:
            warnings.append("⚠️ NSAID with anticoagulant: Increased bleeding risk")
        
        if drug_classes.get('nsaid', 0) > 1:
            warnings.append("⚠️ Multiple NSAIDs: Increased GI and renal risk")
        
        return warnings
    
    @staticmethod
    def check_condition_medication_alignment(
        conditions: List[str],
        medications: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Check if medications align with reported conditions.
        
        Args:
            conditions: List of medical conditions
            medications: List of medications
            
        Returns:
            List of alignment warnings
        """
        warnings = []
        
        # Map conditions to expected medication classes (simplified)
        condition_med_map = {
            'Hypertension': ['ace', 'arb', 'beta', 'calcium', 'diuretic'],
            'Diabetes Type 1': ['insulin'],
            'Diabetes Type 2': ['metformin', 'glipizide', 'insulin'],
            'COPD': ['bronchodilator', 'steroid'],
            'Asthma': ['bronchodilator', 'steroid'],
        }
        
        medication_names = [med.get('name', '').lower() for med in medications]
        
        for condition in conditions:
            expected_classes = condition_med_map.get(condition, [])
            if expected_classes:
                has_expected = any(
                    any(cls in med_name for cls in expected_classes)
                    for med_name in medication_names
                )
                if not has_expected:
                    warnings.append(f"⚠️ No typical medications for condition: {condition}")
        
        return warnings
    
    @staticmethod
    def validate_lab_ranges(
        vitals: Dict[str, Any],
        age: Optional[int] = None
    ) -> List[str]:
        """
        Validate vital signs against age-adjusted normal ranges.
        
        Args:
            vitals: Dictionary of vital signs
            age: Optional age for age-specific ranges
            
        Returns:
            List of range warnings
        """
        warnings = []
        
        # Age-adjusted ranges
        hr = vitals.get('resting_heart_rate')
        if hr is not None:
            if age and age >= 65:
                # Elderly-specific ranges
                if hr < 50 or hr > 100:
                    warnings.append(f"⚠️ Heart rate {hr} bpm outside elderly normal range (50-100)")
            else:
                if hr < 60 or hr > 100:
                    warnings.append(f"⚠️ Heart rate {hr} bpm outside normal range (60-100)")
        
        bp = vitals.get('blood_pressure', {})
        systolic = bp.get('systolic')
        diastolic = bp.get('diastolic')
        
        if systolic is not None and diastolic is not None:
            # Hypertension staging
            if systolic >= 140 or diastolic >= 90:
                warnings.append("⚠️ Blood pressure in hypertensive range")
            elif systolic >= 130 or diastolic >= 85:
                warnings.append("⚠️ Blood pressure in pre-hypertensive range")
        
        return warnings


def validate_profile_completeness(profile_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate profile completeness percentages for each section.
    
    Args:
        profile_data: Complete profile dictionary
        
    Returns:
        Dictionary with completeness percentages per section
    """
    completeness = {}
    
    # Personal info completeness
    personal_info = profile_data.get('personal_info', {})
    personal_fields = ['full_name', 'date_of_birth', 'gender']
    filled_personal = sum(1 for field in personal_fields if personal_info.get(field))
    completeness['personal_info'] = (filled_personal / len(personal_fields)) * 100
    
    # Medical history completeness
    medical_history = profile_data.get('medical_history', {})
    medical_fields = ['chronic_conditions', 'medications', 'allergies']
    filled_medical = sum(1 for field in medical_fields if medical_history.get(field))
    completeness['medical_history'] = (filled_medical / len(medical_fields)) * 100
    
    # Health metrics completeness
    health_metrics = profile_data.get('health_metrics', {})
    health_fields = ['height_cm', 'weight_kg', 'blood_type']
    filled_health = sum(1 for field in health_fields if health_metrics.get(field))
    completeness['health_metrics'] = (filled_health / len(health_fields)) * 100
    
    # Emergency contacts completeness
    emergency_contacts = profile_data.get('emergency_contacts', [])
    if emergency_contacts:
        contact_completeness = []
        for contact in emergency_contacts:
            contact_fields = ['name', 'relationship', 'phone']
            filled_contact = sum(1 for field in contact_fields if contact.get(field))
            contact_completeness.append((filled_contact / len(contact_fields)) * 100)
        completeness['emergency_contacts'] = sum(contact_completeness) / len(contact_completeness)
    else:
        completeness['emergency_contacts'] = 0
    
    # Overall completeness (weighted average)
    weights = {
        'personal_info': 0.25,
        'medical_history': 0.30,
        'health_metrics': 0.20,
        'emergency_contacts': 0.25
    }
    
    overall = sum(completeness[section] * weights[section] for section in weights)
    completeness['overall'] = overall
    
    return completeness
