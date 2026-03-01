"""
Enhanced Profile Management Module for Multi-Elderly Care Platform v1.11

This module provides comprehensive elder profile management including:
- Extended personal information (date of birth, contact info, etc.)
- Medical history (chronic conditions, medications, allergies)
- Health metrics (baseline vitals, mobility status)
- Emergency contacts and care preferences
- Data validation and migration utilities
"""

from .enhanced_profile import EnhancedProfile, create_empty_profile
from .validator import ProfileValidator, MedicalDataValidator
from .templates import DEFAULT_PROFILE_TEMPLATE, COMMON_MEDICAL_CONDITIONS

__all__ = [
    "EnhancedProfile",
    "create_empty_profile",
    "ProfileValidator",
    "MedicalDataValidator",
    "DEFAULT_PROFILE_TEMPLATE",
    "COMMON_MEDICAL_CONDITIONS",
]

__version__ = "1.11.0"
