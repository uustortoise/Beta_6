"""
Streamlit UI components for enhanced profile management.

This module provides reusable UI components for:
- Profile viewing and editing
- Medical history management
- Health metrics input
- Emergency contacts management
- Care preferences configuration
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
from .templates import (
    DEFAULT_PROFILE_TEMPLATE,
    COMMON_MEDICAL_CONDITIONS,
    COMMON_ALLERGIES,
    MOBILITY_STATUS_OPTIONS,
    COGNITIVE_STATUS_OPTIONS,
    BLOOD_TYPE_OPTIONS,
    DIETARY_RESTRICTIONS,
    RELATIONSHIP_OPTIONS
)
from .enhanced_profile import EnhancedProfile
from .validator import ProfileValidator, MedicalDataValidator, validate_profile_completeness


def render_profile_header(profile: EnhancedProfile) -> None:
    """Render profile header with basic information and alerts."""
    # Check if profile is None
    if profile is None:
        st.error("Profile is None. Cannot display profile header.")
        return
    
    # Check if profile.profile exists
    if not hasattr(profile, 'profile') or profile.profile is None:
        st.error("Profile data is missing or corrupted.")
        return
    
    personal_info = profile.profile.get('personal_info', {})
    health_metrics = profile.profile.get('health_metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Name", personal_info.get('full_name', 'Not set'))
    
    with col2:
        age = personal_info.get('age', 'Unknown')
        st.metric("Age", f"{age}" if age else "Unknown")
    
    with col3:
        gender = personal_info.get('gender', 'Not specified')
        st.metric("Gender", gender.capitalize() if gender else "Not specified")
    
    with col4:
        mobility = health_metrics.get('mobility_status', 'Not set')
        st.metric("Mobility", mobility.capitalize() if mobility else "Not set")
    
    # Display medical alerts
    medical_alerts = profile.get_medical_alerts()
    if medical_alerts:
        with st.expander("🚨 Medical Alerts", expanded=True):
            for alert in medical_alerts:
                st.warning(alert)


def render_personal_info_editor(profile: EnhancedProfile) -> None:
    """Render personal information editor form."""
    st.subheader("👤 Personal Information")
    
    personal_info = profile.profile['personal_info']
    
    col1, col2 = st.columns(2)
    
    with col1:
        full_name = st.text_input(
            "Full Name *",
            value=personal_info.get('full_name', ''),
            help="Legal full name of the elder"
        )
        
        date_of_birth = st.text_input(
            "Date of Birth (YYYY-MM-DD)",
            value=personal_info.get('date_of_birth', ''),
            help="Date of birth in ISO format"
        )
        
            # Gender selectbox with safe index calculation
    gender_options = ['', 'male', 'female', 'other', 'prefer_not_to_say']
    current_gender = personal_info.get('gender', '')
    if current_gender in gender_options:
        gender_index = gender_options.index(current_gender)
    else:
        gender_index = 0
    
    gender = st.selectbox(
        "Gender",
        options=gender_options,
        index=gender_index,
        format_func=lambda x: {
            '': 'Select gender',
            'male': 'Male',
            'female': 'Female',
            'other': 'Other',
            'prefer_not_to_say': 'Prefer not to say'
        }[x]
    )
    
    with col2:
        preferred_name = st.text_input(
            "Preferred Name",
            value=personal_info.get('preferred_name', ''),
            help="Name the elder prefers to be called"
        )
        
            # Language preference selectbox with safe index calculation
    lang_options = ['en', 'zh', 'es', 'fr', 'de', 'other']
    current_lang = personal_info.get('language_preference', 'en')
    if current_lang in lang_options:
        lang_index = lang_options.index(current_lang)
    else:
        lang_index = 0
    
    language_preference = st.selectbox(
        "Language Preference",
        options=lang_options,
        index=lang_index,
        format_func=lambda x: {
            'en': 'English',
            'zh': 'Chinese',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'other': 'Other'
        }[x]
    )
    
    # Contact Information
    st.subheader("📞 Contact Information")
    contact_info = personal_info.get('contact_info', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        phone = st.text_input(
            "Phone Number",
            value=contact_info.get('phone', ''),
            help="Primary contact phone number"
        )
        
        email = st.text_input(
            "Email Address",
            value=contact_info.get('email', ''),
            help="Email address for communications"
        )
        
        address = st.text_input(
            "Street Address",
            value=contact_info.get('address', ''),
            help="Street address"
        )
    
    with col2:
        city = st.text_input(
            "City",
            value=contact_info.get('city', ''),
            help="City"
        )
        
        state = st.text_input(
            "State/Province",
            value=contact_info.get('state', ''),
            help="State or province"
        )
        
        postal_code = st.text_input(
            "Postal Code",
            value=contact_info.get('postal_code', ''),
            help="Postal or ZIP code"
        )
        
        country = st.text_input(
            "Country",
            value=contact_info.get('country', ''),
            help="Country"
        )
    
    # Update button
    if st.button("💾 Update Personal Information", type="primary"):
        profile.update_personal_info(
            full_name=full_name,
            date_of_birth=date_of_birth,
            gender=gender,
            preferred_name=preferred_name,
            language_preference=language_preference,
            contact_info={
                'phone': phone,
                'email': email,
                'address': address,
                'city': city,
                'state': state,
                'postal_code': postal_code,
                'country': country
            }
        )
        st.success("Personal information updated!")


def render_medical_history_editor(profile: EnhancedProfile) -> None:
    """Render medical history editor form."""
    st.subheader("🏥 Medical History")
    
    medical_history = profile.profile['medical_history']
    
    # Chronic Conditions
    st.write("#### Chronic Conditions")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_conditions = st.multiselect(
            "Select Chronic Conditions",
            options=COMMON_MEDICAL_CONDITIONS,
            default=medical_history.get('chronic_conditions', []),
            help="Select all chronic medical conditions"
        )
    
    with col2:
        custom_condition = st.text_input(
            "Add Custom Condition",
            value="",
            help="Add a condition not in the list"
        )
        
        if st.button("Add Custom") and custom_condition:
            if custom_condition not in selected_conditions:
                selected_conditions.append(custom_condition)
            st.rerun()
    
    # Update chronic conditions
    if st.button("💾 Update Chronic Conditions"):
        profile.profile['medical_history']['chronic_conditions'] = selected_conditions
        profile._update_timestamps()
        st.success("Chronic conditions updated!")
    
    # Medications Management
    st.write("#### 💊 Medications")
    
    medications = medical_history.get('medications', [])
    
    # Add new medication form
    with st.expander("➕ Add New Medication", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            med_name = st.text_input("Medication Name *", key="med_name")
            med_dosage = st.text_input("Dosage *", key="med_dosage")
            med_frequency = st.text_input("Frequency *", key="med_frequency")
        
        with col2:
            med_purpose = st.text_input("Purpose", key="med_purpose")
            med_doctor = st.text_input("Prescribing Doctor", key="med_doctor")
            med_refill = st.text_input("Last Refill (YYYY-MM-DD)", key="med_refill")
        
        med_notes = st.text_area("Notes", key="med_notes")
        
        if st.button("Add Medication", key="add_med"):
            if med_name and med_dosage and med_frequency:
                profile.add_medication(
                    name=med_name,
                    dosage=med_dosage,
                    frequency=med_frequency,
                    purpose=med_purpose,
                    prescribing_doctor=med_doctor,
                    last_refill=med_refill,
                    notes=med_notes
                )
                st.success("Medication added!")
                st.rerun()
            else:
                st.error("Name, dosage, and frequency are required")
    
    # Display current medications
    if medications:
        st.write("**Current Medications:**")
        for i, med in enumerate(medications):
            with st.expander(f"💊 {med.get('name', 'Unknown')} - {med.get('dosage', '')}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Frequency:** {med.get('frequency', '')}")
                    st.write(f"**Purpose:** {med.get('purpose', 'Not specified')}")
                with col2:
                    st.write(f"**Doctor:** {med.get('prescribing_doctor', 'Not specified')}")
                    st.write(f"**Last Refill:** {med.get('last_refill', 'Not recorded')}")
                st.write(f"**Notes:** {med.get('notes', '')}")
                
                if st.button(f"Remove", key=f"remove_med_{i}"):
                    medications.pop(i)
                    profile._update_timestamps()
                    st.success("Medication removed!")
                    st.rerun()
    
    # Allergies Management
    st.write("#### 🤧 Allergies")
    
    allergies = medical_history.get('allergies', [])
    
    # Add new allergy form
    with st.expander("➕ Add New Allergy", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            allergen = st.text_input("Allergen *", key="allergen")
            if allergen:
                # Suggest common allergies
                matching_allergies = [a for a in COMMON_ALLERGIES if allergen.lower() in a.lower()]
                if matching_allergies:
                    st.caption(f"Suggestions: {', '.join(matching_allergies[:3])}")
        
        with col2:
            reaction = st.text_input("Reaction *", key="reaction")
            severity = st.selectbox(
                "Severity",
                options=['mild', 'moderate', 'severe'],
                index=0,
                key="severity"
            )
        
        allergy_notes = st.text_area("Notes", key="allergy_notes")
        
        if st.button("Add Allergy", key="add_allergy"):
            if allergen and reaction:
                profile.add_allergy(
                    allergen=allergen,
                    reaction=reaction,
                    severity=severity,
                    notes=allergy_notes
                )
                st.success("Allergy added!")
                st.rerun()
            else:
                st.error("Allergen and reaction are required")
    
    # Display current allergies
    if allergies:
        st.write("**Current Allergies:**")
        for i, allergy in enumerate(allergies):
            severity_color = {
                'mild': '🟢',
                'moderate': '🟡',
                'severe': '🔴'
            }.get(allergy.get('severity', 'mild'), '⚪')
            
            with st.expander(f"{severity_color} {allergy.get('allergen', 'Unknown')}", expanded=False):
                st.write(f"**Reaction:** {allergy.get('reaction', '')}")
                st.write(f"**Severity:** {allergy.get('severity', '').capitalize()}")
                st.write(f"**Notes:** {allergy.get('notes', '')}")
                
                if st.button(f"Remove", key=f"remove_allergy_{i}"):
                    allergies.pop(i)
                    profile._update_timestamps()
                    st.success("Allergy removed!")
                    st.rerun()
    
    # Family History
    st.write("#### 👨‍👩‍👧‍👦 Family History")
    
    family_history = medical_history.get('family_history', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        diabetes = st.checkbox("Diabetes", value=family_history.get('diabetes', False))
        hypertension = st.checkbox("Hypertension", value=family_history.get('hypertension', False))
        heart_disease = st.checkbox("Heart Disease", value=family_history.get('heart_disease', False))
    
    with col2:
        stroke = st.checkbox("Stroke", value=family_history.get('stroke', False))
        cancer = st.checkbox("Cancer", value=family_history.get('cancer', False))
        alzheimers = st.checkbox("Alzheimer's/Dementia", value=family_history.get('alzheimers', False))
    
    other_history = st.text_input(
        "Other Family History",
        value=family_history.get('other', ''),
        help="Other family medical history not listed above"
    )
    
    if st.button("💾 Update Family History"):
        profile.profile['medical_history']['family_history'] = {
            'diabetes': diabetes,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'stroke': stroke,
            'cancer': cancer,
            'alzheimers': alzheimers,
            'other': other_history
        }
        profile._update_timestamps()
        st.success("Family history updated!")


def render_health_metrics_editor(profile: EnhancedProfile) -> None:
    """Render health metrics editor form."""
    st.subheader("📊 Health Metrics")
    
    health_metrics = profile.profile['health_metrics']
    
    col1, col2 = st.columns(2)
    
    with col1:
        height_cm = st.number_input(
            "Height (cm)",
            min_value=50.0,
            max_value=250.0,
            value=float(health_metrics.get('height_cm', 170.0)) if health_metrics.get('height_cm') else 170.0,
            step=0.5
        )
        
        weight_kg = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=200.0,
            value=float(health_metrics.get('weight_kg', 70.0)) if health_metrics.get('weight_kg') else 70.0,
            step=0.1
        )
        
        # Blood type selectbox with safe index calculation
        blood_type_options = [''] + BLOOD_TYPE_OPTIONS
        current_blood_type = health_metrics.get('blood_type', '')
        
        # Calculate index safely
        if current_blood_type in blood_type_options:
            blood_type_index = blood_type_options.index(current_blood_type)
        else:
            blood_type_index = 0
        
        blood_type = st.selectbox(
            "Blood Type",
            options=blood_type_options,
            index=blood_type_index,
            format_func=lambda x: 'Select blood type' if x == '' else x
        )
    
    with col2:
        # Mobility status selectbox with safe index calculation
        current_mobility_status = health_metrics.get('mobility_status', 'independent')
        if current_mobility_status in MOBILITY_STATUS_OPTIONS:
            mobility_status_index = MOBILITY_STATUS_OPTIONS.index(current_mobility_status)
        else:
            mobility_status_index = 0
        
        mobility_status = st.selectbox(
            "Mobility Status",
            options=MOBILITY_STATUS_OPTIONS,
            index=mobility_status_index,
            format_func=lambda x: x.capitalize()
        )
        
        # Cognitive status selectbox with safe index calculation
        current_cognitive_status = health_metrics.get('cognitive_status', 'normal')
        if current_cognitive_status in COGNITIVE_STATUS_OPTIONS:
            cognitive_status_index = COGNITIVE_STATUS_OPTIONS.index(current_cognitive_status)
        else:
            cognitive_status_index = 0
        
        cognitive_status = st.selectbox(
            "Cognitive Status",
            options=COGNITIVE_STATUS_OPTIONS,
            index=cognitive_status_index,
            format_func=lambda x: x.replace('_', ' ').capitalize()
        )
    
    # Baseline Vitals
    st.write("#### Baseline Vitals")
    vitals = health_metrics.get('baseline_vitals', {})
    bp = vitals.get('blood_pressure', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        resting_hr = st.number_input(
            "Resting Heart Rate (bpm)",
            min_value=40,
            max_value=120,
            value=int(vitals.get('resting_heart_rate', 72)) if vitals.get('resting_heart_rate') else 72
        )
    
    with col2:
        systolic = st.number_input(
            "Systolic BP (mmHg)",
            min_value=70,
            max_value=200,
            value=int(bp.get('systolic', 120)) if bp.get('systolic') else 120
        )
    
    with col3:
        diastolic = st.number_input(
            "Diastolic BP (mmHg)",
            min_value=40,
            max_value=120,
            value=int(bp.get('diastolic', 80)) if bp.get('diastolic') else 80
        )
    
    with col4:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=35.0,
            max_value=40.0,
            value=float(vitals.get('temperature_c', 37.0)) if vitals.get('temperature_c') else 37.0,
            step=0.1
        )
    
    oxygen_saturation = st.slider(
        "Oxygen Saturation (%)",
        min_value=70,
        max_value=100,
        value=int(vitals.get('oxygen_saturation', 98)) if vitals.get('oxygen_saturation') else 98
    )
    
    if st.button("💾 Update Health Metrics", type="primary"):
        profile.update_health_metrics(
            height_cm=height_cm,
            weight_kg=weight_kg,
            blood_type=blood_type if blood_type else None,
            mobility_status=mobility_status,
            cognitive_status=cognitive_status,
            baseline_vitals_resting_heart_rate=resting_hr,
            blood_pressure=(systolic, diastolic),
            baseline_vitals_temperature_c=temperature,
            baseline_vitals_oxygen_saturation=oxygen_saturation
        )
        st.success("Health metrics updated!")


def render_emergency_contacts_editor(profile: EnhancedProfile) -> None:
    """Render emergency contacts editor form."""
    st.subheader("🚨 Emergency Contacts")
    
    emergency_contacts = profile.profile['emergency_contacts']
    
    # Add new contact form
    with st.expander("➕ Add New Emergency Contact", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            contact_name = st.text_input("Contact Name *", key="contact_name")
            contact_relationship = st.selectbox(
                "Relationship",
                options=RELATIONSHIP_OPTIONS,
                index=0,
                key="contact_relationship",
                format_func=lambda x: x.capitalize()
            )
            contact_phone = st.text_input("Phone Number *", key="contact_phone")
        
        with col2:
            contact_email = st.text_input("Email Address", key="contact_email")
            contact_address = st.text_input("Address", key="contact_address")
            is_primary = st.checkbox("Primary Contact", key="is_primary")
        
        contact_notes = st.text_area("Notes", key="contact_notes")
        
        if st.button("Add Contact", key="add_contact"):
            if contact_name and contact_phone:
                profile.add_emergency_contact(
                    name=contact_name,
                    relationship=contact_relationship,
                    phone=contact_phone,
                    email=contact_email,
                    address=contact_address,
                    is_primary=is_primary,
                    notes=contact_notes
                )
                st.success("Emergency contact added!")
                st.rerun()
            else:
                st.error("Name and phone number are required")
    
    # Display current contacts
    if emergency_contacts:
        st.write("**Current Emergency Contacts:**")
        
        # Check if we have a primary contact
        primary_exists = any(contact.get('is_primary', False) for contact in emergency_contacts)
        
        for i, contact in enumerate(emergency_contacts):
            is_primary = contact.get('is_primary', False)
            primary_badge = " ⭐ PRIMARY" if is_primary else ""
            
            with st.expander(f"👤 {contact.get('name', 'Unknown')}{primary_badge}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Relationship:** {contact.get('relationship', '').capitalize()}")
                    st.write(f"**Phone:** {contact.get('phone', '')}")
                    st.write(f"**Email:** {contact.get('email', 'Not provided')}")
                
                with col2:
                    st.write(f"**Address:** {contact.get('address', 'Not provided')}")
                    st.write(f"**Notes:** {contact.get('notes', 'None')}")
                
                # Mark as primary button (if not already primary)
                if not is_primary and st.button(f"Mark as Primary", key=f"mark_primary_{i}"):
                    # Unmark any existing primary
                    for j, c in enumerate(emergency_contacts):
                        c['is_primary'] = (j == i)
                    profile._update_timestamps()
                    st.success(f"Marked {contact.get('name')} as primary contact!")
                    st.rerun()
                
                # Remove button
                if st.button(f"Remove", key=f"remove_contact_{i}"):
                    emergency_contacts.pop(i)
                    profile._update_timestamps()
                    st.success("Contact removed!")
                    st.rerun()
    else:
        st.info("No emergency contacts added yet. Add at least one contact for safety.")


def render_care_preferences_editor(profile: EnhancedProfile) -> None:
    """Render care preferences editor form."""
    st.subheader("❤️ Care Preferences")
    
    care_preferences = profile.profile['care_preferences']
    daily_routine = care_preferences.get('daily_routine', {})
    
    # Daily Routine
    st.write("#### 📅 Daily Routine")
    
    col1, col2 = st.columns(2)
    
    with col1:
        wake_up = st.text_input(
            "Wake Up Time (HH:MM)",
            value=daily_routine.get('wake_up_time', '07:00'),
            help="Format: HH:MM (24-hour)"
        )
        
        bedtime = st.text_input(
            "Bedtime (HH:MM)",
            value=daily_routine.get('bedtime', '22:00'),
            help="Format: HH:MM (24-hour)"
        )
    
    with col2:
        # Meal times
        meal_times_str = st.text_area(
            "Meal Times (HH:MM, one per line)",
            value="\n".join(daily_routine.get('meal_times', ['08:00', '12:00', '18:00'])),
            help="One meal time per line, format: HH:MM"
        )
        
        medication_times_str = st.text_area(
            "Medication Times (HH:MM, one per line)",
            value="\n".join(daily_routine.get('medication_times', [])),
            help="One medication time per line, format: HH:MM"
        )
    
    preferred_activities = st.text_area(
        "Preferred Activities",
        value="\n".join(daily_routine.get('preferred_activities', [])),
        help="One activity per line"
    )
    
    # Dietary Restrictions
    st.write("#### 🥗 Dietary Restrictions")
    
    dietary_restrictions = st.multiselect(
        "Select Dietary Restrictions",
        options=DIETARY_RESTRICTIONS,
        default=care_preferences.get('dietary_restrictions', []),
        help="Select all applicable dietary restrictions"
    )
    
    custom_restriction = st.text_input(
        "Add Custom Restriction",
        value="",
        help="Add a restriction not in the list"
    )
    
    if st.button("Add Custom Restriction") and custom_restriction:
        if custom_restriction not in dietary_restrictions:
            dietary_restrictions.append(custom_restriction)
        st.rerun()
    
    # Cultural and Religious Preferences
    st.write("#### 🌍 Cultural & Religious Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cultural_prefs = st.text_area(
            "Cultural Preferences",
            value=care_preferences.get('cultural_preferences', {}).get('notes', ''),
            help="Any cultural preferences or considerations"
        )
    
    with col2:
        religious_prefs = st.text_area(
            "Religious Preferences",
            value=care_preferences.get('religious_preferences', {}).get('notes', ''),
            help="Any religious preferences or considerations"
        )
    
    if st.button("💾 Update Care Preferences", type="primary"):
        # Parse meal and medication times
        meal_times = [t.strip() for t in meal_times_str.split('\n') if t.strip()]
        medication_times = [t.strip() for t in medication_times_str.split('\n') if t.strip()]
        preferred_activities_list = [a.strip() for a in preferred_activities.split('\n') if a.strip()]
        
        profile.update_care_preferences(
            'daily_routine',
            {
                'wake_up_time': wake_up,
                'bedtime': bedtime,
                'meal_times': meal_times,
                'medication_times': medication_times,
                'preferred_activities': preferred_activities_list
            }
        )
        
        profile.update_care_preferences('dietary_restrictions', dietary_restrictions)
        
        if cultural_prefs:
            profile.update_care_preferences('cultural_preferences', {'notes': cultural_prefs})
        
        if religious_prefs:
            profile.update_care_preferences('religious_preferences', {'notes': religious_prefs})
        
        st.success("Care preferences updated!")


def render_profile_validation(profile: EnhancedProfile) -> None:
    """Render profile validation results."""
    st.subheader("✅ Profile Validation")
    
    validator = ProfileValidator()
    validation_result = validator.validate_profile(profile.to_dict())
    
    # Medical data validation
    medical_history = profile.profile['medical_history']
    conditions = medical_history.get('chronic_conditions', [])
    medications = medical_history.get('medications', [])
    
    medical_validator = MedicalDataValidator()
    drug_interactions = medical_validator.check_drug_interactions(medications)
    condition_alignment = medical_validator.check_condition_medication_alignment(conditions, medications)
    
    # Completeness score
    completeness = validate_profile_completeness(profile.to_dict())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        error_count = len(validation_result['errors'])
        st.metric("Validation Errors", error_count, delta=None, delta_color="inverse")
    
    with col2:
        warning_count = len(validation_result['warnings'])
        st.metric("Validation Warnings", warning_count, delta=None)
    
    with col3:
        completeness_score = completeness['overall']
        st.metric("Profile Completeness", f"{completeness_score:.1f}%")
    
    # Display errors and warnings
    if validation_result['errors']:
        with st.expander("❌ Validation Errors", expanded=True):
            for error in validation_result['errors']:
                st.error(error)
    
    if validation_result['warnings']:
        with st.expander("⚠️ Validation Warnings", expanded=True):
            for warning in validation_result['warnings']:
                st.warning(warning)
    
    # Display medical validation results
    if drug_interactions:
        with st.expander("💊 Drug Interaction Warnings", expanded=True):
            for warning in drug_interactions:
                st.warning(warning)
    
    if condition_alignment:
        with st.expander("🏥 Condition-Medication Alignment", expanded=True):
            for warning in condition_alignment:
                st.warning(warning)
    
    # Display completeness breakdown
    with st.expander("📊 Completeness Breakdown", expanded=False):
        for section, score in completeness.items():
            if section != 'overall':
                st.progress(int(score), text=f"{section.replace('_', ' ').title()}: {score:.1f}%")


def create_profile_dashboard(profile: EnhancedProfile) -> None:
    """
    Create a complete profile management dashboard.
    
    Args:
        profile: EnhancedProfile instance to manage
    """
    if profile is None:
        st.error("Profile is None. Cannot create dashboard.")
        return
    
    # Profile header
    render_profile_header(profile)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "👤 Personal Info",
        "🏥 Medical History",
        "📊 Health Metrics",
        "🚨 Emergency Contacts",
        "❤️ Care Preferences",
        "✅ Validation"
    ])
    
    with tab1:
        render_personal_info_editor(profile)
    
    with tab2:
        render_medical_history_editor(profile)
    
    with tab3:
        render_health_metrics_editor(profile)
    
    with tab4:
        render_emergency_contacts_editor(profile)
    
    with tab5:
        render_care_preferences_editor(profile)
    
    with tab6:
        render_profile_validation(profile)
    
    # Export and actions section
    st.markdown("---")
    st.subheader("📁 Profile Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Profile to Disk"):
            try:
                profile.save_to_file(f"profile_{profile.elder_id}.json")
                st.success(f"Profile saved to profile_{profile.elder_id}.json")
            except Exception as e:
                st.error(f"Failed to save profile: {str(e)}")
    
    with col2:
        if st.button("📋 Copy Profile JSON"):
            profile_json = profile.to_json()
            st.code(profile_json, language='json')
            st.success("Profile JSON copied to clipboard (view in code block)")
    
    with col3:
        if st.button("🔄 Reset to Defaults"):
            if st.checkbox("Confirm reset - this will clear all profile data"):
                default_profile = EnhancedProfile(profile.elder_id)
                profile.profile = default_profile.profile
                st.success("Profile reset to defaults!")
                st.rerun()


def create_quick_profile_form(elder_id: str) -> Optional[EnhancedProfile]:
    """
    Create a quick profile creation form.
    
    Args:
        elder_id: ID for the new elder
        
    Returns:
        EnhancedProfile instance if created, None otherwise
    """
    st.subheader("🚀 Quick Profile Creation")
    
    with st.form("quick_profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            full_name = st.text_input("Full Name *", help="Legal full name")
            date_of_birth = st.text_input("Date of Birth (YYYY-MM-DD)", help="Optional")
            gender = st.selectbox(
                "Gender",
                options=['', 'male', 'female', 'other', 'prefer_not_to_say'],
                format_func=lambda x: {
                    '': 'Select gender',
                    'male': 'Male',
                    'female': 'Female',
                    'other': 'Other',
                    'prefer_not_to_say': 'Prefer not to say'
                }[x]
            )
        
        with col2:
            phone = st.text_input("Phone Number", help="Primary contact")
            emergency_contact = st.text_input("Emergency Contact Name", help="Primary emergency contact")
            emergency_phone = st.text_input("Emergency Contact Phone", help="Emergency contact phone")
        
        # Quick medical information
        st.write("#### Quick Medical Info")
        chronic_conditions = st.multiselect(
            "Chronic Conditions (select up to 3)",
            options=COMMON_MEDICAL_CONDITIONS,
            help="Select known chronic conditions"
        )[:3]  # Limit to 3 for quick form
        
        allergies = st.multiselect(
            "Allergies (select up to 3)",
            options=COMMON_ALLERGIES,
            help="Select known allergies"
        )[:3]  # Limit to 3 for quick form
        
        submitted = st.form_submit_button("Create Profile")
        
        if submitted:
            if not full_name:
                st.error("Full name is required!")
                return None
            
            # Create profile
            profile = EnhancedProfile(elder_id)
            
            # Update personal info
            profile.update_personal_info(
                full_name=full_name,
                date_of_birth=date_of_birth if date_of_birth else None,
                gender=gender if gender else None,
                contact_info={'phone': phone} if phone else {}
            )
            
            # Add chronic conditions
            for condition in chronic_conditions:
                profile.add_chronic_condition(condition)
            
            # Add allergies
            for allergen in allergies:
                profile.add_allergy(allergen, "Unknown", "mild")
            
            # Add emergency contact if provided
            if emergency_contact and emergency_phone:
                profile.add_emergency_contact(
                    name=emergency_contact,
                    relationship="other",
                    phone=emergency_phone,
                    is_primary=True
                )
            
            st.success("Profile created successfully!")
            return profile
    
    return None
