"""
Profile templates and common medical conditions for enhanced profile management.
"""

DEFAULT_PROFILE_TEMPLATE = {
    "personal_info": {
        "full_name": "",
        "date_of_birth": "",  # ISO format: YYYY-MM-DD
        "age": None,          # Auto-calculated
        "gender": "",         # male/female/other/prefer_not_to_say
        "preferred_name": "",
        "contact_info": {
            "phone": "",
            "email": "",
            "address": "",
            "city": "",
            "state": "",
            "postal_code": "",
            "country": ""
        },
        "language_preference": "en",
        "photo_url": ""
    },
    "medical_history": {
        "chronic_conditions": [],  # List of condition names from COMMON_MEDICAL_CONDITIONS or custom
        "medications": [
            # {
            #     "name": "",
            #     "dosage": "",
            #     "frequency": "",
            #     "purpose": "",
            #     "prescribing_doctor": "",
            #     "last_refill": "",  # ISO format date
            #     "notes": ""
            # }
        ],
        "allergies": [
            # {
            #     "allergen": "",
            #     "reaction": "",
            #     "severity": "mild",  # mild/moderate/severe
            #     "notes": ""
            # }
        ],
        "surgeries": [
            # {
            #     "procedure": "",
            #     "date": "",  # ISO format
            #     "hospital": "",
            #     "surgeon": "",
            #     "notes": ""
            # }
        ],
        "vaccinations": [
            # {
            #     "vaccine": "",
            #     "date": "",  # ISO format
            #     "next_due": "",  # ISO format
            #     "notes": ""
            # }
        ],
        "family_history": {
            "diabetes": False,
            "hypertension": False,
            "heart_disease": False,
            "stroke": False,
            "cancer": False,
            "alzheimers": False,
            "other": ""
        }
    },
    "health_metrics": {
        "height_cm": None,
        "weight_kg": None,
        "blood_type": "",  # A+, A-, B+, B-, O+, O-, AB+, AB-
        "baseline_vitals": {
            "resting_heart_rate": None,  # bpm
            "blood_pressure": {
                "systolic": None,  # mmHg
                "diastolic": None  # mmHg
            },
            "temperature_c": None,  # Celsius
            "oxygen_saturation": None  # percentage
        },
        "mobility_status": "independent",  # independent/assisted/wheelchair/bedridden
        "cognitive_status": "normal",  # normal/mild_impairment/moderate/severe
        "vision_status": "normal",  # normal/glasses/contact_lenses/impaired/blind
        "hearing_status": "normal"  # normal/hearing_aid/impaired/deaf
    },
    "emergency_contacts": [
        # {
        #     "name": "",
        #     "relationship": "",  # spouse, child, friend, etc.
        #     "phone": "",
        #     "email": "",
        #     "address": "",
        #     "is_primary": False,
        #     "notes": ""
        # }
    ],
    "care_preferences": {
        "daily_routine": {
            "wake_up_time": "07:00",
            "bedtime": "22:00",
            "meal_times": ["08:00", "12:00", "18:00"],
            "medication_times": [],
            "preferred_activities": []
        },
        "dietary_restrictions": [],  # vegetarian, vegan, gluten_free, etc.
        "cultural_preferences": {},
        "religious_preferences": {},
        "end_of_life_preferences": {}
    },
    "system_metadata": {
        "created_at": "",  # ISO format
        "last_updated": "",  # ISO format
        "last_assessment": "",  # ISO format
        "data_privacy_level": "standard",  # standard/sensitive/restricted
        "consent_given": False,
        "consent_date": "",  # ISO format
        "version": "1.11.0"
    }
}

COMMON_MEDICAL_CONDITIONS = [
    # Cardiovascular
    "Hypertension",
    "Coronary Artery Disease",
    "Heart Failure",
    "Arrhythmia",
    "Stroke",
    
    # Metabolic/Endocrine
    "Diabetes Type 1",
    "Diabetes Type 2",
    "Thyroid Disorders",
    "Obesity",
    
    # Respiratory
    "COPD",
    "Asthma",
    "Sleep Apnea",
    
    # Musculoskeletal
    "Osteoarthritis",
    "Rheumatoid Arthritis",
    "Osteoporosis",
    "Chronic Back Pain",
    
    # Neurological
    "Alzheimer's Disease",
    "Dementia",
    "Parkinson's Disease",
    "Epilepsy",
    "Migraine",
    
    # Mental Health
    "Depression",
    "Anxiety",
    "Bipolar Disorder",
    
    # Gastrointestinal
    "GERD",
    "Irritable Bowel Syndrome",
    "Inflammatory Bowel Disease",
    
    # Renal
    "Chronic Kidney Disease",
    
    # Others
    "Cancer",
    "HIV/AIDS",
    "Anemia",
    "Glaucoma",
    "Cataracts"
]

COMMON_ALLERGIES = [
    "Penicillin",
    "Sulfa Drugs",
    "Aspirin",
    "Ibuprofen",
    "Codeine",
    "Latex",
    "Pollen",
    "Dust Mites",
    "Mold",
    "Animal Dander",
    "Peanuts",
    "Tree Nuts",
    "Shellfish",
    "Eggs",
    "Milk",
    "Soy",
    "Wheat"
]

MOBILITY_STATUS_OPTIONS = [
    "independent",
    "assisted",  # uses cane, walker, etc.
    "wheelchair",
    "bedridden"
]

COGNITIVE_STATUS_OPTIONS = [
    "normal",
    "mild_impairment",
    "moderate",
    "severe"
]

BLOOD_TYPE_OPTIONS = [
    "A+",
    "A-",
    "B+",
    "B-",
    "O+",
    "O-",
    "AB+",
    "AB-"
]

DIETARY_RESTRICTIONS = [
    "vegetarian",
    "vegan",
    "gluten_free",
    "lactose_free",
    "low_sodium",
    "low_sugar",
    "kosher",
    "halal"
]

RELATIONSHIP_OPTIONS = [
    "spouse",
    "child",
    "grandchild",
    "sibling",
    "friend",
    "neighbor",
    "caregiver",
    "other"
]
