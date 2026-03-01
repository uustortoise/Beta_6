
import pytest
import json
from datetime import datetime
from elderlycare_v1_16.services.profile_service import ProfileService
from elderlycare_v1_16.services.sleep_service import SleepService
from elderlycare_v1_16.services.adl_service import ADLService

def test_profile_service_crud(test_db):
    """
    Test creating, reading, and updating an elder profile.
    """
    svc = ProfileService()
    
    # 1. Create
    elder_id = "HK999"
    data = {
        "personal_info": {
            "full_name": "Test Elder",
            "gender": "Female",
            "date_of_birth": "1940-01-01"
        }
    }
    svc.create_or_update_elder(elder_id, data)
    
    # 2. Read
    profile = svc.get_profile(elder_id)
    assert profile is not None
    assert profile['full_name'] == "Test Elder"
    assert profile['gender'] == "Female"
    
    # 3. Update Medical History
    history_data = {
        "medical_history": {
            "chronic_conditions": ["Hypertension"],
            "medications": ["Amlodipine"]
        }
    }
    svc.create_or_update_elder(elder_id, history_data)
    
    profile_updated = svc.get_profile(elder_id)
    assert "Hypertension" in profile_updated['medical_history']['chronic_conditions']

def test_sleep_service(test_db):
    """
    Test saving and retrieving sleep analysis.
    """
    svc = SleepService()
    elder_id = "HK999"
    # Ensure elder exists (FK constraint)
    ProfileService().create_or_update_elder(elder_id, {"personal_info": {"full_name": "Test"}})
    
    analysis_date = "2025-01-01"
    sleep_result = {
        "duration_hours": 7.5,
        "efficiency": 0.85,
        "quality_score": 80,
        "stages": {"Deep": 20, "Light": 50, "REM": 20, "Awake": 10},
        "sleep_periods": [],
        "insights": ["Good sleep"]
    }
    
    svc.save_sleep_analysis(elder_id, sleep_result, analysis_date)
    
    # Retrieve
    saved = svc.get_latest_sleep(elder_id)
    assert saved is not None
    assert saved['analysis_date'] == analysis_date
    assert saved['duration_hours'] == 7.5
    assert saved['efficiency_percent'] == 85.0 # Logic multiplies by 100? Let's check impl.
    
def test_adl_service(test_db):
    """
    Test ADL event logging.
    """
    svc = ADLService()
    elder_id = "HK999"
    ProfileService().create_or_update_elder(elder_id, {"personal_info": {"full_name": "Test"}})
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    svc.save_adl_event(elder_id, "toileting", "Bathroom", timestamp, confidence=0.95)
    
    # Verify via direct DB query (since ADLService doesn't have a specific 'get_all' method exposed for this yet easily)
    with test_db.get_connection() as conn:
        row = conn.execute("SELECT * FROM adl_history WHERE elder_id = ?", (elder_id,)).fetchone()
        assert row is not None
        assert row['activity_type'] in ('toilet', 'toileting')
        assert row['room'] == 'Bathroom'
