
import pytest
from datetime import datetime
from elderlycare_v1_16.services.profile_service import ProfileService
from elderlycare_v1_16.services.sleep_service import SleepService
from elderlycare_v1_16.services.adl_service import ADLService
from elderlycare_v1_16.services.insight_service import InsightService


def _seed_hypertension_rule(db_mgr):
    conditions = '{"logic":"AND","rules":[{"metric":"deep_sleep_pct","operator":"less_than","value":15,"value_type":"absolute"},{"metric":"night_toilet_visits","operator":"greater_equal","value":3,"value_type":"absolute"}]}'
    with db_mgr.get_connection() as conn:
        conn.execute('''
            INSERT INTO alert_rules_v2
            (rule_name, required_condition, conditions, alert_severity, alert_message, enabled)
            VALUES (?, ?, ?, ?, ?, 1)
        ''', (
            "Hypertension Risk: Poor Sleep Stability",
            "hypertension",
            conditions,
            "high",
            "High risk of morning blood pressure spikes",
        ))
        conn.commit()


def test_hypertension_alert_generation(test_db):
    """
    Verify that an alert is generated when:
    1. Patient has Hypertension history.
    2. Deep Sleep is low (<15%).
    3. Night toileting is frequent (>=3).
    """
    
    elder_id = "HK999"
    _seed_hypertension_rule(test_db)
    
    # 1. Setup Profile with Hypertension
    ProfileService().create_or_update_elder(elder_id, {
        "personal_info": {"full_name": "Test Elder"},
        "medical_history": {
            "chronic_conditions": ["Hypertension", "Diabetes"]
        }
    })
    
    # 2. Setup Poor Sleep (10% Deep Sleep)
    SleepService().save_sleep_analysis(elder_id, {
        "total_duration_hours": 6.0,
        "sleep_efficiency": 0.75,
        "quality_score": 60,
        "stages_summary": {"Deep": 10, "Light": 60, "REM": 20, "Awake": 10}, # 10% Deep
        "sleep_periods": [],
        "insights": []
    }, datetime.now().strftime("%Y-%m-%d"))
    
    # 3. Setup Frequent Toileting (3 times tonight)
    # We need to ensure the count logic matches. 
    # InsightService uses SQL count on 'record_date'.
    adl_svc = ADLService()
    today = datetime.now().strftime("%Y-%m-%d")
    for i in range(3):
        adl_svc.save_adl_event(elder_id, {
            "timestamp": f"{today} 0{i+1}:00:00",
            "activity": "toileting",
            "room": "Bathroom",
            "confidence": 1.0
        })
                              
    # 4. Run Insight Engine
    insight_svc = InsightService()
    alerts = insight_svc.run_daily_analysis(elder_id)
    
    # 5. Verify Alert
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert['title'] == "Hypertension Risk: Poor Sleep Stability"
    assert alert['severity'] == "high"
    assert "High risk of morning blood pressure spikes" in alert['message']
    
    # Verify it was saved to DB
    with test_db.get_connection() as conn:
        row = conn.execute("SELECT * FROM alerts WHERE elder_id = ?", (elder_id,)).fetchone()
        assert row is not None
        assert row['title'] == alert['title']

def test_no_alert_if_healthy(test_db):
    """
    Verify NO alert is generated if sleep is good or no hypertension.
    """
    elder_id = "HK888"
    _seed_hypertension_rule(test_db)
    ProfileService().create_or_update_elder(elder_id, {
        "personal_info": {"full_name": "Healthy Elder"},
        "medical_history": {
            "chronic_conditions": ["Arthritis"] # No Hypertension
        }
    })
    
    # Poor sleep + Toileting -> Should NOT trigger (condition not met)
    SleepService().save_sleep_analysis(elder_id, {
        "stages_summary": {"Deep": 10}
    }, datetime.now().strftime("%Y-%m-%d"))
    
    adl_svc = ADLService()
    today = datetime.now().strftime("%Y-%m-%d")
    for i in range(3):
        adl_svc.save_adl_event(elder_id, {
            "timestamp": f"{today} 01:00:0{i}",
            "activity": "toileting",
            "room": "Bathroom",
            "confidence": 1.0
        })
        
    alerts = InsightService().run_daily_analysis(elder_id)
    assert len(alerts) == 0


def test_fallen_state_alert_generation_from_activity_segment(test_db, monkeypatch):
    elder_id = "HK777"
    today = datetime.now().strftime("%Y-%m-%d")
    monkeypatch.setenv("ENABLE_FALLEN_STATE_ALERTS", "true")
    monkeypatch.setenv("FALLEN_STATE_WARNING_MINUTES", "2")
    monkeypatch.setenv("FALLEN_STATE_CRITICAL_MINUTES", "5")
    monkeypatch.setenv("FALLEN_STATE_RISK_ROOMS", "kitchen,bathroom,entrance,livingroom")

    ProfileService().create_or_update_elder(elder_id, {
        "personal_info": {"full_name": "Fall Pivot Elder"},
        "medical_history": {"chronic_conditions": []},
    })

    with test_db.get_connection() as conn:
        conn.execute(
            """
            INSERT INTO activity_segments
            (elder_id, room, activity_type, start_time, end_time, duration_minutes, avg_confidence, event_count, record_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                elder_id,
                "Kitchen",
                "lying",
                f"{today} 10:00:00",
                f"{today} 10:03:00",
                3.0,
                0.9,
                18,
                today,
            ),
        )
        conn.commit()

    alerts = InsightService().run_daily_analysis(elder_id, analysis_date=today)
    fallen_alerts = [a for a in alerts if a.get("alert_type") == "safety_fallen_state"]
    assert len(fallen_alerts) == 1
    assert fallen_alerts[0]["severity"] == "high"
    assert "Kitchen" in fallen_alerts[0]["title"]


def test_fallen_state_critical_and_deduplicated_in_db(test_db, monkeypatch):
    elder_id = "HK776"
    today = datetime.now().strftime("%Y-%m-%d")
    monkeypatch.setenv("ENABLE_FALLEN_STATE_ALERTS", "true")
    monkeypatch.setenv("FALLEN_STATE_WARNING_MINUTES", "2")
    monkeypatch.setenv("FALLEN_STATE_CRITICAL_MINUTES", "5")
    monkeypatch.setenv("FALLEN_STATE_RISK_ROOMS", "kitchen,bathroom,entrance,livingroom")

    ProfileService().create_or_update_elder(elder_id, {
        "personal_info": {"full_name": "Fall Pivot Elder Critical"},
        "medical_history": {"chronic_conditions": []},
    })

    with test_db.get_connection() as conn:
        conn.execute(
            """
            INSERT INTO activity_segments
            (elder_id, room, activity_type, start_time, end_time, duration_minutes, avg_confidence, event_count, record_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                elder_id,
                "Kitchen",
                "lying_down",
                f"{today} 10:00:00",
                f"{today} 10:10:00",
                10.0,
                0.95,
                60,
                today,
            ),
        )
        conn.commit()

    svc = InsightService()
    first = svc.run_daily_analysis(elder_id, analysis_date=today)
    second = svc.run_daily_analysis(elder_id, analysis_date=today)
    first_fallen = [a for a in first if a.get("alert_type") == "safety_fallen_state"]
    second_fallen = [a for a in second if a.get("alert_type") == "safety_fallen_state"]
    assert len(first_fallen) == 1
    assert first_fallen[0]["severity"] == "critical"
    assert len(second_fallen) == 1

    with test_db.get_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM alerts
            WHERE elder_id = ?
              AND alert_type = 'safety_fallen_state'
              AND alert_date = ?
            """,
            (elder_id, today),
        ).fetchone()
        count = row["cnt"] if hasattr(row, "keys") else row[0]
        assert int(count) == 1
