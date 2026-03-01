"""
Mock Data Generator for Demo Dashboard

Creates realistic elder profiles with ADL, ICOPE, and sleep data
for demonstrating the health advisory chatbot.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any
import random

import sys
from pathlib import Path
# Add the backend directory to path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from models.schemas import (
    MedicalProfile,
    ADLSummary,
    ICOPEAssessment,
    SleepSummary,
    Medication,
    Allergy,
)


class MockElderGenerator:
    """Generates realistic mock elder data for demo purposes."""
    
    SCENARIOS = {
        "margaret": {
            "name": "Margaret Chen",
            "age": 82,
            "gender": "female",
            "scenario": "high_fall_risk",
            "description": "High fall risk due to nocturia and sedative use",
            "conditions": ["Type 2 Diabetes", "Hypertension", "Osteoarthritis"],
            "medications": [
                {"name": "Metformin", "dosage": "500mg", "frequency": "twice daily"},
                {"name": "Amlodipine", "dosage": "5mg", "frequency": "daily"},
                {"name": "Lorazepam", "dosage": "0.5mg", "frequency": "at bedtime"},
            ],
            "mobility": "assisted",
            "cognitive": "normal",
        },
        "robert": {
            "name": "Robert Williams",
            "age": 78,
            "gender": "male",
            "scenario": "cognitive_concerns",
            "description": "Early memory concerns with sleep apnea risk",
            "conditions": ["Hypertension", "Mild Cognitive Impairment", "Sleep Apnea"],
            "medications": [
                {"name": "Lisinopril", "dosage": "10mg", "frequency": "daily"},
                {"name": "Atorvastatin", "dosage": "20mg", "frequency": "daily"},
                {"name": "Donepezil", "dosage": "5mg", "frequency": "daily"},
            ],
            "mobility": "independent",
            "cognitive": "mild_impairment",
        },
        "helen": {
            "name": "Helen Thompson",
            "age": 75,
            "gender": "female",
            "scenario": "generally_healthy",
            "description": "Generally healthy, active lifestyle",
            "conditions": ["Osteoarthritis"],
            "medications": [
                {"name": "Acetaminophen", "dosage": "500mg", "frequency": "as needed"},
                {"name": "Vitamin D", "dosage": "1000 IU", "frequency": "daily"},
            ],
            "mobility": "independent",
            "cognitive": "normal",
        },
    }
    
    @classmethod
    def generate_all_elders(cls) -> Dict[str, Dict[str, Any]]:
        """Generate complete data for all mock elders."""
        elders = {}
        for elder_id, config in cls.SCENARIOS.items():
            elders[elder_id] = cls.generate_elder(elder_id, config)
        return elders
    
    @classmethod
    def generate_elder(cls, elder_id: str, config: Dict) -> Dict[str, Any]:
        """Generate complete data for a single elder."""
        return {
            "profile": cls._generate_medical_profile(elder_id, config),
            "adl": cls._generate_adl_summary(elder_id, config),
            "icope": cls._generate_icope_assessment(elder_id, config),
            "sleep": cls._generate_sleep_summary(elder_id, config),
            "config": config,
        }
    
    @classmethod
    def _generate_medical_profile(cls, elder_id: str, config: Dict) -> MedicalProfile:
        """Generate medical profile from config."""
        medications = [
            Medication(
                name=m["name"],
                dosage=m["dosage"],
                frequency=m["frequency"],
            )
            for m in config["medications"]
        ]
        
        return MedicalProfile(
            elder_id=elder_id,
            full_name=config["name"],
            age=config["age"],
            gender=config["gender"],
            chronic_conditions=config["conditions"],
            medications=medications,
            mobility_status=config["mobility"],
            cognitive_status=config["cognitive"],
            last_updated=datetime.now(),
        )
    
    @classmethod
    def _generate_adl_summary(cls, elder_id: str, config: Dict) -> ADLSummary:
        """Generate ADL summary based on scenario."""
        scenario = config["scenario"]
        
        # Customize based on scenario
        if scenario == "high_fall_risk":
            nighttime_visits = random.randint(3, 5)
            anomaly_count = random.randint(2, 4)
            transition_time = random.uniform(8, 12)
        elif scenario == "cognitive_concerns":
            nighttime_visits = random.randint(2, 4)
            anomaly_count = random.randint(1, 3)
            transition_time = random.uniform(5, 8)
        else:  # healthy
            nighttime_visits = random.randint(0, 2)
            anomaly_count = random.randint(0, 1)
            transition_time = random.uniform(3, 6)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        return ADLSummary(
            period_start=start_date,
            period_end=end_date,
            days_analyzed=7,
            total_activities=random.randint(100, 300),
            activity_breakdown={
                "bathroom": nighttime_visits * 7,
                "kitchen": random.randint(20, 50),
                "bedroom": random.randint(30, 60),
                "living_room": random.randint(40, 80),
            },
            nighttime_activity_count=nighttime_visits * 7,
            nighttime_bathroom_visits=nighttime_visits * 7,
            anomaly_count=anomaly_count,
            anomaly_rate=anomaly_count / 100,
            average_transition_time=transition_time,
        )
    
    @classmethod
    def _generate_icope_assessment(cls, elder_id: str, config: Dict) -> ICOPEAssessment:
        """Generate ICOPE assessment based on scenario."""
        scenario = config["scenario"]
        
        if scenario == "high_fall_risk":
            scores = {
                "cognitive": 75,
                "locomotor": 55,  # Low - mobility issues
                "psychological": 70,
                "sensory": 65,
                "vitality": 60,
            }
        elif scenario == "cognitive_concerns":
            scores = {
                "cognitive": 60,  # Low - cognitive concerns
                "locomotor": 75,
                "psychological": 55,  # Low - depression/anxiety
                "sensory": 70,
                "vitality": 65,
            }
        else:  # healthy
            scores = {
                "cognitive": 85,
                "locomotor": 80,
                "psychological": 85,
                "sensory": 80,
                "vitality": 85,
            }
        
        domains_at_risk = [
            domain for domain, score in scores.items()
            if score < 60
        ]
        
        return ICOPEAssessment(
            assessment_date=datetime.now().date(),
            cognitive_capacity=scores["cognitive"],
            locomotor_capacity=scores["locomotor"],
            psychological_capacity=scores["psychological"],
            sensory_capacity=scores["sensory"],
            vitality_nutrition=scores["vitality"],
            overall_score=sum(scores.values()) / len(scores),
            domains_at_risk=domains_at_risk,
        )
    
    @classmethod
    def _generate_sleep_summary(cls, elder_id: str, config: Dict) -> SleepSummary:
        """Generate sleep summary based on scenario."""
        scenario = config["scenario"]
        
        if scenario == "high_fall_risk":
            duration = random.uniform(5.5, 6.5)
            efficiency = random.uniform(65, 72)
            awakenings = random.randint(5, 8)
            apnea_risk = "moderate"
        elif scenario == "cognitive_concerns":
            duration = random.uniform(6.0, 7.0)
            efficiency = random.uniform(70, 78)
            awakenings = random.randint(4, 7)
            apnea_risk = "high"
        else:  # healthy
            duration = random.uniform(7.0, 8.0)
            efficiency = random.uniform(82, 90)
            awakenings = random.randint(1, 3)
            apnea_risk = "low"
        
        return SleepSummary(
            analysis_date=datetime.now().date(),
            total_duration_hours=duration,
            time_in_bed_hours=duration + random.uniform(0.5, 1.5),
            sleep_efficiency=efficiency,
            light_sleep_minutes=duration * 60 * 0.5,
            deep_sleep_minutes=duration * 60 * 0.2,
            rem_sleep_minutes=duration * 60 * 0.25,
            awake_minutes=duration * 60 * (1 - efficiency/100),
            quality_score=efficiency,
            awakenings_count=awakenings,
            sleep_apnea_risk=apnea_risk,
            insights=[
                f"Sleep efficiency is {efficiency:.0f}%",
                f"Average {awakenings} awakenings per night",
            ],
        )


# Global mock data store
MOCK_ELDERS = MockElderGenerator.generate_all_elders()


def get_mock_elder(elder_id: str) -> Dict[str, Any]:
    """Get mock elder data by ID."""
    return MOCK_ELDERS.get(elder_id)


def list_mock_elders() -> List[Dict[str, Any]]:
    """List all mock elders (summary only)."""
    return [
        {
            "id": elder_id,
            "name": data["config"]["name"],
            "age": data["config"]["age"],
            "scenario": data["config"]["scenario"],
            "description": data["config"]["description"],
        }
        for elder_id, data in MOCK_ELDERS.items()
    ]


if __name__ == "__main__":
    # Test data generation
    print("=== Mock Elder Data ===\n")
    
    for elder_id, data in MOCK_ELDERS.items():
        print(f"Elder: {data['config']['name']} ({elder_id})")
        print(f"  Age: {data['config']['age']}")
        print(f"  Scenario: {data['config']['scenario']}")
        print(f"  Conditions: {', '.join(data['config']['conditions'])}")
        print(f"  ICOPE Overall: {data['icope'].overall_score:.1f}")
        print(f"  Sleep Efficiency: {data['sleep'].sleep_efficiency:.1f}%")
        print(f"  Nighttime Visits: {data['adl'].nighttime_bathroom_visits}")
        print()
