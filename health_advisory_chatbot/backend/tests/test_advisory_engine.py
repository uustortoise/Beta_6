"""
Unit tests for Health Advisory Chatbot

Tests the core advisory engine functionality without external dependencies.
"""

import pytest
from datetime import datetime, date

# Import modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.schemas import (
    MedicalProfile,
    ADLSummary,
    ICOPEAssessment,
    SleepSummary,
    ChatRequest,
    Medication,
)

from chatbot.knowledge_base import get_guidelines_db, get_drug_interaction_db
from chatbot.predictive import get_risk_stratifier, get_fall_risk_calculator
from chatbot.core import get_context_fusion_engine


class TestKnowledgeBase:
    """Test knowledge base components."""
    
    def test_clinical_guidelines_loading(self):
        """Test that guidelines load correctly."""
        db = get_guidelines_db()
        
        # Should have fall prevention guidelines
        fall_guidelines = db.search_by_category("fall_prevention")
        assert len(fall_guidelines) > 0
        
        # Should have medication safety guidelines
        med_guidelines = db.search_by_category("medication_safety")
        assert len(med_guidelines) > 0
    
    def test_drug_interaction_checking(self):
        """Test drug interaction detection."""
        db = get_drug_interaction_db()
        
        # Test known interaction
        interaction = db.check_interaction("warfarin", "aspirin")
        assert interaction is not None
        assert interaction.severity.value == "major"
    
    def test_anticholinergic_burden(self):
        """Test ACB calculation."""
        db = get_drug_interaction_db()
        
        meds = ["diphenhydramine", "oxybutynin", "sertraline"]
        score, contributing = db.get_anticholinergic_burden(meds)
        
        assert score > 0
        assert len(contributing) > 0


class TestPredictiveModels:
    """Test predictive risk models."""
    
    def test_fall_risk_calculation(self):
        """Test fall risk calculation."""
        calculator = get_fall_risk_calculator()
        
        # Create test profile
        profile = MedicalProfile(
            elder_id="test_001",
            age=80,
            chronic_conditions=["hypertension"],
            medications=[],
        )
        
        score, factors = calculator.calculate_tinetti(
            sitting_balance=1,
            rise_from_chair=1,
            immediate_standing_balance=1,
            standing_balance=1,
            balance_with_eyes_closed=1,
            turning_360_degrees=1,
            sitting_down=1,
            gait_initiation=1,
            step_length_height=1,
            step_symmetry=1,
            step_continuity=1,
            path_deviation=1,
            trunk_sway=1,
            walking_stance=1,
        )
        
        assert score.total_score >= 0
        assert score.total_score <= 28
        assert score.risk_category in ["high", "moderate", "low"]
    
    def test_risk_stratification(self):
        """Test comprehensive risk stratification."""
        stratifier = get_risk_stratifier()
        
        profile = MedicalProfile(
            elder_id="test_002",
            age=85,
            chronic_conditions=["diabetes", "hypertension"],
            medications=[
                Medication(name="metformin", dosage="500mg", frequency="twice daily"),
            ],
            mobility_status="assisted",
        )
        
        assessment = stratifier.assess_comprehensive_risk(
            medical_profile=profile,
        )
        
        assert assessment.overall_risk_score >= 0
        assert assessment.overall_risk_score <= 100
        assert assessment.overall_risk_level.value in ["critical", "high", "moderate", "low", "minimal"]


class TestContextFusion:
    """Test context fusion engine."""
    
    def test_context_building(self):
        """Test health context building."""
        engine = get_context_fusion_engine()
        
        profile = MedicalProfile(
            elder_id="test_003",
            full_name="Test Patient",
            age=75,
            chronic_conditions=["hypertension"],
        )
        
        context = engine.build_health_context(
            elder_id="test_003",
            medical_profile=profile,
        )
        
        assert context.elder_id == "test_003"
        assert context.medical_profile is not None
        assert context.data_completeness["medical_profile"] is True
    
    def test_data_quality_scoring(self):
        """Test data quality calculation."""
        engine = get_context_fusion_engine()
        
        profile = MedicalProfile(
            elder_id="test_004",
            age=70,
        )
        
        context = engine.build_health_context(
            elder_id="test_004",
            medical_profile=profile,
        )
        
        score = engine.get_data_quality_score(context)
        assert 0 <= score <= 100


class TestAPI:
    """Test API endpoints."""
    
    def test_chat_request_validation(self):
        """Test chat request validation."""
        # Valid request
        request = ChatRequest(
            elder_id="test_005",
            message="How did I sleep?",
        )
        assert request.elder_id == "test_005"
        assert request.message == "How did I sleep?"
        
        # Should fail with empty message
        with pytest.raises(Exception):
            ChatRequest(elder_id="test_005", message="")


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_advisory(self):
        """Test complete advisory flow (mock mode)."""
        from chatbot.core import get_advisory_engine
        
        engine = get_advisory_engine()
        
        # Create test data
        profile = MedicalProfile(
            elder_id="test_006",
            full_name="Test Patient",
            age=82,
            chronic_conditions=["diabetes", "hypertension"],
            medications=[
                Medication(name="metformin", dosage="500mg", frequency="twice daily"),
            ],
            mobility_status="independent",
        )
        
        adl = ADLSummary(
            period_start=datetime.now(),
            period_end=datetime.now(),
            days_analyzed=7,
            nighttime_bathroom_visits=3,
        )
        
        request = ChatRequest(
            elder_id="test_006",
            message="What are my fall risks?",
        )
        
        # Process request
        response = engine.process_chat_request(
            request=request,
            medical_profile=profile,
            adl_summary=adl,
        )
        
        assert response.session_id is not None
        assert response.message.content is not None
        assert len(response.message.content) > 0
        assert response.current_risks is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
