"""
Context Fusion Engine

Aggregates and synthesizes data from all health sources:
- Medical profile (EnhancedProfile integration)
- ADL history
- ICOPE assessments
- Sleep analysis
- Risk assessments
- Predictive trajectories

Produces a unified HealthContext for advisory generation.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from models.schemas import (
    HealthContext,
    MedicalProfile,
    ADLSummary,
    ICOPEAssessment,
    SleepSummary,
    RiskAssessment,
    TrajectoryPrediction,
)


class ContextFusionEngine:
    """
    Fuses multi-source health data into unified context.
    
    Key capabilities:
    - Data aggregation from Beta 5.5 services
    - Temporal alignment
    - Missing data handling
    - Context summarization
    """
    
    def __init__(self):
        """Initialize context fusion engine."""
        self.default_lookback_days = 7
    
    def build_health_context(
        self,
        elder_id: str,
        medical_profile: Optional[MedicalProfile] = None,
        adl_summary: Optional[ADLSummary] = None,
        icope_assessment: Optional[ICOPEAssessment] = None,
        sleep_summary: Optional[SleepSummary] = None,
        include_risk_assessment: bool = True,
        include_trajectories: bool = True,
    ) -> HealthContext:
        """
        Build comprehensive health context from all available data.
        
        Args:
            elder_id: Elder identifier
            medical_profile: Medical history data
            adl_summary: ADL activity summary
            icope_assessment: ICOPE assessment
            sleep_summary: Sleep analysis
            include_risk_assessment: Calculate risk assessment
            include_trajectories: Calculate trajectories
        
        Returns:
            Unified HealthContext
        """
        context = HealthContext(
            elder_id=elder_id,
            context_timestamp=datetime.now(),
            medical_profile=medical_profile,
            adl_summary=adl_summary,
            icope_assessment=icope_assessment,
            sleep_summary=sleep_summary,
            data_completeness={
                "medical_profile": medical_profile is not None,
                "adl_data": adl_summary is not None,
                "icope_data": icope_assessment is not None,
                "sleep_data": sleep_summary is not None,
            }
        )
        
        # Calculate risk assessment if requested
        if include_risk_assessment and medical_profile:
            from chatbot.predictive.risk_stratifier import get_risk_stratifier
            risk_stratifier = get_risk_stratifier()
            
            risk_assessment = risk_stratifier.assess_comprehensive_risk(
                medical_profile=medical_profile,
                adl_summary=adl_summary,
                icope=icope_assessment,
                sleep=sleep_summary,
            )
            context.risk_assessment = risk_assessment
        
        # Calculate trajectories if requested
        if include_trajectories:
            from chatbot.predictive.trajectory_models import get_trajectory_predictor
            predictor = get_trajectory_predictor()
            
            trajectories = []
            
            # Cognitive trajectory
            if medical_profile:
                cognitive_traj = predictor.predict_cognitive_trajectory(
                    medical_profile=medical_profile,
                    icope_history=[icope_assessment] if icope_assessment else [],
                    prediction_months=12,
                )
                trajectories.append(cognitive_traj)
            
            # Mobility trajectory
            if medical_profile:
                mobility_traj = predictor.predict_mobility_trajectory(
                    medical_profile=medical_profile,
                    adl_history=[adl_summary] if adl_summary else [],
                    icope_history=[icope_assessment] if icope_assessment else [],
                    prediction_months=12,
                )
                trajectories.append(mobility_traj)
            
            # Sleep trajectory
            if sleep_summary:
                sleep_traj = predictor.predict_sleep_trajectory(
                    sleep_history=[sleep_summary],
                    medical_profile=medical_profile,
                    prediction_months=6,
                )
                trajectories.append(sleep_traj)
            
            context.trajectories = trajectories
        
        # Generate human-readable summary
        context.context_summary = self._generate_context_summary(context)
        
        return context
    
    def build_context_from_beta5_services(
        self,
        elder_id: str,
        beta5_base_path: str = "../../../backend",  # Relative to this module
        days_lookback: int = 7,
    ) -> HealthContext:
        """
        Build health context by querying Beta 5.5 services.
        
        This method integrates with existing Beta 5.5 services:
        - ProfileService for medical history
        - ADLService for activity data
        - ICOPEService for assessments
        - SleepService for sleep analysis
        
        Note: This is a bridge method that maintains separation
        between the chatbot module and Beta 5.5 core.
        
        Args:
            elder_id: Elder identifier
            beta5_base_path: Path to Beta 5.5 backend
            days_lookback: Days of history to retrieve
        
        Returns:
            HealthContext with data from Beta 5.5
        """
        # Note: In production, these would call actual Beta 5.5 services
        # For now, we provide the structure for integration
        
        medical_profile = self._fetch_medical_profile(elder_id, beta5_base_path)
        adl_summary = self._fetch_adl_summary(elder_id, days_lookback, beta5_base_path)
        icope = self._fetch_icope_assessment(elder_id, beta5_base_path)
        sleep = self._fetch_sleep_summary(elder_id, beta5_base_path)
        
        return self.build_health_context(
            elder_id=elder_id,
            medical_profile=medical_profile,
            adl_summary=adl_summary,
            icope_assessment=icope,
            sleep_summary=sleep,
        )
    
    def _fetch_medical_profile(
        self,
        elder_id: str,
        base_path: str
    ) -> Optional[MedicalProfile]:
        """
        Fetch medical profile from Beta 5.5 EnhancedProfile.
        
        This is a bridge method - in production, calls ProfileService.
        """
        try:
            # Import Beta 5.5 modules (these are external dependencies)
            import sys
            sys.path.insert(0, base_path)
            
            from elderlycare_v1_16.profile.enhanced_profile import EnhancedProfile
            from elderlycare_v1_16.services.profile_service import ProfileService
            
            # Attempt to load profile
            # Note: Actual implementation would use ProfileService
            profile_data = {}  # Would come from service
            
            if profile_data:
                return self._convert_enhanced_profile_to_model(elder_id, profile_data)
            
        except ImportError:
            # Beta 5.5 not available in test environment
            pass
        except Exception as e:
            print(f"Error fetching medical profile: {e}")
        
        return None
    
    def _convert_enhanced_profile_to_model(
        self,
        elder_id: str,
        profile_data: Dict
    ) -> MedicalProfile:
        """
        Convert Beta 5.5 EnhancedProfile to our MedicalProfile model.
        
        Args:
            elder_id: Elder ID
            profile_data: EnhancedProfile data
        
        Returns:
            MedicalProfile
        """
        personal_info = profile_data.get("personal_info", {})
        medical_history = profile_data.get("medical_history", {})
        health_metrics = profile_data.get("health_metrics", {})
        
        # Convert medications
        from models.schemas import Medication, Allergy
        
        medications = [
            Medication(
                name=m.get("name", ""),
                dosage=m.get("dosage", ""),
                frequency=m.get("frequency", ""),
                purpose=m.get("purpose"),
                prescribing_doctor=m.get("prescribing_doctor"),
                last_refill=m.get("last_refill"),
                notes=m.get("notes"),
            )
            for m in medical_history.get("medications", [])
        ]
        
        # Convert allergies
        allergies = [
            Allergy(
                allergen=a.get("allergen", ""),
                reaction=a.get("reaction", ""),
                severity=a.get("severity", "mild"),
                notes=a.get("notes"),
            )
            for a in medical_history.get("allergies", [])
        ]
        
        # Baseline vitals
        baseline_vitals = health_metrics.get("baseline_vitals", {})
        bp = baseline_vitals.get("blood_pressure", {})
        
        return MedicalProfile(
            elder_id=elder_id,
            full_name=personal_info.get("full_name"),
            age=personal_info.get("age"),
            gender=personal_info.get("gender"),
            chronic_conditions=medical_history.get("chronic_conditions", []),
            acute_conditions=medical_history.get("acute_conditions", []),
            surgical_history=medical_history.get("surgical_history", []),
            family_history=medical_history.get("family_history", {}),
            medications=medications,
            allergies=allergies,
            baseline_blood_pressure=bp if bp else None,
            baseline_heart_rate=baseline_vitals.get("heart_rate"),
            baseline_weight_kg=health_metrics.get("weight_kg"),
            mobility_status=health_metrics.get("mobility_status"),
            cognitive_status=health_metrics.get("cognitive_status"),
            last_updated=datetime.now(),
        )
    
    def _fetch_adl_summary(
        self,
        elder_id: str,
        days: int,
        base_path: str
    ) -> Optional[ADLSummary]:
        """Fetch ADL summary from Beta 5.5 ADLService."""
        # Placeholder for actual service call
        # Would query ADLService for activity data
        return None
    
    def _fetch_icope_assessment(
        self,
        elder_id: str,
        base_path: str
    ) -> Optional[ICOPEAssessment]:
        """Fetch latest ICOPE assessment."""
        # Placeholder for actual service call
        return None
    
    def _fetch_sleep_summary(
        self,
        elder_id: str,
        base_path: str
    ) -> Optional[SleepSummary]:
        """Fetch latest sleep analysis."""
        # Placeholder for actual service call
        return None
    
    def _generate_context_summary(self, context: HealthContext) -> str:
        """Generate human-readable summary of health context."""
        parts = []
        
        # Demographics
        if context.medical_profile:
            profile = context.medical_profile
            demo_parts = []
            if profile.full_name:
                demo_parts.append(profile.full_name)
            if profile.age:
                demo_parts.append(f"age {profile.age}")
            if profile.gender:
                demo_parts.append(profile.gender)
            
            if demo_parts:
                parts.append(f"Patient: {', '.join(demo_parts)}")
            
            # Conditions
            if profile.chronic_conditions:
                parts.append(f"Conditions: {', '.join(profile.chronic_conditions[:5])}")
            
            # Medications
            if profile.medications:
                parts.append(f"Medications: {len(profile.medications)} active")
        
        # ADL Summary
        if context.adl_summary:
            adl = context.adl_summary
            adl_parts = []
            if adl.nighttime_bathroom_visits > 0:
                adl_parts.append(f"{adl.nighttime_bathroom_visits} nightly bathroom visits")
            if adl.anomaly_count > 0:
                adl_parts.append(f"{adl.anomaly_count} anomalies")
            
            if adl_parts:
                parts.append(f"Recent activity: {', '.join(adl_parts)}")
        
        # ICOPE
        if context.icope_assessment:
            icope = context.icope_assessment
            if icope.overall_score:
                parts.append(f"ICOPE overall: {icope.overall_score:.0f}/100")
            
            if icope.domains_at_risk:
                parts.append(f"At-risk domains: {', '.join(icope.domains_at_risk)}")
        
        # Sleep
        if context.sleep_summary:
            sleep = context.sleep_summary
            sleep_parts = []
            if sleep.sleep_efficiency:
                sleep_parts.append(f"{sleep.sleep_efficiency:.0f}% efficiency")
            if sleep.total_duration_hours:
                sleep_parts.append(f"{sleep.total_duration_hours:.1f}h duration")
            
            if sleep_parts:
                parts.append(f"Sleep: {', '.join(sleep_parts)}")
        
        # Risk summary
        if context.risk_assessment:
            risk = context.risk_assessment
            parts.append(f"Overall risk: {risk.overall_risk_level.value} ({risk.overall_risk_score:.0f}/100)")
            
            if risk.critical_alerts:
                parts.append(f"⚠️ Alerts: {', '.join(risk.critical_alerts[:3])}")
        
        return "\n".join(parts) if parts else "Limited health data available"
    
    def get_data_quality_score(self, context: HealthContext) -> float:
        """
        Calculate data quality score (0-100).
        
        Based on completeness and recency of data sources.
        """
        scores = []
        
        # Completeness (60 points)
        completeness = context.data_completeness
        if completeness.get("medical_profile"):
            scores.append(20)
        if completeness.get("adl_data"):
            scores.append(15)
        if completeness.get("icope_data"):
            scores.append(15)
        if completeness.get("sleep_data"):
            scores.append(10)
        
        # Recency (40 points)
        now = datetime.now()
        
        if context.adl_summary:
            days_old = (now - context.adl_summary.period_end).days
            if days_old <= 1:
                scores.append(10)
            elif days_old <= 7:
                scores.append(7)
            elif days_old <= 30:
                scores.append(4)
        
        if context.icope_assessment:
            days_old = (now - datetime.combine(context.icope_assessment.assessment_date, datetime.min.time())).days
            if days_old <= 30:
                scores.append(10)
            elif days_old <= 90:
                scores.append(7)
            elif days_old <= 180:
                scores.append(4)
        
        if context.sleep_summary:
            days_old = (now - datetime.combine(context.sleep_summary.analysis_date, datetime.min.time())).days
            if days_old <= 1:
                scores.append(10)
            elif days_old <= 7:
                scores.append(7)
            elif days_old <= 30:
                scores.append(4)
        
        if context.medical_profile:
            days_old = (now - context.medical_profile.last_updated).days
            if days_old <= 30:
                scores.append(10)
            elif days_old <= 90:
                scores.append(7)
            elif days_old <= 180:
                scores.append(4)
        
        return min(100, sum(scores))


# Singleton
_context_fusion: Optional[ContextFusionEngine] = None


def get_context_fusion_engine() -> ContextFusionEngine:
    """Get or create singleton context fusion engine."""
    global _context_fusion
    if _context_fusion is None:
        _context_fusion = ContextFusionEngine()
    return _context_fusion
