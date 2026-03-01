"""
Multi-Domain Risk Stratifier

Integrates data from all health domains to produce comprehensive risk assessment.
Uses evidence-based scoring algorithms with weighted risk factors.

Risk Domains:
- Fall risk
- Cognitive decline risk
- Sleep disorder risk
- Medication-related risk
- Frailty progression risk
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import math

from models.schemas import (
    RiskAssessment,
    RiskFactor,
    SeverityLevel,
    MedicalProfile,
    ADLSummary,
    ICOPEAssessment,
    SleepSummary,
)


class RiskStratifier:
    """
    Comprehensive risk stratification engine.
    
    Combines multiple risk assessment algorithms to produce
    a unified, evidence-based risk profile.
    """
    
    # Risk score thresholds
    RISK_THRESHOLDS = {
        "critical": 80,
        "high": 60,
        "moderate": 40,
        "low": 20,
    }
    
    def __init__(self):
        """Initialize risk stratifier with scoring weights."""
        self._initialize_scoring_weights()
    
    def _initialize_scoring_weights(self) -> None:
        """Initialize evidence-based risk factor weights."""
        
        # Fall risk weights (based on Tinetti, Morse, and AGS guidelines)
        self.fall_weights = {
            # History (strongest predictor)
            "history_of_falls": 25,
            "multiple_falls": 35,
            
            # Gait/Mobility
            "gait_abnormality": 20,
            "balance_impairment": 20,
            "mobility_aid_use": 15,
            "slow_gait_speed": 15,
            
            # Medical conditions
            "orthostatic_hypotension": 15,
            "vision_impairment": 10,
            "cognitive_impairment": 10,
            "parkinsons_disease": 15,
            "stroke_history": 15,
            "arthritis": 5,
            
            # Medications
            "sedative_use": 15,
            "hypnotic_use": 15,
            "antihypertensive_polypharmacy": 10,
            "diuretic_use": 10,
            
            # Environmental/Functional
            "adl_dependence": 15,
            "fear_of_falling": 10,
            "foot_problems": 5,
        }
        
        # Cognitive decline risk weights
        self.cognitive_weights = {
            # Demographics
            "age_over_85": 10,
            "low_education": 5,
            
            # Medical conditions
            "mild_cognitive_impairment": 30,
            "diabetes": 10,
            "hypertension": 10,
            "cardiovascular_disease": 10,
            "stroke": 20,
            "atrial_fibrillation": 15,
            "depression": 10,
            "sleep_apnea": 10,
            
            # Lifestyle
            "physical_inactivity": 10,
            "social_isolation": 10,
            "cognitive_inactivity": 10,
            
            # Genetic/Family
            "family_history_dementia": 10,
            "apoe4_carrier": 15,
            
            # Current function
            "icope_cognitive_low": 20,
            "memory_complaints": 10,
            "functional_decline": 15,
        }
        
        # Sleep disorder risk weights
        self.sleep_weights = {
            # Symptoms
            "loud_snoring": 15,
            "witnessed_apneas": 20,
            "daytime_sleepiness": 15,
            "insomnia_chronic": 10,
            
            # Risk factors
            "obesity": 15,
            "male_gender": 10,
            "neck_circumference_large": 10,
            "menopause": 10,
            
            # Medical associations
            "hypertension": 10,
            "atrial_fibrillation": 15,
            "heart_failure": 15,
            "diabetes": 10,
            "stroke_history": 10,
            
            # Sleep metrics
            "low_sleep_efficiency": 10,
            "frequent_awakenings": 10,
            "short_sleep_duration": 10,
        }
    
    def calculate_fall_risk(
        self,
        medical_profile: MedicalProfile,
        adl_summary: Optional[ADLSummary] = None,
        icope: Optional[ICOPEAssessment] = None,
    ) -> Tuple[float, List[RiskFactor]]:
        """
        Calculate fall risk score (0-100).
        
        Algorithm based on Tinetti, Morse Fall Scale, and AGS guidelines.
        
        Returns:
            Tuple of (risk_score, risk_factors)
        """
        score = 0.0
        factors = []
        conditions_lower = [c.lower() for c in medical_profile.chronic_conditions]
        meds_lower = [m.name.lower() for m in medical_profile.medications]
        
        # History of falls
        if "fall within last year" in conditions_lower or "recurrent falls" in conditions_lower:
            score += self.fall_weights["history_of_falls"]
            factors.append(RiskFactor(
                factor_name="History of falls",
                category="fall",
                severity=SeverityLevel.HIGH,
                risk_score=self.fall_weights["history_of_falls"],
                weight=2.0,
                weighted_score=self.fall_weights["history_of_falls"] * 2.0,
                evidence_description="Previous fall is strongest predictor of future falls (Tinetti et al.)",
                trend_direction="stable",
            ))
        
        # Mobility status
        if medical_profile.mobility_status in ["assisted", "wheelchair"]:
            score += self.fall_weights["adl_dependence"]
            factors.append(RiskFactor(
                factor_name="Mobility impairment",
                category="fall",
                severity=SeverityLevel.HIGH,
                risk_score=self.fall_weights["adl_dependence"],
                weight=1.5,
                weighted_score=self.fall_weights["adl_dependence"] * 1.5,
                evidence_description="Functional dependence increases fall risk 2-3x",
                trend_direction="stable",
            ))
        
        # ICOPE mobility score
        if icope and icope.locomotor_capacity:
            if icope.locomotor_capacity < 50:
                score += self.fall_weights["gait_abnormality"]
                factors.append(RiskFactor(
                    factor_name="Low ICOPE locomotor score",
                    category="fall",
                    severity=SeverityLevel.HIGH if icope.locomotor_capacity < 40 else SeverityLevel.MODERATE,
                    risk_score=self.fall_weights["gait_abnormality"],
                    weight=1.5,
                    weighted_score=self.fall_weights["gait_abnormality"] * 1.5,
                    contributing_data={"icope_locomotor": icope.locomotor_capacity},
                    evidence_description="ICOPE locomotor score < 50 indicates high fall risk",
                    trend_direction="worsening" if icope.locomotor_trend and icope.locomotor_trend < 0 else "stable",
                ))
        
        # Cognitive impairment
        if medical_profile.cognitive_status in ["mild_impairment", "moderate_impairment"]:
            score += self.fall_weights["cognitive_impairment"]
            factors.append(RiskFactor(
                factor_name="Cognitive impairment",
                category="fall",
                severity=SeverityLevel.HIGH,
                risk_score=self.fall_weights["cognitive_impairment"],
                weight=1.5,
                weighted_score=self.fall_weights["cognitive_impairment"] * 1.5,
                evidence_description="Cognitive impairment doubles fall risk due to impaired judgment",
                trend_direction="stable",
            ))
        
        # Medications
        sedatives = ["lorazepam", "diazepam", "alprazolam", "temazepam", "zolpidem"]
        if any(med in meds_lower for med in sedatives):
            score += self.fall_weights["sedative_use"]
            factors.append(RiskFactor(
                factor_name="Sedative/hypnotic use",
                category="fall",
                severity=SeverityLevel.HIGH,
                risk_score=self.fall_weights["sedative_use"],
                weight=2.0,
                weighted_score=self.fall_weights["sedative_use"] * 2.0,
                evidence_description="Benzodiazepines increase fall risk 44% (AGS Beers Criteria)",
                trend_direction="stable",
            ))
        
        # Diuretics (nocturia risk)
        diuretics = ["furosemide", "hydrochlorothiazide", "chlorthalidone", "torsemide"]
        if any(med in meds_lower for med in diuretics):
            score += self.fall_weights["diuretic_use"]
            factors.append(RiskFactor(
                factor_name="Diuretic use",
                category="fall",
                severity=SeverityLevel.MODERATE,
                risk_score=self.fall_weights["diuretic_use"],
                weight=1.0,
                weighted_score=self.fall_weights["diuretic_use"],
                evidence_description="Diuretics cause nocturia, increasing nighttime fall risk",
                trend_direction="stable",
            ))
        
        # ADL nighttime activity
        if adl_summary:
            if adl_summary.nighttime_bathroom_visits > 2:
                score += 10
                factors.append(RiskFactor(
                    factor_name="Frequent nocturia",
                    category="fall",
                    severity=SeverityLevel.MODERATE,
                    risk_score=10,
                    weight=1.2,
                    weighted_score=12,
                    contributing_data={"nighttime_visits": adl_summary.nighttime_bathroom_visits},
                    evidence_description="Nocturia increases fall risk 1.8x (Nakagawa et al.)",
                    trend_direction="worsening" if adl_summary.nighttime_activity_trend and adl_summary.nighttime_activity_trend > 0 else "stable",
                ))
        
        # Age adjustment
        if medical_profile.age and medical_profile.age > 80:
            score += 5  # Additional age-related risk
        
        return min(score, 100), factors
    
    def calculate_cognitive_decline_risk(
        self,
        medical_profile: MedicalProfile,
        icope: Optional[ICOPEAssessment] = None,
        sleep: Optional[SleepSummary] = None,
    ) -> Tuple[float, List[RiskFactor]]:
        """
        Calculate cognitive decline risk score (0-100).
        
        Based on Lancet Commission modifiable risk factors and ICOPE.
        """
        score = 0.0
        factors = []
        conditions_lower = [c.lower() for c in medical_profile.chronic_conditions]
        
        # Current cognitive status
        if medical_profile.cognitive_status == "mild_impairment":
            score += self.cognitive_weights["mild_cognitive_impairment"]
            factors.append(RiskFactor(
                factor_name="Mild cognitive impairment",
                category="cognitive",
                severity=SeverityLevel.HIGH,
                risk_score=self.cognitive_weights["mild_cognitive_impairment"],
                weight=3.0,
                weighted_score=self.cognitive_weights["mild_cognitive_impairment"] * 3.0,
                evidence_description="MCI progresses to dementia at 10-15% per year",
                trend_direction="worsening",
            ))
        
        # ICOPE cognitive score
        if icope and icope.cognitive_capacity:
            if icope.cognitive_capacity < 60:
                score += self.cognitive_weights["icope_cognitive_low"]
                factors.append(RiskFactor(
                    factor_name="Low ICOPE cognitive score",
                    category="cognitive",
                    severity=SeverityLevel.HIGH,
                    risk_score=self.cognitive_weights["icope_cognitive_low"],
                    weight=2.0,
                    weighted_score=self.cognitive_weights["icope_cognitive_low"] * 2.0,
                    contributing_data={"icope_cognitive": icope.cognitive_capacity},
                    evidence_description="ICOPE cognitive < 60 indicates concerning decline",
                    trend_direction="worsening" if icope.cognitive_trend and icope.cognitive_trend < 0 else "stable",
                ))
        
        # Vascular risk factors
        vascular_conditions = ["diabetes", "hypertension", "cardiovascular disease", "stroke", "atrial fibrillation"]
        for condition in vascular_conditions:
            if condition in conditions_lower:
                weight = self.cognitive_weights.get(condition.replace(" ", "_"), 10)
                score += weight
                factors.append(RiskFactor(
                    factor_name=f"{condition.title()} - vascular risk",
                    category="cognitive",
                    severity=SeverityLevel.MODERATE,
                    risk_score=weight,
                    weight=1.0,
                    weighted_score=weight,
                    evidence_description="Vascular risk factors contribute to vascular dementia and mixed pathologies",
                    trend_direction="stable",
                ))
        
        # Sleep apnea
        if sleep and sleep.sleep_apnea_risk == "high":
            score += self.cognitive_weights["sleep_apnea"]
            factors.append(RiskFactor(
                factor_name="Sleep apnea",
                category="cognitive",
                severity=SeverityLevel.HIGH,
                risk_score=self.cognitive_weights["sleep_apnea"],
                weight=1.5,
                weighted_score=self.cognitive_weights["sleep_apnea"] * 1.5,
                evidence_description="Sleep apnea causes intermittent hypoxia and cognitive decline",
                trend_direction="stable",
            ))
        
        # Age
        if medical_profile.age and medical_profile.age > 85:
            score += self.cognitive_weights["age_over_85"]
            factors.append(RiskFactor(
                factor_name="Advanced age",
                category="cognitive",
                severity=SeverityLevel.MODERATE,
                risk_score=self.cognitive_weights["age_over_85"],
                weight=0.5,
                weighted_score=self.cognitive_weights["age_over_85"] * 0.5,
                evidence_description="Age is strongest non-modifiable risk factor",
                trend_direction="stable",
            ))
        
        return min(score, 100), factors
    
    def calculate_sleep_disorder_risk(
        self,
        medical_profile: MedicalProfile,
        sleep: Optional[SleepSummary] = None,
    ) -> Tuple[float, List[RiskFactor]]:
        """Calculate sleep disorder risk score (0-100)."""
        score = 0.0
        factors = []
        conditions_lower = [c.lower() for c in medical_profile.chronic_conditions]
        
        if not sleep:
            return score, factors
        
        # Sleep efficiency
        if sleep.sleep_efficiency < 70:
            score += self.sleep_weights["low_sleep_efficiency"]
            factors.append(RiskFactor(
                factor_name="Low sleep efficiency",
                category="sleep",
                severity=SeverityLevel.MODERATE,
                risk_score=self.sleep_weights["low_sleep_efficiency"],
                weight=1.5,
                weighted_score=self.sleep_weights["low_sleep_efficiency"] * 1.5,
                contributing_data={"efficiency": sleep.sleep_efficiency},
                evidence_description="Sleep efficiency < 70% indicates sleep disorder",
                trend_direction="worsening" if sleep.efficiency_trend and sleep.efficiency_trend < 0 else "stable",
            ))
        
        # Frequent awakenings
        if sleep.awakenings_count and sleep.awakenings_count > 5:
            score += self.sleep_weights["frequent_awakenings"]
            factors.append(RiskFactor(
                factor_name="Frequent nighttime awakenings",
                category="sleep",
                severity=SeverityLevel.MODERATE,
                risk_score=self.sleep_weights["frequent_awakenings"],
                weight=1.5,
                weighted_score=self.sleep_weights["frequent_awakenings"] * 1.5,
                contributing_data={"awakenings": sleep.awakenings_count},
                evidence_description="> 5 awakenings/night suggests sleep maintenance insomnia or OSA",
                trend_direction="stable",
            ))
        
        # Short sleep
        if sleep.total_duration_hours < 6:
            score += self.sleep_weights["short_sleep_duration"]
            factors.append(RiskFactor(
                factor_name="Short sleep duration",
                category="sleep",
                severity=SeverityLevel.MODERATE,
                risk_score=self.sleep_weights["short_sleep_duration"],
                weight=1.0,
                weighted_score=self.sleep_weights["short_sleep_duration"],
                contributing_data={"duration": sleep.total_duration_hours},
                evidence_description="Sleep < 6 hours associated with health risks",
                trend_direction="stable",
            ))
        
        # Medical associations
        if "hypertension" in conditions_lower:
            score += self.sleep_weights["hypertension"]
            factors.append(RiskFactor(
                factor_name="Hypertension",
                category="sleep",
                severity=SeverityLevel.MODERATE,
                risk_score=self.sleep_weights["hypertension"],
                weight=1.0,
                weighted_score=self.sleep_weights["hypertension"],
                evidence_description="Hypertension strongly associated with sleep apnea",
                trend_direction="stable",
            ))
        
        return min(score, 100), factors
    
    def assess_comprehensive_risk(
        self,
        medical_profile: MedicalProfile,
        adl_summary: Optional[ADLSummary] = None,
        icope: Optional[ICOPEAssessment] = None,
        sleep: Optional[SleepSummary] = None,
    ) -> RiskAssessment:
        """
        Perform comprehensive multi-domain risk assessment.
        
        Args:
            medical_profile: Medical history and conditions
            adl_summary: ADL activity summary
            icope: ICOPE assessment data
            sleep: Sleep analysis summary
        
        Returns:
            Complete RiskAssessment object
        """
        # Calculate domain-specific risks
        fall_score, fall_factors = self.calculate_fall_risk(
            medical_profile, adl_summary, icope
        )
        
        cognitive_score, cognitive_factors = self.calculate_cognitive_decline_risk(
            medical_profile, icope, sleep
        )
        
        sleep_score, sleep_factors = self.calculate_sleep_disorder_risk(
            medical_profile, sleep
        )
        
        # Medication risk (from drug interaction DB)
        from chatbot.knowledge_base.drug_interactions import get_drug_interaction_db
        drug_db = get_drug_interaction_db()
        med_risk = drug_db.assess_medication_risk(
            [m.name for m in medical_profile.medications],
            medical_profile.chronic_conditions,
            medical_profile.age or 75
        )
        med_score = min(med_risk["risk_score"] * 2, 100)  # Scale to 0-100
        
        # Create medication risk factors
        med_factors = []
        if med_risk["interactions"]:
            med_factors.append(RiskFactor(
                factor_name=f"Drug interactions (n={len(med_risk['interactions'])})",
                category="medication",
                severity=SeverityLevel.HIGH if any(i["severity"] == "contraindicated" for i in med_risk["interactions"]) else SeverityLevel.MODERATE,
                risk_score=len(med_risk["interactions"]) * 10,
                weight=2.0,
                weighted_score=len(med_risk["interactions"]) * 20,
                evidence_description="Drug interactions increase adverse event risk",
                trend_direction="stable",
            ))
        
        if med_risk["anticholinergic_burden"]["score"] >= 3:
            med_factors.append(RiskFactor(
                factor_name="High anticholinergic burden",
                category="medication",
                severity=SeverityLevel.HIGH,
                risk_score=20,
                weight=2.0,
                weighted_score=40,
                contributing_data=med_risk["anticholinergic_burden"],
                evidence_description="ACB >= 3 associated with cognitive decline and falls",
                trend_direction="stable",
            ))
        
        # Combine all factors
        all_factors = fall_factors + cognitive_factors + sleep_factors + med_factors
        
        # Sort by weighted score
        all_factors.sort(key=lambda x: x.weighted_score, reverse=True)
        
        # Calculate overall risk (weighted average of domains)
        domain_scores = {
            "fall": fall_score,
            "cognitive": cognitive_score,
            "sleep": sleep_score,
            "medication": med_score,
        }
        
        # Weight domains by clinical importance
        domain_weights = {
            "fall": 1.2,
            "cognitive": 1.3,
            "sleep": 0.8,
            "medication": 1.1,
        }
        
        weighted_sum = sum(
            domain_scores[d] * domain_weights[d] 
            for d in domain_scores 
            if domain_scores[d] > 0
        )
        weight_sum = sum(
            domain_weights[d] 
            for d in domain_scores 
            if domain_scores[d] > 0
        )
        
        overall_score = weighted_sum / weight_sum if weight_sum > 0 else 0
        
        # Determine risk level
        if overall_score >= self.RISK_THRESHOLDS["critical"]:
            overall_level = SeverityLevel.CRITICAL
        elif overall_score >= self.RISK_THRESHOLDS["high"]:
            overall_level = SeverityLevel.HIGH
        elif overall_score >= self.RISK_THRESHOLDS["moderate"]:
            overall_level = SeverityLevel.MODERATE
        elif overall_score >= self.RISK_THRESHOLDS["low"]:
            overall_level = SeverityLevel.LOW
        else:
            overall_level = SeverityLevel.MINIMAL
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            domain_scores, all_factors
        )
        
        # Critical alerts
        critical_alerts = [
            f.factor_name for f in all_factors 
            if f.severity == SeverityLevel.CRITICAL or 
            (f.severity == SeverityLevel.HIGH and f.weighted_score > 30)
        ]
        
        return RiskAssessment(
            assessment_timestamp=datetime.now(),
            fall_risk=fall_score,
            cognitive_decline_risk=cognitive_score,
            sleep_disorder_risk=sleep_score,
            medication_risk=med_score,
            overall_risk_score=round(overall_score, 1),
            overall_risk_level=overall_level,
            risk_factors=all_factors,
            top_risk_factors=all_factors[:5],
            critical_alerts=critical_alerts,
            recommendations=recommendations,
        )
    
    def _generate_recommendations(
        self,
        domain_scores: Dict[str, float],
        factors: List[RiskFactor]
    ) -> List[str]:
        """Generate high-level recommendations based on risks."""
        recommendations = []
        
        # Domain-specific recommendations
        if domain_scores.get("fall", 0) > 60:
            recommendations.append("Urgent: Comprehensive fall risk assessment and intervention")
        elif domain_scores.get("fall", 0) > 40:
            recommendations.append("Recommended: Fall prevention program including exercise")
        
        if domain_scores.get("cognitive", 0) > 60:
            recommendations.append("Urgent: Comprehensive cognitive evaluation")
        elif domain_scores.get("cognitive", 0) > 40:
            recommendations.append("Recommended: Cognitive stimulation and vascular risk management")
        
        if domain_scores.get("medication", 0) > 50:
            recommendations.append("Urgent: Medication review for deprescribing")
        
        if domain_scores.get("sleep", 0) > 50:
            recommendations.append("Recommended: Sleep study evaluation for possible sleep apnea")
        
        return recommendations


# Singleton
_risk_stratifier: Optional[RiskStratifier] = None


def get_risk_stratifier() -> RiskStratifier:
    """Get or create singleton risk stratifier."""
    global _risk_stratifier
    if _risk_stratifier is None:
        _risk_stratifier = RiskStratifier()
    return _risk_stratifier
