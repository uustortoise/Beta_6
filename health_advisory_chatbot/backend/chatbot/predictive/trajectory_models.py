"""
Health Trajectory Predictor

Predicts future health status based on current data and trends.

Domains:
- Cognitive trajectory (normal -> MCI -> dementia)
- Mobility trajectory (independent -> assisted -> dependent)
- Frailty progression
- Functional decline
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

from models.schemas import (
    TrajectoryPrediction,
    MedicalProfile,
    ADLSummary,
    ICOPEAssessment,
)
from .outcome_statistics import get_outcome_statistics


class TrajectoryTrend(str, Enum):
    """Health trajectory trend directions."""
    IMPROVING = "improving"
    STABLE = "stable"
    SLOW_DECLINE = "slow_decline"
    RAPID_DECLINE = "rapid_decline"


class TrajectoryPredictor:
    """
    Predicts health trajectories using trend analysis and clinical models.
    
    Provides forward-looking risk assessment for:
    - Cognitive decline
    - Mobility loss
    - Frailty progression
    - Functional dependence
    """
    
    def __init__(self):
        """Initialize trajectory predictor."""
        self.outcome_stats = get_outcome_statistics()
    
    def predict_cognitive_trajectory(
        self,
        medical_profile: MedicalProfile,
        icope_history: List[ICOPEAssessment],
        prediction_months: int = 12,
    ) -> TrajectoryPrediction:
        """
        Predict cognitive trajectory over time.
        
        Uses:
        - Current cognitive status
        - Rate of decline (from ICOPE history)
        - Vascular risk factors
        - Age
        
        Args:
            medical_profile: Medical history
            icope_history: Historical ICOPE assessments
            prediction_months: Prediction horizon
        
        Returns:
            TrajectoryPrediction
        """
        # Determine current status
        current_status = self._classify_cognitive_status(
            medical_profile.cognitive_status,
            icope_history[-1] if icope_history else None
        )
        
        # Calculate rate of decline
        decline_rate = self._calculate_decline_rate(icope_history)
        
        # Risk factors
        risk_factors = self._identify_cognitive_risk_factors(medical_profile)
        protective_factors = self._identify_protective_factors(medical_profile)
        
        # Calculate transition probabilities
        status_order = ["normal", "mild_impairment", "moderate_impairment", "severe_dementia"]
        current_idx = status_order.index(current_status) if current_status in status_order else 0
        
        # Base transition probability (annual)
        base_transition_prob = {
            "normal": 0.05,           # 5% to MCI
            "mild_impairment": 0.10,  # 10% to moderate
            "moderate_impairment": 0.20,  # 20% to severe
            "severe_dementia": 0.30,  # 30% further decline
        }.get(current_status, 0.05)
        
        # Adjust for decline rate
        if decline_rate > 5:  # Rapid decline (>5 points/year)
            base_transition_prob *= 2.0
        elif decline_rate > 2:  # Moderate decline
            base_transition_prob *= 1.5
        
        # Adjust for risk factors
        risk_multiplier = 1.0 + (len(risk_factors) * 0.1)
        protective_multiplier = max(0.5, 1.0 - (len(protective_factors) * 0.1))
        
        adjusted_prob = base_transition_prob * risk_multiplier * protective_multiplier
        
        # Scale to prediction period
        month_factor = prediction_months / 12
        transition_prob = min(0.95, adjusted_prob * month_factor)
        
        # Determine predicted status
        if transition_prob > 0.6 and current_idx < len(status_order) - 1:
            predicted_status = status_order[current_idx + 1]
        elif transition_prob > 0.3 and current_idx < len(status_order) - 1:
            # Probabilistic - could stay or progress
            predicted_status = f"likely_{status_order[current_idx + 1]}"
        else:
            predicted_status = current_status
        
        # Confidence calculation
        confidence = self._calculate_confidence(
            data_points=len(icope_history),
            prediction_months=prediction_months,
            has_risk_factors=len(risk_factors) > 0
        )
        
        # Integrate Outcome Statistics
        # If patient has diabetes, increase risk/decline rate based on stats
        risk_multiplier = 1.0
        if "Diabetes" in risk_factors:
            risk_multiplier *= 1.2 # Derived from "1.5-2x risk" in stats
            
        final_probability = min(transition_prob * risk_multiplier, 0.95)

        return TrajectoryPrediction(
            domain="cognitive_decline",
            current_status=current_status,
            prediction_horizon=f"{prediction_months} months",
            predicted_status=predicted_status,
            confidence=confidence,
            probability_distribution={
                "stable": round(1 - final_probability, 2),
                "decline": round(final_probability, 2),
            },
            key_drivers=risk_factors[:3],
            protective_factors=protective_factors[:3],
            model_version="1.0",
            features_used=["icope_cognitive", "age", "vascular_risk", "education"],
        )
    
    def predict_mobility_trajectory(
        self,
        medical_profile: MedicalProfile,
        adl_history: List[ADLSummary],
        icope_history: List[ICOPEAssessment],
        prediction_months: int = 12,
    ) -> TrajectoryPrediction:
        """
        Predict mobility/falls trajectory.
        
        Args:
            medical_profile: Medical history
            adl_history: Historical ADL data
            icope_history: Historical ICOPE assessments
            prediction_months: Prediction horizon
        
        Returns:
            TrajectoryPrediction
        """
        current_status = medical_profile.mobility_status or "independent"
        
        # Analyze trend
        trend = self._analyze_mobility_trend(adl_history, icope_history)
        
        # Risk factors
        risk_factors = []
        if medical_profile.age and medical_profile.age > 80:
            risk_factors.append("Advanced age")
        if "stroke" in [c.lower() for c in medical_profile.chronic_conditions]:
            risk_factors.append("History of stroke")
        if "parkinson" in " ".join(medical_profile.chronic_conditions).lower():
            risk_factors.append("Parkinson's disease")
        
        # Predict based on trend and risk
        status_progression = {
            "independent": "assisted",
            "assisted": "wheelchair",
            "wheelchair": "bedridden",
            "bedridden": "bedridden",
        }
        
        if trend == TrajectoryTrend.RAPID_DECLINE:
            decline_prob = 0.7
            predicted_status = status_progression.get(current_status, current_status)
        elif trend == TrajectoryTrend.SLOW_DECLINE:
            decline_prob = 0.4
            predicted_status = f"possible_{status_progression.get(current_status, current_status)}"
        elif trend == TrajectoryTrend.IMPROVING:
            decline_prob = 0.1
            predicted_status = current_status
        else:
            decline_prob = 0.2
            predicted_status = current_status
        
        confidence = self._calculate_confidence(
            data_points=len(adl_history),
            prediction_months=prediction_months,
            has_risk_factors=len(risk_factors) > 0
        )
        
        # Integrate Outcome Statistics for Falls
        stat_risk_modifier = 0.0
        if medical_profile.has_condition("dementia"):
             stat_risk_modifier += 0.2 # 2-3x higher risk
        if medical_profile.has_condition("diabetes"):
             stat_risk_modifier += 0.1
             
        final_decline_prob = min(decline_prob + stat_risk_modifier, 0.95)

        return TrajectoryPrediction(
            domain="mobility_loss",
            current_status=current_status,
            prediction_horizon=f"{prediction_months} months",
            predicted_status=predicted_status,
            confidence=confidence,
            probability_distribution={
                "maintain": round(1 - final_decline_prob, 2),
                "decline": round(final_decline_prob, 2),
            },
            key_drivers=risk_factors,
            protective_factors=["Exercise program", "Physical therapy"] if trend != TrajectoryTrend.RAPID_DECLINE else [],
            model_version="1.0",
            features_used=["mobility_status", "adl_trend", "age", "neurological_conditions"],
        )
    
    def predict_sleep_trajectory(
        self,
        sleep_history: List[Any],  # SleepSummary
        medical_profile: MedicalProfile,
        prediction_months: int = 6,
    ) -> TrajectoryPrediction:
        """
        Predict sleep quality trajectory.
        
        Args:
            sleep_history: Historical sleep data
            medical_profile: Medical history
            prediction_months: Prediction horizon
        
        Returns:
            TrajectoryPrediction
        """
        if not sleep_history:
            return TrajectoryPrediction(
                domain="sleep_quality",
                current_status="unknown",
                prediction_horizon=f"{prediction_months} months",
                predicted_status="unknown",
                confidence=0.0,
                probability_distribution={},
                key_drivers=[],
                protective_factors=[],
                model_version="1.0",
                features_used=[],
            )
        
        # Analyze sleep trend
        recent_sleep = sleep_history[-1]
        
        if recent_sleep.sleep_efficiency > 85:
            current_status = "good"
        elif recent_sleep.sleep_efficiency > 70:
            current_status = "fair"
        else:
            current_status = "poor"
        
        # Trend analysis
        if len(sleep_history) >= 2:
            efficiency_trend = recent_sleep.sleep_efficiency - sleep_history[0].sleep_efficiency
            if efficiency_trend < -10:
                trend = TrajectoryTrend.RAPID_DECLINE
            elif efficiency_trend < -5:
                trend = TrajectoryTrend.SLOW_DECLINE
            elif efficiency_trend > 5:
                trend = TrajectoryTrend.IMPROVING
            else:
                trend = TrajectoryTrend.STABLE
        else:
            trend = TrajectoryTrend.STABLE
        
        # Predict
        if trend == TrajectoryTrend.RAPID_DECLINE:
            predicted_status = "worsening"
            worsening_prob = 0.7
        elif trend == TrajectoryTrend.SLOW_DECLINE:
            predicted_status = "likely_worsening"
            worsening_prob = 0.5
        elif trend == TrajectoryTrend.IMPROVING:
            predicted_status = "improving"
            worsening_prob = 0.1
        else:
            predicted_status = "stable"
            worsening_prob = 0.3
        
        return TrajectoryPrediction(
            domain="sleep_quality",
            current_status=current_status,
            prediction_horizon=f"{prediction_months} months",
            predicted_status=predicted_status,
            confidence=60.0,
            probability_distribution={
                "maintain_improve": round(1 - worsening_prob, 2),
                "worsen": round(worsening_prob, 2),
            },
            key_drivers=["Current sleep efficiency", "Trend direction"],
            protective_factors=["Sleep hygiene", "CBT-I if indicated"],
            model_version="1.0",
            features_used=["sleep_efficiency", "sleep_duration", "awakenings"],
        )
    
    def _classify_cognitive_status(
        self,
        cognitive_status: Optional[str],
        icope: Optional[ICOPEAssessment]
    ) -> str:
        """Classify current cognitive status."""
        if cognitive_status:
            status_map = {
                "normal": "normal",
                "mild_impairment": "mild_impairment",
                "moderate_impairment": "moderate_impairment",
                "severe": "severe_dementia",
            }
            return status_map.get(cognitive_status, "normal")
        
        if icope and icope.cognitive_capacity:
            if icope.cognitive_capacity < 40:
                return "severe_dementia"
            elif icope.cognitive_capacity < 60:
                return "moderate_impairment"
            elif icope.cognitive_capacity < 75:
                return "mild_impairment"
            else:
                return "normal"
        
        return "normal"
    
    def _calculate_decline_rate(
        self,
        icope_history: List[ICOPEAssessment]
    ) -> float:
        """Calculate annual rate of cognitive decline from ICOPE history."""
        if len(icope_history) < 2:
            return 0.0
        
        # Get first and last assessments
        first = icope_history[0]
        last = icope_history[-1]
        
        # Calculate time difference in years
        time_diff_days = (last.assessment_date - first.assessment_date).days
        time_diff_years = time_diff_days / 365.25
        
        if time_diff_years < 0.1:  # Less than ~1 month
            return 0.0
        
        # Calculate score change
        if first.cognitive_capacity and last.cognitive_capacity:
            score_change = first.cognitive_capacity - last.cognitive_capacity
            annual_decline = score_change / time_diff_years
            return annual_decline
        
        return 0.0
    
    def _identify_cognitive_risk_factors(
        self,
        medical_profile: MedicalProfile
    ) -> List[str]:
        """Identify risk factors for cognitive decline."""
        risk_factors = []
        conditions = [c.lower() for c in medical_profile.chronic_conditions]
        
        risk_conditions = [
            ("diabetes", "Diabetes"),
            ("hypertension", "Hypertension"),
            ("stroke", "Stroke history"),
            ("atrial fibrillation", "Atrial fibrillation"),
            ("cardiovascular disease", "Cardiovascular disease"),
            ("depression", "Depression"),
            ("sleep apnea", "Sleep apnea"),
        ]
        
        for condition_key, label in risk_conditions:
            if any(condition_key in c for c in conditions):
                risk_factors.append(label)
        
        if medical_profile.age and medical_profile.age > 85:
            risk_factors.append("Advanced age")
        
        return risk_factors
    
    def _identify_protective_factors(
        self,
        medical_profile: MedicalProfile
    ) -> List[str]:
        """Identify protective factors against cognitive decline."""
        protective = []
        
        # Physical activity (inferred from mobility status)
        if medical_profile.mobility_status == "independent":
            protective.append("Physical activity")
        
        # Social engagement (would need additional data)
        
        return protective
    
    def _analyze_mobility_trend(
        self,
        adl_history: List[ADLSummary],
        icope_history: List[ICOPEAssessment]
    ) -> TrajectoryTrend:
        """Analyze mobility trend from historical data."""
        if not icope_history or len(icope_history) < 2:
            return TrajectoryTrend.STABLE
        
        # Check ICOPE locomotor trend
        recent_icope = icope_history[-1]
        if recent_icope.locomotor_trend:
            if recent_icope.locomotor_trend < -5:
                return TrajectoryTrend.RAPID_DECLINE
            elif recent_icope.locomotor_trend < -2:
                return TrajectoryTrend.SLOW_DECLINE
            elif recent_icope.locomotor_trend > 2:
                return TrajectoryTrend.IMPROVING
        
        # Check ADL trend
        if adl_history and len(adl_history) >= 2:
            recent_adl = adl_history[-1]
            if recent_adl.activity_trend:
                if recent_adl.activity_trend < -20:
                    return TrajectoryTrend.RAPID_DECLINE
                elif recent_adl.activity_trend < -10:
                    return TrajectoryTrend.SLOW_DECLINE
                elif recent_adl.activity_trend > 10:
                    return TrajectoryTrend.IMPROVING
        
        return TrajectoryTrend.STABLE
    
    def _calculate_confidence(
        self,
        data_points: int,
        prediction_months: int,
        has_risk_factors: bool
    ) -> float:
        """Calculate confidence score for prediction."""
        confidence = 50.0  # Base confidence
        
        # More data = higher confidence
        if data_points >= 5:
            confidence += 20
        elif data_points >= 3:
            confidence += 10
        elif data_points >= 1:
            confidence += 5
        
        # Shorter predictions more confident
        if prediction_months <= 6:
            confidence += 10
        elif prediction_months <= 12:
            confidence += 0
        else:
            confidence -= 10
        
        # Risk factors improve model accuracy
        if has_risk_factors:
            confidence += 5
        
        return min(95, max(20, confidence))


# Singleton
_trajectory_predictor: Optional[TrajectoryPredictor] = None


def get_trajectory_predictor() -> TrajectoryPredictor:
    """Get or create singleton trajectory predictor."""
    global _trajectory_predictor
    if _trajectory_predictor is None:
        _trajectory_predictor = TrajectoryPredictor()
    return _trajectory_predictor
