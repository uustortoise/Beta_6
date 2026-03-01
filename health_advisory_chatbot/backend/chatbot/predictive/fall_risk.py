"""
Fall Risk Calculator

Evidence-based fall risk assessment using validated instruments:
- Tinetti Balance and Gait Evaluation
- Morse Fall Scale elements
- Hendrich II Fall Risk Model elements
- Modified for integration with ADL/sensor data
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class GaitSpeedCategory(str, Enum):
    """Gait speed categories based on prior research."""
    NORMAL = "normal"  # > 1.0 m/s
    SLOW = "slow"  # 0.6-1.0 m/s
    VERY_SLOW = "very_slow"  # < 0.6 m/s


@dataclass
class TinettiScore:
    """Tinetti Balance and Gait Evaluation results."""
    balance_score: int  # 0-16
    gait_score: int  # 0-12
    total_score: int  # 0-28
    
    # Interpretation
    risk_category: str  # high/moderate/low
    
    @property
    def interpretation(self) -> str:
        """Get clinical interpretation."""
        if self.total_score <= 18:
            return "High fall risk (> 2x risk)"
        elif self.total_score <= 23:
            return "Moderate fall risk"
        else:
            return "Low fall risk"


class FallRiskCalculator:
    """
    Comprehensive fall risk calculator.
    
    Combines elements from:
    - Tinetti Performance Oriented Mobility Assessment (POMA)
    - Morse Fall Scale
    - Hendrich II Fall Risk Model
    - ADL-derived indicators
    """
    
    # Tinetti scoring thresholds
    TINETTI_HIGH_RISK = 18
    TINETTI_MODERATE_RISK = 23
    
    def __init__(self):
        """Initialize fall risk calculator."""
        pass
    
    def calculate_tinetti(
        self,
        # Balance items (0-16)
        sitting_balance: int = 1,  # 0-1
        rise_from_chair: int = 1,  # 0-2
        immediate_standing_balance: int = 1,  # 0-2
        standing_balance: int = 1,  # 0-2
        balance_with_eyes_closed: int = 1,  # 0-1
        turning_360_degrees: int = 1,  # 0-2
        sitting_down: int = 1,  # 0-2
        
        # Gait items (0-12)
        gait_initiation: int = 1,  # 0-1
        step_length_height: int = 1,  # 0-2
        step_symmetry: int = 1,  # 0-2
        step_continuity: int = 1,  # 0-2
        path_deviation: int = 1,  # 0-2
        trunk_sway: int = 1,  # 0-2
        walking_stance: int = 1,  # 0-2
    ) -> TinettiScore:
        """
        Calculate Tinetti Balance and Gait score.
        
        Args:
            Balance items (0-16 total):
                - sitting_balance: 0=leans/slides, 1=steady
                - rise_from_chair: 0=unable, 1=with arms, 2=without arms
                - immediate_standing_balance: 0=unsteady, 1=steady but uses walker, 2=steady without walker
                - standing_balance: 0=unsteady, 1=steady but wide/narrow, 2=narrow stance
                - balance_with_eyes_closed: 0=unsteady, 1=steady
                - turning_360_degrees: 0=discontinuous, 1=continuous, 2=continuous/graceful
                - sitting_down: 0=unsafe, 1=uses arms/uneven, 2=safe/smooth
            
            Gait items (0-12 total):
                - gait_initiation: 0=hesitant/multiple attempts, 1=immediate
                - step_length_height: 0=one foot passes other, 1=foot passes other, 2=full step
                - step_symmetry: 0=unequal, 1=equal
                - step_continuity: 0=stopping/discontinuous, 1=continuous
                - path_deviation: 0=marked deviation, 1=mild/uses aid, 2=straight/no aid
                - trunk_sway: 0=marked sway/stagger, 1=no sway but knees/trunk flexed, 2=no sway/flexion
                - walking_stance: 0=heels apart, 1=heels almost touch while walking
        
        Returns:
            TinettiScore with interpretation
        """
        balance_score = (
            sitting_balance + rise_from_chair + immediate_standing_balance +
            standing_balance + balance_with_eyes_closed + turning_360_degrees +
            sitting_down
        )
        
        gait_score = (
            gait_initiation + step_length_height + step_symmetry +
            step_continuity + path_deviation + trunk_sway + walking_stance
        )
        
        total_score = balance_score + gait_score
        
        # Determine risk category
        if total_score <= self.TINETTI_HIGH_RISK:
            risk_category = "high"
        elif total_score <= self.TINETTI_MODERATE_RISK:
            risk_category = "moderate"
        else:
            risk_category = "low"
        
        return TinettiScore(
            balance_score=balance_score,
            gait_score=gait_score,
            total_score=total_score,
            risk_category=risk_category
        )
    
    def estimate_tinetti_from_adl(
        self,
        gait_speed: Optional[float] = None,
        nighttime_bathroom_visits: int = 0,
        average_transition_time: Optional[float] = None,
        mobility_aid_use: bool = False,
        fall_history: bool = False,
    ) -> TinettiScore:
        """
        Estimate Tinetti score from ADL-derived indicators.
        
        This is an estimation when direct Tinetti assessment is not available.
        Uses sensor-derived gait speed and activity patterns.
        
        Args:
            gait_speed: Usual gait speed in m/s
            nighttime_bathroom_visits: Average per night
            average_transition_time: Time to transition rooms (seconds)
            mobility_aid_use: Uses walker/cane
            fall_history: History of falls
        
        Returns:
            Estimated TinettiScore
        """
        # Start with neutral scores
        balance_score = 8  # Middle of 0-16
        gait_score = 6  # Middle of 0-12
        
        # Adjust based on gait speed
        if gait_speed:
            if gait_speed < 0.6:
                # Very slow - high risk
                balance_score -= 4
                gait_score -= 4
            elif gait_speed < 1.0:
                # Slow - moderate risk
                balance_score -= 2
                gait_score -= 2
            elif gait_speed > 1.2:
                # Fast - low risk
                balance_score += 2
                gait_score += 2
        
        # Adjust for nighttime activity (nocturia proxy)
        if nighttime_bathroom_visits > 3:
            balance_score -= 2  # More unsteady at night
        
        # Adjust for transition time (mobility indicator)
        if average_transition_time:
            if average_transition_time > 10:
                balance_score -= 3
                gait_score -= 3
            elif average_transition_time > 5:
                balance_score -= 1
                gait_score -= 1
        
        # Mobility aid use
        if mobility_aid_use:
            balance_score -= 2
            gait_score -= 1
        
        # Fall history (strong predictor)
        if fall_history:
            balance_score -= 3
            gait_score -= 2
        
        # Ensure scores are within bounds
        balance_score = max(0, min(16, balance_score))
        gait_score = max(0, min(12, gait_score))
        total_score = balance_score + gait_score
        
        # Determine risk
        if total_score <= self.TINETTI_HIGH_RISK:
            risk_category = "high"
        elif total_score <= self.TINETTI_MODERATE_RISK:
            risk_category = "moderate"
        else:
            risk_category = "low"
        
        return TinettiScore(
            balance_score=balance_score,
            gait_score=gait_score,
            total_score=total_score,
            risk_category=risk_category
        )
    
    def calculate_morse_score(
        self,
        fall_history: bool,
        secondary_diagnosis: bool,
        ambulatory_aid: str,  # none/furniture/cane/crutch/walker
        iv_heparin_lock: bool,
        gait_transferring: str,  # normal/weak/impaired/bedrest
        mental_status: str,  # oriented/forgets
    ) -> Dict:
        """
        Calculate Morse Fall Scale score.
        
        Standardized fall risk assessment used in hospital settings.
        
        Args:
            fall_history: Fall within last 3 months
            secondary_diagnosis: > 1 medical diagnosis
            ambulatory_aid: Type of ambulatory aid used
            iv_heparin_lock: Has IV or heparin lock
            gait_transferring: Gait/transferring status
            mental_status: Mental status
        
        Returns:
            Dictionary with score and risk level
        """
        score = 0
        
        # History of falling (25 points)
        if fall_history:
            score += 25
        
        # Secondary diagnosis (15 points)
        if secondary_diagnosis:
            score += 15
        
        # Ambulatory aid (0-30 points)
        aid_scores = {
            "none": 0,
            "furniture": 30,
            "cane": 15,
            "crutch": 15,
            "walker": 15,
        }
        score += aid_scores.get(ambulatory_aid, 0)
        
        # IV/heparin lock (20 points)
        if iv_heparin_lock:
            score += 20
        
        # Gait/transferring (0-20 points)
        gait_scores = {
            "normal": 0,
            "weak": 10,
            "impaired": 20,
            "bedrest": 0,
        }
        score += gait_scores.get(gait_transferring, 0)
        
        # Mental status (0-15 points)
        mental_scores = {
            "oriented": 0,
            "forgets": 15,
        }
        score += mental_scores.get(mental_status, 0)
        
        # Risk level
        if score >= 50:
            risk_level = "high"
        elif score >= 25:
            risk_level = "moderate"
        else:
            risk_level = "low"
        
        return {
            "score": score,
            "risk_level": risk_level,
            "max_score": 125,
            "interpretation": f"{risk_level.title()} fall risk",
        }
    
    def classify_gait_speed(self, speed_m_per_s: float) -> GaitSpeedCategory:
        """
        Classify gait speed into risk categories.
        
        Based on prior research:
        - < 0.6 m/s: Severe impairment, high fall risk
        - 0.6-1.0 m/s: Moderate impairment
        - > 1.0 m/s: Normal
        
        Args:
            speed_m_per_s: Gait speed in meters per second
        
        Returns:
            GaitSpeedCategory
        """
        if speed_m_per_s < 0.6:
            return GaitSpeedCategory.VERY_SLOW
        elif speed_m_per_s < 1.0:
            return GaitSpeedCategory.SLOW
        else:
            return GaitSpeedCategory.NORMAL
    
    def predict_fall_probability(
        self,
        tinetti_score: Optional[TinettiScore] = None,
        gait_speed: Optional[float] = None,
        fall_history: bool = False,
        medications_count: int = 0,
        cognitive_impairment: bool = False,
    ) -> Dict:
        """
        Predict 30-day fall probability using multiple factors.
        
        Uses logistic regression-inspired algorithm based on
        Tinetti et al. and subsequent validation studies.
        
        Returns:
            Probability and confidence interval
        """
        # Base probability
        logit = -4.0  # Baseline (low risk)
        
        # Tinetti score contribution
        if tinetti_score:
            if tinetti_score.total_score <= 18:
                logit += 2.5
            elif tinetti_score.total_score <= 23:
                logit += 1.2
            else:
                logit -= 1.0
        
        # Gait speed contribution
        if gait_speed:
            if gait_speed < 0.6:
                logit += 1.8
            elif gait_speed < 1.0:
                logit += 0.8
            else:
                logit -= 0.5
        
        # Fall history (strongest predictor)
        if fall_history:
            logit += 1.5
        
        # Medications
        if medications_count >= 10:
            logit += 1.0
        elif medications_count >= 5:
            logit += 0.5
        
        # Cognitive impairment
        if cognitive_impairment:
            logit += 0.8
        
        # Calculate probability
        probability = 1 / (1 + 2.71828 ** (-logit))
        
        # Confidence interval (simplified)
        ci_lower = max(0, probability - 0.15)
        ci_upper = min(1, probability + 0.15)
        
        # Risk category
        if probability >= 0.5:
            risk = "very_high"
        elif probability >= 0.3:
            risk = "high"
        elif probability >= 0.15:
            risk = "moderate"
        else:
            risk = "low"
        
        return {
            "probability_30day": round(probability, 3),
            "confidence_interval": (round(ci_lower, 3), round(ci_upper, 3)),
            "risk_category": risk,
            "interpretation": f"{probability*100:.1f}% probability of fall within 30 days",
        }


# Singleton
_fall_calculator: Optional[FallRiskCalculator] = None


def get_fall_risk_calculator() -> FallRiskCalculator:
    """Get or create singleton fall risk calculator."""
    global _fall_calculator
    if _fall_calculator is None:
        _fall_calculator = FallRiskCalculator()
    return _fall_calculator
