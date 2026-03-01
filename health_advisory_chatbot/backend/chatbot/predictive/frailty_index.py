"""
Frailty Index Calculator

Implementation of the Fried Frailty Phenotype and Frailty Index.

Reference:
Fried LP, et al. Frailty in older adults: evidence for a phenotype.
J Gerontol A Biol Sci Med Sci. 2001;56(3):M146-M156.
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class FrailtyStatus(str, Enum):
    """Fried frailty phenotype categories."""
    ROBUST = "robust"
    PRE_FRAIL = "pre_frail"
    FRAIL = "frail"


@dataclass
class FriedPhenotype:
    """
    Fried Frailty Phenotype assessment.
    
    Five criteria:
    1. Unintentional weight loss (> 10 lbs in past year)
    2. Weakness (grip strength, lowest 20% by gender/BMI)
    3. Exhaustion (self-reported)
    4. Slow gait speed (lowest 20% by gender/height)
    5. Low physical activity (kcal/week, lowest 20%)
    """
    weight_loss: bool
    weakness: bool
    exhaustion: bool
    slow_walking_speed: bool
    low_physical_activity: bool
    
    criteria_count: int
    frailty_status: FrailtyStatus
    
    @property
    def is_frail(self) -> bool:
        """Check if meets frailty criteria (3+)."""
        return self.criteria_count >= 3
    
    @property
    def is_pre_frail(self) -> bool:
        """Check if pre-frail (1-2 criteria)."""
        return 1 <= self.criteria_count <= 2


class FrailtyIndexCalculator:
    """
    Frailty assessment using Fried Phenotype and deficit accumulation.
    
    Provides:
    - Fried Phenotype classification
    - Deficit accumulation frailty index
    - Frailty trajectory prediction
    """
    
    # Grip strength cutoffs (kg) - Fried et al.
    GRIP_STRENGTH_CUTOFFS = {
        "male": {
            "bmi_low": 29,     # BMI <= 24
            "bmi_mid": 30,     # BMI 24.1-28
            "bmi_high": 32,    # BMI > 28
        },
        "female": {
            "bmi_low": 17,
            "bmi_mid": 17.3,
            "bmi_high": 21,
        }
    }
    
    # Gait speed cutoffs (m/s) - Fried et al.
    GAIT_SPEED_CUTOFFS = {
        "male": {
            "height_low": 5.9,   # Height <= 173 cm
            "height_high": 5.5,  # Height > 173 cm
        },
        "female": {
            "height_low": 5.2,   # Height <= 159 cm
            "height_high": 4.8,  # Height > 159 cm
        }
    }
    
    # Physical activity cutoffs (kcal/week) - Fried et al.
    ACTIVITY_CUTOFFS = {
        "male": 383,
        "female": 270,
    }
    
    def __init__(self):
        """Initialize frailty calculator."""
        pass
    
    def assess_fried_phenotype(
        self,
        # Criterion 1: Weight loss
        unintentional_weight_loss_10lbs: bool = False,
        weight_loss_percentage: Optional[float] = None,
        
        # Criterion 2: Weakness
        grip_strength_kg: Optional[float] = None,
        gender: Optional[str] = None,
        bmi: Optional[float] = None,
        
        # Criterion 3: Exhaustion
        exhaustion_frequency: str = "none",  # none/sometimes/most_days
        
        # Criterion 4: Slow walking
        gait_speed_m_per_s: Optional[float] = None,
        height_cm: Optional[float] = None,
        
        # Criterion 5: Low activity
        physical_activity_kcal_per_week: Optional[float] = None,
        
        # Alternative indicators (when direct measures unavailable)
        adl_activity_level: Optional[str] = None,  # high/moderate/low/very_low
    ) -> FriedPhenotype:
        """
        Assess frailty using Fried Phenotype criteria.
        
        Args:
            unintentional_weight_loss_10lbs: > 10 lbs lost unintentionally
            weight_loss_percentage: Percentage weight loss (alternative)
            grip_strength_kg: Hand grip strength in kg
            gender: "male" or "female"
            bmi: Body mass index
            exhaustion_frequency: Self-reported exhaustion
            gait_speed_m_per_s: Walking speed over 4-6 meters
            height_cm: Height in centimeters
            physical_activity_kcal_per_week: Estimated energy expenditure
            adl_activity_level: Activity level from ADL data
        
        Returns:
            FriedPhenotype assessment
        """
        criteria_met = []
        
        # Criterion 1: Weight loss
        weight_loss = unintentional_weight_loss_10lbs or (
            weight_loss_percentage and weight_loss_percentage >= 5
        )
        if weight_loss:
            criteria_met.append("weight_loss")
        
        # Criterion 2: Weakness
        weakness = False
        if grip_strength_kg and gender and gender.lower() in self.GRIP_STRENGTH_CUTOFFS:
            gender_key = gender.lower()
            
            # Determine BMI category
            if bmi:
                if bmi <= 24:
                    cutoff = self.GRIP_STRENGTH_CUTOFFS[gender_key]["bmi_low"]
                elif bmi <= 28:
                    cutoff = self.GRIP_STRENGTH_CUTOFFS[gender_key]["bmi_mid"]
                else:
                    cutoff = self.GRIP_STRENGTH_CUTOFFS[gender_key]["bmi_high"]
            else:
                cutoff = self.GRIP_STRENGTH_CUTOFFS[gender_key]["bmi_mid"]
            
            weakness = grip_strength_kg < cutoff
        
        if weakness:
            criteria_met.append("weakness")
        
        # Criterion 3: Exhaustion
        exhaustion = exhaustion_frequency in ["most_days"]
        if exhaustion:
            criteria_met.append("exhaustion")
        
        # Criterion 4: Slow walking speed
        slow_walking = False
        if gait_speed_m_per_s and gender and gender.lower() in self.GAIT_SPEED_CUTOFFS:
            gender_key = gender.lower()
            
            if height_cm:
                if gender_key == "male":
                    cutoff = self.GAIT_SPEED_CUTOFFS[gender_key]["height_low"] if height_cm <= 173 else self.GAIT_SPEED_CUTOFFS[gender_key]["height_high"]
                else:  # female
                    cutoff = self.GAIT_SPEED_CUTOFFS[gender_key]["height_low"] if height_cm <= 159 else self.GAIT_SPEED_CUTOFFS[gender_key]["height_high"]
            else:
                cutoff = 0.8  # Default cutoff
            
            slow_walking = gait_speed_m_per_s < cutoff
        
        if slow_walking:
            criteria_met.append("slow_walking_speed")
        
        # Criterion 5: Low physical activity
        low_activity = False
        if physical_activity_kcal_per_week and gender and gender.lower() in self.ACTIVITY_CUTOFFS:
            low_activity = physical_activity_kcal_per_week < self.ACTIVITY_CUTOFFS[gender.lower()]
        
        # Alternative: use ADL activity level
        if adl_activity_level and not low_activity:
            low_activity = adl_activity_level in ["low", "very_low"]
        
        if low_activity:
            criteria_met.append("low_physical_activity")
        
        # Count criteria
        criteria_count = len(criteria_met)
        
        # Determine frailty status
        if criteria_count >= 3:
            frailty_status = FrailtyStatus.FRAIL
        elif criteria_count >= 1:
            frailty_status = FrailtyStatus.PRE_FRAIL
        else:
            frailty_status = FrailtyStatus.ROBUST
        
        return FriedPhenotype(
            weight_loss=weight_loss,
            weakness=weakness,
            exhaustion=exhaustion,
            slow_walking_speed=slow_walking,
            low_physical_activity=low_activity,
            criteria_count=criteria_count,
            frailty_status=frailty_status,
        )
    
    def estimate_from_available_data(
        self,
        age: int,
        bmi: Optional[float] = None,
        adl_activity_breakdown: Optional[Dict[str, int]] = None,
        gait_speed: Optional[float] = None,
        icope_vitality: Optional[float] = None,
        medications_count: int = 0,
        chronic_conditions_count: int = 0,
    ) -> FriedPhenotype:
        """
        Estimate frailty phenotype from available health data.
        
        This is used when direct Fried criteria measurements are unavailable,
        leveraging ADL data and other health indicators.
        
        Args:
            age: Patient age
            bmi: Body mass index
            adl_activity_breakdown: Activity counts by type
            gait_speed: Measured or estimated gait speed
            icope_vitality: ICOPE vitality/nutrition score
            medications_count: Number of medications
            chronic_conditions_count: Number of chronic conditions
        
        Returns:
            Estimated FriedPhenotype
        """
        criteria_met = []
        
        # Estimate weight loss from BMI and ICOPE
        if bmi and bmi < 19:
            criteria_met.append("weight_loss")
        elif icope_vitality and icope_vitality < 50:
            criteria_met.append("weight_loss")
        
        # Estimate weakness from gait speed and age
        if gait_speed and gait_speed < 0.8:
            criteria_met.append("weakness")
        elif age > 85 and (not adl_activity_breakdown or sum(adl_activity_breakdown.values()) < 50):
            # Very low activity in very old
            criteria_met.append("weakness")
        
        # Estimate exhaustion from ICOPE and medication count
        if icope_vitality and icope_vitality < 40:
            criteria_met.append("exhaustion")
        
        # Estimate slow walking from gait speed
        if gait_speed and gait_speed < 0.8:
            criteria_met.append("slow_walking_speed")
        
        # Estimate low activity from ADL
        if adl_activity_breakdown:
            total_activities = sum(adl_activity_breakdown.values())
            if total_activities < 30:  # Very low activity
                criteria_met.append("low_physical_activity")
        
        # Deficit accumulation approach
        deficit_score = 0
        if bmi and (bmi < 19 or bmi > 30):
            deficit_score += 1
        if medications_count >= 5:
            deficit_score += 1
        if chronic_conditions_count >= 3:
            deficit_score += 1
        if age > 85:
            deficit_score += 1
        
        # Add deficit-based criteria
        if deficit_score >= 3 and "low_physical_activity" not in criteria_met:
            criteria_met.append("low_physical_activity")
        
        criteria_count = len(set(criteria_met))
        
        if criteria_count >= 3:
            frailty_status = FrailtyStatus.FRAIL
        elif criteria_count >= 1:
            frailty_status = FrailtyStatus.PRE_FRAIL
        else:
            frailty_status = FrailtyStatus.ROBUST
        
        return FriedPhenotype(
            weight_loss="weight_loss" in criteria_met,
            weakness="weakness" in criteria_met,
            exhaustion="exhaustion" in criteria_met,
            slow_walking_speed="slow_walking_speed" in criteria_met,
            low_physical_activity="low_physical_activity" in criteria_met,
            criteria_count=criteria_count,
            frailty_status=frailty_status,
        )
    
    def calculate_deficit_index(
        self,
        deficits: Dict[str, bool],
    ) -> Dict:
        """
        Calculate deficit accumulation frailty index.
        
        The FI is the proportion of deficits present out of total considered.
        
        Args:
            deficits: Dictionary of health deficits (True = present)
        
        Returns:
            Frailty index and interpretation
        """
        total_deficits = len(deficits)
        present_deficits = sum(1 for v in deficits.values() if v)
        
        fi_score = present_deficits / total_deficits if total_deficits > 0 else 0
        
        # Interpretation
        if fi_score < 0.08:
            category = "fit"
            frailty_status = FrailtyStatus.ROBUST
        elif fi_score < 0.25:
            category = "mild_frailty"
            frailty_status = FrailtyStatus.PRE_FRAIL
        else:
            category = "moderate_severe_frailty"
            frailty_status = FrailtyStatus.FRAIL
        
        return {
            "frailty_index": round(fi_score, 3),
            "present_deficits": present_deficits,
            "total_deficits": total_deficits,
            "category": category,
            "frailty_status": frailty_status,
            "interpretation": f"FI = {fi_score:.3f} ({category.replace('_', ' ')})",
        }
    
    def predict_frailty_trajectory(
        self,
        current_phenotype: FriedPhenotype,
        age: int,
        trend_duration_months: int = 12,
    ) -> Dict:
        """
        Predict frailty trajectory over time.
        
        Based on research showing frailty progression rates.
        
        Args:
            current_phenotype: Current frailty assessment
            age: Current age
            trend_duration_months: Prediction horizon
        
        Returns:
            Trajectory prediction
        """
        # Progression rates from literature (approximate annual rates)
        progression_rates = {
            FrailtyStatus.ROBUST: 0.15,      # 15% become pre-frail annually
            FrailtyStatus.PRE_FRAIL: 0.25,   # 25% become frail annually
            FrailtyStatus.FRAIL: 0.10,       # 10% become more frail (die/severe)
        }
        
        current_status = current_phenotype.frailty_status
        annual_rate = progression_rates.get(current_status, 0.15)
        
        # Adjust for age
        if age > 85:
            annual_rate *= 1.5
        elif age > 75:
            annual_rate *= 1.2
        
        # Calculate probability of worsening
        months_factor = trend_duration_months / 12
        worsening_probability = min(0.9, annual_rate * months_factor)
        
        # Predict future status
        if current_status == FrailtyStatus.ROBUST:
            likely_future = FrailtyStatus.PRE_FRAIL if worsening_probability > 0.5 else FrailtyStatus.ROBUST
        elif current_status == FrailtyStatus.PRE_FRAIL:
            if worsening_probability > 0.6:
                likely_future = FrailtyStatus.FRAIL
            elif worsening_probability > 0.3:
                likely_future = FrailtyStatus.PRE_FRAIL
            else:
                likely_future = FrailtyStatus.ROBUST  # Improvement possible
        else:  # FRAIL
            likely_future = FrailtyStatus.FRAIL  # Stays frail or worsens
        
        return {
            "current_status": current_status,
            "prediction_horizon_months": trend_duration_months,
            "worsening_probability": round(worsening_probability, 2),
            "predicted_status": likely_future,
            "confidence": "moderate",
            "key_factors": [
                "Age-related decline",
                "Current frailty status",
                "Activity level maintenance",
            ],
            "recommendations": [
                "Progressive resistance training",
                "Protein supplementation (1.0-1.2 g/kg/day)",
                "Vitamin D 800-1000 IU daily",
            ] if likely_future in [FrailtyStatus.PRE_FRAIL, FrailtyStatus.FRAIL] else [],
        }


# Singleton
_frailty_calculator: Optional[FrailtyIndexCalculator] = None


def get_frailty_calculator() -> FrailtyIndexCalculator:
    """Get or create singleton frailty calculator."""
    global _frailty_calculator
    if _frailty_calculator is None:
        _frailty_calculator = FrailtyIndexCalculator()
    return _frailty_calculator
