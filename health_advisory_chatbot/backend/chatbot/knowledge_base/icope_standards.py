"""
WHO ICOPE (Integrated Care for Older People) Standards

Implementation of WHO ICOPE Guidelines for person-centered assessment
and pathways in primary care for older people.

Reference: WHO ICOPE Guidelines 2017
https://www.who.int/publications/i/item/9789241550109
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ICOPE_Domain(str, Enum):
    """The six intrinsic capacity domains of ICOPE."""
    COGNITIVE = "cognitive"
    LOCOMOTOR = "locomotor"
    NUTRITION = "vitality_nutrition"
    SENSORY = "sensory"
    PSYCHOLOGICAL = "psychological"


class ICOPE_Level(str, Enum):
    """ICOPE assessment levels."""
    NO_DECLINE = "no_decline"  # No further action needed
    SMALL_DECLINE = "small_decline"  # Community-based interventions
    SIGNIFICANT_DECLINE = "significant_decline"  # Specialized assessment


@dataclass
class ICOPEScreening:
    """ICOPE screening tool results."""
    domain: ICOPE_Domain
    screening_result: str
    level: ICOPE_Level
    requires_referral: bool
    recommended_actions: List[str]


class ICOPEStandards:
    """
    WHO ICOPE Standards implementation.
    
    Provides:
    - Screening thresholds for each domain
    - Assessment protocols
    - Care pathway recommendations
    - Domain-specific interventions
    """
    
    # ICOPE Screening Thresholds
    COGNITIVE_THRESHOLDS = {
        "memory_complaint": "small_decline",
        "3_word_recall_failure": "significant_decline",
        "mmse_24_26": "small_decline",
        "mmse_below_24": "significant_decline",
    }
    
    LOCOMOTOR_THRESHOLDS = {
        "gait_speed_normal": "no_decline",  # >1.0 m/s
        "gait_speed_slow": "small_decline",  # 0.6-1.0 m/s
        "gait_speed_very_slow": "significant_decline",  # <0.6 m/s
        "sppb_8_9": "small_decline",
        "sppb_below_8": "significant_decline",
    }
    
    NUTRITION_THRESHOLDS = {
        "unintentional_weight_loss_5_percent": "small_decline",
        "unintentional_weight_loss_10_percent": "significant_decline",
        "mna_sf_8_11": "small_decline",
        "mna_sf_below_8": "significant_decline",
        "bmi_below_19": "significant_decline",
    }
    
    SENSORY_THRESHOLDS = {
        "vision_6_18_to_6_60": "small_decline",
        "vision_below_6_60": "significant_decline",
        "whisper_test_fail": "small_decline",
        "audiometry_moderate": "significant_decline",
    }
    
    PSYCHOLOGICAL_THRESHOLDS = {
        "phq2_positive": "small_decline",
        "gds15_5_9": "small_decline",
        "gds15_above_9": "significant_decline",
    }
    
    def __init__(self):
        """Initialize ICOPE standards."""
        self._interventions = self._load_interventions()
        self._assessment_tools = self._load_assessment_tools()
    
    def _load_interventions(self) -> Dict[str, List[Dict]]:
        """Load evidence-based interventions for each domain."""
        return {
            ICOPE_Domain.COGNITIVE.value: [
                {
                    "name": "Cognitive Stimulation",
                    "description": "Structured cognitive activities and exercises",
                    "evidence": "Moderate - improves cognitive function and quality of life",
                    "frequency": "Weekly sessions",
                    "setting": "Community or home-based",
                },
                {
                    "name": "Physical Exercise",
                    "description": "Aerobic and resistance exercise",
                    "evidence": "Strong - reduces cognitive decline risk",
                    "frequency": "150 minutes/week moderate intensity",
                    "setting": "Community or home",
                },
                {
                    "name": "Social Engagement",
                    "description": "Group activities and social participation",
                    "evidence": "Moderate - protects against cognitive decline",
                    "frequency": "Regular ongoing participation",
                    "setting": "Community centers",
                },
                {
                    "name": "Vascular Risk Management",
                    "description": "Control hypertension, diabetes, cholesterol",
                    "evidence": "Strong - reduces vascular dementia risk",
                    "frequency": "Ongoing medical management",
                    "setting": "Primary care",
                },
            ],
            ICOPE_Domain.LOCOMOTOR.value: [
                {
                    "name": "Progressive Resistance Training",
                    "description": "Strengthening exercises for major muscle groups",
                    "evidence": "Strong - improves strength and function",
                    "frequency": "2-3 times per week",
                    "setting": "Gym or home with equipment",
                },
                {
                    "name": "Balance Training",
                    "description": "Exercises to improve postural control",
                    "evidence": "Strong - reduces fall risk",
                    "frequency": "Daily",
                    "setting": "Home or group classes",
                },
                {
                    "name": "Tai Chi",
                    "description": "Mind-body exercise combining movement and meditation",
                    "evidence": "Strong - improves balance and reduces falls",
                    "frequency": "2-3 times per week",
                    "setting": "Group classes",
                },
                {
                    "name": "Walking Program",
                    "description": "Structured walking with progressive intensity",
                    "evidence": "Moderate - maintains mobility",
                    "frequency": "30 minutes daily",
                    "setting": "Community or home",
                },
            ],
            ICOPE_Domain.NUTRITION.value: [
                {
                    "name": "Protein Supplementation",
                    "description": "Increase protein intake to 1.0-1.2 g/kg/day",
                    "evidence": "Moderate - improves muscle mass",
                    "frequency": "Daily with meals",
                    "setting": "Home",
                },
                {
                    "name": "Vitamin D and Calcium",
                    "description": "Supplementation for bone health",
                    "evidence": "Moderate - reduces fracture risk",
                    "frequency": "Daily (Vit D 800-1000 IU)",
                    "setting": "Home",
                },
                {
                    "name": "Nutritional Counseling",
                    "description": "Dietitian assessment and meal planning",
                    "evidence": "Moderate - improves nutritional status",
                    "frequency": "Initial + follow-up",
                    "setting": "Outpatient",
                },
                {
                    "name": "Oral Nutritional Supplements",
                    "description": "Commercial supplements if underweight",
                    "evidence": "Moderate - promotes weight gain",
                    "frequency": "Between meals",
                    "setting": "Home",
                },
            ],
            ICOPE_Domain.SENSORY.value: [
                {
                    "name": "Cataract Surgery",
                    "description": "Surgical removal of cataracts",
                    "evidence": "Strong - improves vision and function",
                    "frequency": "Once per eye if indicated",
                    "setting": "Hospital",
                },
                {
                    "name": "Refractive Correction",
                    "description": "Updated glasses prescription",
                    "evidence": "Moderate - improves vision",
                    "frequency": "Annual eye exam",
                    "setting": "Optometry",
                },
                {
                    "name": "Hearing Aid Fitting",
                    "description": "Amplification devices for hearing loss",
                    "evidence": "Strong - improves communication",
                    "frequency": "As needed",
                    "setting": "Audiology",
                },
                {
                    "name": "Home Modifications",
                    "description": "Improve lighting, contrast, reduce glare",
                    "evidence": "Moderate - reduces fall risk",
                    "frequency": "One-time assessment",
                    "setting": "Home",
                },
            ],
            ICOPE_Domain.PSYCHOLOGICAL.value: [
                {
                    "name": "Physical Activity",
                    "description": "Structured exercise program",
                    "evidence": "Strong - reduces depressive symptoms",
                    "frequency": "3-5 times per week",
                    "setting": "Community or home",
                },
                {
                    "name": "Social Support Programs",
                    "description": "Group activities and peer support",
                    "evidence": "Moderate - improves mood",
                    "frequency": "Regular participation",
                    "setting": "Community centers",
                },
                {
                    "name": "Problem-Solving Therapy",
                    "description": "Structured psychological intervention",
                    "evidence": "Strong - effective for depression",
                    "frequency": "Weekly sessions x 6-8 weeks",
                    "setting": "Mental health services",
                },
                {
                    "name": "Antidepressant Medication",
                    "description": "Pharmacological treatment if moderate-severe",
                    "evidence": "Moderate - effective for major depression",
                    "frequency": "Daily medication",
                    "setting": "Primary care or psychiatry",
                },
            ],
        }
    
    def _load_assessment_tools(self) -> Dict[str, Dict]:
        """Load ICOPE assessment tools and cutoffs."""
        return {
            "cognitive": {
                "name": "Mini-Cog or MMSE",
                "cutoffs": {
                    "mmse_normal": ">= 27",
                    "mmse_mild": "24-26",
                    "mmse_moderate": "18-23",
                    "mmse_severe": "< 18",
                },
                "time": "5-10 minutes",
            },
            "locomotor": {
                "name": "Gait Speed or SPPB",
                "cutoffs": {
                    "gait_speed_normal": "> 1.0 m/s",
                    "gait_speed_slow": "0.6-1.0 m/s",
                    "gait_speed_very_slow": "< 0.6 m/s",
                    "sppb_normal": ">= 10",
                    "sppb_moderate": "8-9",
                    "sppb_low": "< 8",
                },
                "time": "5 minutes",
            },
            "nutrition": {
                "name": "MNA-SF or Weight History",
                "cutoffs": {
                    "mna_sf_normal": ">= 12",
                    "mna_sf_at_risk": "8-11",
                    "mna_sf_malnourished": "< 8",
                },
                "time": "5 minutes",
            },
            "vision": {
                "name": "Visual Acuity (Snellen)",
                "cutoffs": {
                    "vision_normal": ">= 6/12",
                    "vision_impaired": "6/18 to 6/60",
                    "vision_blind": "< 6/60",
                },
                "time": "3 minutes",
            },
            "hearing": {
                "name": "Whisper Test or Audiometry",
                "cutoffs": {
                    "hearing_normal": "Hears whisper at 2 feet",
                    "hearing_impaired": "Fails whisper test",
                },
                "time": "2 minutes",
            },
            "psychological": {
                "name": "PHQ-2 or GDS-15",
                "cutoffs": {
                    "phq2_negative": "< 3",
                    "phq2_positive": ">= 3",
                    "gds15_normal": "< 5",
                    "gds15_mild": "5-9",
                    "gds15_moderate_severe": ">= 10",
                },
                "time": "3-5 minutes",
            },
        }
    
    def assess_cognitive(
        self,
        memory_complaint: bool,
        word_recall_score: int,
        mmse_score: Optional[int] = None
    ) -> ICOPEScreening:
        """
        Assess cognitive domain per ICOPE.
        
        Args:
            memory_complaint: Patient reports memory problems
            word_recall_score: Number of words recalled (0-3)
            mmse_score: Optional MMSE score
        
        Returns:
            ICOPE screening result
        """
        if word_recall_score < 3 or (mmse_score and mmse_score < 24):
            level = ICOPE_Level.SIGNIFICANT_DECLINE
            return ICOPEScreening(
                domain=ICOPE_Domain.COGNITIVE,
                screening_result="Significant cognitive decline detected",
                level=level,
                requires_referral=True,
                recommended_actions=[
                    "Comprehensive cognitive assessment (MoCA or neuropsychological testing)",
                    "Screen for depression (pseudodementia)",
                    "Laboratory workup (B12, TSH, CBC, CMP)",
                    "Medication review for anticholinergics",
                    "Brain imaging if rapid decline or focal signs",
                    "Caregiver support and education",
                ]
            )
        elif memory_complaint or (mmse_score and 24 <= mmse_score <= 26):
            return ICOPEScreening(
                domain=ICOPE_Domain.COGNITIVE,
                screening_result="Mild cognitive concerns",
                level=ICOPE_Level.SMALL_DECLINE,
                requires_referral=False,
                recommended_actions=[
                    "Cognitive stimulation activities",
                    "Physical exercise program",
                    "Social engagement",
                    "Vascular risk factor management",
                    "Reassess in 6 months",
                ]
            )
        else:
            return ICOPEScreening(
                domain=ICOPE_Domain.COGNITIVE,
                screening_result="No significant cognitive decline",
                level=ICOPE_Level.NO_DECLINE,
                requires_referral=False,
                recommended_actions=["Continue healthy lifestyle"]
            )
    
    def assess_locomotor(
        self,
        gait_speed: Optional[float] = None,
        sppb_score: Optional[int] = None
    ) -> ICOPEScreening:
        """
        Assess locomotor domain per ICOPE.
        
        Args:
            gait_speed: Usual gait speed in m/s (over 4-6m)
            sppb_score: Short Physical Performance Battery score (0-12)
        
        Returns:
            ICOPE screening result
        """
        if (gait_speed and gait_speed < 0.6) or (sppb_score and sppb_score < 8):
            return ICOPEScreening(
                domain=ICOPE_Domain.LOCOMOTOR,
                screening_result="Significant locomotor decline",
                level=ICOPE_Level.SIGNIFICANT_DECLINE,
                requires_referral=True,
                recommended_actions=[
                    "Comprehensive falls assessment",
                    "Physical therapy evaluation",
                    "Home safety assessment",
                    "Medication review (especially sedatives)",
                    "Vision assessment",
                    "Consider mobility aid",
                ]
            )
        elif (gait_speed and 0.6 <= gait_speed < 1.0) or (sppb_score and 8 <= sppb_score <= 9):
            return ICOPEScreening(
                domain=ICOPE_Domain.LOCOMOTOR,
                screening_result="Mild locomotor decline",
                level=ICOPE_Level.SMALL_DECLINE,
                requires_referral=False,
                recommended_actions=[
                    "Progressive resistance training",
                    "Balance exercises",
                    "Tai Chi or similar program",
                    "Walking program",
                    "Vitamin D supplementation if deficient",
                ]
            )
        else:
            return ICOPEScreening(
                domain=ICOPE_Domain.LOCOMOTOR,
                screening_result="Normal locomotor function",
                level=ICOPE_Level.NO_DECLINE,
                requires_referral=False,
                recommended_actions=["Maintain regular physical activity"]
            )
    
    def assess_nutrition(
        self,
        weight_loss_6mo: Optional[float] = None,
        mna_sf_score: Optional[float] = None,
        bmi: Optional[float] = None
    ) -> ICOPEScreening:
        """Assess nutrition domain per ICOPE."""
        significant_decline = (
            (weight_loss_6mo and weight_loss_6mo >= 10) or
            (mna_sf_score and mna_sf_score < 8) or
            (bmi and bmi < 19)
        )
        
        small_decline = (
            (weight_loss_6mo and 5 <= weight_loss_6mo < 10) or
            (mna_sf_score and 8 <= mna_sf_score <= 11)
        )
        
        if significant_decline:
            return ICOPEScreening(
                domain=ICOPE_Domain.NUTRITION,
                screening_result="Malnutrition or significant weight loss",
                level=ICOPE_Level.SIGNIFICANT_DECLINE,
                requires_referral=True,
                recommended_actions=[
                    "Comprehensive nutritional assessment by dietitian",
                    "Evaluate for depression, cognitive impairment, dysphagia",
                    "Medication review for appetite suppressants",
                    "Consider oral nutritional supplements",
                    "Monitor weight weekly",
                    "Evaluate for underlying disease",
                ]
            )
        elif small_decline:
            return ICOPEScreening(
                domain=ICOPE_Domain.NUTRITION,
                screening_result="At risk for malnutrition",
                level=ICOPE_Level.SMALL_DECLINE,
                requires_referral=False,
                recommended_actions=[
                    "Increase protein intake to 1.0-1.2 g/kg/day",
                    "Vitamin D 800-1000 IU daily",
                    "Nutritional counseling",
                    "Meal delivery services if needed",
                    "Social dining opportunities",
                ]
            )
        else:
            return ICOPEScreening(
                domain=ICOPE_Domain.NUTRITION,
                screening_result="Normal nutritional status",
                level=ICOPE_Level.NO_DECLINE,
                requires_referral=False,
                recommended_actions=["Maintain balanced diet"]
            )
    
    def get_domain_interventions(self, domain: str) -> List[Dict]:
        """Get evidence-based interventions for a specific domain."""
        return self._interventions.get(domain, [])
    
    def get_assessment_tool(self, domain: str) -> Dict:
        """Get assessment tool details for a domain."""
        return self._assessment_tools.get(domain, {})
    
    def calculate_overall_icope_score(
        self,
        domain_results: List[ICOPEScreening]
    ) -> Dict:
        """
        Calculate overall ICOPE score and priority.
        
        Args:
            domain_results: Results from all 6 domain assessments
        
        Returns:
            Summary with overall score and priority domains
        """
        score_mapping = {
            ICOPE_Level.NO_DECLINE: 100,
            ICOPE_Level.SMALL_DECLINE: 70,
            ICOPE_Level.SIGNIFICANT_DECLINE: 40,
        }
        
        domain_scores = {}
        significant_declines = []
        small_declines = []
        
        for result in domain_results:
            score = score_mapping.get(result.level, 50)
            domain_scores[result.domain.value] = score
            
            if result.level == ICOPE_Level.SIGNIFICANT_DECLINE:
                significant_declines.append(result.domain.value)
            elif result.level == ICOPE_Level.SMALL_DECLINE:
                small_declines.append(result.domain.value)
        
        overall_score = sum(domain_scores.values()) / len(domain_scores) if domain_scores else 0
        
        return {
            "overall_score": round(overall_score, 1),
            "domain_scores": domain_scores,
            "significant_declines": significant_declines,
            "small_declines": small_declines,
            "requires_specialist_referral": len(significant_declines) > 0,
            "priority_action": "Address significant declines in: " + ", ".join(significant_declines) if significant_declines else "Maintain current function",
        }


# Singleton
_icope_standards: Optional[ICOPEStandards] = None


def get_icope_standards() -> ICOPEStandards:
    """Get or create singleton ICOPE standards instance."""
    global _icope_standards
    if _icope_standards is None:
        _icope_standards = ICOPEStandards()
    return _icope_standards
