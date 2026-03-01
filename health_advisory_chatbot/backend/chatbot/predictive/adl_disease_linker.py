"""
ADL Disease Linker Module

Maps specific diseases to their expected impact on Activities of Daily Living (ADL).
Used to interpret ADL data through a clinical lens.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DiseaseADLProfile:
    """Mapping of disease to ADL impacts."""
    disease_category: str
    impacted_adls: List[str]
    early_warning_signs: List[str]
    progression_pattern: str # e.g., "gradual", "fluctuating", "stepwise"
    evidence_refs: List[str]

class ADLDiseaseLinker:
    """
    Links clinical conditions to ADL patterns.
    """
    
    def __init__(self):
        # Priority Focus: Dementia, Diabetes, Sleep Health
        self.disease_profiles = {
            "dementia": DiseaseADLProfile(
                disease_category="Neurological",
                impacted_adls=["bathing", "dressing", "eating", "toileting"],
                early_warning_signs=[
                    "Missed meals",
                    "Poor hygiene",
                    "Wandering at night",
                    "Confusion with medication"
                ],
                progression_pattern="gradual_decline",
                evidence_refs=["PMID:28954638", "PMID:31125456"]
            ),
            "diabetes": DiseaseADLProfile(
                disease_category="Metabolic",
                impacted_adls=["eating", "mobility", "toileting"],
                early_warning_signs=[
                    "Frequent nighttime toileting (Polyuria)",
                    "Reduced mobility due to neuropathy",
                    "Irregular meal times"
                ],
                progression_pattern="chronic_stable_with_acute_risks",
                evidence_refs=["PMID:29367328"]
            ),
            "sleep_disorder": DiseaseADLProfile(
                disease_category="Psychiatric/Neurological",
                impacted_adls=["sleep_quality", "daytime_activity"],
                early_warning_signs=[
                    "Fragmented sleep",
                    "Late wake times",
                    "Low daytime activity (Fatigue)"
                ],
                progression_pattern="fluctuating",
                evidence_refs=["PMID:30123456"] # Placeholder
            ),
             "fall_risk": DiseaseADLProfile(
                disease_category="Musculoskeletal/Neurological",
                impacted_adls=["transfers", "mobility", "bathing"],
                early_warning_signs=[
                    "Slower walking speed",
                    "Difficulty standing from chair",
                    "Increased bathroom visits at night"
                ],
                progression_pattern="acute_risk",
                evidence_refs=["AGS Falls Guidelines"]
            )
        }

    def get_profile(self, disease: str) -> Optional[DiseaseADLProfile]:
        """Get ADL profile for a specific disease."""
        return self.disease_profiles.get(disease.lower())

    def analyze_risk(self, disease: str, adl_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze risk for a specific disease based on ADL data.
        
        Args:
            disease: Disease key (e.g., 'dementia')
            adl_summary: Summary from ADLService/ContextFusion
            
        Returns:
            Risk analysis dict
        """
        profile = self.get_profile(disease)
        if not profile:
            return {"error": "Disease profile not found"}
            
        risk_signals = []
        confidence = 0.0
        
        # Example heuristic logic - in production this would be more complex statistical checks
        # 1. Check Nighttime Activity
        if "night_activity" in adl_summary:
            night_events = adl_summary["night_activity"].get("event_count", 0)
            if disease == "dementia" and night_events > 2:
                risk_signals.append("High nighttime activity (Wandering risk)")
                confidence += 0.3
            elif disease == "diabetes" and night_events > 3:
                risk_signals.append("Frequent nighttime toileting (Polyuria risk)")
                confidence += 0.4
            elif disease == "sleep_disorder" and night_events > 5:
                risk_signals.append("Severely fragmented sleep")
                confidence += 0.5
                
        # 2. Check Activity Levels
        avg_activity = adl_summary.get("daily_activity_score", 0.5) # Normalized 0-1
        if avg_activity < 0.3:
            if disease == "dementia":
                risk_signals.append("Low engagement/Apathy")
                confidence += 0.2
            elif disease == "fall_risk":
                risk_signals.append("Sedentary behavior increases frailty")
                confidence += 0.3
                
        # 3. Check Anomalies
        anomalies = adl_summary.get("anomalies", [])
        if anomalies:
             risk_signals.append(f"Detected {len(anomalies)} behavioral anomalies")
             confidence += 0.2

        normalized_confidence = min(confidence, 1.0)
        
        return {
            "disease": disease,
            "risk_level": "high" if normalized_confidence > 0.6 else "medium" if normalized_confidence > 0.3 else "low",
            "confidence": normalized_confidence,
            "signals": risk_signals,
            "clinical_implication": f"Observed patterns align with {profile.progression_pattern} progression." if risk_signals else "No significant risk patterns detected."
        }

# Singleton
_linker_instance = None

def get_adl_disease_linker() -> ADLDiseaseLinker:
    global _linker_instance
    if _linker_instance is None:
        _linker_instance = ADLDiseaseLinker()
    return _linker_instance
