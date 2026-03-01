"""
Outcome Statistics Module

Provides evidence-based statistical data for health outcomes.
Used to ground predictive models in real-world clinical data.
"""

from typing import Dict, Any, Optional

class OutcomeStatistics:
    """
    Repository of clinical outcome statistics for elderly care.
    """
    
    # Statistics derived from clinical literature (e.g., PubMed, CDC, WHO)
    # In a real system, this could be populated from a database or external API
    STATS_DB = {
        "dementia": {
            "progression_rate": "3-4 points/year decline on MMSE",
            "fall_risk_increase": "2-3x higher than age-matched controls",
            "hospitalization_risk": "High (UTI, Pneumonia common precipitants)",
            "mortality_factors": ["Advanced age", "Male sex", "Comorbidities"],
            "sources": ["PMID:21514249", "Alzheimer's Association 2023"]
        },
        "diabetes": {
            "hypoglycemia_risk": "High in elderly on insulin/sulfonylureas",
            "cognitive_impact": "1.5-2x increased risk of dementia",
            "fall_risk_increase": "Increased due to neuropathy/vision loss",
            "sources": ["ADA Standards of Care 2023", "PMID:28796598"]
        },
        "falls": {
            "recurrence_rate": "50% within 12 months if untreated",
            "injury_rate": "10-20% result in serious injury (fracture/head trauma)",
            "mortality_risk": "High 1-year mortality after hip fracture (20-30%)",
            "sources": ["CDC STEADI", "AGS Falls Guidelines"]
        },
        "sleep_disorders": {
            "cognitive_impact": "Chronic insomnia associated with cognitive decline",
            "fall_risk_increase": "Sedative use increases fall risk by ~50%",
            "sources": ["AASM Guidelines", "PMID:30601719"]
        }
    }

    def get_statistics(self, condition: str) -> Optional[Dict[str, Any]]:
        """Get outcome statistics for a specific condition."""
        return self.STATS_DB.get(condition.lower())

    def get_risk_multiplier(self, condition: str, outcome: str) -> float:
        """
        Get a specific risk multiplier if available.
        Useful for quantitative adjustments in trajectory models.
        """
        # Placeholder logic - would be more granular in production
        if condition == "dementia" and outcome == "fall_risk":
            return 2.5
        if condition == "diabetes" and outcome == "dementia_risk":
            return 1.75
        if condition == "sleep_disorders" and outcome == "fall_risk":
            return 1.5
            
        return 1.0

# Singleton
_stats_instance = None

def get_outcome_statistics() -> OutcomeStatistics:
    global _stats_instance
    if _stats_instance is None:
        _stats_instance = OutcomeStatistics()
    return _stats_instance
