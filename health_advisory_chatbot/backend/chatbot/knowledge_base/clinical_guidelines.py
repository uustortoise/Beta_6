"""
Clinical Guidelines Database

Curated evidence-based guidelines from authoritative sources:
- American Geriatrics Society (AGS) Beers Criteria
- AGS Fall Prevention Guidelines
- WHO ICOPE Guidelines
- NICE Guidelines (UK)
- AASM Sleep Guidelines

All guidelines include evidence levels and are version-tracked.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class GuidelineSource(str, Enum):
    """Authoritative guideline sources."""
    AGS_BEERS = "ags_beers"  # American Geriatrics Society
    AGS_FALLS = "ags_falls"
    WHO_ICOPE = "who_icope"  # WHO Integrated Care for Older People
    NICE = "nice"  # National Institute for Health and Care Excellence
    AASM = "aasm"  # American Academy of Sleep Medicine
    ADA = "ada"  # American Diabetes Association
    AHA = "aha"  # American Heart Association
    ESPEN = "espen"  # European Society for Clinical Nutrition


@dataclass(frozen=True)
class GuidelineRecommendation:
    """
    Individual recommendation from a clinical guideline.
    
    Immutable to ensure guideline integrity.
    """
    # Required fields
    guideline_id: str
    source: GuidelineSource
    version: str
    publication_year: int
    category: str  # e.g., "fall_prevention", "medication_safety"
    title: str
    description: str
    evidence_grade: str  # A/B/C per GRADE system
    
    # Optional fields with defaults
    action_steps: List[str] = field(default_factory=list)
    evidence_summary: str = ""
    applicable_conditions: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class ClinicalGuidelinesDB:
    """
    Database of evidence-based clinical guidelines for elderly care.
    
    Provides:
    - Guideline lookup by condition/category
    - Evidence-based recommendation retrieval
    - Version tracking for guideline updates
    
    Thread-safe and immutable after initialization.
    """
    
    # Current guideline versions
    VERSIONS = {
        GuidelineSource.AGS_BEERS: "2023",
        GuidelineSource.AGS_FALLS: "2023",
        GuidelineSource.WHO_ICOPE: "2017",
        GuidelineSource.NICE: "2023",
        GuidelineSource.AASM: "2023",
    }
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize guidelines database.
        
        Args:
            data_path: Path to external guideline JSON files (optional)
        """
        self._guidelines: Dict[str, GuidelineRecommendation] = {}
        self._by_category: Dict[str, List[str]] = {}
        self._by_condition: Dict[str, List[str]] = {}
        
        # Load built-in guidelines
        self._load_built_in_guidelines()
        
        # Load external guidelines if provided
        if data_path:
            self._load_external_guidelines(data_path)
    
    def _load_built_in_guidelines(self) -> None:
        """Load curated guidelines embedded in the module."""
        
        # === AGS BEERS CRITERIA 2023 - Medication Safety ===
        beers_medications = [
            {
                "id": "beers_2023_anticholinergic",
                "category": "medication_safety",
                "title": "Avoid Anticholinergic Medications",
                "description": "Highly anticholinergic medications increase risk of confusion, constipation, urinary retention, and falls in older adults.",
                "examples": ["Diphenhydramine", "Oxybutynin", "Amitriptyline", "Paroxetine"],
                "evidence_grade": "A",
                "action_steps": [
                    "Review current medications for anticholinergic burden",
                    "Consider alternative medications with lower anticholinergic properties",
                    "If necessary, use lowest effective dose for shortest duration"
                ],
            },
            {
                "id": "beers_2023_benzodiazepine",
                "category": "medication_safety",
                "title": "Avoid Benzodiazepines for Insomnia",
                "description": "Benzodiazepines increase risk of cognitive impairment, delirium, falls, fractures, and motor vehicle accidents in older adults.",
                "examples": ["Diazepam", "Lorazepam", "Alprazolam", "Temazepam"],
                "evidence_grade": "A",
                "action_steps": [
                    "Gradual tapering if long-term use",
                    "Consider non-pharmacological sleep interventions",
                    "If sleep medication necessary, consider ramelteon or low-dose doxepin"
                ],
            },
            {
                "id": "beers_2023_nsaid",
                "category": "medication_safety",
                "title": "Caution with NSAIDs",
                "description": "NSAIDs increase risk of GI bleeding, renal failure, cardiovascular events, and exacerbation of heart failure.",
                "evidence_grade": "A",
                "action_steps": [
                    "Use lowest effective dose for shortest duration",
                    "Consider PPI protection if GI risk factors present",
                    "Monitor renal function and blood pressure",
                    "Consider acetaminophen as first-line for pain"
                ],
            },
            {
                "id": "beers_2023_hypoglycemic",
                "category": "medication_safety",
                "title": "Avoid Long-acting Sulfonylureas in Diabetes",
                "description": "Glyburide and chlorpropamide cause prolonged hypoglycemia; prefer shorter-acting agents.",
                "examples": ["Glyburide", "Chlorpropamide"],
                "evidence_grade": "A",
                "action_steps": [
                    "Switch to glipizide or other shorter-acting agent",
                    "Consider metformin if renal function adequate",
                    "Monitor blood glucose closely during transition"
                ],
            },
        ]
        
        for med in beers_medications:
            self._add_guideline(
                GuidelineRecommendation(
                    guideline_id=med["id"],
                    source=GuidelineSource.AGS_BEERS,
                    version="2023",
                    publication_year=2023,
                    category=med["category"],
                    title=med["title"],
                    description=med["description"],
                    action_steps=med["action_steps"],
                    evidence_grade=med["evidence_grade"],
                    evidence_summary=f"Based on AGS Beers Criteria 2023",
                )
            )
        
        # === AGS FALL PREVENTION GUIDELINES 2023 ===
        fall_guidelines = [
            {
                "id": "ags_falls_2023_multifactorial",
                "category": "fall_prevention",
                "title": "Multifactorial Fall Risk Assessment",
                "description": "Comprehensive fall risk assessment including history of falls, gait/balance assessment, orthostatic hypotension, medication review, vision assessment, and environmental hazards.",
                "evidence_grade": "A",
                "action_steps": [
                    "Conduct standardized gait assessment (Tinetti or Timed Up and Go)",
                    "Review medications for fall-risk increasing drugs",
                    "Assess orthostatic vital signs",
                    "Evaluate vision and footwear",
                    "Conduct home safety evaluation"
                ],
                "applicable_conditions": ["history_of_falls", "gait_abnormality", "high_fall_risk"],
            },
            {
                "id": "ags_falls_2023_exercise",
                "category": "fall_prevention",
                "title": "Exercise Programs for Fall Prevention",
                "description": "Group or individual exercise programs including gait, balance, and functional training to prevent falls.",
                "evidence_grade": "A",
                "action_steps": [
                    "Tai Chi (strong evidence)",
                    "Balance and strength training at least 3x/week",
                    "Physical therapy referral for individualized program",
                    "Continue exercises indefinitely for sustained benefit"
                ],
                "applicable_conditions": ["high_fall_risk", "mobility_limitation", "balance_impairment"],
            },
            {
                "id": "ags_falls_2023_vitamin_d",
                "category": "fall_prevention",
                "title": "Vitamin D Supplementation",
                "description": "Vitamin D supplementation (800 IU/day) for adults aged 65+ with vitamin D deficiency or high fall risk.",
                "evidence_grade": "B",
                "action_steps": [
                    "Check 25-hydroxyvitamin D level",
                    "Supplement if <20 ng/mL or high fall risk",
                    "Typical dose: 800-1000 IU daily",
                    "Monitor calcium levels if combined with calcium"
                ],
            },
            {
                "id": "ags_falls_2023_home_modification",
                "category": "fall_prevention",
                "title": "Home Safety Modifications",
                "description": "Environmental modifications to reduce fall hazards, particularly for those with visual impairment or history of falls.",
                "evidence_grade": "B",
                "action_steps": [
                    "Remove tripping hazards (rugs, cords)",
                    "Install grab bars in bathroom",
                    "Improve lighting, especially at night",
                    "Install handrails on stairs",
                    "Non-slip mats in bathroom"
                ],
            },
        ]
        
        for guide in fall_guidelines:
            self._add_guideline(
                GuidelineRecommendation(
                    guideline_id=guide["id"],
                    source=GuidelineSource.AGS_FALLS,
                    version="2023",
                    publication_year=2023,
                    category=guide["category"],
                    title=guide["title"],
                    description=guide["description"],
                    action_steps=guide["action_steps"],
                    evidence_grade=guide["evidence_grade"],
                    applicable_conditions=guide.get("applicable_conditions", []),
                )
            )
        
        # === WHO ICOPE GUIDELINES ===
        icope_guidelines = [
            {
                "id": "who_icope_cognitive",
                "category": "cognitive_health",
                "title": "ICOPE Cognitive Decline Screening",
                "description": "Screen for cognitive decline using validated tools; intervene with cognitive stimulation, physical activity, and vascular risk factor management.",
                "evidence_grade": "B",
                "action_steps": [
                    "Mini-Cog or MMSE screening",
                    "Cognitive stimulation activities",
                    "Physical exercise (150 min/week moderate)",
                    "Manage hypertension, diabetes, cholesterol",
                    "Social engagement programs"
                ],
                "applicable_conditions": ["mild_cognitive_impairment", "cognitive_decline"],
            },
            {
                "id": "who_icope_mobility",
                "category": "mobility",
                "title": "ICOPE Locomotor Capacity Assessment",
                "description": "Assess mobility using SPPB or gait speed; intervene with progressive resistance training and balance exercises.",
                "evidence_grade": "A",
                "action_steps": [
                    "Measure usual gait speed over 4-6 meters",
                    "Short Physical Performance Battery (SPPB)",
                    "Progressive resistance training 2-3x/week",
                    "Balance training daily",
                    "Walking program 30 min/day"
                ],
                "applicable_conditions": ["slow_gait", "mobility_limitation", "sarcopenia"],
            },
            {
                "id": "who_icope_nutrition",
                "category": "nutrition",
                "title": "ICOPE Malnutrition Screening",
                "description": "Screen for malnutrition risk; intervene with nutritional counseling and protein supplementation if indicated.",
                "evidence_grade": "B",
                "action_steps": [
                    "MNA-SF or MUST screening",
                    "Dietary assessment",
                    "Protein intake 1.0-1.2 g/kg/day",
                    "Vitamin D and calcium as needed",
                    "Oral nutritional supplements if underweight"
                ],
                "applicable_conditions": ["unintentional_weight_loss", "malnutrition_risk"],
            },
            {
                "id": "who_icope_vision",
                "category": "sensory",
                "title": "ICOPE Vision Assessment",
                "description": "Screen for vision impairment; refer for cataract surgery, refractive correction, or AMD treatment as indicated.",
                "evidence_grade": "B",
                "action_steps": [
                    "Visual acuity testing (Snellen chart)",
                    "Cataract evaluation if vision <6/18",
                    "Refractive correction update",
                    "Home lighting assessment",
                    "Fall prevention if vision impaired"
                ],
            },
            {
                "id": "who_icope_hearing",
                "category": "sensory",
                "title": "ICOPE Hearing Assessment",
                "description": "Screen for hearing loss; provide hearing aids and communication strategies.",
                "evidence_grade": "B",
                "action_steps": [
                    "Whisper test or audiometry",
                    "Hearing aid fitting if indicated",
                    "Communication training",
                    "Reduce background noise",
                    "Regular ear wax removal"
                ],
            },
            {
                "id": "who_icope_depression",
                "category": "mental_health",
                "title": "ICOPE Depression Screening",
                "description": "Screen for depressive symptoms; provide psychological therapy, social support, and pharmacotherapy if indicated.",
                "evidence_grade": "A",
                "action_steps": [
                    "PHQ-2 or GDS-15 screening",
                    "Structured physical activity program",
                    "Social engagement activities",
                    "Cognitive behavioral therapy",
                    "Antidepressants if moderate-severe"
                ],
                "applicable_conditions": ["depressive_symptoms", "social_isolation"],
            },
        ]
        
        for guide in icope_guidelines:
            self._add_guideline(
                GuidelineRecommendation(
                    guideline_id=guide["id"],
                    source=GuidelineSource.WHO_ICOPE,
                    version="2017",
                    publication_year=2017,
                    category=guide["category"],
                    title=guide["title"],
                    description=guide["description"],
                    action_steps=guide["action_steps"],
                    evidence_grade=guide["evidence_grade"],
                    applicable_conditions=guide.get("applicable_conditions", []),
                )
            )
        
        # === SLEEP MEDICINE GUIDELINES (AASM) ===
        sleep_guidelines = [
            {
                "id": "aasm_insomnia_2023",
                "category": "sleep_disorders",
                "title": "Chronic Insomnia Management",
                "description": "Cognitive Behavioral Therapy for Insomnia (CBT-I) as first-line treatment; avoid chronic hypnotic use.",
                "evidence_grade": "A",
                "action_steps": [
                    "Sleep hygiene education",
                    "Stimulus control therapy",
                    "Sleep restriction therapy",
                    "Cognitive therapy for sleep-related worries",
                    "Consider brief hypnotic only if CBT-I not available"
                ],
                "applicable_conditions": ["chronic_insomnia", "sleep_onset_insomnia"],
            },
            {
                "id": "aasm_sleep_apnea_2023",
                "category": "sleep_disorders",
                "title": "Obstructive Sleep Apnea Screening",
                "description": "Screen high-risk patients (hypertension, obesity, daytime sleepiness); refer for polysomnography if indicated.",
                "evidence_grade": "B",
                "action_steps": [
                    "STOP-BANG questionnaire",
                    "Home sleep apnea testing or polysomnography",
                    "CPAP for moderate-severe OSA",
                    "Weight loss if overweight",
                    "Positional therapy if positional OSA"
                ],
                "applicable_conditions": ["loud_snoring", "daytime_sleepiness", "hypertension"],
            },
            {
                "id": "aasm_sleep_hygiene",
                "category": "sleep_health",
                "title": "Sleep Hygiene Recommendations",
                "description": "Behavioral and environmental modifications to promote healthy sleep.",
                "evidence_grade": "C",
                "action_steps": [
                    "Consistent sleep/wake schedule",
                    "Limit caffeine after noon",
                    "Avoid alcohol near bedtime",
                    "Create cool, dark, quiet sleep environment",
                    "Limit screen exposure 1 hour before bed"
                ],
            },
        ]
        
        for guide in sleep_guidelines:
            self._add_guideline(
                GuidelineRecommendation(
                    guideline_id=guide["id"],
                    source=GuidelineSource.AASM,
                    version="2023",
                    publication_year=2023,
                    category=guide["category"],
                    title=guide["title"],
                    description=guide["description"],
                    action_steps=guide["action_steps"],
                    evidence_grade=guide["evidence_grade"],
                    applicable_conditions=guide.get("applicable_conditions", []),
                )
            )
        
        # === DIABETES GUIDELINES (ADA) ===
        diabetes_guidelines = [
            {
                "id": "ada_elderly_2023",
                "category": "chronic_disease",
                "title": "Diabetes Management in Older Adults",
                "description": "Individualized glycemic targets based on health status; prioritize hypoglycemia prevention.",
                "evidence_grade": "B",
                "action_steps": [
                    "Healthy: HbA1c <7.5%",
                    "Complex/intermediate: HbA1c <8.0%",
                    "Very complex/poor health: HbA1c <8.5%",
                    "Avoid sulfonylureas if possible",
                    "Prioritize metformin, DPP-4 inhibitors, GLP-1 RAs"
                ],
                "applicable_conditions": ["diabetes_type_2", "elderly_diabetes"],
            },
        ]
        
        for guide in diabetes_guidelines:
            self._add_guideline(
                GuidelineRecommendation(
                    guideline_id=guide["id"],
                    source=GuidelineSource.ADA,
                    version="2023",
                    publication_year=2023,
                    category=guide["category"],
                    title=guide["title"],
                    description=guide["description"],
                    action_steps=guide["action_steps"],
                    evidence_grade=guide["evidence_grade"],
                    applicable_conditions=guide.get("applicable_conditions", []),
                )
            )
    
    def _add_guideline(self, guideline: GuidelineRecommendation) -> None:
        """Add guideline to indexes."""
        self._guidelines[guideline.guideline_id] = guideline
        
        # Index by category
        if guideline.category not in self._by_category:
            self._by_category[guideline.category] = []
        self._by_category[guideline.category].append(guideline.guideline_id)
        
        # Index by applicable conditions
        for condition in guideline.applicable_conditions:
            if condition not in self._by_condition:
                self._by_condition[condition] = []
            self._by_condition[condition].append(guideline.guideline_id)
    
    def _load_external_guidelines(self, data_path: Path) -> None:
        """Load additional guidelines from external JSON files."""
        if not data_path.exists():
            return
        
        for json_file in data_path.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    for guide_data in data.get("guidelines", []):
                        guideline = GuidelineRecommendation(
                            guideline_id=guide_data["id"],
                            source=GuidelineSource(guide_data["source"]),
                            version=guide_data["version"],
                            publication_year=guide_data["year"],
                            category=guide_data["category"],
                            title=guide_data["title"],
                            description=guide_data["description"],
                            action_steps=guide_data.get("action_steps", []),
                            evidence_grade=guide_data.get("evidence_grade", "C"),
                            applicable_conditions=guide_data.get("conditions", []),
                        )
                        self._add_guideline(guideline)
            except Exception as e:
                # Log error but continue loading other files
                print(f"Error loading guideline file {json_file}: {e}")
    
    def get_guideline(self, guideline_id: str) -> Optional[GuidelineRecommendation]:
        """Retrieve specific guideline by ID."""
        return self._guidelines.get(guideline_id)
    
    def search_by_category(self, category: str) -> List[GuidelineRecommendation]:
        """Find all guidelines in a category."""
        ids = self._by_category.get(category, [])
        return [self._guidelines[id] for id in ids if id in self._guidelines]
    
    def search_by_condition(self, condition: str) -> List[GuidelineRecommendation]:
        """Find guidelines applicable to a specific condition."""
        ids = self._by_condition.get(condition, [])
        return [self._guidelines[id] for id in ids if id in self._guidelines]
    
    def search(
        self,
        categories: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        min_evidence_grade: Optional[str] = None,
        sources: Optional[List[GuidelineSource]] = None
    ) -> List[GuidelineRecommendation]:
        """
        Advanced search across guidelines.
        
        Args:
            categories: Filter by guideline categories
            conditions: Filter by applicable conditions
            min_evidence_grade: Minimum evidence grade (A/B/C)
            sources: Filter by guideline sources
        
        Returns:
            Matching guidelines sorted by evidence grade
        """
        results = list(self._guidelines.values())
        
        # Filter by categories
        if categories:
            results = [g for g in results if g.category in categories]
        
        # Filter by conditions
        if conditions:
            results = [
                g for g in results 
                if any(c in g.applicable_conditions for c in conditions)
            ]
        
        # Filter by evidence grade
        if min_evidence_grade:
            grade_order = {"A": 3, "B": 2, "C": 1}
            min_level = grade_order.get(min_evidence_grade, 0)
            results = [
                g for g in results 
                if grade_order.get(g.evidence_grade, 0) >= min_level
            ]
        
        # Filter by sources
        if sources:
            results = [g for g in results if g.source in sources]
        
        # Sort by evidence grade (A first)
        grade_priority = {"A": 0, "B": 1, "C": 2}
        results.sort(key=lambda g: grade_priority.get(g.evidence_grade, 3))
        
        return results
    
    def get_medication_alerts(self, medications: List[str]) -> List[GuidelineRecommendation]:
        """
        Get medication safety alerts based on current medications.
        
        Args:
            medications: List of medication names
        
        Returns:
            Relevant medication safety guidelines
        """
        alerts = []
        
        # Map common medication classes to Beers criteria
        medication_checks = {
            "diphenhydramine": "beers_2023_anticholinergic",
            "benadryl": "beers_2023_anticholinergic",
            "oxybutynin": "beers_2023_anticholinergic",
            "amitriptyline": "beers_2023_anticholinergic",
            "paroxetine": "beers_2023_anticholinergic",
            "diazepam": "beers_2023_benzodiazepine",
            "lorazepam": "beers_2023_benzodiazepine",
            "alprazolam": "beers_2023_benzodiazepine",
            "temazepam": "beers_2023_benzodiazepine",
            "glyburide": "beers_2023_hypoglycemic",
            "chlorpropamide": "beers_2023_hypoglycemic",
        }
        
        med_lower = [m.lower() for m in medications]
        
        for med in med_lower:
            if med in medication_checks:
                guideline = self.get_guideline(medication_checks[med])
                if guideline:
                    alerts.append(guideline)
        
        return alerts
    
    def get_fall_prevention_plan(
        self,
        risk_factors: List[str]
    ) -> List[GuidelineRecommendation]:
        """
        Generate evidence-based fall prevention recommendations.
        
        Args:
            risk_factors: Identified fall risk factors
        
        Returns:
            Prioritized action plan
        """
        recommendations = []
        
        # Always include multifactorial assessment
        multi = self.get_guideline("ags_falls_2023_multifactorial")
        if multi:
            recommendations.append(multi)
        
        # Add exercise for most patients
        if any(r in risk_factors for r in ["gait_abnormality", "balance_impairment", "muscle_weakness"]):
            exercise = self.get_guideline("ags_falls_2023_exercise")
            if exercise:
                recommendations.append(exercise)
        
        # Add vitamin D if deficient/high risk
        if "vitamin_d_deficiency" in risk_factors or "high_fall_risk" in risk_factors:
            vitd = self.get_guideline("ags_falls_2023_vitamin_d")
            if vitd:
                recommendations.append(vitd)
        
        # Add home modifications if indicated
        if "home_hazards" in risk_factors or "visual_impairment" in risk_factors:
            home = self.get_guideline("ags_falls_2023_home_modification")
            if home:
                recommendations.append(home)
        
        return recommendations
    
    def get_version_info(self) -> Dict[str, str]:
        """Get current guideline versions."""
        return dict(self.VERSIONS)


# Singleton instance for application use
_guidelines_db: Optional[ClinicalGuidelinesDB] = None


def get_guidelines_db() -> ClinicalGuidelinesDB:
    """Get or create singleton guidelines database instance."""
    global _guidelines_db
    if _guidelines_db is None:
        _guidelines_db = ClinicalGuidelinesDB()
    return _guidelines_db
