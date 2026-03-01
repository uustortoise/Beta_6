"""
Drug Interaction Database

Comprehensive drug-drug and drug-condition interaction data for elderly patients.
Sources: OpenFDA, Micromedex, Beers Criteria, primary literature

Provides:
- Drug-drug interaction checking
- Drug-condition contraindications
- Age-related pharmacokinetic considerations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import json
from pathlib import Path


class InteractionSeverity(str, Enum):
    """Drug interaction severity classification."""
    CONTRAINDICATED = "contraindicated"  # Should not be used together
    MAJOR = "major"  # May be life-threatening or require medical intervention
    MODERATE = "moderate"  # May cause significant issues, may need monitoring
    MINOR = "minor"  # Limited clinical effects
    UNKNOWN = "unknown"


class InteractionMechanism(str, Enum):
    """Mechanism of drug interaction."""
    PHARMACOKINETIC = "pharmacokinetic"  # Absorption, distribution, metabolism, excretion
    PHARMACODYNAMIC = "pharmacodynamic"  # Additive/synergistic/antagonistic effects
    PHARMACEUTICAL = "pharmaceutical"  # Incompatibility


@dataclass(frozen=True)
class DrugInteraction:
    """
    Individual drug interaction record.
    
    Immutable to ensure data integrity.
    """
    drug_a: str
    drug_b: str
    severity: InteractionSeverity
    mechanism: InteractionMechanism
    
    # Clinical details
    description: str
    clinical_effects: List[str] = field(default_factory=list)
    
    # Management
    management: str = ""
    monitoring_parameters: List[str] = field(default_factory=list)
    
    # Evidence
    evidence_quality: str = "moderate"  # high/moderate/low
    onset: str = "rapid"  # rapid/delayed
    documentation_level: str = "good"  # excellent/good/fair/poor


@dataclass(frozen=True)
class DrugConditionContraindication:
    """Drug contraindication with specific medical conditions."""
    drug: str
    condition: str
    severity: InteractionSeverity
    description: str
    alternative_drugs: List[str] = field(default_factory=list)


class DrugInteractionDB:
    """
    Drug interaction database for elderly care.
    
    Provides comprehensive interaction checking including:
    - Drug-drug interactions
    - Drug-condition contraindications
    - Age-specific considerations
    
    Data sources:
    - Beers Criteria (AGS)
    - STOPP/START Criteria
    - Lexicomp/Micromedex (structure)
    - Primary literature
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize drug interaction database.
        
        Args:
            data_path: Path to external interaction data files
        """
        self._interactions: Dict[str, List[DrugInteraction]] = {}
        self._contraindications: Dict[str, List[DrugConditionContraindication]] = {}
        self._drug_classes: Dict[str, List[str]] = {}  # class -> drugs
        
        # Load built-in data
        self._load_built_in_interactions()
        self._load_built_in_contraindications()
        
        # Load external data if provided
        if data_path:
            self._load_external_data(data_path)
    
    def _add_interaction(self, interaction: DrugInteraction) -> None:
        """Add interaction to indexes."""
        # Index by both drug names
        key_a = interaction.drug_a.lower()
        key_b = interaction.drug_b.lower()
        
        if key_a not in self._interactions:
            self._interactions[key_a] = []
        self._interactions[key_a].append(interaction)
        
        if key_b not in self._interactions:
            self._interactions[key_b] = []
        self._interactions[key_b].append(interaction)
    
    def _load_built_in_interactions(self) -> None:
        """Load curated high-priority interactions for elderly."""
        
        # === ANTICOAGULANT INTERACTIONS ===
        anticoagulant_interactions = [
            DrugInteraction(
                drug_a="warfarin",
                drug_b="aspirin",
                severity=InteractionSeverity.MAJOR,
                mechanism=InteractionMechanism.PHARMACODYNAMIC,
                description="Increased bleeding risk. Antiplatelet effect of aspirin adds to anticoagulation.",
                clinical_effects=["GI bleeding", "intracranial hemorrhage", "prolonged bleeding"],
                management="Avoid combination unless clear indication (e.g., mechanical heart valve). If used together, monitor INR closely and use GI protection.",
                monitoring_parameters=["INR", "bleeding signs", "hemoglobin", "stool occult blood"],
                evidence_quality="high",
                onset="rapid",
            ),
            DrugInteraction(
                drug_a="warfarin",
                drug_b="amiodarone",
                severity=InteractionSeverity.MAJOR,
                mechanism=InteractionMechanism.PHARMACOKINETIC,
                description="Amiodarone inhibits CYP2C9 and CYP1A2, increasing warfarin levels significantly.",
                clinical_effects=["supratherapeutic INR", "bleeding", "bruising"],
                management="Reduce warfarin dose by 30-50% when starting amiodarone. Monitor INR weekly.",
                monitoring_parameters=["INR"],
                evidence_quality="high",
                onset="delayed",
            ),
            DrugInteraction(
                drug_a="warfarin",
                drug_b="metronidazole",
                severity=InteractionSeverity.MAJOR,
                mechanism=InteractionMechanism.PHARMACOKINETIC,
                description="Metronidazole inhibits warfarin metabolism, increasing INR.",
                clinical_effects=["bleeding", "supratherapeutic INR"],
                management="Reduce warfarin dose temporarily or hold during antibiotic course. Check INR 3-5 days after starting.",
                monitoring_parameters=["INR"],
                evidence_quality="high",
            ),
        ]
        
        for interaction in anticoagulant_interactions:
            self._add_interaction(interaction)
        
        # === DIABETES MEDICATION INTERACTIONS ===
        diabetes_interactions = [
            DrugInteraction(
                drug_a="metformin",
                drug_b="contrast dye",
                severity=InteractionSeverity.MAJOR,
                mechanism=InteractionMechanism.PHARMACODYNAMIC,
                description="Increased risk of lactic acidosis, especially with renal impairment.",
                clinical_effects=["lactic acidosis", "renal failure"],
                management="Hold metformin 48 hours before and after contrast procedure if eGFR < 60. Monitor renal function.",
                monitoring_parameters=["eGFR", "serum creatinine", "lactate"],
                evidence_quality="high",
            ),
            DrugInteraction(
                drug_a="insulin",
                drug_b="beta-blocker",
                severity=InteractionSeverity.MODERATE,
                mechanism=InteractionMechanism.PHARMACODYNAMIC,
                description="Beta-blockers mask hypoglycemia symptoms (tremors, tachycardia) and may prolong hypoglycemia.",
                clinical_effects=["unrecognized hypoglycemia", "prolonged hypoglycemia"],
                management="Monitor blood glucose closely. Selective beta-blockers (metoprolol) preferred over non-selective.",
                monitoring_parameters=["blood glucose", "hypoglycemia symptoms"],
                evidence_quality="high",
            ),
            DrugInteraction(
                drug_a="sulfonylurea",
                drug_b="fluconazole",
                severity=InteractionSeverity.MAJOR,
                mechanism=InteractionMechanism.PHARMACOKINETIC,
                description="Fluconazole inhibits CYP2C9, increasing sulfonylurea levels and hypoglycemia risk.",
                clinical_effects=["severe hypoglycemia", "seizures", "coma"],
                management="Reduce sulfonylurea dose or temporarily discontinue. Monitor glucose closely.",
                monitoring_parameters=["blood glucose", "hypoglycemia symptoms"],
                evidence_quality="high",
            ),
        ]
        
        for interaction in diabetes_interactions:
            self._add_interaction(interaction)
        
        # === CARDIOVASCULAR INTERACTIONS ===
        cv_interactions = [
            DrugInteraction(
                drug_a="amlodipine",
                drug_b="simvastatin",
                severity=InteractionSeverity.MODERATE,
                mechanism=InteractionMechanism.PHARMACOKINETIC,
                description="Amlodipine inhibits simvastatin metabolism, increasing risk of myopathy/rhabdomyolysis.",
                clinical_effects=["muscle pain", "myopathy", "rhabdomyolysis", "elevated CK"],
                management="Limit simvastatin dose to 20mg daily when used with amlodipine. Consider pravastatin or rosuvastatin instead.",
                monitoring_parameters=["CK", "muscle symptoms"],
                evidence_quality="high",
            ),
            DrugInteraction(
                drug_a="digoxin",
                drug_b="amiodarone",
                severity=InteractionSeverity.MAJOR,
                mechanism=InteractionMechanism.PHARMACOKINETIC,
                description="Amiodarone increases digoxin levels by ~70%, risking toxicity.",
                clinical_effects=["nausea", "arrhythmias", "confusion", "visual disturbances", "hyperkalemia"],
                management="Reduce digoxin dose by 50% when starting amiodarone. Monitor levels.",
                monitoring_parameters=["digoxin level", "potassium", "ECG", "renal function"],
                evidence_quality="high",
            ),
            DrugInteraction(
                drug_a="ACE inhibitor",
                drug_b="potassium supplement",
                severity=InteractionSeverity.MAJOR,
                mechanism=InteractionMechanism.PHARMACODYNAMIC,
                description="Additive risk of hyperkalemia, especially with renal impairment.",
                clinical_effects=["hyperkalemia", "arrhythmias", "cardiac arrest"],
                management="Avoid routine potassium supplementation unless documented deficiency. Monitor potassium closely.",
                monitoring_parameters=["serum potassium", "renal function"],
                evidence_quality="high",
            ),
        ]
        
        for interaction in cv_interactions:
            self._add_interaction(interaction)
        
        # === CNS/PSYCHIATRIC INTERACTIONS ===
        cns_interactions = [
            DrugInteraction(
                drug_a="SSRI",
                drug_b="tramadol",
                severity=InteractionSeverity.MAJOR,
                mechanism=InteractionMechanism.PHARMACODYNAMIC,
                description="Risk of serotonin syndrome. SSRIs inhibit serotonin reuptake, tramadol has serotonergic activity.",
                clinical_effects=["serotonin syndrome", "hyperthermia", "confusion", "tremor", "diaphoresis"],
                management="Avoid combination if possible. Use alternative analgesics (acetaminophen, NSAIDs with caution).",
                monitoring_parameters=["mental status", "vital signs", "muscle rigidity"],
                evidence_quality="moderate",
            ),
            DrugInteraction(
                drug_a="warfarin",
                drug_b="SSRI",
                severity=InteractionSeverity.MODERATE,
                mechanism=InteractionMechanism.PHARMACODYNAMIC,
                description="SSRIs impair platelet function (serotonin-mediated), increasing bleeding risk with anticoagulants.",
                clinical_effects=["GI bleeding", "bruising", "prolonged bleeding"],
                management="Monitor closely. Consider GI protection (PPI) if high GI bleed risk.",
                monitoring_parameters=["bleeding signs", "stool occult blood", "hemoglobin"],
                evidence_quality="high",
            ),
            DrugInteraction(
                drug_a="benzodiazepine",
                drug_b="opioid",
                severity=InteractionSeverity.MAJOR,
                mechanism=InteractionMechanism.PHARMACODYNAMIC,
                description="Profound sedation, respiratory depression, coma, and death. FDA Black Box Warning.",
                clinical_effects=["respiratory depression", "coma", "death", "profound sedation"],
                management="Avoid concurrent use when possible. If necessary, use lowest doses, short duration, and monitor closely.",
                monitoring_parameters=["respiratory rate", "sedation level", "oxygen saturation"],
                evidence_quality="high",
            ),
            DrugInteraction(
                drug_a="benzodiazepine",
                drug_b="alcohol",
                severity=InteractionSeverity.MAJOR,
                mechanism=InteractionMechanism.PHARMACODYNAMIC,
                description="Additive CNS depression, increased risk of falls, respiratory depression, and death.",
                clinical_effects=["falls", "respiratory depression", "coma", "death"],
                management="Counsel patients to avoid alcohol completely while taking benzodiazepines.",
                monitoring_parameters=["fall risk", "cognitive status"],
                evidence_quality="high",
            ),
        ]
        
        for interaction in cns_interactions:
            self._add_interaction(interaction)
        
        # === ANTICHOLINERGIC BURDEN INTERACTIONS ===
        anticholinergic_interactions = [
            DrugInteraction(
                drug_a="oxybutynin",
                drug_b="diphenhydramine",
                severity=InteractionSeverity.MODERATE,
                mechanism=InteractionMechanism.PHARMACODYNAMIC,
                description="Additive anticholinergic effects. High anticholinergic burden in elderly causes cognitive impairment, constipation, urinary retention.",
                clinical_effects=["confusion", "delirium", "constipation", "urinary retention", "dry mouth", "blurred vision"],
                management="Avoid combination. Reduce anticholinergic burden. Consider alternatives (mirabegron for bladder, non-sedating antihistamines).",
                monitoring_parameters=["cognitive status", "bowel function", "urinary output"],
                evidence_quality="high",
            ),
        ]
        
        for interaction in anticholinergic_interactions:
            self._add_interaction(interaction)
        
        # Define drug classes for class-level checking
        self._drug_classes = {
            "ssri": ["sertraline", "fluoxetine", "paroxetine", "citalopram", "escitalopram", "fluvoxamine"],
            "snri": ["venlafaxine", "duloxetine", "desvenlafaxine"],
            "benzodiazepine": ["diazepam", "lorazepam", "alprazolam", "temazepam", "clonazepam", "oxazepam"],
            "ace_inhibitor": ["lisinopril", "enalapril", "captopril", "ramipril", "perindopril"],
            "arb": ["losartan", "valsartan", "irbesartan", "candesartan", "olmesartan"],
            "statin": ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin"],
            "sulfonylurea": ["glyburide", "glipizide", "glimepiride"],
            "beta_blocker": ["metoprolol", "atenolol", "propranolol", "carvedilol", "bisoprolol"],
            "nsaid": ["ibuprofen", "naproxen", "diclofenac", "celecoxib"],
            "opioid": ["morphine", "oxycodone", "hydrocodone", "tramadol", "fentanyl", "codeine"],
        }
    
    def _load_built_in_contraindications(self) -> None:
        """Load drug-condition contraindications."""
        
        contraindications = [
            # Anticholinergics in cognitive impairment
            DrugConditionContraindication(
                drug="diphenhydramine",
                condition="dementia",
                severity=InteractionSeverity.MAJOR,
                description="Anticholinergics worsen cognitive function and increase dementia progression risk.",
                alternative_drugs=["non-pharmacologic sleep measures", "low-dose trazodone"],
            ),
            DrugConditionContraindication(
                drug="oxybutynin",
                condition="dementia",
                severity=InteractionSeverity.MAJOR,
                description="Anticholinergic burden worsens cognition in dementia patients.",
                alternative_drugs=["mirabegron", "trospium", "behavioral therapy"],
            ),
            
            # NSAIDs in heart failure
            DrugConditionContraindication(
                drug="ibuprofen",
                condition="heart_failure",
                severity=InteractionSeverity.MAJOR,
                description="NSAIDs cause sodium/water retention and worsen heart failure.",
                alternative_drugs=["acetaminophen", "topical NSAIDs"],
            ),
            
            # Metformin in renal impairment
            DrugConditionContraindication(
                drug="metformin",
                condition="severe_renal_impairment",
                severity=InteractionSeverity.CONTRAINDICATED,
                description="High risk of lactic acidosis when eGFR < 30 mL/min/1.73m².",
                alternative_drugs=["insulin", "DPP-4 inhibitors"],
            ),
            
            # Benzodiazepines in fall risk
            DrugConditionContraindication(
                drug="benzodiazepine",
                condition="high_fall_risk",
                severity=InteractionSeverity.MAJOR,
                description="Benzodiazepines significantly increase fall and fracture risk.",
                alternative_drugs=["non-pharmacologic anxiety treatment", "SSRIs (with caution)"],
            ),
            
            # Anticoagulants in bleeding disorders
            DrugConditionContraindication(
                drug="warfarin",
                condition="active_bleeding",
                severity=InteractionSeverity.CONTRAINDICATED,
                description="Active bleeding is absolute contraindication to anticoagulation.",
                alternative_drugs=["treat bleeding first, then reassess"],
            ),
            
            # Sulfonylureas in elderly
            DrugConditionContraindication(
                drug="glyburide",
                condition="advanced_age",
                severity=InteractionSeverity.MAJOR,
                description="Long-acting sulfonylureas cause prolonged hypoglycemia in elderly.",
                alternative_drugs=["glipizide", "metformin", "DPP-4 inhibitors"],
            ),
        ]
        
        for contraindication in contraindications:
            drug_key = contraindication.drug.lower()
            if drug_key not in self._contraindications:
                self._contraindications[drug_key] = []
            self._contraindications[drug_key].append(contraindication)
    
    def _load_external_data(self, data_path: Path) -> None:
        """Load external interaction data from JSON files."""
        if not data_path.exists():
            return
        
        for json_file in data_path.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                    # Load drug-drug interactions
                    for item in data.get("interactions", []):
                        interaction = DrugInteraction(
                            drug_a=item["drug_a"],
                            drug_b=item["drug_b"],
                            severity=InteractionSeverity(item["severity"]),
                            mechanism=InteractionMechanism(item["mechanism"]),
                            description=item["description"],
                            clinical_effects=item.get("clinical_effects", []),
                            management=item.get("management", ""),
                            monitoring_parameters=item.get("monitoring", []),
                            evidence_quality=item.get("evidence_quality", "moderate"),
                            onset=item.get("onset", "rapid"),
                            documentation_level=item.get("documentation", "good"),
                        )
                        self._add_interaction(interaction)
                    
            except Exception as e:
                print(f"Error loading interaction file {json_file}: {e}")
    
    def check_interaction(
        self,
        drug_a: str,
        drug_b: str
    ) -> Optional[DrugInteraction]:
        """
        Check for interaction between two specific drugs.
        
        Args:
            drug_a: First drug name
            drug_b: Second drug name
        
        Returns:
            DrugInteraction if found, None otherwise
        """
        key_a = drug_a.lower()
        key_b = drug_b.lower()
        
        # Direct lookup
        interactions = self._interactions.get(key_a, [])
        for interaction in interactions:
            if interaction.drug_b.lower() == key_b or interaction.drug_a.lower() == key_b:
                return interaction
        
        # Check drug classes
        for class_name, drugs in self._drug_classes.items():
            if key_a in [d.lower() for d in drugs]:
                # Check if drug_b interacts with any drug in this class
                for drug in drugs:
                    interactions = self._interactions.get(drug.lower(), [])
                    for interaction in interactions:
                        if (interaction.drug_a.lower() == drug.lower() and interaction.drug_b.lower() == key_b) or \
                           (interaction.drug_b.lower() == drug.lower() and interaction.drug_a.lower() == key_b):
                            return interaction
        
        return None
    
    def check_all_interactions(
        self,
        medications: List[str]
    ) -> List[DrugInteraction]:
        """
        Check all pairwise interactions in a medication list.
        
        Args:
            medications: List of medication names
        
        Returns:
            List of all identified interactions
        """
        interactions = []
        seen_pairs = set()
        
        meds_lower = [m.lower() for m in medications]
        
        for i, drug_a in enumerate(meds_lower):
            for drug_b in meds_lower[i+1:]:
                pair_key = tuple(sorted([drug_a, drug_b]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                interaction = self.check_interaction(drug_a, drug_b)
                if interaction:
                    interactions.append(interaction)
        
        # Sort by severity
        severity_order = {
            InteractionSeverity.CONTRAINDICATED: 0,
            InteractionSeverity.MAJOR: 1,
            InteractionSeverity.MODERATE: 2,
            InteractionSeverity.MINOR: 3,
        }
        interactions.sort(key=lambda x: severity_order.get(x.severity, 4))
        
        return interactions
    
    def check_contraindications(
        self,
        medications: List[str],
        conditions: List[str]
    ) -> List[DrugConditionContraindication]:
        """
        Check for drug-condition contraindications.
        
        Args:
            medications: Current medications
            conditions: Patient's medical conditions
        
        Returns:
            List of contraindications
        """
        contraindications = []
        
        for med in medications:
            med_lower = med.lower()
            
            # Direct drug lookup
            drug_contras = self._contraindications.get(med_lower, [])
            for contra in drug_contras:
                if contra.condition.lower() in [c.lower() for c in conditions]:
                    contraindications.append(contra)
            
            # Check drug classes
            for class_name, drugs in self._drug_classes.items():
                if med_lower in [d.lower() for d in drugs]:
                    class_contras = self._contraindications.get(class_name.lower(), [])
                    for contra in class_contras:
                        if contra.condition.lower() in [c.lower() for c in conditions]:
                            contraindications.append(contra)
        
        return contraindications
    
    def get_anticholinergic_burden(
        self,
        medications: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Calculate anticholinergic burden score (ACB).
        
        Uses modified ACB scale:
        - Score 3: High anticholinergic activity
        - Score 2: Moderate activity
        - Score 1: Low activity
        - Score 0: None
        
        Returns:
            Tuple of (total score, list of contributing medications)
        """
        # ACB scoring
        acb_scores = {
            # Score 3
            "amitriptyline": 3,
            "paroxetine": 3,
            "clozapine": 3,
            "olanzapine": 3,
            "promethazine": 3,
            
            # Score 2
            "nortriptyline": 2,
            "imipramine": 2,
            "oxybutynin": 2,
            "tolterodine": 2,
            "diphenhydramine": 2,
            "chlorpheniramine": 2,
            "hydroxyzine": 2,
            "mirtazapine": 2,
            
            # Score 1
            "sertraline": 1,
            "citalopram": 1,
            "escitalopram": 1,
            "fluoxetine": 1,
            "fluvoxamine": 1,
            "loratadine": 1,
            "fexofenadine": 1,
            "ranitidine": 1,  # Though largely replaced by PPIs
        }
        
        total_score = 0.0
        contributing_meds = []
        
        for med in medications:
            med_lower = med.lower()
            score = acb_scores.get(med_lower, 0)
            if score > 0:
                total_score += score
                contributing_meds.append(f"{med} (ACB={score})")
        
        return total_score, contributing_meds
    
    def assess_medication_risk(
        self,
        medications: List[str],
        conditions: List[str],
        age: int
    ) -> Dict[str, any]:
        """
        Comprehensive medication risk assessment.
        
        Args:
            medications: All current medications
            conditions: Medical conditions
            age: Patient age
        
        Returns:
            Risk assessment summary
        """
        # Check drug-drug interactions
        interactions = self.check_all_interactions(medications)
        
        # Check contraindications
        contraindications = self.check_contraindications(medications, conditions)
        
        # Calculate anticholinergic burden
        acb_score, acb_meds = self.get_anticholinergic_burden(medications)
        
        # Count high-risk medications (Beers criteria)
        beers_drugs = [
            "glyburide", "chlorpropamide",  # Hypoglycemia risk
            "diphenhydramine", "amitriptyline", "paroxetine",  # Anticholinergic
            "diazepam", "lorazepam", "alprazolam",  # Benzodiazepines
            "indomethacin", "ketorolac",  # NSAIDs
            "methocarbamol", "carisoprodol",  # Muscle relaxants
            "meperidine",  # Opioid
        ]
        
        beers_count = sum(1 for med in medications if any(bd in med.lower() for bd in beers_drugs))
        
        # Risk stratification
        risk_score = 0
        risk_factors = []
        
        if interactions:
            major_count = sum(1 for i in interactions if i.severity == InteractionSeverity.MAJOR)
            contra_count = sum(1 for i in interactions if i.severity == InteractionSeverity.CONTRAINDICATED)
            risk_score += major_count * 10 + contra_count * 20
            risk_factors.append(f"{len(interactions)} drug interactions identified")
        
        if contraindications:
            risk_score += len(contraindications) * 15
            risk_factors.append(f"{len(contraindications)} drug-condition contraindications")
        
        if acb_score >= 3:
            risk_score += 15
            risk_factors.append(f"High anticholinergic burden (ACB={acb_score})")
        elif acb_score >= 1:
            risk_score += 5
            risk_factors.append(f"Moderate anticholinergic burden (ACB={acb_score})")
        
        if beers_count >= 3:
            risk_score += 10
            risk_factors.append(f"Multiple potentially inappropriate medications (n={beers_count})")
        
        if len(medications) >= 10:
            risk_score += 10
            risk_factors.append("Polypharmacy (≥10 medications)")
        elif len(medications) >= 5:
            risk_score += 5
            risk_factors.append("Hyper-polypharmacy (≥5 medications)")
        
        # Determine risk level
        if risk_score >= 50:
            risk_level = "HIGH"
        elif risk_score >= 25:
            risk_level = "MODERATE"
        elif risk_score >= 10:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "interactions": [
                {
                    "drugs": [i.drug_a, i.drug_b],
                    "severity": i.severity.value,
                    "description": i.description,
                    "management": i.management,
                }
                for i in interactions[:5]  # Top 5
            ],
            "contraindications": [
                {
                    "drug": c.drug,
                    "condition": c.condition,
                    "severity": c.severity.value,
                    "alternatives": c.alternative_drugs,
                }
                for c in contraindications
            ],
            "anticholinergic_burden": {
                "score": acb_score,
                "contributing_medications": acb_meds,
                "interpretation": "High" if acb_score >= 3 else "Moderate" if acb_score >= 1 else "Low",
            },
            "beers_criteria_violations": beers_count,
            "medication_count": len(medications),
        }


# Singleton instance
_interaction_db: Optional[DrugInteractionDB] = None


def get_drug_interaction_db() -> DrugInteractionDB:
    """Get or create singleton drug interaction database."""
    global _interaction_db
    if _interaction_db is None:
        _interaction_db = DrugInteractionDB()
    return _interaction_db
