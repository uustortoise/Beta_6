"""
Medical Knowledge Base Module

Curated evidence sources for health advisory generation:
- Clinical guidelines (AGS, NICE, WHO)
- Drug interactions and safety
- Research paper corpus
- ICOPE standards
"""

from chatbot.knowledge_base.clinical_guidelines import ClinicalGuidelinesDB, get_guidelines_db
from chatbot.knowledge_base.drug_interactions import DrugInteractionDB, get_drug_interaction_db
from chatbot.knowledge_base.research_corpus import ResearchCorpus, get_research_corpus
from chatbot.knowledge_base.icope_standards import ICOPEStandards, get_icope_standards

__all__ = [
    "ClinicalGuidelinesDB",
    "get_guidelines_db",
    "DrugInteractionDB",
    "get_drug_interaction_db",
    "ResearchCorpus",
    "get_research_corpus",
    "ICOPEStandards",
    "get_icope_standards",
]
