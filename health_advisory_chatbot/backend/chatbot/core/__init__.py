"""
Core Advisory Engine Components

- ContextFusionEngine: Aggregates health data from all sources
- LLMService: RAG-augmented generation with evidence grounding
- CitationValidator: Verifies claims against medical knowledge
- AdvisoryEngine: Main orchestration component
"""

from chatbot.core.context_fusion import ContextFusionEngine, get_context_fusion_engine
from chatbot.core.llm_service import LLMService, get_llm_service
from chatbot.core.citation_validator import CitationValidator, get_citation_validator
from chatbot.core.policy_engine import ClinicalPolicyEngine, get_policy_engine
from chatbot.core.safety_gateway import SafetyGateway, UrgencyAssessment, get_safety_gateway
from chatbot.core.advisory_engine import AdvisoryEngine, get_advisory_engine

__all__ = [
    "ContextFusionEngine",
    "get_context_fusion_engine",
    "LLMService",
    "get_llm_service",
    "CitationValidator",
    "get_citation_validator",
    "ClinicalPolicyEngine",
    "get_policy_engine",
    "SafetyGateway",
    "UrgencyAssessment",
    "get_safety_gateway",
    "AdvisoryEngine",
    "get_advisory_engine",
]
