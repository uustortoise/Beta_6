"""
Health Advisory Chatbot Module

A comprehensive, evidence-based health advisory system for elderly care.
Integrates ADL history, medical records, ICOPE assessments, and sleep analysis
to provide predictive, research-grounded health recommendations.

Author: Beta 5.5 Engineering Team
Version: 1.0.0
Status: Development
"""

__version__ = "1.0.0"
__author__ = "Beta 5.5 Engineering Team"

from .core.advisory_engine import AdvisoryEngine
from .core.context_fusion import ContextFusionEngine
from .core.llm_service import LLMService
from .core.citation_validator import CitationValidator

__all__ = [
    "AdvisoryEngine",
    "ContextFusionEngine", 
    "LLMService",
    "CitationValidator",
]
