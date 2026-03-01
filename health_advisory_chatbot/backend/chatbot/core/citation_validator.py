"""
Citation Validator

Validates medical claims against the knowledge base and ensures
accuracy of citations and evidence levels.

Features:
- Claim verification against guidelines
- Evidence level validation
- Citation resolution
- Confidence scoring
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re

from models.schemas import Citation, EvidenceLevel, AdvisoryRecommendation
from chatbot.core.citation_registry import CitationIDRegistry


@dataclass
class ValidationResult:
    """Result of citation validation."""
    is_valid: bool
    verified_claims: List[Dict[str, Any]]
    unverified_claims: List[Dict[str, Any]]
    corrected_citations: List[Citation]
    overall_confidence: float
    warnings: List[str]


class CitationValidator:
    """
    Validates citations and evidence in advisory responses.
    
    Ensures:
    - Claims are supported by evidence
    - Citations resolve to valid sources
    - Evidence levels are appropriate
    - No hallucinated citations
    """
    
    def __init__(self):
        """Initialize citation validator."""
        self.knowledge_bases = self._initialize_knowledge_bases()
        self.citation_registry = CitationIDRegistry(
            guidelines_db=self.knowledge_bases["guidelines"],
            research_corpus=self.knowledge_bases["research"],
        )
    
    def _initialize_knowledge_bases(self) -> Dict:
        """Initialize connections to knowledge bases."""
        from chatbot.knowledge_base import (
            ClinicalGuidelinesDB,
            ResearchCorpus,
            get_guidelines_db,
            get_research_corpus,
        )
        
        return {
            "guidelines": get_guidelines_db(),
            "research": get_research_corpus(),
        }
    
    def validate_response(
        self,
        response_text: str,
        citations_found: List[str],
    ) -> ValidationResult:
        """
        Validate advisory response citations.
        
        Args:
            response_text: Generated response text
            citations_found: List of citation IDs found in text
        
        Returns:
            ValidationResult with verification status
        """
        verified_claims = []
        unverified_claims = []
        corrected_citations = []
        warnings = []
        
        # Validate each citation
        for citation_id in citations_found:
            validation = self._validate_citation(citation_id)
            
            if validation["valid"]:
                corrected_citations.append(validation["citation"])
                verified_claims.append({
                    "citation_id": citation_id,
                    "source": validation["source"],
                    "evidence_level": validation["evidence_level"],
                })
            else:
                unverified_claims.append({
                    "citation_id": citation_id,
                    "reason": validation["reason"],
                })
                warnings.append(f"Unverified citation: {citation_id}")
        
        # Check for common medical claim issues
        medical_claims = self._extract_medical_claims(response_text)
        for claim in medical_claims:
            claim_validation = self._validate_medical_claim(claim)
            if not claim_validation["valid"]:
                warnings.append(f"Questionable claim: {claim}")
        
        # Calculate overall confidence
        if not citations_found:
            overall_confidence = 30.0  # Low confidence without citations
        else:
            verified_ratio = len(verified_claims) / len(citations_found)
            base_confidence = verified_ratio * 100
            
            # Adjust for evidence quality
            evidence_bonus = sum(
                10 if c.evidence_level in [EvidenceLevel.SYSTEMATIC_REVIEW, EvidenceLevel.RCT] else 5
                for c in corrected_citations
            ) / max(len(corrected_citations), 1)
            
            overall_confidence = min(95, base_confidence + evidence_bonus)
        
        # Fail closed for citation validity: any unverified citation invalidates response.
        is_valid = len(unverified_claims) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            verified_claims=verified_claims,
            unverified_claims=unverified_claims,
            corrected_citations=corrected_citations,
            overall_confidence=round(overall_confidence, 1),
            warnings=warnings,
        )
    
    def _validate_citation(self, citation_id: str) -> Dict:
        """
        Validate a single citation ID.
        
        Checks:
        - Clinical guidelines database
        - Research corpus
        - Known citation patterns
        """
        citation_id_clean = citation_id.strip()
        
        route = self.citation_registry.route(citation_id_clean)
        if not route.valid:
            return {
                "valid": False,
                "reason": route.reason or "Citation ID registry validation failed",
                "citation": None,
            }

        if route.source == "clinical_guidelines":
            guideline = self.knowledge_bases["guidelines"].get_guideline(route.normalized_id)
            if guideline:
                return {
                    "valid": True,
                    "source": "clinical_guidelines",
                    "evidence_level": guideline.evidence_grade,
                    "citation": Citation(
                        source_type="clinical_guideline",
                        title=guideline.title,
                        authors=[guideline.source.value],
                        year=guideline.publication_year,
                        evidence_level=EvidenceLevel.CLINICAL_GUIDELINE,
                        confidence_score=90.0,
                    ),
                }
            return {
                "valid": False,
                "reason": "Guideline ID routed but lookup failed",
                "citation": None,
            }

        if route.source == "research_corpus":
            paper = self.knowledge_bases["research"].get_paper(route.normalized_id)
            if paper:
                return {
                    "valid": True,
                    "source": "research_corpus",
                    "evidence_level": paper.evidence_level,
                    "citation": paper.to_citation(),
                }
            return {
                "valid": False,
                "reason": "Research ID routed but lookup failed",
                "citation": None,
            }

        # Unknown routed source
        return {
            "valid": False,
            "reason": "Citation source routing failed",
            "citation": None,
        }

    def _extract_medical_claims(self, text: str) -> List[str]:
        """
        Extract potential medical claims from text.
        
        Looks for patterns like:
        - "X increases Y"
        - "X reduces risk of Y"
        - "X is recommended for Y"
        """
        claims = []
        
        # Pattern: X reduces/increases Y
        patterns = [
            r'([^.]*(?:reduces|increases|decreases|lowers|raises|risk)[^.]*\.)',
            r'([^.]*(?:recommended|indicated|contraindicated)[^.]*\.)',
            r'([^.]*(?:effective|beneficial|harmful)[^.]*\.)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            claims.extend(matches)
        
        return [c.strip() for c in claims if len(c) > 20]
    
    def _validate_medical_claim(self, claim: str) -> Dict:
        """
        Validate a medical claim against guidelines.
        
        Returns validity assessment.
        """
        claim_lower = claim.lower()
        
        # Check for common evidence-based claims
        validated_claims = {
            "exercise reduces fall risk": {
                "valid": True,
                "evidence": "AGS Guidelines 2023 - Exercise reduces falls 30%",
            },
            "vitamin d reduces fall risk": {
                "valid": True,
                "evidence": "AGS Guidelines 2023 - 800 IU/day reduces falls",
            },
            "benzodiazepines increase fall risk": {
                "valid": True,
                "evidence": "AGS Beers Criteria - 44% increased fall risk",
            },
            "sleep apnea causes cognitive decline": {
                "valid": True,
                "evidence": "Multiple studies on hypoxia and cognition",
            },
        }
        
        for key, validation in validated_claims.items():
            if all(word in claim_lower for word in key.split()):
                return validation
        
        # Default - claim not verified
        return {
            "valid": None,  # Unknown
            "evidence": None,
        }
    
    def enrich_citations(
        self,
        citations: List[Citation],
    ) -> List[Citation]:
        """
        Enrich citations with additional metadata.
        
        Adds:
        - Full citation text
        - Access URLs
        - Related citations
        """
        enriched = []
        
        for citation in citations:
            # Build full citation text
            full_citation = self._build_full_citation(citation)
            
            # Add URL if available
            url = None
            if citation.pmid:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/"
            elif citation.doi:
                url = f"https://doi.org/{citation.doi}"
            
            enriched.append(Citation(
                source_type=citation.source_type,
                title=citation.title,
                authors=citation.authors,
                journal=citation.journal,
                year=citation.year,
                doi=citation.doi,
                url=url,
                pmid=citation.pmid,
                evidence_level=citation.evidence_level,
                confidence_score=citation.confidence_score,
            ))
        
        return enriched
    
    def _build_full_citation(self, citation: Citation) -> str:
        """Build full citation text in APA style."""
        authors_text = ", ".join(citation.authors[:3])
        if len(citation.authors) > 3:
            authors_text += " et al."
        
        parts = [
            authors_text,
            f"({citation.year})." if citation.year else "",
            citation.title,
            f"*{citation.journal}*." if citation.journal else "",
        ]
        
        return " ".join(p for p in parts if p)
    
    def calculate_evidence_quality_score(
        self,
        citations: List[Citation],
    ) -> Dict[str, Any]:
        """
        Calculate overall evidence quality score.
        
        Returns:
            Quality metrics
        """
        if not citations:
            return {
                "score": 0,
                "level": "insufficient",
                "description": "No citations provided",
            }
        
        # Score by evidence level
        level_scores = {
            EvidenceLevel.SYSTEMATIC_REVIEW: 100,
            EvidenceLevel.RCT: 90,
            EvidenceLevel.COHORT_STUDY: 70,
            EvidenceLevel.CASE_CONTROL: 50,
            EvidenceLevel.EXPERT_OPINION: 30,
            EvidenceLevel.CLINICAL_GUIDELINE: 95,
            EvidenceLevel.MANUFACTURER_DATA: 40,
        }
        
        scores = []
        for citation in citations:
            score = level_scores.get(citation.evidence_level, 40)
            scores.append(score)
        
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= 80:
            level = "high"
        elif avg_score >= 60:
            level = "moderate"
        else:
            level = "low"
        
        return {
            "score": round(avg_score, 1),
            "level": level,
            "description": f"{level.title()} quality evidence ({len(citations)} sources)",
            "breakdown": {
                "systematic_reviews": sum(1 for c in citations if c.evidence_level == EvidenceLevel.SYSTEMATIC_REVIEW),
                "rcts": sum(1 for c in citations if c.evidence_level == EvidenceLevel.RCT),
                "guidelines": sum(1 for c in citations if c.evidence_level == EvidenceLevel.CLINICAL_GUIDELINE),
                "other": sum(1 for c in citations if c.evidence_level not in [
                    EvidenceLevel.SYSTEMATIC_REVIEW, EvidenceLevel.RCT, EvidenceLevel.CLINICAL_GUIDELINE
                ]),
            },
        }


# Singleton
_citation_validator: Optional[CitationValidator] = None


def get_citation_validator() -> CitationValidator:
    """Get or create singleton citation validator."""
    global _citation_validator
    if _citation_validator is None:
        _citation_validator = CitationValidator()
    return _citation_validator
