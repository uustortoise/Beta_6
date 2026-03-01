"""
Citation ID Registry

Centralized validation and source routing for citation IDs.
"""

from dataclasses import dataclass
import re
from typing import Dict, Optional, Set


@dataclass(frozen=True)
class CitationRoute:
    """Result of citation ID routing."""

    valid: bool
    normalized_id: str
    source: Optional[str] = None  # clinical_guidelines | research_corpus
    reason: Optional[str] = None


class CitationIDRegistry:
    """
    Validates citation ID formats and routes them to the appropriate source.

    Rules:
    - `PMID:<digits>`: research corpus ID
    - `DOI:<value>`: research corpus ID
    - `HASH:<hex>`: research corpus ID
    - `<guideline_id>`: must be known guideline key
    """

    _GUIDELINE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,128}$")
    _PMID_PATTERN = re.compile(r"^PMID:\d{4,10}$")
    _DOI_PATTERN = re.compile(r"^DOI:10\.\d{4,9}/\S+$")
    _HASH_PATTERN = re.compile(r"^HASH:[a-fA-F0-9]{8,64}$")

    def __init__(self, guidelines_db, research_corpus):
        self._guideline_ids = self._extract_guideline_ids(guidelines_db)
        self._research_ids = self._extract_research_ids(research_corpus)

    def route(self, citation_id: str) -> CitationRoute:
        """Validate and route citation ID to source."""
        cid = (citation_id or "").strip()
        if not cid:
            return CitationRoute(valid=False, normalized_id="", reason="Empty citation ID")

        if self._PMID_PATTERN.match(cid):
            if cid in self._research_ids:
                return CitationRoute(valid=True, normalized_id=cid, source="research_corpus")
            return CitationRoute(
                valid=False,
                normalized_id=cid,
                reason="PMID format valid but ID not found in research corpus",
            )

        if self._DOI_PATTERN.match(cid):
            if cid in self._research_ids:
                return CitationRoute(valid=True, normalized_id=cid, source="research_corpus")
            return CitationRoute(
                valid=False,
                normalized_id=cid,
                reason="DOI format valid but ID not found in research corpus",
            )

        if self._HASH_PATTERN.match(cid):
            if cid in self._research_ids:
                return CitationRoute(valid=True, normalized_id=cid, source="research_corpus")
            return CitationRoute(
                valid=False,
                normalized_id=cid,
                reason="HASH format valid but ID not found in research corpus",
            )

        if self._GUIDELINE_ID_PATTERN.match(cid):
            if cid in self._guideline_ids:
                return CitationRoute(valid=True, normalized_id=cid, source="clinical_guidelines")
            return CitationRoute(
                valid=False,
                normalized_id=cid,
                reason="Guideline ID format valid but ID not found in guideline registry",
            )

        return CitationRoute(
            valid=False,
            normalized_id=cid,
            reason="Citation ID does not match allowed registry formats",
        )

    @staticmethod
    def _extract_guideline_ids(guidelines_db) -> Set[str]:
        # Uses internal immutable store from ClinicalGuidelinesDB.
        ids = getattr(guidelines_db, "_guidelines", {}).keys()
        return set(ids)

    @staticmethod
    def _extract_research_ids(research_corpus) -> Set[str]:
        # Uses internal immutable store from ResearchCorpus.
        ids = getattr(research_corpus, "_papers", {}).keys()
        return set(ids)

    def stats(self) -> Dict[str, int]:
        """Registry counts for observability."""
        return {
            "guideline_ids": len(self._guideline_ids),
            "research_ids": len(self._research_ids),
        }

