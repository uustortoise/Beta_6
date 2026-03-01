"""
Tests for citation ID registry strict routing behavior.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot.core.citation_validator import CitationValidator


def test_registry_rejects_malformed_pmid():
    validator = CitationValidator()
    result = validator._validate_citation("PMID:ABC123")

    assert result["valid"] is False
    assert "allowed registry formats" in result["reason"]


def test_registry_rejects_unknown_guideline_id():
    validator = CitationValidator()
    result = validator._validate_citation("ags_falls_2099_nonexistent")

    assert result["valid"] is False
    assert "ID not found in guideline registry" in result["reason"]


def test_registry_accepts_known_guideline_and_pmid():
    validator = CitationValidator()

    known_guideline = validator._validate_citation("beers_2023_anticholinergic")
    known_pmid = validator._validate_citation("PMID:29606030")

    assert known_guideline["valid"] is True
    assert known_guideline["source"] == "clinical_guidelines"
    assert known_pmid["valid"] is True
    assert known_pmid["source"] == "research_corpus"

