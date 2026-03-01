"""
Tests for citation validation fail-closed behavior.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot.core.citation_validator import CitationValidator


def test_rejects_unverified_external_literature_pattern():
    validator = CitationValidator()
    result = validator.validate_response(
        response_text="Exercise helps reduce fall risk [Source: Smith et al. 2024].",
        citations_found=["Smith et al. 2024"],
    )

    assert result.is_valid is False
    assert len(result.unverified_claims) == 1
    assert "does not match allowed registry formats" in result.unverified_claims[0]["reason"]
    assert len(result.corrected_citations) == 0


def test_mixed_valid_and_invalid_citations_fails_closed():
    validator = CitationValidator()
    result = validator.validate_response(
        response_text=(
            "Recommendation A [Source: beers_2023_anticholinergic]. "
            "Recommendation B [Source: Doe et al. 2023]."
        ),
        citations_found=["beers_2023_anticholinergic", "Doe et al. 2023"],
    )

    assert len(result.verified_claims) == 1
    assert len(result.unverified_claims) == 1
    assert result.is_valid is False


def test_known_guideline_and_pmid_citations_remain_valid():
    validator = CitationValidator()
    result = validator.validate_response(
        response_text=(
            "Medication warning [Source: beers_2023_anticholinergic]. "
            "Evidence from review [Source: PMID:29606030]."
        ),
        citations_found=["beers_2023_anticholinergic", "PMID:29606030"],
    )

    assert result.is_valid is True
    assert len(result.unverified_claims) == 0
    assert len(result.corrected_citations) == 2
