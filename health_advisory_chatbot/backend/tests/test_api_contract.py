"""
Contract tests for /api/chat response payload shape.

Canonical API contract is snake_case.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes import ChatbotAPI


def test_chat_response_contract_uses_snake_case():
    api = ChatbotAPI()

    result = asyncio.run(
        api.chat(
            {
                "elder_id": "elder_contract_001",
                "message": "I have chest pain and can't breathe.",
                "language": "en",
            }
        )
    )

    assert result["success"] is True
    assert "session_id" in result
    assert "new_risk_alerts" in result
    assert "response_time_ms" in result

    # Canonical contract keys
    assert "sessionId" not in result
    assert "newRiskAlerts" not in result
    assert "responseTimeMs" not in result

    message = result["message"]
    assert "risk_alerts" in message
    assert "evidence_summary" in message
    assert "citations" in message
    assert isinstance(message["citations"], list)

    assert "riskAlerts" not in message
    assert "evidenceSummary" not in message

    assert "action_plan" in result
    assert result["action_plan"] is not None
    assert "actions" in result["action_plan"]
    assert "confidence" in result["action_plan"]
    assert "policy_version" in result["action_plan"]
    assert "policy_changelog_refs" in result["action_plan"]
