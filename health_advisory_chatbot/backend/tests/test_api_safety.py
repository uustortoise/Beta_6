"""
API-level safety tests to ensure emergency/urgent bypass is honored.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes import ChatbotAPI


class _FailIfCalledLLM:
    def generate_advisory(self, *args, **kwargs):  # pragma: no cover - should never execute
        raise AssertionError("LLM should not be called for urgent/emergency requests")


def test_chat_api_emergency_bypasses_llm():
    api = ChatbotAPI()
    api.advisory_engine._llm_service = _FailIfCalledLLM()

    result = asyncio.run(
        api.chat(
            {
                "elder_id": "elder_api_safe_001",
                "message": "I have chest pain and shortness of breath.",
                "language": "en",
            }
        )
    )

    assert result["success"] is True
    assert "Call 911 immediately" in result["message"]["content"]
    assert any("EMERGENCY" in alert for alert in result["message"]["risk_alerts"])


def test_chat_api_urgent_bypasses_llm():
    api = ChatbotAPI()
    api.advisory_engine._llm_service = _FailIfCalledLLM()

    result = asyncio.run(
        api.chat(
            {
                "elder_id": "elder_api_safe_002",
                "message": "He fell and now has severe hip pain.",
                "language": "en",
            }
        )
    )

    assert result["success"] is True
    assert "Urgent concern detected." in result["message"]["content"]
    assert any("URGENT" in alert for alert in result["message"]["risk_alerts"])

