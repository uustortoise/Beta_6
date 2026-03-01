"""
Unit tests for safety gateway and emergency bypass behavior.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.schemas import ChatRequest
from chatbot.core.advisory_engine import AdvisoryEngine
from chatbot.core.safety_gateway import SafetyGateway


class _FailIfCalledLLM:
    def generate_advisory(self, *args, **kwargs):  # pragma: no cover - should never execute
        raise AssertionError("LLM should not be called for urgent/emergency requests")


def test_detect_emergency_chest_pain():
    gateway = SafetyGateway()
    assessment = gateway.detect_urgency("I have chest pain and can't breathe.")

    assert assessment.level == "emergency"
    assert assessment.llm_allowed is False
    assert "chest pain" in assessment.triggers


def test_detect_routine_message():
    gateway = SafetyGateway()
    assessment = gateway.detect_urgency("How can I improve my sleep routine?")

    assert assessment.level == "routine"
    assert assessment.llm_allowed is True
    assert assessment.triggers == []


def test_build_escalation_message_chinese_template():
    gateway = SafetyGateway()
    assessment = gateway.detect_urgency("I have chest pain and shortness of breath")
    message = gateway.build_escalation_message(assessment, language="zh-CN")

    assert "911" in message
    assert "紧急" in message


def test_advisory_engine_emergency_bypasses_llm():
    engine = AdvisoryEngine()
    engine._llm_service = _FailIfCalledLLM()

    request = ChatRequest(
        elder_id="elder_safe_001",
        message="My mother has chest pain and passed out.",
        language="en",
    )
    response = engine.process_chat_request(request=request)

    assert "Call 911 immediately" in response.message.content
    assert any("EMERGENCY" in alert for alert in response.new_risk_alerts)
    assert response.model_version == "1.1.0-safety-gateway"


def test_advisory_engine_urgent_bypasses_llm():
    engine = AdvisoryEngine()
    engine._llm_service = _FailIfCalledLLM()

    request = ChatRequest(
        elder_id="elder_safe_002",
        message="He fell and has pain in his hip.",
        language="en",
    )
    response = engine.process_chat_request(request=request)

    assert "Urgent concern detected." in response.message.content
    assert any("URGENT" in alert for alert in response.new_risk_alerts)
    assert response.model_version == "1.1.0-safety-gateway"


def test_emergency_telemetry_logs_include_required_fields(caplog):
    caplog.set_level(logging.INFO)

    engine = AdvisoryEngine()
    engine._llm_service = _FailIfCalledLLM()

    request = ChatRequest(
        elder_id="elder_safe_telemetry",
        message="I have chest pain and cannot breathe.",
        language="en",
    )
    _ = engine.process_chat_request(request=request)

    assessment_logs = [r for r in caplog.records if r.message == "safety_gateway_assessment"]
    bypass_logs = [r for r in caplog.records if r.message == "safety_gateway_bypass"]
    emitted_logs = [r for r in caplog.records if r.message == "safety_gateway_response_emitted"]

    assert assessment_logs, "Expected safety_gateway_assessment log"
    assert bypass_logs, "Expected safety_gateway_bypass log"
    assert emitted_logs, "Expected safety_gateway_response_emitted log"

    bypass = bypass_logs[0]
    assert getattr(bypass, "urgency_level", None) == "emergency"
    assert getattr(bypass, "llm_bypassed", None) is True
    assert "chest pain" in getattr(bypass, "trigger_terms", [])
