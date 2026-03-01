"""
Safety Gateway for urgent and emergency message detection.

This module provides deterministic triage before any LLM call.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import re


@dataclass(frozen=True)
class UrgencyAssessment:
    """Structured urgency decision produced by SafetyGateway."""

    level: str  # emergency | urgent | routine
    triggers: List[str]
    recommended_action: str
    llm_allowed: bool


class SafetyGateway:
    """Detect emergency/urgent scenarios and build deterministic responses."""

    def __init__(self):
        self._patterns = self._build_patterns()
        self._templates = self._build_templates()

    def detect_urgency(self, message: str) -> UrgencyAssessment:
        """
        Classify message urgency with deterministic pattern matching.

        Returns:
            UrgencyAssessment with level and action.
        """
        normalized = self._normalize(message)
        emergency_hits = self._match_terms(normalized, self._patterns["emergency"])
        if emergency_hits:
            return UrgencyAssessment(
                level="emergency",
                triggers=emergency_hits,
                recommended_action="Call 911 immediately and seek emergency medical care now.",
                llm_allowed=False,
            )

        urgent_hits = self._match_terms(normalized, self._patterns["urgent"])
        if urgent_hits:
            return UrgencyAssessment(
                level="urgent",
                triggers=urgent_hits,
                recommended_action="Contact your healthcare provider or urgent care now.",
                llm_allowed=False,
            )

        return UrgencyAssessment(
            level="routine",
            triggers=[],
            recommended_action="Proceed with standard advisory workflow.",
            llm_allowed=True,
        )

    def build_escalation_message(self, assessment: UrgencyAssessment, language: str = "en") -> str:
        """Build deterministic escalation response text based on urgency and locale."""
        locale = self._normalize_locale(language)
        templates = self._templates.get(locale, self._templates["en"])
        level_templates = templates.get(assessment.level, templates["routine"])

        if assessment.level == "routine":
            return level_templates["message"]

        trigger_text = ", ".join(assessment.triggers[:3]) if assessment.triggers else level_templates["default_trigger"]
        return (
            f"{level_templates['headline']}\n\n"
            f"{level_templates['action']}\n"
            f"{level_templates['followup']}\n\n"
            f"Detected concern(s): {trigger_text}\n"
            f"{level_templates['disclaimer']}"
        )

    @staticmethod
    def _normalize(text: str) -> str:
        text = (text or "").lower().strip()
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _normalize_locale(language: str) -> str:
        lang = (language or "en").lower()
        if lang.startswith("zh"):
            return "zh"
        return "en"

    @staticmethod
    def _match_terms(text: str, terms: List[Tuple[str, str]]) -> List[str]:
        hits: List[str] = []
        for label, pattern in terms:
            if re.search(pattern, text):
                hits.append(label)
        return hits

    @staticmethod
    def _build_patterns() -> Dict[str, List[Tuple[str, str]]]:
        # Ordered by clinical severity and common phrasing.
        return {
            "emergency": [
                ("chest pain", r"\bchest pain\b"),
                ("can't breathe", r"\b(can('?t| not)\s+breathe|short(ness)? of breath)\b"),
                ("stroke symptoms", r"\b(stroke|face droop|slurred speech|one-sided weakness)\b"),
                ("severe bleeding", r"\b(severe bleeding|bleeding heavily|won'?t stop bleeding)\b"),
                ("suicidal thoughts", r"\b(suicidal|want to die|kill myself|end my life)\b"),
                ("unconsciousness", r"\b(unconscious|passed out|not waking up)\b"),
            ],
            "urgent": [
                ("fall with injury", r"\b(fall|fell|fallen).*(hurt|injury|pain)\b"),
                ("confusion", r"\b(sudden confusion|disoriented|very confused)\b"),
                ("high fever", r"\b(high fever|fever above|fever over)\b"),
                ("blood in stool/urine", r"\b(blood in stool|bloody stool|blood in urine)\b"),
                ("persistent vomiting", r"\b(persistent vomiting|vomiting all day|can('?t| not) keep fluids down)\b"),
            ],
        }

    @staticmethod
    def _build_templates() -> Dict[str, Dict[str, Dict[str, str]]]:
        return {
            "en": {
                "emergency": {
                    "headline": "Emergency concern detected.",
                    "action": "Call 911 immediately. Do not drive yourself.",
                    "followup": "If available, alert a caregiver or nearby person right now.",
                    "default_trigger": "critical symptoms",
                    "disclaimer": "This response is prioritized for immediate safety, not routine advice.",
                },
                "urgent": {
                    "headline": "Urgent concern detected.",
                    "action": "Seek same-day medical care or contact your clinician now.",
                    "followup": "If symptoms worsen, call 911 immediately.",
                    "default_trigger": "urgent symptoms",
                    "disclaimer": "This response is prioritized for urgent triage and follow-up.",
                },
                "routine": {
                    "message": "No emergency or urgent concern detected.",
                },
            },
            "zh": {
                "emergency": {
                    "headline": "检测到紧急健康风险。",
                    "action": "请立即拨打 911，不要自行驾车就医。",
                    "followup": "请立刻通知照护者或附近人员协助。",
                    "default_trigger": "紧急症状",
                    "disclaimer": "该回复用于紧急安全处置，不是常规建议。",
                },
                "urgent": {
                    "headline": "检测到需要尽快处理的健康风险。",
                    "action": "请当天联系医生或前往急诊/急诊门诊。",
                    "followup": "若症状加重，请立即拨打 911。",
                    "default_trigger": "急迫症状",
                    "disclaimer": "该回复用于快速分诊与后续处理。",
                },
                "routine": {
                    "message": "未检测到紧急或急迫风险。",
                },
            },
        }


_safety_gateway = None


def get_safety_gateway() -> SafetyGateway:
    """Get or create singleton safety gateway."""
    global _safety_gateway
    if _safety_gateway is None:
        _safety_gateway = SafetyGateway()
    return _safety_gateway

