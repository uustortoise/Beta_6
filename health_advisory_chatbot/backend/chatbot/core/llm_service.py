"""
LLM Service for Health Advisory

RAG-augmented generation with evidence grounding.
Integrates with OpenAI, Claude, or local models.

Key features:
- Prompt engineering for medical advice
- Context window management
- Evidence retrieval and citation
- Response formatting
"""

import os
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
import json

from models.schemas import (
    HealthContext,
    Message,
    MessageRole,
    Citation,
    AdvisoryRecommendation,
)
from chatbot.rag.retriever import get_retriever, SemanticRetriever
from chatbot.rag.vector_store import COLLECTION_CLINICAL


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: str = "openai"  # openai/anthropic/deepseek/local
    model: str = "gpt-4"
    temperature: float = 0.3  # Lower for medical accuracy
    max_tokens: int = 2000
    api_key: Optional[str] = None
    api_base: Optional[str] = None


class LLMService:
    """
    Large Language Model service for health advisory generation.
    
    Provides:
    - Prompt construction with health context
    - Evidence-grounded response generation
    - Streaming and non-streaming modes
    - Response parsing and validation
    """
    
    SYSTEM_PROMPT = """You are a health advisory assistant for elderly care, grounded in evidence-based medicine.

CORE PRINCIPLES:
1. All recommendations must be supported by clinical guidelines or peer-reviewed research
2. Cite sources for all medical claims using [Source: ID] format
3. Include confidence levels (High/Medium/Low) for each recommendation
4. Flag contraindications and safety concerns prominently
5. Use clear, elder-friendly language
6. Always include a medical disclaimer

RESPONSE STRUCTURE:
1. Brief summary of key findings
2. Risk assessment with specific metrics
3. Evidence-based recommendations with citations
4. Actionable next steps
5. When to seek immediate medical attention

SAFETY RULES:
- Never recommend stopping prescribed medications
- Flag potential drug interactions clearly
- Highlight fall risks and safety concerns
- Recommend professional consultation for significant concerns
- Do not provide specific dosing adjustments

CITATION FORMAT:
Use [Source: ID] after each medical claim. IDs will be resolved to full citations."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM service.
        
        Args:
            config: LLM configuration (uses env vars if not provided)
        """

        self.config = config or self._load_config_from_env()
        self._client = None
        self.retriever = get_retriever()
    
    def _load_config_from_env(self) -> LLMConfig:
        """Load configuration from environment variables."""
        provider = os.getenv("LLM_PROVIDER", "openai")
        
        # Determine API key based on provider
        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
            model = os.getenv("LLM_MODEL", "deepseek-chat")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            api_base = os.getenv("ANTHROPIC_API_BASE")
            model = os.getenv("LLM_MODEL", "claude-3-sonnet-20240229")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            api_base = os.getenv("OPENAI_API_BASE")
            model = os.getenv("LLM_MODEL", "gpt-4")
        
        return LLMConfig(
            provider=provider,
            model=model,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
            api_key=api_key,
            api_base=api_base,
        )
    
    def _get_client(self):
        """Get or create LLM client."""
        if self._client is None:
            if self.config.provider == "openai":
                try:
                    import openai
                    self._client = openai.OpenAI(
                        api_key=self.config.api_key,
                        base_url=self.config.api_base,
                    )
                except ImportError:
                    raise ImportError("OpenAI package not installed. Install with: pip install openai")
            
            elif self.config.provider == "anthropic":
                try:
                    import anthropic
                    self._client = anthropic.Anthropic(api_key=self.config.api_key)
                except ImportError:
                    raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
            
            elif self.config.provider == "deepseek":
                # DeepSeek uses OpenAI-compatible API
                try:
                    import openai
                    self._client = openai.OpenAI(
                        api_key=self.config.api_key,
                        base_url=self.config.api_base or "https://api.deepseek.com",
                    )
                except ImportError:
                    raise ImportError("OpenAI package not installed. Install with: pip install openai")
            
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
        
        return self._client
    
    def generate_advisory(
        self,
        health_context: HealthContext,
        user_question: str,
        conversation_history: Optional[List[Message]] = None,
        retrieved_evidence: Optional[List[Citation]] = None,
    ) -> Dict[str, Any]:
        """
        Generate evidence-based health advisory.
        
        Args:
            health_context: Fused health context
            user_question: User specific question
            conversation_history: Previous messages
            retrieved_evidence: Pre-retrieved evidence citations
        
        Returns:
            Dictionary with response text, citations, and recommendations
        """

        
        # RAG Integration: Retrieve evidence if not provided
        if retrieved_evidence is None:
            try:
                # 1. Retrieve raw evidence
                raw_evidence = self.retriever.retrieve(
                    query=user_question,
                    collection_name=COLLECTION_CLINICAL,
                    n_results=3,
                    min_score=0.5
                )
                
                # 2. Convert to Citation format for prompt
                retrieved_evidence = []
                for item in raw_evidence:
                    retrieved_evidence.append(Citation(
                        source_id=item.source_id,
                        title=item.metadata.get('title', 'Unknown'),
                        authors=[item.metadata.get('source', 'Unknown Source')], # Simplification
                        journal=item.metadata.get('category', 'General'),
                        year=str(item.metadata.get('year', '202X')),
                        url=None,
                        evidence_level=EvidenceLevel.SYSTEMATIC_REVIEW if "review" in item.text.lower() else EvidenceLevel.OBSERVATIONAL,
                        confidence_score=0.8
                    ))
            except Exception as e:
                print(f"RAG Retrieval failed: {e}")
                retrieved_evidence = []

        # Re-build prompt with evidence
        prompt = self._build_prompt(
            health_context=health_context,
            user_question=user_question,
            conversation_history=conversation_history,
            retrieved_evidence=retrieved_evidence,
        )
        
        # Generate response
        try:
            if self.config.provider == "openai":
                response = self._generate_openai(prompt)
            elif self.config.provider == "anthropic":
                response = self._generate_anthropic(prompt)
            elif self.config.provider == "deepseek":
                response = self._generate_deepseek(prompt)
            else:
                response = self._generate_mock(prompt)  # Fallback for testing
            
            # Parse and validate response
            parsed = self._parse_response(response)
            
            return parsed
            
        except Exception as e:
            # Return safe fallback response
            return self._generate_fallback_response(health_context, user_question, str(e))
    
    def _build_prompt(
        self,
        health_context: HealthContext,
        user_question: str,
        conversation_history: Optional[List[Message]],
        retrieved_evidence: Optional[List[Citation]],
    ) -> str:
        """Build comprehensive prompt with context."""
        parts = []
        
        # System prompt
        parts.append(f"SYSTEM: {self.SYSTEM_PROMPT}")
        
        # Health context
        parts.append("\n--- PATIENT HEALTH CONTEXT ---")
        parts.append(health_context.context_summary or "Limited context available")
        
        # Risk assessment
        if health_context.risk_assessment:
            risk = health_context.risk_assessment
            parts.append(f"\nRISK ASSESSMENT:")
            parts.append(f"- Overall Risk: {risk.overall_risk_level.value} ({risk.overall_risk_score}/100)")
            parts.append(f"- Fall Risk: {risk.fall_risk or 'N/A'}/100")
            parts.append(f"- Cognitive Risk: {risk.cognitive_decline_risk or 'N/A'}/100")
            parts.append(f"- Sleep Risk: {risk.sleep_disorder_risk or 'N/A'}/100")
            
            if risk.top_risk_factors:
                parts.append("\nTOP RISK FACTORS:")
                for factor in risk.top_risk_factors[:3]:
                    parts.append(f"- {factor.factor_name} (weighted score: {factor.weighted_score:.1f})")
        
        # Trajectories
        if health_context.trajectories:
            parts.append("\nPREDICTED TRAJECTORIES:")
            for traj in health_context.trajectories:
                parts.append(f"- {traj.domain}: {traj.current_status} -> {traj.predicted_status} "
                           f"(confidence: {traj.confidence:.0f}%)")
        
        # Retrieved evidence
        if retrieved_evidence:
            parts.append("\n--- RETRIEVED EVIDENCE ---")
            for i, citation in enumerate(retrieved_evidence[:5], 1):
                parts.append(f"\n[{i}] {citation.title}")
                parts.append(f"    Authors: {', '.join(citation.authors[:3])}")
                parts.append(f"    Journal: {citation.journal}, {citation.year}")
                parts.append(f"    Evidence Level: {citation.evidence_level.value}")
        
        # Conversation history
        if conversation_history:
            parts.append("\n--- CONVERSATION HISTORY ---")
            for msg in conversation_history[-3:]:  # Last 3 messages
                role = "USER" if msg.role == MessageRole.USER else "ASSISTANT"
                parts.append(f"{role}: {msg.content[:200]}...")
        
        # User question
        parts.append(f"\n--- USER QUESTION ---")
        parts.append(user_question)
        
        parts.append("\n--- RESPONSE INSTRUCTIONS ---")
        parts.append("1. Answer the user's specific question")
        parts.append("2. Cite sources using [Source: ID] format")
        parts.append("3. Include confidence level for each recommendation")
        parts.append("4. Highlight any urgent concerns")
        parts.append("5. End with medical disclaimer")
        
        return "\n".join(parts)
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI API."""
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        return response.choices[0].message.content
    
    def _generate_anthropic(self, prompt: str) -> str:
        """Generate using Anthropic Claude API."""
        client = self._get_client()
        
        response = client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        
        return response.content[0].text
    
    def _generate_deepseek(self, prompt: str) -> str:
        """Generate using DeepSeek API (OpenAI-compatible)."""
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        return response.choices[0].message.content
    
    def _generate_mock(self, prompt: str) -> str:
        """
        Generate mock response for testing without API.
        
        Returns a structured response based on context.
        """
        # Extract key information from prompt for mock response
        has_fall_risk = "Fall Risk:" in prompt and any(x in prompt for x in ["high", "60", "70", "80", "90"])
        has_cognitive_risk = "Cognitive Risk:" in prompt and any(x in prompt for x in ["high", "60", "70", "80", "90"])
        
        response_parts = []
        
        response_parts.append("## Health Advisory Summary")
        response_parts.append("")
        
        if has_fall_risk:
            response_parts.append("🔴 **HIGH FALL RISK DETECTED**")
            response_parts.append("")
            response_parts.append("Based on your recent activity data and health profile, there are significant concerns about fall risk.")
            response_parts.append("")
            response_parts.append("**Key Risk Factors:**")
            response_parts.append("- History of nighttime bathroom visits")
            response_parts.append("- Reduced mobility indicators")
            response_parts.append("- Medications that may affect balance")
            response_parts.append("")
            response_parts.append("**Evidence-Based Recommendations** [Confidence: HIGH]")
            response_parts.append("")
            response_parts.append("1. **Home Safety Assessment** [Source: ags_falls_2023_home_modification]")
            response_parts.append("   - Install grab bars in bathroom")
            response_parts.append("   - Improve nighttime lighting")
            response_parts.append("   - Remove tripping hazards")
            response_parts.append("")
            response_parts.append("2. **Exercise Program** [Source: ags_falls_2023_exercise]")
            response_parts.append("   - Tai Chi classes (2-3x/week)")
            response_parts.append("   - Balance training daily")
            response_parts.append("   - Physical therapy consultation")
            response_parts.append("")
        
        if has_cognitive_risk:
            response_parts.append("🟡 **COGNITIVE HEALTH MONITORING**")
            response_parts.append("")
            response_parts.append("Current indicators suggest monitoring cognitive health.")
            response_parts.append("")
            response_parts.append("**Recommendations:**")
            response_parts.append("1. Cognitive stimulation activities [Source: who_icope_cognitive]")
            response_parts.append("2. Regular physical exercise (150 min/week)")
            response_parts.append("3. Social engagement programs")
            response_parts.append("")
        
        response_parts.append("---")
        response_parts.append("**Medical Disclaimer:** This advisory is for informational purposes only and does not replace professional medical advice. Please consult your healthcare provider for personalized care.")
        
        return "\n".join(response_parts)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured format.
        
        Extracts:
        - Main content
        - Citations
        - Recommendations
        - Risk alerts
        """
        # Extract citations from [Source: ID] format
        import re
        
        citations_found = re.findall(r'\[Source:\s*([^\]]+)\]', response)
        
        # Look for confidence indicators
        confidence_pattern = r'\[Confidence:\s*(HIGH|MEDIUM|LOW)\]'
        confidences = re.findall(confidence_pattern, response, re.IGNORECASE)
        
        # Check for risk alerts
        risk_alerts = []
        if "🔴" in response or "HIGH FALL RISK" in response:
            risk_alerts.append("High fall risk detected")
        if "🟡" in response and "COGNITIVE" in response:
            risk_alerts.append("Cognitive decline monitoring recommended")
        
        # Build recommendations (simplified extraction)
        recommendations = []
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '•', '-')) and len(line) > 20:
                # Look for recommendation pattern
                if any(word in line.lower() for word in ['recommend', 'assessment', 'exercise', 'consultation', 'monitoring']):
                    rec_text = line.strip().lstrip('12345.-• ')
                    recommendations.append({
                        "text": rec_text,
                        "category": "general",
                        "priority": 5,
                    })
        
        return {
            "content": response,
            "citations_found": citations_found,
            "confidence_levels": confidences,
            "risk_alerts": risk_alerts,
            "recommendations_extracted": recommendations,
            "is_valid": True,
        }
    
    def _generate_fallback_response(
        self,
        health_context: HealthContext,
        user_question: str,
        error: str
    ) -> Dict[str, Any]:
        """Generate safe fallback response on error."""
        return {
            "content": (
                "I apologize, but I'm experiencing technical difficulties generating your personalized "
                "health advisory. Please consult with your healthcare provider for immediate concerns.\n\n"
                "**General Wellness Recommendations:**\n"
                "- Maintain regular physical activity as tolerated\n"
                "- Ensure adequate sleep (7-8 hours)\n"
                "- Stay hydrated and maintain balanced nutrition\n"
                "- Attend all scheduled medical appointments\n\n"
                "**Medical Disclaimer:** This information is not a substitute for professional medical advice."
            ),
            "citations_found": [],
            "confidence_levels": [],
            "risk_alerts": [],
            "recommendations_extracted": [],
            "is_valid": False,
            "error": error,
        }
    
    async def generate_streaming(
        self,
        health_context: HealthContext,
        user_question: str,
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response.
        
        Yields text chunks as they become available.
        """
        prompt = self._build_prompt(
            health_context=health_context,
            user_question=user_question,
            conversation_history=None,
            retrieved_evidence=None,
        )
        
        try:
            if self.config.provider in ("openai", "deepseek"):
                # Both OpenAI and DeepSeek use the same streaming interface
                client = self._get_client()
                stream = client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stream=True,
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            
            else:
                # Non-streaming fallback for other providers
                response = self.generate_advisory(health_context, user_question)
                yield response["content"]
        
        except Exception as e:
            yield f"Error generating response: {str(e)}"


# Singleton
_llm_service: Optional[LLMService] = None


def get_llm_service(config: Optional[LLMConfig] = None) -> LLMService:
    """Get or create singleton LLM service."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(config)
    return _llm_service
