"""
Advisory Engine - Main Orchestrator

Coordinates all components to generate evidence-based health advisories:
1. Context Fusion - Aggregate health data
2. Risk Assessment - Calculate risks and trajectories
3. Evidence Retrieval - Fetch relevant research/guidelines
4. LLM Generation - Generate advisory with citations
5. Validation - Verify claims and citations
6. Response Assembly - Format final response
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
import time
import logging

from models.schemas import (
    ChatRequest,
    ChatResponse,
    Message,
    MessageRole,
    HealthContext,
    AdvisoryRecommendation,
    Citation,
    ConversationContext,
    ActionPlan,
)
from chatbot.core.safety_gateway import UrgencyAssessment

logger = logging.getLogger(__name__)


class AdvisoryEngine:
    """
    Main orchestrator for health advisory generation.
    
    Workflow:
    1. Build health context from all sources
    2. Calculate risks and trajectories
    3. Retrieve relevant evidence
    4. Generate LLM response
    5. Validate citations
    6. Assemble final advisory
    """
    
    def __init__(
        self,
        llm_config: Optional[Any] = None,
    ):
        """
        Initialize advisory engine.
        
        Args:
            llm_config: Optional LLM configuration
        """
        # Initialize component references (lazy loading)
        self._context_fusion = None
        self._risk_stratifier = None
        self._trajectory_predictor = None
        self._llm_service = None
        self._citation_validator = None
        self._policy_engine = None
        self._safety_gateway = None
        self._guidelines_db = None
        self._research_corpus = None
        self._llm_config = llm_config
        
        # Session storage (in production, use Redis/database)
        self._sessions: Dict[str, ConversationContext] = {}
    
    @property
    def context_fusion(self):
        """Lazy load context fusion engine."""
        if self._context_fusion is None:
            from .context_fusion import get_context_fusion_engine
            self._context_fusion = get_context_fusion_engine()
        return self._context_fusion
    
    @property
    def risk_stratifier(self):
        """Lazy load risk stratifier."""
        if self._risk_stratifier is None:
            from chatbot.predictive.risk_stratifier import get_risk_stratifier
            self._risk_stratifier = get_risk_stratifier()
        return self._risk_stratifier
    
    @property
    def llm_service(self):
        """Lazy load LLM service."""
        if self._llm_service is None:
            from .llm_service import get_llm_service
            self._llm_service = get_llm_service(self._llm_config)
        return self._llm_service
    
    @property
    def citation_validator(self):
        """Lazy load citation validator."""
        if self._citation_validator is None:
            from .citation_validator import get_citation_validator
            self._citation_validator = get_citation_validator()
        return self._citation_validator
    
    @property
    def guidelines_db(self):
        """Lazy load guidelines database."""
        if self._guidelines_db is None:
            from chatbot.knowledge_base import get_guidelines_db
            self._guidelines_db = get_guidelines_db()
        return self._guidelines_db

    @property
    def policy_engine(self):
        """Lazy load policy engine."""
        if self._policy_engine is None:
            from .policy_engine import get_policy_engine
            self._policy_engine = get_policy_engine()
        return self._policy_engine

    @property
    def safety_gateway(self):
        """Lazy load safety gateway."""
        if self._safety_gateway is None:
            from .safety_gateway import get_safety_gateway
            self._safety_gateway = get_safety_gateway()
        return self._safety_gateway
    
    @property
    def research_corpus(self):
        """Lazy load research corpus."""
        if self._research_corpus is None:
            from chatbot.knowledge_base import get_research_corpus
            self._research_corpus = get_research_corpus()
        return self._research_corpus
    
    def process_chat_request(
        self,
        request: ChatRequest,
        medical_profile: Optional[Any] = None,
        adl_summary: Optional[Any] = None,
        icope_assessment: Optional[Any] = None,
        sleep_summary: Optional[Any] = None,
    ) -> ChatResponse:
        """
        Process a chat request and generate advisory.
        
        Args:
            request: Chat request with user message
            medical_profile: Optional pre-fetched medical profile
            adl_summary: Optional pre-fetched ADL summary
            icope_assessment: Optional pre-fetched ICOPE assessment
            sleep_summary: Optional pre-fetched sleep summary
        
        Returns:
            ChatResponse with advisory and metadata
        """
        start_time = time.time()
        
        # Step 1: Get or create session
        session = self._get_or_create_session(request)

        # Step 2: Safety gateway triage (must occur before LLM/RAG path)
        urgency = self.safety_gateway.detect_urgency(request.message)
        logger.info(
            "safety_gateway_assessment",
            extra={
                "event": "safety_gateway_assessment",
                "urgency_level": urgency.level,
                "trigger_terms": urgency.triggers,
                "elder_id": request.elder_id,
                "session_id": session.session_id,
                "llm_bypassed": not urgency.llm_allowed,
            },
        )
        if not urgency.llm_allowed:
            logger.warning(
                "safety_gateway_bypass",
                extra={
                    "event": "safety_gateway_bypass",
                    "urgency_level": urgency.level,
                    "trigger_terms": urgency.triggers,
                    "elder_id": request.elder_id,
                    "session_id": session.session_id,
                    "llm_bypassed": True,
                },
            )
            emergency_response = self._build_escalation_response(
                request=request,
                session=session,
                urgency=urgency,
                start_time=start_time,
            )
            return emergency_response
        
        # Step 3: Build health context
        health_context = self.context_fusion.build_health_context(
            elder_id=request.elder_id,
            medical_profile=medical_profile,
            adl_summary=adl_summary,
            icope_assessment=icope_assessment,
            sleep_summary=sleep_summary,
            include_risk_assessment=True,
            include_trajectories=True,
        )

        # Step 3.5: Build deterministic action plan
        action_plan = self.policy_engine.build_action_plan(
            health_context=health_context,
            user_question=request.message,
        )
        
        # Step 4: Retrieve relevant evidence
        retrieved_evidence = self._retrieve_evidence(health_context, request.message)
        
        # Step 5: Generate LLM response
        llm_result = self.llm_service.generate_advisory(
            health_context=health_context,
            user_question=request.message,
            conversation_history=session.messages,
            retrieved_evidence=retrieved_evidence,
        )
        
        # Step 6: Validate citations
        validation = self.citation_validator.validate_response(
            response_text=llm_result["content"],
            citations_found=llm_result["citations_found"],
        )
        
        # Step 7: Enrich citations
        enriched_citations = self.citation_validator.enrich_citations(
            validation.corrected_citations
        )
        
        # Step 8: Build recommendations from LLM output and guidelines
        recommendations = self._build_recommendations(
            llm_result=llm_result,
            health_context=health_context,
            citations=enriched_citations,
        )
        
        # Step 9: Create assistant message
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=llm_result["content"],
            timestamp=datetime.now(),
            recommendations=recommendations,
            risk_alerts=llm_result["risk_alerts"],
            citations=enriched_citations,
            evidence_summary=self._generate_evidence_summary(validation, enriched_citations),
        )
        
        # Step 10: Update session
        user_message = Message(
            role=MessageRole.USER,
            content=request.message,
            timestamp=datetime.now(),
        )
        session.messages.append(user_message)
        session.messages.append(assistant_message)
        session.last_activity = datetime.now()
        
        # Step 11: Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Step 12: Build response
        response = ChatResponse(
            session_id=session.session_id,
            message=assistant_message,
            health_context=health_context,
            current_risks=health_context.risk_assessment,
            new_risk_alerts=llm_result["risk_alerts"],
            recommendations=recommendations,
            action_plan=action_plan,
            response_time_ms=processing_time_ms,
            model_version="1.0.0",
        )
        
        return response

    def _build_escalation_response(
        self,
        request: ChatRequest,
        session: ConversationContext,
        urgency: UrgencyAssessment,
        start_time: float,
    ) -> ChatResponse:
        """Build deterministic urgent/emergency response and bypass LLM pipeline."""
        escalation_message = self.safety_gateway.build_escalation_message(
            assessment=urgency,
            language=request.language,
        )

        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=escalation_message,
            timestamp=datetime.now(),
            recommendations=[],
            risk_alerts=[
                f"{urgency.level.upper()} concern detected",
                urgency.recommended_action,
            ],
            citations=[],
            evidence_summary="Deterministic safety triage response (LLM bypass).",
        )

        user_message = Message(
            role=MessageRole.USER,
            content=request.message,
            timestamp=datetime.now(),
        )
        session.messages.append(user_message)
        session.messages.append(assistant_message)
        session.last_activity = datetime.now()

        processing_time_ms = int((time.time() - start_time) * 1000)
        action_plan = self.policy_engine.build_escalation_action_plan(urgency.level)
        logger.warning(
            "safety_gateway_response_emitted",
            extra={
                "event": "safety_gateway_response_emitted",
                "urgency_level": urgency.level,
                "trigger_terms": urgency.triggers,
                "elder_id": request.elder_id,
                "session_id": session.session_id,
                "llm_bypassed": True,
                "response_time_ms": processing_time_ms,
            },
        )
        return ChatResponse(
            session_id=session.session_id,
            message=assistant_message,
            health_context=None,
            current_risks=None,
            new_risk_alerts=assistant_message.risk_alerts,
            recommendations=[],
            action_plan=action_plan,
            response_time_ms=processing_time_ms,
            model_version="1.1.0-safety-gateway",
        )
    
    def _get_or_create_session(
        self,
        request: ChatRequest,
    ) -> ConversationContext:
        """Get existing session or create new one."""
        if request.session_id and request.session_id in self._sessions:
            return self._sessions[request.session_id]
        
        # Create new session
        import uuid
        session_id = str(uuid.uuid4())
        
        session = ConversationContext(
            session_id=session_id,
            elder_id=request.elder_id,
            language=request.language,
            detail_level=request.detail_level,
        )
        
        self._sessions[session_id] = session
        return session
    
    def _retrieve_evidence(
        self,
        health_context: HealthContext,
        user_query: str,
    ) -> List[Citation]:
        """
        Retrieve relevant evidence for the query.
        
        Searches:
        - Clinical guidelines
        - Research corpus
        - Risk-specific recommendations
        """
        citations = []
        
        # Determine query topics
        topics = self._extract_topics(user_query, health_context)
        
        # Search guidelines
        for topic in topics:
            guidelines = self.guidelines_db.search_by_category(topic)
            for guideline in guidelines[:2]:  # Top 2 per topic
                citations.append(Citation(
                    source_type="clinical_guideline",
                    title=guideline.title,
                    authors=[guideline.source.value],
                    year=guideline.publication_year,
                    evidence_level=CitationValidator._map_guideline_to_evidence(guideline.evidence_grade),
                    confidence_score=90.0,
                ))
        
        # Search research corpus
        for topic in topics:
            papers = self.research_corpus.get_papers_by_topic(topic)[:2]
            for paper in papers:
                citations.append(paper.to_citation())
        
        # Add risk-based citations
        if health_context.risk_assessment:
            if health_context.risk_assessment.fall_risk and health_context.risk_assessment.fall_risk > 50:
                fall_guidelines = self.guidelines_db.get_fall_prevention_plan(["high_fall_risk"])
                for guideline in fall_guidelines[:2]:
                    citations.append(Citation(
                        source_type="clinical_guideline",
                        title=guideline.title,
                        authors=[guideline.source.value],
                        year=guideline.publication_year,
                        evidence_level=CitationValidator._map_guideline_to_evidence(guideline.evidence_grade),
                        confidence_score=85.0,
                    ))
        
        # Deduplicate by title
        seen_titles = set()
        unique_citations = []
        for citation in citations:
            if citation.title not in seen_titles:
                seen_titles.add(citation.title)
                unique_citations.append(citation)
        
        return unique_citations[:10]  # Limit to top 10
    
    def _extract_topics(
        self,
        query: str,
        health_context: HealthContext,
    ) -> List[str]:
        """Extract medical topics from query and context.
        
        Uses intelligent matching including:
        - Word variations (fall/fell/falls)
        - Fuzzy matching for typos
        - Related terms and synonyms
        """
        topics = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Query-based topics with expanded variations
        topic_patterns = {
            "fall_prevention": {
                "keywords": ["fall", "balance", "gait", "trip", "slip", "stumble", "tumble"],
                "variations": ["fell", "falling", "falls", "fallen"],
                "related": ["dizzy", "dizziness", "unsteady", "wobble", "near fall", "almost fell"],
            },
            "cognitive_health": {
                "keywords": ["memory", "cognitive", "dementia", "alzheimer", "confusion", "forgetful"],
                "variations": ["forget", "forgot", "forgetting", "confused", "disoriented"],
                "related": ["brain fog", "can't remember", "hard to recall", "mental clarity", "sharpness"],
            },
            "sleep_disorders": {
                "keywords": ["sleep", "insomnia", "apnea", "snore", "tired", "fatigue"],
                "variations": ["slept", "sleeping", "sleepless", "restless"],
                "related": ["night", "bedtime", "can't fall asleep", "wake up", "early morning", "tossing", "turning", "restless leg"],
            },
            "medication_safety": {
                "keywords": ["medication", "drug", "pill", "side effect", "prescription"],
                "variations": ["medicine", "meds", "tablet", "capsule", "dose"],
                "related": ["taking", "prescribed", "doctor gave me", "pharmacy", "drug interaction"],
            },
            "nutrition": {
                "keywords": ["nutrition", "weight", "appetite", "eat", "food", "diet"],
                "variations": ["eating", "ate", "hungry", "full", "meal"],
                "related": ["lost weight", "gained weight", "not hungry", "craving", "protein", "vitamin"],
            },
            "mobility": {
                "keywords": ["walk", "mobility", "exercise", "strength", "move"],
                "variations": ["walking", "walked", "moving", "movement"],
                "related": ["hard to walk", "can't walk", "stiff", "joint", "muscle", "weak", "physical therapy"],
            },
            "cardiovascular": {
                "keywords": ["heart", "blood pressure", "chest pain", "palpitation", "hypertension"],
                "variations": ["bp", "heartbeat", "racing heart"],
                "related": ["short of breath", "dizzy when standing", "swelling legs", "fluid retention"],
            },
            "pain_management": {
                "keywords": ["pain", "ache", "hurt", "sore", "discomfort"],
                "variations": ["painful", "aching", "hurting", "throbbing"],
                "related": ["back pain", "joint pain", "arthritis", "chronic pain", "sharp pain", "dull pain"],
            },
            "mental_health": {
                "keywords": ["depression", "anxiety", "stress", "mood", "sad", "worry"],
                "variations": ["depressed", "anxious", "worried", "stressed"],
                "related": ["feeling down", "low mood", "can't sleep", "nervous", "panic", "lonely", "isolated"],
            },
        }
        
        # Score each topic based on matches
        topic_scores = {}
        for topic, patterns in topic_patterns.items():
            score = 0
            matched_terms = []
            
            # Check all pattern types
            all_patterns = (
                patterns.get("keywords", []) + 
                patterns.get("variations", []) + 
                patterns.get("related", [])
            )
            
            for pattern in all_patterns:
                pattern_lower = pattern.lower()
                # Direct substring match (for multi-word phrases)
                if pattern_lower in query_lower:
                    # Weight: keywords=3, variations=2, related=1
                    if pattern in patterns.get("keywords", []):
                        score += 3
                    elif pattern in patterns.get("variations", []):
                        score += 2
                    else:
                        score += 1
                    matched_terms.append(pattern)
                # Word-level match for single words
                elif " " not in pattern_lower:
                    # Check if pattern is a substring of any query word (for typos/stemming)
                    for word in query_words:
                        # Pattern is substring of word or vice versa
                        if pattern_lower in word or word in pattern_lower:
                            if len(pattern_lower) >= 4:  # Only for meaningful words
                                score += 1
                                matched_terms.append(f"{pattern}~{word}")
                                break
            
            if score > 0:
                topic_scores[topic] = {"score": score, "matches": matched_terms}
                topics.append(topic)
        
        # Context-based topics (always add if risk is high, regardless of query)
        if health_context.risk_assessment:
            if health_context.risk_assessment.fall_risk and health_context.risk_assessment.fall_risk > 40:
                if "fall_prevention" not in topics:
                    topics.append("fall_prevention")
                    topic_scores["fall_prevention"] = {"score": 5, "matches": ["high_fall_risk_context"]}
            if health_context.risk_assessment.cognitive_decline_risk and health_context.risk_assessment.cognitive_decline_risk > 40:
                if "cognitive_health" not in topics:
                    topics.append("cognitive_health")
                    topic_scores["cognitive_health"] = {"score": 5, "matches": ["high_cognitive_risk_context"]}
            if health_context.risk_assessment.sleep_disorder_risk and health_context.risk_assessment.sleep_disorder_risk > 40:
                if "sleep_disorders" not in topics:
                    topics.append("sleep_disorders")
                    topic_scores["sleep_disorders"] = {"score": 5, "matches": ["high_sleep_risk_context"]}
        
        # Debug logging
        if topic_scores:
            print(f"[TopicExtraction] Query: '{query[:50]}...' → Topics: {topic_scores}")
        else:
            print(f"[TopicExtraction] Query: '{query[:50]}...' → No topics matched, using general_health")
        
        return list(set(topics)) if topics else ["general_health"]
    
    def _build_recommendations(
        self,
        llm_result: Dict,
        health_context: HealthContext,
        citations: List[Citation],
    ) -> List[AdvisoryRecommendation]:
        """Build structured recommendations from LLM output."""
        recommendations = []
        
        # Extract from LLM result
        for i, rec_data in enumerate(llm_result.get("recommendations_extracted", [])):
            recommendations.append(AdvisoryRecommendation(
                recommendation_id=f"llm_{i}",
                category="general",
                priority=rec_data.get("priority", 5),
                title=rec_data["text"][:50],
                description=rec_data["text"],
                evidence_level=CitationValidator._get_overall_evidence_level(citations),
                confidence_score=70.0,
                citations=citations[:2],
            ))
        
        # Add guideline-based recommendations for high risks
        if health_context.risk_assessment:
            risk = health_context.risk_assessment
            
            if risk.fall_risk and risk.fall_risk > 60:
                fall_recs = self.guidelines_db.get_fall_prevention_plan(["high_fall_risk"])
                for guideline in fall_recs[:2]:
                    recommendations.append(AdvisoryRecommendation(
                        recommendation_id=f"guideline_{guideline.guideline_id}",
                        category="fall_prevention",
                        priority=9 if risk.fall_risk > 75 else 7,
                        title=guideline.title,
                        description=guideline.description,
                        action_steps=guideline.action_steps,
                        evidence_level=CitationValidator._map_guideline_to_evidence(guideline.evidence_grade),
                        confidence_score=90.0,
                        citations=[Citation(
                            source_type="clinical_guideline",
                            title=guideline.title,
                            authors=[guideline.source.value],
                            year=guideline.publication_year,
                            evidence_level=CitationValidator._map_guideline_to_evidence(guideline.evidence_grade),
                            confidence_score=95.0,
                        )],
                    ))
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _generate_evidence_summary(
        self,
        validation: Any,
        citations: List[Citation],
    ) -> str:
        """Generate human-readable evidence summary."""
        if not citations:
            return "No citations available"
        
        quality = self.citation_validator.calculate_evidence_quality_score(citations)
        
        return (
            f"Based on {len(citations)} sources. "
            f"Evidence quality: {quality['level']} ({quality['score']:.0f}/100). "
            f"Validation: {'Passed' if validation.is_valid else 'Review Needed'}."
        )
    
    def get_session_history(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation history for a session."""
        return self._sessions.get(session_id)
    
    def get_suggested_questions(
        self,
        elder_id: str,
        health_context: Optional[HealthContext] = None,
    ) -> List[Dict[str, str]]:
        """
        Generate suggested questions based on health context.
        
        Returns:
            List of suggested questions with categories
        """
        suggestions = []
        
        # Default questions
        suggestions.extend([
            {"question": "How did I sleep last night?", "category": "sleep"},
            {"question": "What are my fall risk factors?", "category": "safety"},
            {"question": "Am I taking my medications correctly?", "category": "medications"},
        ])
        
        # Context-aware suggestions
        if health_context and health_context.risk_assessment:
            risk = health_context.risk_assessment
            
            if risk.fall_risk and risk.fall_risk > 50:
                suggestions.append({
                    "question": "What exercises can help prevent falls?",
                    "category": "fall_prevention"
                })
            
            if risk.cognitive_decline_risk and risk.cognitive_decline_risk > 50:
                suggestions.append({
                    "question": "How can I maintain my cognitive health?",
                    "category": "cognitive"
                })
            
            if risk.sleep_disorder_risk and risk.sleep_disorder_risk > 50:
                suggestions.append({
                    "question": "Why do I wake up so often at night?",
                    "category": "sleep"
                })
        
        return suggestions[:5]
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a conversation session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


# Helper methods for CitationValidator
class CitationValidator:
    """Extended with helper methods for the advisory engine."""
    
    @staticmethod
    def _map_guideline_to_evidence(grade: str) -> Any:
        """Map guideline grade to evidence level."""
        from models.schemas import EvidenceLevel
        mapping = {
            "A": EvidenceLevel.RCT,
            "B": EvidenceLevel.COHORT_STUDY,
            "C": EvidenceLevel.EXPERT_OPINION,
        }
        return mapping.get(grade, EvidenceLevel.EXPERT_OPINION)
    
    @staticmethod
    def _get_overall_evidence_level(citations: List[Citation]) -> Any:
        """Get overall evidence level from citations."""
        from models.schemas import EvidenceLevel
        if not citations:
            return EvidenceLevel.EXPERT_OPINION
        
        # Return highest level found
        priority = [
            EvidenceLevel.SYSTEMATIC_REVIEW,
            EvidenceLevel.CLINICAL_GUIDELINE,
            EvidenceLevel.RCT,
            EvidenceLevel.COHORT_STUDY,
            EvidenceLevel.CASE_CONTROL,
            EvidenceLevel.EXPERT_OPINION,
        ]
        
        for level in priority:
            if any(c.evidence_level == level for c in citations):
                return level
        
        return EvidenceLevel.EXPERT_OPINION


# Singleton
_advisory_engine: Optional[AdvisoryEngine] = None


def get_advisory_engine(llm_config: Optional[Any] = None) -> AdvisoryEngine:
    """Get or create singleton advisory engine."""
    global _advisory_engine
    if _advisory_engine is None:
        _advisory_engine = AdvisoryEngine(llm_config)
    return _advisory_engine
