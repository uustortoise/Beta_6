"""
API Routes for Health Advisory Chatbot

FastAPI-compatible route handlers for:
- POST /api/chat - Send message and get advisory
- GET /api/chat/history/{session_id} - Get conversation history
- GET /api/chat/suggestions - Get suggested questions
- GET /api/health - Health check
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import os

# Import models
from models.schemas import (
    ChatRequest,
    ChatResponse,
    Message,
    MedicalProfile,
    ADLSummary,
    ICOPEAssessment,
    SleepSummary,
)

from chatbot.core import get_advisory_engine


class ChatbotAPI:
    """
    API handlers for chatbot endpoints.
    
    Can be integrated with FastAPI, Flask, or other frameworks.
    """
    
    def __init__(self):
        """Initialize API handlers."""
        self.advisory_engine = get_advisory_engine()
    
    async def chat(
        self,
        request_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle chat request.
        
        Args:
            request_data: Request body with elder_id, message, etc.
        
        Returns:
            Response dictionary
        """
        try:
            # Parse request
            request = ChatRequest(**request_data)
            
            # In production, fetch data from Beta 5.5 services
            # For now, use mock data if not provided
            medical_profile = request_data.get("_medical_profile")
            adl_summary = request_data.get("_adl_summary")
            icope = request_data.get("_icope_assessment")
            sleep = request_data.get("_sleep_summary")
            
            # Process request
            response = self.advisory_engine.process_chat_request(
                request=request,
                medical_profile=medical_profile,
                adl_summary=adl_summary,
                icope_assessment=icope,
                sleep_summary=sleep,
            )
            
            # Convert to dict
            return self._response_to_dict(response)
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": {
                    "role": "assistant",
                    "content": "I apologize, but I encountered an error processing your request. Please try again or contact support.",
                    "timestamp": datetime.now().isoformat(),
                }
            }
    
    async def get_history(
        self,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Get conversation history.
        
        Args:
            session_id: Session identifier
        
        Returns:
            History response
        """
        session = self.advisory_engine.get_session_history(session_id)
        
        if not session:
            return {
                "success": False,
                "error": "Session not found",
            }
        
        return {
            "success": True,
            "session_id": session_id,
            "elder_id": session.elder_id,
            "started_at": session.started_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "citations": [
                        {
                            "title": c.title,
                            "authors": c.authors,
                            "year": c.year,
                            "evidence_level": c.evidence_level.value,
                        }
                        for c in msg.citations
                    ] if msg.citations else [],
                }
                for msg in session.messages
            ],
        }
    
    async def get_suggestions(
        self,
        elder_id: str,
    ) -> Dict[str, Any]:
        """
        Get suggested questions for elder.
        
        Args:
            elder_id: Elder identifier
        
        Returns:
            Suggestions response
        """
        suggestions = self.advisory_engine.get_suggested_questions(elder_id)
        
        return {
            "success": True,
            "elder_id": elder_id,
            "suggestions": suggestions,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "health_advisory_chatbot",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "advisory_engine": "ok",
                "knowledge_base": "ok",
                "risk_stratifier": "ok",
            },
        }
    
    def _response_to_dict(self, response: ChatResponse) -> Dict[str, Any]:
        """Convert ChatResponse to dictionary."""
        return {
            "success": True,
            "session_id": response.session_id,
            "message": {
                "role": response.message.role.value,
                "content": response.message.content,
                "timestamp": response.message.timestamp.isoformat(),
                "evidence_summary": response.message.evidence_summary,
                "risk_alerts": response.message.risk_alerts,
                "citations": [
                    {
                        "source_type": c.source_type,
                        "title": c.title,
                        "authors": c.authors,
                        "journal": c.journal,
                        "year": c.year,
                        "doi": c.doi,
                        "url": c.url,
                        "pmid": c.pmid,
                        "evidence_level": c.evidence_level.value,
                        "confidence_score": c.confidence_score,
                    }
                    for c in response.message.citations
                ],
            },
            "health_context": {
                "elder_id": response.health_context.elder_id,
                "data_completeness": response.health_context.data_completeness,
                "context_summary": response.health_context.context_summary,
            } if response.health_context else None,
            "risks": {
                "overall_score": response.current_risks.overall_risk_score if response.current_risks else None,
                "overall_level": response.current_risks.overall_risk_level.value if response.current_risks else None,
                "fall_risk": response.current_risks.fall_risk if response.current_risks else None,
                "cognitive_risk": response.current_risks.cognitive_decline_risk if response.current_risks else None,
                "critical_alerts": response.current_risks.critical_alerts if response.current_risks else [],
            } if response.current_risks else None,
            "recommendations": [
                {
                    "id": r.recommendation_id,
                    "title": r.title,
                    "category": r.category,
                    "priority": r.priority,
                    "description": r.description,
                    "evidence_level": r.evidence_level.value,
                    "confidence": r.confidence_score,
                    "citations": [
                        {
                            "title": c.title,
                            "authors": c.authors[:2] if c.authors else [],
                            "year": c.year,
                        }
                        for c in r.citations
                    ],
                }
                for r in response.recommendations
            ],
            "action_plan": {
                "actions": [
                    {
                        "id": a.id,
                        "title": a.title,
                        "description": a.description,
                        "priority": a.priority,
                        "requires_clinician": a.requires_clinician,
                        "policy_refs": a.policy_refs,
                    }
                    for a in response.action_plan.actions
                ],
                "contraindications": response.action_plan.contraindications,
                "confidence": response.action_plan.confidence,
                "policy_version": response.action_plan.policy_version,
                "policy_changelog_refs": response.action_plan.policy_changelog_refs,
            } if response.action_plan else None,
            "new_risk_alerts": response.new_risk_alerts,
            "response_time_ms": response.response_time_ms,
            "model_version": response.model_version,
        }


# FastAPI integration helpers
def create_fastapi_routes(app):
    """
    Register routes with FastAPI app.
    
    Usage:
        from fastapi import FastAPI
        app = FastAPI()
        create_fastapi_routes(app)
    """
    from fastapi import HTTPException
    
    api = ChatbotAPI()
    
    @app.post("/api/chat")
    async def chat_endpoint(request: dict):
        """Chat endpoint."""
        result = await api.chat(request)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    
    @app.get("/api/chat/history/{session_id}")
    async def history_endpoint(session_id: str):
        """Get conversation history."""
        result = await api.get_history(session_id)
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))
        return result
    
    @app.get("/api/chat/suggestions")
    async def suggestions_endpoint(elder_id: str):
        """Get suggested questions."""
        return await api.get_suggestions(elder_id)
    
    @app.get("/api/health")
    async def health_endpoint():
        """Health check."""
        return await api.health_check()


# Flask integration helpers
def create_flask_routes(bp):
    """
    Register routes with Flask blueprint.
    
    Usage:
        from flask import Blueprint
        bp = Blueprint('chatbot', __name__)
        create_flask_routes(bp)
    """
    from flask import request, jsonify
    
    api = ChatbotAPI()
    
    @bp.route('/api/chat', methods=['POST'])
    def chat():
        """Chat endpoint."""
        data = request.get_json()
        result = api.chat(data)  # Note: Flask is sync, use async_to_sync if needed
        return jsonify(result)
    
    @bp.route('/api/chat/history/<session_id>', methods=['GET'])
    def history(session_id):
        """Get conversation history."""
        result = api.get_history(session_id)
        return jsonify(result)
    
    @bp.route('/api/chat/suggestions', methods=['GET'])
    def suggestions():
        """Get suggested questions."""
        elder_id = request.args.get('elder_id')
        result = api.get_suggestions(elder_id)
        return jsonify(result)
    
    @bp.route('/api/health', methods=['GET'])
    def health():
        """Health check."""
        result = api.health_check()
        return jsonify(result)
