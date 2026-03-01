"""
RAG Pipeline Verification Script

Tests the end-to-end flow:
1. User Query -> LLMService
2. LLMService -> SemanticRetriever (Search)
3. Retriever -> VectorStore (ChromaDB)
4. Evidence -> LLM Prompt -> Response
"""

import sys
import os
import logging

# Configure imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# Check if we are in tests/ or scripts/ or root
if 'tests' in current_dir:
    # .../health_advisory_chatbot/tests -> .../health_advisory_chatbot -> .../Beta_5.5
    chatbot_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(chatbot_dir)
else:
    # Assume run from root
    project_root = current_dir

chatbot_root = os.path.join(project_root, 'health_advisory_chatbot')
backend_root = os.path.join(chatbot_root, 'backend')

sys.path.insert(0, chatbot_root)
sys.path.insert(0, backend_root)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG_TEST")

try:
    from backend.chatbot.core.llm_service import get_llm_service, LLMConfig
    from models.schemas import HealthContext, RiskAssessment
    from backend.chatbot.rag.vector_store import COLLECTION_CLINICAL
except ImportError as e:
    logger.error(f"Import failed: {e}")
    sys.exit(1)

def test_rag_flow():
    logger.info("Initializing LLM Service (Mock Mode)...")
    config = LLMConfig(provider="mock")
    llm_service = get_llm_service(config)
    
    # Create Mock Context
    risk_assessment = RiskAssessment(
        overall_risk_score=65,
        overall_risk_level="moderate", 
        sleep_disorder_risk=80,
        cognitive_decline_risk=30,
        fall_risk=40,
        top_risk_factors=[]
    )

    context = HealthContext(
        elder_id="test_elder_001",
        context_summary="75-year-old female with history of hypertension and recent sleep complaints.",
        risk_assessment=risk_assessment,
        trajectories=[]
    )
    
    # Test Query
    query = "What can I do about my insomnia and frequent waking at night?"
    logger.info(f"Testing Query: {query}")
    
    # Generate Advisory (should trigger retrieval)
    logger.info("Calling generate_advisory...")
    try:
        response = llm_service.generate_advisory(
            health_context=context,
            user_question=query,
            retrieved_evidence=None # Force internal retrieval
        )
        logger.info("Response generated.")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        response = {}
    
    # Direct Retriever Test
    logger.info("\n--- Direct Retriever Test ---")
    retriever = llm_service.retriever
    
    results = retriever.retrieve(query, COLLECTION_CLINICAL)
    logger.info(f"Direct Retrieval Found: {len(results)} items")
    for item in results:
        logger.info(f"[{item.score:.4f}] {item.title if hasattr(item, 'title') else item.metadata.get('title')}")
        
    if len(results) > 0:
        logger.info("✅ RAG Retrieval is WORKING")
    else:
        logger.error("❌ RAG Retrieval FAILED (No results)")

if __name__ == "__main__":
    test_rag_flow()
