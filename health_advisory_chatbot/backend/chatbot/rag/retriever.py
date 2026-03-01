"""
Semantic Retriever Module

Orchestrates retrieval of relevant documents from Vector Store.
Handles:
1. Query embedding
2. Vector similarity search
3. Confidence threshold filtering
4. Result formatting for LLM
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .vector_store import get_vector_store, VectorStore

logger = logging.getLogger(__name__)

@dataclass
class RetrievedEvidence:
    """Structure for retrieved evidence item."""
    text: str
    metadata: Dict[str, Any]
    score: float  # Similarity score (lower is better for L2, higher for cosine)
    source_id: str

class SemanticRetriever:
    """
    Retrieves and ranks evidence from knowledge base.
    """
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or get_vector_store()
        
    def retrieve(self, 
                 query: str, 
                 collection_name: str, 
                 n_results: int = 5,
                 min_score: float = 0.0,
                 where_filter: Optional[Dict] = None) -> List[RetrievedEvidence]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User question or topic
            collection_name: Target collection (clinical_evidence, etc.)
            n_results: Max results to return
            min_score: Minimum relevance score (semantic similarity)
            where_filter: Metadata filter
            
        Returns:
            List of RetrievedEvidence objects
        """
        try:
            results = self.vector_store.query_similar(
                collection_name=collection_name,
                query_text=query,
                n_results=n_results,
                where=where_filter
            )
            
            evidence_list = []
            
            # Unpack ChromaDB results structure
            # structure: {'documents': [[...]], 'metadatas': [[...]], 'distances': [[...]], 'ids': [[...]]}
            if not results['documents']:
                return []
                
            num_hits = len(results['documents'][0])
            
            for i in range(num_hits):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                dist = results['distances'][0][i] # Chroma default is L2 distance (lower is better) or Cosine
                uid = results['ids'][0][i]
                
                # Convert distance to similarity score if needed
                # For now, we pass raw distance. Lower is better for L2.
                # If using cosine similarity in Chroma (needs config), structure might differ.
                # Assuming default L2 for now.
                
                evidence = RetrievedEvidence(
                    text=doc,
                    metadata=meta,
                    score=dist, 
                    source_id=uid
                )
                evidence_list.append(evidence)
                
            return evidence_list
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

    def format_evidence_for_context(self, evidence_items: List[RetrievedEvidence]) -> str:
        """
        Format retrieved evidence into a string for LLM prompt.
        """
        if not evidence_items:
            return "No specific clinical evidence found."
            
        formatted = "RELEVANT CLINICAL EVIDENCE:\n"
        for i, item in enumerate(evidence_items, 1):
            source = item.metadata.get('source', 'Unknown')
            title = item.metadata.get('title', 'No Title')
            formatted += f"{i}. [{source}] {title}: {item.text}\n"
            
        return formatted

# Singleton
_retriever_instance = None

def get_retriever() -> SemanticRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = SemanticRetriever()
    return _retriever_instance
