"""
Vector Store Module

Manages ChromaDB vector database for RAG system.
Handles 3 core collections:
1. clinical_evidence - Disease-specific clinical studies
2. adl_correlations - ADL-disease mappings
3. predictive_stats - Risk statistics and outcomes

Note: ChromaDB is optional. If not installed, the vector store will operate in fallback mode.
"""

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import os
import uuid

# Optional ChromaDB import
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not installed. Vector store will operate in fallback mode.")

logger = logging.getLogger(__name__)

# Constants
CHROMA_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data/chroma_db"
)

# Collection Names
COLLECTION_CLINICAL = "clinical_evidence"
COLLECTION_ADL = "adl_correlations"
COLLECTION_PREDICTIVE = "predictive_stats"

class VectorStore:
    """
    ChromaDB wrapper for managing medical knowledge vectors.
    Falls back to simple keyword matching if ChromaDB is not available.
    """
    
    def __init__(self, persistent_path: str = CHROMA_DB_PATH):
        """
        Initialize vector store.
        
        Args:
            persistent_path: Path to store ChromaDB data
        """
        self.persistent_path = persistent_path
        self.fallback_mode = not CHROMADB_AVAILABLE
        
        if self.fallback_mode:
            logger.warning("VectorStore operating in fallback mode (ChromaDB not available)")
            self.client = None
            self.collections = {}
            # Simple in-memory storage for fallback
            self._fallback_data = {
                COLLECTION_CLINICAL: [],
                COLLECTION_ADL: [],
                COLLECTION_PREDICTIVE: []
            }
        else:
            # Ensure directory exists
            Path(self.persistent_path).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Initializing ChromaDB at {self.persistent_path}")
            self.client = chromadb.PersistentClient(path=self.persistent_path)
            
            # Build collections
            self.collections = {
                COLLECTION_CLINICAL: self._get_or_create_collection(COLLECTION_CLINICAL),
                COLLECTION_ADL: self._get_or_create_collection(COLLECTION_ADL),
                COLLECTION_PREDICTIVE: self._get_or_create_collection(COLLECTION_PREDICTIVE)
            }
    
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection using default sentence-transformers."""
        if self.fallback_mode:
            return None
        # Note: Chroma uses 'all-MiniLM-L6-v2' by default if embedding_function is not specified
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        
    def add_documents(self, 
                     collection_name: str, 
                     documents: List[str], 
                     metadatas: List[Dict[str, Any]], 
                     ids: Optional[List[str]] = None):
        """
        Add documents to a specific collection.
        
        Args:
            collection_name: Name of target collection
            documents: List of text content
            metadatas: List of metadata dicts
            ids: Optional list of unique IDs (generated if None)
        """
        if collection_name not in self._fallback_data and collection_name not in self.collections:
            raise ValueError(f"Invalid collection name: {collection_name}")
        
        if self.fallback_mode:
            # Fallback: store in memory
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]
            for i, doc in enumerate(documents):
                self._fallback_data[collection_name].append({
                    "id": ids[i],
                    "document": doc,
                    "metadata": metadatas[i] if i < len(metadatas) else {}
                })
            logger.info(f"Added {len(documents)} documents to {collection_name} (fallback mode)")
        else:
            collection = self.collections[collection_name]
            
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]
                
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to {collection_name}")

    def query_similar(self, 
                     collection_name: str, 
                     query_text: str, 
                     n_results: int = 5,
                     where: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Semantic search in a collection.
        
        Args:
            collection_name: Target collection
            query_text: User query
            n_results: Number of results to return
            where: Metadata filtering (e.g. {"category": "diabetes"})
            
        Returns:
            Dict containing documents, metadatas, distances
        """
        if collection_name not in self._fallback_data and collection_name not in self.collections:
            raise ValueError(f"Invalid collection name: {collection_name}")
        
        if self.fallback_mode:
            # Fallback: simple keyword matching
            query_words = set(query_text.lower().split())
            results = []
            
            for item in self._fallback_data.get(collection_name, []):
                doc_words = set(item["document"].lower().split())
                score = len(query_words & doc_words)
                if score > 0:
                    results.append({
                        "document": item["document"],
                        "metadata": item["metadata"],
                        "score": score
                    })
            
            # Sort by score and take top n
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:n_results]
            
            return {
                "documents": [[r["document"] for r in results]],
                "metadatas": [[r["metadata"] for r in results]],
                "distances": [[1.0 / (r["score"] + 1) for r in results]]
            }
        else:
            collection = self.collections[collection_name]
            
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where
            )
            return results

    def get_stats(self) -> Dict[str, int]:
        """Get doc counts for all collections."""
        if self.fallback_mode:
            return {
                name: len(data) 
                for name, data in self._fallback_data.items()
            }
        
        stats = {}
        for name, col in self.collections.items():
            stats[name] = col.count()
        return stats

    def reset_collection(self, collection_name: str):
        """Clear all data in a collection."""
        if self.fallback_mode:
            if collection_name in self._fallback_data:
                self._fallback_data[collection_name] = []
                logger.warning(f"Reset collection: {collection_name}")
        else:
            if collection_name in self.collections:
                self.client.delete_collection(collection_name)
                self.collections[collection_name] = self._get_or_create_collection(collection_name)
                logger.warning(f"Reset collection: {collection_name}")

# Singleton
_vector_store_instance = None

def get_vector_store() -> VectorStore:
    """Get or create singleton vector store."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
