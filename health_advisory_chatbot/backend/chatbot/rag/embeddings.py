"""
Medical Embeddings Module

Generates vector embeddings for medical text using Sentence Transformers.
Optimized for clinical semantic search.
"""

from typing import List, Union
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class MedicalEmbeddings:
    """
    Wrapper for sentence-transformers to generate medical text embeddings.
    
    Uses 'all-MiniLM-L6-v2' by default:
    - 384 dimensions
    - Fast inference
    - Good general semantic performance
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
        """
        self.model_name = model_name
        self._model = None
        
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model to reduce startup time."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
        
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text or list of texts.
        
        Args:
            text: Single string or list of strings
            
        Returns:
            Numpy array of embeddings
        """
        try:
            embeddings = self.model.encode(text, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero vector as fallback (should handle upstream)
            if isinstance(text, str):
                return np.zeros(384)
            return np.zeros((len(text), 384))

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query (same as embed_text for symmetric models).
        """
        return self.embed_text(query)

# Singleton instance
_embeddings_instance = None

def get_embeddings_model() -> MedicalEmbeddings:
    """Get or create singleton embedding model instance."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = MedicalEmbeddings()
    return _embeddings_instance
