import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VectorDocument:
    """Document with vector embedding"""
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]

class VectorStore:
    """Simple vector store for terminology and context retrieval"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.documents: Dict[str, VectorDocument] = {}
        self.index_built = False
        
        logger.info(f"VectorStore initialized with embedding_dim={embedding_dim}")
    
    def add_document(self, 
                    doc_id: str,
                    content: str,
                    embedding: np.ndarray,
                    metadata: Dict[str, Any] = None) -> bool:
        """Add document to vector store"""
        try:
            if embedding.shape[0] != self.embedding_dim:
                logger.error(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")
                return False
            
            document = VectorDocument(
                id=doc_id,
                content=content,
                embedding=embedding,
                metadata=metadata or {}
            )
            
            self.documents[doc_id] = document
            self.index_built = False  # Invalidate index
            
            logger.debug(f"Added document '{doc_id}' to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    def search(self, 
              query_embedding: np.ndarray,
              top_k: int = 5,
              threshold: float = 0.7) -> List[Tuple[VectorDocument, float]]:
        """Search for similar documents"""
        try:
            if len(self.documents) == 0:
                return []
            
            results = []
            
            for doc in self.documents.values():
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, doc.embedding)
                
                if similarity >= threshold:
                    results.append((doc, similarity))
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID"""
        return self.documents.get(doc_id)
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove document from store"""
        try:
            if doc_id in self.documents:
                del self.documents[doc_id]
                self.index_built = False
                logger.debug(f"Removed document '{doc_id}' from vector store")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing document: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            return {
                "total_documents": len(self.documents),
                "embedding_dim": self.embedding_dim,
                "index_built": self.index_built,
                "memory_usage_mb": self._estimate_memory_usage()
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        try:
            # Rough estimate: each embedding is embedding_dim * 4 bytes (float32)
            # Plus some overhead for strings and metadata
            bytes_per_doc = self.embedding_dim * 4 + 1000  # 1KB overhead per doc
            total_bytes = len(self.documents) * bytes_per_doc
            return total_bytes / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.error(f"Error estimating memory usage: {e}")
            return 0.0
