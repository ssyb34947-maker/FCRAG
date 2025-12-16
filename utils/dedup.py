import hashlib
from typing import List, Dict, Any
import numpy as np

try:
    from simhash import Simhash
    SIMHASH_AVAILABLE = True
except ImportError:
    SIMHASH_AVAILABLE = False

class Deduplicator:
    """Document deduplication utility supporting multiple strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize deduplicator with configuration.
        
        Args:
            config (Dict[str, Any]): Deduplication configuration
        """
        self.config = config
        self.strategy = config.get('strategy', 'md5')
        self.md5_threshold = config.get('md5_threshold', 1.0)
        self.simhash_threshold = config.get('simhash_threshold', 3)
        self.embedding_threshold = config.get('embedding_threshold', 0.95)
        
        # Storage for seen items
        self.seen_md5s = set()
        self.seen_simhashes = []
        self.seen_embeddings = []
    
    def is_duplicate(self, doc: Dict[str, Any], embedding: List[float] = None) -> bool:
        """
        Check if document is a duplicate based on configured strategy.
        
        Args:
            doc (Dict[str, Any]): Document to check
            embedding (List[float], optional): Document embedding for embedding-based dedup
            
        Returns:
            bool: True if document is duplicate, False otherwise
        """
        if self.strategy == 'md5':
            return self._is_md5_duplicate(doc)
        elif self.strategy == 'simhash':
            return self._is_simhash_duplicate(doc)
        elif self.strategy == 'embedding':
            if embedding is None:
                raise ValueError("Embedding required for embedding-based deduplication")
            return self._is_embedding_duplicate(embedding)
        else:
            raise ValueError(f"Unknown deduplication strategy: {self.strategy}")
    
    def _is_md5_duplicate(self, doc: Dict[str, Any]) -> bool:
        """
        Check for MD5 hash duplicate.
        
        Args:
            doc (Dict[str, Any]): Document to check
            
        Returns:
            bool: True if document is duplicate, False otherwise
        """
        text = doc.get('text', '')
        md5_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        if md5_hash in self.seen_md5s:
            return True
            
        self.seen_md5s.add(md5_hash)
        return False
    
    def _is_simhash_duplicate(self, doc: Dict[str, Any]) -> bool:
        """
        Check for SimHash approximate duplicate.
        
        Args:
            doc (Dict[str, Any]): Document to check
            
        Returns:
            bool: True if document is duplicate, False otherwise
        """
        if not SIMHASH_AVAILABLE:
            raise ImportError("Simhash library not available. Install with: pip install simhash")
            
        text = doc.get('text', '')
        doc_simhash = Simhash(text)
        
        for seen_simhash in self.seen_simhashes:
            distance = seen_simhash.distance(doc_simhash)
            if distance <= self.simhash_threshold:
                return True
                
        self.seen_simhashes.append(doc_simhash)
        return False
    
    def _is_embedding_duplicate(self, embedding: List[float]) -> bool:
        """
        Check for embedding-based duplicate using cosine similarity.
        
        Args:
            embedding (List[float]): Document embedding
            
        Returns:
            bool: True if document is duplicate, False otherwise
        """
        if not self.seen_embeddings:
            self.seen_embeddings.append(embedding)
            return False
            
        # Convert to numpy arrays
        emb_array = np.array(embedding)
        seen_arrays = np.array(self.seen_embeddings)
        
        # Normalize vectors
        emb_norm = emb_array / np.linalg.norm(emb_array)
        seen_norms = seen_arrays / np.linalg.norm(seen_arrays, axis=1, keepdims=True)
        
        # Calculate cosine similarities
        similarities = np.dot(seen_norms, emb_norm)
        
        # Check if any similarity exceeds threshold
        if np.any(similarities >= self.embedding_threshold):
            return True
            
        self.seen_embeddings.append(embedding)
        return False