from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.logger import get_logger
import yaml
import os

logger = get_logger(__name__)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class Reranker:
    """Re-ranker for search results."""
    
    def __init__(self):
        """Initialize reranker with configuration."""
        self.mode = config['reranker']['mode']
        self.weights = config['reranker']['weights']
    
    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank search candidates.
        
        Args:
            query (str): Query text
            candidates (List[Dict[str, Any]]): Search candidates
            
        Returns:
            List[Dict[str, Any]]: Reranked candidates
        """
        if not candidates:
            return candidates
            
        if self.mode == 'cross_encoder':
            # For simplicity, we'll simulate cross-encoder scoring with BM25
            # In a real implementation, you would use a cross-encoder model
            logger.warning("Cross-encoder mode simulated with BM25 scoring")
            reranked = self._bm25_rerank(query, candidates)
        elif self.mode == 'embedding_bm25_mixed':
            reranked = self._mixed_rerank(query, candidates)
        else:
            raise ValueError(f"Unknown reranker mode: {self.mode}")
            
        # Sort by final score
        reranked.sort(key=lambda x: x['final_score'], reverse=True)
        logger.info(f"Reranked {len(reranked)} candidates")
        return reranked
    
    def _bm25_rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank using BM25 scoring.
        
        Args:
            query (str): Query text
            candidates (List[Dict[str, Any]]): Search candidates
            
        Returns:
            List[Dict[str, Any]]: BM25 reranked candidates
        """
        # Extract candidate texts
        texts = [c['content'] for c in candidates]
        
        # Fit TF-IDF on candidates + query
        all_texts = texts + [query]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity between query and candidates
        query_vector = tfidf_matrix[-1]
        candidate_vectors = tfidf_matrix[:-1]
        
        # Calculate BM25-like scores
        scores = np.asarray(candidate_vectors.multiply(query_vector).sum(axis=1)).flatten()
        
        # Add scores to candidates
        reranked = []
        for i, candidate in enumerate(candidates):
            candidate_copy = candidate.copy()
            candidate_copy['bm25_score'] = float(scores[i])
            candidate_copy['final_score'] = float(scores[i])
            reranked.append(candidate_copy)
            
        return reranked
    
    def _mixed_rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank using mixed ANN and BM25 scores.
        
        Args:
            query (str): Query text
            candidates (List[Dict[str, Any]]): Search candidates
            
        Returns:
            List[Dict[str, Any]]: Mixed reranked candidates
        """
        # Normalize ANN scores (distances)
        ann_scores = np.array([c.get('distance', 0) for c in candidates])
        # Assuming distances are in [0, 1] range, higher is better (for IP metric)
        # If using L2 metric, you might need to convert differently
        normalized_ann_scores = ann_scores  # Already in appropriate range for IP
        
        # Calculate BM25 scores
        bm25_reranked = self._bm25_rerank(query, candidates)
        bm25_scores = np.array([c['bm25_score'] for c in bm25_reranked])
        
        # Normalize BM25 scores to [0, 1]
        if bm25_scores.max() != bm25_scores.min():
            normalized_bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        else:
            normalized_bm25_scores = np.zeros_like(bm25_scores)
        
        # Combine scores
        w_ann = self.weights.get('ann', 0.7)
        w_bm25 = self.weights.get('bm25', 0.3)
        
        final_scores = w_ann * normalized_ann_scores + w_bm25 * normalized_bm25_scores
        
        # Add final scores to candidates
        reranked = []
        for i, candidate in enumerate(bm25_reranked):
            candidate_copy = candidate.copy()
            candidate_copy['final_score'] = float(final_scores[i])
            reranked.append(candidate_copy)
            
        return reranked