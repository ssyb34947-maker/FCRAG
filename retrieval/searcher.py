from typing import List, Dict, Any, Optional
from embeddings.bairen_embedder import BairenEmbedder
from storage.milvus_client import MilvusClient
from utils.logger import get_logger
import yaml
import os

logger = get_logger(__name__)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class Searcher:
    """Document searcher using vector similarity."""
    
    def __init__(self):
        """Initialize searcher with embedder and Milvus client."""
        self.embedder = BairenEmbedder()
        self.milvus_client = MilvusClient()
        self.top_k = config['retrieval']['top_k']
        self.domain_filter = config['retrieval']['domain_filter']
    
    def search(self, query: str, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query (str): Query text
            domain (Optional[str]): Domain filter
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        # Generate query embedding
        query_embedding = self.embedder.embed([query])[0]
        logger.info("Generated query embedding")
        
        # Search in Milvus
        results = self.milvus_client.search(
            query_embedding=query_embedding,
            domain=domain if self.domain_filter else None,
            limit=self.top_k
        )
        
        logger.info(f"Retrieved {len(results)} candidates")
        return results
    
    def format_results_for_reranking(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format search results for reranking.
        
        Args:
            results (List[Dict[str, Any]]): Raw search results
            
        Returns:
            List[Dict[str, Any]]: Formatted results
        """
        formatted_results = []
        for result in results:
            formatted_result = {
                'content': result.get('content', ''),
                'domain': result.get('domain', ''),
                'metadata': result.get('metadata', {}),
                'score': result.get('distance', 0),
                'source': result.get('source', ''),
                'timestamp': result.get('timestamp', 0)
            }
            formatted_results.append(formatted_result)
            
        return formatted_results